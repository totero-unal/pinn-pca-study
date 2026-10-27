"""Run paired baseline-PINN vs. PCA-preprocessing experiments on Burgers."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from .data import load_burgers_data, make_batches, split_times
from .metrics import mean_std, score_prediction
from .models import init_mlp, parameter_count, spatial_pca_basis
from .plotting import grouped_summary_svg
from .training import (baseline_loss, fit_model, make_update, pca_loss,
                       predict_baseline, predict_pca, warmup_update)


MODEL_BASE = "PINN base"


def _to_jax_basis(basis):
    return tuple(jnp.asarray(component) for component in basis)


def _model_summary(rows, model_name):
    model_rows = [row for row in rows if row["model"] == model_name]
    metrics = ("validation_rmse", "test_rmse", "test_relative_l2", "test_pde_residual_rms",
               "training_seconds", "pipeline_seconds")
    result = {
        "seed_count": len(model_rows),
        "parameters": model_rows[0]["parameters"],
        "k": model_rows[0]["k"],
        "retained_variance_mean": float(np.mean([row["retained_variance"] for row in model_rows if row["retained_variance"] is not None]))
        if any(row["retained_variance"] is not None for row in model_rows) else None,
    }
    for metric in metrics:
        result[metric] = mean_std([row[metric] for row in model_rows])
    return result


def _paired_summary(rows, pca_model_name):
    baseline = {row["seed"]: row for row in rows if row["model"] == MODEL_BASE}
    pca = {row["seed"]: row for row in rows if row["model"] == pca_model_name}
    comparisons = {}
    for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms", "pipeline_seconds"):
        deltas = np.asarray([pca[seed][metric] - baseline[seed][metric] for seed in sorted(baseline)])
        comparisons[metric] = {
            "PCA_minus_base_mean": float(deltas.mean()),
            "PCA_minus_base_std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
            "PCA_better_count": int(np.sum(deltas < 0)),
            "n": int(deltas.size),
        }
    return comparisons


def run_comparison(config: dict, project_root: str | Path):
    """Execute the first paired benchmark and save machine-readable results."""
    project_root = Path(project_root).resolve()
    data_path = project_root / config["data_path"]
    output_dir = project_root / "results" / "burgers" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, t, u = load_burgers_data(data_path)

    baseline_width = int(config["baseline_width"])
    pca_width = int(config["pca_width"])
    k = int(config["pca_components"])
    pca_model_name = f"PCA (k={k})"
    learning_rate = float(config["learning_rate"])
    baseline_params_count = parameter_count([2, baseline_width, baseline_width, baseline_width, 1])
    pca_params_count = parameter_count([1, pca_width, pca_width, pca_width, k])
    if abs(pca_params_count - baseline_params_count) / baseline_params_count > 0.05:
        raise ValueError("Las redes deben quedar dentro de 5% de diferencia en número de parámetros.")

    baseline_update = make_update(baseline_loss, learning_rate)
    pca_update = make_update(pca_loss, learning_rate, with_basis=True)

    rows, partitions = [], {}
    seeds = [int(seed) for seed in config["seeds"]]
    first_seed = seeds[0]
    warm_rng = np.random.default_rng(first_seed)
    warm_fit_ids, _, _ = split_times(
        t.size, first_seed, int(config["test_time_count"]), int(config["validation_time_count"]))
    warm_ic_ids = np.sort(warm_rng.choice(x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
    warm_batches = make_batches(x, t, u, warm_fit_ids, warm_ic_ids, 1,
                                int(config["batch_size"]), first_seed)
    first_basis_start = time.perf_counter()
    first_basis, first_variance = spatial_pca_basis(x, u, warm_fit_ids, k)
    first_pca_fit_s = time.perf_counter() - first_basis_start
    first_basis_jax = _to_jax_basis(first_basis)
    base_warm_params = init_mlp(jax.random.PRNGKey(first_seed), [2, baseline_width, baseline_width, baseline_width, 1])
    pca_warm_params = init_mlp(jax.random.PRNGKey(first_seed), [1, pca_width, pca_width, pca_width, k])
    compile_seconds = {
        MODEL_BASE: warmup_update(baseline_update, base_warm_params, warm_batches[0]),
        MODEL_PCA: warmup_update(pca_update, pca_warm_params, warm_batches[0], first_basis_jax),
    }

    for seed in seeds:
        split_rng = np.random.default_rng(seed)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        ic_ids = np.sort(split_rng.choice(x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = make_batches(x, t, u, fit_ids, ic_ids, int(config["steps"]),
                              int(config["batch_size"]), seed)
        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(),
            "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(),
            "initial_condition_x_indices": ic_ids.tolist(),
        }
        print(f"Seed {seed}: training PINN base and PCA(k={k}) on shared batches")

        base_params = init_mlp(jax.random.PRNGKey(seed), [2, baseline_width, baseline_width, baseline_width, 1])
        base_params, base_train_s, base_loss_value = fit_model(baseline_update, base_params, batches)
        base_pred_start = time.perf_counter()
        base_prediction = predict_baseline(base_params, x, t)
        base_prediction_s = time.perf_counter() - base_pred_start
        base_metrics = score_prediction(base_prediction, u, x, t, test_ids)
        base_val = score_prediction(base_prediction, u, x, t, validation_ids)
        rows.append({
            "model": MODEL_BASE, "seed": seed, "k": None, "retained_variance": None,
            "width": baseline_width, "parameters": baseline_params_count,
            "pca_fit_seconds": 0.0, "training_seconds": base_train_s,
            "pipeline_seconds": base_train_s, "prediction_seconds": base_prediction_s,
            "final_training_loss": base_loss_value,
            "validation_rmse": base_val["rmse"], "test_rmse": base_metrics["rmse"],
            "test_relative_l2": base_metrics["relative_l2"],
            "test_pde_residual_rms": base_metrics["pde_residual_rms"],
        })

        if seed == first_seed:
            pca_fit_s, basis, variance = first_pca_fit_s, first_basis, first_variance
        else:
            pca_start = time.perf_counter()
            basis, variance = spatial_pca_basis(x, u, fit_ids, k)
            pca_fit_s = time.perf_counter() - pca_start
        basis_jax = _to_jax_basis(basis)
        pca_params = init_mlp(jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
        pca_params, pca_train_s, pca_loss_value = fit_model(pca_update, pca_params, batches, basis_jax)
        pca_pred_start = time.perf_counter()
        pca_prediction = predict_pca(pca_params, basis_jax, t)
        pca_prediction_s = time.perf_counter() - pca_pred_start
        pca_metrics = score_prediction(pca_prediction, u, x, t, test_ids)
        pca_val = score_prediction(pca_prediction, u, x, t, validation_ids)
        rows.append({
            "model": pca_model_name, "seed": seed, "k": k, "retained_variance": variance,
            "width": pca_width, "parameters": pca_params_count,
            "pca_fit_seconds": pca_fit_s, "training_seconds": pca_train_s,
            "pipeline_seconds": pca_fit_s + pca_train_s,
            "prediction_seconds": pca_prediction_s,
            "final_training_loss": pca_loss_value,
            "validation_rmse": pca_val["rmse"], "test_rmse": pca_metrics["rmse"],
            "test_relative_l2": pca_metrics["relative_l2"],
            "test_pde_residual_rms": pca_metrics["pde_residual_rms"],
        })
        print(f"  Base: test relL2={base_metrics['relative_l2']:.5g}, "
              f"PDE={base_metrics['pde_residual_rms']:.5g}, train={base_train_s:.2f}s")
        print(f"  PCA:  test relL2={pca_metrics['relative_l2']:.5g}, "
              f"PDE={pca_metrics['pde_residual_rms']:.5g}, pipeline={pca_fit_s+pca_train_s:.2f}s, "
              f"variance={variance:.6%}")

    summary = {
        MODEL_BASE: _model_summary(rows, MODEL_BASE),
        pca_model_name: _model_summary(rows, pca_model_name),
    }
    paired = _paired_summary(rows, pca_model_name)
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    run_metadata = {
        "configuration": config,
        "data": {"path": config["data_path"], "shape_x_t": [int(x.size), int(t.size)],
                 "initial_condition": "single profile supplied by burgers_shock.mat"},
        "parameter_matching": {"baseline": baseline_params_count, "pca": pca_params_count,
                               "relative_difference": abs(pca_params_count-baseline_params_count)/baseline_params_count},
        "one_time_compilation_seconds": compile_seconds,
        "timing_policy": "Compile once per architecture before trials; report compilation separately. PCA pipeline time includes PCA fitting and all optimizer steps; baseline pipeline time includes optimizer steps.",
        "runtime": runtime,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions,
        "summary": summary,
        "paired_differences_PCA_minus_base": paired,
        "runs": rows,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    fields = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped_summary_svg(output_dir / "comparison.svg", summary, rows)
    return {"output_dir": output_dir, "rows": rows, "summary": summary,
            "paired_differences": paired, "compile_seconds": compile_seconds,
            "runtime": runtime, "metadata": run_metadata}
