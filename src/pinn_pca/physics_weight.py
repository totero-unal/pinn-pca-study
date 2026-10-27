"""Empirical PDE-loss weighting sweep for the PCA k=8 Burgers model."""

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

from .controlled import (
    MODEL_BASE,
    _compiled_validation_call,
    _predict_baseline,
    _predict_pca_grid,
    baseline_ad_loss,
    fit_pca_full_basis,
    make_controlled_batches,
    matched_pca_width,
    pca_ad_loss,
    select_pca_components,
)
from .data import load_burgers_data, split_times
from .extended import (
    _checkpoint_summary,
    _paired_final_test_summary,
    _paired_validation_summary,
    _time_to_thresholds,
    _train_to_checkpoints,
)
from .metrics import mean_std, score_prediction
from .models import init_mlp, parameter_count
from .plotting import checkpoint_curve_svg, grouped_summary_svg
from .training import make_update, warmup_update


def run_physics_weight_sweep(config: dict, project_root: str | Path):
    """Compare PDE-loss weights for a fixed k=8 PCA model on paired runs."""
    project_root = Path(project_root).resolve()
    output_dir = project_root / "results" / "burgers" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, t, u = load_burgers_data(project_root / config["data_path"])
    seeds = [int(seed) for seed in config["seeds"]]
    steps = int(config["steps"])
    checkpoints = [int(step) for step in config["checkpoint_steps"]]
    if max(checkpoints) > steps or min(checkpoints) < 1:
        raise ValueError("Los checkpoints deben estar dentro del presupuesto de pasos.")
    eval_every = int(config["eval_every_steps"])
    targets = [float(value) for value in config["validation_rmse_targets"]]
    weights = [float(value) for value in config["physics_weights"]]
    k = int(config["pca_components"])
    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_parameter_count = parameter_count(base_layers)
    pca_width, pca_parameter_count = matched_pca_width(k, base_parameter_count)
    if abs(pca_parameter_count - base_parameter_count) / base_parameter_count > 0.05:
        raise ValueError("La arquitectura PCA debe quedar a menos de 5% de parámetros de la base.")

    base_update = make_update(baseline_ad_loss, float(config["learning_rate"]))
    pca_update = make_update(pca_ad_loss, float(config["learning_rate"]), with_basis=True)
    rows, trajectories, partitions = [], [], {}
    compiled = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        ic_ids = np.sort(rng.choice(
            x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = make_controlled_batches(
            x, t, u, fit_ids, ic_ids, steps, int(config["batch_size"]), seed)
        pca_fit_start = time.perf_counter()
        full_basis, cumulative_variance = fit_pca_full_basis(x, u, fit_ids)
        pca_basis, retained_variance = select_pca_components(full_basis, cumulative_variance, k)
        pca_fit_seconds = time.perf_counter() - pca_fit_start
        weighted_bases = {
            weight: (*pca_basis, jnp.asarray(weight, dtype=jnp.float32)) for weight in weights
        }
        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        # Reuse the exact same initial PCA parameters across all lambda values.
        pca_params = init_mlp(
            jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
        x_device, all_t_device = jnp.asarray(x), jnp.asarray(t)
        val_t_device, val_truth = jnp.asarray(t[validation_ids]), jnp.asarray(u[:, validation_ids])

        if "baseline" not in compiled:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, val_t_device, val_truth)
            compiled["baseline"] = time.perf_counter() - compile_start
        if "pca_k8_weighted" not in compiled:
            compile_start = time.perf_counter()
            first_weight = weights[0]
            warmup_update(pca_update, pca_params, batches[0], weighted_bases[first_weight])
            _compiled_validation_call(pca_params, x_device, val_t_device, val_truth,
                                      weighted_bases[first_weight])
            compiled["pca_k8_weighted"] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(), "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(), "initial_condition_x_indices": ic_ids.tolist(),
        }
        print(f"Seed {seed}: k={k}, var={retained_variance:.6%}, weights={weights}")
        methods = [{"name": MODEL_BASE, "params": base_params, "basis": None,
                    "lambda_pde": None, "k": None, "retained_variance": None,
                    "width": base_width, "parameters": base_parameter_count,
                    "preprocessing_seconds": 0.0}]
        methods.extend({
            "name": f"PCA k={k}, lambda_PDE={weight:g}", "params": pca_params,
            "basis": weighted_bases[weight], "lambda_pde": weight, "k": k,
            "retained_variance": retained_variance, "width": pca_width,
            "parameters": pca_parameter_count, "preprocessing_seconds": pca_fit_seconds,
        } for weight in weights)
        if seed % 2 == 0:
            methods = list(reversed(methods))

        for method in methods:
            basis = method["basis"]
            model_name = method["name"]
            validation_fn = lambda params, b=basis: _compiled_validation_call(
                params, x_device, val_t_device, val_truth, b)
            update = base_update if basis is None else pca_update
            params, training_seconds, loss_value, trace, checkpoint_params = _train_to_checkpoints(
                update, method["params"], batches, validation_fn, checkpoints, eval_every,
                basis_state=basis, preprocessing_seconds=method["preprocessing_seconds"])
            trajectories.extend({"model": model_name, "seed": seed, **row} for row in trace)
            trace_by_step = {row["step"]: row for row in trace}
            for checkpoint in checkpoints:
                checkpoint_params_at_step = checkpoint_params[checkpoint]
                if basis is None:
                    prediction = np.asarray(_predict_baseline(
                        checkpoint_params_at_step, x_device, all_t_device).block_until_ready())
                else:
                    prediction = np.asarray(_predict_pca_grid(
                        checkpoint_params_at_step, basis, all_t_device).block_until_ready())
                validation_metrics = score_prediction(prediction, u, x, t, validation_ids)
                record = {
                    "model": model_name, "seed": seed, "step": checkpoint,
                    "lambda_pde": method["lambda_pde"], "k": method["k"],
                    "retained_variance": method["retained_variance"],
                    "width": method["width"], "parameters": method["parameters"],
                    "pca_fit_seconds": method["preprocessing_seconds"],
                    "elapsed_pipeline_seconds": trace_by_step[checkpoint]["elapsed_pipeline_seconds"],
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_relative_l2": validation_metrics["relative_l2"],
                    "validation_pde_residual_rms": validation_metrics["pde_residual_rms"],
                    "final_training_loss": loss_value if checkpoint == steps else None,
                    "test_rmse": None, "test_relative_l2": None,
                    "test_pde_residual_rms": None,
                }
                if checkpoint == max(checkpoints):
                    test_metrics = score_prediction(prediction, u, x, t, test_ids)
                    record.update({"test_rmse": test_metrics["rmse"],
                                   "test_relative_l2": test_metrics["relative_l2"],
                                   "test_pde_residual_rms": test_metrics["pde_residual_rms"]})
                rows.append(record)
            print(f"  {model_name}: {training_seconds:.2f}s, pipeline "
                  f"{method['preprocessing_seconds'] + training_seconds:.2f}s")

    summary = _checkpoint_summary(rows)
    final_step = max(checkpoints)
    final_rows = [row for row in rows if row["step"] == final_step]
    final_summary = {}
    for model in dict.fromkeys(row["model"] for row in final_rows):
        selected = [row for row in final_rows if row["model"] == model]
        final_summary[model] = {
            metric: mean_std([row[metric] for row in selected])
            for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms",
                           "elapsed_pipeline_seconds")
        }
    paired_validation = _paired_validation_summary(rows)
    paired_test = _paired_final_test_summary(rows, final_step)
    time_targets = _time_to_thresholds(trajectories, targets)
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    metadata = {
        "configuration": config,
        "protocol": {
            "same_samples": "All methods share time splits, supervised minibatches, PDE collocation points, and initial-condition observations within each seed.",
            "same_initialization": "All PCA loss-weight variants start from identical parameter values for a given seed.",
            "same_residual_operator": "JAX autodifferentiation is used for u_t, u_x, and u_xx during training in the baseline and PCA methods.",
            "pca_scope": "PCA k=8 is fitted on training-time snapshots only; validation and test times are excluded.",
            "loss": "L = MSE_data + lambda_PDE * MSE_residual + MSE_initial; only lambda_PDE varies for PCA.",
            "validation_test": "Validation metrics are monitored at checkpoints; test metrics are calculated only at the final checkpoint.",
            "timing": "Compilation is excluded from training time and reported separately. PCA pipeline time includes the training-only PCA fit.",
            "pde_metric": "PDE residual validation and test metrics use the same grid finite-difference scoring operator for every method; training uses JAX autodiff for every method.",
        },
        "compilation_seconds_by_architecture": compiled, "runtime": runtime,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions, "checkpoint_summary": summary,
        "final_test_summary": final_summary,
        "paired_validation_differences": paired_validation,
        "paired_final_test_differences": paired_test,
        "time_to_validation_thresholds": time_targets,
        "checkpoint_metrics": rows, "training_trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("checkpoint_metrics.csv", rows),
                              ("training_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    summary_for_plot = {
        name: {
            "test_relative_l2": mean_std([r["test_relative_l2"] for r in final_rows if r["model"] == name]),
            "test_pde_residual_rms": mean_std([r["test_pde_residual_rms"] for r in final_rows if r["model"] == name]),
            "pipeline_seconds": mean_std([r["elapsed_pipeline_seconds"] for r in final_rows if r["model"] == name]),
        } for name in final_summary
    }
    grouped_summary_svg(output_dir / "final_test_comparison.svg", summary_for_plot, final_rows)
    checkpoint_curve_svg(output_dir / "validation_rmse_vs_steps.svg", rows,
                         "validation_rmse", "RMSE validación por peso físico", "RMSE")
    checkpoint_curve_svg(output_dir / "validation_residual_vs_steps.svg", rows,
                         "validation_pde_residual_rms", "Residuo físico de validación por peso", "RMS del residuo")
    checkpoint_curve_svg(output_dir / "pipeline_time_vs_steps.svg", rows,
                         "elapsed_pipeline_seconds", "Tiempo de pipeline por peso físico", "Segundos")
    return {"output_dir": output_dir, "metadata": metadata, "rows": rows,
            "trajectories": trajectories}
