"""Component-count sweep for the paired periodic-advection benchmark."""

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

from .advection import (
    DEFAULT_ADVECTION_TERMS,
    MODEL_BASE,
    MODEL_PCA,
    _make_advection_batches,
    _make_losses,
    _score,
    make_periodic_advection_data,
)
from .controlled import (
    _compiled_validation_call,
    _predict_baseline,
    _predict_pca_grid,
    _train_trace,
    fit_pca_full_basis,
    matched_pca_width,
    select_pca_components,
)
from .data import split_times
from .metrics import mean_std
from .models import init_mlp, parameter_count
from .plotting import checkpoint_curve_svg, grouped_summary_svg, metric_tradeoff_svg
from .training import make_update, warmup_update


def _summarize(rows, name):
    selected = [row for row in rows if row["model"] == name]
    summary = {
        "seed_count": len(selected),
        "parameters_mean": float(np.mean([row["parameters"] for row in selected])),
        "parameters_min": int(min(row["parameters"] for row in selected)),
        "parameters_max": int(max(row["parameters"] for row in selected)),
        "k": mean_std([row["k"] for row in selected if row["k"] is not None])
        if any(row["k"] is not None for row in selected) else None,
        "retained_variance": mean_std([row["retained_variance"] for row in selected
                                        if row["retained_variance"] is not None])
        if any(row["retained_variance"] is not None for row in selected) else None,
    }
    for metric in ("validation_rmse", "test_rmse", "test_relative_l2",
                   "test_pde_residual_rms", "test_periodic_boundary_rms",
                   "training_seconds", "pipeline_seconds"):
        summary[metric] = mean_std([row[metric] for row in selected])
    return summary


def run_advection_k_sweep(config: dict, project_root: str | Path):
    """Compare variance-selected and fixed-k PCA representations on advection."""
    project_root = Path(project_root).resolve()
    data_path = project_root / config["data_path"]
    fourier_terms = config.get("initial_condition_terms", DEFAULT_ADVECTION_TERMS)
    if data_path.exists():
        with np.load(data_path) as data:
            x, t, u = data["x"], data["t"], data["u"]
            velocity = float(data["velocity"])
            if "fourier_terms_json" in data:
                cached_terms = json.loads(str(data["fourier_terms_json"]))
                if cached_terms != fourier_terms:
                    raise ValueError(
                        f"Los términos de {data_path} no coinciden con la configuración."
                    )
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        velocity = float(config["velocity"])
        x, t, u = make_periodic_advection_data(
            config["nx"], config["nt"], velocity, fourier_terms)
        np.savez_compressed(
            data_path, x=x, t=t, u=u, velocity=velocity,
            fourier_terms_json=np.asarray(json.dumps(fourier_terms, sort_keys=True)),
        )
    output_dir = project_root / "results" / "advection" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_parameters = parameter_count(base_layers)
    baseline_loss, pca_loss = _make_losses(velocity)
    base_update = make_update(baseline_loss, float(config["learning_rate"]))
    pca_update = make_update(pca_loss, float(config["learning_rate"]), with_basis=True)
    seeds = [int(seed) for seed in config["seeds"]]
    extras = [int(k) for k in config["pca_extra_components"]]
    target_values = [float(value) for value in config.get("validation_rmse_targets", [])]
    rows, trajectories, partitions, compilation = [], [], {}, {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        initial_ids = np.sort(rng.choice(
            x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = _make_advection_batches(
            x, t, u, fit_ids, initial_ids, int(config["steps"]),
            int(config["batch_size"]), seed)
        pca_fit_start = time.perf_counter()
        full_basis, cumulative = fit_pca_full_basis(x, u, fit_ids)
        pca_fit_seconds = time.perf_counter() - pca_fit_start
        auto_k = int(np.searchsorted(
            cumulative, float(config["pca_variance_threshold"]), side="left") + 1)
        component_specs = [(f"{MODEL_PCA} 99%", auto_k)]
        component_specs.extend((f"{MODEL_PCA} k={k}", k) for k in extras if k != auto_k)

        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        pca_models = []
        for name, k in component_specs:
            basis, retained_variance = select_pca_components(full_basis, cumulative, k)
            width, parameter_count_pca = matched_pca_width(k, base_parameters)
            if abs(parameter_count_pca - base_parameters) / base_parameters > 0.05:
                raise ValueError(f"No se emparejó el número de parámetros para k={k}.")
            params = init_mlp(jax.random.PRNGKey(seed), [1, width, width, width, k])
            pca_models.append({
                "name": name, "k": k, "basis": basis, "width": width,
                "parameters": parameter_count_pca,
                "retained_variance": retained_variance,
                "params": params,
            })

        x_device = jnp.asarray(x)
        val_times_device = jnp.asarray(t[validation_ids])
        val_truth = jnp.asarray(u[:, validation_ids])
        if MODEL_BASE not in compilation:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, val_times_device, val_truth)
            compilation[MODEL_BASE] = time.perf_counter() - compile_start
        for model in pca_models:
            key = f"{model['name']} width={model['width']}"
            if key not in compilation:
                compile_start = time.perf_counter()
                warmup_update(pca_update, model["params"], batches[0], model["basis"])
                _compiled_validation_call(model["params"], x_device, val_times_device,
                                          val_truth, model["basis"])
                compilation[key] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(), "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(), "initial_condition_x_indices": initial_ids.tolist(),
        }
        methods = [{"name": MODEL_BASE, "params": base_params, "basis": None,
                    "k": None, "retained_variance": None, "width": base_width,
                    "parameters": base_parameters, "preprocessing_seconds": 0.0}]
        methods.extend({**model, "preprocessing_seconds": pca_fit_seconds} for model in pca_models)
        if seed % 2 == 0:
            methods = list(reversed(methods))

        for method in methods:
            basis = method["basis"]
            name = method["name"]
            val_fn = lambda params, b=basis: _compiled_validation_call(
                params, x_device, val_times_device, val_truth, b)
            update = base_update if basis is None else pca_update
            params, train_seconds, final_loss, trace, _, _ = _train_trace(
                update, method["params"], batches, val_fn,
                int(config["eval_every_steps"]), target_values,
                basis_state=basis, preprocessing_seconds=method["preprocessing_seconds"])
            if basis is None:
                prediction = np.asarray(_predict_baseline(
                    params, x_device, jnp.asarray(t)).block_until_ready())
            else:
                prediction = np.asarray(_predict_pca_grid(
                    params, basis, jnp.asarray(t)).block_until_ready())
            validation_metrics = _score(prediction, u, x, t, validation_ids, velocity)
            test_metrics = _score(prediction, u, x, t, test_ids, velocity)
            rows.append({
                "model": name, "seed": seed, "k": method["k"],
                "retained_variance": method["retained_variance"],
                "width": method["width"], "parameters": method["parameters"],
                "pca_fit_seconds": method["preprocessing_seconds"],
                "training_seconds": train_seconds,
                "pipeline_seconds": method["preprocessing_seconds"] + train_seconds,
                "final_training_loss": final_loss,
                "validation_rmse": validation_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "test_relative_l2": test_metrics["relative_l2"],
                "test_pde_residual_rms": test_metrics["pde_residual_rms"],
                "test_periodic_boundary_rms": test_metrics["periodic_boundary_rms"],
            })
            trajectories.extend({"model": name, "seed": seed, **record} for record in trace)
        print(f"Seed {seed}: k99={auto_k}; swept k={[m['k'] for m in pca_models]}")

    model_names = list(dict.fromkeys(row["model"] for row in rows))
    summary = {name: _summarize(rows, name) for name in model_names}
    paired = {}
    baseline = {row["seed"]: row for row in rows if row["model"] == MODEL_BASE}
    for name in model_names:
        if name == MODEL_BASE:
            continue
        selected = {row["seed"]: row for row in rows if row["model"] == name}
        paired[name] = {}
        for metric in ("test_relative_l2", "test_pde_residual_rms",
                       "test_periodic_boundary_rms", "pipeline_seconds"):
            deltas = np.asarray([selected[seed][metric] - baseline[seed][metric]
                                 for seed in sorted(baseline)])
            paired[name][metric] = {
                "PCA_minus_base_mean": float(deltas.mean()),
                "PCA_minus_base_std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "PCA_better_count": int(np.sum(deltas < 0)), "n": int(deltas.size),
            }
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    metadata = {
        "configuration": config,
        "data": {"path": config["data_path"], "shape_x_t": [int(x.size), int(t.size)],
                 "velocity": velocity,
                 "initial_condition": config.get("initial_condition_name", "Fourier series"),
                 "initial_condition_terms": fourier_terms},
        "protocol": {
            "same_sampling": "Each seed shares temporal splits, supervised batches, PDE collocation points, periodic boundary times, and initial-condition locations across models.",
            "pca_scope": "Full PCA basis fit on training-time snapshots only, then truncated at each k.",
            "same_training_residual": "JAX autodifferentiation of u_t + c*u_x for baseline and all PCA models.",
            "periodicity": "All models include a loss penalty on u(-1,t)-u(1,t).",
            "evaluation": "Same test-time indices and common grid-difference residual metric within each seed.",
            "timing": "Compilation is measured separately; each PCA pipeline includes training-only basis fitting.",
        },
        "parameter_matching": {"baseline": parameter_count([2, base_width, base_width, base_width, 1]),
                               "pca_min": min(row["parameters"] for row in rows if row["k"] is not None),
                               "pca_max": max(row["parameters"] for row in rows if row["k"] is not None)},
        "compilation_seconds_by_architecture": compilation,
        "runtime": runtime, "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions, "summary": summary,
        "paired_differences_PCA_minus_base": paired,
        "runs": rows, "training_trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    grouped_summary_svg(output_dir / "comparison.svg", summary, rows,
                        pde_name="advección")
    metric_tradeoff_svg(output_dir / "error_residual_tradeoff.svg", summary,
                        "test_relative_l2", "test_pde_residual_rms",
                        "Error relativo L2 de prueba", "Residuo RMS de advección",
                        "Compromiso entre error de solución y residuo PDE")
    checkpoint_curve_svg(output_dir / "validation_rmse_vs_steps.svg", trajectories,
                         "validation_rmse", "RMSE de validación frente a pasos", "RMSE")
    checkpoint_curve_svg(output_dir / "pipeline_time_vs_steps.svg", trajectories,
                         "elapsed_pipeline_seconds", "Tiempo de pipeline frente a pasos", "Segundos")
    return {"output_dir": output_dir, "summary": summary, "rows": rows,
            "trajectories": trajectories, "metadata": metadata}
