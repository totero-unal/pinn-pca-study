"""Select the PCA physics-loss weight using validation, then test once."""

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
from .data import load_burgers_data
from .extended import _train_to_checkpoints
from .metrics import mean_std, score_prediction
from .models import init_mlp, parameter_count
from .plotting import checkpoint_curve_svg, grouped_summary_svg
from .training import make_update, warmup_update


def _outer_holdout_splits(n_times, outer_seed, seed, outer_test_count, validation_count):
    """Use one fixed outer test split and seed-specific inner validation splits."""
    future = np.arange(1, n_times)
    outer_rng = np.random.default_rng(outer_seed)
    outer_test_ids = np.sort(outer_rng.choice(future, size=outer_test_count, replace=False))
    development = np.setdiff1d(future, outer_test_ids, assume_unique=True)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(development)
    if validation_count >= shuffled.size:
        raise ValueError("No quedan suficientes tiempos para los datos de entrenamiento.")
    validation_ids = np.sort(shuffled[:validation_count])
    fit_ids = np.sort(np.concatenate(([0], shuffled[validation_count:])))
    return fit_ids, validation_ids, outer_test_ids


def _validation_summary(records):
    result = {}
    metrics = ("validation_rmse", "validation_relative_l2",
               "validation_pde_residual_rms", "elapsed_pipeline_seconds")
    models = list(dict.fromkeys(row["model"] for row in records))
    steps = sorted({row["step"] for row in records})
    for model in models:
        result[model] = {}
        for step in steps:
            selected = [r for r in records if r["model"] == model and r["step"] == step]
            if not selected:
                continue
            summary = {"seed_count": len(selected)}
            for metric in metrics:
                summary[metric] = mean_std([r[metric] for r in selected])
            lambda_values = [r["lambda_pde"] for r in selected if r["lambda_pde"] is not None]
            summary["lambda_pde"] = mean_std(lambda_values) if lambda_values else None
            result[model][str(step)] = summary
    return result


def _select_lambda(validation_records, weights, final_step, relative_tolerance):
    candidates = []
    for weight in weights:
        model_name = f"PCA k=8, lambda_PDE={weight:g}"
        rows = [r for r in validation_records
                if r["model"] == model_name and r["step"] == final_step]
        candidates.append({
            "lambda_pde": weight,
            "seed_count": len(rows),
            "validation_rmse": mean_std([r["validation_rmse"] for r in rows]),
            "validation_pde_residual_rms": mean_std(
                [r["validation_pde_residual_rms"] for r in rows]),
        })
    best_rmse = min(row["validation_rmse"]["mean"] for row in candidates)
    cutoff = best_rmse * (1.0 + relative_tolerance)
    feasible = [row for row in candidates if row["validation_rmse"]["mean"] <= cutoff]
    selected = min(feasible, key=lambda row: (
        row["validation_pde_residual_rms"]["mean"],
        row["validation_rmse"]["mean"], row["lambda_pde"],
    ))
    return {
        "rule": "Among PCA weights with mean validation RMSE no more than tolerance above the best PCA mean validation RMSE, select the one with the lowest mean validation PDE residual; ties prefer lower validation RMSE, then lower lambda.",
        "relative_rmse_tolerance": relative_tolerance,
        "best_validation_rmse": best_rmse,
        "validation_rmse_cutoff": cutoff,
        "feasible_lambdas": [row["lambda_pde"] for row in feasible],
        "selected_lambda_pde": selected["lambda_pde"],
        "candidates": candidates,
    }


def run_validation_selected_sweep(config: dict, project_root: str | Path):
    """Tune lambda on validation data and use a fixed outer test set once."""
    project_root = Path(project_root).resolve()
    output_dir = project_root / "results" / "burgers" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, t, u = load_burgers_data(project_root / config["data_path"])
    seeds = [int(v) for v in config["seeds"]]
    steps = int(config["steps"])
    checkpoints = [int(v) for v in config["checkpoint_steps"]]
    if max(checkpoints) > steps or min(checkpoints) < 1:
        raise ValueError("Los checkpoints deben estar dentro del presupuesto de pasos.")
    eval_every = int(config["eval_every_steps"])
    weights = [float(v) for v in config["physics_weights"]]
    targets = [float(v) for v in config["validation_rmse_targets"]]
    outer_seed = int(config["outer_test_seed"])
    outer_count = int(config["outer_test_time_count"])
    validation_count = int(config["validation_time_count"])
    tolerance = float(config["selection_relative_rmse_tolerance"])
    k = int(config["pca_components"])

    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_params_count = parameter_count(base_layers)
    pca_width, pca_params_count = matched_pca_width(k, base_params_count)
    if abs(pca_params_count - base_params_count) / base_params_count > 0.05:
        raise ValueError("Las arquitecturas deben quedar dentro de 5% de diferencia en parámetros.")
    base_update = make_update(baseline_ad_loss, float(config["learning_rate"]))
    pca_update = make_update(pca_ad_loss, float(config["learning_rate"]), with_basis=True)

    validation_rows, trajectories, partitions = [], [], {}
    final_params = {}
    compile_times = {}
    outer_test_ids = None
    outer_test_values_used = False
    for seed in seeds:
        fit_ids, validation_ids, test_ids = _outer_holdout_splits(
            t.size, outer_seed, seed, outer_count, validation_count)
        if outer_test_ids is None:
            outer_test_ids = test_ids
        elif not np.array_equal(outer_test_ids, test_ids):
            raise RuntimeError("La partición externa de prueba debe ser fija para todas las semillas.")

        rng = np.random.default_rng(seed)
        ic_ids = np.sort(rng.choice(
            x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = make_controlled_batches(
            x, t, u, fit_ids, ic_ids, steps, int(config["batch_size"]), seed)
        pca_start = time.perf_counter()
        full_basis, cumulative = fit_pca_full_basis(x, u, fit_ids)
        basis, retained_variance = select_pca_components(full_basis, cumulative, k)
        pca_fit_seconds = time.perf_counter() - pca_start
        weighted_bases = {
            weight: (*basis, jnp.asarray(weight, dtype=jnp.float32)) for weight in weights
        }

        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        pca_params = init_mlp(jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
        x_device = jnp.asarray(x)
        validation_times_device = jnp.asarray(t[validation_ids])
        validation_truth = jnp.asarray(u[:, validation_ids])
        if "baseline" not in compile_times:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, validation_times_device, validation_truth)
            compile_times["baseline"] = time.perf_counter() - compile_start
        if "pca_k8_weighted" not in compile_times:
            compile_start = time.perf_counter()
            first_weight = weights[0]
            warmup_update(pca_update, pca_params, batches[0], weighted_bases[first_weight])
            _compiled_validation_call(pca_params, x_device, validation_times_device,
                                      validation_truth, weighted_bases[first_weight])
            compile_times["pca_k8_weighted"] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(),
            "validation_time_indices": validation_ids.tolist(),
            "outer_test_time_indices": test_ids.tolist(),
            "initial_condition_x_indices": ic_ids.tolist(),
        }
        methods = [{"name": MODEL_BASE, "params": base_params, "basis": None,
                    "lambda_pde": None, "preprocessing_seconds": 0.0}]
        methods.extend({"name": f"PCA k={k}, lambda_PDE={weight:g}",
                        "params": pca_params, "basis": weighted_bases[weight],
                        "lambda_pde": weight, "preprocessing_seconds": pca_fit_seconds}
                       for weight in weights)
        if seed % 2 == 0:
            methods = list(reversed(methods))

        for method in methods:
            basis_state = method["basis"]
            name = method["name"]
            update = base_update if basis_state is None else pca_update
            val_fn = lambda params, b=basis_state: _compiled_validation_call(
                params, x_device, validation_times_device, validation_truth, b)
            params, training_seconds, final_loss, trace, checkpoint_params = _train_to_checkpoints(
                update, method["params"], batches, val_fn, checkpoints, eval_every,
                basis_state=basis_state,
                preprocessing_seconds=method["preprocessing_seconds"])
            final_params[(name, seed)] = (params, basis_state)
            trace_by_step = {row["step"]: row for row in trace}
            trajectories.extend({"model": name, "seed": seed, **row} for row in trace)
            for checkpoint in checkpoints:
                model_params = checkpoint_params[checkpoint]
                if basis_state is None:
                    prediction = np.asarray(_predict_baseline(
                        model_params, x_device, jnp.asarray(t)).block_until_ready())
                else:
                    prediction = np.asarray(_predict_pca_grid(
                        model_params, basis_state, jnp.asarray(t)).block_until_ready())
                metrics = score_prediction(prediction, u, x, t, validation_ids)
                validation_rows.append({
                    "model": name, "seed": seed, "step": checkpoint,
                    "lambda_pde": method["lambda_pde"], "k": None if basis_state is None else k,
                    "retained_variance": None if basis_state is None else retained_variance,
                    "parameters": base_params_count if basis_state is None else pca_params_count,
                    "pca_fit_seconds": method["preprocessing_seconds"],
                    "elapsed_pipeline_seconds": trace_by_step[checkpoint]["elapsed_pipeline_seconds"],
                    "validation_rmse": metrics["rmse"],
                    "validation_relative_l2": metrics["relative_l2"],
                    "validation_pde_residual_rms": metrics["pde_residual_rms"],
                    "test_rmse": None, "test_relative_l2": None,
                    "test_pde_residual_rms": None,
                })
            print(f"Seed {seed}: validation completed for {name} ({training_seconds:.2f}s).")

    final_step = max(checkpoints)
    selection = _select_lambda(validation_rows, weights, final_step, tolerance)
    selected_lambda = selection["selected_lambda_pde"]
    selected_model_name = f"PCA k={k}, lambda_PDE={selected_lambda:g}"

    # Only now, after the validation-based choice is fixed, read test targets for scoring.
    outer_test_values_used = True
    test_rows = []
    for seed in seeds:
        for name in (MODEL_BASE, selected_model_name):
            params, basis_state = final_params[(name, seed)]
            if basis_state is None:
                prediction = np.asarray(_predict_baseline(
                    params, jnp.asarray(x), jnp.asarray(t)).block_until_ready())
            else:
                prediction = np.asarray(_predict_pca_grid(
                    params, basis_state, jnp.asarray(t)).block_until_ready())
            metrics = score_prediction(prediction, u, x, t, outer_test_ids)
            matching_val = [row for row in validation_rows
                            if row["model"] == name and row["seed"] == seed and row["step"] == final_step][0]
            test_rows.append({
                "model": name, "seed": seed, "step": final_step,
                "lambda_pde": None if basis_state is None else selected_lambda,
                "k": None if basis_state is None else k,
                "test_rmse": metrics["rmse"], "test_relative_l2": metrics["relative_l2"],
                "test_pde_residual_rms": metrics["pde_residual_rms"],
                "validation_rmse": matching_val["validation_rmse"],
                "validation_pde_residual_rms": matching_val["validation_pde_residual_rms"],
                "pipeline_seconds": matching_val["elapsed_pipeline_seconds"],
            })
    selected_test_summary = {}
    for name in (MODEL_BASE, selected_model_name):
        chosen = [row for row in test_rows if row["model"] == name]
        selected_test_summary[name] = {
            metric: mean_std([row[metric] for row in chosen])
            for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms", "pipeline_seconds")
        }
    base_test = {row["seed"]: row for row in test_rows if row["model"] == MODEL_BASE}
    selected_test = {row["seed"]: row for row in test_rows if row["model"] == selected_model_name}
    paired_test = {}
    for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms", "pipeline_seconds"):
        deltas = np.asarray([selected_test[seed][metric] - base_test[seed][metric]
                             for seed in sorted(base_test)])
        paired_test[metric] = {
            "selected_PCA_minus_base_mean": float(deltas.mean()),
            "selected_PCA_minus_base_std": float(deltas.std(ddof=1)),
            "selected_PCA_better_count": int(np.sum(deltas < 0)),
            "n": int(deltas.size),
        }

    validation_summary = _validation_summary(validation_rows)
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    metadata = {
        "configuration": config,
        "selection": selection,
        "selected_test_summary": selected_test_summary,
        "paired_selected_test_differences": paired_test,
        "validation_checkpoint_summary": validation_summary,
        "validation_checkpoint_metrics": validation_rows,
        "training_trajectories": trajectories,
        "selected_test_metrics": test_rows,
        "outer_test_indices": outer_test_ids.tolist(),
        "outer_test_targets_used_after_selection": outer_test_values_used,
        "protocol": {
            "nested_split": "One fixed outer test set is excluded from all seed-specific training and validation sets. Validation times are sampled only from the remaining development times.",
            "selection_rule": selection["rule"],
            "selection_cutoff": "15% tolerance over the best mean validation RMSE among PCA weights; select lowest mean validation PDE residual among feasible weights.",
            "test_timing": "PCA weight is chosen from aggregated validation metrics before any test residual/solution metric is computed. Test is evaluated only for the selected lambda and the base comparator.",
            "pca_scope": "PCA is fitted only on supervised training-time snapshots; validation and outer test targets are excluded.",
            "same_residual_operator": "JAX autodifferentiation is used for the training PDE residual in baseline and PCA.",
            "pde_metric": "Validation and final test PDE residual are scored with the same finite-difference operator on the complete prediction grid.",
            "timing": "JAX compilation is measured separately; PCA pipeline time includes its training-only basis fit.",
        },
        "compilation_seconds_by_architecture": compile_times,
        "runtime": runtime, "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions,
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("validation_checkpoint_metrics.csv", validation_rows),
                              ("training_trajectories.csv", trajectories),
                              ("selected_test_metrics.csv", test_rows)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    validation_plot_summary = {}
    final_val_rows = [row for row in validation_rows if row["step"] == final_step]
    for name in dict.fromkeys(row["model"] for row in final_val_rows):
        chosen = [row for row in final_val_rows if row["model"] == name]
        validation_plot_summary[name] = {
            "test_relative_l2": mean_std([row["validation_relative_l2"] for row in chosen]),
            "test_pde_residual_rms": mean_std([row["validation_pde_residual_rms"] for row in chosen]),
            "pipeline_seconds": mean_std([row["elapsed_pipeline_seconds"] for row in chosen]),
        }
    grouped_summary_svg(output_dir / "validation_candidates.svg", validation_plot_summary, final_val_rows)
    checkpoint_curve_svg(output_dir / "validation_rmse_vs_steps.svg", validation_rows,
                         "validation_rmse", "RMSE de validación por peso PDE", "RMSE")
    checkpoint_curve_svg(output_dir / "validation_residual_vs_steps.svg", validation_rows,
                         "validation_pde_residual_rms", "Residuo de validación por peso PDE", "RMS del residuo")
    checkpoint_curve_svg(output_dir / "pipeline_time_vs_steps.svg", validation_rows,
                         "elapsed_pipeline_seconds", "Tiempo de pipeline por peso PDE", "Segundos")
    selected_summary_for_plot = {
        name: {
            "test_relative_l2": selected_test_summary[name]["test_relative_l2"],
            "test_pde_residual_rms": selected_test_summary[name]["test_pde_residual_rms"],
            "pipeline_seconds": selected_test_summary[name]["pipeline_seconds"],
        } for name in selected_test_summary
    }
    grouped_summary_svg(output_dir / "selected_test_comparison.svg", selected_summary_for_plot, test_rows)
    return {"output_dir": output_dir, "metadata": metadata, "validation_rows": validation_rows,
            "test_rows": test_rows}
