"""Longer-budget Burgers benchmark and PCA component-count sweep."""

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
from .metrics import mean_std, score_prediction
from .models import init_mlp, parameter_count
from .plotting import checkpoint_curve_svg, grouped_summary_svg
from .training import adam_init, make_update, warmup_update


def _train_to_checkpoints(update, params, batches, validation_fn, checkpoint_steps,
                          eval_every, basis_state=None, preprocessing_seconds=0.0):
    first_moment, second_moment, adam_step_number = adam_init(params)
    checkpoints = {}
    trace = []
    started = time.perf_counter()
    final_loss = jnp.asarray(jnp.nan)
    checkpoint_steps = set(int(step) for step in checkpoint_steps)
    for step_index, batch in enumerate(batches, start=1):
        if basis_state is None:
            params, first_moment, second_moment, adam_step_number, final_loss = update(
                params, first_moment, second_moment, adam_step_number, batch)
        else:
            params, first_moment, second_moment, adam_step_number, final_loss = update(
                params, first_moment, second_moment, adam_step_number, batch, basis_state)
        loss_value = float(final_loss)
        if step_index % eval_every == 0 or step_index in checkpoint_steps or step_index == len(batches):
            validation_rmse = validation_fn(params)
            elapsed = preprocessing_seconds + time.perf_counter() - started
            trace.append({
                "step": step_index,
                "elapsed_pipeline_seconds": elapsed,
                "validation_rmse": validation_rmse,
                "training_loss": loss_value,
            })
            if step_index in checkpoint_steps:
                # JAX arrays are immutable; retain the checkpoint state without copying to host.
                checkpoints[step_index] = params
    jax.tree.map(lambda value: value.block_until_ready(), params)
    training_seconds = time.perf_counter() - started
    return params, training_seconds, float(final_loss), trace, checkpoints


def _checkpoint_summary(records):
    result = {}
    metrics = ("validation_rmse", "validation_relative_l2", "validation_pde_residual_rms",
               "elapsed_pipeline_seconds", "test_rmse", "test_relative_l2",
               "test_pde_residual_rms")
    models = list(dict.fromkeys(row["model"] for row in records))
    steps = sorted({row["step"] for row in records})
    for model in models:
        result[model] = {}
        for step in steps:
            selected = [row for row in records if row["model"] == model and row["step"] == step]
            if not selected:
                continue
            summary = {
                "seed_count": len(selected),
                "k": mean_std([row["k"] for row in selected if row["k"] is not None])
                if any(row["k"] is not None for row in selected) else None,
                "lambda_pde": mean_std([row["lambda_pde"] for row in selected
                                         if row.get("lambda_pde") is not None])
                if any(row.get("lambda_pde") is not None for row in selected) else None,
                "retained_variance": mean_std([row["retained_variance"] for row in selected
                                                if row["retained_variance"] is not None])
                if any(row["retained_variance"] is not None for row in selected) else None,
            }
            for metric in metrics:
                values = [row[metric] for row in selected if row.get(metric) is not None]
                summary[metric] = mean_std(values) if values else None
            result[model][str(step)] = summary
    return result


def _paired_validation_summary(records, baseline_name=MODEL_BASE):
    models = list(dict.fromkeys(row["model"] for row in records))
    baseline = {(r["seed"], r["step"]): r for r in records if r["model"] == baseline_name}
    result = {}
    for model in models:
        if model == baseline_name:
            continue
        model_rows = {(r["seed"], r["step"]): r for r in records if r["model"] == model}
        result[model] = {}
        for step in sorted({r["step"] for r in records}):
            keys = sorted(key for key in model_rows if key[1] == step and key in baseline)
            metrics = ("validation_rmse", "validation_pde_residual_rms", "elapsed_pipeline_seconds")
            by_metric = {}
            for metric in metrics:
                delta = np.asarray([model_rows[key][metric] - baseline[key][metric] for key in keys])
                by_metric[metric] = {
                    "PCA_minus_base_mean": float(delta.mean()),
                    "PCA_minus_base_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                    "PCA_better_count": int(np.sum(delta < 0)),
                    "n": int(delta.size),
                }
            result[model][str(step)] = by_metric
    return result


def _paired_final_test_summary(records, final_step, baseline_name=MODEL_BASE):
    final_rows = [row for row in records if row["step"] == final_step]
    baseline = {row["seed"]: row for row in final_rows if row["model"] == baseline_name}
    result = {}
    for model in dict.fromkeys(row["model"] for row in final_rows):
        if model == baseline_name:
            continue
        model_rows = {row["seed"]: row for row in final_rows if row["model"] == model}
        result[model] = {}
        for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms",
                       "elapsed_pipeline_seconds"):
            delta = np.asarray([model_rows[seed][metric] - baseline[seed][metric]
                                for seed in sorted(baseline)])
            result[model][metric] = {
                "PCA_minus_base_mean": float(delta.mean()),
                "PCA_minus_base_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                "PCA_better_count": int(np.sum(delta < 0)),
                "n": int(delta.size),
            }
    return result


def _time_to_thresholds(trajectories, targets):
    result = {}
    models = list(dict.fromkeys(row["model"] for row in trajectories))
    for model in models:
        rows = [row for row in trajectories if row["model"] == model]
        seeds = sorted({row["seed"] for row in rows})
        result[model] = {}
        for target in targets:
            times = []
            for seed in seeds:
                seed_rows = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["step"])
                hits = [r for r in seed_rows if r["validation_rmse"] <= target]
                if hits:
                    times.append(hits[0]["elapsed_pipeline_seconds"])
            result[model][str(target)] = {
                "hit_count": len(times), "seed_count": len(seeds),
                "seconds_mean_if_hit": float(np.mean(times)) if times else None,
                "seconds_std_if_hit": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            }
    return result


def run_longer_k_sweep(config: dict, project_root: str | Path):
    """Train baseline and PCA variants to a longer budget with paired batches."""
    project_root = Path(project_root).resolve()
    output_dir = project_root / "results" / "burgers" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, t, u = load_burgers_data(project_root / config["data_path"])
    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_param_count = parameter_count(base_layers)
    base_update = make_update(baseline_ad_loss, float(config["learning_rate"]))
    pca_update = make_update(pca_ad_loss, float(config["learning_rate"]), with_basis=True)
    seeds = [int(seed) for seed in config["seeds"]]
    checkpoints = [int(step) for step in config["checkpoint_steps"]]
    steps = int(config["steps"])
    if max(checkpoints) > steps or min(checkpoints) < 1:
        raise ValueError("Los checkpoints deben estar dentro del presupuesto de pasos.")
    eval_every = int(config["eval_every_steps"])
    targets = [float(v) for v in config["validation_rmse_targets"]]
    requested_extra_ks = [int(v) for v in config["pca_extra_components"]]
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
        basis_fit_start = time.perf_counter()
        full_basis, cumulative_variance = fit_pca_full_basis(x, u, fit_ids)
        pca_fit_seconds = time.perf_counter() - basis_fit_start
        auto_k = int(np.searchsorted(cumulative_variance,
                                     float(config["pca_variance_threshold"]), side="left") + 1)
        settings = [("PCA 99%", auto_k)]
        settings.extend((f"PCA k={k}", k) for k in requested_extra_ks if k != auto_k)
        if max(k for _, k in settings) > len(cumulative_variance):
            raise ValueError("Un k solicitado supera el rango PCA disponible en esta partición.")
        pca_models = []
        for name, k in settings:
            basis, retained_variance = select_pca_components(full_basis, cumulative_variance, k)
            pca_width, pca_param_count = matched_pca_width(k, base_param_count)
            if abs(pca_param_count - base_param_count) / base_param_count > 0.05:
                raise ValueError(f"No se emparejó el número de parámetros para k={k}.")
            params = init_mlp(jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
            pca_models.append({
                "name": name, "k": k, "basis": basis, "width": pca_width,
                "parameters": pca_param_count, "retained_variance": retained_variance,
                "params": params,
            })

        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        x_device, all_t_device = jnp.asarray(x), jnp.asarray(t)
        val_t_device = jnp.asarray(t[validation_ids])
        val_truth = jnp.asarray(u[:, validation_ids])
        base_key = "baseline"
        if base_key not in compiled:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, val_t_device, val_truth)
            compiled[base_key] = time.perf_counter() - compile_start
        for model in pca_models:
            key = f"pca_k{model['k']}_w{model['width']}"
            if key not in compiled:
                compile_start = time.perf_counter()
                warmup_update(pca_update, model["params"], batches[0], model["basis"])
                _compiled_validation_call(model["params"], x_device, val_t_device,
                                          val_truth, model["basis"])
                compiled[key] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(), "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(), "initial_condition_x_indices": ic_ids.tolist(),
        }
        print(f"Seed {seed}: k99={auto_k}; PCA k values={[m['k'] for m in pca_models]}")

        methods = [{"name": MODEL_BASE, "params": base_params, "basis": None,
                    "k": None, "retained_variance": None, "width": base_width,
                    "parameters": base_param_count, "preprocessing_seconds": 0.0}]
        methods.extend({**model, "preprocessing_seconds": pca_fit_seconds} for model in pca_models)
        # Alternate order across seeds to reduce a systematic runtime-order effect.
        if seed % 2 == 0:
            methods = list(reversed(methods))

        for method in methods:
            model_name = method["name"]
            basis = method["basis"]
            validation_fn = lambda params, b=basis: _compiled_validation_call(
                params, x_device, val_t_device, val_truth, b)
            update = base_update if basis is None else pca_update
            params, training_seconds, loss_value, trace, checkpoint_params = _train_to_checkpoints(
                update, method["params"], batches, validation_fn, checkpoints, eval_every,
                basis_state=basis, preprocessing_seconds=method["preprocessing_seconds"])
            for item in trace:
                trajectories.append({"model": model_name, "seed": seed, **item})
            trace_by_step = {record["step"]: record for record in trace}
            for checkpoint in checkpoints:
                checkpoint_model_params = checkpoint_params[checkpoint]
                if basis is None:
                    prediction = np.asarray(_predict_baseline(
                        checkpoint_model_params, x_device, all_t_device).block_until_ready())
                else:
                    prediction = np.asarray(_predict_pca_grid(
                        checkpoint_model_params, basis, all_t_device).block_until_ready())
                validation_metrics = score_prediction(prediction, u, x, t, validation_ids)
                record = {
                    "model": model_name, "seed": seed, "step": checkpoint,
                    "k": method["k"], "retained_variance": method["retained_variance"],
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
            print(f"  {model_name}: {training_seconds:.2f}s entrenamiento, "
                  f"{method['preprocessing_seconds']+training_seconds:.2f}s pipeline")

    summary = _checkpoint_summary(rows)
    final_step = max(checkpoints)
    final_rows = [row for row in rows if row["step"] == final_step]
    final_summary = {}
    for model in dict.fromkeys(row["model"] for row in final_rows):
        selected = [row for row in final_rows if row["model"] == model]
        final_summary[model] = {
            "test_rmse": mean_std([row["test_rmse"] for row in selected]),
            "test_relative_l2": mean_std([row["test_relative_l2"] for row in selected]),
            "test_pde_residual_rms": mean_std([row["test_pde_residual_rms"] for row in selected]),
            "pipeline_seconds": mean_std([row["elapsed_pipeline_seconds"] for row in selected]),
        }
    paired = _paired_validation_summary(rows)
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
            "same_sampling": "All methods share within-seed time partitions, supervised minibatches, PDE collocation points, and initial-condition points.",
            "same_residual_operator": "JAX autodifferentiation of the solution for u_t, u_x, and u_xx in every method.",
            "pca_fit_scope": "The PCA basis uses training times only; a full basis is computed once per seed and truncated for the k sweep.",
            "parameter_matching": "Each PCA MLP hidden width is selected to keep its parameter count within 5% of the base network.",
            "validation_and_test": "Validation metrics are tracked at checkpoints. Test metrics are calculated only at the final checkpoint (maximum configured step).",
            "timing": "Compilation and validation-function compilation are excluded from timed training and reported separately. PCA pipeline elapsed time includes its training-only SVD/basis construction.",
            "pde_metric": "Validation and final test PDE residual use the same grid finite-difference scoring operator for all models; training residuals use the same JAX autodiff operator.",
        },
        "compilation_seconds_by_architecture": compiled,
        "runtime": runtime, "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions, "checkpoint_summary": summary,
        "final_test_summary": final_summary, "paired_validation_differences": paired,
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
                         "validation_rmse", "RMSE de validación frente a pasos", "RMSE")
    checkpoint_curve_svg(output_dir / "validation_residual_vs_steps.svg", rows,
                         "validation_pde_residual_rms", "Residuo de Burgers de validación", "RMS del residuo")
    checkpoint_curve_svg(output_dir / "pipeline_time_vs_steps.svg", rows,
                         "elapsed_pipeline_seconds", "Tiempo acumulado de pipeline", "Segundos")
    return {"output_dir": output_dir, "metadata": metadata, "rows": rows,
            "trajectories": trajectories}
