"""Controlled Burgers comparison with the same autodifferentiated PDE residual."""

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
from .models import NU, burgers_solution, init_mlp, mlp, parameter_count
from .plotting import controlled_trajectory_svg, grouped_summary_svg
from .training import adam_init, adam_step, make_update, warmup_update


MODEL_BASE = "PINN base"
MODEL_PCA = "PCA AD"


def fit_pca_basis(x, u, fit_ids, variance_threshold):
    """Fit a spatial PCA basis using training times only and prepare a C1 spline."""
    snapshots = u[:, fit_ids].T
    mean = snapshots.mean(axis=0).astype(np.float32)
    centered = snapshots - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    cumulative = np.cumsum(variance) / variance.sum()
    k = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    modes = right_vectors[:k].T.astype(np.float32)
    fields = np.column_stack((mean, modes)).astype(np.float32)
    slopes = np.gradient(fields, x, axis=0, edge_order=2).astype(np.float32)
    return tuple(jnp.asarray(a) for a in (x, fields, slopes)), k, float(cumulative[k - 1])


def fit_pca_full_basis(x, u, fit_ids):
    """Fit one full training-only basis so a k sweep shares the same SVD."""
    snapshots = u[:, fit_ids].T
    mean = snapshots.mean(axis=0).astype(np.float32)
    centered = snapshots - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    cumulative = np.cumsum(variance) / variance.sum()
    fields = np.column_stack((mean, right_vectors.T)).astype(np.float32)
    slopes = np.gradient(fields, x, axis=0, edge_order=2).astype(np.float32)
    full_basis = tuple(jnp.asarray(a) for a in (x, fields, slopes))
    return full_basis, cumulative


def select_pca_components(full_basis, cumulative_variance, components):
    """Return the leading k modes and their retained variance."""
    k = int(components)
    if k < 1 or k > full_basis[1].shape[1] - 1:
        raise ValueError(f"k={k} excede el rango PCA disponible.")
    basis = (full_basis[0], full_basis[1][:, :k + 1], full_basis[2][:, :k + 1])
    return basis, float(cumulative_variance[k - 1])


def matched_pca_width(k, target_parameters):
    """Select the one-hidden-width MLP nearest the baseline parameter budget."""
    best_width, best_count = None, None
    for width in range(1, 512):
        count = parameter_count([1, width, width, width, k])
        if best_count is None or abs(count - target_parameters) < abs(best_count - target_parameters):
            best_width, best_count = width, count
    return best_width, best_count


def _pca_field_at_x(basis_state, x_query):
    """Cubic Hermite interpolation of PCA mean and modes; differentiable in x."""
    x_grid, fields, slopes = basis_state[:3]
    dx = x_grid[1] - x_grid[0]
    safe_x = jnp.clip(x_query, x_grid[0], x_grid[-1])
    interval = jnp.clip(jnp.floor((safe_x - x_grid[0]) / dx).astype(jnp.int32),
                        0, x_grid.shape[0] - 2)
    left_x = x_grid[interval]
    right_x = x_grid[interval + 1]
    h = right_x - left_x
    s = (safe_x - left_x) / h
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return (h00 * fields[interval] + h10 * h * slopes[interval]
            + h01 * fields[interval + 1] + h11 * h * slopes[interval + 1])


def pca_solution(params, basis_state, x, t):
    interpolated_fields = _pca_field_at_x(basis_state, x)
    coefficients = mlp(params, jnp.asarray([t]))
    return interpolated_fields[0] + jnp.dot(interpolated_fields[1:], coefficients)


def _burgers_residual(solution_at, x, t):
    u = solution_at(x, t)
    ux = jax.grad(solution_at, argnums=0)(x, t)
    ut = jax.grad(solution_at, argnums=1)(x, t)
    uxx = jax.grad(lambda xx: jax.grad(solution_at, argnums=0)(xx, t))(x)
    return ut + u * ux - NU * uxx


def baseline_ad_loss(params, batch):
    data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u = batch
    solution_at = lambda xx, tt: burgers_solution(params, xx, tt)
    predictions = jax.vmap(solution_at)(data_x, data_t)
    residuals = jax.vmap(lambda xx, tt: _burgers_residual(solution_at, xx, tt))(
        residual_x, residual_t)
    initial_predictions = jax.vmap(
        lambda xx: solution_at(xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
    return (jnp.mean((predictions - data_u) ** 2)
            + jnp.mean(residuals**2)
            + jnp.mean((initial_predictions - initial_u) ** 2))


def pca_ad_loss(params, batch, basis_state):
    data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u = batch
    spatial_basis = basis_state[:3]
    physics_weight = basis_state[3] if len(basis_state) > 3 else 1.0
    solution_at = lambda xx, tt: pca_solution(params, spatial_basis, xx, tt)
    predictions = jax.vmap(solution_at)(data_x, data_t)
    residuals = jax.vmap(lambda xx, tt: _burgers_residual(solution_at, xx, tt))(
        residual_x, residual_t)
    initial_predictions = jax.vmap(
        lambda xx: solution_at(xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
    return (jnp.mean((predictions - data_u) ** 2)
            + physics_weight * jnp.mean(residuals**2)
            + jnp.mean((initial_predictions - initial_u) ** 2))


def make_controlled_batches(x, t, u, fit_ids, ic_ids, steps, batch_size, seed):
    """Create one paired sequence of samples for both models."""
    rng = np.random.default_rng(seed)
    batches = []
    ic_x, ic_u = x[ic_ids], u[ic_ids, 0]
    for _ in range(steps):
        data_x_ids = rng.integers(0, x.size, batch_size)
        data_t_ids = rng.choice(fit_ids, batch_size)
        residual_x = rng.uniform(float(x.min()), float(x.max()), batch_size).astype(np.float32)
        residual_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        batches.append(tuple(jnp.asarray(a) for a in (
            x[data_x_ids], t[data_t_ids], u[data_x_ids, data_t_ids],
            residual_x, residual_t, ic_x, ic_u,
        )))
    return batches


def _evaluate_rmse(params, x, times, truth, basis_state=None):
    if basis_state is None:
        prediction = _predict_baseline(params, jnp.asarray(x), jnp.asarray(times))
    else:
        prediction = _predict_pca_grid(params, basis_state, jnp.asarray(times))
    return jnp.sqrt(jnp.mean((prediction - jnp.asarray(truth)) ** 2))


@jax.jit
def _predict_baseline(params, x, times):
    xx, tt = jnp.meshgrid(x, times, indexing="ij")
    values = jax.vmap(lambda a, b: burgers_solution(params, a, b))(
        xx.reshape(-1), tt.reshape(-1))
    return values.reshape((x.shape[0], times.shape[0]))


@jax.jit
def _predict_pca_grid(params, basis_state, times):
    _, fields, _ = basis_state[:3]
    coefficients = jax.vmap(lambda tt: mlp(params, jnp.asarray([tt])))(times)
    return fields[:, 0, None] + fields[:, 1:] @ coefficients.T


def _compiled_validation_call(params, x, times, truth, basis_state=None):
    value = _evaluate_rmse(params, x, times, truth, basis_state)
    return float(value.block_until_ready())


def _train_trace(update, params, batches, validation_fn, eval_every, target_values,
                 basis_state=None, preprocessing_seconds=0.0, stop_rmse=None):
    first_moment, second_moment, step = adam_init(params)
    target_times = {str(target): None for target in target_values}
    trace = []
    start = time.perf_counter()
    final_loss = jnp.asarray(jnp.nan)
    stopped_step = len(batches)
    for step_index, batch in enumerate(batches, start=1):
        if basis_state is None:
            params, first_moment, second_moment, step, final_loss = update(
                params, first_moment, second_moment, step, batch)
        else:
            params, first_moment, second_moment, step, final_loss = update(
                params, first_moment, second_moment, step, batch, basis_state)
        loss_value = float(final_loss)
        if step_index % eval_every == 0 or step_index == len(batches):
            validation_rmse = validation_fn(params)
            elapsed = preprocessing_seconds + (time.perf_counter() - start)
            trace.append({
                "step": step_index,
                "elapsed_pipeline_seconds": elapsed,
                "validation_rmse": validation_rmse,
                "training_loss": loss_value,
            })
            for target in target_values:
                key = str(target)
                if target_times[key] is None and validation_rmse <= target:
                    target_times[key] = elapsed
            if stop_rmse is not None and validation_rmse <= float(stop_rmse):
                stopped_step = step_index
                break
    jax.tree.map(lambda value: value.block_until_ready(), params)
    training_seconds = time.perf_counter() - start
    return params, training_seconds, float(final_loss), trace, target_times, stopped_step


def _summarize(rows, model_name):
    selected = [row for row in rows if row["model"] == model_name]
    metrics = ("test_rmse", "test_relative_l2", "test_pde_residual_rms",
               "training_seconds", "pipeline_seconds")
    summary = {
        "seed_count": len(selected),
        "parameters_mean": float(np.mean([r["parameters"] for r in selected])),
        "parameters_min": int(min(r["parameters"] for r in selected)),
        "parameters_max": int(max(r["parameters"] for r in selected)),
    }
    for field in ("k", "retained_variance", "pca_fit_seconds"):
        vals = [r[field] for r in selected if r[field] is not None]
        summary[field] = mean_std(vals) if vals else None
    for metric in metrics:
        summary[metric] = mean_std([r[metric] for r in selected])
    target_names = [key for key in selected[0] if key.startswith("time_to_val_rmse_")]
    summary["time_to_target"] = {}
    for target_name in target_names:
        hits = [r[target_name] for r in selected if r[target_name] is not None]
        summary["time_to_target"][target_name.removeprefix("time_to_")] = {
            "hit_count": len(hits), "seed_count": len(selected),
            "seconds_mean_if_hit": float(np.mean(hits)) if hits else None,
            "seconds_std_if_hit": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
        }
    return summary


def _paired_differences(rows, pca_name):
    base = {r["seed"]: r for r in rows if r["model"] == MODEL_BASE}
    pca = {r["seed"]: r for r in rows if r["model"] == pca_name}
    fields = ("test_rmse", "test_relative_l2", "test_pde_residual_rms", "pipeline_seconds")
    result = {}
    for field in fields:
        delta = np.asarray([pca[s][field] - base[s][field] for s in sorted(base)])
        result[field] = {
            "PCA_minus_base_mean": float(delta.mean()),
            "PCA_minus_base_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
            "PCA_better_count": int(np.sum(delta < 0)), "n": int(delta.size),
        }
    return result


def run_controlled_comparison(config: dict, project_root: str | Path):
    """Run a paired benchmark with AD used for both PDE residuals."""
    project_root = Path(project_root).resolve()
    data_path = project_root / config["data_path"]
    output_dir = project_root / "results" / "burgers" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, t, u = load_burgers_data(data_path)
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-4, atol=1e-7):
        raise ValueError("La interpolación cúbica implementada requiere una malla espacial uniforme.")

    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_params_count = parameter_count(base_layers)
    learning_rate = float(config["learning_rate"])
    base_update = make_update(baseline_ad_loss, learning_rate)
    pca_update = make_update(pca_ad_loss, learning_rate, with_basis=True)
    target_values = [float(v) for v in config["validation_rmse_targets"]]
    eval_every = int(config["eval_every_steps"])
    seeds = [int(seed) for seed in config["seeds"]]
    rows, trajectories, partitions = [], [], {}
    compiled_architectures = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        ic_ids = np.sort(rng.choice(x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = make_controlled_batches(
            x, t, u, fit_ids, ic_ids, int(config["steps"]), int(config["batch_size"]), seed)
        basis_fit_start = time.perf_counter()
        basis, k, retained_variance = fit_pca_basis(
            x, u, fit_ids, float(config["pca_variance_threshold"]))
        pca_fit_seconds = time.perf_counter() - basis_fit_start
        pca_width, pca_params_count = matched_pca_width(k, base_params_count)
        if abs(pca_params_count - base_params_count) / base_params_count > 0.05:
            raise ValueError(f"No se logró emparejar parámetros con k={k} dentro del 5%.")
        pca_name = MODEL_PCA

        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        pca_params = init_mlp(jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
        x_device, val_t_device = jnp.asarray(x), jnp.asarray(t[validation_ids])
        val_truth = jnp.asarray(u[:, validation_ids])

        if "baseline" not in compiled_architectures:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, val_t_device, val_truth)
            compiled_architectures["baseline"] = time.perf_counter() - compile_start
        pca_arch_key = f"pca_k{k}_w{pca_width}"
        if pca_arch_key not in compiled_architectures:
            compile_start = time.perf_counter()
            warmup_update(pca_update, pca_params, batches[0], basis)
            _compiled_validation_call(pca_params, x_device, val_t_device, val_truth, basis)
            compiled_architectures[pca_arch_key] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(), "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(), "initial_condition_x_indices": ic_ids.tolist(),
        }
        print(f"Seed {seed}: k={k}, var={retained_variance:.6%}, params={base_params_count}/{pca_params_count}")

        base_validation = lambda params: _compiled_validation_call(
            params, x_device, val_t_device, val_truth)
        base_params, base_train_s, base_loss_value, base_trace, base_targets, _ = _train_trace(
            base_update, base_params, batches, base_validation, eval_every, target_values)
        base_prediction = np.asarray(_predict_baseline(
            base_params, jnp.asarray(x), jnp.asarray(t)).block_until_ready())
        base_metrics = score_prediction(base_prediction, u, x, t, test_ids)
        base_row = {
            "model": MODEL_BASE, "seed": seed, "k": None, "retained_variance": None,
            "width": base_width, "parameters": base_params_count,
            "pca_fit_seconds": 0.0, "training_seconds": base_train_s,
            "pipeline_seconds": base_train_s, "final_training_loss": base_loss_value,
            "validation_rmse": base_trace[-1]["validation_rmse"],
            "test_rmse": base_metrics["rmse"], "test_relative_l2": base_metrics["relative_l2"],
            "test_pde_residual_rms": base_metrics["pde_residual_rms"],
        }
        for target, seconds in base_targets.items():
            base_row[f"time_to_val_rmse_{target.replace('.', '_')}"] = seconds
        rows.append(base_row)

        pca_validation = lambda params: _compiled_validation_call(
            params, x_device, val_t_device, val_truth, basis)
        pca_params, pca_train_s, pca_loss_value, pca_trace, pca_targets, _ = _train_trace(
            pca_update, pca_params, batches, pca_validation, eval_every, target_values,
            basis_state=basis, preprocessing_seconds=pca_fit_seconds)
        pca_prediction = np.asarray(_predict_pca_grid(
            pca_params, basis, jnp.asarray(t)).block_until_ready())
        pca_metrics = score_prediction(pca_prediction, u, x, t, test_ids)
        pca_row = {
            "model": pca_name, "seed": seed, "k": k, "retained_variance": retained_variance,
            "width": pca_width, "parameters": pca_params_count,
            "pca_fit_seconds": pca_fit_seconds, "training_seconds": pca_train_s,
            "pipeline_seconds": pca_fit_seconds + pca_train_s,
            "final_training_loss": pca_loss_value,
            "validation_rmse": pca_trace[-1]["validation_rmse"],
            "test_rmse": pca_metrics["rmse"], "test_relative_l2": pca_metrics["relative_l2"],
            "test_pde_residual_rms": pca_metrics["pde_residual_rms"],
        }
        for target, seconds in pca_targets.items():
            pca_row[f"time_to_val_rmse_{target.replace('.', '_')}"] = seconds
        rows.append(pca_row)
        for model_name, trace in ((MODEL_BASE, base_trace), (pca_name, pca_trace)):
            for item in trace:
                trajectories.append({"model": model_name, "seed": seed, **item})
        print(f"  Base: relL2={base_metrics['relative_l2']:.5g}, pipeline={base_train_s:.2f}s")
        print(f"  PCA:  relL2={pca_metrics['relative_l2']:.5g}, pipeline={pca_fit_seconds+pca_train_s:.2f}s")

    model_names = [MODEL_BASE] + sorted({row["model"] for row in rows if row["model"] != MODEL_BASE})
    summary = {name: _summarize(rows, name) for name in model_names}
    paired = _paired_differences(rows, model_names[1])
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    metadata = {
        "configuration": config,
        "data": {"path": config["data_path"], "shape_x_t": [int(x.size), int(t.size)]},
        "protocol": {
            "same_residual_operator": "JAX autodifferentiation of u, u_x, u_t, and u_xx for both models",
            "pca_interpolation": "Cubic Hermite interpolation of the training-only PCA mean and modes; PDE derivatives flow through this interpolation with JAX autodiff",
            "paired_sampling": "Same supervised minibatches, residual collocation points, times, initial-condition points, and data partitions for both models within each seed",
            "timing": "Compile and validation-function compilation are performed before timed training and reported separately. Training time includes identical periodic validation evaluations. PCA pipeline time additionally includes PCA fitting.",
            "pca_data_scope": "PCA is fitted only on spatial snapshots at training times; validation and test times are excluded.",
        },
        "parameter_matching": {"baseline": base_params_count,
                               "pca_range": [min(r["parameters"] for r in rows if r["k"] is not None),
                                             max(r["parameters"] for r in rows if r["k"] is not None)]},
        "compilation_seconds_by_architecture": compiled_architectures,
        "runtime": runtime, "elapsed_wall_seconds": time.perf_counter() - started,
        "partitions": partitions, "summary": summary,
        "paired_differences_PCA_minus_base": paired, "runs": rows, "trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    grouped_summary_svg(output_dir / "summary.svg", summary, rows)
    controlled_trajectory_svg(output_dir / "validation_vs_time.svg", trajectories, target_values)
    return {"output_dir": output_dir, "rows": rows, "trajectories": trajectories,
            "summary": summary, "metadata": metadata}
