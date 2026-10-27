"""High-dimensional advection with PCA preprocessing of input coordinates."""

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

from .controlled import _train_trace
from .data import split_times
from .metrics import mean_std
from .models import init_mlp, mlp, parameter_count
from .plotting import (
    checkpoint_curve_svg,
    dimension_scaling_svg,
    grouped_summary_svg,
    metric_tradeoff_svg,
)
from .stats import paired_difference_report, paired_reports_by_model
from .training import make_update, warmup_update


MODEL_BASE = "PINN base"
MODEL_PCA = "PCA input 99%"


def _orthogonal_frame(dimension: int, seed: int):
    rng = np.random.default_rng(seed)
    frame, triangular = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    return frame * signs[None, :]


def _sample_coordinates(rng, count, eigenvalues, frame):
    latent = rng.standard_normal((count, eigenvalues.size)) * np.sqrt(eigenvalues)[None, :]
    return (latent @ frame.T).astype(np.float32)


def _exact_solution(x, times, frame, active_directions=(0, 1),
                    frequency_scales=(1.0, 1.0), temporal_rates=(1.0, 0.5)):
    first = x @ frame[:, int(active_directions[0])]
    second = x @ frame[:, int(active_directions[1])]
    phase_1 = np.pi * (frequency_scales[0] * first[:, None]
                       - temporal_rates[0] * times[None, :])
    phase_2 = np.pi * (frequency_scales[1] * second[:, None]
                       - temporal_rates[1] * times[None, :])
    return (np.sin(phase_1) + 0.5 * np.cos(phase_2)).astype(np.float32)


def fit_coordinate_pca(x_train: np.ndarray, variance_threshold: float):
    """Fit a centered coordinate PCA on training points only."""
    mean = x_train.mean(axis=0).astype(np.float32)
    centered = x_train - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].astype(np.float32)
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    k = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    return mean, components, float(cumulative[k - 1]), k, cumulative


def _solution_base(params, x, t):
    return mlp(params, jnp.concatenate((x, jnp.reshape(t, (1,)))))[0]


def _solution_projected(params, basis, x, t):
    mean, components = basis
    projected = (x - mean) @ components
    return mlp(params, jnp.concatenate((projected, jnp.reshape(t, (1,)))))[0]


def _advection_residual(solution, params, x, t, velocity):
    ux = jax.grad(lambda xx: solution(params, xx, t))(x)
    ut = jax.grad(lambda tt: solution(params, x, tt))(t)
    return ut + jnp.dot(velocity, ux)


def _make_losses(velocity):
    velocity_device = jnp.asarray(velocity)

    def baseline_loss(params, batch):
        data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u = batch
        predictions = jax.vmap(lambda x, t: _solution_base(params, x, t))(data_x, data_t)
        residuals = jax.vmap(lambda x, t: _advection_residual(
            _solution_base, params, x, t, velocity_device))(residual_x, residual_t)
        initial_predictions = jax.vmap(
            lambda x: _solution_base(params, x, jnp.asarray(0.0, dtype=x.dtype)))(initial_x)
        return (jnp.mean((predictions - data_u) ** 2)
                + jnp.mean(residuals ** 2)
                + jnp.mean((initial_predictions - initial_u) ** 2))

    def projected_loss(params, batch, basis):
        data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u = batch
        solution = lambda p, x, t: _solution_projected(p, basis, x, t)
        predictions = jax.vmap(lambda x, t: solution(params, x, t))(data_x, data_t)
        residuals = jax.vmap(lambda x, t: _advection_residual(
            solution, params, x, t, velocity_device))(residual_x, residual_t)
        initial_predictions = jax.vmap(
            lambda x: solution(params, x, jnp.asarray(0.0, dtype=x.dtype)))(initial_x)
        return (jnp.mean((predictions - data_u) ** 2)
                + jnp.mean(residuals ** 2)
                + jnp.mean((initial_predictions - initial_u) ** 2))

    return baseline_loss, projected_loss


def _make_batches(x_train, t, u_train, fit_ids, initial_ids,
                  steps, batch_size, seed, residual_bounds=None):
    rng = np.random.default_rng(seed)
    initial_x = x_train[initial_ids]
    initial_u = u_train[initial_ids, 0]
    batches = []
    for _ in range(steps):
        data_ids = rng.integers(0, x_train.shape[0], batch_size)
        data_times = rng.choice(fit_ids, batch_size)
        residual_times = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        if residual_bounds is None:
            residual_ids = rng.integers(0, x_train.shape[0], batch_size)
            residual_x = x_train[residual_ids]
        else:
            low, high = residual_bounds
            residual_x = rng.uniform(low, high, size=(batch_size, low.size)).astype(np.float32)
        batch = (
            x_train[data_ids], t[data_times], u_train[data_ids, data_times],
            residual_x, residual_times, initial_x, initial_u,
        )
        batches.append(tuple(jnp.asarray(value) for value in batch))
    return batches


@jax.jit
def _predict_base_grid(params, x, times):
    n_x, n_t = x.shape[0], times.shape[0]
    xx = jnp.broadcast_to(x[:, None, :], (n_x, n_t, x.shape[1])).reshape((-1, x.shape[1]))
    tt = jnp.broadcast_to(times[None, :], (n_x, n_t)).reshape((-1,))
    values = jax.vmap(lambda a, b: _solution_base(params, a, b))(xx, tt)
    return values.reshape((n_x, n_t))


@jax.jit
def _predict_projected_grid(params, basis, x, times):
    mean, components = basis
    projected = (x - mean) @ components
    n_x, n_t = projected.shape[0], times.shape[0]
    zz = jnp.broadcast_to(
        projected[:, None, :], (n_x, n_t, projected.shape[1])).reshape((-1, projected.shape[1]))
    tt = jnp.broadcast_to(times[None, :], (n_x, n_t)).reshape((-1,))
    values = jax.vmap(
        lambda z, t: mlp(params, jnp.concatenate((z, jnp.reshape(t, (1,)))))[0])(zz, tt)
    return values.reshape((n_x, n_t))


@jax.jit
def _validation_rmse_base(params, x, times, truth):
    prediction = _predict_base_grid(params, x, times)
    return jnp.sqrt(jnp.mean((prediction - truth) ** 2))


@jax.jit
def _validation_rmse_projected(params, basis, x, times, truth):
    prediction = _predict_projected_grid(params, basis, x, times)
    return jnp.sqrt(jnp.mean((prediction - truth) ** 2))


@jax.jit
def _residuals_base(params, x, times, velocity):
    return jax.vmap(lambda xx, tt: _advection_residual(
        _solution_base, params, xx, tt, velocity))(x, times)


@jax.jit
def _residuals_projected(params, basis, x, times, velocity):
    solution = lambda p, xx, tt: _solution_projected(p, basis, xx, tt)
    return jax.vmap(lambda xx, tt: _advection_residual(
        solution, params, xx, tt, velocity))(x, times)


def _nearest_width(input_dimension, target_parameters):
    best_width, best_count = None, None
    for width in range(1, 512):
        count = parameter_count([input_dimension, width, width, width, 1])
        if best_count is None or abs(count - target_parameters) < abs(best_count - target_parameters):
            best_width, best_count = width, count
    return best_width, best_count


def _sample_pde_test_points(x_test, t, test_ids, count, seed, residual_bounds=None):
    rng = np.random.default_rng(seed + 84971)
    times = rng.choice(t[test_ids], size=count).astype(np.float32)
    if residual_bounds is None:
        x_ids = rng.integers(0, x_test.shape[0], size=count)
        return x_test[x_ids], times
    low, high = residual_bounds
    residual_x = rng.uniform(low, high, size=(count, low.size)).astype(np.float32)
    return residual_x, times


def _residual_bounds(x_train, config):
    if config.get("residual_sampling") != "uniform_box":
        return None
    pad = float(config.get("residual_box_padding", 0.1))
    span = np.maximum(x_train.max(axis=0) - x_train.min(axis=0), 1e-6)
    return x_train.min(axis=0) - pad * span, x_train.max(axis=0) + pad * span


def _subspace_capture(components, direction):
    captured = components.T @ np.asarray(direction, dtype=np.float64)
    return float(np.sum(captured ** 2) / max(np.sum(np.asarray(direction, dtype=np.float64) ** 2), 1e-12))


def _score(params, basis, x_test, t, truth, time_ids, residual_x, residual_t, velocity):
    x_device = jnp.asarray(x_test)
    test_times = jnp.asarray(t[time_ids])
    if basis is None:
        prediction = np.asarray(_predict_base_grid(params, x_device, test_times).block_until_ready())
        residuals = np.asarray(_residuals_base(
            params, jnp.asarray(residual_x), jnp.asarray(residual_t), jnp.asarray(velocity)).block_until_ready())
    else:
        prediction = np.asarray(_predict_projected_grid(
            params, basis, x_device, test_times).block_until_ready())
        residuals = np.asarray(_residuals_projected(
            params, basis, jnp.asarray(residual_x), jnp.asarray(residual_t),
            jnp.asarray(velocity)).block_until_ready())
    difference = prediction - truth[:, time_ids]
    return {
        "test_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "test_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(truth[:, time_ids])),
        "test_pde_residual_rms": float(np.sqrt(np.mean(residuals ** 2))),
    }


def _summary(rows, dimension, model):
    selected = [row for row in rows if row["dimension"] == dimension and row["model"] == model]
    result = {
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
    for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms",
                   "validation_rmse", "training_seconds", "pipeline_seconds"):
        result[metric] = mean_std([row[metric] for row in selected])
    return result


def run_high_dim_advection(config: dict, project_root: str | Path):
    """Run paired baseline/input-PCA PINNs across increasing input dimensions."""
    project_root = Path(project_root).resolve()
    output_dir = (project_root / "results" / "high_dimensional_advection"
                  / "input_pca_vs_base" / config["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, trajectories, partitions, compilation = [], [], {}, {}
    dimension_summaries = {}
    threshold = float(config["pca_variance_threshold"])

    for dimension in [int(value) for value in config["dimensions"]]:
        frame = _orthogonal_frame(dimension, int(config["problem_seed"]) + dimension)
        eigenvalues = float(config["covariance_decay"]) ** np.arange(dimension, dtype=np.float64)
        velocity = frame[:, 0] + 0.5 * frame[:, 1]
        baseline_width = int(config["baseline_width"])
        baseline_layers = [dimension + 1, baseline_width, baseline_width, baseline_width, 1]
        baseline_parameters = parameter_count(baseline_layers)
        baseline_loss, projected_loss = _make_losses(velocity)
        baseline_update = make_update(baseline_loss, float(config["learning_rate"]))
        projected_update = make_update(projected_loss, float(config["learning_rate"]), with_basis=True)
        dimension_rows = []

        for seed in [int(value) for value in config["seeds"]]:
            rng = np.random.default_rng(seed + 1009 * dimension)
            x_train = _sample_coordinates(rng, int(config["training_spatial_points"]), eigenvalues, frame)
            x_validation = _sample_coordinates(rng, int(config["validation_spatial_points"]), eigenvalues, frame)
            x_test = _sample_coordinates(rng, int(config["test_spatial_points"]), eigenvalues, frame)
            t = np.linspace(0.0, float(config["final_time"]), int(config["time_points"]), dtype=np.float32)
            u_train = _exact_solution(x_train, t, frame)
            u_validation = _exact_solution(x_validation, t, frame)
            u_test = _exact_solution(x_test, t, frame)
            fit_ids, validation_ids, test_ids = split_times(
                t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))

            pca_start = time.perf_counter()
            mean, components, retained_variance, k, cumulative = fit_coordinate_pca(
                x_train, threshold)
            pca_fit_seconds = time.perf_counter() - pca_start
            basis = (jnp.asarray(mean), jnp.asarray(components[:, :k]))
            projected_width, projected_parameters = _nearest_width(k + 1, baseline_parameters)
            if abs(projected_parameters - baseline_parameters) / baseline_parameters > 0.05:
                raise ValueError(
                    f"No se emparejó el número de parámetros para d={dimension}, k={k}."
                )

            initial_rng = np.random.default_rng(seed + 711)
            initial_ids = np.sort(initial_rng.choice(
                x_train.shape[0], size=min(int(config["initial_condition_points"]), x_train.shape[0]),
                replace=False))
            residual_bounds = _residual_bounds(x_train, config)
            batches = _make_batches(
                x_train, t, u_train, fit_ids, initial_ids, int(config["steps"]),
                int(config["batch_size"]), seed, residual_bounds=residual_bounds)
            x_device, x_val_device = jnp.asarray(x_validation), jnp.asarray(x_validation)
            validation_times = jnp.asarray(t[validation_ids])
            validation_truth = jnp.asarray(u_validation[:, validation_ids])

            base_params = init_mlp(jax.random.PRNGKey(seed), baseline_layers)
            projected_layers = [k + 1, projected_width, projected_width, projected_width, 1]
            projected_params = init_mlp(jax.random.PRNGKey(seed), projected_layers)
            base_key = f"d{dimension}_base"
            projected_key = f"d{dimension}_pca_k{k}_w{projected_width}"
            if base_key not in compilation:
                compile_start = time.perf_counter()
                warmup_update(baseline_update, base_params, batches[0])
                _validation_rmse_base(base_params, x_val_device, validation_times,
                                      validation_truth).block_until_ready()
                compilation[base_key] = time.perf_counter() - compile_start
            if projected_key not in compilation:
                compile_start = time.perf_counter()
                warmup_update(projected_update, projected_params, batches[0], basis)
                _validation_rmse_projected(projected_params, basis, x_val_device,
                                           validation_times, validation_truth).block_until_ready()
                compilation[projected_key] = time.perf_counter() - compile_start

            partitions[f"d{dimension}_seed{seed}"] = {
                "fit_time_indices": fit_ids.tolist(),
                "validation_time_indices": validation_ids.tolist(),
                "test_time_indices": test_ids.tolist(),
                "training_spatial_count": int(x_train.shape[0]),
                "validation_spatial_count": int(x_validation.shape[0]),
                "test_spatial_count": int(x_test.shape[0]),
                "initial_condition_spatial_indices": initial_ids.tolist(),
                "input_pca_k": k,
                "input_pca_retained_variance": retained_variance,
            }

            residual_x, residual_t = _sample_pde_test_points(
                x_test, t, test_ids, int(config["test_residual_points"]), seed,
                residual_bounds=residual_bounds)
            base_validation = lambda params: float(_validation_rmse_base(
                params, x_val_device, validation_times, validation_truth).block_until_ready())
            base_params, base_train_seconds, base_final_loss, base_trace, _, _ = _train_trace(
                baseline_update, base_params, batches, base_validation,
                int(config["eval_every_steps"]), [],
            )
            base_metrics = _score(base_params, None, x_test, t, u_test, test_ids,
                                  residual_x, residual_t, velocity)
            dimension_rows.append({
                "dimension": dimension, "model": MODEL_BASE, "seed": seed,
                "k": None, "retained_variance": None,
                "network_input_dimension": dimension + 1, "width": baseline_width,
                "parameters": baseline_parameters, "pca_fit_seconds": 0.0,
                "training_seconds": base_train_seconds,
                "pipeline_seconds": base_train_seconds,
                "final_training_loss": base_final_loss,
                "validation_rmse": base_trace[-1]["validation_rmse"],
                **base_metrics,
            })
            trajectories.extend({"dimension": dimension, "model": MODEL_BASE,
                                 "seed": seed, **record} for record in base_trace)

            projected_validation = lambda params: float(_validation_rmse_projected(
                params, basis, x_val_device, validation_times, validation_truth).block_until_ready())
            projected_params, projected_train_seconds, projected_final_loss, projected_trace, _, _ = _train_trace(
                projected_update, projected_params, batches, projected_validation,
                int(config["eval_every_steps"]), [], basis_state=basis,
                preprocessing_seconds=pca_fit_seconds,
            )
            projected_metrics = _score(projected_params, basis, x_test, t, u_test, test_ids,
                                       residual_x, residual_t, velocity)
            dimension_rows.append({
                "dimension": dimension, "model": MODEL_PCA, "seed": seed,
                "k": k, "retained_variance": retained_variance,
                "network_input_dimension": k + 1, "width": projected_width,
                "parameters": projected_parameters, "pca_fit_seconds": pca_fit_seconds,
                "training_seconds": projected_train_seconds,
                "pipeline_seconds": pca_fit_seconds + projected_train_seconds,
                "final_training_loss": projected_final_loss,
                "validation_rmse": projected_trace[-1]["validation_rmse"],
                **projected_metrics,
            })
            trajectories.extend({"dimension": dimension, "model": MODEL_PCA,
                                 "seed": seed, **record} for record in projected_trace)
            print(f"d={dimension}, seed={seed}: k99={k}, var={retained_variance:.5%}")

        rows.extend(dimension_rows)
        dimension_summaries[str(dimension)] = {
            MODEL_BASE: _summary(dimension_rows, dimension, MODEL_BASE),
            MODEL_PCA: _summary(dimension_rows, dimension, MODEL_PCA),
        }

    paired = {}
    for dimension in [int(value) for value in config["dimensions"]]:
        baseline_by_seed = {row["seed"]: row for row in rows
                            if row["dimension"] == dimension and row["model"] == MODEL_BASE}
        pca_by_seed = {row["seed"]: row for row in rows
                       if row["dimension"] == dimension and row["model"] == MODEL_PCA}
        paired[str(dimension)] = {}
        for metric in ("test_relative_l2", "test_pde_residual_rms", "pipeline_seconds"):
            paired[str(dimension)][metric] = paired_difference_report(
                [baseline_by_seed[seed][metric] for seed in sorted(baseline_by_seed)],
                [pca_by_seed[seed][metric] for seed in sorted(baseline_by_seed)],
            )

    metadata = {
        "configuration": config,
        "problem": {
            "equation": "u_t + v dot grad_x(u) = 0 on R^d",
            "exact_solution": "sin(pi*(q1 dot x - t)) + 0.5*cos(pi*(q2 dot x - 0.5*t))",
            "velocity": "q1 + 0.5*q2, where q1 and q2 are the first two covariance eigenvectors",
            "coordinate_distribution": "Gaussian with covariance eigenvalues covariance_decay**j in a random orthogonal frame",
            "active_intrinsic_dimension": 2,
            "pca_scope": "Centered PCA is fitted only on training spatial coordinates; JAX autodiff propagates through the fixed linear projection to original coordinates.",
            "sampling": "The baseline and PCA models share each seed's train, validation, test, initial-condition, supervised-batch, and PDE-collocation samples.",
            "test_residual": "Autodifferentiated PDE residual on the same held-out coordinate distribution and held-out time split for both models.",
        },
        "runtime": {
            "python": platform.python_version(), "jax": jax.__version__,
            "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "device": str(jax.devices()[0]),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "compilation_seconds_by_architecture": compilation,
        "partitions": partitions,
        "summary_by_dimension": dimension_summaries,
        "paired_differences_PCA_minus_base": paired,
        "runs": rows,
        "training_trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    for dimension in [int(value) for value in config["dimensions"]]:
        local_rows = [row for row in rows if row["dimension"] == dimension]
        grouped_summary_svg(
            output_dir / f"comparison_d{dimension}.svg",
            dimension_summaries[str(dimension)], local_rows, pde_name="advección",
        )
        local_trajectories = [row for row in trajectories if row["dimension"] == dimension]
        checkpoint_curve_svg(
            output_dir / f"validation_d{dimension}.svg", local_trajectories,
            "validation_rmse", f"Validación en advección, d={dimension}", "RMSE",
        )
    dimension_scaling_svg(output_dir / "dimension_scaling.svg", dimension_summaries)
    return {"output_dir": output_dir, "summary": dimension_summaries,
            "rows": rows, "trajectories": trajectories, "metadata": metadata}


def run_misaligned_high_dim_advection(config: dict, project_root: str | Path):
    """Test whether a low-variance but physically active coordinate is discarded."""
    project_root = Path(project_root).resolve()
    output_dir = (project_root / "results" / "high_dimensional_advection"
                  / "alignment_sensitivity" / config["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dimension = int(config["dimension"])
    frame = _orthogonal_frame(dimension, int(config["problem_seed"]) + dimension)
    eigenvalues = float(config["covariance_decay"]) ** np.arange(dimension, dtype=np.float64)
    active_directions = tuple(int(v) for v in config["active_directions"])
    temporal_rates = tuple(float(v) for v in config.get("temporal_rates", [1.0, 0.5]))
    if config.get("scale_frequency_by_coordinate_std", False):
        frequency_scales = tuple(1.0 / np.sqrt(eigenvalues[index])
                                 for index in active_directions)
    else:
        frequency_scales = tuple(float(v) for v in config.get("frequency_scales", [1.0, 1.0]))
    velocity = np.zeros(dimension, dtype=np.float64)
    for direction, frequency, rate in zip(active_directions, frequency_scales, temporal_rates):
        velocity += frame[:, direction] * (rate / frequency)
    velocity = velocity.astype(np.float32)

    baseline_width = int(config["baseline_width"])
    baseline_layers = [dimension + 1, baseline_width, baseline_width, baseline_width, 1]
    baseline_parameters = parameter_count(baseline_layers)
    baseline_loss, projected_loss = _make_losses(velocity)
    baseline_update = make_update(baseline_loss, float(config["learning_rate"]))
    projected_update = make_update(projected_loss, float(config["learning_rate"]), with_basis=True)
    rows, trajectories, partitions, compilation = [], [], {}, {}
    threshold = float(config["pca_variance_threshold"])
    fixed_components = [int(k) for k in config["fixed_pca_components"]]

    for seed in [int(value) for value in config["seeds"]]:
        rng = np.random.default_rng(seed + 1009 * dimension)
        x_train = _sample_coordinates(rng, int(config["training_spatial_points"]), eigenvalues, frame)
        x_validation = _sample_coordinates(rng, int(config["validation_spatial_points"]), eigenvalues, frame)
        x_test = _sample_coordinates(rng, int(config["test_spatial_points"]), eigenvalues, frame)
        t = np.linspace(0.0, float(config["final_time"]), int(config["time_points"]), dtype=np.float32)
        u_train = _exact_solution(x_train, t, frame, active_directions,
                                  frequency_scales, temporal_rates)
        u_validation = _exact_solution(x_validation, t, frame, active_directions,
                                       frequency_scales, temporal_rates)
        u_test = _exact_solution(x_test, t, frame, active_directions,
                                 frequency_scales, temporal_rates)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))

        pca_start = time.perf_counter()
        mean, full_components, _, auto_k, cumulative = fit_coordinate_pca(x_train, threshold)
        pca_fit_seconds = time.perf_counter() - pca_start
        component_specs = [(MODEL_PCA, auto_k)]
        component_specs.extend((f"PCA input k={k}", k) for k in fixed_components if k != auto_k)
        projected_models = []
        for name, k in component_specs:
            if k < 1 or k > full_components.shape[1]:
                raise ValueError(f"k={k} está fuera del rango PCA para d={dimension}.")
            basis = (jnp.asarray(mean), jnp.asarray(full_components[:, :k]))
            retained = float(cumulative[k - 1])
            width, parameter_total = _nearest_width(k + 1, baseline_parameters)
            if abs(parameter_total - baseline_parameters) / baseline_parameters > 0.05:
                raise ValueError(f"No se emparejaron parámetros para d={dimension}, k={k}.")
            params = init_mlp(jax.random.PRNGKey(seed), [k + 1, width, width, width, 1])
            projected_models.append({
                "name": name, "k": k, "retained_variance": retained,
                "basis": basis, "width": width, "parameters": parameter_total,
                "params": params,
            })

        initial_rng = np.random.default_rng(seed + 711)
        initial_ids = np.sort(initial_rng.choice(
            x_train.shape[0], size=min(int(config["initial_condition_points"]), x_train.shape[0]),
            replace=False))
        batches = _make_batches(
            x_train, t, u_train, fit_ids, initial_ids, int(config["steps"]),
            int(config["batch_size"]), seed)
        x_val_device = jnp.asarray(x_validation)
        validation_times = jnp.asarray(t[validation_ids])
        validation_truth = jnp.asarray(u_validation[:, validation_ids])
        base_params = init_mlp(jax.random.PRNGKey(seed), baseline_layers)
        base_key = f"d{dimension}_base"
        if base_key not in compilation:
            compile_start = time.perf_counter()
            warmup_update(baseline_update, base_params, batches[0])
            _validation_rmse_base(base_params, x_val_device, validation_times,
                                  validation_truth).block_until_ready()
            compilation[base_key] = time.perf_counter() - compile_start
        for model in projected_models:
            key = f"d{dimension}_pca_k{model['k']}_w{model['width']}"
            if key not in compilation:
                compile_start = time.perf_counter()
                warmup_update(projected_update, model["params"], batches[0], model["basis"])
                _validation_rmse_projected(model["params"], model["basis"], x_val_device,
                                           validation_times, validation_truth).block_until_ready()
                compilation[key] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(),
            "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(),
            "initial_condition_spatial_indices": initial_ids.tolist(),
            "input_pca_auto_k": auto_k,
            "training_spatial_count": int(x_train.shape[0]),
            "validation_spatial_count": int(x_validation.shape[0]),
            "test_spatial_count": int(x_test.shape[0]),
        }
        residual_x, residual_t = _sample_pde_test_points(
            x_test, t, test_ids, int(config["test_residual_points"]), seed)
        base_validation = lambda params: float(_validation_rmse_base(
            params, x_val_device, validation_times, validation_truth).block_until_ready())
        base_params, base_train_seconds, base_final_loss, base_trace, _, _ = _train_trace(
            baseline_update, base_params, batches, base_validation,
            int(config["eval_every_steps"]), [],
        )
        base_metrics = _score(base_params, None, x_test, t, u_test, test_ids,
                              residual_x, residual_t, velocity)
        rows.append({
            "dimension": dimension, "model": MODEL_BASE, "seed": seed,
            "k": None, "retained_variance": None,
            "network_input_dimension": dimension + 1, "width": baseline_width,
            "parameters": baseline_parameters, "pca_fit_seconds": 0.0,
            "training_seconds": base_train_seconds, "pipeline_seconds": base_train_seconds,
            "final_training_loss": base_final_loss,
            "validation_rmse": base_trace[-1]["validation_rmse"], **base_metrics,
        })
        trajectories.extend({"dimension": dimension, "model": MODEL_BASE,
                             "seed": seed, **record} for record in base_trace)

        for model in projected_models:
            model_validation = lambda params, b=model["basis"]: float(
                _validation_rmse_projected(params, b, x_val_device,
                                           validation_times, validation_truth).block_until_ready())
            params, train_seconds, final_loss, trace, _, _ = _train_trace(
                projected_update, model["params"], batches, model_validation,
                int(config["eval_every_steps"]), [], basis_state=model["basis"],
                preprocessing_seconds=pca_fit_seconds,
            )
            metrics = _score(params, model["basis"], x_test, t, u_test, test_ids,
                             residual_x, residual_t, velocity)
            rows.append({
                "dimension": dimension, "model": model["name"], "seed": seed,
                "k": model["k"], "retained_variance": model["retained_variance"],
                "network_input_dimension": model["k"] + 1, "width": model["width"],
                "parameters": model["parameters"], "pca_fit_seconds": pca_fit_seconds,
                "training_seconds": train_seconds,
                "pipeline_seconds": pca_fit_seconds + train_seconds,
                "final_training_loss": final_loss,
                "validation_rmse": trace[-1]["validation_rmse"], **metrics,
            })
            trajectories.extend({"dimension": dimension, "model": model["name"],
                                 "seed": seed, **record} for record in trace)
        print(f"seed={seed}: automatic k={auto_k}; fixed k={fixed_components}")

    model_names = list(dict.fromkeys(row["model"] for row in rows))
    summaries = {name: _summary(rows, dimension, name) for name in model_names}
    paired = {}
    base_by_seed = {row["seed"]: row for row in rows if row["model"] == MODEL_BASE}
    for name in model_names:
        if name == MODEL_BASE:
            continue
        pca_by_seed = {row["seed"]: row for row in rows if row["model"] == name}
        paired[name] = {}
        for metric in ("test_relative_l2", "test_pde_residual_rms", "pipeline_seconds"):
            differences = np.asarray([pca_by_seed[seed][metric] - base_by_seed[seed][metric]
                                      for seed in sorted(base_by_seed)])
            paired[name][metric] = {
                "PCA_minus_base_mean": float(differences.mean()),
                "PCA_minus_base_std": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
                "PCA_better_count": int(np.sum(differences < 0)), "n": int(differences.size),
            }

    metadata = {
        "configuration": config,
        "problem": {
            "equation": "u_t + v dot grad_x(u) = 0 on R^d",
            "active_directions": list(active_directions),
            "frequency_scales": list(frequency_scales),
            "temporal_rates": list(temporal_rates),
            "velocity_coefficients": [rate / frequency for rate, frequency
                                       in zip(temporal_rates, frequency_scales)],
            "active_directions_are": "q1 is a leading high-variance direction; q4 lies below the 99%-variance PCA cutoff.",
            "pca_scope": "Centered coordinate PCA is fit on training coordinates only; JAX autodiff propagates through each fixed linear projection.",
            "sampling": "The base and PCA models share the same spatial samples, time partitions, supervised batches, initial-condition points, and PDE residual points per seed.",
        },
        "runtime": {
            "python": platform.python_version(), "jax": jax.__version__,
            "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "device": str(jax.devices()[0]),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "compilation_seconds_by_architecture": compilation,
        "partitions": partitions,
        "summary": summaries,
        "paired_differences_PCA_minus_base": paired,
        "runs": rows,
        "training_trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    grouped_summary_svg(output_dir / "comparison.svg", summaries, rows, pde_name="advección")
    metric_tradeoff_svg(output_dir / "error_residual_tradeoff.svg", summaries,
                        "test_relative_l2", "test_pde_residual_rms",
                        "Error relativo L2 de prueba", "Residuo RMS de advección",
                        "Varianza y relevancia física: comparación en dimensión alta")
    checkpoint_curve_svg(output_dir / "validation_rmse_vs_steps.svg", trajectories,
                         "validation_rmse", "Validación: varianza alineada y desalineada",
                         "RMSE")
    return {"output_dir": output_dir, "summary": summaries,
            "rows": rows, "trajectories": trajectories, "metadata": metadata}
