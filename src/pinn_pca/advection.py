"""Initial paired PINN versus spatial-PCA benchmark for periodic advection."""

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
    _compiled_validation_call,
    _predict_baseline,
    _predict_pca_grid,
    _train_trace,
    fit_pca_basis,
    matched_pca_width,
)
from .data import split_times
from .metrics import mean_std
from .models import burgers_solution, init_mlp, parameter_count
from .plotting import controlled_trajectory_svg, grouped_summary_svg
from .training import make_update, warmup_update


MODEL_BASE = "PINN base"
MODEL_PCA = "PCA AD"


DEFAULT_ADVECTION_TERMS = [
    {"harmonic": 1, "sin": 1.0, "cos": 0.0},
    {"harmonic": 2, "sin": 0.0, "cos": 0.5},
]


def make_periodic_advection_data(nx=256, nt=100, velocity=1.0,
                                  fourier_terms=None):
    """Generate an analytic periodic-advection solution from Fourier terms."""
    x = np.linspace(-1.0, 1.0, int(nx), dtype=np.float32)
    t = np.linspace(0.0, 1.0, int(nt), dtype=np.float32)
    phase = np.pi * (x[:, None] - float(velocity) * t[None, :])
    terms = DEFAULT_ADVECTION_TERMS if fourier_terms is None else fourier_terms
    u = np.zeros_like(phase)
    for term in terms:
        harmonic = int(term["harmonic"])
        u += (float(term.get("sin", 0.0)) * np.sin(harmonic * phase)
              + float(term.get("cos", 0.0)) * np.cos(harmonic * phase))
    return x, t, u.astype(np.float32)


def _make_losses(velocity):
    def residual(solution_at, x_value, t_value):
        ux = jax.grad(solution_at, argnums=0)(x_value, t_value)
        ut = jax.grad(solution_at, argnums=1)(x_value, t_value)
        return ut + velocity * ux

    def baseline_loss(params, batch):
        (data_x, data_t, data_u, residual_x, residual_t,
         initial_x, initial_u, boundary_t) = batch
        solution_at = lambda xx, tt: burgers_solution(params, xx, tt)
        data_predictions = jax.vmap(solution_at)(data_x, data_t)
        residuals = jax.vmap(lambda xx, tt: residual(solution_at, xx, tt))(
            residual_x, residual_t)
        initial_predictions = jax.vmap(
            lambda xx: solution_at(xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
        left = jax.vmap(lambda tt: solution_at(jnp.asarray(-1.0, dtype=tt.dtype), tt))(boundary_t)
        right = jax.vmap(lambda tt: solution_at(jnp.asarray(1.0, dtype=tt.dtype), tt))(boundary_t)
        return (jnp.mean((data_predictions - data_u) ** 2)
                + jnp.mean(residuals**2)
                + jnp.mean((initial_predictions - initial_u) ** 2)
                + jnp.mean((left - right) ** 2))

    def pca_loss(params, batch, basis_state):
        from .controlled import pca_solution

        (data_x, data_t, data_u, residual_x, residual_t,
         initial_x, initial_u, boundary_t) = batch
        solution_at = lambda xx, tt: pca_solution(params, basis_state, xx, tt)
        data_predictions = jax.vmap(solution_at)(data_x, data_t)
        residuals = jax.vmap(lambda xx, tt: residual(solution_at, xx, tt))(
            residual_x, residual_t)
        initial_predictions = jax.vmap(
            lambda xx: solution_at(xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
        left = jax.vmap(lambda tt: solution_at(jnp.asarray(-1.0, dtype=tt.dtype), tt))(boundary_t)
        right = jax.vmap(lambda tt: solution_at(jnp.asarray(1.0, dtype=tt.dtype), tt))(boundary_t)
        return (jnp.mean((data_predictions - data_u) ** 2)
                + jnp.mean(residuals**2)
                + jnp.mean((initial_predictions - initial_u) ** 2)
                + jnp.mean((left - right) ** 2))

    return baseline_loss, pca_loss


def _make_advection_batches(x, t, u, fit_ids, initial_ids, steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    initial_x, initial_u = x[initial_ids], u[initial_ids, 0]
    batches = []
    for _ in range(steps):
        data_ids = rng.integers(0, x.size, batch_size)
        data_times = rng.choice(fit_ids, batch_size)
        residual_x = rng.uniform(float(x.min()), float(x.max()), batch_size).astype(np.float32)
        residual_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        boundary_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        batches.append(tuple(jnp.asarray(value) for value in (
            x[data_ids], t[data_times], u[data_ids, data_times], residual_x, residual_t,
            initial_x, initial_u, boundary_t,
        )))
    return batches


def _advection_residual_grid(prediction, x, t, velocity):
    ux = np.gradient(prediction, x, axis=0, edge_order=2)
    ut = np.gradient(prediction, t, axis=1, edge_order=2)
    return ut + velocity * ux


def _score(prediction, truth, x, t, time_ids, velocity):
    difference = prediction[:, time_ids] - truth[:, time_ids]
    residual = _advection_residual_grid(prediction, x, t, velocity)[1:-1, :][:, time_ids]
    boundary_gap = prediction[0, time_ids] - prediction[-1, time_ids]
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(truth[:, time_ids])),
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "periodic_boundary_rms": float(np.sqrt(np.mean(boundary_gap**2))),
    }


def _summarize(rows, model_name):
    selected = [row for row in rows if row["model"] == model_name]
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
    for metric in ("validation_rmse", "test_rmse", "test_relative_l2", "test_pde_residual_rms",
                   "test_periodic_boundary_rms", "training_seconds", "pipeline_seconds"):
        summary[metric] = mean_std([row[metric] for row in selected])
    return summary


def run_advection_comparison(config: dict, project_root: str | Path):
    """Run a paired base-PINN versus PCA advection pilot and save artifacts."""
    project_root = Path(project_root).resolve()
    data_path = project_root / config["data_path"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        with np.load(data_path) as data:
            x, t, u = data["x"], data["t"], data["u"]
            velocity = float(data["velocity"])
    else:
        velocity = float(config["velocity"])
        x, t, u = make_periodic_advection_data(
            config["nx"], config["nt"], velocity)
        np.savez_compressed(data_path, x=x, t=t, u=u, velocity=velocity)

    output_dir = project_root / "results" / "advection" / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline_width = int(config["baseline_width"])
    baseline_layers = [2, baseline_width, baseline_width, baseline_width, 1]
    baseline_parameters = parameter_count(baseline_layers)
    base_loss, pca_loss = _make_losses(velocity)
    base_update = make_update(base_loss, float(config["learning_rate"]))
    pca_update = make_update(pca_loss, float(config["learning_rate"]), with_basis=True)
    rows, trajectories, partitions = [], [], {}
    compilation_seconds = {}
    seeds = [int(seed) for seed in config["seeds"]]
    target_values = [float(value) for value in config.get("validation_rmse_targets", [])]

    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        initial_ids = np.sort(rng.choice(
            x.size, size=min(int(config["initial_condition_points"]), x.size), replace=False))
        batches = _make_advection_batches(
            x, t, u, fit_ids, initial_ids, int(config["steps"]),
            int(config["batch_size"]), seed)
        pca_started = time.perf_counter()
        basis, k, retained_variance = fit_pca_basis(
            x, u, fit_ids, float(config["pca_variance_threshold"]))
        pca_fit_seconds = time.perf_counter() - pca_started

        base_params = init_mlp(jax.random.PRNGKey(seed), baseline_layers)
        x_device = jnp.asarray(x)
        validation_t_device = jnp.asarray(t[validation_ids])
        validation_truth = jnp.asarray(u[:, validation_ids])
        if MODEL_BASE not in compilation_seconds:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches[0])
            _compiled_validation_call(base_params, x_device, validation_t_device, validation_truth)
            compilation_seconds[MODEL_BASE] = time.perf_counter() - compile_start
        pca_width, pca_parameters = matched_pca_width(k, baseline_parameters)
        if abs(pca_parameters - baseline_parameters) / baseline_parameters > 0.05:
            raise ValueError("Las arquitecturas deben quedar dentro de 5% de diferencia en parámetros.")
        pca_params = init_mlp(jax.random.PRNGKey(seed), [1, pca_width, pca_width, pca_width, k])
        pca_architecture = f"{MODEL_PCA} k={k}, width={pca_width}"
        if pca_architecture not in compilation_seconds:
            compile_start = time.perf_counter()
            warmup_update(pca_update, pca_params, batches[0], basis)
            _compiled_validation_call(pca_params, x_device, validation_t_device,
                                      validation_truth, basis)
            compilation_seconds[pca_architecture] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(), "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(), "initial_condition_x_indices": initial_ids.tolist(),
        }
        base_val_fn = lambda params: _compiled_validation_call(
            params, x_device, validation_t_device, validation_truth)
        base_params, base_train_seconds, base_loss_value, base_trace, _, _ = _train_trace(
            base_update, base_params, batches, base_val_fn,
            int(config["eval_every_steps"]), target_values)
        base_prediction = np.asarray(_predict_baseline(
            base_params, x_device, jnp.asarray(t)).block_until_ready())
        base_validation = _score(base_prediction, u, x, t, validation_ids, velocity)
        base_test = _score(base_prediction, u, x, t, test_ids, velocity)
        rows.append({
            "model": MODEL_BASE, "seed": seed, "k": None, "retained_variance": None,
            "width": baseline_width, "parameters": baseline_parameters,
            "pca_fit_seconds": 0.0, "training_seconds": base_train_seconds,
            "pipeline_seconds": base_train_seconds, "final_training_loss": base_loss_value,
            "validation_rmse": base_validation["rmse"], "test_rmse": base_test["rmse"],
            "test_relative_l2": base_test["relative_l2"],
            "test_pde_residual_rms": base_test["pde_residual_rms"],
            "test_periodic_boundary_rms": base_test["periodic_boundary_rms"],
        })
        trajectories.extend({"model": MODEL_BASE, "seed": seed, **item} for item in base_trace)

        pca_val_fn = lambda params: _compiled_validation_call(
            params, x_device, validation_t_device, validation_truth, basis)
        pca_params, pca_train_seconds, pca_loss_value, pca_trace, _, _ = _train_trace(
            pca_update, pca_params, batches, pca_val_fn,
            int(config["eval_every_steps"]), target_values,
            basis_state=basis, preprocessing_seconds=pca_fit_seconds)
        pca_prediction = np.asarray(_predict_pca_grid(
            pca_params, basis, jnp.asarray(t)).block_until_ready())
        pca_validation = _score(pca_prediction, u, x, t, validation_ids, velocity)
        pca_test = _score(pca_prediction, u, x, t, test_ids, velocity)
        rows.append({
            "model": MODEL_PCA, "seed": seed, "k": k,
            "retained_variance": retained_variance, "width": pca_width,
            "parameters": pca_parameters, "pca_fit_seconds": pca_fit_seconds,
            "training_seconds": pca_train_seconds,
            "pipeline_seconds": pca_fit_seconds + pca_train_seconds,
            "final_training_loss": pca_loss_value,
            "validation_rmse": pca_validation["rmse"], "test_rmse": pca_test["rmse"],
            "test_relative_l2": pca_test["relative_l2"],
            "test_pde_residual_rms": pca_test["pde_residual_rms"],
            "test_periodic_boundary_rms": pca_test["periodic_boundary_rms"],
        })
        trajectories.extend({"model": MODEL_PCA, "seed": seed, **item} for item in pca_trace)
        print(f"Seed {seed}: base relL2={base_test['relative_l2']:.4g}, "
              f"PCA relL2={pca_test['relative_l2']:.4g}, k={k}")

    summary = {name: _summarize(rows, name) for name in (MODEL_BASE, MODEL_PCA)}
    paired = {}
    for metric in ("test_rmse", "test_relative_l2", "test_pde_residual_rms",
                   "test_periodic_boundary_rms", "pipeline_seconds"):
        base = {row["seed"]: row for row in rows if row["model"] == MODEL_BASE}
        pca = {row["seed"]: row for row in rows if row["model"] == MODEL_PCA}
        deltas = np.asarray([pca[seed][metric] - base[seed][metric] for seed in sorted(base)])
        paired[metric] = {
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
                 "velocity": velocity, "initial_condition": "sin(pi*x) + 0.5*cos(2*pi*x)"},
        "protocol": {
            "equation": "u_t + c*u_x = 0, periodic x in [-1,1]",
            "exact_solution": "u(x,t)=u0(x-c*t), generated analytically",
            "paired_sampling": "Same train/validation/test time splits, supervised minibatches, residual collocation points, periodic boundary times, and initial-condition x locations per seed.",
            "pca_scope": "PCA is fit on training-time spatial snapshots only; validation and test times are excluded.",
            "training_residual": "JAX autodifferentiation is used for u_t and u_x in both models.",
            "periodicity": "Both models include a periodic boundary loss comparing u(-1,t) and u(1,t).",
            "test_residual": "A common grid finite-difference operator is used to report the advection residual on held-out times.",
            "timing": "Compilation is excluded from timed training; PCA pipeline time includes basis fitting.",
        },
        "parameter_matching": {
            "baseline": baseline_parameters,
            "pca_min": min(row["parameters"] for row in rows if row["k"] is not None),
            "pca_max": max(row["parameters"] for row in rows if row["k"] is not None),
        },
        "compilation_seconds_by_model": compilation_seconds,
        "runtime": runtime, "elapsed_wall_seconds": time.perf_counter()-started,
        "partitions": partitions, "summary": summary,
        "paired_differences_PCA_minus_base": paired,
        "runs": rows, "training_trajectories": trajectories,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", trajectories)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    grouped_summary_svg(output_dir / "comparison.svg", summary, rows,
                        pde_name="advección")
    controlled_trajectory_svg(output_dir / "validation_vs_time.svg", trajectories, target_values)
    return {"output_dir": output_dir, "summary": summary, "rows": rows,
            "trajectories": trajectories, "metadata": metadata}
