"""Paired 1D PINN study: snapshot ROM, label preprocessing, POD floor, and spline."""

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
    _train_trace,
    matched_pca_width,
    pca_solution,
)
from .data import load_burgers_data, split_times
from .metrics import mean_std
from .models import NU, burgers_solution, init_mlp, parameter_count
from .plotting import (
    checkpoint_curve_svg,
    controlled_trajectory_svg,
    grouped_summary_svg,
    metric_tradeoff_svg,
)
from .projection import (
    components_for_variance,
    fit_snapshot_pca,
    project_field,
    reconstruction_scores,
    spline_coefficient_field,
    truncate_basis,
)
from .stats import paired_reports_by_model
from .training import make_update, warmup_update


MODEL_DATA_PCA = "PINN datos PCA"
MODEL_ROM = "ROM PCA"
MODEL_SPLINE = "Spline POD"
MODEL_FLOOR = "Cota POD"


def make_heat_data(nx=256, nt=80, nu=0.05, final_time=1.0):
    x = np.linspace(0.0, np.pi, int(nx), dtype=np.float32)
    t = np.linspace(0.0, float(final_time), int(nt), dtype=np.float32)
    field = (
        np.sin(x)[:, None] * np.exp(-float(nu) * t)[None, :]
        + 0.5 * np.sin(2.0 * x)[:, None] * np.exp(-4.0 * float(nu) * t)[None, :]
        + 0.25 * np.sin(3.0 * x)[:, None] * np.exp(-9.0 * float(nu) * t)[None, :]
    )
    return x, t, field.astype(np.float32)


def make_traveling_wave_data(nx=256, nt=80, wave_speed=1.0, final_time=1.0):
    x = np.linspace(-1.0, 1.0, int(nx), dtype=np.float32)
    t = np.linspace(0.0, float(final_time), int(nt), dtype=np.float32)
    phase = np.pi * (x[:, None] - float(wave_speed) * t[None, :])
    field = np.sin(phase) + 0.5 * np.cos(2.0 * phase) + 0.25 * np.sin(3.0 * phase)
    return x, t, field.astype(np.float32)


def _grid_residual(equation, prediction, x, t, nu, velocity, wave_speed):
    ux = np.gradient(prediction, x, axis=0, edge_order=2)
    uxx = np.gradient(ux, x, axis=0, edge_order=2)
    ut = np.gradient(prediction, t, axis=1, edge_order=2)
    if equation == "burgers":
        return ut + prediction * ux - nu * uxx
    if equation == "heat":
        return ut - nu * uxx
    if equation == "advection":
        return ut + velocity * ux
    if equation == "wave":
        utt = np.gradient(ut, t, axis=1, edge_order=2)
        return utt - (wave_speed ** 2) * uxx
    raise ValueError(f"Ecuación no soportada: {equation}")


def _score_field(prediction, truth, x, t, time_ids, equation, nu, velocity, wave_speed,
                 periodic):
    difference = prediction[:, time_ids] - truth[:, time_ids]
    residual = _grid_residual(equation, prediction, x, t, nu, velocity, wave_speed)
    residual_slice = residual[1:-1, :][:, time_ids]
    if equation == "wave":
        residual_slice = residual[1:-1, 1:-1][:, np.clip(time_ids, 1, t.size - 2) - 1]
    scores = reconstruction_scores(prediction[:, time_ids], truth[:, time_ids])
    scores["pde_residual_rms"] = float(np.sqrt(np.mean(residual_slice ** 2)))
    if periodic:
        gap = prediction[0, time_ids] - prediction[-1, time_ids]
        scores["periodic_boundary_rms"] = float(np.sqrt(np.mean(gap ** 2)))
    else:
        scores["periodic_boundary_rms"] = 0.0
    return scores


def _pde_residual(equation, solution_at, x, t, nu, velocity, wave_speed):
    ux = jax.grad(solution_at, argnums=0)(x, t)
    ut = jax.grad(solution_at, argnums=1)(x, t)
    if equation == "advection":
        return ut + velocity * ux
    uxx = jax.grad(lambda xx: jax.grad(solution_at, argnums=0)(xx, t))(x)
    if equation == "heat":
        return ut - nu * uxx
    if equation == "burgers":
        return ut + solution_at(x, t) * ux - nu * uxx
    if equation == "wave":
        utt = jax.grad(lambda tt: jax.grad(solution_at, argnums=1)(x, tt))(t)
        return utt - (wave_speed ** 2) * uxx
    raise ValueError(f"Ecuación no soportada: {equation}")


def _make_losses(equation, boundary, nu, velocity, wave_speed, include_ut0):
    def residual(solution_at, x, t):
        return _pde_residual(equation, solution_at, x, t, nu, velocity, wave_speed)

    def full_loss(solution_at, batch, x_left, x_right):
        (data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u,
         initial_ut, boundary_t, left_u, right_u) = batch
        data_predictions = jax.vmap(solution_at)(data_x, data_t)
        residuals = jax.vmap(lambda xx, tt: residual(solution_at, xx, tt))(
            residual_x, residual_t)
        initial_predictions = jax.vmap(
            lambda xx: solution_at(xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
        loss = (jnp.mean((data_predictions - data_u) ** 2)
                + jnp.mean(residuals ** 2)
                + jnp.mean((initial_predictions - initial_u) ** 2))
        if include_ut0:
            initial_ut_pred = jax.vmap(
                lambda xx: jax.grad(solution_at, argnums=1)(
                    xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
            loss = loss + jnp.mean((initial_ut_pred - initial_ut) ** 2)
        if boundary == "dirichlet":
            left = jax.vmap(lambda tt: solution_at(x_left, tt))(boundary_t)
            right = jax.vmap(lambda tt: solution_at(x_right, tt))(boundary_t)
            loss = loss + jnp.mean((left - left_u) ** 2) + jnp.mean((right - right_u) ** 2)
        elif boundary == "periodic":
            left = jax.vmap(lambda tt: solution_at(x_left, tt))(boundary_t)
            right = jax.vmap(lambda tt: solution_at(x_right, tt))(boundary_t)
            loss = loss + jnp.mean((left - right) ** 2)
        return loss

    x_left_const = None
    x_right_const = None

    def bind_bounds(x_left, x_right):
        nonlocal x_left_const, x_right_const
        x_left_const = jnp.asarray(x_left)
        x_right_const = jnp.asarray(x_right)

        def baseline_loss(params, batch):
            solution_at = lambda xx, tt: burgers_solution(params, xx, tt)
            return full_loss(solution_at, batch, x_left_const, x_right_const)

        def rom_loss(params, batch, basis_state):
            spatial = basis_state[:3]
            solution_at = lambda xx, tt: pca_solution(params, spatial, xx, tt)
            return full_loss(solution_at, batch, x_left_const, x_right_const)

        return baseline_loss, rom_loss

    return bind_bounds


def _make_batches(x, t, observed, clean, fit_ids, data_ids, ic_ids, steps, batch_size, seed,
                  include_ut0, wave_speed):
    rng = np.random.default_rng(seed)
    ic_x = x[ic_ids]
    ic_u = observed[ic_ids, 0]
    if include_ut0:
        ut = np.gradient(clean, t, axis=1, edge_order=2)
        ic_ut = ut[ic_ids, 0]
    else:
        ic_ut = np.zeros_like(ic_u)
    batches = []
    for _ in range(steps):
        sample_ids = data_ids[rng.integers(0, data_ids.size, batch_size)]
        data_times = rng.choice(fit_ids, batch_size)
        residual_x = rng.uniform(float(x.min()), float(x.max()), batch_size).astype(np.float32)
        residual_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        boundary_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        left_u = np.interp(boundary_t, t, observed[0]).astype(np.float32)
        right_u = np.interp(boundary_t, t, observed[-1]).astype(np.float32)
        batches.append(tuple(jnp.asarray(value) for value in (
            x[sample_ids], t[data_times], observed[sample_ids, data_times],
            residual_x, residual_t, ic_x, ic_u, ic_ut, boundary_t, left_u, right_u,
        )))
    return batches


def _hermite_basis(x, truncated):
    return (jnp.asarray(x), jnp.asarray(truncated["fields"]), jnp.asarray(truncated["slopes"]))


def _ad_residual_rms(equation, predict_fn, x, t, time_ids, nu, velocity, wave_speed, seed, count):
    rng = np.random.default_rng(seed + 17)
    x_pts = rng.choice(x[1:-1], size=count)
    t_pts = rng.choice(t[time_ids], size=count)
    residual = jax.vmap(lambda xx, tt: _pde_residual(
        equation, predict_fn, xx, tt, nu, velocity, wave_speed))
    values = residual(jnp.asarray(x_pts), jnp.asarray(t_pts))
    return float(np.sqrt(np.mean(np.asarray(values.block_until_ready()) ** 2)))


def _summarize(rows, model_name):
    selected = [row for row in rows if row["model"] == model_name]
    metrics = ("validation_rmse", "test_rmse", "test_relative_l2", "test_pde_residual_rms",
               "test_ad_pde_residual_rms", "projection_test_relative_l2",
               "pipeline_seconds", "stopped_step")
    summary = {
        "seed_count": len(selected),
        "parameters_mean": float(np.mean([row["parameters"] for row in selected])),
        "parameters_min": int(min(row["parameters"] for row in selected)),
        "parameters_max": int(max(row["parameters"] for row in selected)),
    }
    for field in ("k", "retained_variance", "pca_fit_seconds"):
        values = [row[field] for row in selected if row.get(field) is not None]
        summary[field] = mean_std(values) if values else None
    for metric in metrics:
        values = [row[metric] for row in selected if row.get(metric) is not None]
        summary[metric] = mean_std(values) if values else None
    target_names = [key for key in selected[0] if key.startswith("time_to_val_rmse_")]
    summary["time_to_target"] = {}
    for target_name in target_names:
        hits = [row[target_name] for row in selected if row[target_name] is not None]
        summary["time_to_target"][target_name.removeprefix("time_to_")] = {
            "hit_count": len(hits), "seed_count": len(selected),
            "seconds_mean_if_hit": float(np.mean(hits)) if hits else None,
            "seconds_std_if_hit": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
        }
    return summary


def _load_field(config, project_root):
    equation = config["equation"]
    if equation == "burgers":
        return load_burgers_data(project_root / config["data_path"])
    if equation == "heat":
        return make_heat_data(
            nx=int(config.get("nx", 256)), nt=int(config.get("nt", 80)),
            nu=float(config.get("nu", 0.05)), final_time=float(config.get("final_time", 1.0)))
    if equation == "wave":
        return make_traveling_wave_data(
            nx=int(config.get("nx", 256)), nt=int(config.get("nt", 80)),
            wave_speed=float(config.get("wave_speed", 1.0)),
            final_time=float(config.get("final_time", 1.0)))
    raise ValueError(f"Ecuación no soportada: {equation}")


def run_snapshot_study(config: dict, project_root: str | Path):
    """Compare base PINN, PCA-preprocessed labels, ROM-PCA, POD floor, and spline POD."""
    project_root = Path(project_root).resolve()
    output_dir = project_root / "results" / config["result_family"] / "base_vs_pca" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    equation = config["equation"]
    x, t, clean = _load_field(config, project_root)
    nu = float(config.get("nu", NU if equation == "burgers" else 0.05))
    velocity = float(config.get("velocity", 1.0))
    wave_speed = float(config.get("wave_speed", 1.0))
    boundary = config.get("boundary", "dirichlet")
    include_ut0 = equation == "wave"
    bind_bounds = _make_losses(equation, boundary, nu, velocity, wave_speed, include_ut0)
    baseline_loss, rom_loss = bind_bounds(float(x[0]), float(x[-1]))
    learning_rate = float(config["learning_rate"])
    base_update = make_update(baseline_loss, learning_rate)
    rom_update = make_update(rom_loss, learning_rate, with_basis=True)
    base_width = int(config["baseline_width"])
    base_layers = [2, base_width, base_width, base_width, 1]
    base_parameters = parameter_count(base_layers)
    target_values = [float(value) for value in config.get("validation_rmse_targets", [])]
    stop_rmse = config.get("early_stop_validation_rmse")
    stop_rmse = None if stop_rmse is None else float(stop_rmse)
    noise_std = float(config.get("noise_std", 0.0))
    sensor_count = config.get("sensor_count")
    seeds = [int(seed) for seed in config["seeds"]]
    rows, trajectories, partitions, compilation = [], [], {}, {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        observed = clean.copy()
        if noise_std > 0.0:
            observed = observed + rng.normal(0.0, noise_std, size=observed.shape).astype(np.float32)
        fit_ids, validation_ids, test_ids = split_times(
            t.size, seed, int(config["test_time_count"]), int(config["validation_time_count"]))
        if sensor_count is None:
            sensor_ids = np.arange(x.size)
        else:
            sensor_ids = np.sort(rng.choice(x.size, size=min(int(sensor_count), x.size), replace=False))
        ic_ids = np.sort(rng.choice(sensor_ids, size=min(int(config["initial_condition_points"]),
                                                         sensor_ids.size), replace=False))
        pca_start = time.perf_counter()
        full_basis = fit_snapshot_pca(x, observed, fit_ids, sensor_ids)
        auto_k = components_for_variance(full_basis["cumulative_variance"],
                                         float(config["pca_variance_threshold"]))
        k = int(config.get("pca_components", auto_k))
        truncated = truncate_basis(full_basis, k)
        pca_fit_seconds = time.perf_counter() - pca_start
        reconstructed, coeffs_observed = project_field(
            truncated["mean"], truncated["modes"], observed, fit_ids)
        labels = observed.copy()
        labels[:, fit_ids] = reconstructed
        floor_recon_val, _ = project_field(truncated["mean"], truncated["modes"], clean, validation_ids)
        floor_recon_test, _ = project_field(truncated["mean"], truncated["modes"], clean, test_ids)
        floor_val = reconstruction_scores(floor_recon_val, clean[:, validation_ids])
        floor_test = reconstruction_scores(floor_recon_test, clean[:, test_ids])
        spline_field = spline_coefficient_field(
            truncated["mean"], truncated["modes"], t[fit_ids], coeffs_observed, t)
        spline_val = _score_field(spline_field, clean, x, t, validation_ids, equation,
                                  nu, velocity, wave_speed, boundary == "periodic")
        spline_test = _score_field(spline_field, clean, x, t, test_ids, equation,
                                   nu, velocity, wave_speed, boundary == "periodic")

        batches_base = _make_batches(
            x, t, observed, clean, fit_ids, sensor_ids, ic_ids, int(config["steps"]),
            int(config["batch_size"]), seed, include_ut0, wave_speed)
        batches_data = _make_batches(
            x, t, labels, clean, fit_ids, sensor_ids, ic_ids, int(config["steps"]),
            int(config["batch_size"]), seed, include_ut0, wave_speed)
        hermite = _hermite_basis(x, truncated)
        rom_width, rom_parameters = matched_pca_width(k, base_parameters)
        if abs(rom_parameters - base_parameters) / base_parameters > 0.05:
            raise ValueError(f"No se emparejaron parámetros para k={k}.")
        x_device = jnp.asarray(x)
        val_t = jnp.asarray(t[validation_ids])
        val_truth = jnp.asarray(clean[:, validation_ids])
        base_params = init_mlp(jax.random.PRNGKey(seed), base_layers)
        data_params = init_mlp(jax.random.PRNGKey(seed + 91), base_layers)
        rom_params = init_mlp(jax.random.PRNGKey(seed), [1, rom_width, rom_width, rom_width, k])
        if "baseline" not in compilation:
            compile_start = time.perf_counter()
            warmup_update(base_update, base_params, batches_base[0])
            _compiled_validation_call(base_params, x_device, val_t, val_truth)
            compilation["baseline"] = time.perf_counter() - compile_start
        if "data_pca" not in compilation:
            compile_start = time.perf_counter()
            warmup_update(base_update, data_params, batches_data[0])
            compilation["data_pca"] = time.perf_counter() - compile_start
        rom_key = f"rom_k{k}_w{rom_width}"
        if rom_key not in compilation:
            compile_start = time.perf_counter()
            warmup_update(rom_update, rom_params, batches_base[0], hermite)
            _compiled_validation_call(rom_params, x_device, val_t, val_truth, hermite)
            compilation[rom_key] = time.perf_counter() - compile_start

        partitions[str(seed)] = {
            "fit_time_indices": fit_ids.tolist(),
            "validation_time_indices": validation_ids.tolist(),
            "test_time_indices": test_ids.tolist(),
            "sensor_x_indices": sensor_ids.tolist(),
            "initial_condition_x_indices": ic_ids.tolist(),
            "k": k, "retained_variance": truncated["retained_variance"],
        }

        def run_trained(name, params, batches, update, basis, preprocess):
            validation_fn = lambda current: _compiled_validation_call(
                current, x_device, val_t, val_truth, basis)
            trained, train_seconds, final_loss, trace, targets, stopped = _train_trace(
                update, params, batches, validation_fn, int(config["eval_every_steps"]),
                target_values, basis_state=basis, preprocessing_seconds=preprocess,
                stop_rmse=stop_rmse)
            if basis is None:
                prediction = np.asarray(_predict_baseline(
                    trained, x_device, jnp.asarray(t)).block_until_ready())
                predict_fn = lambda xx, tt: burgers_solution(trained, xx, tt)
            else:
                prediction = np.asarray(_predict_pca_grid(
                    trained, basis, jnp.asarray(t)).block_until_ready())
                predict_fn = lambda xx, tt: pca_solution(trained, basis, xx, tt)
            validation = _score_field(prediction, clean, x, t, validation_ids, equation,
                                      nu, velocity, wave_speed, boundary == "periodic")
            test = _score_field(prediction, clean, x, t, test_ids, equation,
                                nu, velocity, wave_speed, boundary == "periodic")
            ad_rms = _ad_residual_rms(
                equation, predict_fn, x, t, test_ids, nu, velocity, wave_speed,
                seed, int(config.get("test_ad_residual_points", 512)))
            row = {
                "model": name, "seed": seed, "k": None if name == MODEL_BASE else k,
                "retained_variance": None if name == MODEL_BASE else truncated["retained_variance"],
                "width": base_width if basis is None else rom_width,
                "parameters": base_parameters if basis is None else rom_parameters,
                "pca_fit_seconds": 0.0 if name == MODEL_BASE else pca_fit_seconds,
                "training_seconds": train_seconds,
                "pipeline_seconds": train_seconds if name == MODEL_BASE else pca_fit_seconds + train_seconds,
                "final_training_loss": final_loss,
                "stopped_step": stopped,
                "validation_rmse": validation["rmse"],
                "test_rmse": test["rmse"],
                "test_relative_l2": test["relative_l2"],
                "test_pde_residual_rms": test["pde_residual_rms"],
                "test_ad_pde_residual_rms": ad_rms,
                "test_periodic_boundary_rms": test["periodic_boundary_rms"],
                "projection_test_relative_l2": floor_test["relative_l2"],
            }
            for target, seconds in targets.items():
                row[f"time_to_val_rmse_{target.replace('.', '_')}"] = seconds
            rows.append(row)
            trajectories.extend({"model": name, "seed": seed, **item} for item in trace)
            print(f"  {name}: relL2={test['relative_l2']:.5g}, pipeline={row['pipeline_seconds']:.2f}s, stop={stopped}")

        print(f"Seed {seed}: k={k}, var={truncated['retained_variance']:.5%}, sensors={sensor_ids.size}")
        run_trained(MODEL_BASE, base_params, batches_base, base_update, None, 0.0)
        run_trained(MODEL_DATA_PCA, data_params, batches_data, base_update, None, pca_fit_seconds)
        run_trained(MODEL_ROM, rom_params, batches_base, rom_update, hermite, pca_fit_seconds)

        spline_row = {
            "model": MODEL_SPLINE, "seed": seed, "k": k,
            "retained_variance": truncated["retained_variance"],
            "width": 0, "parameters": 0, "pca_fit_seconds": pca_fit_seconds,
            "training_seconds": 0.0, "pipeline_seconds": pca_fit_seconds,
            "final_training_loss": None, "stopped_step": 0,
            "validation_rmse": spline_val["rmse"],
            "test_rmse": spline_test["rmse"],
            "test_relative_l2": spline_test["relative_l2"],
            "test_pde_residual_rms": spline_test["pde_residual_rms"],
            "test_ad_pde_residual_rms": spline_test["pde_residual_rms"],
            "test_periodic_boundary_rms": spline_test["periodic_boundary_rms"],
            "projection_test_relative_l2": floor_test["relative_l2"],
        }
        for target in target_values:
            key = f"time_to_val_rmse_{str(target).replace('.', '_')}"
            spline_row[key] = pca_fit_seconds if spline_val["rmse"] <= target else None
        rows.append(spline_row)
        rows.append({
            "model": MODEL_FLOOR, "seed": seed, "k": k,
            "retained_variance": truncated["retained_variance"],
            "width": 0, "parameters": 0, "pca_fit_seconds": pca_fit_seconds,
            "training_seconds": 0.0, "pipeline_seconds": pca_fit_seconds,
            "final_training_loss": None, "stopped_step": 0,
            "validation_rmse": floor_val["rmse"],
            "test_rmse": floor_test["rmse"],
            "test_relative_l2": floor_test["relative_l2"],
            "test_pde_residual_rms": None,
            "test_ad_pde_residual_rms": None,
            "test_periodic_boundary_rms": None,
            "projection_test_relative_l2": floor_test["relative_l2"],
            **{f"time_to_val_rmse_{str(target).replace('.', '_')}": pca_fit_seconds
               if floor_val["rmse"] <= target else None for target in target_values},
        })

    model_names = list(dict.fromkeys(row["model"] for row in rows))
    summary = {name: _summarize(rows, name) for name in model_names}
    trained_rows = [row for row in rows if row["model"] in {MODEL_BASE, MODEL_DATA_PCA, MODEL_ROM, MODEL_SPLINE}]
    paired = paired_reports_by_model(
        trained_rows, MODEL_BASE,
        ("test_rmse", "test_relative_l2", "test_pde_residual_rms", "pipeline_seconds"),
    )
    runtime = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "device": str(jax.devices()[0]),
    }
    metadata = {
        "configuration": config,
        "problem": {
            "equation": equation,
            "boundary": boundary,
            "two_pca_uses": "PINN datos PCA reconstructs training snapshots and trains the same (x,t) MLP; ROM PCA predicts POD coefficients.",
            "projection_floor": "Orthogonal reconstruction of clean held-out snapshots in the training-only truncated basis.",
            "spline_pod": "Cubic spline of training-time POD coefficients, without a neural network.",
            "dirichlet": "Both neural models penalize u at the spatial endpoints against observed boundary traces.",
            "early_stop": "Training stops at the first validation RMSE at or below early_stop_validation_rmse, if set.",
            "noise_and_sensors": "Gaussian noise and sensor subsampling, when configured, affect PCA and supervised labels only; metrics use the clean field.",
            "residual_metrics": "test_pde_residual_rms is a grid finite-difference residual; test_ad_pde_residual_rms is autodifferentiated at held-out collocation points for the neural models.",
        },
        "runtime": runtime,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "compilation_seconds_by_architecture": compilation,
        "partitions": partitions,
        "summary": summary,
        "paired_differences_vs_base": paired,
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
    plot_summary = {name: stats for name, stats in summary.items()
                    if name != MODEL_FLOOR and stats["test_pde_residual_rms"] is not None}
    grouped_summary_svg(output_dir / "comparison.svg", plot_summary, trained_rows, pde_name=equation)
    metric_tradeoff_svg(
        output_dir / "error_residual_tradeoff.svg", plot_summary,
        "test_relative_l2", "test_pde_residual_rms",
        "Error relativo L2 de prueba", "Residuo FD de prueba",
        "Error de solución frente a residuo")
    if trajectories:
        controlled_trajectory_svg(
            output_dir / "validation_vs_time.svg", trajectories, target_values)
        checkpoint_curve_svg(
            output_dir / "validation_rmse_vs_steps.svg", trajectories,
            "validation_rmse", "RMSE de validación", "RMSE")
    return {"output_dir": output_dir, "summary": summary, "rows": rows,
            "trajectories": trajectories, "metadata": metadata}
