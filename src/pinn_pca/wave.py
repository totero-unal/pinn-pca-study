"""Paired standing-wave benchmark with training-only spatial snapshot PCA."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.interpolate import CubicSpline

from .controlled import _train_trace, matched_pca_width
from .data import split_times
from .metrics import mean_std
from .models import burgers_solution, init_mlp, mlp, parameter_count
from .stats import paired_difference_report
from .training import make_update, warmup_update

BASE = "PINN base"
PCA = "PCA espacial 99%"
TERM_NAMES = ["data", "pde", "initial_u", "initial_ut", "boundary"]


def exact_wave(x, t, harmonics, amplitudes, speed=2.0):
    """Standing sine modes on [0,1], with zero initial velocity and boundary values."""
    return sum(a * jnp.sin(n * jnp.pi * x) * jnp.cos(speed * n * jnp.pi * t)
               for n, a in zip(harmonics, amplitudes))


def make_wave_data(nx, nt, final_time, harmonics, amplitudes, speed=2.0):
    if len(harmonics) != len(amplitudes) or not harmonics:
        raise ValueError("Se requieren armónicos y amplitudes de igual longitud.")
    if any(int(n) != n or n < 1 for n in harmonics):
        raise ValueError("Los armónicos deben ser enteros positivos.")
    x = np.linspace(0, 1, nx, dtype=np.float64)
    t = np.linspace(0, final_time, nt, dtype=np.float64)
    u = sum(a * np.sin(n * np.pi * x[:, None]) * np.cos(speed * n * np.pi * t[None, :])
            for n, a in zip(harmonics, amplitudes))
    return x.astype(np.float32), t.astype(np.float32), u.astype(np.float32)


def wave_residual(solution, x, t, speed=2.0):
    uxx = jax.grad(lambda xx: jax.grad(solution, argnums=0)(xx, t))(x)
    utt = jax.grad(lambda tt: jax.grad(solution, argnums=1)(x, tt))(t)
    return utt - speed ** 2 * uxx


def fit_wave_pca(x, u, fit_ids, threshold=0.99):
    """Fit centered PCA only on fit snapshots; C2 interpolation for second derivatives."""
    snapshots = np.asarray(u[:, fit_ids].T, dtype=np.float64)
    mean = snapshots.mean(axis=0)
    _, values, vectors = np.linalg.svd(snapshots - mean, full_matrices=False)
    variance = values ** 2
    if not 0 < threshold <= 1 or variance.sum() <= 0:
        raise ValueError("PCA requiere varianza positiva y umbral en (0,1].")
    cumulative = np.cumsum(variance) / variance.sum()
    k = min(int(np.searchsorted(cumulative, threshold) + 1), len(values))
    modes = vectors[:k].T
    fields = np.column_stack((mean, modes))
    # SciPy supplies the fixed polynomial coefficients, evaluated and differentiated by JAX.
    coefficients = CubicSpline(np.asarray(x, dtype=float), fields, axis=0).c
    basis = (jnp.asarray(x), jnp.asarray(coefficients, dtype=jnp.float32))
    jax.tree.map(lambda v: v.block_until_ready(), basis)
    return basis, {"k": k, "retained_variance": float(cumulative[k - 1]),
                   "cumulative_variance": cumulative.tolist(), "mean": mean, "modes": modes}


def spline_fields(basis, x):
    """Cubic Horner evaluation. Clip the interval index, not x or its derivatives."""
    knots, coeff = basis
    index = jnp.clip(jnp.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    dx = x - knots[index]
    return ((coeff[0, index] * dx + coeff[1, index]) * dx + coeff[2, index]) * dx + coeff[3, index]


def pca_wave_solution(params, basis, x, t):
    fields = spline_fields(basis, x)
    return fields[0] + jnp.dot(fields[1:], mlp(params, jnp.reshape(t, (1,))))


@jax.jit
def predict_base(params, x, t):
    return jax.vmap(lambda xx: jax.vmap(lambda tt: burgers_solution(params, xx, tt))(t))(x)


@jax.jit
def predict_pca(params, basis, x, t):
    fields = jax.vmap(lambda xx: spline_fields(basis, xx))(x)
    coefficients = jax.vmap(lambda tt: mlp(params, jnp.reshape(tt, (1,))))(t)
    return fields[:, :1] + fields[:, 1:] @ coefficients.T


@jax.jit
def validation_base(params, x, t, truth):
    return jnp.sqrt(jnp.mean((predict_base(params, x, t) - truth) ** 2))


@jax.jit
def validation_pca(params, basis, x, t, truth):
    return jnp.sqrt(jnp.mean((predict_pca(params, basis, x, t) - truth) ** 2))


def loss_terms(solution, batch, speed):
    dx, dt, du, rx, rt, ix, iu, bt = batch
    data = jax.vmap(solution)(dx, dt)
    residual = jax.vmap(lambda xx, tt: wave_residual(solution, xx, tt, speed))(rx, rt)
    initial = jax.vmap(lambda xx: solution(xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    initial_ut = jax.vmap(lambda xx: jax.grad(solution, argnums=1)(
        xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    left = jax.vmap(lambda tt: solution(jnp.asarray(0., dtype=tt.dtype), tt))(bt)
    right = jax.vmap(lambda tt: solution(jnp.asarray(1., dtype=tt.dtype), tt))(bt)
    return jnp.stack((jnp.mean((data - du) ** 2), jnp.mean(residual ** 2),
                      jnp.mean((initial - iu) ** 2), jnp.mean(initial_ut ** 2),
                      .5 * (jnp.mean(left ** 2) + jnp.mean(right ** 2))))


def make_losses(speed, weights):
    weights = jnp.asarray(weights)
    def baseline(params, batch):
        return jnp.dot(weights, loss_terms(lambda xx, tt: burgers_solution(params, xx, tt), batch, speed))
    def projected(params, batch, basis):
        return jnp.dot(weights, loss_terms(lambda xx, tt: pca_wave_solution(params, basis, xx, tt), batch, speed))
    return baseline, projected


def make_wave_batches(x, t, u, fit_ids, initial_ids, steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    batches, digest = [], hashlib.sha256()
    for _ in range(steps):
        ids = rng.integers(x.size, size=batch_size)
        tids = rng.choice(fit_ids, size=batch_size)
        rx = rng.uniform(0, 1, batch_size).astype(np.float32)
        rt = rng.uniform(0, float(t[-1]), batch_size).astype(np.float32)
        bt = rng.uniform(0, float(t[-1]), batch_size).astype(np.float32)
        values = (x[ids], t[tids], u[ids, tids], rx, rt, x[initial_ids], u[initial_ids, 0], bt)
        for value in values:
            digest.update(np.ascontiguousarray(value).tobytes())
        batches.append(tuple(jnp.asarray(v) for v in values))
    return batches, digest.hexdigest()


@partial(jax.jit, static_argnames=("projected",))
def diagnostics(params, basis, rx, rt, ix, iu, bt, speed, projected=False):
    solution = (lambda xx, tt: pca_wave_solution(params, basis, xx, tt)) if projected else (
        lambda xx, tt: burgers_solution(params, xx, tt))
    residual = jax.vmap(lambda xx, tt: wave_residual(solution, xx, tt, speed))(rx, rt)
    initial = jax.vmap(lambda xx: solution(xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    initial_ut = jax.vmap(lambda xx: jax.grad(solution, argnums=1)(xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    left = jax.vmap(lambda tt: solution(jnp.asarray(0., dtype=tt.dtype), tt))(bt)
    right = jax.vmap(lambda tt: solution(jnp.asarray(1., dtype=tt.dtype), tt))(bt)
    return {"test_pde_residual_rms": jnp.sqrt(jnp.mean(residual ** 2)),
            "initial_u_rmse": jnp.sqrt(jnp.mean((initial - iu) ** 2)),
            "initial_ut_rms": jnp.sqrt(jnp.mean(initial_ut ** 2)),
            "test_boundary_rms": jnp.sqrt(.5 * (jnp.mean(left ** 2) + jnp.mean(right ** 2)))}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def run_wave_benchmark(config, project_root):
    root = Path(project_root).resolve()
    output = root / "results/wave/base_vs_pca" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    speed, width = float(config["wave_speed"]), int(config["baseline_width"])
    base_layers = [2, width, width, width, 1]
    base_count = parameter_count(base_layers)
    losses = make_losses(speed, config["loss_weights"])
    updates = {BASE: make_update(losses[0], float(config["learning_rate"])),
               PCA: make_update(losses[1], float(config["learning_rate"]), with_basis=True)}
    rows, traces, mode_rows, partitions, compilation = [], [], [], {}, {}
    for case in config["cases"]:
        name = case["name"]
        x, t, u = make_wave_data(config["nx"], config["nt"], config["final_time"],
                                  case["harmonics"], case["amplitudes"], speed)
        data_path = root / "data/generated" / f"wave_{name}.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, x=x, t=t, u=u, speed=speed,
                            harmonics=case["harmonics"], amplitudes=case["amplitudes"])
        partitions[name] = {}
        for seed_index, seed in enumerate(config["seeds"]):
            fit, val, test = split_times(t.size, seed, config["test_time_count"], config["validation_time_count"])
            rng = np.random.default_rng(seed + 17)
            initial_ids = np.sort(rng.choice(x.size, config["initial_condition_points"], replace=False))
            pca_start = time.perf_counter()
            basis, info = fit_wave_pca(x, u, fit, config["pca_variance_threshold"])
            pca_seconds = time.perf_counter() - pca_start
            k = info["k"]
            pw, pc = matched_pca_width(k, base_count)
            if abs(pc - base_count) / base_count > .05:
                raise ValueError("La diferencia de parámetros supera el 5%.")
            batches, digest = make_wave_batches(x, t, u, fit, initial_ids, config["steps"],
                                                 config["batch_size"], seed)
            params = {BASE: init_mlp(jax.random.PRNGKey(seed), base_layers),
                      PCA: init_mlp(jax.random.PRNGKey(seed), [1, pw, pw, pw, k])}
            xv, tv, uv = jnp.asarray(x), jnp.asarray(t[val]), jnp.asarray(u[:, val])
            for model in (BASE, PCA):
                key = "base" if model == BASE else f"pca_k{k}_w{pw}"
                if key not in compilation:
                    start = time.perf_counter()
                    warmup_update(updates[model], params[model], batches[0], basis if model == PCA else None)
                    value = (validation_base(params[model], xv, tv, uv) if model == BASE else
                             validation_pca(params[model], basis, xv, tv, uv))
                    value.block_until_ready()
                    compilation[key] = time.perf_counter() - start
            order = [BASE, PCA] if seed_index % 2 == 0 else [PCA, BASE]
            partitions[name][str(seed)] = {"fit_time_indices": fit.tolist(),
                "validation_time_indices": val.tolist(), "test_time_indices": test.tolist(),
                "initial_x_indices": initial_ids.tolist(), "shared_batch_sha256": digest,
                "model_execution_order": order, "k": k, "retained_variance": info["retained_variance"],
                "cumulative_variance": info["cumulative_variance"]}
            centered = np.asarray(u, dtype=float) - info["mean"][:, None]
            oracle = info["mean"][:, None] + info["modes"] @ (info["modes"].T @ centered)
            floor = float(np.linalg.norm(oracle[:, test] - u[:, test]) / np.linalg.norm(u[:, test]))
            rrng = np.random.default_rng(seed + 1901)
            rx = jnp.asarray(rrng.uniform(0, 1, config["test_residual_points"]), dtype=jnp.float32)
            rt = jnp.asarray(rrng.choice(t[test], config["test_residual_points"]))
            predictions = {}
            print(f"{name}, seed={seed}, k={k}, var={info['retained_variance']:.6f}", flush=True)
            for model in order:
                is_pca = model == PCA
                validation = (lambda p: float(validation_pca(p, basis, xv, tv, uv).block_until_ready())) if is_pca else (
                    lambda p: float(validation_base(p, xv, tv, uv).block_until_ready()))
                fitted, train_seconds, final_loss, trace, targets, _ = _train_trace(
                    updates[model], params[model], batches, validation, config["eval_every_steps"],
                    config["validation_rmse_targets"], basis_state=basis if is_pca else None,
                    preprocessing_seconds=pca_seconds if is_pca else 0)
                prediction = (predict_pca(fitted, basis, xv, jnp.asarray(t)) if is_pca else
                              predict_base(fitted, xv, jnp.asarray(t)))
                prediction = np.asarray(prediction.block_until_ready())
                predictions[model] = prediction
                difference = prediction[:, test] - u[:, test]
                diag = diagnostics(fitted, basis if is_pca else None, rx, rt, xv,
                                   jnp.asarray(u[:, 0]), jnp.asarray(t[test]), speed, projected=is_pca)
                diag = {key: float(value.block_until_ready()) for key, value in diag.items()}
                solution = (lambda xx, tt: pca_wave_solution(fitted, basis, xx, tt)) if is_pca else (
                    lambda xx, tt: burgers_solution(fitted, xx, tt))
                terms = np.asarray(loss_terms(solution, batches[-1], speed))
                row = {"case": name, "model": model, "seed": seed, "k": k if is_pca else None,
                       "retained_variance": info["retained_variance"] if is_pca else None,
                       "parameters": pc if is_pca else base_count, "width": pw if is_pca else width,
                       "pca_fit_seconds": pca_seconds if is_pca else 0,
                       "training_seconds": train_seconds,
                       "pipeline_seconds": train_seconds + (pca_seconds if is_pca else 0),
                       "validation_rmse": trace[-1]["validation_rmse"], "final_training_loss": final_loss,
                       "test_rmse": float(np.sqrt(np.mean(difference ** 2))),
                       "test_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(u[:, test])),
                       "pca_oracle_test_relative_l2": floor if is_pca else None,
                       **diag, **{f"loss_{n}": float(v) for n, v in zip(TERM_NAMES, terms)}}
                row.update({f"time_to_val_rmse_{target.replace('.', '_')}": value for target, value in targets.items()})
                rows.append(row)
                traces.extend({"case": name, "model": model, "seed": seed, **point} for point in trace)
                for harmonic, amplitude in zip(case["harmonics"], case["amplitudes"]):
                    wave = np.sin(harmonic * np.pi * np.asarray(x, dtype=float))
                    coefficient = 2 * np.trapezoid(prediction[:, test] * wave[:, None], x, axis=0)
                    true_coefficient = amplitude * np.cos(speed * harmonic * np.pi * t[test])
                    mode_rows.append({"case": name, "model": model, "seed": seed, "harmonic": harmonic,
                        "amplitude": amplitude, "coefficient_relative_l2": float(np.linalg.norm(
                            coefficient - true_coefficient) / np.linalg.norm(true_coefficient))})
                print(f"  {model}: L2={row['test_relative_l2']:.5f}, PDE={diag['test_pde_residual_rms']:.5f}, "
                      f"pipeline={row['pipeline_seconds']:.2f}s", flush=True)
            if seed_index == 0:
                np.savez_compressed(output / f"example_{name}.npz", x=x, t=t, truth=u,
                                    baseline=predictions[BASE], pca=predictions[PCA], oracle=oracle,
                                    fit_ids=fit, validation_ids=val, test_ids=test, seed=seed)
            for filename, records in (("metrics.csv", rows), ("validation_trajectories.csv", traces),
                                      ("mode_metrics.csv", mode_rows)):
                _write_csv(output / filename, records)
            (output / "progress.json").write_text(json.dumps({"completed_fits": len(rows),
                "planned_fits": 2 * len(config["cases"]) * len(config["seeds"])}), encoding="utf-8")

    summaries, paired = {}, {}
    metrics = ("test_relative_l2", "test_pde_residual_rms", "pipeline_seconds", "validation_rmse",
               "test_boundary_rms", "initial_u_rmse", "initial_ut_rms")
    for case in config["cases"]:
        name = case["name"]
        summaries[name] = {}
        for model in (BASE, PCA):
            selected = [r for r in rows if r["case"] == name and r["model"] == model]
            summaries[name][model] = {key: mean_std([r[key] for r in selected]) for key in metrics}
            summaries[name][model]["k_values"] = sorted({r["k"] for r in selected if r["k"] is not None})
        lookup = {(r["model"], r["seed"]): r for r in rows if r["case"] == name}
        paired[name] = {key: paired_difference_report(
            [lookup[(BASE, s)][key] for s in config["seeds"]],
            [lookup[(PCA, s)][key] for s in config["seeds"]]) for key in metrics[:3]}
    metadata = {"configuration": config, "summary": summaries, "paired_PCA_minus_base": paired,
        "partitions": partitions, "compilation_seconds_by_architecture": compilation,
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "jaxlib": __import__('jaxlib').__version__, "numpy": np.__version__,
                    "scipy": scipy.__version__, "device": str(jax.devices()[0])},
        "elapsed_wall_seconds": time.perf_counter() - started,
        "timing_scope": "Pipeline = PCA fit + spline preparation + training + validation. Compilation, batch generation and final scoring excluded.",
        "pca_scope": "Training-time spatial snapshots only. The neural network predicts temporal coefficients; C2 splines are evaluated and differentiated by JAX.",
        "fairness": "Identical batches, splits, IC locations and PDE test points per pair; matched parameters within 5%; same loss weights; alternating model order. Different architectures use the same seed, not identical weights.",
        "oracle_scope": "Projection uses true test coefficients only as a representation diagnostic, never for training or k selection.",
        "runs": rows}
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    from .wave_report import create_wave_report
    create_wave_report(root, output, metadata, traces, mode_rows)
    return {"output_dir": output, "metadata": metadata}
