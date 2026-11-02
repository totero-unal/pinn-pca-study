"""Burgers reference for Algorithm 1 of arXiv:2501.07700v4.

Uncentered signed residual snapshots, nuclear-sum rank, column-pivoted QR,
convergence pruning and fresh disjoint candidates. No hybrid point filler.
"""
from __future__ import annotations

import csv
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import svd, qr
from scipy.stats import qmc

from .data import load_burgers_data
from .models import NU, init_mlp, mlp
from .training import adam_init, adam_step


def solution(params, x, t):
    # Endpoint correction eliminates float32 sin(pi) roundoff in hard BCs.
    initial = jnp.where(jnp.abs(x) == 1, 0., -jnp.sin(jnp.pi * x))
    return t * (x + 1) * (x - 1) * mlp(params, jnp.stack((x, t)))[0] + initial


def residual(params, x, t):
    f = lambda xx, tt: solution(params, xx, tt)
    ux = jax.grad(f, 0)(x, t)
    ut = jax.grad(f, 1)(x, t)
    uxx = jax.grad(lambda xx: jax.grad(f, 0)(xx, t))(x)
    return ut + f(x, t) * ux - NU * uxx


residuals = jax.jit(jax.vmap(residual, in_axes=(None, 0, 0)))
predictions = jax.jit(jax.vmap(solution, in_axes=(None, 0, 0)))


def learning_rate(step, total, initial):
    return initial * .5 * (1 + jnp.cos(jnp.pi * step / total))


@partial(jax.jit, static_argnames=("period", "total", "observe"))
def train_period(state, points, candidates, *, period, total, initial_lr, observe):
    """One snapshot AFTER each Adam update; optimizer state spans periods."""
    def body(state, _):
        params, m, v, step = state
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean(residuals(p, points[:, 0], points[:, 1]) ** 2)
        )(params)
        new = adam_step(params, grads, m, v, step, learning_rate(step, total, initial_lr))
        snapshot = (residuals(new[0], candidates[:, 0], candidates[:, 1])
                    if observe else jnp.zeros((0,), dtype=jnp.float32))
        return new, (loss, snapshot)
    return jax.lax.scan(body, state, None, length=period)


def select_qr(matrix, epsilon):
    """Eq. 15 uses sum(sigma), NOT explained variance sum(sigma**2)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Snapshot matrix must be nonempty, finite and two-dimensional")
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must lie in [0, 1)")
    modes, singular, _ = svd(matrix, full_matrices=False, check_finite=False)
    if singular.sum() == 0:
        return np.array([], dtype=int), singular, modes[:, :0]
    cumulative = np.cumsum(singular) / singular.sum()
    k = min(len(singular), int(np.searchsorted(cumulative, 1 - epsilon)) + 1)
    basis = modes[:, :k]
    _, _, pivots = qr(basis.T, mode="economic", pivoting=True)
    return pivots[:k], singular, basis


def prune(points, candidates, selected, old_magnitudes, new_magnitudes, candidate_magnitudes,
          floor=1e-12):
    """Return points and correctly aligned next baseline, including inserted points."""
    if floor <= 0 or len(selected) > len(points):
        raise ValueError("Invalid residual floor or replacement count")
    arrays = [old_magnitudes, new_magnitudes, candidate_magnitudes]
    if any(not np.isfinite(a).all() for a in arrays):
        raise FloatingPointError("Nonfinite pruning residual")
    if set(map(tuple, points)) & set(map(tuple, candidates)):
        raise ValueError("Snapshot and training sets must be disjoint")
    d = np.log2(np.maximum(old_magnitudes, floor) / np.maximum(new_magnitudes, floor))
    removed = np.argsort(-d, kind="stable")[:len(selected)]
    accepted = bool(len(selected) and np.all(d[removed] >= 0))
    if not accepted:
        return points.copy(), np.array(new_magnitudes), np.array([], dtype=int), d, False
    keep = np.ones(len(points), dtype=bool)
    keep[removed] = False
    updated = np.concatenate((points[keep], candidates[selected]))
    baseline = np.concatenate((np.asarray(new_magnitudes)[keep],
                               np.asarray(candidate_magnitudes)[selected]))
    if len(set(map(tuple, updated))) != len(points):
        raise ValueError("Replacement introduced duplicates")
    return updated, baseline, removed, d, True


def uniform_points(rng, count, exclude=()):
    forbidden = set(map(tuple, exclude))
    chosen = []
    while len(chosen) < count:
        batch = rng.uniform([-1., 0.], [1., 1.], (count - len(chosen), 2)).astype(np.float32)
        for point in batch:
            key = tuple(point)
            if -1 < point[0] < 1 and 0 < point[1] < 1 and key not in forbidden:
                forbidden.add(key)
                chosen.append(point)
    return np.asarray(chosen)


def sobol_points(count, seed):
    unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    return (unit * [2, 1] - [1, 0]).astype(np.float32)


def evaluate_residual(params, points):
    # Bounded batches avoid very large derivative intermediates in full preset.
    return np.concatenate([np.asarray(residuals(params, p[:, 0], p[:, 1]))
                           for p in np.array_split(points, max(1, int(np.ceil(len(points) / 256))))])


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(c):
    for key in ("steps", "period", "active_points", "snapshot_points", "validation_points", "test_points",
                "hidden_layers", "width"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["steps"] % c["period"] or min(c["snapshot_points"], c["period"]) > c["active_points"]:
        raise ValueError("Periods must divide steps and maximum rank must fit the active set")
    if not 0 <= c["epsilon"] < 1 or c["learning_rate"] <= 0:
        raise ValueError("Invalid epsilon or learning rate")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Provide distinct seeds")
    if c["steps"] not in c["budgets"] or any(b % c["period"] or b <= 0 or b > c["steps"] for b in c["budgets"]):
        raise ValueError("Budgets must be period boundaries and include the final step")


def run(config, root):
    validate_config(config)
    root = Path(root)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory already belongs to a different configuration")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    x, t, reference = load_burgers_data(root / "data/raw/burgers_shock.mat")
    xx, tt = np.meshgrid(x, t, indexing="ij")
    grid = np.column_stack((xx.ravel(), tt.ravel()))
    all_metrics = []
    for seed in config["seeds"]:
        shared_rng = np.random.default_rng(seed)
        initial_points = uniform_points(shared_rng, config["active_points"])
        validation = uniform_points(shared_rng, config["validation_points"], initial_points)
        test = uniform_points(shared_rng, config["test_points"], np.concatenate((initial_points, validation)))
        for method in ("qr_deim", "sobol_resample"):
            folder = output / f"seed_{seed}" / method
            folder.mkdir(parents=True, exist_ok=True)
            # Completed runs can be reused without interrupting long local benchmarks.
            if (folder / "complete.json").exists():
                all_metrics.extend(json.loads((folder / "metrics.json").read_text()))
                print(f"reuse seed={seed} method={method}", flush=True)
                continue
            params = init_mlp(jax.random.PRNGKey(seed), [2] + [config["width"]] * config["hidden_layers"] + [1])
            state = (params, *adam_init(params))
            points = initial_points.copy()
            rng = np.random.default_rng(seed + 100000)
            candidates = uniform_points(rng, config["snapshot_points"], points)
            old = np.abs(evaluate_residual(params, points))
            best_value, best_step, best_params = np.inf, 0, None
            trace, updates, metrics, point_history = [], [], [], []
            start = time.perf_counter()
            print(f"start seed={seed} method={method}", flush=True)
            for step in range(config["period"], config["steps"] + 1, config["period"]):
                point_history.append(points.copy())
                state, (losses, snapshots) = train_period(
                    state, points, candidates, period=config["period"], total=config["steps"],
                    initial_lr=config["learning_rate"], observe=method == "qr_deim")
                snapshots = np.asarray(snapshots).T
                params = state[0]
                val_mse = float(np.mean(evaluate_residual(params, validation).astype(float) ** 2))
                if not np.isfinite(val_mse) or not np.isfinite(np.asarray(losses)).all():
                    raise FloatingPointError(f"Diverged seed={seed} method={method} step={step}")
                if val_mse < best_value:
                    best_value, best_step, best_params = val_mse, step, params
                trace.append(dict(seed=seed, method=method, step=step, validation_mse=val_mse,
                                  best_validation_mse=best_value, best_step=best_step,
                                  train_loss_before_last_update=float(losses[-1]),
                                  elapsed_seconds=time.perf_counter() - start))
                if step in config["budgets"]:
                    for checkpoint, evaluated in (("last", params), ("best_validation", best_params)):
                        pred = np.asarray(predictions(evaluated, grid[:, 0], grid[:, 1])).reshape(reference.shape)
                        error = pred.astype(float) - reference
                        independent = evaluate_residual(evaluated, test)
                        ic = np.asarray(predictions(evaluated, x, np.zeros_like(x)))
                        exact_ic = np.where(np.abs(x) == 1, 0., -np.sin(np.pi * x))
                        bc = np.asarray(predictions(evaluated, np.tile([-1., 1.], len(t)).astype(np.float32), np.repeat(t, 2)))
                        row = dict(seed=seed, method=method, budget=step, checkpoint=checkpoint,
                                   selected_step=step if checkpoint == "last" else best_step,
                                   rmse=float(np.sqrt(np.mean(error ** 2))),
                                   relative_l2=float(np.linalg.norm(error) / np.linalg.norm(reference)),
                                   residual_rms=float(np.sqrt(np.mean(independent.astype(float) ** 2))),
                                   ic_max=float(np.max(np.abs(ic - exact_ic))), bc_max=float(np.max(np.abs(bc))))
                        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                            raise FloatingPointError("Nonfinite evaluation metric")
                        metrics.append(row)
                        if seed == config["seeds"][0]:
                            np.savez_compressed(folder / f"field_{step}_{checkpoint}.npz", x=x, t=t, reference=reference, prediction=pred)
                if method == "qr_deim":
                    selected, singular, modes = select_qr(snapshots, config["epsilon"])
                    current = np.abs(evaluate_residual(params, points))
                    updated, baseline, removed, d, accepted = prune(
                        points, candidates, selected, old, current, np.abs(snapshots[:, -1]))
                    updates.append(dict(step=step, k=len(selected), accepted=accepted, replaced=len(removed),
                                        min_selected_degree=float(np.min(np.sort(d)[-len(selected):])) if len(selected) else 0.,
                                        used_in_next_training=step < config["steps"]))
                    if seed == config["seeds"][0] and step in config["audit_steps"]:
                        np.savez_compressed(folder / f"audit_{step}.npz", snapshots=snapshots, candidates=candidates,
                                            before=points, after=updated, singular_values=singular, modes=modes,
                                            selected=selected, removed=removed, degrees=d, old=old, current=current,
                                            next_baseline=baseline, accepted=accepted)
                    points, old = updated, baseline
                    candidates = uniform_points(rng, config["snapshot_points"], points)
                else:
                    points = sobol_points(config["active_points"], seed + step * 100)
                if step % max(config["period"], config["steps"] // 10) == 0:
                    print(f"seed={seed} method={method} step={step} val={val_mse:.6g}", flush=True)
            write_csv(folder / "trace.csv", trace)
            if updates:
                write_csv(folder / "updates.csv", updates)
            write_csv(folder / "metrics.csv", metrics)
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            np.savez_compressed(folder / "points.npz", active=np.asarray(point_history), final=points,
                                validation=validation, independent_test=test)
            arrays = {f"state_{i}": np.asarray(a) for i, a in enumerate(jax.tree.leaves(state))}
            arrays.update({f"best_{i}": np.asarray(a) for i, a in enumerate(jax.tree.leaves(best_params))})
            np.savez_compressed(folder / "parameters.npz", **arrays)
            (folder / "complete.json").write_text(json.dumps(dict(
                seconds=time.perf_counter() - start, best_step=best_step, best_validation_mse=best_value,
                backend=jax.default_backend(), jax_version=jax.__version__,
                test_time_range=[float(t.min()), float(t.max())])), encoding="utf-8")
            all_metrics.extend(metrics)
    write_csv(output / "metrics.csv", all_metrics)
    return output
