"""One-seed Burgers pilot for residual-history collocation selection."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.stats import qmc

from .controlled import baseline_ad_loss, make_controlled_batches
from .data import load_burgers_data, split_times
from .models import NU, burgers_residual, burgers_solution, init_mlp
from .training import adam_init, adam_step, make_update


METHODS = ("uniform", "sobol", "rad", "pod_qr", "pca_mean_qr")
LABELS = {
    "uniform": "Aleatorio uniforme",
    "sobol": "Sobol",
    "rad": "RAD",
    "pod_qr": "POD + QR",
    "pca_mean_qr": "PCA centrado + media + QR",
}


@jax.jit
def _residual_batch(params, points):
    return jax.vmap(lambda p: burgers_residual(params, p[0], p[1]))(points)


@jax.jit
def _predict_grid(params, x, times):
    xx, tt = jnp.meshgrid(x, times, indexing="ij")
    values = jax.vmap(lambda a, b: burgers_solution(params, a, b))(
        xx.reshape(-1), tt.reshape(-1))
    return values.reshape((x.size, times.size))


def _qmc_points(n: int, seed: int, x_bounds, t_bounds):
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)
    power = int(math.ceil(math.log2(n)))
    unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2(power)[:n]
    return np.column_stack((
        x_bounds[0] + unit[:, 0] * (x_bounds[1] - x_bounds[0]),
        t_bounds[0] + unit[:, 1] * (t_bounds[1] - t_bounds[0]),
    )).astype(np.float32)


def _uniform_points(n: int, rng, x_bounds, t_bounds):
    return np.column_stack((rng.uniform(*x_bounds, n), rng.uniform(*t_bounds, n))).astype(np.float32)


def _residual_field(params, points, chunk=256):
    values = []
    for start in range(0, len(points), chunk):
        result = _residual_batch(params, jnp.asarray(points[start:start + chunk]))
        values.append(np.asarray(result.block_until_ready()))
    return np.concatenate(values)


def _history_basis(history: np.ndarray, method: str, threshold: float, max_modes: int):
    """Build the retained spatial basis, preserving the mean for centered PCA."""
    if method == "pod_qr":
        snapshots = history
        mean = None
    elif method == "pca_mean_qr":
        mean = history.mean(axis=1, keepdims=True)
        snapshots = history - mean
    else:
        raise ValueError(method)

    left, singular, _ = np.linalg.svd(snapshots, full_matrices=False)
    energy = singular**2
    if energy.sum() <= np.finfo(float).tiny:
        if mean is not None and np.linalg.norm(mean) > np.finfo(float).eps:
            return mean / np.linalg.norm(mean), singular, 0
        return np.empty((len(history), 0)), singular, 0
    modes = int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1)
    modes = min(modes, max_modes, left.shape[1])
    basis = left[:, :modes]
    if mean is not None:
        mean_norm = np.linalg.norm(mean)
        if mean_norm > np.finfo(float).eps:
            basis = np.column_stack((mean[:, 0] / mean_norm, basis))

    return basis, singular, modes


def _qr_select(history: np.ndarray, method: str, threshold: float, max_modes: int):
    """Return candidate indices, spectrum, and retained mode count."""
    basis, singular, modes = _history_basis(history, method, threshold, max_modes)
    if basis.shape[1] == 0:
        return np.array([], dtype=int), singular, modes
    _, r, pivots = qr(basis.T, pivoting=True, mode="economic")
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        rank = 0
    else:
        tolerance = np.finfo(float).eps * max(basis.shape) * diagonal[0]
        rank = int(np.sum(diagonal > tolerance))
    return np.asarray(pivots[:rank], dtype=int), singular, modes


def _oversampled_qr_select(history, method, threshold, max_modes, count):
    """QR initialization followed by greedy log-det gains in the same basis.

    Extra pivots of rectangular QR past its rank are not informative. Instead,
    each additional row maximizes b.T @ inv(B_selected.T @ B_selected) @ b.
    A fixed ridge of 1e-10 stabilizes this information matrix; Sherman-Morrison
    updates its inverse. No residual magnitude enters this criterion.
    """
    if not 0 <= count <= len(history):
        raise ValueError("Requested count must fit the candidate pool.")
    ids, _, modes = _qr_select(history, method, threshold, max_modes)
    basis, _, _ = _history_basis(history, method, threshold, max_modes)
    if count == 0 or basis.shape[1] == 0:
        return np.array([], dtype=int), modes
    # Remove dependent columns (the preserved mean may overlap the PCA modes).
    basis, singular, _ = np.linalg.svd(basis.astype(np.float64), full_matrices=False)
    rank = int(np.sum(singular > np.finfo(float).eps * max(history.shape) * singular[0]))
    basis = basis[:, :rank]
    selected = list(ids[:count])
    information = basis[selected].T @ basis[selected] + 1e-10 * np.eye(rank)
    inverse = np.linalg.inv(information)
    available = np.ones(len(history), dtype=bool)
    available[selected] = False
    while len(selected) < count:
        scores = np.einsum('ij,ij->i', basis @ inverse, basis)
        scores[~available] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        available[index] = False
        direction = inverse @ basis[index]
        inverse -= np.outer(direction, direction) / (1.0 + basis[index] @ direction)
        inverse = (inverse + inverse.T) * 0.5
    return np.asarray(selected, dtype=int), modes


def _choose_points(method, pool, history, current_residual, n_active, qr_count,
                   threshold, max_modes, seed, rng, bounds):
    if method == "uniform":
        return _uniform_points(n_active, rng, *bounds), np.array([], dtype=int), 0
    if method == "sobol":
        return _qmc_points(n_active, seed, *bounds), np.array([], dtype=int), 0
    if method == "rad":
        magnitude = np.abs(current_residual).astype(np.float64)
        power = 2.0
        weights = (magnitude - magnitude.min() + 1e-8) ** power
        probability = weights / weights.sum()
        count = min(n_active, len(pool))
        ids = rng.choice(len(pool), count, replace=False, p=probability)
        return pool[ids], ids, 0

    ids, _, modes = _qr_select(history, method, threshold, max_modes)
    ids = ids[:qr_count]
    selected = pool[ids]
    remaining = n_active - len(selected)
    # A common QMC background keeps the active batch size fixed when rank < batch size.
    background = _qmc_points(remaining, seed + 104729, *bounds) if remaining else np.empty((0, 2), np.float32)
    return np.concatenate((selected, background)), ids, modes


def _evaluate(params, x, t, u, ids, residual_eval_points):
    prediction = np.asarray(_predict_grid(params, jnp.asarray(x), jnp.asarray(t[ids])).block_until_ready())
    truth = u[:, ids]
    denom = np.linalg.norm(truth)
    residual = _residual_field(params, residual_eval_points)
    ic = np.asarray(jax.vmap(lambda xx: burgers_solution(params, xx, jnp.asarray(0.0)))(jnp.asarray(x)))
    boundary = np.asarray(jax.vmap(lambda tt: jnp.array([
        burgers_solution(params, jnp.asarray(x[0]), tt),
        burgers_solution(params, jnp.asarray(x[-1]), tt),
    ]))(jnp.asarray(t[ids])))
    return {
        "test_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "test_relative_l2": float(np.linalg.norm(prediction - truth) / denom),
        "independent_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "initial_condition_rmse": float(np.sqrt(np.mean((ic - u[:, 0]) ** 2))),
        "boundary_rms": float(np.sqrt(np.mean(boundary**2))),
    }


def _save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _initial_condition_indices(n_space, count, strategy="left_prefix"):
    """Keep the IC budget fixed while selecting its spatial locations."""
    if not 1 <= count <= n_space:
        raise ValueError("Initial-condition point count must fit the spatial grid.")
    if strategy == "left_prefix":
        return np.arange(count)
    if strategy == "uniform_domain":
        if count < 2:
            raise ValueError("Uniform-domain IC coverage requires at least two points.")
        return np.rint(np.linspace(0, n_space - 1, count)).astype(int)
    raise ValueError(f"Unknown initial-condition strategy: {strategy}")


def _run_collocation_seed(config: dict, root: Path, output: Path, observer=None):
    root = Path(root).resolve()
    residual_fn, evaluate_fn, loss_fn = _residual_field, _evaluate, baseline_ad_loss
    if config.get("hard_conditions"):
        from .collocation_hard_conditions import residual_field, evaluate, loss
        residual_fn, evaluate_fn, loss_fn = residual_field, evaluate, loss
        if config.get("pde_only"):
            from .collocation_hard_conditions import pde_only_loss
            loss_fn = pde_only_loss
    elif config.get("pde_only"):
        raise ValueError("PDE-only control requires hard initial and boundary conditions")
    output.mkdir(parents=True, exist_ok=True)
    x, t, u = load_burgers_data(root / config["data_path"])
    seed = int(config["seed"])
    fit_ids, validation_ids, test_ids = split_times(
        len(t), seed, int(config["test_time_count"]), int(config["validation_time_count"]))
    x_bounds, t_bounds = (float(x.min()), float(x.max())), (float(t.min()), float(t.max()))
    bounds = (x_bounds, t_bounds)
    pool = _qmc_points(int(config["candidate_count"]), seed + 1, *bounds)
    initial_pool = pool.copy()
    independent = _qmc_points(int(config["independent_count"]), seed + 2, *bounds)
    init_ids = _initial_condition_indices(len(x), int(config["initial_condition_points"]),
                                          config.get("initial_condition_strategy", "left_prefix"))
    warm_steps, train_steps = int(config["warmup_steps"]), int(config["train_steps"])
    batch_size, update_every = int(config["batch_size"]), int(config["update_every"])
    batches = make_controlled_batches(x, t, u, fit_ids, init_ids, warm_steps + train_steps, batch_size, seed + 3)

    width = int(config["width"])
    params = init_mlp(jax.random.PRNGKey(seed), [2, width, width, width, 1])
    m1, m2, adam_step_id = adam_init(params)
    update = make_update(loss_fn, float(config["learning_rate"]))
    rng = np.random.default_rng(seed + 4)
    history = []
    snapshots = int(config["history_snapshots"])
    sample_steps = np.unique(np.linspace(1, warm_steps, snapshots, dtype=int))
    start_warm = time.perf_counter()
    for step_i, batch in enumerate(batches[:warm_steps], start=1):
        params, m1, m2, adam_step_id, loss = update(params, m1, m2, adam_step_id, batch)
        if step_i in sample_steps:
            history.append(residual_fn(params, pool))
    jax.tree.map(lambda value: value.block_until_ready(), params)
    warm_seconds = time.perf_counter() - start_warm
    history = np.column_stack(history)
    initial_residual = history[:, -1]

    # Compile the shared evaluation kernels before timing any method.
    setup_start = time.perf_counter()
    warmup_validation = evaluate_fn(params, x, t, u, validation_ids, independent)
    shared_setup_seconds = warm_seconds + (time.perf_counter() - setup_start)

    # Fixed initial state, including Adam moments, is cloned for every method.
    initial_params, initial_m1, initial_m2, initial_adam_step = params, m1, m2, adam_step_id
    if config.get("loss_weights"):
        from .collocation_loss_balance import weighted_loss
        update = make_update(weighted_loss(config["loss_weights"]), float(config["learning_rate"]))
    if observer is not None:
        observer.begin_seed(seed=seed, x=x, t=t, u=u, fit_ids=fit_ids,
                            validation_ids=validation_ids, test_ids=test_ids,
                            pool=pool, independent=independent, initial_ids=init_ids,
                            probe_batch=batches[warm_steps])
    summary_rows, trace_rows, selection_rows, selected_by_method = [], [], [], {}
    first_selection = {}
    first_qr_selection = {}
    t0 = time.perf_counter()
    specs = config.get("method_specs", {key: {"base": key, "label": LABELS[key]} for key in METHODS})
    for method_i, (method, spec) in enumerate(specs.items()):
        base_method = spec["base"]
        label = spec["label"]
        method_params, method_m1, method_m2, method_adam_step = (
            initial_params, initial_m1, initial_m2, initial_adam_step)
        method_history = history.copy()
        method_start = time.perf_counter()
        trace_rows.append({
            "seed": seed, "method": method, "label": label, "step": warm_steps,
            "elapsed_after_warmup_seconds": 0.0,
            "pipeline_seconds": shared_setup_seconds,
            "validation_rmse": warmup_validation["test_rmse"], "training_loss": None,
            "independent_residual_rms": warmup_validation["independent_residual_rms"],
            "initial_condition_rmse": warmup_validation["initial_condition_rmse"],
            "boundary_rms": warmup_validation["boundary_rms"],
        })
        last_selected = np.empty((0, 2), dtype=np.float32)
        last_ids = np.array([], dtype=int)
        for chunk_start in range(warm_steps, warm_steps + train_steps, update_every):
            if observer is not None and hasattr(observer, "prepare_history"):
                pool, method_history = observer.prepare_history(chunk_start, pool, method_history, bounds)
            current_residual = residual_fn(method_params, pool)
            selection_seed = seed + (0 if config.get("common_selection_seed") else method_i * 1009) + chunk_start
            basis_info = None
            if spec.get("rank_schedule") is not None:
                from .collocation_pde_bases import matched_selection
                modes = int(spec["rank_schedule"][str(chunk_start)])
                ids, basis_info = matched_selection(method_history, base_method, modes)
                background = _qmc_points(batch_size - len(ids), selection_seed + 104729, *bounds)
                active = np.concatenate((pool[ids], background))
            elif spec.get("adaptive_basis_threshold") is not None:
                from .collocation_rank_study import select_adaptive_basis
                if spec.get("direct_rank"):
                    from .collocation_rank_study import adaptive_basis
                    basis, basis_info = adaptive_basis(method_history, base_method, float(spec["adaptive_basis_threshold"]))
                    basis_info["actual_dimension"] = basis.shape[1]
                    _, _, pivots = qr(basis.T, pivoting=True, mode="economic")
                    ids = np.asarray(pivots[:basis.shape[1]], dtype=int)
                else:
                    ids, basis_info = select_adaptive_basis(
                        method_history, base_method, float(spec["adaptive_basis_threshold"]),
                        int(spec["oversample_count"]))
                modes = basis_info["actual_dimension"]
                background = _qmc_points(batch_size - len(ids), selection_seed + 104729, *bounds)
                active = np.concatenate((pool[ids], background))
            elif spec.get("fixed_basis_dimension") is not None:
                from .collocation_basis_ablation import select_fixed_basis
                ids, basis_info = select_fixed_basis(
                    method_history, base_method, int(spec["fixed_basis_dimension"]),
                    int(spec["oversample_count"]))
                modes = basis_info["actual_dimension"]
                background = _qmc_points(batch_size - len(ids), selection_seed + 104729, *bounds)
                active = np.concatenate((pool[ids], background))
            elif spec.get("oversample_count") is not None:
                ids, modes = _oversampled_qr_select(
                    method_history, base_method, float(config["energy_threshold"]),
                    int(config["max_modes"]), int(spec["oversample_count"]))
                background = _qmc_points(batch_size - len(ids), selection_seed + 104729, *bounds)
                active = np.concatenate((pool[ids], background))
            else:
                active, ids, modes = _choose_points(
                    base_method, pool, method_history, current_residual,
                    batch_size, int(config["qr_point_count"]),
                    float(config["energy_threshold"]), int(config["max_modes"]),
                    selection_seed, np.random.default_rng(selection_seed), bounds)
                if config.get("common_selection_seed") and base_method == "sobol":
                    active = _qmc_points(batch_size, selection_seed + 104729, *bounds)
            if observer is not None and hasattr(observer, "adjust_active"):
                active, ids, modes = observer.adjust_active(method, chunk_start, method_params,
                    active, ids, modes, pool, method_history, selection_seed, bounds)
            last_selected, last_ids = active, ids
            if observer is not None:
                if hasattr(observer, "selection_details"):
                    observer.selection_details(method, chunk_start, method_history, basis_info)
                observer.selection(method, chunk_start, active, ids, modes)
                if chunk_start == warm_steps:
                    observer.checkpoint(method, warm_steps, method_params, active, chunk_start)
            if chunk_start == warm_steps:
                first_selection[method] = active.copy()
                first_qr_selection[method] = pool[ids].copy() if base_method in ("pod_qr", "pca_qr", "pca_mean_qr") else np.empty((0, 2), np.float32)
            for step_i in range(chunk_start, min(chunk_start + update_every, warm_steps + train_steps)):
                original_batch = batches[step_i]
                batch = (*original_batch[:3], jnp.asarray(active[:, 0]), jnp.asarray(active[:, 1]), *original_batch[5:])
                method_params, method_m1, method_m2, method_adam_step, loss = update(
                    method_params, method_m1, method_m2, method_adam_step, batch)
                if observer is not None and hasattr(observer, "training_step"):
                    observer.training_step(step_i + 1, method_params)
            jax.tree.map(lambda value: value.block_until_ready(), method_params)
            # Keep a rolling history window; the next update adapts to the recent residual evolution.
            method_history = np.column_stack((method_history, residual_fn(method_params, pool)))
            validation = evaluate_fn(method_params, x, t, u, validation_ids, independent)
            trace_rows.append({
                "seed": seed, "method": method, "label": label, "step": min(chunk_start + update_every, warm_steps + train_steps),
                "elapsed_after_warmup_seconds": time.perf_counter() - method_start,
                "pipeline_seconds": shared_setup_seconds + time.perf_counter() - method_start,
                "validation_rmse": validation["test_rmse"], "training_loss": float(loss),
                "independent_residual_rms": validation["independent_residual_rms"],
                "initial_condition_rmse": validation["initial_condition_rmse"],
                "boundary_rms": validation["boundary_rms"],
            })
            if observer is not None:
                observer.checkpoint(method, min(chunk_start + update_every, warm_steps + train_steps),
                                    method_params, active, chunk_start)
            if method_history.shape[1] > snapshots:
                method_history = method_history[:, -snapshots:]
            selection_rows.append({
                "seed": seed, "method": method, "update_step": min(chunk_start + update_every, warm_steps + train_steps),
                "selected_qr_points": int(len(ids)) if base_method in ("pod_qr", "pca_qr", "pca_mean_qr") else 0,
                "requested_basis_points": spec.get("oversample_count"),
                "selection_rule": "qr_then_logdet" if spec.get("oversample_count") is not None else base_method,
                "retained_modes": modes,
                "selected_mean_abs_residual": float(np.mean(np.abs(current_residual[ids]))) if len(ids) else None,
            })
            if config.get("basis_ablation"):
                selection_rows[-1].update(basis_info or dict(actual_dimension=0, centered_variance_captured=None,
                    raw_energy_captured=None, mean_projection_fraction=None, qr_initial_count=0))
            if spec.get("rank_schedule") is not None:
                selection_rows[-1]["selection_rule"] = "matched_rank_direct_qr"
        metrics = evaluate_fn(method_params, x, t, u, test_ids, independent)
        elapsed = time.perf_counter() - method_start
        summary_rows.append({
            "method": method, "label": label, "seed": seed,
            "warmup_steps": warm_steps, "train_steps": train_steps,
            "active_points": batch_size, "update_every": update_every,
            "warmup_seconds_shared": warm_seconds,
            "shared_setup_seconds_including_warmup": shared_setup_seconds,
            "training_seconds": elapsed,
            "pipeline_seconds_including_shared_setup": shared_setup_seconds + elapsed,
            "warmup_validation_rmse": warmup_validation["test_rmse"],
            "final_training_loss": float(loss), **metrics,
        })
        selected_by_method[method] = last_selected

    _save_csv(output / "metrics.csv", summary_rows)
    _save_csv(output / "validation_trajectories.csv", trace_rows)
    _save_csv(output / "selection_diagnostics.csv", selection_rows)
    np.savez_compressed(
        output / "selection_snapshot.npz", candidates=initial_pool, residual_history=history,
        initial_residual=initial_residual,
        **{f"selected_{key}": val for key, val in first_selection.items()},
        **{f"qr_selected_{key}": val for key, val in first_qr_selection.items()},
    )
    config_saved = dict(config)
    config_saved["methods"] = list(specs)
    config_saved["methods_note"] = (
        "Method specs distinguish rank-limited QR from QR-initialized greedy log-det oversampling; remaining slots use a common nested Sobol prefix."
        if config.get("method_specs") else
        "POD/PCA-QR contributes up to qr_point_count selected points; the remaining active slots use a Sobol background with a method-specific scramble.")
    config_saved["timing_note"] = "The uniform warm-up is shared by all methods and included once in each pipeline total."
    (output / "config.json").write_text(json.dumps(config_saved, indent=2), encoding="utf-8")
    if not config.get("skip_seed_plots", False):
        _plot_outputs(output, pool, initial_residual, history, first_selection, first_qr_selection, trace_rows, config)
    summary = {
        "run_id": config["run_id"], "seed": seed, "warmup_seconds_shared": warm_seconds,
        "shared_setup_seconds_including_warmup": shared_setup_seconds,
        "wall_seconds": time.perf_counter() - t0, "methods": summary_rows,
        "candidate_count": len(pool), "history_snapshots": history.shape[1],
        "history_matrix_shape": list(history.shape),
        "qr_count_note": config_saved["methods_note"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if observer is not None:
        observer.end_seed()
    return output


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values))}


def run_collocation_pilot(config: dict, project_root: str | Path):
    """Run one or more paired seeds and write a combined report and notebook."""
    root = Path(project_root).resolve()
    output = root / "pca_collocation" / "results" / "burgers" / "collocation_selection" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 2041)])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("La configuración debe incluir al menos una semilla, sin duplicados.")

    per_seed_summaries, metric_rows, trace_rows, selection_rows = [], [], [], []
    for seed in seeds:
        seed_config = dict(config, seed=seed)
        seed_dir = output / f"seed_{seed}"
        _run_collocation_seed(seed_config, root, seed_dir)
        per_seed_summaries.append(json.loads((seed_dir / "summary.json").read_text(encoding="utf-8")))
        for filename, target in (("metrics.csv", metric_rows),
                                 ("validation_trajectories.csv", trace_rows),
                                 ("selection_diagnostics.csv", selection_rows)):
            with (seed_dir / filename).open(encoding="utf-8", newline="") as stream:
                target.extend(csv.DictReader(stream))

    for row in metric_rows:
        for key in ("seed", "warmup_steps", "train_steps", "active_points", "update_every"):
            if key in row:
                row[key] = int(row[key])
        for key in ("warmup_seconds_shared", "shared_setup_seconds_including_warmup", "training_seconds",
                    "pipeline_seconds_including_shared_setup", "warmup_validation_rmse", "final_training_loss",
                    "test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms"):
            if key in row:
                row[key] = float(row[key])
    for row in trace_rows:
        for key in ("seed", "step"):
            row[key] = int(row[key])
        for key in ("elapsed_after_warmup_seconds", "pipeline_seconds", "validation_rmse"):
            row[key] = float(row[key])
        if row.get("training_loss") not in (None, ""):
            row["training_loss"] = float(row["training_loss"])
        else:
            row["training_loss"] = None
    for row in selection_rows:
        row["seed"] = int(row["seed"])
        for key in ("update_step", "selected_qr_points", "retained_modes"):
            row[key] = int(row[key])
        if row.get("selected_mean_abs_residual") not in (None, ""):
            row["selected_mean_abs_residual"] = float(row["selected_mean_abs_residual"])
        else:
            row["selected_mean_abs_residual"] = None

    _save_csv(output / "metrics_by_seed.csv", metric_rows)
    _save_csv(output / "validation_trajectories_by_seed.csv", trace_rows)
    _save_csv(output / "selection_diagnostics_by_seed.csv", selection_rows)

    baseline_by_seed = {int(row["seed"]): row for row in metric_rows if row["method"] == "uniform"}
    aggregate, paired = [], {}
    numeric_metrics = ("test_rmse", "test_relative_l2", "independent_residual_rms",
                       "initial_condition_rmse", "boundary_rms", "training_seconds",
                       "pipeline_seconds_including_shared_setup")
    for method in METHODS:
        rows = [row for row in metric_rows if row["method"] == method]
        record = {"method": method, "label": LABELS[method], "seed_count": len(rows)}
        for metric in numeric_metrics:
            record[metric] = _mean_std([row[metric] for row in rows])
        aggregate.append(record)
        if method == "uniform":
            continue
        by_seed = {int(row["seed"]): row for row in rows}
        common = sorted(set(by_seed) & set(baseline_by_seed))
        paired[method] = {}
        for metric in numeric_metrics:
            deltas = [by_seed[seed][metric] - baseline_by_seed[seed][metric] for seed in common]
            paired[method][metric] = {
                **_mean_std(deltas),
                "method_better_count": int(sum(delta < 0 for delta in deltas)),
            }

    target = float(config["validation_rmse_target"])
    targets = []
    for seed in seeds:
        for method in METHODS:
            rows = sorted((row for row in trace_rows if row["seed"] == seed and row["method"] == method),
                          key=lambda row: row["step"])
            hit = next((row for row in rows if row["validation_rmse"] <= target), None)
            final = rows[-1]
            targets.append({
                "seed": seed, "method": method, "label": LABELS[method],
                "validation_rmse_target": target,
                "reached": hit is not None,
                "time_to_target_pipeline_seconds": hit["pipeline_seconds"] if hit else None,
                "step_at_target": hit["step"] if hit else None,
                "censor_step": final["step"] if hit is None else None,
                "final_validation_rmse": final["validation_rmse"],
                "observed_pipeline_seconds": final["pipeline_seconds"],
            })
    _save_csv(output / "time_to_validation_target_by_seed.csv", targets)
    target_summary = {}
    for method in METHODS:
        rows = [row for row in targets if row["method"] == method]
        hits = [row["time_to_target_pipeline_seconds"] for row in rows if row["reached"]]
        target_summary[method] = {
            "reached_count": len(hits), "seed_count": len(rows),
            "time_to_target_seconds_among_hits": _mean_std(hits) if hits else None,
            "final_validation_rmse": _mean_std([row["final_validation_rmse"] for row in rows]),
            "censored_count": len(rows) - len(hits),
        }

    combined = {
        "run_id": config["run_id"], "seeds": seeds, "methods": aggregate,
        "paired_differences_vs_uniform": paired,
        "validation_target": {"rmse": target, "by_method": target_summary,
                              "rule": "first recorded validation checkpoint at or below the fixed threshold; no interpolation"},
        "protocol": {
            "warmup_steps": int(config["warmup_steps"]), "continuation_steps": int(config["train_steps"]),
            "active_points": int(config["batch_size"]), "update_every": int(config["update_every"]),
            "candidate_count": int(config["candidate_count"]), "history_snapshots": int(config["history_snapshots"]),
            "energy_threshold": float(config["energy_threshold"]), "max_modes": int(config["max_modes"]),
            "qr_point_count_cap": int(config["qr_point_count"]),
            "fixed_candidate_pool_per_seed": True,
            "rad_rule": "sample without replacement with probabilities proportional to (|r|-min(|r|)+1e-8)^2",
            "qr_rule": "SVD modes retained to the squared-singular-value energy threshold; pivoted QR selects distinct pool indices; Sobol fills remaining active slots",
            "note": "All methods share the per-seed warm-up weights, Adam state, supervised batches, held-out time split, candidate pool, and independent residual evaluation pool.",
        },
        "per_seed_summaries": per_seed_summaries,
    }
    (output / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_confirmation(output, metric_rows, trace_rows, targets, target)
    _build_notebook(root, output.relative_to(root), "07_collocation_selection_confirmation.ipynb")
    return output


def _plot_confirmation(output, metric_rows, trace_rows, targets, target):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [LABELS[method] for method in METHODS]
    for ax, metric, ylabel, title in (
        (axes[0], "test_relative_l2", "Error L2 relativo de prueba", "Calidad a pasos fijos"),
        (axes[1], "independent_residual_rms", "RMS del residuo independiente", "Física fuera del entrenamiento"),
    ):
        for i, method in enumerate(METHODS):
            values = [float(row[metric]) for row in metric_rows if row["method"] == method]
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(len(values))
            ax.scatter(i + jitter, values, s=25, alpha=.75)
            ax.plot([i - .2, i + .2], [np.mean(values)] * 2, color="black", lw=2)
        ax.set_xticks(range(len(METHODS)), labels, rotation=22, ha="right")
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "confirmation_quality.png", dpi=180); fig.savefig(output / "confirmation_quality.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for method in METHODS:
        groups = []
        for seed in sorted({int(row["seed"]) for row in trace_rows}):
            vals = [row["validation_rmse"] for row in trace_rows if row["seed"] == seed and row["method"] == method]
            groups.append(vals)
        checkpoints = sorted({int(row["step"]) for row in trace_rows})
        means = [np.mean([row["validation_rmse"] for row in trace_rows
                          if row["method"] == method and int(row["step"]) == step]) for step in checkpoints]
        stds = [np.std([row["validation_rmse"] for row in trace_rows
                        if row["method"] == method and int(row["step"]) == step], ddof=1) for step in checkpoints]
        ax.plot(checkpoints, means, marker="o", label=LABELS[method])
        ax.fill_between(checkpoints, np.asarray(means) - stds, np.asarray(means) + stds, alpha=.12)
    ax.axhline(target, color="black", linestyle="--", label=f"Objetivo RMSE {target:.2f}")
    ax.set(xlabel="Paso global (incluye calentamiento común)", ylabel="RMSE de validación",
           title="Promedio ± DE entre semillas")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output / "confirmation_validation.png", dpi=180); fig.savefig(output / "confirmation_validation.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    hit_rates = [sum(bool(row["reached"]) for row in targets if row["method"] == method)
                 / max(1, sum(row["method"] == method for row in targets)) for method in METHODS]
    ax.bar([LABELS[m] for m in METHODS], hit_rates, color="#3b7c6b")
    ax.set(ylim=(0, 1.05), ylabel="Fracción de semillas que alcanza el objetivo",
           title=f"Tiempo a RMSE de validación ≤ {target:.2f}")
    ax.tick_params(axis="x", rotation=22); ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "target_hit_rate.png", dpi=180); fig.savefig(output / "target_hit_rate.svg"); plt.close(fig)


def _plot_outputs(output, pool, residual, history, selections, qr_selections, traces, config):
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    sc = ax.scatter(pool[:, 0], pool[:, 1], c=np.abs(residual), s=8, cmap="magma", rasterized=True)
    fig.colorbar(sc, ax=ax, label="|residuo Burgers| tras el calentamiento")
    ax.set(xlabel="x", ylabel="t", title="Residuo en el conjunto candidato")
    fig.savefig(output / "residual_map.png", dpi=180); fig.savefig(output / "residual_map.svg"); plt.close(fig)

    centered = history - history.mean(axis=1, keepdims=True)
    svd_data = {}
    for name, matrix in (("POD sin centrar", history), ("PCA centrado", centered)):
        values = np.linalg.svd(matrix, compute_uv=False)
        energy = values**2 / max(np.sum(values**2), np.finfo(float).tiny)
        svd_data[name] = values
        svd_data[f"{name}_cum"] = np.cumsum(energy)
    np.savez_compressed(output / "spectra.npz", **svd_data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in ("POD sin centrar", "PCA centrado"):
        values = svd_data[name]
        axes[0].semilogy(np.arange(1, len(values) + 1), values / max(values[0], 1e-30), marker="o", label=name)
        axes[1].plot(np.arange(1, len(values) + 1), svd_data[f"{name}_cum"], marker="o", label=name)
    axes[0].set(xlabel="Modo", ylabel="σᵢ / σ₁", title="Espectro singular")
    axes[1].axhline(float(config["energy_threshold"]), color="black", ls="--", lw=1)
    axes[1].set(xlabel="Número de modos", ylabel="Energía acumulada (σ²)", ylim=(0, 1.02), title="Energía retenida")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / "singular_spectrum.png", dpi=180); fig.savefig(output / "singular_spectrum.svg"); plt.close(fig)

    pod_modes = np.linalg.svd(history, full_matrices=False)[0]
    fig, axes = plt.subplots(1, min(3, pod_modes.shape[1]), figsize=(12, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        sc = ax.scatter(pool[:, 0], pool[:, 1], c=pod_modes[:, i], s=8, cmap="coolwarm", rasterized=True)
        fig.colorbar(sc, ax=ax, label=f"Modo {i + 1}")
        ax.set(xlabel="x", ylabel="t", title=f"Modo POD {i + 1}")
    fig.savefig(output / "residual_modes.png", dpi=180); fig.savefig(output / "residual_modes.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True, sharex=True, sharey=True)
    methods = ("uniform", "sobol", "rad", "pod_qr", "pca_mean_qr")
    for ax, method in zip(axes.flat, methods):
        ax.scatter(pool[:, 0], pool[:, 1], s=2, color="#dddddd", rasterized=True)
        points = selections[method]
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=.75, label="lote activo")
        qr_points = qr_selections[method]
        if len(qr_points):
            ax.scatter(qr_points[:, 0], qr_points[:, 1], s=48, marker="*", color="#d62728",
                       edgecolor="white", linewidth=.5, label="índices elegidos por QR")
        ax.legend(fontsize=7, loc="best")
        ax.set(title=LABELS[method], xlabel="x", ylabel="t")
    axes.flat[-1].axis("off")
    fig.suptitle("Puntos activos de la primera actualización (el tamaño del lote es fijo)")
    fig.savefig(output / "selected_points.png", dpi=180); fig.savefig(output / "selected_points.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for method in METHODS:
        rows = [row for row in traces if row["method"] == method]
        ax.plot([row["step"] for row in rows], [row["validation_rmse"] for row in rows],
                marker="o", label=LABELS[method])
    ax.set(xlabel="Paso de entrenamiento", ylabel="RMSE de validación", title="Evolución tras el calentamiento compartido")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output / "validation_rmse.png", dpi=180); fig.savefig(output / "validation_rmse.svg"); plt.close(fig)


def _build_notebook(root, output_relative, notebook_name="06_collocation_selection_pilot.ipynb"):
    output = Path(output_relative).as_posix()
    output_dir = root / output_relative
    if (output_dir / "metrics_by_seed.csv").exists():
        return _build_confirmation_notebook(root, output_relative, notebook_name)
    rows = list(csv.DictReader((output_dir / "metrics.csv").open(encoding="utf-8", newline="")))
    table = [
        "| Método | RMSE prueba | L2 relativo | Residuo independiente | Tiempo pipeline (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['label']} | {float(row['test_rmse']):.4f} | {float(row['test_relative_l2']):.4f} "
            f"| {float(row['independent_residual_rms']):.4f} | {float(row['pipeline_seconds_including_shared_setup']):.3f} |"
        )
    asset = f"../../{Path(output).relative_to('pca_collocation').as_posix()}/"
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Piloto: selección de puntos de colocación con historia del residuo\n\n"
            "Una semilla de Burgers compara aleatorio uniforme, Sobol, RAD, POD sin centrar + QR y PCA centrado con la media explícita + QR. "
            "Todos los métodos comparten un calentamiento uniforme, inicialización, estado de Adam, muestras supervisadas y tamaño/frecuencia de actualización del lote físico. "
            "El piloto es exploratorio, no una comparación estadística."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\n\n"
            "ROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').is_file() and ROOT != ROOT.parent: ROOT = ROOT.parent\n"
            f"OUT = ROOT / '{output}'\n"
            "SUMMARY = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))\n"
            "metrics = pd.read_csv(OUT / 'metrics.csv')\n"
            "trace = pd.read_csv(OUT / 'validation_trajectories.csv')\n"
            "display(pd.Series({k: SUMMARY[k] for k in ['seed','candidate_count','history_snapshots','history_matrix_shape','wall_seconds']}))"]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Qué se observa antes de entrenar con los métodos\n\n"
            "La matriz R tiene una fila por punto candidato y una columna por snapshot del residuo. POD usa R sin centrar; PCA resta la media temporal y conserva esa media como una dirección adicional antes de pivotar QR. "
            "El piloto retiene modos por energía cuadrática acumulada (σ²). El artículo QR-DEIM usa SVD y QR con pivoteo, y actualiza/poda puntos adaptativamente; esta primera versión mantiene fijo el tamaño activo y no implementa poda."]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Resultados del piloto (una semilla)\n\n"
            + "\n".join(table) + "\n\n"
            + "La selección QR usó inicialmente "
            + str(next((r["selected_qr_points"] for r in csv.DictReader((output_dir / "selection_diagnostics.csv").open(encoding="utf-8", newline="")) if r["method"] == "pod_qr"), 0))
            + " índices para POD y "
            + str(next((r["selected_qr_points"] for r in csv.DictReader((output_dir / "selection_diagnostics.csv").open(encoding="utf-8", newline="")) if r["method"] == "pca_mean_qr"), 0))
            + " para PCA con media; el límite configurado es 32. Los espacios restantes hasta 128 puntos activos se completan con Sobol. Esta cuenta pequeña refleja el rango de la historia y no es el tamaño total del lote."]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Residuo, espectro y modos\n\n"
            f"![Magnitud del residuo en candidatos]({asset}residual_map.png)\n\n"
            f"![Espectro singular y energía retenida]({asset}singular_spectrum.png)\n\n"
            f"![Primeros modos POD de la historia residual]({asset}residual_modes.png)\n\n"
            f"![Lotes activos y puntos elegidos por QR en rojo]({asset}selected_points.png)"]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Comportamiento de entrenamiento y evaluación independiente\n\n"
            "El residuo físico se evalúa en un conjunto Sobol independiente del candidato y del entrenamiento. Error de solución, residuo, condición inicial y frontera se reportan por separado. "
            "Los tiempos incluyen las actualizaciones de selección; se informa aparte el calentamiento compartido. La compilación común de JAX se resuelve antes de medir cada método. La comparación por igual número de pasos es el resultado principal de este piloto; la comparación a igual precisión requiere más semillas y una regla de objetivo fijada antes de analizar resultados.\n\n"
            f"![RMSE de validación durante el entrenamiento]({asset}validation_rmse.png)"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n\n"
            "ROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').is_file() and ROOT != ROOT.parent: ROOT = ROOT.parent\n"
            f"OUT = ROOT / '{output}'\n"
            "SUMMARY = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))\n"
            "metrics = pd.read_csv(OUT / 'metrics.csv')\n"
            "trace = pd.read_csv(OUT / 'validation_trajectories.csv')\n"
            "display(metrics.sort_values('test_relative_l2').reset_index(drop=True))\n"
            "display(trace.pivot(index='step', columns='label', values='validation_rmse'))"]},
    ]
    notebook = {
        "cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        }, "nbformat": 4, "nbformat_minor": 5,
    }
    path = root / "pca_collocation" / "notebooks" / "burgers" / notebook_name
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    from .notebook_images import embed_notebook_images
    embed_notebook_images(path, root)


def _build_confirmation_notebook(root, output_relative, notebook_name):
    output_dir = root / output_relative
    output = Path(output_relative).as_posix()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    table = [
        "| Método | L2 relativo prueba (media ± DE) | Residuo independiente | Pipeline (s) |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        def fmt(key):
            value = row[key]
            return f"{value['mean']:.4f} ± {value['std']:.4f}"
        table.append(f"| {row['label']} | {fmt('test_relative_l2')} | "
                     f"{fmt('independent_residual_rms')} | {fmt('pipeline_seconds_including_shared_setup')} |")
    qr_rows = list(csv.DictReader((output_dir / "selection_diagnostics_by_seed.csv").open(encoding="utf-8", newline="")))
    first = [row for row in qr_rows if int(row["update_step"]) == int(summary["protocol"]["warmup_steps"] + summary["protocol"]["update_every"])]
    qr_means = {}
    for method in ("pod_qr", "pca_mean_qr"):
        counts = [int(row["selected_qr_points"]) for row in first if row["method"] == method]
        qr_means[method] = f"{np.mean(counts):.1f} (rango {min(counts)}–{max(counts)})" if counts else "sin selección registrada"
    target_info = summary["validation_target"]
    target_table = ["| Método | Alcanzan el objetivo | Tiempo medio entre aciertos (s) | Censuradas |", "|---|---:|---:|---:|"]
    for method in METHODS:
        item = target_info["by_method"][method]
        timing = item["time_to_target_seconds_among_hits"]
        timing_text = f"{timing['mean']:.3f} ± {timing['std']:.3f}" if timing else "—"
        target_table.append(f"| {LABELS[method]} | {item['reached_count']}/{item['seed_count']} | {timing_text} | {item['censored_count']} |")
    asset = f"../../{Path(output).relative_to('pca_collocation').as_posix()}/"
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Confirmación pareada: selección de puntos de colocación\n\n"
            f"Burgers, {len(summary['seeds'])} semillas. Todos los métodos comparten por semilla el calentamiento uniforme, los pesos y estado de Adam resultantes, los lotes supervisados, el conjunto candidato y el presupuesto de puntos. "
            "Se compara calidad al mismo número de pasos y tiempo observado hasta un objetivo de validación fijado antes de correr.\n\n"
            f"El objetivo es RMSE de validación ≤ {target_info['rmse']:.2f}; se registra el primer checkpoint observado que lo alcanza, sin interpolar. Las semillas que no llegan dentro del presupuesto quedan censuradas."]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Resumen por semilla\n\n" + "\n".join(table) + "\n\n"
            "Diferencias pareadas contra uniforme, promedio entre semillas:\n\n"
            + "| Método | Δ L2 relativo | Semillas con menor L2 | Δ residuo independiente | Semillas con menor residuo |\n"
            + "|---|---:|---:|---:|---:|\n"
            + "\n".join(
                f"| {LABELS[method]} | {summary['paired_differences_vs_uniform'][method]['test_relative_l2']['mean']:.4f} "
                f"± {summary['paired_differences_vs_uniform'][method]['test_relative_l2']['std']:.4f} "
                f"| {summary['paired_differences_vs_uniform'][method]['test_relative_l2']['method_better_count']}/{len(summary['seeds'])} "
                f"| {summary['paired_differences_vs_uniform'][method]['independent_residual_rms']['mean']:.4f} "
                f"± {summary['paired_differences_vs_uniform'][method]['independent_residual_rms']['std']:.4f} "
                f"| {summary['paired_differences_vs_uniform'][method]['independent_residual_rms']['method_better_count']}/{len(summary['seeds'])} |"
                for method in METHODS if method != "uniform") ]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Objetivo de calidad equivalente\n\n" + "\n".join(target_table) + "\n\n"
            "Los tiempos incluyen el calentamiento y preparación compartidos, una vez por corrida. Los promedios de tiempo entre aciertos son descriptivos y no ordenan por sí solos a los métodos: deben leerse junto con la tasa de aciertos y las semillas censuradas."]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Diagnósticos visuales\n\n"
            f"![Calidad en el conjunto de prueba y residuo independiente]({asset}confirmation_quality.png)\n\n"
            f"![Trayectorias de validación, promedio y DE]({asset}confirmation_validation.png)\n\n"
            f"![Tasa de semillas que alcanzan el objetivo]({asset}target_hit_rate.png)"]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Protocolo fijado\n\n"
            "RAD muestrea sin reemplazo con probabilidad proporcional a (|r| − min|r| + 1e−8)². POD usa la historia sin centrar; PCA centra temporalmente y preserva su media como dirección adicional. Se retienen modos hasta el umbral de energía σ² y se eligen índices distintos mediante QR con pivoteo. La selección QR contribuyó inicialmente "
            f"{qr_means['pod_qr']} puntos para POD y {qr_means['pca_mean_qr']} para PCA con media (límite 32); los espacios del lote se completan con Sobol.\n\n"
            "El conjunto de candidatos se mantiene fijo dentro de cada semilla; el conjunto independiente para residuo y los tiempos de prueba quedan fuera de la selección. Este estudio no implementa todavía poda adaptativa y sus conclusiones se limitan a este presupuesto y distribución de RAD."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n\n"
            "ROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').is_file() and ROOT != ROOT.parent: ROOT = ROOT.parent\n"
            f"OUT = ROOT / '{output}'\n"
            "summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))\n"
            "metrics = pd.read_csv(OUT / 'metrics_by_seed.csv')\n"
            "trace = pd.read_csv(OUT / 'validation_trajectories_by_seed.csv')\n"
            "target_times = pd.read_csv(OUT / 'time_to_validation_target_by_seed.csv')\n"
            "display(metrics.groupby('method')[['test_relative_l2','independent_residual_rms','pipeline_seconds_including_shared_setup']].agg(['mean','std']))\n"
            "display(target_times.groupby('method').agg(reached=('reached','sum'), seeds=('seed','count')))"
        ]},
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }, "nbformat": 4, "nbformat_minor": 5}
    path = root / "pca_collocation" / "notebooks" / "burgers" / notebook_name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    from .notebook_images import embed_notebook_images
    embed_notebook_images(path, root)
    return path
