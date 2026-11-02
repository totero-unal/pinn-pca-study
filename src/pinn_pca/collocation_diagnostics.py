"""Observe residual selection without changing the Burgers training objective."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .collocation import _predict_grid, _qmc_points, _residual_batch, _residual_field, _run_collocation_seed, _save_csv
from .models import burgers_solution

COMPONENTS = ("data", "physics", "initial")
METHOD_SPECS = {
    "sobol": {"base": "sobol", "label": "Sobol"},
    "pod_qr_32": {"base": "pod_qr", "label": "POD: 32 puntos", "oversample_count": 32},
    "pca_mean_qr_64": {"base": "pca_mean_qr", "label": "PCA + media: 64 puntos", "oversample_count": 64},
}


def loss_components(params, batch):
    """Exactly the three unit-weight terms of baseline_ad_loss; no BC penalty."""
    data_x, data_t, data_u, residual_x, residual_t, initial_x, initial_u = batch
    prediction = jax.vmap(lambda xx, tt: burgers_solution(params, xx, tt))(data_x, data_t)
    residual = _residual_batch(params, jnp.stack((residual_x, residual_t), axis=1))
    initial = jax.vmap(lambda xx: burgers_solution(params, xx, jnp.asarray(0.0, dtype=xx.dtype)))(initial_x)
    return jnp.stack((jnp.mean((prediction - data_u) ** 2), jnp.mean(residual ** 2), jnp.mean((initial - initial_u) ** 2)))


@jax.jit
def gradient_geometry(params, batch):
    values = loss_components(params, batch)
    jacobian = jax.jacrev(loss_components, argnums=0)(params, batch)
    matrix = jnp.concatenate([leaf.reshape((3, -1)) for leaf in jax.tree.leaves(jacobian)], axis=1)
    gram = matrix @ matrix.T
    norms = jnp.sqrt(jnp.maximum(jnp.diag(gram), 0.0))

    def cosine(i, j):
        denominator = norms[i] * norms[j]
        return jnp.where(denominator > 0, gram[i, j] / denominator, jnp.nan)

    supervised_norm = jnp.linalg.norm(matrix[0] + matrix[2])
    total_norm = jnp.linalg.norm(jnp.sum(matrix, axis=0))
    return {
        **{f"loss_{name}": values[i] for i, name in enumerate(COMPONENTS)},
        **{f"grad_{name}": norms[i] for i, name in enumerate(COMPONENTS)},
        "loss_total": jnp.sum(values), "grad_total": total_norm,
        "grad_data_initial": supervised_norm,
        "cos_physics_data": cosine(1, 0), "cos_physics_initial": cosine(1, 2),
        "cos_data_initial": cosine(0, 2),
        "cos_physics_data_initial": jnp.where(norms[1] * supervised_norm > 0,
            jnp.dot(matrix[1], matrix[0] + matrix[2]) / (norms[1] * supervised_norm), jnp.nan),
        "physics_to_data_initial_grad_ratio": jnp.where(supervised_norm > 0, norms[1] / supervised_norm, jnp.nan),
        "gradient_cancellation_ratio": jnp.where(jnp.sum(norms) > 0, total_norm / jnp.sum(norms), jnp.nan),
    }


def front_mask(points, config):
    """Predefined central late-time window for diagnosis, never for selection."""
    points = np.asarray(points)
    return (np.abs(points[..., 0]) <= config["front_abs_x_max"]) & (points[..., 1] >= config["front_time_min"])


def coverage_statistics(active, selected_count, independent, bounds, config):
    lower = np.asarray([b[0] for b in bounds])
    span = np.asarray([b[1] - b[0] for b in bounds])
    normalized = (active - lower) / span
    bins = int(config["coverage_bins"])
    cell_ids = np.clip((normalized * bins).astype(int), 0, bins - 1)
    distance, _ = cKDTree(normalized).query((independent - lower) / span)
    mask = front_mask(active, config)
    return {
        "front_fraction": float(mask.mean()),
        "basis_front_fraction": float(mask[:selected_count].mean()) if selected_count else None,
        "sobol_front_fraction": float(mask[selected_count:].mean()) if len(active) > selected_count else None,
        "occupied_cell_fraction": float(len(np.unique(cell_ids, axis=0)) / bins ** 2),
        "independent_nearest_distance_mean": float(distance.mean()),
        "independent_nearest_distance_q95": float(np.quantile(distance, .95)),
    }


class DiagnosticRecorder:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.probes, self.points, self.coverage, self.regions = [], [], [], []
        self.fields, self.parameters = {}, {}

    def begin_seed(self, **context):
        self.context = context
        self.seed = context["seed"]
        self.x, self.t, self.u = context["x"], context["t"], context["u"]
        self.bounds = ((float(self.x.min()), float(self.x.max())), (float(self.t.min()), float(self.t.max())))
        self.probe = context["probe_batch"]
        common = _qmc_points(len(self.probe[0]), self.seed + 501, *self.bounds)
        self.common_batch = (*self.probe[:3], jnp.asarray(common[:, 0]), jnp.asarray(common[:, 1]), *self.probe[5:])
        xi = np.unique(np.linspace(0, len(self.x) - 1, min(len(self.x), self.config["grid_x_count"]), dtype=int))
        ti = np.unique(np.linspace(0, len(self.t) - 1, min(len(self.t), self.config["grid_t_count"]), dtype=int))
        self.gx, self.gt = self.x[xi], self.t[ti]
        xx, tt = np.meshgrid(self.gx, self.gt, indexing="ij")
        self.grid_points = np.column_stack((xx.ravel(), tt.ravel()))
        self.fields.update(x=self.gx, t=self.gt, reference=self.u[np.ix_(xi, ti)],
                           initial_indices=context["initial_ids"],
                           initial_x=self.x[context["initial_ids"]], initial_u=self.u[context["initial_ids"], 0],
                           fit_time_indices=context["fit_ids"], validation_time_indices=context["validation_ids"],
                           test_time_indices=context["test_ids"], common_probe_points=common,
                           probe_data_x=np.asarray(self.probe[0]), probe_data_t=np.asarray(self.probe[1]),
                           probe_data_u=np.asarray(self.probe[2]))
        initial_x = self.x[context["initial_ids"]]
        self.metadata = {
            "seed": self.seed, "initial_points": len(initial_x), "spatial_grid_points": len(self.x),
            "initial_condition_strategy": self.config.get("initial_condition_strategy", "left_prefix"),
            "initial_indices": context["initial_ids"].tolist(),
            "initial_x_min": float(initial_x.min()), "initial_x_max": float(initial_x.max()),
            "domain_x_min": float(self.x.min()), "domain_x_max": float(self.x.max()),
            "initial_span_fraction": float(np.ptp(initial_x) / np.ptp(self.x)),
            "boundary_penalty_in_training": False, "loss_weights": {name: 1.0 for name in COMPONENTS},
            "fixed_probe_supervised_points": len(self.probe[0]),
            "probe_rule": "Frozen supervised batch from fit times only; fixed common Sobol residual probe with seed+501. Active probe replaces only residual coordinates with the current training set.",
            "front_window": {k: self.config[k] for k in ("front_abs_x_max", "front_time_min")},
        }

    def selection(self, method, step, active, ids, modes):
        count = len(ids)
        self.coverage.append({"seed": self.seed, "method": method, "selection_step": step,
                              "active_points": len(active), "basis_points": count, "retained_modes": modes,
                              **coverage_statistics(active, count, self.context["independent"], self.bounds, self.config)})
        for i, point in enumerate(active):
            self.points.append({"seed": self.seed, "method": method, "selection_step": step,
                                "point_index": i, "x": float(point[0]), "t": float(point[1]),
                                "source": "basis" if i < count else "sobol",
                                "candidate_index": int(ids[i]) if i < count else None})

    def checkpoint(self, method, step, params, active, selection_step):
        active_batch = (*self.probe[:3], jnp.asarray(active[:, 0]), jnp.asarray(active[:, 1]), *self.probe[5:])
        for name, batch in (("common", self.common_batch), ("active", active_batch)):
            values = gradient_geometry(params, batch)
            self.probes.append({"seed": self.seed, "method": method, "step": step,
                                "probe": name, "active_selection_step": selection_step,
                                **{key: float(value) for key, value in values.items()}})
        if step not in self.config["field_checkpoints"]:
            return
        key = f"{method}__{step}"
        prediction = np.asarray(_predict_grid(params, jnp.asarray(self.gx), jnp.asarray(self.gt)))
        residual = _residual_field(params, self.grid_points).reshape(prediction.shape)
        self.fields[key + "__u"] = prediction
        self.fields[key + "__r"] = residual
        self.fields[key + "__error"] = prediction - self.fields["reference"]
        for i, (weight, bias) in enumerate(params):
            self.parameters[f"{key}__layer_{i}_weight"] = np.asarray(weight)
            self.parameters[f"{key}__layer_{i}_bias"] = np.asarray(bias)
        if step not in self.config.get("regional_test_checkpoints", self.config["field_checkpoints"]):
            return
        test_ids = self.context["test_ids"]
        test_prediction = np.asarray(_predict_grid(params, jnp.asarray(self.x), jnp.asarray(self.t[test_ids])))
        error = test_prediction - self.u[:, test_ids]
        xx, tt = np.meshgrid(self.x, self.t[test_ids], indexing="ij")
        central = front_mask(np.stack((xx, tt), axis=-1), self.config)
        for region, mask in (("all_test", np.ones(error.shape, dtype=bool)), ("central_late_test", central), ("outside_window_test", ~central)):
            self.regions.append({"seed": self.seed, "method": method, "step": step, "region": region,
                                 "n_points": int(mask.sum()),
                                 "rmse": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None})

    def end_seed(self):
        self.output.mkdir(parents=True, exist_ok=True)
        for name, rows in (("loss_gradient_probes", self.probes), ("active_points", self.points),
                           ("coverage", self.coverage), ("regional_test_errors", self.regions)):
            _save_csv(self.output / f"{name}.csv", rows)
        np.savez_compressed(self.output / "fields.npz", **self.fields)
        np.savez_compressed(self.output / "parameters.npz", **self.parameters)
        (self.output / "diagnostic_protocol.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")


def run_diagnostics(config, project_root):
    root = Path(project_root).resolve()
    parent = root / "pca_collocation/results/burgers/collocation_diagnostics"
    output = parent / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    reference = root / "pca_collocation/results/burgers/collocation_selection" / config["reference_run_id"]
    previous_config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    for key in ("seeds", "width", "learning_rate", "warmup_steps", "train_steps", "batch_size", "update_every", "candidate_count", "history_snapshots", "initial_condition_points", "test_time_count", "validation_time_count", "energy_threshold", "max_modes"):
        if config[key] != previous_config[key]:
            raise ValueError(f"Diagnostic replay requires unchanged {key}.")
    last = config["warmup_steps"] + config["train_steps"]
    if config["representative_seed"] not in config["seeds"] or any(s < config["warmup_steps"] or s > last or (s - config["warmup_steps"]) % config["update_every"] for s in config["field_checkpoints"]):
        raise ValueError("Representative seed and field checkpoints must belong to the configured run.")
    saved = dict(config, method_specs=METHOD_SPECS, common_selection_seed=True, skip_seed_plots=True)
    (output / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    all_differences = []
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        recorder = DiagnosticRecorder(config, seed_dir)
        _run_collocation_seed(dict(saved, seed=seed), root, seed_dir, observer=recorder)
        with (reference / f"seed_{seed}/metrics.csv").open(encoding="utf-8", newline="") as file:
            old = {r["method"]: r for r in csv.DictReader(file)}
        with (seed_dir / "metrics.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for metric in ("test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms"):
                difference = float(row[metric]) - float(old[row["method"]][metric])
                all_differences.append({"seed": seed, "method": row["method"], "metric": metric,
                                        "reference_value": float(old[row["method"]][metric]),
                                        "diagnostic_value": float(row[metric]), "difference": difference})
                if abs(difference) > config["replay_tolerance"]:
                    raise ValueError(f"Instrumented replay changed {metric} for {row['method']}, seed {seed}: {difference}")
        print(f"Diagnostic seed {seed} complete; reference metrics reproduced.", flush=True)
    _save_csv(output / "replay_comparison.csv", all_differences)
    from .collocation_diagnostic_report import build_diagnostic_report
    build_diagnostic_report(config, root, output)
    return output
