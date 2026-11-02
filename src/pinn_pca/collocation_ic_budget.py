"""Extend the distributed-IC experiment with an audited identical prefix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .collocation import _evaluate, _run_collocation_seed, _save_csv
from .collocation_diagnostics import DiagnosticRecorder, METHOD_SPECS
from .collocation_diagnostic_report import read_csv
from .collocation_ic_control import INVARIANTS, METRICS


def validate_budget(config, baseline):
    keys = [k for k in INVARIANTS if k not in ("train_steps", "field_checkpoints", "selection_plot_steps")]
    keys.append("initial_condition_strategy")
    for key in keys:
        if config[key] != baseline[key]:
            raise ValueError(f"Budget extension requires unchanged {key}.")
    if config["initial_condition_strategy"] != "uniform_domain":
        raise ValueError("Both runs must use distributed IC nodes.")
    if baseline["method_specs"] != METHOD_SPECS or not baseline["common_selection_seed"]:
        raise ValueError("Reference selectors and selection seeding must match.")
    warm, frequency = config["warmup_steps"], config["update_every"]
    before_end, after_end = warm + baseline["train_steps"], warm + config["train_steps"]
    if after_end <= before_end:
        raise ValueError("Extended budget must exceed the reference budget.")
    checkpoints = config["test_checkpoint_steps"]
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[0] != before_end or checkpoints[-1] != after_end:
        raise ValueError("Test checkpoints must be distinct, ordered and include both budget endpoints.")
    if any(step < before_end or step > after_end or (step - warm) % frequency for step in checkpoints):
        raise ValueError("Test checkpoints must align with the configured training evaluations.")
    if config["regional_test_checkpoints"] != checkpoints:
        raise ValueError("Regional test evaluation must use the fixed test checkpoints.")
    if not set([warm, *checkpoints]).issubset(config["field_checkpoints"]):
        raise ValueError("Field snapshots must cover warmup and every fixed test checkpoint.")
    if any(step < warm or step >= after_end or (step - warm) % frequency for step in config["selection_plot_steps"]):
        raise ValueError("Selection plot steps must be valid selection updates.")
    if config["trend_window_steps"] <= 0 or config["trend_window_steps"] % frequency or after_end - config["trend_window_steps"] < warm:
        raise ValueError("Terminal validation window must align with evaluation checkpoints.")
    return {key: config[key] for key in keys}


def compare_prefix_rows(old, new, tolerance):
    """Compare numerical training monitors while ignoring instrumented elapsed time."""
    lookup = {(r["method"], r["step"]): r for r in new}
    differences = []
    for row in old:
        key = (row["method"], row["step"])
        if key not in lookup:
            raise ValueError(f"Missing prefix checkpoint: {key}")
        current = lookup[key]
        for metric in ("validation_rmse", "independent_residual_rms", "initial_condition_rmse", "boundary_rms", "training_loss"):
            if row.get(metric) in ("", None):
                continue
            difference = float(current[metric]) - float(row[metric])
            if not np.isfinite(difference) or abs(difference) > tolerance:
                raise ValueError(f"Changed prefix checkpoint {key}: {metric}, delta={difference}")
            differences.append({"method": row["method"], "step": row["step"], "metric": metric, "difference": difference})
    return differences


class BudgetRecorder(DiagnosticRecorder):
    def __init__(self, config, output):
        super().__init__(config, output)
        self.budget_metrics = []

    def checkpoint(self, method, step, params, active, selection_step):
        super().checkpoint(method, step, params, active, selection_step)
        if step in self.config["test_checkpoint_steps"]:
            metrics = _evaluate(params, self.x, self.t, self.u, self.context["test_ids"], self.context["independent"])
            self.budget_metrics.append({"seed": self.seed, "method": method, "step": step,
                                        "continuation_steps": step - self.config["warmup_steps"], **metrics})

    def end_seed(self):
        super().end_seed()
        _save_csv(self.output / "budget_metrics.csv", self.budget_metrics)


def experiment_paths(config, root):
    group = config.get("reference_result_group", "ic_coverage_control")
    if group not in ("ic_coverage_control", "ic_long_budget"):
        raise ValueError("Unknown reference result group.")
    baseline = root / "pca_collocation/results/burgers" / group / config["reference_ic_run_id"]
    output = root / "pca_collocation/results/burgers/ic_long_budget" / config["run_id"]
    if baseline.resolve() == output.resolve():
        raise ValueError("Budget output must not overwrite its reference run.")
    return baseline, output


def audit_seed(config, baseline, output):
    checked = []
    with np.load(baseline / "fields.npz", allow_pickle=False) as old, np.load(output / "fields.npz", allow_pickle=False) as new:
        for key in ("x", "t", "reference", "initial_indices", "initial_x", "initial_u", "fit_time_indices",
                    "validation_time_indices", "test_time_indices", "common_probe_points", "probe_data_x", "probe_data_t", "probe_data_u"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"Budget extension changed fixed array {key}.")
            checked.append(key)
    with np.load(baseline / "selection_snapshot.npz", allow_pickle=False) as old, np.load(output / "selection_snapshot.npz", allow_pickle=False) as new:
        for key in old.files:
            np.testing.assert_allclose(new[key], old[key], rtol=0, atol=config["replay_tolerance"], err_msg=key)
    old_points = read_csv(baseline / "active_points.csv")
    old_end = max(int(row["step"]) for row in read_csv(baseline / "validation_trajectories.csv"))
    new_points = [row for row in read_csv(output / "active_points.csv") if int(row["selection_step"]) < old_end]
    if old_points != new_points:
        raise ValueError("The active-point sequence changed inside the common training prefix.")
    differences = compare_prefix_rows(read_csv(baseline / "validation_trajectories.csv"), read_csv(output / "validation_trajectories.csv"), config["replay_tolerance"])
    reference = {row["method"]: row for row in read_csv(baseline / "metrics.csv")}
    for row in read_csv(output / "budget_metrics.csv"):
        if int(row["step"]) == old_end:
            for metric in METRICS:
                delta = float(row[metric]) - float(reference[row["method"]][metric])
                if abs(delta) > config["replay_tolerance"]:
                    raise ValueError(f"Reference test endpoint changed: {row['method']}, {metric}.")
                differences.append({"method": row["method"], "step": str(old_end), "metric": "test_endpoint_" + metric, "difference": delta})
    return checked + ["warmup_history_and_initial_selection", "identical_active_point_prefix", "reference_test_endpoint"], differences


def run_long_budget(config, project_root):
    root = Path(project_root).resolve()
    baseline, output = experiment_paths(config, root)
    validate_budget(config, json.loads((baseline / "config.json").read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=True)
    saved = dict(config, method_specs=METHOD_SPECS, common_selection_seed=True, skip_seed_plots=True)
    (output / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        _run_collocation_seed(dict(saved, seed=seed), root, seed_dir, observer=BudgetRecorder(config, seed_dir))
        audit_seed(config, baseline / f"seed_{seed}", seed_dir)
        print(f"{config['train_steps']}-step continuation: seed {seed} completed; common prefix verified.", flush=True)
    from .collocation_ic_budget_report import build_budget_report
    build_budget_report(config, root)
    return output
