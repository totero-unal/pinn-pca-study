"""Two-stage controlled rank study, with audited reuse of dimension-four runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from .collocation import _run_collocation_seed, _save_csv
from .collocation_basis_ablation import LABELS, BasisRecorder, select_from_basis, read_rows


class RankUnavailable(ValueError):
    """Requested variance cannot be resolved by the available residual history."""


def adaptive_basis(history, method, threshold):
    h = np.asarray(history, dtype=float)
    if h.ndim != 2 or not np.isfinite(h).all() or min(h.shape) < 1 or not 0 < threshold <= 1:
        raise ValueError("Invalid history or variance threshold")
    if method not in ("pod_qr", "pca_qr", "pca_mean_qr"):
        raise ValueError(method)
    mean = h.mean(axis=1)
    centered = h - mean[:, None]
    matrix = h if method == "pod_qr" else centered
    u, s, _ = svd(matrix, full_matrices=False)
    energy = s*s
    numerical_rank = int(np.sum(s > np.finfo(np.float32).eps * s[0])) if s.size else 0
    if not energy.sum():
        k, retained = 0, 1.
    else:
        cumulative = np.cumsum(energy) / energy.sum()
        k = min(len(s), int(np.searchsorted(cumulative, threshold)) + 1)
        if k > numerical_rank:
            raise RankUnavailable(f"Threshold {threshold} requires {k} modes; numerical rank {numerical_rank}")
        retained = float(cumulative[k-1])
    vectors = []
    if method == "pca_mean_qr" and np.linalg.norm(mean) > np.finfo(float).eps * max(1.,np.linalg.norm(h)):
        vectors.append(mean / np.linalg.norm(mean))
    for vector in u[:,:k].T:
        direction = vector.copy()
        for _ in range(2):
            for previous in vectors:
                direction -= previous * np.dot(previous, direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            vectors.append(direction / norm)
    if not vectors:
        raise RankUnavailable("Zero informative rank; cannot claim guided point selection")
    basis = np.column_stack(vectors)
    return basis, dict(threshold=threshold, spectral_modes=k, numerical_rank=numerical_rank,
        spectral_fraction=retained, threshold_quantity="raw_energy" if method=="pod_qr" else "centered_variance")


def select_adaptive_basis(history, method, threshold, count):
    basis, info = adaptive_basis(history, method, threshold)
    ids, captured = select_from_basis(history, basis, count)
    info.update(captured)
    quantity = "raw_energy_captured" if method == "pod_qr" else "centered_variance_captured"
    if info[quantity] + 1e-10 < threshold:
        raise RankUnavailable("Orthonormalization failed to preserve requested variance")
    return ids, info


class RankRecorder(BasisRecorder):
    def __init__(self, config, output, variant):
        super().__init__(config, output)
        self.variant = variant
        self.details = []

    def selection_details(self, method, step, history, info):
        if info:
            self.details.append(dict(step=step, **info))

    def selection(self, method, step, active, ids, modes):
        # Reuse the exact count, uniqueness and common-background audit.
        if self.variant["stage"] == "adaptive":
            self.config["basis_dimension"] = modes
        super().selection(method, step, active, ids, modes)

    def end_seed(self):
        super().end_seed()
        _save_csv(self.output / "rank_history.csv", self.details)
        audit_path = self.output / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit.pop("fixed_dimension",None)
        audit.update(stage=self.variant["stage"], requested_parameter=self.variant["parameter"],
                     min_dimension=min(r["actual_dimension"] for r in self.details),
                     max_dimension=max(r["actual_dimension"] for r in self.details))
        audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")


INVARIANTS = ("data_path", "seeds", "width", "learning_rate", "warmup_steps", "train_steps", "batch_size",
    "update_every", "candidate_count", "history_snapshots", "independent_count", "initial_condition_points",
    "initial_condition_strategy", "test_time_count", "validation_time_count", "guided_points", "test_checkpoints")


def variants(config):
    result = [dict(id="sobol", stage="control", method="sobol", parameter=0)]
    for stage, values in (("fixed",config["dimensions"]),("adaptive",config["thresholds"])):
        for value in values:
            for method in list(LABELS)[1:]:
                token = str(value).replace(".","p")
                result.append(dict(id=f"{stage}_{token}_{method}",stage=stage,method=method,parameter=value))
    return result


def run(config, root, report_only=False):
    root = Path(root).resolve()
    source = root / config["reuse_source"]
    original = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if not original["common_selection_seed"] or original["basis_dimension"] != 4:
        raise ValueError("Reference must be the paired dimension-four experiment")
    for key in INVARIANTS:
        if config[key] != original[key]:
            raise ValueError(f"Reusing results requires unchanged {key}")
    if config["dimensions"] != [2,4,8,12] or config["thresholds"] != [.95,.99,.999]:
        raise ValueError("This preregistered study uses dimensions 2/4/8/12 and thresholds .95/.99/.999")
    output = root / "pca_collocation/results/burgers/rank_study" / config["run_id"]
    output.mkdir(parents=True,exist_ok=True)
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Use another run_id for different settings")
    path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    manifest = []
    for variant in variants(config):
        for seed in config["seeds"]:
            reused = variant["stage"]=="control" or (variant["stage"]=="fixed" and variant["parameter"]==4)
            folder = source / f"seed_{seed}" if reused else output / variant["id"] / f"seed_{seed}"
            failure_path = folder / "infeasible.json"
            if not reused and not report_only and not (folder/"audit.json").exists() and not failure_path.exists():
                print(f"RUN {variant['id']} seed={seed}",flush=True)
                method = variant["method"]
                spec = dict(base=method,label=LABELS[method],oversample_count=config["guided_points"])
                key = "fixed_basis_dimension" if variant["stage"]=="fixed" else "adaptive_basis_threshold"
                spec[key] = variant["parameter"]
                run_config = dict(config,seed=seed,basis_dimension=variant["parameter"] if variant["stage"]=="fixed" else 0,
                    method_specs={method:spec}, common_selection_seed=True, skip_seed_plots=True,basis_ablation=True)
                recorder = RankRecorder(dict(run_config), folder, variant)
                try:
                    _run_collocation_seed(run_config, root, folder, observer=recorder)
                except ValueError as exc:
                    if not isinstance(exc,RankUnavailable) and "Insufficient numerical rank" not in str(exc):
                        raise
                    failure_path.write_text(json.dumps(dict(reason=str(exc),last_successful_selection=recorder.details[-1]["step"] if recorder.details else None),indent=2),encoding="utf-8")
                    print(f"INFEASIBLE {variant['id']} seed={seed}: {exc}",flush=True)
                else:
                    print(f"DONE {variant['id']} seed={seed}",flush=True)
            if not (folder/"audit.json").exists() and not failure_path.exists():
                raise ValueError(f"Missing result {variant['id']} seed={seed}")
            manifest.append(dict(**variant,seed=seed,reused=reused,
                status="infeasible" if failure_path.exists() else "complete",path=folder.relative_to(root).as_posix()))
            (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    from .collocation_rank_report import report
    report(config,root,output)
    return output
