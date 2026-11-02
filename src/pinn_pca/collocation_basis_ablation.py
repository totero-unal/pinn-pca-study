"""Controlled residual-basis ablation: only the basis construction changes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr, svd

from .collocation import _run_collocation_seed, _save_csv, _evaluate, _predict_grid, _qmc_points

LABELS = {"sobol": "Sobol", "pod_qr": "POD", "pca_qr": "PCA centrada", "pca_mean_qr": "PCA + media"}


def fixed_basis(history, method, dimension):
    """Equal orthonormal dimension; mean consumes one slot, never an extra one.

    For PCA+mean retain the mean first, then orthogonalize successive PCA modes.
    Skip linearly dependent modes. Fail rather than silently pad a deficient basis.
    """
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 2 or not np.isfinite(h).all() or not 1 <= dimension <= min(h.shape):
        raise ValueError("Invalid history or fixed dimension")
    mean = h.mean(axis=1)
    centered = h - mean[:, None]
    if method not in ("pod_qr", "pca_qr", "pca_mean_qr"):
        raise ValueError(method)
    matrix = h if method == "pod_qr" else centered
    u, s, _ = svd(matrix, full_matrices=False)
    # Snapshots are float32 measurements: do not count roundoff as a real mode.
    tolerance = np.finfo(np.float32).eps * s[0] if len(s) else 0.
    rank = int(np.sum(s > tolerance))
    vectors = []
    if method == "pca_mean_qr":
        norm = np.linalg.norm(mean)
        if norm <= np.finfo(float).eps * max(1., np.linalg.norm(h)):
            raise ValueError("Mean is zero; equal mean-inclusive dimension cannot be interpreted")
        vectors.append(mean / norm)
    for vector in u[:, :rank].T:
        if len(vectors) == dimension:
            break
        direction = vector.copy()
        # Reorthogonalization protects near-overlap between mean and PCA modes.
        for _ in range(2):
            for previous in vectors:
                direction -= previous * np.dot(previous, direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            vectors.append(direction / norm)
        if len(vectors) == dimension:
            break
    if len(vectors) != dimension:
        raise ValueError(f"Insufficient numerical rank for {method}: requested {dimension}, obtained {len(vectors)}")
    return np.column_stack(vectors)


def select_fixed_basis(history, method, dimension, count):
    if not dimension <= count <= len(history):
        raise ValueError("Guided point count must cover the basis and fit the candidates")
    basis = fixed_basis(history, method, dimension)
    return select_from_basis(history, basis, count)


def select_from_basis(history, basis, count):
    """Shared QR/log-det design for fixed and adaptive orthonormal bases."""
    dimension = basis.shape[1]
    if not 0 < dimension <= count <= len(history):
        raise ValueError("A nonzero basis fitting the guided point budget is required")
    _, _, pivots = qr(basis.T, pivoting=True, mode="economic")
    selected = list(pivots[:dimension])
    information = basis[selected].T @ basis[selected] + 1e-10 * np.eye(dimension)
    inverse = np.linalg.inv(information)
    available = np.ones(len(history), dtype=bool)
    available[selected] = False
    while len(selected) < count:
        scores = np.einsum("ij,ij->i", basis @ inverse, basis)
        scores[~available] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        available[index] = False
        direction = inverse @ basis[index]
        inverse -= np.outer(direction, direction) / (1 + basis[index] @ direction)
        inverse = (inverse + inverse.T) * .5
    h = np.asarray(history, dtype=float)
    mean = h.mean(axis=1)
    centered = h - mean[:, None]
    def captured(matrix):
        total = np.linalg.norm(matrix) ** 2
        return float(np.linalg.norm(basis.T @ matrix) ** 2 / total) if total else 0.
    return np.asarray(selected), dict(actual_dimension=basis.shape[1],
        centered_variance_captured=captured(centered), raw_energy_captured=captured(h),
        mean_projection_fraction=captured(mean), qr_initial_count=dimension)


class BasisRecorder:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.metrics, self.active, self.fields = [], {}, {}

    def begin_seed(self, **context):
        self.c = context

    def selection(self, method, step, active, ids, modes):
        guided = 0 if method == "sobol" else self.config["guided_points"]
        assert len(active) == self.config["batch_size"] and len(ids) == guided
        assert len(set(map(tuple, active))) == len(active)
        assert modes == (0 if method == "sobol" else self.config["basis_dimension"])
        c = self.c
        bounds = ((float(c["x"].min()), float(c["x"].max())), (float(c["t"].min()), float(c["t"].max())))
        background = _qmc_points(len(active)-guided, c["seed"] + step + 104729, *bounds)
        np.testing.assert_array_equal(active[guided:], background)
        self.active[f"{method}_{step}"] = active.copy()

    def checkpoint(self, method, step, params, active, selection_step):
        c = self.c
        if step in self.config["test_checkpoints"]:
            self.metrics.append(dict(seed=c["seed"], method=method, step=step,
                **_evaluate(params, c["x"], c["t"], c["u"], c["test_ids"], c["independent"])))
        if c["seed"] == self.config["seeds"][0] and step == self.config["warmup_steps"] + self.config["train_steps"]:
            self.fields[method] = np.asarray(_predict_grid(params, c["x"], c["t"]))

    def end_seed(self):
        _save_csv(self.output / "budget_metrics.csv", self.metrics)
        np.savez_compressed(self.output / "active_history.npz", **self.active)
        if self.fields:
            np.savez_compressed(self.output / "fields.npz", x=self.c["x"], t=self.c["t"], reference=self.c["u"], **self.fields)
        (self.output / "audit.json").write_text(json.dumps(dict(passed=True,
            checked_selections=len(self.active), fixed_dimension=self.config["basis_dimension"],
            fixed_guided_points=self.config["guided_points"], common_sobol_background=True)), encoding="utf-8")


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(config, root, report_only=False):
    root = Path(root).resolve()
    output = root / "pca_collocation/results/burgers/basis_ablation" / config["run_id"]
    specs = {key: dict(base=key, label=label, **({} if key == "sobol" else
        dict(fixed_basis_dimension=config["basis_dimension"], oversample_count=config["guided_points"]))) for key, label in LABELS.items()}
    saved = dict(config, method_specs=specs, common_selection_seed=True, skip_seed_plots=True, basis_ablation=True)
    if not 1 < config["basis_dimension"] < config["history_snapshots"]:
        raise ValueError("Centered histories need fewer dimensions than snapshots")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Distinct nonempty seeds required")
    if not config["basis_dimension"] <= config["guided_points"] <= min(config["batch_size"],config["candidate_count"]):
        raise ValueError("Invalid guided-point budget")
    end = config["warmup_steps"] + config["train_steps"]
    if config["train_steps"] % config["update_every"] or config["warmup_steps"] < config["history_snapshots"]:
        raise ValueError("Require full training intervals and distinct warmup snapshots")
    if config["test_checkpoints"] != sorted(set(config["test_checkpoints"])) or config["test_checkpoints"][0] != config["warmup_steps"] or config["test_checkpoints"][-1] != end:
        raise ValueError("Ordered test budgets must include warmup and final model")
    if any((step-config["warmup_steps"]) % config["update_every"] for step in config["test_checkpoints"]):
        raise ValueError("Test budgets must align with evaluation intervals")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != saved:
        raise ValueError("Use a new run_id for a changed experiment")
    if report_only and not config_path.exists():
        raise ValueError("No completed experiment")
    config_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    for seed in config["seeds"]:
        folder = output / f"seed_{seed}"
        if not report_only and not (folder / "audit.json").exists():
            print(f"Starting basis ablation seed {seed}", flush=True)
            _run_collocation_seed(dict(saved, seed=seed), root, folder, observer=BasisRecorder(config, folder))
        if not (folder / "audit.json").exists():
            raise ValueError(f"Incomplete seed {seed}")
        print(f"Verified seed {seed}", flush=True)
    from .collocation_basis_report import report
    report(config, root, output)
    return output
