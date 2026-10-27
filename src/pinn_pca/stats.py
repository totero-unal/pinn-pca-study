"""Paired comparisons with Wilcoxon tests and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_difference_report(base_values, other_values, lower_is_better=True, bootstrap_draws=2000):
    """Summarize paired other-minus-base differences, including a signed-rank test."""
    base = np.asarray(base_values, dtype=float)
    other = np.asarray(other_values, dtype=float)
    if base.shape != other.shape:
        raise ValueError("Las series pareadas deben tener la misma longitud.")
    delta = other - base
    n = int(delta.size)
    if n == 0:
        raise ValueError("No hay diferencias pareadas que resumir.")
    better = delta < 0 if lower_is_better else delta > 0
    if np.allclose(delta, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        try:
            result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided",
                              method="auto")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = None, None
    rng = np.random.default_rng(0)
    means = np.empty(int(bootstrap_draws), dtype=float)
    for index in range(int(bootstrap_draws)):
        means[index] = delta[rng.integers(0, n, n)].mean()
    return {
        "other_minus_base_mean": float(delta.mean()),
        "other_minus_base_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "PCA_minus_base_mean": float(delta.mean()),
        "PCA_minus_base_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "other_better_count": int(np.sum(better)),
        "PCA_better_count": int(np.sum(better)),
        "n": n,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "bootstrap_mean_ci_95": [float(np.percentile(means, 2.5)),
                                 float(np.percentile(means, 97.5))],
    }


def paired_reports_by_model(rows, baseline_name, metrics, lower_is_better=None):
    """Build Wilcoxon/bootstrap reports for every non-baseline model."""
    better = lower_is_better or {metric: True for metric in metrics}
    baseline = {row["seed"]: row for row in rows if row["model"] == baseline_name}
    reports = {}
    models = list(dict.fromkeys(row["model"] for row in rows if row["model"] != baseline_name))
    for model in models:
        model_rows = {row["seed"]: row for row in rows if row["model"] == model}
        seeds = sorted(seed for seed in baseline if seed in model_rows)
        reports[model] = {}
        for metric in metrics:
            reports[model][metric] = paired_difference_report(
                [baseline[seed][metric] for seed in seeds],
                [model_rows[seed][metric] for seed in seeds],
                lower_is_better=better.get(metric, True),
            )
    return reports
