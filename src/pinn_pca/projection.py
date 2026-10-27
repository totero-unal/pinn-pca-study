"""Training-only snapshot PCA, reconstruction floors, and spline coefficient models."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .metrics import mean_std


def fit_snapshot_pca(x, field, fit_ids, sensor_ids=None):
    """SVD of training snapshots, optionally on a sensor subset, interpolated to the full grid."""
    sensor_ids = np.arange(x.size) if sensor_ids is None else np.asarray(sensor_ids)
    snapshots = field[np.ix_(sensor_ids, fit_ids)].T
    mean_sensors = snapshots.mean(axis=0).astype(np.float64)
    centered = snapshots - mean_sensors
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = float(np.sum(variance)) if variance.size else 1.0
    cumulative = np.cumsum(variance) / total if variance.size else np.zeros(0)
    modes_sensors = right_vectors.T.astype(np.float64)
    sensor_x = np.asarray(x[sensor_ids], dtype=np.float64)
    full_x = np.asarray(x, dtype=np.float64)
    if sensor_ids.size == x.size and np.array_equal(sensor_ids, np.arange(x.size)):
        mean = mean_sensors.astype(np.float32)
        modes = modes_sensors.astype(np.float32)
    else:
        mean = CubicSpline(sensor_x, mean_sensors)(full_x).astype(np.float32)
        interpolated = [CubicSpline(sensor_x, modes_sensors[:, index])(full_x)
                        for index in range(modes_sensors.shape[1])]
        modes = np.column_stack(interpolated).astype(np.float32) if interpolated else np.zeros((x.size, 0), np.float32)
    slopes = np.gradient(np.column_stack((mean, modes)) if modes.size else mean[:, None],
                         x, axis=0, edge_order=2).astype(np.float32)
    fields = np.column_stack((mean, modes)).astype(np.float32)
    return {
        "mean": mean,
        "modes": modes,
        "fields": fields,
        "slopes": slopes,
        "cumulative_variance": cumulative.astype(np.float64),
        "singular_values": singular_values.astype(np.float64),
    }


def truncate_basis(basis, components):
    k = int(components)
    if k < 1 or k > basis["modes"].shape[1]:
        raise ValueError(f"k={k} está fuera del rango POD disponible.")
    fields = basis["fields"][:, :k + 1]
    slopes = basis["slopes"][:, :k + 1]
    return {
        "mean": basis["mean"],
        "modes": basis["modes"][:, :k],
        "fields": fields,
        "slopes": slopes,
        "k": k,
        "retained_variance": float(basis["cumulative_variance"][k - 1]),
    }


def components_for_variance(cumulative, threshold):
    if cumulative.size == 0:
        return 1
    return int(np.searchsorted(cumulative, float(threshold), side="left") + 1)


def project_field(mean, modes, field, time_ids=None):
    """Orthogonal reconstruction of selected times in the truncated spatial basis."""
    slice_field = field if time_ids is None else field[:, time_ids]
    centered = slice_field - mean[:, None]
    coefficients = modes.T @ centered
    reconstruction = mean[:, None] + modes @ coefficients
    return reconstruction.astype(np.float32), coefficients.astype(np.float32)


def reconstruction_scores(reconstruction, truth):
    difference = reconstruction - truth
    norm = float(np.linalg.norm(truth))
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / norm) if norm else 0.0,
    }


def spline_coefficient_field(mean, modes, t_fit, coefficients_fit, t_query):
    """Cubic spline of training-time POD coefficients, evaluated at query times."""
    order = np.argsort(t_fit)
    times = np.asarray(t_fit, dtype=np.float64)[order]
    coeffs = np.asarray(coefficients_fit, dtype=np.float64)[order]
    query = np.asarray(t_query, dtype=np.float64)
    predicted = np.column_stack([
        CubicSpline(times, coeffs[:, index], extrapolate=True)(query)
        for index in range(modes.shape[1])
    ]).astype(np.float32)
    return (mean[:, None] + modes @ predicted.T).astype(np.float32)


def summarize_floor(rows, model_name, metrics=("test_rmse", "test_relative_l2")):
    selected = [row for row in rows if row["model"] == model_name]
    summary = {"seed_count": len(selected)}
    if selected and selected[0].get("k") is not None:
        summary["k"] = mean_std([row["k"] for row in selected])
        summary["retained_variance"] = mean_std([row["retained_variance"] for row in selected])
    for metric in metrics:
        values = [row[metric] for row in selected if row.get(metric) is not None]
        summary[metric] = mean_std(values) if values else None
    return summary
