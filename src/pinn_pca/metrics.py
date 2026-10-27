"""Evaluation metrics for solution accuracy and the Burgers PDE residual."""

from __future__ import annotations

import numpy as np

from .models import common_grid_residual


def score_prediction(prediction, truth, x, t, time_ids):
    truth_slice = truth[:, time_ids]
    prediction_slice = prediction[:, time_ids]
    difference = prediction_slice - truth_slice
    residual = common_grid_residual(prediction, x, t)[1:-1, :][:, time_ids]
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(truth_slice)),
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0}
