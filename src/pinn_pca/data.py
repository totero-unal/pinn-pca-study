"""Burgers data loading and paired random sampling."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat


def load_burgers_data(path: str | Path):
    raw = loadmat(path)
    x = np.asarray(raw["x"], dtype=np.float32).reshape(-1)
    t = np.asarray(raw["t"], dtype=np.float32).reshape(-1)
    u = np.asarray(raw["usol"], dtype=np.float32)
    if u.shape != (x.size, t.size):
        raise ValueError(f"usol tiene forma {u.shape}; se esperaba {(x.size, t.size)}")
    return x, t, u


def split_times(n_times, seed, test_count=20, validation_count=8):
    """Reserve t=0 for the initial condition and pair splits across methods."""
    rng = np.random.default_rng(seed)
    future = rng.permutation(np.arange(1, n_times))
    if test_count + validation_count >= future.size:
        raise ValueError("La cantidad de tiempos de prueba y validación deja vacíos los datos de ajuste.")
    test_ids = np.sort(future[:test_count])
    validation_ids = np.sort(future[test_count:test_count + validation_count])
    fit_ids = np.sort(np.concatenate(([0], future[test_count + validation_count:])))
    return fit_ids, validation_ids, test_ids


def make_batches(x, t, u, fit_ids, ic_ids, steps, batch_size, seed):
    """Generate one reusable sequence of minibatches shared by both methods."""
    rng = np.random.default_rng(seed)
    batches = []
    ic_x = jnp.asarray(x[ic_ids])
    ic_u = jnp.asarray(u[ic_ids, 0])
    ic_index = jnp.asarray(ic_ids, dtype=jnp.int32)
    for _ in range(steps):
        data_x_index = rng.integers(0, x.size, batch_size)
        data_t_index = rng.choice(fit_ids, batch_size)
        residual_x_index = rng.integers(0, x.size, batch_size)
        residual_t = rng.uniform(float(t.min()), float(t.max()), batch_size).astype(np.float32)
        batches.append((
            jnp.asarray(data_x_index, dtype=jnp.int32),
            jnp.asarray(x[data_x_index]),
            jnp.asarray(t[data_t_index]),
            jnp.asarray(u[data_x_index, data_t_index]),
            jnp.asarray(residual_x_index, dtype=jnp.int32),
            jnp.asarray(x[residual_x_index]),
            jnp.asarray(residual_t),
            ic_index,
            ic_x,
            ic_u,
        ))
    return batches
