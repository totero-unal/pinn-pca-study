"""Shared neural-network primitives and Burgers residual operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


NU = 0.01 / np.pi


def init_mlp(key, layer_sizes):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for subkey, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        bound = np.sqrt(6.0 / (n_in + n_out))
        weight = jax.random.uniform(subkey, (n_in, n_out), minval=-bound, maxval=bound)
        bias = jnp.zeros((n_out,), dtype=jnp.float32)
        params.append((weight, bias))
    return params


def mlp(params, inputs):
    hidden = inputs
    for layer_index, (weight, bias) in enumerate(params):
        hidden = hidden @ weight + bias
        if layer_index < len(params) - 1:
            hidden = jnp.tanh(hidden)
    return hidden


def parameter_count(layer_sizes):
    return int(sum((n_in + 1) * n_out for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])))


def burgers_solution(params, x, t):
    return mlp(params, jnp.stack((x, t)))[0]


def burgers_residual(params, x, t):
    value_fn = lambda xx, tt: burgers_solution(params, xx, tt)
    value = value_fn(x, t)
    ux = jax.grad(value_fn, argnums=0)(x, t)
    ut = jax.grad(value_fn, argnums=1)(x, t)
    uxx = jax.grad(lambda xx: jax.grad(value_fn, argnums=0)(xx, t))(x)
    return ut + value * ux - NU * uxx


def spatial_pca_basis(x, u, fit_ids, components):
    """Fit the spatial PCA basis only on training-time snapshots."""
    snapshots = u[:, fit_ids].T
    mean = snapshots.mean(axis=0).astype(np.float32)
    centered = snapshots - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    k = min(components, right_vectors.shape[0])
    modes = right_vectors[:k].T.astype(np.float32)
    retained_variance = float(np.sum(singular_values[:k] ** 2) / np.sum(singular_values ** 2))
    mean_x = np.gradient(mean, x, edge_order=2).astype(np.float32)
    mean_xx = np.gradient(mean_x, x, edge_order=2).astype(np.float32)
    modes_x = np.gradient(modes, x, axis=0, edge_order=2).astype(np.float32)
    modes_xx = np.gradient(modes_x, x, axis=0, edge_order=2).astype(np.float32)
    return (mean, modes, mean_x, mean_xx, modes_x, modes_xx), retained_variance


def common_grid_residual(prediction, x, t):
    ux = np.gradient(prediction, x, axis=0, edge_order=2)
    uxx = np.gradient(ux, x, axis=0, edge_order=2)
    ut = np.gradient(prediction, t, axis=1, edge_order=2)
    return ut + prediction * ux - NU * uxx
