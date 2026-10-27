"""Training objectives for the baseline PINN and PCA-preprocessed PINN."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .models import NU, burgers_residual, burgers_solution, init_mlp, mlp


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)


def adam_step(params, grads, first_moment, second_moment, step, learning_rate):
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = step + 1
    first_moment = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, first_moment, grads)
    second_moment = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), second_moment, grads)
    first_hat = jax.tree.map(lambda m: m / (1.0 - beta1**step), first_moment)
    second_hat = jax.tree.map(lambda v: v / (1.0 - beta2**step), second_moment)
    params = jax.tree.map(
        lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + epsilon),
        params, first_hat, second_hat,
    )
    return params, first_moment, second_moment, step


def baseline_loss(params, batch):
    (_, data_x, data_t, data_u, _, residual_x, residual_t,
     _, initial_x, initial_u) = batch
    predictions = jax.vmap(lambda xx, tt: burgers_solution(params, xx, tt))(data_x, data_t)
    residuals = jax.vmap(lambda xx, tt: burgers_residual(params, xx, tt))(residual_x, residual_t)
    initial_predictions = jax.vmap(
        lambda xx: burgers_solution(params, xx, jnp.asarray(0.0, dtype=xx.dtype))
    )(initial_x)
    return (jnp.mean((predictions - data_u) ** 2)
            + jnp.mean(residuals**2)
            + jnp.mean((initial_predictions - initial_u) ** 2))


def pca_loss(params, batch, basis_state):
    mean, modes, mean_x, mean_xx, modes_x, modes_xx = basis_state

    def coefficients(tt):
        return mlp(params, jnp.asarray([tt]))

    coefficients_t = jax.jacfwd(coefficients)

    def solution_at_index(index, tt):
        values = coefficients(tt)
        return mean[index] + jnp.sum(modes[index] * values)

    def residual_at_index(index, tt):
        values = coefficients(tt)
        values_t = coefficients_t(tt)
        u = mean[index] + jnp.sum(modes[index] * values)
        ux = mean_x[index] + jnp.sum(modes_x[index] * values)
        uxx = mean_xx[index] + jnp.sum(modes_xx[index] * values)
        ut = jnp.sum(modes[index] * values_t)
        return ut + u * ux - NU * uxx

    (data_indices, _, data_t, data_u, residual_indices, _, residual_t,
     initial_indices, _, initial_u) = batch
    data_predictions = jax.vmap(lambda idx, tt: solution_at_index(idx, tt))(data_indices, data_t)
    residuals = jax.vmap(lambda idx, tt: residual_at_index(idx, tt))(residual_indices, residual_t)
    initial_predictions = jax.vmap(
        lambda idx: solution_at_index(idx, jnp.asarray(0.0, dtype=mean.dtype))
    )(initial_indices)
    return (jnp.mean((data_predictions - data_u) ** 2)
            + jnp.mean(residuals**2)
            + jnp.mean((initial_predictions - initial_u) ** 2))


def make_update(loss_fn, learning_rate, with_basis=False):
    value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

    if with_basis:
        @jax.jit
        def update(params, first_moment, second_moment, step, batch, basis_state):
            loss, grads = value_and_grad(params, batch, basis_state)
            params, first_moment, second_moment, step = adam_step(
                params, grads, first_moment, second_moment, step, learning_rate)
            return params, first_moment, second_moment, step, loss
    else:
        @jax.jit
        def update(params, first_moment, second_moment, step, batch):
            loss, grads = value_and_grad(params, batch)
            params, first_moment, second_moment, step = adam_step(
                params, grads, first_moment, second_moment, step, learning_rate)
            return params, first_moment, second_moment, step, loss
    return update


def warmup_update(update, params, batch, basis_state=None):
    state = adam_init(params)
    start = time.perf_counter()
    if basis_state is None:
        output = update(params, *state, batch)
    else:
        output = update(params, *state, batch, basis_state)
    jax.tree.map(lambda value: value.block_until_ready(), output)
    return time.perf_counter() - start


def fit_model(update, params, batches, basis_state=None):
    first_moment, second_moment, step = adam_init(params)
    start = time.perf_counter()
    loss = jnp.asarray(jnp.nan)
    for batch in batches:
        if basis_state is None:
            params, first_moment, second_moment, step, loss = update(
                params, first_moment, second_moment, step, batch)
        else:
            params, first_moment, second_moment, step, loss = update(
                params, first_moment, second_moment, step, batch, basis_state)
        _ = float(loss)
    jax.tree.map(lambda value: value.block_until_ready(), params)
    return params, time.perf_counter() - start, float(loss)


def predict_baseline(params, x, t):
    prediction = _predict_baseline(params, jnp.asarray(x), jnp.asarray(t))
    return np.asarray(prediction.block_until_ready())


@jax.jit
def _predict_baseline(params, x, t):
    xx, tt = jnp.meshgrid(x, t, indexing="ij")
    prediction = jax.vmap(lambda a, b: burgers_solution(params, a, b))(xx.ravel(), tt.ravel())
    return prediction.reshape((x.shape[0], t.shape[0]))


def predict_pca(params, basis_state, t):
    prediction = _predict_pca(params, basis_state, jnp.asarray(t))
    return np.asarray(prediction.block_until_ready())


@jax.jit
def _predict_pca(params, basis_state, t):
    mean, modes, *_ = basis_state
    coefficients = jax.vmap(lambda tt: mlp(params, jnp.asarray([tt])))(t)
    prediction = mean[:, None] + modes @ coefficients.T
    return prediction
