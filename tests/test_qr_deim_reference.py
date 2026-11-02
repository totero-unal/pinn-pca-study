import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pinn_pca.models import init_mlp, NU
from pinn_pca.training import adam_init
from pinn_pca.qr_deim_reference import (
    solution, residual, residuals, select_qr, prune, uniform_points,
    train_period, learning_rate,
)


class ReferenceTests(unittest.TestCase):
    def test_hard_conditions_and_analytic_residual(self):
        params = init_mlp(jax.random.PRNGKey(3), [2, 8, 1])
        for x in np.linspace(-.9, .9, 9):
            self.assertAlmostEqual(float(solution(params, jnp.float32(x), jnp.float32(0))), -np.sin(np.pi*x), places=6)
        for t in np.linspace(0, 1, 5):
            for x in (-1., 1.):
                self.assertEqual(float(solution(params, jnp.float32(x), jnp.float32(t))), 0)
        zero = jax.tree.map(jnp.zeros_like, params)
        x = .3
        expected = np.pi*np.sin(np.pi*x)*np.cos(np.pi*x) - NU*np.pi**2*np.sin(np.pi*x)
        self.assertAlmostEqual(float(residual(zero, jnp.float32(x), jnp.float32(.4))), expected, places=5)

    def test_rank_is_linear_singular_sum_and_exact_qr_count(self):
        indices, singular, modes = select_qr(np.diag([4., 2., 1.]), .3)
        self.assertEqual(len(indices), 2)  # squared-energy criterion would give 1
        self.assertEqual(len(set(indices)), 2)
        self.assertEqual(modes.shape, (3, 2))
        self.assertEqual(len(select_qr(np.zeros((4, 3)), .005)[0]), 0)
        with self.assertRaises(ValueError):
            select_qr(np.array([[np.nan]]), .005)

    def test_prune_and_align_baseline_with_inserted_coordinates(self):
        points = np.array([[-.8, .1], [0., .5], [.8, .9]])
        candidates = np.array([[-.5, .2], [.5, .7]])
        updated, baseline, removed, d, accepted = prune(points, candidates, [1],
                                                        np.array([8., 2., 1.]), np.array([1., 1., 2.]), np.array([3., 4.]))
        self.assertTrue(accepted)
        np.testing.assert_array_equal(removed, [0])
        np.testing.assert_array_equal(updated, np.vstack([points[1:], candidates[1]]))
        np.testing.assert_array_equal(baseline, [1., 2., 4.])
        updated2, baseline2, removed2, _, accepted2 = prune(points, candidates, [0, 1],
                                                          np.ones(3), np.array([.5, 2., 3.]), np.ones(2))
        self.assertFalse(accepted2)
        np.testing.assert_array_equal(updated2, points)
        np.testing.assert_array_equal(baseline2, [.5, 2., 3.])
        self.assertEqual(len(removed2), 0)
        _, _, _, degrees, _ = prune(points, candidates, [0], np.zeros(3), np.zeros(3), np.zeros(2))
        self.assertTrue(np.isfinite(degrees).all())

    def test_disjoint_and_post_update_snapshots_without_optimizer_reset(self):
        rng = np.random.default_rng(1)
        points = uniform_points(rng, 8)
        candidates = uniform_points(rng, 7, points)
        self.assertFalse(set(map(tuple, points)) & set(map(tuple, candidates)))
        p = init_mlp(jax.random.PRNGKey(1), [2, 4, 1])
        state = (p, *adam_init(p))
        kw = dict(total=4, initial_lr=.001, observe=True)
        two, (_, snapshots) = train_period(state, points, candidates, period=2, **kw)
        one, _ = train_period(state, points, candidates, period=1, **kw)
        sequential, _ = train_period(one, points, candidates, period=1, **kw)
        np.testing.assert_allclose(snapshots[0], residuals(one[0], candidates[:, 0], candidates[:, 1]), atol=1e-6)
        np.testing.assert_allclose(snapshots[1], residuals(two[0], candidates[:, 0], candidates[:, 1]), atol=1e-6)
        unobserved, _ = train_period(state, points, candidates, period=2, total=4, initial_lr=.001, observe=False)
        for a, b, c in zip(jax.tree.leaves(two), jax.tree.leaves(sequential), jax.tree.leaves(unobserved)):
            np.testing.assert_allclose(a, b, atol=1e-7)
            np.testing.assert_allclose(a, c, atol=1e-7)
        self.assertEqual(int(two[-1]), 2)
        self.assertAlmostEqual(float(learning_rate(4, 4, .001)), 0)


if __name__ == "__main__":
    unittest.main()
