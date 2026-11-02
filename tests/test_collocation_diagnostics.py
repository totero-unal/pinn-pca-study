import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pinn_pca.collocation_diagnostics import coverage_statistics, front_mask, gradient_geometry, loss_components
from pinn_pca.controlled import baseline_ad_loss
from pinn_pca.models import init_mlp


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = init_mlp(jax.random.PRNGKey(17), [2, 4, 1])
        x, t = jnp.linspace(-.8, .8, 8), jnp.linspace(.1, .9, 8)
        cls.batch = (x, t, jnp.sin(x), x, t, x[:3], jnp.sin(x[:3]))

    def test_decomposition_and_gradient_sum_match_original_objective(self):
        parts = loss_components(self.params, self.batch)
        self.assertAlmostEqual(float(parts.sum()), float(baseline_ad_loss(self.params, self.batch)), places=6)
        gradients = jax.grad(baseline_ad_loss)(self.params, self.batch)
        expected_norm = np.sqrt(sum(np.sum(np.asarray(v) ** 2) for v in jax.tree.leaves(gradients)))
        measured = gradient_geometry(self.params, self.batch)
        np.testing.assert_allclose(float(measured["grad_total"]), expected_norm, rtol=1e-5)
        for key in ("cos_physics_data", "cos_physics_initial", "cos_data_initial", "cos_physics_data_initial"):
            self.assertLessEqual(abs(float(measured[key])), 1 + 1e-6)
        self.assertLessEqual(float(measured["gradient_cancellation_ratio"]), 1 + 1e-6)

    def test_replacing_only_physics_points_preserves_supervised_probes(self):
        altered = (*self.batch[:3], self.batch[3] * .1, self.batch[4] * .3, *self.batch[5:])
        left, right = gradient_geometry(self.params, self.batch), gradient_geometry(self.params, altered)
        for key in ("loss_data", "loss_initial", "grad_data", "grad_initial", "grad_data_initial"):
            np.testing.assert_allclose(float(left[key]), float(right[key]), rtol=1e-6)
        self.assertNotAlmostEqual(float(left["loss_physics"]), float(right["loss_physics"]), places=6)

    def test_coverage_uses_normalized_distance_and_declared_window(self):
        config = {"coverage_bins": 2, "front_abs_x_max": .1, "front_time_min": .4}
        active = np.array([[-1., 0.], [0., .5], [1., 1.]])
        np.testing.assert_array_equal(front_mask(active, config), [False, True, False])
        measures = coverage_statistics(active, 2, active.copy(), ((-1, 1), (0, 1)), config)
        self.assertEqual(measures["independent_nearest_distance_q95"], 0)
        self.assertAlmostEqual(measures["front_fraction"], 1 / 3)
        self.assertEqual(measures["basis_front_fraction"], .5)
        self.assertEqual(measures["sobol_front_fraction"], 0)
        # Doubling the x domain and both sets leaves normalized distances unchanged.
        scaled = active.copy()
        scaled[:, 0] *= 2
        other = coverage_statistics(scaled, 2, scaled, ((-2, 2), (0, 1)), config)
        self.assertEqual(measures["occupied_cell_fraction"], other["occupied_cell_fraction"])


if __name__ == "__main__":
    unittest.main()
