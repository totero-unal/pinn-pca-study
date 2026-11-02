import unittest

import numpy as np

from pinn_pca.collocation import _initial_condition_indices
from pinn_pca.collocation_ic_control import INVARIANTS, validate_control
from pinn_pca.collocation_diagnostics import METHOD_SPECS
from pinn_pca.controlled import make_controlled_batches


class ICCoverageControlTests(unittest.TestCase):
    def test_same_budget_distributed_nodes_include_endpoints_without_duplicates(self):
        prefix = _initial_condition_indices(256, 32)
        spread = _initial_condition_indices(256, 32, "uniform_domain")
        np.testing.assert_array_equal(prefix, np.arange(32))
        self.assertEqual(len(spread), 32)
        self.assertEqual(len(set(spread)), 32)
        self.assertEqual((spread[0], spread[-1]), (0, 255))
        self.assertLessEqual(np.ptp(np.diff(spread)), 1)
        for count in (0, 257):
            with self.assertRaises(ValueError):
                _initial_condition_indices(256, count, "uniform_domain")
        with self.assertRaises(ValueError):
            _initial_condition_indices(256, 1, "uniform_domain")
        with self.assertRaises(ValueError):
            _initial_condition_indices(256, 32, "unknown")

    def test_ic_placement_preserves_random_supervised_and_residual_batches(self):
        x, t = np.linspace(-1, 1, 256, dtype=np.float32), np.linspace(0, 1, 10, dtype=np.float32)
        u = x[:, None] + t[None, :]
        fit = np.array([0, 1, 3, 6])
        old = make_controlled_batches(x, t, u, fit, _initial_condition_indices(256, 32), 3, 16, 15)
        new = make_controlled_batches(x, t, u, fit, _initial_condition_indices(256, 32, "uniform_domain"), 3, 16, 15)
        for before, after in zip(old, new):
            for i in range(5):
                np.testing.assert_array_equal(np.asarray(before[i]), np.asarray(after[i]))
            self.assertEqual(len(after[5]), 32)
            self.assertFalse(np.array_equal(np.asarray(before[5]), np.asarray(after[5])))
            np.testing.assert_array_equal(np.asarray(after[5]), np.asarray(after[6]))

    def test_control_rejects_changes_to_another_training_setting(self):
        baseline = {key: 32 for key in INVARIANTS}
        baseline.update(seeds=[2051, 2052], method_specs=METHOD_SPECS)
        control = dict(baseline, initial_condition_strategy="uniform_domain")
        validate_control(control, baseline)
        for key in ("learning_rate", "initial_condition_points", "batch_size"):
            changed = dict(control)
            changed[key] += 1
            with self.assertRaisesRegex(ValueError, key):
                validate_control(changed, baseline)


if __name__ == "__main__":
    unittest.main()
