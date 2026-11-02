import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from pinn_pca.collocation import _initial_condition_indices
from pinn_pca.collocation_ic_budget import BudgetRecorder, compare_prefix_rows, experiment_paths, validate_budget
from pinn_pca.controlled import make_controlled_batches

ROOT = Path(__file__).resolve().parents[1]


class LongBudgetTests(unittest.TestCase):
    def test_rejects_changed_training_settings_and_unfixed_test_budgets(self):
        config = json.loads((ROOT / 'pca_collocation/configs/burgers_ic_long_budget.json').read_text(encoding='utf-8'))
        baseline = json.loads((ROOT / 'pca_collocation/results/burgers/ic_coverage_control' / config['reference_ic_run_id'] / 'config.json').read_text(encoding='utf-8'))
        validate_budget(config, baseline)
        for key, value in (('learning_rate', .02), ('initial_condition_strategy', 'left_prefix'), ('train_steps', 2000), ('test_checkpoint_steps', []), ('test_checkpoint_steps', [2100, 3150, 5100])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_budget(dict(config, **{key: value}), baseline)

    def test_prefix_audit_rejects_missing_or_changed_training_values_but_ignores_time(self):
        old = [dict(method='sobol', step='100', validation_rmse='.3', independent_residual_rms='.2', initial_condition_rmse='.1', boundary_rms='.05', training_loss='', pipeline_seconds='1')]
        changed_time = [dict(old[0], pipeline_seconds='2')]
        self.assertTrue(all(row['difference'] == 0 for row in compare_prefix_rows(old, changed_time, 1e-6)))
        with self.assertRaises(ValueError):
            compare_prefix_rows(old, [], 1e-6)
        with self.assertRaises(ValueError):
            compare_prefix_rows(old, [dict(old[0], validation_rmse='.31')], 1e-6)

    def test_longer_batch_generation_preserves_the_prefix(self):
        x, t = np.linspace(-1, 1, 32, dtype=np.float32), np.linspace(0, 1, 10, dtype=np.float32)
        u, ids = x[:, None] + t[None, :], _initial_condition_indices(32, 8, 'uniform_domain')
        before = make_controlled_batches(x, t, u, np.array([0, 1, 4, 6]), ids, 3, 8, 51)
        after = make_controlled_batches(x, t, u, np.array([0, 1, 4, 6]), ids, 8, 8, 51)
        for left, right in zip(before, after):
            for a, b in zip(left, right):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_added_test_metrics_are_evaluated_only_at_the_fixed_checkpoints(self):
        recorder = BudgetRecorder({'test_checkpoint_steps': [2100, 3100], 'warmup_steps': 100}, ROOT)
        recorder.seed, recorder.x, recorder.t, recorder.u = 1, None, None, None
        recorder.context = {'test_ids': [1], 'independent': None}
        with patch('pinn_pca.collocation_diagnostics.DiagnosticRecorder.checkpoint'), patch('pinn_pca.collocation_ic_budget._evaluate', return_value={'test_rmse': .1}) as evaluate:
            for step in (100, 200, 2100, 2200, 3100):
                recorder.checkpoint('sobol', step, None, None, step-100)
            self.assertEqual(evaluate.call_count, 2)
        self.assertEqual([r['step'] for r in recorder.budget_metrics], [2100, 3100])

    def test_ten_thousand_extension_uses_the_five_thousand_reference(self):
        config = json.loads((ROOT / 'pca_collocation/configs/burgers_ic_10000_steps.json').read_text(encoding='utf-8'))
        before, after = experiment_paths(config, ROOT)
        self.assertEqual(before.parent.name, 'ic_long_budget')
        self.assertEqual(before.name, 'ic_uniform_domain_10_seeds_5k_v1')
        self.assertNotEqual(before, after)
        baseline = json.loads((before / 'config.json').read_text(encoding='utf-8'))
        validate_budget(config, baseline)
        self.assertEqual(config['test_checkpoint_steps'], [5100, 6100, 7100, 8100, 9100, 10100])
        with self.assertRaises(ValueError):
            experiment_paths(dict(config, reference_result_group='../elsewhere'), ROOT)
        with self.assertRaises(ValueError):
            experiment_paths(dict(config, run_id=config['reference_ic_run_id']), ROOT)
        with self.assertRaises(ValueError):
            validate_budget(dict(config, test_checkpoint_steps=[2100, 5100, 10100]), baseline)


if __name__ == '__main__':
    unittest.main()
