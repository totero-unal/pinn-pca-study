"""Check censoring and paired horizons with small known trajectories."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn_pca.time_to_target import first_hit, common_horizon_pair, analyze_trajectories


def trace(values, seconds=None):
    times = seconds or list(range(1, len(values) + 1))
    return [{"step": (i + 1) * 100, "elapsed_pipeline_seconds": times[i],
             "validation_rmse": value} for i, value in enumerate(values)]


class TimeToTargetTests(unittest.TestCase):
    def test_first_observed_crossing_is_retained_after_rebound(self):
        result = first_hit(trace([0.8, 0.4, 0.6], [1.2, 2.2, 3.2]), 0.5)
        self.assertEqual(result["hit_step"], 200)
        self.assertEqual(result["hit_pipeline_seconds"], 2.2)
        self.assertEqual(result["previous_checkpoint_seconds"], 1.2)
        self.assertFalse(result["right_censored"])

    def test_failure_is_not_imputed_as_terminal_hit(self):
        result = first_hit(trace([0.8, 0.7]), 0.5)
        self.assertIsNone(result["hit_pipeline_seconds"])
        self.assertIsNone(result["hit_step"])
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["terminal_pipeline_seconds"], 2)

    def test_comparison_caps_later_hit_at_shared_horizon(self):
        base = first_hit(trace([0.8, 0.7], [1, 3]), 0.5)
        other = first_hit(trace([0.8, 0.4], [1, 4]), 0.5)
        result = common_horizon_pair(base, other)
        self.assertEqual(result["common_horizon_seconds"], 3)
        self.assertFalse(result["other_hit_within_horizon"])
        self.assertEqual(result["other_minus_base_capped_seconds"], 0)
        other = first_hit(trace([0.4, 0.3], [1, 4]), 0.5)
        self.assertEqual(common_horizon_pair(base, other)["other_minus_base_capped_seconds"], -2)

    def test_target_uses_only_baseline_final_validation(self):
        rows = []
        for model, values in [("base", [0.8, 0.4]), ("PCA", [0.1, 0.01])]:
            rows.extend({"model": model, "seed": 1, **point} for point in trace(values))
        report = analyze_trajectories(rows, "base", draws=20)
        self.assertEqual(report["target_rmse"], 0.4)
        self.assertEqual(report["paired"]["PCA"]["both_hit_count"], 1)
        with self.assertRaises(ValueError):
            analyze_trajectories(rows[:-1], "base", draws=20)

    def test_invalid_trace_is_rejected(self):
        for points in [trace([0.7, float("nan")]), trace([0.7, 0.6], [2, 1])]:
            with self.assertRaises(ValueError):
                first_hit(points, 0.5)


if __name__ == "__main__":
    unittest.main()
