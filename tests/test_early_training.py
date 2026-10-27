"""Check window integration and observation handling in the retrospective analysis."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pinn_pca.early_training import window_average,last_before,category,_paired_summary


class EarlyTrainingTests(unittest.TestCase):
    def test_window_respects_unequal_checkpoint_intervals(self):
        trace=[{'step':100,'validation_rmse':3.}, {'step':200,'validation_rmse':1.},
               {'step':500,'validation_rmse':2.}]
        self.assertEqual(window_average(trace,100,500),1.625)

    def test_window_cannot_extrapolate_unobserved_endpoints(self):
        trace=[{'step':100,'validation_rmse':3.},{'step':500,'validation_rmse':2.}]
        with self.assertRaises(ValueError):window_average(trace,0,500)
        with self.assertRaises(ValueError):window_average(trace,100,600)

    def test_deadline_uses_previous_observation_and_keeps_missing(self):
        trace=[{'step':100,'elapsed_pipeline_seconds':.6}, {'step':200,'elapsed_pipeline_seconds':1.2}]
        self.assertIsNone(last_before(trace,.5))
        self.assertEqual(last_before(trace,.6)['step'],100)
        self.assertEqual(last_before(trace,1.)['step'],100)
        self.assertEqual(last_before(trace,1.2)['step'],200)

    def test_ratio_of_means_is_not_mean_of_individual_ratios(self):
        result=_paired_summary([{'a':1.,'b':2.},{'a':3.,'b':4.}],'a','b',.05)
        self.assertEqual(result['ratio_of_means'],1.5)
        self.assertEqual(result['paired_delta_mean'],1.)
        self.assertEqual(result['pca_better_count'],0)

    def test_similarity_band_is_inclusive_at_boundaries(self):
        self.assertEqual(category(.95,.05),'Similar ±5%')
        self.assertEqual(category(1.05,.05),'Similar ±5%')
        self.assertEqual(category(.94,.05),'PCA menor >5%')
        self.assertEqual(category(1.06,.05),'PCA mayor >5%')


if __name__=='__main__':unittest.main()
