import unittest
import numpy as np
from pinn_pca.collocation_ic_confirmation import paired_stats

class PairedTests(unittest.TestCase):
    def test_pairing_retains_sign_and_bootstrap_degenerate_case(self):
        result=paired_stats(np.full(10,-.2))
        self.assertEqual(result['wins'],10)
        np.testing.assert_allclose(result['bootstrap95'],[-.2,-.2])
        self.assertAlmostEqual(result['mean_difference'],-.2)
    def test_mixed_differences_keep_dispersion(self):
        result=paired_stats(np.array([-1.,1.]*5))
        self.assertEqual(result['wins'],5)
        self.assertEqual(result['mean_difference'],0)
        self.assertLess(result['bootstrap95'][0],0)
        self.assertGreater(result['bootstrap95'][1],0)

if __name__=='__main__':unittest.main()
