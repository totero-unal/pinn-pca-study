import unittest
import numpy as np
from pinn_pca.collocation_rank_study import adaptive_basis,select_adaptive_basis,RankUnavailable
from pinn_pca.collocation_basis_ablation import select_fixed_basis


class RankTests(unittest.TestCase):
    def test_squared_energy_minimal_rank(self):
        h=np.diag([4.,2.,1.])
        b,info=adaptive_basis(h,"pod_qr",.7)
        self.assertEqual(info["spectral_modes"],1)
        self.assertAlmostEqual(info["spectral_fraction"],16/21)
        for threshold, expected in ((.95,2),(.99,3),(.999,3)):
            b,info=adaptive_basis(h,"pod_qr",threshold)
            self.assertEqual(b.shape[1],expected)

    def test_centered_threshold_and_mean_preservation(self):
        rng=np.random.default_rng(10)
        h=rng.normal(size=(60,20))+rng.normal(size=(60,1))*5
        previous={m:0 for m in ("pod_qr","pca_qr","pca_mean_qr")}
        for threshold in (.95,.99,.999):
            for m in previous:
                ids,info=select_adaptive_basis(h,m,threshold,32)
                self.assertEqual(len(set(ids)),32)
                self.assertGreaterEqual(info["actual_dimension"],previous[m])
                previous[m]=info["actual_dimension"]
                quantity="raw_energy_captured" if m=="pod_qr" else "centered_variance_captured"
                self.assertGreaterEqual(info[quantity]+1e-12,threshold)
                if m=="pca_mean_qr":
                    self.assertAlmostEqual(info["mean_projection_fraction"],1.)
                    self.assertLessEqual(info["actual_dimension"],info["spectral_modes"]+1)
                b,_=adaptive_basis(h,m,threshold)
                np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=1e-12)

    def test_zero_history_and_mean_only_cases(self):
        with self.assertRaises(RankUnavailable):
            adaptive_basis(np.zeros((60,20)),"pod_qr",.99)
        with self.assertRaises(RankUnavailable):
            adaptive_basis(np.ones((60,20)),"pca_qr",.99)
        b,info=adaptive_basis(np.ones((60,20)),"pca_mean_qr",.99)
        self.assertEqual(b.shape[1],1)
        self.assertEqual(info["spectral_modes"],0)
        # The selection guard must regard zero fluctuation energy as fully represented.


if __name__=="__main__":
    unittest.main()
