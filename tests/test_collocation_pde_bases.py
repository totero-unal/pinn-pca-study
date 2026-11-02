import unittest
import numpy as np
from pinn_pca.collocation_pde_bases import matched_selection

class MatchedBasesTests(unittest.TestCase):
    def test_persistent_mode_separates_pod_from_centered_pca(self):
        rng=np.random.default_rng(9);q,_=np.linalg.qr(rng.normal(size=(60,4)))
        temporal=rng.normal(size=(3,20));temporal-=temporal.mean(axis=1,keepdims=True)
        h=10*q[:,:1]+q[:,1:]@temporal
        for method in ('pod_qr','pca_qr','pca_mean_qr'):
            ids,info=matched_selection(h,method,3)
            self.assertEqual(len(set(ids)),3)
            self.assertEqual(info['actual_dimension'],3)
            if method=='pca_qr':self.assertLess(info['mean_projection_fraction'],1e-12)
            elif method=='pca_mean_qr':self.assertAlmostEqual(info['mean_projection_fraction'],1.)
        with self.assertRaises(ValueError):matched_selection(np.ones((60,20)),'pca_qr',3)

if __name__=='__main__':unittest.main()
