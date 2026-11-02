import unittest
from unittest.mock import patch
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pinn_pca.collocation_pde_pruning import PDEPruningRecorder
from pinn_pca.collocation_hard_conditions import residual_field

class PDEPruningTests(unittest.TestCase):
    def test_pruning_baseline_uses_transformed_residual(self):
        r=PDEPruningRecorder({},Path('.'),'pca_partial')
        p=[(jnp.zeros((2,1)),jnp.array([2.]))]
        points=np.random.default_rng(8).uniform([-.9,.1],[.9,.9],(128,2)).astype(np.float32)
        with patch('pinn_pca.collocation_rank_pruning._residual_field',side_effect=AssertionError('Raw residual must not be used')):
            r.adjust_active('pca_partial',100,p,points,np.arange(3),3,None,None,2051,((-1,1),(0,1)))
        np.testing.assert_array_equal(r.baseline,np.abs(residual_field(p,points)))
        self.assertGreater(np.linalg.norm(r.baseline),1.)

if __name__=='__main__':unittest.main()
