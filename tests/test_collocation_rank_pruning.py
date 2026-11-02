import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from pinn_pca.collocation_rank_pruning import RankPruningRecorder

class RankPruningTests(unittest.TestCase):
    def test_control_uses_pca_schedule_and_realigns_baseline(self):
        rng=np.random.default_rng(3)
        points=rng.uniform([-1,0],[1,1],(128,2)).astype(np.float32)
        r=RankPruningRecorder({},Path('.'),'sobol_partial',{200:2,300:5})
        kwargs=dict(pool=np.empty((0,2)),history=None,seed=2051,bounds=((-1,1),(0,1)))
        with patch('pinn_pca.collocation_rank_pruning._residual_field',side_effect=lambda level,p:np.full(len(p),level)):
            r.adjust_active('sobol_partial',100,1.,points,np.arange(3),3,**kwargs)
            updated,ids,k=r.adjust_active('sobol_partial',200,.5,points,np.arange(10),10,**kwargs)
            self.assertEqual(k,2)
            self.assertEqual(len(set(map(tuple,points))-set(map(tuple,updated))),2)
            np.testing.assert_array_equal(r.baseline,np.full(128,.5))
            updated2,_,k=r.adjust_active('sobol_partial',300,.25,points,np.arange(10),10,**kwargs)
            self.assertEqual(k,5)
            self.assertEqual(len(set(map(tuple,updated))-set(map(tuple,updated2))),5)
            self.assertEqual(len(r.exits),7)
            self.assertEqual(len(r.residence),128)

    def test_negative_convergence_cancels_rank_sized_update(self):
        rng=np.random.default_rng(4);points=rng.uniform([-1,0],[1,1],(128,2)).astype(np.float32)
        r=RankPruningRecorder({},Path('.'),'sobol_partial',{200:7})
        kwargs=dict(pool=np.empty((0,2)),history=None,seed=2051,bounds=((-1,1),(0,1)))
        with patch('pinn_pca.collocation_rank_pruning._residual_field',side_effect=lambda level,p:np.full(len(p),level)):
            r.adjust_active('sobol_partial',100,1.,points,np.arange(3),3,**kwargs)
            updated,_,k=r.adjust_active('sobol_partial',200,2.,points,np.arange(10),10,**kwargs)
        self.assertEqual(k,7);np.testing.assert_array_equal(updated,points)
        self.assertFalse(r.pruning_rows[-1]['accepted'])
        self.assertEqual(r.pruning_rows[-1]['replaced'],0)

if __name__=='__main__':unittest.main()
