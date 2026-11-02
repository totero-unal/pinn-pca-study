import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from pinn_pca.collocation_intervals import IntervalRecorder

class IntervalTests(unittest.TestCase):
    def test_each_history_row_keeps_same_coordinate_and_signed_values(self):
        for refresh in (False,True):
            recorder=IntervalRecorder(dict(candidate_count=4),Path('.'),refresh)
            recorder.c=dict(seed=2051)
            initial=np.array([[-.8,.1],[-.4,.3],[.4,.6],[.8,.9]],dtype=np.float32)
            old=np.ones((4,20))
            p,h=recorder.prepare_history(100,initial,old,((-1,1),(0,.99)))
            np.testing.assert_array_equal(h,old)
            observed=recorder.next_pool.copy()
            with patch('pinn_pca.collocation_intervals._residual_field',side_effect=lambda step,pts:pts[:,0]-step):
                for step in range(101,201):recorder.training_step(step,step)
            p,h=recorder.prepare_history(200,p,h,((-1,1),(0,.99)))
            np.testing.assert_array_equal(p,observed)
            np.testing.assert_allclose(h,observed[:,0,None]-np.arange(105,201,5)[None,:],atol=1e-5)
            self.assertTrue(np.all(h<0))
            self.assertEqual(h.shape,(4,20))
            self.assertEqual(np.array_equal(p,initial),not refresh)
            self.assertEqual(len(recorder.columns),0)

if __name__=='__main__':unittest.main()
