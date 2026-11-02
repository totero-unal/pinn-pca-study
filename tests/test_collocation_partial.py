import unittest
import numpy as np
from pinn_pca.collocation_partial import replace_converged

class PartialTests(unittest.TestCase):
    def test_remove_largest_decrease_and_keep_size(self):
        active=np.arange(8).reshape(4,2);new=np.array([[20,21],[22,23]])
        after,removed,d,ok=replace_converged(active,new,np.array([8.,4.,1.,1.]),np.ones(4))
        self.assertTrue(ok)
        np.testing.assert_array_equal(removed,[0,1])
        np.testing.assert_array_equal(after,np.concatenate((active[2:],new)))
    def test_cancel_whole_update_if_one_top_degree_negative(self):
        active=np.arange(8).reshape(4,2);new=np.array([[20,21],[22,23]])
        after,removed,d,ok=replace_converged(active,new,np.ones(4),np.array([.5,2.,3.,4.]))
        self.assertFalse(ok);self.assertEqual(len(removed),0)
        np.testing.assert_array_equal(after,active)
    def test_zero_floor_and_overlap(self):
        active=np.arange(8).reshape(4,2)
        _,_,d,_=replace_converged(active,np.array([[20,21]]),np.zeros(4),np.zeros(4))
        self.assertTrue(np.isfinite(d).all())
        with self.assertRaises(ValueError):replace_converged(active,active[:1],np.ones(4),np.ones(4))

if __name__=="__main__":unittest.main()
