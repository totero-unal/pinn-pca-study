import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from pinn_pca.collocation_memory import MemoryRecorder

class MemoryTests(unittest.TestCase):
    def test_window_cadence_and_gradual_fill(self):
        for frequency,memory in ((5,100),(10,100),(20,100),(5,200),(5,400)):
            r=MemoryRecorder({},Path('.'),frequency,memory)
            pool=np.arange(8).reshape(4,2);h=np.ones((4,20))
            r.prepare_history(100,pool,h,None)
            with patch('pinn_pca.collocation_memory._residual_field',side_effect=lambda step,p:p[:,0]-step):
                for step in range(101,601):
                    r.training_step(step,step)
                    if step%100==0:
                        _,h=r.prepare_history(step,pool,h,None)
                        expected=np.arange(max(100,step-memory)+frequency,step+1,frequency)
                        np.testing.assert_array_equal(r.times,expected)
                        np.testing.assert_array_equal(h,pool[:,0,None]-expected[None,:])
                        self.assertEqual(h.shape[1],min(step-100,memory)//frequency)
                        self.assertEqual(r.log[-1]['full_window'],step>=100+memory)

if __name__=='__main__':unittest.main()
