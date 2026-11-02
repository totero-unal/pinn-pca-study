import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pinn_pca.models import init_mlp,NU
from pinn_pca.collocation_hard_conditions import solution,residual,loss

class HardTests(unittest.TestCase):
    def test_constraints_for_arbitrary_weights(self):
        p=init_mlp(jax.random.PRNGKey(7),[2,8,1])
        for x in np.linspace(-.9,.9,9):self.assertAlmostEqual(float(solution(p,jnp.float32(x),jnp.float32(0))),-np.sin(np.pi*x),places=6)
        for t in np.linspace(0,1,7):
            for x in (-1.,1.):self.assertEqual(float(solution(p,jnp.float32(x),jnp.float32(t))),0)
    def test_transformed_residual_includes_all_product_derivatives(self):
        p=[(jnp.zeros((2,1)),jnp.array([2.]))]
        for x,t in ((-.6,.1),(.2,.5),(.7,.9)):
            u=-np.sin(np.pi*x)+2*t*(1-x*x)
            expected=2*(1-x*x)+u*(-np.pi*np.cos(np.pi*x)-4*t*x)-NU*(np.pi**2*np.sin(np.pi*x)-4*t)
            self.assertAlmostEqual(float(residual(p,jnp.float32(x),jnp.float32(t))),expected,places=5)
    def test_supervised_data_remains_in_objective(self):
        p=[(jnp.zeros((2,1)),jnp.zeros(1))];x=jnp.linspace(-.8,.8,8);t=jnp.linspace(.1,.9,8)
        truth=jax.vmap(lambda xx,tt:solution(p,xx,tt))(x,t)
        a=(x,t,truth,x,t,x,truth);b=(x,t,truth+2,x,t,x,truth)
        self.assertAlmostEqual(float(loss(p,b)-loss(p,a)),4.,places=5)

if __name__=='__main__':unittest.main()
