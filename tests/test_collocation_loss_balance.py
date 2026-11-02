import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pinn_pca.collocation_loss_balance import components,weighted_loss
from pinn_pca.controlled import baseline_ad_loss
from pinn_pca.models import init_mlp

class LossBalanceTests(unittest.TestCase):
    def test_baseline_objective_and_gradients_match_historical_loss(self):
        p=init_mlp(jax.random.PRNGKey(4),[2,4,1])
        x=jnp.linspace(-.9,.9,8);t=jnp.linspace(.1,.9,8)
        batch=(x,t,jnp.sin(x),x,t,x,jnp.sin(x))
        old,g0=jax.value_and_grad(baseline_ad_loss)(p,batch)
        new,g1=jax.value_and_grad(weighted_loss([1,1,1,0]))(p,batch)
        np.testing.assert_allclose(old,new,rtol=1e-6)
        for a,b in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    def test_ic_and_bc_controls_change_only_their_terms(self):
        p=[(jnp.zeros((2,1)),jnp.array([2.]))]
        x=jnp.linspace(-.9,.9,8);t=jnp.linspace(.1,.9,8);zero=jnp.zeros(8)
        batch=(x,t,zero,x,t,x,zero)
        np.testing.assert_allclose(components(p,batch),[4,0,4,4])
        self.assertAlmostEqual(float(weighted_loss([1,1,10,0])(p,batch)),44)
        self.assertAlmostEqual(float(weighted_loss([1,1,1,1])(p,batch)),12)

if __name__=='__main__':unittest.main()
