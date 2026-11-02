import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pinn_pca.collocation_hard_conditions import pde_only_loss,residual_batch
from pinn_pca.models import init_mlp

class PDEOnlyTests(unittest.TestCase):
    def test_labels_and_supervised_coordinates_cannot_affect_loss_or_gradient(self):
        p=init_mlp(jax.random.PRNGKey(4),[2,4,1])
        x=jnp.linspace(-.8,.8,8);t=jnp.linspace(.1,.9,8);nan=jnp.full_like(x,jnp.nan)
        first=(x,t,jnp.sin(x),x,t,x,jnp.cos(x))
        poisoned=(nan,nan,nan,x,t,nan,nan)
        a,g0=jax.value_and_grad(pde_only_loss)(p,first)
        b,g1=jax.value_and_grad(pde_only_loss)(p,poisoned)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,jnp.mean(residual_batch(p,x,t)**2),rtol=1e-6)
        for u,v in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):np.testing.assert_array_equal(u,v)

if __name__=='__main__':unittest.main()
