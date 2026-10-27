"""Physical reference, train-only PCA and spline derivatives for the wave pilot."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from pinn_pca.wave import exact_wave, wave_residual, make_wave_data, fit_wave_pca, spline_fields, make_wave_batches


class WaveTests(unittest.TestCase):
    def test_reference_satisfies_pde_and_initial_conditions(self):
        fn=lambda x,t:exact_wave(x,t,[1,4,9],[1.,.5,.25])
        x=jnp.asarray([.12,.35,.77]);t=jnp.asarray([.16,.49,.83])
        residual=jax.vmap(lambda xx,tt:wave_residual(fn,xx,tt))(x,t)
        np.testing.assert_allclose(residual,0,atol=5e-4)
        np.testing.assert_allclose(jax.vmap(lambda xx:jax.grad(fn,argnums=1)(xx,0.))(x),0,atol=1e-6)
        for bound in [0.,1.]:
            np.testing.assert_allclose(jax.vmap(lambda tt:fn(bound,tt))(t),0,atol=2e-6)

    def test_cubic_second_derivatives_survive_at_boundary(self):
        knots=np.linspace(0,1,8,dtype=np.float32)
        coefficients=CubicSpline(knots,np.column_stack((knots**3,knots**2))).c
        basis=(jnp.asarray(knots),jnp.asarray(coefficients,dtype=jnp.float32))
        second=jax.jacfwd(jax.jacfwd(lambda xx:spline_fields(basis,xx)))
        np.testing.assert_allclose(second(jnp.asarray(1.)),[6.,2.],atol=2e-4)
        knot=float(knots[3])
        np.testing.assert_allclose(second(jnp.asarray(knot-1e-6)),second(jnp.asarray(knot+1e-6)),atol=2e-4)

    def test_pca_cannot_see_held_out_snapshots(self):
        x,t,u=make_wave_data(64,40,1.,[1,4],[1.,.5])
        fit=np.arange(0,40,2)
        basis,a=fit_wave_pca(x,u,fit)
        contaminated=u.copy();contaminated[:,1::2]+=1e5
        other,b=fit_wave_pca(x,contaminated,fit)
        self.assertEqual(a['k'],2)
        self.assertEqual(a['k'],b['k'])
        for left,right in zip(basis,other):np.testing.assert_array_equal(left,right)
        np.testing.assert_allclose(a['modes'].T@a['modes'],np.eye(2),atol=1e-12)

    def test_sampling_prefix_is_independent_of_step_budget(self):
        x,t,u=make_wave_data(16,20,1.,[1],[1.])
        a,_=make_wave_batches(x,t,u,np.arange(10),np.arange(4),2,8,50)
        b,_=make_wave_batches(x,t,u,np.arange(10),np.arange(4),4,8,50)
        for left,right in zip(a,b):
            for av,bv in zip(left,right):np.testing.assert_array_equal(av,bv)


if __name__=='__main__':unittest.main()
