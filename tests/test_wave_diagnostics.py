"""Check the one-factor scaling ablation and validation-only oracle decision."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from pinn_pca import wave
from pinn_pca.models import init_mlp
from pinn_pca.wave_diagnostics import (
    NORMALIZED, fit_coefficient_scaler, normalized_solution, normalized_prediction,
    match_physical_initialization, oracle_gate, oracle_parameters,
    oracle_point_values, parameter_diagnostics)


class WaveDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.t,cls.u=wave.make_wave_data(64,40,1.,[1,4],[1.,.5])
        cls.fit=np.arange(0,40,2)
        cls.basis,cls.info=wave.fit_wave_pca(cls.x,cls.u,cls.fit)
        cls.center,cls.scale=fit_coefficient_scaler(cls.u,cls.info,cls.fit)
        cls.state=(*cls.basis,cls.center,cls.scale)

    def test_scaler_uses_training_only_and_produces_unit_variance(self):
        contaminated=self.u.copy();contaminated[:,1::2]+=1e6
        center,scale=fit_coefficient_scaler(contaminated,self.info,self.fit)
        np.testing.assert_array_equal(center,self.center)
        np.testing.assert_array_equal(scale,self.scale)
        c=(self.u[:,self.fit].T-self.info['mean'])@self.info['modes']
        z=(c-np.asarray(center))/np.asarray(scale)
        np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-7)
        np.testing.assert_allclose(z.std(axis=0),1,atol=1e-7)

    def test_normalized_head_preserves_initial_field_and_derivatives(self):
        raw=init_mlp(jax.random.PRNGKey(99),[1,8,8,8,self.info['k']])
        norm=match_physical_initialization(raw,self.center,self.scale)
        a=lambda x,t:wave.pca_wave_solution(raw,self.basis,x,t)
        b=lambda x,t:normalized_solution(norm,self.state,x,t)
        for x,t in [(.12,.24),(.71,.46),(.98,.85)]:
            xx,tt=jnp.asarray(x),jnp.asarray(t)
            np.testing.assert_allclose(a(xx,tt),b(xx,tt),atol=2e-7)
            np.testing.assert_allclose(wave.wave_residual(a,xx,tt),wave.wave_residual(b,xx,tt),rtol=3e-5,atol=2e-5)
        grid=normalized_prediction(norm,self.state,jnp.asarray(self.x),jnp.asarray(self.t))
        np.testing.assert_allclose(grid,wave.predict_pca(raw,self.basis,jnp.asarray(self.x),jnp.asarray(self.t)),atol=3e-7)

    def test_gate_does_not_use_test_results(self):
        limits={'max_validation_relative_l2':1e-4,'max_normalized_pde_residual':1e-3,'max_oracle_to_null_loss_ratio':.1}
        good={'case':'case','seed':1,'representation':'PCA oráculo spline C2','evaluation':'validation',
              'relative_l2':1e-7,'normalized_pde_residual':1e-4,'oracle_to_null_loss_ratio':.01}
        bad_test={**good,'evaluation':'test','relative_l2':1.}
        self.assertTrue(oracle_gate([good,bad_test],limits)['passed'])
        self.assertFalse(oracle_gate([{**good,'oracle_to_null_loss_ratio':.5}],limits)['passed'])
        self.assertFalse(oracle_gate([],limits)['passed'])

    def test_analytic_pca_oracle_satisfies_the_wave_equation(self):
        oracle=oracle_parameters(self.x,self.info,[1,4],[1.,.5])
        values=oracle_point_values(self.basis,oracle,jnp.asarray([1.,4.]),jnp.asarray([1.,.5]),2.,
            jnp.asarray([.15,.43,.89]),jnp.asarray([.24,.56,.78]),'PCA oráculo seno')
        np.testing.assert_allclose(values[2]-values[3],0,atol=1e-4)

    def test_logged_gradient_norms_match_individual_gradients(self):
        params=init_mlp(jax.random.PRNGKey(8),[1,4,4,4,self.info['k']])
        params=match_physical_initialization(params,self.center,self.scale)
        batches,_=wave.make_wave_batches(self.x,self.t,self.u,self.fit,np.arange(4),1,8,9)
        terms,norms=parameter_diagnostics(params,self.state,batches[0],2.,NORMALIZED)
        fn=lambda p:wave.loss_terms(lambda x,t:normalized_solution(p,self.state,x,t),batches[0],2.)
        np.testing.assert_allclose(terms,fn(params),rtol=1e-5,atol=1e-8)
        for i in range(5):
            leaves=jax.tree.leaves(jax.grad(lambda p:fn(p)[i])(params))
            expected=np.sqrt(sum(float(jnp.sum(v**2)) for v in leaves))
            np.testing.assert_allclose(norms[i],expected,rtol=5e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
