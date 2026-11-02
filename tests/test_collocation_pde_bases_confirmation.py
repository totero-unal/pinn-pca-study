import unittest
from pinn_pca.collocation_pde_bases_confirmation import validate_confirmation,FROZEN

class ConfirmationTests(unittest.TestCase):
    def test_rejects_seed_reuse_and_changes_to_protocol(self):
        prior={key:1 for key in FROZEN};prior['seeds']=list(range(2071,2081))
        new=dict(prior,seeds=list(range(2091,2101)),confirmation=True,primary_comparison='centered_pca_minus_pod_terminal_rmse')
        validate_confirmation(new,prior)
        with self.assertRaises(ValueError):validate_confirmation(dict(new,seeds=prior['seeds']),prior)
        with self.assertRaises(ValueError):validate_confirmation(dict(new,learning_rate=2),prior)
        with self.assertRaises(ValueError):validate_confirmation(dict(new,confirmation=False),prior)

if __name__=='__main__':unittest.main()
