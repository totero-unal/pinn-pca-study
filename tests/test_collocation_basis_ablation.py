import unittest
import numpy as np
from pinn_pca.collocation_basis_ablation import fixed_basis, select_fixed_basis


class BasisTests(unittest.TestCase):
    def test_subspaces_and_equal_dimension(self):
        rng = np.random.default_rng(4)
        h = rng.normal(size=(35, 12)) + rng.normal(size=(35, 1)) * 4
        for method in ("pod_qr", "pca_qr", "pca_mean_qr"):
            b = fixed_basis(h, method, 4)
            self.assertEqual(b.shape, (35, 4))
            np.testing.assert_allclose(b.T @ b, np.eye(4), atol=1e-12)
            if method == "pca_mean_qr":
                mean = h.mean(axis=1)
                np.testing.assert_allclose(b @ b.T @ mean, mean, atol=1e-12)
            else:
                matrix = h if method == "pod_qr" else h - h.mean(axis=1, keepdims=True)
                u, _, _ = np.linalg.svd(matrix, full_matrices=False)
                np.testing.assert_allclose(b @ b.T, u[:, :4] @ u[:, :4].T, atol=1e-12)
            ids, info = select_fixed_basis(h, method, 4, 16)
            self.assertEqual(len(set(ids)), 16)
            self.assertEqual(info["actual_dimension"], 4)
            self.assertEqual(info["qr_initial_count"], 4)

    def test_centering_removes_persistent_component(self):
        rng = np.random.default_rng(9)
        q, _ = np.linalg.qr(rng.normal(size=(30, 5)))
        time = rng.normal(size=(4, 12))
        time -= time.mean(axis=1, keepdims=True)
        h = 10 * q[:, :1] + q[:, 1:] @ time
        centered = fixed_basis(h, "pca_qr", 4)
        self.assertLess(np.linalg.norm(centered.T @ q[:, 0]), 1e-12)
        with_mean = fixed_basis(h, "pca_mean_qr", 4)
        self.assertAlmostEqual(np.linalg.norm(with_mean.T @ q[:, 0]), 1.)

    def test_deficient_histories_fail_instead_of_silent_filler(self):
        for method in ("pod_qr", "pca_qr", "pca_mean_qr"):
            with self.assertRaises(ValueError):
                fixed_basis(np.ones((30, 12)), method, 4)
        with self.assertRaises(ValueError):
            select_fixed_basis(np.eye(10), "pod_qr", 4, 3)

    def test_mean_overlap_does_not_consume_two_dimensions(self):
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(30, 5)))
        time, _ = np.linalg.qr(np.column_stack((np.ones(12), rng.normal(size=(12, 5)))))
        centered = q @ np.diag([5.,4.,3.,2.,1.]) @ time[:,1:].T
        h = centered + 10*q[:, :1]
        b = fixed_basis(h, "pca_mean_qr", 4)
        np.testing.assert_allclose(b @ b.T, q[:,:4] @ q[:,:4].T, atol=1e-12)
        self.assertEqual(fixed_basis(h, "pca_mean_qr", 1).shape, (30,1))


if __name__ == "__main__":
    unittest.main()
