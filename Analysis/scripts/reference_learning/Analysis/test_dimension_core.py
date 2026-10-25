"""Scientific failure modes for anchor-based matrix reconstruction."""
import unittest
import numpy as np

class DimensionCoreTests(unittest.TestCase):

    def test_core_exists(self):
        from pathlib import Path
        self.assertTrue(Path(__file__).with_name('dimension_core.py').exists())

    def test_missing_and_zero_remain_distinct(self):
        from dimension_core import centred
        m = np.array([[0.0, 1.0, np.nan], [0.0, 0.0, 0.0]])
        r = centred(m, 'row')
        self.assertTrue(np.isnan(m[0, 2]))
        np.testing.assert_allclose(r, [[-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])

    def test_known_rank_and_independent_reference(self):
        from dimension_core import spectrum
        rng = np.random.default_rng(7)
        m = rng.normal(size=(100, 1)) @ rng.normal(size=(1, 40))
        self.assertEqual(spectrum(m, 'interaction')['k_var90'], 1)
        self.assertGreater(spectrum(rng.normal(size=(100, 40)), 'interaction')['k_var90'], 15)

    def test_constant_training_matrix_has_no_latent_rank(self):
        from dimension_core import fit_basis
        self.assertEqual(fit_basis(np.full((10, 8), 0.5))['rank'], 0)

    def test_constant_predictions_get_zero_ordering_credit(self):
        from run_existing_evidence import score_predictions
        y = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]])
        pred = np.full_like(y, 0.25)
        rho = score_predictions(y, pred, np.array([0]))[3]
        self.assertEqual(rho[0], 0.0)
        self.assertTrue(np.isnan(rho[1]))

    def test_hidden_targets_cannot_change_prediction(self):
        from dimension_core import reconstruct
        rng = np.random.default_rng(7)
        train = rng.uniform(size=(80, 12))
        test = rng.uniform(size=(3, 12))
        a = np.array([0, 3, 6, 9])
        hidden = np.setdiff1d(np.arange(12), a)
        changed = test.copy()
        changed[:, hidden] = -1000
        for method in ('additive', 'affine', 'low_rank'):
            np.testing.assert_allclose(reconstruct(train, test, a, method), reconstruct(train, changed, a, method))

    def test_affine_recovers_new_row_from_anchors(self):
        from dimension_core import reconstruct
        sig = np.linspace(0.1, 0.6, 20)
        train = np.array([0.02 + a * sig for a in np.linspace(0.3, 1.0, 60)])
        test = np.array([0.04 + 0.9 * sig])
        pred = reconstruct(train, test, np.array([0, 9, 19]), 'affine')
        np.testing.assert_allclose(pred, test, atol=1e-05)

    def test_selector_has_no_duplicates(self):
        from dimension_core import select_panel
        rng = np.random.default_rng(4)
        m = rng.uniform(size=(50, 20))
        for method in ('variance_greedy', 'high_mean', 'random'):
            p = select_panel(m, 12, method, rng)
            self.assertEqual(len(set(p)), 12)
            self.assertTrue(all((0 <= i < 20 for i in p)))

    def test_fixed_targets_are_identical_across_anchor_panels(self):
        from run_existing_evidence import score_predictions
        y = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        pred = np.zeros_like(y)
        targets = np.array([4, 5])
        first = score_predictions(y, pred, np.array([0, 1]), targets=targets)
        second = score_predictions(y, pred, np.array([2, 3]), targets=targets)
        np.testing.assert_allclose(first[1], [0.25 + 0.36])
        np.testing.assert_allclose(first[1], second[1])

    def test_low_rank_recovers_structured_new_rows(self):
        from dimension_core import reconstruct
        rng = np.random.default_rng(91)
        v = rng.normal(size=30)
        train = 0.5 + rng.normal(0, 0.04, (100, 1)) * v
        test = 0.5 + rng.normal(0, 0.04, (10, 1)) * v
        a = np.array([0, 3, 8, 15, 21, 28])
        pred = reconstruct(train, test, a, 'low_rank')
        self.assertLess(np.sqrt(np.mean((pred - test) ** 2)), 0.01)
if __name__ == '__main__':
    unittest.main()
