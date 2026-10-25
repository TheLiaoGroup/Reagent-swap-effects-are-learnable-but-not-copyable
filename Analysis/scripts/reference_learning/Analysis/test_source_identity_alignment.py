"""Identity-specific alignment, fixed mappings and same-cost controls."""
from __future__ import annotations
import importlib
import importlib.util
import itertools
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class SourceIdentityAlignmentTests(unittest.TestCase):

    def producer(self):
        self.assertIsNotNone(importlib.util.find_spec('source_identity_alignment'), 'producer is not implemented')
        return importlib.import_module('source_identity_alignment')

    def test_exact_test_reference_matches_all_permutations_and_covariance(self):
        r = self.producer()
        x, y = (np.array([-0.3, 0.1, 0.6]), np.array([0.2, -0.1, 0.5]))
        result = r.exact_test_reference(y, x, -0.04, 0.7)
        enumerated = np.mean([np.mean((y - (-0.04 + 0.7 * np.array(permutation))) ** 2) for permutation in itertools.permutations(x)])
        self.assertAlmostEqual(result['test_permuted_exact_mse'], enumerated)
        self.assertAlmostEqual(result['delta_test'], 2 * 0.7 * np.mean((x - x.mean()) * (y - y.mean())))

    def test_zero_slope_and_constant_source_are_neutral(self):
        r = self.producer()
        y = np.array([0.2, -0.1, 0.4])
        self.assertAlmostEqual(r.exact_test_reference(y, np.array([-0.3, 0.1, 0.6]), 0.1, 0.0)['delta_test'], 0.0)
        self.assertAlmostEqual(r.exact_test_reference(y, np.full(3, 0.3), -0.1, 2.0)['delta_test'], 0.0)

    def test_test_alignment_can_be_positive_without_net_gain(self):
        r = self.producer()
        x = y = np.array([-0.1, 0.0, 0.1])
        result = r.exact_test_reference(y, x, 1.0, 1.0)
        net = np.mean(y ** 2) - result['real_mse']
        self.assertGreater(result['delta_test'], 0.0)
        self.assertLess(net, 0.0)

    def test_offset_alignment_is_invariant_to_calibration_intercept(self):
        r = self.producer()
        x, y = (np.array([-0.2, 0.0, 0.4]), np.array([-0.1, 0.2, 0.1]))
        first = r.exact_test_reference(y, x, -0.4, 1.0)
        second = r.exact_test_reference(y, x, 0.8, 1.0)
        self.assertAlmostEqual(first['delta_test'], second['delta_test'])
        self.assertNotAlmostEqual(first['real_mse'], second['real_mse'])

    def test_hash_mappings_are_unique_nonidentity_and_outcome_free(self):
        r = self.producer()
        identities = ['a', 'b', 'c', 'd', 'e', 'f']
        first = r.fixed_mappings('S|1', 'T|1', identities)
        second = r.fixed_mappings('S|1', 'T|1', identities)
        self.assertEqual(first, second)
        permutations = [tuple(row['permutation']) for row in first]
        self.assertEqual(len(permutations), 128)
        self.assertEqual(len(set(permutations)), 128)
        self.assertNotIn(tuple(range(6)), permutations)
        self.assertTrue(all((sorted(permutation) == list(range(6)) for permutation in permutations)))
        with self.assertRaises(TypeError):
            r.fixed_mappings('S|1', 'T|1', identities, target_values=np.arange(6))

    def test_small_identity_support_cannot_fabricate_128_mappings(self):
        r = self.producer()
        with self.assertRaisesRegex(ValueError, 'at least six'):
            r.fixed_mappings('S', 'T', list('abcde'))

    def test_repeated_source_vectors_keep_all_identity_mappings(self):
        r = self.producer()
        mappings = r.fixed_mappings('S', 'T', list('abcdef'))
        assigned = [tuple(np.zeros(6)[row['permutation']]) for row in mappings]
        self.assertEqual(len(mappings), 128)
        self.assertEqual(len(set(assigned)), 1)

    def test_batched_ridge_matches_standardized_sklearn_and_scale_invariance(self):
        r = self.producer()
        all_x = np.array([-0.2, 0.3, 0.7, -0.5, 0.1, 0.4])
        xc, yc, xt = (all_x[:3], np.array([0.2, -0.1, 0.4]), all_x[3:])
        result = r.fit_batch(all_x, xc[None, :], yc, xt[None, :])
        fitted = Ridge(alpha=3.0, fit_intercept=True, solver='svd').fit(((xc - all_x.mean()) / all_x.std())[:, None], yc)
        expected = fitted.predict(((xt - all_x.mean()) / all_x.std())[:, None])
        np.testing.assert_allclose(result['source_affine_ridge']['prediction'][0], expected, atol=1e-14)
        scaled = r.fit_batch(7 * all_x, (7 * xc)[None, :], yc, (7 * xt)[None, :])
        np.testing.assert_allclose(result['source_affine_ridge']['prediction'], scaled['source_affine_ridge']['prediction'], atol=1e-14)

    def test_source_zero_affine_falls_back_and_fit_cannot_read_test_targets(self):
        r = self.producer()
        result = r.fit_batch(np.zeros(6), np.zeros((2, 3)), np.array([0.1, 0.2, 0.3]), np.zeros((2, 3)))
        np.testing.assert_allclose(result['source_affine_ridge']['prediction'], 0.2)
        np.testing.assert_array_equal(result['source_affine_ridge']['slope'], 0.0)
        with self.assertRaises(TypeError):
            r.fit_batch(np.zeros(6), np.zeros((1, 3)), np.zeros(3), np.zeros((1, 3)), test_y=np.ones(3))

    def test_direction_weighting_is_not_pooled_design_weighting(self):
        r = self.producer()
        rows = []
        for source, n, value in (('A', 10, 1.0), ('B', 2, 3.0)):
            for design in range(10):
                row = dict(source_screen=source, target_screen='T', source_family=source, target_family='T', relation='cross_family', model='source_affine_ridge', design=design, budget=24, status='scored' if design < n else 'budget_insufficient', b24_scored_domain=design < n, n_test=3, n_calibration=3, actual_spent=12)
                row.update({metric: value if design < n else np.nan for metric in r.PRIMARY_METRICS})
                rows.append(row)
        directions, headline = r.aggregate_designs(pd.DataFrame(rows), secondary=False)
        result = headline[headline.domain.eq('all_scored')].iloc[0]
        self.assertAlmostEqual(result.real_mse, 2.0)
        self.assertEqual(result.n_scored_designs, 12)
if __name__ == '__main__':
    unittest.main()
