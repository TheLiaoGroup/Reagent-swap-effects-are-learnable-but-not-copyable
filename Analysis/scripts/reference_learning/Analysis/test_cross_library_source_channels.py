"""Behavioral contracts for the fixed report70 source-channel design."""
from fractions import Fraction
from itertools import permutations
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
try:
    import cross_library_source_channels as producer
except ModuleNotFoundError:
    producer = None

class CrossLibrarySourceChannelTests(unittest.TestCase):

    def implementation(self):
        self.assertIsNotNone(producer, 'report70 source-channel implementation is absent')
        return producer

    def test_positive_atom_endpoints_preserve_cancelled_common_cost(self):
        module = self.implementation()
        atoms = {('x', 0): Fraction(1, 2), ('y', 0): Fraction(1, 2)}
        contexts = {'x': ('A', 'a', 'b'), 'y': ('A', 'b', 'c')}
        maps = module.expand_channels(atoms, contexts)
        self.assertEqual(maps['d'], {('A', 0, 'a'): Fraction(-1, 2), ('A', 0, 'c'): Fraction(1, 2)})
        self.assertEqual(maps['s']['A', 0, 'b'], Fraction(1, 2))
        self.assertEqual(sum(maps['s'].values()), 1)
        self.assertEqual(set(maps['s']), set(maps['low']) | set(maps['high']))

    def test_collapsed_original_endpoint_multiplicity_is_retained(self):
        module = self.implementation()
        endpoints = {'u': {('x', 0): Fraction(1)}, 'v': {('y', 0): Fraction(1)}}
        self.assertEqual(module.collapse_atoms(['u', 'u', 'v'], endpoints), {('x', 0): Fraction(2, 3), ('y', 0): Fraction(1, 3)})

    def test_invalid_or_non_normalized_atom_weights_are_rejected(self):
        module = self.implementation()
        for atoms in ({('x', 0): Fraction(-1)}, {('x', 0): Fraction(1, 2)}):
            with self.assertRaises(ValueError):
                module.expand_channels(atoms, {'x': ('A', 'a', 'b')})

    def test_d_only_matches_original_standardized_alpha3_svd_ridge(self):
        module = self.implementation()
        values = np.array([[0.1, -0.2], [0.3, 0.1], [0.2, 0.5], [0.7, -0.1], [0.5, 0.3]])
        target = np.array([0.2, -0.1, 0.4])
        fitted = module.ridge_predict(values, values[:3], target, values[3:], 'd')
        x = values[:, 1]
        estimator = Ridge(alpha=3.0, fit_intercept=True, solver='svd').fit(((x[:3] - x.mean()) / x.std())[:, None], target)
        expected = estimator.predict(((x[3:] - x.mean()) / x.std())[:, None])
        np.testing.assert_allclose(fitted['prediction'], expected, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(fitted['coef_d'], estimator.coef_[0] / x.std())
        self.assertEqual(fitted['coef_s'], 0.0)

    def test_constant_channel_drops_without_losing_other_channel(self):
        module = self.implementation()
        values = np.column_stack([np.ones(6) * 0.4, np.arange(6) / 10])
        y = np.array([0.1, -0.2, 0.3])
        joint = module.ridge_predict(values, values[:3], y, values[3:], 'sd')
        difference = module.ridge_predict(values, values[:3], y, values[3:], 'd')
        np.testing.assert_allclose(joint['prediction'], difference['prediction'], atol=1e-14)
        self.assertEqual(joint['coef_s'], 0.0)
        self.assertEqual(joint['n_active_features'], 1)

    def test_batched_joint_refits_match_individual_two_feature_svd_models(self):
        module = self.implementation()
        values = np.array([[0.1, -0.2], [0.3, 0.1], [0.2, 0.5], [0.7, -0.1], [0.5, 0.3], [0.6, 0.2]])
        orders = np.array([[0, 1, 2, 3, 4, 5], [5, 2, 4, 1, 0, 3], [3, 1, 5, 0, 2, 4]])
        assigned = values[orders]
        target = np.array([0.2, -0.1, 0.4])
        batch = module.ridge_batch(values, assigned[:, :3], target, assigned[:, 3:], 'sd')
        mean, sd = (values.mean(axis=0), values.std(axis=0))
        for draw in range(len(orders)):
            estimator = Ridge(alpha=3.0, fit_intercept=True, solver='svd').fit((assigned[draw, :3] - mean) / sd, target)
            expected = estimator.predict((assigned[draw, 3:] - mean) / sd)
            np.testing.assert_allclose(batch['prediction'][draw], expected, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose([batch['coef_s'][draw], batch['coef_d'][draw]], estimator.coef_ / sd, rtol=1e-13, atol=1e-13)
            self.assertAlmostEqual(batch['intercept'][draw], estimator.intercept_ - np.sum(estimator.coef_ * mean / sd))

    def test_all_constant_channels_equal_same_target_calibration_mean(self):
        module = self.implementation()
        values = np.ones((6, 2)) * 0.2
        fitted = module.ridge_predict(values, values[:3], [0.1, 0.2, -0.1], values[3:], 'sd')
        np.testing.assert_allclose(fitted['prediction'], np.mean([0.1, 0.2, -0.1]))
        self.assertEqual(fitted['n_active_features'], 0)

    def test_three_calibration_identities_suffice_for_two_feature_ridge(self):
        module = self.implementation()
        values = np.array([[0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.5, 0.2]])
        self.assertTrue(np.isfinite(module.ridge_predict(values, values[:3], [0.1, 0.2, 0.4], values[3:], 'sd')['prediction']).all())
        with self.assertRaises(ValueError):
            module.ridge_predict(values, values[:2], [0.1, 0.2], values[2:], 'sd')

    def test_joint_permutation_preserves_feature_pairing(self):
        module = self.implementation()
        values = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]])
        reassigned = module.joint_reassign(values, [2, 0, 1])
        np.testing.assert_array_equal(reassigned[:, 1], 2 * reassigned[:, 0])
        with self.assertRaises(ValueError):
            module.joint_reassign(values, [0, 0, 1])

    def test_exact_test_reference_matches_all_joint_permutations(self):
        module = self.implementation()
        y, prediction = (np.array([0.2, -0.1, 0.4]), np.array([0.1, 0.3, -0.2]))
        result = module.exact_reference(y, prediction)
        expected = np.mean([np.mean((y - prediction[list(order)]) ** 2) for order in permutations(range(3))])
        self.assertAlmostEqual(result['test_permuted_exact_mse'], expected)
        self.assertAlmostEqual(result['delta_test'], 2 * np.mean((y - y.mean()) * (prediction - prediction.mean())))
        self.assertLess(abs(result['delta_test_covariance_identity_error']), 1e-14)

    def test_support_checks_include_source_common_union(self):
        module = self.implementation()
        report = module.support_overlaps({1, 2}, {3}, {4}, {5})
        self.assertTrue(all((value == 0 for value in report.values())))
        self.assertEqual(module.support_overlaps({1, 2}, {2}, {4}, {5})['source_cal_test_overlap'], 1)

    def test_direction_macro_preserves_equal_design_weight(self):
        module = self.implementation()
        rows = []
        for source, losses in (('A', [0.0, 2.0]), ('B', [5.0])):
            for design, loss in enumerate(losses):
                rows.append(dict(source_screen=source, target_screen='T', source_family=source, target_family='T', relation='cross_family', budget=24, design=design, channel='s', status='scored', b24_scored_domain=True, real_mse=loss, target_mean_mse=7.0, test_permuted_exact_mse=loss, delta_test=0.0, net_gain=7 - loss, actual_spent=8, n_calibration=3, n_test=3))
        directions, headline = module.aggregate(pd.DataFrame(rows))
        main = headline.query("domain == 'b24_fixed' and macro == 'direction_macro'").iloc[0]
        self.assertEqual(main.real_mse, 3.0)
        self.assertEqual(len(directions.query("domain == 'b24_fixed'")), 2)
if __name__ == '__main__':
    unittest.main()
