"""Scientific channel invariants for report66."""
import importlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from chemical_mode_recovery import contrast_prediction

def producer():
    path = Path(__file__).with_name('chemical_observation_channels.py')
    assert path.exists(), 'report66 producer has not been implemented'
    return importlib.import_module('chemical_observation_channels')

def example_fit():
    loading = np.array([[0.8, 0.1, 0.2, 0.4], [0.1, 0.9, 0.2, 0.1], [0.5, 0.2, 0.8, 0.1], [0.1, 0.7, 0.1, 0.9]])
    return dict(mu=np.array([0.2, 0.5, 0.3, 0.6]), cov=loading @ loading.T + np.eye(4) * 0.2)

class ChemicalObservationChannelsTests(unittest.TestCase):

    def test_channel_transform_centers_and_orients(self):
        mod = producer()
        actual = mod.channel_coordinates(np.array([[0.7, 0.2]]), np.array([0.3, 0.4]), -1)
        np.testing.assert_allclose(actual, [[0.1, -0.6]])

    def test_sd_is_original_full_on_complete_anchors_and_clip_endpoints(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.7, 0.2], [10.0, -10.0], [0.3, 0.6]])
        actual, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], -1, -1)
        raw, clip = contrast_prediction(fit, values, [2, 3], [0, 1], -1)
        for family in ('frozen', 'optimal'):
            np.testing.assert_allclose(actual[f'full_{family}_sd_raw'], raw, atol=1e-13)
            np.testing.assert_allclose(actual[f'full_{family}_sd_clip'], clip, atol=1e-13)
        self.assertLess(abs(clip[1]), abs(raw[1]))

    def test_swapping_reference_endpoints_and_orientation_preserves_predictions(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.7, 0.2], [0.4, 0.8]])
        first, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, -1)
        second, _ = mod.channel_predictions(fit, fit['cov'], values[:, ::-1], [3, 2], [0, 1], 1, 1)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=1e-13)

    def test_reversing_target_orientation_reverses_every_channel(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.7, 0.2], [0.4, 0.8]])
        first, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, 1)
        second, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], -1, 1)
        for key in first:
            np.testing.assert_allclose(first[key], -second[key], atol=1e-13)

    def test_random_difference_sign_is_only_a_coordinate_choice(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.7, 0.2], [0.4, 0.8]])
        first, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, 1)
        flipped, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, -1)
        reordered, _ = mod.channel_predictions(fit, fit['cov'], values[:, ::-1], [3, 2], [0, 1], 1, 1)
        for key in first:
            np.testing.assert_allclose(first[key], flipped[key], atol=1e-13)
            np.testing.assert_allclose(first[key], reordered[key], atol=1e-13)

    def test_both_endpoints_required_for_every_channel(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[np.nan, 0.2], [0.7, np.nan], [np.nan, np.nan]])
        actual, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, 1)
        for prediction in actual.values():
            np.testing.assert_allclose(prediction, fit['mu'][0] - fit['mu'][1])
        bridge, _ = contrast_prediction(fit, values, [2, 3], [0, 1], 1)
        self.assertFalse(np.allclose(bridge[:2], actual['full_optimal_sd_raw'][:2]))

    def test_frozen_sd_sums_centered_s_and_d_contributions(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.7, 0.2], [0.4, 0.8]])
        actual, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, 1)
        marginal = fit['mu'][0] - fit['mu'][1]
        for model in ('full', 'rank1'):
            np.testing.assert_allclose(actual[f'{model}_frozen_sd_raw'], actual[f'{model}_frozen_s_raw'] + actual[f'{model}_frozen_d_raw'] - marginal)

    def test_optimal_scalar_uses_scalar_covariance_not_frozen_joint_weight(self):
        mod = producer()
        fit = example_fit()
        values = np.array([[0.8, 0.4]])
        actual, _ = mod.channel_predictions(fit, fit['cov'], values, [2, 3], [0, 1], 1, 1)
        transform = np.array([[0.5, 0.5], [1.0, -1.0]])
        covz = transform @ fit['cov'][2:, 2:] @ transform.T
        cross = (fit['cov'][0, 2:] - fit['cov'][1, 2:]) @ transform.T
        z = mod.channel_coordinates(values, fit['mu'][2:], 1)[0]
        marginal = fit['mu'][0] - fit['mu'][1]
        self.assertAlmostEqual(actual['full_optimal_s_raw'][0], marginal + z[0] * cross[0] / covz[0, 0])
        self.assertNotAlmostEqual(actual['full_optimal_s_raw'][0], actual['full_frozen_s_raw'][0])

    def test_covariance_degeneracy_is_finite_and_audited(self):
        mod = producer()
        fit = dict(mu=np.array([0.2, 0.5, 0.3, 0.6]), cov=np.zeros((4, 4)))
        actual, audit = mod.channel_predictions(fit, fit['cov'], np.array([[0.7, 0.2]]), [2, 3], [0, 1], 1, 1)
        for value in actual.values():
            np.testing.assert_allclose(value, [-0.3])
        self.assertTrue(all((row['joint_pinv'] for row in audit)))
        self.assertTrue(all((row['zero_s_variance'] and row['zero_d_variance'] for row in audit)))

    def test_masks_are_inherited_11_and_10_and_shared_by_modes(self):
        mod = producer()
        values = np.ones((3, 2))
        other = values.copy()
        other[1, 0] = np.nan
        random = [values.copy() for _ in range(8)]
        random[0][2, 0] = np.nan
        strict = mod.domain_masks(values, values, other, random, 'strict_641')
        relaxed = mod.domain_masks(values, values, other, random, 'relaxed_643')
        np.testing.assert_array_equal(strict['primary'], [True, True, True])
        np.testing.assert_array_equal(strict['all_panels_complete'], [True, False, False])
        np.testing.assert_array_equal(relaxed['all_panels_complete'], [True, True, False])

    def test_increment_specificity_is_difference_of_paired_gains(self):
        mod = producer()
        row = dict(semantic_full_optimal_s_raw_mse=10.0, semantic_full_optimal_sd_raw_mse=7.0, different_other_full_optimal_s_raw_mse=11.0, different_other_full_optimal_sd_raw_mse=10.0)
        actual = mod.channel_derivatives(pd.DataFrame([row])).iloc[0]
        self.assertEqual(actual.semantic_full_optimal_d_given_s_raw_mse, 3.0)
        self.assertEqual(actual.different_other_full_optimal_d_given_s_raw_mse, 1.0)
        self.assertEqual(actual.other_full_optimal_d_given_s_specificity_raw_mse, 2.0)
        self.assertEqual(actual.other_full_optimal_d_given_s_specificity_raw_mse, actual.other_full_optimal_sd_specificity_raw_mse - actual.other_full_optimal_s_specificity_raw_mse)

    def test_disjoint_hidden_endpoints_required(self):
        mod = producer()
        fit = example_fit()
        with self.assertRaises(ValueError):
            mod.channel_predictions(fit, fit['cov'], np.array([[0.7, 0.2]]), [0, 3], [0, 1], 1, 1)
if __name__ == '__main__':
    unittest.main()
