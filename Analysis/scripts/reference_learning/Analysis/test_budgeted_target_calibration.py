"""Tests for fixed-test budgets, leakage guards, and small-sample calibration."""
import unittest
import numpy as np
import pandas as pd
import budgeted_target_calibration as calibration

class BudgetedCalibrationTests(unittest.TestCase):

    def test_split_is_outcome_independent_and_leaves_three_per_side(self):
        frame = pd.DataFrame(dict(unit_identity=list('abcdefg'), target_coordinate_cost=[2, 3, 4, 5, 6, 7, 8], source_value=range(7), target_value=range(7)))
        before = calibration.fixed_split(frame, 's', 't', 0)
        frame.source_value = -1000
        frame.target_value = 999
        self.assertEqual(before, calibration.fixed_split(frame, 's', 't', 0))
        self.assertEqual(len(before[0]), 3)
        self.assertEqual(len(before[1]), 4)
        self.assertFalse(set(before[0]) & set(before[1]))

    def test_prefix_charges_union_and_stays_nested(self):
        supports = {'a': {('t', 0, 'c1'), ('t', 0, 'c2')}, 'b': {('t', 0, 'c2'), ('t', 0, 'c3')}, 'c': {('t', 1, 'c1')}}
        small, used = calibration.budget_prefix(list('abc'), supports, 3)
        large, used_large = calibration.budget_prefix(list('abc'), supports, 4)
        self.assertEqual(small, list('ab'))
        self.assertEqual(len(used), 3)
        self.assertEqual(large, list('abc'))
        self.assertTrue(used <= used_large)

    def test_constant_source_falls_back_to_calibration_mean(self):
        pred, parameters, diagnostics = calibration.predictions_for_models([0] * 6, [0] * 3, [0.1, -0.2, 0.4], [0] * 3)
        np.testing.assert_allclose(pred['source_affine_ridge'], pred['target_calibration_mean'])
        self.assertAlmostEqual(parameters['source_affine_ridge'][0], 0.1)
        self.assertEqual(parameters['source_affine_ridge'][1], 0.0)
        self.assertTrue(diagnostics['ridge_fallback_to_calibration_mean'])

    def test_unpenalized_intercept_and_fixed_slope_shrinkage(self):
        source = np.array([-1.0, 0.0, 1.0])
        pred, parameters, _ = calibration.predictions_for_models(source, source, 10 + 2 * source, source)
        self.assertAlmostEqual(parameters['source_affine_ridge'][0], 10.0)
        self.assertAlmostEqual(parameters['source_affine_ridge'][1], 1.0)
        np.testing.assert_allclose(pred['source_affine_ridge'], 10 + source)

    def test_negative_response_predictions_are_not_clipped(self):
        pred, _, _ = calibration.predictions_for_models([-1, 0, 1], [-1, 0, 1], [-2, -1, 0], [-1.5, 1.5])
        np.testing.assert_allclose(pred['source_offset'], [-2.5, 0.5])

    def test_calibration_guard_requires_three_observed_identities(self):
        with self.assertRaises(ValueError):
            calibration.predictions_for_models([0, 1], [0, 1], [0, 1], [2])

    def test_affine_source_scaling_uses_historical_source_units(self):
        a, _, _ = calibration.predictions_for_models([-2, -1, 0, 1, 2], [-1, 0, 1], [0.1, 0.3, 0.4], [-2, 2])
        b, _, _ = calibration.predictions_for_models([-20, -10, 0, 10, 20], [-10, 0, 10], [0.1, 0.3, 0.4], [-20, 20])
        np.testing.assert_allclose(a['source_affine_ridge'], b['source_affine_ridge'])

    def test_constant_calibration_source_is_valid_without_global_fallback(self):
        pred, parameters, diagnostics = calibration.predictions_for_models([0, 0, 0, 1, 2, 3], [0, 0, 0], [0.1, 0.2, 0.3], [1, 2, 3])
        np.testing.assert_allclose(pred['source_affine_ridge'], [0.2] * 3)
        self.assertAlmostEqual(parameters['source_affine_ridge'][1], 0.0)
        self.assertTrue(diagnostics['calibration_source_constant'])
        self.assertFalse(diagnostics['ridge_fallback_to_calibration_mean'])

    def test_exact_coordinate_overlap_rejects_design(self):
        rows = []
        for index, identity in enumerate('abcdef'):
            target = ['t', index if identity != 'd' else 0, 'c']
            rows.append(dict(source_screen='s', target_screen='t', source_family='sf', target_family='tf', relation='cross_family', unit_identity=identity, source_value=index / 10, target_value=index / 20, source_coordinates_json=calibration.dumps([['s', index, 'c']]), target_coordinates_json=calibration.dumps([target]), source_record_ids_json=calibration.dumps(['s' + identity]), target_record_ids_json=calibration.dumps(['t' + str(target[1])]), target_coordinate_cost=1))
        design = pd.DataFrame([dict(source_screen='s', target_screen='t', source_family='sf', target_family='tf', relation='cross_family', design=0, design_status='eligible_design', n_primary_identities=6, test_identities_json=calibration.dumps(list('abc')), calibration_pool_order_json=calibration.dumps(list('def')), all_source_zero=False, all_target_zero=False)])
        budgets, scores, predictions = calibration.build_budget_results(pd.DataFrame(rows), design)
        self.assertTrue(budgets.status.eq('operator_support_overlap').all())
        self.assertTrue(scores.mse.isna().all())
        self.assertTrue(predictions.empty)
if __name__ == '__main__':
    unittest.main()
