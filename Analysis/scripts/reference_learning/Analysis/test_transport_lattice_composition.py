import unittest
import numpy as np
import pandas as pd
from transport_lattice_composition import classify_transfer_state, dose_overlap_class, filter_registry_support, paired_mse_delta, _prediction_metrics

class TransportLatticeCompositionTests(unittest.TestCase):

    def test_transfer_state_keeps_unscored_and_zero_baseline_separate(self):
        self.assertEqual(classify_transfer_state('zero_eligible', 1.0, 0.5), 'zero_eligible')
        self.assertEqual(classify_transfer_state('scored', np.nan, 0.5), 'supported_unresolved_vs_zero')
        self.assertEqual(classify_transfer_state('scored', 4.0, 5.0), 'supported_positive_vs_zero')
        self.assertEqual(classify_transfer_state('scored', 6.0, 5.0), 'supported_negative_vs_zero')
        self.assertEqual(classify_transfer_state('support_threshold_unavailable', 4.0, 5.0), 'support_threshold_unavailable')

    def test_dose_overlap_class_is_set_based_and_missing_safe(self):
        self.assertEqual(dose_overlap_class('["0.05", "0.1"]', '["0.1", "0.05"]'), 'exact')
        self.assertEqual(dose_overlap_class('["0.05", "0.1"]', '["0.1", "0.2"]'), 'partial')
        self.assertEqual(dose_overlap_class('["0.05"]', '["0.2"]'), 'none')
        self.assertEqual(dose_overlap_class('', '["0.2"]'), 'unknown')

    def test_registry_support_filter_preserves_nested_thresholds_and_overlap(self):
        frame = pd.DataFrame({'source_n_partners': [1, 3, 5, 5], 'target_n_partners': [3, 3, 5, 5], 'shared_partner_count': [0, 0, 1, 0]})
        self.assertEqual(len(filter_registry_support(frame, 1, 'all')), 4)
        self.assertEqual(len(filter_registry_support(frame, 3, 'all')), 3)
        self.assertEqual(len(filter_registry_support(frame, 5, 'all')), 2)
        self.assertEqual(len(filter_registry_support(frame, 1, 'no_shared_partner')), 3)
        self.assertEqual(len(filter_registry_support(frame, 5, 'no_shared_partner')), 1)

    def test_paired_mse_delta_uses_only_common_finite_units(self):
        observed = pd.Series([1.0, 2.0, np.nan, 4.0])
        left = pd.Series([1.0, 4.0, 3.0, np.nan])
        right = pd.Series([2.0, 2.0, 3.0, 5.0])
        result = paired_mse_delta(observed, left, right)
        self.assertEqual(result['n_common'], 2)
        self.assertAlmostEqual(result['mse_left'], 2.0)
        self.assertAlmostEqual(result['mse_right'], 0.5)
        self.assertAlmostEqual(result['delta_left_minus_right'], 1.5)

    def test_prediction_metrics_collapses_repeated_unit_rows_before_error(self):
        frame = pd.DataFrame({'unit_identity': ['u1', 'u1', 'u2'], 'observed': [1.0, 3.0, 2.0], 'model': [1.0, 1.0, 4.0]})
        metrics = _prediction_metrics(frame, 'model', ['unit_identity'])
        self.assertEqual(metrics['n_units'], 2)
        self.assertAlmostEqual(metrics['mse'], 2.5)
if __name__ == '__main__':
    unittest.main()
