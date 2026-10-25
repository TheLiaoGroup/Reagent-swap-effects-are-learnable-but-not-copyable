"""Estimand and isolation tests for controlled molecular bridges."""
import unittest
import numpy as np
import pandas as pd
import controlled_cross_library_bridges as b

class ControlledBridgeTests(unittest.TestCase):

    def test_equal_partner_then_context_balance(self):
        cells = pd.DataFrame(dict(sub_1=['M', 'M', 'M'], sub_2=['A', 'A', 'B'], delta=[0.0, 1.0, 1.0], contrast_id=['c1', 'c2', 'c1']))
        pairs = b.balance_pairs(cells)
        value = pairs.groupby('sub_1').delta.mean().loc['M']
        self.assertAlmostEqual(value, 0.75)
        self.assertNotAlmostEqual(value, cells.delta.mean())

    def test_leave_molecule_out_marginal(self):
        values = np.array([0.9, 0.1, -0.1])
        observed, available = b.leave_one_unit_out(values)
        np.testing.assert_allclose(observed, [0.0, 0.4, 0.5])
        self.assertTrue(available.all())
        fallback, available = b.leave_one_unit_out(np.array([0.9]))
        self.assertFalse(available[0])
        self.assertAlmostEqual(fallback[0], 0.9)

    def test_family_pooling_equalizes_screens(self):
        frame = pd.DataFrame(dict(source_family=['BH', 'BH', 'X'], source_screen=['a', 'b', 'c'], predicted=[0.0, 0.0, 1.0]))
        self.assertAlmostEqual(b.family_source_mean(frame, 'predicted'), 0.5)

    def test_source_scale_ignores_target(self):
        frame = pd.DataFrame(dict(source_family=['A', 'B'], source_screen=['a', 'b'], source_yield_sd=[0.2, 0.4], target_yield_sd=[100.0, 200.0]))
        expected = np.sqrt(0.1)
        self.assertAlmostEqual(b.pooled_source_scale(frame), expected)
        frame.target_yield_sd = 0
        self.assertAlmostEqual(b.pooled_source_scale(frame), expected)

    def test_bh_family_and_no_target_family_sources(self):
        self.assertEqual(b.transformation('Buchwald-Hartwig(CMC)|2'), 'Buchwald-Hartwig')
        self.assertFalse(b.source_relation_allowed('BH', 'BH', 'cross_family'))
        self.assertTrue(b.source_relation_allowed('BH', 'BH', 'within_family'))

    def test_unique_full_pairs_have_one_basis(self):
        data = pd.DataFrame(dict(sub_1=['M', 'N'], sub_2=['P', 'Q'], delta=[0.1, -0.2], n_contexts=[1, 1]))
        result = b.full_pair_units(data)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.basis.eq('full_pair').all())
        self.assertEqual(result.unit_identity.iloc[0], '["M","P"]')

    def test_zero_eligible_score_is_missing(self):
        result = b.error_summary(pd.DataFrame(), 'source_conditioned')
        self.assertEqual(result['status'], 'zero_eligible')
        self.assertEqual(result['n_prediction_units'], 0)
        self.assertTrue(np.isnan(result['family_macro_rmse_pp']))
if __name__ == '__main__':
    unittest.main()
