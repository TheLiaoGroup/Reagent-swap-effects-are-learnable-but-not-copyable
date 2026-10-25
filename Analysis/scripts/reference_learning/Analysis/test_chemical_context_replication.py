"""Focused checks for direction, held-out nuisance fits and shared exclusions."""
import unittest
import numpy as np
import pandas as pd
from chemical_context_replication import context_mean_metrics, corr, crossfit_residual, enrich_edges, other_activity, split_half

class ChemicalContextReplicationTests(unittest.TestCase):

    def test_orientation_is_invariant_to_endpoint_order(self):
        base = dict(screen='x|1', reaction_class='x', contrast_category='Reagent1', contrast_type='identity_swap', fixed_equiv='1', fixed_core_fields_json='{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14}', changed_fields_json='["Reagent1_id"]', condition_a='C1', condition_b='C2', value_a='CMP2', value_b='CMP1', name_a='high', name_b='low')
        reversed_row = dict(base, value_a='CMP1', value_b='CMP2', name_a='low', name_b='high', condition_a='C2', condition_b='C1')
        a, b = (enrich_edges(pd.DataFrame([base])), enrich_edges(pd.DataFrame([reversed_row])))
        self.assertEqual(a.swap_key.iloc[0], b.swap_key.iloc[0])
        self.assertEqual(a.orientation.iloc[0], -b.orientation.iloc[0])
        self.assertEqual(a.context_id.iloc[0], b.context_id.iloc[0])

    def test_nuisance_fit_excludes_evaluated_half(self):
        g = np.linspace(0, 1, 40)
        halves = (np.arange(40) < 20, np.arange(40) >= 20)
        delta = 0.1 + 2 * g
        out = crossfit_residual(delta, g, halves, np.ones(40, bool))
        self.assertLess(np.max(abs(out)), 1e-12)
        altered = delta.copy()
        altered[halves[1]] += np.sin(np.arange(20))
        changed = crossfit_residual(altered, g, halves, np.ones(40, bool))
        np.testing.assert_allclose((altered - changed)[halves[1]], delta[halves[1]], atol=1e-12)

    def test_all_four_targets_excluded_from_activity(self):
        m = np.arange(120, dtype=float).reshape(20, 6)
        g, _ = other_activity(m, [0, 1, 3, 5])
        modified = m.copy()
        modified[:, [0, 1, 3, 5]] = -999
        changed, _ = other_activity(modified, [0, 1, 3, 5])
        np.testing.assert_array_equal(g, changed)
        np.testing.assert_array_equal(g, m[:, [2, 4]].mean(axis=1))

    def test_group_splits_and_centering(self):
        labels = np.repeat(['a', 'b', 'c', 'd'], [5, 7, 10, 12])
        a, b = split_half(labels, np.random.default_rng(2))
        self.assertFalse(set(labels[a]) & set(labels[b]))
        x = np.linspace(-1, 1, 30)
        self.assertAlmostEqual(corr(x + 20, x - 10), 1)
        self.assertTrue(np.isnan(corr(np.ones(30), np.ones(30))))

    def test_unstable_mean_is_penalized_on_held_out_rows(self):
        halves = (np.arange(40) < 20, np.arange(40) >= 20)
        delta = np.r_[np.ones(20), -np.ones(20)][:, None]
        summary = context_mean_metrics(pd.DataFrame({'context_id': ['context']}), delta, halves)
        self.assertEqual(summary['context_mean_sign_agreement'], 0)
        self.assertEqual(summary['context_equal_transferred_mean_gain'], -3)
if __name__ == '__main__':
    unittest.main()
