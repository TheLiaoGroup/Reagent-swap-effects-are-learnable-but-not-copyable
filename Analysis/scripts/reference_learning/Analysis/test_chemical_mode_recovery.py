"""Focused design and leakage checks for held-out chemical contrasts."""
import unittest
import numpy as np
import pandas as pd
from chemical_mode_recovery import contrast_prediction, contrast_risk, edge_panel, fit_clean_covariance, nested_quarter, summarize_edge_scores, training_predictors
from response_mode_ablation import rank1_diagonal_covariance
from shrinkage_covariance_analysis import fit_covariance

class ChemicalModeRecoveryTests(unittest.TestCase):

    def test_panel_is_outcome_independent_and_hides_both_endpoints(self):
        one = edge_panel('SNAr|1', 'edge', 20, (3, 7), [1])
        two = edge_panel('SNAr|1', 'edge', 20, (3, 7), [1])
        self.assertTrue(np.array_equal(one, two))
        self.assertFalse(set(one) & {1, 3, 7})
        self.assertEqual(len(one), 12)
        self.assertEqual(len(set(one)), 12)

    def test_quarter_is_deterministic_nested_and_excludes_test(self):
        pool = np.array([0, 2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 18, 20] * 2)
        pool = np.unique(pool)
        selected = nested_quarter(pool, 's', 'role1', 0, 1)
        self.assertEqual(len(selected), 8)
        self.assertTrue(set(selected) <= set(pool))
        self.assertTrue(np.array_equal(selected, nested_quarter(pool, 's', 'role1', 0, 1)))

    def test_prediction_receives_anchors_only_and_missing_falls_back(self):
        rng = np.random.default_rng(8)
        fit = fit_covariance(rng.uniform(size=(40, 18)))
        anchors = np.array([0, 2, 4, 6, 8, 10])
        values = rng.uniform(size=(7, len(anchors)))
        values[-1] = np.nan
        original = contrast_prediction(fit, values, anchors, (1, 3), 1)
        poisoned = rng.uniform(size=(7, 18))
        poisoned[:, anchors] = values
        poisoned[:, [1, 3]] = 1000000.0
        actual = contrast_prediction(fit, poisoned[:, anchors], anchors, (1, 3), 1)
        np.testing.assert_allclose(original[0], actual[0])
        self.assertAlmostEqual(original[0][-1], fit['mu'][1] - fit['mu'][3])
        self.assertAlmostEqual(original[1][-1], fit['mu'][1] - fit['mu'][3])

    def test_full_expected_risk_no_worse_and_rank1_can_recover_contrasts(self):
        rng = np.random.default_rng(19)
        v = rng.normal(size=14)
        cov = np.outer(v, v) + np.diag(rng.uniform(0.1, 0.6, 14))
        base = rank1_diagonal_covariance(cov)
        full_risk, base_risk, prior = contrast_risk(cov, base, np.arange(2, 8), (0, 1))
        self.assertLessEqual(full_risk, base_risk + 1e-10)
        self.assertLessEqual(full_risk, prior + 1e-10)
        fit = {'mu': np.zeros(14), 'cov': base}
        predicted, _ = contrast_prediction(fit, np.ones((2, 6)), np.arange(2, 8), (0, 1), 1)
        self.assertGreater(np.max(np.abs(predicted)), 1e-06)

    def test_training_predictors_do_not_need_test_outcomes(self):
        rng = np.random.default_rng(27)
        train = rng.uniform(size=(40, 8))
        edges = pd.DataFrame({'condition_a_index': [0, 2], 'condition_b_index': [1, 3], 'orientation': [1, 1], 'swap_key': ['swap', 'swap']})
        roles = np.array([[f'a{i // 2}', f'b{i % 7}'] for i in range(len(train))])
        result = training_predictors(train, edges, roles, 'key')
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isfinite(result[0]['nonleading_contrast_share']))
        self.assertEqual(result[0]['n_other_contexts'], 1)

    def test_ambiguous_column_does_not_affect_modes(self):
        rng = np.random.default_rng(81)
        train = rng.uniform(size=(40, 8))
        original = fit_clean_covariance(train, [3])
        train[:, 3] = 1000000.0 * rng.normal(size=len(train))
        poisoned = fit_clean_covariance(train, [3])
        np.testing.assert_array_equal(original['cov'], poisoned['cov'])
        np.testing.assert_array_equal(original['mu'], poisoned['mu'])

    def test_screen_aggregation_equalizes_contexts(self):
        records = []
        for edge, context, mse in [('a', 'x', 1.0), ('b', 'x', 3.0), ('c', 'y', 10.0)]:
            records.append(dict(screen='s', family='f', split='pair', fraction=1.0, k=12, edge_id=edge, context_id=context, n=2, full_raw_sse=2 * mse, rank1_raw_sse=4 * mse, marginal_sse=6 * mse, zero_sse=8 * mse))
        _, summary = summarize_edge_scores(pd.DataFrame(records))
        self.assertAlmostEqual(summary.iloc[0].full_raw_mse, 6.0)
        self.assertAlmostEqual(summary.iloc[0].extra_gain_raw, 0.5)
if __name__ == '__main__':
    unittest.main()
