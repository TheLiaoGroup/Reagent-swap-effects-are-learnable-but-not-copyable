"""Design, leakage, availability, and weighting checks for report63."""
import importlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

def producer():
    path = Path(__file__).with_name('chemically_aligned_measurements.py')
    assert path.exists(), 'report63 producer has not been implemented'
    return importlib.import_module('chemically_aligned_measurements')

def metadata():
    rows = []
    for edge, a, b, context, swap in (('target', 0, 1, 'c1', 'swap'), ('reference', 2, 3, 'c2', 'swap'), ('shared', 0, 4, 'c3', 'swap'), ('other', 5, 6, 'c4', 'different')):
        rows.append(dict(screen='s|1', family='s', edge_id=edge, context_id=context, swap_key=swap, condition_a_index=a, condition_b_index=b, condition_a=f'C{a}', condition_b=f'C{b}', orientation=-1, category='Reagent1', contrast_type='identity_swap', dose='1'))
    return pd.DataFrame(rows)

class ChemicallyAlignedMeasurementsTests(unittest.TestCase):

    def test_reference_selection_is_metadata_only_disjoint_and_order_invariant(self):
        mod = producer()
        edges = metadata()
        conditions = {'s|1': np.array([f'C{i}' for i in range(12)])}
        first, _ = mod.freeze_registry(edges, conditions, {'s|1': [11]})
        second, _ = mod.freeze_registry(edges.sample(frac=1.0, random_state=4), conditions, {'s|1': [11]})
        first, second = (first.set_index('edge_id'), second.set_index('edge_id'))
        self.assertEqual(first.loc['target', 'reference_edge_id'], 'reference')
        self.assertEqual(first.loc['other', 'eligible'], 0)
        self.assertEqual(first.loc['target', 'random_panels_indices_json'], second.loc['target', 'random_panels_indices_json'])
        panels = json.loads(first.loc['target', 'random_panels_indices_json'])
        self.assertEqual(len(panels), 8)
        for panel in panels:
            self.assertEqual(len(set(panel)), 2)
            self.assertFalse(set(panel) & {0, 1, 11})

    def test_own_orientation_is_applied_to_each_reference(self):
        mod = producer()
        matrix = np.array([[0.1, 0.8, 0.7, 0.2]])
        target = mod.oriented_difference(matrix, 0, 1, -1)
        reference = mod.oriented_difference(matrix, 2, 3, 1)
        np.testing.assert_allclose(target, [0.7])
        np.testing.assert_allclose(reference, [0.5])

    def test_direct_rules_use_joint_train_offset_and_missing_reference_fallback(self):
        mod = producer()
        target = np.array([0.4, 0.5, 0.6, 0.9, np.nan])
        reference = np.array([0.1, 0.2, 0.3, np.nan, 0.7])
        raw, offset, audit = mod.direct_transport(target, reference, np.array([0.2, np.nan]), 0.05)
        np.testing.assert_allclose(raw, [0.2, 0.05])
        np.testing.assert_allclose(offset, [0.5, 0.05])
        self.assertEqual(audit['joint_train_n'], 3)
        self.assertAlmostEqual(audit['offset'], 0.3)
        self.assertTrue(audit['offset_valid'])

    def test_offset_needs_three_joint_training_identities(self):
        mod = producer()
        raw, offset, audit = mod.direct_transport(np.array([0.5, 0.7, np.nan]), np.array([0.1, 0.2, 0.3]), np.array([0.4, np.nan]), 0.12)
        np.testing.assert_allclose(raw, [0.4, 0.12])
        np.testing.assert_allclose(offset, [0.12, 0.12])
        self.assertFalse(audit['offset_valid'])

    def test_anchor_only_predictions_ignore_hidden_endpoint_poison(self):
        mod = producer()
        covariance = np.array([[1.0, 0.1, 0.3, 0.2], [0.1, 1.0, 0.2, 0.4], [0.3, 0.2, 1.0, 0.1], [0.2, 0.4, 0.1, 1.0]])
        fit = {'mu': np.array([0.2, 0.3, 0.4, 0.5]), 'cov': covariance}
        observed = np.array([[0.7, 0.2], [np.nan, 0.2], [np.nan, np.nan]])
        prediction = mod.panel_predictions(fit, covariance, observed, [2, 3], [0, 1], -1)
        poisoned = np.full((3, 4), 1000000000.0)
        poisoned[:, [2, 3]] = observed
        other = mod.panel_predictions(fit, covariance, poisoned[:, [2, 3]], [2, 3], [0, 1], -1)
        np.testing.assert_array_equal(prediction['full_raw'], other['full_raw'])
        self.assertAlmostEqual(prediction['full_raw'][-1], 0.1)
        with self.assertRaises((AssertionError, ValueError)):
            mod.panel_predictions(fit, covariance, observed, [0, 3], [0, 1], -1)

    def test_target_mask_does_not_change_with_anchor_availability(self):
        mod = producer()
        actual = np.array([0.1, 0.2, np.nan, 0.3])
        first = mod.score_values(actual, {'p': np.array([0.1, 0.1, 0.1, 0.1])}, np.ones(4, bool))
        second = mod.score_values(actual, {'p': np.array([0.0, 0.0, 0.0, 0.0])}, np.ones(4, bool))
        self.assertEqual(first['n'], 3)
        self.assertEqual(first['n'], second['n'])
        common = mod.score_values(actual, {'p': np.zeros(4)}, np.array([True, False, True, True]))
        self.assertEqual(common['n'], 2)

    def test_all9_complete_mask_requires_every_frozen_panel(self):
        mod = producer()
        self.assertTrue(hasattr(mod, 'availability_masks'), 'all9 complete-mask policy is missing')
        reference = np.array([[0.1, 0.2], [0.2, 0.3], [np.nan, 0.3]])
        random = [np.array([[0.3, 0.4], [0.4, np.nan], [0.1, 0.2]]) for _ in range(8)]
        masks = mod.availability_masks(reference, random)
        np.testing.assert_array_equal(masks['primary'], [True, True, True])
        np.testing.assert_array_equal(masks['reference_complete'], [True, True, False])
        np.testing.assert_array_equal(masks['all9_complete'], [True, False, False])

    def test_random_draws_average_sse_without_multiplying_target_count(self):
        mod = producer()
        scores = [{'n': 2, 'full_raw_sse': 2.0}, {'n': 2, 'full_raw_sse': 18.0}]
        averaged = mod.average_draw_scores(scores)
        self.assertEqual(averaged['n'], 2)
        self.assertEqual(averaged['full_raw_sse'], 10.0)
        with self.assertRaises(ValueError):
            mod.average_draw_scores([scores[0], {'n': 1, 'full_raw_sse': 3.0}])

    def test_training_sample_must_equal_original_nested_register(self):
        mod = producer()
        pool = np.arange(14)
        selected = mod.nested_quarter(pool, 's', 'pair', 0, 1)
        sample = dict(screen='s', split='pair', repeat=0, fold=1, fraction=0.25, training_indices_json=json.dumps(selected.tolist()), test_indices_json='[14, 15]')
        self.assertTrue(mod.validate_training_sample(sample, 16, np.array([14, 15])))
        sample['training_indices_json'] = json.dumps(np.r_[selected[:-1], 15].tolist())
        with self.assertRaises(ValueError):
            mod.validate_training_sample(sample, 16, np.array([14, 15]))

    def test_context_and_swap_weightings_are_explicit(self):
        mod = producer()
        folds = []
        for edge, context, swap, mse in (('a', 'x', 'p', 1.0), ('b', 'x', 'q', 3.0), ('c', 'y', 'q', 10.0)):
            folds.append(dict(screen='s', family='f', split='pair', fraction=1.0, scope='primary', edge_id=edge, context_id=context, swap_key=swap, n=2, semantic_full_raw_sse=2 * mse, random_full_raw_sse=4 * mse, semantic_rank1_raw_sse=4 * mse, random_rank1_raw_sse=6 * mse, marginal_sse=8 * mse, direct_raw_sse=10 * mse, offset_raw_sse=12 * mse))
        edge, context, screen, family, headline = mod.aggregate_scores(pd.DataFrame(folds))
        self.assertAlmostEqual(screen.loc[screen.weighting.eq('context_balanced'), 'semantic_full_raw_mse'].iloc[0], 6.0)
        self.assertAlmostEqual(screen.loc[screen.weighting.eq('swap_balanced'), 'semantic_full_raw_mse'].iloc[0], 3.75)
        self.assertEqual(headline.n_edges.iloc[0], 3)
if __name__ == '__main__':
    unittest.main()
