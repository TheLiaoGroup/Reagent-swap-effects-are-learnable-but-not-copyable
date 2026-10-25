"""Frozen structural controls and paired scientific estimands for report65."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from chemically_aligned_measurements import freeze_registry

def producer():
    path = Path(__file__).with_name('chemical_observation_specificity.py')
    assert path.exists(), 'report65 producer has not been implemented'
    return importlib.import_module('chemical_observation_specificity')

def source_registry():
    rows = []
    for edge, a, b, context, swap in (('target', 0, 1, 'a', 'x'), ('same', 2, 3, 'b', 'x'), ('loose', 4, 5, 'a', 'y'), ('strict', 6, 7, 'c', 'z'), ('shared', 0, 8, 'd', 'q')):
        rows.append(dict(screen='s|1', family='s', edge_id=edge, condition_a_index=a, condition_b_index=b, condition_a=f'C{a}', condition_b=f'C{b}', context_id=context, swap_key=swap, category='Reagent1', contrast_type='identity_swap', dose='1', orientation=1))
    return freeze_registry(pd.DataFrame(rows), {'s|1': np.array([f'C{i}' for i in range(12)])}, {'s|1': []})[0]

class ChemicalObservationSpecificityTests(unittest.TestCase):

    def test_control_hash_exact_compact_json_and_metadata_selection(self):
        mod = producer()
        expected = hashlib.sha256(json.dumps(['chemical-observation-specificity-v1', 's|1', 'target', 'strict'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(mod.control_hash('s|1', 'target', 'strict'), expected)
        one = mod.add_controls(source_registry()).set_index('edge_id')
        target = one.loc['target']
        self.assertEqual(target.different_other_edge_id, 'strict')
        self.assertEqual(target.n_different_any_candidates, 2)
        self.assertEqual(target.n_different_other_candidates, 1)
        self.assertEqual(target.strict_eligible, 1)
        self.assertFalse({target.different_any_condition_a_index, target.different_any_condition_b_index} & {0, 1})
        two = mod.add_controls(source_registry().sample(frac=1.0, random_state=7)).set_index('edge_id')
        self.assertEqual(target.different_any_edge_id, two.loc['target', 'different_any_edge_id'])

    def test_same_reference_controls_are_retained_and_flagged(self):
        mod = producer()
        source = source_registry()
        source = source[~source.edge_id.isin(['loose', 'shared'])]
        target = mod.add_controls(source).set_index('edge_id').loc['target']
        self.assertEqual(target.different_any_edge_id, target.different_other_edge_id)
        self.assertEqual(target.controls_same_reference, 1)

    def test_domain_complete_masks_follow_11_and_10_frozen_panels(self):
        mod = producer()
        values = np.ones((3, 2))
        other = values.copy()
        other[1, 0] = np.nan
        random = [values.copy() for _ in range(8)]
        random[0][2, 0] = np.nan
        strict = mod.domain_masks(values, values, other, random, 'strict_641')
        loose = mod.domain_masks(values, values, other, random, 'relaxed_643')
        np.testing.assert_array_equal(strict['primary'], [True, True, True])
        np.testing.assert_array_equal(strict['all_panels_complete'], [True, False, False])
        np.testing.assert_array_equal(loose['all_panels_complete'], [True, True, False])

    def test_risk_difference_closure_is_not_a_new_fit(self):
        mod = producer()
        frame = pd.DataFrame(dict(semantic_full_raw_mse=[10.0], different_other_full_raw_mse=[12.0], different_any_full_raw_mse=[13.0], random_full_raw_mse=[15.0], marginal_mse=[20.0]))
        actual = mod.paired_derivatives(frame).iloc[0]
        self.assertEqual(actual.other_specificity_advantage_mse, 2.0)
        self.assertEqual(actual.other_structure_advantage_mse, 3.0)
        self.assertEqual(actual.other_closure_error, 0.0)

    def test_relaxed_domain_unsupported_other_policy_stays_missing(self):
        mod = producer()
        rows = []
        for domain in ('strict_641', 'relaxed_643'):
            rows.append(dict(screen='s', family='f', split='pair', repeat=0, fold=0, fraction=1.0, edge_id='e', context_id='c', swap_key='x', scope='primary', domain=domain, n=2, semantic_full_raw_sse=2.0, semantic_rank1_raw_sse=3.0, random_full_raw_sse=4.0, random_rank1_raw_sse=5.0, marginal_sse=6.0, different_any_full_raw_sse=3.0, different_other_full_raw_sse=3.0 if domain == 'strict_641' else np.nan))
        _, _, screen, _, _ = mod.aggregate_domains(pd.DataFrame(rows))
        self.assertTrue(screen.loc[screen.domain.eq('relaxed_643'), 'different_other_full_raw_mse'].isna().all())

    def test_649_baseline_check_detects_prediction_and_target_changes(self):
        mod = producer()
        row = dict(screen='s', family='f', split='pair', repeat=0, fold=0, fraction=1.0, edge_id='e', n=2, target_mask_sha256='mask', semantic_full_raw_sse=0.4, random_full_raw_sse=0.5, marginal_sse=0.6)
        original = pd.DataFrame([dict(row, scope='primary')])
        checked = mod.check_original_baselines(pd.DataFrame([row]), original)
        self.assertEqual(checked.max_absolute_risk_difference.iloc[0], 0.0)
        changed = pd.DataFrame([dict(row, n=1)])
        with self.assertRaises(ValueError):
            mod.check_original_baselines(changed, original)
        changed = pd.DataFrame([dict(row, semantic_full_raw_sse=0.41)])
        with self.assertRaises(ValueError):
            mod.check_original_baselines(changed, original)
if __name__ == '__main__':
    unittest.main()
