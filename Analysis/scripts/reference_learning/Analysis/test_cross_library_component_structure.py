"""Focused checks for cross-library identity and outcome definitions."""
import unittest
import numpy as np
import pandas as pd
import cross_library_component_structure as c

def recipe(ingredients, solvent='S1', temperature='25', time='2'):
    row = {'solvent_id': solvent, 'temp_id': temperature, 'time_id': time}
    for slot in range(1, 7):
        identity, dose = ingredients.get(slot, ('', ''))
        row[f'Reagent{slot}_id'] = identity
        row[f'Reagent{slot}_HTE_Equiv'] = dose
        row[f'Reagent{slot}_Standardized Name'] = identity
    return row

class ComponentStructureTests(unittest.TestCase):

    def test_slot_invariance_and_background(self):
        a = recipe({1: ('A', '1'), 3: ('X', '2')})
        moved = recipe({4: ('A', '1'), 2: ('X', '2')})
        b = recipe({2: ('B', '1'), 5: ('X', '2')})
        self.assertEqual(c.recipe_key(a), c.recipe_key(moved))
        first = c.detect_perturbation(a, b)
        second = c.detect_perturbation(moved, b)
        self.assertEqual(first['key'], second['key'])
        self.assertEqual(first['background_key'], second['background_key'])
        self.assertEqual(first['orientation'], 1)
        self.assertEqual(c.detect_perturbation(b, a)['orientation'], -1)

    def test_same_dose_and_fixed_physical_fields(self):
        a = recipe({1: ('A', '1')})
        self.assertIsNone(c.detect_perturbation(a, recipe({1: ('B', '2')})))
        self.assertIsNone(c.detect_perturbation(a, recipe({1: ('B', '1')}, time='3')))
        solvent = c.detect_perturbation(a, recipe({4: ('A', '1')}, solvent='S2'))
        self.assertEqual(solvent['kind'], 'solvent_replacement')

    def test_multiset_multiplicity(self):
        a = recipe({1: ('A', '1'), 2: ('A', '1')})
        b = recipe({1: ('B', '1'), 2: ('A', '1')})
        contrast = c.detect_perturbation(a, b)
        self.assertEqual(contrast['background_reagents'], [('A', '1')])

    def test_missing_is_not_zero_and_standardization(self):
        result = c.delta_summary(np.array([0.4, np.nan, 0.2]), np.array([0.6, 0.9, 0.1]))
        self.assertEqual(result['n_overlap'], 2)
        self.assertAlmostEqual(result['mean_delta'], 0.05)
        self.assertAlmostEqual(result['signed_consistency'], 0.05 / np.sqrt(0.025))
        zero = c.delta_summary(np.zeros(3), np.zeros(3))
        self.assertEqual(zero['signed_consistency'], 0)

    def test_family_holdout_groups_bh_and_cmc(self):
        self.assertEqual(c.transformation('Buchwald-Hartwig(CMC)|2'), 'Buchwald-Hartwig')
        self.assertEqual(c.transformation('Buchwald-Hartwig|1'), 'Buchwald-Hartwig')
        self.assertEqual(c.transformation('SNAr|1'), 'SNAr')

    def test_canonical_substrate_keeps_position(self):
        pair = c.canonical_pair('CCO >> C(C)N')
        self.assertEqual(pair, ('CCO', 'CCN'))
        self.assertNotEqual(pair, c.canonical_pair('C(C)N >> CCO'))

    def test_training_weights_balance_families_and_screens(self):
        frame = pd.DataFrame(dict(screen=['A|1', 'A|1', 'A|2', 'B|1'], family=['A', 'A', 'A', 'B'], y=[0.0, 0.0, 0.0, 1.0]))
        weights = c.screen_balanced_weights(frame)
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertAlmostEqual(c.weighted_training_mean(frame, 'y'), 0.5)

    def test_features_do_not_read_target_outcomes(self):
        row = dict(kind='reagent_replacement', low_id='A', high_id='B', perturbation_key='swap', background_reagents_json='[["X","2"]]', fixed_solvent_id='S1', temp_id='25', time_id='2', mean_delta=0.0, signed_consistency=0.0)
        names = dict(A='TEA', B='DIPEA', X='K2CO3', S1='DMF')
        expected = c.feature_dict(row, names, True)
        row.update(mean_delta=0.99, signed_consistency=-1.0, n_overlap=100000)
        self.assertEqual(expected, c.feature_dict(row, names, True))
if __name__ == '__main__':
    unittest.main()
