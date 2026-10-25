"""Focused checks of response identity, weighting, and cancellation boundaries."""
from fractions import Fraction as F
import unittest
import pandas as pd
import response_estimand_lineage as lineage

class ResponseLineageTests(unittest.TestCase):

    def test_different_contrasts_can_define_identical_yield_operator(self):
        contexts = {'ab': ('s', 'a', 'b'), 'bd': ('s', 'b', 'd'), 'ac': ('s', 'a', 'c'), 'cd': ('s', 'c', 'd')}
        split = {('ab', 0): F(1, 2), ('bd', 0): F(1, 2)}
        direct = {('ac', 0): F(1, 2), ('cd', 0): F(1, 2)}
        self.assertNotEqual(split, direct)
        self.assertEqual(lineage.expand_operator(split, contexts), lineage.expand_operator(direct, contexts))

    def test_identical_observed_mean_does_not_establish_operator_identity(self):
        left, right = ({('s', 0, 'a'): F(-1), ('s', 0, 'b'): F(1)}, {('s', 1, 'a'): F(-1), ('s', 1, 'b'): F(1)})
        self.assertNotEqual(lineage.signature(left), lineage.signature(right))
        yields = {key: F(0) for key in set(left) | set(right)}
        self.assertEqual(sum((left[k] * yields[k] for k in left)), sum((right[k] * yields[k] for k in right)))

    def test_nested_equal_partner_weighting_is_not_equal_atom_weighting(self):
        atom = {('a', 0): F(1, 2), ('b', 1): F(1, 4), ('c', 1): F(1, 4)}
        self.assertEqual(sum(atom.values()), 1)
        self.assertEqual(atom['a', 0], atom['b', 1] + atom['c', 1])
        self.assertNotEqual(lineage.signature(atom), lineage.signature({key: F(1, 3) for key in atom}))

    def test_endpoint_reconstructs_nonuniform_context_support(self):
        key = '["replacement","a","b","1"]'
        unit = pd.DataFrame([dict(mode='strict', background_control='background_free', screen='s', family='f', perturbation_key=key, background_key='*', basis='position_1', unit_identity='m', unit_id='u', n_partners=2, n_atomic_cells=3, observed=0.4)])
        contrasts = pd.DataFrame([dict(contrast_id=c, screen='s', perturbation_key=key, condition_low='low' + c, condition_high='high' + c) for c in ('a', 'b', 'c')])
        atomic = pd.DataFrame(dict(contrast_id=['a', 'b', 'c'], pair_index=[0, 1, 1], sub_1=['m'] * 3, sub_2=['p', 'q', 'q'], delta=[0.2, 0.4, 0.8]))
        endpoints, _, _, atoms, _ = lineage.build_endpoint_lineage(atomic, unit, contrasts)
        self.assertEqual(atoms['u'], {('a', 0): F(1, 2), ('b', 1): F(1, 4), ('c', 1): F(1, 4)})
        self.assertAlmostEqual(endpoints.reconstructed_observed.iloc[0], 0.4)

    def test_collapsed_average_accumulates_shared_atoms_exactly(self):
        result = lineage.average_maps([{('a', 0): F(1)}, {('a', 0): F(1, 2), ('b', 0): F(1, 2)}])
        self.assertEqual(result, {('a', 0): F(3, 4), ('b', 0): F(1, 4)})
        self.assertEqual(sum(result.values()), 1)

    def test_canonical_signature_ignores_order_and_zero_coefficients(self):
        self.assertEqual(lineage.signature({('a', 0): F(2, 4), ('b', 0): F(0)}), lineage.signature({('a', 0): F(1, 2)}))
        with self.assertRaises(ValueError):
            lineage.average_maps([])

    def test_duplicate_atoms_rejected(self):
        atoms = pd.DataFrame({'contrast_id': ['a', 'a'], 'pair_index': [0, 0]})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            lineage.build_endpoint_lineage(atoms, pd.DataFrame(), pd.DataFrame())
if __name__ == '__main__':
    unittest.main()
