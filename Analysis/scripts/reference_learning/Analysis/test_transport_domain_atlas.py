"""Semantic tests for observed-domain construction."""
import unittest
import numpy as np
import pandas as pd
import transport_domain_atlas as atlas

class DomainAtlasTests(unittest.TestCase):

    def fixture(self):
        registry = pd.DataFrame([{'source_screen': 'new|1', 'target_screen': 'target|1', 'source_family': 'new', 'target_family': 'target', 'relation': 'cross_family', 'mode': 'dose_relaxed', 'background_control': 'background_free', 'basis': 'position_2', 'unit_identity': 'molecule', 'source_unit_id': 'a', 'target_unit_id': 'b', 'source_n_partners': 1, 'target_n_partners': 1, 'shared_partner_count': 0, 'observed': 0.1, 'source_conditioned': 0.08, 'zero': 0.0}])
        rows = []
        for model in ('source_conditioned', 'zero'):
            for threshold in (1, 3, 5):
                for overlap in ('all', 'no_shared_partner'):
                    rows.append({**{key: registry.iloc[0][key] for key in atlas.PAIR_KEYS + atlas.BASE_KEYS}, 'model': model, 'min_partners_each': threshold, 'partner_overlap': overlap, 'status': 'scored' if threshold == 1 else 'zero_eligible', 'transfer_state': 'baseline_zero' if model == 'zero' else 'supported_positive_vs_zero' if threshold == 1 else 'zero_eligible'})
        spectra = pd.DataFrame([{'screen': screen, 'k_var90': 2, 'participation_rank': 1.5, 'observed_rank': 4} for screen in ('new|1', 'target|1')])
        return (pd.DataFrame(rows), registry, spectra)

    def test_admissible_settings_exclude_invalid_full_pair(self):
        self.assertEqual(len(atlas.admissible_settings()), 52)
        for config in atlas.admissible_settings():
            if config['basis'] == 'full_pair':
                self.assertEqual((config['min_partners_each'], config['partner_overlap']), (1, 'all'))
        with self.assertRaisesRegex(ValueError, 'structurally inapplicable'):
            atlas.validate_configuration({**atlas.ANCHOR, 'basis': 'full_pair', 'min_partners_each': 3})
        with self.assertRaisesRegex(ValueError, 'structurally inapplicable'):
            atlas.validate_configuration({**atlas.ANCHOR, 'basis': 'full_pair', 'partner_overlap': 'no_shared_partner'})

    def test_off_anchor_pair_and_unknown_cells_survive(self):
        paths, registry, spectra = self.fixture()
        result = atlas.build_domain_atlas(paths, registry, spectra)
        self.assertEqual(len(result), 52)
        self.assertEqual(set(result.cohort), {'off_anchor22'})
        self.assertEqual(result.domain_state.value_counts().to_dict(), {'geometry_absent': 46, 'filter_empty': 4, 'scored_positive': 2})
        empty = result[result.domain_state.eq('filter_empty')]
        self.assertFalse(empty.outcome_known.any())
        self.assertEqual(set(empty.zero_baseline_state), {'unknown_zero_baseline'})
        self.assertEqual(set(empty.path_status), {'zero_eligible'})

    def test_duplicate_guard(self):
        paths, registry, spectra = self.fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate pair/config/model'):
            atlas.build_domain_atlas(pd.concat([paths, paths.iloc[[0]]]), registry, spectra)

    def test_prediction_requires_common_finite_zero(self):
        paths, registry, spectra = self.fixture()
        registry['zero'] = np.nan
        result = atlas.build_domain_atlas(paths, registry, spectra)
        self.assertEqual(int(result.domain_state.eq('prediction_unavailable').sum()), 2)
        self.assertFalse(result.outcome_known.any())

    def test_neutral_transition_is_not_reversal(self):
        self.assertEqual(atlas.sign_transition('scored_positive', 'scored_neutral'), 'to_or_from_neutral')
        self.assertEqual(atlas.sign_transition('scored_positive', 'scored_negative'), 'positive_negative_reversal')
        self.assertEqual(atlas.sign_transition('filter_empty', 'scored_negative'), 'unknown_comparison')

    def test_two_scored_marginals_do_not_guarantee_joint_support(self):
        paths, registry, spectra = self.fixture()
        additions, new_paths = ([], [])
        for basis, partners in (('position_1', 1), ('position_2', 3)):
            row = {**registry.iloc[0].to_dict(), 'mode': 'strict', 'basis': basis, 'source_n_partners': partners, 'target_n_partners': partners}
            additions.append(row)
            for config in atlas.admissible_settings():
                if config['mode'] != 'strict' or config['background_control'] != 'background_free' or config['basis'] != basis:
                    continue
                for model in (atlas.MODEL, 'zero'):
                    new_paths.append({**{key: row[key] for key in atlas.PAIR_KEYS}, **config, 'model': model, 'status': 'scored' if partners >= config['min_partners_each'] else 'zero_eligible', 'transfer_state': 'baseline_zero' if model == 'zero' else 'supported_positive_vs_zero'})
        registry = pd.concat([registry, pd.DataFrame(additions)], ignore_index=True)
        paths = pd.concat([paths, pd.DataFrame(new_paths)], ignore_index=True)
        cells = atlas.build_domain_atlas(paths, registry, spectra)
        joint = atlas.build_joint_support(cells, registry)
        row = joint[joint.axis_combination.eq('threshold3+position1')].iloc[0]
        self.assertTrue(row.both_marginals_known)
        self.assertFalse(row.joint_known)
        self.assertEqual(row.joint_state, 'filter_empty')

    def test_mode_qualified_ids_do_not_establish_response_difference(self):
        _, registry, _ = self.fixture()
        row = {**registry.iloc[0].to_dict(), 'background_key': '*', 'source_dose_levels_json': '["1"]', 'target_dose_levels_json': '["1"]', 'source_partner_union_json': '["partner"]', 'target_partner_union_json': '["other"]', 'source_contexts': 1, 'target_contexts': 1}
        strict = {**row, 'mode': 'strict', 'perturbation_key': '["reagent_replacement","A","B","1"]', 'source_unit_id': 'strict_source', 'target_unit_id': 'strict_target'}
        relaxed = {**row, 'perturbation_key': '["reagent_replacement","A","B"]', 'source_unit_id': 'relaxed_source', 'target_unit_id': 'relaxed_target'}
        result = atlas.build_mode_semantics(pd.DataFrame([strict, relaxed]))
        self.assertEqual(len(result), 1)
        self.assertTrue(result.both_mode_qualified_ids_different.iloc[0])
        self.assertEqual(result.observed_difference_pp.iloc[0], 0)
        self.assertEqual(result.source_conditioned_difference_pp.iloc[0], 0)
if __name__ == '__main__':
    unittest.main()
