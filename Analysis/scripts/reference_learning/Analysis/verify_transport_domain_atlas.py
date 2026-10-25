"""Verify report 48 saved tables and atomic support/outcome provenance."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from PIL import Image
import transport_domain_atlas as atlas

def canonical(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns.name = None
    for column in result:
        if pd.api.types.is_bool_dtype(reference[column]):
            result[column] = result[column].astype(str).str.lower().eq('true')
        elif pd.api.types.is_numeric_dtype(reference[column]):
            result[column] = pd.to_numeric(result[column], errors='coerce')
        else:
            result[column] = result[column].astype('string')
    return result

def main() -> None:
    paths, registry, spectra, hashes = atlas.load_inputs()
    outputs = atlas.build_outputs(paths, registry, spectra)
    checks = []
    for name, expected in outputs.items():
        saved = pd.read_csv(atlas.TABLES / name, keep_default_na=False)
        try:
            pd.testing.assert_frame_equal(canonical(saved, expected), canonical(expected, expected), check_dtype=False, rtol=1e-09, atol=1e-10)
        except (AssertionError, KeyError) as error:
            checks.append({'check': 'rebuild:' + name, 'passed': False, 'details': str(error)[:500]})
        else:
            checks.append({'check': 'rebuild:' + name, 'passed': True})
    cells = outputs['transport_domain_atlas_cells.csv']
    pairs = outputs['transport_domain_atlas_pairs.csv']
    materialized = cells[cells.path_present]
    source = paths[paths.model.eq(atlas.MODEL)]
    merged = materialized.merge(source, on=atlas.PAIR_KEYS + atlas.CONFIG_KEYS, validate='one_to_one', suffixes=('_atlas', '_source'))
    finite = merged[merged.domain_state.str.startswith('scored_')]
    joint = outputs['transport_domain_atlas_joint_support.csv']
    counterexample = joint[joint.relation.eq('cross_family') & joint.axis_combination.eq('threshold3+position1')]
    mode_semantics = outputs['transport_domain_atlas_mode_semantics.csv']
    screens = outputs['transport_domain_atlas_screens.csv']
    raw_cross = registry[registry.relation.eq('cross_family') & registry['mode'].eq('strict') & registry.background_control.eq('background_free')]
    pair_columns = ['source_screen', 'target_screen']
    pos1 = raw_cross[raw_cross.basis.eq('position_1')]
    pos2 = raw_cross[raw_cross.basis.eq('position_2')]
    pos1_min3 = pos1[pos1.source_n_partners.ge(3) & pos1.target_n_partners.ge(3)]
    pos2_min3 = pos2[pos2.source_n_partners.ge(3) & pos2.target_n_partners.ge(3)]
    pair_set = lambda frame: set(map(tuple, frame[pair_columns].drop_duplicates().to_numpy()))
    both_marginal_pairs = pair_set(pos1) & pair_set(pos2_min3)
    actual_joint_pairs = pair_set(pos1_min3)
    expected_marginals = {('Buchwald-Hartwig(CMC)|3', 'Heck|1'), ('Heck|1', 'Buchwald-Hartwig(CMC)|3'), ('Heck|1', 'Ullmann|1'), ('Ullmann|1', 'Heck|1')}
    raw_semantics = registry.copy()
    raw_semantics['chemical_key'] = raw_semantics.perturbation_key.map(lambda value: json.dumps(json.loads(value)[:3], separators=(',', ':')))
    signature = atlas.PAIR_KEYS + ['background_control', 'background_key', 'basis', 'unit_identity', 'chemical_key', 'source_dose_levels_json', 'target_dose_levels_json', 'source_partner_union_json', 'target_partner_union_json', 'source_contexts', 'target_contexts']
    semantic_matches = raw_semantics[raw_semantics['mode'].eq('strict')].merge(raw_semantics[raw_semantics['mode'].eq('dose_relaxed')], on=signature, validate='one_to_one', suffixes=('_s', '_r'))
    checks.extend([{'check': 'complete_registered_domain', 'passed': len(cells) == 4056 and len(pairs) == 78 and pairs.n_admissible.eq(52).all()}, {'check': 'cross_family_domain', 'passed': pairs.relation.eq('cross_family').sum() == 64 and pairs.cohort.eq('central42').sum() == 42 and (pairs.cohort.eq('off_anchor22').sum() == 22)}, {'check': 'source_inventory', 'passed': len(source) == 1076 and len(materialized) == 1076 and (len(merged) == 1076)}, {'check': 'raw_support_matches_saved_status', 'passed': merged.n_selected_rows.eq(pd.to_numeric(merged.n_directed_rows)).all() and merged.n_source_finite.eq(pd.to_numeric(merged.n_units)).all()}, {'check': 'raw_rmse_matches_saved_score', 'passed': bool(np.allclose(finite.common_source_rmse_pp, pd.to_numeric(finite.rmse_pp), rtol=1e-09, atol=1e-10) and np.allclose(finite.common_zero_rmse_pp, pd.to_numeric(finite.zero_rmse_pp), rtol=1e-09, atol=1e-10))}, {'check': 'unknown_outcomes_not_negative', 'passed': not cells.loc[cells.domain_state.isin(['geometry_absent', 'filter_empty', 'prediction_unavailable']), 'outcome_known'].any()}, {'check': 'empty_zero_not_scored_baseline', 'passed': cells.loc[cells.domain_state.eq('filter_empty'), 'zero_baseline_state'].eq('unknown_zero_baseline').all()}, {'check': 'spectrum_join_complete', 'passed': cells[['source_participation_rank', 'target_participation_rank']].notna().all().all()}, {'check': 'admissible_full_pair', 'passed': cells.loc[cells.basis.eq('full_pair'), 'min_partners_each'].eq(1).all() and cells.loc[cells.basis.eq('full_pair'), 'partner_overlap'].eq('all').all()}, {'check': 'all_eighteen_spectra_screens_retained', 'passed': len(screens) == 18 and set(screens.loc[screens.n_outgoing_registered_pairs.eq(0), 'screen']) == {'Grignard|1'}}, {'check': 'marginal_joint_counterexample', 'passed': int(counterexample.both_marginals_known.sum()) == 4 and (not (counterexample.both_marginals_known & counterexample.joint_known).any()) and (int(counterexample.joint_known.sum()) == 2)}, {'check': 'independent_registry_partner_counterexample', 'passed': both_marginal_pairs == expected_marginals and actual_joint_pairs == {('Buchwald-Hartwig|1', 'SNAr|1'), ('SNAr|1', 'Buchwald-Hartwig|1')} and (not both_marginal_pairs.intersection(actual_joint_pairs))}, {'check': 'mode_qualified_id_semantic_counterexample', 'passed': len(mode_semantics) == 1752 and mode_semantics.both_mode_qualified_ids_different.all() and np.allclose(mode_semantics.observed_difference_pp, 0, atol=1e-10) and np.allclose(mode_semantics.source_conditioned_difference_pp, 0, atol=1e-10)}, {'check': 'independent_registry_mode_metadata_matches', 'passed': len(semantic_matches) == 1752 and semantic_matches.source_unit_id_s.ne(semantic_matches.source_unit_id_r).all() and semantic_matches.target_unit_id_s.ne(semantic_matches.target_unit_id_r).all() and np.allclose(semantic_matches.observed_s, semantic_matches.observed_r, atol=1e-12) and np.allclose(semantic_matches.source_conditioned_s, semantic_matches.source_conditioned_r, atol=1e-12)}])
    manifest = json.loads((atlas.DATA / 'transport_domain_atlas_manifest.json').read_text(encoding='utf-8'))
    checks.append({'check': 'input_hashes', 'passed': hashes == manifest['input_sha256']})
    for category in ('code_sha256', 'output_sha256'):
        checks.append({'check': category, 'passed': all((atlas.digest(atlas.HERE / name) == value for name, value in manifest[category].items()))})
    with Image.open(atlas.FIGURES / '38_TRANSPORT_DOMAIN_ATLAS.png') as image:
        pixels = np.asarray(image.convert('RGB'))
    checks.append({'check': 'nonblank_figure', 'passed': pixels.shape[0] >= 800 and pixels.shape[1] >= 1400 and (float(np.std(pixels)) > 10)})
    result = {'passed': all((bool(check['passed']) for check in checks)), 'checks_total': len(checks), 'checks_passed': sum((bool(check['passed']) for check in checks)), 'rebuilt_tables': sum((check['check'].startswith('rebuild:') and check['passed'] for check in checks)), 'checks': checks}
    (atlas.DATA / 'transport_domain_atlas_verification.json').write_text(json.dumps(result, indent=2, default=bool), encoding='utf-8')
    print(json.dumps(result, indent=2, default=bool))
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
