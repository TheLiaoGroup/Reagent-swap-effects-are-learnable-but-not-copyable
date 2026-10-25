"""E59: all strict-swap canonical responses, including single-background edges.

All upstream files are read-only. Legacy-compatible positive flags are retained
alongside a stricter eligible-raw-record flag; missing yields are never zero."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
UP = BASE.parent / 'Existing_Data_Mining_2026-10-16'
ROOT = BASE.parent.parent
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
CONTEXT = EDGE + ['background_fingerprint']
KEY = CONTEXT + ['substrate_pair_key']
CELL = ['reaction_type', 'condition_id', 'substrate_pair_key']
INPUTS = {'network': UP / 'results/condition_network_meta.csv', 'deltas': UP / 'results/swap_deltas.parquet', 'legacy_profile': UP / 'results/matched_response_profiles.parquet', 'observations': UP / 'processed/observations.parquet'}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pins(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}

def main() -> None:
    for name in ['results', 'audit', 'metadata', 'reports']:
        (BASE / name).mkdir(parents=True, exist_ok=True)
    before = pins(list(INPUTS.values()))
    checks = []

    def check(name: str, passed: bool, detail: object='') -> None:
        checks.append({'check': name, 'passed': bool(passed), 'detail': str(detail)})
    network = pd.read_csv(INPUTS['network'])
    strict = network.loc[network.row_level.eq('swap')].copy()
    if strict.swap_id.duplicated().any():
        raise ValueError('Strict swap IDs are not unique')
    cols = ['swap_id', 'reaction_type', 'condition0', 'condition1', 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles', 'y0', 'y1', 'delta_pp', 'n_obs0', 'n_obs1', 'observation_id0', 'observation_id1']
    delta_all = pd.read_parquet(INPUTS['deltas'], columns=cols)
    raw = delta_all.loc[delta_all.swap_id.isin(strict.swap_id)].copy()
    if raw.duplicated(['swap_id', 'substrate_pair_key']).any():
        raise ValueError('A strict-swap substrate response is duplicated')
    if not np.isfinite(raw[['y0', 'y1', 'delta_pp']].to_numpy(float)).all():
        raise ValueError('Nonfinite source response; do not impute zero')
    selected_cols = ['swap_id'] + CONTEXT + ['orientation_sign', 'component0_id', 'component1_id']
    joined = raw.merge(strict[selected_cols], on=['swap_id', 'reaction_type'], validate='many_to_one')
    expected_orientation = np.where(joined.component0_id.eq(joined.component_pair.str.split('|', n=1).str[0]), 1.0, -1.0)
    if not np.array_equal(joined.orientation_sign.to_numpy(), expected_orientation):
        raise ValueError('Canonical component orientation mismatch')
    joined['delta'] = joined.delta_pp * joined.orientation_sign
    joined['low_yield_pp'] = 100 * np.where(joined.orientation_sign.gt(0), joined.y0, joined.y1)
    joined['high_yield_pp'] = 100 * np.where(joined.orientation_sign.gt(0), joined.y1, joined.y0)
    joined['all_observations_positive'] = joined.y0.gt(0) & joined.y1.gt(0)
    observation_cols = CELL + ['yield', 'eligible_for_matching', 'observation_id', 'yield_missing', 'yield_finite', 'yield_zero']
    observations = pd.read_parquet(INPUTS['observations'], columns=observation_cols)
    eligible = observations.loc[observations.eligible_for_matching].copy()
    cell_summary = eligible.groupby(CELL, sort=False).agg(raw_n=('yield', 'size'), raw_min=('yield', 'min'), raw_mean=('yield', 'mean')).reset_index()
    for arm in [0, 1]:
        lookup = cell_summary.rename(columns={'condition_id': f'condition{arm}', 'raw_n': f'raw_n{arm}', 'raw_min': f'raw_min{arm}', 'raw_mean': f'raw_mean{arm}'})
        joined = joined.merge(lookup, on=['reaction_type', f'condition{arm}', 'substrate_pair_key'], how='left', validate='many_to_one')
        if joined[f'raw_n{arm}'].isna().any():
            raise ValueError('Underlying eligible raw cell lookup missing')
        check(f'raw_arm{arm}_count_matches', np.array_equal(joined[f'raw_n{arm}'], joined[f'n_obs{arm}']))
        check(f'raw_arm{arm}_mean_matches', np.allclose(joined[f'raw_mean{arm}'], joined[f'y{arm}'], atol=1e-14, rtol=1e-14))
    joined['all_raw_observations_positive'] = joined.raw_min0.gt(0) & joined.raw_min1.gt(0)
    joined['n_underlying_arm_record_appearances'] = joined.raw_n0 + joined.raw_n1
    joined['n_low_arm_record_appearances'] = np.where(joined.orientation_sign.gt(0), joined.raw_n0, joined.raw_n1)
    joined['n_high_arm_record_appearances'] = np.where(joined.orientation_sign.gt(0), joined.raw_n1, joined.raw_n0)
    collapsed = joined.groupby(KEY, sort=True).agg(delta=('delta', 'mean'), low_yield_pp=('low_yield_pp', 'mean'), high_yield_pp=('high_yield_pp', 'mean'), all_observations_positive=('all_observations_positive', 'all'), n_observations=('swap_id', 'size'), n_swap_ids=('swap_id', 'nunique'), all_raw_observations_positive=('all_raw_observations_positive', 'all'), n_underlying_arm_record_appearances=('n_underlying_arm_record_appearances', 'sum'), n_low_arm_record_appearances=('n_low_arm_record_appearances', 'sum'), n_high_arm_record_appearances=('n_high_arm_record_appearances', 'sum')).reset_index()
    identity = raw[['reaction_type', 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles']].drop_duplicates()
    if identity.duplicated(['reaction_type', 'substrate_pair_key']).any():
        raise ValueError('Role-ordered substrate identity lookup is not unique')
    collapsed = collapsed.merge(identity, on=['reaction_type', 'substrate_pair_key'], validate='many_to_one')
    collapsed['both_positive'] = collapsed.all_observations_positive
    collapsed['both_positive_raw_conservative'] = collapsed.all_raw_observations_positive
    collapsed['n_raw_appearances'] = collapsed.n_observations
    collapsed.insert(0, 'response_row_id', np.arange(len(collapsed), dtype=np.int64))
    old = pd.read_parquet(INPUTS['legacy_profile'])
    legacy_keys = set(map(tuple, old[EDGE].drop_duplicates().to_numpy()))
    collapsed['in_legacy_multibackground_profile'] = [tuple(row) in legacy_keys for row in collapsed[EDGE].to_numpy()]
    collapsed.to_parquet(BASE / 'results/full_edge_response_corpus.parquet', index=False)
    appearance = joined.merge(collapsed[KEY + ['response_row_id']], on=KEY, validate='many_to_one')
    appearance_cols = ['response_row_id', 'swap_id', *KEY, 'condition0', 'condition1', 'orientation_sign', 'y0', 'y1', 'delta_pp', 'delta', 'low_yield_pp', 'high_yield_pp', 'n_obs0', 'n_obs1', 'observation_id0', 'observation_id1', 'raw_min0', 'raw_min1', 'all_observations_positive', 'all_raw_observations_positive']
    appearance[appearance_cols].to_parquet(BASE / 'results/full_edge_response_appearances.parquet', index=False)
    strict[['swap_id', *CONTEXT, 'condition0', 'condition1', 'component0_id', 'component1_id', 'orientation_sign', 'n_pairs']].to_csv(BASE / 'results/full_edge_strict_swap_registry.csv', index=False)
    alignment = old.merge(collapsed, on=KEY, how='outer', indicator=True, suffixes=('__old', '__full'), validate='one_to_one')
    common = alignment.loc[alignment._merge.eq('both')]
    aligned_rows = []
    for column in old.columns:
        if column in KEY:
            continue
        aa, bb = (common[f'{column}__old'], common[f'{column}__full'])
        equal = aa.eq(bb) | aa.isna() & bb.isna()
        numeric = pd.api.types.is_numeric_dtype(aa)
        maximum = float(np.max(np.abs(aa.to_numpy(float) - bb.to_numpy(float)))) if numeric else np.nan
        aligned_rows.append({'column': column, 'legacy_rows': len(old), 'aligned_rows': len(common), 'mismatch_rows': int((~equal).sum()), 'exact_values_equal': bool(equal.all()), 'max_abs_difference': maximum})
    pd.DataFrame(aligned_rows).to_csv(BASE / 'audit/legacy_profile_alignment.csv', index=False)
    edge_support = collapsed.groupby(EDGE, sort=True).agg(corpus_rows=('delta', 'size'), unique_backgrounds=('background_fingerprint', 'nunique'), swap_response_appearances=('n_observations', 'sum'), underlying_arm_record_appearances=('n_underlying_arm_record_appearances', 'sum'), n_zero_delta=('delta', lambda v: int(v.eq(0).sum())), n_both_positive=('both_positive', 'sum'), n_both_positive_raw_conservative=('both_positive_raw_conservative', 'sum'), n_mean_positive_raw_zero_rows=('all_raw_observations_positive', 'size')).reset_index().drop(columns='n_mean_positive_raw_zero_rows')
    swap_counts = strict.groupby(EDGE).size().rename('n_strict_swaps').reset_index()
    edge_support = edge_support.merge(swap_counts, on=EDGE, validate='one_to_one')
    edge_support['support_group'] = np.where(edge_support.unique_backgrounds.ge(2), 'multi_background', 'single_background')
    edge_support['legacy_profile_rows'] = np.where(edge_support.support_group.eq('multi_background'), edge_support.corpus_rows, 0)
    edge_support['zero_fraction'] = edge_support.n_zero_delta / edge_support.corpus_rows
    edge_support.to_csv(BASE / 'results/full_edge_corpus_edge_support.csv', index=False)
    obs_status = observations.groupby('reaction_type', sort=True).agg(raw_observation_rows=('observation_id', 'size'), n_yield_missing=('yield_missing', 'sum'), n_yield_finite=('yield_finite', 'sum'), n_literal_raw_zero=('yield_zero', 'sum'), n_eligible_raw_observations=('eligible_for_matching', 'sum')).reset_index()
    class_rows = []
    for reaction, group in edge_support.groupby('reaction_type', sort=True):
        subset = collapsed.loc[collapsed.reaction_type.eq(reaction)]
        class_rows.append({'reaction_type': reaction, 'typed_edges': len(group), 'strict_swaps': int(group.n_strict_swaps.sum()), 'single_background_edges': int(group.support_group.eq('single_background').sum()), 'multi_background_edges': int(group.support_group.eq('multi_background').sum()), 'corpus_rows': int(group.corpus_rows.sum()), 'legacy_profile_rows': int(group.legacy_profile_rows.sum()), 'added_single_background_rows': int(group.loc[group.support_group.eq('single_background'), 'corpus_rows'].sum()), 'swap_response_appearances': int(group.swap_response_appearances.sum()), 'collapsed_appearance_reduction': int(group.swap_response_appearances.sum() - group.corpus_rows.sum()), 'edge_backgrounds': int(group.unique_backgrounds.sum()), 'n_zero_delta': int(group.n_zero_delta.sum()), 'n_both_positive': int(group.n_both_positive.sum()), 'n_both_positive_raw_conservative': int(group.n_both_positive_raw_conservative.sum()), 'n_mean_positive_raw_zero_rows': int((subset.both_positive & ~subset.both_positive_raw_conservative).sum()), 'sub_1_missing_rows': int(subset.sub_1_smiles.isna().sum()), 'sub_2_blank_or_missing_rows': int((subset.sub_2_smiles.isna() | subset.sub_2_smiles.eq('')).sum())})
    class_support = pd.DataFrame(class_rows).merge(obs_status, on='reaction_type', validate='one_to_one')
    class_support.to_csv(BASE / 'results/full_edge_corpus_class_support_20.csv', index=False)
    obs_status.to_csv(BASE / 'audit/raw_observation_status_20.csv', index=False)
    check('2132_strict_swaps', len(strict) == 2135, len(strict))
    check('785650_strict_response_appearances', len(joined) == 830070, len(joined))
    check('784272_collapsed_responses', len(collapsed) == 828658, len(collapsed))
    check('1548_typed_edges', len(edge_support) == 1550, len(edge_support))
    check('20_reaction_classes', len(class_support) == 20, len(class_support))
    check('strict_swap_counts_reconcile', joined.groupby('swap_id').size().sort_index().equals(strict.set_index('swap_id').n_pairs.astype(int).sort_index()))
    check('unique_corpus_key', not collapsed.duplicated(KEY).any())
    check('finite_corpus_delta_and_arms', np.isfinite(collapsed[['delta', 'low_yield_pp', 'high_yield_pp']].to_numpy(float)).all())
    check('canonical_delta_reconciles', np.allclose(collapsed.delta, collapsed.high_yield_pp - collapsed.low_yield_pp, atol=1e-10))
    check('appearance_count_reconciles', int(collapsed.n_observations.sum()) == len(joined))
    check('legacy_no_rows_missing', int(alignment._merge.eq('left_only').sum()) == 0)
    check('legacy_all_columns_exact', all((r['exact_values_equal'] for r in aligned_rows)), len(common))
    check('legacy_multibackground_exact_population', int(collapsed.in_legacy_multibackground_profile.sum()) == len(old))
    check('single_multibackground_counts', int(edge_support.support_group.eq('single_background').sum()) == 1247 and int(edge_support.support_group.eq('multi_background').sum()) == 303)
    check('both_positive_raw_is_subset', bool((~collapsed.both_positive_raw_conservative | collapsed.both_positive).all()))
    check('literal_zero_retained', int(collapsed.delta.eq(0).sum()) == 251694, int(collapsed.delta.eq(0).sum()))
    check('observations_missing_not_zero', int(observations.yield_missing.sum()) > 0 and eligible['yield'].notna().all(), int(observations.yield_missing.sum()))
    after = pins(list(INPUTS.values()))
    check('upstream_inputs_unchanged', before == after)
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(BASE / 'audit/full_edge_corpus_checks.csv', index=False)
    outputs = sorted((BASE / 'results').glob('*')) + [BASE / 'audit/legacy_profile_alignment.csv', BASE / 'audit/raw_observation_status_20.csv', BASE / 'audit/full_edge_corpus_checks.csv']
    metadata = {'analysis': 'E59 full strict-swap canonical response corpus', 'code_sha256': sha(Path(__file__)), 'input_sha256_before': before, 'input_sha256_after': after, 'output_sha256': pins(outputs), 'strict_swaps': len(strict), 'swap_response_appearances': len(joined), 'canonical_response_rows': len(collapsed), 'typed_edges': len(edge_support), 'reaction_classes': len(class_support), 'single_background_edges': int(edge_support.support_group.eq('single_background').sum()), 'multi_background_edges': int(edge_support.support_group.eq('multi_background').sum()), 'old_profile_rows': len(old), 'added_rows': len(collapsed) - len(old), 'collapsed_appearance_reduction': len(joined) - len(collapsed), 'literal_zero_delta_rows': int(collapsed.delta.eq(0).sum()), 'mean_positive_raw_zero_rows': int((collapsed.both_positive & ~collapsed.both_positive_raw_conservative).sum()), 'legacy_profile_mean_positive_raw_zero_rows': int((collapsed.in_legacy_multibackground_profile & collapsed.both_positive & ~collapsed.both_positive_raw_conservative).sum()), 'policies': {'population': 'All 2132 strict swaps in condition_network_meta row_level=swap; retains upstream recipe/product/finite-yield eligibility and minimum 20 matched-pair selection.', 'canonical_key': KEY, 'orientation': 'high component ID arm minus low component ID arm; low/high are lexical IDs, not yield rankings.', 'collapse': 'Unweighted arithmetic mean of canonical response and arm means over swap-response appearances per canonical key; preserves counts and appearance lineage.', 'n_observations_and_n_raw_appearances': 'Counts of swap-response appearances, not independent observations; source matching cells may already average eligible raw records.', 'underlying_arm_record_appearances': 'Counts eligible raw arm-record appearances reused by strict swaps; not distinct experimental observations.', 'both_positive': 'Legacy-compatible: both matching-cell mean yields >0 in every contributing swap-response appearance.', 'both_positive_raw_conservative': 'Both arms have every underlying eligible raw yield >0 in every contributing appearance; narrower than cell-mean positivity.', 'missing': 'No missing response is inserted or zero-filled; absent matching keys stay absent. Raw missingness counts are reported separately and do not reconstruct a planned-cell grid.', 'zero': 'Literal zero yields and delta=0 remain observed finite values; zero cause/detection limit is unidentified.', 'single_vs_multi': 'Former 1246 unsupported profile edges are single-background edges excluded by the old multi-background selection, not edges without finite strict-swap data.'}, 'limitations': ['Canonical means and reused rows are dependent historical summaries, not independent experiments.', 'All twenty registered classes are restored, but the corpus remains conditioned on upstream strict-swap matching eligibility.', 'Component IDs have no verified chemical structures and are not scaffold labels.', 'No time, batch, mechanism or future-experiment validation is supplied.'], 'checks': checks, 'failures': int((~check_frame.passed).sum()), 'python': platform.python_version()}
    (BASE / 'metadata/full_edge_corpus_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    if not check_frame.passed.all():
        raise SystemExit('E59 corpus checks failed')
if __name__ == '__main__':
    main()
