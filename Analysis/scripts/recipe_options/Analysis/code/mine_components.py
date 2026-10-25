"""Reconstruct dose-aware, acquisition-matched component interventions."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TYPE_TO_DATASET = {'acylation': 'Acylation', 'allylic-substitution': 'Allylic-substitution', 'buchwald': 'Buchwald-Hartwig', 'B-H (CMC)': 'Buchwald-Hartwig(CMC)', 'C-N or C-O competitive reactions': 'C-O competitive reactions', 'chan-lam': 'Chan-Lam', 'esterification': 'Esterification', 'finkelstein': 'Finkelstein', 'grignard': 'Grignard', 'heck': 'Heck', 'hiyama': 'Hiyama', 'kumada': 'Kumada', 'minisci': 'Minisci', 'pd-carbonylation': 'Pd-carbonylation', 'reductive-amination': 'Reductive-amination', 'snar': 'SNAr', 'sonogashira': 'Sonogashira', 'suzuki': 'Suzuki', 'suzuki(bpin)': 'Suzuki Bpin', 'ullmann': 'Ullmann'}
SLOTS = [f'Reagent{i}' for i in range(1, 7)] + ['Solvent', 'Temperature', 'Time']
IDENTITY_FIELDS = [f'reagent{i}_id' for i in range(1, 7)] + ['solvent_id', 'temp_id', 'time_id']
REGISTRY_FIELDS = [f'Reagent{i}_id' for i in range(1, 7)] + ['solvent_id', 'temp_id', 'time_id']

def norm_token(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return ''
    try:
        number = float(text)
        if np.isfinite(number):
            return format(number, '.12g')
    except ValueError:
        pass
    return text

def short_hash(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]

def differing_slots(first, second):
    return [slot for slot in first if first[slot] != second[slot]]

def swap_kind(first, second):
    if not first[0] or not second[0]:
        return 'addition_removal'
    if first[0] == second[0]:
        return 'dose_only'
    return 'identity_same_dose' if first[1:] == second[1:] else 'identity_and_dose'

def oracle_gain(delta):
    x = np.asarray(delta, dtype=float)
    return float((np.mean(np.abs(x)) - abs(np.mean(x))) / 2)

def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    if len(order):
        result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return result

def within_plate_delta(rows, first, second):
    keys = [x for x in ['archive', 'plate', 'acquisition', 'target_key'] if x in rows]
    grouped = rows.groupby(keys + ['recipe_id'], dropna=False)['yield_numeric'].mean().unstack('recipe_id')
    if first not in grouped or second not in grouped:
        return pd.DataFrame(columns=keys + ['y0', 'y1', 'delta'])
    result = grouped[[first, second]].dropna().rename(columns={first: 'y0', second: 'y1'}).reset_index()
    result['delta'] = result.y1 - result.y0
    return result

def registry_tokens(row):
    tokens = {}
    for i in range(1, 7):
        identity = norm_token(row[f'Reagent{i}_id'])
        tokens[f'Reagent{i}'] = (identity, norm_token(row[f'Reagent{i}_HTE_Equiv']) if identity else '')
    tokens['Solvent'] = (norm_token(row['solvent_id']),)
    tokens['Temperature'] = (norm_token(row['temp_id']),)
    tokens['Time'] = (norm_token(row['time_id']),)
    return tokens

def note_issue(dataset, condition):
    issues = {('Kumada', 'C9'): 'registry_note_NMP_amount_differs', ('Kumada', 'C13'): 'registry_note_NMP_amount_differs', ('Reductive-amination', 'C33'): 'registry_note_AcOH_amount_differs', ('Reductive-amination', 'C40'): 'registry_note_AcOH_amount_differs', ('Chan-Lam', 'C68'): 'registry_note_uncertain_base_identity'}
    return issues.get((dataset, condition), '')

def resolve_condition_ids(raw, registry_keys):
    exact = raw.dataset_id + '|' + raw.condition_id
    fallback = raw.condition_id.str.replace('^C0+(\\d+)$', 'C\\1', regex=True)
    fallback_keys = raw.dataset_id + '|' + fallback
    use_fallback = ~exact.isin(registry_keys) & fallback_keys.isin(registry_keys)
    return raw.condition_id.where(~use_fallback, fallback)

def acquisition_names(names, conditions):
    suffix = names.str.extract('-([ABC]\\d+)$', expand=False)
    canonical = lambda x: x.str.replace('^([ABC])0+(\\d+)$', '\\1\\2', regex=True)
    if suffix.isna().any() or canonical(suffix).ne(canonical(conditions)).any():
        raise ValueError('Acquisition suffix does not match the supplied condition identifier')
    return names.str.replace('-[ABC]\\d+$', '', regex=True)

def prepare_records(root):
    data = root / 'data'
    raw = pd.read_csv(data / 'raw_reactions.csv.gz', keep_default_na=False, low_memory=False)
    registry = pd.read_csv(data / 'conditions_raw.csv', keep_default_na=False)
    registry['dataset_id'] = registry.reaction_type.map(TYPE_TO_DATASET).fillna('UNUSED:' + registry.reaction_type)
    registry['tokens'] = [registry_tokens(r) for r in registry.to_dict('records')]
    registry['recipe_id'] = ['r_' + short_hash(json.dumps(t, sort_keys=True)) for t in registry.tokens]
    registry['note_issue'] = [note_issue(d, c) for d, c in zip(registry.dataset_id, registry.condition_id)]
    registry['missing_equiv'] = [any((v[0] and (not v[1]) for k, v in t.items() if k.startswith('Reagent'))) for t in registry.tokens]
    registry['registry_key'] = registry.dataset_id + '|' + registry.condition_id
    assert not registry.registry_key.duplicated().any()
    reference = registry.set_index('registry_key')
    normalized_condition = resolve_condition_ids(raw, reference.index)
    raw['condition_id_normalized'] = normalized_condition
    raw['registry_key'] = raw.dataset_id + '|' + normalized_condition
    raw['recipe_id'] = raw.registry_key.map(reference.recipe_id).fillna('')
    raw['metadata_match'] = raw.recipe_id.ne('')
    mismatch_columns = {}
    for actual, expected in zip(IDENTITY_FIELDS, REGISTRY_FIELDS):
        observed = raw[actual].map(norm_token)
        expected_values = raw.registry_key.map(reference[expected]).map(norm_token)
        mismatch = observed.ne(expected_values)
        raw['metadata_match'] &= ~mismatch
        mismatch_columns[actual] = int(mismatch.sum())
    raw['known_note_issue'] = raw.registry_key.map(reference.note_issue).fillna('missing_condition_metadata')
    early_snar = raw.dataset_id.eq('SNAr') & raw.condition_id.eq('C2') & pd.to_numeric(raw.name0.str.extract('^DP220901-(\\d+)-', expand=False), errors='coerce').lt(13)
    raw.loc[early_snar, 'known_note_issue'] = 'SNAr_C2_before_campaign13_was_DIPEA_not_NaOtAm'
    raw['source_check_true'] = raw.check_passed.astype(str).str.strip().str.lower().eq('true')
    raw['yield_numeric'] = pd.to_numeric(raw.yield_numeric, errors='coerce')
    raw['target_key'] = raw.subpair + ' => ' + raw.product_smiles.astype(str)
    raw['acquisition'] = acquisition_names(raw.name0.astype(str), raw.condition_id)
    raw['strict_eligible'] = raw.metadata_match & raw.known_note_issue.eq('') & raw.yield_numeric.between(0, 1) & raw.registry_key.map(reference.missing_equiv).eq(False) & raw.plate.ne('')
    keep = ['record_id', 'dataset_id', 'rxn_class', 'archive', 'plate', 'acquisition', 'condition_id', 'recipe_id', 'subpair', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'sub_1_cas', 'sub_2_cas', 'yield_numeric', 'source_check_true', 'metadata_match', 'known_note_issue', 'strict_eligible', 'condition_id_normalized']
    records = raw[keep].copy()
    records.to_csv(data / 'normalized_records.csv.gz', index=False, compression='gzip')
    issues = records.loc[~records.metadata_match | records.known_note_issue.ne('')]
    issues.to_csv(data / 'mapping_and_note_issues.csv.gz', index=False, compression='gzip')
    exported = registry.drop(columns='tokens').copy()
    for slot in SLOTS:
        exported[slot + '_token'] = [json.dumps(t[slot]) for t in registry.tokens]
    exported.to_csv(data / 'condition_slot_registry.csv', index=False)
    stats = {'raw_rows': len(raw), 'strict_rows': int(raw.strict_eligible.sum()), 'identity_metadata_mismatch_rows': int((~raw.metadata_match).sum()), 'mismatch_by_field': mismatch_columns, 'early_snar_C2_rows': int(early_snar.sum()), 'condition_number_alias_rows': int(raw.condition_id.ne(normalized_condition).sum()), 'source_check_false_finite_rows': int((~raw.source_check_true & raw.yield_numeric.between(0, 1)).sum()), 'note_issue_rows': raw.loc[raw.known_note_issue.ne('')].groupby('known_note_issue').size().to_dict()}
    (data / 'mapping_summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    return (records, registry, stats)

def dataset_recipe_metadata(registry, dataset, recipe_ids):
    selected = registry[registry.dataset_id.eq(dataset) & registry.recipe_id.isin(recipe_ids)]
    metadata = {}
    for recipe, group in selected.groupby('recipe_id', sort=True):
        first = group.iloc[0]
        entry = {'tokens': first.tokens, 'condition_ids': reader_tables.condition_aliases(dataset, recipe, group.condition_id), 'recipe_id': recipe}
        for i in range(1, 7):
            entry[f'Reagent{i}_name'] = first[f'Reagent{i}_Standardized Name'] or 'none'
        entry['Solvent_name'] = first['Standardized Name']
        entry['Temperature_name'] = norm_token(first.temp_id) + ' C'
        entry['Time_name'] = norm_token(first.time_id) + ' h'
        metadata[recipe] = entry
    return metadata

def compare_recipes(first, second, slot):
    a, b = (first['tokens'][slot], second['tokens'][slot])
    if a > b:
        first, second, a, b = (second, first, b, a)
    background = {s: v for s, v in first['tokens'].items() if s != slot}
    kind = swap_kind(a, b)
    if slot in {'Solvent', 'Temperature', 'Time'}:
        kind = 'solvent_identity' if slot == 'Solvent' else 'physical_parameter'
    return {'recipe0': first['recipe_id'], 'recipe1': second['recipe_id'], 'slot': slot, 'component0_id': a[0], 'component1_id': b[0], 'component0_dose': a[1] if len(a) > 1 else '', 'component1_dose': b[1] if len(b) > 1 else '', 'component0_name': first[slot + '_name'], 'component1_name': second[slot + '_name'], 'condition0_ids': first['condition_ids'], 'condition1_ids': second['condition_ids'], 'swap_kind': kind, 'background_id': 'b_' + short_hash(json.dumps(background, sort_keys=True)), 'exchange_id': 'e_' + short_hash((slot, a, b)), 'background_json': json.dumps(background, sort_keys=True)}

def summarize_delta(pair_values):
    x = pair_values.delta.to_numpy(float)
    return {'n_substrate_targets': len(x), 'mean_delta': float(np.mean(x)), 'median_abs_delta': float(np.median(np.abs(x))), 'sd_delta': float(np.std(x, ddof=1)), 'p_delta_gt_5pp': float(np.mean(x > 0.05)), 'p_delta_lt_minus5pp': float(np.mean(x < -0.05)), 'oracle_gain_observed': oracle_gain(x), 'mean_y0': float(pair_values.y0.mean()), 'mean_y1': float(pair_values.y1.mean()), 'p_zero0': float(pair_values.y0.eq(0).mean()), 'p_zero1': float(pair_values.y1.eq(0).mean()), 'n_partner1': pair_values.sub_1_smiles.nunique(), 'n_partner2': pair_values.sub_2_smiles.nunique(), 'q10_delta': float(np.quantile(x, 0.1)), 'q90_delta': float(np.quantile(x, 0.9))}

def run(root=ROOT, min_pairs=20):
    results = root / 'results'
    results.mkdir(exist_ok=True)
    records, registry, mapping_stats = prepare_records(root)
    usable = records[records.strict_eligible].copy()
    target_meta = usable.drop_duplicates(['dataset_id', 'target_key'])[['dataset_id', 'target_key', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']]
    summary_rows, archive_rows, low_rows, replicate_rows, coverage_rows = ([], [], [], [], [])
    long_path = results / 'delta_within_acquisition.csv.gz'
    long_path.unlink(missing_ok=True)
    import gzip
    stream = gzip.open(long_path, 'wt', encoding='utf-8', newline='')
    first_write = True
    try:
        for dataset, rows in usable.groupby('dataset_id', sort=True):
            index = ['archive', 'plate', 'acquisition', 'target_key']
            cell = rows.groupby(index + ['recipe_id'], dropna=False).agg(yield_numeric=('yield_numeric', 'mean'), n_source_records=('record_id', 'size'), source_check_true=('source_check_true', 'all'))
            wide = cell.yield_numeric.unstack('recipe_id')
            flag_wide = cell.source_check_true.unstack('recipe_id')
            same_plate = rows.groupby(['plate', 'target_key', 'recipe_id']).yield_numeric.agg(['size', 'std', 'min', 'max'])
            same_plate = same_plate[same_plate['size'].gt(1)].reset_index()
            same_plate.insert(0, 'dataset_id', dataset)
            replicate_rows.append(same_plate)
            meta = dataset_recipe_metadata(registry, dataset, set(wide.columns))
            comparisons = 0
            accepted = 0
            target_info = target_meta[target_meta.dataset_id.eq(dataset)].drop(columns='dataset_id').set_index('target_key')
            for rid0, rid1 in itertools.combinations(sorted(meta), 2):
                different = differing_slots(meta[rid0]['tokens'], meta[rid1]['tokens'])
                if len(different) != 1:
                    continue
                comparisons += 1
                comp = compare_recipes(meta[rid0], meta[rid1], different[0])
                pair = wide[[comp['recipe0'], comp['recipe1']]].dropna()
                if len(pair) < 3:
                    continue
                pair = pair.rename(columns={comp['recipe0']: 'y0', comp['recipe1']: 'y1'}).reset_index()
                pair['delta'] = pair.y1 - pair.y0
                flags = flag_wide[comp['recipe0']].eq(True) & flag_wide[comp['recipe1']].eq(True)
                pair['both_source_check_true'] = flags.reindex(pd.MultiIndex.from_frame(pair[index])).to_numpy()
                unique = pair.groupby('target_key')[['y0', 'y1', 'delta']].mean().join(target_info).reset_index()
                out = {'dataset_id': dataset, **comp, **summarize_delta(unique), 'n_acquisitions': len(pair), 'n_physical_plates': pair.plate.nunique(), 'n_archives': pair.archive.nunique()}
                out['pair_id'] = 'p_' + short_hash((dataset, comp['recipe0'], comp['recipe1']))
                checked = pair[pair.both_source_check_true].groupby('target_key').delta.mean()
                out['n_targets_source_check_true'] = len(checked)
                out['mean_delta_source_check_true'] = float(checked.mean()) if len(checked) else np.nan
                if len(unique) < min_pairs:
                    low_rows.append(out)
                    continue
                accepted += 1
                summary_rows.append(out)
                for archive, part in pair.groupby('archive', dropna=False):
                    per_target = part.groupby('target_key')[['y0', 'y1', 'delta']].mean().join(target_info)
                    if len(per_target) > 1:
                        archive_rows.append({'dataset_id': dataset, 'pair_id': out['pair_id'], 'archive': archive, **summarize_delta(per_target)})
                pair = pair.join(target_info, on='target_key')
                for key in ['pair_id', 'dataset_id', 'slot', 'exchange_id', 'background_id']:
                    pair[key] = out[key]
                pair.to_csv(stream, index=False, header=first_write)
                first_write = False
            coverage_rows.append({'dataset_id': dataset, 'strict_rows': len(rows), 'unique_recipes': len(meta), 'n_plates': rows.plate.nunique(), 'n_targets': rows.target_key.nunique(), 'one_slot_candidate_pairs': comparisons, 'pairs_with_at_least_20_targets': accepted})
            print(dataset, 'rows', len(rows), 'recipes', len(meta), 'matched_swaps', accepted, flush=True)
    finally:
        stream.close()
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(results / 'matched_swap_summary.csv', index=False)
    pd.DataFrame(archive_rows).to_csv(results / 'matched_swap_by_archive.csv', index=False)
    pd.DataFrame(low_rows).to_csv(results / 'low_support_swap_summary.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(results / 'dataset_matching_coverage.csv', index=False)
    pd.concat(replicate_rows, ignore_index=True).to_csv(results / 'within_plate_repeated_records.csv.gz', index=False)
    slots = summaries.groupby(['dataset_id', 'slot', 'swap_kind']).agg(matched_swaps=('pair_id', 'size'), sum_reused_targets=('n_substrate_targets', 'sum'), median_mean_delta=('mean_delta', 'median'), median_sd_delta=('sd_delta', 'median'), median_oracle_gain=('oracle_gain_observed', 'median'), median_positive_fraction=('p_delta_gt_5pp', 'median'), median_negative_fraction=('p_delta_lt_minus5pp', 'median')).reset_index()
    slots.to_csv(results / 'component_atlas.csv', index=False)
    counts = {**mapping_stats, 'matched_swaps': len(summaries), 'matched_dataset_count': summaries.dataset_id.nunique(), 'summed_reused_substrate_comparisons': int(summaries.n_substrate_targets.sum()), 'low_support_comparisons': len(low_rows), 'minimum_unique_targets': min_pairs, 'pairing_unit': 'same archive, physical plate, acquisition, substrate pair and product definition', 'summary_unit': 'equal-weight substrate-pair/product target means over matched acquisitions'}
    counts['source_check_policy'] = 'Retained as supplied source flag; primary does not presume assay QC meaning; true-flag sensitivity reported.'
    (results / 'mining_summary.json').write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(counts, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--min-pairs', type=int, default=20)
    args = parser.parse_args()
    run(args.root, args.min_pairs)
