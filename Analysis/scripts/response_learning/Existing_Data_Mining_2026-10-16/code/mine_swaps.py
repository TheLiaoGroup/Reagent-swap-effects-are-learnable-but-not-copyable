"""Enumerate reaction-local full-recipe-controlled swaps and matched yield deltas."""
from __future__ import annotations
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
RUN = Path(__file__).resolve().parents[1]
IDENTITIES = [f'reagent{i}_id' for i in range(1, 7)]
DOSES = [f'reagent{i}_equiv' for i in range(1, 7)]
MIN_PAIRS = 20
THRESHOLDS = [15, 20, 30]

def equal(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b

def condition_order(value):
    return tuple((int(piece) if piece.isdigit() else piece for piece in re.split('(\\d+)', str(value))))

def recipe_json(row):
    fields = ['condition_id', *IDENTITIES, *DOSES, 'solvent_id', 'solvent_name', 'temp_id', 'time_id']
    fields += [f'reagent{i}_name' for i in range(1, 7)]
    return json.dumps({field: None if pd.isna(row[field]) else row[field] for field in fields}, sort_keys=True, ensure_ascii=True)

def classify(a, b):
    if not (equal(a.temp_id, b.temp_id) and equal(a.time_id, b.time_id)):
        return None
    id_changes = [i for i in range(6) if not equal(a[IDENTITIES[i]], b[IDENTITIES[i]])]
    dose_changes = [i for i in range(6) if not equal(a[DOSES[i]], b[DOSES[i]])]
    solvent_changes = not equal(a.solvent_id, b.solvent_id)
    if len(id_changes) == 1 and (not solvent_changes):
        slot = id_changes[0]
        if any((i != slot for i in dose_changes)):
            return None
        return (f'reagent{slot + 1}', 'identity_and_dose' if slot in dose_changes else 'strict_identity_only')
    if not id_changes and len(dose_changes) == 1 and (not solvent_changes):
        return (f'reagent{dose_changes[0] + 1}', 'dose_only')
    if not id_changes and (not dose_changes) and solvent_changes:
        return ('solvent', 'solvent_only')
    return None

def unique_join(values):
    return '||'.join(sorted(set((str(x) for x in values.dropna()))))

def stats(y0, y1, prefix=''):
    delta = 100 * (np.asarray(y1) - np.asarray(y0))
    n = len(delta)
    result = {prefix + 'n_pairs': n, prefix + 'mean_delta_pp': float(np.mean(delta)) if n else np.nan, prefix + 'sd_delta_pp': float(np.std(delta, ddof=1)) if n > 1 else np.nan, prefix + 'median_delta_pp': float(np.median(delta)) if n else np.nan, prefix + 'mean_abs_delta_pp': float(np.mean(np.abs(delta))) if n else np.nan}
    for threshold in THRESHOLDS:
        positive = int((delta >= threshold).sum())
        negative = int((delta <= -threshold).sum())
        neutral = n - positive - negative
        result.update({prefix + f'n_pos_{threshold}': positive, prefix + f'n_neg_{threshold}': negative, prefix + f'n_near0_{threshold}': neutral, prefix + f'frac_pos_{threshold}': positive / n if n else np.nan, prefix + f'frac_neg_{threshold}': negative / n if n else np.nan, prefix + f'frac_near0_{threshold}': neutral / n if n else np.nan, prefix + f'balanced_support_{threshold}': 2 * min(positive, negative) / n if n else np.nan})
        assert positive + negative + neutral == n
    return result

def build_cells(observations):
    frame = observations.loc[observations.eligible_for_matching].copy()
    keys = ['reaction_type', 'condition_id', 'substrate_pair_key']
    aggregation = {'y': ('yield', 'mean'), 'n_obs': ('observation_id', 'size'), 'duplicate_cell': ('duplicate_measurement_cell', 'max'), 'check_false': ('check_passed', lambda x: bool(x.eq(False).any()))}
    fields = ['sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive', 'batch_name0_prefix', 'batch_name0_plate_token', 'name0', 'source_file', 'source_sheet', 'observation_id', 'sub_1_FG', 'sub_2_FG']
    for field in fields:
        aggregation[field] = (field, unique_join)
    repeated = frame.duplicated(keys, keep=False)
    singletons = frame.loc[~repeated, keys + ['yield', 'duplicate_measurement_cell', 'check_passed', *fields]].copy()
    singletons = singletons.rename(columns={'yield': 'y', 'duplicate_measurement_cell': 'duplicate_cell'})
    singletons['n_obs'] = 1
    singletons['check_false'] = singletons.check_passed.eq(False)
    singletons = singletons.drop(columns='check_passed')
    for field in fields:
        singletons[field] = singletons[field].fillna('')
    duplicates = frame.loc[repeated].groupby(keys, sort=False, dropna=False).agg(**aggregation).reset_index()
    return pd.concat([singletons, duplicates], ignore_index=True)

def main():
    conditions = pd.read_parquet(RUN / 'processed' / 'conditions.parquet')
    observations = pd.read_parquet(RUN / 'processed' / 'observations.parquet')
    cells = build_cells(observations)
    cells.to_parquet(RUN / 'processed' / 'matching_cells.parquet', index=False)
    candidates, summaries, rejections = ([], [], [])
    for reaction, group in conditions.groupby('reaction_type', sort=True):
        group = group.loc[group.recipe_complete]
        rows = [row for _, row in group.iterrows()]
        rows.sort(key=lambda row: condition_order(row.condition_id))
        for a, b in itertools.combinations(rows, 2):
            classification = classify(a, b)
            if classification is None:
                continue
            slot, match_class = classification
            swap_id = f'{reaction}__{a.condition_id}__{b.condition_id}'
            item = dict(swap_id=swap_id, reaction_type=reaction, condition0=a.condition_id, condition1=b.condition_id, swap_slot=slot, match_class=match_class, recipe0=recipe_json(a), recipe1=recipe_json(b), recipe_complete=True, missing_recipe_fields='')
            if slot == 'solvent':
                item.update(component0_id=a.solvent_id, component1_id=b.solvent_id, component0_name=a.solvent_name, component1_name=b.solvent_name, dose0=np.nan, dose1=np.nan, change_kind='exchange')
            else:
                item.update(component0_id=a[slot + '_id'], component1_id=b[slot + '_id'], component0_name=a[slot + '_name'], component1_name=b[slot + '_name'], dose0=a[slot + '_equiv'], dose1=b[slot + '_equiv'], change_kind='addition' if pd.isna(a[slot + '_id']) else 'removal' if pd.isna(b[slot + '_id']) else 'exchange')
            candidates.append(item)
    print(f'Recipe-controlled candidate swaps: {len(candidates):,}', flush=True)
    writer = None
    delta_path = RUN / 'results' / 'swap_deltas.parquet'
    n_delta_rows = 0
    cell_groups = {key: group.drop(columns=['reaction_type', 'condition_id']) for key, group in cells.groupby(['reaction_type', 'condition_id'], sort=False)}
    for number, item in enumerate(candidates):
        reaction, c0, c1 = (item['reaction_type'], item['condition0'], item['condition1'])
        left, right = (cell_groups.get((reaction, c0)), cell_groups.get((reaction, c1)))
        if left is None or right is None:
            rejections.append(dict(**item, n_common_before_product_filter=0, n_product_disagreement=0, n_pairs=0, rejection='condition_has_no_eligible_yields'))
            continue
        paired = left.merge(right, on='substrate_pair_key', suffixes=('0', '1'), how='inner', validate='one_to_one')
        n_common = len(paired)
        product_disagreement = paired.product_smiles0.ne(paired.product_smiles1)
        n_disagree = int(product_disagreement.sum())
        paired = paired.loc[~product_disagreement].copy()
        if len(paired) < MIN_PAIRS:
            rejections.append(dict(**item, n_common_before_product_filter=n_common, n_product_disagreement=n_disagree, n_pairs=len(paired), rejection='fewer_than_20_common_pairs'))
            continue
        paired['delta_pp'] = 100 * (paired.y1 - paired.y0)
        paired['sub_1_smiles'] = paired.sub_1_smiles0
        paired['sub_2_smiles'] = paired.sub_2_smiles0
        paired['swap_id'] = item['swap_id']
        paired['reaction_type'] = reaction
        paired['condition0'], paired['condition1'] = (c0, c1)
        summary = dict(item)
        summary.update(stats(paired.y0, paired.y1))
        both_positive = paired.y0.gt(0) & paired.y1.gt(0)
        summary.update(stats(paired.loc[both_positive, 'y0'], paired.loc[both_positive, 'y1'], 'both_positive_'))
        nonduplicate = ~paired.duplicate_cell0 & ~paired.duplicate_cell1
        summary.update(stats(paired.loc[nonduplicate, 'y0'], paired.loc[nonduplicate, 'y1'], 'unique_cells_'))
        summary.update(n_common_before_product_filter=n_common, n_product_disagreement=n_disagree, n_sub1=paired.sub_1_smiles.nunique(), n_sub2=paired.loc[paired.sub_2_smiles.ne(''), 'sub_2_smiles'].nunique(), n_duplicated_pairs=int((~nonduplicate).sum()), n_missing_products=int(paired.product_smiles0.eq('').sum()), frac_zero0=float(paired.y0.eq(0).mean()), frac_zero1=float(paired.y1.eq(0).mean()), frac_either_zero=float((paired.y0.eq(0) | paired.y1.eq(0)).mean()), frac_both_zero=float((paired.y0.eq(0) & paired.y1.eq(0)).mean()), n_check_false_pairs=int((paired.check_false0 | paired.check_false1).sum()), archives0=unique_join(paired.archive0), archives1=unique_join(paired.archive1), n_archive_groups0=paired.archive0.nunique(), n_archive_groups1=paired.archive1.nunique(), n_same_archive_literal=int((paired.archive0.eq(paired.archive1) & paired.archive0.ne('')).sum()), n_same_name0_prefix_literal=int((paired.batch_name0_prefix0.eq(paired.batch_name0_prefix1) & paired.batch_name0_prefix0.ne('')).sum()), n_same_plate_token_literal=int((paired.batch_name0_plate_token0.eq(paired.batch_name0_plate_token1) & paired.batch_name0_plate_token0.ne('')).sum()))
        for threshold in THRESHOLDS:
            for sign, mask in [('pos', paired.delta_pp.ge(threshold)), ('neg', paired.delta_pp.le(-threshold))]:
                summary[f'n_sub1_{sign}_{threshold}'] = paired.loc[mask, 'sub_1_smiles'].nunique()
                summary[f'n_sub2_{sign}_{threshold}'] = paired.loc[mask & paired.sub_2_smiles.ne(''), 'sub_2_smiles'].nunique()
        summaries.append(summary)
        paired = paired.rename(columns={'batch_name0_prefix0': 'name0_prefix0', 'batch_name0_prefix1': 'name0_prefix1'})
        paired = paired.drop(columns=['sub_1_smiles0', 'sub_1_smiles1', 'sub_2_smiles0', 'sub_2_smiles1'])
        for column in paired.select_dtypes(include=['object', 'string']).columns:
            paired[column] = paired[column].astype('string')
        table = pa.Table.from_pandas(paired, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(delta_path, table.schema, compression='zstd')
        writer.write_table(table)
        n_delta_rows += len(paired)
        if len(summaries) % 100 == 0:
            print(f'Retained {len(summaries):,} swaps / {n_delta_rows:,} deltas', flush=True)
    if writer:
        writer.close()
    registry = pd.DataFrame(summaries)
    registry = registry.sort_values(['balanced_support_20', 'n_pairs', 'mean_abs_delta_pp'], ascending=False).reset_index(drop=True)
    registry['balanced_support_rank_20'] = np.arange(1, len(registry) + 1)
    registry.to_csv(RUN / 'results' / 'swap_registry.csv', index=False)
    pd.DataFrame(rejections).to_csv(RUN / 'audit' / 'swap_exclusions.csv', index=False)
    metadata = dict(n_recipe_controlled_candidates=len(candidates), n_swaps_retained=len(registry), n_matched_deltas=n_delta_rows, minimum_common_pairs=MIN_PAIRS, delta_definition='100 * (mean_fraction_yield_condition1 - mean_fraction_yield_condition0)', orientation='Natural alphanumeric condition-ID order within reaction class', threshold_boundary='positive delta>=T; negative delta<=-T; near0 abs(delta)<T', ranking='Descending 2*min(n_pos_20,n_neg_20)/n_pairs, then pair count, then mean absolute delta', duplicate_policy='Eligible repeated measurements averaged in each substrate-pair/condition cell. All duplicate cells flagged; unique_cells_ sensitivity drops them.', product_policy='Exclude within-cell product conflicts and across-condition product disagreements; missing equal product labels retained and counted.', recipe_policy='Hold all other reagent identities/equivalents, solvent, temperature and time exactly fixed. Changed-slot dose may differ, labeled identity_and_dose. Active-dose missing fields excluded. Structural empty slots count as absence.', provenance_policy='archive and parsed name0 fields are literal identifiers, not verified experimental batches.')
    (RUN / 'audit' / 'swap_mining_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = pd.read_csv(RUN / 'audit' / 'input_manifest.csv')
    from prepare_data import sha256
    manifest['sha256_after_mining'] = manifest.absolute_path.map(lambda value: sha256(Path(value)))
    manifest['unchanged_after_mining'] = manifest.sha256_before.eq(manifest.sha256_after_mining)
    manifest.to_csv(RUN / 'audit' / 'input_manifest.csv', index=False)
    assert manifest.unchanged_after_mining.all()
    assert (registry.n_pos_20 + registry.n_neg_20 + registry.n_near0_20).eq(registry.n_pairs).all()
    print(json.dumps(metadata, indent=2))
    print(registry[['swap_id', 'match_class', 'component0_name', 'component1_name', 'dose0', 'dose1', 'n_pairs', 'n_pos_20', 'n_neg_20', 'balanced_support_20']].head(20).to_string(index=False))
if __name__ == '__main__':
    main()
