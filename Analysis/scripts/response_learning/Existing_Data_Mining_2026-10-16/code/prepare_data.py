"""Read immutable HTE inputs and retain row-level provenance and quality flags."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
RUN = Path(__file__).resolve().parents[1]
WORKSPACE = RUN.parents[2]
import sys as _m1_sys
_m1_sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
RAW = RAW_DIR
ALIASES = {'acylation': 'acylation', 'allylic-substitution': 'allylic-substitution', 'buchwald': 'buchwald-hartwig', 'buchwald-hartwig': 'buchwald-hartwig', 'b-h (cmc)': 'buchwald-hartwig-cmc', 'buchwald-hartwig(cmc)': 'buchwald-hartwig-cmc', 'c-n or c-o competitive reactions': 'c-o-competitive', 'c-o competitive reactions': 'c-o-competitive', 'suzuki(bpin)': 'suzuki-bpin', 'suzuki bpin': 'suzuki-bpin'}
IDENTITIES = [f'reagent{i}_id' for i in range(1, 7)] + ['solvent_id']
MATCH_FIELDS = IDENTITIES + ['temp_id', 'time_id']

def normalize_reaction(value):
    value = str(value).strip().lower()
    return ALIASES.get(value, value)

def sha256(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def clean_text(series):
    return series.astype('string').str.strip().replace('', pd.NA)

def canonical(series):
    return series.astype('string').fillna('<ABSENT>')

def make_conditions(path):
    tables = reader_tables.read_excel(path, sheet_name=None)
    chunks = []
    schemas = []
    for sheet, frame in tables.items():
        schemas.append(dict(source_file=path.name, source_sheet=sheet, n_rows=len(frame), n_columns=len(frame.columns), role='condition_registry', columns_json=json.dumps(frame.columns.tolist())))
        if 'condition_id' not in frame:
            continue
        frame['source_row'] = reader_tables.source_rows(path, sheet, original=path.name.startswith('0_')) if path.is_absolute() and reader_tables.is_source(path) else np.arange(2, len(frame) + 2)
        frame['source_sheet'] = sheet
        frame['source_file'] = path.name
        frame['reaction_type_original'] = frame['reaction_type']
        frame['reaction_type'] = frame['reaction_type'].map(normalize_reaction)
        rename = {'Standardized Name': 'solvent_name'}
        for i in range(1, 7):
            rename.update({f'Reagent{i}_id': f'reagent{i}_id', f'Reagent{i}_Standardized Name': f'reagent{i}_name', f'Reagent{i}_HTE_CAS': f'reagent{i}_cas', f'Reagent{i}_HTE_Equiv': f'reagent{i}_equiv'})
        frame = frame.rename(columns=rename)
        for column in ['condition_id', *IDENTITIES]:
            frame[column] = clean_text(frame[column])
        missing = [[] for _ in range(len(frame))]
        for i in range(1, 7):
            identity, dose = (f'reagent{i}_id', f'reagent{i}_equiv')
            frame[dose] = pd.to_numeric(frame[dose], errors='coerce')
            active = frame[identity].notna()
            orphan = frame[identity].isna() & (frame[dose].notna() | frame[f'reagent{i}_name'].notna())
            for index in np.flatnonzero(active & frame[dose].isna() | orphan):
                missing[index].append(f'reagent{i}_identity_or_equiv')
        for column in ['solvent_id', 'temp_id', 'time_id']:
            for index in np.flatnonzero(frame[column].isna()):
                missing[index].append(column)
        frame['missing_recipe_fields'] = [';'.join(items) for items in missing]
        frame['recipe_complete'] = frame['missing_recipe_fields'].eq('')
        chunks.append(frame)
    result = pd.concat(chunks, ignore_index=True)
    if result.duplicated(['reaction_type', 'condition_id']).any():
        raise ValueError('Nonunique reaction-local condition IDs in registry')
    return (result, schemas)

def main():
    for folder in ['processed', 'audit', 'results']:
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    sources = sorted((p for p in reader_tables.source_paths(RAW) if p.suffix.lower() in {'.xlsx', '.csv'}))
    manifest = [dict(source_file=p.name, absolute_path=str(p.resolve()), bytes=reader_tables.source_evidence(p)['bytes'], mtime_ns=reader_tables.source_evidence(p)['mtime_ns'], sha256_before=sha256(p)) for p in sources]
    total = next((p for p in sources if p.name.startswith('0_total')))
    conditions, schemas = make_conditions(total)
    frames = []
    for path in sources:
        if path.name.startswith('0_'):
            if path != total:
                for sheet, frame in reader_tables.read_excel(path, sheet_name=None).items():
                    schemas.append(dict(source_file=path.name, source_sheet=sheet, n_rows=len(frame), n_columns=len(frame.columns), role='compound_reference', columns_json=json.dumps(frame.columns.tolist())))
            continue
        tables = reader_tables.read_excel(path, sheet_name=None) if path.suffix.lower() == '.xlsx' else {'CSV': reader_tables.read_csv(path, low_memory=False)}
        reaction = normalize_reaction(re.sub('_.*$', '', path.stem))
        for sheet, frame in tables.items():
            is_observations = {'condition_id', 'sub_1_smiles', 'yield'}.issubset(frame.columns)
            schemas.append(dict(source_file=path.name, source_sheet=sheet, n_rows=len(frame), n_columns=len(frame.columns), role='observations' if is_observations else 'auxiliary_skipped', columns_json=json.dumps(frame.columns.tolist())))
            if not is_observations:
                continue
            frame['reaction_type'] = reaction
            frame['source_file'] = path.name
            frame['source_sheet'] = sheet
            frame['source_row'] = reader_tables.source_rows(path, sheet, original=path.name.startswith('0_')) if path.is_absolute() and reader_tables.is_source(path) else np.arange(2, len(frame) + 2)
            frame['observation_id'] = [f'{path.name}|{sheet}|{r}' for r in frame['source_row']]
            frame['sub_2_missing_by_design'] = reaction == 'finkelstein'
            frames.append(frame)
            print(f'Read {path.name}: {sheet}: {len(frame):,}', flush=True)
    observations = pd.concat(frames, ignore_index=True)
    for column in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'name0', 'archive', 'sub_1_FG', 'sub_2_FG', *IDENTITIES]:
        if column not in observations:
            observations[column] = pd.NA
        observations[column] = clean_text(observations[column])
    observations['yield'] = pd.to_numeric(observations['yield'], errors='coerce')
    observations['yield_missing'] = observations['yield'].isna()
    observations['yield_out_of_range'] = observations['yield'].notna() & ~observations['yield'].between(0, 1)
    observations['yield_zero'] = observations['yield'].eq(0)
    observations['yield_finite'] = np.isfinite(observations['yield'])
    observations['substrate_pair_key'] = observations['sub_1_smiles'].fillna('<MISSING>') + '||' + observations['sub_2_smiles'].fillna('<SINGLE>')
    observations['batch_name0_prefix'] = observations['name0'].str.extract('^(.*?)-PL[^-]+-LC[^-]+-[^-]+$', expand=False)
    observations['batch_name0_plate_token'] = observations['name0'].str.extract('^(.*?-PL[^-]+)-LC[^-]+-[^-]+$', expand=False)
    observations['name0_suffix_matches_condition'] = observations['name0'].str.rsplit('-', n=1).str[-1].eq(observations['condition_id']).fillna(False)
    joined = observations[['reaction_type', 'condition_id', *MATCH_FIELDS]].merge(conditions[['reaction_type', 'condition_id', 'recipe_complete', *MATCH_FIELDS]], how='left', on=['reaction_type', 'condition_id'], suffixes=('_obs', '_registry'), indicator=True, validate='many_to_one')
    observations['recipe_joined'] = joined['_merge'].eq('both').to_numpy()
    observations['recipe_complete'] = joined['recipe_complete'].fillna(False).astype(bool).to_numpy()
    mismatch_fields = []
    for field in MATCH_FIELDS:
        if field in {'temp_id', 'time_id'}:
            a = pd.to_numeric(joined[field + '_obs'], errors='coerce')
            b = pd.to_numeric(joined[field + '_registry'], errors='coerce')
            mismatch = ~(a.eq(b) | a.isna() & b.isna())
        else:
            mismatch = canonical(joined[field + '_obs']).ne(canonical(joined[field + '_registry']))
        observations[f'recipe_mismatch_{field}'] = mismatch.to_numpy() & observations['recipe_joined']
        mismatch_fields.append(f'recipe_mismatch_{field}')
    observations['recipe_consistent'] = observations['recipe_joined'] & ~observations[mismatch_fields].any(axis=1)
    observations['eligible_for_matching'] = observations['yield_finite'] & ~observations['yield_out_of_range'] & observations['recipe_consistent'] & observations['recipe_complete'] & observations['sub_1_smiles'].notna() & (observations['sub_2_smiles'].notna() | observations['sub_2_missing_by_design'])
    keys = ['reaction_type', 'substrate_pair_key', 'condition_id']
    observations['n_replicates_cell'] = observations.groupby(keys, dropna=False)['observation_id'].transform('size')
    observations['duplicate_measurement_cell'] = observations['n_replicates_cell'].gt(1)
    cells = observations.groupby(keys, dropna=False).agg(n_records=('observation_id', 'size'), n_measured=('yield', 'count'), yield_min=('yield', 'min'), yield_max=('yield', 'max'), n_products=('product_smiles', 'nunique'), n_archives=('archive', 'nunique'), source_rows=('observation_id', lambda x: '||'.join(x))).reset_index()
    cells.loc[cells.n_records.gt(1)].to_csv(RUN / 'audit' / 'duplicate_cells.csv', index=False)
    observations['cell_product_conflict'] = observations.set_index(keys).index.isin(cells.loc[cells.n_products.gt(1)].set_index(keys).index)
    observations['eligible_for_matching'] &= ~observations['cell_product_conflict']
    quality = []
    for reaction, frame in observations.groupby('reaction_type', sort=True):
        entry = dict(reaction_type=reaction, n_rows=len(frame), n_conditions=frame.condition_id.nunique(), n_substrate_pairs=frame.substrate_pair_key.nunique(), n_sub1=frame.sub_1_smiles.nunique(), n_sub2=frame.sub_2_smiles.nunique(), n_archives=frame.archive.nunique(), n_name0_prefixes=frame.batch_name0_prefix.nunique(), n_check_passed_true=int(frame.check_passed.eq(True).sum()), n_check_passed_false=int(frame.check_passed.eq(False).sum()))
        for flag in ['yield_missing', 'yield_out_of_range', 'yield_zero', 'recipe_joined', 'recipe_complete', 'recipe_consistent', 'eligible_for_matching', 'duplicate_measurement_cell', 'cell_product_conflict', *mismatch_fields]:
            entry['n_' + flag] = int(frame[flag].sum())
        quality.append(entry)
    for column in observations:
        if observations[column].dtype == object:
            observations[column] = observations[column].astype('string')
    for column in conditions:
        if conditions[column].dtype == object:
            conditions[column] = conditions[column].astype('string')
    observations.to_parquet(RUN / 'processed' / 'observations.parquet', index=False)
    conditions.to_csv(RUN / 'processed' / 'conditions.csv', index=False)
    conditions.to_parquet(RUN / 'processed' / 'conditions.parquet', index=False)
    pd.DataFrame(quality).to_csv(RUN / 'audit' / 'data_quality.csv', index=False)
    pd.DataFrame(schemas).to_csv(RUN / 'audit' / 'input_schemas.csv', index=False)
    observations.loc[observations.recipe_joined & ~observations.recipe_consistent, ['observation_id', 'reaction_type', 'condition_id', *MATCH_FIELDS, *mismatch_fields]].to_csv(RUN / 'audit' / 'recipe_mismatches.csv', index=False)
    observations.loc[~observations.recipe_joined, ['reaction_type', 'condition_id', 'source_file']].value_counts().rename('n_rows').reset_index().to_csv(RUN / 'audit' / 'unjoined_conditions.csv', index=False)
    for item in manifest:
        item['sha256_after'] = sha256(RAW / item['source_file'])
        item['unchanged'] = item['sha256_before'] == item['sha256_after']
    pd.DataFrame(manifest).to_csv(RUN / 'audit' / 'input_manifest.csv', index=False)
    assert all((item['unchanged'] for item in manifest)), 'Raw input checksum changed'
    metadata = dict(n_observations=len(observations), n_condition_rows=len(conditions), n_reaction_files=len(sources) - 2, reaction_aliases=ALIASES, matching_policy='Valid fraction yield [0,1], complete joined consistent recipe, valid substrate fields, no within-cell product conflict. check_passed retained; not assumed analytical acceptance.', duplicate_policy='All raw rows retained. Mining means eligible replicates within reaction/substrate pair/condition; sensitivity excludes duplicate cells.', missing_slot_policy='Empty ID/name/equiv is structural absence. Active ID with missing dose, or orphan name/dose, is an incomplete recipe.', name0_policy='Prefix and PL tokens are literal identifiers only; no date, batch independence, or confirmation inferred.', yield_policy='Raw fraction yields preserved. Missing not zero; out-of-range not clipped and excluded from matching.')
    (RUN / 'audit' / 'ingestion_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(quality).to_string(index=False))
if __name__ == '__main__':
    main()
