"""Read source tables without filtering, repairing, or averaging observations.

Only the user-authorized raw directory is read. CSV exports are text preserving;
Excel exports preserve cached cell values as text. Derived fields are separate."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
import sys as _m1_sys
_m1_sys.path.append(str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR as _M1_RAW_DIR
ROOT = HERE.parents[1]
RAW = _M1_RAW_DIR
DATA = HERE / 'data'
REQUIRED = {'condition_id', 'sub_1_smiles', 'yield'}
CONDITION_FILE = '0_total condition_260810.xlsx'
COMPOUND_FILE = '0_Compound ID_260810.xlsx'
TYPE_MAP = {'acylation': 'Acylation', 'allylic-substitution': 'Allylic-substitution', 'buchwald': 'Buchwald-Hartwig', 'B-H (CMC)': 'Buchwald-Hartwig(CMC)', 'C-N or C-O competitive reactions': 'C-O competitive reactions', 'chan-lam': 'Chan-Lam', 'esterification': 'Esterification', 'finkelstein': 'Finkelstein', 'grignard': 'Grignard', 'heck': 'Heck', 'hiyama': 'Hiyama', 'kumada': 'Kumada', 'minisci': 'Minisci', 'pd-carbonylation': 'Pd-carbonylation', 'reductive-amination': 'Reductive-amination', 'snar': 'SNAr', 'sonogashira': 'Sonogashira', 'suzuki(bpin)': 'Suzuki Bpin', 'suzuki': 'Suzuki', 'ullmann': 'Ullmann'}

def sha256(path: Path) -> str:
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def source_text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ''
    return str(value)

def class_name(path: Path) -> str:
    return re.sub('_\\d{6}$', '', path.stem).rstrip('_')

def read_tables(path: Path):
    if path.suffix.lower() == '.csv':
        yield ('CSV', reader_tables.read_csv(path, dtype=object, encoding='utf-8-sig', keep_default_na=False, na_filter=False, skip_blank_lines=False))
    else:
        with reader_tables.ExcelFile(path, engine='openpyxl') as book:
            for sheet in book.sheet_names:
                yield (sheet, reader_tables.read_excel(book, sheet_name=sheet, dtype=object, keep_default_na=False, na_filter=False))

def provenance(frame: pd.DataFrame, path: Path, sheet: str) -> pd.DataFrame:
    frame = frame.map(source_text)
    for field in ('source_file', 'source_sheet', 'source_row', 'record_id'):
        if field in frame:
            raise ValueError(f'Source column collision: {path.name}/{sheet}/{field}')
    frame['source_file'] = path.name
    frame['source_sheet'] = sheet
    frame['source_row'] = reader_tables.source_rows(path, sheet, original=path.name.startswith('0_')) if path.is_absolute() and reader_tables.is_source(path) else np.arange(2, len(frame) + 2, dtype=np.int64)
    frame['record_id'] = frame['source_row'].map(lambda row: f'{path.name}:{sheet}:{row}')
    return frame

def augment_reactions(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    for field in ('sub_2_smiles', 'name0', 'archive', 'check_passed'):
        if field not in frame:
            frame[field] = ''
    for field in [f'reagent{i}_id' for i in range(1, 7)] + ['solvent_id', 'temp_id', 'time_id']:
        if field not in frame:
            frame[field] = ''
    frame['reaction_class'] = class_name(path)
    frame['pair_id'] = frame['sub_1_smiles'] + ' >> ' + frame['sub_2_smiles']
    frame['raw_yield'] = frame['yield']
    numeric = pd.to_numeric(frame['yield'], errors='coerce')
    finite = np.isfinite(numeric)
    frame['yield_numeric'] = numeric
    frame['yield_finite'] = finite
    status = pd.Series('non_numeric', index=frame.index, dtype=object)
    status.loc[frame['yield'].str.strip().eq('')] = 'missing'
    status.loc[numeric.notna() & ~finite] = 'non_finite'
    status.loc[finite] = 'finite_in_0_1'
    status.loc[finite & ~numeric.between(0, 1)] = 'finite_outside_0_1'
    frame['yield_status'] = status
    frame['qc_pass'] = frame['check_passed'].str.strip().str.lower().isin(['true', '1', '1.0'])
    frame['campaign_id'] = frame['name0'].str.replace('-PL\\d+.*$', '', regex=True)
    frame['plate_id'] = frame['name0'].str.extract('^(.*?-PL\\d+)(?:\\D|$)', expand=False).fillna('')
    return frame

def sorted_json(values) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=True, separators=(',', ':'))

def pair_registry(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cls, pair), group in raw.groupby(['reaction_class', 'pair_id'], sort=True):
        finite = group.loc[group['yield_finite']]
        rows.append({'reaction_class': cls, 'pair_id': pair, 'sub_1_smiles': group['sub_1_smiles'].iloc[0], 'sub_2_smiles': group['sub_2_smiles'].iloc[0], 'n_raw_records': len(group), 'n_finite_records': len(finite), 'n_conditions_all': group['condition_id'].nunique(), 'n_conditions_finite': finite['condition_id'].nunique(), 'n_duplicate_cell_all_records': int(group['duplicate_cell_all'].sum()), 'n_duplicate_cell_finite_records': int(group['duplicate_cell_finite'].sum()), 'n_qc_pass_records': int(group['qc_pass'].sum()), 'condition_ids_all': sorted_json(group['condition_id']), 'condition_ids_finite': sorted_json(finite['condition_id']), 'source_files': sorted_json(group['source_file']), 'archives': sorted_json(group['archive']), 'campaign_ids': sorted_json(group['campaign_id'])})
    return pd.DataFrame(rows)

def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    frames, inventory, registries, metadata = ([], [], {}, [])
    files = sorted((p for p in reader_tables.source_paths(RAW) if p.suffix.lower() in ('.csv', '.xlsx')))
    if any(('PRIVATE_never_release' in str(p) for p in files)):
        raise ValueError('Private source outside authorized scope')
    before = {p.name: sha256(p) for p in files}
    for path in files:
        for sheet, frame in read_tables(path):
            original_columns = list(frame.columns)
            kind = 'reaction' if REQUIRED.issubset(frame.columns) and (not path.name.startswith('0_')) else 'metadata'
            if path.name == CONDITION_FILE:
                kind = 'condition_registry'
            if path.name == COMPOUND_FILE:
                kind = 'compound_registry'
            inventory.append({'source_file': path.name, 'source_sheet': sheet, 'sha256': before[path.name], 'size_bytes': reader_tables.source_evidence(path)['bytes'], 'table_kind': kind, 'n_rows': len(frame), 'n_fields': len(original_columns), 'fields_json': json.dumps(original_columns, ensure_ascii=True), 'reaction_class': class_name(path) if kind == 'reaction' else ''})
            frame = provenance(frame, path, sheet)
            if kind == 'reaction':
                frames.append(augment_reactions(frame, path))
            elif kind in ('condition_registry', 'compound_registry'):
                registries.setdefault(kind, []).append(frame)
            else:
                metadata.append({'source_file': path.name, 'source_sheet': sheet, 'rows': len(frame)})
            print(f'Read {path.name} / {sheet}: {len(frame):,} {kind} rows', flush=True)
    raw = pd.concat(frames, ignore_index=True).fillna('')
    raw['yield_numeric'] = pd.to_numeric(raw['yield_numeric'], errors='coerce')
    raw['yield_finite'] = np.isfinite(raw['yield_numeric'])
    keys = ['reaction_class', 'pair_id', 'condition_id']
    raw['duplicate_cell_all'] = raw.duplicated(keys, keep=False)
    raw['duplicate_cell_finite'] = False
    finite = raw['yield_finite']
    raw.loc[finite, 'duplicate_cell_finite'] = raw.loc[finite].duplicated(keys, keep=False)
    raw['unique_finite_cell'] = finite & ~raw['duplicate_cell_all']
    leading = ['source_file', 'source_sheet', 'source_row', 'record_id', 'reaction_class', 'archive', 'pair_id', 'sub_1_smiles', 'sub_2_smiles', 'condition_id', 'yield', 'raw_yield', 'yield_numeric', 'yield_finite', 'yield_status']
    raw = raw[leading + [field for field in raw.columns if field not in leading]]
    raw.to_csv(DATA / 'raw_long.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(inventory).to_csv(DATA / 'raw_file_inventory.csv', index=False)
    condition = pd.concat(registries['condition_registry'], ignore_index=True)
    condition['reaction_class'] = condition['reaction_type'].map(TYPE_MAP).fillna('')
    condition.to_csv(DATA / 'raw_condition_table.csv', index=False)
    compound = pd.concat(registries['compound_registry'], ignore_index=True)
    compound.to_csv(DATA / 'raw_compound_table.csv', index=False)
    print(f'Saved raw_long: {len(raw):,} observations', flush=True)
    pair = pair_registry(raw)
    pair.to_csv(DATA / 'pair_group_registry.csv', index=False)
    readback = pd.read_csv(DATA / 'raw_long.csv.gz', dtype=str, keep_default_na=False, na_filter=False)
    assert len(readback) == len(raw) and readback['record_id'].is_unique
    original_fields = set().union(*(json.loads(item['fields_json']) for item in inventory if item['table_kind'] == 'reaction'))
    for field in original_fields:
        assert readback[field].equals(raw[field].map(source_text)), f'Export changed source field: {field}'
    after = {p.name: sha256(p) for p in files}
    assert before == after, 'Source hashes changed during extraction'
    summary = {'date': '2026-10-16', 'python': sys.version, 'source_directory': str(RAW), 'files': len(files), 'tables': len(inventory), 'reaction_files': int(pd.DataFrame(inventory).query("table_kind == 'reaction'")['source_file'].nunique()), 'reaction_rows': len(raw), 'condition_rows': len(condition), 'compound_rows': len(compound), 'reaction_classes': sorted(raw['reaction_class'].unique()), 'pair_groups': len(pair), 'yield_status': raw['yield_status'].value_counts().to_dict(), 'duplicate_cell_all_records': int(raw['duplicate_cell_all'].sum()), 'duplicate_cell_finite_records': int(raw['duplicate_cell_finite'].sum()), 'metadata_tables_not_reaction_records': metadata, 'frozen_screen_definition': str(HERE.parent / 'external' / 'hte_common.py'), 'screen_assignment': 'Deferred to downstream import of frozen hte_common; pair_group_registry does not assign screens.', 'preservation': 'All original reaction fields compared after CSV round trip; all raw source SHA-256 checked unchanged.', 'script_sha256': sha256(Path(__file__))}
    (DATA / 'raw_audit_manifest.json').write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
if __name__ == '__main__':
    main()
