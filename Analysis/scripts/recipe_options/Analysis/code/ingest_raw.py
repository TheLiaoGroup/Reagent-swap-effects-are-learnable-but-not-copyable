"""Extract all reaction records without qualification, deduplication, or value repair."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ANALYSIS_DIR.parents[1]
DEFAULT_RAW_DIR = RAW_DIR
REQUIRED_REACTION_COLUMNS = {'condition_id', 'sub_1_smiles', 'yield'}
PROVENANCE_COLUMNS = ['source_file', 'source_sheet', 'source_row', 'record_id', 'dataset_id', 'rxn_class', 'plate', 'subpair', 'yield_numeric', 'yield_status']
CLASS_MAP = {'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig', 'Suzuki Bpin': 'Suzuki'}
CONDITION_FILE = '0_total condition_260810.xlsx'
COMPOUND_FILE = '0_Compound ID_260810.xlsx'

def sha256(path: Path) -> str:
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def dataset_id(path: Path) -> str:
    return re.sub('_\\d{6}$', '', path.stem).rstrip('_')

def source_text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ''
    return str(value)

def stringify(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.map(source_text)

def yield_columns(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(raw, errors='coerce')
    missing = raw.map(lambda value: value is None or str(value).strip() == '')
    status = pd.Series('non_numeric', index=raw.index, dtype=object)
    finite = pd.Series(np.isfinite(numeric), index=raw.index)
    status.loc[finite] = 'finite'
    status.loc[numeric.notna() & ((numeric < 0) | (numeric > 1))] = 'outside_0_1'
    status.loc[missing] = 'missing'
    return (numeric, status)

def augment(frame: pd.DataFrame, path: Path, sheet: str) -> tuple[pd.DataFrame, list[str]]:
    collisions = set(frame.columns).intersection(PROVENANCE_COLUMNS)
    if collisions:
        raise ValueError(f'Derived columns already exist in source: {sorted(collisions)}')
    frame = stringify(frame.copy())
    absent = []
    if 'sub_2_smiles' not in frame:
        frame['sub_2_smiles'] = ''
        absent.append('sub_2_smiles')
    frame['source_file'] = path.name
    frame['source_sheet'] = sheet
    frame['source_row'] = reader_tables.source_rows(path, sheet, original=path.name.startswith('0_')) if path.is_absolute() and reader_tables.is_source(path) else np.arange(2, len(frame) + 2, dtype=np.int64)
    frame['record_id'] = [f'{path.name}:{sheet}:{row}' for row in frame['source_row']]
    frame['dataset_id'] = dataset_id(path)
    frame['rxn_class'] = CLASS_MAP.get(dataset_id(path), dataset_id(path))
    names = frame['name0'] if 'name0' in frame else pd.Series('', index=frame.index)
    frame['plate'] = names.str.extract('^(.*?-PL\\d+)(?:\\D|$)', expand=False).fillna('')
    frame['subpair'] = frame['sub_1_smiles'] + ' >> ' + frame['sub_2_smiles']
    frame['yield_numeric'], frame['yield_status'] = yield_columns(frame['yield'])
    return (frame, absent)

def read_sheets(path: Path):
    if path.suffix.lower() == '.csv':
        yield ('CSV', reader_tables.read_csv(path, dtype=object, keep_default_na=False, na_filter=False, skip_blank_lines=False, encoding='utf-8-sig'))
    else:
        with reader_tables.ExcelFile(path, engine='openpyxl') as book:
            for sheet in book.sheet_names:
                yield (sheet, reader_tables.read_excel(book, sheet_name=sheet, dtype=object, keep_default_na=False, na_filter=False))

def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(count) for key, count in frame[column].value_counts(dropna=False).items()}

def summarize_yields(frame: pd.DataFrame) -> dict:
    numeric = frame['yield_numeric']
    finite = numeric[np.isfinite(numeric)]
    return {'rows': len(frame), 'yield_status': value_counts(frame, 'yield_status'), 'explicit_zero_yield_rows': int((numeric == 0).sum()), 'finite_yield_min': float(finite.min()) if len(finite) else None, 'finite_yield_max': float(finite.max()) if len(finite) else None, 'unparsed_plate_rows': int((frame['plate'] == '').sum()), 'check_passed_values': value_counts(frame, 'check_passed'), 'archive_values': value_counts(frame, 'archive'), 'missing_sub_1_smiles_rows': int((frame['sub_1_smiles'] == '').sum()), 'missing_sub_2_smiles_rows': int((frame['sub_2_smiles'] == '').sum())}

def exception_details(frame: pd.DataFrame) -> dict:
    columns = ['record_id', 'source_file', 'source_sheet', 'source_row', 'yield', 'yield_status']
    details = {}
    for status in ('missing', 'non_numeric', 'outside_0_1'):
        subset = frame.loc[frame['yield_status'] == status]
        details[status] = {'count': len(subset), 'raw_value_counts': value_counts(subset, 'yield')}
        if status != 'missing':
            details[status]['records'] = subset[columns].to_dict('records')
    return details

def self_test() -> None:
    values = pd.Series(['0', '', ' ', 'NA', '0.5', '-0.01', '1.01', '1', 'inf', 'NaN'])
    numeric, status = yield_columns(values)
    assert status.tolist() == ['finite', 'missing', 'missing', 'non_numeric', 'finite', 'outside_0_1', 'outside_0_1', 'finite', 'outside_0_1', 'non_numeric']
    assert numeric.iloc[0] == 0 and pd.isna(numeric.iloc[1])
    assert numeric.iloc[5] == -0.01 and numeric.iloc[6] == 1.01
    raw = pd.DataFrame({'condition_id': ['001', '001'], 'sub_1_smiles': ['C', 'C'], 'yield': ['0', ''], 'name0': ['DP230501-1-1-PL2-LC1-A12', 'unparsed'], 'check_passed': ['False', 'True'], 'archive': ['NA', '0']})
    derived, absent = augment(raw, Path('Finkelstein_260731.csv'), 'CSV')
    assert len(derived) == 2 and derived['condition_id'].tolist() == ['001', '001']
    assert derived['plate'].tolist() == ['DP230501-1-1-PL2', '']
    assert derived['source_row'].tolist() == [2, 3]
    assert derived['subpair'].tolist() == ['C >> ', 'C >> ']
    assert derived['check_passed'].tolist() == ['False', 'True']
    assert derived['archive'].tolist() == ['NA', '0']
    assert absent == ['sub_2_smiles']
    assert dataset_id(Path('Suzuki Bpin_260806.csv')) == 'Suzuki Bpin'
    assert dataset_id(Path('Buchwald-Hartwig(CMC)_260805.csv')) == 'Buchwald-Hartwig(CMC)'
    assert dataset_id(Path('Kumada_.xlsx')) == 'Kumada'

def verify_output(raw: pd.DataFrame, output: Path, inventory: list[dict], registries: dict) -> dict:
    restored = pd.read_csv(output / 'raw_reactions.csv.gz', dtype=str, keep_default_na=False, na_filter=False)
    assert len(restored) == len(raw) > 0, 'Reaction row count changed during export'
    assert list(restored.columns) == list(raw.columns)
    assert restored['record_id'].is_unique, 'Duplicate provenance identifiers'
    assert int(sum((row['reaction_rows'] for row in inventory))) == len(raw)
    assert restored['yield_status'].value_counts().to_dict() == raw['yield_status'].value_counts().to_dict()
    original_columns = [column for column in raw.columns if column not in PROVENANCE_COLUMNS]
    for column in original_columns:
        expected = raw[column].map(source_text)
        assert restored[column].equals(expected), f'Original column changed on export: {column}'
    for column in ('record_id', 'source_file', 'source_sheet', 'dataset_id', 'plate', 'subpair'):
        assert restored[column].equals(raw[column].map(str)), f'Derived column changed: {column}'
    for kind, frame in registries.items():
        readback = pd.read_csv(output / f'{kind}_raw.csv', dtype=str, keep_default_na=False, na_filter=False)
        assert len(readback) == len(frame) > 0
        for column in frame:
            assert readback[column].equals(frame[column].map(source_text)), f'Registry changed: {kind}/{column}'
    return {'status': 'passed', 'reaction_rows_read_back': len(restored), 'unique_record_ids': int(restored['record_id'].nunique()), 'all_original_columns_compared_after_export': original_columns, 'registry_rows_read_back': {kind: len(frame) for kind, frame in registries.items()}}

def write_report(summary: dict, path: Path) -> None:
    lines = ['# Raw data inventory', '', 'Raw inputs: `HTE data/01_原始数据`. All input files were read directly; no published or allocated subset restricted extraction.', '', f"The extraction retained {summary['reaction_rows']:,} reaction records from {summary['reaction_dataset_count']} datasets. No rows were filtered by yield, archive, check_passed, repeated measurements, or screen composition.", '', '| Dataset | Rows | Missing yield | Non-numeric yield | Outside 0-1 | Explicit zero |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, info in summary['datasets'].items():
        counts = info['yield_status']
        lines.append(f"| {key} | {info['rows']:,} | {counts.get('missing', 0):,} | {counts.get('non_numeric', 0):,} | {counts.get('outside_0_1', 0):,} | {info['explicit_zero_yield_rows']:,} |")
    lines.extend(['', f"Conditions: {summary['registries']['conditions']['rows']:,} rows. Compounds: {summary['registries']['compounds']['rows']:,} rows.", '', '## Source fidelity', '', 'All original columns are retained. CSV values are read as text with automatic NA recognition disabled; Excel cell values are converted to their textual representation without editing the workbook. Missing yields remain blank. Explicit zero is retained. The original yield column is separate from numeric conversion.', '', '`yield_numeric` uses pandas numeric conversion without clipping or filling. `yield_status` is `finite` for finite values in [0, 1], `missing` for blank or whitespace-only values, `non_numeric` for other values that do not convert, and `outside_0_1` for converted numbers below 0 or above 1 (including infinities). All exceptional original values are counted in ingestion_summary.json; non-numeric and out-of-range records include exact provenance.', '', '`source_row` starts at 2 for the first data row, including the header as row 1. CSV rows refer to logical CSV records; blank internal rows are retained. `record_id` is filename:sheet:row. `plate` is only the name0 prefix through -PL followed by digits; unmatched names remain blank. No experimental date is inferred.', '', 'Dataset identifiers are filename stems with a terminal six-digit date and trailing underscore removed. CMC and Bpin datasets remain distinct. `rxn_class` groups Buchwald-Hartwig(CMC) with Buchwald-Hartwig and Suzuki Bpin with Suzuki. `subpair` preserves the ordered substrate SMILES separated by ` >> `.', '', '## Source schema differences', ''])
    for detail in summary['source_schema_additions']:
        lines.append(f"- {detail['source_file']} / {detail['source_sheet']}: {', '.join(detail['added_empty_columns'])} absent in the source and added as empty for the shared table interface. No second substrate is inferred.")
    for detail in summary['skipped_sheets']:
        lines.append(f"- {detail['source_file']} / {detail['source_sheet']}: {detail['rows']:,} rows inventoried as metadata; {detail['reason']}.")
    lines.extend(['', '## Outputs', '', '- `data/raw_reactions.csv.gz`: all reaction records, all original columns, and provenance/derived columns.', '- `data/conditions_raw.csv` and `data/compounds_raw.csv`: all rows from the first sheet of each source registry, with provenance.', '- `data/raw_file_inventory.csv`: SHA-256, file size, every sheet, row counts and sheet disposition.', '- `data/ingestion_summary.json`: schemas, counts, raw yield exceptions and verification results.', '', 'The completed export was read back and every original column compared with the extracted source values. Input SHA-256 values were checked again after extraction. No input file was changed.', '', 'Reproduce with the bundled Python runtime: `python -u code/ingest_raw.py` from this Analysis directory. The script also provides `--self-test` for conversion/provenance invariants.', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument('--output-dir', type=Path, default=ANALYSIS_DIR / 'data')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('Conversion and provenance invariant tests passed.', flush=True)
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((path for path in reader_tables.source_paths(args.raw_dir) if path.suffix.lower() in ('.csv', '.xlsx', '.xls')))
    if not files:
        raise FileNotFoundError(f'No source files found: {args.raw_dir}')
    inventory, reactions, skipped, additions, errors = ([], [], [], [], [])
    registries, registry_info = ({}, {})
    schemas = []
    for number, path in enumerate(files, start=1):
        print(f'[{number}/{len(files)}] Reading {path.name}', flush=True)
        file_info = {'source_file': path.name, 'sha256': sha256(path), 'size_bytes': reader_tables.source_evidence(path)['bytes'], 'file_type': path.suffix.lower(), 'reaction_rows': 0, 'registry_rows': 0, 'sheets': [], 'sheet_rowcounts': {}, 'sheet_dispositions': {}, 'status': 'ok'}
        try:
            for index, (sheet, frame) in enumerate(read_sheets(path)):
                original_columns = list(frame.columns)
                file_info['sheets'].append(sheet)
                file_info['sheet_rowcounts'][sheet] = len(frame)
                schemas.append({'source_file': path.name, 'source_sheet': sheet, 'rows': len(frame), 'original_columns': original_columns})
                registry_kind = 'conditions' if path.name == CONDITION_FILE else 'compounds' if path.name == COMPOUND_FILE else None
                if registry_kind and index == 0:
                    extracted = stringify(frame)
                    extracted['source_file'] = path.name
                    extracted['source_sheet'] = sheet
                    extracted['source_row'] = reader_tables.source_rows(path, sheet, original=path.name.startswith('0_')) if path.is_absolute() and reader_tables.is_source(path) else np.arange(2, len(frame) + 2, dtype=np.int64)
                    registries[registry_kind] = extracted
                    registry_info[registry_kind] = {'source_file': path.name, 'source_sheet': sheet, 'rows': len(frame), 'original_columns': original_columns}
                    file_info['registry_rows'] += len(frame)
                    file_info['sheet_dispositions'][sheet] = 'registry_first_sheet_retained'
                elif REQUIRED_REACTION_COLUMNS.issubset(frame.columns):
                    extracted, absent = augment(frame, path, sheet)
                    reactions.append(extracted)
                    file_info['reaction_rows'] += len(frame)
                    file_info['sheet_dispositions'][sheet] = 'all_reaction_rows_retained'
                    if absent:
                        additions.append({'source_file': path.name, 'source_sheet': sheet, 'added_empty_columns': absent})
                else:
                    reason = 'missing reaction columns: ' + ', '.join(sorted(REQUIRED_REACTION_COLUMNS - set(frame.columns)))
                    skipped.append({'source_file': path.name, 'source_sheet': sheet, 'rows': len(frame), 'reason': reason, 'original_columns': original_columns})
                    file_info['sheet_dispositions'][sheet] = 'metadata_not_a_reaction_table'
                print(f"    {sheet}: {len(frame):,} rows ({file_info['sheet_dispositions'][sheet]})", flush=True)
        except Exception as error:
            errors.append({'source_file': path.name, 'error_type': type(error).__name__, 'message': str(error)})
            file_info['status'] = 'error'
            print(f'    ERROR: {type(error).__name__}: {error}', flush=True)
        inventory.append(file_info)
    if errors or set(registries) != {'conditions', 'compounds'} or (not reactions):
        failure = {'status': 'failed', 'errors': errors, 'registries_found': sorted(registries), 'inventory': inventory}
        (output / 'ingestion_summary.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError('Ingestion incomplete; see ingestion_summary.json')
    raw = pd.concat(reactions, ignore_index=True, sort=False)
    for column in raw:
        if column != 'yield_numeric':
            raw[column] = raw[column].map(lambda value: '' if pd.isna(value) else value)
    assert raw['record_id'].is_unique
    print(f'Writing {len(raw):,} reaction records and two registries', flush=True)
    raw.to_csv(output / 'raw_reactions.csv.gz', index=False, encoding='utf-8', compression={'method': 'gzip', 'mtime': 0})
    for kind, frame in registries.items():
        frame.to_csv(output / f'{kind}_raw.csv', index=False, encoding='utf-8')
    inventory_frame = pd.DataFrame(inventory)
    for column in ('sheets', 'sheet_rowcounts', 'sheet_dispositions'):
        inventory_frame[column] = inventory_frame[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    inventory_frame.to_csv(output / 'raw_file_inventory.csv', index=False, encoding='utf-8')
    verification = verify_output(raw, output, inventory, registries)
    for info in inventory:
        assert sha256(args.raw_dir / info['source_file']) == info['sha256'], f"Source changed: {info['source_file']}"
    verification['all_input_sha256_unchanged_after_extraction'] = True
    summary = {'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_directory': str(args.raw_dir.resolve()), 'source_file_count': len(files), 'reaction_dataset_count': int(raw['dataset_id'].nunique()), 'reaction_rows': len(raw), 'reaction_columns': list(raw.columns), 'provenance_and_derived_columns': PROVENANCE_COLUMNS, 'source_schemas': schemas, 'source_schema_additions': additions, 'registries': registry_info, 'skipped_sheets': skipped, 'errors': errors, 'overall': summarize_yields(raw), 'datasets': {key: summarize_yields(frame) for key, frame in raw.groupby('dataset_id', sort=True)}, 'yield_exceptions': exception_details(raw), 'verification': verification, 'definitions': {'source_row': '1-based sheet/logical CSV record row including header; first data row is 2', 'record_id': 'source_file:source_sheet:source_row', 'dataset_id': 'filename stem with terminal _YYMMDD date and trailing underscore removed', 'rxn_class': 'dataset_id except Buchwald-Hartwig(CMC) -> Buchwald-Hartwig and Suzuki Bpin -> Suzuki', 'plate': 'name0 prefix through first -PL followed by digits; unmatched left empty', 'subpair': "ordered raw sub_1_smiles + ' >> ' + raw sub_2_smiles", 'yield_numeric': 'pandas numeric conversion, errors coerced to NaN, no clipping or filling', 'yield_status': 'finite=in-range finite; missing=blank/whitespace; non_numeric=not convertible; outside_0_1=numeric below zero or above one, including infinities', 'original_values': 'CSV parsed as text with automatic NA recognition disabled; Excel values represented as text; no source files changed', 'row_policy': 'all reaction rows retained, including duplicate measurements and every check_passed/archive value'}}
    (output / 'ingestion_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_report(summary, ANALYSIS_DIR / '06_RAW_DATA_INVENTORY.md')
    print(json.dumps({'status': summary['status'], 'reaction_rows': len(raw), 'datasets': summary['reaction_dataset_count'], 'yield_status': summary['overall']['yield_status'], 'registry_rows': verification['registry_rows_read_back'], 'verification': 'passed'}, ensure_ascii=True), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
