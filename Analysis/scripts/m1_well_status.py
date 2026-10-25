"""Write the per-well table with a measurement status for every well.

One row per retained reaction record after the authoritative exclusions.
``measurement_status`` takes four values:

- quantified: a numerical yield was recorded;
- zero_by_rule: blank yield and DESI call ``yield_clf`` = 0 (no product detected);
  recorded as yield 0;
- detected_not_quantified: blank yield and ``yield_clf`` = 1 (product detected,
  no relative yield could be obtained from UPLC); yield left blank;
- no_screen_record: blank yield and no DESI call; yield left blank.

Writes ``out/M1_well_measurement_status.csv.gz``."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import re
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_raw import RAW_DIR
from m1_exclusions import assert_exclusions_reconciled, exclusion_mask, exclusion_summary, load_exclusions
OUT = Path(__file__).resolve().parents[1] / 'out' / 'M1_well_measurement_status.csv.gz'
NEEDED = {'condition_id', 'sub_1_smiles', 'yield'}
KEEP = ['name0', 'condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'yield', 'yield_clf']

def tables():
    for path in sorted((p for p in reader_tables.source_paths(RAW_DIR) if p.suffix.lower() in {'.csv', '.xlsx'} and (not p.name.startswith('0_')))):
        sheets = reader_tables.read_excel(path, sheet_name=None, profile='original') if path.suffix.lower() == '.xlsx' else {'CSV': reader_tables.read_csv(path, low_memory=False, profile='original')}
        for sheet, frame in sheets.items():
            if NEEDED.issubset(frame.columns):
                yield (path, sheet, frame)

def build_wells(source_tables, exclusions):
    parts = []
    matched_ids = []
    source_records = 0
    for path, sheet, frame in source_tables:
        source_records += len(frame)
        excluded = exclusion_mask(frame, exclusions, name_column='name0')
        matched_ids.extend(frame.loc[excluded, 'name0'].tolist())
        source_rows = reader_tables.source_rows(path, sheet, original=True)[~excluded.to_numpy()]
        frame = frame.loc[~excluded]
        out = pd.DataFrame({'source_file': path.name, 'source_sheet': sheet, 'source_row': source_rows, 'reaction_table': re.sub('_.*$', '', path.stem)})
        for column in KEEP:
            out[column] = frame[column].to_numpy() if column in frame else pd.NA
        parts.append(out)
    assert_exclusions_reconciled(matched_ids, exclusions)
    wells = pd.concat(parts, ignore_index=True)
    raw_yield = pd.to_numeric(wells['yield'], errors='coerce')
    call = pd.to_numeric(wells['yield_clf'], errors='coerce')
    if wells['yield'].notna().sum() != raw_yield.notna().sum():
        raise ValueError('non-numeric yield entries')
    if not raw_yield.dropna().between(0, 1).all():
        raise ValueError('yield outside 0-1')
    blank = raw_yield.isna()
    status = np.select([~blank, blank & call.eq(0), blank & call.eq(1), blank & call.isna()], ['quantified', 'zero_by_rule', 'detected_not_quantified', 'no_screen_record'], default='')
    if (status == '').any():
        raise ValueError(f"{int((status == '').sum())} wells have a DESI call other than 0 or 1")
    wells['yield_fraction_0_1'] = raw_yield.where(~blank, np.where(status == 'zero_by_rule', 0.0, np.nan))
    wells['measurement_status'] = status
    wells = wells.drop(columns='yield')[['source_file', 'source_sheet', 'source_row', 'reaction_table', 'name0', 'condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'yield_clf', 'yield_fraction_0_1', 'measurement_status']]
    if wells.name0.isin(exclusions.ids).any():
        raise ValueError('Excluded measurement remains in the per-well status table')
    summary = exclusion_summary(matched_ids, exclusions)
    summary.update(source_records=source_records, retained_records=len(wells))
    return (wells, summary)

def main():
    exclusions = load_exclusions()
    wells, summary = build_wells(tables(), exclusions)
    counts = wells.measurement_status.value_counts()
    if len(wells) != 410983 - len(exclusions.ids) or wells.reaction_table.nunique() != 20 or set(counts.index) != {'quantified', 'zero_by_rule', 'detected_not_quantified', 'no_screen_record'}:
        raise ValueError(f'unexpected corpus shape: {len(wells)} rows, {counts.to_dict()}')
    OUT.parent.mkdir(exist_ok=True)
    wells.to_csv(OUT, index=False, compression={'method': 'gzip', 'mtime': 0})
    back = pd.read_csv(OUT)
    if len(back) != len(wells) or not back.measurement_status.value_counts().equals(counts):
        raise ValueError('read-back mismatch')
    OUT.with_name('M1_well_measurement_status_exclusions.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(back.head(3).to_string())
    print(counts.to_string())
    print(pd.crosstab(back.reaction_table, back.measurement_status).to_string())
if __name__ == '__main__':
    main()
