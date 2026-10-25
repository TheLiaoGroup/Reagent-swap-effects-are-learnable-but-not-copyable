"""Count retained acylation measurement IDs shared with Zhang et al. (2025)."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
from pathlib import Path
import pandas as pd
from m1_raw import M1_ROOT, RAW_DIR
PUBLIC_COMMIT = 'f6fdbe283fc24af53389aab8dcba5e46386ac9c0'
PUBLIC_URL = 'https://raw.githubusercontent.com/aichemeco/amide_coupling/' + PUBLIC_COMMIT + '/data/all_HTE_with_condition.csv'
PUBLIC_SHA256 = '5085e1ae50ee26173bd170a4b10a99c4e3101a46b706d115e425aff17c091b41'
REFERENCE = M1_ROOT / 'Analysis/auxiliary_inputs/zhang_2025.csv'

def overlap_counts(current, published):
    sets = []
    for label, frame in (('retained', current), ('published', published)):
        if frame.empty or 'name0' not in frame:
            raise ValueError(f'{label}: missing measurement IDs')
        ids = frame.name0
        if ids.isna().any() or ids.astype(str).str.strip().eq('').any() or ids.duplicated().any():
            raise ValueError(f'{label}: blank or duplicate measurement IDs')
        sets.append(set(ids.astype(str)))
    retained, public = sets
    if not retained & public:
        raise ValueError('No published measurement IDs match the retained acylation corpus')
    return dict(retained_measurement_ids=len(retained), published_measurement_ids=len(public), overlapping_measurement_ids=len(retained & public), retained_only_measurement_ids=len(retained - public), published_only_measurement_ids=len(public - retained))

def reference():
    if not REFERENCE.is_file():
        raise FileNotFoundError('The packaged auxiliary input Analysis/auxiliary_inputs/zhang_2025.csv is required')
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != PUBLIC_SHA256:
        raise ValueError('Packaged published reference digest differs from the frozen source')
    return pd.read_csv(REFERENCE, low_memory=False)

def main():
    from m1_exclusions import exclusion_mask, load_exclusions
    paths = sorted(reader_tables.source_paths(RAW_DIR, pattern='Acylation_*.xlsx'))
    if len(paths) != 1:
        raise ValueError(f'Expected one acylation workbook, found {len(paths)}')
    frame = reader_tables.read_excel(paths[0])
    exclusions = load_exclusions()
    if exclusion_mask(frame, exclusions).any():
        raise ValueError('Published overlap must be counted after exclusion preprocessing')
    counts = overlap_counts(frame, reference())
    counts.update(reference_url=PUBLIC_URL, reference_commit=PUBLIC_COMMIT, reference_sha256=PUBLIC_SHA256, source_file=paths[0].name, reference_input=REFERENCE.relative_to(M1_ROOT).as_posix(), reference_access='packaged auxiliary input; no network', exclusion_sha256=exclusions.source_sha256, definition='Exact measurement-ID (name0) overlap; current SMILES are retained as written')
    out = M1_ROOT / 'Analysis/out/zhang_2025_overlap.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(counts, indent=2))
if __name__ == '__main__':
    main()
