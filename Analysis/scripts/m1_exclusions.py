"""Read and reconcile the authoritative measurement exclusions for M1."""
from __future__ import annotations
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterable

def ids_sha256(ids: Iterable[str]) -> str:
    """Hash sorted IDs encoded as UTF-8 lines, including the final newline."""
    values = sorted(ids)
    return hashlib.sha256(''.join((value + '\n' for value in values)).encode('utf-8')).hexdigest()

@dataclass(frozen=True)
class Exclusions:
    source_path: Path
    source_sha256: str
    rows: tuple[dict[str, str], ...]
    ids: frozenset[str]
    id_sha256: str

def load_exclusions(path: Path | str | None=None) -> Exclusions:
    if path is None:
        path = os.environ.get('M1_EXCLUDE_CSV')
    if path is None:
        main = Path(__file__).resolve().parents[2]
        path = main / 'Input/EXCLUDE_wells.csv'
    source = Path(path).resolve()
    raw = source.read_bytes()
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'name0', 'plate'}.issubset(reader.fieldnames or []):
            raise ValueError(f'Exclusion CSV requires name0 and plate columns: {source}')
        rows = tuple((dict(row) for row in reader))
    if not rows or any((not row['name0'] or not row['plate'] for row in rows)):
        raise ValueError(f'Exclusion CSV has no IDs or blank name0/plate: {source}')
    counts = Counter((row['name0'] for row in rows))
    if any((count != 1 for count in counts.values())):
        raise ValueError(f'Exclusion CSV has duplicate name0 IDs: {source}')
    ids = frozenset(counts)
    return Exclusions(source, hashlib.sha256(raw).hexdigest(), rows, ids, ids_sha256(ids))

def exclusion_mask(frame, exclusions: Exclusions | None=None, name_column: str='name0'):
    exclusions = exclusions if exclusions is not None else load_exclusions()
    if name_column not in frame:
        raise ValueError(f'Reaction table is missing measurement ID column {name_column}')
    return frame[name_column].isin(exclusions.ids)

def assert_exclusions_reconciled(matched_ids: Iterable[str], exclusions: Exclusions) -> None:
    counts = Counter(matched_ids)
    if not counts:
        raise ValueError('Exclusion CSV matched no measurement rows in the raw corpus')
    missing = exclusions.ids - counts.keys()
    unexpected = counts.keys() - exclusions.ids
    duplicates = sorted((value for value, count in counts.items() if count != 1))
    if missing or unexpected or duplicates:
        raise ValueError(f'Exclusion reconciliation failed: missing={sorted(missing)}, unexpected={sorted(unexpected)}, duplicate={duplicates}')

def exclusion_summary(matched_ids: Iterable[str], exclusions: Exclusions) -> dict:
    matched = sorted(matched_ids)
    if set(matched) - exclusions.ids:
        raise ValueError('Matched ID is absent from the exclusion CSV')
    plate = {row['name0']: row['plate'] for row in exclusions.rows}
    by_plate = dict(sorted(Counter((plate[value] for value in matched)).items()))
    return dict(source_path=str(exclusions.source_path), source_sha256=exclusions.source_sha256, id_sha256=exclusions.id_sha256, count=len(exclusions.ids), matched_count=len(matched), deleted_wells=len(matched), matched_ids=matched, matched_ids_sha256=ids_sha256(matched), per_plate=by_plate, by_plate=by_plate)
