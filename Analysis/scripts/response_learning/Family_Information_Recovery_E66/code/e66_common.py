"""Frozen E66 paths, identities and target conventions."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
ROOT = ANALYSIS.parent
E62 = ANALYSIS / 'Matched_Representation_E62'
E65 = ANALYSIS / 'Structure_Family_Transport_E65'
OLD = ANALYSIS / 'Existing_Data_Mining_2026-10-16'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
BUDGETS = ['0', '1', '2', '4', '8', '16', 'full']
LEARNERS = ['joint', 'calibration_only', 'calibration_centroid']
GEOMETRIES = ['raw_delta', 'centered_delta', 'centered_unit_rms_zeroembed']
ALPHA = 10.0
KERNEL = 'MorganR2_2048_noChirality_roleAdditiveTanimoto'
PLAN_INPUTS = [E62 / 'results/e62_vectors.parquet', E62 / 'results/e62_condition_cells.parquet', E65 / 'results/e65_family_requests.parquet', E65 / 'results/e65_scope_vectors.parquet', E65 / 'results/e65_molecule_families.parquet', E65 / 'metadata/e65_preflight_metadata.json', E65 / 'metadata/e65_metadata.json', E65 / 'results/e65_union_predictions.parquet', E65 / 'results/e65_factor_diagnostics.parquet', OLD / 'processed/observations.parquet', Path(__file__), BASE / 'metadata/e66_feasibility_metadata.json']

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def pins(paths):
    return {str(Path(p).relative_to(ROOT)): sha(p) for p in paths}

def payload(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii')

def identifier(value):
    return hashlib.sha256(payload(value)).hexdigest()

def array_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()

def write_json(path, value):

    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, default=convert, allow_nan=False), encoding='utf-8')

def factor_id(key, train):
    return identifier(['E66_factor_v1', *key, [int(i) for i in train], KERNEL, ALPHA, 'unpenalized_intercept'])

def e65_factor_id(key, train):
    return identifier(['E65_factor_v1', *key, [int(i) for i in train], KERNEL, ALPHA, 'unpenalized_intercept'])

def edge_tag(key):
    return 'edge_' + identifier(['E66_edge_v1', *key])[:16]

def e65_edge_tag(key):
    return 'edge_' + identifier(['E65_edge_v1', *key])[:16]

def center_and_unit(values):
    x = np.asarray(values, float)
    constant = np.all(x == x[:, :1], axis=1)
    mean = x[:, 0] + (x - x[:, :1]).mean(axis=1)
    centered = x - mean[:, None]
    centered[constant] = 0
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    if not np.array_equal(rms == 0, constant):
        raise ValueError('Nonconstant centered RMS underflow')
    unit = np.zeros_like(centered)
    np.divide(centered, rms[:, None], out=unit, where=~constant[:, None])
    return (centered, unit, constant, rms)

def prediction_blocks(block, width):
    return dict(prediction_raw=block[:width].tolist(), prediction_centered=block[width:2 * width].tolist(), prediction_unit=block[2 * width:].tolist())
