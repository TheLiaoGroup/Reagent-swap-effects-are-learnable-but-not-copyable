"""Fixed E65 identifiers, target geometry, paths and file identities."""
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
E64 = ANALYSIS / 'Target_Geometry_E64'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
FOLD = [*EDGE, 'fold_role1', 'fold_role2']
GEOMETRIES = ['raw_delta', 'centered_delta', 'centered_unit_rms_zeroembed']
REGIMES = ['original', 'family'] + [f'thin_{r}' for r in range(5)]
METRICS = ['native_vector_mse', 'native_vector_mae', 'unit0_vector_mse', 'unit0_vector_mae', 'mean_offset_mse', 'centered_error_mse']
ALPHA = 10.0
KERNEL = 'MorganR2_2048_noChirality_roleAdditiveTanimoto'
INPUTS = [E62 / f'results/e62_{n}.parquet' for n in ['vectors', 'membership', 'folds', 'condition_cells']] + [E62 / 'results/e62_support_bridge_1548.csv', E62 / 'results/e62_support_classes_20.csv', E62 / 'metadata/e62_metadata.json', E62 / 'code/matched_representation.py', E64 / 'results/e64_predictions.parquet', E64 / 'metadata/e64_metadata.json', ANALYSIS / 'Target_Geometry_E64/code/target_geometry.py', Path(__file__)]
PLAN_NAMES = ['molecule_families', 'family_requests', 'requests', 'factors', 'scope_vectors']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def pins(paths):
    return {str(Path(p).relative_to(ROOT)): sha(p) for p in paths}

def payload(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii')

def identifier(value):
    return hashlib.sha256(payload(value)).hexdigest()

def array_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()

def write_json(path, value):

    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, default=default, allow_nan=False), encoding='utf-8')

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

def factor_id(key, train):
    return identifier(['E65_factor_v1', *key, [int(i) for i in train], KERNEL, ALPHA, 'unpenalized_intercept'])

def operator_id(factor, query):
    return identifier(['E65_operator_v1', factor, [int(i) for i in query]])

def size_stratum(n):
    return 'n0' if n == 0 else 'n1' if n == 1 else 'n2to4' if n < 5 else 'n5plus'

def edge_tag(key):
    return 'edge_' + identifier(['E65_edge_v1', *key])[:16]
