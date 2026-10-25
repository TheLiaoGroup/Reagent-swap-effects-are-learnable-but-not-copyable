"""Finalize the frozen E62 producer with explicit NumPy JSON conversion.

Default execution only reuses fully verified checkpoints. --allow-fitting is
the complete cold/partial reproduction entry point using the same frozen fit
source, with fitting and reuse reported separately in final provenance."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
FROZEN_SHA = 'e821296e5f468affcc8346f48b16feb3f3058ee937b608b13bc40dd2764d2ac2'

def invocation_provenance(*, allow_fitting: bool, preexisting_checkpoints: int) -> dict:
    if allow_fitting:
        return dict(execution_mode='cold_fit' if preexisting_checkpoints == 0 else 'partial_fit', elapsed_seconds_interpretation='This invocation: fitting, checkpoint validation and aggregation through provenance assembly; preexisting checkpoint fits, if any, are excluded from the current wall time.')
    return dict(execution_mode='checkpoint_only', elapsed_seconds_interpretation='This invocation: checkpoint validation plus resume/aggregation, not the original full-fit wall time.', original_invocation_status='All 92 checkpoint fits and aggregate outputs completed; final metadata serialization failed on numpy.bool_', original_last_checkpoint_reported_elapsed_seconds=101.7)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-fitting', action='store_true', help='Permit cold or partial reproduction; default forbids refitting')
    args = parser.parse_args()
    source = BASE / 'code/matched_representation.py'
    spec = importlib.util.spec_from_file_location('e62_frozen_producer', source)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    if producer.sha(source) != FROZEN_SHA:
        raise ValueError('Frozen executed fit source changed')
    input_pins = producer.pins(producer.INPUTS)
    checkpoints = sorted((BASE / 'parts').glob('edge_*_metadata.json'))
    verified = {}
    for path in checkpoints:
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta['identity']['producer_sha256'] != FROZEN_SHA or meta['identity']['inputs'] != input_pins:
            raise ValueError(f'Checkpoint source/input mismatch: {path.name}')
        tag = path.name.removesuffix('_metadata.json')
        for name, digest in meta['output_sha256'].items():
            part = path.parent / f'{tag}_{name}.parquet'
            if not part.is_file() or producer.sha(part) != digest:
                raise ValueError(f'Checkpoint part mismatch: {part.name}')
        verified[path.name] = producer.sha(path)
    if not args.allow_fitting and len(verified) != 93:
        raise ValueError(f'Default finalization requires 93 verified complete checkpoints; found {len(verified)}')
    original_dumps, original_run_edge, original_argv = (json.dumps, producer.run_edge, sys.argv)
    reuse_count, fitted_count = (0, 0)

    def native_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

    def dumps_native(*values, **kwargs):
        kwargs.setdefault('default', native_scalar)
        return original_dumps(*values, **kwargs)

    def guarded_run(panel, gram, lookup, before):
        nonlocal reuse_count, fitted_count
        tag = f"edge_{panel['number']:04d}_{producer.hashlib.sha256('|'.join(panel['key']).encode()).hexdigest()[:12]}"
        path = BASE / 'parts' / f'{tag}_metadata.json'
        expected = dict(inputs=before, producer_sha256=FROZEN_SHA, edge=list(panel['key']), vector_ids_hash=producer.array_hash(panel['vector_ids']))
        valid = path.name in verified and json.loads(path.read_text(encoding='utf-8'))['identity'] == expected
        if not valid and (not args.allow_fitting):
            raise ValueError(f'Refitting forbidden: checkpoint identity changed for {tag}')
        result = original_run_edge(panel, gram, lookup, before)
        reuse_count += int(valid)
        fitted_count += int(not valid)
        return result
    started = time.perf_counter()
    try:
        json.dumps = dumps_native
        producer.run_edge = guarded_run
        sys.argv = [str(source)]
        producer.main()
    finally:
        json.dumps, producer.run_edge, sys.argv = (original_dumps, original_run_edge, original_argv)
    after = {path.name: producer.sha(path) for path in (BASE / 'parts').glob('edge_*_metadata.json')}
    unchanged = all((after.get(name) == digest for name, digest in verified.items()))
    if not args.allow_fitting and (not unchanged or fitted_count or reuse_count != 93):
        raise ValueError('Checkpoint-only finalization changed execution provenance')
    metadata_path = BASE / 'metadata/e62_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    invocation = invocation_provenance(allow_fitting=args.allow_fitting, preexisting_checkpoints=len(verified))
    metadata['elapsed_seconds_interpretation'] = invocation['elapsed_seconds_interpretation']
    metadata['finalization_provenance'] = dict(frozen_executed_fit_source=str(source.relative_to(producer.ROOT)), frozen_executed_fit_source_sha256=FROZEN_SHA, finalizer_source_sha256=producer.sha(Path(__file__)), finalizer_method='External NumPy scalar JSON default; frozen producer bytes unchanged', **invocation, verified_preexisting_checkpoints=len(verified), reused_edges_this_invocation=reuse_count, fitted_edges_this_invocation=fitted_count, preexisting_checkpoint_hashes_unchanged=unchanged, checkpoint_fit_seconds_sum=sum((json.loads(path.read_text(encoding='utf-8'))['seconds'] for path in (BASE / 'parts').glob('edge_*_metadata.json'))), checkpoint_fit_seconds_sum_source='Sum of checkpoint metadata seconds, including any reused checkpoints from earlier invocations', finalizer_total_seconds=time.perf_counter() - started, finalizer_total_seconds_source='time.perf_counter from immediately before producer.main through provenance assembly', successful_reproduction_command='python scripts/guard.py scripts/response_learning/Matched_Representation_E62/code/finalize_e62.py --allow-fitting')
    metadata_path.write_text(original_dumps(metadata, indent=2), encoding='utf-8')
    print(f'Finalized E62: {reuse_count} reused, {fitted_count} fitted; source and checkpoint provenance retained', flush=True)
if __name__ == '__main__':
    main()
