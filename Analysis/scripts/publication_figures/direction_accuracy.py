"""Reuse direction exports only when their full scientific provenance matches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from publication_figures import direction_accuracy_exporter as exporter
ANALYSIS = exporter.ANALYSIS
OUTPUT = exporter.OUTPUT
OUTPUT_NAMES = {'direction_predictions_raw.csv.gz', 'direction_predictions_targets.csv.gz', 'direction_accuracy_summary.csv', 'direction_accuracy_edge.csv', 'direction_accuracy_context.csv', 'direction_accuracy_screen.csv', 'direction_accuracy_family.csv', 'learned_training_relationships.csv', 'selected_training_pairs.csv'}

def validate(output=OUTPUT):
    """Bind the cached predictions to the unchanged exporter and every input."""
    output = Path(output)
    report = json.loads((output / 'direction_accuracy_verification.json').read_text(encoding='utf-8'))
    if report.get('status') != 'passed' or report.get('exporter_sha256') != exporter.digest(Path(exporter.__file__)):
        raise ValueError('Direction export does not match the verified exporter')
    if report.get('producer_sha256') != exporter.digest(exporter.PRODUCER):
        raise ValueError('Direction export producer changed')
    if Path(report.get('producer', '')).resolve() != exporter.PRODUCER.resolve():
        raise ValueError('Direction export belongs to another analysis directory')
    expected = {'training_fits': 85, 'strict_fold_edge_rows': 3275, 'raw_prediction_rows': 121383, 'unique_target_rows': 121383, 'fold_score_comparison_rows': 3275, 'n_screens': 17, 'n_families': 14, 'n_contexts': 213, 'training_points': 452}
    if any((report.get(key) != value for key, value in expected.items())):
        raise ValueError('Direction export population or replay checks changed')
    if not 0 <= report.get('maximum_sse_bridge_error', float('inf')) <= 1e-10:
        raise ValueError('Direction export does not reproduce the original fold scores')
    inputs = report.get('input_hashes', {})
    if not inputs or inputs.get(str(exporter.PRODUCER)) != report['producer_sha256']:
        raise ValueError('Direction export lacks its producer-bound input inventory')
    root = ANALYSIS.resolve()
    fixed = {exporter.PRODUCER, ANALYSIS / 'scripts/reference_learning/Analysis/data/cv_fold_registry.csv', ANALYSIS / 'scripts/reference_learning/Analysis/Nature_Manuscript/qa/matched_copy_fold_scores.csv.gz', *[ANALYSIS / 'scripts/reference_learning/Analysis/tables' / name for name in ('chemical_observation_channels_target_registry.csv', 'chemical_observation_channels_panel_registry.csv', 'chemical_mode_training_sample_registry.csv')], *[ANALYSIS / 'scripts/reference_learning/Analysis' / name for name in ('chemical_mode_recovery.py', 'chemical_observation_channels.py', 'chemically_aligned_measurements.py', 'shrinkage_covariance_analysis.py')]}
    actual_paths = {Path(value).resolve() for value in inputs}
    if not {path.resolve() for path in fixed}.issubset(actual_paths):
        raise ValueError('Direction export omits a fixed scientific input')
    matrix_paths = actual_paths - {path.resolve() for path in fixed}
    if len(matrix_paths) != 17 or any((path.parent != (ANALYSIS / 'scripts/reference_learning/Analysis/data').resolve() or not path.name.startswith('matrix_') or path.suffix != '.npz' for path in matrix_paths)):
        raise ValueError('Direction export matrix input inventory changed')
    for value, expected_hash in inputs.items():
        path = Path(value).resolve()
        if not path.is_relative_to(root) or exporter.digest(path) != expected_hash:
            raise ValueError(f'Direction scientific input changed: {value}')
    outputs = report.get('output_hashes', {})
    if set(outputs) != OUTPUT_NAMES:
        raise ValueError('Direction export output inventory is incomplete')
    for name, expected_hash in outputs.items():
        if exporter.digest(output / name) != expected_hash:
            raise ValueError(f'Direction export output changed: {name}')
    summary = pd.read_csv(output / 'direction_accuracy_summary.csv')
    if summary.empty:
        raise ValueError('Direction export summary is empty')
    return report

def ensure(output=OUTPUT, allow_replay=True):
    """A clean package recreates absent/stale exports through the original replay."""
    output = Path(output)
    try:
        return validate(output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not allow_replay:
            raise ValueError(f'Direction cache cannot be reused: {error}') from error
        print(f'Direction export requires unchanged replay: {error}', flush=True)
        exporter.replay(output)
        return validate(output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    report = ensure(args.output, allow_replay=not args.verify_only)
    print(json.dumps({'status': 'passed', 'validated_outputs': len(report['output_hashes']), 'scientific_inputs': len(report['input_hashes'])}, indent=2))
