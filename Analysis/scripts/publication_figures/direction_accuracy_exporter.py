"""Repeat-0 direction summaries from the unchanged strict matched-copy producer.

No predictor, split, parameter or mask is modified.
The diagnostic callback is broadened to export individual predictions. Every
replayed fold-edge SSE/count/mask hash is checked against the saved producer."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import numpy as np
import pandas as pd
ANALYSIS = Path(__file__).resolve().parents[2]
PRODUCER = ANALYSIS / 'scripts/reference_learning/Analysis/Nature_Manuscript/compute_matched_copy_comparison.py'
OUTPUT = ANALYSIS / 'out/direction_metrics'
MODEL_COLUMNS = ['marginal_prediction_pp', 'direct_raw_prediction_pp', 'offset_raw_prediction_pp', 'semantic_full_raw_prediction_pp']
TARGET_KEYS = ['screen', 'edge_id', 'pair_index']
METADATA = ['family', 'context_id', 'swap_key', 'substrate_pair']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique_targets(raw):
    selected = raw[raw['repeat'].eq(0) & raw.split.eq('pair') & raw.fraction.eq(1.0)].copy()
    if selected.empty:
        raise ValueError('No repeat-0 full-history pair-held-out target rows')
    for _, group in selected.groupby(TARGET_KEYS, sort=False):
        if not np.allclose(group.target_contrast_pp, group.target_contrast_pp.iloc[0], rtol=0.0, atol=1e-10):
            raise ValueError('Target truth differs across duplicate appearances')
        if any((group[column].nunique(dropna=False) != 1 for column in METADATA)):
            raise ValueError('Target metadata differs across duplicate appearances')
    numeric = ['target_contrast_pp', 'reference_contrast_pp', *MODEL_COLUMNS]
    if not np.isfinite(selected[numeric].to_numpy(float)).all():
        raise ValueError('Non-finite truth or prediction in strict complete targets')
    aggregation = {column: 'first' for column in METADATA}
    aggregation.update({column: 'mean' for column in numeric})
    aggregation['fold'] = lambda values: json.dumps(sorted(map(int, set(values))))
    targets = selected.groupby(TARGET_KEYS, as_index=False, sort=True).agg(aggregation)
    targets = targets.rename(columns={'fold': 'folds_json'})
    counts = selected.groupby(TARGET_KEYS, as_index=False).size().rename(columns={'size': 'prediction_appearances'})
    return targets.merge(counts, on=TARGET_KEYS, validate='one_to_one')

def direction_summaries(targets):
    results = {level: [] for level in ('summary', 'edge', 'context', 'screen', 'family')}
    for threshold in (10.0, 20.0, 30.0):
        selected = targets[targets.target_contrast_pp.abs().ge(threshold)].copy()
        for column in MODEL_COLUMNS:
            model = column.removesuffix('_prediction_pp')
            common = dict(threshold_pp=threshold, model=model)
            if selected.empty:
                for aggregation in ('pooled', 'context_screen_family_macro'):
                    results['summary'].append(dict(**common, aggregation=aggregation, accuracy=np.nan, n_targets=0, n_correct=0, n_zero_predictions=0, n_edges=0, n_contexts=0, n_screens=0, n_families=0))
                continue
            selected['correct'] = np.sign(selected[column]) == np.sign(selected.target_contrast_pp)
            selected['zero_prediction'] = selected[column].eq(0.0)
            edge = selected.groupby(['screen', 'family', 'context_id', 'edge_id'], as_index=False).agg(accuracy=('correct', 'mean'), n_targets=('correct', 'size'), n_correct=('correct', 'sum'), n_zero_predictions=('zero_prediction', 'sum'))
            context = edge.groupby(['screen', 'family', 'context_id'], as_index=False).agg(accuracy=('accuracy', 'mean'), n_targets=('n_targets', 'sum'), n_edges=('edge_id', 'size'))
            screen = context.groupby(['screen', 'family'], as_index=False).agg(accuracy=('accuracy', 'mean'), n_targets=('n_targets', 'sum'), n_edges=('n_edges', 'sum'), n_contexts=('context_id', 'size'))
            family = screen.groupby('family', as_index=False).agg(accuracy=('accuracy', 'mean'), n_targets=('n_targets', 'sum'), n_edges=('n_edges', 'sum'), n_contexts=('n_contexts', 'sum'), n_screens=('screen', 'size'))
            counts = dict(n_targets=len(selected), n_correct=int(selected.correct.sum()), n_zero_predictions=int(selected.zero_prediction.sum()), n_edges=len(edge), n_contexts=len(context), n_screens=len(screen), n_families=len(family))
            for aggregation, accuracy in (('pooled', selected.correct.mean()), ('context_screen_family_macro', family.accuracy.mean())):
                results['summary'].append(dict(**common, aggregation=aggregation, accuracy=float(accuracy), **counts))
            for level, frame in (('edge', edge), ('context', context), ('screen', screen), ('family', family)):
                results[level].extend(frame.assign(**common).to_dict('records'))
    return {level: pd.DataFrame(rows) for level, rows in results.items()}

def load_producer():
    spec = importlib.util.spec_from_file_location('m1_matched_copy_production', PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def replay(output):
    producer = load_producer()
    sys.path.insert(0, str(PRODUCER.parent.parent))
    from threadpoolctl import threadpool_limits
    tables = producer.TABLES
    registry = pd.read_csv(tables / producer.INPUT_NAMES[1])
    panels = pd.read_csv(tables / producer.INPUT_NAMES[2])
    samples = pd.read_csv(tables / producer.INPUT_NAMES[3])
    samples = samples[samples.split.eq('pair') & samples.fraction.eq(1.0) & samples['repeat'].eq(0)].copy()
    cv_path = producer.ANALYSIS / 'data/cv_fold_registry.csv'
    original_folds = pd.read_csv(cv_path)
    frozen_path = producer.QA / 'matched_copy_fold_scores.csv.gz'
    frozen = pd.read_csv(frozen_path)
    frozen = frozen[frozen['repeat'].eq(0)].copy()
    screen_counts = frozen.groupby('screen').n.sum().sort_index().sort_values(ascending=False, kind='stable')
    chosen_screen = str(screen_counts.index[0])
    edge_counts = frozen[frozen.screen.eq(chosen_screen)].groupby('edge_id').n.sum()
    chosen_edge = str(edge_counts.sort_index().sort_values(ascending=False, kind='stable').index[0])
    chosen_fold = int(samples[samples.screen.eq(chosen_screen)].fold.min())
    selection = dict(screen=chosen_screen, edge_id=chosen_edge, fold=chosen_fold, repeat=0, split='pair', fraction=1.0, rule='largest strict complete repeat-0 target count by screen; then largest edge count; lexical ties; lowest fold')
    input_paths = [PRODUCER, cv_path, frozen_path, *[tables / name for name in producer.INPUT_NAMES[1:4]]]
    input_paths += [producer.ANALYSIS / name for name in ('chemical_mode_recovery.py', 'chemical_observation_channels.py', 'chemically_aligned_measurements.py', 'shrinkage_covariance_analysis.py')]
    before = {str(path): digest(path) for path in input_paths}
    relationships, training_points = ([], [])

    def export_rows(matrix, tests, identities, target, reference, fit, key, valid, predictions, registry_lookup):
        endpoints = [int(target.condition_a_index), int(target.condition_b_index)]
        anchors = [int(target.reference_condition_a_index), int(target.reference_condition_b_index)]
        baseline = int(target.orientation) * (fit['mu'][endpoints[0]] - fit['mu'][endpoints[1]])
        basis = np.vstack([fit['mu'][anchors], fit['mu'][anchors] + [1.0, 0.0], fit['mu'][anchors] + [0.0, 1.0]])
        basis_predictions, _ = producer.contrast_prediction(fit, basis, anchors, endpoints, int(target.orientation))
        weights = basis_predictions[1:] - basis_predictions[0]
        ws = float(weights.sum())
        wd = float(int(target.reference_orientation) * (weights[0] - weights[1]) / 2.0)
        means = fit['mu'][anchors]
        reference_mean_s = float(means.mean())
        reference_mean_d = float(int(target.reference_orientation) * (means[0] - means[1]))
        values = matrix[np.ix_(tests, anchors)]
        centred_s = values.mean(axis=1) - reference_mean_s
        centred_d = reference - reference_mean_d
        reconstructed = baseline + ws * centred_s + wd * centred_d
        if not np.allclose(reconstructed[valid], predictions['semantic_full_raw'][valid], rtol=0.0, atol=1e-11):
            raise ValueError('Exact training SD plane does not reconstruct saved predictor')
        relationship = dict(**key, edge_id=target.edge_id, context_id=target.context_id, swap_key=target.swap_key, reference_edge_id=target.reference_edge_id, target_baseline_pp=100.0 * baseline, reference_mean_S_pp=100.0 * reference_mean_s, reference_mean_D_pp=100.0 * reference_mean_d, sd_weight_s=ws, sd_weight_d=wd, centered_S_zero_slice_slope=wd, centered_S_zero_slice_intercept_pp=100.0 * (baseline - wd * reference_mean_d), slice_definition='Exact SD predictor section at reference centred S = 0; full predictions also use centred S', target_condition_a=target.condition_a, target_condition_b=target.condition_b, reference_condition_a=target.reference_condition_a, reference_condition_b=target.reference_condition_b, low_name=target.low_name, high_name=target.high_name)
        relationships.append(relationship)
        frame = pd.DataFrame({'pair_index': tests[valid], 'substrate_pair': identities[tests[valid]], 'target_contrast_pp': 100.0 * producer.oriented_difference(matrix[tests], *endpoints, target.orientation)[valid], 'reference_contrast_pp': 100.0 * reference[valid], 'reference_centered_S_pp': 100.0 * centred_s[valid], 'reference_centered_D_pp': 100.0 * centred_d[valid], **{model + '_prediction_pp': 100.0 * predictions[model][valid] for model in producer.MODELS}})
        for field, value in key.items():
            frame[field] = value
        for field in ('edge_id', 'context_id', 'swap_key', 'reference_edge_id'):
            frame[field] = getattr(target, field)
        if (key['screen'], str(target.edge_id), int(key['fold'])) == (chosen_screen, chosen_edge, chosen_fold):
            sample = samples[samples.screen.eq(chosen_screen) & samples.fold.eq(chosen_fold)]
            if len(sample) != 1:
                raise ValueError('Selected illustration does not identify one frozen fit')
            indices = np.asarray(json.loads(sample.iloc[0].training_indices_json), int)
            observed = matrix[indices]
            ref = producer.oriented_difference(observed, *anchors, target.reference_orientation)
            truth = producer.oriented_difference(observed, *endpoints, target.orientation)
            finite = np.isfinite(observed[:, anchors + endpoints]).all(axis=1)
            for index in np.flatnonzero(finite):
                training_points.append(dict(**key, edge_id=target.edge_id, pair_index=int(indices[index]), substrate_pair=identities[indices[index]], reference_contrast_pp=100.0 * ref[index], target_contrast_pp=100.0 * truth[index], reference_centered_S_pp=100.0 * (observed[index, anchors].mean() - reference_mean_s)))
        return frame.to_dict('records')
    producer._chemical_target = lambda target: True
    producer._chemical_rows = export_rows
    started = time.monotonic()
    with threadpool_limits(limits=1):
        folds, primary, raw, matrices, fits, fallbacks = producer.calculate(registry, panels, samples, original_folds)
    comparison_keys = producer.FOLD_KEYS + ['edge_id']
    error, comparison_rows = producer._maximum_error(folds, frozen[comparison_keys + ['n', 'target_mask_sha256', *producer.SCORE_COLUMNS]], producer.SCORE_COLUMNS, comparison_keys)
    if (fits, len(folds), len(primary), int(folds.n.sum())) != (85, 3275, 3315, 121383):
        raise ValueError('Repeat-0 frozen strict population changed')
    if folds.edge_id.nunique() != 655 or folds.screen.nunique() != 17:
        raise ValueError('Strict target-edge or screen count changed')
    if any((digest(path) != expected for path, expected in before.items())):
        raise ValueError('An input changed during bounded replay')
    before.update(matrices)
    for path, expected in matrices.items():
        if digest(path) != expected:
            raise ValueError('A matrix changed during bounded replay')
    output.mkdir(parents=True, exist_ok=True)
    targets = unique_targets(raw)
    raw_path = output / 'direction_predictions_raw.csv.gz'
    target_path = output / 'direction_predictions_targets.csv.gz'
    raw.to_csv(raw_path, index=False)
    targets.to_csv(target_path, index=False)
    paths = [raw_path, target_path]
    for level, table in direction_summaries(targets).items():
        path = output / f'direction_accuracy_{level}.csv'
        table.to_csv(path, index=False)
        paths.append(path)
    for name, table in (('learned_training_relationships.csv', pd.DataFrame(relationships)), ('selected_training_pairs.csv', pd.DataFrame(training_points))):
        path = output / name
        table.to_csv(path, index=False)
        paths.append(path)
    verification = dict(status='passed', final_pins_modified=False, producer=str(PRODUCER), producer_sha256=digest(PRODUCER), exporter_sha256=digest(Path(__file__)), scope='655 strict edges; all eleven panels finite; pair held out; full history; repeat 0', replay_seconds=time.monotonic() - started, training_fits=fits, strict_fold_edge_rows=len(folds), raw_prediction_rows=len(raw), unique_target_rows=len(targets), duplicate_appearances_removed=len(raw) - len(targets), fold_score_comparison_rows=comparison_rows, maximum_sse_bridge_error=error, n_screens=targets.screen.nunique(), n_families=targets.family.nunique(), n_contexts=len(targets[['screen', 'context_id']].drop_duplicates()), zero_prediction_rule='A predicted zero is incorrect for any included nonzero observed contrast', macro='Equal target mean within edge; equal edge/context; context/screen; screen/family; family macro. At each threshold units with no qualifying targets have undefined accuracy and are omitted; counts are reported.', deduplication='Filter repeat0, pair, full history; average predictions within screen/edge_id/pair_index before sign scoring', illustration=selection, training_points=len(training_points), relationship='Exact unchanged training-covariance SD predictor; one-dimensional line holds centred S at zero', input_hashes=before, output_hashes={path.name: digest(path) for path in paths})
    report = output / 'direction_accuracy_verification.json'
    report.write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    replay(parser.parse_args().output)
