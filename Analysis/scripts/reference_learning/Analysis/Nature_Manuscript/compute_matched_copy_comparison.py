"""Compare fixed copy rules and learned SD on the exact Figure 2 population.

Only manuscript outputs are written. Frozen analysis producers and tables are read."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from chemical_mode_recovery import contrast_prediction, fit_clean_covariance
from chemical_observation_channels import channel_coordinates, channel_predictions
from chemical_response_structure import AMBIGUOUS
from chemically_aligned_measurements import FOLD_KEYS, aggregate_scores, direct_transport, oriented_difference, score_values, validate_training_sample
from shrinkage_covariance_analysis import matrix_path
TABLES = ANALYSIS / 'tables'
SOURCE = HERE / 'source_data'
QA = HERE / 'qa'
SCORE_COLUMNS = ('marginal_sse', 'direct_raw_sse', 'offset_raw_sse', 'semantic_full_raw_sse')
MODELS = {'marginal': 'Training marginal', 'direct_raw': 'Direct copy', 'offset_raw': 'Copy plus training offset', 'semantic_full_raw': 'Learned SD'}
INPUT_NAMES = ('chemical_observation_channels_all649_baseline_check.csv.gz', 'chemical_observation_channels_target_registry.csv', 'chemical_observation_channels_panel_registry.csv', 'chemical_mode_training_sample_registry.csv', 'chemical_observation_channels_fold_scores.csv.gz', 'chemical_observation_channels_headline.csv', 'chemically_aligned_measurements_fold_scores.csv.gz')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def complete_panel_mask(panels):
    if len(panels) != 11:
        raise ValueError('The strict complete population requires all eleven panels')
    mask = np.ones(len(panels[0]), dtype=bool)
    for panel in panels:
        values = np.asarray(panel)
        if values.shape != (len(mask), 2):
            raise ValueError('Every registered panel must have two endpoints')
        mask &= np.isfinite(values).all(axis=1)
    return mask

def summary_tables(folds):
    edge, context, screen, family, headline = aggregate_scores(folds)
    return dict(edge=edge, context=context, screen=screen[screen.weighting.eq('context_balanced')].copy(), family=family[family.weighting.eq('context_balanced')].copy(), headline=headline[headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')].copy())

def _maximum_error(current, previous, columns, keys):
    joined = current.merge(previous, on=keys, how='outer', suffixes=('', '_old'), validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Frozen comparison keys changed')
    if not joined.n.eq(joined.n_old).all():
        raise ValueError('Frozen target counts changed')
    if not joined.target_mask_sha256.eq(joined.target_mask_sha256_old).all():
        raise ValueError('Frozen target identities changed')
    maximum = max((float((joined[column] - joined[column + '_old']).abs().max()) for column in columns))
    if maximum > 1e-10:
        raise ValueError(f'Frozen score bridge differs by {maximum}')
    return (maximum, len(joined))

def _chemical_target(row):
    return row.screen == 'Suzuki|1' and row.low_name == 'Cs2CO3' and (row.high_name == 'K3PO4') and (float(row.dose) == 3.0) or (row.screen == 'Reductive-amination|1' and row.low_name == 'EtOH' and (row.high_name == 'MeOH'))

def _chemical_rows(matrix, tests, identities, target, reference, fit, key, valid, predictions, registry_lookup):
    endpoints = [int(target.condition_a_index), int(target.condition_b_index)]
    anchors = [int(target.reference_condition_a_index), int(target.reference_condition_b_index)]
    values = matrix[np.ix_(tests, anchors)]
    coordinates = channel_coordinates(values, fit['mu'][anchors], int(target.reference_orientation))
    channels, audits = channel_predictions(fit, fit['cov'], values, anchors, endpoints, int(target.orientation), int(target.reference_orientation))
    if not np.allclose(channels['full_optimal_sd_raw'][valid], predictions['semantic_full_raw'][valid], rtol=0.0, atol=1e-11):
        raise ValueError('Chemical SD prediction differs from the main calculation')
    diagnostic = next((row for row in audits if row['model'] == 'full'))
    reference_row = registry_lookup.loc[target.screen, target.reference_edge_id]
    records = []
    for local in np.flatnonzero(valid):
        pair_index = int(tests[local])
        role1, role2 = identities[pair_index].split(' >> ')
        record = {**key, 'edge_id': target.edge_id, 'reference_edge_id': target.reference_edge_id, 'context_id': target.context_id, 'reference_context_id': target.reference_context_id, 'pair_index': pair_index, 'substrate_pair': identities[pair_index], 'role1': role1, 'role2': role2, 'low_name': target.low_name, 'high_name': target.high_name, 'dose': target.dose, 'target_context_json': target.context_json, 'reference_context_json': reference_row.context_json, 'target_condition_a': target.condition_a, 'target_condition_b': target.condition_b, 'reference_condition_a': target.reference_condition_a, 'reference_condition_b': target.reference_condition_b, 'target_orientation': int(target.orientation), 'reference_orientation': int(target.reference_orientation), 'target_yield_a_pp': 100.0 * matrix[pair_index, endpoints[0]], 'target_yield_b_pp': 100.0 * matrix[pair_index, endpoints[1]], 'reference_yield_a_pp': 100.0 * values[local, 0], 'reference_yield_b_pp': 100.0 * values[local, 1], 'target_contrast_pp': 100.0 * int(target.orientation) * (matrix[pair_index, endpoints[0]] - matrix[pair_index, endpoints[1]]), 'reference_contrast_pp': 100.0 * reference[local], 'reference_centered_S_pp': 100.0 * coordinates[local, 0], 'reference_centered_D_pp': 100.0 * coordinates[local, 1], 'learned_S_prediction_pp': 100.0 * channels['full_optimal_s_raw'][local], 'learned_D_prediction_pp': 100.0 * channels['full_optimal_d_raw'][local], 'sd_weight_s': diagnostic['sd_weight_s'], 'sd_weight_d': diagnostic['sd_weight_d'], 'd_only_weight': diagnostic['optimal_d_weight']}
        record.update({model + '_prediction_pp': 100.0 * predictions[model][local] for model in MODELS})
        records.append(record)
    return records

def calculate(registry, panels, samples, original_folds):
    records, primary, chemical = ([], [], [])
    paths, fits, fallback_count = ({}, 0, 0)
    registry_lookup = registry.set_index(['screen', 'edge_id'])
    eligible = registry[registry.eligible.eq(1)]
    start = time.monotonic()
    panel_groups = {(screen, edge): group for (screen, edge), group in panels.groupby(['screen', 'edge_id'], sort=False)}
    for screen, targets in eligible.groupby('screen', sort=True):
        path = matrix_path(screen)
        paths[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            matrix = saved['M']
            conditions = saved['conds'].astype(str)
            identities = saved['sps'].astype(str)
        excluded = [index for index, condition in enumerate(conditions) if (screen.split('|')[0], condition) in AMBIGUOUS]
        folds = original_folds[original_folds.screen.eq(screen)]
        fold_groups = {key: group for key, group in folds.groupby(['split', 'repeat', 'fold'], sort=True)}
        for sample in samples[samples.screen.eq(screen)].itertuples(index=False):
            selected = np.asarray(json.loads(sample.training_indices_json), int)
            tests = np.asarray(json.loads(sample.test_indices_json), int)
            expected = fold_groups[sample.split, sample.repeat, sample.fold]
            validate_training_sample(sample, len(matrix), expected.pair_index.to_numpy(int))
            if identities[tests].tolist() != expected.subpair.tolist():
                raise ValueError('Frozen TEST identities changed')
            train = matrix[selected]
            fit = fit_clean_covariance(train, excluded)
            fits += 1
            key = {field: getattr(sample, field) for field in FOLD_KEYS}
            for target in targets.itertuples(index=False):
                endpoints = [int(target.condition_a_index), int(target.condition_b_index)]
                anchors = [int(target.reference_condition_a_index), int(target.reference_condition_b_index)]
                if len(set(endpoints + anchors)) != 4:
                    raise ValueError('Reference and target endpoints overlap')
                actual = oriented_difference(matrix[tests], *endpoints, target.orientation)
                reference = oriented_difference(matrix[tests], *anchors, target.reference_orientation)
                baseline = int(target.orientation) * (fit['mu'][endpoints[0]] - fit['mu'][endpoints[1]])
                target_train = oriented_difference(train, *endpoints, target.orientation)
                reference_train = oriented_difference(train, *anchors, target.reference_orientation)
                direct, offset, audit = direct_transport(target_train, reference_train, reference, baseline)
                learned, _ = contrast_prediction(fit, matrix[np.ix_(tests, anchors)], anchors, endpoints, int(target.orientation))
                predictions = dict(marginal=np.full(len(tests), baseline), direct_raw=direct, offset_raw=offset, semantic_full_raw=learned)
                edge_key = {**key, 'edge_id': target.edge_id, 'context_id': target.context_id, 'swap_key': target.swap_key}
                finite_target = np.isfinite(actual)
                primary.append({**edge_key, 'scope': 'primary', **score_values(actual, predictions, np.ones(len(tests), bool)), 'target_mask_sha256': hashlib.sha256(tests[finite_target].astype('<i8').tobytes()).hexdigest()})
                if not target.strict_eligible:
                    continue
                registered = panel_groups[screen, target.edge_id]
                if registered.policy.value_counts().to_dict() != {'random': 8, 'semantic': 1, 'different_any': 1, 'different_other': 1}:
                    raise ValueError('The eleven frozen panel policies changed')
                panel_values = [matrix[np.ix_(tests, json.loads(row.anchor_indices_json))] for row in registered.itertuples(index=False)]
                valid = complete_panel_mask(panel_values) & finite_target
                if np.any(valid & ~np.isfinite(reference)):
                    raise ValueError('A complete-panel target lacks its reference contrast')
                n_fallback = int(valid.sum()) if not audit['offset_valid'] else 0
                fallback_count += n_fallback
                records.append({**edge_key, 'scope': 'all_panels_complete', **score_values(actual, predictions, valid), 'target_mask_sha256': hashlib.sha256(tests[valid].astype('<i8').tobytes()).hexdigest(), 'reference_edge_id': target.reference_edge_id, 'offset_joint_train_n': audit['joint_train_n'], 'offset_valid': audit['offset_valid'], 'offset': audit['offset'], 'direct_fallback_rows': 0, 'offset_fallback_rows': n_fallback})
                if _chemical_target(target):
                    chemical.extend(_chemical_rows(matrix, tests, identities, target, reference, fit, key, valid, predictions, registry_lookup))
        print(f'MATCHED_COPY {screen}: {fits} fits, {len(records)} strict fold-edges, {time.monotonic() - start:.1f}s', flush=True)
    return (pd.DataFrame(records), pd.DataFrame(primary), pd.DataFrame(chemical), paths, fits, fallback_count)

def export_summary(tables):
    rows = []
    for level, table in tables.items():
        for item in table.to_dict('records'):
            for model, label in MODELS.items():
                row = {name: item[name] for name in ('screen', 'family', 'context_id', 'edge_id', 'swap_key', 'n', 'n_edges', 'n_scored_edges', 'n_contexts', 'n_screens', 'n_units', 'target_score_count') if name in item}
                row.update(level=level, model=model, model_label=label, domain='strict_641', split='pair', fraction=1.0, scope='all_panels_complete', weighting='context_balanced', mse_pp2=10000.0 * item[model + '_mse'], gain_vs_marginal_pp2=10000.0 * (item['marginal_mse'] - item[model + '_mse']))
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    SOURCE.mkdir(exist_ok=True)
    QA.mkdir(exist_ok=True)
    input_paths = [TABLES / name for name in INPUT_NAMES]
    input_paths += [ANALYSIS / 'data' / 'cv_fold_registry.csv']
    input_paths += [ANALYSIS / name for name in ('chemical_mode_recovery.py', 'chemical_observation_channels.py', 'chemically_aligned_measurements.py', 'shrinkage_covariance_analysis.py')]
    input_hashes = {str(path): digest(path) for path in input_paths}
    baseline = pd.read_csv(input_paths[0])
    registry, panels, samples = (pd.read_csv(TABLES / name) for name in INPUT_NAMES[1:4])
    samples = samples[samples.split.eq('pair') & samples.fraction.eq(1.0)]
    original_folds = pd.read_csv(ANALYSIS / 'data' / 'cv_fold_registry.csv')
    with threadpool_limits(limits=1):
        folds, primary, chemical, matrix_hashes, fits, fallbacks = calculate(registry, panels, samples, original_folds)
    keys = FOLD_KEYS + ['edge_id']
    old63 = pd.read_csv(TABLES / 'chemically_aligned_measurements_fold_scores.csv.gz')
    old63 = old63[old63.split.eq('pair') & old63.fraction.eq(1.0) & old63.scope.eq('primary')]
    baseline63_error, baseline63_rows = _maximum_error(primary, old63[keys + ['n', 'target_mask_sha256', *SCORE_COLUMNS]], SCORE_COLUMNS, keys)
    baseline = baseline[baseline.split.eq('pair') & baseline.fraction.eq(1.0)]
    bridge66_error, bridge66_rows = _maximum_error(primary, baseline[keys + ['n', 'target_mask_sha256', 'marginal_sse', 'semantic_full_raw_sse']], ['marginal_sse', 'semantic_full_raw_sse'], keys)
    old66 = pd.read_csv(TABLES / 'chemical_observation_channels_fold_scores.csv.gz', usecols=keys + ['domain', 'scope', 'n', 'target_mask_sha256', 'marginal_sse', 'semantic_full_optimal_sd_raw_sse'])
    old66 = old66[old66.domain.eq('strict_641') & old66.scope.eq('all_panels_complete') & old66.split.eq('pair') & old66.fraction.eq(1.0)].rename(columns={'semantic_full_optimal_sd_raw_sse': 'semantic_full_raw_sse'})
    complete_error, complete_rows = _maximum_error(folds, old66[keys + ['n', 'target_mask_sha256', 'marginal_sse', 'semantic_full_raw_sse']], ['marginal_sse', 'semantic_full_raw_sse'], keys)
    tables = summary_tables(folds)
    headline = tables['headline'].iloc[0]
    old_head = pd.read_csv(TABLES / 'chemical_observation_channels_headline.csv')
    old_head = old_head[old_head.domain.eq('strict_641') & old_head.scope.eq('all_panels_complete') & old_head.split.eq('pair') & old_head.fraction.eq(1.0) & old_head.weighting.eq('context_balanced') & old_head.macro.eq('family_macro')].iloc[0]
    headline_error_pp2 = 10000.0 * abs(headline.semantic_full_raw_mse - old_head.semantic_full_optimal_sd_raw_mse)
    if headline_error_pp2 > 1e-08:
        raise ValueError('Learned SD no longer matches Figure 2')
    if (fits, len(folds), len(primary), int(folds.n.sum()), len(tables['edge']), len(tables['context']), len(tables['screen']), len(tables['family'])) != (170, 6550, 6630, 242766, 655, 213, 17, 14):
        raise ValueError('Frozen population sizes changed')
    chemical_counts = chemical.groupby('screen').edge_id.nunique().to_dict()
    if chemical_counts != {'Reductive-amination|1': 6, 'Suzuki|1': 9}:
        raise ValueError('The complete chemical examples changed')
    input_hashes.update(matrix_hashes)
    unchanged = all((digest(path) == value for path, value in input_hashes.items()))
    if not unchanged:
        raise ValueError('A frozen input changed during the comparison')
    output = export_summary(tables)
    summary_path = SOURCE / 'Matched_copy_comparison.csv'
    fold_path = QA / 'matched_copy_fold_scores.csv.gz'
    chemical_path = SOURCE / 'Measured_chemical_contrasts.csv'
    output.to_csv(summary_path, index=False)
    folds.to_csv(fold_path, index=False, compression='gzip')
    chemical.to_csv(chemical_path, index=False)
    results = {label: 10000.0 * float(headline[model + '_mse']) for model, label in MODELS.items()}
    verification = {'status': 'passed', 'comparison': 'strict641/full/pair/raw/all11-complete', 'aggregation': 'SSE/count across folds per edge, equal edge/context, context/screen, screen/merged-family, family macro', 'builder_sha256': digest(Path(__file__)), 'input_hashes': input_hashes, 'inputs_unchanged': unchanged, 'baseline_inspected': 'all649_baseline_check has fold-edge totals only; no individual target rows or direct/offset errors', 'training_fits': fits, 'strict_fold_edge_rows': len(folds), 'original649_fold_edge_rows': len(primary), 'target_score_count_including_repeat_use': int(folds.n.sum()), 'edges': len(tables['edge']), 'contexts': len(tables['context']), 'screens': len(tables['screen']), 'merged_families': 14, 'max_original63_primary_sse_error': baseline63_error, 'original63_compared_rows': baseline63_rows, 'max_original66_all649_sse_error': bridge66_error, 'original66_all649_compared_rows': bridge66_rows, 'max_original66_complete_sse_error': complete_error, 'original66_complete_compared_rows': complete_rows, 'learned_sd_figure2_mse_error_pp2': headline_error_pp2, 'direct_fallback_scored_rows': 0, 'offset_fallback_scored_rows': fallbacks, 'headline_mse_pp2': results, 'headline_gain_vs_marginal_pp2': {label: results['Training marginal'] - value for label, value in results.items()}, 'chemical_edge_counts': chemical_counts, 'chemical_prediction_rows': len(chemical), 'chemical_rows_definition': 'All nine Suzuki Cs2CO3/K3PO4 edges and six reductive-amination EtOH/MeOH edges, exact main population, with repeated CV appearances retained', 'output_hashes': {str(path.relative_to(HERE)): digest(path) for path in (summary_path, fold_path, chemical_path)}}
    (QA / 'matched_copy_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'headline_mse_pp2': results, 'figure2_error_pp2': headline_error_pp2, 'chemical_rows': len(chemical), 'inputs_unchanged': unchanged}, indent=2), flush=True)
if __name__ == '__main__':
    main()
