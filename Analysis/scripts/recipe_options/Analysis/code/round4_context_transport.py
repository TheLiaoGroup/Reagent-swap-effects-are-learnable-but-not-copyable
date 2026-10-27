"""Rebuild strict four-condition targets and test cross-background calibration."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round4' / 'context_transport'
INPUT = ROOT / 'round4' / 'rectangle_target_effects.csv.gz'
OLD = ROOT / 'round3' / 'context_rectangles'
BLOCK = ['archive', 'plate', 'acquisition', 'target_key']
YIELDS = ['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']
SCHEMA = ['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', *YIELDS, 'delta_bg0', 'delta_bg1', 'DD', 'n_complete_acquisition_blocks']
SEED = 20261018
MODELS = ['training_mean', 'direct', 'offset', 'affine', 'ridge_anchor_one']
RIDGE_ALPHA = 10.0

def write_csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8')

def build_input():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(OLD / 'rectangle_registry.csv', keep_default_na=False, low_memory=False)
    representatives = registry.sort_values(['rectangle_id', 'representation_id']).drop_duplicates('rectangle_id').copy()
    assert len(representatives) == 830 and representatives.rectangle_id.is_unique
    previous = pd.read_csv(OLD / 'unique_four_recipe_rectangles.csv', keep_default_na=False, low_memory=False).set_index('rectangle_id')
    assert set(representatives.rectangle_id) == set(previous.index)
    representative_ids = representatives.set_index('rectangle_id').representation_id
    assert (representative_ids == previous.loc[representative_ids.index, 'representative_focal_id']).all()
    needed = set(representatives.pair_id_bg0) | set(representatives.pair_id_bg1)
    columns = ['pair_id', 'dataset_id', *BLOCK, 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'y0', 'y1', 'delta', 'both_source_check_true']
    source = pd.read_csv(ROOT / 'results' / 'delta_within_acquisition.csv.gz', usecols=columns, keep_default_na=False, low_memory=False)
    source = source.loc[source.pair_id.isin(needed)].copy()
    assert not source.duplicated(['pair_id', *BLOCK]).any()
    identity = source[['dataset_id', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].drop_duplicates()
    assert not identity.duplicated(['dataset_id', 'target_key']).any()
    assert np.allclose(source.delta, source.y1 - source.y0, rtol=0, atol=1e-14)
    groups = {pair: part[[*BLOCK, 'y0', 'y1', 'delta', 'both_source_check_true']].copy() for pair, part in source.groupby('pair_id', sort=False)}
    del source
    targets, coverage, errors = ([], [], {})
    for index, row in enumerate(representatives.to_dict('records')):
        blocks = groups[row['pair_id_bg0']].merge(groups[row['pair_id_bg1']], on=BLOCK, suffixes=('_bg0', '_bg1'), validate='one_to_one')
        value_columns = [*YIELDS, 'delta_bg0', 'delta_bg1']
        part = blocks.groupby('target_key', as_index=False)[value_columns].mean()
        counts = blocks.groupby('target_key').size()
        part['n_complete_acquisition_blocks'] = part.target_key.map(counts).astype(int)
        part['DD'] = part.delta_bg1 - part.delta_bg0
        for background in (0, 1):
            assert np.allclose(part[f'delta_bg{background}'], part[f'y1_bg{background}'] - part[f'y0_bg{background}'], rtol=0, atol=1e-14)
        assert np.allclose(part.DD, part.y1_bg1 - part.y0_bg1 - part.y1_bg0 + part.y0_bg0, rtol=0, atol=1e-14)
        for column in ['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class']:
            part[column] = row[column]
        part = part.merge(identity, on=['dataset_id', 'target_key'], how='left', validate='many_to_one')
        assert part.sub_1_smiles.notna().all()
        old = previous.loc[row['rectangle_id']]
        assert len(part) == int(old.n_common_targets)
        assert len(blocks) == int(old.n_four_condition_acquisition_blocks)
        if len(part):
            dd = part.DD.to_numpy()
            checks = {'mean_DD': dd.mean(), 'SD_DD': dd.std(ddof=1) if len(dd) > 1 else np.nan, 'centered_profile_RMS_change': np.sqrt(np.mean((dd - dd.mean()) ** 2)), 'RMS_DD': np.sqrt(np.mean(dd ** 2))}
            for column, value in checks.items():
                saved = pd.to_numeric(old[column], errors='coerce')
                assert np.isclose(value, saved, rtol=0, atol=1e-12, equal_nan=True), (row['rectangle_id'], column, value, saved)
                if np.isfinite(value):
                    errors[column] = max(errors.get(column, 0.0), float(abs(value - saved)))
            targets.append(part[SCHEMA])
        coverage.append({**row, 'n_common_targets': len(part), 'n_complete_acquisition_blocks': len(blocks), 'main_n_ge20': len(part) >= 20, 'n_all_four_ge5pct': int(part[YIELDS].ge(0.05).all(axis=1).sum()), 'fraction_four_source_check_true': float((blocks.both_source_check_true_bg0 & blocks.both_source_check_true_bg1).mean()) if len(blocks) else np.nan})
        if (index + 1) % 200 == 0:
            print(f'Input rebuilt: {index + 1}/830 rectangles', flush=True)
    result = pd.concat(targets, ignore_index=True)
    assert not result.duplicated(['rectangle_id', 'target_key']).any()
    result = result.sort_values(['rectangle_id', 'target_key']).reset_index(drop=True)
    result.to_csv(INPUT, index=False, compression='gzip', encoding='utf-8')
    coverage = pd.DataFrame(coverage)
    assert coverage.main_n_ge20.sum() == 825
    write_csv(coverage, 'rectangle_input_coverage.csv')
    manifest = {'status': 'input_ready', 'registered_unique_rectangles': len(coverage), 'rectangles_with_strict_target_support': result.rectangle_id.nunique(), 'main_n_ge20_rectangles': int(coverage.main_n_ge20.sum()), 'target_mean_rows': len(result), 'complete_acquisition_blocks': int(result.n_complete_acquisition_blocks.sum()), 'schema': SCHEMA, 'representative_axis': 'lexicographically first representation_id per rectangle', 'pairing_keys': BLOCK, 'old_unique_summary_max_abs_differences': errors, 'target_input': str(INPUT)}
    (OUT / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return (result, coverage)

def fit_calibration(a_train, b_train, a_test):
    mean_a, mean_b = (float(np.mean(a_train)), float(np.mean(b_train)))
    centered_a, centered_b = (a_train - mean_a, b_train - mean_b)
    variance = float(np.mean(centered_a ** 2))
    slope = float(np.mean(centered_a * centered_b) / variance) if variance > 1e-12 else 0.0
    intercept = mean_b - slope * mean_a
    residual = b_train - a_train
    residual_centered = residual - residual.mean()
    ridge_beta = float(np.sum(centered_a * residual_centered) / (np.sum(centered_a ** 2) + RIDGE_ALPHA))
    ridge_slope = 1 + ridge_beta
    ridge_intercept = mean_b - ridge_slope * mean_a
    predictions = {'training_mean': np.full(len(a_test), mean_b), 'direct': a_test.copy(), 'offset': a_test + mean_b - mean_a, 'affine': intercept + slope * a_test, 'ridge_anchor_one': ridge_intercept + ridge_slope * a_test}
    correlation = float(np.corrcoef(a_train, b_train)[0, 1]) if variance > 1e-12 and np.std(b_train) > 1e-12 else np.nan
    return (predictions, {'mean_a_train': mean_a, 'mean_b_train': mean_b, 'affine_intercept': intercept, 'affine_slope': slope, 'training_correlation': correlation, 'affine_variance_fallback': variance <= 1e-12, 'a_train_variance': variance, 'a_train_min': float(a_train.min()), 'a_train_max': float(a_train.max()), 'ridge_anchor_one_slope': ridge_slope, 'ridge_anchor_one_intercept': ridge_intercept})

def validate_calibration():
    a = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    b = 0.03 + 1.7 * a
    test = np.array([-0.15, 0.05])
    predictions, parameters = fit_calibration(a, b, test)
    assert np.allclose(predictions['affine'], 0.03 + 1.7 * test, atol=1e-14)
    assert np.isclose(parameters['affine_slope'], 1.7)
    predictions, parameters = fit_calibration(np.zeros(5), b, test)
    assert parameters['affine_variance_fallback'] and np.allclose(predictions['affine'], b.mean())
    predictions, _ = fit_calibration(a, a + 0.07, test)
    assert np.allclose(predictions['offset'], test + 0.07, atol=1e-14)
    predictions, parameters = fit_calibration(a, b, test)
    expected_beta = 0.7 * np.sum(a ** 2) / (np.sum(a ** 2) + 10)
    assert np.isclose(parameters['ridge_anchor_one_slope'], 1 + expected_beta)
    assert np.allclose(predictions['ridge_anchor_one'], 0.03 + (1 + expected_beta) * test)

def evaluate_predictions(a, b, predictions, molecules=None):
    weights = np.ones(len(a))
    if molecules is not None:
        _, inverse, degree = np.unique(molecules, return_inverse=True, return_counts=True)
        weights = 1.0 / degree[inverse]
    result = {'n_targets': len(a), 'n_focal_molecules': len(set(molecules)) if molecules is not None else np.nan}
    for model in MODELS:
        residual = predictions[model] - b
        mse = float(np.average(residual ** 2, weights=weights))
        result[f'mse_{model}'] = mse
        result[f'rmse_{model}_pp'] = float(100 * np.sqrt(mse))
        result[f'mae_{model}_pp'] = float(100 * np.average(np.abs(residual), weights=weights))
    base = result['mse_training_mean']
    for model in MODELS:
        result[f'q2_{model}'] = 1 - result[f'mse_{model}'] / base if base > 1e-14 else np.nan
    mean_a, mean_b = (np.average(a, weights=weights), np.average(b, weights=weights))
    variance_a = np.average((a - mean_a) ** 2, weights=weights)
    variance_b = np.average((b - mean_b) ** 2, weights=weights)
    covariance = np.average((a - mean_a) * (b - mean_b), weights=weights)
    result['response_correlation'] = float(covariance / np.sqrt(variance_a * variance_b)) if variance_a > 1e-14 and variance_b > 1e-14 else np.nan
    result['mean_a'] = float(mean_a)
    result['mean_b'] = float(mean_b)
    result['fraction_observed_same_sign'] = float(np.average(a * b > 0, weights=weights))
    result['fraction_observed_opposite_sign'] = float(np.average(a * b < 0, weights=weights))
    both_substantial = (np.abs(a) >= 0.05) & (np.abs(b) >= 0.05)
    result['fraction_observed_same_sign_both_ge5pp'] = float(np.average((a * b > 0) & both_substantial, weights=weights))
    result['fraction_observed_opposite_sign_both_ge5pp'] = float(np.average((a * b < 0) & both_substantial, weights=weights))
    result['n_both_responses_abs_ge5pp'] = int(both_substantial.sum())
    result['same_sign_fraction_among_both_ge5pp'] = float(np.average((a * b > 0)[both_substantial], weights=weights[both_substantial])) if both_substantial.any() else np.nan
    substantial_b = np.abs(b) >= 0.05
    result['n_destination_response_abs_ge5pp'] = int(substantial_b.sum())
    for model in MODELS:
        correct = np.sign(predictions[model]) == np.sign(b)
        result[f'direction_accuracy_destination_ge5pp_{model}'] = float(np.average(correct[substantial_b], weights=weights[substantial_b])) if substantial_b.any() else np.nan
    return result

def case_oof(part, scheme, rectangle):
    molecules = part.focal_molecule.to_numpy() if scheme != 'target' else None
    if scheme == 'target':
        seed = int(hashlib.sha256(f'{SEED}|{rectangle}|target'.encode()).hexdigest()[:8], 16)
        splits = list(KFold(5, shuffle=True, random_state=seed).split(part))
    else:
        splits = list(GroupKFold(5).split(part, groups=molecules))
    fold_number = np.full(len(part), -1, dtype=int)
    output, parameters = ([], [])
    for direction, origin, destination in [('bg0_to_bg1', 'delta_bg0', 'delta_bg1'), ('bg1_to_bg0', 'delta_bg1', 'delta_bg0')]:
        a, b = (part[origin].to_numpy(), part[destination].to_numpy())
        predicted = {model: np.empty(len(part)) for model in MODELS}
        for fold, (train, test) in enumerate(splits):
            assert not set(part.iloc[train].target_key) & set(part.iloc[test].target_key)
            if molecules is not None:
                assert not set(molecules[train]) & set(molecules[test])
            fold_predictions, fitted = fit_calibration(a[train], b[train], a[test])
            for model, values in fold_predictions.items():
                predicted[model][test] = values
            if direction == 'bg0_to_bg1':
                assert (fold_number[test] == -1).all()
                fold_number[test] = fold
            else:
                assert (fold_number[test] == fold).all()
            parameters.append({'direction': direction, 'fold': fold, 'n_train_targets': len(train), 'n_test_targets': len(test), 'n_train_focal_molecules': len(set(molecules[train])) if molecules is not None else np.nan, 'n_test_focal_molecules': len(set(molecules[test])) if molecules is not None else np.nan, 'a_test_min': float(a[test].min()), 'a_test_max': float(a[test].max()), 'n_test_a_outside_training_range': int(((a[test] < a[train].min()) | (a[test] > a[train].max())).sum()), 'max_abs_affine_prediction': float(np.max(np.abs(fold_predictions['affine']))), **fitted})
        assert (fold_number >= 0).all()
        assert all((np.isfinite(values).all() for values in predicted.values()))
        frame = pd.DataFrame({'target_key': part.target_key.to_numpy(), 'direction': direction, 'fold': fold_number, 'a': a, 'b': b, 'focal_molecule': molecules if molecules is not None else ''})
        for model, values in predicted.items():
            frame[f'pred_{model}'] = values
        output.append(frame)
    return (pd.concat(output, ignore_index=True), parameters)

def symmetric_results(directional):
    keys = ['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class', 'panel', 'scheme', 'test_weighting']
    output = []
    for key, part in directional.groupby(keys, sort=False):
        assert set(part.direction) == {'bg0_to_bg1', 'bg1_to_bg0'} and len(part) == 2
        row = dict(zip(keys, key))
        row.update(n_targets=int(part.n_targets.iloc[0]), n_focal_molecules=part.n_focal_molecules.iloc[0])
        for column in part:
            if column.startswith('mse_') or column.startswith('mae_') or column.startswith('fraction_') or column.startswith('direction_accuracy_') or (column in ['response_correlation', 'same_sign_fraction_among_both_ge5pp']):
                row[column] = part[column].mean()
        for model in MODELS:
            row[f'rmse_{model}_pp'] = 100 * np.sqrt(row[f'mse_{model}'])
            row[f'q2_{model}'] = 1 - row[f'mse_{model}'] / row['mse_training_mean'] if row['mse_training_mean'] > 1e-14 else np.nan
        row['offset_gain_over_direct_mse'] = row['mse_direct'] - row['mse_offset']
        row['affine_gain_over_offset_mse'] = row['mse_offset'] - row['mse_affine']
        output.append(row)
    result = pd.DataFrame(output)
    assert not result.duplicated(keys).any()
    return result

def aggregate_rectangles(symmetric):
    output = []
    groupings = [['panel', 'scheme', 'test_weighting'], ['panel', 'scheme', 'test_weighting', 'rxn_class'], ['panel', 'scheme', 'test_weighting', 'dataset_id']]
    for keys in groupings:
        for key, part in symmetric.groupby(keys, sort=False):
            for weighting in ['rectangle_equal', 'reaction_class_equal']:
                numeric = [column for column in part if column.startswith('mse_') or column.startswith('fraction_')]
                units = part[numeric] if weighting == 'rectangle_equal' else part.groupby('rxn_class')[numeric].mean()
                averages = units.mean()
                row = {**dict(zip(keys, key)), 'aggregation': '|'.join(keys), 'aggregation_weighting': weighting, 'n_unique_rectangles': len(part), 'n_datasets': part.dataset_id.nunique(), 'n_reaction_classes': part.rxn_class.nunique(), 'n_aggregation_units': len(units), **averages.to_dict()}
                for model in MODELS:
                    base = row['mse_training_mean']
                    row[f'q2_from_aggregate_mse_{model}'] = 1 - row[f'mse_{model}'] / base if base > 1e-14 else np.nan
                    row[f'rmse_from_aggregate_mse_{model}_pp'] = 100 * np.sqrt(row[f'mse_{model}'])
                    q2 = part[['rxn_class', f'q2_{model}']].copy()
                    row[f'mean_rectangle_q2_{model}'] = q2[f'q2_{model}'].mean() if weighting == 'rectangle_equal' else q2.groupby('rxn_class')[f'q2_{model}'].mean().mean()
                    positive = part.assign(positive=part[f'q2_{model}'].gt(0))
                    row[f'fraction_rectangles_q2_positive_{model}'] = positive.positive.mean() if weighting == 'rectangle_equal' else positive.groupby('rxn_class').positive.mean().mean()
                    row[f'unweighted_median_rectangle_q2_{model}'] = part[f'q2_{model}'].median()
                    row[f'unweighted_q25_rectangle_q2_{model}'] = part[f'q2_{model}'].quantile(0.25)
                    row[f'unweighted_q75_rectangle_q2_{model}'] = part[f'q2_{model}'].quantile(0.75)
                for model in ['offset', 'affine', 'ridge_anchor_one']:
                    for baseline in ['direct', 'offset']:
                        better = part.assign(better=part[f'mse_{model}'].lt(part[f'mse_{baseline}']))
                        row[f'fraction_rectangles_{model}_better_than_{baseline}'] = better.better.mean() if weighting == 'rectangle_equal' else better.groupby('rxn_class').better.mean().mean()
                output.append(row)
    return pd.DataFrame(output)

def choose_examples(symmetric, directional, parameters, targets, coverage):
    eligible = symmetric.loc[symmetric.panel.eq('full') & symmetric.scheme.eq('target') & symmetric.test_weighting.eq('target_equal')].copy()
    supported = eligible.loc[eligible.n_targets.ge(50)]
    selections, selected = ([], set())
    definitions = [('direct_positive', supported.loc[supported.q2_direct.ge(0.1)], 'q2_direct', False), ('calibration_rescue', supported.loc[supported.q2_direct.lt(0) & supported.q2_affine.ge(0.1)], 'q2_affine', False), ('negative_after_calibration', eligible.loc[eligible.q2_affine.lt(0)], 'q2_affine', True)]
    for category, candidates, order, ascending in definitions:
        classes = set()
        for row in candidates.sort_values(order, ascending=ascending).to_dict('records'):
            if row['rectangle_id'] in selected or row['rxn_class'] in classes:
                continue
            row['example_category'] = category
            selections.append(row)
            selected.add(row['rectangle_id'])
            classes.add(row['rxn_class'])
            if len(classes) >= 4:
                break
    finkelstein = eligible.loc[eligible.dataset_id.eq('Finkelstein')].sort_values('q2_affine', ascending=False)
    if len(finkelstein):
        row = finkelstein.iloc[0].to_dict()
        row['example_category'] = 'single_substrate_boundary'
        selections.append(row)
    selected_frame = pd.DataFrame(selections)
    metadata = coverage[['rectangle_id', 'focal_slot', 'component0_name', 'component1_name', 'component0_dose', 'component1_dose', 'context_slot', 'context0_name', 'context1_name', 'context0_token', 'context1_token', 'fixed_background_json', 'condition_c0_bg0', 'condition_c1_bg0', 'condition_c0_bg1', 'condition_c1_bg1']]
    selected_frame = selected_frame.merge(metadata, on='rectangle_id', validate='one_to_one')
    means = targets.groupby('rectangle_id')[YIELDS + ['delta_bg0', 'delta_bg1', 'DD']].mean().add_prefix('mean_')
    selected_frame = selected_frame.merge(means, on='rectangle_id', validate='one_to_one')
    fitted = parameters.loc[parameters.panel.eq('full') & parameters.scheme.eq('target')].groupby(['rectangle_id', 'direction'])[['affine_slope', 'affine_intercept', 'ridge_anchor_one_slope', 'ridge_anchor_one_intercept']].mean().unstack('direction')
    fitted.columns = ['mean_training_' + '_'.join(column) for column in fitted.columns]
    return selected_frame.merge(fitted, on='rectangle_id', validate='one_to_one')

def report(manifest, aggregate, examples, coverage):
    global_rows = aggregate.loc[aggregate.aggregation.eq('panel|scheme|test_weighting') & aggregate.panel.eq('full') & aggregate.test_weighting.eq('target_equal')]
    lines = ['# Round 4: how much of a response transfers across backgrounds after calibration', '', 'The same component swap can produce a substrate response of different size in another condition background; here direct reuse, correction of the overall shift and calibration of response magnitude are separated. A shift or linear map is estimated on training targets and used to predict held-out targets, quantifying how much response knowledge from the measured background is retained.', '', f"Of 830 distinct four-recipe rectangles, 827 have four-cell support with the same archive/plate/acquisition/target, and 825 reach the main range of 20 targets. The unified input contains {manifest['input_target_mean_rows']:,} four-cell target means, encoded on the first axis by representation_id order. DD is invariant to swapping the two axes, whereas response-transfer Q² depends on the predicted axis; the scientific summary further averages the MSE of both directions and both registered axes within each rectangle, so each of the 825 rectangles is counted once.", '', '## First-axis results on full targets', '', 'All results for the first registered axis are kept below, as an axis-choice sensitivity and for interpreting specific conditions. Q² uses the OOF error of the per-fold training mean of the target background as zero. MSE is first averaged over both directions within each rectangle, then aggregated by the stated weighting, and Q² is computed from the MSE ratio. RMSE is in yield percentage points.', '', '| Hold-out | Weighting | Unique rectangles/reaction classes | Constant RMSE | Direct-reuse Q² | Shift-calibrated Q² | Magnitude-calibrated Q² | Magnitude-calibrated RMSE |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in global_rows.itertuples():
        lines.append(f'| {row.scheme} | {row.aggregation_weighting} | {row.n_unique_rectangles}/{row.n_reaction_classes} | {row.rmse_from_aggregate_mse_training_mean_pp:.2f} | {row.q2_from_aggregate_mse_direct:.3f} | {row.q2_from_aggregate_mse_offset:.3f} | {row.q2_from_aggregate_mse_affine:.3f} | {row.rmse_from_aggregate_mse_affine_pp:.2f} |')
    main = global_rows.loc[global_rows.scheme.eq('target') & global_rows.aggregation_weighting.eq('reaction_class_equal')].iloc[0]
    change = f'decreased MSE by {100 * main.q2_from_aggregate_mse_affine:.1f}%' if main.q2_from_aggregate_mse_affine >= 0 else f'increased MSE by {-100 * main.q2_from_aggregate_mse_affine:.1f}%'
    lines.extend(['', f'With reaction classes weighted equally, Q² is {main.q2_from_aggregate_mse_direct:.3f} for direct reuse of the response, {main.q2_from_aggregate_mse_offset:.3f} after correcting only the mean, and {main.q2_from_aggregate_mse_affine:.3f} after also calibrating magnitude. Relative to the training mean of the target background, unconstrained OLS calibration {change}.', f'Median Q² over the 825 rectangles: direct reuse {main.unweighted_median_rectangle_q2_direct:.3f}, shift {main.unweighted_median_rectangle_q2_offset:.3f}, OLS magnitude calibration {main.unweighted_median_rectangle_q2_affine:.3f}; with reaction classes weighted equally, the fractions of rectangles with positive Q² are {100 * main.fraction_rectangles_q2_positive_direct:.1f}%, {100 * main.fraction_rectangles_q2_positive_offset:.1f}% and {100 * main.fraction_rectangles_q2_positive_affine:.1f}%. Medians and aggregate MSE are both kept to show the different influence of typical rectangles and extreme extrapolation on the total.', f'A secondary stability comparison shrinks the slope toward 1 with fixed alpha=10, without standardizing Δ and without penalizing the intercept. Its reaction-class-equal Q² is {main.q2_from_aggregate_mse_ridge_anchor_one:.3f}, and its median rectangle Q² is {main.unweighted_median_rectangle_q2_ridge_anchor_one:.3f}; this comparison does not replace the four main models.', f'With reaction classes weighted equally, OLS magnitude calibration improves on direct reuse in {100 * main.fraction_rectangles_affine_better_than_direct:.1f}% of rectangles, and ridge shrinkage toward 1 in {100 * main.fraction_rectangles_ridge_anchor_one_better_than_direct:.1f}%. This fraction answers a different question from the fraction with positive Q² relative to the constant.', f'On full targets, the fraction where responses in both backgrounds reach 5pp in absolute value with opposite sign is {100 * main.fraction_observed_opposite_sign_both_ge5pp:.1f}% (reaction classes weighted equally, targets weighted equally within rectangle); the fraction with the same sign and both reaching 5pp is {100 * main.fraction_observed_same_sign_both_ge5pp:.1f}%.', '', '## Specific conditions across reactions', '', 'First-axis examples follow. Positive transfer and calibration rescue require at least 50 targets; negative-transfer examples are chosen from all main rectangles to keep the strong extrapolations that drive the total error. One Finkelstein single-substrate example is also kept. Full results for all 825 rectangles are kept.', '', '| Type/reaction | Focal component swap | Background swap | target | Four-cell mean yield/% | Direct/shift/magnitude Q² | Response correlation |', '|---|---|---|---:|---|---|---:|'])
    for row in examples.itertuples():
        values = ', '.join((f"{100 * getattr(row, 'mean_' + column):.1f}" for column in YIELDS))
        lines.append(f'| {row.example_category} / {row.dataset_id} | {row.focal_slot}: {row.component0_name} [{row.component0_dose}] → {row.component1_name} [{row.component1_dose}] | {row.context_slot}: {row.context0_name} → {row.context1_name} | {row.n_targets} | {values} | {row.q2_direct:.3f}/{row.q2_offset:.3f}/{row.q2_affine:.3f} | {row.response_correlation:.3f} |')
    lines.extend(['', 'Four-cell order is component 0/background 0, component 1/background 0, component 0/background 1, component 1/background 1; doses in background tokens, full fixed conditions and original condition IDs are in chemical_examples.csv. The correlation coefficient is between the two measured response vectors and is not OOF predictive performance. Training affine slopes and intercepts for each direction are in fold_calibration.csv and chemical_examples.csv.', '', '## Floor sensitivity', '', 'Results with all four cell target-mean yields ≥5% form a separate sensitivity panel. Target hold-out is run only when ≥20 targets remain after filtering; molecule hold-out also requires at least 20 canonical molecules on that side. The filter uses observed four-cell yields and changes target composition; rectangles that do not meet support are kept in coverage and are not removed from the main results for all 825 rectangles.', '', '| Hold-out | Eligible rectangles after ≥5% | Reaction classes | Direct Q² | Shift Q² | Magnitude Q² |', '|---|---:|---:|---:|---:|---:|'])
    floor = aggregate.loc[aggregate.aggregation.eq('panel|scheme|test_weighting') & aggregate.panel.eq('all_four_ge5pct') & aggregate.test_weighting.eq('target_equal') & aggregate.aggregation_weighting.eq('reaction_class_equal')]
    for row in floor.itertuples():
        lines.append(f'| {row.scheme} | {row.n_unique_rectangles} | {row.n_reaction_classes} | {row.q2_from_aggregate_mse_direct:.3f} | {row.q2_from_aggregate_mse_offset:.3f} | {row.q2_from_aggregate_mse_affine:.3f} |')
    diagnostic_path = OUT / 'calibration_instability.csv'
    if diagnostic_path.exists():
        instability = pd.read_csv(diagnostic_path, low_memory=False)
        worst = instability.nlargest(1, 'mse_affine').iloc[0]
        lines.extend(['', '## Where extreme magnitude calibration occurs', '', f'Rectangle {worst.rectangle_id} ({worst.dataset_id}, {int(worst.n_targets)} targets) accounts for {100 * worst.share_rectangle_equal_mse_affine:.1f}% of the rectangle-equal OLS affine MSE and {100 * worst.share_reaction_class_equal_mse_affine:.1f}% of the reaction-class-equal MSE. In one training fold the source response variance is only {worst.min_a_train_variance:.3g}, the absolute training slope reaches {worst.max_abs_affine_slope:.1f}, and the absolute test prediction reaches {worst.max_abs_affine_prediction:.2f} (Δ on the 0–1 scale). This is an extrapolation failure in real OOF calibration; full results are kept, and the distribution over typical rectangles and the shrink-toward-1 model are shown alongside.', 'calibration_extreme_targets.csv keeps the specific targets causing the largest errors, both substrates and product SMILES, measured four-cell yields, training response range/variance and predictions. It locates the folds that extrapolate from a nearly flat training response to a larger test response.'])
    lines.extend(['', '## Methods and interpretation', '', 'For each rectangle, a and b are the Δ of the same component swap in the two backgrounds. bg0→bg1 and bg1→bg0 use the same five-fold target assignment; complete targets are held out as groups. Molecule hold-out groups by canonical identity from the original structure table, and all targets on that side with valid support are shared; Finkelstein keeps its real single-substrate side, and the empty second substrate is not treated as a molecule.', '', 'The four main models are: training mean(b); direct a; a+training[mean(b)−mean(a)]; within-training OLS intercept+slope×a. When the training variance of a is ≤1e−12, affine falls back to training mean(b). The secondary ridge model sets r=b−a, β=Σ(a−ā)(r−r̄)/[Σ(a−ā)²+10], and predicts a+r̄+β(a−ā). Predictions are not clipped; all models are scored on the same OOF targets. Training fits weight targets equally; molecule hold-out also provides focal-molecule-equal scoring, using the mean loss over the test targets of that molecule as the unit.', '', 'The response a of a test target in the source background is measured experimental information. Even when a whole canonical molecule is held out of the target-background training set, this analysis still uses its measured response in the other background; it addresses cross-background transfer of response knowledge, which differs from independent prediction from the structure of unmeasured molecules.', '', 'The two directions and two focal axes are not counted as independent rectangles; reaction-class equal weighting first averages rectangle errors within each class and then averages classes. Corpus-wide results are also given with rectangles weighted equally and broken down by dataset/reaction class. Residual error after Q² combines nonlinear response change, substrate combinations, measurement and averaging error, and cannot be split directly into a pure chemical interaction.', '', '## Files', '', '- rectangle_target_effects.csv.gz (round4 parent): shared strict four-cell target means.', '- rectangle_input_coverage.csv: all 830 rectangles and representative-axis conditions; model_coverage.csv: support of the 825 main rectangles in each panel/hold-out.', '- directional_results.csv, symmetric_rectangle_results.csv: directional and within-rectangle symmetric errors.', '- aggregate_results.csv: rectangle-equal and reaction-class-equal weighting, global and by dataset/reaction class.', '- fold_calibration.csv, oof_predictions.csv.gz: training slopes, intercepts and predictions on the common test sets.', '- calibration_instability.csv, calibration_extreme_targets.csv: contribution of all 825 rectangles to aggregate error, and the targets, four-cell yields and training ranges of extreme predictions.', '- chemical_examples.csv, run_summary.json: specific chemical examples and reproducible scale.'])
    axis_path = OUT / 'axis_symmetric_aggregate_results.csv'
    if axis_path.exists():
        axis = pd.read_csv(axis_path, low_memory=False)
        selected = axis.loc[axis.aggregation.eq('panel|scheme|test_weighting') & axis.panel.eq('full') & axis.test_weighting.eq('target_equal')]
        section = ['## Scientific summary symmetric over both axes and both directions', '', 'Both axes use identical target/molecule hold-out folds; for each axis MSE is first averaged over both directions, then axes are averaged within each rectangle, and finally results are aggregated over reaction classes or rectangles.', '', '| Hold-out | Weighting | Unique rectangles/reaction classes | Direct Q² | Shift Q² | OLS magnitude Q² | Shrink-toward-1 ridge Q² | Median OLS rectangle Q² |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in selected.itertuples():
            section.append(f'| {row.scheme} | {row.aggregation_weighting} | {row.n_unique_rectangles}/{row.n_reaction_classes} | {row.q2_from_aggregate_mse_direct:.3f} | {row.q2_from_aggregate_mse_offset:.3f} | {row.q2_from_aggregate_mse_affine:.3f} | {row.q2_from_aggregate_mse_ridge_anchor_one:.3f} | {row.unweighted_median_rectangle_q2_affine:.3f} |')
        a = selected.loc[selected.scheme.eq('target') & selected.aggregation_weighting.eq('reaction_class_equal')].iloc[0]
        section.extend(['', f'Symmetric over both axes with reaction classes weighted equally, direct/shift/OLS magnitude/ridge Q² are {a.q2_from_aggregate_mse_direct:.3f}/{a.q2_from_aggregate_mse_offset:.3f}/{a.q2_from_aggregate_mse_affine:.3f}/{a.q2_from_aggregate_mse_ridge_anchor_one:.3f}. OLS improves on direct reuse in {100 * a.fraction_rectangles_affine_better_than_direct:.1f}% of rectangles (reaction classes weighted equally), and the fraction with positive Q² relative to the target-background training mean is {100 * a.fraction_rectangles_q2_positive_affine:.1f}%.', ''])
        sensitivity = axis.loc[axis.aggregation.eq('panel|scheme|test_weighting') & axis.panel.eq('all_four_ge5pct') & axis.scheme.eq('target') & axis.test_weighting.eq('target_equal') & axis.aggregation_weighting.eq('reaction_class_equal')].iloc[0]
        section.extend([f'The two-axis sensitivity with all four cells ≥5% keeps {sensitivity.n_unique_rectangles} rectangles/{sensitivity.n_reaction_classes} reaction classes, with direct/shift/OLS/ridge Q² of {sensitivity.q2_from_aggregate_mse_direct:.3f}/{sensitivity.q2_from_aggregate_mse_offset:.3f}/{sensitivity.q2_from_aggregate_mse_affine:.3f}/{sensitivity.q2_from_aggregate_mse_ridge_anchor_one:.3f}.', ''])
        lines[6:6] = section
        lines.extend(['', 'Alternate-axis errors and per-fold parameters are saved as alternate_axis_directional_results.csv and alternate_axis_fold_calibration.csv; the latter records test-target hashes identical to those of the actual first-axis OOF. axis_symmetric_rectangle_results.csv and axis_symmetric_aggregate_results.csv hold the two-axis symmetric results.'])
    return '\n'.join(lines) + '\n'

def run_analysis(targets, coverage):
    started = time.monotonic()
    validate_calibration()
    features = pd.read_csv(ROOT / 'data' / 'molecular_features.csv', keep_default_na=False, low_memory=False)
    canonical = features.loc[features.parsing_status.eq('valid')].set_index('smiles').canonical_smiles.to_dict()
    main_ids = set(coverage.loc[coverage.n_common_targets.ge(20), 'rectangle_id'])
    assert len(main_ids) == 825
    target_groups = targets.loc[targets.rectangle_id.isin(main_ids)].groupby('rectangle_id', sort=True)
    metadata = coverage.set_index('rectangle_id')
    directional, fitted, support, extreme_targets = ([], [], [], [])
    written, oof_rows = (False, 0)
    with gzip.open(OUT / 'oof_predictions.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        for number, (rectangle, full) in enumerate(target_groups, 1):
            info = metadata.loc[rectangle]
            for panel in ['full', 'all_four_ge5pct']:
                whole = full if panel == 'full' else full.loc[full[YIELDS].ge(0.05).all(axis=1)]
                for scheme, role in [('target', None), ('sub_1_molecule', 'sub_1_smiles'), ('sub_2_molecule', 'sub_2_smiles')]:
                    part = whole.copy()
                    if role is not None:
                        part['focal_molecule'] = part[role].map(canonical)
                        part = part.loc[part.focal_molecule.notna()].copy()
                    n_molecules = part.focal_molecule.nunique() if role is not None else np.nan
                    eligible = len(part) >= 20 and (role is None or n_molecules >= 20)
                    base = {'rectangle_id': rectangle, 'representation_id': info.representation_id, 'dataset_id': info.dataset_id, 'rxn_class': info.rxn_class, 'panel': panel, 'scheme': scheme}
                    support.append({**base, 'n_full_targets': len(full), 'n_panel_targets': len(whole), 'n_valid_targets': len(part), 'n_focal_molecules': n_molecules, 'eligible': eligible, 'reason': 'eligible' if eligible else 'fewer_than20_targets' if len(part) < 20 else 'fewer_than20_molecules'})
                    if not eligible:
                        continue
                    part = part.sort_values('target_key').reset_index(drop=True)
                    oof, parameter_rows = case_oof(part, scheme, rectangle)
                    for column, value in base.items():
                        oof[column] = value
                    oof.to_csv(stream, index=False, header=not written)
                    written = True
                    oof_rows += len(oof)
                    fitted.extend(({**base, **row} for row in parameter_rows))
                    if panel == 'full' and scheme == 'target':
                        diagnostic = oof.copy()
                        diagnostic['affine_squared_error'] = (diagnostic.pred_affine - diagnostic.b) ** 2
                        diagnostic = diagnostic.nlargest(3, 'affine_squared_error').merge(pd.DataFrame(parameter_rows), on=['direction', 'fold'], validate='many_to_one')
                        diagnostic = diagnostic.merge(part[['target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', *YIELDS]], on='target_key', validate='many_to_one')
                        extreme_targets.append(diagnostic)
                    for direction, frame in oof.groupby('direction', sort=False):
                        predictions = {model: frame[f'pred_{model}'].to_numpy() for model in MODELS}
                        for weighting in ['target_equal', 'focal_molecule_equal'] if role else ['target_equal']:
                            molecules = frame.focal_molecule.to_numpy() if weighting == 'focal_molecule_equal' else None
                            metrics = evaluate_predictions(frame.a.to_numpy(), frame.b.to_numpy(), predictions, molecules)
                            metrics['n_focal_molecules'] = n_molecules
                            directional.append({**base, 'direction': direction, 'test_weighting': weighting, **metrics})
            if number % 100 == 0:
                print(f'Calibration: {number}/825 unique rectangles; {time.monotonic() - started:.1f}s', flush=True)
    directional = pd.DataFrame(directional)
    fitted, support = (pd.DataFrame(fitted), pd.DataFrame(support))
    symmetric = symmetric_results(directional)
    aggregate = aggregate_rectangles(symmetric)
    examples = choose_examples(symmetric, directional, fitted, targets, coverage)
    instability = symmetric.loc[symmetric.panel.eq('full') & symmetric.scheme.eq('target') & symmetric.test_weighting.eq('target_equal')].copy()
    class_sizes = instability.groupby('rxn_class').rectangle_id.transform('size')
    for model in MODELS:
        rectangle_contribution = instability[f'mse_{model}'] / len(instability)
        class_contribution = instability[f'mse_{model}'] / class_sizes / instability.rxn_class.nunique()
        instability[f'share_rectangle_equal_mse_{model}'] = rectangle_contribution / rectangle_contribution.sum()
        instability[f'share_reaction_class_equal_mse_{model}'] = class_contribution / class_contribution.sum()
    main_fits = fitted.loc[fitted.panel.eq('full') & fitted.scheme.eq('target')]
    stats = main_fits.groupby('rectangle_id').agg(min_a_train_variance=('a_train_variance', 'min'), max_abs_affine_slope=('affine_slope', lambda values: values.abs().max()), max_abs_affine_prediction=('max_abs_affine_prediction', 'max'), n_folds_abs_affine_slope_gt10=('affine_slope', lambda values: int(values.abs().gt(10).sum())))
    instability = instability.merge(stats, on='rectangle_id', validate='one_to_one')
    diagnostic = pd.concat(extreme_targets, ignore_index=True).nlargest(30, 'affine_squared_error')
    diagnostic = diagnostic.merge(coverage[['rectangle_id', 'focal_slot', 'component0_name', 'component1_name', 'context_slot', 'context0_name', 'context1_name', 'condition_c0_bg0', 'condition_c1_bg0', 'condition_c0_bg1', 'condition_c1_bg1']], on='rectangle_id', validate='many_to_one')
    for frame, filename in [(directional, 'directional_results.csv'), (symmetric, 'symmetric_rectangle_results.csv'), (aggregate, 'aggregate_results.csv'), (fitted, 'fold_calibration.csv'), (support, 'model_coverage.csv'), (examples, 'chemical_examples.csv'), (instability, 'calibration_instability.csv'), (diagnostic, 'calibration_extreme_targets.csv')]:
        write_csv(frame, filename)
    input_manifest = json.loads((OUT / 'input_manifest.json').read_text(encoding='utf-8'))
    manifest = {'status': 'completed', 'registered_unique_rectangles': 830, 'main_unique_rectangles': 825, 'input_target_mean_rows': len(targets), 'main_target_mean_rows': int(targets.rectangle_id.isin(main_ids).sum()), 'directional_result_rows': len(directional), 'symmetric_result_rows': len(symmetric), 'oof_prediction_rows': oof_rows, 'fold_calibration_rows': len(fitted), 'model_coverage_rows': len(support), 'seed': SEED, 'folds': 5, 'input_reconstruction': input_manifest, 'ols_variance_fallback_threshold': 1e-12, 'secondary_ridge_alpha': RIDGE_ALPHA, 'secondary_ridge_definition': 'unstandardized delta input; slope shrunk toward1, unpenalized intercept', 'target_holdout': 'whole product target; deterministic shuffled KFold', 'molecule_holdout': 'canonical focal molecule GroupKFold; minimum20 distinct focal molecules', 'known_test_information': 'a is measured source-background response for the held-out target; not structure-only prediction', 'fitting_weighting': 'target_equal', 'symmetric_summary': 'mean two directional MSE within rectangle, then calculate Q2', 'seconds': time.monotonic() - started}
    (OUT / 'CONTEXT_TRANSPORT_REPORT.md').write_text(report(manifest, aggregate, examples, coverage), encoding='utf-8')
    (OUT / 'run_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'input_reconstruction'}, ensure_ascii=False, indent=2), flush=True)

def verify_saved(targets, coverage):
    """Reconstruct saved predictions and losses with independent arithmetic."""
    started = time.monotonic()
    validate_calibration()
    case_keys = ['rectangle_id', 'panel', 'scheme']
    result_keys = case_keys + ['direction', 'test_weighting']
    results = pd.read_csv(OUT / 'directional_results.csv', low_memory=False).set_index(result_keys)
    model_coverage = pd.read_csv(OUT / 'model_coverage.csv', low_memory=False)
    expected_cases = set(map(tuple, model_coverage.loc[model_coverage.eligible, case_keys].to_numpy()))
    features = pd.read_csv(ROOT / 'data' / 'molecular_features.csv', keep_default_na=False, low_memory=False)
    canonical = features.loc[features.parsing_status.eq('valid')].set_index('smiles').canonical_smiles.to_dict()
    original_groups = targets.groupby('rectangle_id').indices
    seen_cases, seen_results, errors = (set(), set(), {})
    oof_rows = 0

    def close(name, actual, expected):
        a, b = (np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))
        assert a.shape == b.shape and np.allclose(a, b, rtol=1e-09, atol=1e-10, equal_nan=True), (name, key)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            errors[name] = max(errors.get(name, 0.0), float(np.max(np.abs(a[finite] - b[finite]))))

    def verify_case(key, case):
        nonlocal oof_rows
        assert key not in seen_cases
        seen_cases.add(key)
        rectangle, panel, scheme = key
        raw = targets.iloc[original_groups[rectangle]].copy()
        if panel == 'all_four_ge5pct':
            raw = raw.loc[raw[YIELDS].ge(0.05).all(axis=1)]
        role = 'sub_1_smiles' if scheme == 'sub_1_molecule' else 'sub_2_smiles' if scheme == 'sub_2_molecule' else None
        if role:
            raw = raw.loc[raw[role].isin(canonical)]
        raw = raw.set_index('target_key')
        forward = case.loc[case.direction.eq('bg0_to_bg1')].set_index('target_key')
        reverse = case.loc[case.direction.eq('bg1_to_bg0')].set_index('target_key')
        assert forward.index.is_unique and reverse.index.is_unique and (set(forward.index) == set(reverse.index) == set(raw.index))
        reverse = reverse.loc[forward.index]
        assert np.array_equal(forward.fold, reverse.fold)
        close('reverse_a_equals_forward_b', reverse.a, forward.b)
        close('reverse_b_equals_forward_a', reverse.b, forward.a)
        for direction, frame in case.groupby('direction', sort=False):
            assert frame.target_key.is_unique and set(frame.fold) == set(range(5))
            origin, destination = ('delta_bg0', 'delta_bg1') if direction == 'bg0_to_bg1' else ('delta_bg1', 'delta_bg0')
            close('source_a', frame.a, raw.loc[frame.target_key, origin])
            close('destination_b', frame.b, raw.loc[frame.target_key, destination])
            if role:
                expected_molecules = raw.loc[frame.target_key, role].map(canonical).to_numpy()
                assert np.array_equal(expected_molecules, frame.focal_molecule)
                assert frame.groupby('focal_molecule').fold.nunique().max() == 1
            for fold in range(5):
                train, test = (frame.loc[frame.fold.ne(fold)], frame.loc[frame.fold.eq(fold)])
                a, b = (train.a.to_numpy(), train.b.to_numpy())
                expected_mean = b.mean()
                close('training_only_mean', test.pred_training_mean, np.repeat(expected_mean, len(test)))
                close('direct_prediction', test.pred_direct, test.a)
                close('offset_prediction', test.pred_offset, test.a + b.mean() - a.mean())
                if np.var(a) <= 1e-12:
                    expected_affine = np.full(len(test), expected_mean)
                else:
                    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(a)), a]), b, rcond=None)[0]
                    expected_affine = intercept + slope * test.a.to_numpy()
                close('independent_lstsq_affine_prediction', test.pred_affine, expected_affine)
                x = a - a.mean()
                r = b - a
                beta = float(np.dot(x, r - r.mean()) / (np.dot(x, x) + 10))
                close('ridge_anchor_one_prediction', test.pred_ridge_anchor_one, test.a + r.mean() + beta * (test.a - a.mean()))
                if role:
                    assert not set(train.focal_molecule) & set(test.focal_molecule)
            for weighting in ['target_equal', 'focal_molecule_equal'] if role else ['target_equal']:
                rkey = (*key, direction, weighting)
                assert rkey in results.index
                seen_results.add(rkey)
                saved = results.loc[rkey]
                assert int(saved.n_targets) == len(frame)
                for model in MODELS:
                    loss = pd.DataFrame({'squared': (frame[f'pred_{model}'].to_numpy() - frame.b.to_numpy()) ** 2, 'absolute': np.abs(frame[f'pred_{model}'].to_numpy() - frame.b.to_numpy())})
                    if weighting == 'focal_molecule_equal':
                        loss['molecule'] = frame.focal_molecule.to_numpy()
                        loss = loss.groupby('molecule')[['squared', 'absolute']].mean()
                    mse, mae = (loss.squared.mean(), loss.absolute.mean())
                    close('saved_mse', saved[f'mse_{model}'], mse)
                    close('saved_rmse_pp', saved[f'rmse_{model}_pp'], 100 * np.sqrt(mse))
                    close('saved_mae_pp', saved[f'mae_{model}_pp'], 100 * mae)
        oof_rows += len(case)
    carry = None
    for chunk in pd.read_csv(OUT / 'oof_predictions.csv.gz', keep_default_na=False, low_memory=False, chunksize=100000):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = tuple(chunk.iloc[-1][case_keys])
        tail = np.ones(len(chunk), dtype=bool)
        for column, value in zip(case_keys, last):
            tail &= chunk[column].eq(value).to_numpy()
        carry = chunk.loc[tail].copy()
        for key, case in chunk.loc[~tail].groupby(case_keys, sort=False):
            verify_case(key, case)
    if carry is not None:
        key = tuple(carry.iloc[0][case_keys])
        verify_case(key, carry)
    assert seen_cases == expected_cases and seen_results == set(results.index)
    symmetric = pd.read_csv(OUT / 'symmetric_rectangle_results.csv', low_memory=False)
    for row in symmetric.to_dict('records'):
        key = tuple((row[column] for column in case_keys))
        pair = results.loc[[(*key, direction, row['test_weighting']) for direction in ['bg0_to_bg1', 'bg1_to_bg0']]]
        for model in MODELS:
            close('symmetric_mse', row[f'mse_{model}'], pair[f'mse_{model}'].mean())
    aggregate = pd.read_csv(OUT / 'aggregate_results.csv', low_memory=False)
    for row in aggregate.to_dict('records'):
        part = symmetric
        for column in row['aggregation'].split('|'):
            part = part.loc[part[column].eq(row[column])]
        assert len(part) == row['n_unique_rectangles']
        for model in MODELS:
            means = part[f'mse_{model}'] if row['aggregation_weighting'] == 'rectangle_equal' else part.groupby('rxn_class')[f'mse_{model}'].mean()
            close('aggregate_mse', row[f'mse_{model}'], means.mean())
    verified = {'status': 'passed', 'oof_rows': oof_rows, 'eligible_cases': len(seen_cases), 'directional_metric_rows': len(seen_results), 'max_absolute_errors': errors, 'seconds': time.monotonic() - started, 'scope': 'Independent original-target support, train/test identities, reversed directions, training-only predictions (OLS via lstsq), losses, symmetric and aggregate MSE; no refits of prior rounds.'}
    manifest = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    manifest['verification'] = verified
    (OUT / 'run_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(verified, ensure_ascii=False, indent=2), flush=True)

def target_hash(values):
    return hashlib.sha256('\n'.join(sorted(values)).encode('utf-8')).hexdigest()

def reference_fold_hashes():
    keys = ['rectangle_id', 'panel', 'scheme']
    hashes, carry = ({}, None)
    columns = [*keys, 'direction', 'fold', 'target_key']
    for chunk in pd.read_csv(OUT / 'oof_predictions.csv.gz', usecols=columns, keep_default_na=False, low_memory=False, chunksize=100000):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = tuple(chunk.iloc[-1][keys])
        tail = np.ones(len(chunk), dtype=bool)
        for column, value in zip(keys, last):
            tail &= chunk[column].eq(value).to_numpy()
        carry = chunk.loc[tail].copy()
        for key, group in chunk.loc[~tail].groupby(keys + ['direction', 'fold'], sort=False):
            hashes[key] = target_hash(group.target_key)
    if carry is not None:
        for key, group in carry.groupby(keys + ['direction', 'fold'], sort=False):
            hashes[key] = target_hash(group.target_key)
    return hashes

def alternate_axis_analysis(targets, coverage):
    started = time.monotonic()
    registry = pd.read_csv(OLD / 'rectangle_registry.csv', keep_default_na=False, low_memory=False)
    features = pd.read_csv(ROOT / 'data' / 'molecular_features.csv', keep_default_na=False, low_memory=False)
    canonical = features.loc[features.parsing_status.eq('valid')].set_index('smiles').canonical_smiles.to_dict()
    main_ids = set(coverage.loc[coverage.n_common_targets.ge(20), 'rectangle_id'])
    assert len(main_ids) == 825
    by_rectangle = registry.groupby('rectangle_id').indices
    source_rows = coverage.set_index('rectangle_id')
    hashes = reference_fold_hashes()
    directional, fitted = ([], [])
    support_cases, alternate_rectangles, verified_folds = (set(), set(), 0)
    corners = ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']
    for number, (rectangle, original) in enumerate(targets.loc[targets.rectangle_id.isin(main_ids)].groupby('rectangle_id', sort=True), 1):
        reference = source_rows.loc[rectangle]
        axes = registry.iloc[by_rectangle[rectangle]].sort_values('representation_id')
        assert axes.iloc[0].representation_id == reference.representation_id
        if len(axes) < 2:
            continue
        assert len(axes) == 2
        alternate = axes.iloc[1]
        alternate_rectangles.add(rectangle)
        mapping = {reference[corner]: column for corner, column in zip(corners, YIELDS)}
        assert set(mapping) == set(alternate[corners])
        transformed = original.copy()
        for corner, column in zip(corners, YIELDS):
            transformed[column] = original[mapping[alternate[corner]]].to_numpy()
        transformed['delta_bg0'] = transformed.y1_bg0 - transformed.y0_bg0
        transformed['delta_bg1'] = transformed.y1_bg1 - transformed.y0_bg1
        transformed['DD'] = transformed.delta_bg1 - transformed.delta_bg0
        assert np.allclose(transformed.DD, original.DD, rtol=0, atol=1e-12)
        for panel in ['full', 'all_four_ge5pct']:
            whole = transformed if panel == 'full' else transformed.loc[transformed[YIELDS].ge(0.05).all(axis=1)]
            for scheme, role in [('target', None), ('sub_1_molecule', 'sub_1_smiles'), ('sub_2_molecule', 'sub_2_smiles')]:
                part = whole.copy()
                if role:
                    part['focal_molecule'] = part[role].map(canonical)
                    part = part.loc[part.focal_molecule.notna()]
                nm = part.focal_molecule.nunique() if role else np.nan
                if len(part) < 20 or (role and nm < 20):
                    continue
                part = part.sort_values('target_key').reset_index(drop=True)
                support_cases.add((rectangle, panel, scheme))
                base = {'rectangle_id': rectangle, 'representation_id': alternate.representation_id, 'dataset_id': alternate.dataset_id, 'rxn_class': alternate.rxn_class, 'panel': panel, 'scheme': scheme}
                oof, parameters = case_oof(part, scheme, rectangle)
                for row in parameters:
                    frame = oof.loc[oof.direction.eq(row['direction']) & oof.fold.eq(row['fold'])]
                    digest = target_hash(frame.target_key)
                    hkey = (rectangle, panel, scheme, row['direction'], row['fold'])
                    assert hkey in hashes and digest == hashes[hkey], hkey
                    row.update(test_target_sha256=digest, reference_axis_same_test_targets=True, target_train_test_overlap=0, focal_molecule_train_test_overlap=0)
                    verified_folds += 1
                fitted.extend(({**base, **row} for row in parameters))
                for direction, frame in oof.groupby('direction', sort=False):
                    predictions = {model: frame[f'pred_{model}'].to_numpy() for model in MODELS}
                    for weighting in ['target_equal', 'focal_molecule_equal'] if role else ['target_equal']:
                        molecules = frame.focal_molecule.to_numpy() if weighting == 'focal_molecule_equal' else None
                        metrics = evaluate_predictions(frame.a.to_numpy(), frame.b.to_numpy(), predictions, molecules)
                        metrics['n_focal_molecules'] = nm
                        directional.append({**base, 'direction': direction, 'test_weighting': weighting, **metrics})
        if number % 200 == 0:
            print(f'Alternate axis: {number}/825 rectangles', flush=True)
    directional, fitted = (pd.DataFrame(directional), pd.DataFrame(fitted))
    write_csv(directional, 'alternate_axis_directional_results.csv')
    write_csv(fitted, 'alternate_axis_fold_calibration.csv')
    alternate_symmetric = symmetric_results(directional)
    reference_symmetric = pd.read_csv(OUT / 'symmetric_rectangle_results.csv', low_memory=False)
    joined = pd.concat([reference_symmetric, alternate_symmetric], ignore_index=True)
    keys = ['rectangle_id', 'dataset_id', 'rxn_class', 'panel', 'scheme', 'test_weighting']
    output = []
    for key, part in joined.groupby(keys, sort=False):
        assert part.representation_id.nunique() == len(part)
        assert part.n_targets.nunique() == 1
        row = dict(zip(keys, key))
        row.update(n_axes=len(part), axis_representation_ids=';'.join(sorted(part.representation_id)), n_targets=int(part.n_targets.iloc[0]), n_focal_molecules=part.n_focal_molecules.iloc[0])
        for column in part:
            if column.startswith('mse_') or column.startswith('mae_') or column.startswith('fraction_'):
                row[column] = part[column].mean()
        for model in MODELS:
            row[f'rmse_{model}_pp'] = 100 * np.sqrt(row[f'mse_{model}'])
            row[f'q2_{model}'] = 1 - row[f'mse_{model}'] / row['mse_training_mean'] if row['mse_training_mean'] > 1e-14 else np.nan
        output.append(row)
    symmetric = pd.DataFrame(output)
    full_main = symmetric.loc[symmetric.panel.eq('full') & symmetric.scheme.eq('target') & symmetric.test_weighting.eq('target_equal')]
    assert len(full_main) == 825 and full_main.n_axes.eq(2).all()
    expected = pd.read_csv(OUT / 'model_coverage.csv', low_memory=False)
    expected_cases = set(map(tuple, expected.loc[expected.eligible, ['rectangle_id', 'panel', 'scheme']].to_numpy()))
    assert support_cases == expected_cases
    aggregate = aggregate_rectangles(symmetric)
    write_csv(symmetric, 'axis_symmetric_rectangle_results.csv')
    write_csv(aggregate, 'axis_symmetric_aggregate_results.csv')
    manifest = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    manifest['axis_symmetry'] = {'status': 'completed', 'main_rectangles': 825, 'main_rectangles_with_two_axes': len(alternate_rectangles), 'alternate_directional_rows': len(directional), 'alternate_fold_rows': len(fitted), 'matched_reference_axis_test_fold_hashes': verified_folds, 'symmetric_result_rows': len(symmetric), 'method': 'Recipe-ID mapping of shared four corners; mean two directional MSE per axis, then mean axes within unique rectangle', 'seconds': time.monotonic() - started}
    (OUT / 'run_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    original_aggregate = pd.read_csv(OUT / 'aggregate_results.csv', low_memory=False)
    examples = pd.read_csv(OUT / 'chemical_examples.csv', keep_default_na=False, low_memory=False)
    (OUT / 'CONTEXT_TRANSPORT_REPORT.md').write_text(report(manifest, original_aggregate, examples, coverage), encoding='utf-8')
    print(json.dumps(manifest['axis_symmetry'], ensure_ascii=False, indent=2), flush=True)

def verify_axis_saved():
    keys = ['rectangle_id', 'panel', 'scheme', 'test_weighting']
    original = pd.read_csv(OUT / 'symmetric_rectangle_results.csv', low_memory=False).set_index(keys)
    alternate = pd.read_csv(OUT / 'alternate_axis_directional_results.csv', low_memory=False)
    alternate = alternate.groupby(keys)[[f'mse_{model}' for model in MODELS]].mean()
    symmetric = pd.read_csv(OUT / 'axis_symmetric_rectangle_results.csv', low_memory=False).set_index(keys)
    assert set(original.index) == set(alternate.index) == set(symmetric.index)
    alternate, symmetric = (alternate.loc[original.index], symmetric.loc[original.index])
    differences = {}
    for model in MODELS:
        expected = (original[f'mse_{model}'] + alternate[f'mse_{model}']) / 2
        actual = symmetric[f'mse_{model}']
        assert np.allclose(expected, actual, rtol=1e-10, atol=1e-12)
        differences[f'symmetric_mse_{model}'] = float(np.max(np.abs(expected - actual)))
    for model in ['direct', 'offset']:
        assert np.allclose(original[f'mse_{model}'], alternate[f'mse_{model}'], rtol=1e-10, atol=1e-12)
        differences[f'axis_invariant_mse_{model}'] = float(np.max(np.abs(original[f'mse_{model}'] - alternate[f'mse_{model}'])))
    assert symmetric.n_axes.eq(2).all()
    frame = symmetric.reset_index()
    aggregate = pd.read_csv(OUT / 'axis_symmetric_aggregate_results.csv', low_memory=False)
    for row in aggregate.to_dict('records'):
        part = frame
        for column in row['aggregation'].split('|'):
            part = part.loc[part[column].eq(row[column])]
        assert len(part) == row['n_unique_rectangles']
        for model in MODELS:
            expected = part[f'mse_{model}'].mean() if row['aggregation_weighting'] == 'rectangle_equal' else part.groupby('rxn_class')[f'mse_{model}'].mean().mean()
            assert np.isclose(expected, row[f'mse_{model}'], rtol=1e-10, atol=1e-12)
    folds = pd.read_csv(OUT / 'alternate_axis_fold_calibration.csv', low_memory=False)
    assert folds.reference_axis_same_test_targets.all()
    assert folds.target_train_test_overlap.eq(0).all() and folds.focal_molecule_train_test_overlap.eq(0).all()
    hashes = reference_fold_hashes()
    for row in folds.itertuples():
        assert hashes[row.rectangle_id, row.panel, row.scheme, row.direction, row.fold] == row.test_target_sha256
    verification = {'status': 'passed', 'symmetric_cases': len(symmetric), 'matched_actual_reference_oof_fold_hashes': len(folds), 'max_absolute_errors': differences, 'scope': 'Independent two-direction/two-axis MSE aggregation, direct/offset invariance, all aggregate MSE, and actual reference OOF fold hash readback.'}
    manifest = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    manifest['axis_symmetry']['verification'] = verification
    (OUT / 'run_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-only', '--build-input-only', action='store_true', dest='input_only')
    parser.add_argument('--reuse-input', action='store_true')
    parser.add_argument('--verify-existing', action='store_true', help='Audit saved OOF values without rerunning models')
    parser.add_argument('--summarize-existing', action='store_true', help='Refresh aggregate/report fields without rerunning models')
    parser.add_argument('--alternate-axis', action='store_true', help='Calculate second-axis and axis-symmetric summaries')
    parser.add_argument('--verify-axis-existing', action='store_true', help='Audit saved two-axis aggregation and fold hashes')
    args = parser.parse_args()
    if args.reuse_input or args.verify_existing or args.summarize_existing or args.alternate_axis or args.verify_axis_existing:
        targets = pd.read_csv(INPUT, keep_default_na=False, low_memory=False)
        coverage = pd.read_csv(OUT / 'rectangle_input_coverage.csv', keep_default_na=False, low_memory=False)
    else:
        targets, coverage = build_input()
    if args.input_only:
        return
    if args.verify_existing:
        verify_saved(targets, coverage)
        return
    if args.verify_axis_existing:
        verify_axis_saved()
        return
    if args.alternate_axis:
        alternate_axis_analysis(targets, coverage)
        return
    if args.summarize_existing:
        directional = pd.read_csv(OUT / 'directional_results.csv', low_memory=False)
        fitted = pd.read_csv(OUT / 'fold_calibration.csv', low_memory=False)
        symmetric = symmetric_results(directional)
        aggregate = aggregate_rectangles(symmetric)
        examples = choose_examples(symmetric, directional, fitted, targets, coverage)
        write_csv(symmetric, 'symmetric_rectangle_results.csv')
        write_csv(aggregate, 'aggregate_results.csv')
        write_csv(examples, 'chemical_examples.csv')
        manifest = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
        (OUT / 'CONTEXT_TRANSPORT_REPORT.md').write_text(report(manifest, aggregate, examples, coverage), encoding='utf-8')
        return
    run_analysis(targets, coverage)
    alternate_axis_analysis(targets, coverage)
if __name__ == '__main__':
    main()
