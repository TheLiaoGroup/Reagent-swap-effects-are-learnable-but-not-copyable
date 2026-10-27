"""Report66: mean and oriented-difference channels of frozen two-cell panels."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from chemical_mode_recovery import fit_clean_covariance
from chemical_observation_specificity import build_coverage, check_original_baselines, domain_masks
from chemically_aligned_measurements import FOLD_KEYS, aggregate_scores, average_draw_scores, digest, oriented_difference, panel_predictions, score_values, validate_training_sample
from chemical_response_structure import AMBIGUOUS
from response_mode_ablation import rank1_diagonal_covariance
from shrinkage_covariance_analysis import matrix_path
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'chemical_observation_channels'
REPORT = HERE / '66_CHEMICAL_OBSERVATION_CHANNELS_2026-10-17.md'
FIGURE = FIGURES / '52_CHEMICAL_OBSERVATION_CHANNELS.png'
FROZEN_DESIGN = HERE / 'chemical_observation_channels.py'
MODELS, FITS, CHANNELS, KINDS = (('full', 'rank1'), ('frozen', 'optimal'), ('s', 'd', 'sd'), ('raw', 'clip'))
POLICIES = ('semantic', 'different_any', 'different_other', 'random')
DESIGN = {'timing': 'Retrospective follow-up after reports63/65; design frozen before report66 channel scoring, not independent confirmation.', 'registry': 'Exact report65 all1808/18screens registry;649same references;strict641/relaxed643;8fixed random draws;no reselection.', 'coordinates': 'Centered endpoints v=y-mu;S=(v_a+v_b)/2;D=r(v_a-v_b);reference own inherited orientation, random r=+1.', 'cost': 'Every s,d,sd panel requires the same2planned cells;all channel modes require both finite endpoints, else common marginal.', 'frozen': 'Endpoint coefficients jointly fitted on SD under each TRAIN covariance;retain S, D, or both coefficients without refitting.', 'optimal': 'TRAIN-covariance linear-optimal S-only, D-only, SD endpoint coefficients;SD shared with frozen;original mean imputation unchanged.', 'clip': 'Reconstruct both target endpoint conditional means for the selected channels, clip each to[0,1], then oriented difference.', 'training': 'Inherited720TRAIN samples and original folds;quarter/full;pair/role1/role2;fullLW/rank1+diagonal;TRAINmu identical.', 'masks': 'Primary finite target pairs;strictall11/relaxedall10 panel complete shared masks;all modes/policies share denominators.', 'main': 'Strict641/fullhistory/pair/fullLW/optimal/raw/all_panels_complete/context_balanced/family_macro;C=(R_same,S-R_same,SD)-(R_other,S-R_other,SD).', 'aggregation': 'Equal8randomdraw SSE before fold;sumSSE/n edge;equal edge/context,context/screen,screen/family;familymacro primary;swapbalanced sensitivity.', 'bridge': 'Original partially available two-cell covariance prediction kept separately;all649report63 and strict/relaxed report65 scores and masks reproduced.', 'limits': 'Observed selected population;channels correlated;frozen ablation not mechanistic contribution;no independent edge/draw significance or causal intervention.'}

def channel_coordinates(values, anchor_means, reference_orientation):
    if int(reference_orientation) not in (-1, 1):
        raise ValueError('reference orientation must be signed')
    centered = np.asarray(values, float) - np.asarray(anchor_means, float)
    transform = np.array([[0.5, 0.5], [reference_orientation, -reference_orientation]], float)
    return centered @ transform.T

def endpoint_contrasts(predicted, orientation):
    raw = orientation * (predicted[:, 0] - predicted[:, 1])
    clipped = np.clip(predicted, 0.0, 1.0)
    return (raw, orientation * (clipped[:, 0] - clipped[:, 1]))

def channel_predictions(fit, rank1_covariance, values, anchors, endpoints, orientation, reference_orientation):
    """Accept only the two reference columns; every channel has the same availability."""
    anchors, endpoints = (np.asarray(anchors, int), np.asarray(endpoints, int))
    values, mu = (np.asarray(values, float), np.asarray(fit['mu'], float))
    if len(anchors) != 2 or len(set(anchors)) != 2 or len(endpoints) != 2 or set(anchors) & set(endpoints) or (values.ndim != 2) or (values.shape[1] != 2):
        raise ValueError('two disjoint observed anchors and hidden endpoints required')
    if int(orientation) not in (-1, 1):
        raise ValueError('target orientation must be signed')
    z = channel_coordinates(values, mu[anchors], reference_orientation)
    complete = np.isfinite(values).all(axis=1)
    z = np.where(complete[:, None], z, 0.0)
    transform = np.array([[0.5, 0.5], [reference_orientation, -reference_orientation]], float)
    predictions, audits = ({}, [])
    for model, covariance in (('full', fit['cov']), ('rank1', rank1_covariance)):
        aa = covariance[np.ix_(anchors, anchors)]
        zz = transform @ aa @ transform.T
        ez = covariance[np.ix_(endpoints, anchors)] @ transform.T
        used_pinv = False
        try:
            joint = np.linalg.solve(zz, ez.T).T
        except np.linalg.LinAlgError:
            joint = ez @ np.linalg.pinv(zz, rcond=1e-12)
            used_pinv = True
        if not np.isfinite(joint).all() or np.any(np.diag(zz) < -1e-12):
            raise ValueError('invalid channel covariance')
        scalar = np.zeros_like(joint)
        for index in (0, 1):
            if zz[index, index] > 0:
                scalar[:, index] = ez[:, index] / zz[index, index]
        for family in FITS:
            for channel in CHANNELS:
                indices = [0, 1] if channel == 'sd' else [0] if channel == 's' else [1]
                coefficient = joint if family == 'frozen' or channel == 'sd' else scalar
                predicted = np.tile(mu[endpoints], (len(values), 1)) + z[:, indices] @ coefficient[:, indices].T
                raw, clip = endpoint_contrasts(predicted, orientation)
                predictions[f'{model}_{family}_{channel}_raw'] = raw
                predictions[f'{model}_{family}_{channel}_clip'] = clip
        target_cross = orientation * (ez[0] - ez[1])
        target_variance = float(covariance[endpoints[0], endpoints[0]] + covariance[endpoints[1], endpoints[1]] - 2 * covariance[endpoints[0], endpoints[1]])
        joint_target = orientation * (joint[0] - joint[1])
        scalar_target = orientation * (scalar[0] - scalar[1])
        risks = dict(full_sd_expected_risk=float(target_variance - 2 * target_cross @ joint_target + joint_target @ zz @ joint_target), optimal_s_expected_risk=float(target_variance - 2 * target_cross[0] * scalar_target[0] + zz[0, 0] * scalar_target[0] ** 2), optimal_d_expected_risk=float(target_variance - 2 * target_cross[1] * scalar_target[1] + zz[1, 1] * scalar_target[1] ** 2), frozen_s_expected_risk=float(target_variance - 2 * target_cross[0] * joint_target[0] + zz[0, 0] * joint_target[0] ** 2), frozen_d_expected_risk=float(target_variance - 2 * target_cross[1] * joint_target[1] + zz[1, 1] * joint_target[1] ** 2))
        audits.append(dict(model=model, joint_pinv=used_pinv, zero_s_variance=bool(zz[0, 0] <= 0), zero_d_variance=bool(zz[1, 1] <= 0), s_variance=float(zz[0, 0]), d_variance=float(zz[1, 1]), sd_covariance=float(zz[0, 1]), h_ss=float(zz[0, 0]), h_sd=float(zz[0, 1]), h_dd=float(zz[1, 1]), cross_t_s=float(target_cross[0]), cross_t_d=float(target_cross[1]), var_target=target_variance, sd_weight_s=float(joint_target[0]), sd_weight_d=float(joint_target[1]), optimal_s_weight=float(scalar_target[0]), optimal_d_weight=float(scalar_target[1]), endpoint_a_sd_weight_s=float(joint[0, 0]), endpoint_a_sd_weight_d=float(joint[0, 1]), endpoint_b_sd_weight_s=float(joint[1, 0]), endpoint_b_sd_weight_d=float(joint[1, 1]), endpoint_a_optimal_s_weight=float(scalar[0, 0]), endpoint_a_optimal_d_weight=float(scalar[0, 1]), endpoint_b_optimal_s_weight=float(scalar[1, 0]), endpoint_b_optimal_d_weight=float(scalar[1, 1]), d_given_s_conditional_variance=float(zz[1, 1] - zz[0, 1] ** 2 / zz[0, 0]) if zz[0, 0] > 0 else float(zz[1, 1]), s_given_d_conditional_variance=float(zz[0, 0] - zz[0, 1] ** 2 / zz[1, 1]) if zz[1, 1] > 0 else float(zz[0, 0]), expected_d_given_s_gain=risks['optimal_s_expected_risk'] - risks['full_sd_expected_risk'], expected_s_given_d_gain=risks['optimal_d_expected_risk'] - risks['full_sd_expected_risk'], channel_condition_number=float(np.linalg.cond(zz)), **risks))
    return (predictions, audits)

def channel_panel_registry(source, panels):
    lookup = source.set_index(['screen', 'edge_id'])
    rows = []
    for panel in panels.itertuples(index=False):
        target = lookup.loc[panel.screen, panel.edge_id]
        orientation = target.reference_orientation if panel.policy == 'semantic' else 1 if panel.policy == 'random' else target[panel.policy + '_orientation']
        rows.append({**panel._asdict(), 'reference_orientation': int(orientation), 'required_finite_cells': 2, 'channel_cost_cells': 2})
    return pd.DataFrame(rows)

def channel_derivatives(frame):
    output = {}
    for model in MODELS:
        for family in FITS:
            for kind in KINDS:
                increments = {}
                for policy in POLICIES:
                    s, d, sd = [f'{policy}_{model}_{family}_{channel}_{kind}_mse' for channel in CHANNELS]
                    if s in frame and sd in frame:
                        name = f'{policy}_{model}_{family}_d_given_s_{kind}_mse'
                        output[name] = frame[s] - frame[sd]
                        increments[policy] = output[name]
                        marginal = 'marginal_mse' if kind == 'raw' else 'marginal_clip_mse'
                        if marginal in frame:
                            output[f'{policy}_{model}_{family}_sd_gain_vs_marginal_{kind}_mse'] = frame[marginal] - frame[sd]
                            output[f'{policy}_{model}_{family}_sd_recovery_vs_marginal_{kind}'] = 1 - frame[sd] / frame[marginal].replace(0, np.nan)
                    if d in frame and sd in frame:
                        output[f'{policy}_{model}_{family}_s_given_d_{kind}_mse'] = frame[d] - frame[sd]
                if 'semantic' in increments:
                    for other, label in (('different_any', 'any'), ('different_other', 'other'), ('random', 'random')):
                        if other in increments:
                            output[f'{label}_{model}_{family}_d_given_s_specificity_{kind}_mse'] = increments['semantic'] - increments[other]
                    for channel in CHANNELS:
                        semantic = f'semantic_{model}_{family}_{channel}_{kind}_mse'
                        for other, label in (('different_any', 'any'), ('different_other', 'other'), ('random', 'random')):
                            control = f'{other}_{model}_{family}_{channel}_{kind}_mse'
                            if semantic in frame and control in frame:
                                output[f'{label}_{model}_{family}_{channel}_specificity_{kind}_mse'] = frame[control] - frame[semantic]
    return pd.concat([frame, pd.DataFrame(output, index=frame.index)], axis=1)

def aggregate_domains(folds):
    outputs = [[] for _ in range(5)]
    for domain, group in folds.groupby('domain', sort=True):
        group = group.drop(columns=[name for name in group if name.endswith('_sse') and group[name].isna().all()])
        for output, frame in zip(outputs, aggregate_scores(group)):
            output.append(channel_derivatives(frame).assign(domain=domain))
    return tuple((pd.concat(output, ignore_index=True) for output in outputs))

def bridge_baseline_check(current, original):
    keys = FOLD_KEYS + ['edge_id', 'domain', 'scope']
    score_columns = [name for name in original if name.endswith('_sse')]
    renamed = current.rename(columns={name: name.replace('_bridge_', '_') for name in current if '_bridge_' in name})
    columns = ['n', 'target_mask_sha256', *score_columns]
    checked = renamed[keys + columns].merge(original[keys + columns], on=keys, how='outer', suffixes=('', '_65'), validate='one_to_one', indicator=True)
    if not checked._merge.eq('both').all() or not checked.n.eq(checked.n_65).all() or (not checked.target_mask_sha256.eq(checked.target_mask_sha256_65).all()):
        raise ValueError('inherited report65 target mask changed')
    errors = np.stack([(checked[name] - checked[name + '_65']).fillna(0.0).abs().to_numpy() for name in score_columns])
    checked['max_absolute_sse_difference'] = np.max(errors, axis=0)
    if checked.max_absolute_sse_difference.max() > 1e-11:
        raise ValueError('inherited report65 bridge changed')
    return checked[keys + ['n', 'target_mask_sha256', 'max_absolute_sse_difference']]

def availability_summary(folds):
    keys = ['domain', 'screen', 'family', 'split', 'fraction', 'scope']
    columns = [name for name in folds if name.endswith(('_observed_cells', '_complete_rows', '_partial_rows', '_fallback_rows'))]
    summary = folds.groupby(keys, as_index=False)[['n', *columns]].sum(min_count=1)
    for name in columns:
        summary[name + '_per_target'] = summary[name] / summary.n.replace(0, np.nan)
    return summary

def evaluate(registry, panel_registry, samples, original_folds):
    records, draws, audits, all649, diagnostics = ([], [], [], [], [])
    batches = [[] for _ in range(5)]
    scored_rows = 0
    start = time.monotonic()
    for screen, screen_samples in samples.groupby('screen', sort=True):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            matrix, conditions, identities = (saved['M'], saved['conds'].astype(str), saved['sps'].astype(str))
        roles = np.asarray([identity.split(' >> ') for identity in identities])
        excluded = [i for i, condition in enumerate(conditions) if (screen.split('|')[0], condition) in AMBIGUOUS]
        one = registry[registry.screen.eq(screen) & registry.eligible.eq(1)]
        panels_by_target = {key: group for key, group in panel_registry[panel_registry.screen.eq(screen)].groupby('edge_id', sort=False)}
        fold_groups = {key: group for key, group in original_folds[original_folds.screen.eq(screen)].groupby(['split', 'repeat', 'fold'], sort=True)}
        for sample in screen_samples.itertuples(index=False):
            source = fold_groups[sample.split, sample.repeat, sample.fold]
            tests, selected = [np.asarray(json.loads(getattr(sample, name)), int) for name in ('test_indices_json', 'training_indices_json')]
            validate_training_sample(sample, len(matrix), source.pair_index.to_numpy(int))
            if source.subpair.tolist() != identities[tests].tolist():
                raise ValueError('original TEST identities changed')
            if sample.split.startswith('role'):
                role = int(sample.split[-1]) - 1
                if set(roles[selected, role]) & set(roles[tests, role]):
                    raise ValueError('held-out role leaks into TRAIN')
            key = {field: getattr(sample, field) for field in FOLD_KEYS}
            fit = fit_clean_covariance(matrix[selected], excluded)
            rank1 = rank1_diagonal_covariance(fit['cov'])
            audit = {**key, 'n_training': len(selected), 'n_test': len(tests), 'n_original_eligible': len(one), 'n_strict': int(one.strict_eligible.sum()), 'n_relaxed': int(one.relaxed_eligible.sum()), 'n_excluded_ambiguous': len(excluded), 'shrinkage': fit['shrinkage'], 'imputed_training_cells': fit['imputed_cells'], 'train_register_exact': True, 'train_test_disjoint': True, 'quarter_nested_in_full': True, 'n_channel_panel_models': 0, 'n_joint_pinv': 0, 'n_zero_s_variance': 0, 'n_zero_d_variance': 0, 'maximum_complete_sd_prediction_error': 0.0, 'maximum_channel_condition_number': 0.0}
            for target in one.itertuples(index=False):
                endpoints = [int(target.condition_a_index), int(target.condition_b_index)]
                actual = oriented_difference(matrix[tests], *endpoints, target.orientation)
                finite_target = np.isfinite(actual)
                marginal_raw, marginal_clip = endpoint_contrasts(fit['mu'][endpoints][None, :], target.orientation)
                predictions = {'marginal': np.full(len(tests), marginal_raw[0]), 'marginal_clip': np.full(len(tests), marginal_clip[0])}
                panel_values, random_values, random_predictions = ({}, [], [])
                for panel in panels_by_target[target.edge_id].itertuples(index=False):
                    anchors = json.loads(panel.anchor_indices_json)
                    values = matrix[np.ix_(tests, anchors)]
                    prediction, channel_audits = channel_predictions(fit, rank1, values, anchors, endpoints, target.orientation, panel.reference_orientation)
                    bridge = panel_predictions(fit, rank1, values, anchors, endpoints, target.orientation)
                    complete = np.isfinite(values).all(axis=1)
                    for model in MODELS:
                        for kind in KINDS:
                            difference = np.abs(prediction[f'{model}_optimal_sd_{kind}'][complete] - bridge[f'{model}_{kind}'][complete])
                            if len(difference):
                                audit['maximum_complete_sd_prediction_error'] = max(audit['maximum_complete_sd_prediction_error'], float(difference.max()))
                    for row in channel_audits:
                        diagnostics.append({**key, 'edge_id': target.edge_id, 'context_id': target.context_id, 'swap_key': target.swap_key, 'policy': panel.policy, 'draw': panel.draw, 'reference_orientation': panel.reference_orientation, 'anchor_a_index': anchors[0], 'anchor_b_index': anchors[1], 'strict_eligible': target.strict_eligible, 'relaxed_eligible': target.relaxed_eligible, **row})
                        audit['n_channel_panel_models'] += 1
                        for name in ('joint_pinv', 'zero_s_variance', 'zero_d_variance'):
                            audit['n_' + name] += int(row[name])
                        audit['maximum_channel_condition_number'] = max(audit['maximum_channel_condition_number'], row['channel_condition_number'])
                    prediction.update({'bridge_' + name: value for name, value in bridge.items()})
                    if panel.policy == 'random':
                        random_values.append(values)
                        random_predictions.append(prediction)
                    else:
                        panel_values[panel.policy] = values
                        predictions.update({panel.policy + '_' + name: value for name, value in prediction.items()})
                if len(random_values) != 8 or audit['maximum_complete_sd_prediction_error'] > 1e-10:
                    raise ValueError('frozen panels or complete-anchor SD bridge changed')
                edge_key = {**key, 'edge_id': target.edge_id, 'context_id': target.context_id, 'swap_key': target.swap_key}
                original_predictions = {name.replace('_bridge_', '_'): value for name, value in predictions.items() if name.startswith('semantic_bridge_') or name == 'marginal'}
                original_score = score_values(actual, original_predictions, np.ones(len(tests), bool))
                original_random = average_draw_scores([score_values(actual, {name.replace('bridge_', ''): value for name, value in prediction.items() if name.startswith('bridge_')}, np.ones(len(tests), bool)) for prediction in random_predictions])
                all649.append({**edge_key, **original_score, **{'random_' + name: value for name, value in original_random.items() if name.endswith('_sse')}, 'target_mask_sha256': hashlib.sha256(tests[finite_target].astype('<i8').tobytes()).hexdigest()})
                for domain, eligible in (('strict_641', target.strict_eligible), ('relaxed_643', target.relaxed_eligible)):
                    if not eligible:
                        continue
                    active_policies = ['semantic', 'different_any'] + (['different_other'] if domain == 'strict_641' else [])
                    active = {name: value for name, value in predictions.items() if name.startswith('marginal') or any((name.startswith(policy + '_') for policy in active_policies))}
                    masks = domain_masks(panel_values['semantic'], panel_values['different_any'], panel_values.get('different_other'), random_values, domain)
                    for scope, mask in masks.items():
                        score = score_values(actual, active, mask)
                        valid = mask & finite_target
                        draw_scores, random_availability = ([], [])
                        for draw, (values, prediction) in enumerate(zip(random_values, random_predictions)):
                            draw_score = score_values(actual, prediction, mask)
                            observed = np.isfinite(values[valid])
                            counts = dict(observed_cells=int(observed.sum()), complete_rows=int(observed.all(axis=1).sum()), partial_rows=int((observed.sum(axis=1) == 1).sum()), fallback_rows=int((~observed.all(axis=1)).sum()))
                            draws.append({**edge_key, 'domain': domain, 'scope': scope, 'draw': draw, **draw_score, **counts})
                            draw_scores.append(draw_score)
                            random_availability.append(counts)
                        average = average_draw_scores(draw_scores)
                        if score['n'] != average['n']:
                            raise ValueError('channel policies changed target denominator')
                        availability = {}
                        for policy in active_policies:
                            observed = np.isfinite(panel_values[policy][valid])
                            availability.update({policy + '_observed_cells': int(observed.sum()), policy + '_complete_rows': int(observed.all(axis=1).sum()), policy + '_partial_rows': int((observed.sum(axis=1) == 1).sum()), policy + '_fallback_rows': int((~observed.all(axis=1)).sum())})
                        for name in random_availability[0]:
                            availability['random_' + name] = float(np.mean([row[name] for row in random_availability]))
                        records.append({**edge_key, 'domain': domain, 'scope': scope, **score, **{'random_' + name: value for name, value in average.items() if name.endswith('_sse')}, **availability, 'controls_same_reference': target.controls_same_reference, 'target_mask_sha256': hashlib.sha256(tests[valid].astype('<i8').tobytes()).hexdigest()})
            audits.append(audit)
        scored_rows += len(records)
        for batch, rows in zip(batches, (records, draws, audits, all649, diagnostics)):
            batch.append(pd.DataFrame(rows))
            rows.clear()
        print(f'CHANNELS {screen} fold_rows={scored_rows} seconds={time.monotonic() - start:.1f}', flush=True)
    return tuple((pd.concat(batch, ignore_index=True) for batch in batches))

def select_headline(headline, **extra):
    mask = headline.domain.eq('strict_641') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')
    for name, value in extra.items():
        mask &= headline[name].eq(value)
    return headline[mask]

def plot_figure(coverage, screens, headline):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    colors = {'semantic': '#287d78', 'different_other': '#bd5a4a', 'different_any': '#7b74a1', 'random': '#777777'}
    labels = {'semantic': 'Same swap', 'different_other': 'Different swap / other context', 'different_any': 'Different swap / any context', 'random': 'Random'}
    head = select_headline(headline, split='pair', fraction=1.0, scope='all_panels_complete').iloc[0]
    positions = np.arange(3)
    for offset, policy in zip((-0.24, -0.08, 0.08, 0.24), POLICIES):
        values = [10000.0 * head[f'{policy}_full_optimal_{channel}_raw_mse'] for channel in CHANNELS]
        axes[0, 0].scatter(positions + offset, values, color=colors[policy], label=labels[policy], s=34)
    axes[0, 0].set(xticks=positions, xticklabels=['Mean S', 'Difference D', 'Both SD'], ylabel='Family-macro MSE (pp squared)', title='A  TRAIN-optimal channels; common complete rows')
    axes[0, 0].legend(frameon=False, fontsize=8)
    categories = np.arange(4)
    for family, offset, marker in (('optimal', -0.1, 'o'), ('frozen', 0.1, 's')):
        values = [10000.0 * head[f'{policy}_full_{family}_d_given_s_raw_mse'] for policy in POLICIES]
        axes[0, 1].scatter(values, categories + offset, marker=marker, color='#287d78' if family == 'optimal' else '#b47d39', s=34, label=family)
    axes[0, 1].set(yticks=categories, yticklabels=[labels[p] for p in POLICIES], xlabel='MSE(S) - MSE(SD) (pp squared)', title='B  Difference information beyond endpoint mean')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(0, color='#777777', lw=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    one = screens[screens.domain.eq('strict_641') & screens.scope.eq('all_panels_complete') & screens.weighting.eq('context_balanced') & screens.split.eq('pair') & screens.fraction.eq(1.0)].sort_values('screen')
    y = np.arange(len(one))
    short = lambda s: s.replace('Buchwald-Hartwig(CMC)', 'BH(CMC)').replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O')
    axes[1, 0].barh(y, 10000.0 * one.other_full_optimal_d_given_s_specificity_raw_mse, color='#287d78')
    axes[1, 0].set(yticks=y, yticklabels=[short(s) for s in one.screen], xlabel='Same-swap gain - different-swap gain (pp squared)', title='C  All supported screens; context-balanced')
    axes[1, 0].invert_yaxis()
    axes[1, 0].axvline(0, color='#777777', lw=0.8)
    complete = select_headline(headline, scope='all_panels_complete')
    split_colors = {'pair': '#287d78', 'role1': '#bd5a4a', 'role2': '#7b74a1'}
    for split in split_colors:
        values = complete[complete.split.eq(split)].sort_values('fraction')
        axes[1, 1].plot(values.fraction, 10000.0 * values.other_full_optimal_d_given_s_specificity_raw_mse, color=split_colors[split], marker='o', label=split)
    axes[1, 1].set(xticks=[0.25, 1.0], xticklabels=['Quarter', 'Full'], ylabel='Incremental specificity C (pp squared)', title='D  Historical amount and identity split')
    axes[1, 1].axhline(0, color='#777777', lw=0.8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.15)
    fig.suptitle('Mean and difference channels of frozen two-cell chemical observations', fontsize=15)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

def write_report(headline, screens, coverage, availability, audits, bridge63, bridge65, evaluation_seconds):
    head = select_headline(headline, split='pair', fraction=1.0, scope='all_panels_complete').iloc[0]
    c = 10000.0 * head.other_full_optimal_d_given_s_specificity_raw_mse
    eligible_screens = screens[screens.domain.eq('strict_641') & screens.scope.eq('all_panels_complete') & screens.weighting.eq('context_balanced') & screens.split.eq('pair') & screens.fraction.eq(1.0)]
    positive = int(eligible_screens.other_full_optimal_d_given_s_specificity_raw_mse.gt(0).sum())
    lines = ['# 66. Mean and difference channels of chemical observations', '', 'Date: 2026-10-17. Analysis of the existing matrix; continues from the frozen panels of 63 and 65.', '', '## Core results', '', f'With the strict 641 edges, common identities over all 11 panels, full history/pair hold-out, full LW and the TRAIN-optimal linear prediction, the risk improvement from adding the difference D to the same-swap panel is {10000.0 * head.semantic_full_optimal_d_given_s_raw_mse:.3f} pp²; the different-swap, different-context control gives {10000.0 * head.different_other_full_optimal_d_given_s_raw_mse:.3f} pp². The incremental specificity C of the two is {c:.3f} pp², positive in {positive}/{len(eligible_screens)} screens.', '', '| Panel | S-only MSE | D-only MSE | SD MSE | Increment of D over S | Increment of S over D |', '|---|---:|---:|---:|---:|---:|']
    labels = {'semantic': 'Same swap', 'different_any': 'Different swap, any context', 'different_other': 'Different swap, different context', 'random': 'Mean of 8 random panels'}
    for policy in POLICIES:
        values = [head[f'{policy}_full_optimal_{channel}_raw_mse'] for channel in CHANNELS]
        values += [head[f'{policy}_full_optimal_d_given_s_raw_mse'], head[f'{policy}_full_optimal_s_given_d_raw_mse']]
        lines.append('| ' + labels[policy] + ' | ' + ' | '.join((f'{10000.0 * value:.3f}' for value in values)) + ' |')
    lines += ['', 'All values are equal-weight edge→context→screen→merged family macro-average raw MSE, in pp². C compares the risk improvement from adding D within each policy, then takes the difference between same swap and different swap with different context; it directly answers whether the value of the same-swap panel exceeds the mean information of the two reference endpoints.', '', '## Fixed design and estimands', '', 'Fully inherits from 65 the registration of 1,808 edges/384 contexts/18 screens, 649 same-swap references, different-swap any-context/different-context references and 8 random two-cell panels. The strict common domain has 641 edges and the relaxed domain 643; the strict domain covers 203 contexts, 16 screens/14 merged families; the 1,159 edges without a same-swap reference remain registered. The two different-swap controls select the same reference for 550 edges in the strict domain, an overlap within one fixed policy comparison.', '', 'The two reference endpoints are first centred on the same TRAIN mean, defining S=(v_a+v_b)/2, D=r(v_a-v_b). Chemical references use their own frozen direction r; random references use r=+1. The target is the endpoint difference in its own direction. All S, D and SD modes plan to measure two cells, and a channel is used only when both cells are finite; otherwise the same target marginal is used.', '', 'Two sets of mappings are computed, under full LW and rank1+diagonal covariance. The frozen set first estimates SD coefficients jointly and then keeps the S or D coefficients; the optimal set estimates the optimal linear S-only, D-only and SD mappings separately within the same TRAIN covariance. SD is identical in both sets. Only the original 720 TRAIN fits and the original mean imputation supply response information; channels and parameters are not chosen by TEST.', '', 'raw scores the predicted difference between the two target endpoints. The clip sensitivity first rebuilds the two endpoint predictions from the selected channels, clips each to [0,1] and then takes the directed difference; the difference itself is not clipped directly. Additivity of frozen contributions holds only for raw predictions; neither risk differences nor clip form a mechanistic decomposition.', '', 'The primary comparison is fixed as strict641/full history/pair/full LW/optimal/raw/all_panels_complete/context-balanced family-macro. primary still scores all targets with both endpoints finite; the complete mask requires both cells finite in all 11 panels (strict) or 10 panels (relaxed), used jointly by all policies and channels. The complete mask is conditional on availability, and the effects in the primary comparison describe the selected identity population.', '', '## All settings', '', '| Scope | Split | TRAIN | Same swap ΔD given S | Different-context control ΔD given S | C optimal | C frozen | C rank1 optimal | C optimal clip |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in select_headline(headline).sort_values(['scope', 'split', 'fraction']).itertuples(index=False):
        values = [row.semantic_full_optimal_d_given_s_raw_mse, row.different_other_full_optimal_d_given_s_raw_mse, row.other_full_optimal_d_given_s_specificity_raw_mse, row.other_full_frozen_d_given_s_specificity_raw_mse, row.other_rank1_optimal_d_given_s_specificity_raw_mse, row.other_full_optimal_d_given_s_specificity_clip_mse]
        lines.append(f'| {row.scope} | {row.split} | {row.fraction:g} | ' + ' | '.join((f'{10000.0 * value:.3f}' for value in values)) + ' |')
    lines += ['', 'The table above keeps positive and negative results in full; per-screen, per-family, any-context controls, random panels, the relaxed 643 domain and swap-balanced results are in the machine tables. Optimal prediction under the same TRAIN covariance does not guarantee a monotone improvement in held-out risk, so negative increments are reported as they are.', '', '## Availability and bridge to the original model', '', '| Domain | Split | TRAIN | primary target scores | Complete-mask scores | Complete-mask scored edges |', '|---|---|---:|---:|---:|---:|']
    h = headline[headline.macro.eq('family_macro') & headline.weighting.eq('context_balanced')]
    for row in h[h.scope.eq('primary')].sort_values(['domain', 'split', 'fraction']).itertuples(index=False):
        complete = h[h.domain.eq(row.domain) & h.scope.eq('all_panels_complete') & h.split.eq(row.split) & h.fraction.eq(row.fraction)].iloc[0]
        lines.append(f'| {row.domain} | {row.split} | {row.fraction:g} | {row.target_score_count} | {complete.target_score_count} | {complete.n_scored_edges}/{complete.n_edges} |')
    partial = availability[availability.domain.eq('strict_641') & availability.scope.eq('primary') & availability.split.eq('pair') & availability.fraction.eq(1.0)]
    lines += ['', f'In the strict-domain full/pair primary mask, the same-swap panel has {int(partial.semantic_partial_rows.sum())} target scores with only one finite cell; the new channel models fall back to the marginal on these identities, while the original model can still use a single-cell anchor. The original 65 model is therefore listed separately only as a bridge. On the complete mask, SD agrees with the original model prediction by prediction, with maximum absolute difference {audits.maximum_complete_sd_prediction_error.max():.3g}.', '', f'In the original 649 domain, {len(bridge63):,} fold-target risks and target masks were recomputed, with maximum SSE difference from the original 63 output {bridge63.max_absolute_risk_difference.max():.3g}; across the strict/relaxed masks, {len(bridge65):,} scores have maximum SSE difference from the original 65 bridge {bridge65.max_absolute_sse_difference.max():.3g}. quarter/full target masks are identical.', '', '## Scientific interpretation', '', 'S contains the response level shared by the two reference conditions, and D contains the directed response of that reference substitution. The test here is how much additional information adding D gives for predicting the target substitution from fixed two-cell observations, and whether this increment favours the same swap. S and D may be correlated; the frozen ablation also keeps the collinear dependence of the joint mapping. The results do not treat the two channels as independent chemical factors.', '', 'Fixed references come from the same screen and specified recipe semantics, and the background context can vary in several fields at once. Edges, references, folds and random draws are reused; no pseudo-independent significance inference is made over edge or draw counts. This analysis gives channel values and reproducible estimands within the existing matrix; numerical splits are not interpreted as causal mechanisms or prospective experimental gains.', '', '## Files and verification', '', f'producer is `{PREFIX}.py`; focused tests `test_{PREFIX}.py`. Training fits: {len(audits)}; full recomputation took {evaluation_seconds:.1f} s. New tables and data all start with `{PREFIX}_`; Figure 52 is `figures/{FIGURE.name}`. The manifest stores the design, dependencies and the SHA256 of the frozen 65 registry, the TRAIN audit, input invariance and bridge errors to the original 63/65. raw/clip, two models, two fitting modes, three channels, two domains, two scopes, two history fractions, three splits and two weightings are all kept.', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--summaries-only', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    inputs = [TABLES / 'chemical_observation_specificity_target_registry.csv', TABLES / 'chemical_observation_specificity_panel_registry.csv', TABLES / 'chemical_observation_specificity_fold_scores.csv.gz', TABLES / 'chemically_aligned_measurements_fold_scores.csv.gz', TABLES / 'chemical_mode_training_sample_registry.csv', DATA / 'cv_fold_registry.csv', FROZEN_DESIGN, HERE / 'chemical_observation_specificity.py', HERE / 'chemically_aligned_measurements.py', HERE / 'chemical_mode_recovery.py', HERE / 'chemical_response_structure.py', HERE / 'response_mode_ablation.py', HERE / 'shrinkage_covariance_analysis.py', HERE / 'training_size_intervention.py']
    inputs += sorted(DATA.glob('matrix_*.npz'))
    if not FROZEN_DESIGN.exists():
        raise ValueError('channel design must be frozen before registration or scoring')
    source_hashes = {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}
    registry, inherited_panels = (pd.read_csv(inputs[0], keep_default_na=False), pd.read_csv(inputs[1], keep_default_na=False))
    panels, coverage = (channel_panel_registry(registry, inherited_panels), build_coverage(registry))
    if (len(registry), registry.screen.nunique(), int(registry.eligible.sum()), int(registry.strict_eligible.sum()), int(registry.relaxed_eligible.sum()), len(panels)) != (1846, 19, 663, 655, 657, 7279):
        raise ValueError('inherited65 support changed')
    pd.testing.assert_frame_equal(panels[inherited_panels.columns], inherited_panels)
    registry_output, coverage_output, panel_output = [PREFIX + suffix for suffix in ('_target_registry.csv', '_coverage.csv', '_panel_registry.csv')]
    (TABLES / registry_output).write_bytes(inputs[0].read_bytes())
    coverage.to_csv(TABLES / coverage_output, index=False, encoding='utf-8-sig')
    panels.to_csv(TABLES / panel_output, index=False, encoding='utf-8-sig')
    registry_hashes = {name: digest(TABLES / name) for name in (registry_output, coverage_output, panel_output)}
    design_path = DATA / (PREFIX + '_design.json')
    design_path.write_text(json.dumps(dict(date='2026-10-17', design=DESIGN, source_hashes=source_hashes, registry_hashes=registry_hashes), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.register_only:
        print(json.dumps(dict(all_edges=1808, strict=641, relaxed=643, panels=len(panels), registry_hashes=registry_hashes), indent=2))
        return
    cache_names = [PREFIX + suffix for suffix in ('_fold_scores.csv.gz', '_random_draw_scores.csv.gz', '_fit_audit.csv', '_all649_baseline_check.csv.gz', '_report65_bridge_check.csv.gz', '_training_diagnostics.csv.gz')]
    if args.summaries_only:
        prior = json.loads((DATA / (PREFIX + '_manifest.json')).read_text(encoding='utf-8'))
        if prior['source_hashes'] != source_hashes or prior['registry_hashes'] != registry_hashes:
            raise ValueError('cached inputs or policies changed')
        for name in cache_names:
            if digest(TABLES / name) != prior['output_hashes'][name]:
                raise ValueError('cached score bytes changed')
        folds, draws, audits, bridge63, bridge65, diagnostics = [pd.read_csv(TABLES / name) for name in cache_names]
        evaluation_seconds = prior['evaluation_seconds']
    else:
        folds, draws, audits, baselines, diagnostics = evaluate(registry, panels, pd.read_csv(inputs[4]), pd.read_csv(inputs[5]))
        bridge63 = check_original_baselines(baselines, pd.read_csv(inputs[3]))
        bridge65 = bridge_baseline_check(folds, pd.read_csv(inputs[2]))
        evaluation_seconds = time.monotonic() - start
    if (len(folds), len(draws), len(audits), len(bridge63), len(bridge65)) != (104960, 839680, 760, 26520, 104960):
        raise ValueError('inherited evaluation denominator changed')
    if len(diagnostics) != 582320 or diagnostics.duplicated(FOLD_KEYS + ['edge_id', 'policy', 'draw', 'model']).any():
        raise ValueError('TRAIN-only panel diagnostic domain changed')
    paired = folds.pivot(index=FOLD_KEYS[:-1] + ['edge_id', 'domain', 'scope'], columns='fraction', values=['n', 'target_mask_sha256'])
    if not paired['n', 0.25].eq(paired['n', 1.0]).all() or not paired['target_mask_sha256', 0.25].eq(paired['target_mask_sha256', 1.0]).all():
        raise ValueError('historical amount changed target mask')
    edge, context, screen, family, headline = aggregate_domains(folds)
    availability = availability_summary(folds)
    outputs = dict(zip(cache_names, (folds, draws, audits, bridge63, bridge65, diagnostics)))
    outputs.update({PREFIX + suffix: frame for suffix, frame in (('_edge_scores.csv', edge), ('_context_scores.csv', context), ('_screen_scores.csv', screen), ('_family_scores.csv', family), ('_headline.csv', headline), ('_availability.csv', availability))})
    for name, frame in outputs.items():
        if not (args.summaries_only and name in cache_names):
            frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    plot_figure(coverage, screen, headline)
    write_report(headline, screen, coverage, availability, audits, bridge63, bridge65, evaluation_seconds)
    if source_hashes != {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}:
        raise ValueError('source files changed during report66')
    if registry_hashes != {name: digest(TABLES / name) for name in registry_hashes}:
        raise ValueError('frozen panel registry changed')
    output_hashes = {name: digest(TABLES / name) for name in outputs}
    output_hashes.update(registry_hashes)
    manifest = dict(date='2026-10-17', python=platform.python_version(), design=DESIGN, source_hashes=source_hashes, registry_hashes=registry_hashes, inputs_unchanged=True, inherited65_registry_bytes_identical=digest(TABLES / registry_output) == digest(inputs[0]), inherited65_panel_fields_identical=True, code_sha256=digest(Path(__file__)), test_code_sha256=digest(HERE / ('test_' + PREFIX + '.py')), report_sha256=digest(REPORT), figure_sha256=digest(FIGURE), design_sha256=digest(design_path), output_hashes=output_hashes, all_edges=1808, all_screens=18, original_eligible=649, strict_edges=641, relaxed_edges=643, original_unsupported=1159, strict_controls_same_reference=550, training_fits=720, panel_predictions=285000, fold_score_rows=len(folds), random_draw_score_rows=len(draws), training_diagnostic_rows=len(diagnostics), training_diagnostics_include_test_values=False, all649_baseline_rows=len(bridge63), all649_maximum_sse_difference=float(bridge63.max_absolute_risk_difference.max()), report65_bridge_rows=len(bridge65), report65_maximum_sse_difference=float(bridge65.max_absolute_sse_difference.max()), maximum_complete_sd_prediction_error=float(audits.maximum_complete_sd_prediction_error.max()), joint_pinv_count=int(audits.n_joint_pinv.sum()), zero_s_variance_count=int(audits.n_zero_s_variance.sum()), zero_d_variance_count=int(audits.n_zero_d_variance.sum()), target_mask_equal_across_history=True, target_count_equal_across_policies_and_modes=True, all_channels_require_two_finite_cells=True, evaluation_seconds=evaluation_seconds, elapsed_seconds=time.monotonic() - start, cached_score_reuse=args.summaries_only)
    (DATA / (PREFIX + '_manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({name: manifest[name] for name in ('strict_edges', 'relaxed_edges', 'training_fits', 'fold_score_rows', 'random_draw_score_rows', 'maximum_complete_sd_prediction_error', 'all649_maximum_sse_difference', 'report65_maximum_sse_difference', 'evaluation_seconds', 'elapsed_seconds')}, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
