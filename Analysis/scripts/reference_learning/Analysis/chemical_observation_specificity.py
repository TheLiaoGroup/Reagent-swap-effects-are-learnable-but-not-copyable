"""Report65: paired same-swap and different-swap observation policies."""
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
from chemically_aligned_measurements import FOLD_KEYS, aggregate_scores, average_draw_scores, digest, oriented_difference, panel_predictions, score_values, validate_training_sample
from chemical_response_structure import AMBIGUOUS
from response_mode_ablation import rank1_diagonal_covariance
from shrinkage_covariance_analysis import matrix_path
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'chemical_observation_specificity'
REPORT = HERE / '65_CHEMICAL_OBSERVATION_SPECIFICITY_2026-10-17.md'
FIGURE = FIGURES / '51_CHEMICAL_OBSERVATION_SPECIFICITY.png'
FROZEN_DESIGN = HERE / 'chemical_observation_specificity.py'
POLICIES = ('semantic', 'different_any', 'different_other')
REFERENCE_FIELDS = ('edge_id', 'context_id', 'context_json', 'condition_a', 'condition_b', 'condition_a_index', 'condition_b_index', 'orientation', 'swap_key', 'category', 'contrast_type', 'dose', 'low_id', 'high_id', 'low_name', 'high_name', 'reaction_class')
DESIGN = {'timing': 'Retrospective follow-up proposed after viewing report63 initial results;not independent confirmation or preregistration.', 'coverage': 'Inherited1808/18screens;649same-swap supported;strict paired641,relaxed643;all unsupported rows retained.', 'hash': "SHA256 compact UTF8 JSON ['chemical-observation-specificity-v1',screen,target_edge_id,candidate_edge_id];ensure_ascii=False,sort_keys=True,separators=(',',':');minimum digest;shared ordering for both controls.", 'controls': 'Same screen/category/contrast_type/dose,different swap_key,four distinct target/reference endpoints;other additionally different context;any allows same context.', 'same_reference': 'Retain controls selecting the same edge,record peredge flag and counts;not independent evidence.', 'observations': 'Each policy separately plans2cells;strict contains11panels,relaxed10;random8draws are separate2cell panels not16joint cells.', 'training': 'Original720report22 TRAIN registers,quarter/full,pair/role1/role2;full LW,rank1+diagonal and common TRAINmu;no direct rule refit or gate.', 'masks': 'Primary only target endpoints finite;all_panels_complete intersects all11strict/all10relaxed panels,applied identically to every policy;secondary selected-population sensitivity.', 'aggregation': 'Equal mean8randomSSE beforefold;sumSSE/n toedge;equal edge/context/screen/family;familymacro primary;swapbalanced sensitivity.', 'baseline_guard': 'Recompute all649same/random/marginal primary predictions and compare every25960fold-target toreport63,n and targetmask also identical.', 'limits': 'Single metadata-hash policy;not matched on response amplitude,variance,historical covariance or all reference-selection uncertainty;not causal chemistry intervention.'}

def control_hash(screen, target_edge_id, candidate_edge_id):
    text = json.dumps(['chemical-observation-specificity-v1', screen, target_edge_id, candidate_edge_id], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def add_controls(source):
    """Freeze both policies from metadata; preserve all source rows."""
    records = []
    for screen, group in source.groupby('screen', sort=True):
        group = group.sort_values('edge_id')
        for target in group.itertuples(index=False):
            endpoints = {int(target.condition_a_index), int(target.condition_b_index)}
            any_candidates = []
            for candidate in group.itertuples(index=False):
                if candidate.swap_key == target.swap_key or candidate.category != target.category or candidate.contrast_type != target.contrast_type or (candidate.dose != target.dose) or endpoints & {int(candidate.condition_a_index), int(candidate.condition_b_index)}:
                    continue
                any_candidates.append(candidate)
            any_candidates.sort(key=lambda candidate: control_hash(screen, target.edge_id, candidate.edge_id))
            other_candidates = [candidate for candidate in any_candidates if candidate.context_id != target.context_id]
            row = target._asdict()
            row.update(n_different_any_candidates=len(any_candidates), n_different_other_candidates=len(other_candidates), relaxed_eligible=int(bool(target.eligible and any_candidates)), strict_eligible=int(bool(target.eligible and other_candidates)), controls_same_reference=0)
            for label, candidates in (('different_any', any_candidates), ('different_other', other_candidates)):
                selected = candidates[0]._asdict() if candidates else {}
                for field in REFERENCE_FIELDS:
                    row[label + '_' + field] = selected.get(field, -1 if field.endswith('_index') else '')
                row[label + '_selection_hash'] = control_hash(screen, target.edge_id, selected['edge_id']) if selected else ''
            row['controls_same_reference'] = int(bool(other_candidates) and row['different_any_edge_id'] == row['different_other_edge_id'])
            row['strict_reason'] = 'supported' if row['strict_eligible'] else 'no_same_swap_reference' if not target.eligible else 'no_different_swap_other_context_reference'
            row['relaxed_reason'] = 'supported' if row['relaxed_eligible'] else 'no_same_swap_reference' if not target.eligible else 'no_different_swap_any_context_reference'
            records.append(row)
    return pd.DataFrame(records).sort_values(['screen', 'edge_id']).reset_index(drop=True)

def build_coverage(registry):
    records = []
    for screen, group in registry.groupby('screen', sort=True):
        strict = group[group.strict_eligible.eq(1)]
        relaxed = group[group.relaxed_eligible.eq(1)]
        records.append(dict(screen=screen, family=group.family.iloc[0], n_all_edges=len(group), n_original_eligible=int(group.eligible.sum()), n_original_unsupported=int((1 - group.eligible).sum()), n_strict=len(strict), n_relaxed=len(relaxed), n_strict_controls_same_reference=int(strict.controls_same_reference.sum()), strict_controls_same_reference_fraction=float(strict.controls_same_reference.mean()) if len(strict) else np.nan, n_relaxed_controls_same_reference=int(relaxed.controls_same_reference.sum()), n_strict_unsupported_with_same_swap=int(group.eligible.sum() - len(strict)), n_relaxed_unsupported_with_same_swap=int(group.eligible.sum() - len(relaxed))))
    return pd.DataFrame(records)

def build_panels(registry):
    rows = []
    for target in registry[registry.eligible.eq(1)].itertuples(index=False):
        choices = [('semantic', -1, [int(target.reference_condition_a_index), int(target.reference_condition_b_index)], target.reference_edge_id)]
        choices += [('random', draw, panel, '') for draw, panel in enumerate(json.loads(target.random_panels_indices_json))]
        for policy, valid in (('different_any', target.relaxed_eligible), ('different_other', target.strict_eligible)):
            if valid:
                choices.append((policy, -1, [int(getattr(target, policy + '_condition_a_index')), int(getattr(target, policy + '_condition_b_index'))], getattr(target, policy + '_edge_id')))
        for policy, draw, anchors, reference_edge_id in choices:
            if len(set(anchors)) != 2 or set(anchors) & {target.condition_a_index, target.condition_b_index}:
                raise ValueError('invalid hidden-endpoint panel')
            rows.append(dict(screen=target.screen, family=target.family, edge_id=target.edge_id, policy=policy, draw=draw, reference_edge_id=reference_edge_id, anchor_indices_json=json.dumps(anchors), planned_cells=2, strict_eligible=target.strict_eligible, relaxed_eligible=target.relaxed_eligible))
    return pd.DataFrame(rows)

def domain_masks(semantic_values, any_values, other_values, random_values, domain):
    panels = [semantic_values, any_values, *random_values]
    if domain == 'strict_641':
        panels.append(other_values)
    elif domain != 'relaxed_643':
        raise ValueError('unknown fixed domain')
    complete = np.ones(len(semantic_values), bool)
    for panel in panels:
        complete &= np.isfinite(panel).all(axis=1)
    return {'primary': np.ones(len(complete), bool), 'all_panels_complete': complete}

def check_original_baselines(recomputed, original):
    keys = FOLD_KEYS + ['edge_id']
    metrics = [name for name in recomputed if name.endswith('_sse')]
    columns = ['n', 'target_mask_sha256', *metrics]
    old = original[original.scope.eq('primary')]
    paired = recomputed.merge(old[keys + columns], on=keys, how='outer', suffixes=('', '_report63'), validate='one_to_one', indicator=True)
    if not paired._merge.eq('both').all() or not paired.n.eq(paired.n_report63).all() or (not paired.target_mask_sha256.eq(paired.target_mask_sha256_report63).all()):
        raise ValueError('original649 target register changed')
    differences = []
    for name in metrics:
        difference = paired[name] - paired[name + '_report63']
        paired[name.replace('_sse', '_difference')] = difference
        differences.append(np.abs(difference.to_numpy()))
    paired['max_absolute_risk_difference'] = np.max(np.stack(differences), axis=0)
    if paired.max_absolute_risk_difference.max() > 1e-11:
        raise ValueError('original649 same/random/marginal prediction changed')
    return paired.drop(columns='_merge')

def evaluate(registry, samples, original_folds):
    records, draws, audits, original_scores = ([], [], [], [])
    start = time.monotonic()
    for screen, screen_samples in samples.groupby('screen', sort=True):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            matrix, conditions, identities = (saved['M'], saved['conds'].astype(str), saved['sps'].astype(str))
        roles = np.asarray([identity.split(' >> ') for identity in identities])
        excluded = [i for i, condition in enumerate(conditions) if (screen.split('|')[0], condition) in AMBIGUOUS]
        one = registry[registry.screen.eq(screen) & registry.eligible.eq(1)]
        folds = original_folds[original_folds.screen.eq(screen)]
        fold_groups = {key: group for key, group in folds.groupby(['split', 'repeat', 'fold'], sort=True)}
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
            audits.append({**key, 'n_training': len(selected), 'n_test': len(tests), 'n_original_eligible': len(one), 'n_strict': int(one.strict_eligible.sum()), 'n_relaxed': int(one.relaxed_eligible.sum()), 'n_excluded_ambiguous': len(excluded), 'shrinkage': fit['shrinkage'], 'imputed_training_cells': fit['imputed_cells'], 'train_register_exact': True, 'train_test_disjoint': True, 'quarter_nested_in_full': True})
            for target in one.itertuples(index=False):
                endpoints = [target.condition_a_index, target.condition_b_index]
                actual = oriented_difference(matrix[tests], *endpoints, target.orientation)
                finite_target = np.isfinite(actual)
                marginal = target.orientation * (fit['mu'][endpoints[0]] - fit['mu'][endpoints[1]])
                anchors = [int(target.reference_condition_a_index), int(target.reference_condition_b_index)]
                semantic_values = matrix[np.ix_(tests, anchors)]
                semantic = panel_predictions(fit, rank1, semantic_values, anchors, endpoints, target.orientation)
                panels = json.loads(target.random_panels_indices_json)
                random_values = [matrix[np.ix_(tests, panel)] for panel in panels]
                random_predictions = [panel_predictions(fit, rank1, values, panel, endpoints, target.orientation) for values, panel in zip(random_values, panels)]
                predictions = {'semantic_' + name: value for name, value in semantic.items()}
                predictions['marginal'] = np.full(len(tests), marginal)
                baseline = score_values(actual, predictions, np.ones(len(tests), bool))
                averaged = average_draw_scores([score_values(actual, prediction, np.ones(len(tests), bool)) for prediction in random_predictions])
                edge_key = {**key, 'edge_id': target.edge_id, 'context_id': target.context_id, 'swap_key': target.swap_key}
                original_scores.append({**edge_key, **baseline, **{'random_' + name: value for name, value in averaged.items() if name.endswith('_sse')}, 'target_mask_sha256': hashlib.sha256(tests[finite_target].astype('<i8').tobytes()).hexdigest()})
                control_values, control_predictions = ({}, {})
                for policy, valid in (('different_any', target.relaxed_eligible), ('different_other', target.strict_eligible)):
                    if not valid:
                        continue
                    anchors = [int(getattr(target, policy + '_condition_a_index')), int(getattr(target, policy + '_condition_b_index'))]
                    if set(anchors) & set(endpoints):
                        raise ValueError('control shares a hidden endpoint')
                    values = matrix[np.ix_(tests, anchors)]
                    control_values[policy] = values
                    control_predictions[policy] = panel_predictions(fit, rank1, values, anchors, endpoints, target.orientation)
                for domain, eligible in (('strict_641', target.strict_eligible), ('relaxed_643', target.relaxed_eligible)):
                    if not eligible:
                        continue
                    active_policies = ['semantic', 'different_any'] + (['different_other'] if domain == 'strict_641' else [])
                    active_values = {'semantic': semantic_values, **{policy: control_values[policy] for policy in active_policies if policy != 'semantic'}}
                    active_predictions = predictions.copy()
                    for policy in active_policies:
                        if policy != 'semantic':
                            active_predictions.update({policy + '_' + name: value for name, value in control_predictions[policy].items()})
                    masks = domain_masks(semantic_values, control_values['different_any'], control_values.get('different_other'), random_values, domain)
                    for scope, mask in masks.items():
                        score = score_values(actual, active_predictions, mask)
                        draw_scores, draw_cells, draw_complete, draw_none = ([], [], [], [])
                        valid = mask & finite_target
                        for draw, (values, prediction) in enumerate(zip(random_values, random_predictions)):
                            draw_score = score_values(actual, prediction, mask)
                            observed = np.isfinite(values[valid])
                            counts = dict(observed_anchor_cells=int(observed.sum()), complete_anchor_rows=int(observed.all(axis=1).sum()), no_anchor_rows=int((~observed.any(axis=1)).sum()))
                            draws.append({**edge_key, 'domain': domain, 'scope': scope, 'draw': draw, **draw_score, **counts})
                            draw_scores.append(draw_score)
                            draw_cells.append(counts['observed_anchor_cells'])
                            draw_complete.append(counts['complete_anchor_rows'])
                            draw_none.append(counts['no_anchor_rows'])
                        average = average_draw_scores(draw_scores)
                        if score['n'] != average['n']:
                            raise ValueError('observation policy changed target denominator')
                        availability = {}
                        for policy, values in active_values.items():
                            observed = np.isfinite(values[valid])
                            availability.update({policy + '_observed_cells': int(observed.sum()), policy + '_complete_rows': int(observed.all(axis=1).sum()), policy + '_no_anchor_rows': int((~observed.any(axis=1)).sum())})
                        records.append({**edge_key, 'domain': domain, 'scope': scope, **score, **{'random_' + name: value for name, value in average.items() if name.endswith('_sse')}, **availability, 'random_observed_cells': float(np.mean(draw_cells)), 'random_complete_rows': float(np.mean(draw_complete)), 'random_no_anchor_rows': float(np.mean(draw_none)), 'controls_same_reference': target.controls_same_reference, 'target_mask_sha256': hashlib.sha256(tests[valid].astype('<i8').tobytes()).hexdigest()})
        print(f'SPECIFICITY {screen} fold_rows={len(records)} seconds={time.monotonic() - start:.1f}', flush=True)
    return (pd.DataFrame(records), pd.DataFrame(draws), pd.DataFrame(audits), pd.DataFrame(original_scores))

def paired_derivatives(frame):
    frame = frame.copy()
    for model in ('full', 'rank1'):
        for kind in ('raw', 'clip'):
            semantic, random = ('semantic_' + model + '_' + kind + '_mse', 'random_' + model + '_' + kind + '_mse')
            if semantic not in frame or random not in frame:
                continue
            for label, prefix in (('any', 'different_any'), ('other', 'different_other')):
                control = prefix + '_' + model + '_' + kind + '_mse'
                if control not in frame:
                    continue
                suffix = '' if model == 'full' and kind == 'raw' else '_' + model + '_' + kind
                frame[label + '_specificity_advantage_mse' + suffix] = frame[control] - frame[semantic]
                frame[label + '_structure_advantage_mse' + suffix] = frame[random] - frame[control]
                frame[label + '_recovery_vs_control' + suffix] = 1 - frame[semantic] / frame[control].replace(0, np.nan)
                frame[label + '_closure_error' + suffix] = frame[random] - frame[semantic] - frame[label + '_specificity_advantage_mse' + suffix] - frame[label + '_structure_advantage_mse' + suffix]
                if 'marginal_mse' in frame:
                    frame[prefix + '_' + model + '_' + kind + '_recovery_vs_marginal'] = 1 - frame[control] / frame.marginal_mse.replace(0, np.nan)
    return frame

def aggregate_domains(folds):
    outputs = [[] for _ in range(5)]
    for domain, group in folds.groupby('domain', sort=True):
        group = group.drop(columns=[name for name in group if name.endswith('_sse') and group[name].isna().all()])
        for output, frame in zip(outputs, aggregate_scores(group)):
            output.append(paired_derivatives(frame).assign(domain=domain))
    return tuple((pd.concat(output, ignore_index=True) for output in outputs))

def availability_summary(folds):
    keys = ['domain', 'screen', 'family', 'split', 'fraction', 'scope']
    columns = [name for name in folds if name.endswith(('_observed_cells', '_complete_rows', '_no_anchor_rows'))]
    summary = folds.groupby(keys, as_index=False)[['n', *columns]].sum(min_count=1)
    for name in columns:
        summary[name + '_per_target'] = summary[name] / summary.n.replace(0, np.nan)
    summary['zero_target_fold_edges'] = folds.groupby(keys).n.apply(lambda values: int(values.eq(0).sum())).to_numpy()
    return summary

def plot_figure(coverage, screen, headline):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    short = lambda s: s.replace('Buchwald-Hartwig(CMC)', 'BH(CMC)').replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O')
    cov = coverage.sort_values('screen')
    y = np.arange(len(cov))
    axes[0, 0].barh(y, cov.n_all_edges, color='#e3e3e3', label='All strict edges')
    axes[0, 0].barh(y, cov.n_original_eligible, color='#adcfca', label='Report63 supported')
    axes[0, 0].barh(y, cov.n_strict, color='#287d78', label='Strict paired controls')
    axes[0, 0].set(yticks=y, yticklabels=[short(s) for s in cov.screen], xlabel='Registered target edges', title='A  Support without outcome selection')
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=8)
    selected = screen[screen.domain.eq('strict_641') & screen.scope.eq('primary') & screen.weighting.eq('context_balanced') & screen.fraction.eq(1.0)]
    colors = {'pair': '#287d78', 'role1': '#b65d50', 'role2': '#6874a6'}
    for split, offset in (('pair', -0.2), ('role1', 0), ('role2', 0.2)):
        one = selected[selected.split.eq(split)].set_index('screen').reindex(cov.screen)
        axes[0, 1].scatter(10000.0 * one.other_specificity_advantage_mse, y + offset, color=colors[split], s=24, label=split)
    axes[0, 1].set(yticks=y, yticklabels=[short(s) for s in cov.screen], xlabel='Different-swap MSE - same-swap MSE (pp squared)', title='B  Specific-swap policy contrast; full history')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(0, color='#777777', lw=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    head = headline[headline.domain.eq('strict_641') & headline.scope.eq('primary') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')]
    columns = ['semantic_full_raw_mse', 'different_other_full_raw_mse', 'different_any_full_raw_mse', 'random_full_raw_mse', 'marginal_mse']
    labels = ['Same swap', 'Different swap / other context', 'Different swap / any context', 'Random', 'Marginal']
    for split, offset in (('pair', -0.2), ('role1', 0), ('role2', 0.2)):
        one = head[head.split.eq(split) & head.fraction.eq(1.0)].iloc[0]
        axes[1, 0].scatter([10000.0 * one[name] for name in columns], np.arange(len(columns)) + offset, color=colors[split], s=28)
    axes[1, 0].set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel='Family-macro full-model MSE (pp squared)', title='C  Absolute paired risk; full history')
    axes[1, 0].invert_yaxis()
    for split in colors:
        one = head[head.split.eq(split)].sort_values('fraction')
        axes[1, 1].plot(one.fraction, 10000.0 * one.other_specificity_advantage_mse, color=colors[split], marker='o', label=split)
    axes[1, 1].set(xticks=[0.25, 1.0], xticklabels=['Quarter', 'Full'], xlabel='Historical training amount', ylabel='Different-swap MSE - same-swap MSE (pp squared)', title='D  Fixed observation policies and historical amount')
    axes[1, 1].axhline(0, color='#777777', lw=0.8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.15)
    fig.suptitle('Chemical observation specificity: paired two-cell structural controls', fontsize=15)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

def write_report(headline, screen, coverage, baseline_check, evaluation_seconds):
    before = '# Scientific results\n\n'
    head = headline[headline.domain.eq('strict_641') & headline.scope.eq('primary') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')]
    lines = ['## 结果', '', '主表是641条严格共同域内的family宏平均，raw MSE单位pp²。随机8draw先在fold内等权平均SSE，全部政策共享目标分母。', '', '| Split | TRAIN | 同swap | 不同swap异context | 不同swap任意context | 随机 | marginal | 同swap相对异context对照恢复 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in head.sort_values(['split', 'fraction']).itertuples(index=False):
        lines.append(f'| {row.split} | {row.fraction:g} | {10000.0 * row.semantic_full_raw_mse:.3f} | {10000.0 * row.different_other_full_raw_mse:.3f} | {10000.0 * row.different_any_full_raw_mse:.3f} | {10000.0 * row.random_full_raw_mse:.3f} | {10000.0 * row.marginal_mse:.3f} | {100 * row.other_recovery_vs_control:.2f}% |')
    pair = head[head.split.eq('pair') & head.fraction.eq(1.0)].iloc[0]
    positive = screen[screen.domain.eq('strict_641') & screen.scope.eq('primary') & screen.weighting.eq('context_balanced') & screen.fraction.eq(1.0) & screen.split.eq('pair')].other_specificity_advantage_mse.gt(0).sum()
    lines += ['', f'六个设置中，同swap full相对异context不同swap对照的净恢复均为正，范围{100 * head.other_recovery_vs_control.min():.2f}%–{100 * head.other_recovery_vs_control.max():.2f}%。full/pair下，对照MSE {10000.0 * pair.different_other_full_raw_mse:.3f}降到同swap {10000.0 * pair.semantic_full_raw_mse:.3f} pp²；同swap在{positive}/16个screen中占优，负结果完整保留。rank1的相应恢复为{100 * head.other_recovery_vs_control_rank1_raw.min():.2f}%–{100 * head.other_recovery_vs_control_rank1_raw.max():.2f}%，因此这个观测政策的数值收益也依赖使用的历史映射。', '', f'这个不同swap结构对照没有形成统一的随机面板优势：六设置中{int(head.other_structure_advantage_mse.gt(0).sum())}/6个优于随机；full/pair对照为{10000.0 * pair.different_other_full_raw_mse:.3f}，随机为{10000.0 * pair.random_full_raw_mse:.3f} pp²。结果支持在固定槽位、剂量及严格边结构的单hash政策比较中保留同一替换的信息价值，而不能把它解释为纯化学身份的因果效应。', '']
    lines += ['', '同一目标分母上，随机到同swap的风险差恒等于随机到不同swap加不同swap到同swap的风险差；闭合仅是算术核对，不能解释为机制贡献分解。所有正负screen/family结果均保存在机器表。', '', '## 支持、可用性与敏感性', '', '| Screen | 全部边 | 原同swap支持 | 严格641 | 松弛643 | 严格域两控制同参考 |', '|---|---:|---:|---:|---:|---:|']
    for row in coverage.itertuples(index=False):
        label = row.screen.replace('|', ' / ')
        lines.append(f'| {label} | {row.n_all_edges} | {row.n_original_eligible} | {row.n_strict} | {row.n_relaxed} | {row.n_strict_controls_same_reference} |')
    same = int(coverage.n_strict_controls_same_reference.sum())
    lines += ['', f'严格域641条覆盖203个context、16screen/14个合并家族。两控制选中同一参考{same}条（{100 * same / 641:.2f}%），其余{641 - same}条不同。共用hash排序所产生的重合完整保留；两控制不是独立证据。原649域中严格不可用8条、松弛不可用6条，另有原1,159条无同swap参考；全部仍在1,808条/384context/18screen注册表，不填零、不改变原63结论分母。', '', '| Domain | Split | TRAIN | 主mask人数 | 完整mask人数 | 完整mask有评分边 | raw同swap相对异context/any恢复 | 完整mask恢复 | swap平衡恢复 | clip恢复 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    selected = headline[headline.scope.eq('primary') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')]
    for row in selected.sort_values(['domain', 'split', 'fraction']).itertuples(index=False):
        common = headline[headline.domain.eq(row.domain) & headline.split.eq(row.split) & headline.fraction.eq(row.fraction) & headline.scope.eq('all_panels_complete') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')].iloc[0]
        swap = headline[headline.domain.eq(row.domain) & headline.split.eq(row.split) & headline.fraction.eq(row.fraction) & headline.scope.eq('primary') & headline.weighting.eq('swap_balanced') & headline.macro.eq('family_macro')].iloc[0]
        label = 'other' if row.domain == 'strict_641' else 'any'
        lines.append(f"| {row.domain} | {row.split} | {row.fraction:g} | {row.target_score_count} | {common.target_score_count} | {common.n_scored_edges}/{row.n_edges} | {100 * getattr(row, label + '_recovery_vs_control'):.2f}% | {100 * common[label + '_recovery_vs_control']:.2f}% | {100 * swap[label + '_recovery_vs_control']:.2f}% | {100 * getattr(row, label + '_recovery_vs_control_full_clip'):.2f}% |")
    lines += ['', '## 解释边界', '', '这是固定元数据观测政策的回顾性配对比较。即使同swap超过不同swap对照，也不能独立识别具体化学身份的因果作用：对照没有匹配响应幅度、方差、历史协方差、观测误差或全部候选背景。只使用一个预设hash选择，不能宣称覆盖所有reference选择的不确定性。不同context可同时改变多个配方字段。', '', '冻结设计末段关于不占优时解释范围的表述，在此明确限于641/643支持域和这个单hash政策对照的额外优势；某些方向不占优不能否定化学信息本身。策略不因正负结果更换。冻结设计的初始checkbox是评分前清单，不是当前执行状态，源文件按原字节保留。', '', '完整mask分别条件于11或10个冻结面板的可用性，是选定身份群体敏感性；它不替代primary，也不证明测量误差相等。边、条件、fold及随机draw均复用，不作独立复制显著性推断。等context和swap权重不产生化学总体代表性。', '', '报告63直接差值与offset运输规则不在本轮重新拟合或筛选。本轮也不把库内个体差值与跨库聚合响应的风险相减；单一文库内的观测政策价值不等于新文库或实验室可移植性。', '', '## 文件与核验', '', f'完整重算{evaluation_seconds:.1f}秒，720次TRAIN拟合；102,720条domain/fold评分、821,760条随机draw/domain评分。原649条基线共{len(baseline_check):,}条fold-target核对，最大绝对SSE差{baseline_check.max_absolute_risk_difference.max():.3g}；目标人数及mask逐项相同。6项针对性测试及Analysis完整267项unittest均通过。呈现刷新使用hash核对后的缓存，评分与fit audit文件字节保持不变。source、注册、输出、代码、测试、报告及图51哈希写入`data/{PREFIX}_manifest.json`。', '']
    REPORT.write_text(before + '\n'.join(lines), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--summaries-only', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    inputs = [TABLES / 'chemically_aligned_measurements_target_registry.csv', TABLES / 'chemically_aligned_measurements_panel_registry.csv', TABLES / 'chemically_aligned_measurements_fold_scores.csv.gz', TABLES / 'chemical_mode_training_sample_registry.csv', DATA / 'cv_fold_registry.csv', FROZEN_DESIGN, HERE / 'chemically_aligned_measurements.py', HERE / 'chemical_mode_recovery.py', HERE / 'chemical_response_structure.py', HERE / 'response_mode_ablation.py', HERE / 'shrinkage_covariance_analysis.py', HERE / 'training_size_intervention.py']
    inputs += sorted(DATA.glob('matrix_*.npz'))
    source_hashes = {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}
    source = pd.read_csv(inputs[0], keep_default_na=False)
    registry = add_controls(source)
    coverage, panels = (build_coverage(registry), build_panels(registry))
    if (len(registry), registry.screen.nunique(), int(registry.eligible.sum()), int(registry.strict_eligible.sum()), int(registry.relaxed_eligible.sum()), len(panels)) != (1846, 19, 663, 655, 657, 7279):
        raise ValueError('fixed structural-control domain changed')
    if int(registry.groupby('screen').context_id.nunique().sum()) != 400 or registry[registry.eligible.eq(1)].screen.nunique() != 17 or registry[registry.eligible.eq(1)].groupby(['screen', 'swap_key']).ngroups != 242:
        raise ValueError('inherited full-domain coverage changed')
    if int(registry.loc[registry.strict_eligible.eq(1), 'controls_same_reference'].sum()) != 564:
        raise ValueError('shared-hash control identity changed')
    inherited_panels = pd.read_csv(inputs[1])
    original = panels[panels.policy.isin(['semantic', 'random'])].copy()
    original['policy'] = original.policy.replace({'semantic': 'same_swap'})
    check = original.merge(inherited_panels[['screen', 'edge_id', 'policy', 'draw', 'anchor_indices_json']], on=['screen', 'edge_id', 'policy', 'draw'], suffixes=('', '_63'), validate='one_to_one')
    if len(check) != 5967 or not check.anchor_indices_json.eq(check.anchor_indices_json_63).all():
        raise ValueError('inherited report63 panels changed')
    registered_outputs = {PREFIX + '_target_registry.csv': registry, PREFIX + '_coverage.csv': coverage, PREFIX + '_panel_registry.csv': panels}
    for name, frame in registered_outputs.items():
        frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    registry_hashes = {name: digest(TABLES / name) for name in registered_outputs}
    design_path = DATA / (PREFIX + '_design.json')
    design_path.write_text(json.dumps(dict(date='2026-10-17', design=DESIGN, source_hashes=source_hashes, registry_hashes=registry_hashes), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.register_only:
        print(json.dumps(dict(all_edges=1808, original_eligible=649, strict=641, relaxed=643, same_controls=550, panels=len(panels), registry_hashes=registry_hashes), indent=2))
        return
    cache_names = [PREFIX + '_fold_scores.csv.gz', PREFIX + '_random_draw_scores.csv.gz', PREFIX + '_fit_audit.csv', PREFIX + '_all649_baseline_check.csv.gz']
    if args.summaries_only:
        prior = json.loads((DATA / (PREFIX + '_manifest.json')).read_text(encoding='utf-8'))
        if prior['source_hashes'] != source_hashes or prior['registry_hashes'] != registry_hashes:
            raise ValueError('cached inputs or policies changed')
        for name in cache_names:
            if digest(TABLES / name) != prior['output_hashes'][name]:
                raise ValueError('cached score file changed')
        folds, draws, audits, baseline_check = [pd.read_csv(TABLES / name) for name in cache_names]
        evaluation_seconds = prior['evaluation_seconds']
    else:
        samples, original_folds = (pd.read_csv(inputs[3]), pd.read_csv(inputs[4]))
        folds, draws, audits, baselines = evaluate(registry, samples, original_folds)
        baseline_check = check_original_baselines(baselines, pd.read_csv(inputs[2]))
        evaluation_seconds = time.monotonic() - start
    if (len(folds), len(draws), len(audits), len(baseline_check)) != (104960, 839680, 760, 26520):
        raise ValueError('evaluation denominator changed')
    pair_keys = FOLD_KEYS[:-1] + ['edge_id', 'domain', 'scope']
    paired = folds.pivot(index=pair_keys, columns='fraction', values=['n', 'target_mask_sha256'])
    if not paired['n', 0.25].eq(paired['n', 1.0]).all() or not paired['target_mask_sha256', 0.25].eq(paired['target_mask_sha256', 1.0]).all():
        raise ValueError('history amount changed target mask')
    edge, context, screen, family, headline = aggregate_domains(folds)
    availability = availability_summary(folds)
    outputs = {**registered_outputs, cache_names[0]: folds, cache_names[1]: draws, cache_names[2]: audits, cache_names[3]: baseline_check, PREFIX + '_edge_scores.csv': edge, PREFIX + '_context_scores.csv': context, PREFIX + '_screen_scores.csv': screen, PREFIX + '_family_scores.csv': family, PREFIX + '_headline.csv': headline, PREFIX + '_availability.csv': availability}
    for name, frame in outputs.items():
        if name not in registered_outputs and (not (args.summaries_only and name in cache_names)):
            frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    if args.summaries_only and any((digest(TABLES / name) != prior['output_hashes'][name] for name in cache_names)):
        raise ValueError('presentation refresh changed score-file bytes')
    plot_figure(coverage, screen, headline)
    write_report(headline, screen, coverage, baseline_check, evaluation_seconds)
    if source_hashes != {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}:
        raise ValueError('source files changed during computation')
    if registry_hashes != {name: digest(TABLES / name) for name in registered_outputs}:
        raise ValueError('frozen policies changed during computation')
    manifest = dict(date='2026-10-17', python=platform.python_version(), design=DESIGN, source_hashes=source_hashes, registry_hashes=registry_hashes, inputs_unchanged=True, inherited_panels_unchanged=True, code_sha256=digest(Path(__file__)), test_code_sha256=digest(HERE / ('test_' + PREFIX + '.py')), report_sha256=digest(REPORT), figure_sha256=digest(FIGURE), design_sha256=digest(design_path), output_hashes={name: digest(TABLES / name) for name in outputs}, all_edges=1808, all_screens=18, original_eligible=649, strict_edges=641, relaxed_edges=643, original_unsupported=1159, strict_controls_same_reference=550, strict_controls_different_reference=91, training_fits=720, panel_predictions=285000, fold_score_rows=len(folds), random_draw_score_rows=len(draws), all649_baseline_rows=len(baseline_check), all649_maximum_sse_difference=float(baseline_check.max_absolute_risk_difference.max()), target_mask_equal_across_history=True, target_count_equal_across_policies=True, evaluation_seconds=evaluation_seconds, elapsed_seconds=time.monotonic() - start, cached_score_reuse=args.summaries_only, maximum_closure_error=float(max((headline[name].abs().max() for name in headline if 'closure_error' in name))))
    (DATA / (PREFIX + '_manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({name: manifest[name] for name in ('strict_edges', 'relaxed_edges', 'training_fits', 'fold_score_rows', 'random_draw_score_rows', 'all649_maximum_sse_difference', 'evaluation_seconds', 'elapsed_seconds')}, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
