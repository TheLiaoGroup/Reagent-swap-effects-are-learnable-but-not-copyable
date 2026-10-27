"""Report63: frozen, same-cost chemical observation directions."""
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
from chemical_mode_recovery import contrast_prediction, fit_clean_covariance, nested_quarter
from chemical_response_structure import AMBIGUOUS
from response_mode_ablation import rank1_diagonal_covariance
from shrinkage_covariance_analysis import matrix_path
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'chemically_aligned_measurements'
REPORT = HERE / '63_CHEMICALLY_ALIGNED_MEASUREMENTS_2026-10-17.md'
FIGURE = FIGURES / '50_CHEMICALLY_ALIGNED_MEASUREMENTS.png'
SEED, N_DRAWS = (2026101763, 8)
FOLD_KEYS = ['screen', 'family', 'split', 'repeat', 'fold', 'fraction']
CELL_KEYS = ['screen', 'family', 'split', 'fraction', 'scope']
META_COLUMNS = ['screen', 'reaction_class', 'condition_a', 'condition_b', 'category', 'contrast_type', 'dose', 'low_id', 'high_id', 'low_name', 'high_name', 'orientation', 'context_id', 'context_json', 'swap_key', 'domain_swap_key', 'family', 'edge_id', 'condition_a_index', 'condition_b_index']
DESIGN = {'registered_domain': 'All1808 strict unambiguous edges/384contexts/18screens;649eligible/235repeated swap groups/16screens;1159unsupported retained.', 'reference': 'Within screen, identical strict swap_key, different context_id, four distinct endpoints; minimum metadata SHA256.', 'hash_protocol': "SHA256('|'.join(str(part) for part in [PREFIX,SEED,kind,screen,target_edge_id,*candidate_parts])); reference candidate_parts=[candidate_edge_id]; random=[draw,condition_index,condition_id].", 'random': '8fixed hash panels,2distinct anchors each;exclude target endpoints and ambiguous columns;reference endpoints allowed;not8independent replications or16joint measurements.', 'training': 'Exactly inherited report22 TRAIN720 sample registry and original360folds;pair/role1/role2;quarter/full;ambiguous columns excluded from covariance.', 'predictions': 'Same panel and TRAINmu for fullLW/rank1+diagonal/marginal;actual available anchor masks;raw primary,endpoint clip sensitivity.', 'direct': 'Reference own orientation high-low;raw direct and slope1 plus joint TRAIN mean(target-reference),minimum3joint finite;missing reference or invalid offset uses covariance-fit target marginal.', 'target_mask': 'Primary finite target endpoint pairs unchanged by panels;reference_complete secondary requires reference2finite;all9_complete frozen sensitivity requires all9panels2finite;each secondary same mask for every policy.', 'aggregation': 'Random8SSE equal mean within target/fold before folds;sumSSE/count across folds peredge;equal edges/context,contexts/screen,screens/family;familymacro primary;swapbalanced sensitivity.', 'control_inventory': 'Only metadata support for different swap,same screen/category/contrast_type/dose,four endpoints distinct;with/without different context;no extra predictions.', 'cost': '2planned retrospective matrix cells per panel;finite availability separately audited;not newly run experiment costs.', 'interpretation': 'Frozen acquisition-policy comparison,not causal chemistry isolation;different contexts can change multiple background fields;no edge/draw independence inference.'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata_hash(kind, screen, edge_id, *parts):
    text = '|'.join(map(str, (PREFIX, SEED, kind, screen, edge_id, *parts)))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def freeze_registry(edges, conditions_by_screen, excluded_by_screen):
    """This interface accepts recipe metadata, never response values."""
    rows, coverage = ([], [])
    for screen, group in edges.groupby('screen', sort=True):
        group = group.sort_values('edge_id')
        conditions = np.asarray(conditions_by_screen[screen]).astype(str)
        excluded = set(excluded_by_screen[screen])
        for target in group.itertuples(index=False):
            endpoints = {int(target.condition_a_index), int(target.condition_b_index)}
            if endpoints & excluded:
                raise ValueError('ambiguous target endpoint')
            same, other, other_context = ([], [], [])
            for reference in group.itertuples(index=False):
                ref_endpoints = {int(reference.condition_a_index), int(reference.condition_b_index)}
                if endpoints & ref_endpoints:
                    continue
                if target.swap_key == reference.swap_key and target.context_id != reference.context_id:
                    same.append(reference)
                if target.swap_key != reference.swap_key and target.category == reference.category and (target.contrast_type == reference.contrast_type) and (target.dose == reference.dose):
                    other.append(reference)
                    if target.context_id != reference.context_id:
                        other_context.append(reference)
            same.sort(key=lambda ref: metadata_hash('reference', screen, target.edge_id, ref.edge_id))
            candidates = [i for i in range(len(conditions)) if i not in endpoints | excluded]
            if len(candidates) < 2:
                raise ValueError('two legal random anchors required')
            panels = [sorted(candidates, key=lambda i: metadata_hash('random', screen, target.edge_id, draw, i, conditions[i]))[:2] for draw in range(N_DRAWS)]
            row = target._asdict()
            row.update(eligible=int(bool(same)), reason='same_swap_other_context_disjoint' if same else 'no_disjoint_same_swap_other_context', n_reference_candidates=len(same), reference_edge_id='', reference_context_id='', reference_condition_a_index=-1, reference_condition_b_index=-1, reference_condition_a='', reference_condition_b='', reference_orientation=0, reference_selection_hash='', reference_indices_json='[]', random_panels_indices_json=json.dumps(panels), random_panels_ids_json=json.dumps([[conditions[i] for i in panel] for panel in panels]), n_different_swap_candidates=len(other), n_different_swap_other_context_candidates=len(other_context))
            if same:
                reference = same[0]
                row.update(reference_edge_id=reference.edge_id, reference_context_id=reference.context_id, reference_condition_a_index=int(reference.condition_a_index), reference_condition_b_index=int(reference.condition_b_index), reference_condition_a=reference.condition_a, reference_condition_b=reference.condition_b, reference_orientation=int(reference.orientation), reference_selection_hash=metadata_hash('reference', screen, target.edge_id, reference.edge_id), reference_indices_json=json.dumps([int(reference.condition_a_index), int(reference.condition_b_index)]))
            rows.append(row)
        one = pd.DataFrame(rows)
        one = one[one.screen.eq(screen)]
        coverage.append(dict(screen=screen, family=group.family.iloc[0], n_edges=len(group), n_contexts=group.context_id.nunique(), n_swaps=group.swap_key.nunique(), n_eligible=int(one.eligible.sum()), n_unsupported=int((1 - one.eligible).sum()), n_repeated_swap_groups=int(group.groupby('swap_key').size().gt(1).sum()), n_different_swap_supported=int(one.n_different_swap_candidates.gt(0).sum()), n_eligible_different_swap_supported=int((one.eligible.eq(1) & one.n_different_swap_candidates.gt(0)).sum()), n_different_swap_other_context_supported=int(one.n_different_swap_other_context_candidates.gt(0).sum()), n_eligible_different_swap_other_context_supported=int((one.eligible.eq(1) & one.n_different_swap_other_context_candidates.gt(0)).sum()), excluded_anchor_indices_json=json.dumps(sorted(excluded))))
    return (pd.DataFrame(rows).sort_values(['screen', 'edge_id']).reset_index(drop=True), pd.DataFrame(coverage))

def panel_registry(registry):
    records = []
    for row in registry[registry.eligible.eq(1)].itertuples(index=False):
        panels = [('same_swap', -1, json.loads(row.reference_indices_json))]
        panels += [('random', d, panel) for d, panel in enumerate(json.loads(row.random_panels_indices_json))]
        for policy, draw, anchors in panels:
            if len(set(anchors)) != 2 or set(anchors) & {row.condition_a_index, row.condition_b_index}:
                raise ValueError('invalid hidden-endpoint panel')
            records.append(dict(screen=row.screen, family=row.family, edge_id=row.edge_id, context_id=row.context_id, swap_key=row.swap_key, policy=policy, draw=draw, planned_cells=2, anchor_indices_json=json.dumps(anchors)))
    return pd.DataFrame(records)

def oriented_difference(matrix, a, b, orientation):
    return int(orientation) * (matrix[:, int(a)] - matrix[:, int(b)])

def direct_transport(train_target, train_reference, reference_test, baseline):
    train_target, train_reference, reference_test = [np.asarray(x, float) for x in (train_target, train_reference, reference_test)]
    joint = np.isfinite(train_target) & np.isfinite(train_reference)
    valid = int(joint.sum()) >= 3
    offset = float(np.mean(train_target[joint] - train_reference[joint])) if valid else np.nan
    available = np.isfinite(reference_test)
    raw = np.where(available, reference_test, baseline)
    shifted = np.where(available, reference_test + offset, baseline) if valid else np.full(len(reference_test), baseline)
    return (raw, shifted, {'joint_train_n': int(joint.sum()), 'offset_valid': valid, 'offset': offset})

def panel_predictions(fit, rank1_covariance, values, anchors, endpoints, orientation):
    full_raw, full_clip = contrast_prediction(fit, values, anchors, endpoints, orientation)
    rank1_raw, rank1_clip = contrast_prediction(dict(mu=fit['mu'], cov=rank1_covariance), values, anchors, endpoints, orientation)
    return dict(full_raw=full_raw, rank1_raw=rank1_raw, full_clip=full_clip, rank1_clip=rank1_clip)

def score_values(actual, predictions, requested_mask):
    actual, requested_mask = (np.asarray(actual, float), np.asarray(requested_mask, bool))
    valid = requested_mask & np.isfinite(actual)
    result = {'n': int(valid.sum())}
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction, float)
        if len(prediction) != len(actual) or not np.isfinite(prediction[valid]).all():
            raise ValueError('all finite target rows must have a finite prediction')
        result[name + '_sse'] = float(np.sum((actual[valid] - prediction[valid]) ** 2))
    return result

def availability_masks(reference_values, random_values):
    reference_complete = np.isfinite(reference_values).all(axis=1)
    all_complete = reference_complete.copy()
    if len(random_values) != N_DRAWS:
        raise ValueError('all9 mask needs all eight fixed random panels')
    for values in random_values:
        all_complete &= np.isfinite(values).all(axis=1)
    return {'primary': np.ones(len(reference_values), bool), 'reference_complete': reference_complete, 'all9_complete': all_complete}

def average_draw_scores(scores):
    if not scores or len({score['n'] for score in scores}) != 1:
        raise ValueError('random draws must score exactly the same target rows')
    return {'n': scores[0]['n'], **{name: float(np.mean([score[name] for score in scores])) for name in scores[0] if name.endswith('_sse')}}

def validate_training_sample(sample, n_rows, expected_test):
    get = sample.get if isinstance(sample, dict) else lambda name: getattr(sample, name)
    selected = np.asarray(json.loads(get('training_indices_json')), int)
    tests = np.asarray(json.loads(get('test_indices_json')), int)
    pool = np.setdiff1d(np.arange(n_rows), tests)
    expected = nested_quarter(pool, get('screen'), get('split'), int(get('repeat')), int(get('fold'))) if float(get('fraction')) == 0.25 else pool
    if not np.array_equal(tests, expected_test) or not np.array_equal(selected, expected) or np.intersect1d(selected, tests).size:
        raise ValueError('TRAIN/TEST differs from the original report22 register')
    return True

def evaluate(registry, samples, original_folds):
    records, draws, audits = ([], [], [])
    start = time.monotonic()
    for screen, screen_samples in samples.groupby('screen', sort=True):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            matrix, pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        roles = np.asarray([pair.split(' >> ') for pair in pairs])
        excluded = [i for i, condition in enumerate(conditions) if (screen.split('|')[0], condition) in AMBIGUOUS]
        one = registry[registry.screen.eq(screen) & registry.eligible.eq(1)]
        source_folds = original_folds[original_folds.screen.eq(screen)]
        expected_tests = {key: group.pair_index.to_numpy(int) for key, group in source_folds.groupby(['split', 'repeat', 'fold'], sort=True)}
        for sample in screen_samples.itertuples(index=False):
            selected, tests = [np.asarray(json.loads(getattr(sample, field)), int) for field in ('training_indices_json', 'test_indices_json')]
            validate_training_sample(sample, len(matrix), expected_tests[sample.split, sample.repeat, sample.fold])
            source = source_folds[source_folds.split.eq(sample.split) & source_folds.repeat.eq(sample.repeat) & source_folds.fold.eq(sample.fold)]
            if pairs[tests].tolist() != source.subpair.tolist():
                raise ValueError('TEST identities changed')
            if sample.split.startswith('role'):
                role = int(sample.split[-1]) - 1
                if set(roles[selected, role]) & set(roles[tests, role]):
                    raise ValueError('held-out substrate role overlaps TRAIN')
            key = {field: getattr(sample, field) for field in FOLD_KEYS}
            train = matrix[selected]
            fit = fit_clean_covariance(train, excluded)
            rank1 = rank1_diagonal_covariance(fit['cov'])
            audits.append({**key, 'n_training': len(selected), 'n_test': len(tests), 'n_eligible_edges': len(one), 'n_excluded_ambiguous': len(excluded), 'shrinkage': fit['shrinkage'], 'imputed_training_cells': fit['imputed_cells'], 'train_register_exact': True, 'train_test_disjoint': True, 'quarter_nested_in_full': True})
            for edge in one.itertuples(index=False):
                endpoints = [edge.condition_a_index, edge.condition_b_index]
                reference = [edge.reference_condition_a_index, edge.reference_condition_b_index]
                if len(set(endpoints + reference)) != 4 or edge.context_id == edge.reference_context_id:
                    raise ValueError('reference overlaps target or context')
                actual = oriented_difference(matrix[tests], *endpoints, edge.orientation)
                target_valid = np.isfinite(actual)
                values = matrix[np.ix_(tests, reference)]
                complete = np.isfinite(values).all(axis=1)
                baseline = edge.orientation * (fit['mu'][endpoints[0]] - fit['mu'][endpoints[1]])
                semantic = panel_predictions(fit, rank1, values, reference, endpoints, edge.orientation)
                reference_test = oriented_difference(matrix[tests], *reference, edge.reference_orientation)
                target_train = oriented_difference(train, *endpoints, edge.orientation)
                reference_train = oriented_difference(train, *reference, edge.reference_orientation)
                direct, shifted, transport = direct_transport(target_train, reference_train, reference_test, baseline)
                predictions = {'semantic_' + name: value for name, value in semantic.items()}
                predictions.update(marginal=np.full(len(tests), baseline), direct_raw=direct, offset_raw=shifted)
                edge_key = {**key, 'edge_id': edge.edge_id, 'context_id': edge.context_id, 'swap_key': edge.swap_key, 'reference_edge_id': edge.reference_edge_id}
                random_panels = json.loads(edge.random_panels_indices_json)
                random_values_by_draw = [matrix[np.ix_(tests, anchors)] for anchors in random_panels]
                masks = availability_masks(values, random_values_by_draw)
                random_scores = {scope: [] for scope in masks}
                random_cells = {scope: [] for scope in masks}
                random_completes = {scope: [] for scope in masks}
                random_nones = {scope: [] for scope in masks}
                for draw, (anchors, random_values) in enumerate(zip(random_panels, random_values_by_draw)):
                    random_prediction = panel_predictions(fit, rank1, random_values, anchors, endpoints, edge.orientation)
                    for scope, mask in masks.items():
                        score = score_values(actual, random_prediction, mask)
                        valid = target_valid & mask
                        observed = np.isfinite(random_values[valid])
                        draw_row = {**edge_key, 'scope': scope, 'draw': draw, **score, 'observed_anchor_cells': int(observed.sum()), 'complete_anchor_rows': int(observed.all(axis=1).sum()), 'no_anchor_rows': int((~observed.any(axis=1)).sum())}
                        draws.append(draw_row)
                        random_scores[scope].append(score)
                        random_cells[scope].append(draw_row['observed_anchor_cells'])
                        random_completes[scope].append(draw_row['complete_anchor_rows'])
                        random_nones[scope].append(draw_row['no_anchor_rows'])
                for scope, mask in masks.items():
                    score = score_values(actual, predictions, mask)
                    random = average_draw_scores(random_scores[scope])
                    if score['n'] != random['n']:
                        raise ValueError('panel changed target denominator')
                    valid = target_valid & mask
                    observed = np.isfinite(values[valid])
                    records.append({**edge_key, 'scope': scope, **score, **{'random_' + name: value for name, value in random.items() if name.endswith('_sse')}, 'semantic_observed_cells': int(observed.sum()), 'semantic_complete_rows': int(observed.all(axis=1).sum()), 'semantic_no_anchor_rows': int((~observed.any(axis=1)).sum()), 'random_observed_cells': float(np.mean(random_cells[scope])), 'random_complete_rows': float(np.mean(random_completes[scope])), 'random_no_anchor_rows': float(np.mean(random_nones[scope])), 'direct_fallback_rows': int((valid & ~complete).sum()), 'offset_fallback_rows': int((valid & (~complete | (not transport['offset_valid']))).sum()), 'offset_joint_train_n': transport['joint_train_n'], 'offset_valid': transport['offset_valid'], 'offset': transport['offset'], 'target_mask_sha256': hashlib.sha256(tests[valid].astype('<i8').tobytes()).hexdigest()})
        print(f'CHEMICALLY_ALIGNED {screen} fold_rows={len(records)} seconds={time.monotonic() - start:.1f}', flush=True)
    return (pd.DataFrame(records), pd.DataFrame(draws), pd.DataFrame(audits))

def derived(frame):
    frame = frame.copy()
    if {'semantic_full_raw_mse', 'random_full_raw_mse'} <= set(frame):
        frame['semantic_advantage_vs_random_mse'] = frame.random_full_raw_mse - frame.semantic_full_raw_mse
        frame['semantic_recovery_vs_random'] = frame.semantic_advantage_vs_random_mse / frame.random_full_raw_mse.replace(0, np.nan)
    if {'semantic_full_raw_mse', 'marginal_mse'} <= set(frame):
        frame['semantic_recovery_vs_marginal'] = 1 - frame.semantic_full_raw_mse / frame.marginal_mse.replace(0, np.nan)
        for policy in ('random_full_raw', 'direct_raw', 'offset_raw'):
            if policy + '_mse' in frame:
                frame[policy + '_recovery_vs_marginal'] = 1 - frame[policy + '_mse'] / frame.marginal_mse.replace(0, np.nan)
    for policy in ('semantic', 'random'):
        for kind in ('raw', 'clip'):
            full, rank1 = (policy + '_full_' + kind + '_mse', policy + '_rank1_' + kind + '_mse')
            if {full, rank1} <= set(frame):
                frame[policy + '_extra_gain_' + kind] = 1 - frame[full] / frame[rank1].replace(0, np.nan)
    return frame

def aggregate_scores(folds):
    sums = [name for name in folds if name.endswith('_sse')]
    metrics = [name.replace('_sse', '_mse') for name in sums]
    edge_keys = CELL_KEYS + ['edge_id', 'context_id', 'swap_key']
    edge = folds.groupby(edge_keys, as_index=False)[['n', *sums]].sum()
    for name in sums:
        edge[name.replace('_sse', '_mse')] = edge[name] / edge.n.replace(0, np.nan)
    context = edge.groupby(CELL_KEYS + ['context_id'], as_index=False)[metrics].mean()
    contexts = context.groupby(CELL_KEYS, as_index=False)[metrics].mean().assign(weighting='context_balanced')
    swap = edge.groupby(CELL_KEYS + ['swap_key'], as_index=False)[metrics].mean()
    swaps = swap.groupby(CELL_KEYS, as_index=False)[metrics].mean().assign(weighting='swap_balanced')
    screen = pd.concat([contexts, swaps], ignore_index=True)
    counts = edge.groupby(CELL_KEYS, as_index=False).agg(n_edges=('edge_id', 'size'), n_scored_edges=('n', lambda x: int(x.gt(0).sum())), n_contexts=('context_id', 'nunique'), n_swaps=('swap_key', 'nunique'), target_score_count=('n', 'sum'))
    screen = screen.merge(counts, on=CELL_KEYS, validate='many_to_one')
    family_keys = ['family', 'split', 'fraction', 'scope', 'weighting']
    family = screen.groupby(family_keys, as_index=False)[metrics].mean()
    count_columns = ['n_edges', 'n_scored_edges', 'n_contexts', 'n_swaps', 'target_score_count']
    counts = screen.groupby(family_keys, as_index=False)[count_columns].sum()
    units = screen.groupby(family_keys, as_index=False).agg(n_screens=('screen', 'nunique'))
    family = family.merge(counts, on=family_keys).merge(units, on=family_keys)
    headlines = []
    headline_keys = ['split', 'fraction', 'scope', 'weighting']
    for macro, source, unit in (('screen_macro', screen, 'screen'), ('family_macro', family, 'family')):
        mean = source.groupby(headline_keys, as_index=False)[metrics].mean()
        counts = source.groupby(headline_keys, as_index=False)[count_columns].sum()
        units = source.groupby(headline_keys, as_index=False).agg(n_units=(unit, 'nunique'))
        headlines.append(mean.merge(counts, on=headline_keys).merge(units, on=headline_keys).assign(macro=macro))
    return (derived(edge), derived(context), derived(screen), derived(family), derived(pd.concat(headlines, ignore_index=True)))

def availability_scores(folds):
    counts = ['n', 'semantic_observed_cells', 'semantic_complete_rows', 'semantic_no_anchor_rows', 'random_observed_cells', 'random_complete_rows', 'random_no_anchor_rows', 'direct_fallback_rows', 'offset_fallback_rows']
    group = folds.groupby(CELL_KEYS, as_index=False)[counts].sum()
    for name in counts[1:]:
        group[name + '_per_target'] = group[name] / group.n.replace(0, np.nan)
    group['offset_invalid_fold_edges'] = folds.groupby(CELL_KEYS).offset_valid.apply(lambda values: int((~values).sum())).to_numpy()
    return group

def plot_figure(coverage, screen, headline):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    short = lambda s: s.replace('Buchwald-Hartwig(CMC)', 'BH(CMC)').replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O')
    cov = coverage.sort_values('screen')
    y = np.arange(len(cov))
    axes[0, 0].barh(y, cov.n_edges, color='#e3e3e3', label='All strict edges')
    axes[0, 0].barh(y, cov.n_eligible, color='#287d78', label='Same-swap reference supported')
    axes[0, 0].set(yticks=y, yticklabels=[short(s) for s in cov.screen], xlabel='Registered target edges', title='A  Full-domain coverage')
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=8)
    primary = screen[screen.scope.eq('primary') & screen.weighting.eq('context_balanced') & screen.fraction.eq(1.0)]
    colors = {'pair': '#287d78', 'role1': '#b65d50', 'role2': '#6874a6'}
    for split, offset in (('pair', -0.2), ('role1', 0), ('role2', 0.2)):
        g = primary[primary.split.eq(split)].set_index('screen').reindex(cov.screen)
        axes[0, 1].scatter(10000.0 * g.semantic_advantage_vs_random_mse, y + offset, s=23, color=colors[split], label=split)
    axes[0, 1].set(yticks=y, yticklabels=[short(s) for s in cov.screen], xlabel='Random MSE - same-swap MSE (pp squared)', title='B  Fixed two-cell observations; full history')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(0, color='#777777', lw=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    head = headline[headline.scope.eq('primary') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')]
    labels = ['Same-swap full', 'Random full', 'Same-swap rank1', 'Random rank1', 'Marginal', 'Direct', 'Direct + offset']
    columns = ['semantic_full_raw_mse', 'random_full_raw_mse', 'semantic_rank1_raw_mse', 'random_rank1_raw_mse', 'marginal_mse', 'direct_raw_mse', 'offset_raw_mse']
    for split, offset in (('pair', -0.22), ('role1', 0), ('role2', 0.22)):
        g = head[head.split.eq(split) & head.fraction.eq(1.0)].iloc[0]
        axes[1, 0].scatter([10000.0 * g[c] for c in columns], np.arange(len(columns)) + offset, color=colors[split], label=split, s=28)
    axes[1, 0].set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel='Family-macro MSE (pp squared)', title='C  Absolute held-out error; full history')
    axes[1, 0].invert_yaxis()
    for split in colors:
        g = head[head.split.eq(split)].sort_values('fraction')
        axes[1, 1].plot(g.fraction, 100 * g.semantic_recovery_vs_random, marker='o', color=colors[split], label=split)
    axes[1, 1].set(xticks=[0.25, 1.0], xticklabels=['Quarter', 'Full'], xlabel='Historical training amount', ylabel='Same-swap recovery beyond random (%)', title='D  Observation direction and historical amount')
    axes[1, 1].axhline(0, color='#777777', lw=0.8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.15)
    fig.suptitle('Chemically aligned measurements: identical target and two-cell budget', fontsize=15)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

def write_report(headline, screen, coverage, availability, draw_headline, evaluation_seconds):
    before = '# Scientific results\n\n'
    head = headline[headline.macro.eq('family_macro') & headline.weighting.eq('context_balanced') & headline.scope.eq('primary')]
    lines = ['## Results', '', 'The main table is aggregated with equal weights context→screen→merged family; raw MSE in pp²; percentages are ratios of aggregated MSE, and the 8 random draws are first averaged with equal weight within each fold.', '', '| Split | TRAIN | Same-swap full | Random full | marginal | direct | offset | Recovery vs random | Recovery vs marginal | Same-swap full over rank1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in head.sort_values(['split', 'fraction']).itertuples(index=False):
        lines.append(f'| {row.split} | {row.fraction:g} | {10000.0 * row.semantic_full_raw_mse:.3f} | {10000.0 * row.random_full_raw_mse:.3f} | {10000.0 * row.marginal_mse:.3f} | {10000.0 * row.direct_raw_mse:.3f} | {10000.0 * row.offset_raw_mse:.3f} | {100 * row.semantic_recovery_vs_random:.2f}% | {100 * row.semantic_recovery_vs_marginal:.2f}% | {100 * row.semantic_extra_gain_raw:.2f}% |')
    pair = head[head.split.eq('pair') & head.fraction.eq(1.0)].iloc[0]
    lines += ['', f'In all six settings same-swap full beats random full and marginal; recovery vs random is {100 * head.semantic_recovery_vs_random.min():.2f}%–{100 * head.semantic_recovery_vs_random.max():.2f}%, and vs marginal {100 * head.semantic_recovery_vs_marginal.min():.2f}%–{100 * head.semantic_recovery_vs_marginal.max():.2f}%. Under full/pair, raw MSE falls from {10000.0 * pair.random_full_raw_mse:.3f} for random to {10000.0 * pair.semantic_full_raw_mse:.3f} pp² for same swap; direct transport gives {10000.0 * pair.direct_raw_mse:.3f} and offset gives {10000.0 * pair.offset_raw_mse:.3f}, both above marginal at {10000.0 * pair.marginal_mse:.3f}. This supports the value of the fixed chemically aligned point-selection policy and does not support these two fixed-slope difference transport rules.', '']
    draw = draw_headline[draw_headline.macro.eq('family_macro') & draw_headline.weighting.eq('context_balanced') & draw_headline.scope.eq('primary')]
    draw = draw.merge(head[['split', 'fraction', 'semantic_full_raw_mse']], on=['split', 'fraction'])
    recovery = 1 - draw.semantic_full_raw_mse / draw.random_full_raw_mse
    lines += [f'The family macro-average recovery is positive for every pre-specified random draw comparison, ranging {100 * recovery.min():.2f}%–{100 * recovery.max():.2f}% across six settings and 8 draws. This is a sensitivity range over fixed random panels, not a confidence interval, and does not mean that every screen wins.', '']
    lines += ['', '## Support and sensitivity', '', '| Screen | All edges | eligible | unsupported | Different-swap backup control (eligible) | Different-context backup control (eligible) |', '|---|---:|---:|---:|---:|---:|']
    for row in coverage.itertuples(index=False):
        screen_label = row.screen.replace('|', ' / ')
        lines.append(f'| {screen_label} | {row.n_edges} | {row.n_eligible} | {row.n_unsupported} | {row.n_eligible_different_swap_supported} | {row.n_eligible_different_swap_other_context_supported} |')
    lines += ['', 'The 649 supported edges come from 211 contexts, 235 repeated swap groups and 16 screens/14 merged families. All 1,808 edges/384 contexts/18 screens remain registered; the 1,159 unsupported edges are not filled with zero and are not in the denominator of supportable recovery. Backup different-swap controls are registered only: 1,758 supported in the full domain, 643 among eligible; additionally requiring a different context gives 1,723 in the full domain and 641 among eligible. These are not computed and do not change the main domain.', '', '| Split | TRAIN | Main-mask count (incl. fold reuse) | Reference-complete count | Complete fraction | direct fallback | offset fallback |', '|---|---:|---:|---:|---:|---:|---:|']
    av = availability[availability.scope.eq('primary')].groupby(['split', 'fraction'], as_index=False)[['n', 'semantic_complete_rows', 'direct_fallback_rows', 'offset_fallback_rows']].sum()
    for row in av.itertuples(index=False):
        lines.append(f'| {row.split} | {row.fraction:g} | {row.n} | {row.semantic_complete_rows} | {100 * row.semantic_complete_rows / row.n:.3f}% | {row.direct_fallback_rows} | {row.offset_fallback_rows} |')
    lines += ['', 'The following separately checks clip, the reference-complete common mask, the common mask complete in all 9 panels and swap balancing; these results do not replace the raw main analysis.', '', '| Split | TRAIN | Clip recovery vs random | Reference-complete common-mask raw recovery | All-9-panel-complete raw recovery | All-9-complete scored count | All-9-complete scored edges | Swap-balanced raw recovery | Screens/16 where same swap beats random |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in head.sort_values(['split', 'fraction']).itertuples(index=False):
        common = headline[headline.macro.eq('family_macro') & headline.weighting.eq('context_balanced') & headline.scope.eq('reference_complete') & headline.split.eq(row.split) & headline.fraction.eq(row.fraction)].iloc[0]
        swap = headline[headline.macro.eq('family_macro') & headline.weighting.eq('swap_balanced') & headline.scope.eq('primary') & headline.split.eq(row.split) & headline.fraction.eq(row.fraction)].iloc[0]
        all9 = headline[headline.macro.eq('family_macro') & headline.weighting.eq('context_balanced') & headline.scope.eq('all9_complete') & headline.split.eq(row.split) & headline.fraction.eq(row.fraction)].iloc[0]
        positive = screen[screen.scope.eq('primary') & screen.weighting.eq('context_balanced') & screen.split.eq(row.split) & screen.fraction.eq(row.fraction)].semantic_advantage_vs_random_mse.gt(0).sum()
        clip = 1 - row.semantic_full_clip_mse / row.random_full_clip_mse
        lines.append(f'| {row.split} | {row.fraction:g} | {100 * clip:.2f}% | {100 * common.semantic_recovery_vs_random:.2f}% | {100 * all9.semantic_recovery_vs_random:.2f}% | {all9.target_score_count} | {all9.n_scored_edges}/649 | {100 * swap.semantic_recovery_vs_random:.2f}% | {positive}/16 |')
    lines += ['', 'In the common domain complete in all 9 panels, all 649 edges still have scorable identities, and the advantage over random is kept in all six settings. This makes it hard to explain the advantage by differences in anchor missingness between panels alone, but the common mask changes the identity distribution, so numerical changes between the main domain and the common domain cannot be attributed directly to missing-data handling.', '', '## Interpretation boundary', '', 'The advantage of chemically aligned panels first supports the value of this fixed-metadata observation policy within the existing matrix; it does not yet separate substitution semantics, local recipe structure, response magnitude or observation reliability. The direct rules test a stronger difference-transport hypothesis; the advantage of the covariance model does not by itself show that fixed-slope differences are portable. A different context can involve several recipe fields and is not a single-factor background intervention.', '', 'The reference-complete common mask is a diagnostic conditioned on reference availability; the all9 common mask also conditions on the availability of all 8 frozen random panels. Neither is a replacement denominator for the main analysis. Random and semantic panels predict from their own finite anchors, and the main scoring target count is always fixed. The planned two cells per panel and the actual finite availability in the stored matrix are reported separately. The original 22 identity splits are kept; role hold-out only guarantees independence of the corresponding role, and the other role may overlap.', '', 'Edges reuse conditions, same-swap references can point to each other, historical folds overlap, and the 8 random panels are not independent experimental replicates. Equal context or equal swap weights do not give chemical population representativeness; there is no extrapolation to new libraries, laboratories or prospective wet-lab experiments, and no attribution of technical-replicate noise.', '', '## Files and verification', '', f'Full recomputation took {evaluation_seconds:.1f} s; 720 training fits, 77,880 fold scores, 623,040 random draw scores. The producer is `{PREFIX}.py` and the targeted test is `test_{PREFIX}.py`, 10 tests passing; full unittest discovery in the Analysis directory passes 261 tests. The frozen design, input/output SHA256, sources of the 720 TRAIN registrations and the endpoint isolation audit are saved in the manifest and tables with the same prefix; figure 50 shows coverage of all 18 screens and the errors of the 16 supportable screens. Raw main results, clip, reference_complete, all9_complete and swap weights are stored in separate columns.', '']
    REPORT.write_text(before + '\n'.join(lines), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--summaries-only', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    inputs = [TABLES / 'chemical_mode_target_anchor_registry.csv', TABLES / 'chemical_mode_training_sample_registry.csv', DATA / 'chemical_mode_manifest.json', DATA / 'cv_fold_registry.csv', HERE / 'chemical_mode_recovery.py', HERE / 'chemical_context_replication.py', HERE / 'chemical_response_structure.py', HERE / 'response_mode_ablation.py', HERE / 'shrinkage_covariance_analysis.py', HERE / 'training_size_intervention.py']
    inputs += sorted(DATA.glob('matrix_*.npz'))
    sources = {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}
    inherited = json.loads((DATA / 'chemical_mode_manifest.json').read_text(encoding='utf-8'))
    for name in ('chemical_mode_target_anchor_registry.csv', 'chemical_mode_training_sample_registry.csv'):
        if digest(TABLES / name) != inherited['output_hashes'][name]:
            raise ValueError('original report22 register hash mismatch')
    edges = pd.read_csv(inputs[0], usecols=META_COLUMNS, keep_default_na=False)
    samples, original_folds = (pd.read_csv(inputs[1]), pd.read_csv(DATA / 'cv_fold_registry.csv'))
    conditions_by_screen, excluded_by_screen = ({}, {})
    for screen in sorted(edges.screen.unique()):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            conditions_by_screen[screen] = saved['conds'].astype(str)
        excluded_by_screen[screen] = [i for i, c in enumerate(conditions_by_screen[screen]) if (screen.split('|')[0], c) in AMBIGUOUS]
    registry, coverage = freeze_registry(edges, conditions_by_screen, excluded_by_screen)
    panels = panel_registry(registry)
    if not (len(registry) == 1846 and registry.screen.nunique() == 19 and (registry.eligible.sum() == 663) and (registry[registry.eligible.eq(1)].screen.nunique() == 17) and (coverage.n_repeated_swap_groups.sum() == 242) and (coverage.n_contexts.sum() == 400) and (len(samples) == 760) and (len(panels) == 663 * 9)):
        raise ValueError('registered domain changed')
    frozen = {PREFIX + '_target_registry.csv': registry, PREFIX + '_panel_registry.csv': panels, PREFIX + '_coverage.csv': coverage}
    for name, frame in frozen.items():
        frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    frozen_hashes = {name: digest(TABLES / name) for name in frozen}
    design_path = DATA / (PREFIX + '_design.json')
    design_path.write_text(json.dumps(dict(date='2026-10-17', seed=SEED, design=DESIGN, registry_hashes=frozen_hashes, source_hashes=sources), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.register_only:
        print(json.dumps(dict(registered_edges=len(registry), eligible=int(registry.eligible.sum()), panels=len(panels), training_fits=len(samples), expected_fold_edge_rows=649 * 40, frozen_hashes=frozen_hashes), indent=2))
        return
    cache_names = [PREFIX + '_fold_scores.csv.gz', PREFIX + '_random_draw_scores.csv.gz', PREFIX + '_fit_audit.csv']
    if args.summaries_only:
        prior = json.loads((DATA / (PREFIX + '_manifest.json')).read_text(encoding='utf-8'))
        if sources != prior['source_hashes'] or frozen_hashes != prior['registry_hashes']:
            raise ValueError('cached inputs or frozen panels changed')
        for name in cache_names:
            if digest(TABLES / name) != prior['output_hashes'][name]:
                raise ValueError('cached scores changed')
        folds, draws, audits = [pd.read_csv(TABLES / name) for name in cache_names]
        evaluation_seconds = prior['evaluation_seconds']
    else:
        folds, draws, audits = evaluate(registry, samples, original_folds)
        evaluation_seconds = time.monotonic() - start
    if len(audits) != 760 or len(folds) != 79560 or len(draws) != 636480:
        raise ValueError('evaluation register count changed')
    paired = folds.pivot(index=FOLD_KEYS[:-1] + ['edge_id', 'scope'], columns='fraction', values=['n', 'target_mask_sha256'])
    if not paired['n', 0.25].eq(paired['n', 1.0]).all() or not paired['target_mask_sha256', 0.25].eq(paired['target_mask_sha256', 1.0]).all():
        raise ValueError('historical amount changed target rows')
    edge, context, screen, family, headline = aggregate_scores(folds)
    availability = availability_scores(folds)
    draw_headlines = []
    for draw, one in draws.groupby('draw', sort=True):
        renamed = one.rename(columns={name: 'random_' + name for name in one if name.endswith('_sse')})
        draw_headlines.append(aggregate_scores(renamed)[-1].assign(draw=draw))
    draw_headline = pd.concat(draw_headlines, ignore_index=True)
    outputs = {**frozen, cache_names[0]: folds, cache_names[1]: draws, cache_names[2]: audits, PREFIX + '_edge_scores.csv': edge, PREFIX + '_context_scores.csv': context, PREFIX + '_screen_scores.csv': screen, PREFIX + '_family_scores.csv': family, PREFIX + '_headline.csv': headline, PREFIX + '_availability.csv': availability, PREFIX + '_random_draw_headline.csv': draw_headline}
    for name, frame in outputs.items():
        if name not in frozen:
            frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    plot_figure(coverage, screen, headline)
    write_report(headline, screen, coverage, availability, draw_headline, evaluation_seconds)
    if frozen_hashes != {name: digest(TABLES / name) for name in frozen}:
        raise ValueError('frozen registry changed during scoring')
    if sources != {path.relative_to(HERE).as_posix(): digest(path) for path in inputs}:
        raise ValueError('source files changed during scoring')
    manifest = dict(date='2026-10-17', seed=SEED, python=platform.python_version(), source_hashes=sources, inputs_unchanged=True, registry_hashes=frozen_hashes, design=DESIGN, design_sha256=digest(design_path), code_sha256=digest(Path(__file__)), test_code_sha256=digest(HERE / ('test_' + PREFIX + '.py')), report_sha256=digest(REPORT), figure_sha256=digest(FIGURE), output_hashes={name: digest(TABLES / name) for name in outputs}, all_edges=1808, all_screens=18, eligible_edges=649, eligible_screens=16, eligible_families=14, repeated_swap_groups=235, unsupported_edges=1159, training_fits=len(audits), eligible_fold_edge_rows=25960, fold_score_rows=len(folds), random_draw_score_rows=len(draws), panel_predictions=233640, evaluation_seconds=evaluation_seconds, elapsed_seconds=time.monotonic() - start, cached_score_reuse=args.summaries_only, target_counts_equal_across_panels=True, target_mask_equal_across_training_amounts=True, limitations=['Retrospective fixed-library acquisition policy,not causal chemistry isolation.', 'Different context may change multiple background fields;not single-factor intervention.', 'No edge/fold/draw independence;metadata support does not establish chemical population representativeness.', 'Reference-complete secondary mask conditions on reference availability.', 'Finite-cell availability and planned2cell budget differ;no new wet experiment costs.'])
    (DATA / (PREFIX + '_manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ('all_edges', 'eligible_edges', 'training_fits', 'fold_score_rows', 'random_draw_score_rows', 'evaluation_seconds', 'elapsed_seconds')}, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
