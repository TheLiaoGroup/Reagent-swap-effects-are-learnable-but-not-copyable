"""Training-only modes and held-out strict chemical-contrast recovery.

All eligible edges are scored. Each edge has fixed outcome-independent anchors;
neither endpoint can be an anchor. Only frozen Analysis matrices are read."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from chemical_context_replication import split_half
from chemical_response_structure import AMBIGUOUS, sha256
from response_mode_ablation import rank1_diagonal_covariance
from shrinkage_covariance_analysis import fit_covariance, matrix_path
from training_size_intervention import seeded
HERE = Path(__file__).resolve().parent
DATA, TABLES, FIGURES = (HERE / 'data', HERE / 'tables', HERE / 'figures')
SEED = 2026101622
MIN_CONTEXT_HALF = 6
BUDGETS = (6, 12)
FRACTIONS = (0.25, 1.0)
PREDICTORS = ('expected_extra_gain', 'nonleading_contrast_share', 'training_repeated_context_rho', 'training_context_stable_rho', 'training_context_half_rho_gap', 'training_mean_stability', 'n_other_contexts')

def rng_for(*parts):
    text = '|'.join(map(str, (SEED, *parts)))
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))

def family_name(screen):
    return screen.split('|')[0].replace('Buchwald-Hartwig(CMC)', 'Buchwald-Hartwig')

def edge_panel(screen, edge_id, n_conditions, endpoints, excluded):
    candidates = np.setdiff1d(np.arange(n_conditions), np.r_[endpoints, excluded])
    assert len(candidates) >= max(BUDGETS)
    return rng_for('anchors', screen, edge_id).permutation(candidates)[:max(BUDGETS)]

def nested_quarter(pool, screen, split, repeat, fold):
    """Exactly reuse training_size draw-0 sample ordering, never its fitted panel."""
    order = seeded('samples', screen, split, repeat, fold, 0).permutation(pool)
    return np.sort(order[:max(8, int(np.rint(len(pool) * 0.25)))])

def fit_clean_covariance(train, excluded):
    """Embed a chemistry-clean fit without giving ambiguous columns a mode."""
    clean = np.setdiff1d(np.arange(train.shape[1]), excluded)
    fit = fit_covariance(train[:, clean])
    mu = np.zeros(train.shape[1])
    covariance = np.zeros((train.shape[1], train.shape[1]))
    mu[clean] = fit['mu']
    covariance[np.ix_(clean, clean)] = fit['cov']
    return dict(fit, mu=mu, cov=covariance)

def correlation(a, b, minimum=MIN_CONTEXT_HALF):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < minimum:
        return np.nan
    x, y = (a[finite] - np.mean(a[finite]), b[finite] - np.mean(b[finite]))
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den > 1e-12 else np.nan

def average(values):
    values = np.asarray(values, float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan

def contrast_prediction(fit, anchor_values, anchors, endpoints, orientation):
    """Only observed anchor values are accepted; return raw and clipped contrast."""
    anchors = np.asarray(anchors, int)
    values = np.asarray(anchor_values, float)
    endpoints = np.asarray(endpoints, int)
    assert not set(anchors) & set(endpoints)
    assert values.ndim == 2 and values.shape[1] == len(anchors)
    mu, cov = (fit['mu'], fit['cov'])
    predicted = np.tile(mu[endpoints], (len(values), 1))
    masks, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for number, mask in enumerate(masks):
        if not mask.any():
            continue
        rows = np.flatnonzero(membership == number)
        observed = anchors[mask]
        cross = cov[np.ix_(endpoints, observed)]
        try:
            weights = np.linalg.solve(cov[np.ix_(observed, observed)], cross.T).T
        except np.linalg.LinAlgError:
            weights = cross @ np.linalg.pinv(cov[np.ix_(observed, observed)])
        predicted[rows] += (values[np.ix_(rows, mask)] - mu[observed]) @ weights.T
    raw = orientation * (predicted[:, 0] - predicted[:, 1])
    clipped = np.clip(predicted, 0.0, 1.0)
    return (raw, orientation * (clipped[:, 0] - clipped[:, 1]))

def contrast_risk(cov, baseline, anchors, endpoints):
    """Fitted full-covariance linear risk of full and mismatched rank-1 weights."""
    a, b = endpoints
    anchors = np.asarray(anchors, int)
    prior = float(cov[a, a] + cov[b, b] - 2 * cov[a, b])
    if not len(anchors):
        return (prior, prior, prior)
    aa = cov[np.ix_(anchors, anchors)]
    cross = cov[a, anchors] - cov[b, anchors]
    base_cross = baseline[a, anchors] - baseline[b, anchors]
    try:
        full_weights = np.linalg.solve(aa, cross)
    except np.linalg.LinAlgError:
        full_weights = np.linalg.pinv(aa) @ cross
    try:
        base_weights = np.linalg.solve(baseline[np.ix_(anchors, anchors)], base_cross)
    except np.linalg.LinAlgError:
        base_weights = np.linalg.pinv(baseline[np.ix_(anchors, anchors)]) @ base_cross
    full = prior - cross @ full_weights
    base = prior - 2 * cross @ base_weights + base_weights @ aa @ base_weights
    assert full <= base + 1e-10 and full <= prior + 1e-10 and (full >= -1e-10)
    return (max(float(full), 0.0), max(float(base), 0.0), prior)

def training_predictors(train, edges, roles, key, fit=None):
    """Every response-dependent input is restricted to selected training rows."""
    fit = fit_covariance(train) if fit is None else fit
    cov = fit['cov']
    values, vectors = np.linalg.eigh(cov)
    leading_value, leading_vector = (values[-1], vectors[:, -1])
    a = edges.condition_a_index.to_numpy(int)
    b = edges.condition_b_index.to_numpy(int)
    delta = (train[:, a] - train[:, b]) * edges.orientation.to_numpy(int)
    halves = []
    for role in (0, 1):
        split = split_half(roles[:, role], rng_for('training_roles', key, role))
        if split is not None:
            halves.append(split)
    full_context = [[] for _ in range(len(edges))]
    stable_context = [[] for _ in range(len(edges))]
    context_gap = [[] for _ in range(len(edges))]
    for _, group in edges.groupby('swap_key', sort=True):
        indices = group.index.to_numpy(int)
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                rho = correlation(delta[:, i], delta[:, j])
                full_context[i].append(rho)
                full_context[j].append(rho)
                for left, right in halves:
                    r1 = correlation(delta[left, i], delta[left, j])
                    r2 = correlation(delta[right, i], delta[right, j])
                    if np.isfinite(r1) and np.isfinite(r2):
                        stable_context[i].append(0.5 * (r1 + r2))
                        stable_context[j].append(0.5 * (r1 + r2))
                        context_gap[i].append(abs(r1 - r2))
                        context_gap[j].append(abs(r1 - r2))
    counts = edges.groupby('swap_key').swap_key.transform('size').to_numpy() - 1
    result = []
    for i, (aa, bb) in enumerate(zip(a, b)):
        var = cov[aa, aa] + cov[bb, bb] - 2 * cov[aa, bb]
        first = leading_value * (leading_vector[aa] - leading_vector[bb]) ** 2
        means = []
        for left, right in halves:
            x, y = (delta[left, i], delta[right, i])
            if min(np.isfinite(x).sum(), np.isfinite(y).sum()) >= MIN_CONTEXT_HALF:
                m1, m2 = (float(np.nanmean(x)), float(np.nanmean(y)))
                means.append(1 - abs(m1 - m2) / (abs(m1) + abs(m2) + 0.02))
        result.append(dict(nonleading_contrast_share=float((var - first) / var) if var > 1e-16 else np.nan, first_factor_contrast_variance=float(first), fitted_contrast_variance=float(var), training_repeated_context_rho=average(full_context[i]), training_context_stable_rho=average(stable_context[i]), training_context_half_rho_gap=average(context_gap[i]), training_mean_stability=average(means), n_other_contexts=int(counts[i]), n_valid_context_correlations=int(np.isfinite(full_context[i]).sum()), n_valid_context_half_comparisons=len(stable_context[i])))
    return result

def prepare_registry(edges, targets):
    registry, coverage = ([], [])
    for screen, one in edges.groupby('screen', sort=True):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            matrix, conditions = (saved['M'], saved['conds'].astype(str))
        lookup = {condition: i for i, condition in enumerate(conditions)}
        cls = screen.split('|')[0]
        excluded = [i for i, condition in enumerate(conditions) if (cls, condition) in AMBIGUOUS]
        original = set(targets.loc[targets.screen.eq(screen), 'condition_id'])
        both = 0
        for row in one.itertuples(index=False):
            a, b = (lookup[row.condition_a], lookup[row.condition_b])
            assert a not in excluded and b not in excluded
            edge_id = hashlib.sha256((screen + '|' + row.condition_a + '|' + row.condition_b).encode()).hexdigest()[:16]
            panel = edge_panel(screen, edge_id, len(conditions), (a, b), excluded)
            original_both = row.condition_a in original and row.condition_b in original
            both += int(original_both)
            finite = np.isfinite(matrix[:, a]) & np.isfinite(matrix[:, b])
            registry.append(dict(row._asdict(), family=family_name(screen), edge_id=edge_id, condition_a_index=a, condition_b_index=b, anchor_indices_json=json.dumps(panel.tolist()), anchor_ids_json=json.dumps(conditions[panel].tolist()), original_both_target_endpoints=int(original_both), n_observed_contrasts=int(finite.sum()), n_zero_contrasts=int((finite & (matrix[:, a] == matrix[:, b])).sum()), eligible=1, reason='all_strict_unambiguous_edges'))
        coverage.append(dict(screen=screen, family=family_name(screen), n_rows=len(matrix), n_matrix_conditions=len(conditions), n_edges=len(one), n_contexts=one.context_id.nunique(), n_swaps=one.swap_key.nunique(), n_repeated_context_pairs=int(sum((n * (n - 1) // 2 for n in one.groupby('swap_key').size()))), n_original_both_target_edges=both, excluded_anchor_conditions_json=json.dumps(conditions[excluded].tolist()), eligible_edges=len(one), excluded_edges=0))
    registry = pd.DataFrame(registry)
    assert len(registry) == 1846 and registry.screen.nunique() == 19
    return (registry, pd.DataFrame(coverage))

def evaluate(registry, folds):
    records, audits, samples = ([], [], [])
    for screen, one in registry.groupby('screen', sort=True):
        one = one.reset_index(drop=True)
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            matrix, pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        excluded = [i for i, condition in enumerate(conditions) if (screen.split('|')[0], condition) in AMBIGUOUS]
        roles = np.array([pair.split(' >> ') for pair in pairs])
        sf = folds[folds.screen.eq(screen)]
        for (split, repeat, fold), g in sf.groupby(['split', 'repeat', 'fold'], sort=True):
            test_indices = g.pair_index.to_numpy(int)
            assert pairs[test_indices].tolist() == g.subpair.tolist()
            pool = np.setdiff1d(np.arange(len(matrix)), test_indices)
            if split.startswith('role'):
                role = int(split[-1]) - 1
                assert not set(roles[pool, role]) & set(roles[test_indices, role])
            for fraction, selected in ((0.25, nested_quarter(pool, screen, split, repeat, fold)), (1.0, pool)):
                assert not set(selected) & set(test_indices)
                key = dict(screen=screen, family=family_name(screen), split=split, repeat=int(repeat), fold=int(fold), fraction=fraction)
                samples.append(dict(key, training_indices_json=json.dumps(selected.tolist()), test_indices_json=json.dumps(test_indices.tolist())))
                train = matrix[selected]
                fit = fit_clean_covariance(train, excluded)
                base_cov = rank1_diagonal_covariance(fit['cov'])
                base_fit = dict(mu=fit['mu'], cov=base_cov)
                predictors = training_predictors(train, one, roles[selected], str(key), fit=fit)
                audits.append(dict(key, n_training=len(selected), n_test=len(test_indices), shrinkage=fit['shrinkage'], imputed_cells=fit['imputed_cells'], disjoint=True, quarter_nested_in_full=set(selected) <= set(pool), n_excluded_ambiguous_mode_columns=len(excluded)))
                for number, edge in enumerate(one.itertuples(index=False)):
                    endpoints = (edge.condition_a_index, edge.condition_b_index)
                    a, b = endpoints
                    actual = edge.orientation * (matrix[test_indices, a] - matrix[test_indices, b])
                    valid = np.isfinite(actual)
                    anchors_all = np.asarray(json.loads(edge.anchor_indices_json), int)
                    train_delta = edge.orientation * (train[:, a] - train[:, b])
                    train_finite = np.isfinite(train_delta)
                    marginal = edge.orientation * (fit['mu'][a] - fit['mu'][b])
                    for k in BUDGETS:
                        anchors = anchors_all[:k]
                        values = matrix[np.ix_(test_indices, anchors)].copy()
                        full_raw, full_clip = contrast_prediction(fit, values, anchors, endpoints, edge.orientation)
                        rank1_raw, rank1_clip = contrast_prediction(base_fit, values, anchors, endpoints, edge.orientation)
                        predicted = dict(full_raw=full_raw, full_clip=full_clip, rank1_raw=rank1_raw, rank1_clip=rank1_clip, marginal=np.repeat(marginal, len(actual)), zero=np.zeros(len(actual)))
                        risk = np.zeros(3)
                        masks, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
                        for label, mask in enumerate(masks):
                            n = int((valid & (membership == label)).sum())
                            risk += n * np.asarray(contrast_risk(fit['cov'], base_cov, anchors[mask], endpoints))
                        finite_y = actual[valid]
                        metrics = {name + '_sse': float(np.sum((pred[valid] - finite_y) ** 2)) for name, pred in predicted.items()}
                        expected_extra = 1 - risk[0] / risk[1] if risk[1] > 1e-16 else np.nan
                        records.append(dict(key, edge_id=edge.edge_id, context_id=edge.context_id, swap_key=edge.swap_key, category=edge.category, k=k, n=int(valid.sum()), n_zero=int(np.sum(finite_y == 0)), n_train_contrasts=int(train_finite.sum()), train_constant=int(np.ptp(train_delta[train_finite]) < 1e-12) if train_finite.any() else 1, test_constant=int(np.ptp(finite_y) < 1e-12) if len(finite_y) else 1, observed_anchors=int(np.isfinite(values[valid]).sum()), expected_full_sse=risk[0], expected_rank1_sse=risk[1], expected_prior_sse=risk[2], expected_extra_gain=expected_extra, **predictors[number], **metrics))
        print('CHEMICAL_MODE', screen, 'cumulative_fold_edge_rows', len(records), flush=True)
    return (pd.DataFrame(records), pd.DataFrame(audits), pd.DataFrame(samples))

def summarize_edge_scores(folds):
    keys = ['screen', 'family', 'split', 'fraction', 'k', 'edge_id', 'context_id']
    sums = [column for column in folds if column.endswith('_sse')] + ['n']
    average_columns = [column for column in PREDICTORS if column in folds and column != 'expected_extra_gain']
    edge = folds.groupby(keys, as_index=False)[sums].sum()
    if average_columns:
        edge = edge.merge(folds.groupby(keys, as_index=False)[average_columns].mean(), on=keys)
    for column in [c for c in sums if c.endswith('_sse')]:
        edge[column.replace('_sse', '_mse')] = edge[column] / edge.n.replace(0, np.nan)
    edge['extra_gain_raw'] = 1 - edge.full_raw_mse / edge.rank1_raw_mse.replace(0, np.nan)
    edge['recovery_vs_mean_raw'] = 1 - edge.full_raw_mse / edge.marginal_mse.replace(0, np.nan)
    if 'expected_rank1_mse' in edge:
        edge['expected_extra_gain'] = 1 - edge.expected_full_mse / edge.expected_rank1_mse.replace(0, np.nan)
    metrics = [c for c in edge if c.endswith('_mse')] + [p for p in PREDICTORS if p in edge]
    context_keys = ['screen', 'family', 'split', 'fraction', 'k', 'context_id']
    context = edge.groupby(context_keys, as_index=False)[metrics].mean()
    screen_keys = ['screen', 'family', 'split', 'fraction', 'k']
    summary = context.groupby(screen_keys, as_index=False)[metrics].mean()
    counts = edge.groupby(screen_keys, as_index=False).agg(n_edges=('edge_id', 'size'), n_contexts=('context_id', 'nunique'), n_contrast_predictions=('n', 'sum'))
    summary = summary.merge(counts, on=screen_keys)
    for kind in ('raw', 'clip'):
        if 'full_' + kind + '_mse' not in summary:
            continue
        summary['extra_gain_' + kind] = 1 - summary['full_' + kind + '_mse'] / summary['rank1_' + kind + '_mse'].replace(0, np.nan)
        summary['recovery_vs_mean_' + kind] = 1 - summary['full_' + kind + '_mse'] / summary.marginal_mse.replace(0, np.nan)
        summary['recovery_vs_zero_' + kind] = 1 - summary['full_' + kind + '_mse'] / summary.zero_mse.replace(0, np.nan)
    if 'expected_rank1_mse' in summary:
        summary['expected_extra_gain'] = 1 - summary.expected_full_mse / summary.expected_rank1_mse.replace(0, np.nan)
    return (edge, summary)

def aggregate_families(summary):
    keys = ['family', 'split', 'fraction', 'k']
    metrics = [c for c in summary if c.endswith('_mse')] + [p for p in PREDICTORS if p in summary]
    family = summary.groupby(keys, as_index=False)[metrics].mean()
    family = family.merge(summary.groupby(keys, as_index=False).agg(n_screens=('screen', 'nunique')), on=keys)
    for kind in ('raw', 'clip'):
        family['extra_gain_' + kind] = 1 - family['full_' + kind + '_mse'] / family['rank1_' + kind + '_mse'].replace(0, np.nan)
        family['recovery_vs_mean_' + kind] = 1 - family['full_' + kind + '_mse'] / family.marginal_mse.replace(0, np.nan)
    family['expected_extra_gain'] = 1 - family.expected_full_mse / family.expected_rank1_mse.replace(0, np.nan)
    return family

def association_rows(screen, family, edge):
    rows = []
    for unit, frame in (('screen', screen), ('family', family)):
        for group_key, group in frame.groupby(['split', 'fraction', 'k'], sort=True):
            for predictor in PREDICTORS:
                for outcome in ('extra_gain_raw', 'recovery_vs_mean_raw'):
                    good = group[[predictor, outcome]].dropna()
                    valid = len(good) >= 5 and good[predictor].nunique() > 1 and (good[outcome].nunique() > 1)
                    rows.append(dict(zip(['split', 'fraction', 'k'], group_key), unit=unit, predictor=predictor, outcome=outcome, n=len(good), spearman=float(spearmanr(good[predictor], good[outcome]).statistic) if valid else np.nan, reason='' if valid else 'insufficient_or_constant'))
    within = []
    for key, group in edge.groupby(['screen', 'family', 'split', 'fraction', 'k'], sort=True):
        for predictor in PREDICTORS:
            good = group[[predictor, 'extra_gain_raw']].dropna()
            valid = len(good) >= 5 and good[predictor].nunique() > 1 and (good.extra_gain_raw.nunique() > 1)
            within.append(dict(zip(['screen', 'family', 'split', 'fraction', 'k'], key), predictor=predictor, n_edges=len(good), spearman=float(spearmanr(good[predictor], good.extra_gain_raw).statistic) if valid else np.nan))
    return (pd.DataFrame(rows), pd.DataFrame(within))

def transitions(summary):
    keys = ['screen', 'family', 'split', 'k']
    low, high = (summary[summary.fraction.eq(0.25)], summary[summary.fraction.eq(1.0)])
    paired = low.drop(columns='fraction').merge(high.drop(columns='fraction'), on=keys, suffixes=('_quarter', '_full'), validate='one_to_one')
    assert paired.n_contrast_predictions_quarter.eq(paired.n_contrast_predictions_full).all()
    for model in ('full', 'rank1'):
        for kind in ('raw', 'clip'):
            paired[model + '_mse_reduction_' + kind] = 1 - paired[model + '_' + kind + '_mse_full'] / paired[model + '_' + kind + '_mse_quarter']
    paired['extra_gain_change_raw'] = paired.extra_gain_raw_full - paired.extra_gain_raw_quarter
    return paired

def make_figure(summary, association):
    screens = sorted(summary.screen.unique())
    y = np.arange(len(screens))
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.7))
    colors = {'pair': '#3b7588', 'role1': '#26806a', 'role2': '#b05e58'}
    for split, offset in (('pair', -0.18), ('role1', 0), ('role2', 0.18)):
        sub = summary[summary.split.eq(split) & summary.k.eq(12)].set_index(['fraction', 'screen'])
        for fraction, marker in ((0.25, 'o'), (1.0, 's')):
            g = sub.loc[fraction].reindex(screens)
            axes[0].scatter(100 * g.extra_gain_raw, y + offset, s=25, marker=marker, facecolors=colors[split] if fraction == 1.0 else 'none', edgecolors=colors[split])
            axes[1].scatter(100 * g.recovery_vs_mean_raw, y + offset, s=25, marker=marker, facecolors=colors[split] if fraction == 1.0 else 'none', edgecolors=colors[split])
    for ax in axes[:2]:
        ax.set(yticks=y, yticklabels=[s.replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O') for s in screens])
        ax.invert_yaxis()
        ax.axvline(0, color='#777777', lw=0.8)
        ax.grid(axis='x', alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(xlabel='MSE recovery beyond rank-1 + diagonal (%)', title='A  Extra contrast recovery')
    axes[1].set(xlabel='MSE recovery beyond training mean contrast (%)', title='B  Substrate-dependent recovery')
    chosen = association[association.unit.eq('family') & association.fraction.eq(1.0) & association.k.eq(12) & association.outcome.eq('extra_gain_raw')]
    names = ['expected_extra_gain', 'nonleading_contrast_share', 'training_repeated_context_rho', 'training_context_stable_rho', 'training_context_half_rho_gap', 'training_mean_stability']
    labels = ['Fitted extra contrast risk', 'Nonleading variance share', 'Train context correlation', 'Train half-mean context rho', 'Train context half-rho gap', 'Train mean-effect stability']
    for split, offset in (('pair', -0.18), ('role1', 0), ('role2', 0.18)):
        g = chosen[chosen.split.eq(split)].set_index('predictor').reindex(names)
        axes[2].scatter(g.spearman, np.arange(len(names)) + offset, color=colors[split], s=28, label=split)
    axes[2].set(yticks=np.arange(len(names)), yticklabels=labels, xlim=(-1.03, 1.03), xlabel='Spearman rho across merged families', title='C  Training-only predictors; full training')
    axes[2].invert_yaxis()
    axes[2].axvline(0, color='#777777', lw=0.8)
    axes[2].grid(axis='x', alpha=0.2)
    axes[2].spines[['top', 'right']].set_visible(False)
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle('Strict chemical contrasts: held-out recovery and training-only response modes', fontsize=15)
    fig.text(0.19, 0.025, 'All 1,808 edges / 384 contexts / 18 screens. Neither endpoint observed as anchor. k=12.\nOpen circle: quarter training; filled square: full training. Edge -> equal context -> screen; association pools 14 families where supported.', fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.12, top=0.91, wspace=0.7)
    fig.savefig(FIGURES / '15_chemical_mode_heldout_recovery.png', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries-only', action='store_true', help='Reuse hash-verified fold scores to refresh summaries and figure.')
    args = parser.parse_args()
    start = time.time()
    inputs = [TABLES / 'chemical_context_replication_edges.csv', DATA / 'cv_fold_registry.csv', DATA / 'cv_target_registry.csv', DATA / 'chemical_context_replication_manifest.json', HERE / 'chemical_context_replication.py', HERE / 'response_mode_ablation.py', HERE / 'training_size_intervention.py', HERE / 'shrinkage_covariance_analysis.py']
    inputs += sorted(DATA.glob('matrix_*.npz'))
    hashes = {path.relative_to(HERE).as_posix(): sha256(path) for path in inputs}
    inherited = json.loads((DATA / 'chemical_context_replication_manifest.json').read_text(encoding='utf-8'))
    assert hashes['tables/chemical_context_replication_edges.csv'] == inherited['output_hashes']['chemical_context_replication_edges.csv']
    edges, folds, targets = (pd.read_csv(inputs[i]) for i in (0, 1, 2))
    registry, coverage = prepare_registry(edges, targets)
    registry.to_csv(TABLES / 'chemical_mode_target_anchor_registry.csv', index=False, encoding='utf-8-sig')
    coverage.to_csv(TABLES / 'chemical_mode_coverage.csv', index=False, encoding='utf-8-sig')
    registry_hash = sha256(TABLES / 'chemical_mode_target_anchor_registry.csv')
    if args.summaries_only:
        previous = json.loads((DATA / 'chemical_mode_manifest.json').read_text(encoding='utf-8'))
        assert previous['source_hashes'] == hashes
        assert previous['registry_frozen_sha256'] == registry_hash
        cached = ['chemical_mode_fold_edge_scores.csv.gz', 'chemical_mode_fit_audit.csv', 'chemical_mode_training_sample_registry.csv']
        for name in cached:
            assert sha256(TABLES / name) == previous['output_hashes'][name]
        fold_scores, audit, samples = (pd.read_csv(TABLES / name) for name in cached)
        evaluation_seconds = previous.get('evaluation_seconds', previous['elapsed_seconds'])
    else:
        fold_scores, audit, samples = evaluate(registry, folds)
        evaluation_seconds = time.time() - start
    edge_scores, summary = summarize_edge_scores(fold_scores)
    family = aggregate_families(summary)
    associations, within = association_rows(summary, family, edge_scores)
    transition = transitions(summary)
    support = fold_scores.groupby(['screen', 'split', 'fraction', 'k'], as_index=False).agg(fold_edge_rows=('edge_id', 'size'), contrast_predictions=('n', 'sum'), zero_contrast_predictions=('n_zero', 'sum'), constant_training_fold_edges=('train_constant', 'sum'), constant_test_fold_edges=('test_constant', 'sum'), repeated_context_predictor_rows=('training_repeated_context_rho', 'count'), stable_context_predictor_rows=('training_context_stable_rho', 'count'), mean_stability_predictor_rows=('training_mean_stability', 'count'))
    outputs = {'chemical_mode_fold_edge_scores.csv.gz': fold_scores, 'chemical_mode_edge_scores.csv': edge_scores, 'chemical_mode_screen_summary.csv': summary, 'chemical_mode_family_summary.csv': family, 'chemical_mode_associations.csv': associations, 'chemical_mode_within_screen_associations.csv': within, 'chemical_mode_training_transitions.csv': transition, 'chemical_mode_fit_audit.csv': audit, 'chemical_mode_training_sample_registry.csv': samples, 'chemical_mode_support.csv': support}
    for name, frame in outputs.items():
        frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')
    make_figure(summary, associations)
    assert len(audit) == 760 and len(summary) == 228 and (len(family) == 168)
    assert coverage.n_edges.sum() == 1846 and coverage.n_contexts.sum() == 400
    assert coverage.n_original_both_target_edges.sum() == 40
    assert sha256(TABLES / 'chemical_mode_target_anchor_registry.csv') == registry_hash
    assert hashes == {path.relative_to(HERE).as_posix(): sha256(path) for path in inputs}
    output_names = list(outputs) + ['chemical_mode_target_anchor_registry.csv', 'chemical_mode_coverage.csv']
    manifest = dict(date='2026-10-16', seed=SEED, python=platform.python_version(), source_hashes=hashes, inputs_unchanged=True, registry_frozen_sha256=registry_hash, code_sha256=sha256(Path(__file__)), test_code_sha256=sha256(HERE / 'test_chemical_mode_recovery.py'), report_sha256=sha256(HERE / '22_CHEMICAL_MODES_AND_HELDOUT_RECOVERY_2026-10-16.md') if (HERE / '22_CHEMICAL_MODES_AND_HELDOUT_RECOVERY_2026-10-16.md').exists() else None, output_hashes={name: sha256(TABLES / name) for name in output_names}, figure_sha256=sha256(FIGURES / '15_chemical_mode_heldout_recovery.png'), screens=18, families=14, strict_edges=1808, contexts=384, original_both_target_edges=38, original_target_screens=13, inherited_folds=360, fitted_training_covariances=len(audit), fold_edge_score_rows=len(fold_scores), edge_score_rows=len(edge_scores), contrast_predictions=int(fold_scores.n.sum()), elapsed_seconds=time.time() - start, evaluation_seconds=evaluation_seconds, hash_verified_cached_fit_reuse=args.summaries_only, design={'inclusion': 'All inherited strict unambiguous identity_swap/solvent_only edges; no response-based inclusion or example selection.', 'anchors': 'Deterministic seed+screen+edge random order over all columns except both edge endpoints and chemistry-audited ambiguous columns; prefixes6/12, fixed across CV folds and quarter/full training. Ambiguous columns are also excluded from covariance/mode fitting.', 'targets': 'Both endpoints hidden at test time; finite observed endpoint pairs scored. Registry covers every eligible edge; original fixed targets cover38 edges/13screens only.', 'training': 'Inherited360folds; full complement versus nested draw0quarter from training_size ordering, min8; one substrate role disjoint for role splits, other role may overlap.', 'primary': 'Raw signed high-registry-ID minus low-ID contrast fullLW MSE vs leading-eigenfactor+heterogeneous diagonal MSE. Both use same selected training means and anchors. Clipped endpoint sensitivity also saved.', 'baselines': 'Zero contrast and training marginal-mean contrast. Raw homogeneous shared-activity adjustment cancels in any both-hidden-endpoint difference and equals marginal contrast; rank1+diagonal has heterogeneous loadings and can recover nonzero substrate-dependent contrasts.', 'predictors': 'Selected-training-only LW expected contrast risk, leading/nonleading projected contrast variance, repeated-swap other-context correlations and within-selected-training role-half correlation stability; no global report17 response correlation enters prediction.', 'training_context_minimum_half': MIN_CONTEXT_HALF, 'aggregation': 'Sum test-error/count across inherited folds per edge; equal edges within fixed recipe context, equal contexts within screen, equal screens within merged family. Associations across18screens/14families where predictor supported; no edge-level independence claim.', 'budget': '2training endpoints x2anchor budgets x360inherited folds; all1808edges; no search over seeds/ranks/budgets after scores.'}, limitations=['Exploratory fixed-library empirical recovery, not causal mechanism, pure chemical specificity, physical dimension or new-library validation.', 'Response-mode predictors estimated on selected training only; cross-fold reuse remains dependent and associations are descriptive.', 'No replicate-well noise estimate; finite substrate sampling, batch and noise cannot be separated.', 'Context predictors unsupported for many edges/screens, especially quarter training; exact support retained rather than zero-imputed.', 'Contrast edges reuse conditions and rows. Equal-context pooling reduces domination but does not make observations independent.', 'Original target registry is not comparable as a universal contrast denominator; new outcome-independent target/anchor registry is explicitly exploratory.'])
    (DATA / 'chemical_mode_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    selected = summary[summary.k.eq(12)]
    print(selected.groupby(['split', 'fraction'])[['extra_gain_raw', 'recovery_vs_mean_raw']].median().to_string())
    print('elapsed_seconds', time.time() - start, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
