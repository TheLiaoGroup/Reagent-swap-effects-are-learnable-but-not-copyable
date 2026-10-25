"""Controlled nested training-size sensitivity on fixed registered test cells.

The training covariance risk is a fitted linear-risk diagnostic, not a noise
floor. All fractions share test rows/targets and each fixed-random panel."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from dimension_core import select_panel
from response_mode_ablation import rank1_diagonal_covariance
from response_specific_information_risk import expected_risk
from shrinkage_covariance_analysis import fit_covariance, matrix_path
HERE = Path(__file__).resolve().parent
DATA, TABLES, FIGURES = (HERE / 'data', HERE / 'tables', HERE / 'figures')
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
BUDGETS = (6, 12, 24)
DRAWS = 3
SEED = 20261016

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def seeded(*parts):
    key = '|'.join(map(str, (SEED, *parts)))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

def nested_samples(pool, rng):
    pool = np.asarray(pool, int)
    assert len(pool) >= 8 and len(np.unique(pool)) == len(pool)
    order = rng.permutation(pool)
    return [np.sort(order[:max(8, int(np.rint(len(pool) * f)))]) for f in FRACTIONS]

def predict_raw(fit, values, anchors):
    anchors, values = (np.asarray(anchors, int), np.asarray(values, float))
    assert values.ndim == 2 and values.shape[1] == len(anchors)
    out = np.tile(fit['mu'], (len(values), 1))
    patterns, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for number, mask in enumerate(patterns):
        if not mask.any():
            continue
        rows = np.flatnonzero(membership == number)
        observed = anchors[mask]
        weights = np.linalg.solve(fit['cov'][np.ix_(observed, observed)], fit['cov'][:, observed].T).T
        out[rows] += (values[np.ix_(rows, mask)] - fit['mu'][observed]) @ weights.T
    return out

def score_arrays(y, prediction):
    finite = np.isfinite(y)
    n = finite.sum(axis=1)
    assert np.all(n > 0) and np.isfinite(prediction).all()
    error = np.where(finite, prediction - y, 0.0)
    mean_error = error.sum(axis=1) / n
    sse = (error ** 2).sum(axis=1)
    mean_sse = n * mean_error ** 2
    profile = (np.where(finite, error - mean_error[:, None], 0.0) ** 2).sum(axis=1)
    assert np.allclose(sse, mean_sse + profile, rtol=1e-10, atol=1e-12)
    return dict(n_targets=n, sse=sse, mean_sse=mean_sse, profile_sse=profile)

def risk_rows(cov, baseline, anchors, values, targets, finite):
    full = np.empty(len(values))
    base = np.empty(len(values))
    prior = finite @ np.diag(cov)[targets]
    patterns, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for number, mask in enumerate(patterns):
        rows = np.flatnonzero(membership == number)
        f, b, _ = expected_risk(cov, baseline, targets, anchors[mask])
        full[rows], base[rows] = (finite[rows] @ f, finite[rows] @ b)
    assert np.all(full <= base + 1e-10)
    return (full, base, prior)

def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / den) if den > 1e-20 else np.nan

def evaluate():
    paths = [DATA / 'cv_fold_registry.csv', DATA / 'cv_target_registry.csv', TABLES / 'training_selected_panels.csv']
    hashes = {str(p.relative_to(HERE)): digest(p) for p in paths}
    folds, targets, oldpanels = [pd.read_csv(p) for p in paths]
    rows, audits, sample_registry, panel_registry, endpoint = ([], [], [], [], [])
    for screen, sf in folds.groupby('screen', sort=True):
        path = matrix_path(screen)
        hashes[str(path.relative_to(HERE))] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            matrix, pairs, conditions = (z['M'], z['sps'].astype(str), z['conds'].astype(str))
        trg = targets.loc[targets.screen.eq(screen), 'condition_index'].to_numpy(int)
        candidates = np.setdiff1d(np.arange(matrix.shape[1]), trg)
        assert len(candidates) >= max(BUDGETS)
        for (split, repeat, fold), g in sf.groupby(['split', 'repeat', 'fold'], sort=True):
            ti = g.pair_index.to_numpy(int)
            assert list(pairs[ti]) == g.subpair.tolist()
            pool = np.setdiff1d(np.arange(len(matrix)), ti)
            if split.startswith('role'):
                role = int(split[-1]) - 1
                assert not {pairs[i].split(' >> ')[role] for i in pool} & {pairs[i].split(' >> ')[role] for i in ti}
            test = matrix[ti]
            y, finite = (test[:, trg], np.isfinite(test[:, trg]))
            n = finite.sum(axis=1)
            assert np.all(n > 0)
            dy = np.where(finite, y - np.nanmean(y, axis=1, keepdims=True), 0.0)
            subsets = dict(all=np.ones(len(test), bool), productive=np.nanmax(test, axis=1) >= 0.2, informative=(n >= 3) & (np.sqrt((dy * dy).sum(axis=1) / n) > 0.01))
            full_fit = fit_covariance(matrix[pool])
            full_rank1 = rank1_diagonal_covariance(full_fit['cov'])
            full_v = np.linalg.eigh(full_fit['cov'])[1][:, -1]
            residual_reference = full_fit['cov'] - full_rank1
            original = oldpanels[oldpanels.screen.eq(screen) & oldpanels.split.eq(split) & oldpanels.repeat.eq(repeat) & oldpanels.fold.eq(fold) & oldpanels.panel.eq('variance_greedy') & oldpanels.draw.eq(0)].sort_values('position')
            original_order = [np.flatnonzero(conditions == c)[0] for c in original.condition_id]
            for draw in range(DRAWS):
                samples = nested_samples(pool, seeded('samples', screen, split, repeat, fold, draw))
                random_order = seeded('panel', screen, split, repeat, fold, draw).permutation(candidates)[:24]
                for fraction, selected in zip(FRACTIONS, samples):
                    train = matrix[selected]
                    fit = full_fit if fraction == 1.0 else fit_covariance(train)
                    baseline = full_rank1 if fraction == 1.0 else rank1_diagonal_covariance(fit['cov'])
                    nested_order = candidates[select_panel(train[:, candidates], 24, 'variance_greedy', seeded('unused'))]
                    if fraction == 1.0:
                        assert np.array_equal(nested_order, original_order)
                    key = dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), draw=draw, fraction=fraction, training_pairs=len(selected), full_training_pairs=len(pool), test_pairs=len(test))
                    sample_registry.append(dict(key, pair_indices=json.dumps(selected.tolist()), test_indices=json.dumps(ti.tolist())))
                    v = np.linalg.eigh(fit['cov'])[1][:, -1]
                    audits.append(dict(key, shrinkage=fit['shrinkage'], imputed_cells=fit['imputed_cells'], first_direction_abs_cosine_to_full=abs(float(v @ full_v)), offdiagonal_residual_cosine_to_full=cosine(fit['cov'] - baseline, residual_reference), covariance_relative_distance_to_full=float(np.linalg.norm(fit['cov'] - full_fit['cov']) / np.linalg.norm(full_fit['cov'])), disjoint_from_test=not bool(set(selected) & set(ti)), nested_previous=True if fraction == 0.25 else bool(set(samples[FRACTIONS.index(fraction) - 1]) <= set(selected))))
                    for panel, order in [('fixed_random', random_order), ('nested_greedy', nested_order)]:
                        assert not set(order) & set(trg)
                        panel_registry.append(dict(key, panel=panel, condition_indices=json.dumps(order.tolist())))
                        for k in BUDGETS:
                            anchors = order[:k]
                            values = test[:, anchors].copy()
                            raw_f = predict_raw(fit, values, anchors)[:, trg]
                            raw_b = predict_raw(dict(mu=fit['mu'], cov=baseline), values, anchors)[:, trg]
                            clip_f, clip_b = (np.clip(raw_f, 0, 1), np.clip(raw_b, 0, 1))
                            metric = {}
                            for name, pred in [('full_raw', raw_f), ('rank1_raw', raw_b), ('full_clip', clip_f), ('rank1_clip', clip_b), ('marginal', np.tile(fit['mu'][trg], (len(test), 1)))]:
                                scored = score_arrays(y, pred)
                                for label in ('sse', 'mean_sse', 'profile_sse'):
                                    metric[name + '_' + label] = scored[label]
                            er_full, er_base, er_prior = risk_rows(fit['cov'], baseline, anchors, values, trg, finite)
                            metric.update(expected_full_sse=er_full, expected_rank1_sse=er_base, expected_prior_sse=er_prior, full_clipped_cells=np.sum(finite & ((raw_f < 0) | (raw_f > 1)), axis=1))
                            assert np.all(metric['full_clip_sse'] <= metric['full_raw_sse'] + 1e-12)
                            for subset, keep in subsets.items():
                                if not keep.any():
                                    continue
                                rows.append(dict(key, panel=panel, k=k, subset=subset, scored_pairs=int(keep.sum()), n_targets=int(n[keep].sum()), **{name: float(value[keep].sum()) for name, value in metric.items()}))
                            if fraction == 1.0 and draw == 0 and (panel == 'nested_greedy'):
                                for i, pair_index in enumerate(ti):
                                    endpoint.append(dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), pair_index=int(pair_index), k=k, full_sse=float(metric['full_clip_sse'][i]), rank1_sse=float(metric['rank1_clip_sse'][i]), n_targets=int(n[i])))
        print('TRAINING_SIZE', screen, 'fold_rows', len(rows), flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(audits), pd.DataFrame(sample_registry), pd.DataFrame(panel_registry), pd.DataFrame(endpoint), hashes)

def summarize(folds):
    keys = ['screen', 'split', 'fraction', 'panel', 'k', 'subset']
    sums = [c for c in folds if c.endswith('_sse')] + ['n_targets', 'scored_pairs', 'full_clipped_cells']
    result = folds.groupby(keys, as_index=False)[sums].sum()
    result = result.merge(folds.groupby(keys, as_index=False)[['training_pairs', 'full_training_pairs', 'test_pairs']].mean(), on=keys)
    for kind in ('raw', 'clip'):
        result['actual_extra_gain_' + kind] = 1 - result['full_' + kind + '_sse'] / result['rank1_' + kind + '_sse']
        result['actual_total_recovery_' + kind] = 1 - result['full_' + kind + '_sse'] / result.marginal_sse
        result['profile_extra_gain_' + kind] = 1 - result['full_' + kind + '_profile_sse'] / result['rank1_' + kind + '_profile_sse']
        result['full_rmse_pp_' + kind] = 100 * np.sqrt(result['full_' + kind + '_sse'] / result.n_targets)
    result['expected_extra_gain'] = 1 - result.expected_full_sse / result.expected_rank1_sse
    result['expected_total_recovery'] = 1 - result.expected_full_sse / result.expected_prior_sse
    result['raw_risk_ratio'] = result.full_raw_sse / result.expected_full_sse
    result['clipped_cell_fraction'] = result.full_clipped_cells / result.n_targets
    return result

def transitions(summary):
    keys = ['screen', 'split', 'panel', 'k', 'subset']
    low = summary[summary.fraction.eq(0.25)].drop(columns='fraction')
    high = summary[summary.fraction.eq(1.0)].drop(columns='fraction')
    out = low.merge(high, on=keys, suffixes=('_quarter', '_full'), validate='one_to_one')
    assert out.n_targets_quarter.eq(out.n_targets_full).all()
    for model in ('full', 'rank1'):
        out[model + '_clip_mse_reduction'] = 1 - out[model + '_clip_sse_full'] / out[model + '_clip_sse_quarter']
    out['extra_gain_change_clip'] = out.actual_extra_gain_clip_full - out.actual_extra_gain_clip_quarter
    out['expected_extra_gain_change'] = out.expected_extra_gain_full - out.expected_extra_gain_quarter
    out['raw_risk_ratio_change'] = out.raw_risk_ratio_full - out.raw_risk_ratio_quarter
    return out

def associations(summary):
    selected = summary[summary.subset.eq('all')].copy()
    selected['family'] = selected.screen.str.split('|', regex=False).str[0].replace({'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig'})
    rows = []
    for keys, g in selected.groupby(['split', 'fraction', 'panel', 'k'], sort=True):
        for unit in ('screen', 'family'):
            if unit == 'screen':
                x = g
            else:
                fields = ['expected_full_sse', 'expected_rank1_sse', 'full_raw_sse', 'rank1_raw_sse', 'full_clip_sse', 'rank1_clip_sse']
                x = g.groupby('family', as_index=False)[fields].sum()
                x['expected_extra_gain'] = 1 - x.expected_full_sse / x.expected_rank1_sse
                for kind in ('raw', 'clip'):
                    x['actual_extra_gain_' + kind] = 1 - x['full_' + kind + '_sse'] / x['rank1_' + kind + '_sse']
            for actual in ('actual_extra_gain_raw', 'actual_extra_gain_clip'):
                rows.append(dict(zip(['split', 'fraction', 'panel', 'k'], keys), unit=unit, actual=actual, n=len(x), spearman=float(spearmanr(x.expected_extra_gain, x[actual]).statistic), median_predicted_extra=float(x.expected_extra_gain.median()), median_actual_extra=float(x[actual].median())))
    return pd.DataFrame(rows)

def endpoint_check(endpoint):
    path = TABLES / 'response_mode_pair_scores.csv'
    source = pd.read_csv(path, usecols=['screen', 'split', 'repeat', 'fold', 'pair_index', 'k', 'model', 'sse', 'n_targets'])
    keys = ['screen', 'split', 'repeat', 'fold', 'pair_index', 'k']
    checks = []
    for name, model in [('full', 'full_lw'), ('rank1', 'spectral_r1_diag')]:
        ref = source[source.model.eq(model)].drop(columns='model')
        matched = endpoint.merge(ref, on=keys, validate='one_to_one', suffixes=('', '_reference'))
        assert len(matched) == len(endpoint) and matched.n_targets.eq(matched.n_targets_reference).all()
        delta = float(np.max(abs(matched[name + '_sse'] - matched.sse)))
        assert delta < 1e-10
        checks.append(dict(model=model, matched_rows=len(matched), maximum_absolute_sse_difference=delta))
    return (checks, path)

def plot(summary, audits):
    chosen = summary[summary.split.eq('pair') & summary.panel.eq('fixed_random') & summary.subset.eq('all')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {6: '#137c78', 12: '#c05a39', 24: '#6658a0'}
    for k in BUDGETS:
        g = chosen[chosen.k.eq(k)].groupby('fraction').median(numeric_only=True)
        axes[0, 0].plot(100 * g.index, g.actual_extra_gain_clip * 100, 'o-', color=colors[k], label=f'{k} anchors')
        axes[0, 0].plot(100 * g.index, g.expected_extra_gain * 100, '--', color=colors[k], alpha=0.65)
        axes[0, 1].plot(100 * g.index, g.raw_risk_ratio, 'o-', color=colors[k], label=f'{k} anchors')
    axes[0, 0].set(xlabel='Historical training pairs retained (%)', ylabel='Extra gain over rank1 + diagonal (%)', title='Solid: clipped test; dashed: fitted linear risk')
    axes[0, 0].axhline(0, color='black', lw=0.7)
    axes[0, 0].legend()
    axes[0, 1].set(xlabel='Historical training pairs retained (%)', ylabel='Observed raw SSE / fitted full risk', title='Risk discrepancy under fixed random measurements')
    axes[0, 1].axhline(1, color='black', lw=0.7)
    ag = audits[audits.split.eq('pair')].groupby(['screen', 'fraction'])[['first_direction_abs_cosine_to_full', 'offdiagonal_residual_cosine_to_full']].mean().groupby('fraction').median()
    for name, label, col in [('first_direction_abs_cosine_to_full', 'Leading direction', '#137c78'), ('offdiagonal_residual_cosine_to_full', 'Additional off-diagonal structure', '#c05a39')]:
        axes[1, 0].plot(100 * ag.index, ag[name], 'o-', label=label, color=col)
    axes[1, 0].set(xlabel='Historical training pairs retained (%)', ylabel='Agreement with full-training estimate', title='Reference agreement is not ground-truth accuracy', ylim=(-0.05, 1.05))
    axes[1, 0].legend(fontsize=8)
    tr = transitions(summary)
    x = tr[tr.split.eq('pair') & tr.panel.eq('fixed_random') & tr.k.eq(12) & tr.subset.eq('all')].sort_values('extra_gain_change_clip')
    axes[1, 1].barh(x.screen.str.replace('|1', '', regex=False), 100 * x.extra_gain_change_clip, color=np.where(x.extra_gain_change_clip >= 0, '#137c78', '#c05a39'))
    axes[1, 1].axvline(0, color='black', lw=0.7)
    axes[1, 1].set(xlabel='Extra-gain change, quarter to full (percentage points)', title='Heterogeneous learning at 12 anchors')
    axes[1, 1].tick_params(axis='y', labelsize=7)
    fig.savefig(FIGURES / '10_training_size_intervention.png', dpi=180)
    plt.close(fig)

def main():
    started = time.time()
    with threadpool_limits(limits=1):
        folds, audits, samples, panels, endpoint, hashes = evaluate()
        summary = summarize(folds)
        changes = transitions(summary)
        association = associations(summary)
        endpoint_checks, reference_path = endpoint_check(endpoint)
        plot(summary, audits)
    hashes[str(reference_path.relative_to(HERE))] = digest(reference_path)
    outputs = {TABLES / 'training_size_fold_scores.csv.gz': folds, TABLES / 'training_size_fit_audit.csv': audits, TABLES / 'training_size_samples.csv': samples, TABLES / 'training_size_panels.csv': panels, TABLES / 'training_size_summary.csv': summary, TABLES / 'training_size_transitions.csv': changes, TABLES / 'training_size_associations.csv': association, TABLES / 'training_size_full_endpoint_scores.csv.gz': endpoint}
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    for name, expected in hashes.items():
        assert digest(HERE / name) == expected
    files = list(outputs) + [FIGURES / '10_training_size_intervention.png']
    code = [Path(__file__), HERE / 'test_training_size_intervention.py', HERE / 'dimension_core.py', HERE / 'shrinkage_covariance_analysis.py', HERE / 'response_mode_ablation.py', HERE / 'response_specific_information_risk.py']
    manifest = dict(date='2026-10-16', status='exploratory existing-data nested training-size intervention', fractions=FRACTIONS, draws=DRAWS, budgets=BUDGETS, screens=int(folds.screen.nunique()), split_counts=folds.groupby('split').size().to_dict(), fold_rows=len(folds), sample_sets=len(samples), panels=len(panels), endpoint_checks=endpoint_checks, design='Fixed outer test rows and targets; three deterministic nested training samples. Fixed random measurement panels do not depend on training yields; nested_greedy panels are reselected from each subsample only. All models use the same subsample means.', risk='Unclipped model covariance linear risk; baseline coefficients evaluated under full covariance fitted to the same subsample. Empirical errors scored both raw and clipped. Risk omits mean-estimation uncertainty and is not a noise floor.', pooling='Sum errors and target counts across folds and draws, then take ratios; screen medians used in figure. Full-training fits repeat across the three draws, which are not independent replications. Learning reflects row count and substrate diversity jointly.', scope='All18 analysis screens and all existing pair/role folds. Productive/informative subsets are retrospective scoring only; no outer-test selection.', input_sha256=hashes, code_sha256={str(p.relative_to(HERE)): digest(p) for p in code}, output_sha256={str(p.relative_to(HERE)): digest(p) for p in files}, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, seconds=time.time() - started)
    (DATA / 'training_size_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    show = changes[changes.split.eq('pair') & changes.panel.eq('fixed_random') & changes.subset.eq('all')]
    print(show.groupby('k')[['full_clip_mse_reduction', 'extra_gain_change_clip', 'raw_risk_ratio_change']].agg(['median', 'min', 'max']).to_string(), flush=True)
    print('FINISHED_TRAINING_SIZE', len(folds), len(samples), time.time() - started, flush=True)
if __name__ == '__main__':
    main()
