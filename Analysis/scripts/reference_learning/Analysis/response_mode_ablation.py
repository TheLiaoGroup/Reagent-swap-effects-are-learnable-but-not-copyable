"""Training-only covariance-mode ablation on the existing fixed CV experiment.

Retained modes describe deviations from an isotropic residual covariance, not
physical rank. Hidden test targets are supplied only to scoring functions."""
from __future__ import annotations
import argparse
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
import sklearn
from threadpoolctl import threadpool_limits
from dimension_core import select_panel
from run_existing_evidence import score_predictions
from shrinkage_covariance_analysis import conditional_mean, fit_covariance, matrix_path
HERE = Path(__file__).resolve().parent
DATA, TABLES, FIGURES = (HERE / 'data', HERE / 'tables', HERE / 'figures')
BUDGETS = (6, 12, 24)
RANKS = (0, 1, 2, 4, 8, 16)
MODELS = tuple((f'spectral_r{r}' for r in RANKS)) + ('full_lw', 'common_activity', 'spectral_r1_diag')
SUBSETS = ('all', 'productive', 'informative', 'productive_informative')
SEED = 20261016

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def spectral_covariance(covariance, rank, eigensystem=None):
    """Preserve the trace; retain leading eigenvalues above isotropic residual."""
    p = len(covariance)
    r = min(int(rank), p)
    if r == p:
        return (covariance.copy(), 0.0)
    if eigensystem is None:
        values, vectors = np.linalg.eigh(covariance)
        values, vectors = (values[::-1], vectors[:, ::-1])
    else:
        values, vectors = eigensystem
    tau = float(values[r:].mean())
    covariance_r = np.eye(p) * tau
    if r:
        covariance_r += vectors[:, :r] * (values[:r] - tau) @ vectors[:, :r].T
    return (covariance_r, tau)

def common_activity_covariance(covariance):
    """Frobenius projection onto a nonnegative shared offset + isotropic error.

The unit loading vector is fixed to all ones. The residual has equal diagonal
variance, so this is the common-activity analogue of the spectral-r1 model."""
    p = len(covariance)
    u = np.ones(p) / np.sqrt(p)
    alpha = float(u @ covariance @ u)
    tau = float((np.trace(covariance) - alpha) / (p - 1))
    if alpha < tau:
        alpha = tau = float(np.trace(covariance) / p)
    result = tau * np.eye(p) + (alpha - tau) * np.outer(u, u)
    return (result, dict(common_direction_variance=alpha, common_residual_variance=tau, common_offset_variance=(alpha - tau) / p))

def rank1_diagonal_covariance(covariance, eigensystem=None):
    """Retain one eigenfactor and marginalize remaining factors independently."""
    if eigensystem is None:
        values, vectors = np.linalg.eigh(covariance)
        value, vector = (values[-1], vectors[:, -1])
    else:
        values, vectors = eigensystem
        value, vector = (values[0], vectors[:, 0])
    factor = value * np.outer(vector, vector)
    residual = np.diag(covariance - factor)
    assert residual.min() >= -1e-10
    return factor + np.diag(np.maximum(residual, 0.0))

def self_check():
    rng = np.random.default_rng(71)
    a = rng.normal(size=(9, 9))
    covariance = a @ a.T + 0.2 * np.eye(9)
    checked = 0
    for r in (0, 1, 2, 4, 8, 9, 16):
        reduced, _ = spectral_covariance(covariance, r)
        assert np.allclose(reduced, reduced.T)
        assert np.linalg.eigvalsh(reduced).min() >= -1e-10
        assert np.isclose(np.trace(reduced), np.trace(covariance), rtol=1e-12)
        if r >= 9:
            assert np.array_equal(reduced, covariance)
        checked += 1
    common, _ = common_activity_covariance(covariance)
    assert np.linalg.eigvalsh(common).min() >= -1e-10
    assert np.isclose(np.trace(common), np.trace(covariance))
    assert np.allclose(np.diag(common), np.diag(common)[0])
    diagonal_residual = rank1_diagonal_covariance(covariance)
    assert np.linalg.eigvalsh(diagonal_residual).min() >= -1e-10
    assert np.allclose(np.diag(diagonal_residual), np.diag(covariance))
    train = rng.uniform(0.1, 0.7, size=(30, 9))
    train[0, 3] = np.nan
    fit = fit_covariance(train)
    full, _ = spectral_covariance(fit['cov'], 9)
    anchors = np.array([0, 2, 5])
    values = np.array([[0.2, 0.4, 0.5], [np.nan, 0.3, 0.6], [np.nan] * 3])
    reference, _ = conditional_mean(fit, values, anchors)
    actual, _ = conditional_mean(dict(mu=fit['mu'], cov=full), values, anchors)
    assert np.array_equal(actual, reference)
    isotropic, _ = spectral_covariance(fit['cov'], 0)
    pred0, _ = conditional_mean(dict(mu=fit['mu'], cov=isotropic), values, anchors)
    targets = np.setdiff1d(np.arange(9), anchors)
    assert np.allclose(pred0[:, targets], fit['mu'][targets])
    test = rng.uniform(size=(3, 9))
    test[:, anchors] = values
    poisoned = test.copy()
    poisoned[:, targets] = 1000000.0
    before, _ = conditional_mean(fit, test[:, anchors].copy(), anchors)
    after, _ = conditional_mean(fit, poisoned[:, anchors].copy(), anchors)
    assert np.array_equal(before, after)
    assert np.array_equal(actual[2], fit['mu'])
    return dict(psd_trace_rank_checks=checked, common_psd_trace=True, rank1_diagonal_psd_exact_marginal_variances=True, full_exact_equivalence=True, isotropic_targets_equal_training_means=True, hidden_target_poisoning_invariance=True, missing_anchor_fallback=True)

def score(test, predictions, anchors, targets):
    n, sse, sae, rho, regret = score_predictions(test, predictions, anchors, targets)
    y, yp = (test[:, targets], predictions[:, targets])
    finite = np.isfinite(y)
    assert np.array_equal(n, finite.sum(axis=1)) and np.all(n > 0)
    mean_y = np.sum(np.where(finite, y, 0.0), axis=1) / n
    mean_p = np.sum(np.where(finite, yp, 0.0), axis=1) / n
    dy = np.where(finite, y - mean_y[:, None], 0.0)
    dp = np.where(finite, yp - mean_p[:, None], 0.0)
    target_variance = np.sum(dy * dy, axis=1) / n
    profile_sse = np.sum((dp - dy) ** 2, axis=1)
    informative = (n >= 3) & (np.sqrt(target_variance) > 0.01)
    profile_standardized_mse = np.divide(profile_sse, n * target_variance, out=np.full(len(n), np.nan), where=informative)
    return dict(n_targets=n, sse=sse, sae=sae, rho=rho, regret=regret, profile_sse=profile_sse, target_sd=np.sqrt(target_variance), profile_standardized_mse=profile_standardized_mse, productive=np.nanmax(test, axis=1) >= 0.2, informative=informative, n_anchors_observed=np.isfinite(test[:, anchors]).sum(axis=1))

def select_subset(frame, subset):
    if subset == 'all':
        return frame
    if subset == 'productive_informative':
        return frame[frame.productive & frame.informative]
    return frame[frame[subset]]

def run_cv(folds, targets, panels, hashes):
    records, audits, alignments = ([], [], [])
    for screen, screen_folds in folds.groupby('screen', sort=True):
        path = matrix_path(screen)
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            matrix, pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        target_rows = targets[targets.screen.eq(screen)]
        target = target_rows.condition_index.to_numpy(dtype=int)
        assert list(conditions[target]) == target_rows.condition_id.tolist()
        candidates = np.setdiff1d(np.arange(len(conditions)), target)
        lookup = {c: j for j, c in enumerate(conditions)}
        for (split, repeat, fold), registry in screen_folds.groupby(['split', 'repeat', 'fold'], sort=True):
            ti = registry.pair_index.to_numpy(dtype=int)
            assert len(np.unique(ti)) == len(ti) and list(pairs[ti]) == registry.subpair.tolist()
            tr = np.setdiff1d(np.arange(len(matrix)), ti)
            if split.startswith('role'):
                role = int(split[-1]) - 1
                assert not {pairs[j].split(' >> ')[role] for j in tr} & {pairs[j].split(' >> ')[role] for j in ti}
            train, test = (matrix[tr], matrix[ti])
            fit = fit_covariance(train)
            values, vectors = np.linalg.eigh(fit['cov'])
            values, vectors = (values[::-1], vectors[:, ::-1])
            eigensystem = (values, vectors)
            v = vectors[:, 0]
            if v.sum() < 0:
                v = -v
            unit_mean = fit['mu'] / np.linalg.norm(fit['mu']) if np.linalg.norm(fit['mu']) > 0 else np.zeros(len(v))
            alignment = dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), n_conditions=len(v), leading_trace_fraction=float(values[0] / values.sum()), leading_abs_cosine_constant=float(abs(v.sum()) / np.sqrt(len(v))), leading_abs_cosine_mean_yield=float(abs(v @ unit_mean)), leading_same_sign_fraction=float(max(np.mean(v >= 0), np.mean(v <= 0))), training_shrinkage=fit['shrinkage'])
            alignments.append(alignment)
            panel = panels[panels.screen.eq(screen) & panels.split.eq(split) & panels.repeat.eq(repeat) & panels.fold.eq(fold)]
            assert len(panel) == 24 and panel.position.nunique() == 24
            order = np.array([lookup[c] for c in panel.sort_values('position').condition_id])
            assert not set(order) & set(target)
            reconstructed_order = candidates[select_panel(train[:, candidates], 24, 'variance_greedy', np.random.default_rng(SEED))]
            assert np.array_equal(order, reconstructed_order), 'Saved panel must reproduce from training candidate yields only'
            covariance_models = {}
            for r in RANKS:
                covariance_models[f'spectral_r{r}'], _ = spectral_covariance(fit['cov'], r, eigensystem)
            covariance_models['full_lw'] = fit['cov'].copy()
            covariance_models['common_activity'], common_info = common_activity_covariance(fit['cov'])
            covariance_models['spectral_r1_diag'] = rank1_diagonal_covariance(fit['cov'], eigensystem)
            fallback_count = 0
            for model, covariance in covariance_models.items():
                min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
                trace_relative_error = float(abs(np.trace(covariance) - np.trace(fit['cov'])) / max(np.trace(fit['cov']), 1e-20))
                assert min_eigenvalue >= -1e-10 and trace_relative_error < 1e-10
                reduced_fit = dict(mu=fit['mu'], cov=covariance)
                for k in BUDGETS:
                    anchors = order[:k]
                    pred, fallback = conditional_mean(reduced_fit, test[:, anchors].copy(), anchors)
                    fallback_count += fallback
                    info = dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), k=k, panel='variance_greedy', draw=0, model=model)
                    metrics = score(test, pred, anchors, target)
                    for i, pair_index in enumerate(ti):
                        records.append(dict(info, pair_index=int(pair_index), **{key: value[i] for key, value in metrics.items()}))
                audits.append(dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), model=model, training_pairs=len(train), test_pairs=len(test), training_shrinkage=fit['shrinkage'], imputed_training_cells=fit['imputed_cells'], trace=float(np.trace(covariance)), min_eigenvalue=min_eigenvalue, trace_relative_error=trace_relative_error, training_panel_reproduced=True, **common_info if model == 'common_activity' else {}))
            assert fallback_count == 0, 'Unexpected singular covariance needs investigation'
        print('RESPONSE_MODE', screen, 'scores', len(records), flush=True)
    scores = pd.DataFrame(records)
    identity = ['screen', 'split', 'repeat', 'fold', 'pair_index', 'k', 'model']
    assert not scores.duplicated(identity).any()
    matched = scores.groupby(identity[:-1])
    assert matched.model.nunique().eq(len(MODELS)).all()
    for field in ('n_targets', 'productive', 'informative', 'target_sd', 'n_anchors_observed'):
        assert matched[field].nunique().eq(1).all()
    scores.to_csv(TABLES / 'response_mode_pair_scores.csv', index=False)
    pd.DataFrame(audits).to_csv(TABLES / 'response_mode_fit_audit.csv', index=False)
    pd.DataFrame(alignments).to_csv(TABLES / 'response_mode_leading_alignment.csv', index=False)
    return (scores, pd.DataFrame(audits), pd.DataFrame(alignments))

def summarize(scores):
    rows = []
    keys = ['screen', 'split', 'k', 'model']
    for subset in SUBSETS:
        for group_keys, g in select_subset(scores, subset).groupby(keys, sort=True):
            n = int(g.n_targets.sum())
            rows.append(dict(zip(keys, group_keys), subset=subset, pairs=g.pair_index.nunique(), pair_predictions=len(g), target_predictions=n, rmse=float(np.sqrt(g.sse.sum() / n)), mae=float(g.sae.sum() / n), profile_rmse=float(np.sqrt(g.profile_sse.sum() / n)), profile_standardized_rmse=float(np.sqrt(g.profile_standardized_mse.mean())), spearman=float(g.rho.mean()), spearman_rows=int(g.rho.notna().sum()), top1_regret=float(g.regret.mean())))
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / 'response_mode_summary.csv', index=False)
    keys = ['screen', 'split', 'k', 'subset']
    full = summary[summary.model.eq('full_lw')].drop(columns='model')
    baseline = summary[~summary.model.eq('full_lw')].rename(columns={'model': 'baseline'})
    comparison = full.merge(baseline, on=keys, suffixes=('_full', '_baseline'), validate='one_to_many')
    assert comparison.target_predictions_full.eq(comparison.target_predictions_baseline).all()
    for metric in ('rmse', 'mae', 'profile_rmse', 'profile_standardized_rmse', 'spearman', 'top1_regret'):
        comparison[f'{metric}_full_minus_baseline'] = comparison[f'{metric}_full'] - comparison[f'{metric}_baseline']
    comparison.to_csv(TABLES / 'response_mode_comparison.csv', index=False)
    headlines = []
    for keys, g in comparison.groupby(['split', 'k', 'subset', 'baseline'], sort=True):
        headlines.append(dict(zip(['split', 'k', 'subset', 'baseline'], keys), screens=len(g), full_better_screens=int((g.rmse_full_minus_baseline < -1e-12).sum()), median_rmse_full_pp=float(g.rmse_full.median() * 100), median_rmse_baseline_pp=float(g.rmse_baseline.median() * 100), median_paired_rmse_gain_pp=float(-g.rmse_full_minus_baseline.median() * 100), median_paired_profile_rmse_gain_pp=float(-g.profile_rmse_full_minus_baseline.median() * 100), full_better_profile_screens=int((g.profile_rmse_full_minus_baseline < -1e-12).sum()), median_paired_spearman_gain=float(g.spearman_full_minus_baseline.median())))
    headline = pd.DataFrame(headlines)
    headline.to_csv(TABLES / 'response_mode_headline.csv', index=False)
    fractions = []
    for keys, g in summary.groupby(['screen', 'split', 'k', 'subset'], sort=True):
        q = g.set_index('model')
        total_gain = q.loc['spectral_r0', 'rmse'] ** 2 - q.loc['full_lw', 'rmse'] ** 2
        first_gain = q.loc['spectral_r0', 'rmse'] ** 2 - q.loc['spectral_r1', 'rmse'] ** 2
        fractions.append(dict(zip(['screen', 'split', 'k', 'subset'], keys), full_improves_isotropic=bool(total_gain > 1e-12), rank1_fraction_of_full_mse_gain=float(first_gain / total_gain) if total_gain > 1e-12 else np.nan, isotropic_minus_full_mse=float(total_gain), isotropic_minus_rank1_mse=float(first_gain)))
    pd.DataFrame(fractions).to_csv(TABLES / 'response_mode_gain_decomposition.csv', index=False)
    transformation = summary.copy()
    transformation['transformation'] = transformation.screen.str.split('|', regex=False).str[0].replace({'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig'})
    transformation['sse'] = transformation.rmse ** 2 * transformation.target_predictions
    transformation['profile_sse'] = transformation.profile_rmse ** 2 * transformation.target_predictions
    pooled = transformation.groupby(['transformation', 'split', 'k', 'subset', 'model'])[['sse', 'profile_sse', 'target_predictions']].sum().reset_index()
    pooled['rmse'] = np.sqrt(pooled.sse / pooled.target_predictions)
    pooled['profile_rmse'] = np.sqrt(pooled.profile_sse / pooled.target_predictions)
    pooled.to_csv(TABLES / 'response_mode_transformation_summary.csv', index=False)
    return (summary, comparison, headline)

def paired_uncertainty(scores, folds):
    rows = []
    pairs = folds.drop_duplicates(['screen', 'pair_index']).set_index(['screen', 'pair_index']).subpair
    for subset in SUBSETS:
        selected = select_subset(scores, subset)
        for keys, g in selected.groupby(['screen', 'split', 'k'], sort=True):
            screen, split, k = keys
            avg = g.groupby(['pair_index', 'model'])[['sse', 'profile_sse', 'n_targets']].mean().reset_index()
            raw = avg.pivot(index='pair_index', columns='model', values='sse')
            shape = avg.pivot(index='pair_index', columns='model', values='profile_sse')
            count = avg.pivot(index='pair_index', columns='model', values='n_targets')
            assert count.eq(count.full_lw, axis=0).all().all()
            parts = [pairs.loc[screen, j].split(' >> ') for j in raw.index]
            _, i1 = np.unique([p[0] for p in parts], return_inverse=True)
            _, i2 = np.unique([p[1] for p in parts], return_inverse=True)
            seed = SEED + int(hashlib.sha256((screen + split + str(k) + subset).encode()).hexdigest()[:7], 16)
            rng = np.random.default_rng(seed)
            weights = rng.exponential(size=(500, i1.max() + 1))[:, i1] * rng.exponential(size=(500, i2.max() + 1))[:, i2]
            counts = count.full_lw.to_numpy()
            for baseline in ('spectral_r1', 'common_activity', 'spectral_r1_diag'):
                for metric, errors in (('raw', raw), ('target_centered', shape)):
                    base = errors[baseline].to_numpy()
                    full = errors.full_lw.to_numpy()
                    draws = (np.sqrt(weights @ base / (weights @ counts)) - np.sqrt(weights @ full / (weights @ counts))) * 100
                    gain = (np.sqrt(base.sum() / counts.sum()) - np.sqrt(full.sum() / counts.sum())) * 100
                    rows.append(dict(screen=screen, split=split, k=k, subset=subset, baseline=baseline, metric=metric, pairs=len(raw), rmse_gain_pp=float(gain), node_multiplier_q025_pp=float(np.quantile(draws, 0.025)), node_multiplier_q975_pp=float(np.quantile(draws, 0.975)), draws=500))
    uncertainty = pd.DataFrame(rows)
    uncertainty.to_csv(TABLES / 'response_mode_paired_uncertainty.csv', index=False)
    return uncertainty

def check_existing_full(scores, hashes):
    path = TABLES / 'shrinkage_covariance_cv_pair_scores.csv.gz'
    hashes[path.name] = digest(path)
    reference = pd.read_csv(path)
    reference = reference[reference.panel.eq('variance_greedy') & reference.draw.eq(0)]
    current = scores[scores.model.eq('full_lw')]
    keys = ['screen', 'split', 'repeat', 'fold', 'pair_index', 'k']
    paired = current.merge(reference, on=keys, suffixes=('_current', '_existing'), validate='one_to_one')
    assert len(paired) == len(current) == len(reference)
    maxima = {}
    for metric in ('sse', 'sae', 'rho', 'regret', 'n_targets', 'n_anchors_observed'):
        x, y = (paired[f'{metric}_current'], paired[f'{metric}_existing'])
        assert np.allclose(x, y, rtol=1e-10, atol=1e-12, equal_nan=True)
        maxima[metric] = float((x - y).abs().max())
    return dict(matched_pair_scores=len(paired), maximum_absolute_differences=maxima)

def plot(summary, comparison, alignments):
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    order = [f'spectral_r{r}' for r in RANKS] + ['full_lw']
    labels = ['0', '1', '2', '4', '8', '16', 'full']
    colors = {6: '#1f7a8c', 12: '#a94b5b', 24: '#4a8050'}
    for ax, subset, metric, title in ((axs[0, 0], 'all', 'rmse', 'A  Held-out pairs: all profiles'), (axs[0, 1], 'informative', 'profile_rmse', 'B  Variable targets: centered profile error')):
        for k in BUDGETS:
            q = summary[summary.split.eq('pair') & summary.subset.eq(subset) & summary.k.eq(k)]
            medians = q.groupby('model')[metric].median().reindex(order) * 100
            ax.plot(np.arange(len(order)), medians, 'o-', color=colors[k], label=f'{k} anchors')
        ax.set(title=title, xlabel='Retained covariance modes + isotropic residual', ylabel='Median screen RMSE (yield pp)')
        ax.set_xticks(np.arange(len(order)), labels)
        ax.legend(frameon=False)
    ax = axs[1, 0]
    q = comparison[comparison.split.eq('pair') & comparison.subset.eq('informative') & comparison.baseline.eq('spectral_r1_diag')]
    names = q[q.k.eq(24)].sort_values('rmse_full_minus_baseline').screen.tolist()
    for k, offset in zip(BUDGETS, (-0.22, 0.0, 0.22)):
        data = q[q.k.eq(k)].set_index('screen').reindex(names)
        ax.scatter(-data.rmse_full_minus_baseline * 100, np.arange(len(names)) + offset, color=colors[k], s=20, label=str(k))
    ax.axvline(0, color='#999999', linewidth=0.8)
    ax.set_yticks(np.arange(len(names)), [s.replace('|1', '') for s in names], fontsize=7)
    ax.set(title='C  Variable targets: full vs one factor + diagonal', xlabel='Paired RMSE gain from full model (yield pp)')
    ax.legend(title='Anchors', frameon=False, fontsize=8)
    ax = axs[1, 1]
    med = alignments[alignments.split.eq('pair')].groupby('screen').median(numeric_only=True)
    ax.scatter(med.leading_abs_cosine_constant, med.leading_abs_cosine_mean_yield, s=35, color='#1f7a8c')
    offsets = {'Hiyama|1': (-18, -25), 'Acylation|1': (-32, -38), 'Suzuki|1': (7, -18)}
    for screen in ('Suzuki|1', 'Acylation|1', 'Hiyama|1'):
        row = med.loc[screen]
        ax.annotate(screen.replace('|1', ''), (row.leading_abs_cosine_constant, row.leading_abs_cosine_mean_yield), xytext=offsets[screen], textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='-', color='#888888', linewidth=0.6))
    ax.set(xlim=(0, 1.05), ylim=(0, 1.05), title='D  Leading-mode orientation (training folds)', xlabel='Absolute cosine with constant loading', ylabel='Absolute cosine with mean condition yield')
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / '07_response_mode_ablation.png', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    started = time.time()
    with threadpool_limits(limits=1):
        checks = self_check()
        if args.self_check_only:
            print(json.dumps(checks), flush=True)
            return
        paths = [DATA / 'cv_fold_registry.csv', DATA / 'cv_target_registry.csv', TABLES / 'training_selected_panels.csv']
        hashes = {p.name: digest(p) for p in paths}
        folds, targets, all_panels = (pd.read_csv(p) for p in paths)
        panels = all_panels[all_panels.panel.eq('variance_greedy') & all_panels.draw.eq(0)]
        scores, audits, alignments = run_cv(folds, targets, panels, hashes)
        equivalent = check_existing_full(scores, hashes)
        summary, comparison, headline = summarize(scores)
        uncertainty = paired_uncertainty(scores, folds)
        plot(summary, comparison, alignments)
    for path in paths:
        assert digest(path) == hashes[path.name], f'Registry changed during run: {path}'
    outputs = sorted(TABLES.glob('response_mode_*.csv')) + [FIGURES / '07_response_mode_ablation.png']
    manifest = dict(date='2026-10-16', status='Exploratory response-information sensitivity after existing findings; not prospective confirmation', method='Train-only sklearn LedoitWolf; spectral-r covariance=tau*I + U_r diag(lambda_r-tau) U_r.T, tau=mean discarded eigenvalues; full=exact LW', rank_interpretation='r counts retained anisotropic deviations; residual covariance generally full-rank; neither physical dimension nor hard intrinsic rank', common_activity='All-ones loading; Frobenius projection of LW covariance onto shared nonnegative random offset plus isotropic diagonal residual; trace preserved', spectral_rank1_diagonal='Leading LW eigenfactor lambda1*u1*u1.T plus diagonal of discarded covariance; PSD, exact training marginal variances and trace; no extra off-diagonal factors', centering='Model uses raw yields centered only by training condition means; no test-row centering in prediction', profile_scoring='Predicted and observed fixed targets each centered using finite target scoring mask only; targets never passed to prediction', missingness='Training-only condition-mean imputation; finite test anchors only; unobserved targets excluded from scoring', subsets=dict(all='All held-out pairs', productive='Maximum across all observed test-row conditions >=0.20; retrospective descriptive subset', informative='At least 3 finite audit targets and their population SD >0.01; retrospective scoring subset', productive_informative='Intersection of productive and informative'), matching='Same targets, observed target cells, folds and variance_greedy draw0 anchors for all modes; saved panels independently reproduced from each training fold', uncertainty='500 two-role molecular-node exponential multiplier draws on paired fixed-fit errors; repeated pair CV averaged per pair; no refitting, target resampling, or multiplicity correction', leakage_checks='Fold pair identity and role disjointness; targets excluded from anchors; all panels recomputed train-only; conditional_mean receives anchor columns only; hidden-target poisoning invariance', budgets=BUDGETS, ranks=list(RANKS) + ['full'], models=MODELS, self_checks=checks, existing_full_equivalence=equivalent, screens=int(scores.screen.nunique()), unique_pairs=int(scores[['screen', 'pair_index']].drop_duplicates().shape[0]), folds=int(alignments.shape[0]), fit_audit_rows=len(audits), pair_score_rows=len(scores), summary_rows=len(summary), uncertainty_rows=len(uncertainty), split_fit_counts=alignments.groupby('split').size().astype(int).to_dict(), maximum_trace_relative_error=float(audits.trace_relative_error.max()), minimum_eigenvalue=float(audits.min_eigenvalue.min()), input_sha256=hashes, code_sha256={p.name: digest(p) for p in [Path(__file__), HERE / 'shrinkage_covariance_analysis.py', HERE / 'run_existing_evidence.py', HERE / 'dimension_core.py']}, output_sha256={p.name: digest(p) for p in outputs}, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, seconds=time.time() - started)
    (DATA / 'response_mode_manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='utf-8')
    print('FINISHED_RESPONSE_MODE', json.dumps({key: manifest[key] for key in ['screens', 'folds', 'pair_score_rows', 'seconds']}), flush=True)
    print(headline[headline.subset.eq('all') & headline.baseline.eq('spectral_r1')].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
