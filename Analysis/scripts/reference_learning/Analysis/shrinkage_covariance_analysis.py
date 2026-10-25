"""One exploratory Ledoit-Wolf conditional-mean sensitivity, with fixed audits.

The method is added after inspecting the primary model results. It is not a
confirmatory model choice. Existing folds, targets, and anchor panels are reused."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import sklearn
from threadpoolctl import threadpool_limits
from run_existing_evidence import score_predictions
HERE = Path(__file__).resolve().parent
DATA, TABLES = (HERE / 'data', HERE / 'tables')
BUDGETS = (6, 12, 24)
MODEL = 'ledoit_wolf'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def matrix_path(screen):
    return DATA / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')

def fit_covariance(train):
    means = np.nanmean(train, axis=0)
    assert np.isfinite(means).all(), 'A training condition has no finite observations'
    imputed = np.where(np.isfinite(train), train, means)
    estimator = LedoitWolf().fit(imputed)
    assert np.isfinite(estimator.covariance_).all()
    return dict(mu=estimator.location_, cov=estimator.covariance_, shrinkage=float(estimator.shrinkage_), imputed_cells=int((~np.isfinite(train)).sum()))

def conditional_mean(fit, anchor_values, anchors):
    """The test input contains anchor columns only; hidden values are inaccessible."""
    anchors = np.asarray(anchors, dtype=int)
    anchor_values = np.asarray(anchor_values, dtype=float)
    assert anchor_values.ndim == 2 and anchor_values.shape[1] == len(anchors)
    mu, cov = (fit['mu'], fit['cov'])
    out = np.tile(mu, (len(anchor_values), 1))
    masks, membership = np.unique(np.isfinite(anchor_values), axis=0, return_inverse=True)
    fallbacks = 0
    for mask_number, observed in enumerate(masks):
        if not observed.any():
            continue
        rows = np.flatnonzero(membership == mask_number)
        observed_anchors = anchors[observed]
        caa = cov[np.ix_(observed_anchors, observed_anchors)]
        cross = cov[:, observed_anchors]
        residual = anchor_values[np.ix_(rows, observed)] - mu[observed_anchors]
        try:
            weights = np.linalg.solve(caa, residual.T)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(caa) @ residual.T
            fallbacks += 1
        out[rows] += (cross @ weights).T
    return (np.clip(out, 0.0, 1.0), fallbacks)

def self_check():
    fit = dict(mu=np.array([0.2, 0.4, 0.6]), cov=np.array([[0.1, 0.02, 0.03], [0.02, 0.2, 0.04], [0.03, 0.04, 0.3]]))
    anchors = np.array([0, 1])
    values = np.array([[0.3, 0.7], [np.nan, 0.5], [np.nan, np.nan]])
    actual, fallback = conditional_mean(fit, values, anchors)
    expected = fit['mu'] + fit['cov'][:, anchors] @ np.linalg.solve(fit['cov'][np.ix_(anchors, anchors)], values[0] - fit['mu'][anchors])
    assert np.allclose(actual[0], np.clip(expected, 0, 1))
    assert np.allclose(actual[1], np.clip(fit['mu'] + fit['cov'][:, 1] / 0.2 * 0.1, 0, 1))
    assert np.array_equal(actual[2], fit['mu']) and fallback == 0

def score_rows(test, pred, anchors, targets, records, extra):
    counts, sse, sae, rho, regret = score_predictions(test, pred, anchors, targets)
    expected = np.isfinite(test[:, targets]).sum(axis=1)
    assert np.array_equal(counts, expected)
    rows = []
    for index, record in enumerate(records):
        rows.append(dict(record, **extra, model=MODEL, n_targets=int(counts[index]), n_anchors_observed=int(np.isfinite(test[index, anchors]).sum()), sse=float(sse[index]), sae=float(sae[index]), rho=float(rho[index]) if counts[index] else np.nan, regret=float(regret[index]) if counts[index] else np.nan, productive=bool(np.nanmax(test[index]) >= 0.2)))
    return rows

def summarize(scores, historical=False):
    keys = ['screen', 'partition' if historical else 'split', 'panel', 'k', 'model']
    rows = []
    for subset in ('all', 'productive'):
        selected = scores if subset == 'all' else scores[scores.productive]
        for group_keys, group in selected.groupby(keys, sort=True):
            n = int(group.n_targets.sum())
            pair_field = 'subpair' if historical else 'pair_index'
            rows.append(dict(zip(keys, group_keys), subset=subset, pairs=group[pair_field].nunique(), pair_predictions=len(group), target_predictions=n, rmse=float(np.sqrt(group.sse.sum() / n)) if n else np.nan, mae=float(group.sae.sum() / n) if n else np.nan, spearman=float(group.rho.mean()), spearman_rows=int(group.rho.notna().sum()), top1_regret=float(group.regret.mean()), mean_observed_anchors=float(group.n_anchors_observed.mean())))
    return pd.DataFrame(rows)

def compare(summary, reference_path, output_path, historical=False):
    if not reference_path.exists():
        return dict(status='reference_not_available')
    primary = pd.read_csv(reference_path)
    primary = primary[primary.k.isin(BUDGETS)]
    keys = ['screen', 'partition' if historical else 'split', 'panel', 'k', 'subset']
    baseline = primary[primary.model.isin(['additive', 'affine', 'low_rank'])].copy()
    baseline = baseline.rename(columns={'model': 'baseline_model'})
    merged = summary.merge(baseline[keys + ['baseline_model', 'rmse', 'mae', 'spearman', 'top1_regret', 'target_predictions']], on=keys, suffixes=('_lw', '_baseline'), validate='one_to_many')
    assert merged.target_predictions_lw.eq(merged.target_predictions_baseline).all()
    for metric in ('rmse', 'mae', 'spearman', 'top1_regret'):
        merged[f'{metric}_lw_minus_baseline'] = merged[f'{metric}_lw'] - merged[f'{metric}_baseline']
    merged.to_csv(output_path, index=False)
    selected = merged[merged.subset.eq('all') & merged.panel.eq('variance_greedy') & merged.baseline_model.eq('affine')]
    summaries = []
    for group, rows in selected.groupby([keys[1], 'k']):
        summaries.append(dict(regime=group[0], k=int(group[1]), screens=len(rows), rmse_better_screens=int(rows.rmse_lw.lt(rows.rmse_baseline).sum()), median_rmse_pp=float(rows.rmse_lw.median() * 100), median_rmse_change_pp=float(rows.rmse_lw_minus_baseline.median() * 100)))
    return dict(status='compared', variance_greedy_vs_affine=summaries)

def run_cv(folds, targets, panels, input_hashes):
    records, audits = ([], [])
    for screen, screen_folds in folds.groupby('screen', sort=True):
        path = matrix_path(screen)
        input_hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            matrix, pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        condition_lookup = {condition: j for j, condition in enumerate(conditions)}
        target_rows = targets[targets.screen.eq(screen)]
        target = target_rows.condition_index.to_numpy(dtype=int)
        assert list(conditions[target]) == target_rows.condition_id.tolist()
        for (split, repeat, fold), test_registry in screen_folds.groupby(['split', 'repeat', 'fold'], sort=True):
            test_indices = test_registry.pair_index.to_numpy(dtype=int)
            assert len(set(test_indices)) == len(test_indices)
            assert list(pairs[test_indices]) == test_registry.subpair.tolist()
            train_indices = np.setdiff1d(np.arange(len(matrix)), test_indices)
            assert not set(train_indices) & set(test_indices)
            if split.startswith('role'):
                role = int(split[-1]) - 1
                train_roles = {pairs[j].split(' >> ')[role] for j in train_indices}
                test_roles = {pairs[j].split(' >> ')[role] for j in test_indices}
                assert not train_roles & test_roles
            train, test = (matrix[train_indices], matrix[test_indices])
            fit = fit_covariance(train)
            fallback_count = 0
            choices = panels[panels.screen.eq(screen) & panels.split.eq(split) & panels.repeat.eq(repeat) & panels.fold.eq(fold)]
            for (panel, draw), ordered in choices.groupby(['panel', 'draw'], sort=True):
                order = np.array([condition_lookup[c] for c in ordered.sort_values('position').condition_id])
                assert not set(order) & set(target)
                for k in BUDGETS:
                    assert len(order) >= k
                    anchors = order[:k]
                    pred, fallback = conditional_mean(fit, test[:, anchors].copy(), anchors)
                    fallback_count += fallback
                    info = dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), panel=panel, draw=int(draw), k=k, training_shrinkage=fit['shrinkage'])
                    pair_records = [dict(pair_index=int(index)) for index in test_indices]
                    records.extend(score_rows(test, pred, anchors, target, pair_records, info))
            audits.append(dict(screen=screen, split=split, repeat=int(repeat), fold=int(fold), training_pairs=len(train), test_pairs=len(test), training_shrinkage=fit['shrinkage'], imputed_training_cells=fit['imputed_cells'], singular_pinv_fallback_masks=fallback_count))
        print('LW_CV', screen, 'cumulative_scores', len(records), flush=True)
    scores = pd.DataFrame(records)
    identity = ['screen', 'split', 'repeat', 'fold', 'pair_index', 'panel', 'draw', 'k', 'model']
    assert not scores.duplicated(identity).any()
    assert scores.groupby(['screen', 'split', 'repeat', 'fold', 'pair_index']).n_targets.nunique().max() == 1
    scores.to_csv(TABLES / 'shrinkage_covariance_cv_pair_scores.csv.gz', index=False)
    summary = summarize(scores)
    summary.to_csv(TABLES / 'shrinkage_covariance_cv_summary.csv', index=False)
    pd.DataFrame(audits).to_csv(TABLES / 'shrinkage_covariance_cv_fit_audit.csv', index=False)
    comparison = compare(summary, TABLES / 'cv_budget_summary.csv', TABLES / 'shrinkage_covariance_cv_comparison.csv')
    return dict(pair_score_rows=len(scores), unique_screen_pairs=scores[['screen', 'pair_index']].drop_duplicates().shape[0], fit_count=len(audits), singular_fallback_masks=sum((x['singular_pinv_fallback_masks'] for x in audits)), comparison=comparison)

def run_historical(targets, input_hashes):
    panel_path = TABLES / 'historical_holdout_panels.csv'
    observation_path = DATA / 'qualified_observations.csv.gz'
    if not panel_path.exists():
        return dict(status='not_run_no_saved_historical_panels')
    input_hashes[panel_path.name] = digest(panel_path)
    input_hashes[observation_path.name] = digest(observation_path)
    panels = pd.read_csv(panel_path)
    observations = pd.read_csv(observation_path, usecols=['screen', 'partition', 'subpair', 'condition_id', 'yield'])
    records = []
    for screen, panel_rows in panels.groupby('screen', sort=True):
        with np.load(matrix_path(screen), allow_pickle=False) as saved:
            train, train_pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        fit = fit_covariance(train)
        lookup = {condition: j for j, condition in enumerate(conditions)}
        target = targets.loc[targets.screen.eq(screen), 'condition_index'].to_numpy(dtype=int)
        for partition in ('sealed', 'reserve'):
            heldout = observations[observations.screen.eq(screen) & observations.partition.eq(partition)]
            if heldout.empty:
                continue
            test_pairs = sorted(heldout.subpair.unique())
            assert not set(train_pairs) & set(test_pairs)
            test = heldout.pivot(index='subpair', columns='condition_id', values='yield').reindex(index=test_pairs, columns=conditions).to_numpy(dtype=float)
            for (panel, draw), ordered in panel_rows.groupby(['panel', 'draw'], sort=True):
                order = np.array([lookup[c] for c in ordered.sort_values('position').condition_id])
                assert not set(order) & set(target)
                for k in BUDGETS:
                    anchors = order[:k]
                    pred, _ = conditional_mean(fit, test[:, anchors].copy(), anchors)
                    info = dict(screen=screen, partition=partition, panel=panel, draw=int(draw), k=k, training_shrinkage=fit['shrinkage'], training_pairs=len(train))
                    pair_records = [dict(pair_index=j, subpair=pair) for j, pair in enumerate(test_pairs)]
                    records.extend(score_rows(test, pred, anchors, target, pair_records, info))
    scores = pd.DataFrame(records)
    assert not scores.duplicated(['screen', 'partition', 'subpair', 'panel', 'draw', 'k', 'model']).any()
    scores.to_csv(TABLES / 'shrinkage_covariance_historical_pair_scores.csv.gz', index=False)
    summary = summarize(scores, historical=True)
    summary.to_csv(TABLES / 'shrinkage_covariance_historical_summary.csv', index=False)
    comparison = compare(summary, TABLES / 'historical_holdout_summary.csv', TABLES / 'shrinkage_covariance_historical_comparison.csv', historical=True)
    return dict(status='completed', pair_score_rows=len(scores), unique_screen_pairs=scores[['screen', 'partition', 'subpair']].drop_duplicates().shape[0], comparison=comparison)

def main():
    started = time.time()
    self_check()
    fold_path, target_path, panel_path = (DATA / 'cv_fold_registry.csv', DATA / 'cv_target_registry.csv', TABLES / 'training_selected_panels.csv')
    paths = [fold_path, target_path, panel_path]
    hashes = {p.name: digest(p) for p in paths}
    folds, targets, panels = (pd.read_csv(p) for p in paths)
    for reference in (TABLES / 'cv_budget_summary.csv', TABLES / 'historical_holdout_summary.csv'):
        if reference.exists():
            hashes[reference.name] = digest(reference)
    with threadpool_limits(limits=1):
        cv = run_cv(folds, targets, panels, hashes)
        historical = run_historical(targets, hashes)
    for path in paths:
        assert digest(path) == hashes[path.name], f'Input registry changed during run: {path}'
    manifest = dict(date='2026-10-16', status='Exploratory model sensitivity added after primary results; not outcome-blind confirmation', method='sklearn.covariance.LedoitWolf, default data-estimated isotropic shrinkage; raw-yield covariance conditional mean', training_missingness='Column means computed within each training fold only; fill training missing cells with those means', inference='Prediction function receives test anchors only; missing anchors omitted per pair; conditional mean clipped to [0,1]', targets='Exactly the existing fixed audit targets, excluded from all candidate anchor panels', model_selection='One extra standard model; no parameter grid, no outcome-based tuning or best-draw selection', regret='Top-1 regret only within finite fixed audit targets; not full-library final yield regret', productive='Descriptive subset based on max complete observed test-row yield >=0.20; not a prospective eligibility rule', budgets=BUDGETS, cv=cv, historical=historical, input_sha256=hashes, code_sha256={p.name: digest(p) for p in [Path(__file__), HERE / 'run_existing_evidence.py']}, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, seconds=time.time() - started)
    (TABLES / 'shrinkage_covariance_manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='utf-8')
    restored = pd.read_csv(TABLES / 'shrinkage_covariance_cv_pair_scores.csv.gz')
    assert len(restored) == cv['pair_score_rows'] and np.isfinite(restored.sse).all()
    print('FINISHED_LW', json.dumps(manifest, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
