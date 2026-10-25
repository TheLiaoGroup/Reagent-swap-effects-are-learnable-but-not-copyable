"""Train-only expected incremental risk beyond shared-response baselines."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from response_mode_ablation import common_activity_covariance, rank1_diagonal_covariance
from shrinkage_covariance_analysis import fit_covariance, matrix_path
HERE = Path(__file__).resolve().parent
DATA, TABLES = (HERE / 'data', HERE / 'tables')
BASELINES = ('spectral_r1_diag', 'common_activity')
BUDGETS = (6, 12, 24)
KEYS = ['screen', 'repeat', 'fold', 'pair_index', 'k']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def expected_risk(full, baseline, targets, anchors):
    """Per-target linear-prediction error under the full training covariance."""
    stt = full[np.ix_(targets, targets)]
    if not len(anchors):
        return (np.diag(stt).copy(), np.diag(stt).copy(), 0.0)
    saa = full[np.ix_(anchors, anchors)]
    sta = full[np.ix_(targets, anchors)]
    optimum = np.linalg.solve(saa, sta.T).T
    weights = np.linalg.solve(baseline[np.ix_(anchors, anchors)], baseline[np.ix_(targets, anchors)].T).T
    schur = stt - optimum @ sta.T
    general_full = stt - optimum @ sta.T - sta @ optimum.T + optimum @ saa @ optimum.T
    general_baseline = stt - weights @ sta.T - sta @ weights.T + weights @ saa @ weights.T
    full_risk, baseline_risk = (np.diag(schur), np.diag(general_baseline))
    equivalence = float(np.max(abs(schur - general_full)))
    assert equivalence < 1e-10
    assert full_risk.min() >= -1e-10
    assert np.all(full_risk <= baseline_risk + 1e-10)
    return (np.maximum(full_risk, 0.0), np.maximum(baseline_risk, 0.0), equivalence)

def self_check():
    rng = np.random.default_rng(84)
    a = rng.normal(size=(8, 8))
    covariance = a @ a.T + np.eye(8)
    for anchors in (np.array([], dtype=int), np.array([0]), np.array([0, 2, 4])):
        targets = np.array([1, 3, 5, 7])
        for baseline in (rank1_diagonal_covariance(covariance), common_activity_covariance(covariance)[0], covariance):
            full, base, _ = expected_risk(covariance, baseline, targets, anchors)
            assert np.all(full <= base + 1e-10)
        full, base, _ = expected_risk(covariance, covariance, targets, anchors)
        assert np.allclose(full, base)
    return dict(schur_general_formula_equivalence=True, full_optimal_linear_risk=True, equal_full_coefficients_equal_risk=True, zero_anchor_equal_risk=True)

def summarize(rows):
    summaries = []
    for subset in ('all', 'productive', 'informative'):
        selected = rows if subset == 'all' else rows[rows[subset]]
        for unit in ('screen', 'family'):
            selected = selected.copy()
            selected['family'] = selected.screen.str.split('|', regex=False).str[0].replace({'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig'})
            for keys, g in selected.groupby([unit, 'k', 'baseline'], sort=True):
                expected_full, expected_base = (g.expected_sse_full.sum(), g.expected_sse_baseline.sum())
                actual_full, actual_base = (g.actual_sse_full.sum(), g.actual_sse_baseline.sum())
                summaries.append(dict(unit=unit, identity=keys[0], k=int(keys[1]), baseline=keys[2], subset=subset, screens=g.screen.nunique(), pairs=g[['screen', 'pair_index']].drop_duplicates().shape[0], pair_predictions=len(g), target_predictions=int(g.n_targets.sum()), expected_sse_full=float(expected_full), expected_sse_baseline=float(expected_base), actual_sse_full=float(actual_full), actual_sse_baseline=float(actual_base), expected_fractional_extra_gain=float(1 - expected_full / expected_base), actual_fractional_extra_gain=float(1 - actual_full / actual_base), predicted_minus_actual_gain=float(expected_full / expected_base - actual_full / actual_base) * -1))
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / 'response_specific_risk_summary.csv', index=False)
    associations = []
    for keys, g in summary.groupby(['unit', 'k', 'baseline', 'subset'], sort=True):
        result = spearmanr(g.expected_fractional_extra_gain, g.actual_fractional_extra_gain)
        associations.append(dict(zip(['unit', 'k', 'baseline', 'subset'], keys), n=len(g), spearman=float(result.statistic), descriptive_p=float(result.pvalue), median_expected_extra_gain=float(g.expected_fractional_extra_gain.median()), median_actual_extra_gain=float(g.actual_fractional_extra_gain.median()), actual_positive_units=int(g.actual_fractional_extra_gain.gt(0).sum()), median_absolute_gain_prediction_error=float((g.expected_fractional_extra_gain - g.actual_fractional_extra_gain).abs().median())))
    association = pd.DataFrame(associations)
    association.to_csv(TABLES / 'response_specific_risk_associations.csv', index=False)
    return (summary, association)

def main():
    started = time.time()
    checks = self_check()
    paths = [DATA / 'cv_fold_registry.csv', DATA / 'cv_target_registry.csv', TABLES / 'training_selected_panels.csv', TABLES / 'response_mode_pair_scores.csv']
    hashes = {p.name: digest(p) for p in paths}
    folds, target_registry, panels = (pd.read_csv(p) for p in paths[:3])
    folds = folds[folds.split.eq('pair')]
    panels = panels[panels.split.eq('pair') & panels.panel.eq('variance_greedy') & panels.draw.eq(0)]
    rows, fits, max_equivalence = ([], 0, 0.0)
    with threadpool_limits(limits=1):
        for screen, sf in folds.groupby('screen', sort=True):
            path = matrix_path(screen)
            hashes[path.name] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                matrix, pairs, conditions = (z['M'], z['sps'].astype(str), z['conds'].astype(str))
            targets = target_registry[target_registry.screen.eq(screen)].condition_index.to_numpy(dtype=int)
            condition_lookup = {c: i for i, c in enumerate(conditions)}
            for (repeat, fold), registry in sf.groupby(['repeat', 'fold'], sort=True):
                test_indices = registry.pair_index.to_numpy(dtype=int)
                assert list(pairs[test_indices]) == registry.subpair.tolist()
                training_indices = np.setdiff1d(np.arange(len(matrix)), test_indices)
                assert not set(training_indices) & set(test_indices)
                full = fit_covariance(matrix[training_indices])['cov']
                baselines = dict(spectral_r1_diag=rank1_diagonal_covariance(full), common_activity=common_activity_covariance(full)[0])
                panel = panels[panels.screen.eq(screen) & panels.repeat.eq(repeat) & panels.fold.eq(fold)].sort_values('position')
                order = np.array([condition_lookup[c] for c in panel.condition_id])
                assert len(order) == 24 and (not set(order) & set(targets))
                for k in BUDGETS:
                    anchors = order[:k]
                    masks = np.isfinite(matrix[np.ix_(test_indices, anchors)])
                    patterns, membership = np.unique(masks, axis=0, return_inverse=True)
                    for mask_number, observed in enumerate(patterns):
                        selected = np.flatnonzero(membership == mask_number)
                        observed_anchors = anchors[observed]
                        target_masks = np.isfinite(matrix[np.ix_(test_indices[selected], targets)])
                        for baseline, baseline_covariance in baselines.items():
                            risk_full, risk_baseline, equivalence = expected_risk(full, baseline_covariance, targets, observed_anchors)
                            max_equivalence = max(max_equivalence, equivalence)
                            expected_full, expected_baseline = (target_masks @ risk_full, target_masks @ risk_baseline)
                            assert np.all(expected_full <= expected_baseline + 1e-10)
                            for i, index in enumerate(selected):
                                rows.append(dict(screen=screen, repeat=int(repeat), fold=int(fold), pair_index=int(test_indices[index]), k=k, baseline=baseline, n_targets=int(target_masks[i].sum()), n_anchors_observed=int(observed.sum()), expected_sse_full=float(expected_full[i]), expected_sse_baseline=float(expected_baseline[i])))
                fits += 1
    risk_rows = pd.DataFrame(rows)
    scores = pd.read_csv(paths[3], usecols=KEYS + ['split', 'model', 'n_targets', 'sse', 'productive', 'informative', 'n_anchors_observed'])
    scores = scores[scores.split.eq('pair')]
    full_scores = scores[scores.model.eq('full_lw')][KEYS + ['sse', 'productive', 'informative', 'n_targets', 'n_anchors_observed']].rename(columns={'sse': 'actual_sse_full'})
    base_scores = scores[scores.model.isin(BASELINES)][KEYS + ['model', 'sse']].rename(columns={'model': 'baseline', 'sse': 'actual_sse_baseline'})
    risk_rows = risk_rows.merge(full_scores, on=KEYS, validate='many_to_one', suffixes=('', '_score'))
    risk_rows = risk_rows.merge(base_scores, on=KEYS + ['baseline'], validate='one_to_one')
    for field in ('n_targets', 'n_anchors_observed'):
        assert risk_rows[field].eq(risk_rows[field + '_score']).all()
        risk_rows = risk_rows.drop(columns=field + '_score')
    assert len(risk_rows) == 28320 and fits == 180
    risk_rows.to_csv(TABLES / 'response_specific_risk_pair_scores.csv', index=False)
    summary, associations = summarize(risk_rows)
    for p in paths:
        assert digest(p) == hashes[p.name]
    outputs = sorted(TABLES.glob('response_specific_risk_*.csv'))
    manifest = dict(date='2026-10-16', status='Exploratory fixed-fit train-only model-risk sensitivity, after information-frontier analysis', method='Baseline W=Sigma_b[T,A] inv(Sigma_b[A,A]); risk_b=diag(Sigma_full[T,T]-W Sigma_full[A,T]-Sigma_full[T,A] W.T+W Sigma_full[A,A] W.T); full risk is Schur complement', gain='1 - sum(expected_full_risk)/sum(expected_baseline_risk); empirical counterpart 1-SSE_full/SSE_baseline on exactly matching rows and target cells', leakage='180 existing pair-CV training complements only; original targets/variance_greedy draw0 anchors; each test-row finite anchor/target masks used, test target values never enter expected risk', baselines=BASELINES, budgets=BUDGETS, fits=fits, paired_risk_records=len(risk_rows), subset_interpretation='all; productive max observed test-row yield>=.2; informative >=3 finite audit targets with SD>.01; latter two retrospective descriptive scoring only', pooling='Ratios of summed risk/errors, not averages of row ratios; family pools Buchwald-Hartwig and CMC screens; repeated pair-CV predictions retained equally', limits='Training estimated covariance plus mean-imputed missing cells; ideal unbounded linear risk compared with clipped actual predictions; mean shift/covariance misspecification/estimation and target selection affect calibration; descriptive ranks, no refitting uncertainty or multiplicity correction', self_checks=checks, maximum_full_schur_equivalence_error=max_equivalence, full_risk_le_baseline_every_evaluated_target=True, input_sha256=hashes, code_sha256={p.name: digest(p) for p in [Path(__file__), HERE / 'response_mode_ablation.py', HERE / 'shrinkage_covariance_analysis.py']}, output_sha256={p.name: digest(p) for p in outputs}, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, seconds=time.time() - started)
    (DATA / 'response_specific_risk_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(associations.to_string(index=False), flush=True)
    print('FINISHED_RESPONSE_SPECIFIC_RISK', fits, len(risk_rows), time.time() - started, flush=True)
if __name__ == '__main__':
    main()
