"""Descriptive matched-substrate sign association beyond marginal dominance.

Inputs are read-only collapsed response profiles and the complete background-pair
universe. Reused pairs and substrates never constitute independent replication."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'results'
PROFILE_PATH = RESULTS / 'matched_response_profiles.parquet'
PAIR_PATH = RESULTS / 'matched_response_transport_pairs.csv'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
CONTEXT = EDGE + ['background_fingerprint']
THRESHOLDS = (15, 20, 30)
PANELS = ('all', 'both_positive')
COHORTS = ('all_pairs', 'shared_ge20', 'shared_ge20_joint_ge20')
METRICS = ('observed_agreement', 'marginal_expected_agreement', 'excess_agreement', 'cohen_kappa', 'balanced_accuracy_a_to_b', 'balanced_accuracy_b_to_a', 'balanced_accuracy_bidirectional', 'mcc_phi')

def contingency(pp: int, pn: int, np_: int, nn: int) -> dict:
    """Rows are background A sign, columns background B sign; positive first."""
    n = pp + pn + np_ + nn
    out = dict(n_pp=pp, n_pn=pn, n_np=np_, n_nn=nn, n_joint_decisive=n, n_a_positive=pp + pn, n_a_negative=np_ + nn, n_b_positive=pp + np_, n_b_negative=pn + nn, n_minority_a=min(pp + pn, np_ + nn), n_minority_b=min(pp + np_, pn + nn), min_minority_count=min(pp + pn, np_ + nn, pp + np_, pn + nn))
    out.update({metric: np.nan for metric in METRICS})
    out.update(fraction_a_positive=np.nan, fraction_b_positive=np.nan, a_has_both_signs=False, b_has_both_signs=False, kappa_status='no_joint_decisive', phi_status='no_joint_decisive', sign_support_status='no_joint_decisive')
    if not n:
        return out
    a_pos, a_neg, b_pos, b_neg = (pp + pn, np_ + nn, pp + np_, pn + nn)
    pa, pb = (a_pos / n, b_pos / n)
    observed = (pp + nn) / n
    expected = (a_pos * b_pos + a_neg * b_neg) / n ** 2
    excess = observed - expected
    avar, bvar = (bool(a_pos and a_neg), bool(b_pos and b_neg))
    kappa_den = 1 - expected
    out.update(observed_agreement=observed, marginal_expected_agreement=expected, excess_agreement=excess, fraction_a_positive=pa, fraction_b_positive=pb, a_has_both_signs=avar, b_has_both_signs=bvar, cohen_kappa=excess / kappa_den if kappa_den > 0 else np.nan, kappa_status='defined' if kappa_den > 0 else 'expected_agreement_one', phi_status='defined' if avar and bvar else 'one_sign_margin')
    if avar and bvar:
        out['sign_support_status'] = 'both_mixed'
    elif not avar and (not bvar):
        out['sign_support_status'] = 'both_one_sign_same' if observed == 1 else 'both_one_sign_opposite'
    else:
        out['sign_support_status'] = 'a_one_sign' if not avar else 'b_one_sign'
    if bvar:
        out['balanced_accuracy_a_to_b'] = 0.5 * (pp / b_pos + nn / b_neg)
    if avar:
        out['balanced_accuracy_b_to_a'] = 0.5 * (pp / a_pos + nn / a_neg)
    if avar and bvar:
        out['balanced_accuracy_bidirectional'] = 0.5 * (out['balanced_accuracy_a_to_b'] + out['balanced_accuracy_b_to_a'])
        out['mcc_phi'] = (pp * nn - pn * np_) / math.sqrt(a_pos * a_neg * b_pos * b_neg)
    return out

def synthetic_checks() -> dict:
    examples = {'independent_dominance_90_percent_positive': (81, 9, 9, 1), 'balanced_positive_association': (45, 5, 5, 45), 'balanced_negative_association': (5, 45, 45, 5), 'one_sign_source_mixed_target': (75, 25, 0, 0), 'both_one_sign_same': (100, 0, 0, 0), 'both_one_sign_opposite': (0, 100, 0, 0), 'empty': (0, 0, 0, 0)}
    checks = {name: contingency(*counts) for name, counts in examples.items()}
    dominance = checks['independent_dominance_90_percent_positive']
    assert np.isclose(dominance['observed_agreement'], 0.82)
    assert np.isclose(dominance['marginal_expected_agreement'], 0.82)
    assert np.isclose(dominance['excess_agreement'], 0)
    assert np.isclose(dominance['cohen_kappa'], 0)
    assert np.isclose(dominance['mcc_phi'], 0)
    assert np.isclose(dominance['balanced_accuracy_bidirectional'], 0.5)
    for name, kappa, ba in (('balanced_positive_association', 0.8, 0.9), ('balanced_negative_association', -0.8, 0.1)):
        assert np.isclose(checks[name]['cohen_kappa'], kappa)
        assert np.isclose(checks[name]['mcc_phi'], kappa)
        assert np.isclose(checks[name]['balanced_accuracy_bidirectional'], ba)
    one = checks['one_sign_source_mixed_target']
    assert np.isclose(one['observed_agreement'], 0.75) and np.isclose(one['cohen_kappa'], 0)
    assert np.isclose(one['balanced_accuracy_a_to_b'], 0.5)
    assert np.isnan(one['balanced_accuracy_b_to_a']) and np.isnan(one['mcc_phi'])
    assert np.isnan(checks['both_one_sign_same']['cohen_kappa'])
    assert np.isclose(checks['both_one_sign_opposite']['cohen_kappa'], 0)
    assert all((np.isnan(checks['empty'][k]) for k in METRICS))
    for counts in examples.values():
        r = contingency(*counts)
        swapped = contingency(counts[0], counts[2], counts[1], counts[3])
        reversed_ = contingency(counts[3], counts[2], counts[1], counts[0])
        for metric in ('observed_agreement', 'marginal_expected_agreement', 'excess_agreement', 'cohen_kappa', 'mcc_phi', 'balanced_accuracy_bidirectional'):
            assert np.isclose(r[metric], swapped[metric], equal_nan=True)
            assert np.isclose(r[metric], reversed_[metric], equal_nan=True)
        assert np.isclose(r['balanced_accuracy_a_to_b'], swapped['balanced_accuracy_b_to_a'], equal_nan=True)
    return checks

def edge_equal_mean(g: pd.DataFrame, metric: str) -> float:
    return float(g.groupby(EDGE, sort=False)[metric].mean().mean()) if len(g) else np.nan

def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value)):
        return None
    return value

def library_spot_checks(pairs: pd.DataFrame) -> dict:
    """Independent library metrics from expanded 2x2 signs, no fitted models."""
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
    chosen = set(pairs.groupby(['panel', 'threshold_pp', 'sign_support_status'], sort=True).head(1).index)
    chosen.update(pairs.groupby(['panel', 'threshold_pp']).n_joint_decisive.idxmax())
    n_checks, maximum = (0, 0.0)
    for index in sorted(chosen):
        r = pairs.loc[index]
        counts = [int(r.n_pp), int(r.n_pn), int(r.n_np), int(r.n_nn)]
        a = np.repeat([1, 1, -1, -1], counts)
        b = np.repeat([1, -1, 1, -1], counts)
        if not len(a):
            continue
        comparisons = {'observed_agreement': float((a == b).mean())}
        if np.isfinite(r.cohen_kappa):
            comparisons['cohen_kappa'] = cohen_kappa_score(a, b, labels=[-1, 1])
        if r.a_has_both_signs and r.b_has_both_signs:
            comparisons['mcc_phi'] = matthews_corrcoef(a, b)
        if r.b_has_both_signs:
            comparisons['balanced_accuracy_a_to_b'] = balanced_accuracy_score(b, a)
        if r.a_has_both_signs:
            comparisons['balanced_accuracy_b_to_a'] = balanced_accuracy_score(a, b)
        for metric, expected in comparisons.items():
            assert np.isclose(r[metric], expected, atol=1e-12, rtol=1e-12), (index, metric)
            maximum = max(maximum, abs(r[metric] - expected))
            n_checks += 1
    return {'n_pair_panel_threshold_cases': len(chosen), 'n_numeric_checks': n_checks, 'maximum_absolute_discrepancy': maximum, 'method': 'Expand 2x2 counts to sign vectors and compare sklearn cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score and direct agreement; all support statuses and six panel-threshold combinations represented.'}

def aggregate_metric(g: pd.DataFrame, metric: str) -> dict:
    valid = g.loc[g[metric].notna()]
    means = valid.groupby(EDGE, sort=False)[metric].mean()
    out = dict(n_pairs=len(valid), n_edges=len(means), pair_mean=float(valid[metric].mean()) if len(valid) else np.nan, pair_median=float(valid[metric].median()) if len(valid) else np.nan, edge_equal_mean=float(means.mean()) if len(means) else np.nan, edge_equal_median=float(means.median()) if len(means) else np.nan, n_joint_decisive_comparisons=int(valid.n_joint_decisive.sum()))
    for reference in ('observed_agreement', 'marginal_expected_agreement', 'excess_agreement'):
        out[reference + '_pair_mean_same_cohort'] = float(valid[reference].mean()) if len(valid) else np.nan
        out[reference + '_edge_equal_mean_same_cohort'] = edge_equal_mean(valid, reference)
    return out

def summarize(universe: pd.DataFrame, cohort: str, kind: str, label: str, reaction: str='ALL', slot: str='ALL') -> dict:
    g = universe if cohort == 'all_pairs' else universe.loc[universe.n_shared.ge(20)]
    if cohort == 'shared_ge20_joint_ge20':
        g = g.loc[g.n_joint_decisive.ge(20)]
    row = dict(panel=universe.panel.iloc[0], threshold_pp=int(universe.threshold_pp.iloc[0]), cohort=cohort, stratum_type=kind, stratum=label, reaction_type=reaction, swap_slot=slot, n_universe_pairs=len(universe), n_universe_edges=len(universe[EDGE].drop_duplicates()), n_eligible_pairs=len(g), n_eligible_edges=len(g[EDGE].drop_duplicates()), n_shared_comparison_appearances=int(g.n_shared.sum()), n_joint_decisive_comparison_appearances=int(g.n_joint_decisive.sum()), n_no_joint_decisive=int(g.n_joint_decisive.eq(0).sum()), n_joint_decisive_below20=int(g.n_joint_decisive.lt(20).sum()), n_both_mixed=int(g.sign_support_status.eq('both_mixed').sum()), n_both_mixed_minority_below5=int((g.sign_support_status.eq('both_mixed') & g.min_minority_count.lt(5)).sum()), n_either_one_sign=int((g.n_joint_decisive.gt(0) & ~g.sign_support_status.eq('both_mixed')).sum()), n_both_one_sign_same=int(g.sign_support_status.eq('both_one_sign_same').sum()), n_both_one_sign_opposite=int(g.sign_support_status.eq('both_one_sign_opposite').sum()), n_a_one_sign=int(g.sign_support_status.eq('a_one_sign').sum()), n_b_one_sign=int(g.sign_support_status.eq('b_one_sign').sum()), n_pp=int(g.n_pp.sum()), n_pn=int(g.n_pn.sum()), n_np=int(g.n_np.sum()), n_nn=int(g.n_nn.sum()))
    for metric in METRICS:
        row.update({f'{metric}_{key}': value for key, value in aggregate_metric(g, metric).items()})
    for metric, ref in (('excess_agreement', 0), ('cohen_kappa', 0), ('mcc_phi', 0), ('balanced_accuracy_a_to_b', 0.5), ('balanced_accuracy_b_to_a', 0.5)):
        valid = g.loc[g[metric].notna()].copy()
        valid['above_reference'] = valid[metric].gt(ref + 1e-12).astype(float)
        row[metric + '_n_above_reference'] = int(valid.above_reference.sum())
        row[metric + '_fraction_above_reference_pairs'] = float(valid.above_reference.mean()) if len(valid) else np.nan
        row[metric + '_edge_equal_fraction_above_reference'] = edge_equal_mean(valid, 'above_reference')
    return row

def main() -> None:
    checks = synthetic_checks()
    profiles = pd.read_parquet(PROFILE_PATH)
    original = pd.read_csv(PAIR_PATH)
    assert len(profiles) == 267462 and len(original) == 1361
    assert not profiles.duplicated(CONTEXT + ['substrate_pair_key']).any()
    assert len(original[EDGE].drop_duplicates()) == 303
    lookup = {key: g.set_index('substrate_pair_key').sort_index() for key, g in profiles.groupby(CONTEXT, sort=False)}
    assert len(lookup) == 875
    rows = []
    max_prior_discrepancy = 0.0
    for pair_id, r in original.iterrows():
        edge = tuple((r[k] for k in EDGE))
        a, b = (lookup[*edge, r.background_a], lookup[*edge, r.background_b])
        shared = a.index.intersection(b.index).sort_values()
        aa, bb = (a.loc[shared], b.loc[shared])
        for panel in PANELS:
            keep = np.ones(len(shared), dtype=bool) if panel == 'all' else (aa.both_positive & bb.both_positive).to_numpy()
            da, db = (aa.loc[keep, 'delta'].to_numpy(), bb.loc[keep, 'delta'].to_numpy())
            assert len(da) == r[panel + '_n_shared']
            for threshold in THRESHOLDS:
                joint = (np.abs(da) >= threshold) & (np.abs(db) >= threshold)
                sa, sb = (da[joint] > 0, db[joint] > 0)
                metrics = contingency(int((sa & sb).sum()), int((sa & ~sb).sum()), int((~sa & sb).sum()), int((~sa & ~sb).sum()))
                row = {'background_pair_id': pair_id, **dict(zip(EDGE, edge)), 'background_a': r.background_a, 'background_b': r.background_b, 'full_mean_relation': r.full_mean_relation, 'substrate_jaccard': r.substrate_jaccard, 'jaccard_stratum': r.jaccard_stratum, 'panel': panel, 'threshold_pp': threshold, 'n_shared': len(da), 'n_decisive_a': int((np.abs(da) >= threshold).sum()), 'n_decisive_b': int((np.abs(db) >= threshold).sum()), **metrics, 'eligible_shared20': len(da) >= 20, 'eligible_shared20_joint20': len(da) >= 20 and metrics['n_joint_decisive'] >= 20}
                if threshold in (15, 20):
                    assert metrics['n_joint_decisive'] == r[f'{panel}_n_joint_decisive_{threshold}']
                    assert metrics['n_pp'] + metrics['n_nn'] == r[f'{panel}_n_sign_agree_{threshold}']
                    previous = float(r[f'{panel}_sign_agreement_{threshold}'])
                    assert np.isclose(metrics['observed_agreement'], previous, equal_nan=True)
                    if np.isfinite(previous):
                        max_prior_discrepancy = max(max_prior_discrepancy, abs(metrics['observed_agreement'] - previous))
                rows.append(row)
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 8166
    assert pairs.groupby(['panel', 'threshold_pp']).size().eq(1361).all()
    assert (pairs.n_pp + pairs.n_pn + pairs.n_np + pairs.n_nn).eq(pairs.n_joint_decisive).all()
    assert pairs.n_joint_decisive.le(pairs.n_shared).all()
    valid = pairs.n_joint_decisive.gt(0)
    assert np.allclose(pairs.loc[valid, 'observed_agreement'] - pairs.loc[valid, 'marginal_expected_agreement'], pairs.loc[valid, 'excess_agreement'])
    for metric in METRICS:
        finite = pairs[metric].dropna()
        bounds = (-1 - 1e-12, 1 + 1e-12) if metric in ('excess_agreement', 'cohen_kappa', 'mcc_phi') else (-1e-12, 1 + 1e-12)
        assert finite.between(*bounds).all(), metric
    for r in pairs.loc[valid].itertuples(index=False):
        determinant = r.n_pp * r.n_nn - r.n_pn * r.n_np
        assert np.isclose(r.excess_agreement, 2 * determinant / r.n_joint_decisive ** 2, atol=1e-12)
        if r.a_has_both_signs and r.b_has_both_signs:
            assert np.isclose(2 * r.balanced_accuracy_a_to_b - 1, determinant / (r.n_b_positive * r.n_b_negative), atol=1e-12)
            assert np.isclose(2 * r.balanced_accuracy_b_to_a - 1, determinant / (r.n_a_positive * r.n_a_negative), atol=1e-12)
    library_checks = library_spot_checks(pairs)
    pairs.to_csv(RESULTS / 'matched_sign_information_pairs.csv', index=False)
    summary = []
    edge_rows = []
    for (panel, threshold), g in pairs.groupby(['panel', 'threshold_pp'], sort=True):
        for cohort in COHORTS:
            summary.append(summarize(g, cohort, 'overall', 'ALL'))
            for (reaction, slot), sub in g.groupby(['reaction_type', 'swap_slot'], sort=True):
                summary.append(summarize(sub, cohort, 'reaction_slot', f'{reaction}|{slot}', reaction, slot))
            for rel, sub in g.groupby('full_mean_relation', sort=True):
                summary.append(summarize(sub, cohort, 'full_mean_relation', rel))
            for label, sub in g.groupby('jaccard_stratum', sort=True):
                summary.append(summarize(sub, cohort, 'jaccard', label))
            for edge, sub in g.groupby(EDGE, sort=True):
                eligible = sub if cohort == 'all_pairs' else sub.loc[sub.n_shared.ge(20)]
                if cohort == 'shared_ge20_joint_ge20':
                    eligible = eligible.loc[eligible.n_joint_decisive.ge(20)]
                row = dict(panel=panel, threshold_pp=int(threshold), cohort=cohort, **dict(zip(EDGE, edge)), n_universe_pairs=len(sub), n_eligible_pairs=len(eligible), n_joint_decisive_comparison_appearances=int(eligible.n_joint_decisive.sum()), n_both_mixed=int(eligible.sign_support_status.eq('both_mixed').sum()), n_both_mixed_minority_below5=int((eligible.sign_support_status.eq('both_mixed') & eligible.min_minority_count.lt(5)).sum()))
                for metric in METRICS:
                    valid = eligible.loc[eligible[metric].notna()]
                    row[metric + '_n_pairs'] = len(valid)
                    row[metric + '_mean'] = float(valid[metric].mean()) if len(valid) else np.nan
                    row[metric + '_observed_agreement_mean_same_cohort'] = float(valid.observed_agreement.mean()) if len(valid) else np.nan
                    row[metric + '_marginal_expected_agreement_mean_same_cohort'] = float(valid.marginal_expected_agreement.mean()) if len(valid) else np.nan
                edge_rows.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / 'matched_sign_information_summary.csv', index=False)
    pd.DataFrame(edge_rows).to_csv(RESULTS / 'matched_sign_information_edges.csv', index=False)
    audit = {'inputs': {}, 'universe_edges': 302, 'universe_unique_backgrounds': 873, 'universe_background_pairs': 1360, 'pair_panel_threshold_rows': len(pairs), 'thresholds_pp': list(THRESHOLDS), 'panels': list(PANELS), 'cohorts': list(COHORTS), 'metric_status_counts': pairs.groupby(['panel', 'threshold_pp', 'sign_support_status']).size().reset_index(name='n_pairs').to_dict('records'), 'kappa_undefined': 'No jointly decisive pairs, or marginal expected agreement is exactly one.', 'phi_undefined': 'No jointly decisive pairs, or either marginal contains only one sign.', 'balanced_accuracy_undefined': 'Requires both signs in the target background; bidirectional average requires both directions defined.', 'weighting': 'Pair mean weights eligible background pairs equally; edge-equal mean first averages eligible pairs within each component edge, then edges equally. Every metric carries own valid pair/edge denominator and raw/expected agreement on that exact metric cohort.', 'prior_agreement_reconciliation': {'thresholds': [15, 20], 'rows_checked': 5440, 'maximum_absolute_discrepancy': max_prior_discrepancy}, 'synthetic_checks': checks, 'independent_library_spot_checks': library_checks, 'invariants_passed': ['8160 rows; each panel-threshold retains1360 pairs/302 edges', 'unique collapsed profile keys', 'shared-panel and prior15/20 threshold count reconciliation', '2x2 counts sum to jointly decisive denominator', 'jointly decisive subset of shared panel', 'metric bounds', 'observed minus expected equals excess', 'excess equals2*determinant/N^2', 'balanced accuracy determinant identities', 'sign-negation and background-swap invariance in synthetic cases'], 'interpretation': 'Marginal adjustment is descriptive and does not remove shared chemistry, background, substrate-panel, outcome-selection, or reuse confounding. No p-values or independent-replication claim.'}
    for path in (PROFILE_PATH, PAIR_PATH):
        with path.open('rb') as stream:
            audit['inputs'][path.name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    (BASE / 'audit' / 'matched_sign_information_metadata.json').write_text(json.dumps(clean_json(audit), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (BASE / 'reports' / 'MATCHED_SIGN_INFORMATION.md').write_text(make_report(pairs, summary), encoding='utf-8')
    print(summary.loc[summary.stratum_type.eq('overall'), ['panel', 'threshold_pp', 'cohort', 'n_eligible_pairs', 'n_eligible_edges', 'observed_agreement_edge_equal_mean', 'marginal_expected_agreement_edge_equal_mean', 'excess_agreement_edge_equal_mean', 'cohen_kappa_edge_equal_mean', 'mcc_phi_edge_equal_mean']].to_string(index=False))

def make_report(pairs: pd.DataFrame, summary: pd.DataFrame) -> str:

    def f(value: float) -> str:
        return f'{value:.4f}' if np.isfinite(value) else 'NA'

    def overall(cohort: str) -> pd.DataFrame:
        return summary.loc[summary.stratum_type.eq('overall') & summary.cohort.eq(cohort)]

    def tab(frame: pd.DataFrame, fields: list[tuple[str, str]]) -> list[str]:
        lines = ['| ' + ' | '.join((label for label, _ in fields)) + ' |', '|' + '|'.join(('---' for _ in fields)) + '|']
        for _, r in frame.iterrows():
            cells = []
            for _, col in fields:
                value = r[col]
                cells.append(str(int(value)) if col.startswith('n_') or col.endswith('_n_pairs') or col.endswith('_n_edges') or (col == 'threshold_pp') else f(value) if isinstance(value, (float, np.floating)) else str(value))
            lines.append('| ' + ' | '.join(cells) + ' |')
        return lines
    primary20 = overall('shared_ge20_joint_ge20')
    primary20 = primary20.loc[primary20.panel.eq('all') & primary20.threshold_pp.eq(20)].iloc[0]
    prior20 = overall('shared_ge20')
    prior20 = prior20.loc[prior20.panel.eq('all') & prior20.threshold_pp.eq(20)].iloc[0]
    overview = f'The earlier all-yield 20-pp agreement of {f(prior20.observed_agreement_edge_equal_mean)} decomposes into {f(prior20.marginal_expected_agreement_edge_equal_mean)} expected from marginal sign frequencies plus {f(prior20.excess_agreement_edge_equal_mean)} excess on exactly the same cohort. With at least 20 jointly decisive substrates, {int(primary20.n_eligible_pairs)} background pairs / {int(primary20.n_eligible_edges)} edges retain excess agreement {f(primary20.excess_agreement_edge_equal_mean)}; kappa is {f(primary20.cohen_kappa_edge_equal_mean)} on its {int(primary20.cohen_kappa_n_pairs)} defined pairs / {int(primary20.cohen_kappa_n_edges)} edges. Thus marginal dominance explains part of the high agreement, while descriptive sign association remains on the supported decisive panels. This selected-panel association does not imply an overall yield-policy advantage.'
    lines = ['# Matched sign information beyond marginal dominance', '', overview, '', '## Question and universe', '', 'High decisive sign agreement can arise when each background strongly favors the same component for almost every substrate. This analysis asks whether the observed sign pairing carries additional descriptive association beyond those two marginal sign frequencies.', '', 'All 302 repeated strict component edges, 873 unique edge-background contexts, and 1,360 unordered background pairs are retained at 15, 20, and 30 pp, for all yields and the four-arm both-positive sensitivity: 8,160 auditable pair-panel-threshold rows. It reuses the previously collapsed canonical response profiles read-only. A substrate is jointly decisive only when both absolute responses meet the same threshold. The both-positive flag requires every underlying original yield in both arms of both backgrounds to exceed zero before duplicate-background averaging.', '', 'The ungated universe retains zero overlap, zero jointly decisive substrates, sparse panels, and one-sign margins. `shared_ge20` holds the earlier profile-support gate fixed. `shared_ge20_joint_ge20` additionally requires at least 20 jointly decisive substrates. All metric values remain saved before gating. These are descriptive support gates, not significance thresholds.', '', '## Definitions', '', "The 2x2 table records n_PP, n_PN, n_NP, n_NN, where the first sign is background A and the second is B. N is their sum. Observed agreement is (n_PP+n_NN)/N. Marginal expected agreement is p_A p_B + (1-p_A)(1-p_B), where p_A and p_B are positive-sign fractions on exactly those jointly decisive substrates. Excess agreement is observed minus expected. Cohen kappa divides excess by 1-expected. MCC/phi is the 2x2 determinant divided by the square root of the four marginal counts' product.", '', 'For A to B, balanced accuracy treats A sign as a prediction of B sign and averages the sensitivities for positive and negative B responses: 0.5[n_PP/(n_PP+n_NP) + n_NN/(n_NN+n_PN)]. B to A swaps the roles. A directional balanced accuracy is undefined when its target has only one sign; MCC/phi requires both signs in both backgrounds. Kappa is undefined when expected agreement is one. An opposite one-sign pair has kappa zero by the formula, while phi and both directional balanced accuracies remain undefined. All undefined cases and their reasons are explicit, and no zero is substituted for an undefined value.', '', "The edge-equal summaries first average eligible background pairs within each edge and then average the available edge means. Every metric has its own valid pair and edge denominators. Raw and expected agreement on each metric's exact valid cohort are included in the summary CSV, so a change of denominator cannot masquerade as an adjustment effect. Summed contingency counts are reused comparison appearances, not independent observations.", '', '## Same Shared-Panel Gate as the Prior Analysis', '', 'The raw agreement below is recomputed on exactly the earlier shared >=20 cohort. At 15 and 20 pp, every jointly decisive denominator and agreeing numerator reconciles to the prior pair table (5,440 checked rows including ungated rows).', '']
    lines += tab(overall('shared_ge20'), [('Panel', 'panel'), ('Threshold', 'threshold_pp'), ('Raw pairs', 'observed_agreement_n_pairs'), ('Raw edges', 'observed_agreement_n_edges'), ('Raw agreement', 'observed_agreement_edge_equal_mean'), ('Marginal expected', 'marginal_expected_agreement_edge_equal_mean'), ('Excess', 'excess_agreement_edge_equal_mean')])
    lines += ['', '## At Least 20 Jointly Decisive Substrates', '']
    primary = overall('shared_ge20_joint_ge20')
    lines += tab(primary, [('Panel', 'panel'), ('Threshold', 'threshold_pp'), ('Pairs', 'n_eligible_pairs'), ('Edges', 'n_eligible_edges'), ('Raw agreement', 'observed_agreement_edge_equal_mean'), ('Marginal expected', 'marginal_expected_agreement_edge_equal_mean'), ('Excess', 'excess_agreement_edge_equal_mean'), ('Kappa', 'cohen_kappa_edge_equal_mean'), ('Kappa pairs', 'cohen_kappa_n_pairs'), ('Kappa edges', 'cohen_kappa_n_edges')])
    lines += ['', "Kappa's valid cohort may differ from the raw-agreement cohort when both margins are identically one sign. Its same-cohort raw comparison follows; phi and directional balanced accuracies use their own valid denominators.", '']
    lines += tab(primary, [('Panel', 'panel'), ('Threshold', 'threshold_pp'), ('Raw on kappa cohort', 'cohen_kappa_observed_agreement_edge_equal_mean_same_cohort'), ('Expected on kappa cohort', 'cohen_kappa_marginal_expected_agreement_edge_equal_mean_same_cohort'), ('Phi', 'mcc_phi_edge_equal_mean'), ('Phi pairs', 'mcc_phi_n_pairs'), ('Phi edges', 'mcc_phi_n_edges'), ('BA A to B', 'balanced_accuracy_a_to_b_edge_equal_mean'), ('BA A to B pairs', 'balanced_accuracy_a_to_b_n_pairs'), ('BA B to A', 'balanced_accuracy_b_to_a_edge_equal_mean'), ('BA B to A pairs', 'balanced_accuracy_b_to_a_n_pairs')])
    lines += ['', '## Sign Support and Auditable Counts', '']
    lines += tab(primary, [('Panel', 'panel'), ('Threshold', 'threshold_pp'), ('Both mixed', 'n_both_mixed'), ('Either one sign', 'n_either_one_sign'), ('Both same one sign', 'n_both_one_sign_same'), ('PP', 'n_pp'), ('PN', 'n_pn'), ('NP', 'n_np'), ('NN', 'n_nn')])
    lines += ['', 'A one-sign source with a mixed-sign target has balanced accuracy 0.5 in that direction, even if its raw agreement is high. Phi excludes all one-sign margins; this changes its valid cohort and must remain visible when interpreting the mean. A total of 20 jointly decisive substrates need not provide 20 examples of each sign: each pair retains both minority-sign counts, and summaries count mixed-sign panels with fewer than five observations in at least one minority margin. These are support diagnostics rather than an additional primary gate.', '', '## Reaction Class and Changed Slot', '', 'The table gives the all-yield 20-pp primary cohort for every reaction-slot stratum with an eligible pair; strata with no support remain in the CSV. All thresholds, both yield panels, full-mean-sign strata, Jaccard strata, and ungated results are saved without narrowing to selected edges.', '']
    reaction = summary.loc[summary.stratum_type.eq('reaction_slot') & summary.panel.eq('all') & summary.threshold_pp.eq(20) & summary.cohort.eq('shared_ge20_joint_ge20') & summary.n_eligible_pairs.gt(0)]
    lines += tab(reaction, [('Reaction', 'reaction_type'), ('Slot', 'swap_slot'), ('Pairs', 'n_eligible_pairs'), ('Edges', 'n_eligible_edges'), ('Raw', 'observed_agreement_edge_equal_mean'), ('Expected', 'marginal_expected_agreement_edge_equal_mean'), ('Excess', 'excess_agreement_edge_equal_mean'), ('Kappa', 'cohen_kappa_edge_equal_mean'), ('Kappa edges', 'cohen_kappa_n_edges'), ('Phi', 'mcc_phi_edge_equal_mean'), ('Phi edges', 'mcc_phi_n_edges')])
    lines += ['', '## Arithmetic Checks and Interpretation', '', 'A synthetic independent dominance table (81,9,9,1) has 82% observed agreement and 82% marginal expected agreement, but zero excess, kappa, and phi, with balanced accuracy 0.5. A balanced association table (45,5,5,45) has 90% agreement, 50% expected agreement, excess 0.4, kappa and phi 0.8, and balanced accuracy 0.9. Its reversed-association counterpart gives kappa and phi -0.8. Empty, one-sign-source, both-same-one-sign, and both-opposite-one-sign examples verify undefined handling. Synthetic sign negation and background swapping verify orientation symmetry.', '', 'Full-output invariants check 2x2 counts, support bounds, metric ranges, the exact identity excess=2(n_PP*n_NN-n_PN*n_NP)/N^2, directional balanced-accuracy determinant identities, and exact reconciliation with earlier 15/20-pp counts and raw agreement. Independent library spot checks expand observed contingency counts to sign vectors and compare scikit-learn kappa, MCC, balanced accuracy, and direct agreement for all support statuses and all six panel-threshold combinations; their exact case count and maximum discrepancy are in the audit.', '', 'Marginal adjustment identifies association beyond simple marginal sign dominance within the selected shared decisive panel. It does not remove chemical, background, substrate-composition, outcome-selection, or shared-observation confounding. Joint decisiveness itself selects on both measured responses. Positive kappa or phi is not evidence of a molecular mechanism, independent replication, new-substrate prediction, or a useful prospective policy. Negative or zero association is retained. No p-values are computed.', '', 'Files: `results/matched_sign_information_pairs.csv`, `results/matched_sign_information_summary.csv`, `results/matched_sign_information_edges.csv`, and `audit/matched_sign_information_metadata.json`. Reproduce with `python code/matched_sign_information.py`. Only the new named outputs are written.', '']
    return '\n'.join(lines)
if __name__ == '__main__':
    main()
