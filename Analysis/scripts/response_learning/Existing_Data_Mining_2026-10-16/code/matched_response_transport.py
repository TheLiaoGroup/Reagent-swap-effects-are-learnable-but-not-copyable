"""Matched response-profile and observed-substrate policy transport across backgrounds.

All quantities are descriptive. Background pairs and reused substrate panels are
not independent replicates. Original raw data and upstream results are read-only."""
from __future__ import annotations
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'results'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
CONTEXT = EDGE + ['background_fingerprint']
THRESHOLDS = (10, 15, 20)
JAC_LABELS = ['J<0.25', 'J0.25-0.50', 'J0.50-0.75', 'J0.75-0.90', 'J>=0.90']
INPUTS = {name: RESULTS / name for name in ('condition_network_meta.csv', 'swap_deltas.parquet', 'direction_magnitude_background.csv', 'background_portability_overlap.csv')}

def sign(x: float) -> str:
    if not np.isfinite(x):
        return 'unavailable'
    return 'positive' if x > 1e-09 else 'negative' if x < -1e-09 else 'tie'

def relation(a: float, b: float) -> str:
    sa, sb = (sign(a), sign(b))
    if 'unavailable' in (sa, sb):
        return 'unavailable'
    if 'tie' in (sa, sb):
        return 'tied'
    return 'same' if sa == sb else 'opposite'

def profile(a: np.ndarray, b: np.ndarray, low_a: np.ndarray, low_b: np.ndarray, all_mean_a: float, all_mean_b: float) -> dict:
    n = len(a)
    out = {'n_shared': n}
    ma, mb = (float(a.mean()), float(b.mean())) if n else (np.nan, np.nan)
    out.update(mean_delta_a_pp=ma, mean_delta_b_pp=mb, mean_relation=relation(ma, mb), mean_shift_b_minus_a_pp=mb - ma, abs_mean_shift_pp=abs(mb - ma), composition_shift_a_pp=ma - all_mean_a, composition_shift_b_pp=mb - all_mean_b, abs_composition_shift_mean_pp=(abs(ma - all_mean_a) + abs(mb - all_mean_b)) / 2, sign_a=sign(ma), sign_b=sign(mb))
    constant_a = bool(n and np.ptp(a) <= 1e-12)
    constant_b = bool(n and np.ptp(b) <= 1e-12)
    status = 'no_shared' if n == 0 else 'one_shared' if n == 1 else 'both_constant' if constant_a and constant_b else 'a_constant' if constant_a else 'b_constant' if constant_b else 'valid'
    out.update(correlation_status=status, constant_a=constant_a, constant_b=constant_b, pearson=np.nan, spearman=np.nan)
    if status == 'valid':
        out['pearson'] = float(np.corrcoef(a, b)[0, 1])
        out['spearman'] = float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
    out['raw_rmse_pp'] = float(np.sqrt(np.mean((b - a) ** 2))) if n else np.nan
    out['raw_mae_pp'] = float(np.mean(np.abs(b - a))) if n else np.nan
    out['centered_rmse_pp'] = float(np.sqrt(np.mean((b - mb - (a - ma)) ** 2))) if n else np.nan
    out['centered_mae_pp'] = float(np.mean(np.abs(b - mb - (a - ma)))) if n else np.nan
    for threshold in THRESHOLDS:
        da, db = (np.abs(a) >= threshold, np.abs(b) >= threshold)
        joint = da & db
        agree = joint & ((a > 0) == (b > 0))
        denom, num = (int(joint.sum()), int(agree.sum()))
        out.update({f'n_decisive_a_{threshold}': int(da.sum()), f'n_decisive_b_{threshold}': int(db.sum()), f'n_joint_decisive_{threshold}': denom, f'n_sign_agree_{threshold}': num, f'n_sign_opposite_{threshold}': denom - num, f'sign_agreement_{threshold}': num / denom if denom else np.nan, f'joint_decisive_fraction_{threshold}': denom / n if n else np.nan, f'balanced_support_a_{threshold}': 2 * min(int((a >= threshold).sum()), int((a <= -threshold).sum())) / n if n else np.nan, f'balanced_support_b_{threshold}': 2 * min(int((b >= threshold).sum()), int((b <= -threshold).sum())) / n if n else np.nan})
    for direction, source, target, low, source_all_mean in (('a_to_b', a, b, low_b, all_mean_a), ('b_to_a', b, a, low_a, all_mean_b)):
        sm = float(source.mean()) if n else np.nan
        tm = float(target.mean()) if n else np.nan
        inc = float(np.mean((source > 0) * target)) if n else np.nan
        fixed = float(sm > 0) * tm if n else np.nan
        all_fixed = float(source_all_mean > 0) * tm if n else np.nan
        target_fixed = max(0.0, tm) if n else np.nan
        oracle = float(np.maximum(target, 0).mean()) if n else np.nan
        low_mean = float(low.mean()) if n else np.nan
        vals = {'source_policy_increment_pp': inc, 'source_common_fixed_increment_pp': fixed, 'source_all_fixed_increment_pp': all_fixed, 'target_hindsight_fixed_increment_pp': target_fixed, 'target_substrate_hindsight_increment_pp': oracle, 'gain_vs_source_common_fixed_pp': inc - fixed, 'gain_vs_source_all_fixed_pp': inc - all_fixed, 'gain_vs_target_hindsight_fixed_pp': inc - target_fixed, 'regret_vs_target_substrate_hindsight_pp': oracle - inc, 'achieved_target_yield_pp': low_mean + inc, 'target_hindsight_fixed_yield_pp': low_mean + target_fixed, 'n_source_high_selected': int((source > 0).sum()), 'source_common_fixed_selects_high': bool(n and sm > 0), 'source_all_fixed_selects_high': bool(n and source_all_mean > 0)}
        out.update({f'{direction}_{k}': v for k, v in vals.items()})
    for metric in ('gain_vs_source_common_fixed_pp', 'gain_vs_source_all_fixed_pp', 'gain_vs_target_hindsight_fixed_pp', 'regret_vs_target_substrate_hindsight_pp'):
        out[f'bidirectional_{metric}'] = (out[f'a_to_b_{metric}'] + out[f'b_to_a_{metric}']) / 2
    return out

def metric_aggregate(g: pd.DataFrame, col: str) -> dict:
    v = g.loc[g[col].notna()]
    edge_means = v.groupby(EDGE, sort=False)[col].mean() if len(v) else pd.Series(dtype=float)
    return {'n_context_pairs': len(v), 'n_edges': len(edge_means), 'pair_mean': float(v[col].mean()) if len(v) else np.nan, 'pair_median': float(v[col].median()) if len(v) else np.nan, 'edge_equal_mean': float(edge_means.mean()) if len(edge_means) else np.nan, 'edge_equal_median': float(edge_means.median()) if len(edge_means) else np.nan}

def summarize(g: pd.DataFrame, panel: str, scope: str, kind: str, label: str, reaction: str='ALL', slot: str='ALL') -> dict:
    p = f'{panel}_'
    row = dict(panel=panel, eligibility=scope, stratum_type=kind, stratum=label, reaction_type=reaction, swap_slot=slot, n_context_pairs=len(g), n_edges=len(g[EDGE].drop_duplicates()), n_zero_shared=int(g[p + 'n_shared'].eq(0).sum()), n_one_shared=int(g[p + 'n_shared'].eq(1).sum()), n_below_20_shared=int(g[p + 'n_shared'].lt(20).sum()), n_constant_profiles=int(g[p + 'correlation_status'].isin(['both_constant', 'a_constant', 'b_constant']).sum()), n_shared_comparisons=int(g[p + 'n_shared'].sum()))
    metrics = ['spearman', 'pearson', 'centered_rmse_pp', 'centered_mae_pp', 'raw_rmse_pp', 'raw_mae_pp', 'abs_mean_shift_pp', 'abs_composition_shift_mean_pp', 'bidirectional_gain_vs_source_common_fixed_pp', 'bidirectional_gain_vs_source_all_fixed_pp', 'bidirectional_gain_vs_target_hindsight_fixed_pp', 'bidirectional_regret_vs_target_substrate_hindsight_pp']
    for metric in metrics:
        row.update({f'{metric}_{k}': v for k, v in metric_aggregate(g, p + metric).items()})
    for metric in ('spearman', 'bidirectional_gain_vs_target_hindsight_fixed_pp'):
        valid = g[p + metric].notna()
        indicator = g.loc[valid, EDGE].copy()
        indicator['positive'] = g.loc[valid, p + metric].gt(1e-09).astype(float)
        row[f'{metric}_n_positive'] = int(indicator.positive.sum())
        row[f'{metric}_fraction_positive_pairs'] = float(indicator.positive.mean()) if len(indicator) else np.nan
        row[f'{metric}_edge_equal_fraction_positive'] = float(indicator.groupby(EDGE).positive.mean().mean()) if len(indicator) else np.nan
    for threshold in THRESHOLDS:
        den, num = (p + f'n_joint_decisive_{threshold}', p + f'n_sign_agree_{threshold}')
        row[f'n_joint_decisive_{threshold}_comparisons'] = int(g[den].sum())
        row[f'n_sign_agree_{threshold}_comparisons'] = int(g[num].sum())
        row[f'pooled_comparison_sign_agreement_{threshold}'] = g[num].sum() / g[den].sum() if g[den].sum() else np.nan
        row.update({f'sign_agreement_{threshold}_{k}': v for k, v in metric_aggregate(g, p + f'sign_agreement_{threshold}').items()})
    available = g[p + 'mean_relation'].isin(['same', 'opposite'])
    row['n_common_mean_relation_non_tied'] = int(available.sum())
    row['n_common_mean_opposite'] = int(g[p + 'mean_relation'].eq('opposite').sum())
    row['n_full_opposite_common_same'] = int((g.full_mean_relation.eq('opposite') & g[p + 'mean_relation'].eq('same')).sum())
    row['n_full_same_common_opposite'] = int((g.full_mean_relation.eq('same') & g[p + 'mean_relation'].eq('opposite')).sum())
    return row

def validate_profiles() -> None:
    a = np.array([-30.0, -10.0, 20.0, 40.0])
    b = a + 5
    v = profile(a, b, np.zeros(4), np.zeros(4), a.mean(), b.mean())
    assert np.isclose(v['spearman'], 1) and np.isclose(v['centered_rmse_pp'], 0)
    assert np.isclose(v['mean_shift_b_minus_a_pp'], 5)
    assert v['n_joint_decisive_20'] == 3 and v['n_sign_agree_20'] == 3
    assert np.isclose(v['balanced_support_a_20'], 0.5)
    assert np.isclose(v['a_to_b_source_policy_increment_pp'], 17.5)
    assert np.isclose(v['a_to_b_gain_vs_target_hindsight_fixed_pp'], 7.5)
    assert profile(np.zeros(3), np.arange(3.0), np.zeros(3), np.zeros(3), 0, 1)['correlation_status'] == 'a_constant'
    assert profile(np.array([]), np.array([]), np.array([]), np.array([]), 0, 0)['correlation_status'] == 'no_shared'
    assert profile(np.array([1.0]), np.array([-1.0]), np.array([0.0]), np.array([1.0]), 1, -1)['correlation_status'] == 'one_shared'

def independent_spot_checks(raw: pd.DataFrame, selected: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """Rebuild selected panels from original arm yields using separate arithmetic."""
    from collections import defaultdict
    from scipy.stats import rankdata
    choices = {'zero_overlap': pairs.index[pairs.all_n_shared.eq(0)][0], 'constant_profile': pairs.index[pairs.all_correlation_status.ne('valid') & pairs.all_n_shared.gt(1)][0], 'duplicate_background': pairs.index[pairs.n_swap_rows_collapsed_a.gt(1) | pairs.n_swap_rows_collapsed_b.gt(1)][0], 'largest_positive_policy_gain': pairs.all_bidirectional_gain_vs_target_hindsight_fixed_pp.idxmax(), 'largest_negative_policy_gain': pairs.all_bidirectional_gain_vs_target_hindsight_fixed_pp.idxmin(), 'composition_resolved_opposite': pairs.index[pairs.full_mean_relation.eq('opposite') & pairs.all_mean_relation.eq('same')][0], 'large_panel': pairs.all_n_shared.idxmax()}
    checks = []
    for label, idx in choices.items():
        r = pairs.loc[idx]
        rebuilt = []
        n_reversed = 0
        for suffix in ('a', 'b'):
            ids = r[f'swap_ids_{suffix}'].split('||')
            m = selected.set_index('swap_id').loc[ids]
            n_reversed += int(m.orientation_sign.lt(0).sum())
            observations = defaultdict(list)
            for v in raw.loc[raw.swap_id.isin(ids)].itertuples(index=False):
                orient = m.loc[v.swap_id, 'orientation_sign']
                arms = (v.y0 * 100, v.y1 * 100) if orient > 0 else (v.y1 * 100, v.y0 * 100)
                observations[v.substrate_pair_key].append(arms)
            rebuilt.append({key: (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])), all((low > 0 and high > 0 for low, high in vals))) for key, vals in observations.items()})
        all_errors = []
        for panel in ('all', 'both_positive'):
            keys = sorted(set(rebuilt[0]) & set(rebuilt[1]))
            if panel == 'both_positive':
                keys = [key for key in keys if rebuilt[0][key][2] and rebuilt[1][key][2]]
            assert len(keys) == r[f'{panel}_n_shared']
            if not keys:
                continue
            arms_a = np.array([rebuilt[0][key][:2] for key in keys])
            arms_b = np.array([rebuilt[1][key][:2] for key in keys])
            da, db = (arms_a[:, 1] - arms_a[:, 0], arms_b[:, 1] - arms_b[:, 0])
            comparisons = {'mean_delta_a_pp': da.mean(), 'mean_delta_b_pp': db.mean(), 'centered_rmse_pp': np.sqrt(np.mean((db - db.mean() - (da - da.mean())) ** 2))}
            if r[f'{panel}_correlation_status'] == 'valid':
                ar, br = (rankdata(da), rankdata(db))
                ar, br = (ar - ar.mean(), br - br.mean())
                comparisons['spearman'] = np.dot(ar, br) / np.sqrt(np.dot(ar, ar) * np.dot(br, br))
            for t in THRESHOLDS:
                joint = [i for i in range(len(da)) if abs(da[i]) >= t and abs(db[i]) >= t]
                comparisons[f'n_joint_decisive_{t}'] = len(joint)
                comparisons[f'n_sign_agree_{t}'] = sum(((da[i] > 0) == (db[i] > 0) for i in joint))
            for direction, source, target, source_delta in (('a_to_b', arms_a, arms_b, da), ('b_to_a', arms_b, arms_a, db)):
                achieved = np.mean([target[i, 1] if source_delta[i] > 0 else target[i, 0] for i in range(len(keys))])
                source_fixed_arm = int(source[:, 1].mean() > source[:, 0].mean())
                source_fixed = target[:, source_fixed_arm].mean()
                target_fixed = max(target[:, 0].mean(), target[:, 1].mean())
                comparisons[direction + '_achieved_target_yield_pp'] = achieved
                comparisons[direction + '_gain_vs_source_common_fixed_pp'] = achieved - source_fixed
                comparisons[direction + '_gain_vs_target_hindsight_fixed_pp'] = achieved - target_fixed
            for metric, expected in comparisons.items():
                actual = float(r[f'{panel}_{metric}'])
                assert np.isclose(actual, expected, rtol=1e-10, atol=1e-08), (label, panel, metric, actual, expected)
                all_errors.append(abs(actual - expected))
        checks.append({'case': label, 'pair_row_index_zero_based': int(idx), **{k: r[k] for k in EDGE}, 'n_shared': int(r.all_n_shared), 'n_reversed_swap_rows': n_reversed, 'n_numeric_checks': len(all_errors), 'maximum_absolute_numeric_discrepancy': max(all_errors) if all_errors else 0.0})
    return checks

def main() -> None:
    validate_profiles()
    meta = pd.read_csv(INPUTS['condition_network_meta.csv'])
    strict = meta.loc[meta.row_level.eq('swap')].copy()
    assert not strict.swap_id.duplicated().any()
    nbg = strict.groupby(EDGE).background_fingerprint.nunique()
    repeated = nbg[nbg.ge(2)].reset_index()[EDGE]
    selected = strict.merge(repeated, on=EDGE, validate='many_to_one')
    assert len(repeated) == 303 and len(selected) == 876
    expected_orientation = np.where(selected.component0_id.eq(selected.component_pair.str.split('|').str[0]), 1.0, -1.0)
    assert np.array_equal(expected_orientation, selected.orientation_sign.to_numpy())
    cols = ['swap_id', 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles', 'y0', 'y1', 'delta_pp', 'reaction_type', 'condition0', 'condition1']
    raw = pd.read_parquet(INPUTS['swap_deltas.parquet'], columns=cols)
    raw = raw.loc[raw.swap_id.isin(selected.swap_id)].copy()
    assert not raw.duplicated(['swap_id', 'substrate_pair_key']).any()
    assert np.isfinite(raw[['y0', 'y1', 'delta_pp']].to_numpy()).all()
    assert np.allclose(raw.delta_pp, 100 * (raw.y1 - raw.y0), atol=1e-10)
    joined = raw.merge(selected[['swap_id'] + CONTEXT + ['orientation_sign']], on=['swap_id', 'reaction_type'], validate='many_to_one')
    joined['delta'] = joined.delta_pp * joined.orientation_sign
    joined['low_yield_pp'] = 100 * np.where(joined.orientation_sign.gt(0), joined.y0, joined.y1)
    joined['high_yield_pp'] = 100 * np.where(joined.orientation_sign.gt(0), joined.y1, joined.y0)
    joined['all_observations_positive'] = joined.y0.gt(0) & joined.y1.gt(0)
    assert np.allclose(joined.delta, joined.high_yield_pp - joined.low_yield_pp, atol=1e-10)
    swap_recon = joined.groupby('swap_id').delta.agg(['mean', 'size']).join(selected.set_index('swap_id')[['mean_delta_norm_pp', 'n_pairs']])
    assert np.allclose(swap_recon['mean'], swap_recon.mean_delta_norm_pp, atol=1e-08)
    assert np.array_equal(swap_recon['size'], swap_recon.n_pairs)
    collapsed = joined.groupby(CONTEXT + ['substrate_pair_key'], sort=True).agg(delta=('delta', 'mean'), low_yield_pp=('low_yield_pp', 'mean'), high_yield_pp=('high_yield_pp', 'mean'), all_observations_positive=('all_observations_positive', 'all'), n_observations=('swap_id', 'size'), n_swap_ids=('swap_id', 'nunique')).reset_index()
    assert np.allclose(collapsed.delta, collapsed.high_yield_pp - collapsed.low_yield_pp, atol=1e-10)
    identity = raw[['reaction_type', 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles']].drop_duplicates()
    assert not identity.duplicated(['reaction_type', 'substrate_pair_key']).any()
    assert identity[['sub_1_smiles', 'sub_2_smiles']].notna().all().all()
    collapsed = collapsed.merge(identity, on=['reaction_type', 'substrate_pair_key'], validate='many_to_one')
    collapsed['both_positive'] = collapsed.all_observations_positive
    profile_path = RESULTS / 'matched_response_profiles.parquet'
    collapsed.to_parquet(profile_path, index=False)
    context_ids = selected.groupby(CONTEXT).swap_id.agg(lambda x: '||'.join(sorted(x))).to_dict()
    upstream_bg = pd.read_csv(INPUTS['direction_magnitude_background.csv']).set_index(CONTEXT)
    contexts, profiles = ([], {})
    for key, g in collapsed.groupby(CONTEXT, sort=True):
        g = g.set_index('substrate_pair_key').sort_index()
        profiles[key] = g
        d = g.delta.to_numpy()
        all_mean = float(d.mean())
        positive = g.all_observations_positive
        r = {**dict(zip(CONTEXT, key)), 'swap_ids': context_ids[key], 'n_swap_rows_collapsed': len(context_ids[key].split('||')), 'n_substrate_pairs': len(g), 'n_input_responses': int(g.n_observations.sum()), 'n_substrates_with_repeated_observations': int(g.n_observations.gt(1).sum()), 'max_observations_per_substrate': int(g.n_observations.max()), 'mean_delta_pp': all_mean, 'sign': sign(all_mean), 'n_both_positive_substrates': int(positive.sum()), 'both_positive_mean_delta_pp': float(g.loc[positive, 'delta'].mean()), 'balanced_support_20': 2 * min(int((d >= 20).sum()), int((d <= -20).sum())) / len(d), 'upstream_row_weighted_mean_delta_pp': float(upstream_bg.loc[key, 'mean_delta_norm_pp'])}
        contexts.append(r)
    context = pd.DataFrame(contexts)
    assert len(context) == 875
    context_lookup = context.set_index(CONTEXT).to_dict('index')
    reference = pd.read_csv(INPUTS['background_portability_overlap.csv'])
    reference_key = EDGE + ['background_a', 'background_b']
    assert not reference.duplicated(reference_key).any()
    pair_rows = []
    for edge, g in context.groupby(EDGE, sort=True):
        for aa, bb in combinations(g.sort_values('background_fingerprint').background_fingerprint, 2):
            ka, kb = ((*edge, aa), (*edge, bb))
            a, b = (profiles[ka], profiles[kb])
            ca, cb = (context_lookup[ka], context_lookup[kb])
            shared = a.index.intersection(b.index).sort_values()
            union = len(a.index.union(b.index))
            ma, mb = (ca['mean_delta_pp'], cb['mean_delta_pp'])
            row = {**dict(zip(EDGE, edge)), 'background_a': aa, 'background_b': bb, 'swap_ids_a': ca['swap_ids'], 'swap_ids_b': cb['swap_ids'], 'n_substrates_a': len(a), 'n_substrates_b': len(b), 'n_input_responses_a': ca['n_input_responses'], 'n_input_responses_b': cb['n_input_responses'], 'n_swap_rows_collapsed_a': ca['n_swap_rows_collapsed'], 'n_swap_rows_collapsed_b': cb['n_swap_rows_collapsed'], 'n_union_substrates': union, 'substrate_jaccard': len(shared) / union, 'full_mean_a_pp': ma, 'full_mean_b_pp': mb, 'full_sign_a': sign(ma), 'full_sign_b': sign(mb), 'full_mean_relation': relation(ma, mb)}
            aj, bj = (a.loc[shared], b.loc[shared])
            for panel, keep in (('all', np.ones(len(shared), dtype=bool)), ('both_positive', (aj.all_observations_positive & bj.all_observations_positive).to_numpy())):
                av, bv = (aj.loc[keep], bj.loc[keep])
                metrics = profile(av.delta.to_numpy(), bv.delta.to_numpy(), av.low_yield_pp.to_numpy(), bv.low_yield_pp.to_numpy(), ma, mb)
                row.update({f'{panel}_{k}': v for k, v in metrics.items()})
            pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    pairs['jaccard_stratum'] = pd.cut(pairs.substrate_jaccard, [0, 0.25, 0.5, 0.75, 0.9, 1.000000001], labels=JAC_LABELS, right=False).astype(str)
    assert len(pairs) == 1361 and len(pairs[EDGE].drop_duplicates()) == 303

    def unordered_keys(frame: pd.DataFrame) -> pd.Series:
        return frame.apply(lambda r: (*[r[x] for x in EDGE], *sorted([r.background_a, r.background_b])), axis=1)
    ref_counts = dict(zip(unordered_keys(reference), reference.n_shared_substrate_pairs))
    assert set(unordered_keys(pairs)) == set(ref_counts)
    assert all((r.all_n_shared == ref_counts[k] for k, r in zip(unordered_keys(pairs), pairs.itertuples(index=False))))
    assert (pairs.both_positive_n_shared <= pairs.all_n_shared).all()
    for panel in ('all', 'both_positive'):
        p = panel + '_'
        ok = pairs[p + 'n_shared'].gt(0)
        assert np.allclose(pairs.loc[ok, p + 'raw_rmse_pp'] ** 2, pairs.loc[ok, p + 'centered_rmse_pp'] ** 2 + pairs.loc[ok, p + 'mean_shift_b_minus_a_pp'] ** 2, atol=1e-08)
        for t in THRESHOLDS:
            assert (pairs[p + f'n_sign_agree_{t}'] <= pairs[p + f'n_joint_decisive_{t}']).all()
        for d in ('a_to_b', 'b_to_a'):
            assert pairs.loc[ok, p + d + '_regret_vs_target_substrate_hindsight_pp'].ge(-1e-08).all()
            assert pairs.loc[ok, p + d + '_achieved_target_yield_pp'].between(-1e-08, 100 + 1e-08).all()
    spot_checks = independent_spot_checks(raw, selected, pairs)
    pairs.to_csv(RESULTS / 'matched_response_transport_pairs.csv', index=False)
    context.to_csv(RESULTS / 'matched_response_transport_contexts.csv', index=False)
    summary = []
    for panel in ('all', 'both_positive'):
        scopes = ('all_pairs', 'at_least_20_shared', 'both_positive_at_least_20_shared') if panel == 'all' else ('all_pairs', 'at_least_20_shared')
        for scope in scopes:
            g = pairs if scope == 'all_pairs' else pairs.loc[pairs['both_positive_n_shared' if scope == 'both_positive_at_least_20_shared' else panel + '_n_shared'].ge(20)]
            summary.append(summarize(g, panel, scope, 'overall', 'ALL'))
            for rel in ('same', 'opposite', 'tied'):
                summary.append(summarize(g.loc[g.full_mean_relation.eq(rel)], panel, scope, 'full_mean_relation', rel))
            for label in JAC_LABELS:
                summary.append(summarize(g.loc[g.jaccard_stratum.eq(label)], panel, scope, 'jaccard', label))
            for key, sub in g.groupby(['reaction_type', 'swap_slot'], sort=True):
                summary.append(summarize(sub, panel, scope, 'reaction_slot', '|'.join(key), *key))
                for rel, sub2 in sub.groupby('full_mean_relation', sort=True):
                    summary.append(summarize(sub2, panel, scope, 'reaction_slot_full_mean_relation', rel, *key))
                for label, sub2 in sub.groupby('jaccard_stratum', sort=True):
                    summary.append(summarize(sub2, panel, scope, 'reaction_slot_jaccard', label, *key))
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / 'matched_response_transport_summary.csv', index=False)
    edge_rows = []
    for key, g in pairs.groupby(EDGE, sort=True):
        row = {**dict(zip(EDGE, key)), 'n_background_pairs': len(g), 'n_unique_backgrounds': len(set(g.background_a) | set(g.background_b)), 'n_full_mean_same': int(g.full_mean_relation.eq('same').sum()), 'n_full_mean_opposite': int(g.full_mean_relation.eq('opposite').sum())}
        for panel in ('all', 'both_positive'):
            for scope in ('all_pairs', 'at_least_20_shared'):
                sub = g if scope == 'all_pairs' else g.loc[g[panel + '_n_shared'].ge(20)]
                prefix = panel + '_' + scope + '_'
                row[prefix + 'n_pairs'] = len(sub)
                for metric in ('spearman', 'pearson', 'centered_rmse_pp', 'abs_mean_shift_pp', 'sign_agreement_10', 'sign_agreement_15', 'sign_agreement_20', 'bidirectional_gain_vs_source_common_fixed_pp', 'bidirectional_gain_vs_target_hindsight_fixed_pp'):
                    values = sub[panel + '_' + metric]
                    row[prefix + metric + '_mean'] = float(values.mean())
                    row[prefix + metric + '_n_pairs'] = int(values.notna().sum())
        edge_rows.append(row)
    pd.DataFrame(edge_rows).to_csv(RESULTS / 'matched_response_transport_edges.csv', index=False)
    metadata = {'inputs': {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()} for name, path in INPUTS.items()}, 'strict_swap_rows': len(strict), 'repeated_edges': len(repeated), 'repeated_swap_rows': len(selected), 'input_response_rows_repeated_edges': len(joined), 'collapsed_response_rows': len(collapsed), 'unique_edge_backgrounds': len(context), 'background_pairs': len(pairs), 'duplicate_background_contexts': int(context.n_swap_rows_collapsed.gt(1).sum()), 'duplicate_background_substrate_responses': int(collapsed.n_observations.gt(1).sum()), 'response_rows_removed_by_collapse': len(joined) - len(collapsed), 'collapsed_profiles_file': {'path': str(profile_path), 'rows': len(collapsed), 'columns': list(collapsed.columns), 'sha256': hashlib.file_digest(profile_path.open('rb'), 'sha256').hexdigest(), 'smiles_lookup_unique_on': ['reaction_type', 'substrate_pair_key']}, 'orientation_reversed_swap_rows': int(selected.orientation_sign.lt(0).sum()), 'all_correlation_status_counts': pairs.all_correlation_status.value_counts().to_dict(), 'both_positive_correlation_status_counts': pairs.both_positive_correlation_status.value_counts().to_dict(), 'independent_arm_yield_spot_checks': spot_checks, 'thresholds_pp': list(THRESHOLDS), 'correlation_minimum_n': 2, 'primary_summary_minimum_shared_n': 20, 'constant_tolerance_pp': 1e-12, 'mean_sign_tie_tolerance_pp': 1e-09, 'collapse': 'Mean canonical delta and arm yields per edge x unique background x substrate pair; retain input response count; no repeated row treated as independent.', 'both_positive': 'A shared substrate qualifies only when every underlying original yield in both arms of both backgrounds is >0, before duplicate-background averaging.', 'policy': 'Source observed per-substrate delta >0 selects target canonical high arm; ties select low. Source common-panel mean >0 selects source-common fixed arm; target hindsight fixed is max(0,target common mean). Two directed gains averaged for bidirectional output.', 'weighting': 'pair_mean gives one weight per eligible background pair; edge_equal_mean first averages eligible background pairs within each edge, then gives each edge one weight; pooled threshold comparison ratios are secondary descriptive panel-size-weighted ratios.', 'invariants_passed': ['302 repeated edges / 874 swaps / 873 contexts / 1360 pairs', 'canonical component orientation', 'unique substrate pair within swap', 'delta equals yield difference', 'unique reaction-type substrate-pair to role-ordered SMILES lookup', 'swap count and normalized mean reconciliation', 'existing unordered overlap pair keys and shared panel counts', 'both-positive is subset', 'raw RMSE squared equals centered RMSE squared plus mean shift squared', 'threshold agreement numerator <= denominator', 'policy yield in [0,100]', 'nonnegative policy regret to target substrate hindsight', 'synthetic shifts, thresholds, policy, constant, empty and singleton profiles', 'independent rebuild from original arm yields for seven spot cases']}
    report = make_report(pairs, summary, context, metadata)
    (BASE / 'reports' / 'MATCHED_RESPONSE_TRANSPORT.md').write_text(report, encoding='utf-8')
    (BASE / 'audit' / 'matched_response_transport_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(summary.loc[summary.stratum_type.eq('overall'), ['panel', 'eligibility', 'n_context_pairs', 'n_edges', 'spearman_edge_equal_mean', 'centered_rmse_pp_edge_equal_mean', 'sign_agreement_20_edge_equal_mean', 'bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_mean']].to_string(index=False))
    print(json.dumps({k: metadata[k] for k in ('input_response_rows_repeated_edges', 'collapsed_response_rows', 'all_correlation_status_counts', 'both_positive_correlation_status_counts')}, indent=2))

def make_report(pairs: pd.DataFrame, summary: pd.DataFrame, context: pd.DataFrame, metadata: dict) -> str:

    def f(x: float, digits: int=3) -> str:
        return f'{x:.{digits}f}' if np.isfinite(x) else 'NA'

    def row(panel: str, kind: str='overall', label: str='ALL', scope: str='at_least_20_shared') -> pd.Series:
        return summary.loc[summary.panel.eq(panel) & summary.eligibility.eq(scope) & summary.stratum_type.eq(kind) & summary.stratum.eq(label)].iloc[0]
    lines = ['# Matched response profiles and observed-substrate policy transport', '', '## Design and denominator', '', f"All 302 repeated strict component edges are included: 874 strict swap rows, 873 unique edge-background contexts, and 1,360 unordered background pairs. The repeated-edge inputs contain {metadata['input_response_rows_repeated_edges']:,} substrate-response rows; collapsing the duplicate same-background observations leaves {metadata['collapsed_response_rows']:,} unique edge-background-substrate responses. Each response is the canonical high-component-ID yield minus low-component-ID yield in percentage points (pp).", '', 'The 28 duplicate responses belong to one finkelstein reagent2 background for CMP0173|CMP0308. Each duplicate substrate response is averaged before building profiles; its observation count is preserved in the context table. Backgrounds and substrate panels share historical observations and are not independent replications.', '', "The pair table retains zero-overlap, singleton, sparse, and constant profiles. Correlations require at least two shared substrates and nonconstant profiles in both backgrounds; two-point correlations are descriptive and fragile. Primary summaries below require at least 20 shared substrates. For a constant profile correlation is NA, while shift/error and policy quantities remain defined whenever n>0. Centered RMSE removes each background's common-panel mean, separating mean shifts from profile reshaping.", '', 'All summaries show eligible pair and edge denominators. An edge-equal mean first averages a metric across eligible background pairs of one component edge, then averages those edge means. Substrates and background pairs reused within an edge do not acquire independent-replication status. The CSV also retains pair-weighted summaries and pooled threshold ratios as descriptive alternatives.', '', '## Profile portability', '', '| Panel | Eligible pairs / edges | Correlation pairs / edges | Edge-equal Spearman | Edge-equal Pearson | Centered RMSE (pp) | Absolute mean shift (pp) |', '|---|---:|---:|---:|---:|---:|---:|']
    for panel in ('all', 'both_positive'):
        r = row(panel)
        lines.append(f'| {panel} | {int(r.n_context_pairs)} / {int(r.n_edges)} | {int(r.spearman_n_context_pairs)} / {int(r.spearman_n_edges)} | {f(r.spearman_edge_equal_mean)} | {f(r.pearson_edge_equal_mean)} | {f(r.centered_rmse_pp_edge_equal_mean)} | {f(r.abs_mean_shift_pp_edge_equal_mean)} |')
    lines += ['', 'Both-positive sensitivity requires every underlying original yield in all four condition arms (two components in each background) to be strictly positive before duplicate averaging. This removes zero-yield arms, changes the shared panel, and does not establish an uncensored biological population.', '', '| Panel | Threshold (pp) | Jointly decisive comparisons | Agreeing comparisons | Eligible pairs / edges | Edge-equal conditional sign agreement |', '|---|---:|---:|---:|---:|---:|']
    for panel in ('all', 'both_positive'):
        r = row(panel)
        for t in THRESHOLDS:
            lines.append(f"| {panel} | {t} | {int(r[f'n_joint_decisive_{t}_comparisons']):,} | {int(r[f'n_sign_agree_{t}_comparisons']):,} | {int(r[f'sign_agreement_{t}_n_context_pairs'])} / {int(r[f'sign_agreement_{t}_n_edges'])} | {f(r[f'sign_agreement_{t}_edge_equal_mean'])} |")
    lines += ['', 'Decisive sign agreement uses only shared substrates with |delta| >= threshold in both backgrounds. Its denominator is the number jointly decisive, not all shared substrates. Counts pooled across background pairs reuse substrates and are not sample sizes for inference. Balanced support is exactly 2 x min(positive-tail fraction, negative-tail fraction).', '', '## Mean ranking versus response shape', '', '| Full-panel mean signs | Panel | Eligible pairs / edges | Edge-equal Spearman | Centered RMSE (pp) | Gain over target hindsight fixed (pp) |', '|---|---|---:|---:|---:|---:|']
    for rel in ('same', 'opposite'):
        for panel in ('all', 'both_positive'):
            r = row(panel, 'full_mean_relation', rel)
            lines.append(f'| {rel} | {panel} | {int(r.n_context_pairs)} / {int(r.n_edges)} | {f(r.spearman_edge_equal_mean)} | {f(r.centered_rmse_pp_edge_equal_mean)} | {f(r.bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_mean)} |')
    opposite = pairs.full_mean_relation.eq('opposite')
    available = pairs.all_mean_relation.isin(['same', 'opposite'])
    lines += ['', f"Across all 1,360 pairs, full-panel means have opposite signs in {int(opposite.sum())} pairs. Among the {int((opposite & available).sum())} with non-tied common-panel means, {int((opposite & pairs.all_mean_relation.eq('same')).sum())} become same-sign on the identical shared panel; conversely, {int((pairs.full_mean_relation.eq('same') & pairs.all_mean_relation.eq('opposite')).sum())} same-sign full-panel pairs become opposite-sign on the common panel. These are descriptive composition-associated changes: the common-panel contrast holds substrate membership fixed, but does not establish a causal composition mechanism. Pairwise common-minus-full shifts are retained for each background.", '', '## Policy transport on already observed shared substrates', '', "For A to B, the observed sign of each substrate's response in A chooses the corresponding component arm in B; source delta ties choose canonical low. Achieved incremental yield above target low is mean(1[source delta > 0] x target delta). The source-common fixed policy chooses high only if the source common-panel mean is positive; its target increment is 1[source common mean > 0] x target common mean. The target hindsight best fixed policy earns max(0, target common mean). B to A is calculated analogously, and the table reports their arithmetic mean. A source all-panel fixed baseline is also retained to expose panel-composition effects.", '', '| Panel | Eligible pairs / edges | Gain over source-common fixed (pp) | Gain over target hindsight fixed (pp) | Positive gain pairs / eligible pairs | Edge-equal fraction positive |', '|---|---:|---:|---:|---:|---:|']
    for panel in ('all', 'both_positive'):
        r = row(panel)
        lines.append(f'| {panel} | {int(r.n_context_pairs)} / {int(r.n_edges)} | {f(r.bidirectional_gain_vs_source_common_fixed_pp_edge_equal_mean)} | {f(r.bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_mean)} | {int(r.bidirectional_gain_vs_target_hindsight_fixed_pp_n_positive)} / {int(r.bidirectional_gain_vs_target_hindsight_fixed_pp_n_context_pairs)} | {f(r.bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_fraction_positive)} |')
    support = row('all', scope='both_positive_at_least_20_shared')
    lines += ['', f'On the identical {int(support.n_context_pairs)} background pairs / {int(support.n_edges)} edges qualifying for the both-positive sensitivity, the full-panel edge-equal Spearman is {f(support.spearman_edge_equal_mean)} and policy gain over target hindsight fixed is {f(support.bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_mean)} pp. The both-positive differences therefore combine a substrate restriction with changing eligibility; the full-universe versus sensitivity contrast must not be interpreted as a zero-yield effect alone.', '', 'This is transport of observed substrate-specific decisions between backgrounds on the same historical substrates. Source signs are measured on those substrates, and target outcomes are evaluated retrospectively. It is not new-substrate prediction, prospective policy validation, independent chemical replication, or evidence for a molecular mechanism. The hindsight fixed comparator is stronger than choosing the target arm from the source mean, but remains a retrospective descriptive benchmark.', '', '## Jaccard and reaction-slot stratification', '', '| Jaccard bin | Panel | Eligible pairs / edges | Edge-equal Spearman | Gain over target hindsight fixed (pp) |', '|---|---|---:|---:|---:|']
    for label in JAC_LABELS:
        for panel in ('all', 'both_positive'):
            r = row(panel, 'jaccard', label)
            lines.append(f'| {label} | {panel} | {int(r.n_context_pairs)} / {int(r.n_edges)} | {f(r.spearman_edge_equal_mean)} | {f(r.bidirectional_gain_vs_target_hindsight_fixed_pp_edge_equal_mean)} |')
    lines += ['', 'Reaction class x changed-slot summaries and their full-mean-sign and Jaccard strata are supplied in matched_response_transport_summary.csv. These strata can differ in edge composition and panel support; contrasts are descriptive without p-values.', '', '## Files and checks', '', '- `matched_response_transport_pairs.csv`: all 1,360 auditable background pairs, counts, correlation status, profile errors, composition shifts, threshold denominators, and both directed policies for both panels.', '- `matched_response_transport_contexts.csv`: 873 unique context records with duplicate observation counts and normalized means.', '- `matched_response_transport_edges.csv`: all 302 edges, with within-edge pair means and eligible pair counts.', '- `matched_response_transport_summary.csv`: overall, mean-sign, Jaccard, reaction-slot, and crossed descriptive summaries, with pair and edge denominators.', '- `matched_response_profiles.parquet`: reusable 251,835-row collapsed response table with canonical delta, mean low/high arm yields, conservative both-positive flag, original observation count, context keys, substrate-pair key, and validated role-ordered substrate SMILES. This compressed intermediate supports the subsequent partner-additive decomposition without reconstructing profiles again.', '- `audit/matched_response_transport_metadata.json`: input hashes, definitions, coverage, and passed invariants.', '', "The reconstruction reconciles each strict swap's original response count and normalized mean, all 1,360 unordered background-pair keys, and every shared-panel count to the existing overlap analysis. Numerical checks verify the centered-error decomposition, threshold bounds, policy yield bounds, and nonnegative regret relative to per-substrate target hindsight. Synthetic cases check a pure mean shift, sign thresholds, policy arithmetic, constant profiles, zero overlap, and singleton overlap. Seven independent spot cases rebuild canonical arms directly from original y0/y1 values and verify correlation ranks, threshold denominators, centered error, and both directed policies by achieved absolute target yield; the audit records per-case discrepancy and reversed-orientation coverage. Original source files and upstream results are unchanged.", '']
    return '\n'.join(lines)
if __name__ == '__main__':
    main()
