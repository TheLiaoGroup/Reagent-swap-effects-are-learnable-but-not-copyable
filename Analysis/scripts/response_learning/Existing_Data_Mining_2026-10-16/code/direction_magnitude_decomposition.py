"""Decompose cross-background component-edge effects into direction and magnitude.

The input is the existing strict identity-only condition-network table.  The
unit of cross-background comparison is a component edge within a reaction type
and changed slot.  Multiple strict swaps sharing the same background
fingerprint are pooled before edge-level summaries, so they are not counted as
independent backgrounds."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
META = BASE / 'results' / 'condition_network_meta.csv'
REGISTRY = BASE / 'results' / 'swap_registry.csv'
OUT_BACKGROUND = BASE / 'results' / 'direction_magnitude_background.csv'
OUT_EDGE = BASE / 'results' / 'direction_magnitude_edge.csv'
OUT_SUMMARY = BASE / 'results' / 'direction_magnitude_summary.csv'
OUT_AUDIT = BASE / 'audit' / 'direction_magnitude_metadata.json'
OUT_REPORT = BASE / 'reports' / 'DIRECTION_MAGNITUDE_DECOMPOSITION.md'
MAGNITUDE_HIGH_PP = 10.0
MAGNITUDE_LOW_PP = 5.0
DRIFT_HIGH_PP = 10.0
CONSISTENCY_HIGH = 0.75
CONSISTENCY_LOW = 0.5
EDGE_KEYS = ['reaction_type', 'swap_slot', 'component_pair']

def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors='coerce').to_numpy(float)
    w = pd.to_numeric(weights, errors='coerce').to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

def _pairwise_median(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    diffs = [abs(float(a) - float(b)) for a, b in combinations(values, 2)]
    return float(np.median(diffs)) if diffs else np.nan

def _background_direction(delta: float) -> str:
    if not np.isfinite(delta) or abs(delta) < 1e-09:
        return 'tie'
    return 'positive' if delta > 0 else 'negative'

def _direction_class(consistency: float, n_non_tie: int) -> str:
    if n_non_tie == 0 or not np.isfinite(consistency):
        return 'all_tied'
    if consistency >= CONSISTENCY_HIGH:
        return 'direction_preserved'
    if consistency <= CONSISTENCY_LOW:
        return 'direction_unstable'
    return 'direction_mixed'

def _magnitude_class(median_magnitude: float) -> str:
    if not np.isfinite(median_magnitude):
        return 'unquantified'
    if median_magnitude >= MAGNITUDE_HIGH_PP:
        return 'high_magnitude'
    if median_magnitude < MAGNITUDE_LOW_PP:
        return 'low_magnitude'
    return 'intermediate_magnitude'

def _state_label(direction_class: str, magnitude_class: str, drift_range: float, max_magnitude: float) -> str:
    if direction_class == 'all_tied':
        return 'all_tied'
    drift = np.isfinite(drift_range) and drift_range >= DRIFT_HIGH_PP
    if direction_class == 'direction_preserved' and drift:
        return 'direction_preserved_magnitude_drift'
    if direction_class == 'direction_preserved':
        return 'direction_preserved_magnitude_stable'
    if direction_class == 'direction_unstable' and (magnitude_class == 'high_magnitude' or (np.isfinite(max_magnitude) and max_magnitude >= MAGNITUDE_HIGH_PP)):
        return 'high_magnitude_direction_unstable'
    if direction_class == 'direction_unstable':
        return 'direction_unstable_low_or_moderate_magnitude'
    return 'direction_mixed'

def _collapse_backgrounds(g: pd.DataFrame) -> pd.DataFrame:
    """Pool duplicate rows with the same background fingerprint."""
    rows: list[dict[str, object]] = []
    for bg, b in g.groupby('background_fingerprint', sort=True, dropna=False):
        n_pairs = pd.to_numeric(b['n_pairs'], errors='coerce').fillna(0.0)
        pos = pd.to_numeric(b['n_pos_20_norm'], errors='coerce').fillna(0.0).sum()
        neg = pd.to_numeric(b['n_neg_20_norm'], errors='coerce').fillna(0.0).sum()
        total_pairs = float(n_pairs.sum())
        mean_delta = _weighted_mean(b['mean_delta_norm_pp'], n_pairs)
        mean_abs_pair_delta = _weighted_mean(b['mean_abs_delta_pp'], n_pairs)
        balanced = float(2.0 * min(pos, neg) / total_pairs) if total_pairs else np.nan
        direction = _background_direction(mean_delta)
        tail_direction = 'positive' if pos > neg else 'negative' if neg > pos else 'tie'
        rows.append({'reaction_type': b['reaction_type'].iloc[0], 'swap_slot': b['swap_slot'].iloc[0], 'component_pair': b['component_pair'].iloc[0], 'component_pair_orientation': b['component_pair_orientation'].iloc[0], 'background_fingerprint': bg, 'n_swap_rows_collapsed': int(len(b)), 'n_pairs': total_pairs, 'mean_delta_norm_pp': mean_delta, 'abs_mean_delta_norm_pp': abs(mean_delta) if np.isfinite(mean_delta) else np.nan, 'mean_abs_delta_pp': mean_abs_pair_delta, 'n_pos_20_norm': int(pos), 'n_neg_20_norm': int(neg), 'balanced_support_20': balanced, 'direction_sign': direction, 'tail_direction20': tail_direction, 'tail_mixed20': bool(pos > 0 and neg > 0), 'frac_either_zero': _weighted_mean(b['frac_either_zero'], n_pairs), 'n_check_false_pairs': int(pd.to_numeric(b['n_check_false_pairs'], errors='coerce').fillna(0).sum()), 'swap_ids': '||'.join(sorted(b['swap_id'].astype(str)))})
    return pd.DataFrame(rows)

def _summarize_edge(bg: pd.DataFrame) -> dict[str, object]:
    edge_key = {k: bg[k].iloc[0] for k in EDGE_KEYS}
    deltas = pd.to_numeric(bg['mean_delta_norm_pp'], errors='coerce').to_numpy(float)
    magnitudes = np.abs(deltas)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    directions = bg['direction_sign'].astype(str)
    pos = int((directions == 'positive').sum())
    neg = int((directions == 'negative').sum())
    ties = int((directions == 'tie').sum())
    non_tie = pos + neg
    consistency = float(max(pos, neg) / non_tie) if non_tie else np.nan
    tail = bg['tail_direction20'].astype(str)
    tail_pos = int((tail == 'positive').sum())
    tail_neg = int((tail == 'negative').sum())
    tail_tie = int((tail == 'tie').sum())
    tail_non_tie = tail_pos + tail_neg
    tail_consistency = float(max(tail_pos, tail_neg) / tail_non_tie) if tail_non_tie else np.nan
    median_mag = float(np.median(magnitudes)) if len(magnitudes) else np.nan
    mean_mag = float(np.mean(magnitudes)) if len(magnitudes) else np.nan
    signed_sd = float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else np.nan
    mag_sd = float(np.std(magnitudes, ddof=1)) if len(magnitudes) >= 2 else np.nan
    mag_range = float(np.max(magnitudes) - np.min(magnitudes)) if len(magnitudes) else np.nan
    q25, q75 = np.percentile(magnitudes, [25, 75]) if len(magnitudes) else (np.nan, np.nan)
    magnitude_class = _magnitude_class(median_mag)
    direction_class = _direction_class(consistency, non_tie)
    rec: dict[str, object] = {**edge_key, 'n_unique_backgrounds': int(len(bg)), 'n_swap_rows': int(bg['n_swap_rows_collapsed'].sum()), 'n_backgrounds_positive_sign': pos, 'n_backgrounds_negative_sign': neg, 'n_backgrounds_tied_sign': ties, 'n_backgrounds_non_tied_sign': non_tie, 'direction_sign_consistency': consistency, 'direction_fraction_positive': float(pos / non_tie) if non_tie else np.nan, 'direction_fraction_negative': float(neg / non_tie) if non_tie else np.nan, 'direction_class': direction_class, 'n_backgrounds_positive_tail20': tail_pos, 'n_backgrounds_negative_tail20': tail_neg, 'n_backgrounds_tied_tail20': tail_tie, 'tail_direction20_consistency': tail_consistency, 'n_backgrounds_tail_mixed20': int(bg['tail_mixed20'].sum()), 'fraction_backgrounds_tail_mixed20': float(bg['tail_mixed20'].mean()), 'median_abs_mean_delta_pp': median_mag, 'mean_abs_mean_delta_pp': mean_mag, 'min_abs_mean_delta_pp': float(np.min(magnitudes)) if len(magnitudes) else np.nan, 'max_abs_mean_delta_pp': float(np.max(magnitudes)) if len(magnitudes) else np.nan, 'signed_delta_mean_pp': float(np.mean(deltas)) if len(deltas) else np.nan, 'signed_delta_median_pp': float(np.median(deltas)) if len(deltas) else np.nan, 'signed_delta_sd_pp': signed_sd, 'signed_delta_range_pp': float(np.max(deltas) - np.min(deltas)) if len(deltas) else np.nan, 'magnitude_drift_sd_pp': mag_sd, 'magnitude_drift_range_pp': mag_range, 'magnitude_drift_iqr_pp': float(q75 - q25) if np.isfinite(q25) else np.nan, 'magnitude_pairwise_median_abs_diff_pp': _pairwise_median(magnitudes), 'magnitude_drift_range_over_median': float(mag_range / median_mag) if median_mag > 0 else np.nan, 'median_mean_abs_pair_delta_pp': float(np.nanmedian(bg['mean_abs_delta_pp'])), 'mean_mean_abs_pair_delta_pp': float(np.nanmean(bg['mean_abs_delta_pp'])), 'within_to_mean_abs_magnitude_ratio': float(np.nanmedian(bg['mean_abs_delta_pp']) / median_mag) if median_mag > 0 else np.nan, 'median_balanced_support20': float(np.nanmedian(bg['balanced_support_20'])), 'mean_balanced_support20': float(np.nanmean(bg['balanced_support_20'])), 'min_balanced_support20': float(np.nanmin(bg['balanced_support_20'])), 'max_balanced_support20': float(np.nanmax(bg['balanced_support_20'])), 'total_n_pairs': float(bg['n_pairs'].sum()), 'magnitude_class': magnitude_class}
    rec['high_magnitude_any_background'] = bool(np.isfinite(np.max(magnitudes)) and np.max(magnitudes) >= MAGNITUDE_HIGH_PP) if len(magnitudes) else False
    rec['state_label'] = _state_label(direction_class, magnitude_class, mag_range, rec['max_abs_mean_delta_pp'])
    return rec

def _summary_row(label: str, g: pd.DataFrame) -> dict[str, object]:

    def median(col: str) -> float:
        return float(pd.to_numeric(g[col], errors='coerce').median())

    def mean(col: str) -> float:
        return float(pd.to_numeric(g[col], errors='coerce').mean())
    n = len(g)
    return {'reaction_type': label, 'swap_slot': 'ALL' if label == 'ALL' else str(g['swap_slot'].iloc[0]), 'n_edges': n, 'n_unique_backgrounds_total': int(g['n_unique_backgrounds'].sum()), 'median_backgrounds_per_edge': median('n_unique_backgrounds'), 'mean_backgrounds_per_edge': mean('n_unique_backgrounds'), 'median_direction_consistency': median('direction_sign_consistency'), 'mean_direction_consistency': mean('direction_sign_consistency'), 'fraction_direction_consistency_ge_0_75': float((g['direction_sign_consistency'] >= CONSISTENCY_HIGH).mean()), 'fraction_direction_consistency_le_0_50': float((g['direction_sign_consistency'] <= CONSISTENCY_LOW).mean()), 'median_abs_mean_delta_pp': median('median_abs_mean_delta_pp'), 'mean_abs_mean_delta_pp': mean('mean_abs_mean_delta_pp'), 'median_mean_abs_pair_delta_pp': median('median_mean_abs_pair_delta_pp'), 'mean_mean_abs_pair_delta_pp': mean('mean_mean_abs_pair_delta_pp'), 'median_magnitude_drift_range_pp': median('magnitude_drift_range_pp'), 'mean_magnitude_drift_range_pp': mean('magnitude_drift_range_pp'), 'median_magnitude_drift_sd_pp': median('magnitude_drift_sd_pp'), 'median_magnitude_pairwise_diff_pp': median('magnitude_pairwise_median_abs_diff_pp'), 'median_balanced_support20': median('median_balanced_support20'), 'mean_balanced_support20': mean('mean_balanced_support20'), 'median_fraction_backgrounds_tail_mixed20': median('fraction_backgrounds_tail_mixed20'), 'mean_fraction_backgrounds_tail_mixed20': mean('fraction_backgrounds_tail_mixed20'), 'median_within_to_mean_abs_magnitude_ratio': median('within_to_mean_abs_magnitude_ratio'), 'mean_within_to_mean_abs_magnitude_ratio': mean('within_to_mean_abs_magnitude_ratio'), 'fraction_direction_preserved_magnitude_drift': float((g['state_label'] == 'direction_preserved_magnitude_drift').mean()), 'fraction_high_magnitude_direction_unstable': float((g['state_label'] == 'high_magnitude_direction_unstable').mean()), 'fraction_direction_preserved_magnitude_stable': float((g['state_label'] == 'direction_preserved_magnitude_stable').mean()), 'fraction_direction_unstable_low_or_moderate_magnitude': float((g['state_label'] == 'direction_unstable_low_or_moderate_magnitude').mean()), 'n_direction_preserved_magnitude_drift': int((g['state_label'] == 'direction_preserved_magnitude_drift').sum()), 'n_high_magnitude_direction_unstable': int((g['state_label'] == 'high_magnitude_direction_unstable').sum()), 'n_direction_preserved_magnitude_stable': int((g['state_label'] == 'direction_preserved_magnitude_stable').sum()), 'n_direction_unstable_low_or_moderate_magnitude': int((g['state_label'] == 'direction_unstable_low_or_moderate_magnitude').sum())}

def _fmt(x: object, digits: int=2) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return 'NA'
    return f'{float(x):.{digits}f}'

def main() -> None:
    meta = pd.read_csv(META)
    registry = pd.read_csv(REGISTRY)
    strict = registry.loc[registry['match_class'].eq('strict_identity_only'), ['swap_id', 'mean_abs_delta_pp']].copy()
    strict['swap_id'] = strict['swap_id'].astype(str)
    if strict['swap_id'].duplicated().any():
        raise ValueError('duplicate strict swap_id in registry')
    swap = meta.loc[meta['row_level'].eq('swap')].copy()
    swap['swap_id'] = swap['swap_id'].astype(str)
    swap = swap.merge(strict, on='swap_id', how='left', validate='one_to_one')
    if swap['mean_abs_delta_pp'].isna().any():
        raise ValueError('strict swap rows missing registry mean_abs_delta_pp')
    pair = meta.loc[meta['row_level'].eq('component_pair')].copy()
    repeated = pair.loc[pair['pair_n_unique_backgrounds'].ge(2), EDGE_KEYS].drop_duplicates()
    if repeated.empty:
        raise ValueError('no component edges repeated across at least two backgrounds')
    swap = swap.merge(repeated.assign(_repeated=True), on=EDGE_KEYS, how='inner', validate='many_to_one')
    if swap.empty:
        raise ValueError('repeated edge selection produced no swap rows')
    swap['n_pairs'] = pd.to_numeric(swap['n_pairs'], errors='coerce').fillna(0.0)
    for col in ['mean_delta_norm_pp', 'n_pos_20_norm', 'n_neg_20_norm', 'balanced_support_20', 'frac_either_zero', 'n_check_false_pairs']:
        swap[col] = pd.to_numeric(swap[col], errors='coerce')
    collapsed = []
    for _, group in swap.groupby(EDGE_KEYS, sort=True):
        collapsed.append(_collapse_backgrounds(group))
    bg = pd.concat(collapsed, ignore_index=True) if collapsed else pd.DataFrame()
    if bg.empty:
        raise ValueError('background collapse produced no rows')
    edge_rows = [_summarize_edge(g) for _, g in bg.groupby(EDGE_KEYS, sort=True)]
    edge = pd.DataFrame(edge_rows).sort_values(EDGE_KEYS).reset_index(drop=True)
    if (edge['n_unique_backgrounds'] < 2).any():
        raise AssertionError('edge output contains a single-background edge')
    summary_rows = [_summary_row('ALL', edge)]
    for (reaction_type, slot), g in edge.groupby(['reaction_type', 'swap_slot'], sort=True):
        rec = _summary_row(str(reaction_type), g)
        rec['reaction_type'] = reaction_type
        rec['swap_slot'] = slot
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    OUT_BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    OUT_BACKGROUND.write_text(bg.to_csv(index=False), encoding='utf-8-sig')
    OUT_EDGE.write_text(edge.to_csv(index=False), encoding='utf-8-sig')
    OUT_SUMMARY.write_text(summary.to_csv(index=False), encoding='utf-8-sig')
    audit = {'input_meta': str(META), 'input_registry': str(REGISTRY), 'strict_swap_rows_in_meta': int(meta['row_level'].eq('swap').sum()), 'strict_registry_rows': int(len(strict)), 'repeated_edges_at_least_two_unique_backgrounds': int(len(repeated)), 'retained_swap_rows_before_background_collapse': int(len(swap)), 'retained_unique_background_rows': int(len(bg)), 'retained_edge_rows': int(len(edge)), 'reaction_slot_strata': int(edge.groupby(['reaction_type', 'swap_slot']).ngroups), 'same_background_duplicate_rows_collapsed': int(len(swap) - len(bg)), 'thresholds': {'magnitude_high_pp': MAGNITUDE_HIGH_PP, 'magnitude_low_pp': MAGNITUDE_LOW_PP, 'drift_high_pp': DRIFT_HIGH_PP, 'direction_consistency_high': CONSISTENCY_HIGH, 'direction_consistency_low': CONSISTENCY_LOW}, 'state_counts': edge['state_label'].value_counts().to_dict(), 'edge_unique_key_assertion': bool(not edge.duplicated(EDGE_KEYS).any())}
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    all_row = summary.iloc[0]
    top_drift = edge.sort_values(['magnitude_drift_range_pp', 'median_abs_mean_delta_pp'], ascending=False).head(12)
    top_unstable = edge.loc[edge['state_label'].eq('high_magnitude_direction_unstable')].sort_values('median_abs_mean_delta_pp', ascending=False).head(12)
    lines = ['# Strict component edges: direction versus magnitude', '', '本报告把严格身份交换边的跨背景行为拆成两个维度：组件边在规范方向上的方向是否保持，以及效应幅度（背景内平均 Δ 的绝对值）是否漂移。分析只使用既有 `condition_network_meta.csv` 和 `swap_registry.csv`，没有重新读取或修改原始数据。', '', '## 口径', '', f"保留 {int(all_row['n_edges']):,} 条组件边；每条边至少出现在两个唯一 `background_fingerprint` 中。先把同一背景下的重复严格交换按共同底物对数加权合并，再跨背景计算方向和幅度，因此同一背景的多条条件记录不被当作多个背景。规范方向为组件 ID 的低值到高值，`mean_delta_norm_pp` 和方向符号均沿用既有网络表。", '', '- `direction_sign_consistency`：去除背景内平均 Δ 的零平局后，较多方向所占的背景比例；0.75 以上标为 `direction_preserved`，0.50 以下标为 `direction_unstable`。', '- `median_abs_mean_delta_pp` / `mean_abs_mean_delta_pp`：跨背景的平均 Δ 绝对值，单位为产率百分点（pp）。', '- `magnitude_drift_range_pp`：跨背景绝对平均 Δ 的最大值减最小值；10 pp 以上标为幅度漂移。该阈值用于描述性分区，不是统计显著性界线。', '- `median_balanced_support20`：各背景 `balanced_support_20` 的中位数；它反映背景内正负 20 pp 尾部的平衡程度，不是重复概率。', '', '## 全库结果', '', f"跨背景方向保持的中位一致性为 {_fmt(all_row['median_direction_consistency'])}，平均为 {_fmt(all_row['mean_direction_consistency'])}；先在各边的背景内汇总、再跨边取中位数，绝对平均效应为 {_fmt(all_row['median_abs_mean_delta_pp'])} pp，背景内逐底物对平均绝对 Δ 为 {_fmt(all_row['median_mean_abs_pair_delta_pp'])} pp。先逐边计算再取中位数的比值为 {_fmt(all_row['median_within_to_mean_abs_magnitude_ratio'])}；它不等于前述两个全库中位数之比。三角不等式保证该比值至少为 1，应解释为正负抵消的描述量，不能作为分布宽度或隐藏区的独立证明。幅度漂移范围中位数为 {_fmt(all_row['median_magnitude_drift_range_pp'])} pp，背景内 balanced support20 的中位数为 {_fmt(all_row['median_balanced_support20'])}。", '', '| 状态 | 边数 |', '|---|---:|']
    for state, count in edge['state_label'].value_counts().items():
        lines.append(f'| {state} | {int(count)} |')
    lines += ['', '## Reaction type × changed slot', '', '| reaction type | slot | edges | median direction consistency | median abs mean Δ (pp) | median within/mean magnitude ratio | median magnitude drift range (pp) | median balanced support20 | preserved + drift | high magnitude + unstable |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in summary.iloc[1:].sort_values('n_edges', ascending=False).iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {int(x.n_edges)} | {_fmt(x.median_direction_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.median_within_to_mean_abs_magnitude_ratio)} | {_fmt(x.median_magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} | {int(x.n_direction_preserved_magnitude_drift)} | {int(x.n_high_magnitude_direction_unstable)} |')
    lines += ['', '## Direction preserved but magnitude drifted', '', '这些边在大多数背景中保持同一规范方向，但背景间效应幅度相差至少 10 pp。它们是‘方向可携带、幅度需重新校准’的历史候选，不应直接解释为可迁移的固定收益。', '', '| reaction | slot | component pair | backgrounds | direction consistency | median abs mean Δ (pp) | drift range (pp) | median balanced support20 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    preserved_drift = edge.loc[edge['state_label'].eq('direction_preserved_magnitude_drift')].sort_values(['magnitude_drift_range_pp', 'median_abs_mean_delta_pp'], ascending=False).head(25)
    for _, x in preserved_drift.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {_fmt(x.direction_sign_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} |')
    if preserved_drift.empty:
        lines.append('| none | | | | | | | |')
    lines += ['', '## High magnitude but direction unstable', '', '这些边至少有一个背景的绝对平均效应达到 10 pp，但方向一致性不超过 0.50；它们直接揭示背景依赖，不能被压缩成单一组件排序。跨背景的中位数和均值仍在 CSV 中单独给出。', '', '| reaction | slot | component pair | backgrounds | direction consistency | median abs mean Δ (pp) | drift range (pp) | median balanced support20 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for _, x in top_unstable.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {_fmt(x.direction_sign_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} |')
    if top_unstable.empty:
        lines.append('| none | | | | | | | |')
    lines += ['', '## 幅度漂移最大的边（审计入口）', '', '| reaction | slot | component pair | backgrounds | direction class | median abs mean Δ (pp) | drift range (pp) | signed range (pp) |', '|---|---:|---|---:|---|---:|---:|---:|']
    for _, x in top_drift.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {x.direction_class} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.signed_delta_range_pp)} |')
    lines += ['', '## 解释边界', '', '这些是历史 HTE 条件交换的描述性分解。背景之间可能共享底物面板、参照条件、板/批次标签和测量流程；同一组件边的跨背景一致性不等于独立实验重复。方向保持也不说明机理相同，幅度漂移也不单独定位其来源。所有幅度均为匹配产率百分点，零值、缺测和产品身份限制沿用主流水线定义。', '', '## 复现', '', '```powershell', 'python code/direction_magnitude_decomposition.py', '```', '', '输出：`results/direction_magnitude_background.csv`、`results/direction_magnitude_edge.csv`、`results/direction_magnitude_summary.csv`，以及 `audit/direction_magnitude_metadata.json`。']
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote background={OUT_BACKGROUND} rows={len(bg)} edge={len(edge)} summary={len(summary)}')
if __name__ == '__main__':
    main()
