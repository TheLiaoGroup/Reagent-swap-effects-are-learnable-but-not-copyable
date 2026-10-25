"""Separate background portability from substrate-panel overlap.

The unit of the detailed table is an unordered pair of distinct background
contexts for one strict component edge.  A substrate pair reused across two
contexts is not an independent replication; all summaries therefore report
both pairwise context counts and edge-balanced averages."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
META = BASE / 'results' / 'condition_network_meta.csv'
BG = BASE / 'results' / 'direction_magnitude_background.csv'
EDGE = BASE / 'results' / 'direction_magnitude_edge.csv'
SUMMARY = BASE / 'results' / 'direction_magnitude_summary.csv'
PORT = BASE / 'results' / 'context_portability_summary.csv'
DELTAS = BASE / 'results' / 'swap_deltas.parquet'
OUT_PAIRS = BASE / 'results' / 'background_portability_overlap.csv'
OUT_SUMMARY = BASE / 'results' / 'background_portability_overlap_summary.csv'
OUT_AUDIT = BASE / 'audit' / 'background_portability_overlap_metadata.json'
OUT_REPORT = BASE / 'reports' / 'BACKGROUND_PORTABILITY_OVERLAP.md'
EDGE_KEYS = ['reaction_type', 'swap_slot', 'component_pair']
JAC_EDGES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.000000001]
JAC_LABELS = ['J<0.25', 'J0.25-0.50', 'J0.50-0.75', 'J0.75-0.90', 'J>=0.90']
OVERLAP_EDGES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.000000001]
OVERLAP_LABELS = ['Omin<0.25', 'Omin0.25-0.50', 'Omin0.50-0.75', 'Omin0.75-0.90', 'Omin>=0.90']

def _safe_float(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def _direction_agreement(a: object, b: object) -> float:
    a, b = (str(a), str(b))
    if a == 'tie' or b == 'tie' or a == 'nan' or (b == 'nan'):
        return np.nan
    return float(a == b)

def _weighted_mean(g: pd.DataFrame, value: str, weight: str='n_pairs') -> float:
    v = pd.to_numeric(g[value], errors='coerce').to_numpy(float)
    w = pd.to_numeric(g[weight], errors='coerce').to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

def _summarize_bin(g: pd.DataFrame, row_level: str, stratum: str, label: str) -> dict[str, object]:
    """Summarize a fixed overlap bin, retaining pairwise and edge counts."""
    eligible = g.loc[g['direction_agreement'].notna()].copy()
    tail = g.loc[g['tail_agreement'].notna()].copy()
    edge_agree = eligible.groupby(EDGE_KEYS, sort=False)['direction_agreement'].mean() if not eligible.empty else pd.Series(dtype=float)
    edge_tail = tail.groupby(EDGE_KEYS, sort=False)['tail_agreement'].mean() if not tail.empty else pd.Series(dtype=float)
    return {'row_level': row_level, 'stratum_type': stratum, 'stratum': label, 'n_context_pairs': int(len(g)), 'n_direction_pairs': int(len(eligible)), 'n_tail_pairs': int(len(tail)), 'n_edges': int(len(edge_agree)), 'n_edges_tail': int(len(edge_tail)), 'pairwise_direction_agreement': float(eligible['direction_agreement'].mean()) if not eligible.empty else np.nan, 'edge_balanced_direction_agreement': float(edge_agree.mean()) if not edge_agree.empty else np.nan, 'pairwise_tail_agreement': float(tail['tail_agreement'].mean()) if not tail.empty else np.nan, 'edge_balanced_tail_agreement': float(edge_tail.mean()) if not edge_tail.empty else np.nan, 'pairwise_median_abs_delta_difference_pp': float(g['abs_mean_delta_difference_pp'].median()) if len(g) else np.nan, 'pairwise_mean_abs_delta_difference_pp': float(g['abs_mean_delta_difference_pp'].mean()) if len(g) else np.nan, 'edge_balanced_median_abs_delta_difference_pp': float(g.groupby(EDGE_KEYS)['abs_mean_delta_difference_pp'].median().median()) if len(g) else np.nan, 'median_jaccard': float(g['substrate_jaccard'].median()) if len(g) else np.nan, 'median_overlap_min_fraction': float(g['overlap_min_fraction'].median()) if len(g) else np.nan, 'median_shared_substrate_pairs': float(g['n_shared_substrate_pairs'].median()) if len(g) else np.nan, 'reaction_type': 'ALL', 'swap_slot': 'ALL'}

def main() -> None:
    meta = pd.read_csv(META)
    if 'row_level' not in meta or 'swap_id' not in meta:
        raise ValueError('condition_network_meta.csv missing row_level or swap_id')
    swap = meta.loc[meta['row_level'].eq('swap')].copy()
    swap['swap_id'] = swap['swap_id'].astype(str)
    strict_ids = set(swap['swap_id'])
    bg = pd.read_csv(BG)
    edge = pd.read_csv(EDGE)
    if not set(EDGE_KEYS).issubset(bg.columns) or not set(EDGE_KEYS).issubset(edge.columns):
        raise ValueError('direction magnitude inputs missing edge keys')
    repeated_edges = edge.loc[edge['n_unique_backgrounds'].ge(2), EDGE_KEYS].drop_duplicates()
    bg = bg.merge(repeated_edges.assign(_repeated=True), on=EDGE_KEYS, how='inner', validate='many_to_one')
    if bg.empty:
        raise ValueError('no repeated-background edges')
    deltas = pd.read_parquet(DELTAS)
    deltas['swap_id'] = deltas['swap_id'].astype(str)
    deltas = deltas.loc[deltas['swap_id'].isin(strict_ids)].copy()
    sets = deltas.groupby('swap_id')['substrate_pair_key'].agg(lambda s: set(s.astype(str))).to_dict()
    if not sets:
        raise ValueError('no substrate sets for strict swaps')
    context_records: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for _, row in bg.iterrows():
        ids = [x for x in str(row['swap_ids']).split('||') if x and x != 'nan']
        missing = [x for x in ids if x not in sets]
        missing_ids.extend(missing)
        panel = set().union(*(sets.get(x, set()) for x in ids))
        context_records.append({**{k: row[k] for k in EDGE_KEYS}, 'background_fingerprint': row['background_fingerprint'], 'swap_ids': row['swap_ids'], 'n_swap_rows_collapsed': int(row['n_swap_rows_collapsed']), 'n_pairs': _safe_float(row['n_pairs']), 'n_substrate_pairs': int(len(panel)), 'substrate_set': panel, 'mean_delta_norm_pp': _safe_float(row['mean_delta_norm_pp']), 'abs_mean_delta_norm_pp': _safe_float(row['abs_mean_delta_norm_pp']), 'mean_abs_delta_pp': _safe_float(row['mean_abs_delta_pp']), 'direction_sign': row['direction_sign'], 'tail_direction20': row['tail_direction20'], 'balanced_support_20': _safe_float(row['balanced_support_20']), 'tail_mixed20': bool(row['tail_mixed20']), 'frac_either_zero': _safe_float(row['frac_either_zero']), 'n_check_false_pairs': _safe_float(row['n_check_false_pairs'])})
    if missing_ids:
        raise ValueError(f'missing substrate sets for {len(missing_ids)} swap IDs')
    context = pd.DataFrame(context_records)
    pair_rows: list[dict[str, object]] = []
    for key, g in context.groupby(EDGE_KEYS, sort=True):
        if g['background_fingerprint'].duplicated().any():
            raise AssertionError(f'duplicate background context in {key}')
        recs = list(g.to_dict('records'))
        for a, b in combinations(recs, 2):
            sa, sb = (a['substrate_set'], b['substrate_set'])
            shared = len(sa & sb)
            union = len(sa | sb)
            jac = float(shared / union) if union else np.nan
            min_size = min(len(sa), len(sb))
            min_overlap = float(shared / min_size) if min_size else np.nan
            mean_overlap = float(2 * shared / (len(sa) + len(sb))) if len(sa) + len(sb) else np.nan
            pair_rows.append({**{k: a[k] for k in EDGE_KEYS}, 'background_a': a['background_fingerprint'], 'background_b': b['background_fingerprint'], 'swap_ids_a': a['swap_ids'], 'swap_ids_b': b['swap_ids'], 'n_pairs_a': a['n_pairs'], 'n_pairs_b': b['n_pairs'], 'n_substrate_pairs_a': len(sa), 'n_substrate_pairs_b': len(sb), 'n_swap_rows_collapsed_a': a['n_swap_rows_collapsed'], 'n_swap_rows_collapsed_b': b['n_swap_rows_collapsed'], 'panel_repeat_factor_a': float(a['n_pairs'] / len(sa)) if len(sa) else np.nan, 'panel_repeat_factor_b': float(b['n_pairs'] / len(sb)) if len(sb) else np.nan, 'panel_duplicate_context_a': bool(a['n_swap_rows_collapsed'] > 1), 'panel_duplicate_context_b': bool(b['n_swap_rows_collapsed'] > 1), 'n_shared_substrate_pairs': shared, 'n_union_substrate_pairs': union, 'substrate_jaccard': jac, 'overlap_min_fraction': min_overlap, 'overlap_mean_fraction': mean_overlap, 'mean_delta_norm_pp_a': a['mean_delta_norm_pp'], 'mean_delta_norm_pp_b': b['mean_delta_norm_pp'], 'abs_mean_delta_norm_pp_a': a['abs_mean_delta_norm_pp'], 'abs_mean_delta_norm_pp_b': b['abs_mean_delta_norm_pp'], 'abs_mean_delta_difference_pp': abs(a['abs_mean_delta_norm_pp'] - b['abs_mean_delta_norm_pp']), 'signed_mean_delta_difference_pp': a['mean_delta_norm_pp'] - b['mean_delta_norm_pp'], 'mean_abs_delta_pp_a': a['mean_abs_delta_pp'], 'mean_abs_delta_pp_b': b['mean_abs_delta_pp'], 'direction_a': a['direction_sign'], 'direction_b': b['direction_sign'], 'direction_agreement': _direction_agreement(a['direction_sign'], b['direction_sign']), 'tail_direction_a': a['tail_direction20'], 'tail_direction_b': b['tail_direction20'], 'tail_agreement': _direction_agreement(a['tail_direction20'], b['tail_direction20']), 'balanced_support20_a': a['balanced_support_20'], 'balanced_support20_b': b['balanced_support_20'], 'balanced_support20_mean': np.nanmean([a['balanced_support_20'], b['balanced_support_20']]), 'tail_mixed20_a': a['tail_mixed20'], 'tail_mixed20_b': b['tail_mixed20'], 'frac_either_zero_a': a['frac_either_zero'], 'frac_either_zero_b': b['frac_either_zero'], 'n_check_false_pairs_a': a['n_check_false_pairs'], 'n_check_false_pairs_b': b['n_check_false_pairs'], 'n_unique_backgrounds_edge': len(recs)})
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise ValueError('no context pairs generated')
    pairs['jaccard_stratum'] = pd.cut(pairs['substrate_jaccard'], JAC_EDGES, labels=JAC_LABELS, include_lowest=True, right=False).astype(str)
    pairs['overlap_min_stratum'] = pd.cut(pairs['overlap_min_fraction'], OVERLAP_EDGES, labels=OVERLAP_LABELS, include_lowest=True, right=False).astype(str)
    carry_cols = [*EDGE_KEYS, 'direction_sign_consistency', 'direction_class', 'median_abs_mean_delta_pp', 'magnitude_drift_range_pp', 'state_label', 'median_balanced_support20']
    pairs = pairs.merge(edge[carry_cols], on=EDGE_KEYS, how='left', validate='many_to_one')
    association_rows = []
    for overlap_col, overlap_label in [('substrate_jaccard', 'jaccard'), ('overlap_min_fraction', 'overlap_min')]:
        for outcome_col, outcome_label in [('direction_agreement', 'direction'), ('tail_agreement', 'tail')]:
            q = pairs[[overlap_col, outcome_col]].dropna()
            association_rows.append({'level': 'context_pair', 'overlap_metric': overlap_label, 'outcome_metric': outcome_label, 'n': int(len(q)), 'spearman': float(q.corr(method='spearman').iloc[0, 1]) if len(q) >= 3 else np.nan})
            edge_mean = pairs.dropna(subset=[overlap_col, outcome_col]).groupby(EDGE_KEYS, sort=False).agg(overlap=(overlap_col, 'mean'), outcome=(outcome_col, 'mean'))
            association_rows.append({'level': 'edge_balanced', 'overlap_metric': overlap_label, 'outcome_metric': outcome_label, 'n': int(len(edge_mean)), 'spearman': float(edge_mean.corr(method='spearman').iloc[0, 1]) if len(edge_mean) >= 3 else np.nan})
    association = pd.DataFrame(association_rows)
    summary_rows: list[dict[str, object]] = []
    for stratum, col in [('jaccard', 'jaccard_stratum'), ('overlap_min', 'overlap_min_stratum')]:
        for label in JAC_LABELS if stratum == 'jaccard' else OVERLAP_LABELS:
            g = pairs.loc[pairs[col].eq(label)]
            if g.empty:
                continue
            summary_rows.append(_summarize_bin(g, 'overall', stratum, label))
            for (reaction_type, slot), gg in g.groupby(EDGE_KEYS[:2], sort=True):
                rec = _summarize_bin(gg, 'reaction_slot', stratum, label)
                rec['reaction_type'], rec['swap_slot'] = (reaction_type, slot)
                summary_rows.append(rec)
            rec = _summarize_bin(g, 'edge_balanced', stratum, label)
            rec['reaction_type'], rec['swap_slot'] = ('ALL', 'ALL')
            summary_rows.append(rec)
            for (reaction_type, slot), gg in g.groupby(EDGE_KEYS[:2], sort=True):
                rec = _summarize_bin(gg, 'edge_balanced_reaction_slot', stratum, label)
                rec['reaction_type'], rec['swap_slot'] = (reaction_type, slot)
                summary_rows.append(rec)
    contrast_rows = []
    for key, g in pairs.groupby(EDGE_KEYS, sort=True):
        low = g.loc[g['substrate_jaccard'] < 0.75, 'direction_agreement'].dropna()
        high = g.loc[g['substrate_jaccard'] >= 0.75, 'direction_agreement'].dropna()
        if len(low) and len(high):
            contrast_rows.append({'row_level': 'within_edge_jaccard_contrast', **{k: key[i] for i, k in enumerate(EDGE_KEYS)}, 'n_low_jaccard_pairs': int(len(low)), 'n_high_jaccard_pairs': int(len(high)), 'low_jaccard_agreement': float(low.mean()), 'high_jaccard_agreement': float(high.mean()), 'high_minus_low_agreement': float(high.mean() - low.mean())})
    contrast = pd.DataFrame(contrast_rows)
    pairs.to_csv(OUT_PAIRS, index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    contrast_path = BASE / 'results' / 'background_portability_overlap_within_edge.csv'
    contrast.to_csv(contrast_path, index=False, encoding='utf-8-sig')
    port = pd.read_csv(PORT)
    dm_summary = pd.read_csv(SUMMARY)
    audit = {'input_meta': str(META), 'input_background_rows': int(len(bg)), 'input_repeated_edges': int(len(repeated_edges)), 'strict_swap_ids': int(len(strict_ids)), 'strict_swap_ids_with_panel_sets': int(len(sets)), 'context_pair_rows': int(len(pairs)), 'context_pair_edges': int(pairs.groupby(EDGE_KEYS).ngroups), 'within_edge_contrast_edges': int(len(contrast)), 'jaccard_bins': JAC_LABELS, 'overlap_min_bins': OVERLAP_LABELS, 'background_rows_reconciled_to_direction_magnitude': int(len(bg)), 'context_portability_rows_available': int(len(port)), 'direction_magnitude_summary_rows_available': int(len(dm_summary)), 'missing_swap_ids': sorted(set(missing_ids)), 'association_checks': association.to_dict('records'), 'collapsed_contexts_with_multiple_swap_rows': int((context['n_swap_rows_collapsed'] > 1).sum()), 'max_context_panel_repeat_factor': float(np.nanmax(context['n_pairs'] / context['n_substrate_pairs']))}
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

    def f(x: object, n: int=3) -> str:
        try:
            if not np.isfinite(float(x)):
                return 'NA'
            return f'{float(x):.{n}f}'
        except (TypeError, ValueError):
            return 'NA'
    overall = summary.loc[summary['row_level'].eq('overall')].copy()
    jac_overall = overall.loc[overall['stratum_type'].eq('jaccard')]
    overlap_overall = overall.loc[overall['stratum_type'].eq('overlap_min')]
    assoc = association.copy()
    lines = ['# Background portability versus substrate-panel overlap', '', '本报告把跨背景一致性拆成两个来源：条件交换是否在不同背景中保持方向，以及这些背景是否其实重复使用了相近的底物面板。分析只使用已生成的严格 identity-only 网络、direction/magnitude 背景表和对应的 `swap_deltas` 底物键；没有新增实验，也没有改变原始结果。', '', '## 研究单位与防止伪重复', '', f'保留 {len(repeated_edges)} 条至少有两个唯一 `background_fingerprint` 的组件边，展开为 {len(pairs):,} 个同一组件边内的背景两两比较。每个背景先按 `direction_magnitude_background.csv` 合并相同指纹；每个背景的底物面板取其所有 swap ID 的底物对并集。底物对在多个背景重复出现时，`n_shared_substrate_pairs` 只用于计算面板重叠，不被计作独立复制。', f"共有 {int((context['n_swap_rows_collapsed'] > 1).sum())} 个背景指纹由多个 swap 行合并，最大面板重复因子为 {f(np.nanmax(context['n_pairs'] / context['n_substrate_pairs']))}；这些重复只保留为一个背景上下文，并在 CSV 中通过 `n_swap_rows_collapsed` 和 `panel_duplicate_context_*` 标出。", '', '主要指标：', '- `substrate_jaccard = |A∩B|/|A∪B|`；同时报告 `overlap_min_fraction = |A∩B|/min(|A|,|B|)`，以避免仅由一个大面板稀释 Jaccard。', '- `direction_agreement` 比较两个背景的规范化背景平均 Δ 符号；平局不计入分母。`tail_agreement` 比较 +20/-20 pp 尾部多数方向。', '- 结果同时给出 context-pair 加权和 edge-balanced 加权；后者先在每条组件边内平均，再让每条边权重相同。', '- Jaccard 和最小重叠比例均使用固定分箱，而非按结果调节的分位数；空箱不显示。', '', '## 全库分层结果（每个背景对一行）', '', '### Shared substrate Jaccard', '', '| Jaccard stratum | context pairs | edges | pairwise direction agreement | edge-balanced direction agreement | pairwise tail agreement | median abs-Δ difference (pp) | median shared pairs |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in jac_overall.iterrows():
        lines.append(f"| {x['stratum']} | {int(x['n_context_pairs'])} | {int(x['n_edges'])} | {f(x['pairwise_direction_agreement'])} | {f(x['edge_balanced_direction_agreement'])} | {f(x['pairwise_tail_agreement'])} | {f(x['pairwise_median_abs_delta_difference_pp'], 2)} | {f(x['median_shared_substrate_pairs'], 1)} |")
    lines += ['', '### Minimum panel-overlap fraction', '', '| overlap-min stratum | context pairs | edges | pairwise direction agreement | edge-balanced direction agreement | pairwise tail agreement | median abs-Δ difference (pp) | median Jaccard |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in overlap_overall.iterrows():
        lines.append(f"| {x['stratum']} | {int(x['n_context_pairs'])} | {int(x['n_edges'])} | {f(x['pairwise_direction_agreement'])} | {f(x['edge_balanced_direction_agreement'])} | {f(x['pairwise_tail_agreement'])} | {f(x['pairwise_median_abs_delta_difference_pp'], 2)} | {f(x['median_jaccard'])} |")
    lines += ['', '## Within-edge portability test', '', f'共有 {len(contrast)} 条组件边同时包含 Jaccard < 0.75 和 Jaccard ≥ 0.75 的背景对。`background_portability_overlap_within_edge.csv` 给出每条边的高重叠减低重叠方向一致率差值；正值表示高重叠背景对更一致，负值表示低重叠背景对反而更一致。该比较仍是历史描述，且每条边内的背景对共享上下文，不能作独立样本检验。', '', '| quantity | value |', '|---|---:|', f'| within-edge contrasts | {len(contrast)} |', f"| median (high-low agreement) | {f(contrast['high_minus_low_agreement'].median() if len(contrast) else np.nan)} |", f"| mean (high-low agreement) | {f(contrast['high_minus_low_agreement'].mean() if len(contrast) else np.nan)} |", f"| fraction high > low | {f((contrast['high_minus_low_agreement'] > 0).mean() if len(contrast) else np.nan)} |", f"| fraction high < low | {f((contrast['high_minus_low_agreement'] < 0).mean() if len(contrast) else np.nan)} |", '', '## Overlap-association audit', '', 'Spearman 数值仅用于检查面板重叠和一致性的单调关系，不是独立样本显著性检验。`edge_balanced` 先在组件边内平均，再在边之间计算相关。', '', '| level | overlap metric | outcome | n | Spearman |', '|---|---|---|---:|---:|']
    for _, x in assoc.iterrows():
        lines.append(f"| {x['level']} | {x['overlap_metric']} | {x['outcome_metric']} | {int(x['n'])} | {f(x['spearman'])} |")
    lines += ['', '## Reaction class and changed-slot detail', '', '完整的 reaction_type × changed-slot 分层同时保存在 `background_portability_overlap_summary.csv`。这些行用于定位哪些网络具有低重叠仍保持方向的证据；稀疏层级不应被排序成稳定效应。', '', '## Interpretation', '', '高 Jaccard 下的方向一致性只能说明同一组件边在相近底物面板上的条件排序较稳定；它可能来自底物重复、板/批次共线或同一历史筛选网络。低 Jaccard 下若仍保持方向一致，才更接近背景可迁移性的描述性证据，但这里的低重叠仍可能共享底物家族、产品分析流程或其它未记录因素。反过来，高重叠下的方向不一致直接说明“同一面板重复”也不能保证固定排序。', '', '本分析没有把 Jaccard 分箱结果当作独立显著性检验，没有把 context-pair 数当作实验重复数，也没有将其提升为隐藏机制 regime。它的用途是为下一轮验证优先级提供两个维度：背景改变后仍保持方向，以及该判断是否超出了底物面板重用。', '', '## 复现', '', '```powershell', 'python code/background_portability_overlap.py', '```', '', '输出：`results/background_portability_overlap.csv`、`results/background_portability_overlap_summary.csv`、`results/background_portability_overlap_within_edge.csv`。']
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {OUT_PAIRS} rows={len(pairs)}; summary={len(summary)}; within_edge={len(contrast)}')
if __name__ == '__main__':
    main()
