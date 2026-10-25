"""Posthoc condition-network meta-analysis for strict identity-only swaps.

This script intentionally keeps the full strict registry.  It produces one
CSV with three row levels (swap, component_pair, reaction_slot), plus a
focused markdown report.  No source files or prior result tables are changed."""
from __future__ import annotations
import json
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
REGISTRY = BASE / 'results' / 'swap_registry.csv'
DELTAS = BASE / 'results' / 'swap_deltas.parquet'
NEXT = BASE / 'results' / 'next_candidate_screen.csv'
SELECTED = BASE / 'results' / 'selected_candidates.csv'
OUT_CSV = BASE / 'results' / 'condition_network_meta.csv'
OUT_REPORT = BASE / 'reports' / 'CONDITION_NETWORK_META.md'

def safe_json(value: object) -> dict:
    try:
        x = json.loads(value) if isinstance(value, str) else value
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}

def fmt_num(x: object) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return 'NA'
    if isinstance(x, (float, np.floating)):
        return f'{float(x):.4f}'
    return str(x)

def dose(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'NA'
    return f'{float(value):g}'

def background_fingerprint(row: pd.Series) -> str:
    """Canonical recipe fingerprint with the changed slot removed."""
    slot = str(row['swap_slot'])
    r0 = safe_json(row['recipe0'])
    r1 = safe_json(row['recipe1'])
    parts = []
    for i in range(1, 7):
        key = f'reagent{i}'
        if slot == key:
            continue
        cid = r0.get(f'{key}_id')
        equiv = r0.get(f'{key}_equiv')
        if cid is not None or equiv is not None:
            parts.append(f'{key}={cid}@{dose(equiv)}')
        else:
            parts.append(f'{key}=NA')
    parts.extend([f"solvent={r0.get('solvent_id')}@{r0.get('solvent_name')}", f"T={r0.get('temp_id')}", f"t={r0.get('time_id')}"])
    return ';'.join(parts)

def component_label(row: pd.Series, cid: str) -> str:
    if str(row['component0_id']) == cid:
        return str(row['component0_name'])
    return str(row['component1_name'])

def graph_components(edges: list[tuple[str, str]]) -> tuple[dict[str, int], dict[str, int], int]:
    """Return node degree, component size, and number of connected components."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        if pd.isna(a) or pd.isna(b):
            continue
        adj[str(a)].add(str(b))
        adj[str(b)].add(str(a))
    degree = {node: len(neigh) for node, neigh in adj.items()}
    comp_size: dict[str, int] = {}
    ncomp = 0
    seen: set[str] = set()
    for start in adj:
        if start in seen:
            continue
        ncomp += 1
        q = deque([start])
        seen.add(start)
        nodes = []
        while q:
            cur = q.popleft()
            nodes.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        for node in nodes:
            comp_size[node] = len(nodes)
    return (degree, comp_size, ncomp)

def jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    return float(len(a & b) / u) if u else np.nan

def main() -> None:
    reg = pd.read_csv(REGISTRY)
    reg = reg[reg['match_class'].eq('strict_identity_only')].copy()
    if reg['swap_id'].duplicated().any():
        raise ValueError('duplicate strict swap_id')
    reg['component_pair'] = reg.apply(lambda x: '|'.join(sorted([str(x['component0_id']), str(x['component1_id'])])), axis=1)
    reg['background_fingerprint'] = reg.apply(background_fingerprint, axis=1)
    selected = pd.read_csv(SELECTED) if SELECTED.is_file() else pd.DataFrame(columns=['swap_id', 'match_class'])
    selected_any = set(selected['swap_id'].astype(str))
    selected_strict = set(selected.loc[selected['match_class'].eq('strict_identity_only'), 'swap_id'].astype(str))
    nxt = pd.read_csv(NEXT) if NEXT.is_file() else pd.DataFrame(columns=['swap_id', 'screen_rank', 'context_support_tier', 'target_key', 'target_context_n', 'context_peer_comparisons', 'context_median_common_pairs', 'context_median_spearman', 'context_median_sign_agreement20', 'context_median_sign_agreement15', 'context_n_decisive20', 'context_n_agreement20_ge70', 'context_n_with_existing_selected', 'context_median_abs_difference_pp'])
    next_rank = dict(zip(nxt['swap_id'].astype(str), nxt['screen_rank']))
    next_tier = dict(zip(nxt['swap_id'].astype(str), nxt['context_support_tier']))
    next_cols = ['target_key', 'target_context_n', 'context_peer_comparisons', 'context_median_common_pairs', 'context_median_spearman', 'context_median_sign_agreement20', 'context_median_sign_agreement15', 'context_n_decisive20', 'context_n_agreement20_ge70', 'context_n_with_existing_selected', 'context_median_abs_difference_pp']
    next_lookup = nxt.assign(swap_id=nxt['swap_id'].astype(str)).set_index('swap_id')[next_cols].to_dict('index')
    delta = pd.read_parquet(DELTAS)
    delta = delta[delta['swap_id'].astype(str).isin(set(reg['swap_id'].astype(str)))].copy()
    delta['swap_id'] = delta['swap_id'].astype(str)
    substrate_sets = delta.groupby('swap_id')['substrate_pair_key'].agg(lambda s: set(s.astype(str))).to_dict()
    graph_stats: dict[tuple[str, str], dict[str, object]] = {}
    for key, g in reg.groupby(['reaction_type', 'swap_slot'], sort=True):
        condition_edges = [(str(a), str(b)) for a, b in zip(g.condition0, g.condition1)]
        component_edges = [(str(a), str(b)) for a, b in zip(g.component0_id, g.component1_id)]
        cdeg, csize, cncomp = graph_components(condition_edges)
        xdeg, xsize, xncomp = graph_components(component_edges)
        graph_stats[key] = {'condition_degree': cdeg, 'condition_component_size': csize, 'condition_n_components': cncomp, 'condition_n_nodes': len(cdeg), 'component_degree': xdeg, 'component_component_size': xsize, 'component_n_components': xncomp, 'component_n_nodes': len(xdeg)}
    pair_groups = list(reg.groupby(['reaction_type', 'swap_slot', 'component_pair'], sort=True))
    pair_stats: dict[tuple[str, str, str], dict[str, object]] = {}
    for key, g in pair_groups:
        ids = g['swap_id'].astype(str).tolist()
        sets = [substrate_sets.get(i, set()) for i in ids]
        js = []
        intersections = []
        for a, b in combinations(sets, 2):
            js.append(jaccard(a, b))
            intersections.append(len(a & b))
        union = set().union(*sets) if sets else set()
        multiplicity = Counter((x for s in sets for x in s))
        shared_multi = sum((v >= 2 for v in multiplicity.values()))
        low, high = str(key[2]).split('|', 1)
        orient = np.where(g['component0_id'].astype(str).eq(low), 1.0, -1.0)
        mean_norm = g['mean_delta_pp'].to_numpy(float) * orient
        npos = g['n_pos_20'].to_numpy(int)
        nneg = g['n_neg_20'].to_numpy(int)
        pos_norm = np.where(orient > 0, npos, nneg)
        neg_norm = np.where(orient > 0, nneg, npos)
        majority = np.where(pos_norm > neg_norm, 'positive', np.where(neg_norm > pos_norm, 'negative', 'tie'))
        n_pairs = g['n_pairs'].to_numpy(float)
        total_pairs = float(np.nansum(n_pairs))
        pair_stats[key] = {'pair_n_swaps': len(g), 'pair_n_unique_backgrounds': g['background_fingerprint'].nunique(), 'pair_n_substrate_union': len(union), 'pair_n_substrate_in_multiple_contexts': shared_multi, 'pair_mean_context_jaccard': float(np.nanmean(js)) if js else np.nan, 'pair_median_context_jaccard': float(np.nanmedian(js)) if js else np.nan, 'pair_max_context_jaccard': float(np.nanmax(js)) if js else np.nan, 'pair_mean_context_intersection': float(np.nanmean(intersections)) if intersections else np.nan, 'pair_pooled_mean_delta_norm_pp': float(np.nansum(mean_norm * n_pairs) / total_pairs) if total_pairs else np.nan, 'pair_pooled_pos_tail_fraction20': float(np.nansum(pos_norm) / total_pairs) if total_pairs else np.nan, 'pair_pooled_neg_tail_fraction20': float(np.nansum(neg_norm) / total_pairs) if total_pairs else np.nan, 'pair_mean_balanced_support20': float(g['balanced_support_20'].mean()), 'pair_median_balanced_support20': float(g['balanced_support_20'].median()), 'pair_n_context_positive_majority': int(np.sum(majority == 'positive')), 'pair_n_context_negative_majority': int(np.sum(majority == 'negative')), 'pair_n_context_tie_majority': int(np.sum(majority == 'tie')), 'pair_n_context_tail_mixed20': int(np.sum((pos_norm > 0) & (neg_norm > 0))), 'pair_context_majority_consistency': float(max(np.sum(majority == 'positive'), np.sum(majority == 'negative')) / len(g)), 'pair_context_tail_mixing_rate20': float(np.mean((pos_norm > 0) & (neg_norm > 0))), 'pair_orientation_reversed_n': int(np.sum(orient < 0)), 'pair_component_low': low, 'pair_component_high': high}
        for sid, s in zip(ids, sets):
            peer_js = [jaccard(s, t) for other, t in zip(ids, sets) if other != sid]
            pair_stats[key].setdefault('swap_peer_jaccard', {})[sid] = float(np.nanmean(peer_js)) if peer_js else np.nan
    rows: list[dict[str, object]] = []
    for _, row in reg.iterrows():
        sid = str(row['swap_id'])
        key = (row['reaction_type'], row['swap_slot'], row['component_pair'])
        gs = graph_stats[row['reaction_type'], row['swap_slot']]
        ps = pair_stats[key]
        low = ps['pair_component_low']
        orient = 1 if str(row['component0_id']) == low else -1
        pos_norm = int(row['n_pos_20'] if orient > 0 else row['n_neg_20'])
        neg_norm = int(row['n_neg_20'] if orient > 0 else row['n_pos_20'])
        direction = 'positive' if pos_norm > neg_norm else 'negative' if neg_norm > pos_norm else 'tie'
        rec = {'row_level': 'swap', 'swap_id': sid, 'selected_any': sid in selected_any, 'selected_strict': sid in selected_strict, 'in_next_candidate_screen': sid in next_rank, 'next_screen_rank': next_rank.get(sid, np.nan), 'next_context_support_tier': next_tier.get(sid, ''), 'reaction_type': row['reaction_type'], 'swap_slot': row['swap_slot'], 'condition0': row['condition0'], 'condition1': row['condition1'], 'component0_id': row['component0_id'], 'component1_id': row['component1_id'], 'component0_name': row['component0_name'], 'component1_name': row['component1_name'], 'component_pair': row['component_pair'], 'component_pair_orientation': f"{low}->{ps['pair_component_high']}", 'orientation_sign': orient, 'background_fingerprint': row['background_fingerprint'], 'n_pairs': row['n_pairs'], 'mean_delta_pp': row['mean_delta_pp'], 'mean_delta_norm_pp': float(row['mean_delta_pp']) * orient, 'n_pos_20': row['n_pos_20'], 'n_neg_20': row['n_neg_20'], 'n_pos_20_norm': pos_norm, 'n_neg_20_norm': neg_norm, 'direction_majority20': direction, 'tail_mixed20': bool(pos_norm > 0 and neg_norm > 0), 'balanced_support_20': row['balanced_support_20'], 'unique_cells_n_pairs': row['unique_cells_n_pairs'], 'frac_either_zero': row['frac_either_zero'], 'n_check_false_pairs': row['n_check_false_pairs'], 'pair_n_swaps': ps['pair_n_swaps'], 'pair_n_unique_backgrounds': ps['pair_n_unique_backgrounds'], 'pair_n_substrate_union': ps['pair_n_substrate_union'], 'pair_n_substrate_in_multiple_contexts': ps['pair_n_substrate_in_multiple_contexts'], 'pair_mean_context_jaccard': ps['pair_mean_context_jaccard'], 'pair_median_context_jaccard': ps['pair_median_context_jaccard'], 'pair_max_context_jaccard': ps['pair_max_context_jaccard'], 'swap_mean_jaccard_to_peer': ps['swap_peer_jaccard'][sid], 'pair_pooled_mean_delta_norm_pp': ps['pair_pooled_mean_delta_norm_pp'], 'pair_pooled_pos_tail_fraction20': ps['pair_pooled_pos_tail_fraction20'], 'pair_pooled_neg_tail_fraction20': ps['pair_pooled_neg_tail_fraction20'], 'pair_mean_balanced_support20': ps['pair_mean_balanced_support20'], 'pair_median_balanced_support20': ps['pair_median_balanced_support20'], 'pair_n_context_positive_majority': ps['pair_n_context_positive_majority'], 'pair_n_context_negative_majority': ps['pair_n_context_negative_majority'], 'pair_n_context_tail_mixed20': ps['pair_n_context_tail_mixed20'], 'pair_context_majority_consistency': ps['pair_context_majority_consistency'], 'pair_context_tail_mixing_rate20': ps['pair_context_tail_mixing_rate20'], 'condition_graph_node_degree0': gs['condition_degree'].get(str(row['condition0']), 0), 'condition_graph_node_degree1': gs['condition_degree'].get(str(row['condition1']), 0), 'condition_graph_component_size0': gs['condition_component_size'].get(str(row['condition0']), 0), 'condition_graph_component_size1': gs['condition_component_size'].get(str(row['condition1']), 0), 'condition_graph_n_components': gs['condition_n_components'], 'condition_graph_n_nodes': gs['condition_n_nodes'], 'component_graph_degree0': gs['component_degree'].get(str(row['component0_id']), 0), 'component_graph_degree1': gs['component_degree'].get(str(row['component1_id']), 0), 'component_graph_component_size0': gs['component_component_size'].get(str(row['component0_id']), 0), 'component_graph_component_size1': gs['component_component_size'].get(str(row['component1_id']), 0), 'component_graph_n_components': gs['component_n_components'], 'component_graph_n_nodes': gs['component_n_nodes']}
        for c in next_cols:
            rec[f'next_{c}'] = next_lookup.get(sid, {}).get(c, np.nan)
        rows.append(rec)
    for key, g in pair_groups:
        ps = pair_stats[key]
        rec = {'row_level': 'component_pair', 'reaction_type': key[0], 'swap_slot': key[1], 'component_pair': key[2]}
        rec.update(ps)
        rec['selected_strict_n'] = int(g['swap_id'].isin(selected_strict).sum())
        rec['selected_any_n'] = int(g['swap_id'].isin(selected_any).sum())
        rec['next_candidate_n'] = int(g['swap_id'].isin(next_rank).sum())
        rows.append(rec)
    for key, g in reg.groupby(['reaction_type', 'swap_slot'], sort=True):
        gs = graph_stats[key]
        orient = np.where(g['component0_id'].astype(str).eq(g.apply(lambda x: pair_stats[x['reaction_type'], x['swap_slot'], x['component_pair']]['pair_component_low'], axis=1)), 1.0, -1.0)
        n_pairs = g['n_pairs'].to_numpy(float)
        total = float(np.nansum(n_pairs))
        rec = {'row_level': 'reaction_slot', 'reaction_type': key[0], 'swap_slot': key[1], 'n_swaps': len(g), 'n_unique_component_pairs': g['component_pair'].nunique(), 'n_unique_backgrounds': g['background_fingerprint'].nunique(), 'n_conditions': len(set(g.condition0) | set(g.condition1)), 'n_repeated_component_pairs': int((g.groupby('component_pair').size() >= 2).sum()), 'selected_strict_n': int(g['swap_id'].isin(selected_strict).sum()), 'selected_any_n': int(g['swap_id'].isin(selected_any).sum()), 'next_candidate_n': int(g['swap_id'].isin(next_rank).sum()), 'pooled_mean_delta_norm_pp': float(np.nansum(g['mean_delta_pp'].to_numpy(float) * orient * n_pairs) / total) if total else np.nan, 'pooled_pos_tail_fraction20': float(np.nansum(np.where(orient > 0, g['n_pos_20'], g['n_neg_20'])) / total) if total else np.nan, 'pooled_neg_tail_fraction20': float(np.nansum(np.where(orient > 0, g['n_neg_20'], g['n_pos_20'])) / total) if total else np.nan, 'mean_balanced_support20': float(g['balanced_support_20'].mean()), 'median_balanced_support20': float(g['balanced_support_20'].median()), 'n_tail_mixed_contexts20': int(np.sum((np.where(orient > 0, g['n_pos_20'], g['n_neg_20']) > 0) & (np.where(orient > 0, g['n_neg_20'], g['n_pos_20']) > 0))), 'condition_graph_n_nodes': gs['condition_n_nodes'], 'condition_graph_n_components': gs['condition_n_components'], 'component_graph_n_nodes': gs['component_n_nodes'], 'component_graph_n_components': gs['component_n_components'], 'condition_graph_largest_component': max(gs['condition_component_size'].values(), default=0), 'component_graph_largest_component': max(gs['component_component_size'].values(), default=0)}
        rows.append(rec)
    out = pd.DataFrame(rows)
    first = ['row_level', 'swap_id', 'selected_any', 'selected_strict', 'in_next_candidate_screen', 'next_screen_rank', 'next_context_support_tier', 'reaction_type', 'swap_slot', 'condition0', 'condition1', 'component0_id', 'component1_id', 'component0_name', 'component1_name', 'component_pair', 'component_pair_orientation', 'orientation_sign', 'background_fingerprint']
    out = out[first + [c for c in out.columns if c not in first]]
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    pair_out = out[out.row_level.eq('component_pair')].copy()
    slot_out = out[out.row_level.eq('reaction_slot')].copy()
    repeated = pair_out[pair_out.pair_n_swaps.ge(2)].sort_values(['pair_n_swaps', 'pair_context_tail_mixing_rate20'], ascending=False)
    repeated_background = pair_out[pair_out.pair_n_unique_backgrounds.ge(2)]
    mixed = pair_out[pair_out.pair_n_context_tail_mixed20.ge(2)].sort_values('pair_context_tail_mixing_rate20', ascending=False)
    slot_out = slot_out.sort_values(['n_swaps', 'reaction_type', 'swap_slot'], ascending=[False, True, True])
    lines = ['# Condition-network meta-analysis (strict identity-only)', '', '本报告是对现有历史 HTE 数据的事后网络整理。分析只保留 `match_class == strict_identity_only` 的交换，保留 2,132 个候选；未进行新实验，也不把共享底物对当作独立重复。结果文件中的 `row_level` 分为 `swap`（全部候选）、`component_pair`（无向组件交换边）和 `reaction_slot`（reaction type × changed slot）。', '', '## 口径', '', '- 组件边按 `component0_id|component1_id` 的无向规范形式合并；方向性指标将每条边统一到低 ID → 高 ID，避免把反向记录误判为不同交换。', '- `background_fingerprint` 移除 changed slot，保留其余 reagent ID/当量、solvent、temperature、time。不同背景的相同组件边才构成重复背景；相同底物网络的重叠只作为支持范围描述。', '- `tail_mixed20` 表示同一历史候选中同时出现至少一个 +20 pp 和 -20 pp 配对；它描述异质性，不证明机理上的 regime。`balanced_support_20 = 2 × min(n_pos_20,n_neg_20)/n_pairs`，是较小尾部比例的两倍；评分 0.20 对应每侧至少 10%，不是每侧 20%。报告中的均值/中位数用于区分偶然出现两端和较均衡的双向尾部。`direction_majority20` 只表示两类尾部计数的多数方向。', '- `selected_any` 包括既有 selected_candidates 表中所有记录，`selected_strict` 只包括其中严格 identity-only 记录；`in_next_candidate_screen` 标记第二轮候选筛选的 151 个条目。', '', '## 网络总览', '', f"严格交换共 {len(reg):,} 个，覆盖 {reg.reaction_type.nunique()} 个 reaction type 和 {reg.groupby(['reaction_type', 'swap_slot']).ngroups} 个 reaction-slot 层级。全局去重的无向化学身份对为 {reg.component_pair.nunique():,} 个；加入 reaction type 与 changed slot 后，实际网络分析使用 {len(pair_out):,} 条组件边。这两个分母不能混用。按重复交换行计数，有至少两个历史行的边为 {int((pair_out.pair_n_swaps >= 2).sum()):,} 个；真正跨至少两个唯一 `background_fingerprint` 的边为 {int(len(repeated_background)):,} 个。按重复交换行统计，至少两个上下文同时出现正负 20 pp 尾部的边为 {int((pair_out.pair_n_context_tail_mixed20 >= 2).sum()):,} 个；在唯一背景口径下为 {int((repeated_background.pair_n_context_tail_mixed20 >= 2).sum()):,} 个。", '', '### Reaction type × changed slot（按严格候选数）', '', '| reaction type | slot | swaps | component edges | backgrounds | conditions | repeated edges | selected strict | next-screen | pooled mean normalized delta (pp) | +20 tail | -20 tail |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in slot_out.head(30).iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {int(x.n_swaps)} | {int(x.n_unique_component_pairs)} | {int(x.n_unique_backgrounds)} | {int(x.n_conditions)} | {int(x.n_repeated_component_pairs)} | {int(x.selected_strict_n)} | {int(x.next_candidate_n)} | {fmt_num(x.pooled_mean_delta_norm_pp)} | {fmt_num(x.pooled_pos_tail_fraction20)} | {fmt_num(x.pooled_neg_tail_fraction20)} |')
    lines += ['', '表格按候选数截取前 30 个层级；完整层级保存在 CSV 的 `row_level == reaction_slot` 行。normalized delta 仅用于统一无向边方向，不是新的因果估计。', '', '## 重复组件边与方向异质性', '', '下面列出有多个历史交换行的组件边。`pair_n_swaps` 统计交换行，`pair_n_unique_backgrounds` 统计去掉 changed slot 后的唯一背景；同一背景下的重复行不等于跨背景复制。`pair_context_majority_consistency` 接近 1 表示多数方向在上下文间一致；`pair_context_tail_mixing_rate20` 高表示每个上下文内部出现了双向尾部，而 `pair_mean_balanced_support20` 高才表示两端相对更均衡。三者应同时阅读，不能把单一大均值解释为普遍条件优势。', '', '| reaction | slot | component pair | contexts | unique backgrounds | majority consistency | tail mixing | mean balanced support | pooled normalized mean (pp) | mean Jaccard | selected strict |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in repeated.head(40).iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.pair_n_swaps)} | {int(x.pair_n_unique_backgrounds)} | {fmt_num(x.pair_context_majority_consistency)} | {fmt_num(x.pair_context_tail_mixing_rate20)} | {fmt_num(x.pair_mean_balanced_support20)} | {fmt_num(x.pair_pooled_mean_delta_norm_pp)} | {fmt_num(x.pair_mean_context_jaccard)} | {int(x.selected_strict_n)} |')
    lines += ['', '## 共享底物网络', '', '同一组件边的 `pair_n_substrate_union` 是所有上下文覆盖的底物对并集；`pair_n_substrate_in_multiple_contexts` 是出现在至少两个上下文中的底物对数；Jaccard 衡量历史面板重叠。高重叠意味着支持来自相近底物网络，低重叠意味着网络扩展，但两者都不产生独立实验重复。', '', '在当前数据中，严格组件边大多只有一个条件上下文，因此跨背景一致性主要集中在少数网络边；这正是后续候选冻结时需要优先区分的结构。全部边及其共享网络统计见 CSV。', '', '## 解释边界', '', '该分析沿用了候选发现后的历史结果，存在选择偏差；条件 ID、底物家族、板/批次和背景可能共线。严格 identity-only 只保证名义 ID 交换时其他登记字段相同，不保证活性物种、金属负载或分析 QC 已被独立验证。`check_false` 等字段保留在输出中供审计。方向一致性与双向异质性是描述性指标，不能单独建立隐藏机理 regime。', '', '## 复现', '', '```powershell', 'python code/condition_network_meta.py', '```', '', '生成文件：`results/condition_network_meta.csv`。']
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {OUT_CSV} rows={len(out)} swap={len(reg)} pair={len(pair_out)} slot={len(slot_out)}')
if __name__ == '__main__':
    main()
