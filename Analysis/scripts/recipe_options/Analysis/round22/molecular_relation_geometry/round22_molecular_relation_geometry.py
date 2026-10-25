"""Bridge candidate-mask geometry to the existing molecular/path registries.

All contrasts are descriptive.  Endpoint, acquisition-quartet and edit-pair
weighting answer different questions and are reported separately."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
R7 = ROOT / 'round7/joint_reactant_connectivity'
FULL, EVENTS, NON_EVENTS = (7853943, 68367, 7785576)
EXACT = ['neither_exact', 'AB_only_exact', 'CD_only_exact', 'both_exact']
BOOLS = ['distributed_all_target_coverage', 'shared_all_target_coverage', 'all_target_union', 'has_any_all_target_low_candidate', 'all_union_without_full_mask', 'distributed_without_full_mask', 'within_event_negative_alignment']
MEANS = ['candidate_mask_entropy_bits', 'candidate_mask_row_total', 'joint_deficit', 'K_joint', 'min_individual_K', 'individual_K_range', 'minimum_all_eight_endpoint_yields', 'minimum_low_node_route_cost', 'n_common_measured_recipes', 'n_common_graph_edges', 'n_common_graph_components', 'morgan_tanimoto_ab', 'morgan_tanimoto_cd']

def read(path, **kw):
    return pd.read_csv(path, low_memory=False, **kw)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def pack(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def metrics(frame):
    n = len(frame)
    result = {'n_event_views': n, 'n_acquisition_quartets': frame.acquisition_quartet_id.nunique(), 'n_grids': frame.grid_id.nunique()}
    result.update({'n_' + c: int(frame[c].sum()) for c in BOOLS})
    result.update({c + '_fraction': float(frame[c].mean()) if n else np.nan for c in BOOLS})
    result.update({'mean_' + c: float(frame[c].mean()) if n else np.nan for c in MEANS})
    return result

def summarize(frame, keys):
    rows = []
    for labels, part in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        rows.append({**dict(zip(keys, labels)), **metrics(part)})
    return pd.DataFrame(rows)

def weighted_summaries(event):
    columns = BOOLS + MEANS
    rows = []
    for mode, units in [('event_view', None), ('acquisition_quartet', ['acquisition_quartet_id']), ('molecular_grid', ['grid_id']), ('AB_edit', ['dataset_id', 'edit_id_ab']), ('CD_edit', ['dataset_id', 'edit_id_cd']), ('edit_pair', ['dataset_id', 'edit_id_ab', 'edit_id_cd'])]:
        for pattern, part in event.groupby('molecular_exact_pattern', sort=True):
            unit = part if units is None else part.groupby(units, dropna=False)[columns].mean()
            row = {'weighting_unit': mode, 'molecular_exact_pattern': pattern, 'n_event_views': len(part), 'n_weighting_units': len(unit)}
            row.update({'mean_' + c: float(unit[c].mean()) for c in columns})
            rows.append(row)
    return pd.DataFrame(rows)

def denominator_summary(scope, event, keys, include=None):
    den = scope.groupby(keys, as_index=False, dropna=False).agg(n_all_endpoint_views=('n_common_high_endpoint_pairs_with_full_graph_path', 'sum'), n_registered_acquisition_quartets=('acquisition_quartet_id', 'nunique'), n_registered_molecular_grids=('grid_id', 'nunique'))
    if include is not None:
        den = include.merge(den, on=keys, how='left')
    tab = den.merge(summarize(event, keys), on=keys, how='left')
    count = [c for c in tab if c.startswith('n_')]
    tab[count] = tab[count].fillna(0).astype(int)
    tab['n_non_event_views'] = tab.n_all_endpoint_views - tab.n_event_views
    tab['event_rate_full_endpoint_denominator'] = np.where(tab.n_all_endpoint_views > 0, tab.n_event_views / tab.n_all_endpoint_views, np.nan)
    tab['all_union_without_full_mask_rate_full_endpoint_denominator'] = np.where(tab.n_all_endpoint_views > 0, tab.n_all_union_without_full_mask / tab.n_all_endpoint_views, np.nan)
    return tab

def molecular_scope():
    scope = read(R7 / 'all_acquisition_quartet_scope.csv.gz')
    acquisition_columns = [f'{field}_{side}' for field in ['archive', 'plate', 'acquisition'] for side in ['ac', 'bc', 'ad', 'bd']]
    acquisition = read(R7 / 'all_acquisition_quartet_summary.csv.gz', usecols=['acquisition_quartet_id'] + acquisition_columns)
    scope = scope.merge(acquisition, on='acquisition_quartet_id', how='left', validate='one_to_one')
    assert scope[acquisition_columns].notna().all().all()
    annotations = read(R7 / 'complete_grid_molecular_relations.csv.gz', keep_default_na=False)
    assert annotations.grid_id.is_unique and scope.acquisition_quartet_id.is_unique
    assert int(scope.n_common_high_endpoint_pairs_with_full_graph_path.sum()) == FULL
    scope = scope.merge(annotations, on='grid_id', validate='many_to_one', how='left')
    assert scope.has_exact_edit_either_side.notna().all()
    registry = pd.concat([read(ROOT / 'round7/partner_conditioned_edits/molecular_relation_registry.csv.gz', keep_default_na=False), read(R7 / 'supplementary_molecular_relation_registry.csv.gz', keep_default_na=False)], ignore_index=True)
    key = ['dataset_id', 'role', 'molecular_pair_key']
    check = registry.groupby(key)[['edit_id', 'structural_relation', 'exact_graph_method']].nunique()
    assert (check <= 1).all().all()
    registry = registry.drop_duplicates(key)
    for side, role in [('ab', 'sub_1_smiles'), ('cd', 'sub_2_smiles')]:
        sub = registry[registry.role.eq(role)]
        lookup = sub.set_index(['dataset_id', 'molecular_pair_key'])
        pairs = scope.apply(lambda r: (r.dataset_id, pack(sorted([r['molecule_' + x] for x in side]))), axis=1)
        assert pairs.isin(lookup.index).all()
        scope['morgan_tanimoto_' + side] = pairs.map(lookup.morgan_radius2_2048_tanimoto).astype(float)
        scope['molecular_pair_id_' + side] = pairs.map(lookup.molecular_pair_id)
        assert (pairs.map(lookup.edit_id).to_numpy() == scope['edit_id_' + side].to_numpy()).all()
        assert (pairs.map(lookup.has_existing_exact_neighbor_rule).to_numpy() == scope['has_exact_edit_' + side].to_numpy()).all()
    scope['molecular_exact_pattern'] = np.select([scope.has_exact_edit_ab & scope.has_exact_edit_cd, scope.has_exact_edit_ab, scope.has_exact_edit_cd], ['both_exact', 'AB_only_exact', 'CD_only_exact'], default='neither_exact')
    scope['molecular_relation_pattern'] = scope.structural_relation_ab + ';' + scope.structural_relation_cd
    return scope

def conditional_r6_links(scope):
    r6 = read(ROOT / 'round6/path_organization/all_candidate_organization_summary.csv.gz', keep_default_na=False)
    flags = ['any_common_high_pair_equal_count_topk_switch', 'any_common_high_pair_equal_count_same_group_topk_switch', 'any_connectivity_switch_gap10pp']

    def block(row, suffix):
        return tuple((str(row[c + '_' + suffix]) for c in ['archive', 'plate', 'acquisition', 'target_key']))
    lookup = {}
    for row in r6.to_dict('records'):
        k = (row['dataset_id'], *sorted([block(row, 'a'), block(row, 'b')]))
        old = lookup.setdefault(k, {'graph_pairs': set(), 'candidate_ids': set(), **{f: False for f in flags}})
        old['graph_pairs'].add(row['graph_pair_id'])
        old['candidate_ids'].add(row['candidate_id'])
        for f in flags:
            old[f] = old[f] or bool(row[f])
    rows = []
    for row in scope.to_dict('records'):
        for relation, s0, s1 in [('AB_at_C', 'ac', 'bc'), ('AB_at_D', 'ad', 'bd'), ('CD_at_A', 'ac', 'ad'), ('CD_at_B', 'bc', 'bd')]:
            match = lookup.get((row['dataset_id'], *sorted([block(row, s0), block(row, s1)])))
            if match is not None:
                rows.append({'acquisition_quartet_id': row['acquisition_quartet_id'], 'grid_id': row['grid_id'], 'dataset_id': row['dataset_id'], 'relation_with_fixed_partner': relation, 'n_selected_r6_candidates': len(match['candidate_ids']), 'r6_graph_pair_ids': pack(sorted(match['graph_pairs'])), **{f: match[f] for f in flags}})
    links = pd.DataFrame(rows)
    save(links, 'round6_selected_candidate_quartet_links.csv.gz')
    if len(links):
        grouped = links.groupby('acquisition_quartet_id').agg(r6_selected_pair_registry_link_count=('relation_with_fixed_partner', 'size'), r6_any_equal_count_reorganization_link=(flags[0], 'max'), r6_any_equal_count_same_groups_reorganization_link=(flags[1], 'max'), r6_any_connectivity_switch10pp_link=(flags[2], 'max'))
        scope = scope.merge(grouped, on='acquisition_quartet_id', how='left', validate='one_to_one')
    else:
        scope['r6_selected_pair_registry_link_count'] = 0
        for c in ['r6_any_equal_count_reorganization_link', 'r6_any_equal_count_same_groups_reorganization_link', 'r6_any_connectivity_switch10pp_link']:
            scope[c] = False
    scope['r6_selected_pair_registry_link_count'] = scope.r6_selected_pair_registry_link_count.fillna(0).astype(int)
    for c in ['r6_any_equal_count_reorganization_link', 'r6_any_equal_count_same_groups_reorganization_link', 'r6_any_connectivity_switch10pp_link']:
        scope[c] = scope[c].astype('boolean').fillna(False).astype(bool)
    return (scope, links)

def paired_contrasts(event):
    rows = []
    measures = ['joint_deficit', 'K_joint', 'min_individual_K', 'individual_K_range', 'minimum_all_eight_endpoint_yields', 'candidate_mask_entropy_bits']
    levels = [('quartet', ['acquisition_quartet_id']), ('quartet_route_candidate_count', ['acquisition_quartet_id', 'minimum_low_node_route_cost', 'candidate_mask_row_total'])]
    contrasts = [('all_union_vs_partial', 'all_target_union'), ('empty_intersection_all_union_vs_remaining', 'distributed_all_target_coverage'), ('all_union_no_full_mask_vs_partial', 'all_union_without_full_mask')]
    for label, keys in levels:
        for contrast, flag in contrasts:
            eligible = event if contrast != 'all_union_no_full_mask_vs_partial' else event[event.all_union_without_full_mask | ~event.all_target_union]
            groups = eligible.groupby(keys + [flag], dropna=False)
            agg = groups[measures].mean().unstack(flag)
            counts = groups.size().unstack(flag, fill_value=0)
            if True not in counts.columns or False not in counts.columns:
                continue
            comparable = counts.index[(counts[True] > 0) & (counts[False] > 0)]
            for idx in comparable:
                vals = idx if isinstance(idx, tuple) else (idx,)
                row = {'comparison': contrast, 'matching_level': label, **dict(zip(keys, vals)), 'n_flag_event_views': int(counts.loc[idx, True]), 'n_reference_event_views': int(counts.loc[idx, False])}
                for m in measures:
                    row['delta_' + m] = float(agg.loc[idx, (m, True)] - agg.loc[idx, (m, False)])
                rows.append(row)
    paired = pd.DataFrame(rows)
    metadata = event[['acquisition_quartet_id', 'dataset_id', 'rxn_class', 'molecular_exact_pattern']].drop_duplicates()
    return paired.merge(metadata, on='acquisition_quartet_id', validate='many_to_one')

def report(summary, weighted, geometry, paired_summary, scope, event, links, edit_contexts):
    text = ['# Round 22: 分子关系、候选覆盖与路径组织的连接', '', '日期：2026-10-17。', '', '本轮把候选低状态几何接回已有分子关系和四方共同实测配方图。完整口径为 7,994,464 个端点视图 = 70,046 个事件 + 7,924,418 个非事件；18 类反应、11 个零事件类别全部保留。', '', '## 定义需要先分清', '', '`union=15, intersection=0` 表示候选集合覆盖四靶点，且不存在一个在所有候选上都低的靶点。它不排除集合中存在一个单独的四靶点低状态节点。因而本轮进一步记录 `mask1111` 是否存在。', '', 'Round 14 的 `within_event_negative_alignment` 由事件内 Pearson 相关的 `.lt(0)` 定义；Pearson为NaN的未定义事件被计为False。因此本轮沿用该布尔列得到的是全事件中的负值检出比例，包含相关可计算性和可计算事件中符号分布两部分，不能直接解释为可计算事件中反向关系更常见或更少对齐。例如35.46%与80.20%的差异须在Round 23拆分undefined/positive/zero/negative后再判断。当前结果仅区分联合覆盖几何与原布尔检出指标，尚不足以据此断言方向性为独立结构轴。', '', '| 覆盖状态 | 事件视图 | 占事件 | 全事件负值检出 | Kjoint均值 | joint deficit均值 |', '|---|---:|---:|---:|---:|---:|']
    for r in geometry.to_dict('records'):
        text.append(f"| {r['coverage_requirement_class']} | {r['n_event_views']:,} | {r['n_event_views'] / EVENTS:.2%} | {r['within_event_negative_alignment_fraction']:.2%} | {r['mean_K_joint']:.4f} | {r['mean_joint_deficit']:.4f} |")
    distributed = event[event.distributed_all_target_coverage]
    text += ['', f'19,847 个 union=15/intersection=0 事件中，有 {int(distributed.has_any_all_target_low_candidate.sum()):,} 个（{distributed.has_any_all_target_low_candidate.mean():.2%}）包含至少一个 mask1111 节点；其余 {int((~distributed.has_any_all_target_low_candidate).sum()):,} 个必须由多个候选的低靶点集合共同构成四靶点覆盖。后者是更严格的多节点联合覆盖口径；原始四类几何完整保留。', '', '## 分子关系层完整分母', '', 'exact 标记直接继承 Round 6/7 的图规则注册结果。AB/CD 表示两侧分子比较，不表示分子相似度的全序，也不表示已识别出的化学机制。', '', '| 分子关系 | 全端点 | 事件 | 事件率 | 四靶点union | union且无mask1111 | 全事件负值检出 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.to_dict('records'):
        text.append(f"| {r['molecular_exact_pattern']} | {r['n_all_endpoint_views']:,} | {r['n_event_views']:,} | {r['event_rate_full_endpoint_denominator']:.3%} | {r['all_target_union_fraction']:.2%} | {r['all_union_without_full_mask_fraction']:.2%} | {r['within_event_negative_alignment_fraction']:.2%} |")
    text += ['', '分子层之间的差异同时受反应类别、实测图规模、同一 edit 的伙伴覆盖和事件视图复用影响。不能把上述发生率写成局部编辑带来的因果风险。类别×分子关系表保留18×4的72行，零端点分母的发生率记为缺失。', '', '## 事件加权与结构等权', '', '| 分子关系 | 权重单位 | 单位数 | 四靶点union | union且无mask1111 | 全事件负值检出 |', '|---|---|---:|---:|---:|---:|']
    for r in weighted[weighted.weighting_unit.isin(['event_view', 'acquisition_quartet', 'edit_pair'])].to_dict('records'):
        text.append(f"| {r['molecular_exact_pattern']} | {r['weighting_unit']} | {r['n_weighting_units']:,} | {r['mean_all_target_union']:.2%} | {r['mean_all_union_without_full_mask']:.2%} | {r['mean_within_event_negative_alignment']:.2%} |")
    text += ['', 'edit-pair 是同 dataset 下 AB/CD 分子关系的组合；同一结构单位内部先求事件均值，再对单位等权。等权结果仍是测量结构的描述，不能把单位数当独立实验重复。', '', '## 同一精确分子比较跨伙伴关系的变化', '', f'在有事件的子集中，共有 {int((edit_contexts.has_exact_edit & edit_contexts.n_opposite_molecular_pairs.ge(2)).sum()):,} 个dataset×侧别×精确分子比较覆盖至少两个不同的另一侧分子对。其中 {int((edit_contexts.has_exact_edit & edit_contexts.n_opposite_molecular_pairs.ge(2) & edit_contexts.union_present_in_some_contexts_absent_in_others).sum()):,} 个同时出现过有all-target union和完全没有all-target union的另一侧分子对背景。', '', '这里的伙伴背景是一对另一侧分子，不是一个固定伙伴分子。它表明相同精确分子比较并不唯一决定联合覆盖几何。该结果条件于已出现事件的图，且图测量覆盖、端点和路线集合仍可随背景改变；不能解释为单独改变伙伴导致几何变化的实验效应。完整逐edit上下文表保留所有分子比较。', '', '## 在相同四元组内部比较路径量', '', 'Kjoint、各靶点K及deficit均为0-1产率尺度，乘100对应百分点。差值为标记组减参照组。第二层进一步固定最低路线代价及候选节点数，仅保留两种状态同时存在的匹配层。它检验同一已测分子和图背景内的共变，不是匹配后的因果效应。', '', '| 比较 | 匹配层 | 配对层数 | quartet数 | deficit差值均值 | Kjoint差值均值 | K范围差值均值 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in paired_summary[paired_summary.scope.eq('all')].to_dict('records'):
        text.append(f"| {r['comparison']} | {r['matching_level']} | {r['n_matched_strata']:,} | {r['n_acquisition_quartets']:,} | {r['mean_delta_joint_deficit']:.4f} | {r['mean_delta_K_joint']:.4f} | {r['mean_delta_individual_K_range']:.4f} |")
    text += ['', '这些量均由同一事件图构成，部分关联有数学约束。候选集合增大、mask多样性增加与union覆盖之间也有定义关系；本轮用实际分子关系、类别、路线代价及候选数分层显示其边界，不将其视为新的实验重复。', '', f"全体 {event.acquisition_quartet_id.nunique():,} 个有事件的四元组中，只有 {int(event.groupby('acquisition_quartet_id').all_target_union.nunique().eq(2).sum()):,} 个同时提供all-target union与partial union事件；进一步固定路线代价和候选数后，只有13个匹配层。因此同图内部比较的可识别支持较小，不能把大规模事件数当作大量独立的分子关系复现。", '', '## Round 6 等数量连接重组的条件回接', '', f"在完整 {len(scope):,} 个 acquisition quartets 中，{scope.r6_selected_pair_registry_link_count.gt(0).sum():,} 个能回接 Round 6 已选择的配对候选；其事件子集有 {event.loc[event.r6_selected_pair_registry_link_count.gt(0), 'acquisition_quartet_id'].nunique():,} 个 quartet、{event.r6_selected_pair_registry_link_count.gt(0).sum():,} 个事件视图。共保留 {len(links):,} 条 quartet×固定伙伴关系关联。", '', '这一回接只表明其中一对采集分子已在 Round 6 的选择候选内接受过等数量连接分析。Round 6 使用双方共同图，当前事件使用四方共同图；关联不保证端点、候选图或路径相同。因此该关联仅作为证据来源追踪，不用于估计全库等数量重组发生率。', '', '## 科学问题向前推进', '', '现有证据把三个可区分的层级连了起来：分子比较如何改变配方连通瓶颈；四目标能否由同一候选低状态解释；多个候选的低靶点集合怎样共同组成覆盖。中心问题可以保持为：**分子对改变时，组分替换所需的兼容关系如何从单节点状态组织为跨节点、跨靶点的联合约束？**', '', '本轮数据支撑描述性分层和已有证据回接。exact编辑、较高相似度、分布式掩码和较高路线代价都不能单独充当机制标签。后续holdout检验可评估可迁移性，单独不能证成因果关系。测量配方图的替换路径不是连续实验步骤。', '', '## 文件与复算', '', '`round22_event_molecular_geometry.csv.gz` 是全部70,046事件表；`molecular_scope_quartet_registry.csv.gz`保留全部四元组及分母。`molecular_exact_denominator_summary.csv`、`reaction_class_exact_complete.csv`、`reaction_class_complete.csv` 保留全端点口径。`geometry_by_molecular_relation.csv`、`geometry_by_exact_route_candidate_count.csv` 和 `geometry_by_exact_class.csv` 给出完整交叉层。', '', '`weighting_sensitivity.csv`、`molecular_edit_context_summary.csv.gz` 保留复用与结构等权结果；`within_quartet_geometry_contrasts.csv.gz`、`within_quartet_geometry_contrast_summary.csv` 保留同图背景差值。`round6_selected_candidate_quartet_links.csv.gz` 是条件来源关联。', '', '生成脚本与独立 verifier 位于当前目录；独立核验从已保存源表重新构造分母、分子规则连接、mask状态、结构等权和匹配层差值。']
    (OUT / 'ROUND22_MOLECULAR_RELATION_GEOMETRY_REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')

def main():
    scope = molecular_scope()
    scope, links = conditional_r6_links(scope)
    if not all(((ROOT / p).is_file() for p in ['round19/crosslayer_elevation/crosslayer_event_summary.csv.gz', 'round14/target_deficit_alignment/target_deficit_alignment_event_summary.csv.gz', 'round13/target_signature/target_signature_denominator_summary.csv'])):
        save(scope[[c for c in scope if not c.startswith(('target_key', 'product_smiles', 'common_recipe_ids'))]], 'molecular_scope_quartet_registry.csv.gz')
        print('M1 port: wrote molecular_scope_quartet_registry.csv.gz only; event summaries need rounds 13/14/19', flush=True)
        return
    event = read(ROOT / 'round19/crosslayer_elevation/crosslayer_event_summary.csv.gz')
    r14 = read(ROOT / 'round14/target_deficit_alignment/target_deficit_alignment_event_summary.csv.gz', usecols=['event_key', 'endpoint0_recipe', 'endpoint1_recipe', 'K_ac', 'K_bc', 'K_ad', 'K_bd', 'K_joint', 'min_individual_K', 'individual_K_range', 'minimum_all_eight_endpoint_yields'])
    assert len(event) == EVENTS and event.event_key.is_unique and r14.event_key.is_unique
    event = event.merge(r14, on='event_key', validate='one_to_one', how='left')
    take = ['acquisition_quartet_id', 'molecular_exact_pattern', 'molecular_relation_pattern', 'morgan_tanimoto_ab', 'morgan_tanimoto_cd', 'edit_id_ab', 'edit_id_cd', 'structural_relation_ab', 'structural_relation_cd', 'exact_graph_method_ab', 'exact_graph_method_cd', 'has_exact_edit_ab', 'has_exact_edit_cd', 'n_common_measured_recipes', 'n_common_graph_edges', 'n_common_graph_components', 'r6_selected_pair_registry_link_count', 'r6_any_equal_count_reorganization_link', 'r6_any_equal_count_same_groups_reorganization_link', 'r6_any_connectivity_switch10pp_link']
    event = event.merge(scope[take], on='acquisition_quartet_id', validate='many_to_one', how='left')
    assert event[take].notna().all().all() and event.K_joint.notna().all()
    event['all_target_union'] = event.event_union_low_mask_code.eq(15)
    event['has_any_all_target_low_candidate'] = event.candidate_mask_count_1111.gt(0)
    event['all_union_without_full_mask'] = event.all_target_union & ~event.has_any_all_target_low_candidate
    event['distributed_without_full_mask'] = event.distributed_all_target_coverage & ~event.has_any_all_target_low_candidate
    event['coverage_requirement_class'] = np.select([~event.all_target_union, event.all_union_without_full_mask, event.event_intersection_low_mask_code.eq(15)], ['partial_union', 'all_union_without_any_mask1111', 'every_candidate_mask1111'], default='all_union_with_mask1111_mixed')
    denominator = read(ROOT / 'round13/target_signature/target_signature_denominator_summary.csv')
    classes = denominator.loc[denominator.scope.eq('rxn_class'), ['rxn_class']]
    summary = denominator_summary(scope, event, ['molecular_exact_pattern'], pd.DataFrame({'molecular_exact_pattern': EXACT}))
    save(summary, 'molecular_exact_denominator_summary.csv')
    complete = pd.MultiIndex.from_product([classes.rxn_class, EXACT], names=['rxn_class', 'molecular_exact_pattern']).to_frame(index=False)
    class_exact = denominator_summary(scope, event, ['rxn_class', 'molecular_exact_pattern'], complete)
    save(class_exact, 'reaction_class_exact_complete.csv')
    class_summary = denominator_summary(scope, event, ['rxn_class'], classes)
    save(class_summary, 'reaction_class_complete.csv')
    geometry = summarize(event, ['coverage_requirement_class'])
    save(geometry, 'coverage_requirement_summary.csv')
    save(summarize(event, ['molecular_exact_pattern', 'structural_relation_ab', 'structural_relation_cd', 'geometry_class', 'coverage_requirement_class']), 'geometry_by_molecular_relation.csv')
    save(summarize(event, ['molecular_exact_pattern', 'minimum_low_node_route_cost', 'candidate_mask_row_total', 'coverage_requirement_class']), 'geometry_by_exact_route_candidate_count.csv')
    save(summarize(event, ['molecular_exact_pattern', 'rxn_class', 'coverage_requirement_class']), 'geometry_by_exact_class.csv')
    weighted = weighted_summaries(event)
    save(weighted, 'weighting_sensitivity.csv')
    editrows = []
    for side, other in [('ab', 'cd'), ('cd', 'ab')]:
        for keys, part in event.groupby(['dataset_id', 'edit_id_' + side], sort=True):
            contexts = part.groupby('edit_id_' + other).agg(all_target_union_fraction=('all_target_union', 'mean'), all_union_without_full_mask_fraction=('all_union_without_full_mask', 'mean'))
            editrows.append({'dataset_id': keys[0], 'side': side.upper(), 'edit_id': keys[1], 'structural_relation': part['structural_relation_' + side].iloc[0], 'has_exact_edit': bool(part['has_exact_edit_' + side].iloc[0]), 'n_opposite_molecular_pairs': part['edit_id_' + other].nunique(), 'minimum_context_all_union_fraction': contexts.all_target_union_fraction.min(), 'maximum_context_all_union_fraction': contexts.all_target_union_fraction.max(), 'union_present_in_some_contexts_absent_in_others': bool(contexts.all_target_union_fraction.eq(0).any() and contexts.all_target_union_fraction.gt(0).any()), 'minimum_context_union_no_full_mask_fraction': contexts.all_union_without_full_mask_fraction.min(), 'maximum_context_union_no_full_mask_fraction': contexts.all_union_without_full_mask_fraction.max(), 'union_no_full_mask_present_in_some_contexts_absent_in_others': bool(contexts.all_union_without_full_mask_fraction.eq(0).any() and contexts.all_union_without_full_mask_fraction.gt(0).any()), **metrics(part)})
    edits = pd.DataFrame(editrows)
    save(edits, 'molecular_edit_context_summary.csv.gz')
    inventory = event.groupby(['acquisition_quartet_id', 'grid_id', 'dataset_id', 'rxn_class', 'molecular_exact_pattern'], as_index=False).agg(n_event_views=('event_key', 'size'), n_coverage_requirement_classes=('coverage_requirement_class', 'nunique'), n_union_presence_states=('all_target_union', 'nunique'), n_intersection_union_geometry_states=('geometry_class', 'nunique'), all_target_union_fraction=('all_target_union', 'mean'), all_union_without_full_mask_fraction=('all_union_without_full_mask', 'mean'))
    save(inventory, 'quartet_geometry_state_inventory.csv')
    paired = paired_contrasts(event)
    save(paired, 'within_quartet_geometry_contrasts.csv.gz')
    prows = []
    for scope_name, cols in [('all', []), ('molecular_exact_pattern', ['molecular_exact_pattern']), ('rxn_class', ['rxn_class'])]:
        for labels, part in paired.groupby(['comparison', 'matching_level'] + cols, sort=True):
            if not isinstance(labels, tuple):
                labels = (labels,)
            row = {'scope': scope_name, **dict(zip(['comparison', 'matching_level'] + cols, labels)), 'n_matched_strata': len(part), 'n_acquisition_quartets': part.acquisition_quartet_id.nunique()}
            for col in [c for c in paired if c.startswith('delta_')]:
                row['mean_' + col] = part[col].mean()
                row['median_' + col] = part[col].median()
                row['positive_' + col + '_fraction'] = part[col].gt(1e-12).mean()
                row['negative_' + col + '_fraction'] = part[col].lt(-1e-12).mean()
            prows.append(row)
    ps = pd.DataFrame(prows)
    save(ps, 'within_quartet_geometry_contrast_summary.csv')
    save(event, 'round22_event_molecular_geometry.csv.gz')
    scope_columns = [c for c in scope if not c.startswith(('target_key', 'product_smiles', 'common_recipe_ids'))]
    save(scope[scope_columns], 'molecular_scope_quartet_registry.csv.gz')
    report(summary, weighted, geometry, ps, scope, event, links, edits)
    checks = {'full_endpoint_views': int(summary.n_all_endpoint_views.sum()), 'event_views': len(event), 'non_event_views': int(summary.n_non_event_views.sum()), 'reaction_classes': len(class_summary), 'zero_event_classes': int(class_summary.n_event_views.eq(0).sum()), 'acquisition_quartets_with_events': event.acquisition_quartet_id.nunique(), 'unique_event_keys': bool(event.event_key.is_unique), 'distributed_events_with_any_mask1111': int((event.distributed_all_target_coverage & event.has_any_all_target_low_candidate).sum()), 'all_union_without_full_mask': int(event.all_union_without_full_mask.sum()), 'paired_contrast_rows': len(paired)}
    assert checks['full_endpoint_views'] == FULL and checks['event_views'] == EVENTS
    assert checks['non_event_views'] == NON_EVENTS and checks['reaction_classes'] == 18 and (checks['zero_event_classes'] == 11)
    (OUT / 'generation_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))
    print(summary[['molecular_exact_pattern', 'n_all_endpoint_views', 'n_event_views', 'all_target_union_fraction', 'all_union_without_full_mask_fraction']].to_string(index=False))
if __name__ == '__main__':
    main()
