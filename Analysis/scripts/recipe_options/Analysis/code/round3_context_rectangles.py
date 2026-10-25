"""Enumerate registered single-component context rectangles without new significance tests."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round3' / 'context_rectangles'
SLOTS = [f'Reagent{i}' for i in range(1, 7)] + ['Solvent', 'Temperature', 'Time']
BLOCK = ['archive', 'plate', 'acquisition', 'target_key']
VALUE_COLUMNS = ['y0_bg0', 'y1_bg0', 'delta_bg0', 'y0_bg1', 'y1_bg1', 'delta_bg1']
INPUTS = [ROOT / 'results' / 'matched_swap_summary.csv', ROOT / 'results' / 'delta_within_acquisition.csv.gz', ROOT / 'data' / 'condition_slot_registry.csv']

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def digest(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

def combine(values):
    return ';'.join(sorted(set((str(value) for value in values if str(value)))))

def change_kind(slot, token0, token1):
    if slot == 'Solvent':
        return 'solvent_identity'
    if slot in {'Temperature', 'Time'}:
        return 'physical_parameter'
    if not token0[0] or not token1[0]:
        return 'addition_removal'
    if token0[0] == token1[0]:
        return 'dose_only'
    return 'identity_same_dose' if token0[1] == token1[1] else 'identity_and_dose'

def reaction_class(dataset):
    return {'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig', 'Suzuki Bpin': 'Suzuki'}.get(dataset, dataset)

def registry_metadata(registry):
    records = {}
    for key, group in registry.groupby(['dataset_id', 'recipe_id'], sort=False):
        first = group.iloc[0]
        tokens = {slot: tuple(json.loads(first[f'{slot}_token'])) for slot in SLOTS}
        for slot in SLOTS:
            assert group[f'{slot}_token'].map(lambda value: tuple(json.loads(value)) == tokens[slot]).all()
        names = {f'Reagent{i}': first[f'Reagent{i}_Standardized Name'] or 'none' for i in range(1, 7)}
        names.update(Solvent=first['Standardized Name'], Temperature=f'{first.temp_id} C', Time=f'{first.time_id} h')
        records[key] = {'tokens': tokens, 'names': names, 'condition_ids': combine(group.condition_id), 'rxn_class': reaction_class(key[0]), 'registered_reaction_type': combine(group.reaction_type), 'registry_source_file': combine(group.source_file), 'registry_source_sheet': combine(group.source_sheet), 'registry_source_rows': combine(group.source_row)}
    return records

def enumerate_rectangles(swaps, recipes):
    rows = []
    proposed = 0
    for (dataset, exchange), group in swaps.groupby(['dataset_id', 'exchange_id'], sort=True):
        proposed += len(group) * (len(group) - 1) // 2
        for first, second in combinations(group.to_dict('records'), 2):
            assert first['slot'] == second['slot']
            focal = first['slot']
            recipe_a = recipes[dataset, first['recipe0']]
            recipe_b = recipes[dataset, second['recipe0']]
            assert recipe_a['tokens'][focal] == recipe_b['tokens'][focal]
            assert recipes[dataset, first['recipe1']]['tokens'][focal] == recipes[dataset, second['recipe1']]['tokens'][focal]
            changed = [slot for slot in SLOTS if slot != focal and recipe_a['tokens'][slot] != recipe_b['tokens'][slot]]
            if len(changed) != 1:
                continue
            context = changed[0]
            if recipe_a['tokens'][context] > recipe_b['tokens'][context]:
                first, second = (second, first)
                recipe_a, recipe_b = (recipe_b, recipe_a)
            corners = [first['recipe0'], first['recipe1'], second['recipe0'], second['recipe1']]
            assert len(set(corners)) == 4
            for swap in [first, second]:
                left, right = [recipes[dataset, swap[key]]['tokens'] for key in ['recipe0', 'recipe1']]
                assert [slot for slot in SLOTS if left[slot] != right[slot]] == [focal]
            token0, token1 = (recipe_a['tokens'][context], recipe_b['tokens'][context])
            background = {slot: value for slot, value in recipe_a['tokens'].items() if slot not in [focal, context]}
            row = {'rectangle_id': 'q_' + digest([dataset, sorted(corners)]), 'representation_id': 'f_' + digest([dataset, exchange, sorted([first['pair_id'], second['pair_id']])]), 'dataset_id': dataset, 'rxn_class': recipe_a['rxn_class'], 'registered_reaction_type': recipe_a['registered_reaction_type'], 'exchange_id': exchange, 'focal_slot': focal, 'focal_swap_kind': first['swap_kind'], 'component0_id': first['component0_id'], 'component1_id': first['component1_id'], 'component0_dose': first['component0_dose'], 'component1_dose': first['component1_dose'], 'component0_name': first['component0_name'], 'component1_name': first['component1_name'], 'context_slot': context, 'context_change_kind': change_kind(context, token0, token1), 'context0_token': json.dumps(token0), 'context1_token': json.dumps(token1), 'context0_name': recipe_a['names'][context], 'context1_name': recipe_b['names'][context], 'pair_id_bg0': first['pair_id'], 'pair_id_bg1': second['pair_id'], 'recipe_c0_bg0': first['recipe0'], 'recipe_c1_bg0': first['recipe1'], 'recipe_c0_bg1': second['recipe0'], 'recipe_c1_bg1': second['recipe1'], 'condition_c0_bg0': first['condition0_ids'], 'condition_c1_bg0': first['condition1_ids'], 'condition_c0_bg1': second['condition0_ids'], 'condition_c1_bg1': second['condition1_ids'], 'registered_four_recipe_key': json.dumps(sorted(corners)), 'fixed_background_json': json.dumps(background, sort_keys=True), 'bg0_individual_targets': first['n_substrate_targets'], 'bg1_individual_targets': second['n_substrate_targets']}
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(['dataset_id', 'rectangle_id', 'representation_id']).reset_index(drop=True)
    assert not frame.representation_id.duplicated().any()
    assert frame.groupby('rectangle_id').size().le(2).all()
    return (frame, proposed)

def metrics(targets, blocks=None):
    n = len(targets)
    result = {'n_common_targets': n}
    if blocks is not None:
        result.update(n_four_condition_acquisition_blocks=len(blocks), n_archives=blocks.archive.nunique(), n_archive_plate_labels=len(blocks[['archive', 'plate']].drop_duplicates()), fraction_four_source_check_true=float(blocks.four_source_check_true.mean()) if len(blocks) else np.nan)
    if not n:
        return result
    a, b = (targets.delta_bg0.to_numpy(float), targets.delta_bg1.to_numpy(float))
    dd = b - a
    centered = dd - dd.mean()
    rms = float(np.sqrt(np.mean(dd ** 2)))
    centered_rms = float(np.sqrt(np.mean(centered ** 2)))
    sd_a, sd_b = (np.std(a, ddof=1) if n > 1 else np.nan, np.std(b, ddof=1) if n > 1 else np.nan)
    opposite = (a > 0.05) & (b < -0.05) | (a < -0.05) & (b > 0.05)
    y = targets[['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']].to_numpy(float)
    result.update(mean_effect_bg0=float(a.mean()), mean_effect_bg1=float(b.mean()), mean_DD=float(dd.mean()), SD_DD=float(np.std(dd, ddof=1)) if n > 1 else np.nan, RMS_DD=rms, centered_profile_RMS_change=centered_rms, centered_profile_MAE_change=float(np.mean(np.abs(centered))), mean_shift_share_of_DD_square=float(dd.mean() ** 2 / rms ** 2) if rms > 0 else np.nan, centered_profile_share_of_DD_square=float(centered_rms ** 2 / rms ** 2) if rms > 0 else np.nan, SD_effect_bg0=float(sd_a), SD_effect_bg1=float(sd_b), effect_correlation=float(np.corrcoef(a, b)[0, 1]) if n > 1 and sd_a > 1e-12 and (sd_b > 1e-12) else np.nan, n_opposite_effects_both_gt5pp=int(opposite.sum()), fraction_opposite_effects_both_gt5pp=float(opposite.mean()), n_abs_DD_gt5pp=int((np.abs(dd) > 0.05).sum()), fraction_abs_centered_DD_gt5pp=float((np.abs(centered) > 0.05).mean()), n_all_four_yields_ge5pct=int((y >= 0.05).all(axis=1).sum()), n_all_four_yields_le1pct=int((y <= 0.01).all(axis=1).sum()), n_each_background_best_yield_ge10pct=int(((np.maximum(y[:, 0], y[:, 1]) >= 0.1) & (np.maximum(y[:, 2], y[:, 3]) >= 0.1)).sum()), mean_y0_bg0=float(y[:, 0].mean()), mean_y1_bg0=float(y[:, 1].mean()), mean_y0_bg1=float(y[:, 2].mean()), mean_y1_bg1=float(y[:, 3].mean()), mean_context_effect_at_c0=float((y[:, 2] - y[:, 0]).mean()), mean_context_effect_at_c1=float((y[:, 3] - y[:, 1]).mean()))
    assert abs(rms ** 2 - dd.mean() ** 2 - centered_rms ** 2) < 1e-12
    return result

def floor_sensitivity(targets):
    y = targets[['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']].to_numpy(float)
    filters = {'all_four_ge5pct': (y >= 0.05).all(axis=1), 'each_background_best_ge10pct': (np.maximum(y[:, 0], y[:, 1]) >= 0.1) & (np.maximum(y[:, 2], y[:, 3]) >= 0.1)}
    result = {}
    for label, selected in filters.items():
        subset = metrics(targets.loc[selected])
        for metric in ['n_common_targets', 'mean_DD', 'centered_profile_RMS_change', 'effect_correlation', 'n_opposite_effects_both_gt5pp', 'fraction_opposite_effects_both_gt5pp']:
            result[f'{label}_{metric}'] = subset.get(metric, np.nan)
    return result

def paired_blocks(first, second):
    paired = first.merge(second, on=BLOCK, suffixes=('_bg0', '_bg1'), validate='one_to_one')
    paired['four_source_check_true'] = paired.both_source_check_true_bg0 & paired.both_source_check_true_bg1
    paired['DD'] = paired.delta_bg1 - paired.delta_bg0
    assert np.allclose(paired.DD, paired.y1_bg1 - paired.y0_bg1 - paired.y1_bg0 + paired.y0_bg0, rtol=0, atol=1e-14)
    return paired

def unique_rectangle_summary(frame):
    rows = []
    for rectangle, group in frame.groupby('rectangle_id', sort=False):
        first = group.iloc[0]
        if len(group) > 1:
            assert group.n_common_targets.nunique() == 1
            assert group.n_four_condition_acquisition_blocks.nunique() == 1
            for metric in ['mean_DD', 'SD_DD', 'centered_profile_RMS_change', 'RMS_DD']:
                values = group[metric].dropna().to_numpy()
                if len(values):
                    assert np.ptp(values) < 1e-12, (rectangle, metric, values)
        record = {key: first[key] for key in ['rectangle_id', 'dataset_id', 'rxn_class', 'registered_reaction_type', 'registered_four_recipe_key', 'fixed_background_json']}
        record.update(n_focal_representations=len(group), focal_slots=combine(group.focal_slot), representative_focal_id=first.representation_id, n_common_targets=int(first.n_common_targets), n_four_condition_acquisition_blocks=int(first.n_four_condition_acquisition_blocks), n_archives=int(first.n_archives), n_archive_plate_labels=int(first.n_archive_plate_labels))
        for metric in ['mean_DD', 'SD_DD', 'centered_profile_RMS_change', 'RMS_DD', 'mean_shift_share_of_DD_square', 'centered_profile_share_of_DD_square']:
            record[metric] = first.get(metric, np.nan)
        for metric in ['n_common_targets', 'mean_DD', 'centered_profile_RMS_change']:
            column = f'all_four_ge5pct_{metric}'
            values = group[column].dropna().to_numpy()
            assert not len(values) or np.ptp(values) < 1e-12
            record[column] = first[column]
        record['minimum_effect_correlation_over_focal_axes'] = group.effect_correlation.min()
        record['maximum_fraction_opposite_over_focal_axes'] = group.fraction_opposite_effects_both_gt5pp.max()
        rows.append(record)
    return pd.DataFrame(rows)

def unique_floor_summary(unique):
    rows = []
    for baseline_minimum in [1, 20, 50]:
        baseline = unique.loc[unique.n_common_targets.ge(baseline_minimum)]
        for retained_minimum in [1, 20, 50]:
            kept = baseline.loc[baseline.all_four_ge5pct_n_common_targets.ge(retained_minimum)]
            rows.append({'minimum_full_panel_common_targets': baseline_minimum, 'minimum_floor_panel_common_targets': retained_minimum, 'floor_rule': 'all_four_target_mean_yields_ge5pct', 'n_baseline_unique_rectangles': len(baseline), 'n_retained_unique_rectangles': len(kept), 'n_retained_profile_RMS_ge5pp': int(kept.all_four_ge5pct_centered_profile_RMS_change.ge(0.05).sum()), 'median_retained_profile_RMS_pp': 100 * kept.all_four_ge5pct_centered_profile_RMS_change.median(), 'median_full_profile_RMS_pp_in_same_rectangles': 100 * kept.centered_profile_RMS_change.median(), 'median_retained_abs_mean_DD_pp': 100 * kept.all_four_ge5pct_mean_DD.abs().median()})
    return pd.DataFrame(rows)

def summarize_groups(frame, columns, label, universe=None):
    rows = []
    iterator = list(frame.groupby(columns, dropna=False, sort=True)) if columns else [((), frame)]
    if columns and universe is not None:
        existing = {key if isinstance(key, tuple) else (key,) for key, _ in iterator}
        iterator += [(key, frame.iloc[0:0]) for key in universe[columns].drop_duplicates().itertuples(index=False, name=None) if key not in existing]
    for key, group in iterator:
        key = key if isinstance(key, tuple) else (key,)
        for minimum in [1, 20, 50]:
            kept = group.loc[group.n_common_targets.ge(minimum)]
            row = {**dict(zip(columns, key)), 'panel': label, 'minimum_common_targets': minimum, 'n_registered_focal_representations': len(group), 'n_registered_unique_rectangles': group.rectangle_id.nunique(), 'n_zero_support_focal_representations': int(group.n_common_targets.eq(0).sum()), 'n_supported_focal_representations': len(kept), 'n_supported_unique_rectangles': kept.rectangle_id.nunique(), 'median_abs_mean_DD_pp': np.nan, 'median_centered_profile_RMS_change_pp': np.nan}
            for metric in ['n_focal_abs_mean_DD_ge5pp', 'n_unique_abs_mean_DD_ge5pp', 'n_focal_centered_profile_RMS_ge5pp', 'n_unique_centered_profile_RMS_ge5pp', 'n_focal_centered_profile_RMS_ge10pp', 'n_focal_correlation_lt0_5', 'n_focal_opposite_fraction_ge10pct']:
                row[metric] = 0
            if len(kept):
                row.update(median_abs_mean_DD_pp=100 * kept.mean_DD.abs().median(), median_centered_profile_RMS_change_pp=100 * kept.centered_profile_RMS_change.median(), n_focal_abs_mean_DD_ge5pp=int(kept.mean_DD.abs().ge(0.05).sum()), n_unique_abs_mean_DD_ge5pp=kept.loc[kept.mean_DD.abs().ge(0.05), 'rectangle_id'].nunique(), n_focal_centered_profile_RMS_ge5pp=int(kept.centered_profile_RMS_change.ge(0.05).sum()), n_unique_centered_profile_RMS_ge5pp=kept.loc[kept.centered_profile_RMS_change.ge(0.05), 'rectangle_id'].nunique(), n_focal_centered_profile_RMS_ge10pp=int(kept.centered_profile_RMS_change.ge(0.1).sum()), n_focal_correlation_lt0_5=int(kept.effect_correlation.lt(0.5).sum()), n_focal_opposite_fraction_ge10pct=int(kept.fraction_opposite_effects_both_gt5pp.ge(0.1).sum()))
            rows.append(row)
    return pd.DataFrame(rows)

def input_coverage(swaps, recipes, strict):
    scope = swaps.rename(columns={'slot': 'focal_slot', 'swap_kind': 'focal_swap_kind'}).copy()
    scope['rxn_class'] = [recipes[row.dataset_id, row.recipe0]['rxn_class'] for row in swaps.itertuples()]
    scope['registered_reaction_type'] = [recipes[row.dataset_id, row.recipe0]['registered_reaction_type'] for row in swaps.itertuples()]
    columns = ['dataset_id', 'rxn_class', 'focal_slot', 'focal_swap_kind']
    rows = []
    for key, group in scope.groupby(columns, sort=True):
        matched = strict
        for column, value in zip(columns, key):
            matched = matched.loc[matched[column].eq(value)]
        row = {**dict(zip(columns, key)), 'registered_reaction_type': combine(group.registered_reaction_type), 'n_input_swaps': len(group), 'n_input_exchanges': group.exchange_id.nunique(), 'n_registered_focal_representations': len(matched), 'n_registered_unique_rectangles': matched.rectangle_id.nunique()}
        for minimum in [1, 20, 50]:
            supported = matched.loc[matched.n_common_targets.ge(minimum)]
            row[f'n_strict_focal_n_ge{minimum}'] = len(supported)
            row[f'n_strict_unique_rectangles_n_ge{minimum}'] = supported.rectangle_id.nunique()
        rows.append(row)
    return (scope, pd.DataFrame(rows))

def select_cases(strict):
    candidates = strict.loc[strict.n_common_targets.ge(20)].sort_values(['centered_profile_RMS_change', 'n_common_targets'], ascending=False)
    chosen, used = ([], set())
    for _, row in candidates.drop_duplicates('dataset_id').head(10).iterrows():
        chosen.append({**row.to_dict(), 'case_role': 'dataset_high_profile_change'})
        used.add(row.rectangle_id)
    shift = strict.loc[strict.n_common_targets.ge(20)].sort_values('mean_shift_share_of_DD_square', ascending=False)
    for _, row in shift.iterrows():
        if row.rectangle_id not in used:
            chosen.append({**row.to_dict(), 'case_role': 'largest_mean_shift_square_share_reference'})
            used.add(row.rectangle_id)
            break
    sparse = strict.loc[strict.n_common_targets.between(1, 19)].sort_values('centered_profile_RMS_change', ascending=False)
    for _, row in sparse.iterrows():
        if row.rectangle_id not in used:
            chosen.append({**row.to_dict(), 'case_role': 'sparse_common_target_reference'})
            break
    result = pd.DataFrame(chosen)
    result.insert(0, 'case_id', [f'R3C{i:02}' for i in range(1, len(result) + 1)])
    result['case_selection'] = 'illustrations selected after full enumeration; not independent validation'
    return result

def add_optional_source_ids(case_blocks, normalized_path):
    if normalized_path is None:
        return pd.DataFrame()
    norm = pd.read_csv(normalized_path, keep_default_na=False, low_memory=False)
    norm = norm.loc[norm.strict_eligible.astype(str).str.lower().eq('true')]
    norm['yield_numeric'] = pd.to_numeric(norm.yield_numeric, errors='coerce')
    keys = ['dataset_id'] + BLOCK + ['recipe_id']
    cells = norm.groupby(keys, as_index=False).agg(source_yield=('yield_numeric', 'mean'), record_ids=('record_id', combine), source_condition_ids=('condition_id', combine))
    rows = []
    for corner, value in [('c0_bg0', 'y0_bg0'), ('c1_bg0', 'y1_bg0'), ('c0_bg1', 'y0_bg1'), ('c1_bg1', 'y1_bg1')]:
        needed = case_blocks[['case_id', 'rectangle_id', 'representation_id', 'dataset_id'] + BLOCK + [f'recipe_{corner}', value]].rename(columns={f'recipe_{corner}': 'recipe_id', value: 'matched_yield'})
        linked = needed.merge(cells, on=keys, how='left', validate='many_to_one')
        assert linked.record_ids.notna().all()
        assert np.allclose(linked.matched_yield, linked.source_yield, rtol=0, atol=1e-14)
        linked['corner'] = corner
        rows.append(linked)
    return pd.concat(rows, ignore_index=True)

def write_report(strict, target, unique, global_summary, dataset_summary, cases, manifest):

    def number(value, decimals=2):
        return 'NA' if pd.isna(value) else f'{value:.{decimals}f}'
    lines = ['# 第三轮：全库单组件交换的背景矩形', '', '全部来自既有数据；不新增p值、不声称独立验证或通用因果规律。', '', '## 1. 范围与去重', '', f"输入为{manifest['input_focal_swaps']}个既有合格交换，覆盖{manifest['input_datasets']}个dataset、{manifest['input_reaction_classes']}个合并反应类。按同dataset、同exchange_id枚举{manifest['unfiltered_same_exchange_background_pairs']}对背景，保留只改变另一个登记token槽的{len(strict)}个焦点表示，对应{len(unique)}个不同四recipe矩形，覆盖{manifest['datasets_with_rectangles']}个dataset、{manifest['reaction_classes_with_rectangles']}个合并反应类。一个矩形可沿两个组件方向表示；同一矩形的两轴不是独立重复。", '', '本轮仅覆盖已进入原matched_swap_summary的交换；该表本身要求各背景至少20个target。未进入此表的稀疏交换没有被判定为不存在。所有已枚举矩形均保留，包括四条件无共同acquisition或共同target不足20的情况。', '', '方向按登记token固定：焦点组件0→1在两个背景完全相同（包括两剂量），背景0→1也按token字典序固定。DD=Δ背景1−Δ背景0。主分析要求四格同dataset/archive/plate/acquisition/target，随后重复acquisition在target内平均；target仅共同的敏感性允许背景使用不同测量块，证据强度较低。', '', 'plate字段只是原始name0截至-PL数字的名称前缀；n_archive_plate_labels仅计算archive内不同该标签。它用于来源定位与四格匹配，不代表已验证的物理板数、独立实验次数或独立重复。archive也仅按原始归档标签计数。', '', '## 2. 全支持与覆盖敏感性', '', '| 面板 | 最少共同target | 有支持焦点表示 | 不同四recipe矩形 | 去均值profile RMS≥5pp的焦点表示/矩形 | 中位profile RMS/pp | 中位绝对均值DD/pp |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in global_summary.itertuples():
        lines.append(f'| {row.panel} | {row.minimum_common_targets} | {row.n_supported_focal_representations} | {row.n_supported_unique_rectangles} | {row.n_focal_centered_profile_RMS_ge5pp}/{row.n_unique_centered_profile_RMS_ge5pp} | {number(row.median_centered_profile_RMS_change_pp)} | {number(row.median_abs_mean_DD_pp)} |')
    floor_row = unique_floor_summary(unique).query('minimum_full_panel_common_targets == 20 and minimum_floor_panel_common_targets == 20').iloc[0]
    lines += ['', f'以唯一矩形为分母：严格n≥20的{int(floor_row.n_baseline_unique_rectangles)}个中，四格target平均产率均≥5%后仍有n≥20的为{int(floor_row.n_retained_unique_rectangles)}个；其中{int(floor_row.n_retained_profile_RMS_ge5pp)}个profile RMS≥5pp，中位{number(floor_row.median_retained_profile_RMS_pp)}pp。同一批矩形在完整target集的中位profile RMS为{number(floor_row.median_full_profile_RMS_pp_in_same_rectangles)}pp。未保留的矩形是该下限规则下覆盖不足，不能归入下限稳健结论。']
    lines += ['', '阈值5/10pp只用于描述幅度，不是显著性或已验证发现门槛。1-target矩形的中心化profile变化必为零，SD和相关无法估计；相关为空也可能因为某背景全部效应相同，不能补为零。焦点表示的相关和相反偏好取决于选哪条轴；唯一矩形表保留两轴的范围信息。', '', '## 3. 均值改变与底物特异性改变', '', '每个矩形在相同target上保留两个效应向量。均值DD描述整体效应平移；去均值profile RMS描述(Δ1−meanΔ1)与(Δ0−meanΔ0)的差异，SD(DD)是同一异质性信息的样本标准差版本。RMS(DD)^2=mean(DD)^2+centered_profile_RMS^2，脚本逐矩形验证该分解。不能把整体均值平移直接称为底物特异性改变。', '', '同时保存效应相关、两个背景的效应SD、同target两背景效应相反且各>5pp的数量/比例，以及四格均值和产率下限支持。去均值变化仍可能来自测量误差、终点下限和未完全平衡的底物覆盖；它描述观测响应profile随背景改变，尚不独立证明某结构机制。', '', '另逐矩形重算四格target平均产率均≥5%与每个背景至少一格≥10%的target子集。前者两轴使用相同target；后者依赖焦点轴。两种筛选都使用已观测产率，会改变target组成，不能称为独立验证或测量误差校正。完整值保存在all_rectangle_focal_summaries.csv及target_common_support_sensitivity.csv；前者的唯一矩形覆盖另存floor_sensitivity_unique_rectangle_summary.csv。', '', '## 4. 跨数据集覆盖', '', '| dataset | 注册焦点表示/唯一矩形 | 严格n≥20焦点表示/唯一矩形 | profile RMS≥5pp焦点表示/唯一矩形 |', '|---|---:|---:|---:|']
    for row in dataset_summary.loc[dataset_summary.panel.eq('strict_four_condition_acquisition') & dataset_summary.minimum_common_targets.eq(20)].itertuples():
        lines.append(f'| {row.dataset_id} | {row.n_registered_focal_representations}/{row.n_registered_unique_rectangles} | {row.n_supported_focal_representations}/{row.n_supported_unique_rectangles} | {int(row.n_focal_centered_profile_RMS_ge5pp)}/{int(row.n_unique_centered_profile_RMS_ge5pp)} |')
    lines += ['', 'rxn_class按dataset映射合并Buchwald-Hartwig(CMC)→Buchwald-Hartwig与Suzuki Bpin→Suzuki；原始登记名另存registered_reaction_type。20个输入dataset对应18类，有矩形的17个dataset对应16类。不把dataset数改称独立反应类数。按dataset/rxn_class/焦点槽/背景变化槽/交换类型的完整分组CSV包含低支持及零支持。不同分组可复用同一矩形和相同目标，分组计数不可直接累加成独立次数。', '', '## 5. 全来源示例', '', '以下是全枚举后选择的少量示意：优先各dataset的高profile变化、最大均值平方占比参照，以及一个低覆盖实例。其余结果未删除。产率四格顺序为背景0组件0/1、背景1组件0/1；百分数与pp分列。', '', '| 案例 | dataset | 焦点交换[登记dose]；背景交换[登记dose] | target/archive | 四格均值/% | meanDD/pp | profile RMS/pp | 相反>5pp target |', '|---|---|---|---:|---|---:|---:|---:|']
    for row in cases.itertuples():
        ys = '/'.join((number(100 * getattr(row, metric)) for metric in ['mean_y0_bg0', 'mean_y1_bg0', 'mean_y0_bg1', 'mean_y1_bg1']))
        focal_doses = [getattr(row, f'component{i}_dose') for i in [0, 1]]
        focal_names = [f"{getattr(row, f'component{i}_name')}[{focal_doses[i]}]" if pd.notna(focal_doses[i]) and str(focal_doses[i]) else getattr(row, f'component{i}_name') for i in [0, 1]]
        context_tokens = [json.loads(getattr(row, f'context{i}_token')) for i in [0, 1]]
        context_names = [f"{getattr(row, f'context{i}_name')}[{context_tokens[i][1]}]" if len(context_tokens[i]) > 1 and context_tokens[i][1] else getattr(row, f'context{i}_name') for i in [0, 1]]
        lines.append(f"| {row.case_id} | {row.dataset_id} | {'→'.join(focal_names)} ({row.focal_swap_kind})；{row.context_slot}:{'→'.join(context_names)} ({row.context_change_kind}) | {row.n_common_targets}/{row.n_archives} | {ys} | {number(100 * row.mean_DD)} | {number(100 * row.centered_profile_RMS_change)} | {int(row.n_opposite_effects_both_gt5pp)} |")
    supported = strict.loc[strict.n_common_targets.ge(20)]
    max_mean_share = supported.mean_shift_share_of_DD_square.max()
    shift_dominant_count = supported.loc[supported.mean_shift_share_of_DD_square.ge(0.7), 'rectangle_id'].nunique()
    lines += ['', f'严格n≥20矩形的最大mean(DD)^2/RMS(DD)^2为{number(100 * max_mean_share)}%；均值平方占比≥70%的唯一矩形数为{shift_dominant_count}。该范围内的大DD均方主要来自target间变化，仍需检查下限与测量误差。', '', '| 案例 | 四格均≥5%：n / profile RMS pp | 各背景最佳≥10%：n / profile RMS pp |', '|---|---:|---:|']
    for row in cases.itertuples():
        lines.append(f'| {row.case_id} | {int(row.all_four_ge5pct_n_common_targets)} / {number(100 * row.all_four_ge5pct_centered_profile_RMS_change)} | {int(row.each_background_best_ge10pct_n_common_targets)} / {number(100 * row.each_background_best_ge10pct_centered_profile_RMS_change)} |')
    source_description = f"本次selected_case_original_source_links.csv.gz包含{manifest['selected_source_corner_links']}个四格来源链接；每格原始record_id已与normalized记录相连并验证源产率。" if manifest['selected_source_corner_links'] else '本次未提供normalized记录，四格保留完整定位键，不虚构原始record_id。'
    lines += ['', '每例全部配对测量块、四格产率、登记recipe/condition、底物及产物SMILES保存在selected_case_four_condition_blocks.csv.gz；按case和corner对应的四条件登记来源另存selected_case_condition_registry.csv。', source_description, '', '## 6. 文件与重现', '', '- rectangle_registry.csv：全部焦点表示的四recipe、两轴token与固定背景。', '- all_rectangle_focal_summaries.csv：严格同acquisition全矩形统计；包含零支持。', '- target_common_support_sensitivity.csv：各背景先按target平均后的共同target敏感性。', '- unique_four_recipe_rectangles.csv：四recipe去重；不把两焦点轴当重复。', '- summary_global.csv、summary_by_dataset.csv、summary_by_rxn_class.csv、summary_by_dataset_class_slot_background.csv：全支持及n≥20/50分层。', '- input_swap_and_rectangle_coverage.csv：全部输入dataset/登记反应类/槽/交换类型的交换数及矩形覆盖；无矩形保留零行。', '- summary_by_focal_slot.csv、summary_by_context_slot.csv、summary_by_swap_kind.csv：单轴及交换类型覆盖，含无可配对类别。', '- selected_cases.csv及selected_case_*：少量全来源示例。', '- run_manifest.json：输入哈希、计数、计时和可选来源输入。', '', '运行code/round3_context_rectangles.py；--plan-only只做规模评估。所有数值列中未标pp/pct的效应与产率采用0–1单位。保留identity_same_dose、identity_and_dose、addition_removal、dose_only、solvent_identity，不将组件身份和剂量共同改变重写为纯身份效应。', '', '结论应落在观测范围：组件效应的均值和底物响应形状是否随另一组件背景变化，以及变化幅度和覆盖，而非仅凭矩形数量声称多条独立化学规律。']
    (OUT / 'CONTEXT_RECTANGLE_FINDINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main(plan_only=False, source_normalized=None):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    swaps = pd.read_csv(INPUTS[0], keep_default_na=False, low_memory=False)
    registry = pd.read_csv(INPUTS[2], keep_default_na=False, low_memory=False)
    recipes = registry_metadata(registry)
    rectangles, proposed = enumerate_rectangles(swaps, recipes)
    save(rectangles, 'rectangle_registry.csv')
    plan = {'input_focal_swaps': len(swaps), 'unfiltered_same_exchange_background_pairs': proposed, 'input_datasets': swaps.dataset_id.nunique(), 'input_reaction_classes': swaps.dataset_id.map(reaction_class).nunique(), 'registered_focal_representations': len(rectangles), 'unique_four_recipe_rectangles': rectangles.rectangle_id.nunique(), 'datasets_with_rectangles': rectangles.dataset_id.nunique(), 'reaction_classes_with_rectangles': rectangles.rxn_class.nunique(), 'required_swap_pair_ids': len(set(rectangles.pair_id_bg0) | set(rectangles.pair_id_bg1)), 'planned_strategy': 'One long-table read; per-pair indexes; no whole-table Cartesian product; preserve all support levels.'}
    (OUT / 'scale_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SCALE', plan, flush=True)
    if plan_only:
        return
    long = pd.read_csv(INPUTS[1], keep_default_na=False, low_memory=False)
    needed = set(rectangles.pair_id_bg0) | set(rectangles.pair_id_bg1)
    long = long.loc[long.pair_id.isin(needed)].copy()
    assert not long.duplicated(['pair_id'] + BLOCK).any()
    info = long[['dataset_id', 'target_key', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].drop_duplicates()
    assert not info.duplicated(['dataset_id', 'target_key']).any()
    keep = BLOCK + ['y0', 'y1', 'delta', 'both_source_check_true']
    pair_data = {pair_id: group[keep].reset_index(drop=True) for pair_id, group in long.groupby('pair_id', sort=False)}
    target_data = {pair_id: group.groupby('target_key', as_index=False)[['y0', 'y1', 'delta']].mean() for pair_id, group in pair_data.items()}
    del long
    strict_rows, target_rows = ([], [])
    for index, row in enumerate(rectangles.to_dict('records')):
        first, second = (pair_data[row['pair_id_bg0']], pair_data[row['pair_id_bg1']])
        blocks = paired_blocks(first, second)
        targets = blocks.groupby('target_key', as_index=False)[VALUE_COLUMNS].mean()
        strict_rows.append({**row, **metrics(targets, blocks), **floor_sensitivity(targets)})
        common = target_data[row['pair_id_bg0']].merge(target_data[row['pair_id_bg1']], on='target_key', suffixes=('_bg0', '_bg1'), validate='one_to_one')
        target_rows.append({**row, **metrics(common), **floor_sensitivity(common), 'n_shared_archive_labels_between_swap_sets': len(set(first.archive) & set(second.archive))})
        if (index + 1) % 250 == 0:
            print('PROGRESS', index + 1, 'of', len(rectangles), 'seconds', round(time.perf_counter() - started, 1), flush=True)
    strict, target = (pd.DataFrame(strict_rows), pd.DataFrame(target_rows))
    for frame in [strict, target]:
        for column in ['mean_DD', 'SD_DD', 'RMS_DD', 'centered_profile_RMS_change', 'effect_correlation', 'fraction_opposite_effects_both_gt5pp', 'mean_shift_share_of_DD_square']:
            if column not in frame:
                frame[column] = np.nan
    save(strict, 'all_rectangle_focal_summaries.csv')
    save(target, 'target_common_support_sensitivity.csv')
    unique = unique_rectangle_summary(strict)
    save(unique, 'unique_four_recipe_rectangles.csv')
    save(unique_floor_summary(unique), 'floor_sensitivity_unique_rectangle_summary.csv')
    panels = [('strict_four_condition_acquisition', strict), ('target_common_only', target)]
    swap_scope, coverage = input_coverage(swaps, recipes, strict)
    save(coverage, 'input_swap_and_rectangle_coverage.csv')
    globals_ = pd.concat([summarize_groups(frame, [], label) for label, frame in panels], ignore_index=True)
    datasets = pd.concat([summarize_groups(frame, ['dataset_id'], label, swap_scope) for label, frame in panels], ignore_index=True)
    save(globals_, 'summary_global.csv')
    save(datasets, 'summary_by_dataset.csv')
    for columns, name, universe in [(['rxn_class'], 'summary_by_rxn_class.csv', swap_scope), (['focal_slot'], 'summary_by_focal_slot.csv', pd.DataFrame({'focal_slot': SLOTS})), (['context_slot'], 'summary_by_context_slot.csv', pd.DataFrame({'context_slot': SLOTS})), (['focal_swap_kind'], 'summary_by_swap_kind.csv', swap_scope), (['dataset_id', 'rxn_class', 'focal_slot', 'context_slot', 'focal_swap_kind', 'context_change_kind'], 'summary_by_dataset_class_slot_background.csv', None)]:
        save(pd.concat([summarize_groups(frame, columns, label, universe) for label, frame in panels], ignore_index=True), name)
    cases = select_cases(strict)
    save(cases, 'selected_cases.csv')
    case_blocks = []
    for case in cases.to_dict('records'):
        blocks = paired_blocks(pair_data[case['pair_id_bg0']], pair_data[case['pair_id_bg1']])
        for column in ['case_id', 'rectangle_id', 'representation_id', 'dataset_id', 'recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']:
            blocks[column] = case[column]
        case_blocks.append(blocks)
    case_blocks = pd.concat(case_blocks, ignore_index=True).merge(info, on=['dataset_id', 'target_key'], validate='many_to_one')
    save(case_blocks, 'selected_case_four_condition_blocks.csv.gz')
    chosen_recipes = cases.melt(id_vars=['case_id', 'rectangle_id', 'dataset_id'], value_vars=['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1'], var_name='corner', value_name='recipe_id')
    chosen_recipes['corner'] = chosen_recipes.corner.str.removeprefix('recipe_')
    save(chosen_recipes.merge(registry, on=['dataset_id', 'recipe_id'], how='left', validate='many_to_many'), 'selected_case_condition_registry.csv')
    sources = add_optional_source_ids(case_blocks, source_normalized)
    if len(sources):
        save(sources, 'selected_case_original_source_links.csv.gz')
    manifest = {**plan, 'elapsed_seconds': time.perf_counter() - started, 'python': sys.version, 'pandas': pd.__version__, 'numpy': np.__version__, 'plate_semantics': 'name0 prefix through -PL digits; archive/plate labels are provenance keys, not verified physical plates or independent replicates', 'analysis_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'strict_focal_with_any_support': int(strict.n_common_targets.gt(0).sum()), 'strict_focal_n_ge20': int(strict.n_common_targets.ge(20).sum()), 'strict_focal_n_ge50': int(strict.n_common_targets.ge(50).sum()), 'source_normalized_path': str(source_normalized) if source_normalized else None, 'selected_source_corner_links': len(sources), 'no_new_pvalues': True, 'units': '0-1 except names explicitly ending pp/pct; summaries/report convert to pp or percent', 'input_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in INPUTS}}
    if source_normalized:
        manifest['source_normalized_sha256'] = sha256(source_normalized.read_bytes()).hexdigest()
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(strict, target, unique, globals_, datasets, cases, manifest)
    print('GLOBAL', globals_.to_string(index=False), flush=True)
    print('COMPLETE', manifest, flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--source-normalized', type=Path, help='Optional authorized normalized source records for selected-case record IDs')
    args = parser.parse_args()
    main(args.plan_only, args.source_normalized)
