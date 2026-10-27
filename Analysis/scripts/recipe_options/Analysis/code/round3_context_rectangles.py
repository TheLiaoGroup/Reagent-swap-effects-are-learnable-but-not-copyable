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
    lines = ['# Round 3: background rectangles for single-component swaps across the full corpus', '', 'All results come from existing data; no new p-values, no claim of independent validation or general causal rules.', '', '## 1. Scope and deduplication', '', f"The input is {manifest['input_focal_swaps']} existing qualifying swaps, covering {manifest['input_datasets']} datasets and {manifest['input_reaction_classes']} merged reaction classes. Enumerating within the same dataset and exchange_id gives {manifest['unfiltered_same_exchange_background_pairs']} background pairs; keeping those that change only one other registered token slot leaves {len(strict)} focal representations, corresponding to {len(unique)} distinct four-recipe rectangles and covering {manifest['datasets_with_rectangles']} datasets and {manifest['reaction_classes_with_rectangles']} merged reaction classes. A rectangle can be represented along either component direction; the two axes of one rectangle are not independent replicates.", '', 'This round covers only swaps already in the original matched_swap_summary; that table requires at least 20 targets per background. Sparse swaps not in that table are not judged absent. All enumerated rectangles are kept, including those whose four conditions share no acquisition or have fewer than 20 common targets.', '', 'Direction is fixed by the registered token: the focal component 0→1 is identical in both backgrounds (including both doses), and background 0→1 is also fixed by token lexical order. DD=Δbackground1−Δbackground0. The main analysis requires all four cells to share dataset/archive/plate/acquisition/target; repeated acquisitions are then averaged within target. The target-only common-support sensitivity allows backgrounds to use different measurement blocks and is weaker evidence.', '', 'The plate field is only the name prefix of the raw name0 up to the -PL number; n_archive_plate_labels counts distinct labels of this kind within an archive. It is used for source location and four-cell matching, and does not represent verified physical plate counts, independent experiment counts or independent replicates. archive is likewise counted only by raw archive label.', '', '## 2. Full support and coverage sensitivity', '', '| Panel | Minimum common targets | Supported focal representations | Distinct four-recipe rectangles | Focal representations/rectangles with centered profile RMS≥5pp | Median profile RMS/pp | Median absolute mean DD/pp |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in global_summary.itertuples():
        lines.append(f'| {row.panel} | {row.minimum_common_targets} | {row.n_supported_focal_representations} | {row.n_supported_unique_rectangles} | {row.n_focal_centered_profile_RMS_ge5pp}/{row.n_unique_centered_profile_RMS_ge5pp} | {number(row.median_centered_profile_RMS_change_pp)} | {number(row.median_abs_mean_DD_pp)} |')
    floor_row = unique_floor_summary(unique).query('minimum_full_panel_common_targets == 20 and minimum_floor_panel_common_targets == 20').iloc[0]
    lines += ['', f'With unique rectangles as denominator: of the {int(floor_row.n_baseline_unique_rectangles)} strict n≥20 rectangles, {int(floor_row.n_retained_unique_rectangles)} still have n≥20 after requiring all four cell target-mean yields ≥5%; of these, {int(floor_row.n_retained_profile_RMS_ge5pp)} have profile RMS≥5pp, median {number(floor_row.median_retained_profile_RMS_pp)}pp. The same rectangles have median profile RMS on the full target set of {number(floor_row.median_full_profile_RMS_pp_in_same_rectangles)}pp. Rectangles not retained lack coverage under this floor rule and cannot be counted as floor-robust results.']
    lines += ['', 'The 5/10pp thresholds only describe magnitude; they are not significance or validated-discovery cutoffs. For 1-target rectangles the centered profile change is zero by construction and SD and correlation cannot be estimated; an empty correlation can also arise when all effects in one background are identical, and is not filled with zero. Correlations and opposite preferences for focal representations depend on the chosen axis; the unique-rectangle table keeps the range across both axes.', '', '## 3. Mean change and substrate-specific change', '', 'Each rectangle keeps two effect vectors on the same targets. Mean DD describes a shift of the overall effect; centered profile RMS describes the difference between (Δ1−meanΔ1) and (Δ0−meanΔ0), and SD(DD) is the sample-SD version of the same heterogeneity. RMS(DD)^2=mean(DD)^2+centered_profile_RMS^2; the script verifies this decomposition for each rectangle. A shift of the overall mean is not by itself a substrate-specific change.', '', 'Also saved: effect correlation, effect SD in each background, the number/fraction of targets whose effects in the two backgrounds are opposite and each >5pp, and four-cell means and yield-floor support. Centered changes can still come from measurement error, end-point floors and incompletely balanced substrate coverage; they describe how the observed response profile changes with background and do not independently establish a structural mechanism.', '', 'Each rectangle is also recomputed on the target subset with all four cell target-mean yields ≥5% and on the subset with at least one cell ≥10% in each background. The first uses the same targets for both axes; the second depends on the focal axis. Both filters use observed yields and change target composition, so they are not independent validation or measurement-error correction. Full values are in all_rectangle_focal_summaries.csv and target_common_support_sensitivity.csv; unique-rectangle coverage for the first is in floor_sensitivity_unique_rectangle_summary.csv.', '', '## 4. Coverage across datasets', '', '| dataset | Registered focal representations/unique rectangles | Strict n≥20 focal representations/unique rectangles | Profile RMS≥5pp focal representations/unique rectangles |', '|---|---:|---:|---:|']
    for row in dataset_summary.loc[dataset_summary.panel.eq('strict_four_condition_acquisition') & dataset_summary.minimum_common_targets.eq(20)].itertuples():
        lines.append(f'| {row.dataset_id} | {row.n_registered_focal_representations}/{row.n_registered_unique_rectangles} | {row.n_supported_focal_representations}/{row.n_supported_unique_rectangles} | {int(row.n_focal_centered_profile_RMS_ge5pp)}/{int(row.n_unique_centered_profile_RMS_ge5pp)} |')
    lines += ['', 'rxn_class merges Buchwald-Hartwig(CMC)→Buchwald-Hartwig and Suzuki Bpin→Suzuki by dataset mapping; the original registered name is kept in registered_reaction_type. The 20 input datasets map to 18 classes; the 17 datasets with rectangles map to 16 classes. Dataset counts are not reported as counts of independent reaction classes. The full grouped CSV by dataset/rxn_class/focal slot/background slot/swap kind includes low and zero support. Different groups can reuse the same rectangle and targets, so group counts cannot be summed into independent counts.', '', '## 5. Fully sourced examples', '', 'A small set of illustrations selected after full enumeration: high profile change per dataset, the largest mean-square-share reference, and one low-coverage case. No other results were removed. Four-cell yield order is background 0 component 0/1, background 1 component 0/1; percentages and pp are in separate columns.', '', '| Case | dataset | Focal swap [registered dose]; background swap [registered dose] | target/archive | Four-cell means/% | meanDD/pp | profile RMS/pp | Opposite >5pp targets |', '|---|---|---|---:|---|---:|---:|---:|']
    for row in cases.itertuples():
        ys = '/'.join((number(100 * getattr(row, metric)) for metric in ['mean_y0_bg0', 'mean_y1_bg0', 'mean_y0_bg1', 'mean_y1_bg1']))
        focal_doses = [getattr(row, f'component{i}_dose') for i in [0, 1]]
        focal_names = [f"{getattr(row, f'component{i}_name')}[{focal_doses[i]}]" if pd.notna(focal_doses[i]) and str(focal_doses[i]) else getattr(row, f'component{i}_name') for i in [0, 1]]
        context_tokens = [json.loads(getattr(row, f'context{i}_token')) for i in [0, 1]]
        context_names = [f"{getattr(row, f'context{i}_name')}[{context_tokens[i][1]}]" if len(context_tokens[i]) > 1 and context_tokens[i][1] else getattr(row, f'context{i}_name') for i in [0, 1]]
        lines.append(f"| {row.case_id} | {row.dataset_id} | {'→'.join(focal_names)} ({row.focal_swap_kind}); {row.context_slot}:{'→'.join(context_names)} ({row.context_change_kind}) | {row.n_common_targets}/{row.n_archives} | {ys} | {number(100 * row.mean_DD)} | {number(100 * row.centered_profile_RMS_change)} | {int(row.n_opposite_effects_both_gt5pp)} |")
    supported = strict.loc[strict.n_common_targets.ge(20)]
    max_mean_share = supported.mean_shift_share_of_DD_square.max()
    shift_dominant_count = supported.loc[supported.mean_shift_share_of_DD_square.ge(0.7), 'rectangle_id'].nunique()
    lines += ['', f'Among strict n≥20 rectangles the largest mean(DD)^2/RMS(DD)^2 is {number(100 * max_mean_share)}%; the number of unique rectangles with mean-square share ≥70% is {shift_dominant_count}. In this range large mean-square DD mainly comes from variation across targets; floors and measurement error still need checking.', '', '| Case | All four cells ≥5%: n / profile RMS pp | Best cell in each background ≥10%: n / profile RMS pp |', '|---|---:|---:|']
    for row in cases.itertuples():
        lines.append(f'| {row.case_id} | {int(row.all_four_ge5pct_n_common_targets)} / {number(100 * row.all_four_ge5pct_centered_profile_RMS_change)} | {int(row.each_background_best_ge10pct_n_common_targets)} / {number(100 * row.each_background_best_ge10pct_centered_profile_RMS_change)} |')
    source_description = f"In this run, selected_case_original_source_links.csv.gz contains {manifest['selected_source_corner_links']} four-cell source links; each original record_id is linked to the normalized records and the source yield verified." if manifest['selected_source_corner_links'] else 'No normalized records were supplied in this run; the four cells keep full location keys and no original record_id is invented.'
    lines += ['', 'All paired measurement blocks, four-cell yields, registered recipe/condition, and substrate and product SMILES for each case are in selected_case_four_condition_blocks.csv.gz; the registered sources of the four conditions by case and corner are in selected_case_condition_registry.csv.', source_description, '', '## 6. Files and reproduction', '', '- rectangle_registry.csv: four recipes, two-axis tokens and fixed background for all focal representations.', '- all_rectangle_focal_summaries.csv: statistics for all rectangles under strict same-acquisition matching; includes zero support.', '- target_common_support_sensitivity.csv: common-target sensitivity after averaging each background by target.', '- unique_four_recipe_rectangles.csv: deduplicated by four recipes; the two focal axes are not counted as replicates.', '- summary_global.csv, summary_by_dataset.csv, summary_by_rxn_class.csv, summary_by_dataset_class_slot_background.csv: full support and n≥20/50 strata.', '- input_swap_and_rectangle_coverage.csv: swap counts and rectangle coverage for every input dataset/registered reaction class/slot/swap kind; zero rows kept where there are no rectangles.', '- summary_by_focal_slot.csv, summary_by_context_slot.csv, summary_by_swap_kind.csv: single-axis and swap-kind coverage, including categories with no pairable rectangles.', '- selected_cases.csv and selected_case_*: a small set of fully sourced examples.', '- run_manifest.json: input hashes, counts, timing and optional source inputs.', '', 'Run code/round3_context_rectangles.py; --plan-only only estimates the scale. Effects and yields in numeric columns not marked pp/pct are on a 0–1 scale. identity_same_dose, identity_and_dose, addition_removal, dose_only and solvent_identity are kept separate; joint changes of component identity and dose are not rewritten as pure identity effects.', '', 'Conclusions stay within the observed range: whether the mean of a component effect and the shape of the substrate response change with the background of another component, and the size and coverage of that change; rectangle counts alone are not used to claim multiple independent chemical rules.']
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
