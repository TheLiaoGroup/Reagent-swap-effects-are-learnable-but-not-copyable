"""Describe how component backgrounds alter structural response contrasts.

Reads complete-acquisition rectangle target means prepared by round4 ingestion.
All features and both focal sides are retained, including unsupported contrasts.
The alternate condition axis is a reorientation of the same four measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round4' / 'structure_context'
INPUT = ROOT / 'round4' / 'rectangle_target_effects.csv.gz'
SEED = 20261018
CORNERS = ['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']
RECIPE_FIELDS = ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']
DD_COEFFICIENTS = np.array([1.0, -1.0, -1.0, 1.0])
PAIR_COEFFICIENTS = np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [1.0, -1.0, -1.0, 1.0]])
CONTRAST_NAMES = ['C_bg0', 'C_bg1', 'structural_DD']

def recipe_text(registry):
    recipes = {}
    for key, part in registry.groupby(['dataset_id', 'recipe_id'], sort=False):
        row = part.iloc[0]
        details = [str(row[f'Reagent{k}_Standardized Name']) + ' ' + str(row[f'Reagent{k}_HTE_Equiv']) + ' eq' for k in range(1, 7) if row[f'Reagent{k}_Standardized Name']]
        details += [str(row['Standardized Name']), str(row.temp_id) + ' C', str(row.time_id) + ' h']
        recipes[key] = '; '.join(details)
    return recipes

def contrast_values(positive, negative):
    differences = positive - negative
    values = PAIR_COEFFICIENTS @ differences
    assert abs(values[2] - values[1] + values[0]) < 1e-10
    return dict(zip(CONTRAST_NAMES, values))

def response_shape(c0, c1, threshold=0.05):
    if not np.isfinite(c0 + c1):
        return 'not_estimable'
    resolved0, resolved1 = (abs(c0) >= threshold, abs(c1) >= threshold)
    if resolved0 and resolved1 and (c0 * c1 < 0):
        return 'preference_switch'
    if resolved0 and resolved1:
        if abs(c1) - abs(c0) >= threshold:
            return 'preference_amplified'
        if abs(c0) - abs(c1) >= threshold:
            return 'preference_attenuated'
        return 'preference_preserved'
    if not resolved0 and resolved1:
        return 'preference_emerges'
    if resolved0 and (not resolved1):
        return 'preference_disappears'
    return 'both_below5pp'

def focal_equal_means(frame, feature):
    molecule = frame.groupby(['focal', feature], sort=False, observed=True)[CORNERS].mean().reset_index()
    counts = molecule[feature].value_counts().to_dict()
    if not (counts.get(0, 0) and counts.get(1, 0)):
        return ({}, counts)
    result = {}
    means = molecule.groupby(feature)[CORNERS].mean()
    for value, label in [(1, 'positive'), (0, 'negative')]:
        result.update({label + '_' + c: float(means.loc[value, c]) for c in CORNERS})
    result.update(contrast_values(means.loc[1].to_numpy(float), means.loc[0].to_numpy(float)))
    return (result, counts)

def controlled_comparison(frame, feature, seed_text):
    edge = frame.groupby(['partner', 'focal', feature], sort=False, observed=True)[CORNERS].mean().reset_index()
    grouped = edge.groupby(['partner', feature], observed=True)[CORNERS].mean()
    positive = grouped.loc[grouped.index.get_level_values(1) == 1].droplevel(1)
    negative = grouped.loc[grouped.index.get_level_values(1) == 0].droplevel(1)
    shared = sorted(set(positive.index) & set(negative.index) - {''})
    output = {'n_common_partners': len(shared)}
    if not shared:
        return (output, pd.DataFrame())
    positive, negative = (positive.loc[shared], negative.loc[shared])
    common_edges = edge[edge.partner.isin(shared)]
    output['n_positive_molecules_common'] = int(common_edges.loc[common_edges[feature].eq(1), 'focal'].nunique())
    output['n_negative_molecules_common'] = int(common_edges.loc[common_edges[feature].eq(0), 'focal'].nunique())
    common_targets = frame[frame.partner.isin(shared)]
    output['n_common_targets'] = len(common_targets)
    for value, label, group in [(1, 'positive', positive), (0, 'negative', negative)]:
        output.update({label + '_' + c: float(group[c].mean()) for c in CORNERS})
        edges_per_partner = common_edges[common_edges[feature].eq(value)].groupby('partner').focal.nunique()
        output['minimum_' + label + '_molecules_per_partner'] = int(edges_per_partner.min())
        output['median_' + label + '_molecules_per_partner'] = float(edges_per_partner.median())
        zeros = common_targets[common_targets[feature].eq(value)].copy()
        zeros[CORNERS] = zeros[CORNERS].eq(0).astype(float)
        proportions = zeros.groupby(['partner', 'focal'])[CORNERS].mean().groupby('partner').mean().mean()
        output.update({label + '_zero_fraction_' + c: float(proportions[c]) for c in CORNERS})
    output.update(contrast_values(positive.mean().to_numpy(float), negative.mean().to_numpy(float)))
    by_partner = pd.DataFrame({'partner': shared})
    for name, vector in zip(CONTRAST_NAMES, PAIR_COEFFICIENTS):
        by_partner[name] = (positive.to_numpy(float) - negative.to_numpy(float)) @ vector
    output['partner_SD_structural_DD'] = float(by_partner.structural_DD.std(ddof=0))
    output['fraction_partners_T_positive'] = float(by_partner.structural_DD.gt(0).mean())
    output['fraction_partners_T_negative'] = float(by_partner.structural_DD.lt(0).mean())
    if len(shared) >= 6:
        seed = int(hashlib.sha256((str(SEED) + '|' + seed_text).encode()).hexdigest()[:8], 16)
        partition = np.random.default_rng(seed).permutation(len(shared))
        for half, indices in enumerate(np.array_split(partition, 2), 1):
            output[f'half{half}_n_partners'] = len(indices)
            for name in CONTRAST_NAMES:
                output[f'half{half}_' + name] = float(by_partner.iloc[indices][name].mean())
        t1, t2 = (output['half1_structural_DD'], output['half2_structural_DD'])
        output['partner_halves_same_T_sign'] = bool(t1 * t2 > 0)
        output['partner_halves_min_abs_T'] = min(abs(t1), abs(t2))
    return (output, by_partner)

def analyze_axis(targets, metadata, features, feature_names):
    output = []
    for role in ['sub_1_smiles', 'sub_2_smiles']:
        opposite = 'sub_2_smiles' if role == 'sub_1_smiles' else 'sub_1_smiles'
        frame = targets.merge(features[['smiles', 'canonical_smiles'] + feature_names], left_on=role, right_on='smiles', how='left', validate='many_to_one')
        frame['focal'] = frame.canonical_smiles.fillna('')
        canonical = features.set_index('smiles').canonical_smiles.to_dict()
        frame['partner'] = frame[opposite].map(canonical).fillna(frame[opposite]).fillna('')
        valid = frame.focal.ne('') & frame[feature_names].notna().all(axis=1)
        frame = frame.loc[valid].copy()
        one_substrate = bool(targets[opposite].eq('').all())
        for feature in feature_names:
            row = {'rectangle_id': metadata.rectangle_id, 'representation_id': metadata.representation_id, 'dataset_id': metadata.dataset_id, 'rxn_class': metadata.rxn_class, 'focal_slot': metadata.focal_slot, 'context_slot': metadata.context_slot, 'is_representative_axis': metadata.is_representative_axis, 'role': role, 'feature': feature, 'n_all_targets': len(targets), 'n_valid_targets': len(frame), 'primary_rectangle_support': len(targets) >= 20, 'primary_weighting': 'feature_group_molecule_equal_within_common_partner_then_partner_equal', 'one_substrate_without_recorded_partner': one_substrate}
            whole, molecular_counts = focal_equal_means(frame, feature)
            row.update({'n_positive_molecules': int(molecular_counts.get(1, 0)), 'n_negative_molecules': int(molecular_counts.get(0, 0)), **{'molecule_equal_' + k: v for k, v in whole.items()}})
            if frame.empty:
                row.update(status='no_valid_focal_molecules', n_common_partners=0)
            elif not whole:
                row.update(status='feature_constant_in_observed_focal_molecules', n_common_partners=0)
            elif one_substrate:
                row.update(status='single_substrate_molecule_equal_only', n_common_partners=0)
            else:
                controlled, _ = controlled_comparison(frame, feature, metadata.representation_id + '|' + role + '|' + feature)
                row.update({'controlled_' + k: v for k, v in controlled.items()})
                row['n_common_partners'] = controlled['n_common_partners']
                row['status'] = 'shared_partner_estimable' if controlled['n_common_partners'] else 'no_common_partner'
            row['clear_display_support'] = bool(row.get('controlled_n_positive_molecules_common', 0) >= 3 and row.get('controlled_n_negative_molecules_common', 0) >= 3 and (row['n_common_partners'] >= 3))
            row['partner_half_split_available'] = row['n_common_partners'] >= 6
            row['controlled_response_shape_5pp'] = response_shape(row.get('controlled_C_bg0', np.nan), row.get('controlled_C_bg1', np.nan))
            output.append(row)
    return output

def orient_axes(targets, registry):
    representation = targets.representation_id.iloc[0]
    reference = registry.loc[registry.representation_id.eq(representation)].iloc[0]
    recipes = [reference[c] for c in RECIPE_FIELDS]
    assert len(set(recipes)) == 4
    old_yields = targets[CORNERS].to_numpy(float)
    assert np.allclose(old_yields @ DD_COEFFICIENTS, targets.DD, atol=1e-12)
    for _, axis in registry[registry.rectangle_id.eq(reference.rectangle_id)].iterrows():
        indices = [recipes.index(axis[c]) for c in RECIPE_FIELDS]
        oriented = targets.copy()
        oriented[CORNERS] = old_yields[:, indices]
        oriented['delta_bg0'] = oriented.y1_bg0 - oriented.y0_bg0
        oriented['delta_bg1'] = oriented.y1_bg1 - oriented.y0_bg1
        oriented['DD'] = oriented.delta_bg1 - oriented.delta_bg0
        inverse_weights = np.zeros(4)
        inverse_weights[indices] = DD_COEFFICIENTS
        sign = 1 if np.array_equal(inverse_weights, DD_COEFFICIENTS) else -1
        assert np.array_equal(inverse_weights, sign * DD_COEFFICIENTS)
        assert np.allclose(oriented.DD, sign * targets.DD, atol=1e-12)
        axis = axis.copy()
        axis['is_representative_axis'] = axis.representation_id == representation
        axis['DD_sign_vs_representative'] = sign
        yield (oriented, axis)

def enrich_axis_registry(registry, condition_registry):
    recipes = recipe_text(condition_registry)
    rows = []
    for row in registry.to_dict('records'):
        for field in RECIPE_FIELDS:
            row['full_' + field] = recipes[row['dataset_id'], row[field]]
        rows.append(row)
    return pd.DataFrame(rows)

def select_cases(scan, registry, targets_by_id, features):
    candidate = scan[scan.primary_rectangle_support & scan.clear_display_support & scan.partner_half_split_available].copy()
    candidate = candidate[candidate.controlled_partner_halves_same_T_sign.fillna(False) & candidate.controlled_partner_halves_min_abs_T.ge(0.05) & candidate.controlled_structural_DD.abs().ge(0.1)]
    candidate['switch_priority'] = candidate.controlled_response_shape_5pp.eq('preference_switch')
    candidate['rank_score'] = candidate.controlled_partner_halves_min_abs_T
    candidate = candidate.sort_values(['switch_priority', 'rank_score', 'n_common_partners'], ascending=False)
    candidate.head(100).to_csv(OUT / 'ranked_chemical_candidates.csv', index=False, encoding='utf-8-sig')
    selected = candidate.drop_duplicates('dataset_id').head(10).copy()
    selected['case_selection_role'] = 'dataset_first_pass_illustration'
    anchors = scan[scan.rectangle_id.eq('q_a2154d15988dab9b') & scan.role.eq('sub_1_smiles') & scan.feature.eq('NH_heteroaromatic') | scan.rectangle_id.eq('q_215223222cc3f61d') & scan.role.eq('sub_1_smiles') & scan.feature.isin(['aniline', 'tertiary_amine'])].copy()
    anchors['case_selection_role'] = 'alignment_to_separate_transport_prediction'
    selected = pd.concat([selected, anchors], ignore_index=True).drop_duplicates(['rectangle_id', 'representation_id', 'role', 'feature'])
    selected = selected.merge(registry, on=['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class', 'focal_slot', 'context_slot', 'is_representative_axis'], suffixes=('', '_registry'), validate='many_to_one')
    selected.to_csv(OUT / 'selected_structure_context_cases.csv', index=False, encoding='utf-8-sig')
    target_rows, partner_rows = ([], [])
    for case in selected.itertuples():
        stored = targets_by_id[case.rectangle_id]
        axis = registry[registry.representation_id.eq(case.representation_id)].iloc[0]
        reference = registry[registry.representation_id.eq(stored.representation_id.iloc[0])].iloc[0]
        indices = [[reference[c] for c in RECIPE_FIELDS].index(axis[c]) for c in RECIPE_FIELDS]
        part = stored.copy()
        part[CORNERS] = stored[CORNERS].to_numpy(float)[:, indices]
        part['delta_bg0'] = part.y1_bg0 - part.y0_bg0
        part['delta_bg1'] = part.y1_bg1 - part.y0_bg1
        part['DD'] = part.delta_bg1 - part.delta_bg0
        role, feature = (case.role, case.feature)
        opposite = 'sub_2_smiles' if role == 'sub_1_smiles' else 'sub_1_smiles'
        part = part.merge(features[['smiles', 'canonical_smiles', feature]], left_on=role, right_on='smiles', how='left', validate='many_to_one')
        part['focal'] = part.canonical_smiles.fillna('')
        canonical = features.set_index('smiles').canonical_smiles.to_dict()
        part['partner'] = part[opposite].map(canonical).fillna(part[opposite])
        _, by_partner = controlled_comparison(part, feature, case.representation_id + '|' + role + '|' + feature)
        by_partner['rectangle_id'] = case.rectangle_id
        by_partner['representation_id'] = case.representation_id
        by_partner['role'] = role
        by_partner['feature'] = feature
        partner_rows.append(by_partner)
        part = part[part.partner.isin(by_partner.partner)].copy()
        part['representation_id'] = case.representation_id
        part['role'] = role
        part['feature'] = feature
        part['feature_present'] = part[feature]
        for c in RECIPE_FIELDS:
            part[c] = axis[c]
        target_rows.append(part.drop(columns=[feature, 'smiles', 'canonical_smiles']))
    if target_rows:
        pd.concat(target_rows, ignore_index=True).to_csv(OUT / 'selected_case_target_four_yields.csv.gz', index=False, encoding='utf-8-sig')
        pd.concat(partner_rows, ignore_index=True).to_csv(OUT / 'selected_case_partner_contrasts.csv', index=False, encoding='utf-8-sig')
    return selected

def rectangle_evidence_counts(scan):
    main = scan[scan.is_representative_axis & scan.primary_rectangle_support].copy()
    rectangles = main[['rectangle_id', 'dataset_id', 'rxn_class']].drop_duplicates().set_index('rectangle_id')
    assert len(rectangles) == 825
    common = main.status.eq('shared_partner_estimable') & main.n_common_partners.gt(0)
    stages = {'any_common_partner': common, 'clear_support': main.clear_display_support, 'clear_support_and_half_split': main.clear_display_support & main.partner_half_split_available, 'clear_support_and_halves_same_sign': main.clear_display_support & main.partner_half_split_available & main.controlled_partner_halves_same_T_sign.fillna(False)}
    for label, flag in stages.items():
        rectangles[label] = flag.groupby(main.rectangle_id).any().reindex(rectangles.index, fill_value=False)
    denominator = rectangles.any_common_partner
    records, class_records = ([], [])
    for stage, support in stages.items():
        for metric, thresholds in [('absolute_structural_DD', [0.05, 0.1, 0.2]), ('preference_switch', [0.05, 0.1])]:
            for threshold in thresholds:
                if metric == 'absolute_structural_DD':
                    signal = main.controlled_structural_DD.abs().ge(threshold)
                else:
                    signal = main.controlled_C_bg0.abs().ge(threshold) & main.controlled_C_bg1.abs().ge(threshold) & (main.controlled_C_bg0 * main.controlled_C_bg1).lt(0)
                observed = (support & signal).groupby(main.rectangle_id).any().reindex(rectangles.index, fill_value=False)
                field = stage + '__' + metric + '__' + str(round(threshold * 100)) + 'pp'
                rectangles[field] = observed
                by_class = rectangles.assign(observed=observed).groupby('rxn_class').agg(n_primary_rectangles=('observed', 'size'), n_with_signal=('observed', 'sum'), n_common_support=('any_common_partner', 'sum'))
                by_class['fraction_all_primary'] = by_class.n_with_signal / by_class.n_primary_rectangles
                by_class['fraction_common_support'] = by_class.n_with_signal / by_class.n_common_support.replace(0, np.nan)
                base = {'support_stage': stage, 'signal': metric, 'threshold': threshold, 'n_primary_rectangles': len(rectangles), 'n_common_support_rectangles': int(denominator.sum()), 'n_observed_signal_rectangles': int(observed.sum()), 'fraction_all_primary': float(observed.mean()), 'fraction_common_support': float(observed[denominator].mean()), 'reaction_class_equal_fraction_all_primary': float(by_class.fraction_all_primary.mean()), 'reaction_class_equal_fraction_common_support': float(by_class.fraction_common_support.mean()), 'n_all_primary_classes': len(by_class), 'n_common_support_classes': int(by_class.n_common_support.gt(0).sum())}
                records.append(base)
                class_records.extend(({'support_stage': stage, 'signal': metric, 'threshold': threshold, 'rxn_class': key, **row} for key, row in by_class.to_dict('index').items()))
    rectangles.reset_index().to_csv(OUT / 'unique_rectangle_structure_evidence.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(OUT / 'unique_rectangle_threshold_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(class_records).to_csv(OUT / 'unique_rectangle_threshold_by_class.csv', index=False, encoding='utf-8-sig')
    return {'primary_rectangle_denominator': len(rectangles), 'common_partner_comparison_rectangles': int(denominator.sum()), 'clear_support_rectangles': int(rectangles.clear_support.sum()), 'clear_support_and_half_split_rectangles': int(rectangles.clear_support_and_half_split.sum()), 'clear_support_and_halves_same_sign_rectangles': int(rectangles.clear_support_and_halves_same_sign.sum()), 'interpretation': 'Any qualifying feature/side within each representative rectangle; correlated post hoc feature comparisons, not independent discoveries.', 'threshold_results': records}

def original_source_links(selected, targets_by_id):
    normalized = pd.read_csv(ROOT / 'data/normalized_records.csv.gz', keep_default_na=False, low_memory=False)
    normalized = normalized[normalized.strict_eligible.astype(str).str.lower().eq('true')].copy()
    normalized['yield_numeric'] = pd.to_numeric(normalized.yield_numeric, errors='raise')
    block_keys = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
    block_rows, source_rows = ([], [])
    for case in selected.drop_duplicates(['rectangle_id', 'representation_id']).itertuples():
        recipes = [getattr(case, c) for c in RECIPE_FIELDS]
        target_ids = set(targets_by_id[case.rectangle_id].target_key)
        frame = normalized[normalized.dataset_id.eq(case.dataset_id) & normalized.recipe_id.isin(recipes) & normalized.target_key.isin(target_ids)]
        cells = frame.groupby(block_keys + ['recipe_id'], as_index=False).yield_numeric.mean()
        complete = None
        for recipe, corner in zip(recipes, CORNERS):
            cell = cells[cells.recipe_id.eq(recipe)].drop(columns='recipe_id').rename(columns={'yield_numeric': corner})
            complete = cell if complete is None else complete.merge(cell, on=block_keys, how='inner', validate='one_to_one')
        expected = targets_by_id[case.rectangle_id].set_index('target_key')[CORNERS]
        registry_reference = pd.read_csv(ROOT / 'round4/context_transport/rectangle_input_coverage.csv', keep_default_na=False)
        registry_reference = registry_reference[registry_reference.rectangle_id.eq(case.rectangle_id)].iloc[0]
        original_order = [registry_reference[c] for c in RECIPE_FIELDS]
        indices = [original_order.index(recipe) for recipe in recipes]
        expected = expected.iloc[:, indices].copy()
        expected.columns = CORNERS
        reconstructed = complete.groupby('target_key')[CORNERS].mean().reindex(expected.index)
        assert reconstructed.notna().all().all() and np.allclose(reconstructed, expected, atol=1e-12)
        complete['rectangle_id'], complete['representation_id'] = (case.rectangle_id, case.representation_id)
        block_rows.append(complete)
        for recipe, corner in zip(recipes, CORNERS):
            source = complete[block_keys + ['rectangle_id', 'representation_id', corner]].rename(columns={corner: 'matched_cell_yield'})
            source['recipe_id'] = recipe
            source['corner'] = corner
            linked = source.merge(frame, on=block_keys + ['recipe_id'], how='left', validate='one_to_many')
            assert linked.record_id.notna().all()
            source_rows.append(linked[['rectangle_id', 'representation_id', 'corner', *block_keys, 'recipe_id', 'record_id', 'condition_id', 'yield_numeric', 'matched_cell_yield', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']])
    if not source_rows:
        return {'selected_original_record_links': 0, 'selected_complete_source_blocks': 0}
    source = pd.concat(source_rows, ignore_index=True)
    raw = pd.read_csv(ROOT / 'data/raw_reactions.csv.gz', keep_default_na=False, low_memory=False, usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    source = source.merge(raw, on='record_id', how='left', validate='many_to_one')
    assert source.source_file.notna().all()
    source.to_csv(OUT / 'selected_case_original_record_links.csv.gz', index=False, encoding='utf-8-sig')
    blocks = pd.concat(block_rows, ignore_index=True)
    blocks.to_csv(OUT / 'selected_case_complete_acquisition_blocks.csv.gz', index=False, encoding='utf-8-sig')
    return {'selected_original_record_links': len(source), 'selected_complete_source_blocks': len(blocks)}

def write_report(scan, selected, coverage, summary):

    def component_text(name, dose):
        return str(name) + (f' ({dose} eq)' if str(dose) else '')
    named = selected.set_index(['representation_id', 'role', 'feature'], verify_integrity=True)
    snar = named.loc['f_7db35345dd86fb98', 'sub_1_smiles', 'NH_heteroaromatic']
    buchwald = named.loc['f_3894a2ad1416dd6a', 'sub_2_smiles', 'aryl_F']
    suzuki = named.loc['f_869a100c821b92ce', 'sub_1_smiles', 'aryl_Cl']
    aniline = named.loc['f_030b396694c5d66f', 'sub_1_smiles', 'aniline']
    tertiary = named.loc['f_030b396694c5d66f', 'sub_1_smiles', 'tertiary_amine']
    suzuki_yields = suzuki[['controlled_' + group + '_' + corner for group in ['positive', 'negative'] for corner in CORNERS]]
    primary_count = summary['primary_rectangles_ge20']
    primary_classes = scan.loc[scan.primary_rectangle_support, 'rxn_class'].nunique()
    lines = ['# 第四轮：背景组分如何改写组件交换的结构偏好', '', '日期：2026-10-16。现有完整四格记录的探索性结构分析。', '', '## 科学对象', '', '同一个组件交换在两种背景中可能只发生整体产率平移，也可能改变哪些结构受益。本分析比较结构正/负两组的组件效应差：C0=E(Δbg0|feature+)−E(Δbg0|feature−)，C1同理；结构背景对比T=C1−C0=E(DD|feature+)−E(DD|feature−)。T接近零时，背景主要保留这个已定义结构对比；T较大时，背景改变其幅度或符号。它不要求组件总体平均效应很大。', '', f'具体化学判断跨越多类反应：SNAr中，DMSO→t-AmOH对NH杂芳环底物的相对响应在KHCO3与DIPEA背景中从{100 * snar.controlled_C_bg0:+.2f}变为{100 * snar.controlled_C_bg1:+.2f} pp；Buchwald–Hartwig中，BINAP Pd G3→XPhos Pd G3对aryl-F标记的相对响应随Cs2CO3→NaOtBu从{100 * buchwald.controlled_C_bg0:+.2f}变为{100 * buchwald.controlled_C_bg1:+.2f} pp。Suzuki Bpin的aryl-Cl例子则表现为溶剂交换偏好被Na2CO3背景移除（{100 * suzuki.controlled_C_bg0:+.2f}→{100 * suzuki.controlled_C_bg1:+.2f} pp），四角组均值为{100 * suzuki_yields.min():.0f}–{100 * suzuki_yields.max():.0f}%，说明这种模式不只来自接近零产率的地板。它们指向不同的结构×组件×背景关系，不推断共享机理。', '', f'与Acylation主预测矩形对齐时，覆盖更广的aniline对照为{int(aniline.controlled_n_positive_molecules_common)}/{int(aniline.controlled_n_negative_molecules_common)}个分子、{int(aniline.n_common_partners)}个共同酸伙伴：TEA→DIPEA的相对响应在DCC与EDC-HCl背景中为{100 * aniline.controlled_C_bg0:+.2f}→{100 * aniline.controlled_C_bg1:+.3f} pp（T={100 * aniline.controlled_structural_DD:+.2f} pp），两个伙伴半集T为{100 * aniline.controlled_half1_structural_DD:+.2f}/{100 * aniline.controlled_half2_structural_DD:+.2f} pp。tertiary-amine对照的{100 * tertiary.controlled_structural_DD:+.2f} pp背景变化保留为{int(tertiary.controlled_n_positive_molecules_common)}个阳性分子的子群；两项来自同一矩形和共享记录，不能相加当作独立复现。', '', '主估计控制实际共有伙伴：在每一个同时覆盖两组的另一侧partner内，两组焦点分子分别等权，焦点分子下的target先平均，再对共同partner等权。这样组别的伙伴组成有共同支持。另保留直接焦点分子等权结果，用来查看采样覆盖对结论的影响。描述符仍可能与其他结构相关，控制partner不把结构类别变成随机化干预。', '', '## 范围与支持', '', f"全注册{summary['registered_rectangles']}唯一四配方矩形，其中{summary['unique_rectangles']}有完整支持、{primary_count}达到至少20个共同target的主体范围；输入为{summary['target_rectangle_means']:,}目标×矩形均值。默认只扫描代表条件轴；另轴可由同一四recipe角位重排，逐项断言DD与代表轴相同或仅差方向符号，不构成第二次实验。全扫描包括{summary['axis_representations']}轴表示×两底物侧×{len(summary['features'])}个既有二元特征，共{len(scan):,}行，常数特征、无共同partner和缺乏有效分子均保留状态。", '', '| 数据集 | 全库交换 | 注册四格矩形 | 主体≥20 target | 可估计四格目标均值 |', '|---|---:|---:|---:|---:|']
    for row in coverage.itertuples():
        lines.append(f'| {row.dataset_id} | {row.n_swaps} | {row.n_rectangles} | {row.n_primary_rectangles} | {row.n_rectangle_target_means:,} |')
    evidence = summary['rectangle_evidence']
    lines += ['', '表保留全部20数据集/18反应类；无矩形只表示当前完整记录和匹配规则下无本轮估计对象。Finkelstein的一底物表示只做分子等权，不构造真实共同partner。', '', f"按唯一矩形去重：主体{primary_count}个中，{evidence['common_partner_comparison_rectangles']}个至少有一个可共同partner比较的结构特征/焦点侧，{evidence['clear_support_rectangles']}个至少有一个清楚支持对比，{evidence['clear_support_and_half_split_rectangles']}个另有≥6伙伴可分半。其余仍留在{primary_count}分母中，不把不可估计项称为化学无效应。", '', f'| 唯一矩形中的观测信号 | 共同partner即可 | 清楚支持 | 清楚支持且两伙伴半集T同号 | 同号版本/{primary_count} | 同号版本反应类等权 |', '|---|---:|---:|---:|---:|---:|']
    threshold_table = pd.DataFrame(evidence['threshold_results'])
    for metric, threshold, label in [('absolute_structural_DD', 0.05, 'T绝对值≥5pp'), ('absolute_structural_DD', 0.1, 'T绝对值≥10pp'), ('absolute_structural_DD', 0.2, 'T绝对值≥20pp'), ('preference_switch', 0.05, 'C0/C1反号且各≥5pp'), ('preference_switch', 0.1, 'C0/C1反号且各≥10pp')]:
        rows = threshold_table[threshold_table.signal.eq(metric) & threshold_table.threshold.eq(threshold)].set_index('support_stage')
        strict = rows.loc['clear_support_and_halves_same_sign']
        lines.append(f"| {label} | {int(rows.loc['any_common_partner', 'n_observed_signal_rectangles'])} | {int(rows.loc['clear_support', 'n_observed_signal_rectangles'])} | {int(strict.n_observed_signal_rectangles)} | {100 * strict.fraction_all_primary:.1f}% | {100 * strict.reaction_class_equal_fraction_all_primary:.1f}% |")
    lines += ['', f'每格计数只把同一rectangle算一次，即使多个相关特征或两底物侧符合。反应类等权使用{primary_count}矩形所覆盖的{primary_classes}类，类内矩形等权；Finkelstein无真实共同partner，故这里的比率是当前记录获得可展示支持的比例，不是所有反应具有此机制的发生率。共同支持条件分母及逐类数值另存阈值总表。', '', '清楚展示标签要求共同支持中两组各≥3个焦点分子且≥3个共同partner。对≥6个共同partner，用固定seed随机分成两个不重叠伙伴半集，分别报告C0、C1与T的方向和量级。这是共享焦点分子的伙伴敏感性检查，不是独立实验、结构复制或正式p值。全部连续估计和未支持项都在总表中。四角零产率比例按同一层级权重保留；零不替换，未测不填零。', '', '## 跨反应化学实例', '', f"候选仅用于展示：共同支持标签成立、两伙伴半集T同号且各≥5pp、整体|T|≥10pp；优先C0/C1在±5pp之外换号，再按半集支持的幅度排序，每数据集先取一个，再加入主预测矩形的具名对照。{len(summary['features'])}特征、两个条件轴和共享矩形不是独立发现数。完整排名保留，科学解释须回到真实组件、剂量、四角实际产率和结构。", '', '| 数据集/焦点侧/结构 | 组件交换 | 背景交换 | 组+/组−分子；伙伴 | C0→C1/pp | T/pp | 两伙伴半集T/pp |', '|---|---|---|---:|---:|---:|---:|']
    for case in selected.itertuples():
        context0, context1 = (json.loads(case.context0_token), json.loads(case.context1_token))
        context0_dose = context0[1] if len(context0) > 1 else ''
        context1_dose = context1[1] if len(context1) > 1 else ''
        lines.append(f'| {case.dataset_id}/{case.role}/{case.feature} | {component_text(case.component0_name, case.component0_dose)}→{component_text(case.component1_name, case.component1_dose)} | {component_text(case.context0_name, context0_dose)}→{component_text(case.context1_name, context1_dose)} | {int(case.controlled_n_positive_molecules_common)}/{int(case.controlled_n_negative_molecules_common)}; {case.n_common_partners} | {100 * case.controlled_C_bg0:+.1f}→{100 * case.controlled_C_bg1:+.1f} | {100 * case.controlled_structural_DD:+.1f} | {100 * case.controlled_half1_structural_DD:+.1f}/{100 * case.controlled_half2_structural_DD:+.1f} |')
    lines += ['', '四格配方与结构组的实际产率：', '']
    for case in selected.itertuples():
        lines += [f'### {case.dataset_id}：{case.feature}', '', f'矩形`{case.rectangle_id}`，轴`{case.representation_id}`，{case.controlled_response_shape_5pp}。']
        for field in RECIPE_FIELDS:
            lines.append(f"- `{field}`：{getattr(case, 'full_' + field)}。")
        lines += ['', '| 结构组 | Y0/bg0 | Y1/bg0 | Y0/bg1 | Y1/bg1 |', '|---|---:|---:|---:|---:|']
        for label, meaning in [('positive', '含标记'), ('negative', '无标记')]:
            lines.append('| ' + meaning + ' | ' + ' | '.join((f"{100 * getattr(case, 'controlled_' + label + '_' + c):.2f}%" for c in CORNERS)) + ' |')
        lines += ['', '表中产率使用共同partner控制的同一支持，不能把背景共同提高两组产率写成结构偏好改变。逐target的底物/产物SMILES、实际四角产率及伙伴见具名目标表；每partner独立对比见伙伴表。', '']
    lines += ['## 证据如何组织', '', '最有内容的结果是一个具体化学判断：改变哪个背景组分，会保留、增强、移除或反转哪个结构群体对组件交换的响应。高T并不等于共同机理；结构标记可能粗略、组别覆盖不同且共享分子，需把伙伴半集、分子等权和零地板一起读。两轴只是同一四格的两种问题：沿一个轴问某组件偏好是否被背景改变，沿另一个轴交换主语；数学DD对称不使两种叙事成为独立复现。', '', '## 输出与复算', '', '代码`Analysis/code/round4_structure_context.py`。输入由round4统一摄取提供，不重新构建round3模型。', '', '- `all_structure_context_contrasts.csv.gz`：全部矩形×轴×焦点侧×特征，含完整支持、两种权重、四角均值、零地板、伙伴半集及响应形态。', '- `condition_axis_registry.csv`：本次代表轴的角位映射、DD方向、实际名称/剂量/四格配方；原始双轴注册见`round3/context_rectangles/rectangle_registry.csv`。', '- `dataset_coverage.csv`、`structure_context_summary.json`：20数据集/18类覆盖和计数。', '- `ranked_chemical_candidates.csv`、`selected_structure_context_cases.csv`：探索性化学候选与首轮具名展示。', '- `selected_case_target_four_yields.csv.gz`、`selected_case_partner_contrasts.csv`：具名例子的完整四格目标与伙伴对比。', '']
    lines += [f"{len(selected)}个具名结构对比来自{selected.rectangle_id.nunique()}个唯一矩形。源引用按矩形和代表轴去重，已重建{summary.get('selected_complete_source_blocks', 0):,}个完整acquisition四格块、{summary.get('selected_original_record_links', 0):,}个原始record_id链接；每target的四角均值与统一输入逐项核对。`selected_case_original_record_links.csv.gz`提供record_id、源文件/表/行、原始产率、recipe/condition、底物/产物SMILES；`selected_case_complete_acquisition_blocks.csv.gz`保留真正完整的四格匹配块。archive/plate标签不自动等于独立实验板。", '']
    (OUT / 'STRUCTURE_CONTEXT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def self_test():
    yes, no = (np.array([0.1, 0.4, 0.3, 0.6]), np.array([0.2, 0.3, 0.4, 0.5]))
    assert np.allclose(list(contrast_values(yes, no).values()), [0.2, 0.2, 0])
    yes[3] = 0.1
    assert np.allclose(list(contrast_values(yes, no).values()), [0.2, -0.3, -0.5])
    rows = []
    for partner in range(8):
        for group in [0, 1]:
            for molecule in range(3):
                values = np.array([0.1, 0.2 + group * 0.2, 0.2, 0.3 - group * 0.3])
                rows.append({'partner': str(partner), 'focal': str(group) + ':' + str(molecule), 'f': group, **dict(zip(CORNERS, values))})
    result, _ = controlled_comparison(pd.DataFrame(rows), 'f', 'test')
    assert np.isclose(result['C_bg0'], 0.2) and np.isclose(result['C_bg1'], -0.3)
    assert np.isclose(result['structural_DD'], -0.5) and result['partner_halves_same_T_sign']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--include-alternate-axes', action='store_true')
    parser.add_argument('--summarize-existing', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('Structural-context estimand checks passed.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.input, keep_default_na=False, low_memory=False)
    assert not targets.duplicated(['rectangle_id', 'target_key']).any()
    registry = pd.read_csv(ROOT / 'round3/context_rectangles/rectangle_registry.csv', keep_default_na=False, low_memory=False)
    input_coverage = pd.read_csv(ROOT / 'round4/context_transport/rectangle_input_coverage.csv', keep_default_na=False, low_memory=False)
    dictionary = pd.read_csv(ROOT / 'data/feature_dictionary.csv', low_memory=False)
    names = dictionary[dictionary.feature_type.eq('binary')].feature.tolist()
    assert len(names) == 25
    features = pd.read_csv(ROOT / 'data/molecular_features.csv', keep_default_na=False, low_memory=False)
    features[names] = features[names].apply(pd.to_numeric, errors='coerce')
    features = features[features.parsing_status.eq('valid')].copy()
    for column in CORNERS + ['delta_bg0', 'delta_bg1', 'DD']:
        targets[column] = pd.to_numeric(targets[column], errors='raise')
    assert np.allclose(targets.delta_bg1 - targets.delta_bg0, targets.DD, atol=1e-12)
    rows, axis_rows = ([], [])
    targets_by_id = {k: v for k, v in targets.groupby('rectangle_id', sort=True)}
    scan_inputs = [] if args.summarize_existing else list(input_coverage.sort_values('rectangle_id').itertuples())
    for number, reference_row in enumerate(scan_inputs, 1):
        rectangle_id = reference_row.rectangle_id
        part = targets_by_id.get(rectangle_id, targets.iloc[:0].copy())
        targets_by_id[rectangle_id] = part.copy()
        assert part.empty or part.representation_id.nunique() == 1
        if part.empty:
            axis = registry.loc[registry.representation_id.eq(reference_row.representation_id)].iloc[0].copy()
            axis['is_representative_axis'], axis['DD_sign_vs_representative'] = (True, 1)
            orientations = [(part, axis)]
        else:
            orientations = orient_axes(part, registry)
        for oriented, axis in orientations:
            if not args.include_alternate_axes and (not axis.is_representative_axis):
                continue
            axis_rows.append(axis)
            rows.extend(analyze_axis(oriented, axis, features, names))
        if number % 100 == 0:
            print(f'{number}/{len(input_coverage)} rectangles; {len(rows)} contrast rows', flush=True)
    if args.summarize_existing:
        scan = pd.read_csv(OUT / 'all_structure_context_contrasts.csv.gz', low_memory=False)
        axes = pd.read_csv(OUT / 'condition_axis_registry.csv', keep_default_na=False, low_memory=False)
    else:
        scan = pd.DataFrame(rows)
        scan.to_csv(OUT / 'all_structure_context_contrasts.csv.gz', index=False, encoding='utf-8-sig')
        condition_registry = pd.read_csv(ROOT / 'data/condition_slot_registry.csv', keep_default_na=False, low_memory=False)
        axes = enrich_axis_registry(pd.DataFrame(axis_rows), condition_registry)
        axes.to_csv(OUT / 'condition_axis_registry.csv', index=False, encoding='utf-8-sig')
    selected = select_cases(scan, axes, targets_by_id, features)
    evidence = rectangle_evidence_counts(scan)
    swaps = pd.read_csv(ROOT / 'results/matched_swap_summary.csv', keep_default_na=False, low_memory=False)
    mapping = pd.read_csv(ROOT / 'round3/global_scope/dataset_reaction_class_mapping.csv', keep_default_na=False)
    coverage = mapping.merge(swaps.groupby('dataset_id').size().rename('n_swaps'), on='dataset_id')
    coverage = coverage.merge(input_coverage.groupby('dataset_id').size().rename('n_rectangles'), on='dataset_id', how='left')
    coverage = coverage.merge(input_coverage[input_coverage.main_n_ge20].groupby('dataset_id').size().rename('n_primary_rectangles'), on='dataset_id', how='left')
    coverage = coverage.merge(targets.groupby('dataset_id').size().rename('n_rectangle_target_means'), on='dataset_id', how='left').fillna(0)
    coverage[['n_rectangles', 'n_primary_rectangles', 'n_rectangle_target_means']] = coverage[['n_rectangles', 'n_primary_rectangles', 'n_rectangle_target_means']].astype(int)
    coverage.to_csv(OUT / 'dataset_coverage.csv', index=False, encoding='utf-8-sig')
    summary = {'registered_rectangles': len(input_coverage), 'unique_rectangles': int(targets.rectangle_id.nunique()), 'primary_rectangles_ge20': int(input_coverage.main_n_ge20.sum()), 'target_rectangle_means': len(targets), 'axis_representations': len(axes), 'all_scan_rows': len(scan), 'features': names, 'datasets_covered': 20, 'reaction_classes_covered': 18, 'datasets_with_rectangles': int(targets.dataset_id.nunique()), 'classes_with_rectangles': int(targets.rxn_class.nunique()), 'clear_display_rows': int(scan.clear_display_support.sum()), 'selected_case_count': len(selected), 'status_counts': scan.status.value_counts().to_dict(), 'all_axes_are_reorientations_of_the_same_four_recipes': True, 'partner_half_seed': SEED, 'partner_half_interpretation': 'shared focal molecules; partner sensitivity, not independent structural replication or experiments'}
    summary['rectangle_evidence'] = evidence
    summary.update(original_source_links(selected, targets_by_id))
    (OUT / 'structure_context_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(scan, selected, coverage, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
