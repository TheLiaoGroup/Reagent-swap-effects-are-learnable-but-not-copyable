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
    lines = ['# Round 4: how background components rewrite the structural preference of a component swap', '', 'Date: 2026-10-16. Exploratory structural analysis of existing complete four-cell records.', '', '## Scientific object', '', 'In two backgrounds, the same component swap may only shift overall yield, or may change which structures benefit. This analysis compares the component-effect difference between the structure-positive and structure-negative groups: C0=E(Δbg0|feature+)−E(Δbg0|feature−), and likewise C1; the structural background contrast is T=C1−C0=E(DD|feature+)−E(DD|feature−). When T is near zero, the background mainly preserves this defined structural contrast; when T is large, the background changes its magnitude or sign. It does not require a large overall mean component effect.', '', f'The specific chemical judgments span several reaction classes: in SNAr, the relative response of NH-heteroaromatic substrates to DMSO→t-AmOH in the KHCO3 and DIPEA backgrounds changes from {100 * snar.controlled_C_bg0:+.2f} to {100 * snar.controlled_C_bg1:+.2f} pp; in Buchwald–Hartwig, the relative response of aryl-F-flagged substrates to BINAP Pd G3→XPhos Pd G3 changes with Cs2CO3→NaOtBu from {100 * buchwald.controlled_C_bg0:+.2f} to {100 * buchwald.controlled_C_bg1:+.2f} pp. The Suzuki Bpin aryl-Cl example instead shows the solvent-swap preference removed by the Na2CO3 background ({100 * suzuki.controlled_C_bg0:+.2f}→{100 * suzuki.controlled_C_bg1:+.2f} pp), with four-corner group means of {100 * suzuki_yields.min():.0f}–{100 * suzuki_yields.max():.0f}%, so this pattern does not come only from the near-zero-yield floor. They point to different structure×component×background relations; no shared mechanism is inferred.', '', f'Aligned with the main Acylation prediction rectangle, the broader aniline contrast has {int(aniline.controlled_n_positive_molecules_common)}/{int(aniline.controlled_n_negative_molecules_common)} molecules and {int(aniline.n_common_partners)} common acid partners: the relative response to TEA→DIPEA in the DCC and EDC-HCl backgrounds is {100 * aniline.controlled_C_bg0:+.2f}→{100 * aniline.controlled_C_bg1:+.3f} pp (T={100 * aniline.controlled_structural_DD:+.2f} pp), and T in the two partner halves is {100 * aniline.controlled_half1_structural_DD:+.2f}/{100 * aniline.controlled_half2_structural_DD:+.2f} pp. For the tertiary-amine contrast, the {100 * tertiary.controlled_structural_DD:+.2f} pp background change is kept as a subgroup of {int(tertiary.controlled_n_positive_molecules_common)} positive molecules; both come from the same rectangle and shared records and cannot be added as independent replications.', '', 'The main estimate controls for the partners actually shared: within each other-side partner that covers both groups, focal molecules in each group are weighted equally, targets under a focal molecule are averaged first, and common partners are then weighted equally. The partner composition of the groups thus has common support. Results with direct focal-molecule equal weighting are also kept to show the effect of sampling coverage on the conclusions. Descriptors may still correlate with other structures; controlling for partners does not turn structural classes into randomized interventions.', '', '## Scope and support', '', f"Of all {summary['registered_rectangles']} registered unique four-recipe rectangles, {summary['unique_rectangles']} have full support and {primary_count} reach the main range of at least 20 common targets; the input is {summary['target_rectangle_means']:,} target×rectangle means. By default only the representative condition axis is scanned; the other axis can be obtained by rearranging the corners of the same four recipes, with item-wise assertions that DD equals that of the representative axis or differs only in direction sign, so it is not a second experiment. The full scan covers {summary['axis_representations']} axis representations × two substrate sides × {len(summary['features'])} existing binary features, {len(scan):,} rows in total; constant features, no common partner and lack of valid molecules all keep their status.", '', '| Dataset | Corpus swaps | Registered four-cell rectangles | Main ≥20 targets | Estimable four-cell target means |', '|---|---:|---:|---:|---:|']
    for row in coverage.itertuples():
        lines.append(f'| {row.dataset_id} | {row.n_swaps} | {row.n_rectangles} | {row.n_primary_rectangles} | {row.n_rectangle_target_means:,} |')
    evidence = summary['rectangle_evidence']
    lines += ['', 'The table keeps all 20 datasets/18 reaction classes; no rectangles only means that under the current complete records and matching rules there is no estimand in this round. The Finkelstein single-substrate representation uses molecule-equal weighting only and constructs no real common partner.', '', f"Deduplicated by unique rectangle: of the {primary_count} main rectangles, {evidence['common_partner_comparison_rectangles']} have at least one structural feature/focal side comparable within common partners, {evidence['clear_support_rectangles']} have at least one clearly supported contrast, and {evidence['clear_support_and_half_split_rectangles']} additionally have ≥6 partners for splitting into halves. The rest remain in the {primary_count} denominator; non-estimable items are not called chemically null.", '', f'| Observed signal in unique rectangles | Any common partner | Clear support | Clear support and same-sign T in both partner halves | Same-sign version/{primary_count} | Same-sign version, reaction classes equal |', '|---|---:|---:|---:|---:|---:|']
    threshold_table = pd.DataFrame(evidence['threshold_results'])
    for metric, threshold, label in [('absolute_structural_DD', 0.05, 'abs(T)≥5pp'), ('absolute_structural_DD', 0.1, 'abs(T)≥10pp'), ('absolute_structural_DD', 0.2, 'abs(T)≥20pp'), ('preference_switch', 0.05, 'C0/C1 opposite sign, each ≥5pp'), ('preference_switch', 0.1, 'C0/C1 opposite sign, each ≥10pp')]:
        rows = threshold_table[threshold_table.signal.eq(metric) & threshold_table.threshold.eq(threshold)].set_index('support_stage')
        strict = rows.loc['clear_support_and_halves_same_sign']
        lines.append(f"| {label} | {int(rows.loc['any_common_partner', 'n_observed_signal_rectangles'])} | {int(rows.loc['clear_support', 'n_observed_signal_rectangles'])} | {int(strict.n_observed_signal_rectangles)} | {100 * strict.fraction_all_primary:.1f}% | {100 * strict.reaction_class_equal_fraction_all_primary:.1f}% |")
    lines += ['', f'Each cell counts a rectangle once, even if several related features or both substrate sides qualify. Reaction-class equal weighting uses the {primary_count} rectangles, covering {primary_classes} classes, with rectangles weighted equally within class; Finkelstein has no real common partner, so these ratios are the fraction of current records with displayable support, not the incidence of this mechanism across all reactions. Denominators under common-support conditions and per-class values are in the threshold summary table.', '', 'The clear-display label requires ≥3 focal molecules in each group and ≥3 common partners within common support. For ≥6 common partners, a fixed seed randomly splits them into two non-overlapping partner halves, and the direction and magnitude of C0, C1 and T are reported for each. This is a partner sensitivity check with shared focal molecules, not an independent experiment, structural replication or formal p-value. All continuous estimates and unsupported items are in the summary table. Four-corner zero-yield fractions are kept with the same level weights; zeros are not replaced, and unmeasured values are not filled with zero.', '', '## Chemical examples across reactions', '', f"Candidates are for display only: common-support label met, partner-half T of the same sign and each ≥5pp, overall |T|≥10pp; priority to C0/C1 changing sign outside ±5pp, then ordered by half-supported magnitude; one per dataset first, then the named contrasts of the main prediction rectangle are added. {len(summary['features'])} features, two condition axes and shared rectangles are not a count of independent findings. The full ranking is kept; scientific interpretation must go back to the real components, doses, measured four-corner yields and structures.", '', '| Dataset/focal side/structure | Component swap | Background swap | Group+/group− molecules; partners | C0→C1/pp | T/pp | Partner-half T/pp |', '|---|---|---|---:|---:|---:|---:|']
    for case in selected.itertuples():
        context0, context1 = (json.loads(case.context0_token), json.loads(case.context1_token))
        context0_dose = context0[1] if len(context0) > 1 else ''
        context1_dose = context1[1] if len(context1) > 1 else ''
        lines.append(f'| {case.dataset_id}/{case.role}/{case.feature} | {component_text(case.component0_name, case.component0_dose)}→{component_text(case.component1_name, case.component1_dose)} | {component_text(case.context0_name, context0_dose)}→{component_text(case.context1_name, context1_dose)} | {int(case.controlled_n_positive_molecules_common)}/{int(case.controlled_n_negative_molecules_common)}; {case.n_common_partners} | {100 * case.controlled_C_bg0:+.1f}→{100 * case.controlled_C_bg1:+.1f} | {100 * case.controlled_structural_DD:+.1f} | {100 * case.controlled_half1_structural_DD:+.1f}/{100 * case.controlled_half2_structural_DD:+.1f} |')
    lines += ['', 'Measured yields of the four-cell recipes by structural group:', '']
    for case in selected.itertuples():
        lines += [f'### {case.dataset_id}: {case.feature}', '', f'Rectangle `{case.rectangle_id}`, axis `{case.representation_id}`, {case.controlled_response_shape_5pp}.']
        for field in RECIPE_FIELDS:
            lines.append(f"- `{field}`: {getattr(case, 'full_' + field)}.")
        lines += ['', '| Structural group | Y0/bg0 | Y1/bg0 | Y0/bg1 | Y1/bg1 |', '|---|---:|---:|---:|---:|']
        for label, meaning in [('positive', 'Flagged'), ('negative', 'Not flagged')]:
            lines.append('| ' + meaning + ' | ' + ' | '.join((f"{100 * getattr(case, 'controlled_' + label + '_' + c):.2f}%" for c in CORNERS)) + ' |')
        lines += ['', 'Yields in the table use the same partner-controlled common support; a background that raises yields of both groups is not a change in structural preference. Per-target substrate/product SMILES, measured four-corner yields and partners are in the named target table; per-partner contrasts are in the partner table.', '']
    lines += ['## How the evidence is organized', '', 'The most informative result is a specific chemical judgment: changing which background component preserves, strengthens, removes or reverses the response of which structural group to the component swap. A high T does not mean a common mechanism; structural flags can be coarse, group coverage differs and molecules are shared, so partner halves, molecule-equal weighting and the zero floor should be read together. The two axes are two questions on the same four cells: along one axis, whether a component preference is changed by the background; along the other, with the subject swapped. The mathematical symmetry of DD does not make the two readings independent replications.', '', '## Outputs and recomputation', '', 'Code `Analysis/code/round4_structure_context.py`. Input comes from the unified round4 ingestion; the round3 model is not rebuilt.', '', '- `all_structure_context_contrasts.csv.gz`: all rectangles × axes × focal sides × features, with full support, both weightings, four-corner means, zero floor, partner halves and response shape.', '- `condition_axis_registry.csv`: corner mapping, DD direction, actual names/doses/four-cell recipes for the representative axis in this run; the original two-axis registry is `round3/context_rectangles/rectangle_registry.csv`.', '- `dataset_coverage.csv`, `structure_context_summary.json`: coverage and counts for 20 datasets/18 classes.', '- `ranked_chemical_candidates.csv`, `selected_structure_context_cases.csv`: exploratory chemical candidates and the first round of named displays.', '- `selected_case_target_four_yields.csv.gz`, `selected_case_partner_contrasts.csv`: complete four-cell targets and partner contrasts for the named examples.', '']
    lines += [f"{len(selected)} named structural contrasts come from {selected.rectangle_id.nunique()} unique rectangles. Source references are deduplicated by rectangle and representative axis; rebuilt {summary.get('selected_complete_source_blocks', 0):,} complete acquisition four-cell blocks and {summary.get('selected_original_record_links', 0):,} original record_id links; the four-corner means of each target were checked item by item against the unified input. `selected_case_original_record_links.csv.gz` provides record_id, source file/sheet/row, raw yield, recipe/condition and substrate/product SMILES; `selected_case_complete_acquisition_blocks.csv.gz` keeps the truly complete four-cell matched blocks. archive/plate labels do not automatically correspond to independent experimental plates.", '']
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
