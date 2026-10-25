"""Describe complete four-recipe accessibility and coordinated component changes."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round5' / 'joint_accessibility'
CORNERS = ['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']
NAMES = np.array(['00', '10', '01', '11'])
RECIPES = ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']
THRESHOLDS = [('20_5', 0.2, 0.05), ('10_2', 0.1, 0.02), ('20_10', 0.2, 0.1), ('50_5', 0.5, 0.05)]
EVENTS = ['one_high_three_low', 'two_diagonal_high_two_low', 'three_high_one_low']
MARGIN_EVENTS = ['diagonal_margin_ge5pp_high_min20pct', 'diagonal_margin_ge10pp_high_min20pct']
MATCHING_CASES = ['q_c75f1a38f55bfcc7', 'q_95d699c32341cb22', 'q_3040f9f4bed1125a']
COEF = np.array([1.0, -1.0, -1.0, 1.0])

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def classify(y, high=0.2, low=0.05):
    y = np.asarray(y, dtype=float)
    assert y.ndim == 2 and y.shape[1] == 4 and np.isfinite(y).all()
    h, l = (y >= high, y <= low)
    nh, nl = (h.sum(1), l.sum(1))
    diagonal = h[:, 0] & h[:, 3] | h[:, 1] & h[:, 2]
    topology = np.array([f'n{n}' for n in nh], dtype=object)
    topology[(nh == 2) & diagonal] = 'n2_diagonal'
    topology[(nh == 2) & ~diagonal] = 'n2_adjacent'
    strict = np.full(len(y), 'contains_intermediate', dtype=object)
    complete = nh + nl == 4
    strict[complete & (nh == 0)] = 'all_low'
    strict[complete & (nh == 1)] = EVENTS[0]
    strict[complete & (nh == 2) & diagonal] = EVENTS[1]
    strict[complete & (nh == 2) & ~diagonal] = 'two_adjacent_high_two_low'
    strict[complete & (nh == 3)] = EVENTS[2]
    strict[nh == 4] = 'all_high'
    states = np.where(h, 'H', np.where(l, 'L', 'M'))
    patterns = np.array([''.join(row) for row in states])
    return {'n_high': nh, 'n_low': nl, 'n_intermediate': 4 - nh - nl, 'high_topology': topology, 'strict_class': strict, 'state_00_10_01_11': patterns}

def add_target_metrics(targets):
    result = targets.copy()
    y = targets[CORNERS].to_numpy(float)
    order = np.argsort(-y, axis=1, kind='stable')
    row = np.arange(len(y))
    best, baseline = (order[:, 0], 3 - order[:, 0])
    single_a, single_b = (baseline ^ 1, baseline ^ 2)
    maximum, second = (y[row, best], y[row, order[:, 1]])
    result['minimum_yield'] = y.min(1)
    result['maximum_yield'] = maximum
    result['second_highest_yield'] = second
    result['highest_minus_next_yield'] = maximum - second
    result['n_tied_maxima'] = (y == maximum[:, None]).sum(1)
    result['maximum_corner'] = NAMES[best]
    result['opposite_baseline_corner'] = NAMES[baseline]
    result['opposite_baseline_yield'] = y[row, baseline]
    result['single_axis1_corner'] = NAMES[single_a]
    result['single_axis2_corner'] = NAMES[single_b]
    result['single_axis1_yield'] = y[row, single_a]
    result['single_axis2_yield'] = y[row, single_b]
    result['joint_gain_from_opposite_baseline'] = maximum - y[row, baseline]
    result['single_axis1_gain_from_opposite_baseline'] = y[row, single_a] - y[row, baseline]
    result['single_axis2_gain_from_opposite_baseline'] = y[row, single_b] - y[row, baseline]
    result['joint_oriented_DD'] = maximum - y[row, single_a] - y[row, single_b] + y[row, baseline]
    result['DD'] = y @ COEF
    result['abs_DD'] = np.abs(result.DD)
    result['diagonal_00_11_mean'] = y[:, [0, 3]].mean(1)
    result['diagonal_10_01_mean'] = y[:, [1, 2]].mean(1)
    margin0 = y[:, [0, 3]].min(1) - y[:, [1, 2]].max(1)
    margin1 = y[:, [1, 2]].min(1) - y[:, [0, 3]].max(1)
    result['diagonal_00_11_margin'] = margin0
    result['diagonal_10_01_margin'] = margin1
    result['larger_diagonal_min_minus_other_diagonal_max'] = np.maximum(margin0, margin1)
    stronger0 = margin0 >= margin1
    result['larger_margin_diagonal'] = np.where(stronger0, '00_11', '10_01')
    result['dominant_diagonal'] = np.where(np.maximum(margin0, margin1) > 0, result.larger_margin_diagonal, 'neither')
    result['larger_margin_diagonal_min_yield'] = np.where(stronger0, y[:, [0, 3]].min(1), y[:, [1, 2]].min(1))
    for event, margin in zip(MARGIN_EVENTS, [0.05, 0.1]):
        result[event] = (result.larger_diagonal_min_minus_other_diagonal_max >= margin) & (result.larger_margin_diagonal_min_yield >= 0.2)
    for tag, high, low in THRESHOLDS:
        for name, values in classify(y, high, low).items():
            result[name + '_' + tag] = values
    return result

def event_mask(part, event, tag):
    return part[event].astype(bool) if event in MARGIN_EVENTS else part['strict_class_' + tag].eq(event)

def molecule_counts(part):
    return {'n_targets': len(part), 'n_sub1_molecules': part.loc[part.sub_1_smiles.ne(''), 'sub_1_smiles'].nunique(), 'n_sub2_partners': part.loc[part.sub_2_smiles.ne(''), 'sub_2_smiles'].nunique()}

def directional_statistics(targets, rectangles):
    enriched = rectangles.copy().set_index('rectangle_id')
    for margin_tag, event in zip(['5pp', '10pp'], MARGIN_EVENTS):
        rows = []
        for rectangle_id, part in targets[targets[event]].groupby('rectangle_id', sort=False):
            row = {'rectangle_id': rectangle_id}
            sides = {}
            for diagonal in ['00_11', '10_01']:
                selected = part[part.dominant_diagonal.eq(diagonal)]
                counts = molecule_counts(selected)
                for key, value in counts.items():
                    row[f'{key}_{diagonal}_margin{margin_tag}'] = value
                sides[diagonal] = selected
            row[f'n_shared_partners_opposite_directions_margin{margin_tag}'] = len((set(sides['00_11'].sub_2_smiles) & set(sides['10_01'].sub_2_smiles)) - {''})
            row[f'n_shared_sub1_opposite_directions_margin{margin_tag}'] = len((set(sides['00_11'].sub_1_smiles) & set(sides['10_01'].sub_1_smiles)) - {''})
            rows.append(row)
        additions = pd.DataFrame(rows).set_index('rectangle_id').reindex(enriched.index).fillna(0).astype(int)
        enriched = enriched.drop(columns=list(additions.columns), errors='ignore').join(additions).copy()
        for stage in ['any_each_direction', '3targets_each_direction', '3targets_3sub1_3partners_each_direction', '3targets_3sub1_3partners_each_and_3sharedpartners']:
            fields = [f'n_targets_{d}_margin{margin_tag}' for d in ['00_11', '10_01']]
            minimum = 1 if stage == 'any_each_direction' else 3
            valid = enriched[fields].ge(minimum).all(axis=1)
            if '3sub1' in stage:
                fields = [f'{key}_{d}_margin{margin_tag}' for d in ['00_11', '10_01'] for key in ['n_sub1_molecules', 'n_sub2_partners']]
                valid &= enriched[fields].ge(3).all(axis=1)
            if '3sharedpartners' in stage:
                valid &= enriched[f'n_shared_partners_opposite_directions_margin{margin_tag}'].ge(3)
            enriched[f'bidirectional_matching_{stage}_margin{margin_tag}'] = valid
    enriched = enriched.reset_index()
    main = enriched[enriched.main_n_ge20]
    rows = []
    for margin_tag in ['5pp', '10pp']:
        for stage in ['any_each_direction', '3targets_each_direction', '3targets_3sub1_3partners_each_direction', '3targets_3sub1_3partners_each_and_3sharedpartners']:
            column = f'bidirectional_matching_{stage}_margin{margin_tag}'
            class_fractions = main.groupby('rxn_class')[column].mean()
            rows.append({'group_level': 'overall', 'rxn_class': 'all', 'margin_threshold': margin_tag, 'minimum_yield_both_dominant_recipes': 0.2, 'support_stage': stage, 'n_primary_rectangles': len(main), 'n_bidirectional_rectangles': int(main[column].sum()), 'rectangle_occurrence_fraction': main[column].mean(), 'class_equal_occurrence_fraction': class_fractions.mean(), 'n_reaction_classes_with_event': main.loc[main[column], 'rxn_class'].nunique()})
            for key, part in main.groupby('rxn_class', sort=True):
                rows.append({'group_level': 'reaction_class', 'rxn_class': key, 'margin_threshold': margin_tag, 'minimum_yield_both_dominant_recipes': 0.2, 'support_stage': stage, 'n_primary_rectangles': len(part), 'n_bidirectional_rectangles': int(part[column].sum()), 'rectangle_occurrence_fraction': part[column].mean(), 'class_equal_occurrence_fraction': np.nan, 'n_reaction_classes_with_event': int(part[column].any())})
    directional = pd.DataFrame(rows)
    save(directional, 'directional_matching_by_class.csv')
    return (enriched, directional)

def six_directional_targets(targets, rectangles):
    rows = []
    for rectangle_id in MATCHING_CASES:
        part = targets[targets.rectangle_id.eq(rectangle_id) & targets[MARGIN_EVENTS[1]]]
        for direction in ['00_11', '10_01']:
            row = part[part.dominant_diagonal.eq(direction)].sort_values('larger_diagonal_min_minus_other_diagonal_max', ascending=False).iloc[0].to_dict()
            metadata = rectangles[rectangles.rectangle_id.eq(rectangle_id)].iloc[0].to_dict()
            rows.append({**metadata, **row})
    result = pd.DataFrame(rows)
    identity = ['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', *CORNERS, 'DD', 'n_complete_acquisition_blocks', 'dominant_diagonal', 'diagonal_00_11_margin', 'diagonal_10_01_margin', 'larger_diagonal_min_minus_other_diagonal_max', 'larger_margin_diagonal_min_yield', 'n_targets']
    components = ['focal_slot', 'component0_name', 'component1_name', 'component0_dose', 'component1_dose', 'context_slot', 'context0_name', 'context1_name', 'context0_token', 'context1_token', 'fixed_background_json']
    support = [f'{key}_{d}_margin{margin}' for margin in ['5pp', '10pp'] for d in ['00_11', '10_01'] for key in ['n_targets', 'n_sub1_molecules', 'n_sub2_partners']]
    support += [f'n_shared_partners_opposite_directions_margin{margin}' for margin in ['5pp', '10pp']]
    result = result[identity + components + RECIPES + ['full_' + c for c in RECIPES] + support]
    save(result, 'selected_directional_matching_targets.csv')
    fixed_rows = []
    for index in fixed_partner_target_indices(targets):
        target = targets.loc[index].to_dict()
        metadata = rectangles[rectangles.rectangle_id.eq(target['rectangle_id'])].iloc[0].to_dict()
        fixed_rows.append({**metadata, **target})
    fixed = pd.DataFrame(fixed_rows)[result.columns]
    assert len(fixed) == 6
    assert fixed.groupby('rectangle_id').sub_2_smiles.nunique().eq(1).all()
    assert fixed.groupby('rectangle_id').sub_1_smiles.nunique().eq(2).all()
    assert fixed.groupby('rectangle_id').dominant_diagonal.nunique().eq(2).all()
    save(fixed, 'selected_fixed_partner_directional_targets.csv')
    return result

def fixed_partner_target_indices(targets):
    indices = []
    for rectangle_id in MATCHING_CASES:
        part = targets[targets.rectangle_id.eq(rectangle_id) & targets[MARGIN_EVENTS[1]]]
        columns = ['target_key', 'sub_1_smiles', 'sub_2_smiles', 'larger_diagonal_min_minus_other_diagonal_max']
        a = part[part.dominant_diagonal.eq('00_11')][columns]
        b = part[part.dominant_diagonal.eq('10_01')][columns]
        pairs = a.merge(b, on='sub_2_smiles', suffixes=('_00_11', '_10_01'))
        pairs = pairs[pairs.sub_1_smiles_00_11.ne(pairs.sub_1_smiles_10_01)].copy()
        assert len(pairs), rectangle_id
        pairs['minimum_pair_margin'] = pairs[['larger_diagonal_min_minus_other_diagonal_max_00_11', 'larger_diagonal_min_minus_other_diagonal_max_10_01']].min(axis=1)
        pairs['total_pair_margin'] = pairs[['larger_diagonal_min_minus_other_diagonal_max_00_11', 'larger_diagonal_min_minus_other_diagonal_max_10_01']].sum(axis=1)
        pair = pairs.sort_values(['minimum_pair_margin', 'total_pair_margin', 'target_key_00_11', 'target_key_10_01'], ascending=[False, False, True, True]).iloc[0]
        for diagonal in ['00_11', '10_01']:
            index = part[part.target_key.eq(pair['target_key_' + diagonal])].index
            assert len(index) == 1
            indices.append(index[0])
    return indices

def rectangle_summary(targets, metadata):
    rows = []
    grouped = {k: v for k, v in targets.groupby('rectangle_id', sort=False)}
    for meta in metadata.to_dict('records'):
        part = grouped.get(meta['rectangle_id'], targets.iloc[:0])
        row = dict(meta)
        row.update(molecule_counts(part))
        row['main_n_ge20'] = len(part) >= 20
        row['median_highest_minus_next_yield'] = part.highest_minus_next_yield.median()
        row['median_abs_DD'] = part.abs_DD.median()
        row['mean_signed_DD'] = part.DD.mean()
        for tag, high, low in THRESHOLDS:
            for n in range(5):
                row[f'n_active_{n}_{tag}'] = int(part['n_high_' + tag].eq(n).sum())
            events = EVENTS + ['two_adjacent_high_two_low', 'contains_intermediate', 'all_low', 'all_high']
            if tag == '20_5':
                events += MARGIN_EVENTS
            for event in events:
                selected = part.loc[event_mask(part, event, tag)]
                prefix = event + '_' + tag
                counts = molecule_counts(selected)
                row['n_' + prefix] = counts['n_targets']
                row['fraction_' + prefix] = len(selected) / len(part) if len(part) else np.nan
                for side in ['n_sub1_molecules', 'n_sub2_partners']:
                    row[side + '_' + prefix] = counts[side]
                row['display_support_' + prefix] = len(selected) >= 3 and counts['n_sub1_molecules'] >= 3 and (counts['n_sub2_partners'] >= 3)
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate(targets, rectangles):
    rows = []
    for panel, rect in [('all_registered', rectangles), ('primary_ge20', rectangles[rectangles.main_n_ge20])]:
        t = targets[targets.rectangle_id.isin(rect.rectangle_id)]
        for level, key, rpart in [('overall', 'all', rect)] + [('reaction_class', k, v) for k, v in rect.groupby('rxn_class', sort=True)]:
            p = t[t.rectangle_id.isin(rpart.rectangle_id)]
            unique_targets = p[['dataset_id', 'target_key']].drop_duplicates()
            for tag, high, low in THRESHOLDS:
                names = EVENTS + ['two_adjacent_high_two_low', 'contains_intermediate', 'all_low', 'all_high']
                names += [f'active_{n}' for n in range(5)] + ['n2_diagonal', 'n2_adjacent']
                if tag == '20_5':
                    names += MARGIN_EVENTS
                for event in names:
                    if event.startswith('active_'):
                        mask = p['n_high_' + tag].eq(int(event[-1]))
                    elif event.startswith('n2_'):
                        mask = p['high_topology_' + tag].eq(event)
                    else:
                        mask = event_mask(p, event, tag)
                    event_t = p.loc[mask]
                    by_rect = event_t.groupby('rectangle_id').size().reindex(rpart.rectangle_id, fill_value=0)
                    denominators = rpart.set_index('rectangle_id').n_targets
                    fractions = by_rect / denominators.replace(0, np.nan)
                    support = event_t.groupby('rectangle_id').apply(lambda z: len(z) >= 3 and z.sub_1_smiles.nunique() >= 3 and (z.loc[z.sub_2_smiles.ne(''), 'sub_2_smiles'].nunique() >= 3), include_groups=False) if len(event_t) else pd.Series(dtype=bool)
                    class_values = pd.DataFrame({'fraction': fractions}).join(rpart.set_index('rectangle_id').rxn_class)
                    class_fraction = class_values.groupby('rxn_class').fraction.mean().mean()
                    union = len(event_t[['dataset_id', 'target_key']].drop_duplicates())
                    multiplicity = event_t.groupby(['dataset_id', 'target_key']).size()
                    rows.append({'panel': panel, 'group_level': level, 'group': key, 'threshold_tag': tag, 'high_yield_threshold': high, 'low_yield_threshold': low, 'event': event, 'n_registered_rectangles': len(rpart), 'n_rectangles_with_targets': int(rpart.n_targets.gt(0).sum()), 'n_target_rectangle_observations': len(p), 'n_event_observations': int(mask.sum()), 'observation_fraction': float(mask.mean()) if len(p) else np.nan, 'n_rectangles_with_event': int(by_rect.gt(0).sum()), 'rectangle_occurrence_fraction': float(by_rect.gt(0).mean()) if len(rpart) else np.nan, 'rectangle_equal_mean_target_fraction': fractions.mean(), 'class_equal_mean_target_fraction': class_fraction, 'n_rectangles_3targets_3sub1_3partners': int(support.sum()), 'n_dataset_target_union': len(unique_targets), 'n_event_dataset_target_union': union, 'event_dataset_target_union_fraction': union / len(unique_targets) if len(unique_targets) else np.nan, 'event_target_median_rectangle_multiplicity': multiplicity.median(), 'event_target_max_rectangle_multiplicity': multiplicity.max()})
    return pd.DataFrame(rows)

def feature_string(features, smiles):
    row = features.get(smiles, {})
    return ';'.join((k for k, v in row.items() if k not in {'smiles', 'canonical_smiles', 'scaffold', 'scaffold_type', 'parsing_status', 'parsing_error'} and k not in {'MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'aromatic_N_count'} and (str(v) in {'1', '1.0'})))

def select_examples(rectangles, targets):
    features = read(ROOT / 'data/molecular_features.csv').set_index('smiles').to_dict('index')
    candidates = []
    for row in rectangles.to_dict('records'):
        for event in EVENTS + MARGIN_EVENTS:
            prefix = event + '_20_5'
            candidate = {k: v for k, v in row.items() if not any((k.endswith('_' + tag) for tag, _, _ in THRESHOLDS))}
            candidate.update(event=event, n_event_targets=row['n_' + prefix], event_fraction=row['fraction_' + prefix], n_event_sub1_molecules=row['n_sub1_molecules_' + prefix], n_event_sub2_partners=row['n_sub2_partners_' + prefix], three_by_three_support=row['display_support_' + prefix], selected_for_display=False)
            candidates.append(candidate)
    candidates = pd.DataFrame(candidates)
    eligible = candidates[candidates.main_n_ge20 & candidates.three_by_three_support].copy()
    eligible = eligible.sort_values(['event', 'rxn_class', 'event_fraction', 'n_event_targets'], ascending=[True, True, False, False])
    best_by_class = eligible.drop_duplicates(['event', 'rxn_class'])
    chosen = []
    for event, limit in [(EVENTS[0], 8), (EVENTS[1], 5), (EVENTS[2], 2), (MARGIN_EVENTS[1], 6)]:
        pool = best_by_class[best_by_class.event.eq(event)].sort_values('event_fraction', ascending=False)
        chosen.extend(pool.head(limit).index)
    candidates.loc[chosen, 'selected_for_display'] = True
    preferred = candidates.rectangle_id.isin(MATCHING_CASES) & candidates.event.eq(MARGIN_EVENTS[1])
    candidates.loc[preferred, 'selected_for_display'] = True
    candidates['rank_within_event_class'] = candidates.groupby(['event', 'rxn_class']).event_fraction.rank(method='min', ascending=False)
    candidates['candidate_status'] = np.where(~candidates.main_n_ge20, 'under20targets', np.where(candidates.n_event_targets.eq(0), 'zero_event', np.where(candidates.three_by_three_support, 'atleast3targets_3sub1_3partners', 'limited_molecule_or_partner_support')))
    selected = candidates[candidates.selected_for_display]
    display_targets = targets[targets.rectangle_id.isin(selected.rectangle_id)].copy()
    display_targets['sub1_feature_labels'] = display_targets.sub_1_smiles.map(lambda s: feature_string(features, s))
    display_targets['sub2_feature_labels'] = display_targets.sub_2_smiles.map(lambda s: feature_string(features, s))
    display_targets['source_target_selected'] = False
    for case in selected.itertuples():
        p = display_targets[display_targets.rectangle_id.eq(case.rectangle_id)]
        mask = event_mask(p, case.event, '20_5')
        event_t = p[mask]
        score = 'larger_diagonal_min_minus_other_diagonal_max' if case.event == EVENTS[1] or case.event in MARGIN_EVENTS else 'highest_minus_next_yield'
        event_t = event_t.sort_values(score, ascending=False)
        chosen_targets, sub1, sub2 = ([], set(), set())
        for index, row in event_t.iterrows():
            if row.sub_1_smiles not in sub1 or row.sub_2_smiles not in sub2:
                chosen_targets.append(index)
                sub1.add(row.sub_1_smiles)
                sub2.add(row.sub_2_smiles)
            if len(chosen_targets) == 3:
                break
        for index in event_t.index:
            if len(chosen_targets) >= 3:
                break
            if index not in chosen_targets:
                chosen_targets.append(index)
        alternative = p[~mask].sort_values('maximum_yield', ascending=False).head(1)
        display_targets.loc[chosen_targets + alternative.index.tolist(), 'source_target_selected'] = True
    for rectangle_id in MATCHING_CASES:
        p = display_targets[display_targets.rectangle_id.eq(rectangle_id) & display_targets[MARGIN_EVENTS[1]]]
        opposite_examples = p.sort_values('larger_diagonal_min_minus_other_diagonal_max', ascending=False).drop_duplicates('dominant_diagonal')
        display_targets.loc[opposite_examples.index, 'source_target_selected'] = True
    display_targets.loc[fixed_partner_target_indices(targets), 'source_target_selected'] = True
    save(candidates, 'selected_chemical_examples.csv')
    save(display_targets, 'selected_target_four_yields.csv')
    return (selected, display_targets)

def original_links(selected, display_targets):
    requested = display_targets[display_targets.source_target_selected]
    normalized = read(ROOT / 'data/normalized_records.csv.gz')
    normalized = normalized[normalized.strict_eligible.astype(str).str.lower().eq('true')]
    normalized = normalized[normalized.dataset_id.isin(requested.dataset_id) & normalized.target_key.isin(requested.target_key)].copy()
    normalized['yield_numeric'] = pd.to_numeric(normalized.yield_numeric, errors='raise')
    block_keys = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
    linked_rows, n_blocks = ([], 0)
    for case in selected.drop_duplicates('rectangle_id').itertuples():
        expected = requested[requested.rectangle_id.eq(case.rectangle_id)].set_index('target_key')
        recipes = [getattr(case, c) for c in RECIPES]
        frame = normalized[normalized.dataset_id.eq(case.dataset_id) & normalized.target_key.isin(expected.index) & normalized.recipe_id.isin(recipes)]
        cells = frame.groupby(block_keys + ['recipe_id'], as_index=False).yield_numeric.mean()
        complete = None
        for recipe, corner in zip(recipes, CORNERS):
            cell = cells[cells.recipe_id.eq(recipe)].drop(columns='recipe_id').rename(columns={'yield_numeric': corner})
            complete = cell if complete is None else complete.merge(cell, on=block_keys, validate='one_to_one')
        reconstructed = complete.groupby('target_key')[CORNERS].mean().reindex(expected.index)
        assert reconstructed.notna().all().all()
        assert np.allclose(reconstructed, expected[CORNERS].to_numpy(float), atol=1e-12, rtol=0)
        assert complete.groupby('target_key').size().reindex(expected.index).eq(expected.n_complete_acquisition_blocks).all()
        n_blocks += len(complete)
        for recipe, corner in zip(recipes, CORNERS):
            source = complete[block_keys + [corner]].rename(columns={corner: 'matched_cell_yield'}).copy()
            source['recipe_id'], source['corner'] = (recipe, corner)
            source['rectangle_id'] = case.rectangle_id
            linked = source.merge(frame, on=block_keys + ['recipe_id'], validate='one_to_many')
            assert linked.record_id.notna().all()
            linked_rows.append(linked[['rectangle_id', 'corner', *block_keys, 'recipe_id', 'record_id', 'condition_id', 'yield_numeric', 'matched_cell_yield', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']])
    source = pd.concat(linked_rows, ignore_index=True)
    raw = read(ROOT / 'data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row', 'yield', 'yield_numeric'])
    raw = raw.rename(columns={'yield': 'raw_yield_text', 'yield_numeric': 'raw_yield_numeric'})
    source = source.merge(raw, on='record_id', how='left', validate='many_to_one')
    assert source.source_file.notna().all()
    assert np.allclose(pd.to_numeric(source.raw_yield_text), source.yield_numeric, atol=1e-14, rtol=0)
    assert np.allclose(pd.to_numeric(source.raw_yield_numeric), source.yield_numeric, atol=1e-14, rtol=0)
    save(source, 'selected_original_record_links.csv.gz')
    return {'source_targets_verified': len(requested), 'complete_acquisition_blocks_verified': n_blocks, 'original_record_links_verified': len(source), 'unique_original_record_ids_verified': source.record_id.nunique()}

def coverage_table(rectangles, targets):
    coverage = read(ROOT / 'round3/global_scope/dataset_reaction_class_mapping.csv')
    swaps = read(ROOT / 'results/matched_swap_summary.csv')
    coverage = coverage.merge(swaps.groupby('dataset_id').size().rename('n_swaps'), on='dataset_id', how='left')
    rect = rectangles.groupby('dataset_id').agg(n_registered_rectangles=('rectangle_id', 'size'), n_primary_rectangles=('main_n_ge20', 'sum'), n_target_rectangle_observations=('n_targets', 'sum'))
    coverage = coverage.merge(rect, on='dataset_id', how='left').fillna(0)
    for event in EVENTS:
        hits = targets[targets.strict_class_20_5.eq(event)]
        counts = hits.groupby('dataset_id').agg(**{'n_' + event: ('target_key', 'size'), 'n_rectangles_' + event: ('rectangle_id', 'nunique')})
        coverage = coverage.merge(counts, on='dataset_id', how='left').fillna(0)
    coverage['rectangle_support_status'] = np.where(coverage.n_primary_rectangles.gt(0), 'primary_rectangles_available', np.where(coverage.n_registered_rectangles.gt(0), 'under20targets_only', 'no_registered_complete_two_slot_rectangle'))
    return coverage

def write_report(targets, rect, ag, selected, displayed, summary, directional):
    main = ag[(ag.panel == 'primary_ge20') & (ag.group_level == 'overall') & (ag.threshold_tag == '20_5')].set_index('event')
    lines = ['# 分子目标改变组分必须怎样成套匹配', '', '同一催化剂/碱或活化剂/添加剂矩形中，组分的匹配方向随目标分子改变：对一组目标，00与11配方共同占优；对另一组目标，10与01配方共同占优。配方的匹配方向由组分与目标分子的关系共同决定。', '', f'整体研究保持20数据集、18反应类、4,002单槽交换。这里使用全部{len(rect):,}注册矩形、{len(targets):,}个target×rectangle；其中{int(rect.main_n_ge20.sum())}矩形有至少20个完整共同target，构成主体分母。矩形覆盖{targets.dataset_id.nunique()}数据集/{targets.rxn_class.nunique()}反应类，其余数据集在dataset_coverage.csv中逐项保留。每个target的四角来自相同archive/plate/acquisition中的完整四格，先对同一采样匹配，再平均。', '', '矩形等权发生比例先在每个矩形内除以完整目标数，再令矩形等权；反应类等权先令每类内矩形等权，再令16个有矩形支持的反应类等权。无完整两槽矩形的研究反应类保留覆盖状态，不填入零发生比例。低支持的5矩形共34个target×rectangle，其中三高一低2个、一高三低及严格双对角均为0；它们保留在all_registered汇总，未进入825主体计数。', '', '## 1. 同一组件矩形中的匹配方向翻转', '', '对角分离裕度定义为max[min(Y00,Y11)−max(Y10,Y01), min(Y10,Y01)−max(Y00,Y11)]，优势对角的两格均要求≥20%。两种方向发生在不同target上；不把共享矩形或共享分子视为独立实验。', '', '| 对角分离阈值 | 各方向至少1目标：矩形/类 | 各方向≥3目标、3底物1、3伙伴：矩形/类 | 进一步≥3共同伙伴：矩形/类 |', '|---|---:|---:|---:|']
    for margin in ['5pp', '10pp']:
        values = []
        for stage in ['any_each_direction', '3targets_3sub1_3partners_each_direction', '3targets_3sub1_3partners_each_and_3sharedpartners']:
            row = directional[(directional.group_level == 'overall') & (directional.margin_threshold == margin) & (directional.support_stage == stage)].iloc[0]
            values.append(f'{int(row.n_bidirectional_rectangles)}/825；{int(row.n_reaction_classes_with_event)}类')
        lines.append(f'| {margin} | ' + ' | '.join(values) + ' |')
    lines += ['', '三类具名例的相反方向在同一真实矩形内同时出现。以下均要求优势对角两配方≥20%、对角分离≥10pp。', '', '| 反应/矩形 | 00+11占优：目标/底物1/伙伴 | 10+01占优：目标/底物1/伙伴 | 两方向共同伙伴 | 四格总目标 |', '|---|---:|---:|---:|---:|']
    for rectangle_id in MATCHING_CASES:
        row = rect[rect.rectangle_id.eq(rectangle_id)].iloc[0]
        values = ['/'.join((str(int(row[f'{key}_{d}_margin10pp'])) for key in ['n_targets', 'n_sub1_molecules', 'n_sub2_partners'])) for d in ['00_11', '10_01']]
        lines.append(f'| {row.rxn_class} `{rectangle_id}` | {values[0]} | {values[1]} | {int(row.n_shared_partners_opposite_directions_margin10pp)} | {row.n_targets} |')
    lines += ['', '三矩形、六个相反目标的完整SMILES、产物、实际四角、所有固定背景和各槽剂量在selected_directional_matching_targets.csv，六目标均进入同acquisition完整四格与raw转换回查。逐类分母与两阈值支持层级见directional_matching_by_class.csv。', '', '在同一实际伙伴下，仅改变另一侧分子也出现匹配方向翻转。selected_fixed_partner_directional_targets.csv另存三类反应、每类两个相反方向的目标，均满足≥10pp分离和优势两格≥20%，并全部回查源记录。', '', '| 反应 | 固定伙伴SMILES | 改变的底物1 | 优势对角 | 00/10/01/11产率% | 对角分离pp |', '|---|---|---|---|---|---:|']
    fixed_targets = read(OUT / 'selected_fixed_partner_directional_targets.csv')
    for row in fixed_targets.itertuples():
        yields = '/'.join((f'{100 * getattr(row, c):.2f}' for c in CORNERS))
        lines.append(f'| {row.rxn_class} | `{row.sub_2_smiles}` | `{row.sub_1_smiles}` | {row.dominant_diagonal} | {yields} | {100 * row.larger_diagonal_min_minus_other_diagonal_max:.2f} |')
    lines += ['', '## 2. 主体范围中的合成后果', '', '主阈值：H为产率≥20%，L为产率≤5%，M为两者之间。四角顺序为00、10、01、11，每个矩形与target只计一次。', '', '| 四格情形 | target×rectangle | 至少出现一次的矩形 / 825 | 矩形内发生比例等权 | 反应类等权 | ≥3目标/≥3底物1/≥3伙伴的矩形 |', '|---|---:|---:|---:|---:|---:|']
    labels = {EVENTS[0]: '一高三低：从对角低格联合打开', EVENTS[1]: '严格两对角高、两中间态低：成套匹配', EVENTS[2]: '三高一低：孤立不兼容', MARGIN_EVENTS[0]: '对角分离≥5pp，优势对角均≥20%', MARGIN_EVENTS[1]: '对角分离≥10pp，优势对角均≥20%'}
    for event in EVENTS + MARGIN_EVENTS:
        row = main.loc[event]
        lines.append(f'| {labels[event]} | {int(row.n_event_observations):,} | {int(row.n_rectangles_with_event)} / 825 | {100 * row.rectangle_equal_mean_target_fraction:.2f}% | {100 * row.class_equal_mean_target_fraction:.2f}% | {int(row.n_rectangles_3targets_3sub1_3partners)} |')
    lines += ['', '一高三低中的唯一高格相对对角低格需要同时改变两个槽；两条单改动的终点都≤5%。从另两个相邻低格到高格各只需一次改动，因此这个标签只定义该四格对角起点的组合打开。严格两对角高要求两套组合各≥20%、混搭组合各≤5%，直接显示这两组分必须成套匹配。它与仅按≥20%计数得到的n2_diagonal分开，后者可能包含中等产率。', '', '并列连续口径的对角分离裕度为max[min(Y00,Y11)−max(Y10,Y01), min(Y10,Y01)−max(Y00,Y11)]。正值表示某对角的两个配方都优于另两格；分别报告≥5/10个百分点且优势对角最低产率≥20%，保留负值及两个具体对角margin。这描述两套可观产率配方之间，任一单组分改动均损失，协调改变两组分重新得到可观产率。', '', '这些都是已测四配方中的实际结果；四角之外的条件空间未参与路径判断。最高与次高产率差、各条单改动、对角联合gain、带符号及绝对DD、三态排列和四组阈值都逐target保存。严格一高三低的joint_oriented_DD是有限四格差，最高格超出另三格的最小量即highest_minus_next_yield。', '', '## 3. 完整活跃数与阈值', '', '| ≥20%配方数 | target×rectangle | 矩形内比例等权 |', '|---|---:|---:|']
    for n in range(5):
        row = main.loc[f'active_{n}']
        lines.append(f'| {n} | {int(row.n_event_observations):,} | {100 * row.rectangle_equal_mean_target_fraction:.2f}% |')
    lines += ['', '| H/L阈值 | 一高三低：观察/矩形 | 严格对角匹配：观察/矩形 | 三高一低：观察/矩形 |', '|---|---:|---:|---:|']
    for tag, high, low in THRESHOLDS:
        sub = ag[(ag.panel == 'primary_ge20') & (ag.group_level == 'overall') & (ag.threshold_tag == tag)].set_index('event')
        values = [f'{int(sub.loc[e].n_event_observations):,}/{int(sub.loc[e].n_rectangles_with_event)}' for e in EVENTS]
        lines.append(f'| {high * 100:.0f}%/{low * 100:.0f}% | ' + ' | '.join(values) + ' |')
    lines += ['', '## 4. 同一目标重复出现在共享矩形中', '', '| 主阈值情形 | dataset+target并集 / 全目标并集 | 事件目标在矩形中的中位/最大重复次数 |', '|---|---:|---:|']
    for event in EVENTS:
        row = main.loc[event]
        lines.append(f'| {labels[event]} | {int(row.n_event_dataset_target_union):,}/{int(row.n_dataset_target_union):,} | {row.event_target_median_rectangle_multiplicity:.0f}/{row.event_target_max_rectangle_multiplicity:.0f} |')
    lines += ['', '去重结果表示同一dataset+target是否至少在一个已测矩形出现该情形，其分母是进入完整四格的目标并集。它不把同一目标的多个共享矩形当独立实验，也不把这一并集频率替代矩形内发生比例。', '', '## 5. 跨反应具名配方', '', 'selected_chemical_examples.csv保留全部830矩形×三种主情形及两种连续margin口径（包括零事件与支持不足），selected_for_display指定下列具名例。展示优先要求主体矩形且事件覆盖至少3目标、3种底物1及3种实际伙伴，再在每反应类内按事件比例排序；三类双方向配方匹配例另行保留。', '', '| 反应类 / 情形 | 第一槽改动 | 第二槽改动 | 事件数/共同目标 | 底物1/伙伴 | 四角实际配方 |', '|---|---|---|---:|---:|---|']
    for case in selected.itertuples():
        a = f'{case.component0_name} {case.component0_dose}→{case.component1_name} {case.component1_dose} ({case.focal_slot})'
        b = f'{case.context0_name}→{case.context1_name} ({case.context_slot}; token {case.context0_token}→{case.context1_token})'
        recipes = '<br>'.join((f"{n}: {getattr(case, 'full_' + c)}" for n, c in zip(NAMES, RECIPES)))
        lines.append(f'| {case.rxn_class} / {labels[case.event]} [{case.rectangle_id}] | {a} | {b} | {case.n_event_targets}/{case.n_targets} | {case.n_event_sub1_molecules}/{case.n_event_sub2_partners} | {recipes} |')
    lines += ['', '## 6. 实际目标四角与两个单改动', '', '下表每个具名比较给出一个已回查原始记录的事件目标。完整SMILES、结构标记、所有同矩形替代目标在selected_target_four_yields.csv；每矩形另外保留一个非该情形的高产率替代目标进入原始记录回查。', '', '| 反应 / 情形 / 目标 | 00/10/01/11产率% | 最高角 | 对角起点→最高格gain / 两条single gain，pp | DD / oriented DD / 最高次高gap，pp |', '|---|---|---|---|---|']
    for case in selected.itertuples():
        part = displayed[displayed.rectangle_id.eq(case.rectangle_id) & event_mask(displayed, case.event, '20_5') & displayed.source_target_selected]
        score = 'larger_diagonal_min_minus_other_diagonal_max' if case.event == EVENTS[1] or case.event in MARGIN_EVENTS else 'highest_minus_next_yield'
        t = part.sort_values(score, ascending=False).iloc[0]
        yields = '/'.join((f'{100 * t[c]:.2f}' for c in CORNERS))
        gains = '/'.join((f'{100 * t[c]:+.2f}' for c in ['joint_gain_from_opposite_baseline', 'single_axis1_gain_from_opposite_baseline', 'single_axis2_gain_from_opposite_baseline']))
        dds = '/'.join((f'{100 * t[c]:+.2f}' for c in ['DD', 'joint_oriented_DD', 'highest_minus_next_yield']))
        lines.append(f'| {case.rxn_class} / {case.event} / `{t.target_key}`<br>底物1 `{t.sub_1_smiles}`<br>伙伴 `{t.sub_2_smiles}` | {yields} | {t.maximum_corner} | {gains} | {dds} |')
    lines += ['', '对角双高情形应读四角本身及larger_diagonal_min_minus_other_diagonal_max：若以另一活跃对角格为起点，联合改动把一种匹配配方转到另一种匹配配方，而两条单改动落入低产率混搭格；最高角的joint gain可能接近零。这与一高三低中由低起点获得高产率的含义不同。', '', '## 7. 来源与复算', '', f"已对{summary['source_targets_verified']}个展示/替代目标重建{summary['complete_acquisition_blocks_verified']}个同acquisition完整四格块，核对{summary['original_record_links_verified']}条原始record_id链接；raw yield文本、raw numeric与normalized yield一致，重建target四角与统一输入逐项一致。source_file/source_sheet/source_row、archive/plate/acquisition、recipe/condition、SMILES均在selected_original_record_links.csv.gz。archive/plate标签不自动代表独立实验板。", '', '复算：运行code/round5_joint_accessibility.py；--self-test验证三态类别、所有8种矩形对称重排和高格/对角/单步映射。原始数据只读，零产率保留，缺失不填零。', '']
    (OUT / 'JOINT_ACCESSIBILITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def self_test():
    cases = [([0.01, 0.02, 0.03, 0.7], EVENTS[0]), ([0.7, 0.01, 0.02, 0.4], EVENTS[1]), ([0.7, 0.6, 0.02, 0.4], EVENTS[2]), ([0.7, 0.6, 0.02, 0.03], 'two_adjacent_high_two_low'), ([0.7, 0.15, 0.02, 0.03], 'contains_intermediate'), ([0, 0, 0, 0], 'all_low')]
    for values, expected in cases:
        y = np.array([values])
        assert classify(y)['strict_class'][0] == expected
        for flip1, flip2, transpose in product([0, 1], repeat=3):
            indices = []
            for i in range(4):
                a, b = (i & 1, i >> 1)
                if transpose:
                    a, b = (b, a)
                indices.append((a ^ flip1) + 2 * (b ^ flip2))
            assert classify(y[:, indices])['strict_class'][0] == expected
            assert np.isclose(abs(y[:, indices] @ COEF).item(), abs(y @ COEF).item())
            reordered = y[:, indices]
            original_margin = max(min(values[0], values[3]) - max(values[1], values[2]), min(values[1], values[2]) - max(values[0], values[3]))
            reordered_margin = max(reordered[0, [0, 3]].min() - reordered[0, [1, 2]].max(), reordered[0, [1, 2]].min() - reordered[0, [0, 3]].max())
            assert np.isclose(original_margin, reordered_margin)
            if expected == EVENTS[0]:
                best = int(np.argmax(y[:, indices]))
                baseline = 3 - best
                assert y[:, indices][0, baseline] <= 0.05
                assert y[:, indices][0, best ^ 1] <= 0.05 and y[:, indices][0, best ^ 2] <= 0.05
                assert best ^ 1 == baseline ^ 2 and best ^ 2 == baseline ^ 1
    boundary = classify([[0.2, 0.05, 0, 0]])
    assert boundary['strict_class'][0] == EVENTS[0]
    assert boundary['n_high'][0] == 1 and boundary['n_low'][0] == 3
    mapped = add_target_metrics(pd.DataFrame([[0.01, 0.02, 0.03, 0.7]], columns=CORNERS)).iloc[0]
    assert mapped.opposite_baseline_corner == '00' and mapped.maximum_corner == '11'
    assert mapped.single_axis1_corner == '10' and mapped.single_axis2_corner == '01'
    assert mapped.single_axis1_yield == 0.02 and mapped.single_axis2_yield == 0.03
    assert np.isclose(mapped.joint_oriented_DD, 0.66)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--reuse-main-tables', action='store_true', help='Reuse target labels, rectangle counts and aggregate tables for directional/source reporting.')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('Classification, threshold boundary, eight symmetries and diagonal/single-step maps passed.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    targets = read(ROOT / 'round4/rectangle_target_effects.csv.gz')
    assert not targets.duplicated(['rectangle_id', 'target_key']).any()
    metadata = read(ROOT / 'round4/context_transport/rectangle_input_coverage.csv')
    axes = read(ROOT / 'round4/structure_context/condition_axis_registry.csv')
    full_fields = ['rectangle_id', 'representation_id'] + ['full_' + c for c in RECIPES]
    metadata = metadata.merge(axes[full_fields], on=['rectangle_id', 'representation_id'], validate='one_to_one')
    assert len(metadata) == 830 and len(targets) == 176355
    targets[CORNERS + ['DD']] = targets[CORNERS + ['DD']].apply(pd.to_numeric, errors='raise')
    assert np.allclose(targets.DD, targets[CORNERS].to_numpy() @ COEF, atol=1e-12, rtol=0)
    if args.reuse_main_tables:
        targets = read(OUT / 'all_target_accessibility.csv.gz', dtype={c: str for c in ['maximum_corner', 'opposite_baseline_corner', 'single_axis1_corner', 'single_axis2_corner']})
        targets = add_target_metrics(targets)
        rectangles = read(OUT / 'rectangle_accessibility_summary.csv')
    else:
        targets = add_target_metrics(targets)
        rectangles = rectangle_summary(targets, metadata)
    assert int(rectangles.main_n_ge20.sum()) == 825
    rectangles, directional = directional_statistics(targets, rectangles)
    save(targets, 'all_target_accessibility.csv.gz')
    save(rectangles, 'rectangle_accessibility_summary.csv')
    six_directional_targets(targets, rectangles)
    print(directional[directional.group_level.eq('overall')].to_json(orient='records'), flush=True)
    print('Target labels and 830-rectangle summaries saved.', flush=True)
    if args.reuse_main_tables:
        ag = read(OUT / 'aggregate_results.csv')
        categories = {'panel', 'group_level', 'group', 'threshold_tag', 'event'}
        for column in set(ag.columns) - categories:
            ag[column] = pd.to_numeric(ag[column].replace('', np.nan), errors='raise')
    else:
        ag = aggregate(targets, rectangles)
        save(ag, 'aggregate_results.csv')
    coverage = coverage_table(rectangles, targets)
    save(coverage, 'dataset_coverage.csv')
    selected, displayed = select_examples(rectangles, targets)
    summary = {'datasets_in_study': len(coverage), 'reaction_classes_in_study': coverage.rxn_class.nunique(), 'single_slot_swaps_in_study': int(coverage.n_swaps.sum()), 'registered_rectangles': len(rectangles), 'primary_rectangles_ge20': int(rectangles.main_n_ge20.sum()), 'target_rectangle_observations': len(targets), 'rectangle_datasets': targets.dataset_id.nunique(), 'rectangle_reaction_classes': targets.rxn_class.nunique(), 'reaction_classes_in_rectangle_class_equal_means': targets.rxn_class.nunique(), 'candidate_rows_retained': len(rectangles) * (len(EVENTS) + len(MARGIN_EVENTS)), 'selected_display_comparisons': len(selected), 'selected_unique_rectangles': selected.rectangle_id.nunique(), 'thresholds': THRESHOLDS, 'corner_order': NAMES.tolist(), 'synthetic_symmetry_and_mapping_checks_passed': True}
    summary['bidirectional_matching'] = directional[directional.group_level.eq('overall')].to_dict('records')
    primary = ag[(ag.panel == 'primary_ge20') & (ag.group_level == 'overall') & (ag.threshold_tag == '20_5')]
    summary['primary_events'] = primary[primary.event.isin(EVENTS + MARGIN_EVENTS)].to_dict('records')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    summary.update(original_links(selected, displayed))
    write_report(targets, rectangles, ag, selected, displayed, summary, directional)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Complete source blocks/raw conversions verified; report saved.', flush=True)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
