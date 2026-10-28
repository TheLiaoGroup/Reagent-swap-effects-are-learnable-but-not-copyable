"""Apply the predeclared round-three example rule to stored predictions."""
from __future__ import annotations
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from publication_figures import audit_fig1_wells as wells
from publication_figures import common as c
MANIFEST_PATH = c.A / 'figure_assets/reference_selection_rule_20261022.json'
POPULATION_PATH = c.A / 'out/direction_metrics/direction_predictions_targets.csv.gz'
REGISTRY_PATH = c.A / 'scripts/reference_learning/Analysis/tables/chemical_observation_channels_target_registry.csv'
POOL_NAMES = ['flagship_catalyst_pair', 'all_suzuki', 'complete_population']

@lru_cache(maxsize=None)
def drawable(pair):
    substrates = pair.split(' >> ')
    return len(substrates) == 2 and all((s and Chem.MolFromSmiles(s) is not None for s in substrates))

def _number(value):
    return f'{float(value):g}'

def _background_labels(reference, target, category):
    labels = [[], []]
    for i in range(1, 7):
        slot = f'Reagent{i}'
        if slot == category:
            continue
        id_field, dose_field = (f'{slot}_id', f'{slot}_HTE_Equiv')
        if not wells.same(reference[id_field], target[id_field]) or not wells.same(reference[dose_field], target[dose_field]):
            dose_differs = not wells.same(reference[dose_field], target[dose_field])
            for text, recipe in zip(labels, [reference, target]):
                name = str(recipe[f'{slot}_Standardized Name'])
                if dose_differs and recipe[dose_field] != '':
                    name += f' ({_number(recipe[dose_field])} equiv)'
                text.append(name or 'No added reagent')
    for field, name_field, suffix in [('solvent_id', 'Standardized Name', ''), ('temp_id', 'temp_id', ' C'), ('time_id', 'time_id', ' h')]:
        if category in (field, field.removesuffix('_id'), 'temperature' if field == 'temp_id' else ''):
            continue
        if not wells.same(reference[field], target[field]):
            for text, recipe in zip(labels, [reference, target]):
                text.append((_number(recipe[name_field]) if suffix else str(recipe[name_field])) + suffix)
    if not all(labels):
        raise ValueError('Reference and target recipe backgrounds have no displayed difference')
    return ['; '.join(parts) + ' background' for parts in labels]

def _cells(prediction, target, matrices, recipes):
    screen = prediction.screen
    if screen not in matrices:
        path = c.A / 'scripts/reference_learning/Analysis/data' / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
        with np.load(path, allow_pickle=True) as source:
            matrices[screen] = (path, {name: source[name] for name in source.files})
    _, matrix = matrices[screen]
    pair_index = int(prediction.pair_index)
    if matrix['sps'][pair_index] != prediction.substrate_pair:
        raise ValueError('Stored prediction pair index does not match the matrix')
    cls, category = (wells.reaction_class(prediction), target.category)
    rows = []
    for background, prefix, orientation in [('Reference', 'reference_', target.reference_orientation), ('Target', '', target.orientation)]:
        indices = [int(target[prefix + 'condition_a_index']), int(target[prefix + 'condition_b_index'])]
        if int(orientation) not in (-1, 1):
            raise ValueError('Endpoint orientation is not signed')
        if int(orientation) == 1:
            indices.reverse()
        for position, index in enumerate(indices):
            condition = str(matrix['conds'][index])
            recipe = recipes.loc[cls, condition]
            rows.append({'background': background, 'reagent': wells.component_name(recipe, category), 'condition': condition, 'yield_percent': float(100 * matrix['M'][pair_index, index]), 'exchange_category': category, 'endpoint_position': position})
    cells = pd.DataFrame(rows)
    if not np.isfinite(cells.yield_percent).all():
        return None
    if len(set(cells.condition)) != 4:
        raise ValueError('Strict matched reference and target endpoints are not disjoint')
    for background, expected in [('Reference', prediction.reference_contrast_pp), ('Target', prediction.target_contrast_pp)]:
        measured = cells.loc[cells.background.eq(background), 'yield_percent'].to_numpy()
        if not np.isclose(measured[1] - measured[0], expected, rtol=0, atol=1e-08):
            raise ValueError('Stored prediction and measured endpoint orientation disagree')
    reference = recipes.loc[cls, cells.condition.iloc[0]]
    target_recipe = recipes.loc[cls, cells.condition.iloc[2]]
    labels = _background_labels(reference, target_recipe, category)
    cells['background_label'] = cells.background.map(dict(zip(['Reference', 'Target'], labels)))
    return cells

def _exact_errors(row):
    truth = Decimal(str(row.target_contrast_pp))
    return (abs(Decimal(str(row.semantic_full_raw_prediction_pp)) - truth), abs(Decimal(str(row.direct_raw_prediction_pp)) - truth))

def select_example(write_source=True):
    if write_source and (c.OUT.name != 'final_figures' or c.DATA.parent != c.OUT):
        raise ValueError('Final selection may write only to the production figure directory')
    predeclared = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if predeclared.get('predeclared_before_candidate_read') is not True:
        raise ValueError('Selection rule was not declared before reading candidates')
    rule = predeclared['predeclared_rules']['Figure1a']
    if len(rule['pool_order']) != 3 or len(rule['eligibility']) != 6:
        raise ValueError('Unexpected predeclared selection contract')
    rule_text = json.dumps(rule, ensure_ascii=True, sort_keys=True)
    source_paths = wells.paths()
    predictions = c.frame(POPULATION_PATH)
    if predictions.duplicated(['screen', 'edge_id', 'pair_index']).any():
        raise ValueError('Stored target predictions are not unique')
    literals = pd.read_csv(POPULATION_PATH, dtype=str, usecols=['screen', 'edge_id', 'pair_index', 'target_contrast_pp', 'direct_raw_prediction_pp', 'semantic_full_raw_prediction_pp'])
    for field in ['screen', 'edge_id']:
        if not literals[field].eq(predictions[field]).all():
            raise ValueError('Decimal source rows do not align with stored predictions')
    if not pd.to_numeric(literals.pair_index).eq(predictions.pair_index).all():
        raise ValueError('Decimal source pair indices do not align with stored predictions')
    numeric_columns = ['reference_contrast_pp', 'target_contrast_pp', 'direct_raw_prediction_pp', 'semantic_full_raw_prediction_pp']
    finite = np.isfinite(predictions[numeric_columns]).all(axis=1)
    candidates = predictions.loc[finite & predictions.reference_contrast_pp.abs().ge(10) & predictions.target_contrast_pp.abs().ge(20) & (predictions.target_contrast_pp * predictions.direct_raw_prediction_pp).lt(0) & (predictions.target_contrast_pp * predictions.semantic_full_raw_prediction_pp).gt(0)].copy()
    candidates['pair_index'] = pd.to_numeric(candidates.pair_index, errors='raise')
    if not candidates.pair_index.eq(candidates.pair_index.astype(int)).all():
        raise ValueError('Pair indices must be integers')
    candidates['pair_index'] = candidates.pair_index.astype(int)
    numeric_count = len(candidates)
    candidates = candidates.loc[candidates.substrate_pair.map(drawable)].copy()
    drawable_count = len(candidates)
    registry = c.frame(REGISTRY_PATH)
    if registry.duplicated(['screen', 'edge_id']).any():
        raise ValueError('Target registry identities are not unique')
    registry = registry.set_index(['screen', 'edge_id'], drop=False)
    conditions = c.frame(source_paths['conditions']).fillna('')
    needed_classes = set(candidates.screen.str.rsplit('|', n=1).str[0]) | {'Suzuki'}
    conditions = conditions.loc[conditions.reaction_class.isin(needed_classes)].copy()
    if conditions.duplicated(['reaction_class', 'condition_id']).any():
        raise ValueError('Condition registry identities are not unique')
    recipes = conditions.set_index(['reaction_class', 'condition_id'], drop=False).sort_index()
    exclusions = c.frame(source_paths['exclusions'])
    if len(exclusions) != 396 or not exclusions.name0.is_unique:
        raise ValueError('Expected 396 unique authoritative exclusion identities')
    raw = pd.read_csv(source_paths['reference_learning'], usecols=['reaction_class', 'pair_id', 'condition_id', 'name0', 'yield_numeric'], low_memory=False)
    matrices, candidate_cells, measured_indices = ({}, {}, [])
    for index, prediction in candidates.iterrows():
        target = registry.loc[prediction.screen, prediction.edge_id]
        cells = _cells(prediction, target, matrices, recipes)
        if cells is not None:
            candidate_cells[index] = cells
            measured_indices.append(index)
    candidates = candidates.loc[measured_indices].copy()
    requests = []
    for index, prediction in candidates.iterrows():
        for cell in candidate_cells[index].itertuples():
            requests.append({'candidate_index': index, 'reaction_class': wells.reaction_class(prediction), 'pair_id': prediction.substrate_pair, 'condition_id': cell.condition, 'matrix_yield_percent': cell.yield_percent})
    if not requests:
        raise ValueError('No candidates have four finite measured endpoints')
    requested = pd.DataFrame(requests).merge(raw, on=['reaction_class', 'pair_id', 'condition_id'], how='left')
    requested['valid_well'] = requested.name0.notna() & ~requested.name0.isin(exclusions.name0) & np.isfinite(requested.yield_numeric) & np.isclose(100 * requested.yield_numeric, requested.matrix_yield_percent, rtol=0, atol=1e-08)
    accepted = []
    for index, group in requested.groupby('candidate_index', sort=False):
        if len(group) == 4 and group.name0.nunique() == 4 and group.valid_well.all():
            accepted.append(index)
    eligible = candidates.loc[accepted].copy()
    if eligible.empty:
        raise ValueError('No comparison satisfies the predeclared eligibility rule')
    flagship = {str(recipes.loc[('Suzuki', condition), 'Reagent1_id']) for condition in ['C23', 'C26']}
    if flagship != {'CMP0092', 'CMP0097'}:
        raise ValueError('Flagship catalyst identities changed')
    eligible['is_suzuki'] = eligible.screen.str.rsplit('|', n=1).str[0].eq('Suzuki')
    eligible['is_flagship_catalyst_pair'] = [bool(row.is_suzuki and target.category == 'Reagent1' and ({str(target.low_id), str(target.high_id)} == flagship)) for _, row in eligible.iterrows() for target in [registry.loc[row.screen, row.edge_id]]]
    errors = {index: _exact_errors(literals.loc[index]) for index in eligible.index}
    eligible['absolute_learned_error_pp'] = [float(errors[index][0]) for index in eligible.index]
    eligible['absolute_copied_error_pp'] = [float(errors[index][1]) for index in eligible.index]
    eligible['four_name0_json'] = [json.dumps(requested.loc[requested.candidate_index.eq(index), 'name0'].tolist()) for index in eligible.index]
    masks = [eligible.is_flagship_catalyst_pair, eligible.is_suzuki, pd.Series(True, index=eligible.index)]
    counts = {name: int(mask.sum()) for name, mask in zip(POOL_NAMES, masks)}
    pool_index = next((i for i, name in enumerate(POOL_NAMES) if counts[name]))
    pool = eligible.loc[masks[pool_index]].copy()
    ordered_errors = sorted((errors[index][0] for index in pool.index))
    middle = len(ordered_errors) // 2
    median = ordered_errors[middle] if len(ordered_errors) % 2 else (ordered_errors[middle - 1] + ordered_errors[middle]) / 2
    distances = {index: abs(errors[index][0] - median) for index in pool.index}
    nearest = min(distances.values())
    tied = pool.loc[[index for index in pool.index if distances[index] == nearest]]
    selected_index = tied.sort_values(['screen', 'edge_id', 'pair_index'], ascending=True).index[0]
    row = predictions.loc[selected_index].copy()
    row['pair_index'] = int(row.pair_index)
    cells = candidate_cells[selected_index]
    target = registry.loc[row.screen, row.edge_id]
    reference = registry.loc[row.screen, target.reference_edge_id]
    bound_inputs = [POPULATION_PATH, REGISTRY_PATH, source_paths['conditions'], source_paths['reference_learning'], source_paths['exclusions'], *[entry[0] for entry in matrices.values()]]
    eligible['in_selected_pool'] = masks[pool_index]
    eligible['distance_to_selected_pool_median_pp'] = [float(abs(errors[index][0] - median)) for index in eligible.index]
    if write_source:
        c.source('S5_eligible_examples', eligible, bound_inputs, rule_text, 'yield percentage points', 'All three nested pool memberships are retained; exact decimal median distances resolve even-pool ties before lexical identity ties.')
        c.source('S5_pool_counts', pd.DataFrame([{'pool_order': i + 1, 'pool': name, 'eligible_count': counts[name], 'selected_pool': i == pool_index} for i, name in enumerate(POOL_NAMES)]), bound_inputs, rule_text, 'eligible comparisons')
        prediction_export = row.to_frame().T
        prediction_export['reference_mean_yield_percent'] = cells.loc[cells.background.eq('Reference'), 'yield_percent'].mean()
        c.source('S5_selected_prediction', prediction_export, bound_inputs, rule_text, 'yield percentage points')
        c.source('S5_experiment_grid', cells, bound_inputs, rule_text, 'measured yield %')
    audit = wells.check(row, cells)
    selection = {**row.to_dict(), 'eligible_pool_counts': counts, 'eligible_candidates_total': len(eligible), 'eligible_candidates': counts[POOL_NAMES[pool_index]], 'selected_pool': POOL_NAMES[pool_index], 'selected_pool_order': pool_index + 1, 'median_absolute_learned_error_pp': float(median), 'median_absolute_learned_error_decimal': str(median), 'absolute_learned_error_pp': float(errors[selected_index][0]), 'absolute_copied_error_pp': float(errors[selected_index][1]), 'distance_to_median_pp': float(nearest), 'nearest_median_tied_candidates': len(tied), 'tie_break': 'screen lexical, edge_id lexical, pair_index numeric; all ascending', 'eligibility_stages': {'stored_population': len(predictions), 'numeric_sign_and_amplitude': numeric_count, 'drawable_substrate_pairs': drawable_count, 'four_finite_matrix_cells': len(candidates), 'four_unique_current_unexcluded_wells': len(eligible)}, 'reference_context': reference.context_json, 'target_context': target.context_json, 'exchange_category': target.category, 'exchanged_components': cells.reagent.iloc[:2].tolist(), 'background_labels': cells.groupby('background', sort=False).background_label.first().to_dict(), 'source_well_audit': audit, 'rule_sha256': hashlib.sha256(rule_text.encode('utf-8')).hexdigest()}
    return (row, cells, selection)
