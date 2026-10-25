"""One identical measured recipe-substitution path for four reactant pairs.

Inputs are the full strict-source and common-graph scope used in round 7.
Every write stays under Analysis/round9/shared_recipe_paths."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import csv
import gzip
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import round7_joint_reactant_connectivity as prior
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'round7/joint_reactant_connectivity'
OUT = ROOT / 'round9/shared_recipe_paths'
SIDES = prior.SIDES
EPS = 1e-12
FLAGS = ['positive_joint_deficit', 'joint_deficit_ge5pp', 'joint_deficit_ge10pp', 'all_individually_connected20', 'jointly_connected20', 'individual20_but_not_joint20', 'individual20_but_not_joint20_gap5pp', 'individual20_but_not_joint20_gap10pp', 'individual20_but_not_joint20_clearance5pp', 'individual20_but_not_joint20_clearance10pp', 'joint_bottleneck_zero']
SUBSETS = [subset for size in [2, 3] for subset in combinations(range(4), size)]
SUBSET_NAMES = ['_'.join((SIDES[index] for index in subset)) for subset in SUBSETS]
SUBSET_EVENTS = ['all_pairs_shared_but_four_not20', 'all_triples_shared_but_four_not20', 'minimum_blocking_target_count2', 'minimum_blocking_target_count3', 'minimum_blocking_target_count4']
SUBSET_SENSITIVITY_THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3)

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)

def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 1} if name.endswith('.gz') else None)

def metrics(capacities, joint):
    individual = capacities.min(0)
    deficit = individual - joint
    assert (deficit >= -EPS).all()
    deficit[np.abs(deficit) <= EPS] = 0.0
    all_connected = individual >= 0.2 - EPS
    joint_connected = joint >= 0.2 - EPS
    event = all_connected & ~joint_connected
    clearance = np.minimum(individual - 0.2, 0.2 - joint)
    flags = dict(positive_joint_deficit=deficit > EPS, joint_deficit_ge5pp=deficit >= 0.05 - EPS, joint_deficit_ge10pp=deficit >= 0.1 - EPS, all_individually_connected20=all_connected, jointly_connected20=joint_connected, individual20_but_not_joint20=event, individual20_but_not_joint20_gap5pp=event & (deficit >= 0.05 - EPS), individual20_but_not_joint20_gap10pp=event & (deficit >= 0.1 - EPS), individual20_but_not_joint20_clearance5pp=event & (clearance >= 0.05 - EPS), individual20_but_not_joint20_clearance10pp=event & (clearance >= 0.1 - EPS), joint_bottleneck_zero=np.abs(joint) <= EPS)
    assert not (joint_connected & ~all_connected).any()
    return (individual, deficit, clearance, flags)

def induced_labels(y, adjacent, threshold=0.2):
    high = np.flatnonzero(y >= threshold - EPS)
    labels = np.full(len(y), -1, dtype=int)
    if len(high):
        _, groups = connected_components(csr_matrix(adjacent[np.ix_(high, high)]), directed=False)
        labels[high] = groups
    return labels

def shortest_optimal_path(y, adjacent, start, goal, bottleneck):
    """Use SciPy's shortest path within the already verified optimal floor."""
    kept = np.flatnonzero(y >= bottleneck - EPS)
    positions = {int(value): j for j, value in enumerate(kept)}
    local_start, local_goal = (positions[start], positions[goal])
    distance, previous = shortest_path(csr_matrix(adjacent[np.ix_(kept, kept)]), directed=False, indices=local_start, unweighted=True, return_predecessors=True)
    assert np.isfinite(distance[local_goal])
    path = [local_goal]
    while path[-1] != local_start:
        ancestor = int(previous[path[-1]])
        assert ancestor >= 0
        path.append(ancestor)
    result = kept[path[::-1]].tolist()
    assert abs(float(y[result].min()) - bottleneck) <= EPS
    return result

def round8_local_quartets():
    path = ROOT / 'round8/partner_neighborhoods/selected_cross_class_full_16_cell_examples.csv'
    lookup = {}
    if not path.exists():
        return lookup
    for row in read(path).itertuples(index=False):
        first = sorted([row.edit1_structural_molecule_a, row.edit1_structural_molecule_b])
        second = sorted([row.edit2_structural_molecule_a, row.edit2_structural_molecule_b])
        if row.role != 'sub_1_smiles':
            first, second = (second, first)
        key = tuple([row.dataset_id, *first, *second])
        lookup.setdefault(key, []).append(row.local_rectangle_id)
    return lookup

def self_checks():
    adjacent = np.array([[False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]])
    y = np.array([[0.8, 0.6, 0.8, 0.05], [0.8, 0.05, 0.8, 0.6], [0.8, 0.6, 0.8, 0.05], [0.8, 0.05, 0.8, 0.6]])
    individual = np.array([prior.molecular.widest_matrix(row, adjacent) for row in y])
    joint = prior.molecular.widest_matrix(y.min(0), adjacent)
    assert np.isclose(individual[:, 0, 2].min(), 0.6) and np.isclose(joint[0, 2], 0.05)
    assert len(shortest_optimal_path(y.min(0), adjacent, 0, 2, 0.05)) == 3
    assert np.isclose(individual[:, 0, 2].min() - joint[0, 2], 0.55)
    for row in [*y, y.min(0)]:
        np.testing.assert_allclose(prior.molecular.widest_matrix(row, adjacent), prior.mst_capacity(row, adjacent), atol=EPS, rtol=0)
    minimum, deficit, clearance, flags = metrics(individual[:, [0], [2]], joint[[0], [2]])
    assert flags['individual20_but_not_joint20_clearance10pp'].item()
    assert np.isclose(deficit.item(), 0.55) and np.isclose(clearance.item(), 0.15)
    repeated = np.repeat(y[:1], 4, axis=0)
    repeated_individual = np.array([prior.molecular.widest_matrix(row, adjacent) for row in repeated])
    repeated_joint = prior.molecular.widest_matrix(repeated.min(0), adjacent)
    assert np.allclose(repeated_individual.min(0), repeated_joint)
    disconnected = adjacent.copy()
    disconnected[0, :] = False
    disconnected[:, 0] = False
    assert np.isneginf(prior.molecular.widest_matrix(y[0], disconnected)[0, 2])

def load():
    mapping, norm, target, graphs, blocks, by_target, values, block_frame = prior.prepare()
    scope = read(SOURCE / 'all_acquisition_quartet_scope.csv.gz')
    annotations = read(SOURCE / 'complete_grid_molecular_relations.csv.gz')
    scope = scope.merge(annotations, on='grid_id', validate='many_to_one')
    assert len(scope) == 127099 and scope.grid_id.nunique() == 102270
    assert scope.n_common_high_endpoint_pairs_with_full_graph_path.sum() == 7853943
    assert len(norm) == 380910 and len(blocks) == 5889 and (len(target) == 5554)
    source = read(SOURCE / 'all_strict_source_records.csv.gz')
    assert len(source) == len(norm) and source.record_id.is_unique
    aligned = norm.set_index('record_id').loc[source.record_id]
    assert np.allclose(source.yield_numeric, aligned.yield_numeric, atol=EPS)
    for field in [*prior.BLOCK, 'recipe_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']:
        assert np.array_equal(source[field], aligned[field]), field
    counts = source.groupby(['block_id', 'recipe_id']).size().to_dict()
    return (mapping, norm, target, graphs, blocks, values, block_frame, scope, source, counts)

def analyze(scope, graphs, blocks, values, source_counts):
    checks = Counter()
    used_source_nodes = set()

    @lru_cache(maxsize=1024)
    def graph_info(dataset, common):
        adjacent = graphs[dataset]['adjacent'][np.ix_(common, common)]
        _, groups = connected_components(csr_matrix(adjacent), directed=False)
        return (adjacent, groups)

    def checked_capacity(y, adjacent, kind):
        capacity = prior.molecular.widest_matrix(y, adjacent)
        independent = prior.mst_capacity(y, adjacent)
        np.testing.assert_allclose(capacity, independent, atol=EPS, rtol=0)
        labels = induced_labels(y, adjacent)
        high = np.flatnonzero(labels >= 0)
        assert np.array_equal(capacity[np.ix_(high, high)] >= 0.2 - EPS, labels[high, None] == labels[None, high])
        checks[kind + '_full_capacity_matrices_verified'] += 1
        checks[kind + '_capacity_entries_verified'] += capacity.size
        return capacity

    @lru_cache(maxsize=2048)
    def target_capacity(block_id, common):
        dataset = blocks[block_id][0]
        adjacent, _ = graph_info(dataset, common)
        return checked_capacity(values[block_id][list(common)], adjacent, 'individual')
    columns = ['acquisition_quartet_id', 'endpoint0_recipe', 'endpoint1_recipe']
    columns += ['K_' + side for side in SIDES]
    columns += ['min_individual_K', 'K_joint', 'joint_deficit', 'threshold20_clearance']
    columns += ['lower_endpoint_yield_' + side for side in SIDES] + ['minimum_all_eight_endpoint_yields'] + FLAGS
    rows, endpoint_total = ([], 0)
    with gzip.open(OUT / 'all_7994464_endpoint_shared_route_metrics.csv.gz', 'wt', encoding='utf-8-sig', newline='', compresslevel=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for count, original in enumerate(scope.to_dict('records'), 1):
            row = dict(original)
            graph = graphs[row['dataset_id']]
            recipes = json.loads(row['common_recipe_ids'])
            common = tuple((graph['lookup'][r] for r in recipes))
            ys = np.array([values[row['block_id_' + side]][list(common)] for side in SIDES])
            assert ys.shape == (4, len(common)) and np.isfinite(ys).all()
            adjacent, groups = graph_info(row['dataset_id'], common)
            assert int(adjacent.sum() // 2) == row['n_common_graph_edges']
            for side in SIDES:
                key = blocks[row['block_id_' + side]]
                row.update({field + '_' + side: key[j] for j, field in enumerate(prior.BLOCK) if field in ['archive', 'plate', 'acquisition']})
                block_id = row['block_id_' + side]
                row['n_common_recipe_source_records_' + side] = sum((source_counts[block_id, recipe] for recipe in recipes))
                used_source_nodes.update(((block_id, recipe) for recipe in recipes))
            row['n_common_recipe_source_records_all_sides'] = sum((row['n_common_recipe_source_records_' + side] for side in SIDES))
            high = np.flatnonzero((ys >= 0.2).all(0))
            ends = [(i, j) for i, j in combinations(high, 2) if groups[i] == groups[j]]
            assert len(ends) == row['n_common_high_endpoint_pairs_with_full_graph_path']
            for flag in FLAGS:
                row['n_pairs_' + flag], row['any_' + flag] = (0, False)
            for field in ['maximum_joint_deficit', 'maximum_event_joint_deficit', 'maximum_event_threshold_clearance', 'maximum_joint_K', 'minimum_joint_K', 'maximum_min_individual_K']:
                row[field] = np.nan
            row['representative_endpoint0_recipe'] = row['representative_endpoint1_recipe'] = ''
            if ends:
                ii, jj = np.asarray(ends, int).T
                full_capacity = np.array([target_capacity(row['block_id_' + side], common) for side in SIDES])
                joint_full = checked_capacity(ys.min(0), adjacent, 'joint')
                assert (joint_full <= full_capacity.min(0) + EPS).all()
                checks['full_matrix_max_min_inequality_entries_verified'] += joint_full.size
                cap, joint = (full_capacity[:, ii, jj], joint_full[ii, jj])
                lower = np.minimum(ys[:, ii], ys[:, jj])
                minimum, deficit, clearance, flags = metrics(cap, joint)
                assert np.isfinite(cap).all() and np.isfinite(joint).all()
                assert (cap <= lower + EPS).all() and (joint <= lower.min(0) + EPS).all()
                for flag in FLAGS:
                    active = flags[flag]
                    row['n_pairs_' + flag], row['any_' + flag] = (int(active.sum()), bool(active.any()))
                event = flags['individual20_but_not_joint20']
                row.update(maximum_joint_deficit=float(deficit.max()), maximum_joint_K=float(joint.max()), minimum_joint_K=float(joint.min()), maximum_min_individual_K=float(minimum.max()), maximum_event_joint_deficit=float(deficit[event].max()) if event.any() else np.nan, maximum_event_threshold_clearance=float(clearance[event].max()) if event.any() else np.nan)
                chosen = max(range(len(ends)), key=lambda z: (bool(event[z]), float(clearance[z]) if event[z] else float(deficit[z]), float(deficit[z]), float(minimum[z])))
                row['representative_endpoint0_recipe'], row['representative_endpoint1_recipe'] = (recipes[ii[chosen]], recipes[jj[chosen]])
                row.update(representative_min_individual_K=float(minimum[chosen]), representative_K_joint=float(joint[chosen]), representative_joint_deficit=float(deficit[chosen]), representative_threshold20_clearance=float(clearance[chosen]), representative_minimum_all_eight_endpoint_yields=float(lower[:, chosen].min()), representative_is_individual20_but_not_joint20=bool(event[chosen]))
                for z, side in enumerate(SIDES):
                    row['representative_K_' + side] = float(cap[z, chosen])
                    row['representative_lower_endpoint_yield_' + side] = float(lower[z, chosen])
                writer.writerows(([row['acquisition_quartet_id'], recipes[i], recipes[j], *cap[:, z].tolist(), float(minimum[z]), float(joint[z]), float(deficit[z]), float(clearance[z]), *lower[:, z].tolist(), float(lower[:, z].min()), *[bool(flags[flag][z]) for flag in FLAGS]] for z, (i, j) in enumerate(ends)))
                endpoint_total += len(ends)
            rows.append(row)
            if count % 5000 == 0:
                events = sum((r['any_individual20_but_not_joint20'] for r in rows))
                print(f"QUARTETS {count}/{len(scope)} ENDPOINTS {endpoint_total} EVENT_QUARTETS {events} MATRICES {checks['individual_full_capacity_matrices_verified']}+{checks['joint_full_capacity_matrices_verified']}", flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 127099 and result.grid_id.nunique() == 102270 and (endpoint_total == 7853943)
    assert not (result.any_jointly_connected20 & ~result.any_all_individually_connected20).any()
    save(result, 'all_69128_acquisition_quartet_shared_route_summary.csv.gz')
    checks.update(acquisition_quartets=len(result), complete_reactant_grids=result.grid_id.nunique(), endpoint_pairs=endpoint_total)
    source_nodes = pd.DataFrame([dict(block_id=block, recipe_id=recipe, n_source_records=source_counts[block, recipe]) for block, recipe in sorted(used_source_nodes)])
    save(source_nodes, 'all_common_support_source_block_recipe_counts.csv.gz')
    grid = result.groupby(['grid_id', 'dataset_id', 'rxn_class'], sort=False, as_index=False).agg(n_acquisition_quartets=('acquisition_quartet_id', 'size'), n_endpoint_comparable_quartets=('comparability_status', lambda x: int(x.eq('endpoint_comparable').sum())), n_endpoint_pair_views=('n_common_high_endpoint_pairs_with_full_graph_path', 'sum'), maximum_joint_deficit=('maximum_joint_deficit', 'max'), maximum_event_joint_deficit=('maximum_event_joint_deficit', 'max'), maximum_event_threshold_clearance=('maximum_event_threshold_clearance', 'max'), **{'any_' + flag: ('any_' + flag, 'max') for flag in FLAGS})
    save(grid, 'all_52598_molecular_grid_shared_route_summary.csv.gz')
    (OUT / 'matrix_verification_counts.json').write_text(json.dumps(dict(checks), indent=2) + '\n', encoding='utf-8')
    return (result, grid, checks)

def summarize(result, grid):
    aggregate = []
    for grouping, key in [('all', None), ('dataset_id', 'dataset_id'), ('rxn_class', 'rxn_class')]:
        parts = [('all', result)] if key is None else result.groupby(key, sort=True)
        for label, part in parts:
            for flag in FLAGS:
                event = part[part['any_' + flag]]
                aggregate.append(dict(grouping=grouping, group=str(label), metric=flag, n_all_quartets=len(part), n_all_molecular_grids=part.grid_id.nunique(), n_endpoint_comparable_quartets=int(part.comparability_status.eq('endpoint_comparable').sum()), n_endpoint_comparable_grids=part.loc[part.comparability_status.eq('endpoint_comparable'), 'grid_id'].nunique(), n_endpoint_pair_views=int(part.n_common_high_endpoint_pairs_with_full_graph_path.sum()), n_event_quartets=len(event), n_event_grids=event.grid_id.nunique(), n_event_datasets=event.dataset_id.nunique(), n_event_classes=event.rxn_class.nunique(), n_event_endpoint_pairs=int(part['n_pairs_' + flag].sum()), n_non_event_endpoint_pairs=int((part.n_common_high_endpoint_pairs_with_full_graph_path - part['n_pairs_' + flag]).sum())))
    aggregate = pd.DataFrame(aggregate)
    save(aggregate, 'all_event_and_non_event_denominators.csv')
    coverage = read(SOURCE / 'scope_dataset_coverage.csv')
    for flag in FLAGS:
        by_dataset = result.groupby('dataset_id')['n_pairs_' + flag].sum()
        coverage['n_endpoint_pairs_' + flag] = coverage.dataset_id.map(by_dataset).fillna(0).astype(int)
        by_dataset = result.groupby('dataset_id')['any_' + flag].sum()
        coverage['n_quartets_' + flag] = coverage.dataset_id.map(by_dataset).fillna(0).astype(int)
        by_dataset = grid.groupby('dataset_id')['any_' + flag].sum()
        coverage['n_molecular_grids_' + flag] = coverage.dataset_id.map(by_dataset).fillna(0).astype(int)
    assert len(coverage) == 20 and coverage.rxn_class.nunique() == 18 and (coverage.n_legacy_swaps.sum() == 4002)
    save(coverage, 'all_20_dataset_scope_events_and_zeros.csv')
    save(result.groupby(['dataset_id', 'rxn_class', 'comparability_status'], as_index=False).agg(n_quartets=('acquisition_quartet_id', 'size'), n_molecular_grids=('grid_id', 'nunique')), 'all_support_statuses.csv')
    exact = []
    for label, mask in [('all', np.ones(len(result), bool)), ('either_side_exact_edit', result.has_exact_edit_either_side), ('both_sides_exact_edits', result.has_exact_edit_both_sides)]:
        part = result[mask]
        event = part[part.any_individual20_but_not_joint20]
        exact.append(dict(molecular_scope=label, n_all_quartets=len(part), n_all_grids=part.grid_id.nunique(), n_endpoint_pair_views=int(part.n_common_high_endpoint_pairs_with_full_graph_path.sum()), n_event_quartets=len(event), n_event_grids=event.grid_id.nunique(), n_event_classes=event.rxn_class.nunique(), n_event_endpoint_pairs=int(part.n_pairs_individual20_but_not_joint20.sum()), n_event_endpoint_pairs_clearance10pp=int(part.n_pairs_individual20_but_not_joint20_clearance10pp.sum())))
    save(pd.DataFrame(exact), 'exact_edit_subset_denominators.csv')
    return (aggregate, coverage)

def examples(result, graphs, blocks, values, sources):
    candidates = result[result.any_individual20_but_not_joint20].copy()
    r8_lookup = round8_local_quartets()
    candidates['round8_selected_molecular_quartet_ids'] = [packed(r8_lookup.get(tuple([row.dataset_id, row.molecule_a, row.molecule_b, row.molecule_c, row.molecule_d]), [])) for row in candidates.itertuples(index=False)]
    candidates['was_round8_selected_molecular_quartet'] = candidates.round8_selected_molecular_quartet_ids.ne('[]')
    rank = ['maximum_event_threshold_clearance', 'maximum_event_joint_deficit', 'has_exact_edit_either_side', 'has_exact_edit_both_sides', 'n_named_molecules']
    candidates = candidates.sort_values(rank, ascending=False)
    chosen = [candidates.drop_duplicates('rxn_class').assign(selection_reason='strongest_threshold_clearance_in_reaction_class')]
    exact = candidates[candidates.has_exact_edit_either_side]
    if len(exact):
        chosen.append(exact.drop_duplicates('rxn_class').assign(selection_reason='strongest_clearance_with_local_edit_in_class'))
    both = candidates[candidates.has_exact_edit_both_sides]
    if len(both):
        chosen.append(both.drop_duplicates('rxn_class').assign(selection_reason='strongest_clearance_with_both_reactants_locally_edited'))
    r8 = candidates[candidates.was_round8_selected_molecular_quartet]
    if len(r8):
        chosen.append(r8.drop_duplicates('grid_id').assign(selection_reason='shared_route_failure_on_round8_selected_molecular_quartet'))
    selected = pd.concat(chosen, ignore_index=True).drop_duplicates('acquisition_quartet_id').copy()
    selected['case_id'] = 's_' + selected.acquisition_quartet_id.map(prior.sid)
    source_groups = {key: part for key, part in sources.groupby('block_id', sort=False)}
    rows, paths, nodes, links, path_metrics = ([], [], [], [], [])
    for case in selected.to_dict('records'):
        graph = graphs[case['dataset_id']]
        recipes = json.loads(case['common_recipe_ids'])
        common = [graph['lookup'][recipe] for recipe in recipes]
        adjacent = graph['adjacent'][np.ix_(common, common)]
        ys = np.array([values[case['block_id_' + side]][common] for side in SIDES])
        i, j = [recipes.index(case['representative_endpoint' + str(end) + '_recipe']) for end in [0, 1]]
        endpoint_diff = np.flatnonzero(graph['tokens'][common[i]] != graph['tokens'][common[j]])
        case['n_endpoint_slot_differences'] = len(endpoint_diff)
        case['endpoint_changed_slots'] = packed([prior.SLOTS[index] for index in endpoint_diff])
        assert len(endpoint_diff) >= 2
        joint_y = ys.min(0)
        joint_labels = induced_labels(joint_y, adjacent)
        assert joint_labels[i] != joint_labels[j] and joint_labels[i] >= 0 and (joint_labels[j] >= 0)
        case['joint_high20_component_endpoint0'] = int(joint_labels[i])
        case['joint_high20_component_endpoint1'] = int(joint_labels[j])
        case['n_joint_high20_recipes'] = int((joint_labels >= 0).sum())
        for z, side in enumerate(SIDES):
            case['endpoint0_yield_' + side], case['endpoint1_yield_' + side] = (float(ys[z, i]), float(ys[z, j]))
            target_labels = induced_labels(ys[z], adjacent)
            assert target_labels[i] == target_labels[j] and target_labels[i] >= 0
            source = source_groups[case['block_id_' + side]]
            source = source[source.recipe_id.isin(recipes)].copy()
            source['case_id'], source['measured_side'] = (case['case_id'], side)
            links.append(source)
        for optimized_for in [*SIDES, 'joint']:
            objective = joint_y if optimized_for == 'joint' else ys[SIDES.index(optimized_for)]
            expected = case['representative_K_joint'] if optimized_for == 'joint' else case['representative_K_' + optimized_for]
            heap_path = prior.molecular.widest_path(objective, adjacent, i, j)
            assert heap_path and abs(float(objective[heap_path].min()) - expected) <= EPS
            path = shortest_optimal_path(objective, adjacent, i, j, expected)
            assert path and path[0] == i and (path[-1] == j)
            assert abs(float(objective[path].min()) - expected) <= EPS
            independent = prior.mst_capacity(objective, adjacent)
            assert abs(float(independent[i, j]) - expected) <= EPS
            case['path_recipe_ids_' + optimized_for] = packed([recipes[position] for position in path])
            case['path_own_objective_yields_' + optimized_for] = packed(objective[path].tolist())
            path_metrics.append(dict(case_id=case['case_id'], path_optimized_for=optimized_for, n_substitutions=len(path) - 1, own_objective_bottleneck=float(objective[path].min()), minimum_over_all_four_targets_on_this_same_path=float(ys[:, path].min()), **{'bottleneck_' + side: float(ys[z, path].min()) for z, side in enumerate(SIDES)}))
            for step, position in enumerate(path):
                recipe = recipes[position]
                changed, old, new = ('', '', '')
                if step:
                    previous_recipe = recipes[path[step - 1]]
                    diff = np.flatnonzero(graph['tokens'][common[path[step - 1]]] != graph['tokens'][common[position]])
                    assert len(diff) == 1
                    changed = prior.SLOTS[diff[0]]
                    old, new = (graph['labels'][previous_recipe][changed], graph['labels'][recipe][changed])
                paths.append(dict(case_id=case['case_id'], acquisition_quartet_id=case['acquisition_quartet_id'], path_optimized_for=optimized_for, path_step=step, recipe_id=recipe, full_recipe_text=graph['full'][recipe], changed_slot=changed, previous_component=old, new_component=new, minimum_yield_across_four_targets=float(joint_y[position]), **{'yield_' + side: float(ys[z, position]) for z, side in enumerate(SIDES)}))
        for position, recipe in enumerate(recipes):
            nodes.append(dict(case_id=case['case_id'], recipe_id=recipe, full_recipe_text=graph['full'][recipe], is_endpoint0=position == i, is_endpoint1=position == j, minimum_yield_across_four_targets=float(joint_y[position]), joint_high20_component=int(joint_labels[position]), **{'yield_' + side: float(ys[z, position]) for z, side in enumerate(SIDES)}))
        for letter in ['a', 'b', 'c', 'd']:
            name = prior.DISPLAY_NAMES.get(case['molecule_' + letter])
            if name:
                case['molecule_name_' + letter] = name
        rows.append(case)
    selected = pd.DataFrame(rows)
    path_frame, node_frame, path_metric_frame = (pd.DataFrame(paths), pd.DataFrame(nodes), pd.DataFrame(path_metrics))
    selected_sources = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
    save(selected, 'selected_cross_class_shared_route_examples.csv')
    save(path_frame, 'selected_five_paths_actual_yields_and_substitutions.csv')
    save(path_metric_frame, 'selected_five_path_bottleneck_comparisons.csv')
    save(node_frame, 'selected_all_common_recipe_node_yields.csv')
    save(selected_sources, 'selected_all_common_node_original_records.csv.gz')
    if len(selected_sources):
        cell_means = selected_sources.groupby(['case_id', 'measured_side', 'recipe_id']).yield_numeric.mean()
        for record in nodes:
            for side in SIDES:
                assert abs(cell_means.loc[record['case_id'], side, record['recipe_id']] - record['yield_' + side]) <= EPS
    return (selected, path_frame, node_frame, selected_sources, path_metric_frame)

def subset_shared_routes(result, graphs, blocks, values, sources):
    """Threshold-only subset connectivity on the complete base-event support."""
    out_rows, candidates = ([], [])
    columns = ['acquisition_quartet_id', 'endpoint0_recipe', 'endpoint1_recipe', 'individual20_but_not_joint20']
    columns += ['shared20_' + name for name in SUBSET_NAMES]
    columns += ['all_six_pairs_shared20', 'all_four_triples_shared20', 'all_four_shared20', 'all_pairs_shared_but_four_not20', 'all_triples_shared_but_four_not20', 'minimum_blocking_target_count']
    processed_endpoint_rows = 0
    with gzip.open(OUT / 'subset_shared_route_endpoint_flags.csv.gz', 'wt', encoding='utf-8-sig', newline='', compresslevel=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for number, original in enumerate(result.to_dict('records'), 1):
            row = {key: original[key] for key in ['acquisition_quartet_id', 'grid_id', 'dataset_id', 'rxn_class', 'comparability_status', 'has_exact_edit_either_side', 'has_exact_edit_both_sides']}
            row['n_all_eligible_endpoint_pairs'] = int(original['n_common_high_endpoint_pairs_with_full_graph_path'])
            row['n_base_failure_endpoint_pairs'] = int(original['n_pairs_individual20_but_not_joint20'])
            row['subset_connectivity_evaluated'] = bool(original['any_individual20_but_not_joint20'])
            for event in SUBSET_EVENTS:
                row['n_pairs_' + event] = 0
                row['any_' + event] = False
            if row['subset_connectivity_evaluated']:
                graph = graphs[original['dataset_id']]
                recipes = json.loads(original['common_recipe_ids'])
                common = [graph['lookup'][recipe] for recipe in recipes]
                adjacent = graph['adjacent'][np.ix_(common, common)]
                ys = np.array([values[original['block_id_' + side]][common] for side in SIDES])
                _, full_groups = connected_components(csr_matrix(adjacent), directed=False)
                high = np.flatnonzero((ys >= 0.2).all(0))
                ends = [(i, j) for i, j in combinations(high, 2) if full_groups[i] == full_groups[j]]
                assert len(ends) == row['n_all_eligible_endpoint_pairs']
                ii, jj = np.asarray(ends, int).T
                labels_individual = [induced_labels(ys[index], adjacent) for index in range(4)]
                all_individual = np.logical_and.reduce([label[ii] == label[jj] for label in labels_individual])
                labels_joint = induced_labels(ys.min(0), adjacent)
                all_four = labels_joint[ii] == labels_joint[jj]
                subset_connections = np.array([(lambda label: label[ii] == label[jj])(induced_labels(ys[list(subset)].min(0), adjacent)) for subset in SUBSETS])
                all_pairs = subset_connections[:6].all(0)
                all_triples = subset_connections[6:].all(0)
                assert not (all_triples & ~all_pairs).any()
                assert not (all_four & ~all_triples).any()
                assert not (all_pairs & ~all_individual).any()
                base = all_individual & ~all_four
                assert int(base.sum()) == row['n_base_failure_endpoint_pairs']
                pairs_not_four = all_pairs & ~all_four
                triples_not_four = all_triples & ~all_four
                order = np.where(base, np.where(all_triples, 4, np.where(all_pairs, 3, 2)), 0)
                events = dict(all_pairs_shared_but_four_not20=pairs_not_four, all_triples_shared_but_four_not20=triples_not_four, minimum_blocking_target_count2=order == 2, minimum_blocking_target_count3=order == 3, minimum_blocking_target_count4=order == 4)
                assert sum((int(events[f'minimum_blocking_target_count{k}'].sum()) for k in [2, 3, 4])) == int(base.sum())
                for event, flag in events.items():
                    row['n_pairs_' + event], row['any_' + event] = (int(flag.sum()), bool(flag.any()))
                for required_order in [3, 4]:
                    eligible = np.flatnonzero(order == required_order)
                    if len(eligible):
                        min_endpoints = np.minimum(ys[:, ii[eligible]], ys[:, jj[eligible]]).min(0)
                        slot_distance = (graph['tokens'][np.array(common)[ii[eligible]]] != graph['tokens'][np.array(common)[jj[eligible]]]).sum(1)
                        chosen = max(range(len(eligible)), key=lambda z: (-int(slot_distance[z]), float(min_endpoints[z])))
                        z = int(eligible[chosen])
                        candidate = dict(original, minimum_blocking_target_count=required_order, subset_endpoint0_recipe=recipes[ii[z]], subset_endpoint1_recipe=recipes[jj[z]], subset_minimum_all_eight_endpoint_yields=float(min_endpoints[chosen]), subset_n_endpoint_slot_differences=int(slot_distance[chosen]))
                        candidates.append(candidate)
                writer.writerows(([original['acquisition_quartet_id'], recipes[i], recipes[j], bool(base[z]), *subset_connections[:, z].tolist(), bool(all_pairs[z]), bool(all_triples[z]), bool(all_four[z]), bool(pairs_not_four[z]), bool(triples_not_four[z]), int(order[z])] for z, (i, j) in enumerate(ends)))
                processed_endpoint_rows += len(ends)
            out_rows.append(row)
            if number % 15000 == 0:
                print('SUBSET_QUARTETS', number, 'ENDPOINT_ROWS', processed_endpoint_rows, flush=True)
    quartets = pd.DataFrame(out_rows)
    save(quartets, 'subset_shared_route_quartet_summary.csv.gz')
    aggregate = []
    for grouping in ['all', 'dataset_id', 'rxn_class']:
        groups = [('all', quartets)] if grouping == 'all' else quartets.groupby(grouping, sort=True)
        for label, part in groups:
            for event in SUBSET_EVENTS:
                active = part[part['any_' + event]]
                aggregate.append(dict(grouping=grouping, group=str(label), metric=event, denominator_all_endpoint_pairs=int(part.n_all_eligible_endpoint_pairs.sum()), denominator_base_failure_endpoint_pairs=int(part.n_base_failure_endpoint_pairs.sum()), n_event_endpoint_pairs=int(part['n_pairs_' + event].sum()), n_event_quartets=len(active), n_event_molecular_grids=active.grid_id.nunique(), n_event_classes=active.rxn_class.nunique()))
    aggregate = pd.DataFrame(aggregate)
    save(aggregate, 'subset_shared_route_full_denominator_counts.csv')
    coverage = read(SOURCE / 'scope_dataset_coverage.csv')
    for event in SUBSET_EVENTS:
        counts = quartets.groupby('dataset_id')['n_pairs_' + event].sum()
        coverage['n_endpoint_pairs_' + event] = coverage.dataset_id.map(counts).fillna(0).astype(int)
        counts = quartets[quartets['any_' + event]].groupby('dataset_id').grid_id.nunique()
        coverage['n_molecular_grids_' + event] = coverage.dataset_id.map(counts).fillna(0).astype(int)
    assert len(coverage) == 20 and coverage.rxn_class.nunique() == 18
    save(coverage, 'subset_shared_route_all_20_dataset_coverage.csv')
    selected = pd.DataFrame(candidates)
    if len(selected):
        selected = selected.sort_values(['minimum_blocking_target_count', 'has_exact_edit_both_sides', 'has_exact_edit_either_side', 'n_named_molecules', 'subset_n_endpoint_slot_differences', 'subset_minimum_all_eight_endpoint_yields'], ascending=[False, False, False, False, True, False])
        selected = selected.drop_duplicates(['rxn_class', 'minimum_blocking_target_count']).copy()
    selected = subset_examples(selected, graphs, values, sources)
    expected = {event: int(quartets['n_pairs_' + event].sum()) for event in SUBSET_EVENTS}
    observed = Counter()
    readback_rows = 0
    for chunk in read(OUT / 'subset_shared_route_endpoint_flags.csv.gz', chunksize=100000):
        assert np.array_equal(chunk[['shared20_' + name for name in SUBSET_NAMES[:6]]].all(axis=1), chunk.all_six_pairs_shared20)
        assert np.array_equal(chunk[['shared20_' + name for name in SUBSET_NAMES[6:]]].all(axis=1), chunk.all_four_triples_shared20)
        assert np.array_equal(chunk.all_six_pairs_shared20 & ~chunk.all_four_shared20, chunk.all_pairs_shared_but_four_not20)
        assert np.array_equal(chunk.all_four_triples_shared20 & ~chunk.all_four_shared20, chunk.all_triples_shared_but_four_not20)
        observed['all_pairs_shared_but_four_not20'] += int(chunk.all_pairs_shared_but_four_not20.sum())
        observed['all_triples_shared_but_four_not20'] += int(chunk.all_triples_shared_but_four_not20.sum())
        for order in [2, 3, 4]:
            observed['minimum_blocking_target_count' + str(order)] += int(chunk.minimum_blocking_target_count.eq(order).sum())
        readback_rows += len(chunk)
    assert dict(observed) == expected and readback_rows == processed_endpoint_rows
    summary = dict(status='passed', denominator_all_endpoint_pairs=7853943, denominator_base_failure_endpoint_pairs=int(quartets.n_base_failure_endpoint_pairs.sum()), n_evaluated_quartets=int(quartets.subset_connectivity_evaluated.sum()), n_event_quartet_endpoint_rows=processed_endpoint_rows, counts=aggregate[aggregate.grouping.eq('all')].to_dict('records'), all_rows_read_back=True, all_subset_monotonicity_verified=True, selected_examples=len(selected))
    (OUT / 'subset_shared_route_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('SUBSET_COMPLETED', packed(summary), flush=True)
    return summary

def subset_threshold_sensitivity(result, graphs, values):
    """Recompute subset events at nearby thresholds within the retained 20% cohort.

    This is deliberately conditional: only the 2,464 quartets retained because they
    contain a 20% four-target failure are evaluated.  New quartets that would enter
    at another threshold are outside this sensitivity analysis.
    """
    accum = {threshold: dict(threshold_fraction=threshold, threshold_percent=100 * threshold, cohort_quartets=0, cohort_endpoint_pairs=0, threshold_eligible_endpoint_pairs=0, n_event_endpoint_pairs=0, n_event_quartets=0, n_event_molecular_grids=0, n_event_classes=0, all_pairs_shared_but_four_not_threshold=0, all_triples_shared_but_four_not_threshold=0, minimum_blocking_target_count2=0, minimum_blocking_target_count3=0, minimum_blocking_target_count4=0, _grids=set(), _classes=set(), _quartets=set()) for threshold in SUBSET_SENSITIVITY_THRESHOLDS}
    processed_quartets = 0
    for original in result.to_dict('records'):
        if not bool(original['any_individual20_but_not_joint20']):
            continue
        graph = graphs[original['dataset_id']]
        recipes = json.loads(original['common_recipe_ids'])
        common = [graph['lookup'][recipe] for recipe in recipes]
        adjacent = graph['adjacent'][np.ix_(common, common)]
        ys = np.array([values[original['block_id_' + side]][common] for side in SIDES])
        _, full_groups = connected_components(csr_matrix(adjacent), directed=False)
        high20 = np.flatnonzero((ys >= 0.2).all(0))
        ends = [(i, j) for i, j in combinations(high20, 2) if full_groups[i] == full_groups[j]]
        if not ends:
            continue
        processed_quartets += 1
        ii, jj = np.asarray(ends, int).T
        for threshold, row in accum.items():
            row['cohort_quartets'] += 1
            row['cohort_endpoint_pairs'] += len(ends)
            eligible = (ys[:, ii] >= threshold - EPS).all(0) & (ys[:, jj] >= threshold - EPS).all(0)
            row['threshold_eligible_endpoint_pairs'] += int(eligible.sum())
            labels_individual = [induced_labels(ys[index], adjacent, threshold) for index in range(4)]
            individual = np.array([(label[ii] >= 0) & (label[ii] == label[jj]) for label in labels_individual])
            all_individual = individual.all(0)
            labels_joint = induced_labels(ys.min(0), adjacent, threshold)
            all_four = (labels_joint[ii] >= 0) & (labels_joint[ii] == labels_joint[jj])
            subset_connections = np.array([(lambda label: (label[ii] >= 0) & (label[ii] == label[jj]))(induced_labels(ys[list(subset)].min(0), adjacent, threshold)) for subset in SUBSETS])
            all_pairs = subset_connections[:6].all(0)
            all_triples = subset_connections[6:].all(0)
            assert not (all_triples & ~all_pairs).any()
            assert not (all_four & ~all_triples).any()
            base = all_individual & ~all_four
            assert not (base & ~eligible).any()
            pairs_not_four = all_pairs & ~all_four
            triples_not_four = all_triples & ~all_four
            order = np.where(base, np.where(all_triples, 4, np.where(all_pairs, 3, 2)), 0)
            row['n_event_endpoint_pairs'] += int(base.sum())
            row['all_pairs_shared_but_four_not_threshold'] += int(pairs_not_four.sum())
            row['all_triples_shared_but_four_not_threshold'] += int(triples_not_four.sum())
            for order_value in [2, 3, 4]:
                row['minimum_blocking_target_count' + str(order_value)] += int((order == order_value).sum())
            if base.any():
                row['_quartets'].add(original['acquisition_quartet_id'])
                row['_grids'].add(original['grid_id'])
                row['_classes'].add(original['rxn_class'])
    rows = []
    for threshold in SUBSET_SENSITIVITY_THRESHOLDS:
        row = accum[threshold]
        row['n_event_quartets'] = len(row.pop('_quartets'))
        row['n_event_molecular_grids'] = len(row.pop('_grids'))
        row['n_event_classes'] = len(row.pop('_classes'))
        rows.append(row)
    sensitivity = pd.DataFrame(rows)
    row20 = sensitivity.loc[np.isclose(sensitivity.threshold_fraction, 0.2)].iloc[0]
    assert int(row20.n_event_endpoint_pairs) == 68367
    assert int(row20.minimum_blocking_target_count2) == 66945
    assert int(row20.minimum_blocking_target_count3) == 1401
    assert int(row20.minimum_blocking_target_count4) == 21
    save(sensitivity, 'subset_shared_route_threshold_sensitivity.csv')
    summary = dict(status='passed', cohort='quartets with any 20% individual-but-not-joint failure', n_cohort_quartets=processed_quartets, n_cohort_endpoint_pairs=int(sensitivity.cohort_endpoint_pairs.max()), thresholds=[float(value) for value in SUBSET_SENSITIVITY_THRESHOLDS], rows=sensitivity.to_dict('records'))
    (OUT / 'subset_shared_route_threshold_sensitivity.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('SUBSET_SENSITIVITY', packed(summary), flush=True)
    return summary

def subset_report(subset_summary, sensitivity_summary, all_quartets):
    """Write prose explaining the subset obstruction and its conditional sensitivity."""
    selected = read(OUT / 'selected_subset_obstruction_examples.csv')
    flags = read(OUT / 'subset_shared_route_endpoint_flags.csv.gz')
    order4_counts = flags[flags.minimum_blocking_target_count.eq(4)].groupby('acquisition_quartet_id').size().to_dict()
    counts = pd.DataFrame(subset_summary['counts'])
    order4_summary = counts[counts.metric.eq('minimum_blocking_target_count4')].iloc[0]
    cohort_rows = int(subset_summary['n_event_quartet_endpoint_rows'])
    cohort_quartets = int(subset_summary['n_evaluated_quartets'])
    order4_rows = int(order4_summary.n_event_endpoint_pairs)
    lines = ['# Subset connectivity within the four-target shared-route failures', '', 'This layer asks how many target constraints are needed before one >=20% shared recipe-substitution route disappears. It uses the same measured recipe graph, where each edge changes one complete condition slot. The subset route is a graph comparison between measured recipes; it is not a four-factor yield interaction and not a sequential experimental procedure.', '', f"The conditional cohort contains the {cohort_quartets:,} acquisition quartets that have at least one main 20% event. Across those quartets, {cohort_rows:,} eligible endpoint-pair rows were evaluated. The parent denominator remains {int(subset_summary['denominator_all_endpoint_pairs']):,} endpoint views; this subset layer does not add quartets that would become events only at another threshold.", '', '## 20% partition', '', '| Subset result | Endpoint-pair views | Quartets | Molecular grids | Classes |', '|---|---:|---:|---:|---:|']
    labels = {'all_pairs_shared_but_four_not20': 'All six pairs connect; all four fail', 'all_triples_shared_but_four_not20': 'All four triples connect; all four fail', 'minimum_blocking_target_count2': 'Minimum blocking target count = 2', 'minimum_blocking_target_count3': 'Minimum blocking target count = 3', 'minimum_blocking_target_count4': 'Minimum blocking target count = 4'}
    for metric in labels:
        row = counts[counts.metric.eq(metric)].iloc[0]
        lines.append(f'| {labels[metric]} | {int(row.n_event_endpoint_pairs):,} | {int(row.n_event_quartets):,} | {int(row.n_event_molecular_grids):,} | {int(row.n_event_classes)} |')
    lines += ['', f"The minimum blocking count is assigned only within the {int(subset_summary['denominator_base_failure_endpoint_pairs']):,} main-event endpoint rows: count 2 means at least one pair already fails, count 3 means all pairs connect but at least one triple fails, and count 4 means all pairs and triples connect while the four-target route fails. The counts partition the main event exactly.", '', '## Threshold sensitivity', '', f'The following table is conditional on the same retained cohort of {cohort_quartets:,} quartets and {cohort_rows:,} endpoint rows. At each threshold, endpoints and paths are re-evaluated on the original measured graph; threshold-specific events in the other {all_quartets - cohort_quartets:,} quartets are intentionally outside this table.', '', '| Threshold | Endpoint rows eligible at threshold | Individual-but-not-joint endpoint rows | Blocking 2 | Blocking 3 | Blocking 4 |', '|---:|---:|---:|---:|---:|---:|']
    for row in sensitivity_summary['rows']:
        lines.append(f"| {row['threshold_percent']:.0f}% | {int(row['threshold_eligible_endpoint_pairs']):,} | {int(row['n_event_endpoint_pairs']):,} | {int(row['minimum_blocking_target_count2']):,} | {int(row['minimum_blocking_target_count3']):,} | {int(row['minimum_blocking_target_count4']):,} |")
    lines += ['', f'The 20% sensitivity row reproduces the streamed subset partition. All retained rows are already >=20% endpoint pairs, so the 10/15/20% denominators remain {cohort_rows:,}; at 25/30%, some endpoint pairs no longer meet the higher endpoint threshold. Threshold-specific events in this table are therefore conditional on the fixed cohort.', '', f'## The {int(order4_summary.n_event_molecular_grids):,} order-4 molecular grids', '', f'Exactly {order4_rows:,} endpoint rows ({int(order4_summary.n_event_quartets):,} acquisition quartets and {int(order4_summary.n_event_molecular_grids):,} molecular grids) remain connected for every pair and every triple but fail for all four targets. Canonical SMILES are reproduced from the source tables; recipe IDs identify the two endpoints of each obstruction.']
    order4 = selected[selected.minimum_blocking_target_count.eq(4)]
    for case in order4.to_dict('records'):
        qcount = order4_counts.get(case['acquisition_quartet_id'], 0)
        lines += ['', f"### {case['rxn_class']}: `{case['case_id']}` ({qcount} of the {order4_rows:,} endpoint rows)", '', f"Grid `{case['grid_id']}`; quartet `{case['acquisition_quartet_id']}`; endpoint recipes `{case['subset_endpoint0_recipe']}` -> `{case['subset_endpoint1_recipe']}`.", f"A `{case['molecule_a']}`; B `{case['molecule_b']}`; C `{case['molecule_c']}`; D `{case['molecule_d']}`.", f"The four individually optimized bottlenecks are AC/BC/AD/BD = {100 * case['subset_K_ac']:.5f}% / {100 * case['subset_K_bc']:.5f}% / {100 * case['subset_K_ad']:.5f}% / {100 * case['subset_K_bd']:.5f}%. The best one-route bottleneck is **{100 * case['subset_K_joint']:.5f}%**, so the deficit is {100 * case['subset_joint_deficit']:.5f} percentage points.", f"The minimum of the eight endpoint yields is {100 * case['subset_minimum_all_eight_endpoint_yields']:.5f}%; therefore the route failure is a graph bottleneck between high measured endpoints, not an endpoint below-threshold artifact."]
    lines += ['', ' '.join((f"The {case['rxn_class']} case has a best shared four-target route of {100 * case['subset_K_joint']:.5f}%, {100 * (0.2 - case['subset_K_joint']):.5f} percentage points below 20%, while each pair and triple still has a >=20% route." for case in order4.to_dict('records'))), '', '## Files', '', f'- `subset_shared_route_endpoint_flags.csv.gz`: endpoint-level subset labels for all {cohort_rows:,} retained cohort rows.', f'- `subset_shared_route_quartet_summary.csv.gz`: all {all_quartets:,} quartets with explicit zeros and an evaluation flag.', '- `subset_shared_route_full_denominator_counts.csv`: complete subset counts by dataset and reaction class.', '- `subset_shared_route_threshold_sensitivity.csv`: conditional 10/15/20/25/30% sensitivity table.', '- `selected_subset_obstruction_examples.csv`, `selected_all_ten_subset_high_routes_and_best_four_route.csv`, `selected_all_ten_subset_route_statuses.csv`: selected chemistry, routes and statuses.', '', 'All endpoint views can reuse molecules, recipes and source records; they are comparison views rather than independent experimental replicates.', '']
    (OUT / 'SUBSET_SHARED_ROUTES_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def subset_examples(selected, graphs, values, sources):
    case_rows, path_rows, status_rows, node_rows, source_rows = ([], [], [], [], [])
    source_lookup = {key: part for key, part in sources.groupby('block_id', sort=False)}
    for case in selected.to_dict('records'):
        case['case_id'] = 'u_' + prior.sid([case['acquisition_quartet_id'], case['minimum_blocking_target_count'], case['subset_endpoint0_recipe'], case['subset_endpoint1_recipe']])
        graph = graphs[case['dataset_id']]
        recipes = json.loads(case['common_recipe_ids'])
        common = [graph['lookup'][recipe] for recipe in recipes]
        adjacent = graph['adjacent'][np.ix_(common, common)]
        ys = np.array([values[case['block_id_' + side]][common] for side in SIDES])
        i, j = (recipes.index(case['subset_endpoint0_recipe']), recipes.index(case['subset_endpoint1_recipe']))
        cap = np.array([prior.molecular.widest_matrix(y, adjacent)[i, j] for y in ys])
        joint_y = ys.min(0)
        joint = prior.molecular.widest_matrix(joint_y, adjacent)[i, j]
        case['subset_K_joint'] = float(joint)
        case['subset_min_individual_K'] = float(cap.min())
        case['subset_joint_deficit'] = float(cap.min() - joint)
        for z, side in enumerate(SIDES):
            case['subset_K_' + side] = float(cap[z])
            source = source_lookup[case['block_id_' + side]]
            source = source[source.recipe_id.isin(recipes)].copy()
            source['case_id'], source['measured_side'] = (case['case_id'], side)
            source_rows.append(source)
        for subset, name in [(tuple(range(4)), 'all_four_best'), *list(zip(SUBSETS, SUBSET_NAMES))]:
            objective = ys[list(subset)].min(0)
            labels = induced_labels(objective, adjacent)
            connected = labels[i] == labels[j] and labels[i] >= 0
            status_rows.append(dict(case_id=case['case_id'], subset=name, n_targets=len(subset), shared_high20_path_exists=bool(connected), target_sides=packed([SIDES[z] for z in subset])))
            if name == 'all_four_best':
                floor = float(joint)
            elif connected:
                floor = 0.2
            else:
                continue
            path = shortest_optimal_path(objective, adjacent, i, j, floor) if name == 'all_four_best' else None
            if path is None:
                kept = np.flatnonzero(objective >= 0.2 - EPS)
                positions = {int(node): z for z, node in enumerate(kept)}
                _, previous = shortest_path(csr_matrix(adjacent[np.ix_(kept, kept)]), directed=False, indices=positions[i], unweighted=True, return_predecessors=True)
                local_path = [positions[j]]
                while local_path[-1] != positions[i]:
                    local_path.append(int(previous[local_path[-1]]))
                    assert local_path[-1] >= 0
                path = kept[local_path[::-1]].tolist()
                assert float(objective[path].min()) >= 0.2 - EPS
            for step, node in enumerate(path):
                old, new, changed = ('', '', '')
                if step:
                    diff = np.flatnonzero(graph['tokens'][common[path[step - 1]]] != graph['tokens'][common[node]])
                    assert len(diff) == 1
                    changed = prior.SLOTS[diff[0]]
                    old, new = (graph['labels'][recipes[path[step - 1]]][changed], graph['labels'][recipes[node]][changed])
                path_rows.append(dict(case_id=case['case_id'], subset=name, path_step=step, recipe_id=recipes[node], changed_slot=changed, previous_component=old, new_component=new, full_recipe_text=graph['full'][recipes[node]], subset_path_actual_bottleneck=float(objective[path].min()), **{'yield_' + side: float(ys[z, node]) for z, side in enumerate(SIDES)}))
        for node, recipe in enumerate(recipes):
            node_rows.append(dict(case_id=case['case_id'], recipe_id=recipe, is_endpoint0=node == i, is_endpoint1=node == j, full_recipe_text=graph['full'][recipe], **{'yield_' + side: float(ys[z, node]) for z, side in enumerate(SIDES)}))
        case_rows.append(case)
    cases = pd.DataFrame(case_rows)
    save(cases, 'selected_subset_obstruction_examples.csv')
    save(pd.DataFrame(path_rows), 'selected_all_ten_subset_high_routes_and_best_four_route.csv')
    save(pd.DataFrame(status_rows), 'selected_all_ten_subset_route_statuses.csv')
    save(pd.DataFrame(node_rows), 'selected_subset_obstruction_all_common_nodes.csv')
    save(pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame(), 'selected_subset_obstruction_original_records.csv.gz')
    return cases

def verify_endpoint_readback(result):
    counts = Counter()
    per_quartet = Counter()
    max_inequality_error = 0.0
    total = 0
    for chunk in read(OUT / 'all_7994464_endpoint_shared_route_metrics.csv.gz', chunksize=100000):
        cap = chunk[['K_' + side for side in SIDES]].to_numpy(float).T
        minimum, deficit, clearance, flags = metrics(cap, chunk.K_joint.to_numpy(float))
        assert np.allclose(minimum, chunk.min_individual_K, atol=EPS)
        assert np.allclose(deficit, chunk.joint_deficit, atol=EPS)
        assert np.allclose(clearance, chunk.threshold20_clearance, atol=EPS)
        assert np.allclose(chunk[['lower_endpoint_yield_' + side for side in SIDES]].min(axis=1), chunk.minimum_all_eight_endpoint_yields, atol=EPS)
        for flag in FLAGS:
            assert np.array_equal(flags[flag], chunk[flag])
            counts[flag] += int(chunk[flag].sum())
        for quartet, n in chunk.acquisition_quartet_id.value_counts().items():
            per_quartet[quartet] += int(n)
        total += len(chunk)
        max_inequality_error = max(max_inequality_error, float((chunk.K_joint - chunk.min_individual_K).max()))
    assert total == 7853943
    for flag in FLAGS:
        assert counts[flag] == int(result['n_pairs_' + flag].sum())
    assert np.array_equal(result.acquisition_quartet_id.map(per_quartet).fillna(0), result.n_common_high_endpoint_pairs_with_full_graph_path)
    return dict(endpoint_rows_read_back=total, endpoint_flags_and_continuous_metrics_verified=True, every_quartet_endpoint_count_verified=True, maximum_joint_greater_than_individual_violation=max_inequality_error)

def report(result, grid, aggregate, coverage, selected, paths, nodes, source, path_metrics, checks, started, mapping, norm, target, blocks):
    all_rows = aggregate[aggregate.grouping.eq('all')]
    study = json.loads((ROOT / 'results/mining_summary.json').read_text(encoding='utf-8'))
    summary = dict(status='completed', study_datasets=int(study['matched_dataset_count']), study_reaction_classes=int(mapping.rxn_class.nunique()), study_legacy_swaps=int(study['matched_swaps']), strict_source_rows=len(norm), strict_targets=len(target), strict_acquisition_blocks=len(blocks), all_acquisition_quartets=len(result), all_molecular_grids=len(grid), endpoint_comparable_quartets=int(result.comparability_status.eq('endpoint_comparable').sum()), endpoint_comparable_grids=int(grid.n_endpoint_comparable_quartets.gt(0).sum()), endpoint_pair_views=int(result.n_common_high_endpoint_pairs_with_full_graph_path.sum()), support_statuses=result.comparability_status.value_counts().to_dict(), aggregation=all_rows.to_dict('records'), selected_cases=len(selected), selected_paths=5 * len(selected), selected_source_memberships=len(source), selected_distinct_original_records=source.record_id.nunique() if len(source) else 0, elapsed_seconds=time.monotonic() - started)
    verification = dict(status='passed', verified_at_utc=datetime.now(timezone.utc).isoformat(), method='Floyd max-min full matrices matched independently to SciPy maximum spanning forests; induced 20% node connectivity; complete streamed endpoint readback; selected five paths and original record means', checks=dict(checks))
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    event_rows = all_rows[all_rows.metric.eq('individual20_but_not_joint20')].iloc[0]
    lines = ['# Four individually usable replacement routes need not admit one shared route', '', f'Across the complete four-reactant-pair common recipe graphs, {int(event_rows.n_event_endpoint_pairs):,} endpoint-pair views connect at >=20% for every target individually, yet no identical recipe-substitution path between the same two endpoints stays >=20% for all four targets. These views occur in {int(event_rows.n_event_grids):,} molecular grids, {int(event_rows.n_event_quartets):,} acquisition quartets and {int(event_rows.n_event_classes)} reaction classes.', '', 'Each target can therefore retain a high-yield route between the same two successful recipes while the intermediate recipe choices remain incompatible across targets. The graph nodes are measured reaction recipes and edges change exactly one complete condition slot. The path is an order for replacing independently measured recipes, not a sequence of reaction intermediates or an experimentally executed sequential-addition procedure.', '', '## Complete source and denominator', '', f"The parent study remains {summary['study_datasets']:,} datasets, {summary['study_reaction_classes']:,} reaction classes and {summary['study_legacy_swaps']:,} background-specific single-slot exchanges. The analysis starts from all {summary['strict_source_rows']:,} strict source records, {summary['strict_targets']:,} actual targets and {summary['strict_acquisition_blocks']:,} source-located target acquisition blocks. It reuses the complete round-7 enumeration of {summary['all_molecular_grids']:,} molecular 2x2 grids and {summary['all_acquisition_quartets']:,} acquisition quartets, without a molecular-edit or prior-event filter.", '', f"The two endpoints must be measured at >=20% for all four targets and belong to the same component of the four-target common measured graph. This gives all {summary['endpoint_pair_views']:,} eligible endpoint-pair views in {summary['endpoint_comparable_quartets']:,} acquisition quartets and {summary['endpoint_comparable_grids']:,} molecular grids. Other support states remain explicit: fewer than two common measured recipes, fewer than two common high recipes, or high endpoints separated in the full measured graph. A missing comparison is not counted as a zero deficit.", '', 'The four targets each retain one specified strict source block; their blocks need not be the same physical plate or acquisition. Different quartets and endpoint pairs reuse targets, molecules, recipes and source records. Counts describe these measured comparison views and are not independent experimental replicates.', '', '## Quantities and the new question', '', 'For endpoints s,t and target q, K_q = max_path min_{r on path} y_q(r). The worst of the four separately optimized routes is K_separate = min_q K_q. For one identical route required across targets, K_joint = max_path min_{q,r on path} y_q(r). Compute K_joint by assigning each recipe node z(r)=min_q y_q(r) and taking the widest path on the same common measured graph. Always K_joint <= K_separate.', '', 'The joint deficit is K_separate - K_joint. This is a loss in the best attainable route bottleneck when one common route is required, not a yield gain from a chemical transformation. A positive gap can exist on either side of 20%; the main event separately requires K_separate>=20% and K_joint<20%. The 5/10 pp gap criteria require a deficit >=5/10 pp. Threshold clearance g additionally requires K_separate>=20%+g and K_joint<=20%-g.', '', 'Round 7 measured all four K_q on common graphs but optimized the path separately for each target. Its checkerboards and contrasts of four optimized bottlenecks do not measure K_joint. This round changes the order of optimization and minimization while retaining exactly the same source support and endpoint cohort.', '', '## Full event and non-event counts', '', '| Definition | Endpoint-pair views | Acquisition quartets | Molecular grids | Reaction classes |', '|---|---:|---:|---:|---:|']
    for row in all_rows.itertuples(index=False):
        lines.append(f'| {row.metric} | {row.n_event_endpoint_pairs:,} | {row.n_event_quartets:,} | {row.n_event_grids:,} | {row.n_event_classes} |')
    lines += ['', f"All endpoint-pair counts use the {summary['endpoint_pair_views']:,}-view denominator; the machine-readable table includes the corresponding non-event counts. Dataset coverage retains all {summary['study_datasets']:,} rows, including zero events and absent comparable support. Exact-edit subsets are a separate annotation of the complete scan.", '', '## Concrete cross-reaction examples', '']
    for case in selected.to_dict('records'):
        lines += [f"### {case['rxn_class']}: {case['case_id']}", '', f"Grid `{case['grid_id']}`; acquisition quartet `{case['acquisition_quartet_id']}`.", '']
        for letter in ['a', 'b', 'c', 'd']:
            lines.append(f"{letter.upper()}: {case.get('molecule_name_' + letter, '')} `{case['molecule_' + letter]}`.")
        lines += ['', f"Common measured recipes: {case['n_common_measured_recipes']}; minimum of all eight endpoint yields: {100 * case['representative_minimum_all_eight_endpoint_yields']:.2f}%. Four separately optimized bottlenecks AC/BC/AD/BD: " + '/'.join((f"{100 * case['representative_K_' + side]:.2f}%" for side in SIDES)) + f". One shared route: **{100 * case['representative_K_joint']:.2f}%**; joint deficit: **{100 * case['representative_joint_deficit']:.2f} pp**.", '', '| Route optimized for | Substitutions | AC bottleneck/% | BC/% | AD/% | BD/% | Worst across targets/% |', '|---|---:|---:|---:|---:|---:|---:|']
        local = path_metrics[path_metrics.case_id.eq(case['case_id'])]
        for row in local.itertuples(index=False):
            vals = ' | '.join((f"{100 * getattr(row, 'bottleneck_' + side):.2f}" for side in SIDES))
            lines.append(f'| {row.path_optimized_for} | {row.n_substitutions} | {vals} | {100 * row.minimum_over_all_four_targets_on_this_same_path:.2f} |')
        lines += ['', 'One jointly optimal measured recipe path:', '']
        for row in paths[paths.case_id.eq(case['case_id']) & paths.path_optimized_for.eq('joint')].itertuples(index=False):
            vals = '/'.join((f"{100 * getattr(row, 'yield_' + side):.2f}" for side in SIDES))
            lines.append(f'- {row.path_step}. `{row.recipe_id}`: {row.full_recipe_text}; AC/BC/AD/BD={vals}%.')
        lines += ['', 'All four separately optimal paths, all common nodes and original source positions are retained in the accompanying tables.', '']
    lines += ['## Reproduction and files', '', 'Every used individual and joint all-pairs capacity matrix is independently checked against a SciPy maximum spanning forest. Direct connected components of each 20% induced node graph additionally check the threshold connectivity. Synthetic tests explicitly exhibit noncommutation and the identical-target zero-deficit case. The entire streamed endpoint file is read back, all continuous metrics and flags recomputed, and per-quartet totals reconciled. Selected paths first verify the floor with the established heap widest-path solver and maximum spanning forest, then use SciPy shortest_path within that optimal floor to choose the fewest substitutions. Each step is confirmed as one exact slot change. Selections separately retain the strongest events in each class, local-edit events, both-reactant local-edit events and any supported match to the molecular quartets displayed in round 8.', '', '- `all_7994464_endpoint_shared_route_metrics.csv.gz`: every eligible endpoint pair, four individual bottlenecks, joint bottleneck, joint deficit, endpoint yields and event/non-event flags. Yield quantities are fractions, with 0.05=5 pp.', '- `all_69128_acquisition_quartet_shared_route_summary.csv.gz`: every quartet, full support status, molecular/source identities, source counts, continuous extrema and all event counts.', '- `all_52598_molecular_grid_shared_route_summary.csv.gz`: all molecular grids and any-quartet events, without treating quartet views as independent.', '- `all_20_dataset_scope_events_and_zeros.csv`, `all_support_statuses.csv`, `all_event_and_non_event_denominators.csv`, `exact_edit_subset_denominators.csv`: complete scope, events, non-events and exact-edit annotations.', '- `all_common_support_source_block_recipe_counts.csv.gz`: unique common-support source block/recipe nodes and their raw-record multiplicities; the read-only round-7 strict-source table supplies record IDs and original file/sheet/row locations.', '- `selected_cross_class_shared_route_examples.csv`, `selected_five_paths_actual_yields_and_substitutions.csv`, `selected_five_path_bottleneck_comparisons.csv`: all structures, the same endpoints, four separately optimal paths and one jointly optimal path.', '- `selected_all_common_recipe_node_yields.csv`, `selected_all_common_node_original_records.csv.gz`: all measured common-node yields, joint-high components and original record IDs/file/sheet/row positions.', '- `SUBSET_SHARED_ROUTES_REPORT.md`, `subset_shared_route_threshold_sensitivity.csv`: subset obstruction counts, order-4 molecular grids and conditional 10/15/20/25/30% sensitivity within the retained main-event cohort.', '- `run_summary.json`, `verification.json`, `matrix_verification_counts.json`: derived counts and numerical checks.', '', 'Generator: `Analysis/code/round9_shared_recipe_paths.py`. `--postprocess` rebuilds selections/reports/readback from the full saved quartet results; no analysis seed or selected molecular subset controls the primary scan. All CSV reads use `keep_default_na=False, low_memory=False`. All outputs stay under Analysis; input tables and raw files are read-only.', '']
    (OUT / 'SHARED_RECIPE_PATHS_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    self_checks()
    if '--self-check' in sys.argv:
        print('SHARED_ROUTE_SELF_CHECKS_PASSED', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    mapping, norm, target, graphs, blocks, values, block_frame, scope, sources, source_counts = load()
    print('LOADED', len(scope), len(sources), flush=True)
    if '--postprocess' in sys.argv:
        result = read(OUT / 'all_69128_acquisition_quartet_shared_route_summary.csv.gz')
        grid = read(OUT / 'all_52598_molecular_grid_shared_route_summary.csv.gz')
        for column in result:
            if column.startswith('maximum_') or (column.startswith('representative_') and (not column.endswith('_recipe')) and (not column.startswith('representative_is_'))):
                result[column] = pd.to_numeric(result[column], errors='coerce')
        checks = Counter(json.loads((OUT / 'matrix_verification_counts.json').read_text(encoding='utf-8')))
    else:
        result, grid, checks = analyze(scope, graphs, blocks, values, source_counts)
    aggregate, coverage = summarize(result, grid)
    selected, paths, nodes, selected_sources, path_metrics = examples(result, graphs, blocks, values, sources)
    print('SELECTED', len(selected), len(paths), len(selected_sources), flush=True)
    subset_summary = subset_shared_routes(result, graphs, blocks, values, sources)
    sensitivity_summary = subset_threshold_sensitivity(result, graphs, values)
    subset_report(subset_summary, sensitivity_summary, len(result))
    checks['subset_threshold_connectivity_and_endpoint_readback_passed'] = subset_summary['status'] == 'passed'
    checks['subset_threshold_sensitivity_passed'] = sensitivity_summary['status'] == 'passed'
    checks.update(verify_endpoint_readback(result))
    checks['selected_paths_verified'] = 5 * len(selected)
    checks['selected_source_memberships_verified'] = len(selected_sources)
    checks['strict_source_records_aligned'] = len(sources)
    summary = report(result, grid, aggregate, coverage, selected, paths, nodes, selected_sources, path_metrics, checks, started, mapping, norm, target, blocks)
    print('COMPLETED', packed(summary), flush=True)
if __name__ == '__main__':
    main()
