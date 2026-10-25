"""Exact partner neighborhoods and complete two-reactant/two-component rectangles.

All R7 exact-focal partner comparisons remain in the denominator. Molecular
graph adjacency uses the existing exact-edit registry, without a similarity
cutoff. Four-target quantities are descriptive contrasts of matched yields."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import defaultdict, deque
import hashlib
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
from rdkit import Chem, rdBase
import round6_molecular_rewiring as chemistry
import round7_partner_conditioned_edits as previous
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OLD = 'round7/partner_conditioned_edits/'
OUT = ROOT / 'round8/partner_neighborhoods'
Y = chemistry.CORNERS
ROLES = chemistry.ROLES
EPS = 1e-12
SIGNS = np.array([1.0, -1.0, -1.0, 1.0])

def read(name, **kwargs):
    return pd.read_csv(ROOT / name, keep_default_na=False, low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 1} if name.endswith('.gz') else None)

def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def sid(value):
    return hashlib.sha256(packed(value).encode('utf-8')).hexdigest()[:20]

def directions(values, threshold):
    m0 = values[..., [0, 3]].min(axis=-1) - values[..., [1, 2]].max(axis=-1)
    m1 = values[..., [1, 2]].min(axis=-1) - values[..., [0, 3]].max(axis=-1)
    d0 = (m0 >= threshold - EPS) & (values[..., [0, 3]].min(axis=-1) >= 0.2 - EPS)
    reference_learning = (m1 >= threshold - EPS) & (values[..., [1, 2]].min(axis=-1) >= 0.2 - EPS)
    return (np.where(d0, 1, np.where(reference_learning, -1, 0)), m0, m1)

def exact_graphs(relations):
    graphs, distances, graph_rows = ({}, {}, [])
    for key, part in relations.groupby(['dataset_id', 'role'], sort=True):
        graph = defaultdict(set)
        for r in part.itertuples():
            graph[r.structural_molecule_a]
            graph[r.structural_molecule_b]
            if r.has_existing_exact_neighbor_rule:
                graph[r.structural_molecule_a].add(r.structural_molecule_b)
                graph[r.structural_molecule_b].add(r.structural_molecule_a)
        lookup = {}
        for source in graph:
            lookup[source] = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in lookup[source]:
                        lookup[source][neighbor] = lookup[source][current] + 1
                        queue.append(neighbor)
        graphs[key], distances[key] = (graph, lookup)
        seen, component_sizes = (set(), [])
        for source in graph:
            if source in seen:
                continue
            component = set(lookup[source])
            seen.update(component)
            component_sizes.append(len(component))
        graph_rows.append(dict(dataset_id=key[0], role=key[1], n_registered_molecules=len(graph), n_exact_edges=sum(map(len, graph.values())) // 2, n_components=len(component_sizes), n_isolated_molecules=sum((size == 1 for size in component_sizes)), component_sizes_descending=packed(sorted(component_sizes, reverse=True))))
    save(pd.DataFrame(graph_rows), 'dataset_role_exact_graph_coverage.csv')
    return (graphs, distances)

def annotate_pairs(pairs, relations, targets, distances):
    relation_lookup = {(r.dataset_id, r.role, r.molecular_pair_id): r._asdict() for r in relations.itertuples(index=False)}
    columns = ['exact_graph_method', 'exact_transformation', 'exact_common_core', 'exact_fragment_a', 'exact_fragment_b', 'structural_molecule_a', 'structural_molecule_b', 'edit_id', 'single_cut_common_core_ge60pct_ge5atoms', 'single_cut_core', 'single_cut_fragment_a', 'single_cut_fragment_b']
    rows = []
    for row in pairs.itertuples(index=False):
        partner_role = ROLES[1] if row.role == ROLES[0] else ROLES[0]
        other = relation_lookup[row.dataset_id, partner_role, row.partner_molecular_pair_id]
        info = {'partner_' + column: other[column] for column in columns}
        length = distances[row.dataset_id, partner_role].get(row.partner0_smiles, {}).get(row.partner1_smiles, -1)
        info['partner_registered_graph_distance'] = length
        info['partner_registered_graph_relation'] = 'one_exact_edit' if length == 1 else 'two_exact_edits' if length == 2 else 'three_or_more_exact_edits' if length >= 3 else 'no_registered_exact_path'
        info['molecular_scope'] = 'both_sides_exact_local_edits' if row.partner_has_exact_rule else 'only_focal_side_exact_local_edit'
        rows.append(info)
    frame = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    assert np.array_equal(frame.partner_has_exact_rule, frame.partner_registered_graph_distance.eq(1))
    values = targets.set_index('target_row_id')[Y]
    for partner in [0, 1]:
        for side in ['a', 'b']:
            ids = frame[f'target_row_id_{side}_partner{partner}']
            ys = values.loc[ids].to_numpy(float)
            for tag, threshold in [('5pp', 0.05), ('10pp', 0.1)]:
                frame[f'direction_{side}_partner{partner}_{tag}'] = directions(ys, threshold)[0]
    for tag in ['5pp', '10pp']:
        t0, t1 = (frame['transition_partner0_' + tag], frame['transition_partner1_' + tag])
        non0, non1 = (t0.eq('neither_strong_matching'), t1.eq('neither_strong_matching'))
        frame['transition_relation_' + tag] = np.select([non0 & non1, t0.eq(t1) & ~non0, non0 ^ non1], ['both_without_strong_pairing', 'same_informative_transition', 'informative_transition_vs_no_strong_pairing'], default='different_informative_transitions')
        frame['pairing_state_preserved_across_partner_' + tag] = frame[f'direction_a_partner0_{tag}'].eq(frame[f'direction_a_partner1_{tag}']) & frame[f'direction_b_partner0_{tag}'].eq(frame[f'direction_b_partner1_{tag}'])
        assert np.array_equal(frame['same_transition_' + tag], frame['pairing_state_preserved_across_partner_' + tag])
        frame['same_informative_transition_' + tag] = frame['transition_relation_' + tag].eq('same_informative_transition')
        frame['different_informative_transitions_' + tag] = frame['transition_relation_' + tag].eq('different_informative_transitions')
    save(frame, 'all_exact_focal_partner_pairs_with_graph_relations.csv.gz')
    return frame

def pair_summaries(frame):
    rows = []
    for grouping in ['molecular_scope', 'partner_registered_graph_relation', 'partner_exact_graph_method', 'rxn_class']:
        for key, part in frame.groupby(grouping, sort=True):
            row = dict(grouping=grouping, group=key, n_directed_partner_comparisons=len(part), n_edit_rectangle_domains=part.edit_rectangle_id.nunique(), n_focal_edits=part.edit_id.nunique(), n_reaction_classes=part.rxn_class.nunique(), n_opposite_delta_DD_each_ge10pp=int(part.opposite_delta_DD_each_ge10pp.sum()), n_opposite_delta_DD_each_ge20pp=int(part.opposite_delta_DD_each_ge20pp.sum()), median_absolute_fourth_difference=float(part.delta_DD_partner1_minus_partner0.abs().median()))
            for tag in ['5pp', '10pp']:
                for category, n in part['transition_relation_' + tag].value_counts().items():
                    row[category + '_' + tag] = int(n)
                row['opposite_matching_reversal_' + tag] = int(part['opposite_matching_reversal_' + tag].sum())
            rows.append(row)
    save(pd.DataFrame(rows).fillna(0), 'partner_relation_complete_denominators.csv')
    matrix = frame.groupby(['molecular_scope', 'partner_registered_graph_relation', 'transition_partner0_5pp', 'transition_partner1_5pp'], dropna=False).size().reset_index(name='n_directed_comparisons')
    save(matrix, 'all_transition_pairs_5pp.csv')

def local_domains(frame, exact_summary):
    rows = []
    for group_id, part in frame.groupby('edit_rectangle_id', sort=False):
        first = part.iloc[0]
        partners = set(part.partner0_smiles) | set(part.partner1_smiles)
        graph = {partner: set() for partner in partners}
        edges = part[part.partner_has_exact_rule]
        for row in edges.itertuples():
            graph[row.partner0_smiles].add(row.partner1_smiles)
            graph[row.partner1_smiles].add(row.partner0_smiles)
        seen, components, mixed_components = (set(), [], 0)
        responses = {}
        for row in part.itertuples():
            responses[row.partner0_smiles] = row.transition_partner0_5pp
            responses[row.partner1_smiles] = row.transition_partner1_5pp
        for partner in sorted(partners):
            if partner in seen:
                continue
            comp, queue = ({partner}, deque([partner]))
            while queue:
                for neighbor in graph[queue.popleft()]:
                    if neighbor not in comp:
                        comp.add(neighbor)
                        queue.append(neighbor)
            seen.update(comp)
            components.append(len(comp))
            mixed_components += len({responses[p] for p in comp}) > 1
        rows.append(dict(edit_rectangle_id=group_id, edit_id=first.edit_id, rectangle_id=first.rectangle_id, dataset_id=first.dataset_id, rxn_class=first.rxn_class, role=first.role, n_complete_common_partners=len(partners), n_all_partner_pairs=len(part), n_exact_local_partner_edges=len(edges), n_same_informative_exact_edges_5pp=int(edges.same_informative_transition_5pp.sum()), n_both_no_strong_exact_edges_5pp=int(edges.transition_relation_5pp.eq('both_without_strong_pairing').sum()), n_different_transition_exact_edges_5pp=int((~edges.same_transition_5pp).sum()), n_opposite_delta_DD_exact_edges_10pp=int(edges.opposite_delta_DD_each_ge10pp.sum()), n_induced_partner_graph_components=len(components), n_isolated_partners=sum((n == 1 for n in components)), component_sizes_descending=packed(sorted(components, reverse=True)), n_components_with_multiple_transitions_5pp=int(mixed_components)))
    result = pd.DataFrame(rows)
    ref = exact_summary.set_index('edit_rectangle_id')
    assert np.array_equal(result.n_complete_common_partners, ref.loc[result.edit_rectangle_id].n_common_partners)
    result = result.merge(exact_summary.drop(columns=['edit_id', 'rectangle_id', 'dataset_id', 'rxn_class', 'role']), on='edit_rectangle_id', validate='one_to_one')
    save(result, 'all_comparable_exact_edit_partner_neighborhoods.csv.gz')
    return result

def complete_local_rectangles(frame, targets, relations, rectangles):
    both = frame[frame.partner_has_exact_rule].copy()
    both['local_rectangle_id'] = [sid([row.rectangle_id, *sorted([row.edit_id, row.partner_edit_id])]) for row in both.itertuples()]
    sizes = both.groupby('local_rectangle_id').size()
    assert sizes.eq(2).all(), sizes.value_counts().to_dict()
    for _, part in both.groupby('local_rectangle_id', sort=False):
        assert set(part.role) == set(ROLES)
        a, b = (part.iloc[0], part.iloc[1])
        assert set(a[[c for c in both.columns if c.startswith('target_row_id_')]]) == set(b[[c for c in both.columns if c.startswith('target_row_id_')]])
        assert abs(a.delta_DD_partner1_minus_partner0 - b.delta_DD_partner1_minus_partner0) < EPS
    result = both[both.role.eq(ROLES[0])].copy().reset_index(drop=True)
    values = targets.set_index('target_row_id')
    relation_idx = relations.set_index('edit_id')
    cube = np.empty((len(result), 2, 2, 4), dtype=float)
    long = []
    for i, (a, b) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        side = 'a' if a == 0 else 'b'
        ids = result[f'target_row_id_{side}_partner{b}'].to_numpy()
        ys = values.loc[ids, Y].to_numpy(float)
        cube[:, a, b, :] = ys
        for j, corner in enumerate(Y):
            result[f'yield_reactant{a}{b}_{corner}'] = ys[:, j]
            cells = pd.DataFrame(dict(local_rectangle_id=result.local_rectangle_id, dataset_id=result.dataset_id, rxn_class=result.rxn_class, rectangle_id=result.rectangle_id, reactant1_level=a, reactant2_level=b, corner=corner, target_row_id=ids, target_key=values.loc[ids].target_key.to_numpy(), reactant1_smiles=values.loc[ids, ROLES[0]].to_numpy(), reactant2_smiles=values.loc[ids, ROLES[1]].to_numpy(), product_smiles=values.loc[ids].product_smiles.to_numpy(), yield_numeric=ys[:, j]))
            long.append(cells)
    result['edit1_id'], result['edit2_id'] = (result.edit_id, result.partner_edit_id)
    for axis in [1, 2]:
        for field in ['structural_molecule_a', 'structural_molecule_b', 'structural_change_a_to_b', 'exact_graph_method', 'exact_transformation', 'exact_common_core', 'exact_fragment_a', 'exact_fragment_b']:
            result[f'edit{axis}_' + field] = relation_idx.loc[result[f'edit{axis}_id'], field].to_numpy()
    dd = cube @ SIGNS
    for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        result[f'DD_reactant{a}{b}'] = dd[:, a, b]
    fourth = dd[:, 1, 1] - dd[:, 0, 1] - dd[:, 1, 0] + dd[:, 0, 0]
    assert np.allclose(fourth, result.delta_DD_partner1_minus_partner0, atol=EPS)
    result['fourth_difference'] = fourth
    result['absolute_fourth_difference'] = abs(fourth)
    result['edit2_delta_DD_at_reactant1_0'] = dd[:, 0, 1] - dd[:, 0, 0]
    result['edit2_delta_DD_at_reactant1_1'] = dd[:, 1, 1] - dd[:, 1, 0]
    result['opposite_edit2_delta_DD_each_ge10pp'] = (np.minimum(result.edit2_delta_DD_at_reactant1_0, result.edit2_delta_DD_at_reactant1_1) <= -0.1 + EPS) & (np.maximum(result.edit2_delta_DD_at_reactant1_0, result.edit2_delta_DD_at_reactant1_1) >= 0.1 - EPS)
    result['both_exact_edits_opposite_delta_DD_each_ge10pp'] = result.opposite_delta_DD_each_ge10pp & result.opposite_edit2_delta_DD_each_ge10pp
    second_axis = both[both.role.eq(ROLES[1])].set_index('local_rectangle_id').loc[result.local_rectangle_id]
    for tag, threshold in [('5pp', 0.05), ('10pp', 0.1)]:
        for field in ['transition_partner0_', 'transition_partner1_', 'same_informative_transition_', 'transition_relation_']:
            result['edit2_' + field + tag] = second_axis[field + tag].to_numpy()
        result['any_edit_same_informative_transition_' + tag] = result['same_informative_transition_' + tag] | result['edit2_same_informative_transition_' + tag]
        dirs = directions(cube, threshold)[0]
        n_strong = (dirs != 0).sum(axis=(1, 2))
        result['n_strong_pairing_targets_' + tag] = n_strong
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            result[f'direction_reactant{a}{b}_{tag}'] = dirs[:, a, b]
        result['pairing_pattern_00_10_01_11_' + tag] = ['/'.join(map(str, v)) for v in dirs[:, [0, 1, 0, 1], [0, 0, 1, 1]]]
        checker = (dirs[:, 0, 0] == dirs[:, 1, 1]) & (dirs[:, 1, 0] == dirs[:, 0, 1]) & (dirs[:, 0, 0] == -dirs[:, 1, 0]) & (n_strong == 4)
        result['four_strong_checkerboard_' + tag] = checker
        result['single_strong_target_' + tag] = n_strong == 1
        result['all_four_same_strong_pairing_' + tag] = (n_strong == 4) & (dirs.min(axis=(1, 2)) == dirs.max(axis=(1, 2)))
        result['both_pairing_directions_present_' + tag] = (dirs.min(axis=(1, 2)) == -1) & (dirs.max(axis=(1, 2)) == 1)
        result['no_strong_pairing_target_' + tag] = n_strong == 0
        result['nonzero_pairing_00_11_only_' + tag] = (dirs[:, 0, 0] != 0) & (dirs[:, 1, 1] != 0) & (dirs[:, 1, 0] == 0) & (dirs[:, 0, 1] == 0)
        result['nonzero_pairing_10_01_only_' + tag] = (dirs[:, 1, 0] != 0) & (dirs[:, 0, 1] != 0) & (dirs[:, 0, 0] == 0) & (dirs[:, 1, 1] == 0)
    result = result.merge(rectangles[previous.RECIPE_FIELDS], on='rectangle_id', validate='many_to_one')
    long = pd.concat(long, ignore_index=True)
    assert len(long) == 16 * len(result)
    assert not long.duplicated(['local_rectangle_id', 'reactant1_level', 'reactant2_level', 'corner']).any()
    save(result, 'all_unique_both_exact_16_cell_rectangles.csv.gz')
    save(long, 'all_unique_both_exact_16_cells.csv.gz')
    save(both[['local_rectangle_id', 'partner_pair_comparison_id', 'role', 'edit_id', 'partner_edit_id']], 'symmetric_view_deduplication.csv')
    return (result, long)

def rectangle_summaries(local, mapping):
    rows = []
    for dataset in mapping.dataset_id:
        part = local[local.dataset_id.eq(dataset)]
        row = dict(dataset_id=dataset, rxn_class=mapping.set_index('dataset_id').loc[dataset, 'rxn_class'], n_unique_both_exact_rectangles=len(part), n_component_rectangles=part.rectangle_id.nunique(), n_edit1=part.edit1_id.nunique(), n_edit2=part.edit2_id.nunique(), n_distinct_double_edits=len(part[['edit1_id', 'edit2_id']].drop_duplicates()), n_opposite_edit1_delta_DD_each_ge10pp=int(part.opposite_delta_DD_each_ge10pp.sum()), n_opposite_edit2_delta_DD_each_ge10pp=int(part.opposite_edit2_delta_DD_each_ge10pp.sum()), n_both_exact_edits_opposite_delta_DD_each_ge10pp=int(part.both_exact_edits_opposite_delta_DD_each_ge10pp.sum()))
        for tag in ['5pp', '10pp']:
            for metric in ['four_strong_checkerboard', 'single_strong_target', 'all_four_same_strong_pairing', 'both_pairing_directions_present', 'no_strong_pairing_target', 'nonzero_pairing_00_11_only', 'nonzero_pairing_10_01_only']:
                row['n_' + metric + '_' + tag] = int(part[metric + '_' + tag].sum())
        rows.append(row)
    save(pd.DataFrame(rows), 'both_exact_16_cell_coverage_all_20_datasets.csv')
    save(local.groupby(['rxn_class', 'edit1_exact_graph_method', 'edit2_exact_graph_method', 'pairing_pattern_00_10_01_11_5pp']).size().reset_index(name='n_unique_rectangles'), 'all_exact_edit_pair_pairing_patterns_5pp.csv')

def select_examples(local, cells, targets, pairs, relations):
    selected = []
    categories = ['four_strong_checkerboard_5pp', 'both_pairing_directions_present_10pp', 'both_pairing_directions_present_5pp', 'single_strong_target_10pp', 'any_edit_same_informative_transition_5pp', 'all_four_same_strong_pairing_5pp', 'both_exact_edits_opposite_delta_DD_each_ge10pp']
    for category in categories:
        eligible = local[local[category]].sort_values('absolute_fourth_difference', ascending=False)
        for _, group in eligible.groupby('rxn_class', sort=True):
            best = group.iloc[0]
            selected.append(dict(local_rectangle_id=best.local_rectangle_id, selection_reason=category))
    choices = pd.DataFrame(selected).drop_duplicates('local_rectangle_id')
    chosen = choices.merge(local, on='local_rectangle_id', validate='one_to_one')
    save(chosen, 'selected_cross_class_full_16_cell_examples.csv')
    chosen_cells = cells.merge(choices, on='local_rectangle_id', validate='many_to_one')
    save(chosen_cells, 'selected_cross_class_full_16_cells.csv')
    domains = set(chosen.edit_rectangle_id)
    selected_pairs = pairs[pairs.edit_rectangle_id.isin(domains)].copy()
    save(selected_pairs, 'selected_examples_all_comparable_partner_pairs.csv.gz')
    needed_ids = set(chosen_cells.target_row_id)
    for column in [c for c in selected_pairs.columns if c.startswith('target_row_id_')]:
        needed_ids.update(selected_pairs[column])
    selected_targets = targets[targets.target_row_id.isin(needed_ids)].copy()
    save(selected_targets, 'selected_examples_all_complete_partner_targets.csv.gz')
    return (chosen, chosen_cells, selected_targets)

def reconstruct_sources(selected_targets, chosen_cells, rectangles):
    normal = read('data/normalized_records.csv.gz')
    normal = normal[normal.strict_eligible.astype(str).str.lower().eq('true')].copy()
    normal['yield_numeric'] = pd.to_numeric(normal.yield_numeric, errors='raise')
    raw = read('data/raw_reactions.csv.gz')
    requests, blocks = ([], [])
    meta = rectangles.set_index('rectangle_id')
    for rectangle, group in selected_targets.groupby('rectangle_id', sort=False):
        recipes = meta.loc[rectangle, chemistry.RECIPES].tolist()
        ds = group.dataset_id.iloc[0]
        source = normal[normal.dataset_id.eq(ds) & normal.target_key.isin(group.target_key) & normal.recipe_id.isin(recipes)]
        cells = source.groupby([*chemistry.BLOCK, 'recipe_id'], sort=False, as_index=False).yield_numeric.mean()
        full = cells.pivot(index=chemistry.BLOCK, columns='recipe_id', values='yield_numeric').reindex(columns=recipes).dropna().reset_index()
        full = full.rename(columns=dict(zip(recipes, Y)))
        expected = group.set_index('target_key')
        means = full.groupby('target_key')[Y].mean().reindex(expected.index)
        assert np.allclose(means, expected[Y], atol=EPS), rectangle
        full['target_row_id'] = full.target_key.map(expected.target_row_id)
        full['rectangle_id'] = rectangle
        blocks.append(full)
        for corner, recipe in zip(Y, recipes):
            req = full[['rectangle_id', 'target_row_id', *chemistry.BLOCK, corner]].rename(columns={corner: 'matched_cell_yield'})
            req['recipe_id'], req['corner'] = (recipe, corner)
            requests.append(req)
    blocks = pd.concat(blocks, ignore_index=True)
    requests = pd.concat(requests, ignore_index=True)
    links = requests.merge(normal, on=[*chemistry.BLOCK, 'recipe_id'], how='left', validate='many_to_many')
    assert links.record_id.notna().all()
    assert not links.duplicated(['rectangle_id', 'target_row_id', 'archive', 'plate', 'acquisition', 'corner', 'record_id']).any()
    extra = ['record_id', 'source_file', 'source_sheet', 'source_row', 'reagent1_id', 'reagent2_id', 'reagent3_id', 'reagent4_id', 'reagent5_id', 'reagent6_id', 'solvent_id', 'temp_id', 'time_id']
    links = links.merge(raw[extra], on='record_id', validate='many_to_one')
    raw_root = RAW_DIR
    links['source_absolute_path'] = links.source_file.map(lambda name: str(raw_root / name))
    assert all((reader_tables.source_is_file(path) for path in links.source_absolute_path.unique()))
    links['source_record_locator'] = links.source_file.astype(str) + '#' + links.source_sheet.astype(str) + ':' + links.source_row.astype(str)
    save(blocks, 'selected_examples_complete_acquisition_blocks.csv.gz')
    save(links, 'selected_examples_all_partner_original_records.csv.gz')
    membership = chosen_cells[['local_rectangle_id', 'reactant1_level', 'reactant2_level', 'rectangle_id', 'target_row_id', 'corner']]
    full_records = membership.merge(links, on=['rectangle_id', 'target_row_id', 'corner'], validate='one_to_many')
    save(full_records, 'selected_full_16_cell_original_records.csv.gz')
    max_error, verified, checked_fields = (0.0, 0, 0)
    original_fields = [*ROLES, 'product_smiles', 'condition_id', 'reagent1_id', 'reagent2_id', 'reagent3_id', 'reagent4_id', 'reagent5_id', 'reagent6_id', 'solvent_id', 'temp_id', 'time_id']
    wanted = links[['source_file', 'source_sheet', 'source_row', 'yield_numeric', *original_fields]].drop_duplicates()
    for (filename, sheet), part in wanted.groupby(['source_file', 'source_sheet'], sort=False):
        path = raw_root / filename
        if path.suffix.lower() == '.csv':
            original = reader_tables.read_csv(path, keep_default_na=False, low_memory=False)
        else:
            original = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        yield_column = 'yield' if 'yield' in original.columns else None
        assert yield_column is not None, (filename, original.columns.tolist())
        for row in part.itertuples():
            source_row = original.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(row.source_row)]
            value = float(source_row[yield_column])
            error = abs(value - row.yield_numeric)
            max_error = max(max_error, error)
            assert error < EPS, (filename, row.source_row, value, row.yield_numeric)
            for field in original_fields:
                source_value, expected_value = (source_row[field], getattr(row, field))
                if field in ['temp_id', 'time_id'] and str(source_value) != '' and (str(expected_value) != ''):
                    assert float(source_value) == float(expected_value), (filename, row.source_row, field)
                else:
                    assert str(source_value) == str(expected_value), (filename, row.source_row, field, source_value, expected_value)
                checked_fields += 1
            verified += 1
    return dict(n_selected_target_rectangles=len(selected_targets), n_selected_acquisition_blocks=len(blocks), n_source_links=len(links), n_unique_original_records=links.record_id.nunique(), n_full_16_cell_source_memberships=len(full_records), n_original_file_rows_reread=verified, n_original_files=wanted.source_file.nunique(), max_original_yield_error=max_error, n_original_structure_recipe_fields_checked=checked_fields)

def independent_context_crosscheck(local):
    path = ROOT / 'round8/context_decomposition/all_194533_observed_cube_decompositions.csv.gz'
    if not path.is_file():
        return {'status': 'independent_agent_table_not_present'}
    table = pd.read_csv(path, low_memory=False, keep_default_na=False, usecols=['partner_pair_comparison_id', 'fourth_contrast_pp', 'full_checkerboard_gap5pp_high20'])
    joined = local.merge(table, on='partner_pair_comparison_id', validate='one_to_one')
    error = float(np.max(abs(joined.fourth_difference * 100 - joined.fourth_contrast_pp)))
    assert len(joined) == len(local) and error < 1e-10
    assert np.array_equal(joined.four_strong_checkerboard_5pp, joined.full_checkerboard_gap5pp_high20)
    return dict(status='passed', n_unique_local_rectangles=len(joined), max_fourth_contrast_error_pp=error, checkerboard_flags_identical=True)

def report(pairs, local, neighborhoods, chosen, verification, started):
    exact = pairs[pairs.partner_has_exact_rule]
    summary = dict(status='complete', study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, n_all_exact_focal_directed_partner_comparisons=len(pairs), n_both_exact_directed_comparisons=len(exact), n_only_focal_exact_directed_comparisons=int((~pairs.partner_has_exact_rule).sum()), n_unique_full_both_exact_16_cell_rectangles=len(local), n_unique_full_both_exact_16_cells=16 * len(local), n_unique_double_edits=len(local[['dataset_id', 'edit1_id', 'edit2_id']].drop_duplicates()), n_both_exact_datasets=local.dataset_id.nunique(), n_both_exact_classes=local.rxn_class.nunique(), n_comparable_exact_edit_domains=len(neighborhoods), n_domains_with_exact_partner_edge=int(neighborhoods.n_exact_local_partner_edges.gt(0).sum()), n_domains_with_changed_transition_exact_edge_5pp=int(neighborhoods.n_different_transition_exact_edges_5pp.gt(0).sum()), n_domains_with_mixed_connected_component_5pp=int(neighborhoods.n_components_with_multiple_transitions_5pp.gt(0).sum()), n_selected_full_16_cell_examples=len(chosen), verification=verification, seconds=time.monotonic() - started)
    summary['strong_target_count_distribution_5pp'] = {str(k): int(v) for k, v in local.n_strong_pairing_targets_5pp.value_counts().sort_index().items()}
    summary['n_both_exact_edits_opposite_delta_DD_each_ge10pp'] = int(local.both_exact_edits_opposite_delta_DD_each_ge10pp.sum())
    summary['n_unique_rectangles_with_any_informative_preservation_5pp'] = int(local.any_edit_same_informative_transition_5pp.sum())
    for tag in ['5pp', '10pp']:
        summary['both_exact_directed_transition_relations_' + tag] = {str(k): int(v) for k, v in exact['transition_relation_' + tag].value_counts().items()}
        for metric in ['four_strong_checkerboard', 'single_strong_target', 'all_four_same_strong_pairing', 'both_pairing_directions_present', 'no_strong_pairing_target']:
            summary[metric + '_' + tag] = int(local[metric + '_' + tag].sum())
    strong_counts = summary['strong_target_count_distribution_5pp']
    relations5 = summary['both_exact_directed_transition_relations_5pp']
    text = ['# Exact partner neighborhoods and full local two-reactant rectangles', '', 'The same local molecular edit can preserve, establish, eliminate or change component pairing when the other reactant moves by a second exact local edit. The complete local square identifies which four actual reactant combinations carry those pairings.', '', f'The study scope remains 20 datasets, 18 reaction classes and 4,002 background-specific single-slot exchanges. This analysis retains all {len(pairs):,} R7 exact-focal-edit partner comparisons, including preserved responses and non-events. {len(exact):,} directed comparisons have exact edits on both reactants; removing the symmetric role view gives {len(local):,} unique complete 16-cell rectangles, spanning {local.rxn_class.nunique()} classes and {local.dataset_id.nunique()} datasets.', '', f"At 5pp separation and preferred-cell yields at least 20%, {strong_counts.get('0', 0):,} squares have no strong pairing, {strong_counts.get('1', 0):,} have exactly one strong target, {strong_counts.get('2', 0):,} have two, and {strong_counts.get('3', 0):,} have three. {strong_counts.get('4', 0):,} squares have all four targets strongly paired. {summary['both_pairing_directions_present_5pp']:,} contain both pairing directions. Thus the observed local changes most often establish or remove a pairing in one particular reactant combination rather than create a full four-target checkerboard.", '', f"Among the {summary['n_both_exact_directed_comparisons']:,} directed exact-neighbor comparisons, {relations5.get('both_without_strong_pairing', 0):,} have no strong pairing before or after the focal edit under either partner, {relations5.get('informative_transition_vs_no_strong_pairing', 0):,} contrast an informative transition with no strong pairing, {relations5.get('different_informative_transitions', 0):,} have two different informative transitions, and {relations5.get('same_informative_transition', 0):,} preserve the same informative transition. The {relations5.get('same_informative_transition', 0):,} preserved informative responses occur in acylation and Suzuki; they are displayed with the altered responses. The same local partner graph contains mixed transition states in {summary['n_domains_with_mixed_connected_component_5pp']:,} complete edit domains.", '', f"Continuous component responses add a separate chemical relation: in {summary['n_both_exact_edits_opposite_delta_DD_each_ge10pp']:,} complete squares, both exact molecular edits change component DD by at least +10pp in one partner and at most -10pp in the other. These continuous sign changes and the strong-pairing states remain separate quantities.", '', '## Chemistry from complete local squares', '']
    cols = ['reactant00', 'reactant10', 'reactant01', 'reactant11']
    for case in chosen.sort_values(['rxn_class', 'selection_reason']).itertuples():
        text.extend([f'### {case.rxn_class}: {case.selection_reason}', '', f'`{case.local_rectangle_id}`. Reactant 1: `{case.edit1_structural_molecule_a}` to `{case.edit1_structural_molecule_b}` ({case.edit1_structural_change_a_to_b}; {case.edit1_exact_graph_method}). Reactant 2: `{case.edit2_structural_molecule_a}` to `{case.edit2_structural_molecule_b}` ({case.edit2_structural_change_a_to_b}; {case.edit2_exact_graph_method}).', '', f'Components: {case.component0_name} ({case.component0_dose}) / {case.component1_name} ({case.component1_dose}) x {case.context0_name} / {case.context1_name}. Fixed recipe slots: `{case.fixed_background_json}`.', '', f'Four complete recipes: {case.full_recipe_c0_bg0} | {case.full_recipe_c1_bg0} | {case.full_recipe_c0_bg1} | {case.full_recipe_c1_bg1}.', '', '| Reactant levels | y00 % | y10 % | y01 % | y11 % | Pairing at 5pp |', '|---|---:|---:|---:|---:|---:|'])
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            yields = [getattr(case, f'yield_reactant{a}{b}_{corner}') * 100 for corner in Y]
            text.append(f'| {a}{b} | ' + ' | '.join((f'{v:.2f}' for v in yields)) + f" | {getattr(case, f'direction_reactant{a}{b}_5pp')} |")
        text.extend(['', f"Edit 1's transitions under partners 0/1: {case.transition_partner0_5pp} / {case.transition_partner1_5pp}. Edit 2's transitions under reactant 1 levels 0/1: {case.edit2_transition_partner0_5pp} / {case.edit2_transition_partner1_5pp}.", '', f"Edit 1's change in component DD at partner 0/1: {case.delta_DD_partner0 * 100:.2f}/{case.delta_DD_partner1 * 100:.2f} pp. Edit 2's changes at reactant 1 levels 0/1: {case.edit2_delta_DD_at_reactant1_0 * 100:.2f}/{case.edit2_delta_DD_at_reactant1_1 * 100:.2f} pp. Fourth difference: {case.fourth_difference * 100:.2f} pp.", ''])
    text.extend(['## Complete denominators and definitions', '', 'Partner adjacency means one of the existing exact graph rules: H substitution, aromatic CH/N replacement, methylene insertion/deletion, substituent relocation on an identical parent, or a single-cut matched molecular pair with a variable fragment at most four heavy atoms, a core at least five heavy atoms and at least 60% of each molecule. The registry chooses one priority rule per pair; it does not enumerate all possible atom mappings. Fingerprint similarity is retained as an annotation and does not define adjacency.', '', 'The registered partner graph also identifies two-step or longer exact-edit paths. Such a path describes structural relations; an intermediate molecule need not have the same measured component rectangle. The induced graph in each complete response domain uses only its observed common partners.', '', 'The component order is y00,y10,y01,y11. Strong pairing +1 means min(y00,y11)-max(y10,y01) is at least 5pp (or 10pp) and both preferred cells are at least 20%; -1 reverses the diagonal, and 0 denotes neither strong pairing. A four-strong checkerboard requires both molecular diagonals to carry opposite strong component pairings. Single-strong-target cases retain three other fully measured targets without strong pairing. Same informative transition and both-without-strong-pairing are separately counted.', '', "DD=y00-y10-y01+y11. The fourth difference is DD11-DD01-DD10+DD00, equivalently the difference between the same focal edit's changes in DD under the two partner structures. It is a four-axis contrast, not an actual yield gain or a mechanism.", '', "All 16-cell rectangles use four targets' complete within-acquisition component squares, averaged within each target after matching. Shared molecules, recipes, targets, and source records are retained as shared observations. Zero yields remain zero; missing cells are never filled.", '', "`partner_relation_complete_denominators.csv` retains every structural relation and non-event. `both_exact_16_cell_coverage_all_20_datasets.csv` includes zero-coverage datasets. `all_unique_both_exact_16_cells.csv.gz` contains every complete local cell. Selected example tables also include the full focal edit's comparable partner set and original source records.", '', '## Run summary', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', ''])
    (OUT / 'PARTNER_NEIGHBORHOODS_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = read(OLD + 'exact_edit_all_partner_pair_comparisons.csv.gz')
    relations = read(OLD + 'molecular_relation_registry.csv.gz')
    targets = read(OLD + 'all_primary_target_four_yields.csv.gz')
    rectangles = read(OLD + 'all_primary_recipe_rectangles.csv')
    domains = read(OLD + 'exact_edit_rectangle_partner_summary.csv.gz')
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    assert len(pairs) == 204057 and len(targets) == 176321 and (len(rectangles) == 825)
    assert len(mapping) == 20 and mapping.rxn_class.nunique() == 18
    if '--verify-selected' in sys.argv:
        pairs = read('round8/partner_neighborhoods/all_exact_focal_partner_pairs_with_graph_relations.csv.gz')
        neighborhoods = read('round8/partner_neighborhoods/all_comparable_exact_edit_partner_neighborhoods.csv.gz')
        local = read('round8/partner_neighborhoods/all_unique_both_exact_16_cell_rectangles.csv.gz')
        chosen = read('round8/partner_neighborhoods/selected_cross_class_full_16_cell_examples.csv')
        chosen_cells = read('round8/partner_neighborhoods/selected_cross_class_full_16_cells.csv')
        selected_targets = read('round8/partner_neighborhoods/selected_examples_all_complete_partner_targets.csv.gz')
        verification = reconstruct_sources(selected_targets, chosen_cells, rectangles)
        verification['independent_agent_contrast_readback'] = independent_context_crosscheck(local)
        report(pairs, local, neighborhoods, chosen, verification, started)
        return
    if '--resume' in sys.argv:
        pairs = read('round8/partner_neighborhoods/all_exact_focal_partner_pairs_with_graph_relations.csv.gz')
        neighborhoods = read('round8/partner_neighborhoods/all_comparable_exact_edit_partner_neighborhoods.csv.gz')
    else:
        _, distances = exact_graphs(relations)
        print('registered exact graphs complete', flush=True)
        pairs = annotate_pairs(pairs, relations, targets, distances)
        pair_summaries(pairs)
        neighborhoods = local_domains(pairs, domains)
        print('all partner neighborhoods complete', len(neighborhoods), flush=True)
    local, cells = complete_local_rectangles(pairs, targets, relations, rectangles)
    rectangle_summaries(local, mapping)
    print('unique complete local rectangles', len(local), local.groupby('rxn_class').size().to_dict(), flush=True)
    chosen, chosen_cells, selected_targets = select_examples(local, cells, targets, pairs, relations)
    verification = reconstruct_sources(selected_targets, chosen_cells, rectangles)
    verification['independent_agent_contrast_readback'] = independent_context_crosscheck(local)
    report(pairs, local, neighborhoods, chosen, verification, started)
if __name__ == '__main__':
    main()
