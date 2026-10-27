"""Compare molecular substitutions on identical, jointly measured recipe graphs."""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import heapq
from itertools import combinations, product
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
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round6/molecular_connectivity'
BLOCK = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
SLOTS = [f'Reagent{i}' for i in range(1, 7)] + ['Solvent', 'Temperature', 'Time']
TOKENS = [slot + '_token' for slot in SLOTS]
FOUR = ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']
READ = dict(keep_default_na=False, low_memory=False)

def read(path, **kwargs):
    return pd.read_csv(ROOT / path, **READ, **kwargs)

def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def sid(value):
    return sha256(packed(value).encode()).hexdigest()[:18]

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression='gzip' if name.endswith('.gz') else None)

def widest_matrix(y, adjacent):
    """Floyd max-min closure gives all-pair best measured node bottlenecks."""
    capacity = np.full(adjacent.shape, -np.inf)
    capacity[adjacent] = np.minimum(y[:, None], y[None, :])[adjacent]
    np.fill_diagonal(capacity, y)
    for k in range(len(y)):
        capacity = np.maximum(capacity, np.minimum(capacity[:, k, None], capacity[None, k, :]))
    return capacity

def widest_path(y, adjacent, start, goal):
    best, previous = (np.full(len(y), -np.inf), np.full(len(y), -1, dtype=int))
    best[start] = y[start]
    queue = [(-y[start], start)]
    while queue:
        negative, node = heapq.heappop(queue)
        if -negative < best[node] - 1e-15:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(int(previous[path[-1]]))
            return path[::-1]
        for neighbor in np.flatnonzero(adjacent[node]):
            value = min(best[node], y[neighbor])
            if value > best[neighbor] + 1e-15:
                best[neighbor], previous[neighbor] = (value, node)
                heapq.heappush(queue, (-value, neighbor))
    return []

def self_checks():
    adjacent = np.array([[False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]])
    a, b = (np.array([0.8, 0.6, 0.7, 0.1]), np.array([0.9, 0.0, 0.8, 0.05]))
    ca, cb = (widest_matrix(a, adjacent), widest_matrix(b, adjacent))
    assert ca[0, 2] == 0.6 and cb[0, 2] == 0.05
    for values, cap in [(a, ca), (b, cb)]:
        path = widest_path(values, adjacent, 0, 2)
        assert min(values[path]) == cap[0, 2]

def prepare():
    candidates = read('round6/molecular_rewiring/all_opposite_direction_candidates.csv.gz')
    candidates = candidates[candidates.reversal_5pp.astype(str).str.lower().eq('true')].copy()
    assert len(candidates) == 768 and candidates.candidate_id.is_unique
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    registry = read('data/condition_slot_registry.csv')
    registry = registry[registry.dataset_id.isin(mapping.dataset_id)].drop_duplicates(['dataset_id', 'recipe_id'])
    assert not registry.duplicated(['dataset_id', *TOKENS]).any()
    graphs = {}
    for dataset, part in registry.groupby('dataset_id', sort=False):
        part = part.sort_values('recipe_id')
        values = part[TOKENS].to_numpy(str)
        adjacency = (values[:, None, :] != values[None, :, :]).sum(-1) == 1
        labels, full = ({}, {})
        for row in part.to_dict('records'):
            text = {}
            for slot in SLOTS:
                if slot.startswith('Reagent'):
                    name = row[f'{slot}_Standardized Name'] or 'none'
                    text[slot] = name + (f" {row[f'{slot}_HTE_Equiv']} eq" if name != 'none' else '')
                else:
                    text[slot] = {'Solvent': row['Standardized Name'], 'Temperature': str(row['temp_id']) + ' C', 'Time': str(row['time_id']) + ' h'}[slot]
            labels[row['recipe_id']] = text
            full[row['recipe_id']] = '; '.join((f'{slot}={text[slot]}' for slot in SLOTS))
        graphs[dataset] = dict(recipes=part.recipe_id.tolist(), lookup={r: i for i, r in enumerate(part.recipe_id)}, adjacency=adjacency, tokens=values, labels=labels, full=full)
    columns = ['record_id', *BLOCK, 'recipe_id', 'condition_id', 'yield_numeric', 'strict_eligible']
    norm = read('data/normalized_records.csv.gz', usecols=columns)
    norm = norm[norm.strict_eligible.astype(str).str.lower().eq('true') & norm.dataset_id.isin(mapping.dataset_id)].copy()
    norm['yield_numeric'] = pd.to_numeric(norm.yield_numeric, errors='raise')
    assert np.isfinite(norm.yield_numeric).all()
    nodes = norm.groupby([*BLOCK, 'recipe_id'], sort=False, as_index=False).yield_numeric.mean()
    measured = {key: part.set_index('recipe_id').yield_numeric for key, part in nodes.groupby(BLOCK, sort=False)}
    blocks = read('round6/molecular_rewiring/complete_target_acquisition_blocks.csv.gz')
    block_lookup = defaultdict(list)
    for row in blocks.to_dict('records'):
        block_lookup[row['dataset_id'], row['rectangle_id'], row['target_key']].append(tuple((row[k] for k in BLOCK)))
    return (candidates, mapping, graphs, norm, measured, block_lookup)

def analyze(candidates, graphs, measured, block_lookup):
    results, graph_pairs = ([], {})
    for count, candidate in enumerate(candidates.to_dict('records'), 1):
        dataset, rectangle = (candidate['dataset_id'], candidate['rectangle_id'])
        graph = graphs[dataset]
        support_a = block_lookup[dataset, rectangle, candidate['target_key_a']]
        support_b = block_lookup[dataset, rectangle, candidate['target_key_b']]
        assert support_a and support_b
        four = [candidate[key] for key in FOUR]
        for key_a, key_b in product(support_a, support_b):
            measured_a, measured_b = (measured[key_a], measured[key_b])
            recipes = sorted(set(measured_a.index) & set(measured_b.index))
            assert set(four) <= set(recipes)
            indices = [graph['lookup'][recipe] for recipe in recipes]
            adjacent = graph['adjacency'][np.ix_(indices, indices)]
            values_a, values_b = (measured_a.loc[recipes].to_numpy(float), measured_b.loc[recipes].to_numpy(float))
            loc = {recipe: j for j, recipe in enumerate(recipes)}
            q = [loc[recipe] for recipe in four]
            f_a, f_b = (values_a[q], values_b[q])
            margin_a = float(min(f_a[[0, 3]]) - max(f_a[[1, 2]]))
            margin_b = float(min(f_b[[1, 2]]) - max(f_b[[0, 3]]))
            complete_reversal5 = margin_a >= 0.05 and margin_b >= 0.05 and (min(f_a[[0, 3]]) >= 0.2) and (min(f_b[[1, 2]]) >= 0.2)
            full_components, full_labels = connected_components(csr_matrix(adjacent), directed=False)
            highs = [values >= 0.2 for values in [values_a, values_b]]
            high_counts = [connected_components(csr_matrix(adjacent[np.ix_(np.flatnonzero(h), np.flatnonzero(h))]), directed=False, return_labels=False) for h in highs]
            capacities_a, capacities_b = (widest_matrix(values_a, adjacent), widest_matrix(values_b, adjacent))
            endpoints = np.flatnonzero(highs[0] & highs[1])
            pairs = [(a, b) for a, b in combinations(endpoints, 2) if full_labels[a] == full_labels[b]]
            row = {key: candidate[key] for key in ['candidate_id', 'rectangle_id', 'dataset_id', 'rxn_class', 'role', 'molecular_pair_id', 'molecule_a', 'molecule_b', 'partner_smiles', 'target_key_a', 'target_key_b', 'product_smiles_a', 'product_smiles_b', 'has_existing_exact_neighbor_rule', 'has_round4_allowed_exact_rule', 'exact_graph_method', 'structural_relation', 'structural_change_a_to_b', 'minimum_pair_margin', 'reversal_10pp', 'molecule_name_a', 'molecule_name_b', 'partner_name']}
            row.update({f'{key}_{side}': block_key[i] for side, block_key in [('a', key_a), ('b', key_b)] for i, key in enumerate(BLOCK[1:4], 1)})
            graph_pair_id = sid(sorted([list(key_a), list(key_b)]))
            row.update(graph_pair_id=graph_pair_id, target_pair_id=sid([dataset, *sorted([candidate['target_key_a'], candidate['target_key_b']])]), n_measured_recipes_a=len(measured_a), n_measured_recipes_b=len(measured_b), n_common_measured_recipes=len(recipes), common_recipe_ids=packed(recipes), n_common_graph_edges=int(adjacent.sum() // 2), n_common_graph_components=full_components, four_recipe_ids=packed(four), four_yields_a=packed(f_a.tolist()), four_yields_b=packed(f_b.tolist()), acquisition_pair_reversal_5pp=complete_reversal5, acquisition_pair_margin_a=margin_a, acquisition_pair_margin_b=margin_b, n_high_recipes_a=int(highs[0].sum()), n_high_recipes_b=int(highs[1].sum()), n_high_groups_a=int(high_counts[0]), n_high_groups_b=int(high_counts[1]), n_common_high_recipes=len(endpoints), n_common_high_endpoint_pairs=len(endpoints) * (len(endpoints) - 1) // 2, n_common_high_endpoint_pairs_with_full_graph_path=len(pairs), n_pairs_connectivity_switch=0, n_pairs_connectivity_switch_gap5pp=0, n_pairs_connectivity_switch_gap10pp=0, any_connectivity_switch=False, any_connectivity_switch_gap5pp=False, any_connectivity_switch_gap10pp=False, maximum_absolute_bottleneck_difference=np.nan, best_switch_bottleneck_difference=np.nan, maximum_difference_endpoint0_recipe='', maximum_difference_endpoint1_recipe='', maximum_difference_bottleneck_a=np.nan, maximum_difference_bottleneck_b=np.nan, maximum_difference_signed_a_minus_b=np.nan, n_pairs_absolute_bottleneck_difference_ge5pp=0, n_pairs_absolute_bottleneck_difference_ge10pp=0, best_switch_bottleneck_a=np.nan, best_switch_bottleneck_b=np.nan, best_switch_high_path_side='', endpoint0_recipe='', endpoint1_recipe='', endpoint0_yield_a=np.nan, endpoint1_yield_a=np.nan, endpoint0_yield_b=np.nan, endpoint1_yield_b=np.nan, path_recipe_ids_a='[]', path_recipe_ids_b='[]', path_yields_a='[]', path_yields_b='[]')
            if pairs:
                ca = np.array([capacities_a[a, b] for a, b in pairs])
                cb = np.array([capacities_b[a, b] for a, b in pairs])
                assert np.isfinite(ca).all() and np.isfinite(cb).all()
                difference = np.abs(ca - cb)
                switches = (ca >= 0.2) != (cb >= 0.2)
                maximum = int(np.argmax(difference))
                maximum_a, maximum_b = pairs[maximum]
                row.update(maximum_absolute_bottleneck_difference=float(difference.max()), maximum_difference_endpoint0_recipe=recipes[maximum_a], maximum_difference_endpoint1_recipe=recipes[maximum_b], maximum_difference_bottleneck_a=float(ca[maximum]), maximum_difference_bottleneck_b=float(cb[maximum]), maximum_difference_signed_a_minus_b=float(ca[maximum] - cb[maximum]), n_pairs_absolute_bottleneck_difference_ge5pp=int((difference >= 0.05).sum()), n_pairs_absolute_bottleneck_difference_ge10pp=int((difference >= 0.1).sum()), n_pairs_connectivity_switch=int(switches.sum()), n_pairs_connectivity_switch_gap5pp=int((switches & (difference >= 0.05)).sum()), n_pairs_connectivity_switch_gap10pp=int((switches & (difference >= 0.1)).sum()), any_connectivity_switch=bool(switches.any()), any_connectivity_switch_gap5pp=bool((switches & (difference >= 0.05)).any()), any_connectivity_switch_gap10pp=bool((switches & (difference >= 0.1)).any()))
                if switches.any():
                    choices = np.flatnonzero(switches)
                    chosen = max(choices, key=lambda j: (difference[j], min(values_a[list(pairs[j])].min(), values_b[list(pairs[j])].min())))
                    a, b = pairs[chosen]
                    path_a, path_b = (widest_path(values_a, adjacent, a, b), widest_path(values_b, adjacent, a, b))
                    assert min(values_a[path_a]) == ca[chosen] and min(values_b[path_b]) == cb[chosen]
                    row.update(best_switch_bottleneck_difference=float(difference[chosen]), best_switch_bottleneck_a=float(ca[chosen]), best_switch_bottleneck_b=float(cb[chosen]), best_switch_high_path_side='a' if ca[chosen] >= 0.2 else 'b', endpoint0_recipe=recipes[a], endpoint1_recipe=recipes[b], endpoint0_yield_a=float(values_a[a]), endpoint1_yield_a=float(values_a[b]), endpoint0_yield_b=float(values_b[a]), endpoint1_yield_b=float(values_b[b]), path_recipe_ids_a=packed([recipes[i] for i in path_a]), path_recipe_ids_b=packed([recipes[i] for i in path_b]), path_yields_a=packed(values_a[path_a].tolist()), path_yields_b=packed(values_b[path_b].tolist()))
                    graph_pairs[candidate['candidate_id'], graph_pair_id] = (recipes, values_a, values_b, adjacent)
            results.append(row)
        if count % 200 == 0:
            print(f'Compared {count}/{len(candidates)} molecular candidates; {len(results)} complete acquisition pairs', flush=True)
    results = pd.DataFrame(results)
    assert set(results.candidate_id) == set(candidates.candidate_id)
    return (results, graph_pairs)

def summarize(candidates, mapping, results):
    flags = ['any_connectivity_switch', 'any_connectivity_switch_gap5pp', 'any_connectivity_switch_gap10pp']
    by_candidate = results.groupby('candidate_id', as_index=False).agg(n_complete_acquisition_pairs=('graph_pair_id', 'size'), n_unique_graph_pairs=('graph_pair_id', 'nunique'), n_pairs_preserving_original_reversal5pp=('acquisition_pair_reversal_5pp', 'sum'), maximum_common_measured_recipes=('n_common_measured_recipes', 'max'), maximum_common_high_recipes=('n_common_high_recipes', 'max'), maximum_absolute_bottleneck_difference=('maximum_absolute_bottleneck_difference', 'max'), maximum_switch_bottleneck_difference=('best_switch_bottleneck_difference', 'max'), **{flag: (flag, 'max') for flag in flags})
    candidate_summary = candidates.merge(by_candidate, on='candidate_id', validate='one_to_one')
    save(candidate_summary, 'all_769_candidate_summary.csv.gz')
    summary_rows = []
    for name, frame in [('all_769', results), ('exact_neighbor_55', results[results.has_existing_exact_neighbor_rule]), ('round4_rule_30', results[results.has_round4_allowed_exact_rule])]:
        for gap, flag in [(0.0, flags[0]), (0.05, flags[1]), (0.1, flags[2])]:
            event = frame[frame[flag]]
            summary_rows.append(dict(panel=name, minimum_bottleneck_gap=gap, n_candidates=frame.candidate_id.nunique(), n_acquisition_pairs=len(frame), n_unique_graph_pairs=frame.graph_pair_id.nunique(), n_target_pairs=frame.target_pair_id.nunique(), n_event_candidates=event.candidate_id.nunique(), n_event_unique_graph_pairs=event.graph_pair_id.nunique(), n_event_target_pairs=event.target_pair_id.nunique(), n_event_molecular_pairs=event[['dataset_id', 'molecular_pair_id']].drop_duplicates().shape[0], n_event_datasets=event.dataset_id.nunique(), n_event_classes=event.rxn_class.nunique(), n_event_candidates_preserving_original_reversal5pp=event[event.acquisition_pair_reversal_5pp].candidate_id.nunique()))
    save(pd.DataFrame(summary_rows), 'aggregate_results.csv')
    swap = read('results/matched_swap_summary.csv').groupby('dataset_id').size()
    coverage = []
    for row in mapping.to_dict('records'):
        frame = results[results.dataset_id.eq(row['dataset_id'])]
        original = candidates[candidates.dataset_id.eq(row['dataset_id'])]
        coverage.append(dict(row, n_legacy_swaps=int(swap.get(row['dataset_id'], 0)), n_input_reversal5pp_candidates=len(original), n_input_exact_neighbor_candidates=int(original.has_existing_exact_neighbor_rule.sum()), n_complete_acquisition_pairs=len(frame), n_unique_graph_pairs=frame.graph_pair_id.nunique(), n_event_candidates=frame.loc[frame.any_connectivity_switch, 'candidate_id'].nunique(), n_gap5pp_event_candidates=frame.loc[frame.any_connectivity_switch_gap5pp, 'candidate_id'].nunique(), n_gap10pp_event_candidates=frame.loc[frame.any_connectivity_switch_gap10pp, 'candidate_id'].nunique(), support_status='common_measured_recipe_graphs_available' if len(frame) else 'no_original_5pp_molecular_reversal_comparison'))
    save(pd.DataFrame(coverage), 'dataset_coverage.csv')
    return (candidate_summary, pd.DataFrame(summary_rows))

def examples(results, graph_pairs, graphs, norm):
    event = results[results.any_connectivity_switch_gap5pp & results.acquisition_pair_reversal_5pp].copy()
    event['minimum_four_endpoint_yield'] = event[['endpoint0_yield_a', 'endpoint1_yield_a', 'endpoint0_yield_b', 'endpoint1_yield_b']].min(axis=1)
    event = event.sort_values(['best_switch_bottleneck_difference', 'minimum_four_endpoint_yield', 'n_common_measured_recipes'], ascending=False)
    selected = pd.concat([event.drop_duplicates('rxn_class'), event[event.has_existing_exact_neighbor_rule].drop_duplicates('rxn_class')]).drop_duplicates(['candidate_id', 'graph_pair_id'])
    selected['case_id'] = ['m_' + sid([a, b]) for a, b in selected[['candidate_id', 'graph_pair_id']].to_numpy()]
    path_rows, source_requests = ([], [])
    for case in selected.to_dict('records'):
        dataset = case['dataset_id']
        recipes, ya, yb, adjacent = graph_pairs[case['candidate_id'], case['graph_pair_id']]
        lookup, graph = ({recipe: j for j, recipe in enumerate(recipes)}, graphs[dataset])
        all_path = set()
        for side in ['a', 'b']:
            path = json.loads(case['path_recipe_ids_' + side])
            all_path.update(path)
            for step, recipe in enumerate(path):
                node = lookup[recipe]
                slot, old, new = ('', '', '')
                if step:
                    diff = np.flatnonzero(graph['tokens'][graph['lookup'][recipe]] != graph['tokens'][graph['lookup'][path[step - 1]]])
                    assert len(diff) == 1
                    slot = SLOTS[diff[0]]
                    old, new = (graph['labels'][path[step - 1]][slot], graph['labels'][recipe][slot])
                path_rows.append(dict(case_id=case['case_id'], candidate_id=case['candidate_id'], graph_pair_id=case['graph_pair_id'], dataset_id=dataset, path_optimized_for=side, path_step=step, recipe_id=recipe, yield_a=float(ya[node]), yield_b=float(yb[node]), full_recipe_text=graph['full'][recipe], full_recipe_tokens=packed({s: json.loads(graph['tokens'][graph['lookup'][recipe], j]) for j, s in enumerate(SLOTS)}), changed_slot=slot, previous_component=old, new_component=new))
        four = json.loads(case['four_recipe_ids'])
        for side in ['a', 'b']:
            for recipe in sorted(all_path | set(four)):
                source_requests.append(dict(case_id=case['case_id'], measured_side=side, dataset_id=dataset, archive=case['archive_' + side], plate=case['plate_' + side], acquisition=case['acquisition_' + side], target_key=case['target_key_' + side], recipe_id=recipe, in_selected_path=recipe in all_path, original_four_corner=['00', '10', '01', '11'][four.index(recipe)] if recipe in four else ''))
    paths = pd.DataFrame(path_rows)
    source = pd.DataFrame(source_requests).merge(norm, on=[*BLOCK, 'recipe_id'], validate='many_to_many')
    assert not source.duplicated(['case_id', 'measured_side', 'recipe_id', 'record_id']).any()
    locators = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    source = source.merge(locators, on='record_id', how='left', validate='many_to_one')
    assert source.source_file.notna().all()
    raw_root = RAW_DIR
    source['source_absolute_path'] = source.source_file.map(lambda name: str(raw_root / name))
    recovered = source.groupby(['case_id', 'measured_side', 'recipe_id']).yield_numeric.mean()
    for row in paths.to_dict('records'):
        for side in ['a', 'b']:
            assert abs(recovered.loc[row['case_id'], side, row['recipe_id']] - row['yield_' + side]) < 1e-12
    save(selected, 'selected_molecular_pathway_examples.csv')
    save(paths, 'selected_path_steps.csv')
    save(source, 'selected_original_records.csv.gz')
    return (selected, paths, source)

def report(candidates, results, aggregate, selected, paths, source, started):
    summary = dict(study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, input_candidates=len(candidates), input_exact_neighbor_candidates=int(candidates.has_existing_exact_neighbor_rule.sum()), acquisition_pair_rows=len(results), unique_graph_pairs=results.graph_pair_id.nunique(), target_pairs=results.target_pair_id.nunique(), rows_with_same_archive_plate_acquisition=int((results.archive_a.eq(results.archive_b) & results.plate_a.eq(results.plate_b) & results.acquisition_a.eq(results.acquisition_b)).sum()), aggregate_results=aggregate.to_dict('records'), selected_cases=len(selected), selected_path_steps=len(paths), source_record_rows=len(source), unique_source_records=source.record_id.nunique(), elapsed_seconds=round(time.time() - started, 2))
    (OUT / 'run_summary.json').write_text(packed(summary) + '\n', encoding='utf-8')
    lines = ['# Molecular changes rewire alternative paths between recipes', '', f"## All {summary['input_candidates']} comparisons", '', f"Reuses all {summary['input_candidates']} 5pp reversed comparisons from the molecular pairing analysis, including {summary['input_exact_neighbor_candidates']} exact-neighbour comparisons. Recipe yields of each molecule are first averaged by recipe within its original strict dataset/archive/plate/acquisition/target, and only recipes measured for both molecules are kept. The original four cells must be completely measured within the acquisition of each side; combinations of different acquisitions are kept pair by pair, and acquisition pairs are not treated as independent experiments. The full study still retains 20 datasets, 18 reaction classes and 4,002 swaps; the {summary['input_candidates']} input comparisons cover {candidates.rxn_class.nunique()} reaction classes, and the rest are listed individually in dataset_coverage.csv.", '', f"Obtained {len(results)} candidate × complete acquisition pairs, {results.graph_pair_id.nunique()} distinct actual graph pairs and {results.target_pair_id.nunique()} deduplicated target pairs. Comparisons with strictly identical archive/plate/acquisition across molecules: {summary['rows_with_same_archive_plate_acquisition']}; yields on each side come from their own complete four-cell acquisitions.", '', 'Core event requirement: the same two endpoint recipes give ≥20% for both molecules; one side has a path ≥20% throughout within the common measured graph, while the best-path bottleneck on the other side stays <20%. The change therefore lies in the availability of intermediate alternative combinations, while the endpoint recipes work for both molecules.', '', '| Comparison set | Bottleneck difference threshold | Event candidates/all candidates | Deduplicated target pairs | Deduplicated actual graph pairs | Datasets/reaction classes |', '|---|---|---|---:|---:|---|']
    for row in aggregate.to_dict('records'):
        lines.append(f"| {row['panel']} | {100 * row['minimum_bottleneck_gap']:.0f}pp | {row['n_event_candidates']}/{row['n_candidates']} | {row['n_event_target_pairs']} | {row['n_event_unique_graph_pairs']} | {row['n_event_datasets']}/{row['n_event_classes']} |")
    lines += ['', f"The continuous difference is the absolute difference in best-path bottleneck between the two molecules for the same endpoints; the 5/10pp rows in the table also require the event (one side ≥20%, the other <20%). The full candidate summary still has {summary['input_candidates']} rows; the {summary['input_exact_neighbor_candidates']} exact-neighbour rows and {int(candidates.has_round4_allowed_exact_rule.sum())} round-4 rule rows are listed separately and do not replace the total. When the common graph has no two common high-yield endpoints, or the endpoints are not in the same component of the full graph, the bottleneck difference is recorded as missing, not zero.", '', '## Specific molecules and paths', '']
    for case in selected.to_dict('records'):
        lines += [f"### {case['rxn_class']} `{case['case_id']}`", '', f"Fixed partner: `{case['partner_smiles']}`; changed side: {case['role']}. A: `{case['molecule_a']}`; B: `{case['molecule_b']}`. Structural relation: {case['structural_change_a_to_b']} ({case['structural_relation']}).", '', f"Common measured recipes: {int(case['n_common_measured_recipes'])}; {int(case['n_common_graph_edges'])} one-slot edges; ≥20% nodes for A/B: {int(case['n_high_recipes_a'])}/{int(case['n_high_recipes_b'])}; high-yield connected groups: {int(case['n_high_groups_a'])}/{int(case['n_high_groups_b'])}.", '', f"Yields of the same endpoints for A: {100 * case['endpoint0_yield_a']:.1f}%, {100 * case['endpoint1_yield_a']:.1f}%; for B: {100 * case['endpoint0_yield_b']:.1f}%, {100 * case['endpoint1_yield_b']:.1f}%. Best bottleneck A/B: {100 * case['best_switch_bottleneck_a']:.1f}%/{100 * case['best_switch_bottleneck_b']:.1f}%; difference {100 * case['best_switch_bottleneck_difference']:.1f}pp.", '']
        high = case['best_switch_high_path_side']
        steps = paths[paths.case_id.eq(case['case_id']) & paths.path_optimized_for.eq(high)].sort_values('path_step')
        lines.append('Yields at each node for A/B along the best path of the side that stays high-yield throughout: ' + ' → '.join((f'{100 * r.yield_a:.1f}/{100 * r.yield_b:.1f}%' for r in steps.itertuples())) + '.')
        lines.append('Full starting conditions: ' + steps.iloc[0].full_recipe_text + '.')
        for row in steps.iloc[1:].itertuples():
            lines.append(f'- {row.changed_slot}: {row.previous_component} → {row.new_component}; A/B yields {100 * row.yield_a:.1f}/{100 * row.yield_b:.1f}%.')
        lines.append('')
    lines += ['## Outputs and methods', '', 'all_candidate_acquisition_graph_pairs.csv.gz stores all candidate × acquisition pairs, with complete acquisition locations for both sides, common recipes, original four cells, high-yield connected groups on both sides, support for common high-yield endpoint pairs, continuous maximum bottleneck difference and 5/10pp events. all_769_candidate_summary.csv.gz takes the union of acquisition pairs per candidate and keeps all fields of the original molecule comparison. selected_path_steps.csv gives, for each path node, the yields on both sides, full conditions and the actual one-slot change; selected_original_records.csv.gz contains the source rows and record_id of the union of the selected paths on both sides and the original four cells.', '', 'Edges are defined by exact tokens over nine slots: identity and amount for each of six reagents, plus solvent, temperature and time. Both sides have measured yields within the common graph, and zero values are kept; missing measurements do not create low-yield nodes. Best bottlenecks for all endpoint pairs are computed by a max-min Floyd closure, and displayed paths are found by max-min Dijkstra. The high-yield threshold is fixed at 20%; the absolute bottleneck difference is saved continuously, with 5/10pp listed separately. Strong events and exact-neighbour events are selected for each reaction class; the original four-cell reversal still meets 5pp within the selected actual acquisition pair.', '', 'Candidates share molecules, recipes, rectangles and source records. Candidates, target pairs and actual acquisition graph pairs are each deduplicated in the summaries, and no observation is counted as an independent experiment.', '']
    (OUT / 'MOLECULAR_CONNECTIVITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(packed(summary), flush=True)

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    self_checks()
    candidates, mapping, graphs, norm, measured, block_lookup = prepare()
    results, graph_pairs = analyze(candidates, graphs, measured, block_lookup)
    save(results, 'all_candidate_acquisition_graph_pairs.csv.gz')
    _, aggregate = summarize(candidates, mapping, results)
    selected, paths, source = examples(results, graph_pairs, graphs, norm)
    report(candidates, results, aggregate, selected, paths, source, started)
if __name__ == '__main__':
    main()
