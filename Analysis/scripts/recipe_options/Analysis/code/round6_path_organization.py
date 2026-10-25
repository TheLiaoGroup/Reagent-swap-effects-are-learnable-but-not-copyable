"""Separate endpoint loss, yield ordering, and equal-size high-recipe organization."""
from __future__ import annotations
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
import round6_molecular_connectivity as molecular
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round6/path_organization'
EPS = 1e-12

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression='gzip' if name.endswith('.gz') else None)

def rank_quantiles(values):
    if len(values) == 1:
        return np.array([0.5])
    return (pd.Series(values).rank(method='average').to_numpy(float) - 1) / (len(values) - 1)

def group_count(adjacent, selected):
    indices = np.flatnonzero(selected)
    return connected_components(csr_matrix(adjacent[np.ix_(indices, indices)]), directed=False, return_labels=False) if len(indices) else 0

def top_k_comparisons(ya, yb, adjacent, ca=None, cb=None):
    ca = molecular.widest_matrix(ya, adjacent) if ca is None else ca
    cb = molecular.widest_matrix(yb, adjacent) if cb is None else cb
    ordered_a, ordered_b = (np.sort(ya)[::-1], np.sort(yb)[::-1])
    _, full_labels = connected_components(csr_matrix(adjacent), directed=False)
    ii, jj = np.triu_indices(len(ya), 1)
    same_component = full_labels[ii] == full_labels[jj]
    rows = []
    for k in range(2, len(ya)):
        tie_a, tie_b = (ordered_a[k - 1] == ordered_a[k], ordered_b[k - 1] == ordered_b[k])
        status = 'boundary_tie_both' if tie_a and tie_b else 'boundary_tie_a' if tie_a else 'boundary_tie_b' if tie_b else 'available'
        row = dict(k=k, fraction_selected=k / len(ya), n_common_measured_recipes=len(ya), cutoff_yield_a=float(ordered_a[k - 1]), cutoff_yield_b=float(ordered_b[k - 1]), boundary_tie_a=bool(tie_a), boundary_tie_b=bool(tie_b), status=status, n_high_groups_a=np.nan, n_high_groups_b=np.nan, n_common_selected_nodes=np.nan, n_shared_endpoint_pairs=np.nan, n_shared_endpoint_pairs_with_full_graph_path=np.nan, n_endpoint_pairs_connected_a=np.nan, n_endpoint_pairs_connected_b=np.nan, n_endpoint_connectivity_switches=np.nan, any_endpoint_connectivity_switch=False, same_number_high_groups=False, switch_with_same_number_high_groups=False, example_endpoint0_index=-1, example_endpoint1_index=-1)
        if tie_a or tie_b:
            rows.append(row)
            continue
        ha, hb = (ya >= ordered_a[k - 1], yb >= ordered_b[k - 1])
        assert ha.sum() == hb.sum() == k
        common = ha & hb
        eligible = common[ii] & common[jj] & same_component
        connected_a, connected_b = (ca[ii, jj] >= ordered_a[k - 1], cb[ii, jj] >= ordered_b[k - 1])
        switches = eligible & (connected_a != connected_b)
        groups_a, groups_b = (group_count(adjacent, ha), group_count(adjacent, hb))
        row.update(n_high_groups_a=int(groups_a), n_high_groups_b=int(groups_b), n_common_selected_nodes=int(common.sum()), n_shared_endpoint_pairs=int(common.sum() * (common.sum() - 1) // 2), n_shared_endpoint_pairs_with_full_graph_path=int(eligible.sum()), n_endpoint_pairs_connected_a=int((eligible & connected_a).sum()), n_endpoint_pairs_connected_b=int((eligible & connected_b).sum()), n_endpoint_connectivity_switches=int(switches.sum()), any_endpoint_connectivity_switch=bool(switches.any()), same_number_high_groups=groups_a == groups_b, switch_with_same_number_high_groups=bool(switches.any() and groups_a == groups_b))
        if not eligible.any():
            row['status'] = 'no_shared_endpoint_pair_with_full_graph_path'
        if switches.any():
            index = np.flatnonzero(switches)[np.argmax(np.abs(ca[ii[switches], jj[switches]] - cb[ii[switches], jj[switches]]))]
            row['example_endpoint0_index'], row['example_endpoint1_index'] = (int(ii[index]), int(jj[index]))
        rows.append(row)
    return rows

def self_checks():
    adjacent = np.zeros((6, 6), bool)
    for i in range(5):
        adjacent[i, i + 1] = adjacent[i + 1, i] = True
    ya = np.array([0.9, 0.85, 0.8, 0.1, 0.2, 0.7])
    yb = np.array([0.9, 0.1, 0.8, 0.7, 0.6, 0.2])
    ra, rb = (rank_quantiles(ya), rank_quantiles(yb))
    assert np.allclose(ra, rank_quantiles(3 * ya + 2))
    assert np.allclose(ra, rank_quantiles(np.exp(ya)))
    assert np.allclose(rank_quantiles(np.array([0.0, 0.0, 1.0, 2.0])), [1 / 6, 1 / 6, 2 / 3, 1.0])
    ca, cb = (molecular.widest_matrix(ya, adjacent), molecular.widest_matrix(yb, adjacent))
    assert abs(min(ya[0], ya[2]) - ca[0, 2]) < EPS
    assert abs(min(yb[0], yb[2]) - cb[0, 2] - 0.7) < EPS
    rows = top_k_comparisons(ya, yb, adjacent, ca, cb)
    row = next((row for row in rows if row['k'] == 4))
    assert row['n_high_groups_a'] == row['n_high_groups_b'] == 2
    assert row['n_endpoint_connectivity_switches'] == 1
    tied = top_k_comparisons(np.array([0.0, 0.0, 0.8, 0.9]), np.array([0.0, 0.2, 0.8, 0.9]), adjacent[:4, :4])
    assert next((row for row in tied if row['k'] == 3))['status'] == 'boundary_tie_a'

def reconstruct(row, graphs, measured):
    dataset = row['dataset_id']
    keys = {side: tuple([dataset] + [row[field + '_' + side] for field in ['archive', 'plate', 'acquisition', 'target_key']]) for side in ['a', 'b']}
    recipes = json.loads(row['common_recipe_ids'])
    expected = sorted(set(measured[keys['a']].index) & set(measured[keys['b']].index))
    assert recipes == expected
    graph = graphs[dataset]
    indices = [graph['lookup'][recipe] for recipe in recipes]
    adjacent = graph['adjacency'][np.ix_(indices, indices)]
    ya, yb = (measured[keys[side]].loc[recipes].to_numpy(float) for side in ['a', 'b'])
    assert len(recipes) == row['n_common_measured_recipes'] and adjacent.sum() // 2 == row['n_common_graph_edges']
    return (recipes, ya, yb, adjacent, keys)

def analyze(input_rows, graphs, measured):
    observations, pair_rows, k_rows, node_rows, requests = ([], [], [], [], [])
    cache = {}
    for index, original in enumerate(input_rows.to_dict('records'), 1):
        row = dict(original)
        recipes, ya, yb, adjacent, keys = reconstruct(row, graphs, measured)
        n = len(recipes)
        ca, cb = (molecular.widest_matrix(ya, adjacent), molecular.widest_matrix(yb, adjacent))
        ra, rb = (rank_quantiles(ya), rank_quantiles(yb))
        rca, rcb = (molecular.widest_matrix(ra, adjacent), molecular.widest_matrix(rb, adjacent))
        _, full_labels = connected_components(csr_matrix(adjacent), directed=False)
        ranks_a_monotone = rank_quantiles(3.1 * ya + 0.13)
        assert np.allclose(ra, ranks_a_monotone)
        metadata = {key: row[key] for key in ['candidate_id', 'graph_pair_id', 'target_pair_id', 'dataset_id', 'rxn_class']}
        top_rows = top_k_comparisons(ya, yb, adjacent, ca, cb)
        valid_k = [r for r in top_rows if not r['boundary_tie_a'] and (not r['boundary_tie_b'])]
        for kr in top_rows:
            kr.update(metadata)
            for end in [0, 1]:
                pos = kr.pop(f'example_endpoint{end}_index')
                kr[f'example_endpoint{end}_recipe'] = recipes[pos] if pos >= 0 else ''
        k_rows.extend(top_rows)
        high = np.flatnonzero((ya >= 0.2) & (yb >= 0.2))
        pairs = [(i, j) for pos, i in enumerate(high) for j in high[pos + 1:] if full_labels[i] == full_labels[j]]
        local_pairs = []
        for i, j in pairs:
            lower_a, lower_b = (min(ya[i], ya[j]), min(yb[i], yb[j]))
            rank_lower_a, rank_lower_b = (min(ra[i], ra[j]), min(rb[i], rb[j]))
            penalty_a, penalty_b = (max(0.0, lower_a - ca[i, j]), max(0.0, lower_b - cb[i, j]))
            rank_penalty_a, rank_penalty_b = (max(0.0, rank_lower_a - rca[i, j]), max(0.0, rank_lower_b - rcb[i, j]))
            switches_k = [kr for kr in valid_k if lower_a >= kr['cutoff_yield_a'] and lower_b >= kr['cutoff_yield_b'] and ((ca[i, j] >= kr['cutoff_yield_a']) != (cb[i, j] >= kr['cutoff_yield_b']))]
            selected_k = sorted(switches_k, key=lambda r: (not r['same_number_high_groups'], abs(r['fraction_selected'] - 0.5), r['k']))
            pair = dict(metadata, endpoint_pair_id='e_' + molecular.sid([row['graph_pair_id'], recipes[i], recipes[j]]), endpoint0_recipe=recipes[i], endpoint1_recipe=recipes[j], endpoint0_yield_a=float(ya[i]), endpoint1_yield_a=float(ya[j]), endpoint0_yield_b=float(yb[i]), endpoint1_yield_b=float(yb[j]), lower_endpoint_yield_a=float(lower_a), lower_endpoint_yield_b=float(lower_b), best_bottleneck_a=float(ca[i, j]), best_bottleneck_b=float(cb[i, j]), endpoint_penalty_a=float(penalty_a), endpoint_penalty_b=float(penalty_b), endpoint_penalty_signed_a_minus_b=float(penalty_a - penalty_b), endpoint_penalty_absolute_difference=float(abs(penalty_a - penalty_b)), bottleneck_absolute_difference=float(abs(ca[i, j] - cb[i, j])), raw_threshold_connectivity_switch=bool((ca[i, j] >= 0.2) != (cb[i, j] >= 0.2)), rank_lower_endpoint_a=float(rank_lower_a), rank_lower_endpoint_b=float(rank_lower_b), rank_bottleneck_a=float(rca[i, j]), rank_bottleneck_b=float(rcb[i, j]), rank_endpoint_penalty_a=float(rank_penalty_a), rank_endpoint_penalty_b=float(rank_penalty_b), rank_penalty_signed_a_minus_b=float(rank_penalty_a - rank_penalty_b), rank_penalty_absolute_difference=float(abs(rank_penalty_a - rank_penalty_b)), rank_bottleneck_absolute_difference=float(abs(rca[i, j] - rcb[i, j])), n_equal_count_topk_switches=len(switches_k), n_equal_count_same_group_number_topk_switches=sum((kr['same_number_high_groups'] for kr in switches_k)), example_topk_k=selected_k[0]['k'] if selected_k else np.nan, example_topk_groups_a=selected_k[0]['n_high_groups_a'] if selected_k else np.nan, example_topk_groups_b=selected_k[0]['n_high_groups_b'] if selected_k else np.nan, example_topk_cutoff_a=selected_k[0]['cutoff_yield_a'] if selected_k else np.nan, example_topk_cutoff_b=selected_k[0]['cutoff_yield_b'] if selected_k else np.nan, example_topk_selected_count_a=selected_k[0]['k'] if selected_k else np.nan, example_topk_selected_count_b=selected_k[0]['k'] if selected_k else np.nan, example_topk_endpoints_connected_a=bool(ca[i, j] >= selected_k[0]['cutoff_yield_a']) if selected_k else '', example_topk_endpoints_connected_b=bool(cb[i, j] >= selected_k[0]['cutoff_yield_b']) if selected_k else '')
            assert penalty_a <= lower_a + EPS and penalty_b <= lower_b + EPS
            local_pairs.append(pair)
        pair_rows.extend(local_pairs)
        row.update(n_endpoint_pairs_evaluated=len(local_pairs), endpoint_penalty_status='available' if pairs else 'no_common_high_endpoint_pair_with_full_graph_path', n_fixed_topk_levels=n - 2, n_fixed_topk_tie_excluded_levels=len(top_rows) - len(valid_k), n_fixed_topk_boundary_available_levels=len(valid_k), n_fixed_topk_endpoint_comparable_levels=sum((r['n_shared_endpoint_pairs_with_full_graph_path'] > 0 for r in valid_k)), n_fixed_topk_switch_levels=sum((r['any_endpoint_connectivity_switch'] for r in valid_k)), n_fixed_topk_same_group_number_switch_levels=sum((r['switch_with_same_number_high_groups'] for r in valid_k)), any_fixed_topk_connectivity_switch=any((r['any_endpoint_connectivity_switch'] for r in valid_k)), any_fixed_topk_same_group_number_connectivity_switch=any((r['switch_with_same_number_high_groups'] for r in valid_k)), same_raw_high_node_count=row['n_high_recipes_a'] == row['n_high_recipes_b'], same_raw_high_group_count=row['n_high_groups_a'] == row['n_high_groups_b'])
        for metric, column in [('endpoint_penalty', 'endpoint_penalty_absolute_difference'), ('rank_penalty', 'rank_penalty_absolute_difference')]:
            row['maximum_' + metric + '_absolute_difference'] = max((p[column] for p in local_pairs), default=np.nan)
            thresholds = [0.05, 0.1] if metric == 'endpoint_penalty' else [0.1, 0.2]
            for threshold in thresholds:
                row[f'any_{metric}_difference_ge{round(100 * threshold)}'] = any((p[column] >= threshold - EPS for p in local_pairs))
                row[f'n_pairs_{metric}_difference_ge{round(100 * threshold)}'] = sum((p[column] >= threshold - EPS for p in local_pairs))
        row['n_common_high_pairs_with_equal_count_topk_switch'] = sum((p['n_equal_count_topk_switches'] > 0 for p in local_pairs))
        row['any_common_high_pair_equal_count_topk_switch'] = row['n_common_high_pairs_with_equal_count_topk_switch'] > 0
        row['n_common_high_pairs_with_equal_count_same_group_topk_switch'] = sum((p['n_equal_count_same_group_number_topk_switches'] > 0 for p in local_pairs))
        row['any_common_high_pair_equal_count_same_group_topk_switch'] = row['n_common_high_pairs_with_equal_count_same_group_topk_switch'] > 0
        observations.append(row)
        graph = graphs[row['dataset_id']]
        for position, recipe in enumerate(recipes):
            node_rows.append(dict(metadata, recipe_id=recipe, yield_a=float(ya[position]), yield_b=float(yb[position]), rank_quantile_a=float(ra[position]), rank_quantile_b=float(rb[position]), full_recipe_text=graph['full'][recipe], full_recipe_tokens=molecular.packed({slot: json.loads(graph['tokens'][graph['lookup'][recipe], j]) for j, slot in enumerate(molecular.SLOTS)})))
            for side in ['a', 'b']:
                requests.append(dict(metadata, measured_side=side, **dict(zip(molecular.BLOCK[1:], keys[side][1:])), recipe_id=recipe))
        cache[row['candidate_id']] = (recipes, ya, yb, ra, rb, adjacent, keys)
        if index % 150 == 0:
            print(f'Path organization {index}/{len(input_rows)}; endpoint pairs={len(pair_rows)}', flush=True)
    return (pd.DataFrame(observations), pd.DataFrame(pair_rows), pd.DataFrame(k_rows), pd.DataFrame(node_rows), pd.DataFrame(requests), cache)

def summarize(observations, topk, mapping):
    measures = ['any_endpoint_penalty_difference_ge5', 'any_endpoint_penalty_difference_ge10', 'any_rank_penalty_difference_ge10', 'any_rank_penalty_difference_ge20', 'any_fixed_topk_connectivity_switch', 'any_fixed_topk_same_group_number_connectivity_switch', 'any_common_high_pair_equal_count_topk_switch', 'any_common_high_pair_equal_count_same_group_topk_switch']
    aggregate = []
    for panel, part in [('all_769', observations), ('exact_neighbor_55', observations[observations.has_existing_exact_neighbor_rule]), ('round4_rule_30', observations[observations.has_round4_allowed_exact_rule])]:
        for metric in measures:
            event = part[part[metric]]
            aggregate.append(dict(panel=panel, metric=metric, n_input_candidates=part.candidate_id.nunique(), n_graph_pairs=part.graph_pair_id.nunique(), n_endpoint_comparable_candidates=part.loc[part.n_endpoint_pairs_evaluated.gt(0), 'candidate_id'].nunique(), n_topk_endpoint_comparable_candidates=part.loc[part.n_fixed_topk_endpoint_comparable_levels.gt(0), 'candidate_id'].nunique(), n_event_candidates=event.candidate_id.nunique(), n_event_graph_pairs=event.graph_pair_id.nunique(), n_event_target_pairs=event.target_pair_id.nunique(), n_event_datasets=event.dataset_id.nunique(), n_event_reaction_classes=event.rxn_class.nunique(), n_event_candidates_preserving_original_reversal5=event.loc[event.acquisition_pair_reversal_5pp, 'candidate_id'].nunique()))
    aggregate = pd.DataFrame(aggregate)
    swaps = molecular.read('results/matched_swap_summary.csv').groupby('dataset_id').size()
    coverage = []
    for original in mapping.to_dict('records'):
        part = observations[observations.dataset_id.eq(original['dataset_id'])]
        row = dict(original, n_legacy_swaps=int(swaps.get(original['dataset_id'], 0)), n_input_candidates=len(part), n_unique_graph_pairs=part.graph_pair_id.nunique(), n_endpoint_comparable_candidates=int(part.n_endpoint_pairs_evaluated.gt(0).sum()), n_topk_endpoint_comparable_candidates=int(part.n_fixed_topk_endpoint_comparable_levels.gt(0).sum()), status='all_input_common_recipe_graphs_analyzed' if len(part) else 'no_input_5pp_molecular_reversal_candidates')
        row.update({'n_' + metric: int(part[metric].sum()) for metric in measures})
        coverage.append(row)
    coverage = pd.DataFrame(coverage)
    assert len(coverage) == 20 and coverage.rxn_class.nunique() == 18 and (coverage.n_legacy_swaps.sum() == 4002)
    save(aggregate, 'aggregate_results.csv')
    save(coverage, 'dataset_coverage.csv')
    return (aggregate, coverage)
NAMES = {'Nc1ccc(Cl)cc1': '4-chloroaniline', 'CSc1ccc(N)cc1': '4-(methylthio)aniline', 'Cc1cc(C(=O)O)cc(Cl)n1': '2-chloro-6-methylpyridine-4-carboxylic acid', 'CC1(C)OB(c2ccc(CC#N)cc2)OC1(C)C': '4-(cyanomethyl)phenylboronic acid pinacol ester', 'CC1(C)OB(c2ccc(C#N)cc2)OC1(C)C': '4-cyanophenylboronic acid pinacol ester', 'Cc1cnc(Cl)nc1': '2-chloro-5-methylpyrimidine'}

def select_examples(observations, pairs, topk, cache, graphs):
    reference_examples = molecular.read('round6/molecular_connectivity/selected_molecular_pathway_examples.csv')
    candidates = observations[observations.n_endpoint_pairs_evaluated.gt(0)].copy()
    candidates = candidates.sort_values(['has_existing_exact_neighbor_rule', 'maximum_rank_penalty_absolute_difference', 'maximum_endpoint_penalty_absolute_difference'], ascending=False)
    selected = candidates.drop_duplicates('rxn_class').candidate_id.tolist()
    for a, b in [('Nc1ccc(Cl)cc1', 'CSc1ccc(N)cc1'), ('CC1(C)OB(c2ccc(CC#N)cc2)OC1(C)C', 'CC1(C)OB(c2ccc(C#N)cc2)OC1(C)C')]:
        match = candidates[candidates.molecule_a.eq(a) & candidates.molecule_b.eq(b)]
        if len(match):
            selected.append(match.iloc[0].candidate_id)
    selected = list(dict.fromkeys(selected))
    cases, paths = ([], [])
    for candidate_id in selected:
        observation = observations[observations.candidate_id.eq(candidate_id)].iloc[0].to_dict()
        frame = pairs[pairs.candidate_id.eq(candidate_id)].copy()
        frame['equal_count_supported'] = frame.n_equal_count_topk_switches.gt(0)
        frame = frame.sort_values(['equal_count_supported', 'endpoint_penalty_absolute_difference', 'rank_penalty_absolute_difference'], ascending=False)
        chosen = frame.iloc[0].to_dict()
        selection_reason = 'maximum_endpoint_penalty_with_equal_count_support_preferred'
        reference = reference_examples[reference_examples.candidate_id.eq(candidate_id)]
        if observation['molecule_a'] in ['Nc1ccc(Cl)cc1', 'CC1(C)OB(c2ccc(CC#N)cc2)OC1(C)C'] and len(reference):
            original = reference.iloc[0]
            original_pair = frame[frame.endpoint0_recipe.eq(original.endpoint0_recipe) & frame.endpoint1_recipe.eq(original.endpoint1_recipe)]
            assert len(original_pair) == 1 and original_pair.iloc[0].n_equal_count_topk_switches > 0
            chosen = original_pair.iloc[0].to_dict()
            selection_reason = 'original_molecular_path_endpoints'
            preferred_k = 22 if observation['molecule_a'] == 'Nc1ccc(Cl)cc1' else 14
            kr = topk[topk.candidate_id.eq(candidate_id) & topk.k.eq(preferred_k)].iloc[0]
            assert chosen['lower_endpoint_yield_a'] >= float(kr.cutoff_yield_a) and chosen['lower_endpoint_yield_b'] >= float(kr.cutoff_yield_b)
            connected_a = chosen['best_bottleneck_a'] >= float(kr.cutoff_yield_a)
            connected_b = chosen['best_bottleneck_b'] >= float(kr.cutoff_yield_b)
            assert connected_a != connected_b
            chosen.update(example_topk_k=preferred_k, example_topk_selected_count_a=preferred_k, example_topk_selected_count_b=preferred_k, example_topk_groups_a=float(kr.n_high_groups_a), example_topk_groups_b=float(kr.n_high_groups_b), example_topk_cutoff_a=float(kr.cutoff_yield_a), example_topk_cutoff_b=float(kr.cutoff_yield_b), example_topk_endpoints_connected_a=bool(connected_a), example_topk_endpoints_connected_b=bool(connected_b))
        case = dict(observation)
        case.update(chosen)
        case['endpoint_selection_reason'] = selection_reason
        for field in ['best_switch_bottleneck_difference', 'best_switch_bottleneck_a', 'best_switch_bottleneck_b', 'best_switch_high_path_side']:
            case.pop(field, None)
        case['case_id'] = 'o_' + molecular.sid([candidate_id, chosen['endpoint_pair_id']])
        for field in ['molecule_a', 'molecule_b', 'partner_smiles']:
            name_field = 'partner_name' if field == 'partner_smiles' else field.replace('molecule', 'molecule_name')
            case[name_field] = NAMES.get(case[field], case.get(name_field, '') or case[field])
        recipes, ya, yb, ra, rb, adjacent, keys = cache[candidate_id]
        lookup = {recipe: j for j, recipe in enumerate(recipes)}
        i, j = (lookup[chosen['endpoint0_recipe']], lookup[chosen['endpoint1_recipe']])
        for side, values in [('a', ya), ('b', yb)]:
            path = molecular.widest_path(values, adjacent, i, j)
            assert abs(min(values[path]) - chosen['best_bottleneck_' + side]) < EPS
            case['path_recipe_ids_' + side] = molecular.packed([recipes[position] for position in path])
            case['path_yields_' + side] = molecular.packed(values[path].tolist())
            case['path_rank_quantiles_' + side] = molecular.packed((ra if side == 'a' else rb)[path].tolist())
            for step, position in enumerate(path):
                recipe, graph = (recipes[position], graphs[case['dataset_id']])
                changed, previous, new = ('', '', '')
                if step:
                    old_recipe = recipes[path[step - 1]]
                    diff = np.flatnonzero(graph['tokens'][graph['lookup'][recipe]] != graph['tokens'][graph['lookup'][old_recipe]])
                    assert len(diff) == 1
                    changed = molecular.SLOTS[diff[0]]
                    previous, new = (graph['labels'][old_recipe][changed], graph['labels'][recipe][changed])
                paths.append(dict(case_id=case['case_id'], candidate_id=candidate_id, graph_pair_id=case['graph_pair_id'], path_optimized_for=side, path_step=step, recipe_id=recipe, yield_a=float(ya[position]), yield_b=float(yb[position]), rank_quantile_a=float(ra[position]), rank_quantile_b=float(rb[position]), full_recipe_text=graph['full'][recipe], changed_slot=changed, previous_component=previous, new_component=new))
        cases.append(case)
    selected, paths = (pd.DataFrame(cases), pd.DataFrame(paths))
    save(selected, 'selected_chemical_examples.csv')
    save(paths, 'selected_path_steps.csv')
    return (selected, paths)

def original_links(requests, norm, nodes, selected):
    links = requests.merge(norm, on=[*molecular.BLOCK, 'recipe_id'], validate='many_to_many')
    assert not links.duplicated(['candidate_id', 'graph_pair_id', 'measured_side', 'recipe_id', 'record_id']).any()
    raw = molecular.read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    links = links.merge(raw, on='record_id', how='left', validate='many_to_one')
    assert links.source_file.notna().all()
    raw_root = RAW_DIR
    links['source_absolute_path'] = links.source_file.map(lambda name: str(raw_root / name))
    recovered = links.groupby(['candidate_id', 'recipe_id', 'measured_side']).yield_numeric.mean().unstack('measured_side')
    check = nodes.set_index(['candidate_id', 'recipe_id'])
    for side in ['a', 'b']:
        assert np.allclose(recovered.loc[check.index, side], check['yield_' + side], atol=EPS)
    save(links, 'all_common_recipe_original_records.csv.gz')
    selected_links = links[links.candidate_id.isin(selected.candidate_id)].merge(selected[['candidate_id', 'case_id']], on='candidate_id', validate='many_to_one')
    save(selected_links, 'selected_original_records.csv.gz')
    return links

def write_report(observations, pairs, topk, aggregate, coverage, selected, paths, links, started):
    unique = observations.sort_values('candidate_id').drop_duplicates('graph_pair_id')
    summary = dict(study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, candidate_rows=len(observations), unique_graph_pairs=len(unique), target_pairs=observations.target_pair_id.nunique(), n_endpoint_comparable_candidates=int(observations.n_endpoint_pairs_evaluated.gt(0).sum()), n_endpoint_comparable_graph_pairs=observations.loc[observations.n_endpoint_pairs_evaluated.gt(0), 'graph_pair_id'].nunique(), endpoint_pair_rows=len(pairs), unique_graph_endpoint_pairs=pairs[['graph_pair_id', 'endpoint_pair_id']].drop_duplicates().shape[0], fixed_topk_rows=len(topk), n_topk_boundary_tie_excluded_rows=int((topk.boundary_tie_a | topk.boundary_tie_b).sum()), n_topk_endpoint_comparable_candidates=int(observations.n_fixed_topk_endpoint_comparable_levels.gt(0).sum()), n_topk_endpoint_comparable_graph_pairs=observations.loc[observations.n_fixed_topk_endpoint_comparable_levels.gt(0), 'graph_pair_id'].nunique(), n_original_switch_gap10_same_group_count=int((observations.any_connectivity_switch_gap10pp & observations.same_raw_high_group_count).sum()), n_original_switch_gap10_same_high_node_count=int((observations.any_connectivity_switch_gap10pp & observations.same_raw_high_node_count).sum()), source_record_links=len(links), unique_source_records=links.record_id.nunique(), selected_cases=len(selected), aggregation=aggregate.to_dict('records'), elapsed_seconds=round(time.time() - started, 2))
    (OUT / 'run_summary.json').write_text(molecular.packed(summary) + '\n', encoding='utf-8')
    lines = ['# 高产配方的数量与排列：端点损失及等数量连接', '', '## 问题与定义', '', '在同一共同实测配方图上，分子B的较低瓶颈可能来自端点本身较低，也可能来自中间配方排列改变。本专题逐层分开这两种来源，并把两侧高产节点数量固定相等后再比较相同端点的连接。', '', f"全研究保留20数据集、18反应类、4,002单槽交换。本专题完整接收{summary['candidate_rows']}个5pp分子配对反转候选，形成{len(observations)}个候选×采集图对、{summary['unique_graph_pairs']}个不同实际图对；未有输入候选的数据集仍逐项保留。每一侧使用其严格dataset/archive/plate/acquisition/target中的实测配方均值，两侧只保留共同节点，九槽精确token定义共同一槽图。", '', '1. 对两侧都≥20%、在完整共同图中可连的相同端点配方，端点损失penalty=min(两端产率)−最佳路径瓶颈。它量化路径相对本身端点还要下降多少。保存A、B损失及其差，不能把瓶颈差直接当成这一损失差。', '', '2. 每侧在相同N个共同节点内独立排名，分位r=(平均名次−1)/(N−1)，名次归一化到0至1尺度；同产率并列取平均名次（因此实际最低或最高分位可能不达0或1）。再求秩瓶颈及秩端点损失=min(两端分位)−最佳路径秩瓶颈。它们在任意严格递增的整体产率变换下保持不变，因此将整体平移、正比例缩放及其他保持排序的响应变换分离出去。', '', '3. 枚举k=2…N−1，各取本侧最高产的恰好k个节点。仅当两侧第k与k+1名都没有并列才比较；边界并列逐k登记，不随机拆开。只比较双方都入选、且在完整共同图中可连的相同端点。此处最高产指排序前k，不附绝对20%要求。另将双方端点均≥20%的结果单列，作为同样有效端点间连接改变的主体。两侧入选节点数量相同后，相同端点在一侧相连、另一侧分开，直接表示配方排列改变。另列连通组总数也相同而连接对象不同的情形。', '', '## 全量结果', '', f"绝对产率端点比较适用于{summary['n_endpoint_comparable_candidates']}/{summary['candidate_rows']}候选、{summary['n_endpoint_comparable_graph_pairs']}/{summary['unique_graph_pairs']}图对，共{len(pairs):,}个候选×端点对；去重实际图×端点对为{summary['unique_graph_endpoint_pairs']:,}。其余保留不可比较原因，连续量留空，不记零。", '', f"固定top-k共有{len(topk):,}个候选×k，边界并列排除{summary['n_topk_boundary_tie_excluded_rows']:,}行；至少一个k具有可比较共同端点的候选为{summary['n_topk_endpoint_comparable_candidates']}/{summary['candidate_rows']}，去重图对{summary['n_topk_endpoint_comparable_graph_pairs']}/{summary['unique_graph_pairs']}。", '', f"| 判据 | 候选/{summary['candidate_rows']} | 去重图对/{summary['unique_graph_pairs']} | 数据集/反应类 |", '|---|---:|---:|---:|']
    labels = {'any_endpoint_penalty_difference_ge5': '端点损失差≥5pp', 'any_endpoint_penalty_difference_ge10': '端点损失差≥10pp', 'any_rank_penalty_difference_ge10': '秩端点损失差≥0.10', 'any_rank_penalty_difference_ge20': '秩端点损失差≥0.20', 'any_fixed_topk_connectivity_switch': '等k时相同共同端点连接改变', 'any_fixed_topk_same_group_number_connectivity_switch': '等k且连通组数相同，连接对象改变', 'any_common_high_pair_equal_count_topk_switch': '两侧≥20%的共同端点在等k时连接改变', 'any_common_high_pair_equal_count_same_group_topk_switch': '两侧≥20%的共同端点：等k且等组数仍改变连接'}
    for row in aggregate[aggregate.panel.eq('all_769')].to_dict('records'):
        lines.append(f"| {labels[row['metric']]} | {row['n_event_candidates']} | {row['n_event_graph_pairs']} | {row['n_event_datasets']}/{row['n_event_reaction_classes']} |")
    lines += ['', f"原始瓶颈差≥10pp且跨20%阈值的事件中，{summary['n_original_switch_gap10_same_group_count']}个候选两侧高产连通组总数相同，{summary['n_original_switch_gap10_same_high_node_count']}个候选两侧≥20%节点数量相同。等组数与等节点数是不同条件，分别保存。", '', '这里的端点损失差仍随产率整体比例缩放而改变；排序分析与固定top-k承担排除这种解释的责任。固定top-k按全部k保留连续范围，不把挑出的k、端点、矩形或共享分子当作独立实验，也不据其计算显著性。', '', '## 具体化学', '']
    for case in selected.to_dict('records'):
        lines += [f"### {case['rxn_class']}：{case['molecule_name_a']} → {case['molecule_name_b']}", '', f"固定伙伴：{case['partner_name']}，`{case['partner_smiles']}`；变化：`{case['structural_change_a_to_b']}`。候选`{case['candidate_id']}`，图对`{case['graph_pair_id']}`。", '', f"共同配方{int(case['n_common_measured_recipes'])}套。相同端点A产率{100 * case['endpoint0_yield_a']:.2f}%/{100 * case['endpoint1_yield_a']:.2f}%，B为{100 * case['endpoint0_yield_b']:.2f}%/{100 * case['endpoint1_yield_b']:.2f}%；最佳瓶颈A/B={100 * case['best_bottleneck_a']:.2f}%/{100 * case['best_bottleneck_b']:.2f}%。端点损失A/B={100 * case['endpoint_penalty_a']:.2f}/{100 * case['endpoint_penalty_b']:.2f}pp，损失差{100 * case['endpoint_penalty_absolute_difference']:.2f}pp。", '', f"同端点秩瓶颈A/B={case['rank_bottleneck_a']:.3f}/{case['rank_bottleneck_b']:.3f}；秩端点损失A/B={case['rank_endpoint_penalty_a']:.3f}/{case['rank_endpoint_penalty_b']:.3f}，差{case['rank_penalty_absolute_difference']:.3f}。"]
        if case['n_equal_count_topk_switches']:
            lines += ['', f"同一端点还在{k_int(case['example_topk_k'])}个最高产节点的等数量比较中发生连接改变；两侧连通组数={k_int(case['example_topk_groups_a'])}/{k_int(case['example_topk_groups_b'])}，入选产率下界={100 * case['example_topk_cutoff_a']:.2f}%/{100 * case['example_topk_cutoff_b']:.2f}%。"]
        else:
            lines += ['', '该展示端点没有可用的等数量top-k连接翻转；图对的其他端点及全部k仍保留于全量表。']
        for side in ['a', 'b']:
            steps = paths[paths.case_id.eq(case['case_id']) & paths.path_optimized_for.eq(side)].sort_values('path_step')
            lines += ['', f'对{side.upper()}求得的最佳路径（A/B逐点产率）：' + ' → '.join((f'{100 * r.yield_a:.2f}/{100 * r.yield_b:.2f}%' for r in steps.itertuples())) + '。', '起点完整条件：' + steps.iloc[0].full_recipe_text + '。']
            lines += [f'- {r.changed_slot}：{r.previous_component} → {r.new_component}。' for r in steps.iloc[1:].itertuples()]
        lines.append('')
    lines += ['## 文件与复算', '', f"all_candidate_organization_summary.csv.gz完整保存{summary['candidate_rows']}行及原分子/采集信息；unique_graph_pair_summary.csv保留{summary['unique_graph_pairs']}个图对代表（按candidate_id排序取首行，正负方向按该行A/B解释）；all_common_high_endpoint_penalties.csv.gz保存全部共同≥20%端点对及产率、秩瓶颈、损失差和等k支持；all_fixed_topk_comparisons.csv.gz保存全部候选×k、并列排除与端点分母；all_common_recipe_yields.csv.gz保存双方所有共同节点产率、秩分位和完整配方。", '', 'selected_chemical_examples.csv与selected_path_steps.csv给出具名分子及同端点的两侧最佳路径。all_common_recipe_original_records.csv.gz覆盖全部共同节点原始链接；selected_original_records.csv.gz为展示候选的全部共同节点来源。来源包含record_id、source_file/sheet/row、绝对路径与严格采集身份；未改变原始数据。', '', '复算入口：code/round6_path_organization.py。脚本先检验正比例/指数变换下秩不变、并列名次处理及等节点数/等连通组数仍改写端点连接的人工图，再重建全部共同图；所有源行均值逐项与节点表核对。', '']
    (OUT / 'PATH_ORGANIZATION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(molecular.packed(summary), flush=True)
    return summary

def k_int(value):
    return str(int(float(value)))

def refresh_examples():
    previous = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    started = time.time() - previous['elapsed_seconds']
    _, _, graphs, _, measured, _ = molecular.prepare()
    observations, pairs, topk, aggregate, coverage, links = [molecular.read('round6/path_organization/' + name) for name in ['all_candidate_organization_summary.csv.gz', 'all_common_high_endpoint_penalties.csv.gz', 'all_fixed_topk_comparisons.csv.gz', 'aggregate_results.csv', 'dataset_coverage.csv', 'all_common_recipe_original_records.csv.gz']]
    for column in [c for c in pairs if c.startswith('example_topk_') and 'connected' not in c]:
        pairs[column] = pd.to_numeric(pairs[column], errors='coerce')
    cache = {}
    for row in observations.to_dict('records'):
        recipes, ya, yb, adjacent, keys = reconstruct(row, graphs, measured)
        cache[row['candidate_id']] = (recipes, ya, yb, rank_quantiles(ya), rank_quantiles(yb), adjacent, keys)
    selected, paths = select_examples(observations, pairs, topk, cache, graphs)
    selected_links = links[links.candidate_id.isin(selected.candidate_id)].merge(selected[['candidate_id', 'case_id']], on='candidate_id', validate='many_to_one')
    save(selected_links, 'selected_original_records.csv.gz')
    write_report(observations, pairs, topk, aggregate, coverage, selected, paths, links, started)
    print('PATH_ORGANIZATION_EXAMPLES_REFRESHED', flush=True)

def main():
    started = time.time()
    self_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, mapping, graphs, norm, measured, _ = molecular.prepare()
    inputs = molecular.read('round6/molecular_connectivity/all_candidate_acquisition_graph_pairs.csv.gz')
    assert len(inputs) == 768 and inputs.candidate_id.nunique() == 768 and (inputs.graph_pair_id.nunique() == 615)
    observations, pairs, topk, nodes, requests, cache = analyze(inputs, graphs, measured)
    assert set(observations.candidate_id) == set(candidates.candidate_id)
    assert not pairs.duplicated(['candidate_id', 'graph_pair_id', 'endpoint_pair_id']).any()
    save(observations, 'all_candidate_organization_summary.csv.gz')
    save(observations.sort_values('candidate_id').drop_duplicates('graph_pair_id'), 'unique_graph_pair_summary.csv')
    save(pairs, 'all_common_high_endpoint_penalties.csv.gz')
    save(topk, 'all_fixed_topk_comparisons.csv.gz')
    save(nodes, 'all_common_recipe_yields.csv.gz')
    aggregate, coverage = summarize(observations, topk, mapping)
    selected, paths = select_examples(observations, pairs, topk, cache, graphs)
    links = original_links(requests, norm, nodes, selected)
    write_report(observations, pairs, topk, aggregate, coverage, selected, paths, links, started)
    reread = molecular.read('round6/path_organization/all_candidate_organization_summary.csv.gz')
    assert len(reread) == 768 and reread.graph_pair_id.nunique() == 615
    print('PATH_ORGANIZATION_VERIFIED', flush=True)
if __name__ == '__main__':
    if '--self-check' in sys.argv:
        self_checks()
        print('PATH_ORGANIZATION_SELF_CHECKS_PASSED', flush=True)
    elif '--refresh-examples' in sys.argv:
        refresh_examples()
    else:
        main()
