"""Complete acquisition-assignment partner-scope route certificates.

Every exact focal A/B key in the full parent quartet registry contributes every
assignment of one paired acquisition context per distinct measured partner.
Support and acquisition assignments are fixed before examining yields. All
subset routes use that same full-group support, including zero-support groups."""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, product
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
BASE_OUT = Path(__file__).resolve().parent
ANALYSIS = BASE_OUT.parents[1]
FRAME = 'all_coherent_triples' if '--triple-frame' in sys.argv else 'full_partner_assignments'
OUT = BASE_OUT / 'triple_scope' if FRAME == 'all_coherent_triples' else BASE_OUT
THRESHOLDS = [0.1, 0.2, 0.3]
TOKENS = ['Reagent' + str(i) + '_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
SOURCES = {'scope': 'round22/molecular_relation_geometry/molecular_scope_quartet_registry.csv.gz', 'blocks': 'round7/joint_reactant_connectivity/all_strict_target_acquisition_blocks.csv.gz', 'node': 'round7/joint_reactant_connectivity/all_strict_recipe_node_means.csv.gz', 'conditions': 'data/condition_slot_registry.csv', 'records': 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz', 'parent_inventory': 'round26/support_fixed_routes/full_parent_dataset_class_inventory.csv'}

def read(name, **kwargs):
    return pd.read_csv(ANALYSIS / SOURCES[name], keep_default_na=False, low_memory=False, float_precision='round_trip', **kwargs)

def pack(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def write_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def widest(y, adjacency):
    cap = np.where(adjacency, np.minimum(y[:, None], y[None, :]), -np.inf)
    np.fill_diagonal(cap, y)
    for k in range(len(y)):
        np.maximum(cap, np.minimum(cap[:, k, None], cap[None, k, :]), out=cap)
    return cap

def shortest(y, adjacency, start, goal, floor):
    queue, previous = ([start], {start: None})
    for current in queue:
        if current == goal:
            break
        for nxt in np.flatnonzero(adjacency[current] & (y >= floor)):
            nxt = int(nxt)
            if nxt not in previous:
                previous[nxt] = current
                queue.append(nxt)
    assert goal in previous
    route, current = ([], goal)
    while current is not None:
        route.append(current)
        current = previous[current]
    return route[::-1]

def prepare():
    scope = read('scope')
    assert len(scope) == 127099 and scope.acquisition_quartet_id.is_unique
    contexts, keys = ({}, {})
    for row in scope.to_dict('records'):
        for role, aa, bb, entries in [('ab', 'a', 'b', [('c', 'ac', 'bc'), ('d', 'ad', 'bd')]), ('cd', 'c', 'd', [('a', 'ac', 'ad'), ('b', 'bc', 'bd')])]:
            if not row['has_exact_edit_' + role]:
                continue
            focal = sorted([row['molecule_' + aa], row['molecule_' + bb]])
            flip = focal[0] != row['molecule_' + aa]
            key = (row['dataset_id'], role, *focal)
            if key not in keys:
                keys[key] = {'dataset_id': key[0], 'rxn_class': row['rxn_class'], 'focal_role': role, 'molecule_a': focal[0], 'molecule_b': focal[1], 'molecular_pair_id': row['molecular_pair_id_' + role], 'edit_id': row['edit_id_' + role], 'exact_graph_method': row['exact_graph_method_' + role], 'source_exact_transformation': row['exact_transformation_' + role], 'source_structural_change': row['structural_change_a_to_b_' + role], 'canonical_focal_order_reversed_from_source': flip, 'contexts': set(), 'parent_quartets': set(), 'parent_grids': set()}
            keys[key]['parent_quartets'].add(row['acquisition_quartet_id'])
            keys[key]['parent_grids'].add(row['grid_id'])
            for pp, xx, yy in entries:
                x, y = (yy, xx) if flip else (xx, yy)
                ck = (*key, row['molecule_' + pp], row['block_id_' + x], row['block_id_' + y])
                if ck not in contexts:
                    contexts[ck] = {'dataset_id': key[0], 'rxn_class': row['rxn_class'], 'focal_role': role, 'molecule_a': focal[0], 'molecule_b': focal[1], 'partner_smiles': row['molecule_' + pp], 'block_a': row['block_id_' + x], 'block_b': row['block_id_' + y], 'parent_quartets': set(), 'parent_grids': set()}
                    for side, original in [('a', x), ('b', y)]:
                        for field in ['archive', 'plate', 'acquisition']:
                            contexts[ck][field + '_' + side] = row[field + '_' + original]
                contexts[ck]['parent_quartets'].add(row['acquisition_quartet_id'])
                contexts[ck]['parent_grids'].add(row['grid_id'])
                keys[key]['contexts'].add(ck)
    context_rows, context_no = ([], {})
    for no, (ck, row) in enumerate(sorted(contexts.items())):
        context_no[ck] = no
        context_rows.append({'context_id': no, **{k: v for k, v in row.items() if not isinstance(v, set)}, 'parent_quartets': pack(sorted(row['parent_quartets'])), 'parent_grids': pack(sorted(row['parent_grids']))})
    groups, focal_rows = ([], [])
    for no, (key, row) in enumerate(sorted(keys.items())):
        partner_contexts = defaultdict(list)
        for ck in sorted(row['contexts']):
            partner_contexts[ck[4]].append(context_no[ck])
        partners = sorted(partner_contexts)
        choices = [partner_contexts[p] for p in partners]
        full_assignment_count = int(np.prod([len(c) for c in choices], dtype=np.int64))
        assignments = list(product(*choices)) if FRAME == 'full_partner_assignments' else [tuple(cs) for positions in combinations(range(len(partners)), 3) for cs in product(*(choices[p] for p in positions))]
        focal_rows.append({'focal_key_id': no, **{k: v for k, v in row.items() if not isinstance(v, set)}, 'n_partners': len(partners), 'n_acquisition_contexts': sum(map(len, choices)), 'n_assignments': len(assignments), 'n_full_assignments': full_assignment_count, 'partner_smiles': pack(partners), 'acquisition_multiplicity_by_partner': pack([len(x) for x in choices]), 'context_ids_by_partner': pack(choices), 'parent_quartets': pack(sorted(row['parent_quartets'])), 'parent_grids': pack(sorted(row['parent_grids']))})
        for assignment_no, assignment in enumerate(assignments):
            groups.append({'group_id': len(groups), 'focal_key_id': no, 'assignment_within_focal_key': assignment_no, 'dataset_id': key[0], 'rxn_class': row['rxn_class'], 'focal_role': key[1], 'molecular_pair_id': row['molecular_pair_id'], 'edit_id': row['edit_id'], 'n_partners': len(assignment), 'context_ids': pack(assignment), 'frame': FRAME, 'n_assignments_for_focal_key': len(assignments)})
    assert len(groups) == (4632 if FRAME == 'full_partner_assignments' else 44334)
    assert len(focal_rows) == 512 and len(context_rows) == 2911
    block = read('blocks').set_index('block_id')
    node = read('node')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in node.groupby('block_id', sort=False)}
    for c in context_rows:
        a, b = (c['block_a'], c['block_b'])
        c['recipes'] = tuple(sorted(set(values[a]) & set(values[b])))
        assert block.loc[a, 'dataset_id'] == block.loc[b, 'dataset_id'] == c['dataset_id']
        for side, bid in [('a', a), ('b', b)]:
            for field in ['archive', 'plate', 'acquisition']:
                assert c[field + '_' + side] == block.loc[bid, field]
        c['n_native_common_recipes'] = len(c['recipes'])
    conditions = read('conditions').drop_duplicates(['dataset_id', 'recipe_id'])
    graphs = {}
    for dataset, part in conditions.groupby('dataset_id', sort=True):
        part = part.sort_values('recipe_id')
        tok = part[TOKENS].to_numpy(str)
        graphs[dataset] = {'recipes': part.recipe_id.tolist(), 'adjacency': (tok[:, None, :] != tok[None, :, :]).sum(-1) == 1, 'lookup': {r: i for i, r in enumerate(part.recipe_id)}}
    supports, support_lookup = ([], {})
    for group in groups:
        ids = json.loads(group['context_ids'])
        recipes = tuple(sorted(set.intersection(*(set(context_rows[c]['recipes']) for c in ids))))
        sk = (group['dataset_id'], recipes)
        if sk not in support_lookup:
            sno = len(supports)
            support_lookup[sk] = sno
            graph = graphs[sk[0]]
            ix = [graph['lookup'][r] for r in recipes]
            adjacency = graph['adjacency'][np.ix_(ix, ix)]
            nc, labels = connected_components(csr_matrix(adjacency), directed=False)
            supports.append({'support_id': sno, 'dataset_id': sk[0], 'recipes': recipes, 'adjacency': adjacency, 'labels': labels, 'n_nodes': len(recipes), 'n_edges': int(adjacency.sum() // 2), 'n_components': int(nc)})
        su = supports[support_lookup[sk]]
        group.update(support_id=su['support_id'], n_nodes=su['n_nodes'], n_edges=su['n_edges'], n_components=su['n_components'], n_all_measured_endpoint_pairs=su['n_nodes'] * (su['n_nodes'] - 1) // 2)
    save(pd.DataFrame(focal_rows), 'complete_focal_key_registry.csv.gz')
    save(pd.DataFrame([{k: v for k, v in c.items() if k != 'recipes'} | {'native_common_recipe_ids': pack(c['recipes'])} for c in context_rows]), 'complete_context_registry.csv.gz')
    save(pd.DataFrame(groups), 'complete_group_registry_before_outcomes.csv.gz')
    save(pd.DataFrame([{k: v for k, v in su.items() if k not in {'adjacency', 'labels', 'recipes'}} | {'recipe_ids': pack(su['recipes'])} for su in supports]), 'complete_support_registry.csv.gz')
    save(pd.DataFrame([{'support_id': su['support_id'], 'node': i, 'recipe_id': r, 'full_graph_component': int(su['labels'][i])} for su in supports for i, r in enumerate(su['recipes'])]), 'full_support_nodes.csv.gz')
    save(pd.DataFrame([{'support_id': su['support_id'], 'node0': int(i), 'node1': int(j), 'recipe0': su['recipes'][i], 'recipe1': su['recipes'][j]} for su in supports for i, j in zip(*np.where(np.triu(su['adjacency'], 1)))]), 'full_support_edges.csv.gz')
    used_blocks = {c[b] for c in context_rows for b in ['block_a', 'block_b']}
    save(block.loc[sorted(used_blocks)].reset_index(), 'full_source_block_registry.csv.gz')
    save(node[node.block_id.isin(used_blocks)], 'full_source_block_node_yields.csv.gz')
    save(conditions, 'full_condition_registry.csv.gz')
    census = {'frame': FRAME, 'n_focal_keys': len(focal_rows), 'n_contexts': len(context_rows), 'n_groups': len(groups), 'n_groups_ge2': sum((g['n_partners'] >= 2 for g in groups)), 'n_groups_ge3': sum((g['n_partners'] >= 3 for g in groups)), 'max_partners': max((g['n_partners'] for g in groups)), 'n_supports': len(supports), 'n_groups_zero_support': sum((g['n_nodes'] == 0 for g in groups)), 'n_groups_ge2_nodes': sum((g['n_nodes'] >= 2 for g in groups)), 'n_groups_ge3partners_ge2nodes': sum((g['n_partners'] >= 3 and g['n_nodes'] >= 2 for g in groups)), 'group_partner_count_histogram': dict(sorted(Counter((g['n_partners'] for g in groups)).items())), 'group_node_count_histogram': dict(sorted(Counter((g['n_nodes'] for g in groups)).items()))}
    write_json(census, 'design_support_census.json')
    print('DESIGN ' + pack(census), flush=True)
    return (focal_rows, context_rows, groups, supports, values, conditions)

def analyze(focal, contexts, groups, supports, values, conditions):

    @lru_cache(maxsize=4096)
    def context_y(context_id, support_id):
        c, su = (contexts[context_id], supports[support_id])
        return np.array([[values[c['block_' + side]][r] for r in su['recipes']] for side in ['a', 'b']])

    @lru_cache(maxsize=8192)
    def capacity(support_id, ids, projection='ab'):
        y = np.array([context_y(c, support_id) for c in ids])
        if projection == 'a':
            w = y[:, 0].min(0)
        elif projection == 'b':
            w = y[:, 1].min(0)
        else:
            w = y.min((0, 1))
        return widest(w, supports[support_id]['adjacency'])
    summaries, best = ([], None)
    endpoint_rows, pair_rows, deletion_rows = (0, 0, 0)
    output_columns = ['group_id', 'node0', 'node1', 'recipe0', 'recipe1', 'minimum_endpoint_yield', 'full_graph_path', 'K_context_min', 'K_pair_min', 'K_deletion_min', 'K_deletion_max', 'K_a_all_partners', 'K_b_all_partners', 'K_all', 'pair_vs_all_gap', 'two_focal_family_vs_all_gap', 'n_pair_subsets', 'n_deletion_subsets']
    with gzip.open(OUT / 'complete_group_endpoint_capacities.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as stream, gzip.open(OUT / 'complete_pair_subset_endpoint_capacities.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as pstream, gzip.open(OUT / 'complete_deletion_subset_endpoint_capacities.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as dstream:
        stream.write(','.join(output_columns) + '\n')
        pstream.write('group_id,partner_position0,partner_position1,context_id0,context_id1,node0,node1,K_pair\n')
        dstream.write('group_id,deleted_partner_position,deleted_context_id,node0,node1,K_deletion\n')
        for no, group in enumerate(groups):
            ids, sid = (tuple(json.loads(group['context_ids'])), group['support_id'])
            su, m = (supports[sid], len(ids))
            yy = np.array([context_y(c, sid) for c in ids])
            w = yy.min((0, 1))
            high10 = np.flatnonzero(w >= 0.1)
            i, j = np.triu_indices(len(high10), 1)
            i, j = (high10[i], high10[j])
            epmin = np.minimum(w[i], w[j])
            path = su['labels'][i] == su['labels'][j]
            pairs = list(combinations(range(m), 2))
            if len(i):
                full = capacity(sid, ids)[i, j]
                ascope = capacity(sid, ids, 'a')[i, j]
                bscope = capacity(sid, ids, 'b')[i, j]
                single = np.array([capacity(sid, (c,))[i, j] for c in ids])
                pair = np.array([capacity(sid, (ids[a], ids[b]))[i, j] for a, b in pairs])
                delete = np.array([capacity(sid, ids[:a] + ids[a + 1:])[i, j] for a in range(m)])
                min_single, min_pair = (single.min(0), pair.min(0))
                min_delete, max_delete = (delete.min(0), delete.max(0))
                assert np.all(full <= min_delete) and np.all(min_delete <= min_single)
                assert np.all(full <= min_pair) and np.all(full <= np.minimum(ascope, bscope))
                finite = path
                gap, abgap = (np.full(len(i), np.nan), np.full(len(i), np.nan))
                gap[finite] = min_pair[finite] - full[finite]
                abgap[finite] = np.minimum(ascope[finite], bscope[finite]) - full[finite]
                data = {'group_id': no, 'node0': i, 'node1': j, 'recipe0': [su['recipes'][k] for k in i], 'recipe1': [su['recipes'][k] for k in j], 'minimum_endpoint_yield': epmin, 'full_graph_path': path.astype(int), 'K_context_min': min_single, 'K_pair_min': min_pair, 'K_deletion_min': min_delete, 'K_deletion_max': max_delete, 'K_a_all_partners': ascope, 'K_b_all_partners': bscope, 'K_all': full, 'pair_vs_all_gap': gap, 'two_focal_family_vs_all_gap': abgap, 'n_pair_subsets': len(pairs), 'n_deletion_subsets': m}
                pd.DataFrame(data, columns=output_columns).to_csv(stream, index=False, header=False, float_format='%.17g')
                for z, (a, b) in enumerate(pairs):
                    pd.DataFrame({'group_id': no, 'partner_position0': a, 'partner_position1': b, 'context_id0': ids[a], 'context_id1': ids[b], 'node0': i, 'node1': j, 'K_pair': pair[z]}).to_csv(pstream, index=False, header=False, float_format='%.17g')
                for z, c in enumerate(ids):
                    pd.DataFrame({'group_id': no, 'deleted_partner_position': z, 'deleted_context_id': c, 'node0': i, 'node1': j, 'K_deletion': delete[z]}).to_csv(dstream, index=False, header=False, float_format='%.17g')
                endpoint_rows += len(i)
                pair_rows += len(pairs) * len(i)
                deletion_rows += m * len(i)
            else:
                full = ascope = bscope = min_single = min_pair = min_delete = max_delete = np.empty(0)
                gap = abgap = np.empty(0)
            group['n_endpoint_pool10'] = len(i)
            group['n_connected_endpoint_pool10'] = int(path.sum())
            for threshold in THRESHOLDS:
                tag = str(round(threshold * 100))
                eligible = epmin >= threshold
                connected = eligible & path
                cok = min_single >= threshold
                pok = min_pair >= threshold
                dok = min_delete >= threshold
                aok, bok, allok = (ascope >= threshold, bscope >= threshold, full >= threshold)
                event = connected & pok & ~allok & (m >= 3)
                strong = event & aok & bok
                minimal = connected & dok & ~allok & (m >= 3)
                group['n_high_endpoint_pairs_' + tag] = int(eligible.sum())
                group['n_connected_high_endpoint_pairs_' + tag] = int(connected.sum())
                group['support_status_' + tag] = 'zero_common_measured_recipes' if not len(su['recipes']) else 'one_common_measured_recipe' if len(su['recipes']) == 1 else 'no_two_all_target_high_recipes' if not eligible.any() else 'all_high_endpoints_disconnected' if not connected.any() else 'has_connected_all_target_high_endpoints'
                summaries.append({'group_id': no, 'threshold': threshold, 'n_endpoint_pairs': int(eligible.sum()), 'n_connected_endpoint_pairs': int(connected.sum()), 'n_all_context_routes_feasible': int((connected & cok).sum()), 'n_all_pair_routes_feasible': int((connected & pok).sum()), 'n_all_deletion_routes_feasible': int((connected & dok).sum()), 'n_a_all_partner_route_feasible': int((connected & aok).sum()), 'n_b_all_partner_route_feasible': int((connected & bok).sum()), 'n_both_focal_family_routes_feasible': int((connected & aok & bok).sum()), 'n_full_scope_route_feasible': int((connected & allok).sum()), 'n_pairwise_but_not_full_certificate': int(event.sum()), 'n_strong_pairwise_both_focal_but_not_full_certificate': int(strong.sum()), 'n_all_deletions_but_not_full_certificate': int(minimal.sum()), 'n_strong_all_deletions_but_not_full_certificate': int((minimal & aok & bok).sum()), 'sum_pair_vs_all_gap': float(gap[connected].sum()), 'max_pair_vs_all_gap': float(gap[connected].max()) if connected.any() else 0.0, 'sum_two_focal_family_vs_all_gap': float(abgap[connected].sum()), 'sum_all_scope_bottleneck': float(full[connected].sum())})
                for state in product([False, True], repeat=5):
                    mask = connected.copy()
                    for flag, value in zip([pok, dok, aok, bok, allok], state):
                        mask &= flag == value
                    if mask.any():
                        cross_rows.append({'group_id': no, 'threshold': threshold, 'all_pairs_feasible': state[0], 'all_deletions_feasible': state[1], 'a_family_feasible': state[2], 'b_family_feasible': state[3], 'full_scope_feasible': state[4], 'n_endpoint_views': int(mask.sum())})
                if threshold == 0.2:
                    for k in np.flatnonzero(event):
                        clearance = min(min_pair[k] - 0.2, 0.2 - full[k])
                        strong_clearance = min(clearance, ascope[k] - 0.2, bscope[k] - 0.2)
                        rank = (bool(strong[k]), bool(minimal[k]), strong_clearance if strong[k] else clearance, float(gap[k]), -m, -len(su['recipes']), -no, -int(i[k]), -int(j[k]))
                        if best is None or rank > best[0]:
                            best = (rank, no, int(i[k]), int(j[k]))
            if (no + 1) % 1000 == 0:
                print('PROGRESS ' + pack({'groups': no + 1, 'endpoint_rows': endpoint_rows, 'pair_rows': pair_rows, 'closure_cache': str(capacity.cache_info())}), flush=True)
    summary = pd.DataFrame(summaries)
    save(pd.DataFrame(groups), 'complete_group_registry.csv.gz')
    save(summary, 'complete_group_threshold_summary.csv.gz')
    save(pd.DataFrame(cross_rows), 'complete_group_threshold_crossclassification.csv.gz')
    result = {'frame': FRAME, 'n_group_endpoint_rows_pool10': endpoint_rows, 'n_pair_subset_endpoint_rows': pair_rows, 'n_deletion_subset_endpoint_rows': deletion_rows, 'witness_group_id': None if best is None else best[1]}
    write_json(result, 'run_summary.json')
    aggregate(focal, contexts, groups, summary)
    witness(best, focal, contexts, groups, supports, values, conditions, capacity, context_y)
    return result
cross_rows = []

def aggregate(focal, contexts, groups, summary):
    inv = pd.DataFrame(groups)
    merged = summary.merge(inv, on='group_id', validate='many_to_one')
    metrics = [c for c in summary if c.startswith(('n_', 'sum_', 'max_'))]
    rows, coverage = ([], [])
    for stratum, pool in [('all_groups_ge2', merged), ('groups_ge3', merged[merged.n_partners >= 3])]:
        for dim in [None, 'dataset_id', 'rxn_class', 'n_partners']:
            for (threshold, level), part in (pool.assign(level='all') if dim is None else pool.assign(level=pool[dim])).groupby(['threshold', 'level'], sort=True):
                row = {'stratum': stratum, 'dimension': dim or 'all', 'level': level, 'threshold': threshold, 'n_groups': len(part), 'n_focal_keys': part.focal_key_id.nunique(), 'n_exact_molecular_pairs': part.molecular_pair_id.nunique()}
                row.update({c: part[c].max() if c.startswith('max_') else part[c].sum() for c in metrics})
                rows.append(row)
        for threshold in THRESHOLDS:
            part = pool[pool.threshold == threshold]
            for state, selected in [('all_opportunities', part), ('has_high_endpoint_pair', part[part.n_endpoint_pairs > 0]), ('has_connected_high_endpoint_pair', part[part.n_connected_endpoint_pairs > 0]), ('pairwise_but_not_full_certificate', part[part.n_pairwise_but_not_full_certificate > 0]), ('strong_pairwise_both_focal_but_not_full_certificate', part[part.n_strong_pairwise_both_focal_but_not_full_certificate > 0]), ('all_deletions_but_not_full_certificate', part[part.n_all_deletions_but_not_full_certificate > 0])]:
                for dim in [None, 'dataset_id', 'rxn_class']:
                    for level, pp in [('all', selected)] if dim is None else selected.groupby(dim):
                        used = {c for t in pp.context_ids for c in json.loads(t)}
                        blocks = {contexts[c][b] for c in used for b in ['block_a', 'block_b']}
                        qs = {q for c in used for q in json.loads(contexts[c]['parent_quartets'])}
                        coverage.append({'stratum': stratum, 'threshold': threshold, 'state': state, 'dimension': dim or 'all', 'level': level, 'n_groups': len(pp), 'n_focal_keys': pp.focal_key_id.nunique(), 'n_exact_molecular_pairs': pp.molecular_pair_id.nunique(), 'n_contexts': len(used), 'n_source_blocks': len(blocks), 'n_parent_quartets_reused': len(qs), 'n_classes': pp.rxn_class.nunique()})
    save(pd.DataFrame(rows), 'summary.csv')
    save(pd.DataFrame(coverage), 'coverage_and_reuse.csv')
    parent = read('parent_inventory')
    assert len(parent) == 20 and parent.rxn_class.nunique() == 18
    for column, counts in [('n_partner_scope_focal_keys', pd.DataFrame(focal).groupby('dataset_id').size()), ('n_partner_scope_groups_ge2', inv.groupby('dataset_id').size()), ('n_partner_scope_groups_ge3', inv[inv.n_partners >= 3].groupby('dataset_id').size())]:
        parent[column] = parent.dataset_id.map(counts).fillna(0).astype(int)
    primary = merged[(merged.threshold == 0.2) & (merged.n_partners >= 3)]
    for field in ['n_endpoint_pairs', 'n_connected_endpoint_pairs', 'n_all_pair_routes_feasible', 'n_pairwise_but_not_full_certificate', 'n_strong_pairwise_both_focal_but_not_full_certificate', 'n_all_deletions_but_not_full_certificate']:
        parent['partner_scope20_' + field] = parent.dataset_id.map(primary.groupby('dataset_id')[field].sum()).fillna(0).astype(int)
    save(parent, 'complete_20_dataset_18_class_support.csv')
    cross = pd.DataFrame(cross_rows).merge(inv[['group_id', 'dataset_id', 'rxn_class', 'n_partners']], on='group_id')
    columns = ['threshold', 'all_pairs_feasible', 'all_deletions_feasible', 'a_family_feasible', 'b_family_feasible', 'full_scope_feasible']
    complete = []
    for threshold in THRESHOLDS:
        for state in product([False, True], repeat=5):
            mask = (cross.threshold == threshold) & (cross.n_partners >= 3)
            for c, v in zip(columns[1:], state):
                mask &= cross[c] == v
            pp = cross[mask]
            complete.append(dict(zip(columns, [threshold, *state])) | {'n_endpoint_views': int(pp.n_endpoint_views.sum()), 'n_groups': pp.group_id.nunique(), 'n_focal_keys': inv[inv.group_id.isin(pp.group_id)].focal_key_id.nunique()})
    save(pd.DataFrame(complete), 'complete_threshold_crossclassification.csv')

def witness(best, focal, contexts, groups, supports, values, conditions, capacity, context_y):
    if best is None:
        write_json({'status': 'no_pairwise_feasible_full_scope_failed_certificate20'}, 'chemical_witness.json')
        return
    rank, no, start, goal = best
    g = groups[no]
    su = supports[g['support_id']]
    ids = tuple(json.loads(g['context_ids']))
    yy = np.array([context_y(c, g['support_id']) for c in ids])
    w = yy.min((0, 1))
    pair_info, deletion_info, routes, nodes = ([], [], [], [])
    for p, c in enumerate(ids):
        cc = contexts[c]
        for k, recipe in enumerate(su['recipes']):
            nodes.append({'group_id': no, 'partner_position': p, 'context_id': c, 'partner_smiles': cc['partner_smiles'], 'node': k, 'recipe_id': recipe, 'block_a': cc['block_a'], 'block_b': cc['block_b'], 'yield_a': yy[p, 0, k], 'yield_b': yy[p, 1, k], 'full_scope_node_yield': w[k], 'endpoint': k in [start, goal]})
    objectives = [('full_scope', ids, 'ab'), ('a_all_partners', ids, 'a'), ('b_all_partners', ids, 'b')]
    for a, b in combinations(range(len(ids)), 2):
        k = capacity(g['support_id'], (ids[a], ids[b]))[start, goal]
        pair_info.append({'group_id': no, 'partner_position0': a, 'partner_position1': b, 'context_id0': ids[a], 'context_id1': ids[b], 'K_pair': k})
        objectives.append((f'pair_{a}_{b}', (ids[a], ids[b]), 'ab'))
    for a in range(len(ids)):
        sub = ids[:a] + ids[a + 1:]
        k = capacity(g['support_id'], sub)[start, goal]
        deletion_info.append({'group_id': no, 'deleted_partner_position': a, 'deleted_context_id': ids[a], 'K_deletion': k})
        objectives.append((f'delete_{a}', sub, 'ab'))
    for name, subset, projection in objectives:
        ys = np.array([context_y(c, g['support_id']) for c in subset])
        obj = ys[:, 0].min(0) if projection == 'a' else ys[:, 1].min(0) if projection == 'b' else ys.min((0, 1))
        floor = capacity(g['support_id'], subset, projection)[start, goal]
        for step, k in enumerate(shortest(obj, su['adjacency'], start, goal, floor)):
            routes.append({'objective': name, 'path_step': step, 'node': k, 'recipe_id': su['recipes'][k], 'optimized_bottleneck': floor, 'objective_node_yield': obj[k], 'full_scope_node_yield': w[k]})
    reachable = {start}
    queue = [start]
    for k in queue:
        for nxt in np.flatnonzero(su['adjacency'][k] & (w >= 0.2)):
            if int(nxt) not in reachable:
                reachable.add(int(nxt))
                queue.append(int(nxt))
    assert goal not in reachable
    boundary = set(np.flatnonzero(su['adjacency'][list(reachable)].any(0))) - reachable
    cut = []
    for k in sorted(boundary):
        failed = [{'partner_position': int(p), 'focal': 'a' if a == 0 else 'b', 'yield': float(yy[p, a, k])} for p, a in zip(*np.where(yy[:, :, k] < 0.2))]
        assert failed
        cut.append({'node': k, 'recipe_id': su['recipes'][k], 'full_scope_node_yield': w[k], 'failed_targets': pack(failed)})
    full_k = capacity(g['support_id'], ids)[start, goal]
    parent_metadata = focal[g['focal_key_id']]
    identity_fields = ['molecule_a', 'molecule_b', 'exact_graph_method', 'source_exact_transformation', 'source_structural_change', 'canonical_focal_order_reversed_from_source']
    result = {**g, **{k: parent_metadata[k] for k in identity_fields}, **{'parent_' + k: parent_metadata[k] for k in ['n_partners', 'n_acquisition_contexts', 'n_assignments', 'n_full_assignments', 'partner_smiles', 'acquisition_multiplicity_by_partner', 'context_ids_by_partner']}, 'focal_key_parent_quartets': parent_metadata['parent_quartets'], 'focal_key_parent_grids': parent_metadata['parent_grids'], 'n_acquisition_contexts': len(ids), 'partner_smiles': pack([contexts[c]['partner_smiles'] for c in ids]), 'selection': 'Within 20% certificates: strongest both-focal-family flag, all-deletions flag, then largest two-sided clearance, pair-vs-all gap, fewer partners/nodes, smallest IDs.', 'selection_is_outcome_based_existence_witness': True, 'endpoint0_node': start, 'endpoint1_node': goal, 'endpoint0_recipe': su['recipes'][start], 'endpoint1_recipe': su['recipes'][goal], 'minimum_endpoint_yield': float(min(w[start], w[goal])), 'K_all': float(full_k), 'K_pair_min': min((float(x['K_pair']) for x in pair_info)), 'K_a_all_partners': float(capacity(g['support_id'], ids, 'a')[start, goal]), 'K_b_all_partners': float(capacity(g['support_id'], ids, 'b')[start, goal]), 'K_deletion_min': min((float(x['K_deletion']) for x in deletion_info)), 'n_reachable20': len(reachable), 'n_boundary20': len(boundary), 'reachable20_nodes': sorted(reachable), 'partners': [contexts[c]['partner_smiles'] for c in ids]}
    write_json(result, 'chemical_witness.json')
    save(pd.DataFrame(nodes), 'chemical_witness_full_node_yields.csv')
    save(pd.DataFrame(pair_info), 'chemical_witness_pair_certificates.csv')
    save(pd.DataFrame(deletion_info), 'chemical_witness_deletion_certificates.csv')
    save(pd.DataFrame(routes), 'chemical_witness_all_optimal_routes.csv')
    save(pd.DataFrame(cut), 'chemical_witness_joint20_obstruction_boundary.csv')
    save(pd.DataFrame([{'node0': int(a), 'node1': int(b), 'recipe0': su['recipes'][a], 'recipe1': su['recipes'][b]} for a, b in zip(*np.where(np.triu(su['adjacency'], 1)))]), 'chemical_witness_full_edges.csv')
    save(pd.DataFrame([contexts[c] for c in ids]).drop(columns='recipes'), 'chemical_witness_acquisition_contexts.csv')
    save(conditions[(conditions.dataset_id == g['dataset_id']) & conditions.recipe_id.isin(su['recipes'])], 'chemical_witness_conditions.csv')
    used_blocks = {contexts[c][b] for c in ids for b in ['block_a', 'block_b']}
    source = read('records')
    selected = source[source.block_id.isin(used_blocks) & source.recipe_id.isin(su['recipes'])]
    save(selected, 'chemical_witness_source_records.csv.gz')
    remean = selected.groupby(['block_id', 'recipe_id']).yield_numeric.mean()
    for bid in used_blocks:
        for recipe in su['recipes']:
            assert abs(remean.loc[bid, recipe] - values[bid][recipe]) < 1e-14
    result['n_source_records'] = len(selected)
    result['n_source_blocks'] = len(used_blocks)
    result['source_record_node_mean_check'] = 'passed'
    write_json(result, 'chemical_witness.json')

def manifest():
    sources = {}
    for name, relative in SOURCES.items():
        path = ANALYSIS / relative
        sources[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
    sources['generator'] = {'path': str(Path(__file__).resolve()), 'sha256': hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest()}
    write_json(sources, 'source_manifest.json')
if __name__ == '__main__':
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = prepare()
    if '--census-only' not in sys.argv:
        result = analyze(*prepared)
        result['runtime_seconds'] = time.time() - started
        write_json(result, 'run_summary.json')
        manifest()
        print('COMPLETE ' + pack(result), flush=True)
