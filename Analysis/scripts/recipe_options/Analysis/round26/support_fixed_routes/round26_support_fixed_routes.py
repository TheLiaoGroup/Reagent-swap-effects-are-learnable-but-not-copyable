"""Exhaustive exact-pair / shared-partner route transport on fixed recipe support.

The opportunity registry is constructed before reading route outcomes. A pair of
quartets shares an exact focal molecular pair and one opposite-side molecule;
the other opposite-side molecule differs. The stricter stratum additionally
shares both acquisition blocks at the anchor. All writes stay beside this file."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3]
SIDES = ['ac', 'bc', 'ad', 'bd']
SLOTS = [*['Reagent' + str(i) for i in range(1, 7)], 'Solvent', 'Temperature', 'Time']
TOKENS = [s + '_token' for s in SLOTS]
EPS = 1e-12
SOURCES = {'scope': 'round22/molecular_relation_geometry/molecular_scope_quartet_registry.csv.gz', 'r9': 'round9/shared_recipe_paths/all_69128_acquisition_quartet_shared_route_summary.csv.gz', 'nodes': 'round7/joint_reactant_connectivity/all_strict_recipe_node_means.csv.gz', 'conditions': 'data/condition_slot_registry.csv', 'source_rows': 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz', 'class_mapping': 'round3/global_scope/dataset_reaction_class_mapping.csv'}

def read(name, **kw):
    return pd.read_csv(ANALYSIS / SOURCES[name], keep_default_na=False, low_memory=False, **kw)

def pack(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def widest(y, adjacency):
    """Floyd max-min closure, including node endpoint capacities."""
    cap = np.where(adjacency, np.minimum(y[:, None], y[None, :]), -np.inf)
    np.fill_diagonal(cap, y)
    for k in range(len(y)):
        np.maximum(cap, np.minimum(cap[:, k, None], cap[None, k, :]), out=cap)
    return cap

def shortest_at_floor(y, adjacency, start, goal, floor):
    queue, previous = ([start], {start: None})
    for current in queue:
        if current == goal:
            break
        for nxt in np.flatnonzero(adjacency[current] & (y >= floor - EPS)):
            nxt = int(nxt)
            if nxt not in previous:
                previous[nxt] = current
                queue.append(nxt)
    assert goal in previous
    answer, current = ([], goal)
    while current is not None:
        answer.append(current)
        current = previous[current]
    return answer[::-1]

def prepare():
    scope, r9 = (read('scope'), read('r9', usecols=['acquisition_quartet_id', 'common_recipe_ids']))
    assert len(scope) == 127099 and scope.acquisition_quartet_id.is_unique
    assert int(scope.n_common_high_endpoint_pairs_with_full_graph_path.sum()) == 7853943
    scope = scope.merge(r9, on='acquisition_quartet_id', validate='one_to_one')
    conditions = read('conditions').drop_duplicates(['dataset_id', 'recipe_id'])
    recipes = sorted(conditions.recipe_id.unique())
    recipe_number = reader_tables.recipe_number_map(recipes)
    save(pd.DataFrame({'recipe_number': [recipe_number[r] for r in recipes], 'recipe_id': recipes}), 'recipe_number_registry.csv')
    graphs = {}
    for dataset, part in conditions.groupby('dataset_id', sort=True):
        part = part.sort_values('recipe_id')
        tok = part[TOKENS].to_numpy(str)
        graphs[dataset] = {'recipes': part.recipe_id.tolist(), 'adjacency': (tok[:, None, :] != tok[None, :, :]).sum(-1) == 1, 'lookup': {r: i for i, r in enumerate(part.recipe_id)}}
    node = read('nodes')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in node.groupby('block_id', sort=False)}
    quartets = scope.to_dict('records')
    for index, row in enumerate(quartets):
        row['quartet_number'] = index
        row['recipes'] = tuple(json.loads(row['common_recipe_ids']))
        row['recipe_set'] = set(row['recipes'])
        row['native_y'] = np.array([[values[row['block_id_' + s]][r] for r in row['recipes']] for s in SIDES])
    save(pd.DataFrame([{k: v for k, v in row.items() if k not in {'recipes', 'recipe_set', 'native_y'}} for row in quartets]), 'complete_parent_quartet_registry.csv.gz')
    groups = defaultdict(list)
    orientations = [('ab', 'a', 'b', [('c', 'd', [0, 1, 2, 3]), ('d', 'c', [2, 3, 0, 1])]), ('cd', 'c', 'd', [('a', 'b', [0, 2, 1, 3]), ('b', 'a', [1, 3, 0, 2])])]
    for row in quartets:
        for role, a, b, anchors in orientations:
            if not row['has_exact_edit_' + role]:
                continue
            for anchor, other, order in anchors:
                focal = sorted([row['molecule_' + a], row['molecule_' + b]])
                if focal[0] != row['molecule_' + a]:
                    order = [order[1], order[0], order[3], order[2]]
                key = (row['dataset_id'], role, *focal, row['molecule_' + anchor])
                groups[key].append((row['quartet_number'], row['molecule_' + other], order))
    comparisons = []
    group_rows = []
    for group_number, (key, members) in enumerate(sorted(groups.items())):
        dataset, role, a, b, anchor = key
        n_pairs = 0
        for left, right in combinations(sorted(members, key=lambda x: (x[1], x[0])), 2):
            if left[1] == right[1]:
                continue
            lq, rq = (quartets[left[0]], quartets[right[0]])
            lb = [lq['block_id_' + SIDES[i]] for i in left[2]]
            rb = [rq['block_id_' + SIDES[i]] for i in right[2]]
            comparisons.append({'comparison_id': len(comparisons), 'anchor_group_number': group_number, 'dataset_id': dataset, 'rxn_class': lq['rxn_class'], 'focal_role': role, 'focal_molecule_a': a, 'focal_molecule_b': b, 'anchor_partner': anchor, 'partner_q': left[1], 'partner_r': right[1], 'quartet_number_a': left[0], 'quartet_number_b': right[0], 'acquisition_quartet_id_a': lq['acquisition_quartet_id'], 'acquisition_quartet_id_b': rq['acquisition_quartet_id'], 'grid_id_a': lq['grid_id'], 'grid_id_b': rq['grid_id'], 'order_a': pack(left[2]), 'order_b': pack(right[2]), 'aligned_blocks_a': pack(lb), 'aligned_blocks_b': pack(rb), 'identical_anchor_blocks': lb[:2] == rb[:2], 'focal_pair_id': lq['molecular_pair_id_' + role], 'edit_id': lq['edit_id_' + role], 'exact_edit_type': lq['structural_change_a_to_b_' + role], 'exact_graph_method': lq['exact_graph_method_' + role], 'exact_transformation': lq['exact_transformation_' + role]})
            n_pairs += 1
        group_rows.append({'anchor_group_number': group_number, 'dataset_id': dataset, 'focal_role': role, 'focal_molecule_a': a, 'focal_molecule_b': b, 'anchor_partner': anchor, 'n_anchor_records': len(members), 'n_distinct_other_partners': len({m[1] for m in members}), 'n_comparisons': n_pairs})
    assert len(comparisons) == 233260
    save(pd.DataFrame(group_rows), 'complete_anchor_group_registry.csv.gz')
    print('DESIGN ' + pack({'quartets': len(quartets), 'anchor_groups': len(groups), 'comparisons': len(comparisons), 'identical_anchor_blocks': sum((r['identical_anchor_blocks'] for r in comparisons))}), flush=True)
    return (scope, quartets, comparisons, graphs, recipe_number, conditions)

def analyze(scope, quartets, comparisons, graphs, recipe_number, conditions):
    support_registry, support_lookup = ([], {})

    def support(dataset, recipes):
        key = (dataset, recipes)
        if key not in support_lookup:
            graph = graphs[dataset]
            ix = [graph['lookup'][r] for r in recipes]
            adj = graph['adjacency'][np.ix_(ix, ix)]
            ng, labels = connected_components(csr_matrix(adj), directed=False)
            no = len(support_registry)
            support_lookup[key] = no
            support_registry.append({'support_number': no, 'dataset_id': dataset, 'recipes': recipes, 'adjacency': adj, 'labels': labels, 'n_nodes': len(recipes), 'n_edges': int(adj.sum() // 2), 'n_components': int(ng)})
        return support_registry[support_lookup[key]]

    @lru_cache(maxsize=768)
    def matrices(quartet_number, support_number):
        q, su = (quartets[quartet_number], support_registry[support_number])
        index = {r: i for i, r in enumerate(q['recipes'])}
        y = q['native_y'][:, [index[r] for r in su['recipes']]]
        caps = np.array([widest(v, su['adjacency']) for v in y])
        joint = widest(y.min(0), su['adjacency'])
        return (y, caps, joint)

    @lru_cache(maxsize=1024)
    def anchor_matrix(quartet_number, support_number, side0, side1):
        y, _, _ = matrices(quartet_number, support_number)
        return widest(y[[side0, side1]].min(0), support_registry[support_number]['adjacency'])
    for q in quartets:
        q['native_support_number'] = support(q['dataset_id'], q['recipes'])['support_number']
    endpoint_columns = ['comparison_id', 'endpoint0_number', 'endpoint1_number', 'minimum_endpoint_yield', 'fixed_full_graph_path', *['fixed_K_a_' + str(k) for k in range(4)], 'fixed_K_joint_a', *['fixed_K_b_' + str(k) for k in range(4)], 'fixed_K_joint_b', 'fixed_K_anchor_a', 'fixed_K_anchor_b', 'native_K_separate_a', 'native_K_joint_a', 'native_K_separate_b', 'native_K_joint_b']
    summary_rows, best, n_endpoint_rows = ([], None, 0)
    summary_header = True
    with gzip.open(OUT / 'complete_endpoint_comparisons.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as stream, gzip.open(OUT / 'complete_comparison_threshold_summary.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as summary_stream:
        stream.write(','.join(endpoint_columns) + '\n')
        for no, row in enumerate(comparisons):
            qa, qb = (quartets[row['quartet_number_a']], quartets[row['quartet_number_b']])
            recipes = tuple(sorted(qa['recipe_set'] & qb['recipe_set']))
            su = support(row['dataset_id'], recipes)
            row.update(fixed_support_number=su['support_number'], native_support_number_a=qa['native_support_number'], native_support_number_b=qb['native_support_number'], n_fixed_nodes=su['n_nodes'], n_fixed_edges=su['n_edges'], n_fixed_components=su['n_components'], n_native_nodes_a=len(qa['recipes']), n_native_nodes_b=len(qb['recipes']), n_all_fixed_measured_endpoint_pairs=len(recipes) * (len(recipes) - 1) // 2)
            order_a, order_b = (json.loads(row['order_a']), json.loads(row['order_b']))
            qa_index, qb_index = ({r: k for k, r in enumerate(qa['recipes'])}, {r: k for k, r in enumerate(qb['recipes'])})
            ya = qa['native_y'][order_a][:, [qa_index[r] for r in recipes]]
            yb = qb['native_y'][order_b][:, [qb_index[r] for r in recipes]]
            allmin = np.minimum(ya.min(0), yb.min(0)) if recipes else np.empty(0)
            high10 = np.flatnonzero(allmin >= 0.1)
            i, j = np.triu_indices(len(high10), 1)
            i, j = (high10[i], high10[j])
            endpoint_min = np.minimum(allmin[i], allmin[j])
            path = su['labels'][i] == su['labels'][j]
            row['n_endpoint_pool10'] = len(i)
            row['n_endpoint_pool10_full_path'] = int(path.sum())
            row['support_status'] = 'no_common_measured_recipes' if not recipes else 'one_common_measured_recipe' if len(recipes) == 1 else 'fewer_than_two_common_high10_recipes' if len(high10) < 2 else 'all_high10_endpoints_disconnected' if not path.any() else 'has_high10_connected_endpoint'
            if len(i):
                _, ca, ja = matrices(row['quartet_number_a'], su['support_number'])
                _, cb, jb = matrices(row['quartet_number_b'], su['support_number'])
                ca, cb = (ca[order_a], cb[order_b])
                _, nca, nja = matrices(row['quartet_number_a'], qa['native_support_number'])
                _, ncb, njb = matrices(row['quartet_number_b'], qb['native_support_number'])
                na = {r: k for k, r in enumerate(qa['recipes'])}
                nb = {r: k for k, r in enumerate(qb['recipes'])}
                ai, aj = (np.array([na[recipes[k]] for k in i]), np.array([na[recipes[k]] for k in j]))
                bi, bj = (np.array([nb[recipes[k]] for k in i]), np.array([nb[recipes[k]] for k in j]))
                ka, kb, joint_a, joint_b = (ca[:, i, j], cb[:, i, j], ja[i, j], jb[i, j])
                native_a, native_b = (nca[:, ai, aj].min(0), ncb[:, bi, bj].min(0))
                native_joint_a, native_joint_b = (nja[ai, aj], njb[bi, bj])
                fixed_a, fixed_b = (ka.min(0), kb.min(0))
                anchor_a = anchor_matrix(row['quartet_number_a'], su['support_number'], *order_a[:2])[i, j]
                anchor_b = anchor_matrix(row['quartet_number_b'], su['support_number'], *order_b[:2])[i, j]
                if row['identical_anchor_blocks']:
                    np.testing.assert_allclose(anchor_a, anchor_b, atol=EPS, rtol=0)
                assert np.all(joint_a <= fixed_a + EPS) and np.all(joint_b <= fixed_b + EPS)
                assert np.all(joint_a <= native_joint_a + EPS) and np.all(joint_b <= native_joint_b + EPS)
                data = {'comparison_id': np.repeat(no, len(i)), 'endpoint0_number': [recipe_number[recipes[k]] for k in i], 'endpoint1_number': [recipe_number[recipes[k]] for k in j], 'minimum_endpoint_yield': endpoint_min, 'fixed_full_graph_path': path.astype(int), **{'fixed_K_a_' + str(k): ka[k] for k in range(4)}, 'fixed_K_joint_a': joint_a, **{'fixed_K_b_' + str(k): kb[k] for k in range(4)}, 'fixed_K_joint_b': joint_b, 'fixed_K_anchor_a': anchor_a, 'fixed_K_anchor_b': anchor_b, 'native_K_separate_a': native_a, 'native_K_joint_a': native_joint_a, 'native_K_separate_b': native_b, 'native_K_joint_b': native_joint_b}
                pd.DataFrame(data, columns=endpoint_columns).to_csv(stream, index=False, header=False, float_format='%.10g')
                n_endpoint_rows += len(i)
            else:
                fixed_a = fixed_b = joint_a = joint_b = native_a = native_b = native_joint_a = native_joint_b = np.empty(0)
            for t in THRESHOLDS:
                eligible = endpoint_min >= t
                primary = endpoint_min >= 0.2
                tag = str(round(t * 100))
                row['n_high_endpoint_pairs_' + tag] = int(eligible.sum())
                row['n_connected_high_endpoint_pairs_' + tag] = int((eligible & path).sum())
                for cohort, selection in [('threshold_eligible', eligible), ('same_primary20_endpoint_pool', primary)]:
                    for mode, sa, sb, jta, jtb in [('fixed', fixed_a, fixed_b, joint_a, joint_b), ('native', native_a, native_b, native_joint_a, native_joint_b)]:
                        connected = selection & np.isfinite(jta) & np.isfinite(jtb)
                        both_ind = connected & (sa >= t) & (sb >= t)
                        aok, bok = (jta >= t, jtb >= t)
                        ae, be = ((sa >= t) & ~aok, (sb >= t) & ~bok)
                        keep = connected
                        da = np.maximum(0.0, sa[keep] - jta[keep])
                        db = np.maximum(0.0, sb[keep] - jtb[keep])
                        deltas = jtb[keep] - jta[keep]
                        summary_rows.append({'comparison_id': no, 'threshold': t, 'cohort': cohort, 'support_mode': mode, 'n_endpoint_pairs': int(selection.sum()), 'n_threshold_eligible': int((selection & eligible).sum()), 'n_connected_pairs': int(connected.sum()), 'n_both_individually_feasible': int(both_ind.sum()), 'n_neither_individually_feasible': int((connected & (sa < t) & (sb < t)).sum()), 'n_one_individually_feasible': int((connected & ((sa >= t) != (sb >= t))).sum()), 'n_joint_feasibility_switch': int((connected & (aok != bok)).sum()), 'n_joint_switch_both_individually_feasible': int((both_ind & (aok != bok)).sum()), 'n_both_joint_feasible': int((connected & aok & bok).sum()), 'n_neither_joint_feasible': int((connected & ~aok & ~bok).sum()), 'n_both_events': int((connected & ae & be).sum()), 'n_one_event': int((connected & (ae != be)).sum()), 'n_neither_event': int((connected & ~ae & ~be).sum()), 'n_nonzero_joint_difference': int((np.abs(deltas) > EPS).sum()), 'n_nonzero_deficit_difference': int((np.abs(db - da) > EPS).sum()), 'sum_abs_joint_difference': float(np.abs(deltas).sum()), 'sum_abs_deficit_difference': float(np.abs(db - da).sum()), 'sum_joint_a': float(jta[keep].sum()), 'sum_joint_b': float(jtb[keep].sum()), 'sum_deficit_a': float(da.sum()), 'sum_deficit_b': float(db.sum()), 'max_abs_joint_difference': float(np.max(np.abs(deltas))) if len(deltas) else 0.0, 'max_abs_deficit_difference': float(np.max(np.abs(db - da))) if len(db) else 0.0})
                if t == 0.2 and row['identical_anchor_blocks']:
                    witness = eligible & path & (fixed_a >= t) & (fixed_b >= t) & ((joint_a >= t) != (joint_b >= t))
                    for k in np.flatnonzero(witness):
                        clearance = min(max(joint_a[k], joint_b[k]) - t, t - min(joint_a[k], joint_b[k]))
                        rank = (clearance, abs(joint_a[k] - joint_b[k]), -len(recipes))
                        if best is None or rank > best[0]:
                            best = (rank, no, int(i[k]), int(j[k]))
            if (no + 1) % 5000 == 0:
                pd.DataFrame(summary_rows).to_csv(summary_stream, index=False, header=summary_header)
                summary_header = False
                summary_rows.clear()
                print(f'PROGRESS {no + 1}/{len(comparisons)} endpoint_rows={n_endpoint_rows} support_graphs={len(support_registry)}', flush=True)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(summary_stream, index=False, header=summary_header)
    inventory = pd.DataFrame(comparisons)
    save(inventory, 'complete_comparison_registry.csv.gz')
    save(pd.DataFrame([{k: v for k, v in s.items() if k not in {'adjacency', 'labels', 'recipes'}} | {'recipe_ids': pack(s['recipes'])} for s in support_registry]), 'complete_support_graph_registry.csv.gz')
    aggregate(inventory, scope)
    witness_export(best, comparisons, quartets, support_registry, matrices, conditions)
    return {'n_comparisons': len(comparisons), 'n_endpoint_rows': n_endpoint_rows, 'n_support_graphs': len(support_registry), 'witness_comparison_id': None if best is None else best[1]}

def aggregate(inventory, scope):
    meta = ['comparison_id', 'dataset_id', 'rxn_class', 'focal_role', 'exact_edit_type', 'identical_anchor_blocks', 'anchor_group_number', 'focal_pair_id', 'edit_id', 'grid_id_a', 'grid_id_b', 'quartet_number_a', 'quartet_number_b']
    dims = ['threshold', 'cohort', 'support_mode']
    aggregate_parts, primary_parts = ([], [])
    for chunk in pd.read_csv(OUT / 'complete_comparison_threshold_summary.csv.gz', chunksize=100000):
        tab = chunk.merge(inventory[meta], on='comparison_id', validate='many_to_one')
        columns = [c for c in chunk if c.startswith(('n_', 'sum_', 'max_'))]
        for stratum, frame in [('all_anchor_identity', tab), ('identical_anchor_blocks', tab[tab.identical_anchor_blocks])]:
            for dimension in [None, 'dataset_id', 'rxn_class', 'focal_role', 'exact_edit_type']:
                keys = dims + ([] if dimension is None else [dimension])
                group = frame.groupby(keys, as_index=False).agg({c: 'max' if c.startswith('max_') else 'sum' for c in columns})
                group['stratum'], group['dimension'] = (stratum, dimension or 'all')
                group['level'] = 'all' if dimension is None else group[dimension]
                if dimension is not None:
                    group.drop(columns=dimension, inplace=True)
                aggregate_parts.append(group)
        primary_parts.append(tab[(tab.threshold == 0.2) & (tab.cohort == 'threshold_eligible')])
    keys = ['stratum', 'dimension', 'level', *dims]
    aggregated = pd.concat(aggregate_parts, ignore_index=True).groupby(keys, as_index=False).agg({c: 'max' if c.startswith('max_') else 'sum' for c in columns})
    denominators = []
    for stratum, frame in [('all_anchor_identity', inventory), ('identical_anchor_blocks', inventory[inventory.identical_anchor_blocks])]:
        for dimension in [None, 'dataset_id', 'rxn_class', 'focal_role', 'exact_edit_type']:
            groups = [('all', frame)] if dimension is None else frame.groupby(dimension)
            for level, part in groups:
                denominators.append({'stratum': stratum, 'dimension': dimension or 'all', 'level': level, 'n_comparisons': len(part), 'n_anchor_groups': part.anchor_group_number.nunique(), 'n_exact_focal_pairs': part.focal_pair_id.nunique(), 'n_edit_types_registered': part.edit_id.nunique(), 'n_unique_quartets': len(set(part.quartet_number_a) | set(part.quartet_number_b)), 'n_unique_molecular_grids': len(set(part.grid_id_a) | set(part.grid_id_b))})
    aggregated = aggregated.merge(pd.DataFrame(denominators), on=['stratum', 'dimension', 'level'], validate='many_to_one')
    save(aggregated, 'summary.csv')
    save(inventory.groupby(['dataset_id', 'rxn_class', 'identical_anchor_blocks', 'support_status'], as_index=False).agg(n_comparisons=('comparison_id', 'size'), n_anchor_groups=('anchor_group_number', 'nunique'), n_endpoint_pool10=('n_endpoint_pool10', 'sum'), n_high_endpoint_pairs_20=('n_high_endpoint_pairs_20', 'sum'), n_connected_high_endpoint_pairs_20=('n_connected_high_endpoint_pairs_20', 'sum')), 'support_attrition_inventory.csv')
    classes = read('class_mapping')[['dataset_id', 'rxn_class']].drop_duplicates()
    parent = classes.merge(scope.groupby(['dataset_id', 'rxn_class'], as_index=False).agg(n_parent_quartets=('acquisition_quartet_id', 'size'), n_parent_grids=('grid_id', 'nunique'), n_parent_endpoints=('n_common_high_endpoint_pairs_with_full_graph_path', 'sum'), n_parent_events=('n_common_high_endpoint_pairs_with_full_graph_path', lambda x: 0)), how='left')
    events = read('r9', usecols=['dataset_id', 'n_pairs_individual20_but_not_joint20']).groupby('dataset_id').n_pairs_individual20_but_not_joint20.sum()
    parent['n_parent_events'] = parent.dataset_id.map(events).fillna(0).astype(int)
    parent['n_parent_non_events'] = parent.n_parent_endpoints - parent.n_parent_events
    counts = inventory.groupby('dataset_id').size()
    parent['n_anchored_comparisons'] = parent.dataset_id.map(counts).fillna(0).astype(int)
    strict = inventory[inventory.identical_anchor_blocks].groupby('dataset_id').size()
    parent['n_identical_anchor_block_comparisons'] = parent.dataset_id.map(strict).fillna(0).astype(int)
    parent.fillna(0, inplace=True)
    assert parent.rxn_class.nunique() == 18 and int(parent.n_parent_endpoints.sum()) == 7853943
    assert int(parent.n_parent_events.sum()) == 68367
    save(parent, 'full_parent_dataset_class_inventory.csv')
    molecule_rows = []
    for role in ['focal_molecule_a', 'focal_molecule_b', 'anchor_partner', 'partner_q', 'partner_r']:
        part = inventory.groupby(['dataset_id', role]).agg(n_comparisons=('comparison_id', 'size'), n_anchor_groups=('anchor_group_number', 'nunique'), n_exact_focal_pairs=('focal_pair_id', 'nunique')).reset_index()
        part.rename(columns={role: 'molecule_smiles'}, inplace=True)
        part['role'] = role
        molecule_rows.append(part)
    save(pd.concat(molecule_rows, ignore_index=True), 'molecule_reuse_inventory.csv.gz')
    reuse = pd.concat([inventory[['comparison_id', c]].rename(columns={c: 'quartet_number'}) for c in ['quartet_number_a', 'quartet_number_b']])
    save(reuse.groupby('quartet_number', as_index=False).agg(n_comparison_uses=('comparison_id', 'size')), 'quartet_reuse_inventory.csv')
    retained = pd.concat(primary_parts).pivot(index='comparison_id', columns='support_mode', values=['n_joint_feasibility_switch', 'n_joint_switch_both_individually_feasible', 'n_one_event', 'n_connected_pairs'])
    retained.columns = [a + '_' + b for a, b in retained.columns]
    retained = retained.reset_index().merge(inventory[meta], on='comparison_id')
    save(retained, 'native_vs_fixed_primary20_comparison.csv.gz')

def witness_export(best, comparisons, quartets, supports, matrices, conditions):
    if best is None:
        (OUT / 'witness.json').write_text(pack({'status': 'no_identical_anchor_block_joint_switch_with_both_individual_feasible'}), encoding='utf-8')
        return
    rank, no, start, goal = best
    row = comparisons[no]
    fixed = supports[row['fixed_support_number']]
    result = dict(row, endpoint0_recipe=fixed['recipes'][start], endpoint1_recipe=fixed['recipes'][goal], clearance20=rank[0], joint_difference=rank[1])
    nodes, edges, paths = ([], [], [])
    for mode in ['fixed', 'native']:
        for context in ['a', 'b']:
            q = quartets[row['quartet_number_' + context]]
            su = fixed if mode == 'fixed' else supports[q['native_support_number']]
            y, caps, joint = matrices(q['quartet_number'], su['support_number'])
            order = json.loads(row['order_' + context])
            y, caps = (y[order], caps[order])
            source_order = [SIDES[i] for i in order]
            s, t = (su['recipes'].index(result['endpoint0_recipe']), su['recipes'].index(result['endpoint1_recipe']))
            result[mode + '_K_joint_' + context] = float(joint[s, t])
            result[mode + '_K_separate_' + context] = float(caps[:, s, t].min())
            for k, recipe in enumerate(su['recipes']):
                nodes.append({'mode': mode, 'context': context, 'node': k, 'recipe_id': recipe, 'in_fixed_support': recipe in fixed['recipes'], 'endpoint': recipe in [result['endpoint0_recipe'], result['endpoint1_recipe']], **{'yield_' + role: y[z, k] for z, role in enumerate(['AP', 'BP', 'AX', 'BX'])}, **{'block_id_' + role: q['block_id_' + side] for role, side in zip(['AP', 'BP', 'AX', 'BX'], source_order)}, 'joint_node_yield': float(y[:, k].min()), 'joint_high20': bool(y[:, k].min() >= 0.2)})
            ii, jj = np.where(np.triu(su['adjacency'], 1))
            edges.extend(({'mode': mode, 'context': context, 'node0': int(a), 'node1': int(b), 'recipe0': su['recipes'][a], 'recipe1': su['recipes'][b]} for a, b in zip(ii, jj)))
            for objective_no, (objective, floor) in enumerate([*zip(y, caps[:, s, t]), (y.min(0), joint[s, t])]):
                path = shortest_at_floor(objective, su['adjacency'], s, t, float(floor))
                for step, k in enumerate(path):
                    paths.append({'mode': mode, 'context': context, 'objective': ['AP', 'BP', 'AX', 'BX', 'joint'][objective_no], 'path_step': step, 'node': k, 'recipe_id': su['recipes'][k], 'optimized_bottleneck': float(floor), **{'yield_' + role: y[z, k] for z, role in enumerate(['AP', 'BP', 'AX', 'BX'])}})
    save(pd.DataFrame(nodes), 'measured_example_full_graph_nodes.csv')
    save(pd.DataFrame(edges), 'measured_example_full_graph_edges.csv')
    save(pd.DataFrame(paths), 'measured_example_all_optimal_routes.csv')
    used_recipes = {n['recipe_id'] for n in nodes}
    save(conditions[(conditions.dataset_id == row['dataset_id']) & conditions.recipe_id.isin(used_recipes)], 'measured_example_condition_registry.csv')
    blocks = set(json.loads(row['aligned_blocks_a']) + json.loads(row['aligned_blocks_b']))
    source = read('source_rows')
    selected = source[source.block_id.isin(blocks) & source.recipe_id.isin(used_recipes)]
    save(selected, 'measured_example_source_records.csv')
    result['n_source_rows'] = len(selected)
    result['n_distinct_acquisition_blocks'] = len(blocks)
    feasible = 'a' if result['fixed_K_joint_a'] >= 0.2 else 'b'
    other = 'b' if feasible == 'a' else 'a'
    path_table = pd.DataFrame(paths)
    route = path_table[(path_table['mode'] == 'fixed') & (path_table.context == feasible) & (path_table.objective == 'joint')]
    node_table = pd.DataFrame(nodes)
    route = route.merge(node_table[(node_table['mode'] == 'fixed') & (node_table.context == other)][['recipe_id', 'yield_AP', 'yield_BP', 'yield_AX', 'yield_BX', 'joint_node_yield']], on='recipe_id', suffixes=('_feasible', '_other'))
    save(route, 'measured_example_same_route_transported_yields.csv')
    result['feasible_context'] = feasible
    result['other_context'] = other
    result['same_route_bottleneck_other_context'] = float(route.joint_node_yield.min())
    (OUT / 'witness.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def coverage_reuse():
    inv = pd.read_csv(OUT / 'complete_comparison_registry.csv.gz', keep_default_na=False)
    rows = []
    for stratum, frame in [('all_anchor_identity', inv), ('identical_anchor_blocks', inv[inv.identical_anchor_blocks])]:
        for t in THRESHOLDS:
            tag = str(round(t * 100))
            for state, eligible in [('all_registered_opportunities', frame), ('has_two_all_target_high_endpoints', frame[frame['n_high_endpoint_pairs_' + tag] > 0]), ('has_connected_all_target_high_endpoints_on_fixed_graph', frame[frame['n_connected_high_endpoint_pairs_' + tag] > 0])]:
                for dim in [None, 'dataset_id', 'rxn_class', 'focal_role', 'exact_edit_type']:
                    parts = [('all', eligible)] if dim is None else eligible.groupby(dim)
                    for level, part in parts:
                        blocks = {b for text in list(part.aligned_blocks_a) + list(part.aligned_blocks_b) for b in json.loads(text)}
                        rows.append({'stratum': stratum, 'threshold': t, 'coverage_state': state, 'dimension': dim or 'all', 'level': level, 'n_comparisons': len(part), 'n_anchor_groups': part.anchor_group_number.nunique(), 'n_exact_focal_pairs': part.focal_pair_id.nunique(), 'n_registered_edit_ids': part.edit_id.nunique(), 'n_quartets': len(set(part.quartet_number_a) | set(part.quartet_number_b)), 'n_molecular_grids': len(set(part.grid_id_a) | set(part.grid_id_b)), 'n_acquisition_blocks': len(blocks), 'n_high_endpoint_pairs': int(part['n_high_endpoint_pairs_' + tag].sum()), 'n_connected_high_endpoint_pairs_fixed': int(part['n_connected_high_endpoint_pairs_' + tag].sum())})
    save(pd.DataFrame(rows), 'reuse_and_endpoint_coverage_by_threshold.csv')
    annotate_witness()

def annotate_witness(quartet_registry=None):
    wpath = OUT / 'witness.json'
    witness = json.loads(wpath.read_text(encoding='utf-8'))
    if 'comparison_id' in witness:
        q = pd.read_csv(quartet_registry or OUT / 'complete_parent_quartet_registry.csv.gz', keep_default_na=False)
        names = {}
        for r in q[q.quartet_number.isin([witness['quartet_number_a'], witness['quartet_number_b']])].to_dict('records'):
            for at in ['a', 'b', 'c', 'd']:
                if r.get('molecule_name_' + at, ''):
                    names[r['molecule_' + at]] = r['molecule_name_' + at]
        for key in ['focal_molecule_a', 'focal_molecule_b', 'anchor_partner', 'partner_q', 'partner_r']:
            witness[key + '_registered_name'] = names.get(witness[key], '')
        nodes = pd.read_csv(OUT / 'measured_example_full_graph_nodes.csv')
        fixed = nodes[nodes['mode'] == 'fixed']
        anchor_y = fixed[fixed.context == 'a'][['yield_AP', 'yield_BP']].to_numpy().min(1)
        edges = pd.read_csv(OUT / 'measured_example_full_graph_edges.csv')
        e = edges[(edges['mode'] == 'fixed') & (edges.context == 'a')]
        adjacency = np.zeros((len(anchor_y), len(anchor_y)), dtype=bool)
        adjacency[e.node0.to_numpy(int), e.node1.to_numpy(int)] = True
        adjacency |= adjacency.T
        fixeda = fixed[fixed.context == 'a'].set_index('recipe_id')
        i = int(fixeda.loc[witness['endpoint0_recipe'], 'node'])
        j = int(fixeda.loc[witness['endpoint1_recipe'], 'node'])
        witness['fixed_K_anchor'] = float(widest(anchor_y, adjacency)[i, j])
        for at in ['a', 'b']:
            witness['fixed_scope_extension_penalty_' + at] = witness['fixed_K_anchor'] - witness['fixed_K_joint_' + at]
            witness['joint_support_penalty_' + at] = witness['native_K_joint_' + at] - witness['fixed_K_joint_' + at]
        other = fixed[fixed.context == witness['other_context']].sort_values('node').copy()
        high = other.joint_high20.to_numpy(bool)
        reachable = np.zeros(len(high), bool)
        reachable[i] = True
        queue = [i]
        for node in queue:
            for nxt in np.flatnonzero(adjacency[node] & high & ~reachable):
                reachable[nxt] = True
                queue.append(int(nxt))
        assert not reachable[j]
        boundary = adjacency[reachable].any(0) & ~reachable
        assert not high[boundary].any()
        other['reachable_from_endpoint0_in_joint_high20_graph'] = reachable
        other['low_boundary_cut_node'] = boundary
        save(other, 'measured_example_joint20_obstruction_components.csv')
        cut = other[boundary].copy()
        for role in ['AP', 'BP', 'AX', 'BX']:
            cut['low20_' + role] = cut['yield_' + role] < 0.2
        save(cut, 'measured_example_joint20_obstruction_cut.csv')
        witness['n_joint20_reachable_nodes_other_context'] = int(reachable.sum())
        witness['n_joint20_low_boundary_cut_nodes_other_context'] = int(boundary.sum())
        wpath.write_text(json.dumps(witness, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def report(run):
    s = pd.read_csv(OUT / 'summary.csv')
    p = pd.read_csv(OUT / 'full_parent_dataset_class_inventory.csv')
    inv = pd.read_csv(OUT / 'complete_comparison_registry.csv.gz')
    primary = s[(s.dimension == 'all') & (s.cohort == 'threshold_eligible') & (s.threshold == 0.2)]
    lines = ['# Round 26: Partner context and common routes after fixing measured-recipe support', '', 'Date: 2026-10-17.', '', '## Question and pre-built scope', '', f'This round compares, for the same registered exact molecule pair A/B with shared partner P, the two quartets in which the other partner changes from Q to R: (A,B)×(P,Q) versus (A,B)×(P,R). Molecule order is realigned by canonical identity. Comparison opportunities are built from the full registry of {int(p.n_parent_quartets.sum()):,} acquisition quartets, without first filtering to the {int(p.n_parent_events.sum()):,} events or to repeated event edits. Cases where the two acquisition blocks sharing P are identical are reported separately as a stronger descriptive layer.', '', f'There are {len(inv):,} comparison opportunities and {inv.anchor_group_number.nunique():,} anchor groups that actually contribute comparisons, of which {int(inv.identical_anchor_blocks.sum()):,} share identical anchor acquisition blocks. The full parent scope keeps {int(p.n_parent_endpoints.sum()):,} endpoints, {int(p.n_parent_events.sum()):,} events, {int(p.n_parent_non_events.sum()):,} non-events and 18 reaction classes. Parent endpoint counts and new paired endpoint counts use different reuse definitions and cannot be subtracted as a filtering sequence.', '', '## What is fixed', '', 'Each comparison takes the intersection of the common measured recipes of the two quartets and builds one induced adjacency graph in which neighbouring recipes differ in exactly one of the nine condition-slot tokens. Both contexts use identical nodes, edges and endpoints; endpoints must reach the given threshold in all eight target measurements (six distinct target blocks when the anchor blocks are identical). Kindividual is the maximum-bottleneck route for each target separately, Kseparate=min(Kindividual); Kjoint is the maximum-bottleneck route on the per-node worst target yield. joint deficit=Kseparate−Kjoint. Kjoint≤Kseparate is a mathematical identity; the empirical result of this round is the change in route feasibility or in continuous quantities between contexts.', '', 'The fixed graph controls measured node/edge support; it does not control acquisition batch, measurement differences, yield scale or unmeasured conditions of Q/R. The identical-anchor-block layer ensures that the AP/BP yields themselves are identical. The result describes measured partner-dependent common applicability, not causal or mechanistic identification.', '', '## 20% main analysis', '', '| Layer | Graph | Comparison opportunities | Endpoints high-yield in both | Full graph connected in both contexts | Joint feasibility differs | Joint differs when each context is individually feasible |', '|---|---|---:|---:|---:|---:|---:|']
    for r in primary.to_dict('records'):
        lines.append(f"| {r['stratum']} | {r['support_mode']} | {r['n_comparisons']:,} | {r['n_endpoint_pairs']:,} | {r['n_connected_pairs']:,} | {r['n_joint_feasibility_switch']:,} | {r['n_joint_switch_both_individually_feasible']:,} |")
    lines += ['', 'Here "each context individually feasible" means that each of the four targets in each context has a ≥20% route; this column separates changes in individual route feasibility from context differences in the common-route requirement. Non-events are further split into joint feasible in both contexts, infeasible in both, individual route infeasible, and events on one side only / both sides; all are saved in the full threshold table.', '', '## Threshold sensitivity and continuous quantities', '', '| Fixed graph, identical anchor blocks | Threshold | High endpoints at threshold | Joint switches | Switches with both individually feasible | Mean abs. continuous joint difference/pp | Mean abs. deficit difference/pp |', '|---|---:|---:|---:|---:|---:|---:|']
    sens = s[(s.stratum == 'identical_anchor_blocks') & (s.dimension == 'all') & (s.cohort == 'threshold_eligible') & (s.support_mode == 'fixed')]
    for r in sens.to_dict('records'):
        n = r['n_connected_pairs']
        lines.append(f"| fixed | {r['threshold']:.0%} | {r['n_endpoint_pairs']:,} | {r['n_joint_feasibility_switch']:,} | {r['n_joint_switch_both_individually_feasible']:,} | {(100 * r['sum_abs_joint_difference'] / n if n else 0):.4f} | {(100 * r['sum_abs_deficit_difference'] / n if n else 0):.4f} |")
    lines += ['', '10/15/20/25/30% are all recomputed on the same opportunity registry. The full endpoint file keeps 10% high endpoints and records the minimum endpoint yield; threshold_eligible uses, for each threshold, only endpoints that all reach that threshold; same_primary20_endpoint_pool fixes the 20% high-endpoint cohort and also reports n_threshold_eligible, so that endpoints no longer high-yield at 25/30% are not misreported as internal route failures.', '', 'Native and fixed comparisons use the same endpoints; the intersection restriction cannot raise the single-target or joint maximum bottleneck, but deficit need not be monotone. Changes on native support can include changes in graph support. The fixed-graph comparison gives both contexts the same nodes/edges, but the intersection restriction itself can create or remove threshold switches relative to the native graph; yields of excluded recipes not measured in the other context are unknown. The two are not independent samples.', '', '## Missing support and reuse', '', '| Support status | Comparisons |', '|---|---:|']
    for state, n in inv.support_status.value_counts().items():
        lines.append(f'| {state} | {n:,} |')
    lines += ['', 'full_parent_dataset_class_inventory.csv keeps all 18 classes and 20 datasets (zero where no quartet was formed); support_attrition_inventory.csv records missing measurements, no common high endpoint and disconnection by dataset/class/anchor-block layer; complete_anchor_group_registry.csv.gz also keeps anchor groups that cannot form different-partner comparisons. The same molecules, cells, quartets, recipe endpoints and raw rows are reused many times; counts are not numbers of independent experiments. summary.csv gives counts of groups/molecule pairs/cells/quartets, and molecule_reuse_inventory and quartet_reuse_inventory keep the reuse distributions.', '']
    coverage_path = OUT / 'reuse_and_endpoint_coverage_by_threshold.csv'
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        coverage = coverage[(coverage.stratum == 'identical_anchor_blocks') & (coverage.threshold == 0.2) & (coverage.dimension == 'all')]
        lines += ['| 20% coverage layer, identical anchor blocks | Comparisons | Anchor groups | Exact molecule pairs | Molecule cells | Quartets | Acquisition blocks |', '|---|---:|---:|---:|---:|---:|---:|']
        for c in coverage.to_dict('records'):
            lines.append(f"| {c['coverage_state']} | {c['n_comparisons']:,} | {c['n_anchor_groups']:,} | {c['n_exact_focal_pairs']:,} | {c['n_molecular_grids']:,} | {c['n_quartets']:,} | {c['n_acquisition_blocks']:,} |")
        lines += ['', 'Full stratified coverage and reuse denominators are saved in reuse_and_endpoint_coverage_by_threshold.csv.', '']
    w = json.loads((OUT / 'witness.json').read_text(encoding='utf-8'))
    if 'comparison_id' in w:
        lines += ['## Measured chemical witness', '', 'Among candidates with identical anchor blocks, individual feasibility on both sides and a joint threshold switch, this example has the largest two-sided threshold clearance; ties are broken by joint bottleneck difference and then by smaller graph. It is an existence witness of robust separation, not a representative sample.', '', f"Comparison {w['comparison_id']}, dataset {w['dataset_id']}; focal pair `{w['focal_molecule_a']}` and `{w['focal_molecule_b']}`, registered change `{w['exact_edit_type']}`; anchor P=`{w['anchor_partner']}`, Q=`{w['partner_q']}`, R=`{w['partner_r']}`.", '', f"Same endpoints `{w['endpoint0_recipe']}` / `{w['endpoint1_recipe']}`; fixed graph with {w['n_fixed_nodes']} nodes and {w['n_fixed_edges']} edges. Kseparate in contexts a/b is {100 * w['fixed_K_separate_a']:.4f}%/{100 * w['fixed_K_separate_b']:.4f}%, Kjoint {100 * w['fixed_K_joint_a']:.4f}%/{100 * w['fixed_K_joint_b']:.4f}%; joint bottleneck difference {100 * w['joint_difference']:.4f}pp; two-sided clearance of the 20% switch {100 * w['clearance20']:.4f}pp.", '', f"On the native graphs, Kseparate in a/b is {100 * w['native_K_separate_a']:.4f}%/{100 * w['native_K_separate_b']:.4f}%, native Kjoint {100 * w['native_K_joint_a']:.4f}%/{100 * w['native_K_joint_b']:.4f}%. " + ('In this example the native joint route is feasible on both sides; the 20% switch is created by the intersection-graph restriction, and the natively feasible side can detour through excluded recipes. It shows only a partner difference on the retained common support and cannot be described as a native failure that persists after support matching.' if min(w['native_K_joint_a'], w['native_K_joint_b']) >= 0.2 else 'In this example the joint threshold switch is present on both the native and the fixed graphs.'), '', f"Moving the optimal joint route of the feasible side node by node to the other context gives an actual worst yield of {100 * w['same_route_bottleneck_other_context']:.4f}%. The optimal joint bottleneck in the other context is also below 20%, so this is not a single poorly chosen route. measured_example_same_route_transported_yields.csv gives all measured yields on the same route; full_graph_nodes/edges keep the full fixed and native graphs; all_optimal_routes keeps the four individual routes and the joint route; condition_registry gives the original condition tokens and names; source_records gives the original file/sheet/row number for each record.", '']
        if 'fixed_K_anchor' in w:
            lines += [f"K of the shared two-target anchor is {100 * w['fixed_K_anchor']:.4f}%; the bottleneck loss from extending to four targets is, for a/b, {100 * w['fixed_scope_extension_penalty_a']:.4f}/{100 * w['fixed_scope_extension_penalty_b']:.4f}pp. With the anchor bottleneck fixed, the joint difference between the two contexts is exactly the negative of the difference in extension loss. The joint loss due to graph-support restriction for a/b is {100 * w['joint_support_penalty_a']:.4f}/{100 * w['joint_support_penalty_b']:.4f}pp.", '']
        if 'n_joint20_low_boundary_cut_nodes_other_context' in w:
            lines += [f"In the other context, the 20% common high-yield induced graph reaches {w['n_joint20_reachable_nodes_other_context']} nodes from endpoint 0 and does not reach endpoint 1; the outer boundary of the reachable region consists of {w['n_joint20_low_boundary_cut_nodes_other_context']} measured low-yield nodes. joint20_obstruction_cut.csv labels the failing targets for each node; this boundary cuts every ≥20% common route. It is not claimed to be a minimum cut or a chemical mechanism.", '']
    lines += ['## Audit scope', '', 'The generator uses a Floyd max-min closure; the independent verifier uses maximum-spanning-forest path bottlenecks. SHA-256 hashes of all source files are stored in source_hashes.json. Unsupported cases are not merged into chemical negatives. This round does not call the number of low nodes on a route a number of components, and does not infer reaction mechanisms from graph changes.', '']
    verification_path = OUT / 'verification.json'
    if verification_path.exists():
        v = json.loads(verification_path.read_text(encoding='utf-8'))
        crossings = sum(v.get('export_rounding_threshold_crossings', {}).values())
        lines += [f"Independent verification status: {v['status']}; row-by-row verification of {v['verified_endpoint_rows']:,} endpoint comparisons, numerical export tolerance {v['continuous_comparison_tolerance']:.1e}; numerical-rounding crossings of the 10/15/20/25/30% threshold classification, total {crossings}. The main summary is checked against values rebuilt at full precision; the selected example is additionally re-read from the original reaction and condition workbooks, with SHA-256 recorded.", '']
    (OUT / 'ROUND26_SUPPORT_FIXED_ROUTE_TRANSPORT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    if '--report-only' in sys.argv:
        coverage_reuse()
        report(json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8')))
        engine = '\n'.join((inspect.getsource(fn) for fn in [prepare, analyze, aggregate, witness_export, widest, shortest_at_floor]))
        manifest = {'stage': 'numerical generation completed; final reporting/coverage stage rerun', 'numerical_engine_source_sha256': hashlib.sha256(engine.encode('utf-8')).hexdigest(), 'generator_sha256': hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest(), 'verifier_sha256': hashlib.file_digest((OUT / 'verify_round26_support_fixed_routes.py').open('rb'), 'sha256').hexdigest(), 'note': 'Only reporting/coverage/manifest functions were added while numerical run was active; numerical functions unchanged since run start. Full generator now reproduces the same numerical artifacts plus final reporting.'}
        (OUT / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return
    start = time.time()
    scope, quartets, comparisons, graphs, numbers, conditions = prepare()
    result = analyze(scope, quartets, comparisons, graphs, numbers, conditions)
    result['elapsed_seconds'] = time.time() - start
    (OUT / 'run_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    hashes = {name: {'path': str(ANALYSIS / path), 'sha256': hashlib.file_digest((ANALYSIS / path).open('rb'), 'sha256').hexdigest()} for name, path in SOURCES.items()}
    (OUT / 'source_hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    coverage_reuse()
    report(result)
    print('DONE ' + pack(result), flush=True)
if __name__ == '__main__':
    main()
