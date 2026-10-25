"""Independent union-find maximum-spanning-forest verification of all exports."""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parent
OUT = BASE / 'triple_scope' if '--triple-frame' in sys.argv else BASE
TOLERANCE = 2e-15

def read(name, **kwargs):
    return pd.read_csv(OUT / name, keep_default_na=False, float_precision='round_trip', **kwargs)

def forest_closure(y, adjacency):
    """Merge descending-capacity components, assigning each first joined pair."""
    n = len(y)
    result = np.full((n, n), -np.inf)
    np.fill_diagonal(result, y)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = [(float(min(y[i], y[j])), int(i), int(j)) for i, j in zip(*np.where(np.triu(adjacency, 1)))]
    for value, i, j in sorted(edges, reverse=True):
        a, b = (find(i), find(j))
        if a == b:
            continue
        aa, bb = (members.pop(a), members.pop(b))
        result[np.ix_(aa, bb)] = result[np.ix_(bb, aa)] = value
        parent[b] = a
        members[a] = aa + bb
    return result

def contiguous_groups(name):
    pending = None
    for chunk in read(name, chunksize=100000):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last = chunk.group_id.iloc[-1]
        ready = chunk[chunk.group_id != last]
        pending = chunk[chunk.group_id == last]
        for no, part in ready.groupby('group_id', sort=False):
            yield (int(no), part)
    if pending is not None:
        yield (int(pending.group_id.iloc[0]), pending)

def main():
    started = time.time()
    group = read('complete_group_registry.csv.gz').to_dict('records')
    before = read('complete_group_registry_before_outcomes.csv.gz')
    after = pd.DataFrame(group)
    pd.testing.assert_frame_equal(before, after[before.columns], check_dtype=False)
    context = read('complete_context_registry.csv.gz').set_index('context_id').to_dict('index')
    focal = read('complete_focal_key_registry.csv.gz').set_index('focal_key_id').to_dict('index')
    node = read('full_source_block_node_yields.csv.gz')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in node.groupby('block_id', sort=False)}
    su = read('complete_support_registry.csv.gz').to_dict('records')
    conditions = read('full_condition_registry.csv.gz')
    tokens = ['Reagent' + str(i) + '_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
    condition_lookup = conditions.set_index(['dataset_id', 'recipe_id'])
    edge_table = read('full_support_edges.csv.gz')
    edge_lookup = {int(s): p for s, p in edge_table.groupby('support_id')}
    node_table = read('full_support_nodes.csv.gz')
    node_lookup = {int(s): p for s, p in node_table.groupby('support_id')}
    for support in su:
        sid = int(support['support_id'])
        support['recipes'] = tuple(json.loads(support['recipe_ids']))
        n = len(support['recipes'])
        support['adjacency'] = np.zeros((n, n), bool)
        if sid in edge_lookup:
            ep = edge_lookup[sid]
            i, j = (ep.node0.to_numpy(int), ep.node1.to_numpy(int))
            support['adjacency'][i, j] = support['adjacency'][j, i] = True
            assert np.all(ep.recipe0.to_numpy() == np.array(support['recipes'])[i])
            assert np.all(ep.recipe1.to_numpy() == np.array(support['recipes'])[j])
        if n:
            tp = condition_lookup.loc[[(support['dataset_id'], r) for r in support['recipes']], tokens].to_numpy(str)
            expected = (tp[:, None, :] != tp[None, :, :]).sum(-1) == 1
            np.testing.assert_array_equal(expected, support['adjacency'])
            nd = node_lookup[sid].sort_values('node')
            assert nd.recipe_id.tolist() == list(support['recipes'])
        assert int(support['adjacency'].sum() // 2) == support['n_edges']
    for c in context.values():
        expected = tuple(sorted(set(values[c['block_a']]) & set(values[c['block_b']])))
        assert expected == tuple(json.loads(c['native_common_recipe_ids']))
    actual_assignments = defaultdict(set)
    for g in group:
        ids = tuple(json.loads(g['context_ids']))
        assert len(ids) == g['n_partners']
        assert len({context[c]['partner_smiles'] for c in ids}) == len(ids)
        expected_recipes = tuple(sorted(set.intersection(*(set(values[context[c]['block_a']]) & set(values[context[c]['block_b']]) for c in ids))))
        assert expected_recipes == su[g['support_id']]['recipes']
        actual_assignments[g['focal_key_id']].add(ids)
    for no, f in focal.items():
        choices = json.loads(f['context_ids_by_partner'])
        expected = set(product(*choices)) if OUT == BASE else {tuple(c) for positions in combinations(range(len(choices)), 3) for c in product(*(choices[p] for p in positions))}
        assert actual_assignments[no] == expected
        assert len(expected) == f['n_assignments']

    @lru_cache(maxsize=4096)
    def cy(context_id, support_id):
        c = context[context_id]
        return np.array([[values[c['block_' + a]][r] for r in su[support_id]['recipes']] for a in ['a', 'b']])

    @lru_cache(maxsize=4096)
    def closure(support_id, ids, projection='ab'):
        yy = np.array([cy(c, support_id) for c in ids])
        y = yy[:, 0].min(0) if projection == 'a' else yy[:, 1].min(0) if projection == 'b' else yy.min((0, 1))
        return forest_closure(y, su[support_id]['adjacency'])
    primary_summary = read('complete_group_threshold_summary.csv.gz').set_index(['group_id', 'threshold'])
    ep_iter = iter(contiguous_groups('complete_group_endpoint_capacities.csv.gz'))
    pair_iter = iter(contiguous_groups('complete_pair_subset_endpoint_capacities.csv.gz'))
    deletion_iter = iter(contiguous_groups('complete_deletion_subset_endpoint_capacities.csv.gz'))
    max_error = 0.0
    n_rows, pair_rows, deletion_rows = (0, 0, 0)
    checked_counts = Counter()

    def equal(actual, expected):
        nonlocal max_error
        actual, expected = (np.asarray(actual), np.asarray(expected))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        finite = np.isfinite(expected)
        if finite.any():
            error = float(np.max(np.abs(actual[finite] - expected[finite])))
            max_error = max(max_error, error)
            assert error <= TOLERANCE
    for no, g in enumerate(group):
        ids, sid = (tuple(json.loads(g['context_ids'])), g['support_id'])
        yy = np.array([cy(c, sid) for c in ids])
        w = yy.min((0, 1))
        high = np.flatnonzero(w >= 0.1)
        i, j = np.triu_indices(len(high), 1)
        i, j = (high[i], high[j])
        epmin = np.minimum(w[i], w[j])
        expected_n = len(i)
        if expected_n:
            actual_no, pp = next(ep_iter)
            assert actual_no == no
            np.testing.assert_array_equal(pp.node0, i)
            np.testing.assert_array_equal(pp.node1, j)
            equal(pp.minimum_endpoint_yield.to_numpy(float), epmin)
            full = closure(sid, ids)[i, j]
            a = closure(sid, ids, 'a')[i, j]
            b = closure(sid, ids, 'b')[i, j]
            single = np.array([closure(sid, (c,))[i, j] for c in ids]).min(0)
            pairs = list(combinations(range(len(ids)), 2))
            pc = np.array([closure(sid, (ids[x], ids[y]))[i, j] for x, y in pairs])
            dc = np.array([closure(sid, ids[:x] + ids[x + 1:])[i, j] for x in range(len(ids))])
            pair_min, delete_min, delete_max = (pc.min(0), dc.min(0), dc.max(0))
            for name, expected in [('K_all', full), ('K_a_all_partners', a), ('K_b_all_partners', b), ('K_context_min', single), ('K_pair_min', pair_min), ('K_deletion_min', delete_min), ('K_deletion_max', delete_max)]:
                equal(pp[name].to_numpy(float), expected)
            path = np.isfinite(full)
            np.testing.assert_array_equal(pp.full_graph_path.to_numpy(bool), path)
            equal(pp.loc[path, 'pair_vs_all_gap'].to_numpy(float), pair_min[path] - full[path])
            actual_no, pe = next(pair_iter)
            assert actual_no == no
            assert len(pe) == len(pairs) * expected_n
            for z, (x, y) in enumerate(pairs):
                zz = pe[(pe.partner_position0 == x) & (pe.partner_position1 == y)]
                np.testing.assert_array_equal(zz.node0, i)
                np.testing.assert_array_equal(zz.node1, j)
                assert (zz.context_id0 == ids[x]).all() and (zz.context_id1 == ids[y]).all()
                equal(zz.K_pair.to_numpy(float), pc[z])
            actual_no, de = next(deletion_iter)
            assert actual_no == no
            assert len(de) == len(ids) * expected_n
            for z, c in enumerate(ids):
                zz = de[de.deleted_partner_position == z]
                np.testing.assert_array_equal(zz.node0, i)
                np.testing.assert_array_equal(zz.node1, j)
                assert (zz.deleted_context_id == c).all()
                equal(zz.K_deletion.to_numpy(float), dc[z])
            pair_rows += len(pe)
            deletion_rows += len(de)
        else:
            full = a = b = single = pair_min = delete_min = np.empty(0)
            path = np.empty(0, bool)
        n_rows += expected_n
        for threshold in [0.1, 0.2, 0.3]:
            eligible, conn = (epmin >= threshold, (epmin >= threshold) & path)
            pair_ok, delete_ok = (pair_min >= threshold, delete_min >= threshold)
            aok, bok, allok = (a >= threshold, b >= threshold, full >= threshold)
            event = conn & pair_ok & ~allok & (len(ids) >= 3)
            strong = event & aok & bok
            minimal = conn & delete_ok & ~allok & (len(ids) >= 3)
            expected = {'n_endpoint_pairs': int(eligible.sum()), 'n_connected_endpoint_pairs': int(conn.sum()), 'n_all_context_routes_feasible': int((conn & (single >= threshold)).sum()), 'n_all_pair_routes_feasible': int((conn & pair_ok).sum()), 'n_all_deletion_routes_feasible': int((conn & delete_ok).sum()), 'n_a_all_partner_route_feasible': int((conn & aok).sum()), 'n_b_all_partner_route_feasible': int((conn & bok).sum()), 'n_both_focal_family_routes_feasible': int((conn & aok & bok).sum()), 'n_full_scope_route_feasible': int((conn & allok).sum()), 'n_pairwise_but_not_full_certificate': int(event.sum()), 'n_strong_pairwise_both_focal_but_not_full_certificate': int(strong.sum()), 'n_all_deletions_but_not_full_certificate': int(minimal.sum()), 'n_strong_all_deletions_but_not_full_certificate': int((minimal & aok & bok).sum())}
            row = primary_summary.loc[no, threshold]
            for field, value in expected.items():
                assert int(row[field]) == value, (no, threshold, field, row[field], value)
                checked_counts[field] += 1
            assert g['n_high_endpoint_pairs_' + str(round(100 * threshold))] == expected['n_endpoint_pairs']
        if (no + 1) % 5000 == 0:
            print('VERIFIED', no + 1, 'groups', n_rows, 'endpoints', flush=True)
    assert next(ep_iter, None) is None and next(pair_iter, None) is None and (next(deletion_iter, None) is None)
    run = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    assert n_rows == run['n_group_endpoint_rows_pool10']
    assert pair_rows == run['n_pair_subset_endpoint_rows'] and deletion_rows == run['n_deletion_subset_endpoint_rows']
    parent = read('complete_20_dataset_18_class_support.csv')
    assert len(parent) == 20 and parent.rxn_class.nunique() == 18
    assert parent.n_parent_endpoints.sum() == 7853943 and parent.n_parent_events.sum() == 68367
    witness = json.loads((OUT / 'chemical_witness.json').read_text(encoding='utf-8'))
    if 'group_id' in witness:
        routes = read('chemical_witness_all_optimal_routes.csv')
        graph = su[witness['support_id']]['adjacency']
        for name, pp in routes.groupby('objective'):
            pp = pp.sort_values('path_step')
            nn = pp.node.to_numpy(int)
            assert nn[0] == witness['endpoint0_node'] and nn[-1] == witness['endpoint1_node']
            assert all((graph[x, y] for x, y in zip(nn[:-1], nn[1:])))
            equal(np.array([pp.objective_node_yield.min()]), np.array([pp.optimized_bottleneck.iloc[0]]))
    result = {'status': 'passed', 'frame': 'full_partner_assignments' if OUT == BASE else 'all_coherent_triples', 'algorithm': 'descending-edge union-find maximum spanning forest, independent of generator Floyd closure', 'n_groups': len(group), 'n_contexts': len(context), 'n_supports': len(su), 'verified_endpoint_rows': n_rows, 'verified_pair_subset_rows': pair_rows, 'verified_deletion_subset_rows': deletion_rows, 'maximum_continuous_error': max_error, 'numerical_tolerance': TOLERANCE, 'thresholds_exactly_recomputed': [0.1, 0.2, 0.3], 'assignment_registry_completeness': 'passed', 'support_node_edge_and_slot_checks': 'passed', 'summary_counts_checked': dict(checked_counts), 'runtime_seconds': time.time() - started, 'verifier_sha256': hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest()}
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
