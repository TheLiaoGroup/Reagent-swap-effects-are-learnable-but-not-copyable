"""Independent descending-edge union-find capacities and reverse-distance paths.

All endpoint memberships, capacities, flags, conditional heldout sums and policy
macro means are checked; inventory endpoint totals are reconciled separately.
Route values are checked on deterministic stratified endpoints in every active
heldout, plus the selected measured witness; this is not an all-route rerun."""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from itertools import combinations, product
from pathlib import Path
import json
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'round27/partner_scope/triple_scope'
TOL = 3e-14
POLICIES = ['source_maximin', 'source_mean_bottleneck', 'full_graph_shortest']

def read(dest, name, **kwargs):
    return pd.read_csv(dest / name, keep_default_na=False, float_precision='round_trip', low_memory=False, **kwargs)

def forest_capacity(y, edges):
    n = len(y)
    ans = np.full((n, n), -np.inf)
    np.fill_diagonal(ans, y)
    parent, members = (list(range(n)), {i: [i] for i in range(n)})

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for capacity, i, j in sorted([(min(y[i], y[j]), int(i), int(j)) for i, j in edges], reverse=True):
        a, b = (find(i), find(j))
        if a == b:
            continue
        ma, mb = (members.pop(a), members.pop(b))
        ans[np.ix_(ma, mb)] = ans[np.ix_(mb, ma)] = capacity
        parent[b] = a
        members[a] = ma + mb
    return ans

def independent_route(neighbors, objective, start, goal, floor):
    allowed = np.ones(len(neighbors), bool) if objective is None else objective >= floor
    distance = np.full(len(neighbors), -1, int)
    distance[goal] = 0
    q = deque([goal])
    while q:
        v = q.popleft()
        for w in neighbors[v]:
            if allowed[w] and distance[w] < 0:
                distance[w] = distance[v] + 1
                q.append(w)
    assert distance[start] >= 0
    route = [start]
    while route[-1] != goal:
        current = route[-1]
        route.append(min((w for w in neighbors[current] if allowed[w] and distance[w] == distance[current] - 1)))
    counts = np.zeros(len(neighbors), int)
    counts[goal] = 1
    for v in np.argsort(distance):
        if distance[v] <= 0:
            continue
        counts[v] = min(2, sum((counts[w] for w in neighbors[v] if allowed[w] and distance[w] == distance[v] - 1)))
    return (route, counts[start] > 1)

def endpoint_groups(filename):
    pending = None
    for chunk in read(OUT, filename, chunksize=180000):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last = (chunk.group_id.iloc[-1], chunk.heldout_position.iloc[-1])
        mask = (chunk.group_id == last[0]) & (chunk.heldout_position == last[1])
        ready, pending = (chunk[~mask], chunk[mask])
        for key, part in ready.groupby(['group_id', 'heldout_position'], sort=False):
            yield (key, part.reset_index(drop=True))
    if pending is not None:
        yield ((pending.group_id.iloc[0], pending.heldout_position.iloc[0]), pending.reset_index(drop=True))

def main():
    started = time.time()
    groups = read(SOURCE, 'complete_group_registry_before_outcomes.csv.gz').to_dict('records')
    contexts = read(SOURCE, 'complete_context_registry.csv.gz').set_index('context_id').to_dict('index')
    focal = read(SOURCE, 'complete_focal_key_registry.csv.gz').set_index('focal_key_id')
    nodes = read(SOURCE, 'full_source_block_node_yields.csv.gz')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in nodes.groupby('block_id', sort=False)}
    support = read(SOURCE, 'complete_support_registry.csv.gz').to_dict('records')
    edges = read(SOURCE, 'full_support_edges.csv.gz')
    edge_map = {int(k): p[['node0', 'node1']].to_numpy(int) for k, p in edges.groupby('support_id', sort=False)}
    conditions = read(SOURCE, 'full_condition_registry.csv.gz').set_index(['dataset_id', 'recipe_id'])
    tokens = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
    for su in support:
        sid = int(su['support_id'])
        su['recipes'] = json.loads(su['recipe_ids'])
        ee = edge_map.get(sid, np.empty((0, 2), int))
        su['edges'] = ee
        n = len(su['recipes'])
        adjacency = np.zeros((n, n), bool)
        if len(ee):
            a, b = ee.T
            adjacency[a, b] = adjacency[b, a] = True
        if n:
            tt = conditions.loc[[(su['dataset_id'], r) for r in su['recipes']], tokens].to_numpy(str)
            np.testing.assert_array_equal(adjacency, (tt[:, None, :] != tt[None, :, :]).sum(-1) == 1)
        su['neighbors'] = [np.flatnonzero(row).tolist() for row in adjacency]
    assignments = defaultdict(set)
    for g in groups:
        ids = tuple(json.loads(g['context_ids']))
        assignments[int(g['focal_key_id'])].add(ids)
        expected = sorted(set.intersection(*(set(values[contexts[c]['block_a']]) & set(values[contexts[c]['block_b']]) for c in ids)))
        assert expected == support[g['support_id']]['recipes']
    for k, f in focal.iterrows():
        cc = json.loads(f.context_ids_by_partner)
        expected = {tuple(ids) for pos in combinations(range(len(cc)), 3) for ids in product(*(cc[p] for p in pos))}
        assert assignments[k] == expected
    reg = read(OUT, 'complete_heldout_statistics.csv.gz').set_index(['group_id', 'heldout_position'])
    before = read(OUT, 'complete_heldout_registry_before_target_outcomes.csv.gz').set_index(['group_id', 'heldout_position'])
    pd.testing.assert_frame_equal(before, reg[before.columns], check_dtype=False)
    ep = iter(endpoint_groups('complete_endpoint_results.csv.gz'))
    witness = json.loads((OUT / 'measured_witness.json').read_text()) if (OUT / 'measured_witness.json').exists() else None
    rows = route_checks = 0
    max_error = 0.0
    counters = Counter()

    def equal(actual, expected):
        nonlocal max_error
        actual, expected = (np.asarray(actual, float), np.asarray(expected, float))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        if finite.any():
            err = float(np.max(np.abs(actual[finite] - expected[finite])))
            max_error = max(max_error, err)
            assert err <= TOL, err
    for no, g in enumerate(groups):
        sid, ids = (int(g['support_id']), json.loads(g['context_ids']))
        su = support[sid]
        y = np.array([[[values[contexts[c]['block_' + side]][r] for r in su['recipes']] for side in ['a', 'b']] for c in ids])
        union_cap = forest_capacity(y.min((0, 1)), su['edges'])
        for holdout in range(3):
            sy4 = y[[i for i in range(3) if i != holdout]]
            sy, my, ty = (sy4.min((0, 1)), sy4.mean((0, 1)), y[holdout].min(0))
            high = np.flatnonzero(sy >= 0.2)
            i, j = np.triu_indices(len(high), 1)
            u, v = (high[i], high[j])
            rr = reg.loc[no, holdout]
            assert rr.n_source_endpoint_pairs == len(u)
            if not len(u):
                continue
            key, pp = next(ep)
            assert key == (no, holdout) and len(pp) == len(u)
            np.testing.assert_array_equal(pp.node0, u)
            np.testing.assert_array_equal(pp.node1, v)
            sc, tc, mc = (forest_capacity(yy, su['edges']) for yy in [sy, ty, my])
            src, tgt, oracle = (sc[u, v], tc[u, v], union_cap[u, v])
            conn = np.isfinite(src)
            sep, tep = (np.minimum(sy[u], sy[v]), np.minimum(ty[u], ty[v]))
            for name, expected in [('source_optimum', src), ('target_optimum', tgt), ('union_oracle', oracle), ('source_endpoint_minimum', sep), ('target_endpoint_minimum', tep)]:
                equal(pp[name], expected)
            source_ok = conn & (src >= 0.2)
            for field, flag in [('full_graph_connected', conn), ('source_route_feasible20', source_ok), ('target_endpoint_failure20', conn & (tep < 0.2)), ('target_own_internal_failure20', conn & (tep >= 0.2) & (tgt < 0.2)), ('shared_route_impossible20', source_ok & (tep >= 0.2) & (oracle < 0.2)), ('source_target_separately_feasible_shared_impossible20', source_ok & (tgt >= 0.2) & (oracle < 0.2))]:
                np.testing.assert_array_equal(pp[field], flag)
                counters[field] += int(flag.sum())
            assert conn.sum() == rr.n_connected_source_endpoint_pairs
            sample = set()
            good = np.flatnonzero(conn)
            if len(good):
                sample.update([int(good[0]), int(good[-1]), int(good[(no * 37 + holdout * 13) % len(good)])])
                sample.add(int(good[np.argmax((oracle - pp.source_maximin_joint.to_numpy(float))[good])]))
                for flag in [source_ok, pp.source_maximin_shortest_tie.to_numpy(bool), pp.source_maximin_avoidable_selection_failure20.to_numpy(bool), tep < 0.2]:
                    select = np.flatnonzero(conn & flag)
                    if len(select):
                        sample.add(int(select[0]))
            if witness is not None and no == witness['group_id'] and (holdout == witness['heldout_position']):
                match = np.flatnonzero((u == witness['endpoint0_node']) & (v == witness['endpoint1_node']))
                assert len(match) == 1
                sample.add(int(match[0]))
            strata = {'connected': conn, 'source_feasible': source_ok, 'source_feasible_target_endpoints_high': source_ok & (tep >= 0.2), 'source_and_target_separately_feasible': source_ok & (tgt >= 0.2), 'union_feasible': conn & (oracle >= 0.2)}
            for policy, objective, cap in [(POLICIES[0], sy, sc), (POLICIES[1], my, mc), (POLICIES[2], None, None)]:
                ps, pt, pj = (pp[policy + '_' + m].to_numpy(float) for m in ['source', 'target', 'joint'])
                equal(pj, np.minimum(ps, pt))
                gap = np.where(conn, oracle - pj, np.nan)
                equal(pp[policy + '_union_gap'], gap)
                assert (gap[conn] >= -TOL).all()
                if policy == POLICIES[0]:
                    equal(ps[conn], src[conn])
                avoidable = conn & (oracle >= 0.2) & (pj < 0.2)
                internal = conn & (tgt >= 0.2) & (pt < 0.2)
                np.testing.assert_array_equal(pp[policy + '_avoidable_selection_failure20'], avoidable)
                np.testing.assert_array_equal(pp[policy + '_target_internal_failure_despite_own_route20'], internal)
                for stratum, mask in strata.items():
                    prefix = stratum + '__'
                    assert rr[prefix + 'denominator_sum'] == mask.sum()
                    for name, vv in [('joint', pj), ('union_gap', gap), ('success20', pj >= 0.2), ('source_failure20', ps < 0.2), ('avoidable_selection_failure20', avoidable), ('target_internal_failure_despite_own_route20', internal)]:
                        expected_sum = float(vv[mask].sum())
                        assert abs(rr[prefix + policy + '_' + name + '_sum'] - expected_sum) < 1e-09
                for z in sorted(sample):
                    route, tie = independent_route(su['neighbors'], objective, int(u[z]), int(v[z]), cap[u[z], v[z]] if cap is not None else 0)
                    equal([ps[z], pt[z], pj[z]], [sy[route].min(), ty[route].min(), min(sy[route].min(), ty[route].min())])
                    assert pp[policy + '_edges'].iloc[z] == len(route) - 1
                    assert pp[policy + '_shortest_tie'].iloc[z] == tie
                    route_checks += 1
            rows += len(pp)
        if (no + 1) % 2000 == 0:
            print('VERIFIED', no + 1, 'groups', rows, 'endpoint rows', route_checks, 'routes', round(time.time() - started, 1), 'seconds', flush=True)
    assert next(ep, None) is None
    census = json.loads((OUT / 'design_census.json').read_text())
    assert rows == census['n_source_defined_endpoint_pairs']
    key_stats = read(OUT, 'complete_470_focal_key_statistics.csv.gz')
    ds_stats = read(OUT, 'complete_20_dataset_statistics.csv')
    cl_stats = read(OUT, 'complete_18_class_statistics.csv')
    assert len(key_stats) == 512 and len(ds_stats) == 20 and (len(cl_stats) == 18)
    for frame in [key_stats, ds_stats, cl_stats]:
        assert frame.n_source_endpoint_pairs.sum() == rows
        assert frame.n_connected_source_endpoint_pairs.sum() == counters['full_graph_connected']
    weighting = read(OUT, 'continuous_and_threshold_weighting_summary.csv')
    flat_reg = reg.reset_index()
    macro_max_error = 0.0
    for unit in weighting.weighting_unit.unique():
        if unit in ['endpoint', 'heldout']:
            mm = flat_reg
        else:
            grouping = {'focal_key': 'focal_key_id', 'global_exact_molecular_pair': 'molecular_pair_id', 'dataset': 'dataset_id', 'reaction_class': 'rxn_class'}[unit]
            columns = [c for c in flat_reg if '__' in c and c.endswith('_sum')]
            mm = flat_reg.groupby(grouping)[columns].sum()
        for rr in weighting[weighting.weighting_unit == unit].to_dict('records'):
            prefix = rr['stratum'] + '__'
            denominator = mm[prefix + 'denominator_sum']
            metric = mm[prefix + rr['policy'] + '_' + rr['metric'] + '_sum']
            active = denominator > 0
            expected = metric.sum() / denominator.sum() if unit == 'endpoint' else (metric[active] / denominator[active]).mean()
            error = abs(float(expected) - rr['value'])
            macro_max_error = max(macro_max_error, error)
            assert error <= 3e-14
            assert rr['endpoint_denominator'] == denominator.sum()
            if unit != 'endpoint':
                assert rr['n_active_units'] == active.sum()
    output = {'status': 'passed', 'seconds': time.time() - started, 'n_triples': len(groups), 'n_heldouts': len(reg), 'n_parent_focal_keys': len(focal), 'n_endpoint_rows_capacity_and_flag_checked': rows, 'n_deterministically_stratified_routes_checked': route_checks, 'maximum_absolute_capacity_or_route_error': max_error, 'n_policy_weighting_rows_checked': len(weighting), 'maximum_absolute_policy_weighting_error': macro_max_error, 'capacities': 'independent descending-edge union-find maximum-spanning forest; primary uses Floyd max-min closure', 'routes': 'independent reverse-distance greedy lexical reconstruction; primary uses forward BFS predecessor trees', 'route_scope': 'deterministic first/last/hash/max-gap and source-feasible/tie/selection-failure/target-endpoint-failure examples in every active heldout plus explicit measured witness; not every exported route', 'all_graph_edges_checked_against_condition_tokens': True, 'all_triple_acquisition_assignments_reconstructed': True, 'all_endpoint_membership_defined_only_from_sources': True, 'counts': dict(counters), 'tolerance': TOL}
    (OUT / 'independent_verification.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('VERIFICATION', json.dumps(output), flush=True)

def self_test():
    import analyze_heldout_routes as a
    groups, focal, contexts, values, supports = a.load()
    checks = 0
    for gid in [0, 282, 7020]:
        g = groups[gid]
        sid, ids = (g['support_id'], json.loads(g['context_ids']))
        su = supports[sid]
        y = a.yield_array(ids, sid, contexts, values, supports)
        ee = np.array(list(zip(*np.where(np.triu(su['adjacency'], 1))))).astype(int)
        for holdout in range(3):
            sy4 = y[[i for i in range(3) if i != holdout]]
            sy, my, ty = (sy4.min((0, 1)), sy4.mean((0, 1)), y[holdout].min(0))
            high = np.flatnonzero(sy >= 0.2)
            i, j = np.triu_indices(len(high), 1)
            u, v = (high[i], high[j])
            for objective in [sy, my, None]:
                cap = None if objective is None else forest_capacity(objective, ee)
                if cap is not None:
                    np.testing.assert_allclose(cap, a.widest(objective, su['adjacency']), rtol=0, atol=0)
                result = a.policy_results(objective, cap, su, u, v, sy, ty)
                for z, (s, t) in enumerate(zip(u, v)):
                    if su['labels'][s] != su['labels'][t]:
                        continue
                    route, tie = independent_route(su['neighbors'], objective, int(s), int(t), cap[s, t] if cap is not None else 0)
                    assert result['source'][z] == sy[route].min()
                    assert result['target'][z] == ty[route].min()
                    assert result['edges'][z] == len(route) - 1
                    assert result['shortest_tie'][z] == tie
                    checks += 1
    print('SELF_TEST', checks, 'all-pair route reconstructions passed', flush=True)
if __name__ == '__main__':
    self_test() if '--self-test' in sys.argv else main()
