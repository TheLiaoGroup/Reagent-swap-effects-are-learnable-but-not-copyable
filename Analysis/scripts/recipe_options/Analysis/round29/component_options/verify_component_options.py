"""Independent BFS thresholds and descending-node union-find capacities.

No primary optimizer is imported. Every singleton/deletion, pair/triple state,
candidate-set census, summary minimum and saved path is checked."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
SOURCE = ANALYSIS / 'round27/partner_scope/triple_scope'
R28 = ANALYSIS / 'round28/shared_bridges'
SLOTS = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
H = 0.2
TOL = 2e-14

def input_hashes():
    paths = [OUT / name for name in ['all_single_options.csv.gz', 'all_pair_options.csv.gz', 'all_triple_options.csv.gz', 'certificate_option_summary.csv', 'chemical_witness_routes.csv', 'complete_class_option_summary.csv']]
    paths += [SOURCE / name for name in ['complete_group_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_condition_registry.csv.gz', 'full_source_block_node_yields.csv.gz']]
    paths += [R28 / 'all_strong_scope_reconciliation20.csv.gz', Path(__file__).resolve()]
    result = {}
    for path in paths:
        with path.open('rb') as stream:
            result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result

def read(path):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', low_memory=False)

def oid(slot, token):
    return hashlib.sha256((SLOTS[slot] + '|' + token).encode()).hexdigest()[:16]

def capacity_error(actual, expected):
    actual, expected = (float(actual), float(expected))
    assert np.isfinite(actual) or np.isneginf(actual)
    assert np.isfinite(expected) or np.isneginf(expected)
    assert np.isneginf(actual) == np.isneginf(expected)
    error = abs(actual - expected) if np.isfinite(expected) else 0.0
    assert error <= TOL
    return error

def require_registered_certificates(table, registered, label):
    assert set(table.certificate_id) <= registered, f'Unregistered certificate in {label}'

def threshold_connected(weights, neighbors, start, goal, allowed):
    high = allowed & (weights >= H)
    if not high[start] or not high[goal]:
        return False
    seen = {start}
    queue = [start]
    for node in queue:
        if node == goal:
            return True
        for nxt in neighbors[node]:
            if high[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False

def node_activation_capacity(weights, neighbors, start, goal, allowed):
    if not allowed[start] or not allowed[goal]:
        return -np.inf
    parent = list(range(len(weights)))
    size = [1] * len(weights)
    active = np.zeros(len(weights), bool)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for node in sorted(np.flatnonzero(allowed), key=lambda v: (-weights[v], int(v))):
        active[node] = True
        for nxt in neighbors[node]:
            if active[nxt]:
                a, b = (find(node), find(nxt))
                if a != b:
                    if size[a] < size[b]:
                        a, b = (b, a)
                    parent[b] = a
                    size[a] += size[b]
        if active[start] and active[goal] and (find(start) == find(goal)):
            return float(weights[node])
    return -np.inf

def main():
    started = time.time()
    (OUT / 'independent_verification.json').unlink(missing_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    evidence = input_hashes()
    single = read(OUT / 'all_single_options.csv.gz')
    pair = read(OUT / 'all_pair_options.csv.gz')
    triple = read(OUT / 'all_triple_options.csv.gz')
    summary = read(OUT / 'certificate_option_summary.csv')
    certs = read(R28 / 'all_strong_scope_reconciliation20.csv.gz')
    groups = read(SOURCE / 'complete_group_registry.csv.gz').set_index('group_id')
    contexts = read(SOURCE / 'complete_context_registry.csv.gz').set_index('context_id')
    support = read(SOURCE / 'complete_support_registry.csv.gz').set_index('support_id')
    condition = read(SOURCE / 'full_condition_registry.csv.gz').drop_duplicates(['dataset_id', 'recipe_id'])
    condmap = {d: p.set_index('recipe_id') for d, p in condition.groupby('dataset_id')}
    nodes = read(SOURCE / 'full_source_block_node_yields.csv.gz')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in nodes.groupby('block_id', sort=False)}
    expected_certs = {(int(r.group_id), int(r.node0), int(r.node1)) for r in certs.itertuples()}
    actual_certs = {(int(r.group_id), int(r.node0), int(r.node1)) for r in summary.itertuples()}
    assert len(summary) == len(expected_certs) == 285 and expected_certs == actual_certs
    assert summary.certificate_id.is_unique
    assert int(summary.complete_endpoint_corners.sum()) == 239
    registered = set(summary.certificate_id)
    for label, table in [('single', single), ('pair', pair), ('triple', triple)]:
        require_registered_certificates(table, registered, label)
    lookup = {name: {k: p for k, p in table.groupby('certificate_id', sort=False)} for name, table in [('single', single), ('pair', pair), ('triple', triple)]}
    witness = read(OUT / 'chemical_witness_routes.csv')
    require_registered_certificates(witness, registered, 'witness routes')
    state_counts = Counter()
    maximum_error = 0.0
    path_count = 0
    capacity_checks = threshold_checks = 0
    exact_minimum = Counter()

    def compare(capacity, expected):
        nonlocal maximum_error
        maximum_error = max(maximum_error, capacity_error(capacity, expected))
    for ix, row in enumerate(summary.to_dict('records')):
        cid = row['certificate_id']
        gid, start, goal = [int(row[k]) for k in ['group_id', 'node0', 'node1']]
        g = groups.loc[gid]
        recipes = json.loads(support.loc[g.support_id, 'recipe_ids'])
        conditions = condmap[g.dataset_id].loc[recipes]
        tokens = conditions[SLOTS].to_numpy(str)
        adjacency = (tokens[:, None, :] != tokens[None, :, :]).sum(-1) == 1
        neighbors = [np.flatnonzero(r).tolist() for r in adjacency]
        ids = json.loads(g.context_ids)
        y = np.array([[[values[contexts.loc[c, 'block_' + side]][r] for r in recipes] for side in ['a', 'b']] for c in ids])
        weights = y.min((0, 1))
        baseline = (tokens == tokens[start]) | (tokens == tokens[goal])
        base_mask = baseline.all(1)
        options = [(slot, token) for slot in range(len(SLOTS)) for token in sorted(set(tokens[:, slot])) if token not in {tokens[start, slot], tokens[goal, slot]}]
        option_by_id = {oid(slot, token): (slot, token) for slot, token in options}
        assert len(option_by_id) == len(options) == row['n_extra_options']
        expected_corners = math.prod((len(set([tokens[start, k], tokens[goal, k]])) for k in range(len(SLOTS))))
        assert bool(row['complete_endpoint_corners']) == (int(base_mask.sum()) == expected_corners)
        compare(node_activation_capacity(weights, neighbors, start, goal, base_mask), row['K_endpoint_all'])
        compare(node_activation_capacity(weights, neighbors, start, goal, np.ones(len(weights), bool)), row['K_all'])
        assert not threshold_connected(weights, neighbors, start, goal, base_mask)
        assert threshold_connected(weights, neighbors, start, goal, np.ones(len(weights), bool))
        assert min((node_activation_capacity(y[p].min(0), neighbors, start, goal, base_mask) for p in range(3))) >= H

        def expansion(added):
            mask = baseline.copy()
            for slot, token in added:
                mask[:, slot] |= tokens[:, slot] == token
            return mask.all(1)

        def check_path(serialized, allowed, expected):
            nonlocal path_count
            route = json.loads(serialized) if isinstance(serialized, str) else serialized
            if not np.isfinite(float(expected)):
                assert not route
                return
            assert route and route[0] == start and (route[-1] == goal)
            assert all((0 <= n < len(weights) and allowed[n] for n in route))
            assert all((adjacency[a, b] for a, b in zip(route[:-1], route[1:])))
            compare(float(weights[route].min()), expected)
            path_count += 1

        def state(allowed, expected, flag, serialized):
            nonlocal capacity_checks, threshold_checks
            cap = node_activation_capacity(weights, neighbors, start, goal, allowed)
            compare(cap, expected)
            high = threshold_connected(weights, neighbors, start, goal, allowed)
            assert high == bool(flag) == (cap >= H)
            check_path(serialized, allowed, expected)
            capacity_checks += 1
            threshold_checks += 1
            return cap
        ss = lookup['single'].get(cid, pd.DataFrame())
        assert len(ss) == len(options) and (not ss.option_id.duplicated().any())
        assert set(ss.option_id) == set(option_by_id)
        suff, necessary = (set(), set())
        capacity_by_option = {}
        for r in ss.to_dict('records'):
            slot, token = option_by_id[r['option_id']]
            assert r['slot'] == SLOTS[slot] and r['token'] == token
            assert r['endpoint_token0'] == tokens[start, slot] and r['endpoint_token1'] == tokens[goal, slot]
            assert bool(r['endpoint_changing_slot']) == (tokens[start, slot] != tokens[goal, slot])
            allowed = expansion([(slot, token)])
            assert allowed.sum() == r['n_single_allowed_nodes']
            delete = tokens[:, slot] != token
            assert (~delete).sum() == r['n_full_nodes_with_option']
            capacity_by_option[r['option_id']] = state(allowed, r['K_single'], r['single_sufficient20'], r['single_route_nodes'])
            state(delete, r['K_after_full_deletion'], not r['necessary_in_full_graph20'], r['deletion_route_nodes'])
            if r['single_sufficient20']:
                suff.add(r['option_id'])
            if r['necessary_in_full_graph20']:
                necessary.add(r['option_id'])
        assert suff == set(json.loads(row['single_sufficient_option_ids']))
        assert necessary == set(json.loads(row['necessary_option_ids']))
        assert len(suff) == row['n_single_sufficient'] and len(necessary) == row['n_necessary_options_in_full_graph']
        stage_success = {1: len(suff)}
        for order, table_name, flag_field, k_field, mask_field in [(2, 'pair', 'pair_sufficient20', 'K_pair', 'n_pair_allowed_nodes'), (3, 'triple', 'triple_sufficient20', 'K_triple', 'n_triple_allowed_nodes')]:
            run_stage = not bool(suff) if order == 2 else not bool(suff) and stage_success[2] == 0
            pp = lookup[table_name].get(cid, pd.DataFrame())
            expected = set(combinations([oid(*o) for o in options], order)) if run_stage else set()
            actual = {tuple((r[f'option{i}_id'] for i in range(1, order + 1))) for r in pp.to_dict('records')}
            assert len(pp) == len(actual) == len(expected) and actual == expected
            assert row[f'all_{table_name}s_tested'] == run_stage
            assert row[f'n_{table_name}s_tested'] == len(expected)
            successes = 0
            best = -np.inf
            for r in pp.to_dict('records'):
                added = [option_by_id[r[f'option{i}_id']] for i in range(1, order + 1)]
                for i, (slot, token) in enumerate(added, 1):
                    assert r[f'slot{i}'] == SLOTS[slot] and r[f'token{i}'] == token
                allowed = expansion(added)
                assert allowed.sum() == r[mask_field]
                cap = state(allowed, r[k_field], r[flag_field], r['route_nodes'])
                best = max(best, cap)
                successes += bool(r[flag_field])
            stage_success[order] = successes
            assert row[f'n_sufficient_{table_name}s'] == successes
            if run_stage:
                compare(best, row[f'best_{table_name}_K'])
        minimum = next((n for n in [1, 2, 3] if stage_success[n]), None)
        if minimum is None:
            assert row['minimum_extra_options_exact'] == '' and row['minimum_extra_options_lower_bound'] == 4
        else:
            assert int(row['minimum_extra_options_exact']) == minimum == int(row['minimum_extra_options_lower_bound'])
            exact_minimum[minimum] += 1
        for _, pp in witness[witness.certificate_id == cid].groupby(['route_kind', 'option1_id', 'option2_id', 'option3_id'], dropna=False):
            rr = pp.sort_values('step')
            first = rr.iloc[0]
            added = [option_by_id[first[f'option{i}_id']] for i in range(1, 4) if first[f'option{i}_id']]
            if first.route_kind == 'full_shared':
                allowed = np.ones(len(weights), bool)
            elif first.route_kind == 'delete_option_from_full':
                slot, token = added[0]
                allowed = tokens[:, slot] != token
            else:
                allowed = expansion(added)
            nn = rr.node.to_numpy(int)
            assert rr.recipe_id.tolist() == [recipes[n] for n in nn]
            np.testing.assert_allclose(rr.shared_node_yield.to_numpy(float), weights[nn], rtol=0, atol=TOL)
            cap = node_activation_capacity(weights, neighbors, start, goal, allowed)
            compare(cap, first.optimized_bottleneck)
            check_path(nn.tolist(), allowed, cap)
        state_counts['certificates'] += 1
        state_counts['single_options'] += len(ss)
        state_counts['single_sufficient'] += len(suff)
        state_counts['necessary_in_full_graph'] += len(necessary)
        if (ix + 1) % 50 == 0:
            print('VERIFIED', ix + 1, 'certificates', capacity_checks, 'continuous states', round(time.time() - started, 1), 'seconds', flush=True)
    inv = read(OUT / 'complete_class_option_summary.csv')
    assert inv.dataset_id.nunique() == 20 and inv.rxn_class.nunique() == 18 and (len(inv) == 60)
    for row in inv.to_dict('records'):
        pp = summary[summary.dataset_id == row['dataset_id']]
        if row['endpoint_corner_stratum'] != 'all':
            pp = pp[pp.complete_endpoint_corners == (row['endpoint_corner_stratum'] == 'complete')]
        ss = single[single.certificate_id.isin(pp.certificate_id)]
        expected = {'n_strong_certificates': len(pp), 'n_triples': pp.group_id.nunique(), 'n_molecular_pairs': pp.molecular_pair_id.nunique(), 'n_minimum1': int((pp.minimum_extra_options_exact.astype(str) == '1').sum()), 'n_minimum2': int((pp.minimum_extra_options_exact.astype(str) == '2').sum()), 'n_minimum3': int((pp.minimum_extra_options_exact.astype(str) == '3').sum()), 'n_minimum_at_least4': int((pp.minimum_extra_options_lower_bound >= 4).sum()), 'n_multiple_singleton_alternatives': int((pp.n_single_sufficient >= 2).sum()), 'n_with_full_graph_necessary_option': int((pp.n_necessary_options_in_full_graph >= 1).sum()), 'n_tested_options': len(ss), 'n_failed_single_options': int((~ss.single_sufficient20).sum())}
        for k, v in expected.items():
            assert row[k] == v, (k, row[k], v)
    assert input_hashes() == evidence, 'Verification inputs changed during the independent check'
    result = {'status': 'passed', 'seconds': time.time() - started, 'primary_algorithm': 'heap maximum-bottleneck path', 'started_utc': started_utc, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'verification_inputs_sha256': evidence, 'fresh_independent_verification': True, 'independent_continuous_algorithm': 'descending node activation with union-find, stop when endpoints merge', 'independent_threshold_algorithm': 'BFS over allowed nodes with all-six-target minimum>=20%', 'n_certificates': len(summary), 'n_complete_endpoint_corner_certificates': int(summary.complete_endpoint_corners.sum()), 'n_single_options': len(single), 'n_single_deletions': len(single), 'n_pair_options': len(pair), 'n_triple_options': len(triple), 'n_continuous_states_checked': capacity_checks, 'n_threshold_states_checked': threshold_checks, 'n_saved_paths_checked': path_count, 'all_candidate_subsets_exhaustively_reconstructed': True, 'minimum_extra_option_counts': dict(exact_minimum), 'maximum_absolute_capacity_or_path_error': maximum_error, 'tolerance': TOL, 'counts': dict(state_counts), 'n_dataset_corner_summary_rows_checked': len(inv), 'scope': 'all exported single/deletion/pair/triple capacities and threshold states, their saved paths, all witness paths and exact minima; original workbook checks are separate'}
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
