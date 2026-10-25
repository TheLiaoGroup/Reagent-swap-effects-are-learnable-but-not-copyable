"""All-source-optimal alternatives on the unchanged R28 source-high20 frame."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
R28 = ANALYSIS / 'round28/heldout_routes'
R27 = ANALYSIS / 'round27/partner_scope/triple_scope'
H = 0.2

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
a = load_module('r28_readonly', R28 / 'analyze_heldout_routes.py')

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def write_json(value, name):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def endpoint_groups():
    columns = ['group_id', 'heldout_position', 'node0', 'node1', 'full_graph_connected', 'source_optimum', 'target_endpoint_minimum', 'target_optimum', 'union_oracle', 'source_maximin_joint', 'source_maximin_target', 'source_maximin_edges', 'full_graph_shortest_edges']
    pending = None
    for chunk in read(R28 / 'complete_endpoint_results.csv.gz', usecols=columns, chunksize=200000):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last = (chunk.group_id.iloc[-1], chunk.heldout_position.iloc[-1])
        mask = (chunk.group_id == last[0]) & (chunk.heldout_position == last[1])
        ready, pending = (chunk[~mask], chunk[mask])
        for key, part in ready.groupby(['group_id', 'heldout_position'], sort=False):
            yield (key, part.reset_index(drop=True))
    if pending is not None:
        yield ((pending.group_id.iloc[0], pending.heldout_position.iloc[0]), pending.reset_index(drop=True))

def forest_capacity(y, edges):
    """Independent descending-edge union-find, with forbidden nodes removed."""
    n = len(y)
    answer = np.full((n, n), -np.inf)
    np.fill_diagonal(answer, y)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    weighted = [(min(y[i], y[j]), int(i), int(j)) for i, j in edges if np.isfinite(y[i]) and np.isfinite(y[j])]
    for capacity, i, j in sorted(weighted, reverse=True):
        x, z = (find(i), find(j))
        if x == z:
            continue
        mx, mz = (members.pop(x), members.pop(z))
        answer[np.ix_(mx, mz)] = answer[np.ix_(mz, mx)] = capacity
        parent[z] = x
        members[x] = mx + mz
    return answer
STATES = ['disconnected', 'source_route_infeasible20', 'target_endpoint_low20', 'target_own_route_infeasible20', 'separately_feasible_union_infeasible20', 'frozen_source_maximin_success20', 'frozen_failure_tie_rescue_only20', 'frozen_failure_positive_source_compromise20']
FIELDS = ['n_endpoints', *['n_' + s for s in STATES], 'n_union_feasible20', 'n_frozen_failure20', 'n_F_target20_finite', 'F_target20_finite_sum', 'n_positive_sacrifice_union20', 'source_sacrifice_union20_sum', 'source_sacrifice_positive_union20_sum']

def main():
    started = time.time()
    groups, focal, contexts, values, supports = a.load()
    registry = read(R28 / 'complete_heldout_registry_before_target_outcomes.csv.gz')
    native = read(R27 / 'complete_group_pair_support_diagnostic.csv.gz')
    registry = registry.merge(native[['group_id', 'all_native_pair_supports_identical_to_full_support']], on='group_id', validate='many_to_one')
    registry['heldout_id'] = registry.group_id * 3 + registry.heldout_position
    registry = registry.set_index('heldout_id')
    assert list(registry.index) == list(range(133002))
    aggregate = np.zeros((len(registry), 3, len(FIELDS)))
    field_index = {k: i for i, k in enumerate(FIELDS)}
    counts = Counter()
    candidates = []
    max_error = 0.0
    verification_rows = 0
    new_finite_rows = 0
    cached_gid, y = (None, None)
    header = ['group_id', 'heldout_position', 'node0', 'node1', 'F_target20', 'source_sacrifice_if_target20_feasible', 'all_source_optima_fail_target20', 'source_optimal_target20_exists', 'endpoint_topology', 'state20']
    with gzip.open(OUT / 'complete_source_compromise_endpoints.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=2) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for (gid, holdout), part in endpoint_groups():
            gid, holdout = (int(gid), int(holdout))
            g = groups[gid]
            sid, ids = (int(g['support_id']), json.loads(g['context_ids']))
            su = supports[sid]
            if cached_gid != gid:
                y = a.yield_array(ids, sid, contexts, values, supports)
                cached_gid = gid
            sy = y[[i for i in range(3) if i != holdout]].min((0, 1))
            ty = y[holdout].min(0)
            u, v = (part.node0.to_numpy(int), part.node1.to_numpy(int))
            conn = part.full_graph_connected.to_numpy(bool)
            source = part.source_optimum.to_numpy(float)
            target = part.target_optimum.to_numpy(float)
            oracle = part.union_oracle.to_numpy(float)
            endpoint = part.target_endpoint_minimum.to_numpy(float)
            joint = part.source_maximin_joint.to_numpy(float)
            masked_source = np.where(ty >= H, sy, -np.inf)
            cap = a.widest(masked_source, su['adjacency'])
            F = cap[u, v]
            edges = np.column_stack(np.where(np.triu(su['adjacency'], 1)))
            independent = forest_capacity(masked_source, edges)[u, v]
            np.testing.assert_array_equal(np.isneginf(F), np.isneginf(independent))
            finite = np.isfinite(F)
            if finite.any():
                error = float(np.max(np.abs(F[finite] - independent[finite])))
                max_error = max(max_error, error)
                assert error == 0.0
            verification_rows += len(part)
            new_finite_rows += int(finite.sum())
            assert np.all(F <= source)
            np.testing.assert_array_equal(F >= H, oracle >= H)
            np.testing.assert_array_equal(finite, conn & (target >= H))
            optimal_exists = finite & (F == source)
            all_optimal_fail = conn & ~optimal_exists
            union = conn & (oracle >= H)
            frozen_failure = union & (joint < H)
            tie_only = frozen_failure & optimal_exists
            compromise = frozen_failure & ~optimal_exists
            np.testing.assert_array_equal(compromise, union & (F < source))
            sacrifice = np.full(len(part), np.nan)
            sacrifice[finite] = source[finite] - F[finite]
            state = np.full(len(part), 'disconnected', object)
            state[conn & (source < H)] = 'source_route_infeasible20'
            state[conn & (source >= H) & (endpoint < H)] = 'target_endpoint_low20'
            state[conn & (source >= H) & (endpoint >= H) & (target < H)] = 'target_own_route_infeasible20'
            state[conn & (source >= H) & (target >= H) & ~union] = 'separately_feasible_union_infeasible20'
            state[union & ~frozen_failure] = 'frozen_source_maximin_success20'
            state[tie_only] = 'frozen_failure_tie_rescue_only20'
            state[compromise] = 'frozen_failure_positive_source_compromise20'
            adjacent = part.full_graph_shortest_edges.to_numpy(int) == 1
            assert not np.any((tie_only | compromise) & adjacent)
            topology = np.where(adjacent, 'adjacent', 'nonadjacent')
            topology[~conn] = 'disconnected'
            writer.writerows(zip(np.full(len(part), gid), np.full(len(part), holdout), u, v, F, sacrifice, all_optimal_fail, optimal_exists, topology, state))
            data = np.column_stack([np.ones(len(part)), *[state == s for s in STATES], union, frozen_failure, finite, np.where(finite, F, 0.0), compromise, np.where(union, sacrifice, 0.0), np.where(compromise, sacrifice, 0.0)])
            for j, mask in enumerate([np.ones(len(part), bool), adjacent, conn & ~adjacent]):
                aggregate[gid * 3 + holdout, j] = data[mask].sum(0)
            counts.update(state)
            for label, mask in [('positive_source_compromise', compromise), ('tie_rescue_only', tie_only)]:
                for z in np.flatnonzero(mask):
                    candidates.append({'classification': label, 'group_id': gid, 'heldout_position': holdout, 'node0': int(u[z]), 'node1': int(v[z]), 'source_optimum': source[z], 'F_target20': F[z], 'source_sacrifice': sacrifice[z], 'union_oracle': oracle[z], 'frozen_joint': joint[z], 'frozen_target': part.source_maximin_target.iloc[z], 'target_optimum': target[z], 'target_endpoint_minimum': endpoint[z], 'frozen_edges': int(part.source_maximin_edges.iloc[z]), 'focal_key_id': g['focal_key_id'], 'dataset_id': g['dataset_id'], 'rxn_class': g['rxn_class'], 'molecular_pair_id': g['molecular_pair_id'], 'all_native_pair_supports_identical_to_full_support': bool(registry.iloc[gid * 3 + holdout].all_native_pair_supports_identical_to_full_support)})
            if verification_rows % 1000000 < len(part):
                print('PROGRESS', verification_rows, 'rows', round(time.time() - started, 1), 'seconds', flush=True)
    assert verification_rows == 9631618
    assert counts['disconnected'] == 2092626
    assert counts['frozen_failure_tie_rescue_only20'] + counts['frozen_failure_positive_source_compromise20'] == 31145
    assert counts['frozen_source_maximin_success20'] + 31145 == 3730956
    tables = []
    for j, topology in enumerate(['all', 'adjacent', 'nonadjacent']):
        frame = registry.reset_index().join(pd.DataFrame(aggregate[:, j, :], columns=FIELDS))
        frame['endpoint_topology'] = topology
        tables.append(frame)
    heldout = pd.concat(tables, ignore_index=True)
    save(heldout, 'complete_heldout_compromise_statistics.csv.gz')
    save(pd.DataFrame(candidates), 'all_frozen_failure_candidates.csv.gz')
    inventory = read(R27 / 'complete_20_dataset_18_class_support.csv')
    coverage = []
    for unit, keys in [('focal_key', ['focal_key_id']), ('dataset', ['dataset_id']), ('reaction_class', ['rxn_class'])]:
        pooled = heldout.groupby(keys + ['endpoint_topology'], as_index=False)[FIELDS].sum()
        if unit == 'focal_key':
            pooled = focal[['focal_key_id', 'dataset_id', 'rxn_class', 'molecular_pair_id']].merge(pooled, on='focal_key_id', validate='one_to_many')
        save(pooled, 'complete_' + unit + '_compromise_statistics.csv' + ('.gz' if unit == 'focal_key' else ''))
    weight_rows = []
    denominators = ['n_endpoints', 'n_union_feasible20', 'n_frozen_failure20', 'n_positive_sacrifice_union20']
    numerators = ['n_frozen_failure_tie_rescue_only20', 'n_frozen_failure_positive_source_compromise20', 'source_sacrifice_union20_sum', 'source_sacrifice_positive_union20_sum']
    for topology in ['all', 'nonadjacent', 'adjacent']:
        for native_scope in ['all', 'native_support_equal', 'native_support_has_extra']:
            frame = heldout[heldout.endpoint_topology == topology]
            if native_scope != 'all':
                frame = frame[frame.all_native_pair_supports_identical_to_full_support == (native_scope == 'native_support_equal')]
            for unit, key in [('endpoint', None), ('heldout', None), ('focal_key', 'focal_key_id'), ('dataset', 'dataset_id'), ('reaction_class', 'rxn_class')]:
                mm = frame[FIELDS] if key is None else frame.groupby(key)[FIELDS].sum()
                for den in denominators:
                    active = mm[den] > 0
                    for num in numerators:
                        value = (float(mm.loc[active, num].sum() / mm.loc[active, den].sum()) if unit == 'endpoint' else float((mm.loc[active, num] / mm.loc[active, den]).mean())) if active.any() else None
                        weight_rows.append({'endpoint_topology': topology, 'native_scope': native_scope, 'weighting_unit': unit, 'numerator': num, 'denominator': den, 'value': value, 'n_active_units': int(active.sum()) if unit != 'endpoint' else int(mm.loc[active, den].sum()), 'denominator_endpoint_views': int(mm.loc[active, den].sum()), 'unit_rule': 'endpoint pooled; otherwise endpoint ratio within unit then equal mean across units with positive denominator'})
    save(pd.DataFrame(weight_rows), 'compromise_weighting_summary.csv')
    for topology in ['all', 'nonadjacent', 'adjacent']:
        frame = heldout[heldout.endpoint_topology == topology]
        for state_name in STATES + ['union_feasible20', 'frozen_failure20']:
            col = 'n_' + state_name
            active = frame[frame[col] > 0]
            coverage.append({'endpoint_topology': topology, 'state20': state_name, 'n_endpoint_views': int(active[col].sum()), 'n_holdouts': len(active), 'n_triples': active.group_id.nunique(), 'n_focal_keys': active.focal_key_id.nunique(), 'n_exact_molecular_pairs': active.molecular_pair_id.nunique(), 'n_datasets': active.dataset_id.nunique(), 'n_reaction_classes': active.rxn_class.nunique(), 'reaction_classes': ';'.join(sorted(active.rxn_class.unique()))})
    save(pd.DataFrame(coverage), 'class_and_hierarchy_coverage.csv')
    manifest_paths = [R28 / 'complete_endpoint_results.csv.gz', R28 / 'complete_heldout_registry_before_target_outcomes.csv.gz', R28 / 'analyze_heldout_routes.py', R28 / 'ROUND28_HELDOUT_ROUTE_SCOPE_REPORT.md']
    manifest_paths += [R27 / name for name in ['complete_group_registry_before_outcomes.csv.gz', 'complete_focal_key_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_support_edges.csv.gz', 'full_support_nodes.csv.gz', 'full_source_block_node_yields.csv.gz', 'complete_group_pair_support_diagnostic.csv.gz', 'complete_20_dataset_18_class_support.csv']]
    manifest = []
    for path in manifest_paths:
        with path.open('rb') as f:
            manifest.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(f, 'sha256').hexdigest(), 'access': 'read_only'})
    write_json(manifest, 'source_manifest.json')
    finite_sacrifices = pd.DataFrame(candidates).query("classification == 'positive_source_compromise'").source_sacrifice
    summary = {'status': 'completed', 'seconds': time.time() - started, 'n_triples': len(groups), 'n_holdouts': len(registry), 'n_focal_keys': len(focal), 'n_datasets_inventory': int(registry.dataset_id.nunique()), 'n_reaction_classes_inventory': int(registry.rxn_class.nunique()), 'source_endpoint_frame_threshold': H, 'target_route_threshold': H, 'n_endpoint_rows': verification_rows, 'states20': dict(counts), 'n_F_target20_finite': new_finite_rows, 'positive_sacrifice_quantiles_fraction_yield': {str(q): float(finite_sacrifices.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]}, 'no_target_policy': True, 'diagnostic': 'F=max path source bottleneck subject to target node minimum>=20%; all path lengths allowed', 'source_sacrifice': 'B_source-F_target20 when F finite; no route represented by -inf F and undefined sacrifice', 'all_optimal_failure_rule': 'F_target20 < B_source, including target20 infeasibility; source-optimal existence uses exact equality of measured bottleneck values', 'source_high20_frame_unchanged': True, 'prior_rounds_rerun': False}
    write_json(summary, 'run_summary.json')
    write_json({'status': 'passed', 'primary_algorithm': 'max-min Floyd closure of source node yields with target-low nodes masked to -inf', 'independent_algorithm': 'descending finite-edge maximum spanning forest, cross-component capacities assigned at union', 'n_endpoint_rows_checked': verification_rows, 'n_finite_capacities_checked': new_finite_rows, 'maximum_absolute_capacity_error': max_error, 'infinite_capacity_patterns_equal': True, 'F_target20_ge20_equals_existing_union_oracle_ge20_all_rows': True, 'F_target20_finite_equals_connected_target_own_route20_all_rows': True, 'frozen_failure_partition_reconciles_R28_31145': True, 'adjacent_frozen_failures': 0, 'source_manifest': 'source_manifest.json'}, 'independent_verification.json')
    finalize = load_module('r29_finalize', OUT / 'finalize_statistics.py')
    finalize.main()
    print((OUT / 'run_summary.json').read_text(encoding='utf-8'), flush=True)
if __name__ == '__main__':
    main()
