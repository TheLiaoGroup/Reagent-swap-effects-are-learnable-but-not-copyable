"""Frozen source-partner route choice on every R27 coherent triple.

Routes are chosen on the availability-defined whole-triple graph. Target yields
enter evaluation only, after source policies and endpoint membership are fixed."""
from __future__ import annotations
from collections import Counter, deque
from itertools import combinations
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
SOURCE = ANALYSIS / 'round27/partner_scope/triple_scope'
H = 0.2
POLICIES = ('source_maximin', 'source_mean_bottleneck', 'full_graph_shortest')
TOKENS = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']

def read(name, **kwargs):
    return pd.read_csv(SOURCE / name, keep_default_na=False, float_precision='round_trip', low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def json_write(value, name):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def pack(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)

def load():
    groups = read('complete_group_registry_before_outcomes.csv.gz').to_dict('records')
    focal = read('complete_focal_key_registry.csv.gz')
    contexts = read('complete_context_registry.csv.gz').set_index('context_id').to_dict('index')
    nodes = read('full_source_block_node_yields.csv.gz')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in nodes.groupby('block_id', sort=False)}
    supports = read('complete_support_registry.csv.gz').to_dict('records')
    edges = read('full_support_edges.csv.gz')
    edge_lookup = {int(k): p[['node0', 'node1']].to_numpy(int) for k, p in edges.groupby('support_id', sort=False)}
    node_table = read('full_support_nodes.csv.gz')
    label_lookup = {int(k): p.sort_values('node').full_graph_component.to_numpy(int) for k, p in node_table.groupby('support_id', sort=False)}
    for su in supports:
        sid = int(su['support_id'])
        su['recipes'] = json.loads(su['recipe_ids'])
        n = len(su['recipes'])
        adj = np.zeros((n, n), bool)
        if sid in edge_lookup:
            a, b = edge_lookup[sid].T
            adj[a, b] = adj[b, a] = True
        su['adjacency'] = adj
        su['neighbors'] = [np.flatnonzero(row).tolist() for row in adj]
        su['labels'] = label_lookup.get(sid, np.empty(0, int))
        assert su['recipes'] == sorted(su['recipes'])
    assert len(groups) == 44334 and len(focal) == 512
    return (groups, focal, contexts, values, supports)

def yield_array(ids, sid, contexts, values, supports):
    recipes = supports[sid]['recipes']
    return np.array([[[values[contexts[c]['block_' + side]][r] for r in recipes] for side in ['a', 'b']] for c in ids])

def widest(y, adj):
    cap = np.where(adj, np.minimum(y[:, None], y[None, :]), -np.inf)
    np.fill_diagonal(cap, y)
    for k in range(len(y)):
        np.maximum(cap, np.minimum(cap[:, k, None], cap[None, k, :]), out=cap)
    return cap

def census(groups, contexts, values, supports):
    started = time.time()
    rows = []
    for g in groups:
        ids, sid = (json.loads(g['context_ids']), int(g['support_id']))
        su = supports[sid]
        y = yield_array(ids, sid, contexts, values, supports)
        for holdout in range(3):
            source_ids = ids[:holdout] + ids[holdout + 1:]
            sy = y[[i for i in range(3) if i != holdout]].min((0, 1))
            high = np.flatnonzero(sy >= H)
            u, v = np.triu_indices(len(high), 1)
            u, v = (high[u], high[v])
            conn = su['labels'][u] == su['labels'][v]
            status = 'no_common_recipe' if su['n_nodes'] == 0 else 'only_one_common_recipe' if su['n_nodes'] == 1 else 'no_two_source_high_endpoints' if len(high) < 2 else 'all_source_high_pairs_disconnected' if not conn.any() else 'has_connected_source_high_pairs'
            rows.append({k: g[k] for k in ['group_id', 'focal_key_id', 'dataset_id', 'rxn_class', 'molecular_pair_id', 'edit_id', 'support_id', 'n_nodes', 'n_edges']} | {'heldout_position': holdout, 'heldout_context_id': ids[holdout], 'source_context_ids': pack(source_ids), 'n_source_high_nodes': len(high), 'n_source_endpoint_pairs': len(u), 'n_connected_source_endpoint_pairs': int(conn.sum()), 'support_status': status})
    reg = pd.DataFrame(rows)
    save(reg, 'complete_heldout_registry_before_target_outcomes.csv.gz')
    design = {'frame': 'all_R27_coherent_triples_three_leave_one_partner_out_evaluations', 'threshold': H, 'n_parent_focal_keys': 512, 'n_triples': len(groups), 'n_holdouts': len(reg), 'n_supports': len(supports), 'n_source_defined_endpoint_pairs': int(reg.n_source_endpoint_pairs.sum()), 'n_full_graph_connected_source_pairs': int(reg.n_connected_source_endpoint_pairs.sum()), 'holdout_support_status': dict(Counter(reg.support_status)), 'census_seconds': time.time() - started, 'source_only_rule': 'endpoint minimum across four source targets >=0.20; graph support fixed for all six targets by availability', 'policies': list(POLICIES), 'tie_break': 'maximum node-objective bottleneck, then minimum edges, then lexicographic recipe-ID route from lower endpoint recipe-ID', 'memory_strategy': 'per-group matrices and BFS trees; no route or all-pairs cache across groups', 'output_strategy': 'one compact endpoint row containing all three policies, all groups retained in complete registry'}
    json_write(design, 'design_census.json')
    print('CENSUS', pack(design), flush=True)
    return (reg, design)

def bfs_tree(neighbors, start, allowed, sy, ty):
    n = len(neighbors)
    previous = np.full(n, -1, np.int16)
    distance = np.full(n, -1, np.int16)
    count = np.zeros(n, np.int8)
    source_min = np.full(n, np.nan)
    target_min = np.full(n, np.nan)
    previous[start] = start
    distance[start] = 0
    count[start] = 1
    source_min[start], target_min[start] = (sy[start], ty[start])
    queue = [start]
    for cur in queue:
        for nxt in neighbors[cur]:
            if not allowed[nxt]:
                continue
            if distance[nxt] < 0:
                previous[nxt] = cur
                distance[nxt] = distance[cur] + 1
                count[nxt] = count[cur]
                source_min[nxt] = min(source_min[cur], sy[nxt])
                target_min[nxt] = min(target_min[cur], ty[nxt])
                queue.append(nxt)
            elif distance[nxt] == distance[cur] + 1:
                count[nxt] = min(2, count[nxt] + count[cur])
    return (previous, distance, count, source_min, target_min)

def policy_results(objective, capacities, su, u, v, sy, ty):
    """Source-only BFS trees for each distinct optimal threshold and start."""
    m = len(u)
    result = {k: np.full(m, np.nan) for k in ['source', 'target', 'joint']}
    result['edges'] = np.full(m, -1, int)
    result['shortest_tie'] = np.zeros(m, bool)
    for start in np.unique(u):
        positions = np.flatnonzero(u == start)
        floors = capacities[start, v[positions]] if capacities is not None else np.where(su['labels'][start] == su['labels'][v[positions]], 0.0, -np.inf)
        for floor in np.unique(floors):
            if not np.isfinite(floor):
                continue
            at = positions[floors == floor]
            allowed = objective >= floor if capacities is not None else np.ones(len(sy), bool)
            _, dist, count, sm, tm = bfs_tree(su['neighbors'], int(start), allowed, sy, ty)
            goals = v[at]
            assert (dist[goals] >= 0).all()
            result['source'][at], result['target'][at] = (sm[goals], tm[goals])
            result['joint'][at] = np.minimum(sm[goals], tm[goals])
            result['edges'][at], result['shortest_tie'][at] = (dist[goals], count[goals] > 1)
    return result

def main():
    started = time.time()
    groups, focal, contexts, values, supports = load()
    reg, design = census(groups, contexts, values, supports)
    if '--census-only' in sys.argv:
        return
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else len(groups)
    header = ['group_id', 'heldout_position', 'node0', 'node1', 'full_graph_connected', 'source_optimum', 'source_endpoint_minimum', 'target_endpoint_minimum', 'target_optimum', 'union_oracle', 'source_route_feasible20', 'target_endpoint_failure20', 'target_own_internal_failure20', 'shared_route_impossible20', 'source_target_separately_feasible_shared_impossible20']
    for p in POLICIES:
        header += [p + '_' + k for k in ['source', 'target', 'joint', 'union_gap', 'edges', 'shortest_tie', 'avoidable_selection_failure20', 'target_internal_failure_despite_own_route20']]
    aggregate = []
    witness_candidates = []
    filename = 'pilot_endpoint_results.csv.gz' if limit < len(groups) else 'complete_endpoint_results.csv.gz'
    with gzip.open(OUT / filename, 'wt', encoding='utf-8', newline='', compresslevel=2) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for no, g in enumerate(groups[:limit]):
            ids, sid = (json.loads(g['context_ids']), int(g['support_id']))
            su = supports[sid]
            y = yield_array(ids, sid, contexts, values, supports)
            n = su['n_nodes']
            union = y.min((0, 1))
            union_cap = widest(union, su['adjacency']) if n >= 2 else np.empty((n, n))
            for holdout in range(3):
                row = reg.iloc[3 * no + holdout].to_dict()
                sy4 = y[[i for i in range(3) if i != holdout]]
                sy, my, ty = (sy4.min((0, 1)), sy4.mean((0, 1)), y[holdout].min(0))
                high = np.flatnonzero(sy >= H)
                a, b = np.triu_indices(len(high), 1)
                u, v = (high[a], high[b])
                if not len(u):
                    aggregate.append(row)
                    continue
                sc = widest(sy, su['adjacency'])
                tc = widest(ty, su['adjacency'])
                mc = widest(my, su['adjacency'])
                conn = su['labels'][u] == su['labels'][v]
                src, tgt, oracle = (sc[u, v], tc[u, v], union_cap[u, v])
                sep, tep = (np.minimum(sy[u], sy[v]), np.minimum(ty[u], ty[v]))
                source_ok = conn & (src >= H)
                endpoint_fail = conn & (tep < H)
                own_internal_fail = conn & (tep >= H) & (tgt < H)
                shared_impossible = source_ok & (tep >= H) & (oracle < H)
                separate_impossible = source_ok & (tgt >= H) & (oracle < H)
                result = {'source_maximin': policy_results(sy, sc, su, u, v, sy, ty), 'source_mean_bottleneck': policy_results(my, mc, su, u, v, sy, ty), 'full_graph_shortest': policy_results(None, None, su, u, v, sy, ty)}
                strata = {'connected': conn, 'source_feasible': source_ok, 'source_feasible_target_endpoints_high': source_ok & (tep >= H), 'source_and_target_separately_feasible': source_ok & (tgt >= H), 'union_feasible': conn & (oracle >= H)}
                row.update(n_source_feasible_target_endpoint_failures20=int((source_ok & (tep < H)).sum()), n_source_feasible_target_own_internal_failures20=int((source_ok & (tep >= H) & (tgt < H)).sum()))
                cols = [np.full(len(u), int(g['group_id'])), np.full(len(u), holdout), u, v, conn, src, sep, tep, tgt, oracle, source_ok, endpoint_fail, own_internal_fail, shared_impossible, separate_impossible]
                row.update(n_source_route_feasible20=int(source_ok.sum()), n_source_route_failed20=int((conn & ~source_ok).sum()), n_target_endpoint_failures20=int(endpoint_fail.sum()), n_target_own_internal_failures20=int(own_internal_fail.sum()), n_target_own_route_feasible20=int((conn & (tgt >= H)).sum()), n_union_route_feasible20=int((conn & (oracle >= H)).sum()), n_shared_route_impossible20=int(shared_impossible.sum()), n_separately_feasible_shared_impossible20=int(separate_impossible.sum()), union_oracle_sum=float(oracle[conn].sum()), target_optimum_sum=float(tgt[conn].sum()), source_optimum_sum=float(src[conn].sum()))
                for policy, rr in result.items():
                    gap = np.where(conn, oracle - rr['joint'], np.nan)
                    avoidable = conn & (oracle >= H) & (rr['joint'] < H)
                    target_internal = conn & (tgt >= H) & (rr['target'] < H)
                    cols += [rr['source'], rr['target'], rr['joint'], gap, rr['edges'], rr['shortest_tie'], avoidable, target_internal]
                    for metric in ['source', 'target', 'joint']:
                        row[policy + '_' + metric + '_sum'] = float(rr[metric][conn].sum())
                    row[policy + '_union_gap_sum'] = float(gap[conn].sum())
                    row[policy + '_n_success20'] = int((conn & (rr['joint'] >= H)).sum())
                    row[policy + '_n_source_failure20'] = int((conn & (rr['source'] < H)).sum())
                    row[policy + '_n_avoidable_selection_failure20'] = int(avoidable.sum())
                    row[policy + '_n_target_internal_failure_despite_own_route20'] = int(target_internal.sum())
                    row[policy + '_n_shortest_tie'] = int(rr['shortest_tie'].sum())
                    row[policy + '_edges_sum'] = int(rr['edges'][conn].sum())
                    for stratum, mask in strata.items():
                        prefix = stratum + '__'
                        row[prefix + 'denominator_sum'] = int(mask.sum())
                        for field, vv in [('joint', rr['joint']), ('union_gap', gap), ('success20', rr['joint'] >= H), ('source_failure20', rr['source'] < H), ('avoidable_selection_failure20', avoidable), ('target_internal_failure_despite_own_route20', target_internal)]:
                            row[prefix + policy + '_' + field + '_sum'] = float(vv[mask].sum())
                writer.writerows(zip(*cols))
                aggregate.append(row)
                primary = result['source_maximin']
                eligible = np.flatnonzero(source_ok & (tgt >= H) & (oracle >= H) & (primary['joint'] < H))
                if len(eligible):
                    z = int(eligible[np.argmax(oracle[eligible] - primary['joint'][eligible])])
                    witness_candidates.append({'group_id': int(g['group_id']), 'heldout_position': holdout, 'node0': int(u[z]), 'node1': int(v[z]), 'union_oracle': float(oracle[z]), 'achieved_joint': float(primary['joint'][z]), 'gap': float(oracle[z] - primary['joint'][z]), 'source_optimum': float(src[z]), 'target_optimum': float(tgt[z]), 'target_endpoint_minimum': float(tep[z])})
            if (no + 1) % 1000 == 0:
                print('PROGRESS', no + 1, 'groups', round(time.time() - started, 1), 'seconds', flush=True)
    ag = pd.DataFrame(aggregate)
    numeric = [c for c in ag if c.startswith('n_') or c.endswith('_sum') or '_n_' in c]
    ag[numeric] = ag[numeric].fillna(0)
    save(ag, 'pilot_heldout_statistics.csv.gz' if limit < len(groups) else 'complete_heldout_statistics.csv.gz')
    save(pd.DataFrame(witness_candidates).sort_values('gap', ascending=False) if witness_candidates else pd.DataFrame(), 'pilot_witness_candidates.csv' if limit < len(groups) else 'witness_candidates.csv')
    if limit < len(groups):
        print('PILOT', pack({'groups': limit, 'seconds': time.time() - started, 'endpoint_pairs': int(ag.n_source_endpoint_pairs.sum())}), flush=True)
        return
    summarize(ag, focal)
    source_files = ['complete_group_registry_before_outcomes.csv.gz', 'complete_focal_key_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_support_edges.csv.gz', 'full_support_nodes.csv.gz', 'full_source_block_node_yields.csv.gz']
    json_write([{'path': str(SOURCE / f), 'sha256': hashlib.sha256((SOURCE / f).read_bytes()).hexdigest()} for f in source_files], 'source_manifest.json')
    json_write({'status': 'completed', 'seconds': time.time() - started, 'endpoint_rows': int(ag.n_source_endpoint_pairs.sum()), 'connected_endpoint_rows': int(ag.n_connected_source_endpoint_pairs.sum()), 'n_heldouts': len(ag), 'n_triples': len(groups), 'no_target_for_selection': True, 'threshold': H, 'policies': list(POLICIES)}, 'run_summary.json')
    print('COMPLETE', pack(json.loads((OUT / 'run_summary.json').read_text())), flush=True)

def summarize(ag, focal):
    measures = [c for c in ag if c.endswith('_sum') or c.startswith('n_') or '_n_' in c]
    measures = [c for c in measures if c not in {'n_nodes', 'n_edges', 'n_source_high_nodes'}]
    key = ag.groupby('focal_key_id', as_index=False)[measures].sum()
    key = focal.merge(key, on='focal_key_id', how='left', suffixes=('', '_outcome'))
    key[measures] = key[measures].fillna(0)
    save(key, 'complete_470_focal_key_statistics.csv.gz')
    inventory = read('complete_20_dataset_18_class_support.csv')
    dataset = ag.groupby(['dataset_id', 'rxn_class'], as_index=False)[measures].sum()
    dataset = inventory[['dataset_id', 'rxn_class']].merge(dataset, how='left', on=['dataset_id', 'rxn_class'])
    dataset[measures] = dataset[measures].fillna(0)
    save(dataset, 'complete_20_dataset_statistics.csv')
    classes = dataset.groupby('rxn_class', as_index=False)[measures].sum()
    assert len(dataset) == 20 and len(classes) == 18
    save(classes, 'complete_18_class_statistics.csv')
    summary = []
    for unit, frame in [('endpoint', ag), ('heldout', ag), ('focal_key', key), ('dataset', dataset), ('reaction_class', classes)]:
        for stratum in ['connected', 'source_feasible', 'source_feasible_target_endpoints_high', 'source_and_target_separately_feasible', 'union_feasible']:
            prefix = stratum + '__'
            denominator = frame[prefix + 'denominator_sum'].to_numpy(float)
            active = denominator > 0
            for p in POLICIES:
                for metric in ['joint', 'union_gap', 'success20', 'source_failure20', 'avoidable_selection_failure20', 'target_internal_failure_despite_own_route20']:
                    field = prefix + p + '_' + metric + '_sum'
                    value = float(frame[field].sum() / denominator.sum()) if unit == 'endpoint' else float((frame.loc[active, field] / denominator[active]).mean())
                    summary.append({'stratum': stratum, 'weighting_unit': unit, 'policy': p, 'metric': metric, 'value': value, 'n_active_units': int(active.sum()), 'n_all_units': len(frame), 'endpoint_denominator': int(denominator.sum()), 'missing_unit_rule': 'units with zero eligible endpoints retained in inventory, excluded from undefined conditional mean'})
    save(pd.DataFrame(summary), 'continuous_and_threshold_weighting_summary.csv')
    json_write({'pooled_complete_counts': {m: float(ag[m].sum()) for m in measures}, 'conditional_mean_rule': 'zero-denominator units have undefined conditional metric, not imputed zero'}, 'pooled_counts.json')
if __name__ == '__main__':
    main()
