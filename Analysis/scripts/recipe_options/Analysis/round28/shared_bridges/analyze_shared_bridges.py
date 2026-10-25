"""Measured constructive enlargement of the exact six-target route scope."""
from __future__ import annotations
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
SRC = ANALYSIS / 'round27/partner_scope/triple_scope'
TOKENS = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
THRESHOLDS = (0.1, 0.2, 0.3)

def read(name, **kw):
    return pd.read_csv(SRC / name, keep_default_na=False, float_precision='round_trip', **kw)

def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def load():
    groups = read('complete_group_registry.csv.gz')
    contexts = read('complete_context_registry.csv.gz').set_index('context_id')
    conditions = read('full_condition_registry.csv.gz').drop_duplicates(['dataset_id', 'recipe_id'])
    condition_maps = {d: p.set_index('recipe_id') for d, p in conditions.groupby('dataset_id')}
    supports = {}
    for r in read('complete_support_registry.csv.gz').to_dict('records'):
        recipes = json.loads(r['recipe_ids'])
        tok = condition_maps[r['dataset_id']].loc[recipes, TOKENS].to_numpy(str)
        adjacency = (tok[:, None] != tok[None, :]).sum(-1) == 1
        supports[r['support_id']] = {**r, 'recipes': recipes, 'tokens': tok, 'adjacency': adjacency}
    nodes = read('full_source_block_node_yields.csv.gz')
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in nodes.groupby('block_id')}
    return (groups, contexts, conditions, supports, values)

def closure(y, adjacency):
    y = np.asarray(y)
    cap = np.where(adjacency, np.minimum(y[..., :, None], y[..., None, :]), -np.inf)
    ii = np.arange(y.shape[-1])
    cap[..., ii, ii] = y
    for k in range(y.shape[-1]):
        np.maximum(cap, np.minimum(cap[..., :, k, None], cap[..., None, k, :]), out=cap)
    return cap

def masked_capacities(weights, adjacency, masks, starts, goals):
    """Batch equally sized induced graphs; every path is optimized, including detours."""
    unique, inverse = np.unique(np.packbits(masks, axis=1), axis=0, return_inverse=True)
    decoded = np.unpackbits(unique, axis=1)[:, :adjacency.shape[0]].astype(bool)
    sizes = decoded.sum(1)
    result = np.empty((len(starts), len(weights)))
    for size in np.unique(sizes):
        ids = np.flatnonzero(sizes == size)
        ix = np.array([np.flatnonzero(decoded[k]) for k in ids])
        a = adjacency[ix[:, :, None], ix[:, None, :]][:, None]
        yy = weights[:, ix].transpose(1, 0, 2)
        cap = closure(yy, a)
        for z, u in enumerate(ids):
            rows = np.flatnonzero(inverse == u)
            positions = np.full(adjacency.shape[0], -1)
            positions[ix[z]] = np.arange(size)
            result[rows] = cap[z, :, positions[starts[rows]], positions[goals[rows]]]
    return result

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

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    groups, contexts, conditions, supports, values = load()
    cols = ['group_id', 'node0', 'node1', 'minimum_endpoint_yield', 'full_graph_path', 'K_all']
    endpoints = read('complete_group_endpoint_capacities.csv.gz', usecols=cols)
    gids = endpoints.group_id.to_numpy()
    pairdiag = read('complete_group_pair_support_diagnostic.csv.gz').set_index('group_id')
    summaries, candidates = ([], [])
    output_columns = ['group_id', 'node0', 'node1', 'minimum_endpoint_yield', 'full_graph_path', 'K_all', 'endpoint_different_slots', 'n_endpoint_level_nodes', 'n_possible_endpoint_corners', 'complete_endpoint_corners', 'endpoint_graph_path', 'K_endpoint_all', 'K_endpoint_bundle_min', 'K_endpoint_pair_min', 'full_minus_endpoint_gain', 'K_no_fixed_slot_excursion', 'K_no_third_level_in_changed_slots']
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else len(groups)
    prefix = 'pilot_' if limit < len(groups) else ''
    with gzip.open(OUT / (prefix + 'complete_endpoint_bridge_capacities.csv.gz'), 'wt', encoding='utf-8', compresslevel=2, newline='') as stream:
        stream.write(','.join(output_columns) + '\n')
        for g in groups.iloc[:limit].to_dict('records'):
            gid = g['group_id']
            lo, hi = np.searchsorted(gids, [gid, gid + 1])
            ep = endpoints.iloc[lo:hi].copy()
            su = supports[g['support_id']]
            recipes, tok, adjacency = (su['recipes'], su['tokens'], su['adjacency'])
            ids = json.loads(g['context_ids'])
            yy = np.array([[[values[contexts.loc[c, b]][r] for r in recipes] for b in ['block_a', 'block_b']] for c in ids])
            w = yy.min((0, 1))
            if len(ep):
                i, j = (ep.node0.to_numpy(int), ep.node1.to_numpy(int))
                changed = tok[i] != tok[j]
                masks = ((tok[None] == tok[i, None]) | (tok[None] == tok[j, None])).all(2)
                bundles = yy.min(1)
                weights = np.array([w, *bundles, np.minimum(bundles[0], bundles[1]), np.minimum(bundles[0], bundles[2]), np.minimum(bundles[1], bundles[2])])
                k = masked_capacities(weights, adjacency, masks, i, j)
                ep['endpoint_different_slots'] = changed.sum(1)
                ep['n_endpoint_level_nodes'] = masks.sum(1)
                ep['n_possible_endpoint_corners'] = 2 ** changed.sum(1)
                ep['complete_endpoint_corners'] = ep.n_endpoint_level_nodes == ep.n_possible_endpoint_corners
                ep['endpoint_graph_path'] = np.isfinite(k[:, 0])
                ep['K_endpoint_all'] = k[:, 0]
                ep['K_endpoint_bundle_min'] = k[:, 1:4].min(1)
                ep['K_endpoint_pair_min'] = k[:, 4:].min(1)
                finite = np.isfinite(k[:, 0])
                gain = np.full(len(ep), np.nan)
                gain[finite] = ep.K_all.to_numpy()[finite] - k[finite, 0]
                ep['full_minus_endpoint_gain'] = gain
                assert np.all(ep.K_all.to_numpy() + 1e-14 >= k[:, 0])
                assert np.all(k[:, 0] <= k[:, 4:].min(1))
                ep['K_no_fixed_slot_excursion'] = np.nan
                ep['K_no_third_level_in_changed_slots'] = np.nan
                rescue_any = np.zeros(len(ep), bool)
                for h in THRESHOLDS:
                    rescue_any |= (ep.minimum_endpoint_yield >= h) & (ep.K_all >= h) & (ep.K_endpoint_all < h)
                selected = np.flatnonzero(rescue_any)
                if len(selected):
                    match0 = tok[None] == tok[i[selected], None]
                    match1 = tok[None] == tok[j[selected], None]
                    fixed_mask = (match0 | changed[selected, None]).all(2)
                    binary_changed_mask = (match0 | match1 | ~changed[selected, None]).all(2)
                    ep.loc[ep.index[selected], 'K_no_fixed_slot_excursion'] = masked_capacities(w[None], adjacency, fixed_mask, i[selected], j[selected])[:, 0]
                    ep.loc[ep.index[selected], 'K_no_third_level_in_changed_slots'] = masked_capacities(w[None], adjacency, binary_changed_mask, i[selected], j[selected])[:, 0]
                ep[output_columns].to_csv(stream, index=False, header=False, float_format='%.17g')
                ep20 = ep[(ep.minimum_endpoint_yield >= 0.2) & (ep.K_endpoint_bundle_min >= 0.2) & (ep.K_endpoint_all < 0.2) & (ep.K_all >= 0.2)]
                for row in ep20.to_dict('records'):
                    row.update({'focal_key_id': g['focal_key_id'], 'molecular_pair_id': g['molecular_pair_id'], 'dataset_id': g['dataset_id'], 'rxn_class': g['rxn_class'], 'support_id': g['support_id'], 'native_pair_supports_identical': bool(pairdiag.loc[gid, 'all_native_pair_supports_identical_to_full_support'])})
                    row['two_sided_clearance'] = min(row['K_all'] - 0.2, 0.2 - row['K_endpoint_all'], row['K_endpoint_bundle_min'] - 0.2)
                    candidates.append(row)
            for h in THRESHOLDS:
                for complete in ['all', 'complete', 'incomplete']:
                    eligible = ep.minimum_endpoint_yield >= h
                    if len(ep) and complete != 'all':
                        eligible &= ep.complete_endpoint_corners == (complete == 'complete')
                    z = ep[eligible]
                    connected = z[z.full_graph_path == 1]
                    rescue = z[(z.K_all >= h) & (z.K_endpoint_all < h)] if len(z) else z
                    strong = rescue[rescue.K_endpoint_bundle_min >= h] if len(rescue) else rescue
                    metric = {'group_id': gid, 'focal_key_id': g['focal_key_id'], 'dataset_id': g['dataset_id'], 'rxn_class': g['rxn_class'], 'threshold': h, 'endpoint_support_stratum': complete, 'n_high_endpoint_views': len(z), 'n_full_graph_connected_views': len(connected), 'n_endpoint_graph_connected_views': int(z.endpoint_graph_path.sum()) if len(z) else 0, 'n_full_shared_feasible': int((z.K_all >= h).sum()), 'n_endpoint_shared_feasible': int((z.K_endpoint_all >= h).sum()) if len(z) else 0, 'n_rescued_views': len(rescue), 'n_strong_scope_reconciliation': len(strong), 'n_strong_pairwise_endpoint_feasible': int((strong.K_endpoint_pair_min >= h).sum()) if len(strong) else 0, 'n_rescued_endpoint_graph_disconnected': int((~rescue.endpoint_graph_path).sum()) if len(rescue) else 0, 'n_rescued_without_fixed_slot_excursion': int((rescue.K_no_fixed_slot_excursion >= h).sum()) if len(rescue) else 0, 'n_rescued_without_third_level_in_changed_slots': int((rescue.K_no_third_level_in_changed_slots >= h).sum()) if len(rescue) else 0, 'n_rescued_requires_both_types': int(((rescue.K_no_fixed_slot_excursion < h) & (rescue.K_no_third_level_in_changed_slots < h)).sum()) if len(rescue) else 0}
                    summaries.append(metric)
            if (gid + 1) % 1000 == 0:
                print(json.dumps({'groups': gid + 1, 'strong_candidates20': len(candidates), 'seconds': time.time() - started}), flush=True)
    summary = pd.DataFrame(summaries)
    save(summary, prefix + 'complete_group_threshold_summary.csv.gz')
    numeric = [c for c in summary if c.startswith('n_')]
    aggregate = summary.groupby(['threshold', 'endpoint_support_stratum'])[numeric].sum().reset_index()
    save(aggregate, prefix + 'summary.csv')
    inv = read('complete_20_dataset_18_class_support.csv')[['dataset_id', 'rxn_class']].drop_duplicates()
    byclass = summary.groupby(['dataset_id', 'rxn_class', 'threshold', 'endpoint_support_stratum'])[numeric].sum().reset_index()
    for h in THRESHOLDS:
        for s in ['all', 'complete', 'incomplete']:
            missing = inv[~inv.dataset_id.isin(byclass[(byclass.threshold == h) & (byclass.endpoint_support_stratum == s)].dataset_id)].copy()
            for c in numeric:
                missing[c] = 0
            missing['threshold'], missing['endpoint_support_stratum'] = (h, s)
            byclass = pd.concat([byclass, missing], ignore_index=True)
    save(byclass, prefix + 'complete_20_dataset_18_class_summary.csv')
    cand = pd.DataFrame(candidates)
    if len(cand):
        cand = cand.sort_values(['complete_endpoint_corners', 'native_pair_supports_identical', 'two_sided_clearance', 'full_minus_endpoint_gain'], ascending=False)
    save(cand, prefix + 'all_strong_scope_reconciliation20.csv.gz')
    save(groups.iloc[:limit], prefix + 'complete_opportunity_registry.csv.gz')
    dump({'status': 'analyzed', 'n_groups': limit, 'n_endpoint_rows': int(summary[(summary.threshold == 0.1) & (summary.endpoint_support_stratum == 'all')].n_high_endpoint_views.sum()), 'n_strong_candidates20': len(candidates), 'seconds': time.time() - started, 'round27_high10_endpoint_pool_is_sufficient_for_all_six_target_thresholds': True, 'extra_constraint_capacities_calculated_on_union_of_threshold_rescues_only': True}, prefix + 'run_summary.json')
    if len(cand) and (not prefix):
        witness(cand.iloc[0].to_dict(), groups, contexts, conditions, supports, values)
    manifest = {}
    for name in ['complete_group_registry.csv.gz', 'complete_group_endpoint_capacities.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_source_block_node_yields.csv.gz', 'full_condition_registry.csv.gz', 'complete_group_pair_support_diagnostic.csv.gz']:
        p = SRC / name
        with p.open('rb') as f:
            manifest[name] = {'path': str(p), 'sha256': hashlib.file_digest(f, 'sha256').hexdigest()}
    dump(manifest, prefix + 'source_manifest.json')
    print(aggregate.to_string(index=False), flush=True)

def witness(selected, groups, contexts, conditions, supports, values):
    gid = int(selected['group_id'])
    g = groups.set_index('group_id').loc[gid].to_dict()
    su = supports[g['support_id']]
    ids = json.loads(g['context_ids'])
    recipes, tok, adjacency = (su['recipes'], su['tokens'], su['adjacency'])
    yy = np.array([[[values[contexts.loc[c, b]][r] for r in recipes] for b in ['block_a', 'block_b']] for c in ids])
    w = yy.min((0, 1))
    i, j = (int(selected['node0']), int(selected['node1']))
    mask = ((tok == tok[i]) | (tok == tok[j])).all(1)
    endpoint_adj = adjacency & mask[:, None] & mask[None, :]
    objectives = [('full_shared', w, adjacency), ('endpoint_shared', w, endpoint_adj)]
    objectives += [(f'endpoint_partner_{p}', yy[p].min(0), endpoint_adj) for p in range(3)]
    paths = []
    for name, weight, adj in objectives:
        k = closure(weight, adj)[i, j]
        if np.isfinite(k):
            for step, node in enumerate(shortest(weight, adj, i, j, k)):
                paths.append({'objective': name, 'path_step': step, 'node': node, 'recipe_id': recipes[node], 'objective_node_yield': weight[node], 'full_scope_node_yield': w[node], 'optimized_bottleneck': k})
    nodes = [{'partner_position': p, 'context_id': c, 'partner_smiles': contexts.loc[c, 'partner_smiles'], 'node': n, 'recipe_id': r, 'block_a': contexts.loc[c, 'block_a'], 'block_b': contexts.loc[c, 'block_b'], 'yield_a': yy[p, 0, n], 'yield_b': yy[p, 1, n], 'in_endpoint_level_subgraph': bool(mask[n])} for p, c in enumerate(ids) for n, r in enumerate(recipes)]
    cond = conditions[(conditions.dataset_id == g['dataset_id']) & conditions.recipe_id.isin(recipes)]
    save(cond, 'chemical_witness_conditions.csv')
    save(pd.DataFrame(nodes), 'chemical_witness_full_node_yields.csv')
    save(pd.DataFrame(paths), 'chemical_witness_routes.csv')
    save(contexts.loc[ids].reset_index(), 'chemical_witness_acquisition_contexts.csv')
    save(pd.DataFrame([{'node0': a, 'node1': b} for a, b in zip(*np.where(np.triu(adjacency, 1)))]), 'chemical_witness_edges.csv')
    deletions = []
    fullpath = [p['node'] for p in paths if p['objective'] == 'full_shared']
    outside = {(slot, tok[n, slot]) for n in fullpath for slot in range(len(TOKENS)) if tok[n, slot] not in [tok[i, slot], tok[j, slot]]}
    for slot, token in sorted(outside):
        allowed = tok[:, slot] != token
        adj = adjacency & allowed[:, None] & allowed[None, :]
        deletions.append({'slot': TOKENS[slot], 'token': token, 'endpoint_changing_slot': bool(tok[i, slot] != tok[j, slot]), 'K_after_deleting_token': float(closure(w, adj)[i, j])})
    save(pd.DataFrame(deletions), 'chemical_witness_outside_token_deletions.csv')
    meta = contexts.loc[ids[0]]
    dump({**selected, **g, 'group_id': gid, 'molecule_a': meta.molecule_a, 'molecule_b': meta.molecule_b, 'partners': contexts.loc[ids, 'partner_smiles'].tolist(), 'recipe0': recipes[i], 'recipe1': recipes[j], 'selection': 'Outcome-selected existence witness: complete corners, equal native pair supports, maximum two-sided20% clearance, then continuous gain.', 'n_source_blocks': 6, 'source_readback_status': 'pending'}, 'chemical_witness.json')
if __name__ == '__main__':
    main()
