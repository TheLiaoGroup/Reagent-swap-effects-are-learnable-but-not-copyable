"""Shared-route compromise for complete registered two-context opportunities."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
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
PREVIOUS = ANALYSIS / 'round26/support_fixed_routes'
TOKENS = [*[f'Reagent{k}_token' for k in range(1, 7)], 'Solvent_token', 'Temperature_token', 'Time_token']
THRESHOLDS = [0.1, 0.2, 0.3]
SOURCES = {'opportunities': PREVIOUS / 'complete_comparison_registry.csv.gz', 'supports': PREVIOUS / 'complete_support_graph_registry.csv.gz', 'recipe_numbers': PREVIOUS / 'recipe_number_registry.csv', 'nodes': ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_recipe_node_means.csv.gz', 'conditions': ANALYSIS / 'data/condition_slot_registry.csv', 'parent_inventory': PREVIOUS / 'full_parent_dataset_class_inventory.csv'}

def save(frame, name):
    frame.to_csv(OUT / name, index=False, float_format='%.17g', encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def widest(y, adjacency):
    capacities = np.where(adjacency, np.minimum(y[:, None], y[None, :]), -np.inf)
    np.fill_diagonal(capacities, y)
    for k in range(len(y)):
        np.maximum(capacities, np.minimum(capacities[:, k, None], capacities[None, k, :]), out=capacities)
    return capacities

def load():
    inv = pd.read_csv(SOURCES['opportunities'], keep_default_na=False)
    sup = pd.read_csv(SOURCES['supports'], keep_default_na=False).to_dict('records')
    assert len(inv) == 233260 and int(inv.identical_anchor_blocks.sum()) == 133002
    conditions = pd.read_csv(SOURCES['conditions'], keep_default_na=False).drop_duplicates(['dataset_id', 'recipe_id'])
    tokens = {(r.dataset_id, r.recipe_id): np.array([getattr(r, c) for c in TOKENS]) for r in conditions.itertuples()}
    nodes = pd.read_csv(SOURCES['nodes'])
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in nodes.groupby('block_id', sort=False)}
    for s in sup:
        s['recipes'] = tuple(json.loads(s['recipe_ids']))
    numbers = pd.read_csv(SOURCES['recipe_numbers']).set_index('recipe_id').recipe_number.to_dict()
    return (inv, sup, tokens, values, numbers)

def main():
    start = time.time()
    inv, supports, tokens, values, numbers = load()

    @lru_cache(maxsize=512)
    def graph(sno):
        s = supports[sno]
        tok = np.array([tokens[s['dataset_id'], r] for r in s['recipes']])
        adj = (tok[:, None, :] != tok[None, :, :]).sum(-1) == 1
        assert int(adj.sum() // 2) == s['n_edges']
        return adj

    @lru_cache(maxsize=1024)
    def context(blocks_text, sno):
        s = supports[sno]
        blocks = json.loads(blocks_text)
        y = np.array([[values[b][r] for r in s['recipes']] for b in blocks])
        f = y.min(0)
        return (y, f, widest(f, graph(sno)))
    stats = []
    selected = None
    n_export = 0
    columns = ['comparison_id', 'endpoint0_number', 'endpoint1_number', 'minimum_endpoint_yield', 'full_graph_connected', 'B_context_a', 'B_context_b', 'B_union', 'compromise', 'both_contexts_feasible20', 'scope_failure20']
    with gzip.open(OUT / 'complete_primary20_endpoint_views.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=2) as stream:
        stream.write(','.join(columns) + '\n')
        for row in inv.to_dict('records'):
            no, sno = (int(row['comparison_id']), int(row['fixed_support_number']))
            recipes = supports[sno]['recipes']
            if len(recipes) < 2 or not row['n_endpoint_pool10']:
                for threshold in THRESHOLDS:
                    stats.append(dict(comparison_id=no, threshold=threshold, n_high_endpoint_views=0, n_connected_views=0, n_both_contexts_feasible=0, n_scope_failures=0, n_scope_failures_clearance5pp=0, n_compromise_ge5pp=0, sum_compromise=0.0, max_compromise=0.0, sum_compromise_both_feasible=0.0, n_equal_context_optima1pp=0, n_equal_optima_scope_failures=0))
                continue
            ya, f, ca = context(row['aligned_blocks_a'], sno)
            yb, g, cb = context(row['aligned_blocks_b'], sno)
            if row['identical_anchor_blocks']:
                assert np.array_equal(ya[:2], yb[:2])
            allmin = np.minimum(f, g)
            high = np.flatnonzero(allmin >= 0.1)
            ii, jj = np.triu_indices(len(high), 1)
            i, j = (high[ii], high[jj])
            endpoint = np.minimum(allmin[i], allmin[j])
            cu = widest(allmin, graph(sno))
            a, b, union = (ca[i, j], cb[i, j], cu[i, j])
            connected = np.isfinite(union)
            assert np.array_equal(connected, np.isfinite(a) & np.isfinite(b))
            gap = np.zeros(len(i))
            gap[connected] = np.minimum(a[connected], b[connected]) - union[connected]
            assert (gap >= -1e-12).all()
            gap = np.maximum(gap, 0)
            for threshold in THRESHOLDS:
                eligible = endpoint >= threshold
                full = eligible & connected
                both = full & (a >= threshold) & (b >= threshold)
                failure = both & (union < threshold)
                clearance = both & (a >= threshold + 0.05) & (b >= threshold + 0.05) & (union <= threshold - 0.05)
                difference = np.subtract(a, b, where=connected, out=np.full_like(a, np.inf))
                equal = both & (np.abs(difference) <= 0.01)
                stats.append(dict(comparison_id=no, threshold=threshold, n_high_endpoint_views=int(eligible.sum()), n_connected_views=int(full.sum()), n_both_contexts_feasible=int(both.sum()), n_scope_failures=int(failure.sum()), n_scope_failures_clearance5pp=int(clearance.sum()), n_compromise_ge5pp=int((both & (gap >= 0.05)).sum()), sum_compromise=float(gap[full].sum()), max_compromise=float(gap[full].max(initial=0)), sum_compromise_both_feasible=float(gap[both].sum()), n_equal_context_optima1pp=int(equal.sum()), n_equal_optima_scope_failures=int((equal & failure).sum())))
                if threshold == 0.2:
                    records = np.column_stack([np.full(eligible.sum(), no), [numbers[recipes[k]] for k in i[eligible]], [numbers[recipes[k]] for k in j[eligible]], endpoint[eligible], connected[eligible], a[eligible], b[eligible], union[eligible], np.where(connected[eligible], gap[eligible], np.nan), both[eligible], failure[eligible]])
                    np.savetxt(stream, records, delimiter=',', fmt=['%d', '%d', '%d', '%.17g', '%d', '%.17g', '%.17g', '%.17g', '%.17g', '%d', '%d'])
                    n_export += len(records)
                    if row['identical_anchor_blocks']:
                        for at in np.flatnonzero(failure):
                            margin = min(a[at] - 0.2, b[at] - 0.2, 0.2 - union[at])
                            rank = (margin, gap[at], -len(recipes), -no, -int(i[at]), -int(j[at]))
                            if selected is None or rank > selected[0]:
                                selected = (rank, dict(**row, endpoint0_index=int(i[at]), endpoint1_index=int(j[at]), endpoint0_number=numbers[recipes[i[at]]], endpoint1_number=numbers[recipes[j[at]]], B_context_a=float(a[at]), B_context_b=float(b[at]), B_union=float(union[at]), compromise=float(gap[at]), clearance=float(margin)))
            if no % 10000 == 0:
                print(json.dumps(dict(comparison_id=no, elapsed_seconds=round(time.time() - start, 1), exported=n_export)), flush=True)
    tab = pd.DataFrame(stats)
    save(tab, 'complete_comparison_threshold_statistics.csv.gz')
    save(inv, 'complete_opportunity_registry.csv.gz')
    meta = inv[['comparison_id', 'identical_anchor_blocks', 'rxn_class', 'dataset_id', 'focal_pair_id', 'anchor_group_number', 'quartet_number_a', 'quartet_number_b']]
    merged = tab.merge(meta, on='comparison_id', validate='many_to_one')
    summaries = []
    for stratum in ['all_anchor_identities', 'identical_anchor_blocks']:
        part = merged if stratum == 'all_anchor_identities' else merged[merged.identical_anchor_blocks]
        for threshold, tt in part.groupby('threshold', sort=True):
            groups = [('all', 'all', tt)]
            groups += [(dimension, str(key), group) for dimension in ['rxn_class', 'dataset_id'] for key, group in tt.groupby(dimension, sort=True)]
            for dim, key, group in groups:
                output = dict(stratum=stratum, threshold=threshold, dimension=dim, value=key, n_opportunities=len(group))
                for col in ['n_high_endpoint_views', 'n_connected_views', 'n_both_contexts_feasible', 'n_scope_failures', 'n_scope_failures_clearance5pp', 'n_compromise_ge5pp', 'sum_compromise', 'sum_compromise_both_feasible', 'n_equal_context_optima1pp', 'n_equal_optima_scope_failures']:
                    output[col] = group[col].sum()
                output['max_compromise'] = group.max_compromise.max()
                for label, mask in [('evaluable', group.n_connected_views.gt(0)), ('both_feasible', group.n_both_contexts_feasible.gt(0)), ('failure', group.n_scope_failures.gt(0)), ('clearance5pp', group.n_scope_failures_clearance5pp.gt(0))]:
                    gg = group[mask]
                    output[f'{label}_comparisons'] = len(gg)
                    output[f'{label}_anchor_groups'] = gg.anchor_group_number.nunique()
                    output[f'{label}_focal_pairs'] = gg.focal_pair_id.nunique()
                    output[f'{label}_quartets'] = len(set(gg.quartet_number_a) | set(gg.quartet_number_b))
                valid = group[group.n_both_contexts_feasible.gt(0)].copy()
                valid['failure_fraction'] = valid.n_scope_failures / valid.n_both_contexts_feasible
                output['comparison_equal_mean_failure_fraction'] = valid.failure_fraction.mean()
                output['anchor_equal_mean_failure_fraction'] = valid.groupby('anchor_group_number').failure_fraction.mean().mean()
                output['focal_pair_equal_mean_failure_fraction'] = valid.groupby('focal_pair_id').failure_fraction.mean().mean()
                summaries.append(output)
    save(pd.DataFrame(summaries), 'summary.csv')
    inventory = pd.read_csv(SOURCES['parent_inventory'])
    new = inv.groupby(['dataset_id', 'rxn_class'], as_index=False).agg(n_cross_context_opportunities=('comparison_id', 'size'))
    save(inventory.merge(new, on=['dataset_id', 'rxn_class'], how='left').fillna({'n_cross_context_opportunities': 0}), 'complete_20_dataset_inventory.csv')
    write_json('witness_selection.json', selected[1] if selected else {'status': 'no_primary20_failure'})
    write_json('source_manifest.json', {k: dict(path=str(p), sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()) for k, p in SOURCES.items()})
    write_json('run_summary.json', dict(n_opportunities=len(inv), n_primary20_endpoint_views=n_export, n_threshold_comparison_rows=len(tab), elapsed_seconds=time.time() - start))
    print((OUT / 'run_summary.json').read_text(), flush=True)
if __name__ == '__main__':
    main()
