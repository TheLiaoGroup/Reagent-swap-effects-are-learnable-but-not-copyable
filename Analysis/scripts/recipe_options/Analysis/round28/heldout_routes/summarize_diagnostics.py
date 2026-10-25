"""Continuous intrinsic/selection decomposition and endpoint-topology diagnostics."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
POLICIES = ['source_maximin', 'source_mean_bottleneck', 'full_graph_shortest']

def read(name, **kwargs):
    return pd.read_csv(OUT / name, float_precision='round_trip', low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def main():
    started = time.time()
    registry = read('complete_heldout_statistics.csv.gz')
    registry['heldout_id'] = registry.group_id * 3 + registry.heldout_position
    registry = registry.set_index('heldout_id')
    np.testing.assert_array_equal(registry.index.to_numpy(), np.arange(len(registry)))
    fields = ['n_endpoints', 'intrinsic_gap_sum', 'separate_optima_ceiling_sum', 'union_oracle_sum', 'target_endpoint_failure20_count', 'target_own_internal_failure20_count', 'separate_feasible_shared_impossible20_count']
    for policy in POLICIES:
        fields += [policy + '_' + m for m in ['joint_sum', 'selection_gap_sum', 'success20_count', 'source_failure20_count', 'avoidable_selection_failure20_count', 'shortest_tie_count']]
    strata = ['all_connected_source_selected', 'source_route_feasible20', 'union_feasible20']
    topologies = ['all', 'adjacent', 'nonadjacent']
    aggregate = np.zeros((len(registry), len(strata), len(topologies), len(fields)))
    rows = 0
    for chunk in read('complete_endpoint_results.csv.gz', chunksize=200000):
        chunk = chunk[chunk.full_graph_connected].copy()
        u = (chunk.group_id * 3 + chunk.heldout_position).to_numpy(int)
        oracle = chunk.union_oracle.to_numpy(float)
        ceiling = np.minimum(chunk.source_optimum, chunk.target_optimum).to_numpy(float)
        vv = [np.ones(len(chunk)), ceiling - oracle, ceiling, oracle, chunk.target_endpoint_failure20.to_numpy(int), chunk.target_own_internal_failure20.to_numpy(int), chunk.source_target_separately_feasible_shared_impossible20.to_numpy(int)]
        for policy in POLICIES:
            joint = chunk[policy + '_joint'].to_numpy(float)
            vv += [joint, oracle - joint, joint >= 0.2, chunk[policy + '_source'].to_numpy(float) < 0.2, chunk[policy + '_avoidable_selection_failure20'].to_numpy(int), chunk[policy + '_shortest_tie'].to_numpy(int)]
        data = np.array(vv).T
        assert (data[:, 1] >= -1e-14).all()
        adjacent = chunk.full_graph_shortest_edges.to_numpy(int) == 1
        for i, eligible in enumerate([np.ones(len(chunk), bool), chunk.source_route_feasible20.to_numpy(bool), oracle >= 0.2]):
            for j, topology in enumerate([np.ones(len(chunk), bool), adjacent, ~adjacent]):
                mask = eligible & topology
                if mask.any():
                    np.add.at(aggregate[:, i, j, :], u[mask], data[mask])
        rows += len(chunk)
        if rows and rows % 2000000 < 200000:
            print('DIAGNOSTICS', rows, 'connected rows', round(time.time() - started, 1), 'seconds', flush=True)
    base = registry[['group_id', 'heldout_position', 'focal_key_id', 'dataset_id', 'rxn_class', 'molecular_pair_id']]
    summary, class_tables, key_tables, heldout_tables = ([], [], [], [])
    for i, stratum in enumerate(strata):
        for j, topology in enumerate(topologies):
            active = aggregate[:, i, j, 0] > 0
            frame = base.loc[active].reset_index(drop=True).join(pd.DataFrame(aggregate[active, i, j, :], columns=fields))
            frame['stratum'], frame['endpoint_topology'] = (stratum, topology)
            heldout_tables.append(frame)
            for unit in ['endpoint', 'heldout', 'focal_key', 'dataset', 'reaction_class']:
                if unit in ['endpoint', 'heldout']:
                    mm = frame[fields]
                else:
                    key = {'focal_key': 'focal_key_id', 'dataset': 'dataset_id', 'reaction_class': 'rxn_class'}[unit]
                    grouped = frame.groupby(key, as_index=False)[fields].sum()
                    mm = grouped[fields]
                    grouped['stratum'], grouped['endpoint_topology'] = (stratum, topology)
                    if unit == 'focal_key':
                        key_tables.append(grouped)
                    if unit == 'reaction_class':
                        class_tables.append(grouped)
                for field in fields[1:]:
                    metric = float(mm[field].sum() / mm.n_endpoints.sum()) if unit == 'endpoint' else float((mm[field] / mm.n_endpoints).mean())
                    summary.append({'stratum': stratum, 'endpoint_topology': topology, 'weighting_unit': unit, 'metric': field, 'value': metric, 'n_active_units': len(mm), 'n_endpoint_views': int(mm.n_endpoints.sum()), 'within_unit_rule': 'endpoint mean within unit; equal mean across active units; zero-denominator units undefined and excluded'})
    save(pd.DataFrame(summary), 'intrinsic_selection_topology_weighting_summary.csv')
    save(pd.concat(class_tables, ignore_index=True), 'intrinsic_selection_topology_class_diagnostics.csv')
    save(pd.concat(key_tables, ignore_index=True), 'intrinsic_selection_topology_focal_key_diagnostics.csv.gz')
    save(pd.concat(heldout_tables, ignore_index=True), 'intrinsic_selection_topology_heldout_diagnostics.csv.gz')
    pooled = registry.n_connected_source_endpoint_pairs.sum()
    assert rows == pooled
    assert abs(aggregate[:, 0, 1, fields.index('source_maximin_selection_gap_sum')].sum()) < 1e-10
    result = {'status': 'passed', 'seconds': time.time() - started, 'n_connected_endpoint_rows': rows, 'decomposition': 'min(K_source,K_target)-J_policy = [min(K_source,K_target)-K_union] + [K_union-J_policy]', 'intrinsic_gap': 'separate source and target optima ceiling minus full-six-target union oracle; all routes on same fixed measured graph', 'selection_gap': 'full-six-target union oracle minus achieved policy joint bottleneck; no claim target can be used during selection', 'threshold_intrinsic_incompatibility': 'source and target each have own >=20% route, union oracle <20%; target-own failures excluded', 'adjacent_zero_source_maximin_selection_gap_check': True, 'conditioning': 'target/oracle-defined strata are retrospective diagnostics; source-only endpoints and policies frozen before these strata are applied'}
    (OUT / 'diagnostic_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
