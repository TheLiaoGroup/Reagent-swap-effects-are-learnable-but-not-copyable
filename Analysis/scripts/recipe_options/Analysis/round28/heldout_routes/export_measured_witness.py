"""Measured explanatory example; all source policies remain frozen.

The example is selected post hoc from the pilot, not a population estimate.
Original reaction and condition workbooks are opened read-only."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import analyze_heldout_routes as a
from verify_heldout_routes import forest_capacity, independent_route
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
OUT = Path(__file__).resolve().parent
RAW = RAW_DIR

def main():
    groups, focal, contexts, values, supports = a.load()
    candidate = pd.read_csv(OUT / 'pilot_witness_candidates.csv').iloc[0].to_dict()
    gid, holdout, s, t = [int(candidate[k]) for k in ['group_id', 'heldout_position', 'node0', 'node1']]
    g = groups[gid]
    sid, ids = (int(g['support_id']), json.loads(g['context_ids']))
    su = supports[sid]
    y = a.yield_array(ids, sid, contexts, values, supports)
    sy4 = y[[i for i in range(3) if i != holdout]]
    sy, my, ty, uy = (sy4.min((0, 1)), sy4.mean((0, 1)), y[holdout].min(0), y.min((0, 1)))
    ee = np.array(list(zip(*np.where(np.triu(su['adjacency'], 1))))).astype(int)
    sc, mc, tc, uc = [forest_capacity(vv, ee) for vv in [sy, my, ty, uy]]
    objectives = {'source_maximin': (sy, sc[s, t]), 'source_mean_bottleneck': (my, mc[s, t]), 'full_graph_shortest': (None, 0.0), 'union_oracle_evaluation_only': (uy, uc[s, t]), 'target_oracle_evaluation_only': (ty, tc[s, t])}
    source_optimal_target_y = np.where(sy >= sc[s, t], ty, -np.inf)
    source_optimal_target_cap = forest_capacity(source_optimal_target_y, ee)[s, t]
    objectives['best_target_with_source_optimum_evaluation_only'] = (source_optimal_target_y, source_optimal_target_cap)
    route_rows, results = ([], {})
    for name, (objective, floor) in objectives.items():
        route, tie = independent_route(su['neighbors'], objective, s, t, floor)
        achieved_source, achieved_target = (float(sy[route].min()), float(ty[route].min()))
        results[name] = {'source_bottleneck': achieved_source, 'target_bottleneck': achieved_target, 'joint_bottleneck': min(achieved_source, achieved_target), 'n_edges': len(route) - 1, 'multiple_shortest_optimal_routes': bool(tie), 'recipe_route': [su['recipes'][v] for v in route]}
        for step, v in enumerate(route):
            row = {'route': name, 'path_step': step, 'node': v, 'recipe_id': su['recipes'][v], 'source_node_minimum': sy[v], 'target_node_minimum': ty[v], 'union_node_minimum': uy[v]}
            row.update({f'partner{p}_{side}_yield': y[p, q, v] for p in range(3) for q, side in enumerate(['a', 'b'])})
            route_rows.append(row)
    a.save(pd.DataFrame(route_rows), 'measured_witness_routes.csv')
    node_rows = []
    for p, c in enumerate(ids):
        for q, side in enumerate(['a', 'b']):
            for v, recipe in enumerate(su['recipes']):
                node_rows.append({'context_id': c, 'partner_position': p, 'heldout': p == holdout, 'focal': side, 'block_id': contexts[c]['block_' + side], 'node': v, 'recipe_id': recipe, 'yield': y[p, q, v]})
    a.save(pd.DataFrame(node_rows), 'measured_witness_complete_node_yields.csv')
    a.save(pd.DataFrame([{'node0': int(i), 'node1': int(j), 'recipe0': su['recipes'][i], 'recipe1': su['recipes'][j]} for i, j in ee]), 'measured_witness_complete_edges.csv')
    a.save(pd.DataFrame([{'context_id': c, 'partner_position': p, 'heldout': p == holdout, **contexts[c]} for p, c in enumerate(ids)]), 'measured_witness_acquisition_contexts.csv')
    condition = a.read('full_condition_registry.csv.gz')
    condition = condition[(condition.dataset_id == g['dataset_id']) & condition.recipe_id.isin(su['recipes'])]
    a.save(condition, 'measured_witness_conditions.csv')
    block_ids = {contexts[c]['block_' + side] for c in ids for side in ['a', 'b']}
    records = []
    source_path = a.ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
    for chunk in pd.read_csv(source_path, chunksize=150000, keep_default_na=False, low_memory=False, float_precision='round_trip'):
        records.append(chunk[chunk.block_id.isin(block_ids) & chunk.recipe_id.isin(su['recipes'])])
    records = pd.concat(records, ignore_index=True)
    a.save(records, 'measured_witness_source_records.csv.gz')
    hashes, checked, reconstructed = ({}, [], [])
    for (filename, sheet), part in records.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_csv(path, keep_default_na=False) if sheet == 'CSV' else reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in part.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            yy = float(rr['yield'])
            assert abs(yy - record['yield_numeric']) < 1e-12
            for field in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive']:
                assert str(rr[field]) == str(record[field])
            assert str(rr['name0']).startswith(str(record['acquisition']) + '-')
            checked.append({'record_id': record['record_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'raw_yield': yy, 'identity_and_yield_fields_match': True})
            reconstructed.append({'block_id': record['block_id'], 'recipe_id': record['recipe_id'], 'yield': yy})
    a.save(pd.DataFrame(checked), 'measured_witness_raw_reaction_readback.csv')
    means = pd.DataFrame(reconstructed).groupby(['block_id', 'recipe_id'])['yield'].mean()
    for row in node_rows:
        assert abs(means.loc[row['block_id'], row['recipe_id']] - row['yield']) < 1e-12
    checks = []
    fields = ['condition_id', *[f'Reagent{k}_id' for k in range(1, 7)], *[f'Reagent{k}_HTE_Equiv' for k in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    for (filename, sheet), part in condition.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in part.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            for field in fields:
                if str(rr[field]) != str(record[field]):
                    assert abs(float(rr[field]) - float(record[field])) < 1e-12
            checks.append({'recipe_id': record['recipe_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'all_fields_match': True})
    a.save(pd.DataFrame(checks), 'measured_witness_raw_condition_readback.csv')
    f = focal.set_index('focal_key_id').loc[g['focal_key_id']].to_dict()
    witness = {'group_id': gid, 'heldout_position': holdout, 'heldout_context_id': ids[holdout], 'support_id': sid, 'dataset_id': g['dataset_id'], 'rxn_class': g['rxn_class'], 'focal_key_id': g['focal_key_id'], 'molecule_a': f['molecule_a'], 'molecule_b': f['molecule_b'], 'partners': [contexts[c]['partner_smiles'] for c in ids], 'endpoint0_node': s, 'endpoint1_node': t, 'endpoint0_recipe': su['recipes'][s], 'endpoint1_recipe': su['recipes'][t], 'n_nodes': su['n_nodes'], 'n_edges': su['n_edges'], 'source_optimum': float(sc[s, t]), 'target_optimum': float(tc[s, t]), 'union_oracle': float(uc[s, t]), 'best_target_at_source_optimum_evaluation_only': float(source_optimal_target_cap), 'source_endpoint_minimum': float(min(sy[s], sy[t])), 'target_endpoint_minimum': float(min(ty[s], ty[t])), 'routes': results, 'selection': 'posthoc largest observed avoidable maximin-union gap within the first500 pilot triples, not a prevalence estimator', 'source_optimal_tie_warning': 'production policy uses shortest then lexical recipe IDs; target-optimized routes are evaluation-only upper bounds and never policy tie-breaks'}
    a.json_write(witness, 'measured_witness.json')
    verification = {'status': 'passed', 'n_raw_reaction_records': len(checked), 'n_raw_condition_rows': len(checks), 'n_node_target_means_reconstructed': len(node_rows), 'n_exact_zero_raw_reactions': sum((r['raw_yield'] == 0 for r in checked)), 'all_witness_means_reconstructed_from_raw': True, 'condition_fields_checked': fields, 'condition_identities_doses_solvent_temperature_time_verified': True, 'raw_workbook_sha256': hashes}
    a.json_write(verification, 'measured_witness_raw_verification.json')
    print(json.dumps(witness, ensure_ascii=True), flush=True)
    print(json.dumps(verification, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
