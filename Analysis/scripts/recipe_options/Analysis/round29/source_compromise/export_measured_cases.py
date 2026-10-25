"""Posthoc measured cases of positive compromise and source-optimal tie rescue."""
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
import analyze_source_compromise as a
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
OUT = Path(__file__).resolve().parent
RAW = RAW_DIR
v = a.load_module('r28_verify_readonly', a.R28 / 'verify_heldout_routes.py')

def main():
    groups, focal, contexts, values, supports = a.a.load()
    candidates = a.read(OUT / 'all_frozen_failure_candidates.csv.gz')
    previous = json.loads((a.R28 / 'measured_witness.json').read_text(encoding='utf-8'))
    selected = [{'classification': 'positive_source_compromise', 'group_id': previous['group_id'], 'heldout_position': previous['heldout_position'], 'node0': previous['endpoint0_node'], 'node1': previous['endpoint1_node'], 'selection': 'R28 measured witness282 retained for cross-round chemical comparability; posthoc pilot explanatory case'}]
    tied = candidates[candidates.classification == 'tie_rescue_only'].copy()
    if len(tied):
        tied['recovery'] = tied.union_oracle - tied.frozen_joint
        chosen = tied.sort_values(['recovery', 'source_optimum', 'group_id', 'heldout_position', 'node0', 'node1'], ascending=[False, False, True, True, True, True]).iloc[0].to_dict()
        selected.append({k: chosen[k] for k in ['classification', 'group_id', 'heldout_position', 'node0', 'node1']} | {'selection': 'posthoc full-frame largest union-oracle minus frozen-joint loss among tie-only failures; source optimum then IDs break ties'})
    all_cases, all_routes, all_nodes, all_edges, all_contexts, all_frontiers, all_h = ([], [], [], [], [], [], [])
    all_conditions, needed_blocks, recipes_by_block = ([], set(), {})
    for chosen in selected:
        label = chosen['classification']
        gid, holdout, s, t = [int(chosen[k]) for k in ['group_id', 'heldout_position', 'node0', 'node1']]
        g = groups[gid]
        sid, ids = (int(g['support_id']), json.loads(g['context_ids']))
        su = supports[sid]
        y = a.a.yield_array(ids, sid, contexts, values, supports)
        sy, ty = (y[[i for i in range(3) if i != holdout]].min((0, 1)), y[holdout].min(0))
        uy = np.minimum(sy, ty)
        ee = np.column_stack(np.where(np.triu(su['adjacency'], 1)))
        B, T, U = [a.forest_capacity(vv, ee)[s, t] for vv in [sy, ty, uy]]
        target_at_B_y = np.where(sy >= B, ty, -np.inf)
        best_target_at_B = a.forest_capacity(target_at_B_y, ee)[s, t]
        source_at_target20_y = np.where(ty >= 0.2, sy, -np.inf)
        F = a.forest_capacity(source_at_target20_y, ee)[s, t]
        assert F >= 0.2
        assert (F < B) == (label == 'positive_source_compromise')
        objectives = {'frozen_source_maximin': (sy, B), 'best_target_among_all_source_optima': (target_at_B_y, best_target_at_B), 'max_source_subject_target20': (source_at_target20_y, F), 'full_union_oracle': (uy, U), 'target_own_oracle': (ty, T)}
        route_result = {}
        for name, (objective, floor) in objectives.items():
            route, tie = v.independent_route(su['neighbors'], objective, s, t, floor)
            ss, tt = (float(sy[route].min()), float(ty[route].min()))
            route_result[name] = {'source_bottleneck': ss, 'target_bottleneck': tt, 'joint_bottleneck': min(ss, tt), 'n_edges': len(route) - 1, 'multiple_shortest_objective_optimal_paths': bool(tie), 'nodes': [int(z) for z in route], 'recipes': [su['recipes'][z] for z in route]}
            for step, node in enumerate(route):
                all_routes.append({'case': label, 'route': name, 'step': step, 'node': node, 'recipe_id': su['recipes'][node], 'source_minimum': sy[node], 'target_minimum': ty[node], 'union_minimum': uy[node], **{f'partner{p}_{side}_yield': y[p, q, node] for p in range(3) for q, side in enumerate(['a', 'b'])}})
        points = []
        for floor in sorted(set((float(x) for x in sy if x <= B)), reverse=True):
            objective = np.where(sy >= floor, ty, -np.inf)
            target_cap = a.forest_capacity(objective, ee)[s, t]
            route, _ = v.independent_route(su['neighbors'], objective, s, t, target_cap)
            points.append((float(sy[route].min()), float(ty[route].min()), tuple(route)))
        unique = {(ss, tt): route for ss, tt, route in points}
        frontier = [(ss, tt, route) for (ss, tt), route in unique.items() if not any((s2 >= ss and t2 >= tt and (s2 > ss or t2 > tt) for s2, t2 in unique))]
        frontier.sort(key=lambda z: (-z[0], z[1]))
        for rank, (ss, tt, route) in enumerate(frontier):
            all_frontiers.append({'case': label, 'rank_descending_source': rank, 'source_bottleneck': ss, 'target_bottleneck': tt, 'source_sacrifice_from_B': float(B - ss), 'node_route': json.dumps(route), 'recipe_route': json.dumps([su['recipes'][z] for z in route])})
        for h in [0.1, 0.2, 0.3]:
            FF = a.forest_capacity(np.where(ty >= h, sy, -np.inf), ee)[s, t]
            all_h.append({'case': label, 'fixed_source_endpoint_frame': 'source_high20_R28', 'target_threshold': h, 'B_source': B, 'F_target_h': FF, 'source_sacrifice_if_feasible': float(B - FF) if np.isfinite(FF) else None})
        f = focal.set_index('focal_key_id').loc[g['focal_key_id']].to_dict()
        all_cases.append({'case': label, 'group_id': gid, 'heldout_position': holdout, 'support_id': sid, 'focal_key_id': g['focal_key_id'], 'dataset_id': g['dataset_id'], 'rxn_class': g['rxn_class'], 'molecule_a': f['molecule_a'], 'molecule_b': f['molecule_b'], 'partners': [contexts[c]['partner_smiles'] for c in ids], 'context_ids': ids, 'heldout_context_id': ids[holdout], 'endpoint0_node': s, 'endpoint1_node': t, 'n_nodes': su['n_nodes'], 'n_edges': su['n_edges'], 'B_source': float(B), 'F_target20': float(F), 'source_sacrifice': float(B - F), 'best_target_among_all_source_optima': float(best_target_at_B), 'target_own_optimum': float(T), 'union_oracle': float(U), 'routes': route_result, 'selection': chosen['selection']})
        for p, c in enumerate(ids):
            all_contexts.append({'case': label, 'context_id': c, 'partner_position': p, 'heldout': p == holdout, **contexts[c]})
            for q, side in enumerate(['a', 'b']):
                block = contexts[c]['block_' + side]
                needed_blocks.add(block)
                recipes_by_block.setdefault(block, set()).update(su['recipes'])
                for node, recipe in enumerate(su['recipes']):
                    all_nodes.append({'case': label, 'block_id': block, 'context_id': c, 'partner_position': p, 'heldout': p == holdout, 'focal': side, 'node': node, 'recipe_id': recipe, 'yield': y[p, q, node]})
        all_edges.extend(({'case': label, 'node0': int(i), 'node1': int(j), 'recipe0': su['recipes'][i], 'recipe1': su['recipes'][j]} for i, j in ee))
        conditions = a.read(a.R27 / 'full_condition_registry.csv.gz')
        conditions = conditions[(conditions.dataset_id == g['dataset_id']) & conditions.recipe_id.isin(su['recipes'])].copy()
        conditions.insert(0, 'case', label)
        all_conditions.append(conditions)
    a.save(pd.DataFrame(all_routes), 'measured_cases_routes.csv')
    a.save(pd.DataFrame(all_frontiers), 'measured_cases_exact_pareto_frontiers.csv')
    a.save(pd.DataFrame(all_h), 'measured_cases_target_threshold_sensitivity.csv')
    a.save(pd.DataFrame(all_nodes), 'measured_cases_complete_node_yields.csv')
    a.save(pd.DataFrame(all_edges), 'measured_cases_complete_edges.csv')
    a.save(pd.DataFrame(all_contexts), 'measured_cases_acquisition_contexts.csv')
    conditions = pd.concat(all_conditions, ignore_index=True)
    a.save(conditions, 'measured_cases_conditions.csv')
    source_path = a.ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
    parts = []
    for chunk in a.read(source_path, chunksize=150000):
        picked = chunk[chunk.block_id.isin(needed_blocks)]
        mask = [r in recipes_by_block[b] for b, r in zip(picked.block_id, picked.recipe_id)]
        parts.append(picked[mask])
    records = pd.concat(parts, ignore_index=True).drop_duplicates('record_id')
    a.save(records, 'measured_cases_source_records.csv.gz')
    route_nodes = pd.DataFrame(all_routes)[['case', 'route', 'step', 'node', 'recipe_id']].merge(pd.DataFrame(all_nodes), on=['case', 'node', 'recipe_id'], validate='many_to_many')
    pointer_fields = ['block_id', 'recipe_id', 'record_id', 'source_file', 'source_sheet', 'source_row', 'yield_numeric', 'condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'acquisition']
    pointers = route_nodes.merge(records[pointer_fields], on=['block_id', 'recipe_id'], how='left', validate='many_to_many')
    assert pointers.record_id.notna().all()
    condition_pointers = conditions[['case', 'recipe_id', 'source_file', 'source_sheet', 'source_row']].rename(columns={k: 'condition_' + k for k in ['source_file', 'source_sheet', 'source_row']})
    pointers = pointers.merge(condition_pointers, on=['case', 'recipe_id'], how='left', validate='many_to_one')
    a.save(pointers, 'measured_cases_route_raw_pointers.csv')
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
            checked.append({'record_id': record['record_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'raw_yield': yy, 'identity_and_yield_match': True})
            reconstructed.append({'block_id': record['block_id'], 'recipe_id': record['recipe_id'], 'yield': yy})
    a.save(pd.DataFrame(checked), 'measured_cases_raw_reaction_readback.csv')
    means = pd.DataFrame(reconstructed).groupby(['block_id', 'recipe_id'])['yield'].mean()
    for node in all_nodes:
        assert abs(means.loc[node['block_id'], node['recipe_id']] - node['yield']) < 1e-12
    checks = []
    fields = ['condition_id', *[f'Reagent{k}_id' for k in range(1, 7)], *[f'Reagent{k}_HTE_Equiv' for k in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    for (filename, sheet), part in conditions.drop_duplicates(['dataset_id', 'recipe_id']).groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in part.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            for field in fields:
                if str(rr[field]) != str(record[field]):
                    assert abs(float(rr[field]) - float(record[field])) < 1e-12
            checks.append({'recipe_id': record['recipe_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'all_condition_fields_match': True})
    a.save(pd.DataFrame(checks), 'measured_cases_raw_condition_readback.csv')
    a.write_json(all_cases, 'measured_cases.json')
    a.write_json({'status': 'passed', 'n_cases': len(all_cases), 'n_raw_reaction_records': len(checked), 'n_raw_condition_rows': len(checks), 'n_node_target_means_reconstructed': len(all_nodes), 'n_exact_zero_raw_reactions': sum((r['raw_yield'] == 0 for r in checked)), 'raw_workbook_sha256': hashes, 'condition_fields_checked': fields, 'all_exported_node_means_reconstructed_from_raw': True, 'route_reconstruction': 'independent reverse-distance lexical shortest route at specified objective floor', 'frontier': 'exact nondominated source-target path bottleneck pairs, covering every distinct source-node bottleneck'}, 'measured_cases_raw_verification.json')
    print(json.dumps(all_cases, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
