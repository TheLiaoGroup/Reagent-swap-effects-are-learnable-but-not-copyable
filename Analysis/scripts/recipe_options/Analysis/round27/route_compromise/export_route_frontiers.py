"""Raw-verified measured examples and exhaustive two-context route frontiers."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
RAW = RAW_DIR
TOKENS = [*[f'Reagent{k}_token' for k in range(1, 7)], 'Solvent_token', 'Temperature_token', 'Time_token']

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def save(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')

def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def forest(y, adj):
    n = len(y)
    cap = np.full((n, n), -np.inf)
    np.fill_diagonal(cap, y)
    parents = list(range(n))
    members = [[k] for k in range(n)]

    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k
    ii, jj = np.where(np.triu(adj, 1))
    weights = np.minimum(y[ii], y[jj])
    for at in np.argsort(-weights, kind='stable'):
        a, b = (find(int(ii[at])), find(int(jj[at])))
        if a != b:
            cap[np.ix_(members[a], members[b])] = weights[at]
            cap[np.ix_(members[b], members[a])] = weights[at]
            parents[b] = a
            members[a] += members[b]
            members[b] = []
    return cap

def route_at(f, g, adj, start, goal, a, b):
    allowed = (f >= a) & (g >= b)
    queue, prev = (deque([start]), {start: None})
    while queue:
        k = queue.popleft()
        if k == goal:
            break
        for j in np.flatnonzero(adj[k] & allowed):
            j = int(j)
            if j not in prev:
                prev[j] = k
                queue.append(j)
    assert goal in prev
    result, k = ([], goal)
    while k is not None:
        result.append(k)
        k = prev[k]
    return result[::-1]

def export(row, destination, values, supports, conditions, sources):
    destination.mkdir(exist_ok=True)
    recipes = json.loads(supports[int(row['fixed_support_number'])]['recipe_ids'])
    start, goal = (int(row['endpoint0_index']), int(row['endpoint1_index']))
    cond = conditions[conditions.dataset_id.eq(row['dataset_id'])].drop_duplicates('recipe_id').set_index('recipe_id')
    tok = cond.loc[recipes, TOKENS].to_numpy(str)
    adj = (tok[:, None, :] != tok[None, :, :]).sum(-1) == 1
    blocks_a, blocks_b = (json.loads(row['aligned_blocks_a']), json.loads(row['aligned_blocks_b']))
    ya = np.array([[values[b][r] for r in recipes] for b in blocks_a])
    yb = np.array([[values[b][r] for r in recipes] for b in blocks_b])
    assert np.array_equal(ya[:2], yb[:2])
    y = np.concatenate([ya, yb[2:]])
    blocks = blocks_a + blocks_b[2:]
    labels = ['AP', 'BP', 'AQ', 'BQ', 'AR', 'BR']
    f, g, h = (ya.min(0), yb.min(0), y.min(0))
    ca, cb, cu = (forest(f, adj), forest(g, adj), forest(h, adj))
    np.testing.assert_allclose([ca[start, goal], cb[start, goal], cu[start, goal]], [row['B_context_a'], row['B_context_b'], row['B_union']], atol=1e-12)
    rows = []
    for k, recipe in enumerate(recipes):
        rows.append(dict(node=k, recipe_id=recipe, endpoint=k in [start, goal], context_a_node_yield=f[k], context_b_node_yield=g[k], union_node_yield=h[k], **{'yield_' + label: y[i, k] for i, label in enumerate(labels)}, **{'block_' + label: blocks[i] for i, label in enumerate(labels)}))
    save(pd.DataFrame(rows), destination / 'complete_graph_nodes.csv')
    ii, jj = np.where(np.triu(adj, 1))
    edge_rows = []
    for a, b in zip(ii, jj):
        slots = np.flatnonzero(tok[a] != tok[b])
        assert len(slots) == 1
        edge_rows.append(dict(node0=a, node1=b, recipe0=recipes[a], recipe1=recipes[b], changed_slot=TOKENS[slots[0]].removesuffix('_token')))
    save(pd.DataFrame(edge_rows), destination / 'complete_graph_edges.csv')
    candidates = []
    for floor in sorted(set(f)):
        allowed = f >= floor
        if not allowed[start] or not allowed[goal]:
            continue
        subadj = adj & allowed[:, None] & allowed[None, :]
        beta = forest(g, subadj)[start, goal]
        if np.isfinite(beta):
            route = route_at(f, g, adj, start, goal, floor, beta)
            candidates.append((float(f[route].min()), float(g[route].min()), route))
    pairs = sorted(set(((a, b) for a, b, _ in candidates)))
    frontier = [(a, b) for a, b in pairs if not any((aa >= a and bb >= b and (aa > a or bb > b) for aa, bb in pairs))]
    frontier_rows, route_rows = ([], [])
    for k, (a, b) in enumerate(frontier):
        route = route_at(f, g, adj, start, goal, a, b)
        frontier_rows.append(dict(frontier_id=k, context_a_bottleneck=a, context_b_bottleneck=b, minimum_bottleneck=min(a, b), max_regret=max(ca[start, goal] - a, cb[start, goal] - b), route_nodes=json.dumps(route), route_recipes=json.dumps([recipes[j] for j in route])))
        for step, j in enumerate(route):
            route_rows.append(dict(frontier_id=k, step=step, **rows[j]))
    save(pd.DataFrame(frontier_rows), destination / 'complete_pareto_frontier.csv')
    save(pd.DataFrame(route_rows), destination / 'frontier_measured_route_yields.csv')
    assert abs(max((min(a, b) for a, b in frontier)) - cu[start, goal]) < 1e-12
    assert abs(max((a for a, b in frontier)) - ca[start, goal]) < 1e-12
    assert abs(max((b for a, b in frontier)) - cb[start, goal]) < 1e-12
    reached = {start}
    queue = deque([start])
    while queue:
        k = queue.popleft()
        for j in np.flatnonzero(adj[k] & (h >= 0.2)):
            if int(j) not in reached:
                reached.add(int(j))
                queue.append(int(j))
    assert goal not in reached
    boundary = sorted({int(j) for k in reached for j in np.flatnonzero(adj[k])} - reached)
    assert all((h[k] < 0.2 for k in boundary))
    save(pd.DataFrame([dict(**rows[k], failed_targets20='|'.join((label for z, label in enumerate(labels) if y[z, k] < 0.2))) for k in boundary]), destination / 'joint20_obstruction_boundary.csv')
    selected = sources[sources.block_id.isin(blocks) & sources.recipe_id.isin(recipes)]
    save(selected, destination / 'source_reaction_records.csv')
    conditions_selected = conditions[conditions.dataset_id.eq(row['dataset_id']) & conditions.recipe_id.isin(recipes)]
    save(conditions_selected, destination / 'source_condition_records.csv')
    raw_hashes, checks, cc = ({}, [], [])
    for (filename, sheet), group in selected.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        raw_hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_csv(path, keep_default_na=False) if sheet == 'CSV' else reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in group.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            assert abs(float(rr['yield']) - float(record['yield_numeric'])) < 1e-12
            for field in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive']:
                assert str(rr[field]) == str(record[field]), (filename, record['source_row'], field)
            assert str(rr['name0']).startswith(str(record['acquisition']) + '-')
            checks.append(dict(record_id=record['record_id'], source_row=record['source_row'], all_fields_match=True))
    fields = ['condition_id', *[f'Reagent{k}_id' for k in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    for (filename, sheet), group in conditions_selected.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        raw_hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in group.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            for field in fields:
                if str(rr[field]) != str(record[field]):
                    assert abs(float(rr[field]) - float(record[field])) < 1e-12
            cc.append(dict(recipe_id=record['recipe_id'], source_row=record['source_row'], all_fields_match=True))
    back = selected.groupby(['block_id', 'recipe_id']).yield_numeric.mean()
    for i, block in enumerate(blocks):
        for j, recipe in enumerate(recipes):
            assert abs(back[block, recipe] - y[i, j]) < 1e-12
    save(pd.DataFrame(checks), destination / 'raw_reaction_readback_checks.csv')
    save(pd.DataFrame(cc), destination / 'raw_condition_readback_checks.csv')
    row['n_pareto_routes'] = len(frontier)
    row['minimum_max_context_regret'] = min((max(ca[start, goal] - a, cb[start, goal] - b) for a, b in frontier))
    row['identical_native_supports'] = row['n_fixed_nodes'] == row['n_native_nodes_a'] == row['n_native_nodes_b']
    row['start_reachable_joint20_nodes'] = len(reached)
    row['joint20_boundary_nodes'] = len(boundary)
    row['selection_is_outcome_conditioned'] = True
    dump(row, destination / 'witness.json')
    verification = dict(status='passed', n_raw_reaction_records=len(checks), n_raw_condition_records=len(cc), all_means_match_raw=True, independent_maximum_spanning_forests=True, pareto_frontier_sweeps_all_distinct_source_capacities=True, original_workbook_sha256=raw_hashes)
    dump(verification, destination / 'verification.json')
    print(json.dumps(dict(destination=str(destination), **verification)), flush=True)

def main():
    engine = load_module(OUT / 'analyze_route_compromise.py', 'compromise_sources')
    inv, supports, tokens, values, numbers = engine.load()
    conditions = pd.read_csv(ANALYSIS / 'data/condition_slot_registry.csv', keep_default_na=False)
    sources = pd.read_csv(ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz', keep_default_na=False)
    primary = json.loads((OUT / 'witness_selection.json').read_text(encoding='utf-8'))
    export(primary, OUT / 'maximum_clearance_witness', values, supports, conditions, sources)
    stats = pd.read_csv(OUT / 'complete_comparison_threshold_statistics.csv.gz')
    merged = stats.merge(inv, on='comparison_id', validate='many_to_one')
    native = merged[merged.identical_anchor_blocks & merged.n_fixed_nodes.eq(merged.n_native_nodes_a) & merged.n_fixed_nodes.eq(merged.n_native_nodes_b)]
    numeric = ['n_high_endpoint_views', 'n_connected_views', 'n_both_contexts_feasible', 'n_scope_failures', 'n_scope_failures_clearance5pp', 'n_compromise_ge5pp']
    summary = native.groupby('threshold')[numeric].sum().reset_index()
    save(summary, OUT / 'identical_native_support_summary.csv')
    ids = set(native[(native.threshold == 0.2) & native.n_scope_failures.gt(0)].comparison_id)
    selected = None
    for chunk in pd.read_csv(OUT / 'complete_primary20_endpoint_views.csv.gz', chunksize=200000):
        part = chunk[chunk.comparison_id.isin(ids) & chunk.scope_failure20.eq(1)]
        for r in part.to_dict('records'):
            margin = min(r['B_context_a'] - 0.2, r['B_context_b'] - 0.2, 0.2 - r['B_union'])
            rank = (margin, r['compromise'], -int(r['comparison_id']), -int(r['endpoint0_number']), -int(r['endpoint1_number']))
            if selected is None or rank > selected[0]:
                selected = (rank, r)
    if selected:
        row = inv.set_index('comparison_id').loc[selected[1]['comparison_id']].to_dict()
        row.update(selected[1])
        row['comparison_id'] = int(row['comparison_id'])
        num_to_recipe = {v: k for k, v in numbers.items()}
        recipes = json.loads(supports[int(row['fixed_support_number'])]['recipe_ids'])
        for end in [0, 1]:
            row[f'endpoint{end}_index'] = recipes.index(num_to_recipe[int(row[f'endpoint{end}_number'])])
        row['clearance'] = selected[0][0]
        export(row, OUT / 'identical_native_support_witness', values, supports, conditions, sources)
if __name__ == '__main__':
    main()
