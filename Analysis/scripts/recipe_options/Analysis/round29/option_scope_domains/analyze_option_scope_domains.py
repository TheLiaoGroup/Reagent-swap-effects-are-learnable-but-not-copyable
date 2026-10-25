"""Measured option scope on all individually feasible collective conflicts."""
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ROUND = OUT.parent
ANALYSIS = ROUND.parent
SRC = ANALYSIS / 'round27/partner_scope/triple_scope'
R28 = ANALYSIS / 'round28/shared_bridges'
H = 0.2
SLOTS = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', **kwargs)

def save(rows, name):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def activation_capacity(weights, adjacency, start, goal, allowed):
    parent = np.arange(len(weights))
    active = np.zeros(len(weights), bool)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for node in np.argsort(-weights, kind='stable'):
        if not allowed[node]:
            continue
        active[node] = True
        for other in np.flatnonzero(adjacency[node] & active):
            parent[find(other)] = find(node)
        if active[start] and active[goal] and (find(start) == find(goal)):
            return float(weights[node])
    return -np.inf

def high_path(weights, adjacency, start, goal, allowed, threshold):
    valid = allowed & (weights >= threshold)
    if not valid[start] or not valid[goal]:
        return []
    queue, previous = ([start], {start: None})
    for current in queue:
        if current == goal:
            break
        for other in np.flatnonzero(adjacency[current] & valid):
            other = int(other)
            if other not in previous:
                previous[other] = current
                queue.append(other)
    if goal not in previous:
        return []
    path, current = ([], goal)
    while current is not None:
        path.append(current)
        current = previous[current]
    return path[::-1]

def main():
    begun = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    helper_path = ROUND / 'component_options/analyze_component_options.py'
    spec = importlib.util.spec_from_file_location('option_names', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    groups = read(SRC / 'complete_group_registry.csv.gz')
    group_map = groups.set_index('group_id')
    contexts = read(SRC / 'complete_context_registry.csv.gz').set_index('context_id', drop=False)
    supports = read(SRC / 'complete_support_registry.csv.gz').set_index('support_id')
    conditions = read(SRC / 'full_condition_registry.csv.gz').drop_duplicates(['dataset_id', 'recipe_id'])
    condition_maps = {d: p.set_index('recipe_id', drop=False) for d, p in conditions.groupby('dataset_id')}
    values = {b: p.set_index('recipe_id').yield_mean.to_dict() for b, p in read(SRC / 'full_source_block_node_yields.csv.gz').groupby('block_id')}
    diagnostics = read(SRC / 'complete_group_pair_support_diagnostic.csv.gz').set_index('group_id')
    cols = ['group_id', 'node0', 'node1', 'minimum_endpoint_yield', 'K_all', 'K_endpoint_all', 'K_endpoint_bundle_min', 'complete_endpoint_corners']
    conflicts, parent_counts = ([], Counter())
    for chunk in pd.read_csv(R28 / 'complete_endpoint_bridge_capacities.csv.gz', usecols=cols, float_precision='round_trip', chunksize=400000):
        high = chunk.minimum_endpoint_yield >= H
        eligible = high & (chunk.K_endpoint_bundle_min >= H)
        z = chunk[eligible & (chunk.K_endpoint_all < H)]
        conflicts.append(z)
        parent_counts['high20'] += int(high.sum())
        parent_counts['complete_high20'] += int((high & chunk.complete_endpoint_corners).sum())
        parent_counts['complete_individually_feasible20'] += int((eligible & chunk.complete_endpoint_corners).sum())
    conflict = pd.concat(conflicts, ignore_index=True)
    assert len(conflict) == 542 and int(conflict.complete_endpoint_corners.sum()) == 475
    assert parent_counts['complete_individually_feasible20'] == 489953
    option_rows, domain_rows, snapshots = ([], [], {})
    for source in conflict.to_dict('records'):
        gid, start, goal = (int(source[k]) for k in ['group_id', 'node0', 'node1'])
        g = group_map.loc[gid]
        recipes = json.loads(supports.loc[g.support_id, 'recipe_ids'])
        cond = condition_maps[g.dataset_id].loc[recipes]
        tokens = cond[SLOTS].to_numpy(str)
        adjacency = (tokens[:, None] != tokens[None, :]).sum(2) == 1
        cids = json.loads(g.context_ids)
        yy = np.array([[[values[contexts.loc[c, b]][r] for r in recipes] for b in ['block_a', 'block_b']] for c in cids])
        weights = yy.min((0, 1))
        baseline = (tokens == tokens[start]) | (tokens == tokens[goal])
        kbase = activation_capacity(weights, adjacency, start, goal, baseline.all(1))
        kfull = activation_capacity(weights, adjacency, start, goal, np.ones(len(tokens), bool))
        assert kbase == source['K_endpoint_all'] and kfull == source['K_all']
        view = f'g{gid}_n{start}_n{goal}'
        metadata = {**source, 'view_id': view, 'dataset_id': g.dataset_id, 'rxn_class': g.rxn_class, 'focal_key_id': int(g.focal_key_id), 'molecular_pair_id': g.molecular_pair_id, 'support_id': int(g.support_id), 'context_ids': g.context_ids, 'molecule_a': contexts.loc[cids[0], 'molecule_a'], 'molecule_b': contexts.loc[cids[0], 'molecule_b'], 'partner_smiles': json.dumps(contexts.loc[cids, 'partner_smiles'].tolist()), 'recipe0': recipes[start], 'recipe1': recipes[goal], 'native_pair_supports_identical': bool(diagnostics.loc[gid, 'all_native_pair_supports_identical_to_full_support']), 'full_graph_rescues20': kfull >= H}
        tested = []
        for slot in range(len(SLOTS)):
            for token in sorted(set(tokens[:, slot]) - {tokens[start, slot], tokens[goal, slot]}):
                desc = helper.option_description(slot, token, cond)
                expanded = baseline.copy()
                expanded[:, slot] |= tokens[:, slot] == token
                mask = expanded.all(1)
                capacity = activation_capacity(weights, adjacency, start, goal, mask)
                route20 = high_path(weights, adjacency, start, goal, mask, H)
                assert bool(route20) == (capacity >= H)
                tested.append(capacity >= H)
                option_rows.append({**metadata, **desc, 'K_single': capacity, 'single_sufficient20': capacity >= H, 'n_allowed_nodes': int(mask.sum()), 'endpoint_changing_slot': bool(tokens[start, slot] != tokens[goal, slot]), 'route20': json.dumps(route20)})
        domain_rows.append({**metadata, 'n_tested_options': len(tested), 'n_single_sufficient': int(sum(tested))})
        snapshots[view] = (g, recipes, cond, tokens, adjacency, cids, yy)
    options = pd.DataFrame(option_rows)
    domains = pd.DataFrame(domain_rows)
    save(options, 'all_conflict_option_capacities.csv.gz')
    save(domains, 'all_collective_conflict_domains.csv')
    fields = ['group_id', 'focal_key_id', 'molecular_pair_id', 'dataset_id', 'rxn_class', 'support_id', 'context_ids', 'n_nodes']
    ledger = groups[fields].copy()
    grouped = domains.groupby('group_id').agg(n_conflict_views20=('view_id', 'size'), n_complete_conflicts20=('complete_endpoint_corners', 'sum'), n_full_rescued20=('full_graph_rescues20', 'sum'))
    ledger = ledger.merge(grouped, on='group_id', how='left', validate='one_to_one')
    for field in ['n_conflict_views20', 'n_complete_conflicts20', 'n_full_rescued20']:
        ledger[field] = ledger[field].fillna(0).astype(int)
    assert len(ledger) == 44334
    save(ledger, 'complete_group_scope_ledger.csv.gz')
    matched = []
    keys = ['dataset_id', 'focal_key_id', 'support_id', 'recipe0', 'recipe1', 'option_id']
    for key, part in options.groupby(keys, sort=True):
        for a, b in itertools.combinations(part.to_dict('records'), 2):
            ca, cb = (set(json.loads(a['context_ids'])), set(json.loads(b['context_ids'])))
            if len(ca & cb) != 2:
                continue
            unique_a, unique_b = (next(iter(ca - cb)), next(iter(cb - ca)))
            pa, pb = (contexts.loc[unique_a, 'partner_smiles'], contexts.loc[unique_b, 'partner_smiles'])
            molecular_change = pa != pb
            matched.append({'view_a': a['view_id'], 'view_b': b['view_id'], **dict(zip(keys, key)), 'option_name': a['option_name'], 'slot': a['slot'], 'token': a['token'], 'molecule_a': a['molecule_a'], 'molecule_b': a['molecule_b'], 'shared_context_ids': json.dumps(sorted(ca & cb)), 'changed_context_a': unique_a, 'changed_context_b': unique_b, 'changed_partner_a': pa, 'changed_partner_b': pb, 'different_partner_molecule': molecular_change, 'both_complete_corners': a['complete_endpoint_corners'] and b['complete_endpoint_corners'], 'both_native_supports_identical': a['native_pair_supports_identical'] and b['native_pair_supports_identical'], 'K_single_a': a['K_single'], 'K_single_b': b['K_single'], 'sufficient_a': a['single_sufficient20'], 'sufficient_b': b['single_sufficient20'], 'sufficiency_changed': a['single_sufficient20'] != b['single_sufficient20'], 'full_rescues_a': a['full_graph_rescues20'], 'full_rescues_b': b['full_graph_rescues20'], 'full_rescue_changed': a['full_graph_rescues20'] != b['full_graph_rescues20'], 'capacity_difference_pp': 100 * abs(a['K_single'] - b['K_single']), 'nearest_threshold_clearance_pp': 100 * min(abs(a['K_single'] - H), abs(b['K_single'] - H))})
    contrast = pd.DataFrame(matched)
    save(contrast, 'same_support_single_partner_contrasts.csv.gz')
    summaries = []
    for complete in [False, True]:
        d = domains[domains.complete_endpoint_corners == complete]
        summaries.append({'stratum': 'complete' if complete else 'incomplete', 'n_conflict_views': len(d), 'n_triples': d.group_id.nunique(), 'n_exact_molecular_pairs': d.molecular_pair_id.nunique(), 'n_full_graph_rescued': int(d.full_graph_rescues20.sum()), 'n_no_full_graph_rescue': int((~d.full_graph_rescues20).sum()), 'n_any_single_option_rescue': int((d.n_single_sufficient > 0).sum()), 'n_tested_options': int(d.n_tested_options.sum())})
    save(summaries, 'conflict_support_summary.csv')
    contrast_summary = []
    if len(contrast):
        for molecular in [False, True]:
            for complete in [False, True]:
                z = contrast[contrast.different_partner_molecule == molecular]
                if complete:
                    z = z[z.both_complete_corners]
                good = z[z.sufficiency_changed]
                contrast_summary.append({'different_partner_molecule': molecular, 'require_both_complete': complete, 'n_option_contrasts': len(z), 'n_changed_option_contrasts': len(good), 'n_unique_view_pairs': len(z[['view_a', 'view_b']].drop_duplicates()), 'n_changed_view_pairs': len(good[['view_a', 'view_b']].drop_duplicates()), 'n_changed_focal_keys': good.focal_key_id.nunique(), 'n_changed_options': good.option_id.nunique(), 'n_full_rescue_changed_option_contrasts': int(z.full_rescue_changed.sum()), 'n_full_rescue_changed_view_pairs': len(z[z.full_rescue_changed][['view_a', 'view_b']].drop_duplicates())})
    save(contrast_summary, 'matched_option_scope_summary.csv')
    named = options.groupby(['dataset_id', 'slot', 'token', 'option_name'], dropna=False).agg(n_tested_conflict_views=('view_id', 'size'), n_triples=('group_id', 'nunique'), n_exact_molecular_pairs=('molecular_pair_id', 'nunique'), n_sufficient_views=('single_sufficient20', 'sum'), n_full_rescued_views=('full_graph_rescues20', 'sum')).reset_index()
    save(named, 'named_option_conflict_scope.csv')
    inventory = read(SRC / 'complete_20_dataset_18_class_support.csv')[['dataset_id', 'rxn_class']].drop_duplicates()
    cstats = domains.groupby(['dataset_id', 'rxn_class']).agg(n_conflict_views=('view_id', 'size'), n_triples=('group_id', 'nunique'), n_full_rescued=('full_graph_rescues20', 'sum')).reset_index()
    inv = inventory.merge(cstats, how='left', on=['dataset_id', 'rxn_class'], validate='one_to_one')
    inv = inv.fillna(0)
    save(inv, 'complete_20_dataset_18_class_scope.csv')
    overlap = read(ROUND / 'component_options/all_single_options.csv.gz')
    check = options.merge(overlap[['certificate_id', 'option_id', 'K_single']], left_on=['view_id', 'option_id'], right_on=['certificate_id', 'option_id'], suffixes=('_domain', '_certificate'), validate='one_to_one')
    assert len(check) == len(overlap) and np.array_equal(check.K_single_domain, check.K_single_certificate)
    selected = None
    if len(contrast):
        candidates = contrast[contrast.different_partner_molecule & contrast.both_complete_corners & contrast.sufficiency_changed]
        if len(candidates):
            selected = candidates.sort_values(['both_native_supports_identical', 'nearest_threshold_clearance_pp', 'capacity_difference_pp'], ascending=False).iloc[0].to_dict()
            dump(selected, 'measured_scope_witness.json')
            export_witness(selected, snapshots, contexts, options)
    hashes = {}
    for p in [R28 / 'complete_endpoint_bridge_capacities.csv.gz', SRC / 'complete_group_registry.csv.gz', SRC / 'complete_context_registry.csv.gz', SRC / 'complete_support_registry.csv.gz', SRC / 'full_condition_registry.csv.gz', SRC / 'full_source_block_node_yields.csv.gz', SRC / 'complete_group_pair_support_diagnostic.csv.gz', helper_path, ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz']:
        with p.open('rb') as stream:
            hashes[str(p.relative_to(ANALYSIS))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    dump(hashes, 'source_manifest.json')
    result = {'status': 'analyzed', 'n_parent_triples': len(groups), **parent_counts, 'n_conflict_views': len(domains), 'n_option_tests': len(options), 'n_overlap_independently_equal_capacity_tests': len(check), 'n_matched_contrasts': len(contrast), 'selected_witness': selected, 'all_option_threshold_states_verified_by_independent_bfs': True, 'seconds': time.time() - begun}
    dump(result, 'run_summary.json')
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)

def export_witness(selected, snapshots, contexts, options):
    node_rows, condition_rows, routes, context_rows = ([], [], [], [])
    for label in ['a', 'b']:
        view = selected['view_' + label]
        g, recipes, cond, tokens, adjacency, cids, yy = snapshots[view]
        r = options[(options.view_id == view) & (options.option_id == selected['option_id'])].iloc[0]
        start, goal = (int(r.node0), int(r.node1))
        slot = SLOTS.index(r.slot)
        allowed = (tokens == tokens[start]) | (tokens == tokens[goal])
        allowed[:, slot] |= tokens[:, slot] == r.token
        mask = allowed.all(1)
        weights = yy.min((0, 1))
        k = r.K_single
        route = high_path(weights, adjacency, start, goal, mask, k)
        assert route and min(weights[route]) == k
        for p, c in enumerate(cids):
            context_rows.append({'side': label, 'view_id': view, **contexts.loc[c].to_dict()})
            for n, recipe in enumerate(recipes):
                node_rows.append({'side': label, 'view_id': view, 'group_id': int(g.name), 'partner_position': p, 'context_id': c, 'node': n, 'recipe_id': recipe, 'block_a': contexts.loc[c, 'block_a'], 'block_b': contexts.loc[c, 'block_b'], 'yield_a': yy[p, 0, n], 'yield_b': yy[p, 1, n], 'single_option_allowed': bool(mask[n])})
        for step, n in enumerate(route):
            routes.append({'side': label, 'view_id': view, 'step': step, 'node': n, 'recipe_id': recipes[n], 'shared_node_yield': weights[n], 'bottleneck': k})
        condition_rows.extend(cond.to_dict('records'))
    save(node_rows, 'measured_scope_witness_full_node_yields.csv')
    save(routes, 'measured_scope_witness_routes.csv')
    save(context_rows, 'measured_scope_witness_contexts.csv')
    save(pd.DataFrame(condition_rows).drop_duplicates(['dataset_id', 'recipe_id']), 'measured_scope_witness_conditions.csv')
    blocks = sorted(set(pd.DataFrame(node_rows).block_a) | set(pd.DataFrame(node_rows).block_b))
    parts = []
    provenance = ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
    recipes = set(pd.DataFrame(node_rows).recipe_id)
    for chunk in pd.read_csv(provenance, keep_default_na=False, float_precision='round_trip', chunksize=150000):
        part = chunk[chunk.block_id.isin(blocks) & chunk.recipe_id.isin(recipes)]
        if len(part):
            parts.append(part)
    assert parts
    save(pd.concat(parts, ignore_index=True).drop_duplicates('record_id'), 'measured_scope_witness_source_records.csv.gz')
if __name__ == '__main__':
    main()
