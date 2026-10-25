"""Resolve R28 shared-scope bridges to measured, slot-specific chemical options."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import heapq
import itertools
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
R28 = ANALYSIS / 'round28/shared_bridges'
SLOTS = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
H = 0.2

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', **kwargs)

def save(rows, filename):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / filename, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if filename.endswith('.gz') else None)

def dump(obj, filename):
    (OUT / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def widest(weights, neighbors, start, goal, allowed):
    best = np.full(len(weights), -np.inf)
    best[start] = weights[start] if allowed[start] else -np.inf
    heap = [(-best[start], start)]
    previous = {start: None}
    while heap:
        negative, node = heapq.heappop(heap)
        value = -negative
        if value != best[node]:
            continue
        if node == goal:
            route = []
            while node is not None:
                route.append(node)
                node = previous[node]
            return (float(value), route[::-1])
        for nxt in neighbors[node]:
            if allowed[nxt]:
                candidate = min(value, weights[nxt])
                if candidate > best[nxt]:
                    best[nxt] = candidate
                    previous[nxt] = node
                    heapq.heappush(heap, (-candidate, int(nxt)))
    return (-np.inf, [])

def option_key(slot, token):
    return hashlib.sha256((SLOTS[slot] + '|' + token).encode()).hexdigest()[:16]

def option_description(slot, token, cond):
    field = SLOTS[slot]
    row = cond[cond[field] == token].iloc[0]
    values = json.loads(token)
    if slot < 6:
        prefix = f'Reagent{slot + 1}'
        name, identity, dose = (row[prefix + '_Standardized Name'], values[0], values[1])
        cas = row[prefix + '_HTE_CAS']
        unit = 'equiv'
    elif slot == 6:
        name, identity, dose, unit, cas = (row['Standardized Name'], values[0], '', '', '')
    else:
        name, identity, dose, unit, cas = (values[0], values[0], values[0], 'C' if slot == 7 else 'h', '')
    return {'option_id': option_key(slot, token), 'slot': field, 'token': token, 'option_name': name or '(empty)', 'component_id': identity, 'dose': dose, 'dose_unit': unit, 'cas': cas}

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    certs = read(R28 / 'all_strong_scope_reconciliation20.csv.gz')
    groups = read(SRC / 'complete_group_registry.csv.gz').set_index('group_id', drop=False)
    contexts = read(SRC / 'complete_context_registry.csv.gz').set_index('context_id', drop=False)
    cond = read(SRC / 'full_condition_registry.csv.gz').drop_duplicates(['dataset_id', 'recipe_id'])
    condmaps = {d: z.set_index('recipe_id', drop=False) for d, z in cond.groupby('dataset_id')}
    supports = read(SRC / 'complete_support_registry.csv.gz').set_index('support_id', drop=False)
    values = {b: z.set_index('recipe_id').yield_mean.to_dict() for b, z in read(SRC / 'full_source_block_node_yields.csv.gz').groupby('block_id')}
    single_rows, pair_rows, triple_rows, summary, route_rows, node_rows, edge_rows, context_rows = ([], [], [], [], [], [], [], [])
    used_conditions, option_registry, snapshots = ({}, {}, {})
    for counter, source in enumerate(certs.to_dict('records')):
        gid, start, goal = (int(source[k]) for k in ['group_id', 'node0', 'node1'])
        cid = f'g{gid}_n{start}_n{goal}'
        g = groups.loc[gid]
        recipes = json.loads(supports.loc[g.support_id, 'recipe_ids'])
        conditions = condmaps[g.dataset_id].loc[recipes]
        tok = conditions[SLOTS].to_numpy(str)
        adjacency = (tok[:, None] != tok[None]).sum(2) == 1
        neighbors = [np.flatnonzero(row) for row in adjacency]
        cids = json.loads(g.context_ids)
        yy = np.array([[[values[contexts.loc[c, b]][r] for r in recipes] for b in ['block_a', 'block_b']] for c in cids])
        weights = yy.min((0, 1))
        baseline = (tok == tok[start]) | (tok == tok[goal])
        base_mask = baseline.all(1)
        kbase, base_route = widest(weights, neighbors, start, goal, base_mask)
        kfull, full_route = widest(weights, neighbors, start, goal, np.ones(len(tok), bool))
        assert abs(kbase - source['K_endpoint_all']) < 1e-14 and abs(kfull - source['K_all']) < 1e-14
        options = [(slot, token) for slot in range(len(SLOTS)) for token in sorted(set(tok[:, slot])) if token not in [tok[start, slot], tok[goal, slot]]]
        candidates, deletions, suff = ({}, {}, [])
        for slot, token in options:
            desc = option_description(slot, token, conditions)
            option_registry[g.dataset_id, slot, token] = {'dataset_id': g.dataset_id, **desc}
            expanded = baseline.copy()
            expanded[:, slot] |= tok[:, slot] == token
            mask = expanded.all(1)
            ksingle, route = widest(weights, neighbors, start, goal, mask)
            delete = tok[:, slot] != token
            kdelete, delroute = widest(weights, neighbors, start, goal, delete)
            candidates[slot, token] = (ksingle, route, mask)
            deletions[slot, token] = (kdelete, delroute)
            if ksingle >= H:
                suff.append((slot, token))
            single_rows.append({'certificate_id': cid, 'group_id': gid, 'node0': start, 'node1': goal, 'dataset_id': g.dataset_id, 'rxn_class': g.rxn_class, 'molecular_pair_id': g.molecular_pair_id, 'complete_endpoint_corners': source['complete_endpoint_corners'], 'native_pair_supports_identical': source['native_pair_supports_identical'], **desc, 'endpoint_changing_slot': bool(tok[start, slot] != tok[goal, slot]), 'endpoint_token0': tok[start, slot], 'endpoint_token1': tok[goal, slot], 'n_single_allowed_nodes': int(mask.sum()), 'n_full_nodes_with_option': int((~delete).sum()), 'K_single': ksingle, 'single_sufficient20': ksingle >= H, 'K_after_full_deletion': kdelete, 'necessary_in_full_graph20': kdelete < H, 'single_route_nodes': json.dumps(route), 'deletion_route_nodes': json.dumps(delroute)})
        pair_suff, pair_best = ([], -np.inf)
        if not suff:
            for first, second in itertools.combinations(options, 2):
                expanded = baseline.copy()
                for slot, token in [first, second]:
                    expanded[:, slot] |= tok[:, slot] == token
                mask = expanded.all(1)
                kpair, route = widest(weights, neighbors, start, goal, mask)
                pair_best = max(pair_best, kpair)
                if kpair >= H:
                    pair_suff.append((first, second, kpair, route))
                pair_rows.append({'certificate_id': cid, 'group_id': gid, 'node0': start, 'node1': goal, 'option1_id': option_key(*first), 'option2_id': option_key(*second), 'slot1': SLOTS[first[0]], 'token1': first[1], 'slot2': SLOTS[second[0]], 'token2': second[1], 'n_pair_allowed_nodes': int(mask.sum()), 'K_pair': kpair, 'pair_sufficient20': kpair >= H, 'route_nodes': json.dumps(route)})
        triple_suff, triple_best = ([], -np.inf)
        if not suff and (not pair_suff):
            for three in itertools.combinations(options, 3):
                expanded = baseline.copy()
                for slot, token in three:
                    expanded[:, slot] |= tok[:, slot] == token
                mask = expanded.all(1)
                ktriple, route = widest(weights, neighbors, start, goal, mask)
                triple_best = max(triple_best, ktriple)
                if ktriple >= H:
                    triple_suff.append((*three, ktriple, route))
                triple_rows.append({'certificate_id': cid, 'group_id': gid, 'node0': start, 'node1': goal, **{f'option{k + 1}_id': option_key(*o) for k, o in enumerate(three)}, **{f'slot{k + 1}': SLOTS[o[0]] for k, o in enumerate(three)}, **{f'token{k + 1}': o[1] for k, o in enumerate(three)}, 'n_triple_allowed_nodes': int(mask.sum()), 'K_triple': ktriple, 'triple_sufficient20': ktriple >= H, 'route_nodes': json.dumps(route)})
        necessary = [option_key(*o) for o in options if deletions[o][0] < H]
        minexact = 1 if suff else 2 if pair_suff else 3 if triple_suff else ''
        summary.append({**source, 'certificate_id': cid, 'molecule_a': contexts.loc[cids[0], 'molecule_a'], 'molecule_b': contexts.loc[cids[0], 'molecule_b'], 'partner_smiles': json.dumps(contexts.loc[cids, 'partner_smiles'].tolist()), 'context_ids': g.context_ids, 'n_extra_options': len(options), 'n_single_sufficient': len(suff), 'n_necessary_options_in_full_graph': len(necessary), 'necessary_option_ids': json.dumps(necessary), 'single_sufficient_option_ids': json.dumps([option_key(*o) for o in suff]), 'minimum_extra_options_exact': minexact, 'minimum_extra_options_lower_bound': minexact or 4, 'all_pairs_tested': not bool(suff), 'n_pairs_tested': len(options) * (len(options) - 1) // 2 if not suff else 0, 'n_sufficient_pairs': len(pair_suff), 'best_pair_K': pair_best if not suff else '', 'all_triples_tested': not suff and (not pair_suff), 'n_triples_tested': len(options) * (len(options) - 1) * (len(options) - 2) // 6 if not suff and (not pair_suff) else 0, 'n_sufficient_triples': len(triple_suff), 'best_triple_K': triple_best if not suff and (not pair_suff) else ''})
        snapshots[cid] = (source, g, tok, conditions, recipes, yy, cids, neighbors, candidates, deletions, pair_suff, triple_suff, full_route)
        if (counter + 1) % 50 == 0:
            print(json.dumps({'certificates': counter + 1, 'single_states': len(single_rows), 'pair_states': len(pair_rows), 'seconds': time.time() - started}), flush=True)
    single = pd.DataFrame(single_rows)
    pairs = pd.DataFrame(pair_rows)
    summ = pd.DataFrame(summary)
    save(single, 'all_single_options.csv.gz')
    save(pairs, 'all_pair_options.csv.gz')
    save(triple_rows, 'all_triple_options.csv.gz')
    save(summ, 'certificate_option_summary.csv')
    save(list(option_registry.values()), 'named_option_registry.csv')
    save(groups.reset_index(drop=True), 'complete_opportunity_registry.csv.gz')
    scope = []
    support_options = defaultdict(set)
    for su in supports.to_dict('records'):
        dataset = su['dataset_id']
        rr = json.loads(su['recipe_ids'])
        if rr:
            tt = condmaps[dataset].loc[rr, SLOTS].to_numpy(str)
            for slot in range(len(SLOTS)):
                for token in set(tt[:, slot]):
                    support_options[dataset, slot, token].add(su['support_id'])
    for (dataset, slot, token), desc in option_registry.items():
        option_supports = support_options[dataset, slot, token]
        measured = groups[(groups.dataset_id == dataset) & groups.support_id.isin(option_supports)]
        tested = single[(single.dataset_id == dataset) & (single.option_id == desc['option_id'])]
        good = tested[tested.single_sufficient20]
        need = tested[tested.necessary_in_full_graph20]
        scope.append({**desc, 'n_all_opportunities_same_dataset': int((groups.dataset_id == dataset).sum()), 'n_fixed_opportunities_option_measured': len(measured), 'n_measured_molecular_pairs': measured.molecular_pair_id.nunique(), 'n_strong_endpoint_certificates_tested': len(tested), 'n_tested_triples': tested.group_id.nunique(), 'n_tested_molecular_pairs': tested.molecular_pair_id.nunique(), 'n_single_sufficient_certificates': len(good), 'n_sufficient_triples': good.group_id.nunique(), 'n_sufficient_molecular_pairs': good.molecular_pair_id.nunique(), 'n_full_graph_necessary_certificates': len(need), 'n_necessary_triples': need.group_id.nunique(), 'n_necessary_molecular_pairs': need.molecular_pair_id.nunique(), 'n_complete_sufficient': int(good.complete_endpoint_corners.sum()), 'n_incomplete_sufficient': int((~good.complete_endpoint_corners).sum()), 'n_equal_native_support_sufficient': int(good.native_pair_supports_identical.sum())})
    save(scope, 'named_option_descriptive_scope.csv')
    class_rows = []
    inventory = read(SRC / 'complete_20_dataset_18_class_support.csv')[['dataset_id', 'rxn_class']].drop_duplicates()
    for inv in inventory.to_dict('records'):
        for complete in ['all', 'complete', 'incomplete']:
            z = summ[summ.dataset_id == inv['dataset_id']]
            if complete != 'all':
                z = z[z.complete_endpoint_corners == (complete == 'complete')]
            ss = single[single.certificate_id.isin(z.certificate_id)]
            class_rows.append({**inv, 'endpoint_corner_stratum': complete, 'n_original_opportunity_triples': int((groups.dataset_id == inv['dataset_id']).sum()), 'n_strong_certificates': len(z), 'n_triples': z.group_id.nunique(), 'n_molecular_pairs': z.molecular_pair_id.nunique(), 'n_minimum1': int((z.minimum_extra_options_exact == 1).sum()), 'n_minimum2': int((z.minimum_extra_options_exact == 2).sum()), 'n_minimum3': int((z.minimum_extra_options_exact == 3).sum()), 'n_minimum_at_least4': int((z.minimum_extra_options_lower_bound == 4).sum()), 'n_multiple_singleton_alternatives': int((z.n_single_sufficient >= 2).sum()), 'n_with_full_graph_necessary_option': int((z.n_necessary_options_in_full_graph >= 1).sum()), 'n_tested_options': len(ss), 'n_failed_single_options': int((~ss.single_sufficient20).sum())})
    save(class_rows, 'complete_class_option_summary.csv')
    selected = [summ[summ.group_id == 22733].iloc[0].certificate_id, summ[(summ.group_id == 110) & (summ.node0 == 5) & (summ.node1 == 38)].iloc[0].certificate_id]
    need2 = summ[summ.minimum_extra_options_exact == 2].sort_values(['complete_endpoint_corners', 'two_sided_clearance'], ascending=False)
    if len(need2):
        selected.append(need2.iloc[0].certificate_id)
    need3 = summ[summ.minimum_extra_options_exact == 3].sort_values(['complete_endpoint_corners', 'two_sided_clearance'], ascending=False)
    if len(need3):
        selected.append(need3.iloc[0].certificate_id)
    unique_optional = summ[(summ.n_single_sufficient == 1) & (summ.n_necessary_options_in_full_graph == 0)]
    if len(unique_optional):
        selected.append(unique_optional.iloc[0].certificate_id)
    selected = list(dict.fromkeys(selected))
    for cid in selected:
        source, g, tok, conditions, recipes, yy, cids, neighbors, candidates, deletions, pair_suff, triple_suff, full_route = snapshots[cid]
        start, goal = (int(source['node0']), int(source['node1']))
        routes = [('full_shared', '', '', source['K_all'], full_route)]
        for o, (k, rr, mask) in candidates.items():
            if k >= H:
                routes.append(('single_option', option_key(*o), '', k, rr))
        for o, (k, rr) in deletions.items():
            if k < H or candidates[o][0] >= H or (g.group_id == 110 and option_description(*o, conditions)['option_name'] == 'HOAT'):
                routes.append(('delete_option_from_full', option_key(*o), '', k, rr))
        if pair_suff:
            first, second, k, rr = max(pair_suff, key=lambda x: x[2])
            routes.append(('two_option', option_key(*first), option_key(*second), k, rr))
        if triple_suff:
            first, second, third, k, rr = max(triple_suff, key=lambda x: x[3])
            routes.append(('three_option', option_key(*first), option_key(*second), k, rr))
        weights = yy.min((0, 1))
        for kind, oid1, oid2, k, route in routes:
            for step, node in enumerate(route):
                route_rows.append({'certificate_id': cid, 'group_id': g.group_id, 'route_kind': kind, 'option1_id': oid1, 'option2_id': oid2, 'step': step, 'node': node, 'option3_id': option_key(*third) if kind == 'three_option' else '', 'recipe_id': recipes[node], 'shared_node_yield': weights[node], 'optimized_bottleneck': k})
        for p, c in enumerate(cids):
            context_rows.append({'certificate_id': cid, **contexts.loc[c].to_dict()})
            for n, recipe in enumerate(recipes):
                node_rows.append({'certificate_id': cid, 'group_id': g.group_id, 'partner_position': p, 'context_id': c, 'node': n, 'recipe_id': recipe, 'block_a': contexts.loc[c, 'block_a'], 'block_b': contexts.loc[c, 'block_b'], 'yield_a': yy[p, 0, n], 'yield_b': yy[p, 1, n]})
        for a, row in enumerate(neighbors):
            for b in row:
                if a < b:
                    edge_rows.append({'certificate_id': cid, 'node0': a, 'node1': b})
        for r in conditions.to_dict('records'):
            used_conditions[cid, r['recipe_id']] = {'certificate_id': cid, **r}
    save(route_rows, 'chemical_witness_routes.csv')
    save(node_rows, 'chemical_witness_full_node_yields.csv')
    save(context_rows, 'chemical_witness_acquisition_contexts.csv')
    save(edge_rows, 'chemical_witness_edges.csv')
    save(list(used_conditions.values()), 'chemical_witness_conditions.csv')
    witness_provenance = []
    sources = [(selected[0], R28), (selected[1], R28 / 'large_gain_witness')]
    for cid, path in sources:
        existing = read(path / 'chemical_witness_source_records.csv.gz')
        existing.insert(0, 'certificate_id', cid)
        witness_provenance.append(existing)
    if len(selected) > 2:
        nodes = pd.DataFrame(node_rows)
        new_nodes = nodes[nodes.certificate_id.isin(selected[2:])]
        blocks = set(new_nodes.block_a) | set(new_nodes.block_b)
        recipes = set(new_nodes.recipe_id)
        path = ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
        for chunk in pd.read_csv(path, keep_default_na=False, float_precision='round_trip', chunksize=150000):
            relevant = chunk[chunk.block_id.isin(blocks) & chunk.recipe_id.isin(recipes)]
            for cid in selected[2:]:
                nn = new_nodes[new_nodes.certificate_id == cid]
                bb = set(nn.block_a) | set(nn.block_b)
                rr = set(nn.recipe_id)
                part = relevant[relevant.block_id.isin(bb) & relevant.recipe_id.isin(rr)].copy()
                if len(part):
                    part.insert(0, 'certificate_id', cid)
                    witness_provenance.append(part)
    save(pd.concat(witness_provenance, ignore_index=True), 'chemical_witness_source_records.csv.gz')
    manifest = {}
    for path in [SRC / n for n in ['complete_group_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_condition_registry.csv.gz', 'full_source_block_node_yields.csv.gz', 'complete_20_dataset_18_class_support.csv']] + [R28 / 'all_strong_scope_reconciliation20.csv.gz', R28 / 'chemical_witness_source_records.csv.gz', R28 / 'chemical_witness_source_verification.json', R28 / 'large_gain_witness/chemical_witness_source_records.csv.gz', R28 / 'large_gain_witness/chemical_witness_source_verification.json', ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz', Path(__file__)]:
        with path.open('rb') as stream:
            manifest[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    dump(manifest, 'source_manifest.json')
    dump({'reused_raw_verified_witnesses': {selected[0]: str(R28 / 'chemical_witness_source_verification.json'), selected[1]: str(R28 / 'large_gain_witness/chemical_witness_source_verification.json')}, 'new_witnesses_with_all_record_block_condition_pointers': selected[2:], 'new_witness_raw_workbook_readback': 'separate root-level verification; not claimed by this generator'}, 'witness_source_status.json')
    result = {'status': 'analyzed', 'n_certificates': len(summ), 'n_single_options': len(single), 'n_single_sufficient': int(single.single_sufficient20.sum()), 'n_full_graph_necessary_states': int(single.necessary_in_full_graph20.sum()), 'n_pair_options': len(pairs), 'n_pair_sufficient': int(pairs.pair_sufficient20.sum()) if len(pairs) else 0, 'n_triple_options': len(triple_rows), 'n_triple_sufficient': sum((r['triple_sufficient20'] for r in triple_rows)), 'minimum_option_counts': summ.minimum_extra_options_exact.value_counts().to_dict(), 'witness_certificates': selected, 'all_opportunity_registry_rows': len(groups), 'R28_complete_corner_bundle_feasible_endpoint_denominator20': 489953, 'seconds': time.time() - started}
    dump(result, 'run_summary.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
