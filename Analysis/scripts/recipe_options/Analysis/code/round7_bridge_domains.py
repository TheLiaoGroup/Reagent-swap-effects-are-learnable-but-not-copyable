"""Complete six-recipe panels identify molecular domains of bridge components."""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from round2_near_neighbors import exact_neighbors, canonical, parsed
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round7/bridge_domains'
SLOTS = [f'Reagent{i}' for i in range(1, 7)] + ['Solvent', 'Temperature', 'Time']
TOKENS = [f'{x}_token' for x in SLOTS]
BLOCK = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
READ = dict(keep_default_na=False, low_memory=False)

def read(path, **kwargs):
    return pd.read_csv(ROOT / path, **READ, **kwargs)

def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def sid(value):
    return sha256(packed(value).encode()).hexdigest()[:18]

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression='gzip' if name.endswith('.gz') else None)

def label(row, slot):
    if slot.startswith('Reagent'):
        name = row[f'{slot}_Standardized Name'] or 'none'
        return name + (f" {row[f'{slot}_HTE_Equiv']} eq" if name != 'none' else '')
    return {'Solvent': row['Standardized Name'], 'Temperature': f"{row['temp_id']} C", 'Time': f"{row['time_id']} h"}[slot]

def enumerate_panels(registry, classes):
    panels = []
    for dataset, group in registry.groupby('dataset_id', sort=True):
        for first, second in combinations(range(9), 2):
            fixed = [x for x in range(9) if x not in (first, second)]
            buckets = defaultdict(list)
            for row in group[['recipe_id', *TOKENS]].itertuples(index=False, name=None):
                values = row[1:]
                buckets[tuple((values[k] for k in fixed))].append((row[0], values))
            for background, block in sorted(buckets.items()):
                cell = {(values[first], values[second]): recipe for recipe, values in block}
                levels_first = sorted({a for a, b in cell})
                levels_second = sorted({b for a, b in cell})
                for ternary_axis, binary_axis, ternary, binary, transpose in [(first, second, levels_first, levels_second, False), (second, first, levels_second, levels_first, True)]:

                    def recipe(t, b):
                        return cell.get((b, t) if transpose else (t, b))
                    for binary_levels in combinations(binary, 2):
                        common = [t for t in ternary if all((recipe(t, b) for b in binary_levels))]
                        for ternary_levels in combinations(common, 3):
                            recipes = [recipe(t, b) for t in ternary_levels for b in binary_levels]
                            info = dict(dataset_id=dataset, rxn_class=classes[dataset], ternary_slot=SLOTS[ternary_axis], binary_slot=SLOTS[binary_axis], ternary_tokens=packed(list(ternary_levels)), binary_tokens=packed(list(binary_levels)), recipe_ids=packed(recipes), fixed_background=packed({SLOTS[k]: json.loads(v) for k, v in zip(fixed, background)}))
                            info['panel_id'] = 'six_' + sid([dataset, SLOTS[ternary_axis], SLOTS[binary_axis], recipes])
                            panels.append(info)
    result = pd.DataFrame(panels)
    assert result.panel_id.is_unique
    return result

def route_registry(panels, registry):
    lookup = registry.set_index(['dataset_id', 'recipe_id']).to_dict('index')
    rows = []
    for panel in panels.to_dict('records'):
        cells = np.asarray(json.loads(panel['recipe_ids'])).reshape(3, 2)
        for endpoint_levels in combinations(range(3), 2):
            middle = next((i for i in range(3) if i not in endpoint_levels))
            left, right = endpoint_levels
            for diagonal in (0, 1):
                binary = (diagonal, 1 - diagonal)
                ordered = [cells[left, binary[0]], cells[right, binary[0]], cells[left, binary[1]], cells[right, binary[1]], cells[middle, binary[0]], cells[middle, binary[1]]]
                row = dict(panel_id=panel['panel_id'], dataset_id=panel['dataset_id'], rxn_class=panel['rxn_class'], ternary_slot=panel['ternary_slot'], binary_slot=panel['binary_slot'], endpoint_level0=left, endpoint_level1=right, middle_level=middle, diagonal=diagonal, recipe_order=packed(ordered), panel_positions=packed([list(cells.ravel()).index(x) for x in ordered]), endpoint0_component=label(lookup[panel['dataset_id'], ordered[0]], panel['ternary_slot']), endpoint1_component=label(lookup[panel['dataset_id'], ordered[3]], panel['ternary_slot']), bridge_component=label(lookup[panel['dataset_id'], ordered[4]], panel['ternary_slot']), binary0_component=label(lookup[panel['dataset_id'], ordered[0]], panel['binary_slot']), binary1_component=label(lookup[panel['dataset_id'], ordered[3]], panel['binary_slot']))
                row['route_id'] = 'bridge_' + sid([panel['panel_id'], ordered])
                rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == 6 * len(panels) and frame.route_id.is_unique
    return frame

def measurements(mapping, panels):
    fields = ['record_id', *BLOCK, 'recipe_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'yield_numeric', 'strict_eligible', 'condition_id']
    records = read('data/normalized_records.csv.gz', usecols=fields)
    records = records[records.dataset_id.isin(mapping.dataset_id) & records.strict_eligible.astype(str).str.lower().eq('true')].copy()
    records['yield_numeric'] = pd.to_numeric(records.yield_numeric, errors='raise')
    assert np.isfinite(records.yield_numeric).all()
    identity = records[['dataset_id', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].drop_duplicates()
    assert not identity.duplicated(['dataset_id', 'target_key']).any()
    nodes = records.groupby([*BLOCK, 'recipe_id'], as_index=False, sort=False).agg(yield_numeric=('yield_numeric', 'mean'), n_source_rows=('record_id', 'size'))
    source_lookup = nodes.set_index([*BLOCK, 'recipe_id']).yield_numeric
    complete = []
    for dataset, subset in panels.groupby('dataset_id', sort=True):
        source = nodes[nodes.dataset_id.eq(dataset)]
        matrix = source.pivot(index=BLOCK, columns='recipe_id', values='yield_numeric')
        for panel in subset.itertuples(index=False):
            recipes = json.loads(panel.recipe_ids)
            six = matrix.reindex(columns=recipes)
            six = six[six.notna().all(axis=1)]
            if six.empty:
                continue
            frame = six.reset_index()
            frame.columns = [*BLOCK, *[f'y{i}' for i in range(6)]]
            frame['panel_id'] = panel.panel_id
            frame['acquisition_panel_id'] = ['ab_' + sid(list(v) + [panel.panel_id]) for v in frame[BLOCK].itertuples(index=False, name=None)]
            complete.append(frame)
        print('Complete six-grid acquisitions', dataset, sum((len(x) for x in complete if x.dataset_id.iloc[0] == dataset)), flush=True)
    acquisition = pd.concat(complete, ignore_index=True)
    target = acquisition.groupby(['dataset_id', 'target_key', 'panel_id'], sort=False, as_index=False).agg(**{f'y{i}': (f'y{i}', 'mean') for i in range(6)}, n_complete_acquisition_blocks=('acquisition_panel_id', 'size'))
    target = target.merge(identity, on=['dataset_id', 'target_key'], validate='many_to_one')
    target['target_panel_id'] = ['tp_' + sid(v) for v in target[['dataset_id', 'target_key', 'panel_id']].itertuples(index=False, name=None)]
    save(acquisition, 'all_complete_acquisition_six_yields.csv.gz')
    save(target, 'all_complete_target_six_yields.csv.gz')
    return (records, acquisition, target, identity, source_lookup)

def route_metrics(y):
    endpoint = np.minimum(y[:, 0], y[:, 3])
    direct = np.minimum(endpoint, np.maximum(y[:, 1], y[:, 2]))
    bridge = np.minimum(endpoint, np.minimum(y[:, 4], y[:, 5]))
    gain = bridge - direct
    blocked = (endpoint >= 0.2) & (direct < 0.2)
    connects = blocked & (bridge >= 0.2)
    margin = np.minimum(bridge - 0.2, 0.2 - direct)
    return dict(endpoint_minimum=endpoint, direct_bottleneck=direct, bridge_bottleneck=bridge, bridge_gain=gain, event_margin=margin, endpoint_both20=endpoint >= 0.2, direct_two_orders_blocked20=blocked, bridge_connects20=connects, bridge_connects20_gain5=connects & (gain >= 0.05), bridge_connects20_gain10=connects & (gain >= 0.1), bridge_connects20_margin5=connects & (margin >= 0.05), bridge_connects20_margin10=connects & (margin >= 0.1), strict_two_mixed_le5=connects & (y[:, 1] <= 0.05) & (y[:, 2] <= 0.05))

def evaluate_routes(target, routes):
    rows = []
    grouped = {key: part for key, part in target.groupby('panel_id', sort=False)}
    for route in routes.itertuples(index=False):
        part = grouped.get(route.panel_id)
        if part is None:
            continue
        y = part[[f'y{i}' for i in json.loads(route.panel_positions)]].to_numpy(float)
        frame = part[['dataset_id', 'target_key', 'panel_id', 'target_panel_id', 'n_complete_acquisition_blocks', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].copy()
        frame['route_id'] = route.route_id
        for i in range(6):
            frame[f'y{i}'] = y[:, i]
        for key, value in route_metrics(y).items():
            frame[key] = value
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result['route_target_id'] = ['rt_' + sid(v) for v in result[['dataset_id', 'target_key', 'route_id']].itertuples(index=False, name=None)]
    result = result.merge(routes[['route_id', 'rxn_class', 'ternary_slot', 'binary_slot', 'bridge_component']], on='route_id', validate='many_to_one')
    assert result.route_target_id.is_unique
    save(result, 'all_target_bridge_routes.csv.gz')
    return result

def exact_pairs(identity):
    all_relations = []
    for dataset, subset in identity.groupby('dataset_id', sort=True):
        for role in ['sub_1_smiles', 'sub_2_smiles']:
            values = []
            for value in subset[role].unique():
                molecule = parsed(value)
                if molecule is not None and value:
                    values.append(canonical(molecule))
            relations = exact_neighbors(set(values), role)
            if relations.empty:
                continue
            relations['dataset_id'] = dataset
            relations['role'] = role
            all_relations.append(relations)
    frame = pd.concat(all_relations, ignore_index=True)
    save(frame, 'all_dataset_exact_molecular_edits.csv')
    return frame

def molecular_domains(results, identity):
    canonical_lookup = {}
    for value in sorted(set(identity.sub_1_smiles) | set(identity.sub_2_smiles)):
        mol = parsed(value)
        canonical_lookup[value] = canonical(mol) if value and mol is not None else ''
    results = results.copy()
    for role in ['sub_1_smiles', 'sub_2_smiles']:
        results['canonical_' + role] = results[role].map(canonical_lookup)
    relations = exact_pairs(identity)
    all_rows = []
    for dataset, relation_dataset in relations.groupby('dataset_id', sort=True):
        data = results[results.dataset_id.eq(dataset)]
        for role, relation_role in relation_dataset.groupby('role', sort=False):
            partner = 'sub_2_smiles' if role == 'sub_1_smiles' else 'sub_1_smiles'
            left = data.rename(columns={'canonical_' + role: 'molecule_a', 'canonical_' + partner: 'partner_smiles'})
            left = left[left.partner_smiles.ne('')]
            right = left.rename(columns={'molecule_a': 'molecule_b'})
            left = left.merge(relation_role.drop(columns=['dataset_id', 'role']), on='molecule_a', validate='many_to_many')
            key = ['route_id', 'partner_smiles', 'molecule_b']
            joined = left.merge(right, on=key, suffixes=('_a', '_b'), validate='many_to_many')
            if joined.empty:
                continue
            base = joined[['dataset_id_a', 'rxn_class_a', 'route_id', 'partner_smiles', 'molecule_a', 'molecule_b', 'structural_pair_id', 'graph_method', 'transformation', 'common_core', 'fragment_a', 'fragment_b', 'target_key_a', 'target_key_b', 'route_target_id_a', 'route_target_id_b']].copy()
            base = base.rename(columns={'dataset_id_a': 'dataset_id', 'rxn_class_a': 'rxn_class'})
            base['role'] = role
            for field in [*[f'y{i}' for i in range(6)], 'endpoint_minimum', 'direct_bottleneck', 'bridge_bottleneck', 'bridge_gain', 'event_margin', 'endpoint_both20', 'direct_two_orders_blocked20', 'bridge_connects20', 'n_complete_acquisition_blocks']:
                for side in ('a', 'b'):
                    base[field + '_' + side] = joined[field + '_' + side]
            base['both_endpoint_pairs20'] = base.endpoint_both20_a & base.endpoint_both20_b
            base['both_direct_orders_blocked20'] = base.direct_two_orders_blocked20_a & base.direct_two_orders_blocked20_b
            base['opposite_bridge_available20'] = base.both_endpoint_pairs20 & base.bridge_bottleneck_a.ge(0.2).ne(base.bridge_bottleneck_b.ge(0.2))
            base['opposite_bridge20'] = base.both_endpoint_pairs20 & base.bridge_connects20_a.ne(base.bridge_connects20_b) & base.both_direct_orders_blocked20
            base['bridge_difference'] = base.bridge_bottleneck_b - base.bridge_bottleneck_a
            base['bridge_switch_margin'] = np.minimum(np.maximum(base.bridge_bottleneck_a, base.bridge_bottleneck_b) - 0.2, 0.2 - np.minimum(base.bridge_bottleneck_a, base.bridge_bottleneck_b))
            base['opposite_bridge20_margin5'] = base.opposite_bridge20 & base.bridge_switch_margin.ge(0.05)
            base['opposite_bridge20_margin10'] = base.opposite_bridge20 & base.bridge_switch_margin.ge(0.1)
            base['opposite_bridge_available20_margin5'] = base.opposite_bridge_available20 & base.bridge_switch_margin.ge(0.05)
            base['opposite_bridge_available20_margin10'] = base.opposite_bridge_available20 & base.bridge_switch_margin.ge(0.1)
            base['same_specified_bridge_path_switch'] = base.opposite_bridge_available20
            base['same_specified_bridge_path_switch_margin5'] = base.opposite_bridge_available20_margin5
            base['same_specified_bridge_path_switch_margin10'] = base.opposite_bridge_available20_margin10
            base['comparison_id'] = ['domain_' + sid(v) for v in base[['route_id', 'role', 'target_key_a', 'target_key_b']].itertuples(index=False, name=None)]
            all_rows.append(base)
        print('Exact bridge domain comparisons', dataset, sum((len(x) for x in all_rows if x.dataset_id.iloc[0] == dataset)), flush=True)
    frame = pd.concat(all_rows, ignore_index=True)
    assert frame.comparison_id.is_unique
    save(frame, 'all_exact_edit_partner_comparisons.csv.gz')
    return frame

def summarize(mapping, registry, panels, acquisition, target, results, domains):
    swaps = read('results/matched_swap_summary.csv')
    assert len(swaps) == 4002
    rows = []
    for dataset in mapping.dataset_id:
        p = panels[panels.dataset_id.eq(dataset)]
        a = acquisition[acquisition.dataset_id.eq(dataset)]
        t = target[target.dataset_id.eq(dataset)]
        r = results[results.dataset_id.eq(dataset)]
        d = domains[domains.dataset_id.eq(dataset)]
        row = dict(dataset_id=dataset, rxn_class=mapping.set_index('dataset_id').loc[dataset, 'rxn_class'], n_original_swaps=int(swaps.dataset_id.eq(dataset).sum()), n_registered_recipes=int(registry.dataset_id.eq(dataset).sum()), n_complete_registered_six_panels=len(p), n_complete_acquisition_panels=len(a), n_target_panels=len(t), n_target_routes=len(r), n_dataset_targets=r.target_key.nunique(), n_exact_edit_partner_comparisons=len(d), status='no_registered_complete_3x2_panel' if p.empty else 'no_complete_measured_six_grid' if t.empty else 'complete_six_grids_available')
        for field in ['endpoint_both20', 'direct_two_orders_blocked20', 'bridge_connects20', 'bridge_connects20_gain5', 'bridge_connects20_gain10', 'bridge_connects20_margin5', 'bridge_connects20_margin10', 'strict_two_mixed_le5']:
            row['n_' + field] = int(r[field].sum()) if not r.empty else 0
        for field in ['opposite_bridge20', 'opposite_bridge20_margin5', 'opposite_bridge20_margin10', 'opposite_bridge_available20', 'opposite_bridge_available20_margin5', 'opposite_bridge_available20_margin10']:
            row['n_exact_' + field] = int(d[field].sum()) if not d.empty else 0
        rows.append(row)
    coverage = pd.DataFrame(rows)
    save(coverage, 'dataset_coverage.csv')
    bridge = results[results.bridge_connects20]
    counts = results.groupby(['route_id', 'dataset_id', 'rxn_class', 'bridge_component'], as_index=False).agg(n_targets=('target_key', 'nunique'), n_endpoint_pairs20=('endpoint_both20', 'sum'), n_direct_orders_blocked20=('direct_two_orders_blocked20', 'sum'), n_bridge_connects20=('bridge_connects20', 'sum'), n_bridge_margin5=('bridge_connects20_margin5', 'sum'), n_bridge_margin10=('bridge_connects20_margin10', 'sum'), median_bridge_gain=('bridge_gain', 'median'), maximum_bridge_gain=('bridge_gain', 'max'))
    counts['n_blocked_not_bridged20'] = counts.n_direct_orders_blocked20 - counts.n_bridge_connects20
    counts['same_bridge_usable_and_fails'] = counts.n_bridge_connects20.gt(0) & counts.n_blocked_not_bridged20.gt(0)
    save(counts, 'bridge_recipe_domain_summary.csv')
    aggregate = dict(n_datasets=20, n_reaction_classes=18, n_original_swaps=4002, n_panels=len(panels), n_panel_datasets=panels.dataset_id.nunique(), n_acquisition_panels=len(acquisition), n_target_panels=len(target), n_target_routes=len(results), n_dataset_targets=results[['dataset_id', 'target_key']].drop_duplicates().shape[0], n_connecting_target_routes=len(bridge), n_connecting_dataset_targets=bridge[['dataset_id', 'target_key']].drop_duplicates().shape[0], n_connecting_routes=bridge.route_id.nunique(), n_connecting_classes=bridge.rxn_class.nunique(), n_same_bridge_usable_and_fails=int(counts.same_bridge_usable_and_fails.sum()), n_exact_comparisons=len(domains), n_exact_switches=int(domains.opposite_bridge20.sum()), n_exact_switches_margin5=int(domains.opposite_bridge20_margin5.sum()), n_exact_switches_margin10=int(domains.opposite_bridge20_margin10.sum()))
    for field in ['opposite_bridge_available20', 'opposite_bridge_available20_margin5', 'opposite_bridge_available20_margin10']:
        aggregate['n_exact_' + field] = int(domains[field].sum())
    for field in ['opposite_bridge20', 'same_specified_bridge_path_switch', 'same_specified_bridge_path_switch_margin5', 'same_specified_bridge_path_switch_margin10']:
        selected = domains[domains[field]]
        aggregate[field] = dict(n_comparisons=len(selected), n_classes=selected.rxn_class.nunique(), n_datasets=selected.dataset_id.nunique(), n_dataset_edits=selected[['dataset_id', 'role', 'structural_pair_id']].drop_duplicates().shape[0], n_edit_partners=selected[['dataset_id', 'role', 'structural_pair_id', 'partner_smiles']].drop_duplicates().shape[0])
    for field in ['bridge_connects20_gain5', 'bridge_connects20_gain10', 'bridge_connects20_margin5', 'bridge_connects20_margin10', 'strict_two_mixed_le5']:
        aggregate['n_' + field] = int(results[field].sum())
    (OUT / 'run_summary.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding='utf-8')
    return (coverage, counts, aggregate)

def selected_examples(results, domains, routes, registry, records, acquisition):
    interesting = domains[domains.opposite_bridge_available20].copy()
    interesting['absolute_difference'] = interesting.bridge_difference.abs()
    chosen = interesting.sort_values(['opposite_bridge_available20_margin10', 'absolute_difference'], ascending=False).groupby('rxn_class', sort=False).head(1)
    extra = interesting[interesting.graph_method.ne('RDKit_single_cut_MMP')].sort_values('absolute_difference', ascending=False).drop_duplicates(['dataset_id', 'role', 'structural_pair_id']).head(12)
    extra = pd.concat([extra, interesting[interesting.opposite_bridge20]])
    chosen = pd.concat([chosen, extra]).drop_duplicates('comparison_id')
    chosen['case_id'] = ['bd_' + sid(x) for x in chosen.comparison_id]
    metadata = routes.drop(columns=['dataset_id', 'rxn_class'])
    chosen = chosen.merge(metadata, on='route_id', validate='many_to_one')
    save(chosen, 'selected_molecular_bridge_examples.csv')
    partner_keys = chosen[['route_id', 'role', 'structural_pair_id']].drop_duplicates()
    partners = domains.merge(partner_keys, on=['route_id', 'role', 'structural_pair_id'], validate='many_to_one')
    save(partners, 'selected_edits_all_comparable_partners.csv')
    cross = results[results.bridge_connects20].sort_values(['bridge_connects20_margin10', 'bridge_gain', 'endpoint_minimum'], ascending=False).groupby('rxn_class', sort=False).head(1)
    cross = cross.merge(routes.drop(columns=['dataset_id', 'rxn_class', 'panel_id', 'ternary_slot', 'binary_slot', 'bridge_component']), on='route_id', validate='many_to_one')
    cross['case_id'] = ['cross_' + sid(x) for x in cross.route_target_id]
    save(cross, 'selected_cross_reaction_bridge_examples.csv')
    condition_lookup = registry.set_index(['dataset_id', 'recipe_id']).to_dict('index')
    six_rows, source_rows = ([], [])
    for case in chosen.to_dict('records'):
        ordered = json.loads(case['recipe_order'])
        for side in ('a', 'b'):
            dataset, target_key = (case['dataset_id'], case['target_key_' + side])
            complete = acquisition[acquisition.dataset_id.eq(dataset) & acquisition.target_key.eq(target_key) & acquisition.panel_id.eq(case['panel_id'])]
            assert not complete.empty
            for i, recipe in enumerate(ordered):
                condition = condition_lookup[dataset, recipe]
                six_rows.append(dict(case_id=case['case_id'], comparison_id=case['comparison_id'], dataset_id=dataset, side=side, target_key=target_key, molecule=case['molecule_' + side], partner_smiles=case['partner_smiles'], cell=['00', '10', '01', '11', 'x0', 'x1'][i], recipe_id=recipe, yield_numeric=case[f'y{i}_{side}'], full_recipe_text='; '.join((f'{slot}={label(condition, slot)}' for slot in SLOTS)), full_recipe_tokens=packed({slot: json.loads(condition[slot + '_token']) for slot in SLOTS})))
                for block in complete.to_dict('records'):
                    mask = records.recipe_id.eq(recipe)
                    for field in BLOCK:
                        mask &= records[field].eq(block[field])
                    for record in records[mask].to_dict('records'):
                        source_rows.append(dict(case_id=case['case_id'], side=side, cell=['00', '10', '01', '11', 'x0', 'x1'][i], **record))
    for case in cross.to_dict('records'):
        ordered = json.loads(case['recipe_order'])
        complete = acquisition[acquisition.dataset_id.eq(case['dataset_id']) & acquisition.target_key.eq(case['target_key']) & acquisition.panel_id.eq(case['panel_id'])]
        for i, recipe in enumerate(ordered):
            condition = condition_lookup[case['dataset_id'], recipe]
            six_rows.append(dict(case_id=case['case_id'], comparison_id='', dataset_id=case['dataset_id'], side='single_target', target_key=case['target_key'], molecule=case['sub_1_smiles'], partner_smiles=case['sub_2_smiles'], cell=['00', '10', '01', '11', 'x0', 'x1'][i], recipe_id=recipe, yield_numeric=case[f'y{i}'], full_recipe_text='; '.join((f'{slot}={label(condition, slot)}' for slot in SLOTS)), full_recipe_tokens=packed({slot: json.loads(condition[slot + '_token']) for slot in SLOTS})))
            for block in complete.to_dict('records'):
                mask = records.recipe_id.eq(recipe)
                for field in BLOCK:
                    mask &= records[field].eq(block[field])
                for record in records[mask].to_dict('records'):
                    source_rows.append(dict(case_id=case['case_id'], side='single_target', cell=['00', '10', '01', '11', 'x0', 'x1'][i], **record))
    sources = pd.DataFrame(source_rows)
    raw = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    assert raw.record_id.is_unique
    sources = sources.merge(raw, on='record_id', validate='many_to_one')
    sources['source_absolute_path'] = [str(RAW_DIR / x) for x in sources.source_file]
    save(pd.DataFrame(six_rows), 'selected_all_six_yields_and_conditions.csv')
    save(sources, 'selected_original_records.csv.gz')
    requests = []
    for row in partners.to_dict('records'):
        for side in ('a', 'b'):
            requests.append(dict(comparison_id=row['comparison_id'], side=side, dataset_id=row['dataset_id'], target_key=row['target_key_' + side], route_id=row['route_id']))
    requests = pd.DataFrame(requests).merge(routes[['route_id', 'panel_id', 'recipe_order']], on='route_id', validate='many_to_one')
    requests['recipe_id'] = requests.recipe_order.map(json.loads)
    requests = requests.drop(columns='recipe_order').explode('recipe_id', ignore_index=True)
    requests = requests.merge(acquisition[[*BLOCK, 'panel_id']], on=['dataset_id', 'target_key', 'panel_id'], validate='many_to_many')
    partner_sources = requests.merge(records, on=[*BLOCK, 'recipe_id'], validate='many_to_many')
    partner_sources = partner_sources.merge(raw, on='record_id', validate='many_to_one')
    partner_sources['source_absolute_path'] = [str(RAW_DIR / x) for x in partner_sources.source_file]
    save(partner_sources, 'selected_partner_original_records.csv.gz')
    return (chosen, partners, cross)

def write_report(coverage, aggregate, chosen, partners, cross):
    switches = chosen[chosen.opposite_bridge20]
    switch_bottlenecks = switches[['bridge_bottleneck_a', 'bridge_bottleneck_b']].max(axis=1)
    lines = ['# Scope of a third component: from complete six cells to alternative paths changed by molecular structure', '', '## Chemical finding', '', f"Two recipes that each work can both fail when directly mixed, while a third component can keep a high-yield path of stepwise substitutions. These connections come from complete measured six cells and cover {aggregate['n_connecting_classes']} reaction classes; the third choice is not just a higher-yield recipe, it changes whether two working combinations can be reached from each other by single substitutions.", '', f"The full corpus retains 20 datasets, 18 reaction classes and 4,002 original single-slot swaps. Fully registered nine-slot recipes give {aggregate['n_panels']:,} complete 3×2 panels covering {aggregate['n_panel_datasets']} datasets; after complete matching within the same acquisition, {aggregate['n_acquisition_panels']:,} six-cell sets form {aggregate['n_target_panels']:,} target × panel rows and {aggregate['n_target_routes']:,} target × endpoint/relay/diagonal definitions.", '', f"With both endpoints at least 20% and both direct substitution paths below 20%, a third component connects {aggregate['n_connecting_target_routes']:,} target × route cases, involving {aggregate['n_connecting_dataset_targets']:,} deduplicated dataset+target, {aggregate['n_connecting_routes']:,} exact routes and {aggregate['n_connecting_classes']} reaction classes. {aggregate['n_bridge_connects20_margin5']} have a two-sided 5pp threshold margin, {aggregate['n_bridge_connects20_margin10']} have 10pp; in {aggregate['n_strict_two_mixed_le5']}, both original mixed combinations are at most 5%.", '', '## Actual relays across reactions', '', 'The table shows, for each class, the strong-margin instance with the largest bottleneck gain among complete six cells. Endpoints and the four-step bridge path are actual yields for the same target with the other seven slots identical.', '', '| Reaction | Relay choice | Best direct-path bottleneck % | Yields along relay path % (00→x0→x1→11) |', '|---|---|---:|---|']
    for row in cross.to_dict('records'):
        path = ' → '.join((f"{100 * row['y' + str(i)]:.2f}" for i in (0, 4, 5, 3)))
        lines.append(f"| {row['rxn_class']} | {row['bridge_component']} | {100 * row['direct_bottleneck']:.2f} | {path} |")
    lines += ['', 'Actual molecules, complete six cells, nine-slot conditions and sources for all instances are in selected_cross_reaction_bridge_examples.csv, selected_all_six_yields_and_conditions.csv and selected_original_records.csv.gz.', '', '## Can the same relay be used across molecules', '', f"Exact routes that both connect and fail on different targets: {aggregate['n_same_bridge_usable_and_fails']}. With the other reactant also fixed and only exact graph-edit molecule pairs compared, all {aggregate['n_exact_comparisons']:,} molecule edit × partner × route comparisons and their continuous bottlenecks are kept.", '', 'The two levels are counted separately:', '', '| Comparison definition | All comparisons | Two-sided 5pp threshold margin | Two-sided 10pp threshold margin |', '|---|---:|---:|---:|', f"| Both endpoints ≥20% for both molecules, both direct paths <20% for both, relay connects only one | {aggregate['n_exact_switches']} | {aggregate['n_exact_switches_margin5']} | {aggregate['n_exact_switches_margin10']} |", f"| Both endpoints ≥20% for both molecules, the same specified relay path ≥20% for one and <20% for the other | {aggregate['n_exact_opposite_bridge_available20']:,} | {aggregate['n_exact_opposite_bridge_available20_margin5']:,} | {aggregate['n_exact_opposite_bridge_available20_margin10']:,} |", '', f"The first level has {aggregate['n_exact_switches']} comparisons, in acylation and Kumada, with connected-side bottleneck {100 * switch_bottlenecks.min():.2f}–{100 * switch_bottlenecks.max():.2f}%, all kept in the display table. The second level addresses the molecular applicability of a specified relay path; the direct paths of the other molecule may still be high-yield, and direct-path bottlenecks on both sides are listed.", '', '### Representative exact structural changes', '', '| Reaction | Molecule A→B | Fixed partner | Same relay | Bridge bottleneck A/B % | Best direct bottleneck A/B % |', '|---|---|---|---|---:|---:|']
    representative = chosen.drop_duplicates('rxn_class')
    for row in representative.to_dict('records'):
        lines.append(f"| {row['rxn_class']} | `{row['molecule_a']}` → `{row['molecule_b']}` | `{row['partner_smiles']}` | {row['bridge_component']} | {100 * row['bridge_bottleneck_a']:.2f} / {100 * row['bridge_bottleneck_b']:.2f} | {100 * row['direct_bottleneck_a']:.2f} / {100 * row['direct_bottleneck_b']:.2f} |")
    lines += ['', f'For each displayed exact structural edit and the same route, all comparable partners are kept in selected_edits_all_comparable_partners.csv, a total of {len(partners):,} comparisons; these include complete responses where the bridge path works on both molecules, is low-yield on both, or works on one side only. When molecule and partner are not in the same acquisition, each side is first matched completely to six cells within its own acquisition.', '', '## Scientific advance', '', 'Component substitutability depends on molecular structure. Four-cell pairing reversals show which combination works; six-cell relays further show that changing a third component can connect two working combinations, while the scope of the same substitution changes with molecular edits and reaction partners. This finding extends the structure–yield association to a relation between structure, component compatibility and substitution paths.', '', '## Methods and coverage', '', 'A recipe is defined jointly by identity and amount in six reagent slots, solvent, temperature and time. Any two slots are taken as a three-level axis and a two-level axis with the other seven slots fixed; all complete six cells enter the analysis. Each six-cell set keeps three endpoint level/relay choices and two endpoint diagonals; travelling the same path in reverse is not counted twice. The three-level axis can also be solvent, temperature or time, and its actual slot name is kept.', '', 'Six cells are first matched completely within dataset/archive/plate/acquisition/target; source rows in each cell are averaged, then complete acquisitions are averaged by target; zeros are kept and missing values are not filled with zero. The empty reagent slot none is an actually registered condition with the component omitted, distinct from a missing measurement. plate is a prefix for locating the source and is not used to infer independent physical plates. Shared molecules, partners, recipes, panels and paths are not treated as independent experimental replicates.', '', 'With six-cell order 00,10,01,11,x0,x1, the direct-path bottleneck is D=max[min(y00,y10,y11),min(y00,y01,y11)], the specified bridge-path bottleneck is B=min(y00,yx0,yx1,y11), and the continuous gain is B−D. A 20% connection requires both endpoints ≥20%, D<20% and B≥20%; the two-sided threshold margin is min(B−20%,20%−D), saved separately from the 5/10pp gains in B−D.', '', 'Failure of a specified bridge path describes this one complete six-cell path; other paths in the complete measured recipe graph are covered by the round-6 graph analysis. Nodes on a path are independent recipes, not reaction intermediates or sequential reaction steps.', '', '| Dataset | Original swaps | Complete registered six cells | Complete acquisition six cells | target×route connected | Coverage note |', '|---|---:|---:|---:|---:|---|']
    for row in coverage.to_dict('records'):
        lines.append(f"| {row['dataset_id']} | {row['n_original_swaps']} | {row['n_complete_registered_six_panels']} | {row['n_complete_acquisition_panels']} | {row['n_bridge_connects20']} | {row['status']} |")
    lines += ['', 'All outputs are written only to Analysis. Original files are kept read-only. selected_partner_original_records.csv.gz stores original row locations for all comparable partners of the selected exact edits; complete source records and independent recomputation code are kept with the analysis.', '']
    (OUT / 'BRIDGE_DOMAINS_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def partner_domains():
    """Retain every same-edit/same-route partner before identifying crossed domains."""
    data = read('round7/bridge_domains/all_exact_edit_partner_comparisons.csv.gz')
    routes = read('round7/bridge_domains/all_bridge_route_registry.csv')
    group_keys = ['dataset_id', 'rxn_class', 'role', 'structural_pair_id', 'route_id', 'molecule_a', 'molecule_b', 'graph_method', 'transformation']
    a, b = (data.bridge_bottleneck_a.to_numpy(float), data.bridge_bottleneck_b.to_numpy(float))
    both = data.both_endpoint_pairs20.to_numpy(bool)
    data['path_available_only_A'] = both & (a >= 0.2) & (b < 0.2)
    data['path_available_only_B'] = both & (a < 0.2) & (b >= 0.2)
    data['A_switch_margin'] = np.where(data.path_available_only_A, data.bridge_switch_margin, np.nan)
    data['B_switch_margin'] = np.where(data.path_available_only_B, data.bridge_switch_margin, np.nan)
    summary = data.groupby(group_keys, as_index=False, sort=False).agg(n_complete_comparisons=('comparison_id', 'size'), n_partners=('partner_smiles', 'nunique'), n_both_endpoint_pairs20=('both_endpoint_pairs20', 'sum'), n_A_only=('path_available_only_A', 'sum'), n_B_only=('path_available_only_B', 'sum'), best_A_switch_margin=('A_switch_margin', 'max'), best_B_switch_margin=('B_switch_margin', 'max'), minimum_B_minus_A=('bridge_difference', 'min'), maximum_B_minus_A=('bridge_difference', 'max'))
    summary['crossed_partner_path_domains'] = summary.n_A_only.gt(0) & summary.n_B_only.gt(0)
    summary['crossed_partner_margin'] = summary[['best_A_switch_margin', 'best_B_switch_margin']].min(axis=1, skipna=False)
    summary['crossed_partner_path_domains_margin5'] = summary.crossed_partner_path_domains & summary.crossed_partner_margin.ge(0.05)
    summary['crossed_partner_path_domains_margin10'] = summary.crossed_partner_path_domains & summary.crossed_partner_margin.ge(0.1)
    summary = summary.merge(routes[['route_id', 'panel_id']], on='route_id', validate='many_to_one')
    save(summary, 'bridge_edit_partner_domain_summary.csv.gz')
    aggregate = dict(n_all_exact_edit_comparisons=len(data), n_all_edit_route_domains=len(summary), n_domains_with_multiple_partners=int(summary.n_partners.ge(2).sum()))
    for field in ['crossed_partner_path_domains', 'crossed_partner_path_domains_margin5', 'crossed_partner_path_domains_margin10']:
        part = summary[summary[field]]
        aggregate[field] = dict(n_domains=len(part), n_datasets=part.dataset_id.nunique(), n_classes=part.rxn_class.nunique(), n_dataset_edits=part[['dataset_id', 'role', 'structural_pair_id']].drop_duplicates().shape[0], n_edit_panels=part[['dataset_id', 'role', 'structural_pair_id', 'panel_id']].drop_duplicates().shape[0])
    print('PARTNER_DOMAINS', json.dumps(aggregate, ensure_ascii=False), flush=True)
    qualifying = summary[summary.crossed_partner_path_domains].sort_values('crossed_partner_margin', ascending=False)
    selected_domains = qualifying.groupby('rxn_class', sort=False).head(1)
    extra = qualifying[qualifying.graph_method.ne('RDKit_single_cut_MMP')].drop_duplicates(['dataset_id', 'role', 'structural_pair_id']).head(6)
    selected_domains = pd.concat([selected_domains, extra]).drop_duplicates(['dataset_id', 'role', 'structural_pair_id', 'route_id'])
    selected_domains['domain_case_id'] = ['partnerbridge_' + sid(v) for v in selected_domains[['dataset_id', 'role', 'structural_pair_id', 'route_id']].itertuples(index=False, name=None)]
    selected_all = data.merge(selected_domains[['dataset_id', 'role', 'structural_pair_id', 'route_id', 'domain_case_id']], on=['dataset_id', 'role', 'structural_pair_id', 'route_id'], validate='many_to_one')
    selected_rows = []
    for domain_case, subset in selected_all.groupby('domain_case_id', sort=False):
        chosen_partners = []
        chosen_targets = []
        for field in ['path_available_only_A', 'path_available_only_B']:
            row = subset[subset[field]].sort_values('bridge_switch_margin', ascending=False).iloc[0].to_dict()
            row['partner_selection'] = field
            selected_rows.append(row)
            chosen_partners.append(row['partner_smiles'])
            chosen_targets.extend([row['target_key_a'], row['target_key_b']])
        assert len(set(chosen_partners)) == 2 and len(set(chosen_targets)) == 4
    selected = pd.DataFrame(selected_rows)
    selected = selected.merge(routes.drop(columns=['dataset_id', 'rxn_class']), on='route_id', validate='many_to_one')
    save(selected_domains, 'selected_partner_reversal_domain_registry.csv')
    save(selected, 'selected_partner_reversal_cases.csv')
    save(selected_all, 'selected_partner_reversal_all_partners.csv')
    requests = []
    for row in selected_all.to_dict('records'):
        for side in ('a', 'b'):
            requests.append(dict(domain_case_id=row['domain_case_id'], comparison_id=row['comparison_id'], side=side, dataset_id=row['dataset_id'], target_key=row['target_key_' + side], route_id=row['route_id']))
    requests = pd.DataFrame(requests).merge(routes[['route_id', 'panel_id', 'recipe_order']], on='route_id', validate='many_to_one')
    requests['recipe_id'] = requests.recipe_order.map(json.loads)
    requests = requests.drop(columns='recipe_order').explode('recipe_id', ignore_index=True)
    acquisition = read('round7/bridge_domains/all_complete_acquisition_six_yields.csv.gz', usecols=[*BLOCK, 'panel_id'])
    requests = requests.merge(acquisition, on=['dataset_id', 'target_key', 'panel_id'], validate='many_to_many')
    records = read('data/normalized_records.csv.gz', usecols=['record_id', *BLOCK, 'recipe_id', 'yield_numeric', 'strict_eligible', 'condition_id'])
    records = records[records.strict_eligible.astype(str).str.lower().eq('true')]
    source = requests.merge(records, on=[*BLOCK, 'recipe_id'], validate='many_to_many')
    raw = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    source = source.merge(raw, on='record_id', validate='many_to_one')
    source['source_absolute_path'] = [str(RAW_DIR / x) for x in source.source_file]
    save(source, 'selected_partner_reversal_original_records.csv.gz')
    aggregate['n_selected_domains'] = len(selected_domains)
    aggregate['n_selected_complete_partner_comparisons'] = len(selected_all)
    aggregate['n_selected_original_source_links'] = len(source)
    (OUT / 'partner_domain_summary.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Reaction partners rewrite which molecule can use the same relay', '', 'Within the same exact molecular edit × the same six-recipe route, with all comparable partners kept, this checks which molecule allows the specified bridge path to stay ≥20% throughout. Endpoints are ≥20% for both molecules under both partners; the four targets give 24 cells of actual yields.', '', f"All {aggregate['n_all_exact_edit_comparisons']:,} exact edit comparisons form {len(summary):,} edit × route domains, of which {aggregate['n_domains_with_multiple_partners']:,} have at least two complete partners.", '', '| Criterion | Edit × route domains | Datasets | Reaction classes | dataset × exact edit | Edit × six-recipe panel |', '|---|---:|---:|---:|---:|---:|']
    for field in ['crossed_partner_path_domains', 'crossed_partner_path_domains_margin5', 'crossed_partner_path_domains_margin10']:
        entry = aggregate[field]
        lines.append(f"| {field} | {entry['n_domains']} | {entry['n_datasets']} | {entry['n_classes']} | {entry['n_dataset_edits']} | {entry['n_edit_panels']} |")
    lines += ['', '## Complete 24-cell instances', '', '| Domain | Reaction | Fixed molecular edit A→B | Reaction partner | Same bridge component | A/B bridge bottleneck % | A/B direct bottleneck % |', '|---|---|---|---|---|---:|---:|']
    for row in selected.to_dict('records'):
        lines.append(f"| {row['domain_case_id']} | {row['rxn_class']} | `{row['molecule_a']}` → `{row['molecule_b']}` | `{row['partner_smiles']}` | {row['bridge_component']} | {100 * row['bridge_bottleneck_a']:.2f}/{100 * row['bridge_bottleneck_b']:.2f} | {100 * row['direct_bottleneck_a']:.2f}/{100 * row['direct_bottleneck_b']:.2f} |")
    lines += ['', 'The same bridge path lets A pass and B fail with partner 1, and lets B pass and A fail with partner 2. The compatibility of the relay component is thus determined jointly by both reactants. The specified bridge path and both direct paths are saved separately; the whole recipe graph is not judged disconnected on this basis.', '', f'For the {len(selected_domains)} selected domains, all {len(selected_all)} partner comparisons are kept in selected_partner_reversal_all_partners.csv; selected_partner_reversal_cases.csv stores the displayed 24 cells and complete route component labels, and selected_partner_reversal_original_records.csv.gz stores source rows for all partners.', '']
    (OUT / 'PARTNER_DEPENDENT_BRIDGE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    classes = mapping.set_index('dataset_id').rxn_class.to_dict()
    registry = read('data/condition_slot_registry.csv').drop_duplicates(['dataset_id', 'recipe_id'])
    registry = registry[registry.dataset_id.isin(classes)].copy()
    assert len(mapping) == 20 and mapping.rxn_class.nunique() == 18
    assert not registry.duplicated(['dataset_id', *TOKENS]).any()
    selection_only = '--refresh-selection' in sys.argv
    refresh = '--refresh-domains' in sys.argv or selection_only
    panels = read('round7/bridge_domains/complete_six_recipe_registry.csv') if refresh else enumerate_panels(registry, classes)
    save(panels, 'complete_six_recipe_registry.csv')
    print('PANELS', len(panels), panels.groupby('dataset_id').size().to_dict(), flush=True)
    routes = route_registry(panels, registry)
    save(routes, 'all_bridge_route_registry.csv')
    if refresh:
        records = read('data/normalized_records.csv.gz')
        records = records[records.dataset_id.isin(mapping.dataset_id) & records.strict_eligible.astype(str).str.lower().eq('true')].copy()
        records['yield_numeric'] = pd.to_numeric(records.yield_numeric, errors='raise')
        identity = records[['dataset_id', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].drop_duplicates()
        acquisition = read('round7/bridge_domains/all_complete_acquisition_six_yields.csv.gz')
        target = read('round7/bridge_domains/all_complete_target_six_yields.csv.gz')
        results = read('round7/bridge_domains/all_target_bridge_routes.csv.gz')
    else:
        records, acquisition, target, identity, source_lookup = measurements(mapping, panels)
        results = evaluate_routes(target, routes)
    print('Target routes', len(results), 'Connects', int(results.bridge_connects20.sum()), flush=True)
    domains = read('round7/bridge_domains/all_exact_edit_partner_comparisons.csv.gz') if selection_only else molecular_domains(results, identity)
    coverage, counts, aggregate = summarize(mapping, registry, panels, acquisition, target, results, domains)
    chosen, partners, cross = selected_examples(results, domains, routes, registry, records, acquisition)
    write_report(coverage, aggregate, chosen, partners, cross)
    print('SUMMARY', aggregate, flush=True)
    print('Elapsed', round(time.time() - started, 2), flush=True)
    partner_domains()
if __name__ == '__main__':
    if '--partner-domains' in sys.argv:
        partner_domains()
    else:
        main()
