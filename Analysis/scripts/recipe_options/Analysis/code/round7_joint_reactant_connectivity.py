"""Enumerate complete reactant 2x2 grids on four-way measured recipe graphs."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from functools import lru_cache
import gzip
from itertools import combinations, product
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import round6_molecular_connectivity as molecular
from round6_molecular_rewiring import NAMES
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round7/joint_reactant_connectivity'
BLOCK = molecular.BLOCK
SLOTS, TOKENS = (molecular.SLOTS, molecular.TOKENS)
SIDES = ['ac', 'bc', 'ad', 'bd']
EPS = 1e-12
DISPLAY_NAMES = {'C#Cc1cccc(N)c1': '3-ethynylaniline', 'CSc1cccc(N)c1': '3-(methylthio)aniline', 'COc1ccc(Br)nc1': '2-bromo-5-methoxypyridine', 'COc1cccnc1Br': '2-bromo-3-methoxypyridine', 'Cc1cc(C)c[c]([Mg][Br])c1': '3,5-dimethylphenylmagnesium bromide', 'Clc1cc[c]([Mg][Br])cc1': '4-chlorophenylmagnesium bromide', 'Brc1cccc2cc[nH]c12': '7-bromo-1H-indole', 'COC(=O)c1ccc(Br)nc1': 'methyl 6-bromonicotinate', 'C=[CH][Mg][Br]': 'vinylmagnesium bromide', 'C[CH2][Mg][Br]': 'ethylmagnesium bromide', 'CC(C)S(=O)(=O)c1ccccc1N': '2-(propan-2-ylsulfonyl)aniline', 'Cc1cc(C)nc(N)n1': '4,6-dimethylpyrimidin-2-amine', 'Cc1cc(Cl)nc2ccccc12': '2-chloro-4-methylquinoline', 'Clc1ccc2ccccc2n1': '2-chloroquinoline', 'C=Cc1ccc(C)cc1': '4-methylstyrene', 'C=Cc1ccccc1': 'styrene', 'COC(=O)c1cncc(Br)c1': 'methyl 5-bromonicotinate', 'Cc1ccc(Br)c(C)n1': '3-bromo-2,6-dimethylpyridine', 'Cc1ccc(=O)[nH]c1': '5-methylpyridin-2(1H)-one', 'NCC12CC3CC(CC(C3)C1)C2': '(adamantan-1-yl)methanamine', 'CC(C)(C)c1ccc(B(O)O)cc1': '4-tert-butylphenylboronic acid', 'COC(=O)c1ccc(B(O)O)cc1': '4-(methoxycarbonyl)phenylboronic acid', 'Cc1ccc(O)cc1': '4-methylphenol', 'Oc1ccccc1': 'phenol', 'C#CCN1CCOCC1': '4-(prop-2-yn-1-yl)morpholine', 'C#Cc1ccc(N)cc1': '4-ethynylaniline', 'COc1ccc(I)cc1': '4-iodoanisole', 'COc1ccccc1I': '2-iodoanisole', 'O=Cc1ccc(C(F)(F)F)cc1': '4-(trifluoromethyl)benzaldehyde', 'O=Cc1ccc([N+](=O)[O-])cc1': '4-nitrobenzaldehyde', 'CCc1ccc(N)cc1': '4-ethylaniline', 'Nc1ccccc1': 'aniline', 'CCCCN1CCNCC1': '1-butylpiperazine', 'O=[N+]([O-])c1cccnc1F': '2-fluoro-3-nitropyridine', 'Cc1ncc([N+](=O)[O-])n1CC(C)O': 'secnidazole', 'Cc1ncc([N+](=O)[O-])n1CCO': 'metronidazole', 'Cc1cc(C)cc(C(=O)O)c1': '3,5-dimethylbenzoic acid', 'O=C(O)c1cc2ccccc2oc1=O': 'coumarin-3-carboxylic acid', 'C=C(C)CBr': 'methallyl bromide', 'C=CCBr': 'allyl bromide', 'CC(=O)Nc1cccc(B(O)O)c1': '3-acetamidophenylboronic acid', 'O=C1CCc2ccccc2N1': '3,4-dihydroquinolin-2(1H)-one'}

def read(path, **kwargs):
    return pd.read_csv(ROOT / path, keep_default_na=False, low_memory=False, **kwargs)

def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def sid(value):
    return molecular.sid(value)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression='gzip' if name.endswith('.gz') else None)

def prepare():
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    norm = read('data/normalized_records.csv.gz')
    norm = norm[norm.strict_eligible.astype(str).str.lower().eq('true') & norm.dataset_id.isin(mapping.dataset_id)].copy()
    norm['yield_numeric'] = pd.to_numeric(norm.yield_numeric)
    assert len(norm) == 380910 and np.isfinite(norm.yield_numeric).all()
    target = norm[['dataset_id', 'target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']].drop_duplicates()
    assert not target.duplicated(['dataset_id', 'target_key']).any()
    canonical = read('data/molecular_features.csv').set_index('smiles').canonical_smiles.to_dict()
    for role in ['sub_1_smiles', 'sub_2_smiles']:
        target['canonical_' + role] = target[role].map(canonical)
        assert target['canonical_' + role].notna().all()
    assert not target.duplicated(['dataset_id', 'canonical_sub_1_smiles', 'canonical_sub_2_smiles']).any()
    means = norm.groupby([*BLOCK, 'recipe_id'], sort=False, as_index=False).yield_numeric.mean()
    registry = read('data/condition_slot_registry.csv').drop_duplicates(['dataset_id', 'recipe_id'])
    assert not registry.duplicated(['dataset_id', *TOKENS]).any()
    graphs = {}
    for dataset, frame in registry.groupby('dataset_id', sort=False):
        frame = frame.sort_values('recipe_id')
        tokens = frame[TOKENS].to_numpy(str)
        labels, full = ({}, {})
        for row in frame.to_dict('records'):
            text = {}
            for slot in SLOTS:
                if slot.startswith('Reagent'):
                    name = row[f'{slot}_Standardized Name'] or 'none'
                    text[slot] = name + (f" {row[f'{slot}_HTE_Equiv']} eq" if name != 'none' else '')
                else:
                    text[slot] = {'Solvent': row['Standardized Name'], 'Temperature': str(row['temp_id']) + ' C', 'Time': str(row['time_id']) + ' h'}[slot]
            labels[row['recipe_id']] = text
            full[row['recipe_id']] = '; '.join((f'{slot}={text[slot]}' for slot in SLOTS))
        graphs[dataset] = dict(recipes=frame.recipe_id.tolist(), lookup={r: i for i, r in enumerate(frame.recipe_id)}, adjacent=(tokens[:, None, :] != tokens[None, :, :]).sum(-1) == 1, tokens=tokens, labels=labels, full=full)
    blocks, by_target, values, block_rows = ({}, defaultdict(list), {}, [])
    for key, group in means.groupby(BLOCK, sort=False):
        block_id = 'b_' + sid(list(key))
        graph = graphs[key[0]]
        y = np.full(len(graph['recipes']), np.nan)
        for recipe, value in zip(group.recipe_id, group.yield_numeric):
            y[graph['lookup'][recipe]] = value
        blocks[block_id], values[block_id] = (key, y)
        by_target[key[0], key[-1]].append(block_id)
        block_rows.append(dict(block_id=block_id, **dict(zip(BLOCK, key)), n_measured_recipes=len(group), recipe_ids=packed(sorted(group.recipe_id))))
    assert len(blocks) == 5889 and len(target) == 5554
    return (mapping, norm, target, graphs, blocks, by_target, values, pd.DataFrame(block_rows))

def grid_iterator(target, by_target):
    for dataset, frame in target.groupby('dataset_id', sort=True):
        pairs = {(r.canonical_sub_1_smiles, r.canonical_sub_2_smiles): r for r in frame.itertuples()}
        neighbors = defaultdict(set)
        for a, c in pairs:
            if a and c:
                neighbors[a].add(c)
        for a, b in combinations(sorted(neighbors), 2):
            for c, d in combinations(sorted(neighbors[a] & neighbors[b]), 2):
                corner = [pairs[key] for key in [(a, c), (b, c), (a, d), (b, d)]]
                grid_id = 'q_' + sid([dataset, a, b, c, d])
                metadata = dict(grid_id=grid_id, dataset_id=dataset, molecule_a=a, molecule_b=b, molecule_c=c, molecule_d=d)
                for side, row in zip(SIDES, corner):
                    metadata.update({'target_key_' + side: row.target_key, 'product_smiles_' + side: row.product_smiles})
                choices = [by_target[dataset, row.target_key] for row in corner]
                for block_ids in product(*choices):
                    yield dict(metadata, acquisition_quartet_id='g_' + sid([grid_id, *block_ids]), **{'block_id_' + side: key for side, key in zip(SIDES, block_ids)})

def scope(target, graphs, by_target, values, mapping):
    rows, graph_cache = ([], {})
    class_map = mapping.set_index('dataset_id').rxn_class.to_dict()
    for row in grid_iterator(target, by_target):
        dataset = row['dataset_id']
        ys = np.array([values[row['block_id_' + side]] for side in SIDES])
        common = np.flatnonzero(np.isfinite(ys).all(axis=0))
        key = (dataset, tuple(common))
        if key not in graph_cache:
            adjacent = graphs[dataset]['adjacent'][np.ix_(common, common)]
            ngroups, labels = connected_components(csr_matrix(adjacent), directed=False)
            graph_cache[key] = (int(adjacent.sum() // 2), ngroups, labels)
        edges, groups, labels = graph_cache[key]
        high = np.flatnonzero((ys[:, common] >= 0.2).all(axis=0))
        _, sizes = np.unique(labels[high], return_counts=True)
        n_pairs = int(np.sum(sizes * (sizes - 1) // 2))
        status = 'endpoint_comparable' if n_pairs else 'fewer_than_two_common_measured_recipes' if len(common) < 2 else 'fewer_than_two_common_high_recipes' if len(high) < 2 else 'common_high_endpoints_in_different_full_components'
        row.update(rxn_class=class_map[dataset], common_recipe_ids=packed([graphs[dataset]['recipes'][i] for i in common]), n_common_measured_recipes=len(common), n_common_graph_edges=edges, n_common_graph_components=groups, n_common_high_recipes=len(high), n_common_high_endpoint_pairs=len(high) * (len(high) - 1) // 2, n_common_high_endpoint_pairs_with_full_graph_path=n_pairs, comparability_status=status)
        for side, y in zip(SIDES, ys):
            row['n_measured_recipes_' + side] = int(np.isfinite(y).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    save(frame, 'all_acquisition_quartet_scope.csv.gz')
    scope_rows = []
    swaps = read('results/matched_swap_summary.csv').groupby('dataset_id').size()
    for row in mapping.to_dict('records'):
        part = frame[frame.dataset_id.eq(row['dataset_id'])]
        target_part = target[target.dataset_id.eq(row['dataset_id'])]
        scope_rows.append(dict(row, n_legacy_swaps=int(swaps.get(row['dataset_id'], 0)), n_strict_targets=len(target_part), n_complete_reactant_grids=part.grid_id.nunique(), n_acquisition_quartets=len(part), n_endpoint_comparable_quartets=int(part.comparability_status.eq('endpoint_comparable').sum()), n_endpoint_comparable_grids=part.loc[part.comparability_status.eq('endpoint_comparable'), 'grid_id'].nunique(), n_endpoint_pairs=int(part.n_common_high_endpoint_pairs_with_full_graph_path.sum()), maximum_common_recipes=int(part.n_common_measured_recipes.max()) if len(part) else 0))
    coverage = pd.DataFrame(scope_rows)
    save(coverage, 'scope_dataset_coverage.csv')
    summary = dict(strict_rows=380910, strict_targets=5554, strict_acquisition_blocks=5889, study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, complete_reactant_grids=frame.grid_id.nunique(), acquisition_quartets=len(frame), endpoint_comparable_quartets=int(frame.comparability_status.eq('endpoint_comparable').sum()), endpoint_comparable_grids=frame.loc[frame.comparability_status.eq('endpoint_comparable'), 'grid_id'].nunique(), endpoint_pair_rows=int(frame.n_common_high_endpoint_pairs_with_full_graph_path.sum()), unique_common_recipe_subgraphs=len(graph_cache), statuses=frame.comparability_status.value_counts().to_dict())
    (OUT / 'scope_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('SCOPE ' + packed(summary), flush=True)
    return (frame, coverage)

def mst_capacity(y, adjacent):
    edge = np.minimum(y[:, None], y[None, :])
    tree = minimum_spanning_tree(csr_matrix(np.where(adjacent, 1 + float(y.max()) - edge, 0)))
    neighbors = [np.flatnonzero(row).tolist() for row in (tree + tree.T > 0).toarray()]
    capacity = np.full(adjacent.shape, -np.inf)
    for root in range(len(y)):
        stack = [(root, -1, float(y[root]))]
        while stack:
            node, previous, bottleneck = stack.pop()
            capacity[root, node] = bottleneck
            stack.extend(((n, node, min(bottleneck, float(y[n]))) for n in neighbors[node] if n != previous))
    return capacity

def endpoint_metrics(capacity, lower):
    high = capacity >= 0.2
    pattern = (high * np.array([8, 4, 2, 1])[:, None]).sum(axis=0)
    checker = (pattern == 9) | (pattern == 6)
    margin0 = np.minimum(capacity[0], capacity[3]) - np.maximum(capacity[1], capacity[2])
    margin1 = np.minimum(capacity[1], capacity[2]) - np.maximum(capacity[0], capacity[3])
    margin = np.maximum(margin0, margin1)
    clearance0 = np.minimum(np.minimum(capacity[0], capacity[3]) - 0.2, 0.2 - np.maximum(capacity[1], capacity[2]))
    clearance1 = np.minimum(np.minimum(capacity[1], capacity[2]) - 0.2, 0.2 - np.maximum(capacity[0], capacity[3]))
    clearance = np.maximum(clearance0, clearance1)
    penalty = np.maximum(0.0, lower - capacity)
    contrast = np.array([1.0, -1.0, -1.0, 1.0])
    dd = contrast @ capacity
    dd_lower = contrast @ lower
    dd_penalty = contrast @ penalty
    effect_c, effect_d = (capacity[0] - capacity[1], capacity[2] - capacity[3])
    reversing = (effect_c > EPS) & (effect_d < -EPS) | (effect_c < -EPS) & (effect_d > EPS)
    reverse_margin = np.minimum(np.abs(effect_c), np.abs(effect_d))
    flags = dict(checkerboard=checker, checkerboard_gap5pp=checker & (margin >= 0.05 - EPS), checkerboard_gap10pp=checker & (margin >= 0.1 - EPS), checkerboard_clearance5pp=checker & (clearance >= 0.05 - EPS), checkerboard_clearance10pp=checker & (clearance >= 0.1 - EPS), connection_change=(pattern != 0) & (pattern != 15), opposite_continuous_molecular_effects=reversing, opposite_continuous_molecular_effects5pp=reversing & (reverse_margin >= 0.05 - EPS), opposite_continuous_molecular_effects10pp=reversing & (reverse_margin >= 0.1 - EPS), K_interaction5pp=np.abs(dd) >= 0.05 - EPS, K_interaction10pp=np.abs(dd) >= 0.1 - EPS, penalty_interaction5pp=np.abs(dd_penalty) >= 0.05 - EPS, penalty_interaction10pp=np.abs(dd_penalty) >= 0.1 - EPS)
    return dict(pattern=pattern, margin=margin, clearance=clearance, penalty=penalty, DD_K=dd, DD_lower=dd_lower, DD_penalty=dd_penalty, effect_c=effect_c, effect_d=effect_d, reverse_margin=reverse_margin, flags=flags)

def analyze(scope_frame, graphs, blocks, values):
    checks = Counter()

    @lru_cache(maxsize=1024)
    def graph_info(dataset, common):
        adjacent = graphs[dataset]['adjacent'][np.ix_(common, common)]
        _, labels = connected_components(csr_matrix(adjacent), directed=False)
        return (adjacent, labels)

    @lru_cache(maxsize=2048)
    def capacities(block_id, common):
        dataset = blocks[block_id][0]
        adjacent, _ = graph_info(dataset, common)
        y = values[block_id][list(common)]
        answer = molecular.widest_matrix(y, adjacent)
        independent = mst_capacity(y, adjacent)
        np.testing.assert_allclose(answer, independent, atol=EPS, rtol=0)
        high = np.flatnonzero(y >= 0.2)
        _, labels = connected_components(csr_matrix(adjacent[np.ix_(high, high)]), directed=False)
        assert np.array_equal(answer[np.ix_(high, high)] >= 0.2, labels[:, None] == labels[None, :])
        checks['full_capacity_matrices_independently_verified'] += 1
        checks['all_pair_capacity_entries_verified'] += answer.size
        return answer
    columns = ['acquisition_quartet_id', 'endpoint0_recipe', 'endpoint1_recipe']
    columns += ['K_' + side for side in SIDES] + ['lower_endpoint_yield_' + side for side in SIDES]
    columns += ['endpoint_penalty_' + side for side in SIDES]
    columns += ['connection_pattern_ac_bc_ad_bd', 'diagonal_bottleneck_separation_margin', 'threshold_clearance_margin', 'DD_K', 'DD_lower_endpoint_yield', 'DD_endpoint_penalty', 'effect_A_minus_B_at_C', 'effect_A_minus_B_at_D']
    rows, endpoint_total = ([], 0)
    with gzip.open(OUT / 'all_common_high_endpoint_bottlenecks.csv.gz', 'wt', encoding='utf-8-sig', newline='', compresslevel=3) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for count, original in enumerate(scope_frame.to_dict('records'), 1):
            row = dict(original)
            dataset = row['dataset_id']
            graph = graphs[dataset]
            recipes = json.loads(row['common_recipe_ids'])
            common = tuple((graph['lookup'][r] for r in recipes))
            ys = np.array([values[row['block_id_' + side]][list(common)] for side in SIDES])
            for side in SIDES:
                key = blocks[row['block_id_' + side]]
                row.update({field + '_' + side: key[j] for j, field in enumerate(BLOCK) if field in ['archive', 'plate', 'acquisition']})
            _, labels = graph_info(dataset, common)
            high = np.flatnonzero((ys >= 0.2).all(axis=0))
            ends = [(i, j) for i, j in combinations(high, 2) if labels[i] == labels[j]]
            assert len(ends) == row['n_common_high_endpoint_pairs_with_full_graph_path']
            blank = endpoint_metrics(np.empty((4, 0)), np.empty((4, 0)))
            for flag in blank['flags']:
                row['any_' + flag] = False
                row['n_pairs_' + flag] = 0
            for name in ['maximum_absolute_DD_K', 'maximum_absolute_DD_endpoint_penalty', 'maximum_diagonal_bottleneck_separation_margin', 'maximum_opposite_continuous_effect_margin']:
                row[name] = np.nan
            row['connection_pattern_counts'] = packed({})
            row['representative_endpoint0_recipe'] = row['representative_endpoint1_recipe'] = ''
            if ends:
                ii, jj = np.asarray(ends, int).T
                cap = np.array([capacities(row['block_id_' + side], common)[ii, jj] for side in SIDES])
                lower = np.minimum(ys[:, ii], ys[:, jj])
                metrics = endpoint_metrics(cap, lower)
                assert np.isfinite(cap).all()
                np.testing.assert_allclose(metrics['DD_K'], metrics['DD_lower'] - metrics['DD_penalty'], atol=EPS, rtol=0)
                for flag, active in metrics['flags'].items():
                    row['any_' + flag], row['n_pairs_' + flag] = (bool(active.any()), int(active.sum()))
                row.update(maximum_absolute_DD_K=float(np.abs(metrics['DD_K']).max()), maximum_absolute_DD_endpoint_penalty=float(np.abs(metrics['DD_penalty']).max()), maximum_diagonal_bottleneck_separation_margin=float(metrics['margin'].max()), maximum_opposite_continuous_effect_margin=float(metrics['reverse_margin'][metrics['flags']['opposite_continuous_molecular_effects']].max()) if metrics['flags']['opposite_continuous_molecular_effects'].any() else 0.0, connection_pattern_counts=packed({format(int(k), '04b'): int(v) for k, v in zip(*np.unique(metrics['pattern'], return_counts=True))}))
                chosen = max(range(len(ends)), key=lambda j: (metrics['flags']['checkerboard'][j], metrics['margin'][j] if metrics['flags']['checkerboard'][j] else abs(metrics['DD_penalty'][j]), abs(metrics['DD_K'][j])))
                row['representative_endpoint0_recipe'], row['representative_endpoint1_recipe'] = (recipes[ii[chosen]], recipes[jj[chosen]])
                for name, array in [('K', cap), ('lower_endpoint_yield', lower), ('endpoint_penalty', metrics['penalty'])]:
                    for z, side in enumerate(SIDES):
                        row['representative_' + name + '_' + side] = float(array[z, chosen])
                for name in ['pattern', 'margin', 'clearance', 'DD_K', 'DD_lower', 'DD_penalty']:
                    row['representative_' + name] = int(metrics[name][chosen]) if name == 'pattern' else float(metrics[name][chosen])
                writer.writerows(([row['acquisition_quartet_id'], recipes[i], recipes[j], *cap[:, z].tolist(), *lower[:, z].tolist(), *metrics['penalty'][:, z].tolist(), format(int(metrics['pattern'][z]), '04b'), float(metrics['margin'][z]), float(metrics['clearance'][z]), float(metrics['DD_K'][z]), float(metrics['DD_lower'][z]), float(metrics['DD_penalty'][z]), float(metrics['effect_c'][z]), float(metrics['effect_d'][z])] for z, (i, j) in enumerate(ends)))
                endpoint_total += len(ends)
            rows.append(row)
            if count % 5000 == 0:
                print(f"Analyzed {count}/{len(scope_frame)} acquisition quartets; {endpoint_total} endpoint pairs; {checks['full_capacity_matrices_independently_verified']} independent matrices", flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 127099 and result.grid_id.nunique() == 102270 and (endpoint_total == 7853943)
    checks.update(acquisition_quartets=127099, complete_reactant_grids=102270, endpoint_pairs=endpoint_total)
    save(result, 'all_acquisition_quartet_summary.csv.gz')
    flags = list(blank['flags'])
    grid = result.groupby(['grid_id', 'dataset_id', 'rxn_class'], sort=False, as_index=False).agg(n_acquisition_quartets=('acquisition_quartet_id', 'size'), n_endpoint_comparable_quartets=('comparability_status', lambda s: int(s.eq('endpoint_comparable').sum())), maximum_common_measured_recipes=('n_common_measured_recipes', 'max'), maximum_absolute_DD_K=('maximum_absolute_DD_K', 'max'), maximum_absolute_DD_endpoint_penalty=('maximum_absolute_DD_endpoint_penalty', 'max'), **{'any_' + flag: ('any_' + flag, 'max') for flag in flags})
    save(grid, 'all_complete_reactant_grid_summary.csv.gz')
    return (result, grid, checks)

def summarize(result, grid, coverage):
    flags = [c for c in result.columns if c.startswith('any_')]
    aggregate = []
    for metric in flags:
        event = result[result[metric]]
        aggregate.append(dict(metric=metric, n_all_acquisition_quartets=len(result), n_all_reactant_grids=len(grid), n_endpoint_comparable_quartets=int(result.comparability_status.eq('endpoint_comparable').sum()), n_endpoint_comparable_grids=int(grid.n_endpoint_comparable_quartets.gt(0).sum()), n_event_acquisition_quartets=len(event), n_event_reactant_grids=event.grid_id.nunique(), n_event_datasets=event.dataset_id.nunique(), n_event_reaction_classes=event.rxn_class.nunique(), n_event_endpoint_pairs=int(event[metric.replace('any_', 'n_pairs_', 1)].sum())))
    aggregate = pd.DataFrame(aggregate)
    save(aggregate, 'aggregate_results.csv')
    coverage = coverage.copy()
    for metric in flags:
        counts = result.groupby('dataset_id')[metric].sum()
        coverage['n_quartets_' + metric] = coverage.dataset_id.map(counts).fillna(0).astype(int)
        events = result[result[metric]].groupby('dataset_id').grid_id.nunique()
        coverage['n_grids_' + metric] = coverage.dataset_id.map(events).fillna(0).astype(int)
    assert len(coverage) == 20 and coverage.rxn_class.nunique() == 18 and (coverage.n_legacy_swaps.sum() == 4002)
    save(coverage, 'dataset_coverage.csv')
    pattern_rows = []
    for dataset, part in [('all', result), *list(result.groupby('dataset_id', sort=True))]:
        patterns = [(row.grid_id, json.loads(row.connection_pattern_counts)) for row in part.itertuples()]
        for pattern in range(16):
            key = format(pattern, '04b')
            supported = [(grid_id, counts.get(key, 0)) for grid_id, counts in patterns if counts.get(key, 0)]
            pattern_rows.append(dict(dataset_id=dataset, connection_pattern_ac_bc_ad_bd=key, n_endpoint_pairs=sum((count for _, count in supported)), n_acquisition_quartets=len(supported), n_reactant_grids=len({grid_id for grid_id, _ in supported})))
    patterns = pd.DataFrame(pattern_rows)
    assert patterns.loc[patterns.dataset_id.eq('all'), 'n_endpoint_pairs'].sum() == 7853943
    save(patterns, 'connection_pattern_counts.csv')
    return (aggregate, coverage)

def relation_lookup(result):
    path = ROOT / 'round7/partner_conditioned_edits/molecular_relation_registry.csv.gz'
    if not path.exists():
        return {}
    registry = read('round7/partner_conditioned_edits/molecular_relation_registry.csv.gz')
    supplement_path = OUT / 'supplementary_molecular_relation_registry.csv.gz'
    if supplement_path.exists():
        registry = pd.concat([registry, read('round7/joint_reactant_connectivity/supplementary_molecular_relation_registry.csv.gz')], ignore_index=True)
    lookup = {(r['dataset_id'], r['role'], r['molecular_pair_key']): r for r in registry.to_dict('records')}
    missing = {}
    import round6_molecular_rewiring as chemistry
    for row in result.drop_duplicates('grid_id').to_dict('records'):
        for letters, role in [('ab', 'sub_1_smiles'), ('cd', 'sub_2_smiles')]:
            a, b = [row['molecule_' + letter] for letter in letters]
            pair = packed(sorted([a, b]))
            key = (row['dataset_id'], role, pair)
            if key not in lookup:
                missing[key] = dict(dataset_id=key[0], role=role, molecular_pair_key=pair, molecular_pair_id=chemistry.sid(pair), molecule_a=a, molecule_b=b)
    if missing:
        supplement = chemistry.molecular_relations(pd.DataFrame(missing.values()))
        changes = []
        for row in supplement.itertuples():
            same = row.structural_molecule_a == row.exact_molecule_a
            if row.exact_graph_method == 'exact_methylene_insertion':
                change = 'insert_CH2' if same else 'delete_CH2'
            elif row.exact_graph_method == 'exact_substituent_relocation':
                change = 'relocate ' + row.exact_fragment_a + ' on identical parent graph'
            elif row.has_existing_exact_neighbor_rule:
                change = row.exact_fragment_a + ' >> ' + row.exact_fragment_b if same else row.exact_fragment_b + ' >> ' + row.exact_fragment_a
            elif row.single_cut_common_core_ge60pct_ge5atoms:
                change = row.single_cut_fragment_a + ' >> ' + row.single_cut_fragment_b
            else:
                change = 'broader_structural_change'
            changes.append(change)
        supplement['structural_change_a_to_b'] = changes
        supplement['edit_id'] = ['je_' + sid([r.dataset_id, r.role, r.molecular_pair_key]) for r in supplement.itertuples()]
        if supplement_path.exists():
            supplement = pd.concat([read('round7/joint_reactant_connectivity/supplementary_molecular_relation_registry.csv.gz'), supplement], ignore_index=True)
        save(supplement, 'supplementary_molecular_relation_registry.csv.gz')
        lookup.update({(r['dataset_id'], r['role'], r['molecular_pair_key']): r for r in supplement.to_dict('records')})
        print(f'SUPPLEMENTARY_RELATIONS {len(supplement)}', flush=True)
    return lookup

def annotate_relations(result, grid):
    renames = {'any_any_connection_change': 'any_connection_change', 'n_pairs_any_connection_change': 'n_pairs_connection_change'}
    result, grid = (result.rename(columns=renames), grid.rename(columns=renames))
    lookup = relation_lookup(result)
    annotations = []
    known_names = dict(NAMES)
    from round6_path_organization import NAMES as path_names
    known_names.update(path_names)
    for row in result.drop_duplicates('grid_id').to_dict('records'):
        annotation = dict(grid_id=row['grid_id'])
        for letters, role in [('ab', 'sub_1_smiles'), ('cd', 'sub_2_smiles')]:
            pair = packed(sorted([row['molecule_' + letter] for letter in letters]))
            relation = lookup.get((row['dataset_id'], role, pair), {})
            annotation['has_exact_edit_' + letters] = bool(relation.get('has_existing_exact_neighbor_rule', False))
            annotation['has_round4_edit_' + letters] = bool(relation.get('has_round4_allowed_exact_rule', False))
            for field in ['edit_id', 'exact_graph_method', 'exact_transformation', 'structural_change_a_to_b', 'structural_relation']:
                annotation[field + '_' + letters] = relation.get(field, 'not_in_partner_edit_registry')
        annotation['has_exact_edit_either_side'] = annotation['has_exact_edit_ab'] or annotation['has_exact_edit_cd']
        annotation['has_exact_edit_both_sides'] = annotation['has_exact_edit_ab'] and annotation['has_exact_edit_cd']
        annotation['has_methylene_or_relocation_edit'] = any((annotation['exact_graph_method_' + letters] in {'exact_methylene_insertion', 'exact_substituent_relocation'} for letters in ['ab', 'cd']))
        annotation['n_named_molecules'] = sum((row['molecule_' + letter] in known_names for letter in ['a', 'b', 'c', 'd']))
        for letter in ['a', 'b', 'c', 'd']:
            annotation['molecule_name_' + letter] = known_names.get(row['molecule_' + letter], '')
        annotations.append(annotation)
    annotations = pd.DataFrame(annotations)
    result = result.drop(columns=[c for c in annotations if c != 'grid_id' and c in result], errors='ignore').merge(annotations, on='grid_id', validate='many_to_one')
    grid = grid.drop(columns=[c for c in annotations if c != 'grid_id' and c in grid], errors='ignore').merge(annotations, on='grid_id', validate='one_to_one')
    for event in ['checkerboard', 'checkerboard_gap5pp', 'checkerboard_gap10pp', 'opposite_continuous_molecular_effects10pp']:
        for prefix, condition in [('exact_edit_', result.has_exact_edit_either_side), ('methylene_or_relocation_', result.has_methylene_or_relocation_edit)]:
            result['any_' + prefix + event] = result['any_' + event] & condition
            result['n_pairs_' + prefix + event] = result['n_pairs_' + event] * condition.astype(int)
            events = result.groupby('grid_id')['any_' + prefix + event].max()
            grid['any_' + prefix + event] = grid.grid_id.map(events)
    save(result, 'all_acquisition_quartet_summary.csv.gz')
    save(grid, 'all_complete_reactant_grid_summary.csv.gz')
    save(annotations, 'complete_grid_molecular_relations.csv.gz')
    return (result, grid)

def self_checks():
    adjacent = np.array([[False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]])
    y = np.array([[0.8, 0.6, 0.8, 0.1], [0.8, 0.05, 0.8, 0.1], [0.8, 0.1, 0.8, 0.05], [0.8, 0.6, 0.8, 0.1]])
    cap = np.array([molecular.widest_matrix(values, adjacent)[0, 2] for values in y])[:, None]
    for values in y:
        np.testing.assert_allclose(molecular.widest_matrix(values, adjacent), mst_capacity(values, adjacent), atol=EPS, rtol=0)
    result = endpoint_metrics(cap, np.full((4, 1), 0.8))
    assert result['pattern'].tolist() == [9] and result['flags']['checkerboard_gap10pp'].tolist() == [True]
    np.testing.assert_allclose(result['DD_K'], result['DD_lower'] - result['DD_penalty'], atol=EPS, rtol=0)
    shifted = endpoint_metrics(cap + 0.15, np.full((4, 1), 0.95))
    assert shifted['pattern'].tolist() == [15] and (not shifted['flags']['checkerboard'].any())
    np.testing.assert_allclose(result['DD_K'], shifted['DD_K'], atol=EPS, rtol=0)
    for pattern in range(16):
        values = np.array([0.6 if pattern & bit else 0.1 for bit in [8, 4, 2, 1]])[:, None]
        metrics = endpoint_metrics(values, np.full((4, 1), 0.8))
        assert metrics['pattern'].tolist() == [pattern]
        assert metrics['flags']['checkerboard'].tolist() == [pattern in [6, 9]]

def original_sources(norm, block_frame):
    raw = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row', 'yield_numeric'])
    sources = norm.merge(block_frame[[*BLOCK, 'block_id']], on=BLOCK, validate='many_to_one')
    sources = sources.merge(raw, on='record_id', suffixes=('', '_raw'), validate='one_to_one')
    np.testing.assert_allclose(sources.yield_numeric, pd.to_numeric(sources.yield_numeric_raw), atol=EPS, rtol=0)
    raw_root = RAW_DIR
    sources['source_absolute_path'] = sources.source_file.map(lambda name: str(raw_root / name))
    assert all((reader_tables.source_is_file(path) for path in sources.source_absolute_path.unique()))
    save(sources.drop(columns='yield_numeric_raw'), 'all_strict_source_records.csv.gz')
    node = sources.groupby(['block_id', 'recipe_id'], sort=False, as_index=False).agg(yield_mean=('yield_numeric', 'mean'), n_source_records=('record_id', 'size'))
    save(node, 'all_strict_recipe_node_means.csv.gz')
    return sources

def select_examples(result, graphs, blocks, values, sources):
    candidates = result[result.comparability_status.eq('endpoint_comparable')].copy()
    preference = ['any_checkerboard', 'has_methylene_or_relocation_edit', 'has_exact_edit_either_side', 'has_exact_edit_both_sides', 'n_named_molecules', 'any_checkerboard_gap10pp', 'maximum_absolute_DD_endpoint_penalty', 'maximum_absolute_DD_K']
    candidates = candidates.sort_values([column for column in preference if column in candidates], ascending=False)
    selected = candidates.drop_duplicates('rxn_class').copy()
    selected['case_id'] = 'j_' + selected.acquisition_quartet_id.map(sid)
    paths, nodes, links, selected_rows = ([], [], [], [])
    source_groups = {key: frame for key, frame in sources.groupby('block_id', sort=False)}
    for case in selected.to_dict('records'):
        graph = graphs[case['dataset_id']]
        recipes = json.loads(case['common_recipe_ids'])
        common = [graph['lookup'][r] for r in recipes]
        adjacent = graph['adjacent'][np.ix_(common, common)]
        ys = np.array([values[case['block_id_' + side]][common] for side in SIDES])
        i, j = [recipes.index(case['representative_endpoint' + str(end) + '_recipe']) for end in [0, 1]]
        assert (ys[:, [i, j]] >= 0.2).all()
        for z, side in enumerate(SIDES):
            case['endpoint0_yield_' + side], case['endpoint1_yield_' + side] = (float(ys[z, i]), float(ys[z, j]))
            path = molecular.widest_path(ys[z], adjacent, i, j)
            assert abs(float(ys[z, path].min()) - case['representative_K_' + side]) <= EPS
            independent = mst_capacity(ys[z], adjacent)
            assert abs(independent[i, j] - ys[z, path].min()) <= EPS
            case['path_recipe_ids_' + side] = packed([recipes[position] for position in path])
            case['path_yields_' + side] = packed(ys[z, path].tolist())
            for step, position in enumerate(path):
                recipe = recipes[position]
                changed, previous, new = ('', '', '')
                if step:
                    old = recipes[path[step - 1]]
                    diff = np.flatnonzero(graph['tokens'][common[path[step - 1]]] != graph['tokens'][common[position]])
                    assert len(diff) == 1
                    changed = SLOTS[diff[0]]
                    previous, new = (graph['labels'][old][changed], graph['labels'][recipe][changed])
                paths.append(dict(case_id=case['case_id'], acquisition_quartet_id=case['acquisition_quartet_id'], path_optimized_for=side, path_step=step, recipe_id=recipe, **{'yield_' + s: float(ys[k, position]) for k, s in enumerate(SIDES)}, full_recipe_text=graph['full'][recipe], changed_slot=changed, previous_component=previous, new_component=new))
            source = source_groups[case['block_id_' + side]]
            source = source[source.recipe_id.isin(recipes)].copy()
            source['case_id'], source['measured_side'] = (case['case_id'], side)
            links.append(source)
        for position, recipe in enumerate(recipes):
            nodes.append(dict(case_id=case['case_id'], recipe_id=recipe, full_recipe_text=graph['full'][recipe], **{'yield_' + s: float(ys[k, position]) for k, s in enumerate(SIDES)}))
        selected_rows.append(case)
    selected = pd.DataFrame(selected_rows)
    for letter in ['a', 'b', 'c', 'd']:
        if 'molecule_name_' + letter not in selected:
            selected['molecule_name_' + letter] = selected['molecule_' + letter].map(NAMES).fillna('')
        display = selected['molecule_' + letter].map(DISPLAY_NAMES)
        selected['molecule_name_' + letter] = display.fillna(selected['molecule_name_' + letter])
    selected_sources = pd.concat(links, ignore_index=True)
    save(selected, 'selected_chemical_examples.csv')
    save(pd.DataFrame(paths), 'selected_path_steps.csv')
    save(pd.DataFrame(nodes), 'selected_common_recipe_yields.csv')
    save(selected_sources.drop(columns='yield_numeric_raw', errors='ignore'), 'selected_original_records.csv.gz')
    return (selected, pd.DataFrame(paths), selected_sources)

def write_report(result, grid, aggregate, coverage, selected, paths, sources, checks, started):
    summary = dict(status='completed', study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, strict_rows=380910, strict_targets=5554, strict_acquisition_blocks=5889, complete_reactant_grids=len(grid), acquisition_quartets=len(result), endpoint_pairs=int(result.n_common_high_endpoint_pairs_with_full_graph_path.sum()), selected_cases=len(selected), selected_source_links=len(sources), aggregation=aggregate.to_dict('records'), elapsed_seconds=round(time.time() - started, 2))
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    verification = dict(status='passed', verified_at_utc=datetime.now(timezone.utc).isoformat(), method='every used Floyd max-min capacity independently matched to SciPy maximum spanning forest; direct 20% induced components; selected widest paths and source links', checks=dict(checks))
    (OUT / 'verification_summary.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# How complete reactant pairs rewire recipe connectivity', '', '## Scope and definitions', '', f"The study retains 20 datasets, 18 reaction classes and 4,002 original single-slot swaps. This analysis starts from all {summary['strict_rows']:,} strictly qualified records, covering {summary['strict_targets']:,} actual targets and {summary['strict_acquisition_blocks']:,} dataset/archive/plate/acquisition/target acquisition blocks, without relying on earlier molecule-pair candidates or strongly supported rectangles.", '', 'Complete 2×2 reactant grids are enumerated by canonical reactant identity, with A/B on the first side and C/D on the second; the four actual targets are labelled AC, BC, AD, BD. Each reactant pair corresponds to a unique actual target in these data; original target/product identities are saved in a separate table. Each target keeps all of its acquisition blocks independently, and the four sides are combined as a Cartesian product of acquisition blocks; targets are not required or claimed to belong to the same acquisition.', '', 'Each four-block combination uses only recipe nodes measured for all four targets. Two nodes are connected only if the exact tokens of the nine slots (including amounts) differ in one slot; within each target, raw yields are first averaged by acquisition block and recipe. Missing records are not filled with zero, and paths cannot use nodes measured for only some targets.', '', 'The same endpoint pair must be ≥20% for all four targets and lie in the same connected component of the full common graph. K is the lowest yield along the best path connecting the two endpoints for that target within this common graph. 20% high-yield connectivity patterns are encoded in the order AC/BC/AD/BD; 1001 or 0110 are strictly crossed connectivity. The 5/10pp gap requires the smaller K of the two connected corners minus the larger K of the two unconnected corners to be at least 5/10pp; a stricter version with a 5/10pp margin on both sides of the threshold is also saved.', '', 'All four continuous K values and four endpoint losses P=min(endpoint yields)−K are kept for every endpoint pair. Continuous joint quantity DD_K=K_AC−K_BC−K_AD+K_BD, with DD_P defined likewise; DD_endpoint is also saved, and DD_K=DD_endpoint−DD_P always holds. The A minus B bottleneck effect with C and D fixed is also saved in full, so non-crossed connectivity changes and continuous opposite effects can both be checked. Extremes taken after enumerating all endpoints are descriptive only; endpoints, acquisition combinations and shared molecules are not treated as independent experiments for significance.', '', '## Full results', '', f"There are {len(grid):,} complete reactant grids and {len(result):,} four-block acquisition combinations. Combinations supporting common high-yield endpoint comparison: {result.comparability_status.eq('endpoint_comparable').sum():,}, involving {grid.n_endpoint_comparable_quartets.gt(0).sum():,} reactant grids, with {summary['endpoint_pairs']:,} endpoint pairs saved. Other combinations are recorded as insufficient common measurement, insufficient high-yield endpoints or different full-graph components; continuous quantities are left blank.", '', '| Criterion | Four-acquisition combinations | Complete reactant grids | Datasets/reaction classes | Endpoint pairs |', '|---|---:|---:|---:|---:|']
    for row in aggregate.to_dict('records'):
        lines.append(f"| {row['metric']} | {row['n_event_acquisition_quartets']:,} | {row['n_event_reactant_grids']:,} | {row['n_event_datasets']}/{row['n_event_reaction_classes']} | {row['n_event_endpoint_pairs']:,} |")
    lines += ['', '## Examples across reaction classes', '']
    for case in selected.to_dict('records'):
        lines += [f"### {case['rxn_class']}: {case['case_id']}", '', f"A={case['molecule_name_a'] or case['molecule_a']}; B={case['molecule_name_b'] or case['molecule_b']}; C={case['molecule_name_c'] or case['molecule_c']}; D={case['molecule_name_d'] or case['molecule_d']}.", '', f"Common measured recipes for all four targets: {case['n_common_measured_recipes']}; selected same-endpoint connectivity pattern {format(int(case['representative_pattern']), '04b')}; K for AC/BC/AD/BD: " + '/'.join((f"{100 * case['representative_K_' + s]:.2f}%" for s in SIDES)) + f"; DD_K={100 * case['representative_DD_K']:.2f}pp, DD_P={100 * case['representative_DD_penalty']:.2f}pp.", '']
        if 'has_exact_edit_ab' in case:
            lines += ['Molecular relations on both sides: A→B is `' + str(case['exact_graph_method_ab']) + '`, `' + str(case['structural_change_a_to_b_ab']) + '`; C→D is `' + str(case['exact_graph_method_cd']) + '`, `' + str(case['structural_change_a_to_b_cd']) + '`.', '']
        for side in SIDES:
            step = paths[paths.case_id.eq(case['case_id']) & paths.path_optimized_for.eq(side)].sort_values('path_step')
            lines += [f"{side.upper()}: `{case['target_key_' + side]}`; strict acquisition block `{case['archive_' + side]} / {case['plate_' + side]} / {case['acquisition_' + side]}`.", '', 'Yields at each node along the best path (AC/BC/AD/BD): ' + ' → '.join(('/'.join((f"{100 * getattr(r, 'yield_' + s):.2f}" for s in SIDES)) + '%' for r in step.itertuples())) + '.', '', 'Full starting recipe: ' + step.iloc[0].full_recipe_text + '.', '']
            lines += [f'- {r.changed_slot}: {r.previous_component} → {r.new_component}.' for r in step.iloc[1:].itertuples()]
            lines.append('')
    lines += ['## Sources and recomputation', '', f"all_acquisition_quartet_scope.csv.gz and all_acquisition_quartet_summary.csv.gz keep {len(result):,} combinations with their comparability status; all_complete_reactant_grid_summary.csv.gz keeps {len(grid):,} complete molecule grids. all_common_high_endpoint_bottlenecks.csv.gz stores, for all {summary['endpoint_pairs']:,} endpoint pairs, the four K values, endpoints, losses and continuous quantities; connectivity patterns are not limited to positive cases.", '', f"all_strict_target_identities.csv.gz keeps original/canonical molecules and product; all_strict_target_acquisition_blocks.csv.gz keeps the independent acquisition identity and measured recipes of each target; all_strict_recipe_node_means.csv.gz can rebuild each common graph by block_id+recipe_id; all_strict_source_records.csv.gz gives file, sheet, row number and absolute path for all {summary['strict_rows']:,} strict original records.", '', 'selected_chemical_examples.csv, selected_path_steps.csv, selected_common_recipe_yields.csv and selected_original_records.csv.gz give, for the cross-class examples, the four targets, all common nodes, the best path for each target and full sources. Each Floyd max-min capacity matrix actually used is recomputed independently with a SciPy maximum spanning forest, and connectivity is checked against connected groups of the 20% induced graph; selected examples are also verified step by step with an independent widest-path search.', '']
    (OUT / 'JOINT_REACTANT_CONNECTIVITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('COMPLETED ' + packed(summary), flush=True)

def main():
    started = time.time()
    self_checks()
    if '--self-check' in sys.argv:
        print('JOINT_REACTANT_SELF_CHECKS_PASSED', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if '--refresh-names' in sys.argv:
        selected = read('round7/joint_reactant_connectivity/selected_chemical_examples.csv')
        for letter in ['a', 'b', 'c', 'd']:
            selected['molecule_name_' + letter] = selected['molecule_' + letter].map(DISPLAY_NAMES).fillna(selected['molecule_name_' + letter])
        save(selected, 'selected_chemical_examples.csv')
        result = read('round7/joint_reactant_connectivity/all_acquisition_quartet_summary.csv.gz')
        grid = read('round7/joint_reactant_connectivity/all_complete_reactant_grid_summary.csv.gz')
        aggregate = read('round7/joint_reactant_connectivity/aggregate_results.csv')
        coverage = read('round7/joint_reactant_connectivity/dataset_coverage.csv')
        paths = read('round7/joint_reactant_connectivity/selected_path_steps.csv')
        sources = read('round7/joint_reactant_connectivity/selected_original_records.csv.gz')
        checks = Counter(json.loads((OUT / 'verification_summary.json').read_text(encoding='utf-8'))['checks'])
        write_report(result, grid, aggregate, coverage, selected, paths, sources, checks, started)
        return
    mapping, norm, target, graphs, blocks, by_target, values, block_frame = prepare()
    if '--relations-only' in sys.argv:
        relation_lookup(read('round7/joint_reactant_connectivity/all_acquisition_quartet_scope.csv.gz'))
        return
    if '--refresh-examples' in sys.argv:
        result = read('round7/joint_reactant_connectivity/all_acquisition_quartet_summary.csv.gz')
        grid = read('round7/joint_reactant_connectivity/all_complete_reactant_grid_summary.csv.gz')
        for frame in [result, grid]:
            for column in frame:
                if column.startswith('maximum_') or (column.startswith('representative_') and (not column.endswith('_recipe'))):
                    frame[column] = pd.to_numeric(frame[column], errors='coerce')
        coverage = read('round7/joint_reactant_connectivity/scope_dataset_coverage.csv')
        checks = Counter(json.loads((OUT / 'verification_summary.json').read_text(encoding='utf-8'))['checks'])
        result, grid = annotate_relations(result, grid)
        aggregate, coverage = summarize(result, grid, coverage)
        sources = read('round7/joint_reactant_connectivity/all_strict_source_records.csv.gz')
        selected, paths, selected_sources = select_examples(result, graphs, blocks, values, sources)
        checks.update(selected_source_links_verified=len(selected_sources) - checks.get('selected_source_links_verified', 0), selected_paths_verified=4 * len(selected) - checks.get('selected_paths_verified', 0))
        write_report(result, grid, aggregate, coverage, selected, paths, selected_sources, checks, started)
        return
    save(block_frame, 'all_strict_target_acquisition_blocks.csv.gz')
    save(target, 'all_strict_target_identities.csv.gz')
    scope_frame, coverage = scope(target, graphs, by_target, values, mapping)
    print(f'SCOPE_COMPLETED seconds={time.time() - started:.2f}', flush=True)
    if '--scope-only' in sys.argv:
        return
    result, grid, checks = analyze(scope_frame, graphs, blocks, values)
    result, grid = annotate_relations(result, grid)
    aggregate, coverage = summarize(result, grid, coverage)
    sources = original_sources(norm, block_frame)
    selected, paths, selected_sources = select_examples(result, graphs, blocks, values, sources)
    checks.update(strict_source_rows_verified=len(sources), selected_source_links_verified=len(selected_sources), selected_paths_verified=4 * len(selected))
    write_report(result, grid, aggregate, coverage, selected, paths, selected_sources, checks, started)
if __name__ == '__main__':
    main()
