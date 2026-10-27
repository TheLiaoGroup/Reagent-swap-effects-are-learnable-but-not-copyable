"""Follow the same directed molecular edit across all partners and recipe rectangles."""
from __future__ import annotations
from collections import defaultdict
import gzip
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
import round6_molecular_rewiring as chemistry
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round7/partner_conditioned_edits'
Y = chemistry.CORNERS
RECIPES = chemistry.RECIPES
BLOCK = chemistry.BLOCK
EPS = 1e-12
SIGNS = np.array([1.0, -1.0, -1.0, 1.0])
AXIS_PREFERENCES = ['structure_preference_focal_bg0', 'structure_preference_focal_bg1', 'structure_preference_context_c0', 'structure_preference_context_c1']
RELATION_FIELDS = ['dataset_id', 'role', 'molecular_pair_key', 'edit_id', 'has_existing_exact_neighbor_rule', 'has_round4_allowed_exact_rule', 'structural_relation']
RECIPE_FIELDS = ['rectangle_id', 'representation_id', 'exchange_id', 'focal_slot', 'context_slot', 'component0_name', 'component1_name', 'component0_dose', 'component1_dose', 'context0_name', 'context1_name', 'context0_token', 'context1_token', 'fixed_background_json', *RECIPES, *['full_' + name for name in RECIPES]]

def read(name, **kwargs):
    return pd.read_csv(ROOT / name, keep_default_na=False, low_memory=False, **kwargs)

def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def sid(value):
    return chemistry.sid(packed(value))

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 1} if name.endswith('.gz') else None)

def response_metrics(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    result = {}
    for side, values in [('a', a), ('b', b)]:
        result['DD_' + side] = values @ SIGNS
        result['m0_' + side] = values[:, [0, 3]].min(1) - values[:, [1, 2]].max(1)
        result['m1_' + side] = values[:, [1, 2]].min(1) - values[:, [0, 3]].max(1)
        result['diagonal0_min_' + side] = values[:, [0, 3]].min(1)
        result['diagonal1_min_' + side] = values[:, [1, 2]].min(1)
    result['delta_DD'] = result['DD_b'] - result['DD_a']
    result[AXIS_PREFERENCES[0]] = b[:, 1] - b[:, 0] - (a[:, 1] - a[:, 0])
    result[AXIS_PREFERENCES[1]] = b[:, 3] - b[:, 2] - (a[:, 3] - a[:, 2])
    result[AXIS_PREFERENCES[2]] = b[:, 2] - b[:, 0] - (a[:, 2] - a[:, 0])
    result[AXIS_PREFERENCES[3]] = b[:, 3] - b[:, 1] - (a[:, 3] - a[:, 1])
    for tag, threshold, need_high in [('continuous', 0.0, False), ('5pp', 0.05, True), ('10pp', 0.1, True)]:
        directions = []
        for side in ['a', 'b']:
            left = result['m0_' + side] > 0 if not need_high else (result['m0_' + side] >= threshold) & (result['diagonal0_min_' + side] >= 0.2)
            right = result['m1_' + side] > 0 if not need_high else (result['m1_' + side] >= threshold) & (result['diagonal1_min_' + side] >= 0.2)
            direction = np.where(left, 1, np.where(right, -1, 0))
            result['direction_' + side + '_' + tag] = direction
            directions.append(direction)
        da, db = directions
        result['transition_' + tag] = np.select([(da == 1) & (db == -1), (da == -1) & (db == 1), (da == 0) & (db == 1), (da == 0) & (db == -1), (da == 1) & (db == 0), (da == -1) & (db == 0), (da == 1) & (db == 1), (da == -1) & (db == -1)], ['reversal_00_11_to_10_01', 'reversal_10_01_to_00_11', 'established_00_11', 'established_10_01', 'eliminated_00_11', 'eliminated_10_01', 'retained_00_11', 'retained_10_01'], default='neither_strong_matching')
    return result

def load():
    targets = read('round5/joint_accessibility/all_target_accessibility.csv.gz')
    rectangles = read('round5/joint_accessibility/rectangle_accessibility_summary.csv')
    assert len(targets) == 176355 and len(rectangles) == 830 and (rectangles.main_n_ge20.sum() == 825)
    features = read('data/molecular_features.csv')
    mapping = features.set_index('smiles').canonical_smiles.to_dict()
    targets['target_row_id'] = np.arange(len(targets))
    for role in chemistry.ROLES:
        targets['original_' + role] = targets[role]
        targets[role] = targets[role].map(mapping)
        assert targets[role].notna().all()
    assert not targets.duplicated(['rectangle_id', *chemistry.ROLES]).any()
    all_targets = len(targets)
    rectangles = rectangles[rectangles.main_n_ge20].copy()
    targets = targets[targets.rectangle_id.isin(rectangles.rectangle_id)].copy()
    assert len(targets) == 176321
    targets['DD'] = targets[Y].to_numpy(float) @ SIGNS
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    assert len(mapping) == 20 and mapping.rxn_class.nunique() == 18
    return (targets, rectangles, mapping, all_targets)

def all_relations(targets):
    rows = []
    for dataset, part in targets.groupby('dataset_id', sort=True):
        unique = part.drop_duplicates(chemistry.ROLES)
        for role, partner in [(chemistry.ROLES[0], chemistry.ROLES[1]), (chemistry.ROLES[1], chemistry.ROLES[0])]:
            pairs = set()
            for _, group in unique[unique[role].ne('') & unique[partner].ne('')].groupby(partner, sort=False):
                pairs.update(combinations(sorted(set(group[role])), 2))
            for a, b in sorted(pairs):
                key = chemistry.pair_key(a, b)
                rows.append(dict(dataset_id=dataset, role=role, molecule_a=a, molecule_b=b, molecular_pair_key=key, molecular_pair_id=chemistry.sid(key)))
    base = pd.DataFrame(rows)
    relations = chemistry.molecular_relations(base)
    relations['edit_id'] = [sid([r.dataset_id, r.role, r.structural_molecule_a, r.structural_molecule_b]) for r in relations.itertuples()]
    changes = []
    for row in relations.itertuples():
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
    relations['structural_change_a_to_b'] = changes
    for side in ['a', 'b']:
        relations['molecule_name_' + side] = relations['structural_molecule_' + side].map(chemistry.NAMES).fillna('')
    assert relations.edit_id.is_unique
    save(relations, 'molecular_relation_registry.csv.gz')
    return relations

def profile(targets, rectangles, relations):
    lookup = {(row.dataset_id, row.role, row.molecular_pair_key): bool(row.has_existing_exact_neighbor_rule) for row in relations.itertuples()}
    counts = defaultdict(int)
    by_dataset = defaultdict(lambda: defaultdict(int))
    for rectangle, group in targets.groupby('rectangle_id', sort=False):
        dataset = group.dataset_id.iloc[0]
        for role, partner in [(chemistry.ROLES[0], chemistry.ROLES[1]), (chemistry.ROLES[1], chemistry.ROLES[0])]:
            pair_counts = defaultdict(int)
            for _, frame in group[group[role].ne('') & group[partner].ne('')].groupby(partner, sort=False):
                for a, b in combinations(sorted(frame[role]), 2):
                    pair_counts[chemistry.pair_key(a, b)] += 1
            for key, n_partners in pair_counts.items():
                exact = lookup[dataset, role, key]
                for bucket in ['all', 'exact'] if exact else ['all']:
                    values = {bucket + '_partner_observations': n_partners, bucket + '_edit_rectangle_groups': 1, bucket + '_groups_ge2_partners': int(n_partners >= 2), bucket + '_partner_pairs': n_partners * (n_partners - 1) // 2}
                    for field, value in values.items():
                        counts[field] += value
                        by_dataset[dataset][field] += value
    summary = dict(study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, all_registered_rectangles=830, all_target_rectangle_rows=176355, primary_rectangles=len(rectangles), primary_target_rectangle_rows=len(targets), n_structural_relations=len(relations), n_exact_structural_relations=int(relations.has_existing_exact_neighbor_rule.sum()), **dict(counts))
    save(pd.DataFrame([dict(dataset_id=ds, **values) for ds, values in by_dataset.items()]), 'initial_scale_by_dataset.csv')
    (OUT / 'initial_scale.json').write_text(packed(summary) + '\n', encoding='utf-8')
    print('INITIAL_SCALE', packed(summary), flush=True)
    return summary

def reconstruct_all_four_cells(targets, rectangles):
    fields = ['record_id', *BLOCK, 'recipe_id', 'yield_numeric', 'strict_eligible']
    normal = read('data/normalized_records.csv.gz', usecols=fields)
    normal = normal[normal.strict_eligible.astype(str).str.lower().eq('true')].copy()
    normal['yield_numeric'] = pd.to_numeric(normal.yield_numeric, errors='raise')
    cells = normal.groupby([*BLOCK, 'recipe_id'], sort=False, as_index=False).yield_numeric.mean()
    wide = {dataset: part.pivot(index=BLOCK[1:], columns='recipe_id', values='yield_numeric') for dataset, part in cells.groupby('dataset_id', sort=False)}
    expected = targets.set_index(['rectangle_id', 'target_key'])
    blocks, requests = ([], [])
    for rectangle in rectangles.itertuples(index=False):
        recipes = [getattr(rectangle, column) for column in RECIPES]
        complete = wide[rectangle.dataset_id].reindex(columns=recipes).dropna()
        needed = expected.loc[rectangle.rectangle_id]
        complete = complete[complete.index.get_level_values('target_key').isin(needed.index)]
        reconstructed = complete.groupby(level='target_key').mean().reindex(needed.index)
        assert np.allclose(reconstructed, needed[Y], atol=EPS), rectangle.rectangle_id
        complete.columns = Y
        complete = complete.reset_index()
        complete['dataset_id'], complete['rectangle_id'] = (rectangle.dataset_id, rectangle.rectangle_id)
        complete['target_row_id'] = complete.target_key.map(needed.target_row_id)
        blocks.append(complete)
        for corner, recipe in zip(Y, recipes):
            frame = complete[['rectangle_id', 'target_row_id', *BLOCK, corner]].rename(columns={corner: 'matched_cell_yield'})
            frame['corner'], frame['recipe_id'] = (corner, recipe)
            requests.append(frame)
    blocks = pd.concat(blocks, ignore_index=True)
    requests = pd.concat(requests, ignore_index=True)
    links = requests.merge(normal.drop(columns='strict_eligible'), on=[*BLOCK, 'recipe_id'], how='left', validate='many_to_many')
    assert links.record_id.notna().all()
    assert not links.duplicated(['rectangle_id', 'target_row_id', 'archive', 'plate', 'acquisition', 'corner', 'record_id']).any()
    locations = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    links = links.merge(locations, on='record_id', how='left', validate='many_to_one')
    assert links.source_file.notna().all()
    raw_root = RAW_DIR
    links['source_absolute_path'] = links.source_file.map(lambda name: str(raw_root / name))
    save(blocks, 'all_complete_target_acquisition_blocks.csv.gz')
    save(links, 'all_four_cell_original_record_links.csv.gz')
    save(targets, 'all_primary_target_four_yields.csv.gz')
    save(rectangles, 'all_primary_recipe_rectangles.csv')
    print('SOURCES', len(blocks), len(links), links.record_id.nunique(), flush=True)
    return (blocks, links)

def summarize_group(part):
    group = part.groupby('edit_rectangle_id', sort=False)
    summary = group.agg(edit_id=('edit_id', 'first'), rectangle_id=('rectangle_id', 'first'), dataset_id=('dataset_id', 'first'), rxn_class=('rxn_class', 'first'), role=('role', 'first'), n_common_partners=('partner_smiles', 'size'), minimum_delta_DD=('delta_DD', 'min'), maximum_delta_DD=('delta_DD', 'max'), mean_delta_DD=('delta_DD', 'mean'), median_delta_DD=('delta_DD', 'median'), has_existing_exact_neighbor_rule=('has_existing_exact_neighbor_rule', 'first'), has_round4_allowed_exact_rule=('has_round4_allowed_exact_rule', 'first'), structural_relation=('structural_relation', 'first'))
    summary['delta_DD_range'] = summary.maximum_delta_DD - summary.minimum_delta_DD
    summary['cross_partner_comparable'] = summary.n_common_partners.ge(2)
    signs = part[['edit_rectangle_id', 'delta_DD']].assign(positive=part.delta_DD.gt(EPS), negative=part.delta_DD.lt(-EPS)).groupby('edit_rectangle_id', sort=False)[['positive', 'negative']].sum()
    summary['n_delta_DD_positive'], summary['n_delta_DD_negative'] = (signs.positive, signs.negative)
    summary['n_delta_DD_zero'] = summary.n_common_partners - summary.n_delta_DD_positive - summary.n_delta_DD_negative
    for threshold in [0.1, 0.2]:
        summary[f'opposite_delta_DD_each_ge{round(100 * threshold)}pp'] = summary.minimum_delta_DD.le(-threshold + EPS) & summary.maximum_delta_DD.ge(threshold - EPS)
    for tag in ['continuous', '5pp', '10pp']:
        counts = pd.crosstab(part.edit_rectangle_id, part['transition_' + tag]).reindex(summary.index, fill_value=0)
        for category in ['reversal_00_11_to_10_01', 'reversal_10_01_to_00_11', 'established_00_11', 'established_10_01', 'eliminated_00_11', 'eliminated_10_01', 'retained_00_11', 'retained_10_01', 'neither_strong_matching']:
            summary['n_' + category + '_' + tag] = counts.get(category, pd.Series(0, index=summary.index))
        summary['opposite_matching_reversal_' + tag] = summary['n_reversal_00_11_to_10_01_' + tag].gt(0) & summary['n_reversal_10_01_to_00_11_' + tag].gt(0)
        summary['n_distinct_transition_types_' + tag] = (counts > 0).sum(axis=1)
        summary['n_matching_change_' + tag] = summary[['n_' + category + '_' + tag for category in ['reversal_00_11_to_10_01', 'reversal_10_01_to_00_11', 'established_00_11', 'established_10_01', 'eliminated_00_11', 'eliminated_10_01']]].sum(axis=1)
        summary['has_establishment_and_elimination_' + tag] = summary[['n_established_00_11_' + tag, 'n_established_10_01_' + tag]].sum(axis=1).gt(0) & summary[['n_eliminated_00_11_' + tag, 'n_eliminated_10_01_' + tag]].sum(axis=1).gt(0)
    for metric in AXIS_PREFERENCES:
        extremes = group[metric].agg(['min', 'max'])
        summary['minimum_' + metric], summary['maximum_' + metric] = (extremes['min'], extremes['max'])
        for threshold in [0.05, 0.1]:
            summary[f'opposite_{metric}_each_ge{round(100 * threshold)}pp'] = extremes['min'].le(-threshold + EPS) & extremes['max'].ge(threshold - EPS)
    summary['any_component_structure_preference_opposite_5pp'] = summary[['opposite_' + m + '_each_ge5pp' for m in AXIS_PREFERENCES]].any(axis=1)
    summary['any_component_structure_preference_opposite_10pp'] = summary[['opposite_' + m + '_each_ge10pp' for m in AXIS_PREFERENCES]].any(axis=1)
    return summary.reset_index()

def partner_comparisons(part, relation_lookup):
    rows = []
    exact = part[part.has_existing_exact_neighbor_rule]
    for group_id, frame in exact.groupby('edit_rectangle_id', sort=False):
        frame = frame.sort_values('partner_smiles')
        first = frame.iloc[0]
        partner_role = chemistry.ROLES[1] if first.role == chemistry.ROLES[0] else chemistry.ROLES[0]
        for a, b in combinations(frame.to_dict('records'), 2):
            pair_key = chemistry.pair_key(a['partner_smiles'], b['partner_smiles'])
            structural = relation_lookup[first.dataset_id, partner_role, pair_key]
            row = dict(edit_rectangle_id=group_id, edit_id=first.edit_id, rectangle_id=first.rectangle_id, dataset_id=first.dataset_id, rxn_class=first.rxn_class, role=first.role, partner_pair_comparison_id=sid([group_id, a['partner_smiles'], b['partner_smiles']]), observation_id_partner0=a['observation_id'], observation_id_partner1=b['observation_id'], partner0_smiles=a['partner_smiles'], partner1_smiles=b['partner_smiles'], partner_molecular_pair_id=structural['molecular_pair_id'], partner_tanimoto=structural['morgan_radius2_2048_tanimoto'], partner_fingerprint_distance=1 - structural['morgan_radius2_2048_tanimoto'], partner_has_exact_rule=structural['has_existing_exact_neighbor_rule'], partner_structural_relation=structural['structural_relation'], partner_structural_change=structural['structural_change_a_to_b'], delta_DD_partner0=a['delta_DD'], delta_DD_partner1=b['delta_DD'], delta_DD_partner1_minus_partner0=b['delta_DD'] - a['delta_DD'], opposite_delta_DD_each_ge10pp=min(a['delta_DD'], b['delta_DD']) <= -0.1 + EPS and max(a['delta_DD'], b['delta_DD']) >= 0.1 - EPS, opposite_delta_DD_each_ge20pp=min(a['delta_DD'], b['delta_DD']) <= -0.2 + EPS and max(a['delta_DD'], b['delta_DD']) >= 0.2 - EPS)
            for index, observation in enumerate([a, b]):
                for side in ['a', 'b']:
                    row[f'target_row_id_{side}_partner{index}'] = observation['target_row_id_' + side]
            for tag in ['5pp', '10pp']:
                row['transition_partner0_' + tag], row['transition_partner1_' + tag] = (a['transition_' + tag], b['transition_' + tag])
                row['same_transition_' + tag] = a['transition_' + tag] == b['transition_' + tag]
                row['opposite_matching_reversal_' + tag] = {a['transition_' + tag], b['transition_' + tag]} == {'reversal_00_11_to_10_01', 'reversal_10_01_to_00_11'}
                row['at_least_one_matching_change_' + tag] = any((observation['transition_' + tag].startswith(('reversal_', 'established_', 'eliminated_')) for observation in [a, b]))
            values = []
            for metric in AXIS_PREFERENCES:
                row[metric + '_partner0'], row[metric + '_partner1'] = (a[metric], b[metric])
                values.append(min(a[metric], b[metric]) <= -0.1 + EPS and max(a[metric], b[metric]) >= 0.1 - EPS)
            row['any_component_structure_preference_opposite_10pp'] = any(values)
            rows.append(row)
    return pd.DataFrame(rows)

def observations(group, role, relations, edit_ids=None):
    partner = chemistry.ROLES[1] if role == chemistry.ROLES[0] else chemistry.ROLES[0]
    eligible = group[group[role].ne('') & group[partner].ne('')]
    columns = [partner, role, 'target_row_id', *Y]
    values = eligible[columns].rename(columns={partner: 'partner_smiles', role: 'molecule'})
    part = values.merge(values, on='partner_smiles', suffixes=('_a', '_b'), validate='many_to_many')
    part = part[part.molecule_a.lt(part.molecule_b)].copy()
    if not len(part):
        return part
    rectangle = group.rectangle_id.iloc[0]
    part['rectangle_id'], part['dataset_id'], part['rxn_class'], part['role'] = (rectangle, group.dataset_id.iloc[0], group.rxn_class.iloc[0], role)
    part['molecular_pair_key'] = [chemistry.pair_key(a, b) for a, b in part[['molecule_a', 'molecule_b']].to_numpy()]
    part = part.merge(relations[RELATION_FIELDS], on=['dataset_id', 'role', 'molecular_pair_key'], validate='many_to_one')
    if edit_ids is not None:
        part = part[part.edit_id.isin(edit_ids)].copy()
    if not len(part):
        return part
    part['edit_rectangle_id'] = [sid([rectangle, edit]) for edit in part.edit_id]
    part['observation_id'] = [sid([edit, partner]) for edit, partner in part[['edit_rectangle_id', 'partner_smiles']].to_numpy()]
    metrics = response_metrics(part[[c + '_a' for c in Y]], part[[c + '_b' for c in Y]])
    part = pd.concat([part.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    part['n_common_partners'] = part.groupby('edit_rectangle_id').partner_smiles.transform('size')
    return part.drop(columns=['molecule_a', 'molecule_b', 'molecular_pair_key'])

def priority_profiles(targets, rectangles, relations):
    seeds = read('round6/molecular_rewiring/all_opposite_direction_candidates.csv.gz')
    seeds = seeds[seeds.reversal_5pp & seeds.has_existing_exact_neighbor_rule]
    seeds = seeds[['dataset_id', 'role', 'molecular_pair_key', 'rectangle_id', 'candidate_id']].merge(relations[RELATION_FIELDS], on=['dataset_id', 'role', 'molecular_pair_key'], validate='many_to_one')
    frames = []
    for rectangle, seed_group in seeds.groupby('rectangle_id', sort=False):
        group = targets[targets.rectangle_id.eq(rectangle)]
        for role, role_seeds in seed_group.groupby('role', sort=False):
            frames.append(observations(group, role, relations, set(role_seeds.edit_id)))
    frame = pd.concat(frames, ignore_index=True)
    assert frame.observation_id.is_unique
    save(frame, 'priority_seeded_exact_edits_all_partner_observations.csv')
    summary = summarize_group(frame)
    summary = summary.merge(relations.drop(columns=['dataset_id', 'role', 'has_existing_exact_neighbor_rule', 'has_round4_allowed_exact_rule', 'structural_relation']), on='edit_id', validate='many_to_one')
    summary = summary.merge(rectangles[RECIPE_FIELDS], on='rectangle_id', validate='many_to_one')
    seed_ids = seeds.groupby(['edit_id', 'rectangle_id']).candidate_id.agg(lambda values: packed(sorted(values)))
    summary = summary.merge(seed_ids.rename('round6_seed_candidate_ids'), on=['edit_id', 'rectangle_id'], validate='one_to_one')
    summary = summary.sort_values(['opposite_delta_DD_each_ge10pp', 'n_distinct_transition_types_5pp', 'delta_DD_range', 'n_common_partners'], ascending=False)
    summary.insert(0, 'priority_rank', np.arange(1, len(summary) + 1))
    save(summary, 'priority_seeded_exact_edits_all_partner_summary.csv')
    lookup = {(row['dataset_id'], row['role'], row['molecular_pair_key']): row for row in relations.to_dict('records')}
    save(partner_comparisons(frame, lookup), 'priority_seeded_exact_edits_partner_pair_comparisons.csv')
    save(targets, 'all_primary_target_four_yields.csv.gz')
    save(rectangles[RECIPE_FIELDS], 'all_primary_recipe_rectangles.csv')
    print('PRIORITY', len(summary), len(frame), summary.rxn_class.nunique(), flush=True)
    print(summary[['priority_rank', 'edit_rectangle_id', 'rxn_class', 'structural_molecule_a', 'structural_molecule_b', 'structural_change_a_to_b', 'n_common_partners', 'minimum_delta_DD', 'maximum_delta_DD', 'n_distinct_transition_types_5pp']].head(10).to_json(orient='records'), flush=True)
    return (summary, frame)

def analyze(targets, rectangles, relations):
    relation_lookup = {(row['dataset_id'], row['role'], row['molecular_pair_key']): row for row in relations.to_dict('records')}
    group_frames, n_observations, n_pairs = ([], 0, 0)
    all_path = OUT / 'all_edit_partner_observations.csv.gz'
    partner_path = OUT / 'exact_edit_all_partner_pair_comparisons.csv.gz'
    with gzip.open(all_path, 'wt', encoding='utf-8-sig', newline='', compresslevel=1) as all_file, gzip.open(partner_path, 'wt', encoding='utf-8-sig', newline='', compresslevel=1) as pair_file:
        for count, (rectangle, group) in enumerate(targets.groupby('rectangle_id', sort=False), 1):
            for role, partner in [(chemistry.ROLES[0], chemistry.ROLES[1]), (chemistry.ROLES[1], chemistry.ROLES[0])]:
                part = observations(group, role, relations)
                if not len(part):
                    continue
                summary = summarize_group(part)
                group_frames.append(summary)
                part.to_csv(all_file, index=False, header=n_observations == 0)
                n_observations += len(part)
                pairs = partner_comparisons(part, relation_lookup)
                if len(pairs):
                    pairs.to_csv(pair_file, index=False, header=n_pairs == 0)
                    n_pairs += len(pairs)
            if count % 50 == 0:
                print('EDIT_PARTNERS', count, len(rectangles), n_observations, n_pairs, flush=True)
    summaries = pd.concat(group_frames, ignore_index=True)
    assert summaries.edit_rectangle_id.is_unique
    assert summaries.n_common_partners.sum() == n_observations
    save(summaries, 'all_edit_rectangle_partner_summary.csv.gz')
    exact = summaries[summaries.has_existing_exact_neighbor_rule]
    exact = exact.merge(relations.drop(columns=['dataset_id', 'role', 'has_existing_exact_neighbor_rule', 'has_round4_allowed_exact_rule', 'structural_relation']), on='edit_id', validate='many_to_one')
    exact = exact.merge(rectangles[RECIPE_FIELDS], on='rectangle_id', validate='many_to_one')
    save(exact, 'exact_edit_rectangle_partner_summary.csv.gz')
    summaries = summaries.merge(rectangles[['rectangle_id', 'focal_slot', 'context_slot']], on='rectangle_id', validate='many_to_one')
    inventory = summaries.groupby(['dataset_id', 'rxn_class', 'role', 'edit_id'], as_index=False).agg(n_rectangles=('rectangle_id', 'nunique'), n_partner_observations=('n_common_partners', 'sum'), maximum_common_partners=('n_common_partners', 'max'), n_cross_partner_rectangles=('cross_partner_comparable', 'sum'), n_focal_slots=('focal_slot', 'nunique'), n_context_slots=('context_slot', 'nunique'))
    save(inventory.merge(relations.drop(columns=['dataset_id', 'role']), on='edit_id', validate='one_to_one'), 'edit_coverage_across_rectangles.csv.gz')
    return (summaries, n_observations, n_pairs)

def overview(part):
    result = dict(n_edit_rectangle_groups=len(part), n_edits=part.edit_id.nunique(), n_rectangles=part.rectangle_id.nunique(), n_datasets=part.dataset_id.nunique(), n_classes=part.rxn_class.nunique(), n_partner_observations=int(part.n_common_partners.sum()), n_groups_single_partner=int(part.n_common_partners.eq(1).sum()), n_groups_ge2_partners=int(part.cross_partner_comparable.sum()))
    flags = ['opposite_delta_DD_each_ge10pp', 'opposite_delta_DD_each_ge20pp', 'any_component_structure_preference_opposite_5pp', 'any_component_structure_preference_opposite_10pp']
    for tag in ['5pp', '10pp']:
        result['n_partner_observations_matching_change_' + tag] = int(part['n_matching_change_' + tag].sum())
        flags += ['opposite_matching_reversal_' + tag, 'has_establishment_and_elimination_' + tag]
        for event in ['reversal', 'established', 'eliminated', 'retained']:
            columns = [column for column in part if column.startswith('n_' + event + '_') and column.endswith('_' + tag)]
            result['n_partner_observations_' + event + '_' + tag] = int(part[columns].sum().sum())
        result['n_groups_multiple_transition_types_' + tag] = int(part['n_distinct_transition_types_' + tag].gt(1).sum())
        result['n_groups_any_matching_change_' + tag] = int(part['n_matching_change_' + tag].gt(0).sum())
    for flag in flags:
        chosen = part[part[flag]]
        result['n_groups_' + flag] = len(chosen)
        result['n_edits_' + flag] = chosen.edit_id.nunique()
        result['n_classes_' + flag] = chosen.rxn_class.nunique()
    return result

def aggregates(summaries, targets, rectangles, relations, mapping, n_observations, n_pairs):
    prior = read('round6/molecular_rewiring/dataset_role_coverage.csv')
    coverage = prior[['dataset_id', 'rxn_class', 'role', 'n_swaps', 'n_registered_rectangles', 'n_primary_rectangles']].copy()
    rows = []
    for row in coverage.itertuples(index=False):
        part = summaries[summaries.dataset_id.eq(row.dataset_id) & summaries.role.eq(row.role)]
        values = dict(dataset_id=row.dataset_id, role=row.role)
        for label, frame in [('all', part), ('exact', part[part.has_existing_exact_neighbor_rule])]:
            values.update({label + '_' + key: value for key, value in overview(frame).items()})
        other = chemistry.ROLES[1] if row.role == chemistry.ROLES[0] else chemistry.ROLES[0]
        local = targets[targets.dataset_id.eq(row.dataset_id)]
        values['n_primary_targets_missing_partner'] = int(local[other].eq('').sum())
        values['status'] = 'partner_profiles_available' if len(part) else 'one_reactant_representation' if len(local) and local[other].eq('').all() else 'no_primary_partner_edit_profiles'
        rows.append(values)
    coverage = coverage.merge(pd.DataFrame(rows), on=['dataset_id', 'role'], validate='one_to_one')
    assert len(coverage) == 40 and coverage.dataset_id.nunique() == 20 and (coverage.rxn_class.nunique() == 18)
    assert coverage.drop_duplicates('dataset_id').n_swaps.sum() == 4002
    save(coverage, 'dataset_role_full_coverage.csv')
    common_fields = ['n_swaps', 'n_registered_rectangles', 'n_primary_rectangles']
    classes = coverage.drop(columns=common_fields).groupby('rxn_class', as_index=False).sum(numeric_only=True)
    class_scope = coverage.drop_duplicates('dataset_id').groupby('rxn_class', as_index=False)[common_fields].sum()
    classes = classes.merge(class_scope, on='rxn_class', validate='one_to_one')
    save(classes, 'reaction_class_full_coverage.csv')
    transition_rows = []
    for label, part in [('all', summaries), ('exact', summaries[summaries.has_existing_exact_neighbor_rule])]:
        for tag in ['continuous', '5pp', '10pp']:
            for column in part:
                if column.startswith('n_') and column.endswith('_' + tag) and any((column.startswith('n_' + category) for category in ['reversal_', 'established_', 'eliminated_', 'retained_', 'neither_strong_matching'])):
                    transition_rows.append(dict(subset=label, threshold=tag, transition=column[2:-(len(tag) + 1)], n_partner_observations=int(part[column].sum()), denominator_partner_observations=int(part.n_common_partners.sum())))
    save(pd.DataFrame(transition_rows), 'all_partner_transition_counts.csv')
    edit_rule_rows = []
    for relation, part in summaries[summaries.has_existing_exact_neighbor_rule].groupby('structural_relation', sort=True):
        edit_rule_rows.append(dict(structural_relation=relation, **overview(part)))
    save(pd.DataFrame(edit_rule_rows), 'exact_edit_rule_response_summary.csv')
    exact_pairs = read('round7/partner_conditioned_edits/exact_edit_all_partner_pair_comparisons.csv.gz')
    exact_pairs['partner_similarity_bin'] = pd.cut(exact_pairs.partner_tanimoto, [-EPS, 0.25, 0.5, 0.75, 1 + EPS], labels=['0_to_0.25', '0.25_to_0.5', '0.5_to_0.75', '0.75_to_1.0'])
    distance_rows = []
    for grouping in ['partner_similarity_bin', 'partner_structural_relation']:
        for label, part in exact_pairs.groupby(grouping, observed=True, sort=True):
            distance_rows.append(dict(grouping=grouping, group=str(label), n_partner_pairs=len(part), n_edit_rectangle_groups=part.edit_rectangle_id.nunique(), n_focal_edits=part.edit_id.nunique(), median_absolute_delta_DD_difference=part.delta_DD_partner1_minus_partner0.abs().median(), n_opposite_delta_DD_each_ge10pp=int(part.opposite_delta_DD_each_ge10pp.sum()), n_opposite_component_preference_each_ge10pp=int(part.any_component_structure_preference_opposite_10pp.sum()), n_different_transition_5pp=int((~part.same_transition_5pp).sum()), n_at_least_one_matching_change_5pp=int(part.at_least_one_matching_change_5pp.sum())))
    save(pd.DataFrame(distance_rows), 'exact_edit_partner_distance_response_summary.csv')
    scale = json.loads((OUT / 'initial_scale.json').read_text(encoding='utf-8'))
    assert n_observations == scale['all_partner_observations'] and n_pairs == scale['exact_partner_pairs']
    summary = dict(**scale, all=overview(summaries), exact=overview(summaries[summaries.has_existing_exact_neighbor_rule]), exact_partner_pairs_with_exact_partner_edit=int(exact_pairs.partner_has_exact_rule.sum()), exact_partner_pairs_opposite_delta_DD_each_ge10pp=int(exact_pairs.opposite_delta_DD_each_ge10pp.sum()), exact_partner_pairs_different_transition_5pp=int((~exact_pairs.same_transition_5pp).sum()))
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('AGGREGATES', packed(summary), flush=True)
    return summary

def select_complete_examples(summaries, targets, rectangles, relations):
    exact = summaries[summaries.has_existing_exact_neighbor_rule & summaries.n_common_partners.ge(2)].copy()
    exact['n_reversals_5pp'] = exact.n_reversal_00_11_to_10_01_5pp + exact.n_reversal_10_01_to_00_11_5pp
    exact['contains_no_5pp_reversal'] = exact.n_reversals_5pp.eq(0)
    exact = exact.sort_values(['has_establishment_and_elimination_5pp', 'contains_no_5pp_reversal', 'n_distinct_transition_types_5pp', 'delta_DD_range', 'n_common_partners'], ascending=False)
    selected = exact.groupby('rxn_class', sort=False).head(2).copy()
    selected['selection_reason'] = np.select([selected.has_establishment_and_elimination_5pp, selected.n_distinct_transition_types_5pp.gt(1), selected.any_component_structure_preference_opposite_10pp], ['same_directed_edit_establishes_and_eliminates_strong_pairing_across_partners', 'complete_partner_profile_with_multiple_pairing_responses', 'opposite_component_response_at_10pp'], default='complete_partner_profile_in_covered_reaction_class')
    selected = selected.merge(relations.drop(columns=['dataset_id', 'role', 'has_existing_exact_neighbor_rule', 'has_round4_allowed_exact_rule', 'structural_relation']), on='edit_id', validate='many_to_one')
    selected = selected.merge(rectangles[[column for column in RECIPE_FIELDS if column not in selected or column == 'rectangle_id']], on='rectangle_id', validate='many_to_one')
    save(selected, 'selected_complete_local_edit_profiles.csv')
    frames = []
    for rectangle, group in selected.groupby('rectangle_id', sort=False):
        local = targets[targets.rectangle_id.eq(rectangle)]
        for role, part in group.groupby('role', sort=False):
            frames.append(observations(local, role, relations, set(part.edit_id)))
    frame = pd.concat(frames, ignore_index=True)
    assert frame.observation_id.is_unique
    assert len(frame) == selected.n_common_partners.sum()
    save(frame, 'selected_complete_local_edit_partner_observations.csv')
    print('FULL_ENUMERATION_EXAMPLES', len(selected), selected.rxn_class.nunique(), len(frame), flush=True)
    return (selected, frame)

def verify_outputs(targets, summaries):
    source = targets.set_index('target_row_id')
    registry = read('round7/partner_conditioned_edits/molecular_relation_registry.csv.gz').set_index('edit_id')
    counts, lower, upper, declared_counts = (defaultdict(int), {}, {}, {})
    observation_ids, n = (set(), 0)
    transition_counts = defaultdict(int)
    for chunk in pd.read_csv(OUT / 'all_edit_partner_observations.csv.gz', keep_default_na=False, chunksize=50000):
        assert chunk.observation_id.is_unique and (not set(chunk.observation_id) & observation_ids)
        observation_ids.update(chunk.observation_id)
        chemical = registry.loc[chunk.edit_id]
        for side in ['a', 'b']:
            local = source.loc[chunk['target_row_id_' + side]]
            expected = local[Y].to_numpy(float)
            assert np.allclose(expected, chunk[[column + '_' + side for column in Y]], atol=EPS)
            assert np.array_equal(local.rectangle_id.to_numpy(), chunk.rectangle_id.to_numpy())
            assert np.array_equal(local.dataset_id.to_numpy(), chunk.dataset_id.to_numpy())
            role0 = chunk.role.eq(chemistry.ROLES[0]).to_numpy()
            actual_focal = np.where(role0, local[chemistry.ROLES[0]], local[chemistry.ROLES[1]])
            actual_partner = np.where(role0, local[chemistry.ROLES[1]], local[chemistry.ROLES[0]])
            assert np.array_equal(actual_focal, chemical['structural_molecule_' + side].to_numpy())
            assert np.array_equal(actual_partner, chunk.partner_smiles.to_numpy())
        expected = response_metrics(chunk[[column + '_a' for column in Y]], chunk[[column + '_b' for column in Y]])
        for key, values in expected.items():
            if key.startswith('transition_'):
                assert (chunk[key].to_numpy() == values).all()
                for label, value in chunk[key].value_counts().items():
                    transition_counts[key[11:], label] += int(value)
            else:
                assert np.allclose(chunk[key], values, atol=EPS), key
        groups = chunk.groupby('edit_rectangle_id', sort=False).agg(n=('observation_id', 'size'), minimum=('delta_DD', 'min'), maximum=('delta_DD', 'max'), declared_minimum=('n_common_partners', 'min'), declared_maximum=('n_common_partners', 'max'))
        assert np.array_equal(groups.declared_minimum, groups.declared_maximum)
        for key, number, minimum, maximum, declared, _ in groups.itertuples(name=None):
            assert key not in declared_counts or declared_counts[key] == declared
            declared_counts[key] = int(declared)
            counts[key] += number
            lower[key] = min(lower.get(key, np.inf), minimum)
            upper[key] = max(upper.get(key, -np.inf), maximum)
        n += len(chunk)
        if n % 500000 == 0:
            print('READBACK', n, flush=True)
    assert n == summaries.n_common_partners.sum()
    assert len(counts) == len(summaries)
    assert np.array_equal(summaries.edit_rectangle_id.map(counts), summaries.n_common_partners)
    assert counts == declared_counts
    assert np.allclose(summaries.edit_rectangle_id.map(lower), summaries.minimum_delta_DD, atol=EPS)
    assert np.allclose(summaries.edit_rectangle_id.map(upper), summaries.maximum_delta_DD, atol=EPS)
    for (tag, label), count in transition_counts.items():
        assert summaries['n_' + label + '_' + tag].sum() == count
    pairs = read('round7/partner_conditioned_edits/exact_edit_all_partner_pair_comparisons.csv.gz')
    assert pairs.partner_pair_comparison_id.is_unique
    for index in [0, 1]:
        a = source.loc[pairs[f'target_row_id_a_partner{index}'], Y].to_numpy(float)
        b = source.loc[pairs[f'target_row_id_b_partner{index}'], Y].to_numpy(float)
        expected = response_metrics(a, b)
        assert np.allclose(expected['delta_DD'], pairs[f'delta_DD_partner{index}'], atol=EPS)
        for tag in ['5pp', '10pp']:
            assert (expected['transition_' + tag] == pairs[f'transition_partner{index}_' + tag]).all()
    assert np.allclose(pairs.delta_DD_partner1_minus_partner0, pairs.delta_DD_partner1 - pairs.delta_DD_partner0, atol=EPS)
    blocks = read('round7/partner_conditioned_edits/all_complete_target_acquisition_blocks.csv.gz')
    rebuilt = blocks.groupby('target_row_id')[Y].mean().reindex(source.index)
    assert np.allclose(rebuilt, source[Y], atol=EPS)
    links = read('round7/partner_conditioned_edits/all_four_cell_original_record_links.csv.gz')
    mean = links.groupby(['target_row_id', 'rectangle_id', 'archive', 'plate', 'acquisition', 'corner']).yield_numeric.transform('mean')
    assert np.allclose(mean, links.matched_cell_yield, atol=EPS)
    print('VERIFIED', n, len(counts), len(pairs), len(blocks), len(links), flush=True)
    return dict(observations_verified=n, edit_rectangle_groups_verified=len(counts), exact_partner_pairs_verified=len(pairs), complete_four_cell_blocks_verified=len(blocks), original_record_links_verified=len(links), unique_original_records=links.record_id.nunique())

def report(summary):
    exact, all_groups = (summary['exact'], summary['all'])
    priority = read('round7/partner_conditioned_edits/selected_complete_local_edit_profiles.csv')
    observations_table = read('round7/partner_conditioned_edits/selected_complete_local_edit_partner_observations.csv')
    n_priority_seeded = len(read('round7/partner_conditioned_edits/priority_seeded_exact_edits_all_partner_summary.csv'))
    lines = ['# Component pairing changes of one local structural edit across all reaction partners', '', 'Replacing one H with F on 4-methoxyphenylboronic acid, the same local edit gives six types of pairing response across eight Suzuki reaction partners. Within the same four-recipe rectangle formed by Cs2CO3/K3PO4 and PCy3 Pd G3/Pd(DPEPhos)Cl2, the edit can create a strong pairing in either of two opposite directions, or remove either of them; one partner keeps an existing strong pairing, and three partners have no strong pairing before or after the edit. None of the eight partners shows a 5 percentage point pairing reversal.', '', f"With a defined molecule A→B and an actual four-recipe rectangle fixed, the same edit can create, remove or reverse a strong pairing on different reaction partners. All {summary['primary_rectangles']} primary-analysis rectangles give {summary['all_partner_observations']:,} partner observations; exact structural rules support {exact['n_edits']} edits, {summary['exact_edit_rectangle_groups']:,} edit × rectangle rows and {summary['exact_partner_observations']:,} partner observations.", '', f"Among exact edits, {exact['n_groups_ge2_partners']:,} edit × rectangle rows have at least two common partners and {exact['n_groups_single_partner']:,} have only one partner. Within the former denominator, {exact['n_groups_opposite_delta_DD_each_ge10pp']:,} groups have ΔDD increased by at least 10 percentage points on one partner and decreased by at least 10 percentage points on another; {exact['n_groups_has_establishment_and_elimination_5pp']:,} groups contain both creation and removal of a strong pairing; {exact['n_groups_multiple_transition_types_5pp']:,} groups contain different 5 percentage point pairing transitions.", '', f"Across all {summary['exact_partner_observations']:,} exact-edit partner observations, under the 5 percentage point definition there are {exact['n_partner_observations_reversal_5pp']:,} reversals, {exact['n_partner_observations_established_5pp']:,} creations, {exact['n_partner_observations_eliminated_5pp']:,} removals and {exact['n_partner_observations_retained_5pp']:,} retained strong pairings. Counts at 10 percentage points and all continuous values are kept.", '', f"Pairwise comparison of all common partners of each exact edit gives {summary['exact_partner_pairs']:,} pairs; in {summary['exact_partner_pairs_with_exact_partner_edit']:,} of them the partners themselves have an exact local structural relation. Partner fingerprint distance, exact edit and the four target row numbers are saved in the same row, so the 16 actual yields of each partner pair can be recovered directly.", '', '## Full-partner chemical examples', '']
    selected = priority.groupby('rxn_class', sort=False).head(1)
    for example in selected.itertuples(index=False):
        local = observations_table[observations_table.edit_rectangle_id.eq(example.edit_rectangle_id)].sort_values('partner_smiles')
        lines += [f'### {example.rxn_class}: {example.structural_change_a_to_b}', '', f'A `{example.structural_molecule_a}` → B `{example.structural_molecule_b}`; rectangle `{example.rectangle_id}`; all {example.n_common_partners} common partners.', '', f'Horizontal axis: {example.component0_name} → {example.component1_name}; vertical axis: {example.context0_name} → {example.context1_name}.', '', '| Partner canonical SMILES | A four-cell yields/% | B four-cell yields/% | ΔDD/pp | 5pp pairing change |', '|---|---|---|---:|---|']
        for row in local.itertuples(index=False):
            yields = [', '.join((f"{100 * getattr(row, corner + '_' + side):.2f}" for corner in Y)) for side in ['a', 'b']]
            lines.append(f'| `{row.partner_smiles}` | {yields[0]} | {yields[1]} | {100 * row.delta_DD:.2f} | {row.transition_5pp} |')
        lines += ['', 'Four-cell order is 00, 10, 01, 11; full recipes, edit IDs and target row numbers are in the corresponding CSV.', '']
    lines += ['## Definitions and files', '', 'A and B are ordered by fixed lexicographic order of canonical SMILES, and partners are also matched by canonical identity. DD=Y00−Y10−Y01+Y11; ΔDD=DD_B−DD_A. m0=min(Y00,Y11)−max(Y10,Y01), m1=min(Y10,Y01)−max(Y00,Y11). A 5/10 percentage point strong pairing requires the corresponding margin to be at least 0.05/0.10 and both cells of the preferred diagonal to be ≥20%; creation/removal means that only one side of the edit has a strong pairing, and reversal means that both sides have strong pairings in opposite directions.', '', 'The four structure_preference terms record the effect of A→B on the single-axis yield difference under the same component substitution; the difference between its two backgrounds equals ΔDD exactly. Signs are compared only within the same edit × the same recipe rectangle, and signs across unrelated rectangles are not pooled. Partner similarity is Morgan radius=2/2048-bit Tanimoto and describes structural distance; specific edits and larger structural differences are saved separately.', '', 'The full corpus of 20 datasets, 18 reaction classes and 4,002 single-slot swaps is saved in dataset_role_full_coverage.csv. Single-substrate representations without a non-empty second partner remain in the full-corpus coverage and the all-target four-cell table. Single-partner edits are kept in the denominator and per-row tables and are not counted in cross-partner comparisons.', '', '- molecular_relation_registry.csv.gz: directed molecular structures, exact connectivity changes and structural relations, saved once.', f"- all_primary_target_four_yields.csv.gz, all_primary_recipe_rectangles.csv: {summary['primary_target_rectangle_rows']:,} primary-analysis target × rectangle four-cell sets and {summary['primary_rectangles']} actual recipe rectangles.", f"- all_edit_partner_observations.csv.gz: all {summary['all_partner_observations']:,} partner observations, with eight-cell yields, four margins, ΔDD, transition, edit_id and target row numbers.", '- all_edit_rectangle_partner_summary.csv.gz, exact_edit_rectangle_partner_summary.csv.gz: complete partner denominators and response ranges for all edits and exact edits.', '- exact_edit_all_partner_pair_comparisons.csv.gz: all partner pairs of exact edits and partner structural relations.', '- all_partner_transition_counts.csv, exact_edit_partner_distance_response_summary.csv: transition counts and actual denominators stratified by partner distance.', '- exact_edit_rule_response_summary.csv: all-partner responses for H substitution, CH2 insertion/deletion, aromatic CH/N, position shift and small-fragment single-cut MMP.', f'- priority_seeded_exact_edits_all_partner_summary.csv, priority_seeded_exact_edits_all_partner_observations.csv: starting from the round-6 exact reversal seeds, all common partners are recovered for {n_priority_seeded} instances; for display only, not a replacement for the overall denominator.', '- selected_complete_local_edit_profiles.csv, selected_complete_local_edit_partner_observations.csv: all partners of two local edits selected per class from the complete enumeration, prioritising creation/removal differences within the same edit.', '- all_complete_target_acquisition_blocks.csv.gz, all_four_cell_original_record_links.csv.gz: complete four cells of each target within the same archive/plate/acquisition and the location of each original record.', '']
    (OUT / 'PARTNER_CONDITIONED_EDITS_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def self_checks():
    a = np.array([[0.8, 0.1, 0.2, 0.7], [0.1, 0.8, 0.7, 0.2], [0.8, 0.7, 0.6, 0.5]])
    b = np.array([[0.1, 0.8, 0.7, 0.2], [0.8, 0.1, 0.2, 0.7], [0.8, 0.1, 0.2, 0.7]])
    result = response_metrics(a, b)
    assert result['transition_5pp'].tolist() == ['reversal_00_11_to_10_01', 'reversal_10_01_to_00_11', 'established_00_11']
    assert np.allclose(result['delta_DD'], (b - a) @ SIGNS)
    assert np.allclose(result['delta_DD'], result['structure_preference_focal_bg1'] - result['structure_preference_focal_bg0'])
    assert np.allclose(result['delta_DD'], result['structure_preference_context_c1'] - result['structure_preference_context_c0'])
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    self_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    target_table, rectangle_table, mapping, _ = load()
    relations = read('round7/partner_conditioned_edits/molecular_relation_registry.csv.gz') if (OUT / 'molecular_relation_registry.csv.gz').is_file() else all_relations(target_table)
    if '--profile' in sys.argv:
        profile(target_table, rectangle_table, relations)
    elif '--priority' in sys.argv:
        priority_profiles(target_table, rectangle_table, relations)
    elif '--postprocess' in sys.argv:
        started = time.monotonic()
        summaries = read('round7/partner_conditioned_edits/all_edit_rectangle_partner_summary.csv.gz')
        summaries = summaries.merge(rectangle_table[['rectangle_id', 'focal_slot', 'context_slot']], on='rectangle_id', validate='many_to_one')
        scale = json.loads((OUT / 'initial_scale.json').read_text(encoding='utf-8'))
        summary = aggregates(summaries, target_table, rectangle_table, relations, mapping, scale['all_partner_observations'], scale['exact_partner_pairs'])
        select_complete_examples(summaries, target_table, rectangle_table, relations)
        summary['verification'] = verify_outputs(target_table, summaries)
        summary['postprocess_seconds'] = time.monotonic() - started
        summary['status'] = 'completed'
        report(summary)
        (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('COMPLETED', packed(summary), flush=True)
    else:
        started = time.monotonic()
        if not (OUT / 'initial_scale.json').is_file():
            profile(target_table, rectangle_table, relations)
        if '--resume-sources' not in sys.argv:
            priority_profiles(target_table, rectangle_table, relations)
            reconstruct_all_four_cells(target_table, rectangle_table)
        summaries, n_observations, n_pairs = analyze(target_table, rectangle_table, relations)
        summary = aggregates(summaries, target_table, rectangle_table, relations, mapping, n_observations, n_pairs)
        select_complete_examples(summaries, target_table, rectangle_table, relations)
        summary['verification'] = verify_outputs(target_table, summaries)
        summary['seconds'] = time.monotonic() - started
        summary['status'] = 'completed'
        report(summary)
        (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('COMPLETED', packed(summary), flush=True)
