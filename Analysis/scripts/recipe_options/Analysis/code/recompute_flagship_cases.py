"""Independently calculate named matched cases from full ingested raw records."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ANALYSIS = Path(__file__).resolve().parents[1]
ROOT = ANALYSIS.parents[1]
OUT = ANALYSIS / 'cases'
EMPTY = '<missing>'
AMINE_M = 'COc1ccc(CN)cc1'
AMINE_H = 'NCc1ccccc1'
MG_O = 'Cc1ccccc1[Mg]Br'
MG_P = 'Cc1ccc([Mg]Br)cc1'
FIELDS = [f'reagent{i}_id' for i in range(1, 7)] + ['solvent_id', 'temp_id', 'time_id']

def norm(value):
    if pd.isna(value) or str(value).strip() == '':
        return EMPTY
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value).strip()

def write(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def paired(raw, conditions, family, ca, cb, case, target_slot):
    recipes = conditions.loc[conditions.reaction_type.eq(family.lower()) & conditions.condition_id.isin([ca, cb])].set_index('condition_id')
    assert len(recipes) == 2, (case, len(recipes))
    compare_cols = [f'Reagent{i}_{s}' for i in range(1, 7) for s in ['id', 'HTE_Equiv']]
    compare_cols += ['solvent_id', 'temp_id', 'time_id']
    changed = [col for col in compare_cols if norm(recipes.loc[ca, col]) != norm(recipes.loc[cb, col])]
    assert all((c.startswith(f'Reagent{target_slot}_') for c in changed)), (case, changed)
    frame = raw.loc[raw.rxn_class.eq(family) & raw.condition_id.isin([ca, cb])].copy()
    good = np.ones(len(frame), dtype=bool)
    for field in FIELDS:
        registry_field = field[0].upper() + field[1:] if field.startswith('reagent') else field
        expected = frame.condition_id.map(recipes[registry_field].map(norm))
        good &= frame[field].map(norm).eq(expected).to_numpy()
    conflicts = frame.loc[~good].copy()
    conflicts['case'] = case
    frame = frame.loc[good & np.isfinite(frame.yield_numeric)].copy()
    frame['acquisition'] = frame.name0.str.replace('-C\\d+$', '', regex=True)
    assert frame.acquisition.ne(frame.name0).all(), case
    frame['archive'] = frame.archive.fillna(EMPTY)
    keys = ['subpair', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive', 'plate', 'acquisition']
    dup = frame.duplicated(keys + ['condition_id'], keep=False)
    assert not dup.any(), (case, frame.loc[dup, ['record_id', 'name0']].to_dict('records'))
    keep = keys + ['record_id', 'source_file', 'source_sheet', 'source_row', 'name0', 'condition_id', 'yield_numeric']
    result = frame.loc[frame.condition_id.eq(ca), keep].merge(frame.loc[frame.condition_id.eq(cb), keep], on=keys, suffixes=('_A', '_B'), validate='one_to_one')
    result['delta'] = result.yield_numeric_B - result.yield_numeric_A
    result['delta_pp'] = 100 * result.delta
    result['case'] = case
    recipe_summary = {'case': case, 'family': family, 'condition_A': ca, 'condition_B': cb, 'delta_orientation': f'{cb}-{ca}', 'changed_registry_fields': ';'.join(changed), 'input_rows': len(good), 'fingerprint_conflicts': len(conflicts), 'finite_rows_after_fingerprint': len(frame), 'paired_acquisitions': len(result), 'substrate_pairs': result.subpair.nunique(), 'plates': result.plate.nunique()}
    return (result, conflicts, recipe_summary)

def collapse_pairs(frame):
    return frame.groupby(['case', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles'], as_index=False).agg(delta=('delta', 'mean'), yield_A=('yield_numeric_A', 'mean'), yield_B=('yield_numeric_B', 'mean'), n_acquisitions=('acquisition', 'size'), n_archives=('archive', 'nunique'))

def near_neighbor(frame, molecule_a, molecule_b, label):
    collapsed = collapse_pairs(frame)
    a = collapsed.loc[collapsed.sub_1_smiles.eq(molecule_a)]
    b = collapsed.loc[collapsed.sub_1_smiles.eq(molecule_b)]
    joined = a.merge(b, on=['case', 'sub_2_smiles'], suffixes=('_target', '_control'), validate='one_to_one')
    joined['delta_delta'] = joined.delta_target - joined.delta_control
    joined['delta_delta_pp'] = joined.delta_delta * 100
    joined['structural_comparison'] = label
    joined['opposite_sign_both_above_5pp'] = (joined.delta_target * joined.delta_control < 0) & (joined.delta_target.abs() > 0.05) & (joined.delta_control.abs() > 0.05)
    archive_groups = frame.groupby(['case', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive'], as_index=False).agg(delta=('delta', 'mean'), n_acquisitions=('delta', 'size'))
    archive_joined = archive_groups.loc[archive_groups.sub_1_smiles.eq(molecule_a)].merge(archive_groups.loc[archive_groups.sub_1_smiles.eq(molecule_b)], on=['case', 'sub_2_smiles', 'archive'], suffixes=('_target', '_control'), validate='one_to_one')
    archive_joined['delta_delta'] = archive_joined.delta_target - archive_joined.delta_control
    archive_joined['structural_comparison'] = label
    return (joined, archive_joined)

def classify_acid(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'parse_failed'
    query = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    matches = mol.GetSubstructMatches(query)
    if len(matches) != 1:
        return 'acid_count_ambiguous'
    carbon = mol.GetAtomWithIdx(matches[0][0])
    attached = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() != 8]
    if len(attached) != 1 or not attached[0].GetIsAromatic():
        return 'nonaromatic'
    idx = attached[0].GetIdx()
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if all((mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))]
    system = set(next((r for r in rings if idx in r)))
    while True:
        previous = set(system)
        for ring in rings:
            if system & ring:
                system |= ring
        if previous == system:
            break
    return 'heteroaromatic' if any((mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in system)) else 'carbocyclic_aryl'

def validate_prior(all_neighbors, acid_rows):
    prior_near = pd.read_csv(ROOT / 'output' / 'nchem_chemical_discovery_2026-09-24' / 'amide' / 'methoxy_benzyl_matched_contrasts.csv')
    prior_near['case'] = prior_near.activator + '_NMM_minus_DIPEA'
    prior_validation = all_neighbors.merge(prior_near, left_on=['case', 'sub_2_smiles'], right_on=['case', 'acid'])
    prior_validation['absolute_numeric_error'] = (prior_validation.delta_delta + prior_validation.difference_of_base_effects).abs()
    assert len(prior_validation) == 28
    assert prior_validation.absolute_numeric_error.max() < 1e-12
    write(prior_validation[['case', 'acid', 'delta_delta', 'difference_of_base_effects', 'absolute_numeric_error']], 'prior_case_numeric_checks.csv')
    old_classes = pd.read_csv(ROOT / 'JCIM-Mapping the Substrate Dependence of Amide-Coupling Activators from a High-Throughput Reaction Screen' / 'source_data' / 'Figure2_source_data.csv', usecols=['acid_smiles', 'acid_state'])
    class_validation = acid_rows[['sub_2_smiles', 'acid_class']].drop_duplicates().merge(old_classes, left_on='sub_2_smiles', right_on='acid_smiles')
    class_validation['class_agrees'] = class_validation.acid_state.map({'A': 'heteroaromatic', 'B': 'carbocyclic_aryl', 'C': 'nonaromatic'}).eq(class_validation.acid_class)
    assert class_validation.class_agrees.all(), class_validation.loc[~class_validation.class_agrees]
    write(class_validation, 'acid_class_prior_label_validation.csv')
    print('Optional prior validation passed: 28 numerical contrasts and', len(class_validation), 'acid classes.')

def main(validate_prior_requested=False):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ANALYSIS / 'data' / 'raw_reactions.csv.gz', low_memory=False)
    conditions = pd.read_csv(ANALYSIS / 'data' / 'conditions_raw.csv', low_memory=False)
    configs = [('Acylation', 'C33', 'C61', 'PyAOP_NMM_minus_DIPEA', 3), ('Acylation', 'C3', 'C49', 'BTFFH_NMM_minus_DIPEA', 3), ('Acylation', 'C2', 'C48', 'TFFH_NMM_minus_DIPEA', 3), ('Acylation', 'C4', 'C50', 'DCMT_NMM_minus_DIPEA', 3), ('Acylation', 'C3', 'C41', 'EEDQ_minus_BTFFH', 1), ('Acylation', 'C3', 'C95', 'IIDQ_minus_BTFFH', 1), ('Grignard', 'C2', 'C7', 'CeCl3_THF', 1), ('Grignard', 'C42', 'C43', 'CeCl3_2MeTHF', 1)]
    pairs, conflicts, checks, neighbors, archive_neighbors, summaries = ([], [], [], [], [], [])
    for family, ca, cb, case, target_slot in configs:
        pair, conflict, check = paired(raw, conditions, family, ca, cb, case, target_slot)
        pairs.append(pair)
        conflicts.append(conflict)
        checks.append(check)
        per_pair = collapse_pairs(pair)
        summaries.append({'case': case, 'estimate': 'all_substrate_pairs_equal_weight', 'n': len(per_pair), 'mean_delta_pp': per_pair.delta.mean() * 100, 'positive_n': int(per_pair.delta.gt(0).sum()), 'negative_n': int(per_pair.delta.lt(0).sum())})
        if 'minus_DIPEA' in case:
            neighbor, archive_neighbor = near_neighbor(pair, AMINE_M, AMINE_H, 'para_OMe_benzylamine_minus_benzylamine')
            for amine, name in [(AMINE_M, 'methoxybenzylamine'), (AMINE_H, 'benzylamine')]:
                subset = per_pair.loc[per_pair.sub_1_smiles.eq(amine)]
                summaries.append({'case': case, 'estimate': name + '_all_raw_acids', 'n': len(subset), 'mean_delta_pp': subset.delta.mean() * 100, 'positive_n': int(subset.delta.gt(0).sum()), 'negative_n': int(subset.delta.lt(0).sum())})
        elif family == 'Grignard':
            neighbor, archive_neighbor = near_neighbor(pair, MG_O, MG_P, 'ortho_tolylMgBr_minus_para_tolylMgBr')
        else:
            continue
        neighbors.append(neighbor)
        archive_neighbors.append(archive_neighbor)
        summaries.append({'case': case, 'estimate': neighbor.structural_comparison.iloc[0] if len(neighbor) else 'near_neighbor', 'n': len(neighbor), 'mean_delta_delta_pp': neighbor.delta_delta.mean() * 100, 'positive_n': int(neighbor.delta_delta.gt(0).sum()), 'negative_n': int(neighbor.delta_delta.lt(0).sum()), 'opposite_sign_both_above_5pp_n': int(neighbor.opposite_sign_both_above_5pp.sum())})
    all_pairs = pd.concat(pairs, ignore_index=True)
    all_collapsed = collapse_pairs(all_pairs)
    write(all_pairs, 'paired_acquisitions.csv')
    selected_sources = all_pairs.loc[all_pairs.case.str.contains('minus_DIPEA') & all_pairs.sub_1_smiles.isin([AMINE_M, AMINE_H]) | all_pairs.case.str.startswith('CeCl3') & all_pairs.sub_1_smiles.isin([MG_O, MG_P])]
    write(selected_sources, 'flagship_source_rows.csv')
    write(all_collapsed, 'substrate_pair_means.csv')
    write(pd.concat(conflicts, ignore_index=True), 'raw_fingerprint_conflicts.csv')
    write(pd.DataFrame(checks), 'recipe_and_pairing_checks.csv')
    all_neighbors = pd.concat(neighbors, ignore_index=True)
    write(all_neighbors, 'near_neighbor_common_partners.csv')
    write(pd.concat(archive_neighbors, ignore_index=True), 'near_neighbor_same_archive.csv')
    archive_effects = all_pairs.groupby(['case', 'archive', 'sub_1_smiles', 'sub_2_smiles'], as_index=False).agg(delta=('delta', 'mean'), n_acquisitions=('delta', 'size'))
    write(archive_effects, 'archive_substrate_pair_means.csv')
    repeated = all_collapsed.loc[all_collapsed.n_acquisitions.gt(1)]
    write(repeated, 'repeated_acquisition_substrate_pairs.csv')
    sensitivity = []
    for case, group in all_pairs.groupby('case'):
        collapsed = all_collapsed.loc[all_collapsed.case.eq(case)]
        sensitivity.append({'case': case, 'mean_all_acquisitions_pp': group.delta.mean() * 100, 'mean_substrate_pairs_pp': collapsed.delta.mean() * 100, 'substrate_pairs_repeated': int(collapsed.n_acquisitions.gt(1).sum()), 'acquisitions': len(group), 'substrate_pairs': len(collapsed)})
    write(pd.DataFrame(sensitivity), 'acquisition_weighting_sensitivity.csv')
    background_neighbors = all_neighbors.loc[all_neighbors.case.str.contains('minus_DIPEA')]
    common_acids = set.intersection(*[set(g.sub_2_smiles) for _, g in background_neighbors.groupby('case')])
    common_backgrounds = background_neighbors.loc[background_neighbors.sub_2_smiles.isin(common_acids)]
    write(common_backgrounds, 'near_neighbor_common_all_activators.csv')
    for case, subset in common_backgrounds.groupby('case'):
        summaries.append({'case': case, 'estimate': 'near_neighbor_same_acids_all_four_activators', 'n': len(subset), 'mean_delta_delta_pp': subset.delta_delta.mean() * 100, 'positive_n': int(subset.delta_delta.gt(0).sum()), 'negative_n': int(subset.delta_delta.lt(0).sum())})
    acid_rows = all_collapsed.loc[all_collapsed.case.isin(['EEDQ_minus_BTFFH', 'IIDQ_minus_BTFFH'])].copy()
    acid_rows['acid_class'] = acid_rows.sub_2_smiles.map(classify_acid)
    acid_profiles = acid_rows.groupby(['case', 'sub_2_smiles', 'acid_class'], as_index=False).agg(delta=('delta', 'mean'), yield_A=('yield_A', 'mean'), yield_B=('yield_B', 'mean'), n_amines=('sub_1_smiles', 'nunique'))
    write(acid_profiles, 'acylation_acid_class_profiles.csv')
    for (case, group), subset in acid_profiles.groupby(['case', 'acid_class']):
        summaries.append({'case': case, 'estimate': group + '_acid_equal_weight', 'n': len(subset), 'mean_delta_pp': subset.delta.mean() * 100, 'yield_A_pp': subset.yield_A.mean() * 100, 'yield_B_pp': subset.yield_B.mean() * 100, 'positive_n': int(subset.delta.gt(0).sum()), 'negative_n': int(subset.delta.lt(0).sum())})
    summary = pd.DataFrame(summaries)
    write(summary, 'case_summary.csv')
    if validate_prior_requested:
        validate_prior(all_neighbors, acid_rows)
    print(pd.DataFrame(checks).to_string(index=False))
    print(summary.to_string(index=False))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-prior', action='store_true', help='Also compare outputs with optional historical analysis files.')
    args = parser.parse_args()
    if args.validate_prior:
        raise SystemExit('--validate-prior is not supported in the M1 port')
    main(validate_prior_requested=args.validate_prior)
