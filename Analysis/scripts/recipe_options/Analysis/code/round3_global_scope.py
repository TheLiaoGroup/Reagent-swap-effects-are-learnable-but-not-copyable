"""All-library geometry of matched component effects and checked slot semantics.

The fixed denominator is every one of the existing 4,002 matched swaps. This
script describes observed effects; no p/q or effect-size filter changes that
denominator. Thresholds categorize effects, rather than define eligibility."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ANALYSIS = Path(__file__).resolve().parents[1]
OUT = ANALYSIS / 'round3' / 'global_scope'
THRESHOLDS = [0.05, 0.1]
SLOT_SCOPES = {'Acylation': ['carboxylic-acid activator', 'coupling additive / acyl-transfer catalyst', 'base / nucleophilic promoter'], 'Allylic-substitution': ['metal catalyst source / preformed complex', 'ancillary ligand', 'base / basic salt', 'mixed salt / donor / promoter additives'], 'Buchwald-Hartwig': ['metal catalyst source / preformed ligand-metal complex', 'ancillary ligand / coordination additive', 'base / salt promoter'], 'Buchwald-Hartwig(CMC)': ['Pd catalyst source / preformed ligand-Pd complex', 'ancillary ligand / donor additive', 'base / fluoride salt'], 'C-O competitive reactions': ['metal catalyst source / preformed complex', 'ancillary ligand / coordination additive', 'base', 'iodide additive'], 'Chan-Lam': ['metal salt / complex', 'ancillary ligand / N-donor', 'base / carboxylate / salt', 'oxidant / fluoride / carboxylic-acid additive'], 'Esterification': ['mixed acid / Lewis acid / acyl-transfer catalyst / activator', 'base / acyl-transfer catalyst / activator', 'base / coupling reagent / antioxidant', 'coupling reagent / acid derivative / base / antioxidant', 'coupling additive'], 'Finkelstein': ['metal catalyst source', 'N/P-donor ligand / donor cosolvent additive', 'mixed base / Lewis-acid salt / donor / solvent additive', 'iodide source'], 'Grignard': ['mixed Lewis acid / metal salt / donor / reducing or organoboron reagent', 'Lewis-acid salt'], 'Heck': ['Pd catalyst source / preformed complex', 'ancillary ligand / donor or halide additive', 'base / salt promoter', 'salt / phase-transfer / formate additive'], 'Hiyama': ['metal catalyst source / preformed complex', 'ancillary ligand / donor additive', 'base / salt / silyl reagent', 'fluoride / halide / crown-ether additive'], 'Kumada': ['metal catalyst source / preformed complex', 'ancillary ligand / donor additive', 'salt / donor / reducing / olefin promoter'], 'Minisci': ['metal salt / radical-oxidation promoter', 'oxidant', 'acid / metal-salt additive', 'mixed salt / donor additive'], 'Pd-carbonylation': ['Pd catalyst source / preformed complex', 'additional metal complex / carbonyl complex', 'ancillary ligand / base', 'base', 'carbonate base', 'activator / organic additive'], 'Reductive-amination': ['Lewis-acid promoter', 'empty', 'reducing reagent', 'acid additive'], 'SNAr': ['empty', 'empty', 'acid / base'], 'Sonogashira': ['Pd catalyst source / preformed complex', 'metal cocatalyst source', 'ancillary ligand', 'base', 'mixed salt / Lewis acid / organic additive'], 'Suzuki': ['metal catalyst source / preformed ligand-metal complex', 'ancillary ligand / donor / salt additive', 'base / fluoride salt', 'Cu salt additive'], 'Suzuki Bpin': ['metal catalyst source / preformed complex', 'ancillary ligand', 'base / fluoride salt'], 'Ullmann': ['metal catalyst source / preformed complex', 'ancillary ligand / donor / organic additive', 'base / salt promoter', 'metal-salt / oxide additive']}

def normalize(value):
    if str(value).strip() == '':
        return ''
    try:
        return format(float(value), '.12g')
    except ValueError:
        return str(value).strip()

def role_registry(registry, datasets):
    contents, scopes = ([], [])
    for dataset in datasets:
        frame = registry.loc[registry.dataset_id.eq(dataset)]
        assert len(frame) > 0 and dataset in SLOT_SCOPES
        for number in range(1, 7):
            slot = f'Reagent{number}'
            values = frame.loc[frame[f'{slot}_id'].ne('')]
            if values.empty:
                continue
            scope = SLOT_SCOPES[dataset][number - 1]
            assert scope != 'empty', (dataset, slot)
            for (identity, name, dose), rows in values.groupby([f'{slot}_id', f'{slot}_Standardized Name', f'{slot}_HTE_Equiv']):
                contents.append({'dataset_id': dataset, 'slot': slot, 'component_id': identity, 'registered_name': name, 'registered_equiv': dose, 'chemical_scope': scope, 'conditions': ';'.join(sorted(set(rows.condition_id))), 'registry_source_file': '|'.join(sorted(set(rows.source_file))), 'registry_source_rows': '|'.join((str(x) for x in sorted(set(rows.source_row))))})
            scopes.append({'dataset_id': dataset, 'slot': slot, 'chemical_scope': scope, 'n_distinct_components': values[f'{slot}_id'].nunique(), 'all_registered_names': ' | '.join(sorted(set(values[f'{slot}_Standardized Name']))), 'cross_reaction_role_inference': 'Use dataset + registered contents; slot number alone is not a chemical role.'})
        scopes.append({'dataset_id': dataset, 'slot': 'Solvent', 'chemical_scope': 'reaction medium', 'n_distinct_components': frame.solvent_id.nunique(), 'all_registered_names': ' | '.join(sorted(set(frame['Standardized Name']) - {''})), 'cross_reaction_role_inference': 'Registered solvent identity; no solvent-mechanism assignment.'})
    pd.DataFrame(contents).to_csv(OUT / 'registered_component_role_evidence.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scopes).to_csv(OUT / 'dataset_slot_chemical_scopes.csv', index=False, encoding='utf-8-sig')
    return pd.DataFrame(scopes)

def metrics(frame, weights):
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    d, y0, y1 = (frame[c].to_numpy(float) for c in ['delta', 'y0', 'y1'])
    mu = float(weights @ d)
    variance = float(weights @ (d - mu) ** 2)
    second = float(weights @ d ** 2)
    v0 = float(weights @ (y0 - weights @ y0) ** 2)
    v1 = float(weights @ (y1 - weights @ y1) ** 2)
    midpoint = (y0 + y1) / 2
    vmid = float(weights @ (midpoint - weights @ midpoint) ** 2)
    vwithin = (v0 + v1) / 2
    meanabs = float(weights @ np.abs(d))
    assert abs(second - variance - mu ** 2) < 1e-12
    result = {'mean_delta': mu, 'mean_abs_delta': meanabs, 'sd_delta_population': np.sqrt(variance), 'centered_effect_rms': np.sqrt(variance), 'effect_second_moment': second, 'effect_heterogeneity_share': variance / second if second > 0 else np.nan, 'mean_cancellation_fraction': 1 - abs(mu) / meanabs if meanabs > 0 else 0, 'observed_sign_adaptation_gap': (meanabs - abs(mu)) / 2, 'yield_var0': v0, 'yield_var1': v1, 'yield_var_midpoint': vmid, 'yield_var_paired_within': vwithin, 'H_paired_within': variance / vwithin if vwithin > 0 else np.nan, 'H_baseline': variance / v0 if v0 > 0 else np.nan, 'H_midpoint': variance / vmid if vmid > 0 else np.nan, 'mean_y0': float(weights @ y0), 'mean_y1': float(weights @ y1), 'p_y0_exact_zero': float(weights @ (y0 == 0)), 'p_y1_exact_zero': float(weights @ (y1 == 0))}
    if vwithin > 0:
        assert result['H_paired_within'] <= 4 + 1e-10
    for threshold in THRESHOLDS:
        suffix = str(round(threshold * 100)) + 'pp'
        positive, negative = (d >= threshold, d <= -threshold)
        pp, pn = (float(weights @ positive), float(weights @ negative))
        np_, nn = (int(positive.sum()), int(negative.sum()))
        if np_ and nn:
            category = 'bidirectional'
        elif np_ or nn:
            category = 'one_direction_amplitude_heterogeneous' if np.sqrt(variance) >= threshold else 'one_direction_near_parallel'
        else:
            category = 'below_resolved_effect_threshold'
        result.update({f'p_positive_{suffix}': pp, f'p_negative_{suffix}': pn, f'n_positive_{suffix}': np_, f'n_negative_{suffix}': nn, f'minority_response_fraction_{suffix}': min(pp, pn), f'both_directions_any_{suffix}': bool(np_ and nn), f'both_directions_3targets_{suffix}': bool(np_ >= 3 and nn >= 3), f'both_directions_10pct_{suffix}': bool(pp >= 0.1 - 1e-12 and pn >= 0.1 - 1e-12), f'both_directions_3targets_10pct_{suffix}': bool(np_ >= 3 and nn >= 3 and (pp >= 0.1 - 1e-12) and (pn >= 0.1 - 1e-12)), f'response_category_{suffix}': category})
    return result

def weighted_quantile(values, weights, quantile=0.5):
    values, weights = (np.asarray(values, float), np.asarray(weights, float))
    good = np.isfinite(values)
    values, weights = (values[good], weights[good])
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = (values[order], weights[order])
    return float(values[np.searchsorted(np.cumsum(weights), quantile * weights.sum(), side='left')])

def aggregate(frame, scheme):
    if scheme == 'swap_equal':
        weights = np.ones(len(frame))
    elif scheme == 'target_comparison_weighted':
        weights = frame.n_targets.to_numpy(float)
    elif scheme == 'dataset_equal':
        weights = 1 / frame.groupby('dataset_id').pair_id.transform('size').to_numpy(float)
    elif scheme == 'reaction_class_equal':
        datasets_per_class = frame[['dataset_id', 'rxn_class']].drop_duplicates().groupby('rxn_class').size()
        weights = 1 / frame.groupby('dataset_id').pair_id.transform('size').to_numpy(float)
        weights /= frame.rxn_class.map(datasets_per_class).to_numpy(float)
    else:
        raise ValueError(scheme)
    weights /= weights.sum()
    result = {'aggregation': scheme, 'n_swaps': len(frame), 'n_datasets': frame.dataset_id.nunique(), 'n_rxn_classes': frame.rxn_class.nunique(), 'acylation_weight_fraction': float(weights[frame.dataset_id.eq('Acylation')].sum())}
    for key in ['sd_delta_population', 'centered_effect_rms', 'mean_abs_delta', 'effect_heterogeneity_share', 'mean_cancellation_fraction', 'observed_sign_adaptation_gap', 'H_paired_within']:
        values = frame[key].to_numpy(float)
        defined = np.isfinite(values)
        result['defined_weight_fraction_' + key] = float(weights[defined].sum())
        result['weighted_mean_' + key] = float(np.average(values[defined], weights=weights[defined])) if defined.any() else np.nan
        result['weighted_median_' + key] = weighted_quantile(frame[key], weights)
    result['weighted_mean_abs_mean_delta'] = float(weights @ frame.mean_delta.abs())
    for threshold in THRESHOLDS:
        suffix = str(round(threshold * 100)) + 'pp'
        for field in [f'both_directions_any_{suffix}', f'both_directions_3targets_{suffix}', f'both_directions_10pct_{suffix}', f'both_directions_3targets_10pct_{suffix}', f'minority_response_fraction_{suffix}']:
            result['weighted_mean_' + field] = float(weights @ frame[field])
        for category in ['bidirectional', 'one_direction_amplitude_heterogeneous', 'one_direction_near_parallel', 'below_resolved_effect_threshold']:
            result[f'category_fraction_{suffix}_{category}'] = float(weights @ frame[f'response_category_{suffix}'].eq(category))
    result['weighted_fraction_mean_within2pp_sd_atleast15pp'] = float(weights @ (frame.mean_delta.abs().le(0.02) & frame.sd_delta_population.ge(0.15)))
    return result

def select_illustrations(targets, swaps, target_metrics, registry):
    illustrated, target_tables = ([], [])
    for dataset, group in target_metrics.groupby('dataset_id'):
        complete = group.merge(swaps, on='pair_id', suffixes=('', '_swap'), validate='one_to_one')
        resolved = complete.loc[complete.both_directions_3targets_10pct_5pp]
        preferred = resolved.loc[resolved.swap_kind.isin(['identity_same_dose', 'solvent_identity'])]
        candidates = preferred if len(preferred) else resolved
        if candidates.empty:
            illustrated.append({'dataset_id': dataset, 'illustration_status': 'no swap with >=3 targets and >=10pct on both sides at 5pp'})
            continue
        near_zero = candidates.loc[candidates.mean_delta.abs().le(0.02)]
        selected = near_zero.sort_values(['sd_delta_population', 'minority_response_fraction_5pp'], ascending=False).iloc[0] if len(near_zero) else candidates.assign(abs_mean=candidates.mean_delta.abs()).sort_values('abs_mean').iloc[0]
        result = selected.to_dict()
        result['illustration_status'] = 'posthoc named illustration; full-library denominator unchanged'
        part = targets.loc[targets.pair_id.eq(selected.pair_id)].copy()
        positive = part.loc[part.delta.ge(0.05)].sort_values('delta', ascending=False)
        negative = part.loc[part.delta.le(-0.05)].sort_values('delta')
        for label, subset in [('positive_extreme', positive), ('negative_extreme', negative)]:
            row = subset.iloc[0]
            result.update({label + '_' + col: row[col] for col in ['target_key', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'y0', 'y1', 'delta', 'n_acquisitions', 'archives']})
        reg = registry.loc[registry.dataset_id.eq(dataset)].drop_duplicates('recipe_id')
        for side in [0, 1]:
            recipe = reg.loc[reg.recipe_id.eq(selected[f'recipe{side}'])].iloc[0]
            entries = []
            for i in range(1, 7):
                name = recipe[f'Reagent{i}_Standardized Name']
                if name:
                    entries.append(name + ' ' + str(recipe[f'Reagent{i}_HTE_Equiv']) + ' equiv')
            entries += [str(recipe['Standardized Name']), str(recipe.temp_id) + ' C', str(recipe.time_id) + ' h']
            result[f'full_recipe{side}'] = '; '.join(entries)
        illustrated.append(result)
        part['illustration_component0'] = selected.component0_name
        part['illustration_component1'] = selected.component1_name
        target_tables.append(part)
    pd.DataFrame(illustrated).to_csv(OUT / 'named_swap_illustrations_all_datasets.csv', index=False, encoding='utf-8-sig')
    if target_tables:
        pd.concat(target_tables).to_csv(OUT / 'named_swap_full_target_effects.csv.gz', index=False, encoding='utf-8-sig')
    return pd.DataFrame(illustrated)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    swaps = pd.read_csv(ANALYSIS / 'results/matched_swap_summary.csv', keep_default_na=False)
    delta = pd.read_csv(ANALYSIS / 'results/delta_within_acquisition.csv.gz', keep_default_na=False)
    registry = pd.read_csv(ANALYSIS / 'data/condition_slot_registry.csv', keep_default_na=False)
    dataset_map = pd.read_csv(ANALYSIS / 'data/normalized_records.csv.gz', usecols=['dataset_id', 'rxn_class'], keep_default_na=False).drop_duplicates()
    assert len(swaps) == 4002 and swaps.pair_id.is_unique
    assert dataset_map.dataset_id.is_unique and dataset_map.rxn_class.nunique() == 18
    datasets = sorted(swaps.dataset_id.unique())
    assert len(datasets) == 20
    dataset_map = dataset_map.loc[dataset_map.dataset_id.isin(datasets)]
    dataset_map.to_csv(OUT / 'dataset_reaction_class_mapping.csv', index=False, encoding='utf-8-sig')
    scopes = role_registry(registry, datasets)
    targets = delta.groupby(['dataset_id', 'pair_id', 'target_key'], as_index=False).agg(y0=('y0', 'mean'), y1=('y1', 'mean'), delta=('delta', 'mean'), sub_1_smiles=('sub_1_smiles', 'first'), sub_2_smiles=('sub_2_smiles', 'first'), product_smiles=('product_smiles', 'first'), n_acquisitions=('acquisition', 'size'), archives=('archive', lambda s: '|'.join(sorted(set(s)))), both_source_check_true=('both_source_check_true', 'all'))
    assert len(targets) == 1083041
    assert np.max(np.abs(targets.y1 - targets.y0 - targets.delta)) < 1e-12
    expected_counts = swaps.set_index('pair_id').n_substrate_targets
    assert targets.groupby('pair_id').size().eq(expected_counts).all()
    features = pd.read_csv(ANALYSIS / 'data/molecular_features.csv', keep_default_na=False)
    molecular_ids = features.set_index('smiles').canonical_smiles.to_dict()
    for side in [1, 2]:
        source = f'sub_{side}_smiles'
        targets[f'molecule{side}_id'] = targets[source].map(lambda s: molecular_ids.get(s) or s)
    rows = []
    for number, (pair_id, part) in enumerate(targets.groupby('pair_id', sort=False), 1):
        side1_count = part.groupby('molecule1_id').delta.transform('size').to_numpy(float)
        side2_count = part.groupby('molecule2_id').delta.transform('size').to_numpy(float)
        w1, w2 = (1 / side1_count, 1 / side2_count)
        w1 /= w1.sum()
        w2 /= w2.sum()
        one_substrate = part.molecule2_id.eq('').all()
        weight_sets = {'target_equal': np.ones(len(part)), 'side1_molecule_equal': w1, 'side2_molecule_equal': w2, 'role_balanced_molecule_equal': w1 if one_substrate else (w1 + w2) / 2}
        for weighting, weights in weight_sets.items():
            rows.append({'pair_id': pair_id, 'dataset_id': part.dataset_id.iloc[0], 'within_swap_weighting': weighting, 'n_targets': len(part), 'n_molecules_side1': part.molecule1_id.nunique(), 'n_molecules_side2': part.loc[part.molecule2_id.ne(''), 'molecule2_id'].nunique(), 'one_substrate_array': one_substrate, **metrics(part, weights)})
        if number % 500 == 0:
            print('Measured', number, 'of', len(swaps), 'swaps', flush=True)
    measured = pd.DataFrame(rows).merge(dataset_map, on='dataset_id', validate='many_to_one')
    measured = measured.merge(swaps[['pair_id', 'slot', 'swap_kind', 'component0_name', 'component1_name', 'condition0_ids', 'condition1_ids']], on='pair_id', validate='many_to_one')
    measured = measured.merge(scopes[['dataset_id', 'slot', 'chemical_scope']], on=['dataset_id', 'slot'], how='left', validate='many_to_one')
    assert measured.chemical_scope.notna().all()
    measured.to_csv(OUT / 'all_swap_geometry_by_weighting.csv', index=False, encoding='utf-8-sig')
    target_measured = measured.loc[measured.within_swap_weighting.eq('target_equal')]
    reference = target_measured.merge(swaps[['pair_id', 'mean_delta', 'sd_delta']], on='pair_id', suffixes=('', '_old'))
    max_mean_error = float(np.max(np.abs(reference.mean_delta - reference.mean_delta_old)))
    sample_sd = reference.sd_delta_population * np.sqrt(reference.n_targets / (reference.n_targets - 1))
    max_sd_error = float(np.max(np.abs(sample_sd - reference.sd_delta)))
    assert max_mean_error < 1e-12 and max_sd_error < 1e-12
    summary = []
    dataset_summary = []
    role_summary = []
    for weighting, frame in measured.groupby('within_swap_weighting'):
        for scheme in ['swap_equal', 'target_comparison_weighted', 'dataset_equal', 'reaction_class_equal']:
            summary.append({'within_swap_weighting': weighting, **aggregate(frame, scheme)})
        for dataset, group in frame.groupby('dataset_id'):
            dataset_summary.append({'within_swap_weighting': weighting, 'dataset_id': dataset, **aggregate(group, 'swap_equal')})
        for role, group in frame.groupby('chemical_scope'):
            role_summary.append({'within_swap_weighting': weighting, 'chemical_scope': role, **aggregate(group, 'reaction_class_equal')})
    pd.DataFrame(summary).to_csv(OUT / 'global_weighting_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(dataset_summary).to_csv(OUT / 'all_dataset_geometry.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(role_summary).to_csv(OUT / 'chemical_scope_geometry_descriptive.csv', index=False, encoding='utf-8-sig')
    h = target_measured
    illustrations = select_illustrations(targets, swaps, target_measured, registry)
    support_counts = {}
    for threshold in THRESHOLDS:
        suffix = str(round(threshold * 100)) + 'pp'
        flags = h[f'both_directions_3targets_10pct_{suffix}']
        support_counts[suffix] = {'any_both_directions': int(h[f'both_directions_any_{suffix}'].sum()), 'both_directions_atleast3targets': int(h[f'both_directions_3targets_{suffix}'].sum()), 'both_directions_atleast10pct': int(h[f'both_directions_10pct_{suffix}'].sum()), 'both_directions_atleast3targets_and10pct': int(flags.sum()), 'datasets_with_joint_support': int(h.loc[flags].dataset_id.nunique()), 'reaction_classes_with_joint_support': int(h.loc[flags].rxn_class.nunique())}
    counts = {'matched_swaps': len(swaps), 'datasets': len(datasets), 'reaction_classes': dataset_map.rxn_class.nunique(), 'within_acquisition_delta_rows': len(delta), 'target_product_swap_means': len(targets), 'unique_dataset_targets': int(targets[['dataset_id', 'target_key']].drop_duplicates().shape[0]), 'target_comparisons_reuse_original_records': True, 'support_counts': support_counts, 'mean_within2pp_sd_atleast15pp_swaps': int((h.mean_delta.abs().le(0.02) & h.sd_delta_population.ge(0.15)).sum()), 'H_paired_within_above1': int((h.H_paired_within > 1).sum()), 'H_paired_within_max': float(h.H_paired_within.max()), 'H_baseline_denominator_zero': int(h.yield_var0.eq(0).sum()), 'H_midpoint_denominator_zero': int(h.yield_var_midpoint.eq(0).sum()), 'all_zero_effect_swaps': int(h.effect_second_moment.eq(0).sum()), 'max_existing_mean_delta_error': max_mean_error, 'max_existing_sample_SD_error': max_sd_error, 'max_mean_removed_sd_change': float(np.max(np.abs(h.sd_delta_population - h.centered_effect_rms))), 'n_named_illustrations': int(illustrations.pair_id.notna().sum())}
    (OUT / 'global_scope_summary.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding='utf-8')
    assert len(pd.read_csv(OUT / 'all_swap_geometry_by_weighting.csv')) == len(swaps) * 4
    assert len(pd.read_csv(OUT / 'dataset_reaction_class_mapping.csv')) == 20
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    print(pd.DataFrame(summary).loc[lambda v: v.within_swap_weighting.eq('target_equal'), ['within_swap_weighting', 'aggregation', 'weighted_median_sd_delta_population', 'weighted_mean_both_directions_3targets_10pct_5pp', 'weighted_mean_both_directions_3targets_10pct_10pp', 'acylation_weight_fraction']].to_string(index=False), flush=True)
    print(illustrations[['dataset_id', 'component0_name', 'component1_name', 'mean_delta', 'sd_delta_population', 'n_positive_5pp', 'n_negative_5pp', 'n_targets']].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
