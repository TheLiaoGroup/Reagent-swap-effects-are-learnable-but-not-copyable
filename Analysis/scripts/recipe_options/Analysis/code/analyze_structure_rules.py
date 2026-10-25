"""Exploratory molecule-weighted structural contrasts in matched component swaps."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor
ROOT = Path(__file__).resolve().parents[1]
PARTNERS = ['sub_1_smiles', 'sub_2_smiles']
EXPLORATORY_LABEL = 'exploratory_full_data_discovery'
POLICY_LABEL = 'internal_grouped_exploratory_after_full_data_swap_selection'
META_COLUMNS = ['pair_id', 'dataset_id', 'slot', 'exchange_id', 'background_id', 'swap_kind', 'component0_name', 'component1_name', 'component0_id', 'component1_id', 'component0_dose', 'component1_dose', 'condition0_ids', 'condition1_ids', 'recipe0', 'recipe1', 'background_json']
NAMED_RULES = [('acylation_DCC_PFP_to_NHS_aniline', 'p_118f9aa6cf311308', 'sub_1_smiles', 'aniline'), ('acylation_DEPBT_to_TNTU_primary_alkylamine', 'p_250d0c8c4d06f62d', 'sub_1_smiles', 'primary_alkylamine'), ('acylation_TNTU_to_TOTU_primary_alkylamine', 'p_2fa8452dc753300a', 'sub_1_smiles', 'primary_alkylamine'), ('Suzuki_PCy3_to_RuPhos_ortho_ringN', 'p_6742d21c6fcc9bcd', 'sub_1_smiles', 'ortho_ringN_to_leaving_group'), ('SNAr_KHCO3_NMP_to_tAmOH_aniline', 'p_bf73da5201e11aae', 'sub_1_smiles', 'aniline'), ('SNAr_KHCO3_DMSO_to_tAmOH_aniline', 'p_3711390ed3982f89', 'sub_1_smiles', 'aniline')]

def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    if len(order):
        values = p[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(values[::-1])[::-1])
    return adjusted

def welch_contrast(group0, group1) -> dict:
    first, second = (np.asarray(group0, float), np.asarray(group1, float))
    mean0, mean1 = (float(first.mean()), float(second.mean()))
    effect = mean1 - mean0
    variance0, variance1 = (float(first.var(ddof=1)), float(second.var(ddof=1)))
    if variance0 == 0 and variance1 == 0:
        pvalue = 1.0 if effect == 0 else 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pvalue = float(stats.ttest_ind(second, first, equal_var=False).pvalue)
    se_squared = variance0 / len(first) + variance1 / len(second)
    if se_squared > 0:
        degrees = se_squared ** 2 / ((variance0 / len(first)) ** 2 / (len(first) - 1) + (variance1 / len(second)) ** 2 / (len(second) - 1))
        half_width = float(stats.t.ppf(0.975, degrees) * np.sqrt(se_squared))
    else:
        half_width = 0.0
    return {'mean_delta_group0': mean0, 'mean_delta_group1': mean1, 'effect_feature1_minus0': effect, 'abs_effect': abs(effect), 'effect_ci95_low': effect - half_width, 'effect_ci95_high': effect + half_width, 'welch_p': pvalue}

def vectorized_contrasts(outcomes, matrix):
    outcomes, matrix = (np.asarray(outcomes, float), np.asarray(matrix, float))
    mask0, mask1 = (matrix == 0, matrix == 1)
    n0, n1 = (mask0.sum(axis=0), mask1.sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean0, mean1 = (mask0.T @ outcomes / n0, mask1.T @ outcomes / n1)
        variance0 = np.maximum(0, (mask0.T @ outcomes ** 2 - n0 * mean0 ** 2) / (n0 - 1))
        variance1 = np.maximum(0, (mask1.T @ outcomes ** 2 - n1 * mean1 ** 2) / (n1 - 1))
        standard_error2 = variance0 / n0 + variance1 / n1
        degrees = standard_error2 ** 2 / ((variance0 / n0) ** 2 / (n0 - 1) + (variance1 / n1) ** 2 / (n1 - 1))
        effect = mean1 - mean0
        pvalue = 2 * stats.t.sf(np.abs(effect / np.sqrt(standard_error2)), degrees)
        half_width = stats.t.ppf(0.975, degrees) * np.sqrt(standard_error2)
    constant = standard_error2 == 0
    pvalue[constant] = np.where(effect[constant] == 0, 1.0, 0.0)
    half_width[constant] = 0
    return {'n0': n0, 'n1': n1, 'mask0': mask0, 'mask1': mask1, 'mean_delta_group0': mean0, 'mean_delta_group1': mean1, 'effect_feature1_minus0': effect, 'abs_effect': np.abs(effect), 'effect_ci95_low': effect - half_width, 'effect_ci95_high': effect + half_width, 'welch_p': pvalue}

def collapse_acquisitions(delta: pd.DataFrame) -> pd.DataFrame:
    definitions = delta.groupby(['pair_id', 'subpair'], sort=False).target_key.nunique()
    keys = ['pair_id', 'dataset_id', 'target_key', 'subpair', 'sub_1_smiles', 'sub_2_smiles']
    collapsed = delta.groupby(keys, sort=False, dropna=False).agg(y0=('y0', 'mean'), y1=('y1', 'mean'), delta=('delta', 'mean'), n_acquisition_comparisons=('delta', 'size'), n_target_definitions=('target_key', 'nunique')).reset_index()
    assert collapsed.n_target_definitions.eq(1).all()
    collapsed.attrs['n_swap_subpairs_with_multiple_product_target_definitions'] = int(definitions.gt(1).sum())
    return collapsed

def molecule_means(subpairs: pd.DataFrame, partner: str, features: pd.DataFrame) -> pd.DataFrame:
    means = subpairs.groupby(partner, sort=False, dropna=False).agg(delta=('delta', 'mean'), y0=('y0', 'mean'), y1=('y1', 'mean'), n_subpairs=('subpair', 'size'), n_acquisition_comparisons=('n_acquisition_comparisons', 'sum')).reset_index().rename(columns={partner: 'smiles'})
    return means.merge(features, on='smiles', how='left', validate='one_to_one')

def scan_tests(subpairs, summaries, features, binary_features, min_molecules=5):
    all_tests, molecule_tables = ([], {})
    metadata = summaries.set_index('pair_id', drop=False)
    for index, (pair_id, pair) in enumerate(subpairs.groupby('pair_id', sort=False), start=1):
        meta = metadata.loc[pair_id]
        if int(meta.n_substrate_targets) < 20:
            continue
        for partner in PARTNERS:
            molecules = molecule_means(pair, partner, features)
            molecules = molecules[molecules.parsing_status.eq('valid')].copy()
            molecule_tables[pair_id, partner] = molecules
            vector = vectorized_contrasts(molecules.delta, molecules[binary_features].to_numpy(float))
            n_subpairs0 = vector['mask0'].T @ molecules.n_subpairs.to_numpy(int)
            n_subpairs1 = vector['mask1'].T @ molecules.n_subpairs.to_numpy(int)
            for feature_index in np.flatnonzero((vector['n0'] >= min_molecules) & (vector['n1'] >= min_molecules)):
                feature = binary_features[feature_index]
                n0, n1 = (int(vector['n0'][feature_index]), int(vector['n1'][feature_index]))
                contrast = {key: float(vector[key][feature_index]) for key in ['mean_delta_group0', 'mean_delta_group1', 'effect_feature1_minus0', 'abs_effect', 'effect_ci95_low', 'effect_ci95_high', 'welch_p']}
                mean0, mean1 = (contrast['mean_delta_group0'], contrast['mean_delta_group1'])
                opposite = mean0 * mean1 < 0
                reversal = opposite and min(abs(mean0), abs(mean1)) >= 0.05
                if reversal:
                    sign_contrast = 'feature1_prefers_component1_feature0_prefers_component0' if mean1 > 0 else 'feature1_prefers_component0_feature0_prefers_component1'
                elif opposite:
                    sign_contrast = 'opposite_means_below_5pp_margin'
                else:
                    sign_contrast = 'same_or_zero_mean_sign'
                all_tests.append({**{column: meta[column] for column in META_COLUMNS}, 'partner': partner, 'feature': feature, 'n_molecules_group0': n0, 'n_molecules_group1': n1, 'n_molecules_total': n0 + n1, 'n_subpairs_group0': int(n_subpairs0[feature_index]), 'n_subpairs_group1': int(n_subpairs1[feature_index]), 'n_subpairs_total': len(pair), 'n_pairs': len(pair), 'n_targets_source_summary': int(meta.n_substrate_targets), 'n_acquisition_comparisons': int(pair.n_acquisition_comparisons.sum()), **contrast, 'opposite_group_mean_signs': opposite, 'opposite_group_means_at_least_5pp': reversal, 'sign_group_contrast': sign_contrast, 'strong_rule': contrast['abs_effect'] >= 0.1 and min(n0, n1) >= 8 and reversal, 'analysis_status': EXPLORATORY_LABEL})
        if index % 500 == 0:
            print(f'Scanned {index:,} swaps; {len(all_tests):,} eligible structural tests', flush=True)
    tests = pd.DataFrame(all_tests)
    if tests.empty:
        return (tests, molecule_tables)
    tests['bh_q_all_structure_tests'] = bh_adjust(tests.welch_p)
    tests['bh_family_n_tests'] = len(tests)
    tests['q_below_0_05'] = tests.bh_q_all_structure_tests.lt(0.05)
    tests['minimum_group_n'] = tests[['n_molecules_group0', 'n_molecules_group1']].min(axis=1)
    tests = tests.sort_values(['strong_rule', 'q_below_0_05', 'bh_q_all_structure_tests', 'abs_effect', 'minimum_group_n'], ascending=[False, False, True, False, False], kind='stable').reset_index(drop=True)
    tests.insert(0, 'discovery_rank', np.arange(1, len(tests) + 1))
    return (tests, molecule_tables)

def crossbackground_summary(tests, molecule_tables):
    rows = []
    group_columns = ['dataset_id', 'exchange_id', 'partner', 'feature']
    for group_key, group in tests.groupby(group_columns, sort=False):
        background_counts = group.background_id.nunique()
        if background_counts < 2:
            continue
        observed_backgrounds = {}
        for record in group.to_dict('records'):
            molecules = molecule_tables[record['pair_id'], record['partner']]
            for molecule in molecules.canonical_smiles:
                observed_backgrounds.setdefault(molecule, set()).add(record['background_id'])
        shared = sum((len(backgrounds) > 1 for backgrounds in observed_backgrounds.values()))
        signs = np.sign(group.effect_feature1_minus0.to_numpy(float))
        n_positive, n_negative = (int((signs > 0).sum()), int((signs < 0).sum()))
        n_same = max(n_positive, n_negative)
        first = group.iloc[0]
        rows.append({**dict(zip(group_columns, group_key)), 'component0_name': first.component0_name, 'component1_name': first.component1_name, 'component0_dose': first.component0_dose, 'component1_dose': first.component1_dose, 'n_backgrounds': background_counts, 'n_backgrounds_same_effect_sign': n_same, 'all_effect_signs_agree': n_same == len(group), 'majority_effect_sign': 1 if n_positive >= n_negative else -1, 'fraction_backgrounds_same_effect_sign': n_same / len(group), 'minimum_abs_effect_across_backgrounds': float(group.abs_effect.min()), 'median_abs_effect_across_backgrounds': float(group.abs_effect.median()), 'n_backgrounds_strong_rule': int(group.strong_rule.sum()), 'n_backgrounds_opposite_means_at_least_5pp': int(group.opposite_group_means_at_least_5pp.sum()), 'n_backgrounds_q_below_0_05': int(group.q_below_0_05.sum()), 'n_unique_molecules_union': len(observed_backgrounds), 'n_molecules_shared_across_backgrounds': shared, 'support_label': 'shared_molecules_across_backgrounds' if shared else 'nonoverlapping_molecule_supports', 'interpretation': 'exploratory_crossbackground_support_not_independent_replication', 'pair_ids': ';'.join(group.pair_id), 'background_ids': ';'.join(group.background_id)})
    if not rows:
        return pd.DataFrame(columns=group_columns + ['n_backgrounds'])
    return pd.DataFrame(rows).sort_values(['n_backgrounds_strong_rule', 'all_effect_signs_agree', 'minimum_abs_effect_across_backgrounds', 'n_backgrounds'], ascending=False, kind='stable').reset_index(drop=True)

def grouped_policies(candidates, molecule_tables, binary_features, max_pairs=10):
    fold_rows, prediction_rows, summary_rows = ([], [], [])
    selected = candidates.drop_duplicates('pair_id').head(max_pairs)
    for candidate in selected.to_dict('records'):
        pair_id, partner = (candidate['pair_id'], candidate['partner'])
        molecules = molecule_tables[pair_id, partner].copy().reset_index(drop=True)
        if len(molecules) < 10:
            continue
        feature_names = [feature for feature in binary_features if molecules[feature].notna().all()]
        matrix = molecules[feature_names].to_numpy(float)
        outcomes = molecules.delta.to_numpy(float)
        groups = molecules.canonical_smiles.to_numpy(str)
        splitter = GroupKFold(n_splits=5)
        for fold, (training, heldout) in enumerate(splitter.split(matrix, outcomes, groups), start=1):
            assert not set(groups[training]).intersection(groups[heldout])
            model = DecisionTreeRegressor(max_depth=1, min_samples_leaf=5, random_state=20261016)
            model.fit(matrix[training], outcomes[training])
            predicted = model.predict(matrix[heldout])
            train_mean = float(outcomes[training].mean())
            baseline_action = int(train_mean > 0)
            policy_action = (predicted > 0).astype(int)
            gains = outcomes[heldout] * (policy_action - baseline_action)
            selected_feature = feature_names[model.tree_.feature[0]] if model.tree_.feature[0] >= 0 else 'no_split'
            fold_rows.append({'pair_id': pair_id, 'dataset_id': candidate['dataset_id'], 'partner': partner, 'discovery_feature': candidate['feature'], 'fold': fold, 'n_train_molecules': len(training), 'n_test_molecules': len(heldout), 'train_mean_delta': train_mean, 'train_chosen_fixed_component': baseline_action, 'train_selected_stump_feature': selected_feature, 'policy_gain_vs_train_chosen_fixed': float(gains.mean()), 'mae_stump': float(np.abs(predicted - outcomes[heldout]).mean()), 'mae_train_mean': float(np.abs(train_mean - outcomes[heldout]).mean()), 'evaluation_label': POLICY_LABEL})
            for position, prediction, action, gain in zip(heldout, predicted, policy_action, gains):
                row = molecules.iloc[position]
                prediction_rows.append({'pair_id': pair_id, 'partner': partner, 'smiles': row.smiles, 'canonical_smiles': row.canonical_smiles, 'fold': fold, 'observed_mean_y0': row.y0, 'observed_mean_y1': row.y1, 'observed_molecule_mean_delta': row.delta, 'predicted_delta': prediction, 'train_mean_delta': train_mean, 'policy_component': int(action), 'train_chosen_fixed_component': baseline_action, 'observed_policy_yield': row.y0 + int(action) * row.delta, 'observed_fixed_baseline_yield': row.y0 + baseline_action * row.delta, 'policy_gain_vs_train_chosen_fixed': gain, 'n_subpairs': row.n_subpairs, 'evaluation_label': POLICY_LABEL})
        pair_folds = pd.DataFrame([row for row in fold_rows if row['pair_id'] == pair_id])
        predictions = pd.DataFrame([row for row in prediction_rows if row['pair_id'] == pair_id])
        assert len(predictions) == len(molecules) and predictions.canonical_smiles.is_unique
        summary_rows.append({**{column: candidate[column] for column in META_COLUMNS}, 'partner': partner, 'discovery_feature': candidate['feature'], 'n_molecules': len(molecules), 'n_folds': len(pair_folds), 'mean_fold_policy_gain_vs_train_chosen_fixed': float(pair_folds.policy_gain_vs_train_chosen_fixed.mean()), 'sd_fold_policy_gain': float(pair_folds.policy_gain_vs_train_chosen_fixed.std(ddof=1)), 'pooled_molecule_policy_gain_vs_train_chosen_fixed': float(predictions.policy_gain_vs_train_chosen_fixed.mean()), 'mean_fold_mae_stump': float(pair_folds.mae_stump.mean()), 'mean_fold_mae_train_mean': float(pair_folds.mae_train_mean.mean()), 'n_folds_positive_policy_gain': int(pair_folds.policy_gain_vs_train_chosen_fixed.gt(0).sum()), 'train_selected_features_by_fold': ';'.join(pair_folds.train_selected_stump_feature), 'evaluation_label': POLICY_LABEL})
    return (pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows))

def candidate_sources(root, selected_pairs, delta, summaries):
    source = pd.read_csv(root / 'data' / 'normalized_records.csv.gz', keep_default_na=False, low_memory=False)
    if 'strict_eligible' in source:
        source = source[source.strict_eligible.astype(str).str.lower().eq('true')]
    keys = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
    matched = delta[delta.pair_id.isin(selected_pairs)][keys + ['pair_id']].drop_duplicates()
    recipes = summaries[summaries.pair_id.isin(selected_pairs)][['pair_id', 'recipe0', 'recipe1']].melt(id_vars='pair_id', value_vars=['recipe0', 'recipe1'], var_name='component_role', value_name='recipe_id')
    needed = matched.merge(recipes, on='pair_id', validate='many_to_many')
    for table in (needed, source):
        for key in keys + ['recipe_id']:
            table[key] = table[key].astype(str)
    linked = needed.merge(source, on=keys + ['recipe_id'], how='left', validate='many_to_many')
    assert linked.record_id.notna().all(), 'A selected acquisition/recipe could not link to source records'
    columns = ['pair_id', 'component_role', 'record_id', 'dataset_id', 'condition_id', 'recipe_id', 'archive', 'plate', 'acquisition', 'target_key', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'yield_numeric']
    for optional in ['source_check_true', 'quality_pass', 'strict_eligible']:
        if optional in linked:
            columns.append(optional)
    return linked[columns].drop_duplicates().sort_values(['pair_id', 'component_role', 'record_id'])

def shared_partner_adjustments(tests, subpairs, features):
    strong = tests[tests.strong_rule.astype(str).str.lower().eq('true')]
    selections = [strong.head(50)]
    for kind in ('identity_same_dose', 'solvent_identity'):
        selections.append(strong[strong.swap_kind.eq(kind)].head(15))
    selected = pd.concat(selections).drop_duplicates(['pair_id', 'partner', 'feature'])
    lookup = features.set_index('smiles')
    summary_rows, strata_rows = ([], [])
    grouped_pairs = {pair_id: pair for pair_id, pair in subpairs[subpairs.pair_id.isin(selected.pair_id)].groupby('pair_id', sort=False)}
    for candidate in selected.to_dict('records'):
        pair = grouped_pairs[candidate['pair_id']].copy()
        focal = candidate['partner']
        counterpart = PARTNERS[1] if focal == PARTNERS[0] else PARTNERS[0]
        pair['feature_group'] = pair[focal].map(lookup[candidate['feature']])
        pair = pair[pair.feature_group.isin([0, 1])]
        strata = []
        common_molecules0, common_molecules1 = (set(), set())
        for partner_smiles, group in pair.groupby(counterpart, sort=False):
            group0, group1 = (group[group.feature_group.eq(0)], group[group.feature_group.eq(1)])
            if group0.empty or group1.empty:
                continue
            common_molecules0.update(group0[focal])
            common_molecules1.update(group1[focal])
            record = {'pair_id': candidate['pair_id'], 'dataset_id': candidate['dataset_id'], 'partner': focal, 'feature': candidate['feature'], 'counterpart_smiles': partner_smiles, 'n_focal_molecules_group0': group0[focal].nunique(), 'n_focal_molecules_group1': group1[focal].nunique(), 'mean_delta_group0': float(group0.groupby(focal).delta.mean().mean()), 'mean_delta_group1': float(group1.groupby(focal).delta.mean().mean()), 'within_shared_partner_contrast': float(group1.groupby(focal).delta.mean().mean() - group0.groupby(focal).delta.mean().mean())}
            strata.append(record)
            strata_rows.append(record)
        meta = {column: candidate[column] for column in META_COLUMNS}
        row = {**meta, 'partner': focal, 'feature': candidate['feature'], 'discovery_rank': candidate['discovery_rank'], 'unadjusted_effect_feature1_minus0': candidate['effect_feature1_minus0'], 'unadjusted_mean_delta_group0': candidate['mean_delta_group0'], 'unadjusted_mean_delta_group1': candidate['mean_delta_group1'], 'n_counterpart_partners_total': pair[counterpart].nunique(), 'n_common_partners': len(strata), 'n_distinct_focal_molecules_common_group0': len(common_molecules0), 'n_distinct_focal_molecules_common_group1': len(common_molecules1), 'analysis_status': 'exploratory_equal_weight_common_partner_adjustment'}
        if strata:
            values = pd.DataFrame(strata)
            better_supported = values[values.n_focal_molecules_group0.ge(2) & values.n_focal_molecules_group1.ge(2)]
            effect = float(values.within_shared_partner_contrast.mean())
            mean0, mean1 = (float(values.mean_delta_group0.mean()), float(values.mean_delta_group1.mean()))
            row.update({'adjustment_status': 'common_partners_available', 'shared_partner_adjusted_effect': effect, 'shared_partner_mean_delta_group0': mean0, 'shared_partner_mean_delta_group1': mean1, 'same_direction_as_unadjusted': bool(effect * candidate['effect_feature1_minus0'] > 0), 'common_partner_reversal_at_least_5pp': mean0 * mean1 < 0 and min(abs(mean0), abs(mean1)) >= 0.05, 'n_common_partners_ge2_molecules_per_group': len(better_supported), 'adjusted_effect_partners_ge2_molecules_per_group': float(better_supported.within_shared_partner_contrast.mean()) if len(better_supported) else np.nan, 'fraction_common_partners_same_direction': float((values.within_shared_partner_contrast * candidate['effect_feature1_minus0'] > 0).mean())})
        else:
            row.update({'adjustment_status': 'not_estimable_no_shared_partner', 'shared_partner_adjusted_effect': np.nan, 'shared_partner_mean_delta_group0': np.nan, 'shared_partner_mean_delta_group1': np.nan, 'same_direction_as_unadjusted': np.nan, 'common_partner_reversal_at_least_5pp': np.nan, 'n_common_partners_ge2_molecules_per_group': 0, 'adjusted_effect_partners_ge2_molecules_per_group': np.nan, 'fraction_common_partners_same_direction': np.nan})
        summary_rows.append(row)
    return (pd.DataFrame(summary_rows), pd.DataFrame(strata_rows))

def refine_existing(root):
    root = Path(root)
    results = root / 'results'
    tests = pd.read_csv(results / 'structure_all_tests.csv', keep_default_na=False, low_memory=False)
    features = pd.read_csv(root / 'data' / 'molecular_features.csv', keep_default_na=False)
    dictionary = pd.read_csv(root / 'data' / 'feature_dictionary.csv', keep_default_na=False)
    for feature in dictionary.loc[dictionary.feature_type.eq('binary'), 'feature']:
        features[feature] = pd.to_numeric(features[feature], errors='coerce')
    delta = pd.read_csv(results / 'delta_within_acquisition.csv.gz', keep_default_na=False, low_memory=False)
    subpairs = collapse_acquisitions(delta)
    adjusted, strata = shared_partner_adjustments(tests, subpairs, features)
    adjusted.to_csv(results / 'structure_shared_partner_adjustment.csv', index=False)
    strata.to_csv(results / 'structure_shared_partner_strata.csv', index=False)
    summaries = pd.read_csv(results / 'matched_swap_summary.csv', keep_default_na=False, low_memory=False)
    prior = pd.read_csv(results / 'structure_grouped_policy_summary.csv', keep_default_na=False)
    selected_pairs = set(adjusted.pair_id) | set(prior.pair_id)
    sources = candidate_sources(root, selected_pairs, delta, summaries)
    sources.to_csv(results / 'structure_candidate_sources.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary_path = results / 'structure_scan_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update({'n_candidate_source_links': len(sources), 'n_rules_common_partner_checked': len(adjusted), 'n_rules_with_common_partners': int(adjusted.n_common_partners.gt(0).sum()), 'n_rules_common_partner_direction_preserved': int(adjusted.same_direction_as_unadjusted.eq(True).sum()), 'partner_adjustment': 'within each counterpart partner compare feature-group means; average contrasts equally over counterpart partners represented in both groups; report sensitivity requiring >=2 focal molecules/group per partner', 'inference': 'Welch p values describe exploratory structural-group associations across focal molecule means; no randomized structural-group assignment or causal interpretation'})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'common_partner_rules': len(adjusted), 'estimable_rules': summary['n_rules_with_common_partners'], 'same_direction': summary['n_rules_common_partner_direction_preserved'], 'source_record_links': len(sources)}, indent=2), flush=True)
    named_rule_profiles(root, tests=tests, features=features, delta=delta)

def named_rule_profiles(root, tests=None, features=None, delta=None):
    root = Path(root)
    results = root / 'results'
    if tests is None:
        tests = pd.read_csv(results / 'structure_all_tests.csv', keep_default_na=False, low_memory=False)
    if features is None:
        features = pd.read_csv(root / 'data' / 'molecular_features.csv', keep_default_na=False)
    dictionary = pd.read_csv(root / 'data' / 'feature_dictionary.csv', keep_default_na=False)
    binary_features = dictionary.loc[dictionary.feature_type.eq('binary'), 'feature'].tolist()
    for feature in binary_features:
        features[feature] = pd.to_numeric(features[feature], errors='coerce')
    if delta is None:
        delta = pd.read_csv(results / 'delta_within_acquisition.csv.gz', keep_default_na=False, low_memory=False)
    selected_ids = {rule[1] for rule in NAMED_RULES}
    selected_delta = delta[delta.pair_id.isin(selected_ids)].copy()
    variants = {'primary': selected_delta, 'both_source_check_true': selected_delta[selected_delta.both_source_check_true.astype(str).str.lower().eq('true')]}
    profiles, group_rows, contrast_rows = ([], [], [])
    for variant, records in variants.items():
        subpairs = collapse_acquisitions(records)
        for rule_name, pair_id, partner, feature in NAMED_RULES:
            candidate = tests[tests.pair_id.eq(pair_id) & tests.partner.eq(partner) & tests.feature.eq(feature)]
            assert len(candidate) == 1, (rule_name, len(candidate))
            candidate = candidate.iloc[0]
            means = molecule_means(subpairs[subpairs.pair_id.eq(pair_id)], partner, features)
            means = means[means.parsing_status.eq('valid') & means[feature].isin([0, 1])].copy()
            means.insert(0, 'feature_group', means[feature].astype(int))
            means.insert(0, 'feature', feature)
            means.insert(0, 'partner', partner)
            means.insert(0, 'analysis_variant', variant)
            means.insert(0, 'rule_name', rule_name)
            for column in META_COLUMNS:
                means[column] = candidate[column]
            leading = ['rule_name', 'pair_id', 'dataset_id', 'partner', 'feature', 'feature_group', 'smiles', 'canonical_smiles', 'y0', 'y1', 'delta', 'n_subpairs', 'n_acquisition_comparisons', 'analysis_variant']
            means = means[leading + [column for column in means if column not in leading]]
            assert means.canonical_smiles.is_unique
            np.testing.assert_allclose(means.y1 - means.y0, means.delta, atol=1e-12)
            profiles.append(means)
            for group, molecules in means.groupby('feature_group', sort=True):
                group_rows.append({'rule_name': rule_name, 'pair_id': pair_id, 'dataset_id': candidate.dataset_id, 'partner': partner, 'feature': feature, 'feature_group': group, 'analysis_variant': variant, 'n_molecules': len(molecules), 'mean_y0': float(molecules.y0.mean()), 'mean_y1': float(molecules.y1.mean()), 'mean_delta': float(molecules.delta.mean()), 'n_subpairs': int(molecules.n_subpairs.sum()), 'n_acquisition_comparisons': int(molecules.n_acquisition_comparisons.sum()), 'fraction_delta_positive': float(molecules.delta.gt(0).mean()), 'fraction_delta_negative': float(molecules.delta.lt(0).mean()), 'fraction_delta_above_5pp': float(molecules.delta.gt(0.05).mean()), 'fraction_delta_below_minus5pp': float(molecules.delta.lt(-0.05).mean()), 'n_aniline': int(molecules.aniline.eq(1).sum()), 'n_primary_alkylamine': int(molecules.primary_alkylamine.eq(1).sum()), 'n_aniline_and_primary_alkylamine': int((molecules.aniline.eq(1) & molecules.primary_alkylamine.eq(1)).sum())})
            first = means.loc[means.feature_group.eq(0), 'delta']
            second = means.loc[means.feature_group.eq(1), 'delta']
            assert min(len(first), len(second)) >= 2
            contrast = welch_contrast(first, second)
            if variant == 'primary':
                np.testing.assert_allclose(contrast['effect_feature1_minus0'], candidate.effect_feature1_minus0, atol=1e-12)
            contrast_rows.append({'rule_name': rule_name, **{column: candidate[column] for column in META_COLUMNS}, 'partner': partner, 'feature': feature, 'analysis_variant': variant, 'n_molecules_group0': len(first), 'n_molecules_group1': len(second), **contrast, 'primary_bh_q_all_structure_tests': candidate.bh_q_all_structure_tests, 'primary_bh_family_n_tests': int(candidate.bh_family_n_tests), 'p_value_status': 'primary_global_BH_q; sensitivity_Welch_p_descriptive_only'})
    combined = pd.concat(profiles, ignore_index=True)
    combined[combined.analysis_variant.eq('primary')].to_csv(results / 'structure_named_rule_profiles.csv', index=False)
    combined[combined.analysis_variant.eq('both_source_check_true')].to_csv(results / 'structure_named_rule_sourceflag_profiles.csv', index=False)
    pd.DataFrame(group_rows).to_csv(results / 'structure_named_rule_group_summary.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(results / 'structure_named_rule_contrasts.csv', index=False)
    registry = pd.read_csv(root / 'data' / 'condition_slot_registry.csv', keep_default_na=False)
    recipe_rows = []
    for rule_name, pair_id, partner, feature in NAMED_RULES:
        candidate = tests[tests.pair_id.eq(pair_id) & tests.partner.eq(partner) & tests.feature.eq(feature)].iloc[0]
        for role in (0, 1):
            recipe = registry[registry.dataset_id.eq(candidate.dataset_id) & registry.recipe_id.eq(candidate[f'recipe{role}'])].copy()
            assert not recipe.empty
            recipe.insert(0, 'component_role', f'component{role}')
            recipe.insert(0, 'pair_id', pair_id)
            recipe.insert(0, 'rule_name', rule_name)
            recipe_rows.append(recipe)
    pd.concat(recipe_rows, ignore_index=True).to_csv(results / 'structure_named_rule_recipes.csv', index=False)
    strong = tests.strong_rule.astype(str).str.lower().eq('true')
    significant = pd.to_numeric(tests.bh_q_all_structure_tests).lt(0.05)
    summary_path = results / 'structure_scan_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update({'n_strong_rule_tests_q_below_0_05': int((strong & significant).sum()), 'n_distinct_strong_rule_swaps_q_below_0_05': int(tests.loc[strong & significant, 'pair_id'].nunique()), 'n_named_rule_profiles': len(NAMED_RULES), 'n_primary_named_molecule_profiles': int(combined.analysis_variant.eq('primary').sum())})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'named_rules': len(NAMED_RULES), 'primary_molecule_profiles': summary['n_primary_named_molecule_profiles'], 'strong_and_q_below_0_05': summary['n_strong_rule_tests_q_below_0_05']}, indent=2), flush=True)

def self_test():
    np.testing.assert_allclose(bh_adjust([0.01, 0.04, 0.03, 0.002]), [0.02, 0.04, 0.04, 0.008])
    assert welch_contrast([0] * 5, [0] * 5)['welch_p'] == 1
    assert welch_contrast([0] * 5, [1] * 5)['welch_p'] == 0
    rng = np.random.default_rng(17)
    synthetic = rng.normal(0, 0.2, 30)
    matrix = np.column_stack([np.arange(30) % 2, np.arange(30) < 12]).astype(float)
    vector = vectorized_contrasts(synthetic, matrix)
    for index in range(matrix.shape[1]):
        scalar = welch_contrast(synthetic[matrix[:, index] == 0], synthetic[matrix[:, index] == 1])
        for key in scalar:
            np.testing.assert_allclose(vector[key][index], scalar[key], rtol=1e-10, atol=1e-12)
    data = pd.DataFrame({'pair_id': ['p'] * 4, 'dataset_id': ['test'] * 4, 'subpair': ['A >> X', 'A >> X', 'A >> Y', 'B >> X'], 'sub_1_smiles': ['A', 'A', 'A', 'B'], 'sub_2_smiles': ['X', 'X', 'Y', 'X'], 'target_key': ['AX', 'AX', 'AY', 'BX'], 'delta': [0.1, 0.3, 0.6, -0.2], 'y0': [0.1] * 4, 'y1': [0.2, 0.4, 0.7, -0.1]})
    features = pd.DataFrame({'smiles': ['A', 'B'], 'parsing_status': ['valid', 'valid'], 'feature': [1, 0]})
    collapsed = collapse_acquisitions(data)
    molecules = molecule_means(collapsed, 'sub_1_smiles', features).set_index('smiles')
    assert np.isclose(molecules.loc['A', 'delta'], 0.4)
    assert molecules.loc['A', 'n_subpairs'] == 2
    assert molecules.loc['A', 'n_acquisition_comparisons'] == 3
    assert np.isclose(molecules.loc['B', 'delta'], -0.2)
    alternative_product = data.iloc[[0]].copy()
    alternative_product['target_key'] = 'AX_alternative_product'
    with_alternative = collapse_acquisitions(pd.concat([data, alternative_product], ignore_index=True))
    assert len(with_alternative) == 4
    assert with_alternative.attrs['n_swap_subpairs_with_multiple_product_target_definitions'] == 1
    meta = pd.DataFrame({'pair_id': ['p'], 'dataset_id': ['test']}).set_index('pair_id', drop=False)
    assert meta.loc['p', 'pair_id'] == 'p'
    partner_test = pd.DataFrame({'pair_id': ['p'] * 4, 'dataset_id': ['test'] * 4, 'subpair': ['A >> X', 'B >> Y', 'C >> X', 'D >> X'], 'sub_1_smiles': ['A', 'B', 'C', 'D'], 'sub_2_smiles': ['X', 'Y', 'X', 'X'], 'delta': [0.0, 0.9, 0.2, 0.3]})
    partner_features = pd.DataFrame({'smiles': ['A', 'B', 'C', 'D'], 'feature': [0, 0, 1, 1]})
    candidate = {column: '' for column in META_COLUMNS}
    candidate.update({'pair_id': 'p', 'dataset_id': 'test', 'partner': 'sub_1_smiles', 'feature': 'feature', 'discovery_rank': 1, 'strong_rule': True, 'effect_feature1_minus0': -0.2, 'mean_delta_group0': 0.45, 'mean_delta_group1': 0.25})
    adjusted, _ = shared_partner_adjustments(pd.DataFrame([candidate]), partner_test, partner_features)
    assert adjusted.n_common_partners.iloc[0] == 1
    assert np.isclose(adjusted.shared_partner_adjusted_effect.iloc[0], 0.25)
    assert not adjusted.same_direction_as_unadjusted.iloc[0]

def run(root=ROOT, top_pairs=10):
    root = Path(root)
    results = root / 'results'
    summaries = pd.read_csv(results / 'matched_swap_summary.csv', keep_default_na=False, low_memory=False)
    assert summaries.pair_id.is_unique
    features = pd.read_csv(root / 'data' / 'molecular_features.csv', keep_default_na=False)
    dictionary = pd.read_csv(root / 'data' / 'feature_dictionary.csv', keep_default_na=False)
    binary_features = dictionary.loc[dictionary.feature_type.eq('binary'), 'feature'].tolist()
    for feature in binary_features:
        features[feature] = pd.to_numeric(features[feature], errors='coerce')
    delta = pd.read_csv(results / 'delta_within_acquisition.csv.gz', keep_default_na=False, low_memory=False)
    assert set(delta.pair_id).issubset(set(summaries.pair_id))
    subpairs = collapse_acquisitions(delta)
    print(f'Scanning {summaries.pair_id.nunique():,} swaps, {len(subpairs):,} swap-target means, {len(binary_features)} binary features', flush=True)
    tests, molecule_tables = scan_tests(subpairs, summaries, features, binary_features)
    if tests.empty:
        raise RuntimeError('No eligible structural contrasts found')
    tests.to_csv(results / 'structure_all_tests.csv', index=False)
    candidates = tests[tests.abs_effect.ge(0.1) & tests.opposite_group_means_at_least_5pp].copy()
    candidates.to_csv(results / 'structure_ranked_candidates.csv', index=False)
    print(f'Ranked discovery results written: {len(tests):,} globally adjusted tests, {len(candidates):,} reversals with contrast >=10pp', flush=True)
    crossbackground = crossbackground_summary(tests, molecule_tables)
    crossbackground.to_csv(results / 'structure_crossbackground_support.csv', index=False)
    policy_summary, folds, predictions = grouped_policies(candidates, molecule_tables, binary_features, max_pairs=top_pairs)
    policy_summary.to_csv(results / 'structure_grouped_policy_summary.csv', index=False)
    folds.to_csv(results / 'structure_grouped_policy_folds.csv', index=False)
    predictions.to_csv(results / 'structure_grouped_policy_predictions.csv', index=False)
    selected_pairs = candidates.drop_duplicates('pair_id').head(top_pairs).pair_id.tolist()
    sources = candidate_sources(root, selected_pairs, delta, summaries)
    sources.to_csv(results / 'structure_candidate_sources.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary = {'analysis_status': EXPLORATORY_LABEL, 'n_swaps_scanned': int(summaries.pair_id.nunique()), 'n_binary_features': len(binary_features), 'n_eligible_tests': len(tests), 'n_tests_q_below_0_05': int(tests.q_below_0_05.sum()), 'n_reversal_candidates': len(candidates), 'n_strong_rule_tests': int(tests.strong_rule.sum()), 'n_distinct_strong_rule_swaps': int(tests.loc[tests.strong_rule, 'pair_id'].nunique()), 'n_crossbackground_feature_groups': len(crossbackground), 'n_internal_policy_checks': len(policy_summary), 'n_candidate_source_links': len(sources), 'delta_orientation': 'component1 minus component0; contrast is feature1 mean delta minus feature0 mean delta', 'analysis_unit': 'matched acquisitions averaged by substrate-pair/product target_key; target effects averaged equally within each focal molecule', 'n_swap_subpairs_with_multiple_product_target_definitions': subpairs.attrs['n_swap_subpairs_with_multiple_product_target_definitions'], 'multiple_testing': 'Benjamini-Hochberg across all eligible swap/partner/feature contrasts with >=5 molecules per feature group', 'strong_rule_definition': 'absolute mean contrast >=0.10; >=8 molecules per group; opposing mean deltas each at least 0.05 in magnitude', 'policy_evaluation': POLICY_LABEL, 'policy_definition': 'five folds grouped by canonical focal molecule; depth-1 tree with minimum leaf size5 chooses feature within training; compare heldout molecule outcomes with fixed component chosen by training mean delta', 'source_link': 'acquisition/dataset/archive/plate/target_key and recipe identity linked to normalized_records.record_id'}
    (results / 'structure_scan_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    refine_existing(root)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--top-pairs', type=int, default=10)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--refine-existing', action='store_true')
    parser.add_argument('--profiles-only', action='store_true')
    args = parser.parse_args()
    self_test()
    print('Molecule weighting and global correction tests passed.', flush=True)
    if args.profiles_only:
        named_rule_profiles(args.root)
    elif args.refine_existing:
        refine_existing(args.root)
    elif not args.self_test:
        run(args.root, args.top_pairs)
