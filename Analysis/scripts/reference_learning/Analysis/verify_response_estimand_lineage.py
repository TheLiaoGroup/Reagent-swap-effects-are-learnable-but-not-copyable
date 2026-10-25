"""Independent reconstruction from frozen matrices, contrasts, and registry.

Does not call the report 52 producer or use its comparison/signature functions."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent
TABLES, DATA = (HERE / 'tables', HERE / 'data')
PREFIX = 'response_estimand_lineage_'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sig(mapping):
    rows = [[*key, value.numerator, value.denominator] for key, value in sorted(mapping.items()) if value]
    value = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def add_average(maps):
    result = defaultdict(Fraction)
    for mapping in maps:
        for key, value in mapping.items():
            result[key] += value / Fraction(len(maps))
    return {key: value for key, value in result.items() if value}

def main():
    checks = []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
    manifest = json.loads((DATA / (PREFIX + 'manifest.json')).read_text(encoding='utf-8'))
    for category in ('input_sha256', 'code_sha256', 'output_sha256'):
        check(category, all((sha(HERE / name) == value for name, value in manifest[category].items())))
    atomic = pd.read_csv(TABLES / 'controlled_bridge_atomic_effects.csv.gz', keep_default_na=False)
    contexts = pd.read_csv(TABLES / 'cross_library_components_perturbation_contexts.csv', keep_default_na=False)
    registry = pd.read_csv(TABLES / 'controlled_bridge_directed_bridge_registry.csv', keep_default_na=False)
    endpoints = pd.read_csv(TABLES / (PREFIX + 'endpoints.csv.gz'), keep_default_na=False)
    weights = pd.read_csv(TABLES / (PREFIX + 'atom_weights.csv.gz'), keep_default_na=False)
    coefficients = pd.read_csv(TABLES / (PREFIX + 'yield_coefficients.csv.gz'), keep_default_na=False)
    provenance = pd.read_csv(TABLES / (PREFIX + 'record_provenance.csv'), keep_default_na=False)
    collapsed = pd.read_csv(TABLES / (PREFIX + 'collapsed_units.csv.gz'), keep_default_na=False)
    common = pd.read_csv(TABLES / (PREFIX + 'common_identity_comparisons.csv.gz'), keep_default_na=False)
    configs = pd.read_csv(TABLES / (PREFIX + 'config_comparisons.csv'), keep_default_na=False)
    for column in configs.columns:
        if column.endswith('_rmse_pp'):
            configs[column] = pd.to_numeric(configs[column], errors='coerce')
    matches = pd.read_csv(TABLES / (PREFIX + 'mode_matches.csv'), keep_default_na=False)
    cells = pd.read_csv(TABLES / 'transport_domain_atlas_cells.csv', keep_default_na=False)
    matrices, conditions = ({}, {})
    for screen in contexts.screen.unique():
        path = DATA / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            matrices[screen] = saved['M']
            conditions[screen] = {value: index for index, value in enumerate(saved['conds'].astype(str))}
    context = {r.contrast_id: r for r in contexts.itertuples(index=False)}
    qualified = pd.read_csv(DATA / 'qualified_observations.csv.gz', keep_default_na=False)
    raw = pd.read_csv(DATA / 'raw_long.csv.gz', keep_default_na=False, low_memory=False)
    records = qualified[qualified.partition.eq('analysis')].set_index('record_id')
    raw_records = raw.set_index('record_id')
    provenance_ok = True
    for row in provenance.itertuples(index=False):
        q = records.loc[row.record_id]
        r = raw_records.loc[row.record_id]
        provenance_ok &= q.screen == row.screen and q.subpair == row.original_subpair and (q.condition_id == row.condition) and (q.source_file == row.source_file) and (q.source_sheet == row.source_sheet) and (int(q.source_row) == row.source_row)
        provenance_ok &= r.source_file == row.source_file and r.source_sheet == row.source_sheet and (int(r.source_row) == row.source_row)
        provenance_ok &= abs(float(q['yield']) - row.record_yield) < 1e-12 and abs(float(r.yield_numeric) - row.record_yield) < 1e-12
    observed_means = provenance.assign(weighted_yield=provenance.record_yield * provenance.numerator / provenance.denominator).groupby(['screen', 'pair_index', 'condition']).weighted_yield.sum()
    expected_means = provenance.groupby(['screen', 'pair_index', 'condition']).matrix_yield.first()
    check('qualified_and_raw_record_provenance', provenance_ok and np.allclose(observed_means, expected_means, atol=1e-12, rtol=0), source_records=len(provenance))
    check('all_used_coordinates_have_record_provenance', set(map(tuple, provenance[['screen', 'pair_index', 'condition']].to_numpy())) == set(map(tuple, coefficients[['screen', 'pair_index', 'condition']].to_numpy())))
    errors = []
    for row in atomic.itertuples(index=False):
        c = context[row.contrast_id]
        low = matrices[c.screen][row.pair_index, conditions[c.screen][c.condition_low]]
        high = matrices[c.screen][row.pair_index, conditions[c.screen][c.condition_high]]
        errors.extend([abs(low - row.yield_low), abs(high - row.yield_high), abs(high - low - row.delta)])
    check('all_atomic_values_from_frozen_matrix', len(atomic) == 502945 and np.isfinite(errors).all() and (max(errors) < 1e-12), max_error=float(max(errors)))
    saved_atoms = {identifier: {(r.contrast_id, int(r.pair_index)): Fraction(int(r.numerator), int(r.denominator)) for r in part.itertuples(index=False)} for identifier, part in weights.groupby('unit_id')}
    saved_operators = {identifier: {(r.screen, int(r.pair_index), r.condition): Fraction(int(r.numerator), int(r.denominator)) for r in part.itertuples(index=False)} for identifier, part in coefficients.groupby('unit_id')}
    check('unique_saved_weight_coordinates', not weights.duplicated(['unit_id', 'contrast_id', 'pair_index']).any())
    check('unique_saved_yield_coordinates', not coefficients.duplicated(['unit_id', 'screen', 'pair_index', 'condition']).any())
    by_contrast = {k: part for k, part in atomic.groupby('contrast_id')}
    specs = contexts.to_dict('records')
    endpoint_ok, operator_ok, value_ok = (True, True, True)
    for row in endpoints.itertuples(index=False):
        keycol = 'perturbation_key' if row.mode == 'strict' else 'no_cross_screen_dose_key'
        ids = [c['contrast_id'] for c in specs if c['screen'] == row.screen and c[keycol] == row.perturbation_key and (row.background_control == 'background_free' or c['background_key'] == row.background_key)]
        population = pd.concat([by_contrast[k] for k in ids], ignore_index=True)
        if row.basis == 'full_pair':
            a, b = json.loads(row.unit_identity)
            population = population[population.sub_1.eq(a) & population.sub_2.eq(b)]
        else:
            population = population[population['sub_1' if row.basis == 'position_1' else 'sub_2'].eq(row.unit_identity)]
        pair_counts = Counter(zip(population.sub_1, population.sub_2))
        raw = {(r.contrast_id, int(r.pair_index)): Fraction(1, len(pair_counts) * pair_counts[r.sub_1, r.sub_2]) for r in population.itertuples(index=False)}
        endpoint_ok &= raw == saved_atoms[row.unit_id] and sig(raw) == row.atom_signature and (sum(raw.values()) == 1) and (len(raw) == row.n_atoms)
        expanded = defaultdict(Fraction)
        for (contrast, index), weight in raw.items():
            c = context[contrast]
            expanded[c.screen, index, c.condition_low] -= weight
            expanded[c.screen, index, c.condition_high] += weight
        expanded = {key: value for key, value in expanded.items() if value}
        operator_ok &= expanded == saved_operators[row.unit_id] and sig(expanded) == row.operator_signature and (sum(expanded.values()) == 0)
        value = sum((float(weight) * matrices[screen][index, conditions[screen][condition]] for (screen, index, condition), weight in expanded.items()))
        value_ok &= abs(value - row.saved_observed) < 1e-12
    check('independent_all_endpoint_atom_weights', endpoint_ok, endpoints=len(endpoints))
    check('independent_all_endpoint_yield_operators', operator_ok)
    check('all_endpoint_operators_reproduce_response', value_ok)
    endpoint_lookup = endpoints.set_index('unit_id')
    check('full_registry_both_endpoints_resolved', len(registry) == 4828 and set(registry.source_unit_id) | set(registry.target_unit_id) == set(endpoints.unit_id))
    check('directed_source_and_target_response_values', np.allclose(registry.source_unit_id.map(endpoint_lookup.saved_observed), registry.source_conditioned, atol=1e-12, rtol=0) and np.allclose(registry.target_unit_id.map(endpoint_lookup.saved_observed), registry.observed, atol=1e-12, rtol=0))
    mode_ok = True
    for row in matches.itertuples(index=False):
        for role in ('source', 'target'):
            a, b = (getattr(row, role + '_unit_id_strict'), getattr(row, role + '_unit_id_dose_relaxed'))
            mode_ok &= (saved_atoms[a] == saved_atoms[b]) == getattr(row, role + '_same_atom_weights')
            mode_ok &= (saved_operators[a] == saved_operators[b]) == getattr(row, role + '_same_yield_operator')
    check('independent_cross_mode_both_endpoint_comparisons', mode_ok)
    data = registry.copy()
    data['chemical_key'] = data.perturbation_key.map(lambda value: json.dumps(json.loads(value)[:3], ensure_ascii=False, separators=(',', ':')))
    match_keys = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation', 'background_control', 'background_key', 'basis', 'unit_identity', 'chemical_key']
    expected = data[data['mode'].eq('strict')].merge(data[data['mode'].eq('dose_relaxed')], on=match_keys, suffixes=('_strict', '_dose_relaxed'), validate='many_to_one')
    id_columns = [role + '_unit_id_' + mode for role in ('source', 'target') for mode in ('strict', 'dose_relaxed')]
    check('independent_registered_cross_mode_matching_complete', len(matches) == len(expected) == 2014 and set(map(tuple, matches[id_columns].to_numpy())) == set(map(tuple, expected[id_columns].to_numpy())))
    legacy = pd.read_csv(TABLES / 'transport_domain_atlas_mode_semantics.csv', keep_default_na=False)
    meta = matches[matches.declared_metadata_match_report48]
    check('report48_metadata_matches_all_resolved', len(meta) == len(legacy) == 1970 and meta.same_both_yield_operators.all())
    pair = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
    setting = ['mode', 'background_control', 'basis', 'min_partners_each', 'partner_overlap']
    collapsed_ok = True
    rebuilt = {}
    for row in collapsed.to_dict('records'):
        mask = pd.Series(True, index=registry.index)
        for key in pair + ['mode', 'background_control', 'basis', 'unit_identity']:
            mask &= registry[key].eq(row[key])
        mask &= registry.source_n_partners.ge(row['min_partners_each']) & registry.target_n_partners.ge(row['min_partners_each'])
        if row['partner_overlap'] == 'no_shared_partner':
            mask &= registry.shared_partner_count.eq(0)
        selected = registry[mask]
        collapsed_ok &= len(selected) == row['n_registry_rows']
        collapsed_ok &= abs(selected.source_conditioned.mean() - row['source_value']) < 1e-12 and abs(selected.observed.mean() - row['target_value']) < 1e-12
        for role in ('source', 'target'):
            ids = selected[role + '_unit_id'].tolist()
            operator = add_average([saved_operators[k] for k in ids])
            atom = add_average([saved_atoms[k] for k in ids])
            collapsed_ok &= sig(operator) == row[role + '_operator_signature'] and sig(atom) == row[role + '_atom_signature'] and (ids == json.loads(row[role + '_endpoint_ids_json']))
            rebuilt[tuple((row[k] for k in pair + setting + ['unit_identity'])) + (role,)] = operator
    check('independent_collapsed_membership_and_weights', collapsed_ok, collapsed_units=len(collapsed))
    common_ok = True
    for row in common.to_dict('records'):
        for role in ('source', 'target'):
            base = tuple((row[k] for k in pair))
            tail = tuple((row[k] for k in ['background_control', 'basis', 'min_partners_each', 'partner_overlap', 'unit_identity'])) + (role,)
            equal = rebuilt[base + ('strict',) + tail] == rebuilt[base + ('dose_relaxed',) + tail]
            common_ok &= equal == row[role + '_same_operator']
    check('independent_collapsed_both_operator_comparisons', common_ok)
    config_ok = True
    comparison_keys = pair + ['background_control', 'basis', 'min_partners_each', 'partner_overlap']
    for row in configs.to_dict('records'):
        cm = pd.Series(True, index=common.index)
        um = pd.Series(True, index=collapsed.index)
        for key in comparison_keys:
            cm &= common[key].eq(row[key])
            um &= collapsed[key].eq(row[key])
        subset = common[cm]
        units = collapsed[um]
        left = set(units.loc[units['mode'].eq('strict'), 'unit_identity'])
        right = set(units.loc[units['mode'].eq('dose_relaxed'), 'unit_identity'])
        config_ok &= row['n_common_identities'] == len(subset) == len(left & right)
        config_ok &= row['n_strict_identities'] == len(left) and row['n_relaxed_identities'] == len(right)
        config_ok &= row['n_same_both_operators'] == int(subset.same_both_operators.sum()) and row['n_same_target_operator'] == int(subset.target_same_operator.sum())
        config_ok &= row['same_entire_response_population'] == bool(len(subset) > 0 and left == right and subset.same_both_operators.all())
        config_ok &= row['same_entire_target_population'] == bool(len(subset) > 0 and left == right and subset.target_same_operator.all())
        for population, flag in (('both_operators_fixed', 'same_both_operators'), ('target_operator_fixed', 'target_same_operator')):
            ids = set(subset.loc[subset[flag], 'unit_identity'])
            config_ok &= row[population + '_n_identities'] == len(ids)
            if not ids:
                continue
            for mode in ('strict', 'dose_relaxed'):
                scores = units[units['mode'].eq(mode) & units.unit_identity.isin(ids)]
                source = 100 * np.sqrt(np.mean((scores.source_value - scores.target_value) ** 2))
                zero = 100 * np.sqrt(np.mean(scores.target_value ** 2))
                config_ok &= abs(source - row[population + '_' + mode + '_source_rmse_pp']) < 1e-10 and abs(zero - row[population + '_' + mode + '_zero_rmse_pp']) < 1e-10
    check('independent_all_configuration_populations_and_errors', config_ok)
    finite = configs[configs.both_operators_fixed_n_identities.gt(0)]
    check('identical_operators_identical_source_and_zero_errors', np.allclose(finite.both_operators_fixed_strict_source_rmse_pp, finite.both_operators_fixed_dose_relaxed_source_rmse_pp, atol=1e-10, rtol=0) and np.allclose(finite.both_operators_fixed_strict_zero_rmse_pp, finite.both_operators_fixed_dose_relaxed_zero_rmse_pp, atol=1e-10, rtol=0))
    target = configs[configs.target_operator_fixed_n_identities.gt(0)]
    check('fixed_target_common_population_zero_errors_identical', np.allclose(target.target_operator_fixed_strict_zero_rmse_pp, target.target_operator_fixed_dose_relaxed_zero_rmse_pp, atol=1e-10, rtol=0))
    check('all_atlas_pairs_and_admissible_configurations', len(configs) == 2340 and len(configs[pair].drop_duplicates()) == 90 and (len(cells) == 4680))
    dose = pd.read_csv(TABLES / 'transport_domain_atlas_dose_contrasts.csv', keep_default_na=False)
    transitions = configs.merge(dose[comparison_keys + ['outcome_transition']], on=comparison_keys, validate='one_to_one')
    reversals = transitions[transitions.outcome_transition.eq('positive_negative_reversal')]
    check('all_saved_reversals_change_response_population', len(reversals) == 22 and (not reversals.same_entire_response_population.any()) and transitions.loc[transitions.same_entire_response_population, 'outcome_transition'].eq('same_outcome').all())
    with Image.open(HERE / 'figures/41_RESPONSE_ESTIMAND_LINEAGE.png') as image:
        pixels = np.asarray(image.convert('RGB'))
    check('nonblank_figure', pixels.shape[0] >= 700 and pixels.shape[1] >= 2000 and (pixels.std() > 10))
    check('unique_check_names', len({c['check'] for c in checks}) == len(checks))
    result = dict(passed=all((c['passed'] for c in checks)), checks_total=len(checks), checks_passed=sum((c['passed'] for c in checks)), checks=checks, counts=manifest['counts'])
    (DATA / (PREFIX + 'verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
