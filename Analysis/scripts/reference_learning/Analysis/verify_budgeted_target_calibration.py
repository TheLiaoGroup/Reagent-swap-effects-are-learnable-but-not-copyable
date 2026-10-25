"""Independent verification of report 57 design, costs, and closed-form fits."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent
TABLES, DATA = (HERE / 'tables', HERE / 'data')
PREFIX = 'budgeted_target_calibration_'
PAIR = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
BUDGETS = [24, 48, 96, 192]
MODELS = ['zero', 'direct_source', 'target_calibration_mean', 'source_offset', 'source_affine_ridge']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def read(suffix):
    return pd.read_csv(TABLES / (PREFIX + suffix), keep_default_na=False)

def flag(value):
    return str(value).lower() == 'true'

def main():
    checks = []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
    manifest = json.loads((DATA / (PREFIX + 'manifest.json')).read_text(encoding='utf-8'))
    for category in ('input_sha256', 'code_sha256', 'output_sha256'):
        check(category, all((sha(HERE / name) == value for name, value in manifest[category].items())))
    support, designs, budgets, scores, predictions = [read(name) for name in ['unit_support.csv.gz', 'designs.csv', 'budget_designs.csv.gz', 'scores.csv', 'predictions.csv.gz']]
    old = pd.read_csv(TABLES / 'response_estimand_lineage_collapsed_units.csv.gz', keep_default_na=False)
    coefficients = pd.read_csv(TABLES / 'response_estimand_lineage_yield_coefficients.csv.gz', keep_default_na=False)
    provenance = pd.read_csv(TABLES / 'response_estimand_lineage_record_provenance.csv', keep_default_na=False)
    atlas = pd.read_csv(TABLES / 'transport_domain_atlas_cells.csv', keep_default_na=False)
    primary = lambda frame: frame['mode'].eq('strict') & frame.background_control.eq('background_free') & frame.basis.eq('position_2') & frame.min_partners_each.eq(1) & frame.partner_overlap.eq('all')
    old, atlas = (old[primary(old)], atlas[primary(atlas)])
    ep = {identifier: {(r.screen, int(r.pair_index), r.condition): Fraction(int(r.numerator), int(r.denominator)) for r in part.itertuples(index=False)} for identifier, part in coefficients.groupby('unit_id', sort=False)}
    record_lookup = defaultdict(set)
    for row in provenance.itertuples(index=False):
        record_lookup[row.screen, int(row.pair_index), row.condition].add(row.record_id)
    support_lookup = support.set_index(PAIR + ['unit_identity'])
    independently_built = {}
    support_ok = True
    for row in old.to_dict('records'):
        key = tuple((row[k] for k in PAIR + ['unit_identity']))
        saved = support_lookup.loc[key]
        entry = {'x': float(row['source_value']), 'y': float(row['target_value'])}
        support_ok &= abs(entry['x'] - saved.source_value) < 1e-12 and abs(entry['y'] - saved.target_value) < 1e-12
        for role in ('source', 'target'):
            endpoint_ids = json.loads(row[role + '_endpoint_ids_json'])
            operator = defaultdict(Fraction)
            for uid in endpoint_ids:
                for coordinate, value in ep[uid].items():
                    operator[coordinate] += value / Fraction(len(endpoint_ids))
            operator = {k: v for k, v in operator.items() if v}
            canonical = [[*coordinate, value.numerator, value.denominator] for coordinate, value in sorted(operator.items())]
            signature = hashlib.sha256(compact(canonical).encode('utf-8')).hexdigest()
            coords = set(operator)
            records = set((record for coordinate in coords for record in record_lookup[coordinate]))
            support_ok &= signature == saved[role + '_operator_signature'] and coords == set(map(tuple, json.loads(saved[role + '_coordinates_json'])))
            support_ok &= records == set(json.loads(saved[role + '_record_ids_json'])) and len(coords) == saved[role + '_coordinate_cost']
            entry[role] = coords
            entry[role + '_records'] = records
        independently_built[key] = entry
    check('independent_exact_collapsed_response_support', support_ok and len(support) == len(old) == 552)
    expected_pairs = set(map(tuple, atlas[PAIR].to_numpy()))
    check('complete_registered_domain', len(expected_pairs) == 90 and set(map(tuple, designs[PAIR].to_numpy())) == expected_pairs and (len(designs) == 900) and (len(budgets) == 3600) and (len(scores) == 18000))
    check('unique_machine_table_keys', not designs.duplicated(PAIR + ['design']).any() and (not budgets.duplicated(PAIR + ['design', 'budget']).any()) and (not scores.duplicated(PAIR + ['design', 'budget', 'model']).any()) and (not predictions.duplicated(PAIR + ['design', 'budget', 'model', 'unit_identity']).any()))
    design_ok = True
    rebuilt_designs = {}
    for row in designs.to_dict('records'):
        pair = tuple((row[k] for k in PAIR))
        available = {key[-1]: value for key, value in independently_built.items() if key[:-1] == pair}
        domain = atlas[np.logical_and.reduce([atlas[key].eq(value) for key, value in zip(PAIR, pair)])].iloc[0]
        n = len(available)

        def hashed(identity):
            return hashlib.sha256(compact(['budgeted-target-calibration-v1', pair[0], pair[1], int(row['design']), identity]).encode('utf-8')).hexdigest()
        if not str(domain.domain_state).startswith('scored_'):
            test, pool, status = ([], [], str(domain.domain_state))
        elif n < 6:
            test, pool, status = ([], [], 'insufficient_identity_support')
        else:
            ordered = sorted(available, key=hashed)
            count = max(3, int(np.ceil(n / 3)))
            test = ordered[:count]
            pool = sorted(ordered[count:], key=lambda identity: (len(available[identity]['target']), hashed(identity)))
            status = 'eligible_design'
        design_ok &= row['primary_domain_state'] == domain.domain_state and row['n_primary_identities'] == n and (row['design_status'] == status)
        design_ok &= json.loads(row['test_identities_json']) == test and json.loads(row['calibration_pool_order_json']) == pool and (row['n_test'] == len(test)) and (row['n_calibration_pool'] == len(pool))
        rebuilt_designs[pair + (row['design'],)] = (available, test, pool, status)
    check('independent_outcome_free_hash_splits_and_cost_order', design_ok)
    budget_ok, disjoint_ok, fit_ok, prediction_ok, guard_ok = (True, True, True, True, True)
    scores_lookup = scores.set_index(PAIR + ['design', 'budget', 'model'])
    predictions_lookup = {key: part.set_index('unit_identity') for key, part in predictions.groupby(PAIR + ['design', 'budget', 'model'], sort=False)}
    reconstructed_mse = {}
    nested = defaultdict(list)
    max_prediction_error = 0.0
    for row in budgets.to_dict('records'):
        pair = tuple((row[k] for k in PAIR))
        key = pair + (row['design'],)
        available, test, pool, status = rebuilt_designs[key]
        selected = []
        coordinate_union = set()
        for identity in pool:
            new = coordinate_union | available[identity]['target']
            if len(new) > row['budget']:
                break
            coordinate_union = new
            selected.append(identity)

        def union(identities, role):
            return set().union(*(available[i][role] for i in identities)) if identities else set()
        test_coords = union(test, 'target')
        overlap = len(coordinate_union & test_coords)
        source_overlap = len(union(selected, 'source') & union(test, 'source'))
        cross_overlap = len(coordinate_union & union(test, 'source') | union(selected, 'source') & test_coords)
        record_overlap = len(union(selected, 'target_records') & union(test, 'target_records'))
        source_record_overlap = len(union(selected, 'source_records') & union(test, 'source_records'))
        cross_record_overlap = len(union(selected, 'target_records') & union(test, 'source_records') | union(selected, 'source_records') & union(test, 'target_records'))
        disjoint = overlap == source_overlap == cross_overlap == record_overlap == source_record_overlap == cross_record_overlap == 0
        expected_status = status
        if status == 'eligible_design':
            expected_status = 'operator_support_overlap' if not disjoint else 'scored' if len(selected) >= 3 else 'budget_insufficient'
        budget_ok &= row['status'] == expected_status and row['n_calibration'] == len(selected) and (row['actual_spent'] == len(coordinate_union)) and (row['unused_budget'] == row['budget'] - len(coordinate_union))
        budget_ok &= set(map(tuple, json.loads(row['calibration_coordinates_json']))) == coordinate_union and set(map(tuple, json.loads(row['test_coordinates_json']))) == test_coords
        budget_ok &= json.loads(row['calibration_identities_json']) == selected and json.loads(row['test_identities_json']) == test
        disjoint_ok &= row['target_coordinate_overlap'] == overlap and row['source_coordinate_overlap'] == source_overlap and (row['cross_role_coordinate_overlap'] == cross_overlap) and (row['target_record_overlap'] == record_overlap) and (flag(row['operator_disjoint']) == disjoint)
        disjoint_ok &= row['source_record_overlap'] == source_record_overlap and row['cross_role_record_overlap'] == cross_record_overlap
        nested[key].append((row['budget'], selected, test, len(coordinate_union)))
        if expected_status == 'scored':
            xall = np.array([value['x'] for value in available.values()])
            xc = np.array([available[i]['x'] for i in selected])
            yc = np.array([available[i]['y'] for i in selected])
            xt = np.array([available[i]['x'] for i in test])
            yt = np.array([available[i]['y'] for i in test])
            mean, sd = (float(xall.mean()), float(xall.std()))
            if sd <= 1e-12:
                a, b = (float(yc.mean()), 0.0)
            else:
                z = (xc - mean) / sd
                slope = float(np.sum((z - z.mean()) * (yc - yc.mean())) / (np.sum((z - z.mean()) ** 2) + 3.0))
                b = slope / sd
                a = float(yc.mean() - slope * z.mean() - b * mean)
            parameters = {'zero': (0.0, 0.0), 'direct_source': (0.0, 1.0), 'target_calibration_mean': (float(yc.mean()), 0.0), 'source_offset': (float(np.mean(yc - xc)), 1.0), 'source_affine_ridge': (a, b)}
        for model in MODELS:
            score_key = key + (row['budget'], model)
            score = scores_lookup.loc[score_key]
            guard_ok &= score.status == expected_status
            if expected_status != 'scored':
                guard_ok &= score.mse == '' and score_key not in predictions_lookup
                continue
            a, b = parameters[model]
            predicted = a + b * xt
            mse = float(np.mean((predicted - yt) ** 2))
            reconstructed_mse[score_key] = mse
            fit_ok &= abs(float(score.intercept) - a) < 1e-10 and abs(float(score.slope) - b) < 1e-10 and (abs(float(score.mse) - mse) < 1e-12) and (abs(float(score.rmse_pp) - 100 * np.sqrt(mse)) < 1e-09)
            fit_ok &= abs(float(score.source_only_sd) - sd) < 1e-12 and abs(float(score.source_only_mean) - mean) < 1e-12 and (float(score.ridge_alpha) == 3.0)
            saved = predictions_lookup[score_key]
            prediction_ok &= set(saved.index) == set(test) and len(saved) == len(test)
            difference = np.max(np.abs(saved.loc[test, 'predicted'].to_numpy(float) - predicted))
            max_prediction_error = max(max_prediction_error, float(difference))
            prediction_ok &= difference < 1e-10 and np.allclose(saved.loc[test, 'observed'].to_numpy(float), yt, atol=1e-12, rtol=0)
    check('independent_union_cost_prefix_and_budget_guards', budget_ok)
    check('all_source_target_and_record_disjointness', disjoint_ok and budgets.operator_disjoint.all())
    check('independent_closed_form_five_model_fits', fit_ok)
    check('all_heldout_predictions_from_calibration_only', prediction_ok and set(predictions_lookup) == set(reconstructed_mse), max_absolute_prediction_error=max_prediction_error)
    check('unknown_and_guarded_rows_never_scored', guard_ok)
    nested_ok = True
    for entries in nested.values():
        entries = sorted(entries)
        for earlier, later in zip(entries, entries[1:]):
            nested_ok &= earlier[2] == later[2] and later[1][:len(earlier[1])] == earlier[1] and (earlier[3] <= later[3])
    check('tests_fixed_and_prefixes_nested_across_budgets', nested_ok)
    coverage = read('coverage.csv')
    coverage_ok = True
    for row in coverage.to_dict('records'):
        group = budgets[budgets.relation.eq(row['relation']) & budgets.budget.eq(row['budget'])]
        by_direction = group.groupby(['source_screen', 'target_screen']).status.agg(lambda values: int(values.eq('scored').sum()))
        coverage_ok &= row['n_registered_directions'] == len(by_direction) and row['n_directions_any_scored'] == int(by_direction.gt(0).sum()) and (row['n_directions_all_ten_scored'] == int(by_direction.eq(10).sum()))
        for status in budgets.status.unique():
            coverage_ok &= row['n_' + status] == int(group.status.eq(status).sum())
    check('independent_full_domain_coverage_and_guard_counts', coverage_ok)
    directions = read('direction_summary.csv')
    summary_ok = True
    for row in directions.to_dict('records'):
        prefix = tuple((row[k] for k in PAIR))
        values = [value for key, value in reconstructed_mse.items() if key[:5] == prefix and key[-2:] == (row['budget'], row['model'])]
        summary_ok &= row['n_scored_designs'] == len(values)
        if values:
            summary_ok &= abs(float(row['design_macro_mse']) - np.mean(values)) < 1e-12
        else:
            summary_ok &= row['design_macro_mse'] == ''
    check('independent_direction_design_macro_scores', summary_ok and len(directions) == 1800)
    macro = read('model_summary.csv')
    macro_ok = True
    for row in macro.to_dict('records'):
        selected = directions[directions.relation.eq(row['relation']) & directions.budget.eq(row['budget']) & directions.model.eq(row['model']) & directions.status.eq('scored')]
        macro_ok &= row['n_scored_directions'] == len(selected) and abs(float(row['direction_macro_mse']) - pd.to_numeric(selected.design_macro_mse).mean()) < 1e-12
    check('independent_equal_direction_macro_scores', macro_ok)
    paired = read('paired_comparisons.csv.gz')
    paired_ok = True
    for row in paired.to_dict('records'):
        key = tuple((row[k] for k in PAIR + ['design', 'budget']))
        for baseline in ('zero', 'direct_source', 'target_calibration_mean'):
            expected = reconstructed_mse[key + (baseline,)] - reconstructed_mse[key + (row['model'],)]
            paired_ok &= abs(float(row['mse_gain_vs_' + baseline]) - expected) < 1e-12
    check('independent_same_test_same_budget_model_comparisons', paired_ok)
    transitions = read('budget_comparisons.csv.gz')
    transition_ok = True
    for row in transitions.to_dict('records'):
        key = tuple((row[k] for k in PAIR + ['design']))
        left = key + (row['earlier_budget'], row['model'])
        right = key + (row['later_budget'], row['model'])
        known = left in reconstructed_mse and right in reconstructed_mse
        transition_ok &= flag(row['both_scored_same_test']) == known
        if known:
            transition_ok &= abs(float(row['mse_gain_later']) - (reconstructed_mse[left] - reconstructed_mse[right])) < 1e-12
    check('independent_fixed_test_cross_budget_comparisons', transition_ok)
    zero_source = predictions[predictions.source_screen.eq('SNAr|1') & predictions.target_screen.eq('Heck|1')]
    zero_target = predictions[predictions.source_screen.eq('Heck|1') & predictions.target_screen.eq('SNAr|1')]
    check('natural_zero_source_and_zero_target_cases_retained', len(zero_source) > 0 and zero_source.source_value.eq(0).all() and (len(zero_target) > 0) and zero_target.observed.eq(0).all())
    with Image.open(HERE / 'figures/45_BUDGETED_TARGET_CALIBRATION.png') as image:
        pixels = np.asarray(image.convert('RGB'))
    check('nonblank_figure', pixels.shape[0] >= 700 and pixels.shape[1] >= 2400 and (pixels.std() > 10))
    check('unique_check_names', len({c['check'] for c in checks}) == len(checks))
    result = dict(passed=all((c['passed'] for c in checks)), checks_total=len(checks), checks_passed=sum((c['passed'] for c in checks)), checks=checks, counts=manifest['counts'])
    (DATA / (PREFIX + 'verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
