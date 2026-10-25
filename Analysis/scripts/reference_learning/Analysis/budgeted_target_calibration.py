"""Held-out calibration under frozen yield-coordinate acquisition budgets.

Targets remain exact response operators from report 52. Design orders are
outcome-independent sensitivity analyses, never independent replications."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'budgeted_target_calibration_'
REPORT = 'budgeted_target_calibration.py'
PAIR = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
BUDGETS = (24, 48, 96, 192)
MODELS = ('zero', 'direct_source', 'target_calibration_mean', 'source_offset', 'source_affine_ridge')
N_DESIGNS = 10
ALPHA = 3.0
EPS = 1e-12

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(mapping):
    return [[*key, value.numerator, value.denominator] for key, value in sorted(mapping.items()) if value]

def op_signature(mapping):
    return hashlib.sha256(dumps(canonical(mapping)).encode('utf-8')).hexdigest()

def average_operators(operators):
    if not operators:
        raise ValueError('empty response operator population')
    result = defaultdict(Fraction)
    for mapping in operators:
        for key, value in mapping.items():
            result[key] += value / len(operators)
    return {key: value for key, value in result.items() if value}

def order_hash(source, target, design, identity):
    return hashlib.sha256(dumps(['budgeted-target-calibration-v1', source, target, int(design), identity]).encode('utf-8')).hexdigest()

def fixed_split(frame, source, target, design):
    ordered = sorted(frame.unit_identity, key=lambda identity: order_hash(source, target, design, identity))
    n_test = max(3, (len(ordered) + 2) // 3)
    test = ordered[:n_test]
    pool = ordered[n_test:]
    costs = frame.set_index('unit_identity').target_coordinate_cost.to_dict()
    pool = sorted(pool, key=lambda identity: (costs[identity], order_hash(source, target, design, identity)))
    return (test, pool)

def budget_prefix(pool, supports, budget):
    selected, coordinates = ([], set())
    for identity in pool:
        candidate = coordinates | supports[identity]
        if len(candidate) > budget:
            break
        selected.append(identity)
        coordinates = candidate
    return (selected, coordinates)

def predictions_for_models(source_all, source_cal, target_cal, source_test):
    source_all = np.asarray(source_all, float)
    source_cal, target_cal, source_test = [np.asarray(value, float) for value in (source_cal, target_cal, source_test)]
    if len(source_cal) < 3 or len(target_cal) != len(source_cal):
        raise ValueError('at least three calibration identities required')
    source_mean, source_sd = (float(source_all.mean()), float(source_all.std()))
    target_mean = float(target_cal.mean())
    source_constant = bool(source_sd <= EPS)
    train_constant = bool(np.ptp(source_cal) <= EPS)
    if source_constant:
        affine = np.full(len(source_test), target_mean)
        a, b = (target_mean, 0.0)
    else:
        estimator = Ridge(alpha=ALPHA, fit_intercept=True, solver='svd')
        estimator.fit(((source_cal - source_mean) / source_sd)[:, None], target_cal)
        affine = estimator.predict(((source_test - source_mean) / source_sd)[:, None])
        b = float(estimator.coef_[0] / source_sd)
        a = float(estimator.intercept_ - b * source_mean)
    offset = float(np.mean(target_cal - source_cal))
    result = {'zero': np.zeros(len(source_test)), 'direct_source': source_test.copy(), 'target_calibration_mean': np.full(len(source_test), target_mean), 'source_offset': source_test + offset, 'source_affine_ridge': affine}
    parameters = {'zero': (0.0, 0.0), 'direct_source': (0.0, 1.0), 'target_calibration_mean': (target_mean, 0.0), 'source_offset': (offset, 1.0), 'source_affine_ridge': (a, b)}
    return (result, parameters, dict(source_only_mean=source_mean, source_only_sd=source_sd, source_variance_zero=source_constant, calibration_source_constant=train_constant, ridge_alpha=ALPHA, ridge_fallback_to_calibration_mean=source_constant))

def load_inputs():
    names = ['tables/response_estimand_lineage_collapsed_units.csv.gz', 'tables/response_estimand_lineage_yield_coefficients.csv.gz', 'tables/response_estimand_lineage_record_provenance.csv', 'tables/transport_domain_atlas_cells.csv', 'data/response_estimand_lineage_manifest.json', 'data/response_estimand_lineage_verification.json', 'response_estimand_lineage.py', 'controlled_cross_library_bridges.py', 'budgeted_target_calibration.py']
    hashes = {name: digest(HERE / name) for name in names}
    if not json.loads((DATA / 'response_estimand_lineage_verification.json').read_text(encoding='utf-8'))['passed']:
        raise ValueError('report 52 lineage verification did not pass')
    units = pd.read_csv(TABLES / 'response_estimand_lineage_collapsed_units.csv.gz', keep_default_na=False)
    coefficients = pd.read_csv(TABLES / 'response_estimand_lineage_yield_coefficients.csv.gz', keep_default_na=False)
    provenance = pd.read_csv(TABLES / 'response_estimand_lineage_record_provenance.csv', keep_default_na=False)
    cells = pd.read_csv(TABLES / 'transport_domain_atlas_cells.csv', keep_default_na=False)
    mask = lambda frame: frame['mode'].eq('strict') & frame.background_control.eq('background_free') & frame.basis.eq('position_2') & frame.min_partners_each.eq(1) & frame.partner_overlap.eq('all')
    return (units[mask(units)].copy(), coefficients, provenance, cells[mask(cells)].copy(), hashes)

def build_unit_support(units, coefficients, provenance):
    endpoints = {identifier: {(r.screen, int(r.pair_index), r.condition): Fraction(int(r.numerator), int(r.denominator)) for r in group.itertuples(index=False)} for identifier, group in coefficients.groupby('unit_id', sort=False)}
    records = defaultdict(set)
    for row in provenance.itertuples(index=False):
        records[row.screen, int(row.pair_index), row.condition].add(row.record_id)
    rows = []
    for row in units.to_dict('records'):
        item = {key: row[key] for key in PAIR + ['unit_identity', 'source_value', 'target_value', 'n_registry_rows']}
        for role in ('source', 'target'):
            ids = json.loads(row[role + '_endpoint_ids_json'])
            operator = average_operators([endpoints[identifier] for identifier in ids])
            if op_signature(operator) != row[role + '_operator_signature']:
                raise ValueError('report 52 collapsed operator disagrees')
            coordinates = set(operator)
            record_ids = set((record for coordinate in coordinates for record in records[coordinate]))
            item[role + '_operator_signature'] = op_signature(operator)
            item[role + '_coordinates_json'] = dumps([list(key) for key in sorted(coordinates)])
            item[role + '_record_ids_json'] = dumps(sorted(record_ids))
            item[role + '_coordinate_cost'] = len(coordinates)
            item[role + '_record_count'] = len(record_ids)
        rows.append(item)
    return pd.DataFrame(rows).sort_values(PAIR + ['unit_identity']).reset_index(drop=True)

def build_designs(support, cells):
    rows = []
    for pair in cells.to_dict('records'):
        mask = pd.Series(True, index=support.index)
        for key in PAIR:
            mask &= support[key].eq(pair[key])
        units = support[mask]
        for design in range(N_DESIGNS):
            row = {key: pair[key] for key in PAIR}
            row.update(design=design, primary_domain_state=pair['domain_state'], n_primary_identities=len(units))
            if not str(pair['domain_state']).startswith('scored_'):
                status = str(pair['domain_state'])
                test, pool = ([], [])
            elif len(units) < 6:
                status = 'insufficient_identity_support'
                test, pool = ([], [])
            else:
                test, pool = fixed_split(units, pair['source_screen'], pair['target_screen'], design)
                status = 'eligible_design' if len(test) >= 3 and len(pool) >= 3 else 'insufficient_identity_support'
            x, y = (units.source_value.to_numpy(float), units.target_value.to_numpy(float))
            row.update(design_status=status, n_test=len(test), n_calibration_pool=len(pool), test_identities_json=dumps(test), calibration_pool_order_json=dumps(pool), all_source_zero=bool(len(x) > 0 and np.max(np.abs(x)) <= EPS), all_target_zero=bool(len(y) > 0 and np.max(np.abs(y)) <= EPS), all_source_constant=bool(len(x) > 0 and np.ptp(x) <= EPS), all_target_constant=bool(len(y) > 0 and np.ptp(y) <= EPS))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(PAIR + ['design']).reset_index(drop=True)

def support_union(identities, lookup, role, suffix='coordinates'):
    return set((tuple(coordinate) if suffix == 'coordinates' else coordinate for identity in identities for coordinate in json.loads(lookup.loc[identity, role + '_' + suffix + '_json'])))

def build_budget_results(support, designs):
    budgets, scores, predictions = ([], [], [])
    for design in designs.to_dict('records'):
        mask = pd.Series(True, index=support.index)
        for key in PAIR:
            mask &= support[key].eq(design[key])
        units = support[mask].set_index('unit_identity')
        test, pool = (json.loads(design['test_identities_json']), json.loads(design['calibration_pool_order_json']))
        target_support = {identity: set(map(tuple, json.loads(row.target_coordinates_json))) for identity, row in units.iterrows()}
        for budget in BUDGETS:
            selected, coordinates = budget_prefix(pool, target_support, budget)
            target_test = support_union(test, units, 'target') if test else set()
            source_cal = support_union(selected, units, 'source') if selected else set()
            source_test = support_union(test, units, 'source') if test else set()
            target_records = support_union(selected, units, 'target', 'record_ids') if selected else set()
            test_records = support_union(test, units, 'target', 'record_ids') if test else set()
            source_cal_records = support_union(selected, units, 'source', 'record_ids') if selected else set()
            source_test_records = support_union(test, units, 'source', 'record_ids') if test else set()
            overlap = len(coordinates & target_test)
            source_overlap = len(source_cal & source_test)
            cross_role_overlap = len(coordinates & source_test | source_cal & target_test)
            record_overlap = len(target_records & test_records)
            source_record_overlap = len(source_cal_records & source_test_records)
            cross_role_record_overlap = len(target_records & source_test_records | source_cal_records & test_records)
            disjoint = overlap == source_overlap == cross_role_overlap == record_overlap == source_record_overlap == cross_role_record_overlap == 0
            status = design['design_status']
            if status == 'eligible_design':
                status = 'operator_support_overlap' if not disjoint else 'scored' if len(selected) >= 3 else 'budget_insufficient'
            row = {key: design[key] for key in PAIR + ['design']}
            row.update(budget=budget, status=status, n_primary_identities=design['n_primary_identities'], n_test=len(test), n_calibration_pool=len(pool), n_calibration=len(selected), actual_spent=len(coordinates), unused_budget=budget - len(coordinates), test_coordinate_count=len(target_test), calibration_identities_json=dumps(selected), test_identities_json=dumps(test), calibration_coordinates_json=dumps([list(key) for key in sorted(coordinates)]), test_coordinates_json=dumps([list(key) for key in sorted(target_test)]), target_coordinate_overlap=overlap, source_coordinate_overlap=source_overlap, cross_role_coordinate_overlap=cross_role_overlap, target_record_overlap=record_overlap, source_record_overlap=source_record_overlap, cross_role_record_overlap=cross_role_record_overlap, operator_disjoint=disjoint, all_source_zero=design['all_source_zero'], all_target_zero=design['all_target_zero'])
            budgets.append(row)
            if status == 'scored':
                xs = units.loc[test, 'source_value'].to_numpy(float)
                ys = units.loc[test, 'target_value'].to_numpy(float)
                pred, parameters, diagnostics = predictions_for_models(units.source_value, units.loc[selected, 'source_value'], units.loc[selected, 'target_value'], xs)
            else:
                pred, parameters, diagnostics = ({}, {}, {})
            for model in MODELS:
                score = {key: row[key] for key in PAIR + ['design', 'budget', 'status', 'n_test', 'n_calibration', 'actual_spent', 'unused_budget', 'all_source_zero', 'all_target_zero']}
                score.update(model=model, **diagnostics)
                if status == 'scored':
                    error = pred[model] - ys
                    a, b = parameters[model]
                    score.update(mse=float(np.mean(error ** 2)), rmse_pp=float(100 * np.sqrt(np.mean(error ** 2))), mae_pp=float(100 * np.mean(np.abs(error))), signed_bias_pp=float(100 * error.mean()), target_test_rms_pp=float(100 * np.sqrt(np.mean(ys ** 2))), target_test_sd_pp=float(100 * ys.std()), intercept=a, slope=b, predictions_outside_response_range=int((np.abs(pred[model]) > 1).sum()))
                    for identity, x, y, estimate in zip(test, xs, ys, pred[model]):
                        predictions.append({**{key: score[key] for key in PAIR + ['design', 'budget', 'model']}, 'unit_identity': identity, 'source_value': x, 'observed': y, 'predicted': float(estimate), 'target_operator_signature': units.loc[identity, 'target_operator_signature']})
                else:
                    score.update(mse=np.nan, rmse_pp=np.nan, mae_pp=np.nan, signed_bias_pp=np.nan, target_test_rms_pp=np.nan, target_test_sd_pp=np.nan, intercept=np.nan, slope=np.nan, predictions_outside_response_range=0)
                scores.append(score)
    return (pd.DataFrame(budgets), pd.DataFrame(scores), pd.DataFrame(predictions))

def summarize(budgets, scores):
    direction_rows = []
    for keys, part in scores.groupby(PAIR + ['budget', 'model'], sort=True):
        item = dict(zip(PAIR + ['budget', 'model'], keys))
        finite = part[part.status.eq('scored')]
        item.update(n_designs=N_DESIGNS, n_scored_designs=len(finite), status='scored' if len(finite) else str(part.status.iloc[0]), design_macro_mse=float(finite.mse.mean()) if len(finite) else np.nan, design_macro_rmse_pp=float(100 * np.sqrt(finite.mse.mean())) if len(finite) else np.nan, mean_spent=float(finite.actual_spent.mean()) if len(finite) else np.nan, mean_calibration_identities=float(finite.n_calibration.mean()) if len(finite) else np.nan)
        direction_rows.append(item)
    directions = pd.DataFrame(direction_rows)
    coverage = []
    for (relation, budget), part in budgets.groupby(['relation', 'budget'], sort=True):
        count = part.groupby(['source_screen', 'target_screen']).status.agg(lambda x: int(x.eq('scored').sum()))
        item = dict(relation=relation, budget=budget, n_registered_directions=len(count), n_design_budget_rows=len(part), n_directions_any_scored=int(count.gt(0).sum()), n_directions_all_ten_scored=int(count.eq(N_DESIGNS).sum()))
        item.update({'n_' + status: int(part.status.eq(status).sum()) for status in sorted(budgets.status.unique())})
        coverage.append(item)
    macro = []
    for (relation, budget, model), part in directions.groupby(['relation', 'budget', 'model'], sort=True):
        finite = part[part.status.eq('scored')]
        macro.append(dict(relation=relation, budget=budget, model=model, n_directions=len(part), n_scored_directions=len(finite), direction_macro_mse=float(finite.design_macro_mse.mean()), direction_macro_rmse_pp=float(100 * np.sqrt(finite.design_macro_mse.mean()))))
    paired = []
    keys = PAIR + ['design', 'budget']
    lookup = scores.set_index(keys + ['model'])
    for row in scores[scores.status.eq('scored')].to_dict('records'):
        key = tuple((row[k] for k in keys))
        item = {k: row[k] for k in keys + ['model', 'actual_spent', 'n_calibration', 'n_test']}
        item['mse'] = row['mse']
        for baseline in ('zero', 'direct_source', 'target_calibration_mean'):
            reference = lookup.loc[key + (baseline,)]
            item['mse_gain_vs_' + baseline] = float(reference.mse - row['mse'])
            item['rmse_gain_vs_' + baseline + '_pp'] = float(reference.rmse_pp - row['rmse_pp'])
        paired.append(item)
    transitions = []
    columns = PAIR + ['design', 'model']
    for key, part in scores.groupby(columns, sort=True):
        by_budget = part.set_index('budget')
        for earlier, later in ((24, 48), (24, 96), (24, 192)):
            a, b = (by_budget.loc[earlier], by_budget.loc[later])
            both = a.status == b.status == 'scored'
            transitions.append({**dict(zip(columns, key)), 'earlier_budget': earlier, 'later_budget': later, 'earlier_status': a.status, 'later_status': b.status, 'both_scored_same_test': both, 'earlier_spent': a.actual_spent, 'later_spent': b.actual_spent, 'mse_gain_later': float(a.mse - b.mse) if both else np.nan})
    return {PREFIX + 'direction_summary.csv': directions, PREFIX + 'coverage.csv': pd.DataFrame(coverage), PREFIX + 'model_summary.csv': pd.DataFrame(macro), PREFIX + 'paired_comparisons.csv.gz': pd.DataFrame(paired), PREFIX + 'budget_comparisons.csv.gz': pd.DataFrame(transitions)}

def build_outputs():
    units, coefficients, provenance, cells, hashes = load_inputs()
    support = build_unit_support(units, coefficients, provenance)
    designs = build_designs(support, cells)
    budgets, scores, predictions = build_budget_results(support, designs)
    outputs = {PREFIX + 'unit_support.csv.gz': support, PREFIX + 'designs.csv': designs, PREFIX + 'budget_designs.csv.gz': budgets, PREFIX + 'scores.csv': scores, PREFIX + 'predictions.csv.gz': predictions}
    outputs.update(summarize(budgets, scores))
    return (outputs, hashes)

def make_figure(outputs):
    coverage = outputs[PREFIX + 'coverage.csv']
    models = outputs[PREFIX + 'model_summary.csv']
    paired = outputs[PREFIX + 'paired_comparisons.csv.gz']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for relation, part in coverage.groupby('relation'):
        axes[0].plot(part.budget, part.n_directions_any_scored, 'o-', label=relation)
    axes[0].set(xlabel='Frozen target yield-coordinate budget', ylabel='Directions with any scored design', title='Full registered domain retains guards', xticks=BUDGETS)
    axes[0].legend(fontsize=8)
    colors = ['#777777', '#bf5c62', '#6f8790', '#e4ba69', '#438e73']
    for model, color in zip(MODELS, colors):
        part = models[models.relation.eq('cross_family') & models.model.eq(model)]
        axes[1].plot(part.budget, part.direction_macro_rmse_pp, 'o-', color=color, label=model)
    axes[1].set(xlabel='Budget (scored direction domain changes)', ylabel='Equal-direction RMSE (yield pp)', title='Common-budget model comparison', xticks=BUDGETS)
    axes[1].legend(fontsize=7)
    part = paired[paired.relation.eq('cross_family') & paired.model.eq('source_affine_ridge')]
    for budget, group in part.groupby('budget'):
        directions = group.groupby(['source_screen', 'target_screen']).mse_gain_vs_target_calibration_mean.mean()
        axes[2].scatter(np.full(len(directions), budget), directions, s=22, alpha=0.6)
    axes[2].axhline(0, color='gray', lw=0.8)
    axes[2].set(xlabel='Target coordinate budget', ylabel='Target-mean MSE minus affine MSE', title='Added source variation on held-out identities', xticks=BUDGETS)
    path = FIGURES / '45_BUDGETED_TARGET_CALIBRATION.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def counts(outputs):
    get = lambda suffix: outputs[PREFIX + suffix]
    designs, budgets, scores = (get('designs.csv'), get('budget_designs.csv.gz'), get('scores.csv'))
    return dict(registered_directions=int(len(designs) / N_DESIGNS), primary_finite_directions=int(designs.loc[designs.primary_domain_state.str.startswith('scored_'), PAIR].drop_duplicates().shape[0]), primary_collapsed_units=len(get('unit_support.csv.gz')), eligible_directions=int(designs.loc[designs.design_status.eq('eligible_design'), PAIR].drop_duplicates().shape[0]), design_rows=len(designs), budget_design_rows=len(budgets), score_rows=len(scores), prediction_rows=len(get('predictions.csv.gz')), scored_design_budgets=int(budgets.status.eq('scored').sum()), coordinate_overlap_rows=int((~budgets.operator_disjoint).sum()), budgets=list(BUDGETS), designs_per_direction=N_DESIGNS, ridge_alpha=ALPHA)

def main():
    outputs, hashes = build_outputs()
    files = []
    for name, frame in outputs.items():
        path = TABLES / name
        frame.to_csv(path, index=False)
        files.append(path)
    files.extend([make_figure(outputs), HERE / REPORT])
    if hashes != {name: digest(HERE / name) for name in hashes}:
        raise ValueError('input changed during computation')
    manifest = dict(date='2026-10-17', scope='Retrospective fixed-response identity holdout with target coordinate-budget calibration', input_sha256=hashes, code_sha256={name: digest(HERE / name) for name in ['budgeted_target_calibration.py', 'test_budgeted_target_calibration.py', 'verify_budgeted_target_calibration.py']}, output_sha256={str(path.relative_to(HERE)): digest(path) for path in files}, counts=counts(outputs), design=dict(primary='strict/background_free/position_2/min1/all; all 78 registered directions', test='max(3,ceil(n/3)) smallest outcome-independent hashes, fixed across budgets', calibration='remaining pool sorted by exact target operator coordinate cost then same hash; maximal nested prefix within budget; at least3 identities', features='all historical source x used only for source-only mean/SD; no target test data in fitting/scaling', ridge='sklearn Ridge alpha3, standardized source x, unpenalized intercept, slope shrunk to0; global source SD<=1e-12 falls back to target calibration mean', cost='union of nonzero exact target yield-operator coordinates; frozen matrix yield cells, not independent experiments or direct collapsed response measurements', inference='10 fixed hash orders are design sensitivities, not independent replicates; directions share screens and target responses; no target clipping, causal interpretation, prospective claim, or hyperparameter search'))
    (DATA / (PREFIX + 'manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts'], indent=2))
    print(outputs[PREFIX + 'coverage.csv'].to_string(index=False))
    print(outputs[PREFIX + 'model_summary.csv'].to_string(index=False))
if __name__ == '__main__':
    main()
