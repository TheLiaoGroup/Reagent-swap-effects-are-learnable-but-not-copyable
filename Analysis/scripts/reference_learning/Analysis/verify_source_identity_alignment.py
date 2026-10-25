"""Independent closed-form audit of source identity and target adaptation."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent
TABLES, DATA = (HERE / 'tables', HERE / 'data')
PREFIX = 'source_identity_alignment'
PAIR = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
BUDGET_KEYS = PAIR + ['design', 'budget']
MODELS = ('source_affine_ridge', 'source_offset')
N_DRAWS = 128
EPS = 1e-12

def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def closed_form(source_all, source_calibration, target_calibration, affine):
    """Solve the frozen one-predictor model without the producer or sklearn."""
    population = np.asarray(source_all, dtype=float)
    x = np.asarray(source_calibration, dtype=float)
    y = np.asarray(target_calibration, dtype=float)
    if len(x) < 3 or len(y) != len(x):
        raise ValueError('at least three matched calibration identities are required')
    if not affine:
        return (float(np.mean(y - x)), 1.0)
    scale = float(population.std())
    if scale <= EPS:
        return (float(y.mean()), 0.0)
    xc, yc = (x - x.mean(), y - y.mean())
    slope = float(xc @ yc / (xc @ xc + 3.0 * scale * scale))
    return (float(y.mean() - slope * x.mean()), slope)

def frozen_test_reference(source_test, target_test, intercept, slope):
    x, y = (np.asarray(source_test, dtype=float), np.asarray(target_test, dtype=float))
    xmean, ymean = (float(x.mean()), float(y.mean()))
    covariance = float(np.mean((x - xmean) * (y - ymean)))
    true_mse = float(np.mean((y - intercept - slope * x) ** 2))
    permuted_mse = float(np.mean((y - intercept - slope * xmean) ** 2) + slope * slope * np.mean((x - xmean) ** 2))
    return {'true_mse': true_mse, 'permuted_mse': permuted_mse, 'alignment_advantage': 2.0 * slope * covariance, 'test_covariance': covariance}

def compare_frames(actual, expected, keys, columns=None, tolerance=1e-10):
    """Require matching keys before comparing any numeric or metadata fields."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        return (False, {'duplicate_keys': True})
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        return (False, {'key_mismatch': True, 'saved_rows': len(actual), 'rebuilt_rows': len(expected)})
    columns = list(expected.columns) if columns is None else list(columns)
    missing = sorted(set(columns) - set(actual.columns))
    if missing:
        return (False, {'missing_columns': missing})
    maximum = 0.0
    failed = []
    for column in columns:
        left, right = (actual[column], expected[column])
        if pd.api.types.is_numeric_dtype(right):
            x, y = (pd.to_numeric(left, errors='coerce').to_numpy(float), right.to_numpy(float))
            ok = np.isclose(x, y, rtol=1e-09, atol=tolerance, equal_nan=True)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(x[finite] - y[finite]))))
            if not ok.all():
                failed.append(column)
        elif not left.fillna('').astype(str).equals(right.fillna('').astype(str)):
            failed.append(column)
    return (not failed, {'max_absolute_numeric_difference': maximum, 'failed_columns': failed})

def full_role_disjointness(support):
    coordinates_ok, records_ok, n = (True, True, 0)
    for _, part in support.groupby(PAIR, sort=True):
        n += 1
        union = {}
        for role in ('source', 'target'):
            union[role] = set((tuple(value) for text in part[role + '_coordinates_json'] for value in json.loads(text)))
            union[role + '_records'] = set((value for text in part[role + '_record_ids_json'] for value in json.loads(text)))
        coordinates_ok &= not bool(union['source'] & union['target'])
        records_ok &= not bool(union['source_records'] & union['target_records'])
    return (coordinates_ok, records_ok, n)

def baseline_reconstruction(support, budgets, old_scores):
    populations = {key: part.set_index('unit_identity').sort_index() for key, part in support.groupby(PAIR, sort=True)}
    score_lookup = old_scores.set_index(BUDGET_KEYS + ['model'])
    contexts = {}
    fit_ok, score_ok = (True, True)
    max_error, exhaustive_error, exhaustive_count = (0.0, 0.0, 0)
    for row in budgets.to_dict('records'):
        key = tuple((row[name] for name in BUDGET_KEYS))
        if row['status'] != 'scored':
            continue
        population = populations[key[:len(PAIR)]]
        selected = json.loads(row['calibration_identities_json'])
        test = json.loads(row['test_identities_json'])
        xall = population.source_value.to_numpy(float)
        xc = population.loc[selected, 'source_value'].to_numpy(float)
        yc = population.loc[selected, 'target_value'].to_numpy(float)
        xt = population.loc[test, 'source_value'].to_numpy(float)
        yt = population.loc[test, 'target_value'].to_numpy(float)
        reference_mse = float(np.mean((yt - yc.mean()) ** 2))
        score_ok &= np.isclose(reference_mse, score_lookup.loc[key + ('target_calibration_mean',), 'mse'], atol=1e-12, rtol=1e-10)
        entries = {}
        for model in MODELS:
            intercept, slope = closed_form(xall, xc, yc, model == 'source_affine_ridge')
            saved = score_lookup.loc[key + (model,)]
            fit_ok &= np.isclose(intercept, saved.intercept, atol=1e-10, rtol=1e-09)
            fit_ok &= np.isclose(slope, saved.slope, atol=1e-10, rtol=1e-09)
            exact = frozen_test_reference(xt, yt, intercept, slope)
            error = abs(exact['true_mse'] - float(saved.mse))
            max_error = max(max_error, error)
            score_ok &= error < 1e-10
            exact.update(intercept=intercept, slope=slope, target_mean_mse=reference_mse, net_gain=reference_mse - exact['true_mse'])
            entries[model] = exact
            if int(row['design']) == 0 and len(test) <= 4:
                losses = [np.mean((yt - intercept - slope * np.asarray(order)) ** 2) for order in itertools.permutations(xt)]
                exhaustive_error = max(exhaustive_error, abs(float(np.mean(losses)) - exact['permuted_mse']))
                exhaustive_count += len(losses)
        contexts[key] = dict(population=population, selected=selected, test=test, source_all=xall, source_cal=xc, target_cal=yc, source_test=xt, target_test=yt, models=entries)
    return (populations, contexts, {'fits': fit_ok, 'scores': score_ok, 'max_mse_difference': max_error, 'enumerated_patterns': exhaustive_count, 'max_exhaustive_difference': exhaustive_error})

def reconstruct_mappings(pairs, populations):
    mapping_rows, registry_rows, maps = ([], [], {})
    version = 'source-identity-alignment-v1'
    for pair in pairs:
        population = populations.get(pair)
        identities = [] if population is None else population.index.tolist()
        values = np.empty(0) if population is None else population.source_value.to_numpy(float)
        n = len(identities)
        distinct_values = set()
        if n >= 6:
            seen, accepted, candidate = (set(), 0, 0)
            while accepted < N_DRAWS:
                hashes = [hashlib.sha256(compact([version, pair[0], pair[1], candidate, identity]).encode('utf-8')).hexdigest() for identity in identities]
                order = tuple(sorted(range(n), key=lambda j: (hashes[j], identities[j])))
                if order != tuple(range(n)) and order not in seen:
                    seen.add(order)
                    assigned = [identities[j] for j in order]
                    signature = hashlib.sha256(compact([version, pair[0], pair[1], assigned]).encode('utf-8')).hexdigest()
                    vector = values[list(order)]
                    distinct_values.add(tuple(vector))
                    maps[pair + (accepted,)] = dict(candidate=candidate, signature=signature, assigned=dict(zip(identities, assigned)), vector=vector, identity_order=identities)
                    for receiving, donor in zip(identities, assigned):
                        mapping_rows.append({**dict(zip(PAIR, pair)), 'draw': accepted, 'candidate': candidate, 'mapping_sha256': signature, 'unit_identity': receiving, 'assigned_source_identity': donor})
                    accepted += 1
                candidate += 1
        registry_rows.append({**dict(zip(PAIR, pair)), 'n_identities': n, 'mapping_status': 'mapped' if n >= 6 else 'geometry_absent' if n == 0 else 'insufficient_identity_support', 'n_mappings': N_DRAWS if n >= 6 else 0, 'n_distinct_assigned_source_vectors': len(distinct_values), 'all_source_zero': bool(n and np.max(np.abs(values)) <= EPS), 'all_source_constant': bool(n and np.ptp(values) <= EPS)})
    return (pd.DataFrame(mapping_rows), pd.DataFrame(registry_rows), maps)

def reconstruct_scores(budgets, contexts, maps):
    b24 = {tuple((row[name] for name in PAIR + ['design'])): row['status'] == 'scored' for row in budgets[budgets.budget.eq(24)].to_dict('records')}
    exact_rows, draw_rows, secondary_rows = ([], [], [])
    exact_fields = ['intercept', 'slope', 'source_only_mean', 'source_only_sd', 'real_mse', 'target_mean_mse', 'test_permuted_exact_mse', 'delta_test', 'net_gain', 'test_source_mean', 'test_source_variance', 'test_target_mean', 'test_covariance', 'delta_test_covariance_identity_error', 'n_distinct_test_source_values']
    secondary_fields = ['real_mse', 'target_mean_mse', 'mean_permuted_mse', 'delta_pair', 'net_gain']
    for row in budgets.to_dict('records'):
        key = tuple((row[name] for name in BUDGET_KEYS))
        common = {name: row[name] for name in BUDGET_KEYS + ['status', 'n_primary_identities', 'n_test', 'n_calibration', 'actual_spent', 'unused_budget']}
        common['b24_scored_domain'] = b24[key[:-1]]
        for model in MODELS:
            exact = {**common, 'model': model, **dict.fromkeys(exact_fields, np.nan)}
            secondary = {**common, 'model': model, **dict.fromkeys(secondary_fields, np.nan), 'n_draws': 0}
            if row['status'] == 'scored':
                context = contexts[key]
                reference = context['models'][model]
                real, test_reference = (reference['true_mse'], reference['permuted_mse'])
                delta_test = test_reference - real
                exact.update(intercept=reference['intercept'], slope=reference['slope'], source_only_mean=float(context['source_all'].mean()), source_only_sd=float(context['source_all'].std()), real_mse=real, target_mean_mse=reference['target_mean_mse'], test_permuted_exact_mse=test_reference, delta_test=delta_test, net_gain=reference['net_gain'], test_source_mean=float(context['source_test'].mean()), test_source_variance=float(context['source_test'].var()), test_target_mean=float(context['target_test'].mean()), test_covariance=reference['test_covariance'], delta_test_covariance_identity_error=delta_test - reference['alignment_advantage'], n_distinct_test_source_values=len(np.unique(context['source_test'])))
                population, selected, test = (context['population'], context['selected'], context['test'])
                mapping_set = [maps[key[:len(PAIR)] + (draw,)] for draw in range(N_DRAWS)]
                assigned = np.asarray([mapping['vector'] for mapping in mapping_set])
                xc = assigned[:, population.index.get_indexer(selected)]
                xt = assigned[:, population.index.get_indexer(test)]
                means = xc.mean(axis=1)
                if model == 'source_offset':
                    slopes = np.ones(N_DRAWS)
                elif context['source_all'].std() <= EPS:
                    slopes = np.zeros(N_DRAWS)
                else:
                    centered = xc - means[:, None]
                    slopes = centered @ (context['target_cal'] - context['target_cal'].mean())
                    slopes /= np.sum(centered ** 2, axis=1) + 3.0 * context['source_all'].var()
                intercepts = context['target_cal'].mean() - slopes * means
                losses = np.mean((context['target_test'] - intercepts[:, None] - slopes[:, None] * xt) ** 2, axis=1)
                for draw, mapping in enumerate(mapping_set):
                    intercept, slope, loss = (float(intercepts[draw]), float(slopes[draw]), float(losses[draw]))
                    draw_rows.append({**{name: row[name] for name in BUDGET_KEYS}, 'model': model, 'draw': draw, 'candidate': mapping['candidate'], 'mapping_sha256': mapping['signature'], 'b24_scored_domain': common['b24_scored_domain'], 'n_test': row['n_test'], 'n_calibration': row['n_calibration'], 'actual_spent': row['actual_spent'], 'permuted_mse': loss, 'real_mse': real, 'target_mean_mse': reference['target_mean_mse'], 'pair_advantage': loss - real, 'permuted_intercept': intercept, 'permuted_slope': slope})
                average = float(np.mean(losses))
                secondary.update(real_mse=real, target_mean_mse=reference['target_mean_mse'], mean_permuted_mse=average, delta_pair=average - real, net_gain=reference['net_gain'], n_draws=N_DRAWS)
            exact_rows.append(exact)
            secondary_rows.append(secondary)
    return (pd.DataFrame(exact_rows), pd.DataFrame(draw_rows), pd.DataFrame(secondary_rows))

def reconstruct_aggregates(scores, secondary=False):
    signal = 'delta_pair' if secondary else 'delta_test'
    reference = 'mean_permuted_mse' if secondary else 'test_permuted_exact_mse'
    metrics = ['real_mse', 'target_mean_mse', reference, signal, 'net_gain', 'actual_spent', 'n_calibration', 'n_test']
    direction_rows = []
    for group_key, all_designs in scores.groupby(PAIR + ['budget', 'model'], sort=True):
        for domain in ('all_scored', 'b24_fixed'):
            selected = all_designs if domain == 'all_scored' else all_designs[all_designs.b24_scored_domain]
            scored = selected[selected.status.eq('scored')]
            status = 'scored' if len(scored) else 'outside_b24_scored_designs' if not len(selected) else all_designs.status.iloc[0]
            row = {**dict(zip(PAIR + ['budget', 'model'], group_key)), 'domain': domain, 'n_registered_designs': len(all_designs), 'n_domain_designs': len(selected), 'n_scored_designs': len(scored), 'status': status}
            row.update({name: float(scored[name].mean()) if len(scored) else np.nan for name in metrics})
            direction_rows.append(row)
    directions = pd.DataFrame(direction_rows)
    headline_rows = []
    for group_key, all_directions in directions.groupby(['domain', 'relation', 'budget', 'model'], sort=True):
        scored = all_directions[all_directions.status.eq('scored')]
        row = {**dict(zip(['domain', 'relation', 'budget', 'model'], group_key)), 'n_registered_directions': len(all_directions), 'n_scored_directions': len(scored), 'n_scored_designs': int(scored.n_scored_designs.sum()), 'positive_' + signal + '_directions': int(scored[signal].gt(EPS).sum()), 'positive_net_directions': int(scored.net_gain.gt(EPS).sum()), 'positive_delta_without_net_directions': int((scored[signal].gt(EPS) & ~scored.net_gain.gt(EPS)).sum()), 'neutral_' + signal + '_directions': int(scored[signal].abs().le(EPS).sum())}
        row.update({name: float(scored[name].mean()) if len(scored) else np.nan for name in metrics})
        for name in ('real_mse', 'target_mean_mse', reference):
            row[name.replace('_mse', '_rmse_pp')] = 100.0 * np.sqrt(row[name])
        headline_rows.append(row)
    return (directions, pd.DataFrame(headline_rows))

def main():
    checks = []

    def check(name, passed, **details):
        if name in {entry['check'] for entry in checks}:
            raise ValueError('duplicate verification check: ' + name)
        checks.append({'check': name, 'passed': bool(passed), **details})

    def compare(name, actual, expected, keys, columns=None):
        passed, details = compare_frames(actual, expected, keys, columns)
        check(name, passed, saved_rows=len(actual), rebuilt_rows=len(expected), **details)

    def read(name):
        return pd.read_csv(TABLES / (PREFIX + '_' + name))
    support = pd.read_csv(TABLES / 'budgeted_target_calibration_unit_support.csv.gz')
    budgets = pd.read_csv(TABLES / 'budgeted_target_calibration_budget_designs.csv.gz')
    old_scores = pd.read_csv(TABLES / 'budgeted_target_calibration_scores.csv')
    pairs = sorted(set(map(tuple, budgets[PAIR].to_numpy())))
    check('frozen57_complete_domain_and_status_ledger', len(pairs) == 90 and len(budgets) == 3600 and (len(support) == 552) and (len(old_scores) == 18000) and (budgets.status.value_counts().to_dict() == {'scored': 1309, 'geometry_absent': 1280, 'insufficient_identity_support': 800, 'budget_insufficient': 211}))
    check('frozen57_unique_keys', not budgets.duplicated(BUDGET_KEYS).any() and (not support.duplicated(PAIR + ['unit_identity']).any()) and (not old_scores.duplicated(BUDGET_KEYS + ['model']).any()))
    coordinates_ok, records_ok, supported_directions = full_role_disjointness(support)
    check('all54_complete_source_target_coordinate_unions_disjoint', coordinates_ok and supported_directions == 58 and support.source_screen.ne(support.target_screen).all())
    check('all54_complete_source_target_import_record_unions_disjoint', records_ok)
    support_audit_rows = []
    for pair, part in support.groupby(PAIR, sort=True):
        coordinates, records = ({}, {})
        for role in ('source', 'target'):
            coordinates[role] = {tuple(value) for text in part[role + '_coordinates_json'] for value in json.loads(text)}
            records[role] = {value for text in part[role + '_record_ids_json'] for value in json.loads(text)}
        support_audit_rows.append({**dict(zip(PAIR, pair)), 'n_identities': len(part), 'source_coordinate_count': len(coordinates['source']), 'target_coordinate_count': len(coordinates['target']), 'all_source_target_coordinate_overlap': len(coordinates['source'] & coordinates['target']), 'all_source_target_record_overlap': len(records['source'] & records['target'])})
    compare('independent_all54_source_target_support_audit', read('source_target_support_audit.csv'), pd.DataFrame(support_audit_rows), PAIR)
    within_source_ok = True
    for _, part in support.groupby(PAIR, sort=True):
        for suffix in ('coordinates', 'record_ids'):
            seen = set()
            for text in part['source_' + suffix + '_json']:
                values = {tuple(value) if isinstance(value, list) else value for value in json.loads(text)}
                within_source_ok &= not bool(seen & values)
                seen.update(values)
    check('source_identity_coordinate_and_record_supports_disjoint_under_any_bijection', within_source_ok)
    populations, contexts, baseline = baseline_reconstruction(support, budgets, old_scores)
    check('independent_real_affine_and_offset_parameters_match57', baseline['fits'])
    check('independent_real_and_target_mean_risks_match57', baseline['scores'], max_absolute_difference=baseline['max_mse_difference'])
    check('frozen_coefficient_test_reference_matches_explicit_small_test_permutations', baseline['enumerated_patterns'] > 0 and baseline['max_exhaustive_difference'] < 1e-12, enumerated_patterns=baseline['enumerated_patterns'], max_absolute_difference=baseline['max_exhaustive_difference'])
    mappings, registry, maps = reconstruct_mappings(pairs, populations)
    actual_mappings, actual_registry = (read('mappings.csv.gz'), read('mapping_registry.csv'))
    compare('independent_all128_unique_nonidentity_outcome_free_hash_mappings', actual_mappings, mappings, PAIR + ['draw', 'unit_identity'])
    compare('independent_full78_mapping_registry_and_source_degeneracies', actual_registry, registry, PAIR)
    mapping_invariants = True
    for key, part in actual_mappings.groupby(PAIR + ['draw'], sort=True):
        population = populations[key[:len(PAIR)]]
        assigned = population.loc[part.assigned_source_identity, 'source_value'].to_numpy(float)
        original = population.source_value.to_numpy(float)
        mapping_invariants &= set(part.unit_identity) == set(population.index)
        mapping_invariants &= set(part.assigned_source_identity) == set(population.index)
        mapping_invariants &= np.array_equal(np.sort(assigned), np.sort(original))
        mapping_invariants &= np.isclose(assigned.mean(), original.mean(), atol=1e-12, rtol=0)
        mapping_invariants &= np.isclose(assigned.std(), original.std(), atol=1e-12, rtol=0)
    check('all_source_marginals_scales_and_bijections_preserved', mapping_invariants and len(maps) == 38 * N_DRAWS)
    print('Independent baseline, exact-test reference and 128 mappings rebuilt', flush=True)
    exact, draws, secondary = reconstruct_scores(budgets, contexts, maps)
    actual_exact, actual_draws, actual_secondary = (read('design_scores.csv'), read('draw_scores.csv.gz'), read('secondary_scores.csv'))
    score_keys = BUDGET_KEYS + ['model']
    compare('independent_all6240_fixed_coefficient_design_scores', actual_exact, exact, score_keys)
    check('exact_test_correspondence_identity_all_scored_rows', exact.loc[exact.status.eq('scored'), 'delta_test_covariance_identity_error'].abs().max() < 1e-12)
    compare('independent_all330240_secondary_draw_fits_and_losses', actual_draws, draws, score_keys + ['draw'])
    secondary_columns = ['status', 'real_mse', 'target_mean_mse', 'mean_permuted_mse', 'delta_pair', 'net_gain', 'n_draws']
    if 'b24_scored_domain' in actual_secondary:
        secondary_columns.append('b24_scored_domain')
    compare('independent_all6240_secondary_means_and_guard_rows', actual_secondary, secondary, score_keys, secondary_columns)
    check('all_designs_budgets_models_use_same_frozen_direction_mapping', actual_draws.groupby(PAIR + ['draw']).mapping_sha256.nunique().eq(1).all() and actual_draws.groupby(PAIR + ['draw']).candidate.nunique().eq(1).all())
    guard_exact = actual_exact[actual_exact.status.ne('scored')]
    guard_secondary = actual_secondary[actual_secondary.status.ne('scored')]
    check('unknown_and_guards_have_no_scores_or_draws', len(guard_exact) == len(guard_secondary) == (3600 - 1309) * 2 and guard_exact[['real_mse', 'test_permuted_exact_mse', 'delta_test', 'net_gain']].isna().all().all() and guard_secondary[['real_mse', 'mean_permuted_mse', 'delta_pair', 'net_gain']].isna().all().all() and (len(actual_draws) == 1309 * 2 * N_DRAWS))
    for secondary_flag, design_table, stem in ((False, exact, ''), (True, secondary, 'secondary_')):
        directions, headlines = reconstruct_aggregates(design_table, secondary_flag)
        compare('independent_' + stem + 'direction_macro_aggregation', read(stem + 'direction_scores.csv'), directions, PAIR + ['domain', 'budget', 'model'])
        compare('independent_' + stem + 'headline_equal_direction_aggregation', read(stem + 'headline.csv'), headlines, ['domain', 'relation', 'budget', 'model'])
    fixed = budgets[budgets.budget.eq(24) & budgets.status.eq('scored') & budgets.relation.eq('cross_family')]
    fixed_keys = fixed[PAIR + ['design']]
    check('fixed_b24_cross_family_domain_is21directions182designs', len(fixed) == 158 and len(fixed[PAIR].drop_duplicates()) == 17)
    selected = actual_exact.merge(fixed_keys, on=PAIR + ['design'], validate='many_to_one')
    check('fixed_b24_tests_costs_and_domains_not_selected_by_new_losses', len(selected) == 158 * 4 * 2 and selected.status.eq('scored').all() and selected.b24_scored_domain.all())
    offset = selected[selected.model.eq('source_offset')]
    offset_pivot = offset.pivot(index=PAIR + ['design'], columns='budget', values='delta_test')
    check('fixed_test_offset_alignment_budget_invariance_is_algebraic', np.allclose(offset_pivot.to_numpy(), offset_pivot.iloc[:, 0].to_numpy()[:, None], rtol=1e-09, atol=1e-12))
    coverage_rows = []
    for (relation, budget), part in budgets.groupby(['relation', 'budget'], sort=True):
        scored = part[part.status.eq('scored')]
        row = {'relation': relation, 'budget': budget, 'n_registered_directions': len(part[PAIR].drop_duplicates()), 'n_registered_designs': len(part), 'n_scored_directions': len(scored[PAIR].drop_duplicates()), 'n_scored_designs': len(scored)}
        row.update({'n_' + status: int(part.status.eq(status).sum()) for status in sorted(budgets.status.unique())})
        coverage_rows.append(row)
    compare('independent_full_registered_coverage_and_all_guards', read('coverage.csv'), pd.DataFrame(coverage_rows), ['relation', 'budget'])
    manifest = json.loads((DATA / (PREFIX + '_manifest.json')).read_text(encoding='utf-8'))
    for category in ('input_sha256', 'code_sha256', 'output_sha256'):
        entries = manifest[category]
        failures = [name for name, value in entries.items() if not (HERE / name).is_file() or digest(HERE / name) != value]
        check('manifest_' + category, not failures, files=len(entries), mismatched=failures)
    expected_tables = {str(path.relative_to(HERE)).replace('\\', '/') for path in TABLES.glob(PREFIX + '_*')}
    manifest_outputs = {name.replace('\\', '/') for name in manifest['output_sha256']}
    check('manifest_includes_all_eleven_machine_tables_and_independent_verifier', len(expected_tables) == 11 and expected_tables.issubset(manifest_outputs) and (Path(__file__).name in manifest['code_sha256']))
    figures = list((HERE / 'figures').glob('47*SOURCE_IDENTITY*'))
    figure_ok = len(figures) == 1
    if figure_ok:
        with Image.open(figures[0]) as picture:
            pixels = np.asarray(picture.convert('RGB'))
        figure_ok = pixels.shape[0] >= 700 and pixels.shape[1] >= 1200 and (pixels.std() > 10)
    check('nonblank_figure47', figure_ok)
    check('unique_check_names', len({entry['check'] for entry in checks}) == len(checks))
    result = {'passed': all((entry['passed'] for entry in checks)), 'checks_total': len(checks), 'checks_passed': sum((entry['passed'] for entry in checks)), 'counts': {'registered_directions': len(pairs), 'supported_directions': supported_directions, 'mapped_directions': len(maps) // N_DRAWS, 'mappings': len(maps), 'design_model_rows': len(exact), 'draw_model_rows': len(draws), 'fixed_b24_cross_directions': 17, 'fixed_b24_cross_designs': 158}, 'checks': checks}
    output = DATA / (PREFIX + '_verification.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
