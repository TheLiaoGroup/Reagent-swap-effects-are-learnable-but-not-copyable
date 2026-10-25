"""Report70: atom-weighted source level and contrast under fixed target costs."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import budgeted_target_calibration as legacy57
import source_identity_alignment as legacy60
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'cross_library_source_channels'
REPORT = HERE / '70_CROSS_LIBRARY_SOURCE_CHANNELS_2026-10-18.md'
FIGURE = FIGURES / '55_CROSS_LIBRARY_SOURCE_CHANNELS.png'
SPEC = HERE / 'cross_library_source_channels.py'
PAIR = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
UNIT = PAIR + ['unit_identity']
DESIGN = PAIR + ['design', 'budget']
CHANNELS = ('s', 'd', 'sd')
ALL_CHANNELS = (*CHANNELS, 'target_mean')
BUDGETS, ALPHA, EPS = ((24, 48, 96, 192), 3.0, 1e-12)
METRICS = ('real_mse', 'target_mean_mse', 'test_permuted_exact_mse', 'delta_test', 'net_gain')

def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(mapping):
    return [[*key, value.numerator, value.denominator] for key, value in sorted(mapping.items()) if value]

def signature(mapping):
    return hashlib.sha256(compact(canonical(mapping)).encode('utf-8')).hexdigest()

def collapse_atoms(endpoint_ids, endpoints):
    if not endpoint_ids:
        raise ValueError('empty original endpoint population')
    result = defaultdict(Fraction)
    for identifier in endpoint_ids:
        for key, value in endpoints[identifier].items():
            result[key] += value / len(endpoint_ids)
    return {key: value for key, value in result.items() if value}

def expand_channels(atoms, contexts):
    if sum(atoms.values()) != 1 or any((value <= 0 for value in atoms.values())):
        raise ValueError('positive normalized atom weights are required')
    low, high = (defaultdict(Fraction), defaultdict(Fraction))
    for (contrast, pair), weight in atoms.items():
        screen, a, b = contexts[contrast]
        low[screen, int(pair), a] += weight
        high[screen, int(pair), b] += weight
    coordinates = set(low) | set(high)
    output = dict(low=dict(low), high=dict(high))
    output['s'] = {key: (low[key] + high[key]) / 2 for key in coordinates}
    output['d'] = {key: high[key] - low[key] for key in coordinates if high[key] != low[key]}
    return output

def ridge_batch(source_all, calibration, target_calibration, test, channel):
    """Batch SVD implements standardized alpha3 Ridge with unpenalized intercept."""
    all_x, xc, yc, xt = [np.asarray(value, float) for value in (source_all, calibration, target_calibration, test)]
    if all_x.ndim != 2 or all_x.shape[1] != 2 or xc.ndim != 3 or (xt.ndim != 3) or (xc.shape[0] != xt.shape[0]) or (xc.shape[2] != 2) or (xt.shape[2] != 2) or (xc.shape[1] < 3) or (yc.shape != (xc.shape[1],)) or (not all((np.isfinite(value).all() for value in (all_x, xc, yc, xt)))):
        raise ValueError('finite paired S/D and at least three calibration identities required')
    if channel not in ALL_CHANNELS:
        raise ValueError('unregistered source channel')
    mean, sd = (all_x.mean(axis=0), all_x.std(axis=0))
    requested = [] if channel == 'target_mean' else [0] if channel == 's' else [1] if channel == 'd' else [0, 1]
    active = [index for index in requested if sd[index] > EPS]
    coefficients = np.zeros((len(xc), 2))
    intercept = np.full(len(xc), yc.mean())
    if active:
        standardized = (xc[:, :, active] - mean[active]) / sd[active]
        local_mean = standardized.mean(axis=1)
        centered = standardized - local_mean[:, None, :]
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        projection = np.einsum('bni,n->bi', left, yc - yc.mean())
        shrink = np.where(singular > 1e-15, singular / (singular ** 2 + ALPHA), 0.0)
        beta = np.einsum('bij,bi->bj', right, shrink * projection)
        coefficients[:, active] = beta / sd[active]
        intercept -= np.sum(local_mean * beta, axis=1) + coefficients @ mean
    predicted = intercept[:, None] + np.einsum('bni,bi->bn', xt, coefficients)
    return dict(prediction=predicted, intercept=intercept, coef_s=coefficients[:, 0], coef_d=coefficients[:, 1], source_mean_s=float(mean[0]), source_mean_d=float(mean[1]), source_sd_s=float(sd[0]), source_sd_d=float(sd[1]), source_s_constant=bool(sd[0] <= EPS), source_d_constant=bool(sd[1] <= EPS), n_active_features=len(active), calibration_s_constant=np.ptp(xc[:, :, 0], axis=1) <= EPS, calibration_d_constant=np.ptp(xc[:, :, 1], axis=1) <= EPS, ridge_alpha=ALPHA)

def ridge_predict(source_all, source_calibration, target_calibration, source_test, channel):
    fitted = ridge_batch(source_all, np.asarray(source_calibration)[None, :, :], target_calibration, np.asarray(source_test)[None, :, :], channel)
    return {name: value[0] if isinstance(value, np.ndarray) else value for name, value in fitted.items()}

def exact_reference(target, predicted):
    target, predicted = (np.asarray(target, float), np.asarray(predicted, float))
    if target.ndim != 1 or target.shape != predicted.shape or (not len(target)):
        raise ValueError('matched nonempty target and predicted identity vectors required')
    risk = float(np.mean((target - predicted) ** 2))
    reference = float(np.mean((target - predicted.mean()) ** 2) + predicted.var())
    covariance = float(np.mean((target - target.mean()) * (predicted - predicted.mean())))
    return dict(real_mse=risk, test_permuted_exact_mse=reference, delta_test=reference - risk, test_prediction_mean=float(predicted.mean()), test_prediction_variance=float(predicted.var()), test_target_prediction_covariance=covariance, delta_test_covariance_identity_error=reference - risk - 2 * covariance)

def joint_reassign(values, permutation):
    values, permutation = (np.asarray(values), np.asarray(permutation, int))
    if values.ndim != 2 or values.shape[1] != 2 or sorted(permutation.tolist()) != list(range(len(values))):
        raise ValueError('joint S/D permutation must contain each original identity once')
    return values[permutation]

def support_overlaps(source_cal, source_test, target_cal, target_test):
    return dict(source_cal_test_overlap=len(source_cal & source_test), target_cal_test_overlap=len(target_cal & target_test), source_cal_target_test_overlap=len(source_cal & target_test), source_test_target_cal_overlap=len(source_test & target_cal))

def build_sources(collapsed, original_support, atom_table, contexts, provenance):
    endpoint_atoms = {identifier: {(row.contrast_id, int(row.pair_index)): Fraction(int(row.numerator), int(row.denominator)) for row in group.itertuples(index=False)} for identifier, group in atom_table.groupby('unit_id', sort=False)}
    context_map = {row.contrast_id: (row.screen, row.condition_low, row.condition_high) for row in contexts.itertuples(index=False)}
    records = defaultdict(set)
    for row in provenance.itertuples(index=False):
        records[row.screen, int(row.pair_index), row.condition].add(row.record_id)
    matrices, values = ({}, {})

    def observed(coordinate):
        screen, pair, condition = coordinate
        if screen not in matrices:
            path = DATA / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
            with np.load(path, allow_pickle=False) as archive:
                matrices[screen] = (archive['M'], {name: i for i, name in enumerate(archive['conds'].astype(str))})
        matrix, indices = matrices[screen]
        value = float(matrix[pair, indices[condition]])
        if not np.isfinite(value) or coordinate not in records:
            raise ValueError('original atom endpoint lacks finite yield or record provenance')
        values[coordinate] = value
        return value
    support_lookup = original_support.set_index(UNIT)
    sources, atoms_saved, coefficients = ([], [], [])
    for row in collapsed.to_dict('records'):
        key = {name: row[name] for name in UNIT}
        ids = json.loads(row['source_endpoint_ids_json'])
        atoms = collapse_atoms(ids, endpoint_atoms)
        operators = expand_channels(atoms, context_map)
        numeric = {name: sum((float(weight) * observed(coordinate) for coordinate, weight in operator.items())) for name, operator in operators.items()}
        previous = support_lookup.loc[tuple(key.values())]
        old_coordinates = set(map(tuple, json.loads(previous.source_coordinates_json)))
        union = set(operators['s'])
        if signature(operators['d']) != row['source_operator_signature'] or set(operators['d']) != old_coordinates or (not np.isclose(numeric['d'], previous.source_value, atol=1e-12, rtol=0)) or (not np.isclose(numeric['high'] - numeric['low'], numeric['d'], atol=1e-12, rtol=0)):
            raise ValueError('atom-weighted D changed the original report57 response')
        original_records = set(json.loads(previous.source_record_ids_json))
        union_records = set((record for coordinate in union for record in records[coordinate]))
        source = {**key, 'source_low': numeric['low'], 'source_high': numeric['high'], 'source_s': numeric['s'], 'source_d': numeric['d'], 'source_d_57': float(previous.source_value), 'target_value': float(previous.target_value), 'source_endpoint_ids_json': row['source_endpoint_ids_json'], 'n_registry_rows': len(ids), 'n_atoms': len(atoms), 'atom_signature': signature(atoms), 's_operator_signature': signature(operators['s']), 'd_operator_signature': signature(operators['d']), 'source_common_coordinates_json': compact([list(item) for item in sorted(union)]), 'source_common_record_ids_json': compact(sorted(union_records)), 'source_common_coordinate_cost': len(union), 'source_common_record_count': len(union_records), 'source_d_57_coordinate_cost': len(old_coordinates), 'source_canceled_only_coordinates_json': compact([list(item) for item in sorted(union - old_coordinates)]), 'source_canceled_only_coordinate_count': len(union - old_coordinates), 'source_canceled_only_record_ids_json': compact(sorted(union_records - original_records)), 'source_d_reconstruction_error': abs(numeric['d'] - previous.source_value), 'source_common_authorization_sha256': hashlib.sha256(compact([list(item) for item in sorted(union)]).encode()).hexdigest()}
        sources.append(source)
        for (contrast, pair), weight in sorted(atoms.items()):
            atoms_saved.append({**key, 'contrast_id': contrast, 'pair_index': pair, 'numerator': weight.numerator, 'denominator': weight.denominator})
        for name, operator in operators.items():
            for (screen, pair, condition), weight in sorted(operator.items()):
                coefficients.append({**key, 'operator': name, 'screen': screen, 'pair_index': pair, 'condition': condition, 'numerator': weight.numerator, 'denominator': weight.denominator})
    coordinates = pd.DataFrame([dict(screen=screen, pair_index=pair, condition=condition, matrix_yield=value) for (screen, pair, condition), value in sorted(values.items())])
    record_table = coordinates.merge(provenance, on=['screen', 'pair_index', 'condition'], how='left', validate='one_to_many', suffixes=('', '_52'))
    return (pd.DataFrame(sources).sort_values(UNIT).reset_index(drop=True), pd.DataFrame(atoms_saved), pd.DataFrame(coefficients), record_table)

def chemical_trace(collapsed, atom_table, contexts):
    endpoints = {identifier: {(row.contrast_id, int(row.pair_index)): Fraction(int(row.numerator), int(row.denominator)) for row in group.itertuples(index=False)} for identifier, group in atom_table.groupby('unit_id', sort=False)}
    rows = []
    for row in collapsed.to_dict('records'):
        key = {name: row[name] for name in UNIT}
        for role in ('source', 'target'):
            identifiers = json.loads(row[role + '_endpoint_ids_json'])
            atoms = collapse_atoms(identifiers, endpoints)
            weights, counts = (defaultdict(Fraction), defaultdict(int))
            for (contrast, _), weight in atoms.items():
                weights[contrast] += weight
                counts[contrast] += 1
            if sum(weights.values()) != 1:
                raise ValueError('chemical trace lost original endpoint registration weights')
            for contrast, weight in sorted(weights.items()):
                rows.append({**key, 'role': role, 'contrast_id': contrast, 'context_weight_numerator': weight.numerator, 'context_weight_denominator': weight.denominator, 'n_atoms': counts[contrast], 'n_endpoint_registrations': len(identifiers), 'n_unique_endpoints': len(set(identifiers))})
    return pd.DataFrame(rows).merge(contexts, on='contrast_id', how='left', validate='many_to_one')

def source_resources(sources, support, budgets):
    lookup = sources.set_index(UNIT)
    legacy = support.set_index(UNIT)
    direction_rows, support_rows = ([], [])

    def coordinates(part, column):
        return set((tuple(coordinate) for text in part[column] for coordinate in json.loads(text)))
    for key, group in sources.groupby(PAIR, sort=True):
        old = support.set_index(PAIR).loc[[key]].reset_index()
        common = coordinates(group, 'source_common_coordinates_json')
        original = coordinates(old, 'source_coordinates_json')
        target = coordinates(old, 'target_coordinates_json')
        source_records = set((record for text in group.source_common_record_ids_json for record in json.loads(text)))
        target_records = set((record for text in old.target_record_ids_json for record in json.loads(text)))
        direction_rows.append({**dict(zip(PAIR, key)), 'n_identities': len(group), 'source_common_coordinates_json': compact([list(x) for x in sorted(common)]), 'source_common_record_ids_json': compact(sorted(source_records)), 'source_common_coordinate_cost': len(common), 'source_d_57_coordinate_cost': len(original), 'source_canceled_only_coordinate_count': len(common - original), 'source_canceled_only_coordinates_json': compact([list(x) for x in sorted(common - original)]), 'source_target_coordinate_overlap': len(common & target), 'source_target_record_overlap': len(source_records & target_records), 'source_common_authorization_sha256': hashlib.sha256(compact([list(x) for x in sorted(common)]).encode()).hexdigest()})
    directions = pd.DataFrame(direction_rows)
    auth = directions.set_index(PAIR).source_common_authorization_sha256.to_dict()
    for row in budgets.to_dict('records'):
        key = tuple((row[name] for name in PAIR))
        calibration, test = (json.loads(row['calibration_identities_json']), json.loads(row['test_identities_json']))
        source_sets, source_records = ([], [])
        target_sets, target_records = ([], [])
        for identities in (calibration, test):
            local = [lookup.loc[key + (identity,)] for identity in identities]
            original = [legacy.loc[key + (identity,)] for identity in identities]
            source_sets.append(set((tuple(item) for unit in local for item in json.loads(unit.source_common_coordinates_json))))
            source_records.append(set((item for unit in local for item in json.loads(unit.source_common_record_ids_json))))
            target_sets.append(set((tuple(item) for unit in original for item in json.loads(unit.target_coordinates_json))))
            target_records.append(set((item for unit in original for item in json.loads(unit.target_record_ids_json))))
        overlap = support_overlaps(*source_sets, *target_sets)
        record_overlap = {name.replace('overlap', 'record_overlap'): value for name, value in support_overlaps(*source_records, *target_records).items()}
        if any(overlap.values()) or any(record_overlap.values()):
            raise ValueError('expanded common source authorization overlaps held-out target support')
        if set(map(tuple, json.loads(row['calibration_coordinates_json']))) != target_sets[0] or len(target_sets[0]) != row['actual_spent']:
            raise ValueError('original target calibration cost changed')
        support_rows.append({**row, 'source_common_authorization_sha256': auth.get(key, ''), 'source_cal_common_coordinate_cost': len(source_sets[0]), 'source_test_common_coordinate_cost': len(source_sets[1]), **overlap, **record_overlap, 'target_coordinates_unchanged': True})
    if directions[['source_target_coordinate_overlap', 'source_target_record_overlap']].to_numpy().any():
        raise ValueError('full historical source support intersects target labels')
    return (directions, pd.DataFrame(support_rows))

def inherited_mappings(sources, mapping_registry, mapping_table):
    cache = {}
    for key, group in sources.groupby(PAIR, sort=True):
        group = group.sort_values('unit_identity').reset_index(drop=True)
        if len(group) < 6:
            continue
        local = mapping_table
        for name, value in zip(PAIR, key):
            local = local[local[name].eq(value)]
        index = {identity: i for i, identity in enumerate(group.unit_identity)}
        maps = []
        expected = legacy60.fixed_mappings(key[0], key[1], group.unit_identity.tolist())
        for draw, part in local.groupby('draw', sort=True):
            part = part.set_index('unit_identity').loc[group.unit_identity]
            permutation = [index[identity] for identity in part.assigned_source_identity]
            registered = expected[int(draw)]
            if permutation != registered['permutation'] or part.mapping_sha256.nunique() != 1 or part.mapping_sha256.iloc[0] != registered['mapping_sha256']:
                raise ValueError('report60 frozen joint identity mapping changed')
            maps.append(permutation)
        if len(maps) != 128:
            raise ValueError('eligible original direction lacks128 mappings')
        cache[key] = (group, np.asarray(maps, int))
    if len(cache) != int(mapping_registry.mapping_status.eq('mapped').sum()):
        raise ValueError('original mapping guard changed')
    return cache

def build_scores(sources, budgets, mapping_cache):
    primary, secondary, draws, predictions = ([], [], [], [])
    groups = {key: group.set_index('unit_identity').sort_index() for key, group in sources.groupby(PAIR, sort=True)}
    fixed = budgets[budgets.budget.eq(24)].set_index(PAIR + ['design']).status.eq('scored')
    for row in budgets.to_dict('records'):
        key = tuple((row[name] for name in PAIR))
        base = {name: row[name] for name in DESIGN + ['status', 'n_primary_identities', 'n_test', 'n_calibration', 'actual_spent', 'unused_budget']}
        base['b24_scored_domain'] = bool(fixed.loc[key + (row['design'],)])
        if row['status'] == 'scored':
            group = groups[key]
            calibration, test = (json.loads(row['calibration_identities_json']), json.loads(row['test_identities_json']))
            all_x = group[['source_s', 'source_d']].to_numpy(float)
            xc, xt = [group.loc[identities, ['source_s', 'source_d']].to_numpy(float) for identities in (calibration, test)]
            yc, yt = [group.loc[identities, 'target_value'].to_numpy(float) for identities in (calibration, test)]
            target_mean_risk = float(np.mean((yt - yc.mean()) ** 2))
            _, permutations = mapping_cache[key]
            assigned = all_x[permutations]
            indices = {identity: i for i, identity in enumerate(group.index)}
            ci, ti = [[indices[identity] for identity in identities] for identities in (calibration, test)]
        for channel in ALL_CHANNELS:
            item = {**base, 'channel': channel}
            if row['status'] != 'scored':
                primary.append({**item, **{name: np.nan for name in (*METRICS, 'intercept', 'coef_s', 'coef_d', 'source_mean_s', 'source_mean_d', 'source_sd_s', 'source_sd_d', 'n_active_features', 'test_prediction_mean', 'test_prediction_variance', 'test_target_prediction_covariance', 'delta_test_covariance_identity_error', 'source_s_constant', 'source_d_constant', 'calibration_s_constant', 'calibration_d_constant', 'ridge_alpha')}})
                if channel != 'target_mean':
                    secondary.append({**item, 'real_mse': np.nan, 'target_mean_mse': np.nan, 'mean_permuted_mse': np.nan, 'delta_pair': np.nan, 'net_gain': np.nan, 'n_draws': 0})
                continue
            fit = ridge_predict(all_x, xc, yc, xt, channel)
            predicted = fit.pop('prediction')
            reference = exact_reference(yt, predicted)
            primary.append({**item, **fit, **reference, 'target_mean_mse': target_mean_risk, 'net_gain': target_mean_risk - reference['real_mse']})
            for identity, value, x in zip(test, predicted, xt):
                predictions.append({**{name: item[name] for name in DESIGN + ['channel']}, 'unit_identity': identity, 'source_s': float(x[0]), 'source_d': float(x[1]), 'observed': float(group.loc[identity, 'target_value']), 'predicted': float(value)})
            if channel == 'target_mean':
                continue
            batch = ridge_batch(all_x, assigned[:, ci, :], yc, assigned[:, ti, :], channel)
            mse = np.mean((batch['prediction'] - yt[None, :]) ** 2, axis=1)
            for draw, loss in enumerate(mse):
                draws.append({**{name: item[name] for name in DESIGN + ['channel', 'b24_scored_domain']}, 'draw': draw, 'permuted_mse': float(loss), 'real_mse': reference['real_mse'], 'target_mean_mse': target_mean_risk, 'pair_advantage': float(loss - reference['real_mse']), 'permuted_intercept': float(batch['intercept'][draw]), 'permuted_coef_s': float(batch['coef_s'][draw]), 'permuted_coef_d': float(batch['coef_d'][draw])})
            secondary.append({**item, 'real_mse': reference['real_mse'], 'target_mean_mse': target_mean_risk, 'mean_permuted_mse': float(mse.mean()), 'delta_pair': float(mse.mean() - reference['real_mse']), 'net_gain': target_mean_risk - reference['real_mse'], 'n_draws': 128})
    return (pd.DataFrame(primary), pd.DataFrame(secondary), pd.DataFrame(draws), pd.DataFrame(predictions))

def aggregate(scores, secondary=False):
    metrics = ('real_mse', 'target_mean_mse', 'mean_permuted_mse', 'delta_pair', 'net_gain') if secondary else METRICS
    directions = []
    for domain in ('all_scored', 'b24_fixed'):
        for key, group in scores.groupby(PAIR + ['budget', 'channel'], sort=True):
            inside = group if domain == 'all_scored' else group[group.b24_scored_domain]
            scored = inside[inside.status.eq('scored')]
            item = {**dict(zip(PAIR + ['budget', 'channel'], key)), 'domain': domain, 'n_registered_designs': len(group), 'n_domain_designs': len(inside), 'n_scored_designs': len(scored), 'status': 'scored' if len(scored) else 'outside_b24_scored_designs' if not len(inside) and domain == 'b24_fixed' else str(group.status.iloc[0])}
            item.update({name: float(scored[name].mean()) if len(scored) else np.nan for name in (*metrics, 'actual_spent', 'n_calibration', 'n_test')})
            directions.append(item)
    directions = pd.DataFrame(directions)
    head = []
    for key, group in directions.groupby(['domain', 'relation', 'budget', 'channel'], sort=True):
        finite = group[group.status.eq('scored')]
        base = dict(zip(['domain', 'relation', 'budget', 'channel'], key))
        for macro in ('direction_macro', 'target_family_macro'):
            values = finite if macro == 'direction_macro' else finite.groupby('target_family', as_index=False)[list(metrics)].mean()
            item = {**base, 'macro': macro, 'n_registered_directions': len(group), 'n_scored_directions': len(finite), 'n_scored_designs': int(finite.n_scored_designs.sum()), 'n_target_families': finite.target_family.nunique(), 'positive_net_directions': int(finite.net_gain.gt(EPS).sum())}
            item.update({name: float(values[name].mean()) for name in metrics})
            for name in metrics:
                if name.endswith('_mse'):
                    item[name.removesuffix('_mse') + '_rmse_pp'] = 100 * np.sqrt(item[name])
            head.append(item)
    return (directions, pd.DataFrame(head))

def channel_comparisons(scores):
    metrics = ['real_mse', 'target_mean_mse', 'delta_test', 'net_gain']
    table = scores.pivot(index=DESIGN, columns='channel', values=metrics)
    table.columns = [channel + '_' + metric for metric, channel in table.columns]
    table = table.reset_index().merge(scores[scores.channel.eq('sd')][DESIGN + ['status', 'b24_scored_domain', 'n_test', 'n_calibration', 'actual_spent']], on=DESIGN, validate='one_to_one')
    table['d_given_s_gain'] = table.s_real_mse - table.sd_real_mse
    table['s_given_d_gain'] = table.d_real_mse - table.sd_real_mse
    return table

def compare_legacy(primary, secondary, draws, predictions):
    keys = DESIGN
    real = primary[primary.channel.eq('d')]
    old57 = pd.read_csv(TABLES / 'budgeted_target_calibration_scores.csv')
    old57 = old57[old57.model.eq('source_affine_ridge')]
    bridge = real.merge(old57[keys + ['status', 'mse', 'intercept', 'slope']], on=keys, validate='one_to_one', suffixes=('', '_57'))
    bridge['risk_error_57'] = (bridge.real_mse - bridge.mse).abs()
    bridge['intercept_error_57'] = (bridge.intercept - bridge.intercept_57).abs()
    bridge['slope_error_57'] = (bridge.coef_d - bridge.slope).abs()
    old_predictions = pd.read_csv(TABLES / 'budgeted_target_calibration_predictions.csv.gz')
    old_predictions = old_predictions[old_predictions.model.eq('source_affine_ridge')]
    pred = predictions[predictions.channel.eq('d')].merge(old_predictions[keys + ['unit_identity', 'predicted', 'observed', 'target_operator_signature']], on=keys + ['unit_identity'], validate='one_to_one', suffixes=('', '_57'))
    pred['prediction_error_57'] = (pred.predicted - pred.predicted_57).abs()
    old60 = pd.read_csv(TABLES / 'source_identity_alignment_design_scores.csv')
    old60 = old60[old60.model.eq('source_affine_ridge')]
    bridge = bridge.merge(old60[keys + list(METRICS)], on=keys, validate='one_to_one', suffixes=('', '_60'))
    for metric in METRICS:
        bridge[metric + '_error_60'] = (bridge[metric] - bridge[metric + '_60']).abs()
    old_secondary = pd.read_csv(TABLES / 'source_identity_alignment_secondary_scores.csv')
    old_secondary = old_secondary[old_secondary.model.eq('source_affine_ridge')]
    sec = secondary[secondary.channel.eq('d')].merge(old_secondary[keys + ['mean_permuted_mse', 'delta_pair']], on=keys, validate='one_to_one', suffixes=('', '_60'))
    sec['mean_permuted_error_60'] = (sec.mean_permuted_mse - sec.mean_permuted_mse_60).abs()
    sec['delta_pair_error_60'] = (sec.delta_pair - sec.delta_pair_60).abs()
    old_draws = pd.read_csv(TABLES / 'source_identity_alignment_draw_scores.csv.gz')
    old_draws = old_draws[old_draws.model.eq('source_affine_ridge')]
    draw = draws[draws.channel.eq('d')].merge(old_draws[keys + ['draw', 'permuted_mse', 'permuted_intercept', 'permuted_slope']], on=keys + ['draw'], validate='one_to_one', suffixes=('', '_60'))
    for current, prior in (('permuted_mse', 'permuted_mse_60'), ('permuted_intercept', 'permuted_intercept_60'), ('permuted_coef_d', 'permuted_slope')):
        draw[current + '_error_60'] = (draw[current] - draw[prior]).abs()
    maximum = max(bridge.filter(regex='error_').max().max(), pred.prediction_error_57.max(), sec.filter(regex='error_').max().max(), draw.filter(regex='error_').max().max())
    if len(bridge) != 3600 or len(pred) != len(old_predictions) or len(draw) != len(old_draws) or (maximum > 1e-10) or (not bridge.status.eq(bridge.status_57).all()):
        raise ValueError('D-only bridge to original57/60 changed')
    return (bridge, pred, sec, draw, float(maximum))

def plot_figure(headline, secondary, comparisons):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fixed = headline[headline.domain.eq('b24_fixed') & headline.relation.eq('cross_family') & headline.macro.eq('direction_macro')]
    colors = {'s': '#087f8c', 'd': '#c64b5f', 'sd': '#507f36', 'target_mean': '#6c6c6c'}
    for channel in ALL_CHANNELS:
        part = fixed[fixed.channel.eq(channel)].sort_values('budget')
        axes[0, 0].plot(np.arange(4), 10000.0 * part.real_mse, marker='o', color=colors[channel], label=channel)
    axes[0, 0].set(title='A  Fixed cross-library target and calibration designs', ylabel='Held-out MSE (pp squared)')
    for channel in CHANNELS:
        part = fixed[fixed.channel.eq(channel)].sort_values('budget')
        axes[0, 1].plot(np.arange(4), 10000.0 * part.net_gain, marker='o', color=colors[channel], label=channel)
        axes[1, 0].plot(np.arange(4), 10000.0 * part.delta_test, marker='o', color=colors[channel], label=channel)
        part = secondary[secondary.domain.eq('b24_fixed') & secondary.relation.eq('cross_family') & secondary.macro.eq('direction_macro') & secondary.channel.eq(channel)].sort_values('budget')
        axes[1, 1].plot(np.arange(4), 10000.0 * part.delta_pair, marker='o', color=colors[channel], label=channel)
    axes[0, 1].set(title='B  Net value over the same target calibration mean', ylabel='Target mean MSE - model MSE (pp squared)')
    axes[1, 0].set(title='C  Frozen-fit exact TEST identity reference', ylabel='Permuted MSE - real MSE (pp squared)')
    axes[1, 1].set(title='D  Original 128 joint source reassignments and refits', ylabel='Reassigned MSE - real MSE (pp squared)')
    for axis in axes.flat:
        axis.set(xticks=np.arange(4), xticklabels=BUDGETS, xlabel='Target calibration coordinate budget')
        if axis is not axes[0, 0]:
            axis.axhline(0, color='#888888', lw=0.8)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.15)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

def render_report(headline, secondary, directions, context_trace, sources, resources, maximum, counts):
    fixed = headline[headline.domain.eq('b24_fixed') & headline.relation.eq('cross_family') & headline.macro.eq('direction_macro')]
    lines = ['# 70. 跨库来源平均水平与差值的同资源校准', '', '日期：2026-10-18。既有数据的回顾性追加；固定设计见HISTORY_COMPOSITION_AND_SOURCE_CHANNELS_DESIGN_2026-10-18.md。', '', '## 主结果', '', '原B24可评分的跨家族方向与设计固定，四个预算全部保留；先在方向内等设计平均MSE，再等方向平均。目标是报告57原身份内聚合有符号差值，S为来源平均响应水平。MSE及增量单位pp²。', '', '| 预算 | S-only MSE | D-only MSE | SD MSE | target mean MSE | D given S | S given D | SD净值 |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        part = fixed[fixed.budget.eq(budget)].set_index('channel')
        s, d, sd, mean = [part.loc[name].real_mse for name in ALL_CHANNELS]
        lines.append(f'| {budget} | {10000.0 * s:.6f} | {10000.0 * d:.6f} | {10000.0 * sd:.6f} | {10000.0 * mean:.6f} | {10000.0 * (s - sd):+.6f} | {10000.0 * (d - sd):+.6f} | {10000.0 * (mean - sd):+.6f} |')
    lines += ['', '在预定主域，S、D、SD相对同一目标校准均值的净值在四个预算均为负；两个条件增量也均为负。这是本固定alpha=3模型与既定小样本校准设计的平均预测结果，不能据此断言来源水平或差值在所有方向均无信息。', '', '| 预算 | S净值 | D净值 | SD净值 | 正净值方向 S / D / SD（共21） |', '|---:|---:|---:|---:|---|']
    for budget in BUDGETS:
        part = fixed[fixed.budget.eq(budget)].set_index('channel')
        lines.append(f"| {budget} | {10000.0 * part.loc['s'].net_gain:+.6f} | {10000.0 * part.loc['d'].net_gain:+.6f} | {10000.0 * part.loc['sd'].net_gain:+.6f} | " + ' / '.join((str(int(part.loc[name].positive_net_directions)) for name in CHANNELS)) + ' |')
    lines += ['', '## 同资源与响应定义', '', '每个来源身份严格继承52原登记端点及其重复次数：原子权重先在完整底物对内等context，再等伙伴，最后等该身份的登记行。低/高端均值L、H使用同一正权重，S=(L+H)/2、D=H-L；不以D相消后的系数绝对值构造S。目标算子、校准身份、TEST身份、目标坐标预算及guard完全继承57。', '', f"全部{counts['source_units']}来源身份/{counts['finite_directions']}有限方向的D保持原算子，最大来源值重建差为{sources.source_d_reconstruction_error.max():.3g}。新共同source授权由原atom端点并集构成，唯一坐标{counts['legacy_source_coordinates']}增至{counts['common_source_coordinates']}；{counts['expanded_source_identities']}身份/{counts['expanded_source_directions']}方向包含D相消而S需要的坐标，共增加{counts['added_source_coordinates']}个唯一历史坐标。所有坐标已有52记录来源，不是新增实验。S-only、D-only、SD共享相同的完整source授权；不能把这称为与57完全相同的来源信息预算。", '', '完整source池包括未用于目标校准的已知身份，因为标准化继承57对该方向全部source身份取均值和population SD。三种模型固定alpha=3、SVD ridge、截距不罚；各通道独立标准化，常数通道置零，沿用至少3个校准身份。目标raw差值不clip。来源平均水平不是目标活性观测，两个条件增量也不是可相加的信息份额。', '', '## 身份对应与实际净值', '', '精确TEST参照冻结拟合，只在原TEST身份中联合排列(S,D)。以标量预测p计算R_perm=mean((y-mean(p))²)+Var_n(p)，差值等于2Cov_n(y,p)。另继承60全部128个全source身份映射，联合重配S/D后用同一校准目标重新拟合。三通道使用同一映射；两个参照分别保存，不相减为机制贡献。', '', '| 预算 | 通道 | 精确TEST对应优势 | 全source重配重拟合优势 | 相对target mean净值 |', '|---:|---|---:|---:|---:|']
    for row in fixed[fixed.channel.isin(CHANNELS)].sort_values(['budget', 'channel']).itertuples(index=False):
        sec = secondary[secondary.domain.eq('b24_fixed') & secondary.relation.eq('cross_family') & secondary.macro.eq('direction_macro') & secondary.budget.eq(row.budget) & secondary.channel.eq(row.channel)].iloc[0]
        lines.append(f'| {row.budget} | {row.channel} | {10000.0 * row.delta_test:+.6f} | {10000.0 * sec.delta_pair:+.6f} | {10000.0 * row.net_gain:+.6f} |')
    lines += ['', '正的对应优势只表示该模型预测与目标响应之间、相对于指定重排参照的对应结构；它不保证比目标均值预测更准。SD多一个惩罚特征后平均风险上升，是此模型/样本设计下的描述性结果；本分析没有识别其方差成因，也没有据此调整alpha或选择预算。', '', '## 权重与支持敏感性', '', '下表保留同一B24固定跨家族域，但先等方向平均到target family，再等target family平均。它是敏感性，不替代上面的预定主比较。', '', '| 预算 | S净值 | D净值 | SD净值 | S精确TEST对应优势 | D精确TEST对应优势 | SD精确TEST对应优势 |', '|---:|---:|---:|---:|---:|---:|---:|']
    family = headline[headline.domain.eq('b24_fixed') & headline.relation.eq('cross_family') & headline.macro.eq('target_family_macro')]
    for budget in BUDGETS:
        part = family[family.budget.eq(budget)].set_index('channel')
        values = [part.loc[name, metric] for metric in ('net_gain', 'delta_test') for name in CHANNELS]
        lines.append(f'| {budget} | ' + ' | '.join((f'{10000.0 * value:+.6f}' for value in values)) + ' |')
    lines += ['', 'D在预算24的target-family净值为正，而方向macro为负；S和SD的精确TEST对应优势在预算48/96/192改为负。因此，主域方向macro中正的对应优势不能推广为对每种family权重都稳健。', '', '| 预算192域 | 可评分方向 / 设计 | D净值 | SD净值 |', '|---|---|---:|---:|']
    for domain, relation in (('b24_fixed', 'cross_family'), ('all_scored', 'cross_family'), ('b24_fixed', 'within_family'), ('all_scored', 'within_family')):
        part = headline[headline.domain.eq(domain) & headline.relation.eq(relation) & headline.macro.eq('direction_macro') & headline.budget.eq(192)].set_index('channel')
        lines.append(f"| {domain} / {relation} | {int(part.loc['d'].n_scored_directions)} / {int(part.loc['d'].n_scored_designs)} | {10000.0 * part.loc['d'].net_gain:+.6f} | {10000.0 * part.loc['sd'].net_gain:+.6f} |")
    lines += ['', 'all-scored允许预算增加后新进入可评分的设计，不能与B24-fixed曲线作同域收益解释。同家族预算192的D/SD平均净值为正；跨家族平均负值不是普遍失败。', '', '## 具体化学方向', '', '以下是预算192、B24固定域中的描述性实例；全部方向保存在direction_scores，未按例子替换headline。化学替换按52原登记端点（含重复次数）展开到原子，再按contrast_id连接原context；完整身份/来源或目标/context及其精确权重保存在chemical_contexts表。', '', '| 来源 → 目标 | 共同身份 | 原context数 来源 / 目标 | 被平均的名义替换 | D净值 | SD净值 | D given S | S given D |', '|---|---:|---|---|---:|---:|---:|---:|']
    examples = (('Ullmann|1', 'Buchwald-Hartwig|1'), ('Buchwald-Hartwig|1', 'Ullmann|1'), ('SNAr|1', 'Sonogashira|1'), ('Heck|1', 'Buchwald-Hartwig(CMC)|2'))
    for source, target in examples:
        part = directions[directions.source_screen.eq(source) & directions.target_screen.eq(target) & directions.domain.eq('b24_fixed') & directions.budget.eq(192)].set_index('channel')
        trace = context_trace[context_trace.source_screen.eq(source) & context_trace.target_screen.eq(target)]
        swaps = trace[['low_name', 'high_name', 'fixed_equiv']].drop_duplicates()
        labels = [str(row.low_name) + ' → ' + str(row.high_name) + (f' ({float(row.fixed_equiv):g} equiv)' if str(row.fixed_equiv) else '') for row in swaps.itertuples(index=False)]
        label = (source + ' → ' + target).replace('|', '/')
        values = (part.loc['d'].net_gain, part.loc['sd'].net_gain, part.loc['s'].real_mse - part.loc['sd'].real_mse, part.loc['d'].real_mse - part.loc['sd'].real_mse)
        lines.append(f"| {label} | {trace.unit_identity.nunique()} | {trace[trace.role.eq('source')].contrast_id.nunique()} / {trace[trace.role.eq('target')].contrast_id.nunique()} | " + '; '.join(labels) + ' | ' + ' | '.join((f'{10000.0 * value:+.6f}' for value in values)) + ' |')
    lines += ['', 'Ullmann/1 → Buchwald-Hartwig/1中，2 equiv的Cs2CO3 → K3PO4或KOtBu在来源130 °C、16 h的4个context与目标100 °C、16 h的2个context内平均；正的SD净值并不要求增加S后优于D。反向使用同一组共同身份/名义替换，SD净值却为负；即使D given S为正，也可能仍不如目标均值。', '', 'SNAr/1 → Sonogashira/1的MeCN → THF与DMSO → MeCN溶剂替换跨越100 °C、16 h与80 °C、12 h的不同背景，SD有正净值，但D given S为负。Heck/1 → Buchwald-Hartwig(CMC)/2的0.05 equiv Pd(OAc)2 → Pd2(dba)3在两库保留相同名义替换，却分别处于100 °C、16 h和80 °C、12 h背景，SD净值为负。名义替换对齐不等于伙伴、配体/碱/溶剂支持或整体响应相同；这些实例没有识别跨反应化学因果机制。']
    lines += ['', '## 支持与解释', '', '完整保留78个登记方向、3120个设计/预算记录及1290个可评分设计；原几何缺失、身份不足和预算不足均保留原guard而不是补零。新source并集在全部设计内核对校准/TEST及跨角色产率坐标和record不相交；这不制造统计独立性。完整方向、all-scored/B24-fixed、within-family及target-family宏平均敏感性全部另表保存。', '', '跨库对象已经在身份内平均伙伴、配方背景与替换登记；它与69/66的个体反应差值分别评分。这里的方向macro沿用57/62，不在已collapsed对象上再造context重复；target-family macro另列。来源各库仍有各自的伙伴和背景支持，不能从模型通道增量识别化学因果机制。设计、方向、source排列和预算均可能共享身份或screen，不作独立复制的显著性检验。', '', '## 复现与核验', '', f'`python -B cross_library_source_channels.py`生成同前缀tables/data、图55及本报告；`python -B -m unittest test_cross_library_source_channels.py`运行focused tests。D-only对57逐预测/参数/风险和60精确TEST及全部draw的最大桥接差为{maximum:.3g}。生产计数为{compact(counts)}。独立核验由单独verifier完成，最终状态以verification.json为准。']
    return '\n'.join(lines) + '\n'

def main():
    started = time.monotonic()
    input_names = [str(SPEC.relative_to(HERE)), 'tables/response_estimand_lineage_collapsed_units.csv.gz', 'tables/response_estimand_lineage_atom_weights.csv.gz', 'tables/response_estimand_lineage_record_provenance.csv', 'tables/cross_library_components_perturbation_contexts.csv', 'tables/budgeted_target_calibration_unit_support.csv.gz', 'tables/budgeted_target_calibration_budget_designs.csv.gz', 'tables/budgeted_target_calibration_scores.csv', 'tables/budgeted_target_calibration_predictions.csv.gz', 'tables/source_identity_alignment_mapping_registry.csv', 'tables/source_identity_alignment_mappings.csv.gz', 'tables/source_identity_alignment_design_scores.csv', 'tables/source_identity_alignment_secondary_scores.csv', 'tables/source_identity_alignment_draw_scores.csv.gz', 'data/response_estimand_lineage_verification.json', 'data/budgeted_target_calibration_verification.json', 'data/source_identity_alignment_verification.json', 'budgeted_target_calibration.py', 'source_identity_alignment.py']
    input_names += [str(path.relative_to(HERE)) for path in sorted(DATA.glob('matrix_*.npz'))]
    hashes = {name: digest(HERE / name) for name in input_names}
    for name in ('response_estimand_lineage', 'budgeted_target_calibration', 'source_identity_alignment'):
        if not json.loads((DATA / (name + '_verification.json')).read_text(encoding='utf-8'))['passed']:
            raise ValueError('an inherited independent verification is not passing')
    collapsed = pd.read_csv(TABLES / 'response_estimand_lineage_collapsed_units.csv.gz', keep_default_na=False)
    collapsed = collapsed[collapsed['mode'].eq('strict') & collapsed.background_control.eq('background_free') & collapsed.basis.eq('position_2') & collapsed.min_partners_each.eq(1) & collapsed.partner_overlap.eq('all')]
    support = pd.read_csv(TABLES / 'budgeted_target_calibration_unit_support.csv.gz', keep_default_na=False)
    budgets = pd.read_csv(TABLES / 'budgeted_target_calibration_budget_designs.csv.gz', keep_default_na=False)
    atom_table = pd.read_csv(TABLES / 'response_estimand_lineage_atom_weights.csv.gz')
    contexts = pd.read_csv(TABLES / 'cross_library_components_perturbation_contexts.csv', keep_default_na=False)
    sources, atoms, coefficients, records = build_sources(collapsed, support, atom_table, contexts, pd.read_csv(TABLES / 'response_estimand_lineage_record_provenance.csv', keep_default_na=False))
    context_trace = chemical_trace(collapsed, atom_table, contexts)
    resources, support_audit = source_resources(sources, support, budgets)
    mapping_registry = pd.read_csv(TABLES / 'source_identity_alignment_mapping_registry.csv', keep_default_na=False)
    mappings = pd.read_csv(TABLES / 'source_identity_alignment_mappings.csv.gz', keep_default_na=False)
    mapping_cache = inherited_mappings(sources, mapping_registry, mappings)
    early = {'source_units.csv.gz': sources, 'atom_weights.csv.gz': atoms, 'operator_coefficients.csv.gz': coefficients, 'chemical_contexts.csv.gz': context_trace, 'source_records.csv.gz': records, 'direction_resources.csv.gz': resources, 'budget_support_audit.csv.gz': support_audit, 'mapping_registry.csv': mapping_registry, 'mappings.csv.gz': mappings}
    outputs = {}

    def save(name, frame):
        path = TABLES / (PREFIX + '_' + name)
        options = dict(index=False)
        if name.endswith('.gz'):
            options['compression'] = {'method': 'gzip', 'mtime': 0}
        frame.to_csv(path, **options)
        outputs[str(path.relative_to(HERE))] = digest(path)
    for name, frame in early.items():
        save(name, frame)
    frozen = dict(outputs)
    print('REPORT70 source operators and authorization frozen', flush=True)
    primary, secondary, draws, predictions = build_scores(sources, budgets, mapping_cache)
    directions, headline = aggregate(primary)
    secondary_directions, secondary_headline = aggregate(secondary, secondary=True)
    comparisons = channel_comparisons(primary)
    bridge, bridge_predictions, bridge_secondary, bridge_draws, maximum = compare_legacy(primary, secondary, draws, predictions)
    for name, frame in (('design_scores.csv.gz', primary), ('secondary_scores.csv.gz', secondary), ('draw_scores.csv.gz', draws), ('predictions.csv.gz', predictions), ('direction_scores.csv', directions), ('headline.csv', headline), ('secondary_direction_scores.csv', secondary_directions), ('secondary_headline.csv', secondary_headline), ('channel_comparisons.csv.gz', comparisons), ('bridge57_60_design.csv.gz', bridge), ('bridge57_predictions.csv.gz', bridge_predictions), ('bridge60_secondary.csv.gz', bridge_secondary), ('bridge60_draws.csv.gz', bridge_draws)):
        save(name, frame)
    common_coordinates = set((tuple(item) for text in sources.source_common_coordinates_json for item in json.loads(text)))
    old_coordinates = set((tuple(item) for text in support.source_coordinates_json for item in json.loads(text)))
    expanded = sources[sources.source_canceled_only_coordinate_count.gt(0)]
    fixed_domain = budgets[budgets.budget.eq(24) & budgets.status.eq('scored') & budgets.relation.eq('cross_family')]
    counts = dict(registered_directions=len(budgets[PAIR].drop_duplicates()), source_units=len(sources), finite_directions=len(resources), budget_design_rows=len(budgets), scored_designs=int(budgets.status.eq('scored').sum()), design_channel_rows=len(primary), draw_channel_rows=len(draws), prediction_rows=len(predictions), mapped_directions=len(mapping_cache), mapping_rows=len(mappings), legacy_source_coordinates=len(old_coordinates), common_source_coordinates=len(common_coordinates), added_source_coordinates=len(common_coordinates - old_coordinates), expanded_source_identities=len(expanded), expanded_source_directions=len(expanded[PAIR].drop_duplicates()), b24_fixed_cross_family_directions=len(fixed_domain[PAIR].drop_duplicates()), b24_fixed_cross_family_designs=len(fixed_domain))
    if (len(sources), len(resources), len(budgets), counts['scored_designs'], len(primary), len(draws)) != (552, 58, 3600, 1309, 14400, 502656):
        raise ValueError('inherited source or target design support changed')
    if (counts['registered_directions'], counts['legacy_source_coordinates'], counts['common_source_coordinates'], counts['added_source_coordinates'], counts['expanded_source_identities'], counts['expanded_source_directions'], counts['b24_fixed_cross_family_directions'], counts['b24_fixed_cross_family_designs']) != (90, 7597, 7679, 82, 54, 6, 17, 158):
        raise ValueError('frozen resource expansion or B24-fixed comparison domain changed')
    plot_figure(headline, secondary_headline, comparisons)
    REPORT.write_text(render_report(headline, secondary_headline, directions, context_trace, sources, resources, maximum, counts), encoding='utf-8')
    outputs[str(REPORT.relative_to(HERE))], outputs[str(FIGURE.relative_to(HERE))] = (digest(REPORT), digest(FIGURE))
    if hashes != {name: digest(HERE / name) for name in hashes} or any((digest(HERE / name) != value for name, value in frozen.items())):
        raise ValueError('an inherited input or frozen resource authorization changed')
    manifest = dict(date='2026-10-18', source_hashes=hashes, output_hashes=outputs, code_hashes={name: digest(HERE / name) for name in ('cross_library_source_channels.py', 'test_cross_library_source_channels.py')}, counts=counts, design=dict(alpha=ALPHA, eps=EPS, all_channels_common_original_atom_union=True, target57_unchanged=True, source_standardization='per feature, all known source identities within direction, population SD', primary_domain='b24_fixed cross_family direction_macro four budgets', secondary_macro='target_family_macro', independent_context_macro=False, joint_original128_mappings=True), max_d_bridge_error=maximum, max_source_d_error=float(sources.source_d_reconstruction_error.max()), max_exact_identity_error=float(primary.delta_test_covariance_identity_error.abs().max()), elapsed_seconds=time.monotonic() - started)
    (DATA / (PREFIX + '_manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(counts=counts, max_d_bridge_error=maximum, elapsed_seconds=manifest['elapsed_seconds']), indent=2), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
