"""E66 fixed-support risks from saved arrays; no model fitting."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
E62 = ANALYSIS / 'Matched_Representation_E62'
E65 = ANALYSIS / 'Structure_Family_Transport_E65'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
GEOMETRIES = ['raw_delta', 'centered_delta', 'centered_unit_rms_zeroembed']
BUDGETS = ['0', '1', '2', '4', '8', '16', 'full']
LEARNERS = ['joint', 'calibration_only', 'calibration_centroid', 'frozen_family', 'zero']
METRICS = ['native_vector_mse', 'native_vector_mae', 'mean_offset_mse', 'centered_error_mse', 'unit0_vector_mse', 'unit0_vector_mae', 'predicted_null_rate']
POPULATIONS = ['all_original', 'family_supported', 'update_supported', 'common8', 'common16', 'family_n0']
DIAGNOSTIC_POPULATIONS = ['family_n1', 'family_n2to4', 'family_n5plus']
STRATA = ['all', 'true_nonzero', 'true_constant']
WEIGHTS = ['pooled', 'equal_edge', 'equal_class_edge_balanced']
COMPARISONS = ['joint_minus_frozen_family', 'joint_minus_calibration_only', 'joint_minus_calibration_centroid']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def transform(x):
    x = np.asarray(x, float)
    null = np.all(x == x[:, :1], axis=1)
    mean = x[:, 0] + (x - x[:, :1]).mean(axis=1)
    c = x - mean[:, None]
    c[null] = 0
    rms = np.sqrt(np.mean(c * c, axis=1))
    u = np.zeros_like(c)
    np.divide(c, rms[:, None], out=u, where=~null[:, None])
    return (c, u, null)

def population_mask(scope, name):
    nf = scope.n_family_train
    nc = scope.n_original_train - nf
    return {'all_original': pd.Series(True, index=scope.index), 'family_supported': nf.ge(1), 'update_supported': nf.ge(1) & nc.ge(1), 'common8': nf.ge(1) & nc.ge(8), 'common16': nf.ge(1) & nc.ge(16), 'family_n0': nf.eq(0), 'family_n1': nf.eq(1), 'family_n2to4': nf.between(2, 4), 'family_n5plus': nf.ge(5)}[name]

def score_arrays(frame, vectors):
    output = []
    for geometry, column in zip(GEOMETRIES, ['prediction_raw', 'prediction_centered', 'prediction_unit']):
        part = frame.drop(columns=['prediction_raw', 'prediction_centered', 'prediction_unit']).copy()
        for metric in METRICS:
            part[metric] = np.nan
        take = frame.supported.to_numpy(bool)
        raw = np.vstack(vectors.loc[frame.vector_id, 'target_vector'])
        c, u, null = transform(raw)
        target = dict(zip(GEOMETRIES, [raw, c, u]))[geometry]
        part['true_constant_vector'] = null
        if take.any():
            pred = np.vstack(frame.loc[take, column])
            pc, decoded, pn = transform(pred)
            tc = transform(target[take])[0]
            error = pred - target[take]
            values = np.column_stack([np.mean(error * error, axis=1), np.mean(np.abs(error), axis=1), error.mean(axis=1) ** 2, np.mean((pc - tc) ** 2, axis=1), np.mean((decoded - u[take]) ** 2, axis=1), np.mean(np.abs(decoded - u[take]), axis=1), pn.astype(float)])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite saved prediction or loss')
            part.loc[take, METRICS] = values
        part['target_geometry'] = geometry
        output.append(part)
    return pd.concat(output, ignore_index=True)

def sufficient_stats(losses, scope, grouping, populations):
    rows = []
    for population in populations:
        ids = scope.loc[population_mask(scope, population), 'vector_id']
        chosen = losses.loc[losses.vector_id.isin(ids)]
        for stratum in STRATA:
            part = chosen if stratum == 'all' else chosen.loc[chosen.true_constant_vector.eq(stratum == 'true_constant')]
            part = part.loc[part.supported]
            if part.empty:
                continue
            g = part.groupby([*EDGE, *grouping], observed=True, sort=False)
            result = g[METRICS].sum(min_count=1).add_prefix('sum_')
            result['n_scored_vectors'] = g.size()
            result = result.reset_index()
            result['population'] = population
            result['stratum'] = stratum
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def summarize(stats, groupkeys, scope):
    rows, contributions = ([], [])
    for keys, group in stats.groupby(['population', *groupkeys], observed=True, sort=True):
        common = dict(zip(['population', *groupkeys], keys))
        parent = group.loc[group.stratum.eq('all')].set_index(EDGE)
        if parent.empty:
            continue
        n_parent = int(parent.n_scored_vectors.sum())
        n_edges = len(parent)
        edges_per_class = parent.reset_index().groupby('reaction_type').size()
        n_classes = len(edges_per_class)
        n_population = int(population_mask(scope, common['population']).sum())
        for stratum in STRATA:
            part = group.loc[group.stratum.eq(stratum)].set_index(EDGE)
            n = int(part.n_scored_vectors.sum())
            n_e = len(part)
            n_c = part.reset_index().reaction_type.nunique()
            for weighting in WEIGHTS:
                if part.empty:
                    risk = {m: np.nan for m in METRICS}
                else:
                    means = part[['sum_' + m for m in METRICS]].div(part.n_scored_vectors, axis=0)
                    if weighting == 'pooled':
                        values = part[['sum_' + m for m in METRICS]].sum() / n
                    elif weighting == 'equal_edge':
                        values = means.mean()
                    else:
                        values = means.groupby(level='reaction_type').mean().mean()
                    risk = dict(zip(METRICS, values.to_numpy(float)))
                rows.append(dict(**common, stratum=stratum, weighting=weighting, n_population_vectors=n_population, n_parent_scored_vectors=n_parent, n_test_vectors=n, n_test_edges=n_e, n_test_classes=n_c, **risk))
                if stratum == 'all':
                    continue
                aligned = part.reindex(parent.index)
                if weighting == 'pooled':
                    coefficient = np.full(n_edges, 1 / n_parent)
                elif weighting == 'equal_edge':
                    coefficient = 1 / (n_edges * parent.n_scored_vectors.to_numpy())
                else:
                    coefficient = 1 / (n_classes * parent.index.get_level_values('reaction_type').map(edges_per_class).to_numpy() * parent.n_scored_vectors.to_numpy())
                mass = float(coefficient @ aligned.n_scored_vectors.fillna(0).to_numpy())
                values = coefficient @ aligned[['sum_' + m for m in METRICS]].fillna(0).to_numpy(float)
                contributions.append(dict(**common, stratum=stratum, weighting=weighting, n_test_vectors=n, full_sample_weight_mass=mass, **dict(zip(METRICS, values))))
    summary = pd.DataFrame(rows)
    contribution = pd.DataFrame(contributions)
    if len(summary) and 'comparison' not in groupkeys:
        summary['native_rmse'] = np.sqrt(summary.native_vector_mse.clip(lower=0))
    return (summary, contribution)

def complete_grid(stats, support, populations, modelkey, models):
    allstats = stats.loc[stats.stratum.eq('all') & stats.population.isin(populations)].copy()
    keys = ['population', 'target_geometry', 'budget', modelkey]
    combinations = pd.MultiIndex.from_product([populations, GEOMETRIES, BUDGETS, models], names=keys).to_frame(index=False)
    frame = support[[*EDGE, 'panel_status']].merge(combinations, how='cross').merge(allstats[[*EDGE, *keys, 'n_scored_vectors', *['sum_' + m for m in METRICS]]], on=[*EDGE, *keys], how='left', validate='one_to_one')
    frame['n_test_vectors'] = frame.n_scored_vectors.fillna(0).astype(int)
    frame = frame.drop(columns='n_scored_vectors')
    for metric in METRICS:
        frame[metric] = frame.pop('sum_' + metric) / frame.n_test_vectors.replace(0, np.nan)
    frame['supported'] = frame.n_test_vectors.gt(0)
    classes = frame.groupby(['reaction_type', *keys], as_index=False, observed=True).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), **{m: (m, 'mean') for m in METRICS})
    classes['supported'] = classes.n_test_edges.gt(0)
    return (frame, classes)

def main():
    started = time.perf_counter()
    paths = sorted((p for p in (BASE / 'parts').glob('edge_*_predictions.parquet') if not p.name.endswith('_union_predictions.parquet')))
    if len(paths) != 93:
        raise ValueError(f'Expected 93 production edges, got {len(paths)}')
    source_paths = [E62 / 'results/e62_vectors.parquet', E65 / 'results/e65_scope_vectors.parquet', E65 / 'results/e65_predictions.parquet', ANALYSIS / 'Family_Information_Recovery_E66/code/e66_common.py', BASE / 'metadata/e66_fit_metadata.json', BASE / 'metadata/e66_plan_metadata.json']
    before = {str(p.relative_to(ANALYSIS)): sha(p) for p in [*source_paths, *paths]}
    v = pd.read_parquet(source_paths[0]).set_index('vector_id', drop=False)
    scope = pd.read_parquet(source_paths[1])
    old = pd.read_parquet(source_paths[2], columns=[*EDGE, 'vector_id', 'regime', 'target_geometry', 'prediction_vector'])
    old = old.loc[old.regime.eq('family')].pivot(index='vector_id', columns='target_geometry', values='prediction_vector')
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        if not passed:
            raise ValueError(name + ': ' + str(detail))
    check('source_scope', len(v) == 25956 and len(scope) == 25956)
    check('common8_scope', int(population_mask(scope, 'common8').sum()) == 24221)
    check('common16_scope', int(population_mask(scope, 'common16').sum()) == 22963)
    stats, pair_stats, allocation_stats, pair_allocation_stats = ([], [], [], [])
    support = pd.read_csv(E62 / 'results/e62_support_bridge_1548.csv')
    for j, path in enumerate(paths, 1):
        p = pd.read_parquet(path)
        check(path.stem + '_keys', not p.duplicated(['vector_id', 'allocation', 'budget', 'learner']).any())
        ids = p.vector_id.drop_duplicates()
        key = p[EDGE].iloc[0].to_dict()
        s = scope.loc[scope.vector_id.isin(ids)]
        check(path.stem + '_complete_rows', len(p) == len(ids) * 3 * 7 * 3)
        reference = []
        for vid in ids:
            dimension = int(v.loc[vid, 'n_backgrounds'])
            for allocation in ['r0', 'r1', 'r2']:
                for budget in BUDGETS:
                    common = dict(**key, vector_id=int(vid), allocation=allocation, budget=budget, n_backgrounds=dimension)
                    for model in ['frozen_family', 'zero']:
                        exists = model == 'zero' or vid in old.index
                        arrays = [np.zeros(dimension).tolist()] * 3 if model == 'zero' else [old.loc[vid, g] for g in GEOMETRIES] if exists else [None] * 3
                        reference.append(dict(**common, learner=model, supported=exists, **dict(zip(['prediction_raw', 'prediction_centered', 'prediction_unit'], arrays))))
        use = ['vector_id', *EDGE, 'allocation', 'budget', 'learner', 'supported', 'n_backgrounds', 'prediction_raw', 'prediction_centered', 'prediction_unit']
        combined = pd.concat([p[use], pd.DataFrame(reference)[use]], ignore_index=True)
        losses = score_arrays(combined, v)
        check(path.stem + '_unsupported_NA', losses.loc[~losses.supported, METRICS].isna().all().all())
        check(path.stem + '_MSE_decomposition', np.allclose(losses.native_vector_mse, losses.mean_offset_mse + losses.centered_error_mse, atol=1e-08, rtol=0, equal_nan=True))
        group = [*EDGE, 'vector_id', 'budget', 'learner', 'target_geometry', 'true_constant_vector', 'supported']
        averaged = losses.groupby(group, sort=False, observed=True)[METRICS].mean().reset_index()
        check(path.stem + '_three_allocations', losses.groupby(group, sort=False, observed=True).size().eq(3).all())
        destination = BASE / 'parts' / path.name.replace('_predictions', '_mean_losses')
        averaged.to_parquet(destination, index=False)
        stats.append(sufficient_stats(averaged, s, ['target_geometry', 'budget', 'learner'], POPULATIONS + DIAGNOSTIC_POPULATIONS))
        allocation_stats.append(sufficient_stats(losses, s, ['target_geometry', 'budget', 'learner', 'allocation'], ['common8']))
        pairs = []
        order_pairs = []
        for mean, sink in [(averaged, pairs), (losses, order_pairs)]:
            index = [*EDGE, 'vector_id', 'budget', 'target_geometry', 'true_constant_vector']
            if 'allocation' in mean:
                index += ['allocation']
            wide = mean.pivot(index=index, columns='learner', values=METRICS)
            for comparison in COMPARISONS:
                other = comparison.removeprefix('joint_minus_')
                out = wide.index.to_frame(index=False)
                for metric in METRICS:
                    out[metric] = (wide[metric, 'joint'] - wide[metric, other]).to_numpy()
                out['comparison'] = comparison
                out['supported'] = out.native_vector_mse.notna()
                sink.append(out)
        pair_stats.append(sufficient_stats(pd.concat(pairs, ignore_index=True), s, ['target_geometry', 'budget', 'comparison'], POPULATIONS + DIAGNOSTIC_POPULATIONS))
        pair_allocation_stats.append(sufficient_stats(pd.concat(order_pairs, ignore_index=True), s, ['target_geometry', 'budget', 'comparison', 'allocation'], ['common8']))
        print(f'E66 aggregation {j}/92: {path.stem}', flush=True)
    stat = pd.concat(stats, ignore_index=True)
    pair = pd.concat(pair_stats, ignore_index=True)
    stat.to_parquet(BASE / 'results/e66_sufficient_statistics.parquet', index=False)
    pair.to_parquet(BASE / 'results/e66_paired_sufficient_statistics.parquet', index=False)
    for prefix, data, modelkey, models in [('e66', stat, 'learner', LEARNERS), ('e66_paired', pair, 'comparison', COMPARISONS)]:
        summary, contribution = summarize(data, ['target_geometry', 'budget', modelkey], scope)
        summary.to_csv(BASE / f'results/{prefix}_summary.csv', index=False)
        contribution.to_csv(BASE / f'results/{prefix}_full_weight_contributions.csv', index=False)
        edge, classes = complete_grid(data, support, POPULATIONS, modelkey, models)
        edge.to_parquet(BASE / f'results/{prefix}_edge_metrics_1548.parquet', index=False)
        classes.to_csv(BASE / f'results/{prefix}_class_metrics_20.csv', index=False)
        check(prefix + '_complete_NA_grid', len(edge) == 1550 * 6 * 3 * 7 * len(models) and len(classes) == 20 * 6 * 3 * 7 * len(models) and edge.loc[~edge.supported, METRICS].isna().all().all() and classes.loc[~classes.supported, METRICS].isna().all().all())
        keys = ['population', 'target_geometry', 'budget', modelkey, 'weighting']
        a = summary.loc[summary.stratum.eq('all')].set_index(keys)[METRICS].sort_index()
        b = contribution.groupby(keys)[METRICS].sum(min_count=1).sort_index()
        check(prefix + '_parent_contributions_close', np.allclose(a, b, atol=1e-08, rtol=0, equal_nan=True))
        if prefix == 'e66':
            raw = summary.loc[summary.target_geometry.eq('raw_delta')]
            check('summary_raw_decomposition', np.allclose(raw.native_vector_mse, raw.mean_offset_mse + raw.centered_error_mse, atol=1e-08, rtol=0, equal_nan=True))
    for name, data, modelkey in [('e66_allocation_sensitivity', pd.concat(allocation_stats, ignore_index=True), 'learner'), ('e66_paired_allocation_sensitivity', pd.concat(pair_allocation_stats, ignore_index=True), 'comparison')]:
        summary, _ = summarize(data, ['target_geometry', 'budget', modelkey, 'allocation'], scope)
        summary.to_csv(BASE / f'results/{name}.csv', index=False)
    scope_cost = scope.copy()
    scope_cost['n_calibration_pool'] = scope.n_original_train - scope.n_family_train
    observed = []
    for population in POPULATIONS + DIAGNOSTIC_POPULATIONS:
        count = scope.loc[population_mask(scope, population)].groupby(EDGE).size().rename('n_' + population)
        observed.append(count)
    counts = pd.concat(observed, axis=1).reset_index()
    bridge = support.merge(counts, on=EDGE, how='left', validate='one_to_one')
    cols = ['n_' + x for x in POPULATIONS + DIAGNOSTIC_POPULATIONS]
    bridge[cols] = bridge[cols].fillna(0).astype(int)
    bridge.to_csv(BASE / 'results/e66_support_bridge_1548.csv', index=False)
    cls = bridge.groupby('reaction_type', as_index=False).agg(n_registered_edges=('component_pair', 'size'), **{x: (x, 'sum') for x in cols})
    cls.to_csv(BASE / 'results/e66_support_classes_20.csv', index=False)
    check('all20classes1548edges', len(cls) == 20 and len(bridge) == 1550)
    check('sources_unchanged', before == {str(p.relative_to(ANALYSIS)): sha(p) for p in [*source_paths, *paths]})
    pd.DataFrame(checks).to_csv(BASE / 'audit/e66_aggregation_checks.csv', index=False)
    outputs = sorted((BASE / 'results').glob('e66_*')) + sorted((BASE / 'parts').glob('*_mean_losses.parquet'))
    meta = dict(schema_version='E66_aggregation_v1', source_sha256=before, code_sha256=sha(Path(__file__)), output_sha256={str(p.relative_to(BASE)): sha(p) for p in outputs}, checks=checks, failures=0, seconds=time.perf_counter() - started, units={'raw_delta': 'pp2', 'centered_delta': 'pp2', 'centered_unit_rms_zeroembed': 'dimensionless'}, allocation_combination='Arithmetic mean of three losses for each vector, never mean predictions', support='Risks renormalized on declared learner support; primary common8 is fixed and complete for positive budgets')
    (BASE / 'metadata/e66_aggregation_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(dict(checks=len(checks), failures=0, seconds=meta['seconds'])), flush=True)
if __name__ == '__main__':
    main()
