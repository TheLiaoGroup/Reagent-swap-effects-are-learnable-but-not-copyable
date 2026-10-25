"""E65 fixed-population aggregation. No kernel operations or fitting occur here."""
from __future__ import annotations
import pandas as pd
from e65_common import *
POPULATIONS = ['all_original', 'primary', 'sensitivity_n5plus', 'n1', 'n2to4', 'n0']
STRATA = ['all', 'true_nonzero', 'true_constant']
MODELS = ['uncompressed_krr', 'centroid', 'zero']
SUMMARY_REGIMES = REGIMES + ['mean_controls']
WEIGHTS = ['pooled', 'equal_edge', 'equal_class_edge_balanced']
COMPARISONS = ['family_minus_original', 'mean_controls_minus_original', 'family_minus_mean_controls']

def population_ids(scope, population):
    mask = np.ones(len(scope), bool) if population == 'all_original' else scope.primary_supported if population == 'primary' else scope.sensitivity_n5plus if population == 'sensitivity_n5plus' else scope.training_size_stratum.eq(population)
    return set(map(int, scope.loc[mask, 'vector_id']))

def strata(g):
    yield ('all', g)
    yield ('true_nonzero', g.loc[~g.true_constant_vector])
    yield ('true_constant', g.loc[g.true_constant_vector])

def weights(g):
    edges = g[EDGE].drop_duplicates()
    n = g.groupby(EDGE).vector_id.transform('size').to_numpy(float)
    ce = edges.groupby('reaction_type').size()
    classes = edges.reaction_type.nunique()
    return dict(pooled=np.full(len(g), 1 / len(g)), equal_edge=1 / (len(edges) * n), equal_class_edge_balanced=1 / (classes * g.reaction_type.map(ce).to_numpy() * n))

def support_tables(scope):
    support = pd.read_csv(INPUTS[4])
    classes = pd.read_csv(INPUTS[5])
    counts = []
    for population in POPULATIONS:
        ids = population_ids(scope, population)
        q = scope.loc[scope.vector_id.isin(ids)].groupby(EDGE).size().rename('n_' + ('original' if population == 'all_original' else population) + '_vectors')
        counts.append(q)
    observed = pd.concat(counts, axis=1).reset_index()
    support = support.merge(observed, on=EDGE, how='left', validate='one_to_one')
    columns = [x for x in observed.columns if x not in EDGE]
    support[columns] = support[columns].fillna(0).astype(int)
    c = support.groupby('reaction_type', as_index=False)[columns].sum()
    c['n_primary_edges'] = support.groupby('reaction_type').n_primary_vectors.apply(lambda x: int(x.gt(0).sum())).reindex(c.reaction_type).to_numpy()
    classes = classes.merge(c, on='reaction_type', how='left', validate='one_to_one')
    support.to_csv(BASE / 'results/e65_support_bridge_1548.csv', index=False)
    classes.to_csv(BASE / 'results/e65_support_classes_20.csv', index=False)
    return (support, classes)

def aggregate_rows(g, metrics, extra):
    summaries = []
    contributions = []
    for label, part in strata(g):
        for name in WEIGHTS:
            row = dict(**extra, stratum=label, weighting=name, n_test_vectors=len(part), n_test_edges=len(part[EDGE].drop_duplicates()), n_test_classes=part.reaction_type.nunique(), weight_mass=1.0 if len(part) else 0.0)
            if len(part):
                w = weights(part)[name]
                row.update({m: float(w @ part[m].to_numpy()) for m in metrics})
            else:
                row.update({m: np.nan for m in metrics})
            if 'null_indicator' in part:
                row.update(n_true_constant=int(part.true_constant_vector.sum()), n_predicted_null=float(part.null_indicator.sum()))
            summaries.append(row)
    for name in WEIGHTS:
        w = weights(g)[name] if len(g) else np.empty(0)
        for null, label in [(False, 'true_nonzero'), (True, 'true_constant')]:
            mask = g.true_constant_vector.to_numpy() == null
            contributions.append(dict(**extra, stratum=label, weighting=name, n_test_vectors=int(mask.sum()), full_sample_weight_mass=float(w[mask].sum()), **{m: float(w[mask] @ g.loc[mask, m].to_numpy()) if len(g) else np.nan for m in metrics}))
    return (summaries, contributions)

def complete_edges(observed, support, keys, combos, metrics, counts):
    frame = support[[*EDGE, 'panel_status']].merge(combos, how='cross').merge(observed, on=[*EDGE, *keys], how='left', validate='one_to_one')
    for c in counts:
        frame[c] = frame[c].fillna(0)
    frame['n_test_vectors'] = frame.n_test_vectors.astype(int)
    frame['supported'] = frame.n_test_vectors.gt(0)
    if not frame.loc[~frame.supported, metrics].isna().all().all():
        raise ValueError('Unsupported risk must remain NA')
    return frame

def verify_arrays(predictions, baselines, diagnostics, scope, check):
    vectors = pd.read_parquet(INPUTS[0])
    targetmap = vectors.set_index('vector_id').target_vector.to_dict()
    means = diagnostics.set_index('factor_id').train_mean_block.to_dict()
    maximum_loss = maximum_target = maximum_decoder = maximum_decomposition = maximum_norm = maximum_identity = maximum_baseline = 0.0
    for (B, geometry), g in predictions.groupby(['n_backgrounds', 'target_geometry'], sort=False):
        raw = np.vstack([targetmap[int(v)] for v in g.vector_id])
        c, u, null, rms = center_and_unit(raw)
        expected_target = dict(zip(GEOMETRIES, [raw, c, u]))[geometry]
        target = np.vstack(g.target_vector)
        pred = np.vstack(g.prediction_vector)
        decoded = np.vstack(g.unit0_prediction_vector)
        pc, expected_decoded, pn, pr = center_and_unit(pred)
        tc = center_and_unit(target)[0]
        error = pred - target
        expected = dict(native_vector_mse=np.mean(error * error, 1), native_vector_mae=np.mean(np.abs(error), 1), unit0_vector_mse=np.mean((decoded - u) ** 2, 1), unit0_vector_mae=np.mean(np.abs(decoded - u), 1), mean_offset_mse=np.mean(error, 1) ** 2, centered_error_mse=np.mean((pc - tc) ** 2, 1))
        maximum_target = max(maximum_target, float(np.max(np.abs(target - expected_target))))
        maximum_decoder = max(maximum_decoder, float(np.max(np.abs(decoded - expected_decoded))))
        maximum_loss = max(maximum_loss, max((float(np.max(np.abs(g[m].to_numpy() - expected[m]))) for m in METRICS)))
        maximum_decomposition = max(maximum_decomposition, float(np.max(np.abs(g.native_vector_mse - g.mean_offset_mse - g.centered_error_mse))))
        maximum_norm = max(maximum_norm, float(np.max(np.abs(np.mean(decoded * decoded, 1) - ~pn))))
        masks = null | pn
        identity = np.where(null, (~pn).astype(float), 1.0)
        maximum_identity = max(maximum_identity, float(np.max(np.abs(g.unit0_vector_mse.to_numpy()[masks] - identity[masks]))) if masks.any() else 0.0)
        if not np.array_equal(pn, g.unit0_predicted_null) or not np.array_equal(null, g.true_constant_vector):
            raise ValueError('Null flags disagree')
        gi = GEOMETRIES.index(geometry)
        bg = baselines.loc[baselines.n_backgrounds.eq(B) & baselines.target_geometry.eq(geometry)]
        for model, part in bg.groupby('model', sort=False):
            rawb = np.vstack([targetmap[int(v)] for v in part.vector_id])
            cb, ub, nb, rb = center_and_unit(rawb)
            tb = dict(zip(GEOMETRIES, [rawb, cb, ub]))[geometry]
            bp = np.zeros_like(tb) if model == 'zero' else np.vstack([means[f][gi * B:(gi + 1) * B] for f in part.factor_id])
            bc, bd, bn, br = center_and_unit(bp)
            errorb = bp - tb
            tbc = center_and_unit(tb)[0]
            ex = dict(native_vector_mse=np.mean(errorb * errorb, 1), native_vector_mae=np.mean(np.abs(errorb), 1), unit0_vector_mse=np.mean((bd - ub) ** 2, 1), unit0_vector_mae=np.mean(np.abs(bd - ub), 1), mean_offset_mse=np.mean(errorb, 1) ** 2, centered_error_mse=np.mean((bc - tbc) ** 2, 1))
            maximum_baseline = max(maximum_baseline, max((float(np.max(np.abs(part[m].to_numpy() - ex[m]))) for m in METRICS)))
            if not np.array_equal(part.unit0_predicted_null, bn):
                raise ValueError('Baseline null flags disagree')
    check('all_saved_target_arrays_reconstructed', maximum_target < 1e-12, maximum_target)
    check('all_saved_common_decoder_arrays_reconstructed', maximum_decoder < 1e-12, maximum_decoder)
    check('all_saved_six_prediction_metrics_reconstructed', maximum_loss < 1e-08, maximum_loss)
    check('all_saved_baseline_metrics_reconstructed', maximum_baseline < 1e-08, maximum_baseline)
    check('native_MSE_mean_plus_centered_identity', maximum_decomposition < 1e-08, maximum_decomposition)
    check('unit0_norm_null_or_one', maximum_norm < 1e-12, maximum_norm)
    check('unit0_null_loss_identities', maximum_identity < 1e-12, maximum_identity)
    old = pd.read_parquet(INPUTS[8], columns=['vector_id', 'geometry', 'nominal_k', 'model', 'prediction_vector'])
    old = old.loc[old.nominal_k.eq(2) & old.model.eq('uncompressed_krr')].rename(columns={'geometry': 'target_geometry', 'prediction_vector': 'old_prediction_vector'})
    original = predictions.loc[predictions.regime.eq('original') & predictions.target_geometry.ne('raw_delta')].merge(old[['vector_id', 'target_geometry', 'old_prediction_vector']], on=['vector_id', 'target_geometry'], validate='one_to_one')
    agreement = max((float(np.max(np.abs(np.vstack(g.prediction_vector) - np.vstack(g.old_prediction_vector)))) for _, g in original.groupby('n_backgrounds')))
    check('all_original_centered_unit_agree_E64', len(original) == 51912 and agreement < 1e-08, agreement)
    check('fixed_linear_centering_commutes_all_factors', diagnostics.maximum_centering_commute_error.max() < 1e-08, float(diagnostics.maximum_centering_commute_error.max()))
    check('all_actual_singleton_predictions_equal_centroid', diagnostics.loc[diagnostics.n_train.eq(1), 'maximum_singleton_centroid_error'].max() < 1e-12)

def run():
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
    p = pd.read_parquet(BASE / 'results/e65_predictions.parquet')
    b = pd.read_parquet(BASE / 'results/e65_baseline_metrics.parquet')
    d = pd.read_parquet(BASE / 'results/e65_factor_diagnostics.parquet')
    scope = pd.read_parquet(BASE / 'results/e65_scope_vectors.parquet')
    check('all516606_full_prediction_keys', len(p) == 537228 and (not p.duplicated(['vector_id', 'regime', 'target_geometry']).any()))
    check('all1033212_baseline_keys', len(b) == 1074456 and (not b.duplicated(['vector_id', 'regime', 'target_geometry', 'model']).any()))
    check('all71328_logical_factors', len(d) == 72928 and d.factor_id.nunique() == 72928)
    check('only_original_learned_predictions_on_n0', p.loc[p.training_size_stratum.eq('n0'), 'regime'].eq('original').all() and len(p.loc[p.training_size_stratum.eq('n0')]) == 436 * 3)
    check('finite_nonnegative_prediction_and_baseline_losses', np.isfinite(p[METRICS]).all().all() and np.isfinite(b[METRICS]).all().all() and p[METRICS].ge(0).all().all() and b[METRICS].ge(0).all().all())
    check('family_crossrole_scaffold_absence', p.loc[p.regime.eq('family'), 'n_held_families_present_either_role'].eq(0).all())
    verify_arrays(p, b, d, scope, check)
    null = p[['vector_id', 'true_constant_vector']].drop_duplicates()
    scope = scope.merge(null, on='vector_id', validate='one_to_one')
    scope.to_parquet(BASE / 'results/e65_scope_targets.parquet', index=False)
    support, classes = support_tables(scope)
    keys = [*EDGE, 'vector_id', 'target_geometry', 'model', 'regime', 'true_constant_vector']
    metrics = pd.concat([p[[*keys, *METRICS, 'unit0_predicted_null']], b[[*keys, *METRICS, 'unit0_predicted_null']]], ignore_index=True)
    metrics['null_indicator'] = metrics.unit0_predicted_null.astype(float)
    metrics = metrics.drop(columns='unit0_predicted_null')
    index = [*EDGE, 'vector_id', 'target_geometry', 'model', 'true_constant_vector']
    controls = metrics.loc[metrics.regime.str.startswith('thin_')]
    check('allfive_controls_same_population', controls.groupby(['vector_id', 'target_geometry', 'model']).size().eq(5).all() and len(controls) == 25520 * 3 * 3 * 5)
    mean = controls.groupby(index, as_index=False)[[*METRICS, 'null_indicator']].mean()
    mean['regime'] = 'mean_controls'
    metrics = pd.concat([metrics, mean], ignore_index=True)
    edgekeys = ['population', 'target_geometry', 'regime', 'model', 'stratum']
    edge_rows = []
    comborows = []
    summaries = []
    contributions = []
    for population in POPULATIONS:
        ids = population_ids(scope, population)
        data = metrics.loc[metrics.vector_id.isin(ids)]
        regimes = ['original'] if population == 'all_original' else SUMMARY_REGIMES
        for geometry in GEOMETRIES:
            for regime in regimes:
                for model in MODELS:
                    g = data.loc[data.target_geometry.eq(geometry) & data.regime.eq(regime) & data.model.eq(model)]
                    extra = dict(population=population, target_geometry=geometry, regime=regime, model=model)
                    a, c = aggregate_rows(g, METRICS, extra)
                    summaries.extend(a)
                    contributions.extend(c)
                    for label, part in strata(g):
                        comborows.append(dict(**extra, stratum=label))
                        q = part.groupby(EDGE, as_index=False).agg(n_test_vectors=('vector_id', 'size'), n_true_constant=('true_constant_vector', 'sum'), n_predicted_null=('null_indicator', 'sum'), **{m: (m, 'mean') for m in METRICS})
                        for k, v in dict(**extra, stratum=label).items():
                            q[k] = v
                        edge_rows.append(q)
    e = complete_edges(pd.concat(edge_rows, ignore_index=True), support, edgekeys, pd.DataFrame(comborows), METRICS, ['n_test_vectors', 'n_true_constant', 'n_predicted_null'])
    e['native_rmse'] = np.sqrt(e.native_vector_mse)
    e['unit0_rmse'] = np.sqrt(e.unit0_vector_mse)
    e.to_csv(BASE / 'results/e65_edge_metrics_1548.csv', index=False)
    c = e.groupby(['reaction_type', *edgekeys], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), n_true_constant=('n_true_constant', 'sum'), n_predicted_null=('n_predicted_null', 'sum'), **{m: (m, 'mean') for m in METRICS})
    c['supported'] = c.n_test_edges.gt(0)
    c['native_rmse'] = np.sqrt(c.native_vector_mse)
    c['unit0_rmse'] = np.sqrt(c.unit0_vector_mse)
    c.to_csv(BASE / 'results/e65_class_metrics_20.csv', index=False)
    summary = pd.DataFrame(summaries)
    contrib = pd.DataFrame(contributions)
    summary['native_rmse'] = np.sqrt(summary.native_vector_mse)
    summary['unit0_rmse'] = np.sqrt(summary.unit0_vector_mse)
    summary.to_csv(BASE / 'results/e65_summary.csv', index=False)
    contrib.to_csv(BASE / 'results/e65_full_weight_stratum_contributions.csv', index=False)
    check('complete20class1548edge_metric_grids', len(e) == 1550 * len(comborows) and len(c) == 20 * len(comborows) and e.loc[~e.supported, METRICS].isna().all().all() and c.loc[~c.supported, METRICS].isna().all().all())
    skeys = ['population', 'target_geometry', 'regime', 'model', 'weighting']
    total = summary.loc[summary.stratum.eq('all')].set_index(skeys)[METRICS].sort_index()
    recovered = contrib.groupby(skeys)[METRICS].sum(min_count=1).sort_index()
    check('all_model_strata_contributions_close', np.allclose(total, recovered, atol=1e-08, rtol=0, equal_nan=True))
    full = metrics.loc[metrics.model.eq('uncompressed_krr') & metrics.vector_id.isin(population_ids(scope, 'primary'))]
    wide = full.pivot(index=[*EDGE, 'vector_id', 'target_geometry', 'true_constant_vector'], columns='regime', values=METRICS)
    pv = wide.index.to_frame(index=False)
    for metric in METRICS:
        for regime in SUMMARY_REGIMES:
            pv[metric + '__' + regime] = wide[metric, regime].to_numpy()
        pv[metric + '__family_minus_original'] = pv[metric + '__family'] - pv[metric + '__original']
        pv[metric + '__mean_controls_minus_original'] = pv[metric + '__mean_controls'] - pv[metric + '__original']
        pv[metric + '__family_minus_mean_controls'] = pv[metric + '__family'] - pv[metric + '__mean_controls']
    pv.to_parquet(BASE / 'results/e65_paired_vector_losses.parquet', index=False)
    pair_summaries = []
    pair_contrib = []
    pair_edge = []
    pair_combos = []
    diffmetrics = ['difference_' + m for m in METRICS]
    for population in ['primary', 'sensitivity_n5plus', 'n1', 'n2to4', 'n0']:
        ids = population_ids(scope, population)
        pop = pv.loc[pv.vector_id.isin(ids)]
        for geometry in GEOMETRIES:
            for comparison in COMPARISONS:
                g = pop.loc[pop.target_geometry.eq(geometry)].copy()
                for metric, out in zip(METRICS, diffmetrics):
                    g[out] = g[metric + '__' + comparison]
                extra = dict(population=population, target_geometry=geometry, comparison=comparison, model='uncompressed_krr')
                a, co = aggregate_rows(g, diffmetrics, extra)
                pair_summaries.extend(a)
                pair_contrib.extend(co)
                for label, part in strata(g):
                    pair_combos.append(dict(**extra, stratum=label))
                    q = part.groupby(EDGE, as_index=False).agg(n_test_vectors=('vector_id', 'size'), **{m: (m, 'mean') for m in diffmetrics})
                    for k, v in dict(**extra, stratum=label).items():
                        q[k] = v
                    pair_edge.append(q)
    pairkeys = ['population', 'target_geometry', 'comparison', 'model', 'stratum']
    pe = complete_edges(pd.concat(pair_edge, ignore_index=True), support, pairkeys, pd.DataFrame(pair_combos), diffmetrics, ['n_test_vectors'])
    pe.to_csv(BASE / 'results/e65_paired_edge_metrics_1548.csv', index=False)
    pc = pe.groupby(['reaction_type', *pairkeys], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), **{m: (m, 'mean') for m in diffmetrics})
    pc['supported'] = pc.n_test_edges.gt(0)
    pc.to_csv(BASE / 'results/e65_paired_class_metrics_20.csv', index=False)
    ps = pd.DataFrame(pair_summaries)
    pco = pd.DataFrame(pair_contrib)
    ps.to_csv(BASE / 'results/e65_paired_summary.csv', index=False)
    pco.to_csv(BASE / 'results/e65_paired_full_weight_stratum_contributions.csv', index=False)
    pkeys = ['population', 'target_geometry', 'comparison', 'model', 'weighting']
    check('all_paired_strata_contributions_close', np.allclose(ps.loc[ps.stratum.eq('all')].set_index(pkeys)[diffmetrics].sort_index(), pco.groupby(pkeys)[diffmetrics].sum(min_count=1).sort_index(), atol=1e-08, rtol=0, equal_nan=True))
    check('paired_vector_support_and_loss_averaging', len(pv) == 25520 * 3 and all((np.allclose(pv[m + '__mean_controls'], pv[[m + f'__thin_{r}' for r in range(5)]].mean(axis=1), atol=1e-12, rtol=1e-15) for m in METRICS)))
    check('complete_paired20class1548edge_NA_grids', len(pe) == 1550 * len(pair_combos) and len(pc) == 20 * len(pair_combos) and pe.loc[~pe.supported, diffmetrics].isna().all().all())
    spread = []
    for population in ['primary', 'sensitivity_n5plus', 'n1', 'n2to4']:
        ids = population_ids(scope, population)
        for geometry, g in pv.loc[pv.vector_id.isin(ids)].groupby('target_geometry'):
            for label, part in strata(g):
                if not len(part):
                    continue
                for name, w in weights(part).items():
                    row = dict(population=population, target_geometry=geometry, stratum=label, weighting=name, n_test_vectors=len(part))
                    for m in METRICS:
                        values = part[[m + f'__thin_{r}' for r in range(5)]].to_numpy()
                        replicate = w @ values
                        row[m + '__control_risk_min'] = float(replicate.min())
                        row[m + '__control_risk_max'] = float(replicate.max())
                        row[m + '__control_risk_sd_population'] = float(replicate.std(ddof=0))
                        row[m + '__mean_within_vector_control_sd'] = float(w @ values.std(axis=1, ddof=0))
                    spread.append(row)
    pd.DataFrame(spread).to_csv(BASE / 'results/e65_control_spread.csv', index=False)
    structural = []
    raw = p.loc[p.target_geometry.eq('raw_delta')]
    for population in POPULATIONS:
        ids = population_ids(scope, population)
        for regime, g in raw.loc[raw.vector_id.isin(ids)].groupby('regime'):
            if population == 'all_original' and regime != 'original':
                continue
            for role in [1, 2]:
                values = g[f'nearest_training_tanimoto_role{role}']
                q = values.quantile([0, 0.1, 0.5, 0.9, 1])
                structural.append(dict(population=population, regime=regime, ordered_role=role, n_test_vectors=len(g), nearest_tanimoto_min=q.loc[0], nearest_tanimoto_q10=q.loc[0.1], nearest_tanimoto_median=q.loc[0.5], nearest_tanimoto_q90=q.loc[0.9], nearest_tanimoto_max=q.loc[1], nearest_tanimoto_mean=values.mean(), mean_n_train=g.n_train.mean(), min_n_train=g.n_train.min(), max_n_train=g.n_train.max(), mean_n_training_identities_role=g[f'n_training_identities_role{role}'].mean(), mean_n_training_families_role=g[f'n_training_families_role{role}'].mean(), fraction_no_held_family_in_training=g.n_held_families_present_either_role.eq(0).mean()))
    pd.DataFrame(structural).to_csv(BASE / 'results/e65_structural_diagnostics.csv', index=False)
    expected = dict(all_original=25956, primary=25520, sensitivity_n5plus=24970, n1=114, n2to4=436, n0=436)
    check('frozen_population_counts', all((len(population_ids(scope, k)) == v for k, v in expected.items())))
    check('all_paired_population_denominators', all((ps.loc[ps.population.eq(k) & ps.stratum.eq('all'), 'n_test_vectors'].eq(v if k != 'n0' else 0).all() for k, v in expected.items() if k != 'all_original')))
    zero = summary.loc[summary.model.eq('zero') & summary.target_geometry.eq(GEOMETRIES[2])]
    mass_error = 0.0
    for population in POPULATIONS:
        pop = scope.loc[scope.vector_id.isin(population_ids(scope, population))]
        for label, part in strata(pop):
            if not len(part):
                continue
            for name, w in weights(part).items():
                expected_mass = float(w @ (~part.true_constant_vector).to_numpy(float))
                observed = zero.loc[zero.population.eq(population) & zero.stratum.eq(label) & zero.weighting.eq(name), 'native_vector_mse'].dropna()
                if len(observed):
                    mass_error = max(mass_error, float(np.max(np.abs(observed - expected_mass))))
    check('zero_baseline_unit_population_mass', mass_error < 1e-12 and all((np.allclose(g.native_vector_mse, g.unit0_vector_mse, atol=1e-12, rtol=0, equal_nan=True) for _, g in zero.groupby('population'))), mass_error)
    pd.DataFrame(checks).to_csv(BASE / 'audit/e65_production_checks.csv', index=False)
    if not all((x['passed'] for x in checks)):
        raise ValueError('Production checks failed: ' + str([x for x in checks if not x['passed']]))
if __name__ == '__main__':
    run()
