"""E64 aggregation and fixed-support consistency checks; never fits models."""
from __future__ import annotations
import numpy as np
import pandas as pd
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
GROUP = ['geometry', 'nominal_k', 'model']
METRICS = ['native_vector_mse', 'native_vector_mae', 'unit0_vector_mse', 'unit0_vector_mae']
FIVE = ['centroid', 'codebook_continuous', 'codebook_hard', 'codebook_mixture', 'pca_continuous']
MODELS = FIVE + ['uncompressed_krr', 'zero'] + ['raw_postmapped__' + m for m in FIVE]
GEOMETRIES = ['centered_delta', 'centered_unit_rms_zeroembed']
STRATA = ['all', 'true_nonzero', 'true_constant']

def weights(g):
    edges = g[EDGE].drop_duplicates()
    sizes = g.groupby(EDGE).vector_id.transform('size').to_numpy(float)
    classedges = edges.groupby('reaction_type').size()
    return dict(pooled=np.full(len(g), 1 / len(g)), equal_edge=1 / (len(edges) * sizes), equal_class_edge_balanced=1 / (edges.reaction_type.nunique() * g.reaction_type.map(classedges).to_numpy() * sizes))

def stratum_rows(g):
    yield ('all', g)
    yield ('true_nonzero', g.loc[~g.true_constant_vector])
    yield ('true_constant', g.loc[g.true_constant_vector])

def summary_rows(g, metrics, extra):
    for s, part in stratum_rows(g):
        if not len(part):
            continue
        for name, w in weights(part).items():
            yield dict(**extra, stratum=s, weighting=name, n_test_vectors=len(part), n_test_edges=len(part[EDGE].drop_duplicates()), n_test_classes=part.reaction_type.nunique(), weight_mass=float(w.sum()), **{c: float(w @ part[c].to_numpy()) for c in metrics})

def contributions(g, metrics, extra):
    for name, w in weights(g).items():
        for null, label in [(False, 'true_nonzero'), (True, 'true_constant')]:
            mask = g.true_constant_vector.to_numpy() == null
            yield dict(**extra, stratum=label, weighting=name, n_test_vectors=int(mask.sum()), full_sample_weight_mass=float(w[mask].sum()), **{c: float(w[mask] @ g.loc[mask, c].to_numpy()) for c in metrics})

def complete_grid(observed, support, combos, keys, metrics, countcols):
    out = support[[*EDGE, 'panel_status']].merge(combos, how='cross').merge(observed, on=[*EDGE, *keys], how='left', validate='one_to_one')
    out[countcols] = out[countcols].fillna(0).astype(int)
    out['supported'] = out.n_test_vectors.gt(0)
    if not out.loc[~out.supported, metrics].isna().all().all():
        raise ValueError('Unsupported metrics must remain NA')
    return out

def run(base, inputs):
    p = pd.read_parquet(base / 'results/e64_predictions.parquet')
    support = pd.read_csv(inputs[5])
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))

    def save(frame, name):
        frame.to_csv(base / 'results' / name, index=False)
    check('all_expected_keys_and_support', len(p) == 1245888 and p.vector_id.nunique() == 25956 and p.groupby('vector_id').size().eq(48).all() and (not p.duplicated(['vector_id', *GROUP]).any()), f'{len(p)} prediction rows')
    check('all_native_and_decoder_losses_finite_nonnegative', np.isfinite(p[METRICS]).all().all() and p[METRICS].ge(0).all().all())
    check('all48families_have_same_fixed_population', p.groupby(GROUP).size().eq(25956).all() and p.groupby(GROUP).true_constant_vector.sum().eq(9924).all())
    maximum_loss_error = 0.0
    maximum_decoder_norm_error = 0.0
    decoder_identity_error = 0.0
    for b, g in p.groupby('n_backgrounds', sort=True):
        target = np.vstack(g.target_vector)
        prediction = np.vstack(g.prediction_vector)
        decoded = np.vstack(g.unit0_prediction_vector)
        cm = target[:, 0] + (target - target[:, :1]).mean(axis=1)
        ct = target - cm[:, None]
        rms = np.sqrt(np.mean(ct * ct, axis=1))
        unit = np.zeros_like(ct)
        np.divide(ct, rms[:, None], out=unit, where=~g.true_constant_vector.to_numpy()[:, None])
        expected = [np.mean((prediction - target) ** 2, axis=1), np.mean(np.abs(prediction - target), axis=1), np.mean((decoded - unit) ** 2, axis=1), np.mean(np.abs(decoded - unit), axis=1)]
        maximum_loss_error = max(maximum_loss_error, max((float(np.max(np.abs(g[c].to_numpy() - v))) for c, v in zip(METRICS, expected))))
        expected_norm = (~g.unit0_predicted_null).to_numpy(float)
        maximum_decoder_norm_error = max(maximum_decoder_norm_error, float(np.max(np.abs(np.mean(decoded * decoded, axis=1) - expected_norm))))
        null = g.true_constant_vector.to_numpy()
        pn = g.unit0_predicted_null.to_numpy()
        masks = null | pn
        identity = np.where(null, ~pn, np.ones(len(g), bool)).astype(float)
        decoder_identity_error = max(decoder_identity_error, float(np.max(np.abs(g.unit0_vector_mse.to_numpy()[masks] - identity[masks]))))
    check('all_saved_native_and_decoder_losses_match_arrays', maximum_loss_error < 1e-08, f'max difference {maximum_loss_error:.17g}')
    check('unit0_decoded_RMS_is_exact_null_or_one', maximum_decoder_norm_error < 1e-12, f'max difference {maximum_decoder_norm_error:.17g}')
    check('unit0_null_embedding_loss_identities', decoder_identity_error < 1e-12, f'max difference {decoder_identity_error:.17g}')
    z = p.loc[p.geometry.eq(GEOMETRIES[1]) & p.model.eq('zero')]
    check('native_unit_zero_baseline_one_or_zero', np.allclose(z.native_vector_mse, (~z.true_constant_vector).astype(float), atol=1e-12, rtol=0))
    zd = p.loc[p.model.eq('zero')]
    check('fixed_zero_predictions_are_null_and_have_no_fit', zd.unit0_predicted_null.all() and zd.origin.eq('fixed_zero').all() and zd.k_independent.all())
    ki = p.loc[p.model.isin(['uncompressed_krr', 'zero'])]
    for metric in METRICS:
        wide = ki.pivot(index=['geometry', 'model', 'vector_id'], columns='nominal_k', values=metric)
        check('K_independent_repeats_' + metric, np.array_equal(wide[2], wide[3]))
    check('K_independent_fit_identity_reused', ki.groupby(['geometry', 'model', 'vector_id']).fit_identity.nunique().eq(1).all())
    components = pd.read_parquet(inputs[9], columns=['vector_id', 'nominal_k', 'model', 'amplitude_mse_pp2', 'alignment_mse_pp2'])
    refs = p.loc[p.geometry.eq(GEOMETRIES[0]) & p.origin.eq('frozen_raw_postmapped'), ['vector_id', 'nominal_k', 'model', 'native_vector_mse']].copy()
    refs['model'] = refs.model.str.removeprefix('raw_postmapped__')
    refs = refs.merge(components, on=['vector_id', 'nominal_k', 'model'], validate='one_to_one')
    ref_error = float((refs.native_vector_mse - refs.amplitude_mse_pp2 - refs.alignment_mse_pp2).abs().max())
    check('centered_raw_references_equal_E63_centered_MSE', ref_error < 1e-08, f'max difference {ref_error:.17g}')
    source_members = pd.read_parquet(inputs[1])
    current_members = pd.read_parquet(base / 'results/e64_membership.parquet')
    check('exact_E62_membership_reused', source_members.equals(current_members))
    f = pd.read_parquet(base / 'results/e64_folds.parquet')
    sf = pd.read_parquet(inputs[2])
    common = [*EDGE, 'fold_role1', 'fold_role2', 'nominal_k', 'status', 'n_train_vectors', 'n_test_vectors', 'train_vector_hash', 'test_vector_hash', 'train_cell_hash', 'test_cell_hash', 'n_cell_overlap']
    check('all_frozen_folds_copied_for_each_target', len(f) == 9300 and all((g[common].sort_values(common[:6]).reset_index(drop=True).equals(sf[common].sort_values(common[:6]).reset_index(drop=True)) for _, g in f.groupby('geometry'))))
    scored = f.loc[f.status.eq('scored')]
    check('all1967folds_and_degenerate_training_retained', len(scored) == 7936 and scored.n_train_true_constant.add(scored.n_train_true_nonzero).eq(scored.n_train_vectors).all() and (scored.n_train_true_nonzero.eq(0).sum() == 160) and (scored.n_train_true_nonzero.le(1).sum() == 376), f'{len(scored)} target/K fold rows')
    edge_rows = []
    summary = []
    full_contrib = []
    for s, g in stratum_rows(p):
        q = g.groupby([*EDGE, *GROUP], as_index=False).agg(n_test_vectors=('vector_id', 'size'), n_true_constant=('true_constant_vector', 'sum'), n_predicted_null=('unit0_predicted_null', 'sum'), **{c: (c, 'mean') for c in METRICS})
        q['stratum'] = s
        edge_rows.append(q)
    comb = pd.DataFrame([(g, k, m, s) for g in GEOMETRIES for k in [2, 3] for m in MODELS for s in STRATA], columns=[*GROUP, 'stratum'])
    e = complete_grid(pd.concat(edge_rows, ignore_index=True), support, comb, [*GROUP, 'stratum'], METRICS, ['n_test_vectors', 'n_true_constant', 'n_predicted_null'])
    e['native_rmse'] = np.sqrt(e.native_vector_mse)
    e['unit0_rmse'] = np.sqrt(e.unit0_vector_mse)
    save(e, 'e64_edge_metrics_1548.csv')
    c = e.groupby(['reaction_type', *GROUP, 'stratum'], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), n_true_constant=('n_true_constant', 'sum'), n_predicted_null=('n_predicted_null', 'sum'), **{x: (x, 'mean') for x in METRICS})
    c['supported'] = c.n_test_edges.gt(0)
    c['native_rmse'] = np.sqrt(c.native_vector_mse)
    c['unit0_rmse'] = np.sqrt(c.unit0_vector_mse)
    save(c, 'e64_class_metrics_20.csv')
    for key, g in p.groupby(GROUP, sort=True):
        extra = dict(zip(GROUP, key))
        rows = list(summary_rows(g, METRICS, extra))
        for row in rows:
            subset = g if row['stratum'] == 'all' else g.loc[g.true_stratum.eq(row['stratum'])]
            row.update(n_true_constant=int(subset.true_constant_vector.sum()), n_predicted_null=int(subset.unit0_predicted_null.sum()), native_rmse=np.sqrt(row['native_vector_mse']), unit0_rmse=np.sqrt(row['unit0_vector_mse']))
        summary.extend(rows)
        full_contrib.extend(contributions(g, METRICS, extra))
    summary = pd.DataFrame(summary)
    full_contrib = pd.DataFrame(full_contrib)
    save(summary, 'e64_summary.csv')
    save(full_contrib, 'e64_full_weight_stratum_contributions.csv')
    check('model_grids_complete_with_NA', len(e) == 1550 * 144 and len(c) == 20 * 144 and (len(summary) == 432) and e.loc[~e.supported, METRICS].isna().all().all() and c.loc[~c.supported, METRICS].isna().all().all())
    full = summary.loc[summary.stratum.eq('all')].set_index([*GROUP, 'weighting'])[METRICS].sort_index()
    rec = full_contrib.groupby([*GROUP, 'weighting'])[METRICS].sum().sort_index()
    check('stratum_full_weight_contributions_recover_model_totals', np.allclose(full, rec, atol=1e-08, rtol=0))
    pooled_zero = summary.loc[summary.geometry.eq(GEOMETRIES[1]) & summary.model.eq('zero') & summary.stratum.eq('all') & summary.weighting.eq('pooled')]
    check('unit_zero_pooled_population_mass', np.allclose(pooled_zero.native_vector_mse, 16032 / 25956, atol=1e-12, rtol=0))
    comparisons = [('centroid_minus_' + m, 'centroid', m) for m in MODELS] + [('zero_minus_' + m, 'zero', m) for m in MODELS] + [('hard_minus_continuous', 'codebook_hard', 'codebook_continuous')] + [('uncompressed_minus_' + m, 'uncompressed_krr', m) for m in FIVE]
    paircols = ['geometry', 'nominal_k', 'comparison', 'lhs_model', 'rhs_model']
    difference = ['difference_' + x for x in METRICS]
    pes = []
    ps = []
    pcs = []
    for (geometry, k), g in p.groupby(['geometry', 'nominal_k'], sort=True):
        wide = g.pivot(index=[*EDGE, 'vector_id', 'true_constant_vector'], columns='model', values=METRICS)
        for name, lhs, rhs in comparisons:
            v = wide.index.to_frame(index=False)
            for metric, out in zip(METRICS, difference):
                v[out] = (wide[metric, lhs] - wide[metric, rhs]).to_numpy()
            extra = dict(geometry=geometry, nominal_k=int(k), comparison=name, lhs_model=lhs, rhs_model=rhs)
            ps.extend(summary_rows(v, difference, extra))
            pcs.extend(contributions(v, difference, extra))
            for s, sub in stratum_rows(v):
                q = sub.groupby(EDGE, as_index=False).agg(n_test_vectors=('vector_id', 'size'), **{x: (x, 'mean') for x in difference})
                for key, value in dict(**extra, stratum=s).items():
                    q[key] = value
                pes.append(q)
    pg = pd.DataFrame([(geometry, k, name, lhs, rhs, s) for geometry in GEOMETRIES for k in [2, 3] for name, lhs, rhs in comparisons for s in STRATA], columns=[*paircols, 'stratum'])
    pe = complete_grid(pd.concat(pes, ignore_index=True), support, pg, [*paircols, 'stratum'], difference, ['n_test_vectors'])
    save(pe, 'e64_paired_edge_metrics_1548.csv')
    pc = pe.groupby(['reaction_type', *paircols, 'stratum'], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), **{x: (x, 'mean') for x in difference})
    pc['supported'] = pc.n_test_edges.gt(0)
    save(pc, 'e64_paired_class_metrics_20.csv')
    ps, pcs = (pd.DataFrame(ps), pd.DataFrame(pcs))
    save(ps, 'e64_paired_summary.csv')
    save(pcs, 'e64_paired_full_weight_stratum_contributions.csv')
    pair_all = ps.loc[ps.stratum.eq('all')].set_index([*paircols, 'weighting'])[difference].sort_index()
    pair_rec = pcs.groupby([*paircols, 'weighting'])[difference].sum().sort_index()
    check('paired_full_weight_contributions_recover_all_totals', np.allclose(pair_all, pair_rec, atol=1e-08, rtol=0))
    check('paired_grids_keep_all_registered_edges_classes', len(pe) == 1550 * 360 and len(pc) == 20 * 360 and (len(ps) == 1080) and (len(pcs) == 720))
    save(support, 'e64_support_bridge_1548.csv')
    save(pd.read_csv(inputs[6]), 'e64_support_classes_20.csv')
    pd.DataFrame(checks).to_csv(base / 'audit/e64_production_checks.csv', index=False)
    if not all((row['passed'] for row in checks)):
        raise ValueError('E64 production aggregation checks failed: ' + str([x for x in checks if not x['passed']]))
