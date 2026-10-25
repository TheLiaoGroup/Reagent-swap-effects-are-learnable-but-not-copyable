"""E62 fixed, descriptive postprocessing; no fitting or threshold search."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
WEIGHTS = ['pooled', 'equal_edge', 'equal_class_edge_balanced']
MODELS = ['centroid', 'codebook_continuous', 'codebook_hard', 'codebook_mixture', 'pca_continuous']
PAIRS = [('codebook_hard', 'codebook_continuous'), ('codebook_mixture', 'codebook_continuous'), ('pca_continuous', 'codebook_continuous'), ('codebook_continuous', 'centroid'), ('codebook_hard', 'centroid'), ('codebook_mixture', 'centroid'), ('pca_continuous', 'centroid')]
INPUTS = [BASE / f'results/e62_{name}' for name in ['predictions.parquet', 'oracles.parquet', 'folds.parquet', 'summary.csv', 'support_bridge_1548.csv']]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pins(paths):
    return {str(path.relative_to(BASE)): sha(path) for path in paths}

def row_weights(group):
    sizes = group.groupby(EDGE).vector_id.transform('size').to_numpy(float)
    edges = group[EDGE].drop_duplicates()
    n_edge_class = edges.groupby('reaction_type').size()
    return dict(pooled=np.full(len(group), 1 / len(group)), equal_edge=1 / (len(edges) * sizes), equal_class_edge_balanced=1 / (edges.reaction_type.nunique() * group.reaction_type.map(n_edge_class).to_numpy() * sizes))

def weighted_rows(group, columns, extra):
    rows = []
    for weighting, weights in row_weights(group).items():
        rows.append(dict(**extra, weighting=weighting, n_vectors=len(group), n_edges=len(group[EDGE].drop_duplicates()), n_classes=group.reaction_type.nunique(), weight_sum=float(weights.sum()), **{column: float(weights @ group[column].to_numpy(float)) for column in columns}))
    return rows

def direction_rates(table):
    table = table.copy()
    table['direction_accuracy'] = table.n_correct / table.n_selected.replace(0, np.nan)
    table['positive_recall'] = table.n_positive_correct / table.n_positive.replace(0, np.nan)
    table['negative_recall'] = table.n_negative_correct / table.n_negative.replace(0, np.nan)
    table['balanced_accuracy'] = (table.positive_recall + table.negative_recall) / 2
    return table

def main():
    before = pins(INPUTS)
    p = pd.read_parquet(INPUTS[0])
    support = pd.read_csv(INPUTS[4])
    checks, written = ([], [])

    def save(frame, name):
        path = BASE / 'results' / name
        if path.suffix == '.parquet':
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        written.append(path)

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
    idcols = [*EDGE, 'nominal_k', 'vector_id', 'allzero_vector', 'n_backgrounds']
    mse = p.pivot(index=idcols, columns='model', values='vector_mse_pp2')
    mae = p.pivot(index=idcols, columns='model', values='vector_mae_pp')
    check('all_models_paired_on_identical_vectors', not mse.isna().any().any() and (not mae.isna().any().any()), f'{len(mse)} vector-capacity rows')
    paired = []
    for lhs, rhs in PAIRS:
        frame = mse.reset_index()[idcols].copy()
        frame['lhs_model'], frame['rhs_model'] = (lhs, rhs)
        frame['mse_difference_pp2'] = (mse[lhs] - mse[rhs]).to_numpy()
        frame['mae_difference_pp'] = (mae[lhs] - mae[rhs]).to_numpy()
        paired.append(frame)
    paired = pd.concat(paired, ignore_index=True)
    save(paired, 'e62_paired_vector_losses.parquet')
    paircols, losscols = (['nominal_k', 'lhs_model', 'rhs_model'], ['mse_difference_pp2', 'mae_difference_pp'])
    edge = paired.groupby([*EDGE, *paircols], as_index=False).agg(n_vectors=('vector_id', 'size'), **{c: (c, 'mean') for c in losscols})
    pairgrid = pd.DataFrame([(k, lhs, rhs) for k in [2, 3] for lhs, rhs in PAIRS], columns=paircols)
    edge = support[EDGE].merge(pairgrid, how='cross').merge(edge, on=[*EDGE, *paircols], how='left', validate='one_to_one')
    edge['n_vectors'] = edge.n_vectors.fillna(0).astype(int)
    edge['supported'] = edge.n_vectors.gt(0)
    save(edge, 'e62_paired_edge_losses_1548.csv')
    cls = edge.groupby(['reaction_type', *paircols], as_index=False).agg(n_edges=('supported', 'sum'), n_vectors=('n_vectors', 'sum'), **{c: (c, 'mean') for c in losscols})
    cls['supported'] = cls.n_edges.gt(0)
    save(cls, 'e62_paired_class_losses_20.csv')
    summaries, contributions = ([], [])
    for key, group in paired.groupby(paircols, sort=True):
        extras = dict(zip(paircols, key))
        summaries.extend(weighted_rows(group, losscols, extras))
        for weighting, weights in row_weights(group).items():
            for zero, label in [(True, 'allzero'), (False, 'nonzero')]:
                mask = group.allzero_vector.to_numpy() == zero
                contributions.append(dict(**extras, weighting=weighting, stratum=label, n_vectors=int(mask.sum()), full_sample_weight_mass=float(weights[mask].sum()), **{c: float(weights[mask] @ group.loc[mask, c].to_numpy()) for c in losscols}))
    summaries, contributions = (pd.DataFrame(summaries), pd.DataFrame(contributions))
    save(summaries, 'e62_paired_summary.csv')
    save(contributions, 'e62_zero_nonzero_full_weight_contributions.csv')
    reconstructed = contributions.groupby([*paircols, 'weighting'])[losscols].sum().sort_index()
    expected = summaries.set_index([*paircols, 'weighting'])[losscols].sort_index()
    err = float(np.max(np.abs(reconstructed.to_numpy() - expected.to_numpy())))
    check('zero_nonzero_fixed_weight_contributions_sum_to_total', err < 1e-10, f'max difference {err:.3g}')
    check('all_three_weight_masses_equal_one', np.allclose(summaries.weight_sum, 1, atol=1e-12, rtol=0), summaries.weight_sum.min())
    strata, strat_edge = ([], [])
    for (k, model, zero), group in p.groupby(['nominal_k', 'model', 'allzero_vector'], sort=True):
        extra = dict(nominal_k=int(k), model=model, stratum='allzero' if zero else 'nonzero')
        rows = weighted_rows(group, ['vector_mse_pp2', 'vector_mae_pp', 'vector_bias_pp'], extra)
        for row in rows:
            row['rmse_pp'] = np.sqrt(row['vector_mse_pp2'])
        strata.extend(rows)
        e = group.groupby(EDGE, as_index=False).agg(n_vectors=('vector_id', 'size'), mse_pp2=('vector_mse_pp2', 'mean'), mae_pp=('vector_mae_pp', 'mean'), bias_pp=('vector_bias_pp', 'mean'))
        for column, value in extra.items():
            e[column] = value
        strat_edge.append(e)
    save(pd.DataFrame(strata), 'e62_zero_nonzero_summary.csv')
    se = pd.concat(strat_edge, ignore_index=True)
    segrid = pd.DataFrame([(k, model, s) for k in [2, 3] for model in MODELS for s in ['allzero', 'nonzero']], columns=['nominal_k', 'model', 'stratum'])
    se = support[EDGE].merge(segrid, how='cross').merge(se, on=[*EDGE, 'nominal_k', 'model', 'stratum'], how='left', validate='one_to_one')
    se['n_vectors'] = se.n_vectors.fillna(0).astype(int)
    se['supported'] = se.n_vectors.gt(0)
    save(se, 'e62_zero_nonzero_edge_metrics_1548.csv')
    sc = se.groupby(['reaction_type', 'nominal_k', 'model', 'stratum'], as_index=False).agg(n_edges=('supported', 'sum'), n_vectors=('n_vectors', 'sum'), mse_pp2=('mse_pp2', 'mean'), mae_pp=('mae_pp', 'mean'), bias_pp=('bias_pp', 'mean'))
    sc['supported'] = sc.n_edges.gt(0)
    save(sc, 'e62_zero_nonzero_class_metrics_20.csv')
    rebuilt = []
    for (k, model), group in p.groupby(['nominal_k', 'model'], sort=True):
        rebuilt.extend(weighted_rows(group, ['vector_mse_pp2', 'vector_mae_pp'], dict(nominal_k=k, model=model)))
    rebuilt = pd.DataFrame(rebuilt).set_index(['nominal_k', 'model', 'weighting']).sort_index()
    original = pd.read_csv(INPUTS[3]).set_index(['nominal_k', 'model', 'weighting']).sort_index()
    aggerr = max(float(np.max(np.abs(rebuilt['vector_mse_pp2'] - original.mse_pp2))), float(np.max(np.abs(rebuilt['vector_mae_pp'] - original.mae_pp))))
    check('independent_full_weight_reconstruction_matches_main', aggerr < 1e-10, f'max difference {aggerr:.3g}')
    direction = p[idcols + ['model']].copy()
    counts, max_loss_err = ([], 0.0)
    for row in p.itertuples(index=False):
        y, pred = (np.asarray(row.target_vector), np.asarray(row.prediction_vector))
        e = pred - y
        max_loss_err = max(max_loss_err, abs(float(np.mean(e * e)) - row.vector_mse_pp2), abs(float(np.mean(np.abs(e))) - row.vector_mae_pp))
        pos, neg = (y >= 20, y <= -20)
        selected = pos | neg
        pc, nc = (int(np.count_nonzero(pos & (pred > 0))), int(np.count_nonzero(neg & (pred < 0))))
        counts.append((int(selected.sum()), int(pos.sum()), int(neg.sum()), pc, nc, pc + nc, int(np.count_nonzero(selected & (pred == 0)))))
    countcols = ['n_selected', 'n_positive', 'n_negative', 'n_positive_correct', 'n_negative_correct', 'n_correct', 'n_prediction_ties']
    direction[countcols] = np.asarray(counts, dtype=np.int64)
    save(direction, 'e62_direction_fixed20_vector_counts.parquet')
    check('saved_losses_recompute_from_all_prediction_arrays', max_loss_err < 1e-10, f'max difference {max_loss_err:.3g}')
    d = direction.groupby([*EDGE, 'nominal_k', 'model'], as_index=False)[countcols].sum()
    modelgrid = pd.DataFrame([(k, model) for k in [2, 3] for model in MODELS], columns=['nominal_k', 'model'])
    d = support[EDGE].merge(modelgrid, how='cross').merge(d, on=[*EDGE, 'nominal_k', 'model'], how='left', validate='one_to_one')
    d[countcols] = d[countcols].fillna(0).astype(int)
    d = direction_rates(d)
    d['supported'] = d.n_selected.gt(0)
    save(d, 'e62_direction_fixed20_edges_1548.csv')
    rates = ['direction_accuracy', 'positive_recall', 'negative_recall', 'balanced_accuracy']
    dc = d.groupby(['reaction_type', 'nominal_k', 'model'], as_index=False).agg(**{c: (c, 'mean') for c in rates}, **{c: (c, 'sum') for c in countcols}, n_supported_edges=('supported', 'sum'))
    dc['supported'] = dc.n_supported_edges.gt(0)
    save(dc, 'e62_direction_fixed20_classes_20.csv')
    ds = []
    for (k, model), group in d.groupby(['nominal_k', 'model'], sort=True):
        cgroup = dc.loc[dc.nominal_k.eq(k) & dc.model.eq(model)]
        pooled = direction_rates(pd.DataFrame([group[countcols].sum()])).iloc[0]
        for weighting, source in [('pooled', pooled), ('equal_edge', group[rates].mean()), ('equal_class_edge_balanced', cgroup[rates].mean())]:
            ds.append(dict(nominal_k=k, model=model, weighting=weighting, **{c: int(group[c].sum()) for c in countcols}, n_supported_edges=int(group.supported.sum()), n_supported_classes=int(cgroup.supported.sum()), **{c: float(source[c]) for c in rates}, n_edges_positive_support=int(group.positive_recall.notna().sum()), n_edges_negative_support=int(group.negative_recall.notna().sum()), n_edges_both_direction_support=int(group.balanced_accuracy.notna().sum())))
    save(pd.DataFrame(ds), 'e62_direction_fixed20_summary.csv')
    check('direction_counts_valid_and_ties_never_correct', bool((direction.n_correct <= direction.n_selected - direction.n_prediction_ties).all() and (direction.n_correct == direction.n_positive_correct + direction.n_negative_correct).all()), 'strict pred>0 or pred<0; exact zero ties count incorrect')
    check('complete_edge_class_accounting_in_downstream_grids', len(edge) == 1550 * 14 and len(cls) == 20 * 14 and (len(d) == 1550 * 10) and (len(dc) == 20 * 10), f'paired edges/classes {len(edge)}/{len(cls)}, direction {len(d)}/{len(dc)}')
    oracle = pd.read_parquet(INPUTS[1])
    osummary = []
    for (k, kind), group in oracle.groupby(['nominal_k', 'oracle_kind'], sort=True):
        osummary.extend(weighted_rows(group, ['vector_mse_pp2'], dict(nominal_k=k, oracle_kind=kind, predictive=False)))
    save(pd.DataFrame(osummary), 'e62_oracle_summary.csv')
    folds = pd.read_parquet(INPUTS[2])
    diag = folds.groupby(['nominal_k', 'status', 'effective_k', 'effective_rank'], dropna=False, as_index=False).agg(n_fold_rows=('status', 'size'), n_test_vector_exposures=('n_test_vectors', 'sum'), n_train_vector_exposures=('n_train_vectors', 'sum'))
    save(diag, 'e62_effective_capacity_folds.csv')
    check('upstream_outputs_unchanged', before == pins(INPUTS), 'all five postprocess inputs pinned')
    auditpath = BASE / 'audit/e62_postprocess_checks.csv'
    pd.DataFrame(checks).to_csv(auditpath, index=False)
    written.append(auditpath)
    metadata = dict(analysis='E62 matched representation fixed descriptive postprocessing', code_sha256=sha(Path(__file__)), input_sha256=before, output_sha256=pins(written), checks=checks, failures=sum((not c['passed'] for c in checks)), paired_sign='lhs minus rhs; positive means lhs larger loss', fitted_models=0, stratum_weighting='Zero/nonzero summary renormalizes within stratum. Contributions retain original full-sample vector weights and sum exactly to overall paired loss difference.', direction_endpoint=dict(threshold_abs_true_delta_pp=20, selection='Observed outcome-conditioned coordinate subset; no threshold scan or noise calibration', ties='prediction==0 is incorrect', pooling='Within edge: coordinate-count rates; equal-edge averages supported edge rates; equal-class averages class rates formed by equal-edge means. Per-direction support may differ and missing stays NA; balanced accuracy requires both directions at each edge.', interpretation='Descriptive known-background response direction, not a prospective subset selection rule or mechanism test'))
    (BASE / 'metadata/e62_postprocess_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if metadata['failures']:
        raise SystemExit('E62 postprocess checks failed')
    print(f'E62 postprocess: {len(checks)}/{len(checks)} checks, {len(written)} outputs, no fitting')
if __name__ == '__main__':
    main()
