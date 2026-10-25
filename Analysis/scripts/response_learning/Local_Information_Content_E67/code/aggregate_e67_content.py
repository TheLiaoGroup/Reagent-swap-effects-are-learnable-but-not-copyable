"""Score E67 saved fixed-operator arrays and preserve the full support ledger."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'code'))
from fit_e67_content import EDGE, E62, E66, RAW_VARIANTS, UNIT_VARIANTS, center_and_unit, pins, sha, write_json
METRICS = ['native_mse', 'native_mae', 'level_mse', 'modulation_mse', 'unit_action_mse', 'half_action_mse', 'half_action_activity']
DECOMP = ['full_update', 'augmented_zero_change', 'local_level_content', 'local_modulation_content', 'orthogonal_cross_term', 'reconstruction_error']
WEIGHTS = ['pooled', 'equal_edge', 'equal_class_edge_balanced']

def strata_rows(frame):
    for name, mask in [('all', np.ones(len(frame), bool)), ('true_nonconstant', ~frame.true_constant.to_numpy(bool)), ('true_constant', frame.true_constant.to_numpy(bool))]:
        yield (name, frame.loc[mask])

def summarize(frame, group, metrics):
    keys = [*EDGE, *group]
    edge = frame.groupby(keys, sort=True, observed=True).agg(n_test_vectors=('vector_id', 'size'), **{m: (m, 'mean') for m in metrics}).reset_index()
    cls = edge.groupby(['reaction_type', *group], sort=True, observed=True).agg(n_test_vectors=('n_test_vectors', 'sum'), n_test_edges=('component_pair', 'size'), **{m: (m, 'mean') for m in metrics}).reset_index()
    summaries = []
    for key, g in frame.groupby(group, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        sel = pd.Series(True, index=edge.index)
        sc = pd.Series(True, index=cls.index)
        for name, val in zip(group, key):
            sel &= edge[name].eq(val)
            sc &= cls[name].eq(val)
        e, c = (edge.loc[sel], cls.loc[sc])
        for weighting, source in zip(WEIGHTS, [g, e, c]):
            summaries.append(dict(**dict(zip(group, key)), weighting=weighting, n_test_vectors=len(g), n_test_edges=len(e), n_test_classes=len(c), **{m: source[m].mean() for m in metrics}))
    return (edge, cls, pd.DataFrame(summaries))

def complete_ledgers(edge, cls, ledger, classes, combinations, group):
    grid = ledger[EDGE].merge(combinations, how='cross')
    edge = grid.merge(edge, on=[*EDGE, *group], how='left', validate='one_to_one')
    edge['supported'] = edge.n_test_vectors.notna()
    edge['n_test_vectors'] = edge.n_test_vectors.fillna(0).astype(int)
    cgrid = classes[['reaction_type']].merge(combinations, how='cross')
    cls = cgrid.merge(cls, on=['reaction_type', *group], how='left', validate='one_to_one')
    cls['supported'] = cls.n_test_vectors.notna()
    for n in ['n_test_vectors', 'n_test_edges']:
        cls[n] = cls[n].fillna(0).astype(int)
    return (edge, cls)

def main():
    started = time.perf_counter()
    fm = BASE / 'metadata/e67_fit_metadata.json'
    fit = json.loads(fm.read_text())
    if fit['n_edges'] != 93 or fit['n_predictions'] != 1162608:
        raise ValueError('E67 fits incomplete')
    if any((sha(BASE.parent.parent / p) != h for p, h in fit['source_sha256'].items())):
        raise ValueError('E67 frozen sources changed')
    vpath = E62 / 'results/e62_vectors.parquet'
    vectors = pd.read_parquet(vpath).set_index('vector_id')
    ledger = pd.read_csv(E62 / 'results/e62_support_bridge_1548.csv')
    classes = pd.read_csv(E66 / 'results/e66_support_classes_20.csv')
    inputs = [fm, vpath, E62 / 'results/e62_support_bridge_1548.csv', E66 / 'results/e66_support_classes_20.csv', E66 / 'results/e66_scope_vectors.parquet']
    losses, decomposition, checks = ([], [], [])
    max_reconstruction = max_levelmod = max_geometry = 0.0
    for name, digest in fit['checkpoint_manifest'].items():
        mp = BASE / 'parts' / name
        if sha(mp) != digest:
            raise ValueError('E67 checkpoint metadata changed')
        meta = json.loads(mp.read_text())
        pp = BASE / 'parts' / name.replace('_metadata.json', '_predictions.parquet')
        if sha(pp) != meta['prediction_sha256']:
            raise ValueError('E67 predictions changed')
        inputs.extend([mp, pp])
        frame = pd.read_parquet(pp)
        for geometry, g in frame.groupby('target_geometry', sort=True):
            raw = np.stack(vectors.loc[g.vector_id, 'target_vector'])
            _, unit, null, _ = center_and_unit(raw)
            target = raw if geometry == 'raw_delta' else unit
            pred = np.stack(g.prediction)
            err = pred - target
            pc, direction, pn, rms = center_and_unit(pred)
            tc = center_and_unit(target)[0]
            val = g.drop(columns='prediction').copy()
            val['true_constant'] = null
            val['native_mse'] = np.mean(err * err, axis=1)
            val['native_mae'] = np.mean(np.abs(err), axis=1)
            val['level_mse'] = err.mean(1) ** 2
            val['modulation_mse'] = np.mean((pc - tc) ** 2, axis=1)
            if geometry == 'centered_unit_rms_zeroembed':
                active = rms > 0.5
                val['unit_action_mse'] = np.mean((direction - unit) ** 2, axis=1)
                val['half_action_mse'] = np.mean((direction * active[:, None] - unit) ** 2, axis=1)
                val['half_action_activity'] = active.astype(float)
            else:
                val[['unit_action_mse', 'half_action_mse', 'half_action_activity']] = np.nan
            max_geometry = max(max_geometry, float(np.max(np.abs(val.native_mse - val.level_mse - val.modulation_mse))))
            losses.append(val)
        rawf = frame.loc[frame.target_geometry.eq('raw_delta')]
        index = ['family_request_id', 'allocation', 'vector_id']
        table = rawf.pivot(index=index, columns='variant', values='prediction')
        truth = np.stack(vectors.loc[table.index.get_level_values('vector_id'), 'target_vector'])
        a = {k: np.stack(table[k]) for k in RAW_VARIANTS}
        risk = {k: np.mean((p - truth) ** 2, axis=1) for k, p in a.items()}
        z = a['zero_local']
        dl = a['level_only'] - z
        dm = a['modulation_only'] - z
        cross = 2 * np.mean(dl * dm, axis=1)
        full = risk['full'] - risk['frozen_family']
        op = risk['zero_local'] - risk['frozen_family']
        lc = risk['level_only'] - risk['zero_local']
        mc = risk['modulation_only'] - risk['zero_local']
        reconstruction = full - op - lc - mc - cross
        max_reconstruction = max(max_reconstruction, float(np.max(np.abs(reconstruction))))
        max_levelmod = max(max_levelmod, float(np.max(np.abs(dl - dl.mean(1)[:, None]))), float(np.max(np.abs(dm.mean(1)))))
        d = table.reset_index()[index]
        key = frame.iloc[0]
        for k in EDGE:
            d[k] = key[k]
        d['true_constant'] = center_and_unit(truth)[2]
        for n, x in zip(DECOMP, [full, op, lc, mc, cross, reconstruction]):
            d[n] = x
        decomposition.append(d)
    if max_reconstruction > 1e-08 or max_levelmod > 1e-09 or max_geometry > 1e-08:
        raise ValueError('E67 risk geometry failed')
    loss = pd.concat(losses, ignore_index=True)
    decomp = pd.concat(decomposition, ignore_index=True)
    before = pins(inputs)
    outputs = []

    def save(df, name):
        path = BASE / 'results' / name
        if path.suffix == '.parquet':
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        outputs.append(path)
    save(loss, 'e67_vector_losses.parquet')
    save(decomp, 'e67_decomposition_vectors.parquet')
    group = ['target_geometry', 'variant', 'stratum']
    avg = loss.groupby([*EDGE, 'vector_id', 'target_geometry', 'variant', 'true_constant'], sort=True)[METRICS].mean().reset_index()
    scored = pd.concat([g.assign(stratum=s) for s, g in strata_rows(avg)], ignore_index=True)
    e, c, s = summarize(scored, group, METRICS)
    combos = scored[group].drop_duplicates()
    e, c = complete_ledgers(e, c, ledger, classes, combos, group)
    save(e, 'e67_edge_metrics_1548.parquet')
    save(c, 'e67_class_metrics_20.csv')
    save(s, 'e67_summary.csv')
    allocations = pd.concat([g.assign(stratum=s) for s, g in strata_rows(loss)], ignore_index=True)
    _, _, sa = summarize(allocations, ['allocation', *group], METRICS)
    save(sa, 'e67_allocation_summary.csv')
    davg = decomp.groupby([*EDGE, 'vector_id', 'true_constant'], sort=True)[DECOMP].mean().reset_index()
    ds = pd.concat([g.assign(stratum=s) for s, g in strata_rows(davg)], ignore_index=True)
    de, dc, dsum = summarize(ds, ['stratum'], DECOMP)
    de, dc = complete_ledgers(de, dc, ledger, classes, ds[['stratum']].drop_duplicates(), ['stratum'])
    save(de, 'e67_decomposition_edges_1548.parquet')
    save(dc, 'e67_decomposition_classes_20.csv')
    save(dsum, 'e67_decomposition_summary.csv')
    da = pd.concat([g.assign(stratum=s) for s, g in strata_rows(decomp)], ignore_index=True)
    _, _, dasum = summarize(da, ['allocation', 'stratum'], DECOMP)
    save(dasum, 'e67_decomposition_allocations.csv')
    scope = pd.read_parquet(E66 / 'results/e66_scope_vectors.parquet')
    supported = scope.loc[scope.common8].copy()
    supported['true_constant'] = [bool(np.all(x == x[0])) for x in vectors.loc[supported.vector_id, 'target_vector']]
    support = supported.groupby(EDGE).agg(n_common8=('vector_id', 'size'), n_true_constant=('true_constant', 'sum')).reset_index()
    support = ledger.merge(support, on=EDGE, how='left', validate='one_to_one')
    for n in ['n_common8', 'n_true_constant']:
        support[n] = support[n].fillna(0).astype(int)
    save(support, 'e67_support_bridge_1548.csv')
    cs = support.groupby('reaction_type', sort=True).agg(n_registered_edges=('component_pair', 'size'), n_common8=('n_common8', 'sum'), n_true_constant=('n_true_constant', 'sum')).reset_index()
    save(cs, 'e67_support_classes_20.csv')
    for geometry, variants in [('raw_delta', RAW_VARIANTS), ('centered_unit_rms_zeroembed', UNIT_VARIANTS)]:
        take = loss.loc[loss.target_geometry.eq(geometry)]
        checks.append(dict(check=geometry + '_full_common8_each_order_variant', passed=bool(len(take) == 24221 * 3 * len(variants) and take.groupby(['allocation', 'variant']).vector_id.nunique().eq(24221).all())))
    checks.extend([dict(check='full_support_20_1548', passed=bool(len(support) == 1550 and len(cs) == 20 and (support.n_common8.sum() == 24221))), dict(check='unsupported_metrics_NA', passed=bool(e.loc[~e.supported, METRICS].isna().all().all() and c.loc[~c.supported, METRICS].isna().all().all())), dict(check='all_scalar_losses_finite', passed=bool(np.isfinite(loss[['native_mse', 'native_mae', 'level_mse', 'modulation_mse']]).all().all())), dict(check='allocation_mean_equals_summary', passed=bool(all((np.allclose(s.set_index([*group, 'weighting'])[m].sort_index(), sa.groupby([*group, 'weighting'])[m].mean().sort_index(), atol=1e-08, rtol=0, equal_nan=True) for m in METRICS))))])
    if before != pins(inputs):
        raise ValueError('E67 inputs changed during aggregation')
    save(pd.DataFrame(checks), 'e67_score_checks.csv')
    metadata = dict(source_sha256=before, output_sha256=pins(outputs), code_sha256=sha(Path(__file__)), n_loss_rows=len(loss), n_decomposition_rows=len(decomp), n_checks=len(checks), failures=sum((not x['passed'] for x in checks)), maximum_risk_reconstruction_error=max_reconstruction, maximum_level_modulation_geometry_error=max_levelmod, maximum_risk_geometry_error=max_geometry, seconds=time.perf_counter() - started)
    write_json(BASE / 'metadata/e67_score_metadata.json', metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k not in ['source_sha256', 'output_sha256']}), flush=True)
    if metadata['failures']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
