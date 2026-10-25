"""E65 full-kernel prediction with frozen chemistry-only membership and edge checkpoints."""
from __future__ import annotations
import argparse
import platform
import time
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits
from e65_common import *

def fingerprints(canonical):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False, useBondTypes=True, includeRingMembership=True, countSimulation=False)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in canonical]
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], float)

def solve(kernel, cross, target):
    col = kernel.mean(axis=0)
    grand = float(col.mean())
    centered = kernel - col[None, :] - col[:, None] + grand
    symmetry = float(np.max(np.abs(centered - centered.T)))
    if symmetry > 1e-12:
        raise ValueError('Kernel symmetry differs')
    centered = (centered + centered.T) * 0.5
    centered.flat[::len(centered) + 1] += ALPHA
    mean = target.mean(axis=0)
    factor = cho_factor(centered, lower=True, check_finite=False)
    coefficients = cho_solve(factor, target - mean, check_finite=False)
    query = cross - cross.mean(axis=1)[:, None] - col[None, :] + grand
    return (mean + query @ coefficients, mean, symmetry)

def formula_checks():
    rows = []

    def check(name, value):
        rows.append(dict(check=name, passed=bool(value)))
    kernel = np.array([[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1.0]])
    cross = np.array([[0.4, 0.2, 0.6], [0.8, 0.3, 0.1]])
    y = np.array([[0, 0, 0], [1, 3, 7], [-2, 1, 6.0]], float)
    c, u, null, rms = center_and_unit(y)
    block = np.hstack([y, c, u])
    prediction, mean, _ = solve(kernel, cross, block)
    check('linear_centering_commutes', np.allclose(center_and_unit(prediction[:, :3])[0], prediction[:, 3:6], atol=1e-12, rtol=0))
    singleton = solve(kernel[1:2, 1:2], cross[:, 1:2], block[1:2])[0]
    check('singleton_equals_nonzero_training_target', np.array_equal(singleton, np.broadcast_to(block[1:2], singleton.shape)))
    zero = solve(kernel, cross, np.zeros_like(block))[0]
    check('allzero_training_retained', np.array_equal(zero, np.zeros_like(zero)))
    xc = kernel - kernel.mean(0) - kernel.mean(1)[:, None] + kernel.mean()
    q = cross - cross.mean(1)[:, None] - kernel.mean(0) + kernel.mean()
    direct = np.linalg.solve(xc + ALPHA * np.eye(3), block - block.mean(0))
    check('Cholesky_equals_direct_fixed_intercept', np.allclose(prediction, block.mean(0) + q @ direct, atol=1e-12, rtol=0))
    pc, pu, pn, pr = center_and_unit(prediction[:, :3])
    error = prediction[:, :3] - y[:2]
    check('raw_MSE_level_modulation_identity', np.allclose(np.mean(error ** 2, 1), np.mean(error, 1) ** 2 + np.mean((pc - c[:2]) ** 2, 1), atol=1e-12, rtol=0))
    check('unit_target_zero_action_identity', np.allclose(np.mean(u * u, 1), (~null).astype(float), atol=1e-12, rtol=0))
    return rows

def losses(prediction, target, true_unit):
    pc, decoded, pnull, prms = center_and_unit(prediction)
    tc = center_and_unit(target)[0]
    error = prediction - target
    return (dict(native_vector_mse=np.mean(error * error, axis=1), native_vector_mae=np.mean(np.abs(error), axis=1), unit0_vector_mse=np.mean((decoded - true_unit) ** 2, axis=1), unit0_vector_mae=np.mean(np.abs(decoded - true_unit), axis=1), mean_offset_mse=np.mean(error, axis=1) ** 2, centered_error_mse=np.mean((pc - tc) ** 2, axis=1)), decoded, pnull, prms)

def run_edge(key, vectors, scope, requests, factors, gram, molindex, familymap, before, plan_hash):
    tag = edge_tag(key)
    names = ['predictions', 'baseline_metrics', 'union_predictions', 'factor_diagnostics']
    paths = {n: BASE / f'parts/{tag}_{n}.parquet' for n in names}
    mp = BASE / f'parts/{tag}_metadata.json'
    identity = dict(edge=list(key), input_sha256=before, preflight_metadata_sha256=plan_hash, producer_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e65_common.py'), vector_ids_hash=array_hash(sorted(vectors.vector_id)), factor_ids=sorted(factors.factor_id))
    if mp.exists():
        saved = json.loads(mp.read_text(encoding='utf-8'))
        if saved.get('identity') == identity and all((p.exists() and sha(p) == saved['output_sha256'].get(n) for n, p in paths.items())):
            print(f'Reused {tag}', flush=True)
            return dict(paths=paths, metadata=mp, meta=saved, reused=True)
    started = time.perf_counter()
    vectors = vectors.sort_values('vector_id')
    ids = vectors.vector_id.to_numpy(np.int64)
    positions = {int(x): i for i, x in enumerate(ids)}
    scope = scope.set_index('vector_id')
    y = np.vstack(vectors.target_vector)
    c, u, true_null, true_rms = center_and_unit(y)
    B = y.shape[1]
    block = np.hstack([y, c, u])
    targets = dict(zip(GEOMETRIES, [y, c, u]))
    r1 = np.array([molindex[x] for x in vectors.canonical_role1])
    r2 = np.array([molindex[x] for x in vectors.canonical_role2])
    kernel = (gram[np.ix_(r1, r1)] + gram[np.ix_(r2, r2)]) / 2
    fam1 = np.array([familymap[x] for x in vectors.canonical_role1])
    fam2 = np.array([familymap[x] for x in vectors.canonical_role2])
    cache = {}
    unions = []
    diagnostics = []
    maximum_commute_error = 0.0
    maximum_singleton_error = 0.0
    maximum_symmetry_error = 0.0
    for factor in factors.to_dict('records'):
        fid = factor['factor_id']
        trids = factor['train_vector_ids']
        qids = factor['union_query_vector_ids']
        tr = np.array([positions[int(i)] for i in trids])
        query = np.array([positions[int(i)] for i in qids])
        prediction, mean, symmetry = solve(kernel[np.ix_(tr, tr)], kernel[np.ix_(query, tr)], block[tr])
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite full prediction')
        commute = float(np.max(np.abs(center_and_unit(prediction[:, :B])[0] - prediction[:, B:2 * B])))
        maximum_commute_error = max(maximum_commute_error, commute)
        maximum_symmetry_error = max(maximum_symmetry_error, symmetry)
        singleton = float(np.max(np.abs(prediction - block[tr[0]]))) if len(tr) == 1 else 0.0
        maximum_singleton_error = max(maximum_singleton_error, singleton)
        n1 = len(np.unique(r1[tr]))
        n2 = len(np.unique(r2[tr]))
        f1 = set(fam1[tr])
        f2 = set(fam2[tr])
        allf = f1 | f2
        near1 = gram[np.ix_(r1[query], np.unique(r1[tr]))].max(axis=1)
        near2 = gram[np.ix_(r2[query], np.unique(r2[tr]))].max(axis=1)
        diag = dict(**dict(zip(EDGE, key)), factor_id=fid, n_train=len(tr), n_union_query=len(query), n_backgrounds=B, train_vector_hash=array_hash(trids), union_query_vector_hash=array_hash(qids), train_mean_block=mean.tolist(), n_train_true_constant=int(true_null[tr].sum()), n_train_true_nonzero=int((~true_null[tr]).sum()), n_training_identities_role1=n1, n_training_identities_role2=n2, n_training_identities_either_role=len(set(r1[tr]) | set(r2[tr])), n_training_families_role1=len(f1), n_training_families_role2=len(f2), n_training_families_either_role=len(allf), maximum_kernel_symmetry_error=symmetry, maximum_centering_commute_error=commute, maximum_singleton_centroid_error=singleton)
        diagnostics.append(diag)
        cache[fid] = dict(query_index={int(v): i for i, v in enumerate(qids)}, prediction=prediction, mean=mean, near1=near1, near2=near2, families=allf, diag=diag)
        for j, vid in enumerate(qids):
            unions.append(dict(**dict(zip(EDGE, key)), factor_id=fid, union_request_id=factor['union_request_id'], vector_id=int(vid), prediction_block=prediction[j].tolist(), nearest_training_tanimoto_role1=float(near1[j]), nearest_training_tanimoto_role2=float(near2[j])))
    if maximum_commute_error > 1e-08 or maximum_singleton_error > 1e-12:
        raise ValueError('Centering/singleton consistency failed')
    predictions = []
    baselines = []
    for request in requests.to_dict('records'):
        fid = request['factor_id']
        item = cache[fid]
        qids = request['query_vector_ids']
        ii = np.array([positions[int(v)] for v in qids])
        qi = np.array([item['query_index'][int(v)] for v in qids])
        for gi, geometry in enumerate(GEOMETRIES):
            pred = item['prediction'][qi, gi * B:(gi + 1) * B]
            target = targets[geometry][ii]
            values, decoded, pnull, prms = losses(pred, target, u[ii])
            basevalues = {}
            for model, bp in [('centroid', np.broadcast_to(item['mean'][gi * B:(gi + 1) * B], pred.shape)), ('zero', np.zeros_like(pred))]:
                metrics, _, null, rms = losses(bp, target, u[ii])
                basevalues[model] = (metrics, null, rms)
            for j, vid in enumerate(qids):
                sc = scope.loc[int(vid)]
                d = item['diag']
                held = {sc.family_role1, sc.family_role2}
                common = dict(**dict(zip(EDGE, key)), vector_id=int(vid), fold_role1=int(request['fold_role1']), fold_role2=int(request['fold_role2']), regime=request['regime'], target_geometry=geometry, factor_id=fid, request_id=request['request_id'], plan_request_id=request['plan_request_id'], family_request_id=sc.family_request_id, n_backgrounds=B, n_train=request['n_train'], n_original_train=int(sc.n_original_train), n_family_train=int(sc.n_family_train), training_size_stratum=sc.training_size_stratum, primary_supported=bool(sc.primary_supported), sensitivity_n5plus=bool(sc.sensitivity_n5plus), true_constant_vector=bool(true_null[ii[j]]), true_stratum='true_constant' if true_null[ii[j]] else 'true_nonzero', target_units='dimensionless' if gi == 2 else 'pp', native_mse_units='dimensionless' if gi == 2 else 'pp2', nearest_training_tanimoto_role1=float(item['near1'][qi[j]]), nearest_training_tanimoto_role2=float(item['near2'][qi[j]]), n_held_families_present_either_role=len(held & item['families']), **{n: d[n] for n in ['n_training_identities_role1', 'n_training_identities_role2', 'n_training_identities_either_role', 'n_training_families_role1', 'n_training_families_role2', 'n_training_families_either_role']})
                predictions.append(dict(**common, model='uncompressed_krr', origin='native_full_krr', target_vector=target[j].tolist(), prediction_vector=pred[j].tolist(), unit0_prediction_vector=decoded[j].tolist(), unit0_predicted_null=bool(pnull[j]), predicted_centered_rms=float(prms[j]), **{n: float(values[n][j]) for n in METRICS}))
                for model, (metrics, null, rms) in basevalues.items():
                    baselines.append(dict(**common, model=model, origin='regime_training_centroid' if model == 'centroid' else 'fixed_zero', unit0_predicted_null=bool(null[j]), predicted_centered_rms=float(rms[j]), **{n: float(metrics[n][j]) for n in METRICS}))
    tables = dict(predictions=pd.DataFrame(predictions), baseline_metrics=pd.DataFrame(baselines), union_predictions=pd.DataFrame(unions), factor_diagnostics=pd.DataFrame(diagnostics))
    for name, frame in tables.items():
        frame.to_parquet(paths[name], index=False)
    meta = dict(identity=identity, output_sha256={n: sha(p) for n, p in paths.items()}, n_factors=len(factors), n_union_queries=len(unions), n_prediction_rows=len(predictions), n_baseline_rows=len(baselines), n_regime_requests=len(requests), maximum_centering_commute_error=maximum_commute_error, maximum_singleton_centroid_error=maximum_singleton_error, maximum_kernel_symmetry_error=maximum_symmetry_error, seconds=time.perf_counter() - started)
    write_json(mp, meta)
    print(f"Completed {tag}: {len(factors)} factors, {len(predictions)} predictions, {meta['seconds']:.1f}s", flush=True)
    return dict(paths=paths, metadata=mp, meta=meta, reused=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-edges', type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    before = pins(INPUTS)
    preflight = BASE / 'metadata/e65_preflight_metadata.json'
    plan = json.loads(preflight.read_text(encoding='utf-8'))
    plan_hash = sha(preflight)
    for path, digest in plan['output_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen plan output changed: ' + path)
    if sha(BASE / 'code/e65_common.py') != plan['common_code_sha256']:
        raise ValueError('Frozen common source changed')
    for path, digest in plan['source_sha256'].items():
        if before.get(path) != digest:
            raise ValueError('Plan source changed: ' + path)
    checks = formula_checks()
    pd.DataFrame(checks).to_csv(BASE / 'audit/e65_formula_checks.csv', index=False)
    if not all((x['passed'] for x in checks)):
        raise ValueError('Formula checks failed')
    molecule = pd.read_parquet(BASE / 'results/e65_molecule_families.parquet')
    canonical = molecule.canonical_smiles.tolist()
    familymap = molecule.set_index('canonical_smiles').family.to_dict()
    molindex = {s: i for i, s in enumerate(canonical)}
    v = pd.read_parquet(INPUTS[0])
    s = pd.read_parquet(BASE / 'results/e65_scope_vectors.parquet')
    r = pd.read_parquet(BASE / 'results/e65_requests.parquet')
    f = pd.read_parquet(BASE / 'results/e65_factors.parquet')
    vg = dict(tuple(v.groupby(EDGE, sort=True)))
    sg = dict(tuple(s.groupby(EDGE, sort=True)))
    rg = dict(tuple(r.groupby(EDGE, sort=True)))
    fg = dict(tuple(f.groupby(EDGE, sort=True)))
    keys = list(vg)[:args.max_edges] if args.max_edges else list(vg)
    completed = []
    with threadpool_limits(limits=1):
        gram = fingerprints(canonical)
        for j, key in enumerate(keys, 1):
            completed.append(run_edge(key, vg[key], sg[key], rg[key], fg[key], gram, molindex, familymap, before, plan_hash))
            print(f'Progress {j}/{len(keys)}; elapsed {time.perf_counter() - started:.1f}s', flush=True)
    if before != pins(INPUTS) or sha(preflight) != plan_hash:
        raise ValueError('Inputs changed during fitting')
    if args.max_edges:
        write_json(BASE / 'audit/e65_smoke_metadata.json', dict(producer_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e65_common.py'), input_sha256=before, preflight_sha256=plan_hash, n_edges=len(completed), n_factors=sum((x['meta']['n_factors'] for x in completed)), fit_seconds=sum((x['meta']['seconds'] for x in completed)), elapsed_seconds=time.perf_counter() - started, checkpoints={x['metadata'].name: sha(x['metadata']) for x in completed}, formula_checks=checks))
        print('Smoke complete; final aggregation not invoked', flush=True)
        return
    for name in ['predictions', 'baseline_metrics', 'union_predictions', 'factor_diagnostics']:
        pd.concat([pd.read_parquet(x['paths'][name]) for x in completed], ignore_index=True).to_parquet(BASE / f'results/e65_{name}.parquet', index=False)
    import aggregate_family_transport
    aggregate_family_transport.run()
    outputs = sorted((BASE / 'results').glob('*')) + [BASE / 'audit/e65_formula_checks.csv', BASE / 'audit/e65_production_checks.csv']
    production = pd.read_csv(BASE / 'audit/e65_production_checks.csv').fillna('').to_dict('records')
    metadata = dict(schema_version='E65_fit_v1', analysis='Structural-family transport versus equal-size deterministic deletion', code_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e65_common.py'), aggregator_code_sha256=sha(BASE / 'code/aggregate_family_transport.py'), input_sha256_before=before, input_sha256_after=pins(INPUTS), preflight_metadata_sha256=plan_hash, output_sha256=pins(outputs), checkpoint_manifest={x['metadata'].name: sha(x['metadata']) for x in completed}, n_factorizations=sum((x['meta']['n_factors'] for x in completed)), n_logical_factors=sum((x['meta']['n_factors'] for x in completed)), n_factorizations_executed_this_invocation=sum((x['meta']['n_factors'] for x in completed if not x['reused'])), n_factors_reused_from_verified_checkpoints=sum((x['meta']['n_factors'] for x in completed if x['reused'])), n_edges_reused=sum((x['reused'] for x in completed)), n_union_queries=sum((x['meta']['n_union_queries'] for x in completed)), n_prediction_rows=sum((x['meta']['n_prediction_rows'] for x in completed)), n_baseline_rows=sum((x['meta']['n_baseline_rows'] for x in completed)), n_formula_checks=len(checks), n_formula_passed=sum((x['passed'] for x in checks)), checks=production, failures=sum((not x['passed'] for x in production)), parameters=dict(alpha=ALPHA, kernel=KERNEL, seed=20261017, factorization='Cholesky, all3targetblocks solved together, exact fixed unpenalized intercept', thin_controls='5 deterministic hashed subsets; mean losses not prediction ensemble'), versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, rdkit=rdBase.rdkitVersion), seconds=time.perf_counter() - started)
    write_json(BASE / 'metadata/e65_metadata.json', metadata)
    if metadata['failures']:
        raise ValueError('E65 production checks failed')
    print(f"E65 complete: {metadata['n_prediction_rows']} predictions, {metadata['n_factorizations']} factors, {len(production)} checks, {metadata['seconds']:.1f}s", flush=True)
if __name__ == '__main__':
    main()
