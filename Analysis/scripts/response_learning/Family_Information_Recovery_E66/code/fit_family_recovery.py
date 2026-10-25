"""Fit frozen E66 operators edge by edge, reusing E65 endpoint arrays."""
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
from e66_common import *

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
    kernel = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    cross = np.array([[0.4, 0.2, 0.6], [0.8, 0.3, 0.1]])
    y = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 7.0], [-2.0, 1.0, 6.0]])
    c, u, null, _ = center_and_unit(y)
    block = np.hstack([y, c, u])
    pred, mean, _ = solve(kernel, cross, block)
    kc = kernel - kernel.mean(0) - kernel.mean(1)[:, None] + kernel.mean()
    qc = cross - cross.mean(1)[:, None] - kernel.mean(0) + kernel.mean()
    direct = mean + qc @ np.linalg.solve(kc + ALPHA * np.eye(3), block - mean)
    singleton = solve(kernel[1:2, 1:2], cross[:, 1:2], block[1:2])[0]
    return [dict(check=name, passed=bool(value)) for name, value in [('Cholesky_equals_dense_fixed_intercept', np.allclose(pred, direct, rtol=0, atol=1e-12)), ('linear_centering_commutes', np.allclose(center_and_unit(pred[:, :3])[0], pred[:, 3:6], rtol=0, atol=1e-12)), ('singleton_equals_training_block', np.array_equal(singleton, np.broadcast_to(block[1:2], singleton.shape))), ('allzero_training_predicts_zero', np.array_equal(solve(kernel, cross, np.zeros_like(block))[0], np.zeros_like(pred))), ('unit_null_embedding_zero_loss', np.allclose(np.mean(u * u, 1), (~null).astype(float), rtol=0, atol=1e-12)), ('normalization_is_not_linear_centering', not np.allclose(center_and_unit(pred[:, :3])[1], pred[:, 6:], atol=1e-08))]]

def run_edge(key, vectors, plan, gram, molindex, before, plan_hash):
    started = time.perf_counter()
    tag = edge_tag(key)
    rqpath = BASE / f'parts/{tag}_requests.parquet'
    fpath = BASE / f'parts/{tag}_factors.parquet'
    paths = {n: BASE / f'parts/{tag}_{n}.parquet' for n in ['predictions', 'union_predictions', 'factor_diagnostics']}
    mp = BASE / f'parts/{tag}_metadata.json'
    identity = dict(edge=list(key), source_sha256=before, plan_metadata_sha256=plan_hash, request_sha256=sha(rqpath), factor_sha256=sha(fpath), producer_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e66_common.py'))
    if mp.exists():
        saved = json.loads(mp.read_text(encoding='utf-8'))
        if saved.get('identity') == identity and all((p.exists() and sha(p) == saved['output_sha256'].get(n) for n, p in paths.items())):
            print(f'Reused {tag}', flush=True)
            return dict(meta=saved, metadata=mp, paths=paths, checkpoint_reused=True)
    vectors = vectors.sort_values('vector_id').reset_index(drop=True)
    positions = {int(v): i for i, v in enumerate(vectors.vector_id)}
    r1 = np.array([molindex[s] for s in vectors.canonical_role1])
    r2 = np.array([molindex[s] for s in vectors.canonical_role2])
    kernel = (gram[np.ix_(r1, r1)] + gram[np.ix_(r2, r2)]) * 0.5
    y = np.stack(vectors.target_vector.to_numpy())
    width = y.shape[1]
    c, u, true_constant, _ = center_and_unit(y)
    block = np.hstack([y, c, u])
    requests = pd.read_parquet(rqpath)
    factors = pd.read_parquet(fpath)
    refpath = E65 / f'parts/{e65_edge_tag(key)}_union_predictions.parquet'
    diagpath = E65 / f'parts/{e65_edge_tag(key)}_factor_diagnostics.parquet'
    oldmeta_path = E65 / f'parts/{e65_edge_tag(key)}_metadata.json'
    oldmanifest = json.loads((E65 / 'metadata/e65_metadata.json').read_text(encoding='utf-8'))
    if sha(oldmeta_path) != oldmanifest['checkpoint_manifest'][oldmeta_path.name]:
        raise ValueError('E65 checkpoint metadata changed')
    oldmeta = json.loads(oldmeta_path.read_text(encoding='utf-8'))
    if sha(refpath) != oldmeta['output_sha256']['union_predictions'] or sha(diagpath) != oldmeta['output_sha256']['factor_diagnostics']:
        raise ValueError('E65 reference arrays changed')
    reference = pd.read_parquet(refpath, columns=['factor_id', 'vector_id', 'prediction_block'])
    ref = {(r.factor_id, int(r.vector_id)): np.asarray(r.prediction_block, float) for r in reference.itertuples(index=False)}
    refmeans = pd.read_parquet(diagpath, columns=['factor_id', 'train_mean_block']).set_index('factor_id').train_mean_block.to_dict()
    cache = {}
    unions, diagnostics = ([], [])
    max_commute = max_singleton = max_symmetry = max_ref_mean = 0.0
    executed = reused = analytic = additional_query_factors = reused_query_rows = 0
    for f in factors.itertuples(index=False):
        trids = list(map(int, f.train_vector_ids))
        qids = list(map(int, f.union_query_vector_ids))
        tr = np.array([positions[i] for i in trids])
        query = np.array([positions[i] for i in qids])
        mean = block[tr].mean(axis=0)
        symmetry = 0.0
        oldfid = e65_factor_id(key, trids)
        available = np.array([(oldfid, i) in ref for i in qids])
        missing = ~available
        if oldfid in refmeans:
            max_ref_mean = max(max_ref_mean, float(np.max(np.abs(mean - refmeans[oldfid]))))
        if available.all():
            prediction = np.stack([ref[oldfid, i] for i in qids])
            origin = 'frozen_E65_operator'
            reused += 1
        elif len(tr) == 1:
            prediction = np.broadcast_to(block[tr[0]], (len(query), block.shape[1])).copy()
            origin = 'analytic_singleton'
            analytic += 1
        elif not f.requires_krr:
            prediction = np.broadcast_to(mean, (len(query), block.shape[1])).copy()
            origin = 'centroid_only_no_factorization'
        else:
            novel, mean, symmetry = solve(kernel[np.ix_(tr, tr)], kernel[np.ix_(query[missing], tr)], block[tr])
            prediction = np.empty((len(query), block.shape[1]))
            prediction[missing] = novel
            origin = 'new_KRR'
            executed += 1
            additional_query_factors += int(oldfid in refmeans)
        for j in np.flatnonzero(available):
            prediction[j] = ref[oldfid, qids[j]]
        reused_query_rows += int(available.sum())
        query_origins = np.where(available, 'frozen_E65_operator', origin)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite prediction')
        commute = float(np.max(np.abs(center_and_unit(prediction[:, :width])[0] - prediction[:, width:2 * width])))
        singleton = float(np.max(np.abs(prediction - block[tr[0]]))) if len(tr) == 1 else 0.0
        max_commute = max(max_commute, commute)
        max_singleton = max(max_singleton, singleton)
        max_symmetry = max(max_symmetry, symmetry)
        diagnostics.append(dict(**dict(zip(EDGE, key)), factor_id=f.factor_id, n_train=len(tr), n_union_query=len(query), n_backgrounds=width, origin=origin, reuse_e65_factor_id=f.reuse_e65_factor_id, train_vector_hash=f.train_vector_hash, union_query_vector_hash=f.union_query_vector_hash, train_mean_block=mean.tolist(), maximum_centering_commute_error=commute, maximum_singleton_centroid_error=singleton, maximum_kernel_symmetry_error=symmetry, n_train_constant=int(true_constant[tr].sum()), n_reused_E65_query_rows=int(available.sum()), n_new_query_rows=int(missing.sum()), E65_training_set_already_exists=oldfid in refmeans))
        cache[f.factor_id] = dict(prediction=prediction, mean=mean, query_index={i: j for j, i in enumerate(qids)}, origin=origin, query_origins=query_origins)
        for j, vid in enumerate(qids):
            unions.append(dict(**dict(zip(EDGE, key)), factor_id=f.factor_id, vector_id=vid, origin=str(query_origins[j]), **prediction_blocks(prediction[j], width)))
    if max_commute > 1e-08 or max_singleton > 1e-12 or max_ref_mean > 1e-10:
        raise ValueError('Centering/singleton/reference-mean check failed')
    predictions = []
    family_by_id = {r.family_request_id: r for r in plan}
    endpoint_checks = dict(n_budget0_reference=0, n_full_reference=0, n_C0_reference=0, n_n0_equal=0)
    maximum_endpoint_error = 0.0
    compare = {}
    for rq in requests.itertuples(index=False):
        item = cache.get(rq.factor_id)
        for vid in map(int, rq.query_vector_ids):
            supported = bool(rq.supported)
            pred = None
            origin = 'unsupported_n0'
            if supported:
                pred = item['mean'] if rq.learner == 'calibration_centroid' else item['prediction'][item['query_index'][vid]]
                origin = 'calibration_centroid' if rq.learner == 'calibration_centroid' else str(item['query_origins'][item['query_index'][vid]])
                if rq.learner == 'joint' and (rq.budget == '0' or rq.budget == 'full' or rq.n_calibration == 0 or (rq.n_calibration == rq.n_calibration_available)):
                    endpoint_ids = []
                    use_original = rq.budget == 'full' or (rq.n_calibration > 0 and rq.n_calibration == rq.n_calibration_available)
                    if use_original:
                        endpoint_ids = family_by_id[rq.family_request_id].original_train_vector_ids
                        endpoint_checks['n_full_reference'] += 1
                        origin = 'frozen_E65_original'
                    else:
                        endpoint_ids = family_by_id[rq.family_request_id].family_train_vector_ids
                        endpoint_checks['n_budget0_reference'] += 1
                        origin = 'frozen_E65_family'
                    endpoint_pred = ref[e65_factor_id(key, endpoint_ids), vid]
                    maximum_endpoint_error = max(maximum_endpoint_error, float(np.max(np.abs(pred - endpoint_pred))))
                    pred = endpoint_pred
                    if rq.n_calibration == 0:
                        endpoint_checks['n_C0_reference'] += int(rq.n_calibration_available == 0)
                if rq.n_family_train == 0 and rq.learner in ['joint', 'calibration_only']:
                    checkkey = (vid, rq.allocation, rq.budget)
                    if checkkey in compare:
                        if not np.array_equal(compare.pop(checkkey), pred):
                            raise ValueError('TF0 joint differs from calibration-only')
                        endpoint_checks['n_n0_equal'] += 1
                    else:
                        compare[checkkey] = pred
            row = dict(**dict(zip(EDGE, key)), vector_id=vid, family_request_id=rq.family_request_id, allocation=rq.allocation, budget=rq.budget, learner=rq.learner, factor_id=rq.factor_id, plan_request_id=rq.plan_request_id, allocation_id=rq.allocation_id, n_train=int(rq.n_train), n_calibration=int(rq.n_calibration), n_backgrounds=width, supported=supported, origin=origin)
            row.update(prediction_blocks(pred, width) if supported else dict(prediction_raw=None, prediction_centered=None, prediction_unit=None))
            predictions.append(row)
    if maximum_endpoint_error > 1e-10 or compare:
        raise ValueError('Endpoint or TF0 equality mismatch')
    for name, rows in [('predictions', predictions), ('union_predictions', unions), ('factor_diagnostics', diagnostics)]:
        pd.DataFrame(rows).to_parquet(paths[name], index=False)
    meta = dict(identity=identity, output_sha256={n: sha(p) for n, p in paths.items()}, n_factors=len(factors), n_factorizations_executed=executed, n_factors_reused_from_E65=reused, n_analytic_singletons=analytic, n_additional_query_operator_solves=additional_query_factors, n_E65_query_rows_reused=reused_query_rows, n_union_queries=len(unions), n_prediction_rows=len(predictions), maximum_centering_commute_error=max_commute, maximum_singleton_centroid_error=max_singleton, maximum_kernel_symmetry_error=max_symmetry, maximum_reference_mean_error=max_ref_mean, maximum_endpoint_error=maximum_endpoint_error, endpoint_checks=endpoint_checks, seconds=time.perf_counter() - started)
    write_json(mp, meta)
    print(f"Completed {tag}: {executed} new / {reused} reused / {analytic} singleton; {len(predictions)} rows; {meta['seconds']:.1f}s", flush=True)
    return dict(meta=meta, metadata=mp, paths=paths, checkpoint_reused=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-edges', type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    before = pins(PLAN_INPUTS)
    planpath = BASE / 'metadata/e66_plan_metadata.json'
    plan = json.loads(planpath.read_text(encoding='utf-8'))
    plan_hash = sha(planpath)
    if before != plan['source_sha256']:
        raise ValueError('Frozen source pins changed')
    if sha(BASE / 'code/e66_common.py') != plan['common_code_sha256']:
        raise ValueError('Frozen common code changed')
    for path, digest in plan['output_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen plan artifact changed: ' + path)
    checks = formula_checks()
    pd.DataFrame(checks).to_csv(BASE / 'audit/e66_formula_checks.csv', index=False)
    if not all((c['passed'] for c in checks)):
        raise ValueError('Formula check failed')
    if rdBase.rdkitVersion != '2025.03.2':
        raise ValueError('Frozen RDKit version differs')
    molecules = pd.read_parquet(E65 / 'results/e65_molecule_families.parquet').canonical_smiles.tolist()
    molindex = {s: i for i, s in enumerate(molecules)}
    vectors = pd.read_parquet(E62 / 'results/e62_vectors.parquet')
    family = pd.read_parquet(E65 / 'results/e65_family_requests.parquet')
    familygroup = dict(tuple(family.groupby(EDGE, sort=True)))
    completed = []
    groups = list(vectors.groupby(EDGE, sort=True))
    if args.max_edges:
        groups = groups[:args.max_edges]
    with threadpool_limits(limits=1):
        gram = fingerprints(molecules)
        for i, (key, g) in enumerate(groups, 1):
            completed.append(run_edge(tuple(key), g, list(familygroup[key].itertuples(index=False)), gram, molindex, before, plan_hash))
            print(f'Progress {i}/{len(groups)}; elapsed {time.perf_counter() - started:.1f}s', flush=True)
    if before != pins(PLAN_INPUTS) or sha(planpath) != plan_hash:
        raise ValueError('Frozen source or plan changed during fitting')
    counts = {n: sum((c['meta'][n] for c in completed)) for n in ['n_factors', 'n_factorizations_executed', 'n_factors_reused_from_E65', 'n_analytic_singletons', 'n_union_queries', 'n_prediction_rows', 'n_additional_query_operator_solves', 'n_E65_query_rows_reused']}
    metadata = dict(schema_version='E66_fit_v1', source_sha256_before=before, source_sha256_after=pins(PLAN_INPUTS), plan_metadata_sha256=plan_hash, code_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e66_common.py'), checkpoint_manifest={c['metadata'].name: sha(c['metadata']) for c in completed}, counts=counts, n_edges=len(completed), n_edges_reused_from_checkpoints=sum((c['checkpoint_reused'] for c in completed)), factorization_count_this_invocation=sum((c['meta']['n_factorizations_executed'] for c in completed if not c['checkpoint_reused'])), formula_checks=checks, failures=0, maximum_endpoint_error=max((c['meta']['maximum_endpoint_error'] for c in completed)), maximum_centering_commute_error=max((c['meta']['maximum_centering_commute_error'] for c in completed)), versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, rdkit=rdBase.rdkitVersion), seconds=time.perf_counter() - started)
    dest = BASE / ('audit/e66_smoke_metadata.json' if args.max_edges else 'metadata/e66_fit_metadata.json')
    write_json(dest, metadata)
    print(json.dumps(dict(counts=counts, n_edges=len(completed), seconds=metadata['seconds'])), flush=True)
if __name__ == '__main__':
    main()
