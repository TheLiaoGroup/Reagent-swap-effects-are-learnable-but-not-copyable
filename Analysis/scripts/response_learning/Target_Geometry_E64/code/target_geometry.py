"""E64 fixed-support target retraining; only this analysis directory is writable."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits
BASE = Path(__file__).resolve().parents[1]
ANALYSIS, ROOT = (BASE.parent, BASE.parent.parent)
E62, E63 = (ANALYSIS / 'Matched_Representation_E62', ANALYSIS / 'Response_Information_Decomposition_E63')
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
FOLD = [*EDGE, 'fold_role1', 'fold_role2']
GEOMETRIES = ['centered_delta', 'centered_unit_rms_zeroembed']
FIVE = ['centroid', 'codebook_continuous', 'codebook_hard', 'codebook_mixture', 'pca_continuous']
MODELS = FIVE + ['uncompressed_krr', 'zero'] + ['raw_postmapped__' + m for m in FIVE]
INPUTS = [E62 / f'results/e62_{n}.parquet' for n in ['vectors', 'membership', 'folds', 'predictions', 'condition_cells']] + [E62 / 'results/e62_support_bridge_1548.csv', E62 / 'results/e62_support_classes_20.csv', E62 / 'metadata/e62_metadata.json', E62 / 'code/matched_representation.py', E63 / 'results/e63_vector_components.parquet', E63 / 'metadata/e63_metadata.json', Path(__file__)]
HELPER_SHA = 'e821296e5f468affcc8346f48b16feb3f3058ee937b608b13bc40dd2764d2ac2'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pins(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def native_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=native_json), encoding='utf-8')

def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()

def center_and_unit(values):
    values = np.asarray(values, dtype=float)
    constant = np.all(values == values[:, :1], axis=1)
    mean = values[:, 0] + (values - values[:, :1]).mean(axis=1)
    centered = values - mean[:, None]
    centered[constant] = 0.0
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    if not np.array_equal(rms == 0, constant):
        raise ValueError('Nonconstant vector underflowed to zero centered RMS')
    unit = np.zeros_like(centered)
    np.divide(centered, rms[:, None], out=unit, where=~constant[:, None])
    return (centered, unit, constant, rms)

def load_helper():
    if sha(INPUTS[8]) != HELPER_SHA:
        raise ValueError('Frozen E62 helper changed')
    spec = importlib.util.spec_from_file_location('e64_frozen_e62_helpers', INPUTS[8])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def formula_checks():
    rows = []

    def check(name, value):
        rows.append(dict(check=name, passed=bool(value)))
    a = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [-1, 0, 1], [1e-12, 0, -1e-12]], float)
    c, u, null, rms = center_and_unit(a)
    check('exact_constants_zero_embedded', np.array_equal(null, [1, 1, 0, 0]) and np.all(u[:2] == 0) and np.all(c[:2] == 0))
    check('tiny_nonconstant_retained', rms[3] > 0 and np.allclose(np.mean(u[2:] ** 2, axis=1), 1))
    check('unit_nonnull_zero_prediction_loss_one', np.allclose(np.mean(u[2:] ** 2, axis=1), 1, atol=1e-14))
    check('both_null_loss_zero', np.all(np.mean((u[:2] - 0) ** 2, axis=1) == 0))
    _, opposite, _, _ = center_and_unit(-a)
    check('opposite_direction_loss_four', np.allclose(np.mean((u[2:] - opposite[2:]) ** 2, axis=1), 4))
    check('native_conditional_mean_not_renormalized', np.allclose(np.mean((0.25 * u[2:] - u[2:]) ** 2, axis=1), 0.75 ** 2))
    _, dec, _, _ = center_and_unit(0.25 * u)
    check('common_decoder_removes_nonzero_length', np.allclose(dec, u, atol=1e-14))
    check('shift_invariant_centered_and_unit', np.allclose(center_and_unit(a + 3)[0][:3], c[:3], atol=1e-14) and np.allclose(center_and_unit(a + 3)[1][:3], u[:3], atol=1e-14))
    return rows

def validate_inputs(before):
    metadata = json.loads(INPUTS[7].read_text(encoding='utf-8'))
    for p in INPUTS[:7]:
        if metadata['output_sha256'].get(str(p.relative_to(ROOT))) != before[str(p.relative_to(ROOT))]:
            raise ValueError(f'E62 snapshot mismatch {p.name}')
    if metadata['code_sha256'] != HELPER_SHA:
        raise ValueError('E62 fit helper identity mismatch')
    e63 = json.loads(INPUTS[10].read_text(encoding='utf-8'))
    if e63['output_sha256'].get(str(INPUTS[9].relative_to(ROOT))) != before[str(INPUTS[9].relative_to(ROOT))]:
        raise ValueError('E63 component snapshot mismatch')

def run_edge(key, vectors, membership, source_folds, raw, gram, lookup, helper, before):
    tag = 'edge_' + hashlib.sha256('|'.join(key).encode()).hexdigest()[:16]
    paths = {name: BASE / 'parts' / f'{tag}_{name}.parquet' for name in ['predictions', 'folds', 'prototypes']}
    mp = BASE / 'parts' / f'{tag}_metadata.json'
    identity = dict(inputs=before, producer_sha256=sha(Path(__file__)), edge=list(key), vector_ids_hash=array_hash(vectors.vector_id))
    if mp.exists():
        meta = json.loads(mp.read_text(encoding='utf-8'))
        if meta.get('identity') == identity and all((p.exists() and sha(p) == meta['output_sha256'].get(n) for n, p in paths.items())):
            print(f'Reused {tag}', flush=True)
            return dict(paths=paths, metadata_path=mp, meta=meta)
    started = time.perf_counter()
    y = np.vstack(vectors.target_vector)
    vector_ids = vectors.vector_id.to_numpy(np.int64)
    centered, unit, true_null, true_rms = center_and_unit(y)
    target = {GEOMETRIES[0]: centered, GEOMETRIES[1]: unit}
    index = {int(v): i for i, v in enumerate(vectors.vector_id)}
    role1 = np.array([lookup[x] for x in vectors.canonical_role1])
    role2 = np.array([lookup[x] for x in vectors.canonical_role2])
    kernel = (gram[np.ix_(role1, role1)] + gram[np.ix_(role2, role2)]) / 2
    rawmap = {(int(k), m): g.set_index('vector_id').prediction_vector.to_dict() for (k, m), g in raw.groupby(['nominal_k', 'model'], sort=False)}
    members = {(int(a), int(b), purpose): g.vector_id.to_numpy(np.int64) for (a, b, purpose), g in membership.groupby(['fold_role1', 'fold_role2', 'purpose'], sort=False)}
    prediction_rows, fold_rows, prototypes = ([], [], [])
    n_operator = n_capacity = n_uncompressed = 0
    coverage = np.zeros(len(vectors), int)
    common_cols = [*FOLD, 'n_train_vectors', 'n_test_vectors', 'n_cross_role_purged', 'n_lineage_cell_purged', 'n_train_cells', 'n_test_cells', 'n_cell_overlap', 'train_vector_hash', 'test_vector_hash', 'train_cell_hash', 'test_cell_hash', 'status']
    for sf in source_folds.loc[source_folds.nominal_k.eq(2)].to_dict('records'):
        f1, f2 = (int(sf['fold_role1']), int(sf['fold_role2']))
        common = {c: sf[c] for c in common_cols}
        if sf['status'] != 'scored':
            for geometry in GEOMETRIES:
                for k in [2, 3]:
                    fold_rows.append(dict(**common, geometry=geometry, nominal_k=k))
            continue
        trids, teids = (members[f1, f2, 'train'], members[f1, f2, 'test'])
        tr = np.array([index[int(v)] for v in trids])
        te = np.array([index[int(v)] for v in teids])
        coverage[te] += 1
        trcells = np.unique(np.concatenate(vectors.iloc[tr].condition_cell_ids.to_numpy()))
        tecells = np.unique(np.concatenate(vectors.iloc[te].condition_cell_ids.to_numpy()))
        if array_hash(trids) != sf['train_vector_hash'] or array_hash(teids) != sf['test_vector_hash'] or array_hash(trcells) != sf['train_cell_hash'] or (array_hash(tecells) != sf['test_cell_hash']):
            raise ValueError('Exact E62 membership/cell hash changed')
        if np.intersect1d(trcells, tecells).size or (set(vectors.iloc[te].canonical_role1) | set(vectors.iloc[te].canonical_role2)) & (set(vectors.iloc[tr].canonical_role1) | set(vectors.iloc[tr].canonical_role2)):
            raise ValueError('Identity/cell disjointness failed')
        operator, min_eigen = helper.kernel_operator(kernel[np.ix_(tr, tr)], kernel[np.ix_(te, tr)])
        n_operator += 1
        for geometry in GEOMETRIES:
            yt = target[geometry]
            full = helper.kernel_predict(operator, yt[tr])
            n_uncompressed += 1
            fit_prefix = tag + f'|{f1}|{f2}|{geometry}'
            for k in [2, 3]:
                fit = helper.fit_capacity(yt[tr], operator, k)
                n_capacity += 1
                fold_rows.append(dict(**common, geometry=geometry, nominal_k=k, effective_k=fit['effective_k'], effective_rank=fit['effective_rank'], n_unique_train_vectors=fit['n_unique_train_vectors'], n_train_true_constant=int(true_null[tr].sum()), n_train_true_nonzero=int((~true_null[tr]).sum()), scale_rms=fit['scale'], scale_used=fit['scale_used'], kmeans_inertia_scaled=fit['inertia_scaled'], minimum_kernel_eigenvalue=min_eigen, dictionary_origin=fit['origin'].tolist(), training_mean=fit['training_mean'].tolist(), dictionary_basis=fit['basis'].tolist(), pca_basis=fit['pca_basis'].tolist(), dictionary_singular=fit['dictionary_singular'].tolist(), data_singular=fit['data_singular'].tolist(), dictionary_rank_tolerance=fit['dictionary_tolerance'], data_rank_tolerance=fit['data_tolerance']))
                for state, center in enumerate(fit['centers']):
                    prototypes.append(dict(**{c: common[c] for c in FOLD}, geometry=geometry, nominal_k=k, state=state, center_vector=center.tolist(), n_training_assigned=int(fit['cluster_counts'][state])))
                allpred = dict(fit['predictions'], uncompressed_krr=full, zero=np.zeros_like(full))
                for m in FIVE:
                    rp = np.vstack([rawmap[k, m][int(i)] for i in teids])
                    cp, up, _, _ = center_and_unit(rp)
                    allpred['raw_postmapped__' + m] = cp if geometry == GEOMETRIES[0] else up
                for model, p in allpred.items():
                    if not np.isfinite(p).all():
                        raise ValueError('Nonfinite prediction')
                    _, decoded, pnull, prms = center_and_unit(p)
                    nmse = np.mean((p - yt[te]) ** 2, axis=1)
                    nmae = np.mean(np.abs(p - yt[te]), axis=1)
                    dmse = np.mean((decoded - unit[te]) ** 2, axis=1)
                    dmae = np.mean(np.abs(decoded - unit[te]), axis=1)
                    origin = 'frozen_raw_postmapped' if model.startswith('raw_postmapped__') else 'fixed_zero' if model == 'zero' else 'native_fit'
                    independent = model in ['uncompressed_krr', 'zero']
                    fit_id = 'fixed_zero|' + geometry if model == 'zero' else fit_prefix + '|uncompressed_krr' if model == 'uncompressed_krr' else fit_prefix + f'|raw|{k}|{model}' if origin == 'frozen_raw_postmapped' else fit_prefix + f'|capacity{k}'
                    for j, i in enumerate(te):
                        prediction_rows.append(dict(**{c: common[c] for c in FOLD}, geometry=geometry, nominal_k=k, model=model, origin=origin, vector_id=int(vector_ids[i]), n_backgrounds=y.shape[1], true_constant_vector=bool(true_null[i]), true_stratum='true_constant' if true_null[i] else 'true_nonzero', target_vector=yt[i].tolist(), prediction_vector=p[j].tolist(), unit0_prediction_vector=decoded[j].tolist(), native_vector_mse=float(nmse[j]), native_vector_mae=float(nmae[j]), unit0_vector_mse=float(dmse[j]), unit0_vector_mae=float(dmae[j]), predicted_constant_vector=bool(pnull[j]), unit0_predicted_null=bool(pnull[j]), predicted_centered_rms=float(prms[j]), effective_k=fit['effective_k'] if model in FIVE else -1, effective_rank=fit['effective_rank'] if model in FIVE else -1, mixture_weights=fit['probabilities'][j].tolist() if model == 'codebook_mixture' else [], predicted_coordinates=fit['coordinates'][j].tolist() if model == 'codebook_continuous' else [], hard_assigned_state=int(fit['hard_labels'][j]) if model == 'codebook_hard' else -1, fit_identity=fit_id, k_independent=independent, k_prediction_reuse=bool(independent and k == 3)))
    if not np.all(coverage == 1):
        raise ValueError('Frozen E62 test coverage changed')
    for name, rows in [('predictions', prediction_rows), ('folds', fold_rows), ('prototypes', prototypes)]:
        pd.DataFrame(rows).to_parquet(paths[name], index=False)
    meta = dict(identity=identity, output_sha256={n: sha(p) for n, p in paths.items()}, n_vectors=len(vectors), n_prediction_rows=len(prediction_rows), n_nonempty_folds=n_operator, n_structural_operator_calls=n_operator, n_codebook_capacity_calls=n_capacity, n_uncompressed_target_calls=n_uncompressed, seconds=time.perf_counter() - started)
    write_json(mp, meta)
    print(f"Completed {tag}: {len(vectors)} vectors, {n_operator} folds, {meta['seconds']:.1f}s", flush=True)
    return dict(paths=paths, metadata_path=mp, meta=meta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-edges', type=int, default=0)
    args = parser.parse_args()
    for folder in ['parts', 'results', 'audit', 'metadata', 'reports']:
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    before = pins(INPUTS)
    validate_inputs(before)
    boundaries = formula_checks()
    pd.DataFrame(boundaries).to_csv(BASE / 'audit/e64_formula_checks.csv', index=False)
    if not all((x['passed'] for x in boundaries)):
        raise ValueError('E64 formula smoke checks failed')
    helper = load_helper()
    vectors = pd.read_parquet(INPUTS[0])
    membership = pd.read_parquet(INPUTS[1])
    folds = pd.read_parquet(INPUTS[2])
    raw = pd.read_parquet(INPUTS[3])
    if len(vectors) != 25956 or len(membership) != 429980 or len(folds) != 4650 or (len(raw) != 259560):
        raise ValueError('Frozen E62 source dimensions changed')
    canonical = sorted(set(vectors.canonical_role1) | set(vectors.canonical_role2))
    lookup = {v: i for i, v in enumerate(canonical)}
    vg = dict(tuple(vectors.groupby(EDGE, sort=True)))
    mg = dict(tuple(membership.groupby(EDGE, sort=True)))
    fg = dict(tuple(folds.groupby(EDGE, sort=True)))
    rg = dict(tuple(raw.groupby(EDGE, sort=True)))
    keys = list(vg)[:args.max_edges] if args.max_edges else list(vg)
    started = time.perf_counter()
    completed = []
    print(f'E64 start {len(keys)}/93 edges; 25,956 fixed vectors; 2 geometries; 12 prediction families', flush=True)
    with threadpool_limits(limits=1):
        gram = helper.fingerprint_gram(canonical)
        for n, key in enumerate(keys, 1):
            completed.append(run_edge(key, vg[key], mg[key], fg[key], rg[key], gram, lookup, helper, before))
            print(f'Progress {n}/{len(keys)}; elapsed {time.perf_counter() - started:.1f}s', flush=True)
    if before != pins(INPUTS):
        raise ValueError('Inputs changed during E64')
    if args.max_edges:
        write_json(BASE / 'audit/e64_smoke_metadata.json', dict(code_sha256=sha(Path(__file__)), input_sha256=before, edges=len(completed), formula_checks=boundaries, parts={x['metadata_path'].name: sha(x['metadata_path']) for x in completed}, elapsed_seconds=time.perf_counter() - started))
        print('E64 smoke completed; final aggregate not written', flush=True)
        return
    for name in ['predictions', 'folds', 'prototypes']:
        pd.concat([pd.read_parquet(x['paths'][name]) for x in completed], ignore_index=True).to_parquet(BASE / f'results/e64_{name}.parquet', index=False)
    vector_chunks = []
    for _, g in vectors.groupby('n_backgrounds', sort=True):
        c, u, null, rms = center_and_unit(np.vstack(g.target_vector))
        q = g.copy()
        q['centered_delta_target'] = [x.tolist() for x in c]
        q['centered_unit_rms_zeroembed_target'] = [x.tolist() for x in u]
        q['true_constant_vector'] = null
        q['true_centered_rms_pp'] = rms
        vector_chunks.append(q)
    pd.concat(vector_chunks).sort_index().to_parquet(BASE / 'results/e64_vectors.parquet', index=False)
    membership.to_parquet(BASE / 'results/e64_membership.parquet', index=False)
    pd.read_parquet(INPUTS[4]).to_parquet(BASE / 'results/e64_condition_cells.parquet', index=False)
    import summarize_target_geometry as aggregate
    aggregate.run(BASE, INPUTS)
    outputs = sorted((BASE / 'results').glob('*')) + [BASE / 'audit/e64_formula_checks.csv', BASE / 'audit/e64_production_checks.csv']
    checks = pd.read_csv(BASE / 'audit/e64_production_checks.csv').to_dict('records')
    metadata = dict(analysis='E64 fixed-support independently trained target geometries', code_sha256=sha(Path(__file__)), aggregator_code_sha256=sha(BASE / 'code/summarize_target_geometry.py'), input_sha256_before=before, input_sha256_after=pins(INPUTS), output_sha256=pins(outputs), checks=checks, failures=sum((not x['passed'] for x in checks)), n_formula_checks=len(boundaries), n_formula_passed=sum((x['passed'] for x in boundaries)), n_edges=93, n_scored_classes=14, n_registered_edges=1550, n_registered_classes=20, n_vectors=25956, n_allzero_vectors=9924, n_prediction_rows=sum((x['meta']['n_prediction_rows'] for x in completed)), n_unique_nonempty_folds=sum((x['meta']['n_nonempty_folds'] for x in completed)), n_structural_operator_calls=sum((x['meta']['n_structural_operator_calls'] for x in completed)), n_codebook_capacity_calls=sum((x['meta']['n_codebook_capacity_calls'] for x in completed)), n_uncompressed_target_calls=sum((x['meta']['n_uncompressed_target_calls'] for x in completed)), checkpoint_manifest={x['metadata_path'].name: sha(x['metadata_path']) for x in completed}, parameters=dict(geometries=GEOMETRIES, models=MODELS, nominal_k=[2, 3], alpha_summed_loss=10, seed=20261017, KMeans_n_init=20, membership='Exact E62 saved vectors/membership/folds/cells; no eligibility recalculation', target_transform='Stable center x0+mean(x-x0); exact constant maps zero; positive centered RMS divides without epsilon', unit0_decoder='Same center/unit/null map on every prediction; native unit risk retains conditional-mean length', K_independent='uncompressed_krr fits once/target/fold and repeats K; zero never fits; repeated K is not extra evidence'), limitations=['Native risks in different target geometries have different units/estimands and cannot be directly compared by magnitude or ratios.', 'Zero-embedded shape risk and common unit0 decoder loss include null decisions; neither is pure angular accuracy.', 'Raw-postmapped reference changes output decoder as well as training target; common unit0 endpoint shares final action.', 'Known edge/background, dependent historical complete-panel identity holdout; no scaffold/time/new-background/mechanism/causal/state-count claim.'], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__), elapsed_seconds=time.perf_counter() - started)
    write_json(BASE / 'metadata/e64_metadata.json', metadata)
    if metadata['failures']:
        raise SystemExit('E64 production checks failed')
    print(f"E64 complete: {metadata['n_prediction_rows']} predictions; {len(checks)}/{len(checks)} checks; {metadata['elapsed_seconds']:.1f}s", flush=True)
if __name__ == '__main__':
    main()
