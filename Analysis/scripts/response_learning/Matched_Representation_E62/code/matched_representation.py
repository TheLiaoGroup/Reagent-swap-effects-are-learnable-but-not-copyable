"""E62 matched finite-codebook and continuous response predictions, raw delta.

Targets are whole within-edge response vectors on known backgrounds. Only
training substrate vectors enter numeric fitting; endpoint lineage is read
without arm yields to enforce disjoint condition cells. No E60/E61 predictions
or targets are read. Edge checkpoints commit only after all part hashes exist.
"""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
ROOT = ANALYSIS.parent
UP = ANALYSIS / 'Existing_Data_Mining_2026-10-16'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
MODELS = ['centroid', 'codebook_continuous', 'codebook_hard', 'codebook_mixture', 'pca_continuous']
CAPACITIES = [2, 3]
SEED, ALPHA = (20261017, 10.0)
INPUTS = [ANALYSIS / 'Full_Edge_Corpus_E59/results/full_edge_response_corpus.parquet', ANALYSIS / 'Full_Edge_Corpus_E59/results/full_edge_response_appearances.parquet', UP / 'processed/molecular_features.parquet', Path(__file__), UP / 'code/structure_context_geometry.py', UP / 'code/both_new_structure_context.py', UP / 'code/structural_kernel_context.py']

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pins(paths: list[Path]) -> dict:
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()

def identity_fold(value: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{SEED}|geometry_molecule|{value}'.encode()).digest()[:8], 'little') % 5

def holdout(a: np.ndarray, b: np.ndarray, f1: int, f2: int) -> tuple[np.ndarray, np.ndarray, int]:
    h1, h2 = (np.array([identity_fold(v) for v in a]), np.array([identity_fold(v) for v in b]))
    test, train = ((h1 == f1) & (h2 == f2), (h1 != f1) & (h2 != f2))
    before = int(train.sum())
    held = set(a[test]) | set(b[test])
    train &= ~np.isin(a, list(held)) & ~np.isin(b, list(held))
    return (train, test, before - int(train.sum()))

def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative probabilities summing to one."""
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values, axis=1)[:, ::-1]
    excess = np.cumsum(ordered, axis=1) - 1.0
    j = np.arange(1, values.shape[1] + 1)
    active = ordered - excess / j > 0
    rho = active.sum(axis=1) - 1
    theta = excess[np.arange(len(values)), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0.0)

def orthogonal_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = 1e-10 * max(float(singular[0]) if len(singular) else 0.0, 1.0)
    rank = int(np.count_nonzero(singular > tolerance))
    return (vt[:rank].T, singular, tolerance)

def kernel_operator(kernel: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, float]:
    mean = kernel.mean(axis=0)
    grand = float(mean.mean())
    centered = kernel - mean[None, :] - mean[:, None] + grand
    cross_centered = cross - cross.mean(axis=1)[:, None] - mean[None, :] + grand
    values, vectors = eigh(centered, check_finite=False)
    if float(values.min()) < -1e-08:
        raise ValueError('Tanimoto centered kernel is not positive semidefinite')
    operator = cross_centered @ vectors / (values + ALPHA) @ vectors.T
    return (operator, float(values.min()))

def kernel_predict(operator: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean = target.mean(axis=0)
    return mean + operator @ (target - mean)

def fit_capacity(train_y: np.ndarray, operator: np.ndarray, nominal_k: int) -> dict:
    """Only training responses and the structure-only kernel operator enter."""
    scale = float(np.sqrt(np.mean(train_y * train_y)))
    scale_used = scale if scale > 0 else 1.0
    y = train_y / scale_used
    n_unique = len(np.unique(y, axis=0))
    requested_k = min(nominal_k, n_unique)
    if requested_k == 1:
        centers = y[:1].copy()
        inertia = 0.0
    else:
        fit = KMeans(n_clusters=requested_k, random_state=SEED, n_init=20, max_iter=300, algorithm='lloyd').fit(y)
        centers, inertia = (fit.cluster_centers_, float(fit.inertia_))
    centers = np.unique(centers, axis=0)
    distances = np.square(y[:, None, :] - centers[None, :, :]).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    occupied = np.bincount(labels, minlength=len(centers)) > 0
    centers = centers[occupied]
    labels = np.argmin(np.square(y[:, None, :] - centers[None, :, :]).sum(axis=2), axis=1)
    origin = centers.mean(axis=0)
    basis, dictionary_singular, dictionary_tolerance = orthogonal_basis(centers - origin)
    data_mean = y.mean(axis=0)
    pca_basis, data_singular, data_tolerance = orthogonal_basis(y - data_mean)
    effective_rank = min(nominal_k - 1, basis.shape[1], pca_basis.shape[1])
    basis = basis[:, :effective_rank]
    pca_basis = pca_basis[:, :effective_rank]
    coordinates = kernel_predict(operator, (y - origin) @ basis)
    continuous = origin + coordinates @ basis.T
    hard_labels = np.argmin(np.square(continuous[:, None, :] - centers[None, :, :]).sum(axis=2), axis=1)
    scores = kernel_predict(operator, np.eye(len(centers))[labels])
    probabilities = project_simplex(scores)
    mixture = probabilities @ centers
    pca = data_mean + kernel_predict(operator, (y - data_mean) @ pca_basis) @ pca_basis.T
    centroid = np.broadcast_to(data_mean, continuous.shape).copy()
    return dict(predictions={'centroid': centroid * scale_used, 'codebook_continuous': continuous * scale_used, 'codebook_hard': centers[hard_labels] * scale_used, 'codebook_mixture': mixture * scale_used, 'pca_continuous': pca * scale_used}, centers=centers * scale_used, origin=origin * scale_used, basis=basis, pca_basis=pca_basis, training_mean=data_mean * scale_used, coordinates=coordinates * scale_used, probabilities=probabilities, hard_labels=hard_labels, cluster_counts=np.bincount(labels, minlength=len(centers)), effective_k=len(centers), requested_k_after_unique=requested_k, n_unique_train_vectors=n_unique, effective_rank=effective_rank, scale=scale, scale_used=scale_used, inertia_scaled=inertia, dictionary_singular=dictionary_singular, data_singular=data_singular, dictionary_tolerance=dictionary_tolerance, data_tolerance=data_tolerance)

def fingerprint_gram(canonical: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False, useBondTypes=True, includeRingMembership=True, countSimulation=False)
    molecules = [Chem.MolFromSmiles(value) for value in canonical]
    if any((m is None for m in molecules)):
        raise ValueError('A canonical substrate failed to parse')
    fps = [generator.GetFingerprint(m) for m in molecules]
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=float)

def panel_data(corpus: pd.DataFrame, feature_map: dict) -> tuple[list[dict], pd.DataFrame]:
    panels, ledger = ([], [])
    for number, (key, group) in enumerate(corpus.groupby(EDGE, sort=True)):
        backgrounds = sorted(group.background_fingerprint.unique())
        matrix = group.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta').reindex(columns=backgrounds)
        complete = matrix.notna().all(axis=1)
        identity = group.drop_duplicates('substrate_pair_key').set_index('substrate_pair_key').reindex(matrix.index)
        a, b = (identity.sub_1_smiles.map(feature_map), identity.sub_2_smiles.map(feature_map))
        actual = identity.sub_1_smiles.notna() & identity.sub_2_smiles.notna() & identity.sub_1_smiles.ne('') & identity.sub_2_smiles.ne('')
        valid = a.notna() & b.notna()
        keep = complete & actual & valid
        count = int(keep.sum())
        status = 'eligible' if len(backgrounds) >= 3 and count >= 20 else 'single_background' if len(backgrounds) == 1 else 'fewer_than3backgrounds' if len(backgrounds) < 3 else 'fewer_than20valid_vectors'
        ledger.append(dict(**dict(zip(EDGE, key)), edge_number=number, n_corpus_response_rows=len(group), n_backgrounds=len(backgrounds), n_substrate_keys=len(matrix), n_complete_vectors=int(complete.sum()), n_incomplete_vectors=int((~complete).sum()), n_complete_single_role=int((complete & ~actual).sum()), n_complete_invalid_structure=int((complete & actual & ~valid).sum()), n_candidate_vectors=count, n_complete_allzero_vectors=int(np.count_nonzero(np.linalg.norm(matrix.loc[complete].to_numpy(), axis=1) == 0)), panel_status=status))
        if status != 'eligible':
            continue
        pairs = matrix.index[keep].to_numpy()
        response_ids = group.pivot(index='substrate_pair_key', columns='background_fingerprint', values='response_row_id').reindex(index=pairs, columns=backgrounds).to_numpy(np.int64)
        panels.append(dict(key=key, number=number, backgrounds=backgrounds, pairs=pairs, y=matrix.loc[keep].to_numpy(float), a=a[keep].to_numpy(str), b=b[keep].to_numpy(str), response_ids=response_ids, vector_ids=response_ids.min(axis=1)))
    return (panels, pd.DataFrame(ledger))

def lineage_cells(panels: list[dict]) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    use = np.unique(np.concatenate([p['response_ids'].ravel() for p in panels]))
    columns = ['response_row_id', 'reaction_type', 'substrate_pair_key', 'condition0', 'condition1', 'swap_id']
    appearance = pd.read_parquet(INPUTS[1], columns=columns)
    appearance = appearance.loc[appearance.response_row_id.isin(use)]
    arms = pd.concat([appearance[['response_row_id', 'reaction_type', 'substrate_pair_key', f'condition{arm}']].rename(columns={f'condition{arm}': 'condition_id'}) for arm in [0, 1]], ignore_index=True)
    index = pd.MultiIndex.from_frame(arms[['reaction_type', 'substrate_pair_key', 'condition_id']])
    codes, cells = pd.factorize(index, sort=True)
    arms['cell_id'] = codes
    arm_map = arms[['response_row_id', 'cell_id']].drop_duplicates().groupby('response_row_id').cell_id.agg(list).to_dict()
    if set(use) != set(arm_map):
        raise ValueError('Incomplete canonical response to condition-cell lineage')
    cell_frame = cells.to_frame(index=False)
    cell_frame.insert(0, 'cell_id', np.arange(len(cell_frame)))
    for panel in panels:
        panel['cells'] = [np.unique([cell for response in row for cell in arm_map[int(response)]]).astype(np.int64) for row in panel['response_ids']]
    return (cell_frame, arm_map)

def disjoint_training(train: np.ndarray, test: np.ndarray, cells: list[np.ndarray]) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    test_cells = np.unique(np.concatenate([cells[i] for i in np.flatnonzero(test)])) if test.any() else np.array([], dtype=np.int64)
    clean = train.copy()
    removed = 0
    for i in np.flatnonzero(train):
        if np.intersect1d(cells[i], test_cells).size:
            clean[i] = False
            removed += 1
    train_cells = np.unique(np.concatenate([cells[i] for i in np.flatnonzero(clean)])) if clean.any() else np.array([], dtype=np.int64)
    if np.intersect1d(train_cells, test_cells).size:
        raise ValueError('Training/test condition cells overlap after whitelist purge')
    return (clean, removed, train_cells, test_cells)

def score_rows(panel: dict, test: np.ndarray, f1: int, f2: int, nominal_k: int, fit: dict) -> tuple[list, list]:
    predictions, oracles = ([], [])
    y = panel['y'][test]
    for j, index in enumerate(np.flatnonzero(test)):
        common = dict(**dict(zip(EDGE, panel['key'])), vector_id=int(panel['vector_ids'][index]), substrate_pair_key=panel['pairs'][index], fold_role1=f1, fold_role2=f2, nominal_k=nominal_k, n_backgrounds=y.shape[1], effective_k=fit['effective_k'], effective_rank=fit['effective_rank'], allzero_vector=bool(np.all(y[j] == 0)))
        baseline_mse = float(np.mean((fit['predictions']['centroid'][j] - y[j]) ** 2))
        for model in MODELS:
            prediction = fit['predictions'][model][j]
            error = prediction - y[j]
            predictions.append(dict(**common, model=model, target_vector=y[j].tolist(), prediction_vector=prediction.tolist(), vector_mse_pp2=float(np.mean(error * error)), vector_mae_pp=float(np.mean(np.abs(error))), vector_bias_pp=float(np.mean(error)), centroid_mse_pp2=baseline_mse, gain_over_centroid_pp2=baseline_mse - float(np.mean(error * error)), mixture_weights=fit['probabilities'][j].tolist() if model == 'codebook_mixture' else [], predicted_coordinates=fit['coordinates'][j].tolist() if model == 'codebook_continuous' else [], hard_assigned_state=int(fit['hard_labels'][j]) if model == 'codebook_hard' else -1))
        nearest = np.argmin(np.square(fit['centers'] - y[j]).sum(axis=1))
        oracle_hard = fit['centers'][nearest]
        basis = fit['pca_basis']
        oracle_pca = fit['training_mean'] + (y[j] - fit['training_mean']) @ basis @ basis.T
        for kind, value in [('oracle_nearest_prototype', oracle_hard), ('oracle_pca_projection', oracle_pca)]:
            oracles.append(dict(**common, oracle_kind=kind, oracle_vector=value.tolist(), vector_mse_pp2=float(np.mean((value - y[j]) ** 2)), predictive=False))
    return (predictions, oracles)

def run_edge(panel: dict, gram: np.ndarray, lookup: dict, input_pins: dict) -> dict:
    key = panel['key']
    tag = f"edge_{panel['number']:04d}_" + hashlib.sha256('|'.join(key).encode()).hexdigest()[:12]
    directory = BASE / 'parts'
    paths = {name: directory / f'{tag}_{name}.parquet' for name in ['predictions', 'oracles', 'folds', 'prototypes', 'vectors', 'membership']}
    metapath = directory / f'{tag}_metadata.json'
    identity = dict(inputs=input_pins, producer_sha256=sha(Path(__file__)), edge=list(key), vector_ids_hash=array_hash(panel['vector_ids']))
    if metapath.is_file():
        meta = json.loads(metapath.read_text(encoding='utf-8'))
        if meta.get('identity') == identity and all((p.is_file() and sha(p) == meta['output_sha256'].get(name) for name, p in paths.items())):
            print(f'Reused {tag}', flush=True)
            return dict(panel=panel, paths=paths, meta=meta, metapath=metapath)
    started = time.perf_counter()
    rowids = [np.array([lookup[s] for s in panel[role]]) for role in ['a', 'b']]
    kernel = (gram[np.ix_(rowids[0], rowids[0])] + gram[np.ix_(rowids[1], rowids[1])]) / 2
    predictions, oracles, folds, prototypes, membership = ([], [], [], [], [])
    coverage = np.zeros(len(panel['y']), dtype=int)
    for f1 in range(5):
        for f2 in range(5):
            train, test, n_role_purged = holdout(panel['a'], panel['b'], f1, f2)
            train, n_cell_purged, trcells, tecells = disjoint_training(train, test, panel['cells'])
            tr, te = (np.flatnonzero(train), np.flatnonzero(test))
            held = set(panel['a'][test]) | set(panel['b'][test])
            if held & (set(panel['a'][train]) | set(panel['b'][train])):
                raise ValueError('Both-partner identity purge failed')
            if set(panel['pairs'][train]) & set(panel['pairs'][test]):
                raise ValueError('Training/test substrate pair overlap')
            coverage += test
            status = 'no_test' if not len(te) else 'insufficient_training' if len(tr) < 5 else 'scored'
            common = dict(**dict(zip(EDGE, key)), fold_role1=f1, fold_role2=f2, n_train_vectors=len(tr), n_test_vectors=len(te), n_cross_role_purged=n_role_purged, n_lineage_cell_purged=n_cell_purged, n_train_cells=len(trcells), n_test_cells=len(tecells), n_cell_overlap=0, train_vector_hash=array_hash(panel['vector_ids'][tr]), test_vector_hash=array_hash(panel['vector_ids'][te]), train_cell_hash=array_hash(trcells), test_cell_hash=array_hash(tecells), status=status)
            if status == 'scored':
                for purpose, indices in [('train', tr), ('test', te)]:
                    membership.extend((dict(**dict(zip(EDGE, key)), fold_role1=f1, fold_role2=f2, purpose=purpose, vector_id=int(panel['vector_ids'][i])) for i in indices))
                operator, minimum_eigenvalue = kernel_operator(kernel[np.ix_(tr, tr)], kernel[np.ix_(te, tr)])
                for nominal_k in CAPACITIES:
                    fit = fit_capacity(panel['y'][tr], operator, nominal_k)
                    if not np.allclose(fit['probabilities'].sum(axis=1), 1, atol=1e-12) or np.any(fit['probabilities'] < -1e-12):
                        raise ValueError('Mixture probabilities invalid')
                    if any((not np.isfinite(value).all() for value in fit['predictions'].values())):
                        raise ValueError('A predictive vector is nonfinite')
                    ps, os = score_rows(panel, test, f1, f2, nominal_k, fit)
                    predictions.extend(ps)
                    oracles.extend(os)
                    folds.append(dict(**common, nominal_k=nominal_k, effective_k=fit['effective_k'], effective_rank=fit['effective_rank'], n_unique_train_vectors=fit['n_unique_train_vectors'], scale_rms_pp=fit['scale'], scale_used_pp=fit['scale_used'], kmeans_inertia_scaled=fit['inertia_scaled'], minimum_kernel_eigenvalue=minimum_eigenvalue, dictionary_origin=fit['origin'].tolist(), training_mean=fit['training_mean'].tolist(), dictionary_basis=fit['basis'].tolist(), pca_basis=fit['pca_basis'].tolist(), dictionary_singular=fit['dictionary_singular'].tolist(), data_singular=fit['data_singular'].tolist(), dictionary_rank_tolerance=fit['dictionary_tolerance'], data_rank_tolerance=fit['data_tolerance']))
                    for state, center in enumerate(fit['centers']):
                        prototypes.append(dict(**dict(zip(EDGE, key)), fold_role1=f1, fold_role2=f2, nominal_k=nominal_k, state=state, center_vector_pp=center.tolist(), n_training_assigned=int(fit['cluster_counts'][state])))
            else:
                for nominal_k in CAPACITIES:
                    folds.append(dict(**common, nominal_k=nominal_k))
    if not np.all(coverage == 1):
        raise ValueError('25 test groups do not partition eligible vectors')
    vector_rows = [dict(**dict(zip(EDGE, key)), vector_id=int(panel['vector_ids'][i]), substrate_pair_key=panel['pairs'][i], canonical_role1=panel['a'][i], canonical_role2=panel['b'][i], backgrounds=panel['backgrounds'], response_row_ids=panel['response_ids'][i].tolist(), condition_cell_ids=panel['cells'][i].tolist(), target_vector=panel['y'][i].tolist(), n_backgrounds=len(panel['backgrounds']), allzero_vector=bool(np.all(panel['y'][i] == 0))) for i in range(len(panel['y']))]
    for name, rows in [('predictions', predictions), ('oracles', oracles), ('folds', folds), ('prototypes', prototypes), ('vectors', vector_rows), ('membership', membership)]:
        pd.DataFrame(rows).to_parquet(paths[name], index=False)
    meta = dict(identity=identity, output_sha256={name: sha(p) for name, p in paths.items()}, n_vectors=len(vector_rows), n_scored_vectors=len({row['vector_id'] for row in predictions}), n_nonempty_trainable_folds=sum((row['status'] == 'scored' for row in folds)) // 2, n_prediction_rows=len(predictions), n_oracle_rows=len(oracles), seconds=time.perf_counter() - started)
    metapath.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"Completed {tag}: {len(vector_rows)} vectors, {meta['n_nonempty_trainable_folds']} folds, {meta['seconds']:.1f}s", flush=True)
    return dict(panel=panel, paths=paths, meta=meta, metapath=metapath)

def aggregate(completed: list[dict], ledger: pd.DataFrame) -> None:
    for name in ['predictions', 'oracles', 'folds', 'prototypes', 'vectors', 'membership']:
        pd.concat([pd.read_parquet(item['paths'][name]) for item in completed], ignore_index=True).to_parquet(BASE / f'results/e62_{name}.parquet', index=False)
    predictions = pd.read_parquet(BASE / 'results/e62_predictions.parquet')
    grouping = [*EDGE, 'nominal_k', 'model']
    edge_scores = predictions.groupby(grouping, as_index=False).agg(n_test_vectors=('vector_id', 'size'), n_allzero_vectors=('allzero_vector', 'sum'), mse_pp2=('vector_mse_pp2', 'mean'), mae_pp=('vector_mae_pp', 'mean'), bias_pp=('vector_bias_pp', 'mean'), centroid_mse_pp2=('centroid_mse_pp2', 'mean'), gain_over_centroid_pp2=('gain_over_centroid_pp2', 'mean'))
    edge_scores['rmse_pp'] = np.sqrt(edge_scores.mse_pp2)
    grid = ledger[EDGE].merge(pd.DataFrame([(k, model) for k in CAPACITIES for model in MODELS], columns=['nominal_k', 'model']), how='cross')
    edge_scores = grid.merge(edge_scores, on=grouping, how='left', validate='one_to_one')
    for column in ['n_test_vectors', 'n_allzero_vectors']:
        edge_scores[column] = edge_scores[column].fillna(0).astype(int)
    edge_scores['supported'] = edge_scores.n_test_vectors.gt(0)
    edge_scores.to_csv(BASE / 'results/e62_edge_metrics_1548.csv', index=False)
    class_rows, summary = ([], [])
    for (k, model), all_edges in edge_scores.groupby(['nominal_k', 'model'], sort=True):
        supported = all_edges.loc[all_edges.supported]
        for reaction in sorted(ledger.reaction_type.unique()):
            group = supported.loc[supported.reaction_type.eq(reaction)]
            class_rows.append(dict(nominal_k=k, model=model, reaction_type=reaction, n_test_edges=len(group), n_test_vectors=int(group.n_test_vectors.sum()), n_allzero_vectors=int(group.n_allzero_vectors.sum()), supported=bool(len(group)), rmse_pp=float(np.sqrt(group.mse_pp2.mean())), **{c: float(group[c].mean()) for c in ['mse_pp2', 'mae_pp', 'bias_pp', 'centroid_mse_pp2', 'gain_over_centroid_pp2']}))
        rows = predictions.loc[predictions.nominal_k.eq(k) & predictions.model.eq(model)]
        pooled = dict(mse_pp2=float(rows.vector_mse_pp2.mean()), rmse_pp=float(np.sqrt(rows.vector_mse_pp2.mean())), mae_pp=float(rows.vector_mae_pp.mean()), bias_pp=float(rows.vector_bias_pp.mean()), centroid_mse_pp2=float(rows.centroid_mse_pp2.mean()), gain_over_centroid_pp2=float(rows.gain_over_centroid_pp2.mean()))
        classes = pd.DataFrame(class_rows).query('nominal_k == @k and model == @model and supported')
        columns = list(pooled)
        for weighting, scores in [('pooled', pooled), ('equal_edge', {c: float(supported[c].mean()) for c in columns}), ('equal_class_edge_balanced', {c: float(classes[c].mean()) for c in columns})]:
            scores['rmse_pp'] = float(np.sqrt(scores['mse_pp2']))
            summary.append(dict(nominal_k=k, model=model, weighting=weighting, n_test_vectors=len(rows), n_test_edges=len(supported), n_test_classes=classes.reaction_type.nunique(), **scores))
    pd.DataFrame(class_rows).to_csv(BASE / 'results/e62_class_metrics_20.csv', index=False)
    pd.DataFrame(summary).to_csv(BASE / 'results/e62_summary.csv', index=False)
    oracle = pd.read_parquet(BASE / 'results/e62_oracles.parquet')
    oracle.groupby([*EDGE, 'nominal_k', 'oracle_kind'], as_index=False).agg(n_test_vectors=('vector_id', 'size'), mse_pp2=('vector_mse_pp2', 'mean')).to_csv(BASE / 'results/e62_oracle_edge_metrics.csv', index=False)
    scored = predictions.loc[predictions.nominal_k.eq(2) & predictions.model.eq('centroid')].groupby(EDGE).size().rename('n_scored_vectors').reset_index()
    ledger = ledger.merge(scored, on=EDGE, how='left', validate='one_to_one')
    ledger['n_scored_vectors'] = ledger.n_scored_vectors.fillna(0).astype(int)
    ledger['complete_coverage'] = ledger.panel_status.eq('eligible') & ledger.n_scored_vectors.eq(ledger.n_candidate_vectors)
    ledger.to_csv(BASE / 'results/e62_support_bridge_1548.csv', index=False)
    ledger.groupby('reaction_type', as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_corpus_response_rows=('n_corpus_response_rows', 'sum'), n_eligible_edges=('complete_coverage', 'sum'), n_candidate_vectors=('n_candidate_vectors', lambda v: int(v.loc[ledger.loc[v.index, 'panel_status'].eq('eligible')].sum())), n_scored_vectors=('n_scored_vectors', 'sum')).to_csv(BASE / 'results/e62_support_classes_20.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-edges', type=int, default=0, help='Smoke subset; does not write final aggregate outputs')
    args = parser.parse_args()
    started = time.perf_counter()
    for folder in ['results', 'parts', 'audit', 'metadata', 'reports', 'figures']:
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    before = pins(INPUTS)
    corpus_columns = ['response_row_id', *EDGE, 'background_fingerprint', 'substrate_pair_key', 'delta', 'sub_1_smiles', 'sub_2_smiles']
    corpus = pd.read_parquet(INPUTS[0], columns=corpus_columns)
    features = pd.read_parquet(INPUTS[2], columns=['smiles', 'canonical_smiles', 'parse_valid'])
    if features.smiles.duplicated().any():
        raise ValueError('Nonunique molecular lookup')
    feature_map = features.loc[features.parse_valid & features.canonical_smiles.notna()].set_index('smiles').canonical_smiles.to_dict()
    panels, ledger = panel_data(corpus, feature_map)
    if len(ledger) != 1550 or ledger.reaction_type.nunique() != 20:
        raise ValueError('Complete edge/class accounting changed')
    print(f"Eligible panels: {len(panels)}; vectors: {sum((len(p['y']) for p in panels))}; allzero: {sum((np.count_nonzero(np.all(p['y'] == 0, axis=1)) for p in panels))}", flush=True)
    cells, _ = lineage_cells(panels)
    if args.max_edges:
        panels = panels[:args.max_edges]
    canonical = sorted({v for panel in panels for role in ['a', 'b'] for v in panel[role]})
    lookup = {v: i for i, v in enumerate(canonical)}
    completed = []
    with threadpool_limits(limits=1):
        gram = fingerprint_gram(canonical)
        for i, panel in enumerate(panels):
            completed.append(run_edge(panel, gram, lookup, before))
            print(f'Progress {i + 1}/{len(panels)}; elapsed {time.perf_counter() - started:.1f}s', flush=True)
            gc.collect()
    if args.max_edges:
        print('Smoke subset complete; final aggregates not written', flush=True)
        return
    cells.to_parquet(BASE / 'results/e62_condition_cells.parquet', index=False)
    aggregate(completed, ledger)
    pred = pd.read_parquet(BASE / 'results/e62_predictions.parquet')
    fold = pd.read_parquet(BASE / 'results/e62_folds.parquet')
    support = pd.read_csv(BASE / 'results/e62_support_bridge_1548.csv')
    checks = [dict(check='all1548edges20classes', passed=len(support) == 1550 and support.reaction_type.nunique() == 20), dict(check='all_predictions_finite', passed=bool(np.isfinite(pred.vector_mse_pp2).all())), dict(check='all_ten_models_same_vector_support', passed=bool(pred.groupby('vector_id').size().eq(10).all()) and (not pred.duplicated(['vector_id', 'nominal_k', 'model']).any())), dict(check='all_candidate_vectors_scored', passed=bool(support.loc[support.panel_status.eq('eligible'), 'complete_coverage'].all())), dict(check='all_folds_cell_disjoint', passed=bool(fold.n_cell_overlap.eq(0).all())), dict(check='all_zero_vectors_preserved', passed=pred.loc[pred.model.eq('centroid') & pred.nominal_k.eq(2), 'allzero_vector'].sum() == 9924)]
    after = pins(INPUTS)
    checks.append(dict(check='upstream_inputs_unchanged', passed=before == after))
    pd.DataFrame(checks).to_csv(BASE / 'audit/e62_production_checks.csv', index=False)
    outputs = sorted((BASE / 'results').glob('*')) + [BASE / 'audit/e62_production_checks.csv']
    metadata = dict(analysis='E62 matched discrete and continuous raw response-vector representations', code_sha256=sha(Path(__file__)), input_sha256_before=before, input_sha256_after=after, output_sha256=pins(outputs), checks=checks, failures=sum((not c['passed'] for c in checks)), elapsed_seconds=time.perf_counter() - started, n_edges=len(panels), n_classes=pred.reaction_type.nunique(), n_vectors=pred.vector_id.nunique(), n_allzero_vectors=int(pred.loc[pred.model.eq('centroid') & pred.nominal_k.eq(2), 'allzero_vector'].sum()), n_prediction_rows=len(pred), n_trainable_nonempty_folds=int(fold.status.eq('scored').sum() // 2), checkpoint_manifest={item['metapath'].name: sha(item['metapath']) for item in completed}, parameters=dict(capacities=CAPACITIES, alpha_summed_loss=ALPHA, KMeans_n_init=20, seed=SEED, target='Raw pp delta vector; train-only common RMS scaling; allzero vectors retained', fingerprint='Morgan radius2 2048bit includeChirality=False; additive role Tanimoto kernel', split='E48 two-partner canonical identity 5x5 groups, cross-role purge, exact-cell whitelist purge', weighting='per-vector mean over backgrounds; equal_edge averages edge MSE; equal_class_edge_balanced averages edge MSE within class then classes. All displayed RMSE=sqrt(the corresponding aggregated MSE), unlike E60 mean-edge-RMSE.'), limitations=['Known edge and known background coordinates; support-conditioned complete panels, no new-background validation.', 'Hard codebook and continuous decoder share fitted coordinates; mixture learns assignments from the same training labels and has different constraints.', 'Oracle prototype assignment and PCA projection use test outcomes solely as representation lower bounds, not predictive models.', 'Finite-state conditional means can vary continuously; this comparison does not identify latent chemical state counts or mechanisms.', 'Historical observations are reused; no new experiment, temporal, scaffold or causal validation.'], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, rdkit=rdBase.rdkitVersion))
    (BASE / 'metadata/e62_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if metadata['failures']:
        raise SystemExit('E62 production checks failed')
    print(f"E62 complete: {metadata['n_vectors']} vectors, {metadata['n_trainable_nonempty_folds']} folds, {metadata['elapsed_seconds']:.1f}s", flush=True)
if __name__ == '__main__':
    main()
