"""Fixed augmented operators, changed local labels; immutable E66 allocations."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
E66 = ANALYSIS / 'Family_Information_Recovery_E66'
sys.path.insert(0, str(E66 / 'code'))
from e66_common import EDGE, E62, E65, ALPHA, center_and_unit, edge_tag, factor_id, pins, sha, write_json, payload
RAW_VARIANTS = ['full', 'level_only', 'modulation_only', 'zero_local', 'permutation_p0', 'permutation_p1', 'permutation_p2', 'frozen_family', 'literal_zero']
UNIT_VARIANTS = ['full', 'zero_local', 'permutation_p0', 'permutation_p1', 'permutation_p2', 'frozen_family', 'literal_zero']

def permutation(ids, request, allocation, p):
    return sorted(ids, key=lambda i: (hashlib.sha256(payload(['E67_local_permutation_v1', request, allocation, p, int(i)])).digest(), i))

def source_paths():
    return [Path(__file__), E66 / 'metadata/e66_plan_metadata.json', E66 / 'metadata/e66_fit_metadata.json', E66 / 'code/e66_common.py', E66 / 'results/e66_scope_vectors.parquet', E62 / 'results/e62_vectors.parquet', E65 / 'results/e65_family_requests.parquet', E65 / 'results/e65_molecule_families.parquet', E62 / 'results/e62_support_bridge_1548.csv', E66 / 'results/e66_support_classes_20.csv']

def freeze():
    for name in ['parts', 'results', 'reports', 'metadata', 'audit', 'figures']:
        (BASE / name).mkdir(exist_ok=True, parents=True)
    paths = source_paths()
    planmeta = json.loads((E66 / 'metadata/e66_plan_metadata.json').read_text())
    fitmeta = json.loads((E66 / 'metadata/e66_fit_metadata.json').read_text())
    scope = pd.read_parquet(E66 / 'results/e66_scope_vectors.parquet')
    selected = scope.loc[scope.common8]
    rows = []
    for key, g in selected.groupby(EDGE, sort=True):
        tag = edge_tag(key)
        ap = E66 / f'parts/{tag}_allocations.parquet'
        pp = E66 / f'parts/{tag}_predictions.parquet'
        mp = E66 / f'parts/{tag}_metadata.json'
        if sha(ap) != planmeta['output_sha256'][str(ap.relative_to(ANALYSIS.parent))]:
            raise ValueError('Frozen E66 allocation changed')
        if sha(mp) != fitmeta['checkpoint_manifest'][mp.name]:
            raise ValueError('Frozen E66 fit metadata changed')
        m = json.loads(mp.read_text())
        if sha(pp) != m['output_sha256']['predictions']:
            raise ValueError('Frozen E66 predictions changed')
        paths.extend([ap, pp, mp])
        alloc = pd.read_parquet(ap)
        alloc = alloc.loc[alloc.budget.eq('8') & alloc.family_request_id.isin(g.family_request_id)]
        for a in alloc.itertuples(index=False):
            ids = sorted(map(int, a.calibration_vector_ids))
            if len(ids) != 8 or a.n_family_train < 1:
                raise ValueError('Not exact-eight family-supported allocation')
            row = a._asdict()
            row.update({f'permutation_p{p}_source_ids': permutation(ids, a.family_request_id, a.allocation, p) for p in range(3)})
            rows.append(row)
    plan = pd.DataFrame(rows)
    if len(selected) != 24221 or len(plan) != 43518 or plan.n_query.sum() != 72663:
        raise ValueError('Frozen common8 membership differs')
    dest = BASE / 'results/e67_plan.parquet'
    before = pins(paths)
    plan.to_parquet(dest, index=False)
    write_json(BASE / 'metadata/e67_preflight_metadata.json', dict(source_sha256=before, plan_sha256=sha(dest), code_sha256=sha(Path(__file__)), n_query_vectors=len(selected), n_requests=len(plan), n_query_order_rows=int(plan.n_query.sum()), n_edges=selected[EDGE].drop_duplicates().shape[0], n_scored_classes=selected.reaction_type.nunique(), outcome_access='Only saved allocation, support and structure fields; prediction/outcome files pinned but not read', permutations='p0/p1/p2 per DESIGN_E67; fixed points retained'))
    print('Frozen E67 plan: 43518 request-orders / 72663 query-orders', flush=True)

def weights(kernel, cross):
    n = len(kernel)
    col = kernel.mean(0)
    grand = col.mean()
    kc = kernel - col[None, :] - col[:, None] + grand
    kc = (kc + kc.T) * 0.5
    kc.flat[::n + 1] += ALPHA
    qc = cross - cross.mean(1)[:, None] - col[None, :] + grand
    w = cho_solve(cho_factor(kc, lower=True, check_finite=False), qc.T, check_finite=False).T
    w += (1 - w.sum(1))[:, None] / n
    return w

def run_edge(key, v, plan, family, gram, molindex, before):
    started = time.perf_counter()
    tag = edge_tag(key)
    dest = BASE / f'parts/{tag}_predictions.parquet'
    mp = BASE / f'parts/{tag}_metadata.json'
    identity = dict(edge=list(key), source_sha256=before, plan_sha256=sha(BASE / 'results/e67_plan.parquet'), code_sha256=sha(Path(__file__)))
    if mp.exists():
        old = json.loads(mp.read_text())
        if old['identity'] == identity and sha(dest) == old['prediction_sha256']:
            print(f'Reused {tag}', flush=True)
            return mp
    v = v.sort_values('vector_id').reset_index(drop=True)
    positions = {int(vid): i for i, vid in enumerate(v.vector_id)}
    r1 = np.array([molindex[s] for s in v.canonical_role1])
    r2 = np.array([molindex[s] for s in v.canonical_role2])
    kernel = 0.5 * (gram[np.ix_(r1, r1)] + gram[np.ix_(r2, r2)])
    y = np.stack(v.target_vector.to_numpy())
    centered, unit, _, _ = center_and_unit(y)
    level = y - centered
    width = y.shape[1]
    fam = family.set_index('family_request_id')
    old = pd.read_parquet(E66 / f'parts/{tag}_predictions.parquet')
    old8 = old.loc[old.learner.eq('joint') & old.budget.eq('8')]
    ref = {(r.family_request_id, r.allocation, int(r.vector_id)): r for r in old8.itertuples(index=False)}
    frozen = {(r.family_request_id, int(r.vector_id)): r for r in old.loc[old.learner.eq('joint') & old.budget.eq('0') & old.allocation.eq('r0')].itertuples(index=False)}
    operators = {}
    for a in plan.itertuples(index=False):
        tf = list(map(int, fam.loc[a.family_request_id].family_train_vector_ids))
        local = list(map(int, a.calibration_vector_ids))
        train = sorted(tf + local)
        fid = factor_id(key, train)
        item = operators.setdefault(fid, dict(train=train, query=set()))
        item['query'].update(map(int, a.query_vector_ids))
    cache = {}
    max_ref = max_identity = max_orthogonal = max_weight = 0.0
    for fid, item in operators.items():
        tr = np.array([positions[i] for i in item['train']])
        qids = sorted(item['query'])
        query = np.array([positions[i] for i in qids])
        w = weights(kernel[np.ix_(tr, tr)], kernel[np.ix_(query, tr)])
        max_weight = max(max_weight, float(np.max(np.abs(w.sum(1) - 1))))
        cache[fid] = dict(w=w, raw=w @ y[tr], unit=w @ unit[tr], trainpos={i: j for j, i in enumerate(item['train'])}, querypos={i: j for j, i in enumerate(qids)})
    rows = []
    for a in plan.itertuples(index=False):
        localids = list(map(int, a.calibration_vector_ids))
        train = sorted(list(map(int, fam.loc[a.family_request_id].family_train_vector_ids)) + localids)
        fid = factor_id(key, train)
        item = cache[fid]
        loc = np.array([positions[i] for i in localids])
        loccol = np.array([item['trainpos'][i] for i in localids])
        for vid in map(int, a.query_vector_ids):
            qi = item['querypos'][vid]
            wlocal = item['w'][qi, loccol]
            raw = item['raw'][qi]
            u = item['unit'][qi]
            reference = ref[a.family_request_id, a.allocation, vid]
            max_ref = max(max_ref, float(np.max(np.abs(raw - reference.prediction_raw))), float(np.max(np.abs(u - reference.prediction_unit))))
            z = raw - wlocal @ y[loc]
            zu = u - wlocal @ unit[loc]
            rawpred = dict(full=raw, zero_local=z, level_only=z + wlocal @ level[loc], modulation_only=z + wlocal @ centered[loc], frozen_family=np.asarray(frozen[a.family_request_id, vid].prediction_raw), literal_zero=np.zeros(width))
            unitpred = dict(full=u, zero_local=zu, frozen_family=np.asarray(frozen[a.family_request_id, vid].prediction_unit), literal_zero=np.zeros(width))
            for p in range(3):
                source = np.array([positions[int(i)] for i in getattr(a, f'permutation_p{p}_source_ids')])
                rawpred[f'permutation_p{p}'] = z + wlocal @ y[source]
                unitpred[f'permutation_p{p}'] = zu + wlocal @ unit[source]
            ident = raw - rawpred['level_only'] - rawpred['modulation_only'] + z
            aa, bb = (rawpred['level_only'] - z, rawpred['modulation_only'] - z)
            max_identity = max(max_identity, float(np.max(np.abs(ident))))
            max_orthogonal = max(max_orthogonal, abs(float(2 * np.mean(aa * bb))))
            for geometry, predictions in [('raw_delta', rawpred), ('centered_unit_rms_zeroembed', unitpred)]:
                for variant, prediction in predictions.items():
                    rows.append(dict(**dict(zip(EDGE, key)), vector_id=vid, family_request_id=a.family_request_id, allocation=a.allocation, allocation_id=a.allocation_id, factor_id=fid, target_geometry=geometry, variant=variant, n_train=len(train), n_calibration=8, n_backgrounds=width, prediction=prediction.tolist()))
    if max_ref > 1e-08 or max_identity > 1e-09 or max_orthogonal > 1e-08 or (max_weight > 1e-12):
        raise ValueError(f'E67 algebra/ref check failed {(max_ref, max_identity, max_orthogonal, max_weight)}')
    pd.DataFrame(rows).to_parquet(dest, index=False)
    write_json(mp, dict(identity=identity, prediction_sha256=sha(dest), n_operators=len(operators), n_predictions=len(rows), maximum_E66_ref_error=max_ref, maximum_prediction_identity_error=max_identity, maximum_orthogonal_cross_term=max_orthogonal, maximum_weight_sum_error=max_weight, seconds=time.perf_counter() - started))
    print(f'{tag}: {len(operators)} operators / {len(rows)} predictions / {time.perf_counter() - started:.1f}s', flush=True)
    return mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--max-edges', type=int, default=0)
    args = parser.parse_args()
    if not (BASE / 'metadata/e67_preflight_metadata.json').exists():
        freeze()
    if args.freeze_only:
        return
    started = time.perf_counter()
    meta = json.loads((BASE / 'metadata/e67_preflight_metadata.json').read_text())
    before = meta['source_sha256']
    for path, digest in before.items():
        if sha(ANALYSIS.parent / path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    if sha(Path(__file__)) != meta['code_sha256'] or sha(BASE / 'results/e67_plan.parquet') != meta['plan_sha256']:
        raise ValueError('Frozen E67 producer or plan changed')
    if rdBase.rdkitVersion != '2025.03.2':
        raise ValueError('RDKit version differs')
    vectors = pd.read_parquet(E62 / 'results/e62_vectors.parquet')
    family = pd.read_parquet(E65 / 'results/e65_family_requests.parquet')
    plan = pd.read_parquet(BASE / 'results/e67_plan.parquet')
    molecules = pd.read_parquet(E65 / 'results/e65_molecule_families.parquet').canonical_smiles.tolist()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False, useBondTypes=True, includeRingMembership=True, countSimulation=False)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in molecules]
    gram = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    molindex = {s: i for i, s in enumerate(molecules)}
    vg, fg = (dict(tuple(vectors.groupby(EDGE))), dict(tuple(family.groupby(EDGE))))
    groups = list(plan.groupby(EDGE, sort=True))
    if args.max_edges:
        groups = groups[:args.max_edges]
    manifests = []
    with threadpool_limits(limits=1):
        for key, p in groups:
            manifests.append(run_edge(tuple(key), vg[key], p, fg[key], gram, molindex, before))
    if any((sha(ANALYSIS.parent / p) != h for p, h in before.items())):
        raise ValueError('Frozen source changed during fitting')
    ms = [json.loads(p.read_text()) for p in manifests]
    output = dict(source_sha256=before, preflight_sha256=sha(BASE / 'metadata/e67_preflight_metadata.json'), code_sha256=sha(Path(__file__)), checkpoint_manifest={p.name: sha(p) for p in manifests}, n_edges=len(ms), n_operators=sum((m['n_operators'] for m in ms)), n_predictions=sum((m['n_predictions'] for m in ms)), maximum_E66_ref_error=max((m['maximum_E66_ref_error'] for m in ms)), maximum_prediction_identity_error=max((m['maximum_prediction_identity_error'] for m in ms)), maximum_orthogonal_cross_term=max((m['maximum_orthogonal_cross_term'] for m in ms)), seconds=time.perf_counter() - started, failures=0)
    write_json(BASE / ('audit/e67_smoke_metadata.json' if args.max_edges else 'metadata/e67_fit_metadata.json'), output)
    print(json.dumps({k: v for k, v in output.items() if k not in ['source_sha256', 'checkpoint_manifest']}), flush=True)
if __name__ == '__main__':
    main()
