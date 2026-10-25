"""Chemistry-only E65 membership freeze. This stage never reads target columns."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from e65_common import *

def main():
    for name in ['results', 'metadata', 'audit', 'parts', 'reports']:
        (BASE / name).mkdir(exist_ok=True, parents=True)
    sources = INPUTS[:8] + INPUTS[10:]
    before = pins(sources)
    started = time.perf_counter()
    cols = [*EDGE, 'vector_id', 'canonical_role1', 'canonical_role2', 'condition_cell_ids']
    v = pd.read_parquet(INPUTS[0], columns=cols)
    membership = pd.read_parquet(INPUTS[1])
    folds = pd.read_parquet(INPUTS[2])
    canonical = sorted(set(v.canonical_role1) | set(v.canonical_role2))
    molecules = []
    if rdBase.rdkitVersion != '2025.03.2':
        raise ValueError('Frozen RDKit version differs')
    for smi in canonical:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid canonical molecule')
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
        family = scaffold or '__ACYCLIC__'
        molecules.append(dict(canonical_smiles=smi, scaffold_smiles=scaffold, family=family, family_id=identifier(['E65_family_v1', family]), acyclic=not bool(scaffold)))
    molecule = pd.DataFrame(molecules)
    family = molecule.set_index('canonical_smiles').family.to_dict()
    v['family_role1'] = v.canonical_role1.map(family)
    v['family_role2'] = v.canonical_role2.map(family)
    famrequests = []
    requests = []
    scopes = []
    factorrows = {}
    fold_count = 0
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))

    def add_request(key, f1, f2, regime, train, query, lineage, held1='', held2=''):
        train = np.sort(np.asarray(train, dtype=np.int64))
        query = np.sort(np.asarray(query, dtype=np.int64))
        fid = factor_id(key, train)
        oid = operator_id(fid, query)
        logical = identifier(['E65_regime_request_v1', *key, int(f1), int(f2), regime, lineage])
        row = dict(**dict(zip(EDGE, key)), fold_role1=int(f1), fold_role2=int(f2), regime=regime, replicate=int(regime[-1]) if regime.startswith('thin_') else -1, plan_request_id=logical, lineage_id=lineage, family_role1=held1, family_role2=held2, train_vector_ids=train.tolist(), query_vector_ids=query.tolist(), train_vector_hash=array_hash(train), query_vector_hash=array_hash(query), n_train=len(train), n_query=len(query), factor_id=fid, request_id=oid)
        requests.append(row)
        if fid not in factorrows:
            factorrows[fid] = dict(**dict(zip(EDGE, key)), factor_id=fid, train_vector_ids=train.tolist(), train_vector_hash=array_hash(train), n_train=len(train), queries=set(), regimes=set(), request_ids=set(), plan_request_ids=set())
        f = factorrows[fid]
        if f['train_vector_ids'] != train.tolist():
            raise ValueError('Factor hash collision')
        f['queries'].update(map(int, query))
        f['regimes'].add(regime)
        f['request_ids'].add(oid)
        f['plan_request_ids'].add(logical)
    vg = dict(tuple(v.groupby(EDGE, sort=True)))
    mg = dict(tuple(membership.groupby(EDGE, sort=True)))
    frozen = folds.loc[folds.nominal_k.eq(2) & folds.status.eq('scored')]
    for key, g in vg.items():
        indexed = g.set_index('vector_id')
        m = mg[key]
        for (f1, f2), fm in m.groupby(['fold_role1', 'fold_role2'], sort=True):
            tr = np.sort(fm.loc[fm.purpose.eq('train'), 'vector_id'].to_numpy(np.int64))
            te = np.sort(fm.loc[fm.purpose.eq('test'), 'vector_id'].to_numpy(np.int64))
            if not len(te):
                continue
            if not len(tr):
                raise ValueError('Original empty training set')
            fold_count += 1
            tg = indexed.loc[tr]
            qg = indexed.loc[te]
            sf = frozen.loc[(frozen[EDGE] == pd.Series(key, index=EDGE)).all(axis=1) & frozen.fold_role1.eq(f1) & frozen.fold_role2.eq(f2)].iloc[0]
            if array_hash(tr) != sf.train_vector_hash or array_hash(te) != sf.test_vector_hash:
                raise ValueError('Frozen original IDs changed')
            trcells = np.unique(np.concatenate(tg.condition_cell_ids.to_numpy()))
            tecells = np.unique(np.concatenate(qg.condition_cell_ids.to_numpy()))
            if np.intersect1d(trcells, tecells).size:
                raise ValueError('Condition-cell overlap')
            if (set(tg.canonical_role1) | set(tg.canonical_role2)) & (set(qg.canonical_role1) | set(qg.canonical_role2)):
                raise ValueError('Cross-role identity overlap')
            lineage = identifier(['E65_original_fold_v1', *key, int(f1), int(f2)])
            add_request(key, f1, f2, 'original', tr, te, lineage)
            rankings = []
            for replicate in range(5):
                rankings.append(sorted(map(int, tr), key=lambda i: (hashlib.sha256(payload([20261017 + replicate, 'E65_thin', *key, int(f1), int(f2), i])).digest(), i)))
            bysize = {}
            trainfamilies = set(tg.family_role1) | set(tg.family_role2)
            for (s1, s2), part in qg.groupby(['family_role1', 'family_role2'], sort=True):
                held = {s1, s2}
                keep = ~tg.family_role1.isin(held) & ~tg.family_role2.isin(held)
                ft = tr[keep.to_numpy()]
                qt = np.sort(part.index.to_numpy(np.int64))
                n = len(ft)
                frid = identifier(['E65_family_request_v1', *key, int(f1), int(f2), s1, s2])
                famrequests.append(dict(**dict(zip(EDGE, key)), fold_role1=int(f1), fold_role2=int(f2), family_request_id=frid, family_role1=s1, family_role2=s2, query_vector_ids=qt.tolist(), query_vector_hash=array_hash(qt), n_query=len(qt), original_train_vector_ids=tr.tolist(), original_train_vector_hash=array_hash(tr), n_original_train=len(tr), family_train_vector_ids=ft.tolist(), family_train_vector_hash=array_hash(ft), n_family_train=n, training_size_stratum=size_stratum(n), support_status='supported' if n else 'no_family_training'))
                for vid in qt:
                    row = part.loc[vid]
                    scopes.append(dict(**dict(zip(EDGE, key)), vector_id=int(vid), fold_role1=int(f1), fold_role2=int(f2), canonical_role1=row.canonical_role1, canonical_role2=row.canonical_role2, family_role1=s1, family_role2=s2, family_request_id=frid, n_original_train=len(tr), n_family_train=n, training_size_stratum=size_stratum(n), primary_supported=n >= 1, sensitivity_n5plus=n >= 5, no_training_reason='' if n else 'no_family_training', original_family_role1_present=s1 in trainfamilies, original_family_role2_present=s2 in trainfamilies, either_acyclic='__ACYCLIC__' in held))
                if n:
                    if set(ft) - set(tr) or (set(tg.loc[ft].family_role1) | set(tg.loc[ft].family_role2)) & held:
                        raise ValueError('Family exclusion failed')
                    add_request(key, f1, f2, 'family', ft, qt, frid, s1, s2)
                    bysize.setdefault(n, []).extend(map(int, qt))
            for n, query in sorted(bysize.items()):
                for replicate in range(5):
                    train = rankings[replicate][:n]
                    if n == len(tr) and set(train) != set(tr):
                        raise ValueError('Full-size control differs')
                    add_request(key, f1, f2, f'thin_{replicate}', train, query, identifier(['E65_count_request_v1', *key, int(f1), int(f2), int(n)]))
    for row in factorrows.values():
        row['union_query_vector_ids'] = sorted(row.pop('queries'))
        row['union_query_vector_hash'] = array_hash(row['union_query_vector_ids'])
        row['n_union_query'] = len(row['union_query_vector_ids'])
        row['requested_regimes'] = sorted(row.pop('regimes'))
        row['request_ids'] = sorted(row['request_ids'])
        row['plan_request_ids'] = sorted(row['plan_request_ids'])
        row['union_request_id'] = operator_id(row['factor_id'], row['union_query_vector_ids'])
        row['checkpoint_path'] = 'parts/' + edge_tag([row[x] for x in EDGE]) + '_union_predictions.parquet'
        row['checkpoint_status'] = 'planned'
    fs = pd.DataFrame(famrequests)
    rq = pd.DataFrame(requests)
    ft = pd.DataFrame(factorrows.values())
    scope = pd.DataFrame(scopes).sort_values('vector_id').reset_index(drop=True)
    counts = dict(n_molecules=len(molecule), n_families=molecule.family.nunique(), n_acyclic=int(molecule.acyclic.sum()), n_original_folds=fold_count, n_family_requests=len(fs), n_nonempty_family_requests=int(fs.n_family_train.gt(0).sum()), n_control_count_requests=rq.loc[rq.regime.eq('thin_0')].shape[0], n_regime_requests=len(rq), n_exact_requests=rq.request_id.nunique(), n_factors=len(ft), n_union_queries=int(ft.n_union_query.sum()), max_union_query=int(ft.n_union_query.max()), max_train=int(ft.n_train.max()), n_vectors=len(scope), n_primary=int(scope.primary_supported.sum()), n_sensitivity=int(scope.sensitivity_n5plus.sum()), n_n1=int(scope.n_family_train.eq(1).sum()), n_n2to4=int(scope.n_family_train.between(2, 4).sum()), n_n0=int(scope.n_family_train.eq(0).sum()), n_prediction_rows=3 * (len(scope) + 6 * int(scope.primary_supported.sum())))
    expected = dict(n_molecules=872, n_families=115, n_acyclic=40, n_original_folds=1984, n_family_requests=16020, n_nonempty_family_requests=15787, n_control_count_requests=12185, n_regime_requests=78696, n_exact_requests=76799, n_factors=72928, n_union_queries=175471, max_union_query=80, max_train=642, n_vectors=25956, n_primary=25520, n_sensitivity=24970, n_n1=114, n_n2to4=436, n_n0=436, n_prediction_rows=537228)
    for k, wanted in expected.items():
        check(k, counts[k] == wanted, f'{counts[k]} expected {wanted}')
    perregime = rq.groupby('regime').factor_id.nunique().to_dict()
    check('per_regime_factor_counts', perregime == dict(original=1984, family=12477, thin_0=12184, thin_1=12183, thin_2=12180, thin_3=12184, thin_4=12184), perregime)
    check('single_original_test_coverage', scope.vector_id.nunique() == len(v) and set(scope.vector_id) == set(v.vector_id))
    check('requests_exact_unique_lineage', not rq.plan_request_id.duplicated().any())
    check('request_query_coverage', rq.n_query.sum() == 179076, int(rq.n_query.sum()))
    check('all_92_edges_14classes_primary', scope.loc[scope.primary_supported, EDGE].drop_duplicates().shape[0] == 93 and scope.loc[scope.primary_supported].reaction_type.nunique() == 14)
    check('sources_unchanged', before == pins(sources))
    if not all((x['passed'] for x in checks)):
        raise ValueError(str([x for x in checks if not x['passed']]))
    for name, frame in zip(PLAN_NAMES, [molecule, fs, rq, ft, scope]):
        frame.to_parquet(BASE / f'results/e65_{name}.parquet', index=False)
    pd.DataFrame(checks).to_csv(BASE / 'audit/e65_preflight_checks.csv', index=False)
    files = [BASE / f'results/e65_{name}.parquet' for name in PLAN_NAMES] + [BASE / 'audit/e65_preflight_checks.csv']
    write_json(BASE / 'metadata/e65_preflight_metadata.json', dict(schema_version='E65_plan_v1', code_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e65_common.py'), source_sha256=before, output_sha256=pins(files), counts=counts, per_regime_factor_counts=perregime, checks=checks, failures=0, rdkit_version=rdBase.rdkitVersion, seconds=time.perf_counter() - started, outcome_access='Only structure/identity/cell columns read. No target, prediction, or outcome column read.', family_id_payload=['E65_family_v1', 'literal_family_string'], hash_rule='compact ensure_ascii JSON to ASCII; sorted IDs int64 little endian for array hashes'))
    print(json.dumps(dict(counts=counts, checks_passed=len(checks), seconds=time.perf_counter() - started)), flush=True)
if __name__ == '__main__':
    main()
