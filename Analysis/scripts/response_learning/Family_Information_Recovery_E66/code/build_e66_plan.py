"""Freeze E66 allocations, information costs and deduplicated operators without outcomes."""
from __future__ import annotations
import time
import pandas as pd
from e66_common import *

def main():
    started = time.perf_counter()
    for name in ['parts', 'results', 'metadata', 'audit']:
        (BASE / name).mkdir(exist_ok=True, parents=True)
    before = pins(PLAN_INPUTS)
    vectors = pd.read_parquet(PLAN_INPUTS[0], columns=[*EDGE, 'vector_id', 'canonical_role1', 'canonical_role2', 'n_backgrounds', 'condition_cell_ids'])
    cells = pd.read_parquet(PLAN_INPUTS[1]).rename(columns={'0': 'reaction_type', '1': 'substrate_pair_key', '2': 'condition_id'})
    obs = pd.read_parquet(OLD / 'processed/observations.parquet', columns=['reaction_type', 'substrate_pair_key', 'condition_id', 'observation_id', 'eligible_for_matching'])
    obs = obs.loc[obs.eligible_for_matching]
    if obs.observation_id.duplicated().any():
        raise ValueError('Raw observation IDs not unique')
    joined = cells.merge(obs, on=['reaction_type', 'substrate_pair_key', 'condition_id'], how='left', validate='one_to_many')
    if joined.observation_id.isna().any():
        raise ValueError('Condition cell without eligible observation')
    cellcost = joined.groupby('cell_id', sort=True).agg(n_raw_records=('observation_id', 'size'), raw_observation_ids=('observation_id', list)).reset_index()
    cellcost = cells.merge(cellcost, on='cell_id', validate='one_to_one')
    cellcost.to_parquet(BASE / 'results/e66_cell_costs.parquet', index=False)
    costs = cellcost.set_index('cell_id').n_raw_records.to_dict()
    vc = []
    for row in vectors.itertuples(index=False):
        ids = list(map(int, row.condition_cell_ids))
        if len(set(ids)) != 2 * row.n_backgrounds:
            raise ValueError('Vector cost is not 2B distinct cells')
        vc.append(dict(vector_id=int(row.vector_id), n_backgrounds=int(row.n_backgrounds), n_scalar_labels=int(row.n_backgrounds), n_condition_cells=len(ids), n_raw_records=sum((costs[i] for i in ids))))
    vcost = pd.DataFrame(vc).set_index('vector_id')
    vcost.reset_index().to_parquet(BASE / 'results/e66_vector_costs.parquet', index=False)
    family = pd.read_parquet(E65 / 'results/e65_family_requests.parquet')
    scope = pd.read_parquet(E65 / 'results/e65_scope_vectors.parquet')
    scope['n_calibration_available'] = scope.n_original_train - scope.n_family_train
    scope['common8'] = scope.n_family_train.ge(1) & scope.n_calibration_available.ge(8)
    scope['common16'] = scope.n_family_train.ge(1) & scope.n_calibration_available.ge(16)
    scope['family_supported'] = scope.n_family_train.ge(1)
    scope['update_supported'] = scope.n_family_train.ge(1) & scope.n_calibration_available.ge(1)
    scope['family_n0'] = scope.n_family_train.eq(0)
    scope = scope.merge(vcost.reset_index(), on='vector_id', validate='one_to_one')
    scope.to_parquet(BASE / 'results/e66_scope_vectors.parquet', index=False)
    vg = dict(tuple(vectors.groupby(EDGE, sort=True)))
    outputs = [BASE / f'results/e66_{n}.parquet' for n in ['cell_costs', 'vector_costs', 'scope_vectors']]
    total = dict(n_edges=0, n_allocations=0, n_requests=0, n_factors=0, n_union_queries=0, n_reusable_factors=0, n_novel_factors=0, n_prediction_rows=0)
    manifests = {}
    for key, fs in family.groupby(EDGE, sort=True):
        key = tuple(key)
        g = vg[key].set_index('vector_id')
        flat_cells = np.concatenate(g.condition_cell_ids.to_numpy())
        if len(np.unique(flat_cells)) != len(flat_cells):
            raise ValueError('Within-edge cell cost cannot be summed')
        allocations, requests, factors = ([], [], {})
        endpoint = {}
        for f in fs.itertuples(index=False):
            for train in [f.original_train_vector_ids, f.family_train_vector_ids]:
                if len(train):
                    endpoint.setdefault(tuple(map(int, train)), set()).update(map(int, f.query_vector_ids))
        rankings = {}
        for f in fs.itertuples(index=False):
            t0 = sorted(map(int, f.original_train_vector_ids))
            tf = sorted(map(int, f.family_train_vector_ids))
            pool = sorted(set(t0) - set(tf))
            query = sorted(map(int, f.query_vector_ids))
            if set(tf) & set(pool) or set(tf) | set(pool) != set(t0):
                raise ValueError('Family/calibration partition differs')
            if set(t0) & set(query):
                raise ValueError('Training/query overlap')
            held = {f.family_role1, f.family_role2}
            trmols = set(g.loc[t0].canonical_role1) | set(g.loc[t0].canonical_role2)
            qmols = set(g.loc[query].canonical_role1) | set(g.loc[query].canonical_role2)
            if trmols & qmols:
                raise ValueError('Cross-role test partner leakage')
            for r in range(3):
                rankkey = (int(f.fold_role1), int(f.fold_role2), r)
                if rankkey not in rankings:
                    rankings[rankkey] = {i: hashlib.sha256(payload([20261017 + r, 'E66_calibration', *key, int(f.fold_role1), int(f.fold_role2), i])).digest() for i in t0}
                ranking = sorted(pool, key=lambda i: (rankings[rankkey][i], i))
                raw_prefix = np.r_[0, np.cumsum(vcost.loc[ranking, 'n_raw_records'].to_numpy(np.int64))]
                width = int(g.n_backgrounds.iloc[0])
                previous = set()
                for budget in BUDGETS:
                    n = len(pool) if budget == 'full' else min(int(budget), len(pool))
                    calibration = sorted(ranking[:n])
                    if not previous <= set(calibration):
                        raise ValueError('Calibration allocations not nested')
                    previous = set(calibration)
                    cid = identifier(['E66_allocation_v1', f.family_request_id, r, budget])
                    row = dict(**dict(zip(EDGE, key)), family_request_id=f.family_request_id, fold_role1=int(f.fold_role1), fold_role2=int(f.fold_role2), allocation=f'r{r}', budget=budget, allocation_id=cid, calibration_vector_ids=calibration, calibration_vector_hash=array_hash(calibration), query_vector_ids=query, query_vector_hash=array_hash(query), n_query=len(query), n_calibration=n, n_calibration_available=len(pool), n_original_train=len(t0), n_family_train=len(tf), n_scalar_labels=n * width, n_condition_cells=2 * n * width, n_raw_records=int(raw_prefix[n]))
                    allocations.append(row)
                    for learner in LEARNERS:
                        train = sorted(tf + calibration) if learner == 'joint' else calibration
                        fid = factor_id(key, train) if train else ''
                        rq = {k: val for k, val in row.items() if k != 'calibration_vector_ids'}
                        rq.update(learner=learner, factor_id=fid, n_train=len(train), supported=bool(train), plan_request_id=identifier(['E66_request_v1', cid, learner]))
                        requests.append(rq)
                        if not train:
                            continue
                        if fid not in factors:
                            factors[fid] = dict(**dict(zip(EDGE, key)), factor_id=fid, train_vector_ids=train, train_vector_hash=array_hash(train), n_train=len(train), queries=set(), requires_krr=False, requested_learners=set())
                        item = factors[fid]
                        if item['train_vector_ids'] != train:
                            raise ValueError('Factor identity collision')
                        item['queries'].update(query)
                        item['requires_krr'] |= learner != 'calibration_centroid'
                        item['requested_learners'].add(learner)
        for item in factors.values():
            query = sorted(item.pop('queries'))
            item['union_query_vector_ids'] = query
            item['union_query_vector_hash'] = array_hash(query)
            item['n_union_query'] = len(query)
            item['requested_learners'] = sorted(item['requested_learners'])
            train = tuple(item['train_vector_ids'])
            reuse = train in endpoint and set(query) <= endpoint[train]
            item['reuse_e65_factor_id'] = e65_factor_id(key, train) if reuse else ''
            item['origin'] = 'frozen_E65_operator' if reuse else 'analytic_singleton' if len(train) == 1 else 'new_KRR'
        tag = edge_tag(key)
        paths = {name: BASE / f'parts/{tag}_{name}.parquet' for name in ['allocations', 'requests', 'factors']}
        for name, rows in [('allocations', allocations), ('requests', requests), ('factors', list(factors.values()))]:
            pd.DataFrame(rows).to_parquet(paths[name], index=False)
        counts = dict(n_allocations=len(allocations), n_requests=len(requests), n_factors=len(factors), n_union_queries=sum((i['n_union_query'] for i in factors.values())), n_reusable_factors=sum((bool(i['reuse_e65_factor_id']) for i in factors.values())), n_novel_factors=sum((not i['reuse_e65_factor_id'] for i in factors.values())), n_prediction_rows=sum((i['n_query'] for i in requests)))
        for name, count in counts.items():
            total[name] += count
        total['n_edges'] += 1
        meta = BASE / f'parts/{tag}_plan_metadata.json'
        write_json(meta, dict(edge=list(key), counts=counts, output_sha256=pins(paths.values())))
        manifests[meta.name] = sha(meta)
        outputs.extend(paths.values())
        outputs.append(meta)
        print(f'Planned {tag}: {len(factors)} factors; {len(requests)} requests', flush=True)
    if before != pins(PLAN_INPUTS):
        raise ValueError('Source changed during outcome-blind plan')
    expected = dict(n_vectors=25956, common8=24221, common16=22963, family_supported=25520, update_supported=24943, family_n0=436)
    actual = dict(n_vectors=len(scope), **{n: int(scope[n].sum()) for n in expected if n != 'n_vectors'})
    if actual != expected or total['n_prediction_rows'] != 25956 * 3 * 7 * 3:
        raise ValueError(f'Frozen support mismatch {actual}, {total}')
    write_json(BASE / 'metadata/e66_plan_metadata.json', dict(schema_version='E66_plan_v1', source_sha256=before, output_sha256=pins(outputs), code_sha256=sha(Path(__file__)), common_code_sha256=sha(BASE / 'code/e66_common.py'), counts=total, support=actual, checkpoint_manifest=manifests, outcome_access='Identity, structure, cell membership and eligible observation ID columns only; no outcome or prediction columns read', seconds=time.perf_counter() - started))
    print(json.dumps(dict(counts=total, support=actual, seconds=time.perf_counter() - started)), flush=True)
if __name__ == '__main__':
    main()
