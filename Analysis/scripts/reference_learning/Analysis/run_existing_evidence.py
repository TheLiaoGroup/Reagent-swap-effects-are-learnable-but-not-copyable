"""Rebuild historical screens and mine predictive response complexity from raw data."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold, GroupKFold
from threadpoolctl import threadpool_limits
from dimension_core import spectrum, centred, fit_basis, select_panel, reconstruct
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, TABLES = (HERE / 'data', HERE / 'tables')
SEED = 20261016
BUDGETS = (2, 4, 6, 8, 12, 16, 24)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_screens():
    ref = HERE.parent / 'external' / 'hte_common.py'
    spec = importlib.util.spec_from_file_location('frozen_hte', ref)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    u = pd.read_csv(DATA / 'raw_long.csv.gz', low_memory=False)
    u['yield'] = pd.to_numeric(u['yield_numeric'], errors='coerce')
    u['rxn_class'] = u['reaction_class']
    u['subpair'] = u['sub_1_smiles'].fillna('nan') + ' >> ' + u['sub_2_smiles'].fillna('nan')
    numeric = u[np.isfinite(u['yield']) & u['rxn_class'].ne('Finkelstein')].copy()
    clean, dup = h.drop_duplicated_cells(numeric)
    screens = h.build_screens(clean)
    alloc = h.allocate(screens)
    registry, matrices, observations, checks = ([], {}, [], [])
    for key, s in sorted(screens.items()):
        idx = alloc[key]['analysis']
        matrices[key] = dict(M=s['M'][idx], sps=np.array(s['sps'])[idx], conds=np.array(s['conds']), cls=s['cls'])
        lookup = {sp: label for label, ix in alloc[key].items() for sp in np.array(s['sps'])[ix]}
        g = clean[clean.rxn_class.eq(s['cls']) & clean.subpair.isin(s['sps']) & clean.condition_id.isin(s['conds'])].copy()
        g['screen'], g['partition'] = (key, g.subpair.map(lookup))
        observations.append(g[['record_id', 'source_file', 'source_sheet', 'source_row', 'screen', 'partition', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'condition_id', 'yield', 'name0', 'archive']])
        for i, sp in enumerate(s['sps']):
            registry.append(dict(screen=key, subpair=sp, partition=lookup[sp], n_observed=int(np.isfinite(s['M'][i]).sum())))
        m = matrices[key]['M']
        checks.append(dict(screen=key, all_pairs=len(s['sps']), analysis_pairs=len(idx), conditions=m.shape[1], analysis_wells=int(np.isfinite(m).sum()), all_wells=int(np.isfinite(s['M']).sum()), zero_fraction=float(np.nanmean(m[np.isfinite(m)] == 0)), all_zero_pairs=int(np.sum(np.nanmax(m, axis=1) == 0)), completeness=float(np.isfinite(m).mean())))
    pd.DataFrame(registry).to_csv(DATA / 'screen_partition_registry.csv', index=False)
    pd.concat(observations).to_csv(DATA / 'qualified_observations.csv.gz', index=False)
    pd.DataFrame(checks).to_csv(TABLES / 'screen_inventory.csv', index=False)
    dup[['record_id', 'rxn_class', 'subpair', 'condition_id', 'yield']].to_csv(TABLES / 'excluded_duplicate_cells.csv', index=False)
    raw = pd.concat(observations)
    raw = raw[raw.partition.eq('analysis')]
    provenance = dict(raw_rows=len(u), finite_yields=int(np.isfinite(u['yield']).sum()), duplicate_rows_excluded=len(dup), qualifying_screens=len(screens), analysis_pairs=len(registry) - sum((len(alloc[k]['sealed']) + len(alloc[k]['reserve']) for k in alloc)), analysis_wells=len(raw), definition_path=str(ref), definition_sha256=digest(ref), raw_long_sha256=digest(DATA / 'raw_long.csv.gz'))
    assert (provenance['raw_rows'], provenance['finite_yields'], provenance['duplicate_rows_excluded'], provenance['qualifying_screens'], provenance['analysis_pairs'], provenance['analysis_wells']) == (410587, 381688, 338, 19, 2454, 196491), provenance
    (DATA / 'screen_reconstruction_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('SCREENS', json.dumps(provenance), flush=True)
    return (matrices, clean)

def geometry(matrices):
    rng = np.random.default_rng(SEED)
    rows, boot, refs, matched = ([], [], [], [])
    for key, item in matrices.items():
        m = item['M']
        productive = np.nanmax(m, axis=1) >= 0.2
        complete = np.isfinite(m).all(axis=1)
        for view in ('row', 'interaction'):
            variants = dict(all=m, complete_rows=m[complete], productive_rows=m[productive])
            if key.startswith('SNAr|'):
                variants['drop_recipe_ambiguity'] = m[:, item['conds'] != 'C2']
            if key.startswith('Chan-Lam|'):
                variants['drop_recipe_ambiguity'] = m[:, item['conds'] != 'C68']
            for label, x in variants.items():
                if len(x) >= 5:
                    rows.append(dict(screen=key, view=view, subset=label, n_pairs=x.shape[0], n_conditions=x.shape[1], **spectrum(x, view)))
        for b in range(100):
            ix = rng.integers(0, len(m), len(m))
            boot.append(dict(screen=key, draw=b, **spectrum(m[ix], 'interaction')))
        for b in range(30):
            for kind in ('within_row', 'within_column'):
                x = m.copy()
                if kind == 'within_row':
                    for a in x:
                        ok = np.isfinite(a)
                        a[ok] = rng.permutation(a[ok])
                else:
                    for j in range(x.shape[1]):
                        ok = np.isfinite(x[:, j])
                        x[ok, j] = rng.permutation(x[ok, j])
                refs.append(dict(screen=key, reference=kind, draw=b, **spectrum(x, 'interaction')))
        for b in range(50):
            ri = rng.choice(len(m), 40, replace=False)
            ci = rng.choice(m.shape[1], 32, replace=False)
            matched.append(dict(screen=key, draw=b, subset='all', **spectrum(m[np.ix_(ri, ci)], 'interaction')))
            if productive.sum() >= 20:
                ri = rng.choice(np.flatnonzero(productive), 20, replace=False)
                matched.append(dict(screen=key, draw=b, subset='productive_20_pairs', **spectrum(m[np.ix_(ri, ci)], 'interaction')))
        print('GEOMETRY', key, flush=True)
    pd.DataFrame(rows).to_csv(TABLES / 'dimension_sensitivity.csv', index=False)
    pd.DataFrame(boot).to_csv(TABLES / 'dimension_bootstrap.csv', index=False)
    pd.DataFrame(refs).to_csv(TABLES / 'permutation_reference.csv', index=False)
    pd.DataFrame(matched).to_csv(TABLES / 'matched_size_spectra.csv', index=False)

def score_predictions(y, pred, anchors, targets=None):
    mask = np.isfinite(y)
    if targets is not None:
        target_mask = np.zeros(y.shape[1], dtype=bool)
        target_mask[targets] = True
        mask &= target_mask
    mask[:, anchors] = False
    errors = np.where(mask, pred - y, 0.0)
    counts = mask.sum(axis=1)
    actual = np.where(mask, y, np.nan)
    guessed = np.where(mask, pred, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ra = rankdata(actual, axis=1, nan_policy='omit')
        rb = rankdata(guessed, axis=1, nan_policy='omit')
        da = ra - np.nanmean(ra, axis=1, keepdims=True)
        db = rb - np.nanmean(rb, axis=1, keepdims=True)
        den = np.sqrt(np.nansum(da * da, axis=1) * np.nansum(db * db, axis=1))
        rho = np.divide(np.nansum(da * db, axis=1), den, out=np.full(len(y), np.nan), where=den > 1e-12)
        true_variable = np.nansum(da * da, axis=1) > 1e-12
        rho[true_variable & (den <= 1e-12)] = 0.0
    choice = np.argmax(np.where(mask, pred, -np.inf), axis=1)
    regret = np.nanmax(actual, axis=1) - y[np.arange(len(y)), choice]
    return (counts, np.sum(errors * errors, axis=1), np.sum(abs(errors), axis=1), rho, regret)

def crossvalidate(matrices):
    scores, folds, panels, target_registry = ([], [], [], [])
    for sk, item in matrices.items():
        m, sps = (item['M'], item['sps'])
        target_rng = np.random.default_rng(SEED + int(hashlib.sha256(sk.encode()).hexdigest()[:6], 16))
        targets = np.sort(target_rng.choice(m.shape[1], max(6, int(round(m.shape[1] * 0.2))), replace=False))
        candidates = np.setdiff1d(np.arange(m.shape[1]), targets)
        for j in targets:
            target_registry.append(dict(screen=sk, condition_id=item['conds'][j], condition_index=int(j)))
        roles = [sp.split(' >> ') for sp in sps]
        schemes = []
        for repeat in range(2):
            schemes.extend((('pair', repeat, f, tr, te) for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + repeat).split(m))))
        for role in (0, 1):
            groups = [x[role] for x in roles]
            ng = len(set(groups))
            if ng >= 5:
                schemes.extend(((f'role{role + 1}', 0, f, tr, te) for f, (tr, te) in enumerate(GroupKFold(5).split(m, groups=groups))))
        for split, repeat, fold, tr, te in schemes:
            assert not set(tr) & set(te)
            if split.startswith('role'):
                role = int(split[-1]) - 1
                assert not {roles[i][role] for i in tr} & {roles[i][role] for i in te}
            train, test = (m[tr], m[te])
            fitted = fit_basis(train)
            for idx in te:
                folds.append(dict(screen=sk, split=split, repeat=repeat, fold=fold, pair_index=idx, subpair=sps[idx]))
            orders = []
            for pm, draw in [('variance_greedy', 0), ('high_mean', 0)] + [('random', j) for j in range(3)]:
                rng = np.random.default_rng(SEED + repeat * 1000 + fold * 30 + draw)
                order = candidates[select_panel(train[:, candidates], max(BUDGETS), pm, rng)]
                orders.append((pm, draw, order))
                for pos, j in enumerate(order):
                    panels.append(dict(screen=sk, split=split, repeat=repeat, fold=fold, panel=pm, draw=draw, position=pos + 1, condition_id=item['conds'][j]))
            for k in BUDGETS:
                if k >= m.shape[1]:
                    continue
                for pm, draw, order in orders:
                    anchors = order[:k]
                    for method in ('additive', 'affine', 'low_rank'):
                        pred = reconstruct(train, test, anchors, method, fitted)
                        assert not set(anchors) & set(targets)
                        counts, sse, sae, rho, regret = score_predictions(test, pred, anchors, targets)
                        for ii, idx in enumerate(te):
                            scores.append(dict(screen=sk, split=split, repeat=repeat, fold=fold, pair_index=int(idx), panel=pm, draw=draw, k=k, model=method, n_targets=int(counts[ii]), n_anchors_observed=int(np.isfinite(test[ii, anchors]).sum()), sse=float(sse[ii]), sae=float(sae[ii]), rho=float(rho[ii]), regret=float(regret[ii]), productive=bool(np.nanmax(test[ii]) >= 0.2)))
        print('CV', sk, 'folds', len(schemes), 'cumulative_scores', len(scores), flush=True)
    pd.DataFrame(folds).to_csv(DATA / 'cv_fold_registry.csv', index=False)
    pd.DataFrame(target_registry).to_csv(DATA / 'cv_target_registry.csv', index=False)
    pd.DataFrame(panels).to_csv(TABLES / 'training_selected_panels.csv', index=False)
    s = pd.DataFrame(scores)
    s.to_csv(TABLES / 'cv_pair_scores.csv.gz', index=False)
    summaries = []
    for subset in ('all', 'productive'):
        v = s if subset == 'all' else s[s.productive]
        for keys, g in v.groupby(['screen', 'split', 'panel', 'k', 'model'], sort=True):
            n = int(g.n_targets.sum())
            summaries.append(dict(zip(['screen', 'split', 'panel', 'k', 'model'], keys), subset=subset, pairs=g.pair_index.nunique(), target_predictions=n, rmse=float(np.sqrt(g.sse.sum() / max(n, 1))), mae=float(g.sae.sum() / max(n, 1)), spearman=float(g.rho.mean()), top1_regret=float(g.regret.mean())))
    pd.DataFrame(summaries).to_csv(TABLES / 'cv_budget_summary.csv', index=False)

def qc_repeats(clean):
    result, values = ([], [])
    for cls, a, b in [('Hiyama', 'C43', 'C46'), ('Suzuki', 'C74', 'C77')]:
        d = clean[clean.rxn_class.eq(cls) & clean.condition_id.isin([a, b])]
        p = d.pivot(index='subpair', columns='condition_id', values='yield')
        if a not in p or b not in p:
            continue
        p = p[[a, b]].dropna()
        for subset in ('all', 'either_above_5pct'):
            z = p if subset == 'all' else p[p.max(axis=1) > 0.05]
            e = z[a] - z[b]
            result.append(dict(reaction_class=cls, first=a, second=b, subset=subset, n_pairs=len(z), mad_pp=float(e.abs().mean() * 100), rmse_pp=float(np.sqrt((e * e).mean()) * 100), both_zero=int(z.eq(0).all(axis=1).sum()), pearson=float(z.corr().iloc[0, 1])))
        q = p.reset_index()
        q['reaction_class'] = cls
        q['first'] = a
        q['second'] = b
        q = q.rename(columns={a: 'yield_first', b: 'yield_second'})
        values.append(q)
    pd.DataFrame(result).to_csv(TABLES / 'local_qc_repeat_summary.csv', index=False)
    if values:
        pd.concat(values).to_csv(TABLES / 'local_qc_repeat_values.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'prepare', 'geometry', 'cv'], default='all')
    args = parser.parse_args()
    TABLES.mkdir(exist_ok=True)
    start = time.time()
    with threadpool_limits(limits=1):
        matrices, clean = load_screens()
        qc_repeats(clean)
        for sk, it in matrices.items():
            name = sk.replace('|', '_').replace(' ', '_')
            np.savez_compressed(DATA / f'matrix_{name}.npz', M=it['M'], sps=it['sps'].astype(str), conds=it['conds'].astype(str))
        if args.stage in ('all', 'geometry'):
            geometry(matrices)
        if args.stage in ('all', 'cv'):
            crossvalidate(matrices)
    manifest = dict(date='2026-10-16', seed=SEED, budgets=BUDGETS, stage=args.stage, seconds=time.time() - start, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, bootstrap=100, permutation_draws=30, matched_size_draws=50, cv='5-fold x2 pair; 5-fold x1 role1 and role2', ridge_penalty=0.01, training_rank='min(number observed anchors - 1, training interaction k_var90)', evaluation='Fixed random 20% of conditions (minimum 6) per screen; excluded from every anchor panel across all budgets; same targets for all models and panels', status='Exploratory; primary fits exclude historical sealed and reserve pairs; no new prospective validation', code_hashes={p.name: digest(p) for p in [Path(__file__), HERE / 'dimension_core.py']})
    (DATA / f'analysis_manifest_{args.stage}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('FINISHED', json.dumps(manifest), flush=True)
if __name__ == '__main__':
    main()
