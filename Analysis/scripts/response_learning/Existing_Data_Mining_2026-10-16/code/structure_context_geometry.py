"""Historical structural prediction of scale-invariant context directions.

All-background outcome vectors define the target, never predictors. A training
line decomposition distinguishes signed-amplitude prediction from information
orthogonal to a common background direction. Neither is mechanistic evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from chemical_features import DESCRIPTOR_COLUMNS
from decision_opportunity_audit import markdown_table
BASE = Path(__file__).resolve().parents[1]
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
SPLITS = ('pair', 'role1_new_purged', 'role2_new_purged')
MODELS = ('training_centroid', 'descriptor_ridge_1', 'descriptor_ridge_10', 'descriptor_ridge_100')
SEED = 20261017
MIN_TRAIN = 5
METRICS = ('loss', 'centroid_loss', 'improvement', 'along_improvement', 'orthogonal_improvement')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def assignment(value, kind):
    return int.from_bytes(hashlib.sha256(f'{SEED}|geometry_{kind}|{value}'.encode()).digest()[:8], 'little') % 5

def split_masks(p1, p2, split, fold):
    if split == 'pair':
        groups = np.array([assignment(f'{a}\t{b}', 'pair') for a, b in zip(p1, p2)])
    else:
        own = p1 if split == 'role1_new_purged' else p2
        groups = np.array([assignment(a, 'molecule') for a in own])
    test, train = (groups == fold, groups != fold)
    before = int(train.sum())
    held = set()
    if split != 'pair':
        own = p1 if split == 'role1_new_purged' else p2
        held = set(own[test]) - {'<SINGLE>'}
        train &= ~np.isin(p1, list(held)) & ~np.isin(p2, list(held))
    return (train, test, before - int(train.sum()), held)

def fit_models(x, y, xt):
    mu, sd = (x.mean(axis=0), x.std(axis=0))
    sd[sd == 0] = 1.0
    z, zt = ((x - mu) / sd, (xt - mu) / sd)
    center = y.mean(axis=0)
    predictions = {'training_centroid': np.broadcast_to(center, (len(xt), len(center))).copy()}
    for alpha in (1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky')
        predictions[f'descriptor_ridge_{alpha:g}'] = model.fit(z, y).predict(zt)
    _, _, vt = np.linalg.svd(y, full_matrices=False)
    return (predictions, vt[0])

def losses(target, prediction, baseline, axis):
    e, e0 = (target - prediction, target - baseline)
    total, base = (np.square(e).sum(), np.square(e0).sum())
    along = np.square(e0 @ axis).sum() - np.square(e @ axis).sum()
    e_perp = e - (e @ axis)[:, None] * axis
    e0_perp = e0 - (e0 @ axis)[:, None] * axis
    orth = np.square(e0_perp).sum() - np.square(e_perp).sum()
    assert np.isclose(base - total, along + orth, atol=1e-08, rtol=1e-10)
    return dict(loss_sum=float(total), centroid_loss_sum=float(base), improvement_sum=float(base - total), along_improvement_sum=float(along), orthogonal_improvement_sum=float(orth))

def summary(frame, outcome, split, model, gate, kind, label):
    r = dict(outcome=outcome, split=split, model=model, gate=gate, stratum_type=kind, stratum=label, n_edges=len(frame), n_classes=frame.reaction_type.nunique(), n_scored_rows=int(frame.n_scored.sum()))
    for field in METRICS:
        r[f'{field}_equal_edge_mean'] = float(frame[field].mean())
        r[f'{field}_equal_edge_median'] = float(frame[field].median())
    r['positive_orthogonal_fraction'] = float(frame.orthogonal_improvement.gt(0).mean()) if len(frame) else np.nan
    return r

def main():
    inputs = ['results/matched_response_profiles.parquet', 'processed/molecular_features.parquet', 'results/condition_network_meta.csv']
    hashes = {p: sha(BASE / p) for p in inputs}
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        if not passed:
            raise AssertionError(f'{name}: {detail}')
    x = np.arange(36.0).reshape(12, 3)
    y = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 0]), x[:, 0] / 20])
    preds, v = fit_models(x[:8], y[:8], x[8:])
    center = preds['training_centroid']
    for name, pred in preds.items():
        losses(y[8:], pred, center, v)
    check('toy_orthogonal_loss_closure', True)
    rank1 = np.outer(np.linspace(-2, 2, 12), np.array([1.0, -2.0, 3.0]))
    rank1 /= np.linalg.norm(rank1, axis=1)[:, None]
    pp, vv = fit_models(x[:8], rank1[:8], x[8:])
    check('signed_line_toy_cannot_create_orthogonal_gain', all((abs(losses(rank1[8:], p, pp['training_centroid'], vv)['orthogonal_improvement_sum']) < 1e-20 for p in pp.values())))
    z = (x[:8] - x[:8].mean(axis=0)) / x[:8].std(axis=0)
    zt = (x[8:] - x[:8].mean(axis=0)) / x[:8].std(axis=0)
    yc = y[:8].mean(axis=0)
    beta = np.linalg.solve(z.T @ z + 10 * np.eye(3), z.T @ (y[:8] - yc))
    check('dense_penalty_equations_match_sklearn', np.allclose(zt @ beta + yc, preds['descriptor_ridge_10'], atol=1e-12))
    p1, p2 = (np.array(['a', 'b', 'c', 'd', 'e', 'f']), np.array(['x', 'a', 'z', 'x', 'z', 'y']))
    for split in SPLITS[1:]:
        for fold in range(5):
            train, test, _, held = split_masks(p1, p2, split, fold)
            check(f'toy_cross_role_purge_{split}_{fold}', not (set(p1[train]) | set(p2[train])) & held)
    profiles = pd.read_parquet(BASE / inputs[0])
    features = pd.read_parquet(BASE / inputs[1])
    network = pd.read_csv(BASE / inputs[2])
    classes = sorted(network.loc[network.row_level.eq('swap'), 'reaction_type'].unique())
    cmap = features.set_index('smiles').canonical_smiles.to_dict()
    fmap = features.set_index('smiles')[DESCRIPTOR_COLUMNS]
    check('twelve_frozen_numeric_descriptors', len(DESCRIPTOR_COLUMNS) == 12 and features.parse_valid.all())
    fold_rows, edge_rows, panels, signal_rows = ([], [], [], [])
    for count, (key, g) in enumerate(profiles.groupby(EDGE, sort=True), 1):
        backgrounds = sorted(g.background_fingerprint.unique())
        common = sorted(set.intersection(*(set(t.substrate_pair_key) for _, t in g.groupby('background_fingerprint'))))
        ident = g.drop_duplicates('substrate_pair_key').set_index('substrate_pair_key').reindex(common)
        matrix = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta').reindex(index=common, columns=backgrounds).to_numpy(float)
        pos = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='both_positive').reindex(index=common, columns=backgrounds).all(axis=1).to_numpy()
        identities, blocks, valid = ([], [], np.ones(len(common), dtype=bool))
        for role in (1, 2):
            raw = ident[f'sub_{role}_smiles']
            single = raw.isna() | raw.eq('')
            canonical = raw.map(cmap)
            canonical.loc[single] = '<SINGLE>'
            values = fmap.reindex(raw).to_numpy(float)
            values[single] = 0.0
            valid &= canonical.notna().to_numpy() & np.isfinite(values).all(axis=1)
            identities.append(canonical.fillna('<INVALID>').to_numpy(str))
            blocks.append(values)
        xfull = np.column_stack(blocks)
        norms = np.linalg.norm(matrix, axis=1)
        check(f'complete_and_canonical_unique_{count}', np.isfinite(matrix).all() and len(set(zip(*identities))) == len(common))
        for outcome in ('all', 'both_positive'):
            population = np.ones(len(common), dtype=bool) if outcome == 'all' else pos
            keep = population & (norms > 0) & valid
            meta = dict(zip(EDGE, key), outcome=outcome, n_backgrounds=len(backgrounds), n_intersection_rows=len(common), n_population_rows=int(population.sum()), n_zero_rows=int((population & (norms == 0)).sum()), n_invalid_features=int((population & ~valid).sum()), n_nonzero_valid=int(keep.sum()))
            panels.append(meta)
            xx, yy = (xfull[keep], matrix[keep] / norms[keep, None])
            row_rms = norms[keep] / np.sqrt(len(backgrounds))
            a, b = (identities[0][keep], identities[1][keep])
            for split in SPLITS:
                per_model = {model: [] for model in MODELS}
                for fold in range(5):
                    train, test, n_purged, held = split_masks(a, b, split, fold)
                    nt, nv = (int(train.sum()), int(test.sum()))
                    status = 'no_test' if not nv else 'insufficient_training' if nt < MIN_TRAIN else 'scored'
                    fmeta = dict(**meta, split=split, fold=fold, n_train=nt, n_test=nv, n_cross_role_purged=n_purged, n_scored=nv if status == 'scored' else 0, n_test_role1_identities=len(set(a[test])), n_test_role2_identities=len(set(b[test])), n_test_role1_also_in_train=int(np.isin(a[test], list(set(a[train]) | set(b[train]))).sum()), n_test_role2_also_in_train=int(np.isin(b[test], list(set(a[train]) | set(b[train]))).sum()), status=status)
                    if split != 'pair':
                        assert not (set(a[train]) | set(b[train])) & held
                    if status == 'scored':
                        pred, axis = fit_models(xx[train], yy[train], xx[test])
                        baseline = pred['training_centroid']
                    for model in MODELS:
                        record = {**fmeta, 'model': model, **({f'{m}_sum': 0.0 for m in METRICS} if status != 'scored' else losses(yy[test], pred[model], baseline, axis))}
                        fold_rows.append(record)
                        per_model[model].append(record)
                        if status == 'scored':
                            for floor in (0.0, 1.0, 5.0, 10.0):
                                strong = row_rms[test] >= floor
                                signal_rows.append({**fmeta, 'model': model, 'test_rms_floor_pp': floor, 'n_scored': int(strong.sum()), **losses(yy[test][strong], pred[model][strong], baseline[strong], axis)})
                for model, records in per_model.items():
                    n = sum((r['n_scored'] for r in records))
                    row = dict(**meta, split=split, model=model, n_scored=n, n_scored_folds=sum((r['status'] == 'scored' for r in records)), n_cross_role_purged_sum=sum((r['n_cross_role_purged'] for r in records)), coverage=n / len(yy) if len(yy) else np.nan)
                    for m in METRICS:
                        row[f'{m}_sum'] = sum((r[f'{m}_sum'] for r in records))
                        row[m] = row[f'{m}_sum'] / n if n else np.nan
                    edge_rows.append(row)
        if count % 50 == 0:
            print(f'Structural context prediction {count}/302 edges', flush=True)
    folds, edges, panels = (pd.DataFrame(fold_rows), pd.DataFrame(edge_rows), pd.DataFrame(panels))
    check('all302_edges_2populations_3splits_4models', len(edges) == 302 * 2 * 3 * 4)
    check('all_fivefold_rows_retained', len(folds) == 302 * 2 * 3 * 4 * 5)
    check('full_loss_decomposition', np.allclose(edges.improvement_sum, edges.along_improvement_sum + edges.orthogonal_improvement_sum, atol=1e-08))
    check('centroid_zero_improvement', edges.loc[edges.model.eq('training_centroid'), 'improvement_sum'].eq(0).all())
    check('all_unit_targets', np.allclose(edges.loc[edges.n_scored.gt(0), 'loss'], edges.loc[edges.n_scored.gt(0), 'loss_sum'] / edges.loc[edges.n_scored.gt(0), 'n_scored']))
    for split, own in (('role1_new_purged', 1), ('role2_new_purged', 2)):
        check(f'real_no_training_identity_leak_{split}', folds.loc[folds.split.eq(split), f'n_test_role{own}_also_in_train'].eq(0).all())
    summary_rows = []
    for (outcome, split, model), g in edges.groupby(['outcome', 'split', 'model']):
        gates = {'any_scored': g.loc[g.n_scored.gt(0)], 'bg_ge2_nonzero_ge20_full': g.loc[g.n_backgrounds.ge(2) & g.n_nonzero_valid.ge(20) & g.coverage.eq(1)], 'bg_ge3_nonzero_ge20_full_primary': g.loc[g.n_backgrounds.ge(3) & g.n_nonzero_valid.ge(20) & g.coverage.eq(1)]}
        for gate, sub in gates.items():
            summary_rows.append(summary(sub, outcome, split, model, gate, 'corpus', 'ALL'))
            for reaction in classes:
                summary_rows.append(summary(sub.loc[sub.reaction_type.eq(reaction)], outcome, split, model, gate, 'reaction_class', reaction))
    summaries = pd.DataFrame(summary_rows)
    common_rows = []
    for outcome in ('all', 'both_positive'):
        s = edges.loc[edges.outcome.eq(outcome) & edges.n_backgrounds.ge(3) & edges.n_nonzero_valid.ge(20) & edges.coverage.eq(1)]
        sets = [set(map(tuple, s.loc[s.split.eq(split), EDGE].drop_duplicates().to_numpy())) for split in SPLITS]
        common_edges = set.intersection(*sets)
        selected = s.loc[[tuple(k) in common_edges for k in s[EDGE].to_numpy()]]
        for (split, model), sub in selected.groupby(['split', 'model']):
            common_rows.append(summary(sub, outcome, split, model, 'all_splits_common_primary', 'corpus', 'ALL'))
    common_summary = pd.DataFrame(common_rows)
    signal_folds = pd.DataFrame(signal_rows)
    sumfields = ['n_scored'] + [f'{m}_sum' for m in METRICS]
    signal_edges = signal_folds.groupby(EDGE + ['outcome', 'split', 'model', 'test_rms_floor_pp'], sort=True)[sumfields].sum().reset_index()
    signal_edges = signal_edges.merge(panels, on=EDGE + ['outcome'], validate='many_to_one')
    for m in METRICS:
        signal_edges[m] = signal_edges[f'{m}_sum'] / signal_edges.n_scored.replace(0, np.nan)
    signal_summary_rows = []
    for (outcome, split, model, floor), sub in signal_edges.groupby(['outcome', 'split', 'model', 'test_rms_floor_pp']):
        selected = sub.loc[sub.n_backgrounds.ge(3) & sub.n_scored.ge(20)]
        signal_summary_rows.append({**summary(selected, outcome, split, model, 'test_signal_ge20_bg_ge3', 'corpus', 'ALL'), 'test_rms_floor_pp': floor})
    signal_summary = pd.DataFrame(signal_summary_rows)
    check('all20_classes_retained', summaries.loc[summaries.stratum_type.eq('reaction_class'), 'stratum'].nunique() == 20)
    check('inputs_unchanged', all((sha(BASE / p) == h for p, h in hashes.items())))
    for name, data in (('folds', folds), ('edges', edges), ('panels', panels), ('summary', summaries), ('common_summary', common_summary), ('signal_folds', signal_folds), ('signal_edges', signal_edges), ('signal_summary', signal_summary), ('checks', pd.DataFrame(checks))):
        data.to_csv(BASE / f'results/structure_context_geometry_{name}.csv', index=False)
    metadata = dict(inputs_sha256=hashes, n_checks=len(checks), n_failed_checks=0, descriptors=DESCRIPTOR_COLUMNS, models=MODELS, minimum_training_rows=MIN_TRAIN, objective='sum squared row-unit-vector errors + alpha * squared standardized-feature coefficients; intercept unpenalized', identity_split='SHA256(seed|geometry_molecule|canonicalSMILES), role-specific test, test-role molecules purged from either training role', target='delta vector across all recorded backgrounds / row L2 norm; exact-zero directions undefined; all-background nonzero and both-positive populations are outcome-conditioned', information='Only 12 frozen descriptors per role; no source test response, no test target used in fitting/preprocessing/training SVD. Known backgrounds; new canonical identities within one edge only.', orthogonal='Leading right singular vector of uncentered TRAINING unit targets. Fold-specific line; along + orthogonal improvements close algebraically. Orthogonal empirical gain is distinct from signed-amplitude information.', signal_sensitivity='Only TEST scoring is restricted to row RMS>=0/1/5/10pp. Every fit uses the same all-nonzero training rows. Positive signal floors change test population and are not scale-invariant or noise corrections.', scope='Historical complete-intersection panels; no scaffold-heldout, prospective or new-reaction/background validation. Descriptors are exploratory fixed representation; no target-based selection of penalties.')
    (BASE / 'audit/structure_context_geometry_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    show = summaries.loc[summaries.outcome.eq('all') & summaries.stratum_type.eq('corpus') & summaries.gate.eq('bg_ge3_nonzero_ge20_full_primary')]
    columns = ['split', 'model', 'n_edges', 'n_classes', 'n_scored_rows', 'loss_equal_edge_mean', 'improvement_equal_edge_mean', 'along_improvement_equal_edge_mean', 'orthogonal_improvement_equal_edge_mean']
    report = f"# Molecular structure and scale-invariant context direction\n\n## Scientific question and competing explanation\n\nDoes molecular structure describe the recorded context-response direction when test molecules contribute no measured responses to training? Each target is the vector of component-swap deltas across all recorded backgrounds of an edge, divided by its L2 norm. Stable positive per-product scaling cancels. All302 repeated edges remain in panel/fold/edge tables, and all20 reaction classes remain in summary support accounting. All-zero vectors are undefined directions and counted separately. The common all-background intersection selects densely covered panels.\n\nTwelve previously frozen continuous descriptors for each role give24 predictors. They were not chosen by these targets. Training-only mean/std scaling and multioutput ridge with fixed penalties1/10/100 predict the unnormalized conditional mean of the unit vector. The objective is summed squared error plus alpha times squared coefficients; the intercept is unpenalized. The baseline is the training unit-vector centroid. No penalty is chosen from test results. Predictions are not renormalized, so squared loss is a proper mean-vector prediction criterion rather than angular error.\n\n## Holdout and support\n\nFive folds are deterministic hashes of canonical ordered pairs, or of the canonical molecule in role1/role2. For molecule splits, all rows containing any test-role molecule in either training role are purged. This guarantees that the held-out role's identities are absent from that edge's training in either role. The other partner can be known or unseen; this is not a both-partners-unseen claim. Pair folds need not have both partners known. No source test response is provided to these models. Background identities are already known, and other edge-specific fits can contain the same molecules; there is no cross-edge pooling.\n\nAll fold cells, including no-test and fewer-than5-training cases, remain saved. Primary summaries require>=3backgrounds, >=20 nonzero valid-feature rows, and complete test coverage. The full two-background and partial-coverage sensitivities remain saved. Missing observation is not zero. Zero-direction and both-positive eligibility are defined using observed outcomes, making these retrospective conditional populations. Test normalization defines the scoring target and is never used as an input, preprocessing estimate or training label. A test row's complete response vector is withheld together.\n\n## Information beyond a common signed background line\n\nPredicting opposite signs of a substrate amplitude can improve the centroid baseline even in a perfectly signed-rank1 matrix. Therefore each fold obtains the leading uncentered right singular vector from its TRAINING unit targets only. Squared-error reduction versus the training centroid is separated into along-line and orthogonal-to-line reductions. The closure is algebraic; the empirical orthogonal reduction asks whether structure carries information beyond that training line. The line changes across folds and is not a noise-free universal direction.\n\n{markdown_table(show[columns])}\n\nThe following comparison holds the primary edge cohort common across all three split designs; the underlying heldout fold composition still differs by design.\n\n{markdown_table(common_summary.loc[common_summary.outcome.eq('all'), columns])}\n\n## Weak-signal sensitivity with unchanged fits\n\nRestrict only test scoring to row RMS>=1/5/10pp, retaining the same all-nonzero training rows, preprocessing and training line. Thus these are outcome-conditioned signal-scoring sensitivities, not models refitted on strong signals. At least20 scored strong-signal rows and3backgrounds define each reported edge. Changing the threshold changes the test cohort; these gates are not scale-invariant, do not remove assay bias and do not estimate a noise-free effect.\n\n{markdown_table(signal_summary.loc[signal_summary.outcome.eq('all') & signal_summary.model.eq('descriptor_ridge_100'), ['test_rms_floor_pp', *columns]])}\n\n## Interpretation boundaries\n\nPositive structural prediction is a historical association between this descriptor representation and recorded direction. It does not identify chemical states, explain all geometric residual, validate assays, or establish a causal mechanism. Condition-dependent analytical response, omitted variables and measurement noise remain explanations. Exact normalization invariance does not remove those explanations. A negative result would constrain only these fixed additive descriptor predictors and holdouts, not all molecular representations or hidden regimes. Direction loss is dimensionless and cannot be reported as yield gain. Outcome-selected nonzero directions, complete intersections, shared molecules/edges/backgrounds and the two populations are dependent. No p-values or independent replication claims are made.\n\n## Artifacts and checks\n\nRun `python code/structure_context_geometry.py`. All nine `results/structure_context_geometry_*.csv` tables and `audit/structure_context_geometry_metadata.json` are saved. {len(checks)} checks pass, including all302edges, every fold, all20classes, real cross-role identity exclusion, dense ridge normal equations, an exactly signed-rank1 toy, and loss decomposition. Input hashes are unchanged. These check computation, not independent scientific replication.\n"
    (BASE / 'reports/STRUCTURE_CONTEXT_GEOMETRY.md').write_text(report, encoding='utf-8')
    print(show[columns].to_string(index=False))
    print(json.dumps({'n_checks': len(checks), 'n_failed': 0}))
if __name__ == '__main__':
    main()
