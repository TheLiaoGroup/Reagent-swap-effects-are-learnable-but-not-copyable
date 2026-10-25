"""Fixed fingerprint kernel prediction of the frozen E42 context directions.

All E42 support, targets, folds, training lines and signal gates are retained.
Models use molecular structure only; no model or penalty is selected by test loss."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits
from chemical_features import DESCRIPTOR_COLUMNS
from structure_context_geometry import EDGE, SPLITS, MIN_TRAIN, METRICS, split_masks, losses, summary
BASE = Path(__file__).resolve().parents[1]
ALPHAS = (1.0, 10.0, 100.0)
KERNEL_MODELS = tuple((f'morgan_{family}_ridge_{alpha:g}' for family in ('add', 'product') for alpha in ALPHAS))
MODELS = ('training_centroid', 'descriptor_ridge_1', 'descriptor_ridge_10', 'descriptor_ridge_100', *KERNEL_MODELS)
CHECKS = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()

def check(name, passed, detail=''):
    CHECKS.append(dict(check=name, passed=bool(passed), detail=str(detail)))
    if not passed:
        raise AssertionError(f'{name}: {detail}')

def centered_krr(kernel, cross_kernel, target, alphas=ALPHAS):
    """Exact summed-loss kernel ridge; the intercept is not penalized."""
    mean = kernel.mean(axis=0)
    grand = mean.mean()
    centered = kernel - mean[None, :] - mean[:, None] + grand
    cross = cross_kernel - cross_kernel.mean(axis=1)[:, None] - mean[None, :] + grand
    values, vectors = eigh(centered, check_finite=False)
    center = target.mean(axis=0)
    projected = vectors.T @ (target - center)
    return {alpha: center + cross @ (vectors @ (projected / (values[:, None] + alpha))) for alpha in alphas}

def fingerprints(canonical):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False, useBondTypes=True, includeRingMembership=True, countSimulation=False)
    fps = []
    for smiles in canonical:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f'Canonical molecule did not parse: {smiles}')
        fps.append(generator.GetFingerprint(molecule))
    gram = np.zeros((len(fps) + 1, len(fps) + 1))
    for i, fp in enumerate(fps):
        gram[i, :len(fps)] = DataStructs.BulkTanimotoSimilarity(fp, fps)
    gram[-1, -1] = 1.0
    return (fps, gram)

def toy_checks():
    rng = np.random.default_rng(20261017)
    x = rng.normal(size=(16, 5))
    y = rng.normal(size=(10, 3))
    kernel = x[:10] @ x[:10].T
    cross = x[10:] @ x[:10].T
    predicted = centered_krr(kernel, cross, y)
    for alpha, pred in predicted.items():
        xc = x[:10] - x[:10].mean(axis=0)
        beta = solve(xc.T @ xc + alpha * np.eye(5), xc.T @ (y - y.mean(axis=0)), assume_a='pos')
        dense = (x[10:] - x[:10].mean(axis=0)) @ beta + y.mean(axis=0)
        check(f'toy_dense_primal_alpha{alpha:g}', np.allclose(pred, dense, atol=1e-11, rtol=1e-11))
        check(f'toy_unpenalized_target_offset_alpha{alpha:g}', np.allclose(centered_krr(kernel, cross, y + 5, [alpha])[alpha], pred + 5, atol=1e-11, rtol=1e-11))
    first, second = (rng.normal(size=(12, 3)), rng.normal(size=(12, 2)))
    k1, k2 = (first @ first.T, second @ second.T)
    additive = np.column_stack([first, second]) / np.sqrt(2)
    product = np.einsum('ij,ik->ijk', first, second).reshape(12, -1)
    check('toy_additive_explicit_feature_gram', np.allclose((k1 + k2) / 2, additive @ additive.T))
    check('toy_product_tensor_feature_gram', np.allclose(k1 * k2, product @ product.T))
    rank1 = np.outer(np.linspace(-2, 2, 10), [1.0, -2.0, 3.0])
    rank1 /= np.linalg.norm(rank1, axis=1)[:, None]
    axis = np.linalg.svd(rank1, full_matrices=False)[2][0]
    for alpha, pred in centered_krr(kernel, cross, rank1).items():
        target = np.tile(rank1[0], (len(cross), 1))
        baseline = np.broadcast_to(rank1.mean(axis=0), target.shape)
        score = losses(target, pred, baseline, axis)
        check(f'toy_signed_rank1_no_orthogonal_improvement_alpha{alpha:g}', abs(score['orthogonal_improvement_sum']) < 1e-20)
    fps, gram = fingerprints(['CC', 'CO'])
    check('toy_SINGLE_unit_and_orthogonal', gram[-1, -1] == 1 and np.count_nonzero(gram[-1, :-1]) == 0)
    manual = len(set(fps[0].GetOnBits()) & set(fps[1].GetOnBits())) / len(set(fps[0].GetOnBits()) | set(fps[1].GetOnBits()))
    check('toy_Tanimoto_bit_intersection', gram[0, 1] == manual)
    check('toy_Tanimoto_PSD', np.linalg.eigvalsh(gram).min() >= -1e-10)
    return len(CHECKS)

def compare_tables(name, left, right, keys, columns):
    a, b = (left.set_index(keys).sort_index(), right.set_index(keys).sort_index())
    check(name + '_keys', a.index.equals(b.index))
    for column in columns:
        if pd.api.types.is_numeric_dtype(a[column]):
            passed = np.allclose(a[column].to_numpy(float), b[column].to_numpy(float), atol=5.1e-09, rtol=1e-10, equal_nan=True)
        else:
            passed = ((a[column] == b[column]) | a[column].isna() & b[column].isna()).all()
        check(name + '_' + column, passed)

def aggregate(folds, panels, signal_folds, classes):
    groupkeys = EDGE + ['outcome', 'split', 'model']
    sumfields = ['n_scored'] + [f'{m}_sum' for m in METRICS]
    edges = folds.groupby(groupkeys, sort=True)[sumfields].sum().reset_index()
    extra = folds.assign(scored=folds.status.eq('scored')).groupby(groupkeys).agg(n_scored_folds=('scored', 'sum'), n_cross_role_purged_sum=('n_cross_role_purged', 'sum')).reset_index()
    edges = edges.merge(extra, on=groupkeys, validate='one_to_one').merge(panels, on=EDGE + ['outcome'], validate='many_to_one')
    edges['coverage'] = edges.n_scored / edges.n_nonzero_valid.replace(0, np.nan)
    for m in METRICS:
        edges[m] = edges[f'{m}_sum'] / edges.n_scored.replace(0, np.nan)
    rows = []
    for (outcome, split, model), g in edges.groupby(['outcome', 'split', 'model']):
        gates = {'any_scored': g[g.n_scored > 0], 'bg_ge2_nonzero_ge20_full': g[(g.n_backgrounds >= 2) & (g.n_nonzero_valid >= 20) & g.coverage.eq(1)], 'bg_ge3_nonzero_ge20_full_primary': g[(g.n_backgrounds >= 3) & (g.n_nonzero_valid >= 20) & g.coverage.eq(1)]}
        for gate, sub in gates.items():
            rows.append(summary(sub, outcome, split, model, gate, 'corpus', 'ALL'))
            for reaction in classes:
                rows.append(summary(sub[sub.reaction_type == reaction], outcome, split, model, gate, 'reaction_class', reaction))
    summaries = pd.DataFrame(rows)
    common = []
    for outcome in ('all', 'both_positive'):
        g = edges[(edges.outcome == outcome) & (edges.n_backgrounds >= 3) & (edges.n_nonzero_valid >= 20) & edges.coverage.eq(1)]
        cohorts = [set(map(tuple, g.loc[g.split.eq(s), EDGE].to_numpy())) for s in SPLITS]
        shared = set.intersection(*cohorts)
        g = g.loc[[tuple(e) in shared for e in g[EDGE].to_numpy()]]
        for (split, model), sub in g.groupby(['split', 'model']):
            common.append(summary(sub, outcome, split, model, 'all_splits_common_primary', 'corpus', 'ALL'))
    signals = signal_folds.groupby(groupkeys + ['test_rms_floor_pp'], sort=True)[sumfields].sum().reset_index()
    signals = signals.merge(panels, on=EDGE + ['outcome'], validate='many_to_one')
    for m in METRICS:
        signals[m] = signals[f'{m}_sum'] / signals.n_scored.replace(0, np.nan)
    signal_rows = []
    for (outcome, split, model, floor), g in signals.groupby(['outcome', 'split', 'model', 'test_rms_floor_pp']):
        selected = g[(g.n_backgrounds >= 3) & (g.n_scored >= 20)]
        for kind, label, sub in [('corpus', 'ALL', selected)] + [('reaction_class', c, selected[selected.reaction_type == c]) for c in classes]:
            signal_rows.append({**summary(sub, outcome, split, model, 'test_signal_ge20_bg_ge3', kind, label), 'test_rms_floor_pp': floor})
    equal = []
    for keys, g in summaries[summaries.stratum_type.eq('reaction_class')].groupby(['outcome', 'split', 'model', 'gate']):
        record = dict(zip(['outcome', 'split', 'model', 'gate'], keys), n_classes_retained=len(g), n_classes_scoreable=int(g.n_edges.gt(0).sum()))
        for m in METRICS:
            record[m + '_equal_class_mean'] = g[m + '_equal_edge_mean'].mean()
        equal.append(record)
    return (edges, summaries, pd.DataFrame(common), signals, pd.DataFrame(signal_rows), pd.DataFrame(equal))

def table(frame, columns):

    def fmt(x):
        return 'NA' if pd.isna(x) else f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(('---' for _ in columns)) + ' |'] + ['| ' + ' | '.join((fmt(x) for x in row)) + ' |' for row in frame[columns].itertuples(index=False, name=None)])

def render_report(summaries, signal_summary, equal, metadata):
    primary = summaries[summaries.outcome.eq('all') & summaries.stratum_type.eq('corpus') & summaries.gate.eq('bg_ge3_nonzero_ge20_full_primary')]
    columns = ['split', 'model', 'n_edges', 'n_classes', 'n_scored_rows', 'loss_equal_edge_mean', 'improvement_equal_edge_mean', 'along_improvement_equal_edge_mean', 'orthogonal_improvement_equal_edge_mean', 'positive_orthogonal_fraction']
    signals = signal_summary[signal_summary.outcome.eq('all') & signal_summary.stratum_type.eq('corpus') & signal_summary.model.isin(['descriptor_ridge_100', 'morgan_add_ridge_10', 'morgan_product_ridge_10'])]
    class_rows = []
    classes = sorted(summaries.loc[summaries.stratum_type.eq('reaction_class'), 'stratum'].unique())
    byclass = summaries[summaries.outcome.eq('all') & summaries.stratum_type.eq('reaction_class') & summaries.gate.eq('bg_ge3_nonzero_ge20_full_primary')]
    for reaction in classes:
        row = dict(reaction_type=reaction)
        for split in SPLITS[1:]:
            for family in ('add', 'product'):
                x = byclass[(byclass.stratum == reaction) & (byclass.split == split) & (byclass.model == f'morgan_{family}_ridge_10')].iloc[0]
                row[f'{split}_{family}_orthogonal'] = x.orthogonal_improvement_equal_edge_mean
                row['n_edges'] = x.n_edges
        class_rows.append(row)
    eq = equal[equal.outcome.eq('all') & equal.gate.eq('bg_ge3_nonzero_ge20_full_primary')]
    return '\n'.join(['# Fixed Structural Kernels for Context Geometry', '', 'This extension tests whether a richer fixed molecular representation predicts the E42 row-unit context-response vector beyond a training-only common signed background line. All302 edges, both original outcome populations, all20 class support rows, exact E42 pair/role-new splits and cross-role purge are retained. Descriptors and the training centroid are copied from the audited E42 outputs; they are not refitted or retuned.', '', '## Fixed Kernels and Exact Fit', '', 'Every valid canonical molecule receives a fixed RDKit Morgan radius2,2048-bit fingerprint, with bond types and ring membership included and chirality excluded. Fingerprints are unsupervised and not selected by the response targets. Tanimoto similarities K1 and K2 describe the ordered roles. The additive kernel is (K1+K2)/2; the product kernel is K1*K2 elementwise, corresponding to a tensor product of role feature spaces. Both have unit diagonal. A missing SINGLE role is a separate unit feature: SINGLE-SINGLE=1 and SINGLE-molecule=0, rather than a zero molecular fingerprint.', '', "Centered kernel ridge uses fixed alpha1/10/100 and summed squared unit-vector error plus alpha times squared RKHS norm. Train and cross kernels are centered using training means only; the target intercept is the training centroid and unpenalized. All penalties are saved, with no selection from heldout results. The kernel scales differ from E42's standardized descriptors, so equal alpha values are not equivalent regularization strengths across representations. Exact dense eigensolutions are feasible because the largest eligible panel has773 rows; BLAS threads are limited to1. Predictions are not normalized.", '', "A test row's complete response vector is withheld together. Its unit normalization defines the target and is not a predictor or preprocessing estimate. Fivefold canonical identities, the minimum5 training rows, empty folds, and partial coverage follow E42 exactly. In role-new splits, the test role's identities are absent from either training role within this edge; the other partner may be known or unseen. This is neither a both-partners-new nor scaffold-heldout design. Similar molecules can still interpolate through Tanimoto, and fits do not pool edges or test unseen backgrounds.", '', '## Primary All-Yield Results', '', 'The primary cohort has>=3 recorded backgrounds,>=20 nonzero valid rows and full test coverage. It contains the same63 edges/10classes/15,877 rows under every split. Loss is dimensionless and is not yield gain. Along-line and orthogonal improvements are measured versus the training centroid using the leading uncentered right singular vector of TRAINING unit targets. Their closure is algebraic; their observed size/sign is the empirical result.', '', table(primary, columns), '', '## Equal-Class Weighting', '', 'These means first average edges inside each class, then average the supported classes equally. They do not assign zero effects to unsupported classes. Every unsupported class is still retained in the detailed tables.', '', table(eq, ['split', 'model', 'n_classes_retained', 'n_classes_scoreable', 'improvement_equal_class_mean', 'orthogonal_improvement_equal_class_mean']), '', '## Signal Scoring Sensitivity', '', 'Fits, training rows, kernels, centering and training lines are unchanged. Only TEST scoring is restricted to recorded row RMS>=0/1/5/10pp, with>=20 scored rows and>=3 backgrounds per reported edge. These gates are outcome-conditioned and not scale-invariant; they change the test cohort and are not noise corrections. Alpha10 is a fixed compact display for both kernels, not a selected optimum; all penalties and all20 class summaries remain in the CSVs.', '', table(signals, ['test_rms_floor_pp', *columns[:-1]]), '', '## Class Heterogeneity', '', 'The following fixed-alpha10 table displays orthogonal improvement on the primary cohort for every class; NA indicates no eligible edge. Other penalties and pair splits remain in the full summary.', '', table(pd.DataFrame(class_rows), list(pd.DataFrame(class_rows).columns)), '', '## Interpretation and Verification', '', 'The product kernel is a statistical joint representation, not evidence of a chemical interaction mechanism. Product-vs-additive comparisons are not a nested test and do not uniquely isolate interactions; normalizing response vectors can itself introduce nonlinear role dependence. Positive prediction constrains a fixed historical representation/holdout; negative results constrain these models rather than all molecular explanations. Outcome-conditioned nonzero directions, both-positive panels, complete intersections and signal floors select dependent retrospective populations. Noise, clipping, assay calibration, omitted variables and close-analogue overlap remain explanations. No mechanistic, independent replication, p-value or prospective-success claims follow.', '', f"Passed {metadata['n_checks']} checks. Kernel symmetry, unit diagonals, PSD, tensor multiplicativity and the SINGLE convention are verified. Independent explicit-feature primal and augmented dual/intercept systems check the solver; representative actual-fold augmented solves and predictions are saved with kernel hashes. Every E42 panel/fold count, status, identity purge and centroid loss reconciles; descriptors are copied exactly. Inputs remain SHA-256 identical. The report and output hashes/versions are recorded in audit/structural_kernel_context_metadata.json.", '', 'Run `python audit/structural_kernel_context_tests.py`, then `python code/structural_kernel_context.py`. Outputs use results/structural_kernel_context_: panels, folds, edges, summary, common_summary, equal_class_summary, signal_folds, signal_edges, signal_summary, kernel_audit, spotchecks, spot_predictions and checks. All old audited code and shared synthesis files are untouched.', ''])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    toy_checks()
    if args.test_only:
        print(f'Passed {len(CHECKS)} analytical/kernel checks')
        return
    inputs = ['results/matched_response_profiles.parquet', 'processed/molecular_features.parquet', 'results/condition_network_meta.csv', 'code/structure_context_geometry.py', 'code/chemical_features.py']
    inputs += [f'results/structure_context_geometry_{name}.csv' for name in ('panels', 'folds', 'edges', 'summary', 'signal_folds', 'signal_summary')]
    before = {p: sha(BASE / p) for p in inputs}
    profiles = pd.read_parquet(BASE / inputs[0])
    features = pd.read_parquet(BASE / inputs[1])
    network = pd.read_csv(BASE / inputs[2])
    classes = sorted(network.loc[network.row_level.eq('swap'), 'reaction_type'].unique())
    cmap = features.set_index('smiles').canonical_smiles.to_dict()
    fmap = features.set_index('smiles')[DESCRIPTOR_COLUMNS]
    canonical = sorted(features.canonical_smiles.dropna().unique())
    fps, global_kernel = fingerprints(canonical)
    ids = {s: i for i, s in enumerate(canonical)}
    ids['<SINGLE>'] = len(canonical)
    check('every_actual_canonical_molecule_fingerprinted', len(fps) == len(canonical) == 1228)
    check('global_Tanimoto_symmetric_unit_diagonal', np.allclose(global_kernel, global_kernel.T, atol=1e-15) and np.all(np.diag(global_kernel) == 1))
    global_min = float(eigh(global_kernel, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0])
    check('global_Tanimoto_with_SINGLE_PSD', global_min >= -1e-09, global_min)
    check('SINGLE_not_similar_to_any_actual_molecule', np.count_nonzero(global_kernel[-1, :-1]) == 0)
    global_bits = [set(fp.GetOnBits()) for fp in fps]
    for i, j in [(0, len(fps) - 1), (12, 120), (144, 545), (340, 850), (1, 2)]:
        manual = len(global_bits[i] & global_bits[j]) / len(global_bits[i] | global_bits[j])
        check(f'real_manual_Tanimoto_{i}_{j}', global_kernel[i, j] == manual)
    old_panels = pd.read_csv(BASE / 'results/structure_context_geometry_panels.csv')
    old_folds = pd.read_csv(BASE / 'results/structure_context_geometry_folds.csv')
    old_signals = pd.read_csv(BASE / 'results/structure_context_geometry_signal_folds.csv')
    fold_rows = []
    signal_rows = []
    panel_rows = []
    kernel_audit = []
    spots = []
    predictions = []
    sampled = set()
    for counter, (edge, g) in enumerate(profiles.groupby(EDGE, sort=True), 1):
        backgrounds = sorted(g.background_fingerprint.unique())
        common = sorted(set.intersection(*(set(t.substrate_pair_key) for _, t in g.groupby('background_fingerprint'))))
        ident = g.drop_duplicates('substrate_pair_key').set_index('substrate_pair_key').reindex(common)
        matrix = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta').reindex(index=common, columns=backgrounds).to_numpy(float)
        positive = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='both_positive').reindex(index=common, columns=backgrounds).all(axis=1).to_numpy()
        identities = []
        valid = np.ones(len(common), bool)
        for role in (1, 2):
            raw = ident[f'sub_{role}_smiles']
            single = raw.isna() | raw.eq('')
            molecule = raw.map(cmap)
            molecule.loc[single] = '<SINGLE>'
            descriptors = fmap.reindex(raw).to_numpy(float)
            descriptors[single] = 0.0
            valid &= molecule.notna().to_numpy() & np.isfinite(descriptors).all(axis=1)
            identities.append(molecule.fillna('<INVALID>').to_numpy(str))
        norms = np.linalg.norm(matrix, axis=1)
        check(f'panel_complete_unique_{counter}', np.isfinite(matrix).all() and len(set(zip(*identities))) == len(common))
        all_keep = (norms > 0) & valid
        role_matrices = []
        for identity in identities:
            index = np.array([ids[s] for s in identity[all_keep]], dtype=int)
            role_matrices.append(global_kernel[np.ix_(index, index)])
        full_kernels = {'add': (role_matrices[0] + role_matrices[1]) / 2, 'product': role_matrices[0] * role_matrices[1]}
        for family, k in full_kernels.items():
            mineig = float(eigh(k, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]) if len(k) else np.nan
            check(f'actual_{family}_kernel_symmetric_PSD_unitdiag_{counter}', not len(k) or (np.allclose(k, k.T, atol=1e-15) and np.all(np.diag(k) == 1) and (mineig >= -1e-09)), mineig)
            kernel_audit.append(dict(zip(EDGE, edge), family=family, n_rows=len(k), minimum_eigenvalue=mineig, kernel_sha256=array_hash(k)))
        for outcome in ('all', 'both_positive'):
            population = np.ones(len(common), bool) if outcome == 'all' else positive
            keep = population & all_keep
            meta = dict(zip(EDGE, edge), outcome=outcome, n_backgrounds=len(backgrounds), n_intersection_rows=len(common), n_population_rows=int(population.sum()), n_zero_rows=int((population & (norms == 0)).sum()), n_invalid_features=int((population & ~valid).sum()), n_nonzero_valid=int(keep.sum()))
            panel_rows.append(meta)
            yy = matrix[keep] / norms[keep, None]
            row_rms = norms[keep] / np.sqrt(len(backgrounds))
            a, b = (identities[0][keep], identities[1][keep])
            rowids = np.array(common)[keep]
            subset = np.flatnonzero(population[all_keep])
            kernels = {name: k[np.ix_(subset, subset)] for name, k in full_kernels.items()}
            for split in SPLITS:
                for fold in range(5):
                    train, test, purged, held = split_masks(a, b, split, fold)
                    nt, nv = (int(train.sum()), int(test.sum()))
                    status = 'no_test' if not nv else 'insufficient_training' if nt < MIN_TRAIN else 'scored'
                    fmeta = dict(**meta, split=split, fold=fold, n_train=nt, n_test=nv, n_cross_role_purged=purged, n_scored=nv if status == 'scored' else 0, n_test_role1_identities=len(set(a[test])), n_test_role2_identities=len(set(b[test])), n_test_role1_also_in_train=int(np.isin(a[test], list(set(a[train]) | set(b[train]))).sum()), n_test_role2_also_in_train=int(np.isin(b[test], list(set(a[train]) | set(b[train]))).sum()), status=status)
                    if split != 'pair':
                        check(f'actual_identity_purge_{counter}_{outcome}_{split}_{fold}', not (set(a[train]) | set(b[train])) & held)
                    pred = {}
                    if status == 'scored':
                        center = yy[train].mean(axis=0)
                        baseline = np.broadcast_to(center, (nv, len(center))).copy()
                        axis = np.linalg.svd(yy[train], full_matrices=False)[2][0]
                        for family, kernel in kernels.items():
                            training = kernel[np.ix_(train, train)]
                            cross = kernel[np.ix_(test, train)]
                            fitted = centered_krr(training, cross, yy[train])
                            pred.update({f'morgan_{family}_ridge_{alpha:g}': value for alpha, value in fitted.items()})
                        samplekey = (outcome, split)
                        if len(backgrounds) >= 3 and len(yy) >= 20 and (samplekey not in sampled):
                            sampled.add(samplekey)
                            for family, kernel in kernels.items():
                                training = kernel[np.ix_(train, train)]
                                cross = kernel[np.ix_(test, train)]
                                for alpha in ALPHAS:
                                    augmented = np.block([[training + alpha * np.eye(nt), np.ones((nt, 1))], [np.ones((1, nt)), np.zeros((1, 1))]])
                                    independent = solve(augmented, np.vstack([yy[train], np.zeros((1, yy.shape[1]))]), assume_a='sym')
                                    alternative = cross @ independent[:-1] + independent[-1]
                                    model = f'morgan_{family}_ridge_{alpha:g}'
                                    error = float(abs(alternative - pred[model]).max(initial=0))
                                    check(f'real_augmented_dual_{outcome}_{split}_{model}', error < 1e-09, error)
                                    spots.append(dict(**fmeta, model=model, alpha=alpha, max_prediction_difference=error, training_kernel_sha256=array_hash(training), cross_kernel_sha256=array_hash(cross), train_identity_sha256=hashlib.sha256('\n'.join(rowids[train]).encode()).hexdigest(), test_identity_sha256=hashlib.sha256('\n'.join(rowids[test]).encode()).hexdigest()))
                                    for j, identity in enumerate(rowids[test]):
                                        predictions.append(dict(zip(EDGE, edge), outcome=outcome, split=split, fold=fold, model=model, substrate_pair_key=identity, target=json.dumps(yy[test][j].tolist()), prediction=json.dumps(pred[model][j].tolist()), baseline=json.dumps(baseline[j].tolist()), training_axis=json.dumps(axis.tolist()), backgrounds=json.dumps(backgrounds)))
                    for model in KERNEL_MODELS:
                        score = {f'{m}_sum': 0.0 for m in METRICS} if status != 'scored' else losses(yy[test], pred[model], baseline, axis)
                        fold_rows.append(dict(**fmeta, model=model, **score))
                        if status == 'scored':
                            for floor in (0.0, 1.0, 5.0, 10.0):
                                strong = row_rms[test] >= floor
                                signal_rows.append({**fmeta, 'model': model, 'test_rms_floor_pp': floor, 'n_scored': int(strong.sum()), **losses(yy[test][strong], pred[model][strong], baseline[strong], axis)})
        if counter % 40 == 0:
            print(f'Structural kernel prediction: {counter}/302 edges', flush=True)
    panels = pd.DataFrame(panel_rows)
    new_folds = pd.DataFrame(fold_rows)
    new_signals = pd.DataFrame(signal_rows)
    foldkeys = EDGE + ['outcome', 'split', 'fold']
    ref = old_folds[old_folds.model.eq('training_centroid')].drop(columns='model')
    one = new_folds[new_folds.model.eq(KERNEL_MODELS[0])]
    supportcols = [c for c in ref if c not in foldkeys and (not c.endswith('_sum'))]
    compare_tables('E42_panel', panels, old_panels, EDGE + ['outcome'], [c for c in panels if c not in EDGE + ['outcome']])
    compare_tables('E42_fold_support', one, ref, foldkeys, supportcols)
    compare_tables('E42_centroid_scoring', one, ref, foldkeys, ['centroid_loss_sum'])
    compare_tables('E42_signal_support', new_signals[new_signals.model.eq(KERNEL_MODELS[0])], old_signals[old_signals.model.eq('training_centroid')], foldkeys + ['test_rms_floor_pp'], ['n_scored', 'n_train', 'n_test', 'n_cross_role_purged', 'centroid_loss_sum'])
    folds = pd.concat([old_folds, new_folds], ignore_index=True)
    signal_folds = pd.concat([old_signals, new_signals], ignore_index=True)
    edges, summaries, common, signal_edges, signal_summary, equal = aggregate(folds, panels, signal_folds, classes)
    old_edges = pd.read_csv(BASE / 'results/structure_context_geometry_edges.csv')
    compare_tables('E42_saved_models_edges', edges[edges.model.isin(old_edges.model.unique())], old_edges, EDGE + ['outcome', 'split', 'model'], ['n_scored', 'n_scored_folds', 'coverage', *[f'{m}_sum' for m in METRICS], *METRICS])
    old_summary = pd.read_csv(BASE / 'results/structure_context_geometry_summary.csv')
    compare_tables('E42_saved_models_summary', summaries[summaries.model.isin(old_edges.model.unique())], old_summary, ['outcome', 'split', 'model', 'gate', 'stratum_type', 'stratum'], ['n_edges', 'n_classes', 'n_scored_rows', *[f'{m}_equal_edge_mean' for m in METRICS]])
    check('complete302_edges_bothpop_3splits_10models', len(edges) == 302 * 2 * 3 * 10)
    check('complete_all5fold_records', len(folds) == 302 * 2 * 3 * 10 * 5)
    check('all20class_support_retained', summaries.loc[summaries.stratum_type.eq('reaction_class'), 'stratum'].nunique() == 20)
    check('loss_decomposition_all_edge_models', np.allclose(edges.improvement_sum, edges.along_improvement_sum + edges.orthogonal_improvement_sum, atol=1e-08, rtol=1e-10))
    for split, own in [('role1_new_purged', 1), ('role2_new_purged', 2)]:
        check(f'all_saved_identity_exclusion_{split}', folds.loc[folds.split.eq(split), f'n_test_role{own}_also_in_train'].eq(0).all())
    primary = summaries[summaries.outcome.eq('all') & summaries.gate.eq('bg_ge3_nonzero_ge20_full_primary') & summaries.stratum_type.eq('corpus')]
    check('same_primary63edges10classes15877rows', primary.n_edges.eq(63).all() and primary.n_classes.eq(10).all() and primary.n_scored_rows.eq(15877).all())
    check('all_inputs_unchanged', all((sha(BASE / p) == h for p, h in before.items())))
    outputs = {'panels': panels, 'folds': folds, 'edges': edges, 'summary': summaries, 'common_summary': common, 'equal_class_summary': equal, 'signal_folds': signal_folds, 'signal_edges': signal_edges, 'signal_summary': signal_summary, 'kernel_audit': pd.DataFrame(kernel_audit), 'spotchecks': pd.DataFrame(spots), 'spot_predictions': pd.DataFrame(predictions), 'checks': pd.DataFrame(CHECKS)}
    for name, frame in outputs.items():
        frame.to_csv(BASE / f'results/structural_kernel_context_{name}.csv', index=False)
    metadata = dict(n_checks=len(CHECKS), n_failed_checks=0, input_sha256=before, code_sha256=sha(__file__), output_sha256={name: sha(BASE / f'results/structural_kernel_context_{name}.csv') for name in outputs}, output_rows={name: len(frame) for name, frame in outputs.items()}, models=MODELS, alphas=ALPHAS, fingerprint=dict(family='Morgan', radius=2, n_bits=2048, chirality=False, bond_types=True, ring_membership=True, count_simulation=False, n_canonical_molecules=len(canonical)), objective='summed squared unit-vector error+alpha squared RKHS norm; unpenalized target centroid intercept; exact centered kernel eigensolve', kernels='Kadd=(K1+K2)/2; Kproduct=K1*K2; role Tanimoto; SINGLE-SINGLE1,SINGLE-molecule0; unit diagonal', descriptor_source='Exact copied E42 folds and signal folds; no descriptor refits or retuning', global_kernel_min_eigenvalue=global_min, global_kernel_sha256=array_hash(global_kernel), n_spotcheck_folds=len(sampled), max_spotcheck_prediction_difference=max((x['max_prediction_difference'] for x in spots)), blas_thread_limit='library_default_unrestricted', target='Exact E42 outcome-conditioned all-background delta unit vector; zero undefined; entire test vectors withheld', information='Only fixed canonical fingerprints. No heldout target in fit, training centering or line. Frozen E42 splits and purge; no cross-edge pooling', limits='Role-new is not both-new/scaffold; close analogues remain. Product kernel is statistical joint representation, not mechanism/nested interaction test. Normalization may itself introduce nonlinear role dependence. Positive signal floors are retrospective outcome gates. Alpha scales differ from standardized descriptors. No model/penalty selection or significance/prospective claim', versions=dict(numpy=np.__version__, pandas=pd.__version__, rdkit=rdkit.__version__))
    report = render_report(summaries, signal_summary, equal, metadata)
    (BASE / 'reports/STRUCTURAL_KERNEL_CONTEXT.md').write_text(report, encoding='utf-8')
    metadata['report_sha256'] = sha(BASE / 'reports/STRUCTURAL_KERNEL_CONTEXT.md')
    (BASE / 'audit/structural_kernel_context_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(primary[['split', 'model', 'n_edges', 'improvement_equal_edge_mean', 'along_improvement_equal_edge_mean', 'orthogonal_improvement_equal_edge_mean']].to_string(index=False))
    print(f'Passed {len(CHECKS)} checks; all E42 support reconciles', flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
