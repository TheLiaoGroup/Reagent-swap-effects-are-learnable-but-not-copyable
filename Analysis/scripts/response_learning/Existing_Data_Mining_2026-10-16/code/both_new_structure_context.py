"""Structure-only context prediction when both actual partners are unseen.

Cartesian identity-group holdouts cover each eligible row once. Structure-only
similarity gates describe applicability; they are not scaffold splits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from chemical_features import DESCRIPTOR_COLUMNS
from decision_opportunity_audit import markdown_table
from structural_kernel_context import centered_krr, fingerprints, ALPHAS, MODELS, sha
from structure_context_geometry import EDGE, assignment, fit_models, losses, METRICS, MIN_TRAIN
BASE = Path(__file__).resolve().parents[1]
NOVELTY_CEILINGS = (0.4, 0.6, 0.8, 0.95, 1.0)

def holdout(a, b, f1, f2):
    ha = np.array([assignment(x, 'molecule') for x in a])
    hb = np.array([assignment(x, 'molecule') for x in b])
    test = (ha == f1) & (hb == f2)
    train = (ha != f1) & (hb != f2)
    before = int(train.sum())
    held = set(a[test]) | set(b[test])
    train &= ~np.isin(a, list(held)) & ~np.isin(b, list(held))
    return (train, test, before - int(train.sum()), held)

def summarize(frame, classes, gate):
    rows = []
    for (outcome, model), group in frame.groupby(['outcome', 'model']):
        eligible = group.loc[group.n_backgrounds.ge(3) & group.n_two_actual_valid.ge(20) & group.coverage.eq(1)] if gate == 'full_primary' else group.loc[group.n_scored.gt(0)]
        for label in ('ALL', *classes):
            sub = eligible if label == 'ALL' else eligible.loc[eligible.reaction_type.eq(label)]
            row = dict(outcome=outcome, model=model, gate=gate, stratum=label, n_edges=len(sub), n_classes=sub.reaction_type.nunique(), n_scored_rows=int(sub.n_scored.sum()))
            for m in METRICS:
                row[m + '_equal_edge_mean'] = float(sub[m].mean())
                row[m + '_equal_edge_median'] = float(sub[m].median())
            row['positive_orthogonal_edge_fraction'] = float(sub.orthogonal_improvement.gt(0).mean()) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    checks = []

    def check(name, passed, detail=''):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        if not passed:
            raise AssertionError(name + ': ' + str(detail))
    a = np.array(['C', 'N', 'O', 'C', 'S', 'P'])
    b = np.array(['N', 'C', 'Cl', 'S', 'O', 'F'])
    cover = np.zeros(len(a), int)
    for f1 in range(5):
        for f2 in range(5):
            tr, te, _, held = holdout(a, b, f1, f2)
            check(f'toy_purge_{f1}_{f2}', not held & (set(a[tr]) | set(b[tr])))
            cover += te
    check('toy_all25fold_partition', np.all(cover == 1))
    inputs = ['results/matched_response_profiles.parquet', 'processed/molecular_features.parquet', 'results/condition_network_meta.csv', 'results/structural_kernel_context_panels.csv', 'code/structural_kernel_context.py', 'code/structure_context_geometry.py']
    hashes = {p: sha(BASE / p) for p in inputs}
    profiles = pd.read_parquet(BASE / inputs[0])
    features = pd.read_parquet(BASE / inputs[1])
    classes = sorted(pd.read_csv(BASE / inputs[2]).reaction_type.unique())
    canonical = sorted(features.canonical_smiles.unique())
    _, gram = fingerprints(canonical)
    ids = {s: i for i, s in enumerate(canonical)}
    cmap = features.set_index('smiles').canonical_smiles.to_dict()
    fmap = features.set_index('smiles')[DESCRIPTOR_COLUMNS]
    fold_rows, panels, sensitivity, rowmeta, spots = ([], [], [], [], [])
    sampled = set()
    for number, (edge, g) in enumerate(profiles.groupby(EDGE, sort=True), 1):
        backgrounds = sorted(g.background_fingerprint.unique())
        common = sorted(set.intersection(*(set(t.substrate_pair_key) for _, t in g.groupby('background_fingerprint'))))
        ident = g.drop_duplicates('substrate_pair_key').set_index('substrate_pair_key').reindex(common)
        matrix = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta').reindex(index=common, columns=backgrounds).to_numpy(float)
        positive = g.pivot(index='substrate_pair_key', columns='background_fingerprint', values='both_positive').reindex(index=common, columns=backgrounds).all(axis=1).to_numpy()
        identities, blocks = ([], [])
        valid = np.ones(len(common), bool)
        two_actual = np.ones(len(common), bool)
        for role in (1, 2):
            raw = ident[f'sub_{role}_smiles']
            single = raw.isna() | raw.eq('')
            molecule = raw.map(cmap)
            molecule.loc[single] = '<SINGLE>'
            descriptors = fmap.reindex(raw).to_numpy(float)
            descriptors[single] = 0.0
            identities.append(molecule.fillna('<INVALID>').to_numpy(str))
            blocks.append(descriptors)
            valid &= molecule.notna().to_numpy() & np.isfinite(descriptors).all(axis=1)
            two_actual &= ~single.to_numpy()
        norms = np.linalg.norm(matrix, axis=1)
        check(f'panel_unique_finite_{number}', np.isfinite(matrix).all() and len(set(zip(*identities))) == len(common))
        for outcome in ('all', 'both_positive'):
            population = np.ones(len(common), bool) if outcome == 'all' else positive
            keep = population & valid & two_actual & (norms > 0)
            meta = dict(zip(EDGE, edge), outcome=outcome, n_backgrounds=len(backgrounds), n_intersection_rows=len(common), n_population_rows=int(population.sum()), n_single_role_rows=int((population & ~two_actual).sum()), n_two_actual_valid=int(keep.sum()), n_zero_rows=int((population & (norms == 0)).sum()))
            panels.append(meta)
            a, b = (identities[0][keep], identities[1][keep])
            rowids = np.array(common)[keep]
            x = np.column_stack(blocks)[keep]
            y = matrix[keep] / norms[keep, None]
            rms = norms[keep] / np.sqrt(len(backgrounds))
            ri = [np.array([ids[s] for s in role], dtype=int) for role in (a, b)]
            rolek = [gram[np.ix_(i, i)] for i in ri]
            kernels = {'add': (rolek[0] + rolek[1]) / 2, 'product': rolek[0] * rolek[1]}
            coverage = np.zeros(len(y), int)
            for f1 in range(5):
                for f2 in range(5):
                    train, test, purged, held = holdout(a, b, f1, f2)
                    nt, nv = (int(train.sum()), int(test.sum()))
                    coverage += test
                    check(f'real_purge_{number}_{outcome}_{f1}_{f2}', not held & (set(a[train]) | set(b[train])))
                    status = 'no_test' if not nv else 'insufficient_training' if nt < MIN_TRAIN else 'scored'
                    fmeta = dict(**meta, fold_role1=f1, fold_role2=f2, n_train=nt, n_test=nv, n_cross_role_purged=purged, n_scored=nv if status == 'scored' else 0, status=status, n_test_identities=len(held), n_test_role1_identities=len(set(a[test])), n_test_role2_identities=len(set(b[test])))
                    if status == 'scored':
                        pred, axis = fit_models(x[train], y[train], x[test])
                        baseline = pred['training_centroid']
                        for family, kernel in kernels.items():
                            kk, kt = (kernel[np.ix_(train, train)], kernel[np.ix_(test, train)])
                            fits = centered_krr(kk, kt, y[train])
                            pred.update({f'morgan_{family}_ridge_{alpha:g}': value for alpha, value in fits.items()})
                        nearest = [gram[np.ix_(i[test], np.unique(i[train]))].max(axis=1) for i in ri]
                        both_ceiling = np.maximum(*nearest)
                        for j, identity in enumerate(rowids[test]):
                            rowmeta.append(dict(zip(EDGE, edge), outcome=outcome, substrate_pair_key=identity, fold_role1=f1, fold_role2=f2, n_train=nt, role1_nearest_training_similarity=float(nearest[0][j]), role2_nearest_training_similarity=float(nearest[1][j]), both_roles_max_similarity=float(both_ceiling[j]), role1_identical_fingerprint=bool(nearest[0][j] == 1), role2_identical_fingerprint=bool(nearest[1][j] == 1)))
                        if edge[0] not in sampled and len(backgrounds) >= 3 and (len(y) >= 20):
                            sampled.add(edge[0])
                            for family, kernel in kernels.items():
                                kk, kt = (kernel[np.ix_(train, train)], kernel[np.ix_(test, train)])
                                for alpha in ALPHAS:
                                    system = np.block([[kk + alpha * np.eye(nt), np.ones((nt, 1))], [np.ones((1, nt)), np.zeros((1, 1))]])
                                    coef = solve(system, np.vstack((y[train], np.zeros((1, y.shape[1])))), assume_a='sym')
                                    alternative = kt @ coef[:-1] + coef[-1]
                                    error = float(np.max(abs(alternative - pred[f'morgan_{family}_ridge_{alpha:g}'])))
                                    check(f'augmented_{number}_{family}_{alpha}', error < 1e-09, error)
                                    spots.append(dict(**fmeta, family=family, alpha=alpha, max_prediction_error=error))
                    for model in MODELS:
                        value = losses(y[test], pred[model], baseline, axis) if status == 'scored' else {m + '_sum': 0.0 for m in METRICS}
                        fold_rows.append(dict(**fmeta, model=model, **value))
                        if status == 'scored':
                            gates = [('test_rms_floor', floor, rms[test] >= floor) for floor in (0.0, 1.0, 5.0, 10.0)]
                            gates += [('both_role_similarity_ceiling', ceiling, both_ceiling <= ceiling) for ceiling in NOVELTY_CEILINGS]
                            for kind, threshold, selected in gates:
                                sensitivity.append({**fmeta, 'model': model, 'scoring_kind': kind, 'threshold': threshold, 'n_scored': int(selected.sum()), **losses(y[test][selected], pred[model][selected], baseline[selected], axis)})
            check(f'exact_once_{number}_{outcome}', np.all(coverage == 1))
        if number % 40 == 0:
            print(f'Both-new structure {number}/302', flush=True)
    folds, panels = (pd.DataFrame(fold_rows), pd.DataFrame(panels))
    keys = EDGE + ['outcome', 'model']
    sums = ['n_scored', *[m + '_sum' for m in METRICS]]
    edges = folds.groupby(keys)[sums].sum().reset_index().merge(panels, on=EDGE + ['outcome'], validate='many_to_one')
    edges['coverage'] = edges.n_scored / edges.n_two_actual_valid.replace(0, np.nan)
    for m in METRICS:
        edges[m] = edges[m + '_sum'] / edges.n_scored.replace(0, np.nan)
    summary = pd.concat([summarize(edges, classes, gate) for gate in ('any_scored', 'full_primary')], ignore_index=True)
    sens = pd.DataFrame(sensitivity)
    sense = sens.groupby(keys + ['scoring_kind', 'threshold'])[sums].sum().reset_index().merge(panels, on=EDGE + ['outcome'], validate='many_to_one')
    for m in METRICS:
        sense[m] = sense[m + '_sum'] / sense.n_scored.replace(0, np.nan)
    ss = []
    for (outcome, model, kind, threshold), g in sense.groupby(['outcome', 'model', 'scoring_kind', 'threshold']):
        for label in ('ALL', *classes):
            sub = g.loc[g.n_backgrounds.ge(3) & g.n_scored.ge(20)]
            if label != 'ALL':
                sub = sub.loc[sub.reaction_type.eq(label)]
            record = dict(outcome=outcome, model=model, scoring_kind=kind, threshold=threshold, stratum=label, n_edges=len(sub), n_classes=sub.reaction_type.nunique(), n_scored_rows=int(sub.n_scored.sum()))
            record.update({m + '_equal_edge_mean': float(sub[m].mean()) for m in METRICS})
            ss.append(record)
    ss = pd.DataFrame(ss)
    equal = summary.loc[summary.stratum.ne('ALL')].groupby(['outcome', 'model', 'gate']).agg(n_classes_supported=('n_edges', lambda s: int(s.gt(0).sum())), improvement_equal_class_mean=('improvement_equal_edge_mean', 'mean'), orthogonal_improvement_equal_class_mean=('orthogonal_improvement_equal_edge_mean', 'mean')).reset_index()
    check('all302edges2pop10models25folds', len(folds) == 302 * 2 * 10 * 25)
    check('all20classes_retained', summary.loc[summary.stratum.ne('ALL'), 'stratum'].nunique() == 20)
    check('all_loss_closure', np.allclose(edges.improvement_sum, edges.along_improvement_sum + edges.orthogonal_improvement_sum, atol=1e-08))
    check('all_inputs_unchanged', hashes == {p: sha(BASE / p) for p in inputs})
    for name, frame in (('panels', panels), ('edges', edges), ('summary', summary), ('signal_novelty_edges', sense), ('signal_novelty_summary', ss), ('equal_class_summary', equal), ('spotchecks', pd.DataFrame(spots)), ('checks', pd.DataFrame(checks))):
        frame.to_csv(BASE / f'results/both_new_structure_context_{name}.csv', index=False)
    folds.to_parquet(BASE / 'results/both_new_structure_context_folds.parquet', index=False)
    sens.to_parquet(BASE / 'results/both_new_structure_context_scoring_folds.parquet', index=False)
    rowmeta = pd.DataFrame(rowmeta)
    rowmeta.to_parquet(BASE / 'results/both_new_structure_context_row_novelty.parquet', index=False)
    primary = summary.loc[summary.outcome.eq('all') & summary.gate.eq('full_primary') & summary.stratum.eq('ALL')]
    columns = ['model', 'n_edges', 'n_classes', 'n_scored_rows', 'improvement_equal_edge_mean', 'along_improvement_equal_edge_mean', 'orthogonal_improvement_equal_edge_mean', 'positive_orthogonal_edge_fraction']
    metadata = dict(checks=len(checks), failures=0, input_hashes=hashes, code_sha256=sha(__file__), fold_rows=len(folds), scoring_fold_rows=len(sens), novelty_rows=len(rowmeta), novelty_ceilings=NOVELTY_CEILINGS, models=MODELS, sampled_augmented_classes=sorted(sampled), primary=primary.to_dict('records'), split='25 Cartesian canonical identity hash folds; both roles actual molecules; purge held test molecules from either training role', limitation='Historical within-edge/background interpolation; no scaffold/temporal/prospective test. Full intersections and nonzero directions outcome-conditioned. Fingerprint equality may remain despite canonical identity purge.')
    report = "# Both-Partner-New Structural Context Prediction\n\n## Information condition\n\nBoth actual molecular partners of every scored test row are absent from either role of this edge's training set. The response target, fixed Morgan fingerprints, additive/product Tanimoto kernels, all three penalties, training-centroid reference and training-only uncentered line follow E42/E44. All302 registered edges and20classes remain in accounting; SINGLE roles cannot satisfy both-partner novelty and are explicitly excluded. Unit normalization is target definition, not an input feature. No model is selected on heldout scores.\n\nEach role uses the frozen E42 five canonical-molecule hash groups. The25 Cartesian group pairs partition eligible test rows exactly once. For test group(f1,f2), training excludes role1groupf1 and role2groupf2, then removes every test molecule from either training role. Test rows are scored only with at least5training rows. Empty and insufficient-training folds remain in saved outputs. Both-new denotes identity, not scaffold or remote chemical extrapolation. Models stay within a known edge and known background coordinate system.\n\nTraining size and purging differ from E44; comparisons are not a pure estimate of an identity-leakage effect. The bothactual population is a further explicit scope condition. All fixed penalties, positive and negative results, and full/partial support remain visible.\n\n## Full-Coverage Primary Cohort\n\nAt least3backgrounds,20eligible nonzero rows, and every eligible row scored are required. Loss improvements are dimensionless per-row squared unit-vector error, not yield gains. Along-line and orthogonal gains use the training-only leading signed line; closure is arithmetic and is not independent evidence.\n\n" + markdown_table(primary[columns]) + '\n\n'
    report += '## Equal-Class Means\n\n' + markdown_table(equal.loc[equal.outcome.eq('all') & equal.gate.eq('full_primary')]) + '\n\n'
    display = ss.loc[ss.outcome.eq('all') & ss.stratum.eq('ALL') & ss.model.isin(['descriptor_ridge_100', 'morgan_add_ridge_10', 'morgan_product_ridge_10'])]
    report += '## Fixed-Fit Signal and Structural Applicability\n\nThese gates change test scoring only. Signal floors0/1/5/10pp depend on observed target amplitudes and are not noise corrections. Structural ceilings0.4/0.6/0.8/0.95/1 are fixed outcome-blind nearest-training Tanimoto restrictions: BOTH role-specific maxima must be at most the ceiling. Each maximum uses molecules in the same ordered training role. These gates do not exclude close scaffolds categorically, and low-support subsets are retained with undefined means where necessary. Fingerprint maxsimilarity1 can remain after canonical identity exclusion because chirality is omitted and bit representations may coincide. Detailed per-row novelty is saved.\n\nThe scoring gate requires at least20scored rows and3backgrounds, and can admit partially covered edges; it therefore has a separate support denominator from full-primary. The centroid and every model use identical cells inside a gate.\n\n' + markdown_table(display[['model', 'scoring_kind', 'threshold', 'n_edges', 'n_classes', 'n_scored_rows', 'improvement_equal_edge_mean', 'orthogonal_improvement_equal_edge_mean']]) + '\n\n'
    support = summary.loc[summary.gate.eq('full_primary') & summary.model.eq('training_centroid') & summary.stratum.ne('ALL'), ['outcome', 'stratum', 'n_edges', 'n_classes', 'n_scored_rows']]
    report += '## All-Class Support\n\n' + markdown_table(support) + '\n\n'
    report += f'## Verification and Limits\n\nProducer checks:{len(checks)}, failures:0. The25fold partition, bothrole exclusion, all302edges and20classes, numerical loss closure and input hashes were checked. Augmented-intercept dual solves independently replay actual kernel predictions in{len(sampled)}classes. Run `python code/both_new_structure_context.py` from the analysis directory. Outputs use prefix`both_new_structure_context_`; parquet retains all150,? fold rows with actual row count in metadata, every scoring gate and per-row structural novelty.\n\nThe target is a retrospective complete observed context direction; nonzero and bothpositive selection remain outcome-conditioned. Purged canonical identities do not establish scaffold novelty, measurement calibration or a mechanism. Fingerprints can capture consistent measured-response structure, including assay-correlated effects. A product kernel is a statistical joint representation and its comparison with the additive kernel is not a nested mechanistic interaction test. No independent experiment, formal sampling uncertainty or prospective guarantee is added.\n'.replace('all150,? fold rows', f'all{len(folds):,}fold rows')
    (BASE / 'reports/BOTH_NEW_STRUCTURE_CONTEXT.md').write_text(report, encoding='utf-8')
    (BASE / 'audit/both_new_structure_context_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(primary[columns].to_string(index=False))
    print(json.dumps(dict(checks=len(checks), failures=0, folds=len(folds), primary_edges=primary.n_edges.unique().tolist())))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
