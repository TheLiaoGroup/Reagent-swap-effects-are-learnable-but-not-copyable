"""E63 exact descriptive decomposition of frozen E62 raw-vector MSE.

No fitting, resplitting, threshold search, outcome exclusion or source edits.
Centered alignment is amplitude weighted; it is not unit-shape prediction."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
ROOT = ANALYSIS.parent
E62 = ANALYSIS / 'Matched_Representation_E62'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
MODELS = ['centroid', 'codebook_continuous', 'codebook_hard', 'codebook_mixture', 'pca_continuous']
INPUTS = [E62 / f'results/e62_{name}' for name in ['predictions.parquet', 'edge_metrics_1548.csv', 'class_metrics_20.csv', 'summary.csv', 'support_bridge_1548.csv', 'support_classes_20.csv']] + [E62 / 'metadata/e62_metadata.json', Path(__file__)]
NEGATIVE_ROUNDOFF_TOL_PP2 = 1e-09
CLOSURE_TOL_PP2 = 1e-08
LOSS = ['offset_mse_pp2', 'amplitude_mse_pp2', 'alignment_mse_pp2', 'total_mse_pp2']
DIFF = ['offset_difference_pp2', 'amplitude_difference_pp2', 'alignment_difference_pp2', 'mse_difference_pp2']
PAIR = ['nominal_k', 'comparison', 'lhs_model', 'rhs_model']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pins(paths):
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}

def decompose(y, p):
    """Stable centering keeps exactly constant saved vectors exactly centered zero."""
    y, p = (np.asarray(y, dtype=float), np.asarray(p, dtype=float))
    m = y[:, 0] + (y - y[:, :1]).mean(axis=1)
    n = p[:, 0] + (p - p[:, :1]).mean(axis=1)
    yc, pc = (y - m[:, None], p - n[:, None])
    a, d = (np.sqrt(np.mean(yc * yc, axis=1)), np.sqrt(np.mean(pc * pc, axis=1)))
    cov = np.mean(yc * pc, axis=1)
    alignment_raw = 2 * (a * d - cov)
    if np.min(alignment_raw) < -NEGATIVE_ROUNDOFF_TOL_PP2:
        raise ValueError(f'Alignment violates nonnegativity: {np.min(alignment_raw)} pp2')
    alignment = np.maximum(alignment_raw, 0)
    defined = (a > 0) & (d > 0)
    cosine = np.full(len(y), np.nan)
    np.divide(cov, a * d, out=cosine, where=defined)
    offset, amplitude = ((m - n) ** 2, (a - d) ** 2)
    direct = np.mean((y - p) ** 2, axis=1)
    component_sum = offset + amplitude + alignment
    return dict(true_mean_pp=m, predicted_mean_pp=n, true_centered_rms_pp=a, predicted_centered_rms_pp=d, centered_cross_product_pp2=cov, centered_cosine=cosine, cosine_defined=defined, true_constant_vector=np.all(y == y[:, :1], axis=1), predicted_constant_vector=np.all(p == p[:, :1], axis=1), offset_mse_pp2=offset, amplitude_mse_pp2=amplitude, alignment_mse_pp2=alignment, alignment_raw_pp2=alignment_raw, alignment_roundoff_adjustment_pp2=alignment - alignment_raw, total_mse_pp2=direct, component_sum_pp2=component_sum, closure_error_pp2=component_sum - direct, raw_closure_error_pp2=offset + amplitude + alignment_raw - direct)

def synthetic_checks():
    rows = []
    cases = [('constant_offset', [3, 3, 3], [5, 5, 5], [4, 0, 0], None), ('same_direction_amplitude', [-1, 0, 1], [-2, 0, 2], [0, 2 / 3, 0], 1), ('opposite_direction', [-1, 0, 1], [1, 0, -1], [0, 0, 8 / 3], -1), ('orthogonal_centered', [-1, 1, 0, 0], [0, 0, -1, 1], [0, 0, 1], 0), ('zero_true_not_missing', [0, 0, 0], [1, 2, 3], [4, 2 / 3, 0], None), ('zero_prediction', [-1, 0, 1], [0, 0, 0], [0, 2 / 3, 0], None), ('noninteger_constants', [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.01, 0, 0], None)]
    for name, y, p, expected, cosine in cases:
        values = decompose([y], [p])
        observed = [values[c][0] for c in LOSS[:3]]
        cosine_ok = np.isnan(values['centered_cosine'][0]) if cosine is None else abs(values['centered_cosine'][0] - cosine) < 1e-12
        rows.append(dict(check=name, passed=bool(np.allclose(observed, expected, atol=1e-12, rtol=0) and cosine_ok and (abs(values['closure_error_pp2'][0]) < 1e-12))))
    rng = np.random.default_rng(20261017)
    values = decompose(rng.normal(size=(100, 15)), rng.normal(size=(100, 15)))
    rows.append(dict(check='fixed_random_formula_closure', passed=bool(np.max(np.abs(values['closure_error_pp2'])) < 1e-12)))
    return rows

def row_weights(group):
    edges = group[EDGE].drop_duplicates()
    sizes = group.groupby(EDGE).vector_id.transform('size').to_numpy(float)
    n_class_edges = edges.groupby('reaction_type').size()
    return dict(pooled=np.full(len(group), 1 / len(group)), equal_edge=1 / (len(edges) * sizes), equal_class_edge_balanced=1 / (edges.reaction_type.nunique() * group.reaction_type.map(n_class_edges).to_numpy() * sizes))

def weighted_summary(group, columns, extra):
    rows = []
    for weighting, w in row_weights(group).items():
        rows.append(dict(**extra, weighting=weighting, n_test_vectors=len(group), n_test_edges=len(group[EDGE].drop_duplicates()), n_test_classes=group.reaction_type.nunique(), full_weight_mass=float(w.sum()), **{c: float(w @ group[c].to_numpy(float)) for c in columns}))
    return rows

def main():
    started = time.perf_counter()
    for folder in ['results', 'audit', 'metadata', 'reports']:
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    before = pins(INPUTS)
    source_sha = sha(Path(__file__))
    upstream = json.loads(INPUTS[6].read_text(encoding='utf-8'))
    for path in INPUTS[:6]:
        key = str(path.relative_to(ROOT))
        if upstream['output_sha256'].get(key) != before[key]:
            raise ValueError(f'Input differs from frozen E62 manifest: {key}')
    boundary = synthetic_checks()
    boundary_path = BASE / 'audit/e63_formula_checks.csv'
    pd.DataFrame(boundary).to_csv(boundary_path, index=False)
    if not all((r['passed'] for r in boundary)):
        raise ValueError('Synthetic formula checks failed')
    p = pd.read_parquet(INPUTS[0], columns=[*EDGE, 'vector_id', 'substrate_pair_key', 'nominal_k', 'model', 'n_backgrounds', 'allzero_vector', 'target_vector', 'prediction_vector', 'vector_mse_pp2'])
    support = pd.read_csv(INPUTS[4])
    vectors = p.drop(columns=['target_vector', 'prediction_vector']).rename(columns={'vector_mse_pp2': 'e62_mse_pp2'})
    chunks = []
    zero_flags_ok = True
    for b, g in p.groupby('n_backgrounds', sort=True):
        y, prediction = (np.vstack(g.target_vector), np.vstack(g.prediction_vector))
        if y.shape != prediction.shape or y.shape[1] != b:
            raise ValueError('Background dimension mismatch')
        zero_flags_ok &= bool(np.array_equal(np.all(y == 0, axis=1), g.allzero_vector))
        chunk = pd.DataFrame(decompose(y, prediction), index=g.index)
        chunks.append(chunk)
    vectors = vectors.join(pd.concat(chunks).sort_index())
    vectors['roundoff_clamped'] = vectors.alignment_roundoff_adjustment_pp2.gt(0)
    written = [boundary_path]
    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))

    def save(frame, name):
        path = BASE / 'results' / name
        if path.suffix == '.parquet':
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        written.append(path)
    check('complete_expected_prediction_support', len(vectors) == 259560 and vectors.vector_id.nunique() == 25956 and (vectors.reaction_type.nunique() == 14) and (len(vectors[EDGE].drop_duplicates()) == 93), f'{len(vectors)} rows')
    check('all_ten_models_share_all_vectors', vectors.groupby('vector_id').size().eq(10).all() and (not vectors.duplicated(['vector_id', 'nominal_k', 'model']).any()) and (set(vectors.model) == set(MODELS)), 'All 25,956 vectors retained for 2K x 5models')
    check('literal_allzero_flags_match_saved_arrays', zero_flags_ok and int(vectors.allzero_vector.sum()) == 99240, '9,924 zero vectors per model/K; missing was never filled')
    check('nonnegative_components', np.isfinite(vectors[LOSS].to_numpy()).all() and vectors[LOSS].ge(0).all().all(), f'raw alignment minimum {vectors.alignment_raw_pp2.min():.17g}')
    closure = float(vectors.closure_error_pp2.abs().max())
    check('per_vector_three_components_close_to_direct_MSE', closure < CLOSURE_TOL_PP2, f'max absolute residual {closure:.17g} pp2')
    loss_error = float((vectors.total_mse_pp2 - vectors.e62_mse_pp2).abs().max())
    check('direct_MSE_matches_frozen_E62_all_rows', loss_error < CLOSURE_TOL_PP2, f'max difference {loss_error:.17g} pp2')
    defined = vectors.cosine_defined
    zero_centered = vectors.true_centered_rms_pp.eq(0) | vectors.predicted_centered_rms_pp.eq(0)
    check('cosine_NA_exactly_on_centered_degeneracy', np.array_equal(~defined, zero_centered) and vectors.loc[~defined, 'centered_cosine'].isna().all() and np.isfinite(vectors.loc[defined, 'centered_cosine']).all(), f'{int((~defined).sum())} undefined model/K rows')
    check('alignment_zero_when_either_centered_RMS_zero', vectors.loc[zero_centered, 'alignment_mse_pp2'].eq(0).all(), int(zero_centered.sum()))
    check('cosine_range_with_machine_tolerance', vectors.loc[defined, 'centered_cosine'].abs().le(1 + 1e-12).all(), f'max abs {vectors.centered_cosine.abs().max():.17g}; cosine not clamped')
    check('exact_constant_vectors_center_to_exact_zero', np.array_equal(vectors.true_constant_vector, vectors.true_centered_rms_pp.eq(0)) and np.array_equal(vectors.predicted_constant_vector, vectors.predicted_centered_rms_pp.eq(0)), 'Stable centering, no chemical-zero tolerance')
    save(vectors, 'e63_vector_components.parquet')
    counter = dict(n_test_vectors=('vector_id', 'size'), n_allzero_vectors=('allzero_vector', 'sum'), n_cosine_defined=('cosine_defined', 'sum'), n_true_constant_vectors=('true_constant_vector', 'sum'), n_predicted_constant_vectors=('predicted_constant_vector', 'sum'), n_roundoff_clamped=('roundoff_clamped', 'sum'))
    edge = vectors.groupby([*EDGE, 'nominal_k', 'model'], as_index=False).agg(**counter, **{c: (c, 'mean') for c in LOSS}, e62_mse_pp2=('e62_mse_pp2', 'mean'), closure_error_pp2=('closure_error_pp2', 'mean'), max_abs_closure_pp2=('closure_error_pp2', lambda s: s.abs().max()))
    modelgrid = pd.DataFrame([(k, model) for k in [2, 3] for model in MODELS], columns=['nominal_k', 'model'])
    edge = support[[*EDGE, 'panel_status', 'n_candidate_vectors', 'n_scored_vectors']].merge(modelgrid, how='cross').merge(edge, on=[*EDGE, 'nominal_k', 'model'], how='left', validate='one_to_one')
    countcols = list(counter)
    edge[countcols] = edge[countcols].fillna(0).astype(int)
    edge['supported'] = edge.n_test_vectors.gt(0)
    edge['rmse_pp'] = np.sqrt(edge.total_mse_pp2)
    save(edge, 'e63_edge_components_1548.csv')
    classes = edge.groupby(['reaction_type', 'nominal_k', 'model'], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), **{c: (c, 'sum') for c in countcols}, **{c: (c, 'mean') for c in LOSS}, e62_mse_pp2=('e62_mse_pp2', 'mean'), closure_error_pp2=('closure_error_pp2', 'mean'))
    classes['supported'] = classes.n_test_edges.gt(0)
    classes['rmse_pp'] = np.sqrt(classes.total_mse_pp2)
    save(classes, 'e63_class_components_20.csv')
    summary, zero_summary, roundoff = ([], [], [])
    for (k, model), g in vectors.groupby(['nominal_k', 'model'], sort=True):
        extra = dict(nominal_k=int(k), model=model)
        rows = weighted_summary(g, LOSS + ['e62_mse_pp2', 'alignment_roundoff_adjustment_pp2'], extra)
        for row in rows:
            row.update(n_allzero_vectors=int(g.allzero_vector.sum()), n_cosine_defined=int(g.cosine_defined.sum()), n_true_constant_vectors=int(g.true_constant_vector.sum()), n_predicted_constant_vectors=int(g.predicted_constant_vector.sum()), rmse_pp=float(np.sqrt(row['total_mse_pp2'])))
        summary.extend(rows)
        for zero, sg in g.groupby('allzero_vector', sort=True):
            zero_summary.extend(weighted_summary(sg, LOSS, dict(**extra, stratum='allzero' if zero else 'nonzero', weighting_population='renormalized_within_stratum')))
        roundoff.append(dict(**extra, n_rows=len(g), n_clamped=int(g.roundoff_clamped.sum()), raw_alignment_min_pp2=float(g.alignment_raw_pp2.min()), max_clamp_pp2=float(g.alignment_roundoff_adjustment_pp2.max()), max_abs_raw_closure_pp2=float(g.raw_closure_error_pp2.abs().max()), max_abs_clamped_closure_pp2=float(g.closure_error_pp2.abs().max()), max_abs_e62_MSE_difference_pp2=float((g.total_mse_pp2 - g.e62_mse_pp2).abs().max())))
    summary = pd.DataFrame(summary)
    save(summary, 'e63_summary.csv')
    save(pd.DataFrame(zero_summary), 'e63_zero_nonzero_component_summary.csv')
    save(pd.DataFrame(roundoff), 'e63_roundoff_diagnostics.csv')
    original = pd.read_csv(INPUTS[3]).set_index(['nominal_k', 'model', 'weighting']).sort_index()
    current = summary.set_index(['nominal_k', 'model', 'weighting']).sort_index()
    aggregate_error = float(np.max(np.abs(original.mse_pp2 - current.total_mse_pp2)))
    check('all30_weighted_totals_match_frozen_E62', aggregate_error < CLOSURE_TOL_PP2 and np.allclose(current.full_weight_mass, 1, atol=1e-12, rtol=0), f'max aggregate MSE difference {aggregate_error:.17g}')
    e62edge = pd.read_csv(INPUTS[1]).set_index([*EDGE, 'nominal_k', 'model']).sort_index()
    e63edge = edge.set_index([*EDGE, 'nominal_k', 'model']).sort_index()
    e62class = pd.read_csv(INPUTS[2]).set_index(['reaction_type', 'nominal_k', 'model']).sort_index()
    e63class = classes.set_index(['reaction_type', 'nominal_k', 'model']).sort_index()
    check('all_edge_class_MSEs_match_E62_with_NA', np.allclose(e62edge.mse_pp2, e63edge.total_mse_pp2, atol=CLOSURE_TOL_PP2, rtol=0, equal_nan=True) and np.allclose(e62class.mse_pp2, e63class.total_mse_pp2, atol=CLOSURE_TOL_PP2, rtol=0, equal_nan=True), 'Full registered grids')
    check('all1548edges20classes_and_unsupported_NA', len(edge) == 15500 and len(classes) == 200 and (edge[EDGE].drop_duplicates().shape[0] == 1550) and (classes.reaction_type.nunique() == 20) and edge.loc[~edge.supported, LOSS].isna().all().all() and classes.loc[~classes.supported, LOSS].isna().all().all(), 'Unsupported losses remain NA')
    index = [*EDGE, 'nominal_k', 'vector_id', 'allzero_vector']
    wide = vectors.pivot(index=index, columns='model', values=LOSS)
    comparisons = [('centroid_minus_' + model, 'centroid', model) for model in MODELS] + [('hard_minus_continuous', 'codebook_hard', 'codebook_continuous')]
    paired, pair_summary, contributions = ([], [], [])
    for comparison, lhs, rhs in comparisons:
        frame = wide.index.to_frame(index=False)
        frame['comparison'], frame['lhs_model'], frame['rhs_model'] = (comparison, lhs, rhs)
        for loss, difference in zip(LOSS, DIFF):
            frame[difference] = (wide[loss, lhs] - wide[loss, rhs]).to_numpy()
        for k, g in frame.groupby('nominal_k', sort=True):
            extra = dict(nominal_k=int(k), comparison=comparison, lhs_model=lhs, rhs_model=rhs)
            pair_summary.extend(weighted_summary(g, DIFF, extra))
            for weighting, w in row_weights(g).items():
                for zero, label in [(True, 'allzero'), (False, 'nonzero')]:
                    mask = g.allzero_vector.to_numpy() == zero
                    contributions.append(dict(**extra, weighting=weighting, stratum=label, n_test_vectors=int(mask.sum()), full_sample_weight_mass=float(w[mask].sum()), **{c: float(w[mask] @ g.loc[mask, c].to_numpy()) for c in DIFF}))
        paired.append(frame)
    paired = pd.concat(paired, ignore_index=True)
    pair_summary, contributions = (pd.DataFrame(pair_summary), pd.DataFrame(contributions))
    save(pair_summary, 'e63_paired_component_summary.csv')
    save(contributions, 'e63_zero_nonzero_full_weight_contributions.csv')
    pairgrid = pd.DataFrame([(k, comparison, lhs, rhs) for k in [2, 3] for comparison, lhs, rhs in comparisons], columns=PAIR)
    pe = paired.groupby([*EDGE, *PAIR], as_index=False).agg(n_test_vectors=('vector_id', 'size'), **{c: (c, 'mean') for c in DIFF})
    pe = support[[*EDGE, 'panel_status']].merge(pairgrid, how='cross').merge(pe, on=[*EDGE, *PAIR], how='left', validate='one_to_one')
    pe['n_test_vectors'] = pe.n_test_vectors.fillna(0).astype(int)
    pe['supported'] = pe.n_test_vectors.gt(0)
    save(pe, 'e63_paired_edge_components_1548.csv')
    pc = pe.groupby(['reaction_type', *PAIR], as_index=False).agg(n_registered_edges=('component_pair', 'size'), n_test_edges=('supported', 'sum'), n_test_vectors=('n_test_vectors', 'sum'), **{c: (c, 'mean') for c in DIFF})
    pc['supported'] = pc.n_test_edges.gt(0)
    save(pc, 'e63_paired_class_components_20.csv')
    pair_closure = (paired[DIFF[:3]].sum(axis=1) - paired.mse_difference_pp2).abs().max()
    check('paired_components_sum_to_total_difference', pair_closure < 2 * CLOSURE_TOL_PP2, f'max residual {pair_closure:.17g} pp2')
    contribution_sums = contributions.groupby([*PAIR, 'weighting'])[DIFF].sum().sort_index()
    pair_totals = pair_summary.set_index([*PAIR, 'weighting'])[DIFF].sort_index()
    contribution_error = float(np.max(np.abs(contribution_sums.to_numpy() - pair_totals.to_numpy())))
    check('zero_nonzero_contributions_recover_all_full_weight_differences', contribution_error < CLOSURE_TOL_PP2, f'max difference {contribution_error:.17g} pp2')
    check('paired_grids_keep_complete_registered_support', len(pe) == 1550 * 12 and len(pc) == 20 * 12 and pe.loc[~pe.supported, DIFF].isna().all().all() and pc.loc[~pc.supported, DIFF].isna().all().all(), 'Six comparisons x two K, including zero centroid baseline self-gain')
    check('inputs_and_source_unchanged', before == pins(INPUTS) and source_sha == sha(Path(__file__)), 'E62 and discriminant memo are read-only')
    checkpath = BASE / 'audit/e63_production_checks.csv'
    pd.DataFrame(checks).to_csv(checkpath, index=False)
    written.append(checkpath)
    meta = dict(analysis='E63 frozen raw response information loss decomposition', code_sha256=source_sha, fitted_models=0, input_sha256_before=before, input_sha256_after=pins(INPUTS), output_sha256=pins(written), n_formula_checks=len(boundary), n_formula_passed=sum((r['passed'] for r in boundary)), production_checks=checks, failures=sum((not r['passed'] for r in boundary)) + sum((not r['passed'] for r in checks)), n_prediction_rows=len(vectors), n_vectors=vectors.vector_id.nunique(), n_scored_edges=93, n_registered_edges=1550, n_scored_classes=14, n_registered_classes=20, n_allzero_vectors=9924, n_cosine_undefined_model_capacity_rows=int((~vectors.cosine_defined).sum()), n_roundoff_clamped=int(vectors.roundoff_clamped.sum()), max_roundoff_adjustment_pp2=float(vectors.alignment_roundoff_adjustment_pp2.max()), max_abs_closure_pp2=closure, parameters=dict(target='Frozen E62 raw pp delta vectors over all known backgrounds', formula='offset=(mean(y)-mean(p))^2; amplitude=(RMS(center(y))-RMS(center(p)))^2; alignment=2*(RMS(center(y))*RMS(center(p))-mean(center(y)*center(p)))', stable_centering='mean(x)=x[0]+mean(x-x[0]); exact constant vectors stay centered zero', negative_alignment_roundoff_tolerance_pp2=NEGATIVE_ROUNDOFF_TOL_PP2, closure_tolerance_pp2=CLOSURE_TOL_PP2, roundoff='Raw alignment and positive clamp adjustment retained per row and in summary diagnostics; any more-negative value raises. Cosine is not clamped.', cosine='NA exactly if either centered RMS is zero; model-specific defined supports cannot replace common full population', weighting='E62: within-vector mean over backgrounds; pooled vectors; edge equal with vector equal within edge; class equal with edges equal within class. RMSE=sqrt(aggregated MSE).', contrasts='centroid-minus-model positive is component gain; hard-minus-continuous positive is greater hard component loss; all signed terms retained', strata='Full-weight contributions retain original population row weights; stratum summary separately renormalizes within observed zero/nonzero population'), limitations=['Exact algebraic decomposition of existing MSE, not independent experimental evidence.', 'Alignment loss is centered-amplitude weighted, not normalized-shape prediction or shape retraining.', 'Known edges/backgrounds and support-conditioned double-identity holdout only; no new-background/scaffold/time validation.', 'No mechanism, causal effect, true state number, zero-generating process or significance identified.', 'Allzero is observed data; missing is not zero; no model-specific directional population substitution.'], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__), elapsed_seconds=time.perf_counter() - started)
    (BASE / 'metadata/e63_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if meta['failures']:
        raise SystemExit('E63 consistency checks failed')
    print(f'E63 complete: {len(vectors)} rows, {len(boundary)}/{len(boundary)} formula, {len(checks)}/{len(checks)} consistency checks, zero fitting', flush=True)
if __name__ == '__main__':
    main()
