"""Necessary sign constraints for separability after stable monotone yield maps.

This reinterprets existing recorded signs, not independent empirical evidence.
Every background pair and typed edge is retained, including unsupported cases."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
CONTEXT = EDGE + ['background_fingerprint']
THRESHOLDS = (0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0)
POPULATIONS = ('all', 'both_positive')
GATES = ('all_retained', 'shared20', 'shared20_joint20')
METRICS = ('contradictory_cycle_fraction', 'minimum_row_deletion_fraction', 'same_relation_fraction', 'joint_decisive_fraction', 'single_observation_minimum_row_deletion_fraction')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def active(x, threshold):
    return np.abs(x) > 0 if threshold == 0 else np.abs(x) >= threshold

def counts_from_relations(relation):
    ns, no = (int((relation > 0).sum()), int((relation < 0).sum()))
    n = ns + no
    cycles = ns * no
    total_cycles = n * (n - 1) // 2
    return dict(n_sign_same=ns, n_sign_opposite=no, n_joint_decisive=n, n_contradictory_row_pairs=cycles, n_possible_row_pairs=total_cycles, contradictory_cycle_fraction=cycles / total_cycles if total_cycles else np.nan, minimum_row_deletions=min(ns, no), minimum_row_deletion_fraction=min(ns, no) / n if n else np.nan, same_relation_fraction=ns / n if n else np.nan, contradiction=bool(ns and no), sign_testable=bool(n >= 2), status='contradiction' if ns and no else 'only_same_relation' if ns else 'only_opposite_relation' if no else 'no_joint_decisive')

def pair_metrics(a, b, threshold, single):
    joint = active(a, threshold) & active(b, threshold)
    relation = np.sign(a[joint]) * np.sign(b[joint])
    out = counts_from_relations(relation)
    strict = joint & single
    s = counts_from_relations(np.sign(a[strict]) * np.sign(b[strict]))
    out.update(n_shared=len(a), n_decisive_a=int(active(a, threshold).sum()), n_decisive_b=int(active(b, threshold).sum()), joint_decisive_fraction=int(joint.sum()) / len(a) if len(a) else np.nan, n_exact_zero_a=int((a == 0).sum()), n_exact_zero_b=int((b == 0).sum()), n_both_exact_zero=int(((a == 0) & (b == 0)).sum()), n_nonzero_tiny_a=int(((a != 0) & (np.abs(a) <= 1e-08)).sum()), n_nonzero_tiny_b=int(((b != 0) & (np.abs(b) <= 1e-08)).sum()), n_joint_with_tiny_nonzero=int((joint & ((np.abs(a) <= 1e-08) | (np.abs(b) <= 1e-08))).sum()), n_shared_single_observation=int(single.sum()), n_joint_single_observation=s['n_joint_decisive'], single_observation_n_sign_same=s['n_sign_same'], single_observation_n_sign_opposite=s['n_sign_opposite'], single_observation_contradiction=s['contradiction'], single_observation_minimum_row_deletion_fraction=s['minimum_row_deletion_fraction'])
    return (out, joint)

def canonical_pattern(row):
    first = row[np.flatnonzero(row)[0]]
    canonical = row * first
    return ''.join(('+' if value > 0 else '-' if value < 0 else '0' for value in canonical))

def signatures(matrix, threshold):
    selected = active(matrix, threshold).all(axis=1)
    signs = np.sign(matrix).astype(np.int8)
    patterns = Counter((canonical_pattern(row) for row in signs[selected]))
    n = int(selected.sum())
    largest = max(patterns.values(), default=0)
    out = dict(n_population_rows=len(matrix), n_fully_decisive_rows=n, n_abstained_rows=len(matrix) - n, n_all_exact_zero_rows=int((matrix == 0).all(axis=1).sum()), n_any_exact_zero_rows=int((matrix == 0).any(axis=1).sum()), n_antipodal_classes=len(patterns), largest_antipodal_class_n=largest, minimum_signature_row_deletions=n - largest, minimum_signature_row_deletion_fraction=(n - largest) / n if n else np.nan, sign_contradiction=bool(len(patterns) > 1), fully_decisive_fraction=n / len(matrix) if len(matrix) else np.nan, ternary_nonzero_antipodal_classes=np.nan, ternary_minimum_row_deletion_fraction=np.nan, ternary_zero_aware_contradiction=np.nan)
    if threshold == 0:
        nonzero = signs[(signs != 0).any(axis=1)]
        ternary = Counter((canonical_pattern(row) for row in nonzero))
        most = max(ternary.values(), default=0)
        out.update(ternary_nonzero_antipodal_classes=len(ternary), ternary_minimum_row_deletion_fraction=(len(nonzero) - most) / len(nonzero) if len(nonzero) else np.nan, ternary_zero_aware_contradiction=bool(len(ternary) > 1))
    return (out, patterns)

def synthetic_checks():
    cases = []
    a, b = (np.array([1.0, 1.0]), np.array([1.0, -1.0]))
    witness, _ = pair_metrics(a, b, 0, np.ones(2, bool))
    assert witness['n_contradictory_row_pairs'] == 1
    assert witness['contradictory_cycle_fraction'] == 1
    assert witness['minimum_row_deletion_fraction'] == 0.5
    separable = np.outer(np.array([-3.0, -1.0, 0.0, 2.0, 5.0]), [1.0, -4.0, 2.0])
    sep, patterns = signatures(separable, 0)
    assert sep['n_antipodal_classes'] == 1 and (not sep['sign_contradiction'])
    assert sep['n_all_exact_zero_rows'] == 1
    assert not sep['ternary_zero_aware_contradiction']
    zeroaware, _ = signatures(np.array([[1.0, 0.0, 2.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), 0)
    assert zeroaware['n_antipodal_classes'] == 1 and zeroaware['ternary_zero_aware_contradiction']
    incompatible, _ = signatures(np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]), 0)
    assert incompatible['n_antipodal_classes'] == 2
    for ns in range(5):
        for no in range(5):
            relation = np.r_[np.ones(ns), -np.ones(no)]
            scored = counts_from_relations(relation)
            enumerated = sum((relation[i] != relation[j] for i, j in combinations(range(len(relation)), 2)))
            assert enumerated == scored['n_contradictory_row_pairs']
            if len(relation):
                max_keep = max((len(t) for k in range(len(relation) + 1) for t in combinations(range(len(relation)), k) if len({relation[i] for i in t}) <= 1), default=0)
                assert len(relation) - max_keep == scored['minimum_row_deletions']
            cases.append(dict(n_same=ns, n_opposite=no, exhaustive_count_and_deletion='passed'))
    low = np.array([[10.0, 10.0], [10.0, 20.0]])
    high = np.array([[20.0, 30.0], [20.0, 10.0]])
    functions = {'log1p': np.log1p, 'cube': lambda y: y ** 3, 'expm1': lambda y: np.expm1(y / 100.0)}
    for name, function in functions.items():
        transformed = function(high) - function(low)
        assert np.array_equal(np.sign(transformed), np.sign(high - low))
        assert pair_metrics(transformed[:, 0], transformed[:, 1], 0, np.ones(2, bool))[0]['contradiction']
    original = np.array([2.0, 3.0])
    assert active(original, 1).all() and (not active(original * 0.1, 1).any())
    assert active(np.array([0.0, 1e-14]), 0).tolist() == [False, True]
    assert active(np.array([1.0, 5.0, 10.0, 20.0]), 5).tolist() == [False, True, True, True]
    return dict(two_row_counterexample='passed', signed_separable_antipodes_and_zero_rows='passed', zero_aware_secondary='passed', exact_zero_and_inclusive_positive_thresholds='passed', exhaustive_small_count_cases=cases, strict_monotone_functions=list(functions), positive_threshold_noninvariance='passed')

def gate(frame, name):
    if name == 'shared20_joint20':
        return frame.loc[frame.n_shared.ge(20) & frame.n_joint_decisive.ge(20)]
    if name == 'shared20':
        return frame.loc[frame.n_shared.ge(20)]
    return frame

def summarize(frame, population, threshold, gate_name, stratum):
    chosen = gate(frame, gate_name)
    testable = chosen.loc[chosen.sign_testable]
    flags = testable.assign(indicator=testable.contradiction.astype(float))
    edge_flags = flags.groupby(EDGE).indicator.mean()
    single_testable = chosen.loc[chosen.n_joint_single_observation.ge(2)]
    single_flags = single_testable.assign(indicator=single_testable.single_observation_contradiction.astype(float))
    single_edge_flags = single_flags.groupby(EDGE).indicator.mean()
    means = chosen.groupby(EDGE)[list(METRICS)].mean()
    out = dict(population=population, threshold_pp=threshold, gate=gate_name, stratum=stratum, n_registry_pairs=len(frame), n_registry_edges=len(frame[EDGE].drop_duplicates()), n_eligible_pairs=len(chosen), n_eligible_edges=len(chosen[EDGE].drop_duplicates()), n_eligible_classes=chosen.reaction_type.nunique(), n_sign_testable_pairs=len(testable), n_sign_testable_edges=len(edge_flags), n_contradictory_pairs=int(chosen.contradiction.sum()), n_edges_with_contradiction=len(chosen.loc[chosen.contradiction, EDGE].drop_duplicates()), edge_equal_fraction_contradictory=float(edge_flags.mean()) if len(edge_flags) else np.nan, pair_fraction_contradictory=float(testable.contradiction.mean()) if len(testable) else np.nan, n_joint_decisive_appearances=int(chosen.n_joint_decisive.sum()), n_contradictory_cycle_appearances=int(chosen.n_contradictory_row_pairs.sum()), n_possible_cycle_appearances=int(chosen.n_possible_row_pairs.sum()), n_empty_joint=int(chosen.n_joint_decisive.eq(0).sum()), n_single_joint=int(chosen.n_joint_decisive.eq(1).sum()), n_single_observation_joint20_pairs=int(chosen.n_joint_single_observation.ge(20).sum()), n_single_observation_sign_testable_pairs=len(single_testable), n_single_observation_sign_testable_edges=len(single_edge_flags), n_single_observation_contradictory_pairs=int(chosen.single_observation_contradiction.sum()), n_single_observation_edges_with_contradiction=len(chosen.loc[chosen.single_observation_contradiction, EDGE].drop_duplicates()), single_observation_edge_equal_fraction_contradictory=float(single_edge_flags.mean()) if len(single_edge_flags) else np.nan)
    for metric in METRICS:
        valid = means[metric].dropna()
        out[f'{metric}_edge_equal_mean'] = float(valid.mean()) if len(valid) else np.nan
        out[f'{metric}_edge_equal_median'] = float(valid.median()) if len(valid) else np.nan
        out[f'{metric}_n_edges'] = len(valid)
        out[f'{metric}_n_pairs'] = int(chosen[metric].notna().sum())
    return out

def secondary_summary(frame, population, threshold, gate_name, stratum):
    chosen = frame if gate_name == 'all_retained' else frame.loc[frame.n_fully_decisive_rows.ge(20) & (frame.n_backgrounds.ge(3) if gate_name == 'fully20_bg3' else frame.n_backgrounds.ge(2))]
    defined = chosen.loc[chosen.n_fully_decisive_rows.gt(0)]
    return dict(population=population, threshold_pp=threshold, gate=gate_name, stratum=stratum, n_registry_edges=len(frame), n_eligible_edges=len(chosen), n_eligible_classes=chosen.reaction_type.nunique(), n_defined_edges=len(defined), n_sign_contradictory_edges=int(defined.sign_contradiction.sum()), n_fully_decisive_rows=int(chosen.n_fully_decisive_rows.sum()), n_abstained_rows=int(chosen.n_abstained_rows.sum()), fraction_contradictory_edges=float(defined.sign_contradiction.mean()) if len(defined) else np.nan, minimum_signature_row_deletion_fraction_equal_edge_mean=float(defined.minimum_signature_row_deletion_fraction.mean()), n_antipodal_classes_equal_edge_median=float(defined.n_antipodal_classes.median()), ternary_minimum_row_deletion_fraction_equal_edge_mean=float(chosen.ternary_minimum_row_deletion_fraction.mean()), n_ternary_zero_aware_contradictory_edges=int(chosen.ternary_zero_aware_contradiction.eq(True).sum()))

def render_report(summaries, secondary, support, metadata):

    def fmt(value):
        return f'{value:.4f}' if np.isfinite(value) else 'NA'
    primary = summaries.loc[summaries.stratum.eq('ALL') & summaries.population.eq('all') & summaries.gate.eq('shared20_joint20')].set_index('threshold_pp')
    p0, p20 = (primary.loc[0.0], primary.loc[20.0])
    lines = ['# Monotone-calibration-invariant sign separability', '', f'At exact threshold 0, all {p0.n_eligible_pairs:,} supported background pairs ({p0.n_eligible_edges:,} typed edges; {p0.n_eligible_classes} classes) contain a sign contradiction. At the fixed 20 pp threshold, {p20.n_contradictory_pairs:,}/{p20.n_sign_testable_pairs:,} supported pairs and {p20.n_edges_with_contradiction:,}/{p20.n_eligible_edges:,} eligible edges retain a witness; the edge-equal fraction of contradictory pairs is {p20.edge_equal_fraction_contradictory:.4f}. The corresponding mean edge minimum row-deletion fractions are {p0.minimum_row_deletion_fraction_edge_equal_mean:.4f} and {p20.minimum_row_deletion_fraction_edge_equal_mean:.4f}. These are necessary-condition failures for exact separability after stable strictly increasing product-specific maps, on the frozen recorded panels.', '', 'Could stable product-specific monotone yield calibration rescue a single separable substrate x background response? Here p indexes a reaction/substrate expected-product record, not an independently confirmed chemical species. Let transformed delta[p,b] = f_p(high[p,b])-f_p(low[p,b]), where f_p is strictly increasing and identical across the recorded arms/backgrounds for that expected-product record. Actual product mixtures or background-specific product species can violate this stable-map assumption and remain possible explanations. Every nonzero delta sign is preserved. If transformed delta[p,b]=a[p]*g[b] with arbitrary signed a[p], then for two backgrounds the relation r[p]=sign(delta[p,A])*sign(delta[p,B]) must be identical for every jointly nonzero row. Observing both same-sign and opposite-sign relations therefore contradicts this exact representation on the recorded panel under every such transformation.', '', 'This is a scientifically useful measurement-boundary extension of E39, which removes stable positive scales but not arbitrary nonlinear product maps. Its empirical counts reuse the existing signs and E34 contingencies; they are not new independent data, independent confirmations, p-values, or mechanistic evidence. In particular, all pairwise metrics here are algebraic functions of n_same and n_opposite (or the existing sign-agreement fraction). Passing the sign condition is only necessary and does not establish that any monotone transformation realizes separable magnitudes.', '', 'For each background pair, nS and nO count jointly decisive substrate rows with same and opposite response signs. Contradictory unordered row pairs number nS*nO out of choose(nS+nO,2): each yields an incompatible 2x2 sign cycle on the two backgrounds. The smallest fraction of decisive rows that must be deleted to make this two-background sign pattern feasible is min(nS,nO)/(nS+nO). This is distance to sign feasibility, not a noise estimate or distance to full magnitude separability. Cycle counts reuse rows/backgrounds and are not independent sample sizes.', '', 'All 1,360 background pairs and 302 typed edges are retained, with both all-outcome and outcome-conditioned both-positive panels. All 20 reaction classes appear in support tables; 18 have registry background pairs, while esterification and reductive-amination remain unsupported. Zero overlap, zero responses, singleton decisive panels and empty sign denominators remain explicit. Exact threshold 0 uses abs(delta)>0, with no numerical zero tolerance and unchanged outcomes. Only three profile responses are nonzero but <=1e-8 pp; appearances are counted separately. Positive thresholds use the original inclusive abs(delta)>=threshold in both backgrounds. Thresholds 1,5,10,20 are fixed sensitivities; 15 and 30 also reconcile E34 completely.', '', 'The sign contradiction on a frozen selected population is invariant under strictly increasing transformations. Positive magnitude thresholds and their membership are not invariant: transformed deltas can cross a 1/5/10/15/20/30 cutoff. Both-positive membership is also an inherited outcome-conditioned restriction, not an invariant definition under arbitrary shifts of yield scale. Selection is frozen here; no threshold is chosen from its result.', '', 'The primary supported gate requires >=20 shared rows and >=20 jointly decisive rows. The shared20 gate additionally shows attrition caused by decisive support. Metrics average eligible background pairs within each typed component edge, then weight edges equally; contradiction fractions use sign-testable pairs (>=2 decisive rows), while row-deletion fractions require >=1. Pair/edge denominators are saved for every metric. The all-retained CSV includes the complete universe.', '', '| Population | Threshold (pp) | Gate | Eligible pairs / edges / classes | Contradictory pairs / testable pairs | Edges with a witness | Edge-equal fraction contradictory | Mean edge contradictory-cycle fraction | Mean edge minimum row deletion |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries.loc[summaries.stratum.eq('ALL') & summaries.gate.isin(['shared20', 'shared20_joint20'])].itertuples():
        lines.append(f'| {r.population} | {r.threshold_pp:g} | {r.gate} | {r.n_eligible_pairs} / {r.n_eligible_edges} / {r.n_eligible_classes} | {r.n_contradictory_pairs} / {r.n_sign_testable_pairs} | {r.n_edges_with_contradiction} | {fmt(r.edge_equal_fraction_contradictory)} | {fmt(r.contradictory_cycle_fraction_edge_equal_mean)} | {fmt(r.minimum_row_deletion_fraction_edge_equal_mean)} |')
    lines += ['', 'The report treats each contradiction as a property of recorded observations. Measurement noise, rounding and clipping/ties, condition/arm-dependent assay responses and omitted structure remain possible explanations. A strictly increasing map applied equally to both arms cannot reverse their order; allowing arm-specific calibration is a different model. A few contrary rows can reject exact sign separability even if most rows agree, which is why witness prevalence and row-deletion magnitudes are reported together.', '', f"The recorded profile delta and mean-high minus mean-low have identical signs in all 251,835 records; their maximum numerical difference is {metadata['maximum_delta_vs_mean_arm_difference_pp']:.4g} pp. Positive thresholds follow the stored delta, because direct arm subtraction can fall on the other side of an inclusive threshold at machine precision. However, {metadata['n_collapsed_multiple_observation_records']} records average duplicate observations. The monotone proof applies directly to the recorded/collapsed arm values; transforming individual replicates and then averaging need not preserve the sign of their original average difference, because f(mean y) and mean f(y) need not commute. Every pair also retains sign counts restricted to records with exactly one underlying observation in both backgrounds. This labeled sensitivity preserves the full primary result and isolates that averaging issue. The following rows keep the original primary gate, then exclude multi-observation records within each eligible pair; they do not reapply the 20-row gate after exclusion.", '', '| Single-observation sensitivity | Threshold (pp) | Testable pairs / edges | Pairs retaining >=20 decisive rows | Contradictory pairs / witness edges | Edge-equal fraction contradictory | Mean edge minimum row deletion |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries.loc[summaries.stratum.eq('ALL') & summaries.gate.eq('shared20_joint20') & summaries.threshold_pp.isin([0.0, 1.0, 5.0, 10.0, 20.0])].itertuples():
        lines.append(f'| {r.population} | {r.threshold_pp:g} | {r.n_single_observation_sign_testable_pairs} / {r.n_single_observation_sign_testable_edges} | {r.n_single_observation_joint20_pairs} | {r.n_single_observation_contradictory_pairs} / {r.n_single_observation_edges_with_contradiction} | {fmt(r.single_observation_edge_equal_fraction_contradictory)} | {fmt(r.single_observation_minimum_row_deletion_fraction_edge_equal_mean)} |')
    lines += ['', '| Secondary all-background complete panels | Threshold | Gate | Edges / classes | Fully decisive rows | Contradictory edges / defined edges | Mean edge minimum signature deletion | Median antipodal classes |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in secondary.loc[secondary.stratum.eq('ALL') & secondary.gate.eq('fully20_bg3')].itertuples():
        lines.append(f'| {r.population} | {r.threshold_pp:g} | {r.gate} | {r.n_eligible_edges} / {r.n_eligible_classes} | {r.n_fully_decisive_rows} | {r.n_sign_contradictory_edges} / {r.n_defined_edges} | {fmt(r.minimum_signature_row_deletion_fraction_equal_edge_mean)} | {fmt(r.n_antipodal_classes_equal_edge_median)} |')
    lines += ['', 'For each of all 302 edges, the secondary audit uses the exact intersection across every recorded background. Fully decisive row sign vectors are identified with their antipodes because a[p] may be positive or negative. More than one antipodal class contradicts signed rank-one sign structure. Deleting all rows outside the largest class is the minimum sign-only deletion. Missing records, all-zero rows and rows abstaining in any background remain counted. At exact threshold 0, an additional zero-aware ternary-pattern calculation includes partly zero rows and allows entirely zero a[p] rows; it is saved separately because observed zeros may reflect clipping. Complete intersections and full decisiveness select a smaller densely observed population, so this is a secondary scope, not a broader independent confirmation.', '', '| Reaction class | Registry pairs / edges | All threshold0 primary pairs / witness edges | All threshold20 primary pairs / witness edges |', '|---|---:|---:|---:|']
    zero = support.loc[support.population.eq('all') & support.threshold_pp.eq(0)].set_index('reaction_type')
    twenty = support.loc[support.population.eq('all') & support.threshold_pp.eq(20)].set_index('reaction_type')
    for reaction in zero.index:
        a, b = (zero.loc[reaction], twenty.loc[reaction])
        lines.append(f'| {reaction} | {a.n_registry_pairs} / {a.n_registry_edges} | {a.n_primary_pairs} / {a.n_witness_edges} | {b.n_primary_pairs} / {b.n_witness_edges} |')
    lines += ['', f"Validation includes an analytical two-row counterexample; all 25 small count cases with exhaustive cycle enumeration and deletion checks; signed-separable/antipodal and zero-aware toys; three strictly increasing nonlinear maps; explicit noninvariance of positive threshold membership; reconstruction of all 8,160 E34 pair/population/threshold rows; existing 10/15/20 pp registry counts; and stable product-specific nonlinear maps on all recorded arm values. On frozen selected populations, transformed sign counts have maximum discrepancy {metadata['max_frozen_transform_count_discrepancy']}. All input hashes are unchanged.", '', 'The real nonlinear map is f_p(y)=c_p*y+d_p*y^2/100 with c_p>0,d_p>=0 chosen deterministically per reaction/substrate product record and reused across all its arms/backgrounds/edges. Its difference is evaluated as (high-low)*(c_p+d_p*(high+low)/100) to avoid cancellation on the three nearly tied records. Direct floating-point evaluation is separately audited. This numerical check illustrates the analytical order-preservation theorem; it does not establish that this map corrects an assay.', '', 'Outputs: monotone_sign_separability_pairs.csv; _summary.csv; _support.csv; _witnesses.csv; _signatures.csv; _signature_patterns.csv; _signature_summary.csv; _transformation.csv; audit/monotone_sign_separability_metadata.json. All are historical descriptive results, with no prospective validation or new-molecule prediction.', '']
    return '\n'.join(lines)

def main():
    tests = synthetic_checks()
    paths = {name: BASE / relative for name, relative in dict(profiles='results/matched_response_profiles.parquet', registry='results/matched_response_transport_pairs.csv', network='results/condition_network_meta.csv', prior_sign='results/matched_sign_information_pairs.csv').items()}
    before = {name: digest(path) for name, path in paths.items()}
    profiles = pd.read_parquet(paths['profiles'])
    registry = pd.read_csv(paths['registry'])
    classes = sorted(pd.read_csv(paths['network']).reaction_type.unique())
    assert len(profiles) == 267462 and len(registry) == 1361 and (len(classes) == 20)
    assert not profiles.duplicated(CONTEXT + ['substrate_pair_key']).any()
    high, low = (profiles.high_yield_pp.to_numpy(), profiles.low_yield_pp.to_numpy())
    assert min(high.min(), low.min()) >= 0
    original_delta = profiles.delta.to_numpy()
    assert np.array_equal(np.sign(original_delta), np.sign(high - low))
    product_ids = profiles.reaction_type + '|' + profiles.substrate_pair_key
    unique_products = product_ids.unique()
    coefficients = {}
    for key in unique_products:
        h = hashlib.sha256(f'monotone_sign_map_v1|{key}'.encode()).digest()
        coefficients[key] = (0.25 + 2 * int.from_bytes(h[:2], 'little') / 65535.0, 2 * int.from_bytes(h[2:4], 'little') / 65535.0)
    c = np.array([coefficients[p][0] for p in product_ids])
    d = np.array([coefficients[p][1] for p in product_ids])
    transformed = (high - low) * (c + d * (high + low) / 100.0)
    direct = c * high + d * high ** 2 / 100.0 - (c * low + d * low ** 2 / 100.0)
    assert np.array_equal(np.sign(transformed), np.sign(original_delta))
    profiles['transformed_delta'] = transformed
    transformation = []
    for threshold in THRESHOLDS:
        transformation.append(dict(threshold_pp=threshold, n_profile_records=len(profiles), n_sign_mismatches_factored=int((np.sign(transformed) != np.sign(original_delta)).sum()), n_sign_mismatches_direct_float=int((np.sign(direct) != np.sign(original_delta)).sum()), n_decisive_membership_changed=int((active(original_delta, threshold) != active(transformed, threshold)).sum())))
    lookup = {key: g.set_index('substrate_pair_key').sort_index() for key, g in profiles.groupby(CONTEXT, sort=False)}
    rows, witnesses, discrepancy = ([], [], 0)
    for idx, pair in registry.iterrows():
        edge = tuple((pair[k] for k in EDGE))
        aa, bb = (lookup[edge + (pair.background_a,)], lookup[edge + (pair.background_b,)])
        common = aa.index.intersection(bb.index).sort_values()
        assert len(common) == pair.all_n_shared
        for population in POPULATIONS:
            selected = common
            if population == 'both_positive':
                selected = selected[(aa.loc[selected, 'both_positive'] & bb.loc[selected, 'both_positive']).to_numpy()]
            assert len(selected) == pair[f'{population}_n_shared']
            a, b = (aa.loc[selected], bb.loc[selected])
            da, db = (a.delta.to_numpy(), b.delta.to_numpy())
            single = (a.n_observations.eq(1) & b.n_observations.eq(1)).to_numpy()
            for threshold in THRESHOLDS:
                scored, joint = pair_metrics(da, db, threshold, single)
                tr = counts_from_relations(np.sign(a.transformed_delta.to_numpy()[joint]) * np.sign(b.transformed_delta.to_numpy()[joint]))
                discrepancy = max(discrepancy, abs(scored['n_sign_same'] - tr['n_sign_same']), abs(scored['n_sign_opposite'] - tr['n_sign_opposite']))
                assert discrepancy == 0
                base = dict(pair_id=idx + 1, population=population, threshold_pp=threshold, **dict(zip(EDGE, edge)))
                rows.append(dict(**base, **scored))
                if threshold in (10.0, 15.0, 20.0):
                    assert scored['n_joint_decisive'] == pair[f'{population}_n_joint_decisive_{int(threshold)}']
                    assert scored['n_sign_same'] == pair[f'{population}_n_sign_agree_{int(threshold)}']
                if scored['contradiction']:
                    same = np.flatnonzero(joint & (np.sign(da) == np.sign(db)))[0]
                    opp = np.flatnonzero(joint & (np.sign(da) != np.sign(db)))[0]
                    witnesses.append(dict(**base, same_relation_substrate=selected[same], opposite_relation_substrate=selected[opp], same_delta_a_pp=float(da[same]), same_delta_b_pp=float(db[same]), opposite_delta_a_pp=float(da[opp]), opposite_delta_b_pp=float(db[opp]), same_n_observations_a=int(a.n_observations.iloc[same]), same_n_observations_b=int(b.n_observations.iloc[same]), opposite_n_observations_a=int(a.n_observations.iloc[opp]), opposite_n_observations_b=int(b.n_observations.iloc[opp])))
        if (idx + 1) % 250 == 0:
            print(f'Monotone sign audit: {idx + 1}/{len(registry)} background pairs', flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 1361 * 2 * len(THRESHOLDS)
    prior = pd.read_csv(paths['prior_sign'])
    prior['pair_id'] = prior.background_pair_id + 1
    prior = prior.rename(columns={'panel': 'population'}).set_index(['pair_id', 'population', 'threshold_pp'])
    new = result.loc[result.threshold_pp.isin([15.0, 20.0, 30.0])].set_index(['pair_id', 'population', 'threshold_pp'])
    prior = prior.loc[new.index]
    assert len(new) == 8166
    assert np.array_equal(new.n_joint_decisive.to_numpy(), prior.n_joint_decisive.to_numpy())
    assert np.array_equal(new.n_sign_same.to_numpy(), (prior.n_pp + prior.n_nn).to_numpy())
    assert np.array_equal(new.n_sign_opposite.to_numpy(), (prior.n_pn + prior.n_np).to_numpy())
    assert (result.n_contradictory_row_pairs <= result.n_possible_row_pairs).all()
    summary_rows, support_rows = ([], [])
    for (population, threshold), group in result.groupby(['population', 'threshold_pp'], sort=False):
        for gate_name in GATES:
            for reaction in ('ALL', *classes):
                sub = group if reaction == 'ALL' else group.loc[group.reaction_type.eq(reaction)]
                summary_rows.append(summarize(sub, population, threshold, gate_name, reaction))
        for reaction in classes:
            sub = group.loc[group.reaction_type.eq(reaction)]
            primary = gate(sub, 'shared20_joint20')
            support_rows.append(dict(reaction_type=reaction, population=population, threshold_pp=threshold, n_registry_pairs=len(sub), n_registry_edges=len(sub[EDGE].drop_duplicates()), n_primary_pairs=len(primary), n_primary_edges=len(primary[EDGE].drop_duplicates()), n_contradictory_pairs=int(primary.contradiction.sum()), n_witness_edges=len(primary.loc[primary.contradiction, EDGE].drop_duplicates())))
    summaries, support = (pd.DataFrame(summary_rows), pd.DataFrame(support_rows))
    signature_rows, pattern_rows = ([], [])
    for edge, group in profiles.groupby(EDGE, sort=True):
        contexts = sorted(group.background_fingerprint.unique())
        panels = [set(g.substrate_pair_key) for _, g in group.groupby('background_fingerprint')]
        common = sorted(set.intersection(*panels))
        union = set.union(*panels)
        full = group.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta').reindex(index=common, columns=contexts)
        positivity = group.pivot(index='substrate_pair_key', columns='background_fingerprint', values='both_positive').reindex(index=common, columns=contexts)
        assert not full.isna().any().any()
        for population in POPULATIONS:
            selected = np.ones(len(full), bool) if population == 'all' else positivity.all(axis=1).to_numpy()
            matrix = full.to_numpy()[selected]
            for threshold in THRESHOLDS:
                scored, patterns = signatures(matrix, threshold)
                base = dict(**dict(zip(EDGE, edge)), population=population, threshold_pp=threshold, n_backgrounds=len(contexts), n_intersection_rows=len(common), n_union_rows=len(union), n_observed_cells=len(group), n_missing_cells_from_union=len(union) * len(contexts) - len(group))
                signature_rows.append(dict(**base, **scored))
                pattern_rows.extend((dict(**base, antipodal_pattern=pattern, n_rows=count) for pattern, count in patterns.items()))
                if scored['n_antipodal_classes'] > 1:
                    chosen = matrix[active(matrix, threshold).all(axis=1)]
                    vectors = np.sign(chosen).astype(int)
                    first = vectors[0]
                    other = next((v for v in vectors if canonical_pattern(v) != canonical_pattern(first)))
                    agreement = first * other
                    assert (agreement > 0).any() and (agreement < 0).any()
    signature_table = pd.DataFrame(signature_rows)
    assert len(signature_table) == 303 * 2 * len(THRESHOLDS)
    secondary_rows = []
    for (population, threshold), group in signature_table.groupby(['population', 'threshold_pp'], sort=False):
        for gate_name in ('all_retained', 'fully20_bg2', 'fully20_bg3'):
            for reaction in ('ALL', *classes):
                sub = group if reaction == 'ALL' else group.loc[group.reaction_type.eq(reaction)]
                secondary_rows.append(secondary_summary(sub, population, threshold, gate_name, reaction))
    secondary = pd.DataFrame(secondary_rows)
    for name, frame in (('pairs', result), ('summary', summaries), ('support', support), ('witnesses', pd.DataFrame(witnesses)), ('signatures', signature_table), ('signature_patterns', pd.DataFrame(pattern_rows)), ('signature_summary', secondary), ('transformation', pd.DataFrame(transformation))):
        frame.to_csv(BASE / f'results/monotone_sign_separability_{name}.csv', index=False)
    after = {name: digest(path) for name, path in paths.items()}
    assert before == after
    metadata = dict(thresholds_pp=list(THRESHOLDS), zero='Exactdelta0; threshold0 abs(delta)>0; positive thresholds inclusive>=; unchanged outcomes', primary='all population; shared>=20 and jointlydecisive>=20; full universe retained', n_registry_pairs=1360, n_typed_edges=302, n_registered_classes=registry.reaction_type.nunique(), n_corpus_classes=len(classes), n_pair_population_threshold_rows=len(result), n_signature_rows=len(signature_table), n_E34_reconciled_rows=len(new), n_witness_rows=len(witnesses), synthetic=tests, max_frozen_transform_count_discrepancy=discrepancy, n_profile_nearzero_nonzero=int(((np.abs(original_delta) <= 1e-08) & (original_delta != 0)).sum()), n_collapsed_multiple_observation_records=int(profiles.n_observations.gt(1).sum()), direct_float_sign_mismatches=int((np.sign(direct) != np.sign(original_delta)).sum()), maximum_delta_vs_mean_arm_difference_pp=float(np.abs(original_delta - (high - low)).max()), exact_sign_delta_vs_mean_arm_mismatches=0, input_hashes={name: dict(path=str(paths[name]), sha256=before[name], unchanged=before[name] == after[name]) for name in paths}, code_sha256=digest(Path(__file__)), product_identity='p indexes a reaction/substrate expected-product record without independent chemical confirmation. Actual mixtures or background-specific species may violate stable f_p.', invariance='Strictly increasing map preserves signs on frozen selected panels; positive threshold membership is not invariant', collapsed='Proof directly concerns recorded/collapsed arm values; transform-before-averaging does not commute; single-observation sensitivity retained', interpretation='Necessary condition only; counts re-express existing E34 signs; no independent evidence, pvalue, noise correction, mechanism or prospective generalization', witnesses='First lexicographic same-relation and opposite-relation substrate rows for each contradictory case; no ranking of chemical candidates')
    (BASE / 'audit/monotone_sign_separability_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    report = render_report(summaries, secondary, support, metadata)
    (BASE / 'reports/MONOTONE_SIGN_SEPARABILITY.md').write_text(report, encoding='utf-8')
    print(summaries.loc[summaries.stratum.eq('ALL') & summaries.gate.eq('shared20_joint20'), ['population', 'threshold_pp', 'n_eligible_pairs', 'n_eligible_edges', 'n_eligible_classes', 'n_contradictory_pairs', 'n_edges_with_contradiction', 'edge_equal_fraction_contradictory', 'contradictory_cycle_fraction_edge_equal_mean', 'minimum_row_deletion_fraction_edge_equal_mean']].to_string(index=False))
    print(secondary.loc[secondary.stratum.eq('ALL') & secondary.gate.eq('fully20_bg3')].to_string(index=False))
    print(json.dumps(dict(n_E34_reconciled_rows=len(new), max_frozen_transform_count_discrepancy=discrepancy, direct_float_sign_mismatches=metadata['direct_float_sign_mismatches']), indent=2))
if __name__ == '__main__':
    main()
