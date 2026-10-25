"""Sparse sign-constraint feasibility for all recorded context profiles.

Missing and threshold-abstaining cells impose no constraint. This is exact sign
fitting on observed support, not magnitude rank-one fitting or heldout prediction."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
BASE = Path(__file__).resolve().parents[1]
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
THRESHOLDS = (0.0, 1.0, 5.0, 10.0, 20.0)
CHUNK = 1024
CHECKS = []

def check(name, passed, detail=''):
    CHECKS.append(dict(check=name, passed=bool(passed), detail=str(detail)))
    if not passed:
        raise AssertionError(f'{name}: {detail}')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def candidates(start, stop, n_backgrounds):
    case = np.arange(start, stop, dtype=np.uint32)
    remaining = 1 - 2 * (case[:, None] >> np.arange(n_backgrounds - 2, -1, -1, dtype=np.int32) & 1).astype(np.int8)
    return np.column_stack([np.ones(len(case), dtype=np.int8), remaining])

def solve_component(rr, bb, sign, mass, n_rows, n_backgrounds):
    assert n_backgrounds <= 18
    count_matrix = np.zeros((n_rows, n_backgrounds))
    weighted_matrix = np.zeros_like(count_matrix)
    count_matrix[rr, bb] = sign
    weighted_matrix[rr, bb] = sign * mass
    n_cases = 1 << n_backgrounds - 1
    count_scores = np.empty(n_cases)
    weight_scores = np.empty(n_cases)
    interacting = np.count_nonzero(count_matrix, axis=1) >= 2
    for start in range(0, n_cases, CHUNK):
        stop = min(n_cases, start + CHUNK)
        cs = candidates(start, stop, n_backgrounds)
        count_scores[start:stop] = np.abs(count_matrix[interacting] @ cs.T).sum(axis=0)
        weight_scores[start:stop] = np.abs(weighted_matrix[interacting] @ cs.T).sum(axis=0)
    first_count = int(np.argmax(count_scores))
    count_c = candidates(first_count, first_count + 1, n_backgrounds)[0]
    count_r = np.where(count_matrix @ count_c >= 0, 1, -1).astype(np.int8)
    total_mass = math.fsum(mass)
    error = 64 * np.finfo(float).eps * (n_rows + n_backgrounds + 1) * total_mass
    near = np.flatnonzero(weight_scores >= weight_scores.max() - 2 * error)
    ratios = [float(w).as_integer_ratio() for w in mass]
    power = max((den.bit_length() - 1 for _, den in ratios), default=0)
    exact = [num << power - (den.bit_length() - 1) for num, den in ratios]
    best_cost = None
    best_case = None
    best_r = None
    best_c = None
    for case in near:
        cs = candidates(int(case), int(case) + 1, n_backgrounds)[0]
        balances = [0] * n_rows
        for row, bg, s, w in zip(rr, bb, sign, exact):
            balances[int(row)] += int(s) * int(cs[bg]) * w
        rs = np.array([1 if value >= 0 else -1 for value in balances], dtype=np.int8)
        wrong = rs[rr] * cs[bb] != sign
        cost = sum((w for w, bad in zip(exact, wrong) if bad))
        if best_cost is None or cost < best_cost:
            best_cost, best_case, best_r, best_c = (cost, int(case), rs, cs)
    weighted_wrong = best_r[rr] * best_c[bb] != sign
    exact_float = math.ldexp(float(best_cost), -power)
    fsum_cost = math.fsum(mass[weighted_wrong])
    assert abs(exact_float - fsum_cost) <= 8 * np.finfo(float).eps * max(1, total_mass)
    return dict(count_row=count_r, count_bg=count_c, mass_row=best_r, mass_bg=best_c, enumeration_cases=n_cases, weighted_precision_candidate_cases=len(near), float_screen_error_bound_pp=error, weighted_exact_cost_scaled_integer=str(best_cost), weighted_exact_denominator_power=power, first_count_case=first_count, first_mass_case=best_case)

def solve_graph(row_index, background_index, sign, mass, n_rows, n_backgrounds):
    rr = np.asarray(row_index, dtype=int)
    bb = np.asarray(background_index, dtype=int)
    signs = np.asarray(sign, dtype=np.int8)
    weights = np.asarray(mass, dtype=float)
    assert len(rr) == len(bb) == len(signs) == len(weights)
    assert np.isin(signs, [-1, 1]).all() and np.all(weights > 0)
    assert len(set(zip(rr, bb))) == len(rr)
    degrees = np.bincount(rr, minlength=n_rows)
    bgdegrees = np.bincount(bb, minlength=n_backgrounds)
    adjacency = coo_matrix((np.ones(2 * len(rr)), (np.r_[rr, n_rows + bb], np.r_[n_rows + bb, rr])), shape=(n_rows + n_backgrounds, n_rows + n_backgrounds)).tocsr()
    all_components, labels = connected_components(adjacency, directed=False)
    active_labels = sorted(set(labels[rr]), key=lambda label: int(np.flatnonzero(labels[n_rows:] == label)[0]))
    solutions = {name: dict(row_signs=np.ones(n_rows, dtype=np.int8), background_signs=np.ones(n_backgrounds, dtype=np.int8)) for name in ('count', 'mass')}
    components = []
    for index, label in enumerate(active_labels, 1):
        rows = np.flatnonzero((labels[:n_rows] == label) & (degrees > 0))
        bgs = np.flatnonzero((labels[n_rows:] == label) & (bgdegrees > 0))
        selected = labels[rr] == label
        local_r = np.searchsorted(rows, rr[selected])
        local_b = np.searchsorted(bgs, bb[selected])
        result = solve_component(local_r, local_b, signs[selected], weights[selected], len(rows), len(bgs))
        for name in ('count', 'mass'):
            solutions[name]['row_signs'][rows] = result[name + '_row']
            solutions[name]['background_signs'][bgs] = result[name + '_bg']
        components.append(dict(component=index, n_rows=len(rows), n_backgrounds=len(bgs), n_cells=int(selected.sum()), cycle_rank=int(selected.sum()) - len(rows) - len(bgs) + 1, **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)}))
    for name, result in solutions.items():
        wrong = result['row_signs'][rr] * result['background_signs'][bb] != signs
        result.update(mismatched=wrong, n_mismatched=int(wrong.sum()), mismatched_mass_pp=math.fsum(weights[wrong]))
    active_vertices = int(np.count_nonzero(degrees) + np.count_nonzero(bgdegrees))
    cycle_rank = len(rr) - active_vertices + len(active_labels)
    topology = dict(n_active_cells=len(rr), n_active_rows=int(np.count_nonzero(degrees)), n_active_backgrounds=int(np.count_nonzero(bgdegrees)), n_active_vertices=active_vertices, n_isolated_rows=int((degrees == 0).sum()), n_isolated_backgrounds=int((bgdegrees == 0).sum()), n_active_components=len(active_labels), n_components_including_isolates=all_components, cycle_rank=cycle_rank, n_cyclic_components=sum((c['cycle_rank'] > 0 for c in components)), n_cells_in_cyclic_components=sum((c['n_cells'] for c in components if c['cycle_rank'] > 0)), n_degree1_rows=int((degrees == 1).sum()), n_cells_on_degree1_rows=int((degrees[rr] == 1).sum()), n_degree_ge2_rows=int((degrees >= 2).sum()), n_cells_on_degree_ge2_rows=int((degrees[rr] >= 2).sum()), enumeration_cases=sum((c['enumeration_cases'] for c in components)), weighted_precision_candidate_cases=sum((c['weighted_precision_candidate_cases'] for c in components)))
    assert solutions['count']['n_mismatched'] <= solutions['mass']['n_mismatched']
    assert solutions['mass']['mismatched_mass_pp'] <= solutions['count']['mismatched_mass_pp'] + 1e-10
    assert cycle_rank >= 0 and (cycle_rank > 0 or solutions['count']['n_mismatched'] == 0)
    compmap = {label: index for index, label in enumerate(active_labels, 1)}
    return dict(**solutions, topology=topology, components=components, row_components=np.array([compmap.get(x, 0) for x in labels[:n_rows]], dtype=np.int32), background_components=np.array([compmap.get(x, 0) for x in labels[n_rows:]], dtype=np.int32))

class ParquetSink:

    def __init__(self, path):
        self.path = path
        self.writer = None
        self.n_rows = 0

    def append(self, data):
        frame = pd.DataFrame(data)
        if not len(frame):
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='zstd')
        self.writer.write_table(table)
        self.n_rows += len(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()

def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan

def summarize(solutions, classes):
    metrics = ['mismatched_cell_fraction_decisive', 'mismatched_cell_fraction_panel', 'mismatched_cell_fraction_population', 'mismatched_mass_fraction_decisive', 'mismatched_mass_fraction_panel', 'mismatched_mass_pp_per_decisive_cell', 'mismatched_mass_pp_per_panel_cell', 'mismatched_mass_pp_per_population_cell']
    result = []
    for keys, allrows in solutions.groupby(['population', 'panel', 'threshold_pp', 'objective'], sort=True):
        for label in ['ALL', *classes]:
            x = allrows if label == 'ALL' else allrows[allrows.reaction_type == label]
            valid = x[x.n_decisive_cells > 0]
            cyclic = x[x.cycle_rank > 0]
            record = dict(zip(['population', 'panel', 'threshold_pp', 'objective'], keys), stratum=label, n_retained_edges=len(x), n_observed_edges=int(x.n_panel_observed_cells.gt(0).sum()), n_decisive_edges=len(valid), n_decisive_classes=valid.reaction_type.nunique(), n_cycle_edges=len(cyclic), n_conflicting_edges=int(x.n_mismatched.gt(0).sum()), conflict_fraction_decisive_edges=ratio(int(x.n_mismatched.gt(0).sum()), len(valid)), conflict_fraction_cycle_edges=ratio(int(x.n_mismatched.gt(0).sum()), len(cyclic)), n_population_cell_appearances=int(x.n_population_observed_cells.sum()), n_panel_cell_appearances=int(x.n_panel_observed_cells.sum()), n_decisive_cell_appearances=int(x.n_decisive_cells.sum()), n_mismatched_cell_appearances=int(x.n_mismatched.sum()), mismatched_mass_pp_sum=float(x.mismatched_mass_pp.sum()), pooled_mismatched_cell_fraction_decisive=ratio(x.n_mismatched.sum(), x.n_decisive_cells.sum()), pooled_mismatched_mass_fraction_decisive=ratio(x.mismatched_mass_pp.sum(), x.decisive_abs_mass_pp.sum()))
            for metric in metrics:
                record[metric + '_equal_edge_mean'] = valid[metric].mean()
                record[metric + '_equal_edge_median'] = valid[metric].median()
            result.append(record)
    summary = pd.DataFrame(result)
    equal = []
    for keys, g in summary[summary.stratum.ne('ALL')].groupby(['population', 'panel', 'threshold_pp', 'objective'], sort=True):
        x = g[g.n_decisive_edges > 0]
        record = dict(zip(['population', 'panel', 'threshold_pp', 'objective'], keys), n_classes_retained=len(g), n_classes_decisive=len(x))
        for metric in metrics:
            record[metric + '_equal_class_mean'] = x[metric + '_equal_edge_mean'].mean()
        equal.append(record)
    return (summary, pd.DataFrame(equal))

def markdown(frame, columns):

    def value(x):
        return 'NA' if pd.isna(x) else f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(('---' for _ in columns)) + ' |'] + ['| ' + ' | '.join((value(x) for x in r)) + ' |' for r in frame[columns].itertuples(index=False, name=None)])

def report(summary, equal, sensitivity, metadata):
    show = summary[summary.stratum.eq('ALL') & summary.population.eq('all')]
    columns = ['panel', 'threshold_pp', 'objective', 'n_retained_edges', 'n_decisive_edges', 'n_decisive_classes', 'n_cycle_edges', 'n_conflicting_edges', 'mismatched_cell_fraction_decisive_equal_edge_mean', 'mismatched_mass_fraction_decisive_equal_edge_mean', 'mismatched_mass_pp_per_panel_cell_equal_edge_mean']
    classes = summary[summary.stratum.ne('ALL') & summary.population.eq('all') & summary.panel.eq('observed_sparse') & summary.threshold_pp.eq(0) & summary.objective.eq('count')]
    return '\n'.join(['# Sparse Sign Feasibility Across Recorded Backgrounds', '', 'This extension broadens E43 from complete intersections to every observed substrate-background cell in all302 repeated typed edges. A decisive recorded response imposes sign(delta[i,b])=r[i]*c[b], with unrestricted signed row and background factors. Missing, exact-zero and threshold-abstaining cells impose no constraint; they are never imputed as zero response constraints. This is a global descriptive fit, not a heldout predictive model.', '', '## Population, Panels and Exact Optimization', '', 'The all population retains all recorded cells. The inherited both_positive population retains a cell only when both of its recorded arm yields are positive; this is outcome-conditioned. observed_sparse uses their union graph. complete_population_rows first requires the row to have a record in every original background within that population, then applies the magnitude threshold. Backgrounds absent after filtering remain explicit isolates. All20 reaction classes remain in summaries; the302 repeated edges cover18 classes and unsupported classes retain empty support.', '', 'Threshold0 uses exact abs(delta)>0 with no zero tolerance. The fixed1/5/10/20pp sensitivities use inclusive abs(delta)>=threshold on original stored deltas. Components are recomputed after every population, panel and threshold mask. Each active bipartite component fixes the first lexicographic background sign to+1 and enumerates all2^(B-1) remaining background assignments. Conditional row signs minimize the objective exactly. Background ties use lexicographic+1 before-1; conditional row ties use+1. Isolated nodes use+1 without affecting loss. The largest original background set is18, so this enumeration is finite and exact; NumPy case chunks are1024 and BLAS threads unrestricted (library defaults).', '', 'Two objectives are solved separately: count weights1 and weights abs(original delta). The count optimum can have a larger mismatched response mass than the mass optimum; the mass optimum can mismatch more cells. Both residual counts and masses are saved for both assignments. Weighted enumeration first uses a conservative floating error screen, then rescales every recorded binary float to a common integer denominator and compares all possible close optima and row majorities exactly. Reported pp sums are checked against correctly rounded sums of the chosen residual weights.', '', '## All-Yield Results', '', 'Fractions below use decisive-cell or decisive-mass denominators, as labeled. Saved tables also expose full panel/population-cell denominators, including zero and abstaining observations, and pp per decisive, panel and population cell. Empty decisive support gives undefined fractions, not a zero-effect claim. Summary means weight supported typed edges equally; all retained/no-cycle cases remain visible. n_cycle_edges counts typed edges whose graphs have positive cycle rank, not individual cycle cells.', '', markdown(show, columns), '', '## Equal-Class Results', '', 'These means first average edges inside each supported class, then average supported classes equally. Missing classes are not assigned a zero effect.', '', markdown(equal[equal.population.eq('all')], ['panel', 'threshold_pp', 'objective', 'n_classes_retained', 'n_classes_decisive', 'mismatched_cell_fraction_decisive_equal_class_mean', 'mismatched_mass_fraction_decisive_equal_class_mean']), '', '## Complete Class Accounting', '', markdown(classes, ['stratum', 'n_retained_edges', 'n_observed_edges', 'n_decisive_edges', 'n_cycle_edges', 'n_conflicting_edges', 'mismatched_cell_fraction_decisive_equal_edge_mean']), '', '## What This Distance Means', '', 'Deleting mismatched constraints leaves a sign-factorable observed graph. It does not mean replacing their responses by zero: arbitrary zeros can violate rank-one zero patterns. The weighted objective is distance only to sign-compatible orthants with arbitrary per-cell magnitudes. It is not distance to the numerical rank-one magnitude cone, a realizable product-specific monotone map, or an assay/noise correction. Sign feasibility is necessary, not sufficient, for those stronger models. A zero residual on a tree or sparse graph can be uninformative; cycle rank E-V+C, components, degree1 rows/cells, and degree>=2 support describe topology, not independent empirical confirmation or numbers of independent contradictions.', '', 'On frozen support, strictly increasing maps applied identically to both arms preserve response signs and the count objective. Positive magnitude thresholds, both-positive membership under arbitrary shifts, weighted pp losses and mass fractions are not invariant. The panel/population is frozen here. Shared observations, nested thresholds and backgrounds make the summaries dependent. This extension reuses E43/profile signs; it adds coverage and a different necessary-constraint calculation, not new independent observations, p-values, mechanisms or prospective validation.', '', '## E43 and Single-Observation Checks', '', "The E43 overlap table reconstructs exactly its fully decisive complete rows at each matching threshold and verifies their antipodal pattern counts and contradiction flags. E47 complete_population_rows is a different mask because it also retains rows with only some decisive backgrounds. E43's zero-aware patterns additionally treat observed zeros as constraints; E47 does not. These panels are never silently equated. A contradiction within the identical E43 fully decisive subset must also occur in the containing E47 complete graph.", '', f"There are28 collapsed duplicate profile records, including8 nonzero responses. The single-observation sensitivity drops such cells before population-complete row selection and thresholds, and recomputes every affected graph. Unaffected edge strata reference the identical original assignment. The sensitivity table retains all original edge/population/panel/threshold/objective combinations; {metadata['n_single_observation_changed_solution_rows']} solution rows change a count or response-mass optimum. This isolates transform-before-averaging concerns without altering the main recorded-mean result.", '', f"Passed {metadata['n_checks']} checks. Independent exhaustive enumeration over both row and background signs tests61 small graphs, distinct count/mass optima, isolates and sparse masks. Every saved assignment reconstructs residual cells, objectives, cycle and support accounting; no-cycle graphs have zero count loss. All302edges, both populations, both panels, all five thresholds and all20 class rows remain. Inputs are SHA-256 unchanged.", '', 'Run `python audit/sparse_sign_feasibility_tests.py`, then `python code/sparse_sign_feasibility.py`. Results include panels, solutions, components, summary, equal_class_summary, single_observation_sensitivity, single_observation_summary, E43_overlap, nodes, assignments.parquet, mismatched_cells.parquet and checks. Parquet assignments use panel_id, objective0=count/1=mass, node_kind0=row/1=background, and edge-local node_index resolved through nodes.csv. Mismatched cell_id is one-based row position in the hash-pinned original matched_response_profiles.parquet. Component0 means isolated. Metadata pins versions, inputs and all outputs. All raw inputs, old audited analyses and shared integration files are untouched.', ''])

def main():
    paths = ['results/matched_response_profiles.parquet', 'results/condition_network_meta.csv', 'results/monotone_sign_separability_signatures.csv']
    before = {p: sha(BASE / p) for p in paths}
    profiles = pd.read_parquet(BASE / paths[0]).reset_index(drop=True)
    profiles['cell_id'] = np.arange(1, len(profiles) + 1, dtype=np.int64)
    classes = sorted(pd.read_csv(BASE / paths[1]).reaction_type.unique())
    prior = pd.read_csv(BASE / paths[2]).set_index(EDGE + ['population', 'threshold_pp'])
    check('full_input_universe', len(profiles) == 267462 and len(classes) == 20)
    check('profile_row_background_unique', not profiles.duplicated(EDGE + ['substrate_pair_key', 'background_fingerprint']).any())
    check('both_positive_original_arm_membership', np.array_equal(profiles.both_positive, (profiles.high_yield_pp > 0) & (profiles.low_yield_pp > 0)))
    check('duplicate_profile_count', profiles.n_observations.gt(1).sum() == 38)
    panels = []
    solutions = []
    components = []
    nodes = []
    sensitivity = []
    overlaps = []
    assign = ParquetSink(BASE / 'results/sparse_sign_feasibility_assignments.parquet')
    mismatched = ParquetSink(BASE / 'results/sparse_sign_feasibility_mismatched_cells.parquet')
    panel_id = 0
    main_panels = 0
    changed_single = 0
    try:
        for edge_id, (edge, g) in enumerate(profiles.groupby(EDGE, sort=True), 1):
            g = g.copy()
            rowkeys = sorted(g.substrate_pair_key.unique())
            backgrounds = sorted(g.background_fingerprint.unique())
            rowmap = {key: i for i, key in enumerate(rowkeys)}
            bgmap = {key: i for i, key in enumerate(backgrounds)}
            g['row_index'] = g.substrate_pair_key.map(rowmap)
            g['background_index'] = g.background_fingerprint.map(bgmap)
            for kind, values in [(0, rowkeys), (1, backgrounds)]:
                nodes.extend((dict(edge_id=edge_id, node_kind=kind, node_index=i, label=value, **dict(zip(EDGE, edge))) for i, value in enumerate(values)))
            nbg = len(backgrounds)
            check(f'background_enum_bound_{edge_id}', nbg <= 18)
            affected = g.n_observations.gt(1).any()
            for population in ('all', 'both_positive'):
                gpop = g if population == 'all' else g[g.both_positive]
                for panel_name in ('observed_sparse', 'complete_population_rows'):
                    for threshold in THRESHOLDS:
                        original = None
                        original_id = None
                        for observation_filter in ['all', 'single_observation'] if affected else ['all']:
                            gp = gpop if observation_filter == 'all' else gpop[gpop.n_observations.eq(1)]
                            rowcounts = gp.groupby('row_index').size()
                            selectedrows = rowcounts.index if panel_name == 'observed_sparse' else rowcounts[rowcounts.eq(nbg)].index
                            panel = gp[gp.row_index.isin(selectedrows)]
                            rowids = np.sort(panel.row_index.unique())
                            decisive = panel[panel.delta.ne(0)] if threshold == 0 else panel[panel.delta.abs().ge(threshold)]
                            rr = np.searchsorted(rowids, decisive.row_index.to_numpy())
                            bb = decisive.background_index.to_numpy(int)
                            signs = np.sign(decisive.delta.to_numpy()).astype(np.int8)
                            mass = decisive.delta.abs().to_numpy()
                            fitted = solve_graph(rr, bb, signs, mass, len(rowids), nbg)
                            panel_id += 1
                            if observation_filter == 'all':
                                main_panels += 1
                                original_id = panel_id
                            meta = dict(panel_id=panel_id, edge_id=edge_id, **dict(zip(EDGE, edge)), population=population, panel=panel_name, threshold_pp=threshold, observation_filter=observation_filter, n_original_backgrounds=nbg, n_population_rows=int(gp.row_index.nunique()), n_population_observed_cells=len(gp), n_population_missing_cells_from_union=int(gp.row_index.nunique()) * nbg - len(gp), n_panel_rows=len(rowids), n_panel_observed_cells=len(panel), n_panel_missing_cells_from_union=len(rowids) * nbg - len(panel), n_decisive_cells=len(decisive), n_abstaining_observed_cells=len(panel) - len(decisive), population_abs_mass_pp=math.fsum(gp.delta.abs()), panel_abs_mass_pp=math.fsum(panel.delta.abs()), decisive_abs_mass_pp=math.fsum(mass), **fitted['topology'])
                            panels.append(meta)
                            components.extend((dict(panel_id=panel_id, **c) for c in fitted['components']))
                            records = []
                            for objective_index, objective in enumerate(('count', 'mass')):
                                fit = fitted[objective]
                                loss = fit['n_mismatched']
                                weight = fit['mismatched_mass_pp']
                                record = dict(**meta, objective=objective, n_mismatched=loss, mismatched_mass_pp=weight, minimum_count_objective=objective == 'count', minimum_mass_objective=objective == 'mass', mismatched_cell_fraction_decisive=ratio(loss, len(decisive)), mismatched_cell_fraction_panel=ratio(loss, len(panel)), mismatched_cell_fraction_population=ratio(loss, len(gp)), mismatched_mass_fraction_decisive=ratio(weight, meta['decisive_abs_mass_pp']), mismatched_mass_fraction_panel=ratio(weight, meta['panel_abs_mass_pp']), mismatched_mass_pp_per_decisive_cell=ratio(weight, len(decisive)), mismatched_mass_pp_per_panel_cell=ratio(weight, len(panel)), mismatched_mass_pp_per_population_cell=ratio(weight, len(gp)))
                                records.append(record)
                                if observation_filter == 'all':
                                    solutions.append(record)
                                else:
                                    sensitivity.append(dict(**record, reference_panel_id=original_id, reused_original_assignment=False))
                                nodecount = len(rowids) + nbg
                                assign.append(dict(panel_id=np.full(nodecount, panel_id, dtype=np.int32), objective=np.full(nodecount, objective_index, dtype=np.int8), node_kind=np.r_[np.zeros(len(rowids), np.int8), np.ones(nbg, np.int8)], node_index=np.r_[rowids, np.arange(nbg)].astype(np.int32), component=np.r_[fitted['row_components'], fitted['background_components']], sign=np.r_[fit['row_signs'], fit['background_signs']]))
                                wrong = fit['mismatched']
                                mismatched.append(dict(panel_id=np.full(loss, panel_id, dtype=np.int32), objective=np.full(loss, objective_index, dtype=np.int8), cell_id=decisive.cell_id.to_numpy(np.int64)[wrong], abs_delta_pp=mass[wrong], observed_sign=signs[wrong], fitted_sign=(fit['row_signs'][rr] * fit['background_signs'][bb])[wrong]))
                                check(f'reconstruct_assignment_{panel_id}_{objective}', int(wrong.sum()) == loss and np.array_equal(wrong, fit['row_signs'][rr] * fit['background_signs'][bb] != signs))
                            check(f'no_cycle_has_zero_loss_{panel_id}', meta['cycle_rank'] > 0 or fitted['count']['n_mismatched'] == 0)
                            check(f'all_support_cells_partition_{panel_id}', meta['n_decisive_cells'] + meta['n_abstaining_observed_cells'] == len(panel) and meta['n_cells_on_degree1_rows'] + meta['n_cells_on_degree_ge2_rows'] == len(decisive))
                            if observation_filter == 'all':
                                original = records
                                if panel_name == 'complete_population_rows':
                                    matrix = panel.pivot(index='row_index', columns='background_index', values='delta').reindex(columns=range(nbg)).to_numpy()
                                    full = (matrix != 0).all(axis=1) if threshold == 0 else (abs(matrix) >= threshold).all(axis=1)
                                    ss = np.sign(matrix[full]).astype(np.int8)
                                    patterns = Counter((tuple(row * row[0]) for row in ss))
                                    old = prior.loc[edge + (population, threshold)]
                                    contradiction = len(patterns) > 1
                                    check(f'E43_identical_fully_decisive_subset_{edge_id}_{population}_{threshold:g}', len(rowids) == old.n_population_rows and len(ss) == old.n_fully_decisive_rows and (len(patterns) == old.n_antipodal_classes) and (contradiction == old.sign_contradiction))
                                    check(f'E43_subset_conflict_implies_sparse_conflict_{edge_id}_{population}_{threshold:g}', not contradiction or fitted['count']['n_mismatched'] > 0)
                                    overlaps.append(dict(edge_id=edge_id, **dict(zip(EDGE, edge)), population=population, threshold_pp=threshold, n_E47_complete_population_rows=len(rowids), n_E43_fully_decisive_rows=len(ss), n_antipodal_classes=len(patterns), E43_fully_decisive_contradiction=contradiction, E47_complete_sparse_contradiction=fitted['count']['n_mismatched'] > 0, n_E47_decisive_cells=len(decisive), n_E43_fully_decisive_cells=len(ss) * nbg))
                            else:
                                changed_single += sum((old['n_mismatched'] != new['n_mismatched'] or abs(old['mismatched_mass_pp'] - new['mismatched_mass_pp']) > 1e-10 for old, new in zip(original, records)))
                        if not affected:
                            sensitivity.extend((dict(**r, reference_panel_id=r['panel_id'], reused_original_assignment=True, sensitivity_observation_filter='single_observation') for r in original))
            if edge_id % 30 == 0:
                print(f'Sparse sign graph: {edge_id}/302 edges; {panel_id} solved panel masks', flush=True)
    finally:
        assign.close()
        mismatched.close()
    panels = pd.DataFrame(panels)
    solutions = pd.DataFrame(solutions)
    sensitivity = pd.DataFrame(sensitivity)
    check('all302_edges_bothpop_bothpanels_5thresholds', main_panels == 303 * 2 * 2 * 5 and len(solutions) == 303 * 2 * 2 * 5 * 2)
    check('all302_retained_every_solution_stratum', solutions.groupby(['population', 'panel', 'threshold_pp', 'objective']).size().eq(303).all())
    check('single_observation_complete_stratum_universe', len(sensitivity) == len(solutions))
    check('no_negative_or_overfull_count_distance', solutions.n_mismatched.ge(0).all() and solutions.n_mismatched.le(solutions.n_decisive_cells).all())
    check('weighted_mass_bound', solutions.mismatched_mass_pp.ge(0).all() and (solutions.mismatched_mass_pp <= solutions.decisive_abs_mass_pp + 1e-10).all())
    summary, equal = summarize(solutions, classes)
    singlesummary, _ = summarize(sensitivity, classes)
    check('all20_classes_every_summary_stratum', summary[summary.stratum.ne('ALL')].groupby(['population', 'panel', 'threshold_pp', 'objective']).size().eq(20).all())
    check('all_input_hashes_unchanged', all((sha(BASE / p) == h for p, h in before.items())))
    outputs = dict(panels=panels, solutions=solutions, components=pd.DataFrame(components), summary=summary, equal_class_summary=equal, single_observation_sensitivity=sensitivity, single_observation_summary=singlesummary, E43_overlap=pd.DataFrame(overlaps), nodes=pd.DataFrame(nodes), checks=pd.DataFrame(CHECKS))
    for name, frame in outputs.items():
        frame.to_csv(BASE / f'results/sparse_sign_feasibility_{name}.csv', index=False)
    output_paths = {name: f'results/sparse_sign_feasibility_{name}.csv' for name in outputs}
    output_paths.update(assignments='results/sparse_sign_feasibility_assignments.parquet', mismatched_cells='results/sparse_sign_feasibility_mismatched_cells.parquet')
    metadata = dict(n_checks=len(CHECKS), n_failed_checks=0, input_sha256=before, code_sha256=sha(__file__), test_code_sha256=sha(BASE / 'audit/sparse_sign_feasibility_tests.py'), output_sha256={name: sha(BASE / path) for name, path in output_paths.items()}, output_rows={**{name: len(frame) for name, frame in outputs.items()}, 'assignments': assign.n_rows, 'mismatched_cells': mismatched.n_rows}, n_original_edges=302, n_classes=20, thresholds_pp=THRESHOLDS, chunk_cases=CHUNK, blas_threads='library_default_unrestricted', n_main_graph_masks=main_panels, n_solved_graph_masks=len(panels), n_single_observation_changed_solution_rows=changed_single, objective='Independent countweight1 and abs(original delta) objectives; conditional row signs exact; componentwise background enumeration', weighted_precision='Conservative float64 absolute-score screen 64eps*(nrows+nbg+1)*mass; every candidate within twice bound compared using exact common-denominator binary integers', tie_rule='Fix first sorted active background per component +1; background vector lexicographic +1 before -1; row ties +1; isolated signs +1', assignment_schema='panel_id; objective0count/1mass; node_kind0row/1background; edge-local node_index resolves nodes.csv; component0isolated', mismatched_cell_id='One-based row in hash-pinned original matched_response_profiles.parquet', interpretation='Global in-sample necessary sign-constraint fit. Deleting constraints is not zeroing cells. Weighted loss is distance only to sign-compatible orthants with arbitrary percell magnitudes, not rankone magnitude cone or realizable monotone map. Reuses E43 signs; no independent evidence, pvalues, mechanism, noise correction or prospective claim', versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, pyarrow=pa.__version__))
    (BASE / 'reports/SPARSE_SIGN_FEASIBILITY.md').write_text(report(summary, equal, sensitivity, metadata), encoding='utf-8')
    metadata['report_sha256'] = sha(BASE / 'reports/SPARSE_SIGN_FEASIBILITY.md')
    (BASE / 'audit/sparse_sign_feasibility_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    show = summary[summary.stratum.eq('ALL') & summary.population.eq('all')]
    print(show[['panel', 'threshold_pp', 'objective', 'n_decisive_edges', 'n_cycle_edges', 'n_conflicting_edges', 'mismatched_cell_fraction_decisive_equal_edge_mean', 'mismatched_mass_fraction_decisive_equal_edge_mean']].to_string(index=False))
    print(f'Passed {len(CHECKS)} checks; {main_panels} main and {len(panels) - main_panels} single-observation graph masks', flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
