"""Read every M1 headline number from the regenerated pipeline outputs.

Each number is pinned to the approved M1 value, at its displayed
precision. A filter that matches no rows, or a value outside half a display
unit, fails the run. Writes ``out/m1_numbers.csv`` and ``out/corpus_nesting.csv``."""
from __future__ import annotations
import gzip
import json
import ast
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import pandas as pd
S = Path(__file__).resolve().parent
OUT = S.parent / 'out'
REFERENCE_LEARNING = S / 'reference_learning' / 'Analysis'
RECIPE_OPTIONS = S / 'recipe_options' / 'Analysis'
RESPONSE_LEARNING = S / 'response_learning'
EDM = RESPONSE_LEARNING / 'Existing_Data_Mining_2026-10-16'
rows, failures = ([], [])

def one(frame: pd.DataFrame, **keys) -> pd.Series:
    sel = frame
    for column, value in keys.items():
        sel = sel[sel[column] == value]
    if len(sel) != 1:
        raise ValueError(f'expected one row for {keys}, found {len(sel)}')
    return sel.iloc[0]

def pin(section: str, item: str, value: float, shown: str, unit: float):
    """Record value; it must round to the plan's shown number within unit/2."""
    target = float(shown.replace(',', ''))
    ok = abs(value - target) <= unit / 2 + 1e-12
    rows.append(dict(section=section, item=item, value=value, plan_value=shown, matches=ok))
    if not ok:
        failures.append(f'{section} | {item}: {value!r} vs plan {shown}')

def rounded_pin(section, item, value, shown, source='', selector=''):
    """Pin an actual displayed decimal, including exact half-up yield ties."""
    quantum = Decimal(shown.replace(',', ''))
    rounded = Decimal(str(value)).quantize(Decimal(1).scaleb(quantum.as_tuple().exponent), rounding=ROUND_HALF_UP)
    ok = rounded == quantum
    rows.append(dict(section=section, item=item, value=float(value), plan_value=shown, matches=ok, source=str(source), selector=selector))
    if not ok:
        failures.append(f'{section} | {item}: displayed {rounded} vs {shown}')
ex = pd.read_csv(EDM / 'results/focused_chemical_examples.csv')
ex = ex[ex.target_swap_id.eq('suzuki__C23__C26') & ex.sub_2_smiles.eq('COc1ccc(B(O)O)cc1')]
for smiles, label in [('Brc1ccc2ncccc2c1', 'bromoquinoline'), ('Brc1ccc2[nH]ncc2c1', 'bromo-1H-indazole')]:
    r = one(ex, sub_1_smiles=smiles)
    shown = {'bromoquinoline': ('4.98', '93.36'), 'bromo-1H-indazole': ('83.86', '11.55')}[label]
    pin('1', f'Suzuki PdCl2(dtbpf)->PdCl2(PPh3)2, {label}, y0 %', 100 * r.y0, shown[0], 0.01)
    pin('1', f'Suzuki PdCl2(dtbpf)->PdCl2(PPh3)2, {label}, y1 %', 100 * r.y1, shown[1], 0.01)
cells = pd.read_csv(RECIPE_OPTIONS / 'round9/pairing_repair/selected_pairing_repair_all_12_yields_and_conditions.csv')
cells = cells[cells.action_id.eq('0e5d6cf7a808adc5a0a4_d0_s01_r1')]
if len(cells) != 12:
    raise ValueError('Suzuki methyl-shift panel must have 12 cells')

def cell(side, catalyst, base):
    sel = cells[cells.molecular_side.eq(side) & cells.ternary_component.str.contains(catalyst, regex=False) & cells.binary_component.str.contains(base, regex=False)]
    if len(sel) != 1:
        raise ValueError(f'{side} {catalyst} {base}: {len(sel)} rows')
    return 100 * float(sel.yield_numeric.iloc[0])
pin('1', 'Suzuki 3->2 methyl, PdCl2(PCy3)2/K3PO4 before %', cell('before', 'PCy3', 'K3PO4'), '47.62', 0.01)
pin('1', 'Suzuki 3->2 methyl, PdCl2(PCy3)2/K3PO4 after %', cell('after', 'PCy3', 'K3PO4'), '0', 0.01)
pin('1', 'Suzuki 3->2 methyl, SPhos Pd G3/K3PO4 after %', cell('after', 'SPhos', 'K3PO4'), '92.48', 0.01)
pin('1', 'Suzuki 3->2 methyl, SPhos Pd G3/K3PO4 before %', cell('before', 'SPhos', 'K3PO4'), '13.49', 0.01)
acy = pd.read_csv(RECIPE_OPTIONS / 'round8/context_decomposition/selected_actual_yields_and_lower_order_reconstructions.csv')
acy = acy[acy.partner_pair_comparison_id.astype(str).eq('5e3f5818ec51384251')]
if len(acy) != 16:
    raise ValueError('acylation fluorination panel must have 16 cells')
RECIPES = {0: 'DCC/Oxyma', 10: 'DCC/Cl-HOBT', 1: 'DIC/Oxyma', 11: 'DIC/Cl-HOBT'}
for (edit, partner), shown in {('A', 'C'): ('0', '79.84', '77.16', '0'), ('A', 'D'): ('73.21', '0', '0', '72.53'), ('B', 'C'): ('76.61', '0', '73.00', '77.73'), ('B', 'D'): ('80.50', '81.21', '82.10', '80.68')}.items():
    block = acy[acy.edit_level.eq(edit) & acy.partner_level.eq(partner)].set_index('corner').observed_yield_pct
    for corner, target in zip([0, 10, 1, 11], shown):
        pin('1', f'amide fluorination acid {edit} amine {partner} {RECIPES[corner]} %', block.loc[corner], target, 0.01 + 1e-09)
act = pd.read_csv(RECIPE_OPTIONS / 'cases/figure_common8_activators.csv').groupby('activator').structural_double_difference_pp.agg(['mean', 'size'])
for name, shown in [('PyAOP', '23.50'), ('TFFH', '-4.81'), ('BTFFH', '-3.25'), ('DCMT', '-0.10')]:
    if act.loc[name, 'size'] != 8:
        raise ValueError(f'{name}: expected 8 shared acids')
    pin('1', f'NMM->DIPEA, 4-MeO-benzylamine minus benzylamine, {name} pp', act.loc[name, 'mean'], shown, 0.01)
blocks = pd.read_csv(EDM / 'results/focused_chemical_partner_blocks.csv')
fdpp = blocks[blocks.target_swap_id.eq('acylation__C18__C24') & blocks.swap_id.eq('acylation__C18__C24') & blocks.scope.eq('all') & blocks.comparison.eq('primary_aniline_vs_primary_alkyl')]
pin('1', 'FDPP->TOTU aryl minus alkyl primary amine, acid blocks', len(fdpp), '42', 1)
pin('1', 'FDPP->TOTU aryl minus alkyl primary amine, mean contrast pp', fdpp.contrast_pp.mean(), '33.87', 0.01)
sign = pd.read_csv(EDM / 'results/sparse_sign_feasibility_summary.csv')
for threshold, cyc, con in [(0, '293', '281'), (20, '218', '148')]:
    r = one(sign, population='all', panel='observed_sparse', objective='count', stratum='ALL', threshold_pp=threshold)
    pin('2', f'sign test |response|>={threshold} pp, cyclic exchanges', r.n_cycle_edges, cyc, 1)
    pin('2', f'sign test |response|>={threshold} pp, conflicting exchanges', r.n_conflicting_edges, con, 1)
for threshold, cyc, con in [(0, '283', '271'), (20, '212', '143')]:
    excluded = one(sign, population='all', panel='observed_sparse', objective='count', stratum='finkelstein', threshold_pp=threshold)
    total = one(sign, population='all', panel='observed_sparse', objective='count', stratum='ALL', threshold_pp=threshold)
    pin('2', f'sign test 19 arrays |response|>={threshold} pp, cyclic exchanges', total.n_cycle_edges - excluded.n_cycle_edges, cyc, 1)
    pin('2', f'sign test 19 arrays |response|>={threshold} pp, conflicting exchanges', total.n_conflicting_edges - excluded.n_conflicting_edges, con, 1)
copy = pd.read_csv(REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv')
copy = copy[copy.level.eq('headline')].set_index('model').mse_pp2
marginal = copy['marginal']
pin('2', 'copy reference difference: error change vs training marginal %', 100 * (copy['direct_raw'] / marginal - 1), '42.4', 0.1)
pin('2', 'copy plus training offset: error change vs marginal %', 100 * (copy['offset_raw'] / marginal - 1), '40.7', 0.1)
cross = pd.read_csv(REFERENCE_LEARNING / 'tables/cross_library_source_channels_headline.csv')
r = one(cross, domain='b24_fixed', relation='cross_family', macro='direction_macro', channel='d', budget=192)
pin('2', 'cross-library budget 192: correct correspondence minus reassignment pp2', 10000.0 * r.delta_test, '-8.662', 0.001)
pin('2', 'cross-library budget 192: correct correspondence penalty pp2', 10000.0 * (r.real_mse - r.test_permuted_exact_mse), '8.662', 0.001)
pin('2', 'cross-library budget 192: excess over target-only calibration pp2', -10000.0 * r.net_gain, '17.308', 0.001)
spec = pd.read_csv(REFERENCE_LEARNING / 'tables/chemical_observation_specificity_headline.csv')
spec = spec[spec.domain.eq('strict_641') & spec.scope.eq('all_panels_complete') & spec.weighting.eq('context_balanced') & spec.macro.eq('family_macro')]
if len(spec) != 6:
    raise ValueError('expected six split x history settings')
pin('3', 'same-swap reference vs matched control, lowest error reduction %', 100 * spec.other_recovery_vs_control.min(), '8.94', 0.01)
pin('3', 'same-swap reference vs matched control, highest error reduction %', 100 * spec.other_recovery_vs_control.max(), '11.90', 0.01)
pin('3', 'learned relationship: error change vs training marginal %', 100 * (copy['semantic_full_raw'] / marginal - 1), '-13.7', 0.1)
chan = pd.read_csv(REFERENCE_LEARNING / 'tables/chemical_observation_channels_headline.csv')
r = one(chan, domain='strict_641', scope='all_panels_complete', split='pair', fraction=1.0, weighting='context_balanced', macro='family_macro')
pin('3', 'D beyond S, same swap pp2', 10000.0 * r.semantic_full_optimal_d_given_s_raw_mse, '13.372', 0.001)
pin('3', 'D beyond S, matched control swap pp2', 10000.0 * r.different_other_full_optimal_d_given_s_raw_mse, '2.447', 0.001)
scr = pd.read_csv(REFERENCE_LEARNING / 'tables/chemical_observation_channels_screen_scores.csv')
scr = scr[scr.domain.eq('strict_641') & scr.scope.eq('all_panels_complete') & scr.split.eq('pair') & scr.fraction.eq(1.0) & scr.weighting.eq('context_balanced')]
pin('3', 'libraries scored', len(scr), '17', 1)
pin('3', 'libraries with positive D-beyond-S specificity', int((scr.other_full_optimal_d_given_s_specificity_raw_mse > 0).sum()), '13', 1)
e63 = pd.read_csv(RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results/e63_paired_component_summary.csv')
e63 = e63[e63.comparison.eq('centroid_minus_codebook_continuous')]
if len(e63) != 6:
    raise ValueError('expected two dictionary sizes x three weightings')
share = 100 * e63.offset_difference_pp2 / e63.mse_difference_pp2
pin('3', "structure model: share of net gain from each pair's own mean over backgrounds, min %", share.min(), '91.1', 0.1)
pin('3', "structure model: share of net gain from each pair's own mean over backgrounds, max %", share.max(), '96.6', 0.1)
summary = pd.read_csv(RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results/e63_summary.csv')
k3 = e63[(e63.nominal_k == 3) & e63.weighting.eq('equal_class_edge_balanced')].iloc[0]
cen = summary[summary.model.eq('centroid') & (summary.nominal_k == 3) & summary.weighting.eq('equal_class_edge_balanced')]
if len(cen) != 1:
    raise ValueError('centroid k=3 class-balanced row')
cen = cen.iloc[0]
pin('3', 'modulation error removed by structure model, k=3 class-balanced %', 100 * (k3.amplitude_difference_pp2 + k3.alignment_difference_pp2) / (cen.amplitude_mse_pp2 + cen.alignment_mse_pp2), '0.22', 0.01)
e66 = pd.read_csv(RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results/e66_summary.csv')
key = dict(population='common8', target_geometry='raw_delta', stratum='all', weighting='equal_class_edge_balanced', budget='8')
joint, local = (one(e66, learner='joint', **key), one(e66, learner='calibration_only', **key))
pin('3', 'history + 8 local profiles, class-balanced MSE pp2', joint.native_vector_mse, '195.30', 0.01)
pin('3', '8 local profiles alone, class-balanced MSE pp2', local.native_vector_mse, '216.29', 0.01)
pin('3', 'history alone, class-balanced MSE pp2', one(e66, learner='frozen_family', **key).native_vector_mse, '200.90', 0.01)
pin('3', 'local-update queries', joint.n_test_vectors, '24221', 1)
e67 = pd.read_csv(RESPONSE_LEARNING / 'Local_Information_Content_E67/results/e67_decomposition_classes_20.csv')
for cls, lvl, mod in [('buchwald-hartwig', '-10.125', '-13.276'), ('ullmann', '3.632', '0.092')]:
    r = one(e67, reaction_type=cls, stratum='all')
    pin('4', f'{cls}: local level labels, change in MSE pp2', r.local_level_content, lvl, 0.001)
    pin('4', f'{cls}: local modulation labels, change in MSE pp2', r.local_modulation_content, mod, 0.001)
rounded_pin('4', 'ullmann local modulation near-zero display pp2', one(e67, reaction_type='ullmann', stratum='all').local_modulation_content, '+0.09')
e67_summary = pd.read_csv(RESPONSE_LEARNING / 'Local_Information_Content_E67/results/e67_decomposition_summary.csv')
r = one(e67_summary, stratum='all', weighting='equal_class_edge_balanced')
pin('audit', 'overall local modulation content, array-balanced MSE pp2', r.local_modulation_content, '0.234', 0.001)
e65 = pd.read_csv(RESPONSE_LEARNING / 'Structure_Family_Transport_E65/results/e65_paired_summary.csv')
r = one(e65, population='primary', target_geometry='raw_delta', comparison='family_minus_mean_controls', model='uncompressed_krr', stratum='all', weighting='equal_class_edge_balanced')
pin('SI', 'scaffold-family exclusion extra cost MSE pp2', r.difference_native_vector_mse, '0.49', 0.01)
repair = json.loads((RECIPE_OPTIONS / 'round9/pairing_repair/run_summary.json').read_text(encoding='utf-8'))
pin('4', 'distinct failed-to-recovered comparisons', repair['n_unique_measured_failed_to_repaired_cell_pairs'], '117', 1)
rep = pd.read_csv(RECIPE_OPTIONS / 'round9/pairing_repair/all_actual_yield_repairs_with_12cells.csv')
pin('4', 'recovery actions', len(rep), '132', 1)
pin('4', 'recovery: exact molecular edits', rep.edit_id.nunique(), '50', 1)
pin('4', 'recovery: arrays', rep.dataset_id.nunique(), '6', 1)
conf = pd.read_csv(RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv')
pin('4', 'shared-scope conflict views', len(conf), '542', 1)
pin('4', 'recoverable within tested recipes', int(conf.full_graph_rescues20.sum()), '285', 1)
pin('4', 'not recoverable within tested recipes', int((~conf.full_graph_rescues20.astype(bool)).sum()), '257', 1)
opt = pd.read_csv(RECIPE_OPTIONS / 'round29/component_options/minimum_option_distribution.csv')
opt = opt[opt.endpoint_corner_stratum.eq('all')].set_index('minimum_extra_options').n_certificates
for k, shown in [(1, '266'), (2, '16'), (3, '3')]:
    pin('4', f'recoverable views needing {k} extra option(s)', opt.loc[k], shown, 1)
wit = json.loads((RECIPE_OPTIONS / 'round27/route_compromise/identical_native_support_witness/witness.json').read_text(encoding='utf-8'))
pin('SI', 'cost of merging two Suzuki scopes pp', 100 * wit['compromise'], '13.48', 0.01)
front = pd.read_csv(RECIPE_OPTIONS / 'round29/source_compromise/measured_cases_exact_pareto_frontiers.csv')
front = front[front.case.eq('positive_source_compromise')]
pin('SI', 'acylation minimum source sacrifice keeping target >= 20% pp', 100 * front.loc[front.target_bottleneck >= 0.2, 'source_sacrifice_from_B'].min(), '5.00', 0.01)
import m1_storyboard as storyboard
storyboard.RECORDS.clear()
storyboard.load_data(verify_pins=False)
figure_records = pd.DataFrame(storyboard.RECORDS)
figure_baselines = {(1, 'a'): '4.98 93.36 83.86 11.55', (1, 'b'): '70.70 30.06 29.50 47.62 7.35 13.49 70.82 51.60 0.00 0.00 51.13 92.48', (4, 'c'): '91.34 84.79 84.44 94.04 89.73 77.92 0.00 41.25 79.00 85.79 92.79 78.26', (2, 'a'): '-72.31 83.86 11.55 -28.38 51.04 22.66 -20.63 64.81 44.19 33.87 18.44 52.31', (2, 'b'): '271 283 143 212', (2, 'c'): '122.99 140.30 131.64', (3, 'a'): '160.02 227.80 225.11 138.07', (3, 'b'): '8.94 10.95 10.23 10.16 11.90 11.25', (3, 'c'): '93.9 95.1 96.5 91.1 92.8 96.6 84.07 0.19 0.22', (4, 'a'): '216.29 200.90 195.30', (4, 'b'): '266 16 3 257'}
if len(figure_records) != 73:
    raise ValueError(f'Expected 73 approved storyboard quantities, found {len(figure_records)}')
for (fig, panel), expected in figure_baselines.items():
    selected = figure_records[figure_records.figure.eq(fig) & figure_records.panel.eq(panel)]
    expected = expected.split()
    if len(selected) != len(expected):
        raise ValueError(f'Figure {fig}{panel}: changed source population')
    for index, (record, shown) in enumerate(zip(selected.to_dict('records'), expected), 1):
        rounded_pin('figure', f"Figure {fig}{panel}: {record['label']} [row {index}]", record['value'], shown, record['source'], record['selector'])
amide = one(rep, action_id='4bcee624ba37f2c2cef3_d0_s01_r0')
if amide.measured_cell_repair_id != 'f8790df12a47c7b2480c':
    raise ValueError('The HBTU/TPTU measured recovery identity changed')
rounded_pin('4', 'unique measured failure-to-recovery identities', rep.measured_cell_repair_id.nunique(), '117')
for column, shown in [('y0_a', '91.34'), ('y0_b', '0.00'), ('y4_b', '92.79'), ('y4_a', '89.73')]:
    rounded_pin('4', f'HBTU to TPTU methylthio relocation {column} yield %', 100 * amide[column], shown, 'scripts/recipe_options/Analysis/round9/pairing_repair/all_actual_yield_repairs_with_12cells.csv', 'action_id=4bcee624ba37f2c2cef3_d0_s01_r0; measured_cell_repair_id=f8790df12a47c7b2480c')
for column, label, shown in [('y0_a', 'HBTU before', '91.34'), ('y4_a', 'TPTU before', '89.73'), ('y0_b', 'HBTU after', '0.00'), ('y4_b', 'TPTU after', '92.79')]:
    rounded_pin('4', 'amide SMe ' + label + ' %', 100 * amide[column], shown, 'scripts/recipe_options/Analysis/round9/pairing_repair/all_actual_yield_repairs_with_12cells.csv', 'action_id=4bcee624ba37f2c2cef3_d0_s01_r0')
fdpp_expected = '45.846913 69.978857 53.048615 66.330379 91.804417 63.460334 114.711708\n26.759959 43.382828 0.572772 50.766195 16.073263 12.627276 49.386109 22.638306 0.000000\n70.074098 46.105485 2.498942 13.782395 38.164067 31.274792 -4.809001 29.670622 9.262472\n18.185419 11.321311 24.312973 21.132088 -6.780081 1.252579 32.988627 25.740918 32.364832\n35.327544 -0.410685 62.709245 45.302430 37.522138 67.446938 24.459197 26.449799'.split()
if len(fdpp) != len(fdpp_expected) or fdpp.shared_partner_smiles.nunique() != 42:
    raise ValueError('The supplementary FDPP/TOTU acid population changed')
for index, (record, shown) in enumerate(zip(fdpp.itertuples(index=False), fdpp_expected), 1):
    rounded_pin('SI figure', f'Supplementary Figure S1b acid contrast row {index}', record.contrast_pp, shown, 'scripts/response_learning/Existing_Data_Mining_2026-10-16/results/focused_chemical_partner_blocks.csv', 'acylation__C18__C24; all; primary_aniline_vs_primary_alkyl; shared_partner_smiles=' + record.shared_partner_smiles)
base_points = pd.read_csv(RECIPE_OPTIONS / 'cases/figure_common8_activators.csv')
acid_order = 'A05 A10 A04 A06 A01 A07 A09 A03'.split()
base_expected = {'PyAOP': '25.088670 10.574320 28.144760 20.431440 44.279800 16.861920 12.770790 29.841750', 'BTFFH': '-1.473920 -16.703110 -0.009500 -0.990640 9.081450 -22.348850 5.786900 0.681930', 'TFFH': '-0.934230 -10.027870 -0.282140 0.546560 -1.670300 -26.432080 1.042460 -0.758600', 'DCMT': '-0.489580 -1.018370 0.000500 -0.273280 2.620750 -1.466620 -0.155930 -0.037930'}
for activator, expected in base_expected.items():
    for acid, shown in zip(acid_order, expected.split()):
        record = one(base_points, activator=activator, acid_label=acid)
        rounded_pin('SI figure', f'Supplementary Figure S1c {activator} acid {acid} difference-in-differences pp', record.structural_double_difference_pp, shown, 'scripts/recipe_options/Analysis/cases/figure_common8_activators.csv', f'activator={activator}; acid_label={acid}')
specificity_expected = {'Acylation|1': '36.466603', 'Buchwald-Hartwig(CMC)|2': '0.014636', 'Buchwald-Hartwig|1': '14.008188', 'Buchwald-Hartwig|2': '-1.207115', 'C-O competitive reactions|1': '-4.820697', 'Chan-Lam|1': '7.823537', 'Chan-Lam|2': '-1.864878', 'Esterification|1': '16.069462', 'Grignard|1': '2.441652', 'Heck|1': '4.891411', 'Hiyama|1': '0.789745', 'Kumada|1': '45.716110', 'Reductive-amination|1': '-2.293209', 'SNAr|1': '6.991821', 'Sonogashira|1': '6.688697', 'Suzuki|1': '21.314853', 'Ullmann|1': '11.450975'}
for screen, shown in specificity_expected.items():
    record = one(scr, screen=screen)
    rounded_pin('SI figure', f'Supplementary Figure S2a {screen} D beyond S specificity pp2', 10000.0 * record.other_full_optimal_d_given_s_specificity_raw_mse, shown, 'scripts/reference_learning/Analysis/tables/chemical_observation_channels_screen_scores.csv', 'strict_641; all_panels_complete; pair; fraction=1; context_balanced; screen=' + screen)
for key, shown in [('B_context_a', '25.615'), ('B_context_b', '23.761'), ('B_union', '10.285')]:
    rounded_pin('SI figure', 'Supplementary Figure S3a ' + key + ' bottleneck yield percent', 100 * wit[key], shown, 'scripts/recipe_options/Analysis/round27/route_compromise/identical_native_support_witness/witness.json')
for index, (record, expected) in enumerate(zip(front.itertuples(index=False), [('0.000000', '0.000000'), ('4.996850', '66.280880'), ('10.171710', '70.086960'), ('20.909550', '72.540880')]), 1):
    for column, shown in zip(['source_sacrifice_from_B', 'target_bottleneck'], expected):
        rounded_pin('SI figure', f'Supplementary Figure S3b Pareto point {index} {column} percent', 100 * getattr(record, column), shown, 'scripts/recipe_options/Analysis/round29/source_compromise/measured_cases_exact_pareto_frontiers.csv', 'case=positive_source_compromise; row=' + str(index))
if len(front) != 4:
    raise ValueError('The supplementary amide Pareto population changed')
e62meta = json.loads((RESPONSE_LEARNING / 'Matched_Representation_E62/metadata/e62_metadata.json').read_text())
for key, shown in [('n_edges', '93'), ('n_classes', '14'), ('n_vectors', '25956'), ('n_allzero_vectors', '9924'), ('n_trainable_nonempty_folds', '1984')]:
    rounded_pin('protocol', 'structure model ' + key, e62meta[key], shown)
for key, shown in [('edges', '655'), ('contexts', '213'), ('screens', '17'), ('merged_families', '14'), ('target_score_count_including_repeat_use', '242766')]:
    verified = json.loads((REFERENCE_LEARNING / 'Nature_Manuscript/qa/matched_copy_verification.json').read_text())
    rounded_pin('protocol', 'paired copy-learning ' + key, verified[key], shown)
cross = pd.read_csv(REFERENCE_LEARNING / 'tables/cross_library_source_channels_headline.csv')
cross = one(cross, domain='b24_fixed', relation='cross_family', macro='direction_macro', channel='d', budget=192)
for key, shown in [('n_scored_directions', '17'), ('n_scored_designs', '158'), ('n_target_families', '6')]:
    rounded_pin('protocol', 'cross-library ' + key, cross[key], shown)
panel = pd.read_csv(REFERENCE_LEARNING / 'tables/chemical_observation_channels_panel_registry.csv')
counts = panel[panel.strict_eligible.eq(1)].groupby(['screen', 'edge_id']).size().unique()
if len(counts) != 1:
    raise ValueError('Strict targets do not share a fixed panel count')
rounded_pin('protocol', 'observation policies separate two-well panels', counts[0], '11')
rounded_pin('protocol', 'observations per panel', panel.planned_cells.unique().item(), '2')
samples = pd.read_csv(REFERENCE_LEARNING / 'tables/chemical_mode_training_sample_registry.csv')
for value, shown in zip(sorted(samples.fraction.unique()), ['0.25', '1']):
    rounded_pin('protocol', 'training fraction ' + shown, value, shown)
folds = pd.read_csv(REFERENCE_LEARNING / 'data/cv_fold_registry.csv')
for split, repeats in [('pair', '2'), ('role1', '1'), ('role2', '1')]:
    selected = folds[folds.split.eq(split)]
    rounded_pin('protocol', split + ' holdout folds', selected.fold.nunique(), '5')
    rounded_pin('protocol', split + ' holdout repeats', selected.repeat.nunique(), repeats)
strict_registry_path = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_target_registry.csv'
strict_registry = pd.read_csv(strict_registry_path)
strict_screens = set(strict_registry.loc[strict_registry.strict_eligible.eq(1), 'screen'])
visibility = []
for screen, screen_samples in samples[samples.screen.isin(strict_screens)].groupby('screen'):
    matrix_path = REFERENCE_LEARNING / 'data' / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
    with np.load(matrix_path, allow_pickle=False) as saved:
        roles = np.asarray([pair.split(' >> ') for pair in saved['sps'].astype(str)])
    for sample in screen_samples.itertuples(index=False):
        if sample.split not in {'role1', 'role2'}:
            continue
        train = np.asarray(json.loads(sample.training_indices_json), int)
        test = np.asarray(json.loads(sample.test_indices_json), int)
        held_role = int(sample.split[-1]) - 1
        if np.isin(roles[test, held_role], roles[train, held_role]).any():
            raise ValueError('Held-out role identity reappears in the registered training set')
        other_role = 1 - held_role
        seen = np.isin(roles[test, other_role], roles[train, other_role])
        visibility.append(dict(split=sample.split, fraction=sample.fraction, n_test=len(test), n_other_partner_seen=int(seen.sum())))
visibility = pd.DataFrame(visibility)
for split, fraction, shown in [('role1', 0.25, '62.83'), ('role1', 1.0, '88.15'), ('role2', 0.25, '58.75'), ('role2', 1.0, '85.39')]:
    selected = visibility[visibility.split.eq(split) & visibility.fraction.eq(fraction)]
    selector = f'screens with strict_eligible targets; split={split}; fraction={fraction:g}; test-pair instances before endpoint availability; same-role partner identity in TRAIN'
    source = 'scripts/reference_learning/Analysis/tables/chemical_mode_training_sample_registry.csv; scripts/reference_learning/Analysis/data/matrix_*.npz'
    rounded_pin('protocol', f'{split} fraction {fraction:g} other partner visible percent', 100 * selected.n_other_partner_seen.sum() / selected.n_test.sum(), shown, source, selector)
    rounded_pin('protocol', f'{split} fraction {fraction:g} visibility test-pair instances', selected.n_test.sum(), '2354', source, selector)
specificity_path = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_headline.csv'
for split in ('pair', 'role1', 'role2'):
    selected = one(spec, split=split, fraction=1.0)
    selector = f'strict_641; all_panels_complete; context_balanced; family_macro; split={split}; fraction=1'
    for field, shown in [('n_scored_edges', '655'), ('n_contexts', '213'), ('n_swaps', '239'), ('target_score_count', '242766' if split == 'pair' else '121383')]:
        rounded_pin('protocol', f'same-swap complete-panel {split} {field}', selected[field], shown, specificity_path.relative_to(S.parent), selector)
family_scope_path = RESPONSE_LEARNING / 'Structure_Family_Transport_E65/results/e65_scope_vectors.parquet'
family_scope = pd.read_parquet(family_scope_path)
if family_scope.vector_id.nunique() != len(family_scope):
    raise ValueError('Family-exclusion support must contain one row per test vector')
family_counts = {'original vectors': (len(family_scope), '25956'), 'primary vectors': (int(family_scope.primary_supported.sum()), '25520'), 'at least five training vectors': (int(family_scope.sensitivity_n5plus.sum()), '24970'), 'zero training vectors': (int(family_scope.n_family_train.eq(0).sum()), '436'), 'one training vector': (int(family_scope.n_family_train.eq(1).sum()), '114'), 'two to four training vectors': (int(family_scope.n_family_train.between(2, 4).sum()), '436'), 'both families originally present': (int((family_scope.original_family_role1_present & family_scope.original_family_role2_present).sum()), '18215'), 'neither family originally present': (int((~family_scope.original_family_role1_present & ~family_scope.original_family_role2_present).sum()), '577'), 'at least one acyclic partner': (int(family_scope.either_acyclic.sum()), '2946')}
for item, (actual, shown) in family_counts.items():
    rounded_pin('protocol', 'scaffold-family exclusion ' + item, actual, shown, family_scope_path.relative_to(S.parent), 'one row per identity-purged test vector; n_family_train after both query families are excluded across roles')
primary_family = family_scope[family_scope.primary_supported]
for item, actual, shown in [('primary swaps', len(primary_family[['reaction_type', 'swap_slot', 'component_pair']].drop_duplicates()), '93'), ('primary arrays', primary_family.reaction_type.nunique(), '14')]:
    rounded_pin('protocol', 'scaffold-family exclusion ' + item, actual, shown, family_scope_path.relative_to(S.parent), 'primary_supported=True (n_family_train>=1)')

def source_constants(path):
    found = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = value
            elif isinstance(target, ast.Tuple):
                found.update({item.id: val for item, val in zip(target.elts, value) if isinstance(item, ast.Name)})
    return found
representation = RESPONSE_LEARNING / 'Matched_Representation_E62/code/matched_representation.py'
constants = source_constants(representation)
for index, capacity in enumerate(constants['CAPACITIES']):
    rounded_pin('protocol', f'dictionary capacity {index + 1}', capacity, ['2', '3'][index])
rounded_pin('protocol', 'structural ridge penalty', constants['ALPHA'], '10')
fingerprints = [node for node in ast.walk(ast.parse(representation.read_text())) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'GetMorganGenerator')]
if len(fingerprints) != 1:
    raise ValueError('Expected one structural fingerprint specification')
keywords = {kw.arg: ast.literal_eval(kw.value) for kw in fingerprints[0].keywords}
rounded_pin('protocol', 'Morgan fingerprint radius', keywords['radius'], '2')
rounded_pin('protocol', 'Morgan fingerprint bits', keywords['fpSize'], '2048')
for item, value, shown in [('Morgan radius', keywords['radius'], '2'), ('Morgan bits', keywords['fpSize'], '2048'), ('ridge alpha', constants['ALPHA'], '10'), ('structural seed', constants['SEED'], '20261017')]:
    rounded_pin('protocol', item, value, shown, representation.relative_to(S))
tree = ast.parse(representation.read_text(encoding='utf-8'))
kmeans = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'KMeans')]
if len(kmeans) != 1:
    raise ValueError('Expected one dictionary KMeans specification')
kmeans_keywords = {kw.arg: ast.literal_eval(kw.value) for kw in kmeans[0].keywords if kw.arg in {'n_init', 'max_iter'}}
rounded_pin('protocol', 'kmeans starts', kmeans_keywords['n_init'], '20', representation.relative_to(S))
rounded_pin('protocol', 'kmeans iterations', kmeans_keywords['max_iter'], '300', representation.relative_to(S))
identity_fold = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'identity_fold'))
moduli = [ast.literal_eval(node.right) for node in ast.walk(identity_fold) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)]
if len(moduli) != 1:
    raise ValueError('Expected one identity-fold modulus')
rounded_pin('protocol', 'structural identity fold groups', moduli[0], '5', representation.relative_to(S))
rounded_pin('protocol', 'structural role-fold combinations', moduli[0] ** 2, '25', representation.relative_to(S))
cross_constants = source_constants(REFERENCE_LEARNING / 'cross_library_source_channels.py')
rounded_pin('protocol', 'cross-library ridge penalty', cross_constants['ALPHA'], '3')
for index, budget in enumerate(cross_constants['BUDGETS']):
    rounded_pin('protocol', f'cross-library coordinate budget {index + 1}', budget, ['24', '48', '96', '192'][index])
rounded_pin('protocol', 'cross calibration cap', max(cross_constants['BUDGETS']), '192')
designs = pd.read_csv(REFERENCE_LEARNING / 'tables/cross_library_source_channels_design_scores.csv.gz')
designs = designs[designs.b24_scored_domain & designs.relation.eq('cross_family') & designs.channel.eq('d') & designs.budget.eq(192) & designs.status.eq('scored')]
if len(designs) != int(cross.n_scored_designs):
    raise ValueError('Cross-library protocol support differs from the headline support')
for operation, shown in [('min', '16'), ('max', '186')]:
    rounded_pin('protocol', 'cross actual ' + {'min': 'minimum', 'max': 'maximum'}[operation], getattr(designs.actual_spent, operation)(), shown)
scope = pd.read_parquet(RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results/e66_scope_vectors.parquet')
scope = scope[scope.common8]
for operation, shown in [('min', '3'), ('max', '15')]:
    backgrounds = getattr(scope.n_backgrounds, operation)()
    rounded_pin('protocol', 'local complete profile backgrounds ' + operation, backgrounds, shown)
    rounded_pin('protocol', 'eight local profiles condition wells ' + operation, 16 * backgrounds, {'min': '48', 'max': '240'}[operation])
    suffix = {'min': 'minimum', 'max': 'maximum'}[operation]
    rounded_pin('protocol', 'background ' + suffix, backgrounds, shown)
    rounded_pin('protocol', 'local well ' + suffix, 16 * backgrounds, {'min': '48', 'max': '240'}[operation])
rounded_pin('protocol', 'local allocations per query', pd.read_csv(RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results/e66_allocation_sensitivity.csv').allocation.nunique(), '3')
sign_source = EDM / 'code/sparse_sign_feasibility.py'
rounded_pin('protocol', 'sign threshold', max(source_constants(sign_source)['THRESHOLDS']), '20', sign_source.relative_to(S))
option_source = RECIPE_OPTIONS / 'round29/option_scope_domains/analyze_option_scope_domains.py'
rounded_pin('protocol', 'conflict recovery yield threshold percent', 100 * source_constants(option_source)['H'], '20', option_source.relative_to(S))
repair_source = RECIPE_OPTIONS / 'code/round9_pairing_repair.py'
repair_constants = source_constants(repair_source)
rounded_pin('protocol', 'repair preference margin pp', 100 * dict(repair_constants['THRESHOLDS'])['5pp'], '5', repair_source.relative_to(S))
rounded_pin('protocol', 'repair pairing margin', 100 * dict(repair_constants['THRESHOLDS'])['5pp'], '5', repair_source.relative_to(S))
repair_tree = ast.parse(repair_source.read_text(encoding='utf-8'))
gain_loop = next((node for node in ast.walk(repair_tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and all((isinstance(n, ast.Name) for n in node.target.elts)) and ([n.id for n in node.target.elts] == ['gain_tag', 'cutoff'])))
rounded_pin('protocol', 'repair gain threshold pp', 100 * dict(ast.literal_eval(gain_loop.iter))['gain10'], '10', repair_source.relative_to(S))
rounded_pin('protocol', 'repair gain threshold', 100 * dict(ast.literal_eval(gain_loop.iter))['gain10'], '10', repair_source.relative_to(S))
event = next((node.value for node in ast.walk(repair_tree) if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'event' for target in node.targets))))
yield_thresholds = [ast.literal_eval(node.comparators[0].left) for node in ast.walk(event) if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and (node.left.id == 'new')]
if len(yield_thresholds) != 1:
    raise ValueError('Expected one measured recovery yield threshold')
rounded_pin('protocol', 'repair yield threshold percent', 100 * yield_thresholds[0], '20', repair_source.relative_to(S))
rounded_pin('protocol', 'repair yield threshold', 100 * yield_thresholds[0], '20', repair_source.relative_to(S))
full_meta = json.loads((RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/metadata/full_edge_corpus_metadata.json').read_text())
rounded_pin('protocol', 'exact swaps', full_meta['strict_swaps'], '2135')
screen_source = S / 'reference_learning/external/hte_common.py'
screen_constants = source_constants(screen_source)
for key, item, shown in [('FILL', 'REFERENCE_LEARNING minimum observed fraction', '0.70'), ('JAC', 'REFERENCE_LEARNING condition signature Jaccard threshold', '0.90'), ('MIN_SUB', 'REFERENCE_LEARNING minimum substrate pairs', '50'), ('MIN_COND', 'REFERENCE_LEARNING minimum conditions', '12'), ('HOLD', 'REFERENCE_LEARNING GO sealed fraction', '0.20'), ('FRAC_NONGO', 'REFERENCE_LEARNING nonGO analysis fraction', '0.70'), ('SEED', 'REFERENCE_LEARNING partition seed', '20260818')]:
    rounded_pin('protocol', item, screen_constants[key], shown, screen_source.relative_to(S))
rounded_pin('protocol', 'REFERENCE_LEARNING prespecified GO screens', len(screen_constants['GO_SCREENS']), '8', screen_source.relative_to(S))
screen_registry = pd.read_csv(REFERENCE_LEARNING / 'data/screen_partition_registry.csv')
registered_screens = set(screen_registry.screen)
if not screen_constants['GO_SCREENS'].issubset(registered_screens):
    raise ValueError('A prespecified GO screen is absent from the REFERENCE_LEARNING reconstructed screens')
rounded_pin('protocol', 'REFERENCE_LEARNING qualifying nonGO screens', len(registered_screens - screen_constants['GO_SCREENS']), '11')
from m1_exclusions import load_exclusions
excluded = load_exclusions()
cleanup_path = OUT / 'M1_well_measurement_status_exclusions.json'
cleanup = json.loads(cleanup_path.read_text(encoding='utf-8'))
if cleanup['source_sha256'] != excluded.source_sha256 or cleanup['id_sha256'] != excluded.id_sha256 or cleanup['deleted_wells'] != len(excluded.ids) or (set(cleanup['matched_ids']) != excluded.ids) or (cleanup['source_records'] - cleanup['deleted_wells'] != cleanup['retained_records']):
    raise ValueError('Cleanup accounting differs from the authoritative exclusion CSV')
for key, item, shown in [('source_records', 'original source reaction records', '410983'), ('deleted_wells', 'excluded measurement records', '396'), ('retained_records', 'retained reaction records', '410587')]:
    rounded_pin('cleanup', item, cleanup[key], shown, cleanup_path.relative_to(S.parent), key)
rounded_pin('cleanup', 'excluded fraction of source corpus percent', 100 * cleanup['deleted_wells'] / cleanup['source_records'], '0.1', cleanup_path.relative_to(S.parent), '100*deleted_wells/source_records')
overlap_path = OUT / 'zhang_2025_overlap.json'
overlap = json.loads(overlap_path.read_text(encoding='utf-8'))
if overlap['exclusion_sha256'] != excluded.source_sha256:
    raise ValueError('Published overlap used a different exclusion CSV')
for key, item, shown in [('retained_measurement_ids', 'retained acylation measurement IDs', '114733'), ('published_measurement_ids', 'published Zhang measurement IDs', '47015'), ('overlapping_measurement_ids', 'published Zhang overlapping measurement IDs', '45546')]:
    rounded_pin('cleanup', item, overlap[key], shown, overlap_path.relative_to(S.parent), key)
obs = pd.read_parquet(EDM / 'processed/observations.parquet', columns=['reaction_type', 'yield', 'eligible_for_matching'])
recipe_options_map = json.loads((RECIPE_OPTIONS / 'data/mapping_summary.json').read_text(encoding='utf-8'))
reference_learning_manifest = json.loads((REFERENCE_LEARNING / 'data/screen_reconstruction_manifest.json').read_text(encoding='utf-8'))
corpus = pd.read_parquet(RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/results/full_edge_response_corpus.parquet', columns=['reaction_type'])
e62 = pd.read_csv(RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results/e63_paired_component_summary.csv').iloc[0]
nest = [('corpus', '20 reaction tables in 01_原始数据', len(obs), obs.reaction_type.nunique()), ('corpus', 'records with a finite yield', int(np.isfinite(obs['yield']).sum()), None), ('RECIPE_OPTIONS', 'strictly eligible records (recipe matched, yield in [0,1])', recipe_options_map['strict_rows'], None), ('RESPONSE_LEARNING', 'eligible for exact matching (complete, consistent recipe)', int(obs.eligible_for_matching.sum()), None), ('RESPONSE_LEARNING', 'canonical yield-difference records over 2,135 exact swaps', len(corpus), corpus.reaction_type.nunique()), ('RESPONSE_LEARNING', 'complete response vectors for the structure model', int(e62.n_test_vectors), int(e62.n_test_classes)), ('RESPONSE_LEARNING', 'local-update queries (common8)', int(joint.n_test_vectors), int(joint.n_test_classes)), ('REFERENCE_LEARNING', 'analysis-partition wells in 19 qualified screens (Finkelstein excluded)', reference_learning_manifest['analysis_wells'], reference_learning_manifest['qualifying_screens']), ('REFERENCE_LEARNING', 'substrate-pair rows in those screens', reference_learning_manifest['analysis_pairs'], reference_learning_manifest['qualifying_screens'])]
for label, n in [('raw_rows', len(obs)), ('finite_yields', int(np.isfinite(obs['yield']).sum()))]:
    if reference_learning_manifest[label] != n or (label == 'raw_rows' and recipe_options_map['raw_rows'] != n):
        failures.append(f'corpus {label}: REFERENCE_LEARNING/RECIPE_OPTIONS pipelines disagree with the shared observations table')
for (_, support, count, classes), shown in zip(nest, ['410587', '381688', '380910', '381686', '828658', '25956', '24221', '196491', '2454']):
    rounded_pin('corpus', support, count, shown)
for index, shown in [(0, '20'), (4, '20'), (5, '14'), (6, '14'), (7, '19'), (8, '19')]:
    rounded_pin('corpus', nest[index][1] + ' classes or screens', nest[index][3], shown)
arrays = obs[obs.reaction_type.ne('finkelstein')]
rounded_pin('corpus', '19 arrays', len(arrays), '406909')
rounded_pin('corpus', '19 arrays count', arrays.reaction_type.nunique(), '19')
rounded_pin('corpus', 'coupling-array records', len(arrays), '406909')
rounded_pin('corpus', 'coupling-array count', arrays.reaction_type.nunique(), '19')
from m1_update_companions import build as update_companions, GRID
steps = update_companions()
rounded_pin('protocol', 'illustrative update displacement fraction', GRID[1], '0.25')
step_baselines = {('centered_unit_rms_zeroembed', 'zero_local_to_full', 'pooled'): ('0.0009781', '0.0031952', '-0.0002893', '0.0012391'), ('centered_unit_rms_zeroembed', 'zero_local_to_full', 'equal_edge'): ('0.0032876', '0.0101741', '-0.0010079', '0.0035989'), ('centered_unit_rms_zeroembed', 'zero_local_to_full', 'equal_class_edge_balanced'): ('0.0085996', '0.0187770', '-0.0031262', '0.0015779'), ('raw_delta', 'zero_local_to_level_only', 'pooled'): ('0.8263729', '1.4446278', '-0.3228972', '-0.2081180'), ('raw_delta', 'zero_local_to_level_only', 'equal_edge'): ('1.5855737', '2.6209657', '-0.6289765', '-0.5501817'), ('raw_delta', 'zero_local_to_level_only', 'equal_class_edge_balanced'): ('3.0129060', '3.9798837', '-1.2577102', '-2.0459282'), ('raw_delta', 'zero_local_to_modulation_only', 'pooled'): ('0.1692277', '0.4799933', '-0.0546143', '0.1415379'), ('raw_delta', 'zero_local_to_modulation_only', 'equal_edge'): ('0.4113121', '1.3241650', '-0.1228957', '0.5015408'), ('raw_delta', 'zero_local_to_modulation_only', 'equal_class_edge_balanced'): ('1.0598246', '2.3536840', '-0.3828070', '0.2340349')}
for (geometry, path, weighting), expected in step_baselines.items():
    selected = one(steps, target_geometry=geometry, path=path, weighting=weighting, allocation='mean_of_orders', stratum='all', risk_component='native')
    for column, shown in zip(['A', 'B', 'change_t025', 'change_t1'], expected):
        rounded_pin('audit' if path == 'zero_local_to_modulation_only' else 'SI', f'update path {geometry} {path} {weighting} {column}', selected[column], shown, 'out/m1_update_step_summary.csv', 'mean_of_orders; all; native')
reconciliation_path = OUT / 'corpus_numeric_pins.json'
reconciliation = json.loads(reconciliation_path.read_text(encoding='utf-8'))
if not isinstance(reconciliation, list) or not reconciliation:
    raise ValueError('Corpus reconciliation pins must be a nonempty list')
if len({entry['item'] for entry in reconciliation}) != len(reconciliation):
    raise ValueError('Duplicate corpus reconciliation pin identifiers')
for entry in reconciliation:
    rounded_pin('reconciliation', entry['item'], entry['actual'], entry['shown'], entry['source'], entry['selector'])
from publication_figures.direction_accuracy import ensure as ensure_direction_predictions
from publication_figures.display_pins import apply as pin_final_displays
ensure_direction_predictions()
pin_final_displays(rounded_pin)
transformation_codes = arrays.reaction_type.replace({'buchwald-hartwig-cmc': 'buchwald-hartwig', 'suzuki-bpin': 'suzuki'})
rounded_pin('corpus', 'coupling-transformation count', transformation_codes.nunique(), '17', 'scripts/response_learning/Existing_Data_Mining_2026-10-16/processed/observations.parquet', 'non-Finkelstein arrays; BH(CMC)=BH; Suzuki Bpin=Suzuki; Ullmann and C-O competition separate')
OUT.mkdir(exist_ok=True)
pd.DataFrame(nest, columns=['analysis', 'support', 'n_records', 'n_classes_or_screens']).to_csv(OUT / 'corpus_nesting.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'm1_numbers.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(nest).to_string(index=False))
if failures:
    raise SystemExit('MISMATCH:\n' + '\n'.join(failures))
print(f'{len(rows)} numbers match the approved pins')
