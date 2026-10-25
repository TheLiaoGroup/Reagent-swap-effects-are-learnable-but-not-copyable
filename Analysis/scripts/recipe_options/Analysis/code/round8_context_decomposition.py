"""Decompose every observed exact-edit/partner-pair 16-cell yield cube.

Run with the project Python runtime. Inputs are read-only round-7 tables;
all outputs are confined to Analysis/round8/context_decomposition."""
from __future__ import annotations
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'round7/partner_conditioned_edits'
OUT = ROOT / 'round8/context_decomposition'
CORNERS = ['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']
CORNER_LABELS = ['00', '10', '01', '11']
TARGET_KEYS = ['target_row_id_a_partner0', 'target_row_id_b_partner0', 'target_row_id_a_partner1', 'target_row_id_b_partner1']
TARGET_LABELS = ['a_C', 'b_C', 'a_D', 'b_D']
FACTORS = ['E', 'P', 'X', 'Y']
TERMS = [()] + [term for order in range(1, 5) for term in combinations(range(4), order)]
TERM_NAMES = ['intercept' if not term else ''.join((FACTORS[i] for i in term)) for term in TERMS]
ORDER = np.array([len(term) for term in TERMS])
LEVELS = np.array([[e, p, x, y] for p in (0, 1) for e in (0, 1) for y in (0, 1) for x in (0, 1)])
CODED = 2 * LEVELS - 1
H = np.column_stack([np.prod(CODED[:, term], axis=1) if term else np.ones(16) for term in TERMS])
DD_SIGNS = np.array([1.0, -1.0, -1.0, 1.0])
EPS = 1e-10
MODEL_ORDER = {'additive': 1, 'pairwise': 2, 'threeway': 3}
RECIPE_FIELDS = ['rectangle_id', 'representation_id', 'dataset_id', 'rxn_class', 'exchange_id', 'focal_slot', 'context_slot', 'component0_name', 'component1_name', 'component0_dose', 'component1_dose', 'context0_name', 'context1_name', 'context0_token', 'context1_token', 'fixed_background_json', 'recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1', 'full_recipe_c0_bg0', 'full_recipe_c1_bg0', 'full_recipe_c0_bg1', 'full_recipe_c1_bg1']
RELATION_FIELDS = ['edit_id', 'structural_molecule_a', 'structural_molecule_b', 'molecule_name_a', 'molecule_name_b', 'exact_graph_method', 'structural_change_a_to_b', 'exact_common_core', 'exact_fragment_a', 'exact_fragment_b']

def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 1} if name.endswith('.gz') else None)

def sid(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()[:20]

def matching(cubes, gap_pp, high_floor_pct):
    """Require both cells of one diagonal to exceed both others by gap."""
    diagonal0 = cubes[:, :, [0, 3]]
    diagonal1 = cubes[:, :, [1, 2]]
    m0 = diagonal0.min(2) - diagonal1.max(2)
    m1 = diagonal1.min(2) - diagonal0.max(2)
    gap0 = m0 > EPS if gap_pp == 0 else m0 >= gap_pp - EPS
    gap1 = m1 > EPS if gap_pp == 0 else m1 >= gap_pp - EPS
    return np.where(gap0 & (diagonal0.min(2) >= high_floor_pct - EPS), 1, np.where(gap1 & (diagonal1.min(2) >= high_floor_pct - EPS), -1, 0))

def checkerboard(direction):
    return (direction[:, 0] != 0) & (direction[:, 0] == -direction[:, 1]) & (direction[:, 0] == -direction[:, 2]) & (direction[:, 0] == direction[:, 3])

def transition(direction):
    a, b = (direction[:, 0], direction[:, 1])
    return np.select([(a == 1) & (b == -1), (a == -1) & (b == 1), (a == 0) & (b == 1), (a == 0) & (b == -1), (a == 1) & (b == 0), (a == -1) & (b == 0), (a == 1) & (b == 1), (a == -1) & (b == -1)], ['reversal_00_11_to_10_01', 'reversal_10_01_to_00_11', 'established_00_11', 'established_10_01', 'eliminated_00_11', 'eliminated_10_01', 'retained_00_11', 'retained_10_01'], default='neither_strong_matching')

def self_checks():
    assert H.shape == (16, 16) and np.allclose(H.T @ H, 16 * np.eye(16))
    pure = 50 + 12 * H[:, -1]
    beta = pure @ H / 16
    assert np.isclose(beta[-1], 12) and np.isclose(np.abs(pure @ H[:, -1]), 192)
    assert np.allclose(pure - beta[ORDER <= 3] @ H[:, ORDER <= 3].T, 12 * H[:, -1])
    additive = 40 + 3 * CODED[:, 0] - 4 * CODED[:, 3]
    assert np.allclose((additive @ H / 16)[ORDER > 1], 0)
    test = np.array([[[80.0, 10.0, 10.0, 80.0], [10.0, 80.0, 80.0, 10.0], [10.0, 80.0, 80.0, 10.0], [80.0, 10.0, 10.0, 80.0]]])
    assert checkerboard(matching(test, 10, 20)).item()
    assert not checkerboard(matching(test, 75, 20)).item()
    assert not checkerboard(matching(np.ones((1, 4, 4)) * 50, 0, 0)).item()

def load():
    pairs = read(SOURCE / 'exact_edit_all_partner_pair_comparisons.csv.gz')
    targets = read(SOURCE / 'all_primary_target_four_yields.csv.gz')
    rectangles = read(SOURCE / 'all_primary_recipe_rectangles.csv', usecols=RECIPE_FIELDS)
    relations = read(SOURCE / 'molecular_relation_registry.csv.gz')
    assert len(pairs) == 204057 and pairs.partner_pair_comparison_id.is_unique
    assert len(targets) == 176321 and targets.target_row_id.is_unique
    assert len(rectangles) == 825 and rectangles.rectangle_id.is_unique
    assert pairs.edit_rectangle_id.nunique() == 15285
    source = targets.set_index('target_row_id')
    cubes = np.stack([source.loc[pairs[key], CORNERS].to_numpy(float) for key in TARGET_KEYS], axis=1) * 100
    for key in TARGET_KEYS:
        assert np.array_equal(source.loc[pairs[key], 'rectangle_id'], pairs.rectangle_id)
        assert np.array_equal(source.loc[pairs[key], 'dataset_id'], pairs.dataset_id)
    assert np.isfinite(cubes).all()
    registry = relations.set_index('edit_id')
    assert registry.loc[pairs.edit_id, 'has_existing_exact_neighbor_rule'].all()
    for partner_index, partner_name in enumerate(['partner0_smiles', 'partner1_smiles']):
        for edit_index, edit_name in enumerate(['structural_molecule_a', 'structural_molecule_b']):
            local = source.loc[pairs[TARGET_KEYS[2 * partner_index + edit_index]]]
            role1 = pairs.role.eq('sub_1_smiles').to_numpy()
            focal = np.where(role1, local.sub_1_smiles, local.sub_2_smiles)
            partner = np.where(role1, local.sub_2_smiles, local.sub_1_smiles)
            assert np.array_equal(focal, registry.loc[pairs.edit_id, edit_name])
            assert np.array_equal(partner, pairs[partner_name])
    return (pairs, targets, rectangles, relations, cubes)

def decompose(pairs, cubes):
    yields = cubes.reshape(len(cubes), 16)
    beta = yields @ H / 16
    assert np.allclose(beta @ H.T, yields, atol=EPS)
    result = pairs.copy()
    result['cube_id'] = [sid(sorted(map(int, row))) for row in pairs[TARGET_KEYS].to_numpy()]
    result['n_views_of_same_cube'] = result.groupby('cube_id').cube_id.transform('size')
    columns = {}
    for t, label in enumerate(TARGET_LABELS):
        for c, corner in enumerate(CORNER_LABELS):
            columns[f'yield_{label}_{corner}_pct'] = cubes[:, t, c]
        columns[f'DD_{label}_pp'] = cubes[:, t] @ DD_SIGNS
        columns[f'margin_00_11_{label}_pp'] = cubes[:, t, [0, 3]].min(1) - cubes[:, t, [1, 2]].max(1)
        columns[f'margin_10_01_{label}_pp'] = cubes[:, t, [1, 2]].min(1) - cubes[:, t, [0, 3]].max(1)
    for j, name in enumerate(TERM_NAMES):
        columns['coefficient_' + name + '_pp'] = beta[:, j]
    columns['fourth_contrast_pp'] = yields @ H[:, -1]
    columns['fourth_component_amplitude_pp'] = np.abs(beta[:, -1])
    centered_ms = np.square(beta[:, 1:]).sum(1)
    for name, order in MODEL_ORDER.items():
        residual = yields - beta[:, ORDER <= order] @ H[:, ORDER <= order].T
        mse = np.square(residual).mean(1)
        assert np.allclose(mse, np.square(beta[:, ORDER > order]).sum(1), atol=EPS)
        columns[f'{name}_rmse_pp'] = np.sqrt(mse)
        columns[f'{name}_mae_pp'] = np.abs(residual).mean(1)
        columns[f'{name}_max_abs_error_pp'] = np.abs(residual).max(1)
        columns[f'{name}_unexplained_fraction_of_centered_ss'] = np.divide(mse, centered_ms, out=np.zeros_like(mse), where=centered_ms > EPS)
        predictions = yields - residual
        columns[f'{name}_prediction_outside_0_100'] = ((predictions < -EPS) | (predictions > 100 + EPS)).any(1)
    assert np.allclose(columns['threeway_rmse_pp'], columns['fourth_component_amplitude_pp'], atol=EPS)
    assert np.allclose(columns['threeway_max_abs_error_pp'], columns['fourth_component_amplitude_pp'], atol=EPS)
    assert np.allclose(columns['fourth_contrast_pp'], 100 * pairs.delta_DD_partner1_minus_partner0, atol=EPS)
    for gap in [0, 1, 2, 5, 10, 15, 20]:
        direction = matching(cubes, gap, 20)
        columns[f'full_checkerboard_gap{gap}pp_high20'] = checkerboard(direction)
        if gap in [5, 10]:
            for j, label in enumerate(TARGET_LABELS):
                columns[f'direction_{label}_{gap}pp_high20'] = direction[:, j]
            for j, name in enumerate(['C', 'D']):
                computed = transition(direction[:, [2 * j, 2 * j + 1]])
                assert np.array_equal(computed, pairs[f'transition_partner{j}_{gap}pp'])
                columns[f'transition_{name}_{gap}pp'] = computed
            columns[f'at_least_one_actual_edit_reversal_{gap}pp'] = (direction[:, 0] * direction[:, 1] == -1) | (direction[:, 2] * direction[:, 3] == -1)
    result = pd.concat([result, pd.DataFrame(columns, index=result.index)], axis=1)
    for gap in [5, 10]:
        assert np.array_equal(result[f'full_checkerboard_gap{gap}pp_high20'], pairs[f'opposite_matching_reversal_{gap}pp'])
    save(result, 'all_194533_observed_cube_decompositions.csv.gz')
    save(rectangle_metadata(), 'factorial_design_and_terms.csv')
    return (result, beta)

def rectangle_metadata():
    rows = []
    for cell, (e, p, x, y) in enumerate(LEVELS):
        row = dict(cell_index=cell, edit_level='A' if e == 0 else 'B', partner_level='C' if p == 0 else 'D', X=x, Y=y, corner=f'{x}{y}', target_label=TARGET_LABELS[2 * p + e])
        row.update({'sign_' + name: H[cell, j] for j, name in enumerate(TERM_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)

def count_unit(frame, mask):
    selected = frame[mask]
    return dict(n_comparison_views=len(selected), n_unique_cubes=selected.cube_id.nunique(), n_edit_rectangle_groups=selected.edit_rectangle_id.nunique(), n_edits=selected.edit_id.nunique(), n_datasets=selected.dataset_id.nunique(), n_classes=selected.rxn_class.nunique())

def summaries(frame, cubes):
    rows = []
    for grouping in ['all', 'unique_cube', 'dataset_id', 'rxn_class', 'partner_has_exact_rule']:
        if grouping == 'all':
            groups = [('all', frame)]
        elif grouping == 'unique_cube':
            groups = [('all_unique_cubes', frame.drop_duplicates('cube_id'))]
        else:
            groups = frame.groupby(grouping, sort=True)
        for label, part in groups:
            for model in MODEL_ORDER:
                errors = part[f'{model}_rmse_pp']
                rows.append(dict(grouping=grouping, group=str(label), model=model, n_comparison_views=len(part), n_unique_cubes=part.cube_id.nunique(), n_exact_zero_error=int(errors.le(EPS).sum()), q25_rmse_pp=errors.quantile(0.25), median_rmse_pp=errors.median(), q75_rmse_pp=errors.quantile(0.75), q90_rmse_pp=errors.quantile(0.9), q95_rmse_pp=errors.quantile(0.95), q99_rmse_pp=errors.quantile(0.99), maximum_rmse_pp=errors.max(), median_max_abs_error_pp=part[f'{model}_max_abs_error_pp'].median(), maximum_max_abs_error_pp=part[f'{model}_max_abs_error_pp'].max(), pooled_observed_cell_rmse_pp=np.sqrt(np.square(errors).mean()), median_unexplained_centered_ss=part[f'{model}_unexplained_fraction_of_centered_ss'].median(), n_prediction_outside_0_100=int(part[f'{model}_prediction_outside_0_100'].sum())))
    save(pd.DataFrame(rows), 'model_reconstruction_error_summary.csv')
    sensitivity = []
    for model in MODEL_ORDER:
        for threshold in [0, 1, 2, 5, 10, 15, 20]:
            errors = frame[f'{model}_rmse_pp']
            mask = errors > EPS if threshold == 0 else errors >= threshold - EPS
            sensitivity.append(dict(model=model, threshold_rmse_pp=threshold, denominator_views=len(frame), denominator_unique_cubes=frame.cube_id.nunique(), **count_unit(frame, mask), n_non_event_views=int((~mask).sum()), n_with_actual_edit_reversal5=int(frame.loc[mask, 'at_least_one_actual_edit_reversal_5pp'].sum()), n_with_full_checkerboard5=int(frame.loc[mask, 'full_checkerboard_gap5pp_high20'].sum())))
    save(pd.DataFrame(sensitivity), 'reconstruction_error_threshold_sensitivity.csv')
    sensitivity = []
    for gap in [0, 1, 2, 5, 10, 15, 20]:
        for high in [0, 10, 20, 30, 50]:
            mask = checkerboard(matching(cubes, gap, high))
            sensitivity.append(dict(diagonal_gap_pp=gap, high_diagonal_min_pct=high, denominator_views=len(frame), denominator_unique_cubes=frame.cube_id.nunique(), n_non_event_views=int((~mask).sum()), **count_unit(frame, mask)))
    save(pd.DataFrame(sensitivity), 'actual_checkerboard_threshold_sensitivity.csv')
    transition_groups = frame.groupby(['transition_C_5pp', 'transition_D_5pp'], sort=True)
    transition_rows = []
    for (left, right), part in transition_groups:
        transition_rows.append(dict(transition_C_5pp=left, transition_D_5pp=right, n_comparison_views=len(part), n_unique_cubes=part.cube_id.nunique(), n_fourth_error_ge5pp=int(part.threeway_rmse_pp.ge(5 - EPS).sum()), maximum_fourth_error_pp=part.threeway_rmse_pp.max(), median_fourth_error_pp=part.threeway_rmse_pp.median()))
    save(pd.DataFrame(transition_rows), 'actual_transition_pair_complete_counts.csv')
    groups = read(SOURCE / 'exact_edit_rectangle_partner_summary.csv.gz')
    aggregate = frame.groupby('edit_rectangle_id', sort=False).agg(n_partner_pair_comparisons=('cube_id', 'size'), n_unique_cubes=('cube_id', 'nunique'), max_fourth_error_pp=('threeway_rmse_pp', 'max'), median_fourth_error_pp=('threeway_rmse_pp', 'median'), n_actual_checkerboards5=('full_checkerboard_gap5pp_high20', 'sum'), max_pairwise_rmse_pp=('pairwise_rmse_pp', 'max'))
    groups = groups.merge(aggregate, on='edit_rectangle_id', how='left', validate='one_to_one')
    for key in ['n_partner_pair_comparisons', 'n_unique_cubes', 'n_actual_checkerboards5']:
        groups[key] = pd.to_numeric(groups[key], errors='coerce').fillna(0).astype(int)
    assert np.array_equal(groups.n_partner_pair_comparisons, groups.n_common_partners * (groups.n_common_partners - 1) // 2)
    save(groups, 'all_exact_edit_rectangle_scope_including_single_partners.csv.gz')
    mapping = read(ROOT / 'round3/global_scope/dataset_reaction_class_mapping.csv')
    old = read(SOURCE / 'dataset_role_full_coverage.csv')
    coverage = old.groupby('dataset_id', as_index=False).agg(n_legacy_swaps=('n_swaps', 'max'), n_primary_rectangles=('n_primary_rectangles', 'max'), n_exact_partner_observations=('exact_n_partner_observations', 'sum'), n_exact_edit_rectangles=('exact_n_edit_rectangle_groups', 'sum'), n_single_partner_edit_rectangles=('exact_n_groups_single_partner', 'sum'))
    mapping = mapping.merge(coverage, on='dataset_id', how='left', validate='one_to_one')
    counts = frame.groupby('dataset_id').agg(n_comparison_views=('cube_id', 'size'), n_unique_cubes=('cube_id', 'nunique'), n_fourth_error_nonzero=('threeway_rmse_pp', lambda x: int(x.gt(EPS).sum())), n_fourth_error_ge5pp=('threeway_rmse_pp', lambda x: int(x.ge(5 - EPS).sum())), n_fourth_error_ge10pp=('threeway_rmse_pp', lambda x: int(x.ge(10 - EPS).sum())), n_full_checkerboard5=('full_checkerboard_gap5pp_high20', 'sum'))
    mapping = mapping.merge(counts, on='dataset_id', how='left', validate='one_to_one')
    for name in counts.columns:
        mapping[name] = mapping[name].fillna(0).astype(int)
    mapping['decomposition_status'] = np.where(mapping.n_comparison_views.gt(0), 'observed_complete_16_cell_comparisons', 'no_observed_exact_edit_with_two_partners_in_primary_rectangles')
    assert len(mapping) == 20 and mapping.rxn_class.nunique() == 18 and (mapping.n_legacy_swaps.sum() == 4002)
    save(mapping, 'all_20_dataset_scope_and_non_events.csv')
    return (pd.DataFrame(rows), pd.DataFrame(sensitivity), mapping)

def select_examples(frame, targets, rectangles, relations, beta):
    priority = read(SOURCE / 'priority_seeded_exact_edits_all_partner_summary.csv')
    r7_display = read(SOURCE / 'selected_complete_local_edit_profiles.csv')
    frame['was_round7_priority_edit_rectangle'] = frame.edit_rectangle_id.isin(priority.edit_rectangle_id)
    frame['was_round7_selected_complete_profile'] = frame.edit_rectangle_id.isin(r7_display.edit_rectangle_id)
    ranked = frame.sort_values('threeway_rmse_pp', ascending=False)
    selected = pd.concat([ranked.groupby('rxn_class', sort=False).head(1), ranked[~ranked.was_round7_priority_edit_rectangle & ~ranked.was_round7_selected_complete_profile].groupby('rxn_class', sort=False).head(1), frame[frame.full_checkerboard_gap5pp_high20]], ignore_index=False)
    selected = selected.drop_duplicates('partner_pair_comparison_id').copy()
    selected['selection_reason'] = np.where(selected.full_checkerboard_gap5pp_high20, 'actual_opposite_strong_matching_reversals_at_both_partners', 'largest_irreducible_threeway_yield_error_in_class_or_new_full_enumeration_profile')
    selected['original_frame_index'] = selected.index
    selected = selected.merge(relations[RELATION_FIELDS], on='edit_id', validate='many_to_one')
    selected = selected.merge(rectangles.drop(columns=['dataset_id', 'rxn_class']), on='rectangle_id', validate='many_to_one')
    save(selected, 'selected_chemical_examples.csv')
    long = []
    complete_profiles = []
    profile_ids = set()
    target_lookup = targets.set_index('target_row_id')
    for row in selected.itertuples(index=False):
        j = row.original_frame_index
        predicted = {model: beta[j, ORDER <= order] @ H[:, ORDER <= order].T for model, order in MODEL_ORDER.items()}
        for cell, (e, p, x, y) in enumerate(LEVELS):
            target_id = getattr(row, TARGET_KEYS[2 * p + e])
            record = dict(partner_pair_comparison_id=row.partner_pair_comparison_id, cube_id=row.cube_id, edit_rectangle_id=row.edit_rectangle_id, rectangle_id=row.rectangle_id, dataset_id=row.dataset_id, rxn_class=row.rxn_class, target_row_id=target_id, edit_level='A' if e == 0 else 'B', partner_level='C' if p == 0 else 'D', X=x, Y=y, corner=f'{x}{y}', source_corner=CORNERS[2 * y + x], observed_yield_pct=100 * target_lookup.loc[target_id, CORNERS[2 * y + x]], n_complete_acquisition_blocks=int(target_lookup.loc[target_id, 'n_complete_acquisition_blocks']), recipe_id=getattr(row, ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1'][2 * y + x]))
            for model in MODEL_ORDER:
                record[model + '_fitted_yield_pct'] = predicted[model][cell]
                record[model + '_observed_minus_fitted_pp'] = record['observed_yield_pct'] - predicted[model][cell]
            long.append(record)
        if row.edit_rectangle_id in profile_ids:
            continue
        profile_ids.add(row.edit_rectangle_id)
        partner_role = 'sub_2_smiles' if row.role == 'sub_1_smiles' else 'sub_1_smiles'
        local = targets[targets.rectangle_id.eq(row.rectangle_id)]
        a = local[local[row.role].eq(row.structural_molecule_a)].set_index(partner_role)
        b = local[local[row.role].eq(row.structural_molecule_b)].set_index(partner_role)
        partners = sorted(set(a.index) & set(b.index) - {''})
        for partner in partners:
            entry = dict(edit_rectangle_id=row.edit_rectangle_id, edit_id=row.edit_id, rectangle_id=row.rectangle_id, dataset_id=row.dataset_id, rxn_class=row.rxn_class, role=row.role, partner_smiles=partner, n_all_common_partners=len(partners), target_row_id_a=int(a.loc[partner, 'target_row_id']), target_row_id_b=int(b.loc[partner, 'target_row_id']))
            values = np.stack([a.loc[partner, CORNERS].to_numpy(float), b.loc[partner, CORNERS].to_numpy(float)]) * 100
            for s, label in enumerate(['a', 'b']):
                entry.update({f'yield_{label}_{corner}_pct': values[s, c] for c, corner in enumerate(CORNER_LABELS)})
                entry['DD_' + label + '_pp'] = values[s] @ DD_SIGNS
            entry['delta_DD_pp'] = entry['DD_b_pp'] - entry['DD_a_pp']
            entry['transition_5pp'] = transition(matching(values[None, :, :], 5, 20))[0]
            complete_profiles.append(entry)
    long = pd.DataFrame(long)
    save(long, 'selected_actual_yields_and_lower_order_reconstructions.csv')
    save(pd.DataFrame(complete_profiles), 'selected_edit_rectangle_all_common_partners.csv')
    source_links = read(SOURCE / 'all_four_cell_original_record_links.csv.gz')
    used_targets = set(frame[TARGET_KEYS].to_numpy().ravel())
    links = source_links[source_links.target_row_id.isin(used_targets)].copy()
    save(links, 'all_participating_target_original_record_links.csv.gz')
    selected_targets = set(long.target_row_id)
    save(links[links.target_row_id.isin(selected_targets)], 'selected_example_original_record_links.csv.gz')
    block = read(SOURCE / 'all_complete_target_acquisition_blocks.csv.gz')
    rebuilt = block.groupby('target_row_id')[CORNERS].mean().loc[sorted(used_targets)]
    assert np.allclose(rebuilt, target_lookup.loc[rebuilt.index, CORNERS], atol=EPS)
    assert links.target_row_id.nunique() == len(used_targets)
    assert set(zip(long.target_row_id, long.source_corner)).issubset(set(zip(links.target_row_id, links.corner)))
    cell_keys = ['target_row_id', 'archive', 'plate', 'acquisition', 'corner']
    original_cell_means = links.groupby(cell_keys).yield_numeric.transform('mean')
    assert np.allclose(original_cell_means, links.matched_cell_yield, atol=EPS)
    original_target_means = links.drop_duplicates(cell_keys).groupby(['target_row_id', 'corner']).matched_cell_yield.mean().unstack()
    assert np.allclose(original_target_means[CORNERS], target_lookup.loc[original_target_means.index, CORNERS], atol=EPS)
    return (selected, dict(n_participating_target_rectangle_rows=len(used_targets), n_original_cell_record_links=len(links), n_distinct_original_records=links.record_id.nunique(), n_selected_examples=len(selected), n_selected_complete_profiles=len(profile_ids), n_selected_all_partner_observations=len(complete_profiles)))

def report(frame, selected, error_summary, mapping, source_counts, seconds):
    top = selected.sort_values('threeway_rmse_pp', ascending=False)
    actual = frame[frame.full_checkerboard_gap5pp_high20]
    scope = json.loads((SOURCE / 'initial_scale.json').read_text(encoding='utf-8'))
    counts = {f'fourth_error_ge{threshold}pp': count_unit(frame, frame.threeway_rmse_pp.ge(threshold - EPS)) for threshold in [1, 2, 5, 10, 15, 20]}
    summary = dict(study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, n_comparison_views=len(frame), n_unique_cubes=frame.cube_id.nunique(), n_duplicate_views=len(frame) - frame.cube_id.nunique(), n_edit_rectangle_groups=frame.edit_rectangle_id.nunique(), n_observed_edits=frame.edit_id.nunique(), n_classes=frame.rxn_class.nunique(), n_source_exact_registered_edits=scope['n_exact_structural_relations'], n_source_exact_edit_rectangles=scope['exact_edit_rectangle_groups'], n_source_single_partner_exact_edit_rectangles=scope['exact_edit_rectangle_groups'] - scope['exact_groups_ge2_partners'], n_fourth_error_zero=int(frame.threeway_rmse_pp.le(EPS).sum()), n_fourth_error_nonzero=int(frame.threeway_rmse_pp.gt(EPS).sum()), median_fourth_error_pp=frame.threeway_rmse_pp.median(), q99_fourth_error_pp=frame.threeway_rmse_pp.quantile(0.99), maximum_fourth_error_pp=frame.threeway_rmse_pp.max(), full_checkerboard5=count_unit(frame, frame.full_checkerboard_gap5pp_high20), full_checkerboard10=count_unit(frame, frame.full_checkerboard_gap10pp_high20), **counts, **source_counts, elapsed_seconds=seconds, status='complete')
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Four factors in sixteen observed yields: chemical context and lower-order reconstruction', '', f"A fluorine added to 4-methylbenzoic acid changes the DCC/DIC x coupling-additive yield pattern differently for two anilines. Across the complete observed comparison set, this is the largest departure from every model containing only one-, two-, and three-factor terms: the best reconstruction misses each of its sixteen measured yields by {summary['maximum_fourth_error_pp']:.2f} percentage points. The actual patterns are loss of opposite preferred pairings in the two partners, rather than two opposite edit-induced flips.", '', '## Complete comparison scope', '', f'The parent study remains **20 datasets, 18 reaction classes, 4,002 background-specific single-slot exchanges**. This analysis starts from all **{len(frame):,}** observed exact-edit x fixed-rectangle x partner-pair comparisons, not from selected reversal seeds. These comparisons contain {frame.cube_id.nunique():,} unique molecular-quartet/rectangle cubes, {frame.edit_rectangle_id.nunique():,} exact-edit x rectangle groups, {frame.edit_id.nunique()} focal edits, and {frame.rxn_class.nunique()} reaction classes. The focal edit follows the existing exact-graph/MMP rule registry; partners need not themselves be nearest neighbors.', '', f"The source contains {scope['exact_edit_rectangle_groups']:,} exact-edit x rectangle groups and {scope['exact_partner_observations']:,} fixed-partner A/B observations. The {scope['exact_edit_rectangle_groups'] - scope['exact_groups_ge2_partners']:,} groups with only one common partner remain in the scope table but have no complete four-factor cube. Dataset rows with no eligible comparison remain in `all_20_dataset_scope_and_non_events.csv`. An event count of zero and absence of an eligible cube are separately labelled.", '', 'All sixteen cells are observed target-level yields. Each four-cell target is assembled within dataset/archive/plate/acquisition/target before averaging complete acquisition blocks. Different targets need not occupy the same acquisition block. Missing yields are never filled with zero. All four exact recipe IDs, structural identities, original record IDs, source paths, sheets and rows are retained. Shared molecules, recipes, cells and role views are not independent experiments; counts and reconstruction errors are descriptive, without artificial p-values.', '', '## What the factorial quantities mean', '', 'E denotes A/B of the focal edit, P denotes partners C/D, and X/Y are the two actual component choices of the fixed recipe rectangle. Code each level as -1/+1. The saturated expression is y = sum_S beta_S product_{j in S}(s_j), over the 16 subsets of E,P,X,Y, where beta_S = (1/16) sum_cells y product(s_j). Yields and coefficients here use percentage-point units.', '', 'For one target, DD = y00 - y10 - y01 + y11. Define deltaDD_C = DD_BC - DD_AC and deltaDD_D = DD_BD - DD_AD. The fourth contrast is C4 = deltaDD_D - deltaDD_C = 16 beta_EPXY. A large C4 describes a difference of changes in component interaction; it is neither a yield gain nor a mechanistic assignment. Its sign depends on the declared factor ordering.', '', 'The additive fit retains intercept and four main effects (5 parameters); the pairwise fit retains all terms through order two (11); the three-way fit retains all terms through order three (15). These are least-squares projections of the same sixteen observed cells, not predictions on held-out data. The three-way residual is exactly beta_EPXY s_E s_P s_X s_Y. Thus **abs(C4)/16 is the RMSE, MAE and maximum absolute per-cell error of the best three-way fit**. It is also the minimum attainable maximum error among all three-way functions: the alternating residual contrast forces at least one cell to miss by that amount. It is not a claim that every possible lower-order fit must miss every cell equally; the least-squares projection does.', '', 'For additive and pairwise fits, RMSE = sqrt(sum of squared omitted beta coefficients). These models may produce unconstrained fitted yields outside 0-100%; enforcing bounds cannot improve their least-squares minimum. A nonzero fourth coefficient proves exact algebraic non-representability on this cube. Experimental or rounding relevance requires the yield-scale error and sensitivity thresholds below, not mere nonzero arithmetic.', '', '## Yield-scale departures across the full denominator', '', '| Model | Median RMSE / pp | 95th percentile / pp | 99th percentile / pp | Maximum / pp | Exact-zero RMSE views |', '|---|---:|---:|---:|---:|---:|']
    for row in error_summary[error_summary.grouping.eq('all')].itertuples(index=False):
        lines.append(f'| {row.model} | {row.median_rmse_pp:.3f} | {row.q95_rmse_pp:.3f} | {row.q99_rmse_pp:.3f} | {row.maximum_rmse_pp:.3f} | {row.n_exact_zero_error:,} |')
    lines += ['', '| Minimum irreducible three-way error / pp | Comparison views | Unique cubes | Exact-edit x rectangle groups | Reaction classes |', '|---:|---:|---:|---:|---:|']
    for threshold, entry in [(t, counts[f'fourth_error_ge{t}pp']) for t in [1, 2, 5, 10, 15, 20]]:
        lines.append(f"| {threshold} | {entry['n_comparison_views']:,} | {entry['n_unique_cubes']:,} | {entry['n_edit_rectangle_groups']:,} | {entry['n_classes']} |")
    lines += ['', f"The median absolute fourth contrast is {16 * frame.threeway_rmse_pp.median():.4f} pp, corresponding to only {frame.threeway_rmse_pp.median():.4f} pp per observed yield. A 5 pp per-cell departure requires an 80 pp fourth contrast. There are {summary['n_fourth_error_zero']:,} zero-fourth-error views and {summary['n_fourth_error_nonzero']:,} nonzero views; all remain in the output. Reused cells and overlapping cubes mean these fractions characterize the observed comparison collection, not independent sampling frequencies.", '', '## Actual directional flips are a stricter outcome', '', 'A strong XY matching requires both cells of one diagonal to exceed both cells of the other diagonal by at least g pp, and both preferred-diagonal yields to be at least h%. A full checkerboard requires matching directions (AC, BC, AD, BD) = (+,-,-,+) or its inverse. This simultaneously reverses the actual four-corner pairing under the edit at both partners, in opposite directions. Large C4, opposite signed deltaDD, establishment, and elimination do not by themselves satisfy this condition.', '', f'At g=5 pp and h=20%, **{len(actual)} of {len(frame):,} views** satisfy the full checkerboard; at g=10 pp, **{int(frame.full_checkerboard_gap10pp_high20.sum())}** do. The 5 pp example is the previously displayed Suzuki methoxy-position domain and is labelled as such. This rarity is kept separate from the broader algebraic yield departures. `actual_checkerboard_threshold_sensitivity.csv` retains all 35 gap/floor combinations and their non-events.', '', '## Chemical cases selected from the full scan', '']
    wanted = [top.iloc[0].partner_pair_comparison_id]
    for reaction in ['SNAr', 'Suzuki', 'Kumada', 'Buchwald-Hartwig', 'Grignard']:
        local = top[top.rxn_class.eq(reaction)]
        if len(local):
            wanted.append(local.iloc[0].partner_pair_comparison_id)
    wanted += actual.partner_pair_comparison_id.to_list()
    for key in dict.fromkeys(wanted):
        row = selected[selected.partner_pair_comparison_id.eq(key)].iloc[0]
        lines += [f'### {row.rxn_class}: {row.structural_change_a_to_b}', '', f'Pair ID `{key}`; cube `{row.cube_id}`; edit `{row.edit_id}`; rectangle `{row.rectangle_id}`.', '', f'A `{row.structural_molecule_a}` -> B `{row.structural_molecule_b}`. C `{row.partner0_smiles}`; D `{row.partner1_smiles}`.', '', f'X0/X1: {row.component0_name} / {row.component1_name}; Y0/Y1: {row.context0_name} / {row.context1_name}.', '', '| Target | 00 / % | 10 / % | 01 / % | 11 / % | DD / pp |', '|---|---:|---:|---:|---:|---:|']
        for label, display in zip(TARGET_LABELS, ['A+C', 'B+C', 'A+D', 'B+D']):
            values = ' | '.join((f"{row[f'yield_{label}_{corner}_pct']:.2f}" for corner in CORNER_LABELS))
            lines.append(f"| {display} | {values} | {row[f'DD_{label}_pp']:.2f} |")
        lines += ['', f'C4 = {row.fourth_contrast_pp:.2f} pp; irreducible three-way per-cell error = **{row.threeway_rmse_pp:.2f} pp**; pairwise RMSE = {row.pairwise_rmse_pp:.2f} pp; additive RMSE = {row.additive_rmse_pp:.2f} pp. Actual 5 pp outcomes: C `{row.transition_C_5pp}`, D `{row.transition_D_5pp}`; full checkerboard `{bool(row.full_checkerboard_gap5pp_high20)}`.', '', f'Full recipe 00: {row.full_recipe_c0_bg0}. Full recipe 10: {row.full_recipe_c1_bg0}. Full recipe 01: {row.full_recipe_c0_bg1}. Full recipe 11: {row.full_recipe_c1_bg1}.', '']
    lines += ['## Reproducibility and interpretation', '', f"The strongest new result is a quantified yield departure that remains after fitting every lower-order contribution of both reactants and both recipe components. The largest acylation case gives {summary['maximum_fourth_error_pp']:.2f} pp per measured yield; other concrete transformations include aromatic CH/N replacement in SNAr, bromide relocation in Suzuki, methoxy relocation in Kumada and a Cl/F replacement in Grignard. These results identify specific observed recipe-response structures. They do not assign molecular mechanisms or establish prospective portability.", '', 'The fourth contrast alone is insufficient as a headline about component-pair reversals: its median yield-scale amplitude is small, and the full strong checkerboard is sparse. The substantial-error subset and the rare actual checkerboard answer different scientific questions. Complete profiles for the selected edits retain every common partner, including partners without strong pairing or reversal.', '', '- `all_194533_observed_cube_decompositions.csv.gz`: every comparison, all sixteen observed yields, sixteen factorial coefficients, full-model identity, lower-order reconstruction errors, matching margins, original target IDs and role-view multiplicity.', f"- `all_primary_recipe_rectangles.csv`: exact recipe IDs, labels, doses and fixed background for all {scope['primary_rectangles']} primary rectangles.", f"- `all_exact_edit_rectangle_scope_including_single_partners.csv.gz`: all {summary['n_source_exact_edit_rectangles']:,} exact-edit x rectangle groups, including single-partner non-comparability.", '- `all_20_dataset_scope_and_non_events.csv`: all 20 datasets, actual denominators and zeros.', '- `model_reconstruction_error_summary.csv` and `reconstruction_error_threshold_sensitivity.csv`: additive/pairwise/three-way model errors and threshold denominators.', '- `actual_checkerboard_threshold_sensitivity.csv` and `actual_transition_pair_complete_counts.csv`: actual directional outcomes, complete non-events and gap/floor sensitivity.', '- `selected_chemical_examples.csv`, `selected_actual_yields_and_lower_order_reconstructions.csv`, `selected_edit_rectangle_all_common_partners.csv`: concrete chemical identities, actual-versus-fitted sixteen cells and complete local partner profiles.', '- `all_participating_target_original_record_links.csv.gz` and `selected_example_original_record_links.csv.gz`: original record IDs and absolute file/sheet/row positions. These are normalized source observations, linked through target-row IDs; they are not new experiments.', '- `factorial_design_and_terms.csv`: exact level order and all orthogonal term signs.', '- `run_summary.json` and `verification.json`: derived counts and read-back verification.', '', 'Generator: `Analysis/code/round8_context_decomposition.py`. All pandas CSV reads use `keep_default_na=False, low_memory=False`. The full comparison table comes directly from the round-7 exhaustive exact-edit partner-pair enumeration; no seed filter is applied.', '']
    (OUT / 'CONTEXT_DECOMPOSITION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary

def verify(frame, beta, cubes):
    reread = read(OUT / 'all_194533_observed_cube_decompositions.csv.gz')
    assert len(reread) == 204057 and reread.partner_pair_comparison_id.is_unique
    assert np.allclose(reread.fourth_contrast_pp / 16, reread.coefficient_EPXY_pp, atol=EPS)
    assert np.allclose(reread.threeway_rmse_pp, reread.fourth_contrast_pp.abs() / 16, atol=EPS)
    selected_long = read(OUT / 'selected_actual_yields_and_lower_order_reconstructions.csv')
    direct_lstsq_max_error = 0.0
    for _, part in selected_long.groupby('partner_pair_comparison_id', sort=False):
        signs = np.column_stack([part.edit_level.map({'A': -1.0, 'B': 1.0}), part.partner_level.map({'C': -1.0, 'D': 1.0}), 2 * part.X - 1, 2 * part.Y - 1])
        design = np.column_stack([np.ones(16)] + [np.prod(signs[:, term], axis=1) for order in [1, 2, 3] for term in combinations(range(4), order)])
        direct_fit = design @ np.linalg.lstsq(design, part.observed_yield_pct, rcond=None)[0]
        direct_lstsq_max_error = max(direct_lstsq_max_error, np.abs(direct_fit - part.threeway_fitted_yield_pct).max())
    assert direct_lstsq_max_error < EPS
    for model in MODEL_ORDER:
        recovered = selected_long.groupby('partner_pair_comparison_id')[model + '_observed_minus_fitted_pp'].apply(lambda x: np.sqrt(np.mean(np.square(x))))
        declared = frame.set_index('partner_pair_comparison_id').loc[recovered.index, model + '_rmse_pp']
        assert np.allclose(recovered, declared, atol=EPS)
    sensitivity = read(OUT / 'actual_checkerboard_threshold_sensitivity.csv')
    assert (sensitivity.n_comparison_views + sensitivity.n_non_event_views).eq(204057).all()
    errors = read(OUT / 'reconstruction_error_threshold_sensitivity.csv')
    assert (errors.n_comparison_views + errors.n_non_event_views).eq(204057).all()
    check = dict(status='passed', observed_comparison_views=len(reread), exact_orthogonality=True, synthetic_additive_and_pure_fourth_cases=True, all_structural_and_partner_identities_verified=True, full_saturated_reconstruction_max_error_pp=float(np.abs(beta @ H.T - cubes.reshape(len(cubes), 16)).max()), fourth_contrast_matches_round7_for_all_views=True, actual_strong_flips_match_round7_for_all_views=True, complete_source_block_means_verified=True, all_threshold_denominators_read_back=True, original_record_to_cell_to_target_means_verified=True, independent_direct_least_squares_selected_max_disagreement_pp=float(direct_lstsq_max_error), selected_observed_and_fitted_cell_errors_read_back=True, all_comparison_csv_read_back=True)
    (OUT / 'verification.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8')
    return check
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    self_checks()
    pairs, targets, rectangles, relations, cubes = load()
    print('LOADED', len(pairs), cubes.shape, flush=True)
    frame, beta = decompose(pairs, cubes)
    error_summary, _, mapping = summaries(frame, cubes)
    print('DECOMPOSED', frame.threeway_rmse_pp.describe().to_dict(), flush=True)
    save(rectangles, 'all_primary_recipe_rectangles.csv')
    selected, source_counts = select_examples(frame, targets, rectangles, relations, beta)
    print('SELECTED', source_counts, flush=True)
    summary = report(frame, selected, error_summary, mapping, source_counts, time.monotonic() - started)
    summary['verification'] = verify(frame, beta, cubes)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('COMPLETED', json.dumps(summary, ensure_ascii=False), flush=True)
