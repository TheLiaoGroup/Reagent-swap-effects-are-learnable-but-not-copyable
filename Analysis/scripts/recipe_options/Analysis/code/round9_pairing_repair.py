"""Repair a lost component pairing with a measured third component option.

The original six-grid acquisition matching is retained. A restored pairing
state and recovery of an actually failed preferred cell are distinct events."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import round7_bridge_domains as six
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OLD = 'round7/bridge_domains/'
OUT = ROOT / 'round9/pairing_repair'
EPS = 1e-12
YS = [f'y{i}' for i in range(6)]
BASE_FIELDS = ['dataset_id', 'rxn_class', 'role', 'partner_smiles', 'molecule_a', 'molecule_b', 'structural_pair_id', 'graph_method', 'transformation', 'common_core', 'fragment_a', 'fragment_b', 'target_key_a', 'target_key_b']
THRESHOLDS = [('5pp', 0.05), ('10pp', 0.1)]

def read(name, **kwargs):
    return pd.read_csv(ROOT / name, keep_default_na=False, low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', compression={'method': 'gzip', 'compresslevel': 1} if name.endswith('.gz') else None)

def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def sid(value):
    return sha256(packed(value).encode('utf-8')).hexdigest()[:20]

def directed_graph_change(row):
    if row.graph_method == 'exact_substituent_relocation':
        return 'relocate ' + row.fragment_a + ' on identical parent graph'
    if row.graph_method == 'exact_methylene_insertion':
        return 'insert_CH2' if row.before_side == 0 else 'delete_CH2'
    if row.graph_method == 'exact_aromatic_atom_replacement':
        return 'aromatic_CH >> aromatic_N' if row.before_side == 0 else 'aromatic_N >> aromatic_CH'
    if ' >> ' in row.transformation:
        first, second = row.transformation.split(' >> ', 1)
        return first + ' >> ' + second if row.before_side == 0 else second + ' >> ' + first
    return row.transformation

def base_objects():
    panels = read(OLD + 'complete_six_recipe_registry.csv')
    routes = read(OLD + 'all_bridge_route_registry.csv', usecols=['route_id', 'panel_id'])
    legacy = read(OLD + 'all_exact_edit_partner_comparisons.csv.gz', usecols=[*BASE_FIELDS, 'route_id'])
    assert len(legacy) == 573894
    legacy = legacy.merge(routes, on='route_id', validate='many_to_one')
    key = ['dataset_id', 'role', 'target_key_a', 'target_key_b', 'panel_id']
    counts = legacy.groupby(key, sort=False).size()
    assert counts.eq(6).all(), counts.value_counts().to_dict()
    base = legacy.drop_duplicates(key).drop(columns='route_id').reset_index(drop=True)
    assert len(base) * 6 == len(legacy)
    base['base12_id'] = [sid(list(row)) for row in base[key].itertuples(index=False, name=None)]
    assert base.base12_id.is_unique
    base['edit_id'] = [sid([r.dataset_id, r.role, *sorted([r.molecule_a, r.molecule_b])]) for r in base.itertuples()]
    base['edit_panel_id'] = [sid([r.edit_id, r.panel_id]) for r in base.itertuples()]
    target = read(OLD + 'all_complete_target_six_yields.csv.gz')
    target_columns = ['dataset_id', 'panel_id', 'target_key', 'target_panel_id', *YS, 'n_complete_acquisition_blocks', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']
    for side in ['a', 'b']:
        table = target[target_columns].rename(columns={column: column + '_' + side for column in target_columns if column not in ['dataset_id', 'panel_id']})
        base = base.merge(table, on=['dataset_id', 'panel_id', 'target_key_' + side], validate='many_to_one')
    save(base, 'all_exact_edit_partner_panel_12cells.csv.gz')
    save(panels, 'complete_six_recipe_registry.csv')
    panel_coverage = panels.groupby('dataset_id').size().rename('n_registered_six_panels')
    target_coverage = target.groupby('dataset_id').size().rename('n_complete_target_six_panels')
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    assert len(mapping) == 20 and mapping.rxn_class.nunique() == 18
    coverage = mapping.merge(panel_coverage, on='dataset_id', how='left').merge(target_coverage, on='dataset_id', how='left').fillna(0)
    base_coverage = base.groupby('dataset_id').agg(n_complete_exact_edit_fixed_partner_panels=('base12_id', 'size'), n_exact_edits=('edit_id', 'nunique'), n_edit_panels=('edit_panel_id', 'nunique'))
    coverage = coverage.merge(base_coverage, on='dataset_id', how='left').fillna(0)
    save(coverage, 'all_20_dataset_denominator_coverage.csv')
    print('BASE', len(panels), len(target), len(base), base.groupby('rxn_class').size().to_dict(), flush=True)
    return (base, panels, target, coverage)

def square_metrics(square):
    preferred0 = np.stack([square[:, 0, 0], square[:, 1, 1]], axis=1)
    other0 = np.stack([square[:, 0, 1], square[:, 1, 0]], axis=1)
    m0 = preferred0.min(axis=1) - other0.max(axis=1)
    m1 = other0.min(axis=1) - preferred0.max(axis=1)
    orientation = np.where(m0 >= m1, 1, -1)
    preferred = np.where((orientation == 1)[:, None], preferred0, other0)
    others = np.where((orientation == 1)[:, None], other0, preferred0)
    return (orientation, preferred, others)

def evaluate(base):
    a = base[[y + '_a' for y in YS]].to_numpy(float).reshape(-1, 3, 2)
    b = base[[y + '_b' for y in YS]].to_numpy(float).reshape(-1, 3, 2)
    indices = np.arange(len(base))
    frames, responses = ([], [])
    common = ['base12_id', 'edit_id', 'edit_panel_id', 'dataset_id', 'rxn_class', 'panel_id', 'role', 'partner_smiles']
    for before_side, before, after in [(0, a, b), (1, b, a)]:
        for old0, old1 in combinations(range(3), 2):
            third = next((i for i in range(3) if i not in [old0, old1]))
            direction, preferred, others = square_metrics(before[:, [old0, old1], :])
            before_margin = preferred.min(axis=1) - others.max(axis=1)
            before_minimum = preferred.min(axis=1)
            binary0 = (direction == -1).astype(int)
            preferred_bins = [binary0, 1 - binary0]
            after_preferred = np.stack([after[indices, old0, preferred_bins[0]], after[indices, old1, preferred_bins[1]]], axis=1)
            after_others = np.stack([after[indices, old0, 1 - preferred_bins[0]], after[indices, old1, 1 - preferred_bins[1]]], axis=1)
            after_margin = after_preferred.min(axis=1) - after_others.max(axis=1)
            after_opposite_margin = after_others.min(axis=1) - after_preferred.max(axis=1)
            response = base[common].copy()
            suffix = f'_d{before_side}_s{old0}{old1}'
            response['response_id'] = response.base12_id + suffix
            response['domain_id'] = response.edit_panel_id + suffix
            response['before_side'], response['old_level0'], response['old_level1'], response['third_level'] = (before_side, old0, old1, third)
            response['original_assignment'] = direction
            response['before_original_pairing_margin'] = before_margin
            response['after_original_pairing_margin'] = after_margin
            choices = {}
            for tag, threshold in THRESHOLDS:
                original_strong = (before_margin >= threshold - EPS) & (before_minimum >= 0.2 - EPS)
                after_original_strong = (after_margin >= threshold - EPS) & (after_preferred.min(axis=1) >= 0.2 - EPS)
                after_opposite_strong = (after_opposite_margin >= threshold - EPS) & (after_others.min(axis=1) >= 0.2 - EPS)
                response['original_strong_' + tag] = original_strong
                response['pairing_loss_' + tag] = original_strong & ~after_original_strong
                response['after_pairing_state_relative_to_original_' + tag] = np.where(after_original_strong, 1, np.where(after_opposite_strong, -1, 0))
                response['loss_all_four_after_ge20_' + tag] = response['pairing_loss_' + tag] & (after[:, [old0, old1], :].min(axis=(1, 2)) >= 0.2 - EPS)
                choices[tag] = []
            for replace_index, replace_level in enumerate([old0, old1]):
                keep_index = 1 - replace_index
                keep_level = [old0, old1][keep_index]
                replace_bin, keep_bin = (preferred_bins[replace_index], preferred_bins[keep_index])
                retained = after[indices, keep_level, keep_bin]
                retained_other = after[indices, keep_level, 1 - keep_bin]
                failed = after[indices, replace_level, replace_bin]
                new = after[indices, third, replace_bin]
                new_other = after[indices, third, 1 - replace_bin]
                new_margin = np.minimum(retained, new) - np.maximum(retained_other, new_other)
                new_before_margin = np.minimum(before[indices, keep_level, keep_bin], before[indices, third, replace_bin]) - np.maximum(before[indices, keep_level, 1 - keep_bin], before[indices, third, 1 - replace_bin])
                gain = new - failed
                frame = response.copy()
                frame['action_id'] = frame.response_id + f'_r{replace_index}'
                frame['replacement_index'] = replace_index
                frame['replaced_level'], frame['retained_level'] = (replace_level, keep_level)
                frame['retained_preferred_binary_level'], frame['third_preferred_binary_level'] = (keep_bin, replace_bin)
                frame['after_retained_preferred_yield'] = retained
                frame['after_replaced_old_preferred_yield'] = failed
                frame['after_third_preferred_yield'] = new
                frame['after_retained_other_yield'], frame['after_third_other_yield'] = (retained_other, new_other)
                frame['actual_preferred_cell_gain'] = gain
                frame['after_new_aligned_pairing_margin'] = new_margin
                frame['before_new_aligned_pairing_margin'] = new_before_margin
                gain_events = {}
                for tag, threshold in THRESHOLDS:
                    loss = response['pairing_loss_' + tag].to_numpy(bool)
                    new_strong = (new_margin >= threshold - EPS) & (np.minimum(retained, new) >= 0.2 - EPS)
                    state_repair = loss & new_strong
                    opportunity = loss & (retained >= 0.2 - EPS) & (failed < 0.2 - EPS)
                    frame['new_aligned_strong_pairing_' + tag] = new_strong
                    frame['pairing_state_restored_' + tag] = state_repair
                    frame['failed_cell_repair_opportunity_' + tag] = opportunity
                    frame['state_restored_from_all_high_loss_' + tag] = state_repair & response['loss_all_four_after_ge20_' + tag]
                    for gain_tag, cutoff in [('gain5', 0.05), ('gain10', 0.1), ('gain20', 0.2)]:
                        event = state_repair & opportunity & (new >= 0.2 - EPS) & (gain >= cutoff - EPS)
                        field = 'actual_yield_repair_' + tag + '_' + gain_tag
                        frame[field] = event
                        gain_events[tag, gain_tag] = event
                    frame['actual_yield_repair_' + tag + '_gain10_threshold_margin5'] = gain_events[tag, 'gain10'] & (retained >= 0.25 - EPS) & (failed <= 0.15 + EPS) & (new >= 0.25 - EPS)
                    frame['actual_yield_repair_' + tag + '_gain10_threshold_margin10'] = gain_events[tag, 'gain10'] & (retained >= 0.3 - EPS) & (failed <= 0.1 + EPS) & (new >= 0.3 - EPS)
                    frame['state_restored_without_actual_yield_repair_' + tag] = state_repair & ~gain_events[tag, 'gain10']
                    frame['both_before_and_after_new_same_pairing_' + tag] = state_repair & (new_before_margin >= threshold - EPS) & (np.minimum(before[indices, keep_level, keep_bin], before[indices, third, replace_bin]) >= 0.2 - EPS)
                    choices[tag].append((state_repair, gain_events[tag, 'gain10'], opportunity, gain, new_margin))
                frames.append(frame)
            for tag, threshold in THRESHOLDS:
                first, second = choices[tag]
                assert not np.any(first[0] & second[0]), 'Both third-level binary choices cannot be strongly preferred'
                response['restored_pairing_choice_' + tag] = np.where(first[0], 0, np.where(second[0], 1, -1))
                response['actual_yield_repair_choice_' + tag] = np.where(first[1], 0, np.where(second[1], 1, -1))
                for option, arrays in enumerate([first, second]):
                    response[f'replacement{option}_failed_cell_opportunity_{tag}'] = arrays[2]
                    response[f'replacement{option}_actual_gain'] = arrays[3]
                    response[f'replacement{option}_new_pairing_margin'] = arrays[4]
            responses.append(response)
            print('ACTIONS', before_side, old0, old1, 'state5', sum((int(v[0].sum()) for v in choices['5pp'])), 'actual5_gain10', sum((int(v[1].sum()) for v in choices['5pp'])), flush=True)
    actions = pd.concat(frames, ignore_index=True)
    responses = pd.concat(responses, ignore_index=True)
    assert len(actions) == 12 * len(base) and len(responses) == 6 * len(base)
    assert actions.action_id.is_unique and responses.response_id.is_unique
    assert not responses.duplicated(['domain_id', 'partner_smiles']).any(), 'Separate product definitions require explicit partner-context handling'
    save(actions, 'all_pairing_repair_actions.csv.gz')
    save(responses, 'all_partner_response_choices.csv.gz')
    return (actions, responses)

def summarize(actions, responses, coverage):
    metrics = ['original_strong_5pp', 'pairing_loss_5pp', 'loss_all_four_after_ge20_5pp', 'pairing_state_restored_5pp', 'failed_cell_repair_opportunity_5pp', 'state_restored_from_all_high_loss_5pp', 'actual_yield_repair_5pp_gain5', 'actual_yield_repair_5pp_gain10', 'actual_yield_repair_5pp_gain20', 'actual_yield_repair_10pp_gain10', 'actual_yield_repair_5pp_gain10_threshold_margin5', 'actual_yield_repair_5pp_gain10_threshold_margin10', 'both_before_and_after_new_same_pairing_5pp']
    records = []
    for dataset in coverage.dataset_id:
        part = actions[actions.dataset_id.eq(dataset)]
        row = dict(dataset_id=dataset, n_replacement_actions=len(part), n_molecular_panel_original_square_responses=len(part) // 2)
        for metric in metrics:
            row['n_' + metric + '_actions'] = int(part[metric].sum())
        repair = part[part.actual_yield_repair_5pp_gain10]
        row['n_distinct_yield_repair_edits'] = repair.edit_id.nunique()
        row['n_distinct_yield_repair_edit_panels'] = repair.edit_panel_id.nunique()
        records.append(row)
    table = coverage.merge(pd.DataFrame(records), on='dataset_id', validate='one_to_one')
    save(table, 'all_20_dataset_event_and_denominator_coverage.csv')
    summary = dict(study_datasets=20, study_reaction_classes=18, study_legacy_swaps=4002, n_base_12_cell_objects=len(actions) // 12, n_directed_original_square_responses=len(responses), n_replacement_actions=len(actions), n_edit_panel_original_square_domains=responses.domain_id.nunique(), n_registered_six_panels=int(coverage.n_registered_six_panels.sum()), n_complete_target_panels=int(coverage.n_complete_target_six_panels.sum()))
    summary['action_counts'] = {metric: int(actions[metric].sum()) for metric in metrics}
    for tag, threshold in THRESHOLDS:
        loss = responses[responses['pairing_loss_' + tag]]
        actual = actions[actions['actual_yield_repair_' + tag + '_gain10']]
        state = actions[actions['pairing_state_restored_' + tag]]
        summary[tag] = dict(n_original_strong_responses=int(responses['original_strong_' + tag].sum()), n_loss_responses=len(loss), n_loss_all_four_after_ge20=int(loss['loss_all_four_after_ge20_' + tag].sum()), n_state_restored=len(state), n_actual_yield_repaired=len(actual), n_actual_repair_edits=actual.edit_id.nunique(), n_actual_repair_classes=actual.rxn_class.nunique(), n_actual_repair_edit_panels=actual.edit_panel_id.nunique(), classes=sorted(actual.rxn_class.unique()))
    return summary

def partner_domains(actions, responses):
    domain = responses.groupby('domain_id', sort=False).agg(edit_id=('edit_id', 'first'), edit_panel_id=('edit_panel_id', 'first'), dataset_id=('dataset_id', 'first'), rxn_class=('rxn_class', 'first'), panel_id=('panel_id', 'first'), role=('role', 'first'), before_side=('before_side', 'first'), old_level0=('old_level0', 'first'), old_level1=('old_level1', 'first'), third_level=('third_level', 'first'), n_complete_partners=('response_id', 'size'), n_original_strong_5pp=('original_strong_5pp', 'sum'), n_pairing_loss_5pp=('pairing_loss_5pp', 'sum'))
    save(domain.reset_index(), 'all_complete_partner_domain_denominators.csv.gz')
    groups = []
    for tag, threshold in THRESHOLDS:
        eligible = actions[actions['failed_cell_repair_opportunity_' + tag]].copy()
        key = ['domain_id', 'replacement_index', 'original_assignment']
        group = eligible.groupby(key, sort=False).agg(n_comparable_failed_cell_partners=('partner_smiles', 'size'), n_actual_repair_partners=('actual_yield_repair_' + tag + '_gain10', 'sum'), dataset_id=('dataset_id', 'first'), rxn_class=('rxn_class', 'first'), edit_id=('edit_id', 'first'), edit_panel_id=('edit_panel_id', 'first'))
        group['n_failed_repair_partners'] = group.n_comparable_failed_cell_partners - group.n_actual_repair_partners
        group['has_repair_success_and_failure'] = group.n_actual_repair_partners.gt(0) & group.n_failed_repair_partners.gt(0)
        group['n_success_failure_partner_pairs'] = group.n_actual_repair_partners * group.n_failed_repair_partners
        group['pairing_threshold'] = tag
        groups.append(group.reset_index())
    comparison = pd.concat(groups, ignore_index=True)
    save(comparison, 'all_fixed_repair_choice_partner_opportunity_domains.csv.gz')
    reversal_rows = []
    for tag, threshold in THRESHOLDS:
        repaired = responses[responses['actual_yield_repair_choice_' + tag].ge(0)]
        for key, group in repaired.groupby(['domain_id', 'original_assignment'], sort=False):
            first = group[group['actual_yield_repair_choice_' + tag].eq(0)]
            second = group[group['actual_yield_repair_choice_' + tag].eq(1)]
            for a in first.itertuples():
                for b in second.itertuples():
                    reversal_rows.append(dict(partner_case_id=sid([tag, a.response_id, b.response_id]), kind='replacement_choice_changes', pairing_threshold=tag, domain_id=key[0], original_assignment=key[1], dataset_id=a.dataset_id, rxn_class=a.rxn_class, edit_id=a.edit_id, edit_panel_id=a.edit_panel_id, response_id_partner0=a.response_id, response_id_partner1=b.response_id, action_id_partner0=a.response_id + '_r0', action_id_partner1=b.response_id + '_r1', partner0_smiles=a.partner_smiles, partner1_smiles=b.partner_smiles, actual_gain_partner0=a.replacement0_actual_gain, actual_gain_partner1=b.replacement1_actual_gain, minimum_actual_gain=min(a.replacement0_actual_gain, b.replacement1_actual_gain)))
    reversals = pd.DataFrame(reversal_rows)
    if reversals.empty:
        reversals = pd.DataFrame(columns=['partner_case_id', 'kind', 'pairing_threshold', 'domain_id', 'dataset_id', 'rxn_class', 'edit_id', 'edit_panel_id', 'response_id_partner0', 'response_id_partner1', 'action_id_partner0', 'action_id_partner1', 'partner0_smiles', 'partner1_smiles', 'actual_gain_partner0', 'actual_gain_partner1', 'minimum_actual_gain'])
    save(reversals, 'all_partner_replacement_choice_changes.csv.gz')
    success_failure_rows = []
    interesting = comparison[comparison.has_repair_success_and_failure]
    for row in interesting.itertuples():
        part = actions[actions.domain_id.eq(row.domain_id) & actions.replacement_index.eq(row.replacement_index) & actions.original_assignment.eq(row.original_assignment) & actions['failed_cell_repair_opportunity_' + row.pairing_threshold]]
        success = part[part['actual_yield_repair_' + row.pairing_threshold + '_gain10']].sort_values('actual_preferred_cell_gain', ascending=False).iloc[0]
        failure = part[~part['actual_yield_repair_' + row.pairing_threshold + '_gain10']].sort_values('actual_preferred_cell_gain').iloc[0]
        success_failure_rows.append(dict(partner_case_id=sid([row.pairing_threshold, success.action_id, failure.action_id]), kind='same_replacement_success_and_failure', pairing_threshold=row.pairing_threshold, domain_id=row.domain_id, original_assignment=row.original_assignment, dataset_id=row.dataset_id, rxn_class=row.rxn_class, edit_id=row.edit_id, edit_panel_id=row.edit_panel_id, response_id_partner0=success.response_id, response_id_partner1=failure.response_id, action_id_partner0=success.action_id, action_id_partner1=failure.action_id, partner0_smiles=success.partner_smiles, partner1_smiles=failure.partner_smiles, actual_gain_partner0=success.actual_preferred_cell_gain, actual_gain_partner1=failure.actual_preferred_cell_gain, minimum_actual_gain=min(success.actual_preferred_cell_gain, failure.actual_preferred_cell_gain)))
    success_failure = pd.DataFrame(success_failure_rows, columns=reversals.columns)
    save(success_failure, 'representative_partner_success_failure_cases.csv')
    return (domain.reset_index(), comparison, reversals, success_failure)

def selected_examples(base, panels, actions, responses, reversals, success_failure):
    picked = []
    for metric in ['actual_yield_repair_10pp_gain20', 'actual_yield_repair_5pp_gain10', 'state_restored_from_all_high_loss_5pp']:
        part = actions[actions[metric]].sort_values(['actual_preferred_cell_gain', 'after_new_aligned_pairing_margin'], ascending=False)
        for _, group in part.groupby('rxn_class', sort=True):
            picked.append(dict(action_id=group.iloc[0].action_id, selection_reason=metric))
    selections = pd.DataFrame(picked).drop_duplicates('action_id')
    cases = []
    for table in [reversals, success_failure]:
        if table.empty:
            continue
        preferred = table.sort_values(['pairing_threshold', 'minimum_actual_gain'], ascending=[True, False])
        for _, group in preferred.groupby('rxn_class', sort=True):
            case = group.iloc[0]
            cases.append(case)
            for side in [0, 1]:
                selections = pd.concat([selections, pd.DataFrame([dict(action_id=case[f'action_id_partner{side}'], selection_reason=case.kind)])], ignore_index=True)
    selections = selections.drop_duplicates('action_id')
    chosen = selections.merge(actions, on='action_id', validate='one_to_one')
    chosen = chosen.merge(base.drop(columns=[c for c in base.columns if c in chosen.columns and c != 'base12_id']), on='base12_id', validate='many_to_one')
    chosen = chosen.merge(panels.drop(columns=['dataset_id', 'rxn_class']), on='panel_id', validate='many_to_one')
    registry = read('data/condition_slot_registry.csv').drop_duplicates(['dataset_id', 'recipe_id'])
    lookup = registry.set_index(['dataset_id', 'recipe_id']).to_dict('index')
    for prefix, level_column, slot_column in [('replaced', 'replaced_level', 'ternary_slot'), ('retained', 'retained_level', 'ternary_slot'), ('third', 'third_level', 'ternary_slot')]:
        chosen[prefix + '_component'] = [six.label(lookup[r.dataset_id, json.loads(r.recipe_ids)[2 * getattr(r, level_column)]], getattr(r, slot_column)) for r in chosen.itertuples()]
    for binary in [0, 1]:
        chosen[f'binary{binary}_component'] = [six.label(lookup[r.dataset_id, json.loads(r.recipe_ids)[binary]], r.binary_slot) for r in chosen.itertuples()]
    chosen['molecule_before'] = np.where(chosen.before_side.eq(0), chosen.molecule_a, chosen.molecule_b)
    chosen['molecule_after'] = np.where(chosen.before_side.eq(0), chosen.molecule_b, chosen.molecule_a)
    chosen['directed_graph_transformation'] = [directed_graph_change(r) for r in chosen.itertuples()]
    save(chosen, 'selected_pairing_repair_12cell_examples.csv')
    partner_cases = pd.DataFrame(cases)
    if partner_cases.empty:
        partner_cases = reversals.iloc[:0].copy()
    save(partner_cases, 'selected_partner_repair_24cell_cases.csv')
    domains = set(chosen.domain_id)
    full_responses = responses[responses.domain_id.isin(domains)].merge(base.drop(columns=[c for c in base.columns if c in responses.columns and c != 'base12_id']), on='base12_id', validate='many_to_one')
    save(full_responses, 'selected_domains_all_complete_partner_12cell_responses.csv.gz')
    save(actions[actions.domain_id.isin(domains)], 'selected_domains_all_partner_replacement_actions.csv.gz')
    long = []
    for row in chosen.itertuples():
        for side, source_side in [('before', 'a' if row.before_side == 0 else 'b'), ('after', 'b' if row.before_side == 0 else 'a')]:
            for position, recipe in enumerate(json.loads(row.recipe_ids)):
                condition = lookup[row.dataset_id, recipe]
                long.append(dict(action_id=row.action_id, base12_id=row.base12_id, domain_id=row.domain_id, dataset_id=row.dataset_id, rxn_class=row.rxn_class, panel_id=row.panel_id, molecular_side=side, target_panel_id=getattr(row, 'target_panel_id_' + source_side), target_key=getattr(row, 'target_key_' + source_side), ternary_level=position // 2, binary_level=position % 2, panel_position=position, recipe_id=recipe, yield_numeric=getattr(row, f'y{position}_' + source_side), ternary_component=six.label(condition, row.ternary_slot), binary_component=six.label(condition, row.binary_slot), full_recipe='; '.join((slot + '=' + six.label(condition, slot) for slot in six.SLOTS)), reactant1_smiles=getattr(row, 'sub_1_smiles_' + source_side), reactant2_smiles=getattr(row, 'sub_2_smiles_' + source_side), product_smiles=getattr(row, 'product_smiles_' + source_side)))
    cells = pd.DataFrame(long)
    assert len(cells) == 12 * len(chosen)
    save(cells, 'selected_pairing_repair_all_12_yields_and_conditions.csv')
    return (chosen, partner_cases, full_responses, cells)

def all_repair_metadata(actions, base, panels):
    actual = actions[actions.actual_yield_repair_5pp_gain10].copy()
    actual = actual.merge(base.drop(columns=[c for c in base.columns if c in actual.columns and c != 'base12_id']), on='base12_id', validate='many_to_one')
    actual = actual.merge(panels.drop(columns=['dataset_id', 'rxn_class']), on='panel_id', validate='many_to_one')
    actual['molecule_before'] = np.where(actual.before_side.eq(0), actual.molecule_a, actual.molecule_b)
    actual['molecule_after'] = np.where(actual.before_side.eq(0), actual.molecule_b, actual.molecule_a)
    actual['directed_graph_transformation'] = [directed_graph_change(r) for r in actual.itertuples()]
    actual['target_key_after'] = np.where(actual.before_side.eq(0), actual.target_key_b, actual.target_key_a)
    actual['failed_preferred_recipe_id'] = [json.loads(r.recipe_ids)[2 * r.replaced_level + r.third_preferred_binary_level] for r in actual.itertuples()]
    actual['repaired_preferred_recipe_id'] = [json.loads(r.recipe_ids)[2 * r.third_level + r.third_preferred_binary_level] for r in actual.itertuples()]
    actual['retained_preferred_recipe_id'] = [json.loads(r.recipe_ids)[2 * r.retained_level + r.retained_preferred_binary_level] for r in actual.itertuples()]
    actual['measured_cell_repair_id'] = [sid([r.dataset_id, r.target_key_after, r.failed_preferred_recipe_id, r.repaired_preferred_recipe_id]) for r in actual.itertuples()]
    save(actual, 'all_actual_yield_repairs_with_12cells.csv')
    return actual

def state_partner_choice_scope(responses):
    frames = []
    for tag, _ in THRESHOLDS:
        part = responses[['domain_id', 'original_assignment', 'dataset_id', 'rxn_class', 'pairing_loss_' + tag, 'restored_pairing_choice_' + tag]].copy()
        for option in [0, 1, -1]:
            part['choice' + str(option)] = part['pairing_loss_' + tag] & part['restored_pairing_choice_' + tag].eq(option)
        group = part.groupby(['domain_id', 'original_assignment'], sort=False).agg(dataset_id=('dataset_id', 'first'), rxn_class=('rxn_class', 'first'), n_complete_partners=('dataset_id', 'size'), n_lost_pairing_partners=('pairing_loss_' + tag, 'sum'), n_state_restored_replacing0=('choice0', 'sum'), n_state_restored_replacing1=('choice1', 'sum'), n_no_state_restoration=('choice-1', 'sum')).reset_index()
        group['has_state_replacement_choice_change'] = group.n_state_restored_replacing0.gt(0) & group.n_state_restored_replacing1.gt(0)
        group['pairing_threshold'] = tag
        frames.append(group)
    result = pd.concat(frames, ignore_index=True)
    save(result, 'all_partner_state_restoration_choice_scope.csv.gz')
    return {tag: dict(n_same_assignment_loss_domains=int((result.pairing_threshold.eq(tag) & result.n_lost_pairing_partners.gt(0)).sum()), n_choice_change_domains=int((result.pairing_threshold.eq(tag) & result.has_state_replacement_choice_change).sum()), n_choice_change_classes=result[result.pairing_threshold.eq(tag) & result.has_state_replacement_choice_change].rxn_class.nunique()) for tag, _ in THRESHOLDS}

def source_reconstruction(base, full_responses, actual, panels, chosen_cells, prefix=''):
    all_target = read(OLD + 'all_complete_target_six_yields.csv.gz')
    acquisition = read(OLD + 'all_complete_acquisition_six_yields.csv.gz')
    keys = ['dataset_id', 'target_key', 'panel_id']
    mean_all = acquisition.groupby(keys, sort=False)[YS].mean()
    expected_all = all_target.set_index(keys)[YS]
    assert len(mean_all) == len(expected_all) and np.allclose(mean_all.reindex(expected_all.index), expected_all, atol=EPS)
    required = base[base.base12_id.isin(set(full_responses.base12_id) | set(actual.base12_id))]
    wanted = pd.concat([required[['dataset_id', 'panel_id', 'target_key_' + side]].rename(columns={'target_key_' + side: 'target_key'}) for side in ['a', 'b']]).drop_duplicates()
    target = wanted.merge(all_target, on=keys, validate='one_to_one')
    normal = read('data/normalized_records.csv.gz')
    normal = normal[normal.strict_eligible.astype(str).str.lower().eq('true')].copy()
    normal['yield_numeric'] = pd.to_numeric(normal.yield_numeric, errors='raise')
    raw = read('data/raw_reactions.csv.gz')
    pmeta = panels.set_index('panel_id')
    blocks, requests = ([], [])
    for panel, group in target.groupby('panel_id', sort=False):
        dataset = group.dataset_id.iloc[0]
        recipes = json.loads(pmeta.loc[panel, 'recipe_ids'])
        source = normal[normal.dataset_id.eq(dataset) & normal.target_key.isin(group.target_key) & normal.recipe_id.isin(recipes)]
        cells = source.groupby([*six.BLOCK, 'recipe_id'], sort=False, as_index=False).yield_numeric.mean()
        block = cells.pivot(index=six.BLOCK, columns='recipe_id', values='yield_numeric').reindex(columns=recipes).dropna().reset_index()
        block = block.rename(columns=dict(zip(recipes, YS)))
        expected = group.set_index('target_key')
        recon = block.groupby('target_key')[YS].mean().reindex(expected.index)
        assert np.allclose(recon, expected[YS], atol=EPS), panel
        block['target_panel_id'] = block.target_key.map(expected.target_panel_id)
        block['panel_id'] = panel
        blocks.append(block)
        for position, recipe in enumerate(recipes):
            request = block[['panel_id', 'target_panel_id', *six.BLOCK, YS[position]]].rename(columns={YS[position]: 'matched_cell_yield'})
            request['panel_position'], request['recipe_id'] = (position, recipe)
            requests.append(request)
    blocks, requests = (pd.concat(blocks, ignore_index=True), pd.concat(requests, ignore_index=True))
    links = requests.merge(normal, on=[*six.BLOCK, 'recipe_id'], how='left', validate='many_to_many')
    assert links.record_id.notna().all()
    assert not links.duplicated(['panel_id', 'target_panel_id', 'archive', 'plate', 'acquisition', 'panel_position', 'record_id']).any()
    fields = ['record_id', 'source_file', 'source_sheet', 'source_row', *[f'reagent{i}_id' for i in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    links = links.merge(raw[fields], on='record_id', validate='many_to_one')
    raw_root = RAW_DIR
    links['source_absolute_path'] = links.source_file.map(lambda name: str(raw_root / name))
    assert all((reader_tables.source_is_file(path) for path in links.source_absolute_path.unique()))
    save(target, prefix + 'all_actual_repairs_and_selected_partner_target_six_yields.csv.gz')
    save(blocks, prefix + 'all_actual_repairs_and_selected_partner_complete_acquisition_blocks.csv.gz')
    save(links, prefix + 'all_actual_repairs_and_selected_partner_original_record_links.csv.gz')
    memberships = chosen_cells[['action_id', 'molecular_side', 'target_panel_id', 'panel_position']].merge(links, on=['target_panel_id', 'panel_position'], validate='many_to_many')
    save(memberships, prefix + 'selected_12cell_original_records.csv.gz')
    original_fields = ['sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'condition_id', *[f'reagent{i}_id' for i in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    wanted_raw = links[['source_file', 'source_sheet', 'source_row', 'yield_numeric', *original_fields]].drop_duplicates()
    verified, checked, max_error = (0, 0, 0.0)
    for (filename, sheet), group in wanted_raw.groupby(['source_file', 'source_sheet'], sort=False):
        path = raw_root / filename
        original = reader_tables.read_csv(path, low_memory=False, keep_default_na=False) if path.suffix.lower() == '.csv' else reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for row in group.itertuples():
            source_row = original.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(row.source_row)]
            error = abs(float(source_row['yield']) - row.yield_numeric)
            assert error < EPS, (filename, row.source_row, error)
            max_error = max(max_error, error)
            for field in original_fields:
                observed, expected = (source_row[field], getattr(row, field))
                if field in ['temp_id', 'time_id'] and str(observed) != '' and (str(expected) != ''):
                    assert float(observed) == float(expected), (filename, row.source_row, field)
                else:
                    assert str(observed) == str(expected), (filename, row.source_row, field, observed, expected)
                checked += 1
            verified += 1
    return dict(n_all_target_panels_reconstructed_from_complete_blocks=len(expected_all), n_selected_target_panels=len(target), n_selected_complete_acquisition_blocks=len(blocks), n_source_links=len(links), n_unique_original_records=links.record_id.nunique(), n_selected_12cell_memberships=len(memberships), n_original_rows_reread=verified, n_original_files=wanted_raw.source_file.nunique(), n_original_structure_recipe_fields_checked=checked, max_original_yield_error=max_error)

def state_choice_details(base, panels, actions, responses):
    scope = read('round9/pairing_repair/all_partner_state_restoration_choice_scope.csv.gz')
    changed = scope[scope.has_state_replacement_choice_change]
    cases, chosen_actions = ([], [])
    for domain in changed.itertuples():
        part = responses[responses.domain_id.eq(domain.domain_id) & responses.original_assignment.eq(domain.original_assignment)]
        pair = [part[part['restored_pairing_choice_' + domain.pairing_threshold].eq(option)].iloc[0] for option in [0, 1]]
        case_id = sid([domain.pairing_threshold, pair[0].response_id, pair[1].response_id, 'state_only_choice_change'])
        cases.append(dict(partner_case_id=case_id, kind='state_replacement_choice_change', pairing_threshold=domain.pairing_threshold, domain_id=domain.domain_id, original_assignment=domain.original_assignment, dataset_id=domain.dataset_id, rxn_class=domain.rxn_class, response_id_partner0=pair[0].response_id, response_id_partner1=pair[1].response_id, action_id_partner0=pair[0].response_id + '_r0', action_id_partner1=pair[1].response_id + '_r1', partner0_smiles=pair[0].partner_smiles, partner1_smiles=pair[1].partner_smiles))
        for option in [0, 1]:
            chosen_actions.append(dict(partner_case_id=case_id, partner_level=option, action_id=pair[option].response_id + f'_r{option}'))
    cases = pd.DataFrame(cases)
    selection = pd.DataFrame(chosen_actions)
    chosen = selection.merge(actions, on='action_id', validate='one_to_one')
    chosen = chosen.merge(base.drop(columns=[c for c in base.columns if c in chosen.columns and c != 'base12_id']), on='base12_id', validate='many_to_one')
    chosen = chosen.merge(panels.drop(columns=['dataset_id', 'rxn_class']), on='panel_id', validate='many_to_one')
    chosen['directed_graph_transformation'] = [directed_graph_change(r) for r in chosen.itertuples()]
    full = responses[responses.domain_id.isin(set(cases.domain_id))].merge(base.drop(columns=[c for c in base.columns if c in responses.columns and c != 'base12_id']), on='base12_id', validate='many_to_one')
    save(cases, 'selected_state_choice_change_24cell_cases.csv')
    save(chosen, 'selected_state_choice_change_actions_with_12cells.csv')
    save(full, 'selected_state_choice_change_all_partner_12cells.csv.gz')
    registry = read('data/condition_slot_registry.csv').drop_duplicates(['dataset_id', 'recipe_id'])
    lookup = registry.set_index(['dataset_id', 'recipe_id']).to_dict('index')
    rows = []
    for row in chosen.itertuples():
        for side, source_side in [('before', 'a' if row.before_side == 0 else 'b'), ('after', 'b' if row.before_side == 0 else 'a')]:
            for position, recipe in enumerate(json.loads(row.recipe_ids)):
                condition = lookup[row.dataset_id, recipe]
                rows.append(dict(partner_case_id=row.partner_case_id, partner_level=row.partner_level, action_id=row.action_id, molecular_side=side, dataset_id=row.dataset_id, rxn_class=row.rxn_class, panel_id=row.panel_id, target_panel_id=getattr(row, 'target_panel_id_' + source_side), panel_position=position, ternary_level=position // 2, binary_level=position % 2, yield_numeric=getattr(row, f'y{position}_' + source_side), ternary_component=six.label(condition, row.ternary_slot), binary_component=six.label(condition, row.binary_slot), reactant1_smiles=getattr(row, 'sub_1_smiles_' + source_side), reactant2_smiles=getattr(row, 'sub_2_smiles_' + source_side), product_smiles=getattr(row, 'product_smiles_' + source_side), recipe_id=recipe, full_recipe='; '.join((slot + '=' + six.label(condition, slot) for slot in six.SLOTS))))
    cells = pd.DataFrame(rows)
    assert len(cells) == 24 * len(cases)
    save(cells, 'selected_state_choice_change_all_24_yields_and_conditions.csv')
    verification = source_reconstruction(base, full, pd.DataFrame(columns=['base12_id']), panels, cells, prefix='state_choice_')
    return dict(n_selected_state_choice_cases=len(cases), n_all_complete_partner_responses=len(full), verification=verification)

def write_report(summary, chosen, partner_cases, cells):
    s5, s10 = (summary['5pp'], summary['10pp'])
    lines = ['# A third component choice restores pairing broken by a molecular edit', '', 'This analysis checks complete six-recipe sets before and after the same exact molecular edit: the original molecule A has a strong pairing over two component options × two binary components; the edited molecule B loses this pairing. One component option is then kept and the other is replaced by a measured third choice, to check whether a relation with the same pairing identity as in A is re-established, and separately whether the originally failed actual yield cells recover.', '', f"The study still retains 20 datasets, 18 reaction classes and 4,002 context-specific single-slot swaps. All {summary['n_registered_six_panels']:,} registered six-recipe panels and {summary['n_complete_target_panels']:,} complete target × panel rows form {summary['n_base_12_cell_objects']:,} 12-cell sets of exact molecule pair × fixed partner × panel. The six route views of each R7 12-cell set are first deduplicated, then three original four-cell sets, two molecular directions and two replacement choices are enumerated, giving {summary['n_replacement_actions']:,} actions; cases without pairing loss and without recovery are all kept.", '', f"With a 5pp advantage and both originally preferred cells ≥20%, {s5['n_original_strong_responses']:,} directed original four-cell sets have a strong pairing, and {s5['n_loss_responses']:,} lose the original pairing after the edit. Of these, in {s5['n_loss_all_four_after_ge20']:,} all four cells of B are ≥20%, so there is no failed preferred cell to begin with. The third choice restores the pairing state in {s5['n_state_restored']:,} actions; {s5['n_actual_yield_repaired']:,} of them also raise one original preferred cell from below 20% to ≥20% with a gain of at least 10pp, covering {s5['n_actual_repair_edits']} exact edits, {s5['n_actual_repair_edit_panels']} edit × panel rows and {s5['n_actual_repair_classes']} reaction classes.", '', f"With the pairing advantage raised to 10pp, actual recovery remains in {s10['n_actual_yield_repaired']} actions and {s10['n_actual_repair_classes']} classes. With an additional 5pp/10pp margin on both sides of the 20% threshold for the original failed cell, the newly recovered cell and the retained preferred cell, the counts under the 5pp pairing definition are {summary['action_counts']['actual_yield_repair_5pp_gain10_threshold_margin5']}/{summary['action_counts']['actual_yield_repair_5pp_gain10_threshold_margin10']}.", '', '## Complete 12-cell chemistry', '']
    for case in chosen.itertuples():
        values = cells[cells.action_id.eq(case.action_id)]
        lines.extend([f'### {case.rxn_class}: {case.selection_reason}', '', f'`{case.action_id}`. Structure changes from `{case.molecule_before}` to `{case.molecule_after}`, fixed partner `{case.partner_smiles}`; graph rule {case.graph_method}, directed change in this example `{case.directed_graph_transformation}`.', '', f'Of the original three levels, {case.retained_component} is kept and {case.replaced_component} is replaced by {case.third_component}; the other component is {case.binary0_component}/{case.binary1_component}.', '', '| Molecule | Ternary component choice | Binary 0 yield % | Binary 1 yield % |', '|---|---|---:|---:|'])
        for side in ['before', 'after']:
            for level in range(3):
                pair = values[values.molecular_side.eq(side) & values.ternary_level.eq(level)].sort_values('binary_level')
                lines.append(f'| {side} | {pair.iloc[0].ternary_component} | {100 * pair.iloc[0].yield_numeric:.2f} | {100 * pair.iloc[1].yield_numeric:.2f} |')
        lines.extend(['', f'Original preferred cell of B {100 * case.after_replaced_old_preferred_yield:.2f}% → third choice with the same binary partner {100 * case.after_third_preferred_yield:.2f}%; retained preferred cell {100 * case.after_retained_preferred_yield:.2f}%, new pairing advantage {100 * case.after_new_aligned_pairing_margin:.2f}pp. actual_yield_repair_5pp_gain10={case.actual_yield_repair_5pp_gain10}. Full nine-slot recipes are in the per-cell table; the registered none denotes an omitted component.', ''])
    lines.extend(['## Partner dependence and boundaries', '', f"With the same original A pairing direction, the same original four-cell set and the same third substitute choice, both recovery and non-recovery occur among partners with the same failed-cell opportunity: under the 5pp definition, {summary['fixed_choice_success_and_failure']['5pp']['n_opportunity_groups']} option × direction domains; under 10pp, {summary['fixed_choice_success_and_failure']['10pp']['n_opportunity_groups']}. Among strict actual recoveries, cases where different partners require the third choice to replace different original levels: {summary['partner_choice_changes']['5pp']['n_pairs']} partner pairs.", '', 'In the complete 24-cell instances, the new pairing advantage and single-cell yields must be read together: in the selected acylation comparison, the same replacement raised the failed cell for both partners, but only one re-formed the pairing with the original identity. The failure for the other partner therefore means that the pairing relation was not restored, not that the yield of that cell did not recover. All complete partners and both actions of all selected domains remain kept.', '', f"In addition, {summary['state_restoration_partner_choice_scope']['5pp']['n_choice_change_domains']} Suzuki domains show, under the 5pp definition, a partner-dependent change in the replacement option needed to restore the pairing state; under the 10pp definition, {summary['state_restoration_partner_choice_scope']['10pp']['n_choice_change_domains']}; their complete 24 cells, all partners and sources are saved separately in selected_state_choice_change_*. Some individual partners meet actual failed-cell recovery, but no partner pair meets this strict condition jointly, so these are not counted as choice reversals of actual yield recovery.", '', '## Definitions and reproducibility', '', 'Old four-cell sets are defined by ternary options old0/old1 and binary options bin0/bin1. A 00/11 or 01/10 pairing for A requires both preferred cells ≥20% and min(preferred)-max(other two cells)≥5pp or 10pp. B loses the original pairing either by having no strong pairing or by switching to the opposite direction. The new four-cell set keeps the original binary preferred identity of the retained option; the third option should prefer the opposite binary identity. state_restored checks only this aligned pairing; actual_yield_repair also requires the retained preferred cell of B ≥20%, the replaced original preferred cell <20%, the same-binary cell of the third option ≥20%, and an actual gain ≥5/10/20pp.', '', 'Complete six cells are always matched first within dataset/archive/plate/acquisition/target and then averaged within target. Missing measurements are not filled with zero, and zero yields are kept. The same recipes, source rows, partners and targets are reused in different pairing definitions; action counts and measured cell positions are not counts of independent experiments. Actual recovery corresponds to failure of one preferred endpoint of B in the original four cells, and is defined separately from the R7 specified bridge-path comparison, which requires both original endpoints to work.', '', 'all_pairing_repair_actions.csv.gz stores all actions, all_partner_response_choices.csv.gz stores directed original four-cell sets with duplicates from the two replacement actions removed, and all_exact_edit_partner_panel_12cells.csv.gz stores all actual 12-cell sets. all_actual_yield_repairs_with_12cells.csv also gives all strict recoveries as actual failed recipe → recovered recipe; measured_cell_repair_id identifies the same measured cell change reused by multiple structural relations.', '', '## Summary and verification of original files', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', ''])
    (OUT / 'PAIRING_REPAIR_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if '--refresh-descriptions' in sys.argv:
        for filename in ['selected_pairing_repair_12cell_examples.csv', 'all_actual_yield_repairs_with_12cells.csv', 'selected_state_choice_change_actions_with_12cells.csv']:
            table = read('round9/pairing_repair/' + filename)
            table['directed_graph_transformation'] = [directed_graph_change(r) for r in table.itertuples()]
            save(table, filename)
        summary = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
        write_report(summary, read('round9/pairing_repair/selected_pairing_repair_12cell_examples.csv'), read('round9/pairing_repair/selected_partner_repair_24cell_cases.csv'), read('round9/pairing_repair/selected_pairing_repair_all_12_yields_and_conditions.csv'))
        print('Directed structural descriptions and report refreshed', flush=True)
        return
    if '--state-details' in sys.argv:
        base = read('round9/pairing_repair/all_exact_edit_partner_panel_12cells.csv.gz')
        panels = read(OLD + 'complete_six_recipe_registry.csv')
        actions = read('round9/pairing_repair/all_pairing_repair_actions.csv.gz')
        responses = read('round9/pairing_repair/all_partner_response_choices.csv.gz')
        summary = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
        summary['state_choice_details'] = state_choice_details(base, panels, actions, responses)
        chosen = read('round9/pairing_repair/selected_pairing_repair_12cell_examples.csv')
        partner_cases = read('round9/pairing_repair/selected_partner_repair_24cell_cases.csv')
        cells = read('round9/pairing_repair/selected_pairing_repair_all_12_yields_and_conditions.csv')
        write_report(summary, chosen, partner_cases, cells)
        (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary['state_choice_details'], ensure_ascii=False, indent=2), flush=True)
        return
    if '--resume' in sys.argv:
        base = read('round9/pairing_repair/all_exact_edit_partner_panel_12cells.csv.gz')
        panels = read(OLD + 'complete_six_recipe_registry.csv')
        coverage = read('round9/pairing_repair/all_20_dataset_denominator_coverage.csv')
        actions = read('round9/pairing_repair/all_pairing_repair_actions.csv.gz')
        responses = read('round9/pairing_repair/all_partner_response_choices.csv.gz')
    else:
        base, panels, targets, coverage = base_objects()
        actions, responses = evaluate(base)
    summary = summarize(actions, responses, coverage)
    domains, comparisons, reversals, success_failure = partner_domains(actions, responses)
    summary['partner_choice_changes'] = {tag: dict(n_pairs=len(reversals[reversals.pairing_threshold.eq(tag)]), n_domains=reversals[reversals.pairing_threshold.eq(tag)].domain_id.nunique(), n_classes=reversals[reversals.pairing_threshold.eq(tag)].rxn_class.nunique()) for tag, _ in THRESHOLDS}
    summary['fixed_choice_success_and_failure'] = {tag: dict(n_opportunity_groups=int((comparisons.pairing_threshold.eq(tag) & comparisons.has_repair_success_and_failure).sum())) for tag, _ in THRESHOLDS}
    chosen, partner_cases, full_responses, cells = selected_examples(base, panels, actions, responses, reversals, success_failure)
    actual = all_repair_metadata(actions, base, panels)
    summary['n_unique_measured_failed_to_repaired_cell_pairs'] = actual.measured_cell_repair_id.nunique()
    summary['state_restoration_partner_choice_scope'] = state_partner_choice_scope(responses)
    summary['verification'] = source_reconstruction(base, full_responses, actual, panels, cells)
    summary['state_choice_details'] = state_choice_details(base, panels, actions, responses)
    summary['n_selected_actions'] = len(chosen)
    summary['n_selected_partner_24cell_cases'] = len(partner_cases)
    summary['seconds'] = time.monotonic() - started
    summary['status'] = 'complete'
    write_report(summary, chosen, partner_cases, cells)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
