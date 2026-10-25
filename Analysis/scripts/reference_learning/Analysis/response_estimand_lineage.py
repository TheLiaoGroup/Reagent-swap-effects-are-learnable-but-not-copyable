"""Exact retrospective lineage of directed bridge response linear operators.

Atoms are contrast_id x pair_index paired differences. Operators expand atoms
onto screen x pair_index x condition yield coordinates, with rational weights."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import controlled_cross_library_bridges as bridge
import transport_domain_atlas as atlas
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'response_estimand_lineage_'
REPORT = 'response_estimand_lineage.py'
PAIR = atlas.PAIR_KEYS
CONFIG = ['background_control', 'basis', 'min_partners_each', 'partner_overlap']

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_map(weights):
    return [[*key, value.numerator, value.denominator] for key, value in sorted(weights.items()) if value]

def signature(weights):
    return hashlib.sha256(dumps(canonical_map(weights)).encode('utf-8')).hexdigest()

def average_maps(maps):
    if not maps:
        raise ValueError('cannot average an empty response population')
    result = defaultdict(Fraction)
    for weights in maps:
        for key, value in weights.items():
            result[key] += value / len(maps)
    return {key: value for key, value in result.items() if value}

def expand_operator(atom_weights, contexts):
    result = defaultdict(Fraction)
    for (contrast, pair_index), weight in atom_weights.items():
        screen, low, high = contexts[contrast]
        result[screen, int(pair_index), low] -= weight
        result[screen, int(pair_index), high] += weight
    return {key: value for key, value in result.items() if value}

def chemical_key(value):
    decoded = json.loads(value)
    if not isinstance(decoded, list) or len(decoded) not in (3, 4):
        raise ValueError('unexpected perturbation key schema')
    return dumps(decoded[:3])

def load_inputs():
    names = ['tables/controlled_bridge_atomic_effects.csv.gz', 'tables/controlled_bridge_conditional_estimands.csv.gz', 'tables/controlled_bridge_directed_bridge_registry.csv', 'tables/cross_library_components_perturbation_contexts.csv', 'tables/transport_domain_atlas_cells.csv', 'tables/transport_domain_atlas_dose_contrasts.csv', 'tables/transport_domain_atlas_mode_semantics.csv', 'data/controlled_bridge_manifest.json', 'data/cross_library_components_manifest.json', 'data/qualified_observations.csv.gz', 'data/raw_long.csv.gz', 'run_existing_evidence.py', 'controlled_cross_library_bridges.py', 'cross_library_component_structure.py', 'transport_domain_atlas.py']
    hashes = {name: digest(HERE / name) for name in names}
    source_helper = HERE.parent / 'external' / 'hte_common.py'
    hashes[str(source_helper)] = digest(source_helper)
    atomic = pd.read_csv(TABLES / 'controlled_bridge_atomic_effects.csv.gz', keep_default_na=False)
    units = pd.read_csv(TABLES / 'controlled_bridge_conditional_estimands.csv.gz', keep_default_na=False)
    registry = pd.read_csv(TABLES / 'controlled_bridge_directed_bridge_registry.csv', keep_default_na=False)
    contrasts = pd.read_csv(TABLES / 'cross_library_components_perturbation_contexts.csv', keep_default_na=False)
    cells = pd.read_csv(TABLES / 'transport_domain_atlas_cells.csv', keep_default_na=False)
    used = set(registry.source_unit_id) | set(registry.target_unit_id)
    units = units[units.unit_id.isin(used)].copy()
    if set(units.unit_id) != used or units.unit_id.duplicated().any():
        raise ValueError('registered endpoint unit lookup is not complete and unique')
    inventory = pd.read_csv(TABLES / 'cross_library_components_screen_inventory.csv', keep_default_na=False)
    for screen in inventory.screen:
        path = DATA / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
        hashes[str(path.relative_to(HERE))] = digest(path)
    hashes['tables/cross_library_components_screen_inventory.csv'] = digest(TABLES / 'cross_library_components_screen_inventory.csv')
    return (atomic, units, registry, contrasts, cells, hashes)

def build_record_provenance(coefficients):
    qualified = pd.read_csv(DATA / 'qualified_observations.csv.gz', keep_default_na=False)
    qualified = qualified[qualified.partition.eq('analysis')]
    coordinates = coefficients[['screen', 'pair_index', 'condition']].drop_duplicates().sort_values(['screen', 'pair_index', 'condition'])
    rows = []
    for screen, part in coordinates.groupby('screen', sort=True):
        path = DATA / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            matrix, pairs, conditions = (saved['M'], saved['sps'].astype(str), saved['conds'].astype(str))
        index = {condition: i for i, condition in enumerate(conditions)}
        for coordinate in part.itertuples(index=False):
            pair = pairs[coordinate.pair_index]
            value = float(matrix[coordinate.pair_index, index[coordinate.condition]])
            rows.append(dict(screen=screen, pair_index=coordinate.pair_index, condition=coordinate.condition, original_subpair=pair, matrix_yield=value))
    coordinates = pd.DataFrame(rows)
    qualified = qualified.rename(columns={'condition_id': 'condition', 'subpair': 'original_subpair', 'yield': 'record_yield'})
    result = coordinates.merge(qualified[['screen', 'condition', 'original_subpair', 'record_id', 'source_file', 'source_sheet', 'source_row', 'record_yield']], on=['screen', 'condition', 'original_subpair'], how='left', validate='one_to_many')
    if result.record_id.isna().any():
        raise ValueError('matrix coordinate has no qualified record')
    groups = result.groupby(['screen', 'pair_index', 'condition'])
    result['qualified_records_in_coordinate'] = groups.record_id.transform('size')
    if not np.allclose(groups.record_yield.transform('mean'), result.matrix_yield, atol=1e-12, rtol=0):
        raise ValueError('qualified record aggregation disagrees with frozen matrix')
    result['numerator'], result['denominator'] = (1, result.qualified_records_in_coordinate)
    return result

def build_endpoint_lineage(atomic, units, contrasts):
    if atomic.duplicated(['contrast_id', 'pair_index']).any():
        raise ValueError('duplicate contrast x pair atom')
    context_map = {r.contrast_id: (r.screen, r.condition_low, r.condition_high) for r in contrasts.itertuples(index=False)}
    atoms, operators, summaries, weights_rows, coefficient_rows = ({}, {}, [], [], [])
    group_columns = bridge.GROUP
    for keys, group in units.groupby(group_columns, sort=True):
        mode, control, screen, family, key, background = keys
        key_column = bridge.MODES[mode]
        selected = contrasts[contrasts.screen.eq(screen) & contrasts[key_column].eq(key)]
        if control == 'exact_background':
            selected = selected[selected.background_key.eq(background)]
        cells = atomic[atomic.contrast_id.isin(selected.contrast_id)].copy()
        counts = cells.groupby(['sub_1', 'sub_2']).size().to_dict()
        for unit in group.itertuples(index=False):
            if unit.basis == 'full_pair':
                sub1, sub2 = json.loads(unit.unit_identity)
                part = cells[cells.sub_1.eq(sub1) & cells.sub_2.eq(sub2)]
            else:
                column = 'sub_1' if unit.basis == 'position_1' else 'sub_2'
                part = cells[cells[column].eq(unit.unit_identity)]
            n_partners = len(part[['sub_1', 'sub_2']].drop_duplicates())
            if not n_partners or n_partners != unit.n_partners or len(part) != unit.n_atomic_cells:
                raise ValueError('registered unit partner/atom population changed')
            atom = {(r.contrast_id, int(r.pair_index)): Fraction(1, n_partners * counts[r.sub_1, r.sub_2]) for r in part.itertuples(index=False)}
            if sum(atom.values()) != 1:
                raise ValueError('response atom weights do not normalize')
            reconstructed = sum((float(atom[r.contrast_id, int(r.pair_index)]) * r.delta for r in part.itertuples(index=False)))
            if not np.isclose(reconstructed, unit.observed, rtol=0, atol=1e-12):
                raise ValueError('reconstructed endpoint disagrees with legacy response')
            operator = expand_operator(atom, context_map)
            if sum(operator.values()) != 0:
                raise ValueError('yield difference operator does not annihilate constants')
            atoms[unit.unit_id], operators[unit.unit_id] = (atom, operator)
            summaries.append(dict(unit_id=unit.unit_id, mode=mode, background_control=control, screen=screen, family=family, perturbation_key=key, background_key=background, basis=unit.basis, unit_identity=unit.unit_identity, n_partners=n_partners, n_atoms=len(atom), n_yield_coordinates=len(operator), atom_weight_sum=1, yield_coefficient_sum=0, atom_signature=signature(atom), operator_signature=signature(operator), reconstructed_observed=reconstructed, saved_observed=unit.observed, absolute_reconstruction_error=abs(reconstructed - unit.observed), resolved=True))
            for (contrast, pair_index), weight in sorted(atom.items()):
                weights_rows.append((unit.unit_id, contrast, pair_index, weight.numerator, weight.denominator))
            for (scr, pair_index, condition), weight in sorted(operator.items()):
                coefficient_rows.append((unit.unit_id, scr, pair_index, condition, weight.numerator, weight.denominator))
    return (pd.DataFrame(summaries).sort_values('unit_id').reset_index(drop=True), pd.DataFrame(weights_rows, columns=['unit_id', 'contrast_id', 'pair_index', 'numerator', 'denominator']), pd.DataFrame(coefficient_rows, columns=['unit_id', 'screen', 'pair_index', 'condition', 'numerator', 'denominator']), atoms, operators)

def compare_maps(left, right):
    shared = set(left) & set(right)
    return dict(n_shared_atoms=len(shared), n_strict_only_atoms=len(set(left) - set(right)), n_relaxed_only_atoms=len(set(right) - set(left)), same_atom_support=set(left) == set(right), same_atom_weights=left == right, common_atom_weight_l1=float(sum((abs(left[k] - right[k]) for k in shared))))

def build_mode_matches(registry, atoms, operators):
    data = registry.copy()
    data['chemical_key'] = data.perturbation_key.map(chemical_key)
    keys = PAIR + ['background_control', 'background_key', 'basis', 'unit_identity', 'chemical_key']
    strict, relaxed = (data[data['mode'].eq('strict')], data[data['mode'].eq('dose_relaxed')])
    matched = strict.merge(relaxed, on=keys, suffixes=('_strict', '_dose_relaxed'), validate='many_to_one')
    rows = []
    metadata = ['source_dose_levels_json', 'target_dose_levels_json', 'source_partner_union_json', 'target_partner_union_json', 'source_contexts', 'target_contexts']
    for row in matched.to_dict('records'):
        item = {key: row[key] for key in keys}
        item['strict_perturbation_key'] = row['perturbation_key_strict']
        item['declared_metadata_match_report48'] = all((row[k + '_strict'] == row[k + '_dose_relaxed'] for k in metadata))
        for role in ('source', 'target'):
            left_id, right_id = (row[role + '_unit_id_strict'], row[role + '_unit_id_dose_relaxed'])
            item[role + '_unit_id_strict'], item[role + '_unit_id_dose_relaxed'] = (left_id, right_id)
            item.update({role + '_' + key: value for key, value in compare_maps(atoms[left_id], atoms[right_id]).items()})
            item[role + '_same_yield_operator'] = operators[left_id] == operators[right_id]
            item[role + '_operator_signature_strict'] = signature(operators[left_id])
            item[role + '_operator_signature_dose_relaxed'] = signature(operators[right_id])
        item['same_both_yield_operators'] = item['source_same_yield_operator'] and item['target_same_yield_operator']
        item['same_both_atom_weights'] = item['source_same_atom_weights'] and item['target_same_atom_weights']
        rows.append(item)
    return pd.DataFrame(rows).sort_values(keys + ['strict_perturbation_key']).reset_index(drop=True)

def build_collapsed(registry, cells, atoms, operators):
    summaries, maps = ([], {})
    for cell in cells.to_dict('records'):
        selected = atlas.select_rows(registry, cell, cell)
        for identity, part in selected.groupby('unit_identity', sort=True):
            item = {k: cell[k] for k in PAIR + atlas.CONFIG_KEYS}
            item.update(unit_identity=identity, n_registry_rows=len(part), source_value=float(part.source_conditioned.mean()), target_value=float(part.observed.mean()))
            for role in ('source', 'target'):
                ids = part[role + '_unit_id'].tolist()
                atom, operator = (average_maps([atoms[k] for k in ids]), average_maps([operators[k] for k in ids]))
                item[role + '_endpoint_ids_json'] = dumps(ids)
                item[role + '_atom_signature'], item[role + '_operator_signature'] = (signature(atom), signature(operator))
                item[role + '_n_atoms'], item[role + '_n_yield_coordinates'] = (len(atom), len(operator))
                maps[tuple((item[k] for k in PAIR + atlas.CONFIG_KEYS)) + (identity, role)] = (atom, operator)
            summaries.append(item)
    return (pd.DataFrame(summaries), maps)

def metric(frame, suffix, identities):
    subset = frame[frame.unit_identity.isin(identities)]
    if subset.empty:
        return (np.nan, np.nan)
    target, source = (subset['target_value_' + suffix], subset['source_value_' + suffix])
    return (float(100 * np.sqrt(np.mean((source - target) ** 2))), float(100 * np.sqrt(np.mean(target ** 2))))

def build_config_comparisons(collapsed, cells):
    keys = PAIR + CONFIG + ['unit_identity']
    left, right = (collapsed[collapsed['mode'].eq('strict')], collapsed[collapsed['mode'].eq('dose_relaxed')])
    common = left.merge(right, on=keys, suffixes=('_strict', '_dose_relaxed'), validate='one_to_one')
    for role in ('source', 'target'):
        for level in ('atom', 'operator'):
            common[role + '_same_' + level] = common[role + '_' + level + '_signature_strict'].eq(common[role + '_' + level + '_signature_dose_relaxed'])
    common['same_both_operators'] = common.source_same_operator & common.target_same_operator
    common['same_both_atom_weights'] = common.source_same_atom & common.target_same_atom
    summaries = []
    for strict_cell in cells[cells['mode'].eq('strict')].to_dict('records'):
        mask = pd.Series(True, index=common.index)
        lm, rm = (pd.Series(True, index=left.index), pd.Series(True, index=right.index))
        for key in PAIR + CONFIG:
            mask &= common[key].eq(strict_cell[key])
            lm &= left[key].eq(strict_cell[key])
            rm &= right[key].eq(strict_cell[key])
        part = common[mask]
        li, ri = (set(left[lm].unit_identity), set(right[rm].unit_identity))
        item = {k: strict_cell[k] for k in PAIR + CONFIG}
        item.update(n_strict_identities=len(li), n_relaxed_identities=len(ri), n_common_identities=len(part), n_strict_only_identities=len(li - ri), n_relaxed_only_identities=len(ri - li), n_same_both_operators=int(part.same_both_operators.sum()), n_same_both_atom_weights=int(part.same_both_atom_weights.sum()), n_same_target_operator=int(part.target_same_operator.sum()), n_same_source_operator=int(part.source_same_operator.sum()), n_target_changed=int((~part.target_same_operator).sum()), n_source_changed=int((~part.source_same_operator).sum()), same_entire_response_population=bool(len(part) > 0 and li == ri and part.same_both_operators.all()), same_entire_target_population=bool(len(part) > 0 and li == ri and part.target_same_operator.all()), strict_domain_state=strict_cell['domain_state'])
        for population, condition in (('both_operators_fixed', part.same_both_operators), ('target_operator_fixed', part.target_same_operator)):
            identities = set(part.loc[condition, 'unit_identity'])
            item[population + '_n_identities'] = len(identities)
            for mode in ('strict', 'dose_relaxed'):
                source_rmse, zero_rmse = metric(part, mode, identities)
                item[population + '_' + mode + '_source_rmse_pp'] = source_rmse
                item[population + '_' + mode + '_zero_rmse_pp'] = zero_rmse
        summaries.append(item)
    return (common, pd.DataFrame(summaries))

def build_outputs():
    atomic, units, registry, contrasts, cells, hashes = load_inputs()
    endpoints, weights, coefficients, atoms, operators = build_endpoint_lineage(atomic, units, contrasts)
    provenance = build_record_provenance(coefficients)
    matches = build_mode_matches(registry, atoms, operators)
    collapsed, _ = build_collapsed(registry, cells, atoms, operators)
    comparisons, configs = build_config_comparisons(collapsed, cells)
    coverage = registry[PAIR + atlas.BASE_KEYS + ['perturbation_key', 'background_key', 'unit_identity', 'source_unit_id', 'target_unit_id']].copy()
    for role in ('source', 'target'):
        coverage = coverage.merge(endpoints[['unit_id', 'atom_signature', 'operator_signature', 'n_atoms', 'n_yield_coordinates', 'resolved']].rename(columns={c: role + '_' + c for c in ['unit_id', 'atom_signature', 'operator_signature', 'n_atoms', 'n_yield_coordinates', 'resolved']}), on=role + '_unit_id', validate='many_to_one')
    return ({PREFIX + 'endpoints.csv.gz': endpoints, PREFIX + 'atom_weights.csv.gz': weights, PREFIX + 'record_provenance.csv': provenance, PREFIX + 'yield_coefficients.csv.gz': coefficients, PREFIX + 'registry_coverage.csv': coverage, PREFIX + 'mode_matches.csv': matches, PREFIX + 'collapsed_units.csv.gz': collapsed, PREFIX + 'common_identity_comparisons.csv.gz': comparisons, PREFIX + 'config_comparisons.csv': configs}, hashes)

def counts(outputs):
    get = lambda suffix: outputs[PREFIX + suffix]
    endpoints, matches, configs = (get('endpoints.csv.gz'), get('mode_matches.csv'), get('config_comparisons.csv'))
    common = get('common_identity_comparisons.csv.gz')
    return dict(registered_rows=len(get('registry_coverage.csv')), resolved_endpoints=len(endpoints), endpoint_atom_weight_rows=len(get('atom_weights.csv.gz')), endpoint_yield_coefficient_rows=len(get('yield_coefficients.csv.gz')), source_record_provenance_rows=len(get('record_provenance.csv')), used_yield_coordinates=len(get('record_provenance.csv')[['screen', 'pair_index', 'condition']].drop_duplicates()), maximum_qualified_records_per_coordinate=int(get('record_provenance.csv').qualified_records_in_coordinate.max()), cross_mode_registered_matches=len(matches), metadata_matches_report48=int(matches.declared_metadata_match_report48.sum()), registered_matches_same_both_operators=int(matches.same_both_yield_operators.sum()), registered_matches_same_both_atom_weights=int(matches.same_both_atom_weights.sum()), pair_configuration_comparisons=len(configs), collapsed_common_identities=len(common), collapsed_same_both_operators=int(common.same_both_operators.sum()), collapsed_same_target_operator=int(common.target_same_operator.sum()), nonempty_common_configs=int(configs.n_common_identities.gt(0).sum()), configs_with_identical_entire_response_population=int(configs.same_entire_response_population.sum()), configs_with_identical_entire_target_population=int(configs.same_entire_target_population.sum()), atom_different_operator_equal_registered=int((matches.same_both_yield_operators & ~matches.same_both_atom_weights).sum()), atom_different_operator_equal_collapsed=int((common.same_both_operators & ~common.same_both_atom_weights).sum()))

def make_figure(outputs):
    matches = outputs[PREFIX + 'mode_matches.csv']
    configs = outputs[PREFIX + 'config_comparisons.csv']
    common = outputs[PREFIX + 'common_identity_comparisons.csv.gz']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    categories = ['Both operators equal', 'Target only equal', 'Source only equal', 'Both operators changed']

    def categories_count(frame, source, target):
        return [int((frame[source] & frame[target]).sum()), int((~frame[source] & frame[target]).sum()), int((frame[source] & ~frame[target]).sum()), int((~frame[source] & ~frame[target]).sum())]
    colors = ['#438e73', '#e4ba69', '#6f8790', '#bf5c62']
    axes[0].barh(categories, categories_count(matches, 'source_same_yield_operator', 'target_same_yield_operator'), color=colors)
    axes[0].set(title='Registered chemical-identity matches', xlabel='Directed registry row matches')
    axes[1].barh(categories, categories_count(common, 'source_same_operator', 'target_same_operator'), color=colors)
    axes[1].set(title='After original identity-level averaging', xlabel='Common identity x pair x configuration')
    for relation, part in configs.groupby('relation'):
        axes[2].scatter(part.n_common_identities, part.n_same_both_operators, s=28, alpha=0.55, label=relation)
    upper = max(configs.n_common_identities.max(), 1)
    axes[2].plot([0, upper], [0, upper], color='gray', lw=0.8)
    axes[2].set(xlabel='Common molecular identities', ylabel='Identical source and target operators', title='Identity overlap versus response equivalence')
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.tick_params(axis='y', labelsize=8)
    for ax in axes[:2]:
        ax.bar_label(ax.containers[0], padding=3, fontsize=8)
        ax.margins(x=0.12)
    path = FIGURES / '41_RESPONSE_ESTIMAND_LINEAGE.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def main():
    outputs, hashes = build_outputs()
    files = []
    for name, frame in outputs.items():
        path = TABLES / name
        frame.to_csv(path, index=False)
        files.append(path)
    files += [make_figure(outputs), HERE / REPORT]
    if hashes != {name: digest(HERE / name) for name in hashes}:
        raise ValueError('input changed during reconstruction')
    manifest = dict(date='2026-10-17', scope='Retrospective exact response operator reconstruction; no causal mode effect or independent replication claim', input_sha256=hashes, code_sha256={name: digest(HERE / name) for name in ['response_estimand_lineage.py', 'test_response_estimand_lineage.py', 'verify_response_estimand_lineage.py']}, output_sha256={str(path.relative_to(HERE)): digest(path) for path in files}, counts=counts(outputs), definitions=dict(atom='contrast_id x pair_index, high minus low in frozen matrix', endpoint='distinct observed partners equal weight, observed contrasts within full substrate pair equal weight', operator='exact reduced rational signed yield coefficients on screen x pair_index x condition; zero coefficients removed', collapsed='equal registered perturbation/background rows within unit_identity after original both-endpoint partner filters', same_estimand='source and target yield operators both identical; full comparison also needs identical identity population and equal identity scoring weights', provenance='every used matrix coordinate mapped through original saved subpair to qualified record_id and source workbook/sheet/row, with explicit equal-record mean weights; excluded duplicate cells are not reinstated', limitation='conditional population response operators; no replicate-noise model or causal cross-mode contrast'))
    (DATA / (PREFIX + 'manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts'], indent=2))
if __name__ == '__main__':
    main()
