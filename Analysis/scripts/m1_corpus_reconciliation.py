"""Reconcile the M1 corpus and existing analysis supports without fitting models.

The production entry point reads MAIN outputs only. ``collect`` performs no
writes, so the same selectors can be checked against a read-only reference.
All displayed quantities have independently frozen numerical pins."""
from __future__ import annotations
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd
from m1_exclusions import assert_exclusions_reconciled, exclusion_summary, load_exclusions
from m1_reader_labels import reader_text
S = Path(__file__).resolve().parent
OUT = S.parent / 'out'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
SCREEN_EXPECTED = {'Acylation|1': (94, 596, 54874, 256, 23580, 0, 0), 'Buchwald-Hartwig(CMC)|1': (95, 50, 4673, 8, 760, 0, 0), 'Buchwald-Hartwig(CMC)|2': (95, 50, 4657, 6, 563, 0, 0), 'Buchwald-Hartwig(CMC)|3': (91, 50, 4519, 4, 354, 0, 0), 'Buchwald-Hartwig|1': (95, 213, 19823, 0, 0, 53, 4917), 'Buchwald-Hartwig|2': (89, 57, 5027, 0, 0, 14, 1239), 'C-O competitive reactions|1': (92, 50, 4557, 16, 1459, 0, 0), 'Chan-Lam|1': (32, 105, 3334, 0, 0, 26, 830), 'Chan-Lam|2': (33, 50, 1639, 0, 0, 0, 0), 'Esterification|1': (95, 50, 4656, 2, 187, 0, 0), 'Grignard|1': (46, 62, 2741, 0, 0, 16, 710), 'Heck|1': (90, 142, 12410, 61, 5345, 0, 0), 'Hiyama|1': (91, 74, 6647, 31, 2771, 0, 0), 'Kumada|1': (89, 126, 10917, 0, 0, 32, 2790), 'Reductive-amination|1': (47, 113, 5243, 48, 2224, 0, 0), 'SNAr|1': (47, 229, 10660, 0, 0, 57, 2637), 'Sonogashira|1': (95, 86, 7952, 0, 0, 22, 2040), 'Suzuki|1': (94, 242, 22002, 0, 0, 61, 5540), 'Ullmann|1': (95, 109, 10160, 47, 4393, 0, 0)}

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def select(frame: pd.DataFrame, **keys) -> pd.DataFrame:
    selected = frame
    for column, value in keys.items():
        selected = selected.loc[selected[column].eq(value)]
    require(not selected.empty, f'Empty selector: {keys}')
    return selected

def one(frame: pd.DataFrame, **keys) -> pd.Series:
    selected = select(frame, **keys)
    require(len(selected) == 1, f'Nonunique selector: {keys}; rows={len(selected)}')
    return selected.iloc[0]

def boolean(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower()
    require(values.isin(['true', 'false']).all(), f'Nonboolean values in {series.name}')
    return values.eq('true')

def table(headers: list[str], rows: list[list[str]]) -> str:

    def line(values):
        return '| ' + ' | '.join((str(v).replace('|', '\\|').replace('\n', ' ') for v in values)) + ' |'
    return '\n'.join([line(headers), line(['---'] * len(headers)), *[line(row) for row in rows]]) + '\n'

def cleanup_provenance(wells: pd.DataFrame, well_status_path: Path, exclusions=None) -> dict:
    """Reconcile reaction rows using the status writer's raw-row accounting."""
    exclusions = load_exclusions() if exclusions is None else exclusions
    path = Path(well_status_path).with_name('M1_well_measurement_status_exclusions.json')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    expected = exclusion_summary(exclusions.ids, exclusions)
    for key in ('source_sha256', 'id_sha256', 'count', 'deleted_wells', 'by_plate'):
        require(metadata.get(key) == expected[key], f'Well-status exclusion metadata mismatch: {key}')
    assert_exclusions_reconciled(metadata.get('matched_ids', []), exclusions)
    require('name0' in wells, 'Well-status corpus lacks measurement IDs')
    require(not wells.name0.isin(exclusions.ids).any(), 'Excluded measurement remains in retained corpus')
    require(metadata.get('retained_records') == len(wells), 'Retained reaction-record count mismatch')
    require(metadata.get('source_records') == len(wells) + metadata['deleted_wells'], 'Original source records do not equal excluded plus retained reaction records')
    return metadata

def collect(script_root: Path, well_status_path: Path):
    """Return CSV frames, tokenized Markdown and numerical pins; never write."""
    script_root, well_status_path = (Path(script_root), Path(well_status_path))
    reference_learning, recipe_options, response_learning = (script_root / 'reference_learning/Analysis', script_root / 'recipe_options/Analysis', script_root / 'response_learning')
    edm = response_learning / 'Existing_Data_Mining_2026-10-16'
    e59, e62 = (response_learning / 'Full_Edge_Corpus_E59', response_learning / 'Matched_Representation_E62')
    e66 = response_learning / 'Family_Information_Recovery_E66'
    pins, pin_map = ([], {})

    def source(path):
        if isinstance(path, (tuple, list)):
            return '; '.join(dict.fromkeys((source(part) for part in path)))
        path = Path(path)
        return 'out/' + path.name if path.parent == well_status_path.parent else 'scripts/' + path.relative_to(script_root).as_posix()

    def pin(item, actual, expected, path, selector):
        require(item not in pin_map, f'Duplicate pin: {item}')
        shown = f'{expected:,}' if isinstance(expected, int) else expected
        quantum = Decimal(shown.replace(',', ''))
        rounded = Decimal(str(actual)).quantize(Decimal(1).scaleb(quantum.as_tuple().exponent), rounding=ROUND_HALF_UP)
        require(rounded == quantum, f'Frozen count/value failed: {item}: {actual} vs {shown}')
        entry = dict(item=item, actual=int(actual) if isinstance(expected, int) else float(actual), shown=shown, source=source(path), selector=selector)
        pins.append(entry)
        pin_map[item] = entry
        return '{{' + item + '}}'
    reconciliation, recon_md = ([], [])

    def reconcile(label, available, used, unit, criterion, path, selector, key, expected, parent=None):
        require(0 <= used <= available, f'Invalid nested support: {label}: {used}/{available}')
        parent_selectors = {'recon_arrays19_records': 'all retained reaction records', 'recon_tables20_finite': 'all retained reaction records', 'recon_arrays19_finite': 'reaction_table != Finkelstein; count records', 'recon_recipe_options_tables20_strict': 'yield_numeric in [0,1]; count records', 'recon_recipe_options_arrays19_strict': 'dataset_id != Finkelstein; yield_numeric in [0,1]; count records', 'recon_response_learning_tables20_eligible_records': 'sum(n_yield_finite)', 'recon_response_learning_arrays19_eligible_records': 'reaction_type != finkelstein; sum(n_yield_finite)', 'recon_response_learning_arrays19_exact_swaps': 'sum(strict_swaps)', 'recon_response_learning_arrays19_typed_edges': 'sum(typed_edges)', 'recon_response_learning_arrays19_responses': 'sum(corpus_rows)', 'recon_response_learning_arrays19_multibackground_types': 'sum(multi_background_edges)', 'recon_response_learning_arrays19_multibackground_responses': 'sum(legacy_profile_rows)', 'recon_response_learning_arrays19_structure_types': 'reaction_type != finkelstein; n_backgrounds >= 2; count rows', 'recon_response_learning_family_supported': 'sum(n_all_original)', 'recon_response_learning_update_supported': 'sum(n_all_original)', 'recon_response_learning_arrays19_local_common8': 'sum(n_all_original)', 'recon_reference_learning_qualified_wells': 'reaction_table != Finkelstein; yield_fraction_0_1 present; count records'}
        available_path = well_status_path if key == 'recon_reference_learning_qualified_wells' else path
        available_selector = 'all retained reaction records' if key.startswith('recon_status_') else 'sum(n_observed)' if key.startswith('recon_reference_learning_') and key.endswith('_wells') and (key != 'recon_reference_learning_qualified_wells') else parent_selectors[key]
        parent_key = key + '_available'
        available_token = pin(parent_key, available, expected[0], available_path, available_selector)
        used_token = pin(key, used, expected[1], path, selector)
        difference_token = pin(key + '_difference', available - used, expected[0] - expected[1], [available_path, path], f'available ({available_selector}) minus used ({selector})')
        reconciliation.append(dict(population=label, available=int(available), used=int(used), difference=int(available - used), unit=unit, criterion=criterion, source=source([available_path, path]), selector=f'available: {available_selector}; used: {selector}; difference: available - used'))
        recon_md.append([f'{parent or label}: {available_token} {unit}', f'{label}: {used_token} {unit}', f'{criterion} Difference: {difference_token} {unit}.'])
        return used_token
    wells = pd.read_csv(well_status_path, low_memory=False)
    cleanup = cleanup_provenance(wells, well_status_path)
    cleanup_path = well_status_path.with_name('M1_well_measurement_status_exclusions.json')
    require(not wells.duplicated(['source_file', 'source_sheet', 'source_row']).any(), 'Duplicate raw-well identity')
    counts = wells.measurement_status.value_counts()
    expected_status = dict(quantified=360034, zero_by_rule=21654, detected_not_quantified=24286, no_screen_record=4613)
    require(set(counts.index) == set(expected_status), 'Unexpected measurement status')
    source_total = pin('recon_source_tables20_records', cleanup['source_records'], 410983, cleanup_path, 'source_records; reaction sheets only')
    excluded_total = pin('recon_excluded_records', cleanup['deleted_wells'], 396, cleanup_path, 'deleted_wells; exact authoritative measurement IDs')
    total = pin('recon_tables20_records', len(wells), 410587, well_status_path, 'all retained reaction records')
    cleanup_criterion = 'Apply the authoritative measurement-ID exclusion list before yield processing or analysis.'
    reconciliation.append(dict(population='Retained reaction records', available=cleanup['source_records'], used=len(wells), difference=cleanup['deleted_wells'], unit='records', criterion=cleanup_criterion, source=source([cleanup_path, well_status_path]), selector='available: source_records; used: retained reaction rows; difference: deleted_wells'))
    recon_md.append([f'Original source: {source_total} records', f'Retained corpus: {total} records', f'{cleanup_criterion} Excluded: {excluded_total} records.'])
    ntables = pin('recon_tables20_count', wells.source_file.nunique(), 20, well_status_path, 'nunique(source_file)')
    arrays = wells.loc[wells.reaction_table.ne('Finkelstein')]
    narrays = pin('recon_arrays19_count', arrays.source_file.nunique(), 19, well_status_path, 'reaction_table != Finkelstein; nunique(source_file)')
    array_records = reconcile('Coupling-array records', len(wells), len(arrays), 'records', 'Exclude the Finkelstein table, which has one substrate role.', well_status_path, 'reaction_table != Finkelstein', 'recon_arrays19_records', (410587, 406909), 'Retained tables')
    finite = wells.yield_fraction_0_1.notna()
    require(finite.equals(wells.measurement_status.isin(['quantified', 'zero_by_rule'])), 'Finite-yield/status mismatch')
    require(wells.loc[wells.measurement_status.eq('zero_by_rule'), 'yield_fraction_0_1'].eq(0).all(), 'Nonzero rule-assigned yield')
    numeric = reconcile('Numerical yields', len(wells), int(finite.sum()), 'records', 'Retain recorded numerical yields and blank yields with a negative product-detection call.', well_status_path, 'measurement_status in [quantified, zero_by_rule]', 'recon_tables20_finite', (410587, 381688))
    for status, expected in expected_status.items():
        reconcile(status, len(wells), int(counts[status]), 'records', {'quantified': 'A numerical yield is recorded.', 'zero_by_rule': 'Yield is blank and the product-detection call is negative; assign zero.', 'detected_not_quantified': 'Yield is blank and product is detected; retain a missing numerical yield.', 'no_screen_record': 'Yield and the product-detection call are both absent; retain a missing numerical yield.'}[status], well_status_path, f'measurement_status == {status}', 'recon_status_' + status, (410587, expected))
    reconcile('Numerical yields in coupling arrays', len(arrays), int(arrays.yield_fraction_0_1.notna().sum()), 'records', 'Apply the numerical-yield rule within coupling arrays.', well_status_path, 'reaction_table != Finkelstein; numerical yield present', 'recon_arrays19_finite', (406909, 378070))
    recipe_options_path = recipe_options / 'data/normalized_records.csv.gz'
    records = pd.read_csv(recipe_options_path, usecols=['dataset_id', 'strict_eligible'])
    require(len(records) == len(wells), 'RECIPE_OPTIONS raw-record count differs from well-status corpus')
    strict = boolean(records.strict_eligible)
    recipe_options_manifest = json.loads((recipe_options / 'data/mapping_summary.json').read_text(encoding='utf-8'))
    require(int(strict.sum()) == recipe_options_manifest['strict_rows'], 'RECIPE_OPTIONS strict-count summary mismatch')
    recipe_options_criterion = 'Require recipe metadata agreement, no flagged registry/campaign issue, numerical yield in range, complete equivalents and a recorded plate.'
    reconcile('RECIPE_OPTIONS strict records, all tables', int(finite.sum()), int(strict.sum()), 'records', recipe_options_criterion, recipe_options_path, 'strict_eligible == True', 'recon_recipe_options_tables20_strict', (381688, 380910), 'Numerical yields')
    reconcile('RECIPE_OPTIONS strict records, coupling arrays', int(arrays.yield_fraction_0_1.notna().sum()), int((strict & records.dataset_id.ne('Finkelstein')).sum()), 'records', recipe_options_criterion, recipe_options_path, 'strict_eligible == True; dataset_id != Finkelstein', 'recon_recipe_options_arrays19_strict', (378070, 377292))
    class_path = e59 / 'results/full_edge_corpus_class_support_20.csv'
    classes = pd.read_csv(class_path)
    require(classes.reaction_type.is_unique and len(classes) == 20, 'E59 table support identities')
    require(int(classes.raw_observation_rows.sum()) == len(wells), 'E59 raw-count mismatch')
    require(int(classes.n_yield_finite.sum()) == int(finite.sum()), 'E59 finite-count mismatch')
    coupling = classes.loc[classes.reaction_type.ne('finkelstein')]
    edge_path = e59 / 'results/full_edge_corpus_edge_support.csv'
    edges = pd.read_csv(edge_path)
    require(not edges.duplicated(EDGE).any(), 'Duplicate typed swap support')
    registry_path = e59 / 'results/full_edge_strict_swap_registry.csv'
    swaps = pd.read_csv(registry_path)
    require(swaps.swap_id.is_unique, 'Duplicate exact swap identity')
    for column, actual in [('typed_edges', len(edges)), ('strict_swaps', len(swaps)), ('corpus_rows', int(edges.corpus_rows.sum()))]:
        require(int(classes[column].sum()) == actual, f'E59 support mismatch: {column}')
    response_learning_criterion = 'Require a numerical yield in range, complete registry-matched recipe, both substrate roles where designed, and no product conflict within a condition cell.'
    reconcile('RESPONSE_LEARNING exact-matching records, all tables', int(finite.sum()), int(classes.n_eligible_raw_observations.sum()), 'records', response_learning_criterion, class_path, 'sum(n_eligible_raw_observations)', 'recon_response_learning_tables20_eligible_records', (381688, 381686))
    reconcile('RESPONSE_LEARNING exact-matching records, coupling arrays', int(arrays.yield_fraction_0_1.notna().sum()), int(coupling.n_eligible_raw_observations.sum()), 'records', response_learning_criterion, class_path, 'reaction_type != finkelstein; sum(n_eligible_raw_observations)', 'recon_response_learning_arrays19_eligible_records', (378070, 378068))
    descriptions = {'strict_swaps': ('Exact recipe swaps', 'exact recipe swaps', 'Change exactly one component, preserve every other recipe slot and match at least twenty common substrate pairs with agreeing products.', 'exact_swaps', 2135, 2021), 'typed_edges': ('Typed component swaps', 'typed component swaps', 'Group exact recipe swaps by component pair and the slot exchanged.', 'typed_edges', 1550, 1475), 'corpus_rows': ('Canonical responses', 'responses', 'One yield difference per typed swap, substrate pair and background; collapse duplicate appearances of the same response.', 'responses', 828658, 824473), 'multi_background_edges': ('Multiple-background swaps', 'typed component swaps', 'Require at least two observed backgrounds for the same typed component swap.', 'multibackground_types', 303, 293), 'legacy_profile_rows': ('Multiple-background responses', 'responses', 'Retain canonical responses belonging to multiple-background typed component swaps.', 'multibackground_responses', 267462, 265714)}
    for column, (label, unit, criterion, key, expected_all, expected_array) in descriptions.items():
        reconcile(label + ', coupling arrays', int(classes[column].sum()), int(coupling[column].sum()), unit, criterion + ' The difference is the Finkelstein contribution.', class_path, f'reaction_type != finkelstein; sum({column})', 'recon_response_learning_arrays19_' + key, (expected_all, expected_array), label + ', all tables')
    ce = edges.loc[edges.reaction_type.ne('finkelstein')]
    require(int(ce.corpus_rows.sum()) == int(coupling.corpus_rows.sum()), 'Array response support mismatch')
    require(int(ce.unique_backgrounds.ge(2).sum()) == int(coupling.multi_background_edges.sum()), 'Multiple-background typed support mismatch')
    require(int(ce.loc[ce.unique_backgrounds.ge(2), 'corpus_rows'].sum()) == int(coupling.legacy_profile_rows.sum()), 'Multiple-background response support mismatch')
    require(int(swaps.reaction_type.ne('finkelstein').sum()) == int(coupling.strict_swaps.sum()), 'Array exact-swap support mismatch')
    bridge_path = e62 / 'results/e62_support_bridge_1548.csv'
    bridge = pd.read_csv(bridge_path)
    require(not bridge.duplicated(EDGE).any(), 'Duplicate E62 typed swap')
    eligible = bridge.loc[bridge.panel_status.eq('eligible')]
    require(eligible.n_backgrounds.ge(3).all() and eligible.n_candidate_vectors.ge(20).all(), 'E62 eligibility definition mismatch')
    require(eligible.n_candidate_vectors.eq(eligible.n_scored_vectors).all(), 'E62 candidate/scored coverage mismatch')
    require(not eligible.reaction_type.eq('finkelstein').any(), 'Finkelstein entered the two-substrate structure panel')
    structure_summary = pd.read_csv(e62 / 'results/e62_support_classes_20.csv')
    require(int(structure_summary.n_eligible_edges.sum()) == len(eligible) and int(structure_summary.n_scored_vectors.sum()) == int(eligible.n_scored_vectors.sum()), 'E62 table/typed-swap support mismatch')
    structure_types = reconcile('Structure-model swaps', int(coupling.multi_background_edges.sum()), len(eligible), 'typed component swaps', 'Require at least three backgrounds and twenty complete substrate-pair vectors with both actual substrate structures and valid fingerprints.', bridge_path, 'panel_status == eligible; reaction_type != finkelstein', 'recon_response_learning_arrays19_structure_types', (293, 93))
    vectors = pin('recon_response_learning_arrays19_structure_vectors', int(eligible.n_scored_vectors.sum()), 25956, bridge_path, 'panel_status == eligible; sum(n_scored_vectors)')
    model_arrays = pin('recon_response_learning_structure_arrays', eligible.reaction_type.nunique(), 14, bridge_path, 'panel_status == eligible; nunique(reaction_type)')
    reconciliation.append(dict(population='Complete scored response vectors', available=int(eligible.n_candidate_vectors.sum()), used=int(eligible.n_scored_vectors.sum()), difference=0, unit='response vectors', criterion='A substrate pair has a numerical response at every background of an eligible typed swap and valid structures for both substrate roles.', source=source(bridge_path), selector='panel_status == eligible; sum(n_scored_vectors)'))
    recon_md.append([f'Eligible structure panel: {structure_types} typed component swaps', f'{vectors} complete scored response vectors across {model_arrays} arrays', "A vector contains one substrate pair's responses at every background of its typed swap; this changes the unit from responses to vectors."])
    e66_support_path = e66 / 'results/e66_support_classes_20.csv'
    support = pd.read_csv(e66_support_path)
    require(int(support.n_all_original.sum()) == int(eligible.n_scored_vectors.sum()), 'E66/E62 vector count mismatch')
    reconcile('History-supported vectors', int(support.n_all_original.sum()), int(support.n_family_supported.sum()), 'response vectors', 'Require at least one historical training vector outside both query substrate families.', e66_support_path, 'sum(n_family_supported)', 'recon_response_learning_family_supported', (25956, 25520))
    reconcile('History and local-update supported vectors', int(support.n_all_original.sum()), int(support.n_update_supported.sum()), 'response vectors', 'Require at least one historical training vector outside both query substrate families and at least one local calibration profile sharing a query substrate family.', e66_support_path, 'sum(n_update_supported)', 'recon_response_learning_update_supported', (25956, 24943))
    common8 = reconcile('Common local-update queries', int(support.n_all_original.sum()), int(support.n_common8.sum()), 'response vectors', 'Require at least one historical training vector outside both query substrate families and at least eight local calibration profiles sharing a query substrate family; all compared learners share these queries.', e66_support_path, 'sum(n_common8)', 'recon_response_learning_arrays19_local_common8', (25956, 24221))
    manifest_path = reference_learning / 'data/screen_reconstruction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['raw_rows'] == len(wells) and manifest['finite_yields'] == int(finite.sum()), 'REFERENCE_LEARNING raw/finite corpus mismatch')
    pin('recon_reference_learning_duplicate_rows', manifest['duplicate_rows_excluded'], 338, manifest_path, 'duplicate_rows_excluded')
    inventory_path = reference_learning / 'tables/screen_inventory.csv'
    inventory = pd.read_csv(inventory_path)
    partition_path = reference_learning / 'data/screen_partition_registry.csv'
    partitions = pd.read_csv(partition_path)
    observation_path = reference_learning / 'data/qualified_observations.csv.gz'
    observed = pd.read_csv(observation_path, usecols=['screen', 'partition', 'subpair', 'source_file'])
    require(not partitions.duplicated(['screen', 'subpair']).any(), 'REFERENCE_LEARNING pair occurs in multiple partitions')
    require(set(partitions.partition) == {'analysis', 'reserve', 'sealed'}, 'Unexpected REFERENCE_LEARNING partition')
    actual_partition = observed.groupby(['screen', 'subpair', 'partition']).size().rename('actual_n_observed').reset_index()
    check = partitions.merge(actual_partition, on=['screen', 'subpair', 'partition'], how='outer', validate='one_to_one', indicator=True)
    require(check._merge.eq('both').all() and check.n_observed.eq(check.actual_n_observed).all(), 'REFERENCE_LEARNING partition/observation count mismatch')
    parent_map = observed[['screen', 'source_file']].drop_duplicates()
    require(parent_map.screen.is_unique, 'REFERENCE_LEARNING screen maps to multiple raw tables')
    raw_inventory_path = reference_learning / 'data/raw_file_inventory.csv'
    raw_inventory = pd.read_csv(raw_inventory_path)
    raw_inventory = raw_inventory.loc[raw_inventory.table_kind.eq('reaction'), ['source_file', 'reaction_class', 'n_rows']]
    require(raw_inventory.source_file.is_unique and raw_inventory.n_rows.sum() == len(wells), 'REFERENCE_LEARNING raw-file inventory mismatch')
    mapping = inventory[['screen', 'conditions', 'all_pairs', 'all_wells', 'analysis_pairs', 'analysis_wells']].merge(parent_map.merge(raw_inventory, on='source_file', how='left', validate='many_to_one'), on='screen', validate='one_to_one')
    require(mapping.reaction_class.notna().all() and (not mapping.reaction_class.eq('Finkelstein').any()), 'REFERENCE_LEARNING parent-array mapping incomplete')
    mapping = mapping.rename(columns={'reaction_class': 'parent_array'})
    require(set(mapping.screen) == set(SCREEN_EXPECTED) and len(mapping) == manifest['qualifying_screens'], 'REFERENCE_LEARNING screen identities changed')
    mapping = mapping.set_index('screen').loc[list(SCREEN_EXPECTED)].reset_index()
    mapping_md = []
    for index, row in mapping.iterrows():
        expected = SCREEN_EXPECTED[row.screen]
        tokens = []
        for j, column in enumerate(['conditions', 'analysis_pairs', 'analysis_wells', 'reserve_pairs', 'reserve_wells', 'sealed_pairs', 'sealed_wells']):
            if column.startswith(('reserve', 'sealed')):
                part, metric = column.split('_')
                block = partitions.loc[partitions.screen.eq(row.screen) & partitions.partition.eq(part)]
                value = len(block) if metric == 'pairs' else int(block.n_observed.sum())
                mapping.loc[index, column] = value
            else:
                value = int(row[column])
            tokens.append(pin(f'recon_reference_learning_screen_{index + 1:02d}_{column}', int(value), expected[j], inventory_path if column == 'conditions' else partition_path, f'screen == {row.screen}; conditions' if column == 'conditions' else f"screen == {row.screen}; partition == {column.split('_')[0]}; " + ('count rows' if column.endswith('pairs') else 'sum(n_observed)')))
        analysis_partition = partitions.loc[partitions.screen.eq(row.screen) & partitions.partition.eq('analysis')]
        require(len(analysis_partition) == int(row.analysis_pairs) and int(analysis_partition.n_observed.sum()) == int(row.analysis_wells), 'REFERENCE_LEARNING per-screen analysis partition mismatch: ' + row.screen)
        require(int(row.all_pairs) == sum((int(mapping.loc[index, c]) for c in ['analysis_pairs', 'reserve_pairs', 'sealed_pairs'])), 'REFERENCE_LEARNING screen pair partition mismatch')
        require(int(row.all_wells) == sum((int(mapping.loc[index, c]) for c in ['analysis_wells', 'reserve_wells', 'sealed_wells'])), 'REFERENCE_LEARNING screen well partition mismatch')
        mapping_md.append([row.screen, row.parent_array, f'{tokens[1]} / {tokens[2]}', f'{tokens[3]} / {tokens[4]}', f'{tokens[5]} / {tokens[6]}'])
    integer_columns = ['conditions', 'analysis_pairs', 'analysis_wells', 'reserve_pairs', 'reserve_wells', 'sealed_pairs', 'sealed_wells']
    mapping[integer_columns] = mapping[integer_columns].astype(int)
    screen_count = pin('recon_reference_learning_screens', len(mapping), 19, inventory_path, 'nunique(screen)')
    parent_count = pin('recon_reference_learning_parent_arrays', mapping.parent_array.nunique(), 15, observation_path, 'unique screen-to-source_file parents')
    unrepresented = sorted(set(raw_inventory.reaction_class) - {'Finkelstein'} - set(mapping.parent_array))
    require(unrepresented == ['Allylic-substitution', 'Minisci', 'Pd-carbonylation', 'Suzuki Bpin'], 'REFERENCE_LEARNING arrays without a qualified screen changed')
    unrepresented_count = pin('recon_reference_learning_arrays_without_screen', len(unrepresented), 4, raw_inventory_path, 'reaction tables excluding Finkelstein minus mapped qualified-screen parent arrays')
    qualified = reconcile('REFERENCE_LEARNING qualified-screen wells', int(arrays.yield_fraction_0_1.notna().sum()), len(observed), 'records', 'Exclude duplicated condition cells; require independent condition-signature blocks, a dense numerical-yield core, sufficient substrate pairs and conditions, valid first-substrate fingerprints, and removal of registered QC twin columns.', observation_path, 'all qualified observations', 'recon_reference_learning_qualified_wells', (378070, 258830))
    for part, expected_pairs, expected_wells in [('analysis', 2454, 196491), ('reserve', 479, 41636), ('sealed', 281, 20703)]:
        block = partitions.loc[partitions.partition.eq(part)]
        reconcile('REFERENCE_LEARNING ' + part + ' wells', len(observed), int(block.n_observed.sum()), 'records', {'analysis': 'Use the analysis substrate-pair partition: similarity-based holdout in designated screens, otherwise the seeded sample with the minimum substrate-pair floor.', 'reserve': 'Retain the unallocated substrate pairs from screens with seeded analysis sampling.', 'sealed': 'Hold out the least similar substrate pairs in designated screens, using similarity from both substrate roles.'}[part], partition_path, f'partition == {part}; sum(n_observed)', 'recon_reference_learning_' + part + '_wells', (258830, expected_wells))
        pin('recon_reference_learning_' + part + '_pairs', len(block), expected_pairs, partition_path, f'partition == {part}; row count')
    require(int(mapping.analysis_pairs.sum()) == manifest['analysis_pairs'] and int(mapping.analysis_wells.sum()) == manifest['analysis_wells'], 'REFERENCE_LEARNING analysis manifest mismatch')
    broad, broad_md = ([], [])

    def estimate(evidence, support_label, model, value, expected, unit, path, selector, key, n_edges=None, n_scores=None):
        token = pin(key, value, expected, path, selector)
        broad.append(dict(evidence=evidence, support=support_label, model=model, estimate=float(value), unit=unit, n_edges=n_edges, n_scores=n_scores, source=source(path), selector=selector))
        return token
    copy_path = reference_learning / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv'
    copy = select(pd.read_csv(copy_path), level='headline')
    marginal = one(copy, model='marginal').mse_pp2
    broad_path = reference_learning / 'tables/chemically_aligned_measurements_headline.csv'
    broad_copy = one(pd.read_csv(broad_path), split='pair', fraction=1, scope='primary', weighting='context_balanced', macro='family_macro')
    wide_edges = pin('broad_copy_edges', broad_copy.n_edges, 663, broad_path, 'split=pair; fraction=1; scope=primary; context_balanced; family_macro; n_edges')
    wide_scores = pin('broad_copy_scores', broad_copy.target_score_count, 326004, broad_path, 'split=pair; fraction=1; scope=primary; context_balanced; family_macro; target_score_count')
    main_copy_row = one(copy, model='direct_raw')
    pin('broad_copy_main_edges', main_copy_row.n_edges, 655, copy_path, 'level=headline; model=direct_raw; n_edges')
    pin('broad_copy_main_scores', main_copy_row.target_score_count, 242766, copy_path, 'level=headline; model=direct_raw; target_score_count')
    for model, label, column, expected_main, expected_broad, sign in [('direct_raw', 'Copying: error increase', 'direct_raw_recovery_vs_marginal', '42.4', '45.8', -1), ('semantic_full_raw', 'Learning: error decrease', 'semantic_recovery_vs_marginal', '13.7', '12.3', 1)]:
        r = one(copy, model=model)
        main_value = sign * 100 * (1 - r.mse_pp2 / marginal)
        main_token = estimate(label, 'complete panels', model, main_value, expected_main, '%', copy_path, f'level=headline; model={model}; relative to marginal', 'broad_' + model + '_main', int(r.n_edges), int(r.target_score_count))
        wide_token = estimate(label, 'all scored edges', model, sign * 100 * broad_copy[column], expected_broad, '%', broad_path, f'split=pair; fraction=1; scope=primary; context_balanced; family_macro; {column}', 'broad_' + model + '_wide', int(broad_copy.n_edges), int(broad_copy.target_score_count))
        broad_md.append([label, main_token + '%', wide_token + '%', f'{wide_edges} edges; {wide_scores} target scores', 'Retain scored edges without requiring a complete comparison panel.'])
    specificity_path = reference_learning / 'tables/chemical_observation_specificity_headline.csv'
    specificity = pd.read_csv(specificity_path)
    ranges = []
    for scope, tag, expected_min, expected_max in [('all_panels_complete', 'main', '8.94', '11.90'), ('primary', 'wide', '8.0', '11.2')]:
        selected = select(specificity, domain='strict_641', scope=scope, weighting='context_balanced', macro='family_macro')
        require(len(selected) == 6 and (not selected.duplicated(['split', 'fraction']).any()), 'Specificity split/history settings changed')
        bounds = []
        for suffix, value, expected in [('min', 100 * selected.other_recovery_vs_control.min(), expected_min), ('max', 100 * selected.other_recovery_vs_control.max(), expected_max)]:
            bounds.append(estimate('Same-swap reference: error decrease vs matched control', scope, suffix, value, expected, '%', specificity_path, f'domain=strict_641; scope={scope}; context_balanced; family_macro; {suffix}(100*other_recovery_vs_control)', 'broad_specificity_' + tag + '_' + suffix))
        ranges.append(' to '.join(bounds) + '%')
    broad_md.append(['Same-swap reference vs matched control', *ranges, 'Across the existing split and history settings', 'Retain scored comparisons without the complete-panel requirement.'])
    cross_path = reference_learning / 'tables/cross_library_source_channels_headline.csv'
    cross = pd.read_csv(cross_path)
    directions = pin('broad_cross_scored_directions', one(cross, domain='all_scored', relation='cross_family', macro='direction_macro', channel='d', budget=192).n_scored_directions, 28, cross_path, 'domain=all_scored; relation=cross_family; direction_macro; channel=d; budget=192; n_scored_directions')
    for column, label, expected_main, expected_wide, sign in [('delta_test', 'Correct correspondence minus reassignment: error reduction', '-8.66', '-5.03', 1), ('net_gain', 'Cross-array transfer: excess error over target-only calibration', '17.31', '9.40', -1)]:
        values = []
        for domain, tag, expected in [('b24_fixed', 'main', expected_main), ('all_scored', 'wide', expected_wide)]:
            r = one(cross, domain=domain, relation='cross_family', macro='direction_macro', channel='d', budget=192)
            values.append(estimate(label, domain, 'cross_family', sign * 10000.0 * r[column], expected, 'pp2', cross_path, f'domain={domain}; relation=cross_family; direction_macro; channel=d; budget=192; {column}', 'broad_cross_' + column + '_' + tag))
        broad_md.append([label, values[0] + ' pp2', values[1] + ' pp2', directions + ' scored directions', 'Retain every scored cross-array direction at the same calibration budget.'])
    recovery_path = e66 / 'results/e66_summary.csv'
    recovery = pd.read_csv(recovery_path, dtype={'budget': str})
    for model, label, main_expected, wide_expected, wide_n in [('joint', 'History plus local calibration', '195.30', '190.49', 25956), ('calibration_only', 'Local calibration alone', '216.29', '214.35', 25379), ('frozen_family', 'History alone', '200.90', '192.34', 25520)]:
        values, query_token = ([], None)
        for population, tag, expected in [('common8', 'main', main_expected), ('all_original', 'wide', wide_expected)]:
            r = one(recovery, population=population, target_geometry='raw_delta', budget='8', learner=model, stratum='all', weighting='equal_class_edge_balanced')
            require(int(r.n_test_edges) == len(eligible) and int(r.n_test_classes) == eligible.reaction_type.nunique(), 'E66 model edge/array support mismatch')
            count_token = pin('broad_recovery_' + model + '_' + tag + '_queries', r.n_test_vectors, 24221 if tag == 'main' else wide_n, recovery_path, f'population={population}; raw_delta; budget=8; learner={model}; all; equal_class_edge_balanced; n_test_vectors')
            values.append(estimate(label, population, model, r.native_vector_mse, expected, 'pp2', recovery_path, f'population={population}; raw_delta; budget=8; learner={model}; all; equal_class_edge_balanced; native_vector_mse', 'broad_recovery_' + model + '_' + tag, int(r.n_test_edges), int(r.n_test_vectors)))
            if tag == 'wide':
                query_token = count_token
        broad_md.append([label + ': prediction error', values[0] + ' pp2', values[1] + ' pp2', query_token + ' queries', 'Each learner uses its own widest scored support; the common-support comparison uses ' + common8 + ' queries.'])
    sign_path = edm / 'results/sparse_sign_feasibility_summary.csv'
    signs = pd.read_csv(sign_path)
    solutions = pd.read_csv(edm / 'results/sparse_sign_feasibility_solutions.csv')
    threshold_token = pin('broad_sign_large_threshold_pp', one(signs, population='all', panel='observed_sparse', objective='count', stratum='ALL', threshold_pp=20).threshold_pp, 20, sign_path, 'population=all; observed_sparse; objective=count; stratum=ALL; threshold_pp=20')
    sign_display = []
    for threshold, expected_19, expected_20 in [(0, (283, 271), (293, 281)), (20, (212, 143), (218, 148))]:
        total_sign = one(signs, population='all', panel='observed_sparse', objective='count', stratum='ALL', threshold_pp=threshold)
        fink = one(signs, population='all', panel='observed_sparse', objective='count', stratum='finkelstein', threshold_pp=threshold)
        actual = select(solutions, population='all', panel='observed_sparse', objective='count', threshold_pp=threshold)
        actual = actual.loc[actual.reaction_type.ne('finkelstein')]
        require(not actual.duplicated(EDGE).any() and len(actual) == int(coupling.multi_background_edges.sum()), 'Sign-test typed support mismatch')
        require(int(actual.cycle_rank.gt(0).sum()) == total_sign.n_cycle_edges - fink.n_cycle_edges, 'Sign-test cycle summary mismatch')
        require(int(actual.n_mismatched.gt(0).sum()) == total_sign.n_conflicting_edges - fink.n_conflicting_edges, 'Sign-test conflict summary mismatch')
        tokens = {}
        for size, expected in [(19, expected_19), (20, expected_20)]:
            for column, suffix, target in [('n_cycle_edges', 'cyclic', expected[0]), ('n_conflicting_edges', 'conflicting', expected[1])]:
                value = total_sign[column] - (fink[column] if size == 19 else 0)
                tokens[size, suffix] = estimate('Sign conflicts', ('coupling arrays' if size == 19 else 'all raw tables') + '; observed sparse support', suffix, value, target, 'typed component swaps', sign_path, f'population=all; panel=observed_sparse; objective=count; threshold_pp={threshold}; ALL' + (' minus finkelstein' if size == 19 else ''), f'broad_sign_{size}_{threshold}_{suffix}', int(actual.shape[0]) if size == 19 else int(total_sign.n_retained_edges))
        sign_display.append(['Sign conflicts' + (': response magnitude at least ' + threshold_token + ' pp' if threshold else ': all nonzero responses'), tokens[19, 'conflicting'] + ' / ' + tokens[19, 'cyclic'], tokens[19, 'conflicting'] + ' / ' + tokens[19, 'cyclic'], 'All eligible multiple-background swaps in coupling arrays', 'Full observed sparse support; the all-table supplement is ' + tokens[20, 'conflicting'] + ' / ' + tokens[20, 'cyclic'] + '.'])
    broad_md.extend(sign_display)
    component_path = response_learning / 'Response_Information_Decomposition_E63/results/e63_paired_component_summary.csv'
    components = select(pd.read_csv(component_path), comparison='centroid_minus_codebook_continuous')
    require(len(components) == 6, 'Structure decomposition settings changed')
    shares = 100 * components.offset_difference_pp2 / components.mse_difference_pp2
    share_tokens = [estimate("Structure gain from the pair's own mean response", 'full structure panel', suffix, value, expected, '%', component_path, 'comparison=centroid_minus_codebook_continuous; ' + suffix + '(100*offset_difference_pp2/mse_difference_pp2)', 'broad_structure_' + suffix) for suffix, value, expected in [('min', shares.min(), '91.1'), ('max', shares.max(), '96.6')]]
    broad_md.append(["Structure gain from the pair's own mean response", ' to '.join(share_tokens) + '%', ' to '.join(share_tokens) + '%', vectors + ' complete response vectors', 'The main result already uses the full eligible structure panel.'])
    conflict_path = recipe_options / 'round29/option_scope_domains/all_collective_conflict_domains.csv'
    conflicts = pd.read_csv(conflict_path)
    conflict_token = estimate('Shared-scope conflicts', 'full enumerated support', 'all views', len(conflicts), 542, 'conflict views', conflict_path, 'all rows', 'broad_shared_scope_conflicts')
    broad_md.append(['Shared-scope conflicts', conflict_token, conflict_token, 'All enumerated conflict views', 'The main count already uses the full enumerated support.'])
    recon_text = f'A table is one of the {ntables} original reaction files. The original source contains {source_total} reaction records; excluding {excluded_total} listed wells leaves {total} retained reaction records. Auxiliary workbook sheets are not reaction records. An array is one of the {narrays} retained tables remaining after Finkelstein is excluded ({array_records} records). The numerical-yield corpus contains {numeric} records. A screen is a qualified REFERENCE_LEARNING subset; {screen_count} screens derive from {parent_count} arrays.\n\n' + table(['Corpus / parent support', 'Used in this analysis', 'Criterion accounting for difference'], recon_md) + '\nDifferences are calculated only within the same unit. A response can appear in several exact recipe swaps; canonical responses collapse those appearances. Complete vectors collect responses across backgrounds and are therefore a separate support unit. REFERENCE_LEARNING partition wells count observed numerical-yield records, not imputed cells.\n'
    mapping_text = f'The {screen_count} reconstructed screens derive from {parent_count} parent arrays. Pair / well counts refer to substrate pairs / observed numerical-yield wells in each partition.\n\n' + table(['Screen', 'Parent array', 'Analysis pairs / wells', 'Reserve pairs / wells', 'Sealed pairs / wells'], mapping_md) + '\nAcross all qualified screens, the analysis partition contains {{recon_reference_learning_analysis_pairs}} pairs / {{recon_reference_learning_analysis_wells}} wells; the reserve contains {{recon_reference_learning_reserve_pairs}} pairs / {{recon_reference_learning_reserve_wells}} wells; the sealed partition contains {{recon_reference_learning_sealed_pairs}} pairs / {{recon_reference_learning_sealed_wells}} wells.\n' + f'\nThe {unrepresented_count} other arrays have no qualified REFERENCE_LEARNING screen under the same reconstruction rules: ' + ', '.join(unrepresented) + '.\n'
    broad_text = 'These summaries use the existing predictions and measured-response outputs. Error reduction is positive when the named procedure lowers error; excess error is positive when transfer is worse than target-only calibration. The widest scored supports for the history and local-calibration learners differ and do not form a paired model comparison.\n\n' + table(['Evidence', 'Main support', 'Widest support', 'Widest denominator', 'Support definition'], broad_md)
    frames = {'corpus_reconciliation': pd.DataFrame(reconciliation), 'reference_learning_screen_array_mapping': mapping[['screen', 'parent_array', 'source_file', *integer_columns]], 'broadest_support': pd.DataFrame(broad)}
    texts = dict(corpus_reconciliation=recon_text, reference_learning_screen_array_mapping=mapping_text, broadest_support=broad_text)
    return (frames, texts, pins)

def main():
    frames, texts, pins = collect(S, OUT / 'M1_well_measurement_status.csv.gz')
    OUT.mkdir(exist_ok=True)
    replacements = {entry['item']: entry['shown'] for entry in pins}
    for name, frame in frames.items():
        frame.to_csv(OUT / (name + '.csv'), index=False)
        tokenized = reader_text(texts[name])
        rendered = tokenized
        for item, shown in replacements.items():
            rendered = rendered.replace('{{' + item + '}}', shown)
        require('{{' not in rendered, f'Unresolved tokens in {name}')
        (OUT / (name + '.md.in')).write_text(tokenized, encoding='utf-8')
        (OUT / (name + '.md')).write_text(rendered, encoding='utf-8')
    (OUT / 'corpus_numeric_pins.json').write_text(json.dumps(pins, indent=2) + '\n', encoding='utf-8')
    print(f'Reconciled corpus; {len(pins)} frozen numerical pins; ' + ', '.join((f'{name}: {len(frame)} rows' for name, frame in frames.items())))
if __name__ == '__main__':
    main()
