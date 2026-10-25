"""Read original workbooks for every exported component-option chemical witness.

M1 port of the RECIPE_OPTIONS R29 raw verifier: retain raw identity/condition/mean checks,
check all five current witnesses directly, and never reuse a historical receipt."""
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys
sys.dont_write_bytecode = True
import openpyxl
import pandas as pd
SCRIPTS = next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))
sys.path.insert(0, str(SCRIPTS))
from m1_raw import RAW_DIR
from m1_exclusions import load_exclusions
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'round27/partner_scope/triple_scope'
STRICT_SOURCE = OUT.parents[1] / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
RAW = RAW_DIR
CONDITION_FIELDS = ['condition_id', *[f'Reagent{k}_id' for k in range(1, 7)], *[f'Reagent{k}_HTE_Equiv' for k in range(1, 7)], *[f'Reagent{k}_Standardized Name' for k in range(1, 7)], 'solvent_id', 'Standardized Name', 'temp_id', 'time_id']

def read(name):
    return pd.read_csv(OUT / name, keep_default_na=False, float_precision='round_trip')

def same(actual, expected):
    actual = '' if actual is None else actual
    if str(actual) == str(expected):
        return True
    try:
        return abs(float(actual) - float(expected)) < 1e-12
    except (ValueError, TypeError):
        return False

def selected_rows(path, sheet, row_numbers):
    assert row_numbers and min(row_numbers) >= 2
    result = {number: reader_tables.read_row(path, sheet, number, original=path.name.startswith('0_'), dtype=object, keep_default_na=False).to_dict() for number in row_numbers}
    assert set(result) == row_numbers
    return result

def raw_yield(actual):
    value = actual['yield']
    if value is None or str(value).strip() == '' or pd.isna(value):
        assert same(actual.get('yield_clf'), 0), 'Unquantified nonzero/unknown DESI call in a scored witness'
        return 0.0
    result = float(value)
    assert math.isfinite(result) and 0 <= result <= 1
    return result

def sha256(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(frame, name):
    pd.DataFrame(frame).to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g')

def verify_registry_linkage(nodes, conditions, routes, summary):
    """Join both QA branches to the same registered six-target chemical graph."""
    groups = pd.read_csv(SOURCE / 'complete_group_registry.csv.gz', keep_default_na=False).set_index('group_id')
    contexts = pd.read_csv(SOURCE / 'complete_context_registry.csv.gz', keep_default_na=False).set_index('context_id')
    support = pd.read_csv(SOURCE / 'complete_support_registry.csv.gz', keep_default_na=False).set_index('support_id')
    registry_conditions = pd.read_csv(SOURCE / 'full_condition_registry.csv.gz', keep_default_na=False, float_precision='round_trip').drop_duplicates(['dataset_id', 'recipe_id']).set_index(['dataset_id', 'recipe_id'])
    registry_values = pd.read_csv(SOURCE / 'full_source_block_node_yields.csv.gz', keep_default_na=False, float_precision='round_trip').set_index(['block_id', 'recipe_id']).yield_mean
    assert registry_values.index.is_unique
    assert summary.certificate_id.is_unique
    by_certificate = summary.set_index('certificate_id')
    assert set(nodes.certificate_id) <= set(by_certificate.index)
    n_values = 0
    for cid, nn in nodes.groupby('certificate_id', sort=True):
        group = groups.loc[by_certificate.loc[cid, 'group_id']]
        recipes = json.loads(support.loc[group.support_id, 'recipe_ids'])
        context_ids = json.loads(group.context_ids)
        assert len(context_ids) == 3
        expected_keys = {(context, node) for context in context_ids for node in range(len(recipes))}
        assert not nn.duplicated(['context_id', 'node']).any()
        assert set(zip(nn.context_id, nn.node)) == expected_keys
        for row in nn.to_dict('records'):
            assert row['recipe_id'] == recipes[int(row['node'])]
            context = contexts.loc[row['context_id']]
            for side in ['a', 'b']:
                assert row['block_' + side] == context['block_' + side]
                expected = float(registry_values.loc[context['block_' + side], row['recipe_id']])
                assert math.isfinite(expected) and abs(float(row['yield_' + side]) - expected) < 2e-15
                n_values += 1
        cc = conditions[conditions.certificate_id == cid]
        assert not cc.duplicated('recipe_id').any()
        assert set(cc.recipe_id) == set(recipes)
        for row in cc.to_dict('records'):
            assert row['dataset_id'] == group.dataset_id
            expected = registry_conditions.loc[group.dataset_id, row['recipe_id']]
            for field in ['source_file', 'source_sheet', 'source_row', *CONDITION_FIELDS]:
                assert same(row[field], expected[field]), (cid, row['recipe_id'], field)
        minima = nn.assign(shared=nn[['yield_a', 'yield_b']].min(axis=1)).groupby(['node', 'recipe_id']).shared.min()
        for row in routes[routes.certificate_id == cid].to_dict('records'):
            if 'group_id' in row:
                assert row['group_id'] == by_certificate.loc[cid, 'group_id']
            assert (row['node'], row['recipe_id']) in minima.index
            assert abs(float(row['shared_node_yield']) - minima.loc[row['node'], row['recipe_id']]) < 2e-14
    return n_values

def verify_source_membership(source, nodes):
    """Check original-well membership and block/recipe assignment by record ID."""
    assert not source.duplicated(['certificate_id', 'record_id']).any(), 'Repeated raw record within a witness'
    expected_pairs = {(row['block_' + side], row['recipe_id']) for row in nodes.to_dict('records') for side in ['a', 'b']}
    columns = ['record_id', 'block_id', 'recipe_id', 'source_file', 'source_sheet', 'source_row', 'yield_numeric']
    parts = []
    for chunk in pd.read_csv(STRICT_SOURCE, keep_default_na=False, float_precision='round_trip', usecols=columns, chunksize=150000):
        selected = pd.MultiIndex.from_frame(chunk[['block_id', 'recipe_id']]).isin(expected_pairs)
        if selected.any():
            parts.append(chunk[selected])
    assert parts, 'No registered raw records for the witness graphs'
    registered = pd.concat(parts, ignore_index=True)
    assert registered.record_id.is_unique
    indexed = registered.set_index('record_id')
    for row in source.drop_duplicates('record_id').to_dict('records'):
        assert row['record_id'] in indexed.index
        original = indexed.loc[row['record_id']]
        for field in columns[1:]:
            assert same(row[field], original[field]), (row['record_id'], field, 'registered original-well assignment')
    for cid, nn in nodes.groupby('certificate_id', sort=True):
        pairs = {(row['block_' + side], row['recipe_id']) for row in nn.to_dict('records') for side in ['a', 'b']}
        expected_ids = registered[pd.MultiIndex.from_frame(registered[['block_id', 'recipe_id']]).isin(pairs)].record_id
        assert set(source[source.certificate_id == cid].record_id) == set(expected_ids), (cid, 'incomplete raw membership')
    return len(source)

def main():
    (OUT / 'chemical_witness_raw_verification.json').unlink(missing_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    exclusions = load_exclusions()
    evidence_paths = [OUT / name for name in ['chemical_witness_source_records.csv.gz', 'chemical_witness_conditions.csv', 'chemical_witness_full_node_yields.csv', 'chemical_witness_routes.csv', 'certificate_option_summary.csv']]
    evidence_paths += [SOURCE / name for name in ['complete_group_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_condition_registry.csv.gz', 'full_source_block_node_yields.csv.gz']]
    evidence_paths += [STRICT_SOURCE, Path(__file__).resolve(), SCRIPTS / 'm1_exclusions.py', exclusions.source_path]
    evidence = {str(path): sha256(path) for path in evidence_paths}
    assert evidence[str(exclusions.source_path)] == exclusions.source_sha256, 'Exclusion input changed after loading'
    source = read('chemical_witness_source_records.csv.gz')
    conditions = read('chemical_witness_conditions.csv')
    nodes = read('chemical_witness_full_node_yields.csv')
    routes = read('chemical_witness_routes.csv')
    summary = read('certificate_option_summary.csv')
    certificates = set(nodes.certificate_id)
    assert len(certificates) == 5
    assert set(source.certificate_id) == set(conditions.certificate_id) == set(routes.certificate_id) == certificates
    linked_values = verify_registry_linkage(nodes, conditions, routes, summary)
    registered_records = verify_source_membership(source, nodes)
    unique = source.drop(columns='certificate_id').drop_duplicates()
    assert unique.record_id.is_unique, 'Inconsistent source metadata for a reused raw record'
    condition_keys = ['dataset_id', 'recipe_id']
    pointer_fields = ['source_file', 'source_sheet', 'source_row']
    unique_conditions = conditions[condition_keys + pointer_fields + CONDITION_FIELDS].drop_duplicates()
    assert not unique_conditions.duplicated(condition_keys).any(), 'Inconsistent reused condition metadata'
    hashes, reaction_checks, condition_checks, raw_values = ({}, [], [], {})
    for (filename, sheet), part in unique.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        print(json.dumps({'reading_raw_reactions': str(path), 'n_requested_records': len(part)}), flush=True)
        hashes[str(path)] = sha256(path)
        rows = selected_rows(path, sheet, set(part.source_row.astype(int)))
        for r in part.to_dict('records'):
            actual = rows[int(r['source_row'])]
            assert r['record_id'] == f"{filename}:{sheet}:{int(r['source_row'])}"
            assert str(actual['name0']) not in exclusions.ids, 'Excluded well in a chemical witness'
            value = raw_yield(actual)
            assert value == r['yield_numeric'], r['record_id']
            for field in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive']:
                assert same(actual[field], r[field]), (r['record_id'], field)
            if 'name0' in r:
                assert same(actual['name0'], r['name0']), (r['record_id'], 'name0')
            assert str(actual['name0']).startswith(str(r['acquisition']) + '-'), r['record_id']
            raw_values[r['record_id']] = value
            reaction_checks.append({'record_id': r['record_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': r['source_row'], 'yield': value, 'structure_yield_condition_archive_acquisition_match': True, **{field: actual[field] for field in ['sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive', 'condition_id', 'name0']}})
    for (filename, sheet), part in unique_conditions.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        print(json.dumps({'reading_raw_conditions': str(path), 'n_requested_conditions': len(part)}), flush=True)
        hashes[str(path)] = sha256(path)
        rows = selected_rows(path, sheet, set(part.source_row.astype(int)))
        for r in part.to_dict('records'):
            actual = rows[int(r['source_row'])]
            for field in CONDITION_FIELDS:
                assert same(actual[field], r[field]), (r['recipe_id'], field, actual[field], r[field])
            condition_checks.append({'dataset_id': r['dataset_id'], 'recipe_id': r['recipe_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': r['source_row'], 'all_identity_dose_background_fields_match': True, **{field: actual[field] for field in CONDITION_FIELDS}})
    source = source.copy()
    source['raw_yield'] = source.record_id.map(raw_values)
    assert source.raw_yield.notna().all()
    mean_checks, by_witness = ([], [])
    for cid in sorted(certificates):
        ss = source[source.certificate_id == cid].drop_duplicates('record_id')
        nn = nodes[nodes.certificate_id == cid]
        cc = conditions[conditions.certificate_id == cid]
        expected = {(row['block_' + side], row['recipe_id']) for row in nn.to_dict('records') for side in ['a', 'b']}
        assert set(zip(ss.block_id, ss.recipe_id)) == expected
        assert set(cc.recipe_id) == set(nn.recipe_id)
        means = ss.groupby(['block_id', 'recipe_id']).raw_yield.mean()
        errors = []
        for row in nn.to_dict('records'):
            for side in ['a', 'b']:
                actual = means.loc[row['block_' + side], row['recipe_id']]
                error = abs(actual - row['yield_' + side])
                assert error < 2e-15
                errors.append(error)
                mean_checks.append({'certificate_id': cid, 'context_id': row['context_id'], 'side': side, 'block_id': row['block_' + side], 'recipe_id': row['recipe_id'], 'raw_reconstructed_yield': actual, 'exported_yield': row['yield_' + side], 'absolute_error': error})
        by_witness.append({'certificate_id': cid, 'n_original_reaction_records': len(ss), 'n_original_conditions': len(cc), 'n_target_recipe_values_reconstructed': 2 * len(nn), 'n_original_exact_zero_records': int((ss.raw_yield == 0).sum()), 'maximum_yield_error': max(errors), 'identity_yield_acquisition_dose_background_checks': 'passed'})
    assert {str(path): sha256(path) for path in evidence_paths} == evidence, 'Witness evidence changed during raw readback'
    assert {path: sha256(Path(path)) for path in hashes} == hashes, 'Original workbook changed during raw readback'
    save(reaction_checks, 'chemical_witness_raw_reaction_readback.csv')
    save(condition_checks, 'chemical_witness_raw_condition_readback.csv')
    save(mean_checks, 'chemical_witness_raw_reconstructed_node_yields.csv')
    result = {'status': 'passed', 'new_witnesses': by_witness, 'started_utc': started_utc, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'witness_certificates': sorted(certificates), 'fresh_raw_readback': True, 'n_registry_linked_target_recipe_values': linked_values, 'n_registered_witness_source_records_checked': registered_records, 'witness_selection': 'all five producer-exported witnesses; no new group selection', 'n_deduplicated_raw_reaction_records': len(unique), 'n_deduplicated_raw_conditions': len(unique_conditions), 'n_witness_linked_raw_reaction_records': len(source), 'n_target_recipe_values_reconstructed': len(mean_checks), 'n_deduplicated_exact_zero_raw_records': sum((r['yield'] == 0 for r in reaction_checks)), 'original_workbook_sha256': hashes, 'verification_inputs_sha256': evidence, 'reused_R28_verifications': {}, 'all_five_witnesses_raw_readback_status': 'passed'}
    with (OUT / 'chemical_witness_raw_verification.json').open('w', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
