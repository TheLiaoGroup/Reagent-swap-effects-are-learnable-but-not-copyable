"""Add support diagnostics, raw-source witness readback, and concise evidence tables."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parents[1]
RAW = RAW_DIR

def read(dest, name, **kwargs):
    return pd.read_csv(dest / name, keep_default_na=False, float_precision='round_trip', **kwargs)

def save(dest, frame, name):
    frame.to_csv(dest / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def fix_witness_scope_metadata(dest):
    path = dest / 'chemical_witness.json'
    w = json.loads(path.read_text(encoding='utf-8'))
    if 'group_id' not in w:
        return
    g = read(dest, 'complete_group_registry.csv.gz').set_index('group_id').loc[w['group_id']].to_dict()
    f = read(dest, 'complete_focal_key_registry.csv.gz').set_index('focal_key_id').loc[g['focal_key_id']].to_dict()
    for key in ['n_partners', 'n_acquisition_contexts', 'n_assignments', 'n_full_assignments', 'partner_smiles', 'acquisition_multiplicity_by_partner', 'context_ids_by_partner']:
        w['parent_' + key] = f[key]
    for key in ['n_assignments', 'n_full_assignments', 'acquisition_multiplicity_by_partner', 'context_ids_by_partner', 'parent_quartets', 'parent_grids']:
        w.pop(key, None)
    w['focal_key_parent_quartets'], w['focal_key_parent_grids'] = (f['parent_quartets'], f['parent_grids'])
    w['n_partners'] = int(g['n_partners'])
    w['n_acquisition_contexts'] = len(json.loads(g['context_ids']))
    w['partner_smiles'] = json.dumps(w['partners'], ensure_ascii=False, separators=(',', ':'))
    assert w['n_partners'] == w['n_acquisition_contexts'] == len(w['partners'])
    path.write_text(json.dumps(w, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def diagnostic(dest):
    groups = read(dest, 'complete_group_registry.csv.gz')
    contexts = read(dest, 'complete_context_registry.csv.gz').set_index('context_id')
    support = {i: set(json.loads(r)) for i, r in zip(contexts.index, contexts.native_common_recipe_ids)}
    rows = []
    for g in groups.to_dict('records'):
        ids = json.loads(g['context_ids'])
        full = set.intersection(*(support[c] for c in ids))
        pair_extra = [len((support[a] & support[b]) - full) for a, b in combinations(ids, 2)]
        rows.append({'group_id': g['group_id'], 'n_native_pair_supports': len(pair_extra), 'all_native_pair_supports_identical_to_full_support': all((n == 0 for n in pair_extra)), 'n_native_pair_supports_with_extra_nodes': sum((n > 0 for n in pair_extra)), 'min_extra_native_pair_nodes': min(pair_extra), 'max_extra_native_pair_nodes': max(pair_extra), 'extra_nodes_by_ordered_partner_pair': json.dumps(pair_extra, separators=(',', ':'))})
    diagnostics = pd.DataFrame(rows)
    save(dest, diagnostics, 'complete_group_pair_support_diagnostic.csv.gz')
    summary = read(dest, 'complete_group_threshold_summary.csv.gz').merge(groups, on='group_id').merge(diagnostics, on='group_id')
    result = []
    for threshold in [0.1, 0.2, 0.3]:
        for identical in [False, True]:
            pp = summary[(summary.threshold == threshold) & (summary.n_partners >= 3) & (summary.all_native_pair_supports_identical_to_full_support == identical)]
            for event, field in [('all_opportunities', None), ('has_connected_high_endpoints', 'n_connected_endpoint_pairs'), ('pairwise_but_not_full', 'n_pairwise_but_not_full_certificate'), ('strong_pairwise_both_focal_but_not_full', 'n_strong_pairwise_both_focal_but_not_full_certificate')]:
                eligible = pp if field is None else pp[pp[field] > 0]
                result.append({'threshold': threshold, 'all_native_pair_supports_identical_to_full_support': identical, 'state': event, 'n_groups': len(eligible), 'n_focal_keys': eligible.focal_key_id.nunique(), 'n_exact_molecular_pairs': eligible.molecular_pair_id.nunique(), 'n_classes': eligible.rxn_class.nunique(), 'n_endpoint_views': int(eligible[field].sum()) if field else 0})
    save(dest, pd.DataFrame(result), 'native_pair_support_certificate_crossclassification.csv')
    means = []
    for threshold in [0.1, 0.2, 0.3]:
        pp = summary[(summary.threshold == threshold) & (summary.n_partners >= 3) & (summary.n_connected_endpoint_pairs > 0)].copy()
        pp['group_mean_gap'] = pp.sum_pair_vs_all_gap / pp.n_connected_endpoint_pairs
        means.append({'threshold': threshold, 'n_supported_groups': len(pp), 'n_supported_focal_keys': pp.focal_key_id.nunique(), 'endpoint_weighted_mean_pair_vs_all_gap': pp.sum_pair_vs_all_gap.sum() / pp.n_connected_endpoint_pairs.sum(), 'equal_assignment_mean_pair_vs_all_gap': pp.group_mean_gap.mean(), 'equal_focal_key_mean_pair_vs_all_gap': pp.groupby('focal_key_id').group_mean_gap.mean().mean(), 'maximum_pair_vs_all_gap': pp.max_pair_vs_all_gap.max()})
    save(dest, pd.DataFrame(means), 'continuous_gap_weighting_summary.csv')
    witness = json.loads((dest / 'chemical_witness.json').read_text(encoding='utf-8'))
    if 'group_id' in witness:
        witness['native_pair_support_diagnostic'] = diagnostics.set_index('group_id').loc[witness['group_id']].to_dict()
        witness['native_pair_support_diagnostic'] = {k: v.item() if hasattr(v, 'item') else v for k, v in witness['native_pair_support_diagnostic'].items()}
        (dest / 'chemical_witness.json').write_text(json.dumps(witness, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(str(dest.name), 'SUPPORT', json.dumps(result), flush=True)

def readback(dest):
    witness = json.loads((dest / 'chemical_witness.json').read_text(encoding='utf-8'))
    if 'group_id' not in witness:
        return
    source = read(dest, 'chemical_witness_source_records.csv.gz')
    hashes, checked, all_raw = ({}, [], [])
    for (filename, sheet), part in source.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_csv(path, keep_default_na=False) if sheet == 'CSV' else reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in part.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            y = float(rr['yield'])
            assert abs(y - record['yield_numeric']) < 1e-12
            for field in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive']:
                assert str(rr[field]) == str(record[field]), (filename, record['source_row'], field)
            assert str(rr['name0']).startswith(str(record['acquisition']) + '-')
            checked.append({'record_id': record['record_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'raw_yield': y, 'raw_yield_is_exact_zero': y == 0.0, 'all_identity_and_yield_fields_match': True})
            all_raw.append({'block_id': record['block_id'], 'recipe_id': record['recipe_id'], 'yield': y})
    save(dest, pd.DataFrame(checked), 'chemical_witness_raw_reaction_readback.csv')
    cond = read(dest, 'chemical_witness_conditions.csv')
    condition_checks = []
    fields = ['condition_id', *['Reagent' + str(k) + '_id' for k in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    for (filename, sheet), part in cond.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for record in part.to_dict('records'):
            rr = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(record['source_row'])]
            for field in fields:
                if str(rr[field]) != str(record[field]):
                    assert abs(float(rr[field]) - float(record[field])) < 1e-12
            condition_checks.append({'recipe_id': record['recipe_id'], 'condition_id': record['condition_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': record['source_row'], 'all_fields_match': True})
    save(dest, pd.DataFrame(condition_checks), 'chemical_witness_raw_condition_readback.csv')
    reconstructed = pd.DataFrame(all_raw).groupby(['block_id', 'recipe_id'])['yield'].mean()
    nodes = read(dest, 'chemical_witness_full_node_yields.csv')
    zero_nodes = []
    for record in nodes.to_dict('records'):
        for side in ['a', 'b']:
            actual = reconstructed.loc[record['block_' + side], record['recipe_id']]
            assert abs(actual - record['yield_' + side]) < 1e-12
            if record['yield_' + side] == 0.0:
                assert actual == 0.0
                zero_nodes.append({'context_id': record['context_id'], 'partner_position': record['partner_position'], 'focal': side, 'node': record['node'], 'recipe_id': record['recipe_id'], 'block_id': record['block_' + side], 'raw_source_mean': actual, 'all_source_records_exact_zero': True})
    save(dest, pd.DataFrame(zero_nodes), 'chemical_witness_exact_zero_node_readback.csv')
    result = {'status': 'passed', 'group_id': witness['group_id'], 'n_original_reaction_records': len(checked), 'n_original_condition_rows': len(condition_checks), 'n_original_zero_yield_records': sum((r['raw_yield_is_exact_zero'] for r in checked)), 'n_node_target_exact_zeros_verified': len(zero_nodes), 'all_exported_node_yields_reconstructed_from_original_workbook': True, 'zeros_are_measured_source_values_not_missing_value_imputations': True, 'original_workbook_sha256': hashes}
    (dest / 'chemical_witness_source_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(str(dest.name), 'RAW', json.dumps(result, ensure_ascii=True), flush=True)

def delivery_manifests():
    for dest in [OUT, OUT / 'triple_scope']:
        verification = json.loads((dest / 'independent_verification.json').read_text(encoding='utf-8'))
        assert verification['status'] == 'passed'
        witness = json.loads((dest / 'chemical_witness.json').read_text(encoding='utf-8'))
        if 'status' in witness:
            assert witness['status'] == 'no_pairwise_feasible_full_scope_failed_certificate20'
            assert 'group_id' not in witness
            witness_status = witness['status']
            witness_check = 'skipped_no_witness'
            witness_workbooks = {}
        else:
            assert 'group_id' in witness
            raw = json.loads((dest / 'chemical_witness_source_verification.json').read_text(encoding='utf-8'))
            assert raw['status'] == 'passed'
            assert witness['n_partners'] == witness['n_acquisition_contexts'] == len(witness['partners'])
            witness_status = 'certificate_available'
            witness_check = 'passed'
            witness_workbooks = raw['original_workbook_sha256']
        cross = read(dest, 'complete_threshold_crossclassification.csv')
        summary = read(dest, 'summary.csv')
        for threshold in [0.1, 0.2, 0.3]:
            row = summary[(summary.stratum == 'groups_ge3') & (summary.dimension == 'all') & (summary.threshold == threshold)].iloc[0]
            assert cross[cross.threshold == threshold].n_endpoint_views.sum() == row.n_connected_endpoint_pairs
        path = dest / 'source_manifest.json'
        source = json.loads(path.read_text(encoding='utf-8'))
        source.setdefault('generator_at_initial_numeric_run', source['generator'])
        for key, filename in [('generator', 'round27_partner_scope.py'), ('independent_verifier', 'verify_partner_scope.py'), ('finalizer', 'finalize_partner_scope.py')]:
            code = OUT / filename
            source[key] = {'path': str(code), 'sha256': hashlib.file_digest(code.open('rb'), 'sha256').hexdigest()}
        source['metadata_normalization'] = 'Finalizer prefixes focal-key parent fields in witness; numeric group registries and route capacities are unchanged.'
        source['chemical_witness_status'] = witness_status
        source['original_witness_workbooks'] = witness_workbooks
        path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        files = []
        for file in sorted(dest.iterdir()):
            if file.is_file() and file.name != 'delivery_manifest.json':
                files.append({'name': file.name, 'bytes': file.stat().st_size, 'sha256': hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()})
        result = {'status': 'passed', 'n_files': len(files), 'total_bytes': sum((f['bytes'] for f in files)), 'chemical_witness_status': witness_status, 'scope_metadata_check': witness_check, 'witness_source_verification_check': witness_check, 'crossclassification_sum_check': 'passed', 'files': files}
        (dest / 'delivery_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(dest.name, 'DELIVERY', len(files), result['total_bytes'], flush=True)
if __name__ == '__main__':
    started = time.time()
    if '--manifest-only' not in sys.argv:
        for dest in [OUT, OUT / 'triple_scope']:
            fix_witness_scope_metadata(dest)
            diagnostic(dest)
            readback(dest)
    delivery_manifests()
    print('FINALIZED', time.time() - started, flush=True)
