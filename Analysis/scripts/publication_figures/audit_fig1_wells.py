"""Trace a selected draft example to current raw wells and exclusions."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from publication_figures import common as c

def paths():
    return {'reference_learning': c.A / 'scripts/reference_learning/Analysis/data/raw_long.csv.gz', 'recipe_options': c.A / 'scripts/recipe_options/Analysis/data/raw_reactions.csv.gz', 'conditions': c.A / 'scripts/reference_learning/Analysis/data/raw_condition_table.csv', 'exclusions': __import__('m1_exclusions').load_exclusions().source_path}

def reaction_class(prediction):
    return str(prediction.screen).rsplit('|', 1)[0]

def clean(value):
    return '' if value is None or pd.isna(value) else value

def same(left, right):
    left, right = (clean(left), clean(right))
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return bool(np.isclose(left, right, rtol=0, atol=1e-10))
    return left == right

def component_name(recipe, category):
    if category.startswith('Reagent'):
        return str(recipe[f'{category}_Standardized Name'])
    if category in ('solvent', 'solvent_id'):
        return str(recipe['Standardized Name'])
    if category in ('temp', 'temperature', 'temp_id'):
        return f'{float(recipe.temp_id):g} C'
    if category in ('time', 'time_id'):
        return f'{float(recipe.time_id):g} h'
    raise ValueError(f'Unrecognized exchanged component: {category}')

def _selected_raw(path, class_field, pair_field, prediction, cells):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
        local = chunk.loc[chunk[class_field].eq(reaction_class(prediction)) & chunk[pair_field].eq(prediction.substrate_pair) & chunk.condition_id.isin(cells.condition)]
        if len(local):
            chunks.append(local)
    if not chunks:
        raise ValueError(f'Selected wells absent from {path}')
    raw = pd.concat(chunks, ignore_index=True).fillna('')
    if len(raw) != 4 or not raw.condition_id.is_unique or (not raw.name0.is_unique):
        raise ValueError(f'Four source wells do not resolve uniquely in {path}')
    return raw.set_index('condition_id', drop=False)

def _direct_source_rows(rows):
    source_paths = []
    found = 0
    by_file = {}
    for row in rows:
        by_file.setdefault(row['source_file'], []).append(row)
    for filename, selected in by_file.items():
        matches = reader_tables.source_paths(pattern=filename)
        if len(matches) != 1:
            raise ValueError(f'No unique current CSV source for {filename}')
        source = matches[0]
        source_paths.append(source)
        for expected in selected:
            sheet_name, source_row = (expected['source_sheet'], expected['source_row'])
            actual = reader_tables.read_row(source, sheet_name, source_row, dtype=object, keep_default_na=False).to_dict()
            for field in ['name0', 'idx', 'condition_id', 'sub_1_smiles', 'sub_2_smiles', *[f'reagent{i}_id' for i in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']:
                if not same(actual[field], expected[field]):
                    raise ValueError(f'Direct source mismatch: {filename}:{sheet_name}:{source_row}:{field}')
            if not np.isclose(100 * float(actual['yield']), expected['yield_percent'], rtol=0, atol=1e-08):
                raise ValueError(f'Direct source yield mismatch: {filename}:{source_row}')
            if str(actual['check_passed']).lower() not in ('true', '1', '1.0'):
                raise ValueError('Selected source well did not pass QC')
            expected['direct_workbook_match'] = False
            expected['direct_source_match'] = True
            found += 1
    if found != 4:
        raise ValueError('Four direct source rows were not read back')
    return (source_paths, found)

def check(prediction, cells):
    source = paths()
    cls = reaction_class(prediction)
    if len(cells) != 4 or not cells.condition.is_unique:
        raise ValueError('Example must have four distinct endpoint conditions')
    raw1 = _selected_raw(source['reference_learning'], 'reaction_class', 'pair_id', prediction, cells)
    raw2 = _selected_raw(source['recipe_options'], 'rxn_class', 'subpair', prediction, cells)
    conditions = pd.read_csv(source['conditions']).fillna('')
    conditions = conditions.loc[conditions.reaction_class.eq(cls) & conditions.condition_id.isin(cells.condition)].set_index('condition_id')
    if len(conditions) != 4 or not conditions.index.is_unique:
        raise ValueError('Example recipes do not resolve uniquely')
    exclusions = pd.read_csv(source['exclusions'])
    if len(exclusions) != 396 or not exclusions.name0.is_unique:
        raise ValueError('Authoritative exclusion list does not contain 396 unique wells')
    rows = []
    identity_fields = ['name0', 'idx', 'source_file', 'source_sheet', 'source_row', 'record_id', 'archive', 'sub_1_smiles', 'sub_2_smiles', *[f'reagent{i}_id' for i in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']
    for cell in cells.itertuples():
        one, two, recipe = (raw1.loc[cell.condition], raw2.loc[cell.condition], conditions.loc[cell.condition])
        for field in identity_fields:
            if not same(one[field], two[field]):
                raise ValueError(f'Retained raw identity mismatch: {cell.condition}:{field}')
        if one.plate_id != two.plate:
            raise ValueError('Retained raw plate identities disagree')
        if not np.isclose(100 * one.yield_numeric, cell.yield_percent, rtol=0, atol=1e-08):
            raise ValueError('Matrix and raw well yield disagree')
        if not np.isclose(one.yield_numeric, two.yield_numeric, rtol=0, atol=1e-10):
            raise ValueError('Retained raw well yields disagree')
        for field in ['check_passed', 'qc_pass', 'unique_finite_cell']:
            if str(one[field]).lower() != 'true':
                raise ValueError(f'Selected well failed {field}')
        if str(two.check_passed).lower() != 'true':
            raise ValueError('RECIPE_OPTIONS selected well failed QC')
        for field in ['duplicate_cell_all', 'duplicate_cell_finite']:
            if str(one[field]).lower() != 'false':
                raise ValueError('Selected well is a duplicate cell')
        for i in range(1, 7):
            if not same(recipe[f'Reagent{i}_id'], one[f'reagent{i}_id']):
                raise ValueError('Condition registry and raw reagent identities disagree')
        for field in ['solvent_id', 'temp_id', 'time_id']:
            if not same(recipe[field], one[field]):
                raise ValueError(f'Condition registry and raw {field} disagree')
        if hasattr(cell, 'exchange_category') and component_name(recipe, cell.exchange_category) != cell.reagent:
            raise ValueError('Displayed exchanged component and raw condition registry disagree')
        matches = int(exclusions.name0.eq(one.name0).sum())
        if matches:
            raise ValueError(f'Selected well is excluded: {one.name0}')
        rows.append({'background': cell.background, 'reagent': cell.reagent, 'condition_id': cell.condition, 'background_label': getattr(cell, 'background_label', cell.background), 'name0': one.name0, 'plate_id': one.plate_id, 'archive': one.archive, 'idx': int(one.idx), 'source_file': one.source_file, 'source_sheet': one.source_sheet, 'source_row': int(one.source_row), 'record_id': one.record_id, 'substrate_pair': prediction.substrate_pair, 'sub_1_smiles': one.sub_1_smiles, 'sub_2_smiles': one.sub_2_smiles, 'yield_percent': float(cell.yield_percent), **{f'reagent{i}_id': one[f'reagent{i}_id'] for i in range(1, 7)}, 'solvent_id': one.solvent_id, 'temp_id': float(one.temp_id), 'time_id': float(one.time_id), 'recipe_names_json': json.dumps({f'Reagent{i}': recipe[f'Reagent{i}_Standardized Name'] for i in range(1, 7)}), 'solvent': recipe['Standardized Name'], 'qc_pass': True, 'unique_finite_cell': True, 'duplicate_cell': False, 'excluded_name0_matches': matches})
    raw_sources, found = _direct_source_rows(rows)
    plate_matches = int(exclusions.plate.isin({r['plate_id'] for r in rows}).sum())
    sources = [*raw_sources, *source.values()]
    c.source('S5_source_well_audit', pd.DataFrame(rows), sources, 'Exact class/pair/condition lookup in both retained raw tables; direct source rows; exact name0 exclusion check', 'measured yield %', 'Four unique finite QC-passing wells have zero authoritative name0 exclusion matches.')
    report = {'status': 'passed', 'screen': prediction.screen, 'edge_id': prediction.edge_id, 'pair_index': int(prediction.pair_index), 'reaction_class': cls, 'cell_count': 4, 'authoritative_exclusion_rows': len(exclusions), 'exact_name0_matches': 0, 'selected_plate_matches': plate_matches, 'direct_source_matches': found, 'direct_workbook_matches': sum((r['direct_workbook_match'] for r in rows)), 'retained_raw_tables_agree': True, 'cells': rows, 'sources': [{'path': str(p), 'sha256': reader_tables.source_evidence(p)['sha256'] if reader_tables.is_source(p) else hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    (c.OUT / 'FIG1A_WELL_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
