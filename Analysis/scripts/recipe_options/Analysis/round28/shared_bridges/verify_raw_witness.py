"""Read back the selected bridge's original reaction and condition workbooks."""
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import pandas as pd
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
BASE = Path(__file__).resolve().parent
OUT = BASE / sys.argv[sys.argv.index('--witness-dir') + 1] if '--witness-dir' in sys.argv else BASE
ANALYSIS = BASE.parents[1]
RAW = RAW_DIR

def read(name):
    return pd.read_csv(OUT / name, keep_default_na=False, float_precision='round_trip')

def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g')

def main():
    nodes = read('chemical_witness_full_node_yields.csv')
    blocks = set(nodes.block_a) | set(nodes.block_b)
    recipes = set(nodes.recipe_id)
    source_path = ANALYSIS / 'round7/joint_reactant_connectivity/all_strict_source_records.csv.gz'
    parts = []
    for chunk in pd.read_csv(source_path, keep_default_na=False, float_precision='round_trip', chunksize=150000):
        part = chunk[chunk.block_id.isin(blocks) & chunk.recipe_id.isin(recipes)]
        if len(part):
            parts.append(part)
    source = pd.concat(parts, ignore_index=True)
    source.to_csv(OUT / 'chemical_witness_source_records.csv.gz', index=False, encoding='utf-8-sig', float_format='%.17g', compression='gzip')
    hashes, checked, raw_values = ({}, [], [])
    for (filename, sheet), part in source.groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_csv(path, keep_default_na=False) if sheet == 'CSV' else reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for r in part.to_dict('records'):
            actual = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(r['source_row'])]
            assert float(actual['yield']) == r['yield_numeric']
            for field in ['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'archive']:
                assert str(actual[field]) == str(r[field])
            assert str(actual['name0']).startswith(str(r['acquisition']) + '-')
            checked.append({'record_id': r['record_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': r['source_row'], 'yield': float(actual['yield']), 'identity_yield_acquisition_match': True})
            raw_values.append({'block_id': r['block_id'], 'recipe_id': r['recipe_id'], 'yield': float(actual['yield'])})
    means = pd.DataFrame(raw_values).groupby(['block_id', 'recipe_id'])['yield'].mean()
    for row in nodes.to_dict('records'):
        for side in ['a', 'b']:
            assert abs(means.loc[row['block_' + side], row['recipe_id']] - row['yield_' + side]) < 2e-15
    condchecks = []
    for (filename, sheet), part in read('chemical_witness_conditions.csv').groupby(['source_file', 'source_sheet']):
        path = RAW / filename
        hashes[str(path)] = reader_tables.source_evidence(path)['sha256']
        raw = reader_tables.read_excel(path, sheet_name=sheet, keep_default_na=False)
        for r in part.to_dict('records'):
            actual = raw.set_axis(reader_tables.source_rows(path, sheet, original=path.name.startswith('0_'))).loc[int(r['source_row'])]
            for field in ['condition_id', *[f'Reagent{k}_id' for k in range(1, 7)], *[f'Reagent{k}_HTE_Equiv' for k in range(1, 7)], 'solvent_id', 'temp_id', 'time_id']:
                if str(actual[field]) != str(r[field]):
                    assert abs(float(actual[field]) - float(r[field])) < 1e-12
            condchecks.append({'recipe_id': r['recipe_id'], 'source_file': filename, 'source_sheet': sheet, 'source_row': r['source_row'], 'all_identity_dose_background_fields_match': True})
    save(pd.DataFrame(checked), 'chemical_witness_raw_reaction_readback.csv')
    save(pd.DataFrame(condchecks), 'chemical_witness_raw_condition_readback.csv')
    result = {'status': 'passed', 'n_raw_reaction_records': len(checked), 'n_raw_condition_rows': len(condchecks), 'n_target_recipe_means_reconstructed': 2 * len(nodes), 'n_exact_zero_raw_records': sum((r['yield'] == 0 for r in checked)), 'all_identity_yield_acquisition_dose_background_checks': 'passed', 'original_workbook_sha256': hashes}
    (OUT / 'chemical_witness_source_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    witness = json.loads((OUT / 'chemical_witness.json').read_text(encoding='utf-8'))
    witness['source_readback_status'] = 'passed'
    witness['n_original_reaction_rows'] = len(checked)
    witness['n_original_condition_rows'] = len(condchecks)
    (OUT / 'chemical_witness.json').write_text(json.dumps(witness, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
if __name__ == '__main__':
    main()
