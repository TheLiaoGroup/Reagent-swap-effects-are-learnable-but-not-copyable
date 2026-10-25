"""Audit condition recipes directly from the original Excel registries.

Fingerprints are scoped within reaction_type. No yields or prior filtered tables
enter this audit, and no original workbook is changed."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
import sys as _m1_sys
_m1_sys.path.append(str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR as _M1_RAW_DIR
ROOT = HERE.parents[1]
RAW = _M1_RAW_DIR
OUT = HERE / 'tables'
CONDITION_FILE = RAW / '0_total condition_260810.xlsx'
COMPOUND_FILE = RAW / '0_Compound ID_260810.xlsx'
CORE_FIELDS = [field for i in range(1, 7) for field in (f'Reagent{i}_id', f'Reagent{i}_HTE_Equiv')]
CORE_FIELDS += ['solvent_id', 'temp_id', 'time_id']
IDENTITY_FIELDS = [f'Reagent{i}_id' for i in range(1, 7)]
IDENTITY_FIELDS += ['solvent_id', 'temp_id', 'time_id']
FAMILY_RULE_VERSION = 'name_based_putative_v1'

def sha256(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def token(value):
    text = str(value).strip()
    if not text:
        return ''
    try:
        number = Decimal(text)
        return format(number.normalize(), 'f') if number.is_finite() else text
    except InvalidOperation:
        return text

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def fingerprint(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()

def families(name):
    """Broad name patterns, not roles, active species, or mechanistic assignments."""
    labels = []
    for metal in ('Pd', 'Ni', 'Cu', 'Fe', 'Co', 'Mn', 'Zn', 'Ag', 'Au', 'Ir', 'Ru', 'Ti', 'Al', 'Sn', 'In', 'Ce', 'Mg', 'Zr', 'Pb', 'Cd'):
        if re.search(f'(?<![A-Za-z]){metal}(?=[^a-z]|$)', name):
            labels.append('metal_' + metal)
    if 'nickel(' in name.lower():
        labels.append('metal_Ni')
    if name in {'K2CO3', 'Na2CO3', 'Cs2CO3', 'Li2CO3', 'NaHCO3', 'KHCO3'}:
        labels.append('carbonate_bicarbonate')
    if name in {'K3PO4', 'K3PO4·3H2O', 'K2HPO4'}:
        labels.append('phosphate')
    if name in {'LiOH', 'LiOH·H2O', 'NaOH', 'KOH', 'CsOH', 'CsOH·H2O', 'NH3.H2O'}:
        labels.append('hydroxide')
    if name in {'NaOMe', 'EtONa', 'NaOtBu', 'NaOtAm', 'KOtBu', 'KOtAm', 'LiOtBu', 'KOPh', 'NaOSiMe3', 'KOSiMe3'}:
        labels.append('alkoxide_phenoxide_silanolate')
    if name in {'NaHMDS', 'KHMDS', 'LiHMDS', 'LDA', 'NaH', 'n-Butyllithium'}:
        labels.append('strong_base_organometallic')
    if name in {'TEA', 'DIPEA', 'DBU', 'DBN', 'TBD', 'MTBD', 'DABCO', 'NMM', 'NMI', 'Py', 'DMAP', '2,6-lutidine', '2-Picoline', 'iPr2NH', 'NHEt2', 'Cy2NMe', '4-pyrrolidinopyridine'}:
        labels.append('amine_aza_heterocycle')
    if name in {'TBAF', 'TBAT', 'CsF', 'KF', 'AgF', 'H2SiF6', 'CuF2'}:
        labels.append('fluoride_source')
    if any((piece in name for piece in ('Phos', 'BINAP', 'BIPHEP', 'phosphin', 'phosphon', 'PPh', 'PCy', 'PBu', 'P(tBu)', 'P(o-', 'P(p-', 'P(m-', 'P(2-', 'P(Mes)', 'P(NEt2)', 'P(NMe2)', 'P(OPh)', 'dpp', 'DPP', 'dtbpf', 'DTBPF', 'dcype', 'CyPF', 'QPhos'))):
        labels.append('phosphorus_ligand_or_complex')
    if any((piece in name for piece in ('IPr', 'SIPr', 'IMes', 'PEPPSI'))):
        labels.append('NHC_precursor_or_complex')
    if any((piece in name.lower() for piece in ('bipyridin', 'terpyridin', 'phenanthrol', 'diamine', 'ethylenediam', 'picolin'))) or name in {'1,10-Phen', '4Me2Phen', 'Me4Phen', 'TMEDA', 'DMEDA', 'DMEN', 'DETA', 'TAEA', 'DTBPY', 'BINAM', '(1R,2R)-DACH', '1,2-diaminocyclohexane'}:
        labels.append('nitrogen_donor_candidate')
    if name in {'NaBH4', 'NaBH3CN', 'NaBH(OAc)3', 'pic-BH3', 'BMS', 'LAH', 'DIBAL-H'}:
        labels.append('borane_borohydride_hydride')
    if name in {'PMHS', 'PhSiH3'}:
        labels.append('silane_reductant')
    if name in {'DCC', 'DIC', 'EDC-HCl'}:
        labels.append('carbodiimide')
    if name in {'HATU', 'HBTU', 'HCTU', 'TBTU', 'COMU', 'TCFH', 'TSTU', 'TCTU', 'TOTU', 'TNTU', 'TPTU', 'TDBTU', 'HBPYU', 'HDMC', 'CITU', 'CIP', 'DMTMM', 'CDMT', 'DCMT', 'CDI', 'BOP', 'BOPCl', 'PyBOP', 'PyAOP', 'PyBrOP', 'PyClOP', 'PyCIU', 'PyClocK', 'PyOxim', 'BTFFH', 'TFFH', 'FDPP', 'DEPBT', 'EEDQ', 'IIDQ', '(COCl)2', 'IBCF', 'DPP-Cl'}:
        labels.append('coupling_activation_reagent')
    if name in {'DEAD', 'DIAD', 'ADDP', 'TMAD', 'DBAD', 'DCAD', 'DMEAD', 'Di-tert-butyl azodicarboxylate'}:
        labels.append('azo_activation_reagent')
    if name in {'AcOH', 'HCOOH', 'TFA', 'PTSA', 'PTSA monohydrate', 'MsOH', '10-CSA', 'HCl', 'H2SO4', 'KHSO4'} or name.endswith(' acid'):
        labels.append('acid_candidate')
    if name in {'HOBt', 'Cl-HOBT', 'HOAT', 'NHS', 'oxyma-B', 'ethyl cyanoglyoxylate-2-oxime'}:
        labels.append('coupling_additive_candidate')
    if name in {'APS', 'K2S2O8', 'Na2S2O8', 'CAN', 'PIFA', 'Selectfluor', 'DTBP', 'MnO2', 'quinone'}:
        labels.append('oxidant_candidate')
    return sorted(set(labels)) or ['unclassified']
SOLVENT_GROUPS = {'amide_urea': {'DMF', 'DMA', 'NMP', 'DMI', 'DMPU'}, 'sulfoxide': {'DMSO'}, 'ether': {'THF', '2-MeTHF', 'Et2O', 'DME', 'diglyme', '1,4-dioxane', 'CPME', 'MTBE'}, 'nitrile': {'MeCN'}, 'alcohol_polyol': {'MeOH', 'EtOH', 'IPA', 't-AmOH', 't-BuOH', 'tBuOH', 'TFE', '1-Pentanol', '1-Propanol', 'EG', 'propane-1,2,3-triol'}, 'water': {'H2O'}, 'aromatic_hydrocarbon': {'benzene', 'toluene', 'xylene', 'p-xylene', 'o-xylene', 'm-xylene', 'mesitylene'}, 'halogenated': {'DCE', 'CCl4', 'PhCl', 'benzotrifluoride'}, 'amine_aza_heterocycle': {'TEA', 'iPr2NH', 'sBuNH2', 'Py'}, 'ester': {'EtOAc', 'DEM'}, 'alkane': {'hexane', 'Cyclohexane'}, 'phosphoramide': {'HMPA'}}

def solvent_features(name):
    text = re.sub('\\s*\\([0-9:. ]+\\)\\s*$', '', name)
    ingredients = text.split('/')
    labels = []
    unknown = []
    for ingredient in ingredients:
        hit = [group for group, members in SOLVENT_GROUPS.items() if ingredient in members]
        labels.extend(hit)
        if not hit:
            unknown.append(ingredient)
    return (ingredients, sorted(set(labels)), unknown)

def note_flags(note):
    flags = []
    patterns = {'explicit_duplicate_note': ('重复', '一致', '一样'), 'invalid_or_uncollected_note': ('无效', '作废', '无有效', '无数据', '未收集'), 'historical_change_or_correction_note': ('更换', '改正'), 'amount_difference_note': ('当量不同',), 'uncertain_dispensed_identity_note': ('疑似',), 'standard_control_note': ('标准孔',)}
    for label, words in patterns.items():
        if any((word in note for word in words)):
            flags.append(label)
    return flags

def csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recipes = reader_tables.read_excel(CONDITION_FILE, sheet_name='Sheet2', dtype=object, keep_default_na=False)
    compounds = reader_tables.read_excel(COMPOUND_FILE, sheet_name='Sheet1', dtype=object, keep_default_na=False)
    reader_tables.validate_registry_scope(recipes, compounds)
    assert not recipes.duplicated(['reaction_type', 'condition_id']).any()
    exact_fields = [c for c in recipes.columns if c not in {'note', 'reaction_type', 'condition_id'}]
    details, components = ([], [])
    for idx, raw in recipes.iterrows():
        row = {k: token(v) for k, v in raw.items()}
        family_list, missing_ids, missing_eq, orphan_eq, component_ids = ([], [], [], [], [])
        for i in range(1, 7):
            identity, dose, name = (row[f'Reagent{i}_id'], row[f'Reagent{i}_HTE_Equiv'], row[f'Reagent{i}_Standardized Name'])
            if identity or name or row[f'Reagent{i}_HTE_CAS']:
                component_ids.append(identity)
                if not identity:
                    missing_ids.append(i)
                if not dose:
                    missing_eq.append(i)
                labels = families(name)
                family_list.extend(labels)
                components.append({'reaction_type': row['reaction_type'], 'condition_id': row['condition_id'], 'slot': i, 'component_id': identity, 'component_name': name, 'component_cas': row[f'Reagent{i}_HTE_CAS'], 'equiv': dose, 'putative_name_based_families_json': dumps(labels), 'family_rule_version': FAMILY_RULE_VERSION})
            elif dose:
                orphan_eq.append(i)
        invariant = sorted([(row[f'Reagent{i}_id'], row[f'Reagent{i}_HTE_Equiv']) for i in range(1, 7) if row[f'Reagent{i}_id']])
        invariant += [('solvent', row['solvent_id']), ('temperature', row['temp_id']), ('time', row['time_id'])]
        ingredients, solvent_groups, unknown_solvents = solvent_features(row['Standardized Name'])
        details.append({**row, 'source_sheet': 'Sheet2', 'source_row': int(reader_tables.source_rows(CONDITION_FILE, 'Sheet2', original=True)[idx]), 'recipe_full_id_dose_sha256': fingerprint([row[c] for c in CORE_FIELDS]), 'recipe_all_28_metadata_fields_sha256': fingerprint([row[c] for c in exact_fields]), 'recipe_ignore_6_doses_sha256': fingerprint([row[c] for c in IDENTITY_FIELDS]), 'recipe_slot_invariant_id_dose_sha256': fingerprint(invariant), 'component_count': len(component_ids), 'component_ids_json': dumps(component_ids), 'putative_name_based_families_json': dumps(sorted(set(family_list))), 'missing_component_id_slots_json': dumps(missing_ids), 'missing_dose_slots_json': dumps(missing_eq), 'orphan_dose_slots_json': dumps(orphan_eq), 'solvent_ingredients_json': dumps(ingredients), 'solvent_putative_families_json': dumps(solvent_groups), 'unclassified_solvent_ingredients_json': dumps(unknown_solvents), 'note_flags_json': dumps(note_flags(row['note']))})
    detail = pd.DataFrame(details)
    component = pd.DataFrame(components)
    signatures = ['recipe_full_id_dose_sha256', 'recipe_all_28_metadata_fields_sha256', 'recipe_ignore_6_doses_sha256', 'recipe_slot_invariant_id_dose_sha256']
    duplicate_rows, summaries = ([], [])
    for rxn, group in detail.groupby('reaction_type', sort=True):
        summary = {'reaction_type': rxn, 'condition_rows': len(group), 'condition_ids': group.condition_id.nunique(), 'component_id_count': len(set((x for value in group.component_ids_json for x in json.loads(value)))), 'component_count_min': int(group.component_count.min()), 'component_count_max': int(group.component_count.max()), 'solvent_id_count': group.solvent_id.nunique(), 'temperature_levels_json': dumps(sorted(set(group.temp_id))), 'time_levels_json': dumps(sorted(set(group.time_id))), 'temperature_level_count': group.temp_id.nunique(), 'time_level_count': group.time_id.nunique(), 'nonempty_note_rows': int(group.note.ne('').sum()), 'recipe_fields_in_primary_fingerprint': len(CORE_FIELDS), 'redundant_name_cas_fields_ignored': len(exact_fields) - len(CORE_FIELDS), 'unknown_solvent_recipe_rows': int(group.unclassified_solvent_ingredients_json.ne('[]').sum()), 'missing_component_id_rows': int(group.missing_component_id_slots_json.ne('[]').sum()), 'missing_dose_rows': int(group.missing_dose_slots_json.ne('[]').sum()), 'orphan_dose_rows': int(group.orphan_dose_slots_json.ne('[]').sum())}
        fam_counts = Counter((x for value in group.putative_name_based_families_json for x in json.loads(value)))
        summary['putative_family_recipe_counts_json'] = dumps(dict(sorted(fam_counts.items())))
        solvent_counts = Counter((x for value in group.solvent_putative_families_json for x in json.loads(value)))
        summary['putative_solvent_family_recipe_counts_json'] = dumps(dict(sorted(solvent_counts.items())))
        summary['unclassified_component_occurrences'] = int(sum(('unclassified' in json.loads(v) for v in component.loc[component.reaction_type.eq(rxn), 'putative_name_based_families_json'])))
        for signature in signatures:
            counts = group[signature].value_counts()
            summary[signature + '_unique'] = len(counts)
            summary[signature + '_duplicate_groups'] = int((counts > 1).sum())
            summary[signature + '_duplicate_excess'] = len(group) - len(counts)
            for value, count in counts[counts > 1].items():
                matching = group[group[signature].eq(value)]
                duplicate_rows.append({'reaction_type': rxn, 'fingerprint_definition': signature, 'fingerprint': value, 'group_size': count, 'condition_ids_json': dumps(matching.condition_id.tolist()), 'source_rows_json': dumps(matching.source_row.tolist()), 'notes_json': dumps(matching.note.tolist()), 'full_id_dose_unique_within_group': matching.recipe_full_id_dose_sha256.nunique(), 'all_28_fields_unique_within_group': matching.recipe_all_28_metadata_fields_sha256.nunique()})
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    duplicates = pd.DataFrame(duplicate_rows)
    source_hashes = {p.name: {'sha256': sha256(p), 'bytes': reader_tables.source_evidence(p)['bytes']} for p in (CONDITION_FILE, COMPOUND_FILE)}
    compound_collisions = compounds.loc[compounds.duplicated('Compound ID', keep=False)].copy()
    compound_collisions['source_row'] = reader_tables.source_rows(COMPOUND_FILE, 'Sheet1', original=True)[compound_collisions.index]
    referenced = set(component.component_id) | set(detail.solvent_id)
    registry_ids = set(compounds['Compound ID'].map(token))
    registry_name_sets = compounds.groupby('Compound ID')['Standardized Name'].agg(lambda s: set(map(token, s))).to_dict()
    component['registry_id_found'] = component.component_id.isin(registry_ids)
    component['registry_name_matches'] = [name in registry_name_sets.get(cid, set()) for cid, name in zip(component.component_id, component.component_name)]
    audit = {'source_hashes': source_hashes, 'condition_rows': len(detail), 'reaction_types': detail.reaction_type.nunique(), 'compound_rows': len(compounds), 'compound_unique_ids': compounds['Compound ID'].nunique(), 'compound_duplicate_id_rows': len(compound_collisions), 'referenced_ids_absent_from_compound_registry': sorted(referenced - registry_ids), 'component_name_mismatches_to_registry_occurrences': int((~component.registry_name_matches).sum()), 'primary_fingerprint_fields': CORE_FIELDS, 'all_metadata_fingerprint_fields': exact_fields, 'scope': 'Group fingerprints within reaction_type only; no cross-reaction condition-ID joins', 'numeric_normalization': 'Decimal normalization; stripped leading/trailing whitespace; no missing-value imputation', 'ignored_primary_fields': [c for c in recipes if c not in CORE_FIELDS], 'sum_primary_duplicate_excess': int(summary.recipe_full_id_dose_sha256_duplicate_excess.sum()), 'sum_all_metadata_duplicate_excess': int(summary.recipe_all_28_metadata_fields_sha256_duplicate_excess.sum()), 'sum_dose_ignored_duplicate_excess': int(summary.recipe_ignore_6_doses_sha256_duplicate_excess.sum()), 'sum_slot_invariant_duplicate_excess': int(summary.recipe_slot_invariant_id_dose_sha256_duplicate_excess.sum()), 'family_rule_version': FAMILY_RULE_VERSION, 'full_registry_not_observation_subset': True, 'not_encoded': ['concentration', 'total volumes', 'water content beyond solvent label', 'gas pressure/atmosphere', 'order/rate of addition', 'mixing', 'light exposure', 'actual chemical inventory/stock', 'batch/date/plate', 'calibration', 'time-varying recipe corrections']}
    csv(summary, 'condition_recipe_audit.csv')
    csv(detail, 'condition_recipe_audit_details.csv')
    csv(duplicates, 'condition_recipe_audit_duplicates.csv')
    csv(component, 'condition_recipe_audit_components.csv')
    csv(compound_collisions, 'condition_recipe_audit_compound_collisions.csv')
    (OUT / 'condition_recipe_audit_provenance.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    restored = pd.read_csv(OUT / 'condition_recipe_audit_details.csv', keep_default_na=False)
    assert len(restored) == len(recipes)
    assert not restored.duplicated(['reaction_type', 'condition_id']).any()
    assert len(CORE_FIELDS) == 15 and len(exact_fields) == 28
    assert sum(summary.condition_rows) == len(recipes)
    assert not referenced - registry_ids, 'Referenced compound IDs absent from registry'
    assert component.registry_name_matches.all(), 'Component name/registry mismatch needs review'
    assert not detail.missing_component_id_slots_json.ne('[]').any(), 'Primary fingerprint requires component IDs'
    assert not detail.orphan_dose_slots_json.ne('[]').any(), 'Inspect orphan doses before using slot-invariant fingerprint'
    hiyama = detail[detail.reaction_type.eq('hiyama') & detail.condition_id.isin(['C43', 'C46'])]
    assert len(hiyama) == 2 and hiyama.recipe_all_28_metadata_fields_sha256.nunique() == 1
    print(summary[['reaction_type', 'condition_rows', 'recipe_full_id_dose_sha256_unique', 'recipe_full_id_dose_sha256_duplicate_excess', 'recipe_ignore_6_doses_sha256_duplicate_excess', 'solvent_id_count', 'temperature_level_count', 'time_level_count']].to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
