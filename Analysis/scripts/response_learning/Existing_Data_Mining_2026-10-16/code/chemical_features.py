"""Deterministic, interpretable substrate annotations from the designated raw HTE corpus.

Run with Python plus pandas, openpyxl, pyarrow and RDKit. Inputs remain read-only."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
SMARTS = {'aldehyde': '[CX3H1](=[OX1])[#6]', 'ketone': '[#6][#6X3](=[OX1])[#6]', 'phenol': '[OX2H1]-[c]', 'carboxylic_acid': '[CX3](=[OX1])[OX2H1]', 'carboxylate': '[CX3](=[OX1])[O-]', 'ester': '[$([CX3H1]),$([CX3]-[#6])](=[OX1])[OX2][#6]', 'aryl_halide': '[c]-[F,Cl,Br,I]', 'halogen': '[F,Cl,Br,I]', 'fluorinated_aromatic': '[c]-[F]', 'nitro': '[N+](=[O])[O-]', 'nitrile': '[C]#[N]', 'amide': '[NX3,n]-,:[CX3,c](=[OX1])', 'sulfonamide': '[NX3][SX4](=[OX1])(=[OX1])', 'boronic_acid_or_ester': '[#6]-[B](-[O])-[O]'}
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}
DESCRIPTOR_COLUMNS = ['desc_mw', 'desc_logp', 'desc_tpsa', 'desc_hbd', 'desc_hba', 'desc_rotatable_bonds', 'desc_formal_charge', 'desc_aromatic_rings', 'desc_heavy_atoms', 'desc_fraction_csp3', 'desc_heteroaryl_n_count', 'desc_max_ortho_heavy_substituents']
AMINE_FLAGS = [f'flag_{degree}_{kind}' for degree in ('primary', 'secondary', 'tertiary') for kind in ('aniline', 'alkyl_amine', 'amine')]
CARBONYL_FLAGS = [f'flag_{kind}_{group}' for kind in ('aromatic', 'aliphatic') for group in ('aldehyde', 'ketone')]
FLAG_COLUMNS = AMINE_FLAGS + CARBONYL_FLAGS + ['flag_any_amine', 'flag_aldehyde', 'flag_ketone', 'flag_pyridine_like', 'flag_quinoline_like', 'flag_azole', 'flag_pyrrole_like', 'flag_ortho_substituted_aryl_halide', 'flag_ortho_substituted_aniline'] + [f'flag_{name}' for name in SMARTS if name not in ('aldehyde', 'ketone')]

def amine_atoms(mol: Chem.Mol) -> list[tuple[int, str, str]]:
    """Neutral carbon-substituted amines, excluding conjugated acyl/S/P nitrogens."""
    result = []
    degree_names = {1: 'primary', 2: 'secondary', 3: 'tertiary'}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetIsAromatic() or atom.GetFormalCharge():
            continue
        neighbors = list(atom.GetNeighbors())
        if not neighbors or not all((n.GetAtomicNum() == 6 for n in neighbors)):
            continue
        if not all((b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())):
            continue
        if len(neighbors) + atom.GetTotalNumHs() != 3:
            continue
        conjugated = any((b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() in (7, 8, 16) for n in neighbors for b in n.GetBonds()))
        if conjugated:
            continue
        kind = 'aniline' if any((n.GetIsAromatic() for n in neighbors)) else 'alkyl_amine'
        result.append((atom.GetIdx(), degree_names[len(neighbors)], kind))
    return result

def ring_flags(mol: Chem.Mol) -> dict[str, int]:
    aromatic_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if all((mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))]
    six_n = [r for r in aromatic_rings if len(r) == 6 and any((mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and mol.GetAtomWithIdx(i).GetTotalNumHs() == 0 and (mol.GetAtomWithIdx(i).GetDegree() == 2) for i in r))]
    benzene = [r for r in aromatic_rings if len(r) == 6 and all((mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in r))]
    one_n = [r for r in six_n if sum((mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in r)) == 1]
    five_n = [r for r in aromatic_rings if len(r) == 5 and any((mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in r))]
    pyrrole_like = any((mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and (mol.GetAtomWithIdx(i).GetTotalNumHs() == 1 or mol.GetAtomWithIdx(i).GetDegree() == 3) for r in five_n for i in r))
    return {'flag_pyridine_like': int(bool(six_n)), 'flag_quinoline_like': int(any((len(a & b) == 2 for a in one_n for b in benzene))), 'flag_azole': int(bool(five_n)), 'flag_pyrrole_like': int(pyrrole_like)}

def ortho_occupancy(mol: Chem.Mol, anchor: int) -> int:
    """Maximum occupied adjacent sites in any aromatic six-ring containing anchor.

    Exocyclic heavy substituent bonds must be outside rings, so fused ring bonds
    themselves are not counted as ortho substituents.
    """
    counts = []
    for ring in mol.GetRingInfo().AtomRings():
        if anchor not in ring or len(ring) != 6:
            continue
        if not all((mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
            continue
        adjacent = [n for n in mol.GetAtomWithIdx(anchor).GetNeighbors() if n.GetIdx() in ring]
        counts.append(sum((any((b.GetOtherAtom(n).GetAtomicNum() > 1 and (not b.IsInRing()) for b in n.GetBonds())) for n in adjacent)))
    return max(counts, default=0)

def annotate(smiles: str) -> dict[str, Any]:
    row = {'smiles': smiles, 'canonical_smiles': None, 'scaffold_smiles': None, 'parse_valid': False, 'parse_error': None}
    row.update({c: np.nan for c in DESCRIPTOR_COLUMNS + FLAG_COLUMNS})
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        row['parse_error'] = 'RDKit MolFromSmiles returned None or empty molecule'
        return row
    row.update({'canonical_smiles': Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), 'scaffold_smiles': MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True), 'parse_valid': True, 'desc_mw': Descriptors.MolWt(mol), 'desc_logp': Crippen.MolLogP(mol), 'desc_tpsa': rdMolDescriptors.CalcTPSA(mol), 'desc_hbd': Lipinski.NumHDonors(mol), 'desc_hba': Lipinski.NumHAcceptors(mol), 'desc_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol), 'desc_formal_charge': Chem.GetFormalCharge(mol), 'desc_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol), 'desc_heavy_atoms': mol.GetNumHeavyAtoms(), 'desc_fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol), 'desc_heteroaryl_n_count': sum((a.GetAtomicNum() == 7 and a.GetIsAromatic() for a in mol.GetAtoms()))})
    row.update({c: 0 for c in FLAG_COLUMNS})
    for name, pattern in PATTERNS.items():
        row[f'flag_{name}'] = int(mol.HasSubstructMatch(pattern))
    amines = amine_atoms(mol)
    aniline_anchors = []
    for atom_idx, degree, kind in amines:
        row[f'flag_{degree}_{kind}'] = 1
        row[f'flag_{degree}_amine'] = 1
        if kind == 'aniline':
            aniline_anchors.extend((n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if n.GetIsAromatic()))
    row['flag_any_amine'] = int(bool(amines))
    carbonyl_anchors = []
    for group in ('aldehyde', 'ketone'):
        for match in mol.GetSubstructMatches(PATTERNS[group]):
            carbonyl_idx = match[0] if group == 'aldehyde' else match[1]
            attached_c = [n for n in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors() if n.GetAtomicNum() == 6]
            aromatic = any((n.GetIsAromatic() for n in attached_c))
            row[f"flag_{('aromatic' if aromatic else 'aliphatic')}_{group}"] = 1
            carbonyl_anchors.extend((n.GetIdx() for n in attached_c if n.GetIsAromatic()))
    halide_anchors = [match[0] for match in mol.GetSubstructMatches(PATTERNS['aryl_halide'])]
    row['flag_ortho_substituted_aryl_halide'] = int(any((ortho_occupancy(mol, i) for i in halide_anchors)))
    row['flag_ortho_substituted_aniline'] = int(any((ortho_occupancy(mol, i) for i in aniline_anchors)))
    row['desc_max_ortho_heavy_substituents'] = max((ortho_occupancy(mol, i) for i in halide_anchors + aniline_anchors + carbonyl_anchors), default=0)
    row.update(ring_flags(mol))
    return row

def self_test() -> None:
    cases = {'Nc1ccccc1': {'flag_primary_aniline': 1, 'flag_primary_alkyl_amine': 0}, 'NCc1ccccc1': {'flag_primary_aniline': 0, 'flag_primary_alkyl_amine': 1}, 'CNc1ccccc1': {'flag_secondary_aniline': 1}, 'CN(C)c1ccccc1': {'flag_tertiary_aniline': 1}, 'CC(=O)Nc1ccccc1': {'flag_any_amine': 0, 'flag_amide': 1}, 'O=Cc1ccccc1': {'flag_aromatic_aldehyde': 1, 'flag_ketone': 0}, 'CC=O': {'flag_aliphatic_aldehyde': 1, 'flag_aromatic_aldehyde': 0}, 'CC(=O)c1ccccc1': {'flag_aromatic_ketone': 1, 'flag_aldehyde': 0}, 'CC(C)=O': {'flag_aliphatic_ketone': 1}, 'n1ccccc1': {'flag_pyridine_like': 1, 'flag_pyrrole_like': 0, 'flag_azole': 0}, 'c1cc[nH]c1': {'flag_pyridine_like': 0, 'flag_pyrrole_like': 1, 'flag_azole': 1}, 'Cn1cccc1': {'flag_pyrrole_like': 1}, 'c1ccc2ncccc2c1': {'flag_quinoline_like': 1}, 'Cc1ccccc1Br': {'flag_ortho_substituted_aryl_halide': 1, 'desc_max_ortho_heavy_substituents': 1}, 'Cc1ccc(Br)cc1': {'flag_ortho_substituted_aryl_halide': 0}, 'CC(=O)OC': {'flag_ester': 1, 'flag_ketone': 0, 'flag_carboxylic_acid': 0}, 'CCOC=O': {'flag_ester': 1, 'flag_aldehyde': 0}, 'COC(=O)OC': {'flag_ester': 0}, 'COC(=O)N': {'flag_ester': 0, 'flag_amide': 1, 'flag_any_amine': 0}, 'Cc1ccc(=O)[nH]c1': {'flag_amide': 1, 'flag_pyridine_like': 0}, 'O=Cc1coc2ccccc2c1=O': {'flag_ketone': 1, 'flag_aldehyde': 1}, 'CC(=O)O': {'flag_carboxylic_acid': 1, 'flag_ester': 0}, 'Oc1ccccc1': {'flag_phenol': 1}}
    for smiles, expected in cases.items():
        actual = annotate(smiles)
        for column, value in expected.items():
            assert actual[column] == value, (smiles, column, actual[column], value)
    assert annotate('invalid_smiles')['parse_valid'] is False
    print(f'Chemical distinction self-tests passed: {len(cases)} structures plus invalid SMILES.')

def read_substrates(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, sheets = ([], [])
    wanted = {f'sub_{slot}_{kind}' for slot in (1, 2) for kind in ('smiles', 'FG')}
    for path in sorted(reader_tables.source_paths(raw_root)):
        if path.suffix.lower() not in ('.csv', '.xlsx'):
            continue
        if path.suffix.lower() == '.csv':
            tables = {'CSV': reader_tables.read_csv(path, usecols=lambda c: c in wanted, dtype=str)}
        else:
            tables = {}
            with reader_tables.ExcelFile(path) as workbook:
                for sheet in workbook.sheet_names:
                    header = reader_tables.read_excel(workbook, sheet_name=sheet, nrows=0)
                    columns = [c for c in header if c in wanted]
                    if columns:
                        tables[sheet] = reader_tables.read_excel(workbook, sheet_name=sheet, usecols=columns, dtype=str)
                    else:
                        sheets.append({'source_file': path.name, 'source_sheet': sheet, 'included': False, 'reason': 'No substrate SMILES columns', 'n_rows': 0})
        for sheet, table in tables.items():
            included = any((f'sub_{slot}_smiles' in table for slot in (1, 2)))
            sheets.append({'source_file': path.name, 'source_sheet': sheet, 'included': included, 'reason': 'Substrate columns' if included else 'No substrate SMILES columns', 'n_rows': len(table)})
            if not included:
                continue
            for slot in (1, 2):
                column = f'sub_{slot}_smiles'
                if column not in table:
                    continue
                fg = f'sub_{slot}_FG'
                part = pd.DataFrame({'smiles': table[column], 'source_fg': table[fg] if fg in table else None})
                part = part[part.smiles.notna() & part.smiles.str.strip().ne('')]
                part['source_file'], part['source_sheet'], part['substrate_slot'] = (path.name, sheet, slot)
                records.append(part.groupby(['smiles', 'source_fg', 'source_file', 'source_sheet', 'substrate_slot'], dropna=False).size().rename('n_reaction_rows').reset_index())
    return (pd.concat(records, ignore_index=True), pd.DataFrame(sheets))
FG_TO_FLAGS = {'primary amine': ['flag_primary_amine'], 'secondary amine': ['flag_secondary_amine'], 'tertiary amine': ['flag_tertiary_amine'], 'aldehyde': ['flag_aldehyde'], 'ketone': ['flag_ketone'], 'phenol': ['flag_phenol'], 'carboxylic acid': ['flag_carboxylic_acid', 'flag_carboxylate'], 'ester': ['flag_ester'], 'aryl halides': ['flag_aryl_halide'], 'pyridine': ['flag_pyridine_like'], 'quinoline': ['flag_quinoline_like'], 'isoquinoline': ['flag_quinoline_like'], 'pyrrole ring': ['flag_pyrrole_like'], 'nitro': ['flag_nitro'], 'cyano': ['flag_nitrile'], 'amide': ['flag_amide'], 'sulfonamide': ['flag_sulfonamide'], 'boronic acid': ['flag_boronic_acid_or_ester'], 'boronic acid ester': ['flag_boronic_acid_or_ester']}

def write_outputs(raw_root: Path, output_root: Path) -> None:
    audit = output_root / 'audit'
    processed = output_root / 'processed'
    reports = output_root / 'reports'
    for folder in (audit, processed, reports):
        folder.mkdir(parents=True, exist_ok=True)
    usage, sheets = read_substrates(raw_root)
    smiles = sorted(usage.smiles.unique())
    features = pd.DataFrame([annotate(s) for s in smiles])
    features.to_parquet(processed / 'molecular_features.parquet', index=False)
    sheets.to_csv(audit / 'molecular_feature_source_sheets.csv', index=False)
    usage.to_csv(audit / 'molecular_feature_substrate_usage.csv', index=False)
    features.loc[~features.parse_valid].to_csv(audit / 'molecular_feature_parse_failures.csv', index=False)
    aliases = features[features.parse_valid].copy()
    aliases['n_supplied_aliases'] = aliases.groupby('canonical_smiles').smiles.transform('size')
    aliases.loc[aliases.n_supplied_aliases > 1, ['smiles', 'canonical_smiles', 'scaffold_smiles', 'n_supplied_aliases']].to_csv(audit / 'molecular_feature_canonical_aliases.csv', index=False)
    examples = []
    for flag in FLAG_COLUMNS:
        selected = features[features[flag].eq(1)].sort_values('smiles')
        for _, row in selected.head(5).iterrows():
            examples.append({'flag': flag, 'n_positive_strings': len(selected), 'smiles': row.smiles, 'canonical_smiles': row.canonical_smiles})
    pd.DataFrame(examples).to_csv(audit / 'molecular_feature_flag_examples.csv', index=False)
    agreement = []
    joined = usage.merge(features, on='smiles', validate='many_to_one')
    for tag, flags in FG_TO_FLAGS.items():
        tagged = joined[joined.source_fg.fillna('').map(lambda s: tag in {t.strip().lower() for t in s.split(';')})]
        tagged = tagged.drop_duplicates(['smiles', 'source_fg', 'source_file', 'substrate_slot'])
        for _, row in tagged.iterrows():
            agreement.append({'source_tag': tag, 'smiles': row.smiles, 'source_file': row.source_file, 'substrate_slot': row.substrate_slot, 'source_fg': row.source_fg, 'mapped_flags': ';'.join(flags), 'parse_valid': row.parse_valid, 'annotation_agrees': bool(row[flags].eq(1).any()) if row.parse_valid else None})
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(audit / 'molecular_feature_fg_agreement.csv', index=False)
    agreement[agreement.annotation_agrees.eq(False)].to_csv(audit / 'molecular_feature_fg_disagreements.csv', index=False)
    agreement.groupby('source_tag').agg(n_tagged_contexts=('smiles', 'size'), n_unique_smiles=('smiles', 'nunique'), n_agree=('annotation_agrees', 'sum')).reset_index().to_csv(audit / 'molecular_feature_fg_agreement_summary.csv', index=False)
    summary = {'rdkit_version': rdBase.rdkitVersion, 'raw_root': str(raw_root.resolve()), 'n_unique_supplied_smiles': len(features), 'n_parse_valid': int(features.parse_valid.sum()), 'n_canonical_identities': int(features.canonical_smiles.nunique()), 'n_scaffolds_including_acyclic_empty': int(features.scaffold_smiles.nunique()), 'n_source_reaction_rows': int(sheets.loc[sheets.included, 'n_rows'].sum()), 'n_substrate_occurrences': int(usage.n_reaction_rows.sum()), 'n_alias_strings': int(aliases.n_supplied_aliases.gt(1).sum()), 'n_fg_contexts_audited': len(agreement), 'n_fg_disagreements': int(agreement.annotation_agrees.eq(False).sum()), 'feature_columns': DESCRIPTOR_COLUMNS + FLAG_COLUMNS}
    pd.DataFrame([{'metric': key, 'value': json.dumps(value) if isinstance(value, list) else value} for key, value in summary.items()]).to_csv(audit / 'molecular_feature_summary.csv', index=False)
    write_methods(reports / 'CHEMICAL_FEATURE_METHODS.md', summary)
    print(json.dumps(summary, indent=2))

def write_methods(path: Path, summary: dict[str, Any]) -> None:
    text = f"# Molecular annotation methods\n\n## Scope and provenance\n\nOnly `HTE data/01_原始数据` is read. Every CSV and every XLSX sheet is inspected for `sub_1_smiles` and `sub_2_smiles`; metadata-only sheets and products are excluded. The original supplied strings are retained in `smiles`; outer whitespace is removed only for RDKit parsing. No external structures, labels, corpora, fingerprints, outcomes, or model fits are used. Source columns and annotation functions are defined in `code/chemical_features.py`.\n\nThe run contains {summary['n_source_reaction_rows']:,} raw reaction rows, {summary['n_substrate_occurrences']:,} nonempty substrate occurrences, {summary['n_unique_supplied_smiles']:,} supplied strings, {summary['n_parse_valid']:,} valid strings, and {summary['n_canonical_identities']:,} distinct canonical isomeric SMILES. RDKit version: `{summary['rdkit_version']}`. Run `python code/chemical_features.py --self-test` for chemically meaningful distinction tests; run without arguments to regenerate all outputs.\n\n## Identity and descriptors\n\n`canonical_smiles` uses RDKit canonical isomeric SMILES after standard sanitization. Salts are retained, charges are retained, and tautomers are not normalized. `scaffold_smiles` is the RDKit Bemis-Murcko scaffold with chirality retained; acyclic molecules have the empty scaffold. These are identity/grouping variables, not fitted predictors. A molecule that cannot be parsed has `parse_valid=False`, an error message and missing numeric features, never a fabricated negative flag.\n\nDescriptors are RDKit MolWt (g/mol), Crippen MolLogP, topological polar surface area (square angstrom), Lipinski H-bond donors/acceptors, default RDKit strict rotatable bonds, formal charge, aromatic ring count, heavy-atom count and fraction of sp3 carbons. `desc_heteroaryl_n_count` counts aromatic N atoms of all aromatic N types. Units and unscaled raw values are retained.\n\n## Interpretable structural flags\n\nFlags are binary presence/absence, allowing multiple motifs per molecule. Primary/secondary/tertiary amines are neutral nonaromatic nitrogens with respectively one/two/three carbon neighbors, only single bonds, and total carbon-neighbor plus H count three. Nitrogens directly attached to C(=O/S/N) are excluded. This excludes amides, carbamates, ureas and amidines; N-O, N-N, N-S and N-P compounds are not classified as ordinary amines. `aniline` means direct attachment to at least one aromatic carbon (including heteroaromatic carbon); mixed aryl/alkyl amines belong to this class. `alkyl_amine` means all carbon neighbors are nonaromatic. Thus benzylamine is alkyl amine, whereas aniline is aryl amine. Protonated and quaternary ammonium are retained as structures but do not receive neutral amine flags.\n\nAldehydes require C(=O)(H)-C, so formic acid/formamides are excluded; ketones require C-C(=O)-C. Aromatic carbonyl means the carbonyl carbon is directly bonded to an aromatic carbon; aliphatic means no such direct bond. An aryl/alkyl ketone is aromatic. Additional functional group SMARTS appear below.\n\n`pyridine_like` indicates at least one aromatic six-membered ring containing aromatic N with zero H and degree two; diazines and fused examples may also qualify. `quinoline_like` requires a six-membered aromatic ring with exactly one N sharing two atoms with an all-carbon aromatic six-ring (includes isoquinoline). `azole` means any aromatic five-ring containing N, including pyrrole and fused five-rings. `pyrrole_like` requires a five-ring aromatic N with one H or degree three, including N-substituted pyrrole and indole. These are graph motifs, not claims about nitrogen basicity or coordination.\n\nFor aryl halide, aryl amine and aromatic carbonyl attachment atoms, `desc_max_ortho_heavy_substituents` is the maximum number (0-2) of adjacent positions on a six-membered aromatic ring that carry an exocyclic heavy atom through a nonring bond. Fused ring bonds themselves are excluded. `flag_ortho_substituted_aryl_halide` and `flag_ortho_substituted_aniline` indicate at least one occupied adjacent position at their respective attachment site. The descriptor is zero when no eligible site exists. These topological occupancy measures do not measure 3D steric energy, conformation, catalyst accessibility or mechanism.\n\n## SMARTS panel\n\n| Flag suffix | RDKit SMARTS |\n| --- | --- |\n"
    text += '\n'.join((f'| `{name}` | `{smarts}` |' for name, smarts in SMARTS.items()))
    text += f"\n\n`fluorinated_aromatic` specifically means aromatic C-F, not every aryl CF3 group. Carboxylates are separated from neutral acids. The ester pattern accepts formates and carbon-substituted carboxylic esters, requiring carbon on the O side; carbonates/carbamates are excluded. The amide query accepts aromatic lactams such as pyridones. Ketone carbonyl carbon may be aromatic in RDKit's representation of a conjugated ring. Halogens include covalent or disconnected F/Cl/Br/I; aryl halide is the covalent aromatic C-X subset.\n\n## Audits and limitations\n\n`audit/molecular_feature_source_sheets.csv` enumerates input sheets. `molecular_feature_substrate_usage.csv` preserves source file, sheet, slot, source functional group tags and row frequency. `molecular_feature_parse_failures.csv` enumerates every failed supplied string. `molecular_feature_canonical_aliases.csv` lists all {summary['n_alias_strings']:,} strings belonging to canonical identities with multiple supplied spellings; no alias is silently dropped. `molecular_feature_flag_examples.csv` gives up to five deterministic examples per positive flag.\n\nThe source functional group audit maps only explicit meaningful tags to corresponding graph flags, checks positive source assertions, and reports all comparisons and disagreements. It does not treat absence of a source tag as a negative label: source tags are not guaranteed exhaustive and are not ground truth. Contexts are deduplicated by supplied string, source tag set, file and substrate slot, not weighted by thousands of repeated reaction rows. There are {summary['n_fg_contexts_audited']:,} audited contexts and {summary['n_fg_disagreements']:,} disagreements. Narrow neutral-amine definitions, broad source labels, protonation, fused rings and source labeling errors can produce disagreement; inspect individual structures before choosing labels for a scientific claim.\n\nThe supplied graph annotations describe structure only. Any later associations with retrospective yield regimes are observational and cannot establish causality, mechanism, selectivity, conversion or prospective performance. Scaffold grouping based on empty acyclic scaffolds may be especially coarse. Numeric and flag columns are listed verbatim in `audit/molecular_feature_summary.csv`; strings and `parse_valid` must be excluded from a numeric model unless explicitly used for identity grouping.\n"
    path.write_text(text, encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    output_root = args.output_root or Path(__file__).resolve().parents[1]
    import sys as _m1_sys
    _m1_sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
    from m1_raw import RAW_DIR
    raw_root = args.raw_root or RAW_DIR
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    write_outputs(raw_root, output_root)
if __name__ == '__main__':
    main()
