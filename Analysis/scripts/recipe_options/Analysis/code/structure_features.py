"""Interpretable RDKit features for every unique raw substrate SMILES."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
AMINE_EXCLUSIONS = ';!$(N-C=O);!$(N-S(=O)=O);!$(N-P(=O))'
PRIMARY_N = '[N;X3;H2;+0' + AMINE_EXCLUSIONS + ']'
SMARTS_FEATURES = {'primary_arylamine': (PRIMARY_N + '-[c]', 'Neutral primary amine directly attached to aromatic carbon; carbonyl, sulfonyl and phosphoryl N excluded.'), 'primary_alkylamine': (PRIMARY_N + '-[CX4]', 'Neutral primary amine directly attached to an sp3 carbon; carbonyl, sulfonyl and phosphoryl N excluded.'), 'secondary_amine': ('[N;X3;H1;+0' + AMINE_EXCLUSIONS + ']([#6])[#6]', 'Neutral nonamide amine nitrogen bearing one H and two carbon substituents.'), 'aniline': ('[N;X3;+0' + AMINE_EXCLUSIONS + ']-[c]', 'Neutral nonamide amine N directly bonded to aromatic carbon, including substituted anilines.'), 'NH_heteroaromatic': ('[nH]', 'An aromatic nitrogen bearing H in the recorded molecular graph.'), 'acid_present': ('[CX3](=[OX1])[OX2H1,OX1-]', 'Carboxylic acid or carboxylate present; esters excluded.'), 'ketone': ('[CX3](=[OX1])([#6])[#6]', 'Carbonyl carbon bearing two carbon substituents.'), 'pyridine_like_N': ('[nH0;+0]', 'Neutral aromatic nitrogen bearing no hydrogen (pyridine-like lone pair).'), 'aryl_halide': ('[c]-[F,Cl,Br,I]', 'F, Cl, Br or I directly attached to aromatic carbon.'), 'aryl_F': ('[c]-[F]', 'Fluorine directly attached to aromatic carbon.'), 'aryl_Cl': ('[c]-[Cl]', 'Chlorine directly attached to aromatic carbon.'), 'aryl_Br': ('[c]-[Br]', 'Bromine directly attached to aromatic carbon.'), 'methoxy': ('[OX2]([CH3])[#6;!$(C=O)]', 'Methyl ether group; methyl esters, carbonates and carbamates excluded.'), 'benzylic_primary_amine': (PRIMARY_N + '-[CX4]-[c]', 'Neutral primary amine attached to an sp3 carbon directly attached to aromatic carbon; alpha substitution permitted.'), 'free_alcohol': ('[OX2H1]-[CX4]', 'Unprotected O-H directly attached to an sp3 carbon.'), 'phenol': ('[OX2H1]-[c]', 'O-H directly attached to aromatic carbon.'), 'amide': ('[CX3](=[OX1])-[#7]', 'Carbonyl carbon directly attached to nitrogen, including lactams and ureas.'), 'tertiary_amine': ('[N;X3;H0;+0' + AMINE_EXCLUSIONS + ']([#6])([#6])[#6]', 'Neutral nonamide amine nitrogen with three carbon substituents and no H.')}
QUERIES = {name: Chem.MolFromSmarts(pattern) for name, (pattern, _) in SMARTS_FEATURES.items()}
assert all((query is not None for query in QUERIES.values()))
ACID_QUERY = QUERIES['acid_present']
CARBONYL_QUERY = Chem.MolFromSmarts('[CX3]=[OX1]')
ALDEHYDE_QUERIES = [Chem.MolFromSmarts(pattern) for pattern in ('[CX3H1](=[OX1])[#6]', '[CX3H2]=[OX1]')]
ARYL_METHOXY_QUERY = Chem.MolFromSmarts('[c]-[OX2]-[CH3]')
GRAPH_FEATURES = {'acid_attached_aromatic': 'At least one carboxylic acid/carboxylate carbonyl C is directly bonded to an aromatic atom.', 'acid_attached_heteroaromatic_ring_system': 'An acid carbonyl C is directly bonded to an aromatic ring system containing a noncarbon atom; ring system connectivity follows aromatic bonds, including fused rings but not single-bond-linked rings.', 'acid_nonaromatic': 'At least one acid/carboxylate carbonyl C has no directly bonded aromatic atom; aromatic rings separated by a methylene do not count as direct attachment.', 'aldehyde': 'R-CHO or formaldehyde; carbonyl H plus carbon or H substituent, excluding formamides and formate esters.', 'carbonyl_adjacent_aromatic_N': 'A carbonyl C is bonded to aromatic carbon whose adjacent aromatic-bonded ring atom is nitrogen (2-pyridyl relationship).', 'ortho_ringN_to_leaving_group': 'An aromatic carbon bearing any F, Cl, Br or I has an adjacent aromatic-bonded ring nitrogen. This topological feature can be triggered by a spectator halogen, including F when a different aryl halogen reacts; it does not identify the proven reactive site. The feature ID is retained for compatibility.', 'para_methoxy_benzylamine': 'A neutral Ar-CH2-NH2 group and an Ar-O-CH3 group occupy opposite positions of the same aromatic six-carbon ring; other ring substituents are permitted.'}
DESCRIPTOR_COLUMNS = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'aromatic_N_count']
BINARY_COLUMNS = list(SMARTS_FEATURES) + list(GRAPH_FEATURES)

def aromatic_component(molecule, start: int) -> set[int]:
    seen, pending = ({start}, [start])
    while pending:
        atom = molecule.GetAtomWithIdx(pending.pop())
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if bond.GetIsAromatic() and other.GetIsAromatic() and (other.GetIdx() not in seen):
                seen.add(other.GetIdx())
                pending.append(other.GetIdx())
    return seen

def adjacent_aromatic_n(atom) -> bool:
    return any((bond.GetIsAromatic() and bond.GetOtherAtom(atom).GetAtomicNum() == 7 and bond.GetOtherAtom(atom).GetIsAromatic() for bond in atom.GetBonds()))

def para_methoxy_benzylamine(molecule) -> int:
    benzyl_positions = {match[2] for match in molecule.GetSubstructMatches(QUERIES['benzylic_primary_amine']) if molecule.GetAtomWithIdx(match[1]).GetTotalNumHs() == 2}
    methoxy_positions = {match[0] for match in molecule.GetSubstructMatches(ARYL_METHOXY_QUERY)}
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) != 6 or not all((molecule.GetAtomWithIdx(index).GetIsAromatic() and molecule.GetAtomWithIdx(index).GetAtomicNum() == 6 for index in ring)):
            continue
        for position, index in enumerate(ring):
            if index in benzyl_positions and ring[(position + 3) % 6] in methoxy_positions:
                return 1
    return 0

def molecule_features(smiles: str) -> dict:
    raw_smiles = '' if smiles is None or pd.isna(smiles) else str(smiles)
    result = {'smiles': raw_smiles, 'canonical_smiles': '', 'scaffold': '', 'scaffold_type': '', 'parsing_status': 'empty' if not raw_smiles.strip() else 'invalid', 'parsing_error': '', **{name: np.nan for name in BINARY_COLUMNS + DESCRIPTOR_COLUMNS}}
    if not raw_smiles.strip():
        result['parsing_error'] = 'Empty source SMILES'
        return result
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(raw_smiles)
        if molecule is None:
            result['parsing_error'] = 'RDKit could not parse and sanitize the recorded SMILES'
            return result
        result['canonical_smiles'] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        murcko = MurckoScaffold.GetScaffoldForMol(molecule)
        if murcko.GetNumAtoms():
            result['scaffold'] = Chem.MolToSmiles(murcko, canonical=True, isomericSmiles=True)
            result['scaffold_type'] = 'murcko'
        else:
            result['scaffold'] = result['canonical_smiles']
            result['scaffold_type'] = 'acyclic_identity'
        result.update({name: int(molecule.HasSubstructMatch(query)) for name, query in QUERIES.items()})
        result.update({name: 0 for name in GRAPH_FEATURES})
        for match in molecule.GetSubstructMatches(ACID_QUERY):
            acid_carbon = molecule.GetAtomWithIdx(match[0])
            aromatic_neighbors = [atom for atom in acid_carbon.GetNeighbors() if atom.GetIsAromatic()]
            if aromatic_neighbors:
                result['acid_attached_aromatic'] = 1
                for neighbor in aromatic_neighbors:
                    component = aromatic_component(molecule, neighbor.GetIdx())
                    if any((molecule.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in component)):
                        result['acid_attached_heteroaromatic_ring_system'] = 1
            else:
                result['acid_nonaromatic'] = 1
        result['aldehyde'] = int(any((molecule.HasSubstructMatch(query) for query in ALDEHYDE_QUERIES)))
        for match in molecule.GetSubstructMatches(CARBONYL_QUERY):
            if any((atom.GetAtomicNum() == 6 and atom.GetIsAromatic() and adjacent_aromatic_n(atom) for atom in molecule.GetAtomWithIdx(match[0]).GetNeighbors())):
                result['carbonyl_adjacent_aromatic_N'] = 1
        for atom in molecule.GetAtoms():
            if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and adjacent_aromatic_n(atom):
                if any((neighbor.GetAtomicNum() in (9, 17, 35, 53) for neighbor in atom.GetNeighbors())):
                    result['ortho_ringN_to_leaving_group'] = 1
        result['para_methoxy_benzylamine'] = para_methoxy_benzylamine(molecule)
        result.update({'MolWt': float(Descriptors.MolWt(molecule)), 'LogP': float(Crippen.MolLogP(molecule)), 'TPSA': float(rdMolDescriptors.CalcTPSA(molecule)), 'HBD': int(Lipinski.NumHDonors(molecule)), 'HBA': int(Lipinski.NumHAcceptors(molecule)), 'aromatic_N_count': sum((atom.GetIsAromatic() and atom.GetAtomicNum() == 7 for atom in molecule.GetAtoms()))})
        result['parsing_status'] = 'valid'
    except Exception as error:
        result['parsing_error'] = f'{type(error).__name__}: {error}'
        result['parsing_status'] = 'invalid'
        for name in BINARY_COLUMNS + DESCRIPTOR_COLUMNS:
            result[name] = np.nan
    return result

def feature_dictionary() -> pd.DataFrame:
    rows = [{'feature': name, 'feature_type': 'binary', 'description': description, 'smarts_or_graph_rule': pattern, 'units': '0/1'} for name, (pattern, description) in SMARTS_FEATURES.items()]
    rows.extend(({'feature': name, 'feature_type': 'binary', 'description': description, 'smarts_or_graph_rule': 'RDKit molecular graph: ' + description, 'units': '0/1'} for name, description in GRAPH_FEATURES.items()))
    descriptor_definitions = {'MolWt': ('RDKit average molecular weight of the complete recorded molecular graph.', 'Descriptors.MolWt', 'g/mol'), 'LogP': ('RDKit Crippen calculated logP.', 'Crippen.MolLogP', 'dimensionless'), 'TPSA': ('RDKit topological polar surface area.', 'rdMolDescriptors.CalcTPSA', 'angstrom^2'), 'HBD': ('RDKit Lipinski hydrogen-bond donor count.', 'Lipinski.NumHDonors', 'count'), 'HBA': ('RDKit Lipinski hydrogen-bond acceptor count.', 'Lipinski.NumHAcceptors', 'count'), 'aromatic_N_count': ('Number of nitrogen atoms marked aromatic by RDKit, including [nH].', 'atom.GetIsAromatic() and atom.GetAtomicNum()==7', 'count')}
    rows.extend(({'feature': name, 'feature_type': 'descriptor', 'description': description, 'smarts_or_graph_rule': rule, 'units': units} for name, (description, rule, units) in descriptor_definitions.items()))
    metadata = {'smiles': 'Exact raw substrate SMILES; raw strings are the table key.', 'canonical_smiles': 'RDKit canonical isomeric SMILES of the full recorded graph; no fragment stripping or neutralization.', 'scaffold': 'Canonical Murcko scaffold if any ring scaffold exists; otherwise the complete canonical molecular identity, so unrelated acyclic molecules do not collapse to an empty scaffold.', 'scaffold_type': 'murcko or acyclic_identity for valid structures; empty otherwise.', 'parsing_status': 'valid, invalid or empty. Invalid and empty rows retain the raw string and missing feature values.', 'parsing_error': 'Reason an empty or invalid SMILES has no feature values.'}
    rows.extend(({'feature': name, 'feature_type': 'metadata', 'description': description, 'smarts_or_graph_rule': '', 'units': ''} for name, description in metadata.items()))
    dictionary = pd.DataFrame(rows)
    dictionary['rdkit_version'] = rdBase.rdkitVersion
    return dictionary

def build_features(raw_path: str | Path, outdir: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, usecols=['sub_1_smiles', 'sub_2_smiles'], dtype=str, keep_default_na=False, na_filter=False)
    smiles_values = sorted(set(raw['sub_1_smiles']) | set(raw['sub_2_smiles']))
    print(f'Generating graph features for {len(smiles_values):,} unique raw substrate strings', flush=True)
    features = pd.DataFrame((molecule_features(smiles) for smiles in smiles_values))
    assert len(features) == len(smiles_values) and features['smiles'].is_unique
    valid = features['parsing_status'] == 'valid'
    assert features.loc[valid, BINARY_COLUMNS].isin([0, 1]).all().all()
    assert features.loc[~valid, BINARY_COLUMNS + DESCRIPTOR_COLUMNS].isna().all().all()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features.to_csv(outdir / 'molecular_features.csv', index=False, encoding='utf-8')
    feature_dictionary().to_csv(outdir / 'feature_dictionary.csv', index=False, encoding='utf-8')
    readback = pd.read_csv(outdir / 'molecular_features.csv', keep_default_na=False, na_filter=False)
    assert len(readback) == len(smiles_values) and readback['smiles'].tolist() == smiles_values
    assert readback['parsing_status'].tolist() == features['parsing_status'].tolist()
    print('Feature table verified: ' + str(features['parsing_status'].value_counts().to_dict()), flush=True)
    return features

def self_test() -> None:
    cases = [('Nc1ccccc1', {'primary_arylamine': 1, 'primary_alkylamine': 0, 'aniline': 1, 'benzylic_primary_amine': 0}), ('NCc1ccccc1', {'primary_arylamine': 0, 'primary_alkylamine': 1, 'aniline': 0, 'benzylic_primary_amine': 1}), ('COc1ccc(CN)cc1', {'para_methoxy_benzylamine': 1, 'methoxy': 1, 'benzylic_primary_amine': 1}), ('COc1cccc(CN)c1', {'para_methoxy_benzylamine': 0, 'benzylic_primary_amine': 1}), ('O=C(O)c1cccnc1', {'acid_present': 1, 'acid_attached_aromatic': 1, 'acid_attached_heteroaromatic_ring_system': 1, 'acid_nonaromatic': 0}), ('O=C(O)c1ccccc1', {'acid_present': 1, 'acid_attached_aromatic': 1, 'acid_attached_heteroaromatic_ring_system': 0, 'acid_nonaromatic': 0}), ('O=C(O)Cc1ccccc1', {'acid_present': 1, 'acid_attached_aromatic': 0, 'acid_attached_heteroaromatic_ring_system': 0, 'acid_nonaromatic': 1}), ('O=C(O)c1ccc2[nH]ccc2c1', {'acid_attached_heteroaromatic_ring_system': 1, 'NH_heteroaromatic': 1}), ('O=C(O)c1ccc(-c2ccccn2)cc1', {'acid_attached_heteroaromatic_ring_system': 0}), ('O=Cc1ccccn1', {'aldehyde': 1, 'ketone': 0, 'carbonyl_adjacent_aromatic_N': 1}), ('O=Cc1cccnc1', {'aldehyde': 1, 'carbonyl_adjacent_aromatic_N': 0}), ('Clc1ccccn1', {'aryl_halide': 1, 'aryl_Cl': 1, 'aryl_F': 0, 'ortho_ringN_to_leaving_group': 1}), ('Clc1cccnc1', {'ortho_ringN_to_leaving_group': 0}), ('CC(=O)N', {'amide': 1, 'primary_alkylamine': 0, 'primary_arylamine': 0, 'ketone': 0}), ('CCNCC', {'secondary_amine': 1, 'tertiary_amine': 0}), ('CCN(CC)CC', {'secondary_amine': 0, 'tertiary_amine': 1}), ('Oc1ccccc1', {'phenol': 1, 'free_alcohol': 0}), ('OCC', {'phenol': 0, 'free_alcohol': 1}), ('COC(C)=O', {'methoxy': 0, 'acid_present': 0})]
    for smiles, expected in cases:
        observed = molecule_features(smiles)
        assert observed['parsing_status'] == 'valid', (smiles, observed)
        for feature, value in expected.items():
            assert observed[feature] == value, (smiles, feature, observed[feature], value)
    invalid = molecule_features('this is not a SMILES')
    assert invalid['parsing_status'] == 'invalid' and pd.isna(invalid['primary_arylamine'])
    assert molecule_features('')['parsing_status'] == 'empty'
    acyclic = molecule_features('CCN')
    assert acyclic['scaffold_type'] == 'acyclic_identity'
    assert acyclic['scaffold'] == acyclic['canonical_smiles']

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-path', type=Path, default=ANALYSIS_DIR / 'data' / 'raw_reactions.csv.gz')
    parser.add_argument('--outdir', type=Path, default=ANALYSIS_DIR / 'data')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    print('Chemical graph feature tests passed.', flush=True)
    if not args.self_test:
        build_features(args.raw_path, args.outdir)
if __name__ == '__main__':
    main()
