"""Exact graph near-neighbors and component-effect double differences.

Uses RDKit single-cut MMP fragmentation, exact graph deletion/insertion and
aromatic CH-to-N replacement. All yield effects begin with the existing
within-acquisition paired deltas; no unpaired raw-yield contrasts are used."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMMPA
ANALYSIS = Path(__file__).resolve().parents[1]
OUT = ANALYSIS / 'round2' / 'near_neighbors'
EMPTY = '<missing>'
ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
AMINE = Chem.MolFromSmarts('[N;X3;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-P(=O))]')

def canonical(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def parsed(smiles):
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(str(smiles))

def stable_id(value):
    return hashlib.sha256(value.encode()).hexdigest()[:14]

def no_dummy(core):
    molecule = parsed(core)
    dummy = [a.GetIdx() for a in molecule.GetAtoms() if not a.GetAtomicNum()]
    assert len(dummy) == 1
    edit = Chem.RWMol(molecule)
    edit.RemoveAtom(dummy[0])
    result = edit.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(result)
    return canonical(result)

def anchor_distance(core, role):
    mol = parsed(core)
    dummy = next((a for a in mol.GetAtoms() if not a.GetAtomicNum()))
    root = dummy.GetNeighbors()[0].GetIdx()
    query = AMINE if role == 'sub_1_smiles' else ACID
    anchors = [m[0] for m in mol.GetSubstructMatches(query)]
    return min((0 if root == a else len(Chem.GetShortestPath(mol, root, a)) - 1 for a in anchors), default=-1)

def exact_neighbors(smiles_values, role):
    molecules = {s: parsed(s) for s in sorted(smiles_values)}
    molecules = {s: m for s, m in molecules.items() if m is not None and '.' not in s}
    by_core = defaultdict(list)
    by_parent_frag = defaultdict(list)
    candidates = []

    def add(a, b, kind, transform, core, fragment_a, fragment_b):
        if a == b:
            return
        candidates.append({'molecule_a': a, 'molecule_b': b, 'graph_method': kind, 'transformation': transform, 'common_core': core, 'fragment_a': fragment_a, 'fragment_b': fragment_b})
    for smiles, molecule in molecules.items():
        n_heavy = molecule.GetNumHeavyAtoms()
        for _, fragments in rdMMPA.FragmentMol(molecule, maxCuts=1, resultsAsMols=False):
            parts = fragments.split('.')
            assert len(parts) == 2
            for core, fragment in [parts, parts[::-1]]:
                n_fragment = parsed(fragment).GetNumHeavyAtoms()
                n_core = parsed(core).GetNumHeavyAtoms()
                if n_fragment > 4 or n_core < 5 or n_core < 0.6 * n_heavy:
                    continue
                by_core[core].append((smiles, fragment))
                try:
                    parent = no_dummy(core)
                except Exception:
                    continue
                by_parent_frag[parent, fragment].append((smiles, core))
                if parent in molecules:
                    add(parent, smiles, 'exact_H_substitution', '[H] >> ' + fragment, core, '[H]', fragment)
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetIsAromatic() and (atom.GetTotalNumHs() == 1):
                edit = Chem.RWMol(molecule)
                replacement = edit.GetAtomWithIdx(atom.GetIdx())
                replacement.SetAtomicNum(7)
                replacement.SetNumExplicitHs(0)
                replacement.SetNoImplicit(True)
                test = edit.GetMol()
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(test)
                    other = canonical(test)
                except Exception:
                    continue
                if other in molecules:
                    add(smiles, other, 'exact_aromatic_atom_replacement', 'aromatic_CH >> aromatic_N', smiles, '[cH]', '[n]')
            if atom.GetAtomicNum() == 6 and (not atom.GetIsAromatic()) and (not atom.IsInRing()) and (atom.GetDegree() == 2) and (atom.GetTotalNumHs() == 2) and all((b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())):
                neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
                if molecule.GetBondBetweenAtoms(*neighbors) is not None:
                    continue
                edit = Chem.RWMol(molecule)
                edit.AddBond(*neighbors, Chem.BondType.SINGLE)
                edit.RemoveAtom(atom.GetIdx())
                test = edit.GetMol()
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(test)
                    other = canonical(test)
                except Exception:
                    continue
                if other in molecules:
                    add(other, smiles, 'exact_methylene_insertion', 'insert_CH2', other, 'bond', '[CH2]')
    for core, values in by_core.items():
        unique = sorted(set(values), key=lambda item: (item[1], item[0]))
        for i, (a, frag_a) in enumerate(unique):
            for b, frag_b in unique[i + 1:]:
                if frag_a != frag_b:
                    add(a, b, 'RDKit_single_cut_MMP', frag_a + ' >> ' + frag_b, core, frag_a, frag_b)
    for (parent, frag), values in by_parent_frag.items():
        unique = sorted(set(values))
        for i, (a, core_a) in enumerate(unique):
            for b, core_b in unique[i + 1:]:
                if core_a == core_b or a == b:
                    continue
                aa, bb = (a, b)
                da, db = (anchor_distance(core_a, role), anchor_distance(core_b, role))
                if da > db:
                    aa, bb, da, db = (bb, aa, db, da)
                add(aa, bb, 'exact_substituent_relocation', f'move_{frag}:anchor_distance_{da}_to_{db}', parent, frag, frag)
    if not candidates:
        return pd.DataFrame()
    found = pd.DataFrame(candidates)
    priority = {'exact_H_substitution': 0, 'exact_methylene_insertion': 1, 'exact_aromatic_atom_replacement': 2, 'exact_substituent_relocation': 3, 'RDKit_single_cut_MMP': 4}
    found['_priority'] = found.graph_method.map(priority)
    found['_key'] = found.apply(lambda row: '|'.join(sorted([row.molecule_a, row.molecule_b])), axis=1)
    found = found.sort_values(['_priority', 'transformation']).drop_duplicates('_key')
    found['structural_pair_id'] = found._key.map(stable_id)
    return found.drop(columns=['_priority', '_key']).reset_index(drop=True)

def assess_join(joined):
    dd = joined.delta_b - joined.delta_a
    return {'n_shared_partners': len(joined), 'mean_dd': dd.mean(), 'median_dd': dd.median(), 'mean_delta_a': joined.delta_a.mean(), 'mean_delta_b': joined.delta_b.mean(), 'fraction_dd_positive': (dd > 0).mean(), 'fraction_dd_negative': (dd < 0).mean(), 'n_preference_reversals_5pp': int(((joined.delta_a * joined.delta_b < 0) & (joined.delta_a.abs() > 0.05) & (joined.delta_b.abs() > 0.05)).sum()), 'mean_y0_a': joined.y0_a.mean(), 'mean_y1_a': joined.y1_a.mean(), 'mean_y0_b': joined.y0_b.mean(), 'mean_y1_b': joined.y1_b.mean(), 'n_both_high_max20': int(((joined[['y0_a', 'y1_a']].max(axis=1) >= 0.2) & (joined[['y0_b', 'y1_b']].max(axis=1) >= 0.2)).sum()), 'n_both_high_max50': int(((joined[['y0_a', 'y1_a']].max(axis=1) >= 0.5) & (joined[['y0_b', 'y1_b']].max(axis=1) >= 0.5)).sum()), 'n_a_exact_zero_both': int(((joined.y0_a == 0) & (joined.y1_a == 0)).sum()), 'n_b_exact_zero_both': int(((joined.y0_b == 0) & (joined.y1_b == 0)).sum())}

def mine(dataset, delta, swaps, mapping):
    rows = delta.loc[delta.dataset_id.eq(dataset)].copy()
    for column in ['sub_1_smiles', 'sub_2_smiles', 'product_smiles']:
        rows[column] = rows[column].map(mapping)
    assert rows[['sub_1_smiles', 'sub_2_smiles', 'product_smiles']].notna().all().all()
    product_counts = rows.groupby(['sub_1_smiles', 'sub_2_smiles']).product_smiles.nunique()
    ambiguous = product_counts[product_counts > 1].reset_index(name='n_products')
    ambiguous['dataset_id'] = dataset
    ambiguous_keys = set(map(tuple, ambiguous[['sub_1_smiles', 'sub_2_smiles']].to_numpy()))
    keep = [tuple(v) not in ambiguous_keys for v in rows[['sub_1_smiles', 'sub_2_smiles']].to_numpy()]
    rows = rows.loc[np.asarray(keep, dtype=bool) & rows.product_smiles.ne('')]
    summaries, supports, neighbors = ([], [], [])
    for role, counterpart in [('sub_1_smiles', 'sub_2_smiles'), ('sub_2_smiles', 'sub_1_smiles')]:
        candidates = exact_neighbors(set(rows[role]), role)
        if candidates.empty:
            continue
        candidates['dataset_id'] = dataset
        candidates['role'] = role
        neighbors.append(candidates)
        print(dataset, role, 'exact neighbor pairs:', len(candidates), flush=True)
        keys = ['pair_id', role, counterpart, 'product_smiles']
        target = rows.groupby(keys, as_index=False).agg(y0=('y0', 'mean'), y1=('y1', 'mean'), delta=('delta', 'mean'), n_acquisitions=('acquisition', 'size'), archives=('archive', lambda s: '|'.join(sorted(set(s)))), check_true=('both_source_check_true', 'all'))
        archive_target = rows.groupby(keys + ['archive'], as_index=False).agg(y0=('y0', 'mean'), y1=('y1', 'mean'), delta=('delta', 'mean'), n_acquisitions=('acquisition', 'size'), check_true=('both_source_check_true', 'all'))
        target_groups = dict(tuple(target.groupby(role)))
        archive_groups = dict(tuple(archive_target.groupby(role)))
        for n, pair in enumerate(candidates.to_dict('records')):
            a, b = (pair['molecule_a'], pair['molecule_b'])
            joined = target_groups[a].merge(target_groups[b], on=['pair_id', counterpart], suffixes=('_a', '_b'), validate='one_to_one')
            if joined.empty:
                continue
            joined['partner_smiles'] = joined[counterpart]
            for key in ['molecule_a', 'molecule_b', 'structural_pair_id', 'transformation', 'graph_method', 'role']:
                joined[key] = pair[key]
            joined['dataset_id'] = dataset
            joined['dd'] = joined.delta_b - joined.delta_a
            same_archive = archive_groups[a].merge(archive_groups[b], on=['pair_id', counterpart, 'archive'], suffixes=('_a', '_b'), validate='one_to_one')
            same_archive['dd'] = same_archive.delta_b - same_archive.delta_a
            for swap_id, part in joined.groupby('pair_id'):
                result = {**pair, 'pair_id': swap_id, **assess_join(part)}
                archive_part = same_archive.loc[same_archive.pair_id.eq(swap_id)]
                if len(archive_part):
                    by_partner = archive_part.groupby(counterpart, as_index=False).mean(numeric_only=True)
                    result.update({'n_shared_same_archive_partners': len(by_partner), 'same_archive_mean_dd': by_partner.dd.mean(), 'same_archive_fraction_agree': float((by_partner.dd * result['mean_dd'] > 0).mean()), 'shared_archives': '|'.join(sorted(set(archive_part.archive)))})
                else:
                    result.update({'n_shared_same_archive_partners': 0, 'same_archive_mean_dd': np.nan, 'same_archive_fraction_agree': np.nan, 'shared_archives': ''})
                checked = part.loc[part.check_true_a & part.check_true_b]
                result['n_source_check_true_partners'] = len(checked)
                result['source_check_true_mean_dd'] = checked.dd.mean()
                result['candidate_priority'] = len(part) >= 5 and abs(result['mean_dd']) >= 0.1 and (result['n_both_high_max20'] >= 3) and (max(result['fraction_dd_positive'], result['fraction_dd_negative']) >= 0.75)
                summaries.append(result)
                if result['candidate_priority']:
                    supports.append(part)
            if (n + 1) % 50 == 0:
                print(dataset, role, n + 1, 'pairs searched', flush=True)
    return (pd.DataFrame(summaries), pd.concat(supports, ignore_index=True) if supports else pd.DataFrame(), pd.concat(neighbors, ignore_index=True) if neighbors else pd.DataFrame(), ambiguous)

def source_trace(supports, normalized, swaps):
    wanted = []
    for suffix in ['a', 'b']:
        temp = supports[['dataset_id', 'pair_id', 'structural_pair_id', 'role', 'molecule_' + suffix, 'partner_smiles', 'product_smiles_' + suffix]].copy()
        temp['side'] = suffix
        temp['sub_1_smiles'] = np.where(temp.role.eq('sub_1_smiles'), temp['molecule_' + suffix], temp.partner_smiles)
        temp['sub_2_smiles'] = np.where(temp.role.eq('sub_2_smiles'), temp['molecule_' + suffix], temp.partner_smiles)
        temp['product_smiles'] = temp['product_smiles_' + suffix]
        wanted.append(temp[['dataset_id', 'pair_id', 'structural_pair_id', 'side', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']])
    targets = pd.concat(wanted).drop_duplicates().merge(swaps[['dataset_id', 'pair_id', 'recipe0', 'recipe1']], on=['dataset_id', 'pair_id'])
    traces = []
    for index in [0, 1]:
        targets['recipe_id'] = targets[f'recipe{index}']
        traced = targets.merge(normalized, on=['dataset_id', 'recipe_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles'], how='inner')
        traced['condition_side'] = index
        traces.append(traced)
    traced = pd.concat(traces, ignore_index=True)
    keys = ['dataset_id', 'pair_id', 'structural_pair_id', 'side', 'archive', 'plate', 'acquisition', 'target_key']
    paired = traced.groupby(keys).condition_side.transform('nunique').eq(2)
    return traced.loc[paired]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Acylation'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    swaps = pd.read_csv(ANALYSIS / 'results/matched_swap_summary.csv', keep_default_na=False)
    delta = pd.read_csv(ANALYSIS / 'results/delta_within_acquisition.csv.gz', keep_default_na=False)
    normalized = pd.read_csv(ANALYSIS / 'data/normalized_records.csv.gz', keep_default_na=False)
    features = pd.read_csv(ANALYSIS / 'data/molecular_features.csv', keep_default_na=False)
    mapping = features.set_index('smiles').canonical_smiles.to_dict()
    products = set(normalized.product_smiles) - set(mapping)
    mapping.update({s: canonical(m) if (m := parsed(s)) is not None else '' for s in products})
    for col in ['sub_1_smiles', 'sub_2_smiles', 'product_smiles']:
        normalized['raw_' + col] = normalized[col]
        normalized[col] = normalized[col].map(mapping)
    normalized = normalized.loc[normalized.strict_eligible.eq(True)].copy()
    outputs = [mine(dataset, delta, swaps, mapping) for dataset in args.datasets]
    summary, supports, pairs, ambiguous = [pd.concat([r[i] for r in outputs], ignore_index=True) for i in range(4)]
    summary = summary.merge(swaps.drop(columns=['dataset_id']), on='pair_id', suffixes=('', '_swap'), validate='many_to_one')
    summary['abs_mean_dd'] = summary.mean_dd.abs()
    summary = summary.sort_values(['candidate_priority', 'n_shared_same_archive_partners', 'abs_mean_dd'], ascending=False)
    summary.to_csv(OUT / 'all_neighbor_swap_summary.csv', index=False, encoding='utf-8-sig')
    pairs.to_csv(OUT / 'exact_graph_neighbor_pairs.csv', index=False, encoding='utf-8-sig')
    ambiguous.to_csv(OUT / 'multiple_product_substrate_pairs_excluded.csv', index=False, encoding='utf-8-sig')
    if len(supports):
        supports.to_csv(OUT / 'priority_shared_partner_support.csv.gz', index=False, encoding='utf-8-sig')
        source_trace(supports, normalized, swaps).to_csv(OUT / 'priority_source_records.csv.gz', index=False, encoding='utf-8-sig')
    group = summary.loc[summary.n_shared_partners.ge(3)].groupby(['dataset_id', 'role', 'transformation', 'pair_id'], as_index=False).agg(n_structural_pairs=('structural_pair_id', 'nunique'), mean_structural_dd=('mean_dd', 'mean'), minimum_pair_dd=('mean_dd', 'min'), maximum_pair_dd=('mean_dd', 'max'), n_same_archive_structural_pairs=('n_shared_same_archive_partners', lambda v: int((v >= 3).sum())))
    group = group.merge(swaps[['pair_id', 'component0_name', 'component1_name', 'slot', 'condition0_ids', 'condition1_ids']], on='pair_id')
    group.to_csv(OUT / 'repeated_transformations.csv', index=False, encoding='utf-8-sig')
    counts = {'datasets': args.datasets, 'rdkit_version': rdBase.rdkitVersion, 'exact_graph_pairs': len(pairs), 'neighbor_swap_comparisons': len(summary), 'priority_candidates': int(summary.candidate_priority.sum()), 'priority_partner_rows': len(supports), 'ambiguous_substrate_pairs_excluded': len(ambiguous), 'scope': 'Posthoc discovery; target/product means of same-acquisition component deltas. Shared partners are not independent structural transformations.'}
    (OUT / 'run_summary.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding='utf-8')
    assert pd.read_csv(OUT / 'all_neighbor_swap_summary.csv').shape[0] == len(summary)
    assert pairs.structural_pair_id.notna().all()
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    columns = ['role', 'molecule_a', 'molecule_b', 'transformation', 'component0_name', 'component1_name', 'n_shared_partners', 'mean_dd', 'n_shared_same_archive_partners', 'same_archive_mean_dd', 'n_both_high_max50']
    print(summary.loc[summary.candidate_priority, columns].head(30).to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
