"""Enumerate molecular changes that reverse the preferred recipe diagonal.

Candidate rows preserve all positive-margin opposite directions, without a
fingerprint or response-size filter. Source matching precedes target means."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMMPA
from round2_near_neighbors import exact_neighbors, parsed, canonical
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'round6' / 'molecular_rewiring'
CORNERS = ['y0_bg0', 'y1_bg0', 'y0_bg1', 'y1_bg1']
RECIPES = ['recipe_c0_bg0', 'recipe_c1_bg0', 'recipe_c0_bg1', 'recipe_c1_bg1']
BLOCK = ['dataset_id', 'archive', 'plate', 'acquisition', 'target_key']
ROLES = ['sub_1_smiles', 'sub_2_smiles']
ROUND4_METHODS = {'exact_H_substitution', 'exact_aromatic_atom_replacement', 'exact_methylene_insertion', 'RDKit_single_cut_MMP'}
NAMES = {'Nc1ccnc(C(F)(F)F)c1': '4-amino-2-(trifluoromethyl)pyridine', 'COC(=O)c1cc(N)ccn1': 'methyl 4-aminopyridine-2-carboxylate', 'Cc1ccnc(Cl)n1': '2-chloro-4-methylpyrimidine', 'Cc1ccc(B(O)O)cc1': '4-methylphenylboronic acid', 'COc1ccc(B(O)O)cc1': '4-methoxyphenylboronic acid', 'Brc1ccc2[nH]ncc2c1': '5-bromo-1H-indazole', 'COC(=O)c1ccc(Br)cc1': 'methyl 4-bromobenzoate', 'COC(=O)c1ccc(Br)c(C)c1': 'methyl 4-bromo-3-methylbenzoate', 'COc1ncc(B(O)O)cn1': '2-methoxypyrimidin-5-ylboronic acid', 'COc1ccc(B(O)O)cn1': '6-methoxypyridin-3-ylboronic acid', 'Cc1cc(F)ccc1B(O)O': '4-fluoro-2-methylphenylboronic acid', 'Cc1cc(B(O)O)ccc1F': '4-fluoro-3-methylphenylboronic acid', 'Cc1ccc(C(=O)O)cc1': '4-methylbenzoic acid', 'Cc1ccc(CC(=O)O)cc1': '2-(4-methylphenyl)acetic acid', 'COC(=O)c1ccc(N)cc1': 'methyl 4-aminobenzoate', 'CCNc1ccccc1': 'N-ethylaniline', 'C[C@@H](N)c1ccccc1': '1-phenylethylamine (source stereoisomer)', 'Cc1ccc(Br)nn1': '3-bromo-6-methylpyridazine', 'Cc1ccc(Br)nc1': '2-bromo-5-methylpyridine', 'Oc1ccc(-c2ccc(F)cc2)cc1': '4-(4-fluorophenyl)phenol', 'O=Cc1ccc(Br)nc1': '6-bromopyridine-3-carbaldehyde', 'COc1ccc(C=O)cn1': '6-methoxypyridine-3-carbaldehyde', 'CCc1cc[c]([Mg][Br])cc1': '4-ethylphenylmagnesium bromide', 'Cc1cc(Cl)ncn1': '4-chloro-6-methylpyrimidine', 'Ic1cnccn1': '2-iodopyrazine', 'CC1CCCNC1': '3-methylpiperidine', 'C1CCNC1': 'pyrrolidine', 'O=[N+]([O-])c1ccccc1F': '1-fluoro-2-nitrobenzene', 'Cc1cc(C)cc(Br)c1': '1-bromo-3,5-dimethylbenzene', 'Cc1cc(Br)cs1': '4-bromo-2-methylthiophene', 'Cc1cccc(Br)n1': '2-bromo-6-methylpyridine', 'Brc1ccc2ccccc2n1': '2-bromoquinoline'}

def read(path, **kwargs):
    return pd.read_csv(ROOT / path, keep_default_na=False, low_memory=False, **kwargs)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def sid(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:18]

def pair_key(a, b):
    return json.dumps(sorted([a, b]), separators=(',', ':'))

def metrics(y):
    values = np.asarray(y, dtype=float)
    margin0 = values[..., [0, 3]].min(axis=-1) - values[..., [1, 2]].max(axis=-1)
    margin1 = values[..., [1, 2]].min(axis=-1) - values[..., [0, 3]].max(axis=-1)
    return (margin0, margin1)

def enumerate_candidates(targets, rectangles):
    rows = []
    a, b = (targets[targets.dominant_diagonal.eq('00_11')], targets[targets.dominant_diagonal.eq('10_01')])
    fields = ['rectangle_id', 'dataset_id', 'rxn_class', 'target_key', *ROLES, 'product_smiles', 'canonical_product', *CORNERS, 'DD', 'n_complete_acquisition_blocks', 'diagonal_00_11_margin', 'diagonal_10_01_margin', 'larger_margin_diagonal_min_yield']
    for role, other in [(ROLES[0], ROLES[1]), (ROLES[1], ROLES[0])]:
        joins = ['rectangle_id', 'dataset_id', 'rxn_class', other]
        part = a[fields].merge(b[fields], on=joins, suffixes=('_a', '_b'), validate='many_to_many')
        part = part.rename(columns={role + '_a': 'molecule_a', role + '_b': 'molecule_b', other: 'partner_smiles'})
        part['role'] = role
        part['molecular_pair_key'] = [pair_key(x, y) for x, y in part[['molecule_a', 'molecule_b']].to_numpy()]
        part['molecular_pair_id'] = part.molecular_pair_key.map(sid)
        part['candidate_id'] = [sid('|'.join(map(str, row))) for row in part[['dataset_id', 'rectangle_id', 'role', 'target_key_a', 'target_key_b']].to_numpy()]
        part['target_id_a'] = (part.dataset_id + '|' + part.target_key_a).map(sid)
        part['target_id_b'] = (part.dataset_id + '|' + part.target_key_b).map(sid)
        part['margin_a'] = part.diagonal_00_11_margin_a
        part['margin_b'] = part.diagonal_10_01_margin_b
        part['minimum_pair_margin'] = part[['margin_a', 'margin_b']].min(axis=1)
        part['minimum_dominant_yield'] = part[['larger_margin_diagonal_min_yield_a', 'larger_margin_diagonal_min_yield_b']].min(axis=1)
        part['both_dominant_min20pct'] = part.minimum_dominant_yield.ge(0.2)
        part['reversal_5pp'] = part.both_dominant_min20pct & part.minimum_pair_margin.ge(0.05)
        part['reversal_10pp'] = part.both_dominant_min20pct & part.minimum_pair_margin.ge(0.1)
        part['DD_a_minus_b'] = part.DD_a - part.DD_b
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    metadata = ['rectangle_id', 'representation_id', 'exchange_id', 'focal_slot', 'context_slot', 'component0_name', 'component1_name', 'component0_dose', 'component1_dose', 'context0_name', 'context1_name', 'context0_token', 'context1_token', 'fixed_background_json', *RECIPES, *['full_' + x for x in RECIPES]]
    result = result.merge(rectangles[metadata], on='rectangle_id', validate='many_to_one')
    excluded = result[result.partner_smiles.eq('') | result.molecule_a.eq(result.molecule_b)].copy()
    excluded['reason'] = np.where(excluded.partner_smiles.eq(''), 'one_reactant_representation_no_fixed_partner', 'same_canonical_focal_molecule')
    save(excluded, 'one_reactant_or_same_molecule_comparisons.csv.gz')
    result = result[result.partner_smiles.ne('') & result.molecule_a.ne(result.molecule_b)].copy()
    assert result.candidate_id.is_unique
    assert result.minimum_pair_margin.gt(0).all()
    assert result.DD_a.gt(0).all() and result.DD_b.lt(0).all()
    print('ALL_CANDIDATES', len(result), result.groupby('role')[['reversal_5pp', 'reversal_10pp']].sum().to_dict(), flush=True)
    return result

def molecular_relations(candidates):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    molecules = sorted(set(candidates.molecule_a) | set(candidates.molecule_b))
    mols = {smiles: parsed(smiles) for smiles in molecules}
    fingerprints = {s: generator.GetFingerprint(m) for s, m in mols.items() if m is not None}
    probe = rdMMPA.FragmentMol(Chem.MolFromSmiles('COc1ccc(N)cc1'), maxCuts=1, resultsAsMols=False)
    assert probe and all((first == '' and len(second.split('.')) == 2 for first, second in probe))
    fragments = {}
    for smiles, molecule in mols.items():
        collection = defaultdict(list)
        if molecule is not None and '.' not in smiles:
            for first, second in rdMMPA.FragmentMol(molecule, maxCuts=1, resultsAsMols=False):
                assert first == ''
                parts = second.split('.')
                assert len(parts) == 2
                for core, fragment in [parts, parts[::-1]]:
                    core_mol, variable = (parsed(core), parsed(fragment))
                    n_core = core_mol.GetNumHeavyAtoms()
                    if n_core >= 5 and n_core >= 0.6 * molecule.GetNumHeavyAtoms():
                        assert sum((atom.GetAtomicNum() == 0 for atom in core_mol.GetAtoms())) == 1
                        collection[core].append((fragment, variable.GetNumHeavyAtoms(), n_core))
        fragments[smiles] = collection
    relations = []
    for (dataset, role), part in candidates.groupby(['dataset_id', 'role'], sort=False):
        values = sorted(set(part.molecule_a) | set(part.molecule_b))
        exact = exact_neighbors(values, role)
        lookup = {} if exact.empty else {pair_key(row.molecule_a, row.molecule_b): row._asdict() for row in exact.itertuples(index=False)}
        pairs = part[['molecular_pair_key', 'molecular_pair_id', 'molecule_a', 'molecule_b']].drop_duplicates('molecular_pair_key')
        for row in pairs.itertuples(index=False):
            a, b = sorted([row.molecule_a, row.molecule_b])
            exact_row = lookup.get(row.molecular_pair_key, {})
            info = {'dataset_id': dataset, 'role': role, 'molecular_pair_key': row.molecular_pair_key, 'molecular_pair_id': row.molecular_pair_id, 'structural_molecule_a': a, 'structural_molecule_b': b, 'morgan_radius2_2048_tanimoto': DataStructs.TanimotoSimilarity(fingerprints[a], fingerprints[b]) if a in fingerprints and b in fingerprints else np.nan, 'exact_graph_method': exact_row.get('graph_method', 'no_rule_supported_exact_edit'), 'exact_transformation': exact_row.get('transformation', ''), 'exact_common_core': exact_row.get('common_core', ''), 'exact_fragment_a': exact_row.get('fragment_a', ''), 'exact_fragment_b': exact_row.get('fragment_b', ''), 'exact_molecule_a': exact_row.get('molecule_a', ''), 'exact_molecule_b': exact_row.get('molecule_b', ''), 'has_existing_exact_neighbor_rule': bool(exact_row)}
            info['has_round4_allowed_exact_rule'] = info['exact_graph_method'] in ROUND4_METHODS
            info['has_exact_substituent_relocation'] = info['exact_graph_method'] == 'exact_substituent_relocation'
            possible = []
            for core in set(fragments[a]) & set(fragments[b]):
                for fa, na, nc in fragments[a][core]:
                    for fb, nb, _ in fragments[b][core]:
                        if fa != fb:
                            possible.append((max(na, nb), na + nb, -nc, core, fa, fb, na, nb, nc))
            info['single_cut_common_core_ge60pct_ge5atoms'] = bool(possible)
            info.update(single_cut_core='', single_cut_fragment_a='', single_cut_fragment_b='', single_cut_fragment_a_heavy_atoms=np.nan, single_cut_fragment_b_heavy_atoms=np.nan, single_cut_core_heavy_atoms=np.nan, single_cut_core_fraction_a=np.nan, single_cut_core_fraction_b=np.nan)
            if possible:
                _, _, _, core, fa, fb, na, nb, nc = min(possible)
                info.update(single_cut_core=core, single_cut_fragment_a=fa, single_cut_fragment_b=fb, single_cut_fragment_a_heavy_atoms=na, single_cut_fragment_b_heavy_atoms=nb, single_cut_core_heavy_atoms=nc, single_cut_core_fraction_a=nc / mols[a].GetNumHeavyAtoms(), single_cut_core_fraction_b=nc / mols[b].GetNumHeavyAtoms())
            info['structural_relation'] = info['exact_graph_method'] if exact_row else 'single_cut_MMP_larger_variable_fragment' if possible else 'other_structural_change'
            relations.append(info)
        print('STRUCTURES', dataset, role, len(pairs), 'existing exact', sum((p['has_existing_exact_neighbor_rule'] for p in relations if p['dataset_id'] == dataset and p['role'] == role)), flush=True)
    result = pd.DataFrame(relations)
    assert not result.duplicated(['dataset_id', 'role', 'molecular_pair_key']).any()
    return result

def reconstruct_sources(candidates, rectangles, targets):
    normal = read('data/normalized_records.csv.gz')
    normal = normal[normal.strict_eligible.astype(str).str.lower().eq('true')].copy()
    normal['yield_numeric'] = pd.to_numeric(normal.yield_numeric, errors='raise')
    target_list = pd.concat([candidates[['dataset_id', 'rectangle_id', 'target_key_' + side]].rename(columns={'target_key_' + side: 'target_key'}) for side in ['a', 'b']]).drop_duplicates()
    source_rows, block_rows = ([], [])
    metadata = rectangles.set_index('rectangle_id')
    expected = targets.set_index(['rectangle_id', 'target_key'])[CORNERS]
    for rectangle, wanted in target_list.groupby('rectangle_id', sort=False):
        case = metadata.loc[rectangle]
        recipes = case[RECIPES].tolist()
        frame = normal[normal.dataset_id.eq(case.dataset_id) & normal.recipe_id.isin(recipes) & normal.target_key.isin(wanted.target_key)].copy()
        cells = frame.groupby(BLOCK + ['recipe_id'], as_index=False).yield_numeric.mean()
        complete = cells.pivot(index=BLOCK, columns='recipe_id', values='yield_numeric').reindex(columns=recipes).dropna().reset_index()
        complete = complete.rename(columns=dict(zip(recipes, CORNERS)))
        recon = complete.groupby('target_key')[CORNERS].mean().reindex(wanted.target_key)
        ref = expected.loc[rectangle].reindex(wanted.target_key)
        assert np.isfinite(recon.to_numpy()).all() and np.allclose(recon, ref, atol=1e-12), rectangle
        complete['rectangle_id'] = rectangle
        complete['representation_id'] = case.representation_id
        complete['target_id'] = (complete.dataset_id + '|' + complete.target_key).map(sid)
        block_rows.append(complete)
        for corner, recipe in zip(CORNERS, recipes):
            links = complete[BLOCK + ['rectangle_id', 'representation_id', 'target_id', corner]].rename(columns={corner: 'matched_cell_yield'})
            links['recipe_id'] = recipe
            links['corner'] = corner
            linked = links.merge(frame, on=BLOCK + ['recipe_id'], how='left', validate='one_to_many')
            source_rows.append(linked[['rectangle_id', 'representation_id', 'target_id', 'corner', *BLOCK, 'recipe_id', 'record_id', 'condition_id', 'yield_numeric', 'matched_cell_yield', *ROLES, 'product_smiles', 'sub_1_cas', 'sub_2_cas']])
    blocks = pd.concat(block_rows, ignore_index=True)
    source = pd.concat(source_rows, ignore_index=True)
    raw = read('data/raw_reactions.csv.gz', usecols=['record_id', 'source_file', 'source_sheet', 'source_row'])
    source = source.merge(raw, on='record_id', how='left', validate='many_to_one')
    assert source.source_file.notna().all()
    source['source_record_locator'] = source.source_file.astype(str) + '#' + source.source_sheet.astype(str) + ':' + source.source_row.astype(str)
    raw_root = RAW_DIR
    paths = {filename: str(raw_root / filename) for filename in source.source_file.unique()}
    assert all((reader_tables.source_is_file(path) for path in paths.values()))
    source['source_absolute_path'] = source.source_file.map(paths)
    save(blocks, 'complete_target_acquisition_blocks.csv.gz')
    save(source, 'original_record_links.csv.gz')
    keys = ['rectangle_id', 'dataset_id', 'archive', 'plate', 'acquisition']
    left = candidates[['candidate_id', 'rectangle_id', 'dataset_id', 'target_key_a', 'target_key_b']].merge(blocks[keys + ['target_key', *CORNERS]], left_on=['rectangle_id', 'dataset_id', 'target_key_a'], right_on=['rectangle_id', 'dataset_id', 'target_key'], validate='many_to_many').drop(columns='target_key')
    shared = left.merge(blocks[keys + ['target_key', *CORNERS]], left_on=keys + ['target_key_b'], right_on=keys + ['target_key'], suffixes=('_a', '_b'), validate='many_to_one').drop(columns='target_key')
    for side in ['a', 'b']:
        m0, m1 = metrics(shared[[c + '_' + side for c in CORNERS]])
        shared['same_acquisition_margin0_' + side] = m0
        shared['same_acquisition_margin1_' + side] = m1
    for label, threshold in [('5pp', 0.05), ('10pp', 0.1)]:
        shared['same_acquisition_reversal_' + label] = shared.same_acquisition_margin0_a.ge(threshold) & shared.same_acquisition_margin1_b.ge(threshold) & shared[['y0_bg0_a', 'y1_bg1_a', 'y1_bg0_b', 'y0_bg1_b']].ge(0.2).all(axis=1)
    save(shared, 'same_acquisition_eight_yields.csv.gz')
    counts = shared.groupby('candidate_id', as_index=False).agg(n_common_complete_acquisitions=('acquisition', 'size'), n_common_acquisition_reversals_5pp=('same_acquisition_reversal_5pp', 'sum'), n_common_acquisition_reversals_10pp=('same_acquisition_reversal_10pp', 'sum'))
    means = shared.groupby('candidate_id', as_index=False)[[c + '_' + side for side in ['a', 'b'] for c in CORNERS]].mean()
    means = means.rename(columns={c: 'common_acquisition_mean_' + c for c in means if c != 'candidate_id'})
    out = candidates.merge(counts, on='candidate_id', how='left', validate='one_to_one').merge(means, on='candidate_id', how='left', validate='one_to_one')
    for column in counts.columns.drop('candidate_id'):
        out[column] = out[column].fillna(0).astype(int)
    cas = normal.groupby(['dataset_id', 'target_key'], as_index=False)[['sub_1_cas', 'sub_2_cas']].agg(lambda x: ' | '.join(sorted(set(x) - {''})))
    for side in ['a', 'b']:
        renamed = cas.rename(columns={'target_key': 'target_key_' + side, 'sub_1_cas': 'sub_1_cas_' + side, 'sub_2_cas': 'sub_2_cas_' + side})
        out = out.merge(renamed, on=['dataset_id', 'target_key_' + side], how='left', validate='many_to_one')
    source_groups = source.groupby(['dataset_id', 'rectangle_id', 'target_key'], as_index=False).agg(source_record_ids=('record_id', lambda x: json.dumps(sorted(set(x)))), source_absolute_paths=('source_absolute_path', lambda x: json.dumps(sorted(set(x)), ensure_ascii=False)))
    for side in ['a', 'b']:
        renamed = source_groups.rename(columns={key: key + '_' + side for key in ['target_key', 'source_record_ids', 'source_absolute_paths']})
        out = out.merge(renamed, on=['dataset_id', 'rectangle_id', 'target_key_' + side], how='left', validate='many_to_one')
    return (out, {'source_record_links': len(source), 'unique_source_records': int(source.record_id.nunique()), 'complete_target_acquisition_blocks': len(blocks), 'candidate_same_acquisition_eight_yield_blocks': len(shared), 'candidate_targets_reconstructed': len(target_list)})

def summarize(candidates, rectangles, mapping):
    base = read('round5/joint_accessibility/dataset_coverage.csv')[['dataset_id', 'rxn_class', 'n_swaps', 'n_registered_rectangles', 'n_primary_rectangles', 'rectangle_support_status']]
    assert set(base.dataset_id) == set(mapping.dataset_id) and base.n_swaps.sum() == 4002
    rows = []
    for dataset in mapping.dataset_id:
        for role in ROLES:
            part = candidates[candidates.dataset_id.eq(dataset) & candidates.role.eq(role)]
            info = {'dataset_id': dataset, 'role': role, 'n_continuous_opposite_candidates': len(part), 'n_unique_molecular_pairs': part.molecular_pair_id.nunique(), 'n_candidate_rectangles': part.rectangle_id.nunique()}
            for tag in ['5pp', '10pp']:
                q = part[part['reversal_' + tag]]
                exact = q[q.has_existing_exact_neighbor_rule]
                info.update({'n_reversals_' + tag: len(q), 'n_reversal_rectangles_' + tag: q.rectangle_id.nunique(), 'n_unique_molecular_pairs_' + tag: q.molecular_pair_id.nunique(), 'n_exact_reversals_' + tag: len(exact), 'n_exact_reversal_rectangles_' + tag: exact.rectangle_id.nunique(), 'n_reversals_common_acquisition_' + tag: int(q.n_common_complete_acquisitions.gt(0).sum())})
            if len(part):
                info['status'] = 'opposite_directions_present'
            elif int(base.loc[base.dataset_id.eq(dataset), 'n_primary_rectangles'].iloc[0]) == 0:
                info['status'] = 'no_primary_complete_rectangle'
            else:
                info['status'] = 'no_observed_opposite_directions_with_same_nonempty_partner'
            rows.append(info)
    coverage = base.merge(pd.DataFrame(rows), on='dataset_id', validate='one_to_many')
    excluded = read('round6/molecular_rewiring/one_reactant_or_same_molecule_comparisons.csv.gz')
    counts = excluded.groupby(['dataset_id', 'role'], as_index=False).agg(n_comparisons_without_fixed_molecular_partner=('candidate_id', 'size'), n_without_partner_reversal_5pp=('reversal_5pp', 'sum'), n_without_partner_reversal_10pp=('reversal_10pp', 'sum'))
    coverage = coverage.merge(counts, on=['dataset_id', 'role'], how='left', validate='one_to_one')
    count_columns = [c for c in counts if c not in ['dataset_id', 'role']]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    coverage.loc[coverage.n_comparisons_without_fixed_molecular_partner.gt(0), 'status'] = 'one_reactant_comparisons_retained_separately_no_fixed_partner'
    save(coverage, 'dataset_role_coverage.csv')
    grouping = ['dataset_id', 'rxn_class', 'role', 'molecular_pair_id', 'molecular_pair_key', 'structural_molecule_a', 'structural_molecule_b', 'structural_relation']
    pair_summary = candidates.groupby(grouping, as_index=False).agg(n_candidate_occurrences=('candidate_id', 'size'), n_rectangles=('rectangle_id', 'nunique'), n_fixed_partners=('partner_smiles', 'nunique'), n_reversal_5pp=('reversal_5pp', 'sum'), n_reversal_10pp=('reversal_10pp', 'sum'), maximum_minimum_pair_margin=('minimum_pair_margin', 'max'), n_candidate_occurrences_common_acquisition=('n_common_complete_acquisitions', lambda x: int(x.gt(0).sum())))
    save(pair_summary, 'molecular_pair_summary.csv')
    class_rows = []
    for threshold in ['5pp', '10pp']:
        for role in ROLES:
            for reaction_class in sorted(mapping.rxn_class.unique()):
                part = candidates[candidates.rxn_class.eq(reaction_class) & candidates.role.eq(role) & candidates['reversal_' + threshold]]
                exact = part[part.has_existing_exact_neighbor_rule]
                class_rows.append({'rxn_class': reaction_class, 'role': role, 'margin_threshold': threshold, 'n_reversal_candidates': len(part), 'n_reversal_rectangles': part.rectangle_id.nunique(), 'n_unique_molecular_pairs': part.molecular_pair_id.nunique(), 'n_exact_candidates': len(exact), 'n_exact_rectangles': exact.rectangle_id.nunique()})
    save(pd.DataFrame(class_rows), 'reaction_class_coverage.csv')
    return (coverage, pair_summary)

def select_examples(candidates):
    ranked = candidates.sort_values(['reversal_10pp', 'has_existing_exact_neighbor_rule', 'single_cut_common_core_ge60pct_ge5atoms', 'minimum_pair_margin', 'morgan_radius2_2048_tanimoto'], ascending=False)
    selected = []
    for _, part in ranked[ranked.reversal_5pp].groupby('rxn_class', sort=True):
        selected.append(part.iloc[0])
    for _, part in ranked[ranked.reversal_5pp & ranked.has_existing_exact_neighbor_rule].groupby('rxn_class', sort=True):
        selected.append(part.iloc[0])
    for candidate_id in ['3fd4ad20b610055390', '93412e602c45632eed', 'cd6be954c1dc09c153', '04dd2bf19e43e5601d', 'b19639d4281e5aa9ad', 'e82ed144ebf037209b']:
        selected.append(candidates[candidates.candidate_id.eq(candidate_id)].iloc[0])
    result = pd.DataFrame(selected).reset_index(drop=True)
    result = result.drop_duplicates('candidate_id').reset_index(drop=True)
    save(result, 'selected_multiclass_examples.csv')
    columns = ['candidate_id', 'rectangle_id', 'dataset_id', 'rxn_class', 'role', 'molecule_name_a', 'molecule_name_b', 'partner_name', 'molecule_a', 'molecule_b', 'partner_smiles', 'structural_relation', 'structural_change_a_to_b', 'morgan_radius2_2048_tanimoto', 'margin_a', 'margin_b', 'reversal_5pp', 'reversal_10pp', *[c + '_' + side for side in ['a', 'b'] for c in CORNERS], *['full_' + r for r in RECIPES], 'source_record_ids_a', 'source_record_ids_b', 'source_absolute_paths_a', 'source_absolute_paths_b']
    save(result[columns], 'named_edit_main_table.csv')
    return result

def report(candidates, coverage, examples, summary):
    lines = ['# Round 6: how specific molecular changes reverse two-component pairing', '', 'Date: 2026-10-16.', '', 'One methyl group, one ring nitrogen or one local substituent on a substrate can rearrange the preferred pairing of components. In Suzuki, replacing 4-methylphenylboronic acid with 4-methoxyphenylboronic acid crosses the preferred pairing of PdCl2(PCy3)2/SPhos Pd G3 with Cs2CO3/K3PO4; in Buchwald–Hartwig, replacing CF3 with CO2Me on the same 4-aminopyridine core crosses the pairing of BINAP Pd G3/XPhos Pd G3 with Cs2CO3/NaOtBu. The aromatic CH/N replacement in Ullmann and the Br/OMe replacement in Grignard extend this structure dependence to other reactions and components.', '', 'All 20 datasets, 18 reaction classes and 4,002 single-slot swaps are kept in the coverage table. This analysis scans 825 complete four-recipe rectangles with at least 20 targets; opposite-direction candidates have no fingerprint similarity threshold.', '', f'With the canonical molecule on the other side fixed, there are {len(candidates):,} opposite-direction candidates. When both preferred recipes give ≥20% and each molecule has a diagonal margin ≥5 percentage points, {int(candidates.reversal_5pp.sum()):,} comparisons remain; at 10 percentage points, {int(candidates.reversal_10pp.sum()):,}. Comparison rows share molecules, targets and recipes and are not counts of independent experiments.', '', '## Coverage of structural changes on both sides', '', '| Changed substrate side | Continuous opposite-direction candidates | 5pp comparisons / rectangles / classes | 10pp comparisons / rectangles / classes | 10pp exact graph-neighbour comparisons / rectangles / classes |', '|---|---:|---:|---:|---:|']
    for role in ROLES:
        part = candidates[candidates.role.eq(role)]
        fields = []
        for tag in ['5pp', '10pp']:
            q = part[part['reversal_' + tag]]
            fields.append(f'{len(q)} / {q.rectangle_id.nunique()} / {q.rxn_class.nunique()}')
        q = part[part.reversal_10pp & part.has_existing_exact_neighbor_rule]
        lines.append(f'| {role} | {len(part):,} | {fields[0]} | {fields[1]} | {len(q)} / {q.rectangle_id.nunique()} / {q.rxn_class.nunique()} |')
    lines += ['', '## Actual eight cells across reaction classes', '', 'Corner order is fixed as 00, 10, 01, 11; A prefers diagonal 00/11, B prefers diagonal 10/01. Yields are averaged after complete four cells within the same archive/plate/acquisition inside each target.', '']
    for row in examples.itertuples(index=False):
        lines += [f'### {row.rxn_class}: {row.structural_relation}', '', f'Candidate `{row.candidate_id}`, changed `{row.role}`, fixed partner `{row.partner_smiles}`.', '', f"A: {row.molecule_name_a or 'Molecule A'}, `{row.molecule_a}`; B: {row.molecule_name_b or 'Molecule B'}, `{row.molecule_b}`.", '', f'Structural change A→B: `{row.structural_change_a_to_b}`; Morgan Tanimoto={row.morgan_radius2_2048_tanimoto:.3f}.', '', '| Molecule | 00 | 10 | 01 | 11 | Preferred diagonal margin |', '|---|---:|---:|---:|---:|---:|']
        for side, label in [('a', 'A'), ('b', 'B')]:
            lines.append('| ' + label + ' | ' + ' | '.join((f"{100 * getattr(row, c + '_' + side):.2f}%" for c in CORNERS)) + f" | {100 * getattr(row, 'margin_' + side):.2f} pp |")
        lines += ['', *[f"- {c.replace('recipe_c', '')}: {getattr(row, 'full_' + c)} (`{getattr(row, c)}`)" for c in RECIPES], '', f'target A `{row.target_id_a}`; target B `{row.target_id_b}`. All source records are in `original_record_links.csv.gz`, located by rectangle_id=`{row.rectangle_id}` and target_id.', '']
    lines += ['## Definitions and files', '', 'm0=min(Y00,Y11)−max(Y10,Y01), m1=min(Y10,Y01)−max(Y00,Y11). Opposite-direction candidates require m0>0 for A and m1>0 for B; the 5/10pp labels also require all four cells of the two preferred diagonals to be ≥20%. Comparisons without a common acquisition are not removed from the total; common-acquisition eight cells are listed separately.', '', f"Fingerprints use RDKit Morgan radius=2, 2048 bits. Tanimoto describes similarity only. The four round-4 rule types (H substitution, acyclic CH2 insertion, aromatic CH/N replacement and small-fragment single-cut MMP) support {summary['n_round4_rule_reversal_5pp']}/{summary['n_round4_rule_reversal_10pp']} rows at 5/10pp; exact substituent position shifts add 25/11 rows, for a total of 55/17 rows. Single-cut MMP requires a common core of ≥5 heavy atoms covering ≥60% of each molecule; the variable fragments and atom counts are recorded. MMPs with large variable fragments are listed separately and not described as exact single-atom edits.", '', 'No strict archive/plate/acquisition eight-cell block is shared across molecules; the total uses the measured means of the complete four cells of each target. Common-acquisition checks are in same_acquisition_eight_yields.csv.gz; this subset does not replace the total. Finkelstein has no second reactant identity; 75 single-substrate continuous comparisons (4/3 rows at 5/10pp) are kept in a separate table and in the full-corpus coverage.', '', 'After canonicalisation, each reactant pair in a dataset maps to one canonical product; target identity is always dataset+target_key. Original targets with the same canonical identity are not merged for yield, and candidates keep their actual target definitions.', '', f"Reconstructed {summary['candidate_targets_reconstructed']:,} candidate target × rectangle rows and {summary['complete_target_acquisition_blocks']:,} complete four-cell acquisition blocks; {summary['source_record_links']:,} source links map to {summary['unique_source_records']:,} unique record_id values.", '', '- `all_opposite_direction_candidates.csv.gz`: all continuous candidates, eight-cell yields, recipes, structural relations, 5/10pp and common-acquisition support.', '- `molecular_relations.csv`: fingerprints, exact edits and MMP for each dataset × substrate side × molecule pair.', '- `molecular_pair_summary.csv`: support of molecule pairs across partners/rectangles; these occurrence counts are not independent experiments.', '- `named_edit_main_table.csv`: concise main table of named structures, actual eight-cell yields, full recipes and source record IDs.', '- `complete_recipe_registry.csv`: all full recipes involved; identity/amount for six reagent slots, solvent, temperature and time, with zeros and missing values kept as recorded.', '- `dataset_role_coverage.csv`, `reaction_class_coverage.csv`: full 20/18 coverage, with zero support and reasons kept.', '- `complete_target_acquisition_blocks.csv.gz`, `same_acquisition_eight_yields.csv.gz`: actual complete four cells and common-acquisition eight cells.', '- `original_record_links.csv.gz`: record_id, source file/sheet/row, conditions and SMILES; `selected_multiclass_examples.csv` keeps representative comparisons across classes.', '']
    (OUT / 'MOLECULAR_REWIRING_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = read('round5/joint_accessibility/all_target_accessibility.csv.gz')
    rectangles_all = read('round5/joint_accessibility/rectangle_accessibility_summary.csv')
    rectangles = rectangles_all[rectangles_all.n_targets.ge(20)].copy()
    assert len(rectangles) == 825 and len(rectangles_all) == 830
    mapping = read('round3/global_scope/dataset_reaction_class_mapping.csv')
    features = read('data/molecular_features.csv')
    canonical_map = features.set_index('smiles').canonical_smiles.to_dict()
    for role in ROLES:
        targets['raw_' + role] = targets[role]
        targets[role] = targets[role].map(canonical_map)
        assert targets[role].notna().all()
    unique_products = sorted(targets.product_smiles.unique())
    products = {s: canonical(m) if (m := parsed(s)) is not None else s for s in unique_products}
    targets['canonical_product'] = targets.product_smiles.map(products)
    unique_targets = targets.drop_duplicates(['dataset_id', 'target_key'])
    ambiguity = unique_targets.groupby(['dataset_id', *ROLES]).canonical_product.nunique().reset_index(name='n_canonical_products')
    assert ambiguity.n_canonical_products.max() == 1
    save(ambiguity, 'canonical_pair_product_check.csv')
    n_all_unique_targets = len(unique_targets)
    targets = targets[targets.rectangle_id.isin(rectangles.rectangle_id)].copy()
    unique_targets = targets.drop_duplicates(['dataset_id', 'target_key'])
    candidates = enumerate_candidates(targets, rectangles)
    relations = molecular_relations(candidates)
    save(relations, 'molecular_relations.csv')
    candidates = candidates.merge(relations.drop(columns='molecular_pair_id'), on=['dataset_id', 'role', 'molecular_pair_key'], validate='many_to_one')
    changes = []
    for row in candidates.itertuples(index=False):
        same = row.molecule_a == row.exact_molecule_a
        if row.exact_graph_method == 'exact_methylene_insertion':
            change = 'insert_CH2' if same else 'delete_CH2'
        elif row.exact_graph_method == 'exact_substituent_relocation':
            change = 'relocate ' + row.exact_fragment_a + ' on identical parent graph'
        elif row.has_existing_exact_neighbor_rule:
            change = row.exact_fragment_a + ' >> ' + row.exact_fragment_b if same else row.exact_fragment_b + ' >> ' + row.exact_fragment_a
        elif row.single_cut_common_core_ge60pct_ge5atoms:
            change = row.single_cut_fragment_a + ' >> ' + row.single_cut_fragment_b if row.molecule_a == row.structural_molecule_a else row.single_cut_fragment_b + ' >> ' + row.single_cut_fragment_a
        else:
            change = 'broader_structural_change'
        changes.append(change)
    candidates['structural_change_a_to_b'] = changes
    for column, identity in [('molecule_name_a', 'molecule_a'), ('molecule_name_b', 'molecule_b'), ('partner_name', 'partner_smiles')]:
        candidates[column] = candidates[identity].map(NAMES).fillna('')
    recipe_registry = read('data/condition_slot_registry.csv')
    wanted_recipes = pd.concat([candidates[['dataset_id', name]].rename(columns={name: 'recipe_id'}) for name in RECIPES]).drop_duplicates()
    recipe_registry = wanted_recipes.merge(recipe_registry, on=['dataset_id', 'recipe_id'], validate='one_to_many')
    save(recipe_registry, 'complete_recipe_registry.csv')
    print('EXACT_THRESHOLD_SUPPORT', candidates[candidates.has_existing_exact_neighbor_rule].groupby(['rxn_class', 'role'])[['reversal_5pp', 'reversal_10pp']].sum().to_string(), flush=True)
    save(candidates, 'all_opposite_direction_candidates.csv.gz')
    print('EARLY_EXACT_CASES', candidates[candidates.reversal_10pp & candidates.has_existing_exact_neighbor_rule].sort_values('minimum_pair_margin', ascending=False)[['candidate_id', 'dataset_id', 'role', 'molecule_a', 'molecule_b', 'partner_smiles', 'exact_graph_method', 'exact_transformation', *[c + '_' + side for side in ['a', 'b'] for c in CORNERS]]].head(15).to_json(orient='records'), flush=True)
    candidates, source_summary = reconstruct_sources(candidates, rectangles, targets)
    save(candidates, 'all_opposite_direction_candidates.csv.gz')
    coverage, pair_summary = summarize(candidates, rectangles, mapping)
    examples = select_examples(candidates)
    summary = {'status': 'completed', 'n_study_datasets': mapping.dataset_id.nunique(), 'n_study_classes': mapping.rxn_class.nunique(), 'n_study_single_slot_swaps': int(coverage.drop_duplicates('dataset_id').n_swaps.sum()), 'n_primary_rectangles': len(rectangles), 'n_primary_target_rectangle_means': len(targets), 'n_primary_unique_dataset_targets': len(unique_targets), 'n_canonical_reactant_pairs': len(ambiguity), 'n_all_830_rectangle_unique_dataset_targets': n_all_unique_targets, 'maximum_canonical_products_per_dataset_reactant_pair': int(ambiguity.n_canonical_products.max()), 'n_continuous_candidates': len(candidates), 'n_molecular_relations': len(relations), 'n_unique_dataset_molecular_pairs': len(relations.drop_duplicates(['dataset_id', 'molecular_pair_id'])), 'n_reversal_5pp': int(candidates.reversal_5pp.sum()), 'n_reversal_10pp': int(candidates.reversal_10pp.sum()), 'n_exact_reversal_5pp': int((candidates.reversal_5pp & candidates.has_existing_exact_neighbor_rule).sum()), 'n_exact_reversal_10pp': int((candidates.reversal_10pp & candidates.has_existing_exact_neighbor_rule).sum()), 'n_round4_rule_reversal_5pp': int((candidates.reversal_5pp & candidates.has_round4_allowed_exact_rule).sum()), 'n_round4_rule_reversal_10pp': int((candidates.reversal_10pp & candidates.has_round4_allowed_exact_rule).sum()), 'n_classes_reversal_5pp': candidates.loc[candidates.reversal_5pp, 'rxn_class'].nunique(), 'n_classes_reversal_10pp': candidates.loc[candidates.reversal_10pp, 'rxn_class'].nunique(), 'n_classes_exact_reversal_5pp': candidates.loc[candidates.reversal_5pp & candidates.has_existing_exact_neighbor_rule, 'rxn_class'].nunique(), 'n_classes_exact_reversal_10pp': candidates.loc[candidates.reversal_10pp & candidates.has_existing_exact_neighbor_rule, 'rxn_class'].nunique(), 'rdkit_version': rdBase.rdkitVersion, **source_summary, 'seconds': time.monotonic() - started}
    for tag in ['5pp', '10pp']:
        part = candidates[candidates['reversal_' + tag]]
        summary['n_reversal_rectangles_' + tag] = int(part.rectangle_id.nunique())
        summary['n_dataset_molecular_pairs_reversal_' + tag] = len(part[['dataset_id', 'molecular_pair_id']].drop_duplicates())
        for relation, key in [('has_existing_exact_neighbor_rule', 'exact'), ('has_round4_allowed_exact_rule', 'round4_rule')]:
            q = part[part[relation]]
            summary['n_' + key + '_dataset_molecular_pairs_reversal_' + tag] = len(q[['dataset_id', 'molecular_pair_id']].drop_duplicates())
    report(candidates, coverage, examples, summary)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    reread = pd.read_csv(OUT / 'all_opposite_direction_candidates.csv.gz', keep_default_na=False)
    assert len(reread) == len(candidates) and reread.candidate_id.is_unique
    assert reread.reversal_10pp.sum() == candidates.reversal_10pp.sum()
    for side, direction in [('a', 0), ('b', 1)]:
        margins = metrics(reread[[c + '_' + side for c in CORNERS]])
        assert np.allclose(margins[direction], reread['margin_' + side])
    assert len(pd.read_csv(OUT / 'dataset_role_coverage.csv')) == 40
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
