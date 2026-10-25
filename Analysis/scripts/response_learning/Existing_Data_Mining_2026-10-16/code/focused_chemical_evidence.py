"""Posthoc structure contrasts for selected exact recipe swaps; no inferential p values."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
TARGETS = ['acylation__C18__C40', 'acylation__C18__C24', 'suzuki__C23__C26', 'buchwald-hartwig__C26__C27', 'buchwald-hartwig__C26__C35', 'buchwald-hartwig__C80__C90']
FOCUSED_SMARTS = {'aryl_chloride': '[c]-Cl', 'aryl_bromide': '[c]-Br', 'aryl_iodide': '[c]-I', 'aryl_fluoride': '[c]-F', 'aryl_triflate': '[c]-O-S(=O)(=O)C(F)(F)F'}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FOCUSED_SMARTS.items()}
ACID_PATTERN = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1,O-]')

def focused_annotations(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    result = {f'chem_{k}': int(mol.HasSubstructMatch(q)) for k, q in PATTERNS.items()}
    result.update({f'chem_{k}': 0 for k in ('aromatic_acid', 'aliphatic_acid', 'acid_alpha_branched', 'acid_alpha_hetero_substituted', 'acid_alpha_aryl_substituted')})
    for match in mol.GetSubstructMatches(ACID_PATTERN):
        acid_c = mol.GetAtomWithIdx(match[0])
        adjacent_c = [n for n in acid_c.GetNeighbors() if n.GetAtomicNum() == 6]
        for alpha in adjacent_c:
            if alpha.GetIsAromatic():
                result['chem_aromatic_acid'] = 1
                continue
            result['chem_aliphatic_acid'] = 1
            other_neighbors = [n for n in alpha.GetNeighbors() if n.GetIdx() != acid_c.GetIdx()]
            result['chem_acid_alpha_branched'] |= int(sum((n.GetAtomicNum() == 6 for n in other_neighbors)) >= 2)
            result['chem_acid_alpha_hetero_substituted'] |= int(any((n.GetAtomicNum() in (7, 8, 16) for n in other_neighbors)))
            result['chem_acid_alpha_aryl_substituted'] |= int(any((n.GetAtomicNum() == 6 and n.GetIsAromatic() for n in other_neighbors)))
    return result

def recipe_differences(recipe0: dict, recipe1: dict) -> list[str]:
    ignored = {'condition_id'} | {f'reagent{i}_name' for i in range(1, 7)} | {'solvent_name'}
    return [k for k in recipe0 if k not in ignored and recipe0[k] != recipe1[k]]

def exact_family_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_id in TARGETS:
        target = registry.loc[registry.swap_id.eq(target_id)].iloc[0]
        if target.match_class != 'strict_identity_only':
            raise ValueError(f'Expected strict target {target_id}')
        for _, candidate in registry[registry.reaction_type.eq(target.reaction_type) & registry.match_class.eq('strict_identity_only')].iterrows():
            direct = candidate.component0_id == target.component0_id and candidate.component1_id == target.component1_id
            reverse = candidate.component0_id == target.component1_id and candidate.component1_id == target.component0_id
            if not (direct or reverse) or candidate.swap_slot != target.swap_slot:
                continue
            dose0, dose1 = (candidate.dose0, candidate.dose1) if direct else (candidate.dose1, candidate.dose0)
            if not np.isclose(dose0, target.dose0) or not np.isclose(dose1, target.dose1):
                continue
            recipe0, recipe1 = (json.loads(candidate.recipe0), json.loads(candidate.recipe1))
            differences = recipe_differences(recipe0, recipe1)
            if differences != [f'{target.swap_slot}_id']:
                raise ValueError((candidate.swap_id, differences))
            background = {k: v for k, v in recipe0.items() if k not in {'condition_id', f'{target.swap_slot}_id', f'{target.swap_slot}_name'}}
            rows.append({'target_swap_id': target_id, 'swap_id': candidate.swap_id, 'direction_multiplier': 1 if direct else -1, 'component0_name': target.component0_name, 'component1_name': target.component1_name, 'dose0': dose0, 'dose1': dose1, 'n_pairs_registry': candidate.n_pairs, 'strict_changed_field': differences[0], 'background_json': json.dumps(background, sort_keys=True), 'recipe0_json': candidate.recipe0, 'recipe1_json': candidate.recipe1})
    return pd.DataFrame(rows).drop_duplicates(['target_swap_id', 'swap_id'])

def chosen_flags(reaction_type: str) -> dict[int, list[str]]:
    amines = ['flag_primary_aniline', 'flag_primary_alkyl_amine', 'flag_secondary_aniline', 'flag_secondary_alkyl_amine', 'flag_pyridine_like', 'flag_ortho_substituted_aniline']
    if reaction_type == 'acylation':
        return {1: amines, 2: ['chem_aromatic_acid', 'chem_aliphatic_acid', 'chem_acid_alpha_branched', 'chem_acid_alpha_hetero_substituted', 'chem_acid_alpha_aryl_substituted', 'flag_pyridine_like', 'flag_azole']}
    electrophile = ['chem_aryl_chloride', 'chem_aryl_bromide', 'chem_aryl_iodide', 'chem_aryl_triflate', 'flag_pyridine_like', 'flag_azole', 'flag_pyrrole_like', 'flag_ortho_substituted_aryl_halide']
    if reaction_type == 'suzuki':
        return {1: electrophile, 2: ['flag_pyridine_like', 'flag_azole', 'flag_pyrrole_like', 'flag_phenol', 'chem_aryl_fluoride']}
    return {1: amines + ['flag_pyrrole_like'], 2: electrophile}

def feature_contrasts(d: pd.DataFrame, target_id: str, swap_id: str, scope: str, role: int, feature: str, comparison: str='flag_vs_complement') -> tuple[dict, list[dict]]:
    own, partner = (f'sub_{role}_smiles', f'sub_{3 - role}_smiles')
    flag = f's{role}_{feature}'
    summaries = []
    for level in (0, 1):
        subset = d[d[flag].eq(level)]
        molecule_means = subset.groupby(own).delta_pp.mean()
        summaries.append({f'n_pairs{level}': len(subset), f'n_molecules{level}': subset[own].nunique(), f'n_partners{level}': subset[partner].nunique(), f'pair_mean_delta{level}': subset.delta_pp.mean(), f'molecule_mean_delta{level}': molecule_means.mean(), f'median_delta{level}': subset.delta_pp.median(), f'n_positive20_{level}': int(subset.delta_pp.ge(20).sum()), f'n_negative20_{level}': int(subset.delta_pp.le(-20).sum()), f'n_either_zero{level}': int((subset.y0.eq(0) | subset.y1.eq(0)).sum())})
    blocks = []
    for other, block in d.groupby(partner):
        zero, one = (block[block[flag].eq(0)], block[block[flag].eq(1)])
        if zero[own].nunique() < 2 or one[own].nunique() < 2:
            continue
        contrast = one.groupby(own).delta_pp.mean().mean() - zero.groupby(own).delta_pp.mean().mean()
        blocks.append({'target_swap_id': target_id, 'swap_id': swap_id, 'scope': scope, 'feature_role': role, 'feature': feature, 'comparison': comparison, 'shared_partner_smiles': other, 'n_molecules0': zero[own].nunique(), 'n_molecules1': one[own].nunique(), 'mean_delta0_pp': zero.delta_pp.mean(), 'mean_delta1_pp': one.delta_pp.mean(), 'contrast_pp': contrast})
    contrasts = pd.Series([b['contrast_pp'] for b in blocks], dtype=float)
    output = {'target_swap_id': target_id, 'swap_id': swap_id, 'scope': scope, 'feature_role': role, 'feature': feature, 'comparison': comparison, **summaries[0], **summaries[1], 'pair_weighted_contrast_pp': summaries[1]['pair_mean_delta1'] - summaries[0]['pair_mean_delta0'], 'molecule_weighted_contrast_pp': summaries[1]['molecule_mean_delta1'] - summaries[0]['molecule_mean_delta0'], 'n_eligible_partner_blocks': len(blocks), 'partner_block_mean_contrast_pp': contrasts.mean(), 'partner_block_median_contrast_pp': contrasts.median(), 'partner_block_min_contrast_pp': contrasts.min(), 'partner_block_max_contrast_pp': contrasts.max(), 'partner_block_positive_fraction': contrasts.gt(0).mean() if len(contrasts) else np.nan}
    return (output, blocks)

def restricted_comparisons(d: pd.DataFrame, reaction_type: str) -> list[tuple[pd.DataFrame, int, str, str]]:
    result = []
    if reaction_type == 'acylation':
        selected = d[d.s1_flag_primary_aniline.ne(d.s1_flag_primary_alkyl_amine)]
        result.append((selected, 1, 'flag_primary_aniline', 'primary_aniline_vs_primary_alkyl'))
        result.append((d[d.s1_flag_primary_aniline.eq(1)], 1, 'flag_ortho_substituted_aniline', 'within_primary_anilines'))
    elif reaction_type == 'suzuki':
        for flag in ('flag_pyridine_like', 'flag_pyrrole_like', 'flag_azole'):
            result.append((d[d.s1_chem_aryl_bromide.eq(1)], 1, flag, 'within_aryl_bromides'))
        selected = d[d.s1_flag_pyridine_like.ne(d.s1_flag_pyrrole_like)]
        result.append((selected, 1, 'flag_pyridine_like', 'pyridine_only_vs_pyrrole_only'))
    return result

def annotation_self_tests() -> None:
    cases = {'O=C(O)c1ccccc1': {'chem_aromatic_acid': 1, 'chem_aliphatic_acid': 0}, 'O=C(O)Cc1ccccc1': {'chem_aromatic_acid': 0, 'chem_aliphatic_acid': 1, 'chem_acid_alpha_aryl_substituted': 1, 'chem_acid_alpha_branched': 0}, 'CC(C)C(=O)O': {'chem_acid_alpha_branched': 1}, 'NCC(=O)O': {'chem_acid_alpha_hetero_substituted': 1}, 'Brc1ccccc1': {'chem_aryl_bromide': 1, 'chem_aryl_chloride': 0}, 'CCBr': {'chem_aryl_bromide': 0}, 'O=S(=O)(Oc1ccccc1)C(F)(F)F': {'chem_aryl_triflate': 1}}
    for smiles, expected in cases.items():
        values = focused_annotations(smiles)
        for flag, value in expected.items():
            assert values[flag] == value, (smiles, flag, values[flag])
    print(f'Focused structural definition tests passed: {len(cases)} structures.')

def representative_rows(d: pd.DataFrame, target_id: str) -> list[dict]:
    output = []
    for sign, mask in (('positive20', d.delta_pp.ge(20)), ('negative20', d.delta_pp.le(-20))):
        selected = d[mask & d.y0.gt(0) & d.y1.gt(0)].copy()
        selected['abs_delta'] = selected.delta_pp.abs()
        used_amine, used_acid = (set(), set())
        for _, row in selected.sort_values(['abs_delta', 'substrate_pair_key'], ascending=[False, True]).iterrows():
            if row.sub_1_smiles in used_amine or row.sub_2_smiles in used_acid:
                continue
            used_amine.add(row.sub_1_smiles)
            used_acid.add(row.sub_2_smiles)
            fields = ['substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles', 'y0', 'y1', 'delta_pp', 'sub_1_FG0', 'sub_2_FG0', 'source_file0', 'source_sheet0', 'observation_id0', 'observation_id1', 'name00', 'name01', 'archive0', 'archive1', 'duplicate_cell0', 'duplicate_cell1']
            record = {'target_swap_id': target_id, 'effect_sign': sign, **{k: row[k] for k in fields}}
            record.update({f's{role}_positive_annotations': ';'.join((c.removeprefix(f's{role}_') for c in row.index if c.startswith((f's{role}_flag_', f's{role}_chem_')) and row[c] == 1)) for role in (1, 2)})
            output.append(record)
            if len(used_amine) == 4:
                break
    return output

def main() -> None:
    annotation_self_tests()
    root = Path(__file__).resolve().parents[1]
    results = root / 'results'
    registry = pd.read_csv(results / 'swap_registry.csv')
    families = exact_family_rows(registry)
    family_ids = set(families.swap_id)
    deltas = pd.read_parquet(results / 'swap_deltas.parquet', filters=[('swap_id', 'in', list(family_ids))])
    features = pd.read_parquet(root / 'processed' / 'molecular_features.parquet')
    extra = pd.DataFrame([{'smiles': s, **focused_annotations(s)} for s in features.smiles])
    features = features.merge(extra, on='smiles', validate='one_to_one')
    keep = ['smiles'] + [c for c in features if c.startswith(('flag_', 'chem_'))]
    for role in (1, 2):
        renamed = features[keep].rename(columns={'smiles': f'sub_{role}_smiles', **{c: f's{role}_{c}' for c in keep if c != 'smiles'}})
        deltas = deltas.merge(renamed, on=f'sub_{role}_smiles', how='left', validate='many_to_one')
    summaries, blocks, examples, context_summaries, overlaps = ([], [], [], [], [])
    for _, context in families.iterrows():
        d = deltas[deltas.swap_id.eq(context.swap_id)].copy()
        assert not d.substrate_pair_key.duplicated().any(), context.swap_id
        if context.direction_multiplier == -1:
            d['delta_pp'] *= -1
            d[['y0', 'y1']] = d[['y1', 'y0']].to_numpy()
        target = registry[registry.swap_id.eq(context.target_swap_id)].iloc[0]
        primary = deltas[deltas.swap_id.eq(context.target_swap_id)]
        overlap_keys = set(primary.substrate_pair_key) & set(d.substrate_pair_key)
        context_summaries.append({**context.to_dict(), 'n_pairs': len(d), 'n_sub1': d.sub_1_smiles.nunique(), 'n_sub2': d.sub_2_smiles.nunique(), 'mean_delta_pp': d.delta_pp.mean(), 'median_delta_pp': d.delta_pp.median(), 'n_positive20': int(d.delta_pp.ge(20).sum()), 'n_negative20': int(d.delta_pp.le(-20).sum()), 'n_both_positive': int((d.y0.gt(0) & d.y1.gt(0)).sum()), 'n_overlap_target_pairs': len(overlap_keys)})
        if context.swap_id != context.target_swap_id and overlap_keys:
            matched = primary[['substrate_pair_key', 'delta_pp']].merge(d[['substrate_pair_key', 'delta_pp']], on='substrate_pair_key', suffixes=('_target', '_context'))
            active = matched[matched.delta_pp_target.abs().ge(20)]
            overlaps.append({'target_swap_id': context.target_swap_id, 'swap_id': context.swap_id, 'n_common_pairs': len(matched), 'pearson_delta_correlation': matched.delta_pp_target.corr(matched.delta_pp_context), 'n_target_abs20': len(active), 'n_same_sign_of_target_abs20': int((np.sign(active.delta_pp_target) == np.sign(active.delta_pp_context)).sum()), 'n_context_abs20_same_sign': int(((np.sign(active.delta_pp_target) == np.sign(active.delta_pp_context)) & active.delta_pp_context.abs().ge(20)).sum())})
        scope_sets = [('all', d), ('both_yields_positive', d[d.y0.gt(0) & d.y1.gt(0)]), ('unique_cells', d[~d.duplicate_cell0 & ~d.duplicate_cell1])]
        family_members = families[families.target_swap_id.eq(context.target_swap_id)].swap_id
        if len(family_members) > 1:
            common_keys = set.intersection(*[set(deltas.loc[deltas.swap_id.eq(member), 'substrate_pair_key']) for member in family_members])
            scope_sets.append(('shared_target_pair_panel', d[d.substrate_pair_key.isin(common_keys)]))
        for scope, scoped in scope_sets:
            for role, flags in chosen_flags(target.reaction_type).items():
                for flag in flags:
                    summary, block_rows = feature_contrasts(scoped, context.target_swap_id, context.swap_id, scope, role, flag)
                    summaries.append(summary)
                    blocks.extend(block_rows)
            for selected, role, flag, comparison in restricted_comparisons(scoped, target.reaction_type):
                summary, block_rows = feature_contrasts(selected, context.target_swap_id, context.swap_id, scope, role, flag, comparison)
                summaries.append(summary)
                blocks.extend(block_rows)
        if context.swap_id == context.target_swap_id:
            examples.extend(representative_rows(d, context.target_swap_id))
    summary = pd.DataFrame(summaries)
    block_df = pd.DataFrame(blocks)
    context_df = pd.DataFrame(context_summaries)
    summary.to_csv(results / 'focused_chemical_subgroups.csv', index=False)
    block_df.to_csv(results / 'focused_chemical_partner_blocks.csv', index=False)
    context_df.to_csv(results / 'focused_chemical_contexts.csv', index=False)
    pd.DataFrame(overlaps).to_csv(results / 'focused_chemical_context_overlap.csv', index=False)
    pd.DataFrame(examples).to_csv(results / 'focused_chemical_examples.csv', index=False)
    extra.to_csv(results / 'focused_chemical_extra_annotations.csv', index=False)
    write_report(root, registry, summary, context_df, pd.DataFrame(overlaps), pd.DataFrame(examples))
    leading = summary[(summary.swap_id == summary.target_swap_id) & summary.scope.eq('all') & summary.comparison.eq('flag_vs_complement') & summary.n_molecules0.ge(5) & summary.n_molecules1.ge(5) & summary.n_eligible_partner_blocks.ge(3)]
    leading = leading.assign(strength=leading.partner_block_mean_contrast_pp.abs()).sort_values(['target_swap_id', 'strength'], ascending=[True, False])
    print(leading.groupby('target_swap_id').head(5)[['target_swap_id', 'feature_role', 'feature', 'n_molecules0', 'n_molecules1', 'molecule_mean_delta0', 'molecule_mean_delta1', 'n_eligible_partner_blocks', 'partner_block_mean_contrast_pp', 'partner_block_median_contrast_pp', 'partner_block_positive_fraction']].to_string(index=False))

def write_report(root: Path, registry: pd.DataFrame, summary: pd.DataFrame, contexts: pd.DataFrame, overlaps: pd.DataFrame, examples: pd.DataFrame) -> None:
    lines = ['# Focused posthoc chemical evidence', '', 'These structure comparisons were selected after examining the retrospective swap scan. They are descriptive hypotheses, not prespecified tests or mechanistic conclusions. Delta is condition1 minus condition0 in percentage points; positive values favor the second named component. All selected swaps are strict identity-only changes at fixed target dose, verified from the full recipe JSON. No model is fit and no p value treats repeated molecules or partners as independent observations.', '', '## Chemical readout', '', 'The strongest acylation contrast is between primary aryl/heteroaryl amines (the `primary_aniline` code flag includes heteroaryl carbon attachment) and primary alkyl amines. The primary alkyl groups have negative average effects for replacing FDPP with either comparator, while primary aryl/heteroaryl groups have modest positive averages. Large within-acid contrasts do not mean that each chemical group has an absolute effect of that size: the contrast is the difference between two group means on each shared acid. Both signs occur among aryl amines, including a -90.38 pp PyClocK example, so this is not a deterministic structure rule. Secondary-amine subgroup averages are much closer to zero.', '', '| Acylation exchange | Primary aryl/heteroaryl N / molecule mean (pp) | Primary alkyl N / molecule mean (pp) | Secondary aryl N / molecule mean (pp) | Secondary alkyl N / molecule mean (pp) |', '| --- | --- | --- | --- | --- |']
    for target in TARGETS[:2]:
        cells = []
        for flag in ('flag_primary_aniline', 'flag_primary_alkyl_amine', 'flag_secondary_aniline', 'flag_secondary_alkyl_amine'):
            e = summary[summary.target_swap_id.eq(target) & summary.swap_id.eq(target) & summary.scope.eq('all') & summary.comparison.eq('flag_vs_complement') & summary.feature.eq(flag)].iloc[0]
            cells.append(f'{int(e.n_molecules1)} / {e.molecule_mean_delta1:.2f}')
        lines.append(f'| {target} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Suzuki electrophiles with pyridine-like nitrogen favor the PPh3 catalyst relative to the complement; pyrrole-like electrophiles favor dtbpf on average. These contrasts remain when restricting electrophiles to aryl bromides, so differences between bromides and triflates cannot explain the entire motif contrast. The exact same catalyst exchange has a second, K2CO3 base context, but 412 substrate pairs are reused. The shared-panel motif contrasts retain their directions across bases. Individual pair effects replicate only partially. The large triflate contrast has only three eligible boronic-partner blocks and only one positive-yield block; it is weaker evidence of chemical breadth than the nitrogen pattern.', '', '## How the contrasts are calculated', '', "Each exact swap contains at most one matched record per substrate pair. A molecule mean first averages its matched effects over its observed opposite partners; a molecule-weighted group mean averages these molecule means equally. This changes the weighting but cannot remove partner selection. The stronger descriptive check holds the other reaction partner fixed: only partner blocks with at least two distinct molecules at each feature level are eligible; contrast is mean(delta | flag=1) minus mean(delta | flag=0). Blocks receive equal weight. The median, range and sign fraction are reported. Blocks often reuse the same molecules and therefore are not independent replicates. A flag's complement can contain several different chemical families.", '', 'Sensitivity scopes require both yields strictly positive or remove cells formed from repeated measurements. The positive-only restriction changes the observed population and is a sensitivity check, not a correction. Molecular/partner counts always count unique supplied SMILES. Numeric descriptor and arbitrary fingerprint screening is not used here. The small selected panel examines amine type, acid attachment/alpha substitution, ring nitrogen, aromatic leaving group and ortho occupancy.', '', '## Exact recipe swaps and subgroup evidence', '']
    for target in TARGETS:
        row = registry[registry.swap_id.eq(target)].iloc[0]
        lines += [f'### {target}', '', f'Exchange: **{row.component0_name} -> {row.component1_name}**, dose {row.dose0:g} -> {row.dose1:g} equivalents (source units; catalyst entries use fractions of substrate equivalents). Full background recipes are in `focused_chemical_contexts.csv`. Primary panel: {int(row.n_pairs)} pairs, {int(row.n_sub1)} substrate-1 molecules and {int(row.n_sub2)} substrate-2 molecules. Overall mean delta {row.mean_delta_pp:.2f} pp.', '']
        eligible = summary[summary.swap_id.eq(target) & summary.target_swap_id.eq(target) & summary.scope.eq('all') & summary.comparison.eq('flag_vs_complement') & summary.n_molecules0.ge(5) & summary.n_molecules1.ge(5) & summary.n_eligible_partner_blocks.ge(3)].copy()
        eligible['abs_contrast'] = eligible.partner_block_mean_contrast_pp.abs()
        eligible = eligible.sort_values('abs_contrast', ascending=False).head(5)
        lines += ['The following rows are ranked posthoc by absolute mean within-partner contrast among flags with at least five molecules per level and three eligible partner blocks. This ranking itself is selection and must not be treated as validation.', '', '| Role / flag | Molecules 0 / 1 | Molecule means 0 / 1 (pp) | Shared partner blocks | Block contrast mean / median (pp) | Positive blocks |', '| --- | --- | --- | --- | --- | --- |']
        for _, e in eligible.iterrows():
            lines.append(f'| S{int(e.feature_role)} `{e.feature}` | {int(e.n_molecules0)} / {int(e.n_molecules1)} | {e.molecule_mean_delta0:.2f} / {e.molecule_mean_delta1:.2f} | {int(e.n_eligible_partner_blocks)} | {e.partner_block_mean_contrast_pp:.2f} / {e.partner_block_median_contrast_pp:.2f} | {e.partner_block_positive_fraction:.0%} |')
        lines += ['', 'Positive-yield sensitivity for those same selected flags:', '', '| Role / flag | Molecules 0 / 1 | Blocks | Block contrast mean / median (pp) | Positive blocks |', '| --- | --- | --- | --- | --- |']
        for _, e in eligible.iterrows():
            sensitivity = summary[summary.target_swap_id.eq(target) & summary.swap_id.eq(target) & summary.scope.eq('both_yields_positive') & summary.comparison.eq('flag_vs_complement') & summary.feature_role.eq(e.feature_role) & summary.feature.eq(e.feature)].iloc[0]
            lines.append(f'| S{int(e.feature_role)} `{e.feature}` | {int(sensitivity.n_molecules0)} / {int(sensitivity.n_molecules1)} | {int(sensitivity.n_eligible_partner_blocks)} | {sensitivity.partner_block_mean_contrast_pp:.2f} / {sensitivity.partner_block_median_contrast_pp:.2f} | {sensitivity.partner_block_positive_fraction:.0%} |')
        family = contexts[contexts.target_swap_id.eq(target)]
        lines += ['', f'The same exchange and doses occur in {len(family)} strict background recipe comparisons, including this target. Their panels are observed contexts, not randomized independent replications. Context-specific feature results and panel overlap are supplied as CSVs.', '']
        restricted = summary[summary.swap_id.eq(target) & summary.target_swap_id.eq(target) & summary.scope.isin(['all', 'both_yields_positive']) & summary.comparison.ne('flag_vs_complement')]
        if len(restricted):
            lines += ['Restricted comparisons (specified after observing the broad patterns; still posthoc). For the direct primary comparison, level 0 is primary alkyl and level 1 primary aryl/heteroaryl; molecules positive for both flags are excluded. For the exclusive N comparison, level 0 is pyrrole-only and level 1 pyridine-only.', '', '| Restriction / flag | Scope | Molecules 0 / 1 | Molecule means 0 / 1 (pp) | Blocks | Block mean / median (pp) |', '| --- | --- | --- | --- | --- | --- |']
            for _, e in restricted.iterrows():
                lines.append(f'| `{e.comparison}` / `{e.feature}` | {e.scope} | {int(e.n_molecules0)} / {int(e.n_molecules1)} | {e.molecule_mean_delta0:.2f} / {e.molecule_mean_delta1:.2f} | {int(e.n_eligible_partner_blocks)} | {e.partner_block_mean_contrast_pp:.2f} / {e.partner_block_median_contrast_pp:.2f} |')
            lines.append('')
        if len(family) > 1:
            selected_context = summary[summary.target_swap_id.eq(target) & summary.scope.eq('shared_target_pair_panel') & summary.comparison.eq('flag_vs_complement') & summary.feature.isin(['flag_pyridine_like', 'flag_pyrrole_like'])]
            lines += ['Nitrogen motifs on the shared substrate panel, direction aligned to the target exchange:', '', '| Context | Role / flag | Molecules 0 / 1 | Blocks | Block mean / median (pp) |', '| --- | --- | --- | --- | --- |']
            for _, e in selected_context.iterrows():
                lines.append(f'| {e.swap_id} | S{int(e.feature_role)} `{e.feature}` | {int(e.n_molecules0)} / {int(e.n_molecules1)} | {int(e.n_eligible_partner_blocks)} | {e.partner_block_mean_contrast_pp:.2f} / {e.partner_block_median_contrast_pp:.2f} |')
            lines.append('')
            for _, overlap in overlaps[overlaps.target_swap_id.eq(target)].iterrows():
                lines += [f'Across {int(overlap.n_common_pairs)} common pairs, the **Pearson** correlation between aligned context deltas is {overlap.pearson_delta_correlation:.3f}. Of {int(overlap.n_target_abs20)} target pairs with |delta| >=20 pp, {int(overlap.n_same_sign_of_target_abs20)} have the same nonzero effect sign in the other context and {int(overlap.n_context_abs20_same_sign)} also retain |delta| >=20 pp. These thresholds/populations differ from analyses conditioning on both contexts being decisive.', '']
    lines += ['## Additional structural definitions', '', 'The frozen molecular annotations and their definitions are documented in `CHEMICAL_FEATURE_METHODS.md`. Extra focused annotations never modify that table:', '', '| Feature | Definition |', '| --- | --- |']
    lines += [f'| `chem_{key}` | `{query}` |' for key, query in FOCUSED_SMARTS.items()]
    lines += ['| `chem_aromatic_acid` | A neutral/deprotonated carboxyl carbon directly attached to aromatic carbon |', '| `chem_aliphatic_acid` | A neutral/deprotonated carboxyl carbon directly attached to nonaromatic carbon |', '| `chem_acid_alpha_branched` | The nonaromatic alpha carbon has at least two carbon neighbors other than the carboxyl carbon |', '| `chem_acid_alpha_hetero_substituted` | The nonaromatic alpha carbon is directly attached to N, O or S |', '| `chem_acid_alpha_aryl_substituted` | The nonaromatic alpha carbon is directly attached to aromatic carbon |', '', 'Acid flags may overlap and are topological. Alpha branching is not a computed steric energy. Halide flags can coexist when a structure contains several halogens and do not identify which bond reacts. The leaving-group assignment therefore requires inspection of the supplied product for ambiguous structures. Source roles differ: acylation S1 is amine and S2 acid; Suzuki S1 electrophile and S2 boronic partner; Buchwald-Hartwig S1 amine and S2 electrophile. The raw source functional group strings are included in examples for inspection.', '', "The two acylation comparisons share 906 exact substrate pairs and the same FDPP C18 reference arm; their chemical pattern agreement is correlated evidence, not independent replication of the same swap. Suzuki's second context has reversed condition ordering, and its effects are multiplied by -1 to align with dtbpf-to-PPh3. Complete recipe JSON retains original condition ordering, with the alignment multiplier explicit in the context table. The two Suzuki contexts share 412 pairs, so this is primarily a within-panel base-context comparison.", '', '## Traceability and limits', '', '`focused_chemical_examples.csv` contains up to four distinct-molecule examples per sign (|delta| >=20 pp) for each target, restricted to yields >0 in both arms. It retains both SMILES, yields, effect, source functional-group strings, file/sheet, exact observation IDs, original sample labels, archive and repeated-cell flags. Each example is an observed pair, not prospective evidence.', '', '`focused_chemical_subgroups.csv` contains all selected-panel subgroup results for all strict matched backgrounds and sensitivity scopes, including small or unsupported groups rather than suppressing them. `focused_chemical_partner_blocks.csv` exposes every eligible shared-partner contrast. `focused_chemical_contexts.csv` contains complete recipes; `focused_chemical_context_overlap.csv` quantifies shared panels and the descriptive effect agreement. Reusing the same substrate panel across backgrounds provides a within-panel context comparison, not an external chemical validation.', '', 'These results cannot distinguish catalyst/coupling-reagent mechanism, yield quantitation limitations, batch effects, source selection, or true substrate interactions. Regime claims need the separate grouped and selection-aware validation analyses. The group contrasts here are selected, dependent, retrospective and potentially confounded beyond the held-fixed opposite partner.', '']
    lines += ['## Representative nonzero pairs', '', 'Both arms are strictly above zero in every row below. Percentages are observed source yields; the full example CSV retains source functional-group strings and sample labels.', '', '| Exchange / sign | S1 supplied SMILES | S2 supplied SMILES | Yield 0 -> 1 (%) | Delta (pp) | Source observation IDs 0 / 1 |', '| --- | --- | --- | --- | --- | --- |']
    for target in (TARGETS[0], TARGETS[2]):
        for sign in ('positive20', 'negative20'):
            selected = examples[examples.target_swap_id.eq(target) & examples.effect_sign.eq(sign)]
            if target == TARGETS[0] and sign == 'negative20':
                selected = selected[selected.s1_positive_annotations.str.contains('flag_primary_alkyl_amine')]
            if len(selected):
                e = selected.iloc[0]
                lines.append(f'| {target} / {sign} | `{e.sub_1_smiles}` | `{e.sub_2_smiles}` | {100 * e.y0:.2f} -> {100 * e.y1:.2f} | {e.delta_pp:.2f} | `{e.observation_id0}` / `{e.observation_id1}` |')
    lines.append('')
    (root / 'reports' / 'FOCUSED_CHEMICAL_EVIDENCE.md').write_text('\n'.join(lines), encoding='utf-8')
if __name__ == '__main__':
    main()
