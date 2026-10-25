"""Final context-dependence and learning figures from reproducible M1 outputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from rdkit import Chem
from publication_figures import common as c
REFERENCE_LEARNING = c.A / 'scripts/reference_learning/Analysis'
RESPONSE_LEARNING = c.A / 'scripts/response_learning'
EDM = RESPONSE_LEARNING / 'Existing_Data_Mining_2026-10-16/results'
E59 = RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/results'
E63 = RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results'
METRICS = c.A / 'out/direction_metrics'
DELTA = LinearSegmentedColormap.from_list('final_delta', [c.RED, 'white', c.BLUE])
CLASSES = {'acylation': 'Amide coupling', 'allylic-substitution': 'Allylic substitution', 'buchwald-hartwig': 'Buchwald–Hartwig', 'buchwald-hartwig-cmc': 'Buchwald–Hartwig (CMC)', 'c-o-competitive': 'C–O competition', 'chan-lam': 'Chan–Lam', 'esterification': 'Esterification', 'grignard': 'Grignard', 'heck': 'Heck', 'hiyama': 'Hiyama', 'kumada': 'Kumada', 'minisci': 'Minisci', 'pd-carbonylation': 'Pd carbonylation', 'reductive-amination': 'Reductive amination', 'snar': 'SNAr', 'sonogashira': 'Sonogashira', 'suzuki': 'Suzuki', 'suzuki-bpin': 'Suzuki BPin', 'ullmann': 'Ullmann'}

def screen_label(value):
    name, number = str(value).rsplit('|', 1) if '|' in str(value) else (str(value), '')
    normalized = name.lower().replace('buchwald-hartwig(cmc)', 'buchwald-hartwig-cmc')
    normalized = normalized.replace('c-o competitive reactions', 'c-o-competitive')
    name = CLASSES.get(normalized, name)
    return f'{name} {number}'.strip()

def source(name, data, paths, selection, unit='', note=''):
    return c.source(name, data, paths, selection, unit, note)

def context_data():
    path = EDM / 'matched_response_profiles.parquet'
    profiles = c.frame(path)
    selected = profiles.loc[profiles.reaction_type.eq('suzuki') & profiles.swap_slot.eq('reagent1') & profiles.component_pair.eq('CMP0092|CMP0097')].copy()
    if selected.empty or selected.duplicated(['substrate_pair_key', 'background_fingerprint']).any():
        raise ValueError('Flagship response cells are not unique')
    backgrounds = sorted(selected.background_fingerprint.unique())
    if len(backgrounds) != 2:
        raise ValueError('Flagship substitution lost one of its two backgrounds')
    cs = next((x for x in backgrounds if 'CMP0333@3' in x))
    kk = next((x for x in backgrounds if 'CMP0334@3' in x))
    wide = selected.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta')
    wide = wide[[cs, kk]].dropna().rename(columns={cs: 'Cs2CO3_delta_pp', kk: 'K2CO3_delta_pp'}).reset_index()
    wide['opposite_sign'] = wide.Cs2CO3_delta_pp * wide.K2CO3_delta_pp < 0
    if len(wide) != 412 or wide.opposite_sign.sum() != 145:
        raise ValueError('Shared-background headline changed')
    source('Fig2_a_shared_backgrounds', wide, path, 'Suzuki; reagent1; CMP0092|CMP0097; both observed backgrounds, finite shared substrate pairs', 'yield change pp')
    witness = selected.loc[selected.sub_1_smiles.isin(['Brc1ccc2[nH]ncc2c1', 'COc1cccnc1Br']) & selected.sub_2_smiles.eq('COc1ccc(B(O)O)cc1')].copy()
    pivot = witness.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta')[[cs, kk]]
    if pivot.shape != (2, 2) or abs(pivot.to_numpy()).min() < 20 or np.prod(np.sign(pivot.to_numpy())) >= 0:
        raise ValueError('Original strong nonseparable four-cell witness changed')
    source('Fig2_a_four_cell_witness', witness, path, 'Original indazole/methoxybromopyridine sign witness; common 4-methoxyphenylboronic acid', 'yield change pp')
    return (wide, witness, pivot)

def six_swaps():
    registry_path = E59 / 'full_edge_strict_swap_registry.csv'
    corpus_path = E59 / 'full_edge_response_corpus.parquet'
    registry, corpus = (c.frame(registry_path), c.frame(corpus_path))
    selected, response_sets = ([], {})
    rule = 'Largest substrate-pair support in each prescribed transformation, descending n_pairs and lexical swap_id; the mandated Suzuki flagship suzuki__C23__C26 overrides the transformation maximum'
    for cls in ['suzuki', 'acylation', 'buchwald-hartwig', 'snar', 'ullmann', 'kumada']:
        candidates = registry.loc[registry.reaction_type.eq(cls)].sort_values(['n_pairs', 'swap_id'], ascending=[False, True])
        if candidates.empty:
            raise ValueError(f'No measured substitution in {cls}')
        chosen = registry.loc[registry.swap_id.eq('suzuki__C23__C26')].iloc[0] if cls == 'suzuki' else candidates.iloc[0]
        rows = corpus.loc[corpus.reaction_type.eq(cls) & corpus.swap_slot.eq(chosen.swap_slot) & corpus.component_pair.eq(chosen.component_pair) & corpus.background_fingerprint.eq(chosen.background_fingerprint)].copy()
        if len(rows) != int(chosen.n_pairs) or len(rows) < 50:
            raise ValueError('Six-class substitution support changed')
        rows['display_delta_pp'] = rows.delta * chosen.orientation_sign
        rows['swap_id'] = chosen.swap_id
        source(f"Fig2_b_{cls.replace('buchwald-hartwig', 'BH')}", rows, [registry_path, corpus_path], rule + '; delta oriented to component0_id to component1_id', 'yield change pp')
        selected.append(chosen)
        response_sets[cls] = rows
    selected = pd.DataFrame(selected)
    source('Fig2_b_selected_swaps', selected, registry_path, rule, 'exact substitutions and pair counts')
    return (selected, response_sets)

def sign_data():
    path = EDM / 'sparse_sign_feasibility_summary.csv'
    raw = c.frame(path)
    rows = raw.loc[raw.population.eq('all') & raw.panel.eq('observed_sparse') & raw.objective.eq('count') & raw.threshold_pp.isin([0, 20]) & ~raw.stratum.isin(['ALL', 'finkelstein'])].copy()
    if len(rows) != 38 or rows.stratum.nunique() != 19:
        raise ValueError('Sign-conflict panel requires all nineteen arrays at both thresholds')
    rows['conflict_percent'] = np.where(rows.n_cycle_edges.gt(0), 100 * rows.n_conflicting_edges / rows.n_cycle_edges, np.nan)
    totals = rows.groupby('threshold_pp')[['n_cycle_edges', 'n_conflicting_edges']].sum()
    if tuple(totals.loc[0]) != (283, 271) or tuple(totals.loc[20]) != (212, 143):
        raise ValueError('Sign-conflict headline changed')
    source('Fig2_c_array_conflicts', rows, path, 'all; observed_sparse; count; threshold0/20pp; all19 arrays excluding Finkelstein', 'conflict percent and counts', 'No cyclic substitution means an undefined percentage, not zero conflict')
    return rows

def figure2():
    wide, witness, pivot = context_data()
    swaps, response_sets = six_swaps()
    signs = sign_data()
    fig = c.newfig(165)
    a = c.panel(fig, [0.085, 0.535, 0.445, 0.385], 'a', 'Catalyst responses across base backgrounds')
    for rect in [(-100, 0), (0, -100)]:
        a.add_patch(Rectangle(rect, 100, 100, fc=c.RED, alpha=0.065, lw=0))
    a.scatter(wide.Cs2CO3_delta_pp, wide.K2CO3_delta_pp, color=c.GRAY, s=5, alpha=0.42, lw=0)
    a.axhline(0, color=c.GRAY, lw=0.5)
    a.axvline(0, color=c.GRAY, lw=0.5)
    a.set(xlim=(-100, 100), ylim=(-100, 100), xticks=[-100, 0, 100], yticks=[-100, 0, 100])
    c.clean(a, 'Cs₂CO₃ response (pp)', 'K₂CO₃ response (pp)')
    correlation = wide[['Cs2CO3_delta_pp', 'K2CO3_delta_pp']].corr().iloc[0, 1]
    a.text(0.035, 0.96, f'r = {correlation:.2f}\nOpposite signs: 145/412', transform=a.transAxes, va='top', fontsize=6)
    numbered_pairs = []
    for i, (smiles, color) in enumerate([('Brc1ccc2[nH]ncc2c1', c.RED), ('COc1cccnc1Br', c.BLUE)]):
        pair = witness.loc[witness.sub_1_smiles.eq(smiles), 'substrate_pair_key'].iloc[0]
        numbered_pairs.append(pair)
        x, y = pivot.loc[pair].to_numpy()
        a.scatter([x], [y], s=22, color=color, edgecolors='white', lw=0.45, zorder=5)
        a.annotate(str(i + 1), (x, y), xytext=(4, 4), textcoords='offset points', color=color, fontsize=6)
        cy = 0.815 - i * 0.145
        c.molecule(fig, [0.6, cy - 0.025, 0.18, 0.09], smiles)
        fig.text(0.8, cy + 0.015, str(i + 1), color=color, fontsize=6, va='center')
    fig.text(0.6, 0.902, 'PdCl₂(dtbpf) → PdCl₂(PPh₃)₂', fontsize=6)
    fig.text(0.6, 0.875, 'Common 4-methoxyphenylboronic acid', fontsize=5.5)
    grid = fig.add_axes([0.825, 0.64, 0.125, 0.14])
    numbered_responses = pivot.loc[numbered_pairs].to_numpy()
    grid.pcolormesh(np.arange(3) - 0.5, np.arange(3) - 0.5, numbered_responses, cmap=DELTA, norm=TwoSlopeNorm(0, -100, 100), rasterized=False)
    for (i, j), value in np.ndenumerate(numbered_responses):
        grid.text(j, i, f'{value:+.0f}', ha='center', va='center', fontsize=6, color='white' if abs(value) > 60 else c.INK)
    grid.set(xlim=(-0.5, 1.5), ylim=(1.5, -0.5))
    grid.set_xticks([0, 1], ['Cs₂CO₃', 'K₂CO₃'], fontsize=5)
    grid.set_yticks([0, 1], ['1', '2'], fontsize=5.5)
    grid.tick_params(length=0, pad=2)
    grid.spines[:].set_visible(False)
    fig.text(0.885, 0.6, 'Response (pp)', ha='center', fontsize=5.5)
    fig.text(0.6, 0.548, 'Three negative, one positive:\nno pair sign × background sign can reproduce all four.', fontsize=5.5, linespacing=1.2)
    b = c.panel(fig, [0.06, 0.075, 0.51, 0.325], 'b', 'Six chemical transformations')
    b.axis('off')
    titles = {'suzuki': 'PdCl₂(dtbpf) → PdCl₂(PPh₃)₂', 'acylation': 'TFFH → DCMT', 'buchwald-hartwig': 'RuPhos Pd G3 →\nBrettPhos Pd G4', 'snar': 'Triethylamine →\n2,6-lutidine', 'ullmann': 'N-Methyl-L-proline →\nL-proline', 'kumada': 'Tetramethylurea → DMI'}
    for i, row in enumerate(swaps.itertuples()):
        col, rr = (i % 3, i // 3)
        x, y = (0.065 + col * 0.177, 0.275 - rr * 0.165)
        ax = fig.add_axes([x, y, 0.145, 0.055])
        values = response_sets[row.reaction_type].display_delta_pp.to_numpy()
        jitter = (np.arange(len(values)) * 0.61803398875 % 1 - 0.5) * 0.55
        ax.scatter(values, jitter, color=np.where(values > 0, c.BLUE, c.RED), s=1.7, alpha=0.5, lw=0)
        ax.axvline(0, color=c.GRAY, lw=0.4)
        ax.set(xlim=(-100, 100), ylim=(-0.5, 0.5), yticks=[])
        ax.set_xticks([-100, 0, 100], ['−100', '0', '100'] if rr else [])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(length=2, pad=1, labelsize=5)
        fig.text(x, y + 0.072, CLASSES[row.reaction_type], fontsize=5.5, weight='bold')
        fig.text(x, y + 0.058, f'n={len(values):,}', fontsize=5)
        fig.text(x, y + 0.112, titles[row.reaction_type], fontsize=5, linespacing=1.05)
    fig.text(0.29, 0.05, 'Yield change (pp)', ha='center', fontsize=6)
    ax = c.panel(fig, [0.78, 0.075, 0.115, 0.325], 'c', 'Sign conflicts', title_x=0.64)
    order = signs.loc[signs.threshold_pp.eq(20)].sort_values(['conflict_percent', 'stratum'], ascending=[False, True], na_position='last').stratum.tolist()
    for i, name in enumerate(order):
        group = signs.loc[signs.stratum.eq(name)].set_index('threshold_pp')
        values = group.loc[[0, 20], 'conflict_percent'].to_numpy()
        if np.isfinite(values).all():
            ax.plot(values, [i, i], color=c.GRAY, lw=0.45)
        for value, color, marker in zip(values, [c.BLUE, c.RED], ['o', 'D']):
            if np.isfinite(value):
                ax.scatter(value, i, color=color, marker=marker, s=7, lw=0, zorder=3)
        if not np.isfinite(values).any():
            ax.text(2, i, 'No cycles', color=c.GRAY, fontsize=4.8, va='center')
        counts = ' · '.join((f"{int(group.loc[t, 'n_conflicting_edges'])}/{int(group.loc[t, 'n_cycle_edges'])}" for t in [0, 20]))
        ax.text(1.07, i, counts, transform=ax.get_yaxis_transform(), va='center', fontsize=5)
    ax.set_yticks(range(len(order)), [CLASSES[x] for x in order], fontsize=4.8)
    ax.set(xlim=(-3, 105), ylim=(len(order) - 0.5, -0.5), xticks=[0, 50, 100])
    c.clean(ax, 'Conflicting swaps (%)')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=c.BLUE, label='0 pp', markersize=3), Line2D([], [], marker='D', ls='', color=c.RED, label='20 pp', markersize=3)], loc='upper left', bbox_to_anchor=(-0.05, 1.13), frameon=False, ncol=2, fontsize=5, handletextpad=0.3, columnspacing=0.7)
    fig.text(0.903, 0.405, '0 pp · 20 pp', fontsize=5, va='bottom')
    c.save(fig, 'Figure_2', 'Substitution responses depend jointly on substrate and recipe background')

def main_screen_data():
    path = REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv'
    raw = c.frame(path)
    screens = raw.loc[raw.level.eq('screen')].pivot(index='screen', columns='model', values='mse_pp2')
    if len(screens) != 17:
        raise ValueError('Matched comparison no longer has seventeen screens')
    headline = raw.loc[raw.level.eq('headline')].set_index('model')
    overall = {'screen': 'Overall', 'marginal': headline.loc['marginal', 'mse_pp2']}
    for model in ['direct_raw', 'semantic_full_raw']:
        screens[model + '_relative_pct'] = 100 * (screens[model] / screens.marginal - 1)
        overall[model] = headline.loc[model, 'mse_pp2']
        overall[model + '_relative_pct'] = 100 * (overall[model] / overall['marginal'] - 1)
    shown = screens.reset_index()
    shown = shown.sort_values(['semantic_full_raw_relative_pct', 'screen'])
    shown = pd.concat([pd.DataFrame([overall]), shown], ignore_index=True)
    source('Fig3_a_screen_comparison', shown, path, 'Matched strict pair-held-out/full-history population; Overall hierarchical group average first; screens ordered by learned error change ascending', 'error change versus historical marginal percent')
    return shown

def direction_errors():
    path = METRICS / 'direction_accuracy_summary.csv'
    raw = c.frame(path)
    rows = raw.loc[raw.threshold_pp.eq(20) & raw.aggregation.eq('pooled') & raw.model.isin(['direct_raw', 'semantic_full_raw'])].copy()
    rows = rows.set_index('model').loc[['direct_raw', 'semantic_full_raw']].reset_index()
    rows['wrong_per_100'] = 100 * (1 - rows.accuracy)
    rows['n_wrong'] = rows.n_targets - rows.n_correct
    if rows.n_targets.tolist() != [21997, 21997] or rows.n_wrong.tolist() != [4097, 2982]:
        raise ValueError('Direction-error headline changed')
    source('Fig3_b_direction_errors', rows, path, 'Pooled unique repeat0 targets with measured absolute change at least20pp; zero predictions count as errors', 'errors per100 and target counts')
    return rows

def training_data():
    path, coef_path = (METRICS / 'selected_training_pairs.csv', METRICS / 'learned_training_relationships.csv')
    points, coefficients = (c.frame(path), c.frame(coef_path))
    for key in ['screen', 'edge_id', 'repeat', 'fold']:
        unique = points[key].unique()
        if len(unique) != 1:
            raise ValueError('Training illustration spans multiple frozen relationships')
        coefficients = coefficients.loc[coefficients[key].eq(unique[0])]
    if len(points) != 452 or len(coefficients) != 1:
        raise ValueError('Stored training-relationship support changed')
    source('Fig3_c_training_pairs', points, path, 'Unchanged support-selected screen/edge/lowest fold; training pairs only', 'yield change pp')
    source('Fig3_c_training_relation', coefficients, coef_path, 'Stored full S+D relationship; section at centred reference S=0, no new fit', 'slope and intercept pp')
    return (points, coefficients.iloc[0])

def reagent_identities():
    path = REFERENCE_LEARNING / 'data/raw_compound_table.csv'
    raw = c.frame(path)
    records = [{'compound_id': 'CMP0482', 'registry_name': 'pentafluorophenol', 'display_name': 'Pentafluorophenol', 'cas': '771-61-9', 'smiles': 'Oc1c(F)c(F)c(F)c(F)c1F', 'inchikey': 'XBNGYFFABRKICK-UHFFFAOYSA-N', 'identity_source': 'https://pubchem.ncbi.nlm.nih.gov/compound/13041'}, {'compound_id': 'CMP0484', 'registry_name': 'NHPI', 'display_name': 'N-Hydroxyphthalimide', 'cas': '524-38-9', 'smiles': 'O=C1c2ccccc2C(=O)N1O', 'inchikey': 'CFMZSMGAMPBRBE-UHFFFAOYSA-N', 'identity_source': 'https://pubchem.ncbi.nlm.nih.gov/compound/10665'}]
    for row in records:
        match = raw.loc[raw['Compound ID'].eq(row['compound_id'])]
        if len(match) != 1 or match['Standardized Name'].iloc[0] != row['registry_name'] or row['cas'] not in str(match.cas.iloc[0]):
            raise ValueError('Actual reagent registry identity changed')
        if Chem.MolToInchiKey(Chem.MolFromSmiles(row['smiles'])) != row['inchikey']:
            raise ValueError('Actual reagent structure identity changed')
    source('Fig3_c_reagent_identities', pd.DataFrame(records), path, 'CAS-linked identities checked2026-10-22; actual substituted reagents CMP0482/CMP0484', 'chemical identities')
    return records

def specificity_data():
    path = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_screen_scores.csv'
    hp = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_headline.csv'
    raw, headline = (c.frame(path), c.frame(hp))
    data = raw.loc[raw.domain.eq('strict_641') & raw.scope.eq('all_panels_complete') & raw.weighting.eq('context_balanced')].copy()
    head = headline.loc[headline.domain.eq('strict_641') & headline.scope.eq('all_panels_complete') & headline.weighting.eq('context_balanced') & headline.macro.eq('family_macro')].copy()
    if len(data) != 102 or len(head) != 6:
        raise ValueError('Same-substitution specificity population changed')
    source('Fig3_d_specificity_screens', data, path, 'strict_641; all_panels_complete; context_balanced; all six settings', 'fraction of matched-control MSE removed')
    source('Fig3_d_specificity_macro', head, hp, 'Same settings; original hierarchical group average (macro=family_macro)', 'fraction of matched-control MSE removed')
    return (data, head)

def figure3():
    shown, errors = (main_screen_data(), direction_errors())
    points, coefficient = training_data()
    identities = reagent_identities()
    data, headline = specificity_data()
    fig = c.newfig(170)
    a = c.panel(fig, [0.51, 0.535, 0.44, 0.385], 'a', 'Prediction error with the same two reference wells', title_x=0.06)
    logic = fig.add_axes([0.04, 0.645, 0.205, 0.22])
    logic.set_gid('same_pair_reference_schematic')
    logic.set(xlim=(0, 1), ylim=(0, 1))
    logic.axis('off')
    fig.text(0.1425, 0.89, 'Same substrate pair', ha='center', fontsize=6, weight='bold')
    for j, background in enumerate(['Reference\nbackground', 'Target\nbackground']):
        xx = 0.22 + j * 0.45
        logic.text(xx + 0.115, 0.91, background, ha='center', va='bottom', fontsize=5.5, linespacing=1.05)
        for i, reagent in enumerate(['X', 'Y']):
            yy = 0.66 - i * 0.23
            if j == 0:
                logic.text(0.055, yy + 0.085, reagent, ha='center', va='center', fontsize=5.5)
            logic.add_patch(Rectangle((xx, yy), 0.23, 0.17, fc='#DAEAE8' if j == 0 else 'white', ec=c.GRAY, lw=0.55))
            value = ('yₓ' if i == 0 else 'yᵧ') if j == 0 else '?'
            logic.text(xx + 0.115, yy + 0.085, value, ha='center', va='center', fontsize=6)
    logic.plot([0.2, 0.16, 0.16, 0.2], [0.745, 0.745, 0.515, 0.515], color=c.GRAY, lw=0.55)
    logic.plot([0.16, 0.16], [0.515, 0.43], color=c.GRAY, lw=0.55)
    for endpoint, label, color in [((0.2, 0.2), 'Copy Δ', c.RED), ((0.73, 0.2), 'Learn\n+ history', c.TEAL)]:
        logic.add_patch(FancyArrowPatch((0.16, 0.43), endpoint, arrowstyle='-|>', mutation_scale=5, lw=0.65, color=color))
        logic.text(endpoint[0], 0.09, label, ha='center', va='top', color=color, fontsize=5.5, linespacing=1.05)
    limit = 80
    for i, row in enumerate(shown.itertuples()):
        learned, copied = (row.semantic_full_raw_relative_pct, row.direct_raw_relative_pct)
        a.plot([learned, min(copied, limit - 2)], [i, i], color=c.GRAY, lw=0.65)
        a.scatter(learned, i, color=c.TEAL, s=17 if i == 0 else 10, zorder=3)
        if copied >= limit:
            a.plot(limit - 2, i, marker='>', color=c.RED, ms=4)
            a.text(limit - 5, i - 0.3, f'{copied:.0f}%', color=c.RED, ha='right', fontsize=5.3)
        else:
            a.scatter(copied, i, color=c.RED, s=17 if i == 0 else 10, zorder=3)
    a.axvline(0, color=c.GRAY, ls='--', lw=0.5)
    a.set_yticks(range(len(shown)), [screen_label(x) for x in shown.screen], fontsize=5.5)
    a.get_yticklabels()[0].set_weight('bold')
    a.set(xlim=(-45, 80), ylim=(len(shown) - 0.5, -0.8), xticks=[-40, 0, 40, 80])
    c.clean(a, 'Error change vs history alone (no reference) (%)')
    for value, color, ha in [(shown.iloc[0].semantic_full_raw_relative_pct, c.TEAL, 'right'), (shown.iloc[0].direct_raw_relative_pct, c.RED, 'left')]:
        a.text(value + (-1.7 if ha == 'right' else 1.7), -0.17, f'{value:+.1f}%', color=color, ha=ha, va='bottom', fontsize=6)
    a.legend(handles=[Line2D([], [], marker='o', ls='', color=c.RED, label='Copy', markersize=3), Line2D([], [], marker='o', ls='', color=c.TEAL, label='Learn', markersize=3)], loc='upper right', bbox_to_anchor=(1, 1.105), frameon=False, ncol=2, fontsize=6)
    b = c.panel(fig, [0.105, 0.115, 0.14, 0.285], 'b', 'Missed or wrong directions', title_x=0.065)
    b._draft_panel_title.set_y(0.436)
    b._draft_panel_letter.set_y(0.436)
    b.bar([0, 1], errors.wrong_per_100, color=[c.RED, c.TEAL], width=0.55)
    for i, row in enumerate(errors.itertuples()):
        b.text(i, row.wrong_per_100 + 0.8, f'{row.wrong_per_100:.1f}\n({int(row.n_wrong):,})', ha='center', fontsize=5.5)
    b.set(xlim=(-0.6, 1.6), ylim=(0, 25), yticks=[0, 10, 20])
    b.set_xticks([0, 1], ['Copy', 'Learn'], fontsize=5.5)
    c.clean(b, ylabel='Errors per 100 large changes')
    b.text(0.5, -0.23, '21,997 targets; |Δ| ≥ 20 pp', ha='center', transform=b.transAxes, fontsize=5)
    ax = c.panel(fig, [0.385, 0.125, 0.205, 0.205], 'c', 'Reference and target responses', title_x=0.35)
    ax._draft_panel_title.set_y(0.436)
    ax._draft_panel_letter.set_y(0.436)
    ax.scatter(points.reference_contrast_pp, points.target_contrast_pp, color=c.GRAY, s=3.7, alpha=0.4, lw=0)
    xx = np.array([-100, 100])
    ax.plot(xx, xx, color=c.RED, lw=0.8)
    ax.plot(xx, coefficient.centered_S_zero_slice_intercept_pp + coefficient.centered_S_zero_slice_slope * xx, color=c.TEAL, lw=0.9)
    ax.text(42, 62, 'Copy (slope 1)', color=c.RED, fontsize=5.5, rotation=39)
    learned_x = 28
    learned_y = coefficient.centered_S_zero_slice_intercept_pp + coefficient.centered_S_zero_slice_slope * learned_x
    ax.text(learned_x, learned_y - 16, 'Learned', color=c.TEAL, fontsize=5.5)
    ax.axhline(0, color=c.LIGHT, lw=0.45)
    ax.axvline(0, color=c.LIGHT, lw=0.45)
    ax.set(xlim=(-100, 100), ylim=(-100, 100), xticks=[-100, 0, 100], yticks=[-100, 0, 100])
    c.clean(ax, 'Reference (pp)', 'Target (pp)')
    ax.text(0.04, 0.94, '452 training pairs', transform=ax.transAxes, va='top', fontsize=5)
    for x, identity in zip([0.35, 0.498], identities):
        c.molecule(fig, [x, 0.355, 0.13, 0.055], identity['smiles'])
    fig.add_artist(FancyArrowPatch((0.48, 0.38), (0.5, 0.38), transform=fig.transFigure, arrowstyle='->', mutation_scale=5, lw=0.55, color=c.INK))
    fig.text(0.49, 0.405, 'reagent swap', ha='center', fontsize=5)
    ax = c.panel(fig, [0.835, 0.115, 0.12, 0.285], 'd', 'Same-substitution reference', title_x=0.65)
    ax._draft_panel_title.set_y(0.436)
    ax._draft_panel_letter.set_y(0.436)
    settings = [(s, f) for s in ['pair', 'role1', 'role2'] for f in [0.25, 1.0]]
    labels = {'pair': 'Pair', 'role1': 'First reactant', 'role2': 'Second reactant'}
    for i, (split, fraction) in enumerate(settings):
        local = data.loc[data.split.eq(split) & data.fraction.eq(fraction)].sort_values('screen')
        value = headline.loc[headline.split.eq(split) & headline.fraction.eq(fraction)]
        if len(local) != 17 or len(value) != 1:
            raise ValueError('Specificity setting no longer contains seventeen screens')
        ax.scatter(100 * local.other_recovery_vs_control, i + np.linspace(-0.17, 0.17, len(local)), color=c.GRAY, alpha=0.65, s=5, lw=0)
        ax.scatter(100 * value.other_recovery_vs_control, [i], color=c.TEAL, s=17, edgecolors='white', lw=0.3, zorder=4)
        ax.text(-5 / 3, i, labels[split], transform=ax.get_yaxis_transform(), va='center', fontsize=5.5)
        ax.text(-0.23, i, f'{int(100 * fraction)}%', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.5)
    ax.axvline(0, color=c.GRAY, ls='--', lw=0.5)
    ax.set_yticks(range(6), [''] * len(settings))
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(5.8, -0.6)
    fig.text(0.635, 0.405, 'Held out', fontsize=5.5, weight='bold')
    fig.text(0.8074, 0.405, 'History', ha='right', fontsize=5.5, weight='bold')
    c.clean(ax, 'Error removed versus\nmatched control (%)')
    c.save(fig, 'Figure_3', 'Historical relationships make the same reference measurements useful')

def cross_data():
    path = REFERENCE_LEARNING / 'tables/cross_library_source_channels_direction_scores.csv'
    raw = c.frame(path)
    rows = raw.loc[raw.domain.eq('b24_fixed') & raw.relation.eq('cross_family') & raw.channel.eq('d') & raw.budget.eq(192) & raw.status.eq('scored')].sort_values(['source_screen', 'target_screen']).copy()
    if len(rows) != 17:
        raise ValueError('Cross-library population changed')
    for field in ['target_mean_mse', 'real_mse', 'test_permuted_exact_mse']:
        rows[field + '_relative_pct'] = 100 * rows[field] / rows.target_mean_mse
    rows['calibrated_worse_than_target_only'] = rows.real_mse.gt(rows.target_mean_mse)
    if rows.calibrated_worse_than_target_only.sum() != 15:
        raise ValueError('Cross-library direction headline changed')
    source('S3_a_transfer_directions', rows, path, 'b24_fixed; transfer between libraries (relation=cross_family); d; budget192; status scored; all17 directions', 'error percent of target-only baseline')
    return rows

def extended_cross(stem='Extended_Data_Figure_3'):
    rows = cross_data()
    fig = c.newfig(100)
    ax = c.panel(fig, [0.12, 0.18, 0.8, 0.65], 'a', 'Transfer between libraries')
    fields = ['target_mean_mse_relative_pct', 'real_mse_relative_pct', 'test_permuted_exact_mse_relative_pct']
    for row in rows.itertuples():
        color = c.RED if row.calibrated_worse_than_target_only else c.TEAL
        values = [getattr(row, field) for field in fields]
        ax.plot([0, 1, 2], values, color=color, alpha=0.5, lw=0.7)
        ax.scatter([0, 1, 2], values, c=[c.BLUE, color, c.GRAY], s=10, zorder=3)
    ax.axhline(100, color=c.GRAY, lw=0.5, ls='--')
    ax.set(xlim=(-0.2, 2.2))
    ax.set_xticks([0, 1, 2], ['Target only', 'Calibrated transfer', 'Reassigned identities'])
    c.clean(ax, ylabel='Error (% of target only)')
    ax.text(0.025, 0.95, 'Calibration worsens 15/17 directions', transform=ax.transAxes, va='top', fontsize=6)
    c.save(fig, stem, 'Calibration between libraries usually worsens the target-only baseline')

def structure_data():
    component_path, profile_path = (E63 / 'e63_vector_components.parquet', EDM / 'matched_response_profiles.parquet')
    population = c.frame(component_path)
    population = population.loc[population.nominal_k.eq(3) & population.model.eq('centroid')].copy()
    keys = ['reaction_type', 'swap_slot', 'component_pair']
    support = population.groupby(keys).size().rename('n_vectors').reset_index()
    chosen = support.sort_values(['n_vectors'] + keys, ascending=[False, True, True, True]).iloc[0]
    block = population
    for key in keys:
        block = block.loc[block[key].eq(chosen[key])]
    means = block[keys + ['vector_id', 'substrate_pair_key', 'true_mean_pp']].drop_duplicates()
    if not means.vector_id.is_unique or not means.substrate_pair_key.is_unique:
        raise ValueError('Response vector selection identities are not unique')
    low = means.sort_values(['true_mean_pp', 'substrate_pair_key', 'vector_id']).iloc[0]
    high = means.sort_values(['true_mean_pp', 'substrate_pair_key', 'vector_id'], ascending=[False, True, True]).iloc[0]
    selected = means.loc[means.vector_id.isin([low.vector_id, high.vector_id])]
    profiles = c.frame(profile_path)
    vectors = profiles.merge(selected, on=keys + ['substrate_pair_key'], how='inner', validate='many_to_one')
    vectors['background_rank'] = vectors.background_fingerprint.map({v: i + 1 for i, v in enumerate(sorted(vectors.background_fingerprint.unique()))})
    actual = vectors.groupby('vector_id').delta.mean()
    if vectors.vector_id.nunique() != 2 or not np.allclose(actual, selected.set_index('vector_id').loc[actual.index, 'true_mean_pp']):
        raise ValueError('Measured response means do not match decomposition')
    source('S4_a_selection_population', means, component_path, 'k3 centroid; substitution with largest vector support, lexical ties; complete population', 'own mean response pp')
    source('S4_a_response_vectors', vectors, [component_path, profile_path], 'Minimum and maximum mean vectors in that full substitution population; lexical ties', 'yield change pp')
    summary_path, gain_path = (E63 / 'e63_summary.csv', E63 / 'e63_paired_component_summary.csv')
    summary, paired = (c.frame(summary_path), c.frame(gain_path))
    baseline = summary.loc[summary.model.eq('centroid') & summary.nominal_k.eq(3) & summary.weighting.eq('equal_class_edge_balanced')]
    gains = paired.loc[paired.comparison.eq('centroid_minus_codebook_continuous')].copy()
    selected_gain = gains.loc[gains.nominal_k.eq(3) & gains.weighting.eq('equal_class_edge_balanced')]
    if len(baseline) != 1 or len(selected_gain) != 1:
        raise ValueError('Decomposition baseline is not unique')
    base, gain = (baseline.iloc[0], selected_gain.iloc[0])
    layers = pd.DataFrame([{'layer': "Pair's own level", 'baseline_error_pp2': base.offset_mse_pp2, 'removed_error_pp2': gain.offset_difference_pp2}, {'layer': 'Background fluctuation', 'baseline_error_pp2': base.amplitude_mse_pp2 + base.alignment_mse_pp2, 'removed_error_pp2': gain.amplitude_difference_pp2 + gain.alignment_difference_pp2}])
    layers['within_layer_error_removed_pct'] = 100 * layers.removed_error_pp2 / layers.baseline_error_pp2
    gains['own_mean_share_pct'] = 100 * gains.offset_difference_pp2 / gains.mse_difference_pp2
    source('S4_b_within_layer_improvement', layers, [summary_path, gain_path], 'Group/edge-balanced k3; removed MSE divided by same-layer centroid MSE', 'within-layer error removed percent')
    source('S4_c_net_gain_components', gains, gain_path, 'centroid_minus_codebook_continuous; k2/k3 and all three original weightings', 'own level share of total net improvement percent')
    return (vectors, layers, gains)

def extended_structure(stem='Extended_Data_Figure_4'):
    vectors, layers, gains = structure_data()
    fig = c.newfig(140)
    ax = c.panel(fig, [0.09, 0.38, 0.39, 0.48], 'a', 'Structure-only prediction, both partners unseen')
    for i, (_, group) in enumerate(vectors.groupby('vector_id', sort=False)):
        group = group.sort_values('background_rank')
        color = [c.BLUE, c.TEAL][i]
        ax.plot(group.background_rank, group.delta, 'o-', color=color, lw=0.8, ms=3)
        ax.axhline(group.true_mean_pp.iloc[0], color=color, ls='--', lw=0.55)
    ax.set_xticks(sorted(vectors.background_rank.unique()))
    c.clean(ax, 'Recipe background', 'Measured response (pp)')
    ax.legend(handles=[Line2D([], [], color=c.GRAY, label='Response across backgrounds'), Line2D([], [], color=c.GRAY, ls='--', label="Pair's own mean")], loc='center left', frameon=False, fontsize=5.5)
    ax = c.panel(fig, [0.65, 0.38, 0.28, 0.48], 'b', 'Within-layer prediction improvement', title_x=0.59)
    ax.bar([0, 1], layers.within_layer_error_removed_pct, color=[c.BLUE, c.TEAL], width=0.55)
    for i, row in enumerate(layers.itertuples()):
        ax.text(i, row.within_layer_error_removed_pct + 0.12, f'{row.within_layer_error_removed_pct:.2g}%', ha='center', fontsize=6)
    ax.set_xticks([0, 1], ["Pair's own\nlevel", 'Background\nfluctuation'])
    ax.set(ylim=(0, layers.within_layer_error_removed_pct.max() * 1.25))
    c.clean(ax, ylabel='Error removed within each layer (%)')
    ax = c.panel(fig, [0.28, 0.1, 0.65, 0.15], 'c', 'Share of net total improvement', title_x=0.09)
    gains = gains.sort_values(['nominal_k', 'weighting'])
    weights = {'pooled': 'Pooled', 'equal_edge': 'Equal edge', 'equal_class_edge_balanced': 'Group/edge balanced'}
    for i, row in enumerate(gains.itertuples()):
        ax.barh(i, row.own_mean_share_pct, color=c.BLUE, height=0.65)
        ax.barh(i, 100 - row.own_mean_share_pct, left=row.own_mean_share_pct, color=c.TEAL, height=0.65)
    ax.set_yticks(range(len(gains)), [f'k={int(r.nominal_k)}; {weights[r.weighting]}' for r in gains.itertuples()], fontsize=5)
    ax.set(xlim=(0, 100), ylim=(len(gains) - 0.5, -0.5), xticks=[0, 50, 100])
    c.clean(ax, 'Net improvement (%)')
    c.save(fig, stem, "Structure-based prediction primarily recovers each pair's own response level")

def extended_reference(stem='Extended_Data_Figure_5'):
    from publication_figures.select_fig1_example import select_example
    row, cells, selection = select_example()
    if selection['eligible_pool_counts'] != {'flagship_catalyst_pair': 0, 'all_suzuki': 136, 'complete_population': 529}:
        raise ValueError('Declared example-selection population changed')
    fig = c.newfig(125)
    ax = c.panel(fig, [0.07, 0.17, 0.55, 0.66], 'a', 'Matched reference and target backgrounds')
    ax.axis('off')
    for i, smiles in enumerate(row.substrate_pair.split(' >> ')):
        c.molecule(fig, [0.09 + i * 0.25, 0.71, 0.18, 0.1], smiles)
    fig.text(0.31, 0.76, '+', ha='center', fontsize=6)
    reference = cells.loc[cells.background.eq('Reference')]
    for j, reagent in enumerate(reference.reagent):
        ax.text(0.4 + j * 0.25, 0.65, reagent, ha='center', fontsize=6)
    for i, background in enumerate(['Reference', 'Target']):
        subset = cells.loc[cells.background.eq(background)]
        yy = 0.48 - i * 0.27
        ax.text(0.22, yy, background + '\n' + subset.background_label.iloc[0], ha='right', va='center', fontsize=5.5)
        for j, value in enumerate(subset.yield_percent):
            ax.add_patch(Rectangle((0.28 + j * 0.25, yy - 0.08), 0.24, 0.16, fc='white' if i == 0 else c.LIGHT, ec=c.GRAY, lw=0.5))
            ax.text(0.4 + j * 0.25, yy, f'{value:.1f}%', ha='center', va='center', fontsize=6)
    ax = c.panel(fig, [0.75, 0.23, 0.2, 0.53], 'b', 'Predicted and measured changes', title_x=0.66)
    values = [row.direct_raw_prediction_pp, row.semantic_full_raw_prediction_pp, row.target_contrast_pp]
    for i, value in enumerate(values):
        color = [c.GOLD, c.TEAL, c.INK][i]
        ax.barh(2 - i, value, color=color, height=0.45)
        ax.text(value + (-3 if value < 0 else 3), 2 - i, f'{value:+.1f}', ha='right' if value < 0 else 'left', va='center', color=color, fontsize=6)
    ax.axvline(0, color=c.GRAY, lw=0.5)
    ax.set_yticks([2, 1, 0], ['Copy', 'Learn', 'Measured'])
    ax.set(xlim=(-60, 60), ylim=(-0.6, 2.6), xticks=[-50, 0, 50])
    c.clean(ax, 'Change (pp)')
    c.save(fig, stem, 'A representative matched-reference comparison illustrates direction and magnitude separately')
CAPTIONS = {'Figure_2': '**Figure 2 | Substitution responses depend jointly on substrate and recipe background.** **a**, Responses to PdCl₂(dtbpf) → PdCl₂(PPh₃)₂ in 412 substrate pairs measured under both Cs₂CO₃ and K₂CO₃; 145 have opposite signs (shaded quadrants; Pearson r = 0.55). Coloured, numbered points identify the two aryl bromides; the common partner is 4-methoxyphenylboronic acid. Their four responses cannot be factorized into pair and background signs. **b**, Complete response distributions for one exact substitution in each of six transformations. Substitutions have the largest substrate-pair support within their transformation, except for the prescribed Suzuki flagship. Every point is a measured pair; blue and red denote positive and negative responses. **c**, Fraction of cycle-containing substitutions with sign conflict in each of nineteen arrays, using all nonzero changes (circles) or changes of at least 20 points (diamonds). The right column gives conflicting/cycle-containing counts at both thresholds; totals are 271/283 and 143/212. Arrays without cyclic substitutions have undefined fractions. The sign test asks whether the product of pair and background signs describes the observed response directions.', 'Figure_3': '**Figure 3 | Historical relationships make the same reference measurements useful.** **a**, The schematic shows the same substrate pair in reference and target backgrounds: two measured reference wells supply either copying or learning, while the target wells are withheld. Prediction-error changes relative to history alone (no reference) for seventeen screens and the hierarchical group average (Overall). Copying and learning use the same two reference wells; screens are ordered by learned error change. Overall, copying increases error by 42.4%, whereas learning reduces it by 13.7%. The arrow marks a copied-error value beyond the plotted range. **b**, Missed or wrong directions among 21,997 unique held-out targets with measured absolute changes of at least 20 percentage points: copying makes 4,097 errors (18.6 per 100), learning 2,982 (13.6 per 100). A zero prediction counts as an error; the 486 zero copied predictions are not opposite-sign choices. **c**, The 452 training pairs for the support-selected pentafluorophenol → N-hydroxyphthalimide substitution. Red denotes copying; teal is the learned relationship at mean reference yield. **d**, Error removed by an aligned reference relative to its matched control. Grey points denote screens and teal points hierarchical group averages. Rows combine substrate-pair or reactant-role exclusion with 25% or 100% of history. Structure-only prediction with both partners unseen is a separate task (Extended Data Fig. 4).', 'Extended_Data_Figure_3': '**Extended Data Figure 3 | Calibration between libraries usually worsens the target-only baseline.** **a**, All seventeen fixed-support, scored transfers between libraries. Each line follows one direction from target-only prediction to calibrated transfer and the source-identity reassignment control. Errors are normalized to the target-only error for that direction. Calibration worsens fifteen directions (red) and improves two (teal). The source difference channel, budget of 192 and fixed target support follow the original cross-library analysis.', 'Extended_Data_Figure_4': "**Extended Data Figure 4 | Structure-based prediction primarily recovers each pair's own response level.** This structure-only task excludes both partner molecules from training and is distinct from the two-reference-well learning task in Figure 3. **a**, Two measured response vectors from the substitution with the most supported complete vectors in the three-group decomposition; the displayed pairs have the minimum and maximum own mean response within that population. Solid lines show responses across backgrounds; dashed lines show each pair's own mean. **b**, Error removed within each layer, divided by that layer's centroid-baseline error, for the three-group, group-and-edge-balanced comparison. Approximately 5.0% of pair-level error and 0.22% of background-fluctuation error are removed. **c**, Pair-level (blue) and fluctuation (teal) contributions to net total improvement for two or three groups and three weightings. Each pair's own level accounts for 91.1–96.6% of net improvement. This share uses a different denominator from panel b.", 'Extended_Data_Figure_5': '**Extended Data Figure 5 | A representative matched-reference comparison illustrates direction and magnitude separately.** **a**, Replacing PdCl₂(dtbpf) with XPhos Pd G3 gives opposite responses in the K₃PO₄ reference and Cs₂CO₃ target backgrounds of the same substrate pair. Target wells were withheld from prediction. **b**, Copying predicts −10.5 percentage points, learning +5.6 points and the measured target change is +34.4 points. Learning identifies the direction but underestimates its magnitude. The predeclared rule requires a reference change of at least 10 points, a measured target change of at least 20 points, an incorrect copied sign and a correct learned sign, plus four retained measured endpoints. The nested eligible pools contain zero flagship-catalyst, 136 Suzuki and 529 total comparisons. The selected Suzuki comparison is nearest the median absolute learned error; lexical identities resolve ties. Detailed selection and well verification are in Methods and Source Data.'}
