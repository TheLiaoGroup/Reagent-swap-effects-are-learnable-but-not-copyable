"""Flagship reversal and measured rescue decisions from the M1 outputs."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from publication_figures import common as c
RECIPE_OPTIONS = c.A / 'scripts/recipe_options/Analysis'
RESPONSE_LEARNING = c.A / 'scripts/response_learning'
EDM = RESPONSE_LEARNING / 'Existing_Data_Mining_2026-10-16/results'
E59 = RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/results'
E66 = RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results'
PAIRING = RECIPE_OPTIONS / 'round9/pairing_repair'
FLAGSHIP = 'suzuki__C23__C26'
BORONIC = 'COc1ccc(B(O)O)cc1'
QUINOLINE = 'Brc1ccc2ncccc2c1'
INDAZOLE = 'Brc1ccc2[nH]ncc2c1'
METHYL_ACTION = '0e5d6cf7a808adc5a0a4_d0_s01_r1'
AMIDE_ACTION = '4bcee624ba37f2c2cef3_d0_s01_r0'
YIELD_CMAP = LinearSegmentedColormap.from_list('measured_yield', ['#FAFCFC', '#B4D7D0', c.TEAL])

def _arrow(ax, start, end, color=c.GRAY, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=6, color=color, lw=0.7, **kwargs))

def flagship_data():
    example_path = EDM / 'focused_chemical_examples.csv'
    examples = c.frame(example_path)
    examples = examples.loc[examples.target_swap_id.eq(FLAGSHIP) & examples.sub_2_smiles.eq(BORONIC) & examples.sub_1_smiles.isin([QUINOLINE, INDAZOLE])].copy()
    if len(examples) != 2 or examples.sub_1_smiles.nunique() != 2:
        raise ValueError('The two prescribed flagship examples are not unique')
    raw_path = reader_tables.data_dir() / 'Suzuki_260806.xlsx'
    raw = reader_tables.read_excel(raw_path, usecols=['name0', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles', 'yield'])
    records = []
    for example in examples.itertuples():
        for endpoint, field in [(0, 'name00'), (1, 'name01')]:
            matched = raw.loc[raw.name0.eq(getattr(example, field))]
            if len(matched) != 1:
                raise ValueError('Flagship raw record is not unique')
            row = matched.iloc[0]
            if row.sub_1_smiles != example.sub_1_smiles or row.sub_2_smiles != BORONIC:
                raise ValueError('Flagship chemistry differs from its retained raw well')
            if not np.isclose(row['yield'], getattr(example, f'y{endpoint}'), rtol=0, atol=1e-12):
                raise ValueError('Flagship raw yield differs from the paired response')
            records.append(dict(substrate_smiles=row.sub_1_smiles, partner_smiles=row.sub_2_smiles, product_smiles=row.product_smiles, endpoint=endpoint, name0=row.name0, catalyst='PdCl2(dtbpf)' if endpoint == 0 else 'PdCl2(PPh3)2', yield_percent=100 * row['yield']))
    records = pd.DataFrame(records)
    c.source('Fig1_a_flagship_wells', records, [example_path, raw_path], 'Prescribed suzuki__C23__C26; fixed quinoline/indazole and common 4-methoxyphenylboronic acid; exact name0 joins', 'yield percent', 'Product structures are read from the retained raw records.')
    registry_path = E59 / 'full_edge_strict_swap_registry.csv'
    registry = c.frame(registry_path)
    selected = registry.loc[registry.swap_id.eq(FLAGSHIP)]
    if len(selected) != 1:
        raise ValueError('Flagship exact-swap registry is not unique')
    row = selected.iloc[0]
    corpus_path = E59 / 'full_edge_response_corpus.parquet'
    corpus = pd.read_parquet(corpus_path, filters=[('reaction_type', '=', row.reaction_type), ('swap_slot', '=', row.swap_slot), ('component_pair', '=', row.component_pair), ('background_fingerprint', '=', row.background_fingerprint)])
    if len(corpus) != int(row.n_pairs) or len(corpus) != 448:
        raise ValueError('The fixed flagship distribution lost its 448 substrate pairs')
    corpus['display_delta_pp'] = corpus.delta * row.orientation_sign
    c.source('Fig1_a_448_responses', corpus, [registry_path, corpus_path], 'All substrate pairs for prescribed exact swap suzuki__C23__C26; oriented condition C23 to C26', 'yield change pp')
    return (records, corpus)

def figure1():
    records, corpus = flagship_data()
    fig = c.newfig(height_mm=125, width_mm=180)
    main = c.panel(fig, [0.065, 0.09, 0.88, 0.84], 'a', 'One catalyst substitution')
    main.axis('off')
    for substrate, y, color in [(QUINOLINE, 0.865, c.BLUE), (INDAZOLE, 0.735, c.RED)]:
        selected = records.loc[records.substrate_smiles.eq(substrate)]
        product = selected.product_smiles.unique()
        if len(product) != 1:
            raise ValueError('Flagship product identity depends on catalyst')
        c.molecule(fig, [0.075, y - 0.035, 0.17, 0.07], substrate)
        c.molecule(fig, [0.705, y - 0.035, 0.26, 0.07], product[0])
        fig.add_artist(Line2D([0.06, 0.075], [y, y], transform=fig.transFigure, color=color, lw=1.5))
    c.molecule(fig, [0.275, 0.765, 0.17, 0.07], BORONIC)
    fig.text(0.255, 0.8, '+', ha='center', va='center', fontsize=6)
    fig.add_artist(FancyArrowPatch((0.46, 0.8), (0.69, 0.8), transform=fig.transFigure, arrowstyle='->', mutation_scale=7, color=c.INK, lw=0.8))
    fig.text(0.575, 0.89, 'PdCl₂(dtbpf) or PdCl₂(PPh₃)₂', ha='center', fontsize=6)
    fig.text(0.575, 0.862, 'Cs₂CO₃, 1,4-dioxane/H₂O', ha='center', fontsize=6)
    fig.text(0.575, 0.834, '100 °C, 12 h', ha='center', fontsize=6)
    slope = fig.add_axes([0.1, 0.105, 0.58, 0.52])
    for substrate, color, label in [(QUINOLINE, c.BLUE, 'bromoquinoline'), (INDAZOLE, c.RED, 'bromo-1H-indazole')]:
        selected = records.loc[records.substrate_smiles.eq(substrate)].sort_values('endpoint')
        yields = selected.yield_percent.to_numpy()
        slope.plot([0, 1], yields, 'o-', color=color, ms=4, lw=1.6)
        for endpoint, value in enumerate(yields):
            slope.annotate(f'{value:.2f}%', (endpoint, value), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', color=color, fontsize=6)
        slope.text(1.035, yields[1], label, ha='left', va='center', color=color, fontsize=6)
    slope.set(xlim=(-0.08, 1.08), ylim=(-2, 104), yticks=[0, 50, 100])
    slope.set_xticks([0, 1], ['PdCl₂(dtbpf)', 'PdCl₂(PPh₃)₂'])
    c.clean(slope, ylabel='Yield (%)')
    slope.spines['bottom'].set_visible(False)
    slope.tick_params(axis='x', length=0, pad=5, labelsize=6)
    distribution = fig.add_axes([0.86, 0.105, 0.07, 0.52])
    values = corpus.display_delta_pp.to_numpy()
    jitter = (np.arange(len(values)) * 0.61803398875 % 1 - 0.5) * 0.7
    distribution.scatter(jitter, values, s=2, c='#B7BFC5', alpha=0.55, lw=0)
    distribution.axhline(0, color=c.GRAY, lw=0.45)
    for substrate, color in [(QUINOLINE, c.BLUE), (INDAZOLE, c.RED)]:
        selected = corpus.loc[corpus.sub_1_smiles.eq(substrate) & corpus.sub_2_smiles.eq(BORONIC)]
        if len(selected) != 1:
            raise ValueError('Flagship highlight is not unique within its full distribution')
        distribution.scatter(0, selected.display_delta_pp.iloc[0], s=12, c=color, edgecolors='white', linewidths=0.35, zorder=4)
    distribution.set(xlim=(-0.5, 0.5), ylim=(-100, 100), yticks=[-100, 0, 100], xticks=[])
    distribution.yaxis.tick_right()
    distribution.yaxis.set_label_position('right')
    c.clean(distribution, ylabel='Yield change (pp)')
    distribution.spines[['left', 'bottom', 'top']].set_visible(False)
    distribution.text(0.5, 1.06, '448 pairs', transform=distribution.transAxes, ha='center', fontsize=6)
    main._draft_panel_letter.set_fontsize(7)
    c.save(fig, 'Figure_1', 'A catalyst substitution reverses between related substrates')

def repair_data(action, source_name):
    path = PAIRING / 'selected_pairing_repair_all_12_yields_and_conditions.csv'
    selected = c.frame(path).loc[lambda d: d.action_id.eq(action)].copy()
    if len(selected) != 12 or selected.groupby('molecular_side').size().tolist() != [6, 6]:
        raise ValueError('A fixed rescue example requires all twelve measured cells')
    c.source(source_name, selected, path, f'Prescribed action_id={action}; all before/after recipe cells', 'yield fraction')
    return selected

def learning_data():
    path = E66 / 'e66_summary.csv'
    raw = c.frame(path)
    q = raw.loc[raw.population.eq('common8') & raw.target_geometry.eq('raw_delta') & raw.stratum.eq('all') & raw.weighting.eq('equal_class_edge_balanced') & raw.learner.isin(['joint', 'calibration_only']) & raw.budget.isin(['1', '2', '4', '8'])].copy()
    if len(q) != 8 or not q.n_test_vectors.eq(24221).all():
        raise ValueError('The fixed local-learning population or budgets changed')
    q['n_vectors'] = q.budget.astype(int)
    joint = q.loc[q.learner.eq('joint')].set_index('n_vectors')
    local = q.loc[q.learner.eq('calibration_only')].set_index('n_vectors')
    comparison = pd.DataFrame({'n_vectors': [1, 2, 4, 8]})
    comparison['joint_mse_pp2'] = comparison.n_vectors.map(joint.native_vector_mse)
    comparison['local_only_mse_pp2'] = comparison.n_vectors.map(local.native_vector_mse)
    comparison['error_change_pct'] = 100 * (comparison.joint_mse_pp2 / comparison.local_only_mse_pp2 - 1)
    if f"{comparison.loc[comparison.n_vectors.eq(8), 'error_change_pct'].iloc[0]:.1f}" != '-9.7':
        raise ValueError('The eight-profile local comparison changed')
    scope_path = E66 / 'e66_scope_vectors.parquet'
    scope = pd.read_parquet(scope_path, columns=['common8', 'n_backgrounds'])
    selected = scope.loc[scope.common8]
    if len(selected) != 24221:
        raise ValueError('Local profile costs use a different query population')
    costs = []
    for n in [1, 2, 4, 8]:
        for backgrounds, count in selected.n_backgrounds.value_counts().sort_index().items():
            costs.append(dict(n_vectors=n, n_backgrounds=int(backgrounds), local_condition_cells=2 * int(backgrounds) * n, n_query_vectors=int(count)))
    costs = pd.DataFrame(costs)
    c.source('Fig4_c_local_learning', comparison, path, 'common8/raw_delta/all/equal_class_edge_balanced; identical 24221 queries; joint versus calibration_only at budgets1/2/4/8', 'percent MSE change relative to local only', 'Each plotted point compares the original fixed-budget models; none is refitted.')
    c.source('Fig4_c_measurement_cost', costs, scope_path, 'All common8 queries; distinct condition-cell cost=2*number of backgrounds*number of profiles', 'distinct condition cells', 'The x axis consists of ordered budget categories, not a cost-scaled continuous axis.')
    return (comparison, costs)

def scope_partition():
    views_path = RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv'
    options_path = RECIPE_OPTIONS / 'round29/component_options/minimum_option_distribution.csv'
    views = c.frame(views_path)
    options = c.frame(options_path).loc[lambda x: x.endpoint_corner_stratum.eq('all')]
    values = [int(options.loc[options.minimum_extra_options.eq(k), 'n_certificates'].sum()) for k in [1, 2, 3]]
    values.append(int((~views.full_graph_rescues20).sum()))
    if values != [266, 16, 3, 257] or sum(values) != len(views):
        raise ValueError('The complete shared-scope partition changed')
    partition = pd.DataFrame({'category': ['1 extra option', '2 extra options', '3 extra options', 'Unrecovered'], 'n': values})
    c.source('Fig4_e_scope_partition', partition, [views_path, options_path], 'All542 conflict views; all285 recoverable and257 unrecovered; minimum options1/2/3 partition recoverable subset', 'conflict views')
    return partition

def _amide_network(fig, cells):
    c.molecule(fig, [0.075, 0.852, 0.2, 0.072], 'CSc1ccc(N)cc1')
    c.molecule(fig, [0.355, 0.852, 0.22, 0.072], 'O=C(O)COc1ccccc1')
    fig.text(0.315, 0.888, '+', ha='center', va='center', fontsize=6)
    starting = cells.loc[cells.ternary_level.eq(0) & cells.binary_level.eq(0)].set_index('molecular_side')
    if len(starting) != 2 or set(starting.index) != {'before', 'after'}:
        raise ValueError('The amide relocation requires its two HBTU/no-HOBt starting wells')
    before, failure = (100 * starting.loc['before', 'yield_numeric'], 100 * starting.loc['after', 'yield_numeric'])
    fig.text(0.075, 0.809, f'HBTU, no HOBt: meta {before:.2f}% → para {failure:.2f}%', fontsize=6)
    after = cells.loc[cells.molecular_side.eq('after')].sort_values('panel_position')
    ax = fig.add_axes([0.125, 0.625, 0.46, 0.163])
    ax.set(xlim=(-0.7, 2.4), ylim=(-0.4, 1.7))
    ax.axis('off')
    edges = []
    for i, row in after.iterrows():
        for j, other in after.iterrows():
            if j <= i or int(row.ternary_level != other.ternary_level) + int(row.binary_level != other.binary_level) != 1:
                continue
            rescue = row.binary_level == other.binary_level == 0 and {row.ternary_level, other.ternary_level} == {0, 2}
            if not rescue:
                ax.plot([row.ternary_level, other.ternary_level], [row.binary_level, other.binary_level], color='#C6CBCF', lw=0.5, zorder=0)
            edges.append(dict(recipe0=row.recipe_id, recipe1=other.recipe_id, differing_slots=1, selected_recovery=bool(rescue)))
    _arrow(ax, (0.16, -0.04), (1.82, -0.04), c.TEAL, connectionstyle='arc3,rad=-.34')
    for row in after.itertuples():
        selected = row.binary_level == 0 and row.ternary_level in [0, 2]
        color = c.RED if row.binary_level == 0 and row.ternary_level == 0 else c.TEAL if selected else c.GRAY
        ax.add_patch(Rectangle((row.ternary_level - 0.2, row.binary_level - 0.17), 0.4, 0.34, facecolor='white', edgecolor=color, lw=1 if selected else 0.6, zorder=2))
        ax.text(row.ternary_level, row.binary_level, f'{100 * row.yield_numeric:.0f}%', ha='center', va='center', fontsize=6, color=color if selected else c.INK, zorder=3)
    for i, label in enumerate(['HBTU', 'PyBOP', 'TPTU']):
        ax.text(i, 1.5, label, ha='center', fontsize=6)
    ax.text(-0.35, 0, 'No HOBt', ha='right', va='center', fontsize=6)
    ax.text(-0.35, 1, '+ HOBt', ha='right', va='center', fontsize=6)
    c.source('Fig4_a_recipe_edges', pd.DataFrame(edges), PAIRING / 'selected_pairing_repair_all_12_yields_and_conditions.csv', 'Prescribed amide after-edit six-node grid; edge iff one measured recipe slot differs', 'edges')
    fig.text(0.075, 0.597, 'DIPEA, DMF, 25 °C, 2 h', fontsize=6)

def _suzuki_rescue(fig, cells):
    for j, side in enumerate(['before', 'after']):
        rows = cells.loc[cells.molecular_side.eq(side)].sort_values('panel_position')
        values = rows.yield_numeric.to_numpy().reshape(3, 2) * 100
        x = 0.245 + j * 0.185
        c.molecule(fig, [x - 0.005, 0.455, 0.15, 0.075], rows.reactant2_smiles.iloc[0], highlight_methyl=True, template='OB(O)c1ccccc1', halo=True)
        fig.text(x + 0.075, 0.443, '3-methyl' if j == 0 else '2-methyl', ha='center', fontsize=6)
        ax = fig.add_axes([x, 0.255, 0.15, 0.17])
        image = ax.pcolormesh(np.arange(3) - 0.5, np.arange(4) - 0.5, values, cmap=YIELD_CMAP, vmin=0, vmax=100, rasterized=False)
        ax.set(xlim=(-0.5, 1.5), ylim=(2.5, -0.5))
        ax.set_xticks([0, 1], ['Cs₂CO₃', 'K₃PO₄'], fontsize=6)
        ax.set_yticks([0, 1, 2], ['PdCl₂(dtbpf)', 'PdCl₂(PCy₃)₂', 'SPhos Pd G3'] if j == 0 else [], fontsize=6)
        ax.tick_params(length=0, pad=2)
        ax.spines[:].set_visible(False)
        for (r, k), value in np.ndenumerate(values):
            ax.text(k, r, f'{value:.0f}', ha='center', va='center', fontsize=6, color='white' if value > 72 else c.INK)
        if j == 1:
            for r, color in [(1, c.RED), (2, c.TEAL)]:
                ax.add_patch(Rectangle((0.52, r - 0.48), 0.96, 0.96, fill=False, edgecolor=color, linewidth=1.1, zorder=4))
            _arrow(ax, (1.58, 1.02), (1.58, 1.96), c.TEAL, connectionstyle='arc3,rad=-.45', clip_on=False)
    colorbar = fig.colorbar(image, cax=fig.add_axes([0.245, 0.218, 0.335, 0.007]), orientation='horizontal', ticks=[0, 50, 100])
    colorbar.solids.set_rasterized(False)
    colorbar.ax.tick_params(length=2, pad=1, labelsize=6)
    colorbar.set_label('Yield (%)', labelpad=1, fontsize=6)
    fig.text(0.245, 0.178, 'Same bromoindazole; 100 °C, 12 h', fontsize=6)
    fig.text(0.245, 0.16, '1,4-dioxane/H₂O', fontsize=6)

def figure4():
    amide = repair_data(AMIDE_ACTION, 'Fig4_a_amide_12_cells')
    methyl = repair_data(METHYL_ACTION, 'Fig4_b_methyl_12_cells')
    repair_path = PAIRING / 'all_actual_yield_repairs_with_12cells.csv'
    repairs = c.frame(repair_path)
    chosen = repairs.loc[repairs.action_id.eq(AMIDE_ACTION)]
    if len(chosen) != 1 or not bool(chosen.actual_yield_repair_5pp_gain10.iloc[0]):
        raise ValueError('The prescribed amide rescue is outside the117 measured recovery contrasts')
    counts = repairs.groupby('dataset_id').measured_cell_repair_id.nunique().rename('n_measured_repairs').reset_index()
    if counts.n_measured_repairs.sum() != 117:
        raise ValueError('The measured rescue-contrast count changed')
    c.source('Fig4_a_recovery_classes', counts, repair_path, 'All eligible repairs; distinct measured_cell_repair_id within each reaction array', 'measured contrasts')
    q, costs = learning_data()
    partition = scope_partition()
    fig = c.newfig(height_mm=170, width_mm=180)
    panel = c.panel(fig, [0.06, 0.61, 0.55, 0.33], 'a', 'Amide recipe alternatives')
    panel.axis('off')
    _amide_network(fig, amide)
    panel = c.panel(fig, [0.06, 0.205, 0.55, 0.315], 'b', 'Suzuki recipe alternatives')
    panel.axis('off')
    _suzuki_rescue(fig, methyl)
    ax = c.panel(fig, [0.73, 0.675, 0.24, 0.235], 'c', 'History beyond local\nprofiles', title_x=0.705)
    x = np.arange(len(q))
    ax.plot(x, q.error_change_pct, 'o-', color=c.TEAL, lw=1.15, ms=3.5)
    ax.axhline(0, color=c.GRAY, lw=0.6)
    labels = []
    for n in q.n_vectors:
        selection = costs.loc[costs.n_vectors.eq(n), 'local_condition_cells']
        labels.append(f'{n}\n{int(selection.min())}–{int(selection.max())}')
    ax.set_xticks(x, labels, fontsize=6)
    ax.set(xlim=(-0.1, 3.2), ylim=(-52, 4), yticks=[-50, -25, 0])
    ax.text(3.05, q.error_change_pct.iloc[-1] - 4, f'{q.error_change_pct.iloc[-1]:.1f}%', ha='right', va='top', fontsize=6, color=c.TEAL)
    ax.text(0.97, 0.95, 'Local-only baseline', transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=c.GRAY)
    c.clean(ax, 'Local profiles\n(endpoint wells)', 'Error change from\nadding history (%)')
    path = RESPONSE_LEARNING / 'Local_Information_Content_E67/results/e67_decomposition_classes_20.csv'
    content = c.frame(path).loc[lambda d: d.stratum.eq('all') & d.reaction_type.isin(['buchwald-hartwig', 'ullmann'])].copy()
    if len(content) != 2:
        raise ValueError('The fixed BH/Ullmann local-information comparison is missing')
    c.source('Fig4_d_local_information', content, path, 'all stratum; Buchwald-Hartwig and Ullmann', 'MSE change pp2', 'The Ullmann fluctuation contribution is near zero; no directional conclusion is drawn.')
    ax = c.panel(fig, [0.79, 0.25, 0.18, 0.22], 'd', 'Local information\ncomponents', title_x=0.705)
    ax._draft_panel_title.set_y(0.52)
    ax._draft_panel_letter.set_y(0.52)
    for i, reaction in enumerate(['buchwald-hartwig', 'ullmann']):
        row = content.loc[content.reaction_type.eq(reaction)].iloc[0]
        for j, (column, color) in enumerate([('local_level_content', c.BLUE), ('local_modulation_content', c.TEAL)]):
            value, y = (float(row[column]), i * 3 + j)
            ax.barh(y, value, color=color, height=0.65)
            ax.text(value + (-0.3 if value < -1 else 0.25), y, '≈0' if reaction == 'ullmann' and j == 1 else f'{value:+.1f}', ha='right' if value < -1 else 'left', va='center', fontsize=6)
    ax.axvline(0, color=c.GRAY, lw=0.5)
    ax.set_yticks([0.5, 3.5], ['Buchwald–\nHartwig', 'Ullmann'], fontsize=6)
    ax.set(xlim=(-18, 7), ylim=(5, -1))
    c.clean(ax, 'Error change (pp²)')
    ax.text(-0.42, 1.08, 'Level', color=c.BLUE, fontsize=6, transform=ax.transAxes)
    ax.text(0.52, 1.08, 'Fluctuation', color=c.TEAL, fontsize=6, transform=ax.transAxes)
    ax = c.panel(fig, [0.095, 0.091, 0.52, 0.023], 'e', 'Shared-scope conflicts')
    colors = [c.TEAL, c.BLUE, '#B6862C', '#C6CBD0']
    left = 0
    for count, color in zip(partition.n, colors):
        ax.barh(0, count, left=left, height=0.65, color=color, edgecolor='white', linewidth=0.3)
        left += count
    ax.set(xlim=(0, int(partition.n.sum())), ylim=(-0.6, 0.6))
    ax.axis('off')
    ax.text(0, 1.02, 'Recoverable: 285/542', transform=ax.transAxes, fontsize=6, va='bottom')
    ax.text(1, 1.02, 'Unrecovered: 257/542', transform=ax.transAxes, ha='right', fontsize=6, va='bottom')
    for i, (row, color) in enumerate(zip(partition.itertuples(), colors)):
        xx, yy = (0.095 + i % 2 * 0.27, 0.061 - i // 2 * 0.022)
        fig.add_artist(Rectangle((xx, yy), 0.008, 0.009, transform=fig.transFigure, fc=color, lw=0))
        fig.text(xx + 0.013, yy + 0.0045, f'{row.category}: {row.n}', va='center', fontsize=6)
    for _, panel, _ in fig._m1_panels:
        panel._draft_panel_letter.set_fontsize(7)
    c.save(fig, 'Figure_4', 'Local measurements and measured alternatives guide the next decision')
CAPTIONS = {'Figure_1': '**Figure 1 | A catalyst substitution reverses between related substrates.** **a**, The same Suzuki catalyst substitution raises the bromoquinoline yield from 4.98% to 93.36% but lowers the bromo-1H-indazole yield from 83.86% to 11.55%, with the common boronic acid and fixed background shown. The crossing lines show measured yields. The adjacent point strip contains all 448 measured substrate-pair responses for this exact substitution and background; coloured points identify the two examples.', 'Figure_4': '**Figure 4 | Local measurements and measured alternatives guide the next decision.** **a**, Moving the methylthio group from meta to para lowers the HBTU/no-HOBt yield from 91.34% to 0%. Six para-methylthioaniline/phenoxyacetic-acid recipes show recovery to 92.79% with TPTU without HOBt. DIPEA, DMF, 25 °C and 2 h are fixed; lines join recipes differing in one slot. **b**, Twelve Suzuki yields before and after a boronic-acid methyl relocation. At fixed K₃PO₄, the PdCl₂(PCy₃)₂ yield falls from 47.62% to 0%; SPhos Pd G3 gives 92.48% after relocation. Halos identify methyl positions; red and teal outlines mark failure and recovery. **c**, Adding history to local profiles, relative to profiles alone, for the same 24,221 queries. The 1, 2, 4 and 8 profiles have endpoint-well counts on the second line. At eight profiles (48–240 endpoint wells here), history lowers error by 9.7%. Its additional benefit decreases with more local data. **d**, Local level and fluctuation contributions under the fixed update rule; Ullmann fluctuation changes error negligibly (+0.09 pp²). **e**, Measured alternatives recover 285 of 542 shared-scope conflicts; 257 remain unrecovered. One, two or three extra component options suffice in 266, 16 and 3 recoverable views. The amide example belongs to 117 measured rescue contrasts across six transformations.'}
