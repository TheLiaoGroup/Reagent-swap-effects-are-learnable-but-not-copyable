"""Build four M1 figure drafts from the M1 pipeline outputs only."""
from __future__ import annotations
import io
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
S = Path(__file__).resolve().parent
A = S.parent
OUT = A / 'out' / 'storyboard'
REFERENCE_LEARNING = S / 'reference_learning' / 'Analysis'
RECIPE_OPTIONS = S / 'recipe_options' / 'Analysis'
RESPONSE_LEARNING = S / 'response_learning'
EDM = RESPONSE_LEARNING / 'Existing_Data_Mining_2026-10-16'
INK, GRAY, TEAL, RED, BLUE, GOLD = ('#20242a', '#87919b', '#167e72', '#ba4852', '#337ca8', '#bc8c29')
YIELD_CMAP = LinearSegmentedColormap.from_list('yield', ['#fffefe', '#c5e0d8', '#277a6b'])
TITLES = ['A reagent swap can reverse its effect between related substrates', 'Substrate and recipe jointly determine the effect of a reagent swap', 'Learning turns the same reference measurements into useful predictions', 'Local response measurements complement historical swap knowledge']
RECORDS = []

def select_one(frame, **keys):
    selected = frame
    for name, value in keys.items():
        selected = selected.loc[selected[name].eq(value)]
    if len(selected) != 1:
        raise ValueError(f'Expected one row for {keys}, found {len(selected)}')
    return selected.iloc[0]

def display_yield(value):
    return str(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def sign_counts(signs, threshold, exclude_finkelstein=True):
    keys = dict(population='all', panel='observed_sparse', objective='count', threshold_pp=threshold)
    total = select_one(signs, stratum='ALL', **keys).copy()
    if exclude_finkelstein:
        excluded = select_one(signs, stratum='finkelstein', **keys)
        for name in ('n_conflicting_edges', 'n_cycle_edges'):
            total[name] -= excluded[name]
        if not 0 <= total.n_conflicting_edges <= total.n_cycle_edges:
            raise ValueError('Invalid sign counts after excluding Finkelstein')
    return total

def conflict_partition(views, options):
    flags = views.full_graph_rescues20
    if len(views) == 0 or flags.isna().any() or (not flags.isin([True, False]).all()):
        raise ValueError('Expected nonempty Boolean recoverability flags')
    recovered = int(flags.sum())
    counts = []
    for k in (1, 2, 3):
        selected = options.loc[options.minimum_extra_options.eq(k), 'n_certificates']
        if len(selected) > 1:
            raise ValueError('Duplicate option counts')
        counts.append(int(selected.iloc[0]) if len(selected) else 0)
    if sum(counts) != recovered:
        raise ValueError('Option counts must partition the recoverable subset')
    counts.append(len(views) - recovered)
    return counts

def record(fig, panel, label, value, unit, source, selector='', **extra):
    RECORDS.append(dict(figure=fig, panel=panel, label=label, value=value, unit=unit, source=str(source.relative_to(A)).replace('\\', '/'), selector=selector, **extra))

def frame(path):
    result = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    if len(result) == 0:
        raise ValueError(f'Empty source: {path}')
    return result

def load_data(verify_pins=True):
    if verify_pins:
        pins = frame(A / 'out/m1_numbers.csv')
        if len(pins) < 67 or not pins.matches.eq(True).all():
            raise ValueError('Re-run m1_numbers.py: every pinned number must match')
    path = EDM / 'results/focused_chemical_examples.csv'
    ex = frame(path)
    ex = ex.loc[ex.target_swap_id.eq('suzuki__C23__C26') & ex.sub_2_smiles.eq('COc1ccc(B(O)O)cc1')]
    flip = []
    for smi in ('Brc1ccc2ncccc2c1', 'Brc1ccc2[nH]ncc2c1'):
        r = select_one(ex, sub_1_smiles=smi)
        flip.append(r)
        for name in ('y0', 'y1'):
            record(1, 'a', f'{smi} {name}', 100 * r[name], 'yield %', path, f'suzuki__C23__C26; sub_1={smi}; sub_2={r.sub_2_smiles}', observation_id=r['observation_id0' if name == 'y0' else 'observation_id1'])
    path = RECIPE_OPTIONS / 'round9/pairing_repair/selected_pairing_repair_all_12_yields_and_conditions.csv'
    repairs = frame(path)
    methyl = repairs.loc[repairs.action_id.eq('0e5d6cf7a808adc5a0a4_d0_s01_r1')].copy()
    amide = repairs.loc[repairs.action_id.eq('4bcee624ba37f2c2cef3_d0_s01_r0')].copy()
    for fig, panel, block in ((1, 'b', methyl), (4, 'c', amide)):
        if len(block) != 12:
            raise ValueError('A recovery example requires all 12 measured cells')
        for _, r in block.iterrows():
            record(fig, panel, f'{r.molecular_side}: {r.ternary_component}; {r.binary_component}', 100 * r.yield_numeric, 'yield %', path, f'action_id={r.action_id}; side={r.molecular_side}; position={r.panel_position}', reactant1_smiles=r.reactant1_smiles, reactant2_smiles=r.reactant2_smiles, recipe=r.full_recipe)
    path = EDM / 'results/matched_response_profiles.parquet'
    profiles = frame(path)
    witness = profiles.loc[profiles.reaction_type.eq('suzuki') & profiles.component_pair.eq('CMP0092|CMP0097') & profiles.sub_1_smiles.isin(['Brc1ccc2[nH]ncc2c1', 'COc1cccnc1Br']) & profiles.sub_2_smiles.eq('COc1ccc(B(O)O)cc1')].copy()
    if len(witness) != 4:
        raise ValueError('Sign witness must contain four observed cells')
    for _, r in witness.iterrows():
        if not np.isclose(r.high_yield_pp - r.low_yield_pp, r.delta) or abs(r.delta) < 20:
            raise ValueError('Incorrect sign orientation or weak witness')
        for name, unit in (('delta', 'yield pp'), ('low_yield_pp', 'yield %'), ('high_yield_pp', 'yield %')):
            record(2, 'a', f'{r.sub_1_smiles}: {name}', r[name], unit, path, f'CMP0092|CMP0097; {r.substrate_pair_key}; {r.background_fingerprint}')
    path = EDM / 'results/sparse_sign_feasibility_summary.csv'
    signs = frame(path)
    sign_rows = []
    for threshold in (0, 20):
        r = sign_counts(signs, threshold)
        sign_rows.append(r)
        for name in ('n_conflicting_edges', 'n_cycle_edges'):
            record(2, 'b', f'threshold {threshold}: {name}', r[name], 'swaps', path, f'all; observed_sparse; count; ALL minus finkelstein; threshold_pp={threshold}')
    path = REFERENCE_LEARNING / 'tables/cross_library_source_channels_headline.csv'
    cross = select_one(frame(path), domain='b24_fixed', relation='cross_family', macro='direction_macro', channel='d', budget=192)
    for name in ('target_mean_mse', 'real_mse', 'test_permuted_exact_mse'):
        record(2, 'c', name, 10000.0 * cross[name], 'MSE pp2', path, 'b24_fixed; cross_family; direction_macro; d; budget=192')
    path = REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv'
    copy = frame(path)
    copy = copy.loc[copy.level.eq('headline')].set_index('model')
    if len(copy) != 4 or copy.n_edges.ne(655).any() or copy.target_score_count.ne(242766).any():
        raise ValueError('Copy/learn panel must use the same strict population')
    for model, r in copy.iterrows():
        record(3, 'a', model, r.mse_pp2, 'MSE pp2', path, f'level=headline; model={model}')
    path = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_headline.csv'
    spec = frame(path)
    spec = spec.loc[spec.domain.eq('strict_641') & spec.scope.eq('all_panels_complete') & spec.weighting.eq('context_balanced') & spec.macro.eq('family_macro')].copy()
    if len(spec) != 6:
        raise ValueError('Expected six split/history settings')
    for _, r in spec.iterrows():
        record(3, 'b', f'{r.split}; history fraction {r.fraction}', 100 * r.other_recovery_vs_control, 'MSE reduction %', path, 'strict_641; all_panels_complete; context_balanced; family_macro')
    path = RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results/e63_paired_component_summary.csv'
    components = frame(path)
    components = components.loc[components.comparison.eq('centroid_minus_codebook_continuous')].copy()
    if len(components) != 6:
        raise ValueError('Expected six component decompositions')
    components['own_mean_share'] = 100 * components.offset_difference_pp2 / components.mse_difference_pp2
    for _, r in components.iterrows():
        record(3, 'c', f'k={r.nominal_k}; {r.weighting}', r.own_mean_share, 'net gain %', path, 'centroid_minus_codebook_continuous; offset_difference_pp2/mse_difference_pp2')
    baseline_path = RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results/e63_summary.csv'
    baseline = select_one(frame(baseline_path), model='centroid', nominal_k=3, weighting='equal_class_edge_balanced')
    gain = select_one(components, nominal_k=3, weighting='equal_class_edge_balanced')
    modulation_baseline = baseline.amplitude_mse_pp2 + baseline.alignment_mse_pp2
    modulation_gain = gain.amplitude_difference_pp2 + gain.alignment_difference_pp2
    modulation_pct = 100 * modulation_gain / modulation_baseline
    record(3, 'c', 'modulation baseline error', modulation_baseline, 'MSE pp2', baseline_path, 'centroid; k=3; equal_class_edge_balanced; amplitude_mse_pp2+alignment_mse_pp2')
    record(3, 'c', 'modulation error reduction', modulation_gain, 'MSE pp2', path, 'centroid_minus_codebook_continuous; k=3; equal_class_edge_balanced; amplitude_difference_pp2+alignment_difference_pp2')
    record(3, 'c', 'centred-modulation error removed', modulation_pct, 'baseline modulation error %', path, '100 * modulation error reduction / modulation baseline error; baseline source=e63_summary.csv')
    path = RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results/e66_summary.csv'
    local = frame(path)
    local = local.loc[local.population.eq('common8') & local.target_geometry.eq('raw_delta') & local.stratum.eq('all') & local.weighting.eq('equal_class_edge_balanced') & local.budget.astype(str).eq('8')]
    local = local.set_index('learner')
    for learner in ('calibration_only', 'frozen_family', 'joint'):
        if learner not in local.index or local.loc[learner, 'n_test_vectors'] != 24221:
            raise ValueError('Local/history comparison must share all 24,221 queries')
        record(4, 'a', learner, local.loc[learner, 'native_vector_mse'], 'MSE pp2', path, 'common8; raw_delta; all; equal_class_edge_balanced; budget=8')
    path = RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv'
    conf = frame(path)
    opt_path = RECIPE_OPTIONS / 'round29/component_options/minimum_option_distribution.csv'
    opt = frame(opt_path)
    opt = opt.loc[opt.endpoint_corner_stratum.eq('all')]
    partition = conflict_partition(conf, opt)
    if partition != [266, 16, 3, 257] or len(conf) != 542:
        raise ValueError('Shared-scope denominator changed')
    for k, n in zip((1, 2, 3, 'unrecoverable'), partition):
        record(4, 'b', f'extra options: {k}', n, 'conflict views', opt_path if k != 'unrecoverable' else path, 'endpoint_corner_stratum=all' if k != 'unrecoverable' else 'full_graph_rescues20=False')
    return dict(flip=flip, methyl=methyl, amide=amide, witness=witness, signs=sign_rows, cross=cross, copy=copy, spec=spec, components=components, modulation_pct=modulation_pct, local=local, partition=partition)

def new_figure(number):
    fig = plt.figure(figsize=(11.4, 8.0), facecolor='white')
    fig.text(0.045, 0.96, f'Figure {number} | {TITLES[number - 1]}', fontsize=14, fontweight='bold', va='top', color=INK)
    fig.text(0.045, 0.025, 'M1 | figure storyboard | measured data and held-out predictions', fontsize=8, color=GRAY)
    fig.text(0.955, 0.025, str(number), ha='right', fontsize=8, color=GRAY)
    return fig

def label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=13, fontweight='bold', va='top', color=INK)
    fig.text(x + 0.025, y - 0.001, title, fontsize=11, fontweight='bold', va='top', color=INK)

def molecule(fig, rect, smiles, highlight_methyl=False, template=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid source SMILES: {smiles}')
    rdDepictor.Compute2DCoords(mol)
    if template:
        core = Chem.MolFromSmiles(template)
        rdDepictor.Compute2DCoords(core)
        rdDepictor.GenerateDepictionMatching2DStructure(mol, core)
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 280)
    opts = drawer.drawOptions()
    opts.fixedBondLength = 35
    opts.baseFontSize = 0.65
    opts.padding = 0.12
    opts.useBWAtomPalette()
    atoms = [a.GetIdx() for a in mol.GetAtoms() if highlight_methyl and a.GetAtomicNum() == 6 and (a.GetDegree() == 1) and (a.GetTotalNumHs() == 3)]
    drawer.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors={i: (0.91, 0.7, 0.32) for i in atoms})
    drawer.FinishDrawing()
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())))
    ax.axis('off')

def clean(ax, xlabel='', ylabel=''):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c4c9ce')
    ax.tick_params(labelsize=9, color=GRAY)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

def table_heatmap(fig, rect, values, row_names, col_names, highlight=()):
    ax = fig.add_axes(rect)
    ax.imshow(values, vmin=0, vmax=100, cmap=YIELD_CMAP, aspect='auto')
    ax.set_xticks(range(len(col_names)), col_names, fontsize=9)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=9)
    ax.tick_params(length=0, pad=8)
    ax.spines[:].set_visible(False)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, display_yield(value), ha='center', va='center', fontsize=11, color='white' if value > 72 else INK, fontweight='bold' if (i, j) in highlight else 'normal')
    for i, j in highlight:
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=GOLD, lw=2))
    return ax

def figure1(d):
    fig = new_figure(1)
    label(fig, 0.045, 0.897, 'a', 'One catalyst swap, opposite yield changes')
    fig.text(0.32, 0.842, 'PdCl2(dtbpf)  ->  PdCl2(PPh3)2', fontsize=12, fontweight='bold')
    fig.text(0.32, 0.807, '5 mol% Pd; Cs2CO3, 3 equiv; 100 C, 12 h', fontsize=9, color=GRAY)
    molecule(fig, [0.04, 0.63, 0.21, 0.17], d['flip'][0].sub_1_smiles)
    molecule(fig, [0.04, 0.44, 0.21, 0.17], d['flip'][1].sub_1_smiles)
    for y, r in zip((0.7, 0.51), d['flip']):
        delta = 100 * (r.y1 - r.y0)
        fig.text(0.29, y, f'{100 * r.y0:.2f}%', fontsize=20, va='center')
        fig.text(0.43, y, '->', fontsize=18, va='center', color=GRAY)
        fig.text(0.5, y, f'{100 * r.y1:.2f}%', fontsize=20, va='center', color=TEAL if delta > 0 else RED)
        fig.text(0.69, y, f'{delta:+.2f} pp', fontsize=14, va='center', color=TEAL if delta > 0 else RED)
    molecule(fig, [0.8, 0.61, 0.16, 0.15], d['flip'][0].sub_2_smiles)
    fig.text(0.88, 0.59, 'Same boronic acid', ha='center', fontsize=9, color=GRAY)
    fig.text(0.3, 0.438, 'Same solvent: 1,4-dioxane/water (4:1)', fontsize=9, color=GRAY)
    label(fig, 0.045, 0.386, 'b', 'Moving one methyl group changes which catalyst rescues the coupling')
    for side, x in (('before', 0.31), ('after', 0.69)):
        block = d['methyl'].loc[d['methyl'].molecular_side.eq(side)].sort_values('panel_position')
        molecule(fig, [x, 0.244, 0.22, 0.095], block.reactant2_smiles.iloc[0], True, template='OB(O)c1ccccc1')
        fig.text(x + 0.11, 0.241, '3-methyl' if side == 'before' else '2-methyl', ha='center', fontsize=9)
        table_heatmap(fig, [x, 0.087, 0.24, 0.135], 100 * block.yield_numeric.to_numpy().reshape(3, 2), ['PdCl2(dtbpf)', 'PdCl2(PCy3)2', 'SPhos Pd G3'] if side == 'before' else ['', '', ''], ['Cs2CO3', 'K3PO4'], [(1, 1), (2, 1)])
    molecule(fig, [0.045, 0.15, 0.14, 0.12], d['methyl'].reactant1_smiles.iloc[0])
    fig.text(0.115, 0.125, 'Same electrophile', ha='center', fontsize=8, color=GRAY)
    fig.text(0.33, 0.043, 'Yield (%) | each base, 3 equiv; catalyst, 5 mol%', fontsize=8, color=GRAY)
    return fig

def figure2(d):
    fig = new_figure(2)
    label(fig, 0.045, 0.897, 'a', 'A background flip cannot reconcile these preferences')
    fig.text(0.065, 0.84, 'PdCl2(dtbpf) -> PdCl2(PPh3)2', fontsize=11, fontweight='bold')
    fig.text(0.065, 0.807, 'Same 4-methoxyphenylboronic acid; 5 mol% catalyst', fontsize=9, color=GRAY)
    fig.text(0.31, 0.753, 'Cs2CO3', ha='center', fontsize=11)
    fig.text(0.49, 0.753, 'K2CO3', ha='center', fontsize=11)
    signs = []
    for i, smi in enumerate(('Brc1ccc2[nH]ncc2c1', 'COc1cccnc1Br')):
        y = 0.6 - i * 0.23
        molecule(fig, [0.04, y - 0.035, 0.17, 0.15], smi)
        for j, component in enumerate(('CMP0333@3', 'CMP0334@3')):
            q = d['witness'].loc[d['witness'].sub_1_smiles.eq(smi) & d['witness'].background_fingerprint.str.contains(component, regex=False)]
            r = select_one(q)
            signs.append(np.sign(r.delta))
            x = 0.31 + j * 0.18
            fig.text(x, y + 0.035, f'{r.delta:+.2f}', ha='center', fontsize=19, color=TEAL if r.delta > 0 else RED, fontweight='bold')
            fig.text(x, y - 0.005, f'{r.low_yield_pp:.2f} -> {r.high_yield_pp:.2f}%', ha='center', fontsize=9, color=GRAY)
    if np.prod(signs) != -1:
        raise ValueError('The plotted witness must contradict row-sign x column-sign factorization')
    fig.text(0.07, 0.22, 'Three negative responses, one positive response', fontsize=11, fontweight='bold')
    fig.text(0.07, 0.175, 'No substrate sign x background sign can fit all four.\nAll four changes exceed 20 yield percentage points.', fontsize=10, linespacing=1.6)
    label(fig, 0.635, 0.897, 'b', 'Strict sign conflicts persist')
    ax = fig.add_axes([0.68, 0.63, 0.25, 0.16])
    for i, r in enumerate(d['signs']):
        ax.barh(i, r.n_cycle_edges, color='#e3e7e9', height=0.55)
        ax.barh(i, r.n_conflicting_edges, color=RED, height=0.55)
        ax.text(r.n_cycle_edges + 7, i, f'{int(r.n_conflicting_edges)}/{int(r.n_cycle_edges)}', va='center', fontsize=10)
    ax.set_yticks([0, 1], ['Nonzero', '|change| >=20 pp'], fontsize=9)
    ax.set_xlim(0, 365)
    ax.invert_yaxis()
    clean(ax, xlabel='Swaps with an observed cycle')
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    label(fig, 0.635, 0.49, 'c', 'Transfer loses to target-only\ncalibration and reassignment')
    cross = d['cross']
    vals = [10000.0 * cross.target_mean_mse, 10000.0 * cross.real_mse, 10000.0 * cross.test_permuted_exact_mse]
    ax = fig.add_axes([0.79, 0.25, 0.15, 0.16])
    ax.barh([0, 1, 2], vals, color=[TEAL, RED, GRAY], height=0.55)
    ax.set_yticks([0, 1, 2], ['Target-only', 'Calibrated transfer', 'TEST identities\nreassigned'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    clean(ax, xlabel='MSE (pp2)')
    for i, v in enumerate(vals):
        ax.text(v + 2, i, f'{v:.2f}', va='center', fontsize=9)
    fig.text(0.66, 0.13, f'Transfer penalty vs target-only: +{vals[1] - vals[0]:.3f} pp2\nCorrect correspondence penalty: +{vals[1] - vals[2]:.3f} pp2', fontsize=9, linespacing=1.6)
    fig.text(0.66, 0.067, f'{int(cross.n_scored_directions)} cross-array directions; budget <=192 coordinates', fontsize=8, color=GRAY)
    return fig

def figure3(d):
    fig = new_figure(3)
    label(fig, 0.045, 0.897, 'a', 'Identical targets and two-well references: copy versus learn')
    order = ['marginal', 'direct_raw', 'offset_raw', 'semantic_full_raw']
    vals = d['copy'].loc[order, 'mse_pp2'].to_numpy()
    ax = fig.add_axes([0.18, 0.53, 0.34, 0.29])
    ax.barh(range(4), vals, color=[GRAY, RED, '#d39399', TEAL], height=0.6)
    ax.set_yticks(range(4), ['Training marginal', 'Direct copy', 'Copy + offset', 'Learned relation'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 290)
    clean(ax, xlabel='Held-out error, MSE (pp2)')
    for i, v in enumerate(vals):
        suffix = '' if i == 0 else f' ({100 * (v / vals[0] - 1):+.1f}%)'
        ax.text(v + 3, i, f'{v:.2f}{suffix}', va='center', fontsize=9)
    fig.text(0.08, 0.426, "Same 655 contrasts in 17 screens; 14 screen-group macro-average\nReference: the target pair's two yields in another background", fontsize=9, linespacing=1.6, color=GRAY)
    label(fig, 0.59, 0.897, 'b', 'The same swap beats\na matched different swap')
    ax = fig.add_axes([0.71, 0.56, 0.23, 0.23])
    for k, (fraction, color) in enumerate(((0.25, BLUE), (1.0, TEAL))):
        data = [select_one(d['spec'], split=split, fraction=fraction) for split in ('pair', 'role1', 'role2')]
        vals2 = [100 * r.other_recovery_vs_control for r in data]
        ax.scatter(vals2, np.arange(3) + (k - 0.5) * 0.13, color=color, s=45, label='25% history' if fraction == 0.25 else 'Full history')
    ax.set_yticks(range(3), ['Pair held out', 'Role 1 held out', 'Role 2 held out'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 13)
    clean(ax, xlabel='Error reduction (%)')
    ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(-0.02, -0.48), ncol=2)
    label(fig, 0.045, 0.363, 'c', "Structure mainly predicts each pair's own response level")
    ax = fig.add_axes([0.2, 0.105, 0.33, 0.18])
    comp = d['components'].sort_values(['nominal_k', 'weighting'])
    values = comp.own_mean_share.to_numpy()
    ax.barh(range(6), 100, color='#e5e9eb', height=0.65)
    ax.barh(range(6), values, color=BLUE, height=0.65)
    names = {'pooled': 'Pooled', 'equal_edge': 'Swap-balanced', 'equal_class_edge_balanced': 'Array-balanced'}
    ax.set_yticks(range(6), [f'k={r.nominal_k}: {names[r.weighting]}' for _, r in comp.iterrows()], fontsize=8)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    clean(ax, xlabel='Share of net prediction gain (%)')
    for i, v in enumerate(values):
        ax.text(v - 1.2, i, f'{v:.1f}', va='center', ha='right', fontsize=8, color='white')
    fig.text(0.6, 0.29, f'{values.min():.1f}-{values.max():.1f}%', fontsize=24, color=BLUE, fontweight='bold')
    fig.text(0.6, 0.237, "Gain from each substrate pair's own\nmean response across recipe backgrounds", fontsize=10, linespacing=1.5)
    fig.text(0.6, 0.155, f"{d['modulation_pct']:.2f}%", fontsize=22, color=GRAY, fontweight='bold')
    fig.text(0.6, 0.1, 'Centred-modulation error removed\n(k=3, array-balanced; different denominator)', fontsize=9, linespacing=1.5, color=GRAY)
    return fig

def figure4(d):
    fig = new_figure(4)
    label(fig, 0.045, 0.897, 'a', 'History and local profiles work best together')
    learners = ['calibration_only', 'frozen_family', 'joint']
    vals = d['local'].loc[learners, 'native_vector_mse'].to_numpy()
    ax = fig.add_axes([0.205, 0.635, 0.28, 0.17])
    ax.barh(range(3), vals, color=[GRAY, BLUE, TEAL], height=0.6)
    ax.set_yticks(range(3), ['8 local profiles', 'History only', 'History + 8 local'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 275)
    clean(ax, xlabel='Held-out error, MSE (pp2)')
    for i, v in enumerate(vals):
        ax.text(v + 3, i, f'{v:.2f}', va='center', fontsize=10)
    support = d['local'].loc['joint']
    fig.text(0.07, 0.51, f'{int(support.n_test_vectors):,} queries; {int(support.n_test_classes)} arrays\nEach local profile spans all B backgrounds: 2B wells\n8 profiles = 8B scalar responses = 16B wells', fontsize=9, color=GRAY, linespacing=1.5)
    recovered = sum(d['partition'][:3])
    unrecovered = d['partition'][3]
    label(fig, 0.58, 0.897, 'b', f'Measured alternatives resolve\n{recovered} of {recovered + unrecovered} shared-scope conflicts')
    ax = fig.add_axes([0.61, 0.7, 0.33, 0.07])
    ax.barh(0, recovered, color=TEAL)
    ax.barh(0, unrecovered, left=recovered, color='#c8cdd2')
    ax.text(recovered / 2, 0, f'Recovered\n{recovered}', ha='center', va='center', fontsize=11, color='white')
    ax.text(recovered + unrecovered / 2, 0, f'Unrecovered\n{unrecovered}', ha='center', va='center', fontsize=11)
    ax.set_xlim(0, recovered + unrecovered)
    ax.axis('off')
    fig.text(0.61, 0.81, f'{recovered + unrecovered} shared-scope conflict views', fontsize=11, fontweight='bold')
    fig.text(0.61, 0.63, f'Within the {recovered} recoverable views only:', fontsize=10)
    fig.text(0.61, 0.58, f"1 extra option: {d['partition'][0]}    2: {d['partition'][1]}    3: {d['partition'][2]}", fontsize=10)
    fig.text(0.61, 0.53, 'Retrospective; recovery within tested recipes', fontsize=9, color=GRAY)
    label(fig, 0.045, 0.438, 'c', 'An alternative activator restores yield after a molecular edit')
    am = d['amide']
    for side, x in (('before', 0.3), ('after', 0.65)):
        block = am.loc[am.molecular_side.eq(side)]
        molecule(fig, [x, 0.266, 0.2, 0.12], block.reactant1_smiles.iloc[0], template='Nc1ccccc1')
        fig.text(x + 0.1, 0.257, '3-SMe aniline' if side == 'before' else '4-SMe aniline', ha='center', fontsize=10)
        ys = []
        for activator in ('HBTU', 'TPTU'):
            q = block.loc[block.full_recipe.str.contains(f'Reagent1={activator} ', regex=False) & block.full_recipe.str.contains('Reagent2=none', regex=False)]
            ys.append(100 * select_one(q).yield_numeric)
        table_heatmap(fig, [x, 0.11, 0.2, 0.11], np.array(ys).reshape(2, 1), ['HBTU', 'TPTU'] if side == 'before' else ['', ''], ['Yield (%)'], [(0, 0), (1, 0)])
    molecule(fig, [0.05, 0.21, 0.16, 0.13], am.reactant2_smiles.iloc[0])
    fig.text(0.13, 0.18, 'Same acid', fontsize=9, ha='center', color=GRAY)
    fig.text(0.29, 0.065, '117 distinct failure-to-recovery comparisons; 50 molecular edits; 6 arrays', fontsize=9)
    return fig
