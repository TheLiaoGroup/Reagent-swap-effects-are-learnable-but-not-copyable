"""Generate submission figures and panel source data using only M1 outputs.

Run through guard.py. The existing storyboard supplies the approved panel
selection; this module owns the final vector artwork and source workbooks."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import copy
import json
import re
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pymupdf
import m1_storyboard as sb
from m1_chemistry import Chemistry
from m1_figure_text import numeric_minus
from m1_reader_labels import reader_table, reader_text
A = Path(__file__).resolve().parents[1]
M1 = A.parent
OUT = A / 'out/final_figures'
FIG = M1 / 'Submission/02_Figures'
DATA = M1 / 'Submission/04_Source_Data'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
WIDTH_MM = 180
WIDTH_PT = WIDTH_MM / 25.4 * 72
HEIGHT_PT = WIDTH_PT * 8 / 11.4
PLACEMENTS = []
TABLES = []
QA = []
CHEM = None
TITLES = list(sb.TITLES)

def frame(path):
    return sb.frame(path)

def add_table(name, data, source, selector, unit, note=''):
    if len(data) == 0:
        raise ValueError(f'No rows selected for {name}')
    data = reader_table(data).reset_index(drop=True)
    selector, note = (reader_text(selector), reader_text(note))
    if 'source_absolute_path' in data:
        derived = reader_tables.data_dir().resolve()
        pointers = []
        for value in data.source_absolute_path:
            path = Path(value).resolve()
            if not Path(value).is_absolute() or not path.is_relative_to(derived) or path.parent != derived or (not reader_tables.is_source(path)) or (not reader_tables.source_is_file(path)) or ('source_file' in data and data.loc[data.source_absolute_path.eq(value), 'source_file'].ne(path.name).any()):
                raise ValueError(f'Source pointer outside the generated raw corpus: {value}')
            pointers.append((Path('data/raw_zero_rule/01_原始数据') / path.relative_to(derived)).as_posix())
        data = data.rename(columns={'source_absolute_path': 'source_analysis_relative_path'})
        data['source_analysis_relative_path'] = pointers
    if 'unit' not in data:
        data['unit'] = unit
    data['provenance_source_file'] = str(source.relative_to(A)).replace('\\', '/')
    data['source_filter'] = selector
    if note:
        data['definition'] = note
    path = OUT / 'source_tables' / (name + '.csv')
    path.parent.mkdir(exist_ok=True)
    data.to_csv(path, index=False)
    TABLES.append({'name': name, 'csv': str(path), 'rows': len(data), 'source': str(source), 'selector': selector, 'unit': unit, 'note': note, 'numeric_columns': list(data.select_dtypes(include='number').columns)})
    return data

def new_figure(number):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_PT / 72), facecolor='white', dpi=180)
    ax = fig.add_axes([0, 0, 1, 1], label=f'title_{number}')
    ax.set_axis_off()
    ax.set_title(TITLES[number - 1], loc='left', x=0.045, y=0.965, va='top', fontsize=9, fontweight='bold', color=sb.INK, pad=0)
    return fig

def molecule(fig, rect, smiles, highlight_methyl=False, template=None):
    if rect == [0.31, 0.244, 0.22, 0.095] or rect == [0.69, 0.244, 0.22, 0.095]:
        rect = [rect[0], 0.258, rect[2], rect[3]]
    if len(rect) == 4 and rect[1:] == [0.266, 0.2, 0.12]:
        rect = [rect[0], 0.281, rect[2], rect[3]]
    marked = highlight_methyl or ('CSc1' in smiles and template == 'Nc1ccccc1')
    entry = CHEM.get(smiles, marked, template)
    PLACEMENTS.append({'figure_id': id(fig), 'rect': list(rect), 'asset': entry})

def table_heatmap(fig, rect, values, row_names, col_names, highlight=()):
    ax = fig.add_axes(rect)
    rows, columns = values.shape
    for (i, j), value in np.ndenumerate(values):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=sb.YIELD_CMAP(value / 100), edgecolor='white', linewidth=0.35))
        ax.text(j, i, sb.display_yield(value), ha='center', va='center', fontsize=11, color='white' if value > 72 else sb.INK, fontweight='bold' if (i, j) in highlight else 'normal')
    for i, j in highlight:
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=sb.GOLD, lw=1.2))
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(range(columns), col_names, fontsize=9)
    ax.set_yticks(range(rows), row_names, fontsize=9)
    ax.tick_params(length=0, pad=8)
    ax.spines[:].set_visible(False)
    return ax

def formatted(text):
    replacements = {'PdCl2': 'PdCl₂', 'PPh3': 'PPh₃', 'PCy3': 'PCy₃', 'Cs2CO3': 'Cs₂CO₃', 'K2CO3': 'K₂CO₃', 'K3PO4': 'K₃PO₄', 'pp2': 'pp²', ' -> ': ' → ', '->': '→', ' >=': ' ≥', ' <=': ' ≤', '100 C': '100 °C', '25 C': '25 °C', ' x ': ' × '}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace('(PPh₃)2', '(PPh₃)₂').replace('(PCy₃)2', '(PCy₃)₂')
    return numeric_minus(text)

def finalize_text(fig):
    title = fig.axes[0].get_title(loc='left')
    if not title:
        raise ValueError('Empty left-hand figure title')
    for ax in fig.axes:
        if not ax.axison:
            continue
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            ticks = axis.get_majorticklocs()
            axis.set_ticks([t for t in ticks if low <= t <= high])
            labels = [formatted(t.get_text()) for t in axis.get_ticklabels()]
            axis.set_ticklabels(labels)
    for item in fig.findobj(Text):
        original = item.get_text()
        if original.startswith('Yield (%) | each base'):
            item.set_y(0.015)
        if original in ('3-SMe aniline', '4-SMe aniline'):
            item.set_y(0.242)
        if original.startswith('117 distinct failure-to-recovery comparisons'):
            item.set_y(0.025)
        item.set_text(formatted(original))
        size = item.get_fontsize()
        is_title = item is fig.axes[0]._left_title
        item.set_fontsize(9 if is_title else max(5.2, min(13, size * WIDTH_MM / (11.4 * 25.4))))
        item.set_fontfamily('DejaVu Sans')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_window_extent(renderer)
    outside = []
    for item in fig.findobj(Text):
        if not item.get_text() or not item.get_visible():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width + 1 or (box.y1 > bounds.height + 1)):
            outside.append({'text': item.get_text(), 'bounds': list(box.bounds)})
    if outside:
        raise ValueError(f'Text outside final canvas: {outside}')
    return title

def save_vector(fig, stem, title):
    readback = finalize_text(fig)
    if formatted(title) != readback:
        raise ValueError('Figure title / legend title mismatch')
    temp = OUT / (stem + '_plot.svg')
    fig.savefig(temp, format='svg', facecolor='white')
    base_pdf = OUT / (stem + '_plot.pdf')
    fig.savefig(base_pdf, format='pdf', facecolor='white')
    root = ET.fromstring(temp.read_text(encoding='utf-8'))
    width_pt, height_pt = fig.get_size_inches() * 72
    scale = 0.75
    chemistry_boxes = []
    for placement in [x for x in PLACEMENTS if x['figure_id'] == id(fig)]:
        x, y, w, h = placement['rect']
        entry = placement['asset']
        ew, eh = (entry['width_pt'], entry['height_pt'])
        px = x * width_pt + (w * width_pt - ew * scale) / 2
        py = (1 - y - h) * height_pt + (h * height_pt - eh * scale) / 2
        if min(px, py) < 0 or px + ew * scale > width_pt or py + eh * scale > height_pt:
            raise ValueError(f"Chemical object outside figure: {entry['smiles']}")
        group = ET.SubElement(root, f'{{{NS}}}g', {'transform': f'translate({px:.6f},{py:.6f}) scale({scale})'})
        native = ET.fromstring(Path(entry['svg']).read_text(encoding='utf-8-sig'))
        view = native.get('viewBox', f'0 0 {ew} {eh}').replace(',', ' ').split()
        vx, vy, vw, vh = map(float, view)
        inner = ET.SubElement(group, f'{{{NS}}}g', {'transform': f'scale({ew / vw:.9f},{eh / vh:.9f}) translate({-vx},{-vy})'})
        for child in native:
            inner.append(copy.deepcopy(child))
        chemistry_boxes.append({'smiles': entry['smiles'], 'x_pt': px, 'y_pt': py, 'width_pt': ew * scale, 'height_pt': eh * scale, 'scale': scale})
    svg_path = FIG / (stem + '.svg')
    svg_path.write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')
    if any((n.tag.split('}')[-1] == 'image' for n in root.iter())):
        raise ValueError('Raster image embedded in final SVG')
    pdf = pymupdf.open(base_pdf)
    for placement, box in zip([x for x in PLACEMENTS if x['figure_id'] == id(fig)], chemistry_boxes):
        native = pymupdf.open(placement['asset']['pdf'])
        rect = pymupdf.Rect(box['x_pt'], box['y_pt'], box['x_pt'] + box['width_pt'], box['y_pt'] + box['height_pt'])
        pdf[0].show_pdf_page(rect, native, 0)
        native.close()
    pdf.set_metadata({'title': formatted(title), 'creator': 'M1 measured data; native ChemDraw vector structures'})
    pdf_path = FIG / (stem + '.pdf')
    pdf.save(pdf_path)
    pix = pdf[0].get_pixmap(dpi=300, alpha=False)
    pix.save(FIG / (stem + '.png'))
    if pdf[0].get_images() or len(pdf[0].get_drawings()) == 0:
        raise ValueError('Final PDF must contain vector paths and no raster images')
    QA.append({'figure': stem, 'title': formatted(title), 'width_mm': width_pt / 72 * 25.4, 'height_mm': height_pt / 72 * 25.4, 'vector_paths': len(pdf[0].get_drawings()), 'raster_images': 0, 'chemical_boxes': chemistry_boxes, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300, 'pixel_bounds_pass': True})
    pdf.close()
    plt.close(fig)

def final_figure3(data):
    fig = sb.figure3(data)
    for text in list(fig.texts):
        x, y = text.get_position()
        if x == 0.6 and y in (0.29, 0.237, 0.155, 0.1):
            text.remove()
    values = data['components'].own_mean_share
    fig.text(0.61, 0.31, 'DENOMINATOR: NET PREDICTION GAIN', fontsize=8, color=sb.BLUE, fontweight='bold')
    fig.text(0.61, 0.267, f'{values.min():.1f}–{values.max():.1f}%', fontsize=19, color=sb.BLUE, fontweight='bold')
    fig.text(0.61, 0.222, "From each pair's own mean across backgrounds", fontsize=8)
    fig.add_artist(Line2D([0.61, 0.95], [0.196, 0.196], transform=fig.transFigure, lw=0.5, color='#c8cdd2'))
    fig.text(0.61, 0.176, 'DENOMINATOR: BASELINE MODULATION ERROR', fontsize=8, color=sb.GRAY, fontweight='bold')
    fig.text(0.61, 0.131, f"{data['modulation_pct']:.2f}%", fontsize=18, color=sb.GRAY, fontweight='bold')
    fig.text(0.61, 0.087, 'Removed by structure prediction (k = 3, array-balanced)', fontsize=8)
    return fig

def final_figure2(data):
    fig = sb.figure2(data)
    for _, row in data['witness'].iterrows():
        shown_delta = Decimal(sb.display_yield(row.high_yield_pp)) - Decimal(sb.display_yield(row.low_yield_pp))
        for text in fig.texts:
            if text.get_text() == f'{row.delta:+.2f}':
                text.set_text(f'{shown_delta:+.2f}')
    cross = data['cross']
    labels = {f'{10000.0 * cross[name]:.2f}': f'{10000.0 * cross[name]:.3f}' for name in ('target_mean_mse', 'real_mse', 'test_permuted_exact_mse')}
    for text in fig.findobj(Text):
        if text.get_text() in labels:
            text.set_text(labels[text.get_text()])
    return fig

def source_main(data):
    records = pd.DataFrame(sb.RECORDS)
    records.to_csv(OUT / 'figure_data.csv', index=False)
    for number, sub in records.groupby('figure', sort=True):
        for panel, rows in sub.groupby('panel', sort=True):
            add_table(f'Fig{number}_{panel}', rows.drop(columns=['figure', 'panel']), A / 'scripts/m1_storyboard.py', f'M1 source selection for Figure {number}{panel}; source and selector columns retain original provenance', 'see unit column')
    path = sb.RECIPE_OPTIONS / 'round9/pairing_repair/selected_12cell_original_records.csv.gz'
    raw = frame(path)
    for num, action in ((1, '0e5d6cf7a808adc5a0a4_d0_s01_r1'), (4, '4bcee624ba37f2c2cef3_d0_s01_r0')):
        selected = raw.loc[raw.action_id.eq(action)]
        add_table(f'Fig{num}_raw_identity', selected[['action_id', 'molecular_side', 'panel_position', 'record_id', 'source_file', 'source_sheet', 'source_row', 'condition_id', 'archive', 'plate', 'acquisition', 'yield_numeric', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles']], path, f'action_id={action}', 'yield_numeric: fraction')
    repair_path = sb.RECIPE_OPTIONS / 'round9/pairing_repair/all_actual_yield_repairs_with_12cells.csv'
    repairs = frame(repair_path)
    selected = repairs.loc[repairs.action_id.eq('4bcee624ba37f2c2cef3_d0_s01_r0')]
    if len(selected) != 1 or selected.measured_cell_repair_id.nunique() != 1:
        raise ValueError('Figure 4c must be a member of the measured repair set')
    counts = repairs.groupby('dataset_id').agg(n_actions=('action_id', 'size'), n_distinct_comparisons=('measured_cell_repair_id', 'nunique'), n_edits=('edit_id', 'nunique')).reset_index()
    add_table('Fig4_c_repair_counts', counts, repair_path, 'all actual_yield_repair_5pp_gain10; distinct by measured_cell_repair_id', 'count')
    conflicts = frame(sb.RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv')
    add_table('Fig4_b_conflict_views', conflicts, sb.RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv', f'all {len(conflicts)} rows; Boolean full_graph_rescues20', 'yields and capacities: fraction')

def si_chemical(data):
    title = 'Additional molecular changes reshape reagent preferences'
    fig = new_figure(1)
    fig.axes[0].set_title(title, loc='left', x=0.045, y=0.965, va='top', fontsize=9, fontweight='bold', pad=0)
    path = sb.RECIPE_OPTIONS / 'round8/context_decomposition/selected_actual_yields_and_lower_order_reconstructions.csv'
    acy = frame(path)
    acy = acy.loc[acy.partner_pair_comparison_id.astype(str).eq('5e3f5818ec51384251')].copy()
    add_table('S1_a_amide16', acy[['edit_level', 'partner_level', 'corner', 'observed_yield_pct', 'target_row_id', 'recipe_id']], path, 'partner_pair_comparison_id=5e3f5818ec51384251', 'yield %')
    meta_path = sb.RECIPE_OPTIONS / 'round8/context_decomposition/selected_chemical_examples.csv'
    meta = sb.select_one(frame(meta_path), partner_pair_comparison_id='5e3f5818ec51384251')
    raw_path = sb.RECIPE_OPTIONS / 'round8/context_decomposition/selected_example_original_record_links.csv.gz'
    raw = frame(raw_path)
    raw = raw.loc[raw.rectangle_id.eq('q_d0ddc4bc626d5687') & raw.target_row_id.isin(acy.target_row_id)]
    add_table('S1_a_raw_identity', raw, raw_path, 'rectangle_id=q_d0ddc4bc626d5687; target_row_id in the sixteen displayed cells', 'yield: fraction')
    sb.label(fig, 0.045, 0.89, 'a', 'Acid fluorination removes two opposite diagonal preferences')
    for smi, x in ((meta.structural_molecule_a, 0.11), (meta.structural_molecule_b, 0.3), (meta.partner0_smiles, 0.56), (meta.partner1_smiles, 0.78)):
        molecule(fig, [x, 0.68, 0.15, 0.14], smi, False, 'c1ccccc1')
    for x, name in ((0.185, 'Acid A'), (0.375, 'Acid B'), (0.635, 'Aniline C'), (0.855, 'Aniline D')):
        fig.text(x, 0.678, name, ha='center', fontsize=8, color=sb.GRAY)
    for (edit, partner), x in zip((('A', 'C'), ('B', 'C'), ('A', 'D'), ('B', 'D')), (0.1, 0.3, 0.56, 0.76)):
        block = acy.loc[acy.edit_level.eq(edit) & acy.partner_level.eq(partner)].copy()
        block['corner_str'] = block.corner.astype(int).map({0: '00', 10: '10', 1: '01', 11: '11'})
        vals = block.set_index('corner_str').loc[['00', '10', '01', '11'], 'observed_yield_pct'].to_numpy().reshape(2, 2).T
        sb.table_heatmap(fig, [x, 0.46, 0.16, 0.17], vals, ['Oxyma', 'Cl-HOBT'] if x in (0.1, 0.56) else ['', ''], ['DCC', 'DIC'])
        fig.text(x + 0.08, 0.65, f'Acid {edit} / aniline {partner}', ha='center', fontsize=9)
    path = sb.EDM / 'results/focused_chemical_partner_blocks.csv'
    fdpp = frame(path)
    fdpp = fdpp.loc[fdpp.target_swap_id.eq('acylation__C18__C24') & fdpp.swap_id.eq('acylation__C18__C24') & fdpp.scope.eq('all') & fdpp.comparison.eq('primary_aniline_vs_primary_alkyl')]
    add_table('S1_b_FDPP_TOTU', fdpp, path, 'acylation__C18__C24; scope=all; comparison=primary_aniline_vs_primary_alkyl', 'yield difference pp')
    sb.label(fig, 0.045, 0.36, 'b', 'FDPP → TOTU distinguishes amine categories')
    ax = fig.add_axes([0.09, 0.12, 0.34, 0.16])
    vals = fdpp.contrast_pp.sort_values().to_numpy()
    ax.scatter(vals, np.arange(len(vals)), s=9, color=sb.BLUE)
    ax.axvline(vals.mean(), color=sb.RED, lw=1)
    ax.annotate(f'mean {vals.mean():.2f}', (vals.mean(), 1.02), xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords='offset points', color=sb.RED, fontsize=9, va='bottom')
    sb.clean(ax, xlabel='Aryl minus alkyl primary-amine response (pp)', ylabel='Acid partner')
    path = sb.RECIPE_OPTIONS / 'cases/figure_common8_activators.csv'
    base = frame(path)
    lead = ['activator', 'acid_label', 'structural_double_difference_pp', 'sub_2_smiles', 'sub_1_smiles_target', 'sub_1_smiles_control']
    source_base = base[lead + [c for c in base if c not in lead]]
    add_table('S1_c_base_DiD', source_base, path, 'all four activators on the eight shared acid partners', 'structural_double_difference_pp: pp; raw yield and delta fields: fraction', '([DIPEA-NMM] for 4-methoxybenzylamine) - ([DIPEA-NMM] for benzylamine)')
    sb.label(fig, 0.53, 0.36, 'c', 'NMM → DIPEA depends on activator and amine')
    ax = fig.add_axes([0.62, 0.12, 0.31, 0.16])
    groups = ['PyAOP', 'TFFH', 'BTFFH', 'DCMT']
    group_labels = []
    for k, group in enumerate(groups):
        v = base.loc[base.activator.eq(group), 'structural_double_difference_pp'].to_numpy()
        ax.scatter(v, np.repeat(k, len(v)), s=10, color=sb.BLUE, alpha=0.6)
        ax.plot([v.mean()], [k], '|', color=sb.RED, ms=12, mew=2)
        group_labels.append(f'{group} (mean {v.mean():+.2f})')
    ax.set_yticks(range(len(groups)), group_labels)
    ax.axvline(0, color=sb.GRAY, lw=0.5)
    ax.invert_yaxis()
    sb.clean(ax, xlabel='4-MeO-benzylamine minus benzylamine (pp)')
    return (fig, title)

def si_information():
    title = 'Measured response differences and local information contribute differently'
    fig = new_figure(1)
    fig.axes[0].set_title(title, loc='left', x=0.045, y=0.965, va='top', fontsize=9, fontweight='bold', pad=0)
    path = sb.REFERENCE_LEARNING / 'tables/chemical_observation_channels_screen_scores.csv'
    screens = frame(path)
    screens = screens.loc[screens.domain.eq('strict_641') & screens.scope.eq('all_panels_complete') & screens.split.eq('pair') & screens.fraction.eq(1.0) & screens.weighting.eq('context_balanced')].copy()
    cols = ['screen', 'family', 'semantic_full_optimal_d_given_s_raw_mse', 'different_other_full_optimal_d_given_s_raw_mse', 'other_full_optimal_d_given_s_specificity_raw_mse']
    table = screens[cols].copy()
    table[cols[2:]] *= 10000.0
    add_table('S2_a_DgivenS_17screens', table, path, 'strict_641; all_panels_complete; pair; fraction=1; context_balanced', 'MSE pp2')
    sb.label(fig, 0.045, 0.89, 'a', 'The difference adds information beyond mean activity')
    ax = fig.add_axes([0.28, 0.22, 0.29, 0.59])
    ordered = table.sort_values(cols[-1])
    ax.barh(range(len(ordered)), ordered[cols[-1]], color=[sb.TEAL if x > 0 else sb.RED for x in ordered[cols[-1]]])
    ax.set_yticks(range(len(ordered)), ordered.screen, fontsize=8)
    ax.axvline(0, color=sb.GRAY, lw=0.5)
    sb.clean(ax, xlabel='Same-swap specificity of D beyond S (pp²)')
    path = sb.RESPONSE_LEARNING / 'Local_Information_Content_E67/results/e67_decomposition_classes_20.csv'
    local = frame(path)
    local = local.loc[local.stratum.eq('all')].copy()
    add_table('S2_b_local_content', local, path, 'stratum=all; all 20 source tables, including unsupported rows', 'change in MSE pp2', 'Rows marked supported=False have zero test support and no error estimate; Finkelstein is a source table, not a coupling array.')
    selected = local.loc[local.reaction_type.isin(['buchwald-hartwig', 'ullmann'])].set_index('reaction_type')
    sb.label(fig, 0.65, 0.89, 'b', 'Distinct roles for\nlocal information')
    ax = fig.add_axes([0.73, 0.5, 0.2, 0.27])
    for i, cls in enumerate(['buchwald-hartwig', 'ullmann']):
        r = selected.loc[cls]
        for j, name in enumerate(['local_level_content', 'local_modulation_content']):
            v = r[name]
            if cls == 'ullmann' and name == 'local_modulation_content':
                ax.plot(v, i * 3 + j, '|', color=sb.TEAL, ms=7, mew=1)
                ax.text(v + 0.3, i * 3 + j, f'Nearly unchanged\n({v:+.2f})', ha='left', va='center', fontsize=7)
                continue
            ax.barh(i * 3 + j, v, color=sb.BLUE if j == 0 else sb.TEAL)
            ax.text(v + (0.3 if v >= 0 else -0.3), i * 3 + j, f'{v:+.3f}', ha='left' if v >= 0 else 'right', va='center', fontsize=8)
    ax.set_yticks([0.5, 3.5], ['Buchwald–Hartwig', 'Ullmann'])
    ax.set_xlim(-19, 6)
    ax.set_ylim(-0.65, 4.8)
    ax.axvline(0, color=sb.GRAY, lw=0.5)
    ax.invert_yaxis()
    sb.clean(ax, xlabel='Change in prediction error (pp²)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=sb.BLUE, label='Local level labels'), Patch(facecolor=sb.TEAL, label='Local modulation labels')],
              loc='upper left', bbox_to_anchor=(-0.05, -0.28), frameon=False, fontsize=8, handlelength=1.0, handleheight=0.8)
    return (fig, title)

def si_routes():
    title = 'Combining reaction scopes can require a yield sacrifice'
    fig = new_figure(1)
    fig.axes[0].set_title(title, loc='left', x=0.045, y=0.965, va='top', fontsize=9, fontweight='bold', pad=0)
    witness_path = sb.RECIPE_OPTIONS / 'round27/route_compromise/identical_native_support_witness/witness.json'
    wit = json.loads(witness_path.read_text(encoding='utf-8'))
    row = pd.DataFrame([{k: wit[k] for k in ('focal_molecule_a', 'focal_molecule_b', 'anchor_partner', 'partner_q', 'partner_r', 'B_context_a', 'B_context_b', 'B_union', 'compromise')}])
    add_table('S3_a_Suzuki_scope', row, witness_path, 'identical_native_support_witness; selected comparison_id', 'yield bottleneck and sacrifice: fraction')
    for suffix, filename in (('nodes', 'complete_graph_nodes.csv'), ('edges', 'complete_graph_edges.csv'), ('routes', 'frontier_measured_route_yields.csv'), ('raw_reactions', 'source_reaction_records.csv'), ('raw_conditions', 'source_condition_records.csv')):
        path = witness_path.parent / filename
        add_table('S3_a_' + suffix, frame(path), path, 'complete identical-support Suzuki witness', 'yield fields: fraction; other fields identify routes and records')
    sb.label(fig, 0.045, 0.89, 'a', 'A common Suzuki route lowers the scope bottleneck')
    for smi, x, scope in ((wit['focal_molecule_a'], 0.08, 'Scope 1'), (wit['focal_molecule_b'], 0.28, 'Scope 2')):
        molecule(fig, [x, 0.64, 0.16, 0.15], smi, True, 'OB(O)c1ccccc1')
        fig.text(x + 0.08, 0.615, scope, ha='center', va='top', fontsize=9)
    ax = fig.add_axes([0.16, 0.22, 0.29, 0.34])
    values = [100 * wit[k] for k in ('B_context_a', 'B_context_b', 'B_union')]
    ax.barh(range(len(values)), values, color=[sb.BLUE, sb.BLUE, sb.RED])
    ax.set_yticks(range(len(values)), ['Scope 1', 'Scope 2', 'Combined scopes'])
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.annotate(f'{v:.2f}', (v, i), xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=9)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_ylim(3.25, -0.6)
    ax.annotate('', (values[2], 2.45), (values[1], 2.45), arrowprops=dict(arrowstyle='|-|', color=sb.RED, lw=0.7))
    ax.text((values[1] + values[2]) / 2, 2.65, f"{100 * wit['compromise']:.2f} pp", ha='center', va='top', fontsize=9, color=sb.RED)
    sb.clean(ax, xlabel='Best route bottleneck yield (%)')
    path = sb.RECIPE_OPTIONS / 'round29/source_compromise/measured_cases_exact_pareto_frontiers.csv'
    front = frame(path)
    front = front.loc[front.case.eq('positive_source_compromise')].copy()
    add_table('S3_b_amide_frontier', front, path, 'case=positive_source_compromise; all exact Pareto points', 'bottleneck yields and sacrifice: fraction')
    for suffix, filename in (('nodes', 'measured_cases_complete_node_yields.csv'), ('edges', 'measured_cases_complete_edges.csv'), ('routes', 'measured_cases_routes.csv'), ('raw_pointers', 'measured_cases_route_raw_pointers.csv')):
        path = path.parent / filename
        raw = frame(path)
        add_table('S3_b_' + suffix, raw.loc[raw.case.eq('positive_source_compromise')], path, 'case=positive_source_compromise', 'yield fields: fraction; other fields identify routes and records')
    sb.label(fig, 0.58, 0.89, 'b', 'An amide target scope requires source sacrifice')
    ax = fig.add_axes([0.65, 0.22, 0.28, 0.58])
    x, y = (100 * front.source_sacrifice_from_B, 100 * front.target_bottleneck)
    ax.plot(x, y, 'o-', color=sb.TEAL, markersize=4)
    ax.axhline(20, color=sb.GRAY, ls='--', lw=0.7)
    ax.annotate('20%', (0.98, 20), xycoords=ax.get_yaxis_transform(), xytext=(0, 3), textcoords='offset points', ha='right', va='bottom', color=sb.GRAY, fontsize=9)
    sb.clean(ax, xlabel='Source-scope sacrifice (pp)', ylabel='Target-scope bottleneck yield (%)')
    eligible = front.loc[front.target_bottleneck.ge(0.2)]
    selected = eligible.loc[eligible.source_sacrifice_from_B.idxmin()]
    cost = 100 * front.loc[front.target_bottleneck.ge(0.2), 'source_sacrifice_from_B'].min()
    target = 100 * selected.target_bottleneck
    ax.annotate(f'{cost:.2f} pp', (cost, target), xytext=(5, -14), textcoords='offset points', fontsize=9, color=sb.RED, va='top')
    return (fig, title)

def graphical_abstract(data):
    title = 'Reagent-swap effects'
    fig = plt.figure(figsize=(90 / 25.4, 50 / 25.4), facecolor='white', dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, loc='left', x=0.05, y=0.98, va='top', fontsize=9, fontweight='bold', pad=0)
    fig.text(0.05, 0.81, 'Learnable but not copyable', fontsize=12, fontweight='bold', color=sb.BLUE)
    for y, row in zip((0.57, 0.28), data['flip']):
        molecule(fig, [0.035, y - 0.13, 0.26, 0.25], row.sub_1_smiles)
        gain = row.y1 > row.y0
        fig.text(0.34, y, f'{100 * row.y0:.2f}% → {100 * row.y1:.2f}%', fontsize=13, va='center', color=sb.TEAL if gain else sb.RED)
    fig.text(0.05, 0.065, 'PdCl₂(dtbpf) → PdCl₂(PPh₃)₂; fixed boronic acid', fontsize=8, color=sb.GRAY)
    return (fig, title)

def technical_tables():
    path = sb.REFERENCE_LEARNING / 'tables/chemical_observation_channels_panel_registry.csv'
    panels = frame(path)
    add_table('S4_observation_panels', panels, path, 'all registered observation panels', 'two wells per panel', 'Policies are scored independently; the complete-11-panel population requires availability of each candidate panel.')
    for label, rel in (('S4_unit_RMS', 'Target_Geometry_E64/results/e64_summary.csv'), ('S4_local_allocations', 'Family_Information_Recovery_E66/results/e66_allocation_sensitivity.csv')):
        path = sb.RESPONSE_LEARNING / rel
        raw = frame(path)
        if label == 'S4_unit_RMS':
            raw = raw.loc[raw.geometry.eq('centered_unit_rms_zeroembed') & raw.stratum.eq('all')]
            raw = raw.drop(columns=['unit0_vector_mse', 'unit0_vector_mae', 'unit0_rmse'])
            selector = 'geometry=centered_unit_rms_zeroembed; stratum=all'
            unit = 'native_vector_mse: dimensionless squared normalized response'
        else:
            raw = raw.loc[raw.population.eq('common8') & raw.target_geometry.eq('raw_delta') & raw.stratum.eq('all')]
            selector = 'population=common8; target_geometry=raw_delta; stratum=all; each fixed allocation'
            unit = 'native_vector_mse: pp2; native_vector_mae: pp'
        add_table(label, raw, path, selector, unit)
    update_path = A / 'out/m1_update_step_summary.csv'
    if update_path.is_file():
        updates = frame(update_path)
        updates = updates.loc[updates.target_geometry.eq('raw_delta') & updates.path.eq('zero_local_to_level_only') | updates.target_geometry.eq('centered_unit_rms_zeroembed') & updates.path.eq('zero_local_to_full')].copy()
        add_table('S4_update_steps', updates, update_path, 'raw_delta/zero_local_to_level_only or centered_unit_rms_zeroembed/zero_local_to_full; existing-prediction arithmetic; no new model fit', 'see target_geometry and metric headers')
    cross_library = (('S5_cross_library_caps', 'headline.csv', 'all original domains, channels and macros; caps=24,48,96,192', 'MSE and gain: yield fraction squared; *_rmse_pp: percentage points; support: counts'), ('S5_cross_library_directions', 'direction_scores.csv', 'all registered directions and statuses; caps=24,48,96,192', 'MSE and gain: yield fraction squared; actual_spent: target coordinates; n_calibration/n_test: identities'), ('S5_cross_library_budget_audit', 'budget_support_audit.csv.gz', 'all registered designs and statuses; caps=24,48,96,192', 'budget, cost and overlap: coordinate or record counts; n_*: identity counts; JSON fields retain native identifiers'))
    for label, suffix, selector, unit in cross_library:
        path = sb.REFERENCE_LEARNING / ('tables/cross_library_source_channels_' + suffix)
        add_table(label, frame(path), path, selector, unit)
    path = sb.EDM / 'results/sparse_sign_feasibility_summary.csv'
    signs = frame(path)
    signs = signs.loc[signs.population.eq('all') & signs.panel.eq('observed_sparse') & signs.objective.eq('count') & signs.threshold_pp.isin([0, 20]) & signs.stratum.isin(['ALL', 'finkelstein'])].copy()
    if len(signs) != 4:
        raise ValueError('Expected the all-table and Finkelstein sign counts at both thresholds')
    add_table('S6_sign_20_tables', signs, path, 'all; observed_sparse; count; stratum=ALL or finkelstein; threshold_pp=0 or 20', 'threshold_pp and *_pp_*: yield percentage points; *_fraction_*: fractions (0-1); n_*_edges: typed swaps; n_decisive_classes: source tables; n_*_appearances: response appearances', 'mismatched_mass_pp_sum sums absolute conflicting responses; *_pp_per_* fields are mean or median mass per response appearance.')
    reconciliation_tables = (('S6_corpus_reconciliation', 'corpus_reconciliation.csv', {'population', 'available', 'used', 'difference', 'unit', 'criterion', 'source', 'selector'}, "all reconciliation populations; available-used differences are within the row's declared unit", 'available, used and difference: row-specific unit', 'Branch supports are distinct populations; counts in different units must not be subtracted.'), ('S6_screen_array_mapping', 'reference_learning_screen_array_mapping.csv', {'screen', 'parent_array', 'source_file', 'conditions', 'analysis_pairs', 'analysis_wells', 'reserve_pairs', 'reserve_wells', 'sealed_pairs', 'sealed_wells'}, 'all REFERENCE_LEARNING qualifying screens, linked to their source array and raw table', 'conditions: recipe coordinates; *_pairs: substrate-pair rows; *_wells: measured condition cells', 'A reconstructed screen is a subset of a source array; screen and array counts are not interchangeable.'), ('S6_broadest_support', 'broadest_support.csv', {'evidence', 'support', 'model', 'estimate', 'unit', 'n_edges', 'n_scores', 'source', 'selector'}, 'all prespecified broadest-support comparisons; each learner retains its own eligible denominator', 'estimate: row-specific unit; n_edges and n_scores: support counts defined by the row', 'Support is analysis-specific; different eligible query counts do not constitute a paired comparison.'))
    for label, filename, required, selector, unit, note in reconciliation_tables:
        path = A / 'out' / filename
        data = frame(path)
        if (missing := (required - set(data.columns))):
            raise ValueError(f'Missing reconciliation columns for {label}: {sorted(missing)}')
        if 'unit' in data and data.unit.fillna('').astype(str).str.strip().eq('').any():
            raise ValueError(f'Every row in {label} must declare its estimate/count unit')
        add_table(label, data, path, selector, unit, note)

def main():
    from publication_figures.build import main as build_final
    build_final()
if __name__ == '__main__':
    main()
