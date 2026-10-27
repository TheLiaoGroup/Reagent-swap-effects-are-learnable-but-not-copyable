"""Approved production layouts; scientific selections remain in the original modules."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_cdt
from publication_figures import common as c, figures14 as old14, figures23 as old23
BLUE, RED, ORANGE, TEAL = ('#2F6F9F', '#B8434E', '#D9822B', '#0E7C74')
GRAY, INK, LIGHT = ('#89939B', '#25292C', '#E9EDEF')
SOURCE_RENAMES = {'Fig3_b_direction_errors': 'Fig3_a_direction_errors', 'Fig3_c_training_pairs': 'Fig3_b_training_pairs', 'Fig3_c_training_relation': 'Fig3_b_training_relation', 'Fig3_c_reagent_identities': 'Fig3_b_reagent_identities', 'Fig3_d_specificity_screens': 'Fig3_c_specificity_screens', 'Fig3_d_specificity_macro': 'Fig3_c_specificity_macro', 'Fig4_c_local_learning': 'Fig4_d_local_learning', 'Fig4_c_measurement_cost': 'Fig4_d_measurement_cost', 'Fig4_e_scope_partition': 'Fig4_c_scope_partition'}

@contextmanager
def panel_source_names():
    original = c.source

    def source(name, *args, **kwargs):
        return original(SOURCE_RENAMES.get(name, name), *args, **kwargs)
    c.source = source
    try:
        yield
    finally:
        c.source = original

def figure(height):
    c.setup()
    fig = plt.figure(figsize=(183 / 25.4, height / 25.4), dpi=180)
    fig._m1_chemistry, fig._m1_panels = ([], [])
    return fig

def title(fig, x, y, letter, text):
    fig.text(x - 0.023, y, letter.upper(), weight='bold', fontsize=8, va='bottom')
    fig.text(x, y, text, weight='bold', fontsize=6, va='bottom')

def panel(fig, rect, letter, text, title_x=None, title_y=None):
    ax = c.panel(fig, rect, letter, text, title_x=title_x)
    ax._draft_panel_letter.set_fontsize(8)
    if title_y is not None:
        ax._draft_panel_letter.set_y(title_y)
        ax._draft_panel_title.set_y(title_y)
    return ax

def arrow(ax, start, end, color=GRAY, **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=5, lw=0.65, color=color, **kwargs)
    ax.add_patch(patch)
    return patch

def molecule(fig, rect, smiles, **kwargs):
    c.molecule(fig, rect, smiles, **kwargs)
    if not (kwargs.get('halo') or kwargs.get('highlight_methyl')):
        return
    entry = fig._m1_chemistry[-1]
    root = c.ET.fromstring(Path(entry['normalized_svg']).read_text(encoding='utf-8'))
    for elem in root.iter():
        if elem.tag.rsplit('}', 1)[-1] == 'circle':
            elem.set('fill', GRAY)
            elem.set('fill-opacity', '.28')
    svg = c.ET.tostring(root, encoding='unicode')
    key = hashlib.sha256(svg.encode()).hexdigest()[:14]
    vector = c.OUT / 'render_work' / f'chem_gray_{key}.svg'
    pdf_path = vector.with_suffix('.pdf')
    vector.write_text(svg, encoding='utf-8')
    if not pdf_path.exists():
        with c.pymupdf.open(stream=svg.encode(), filetype='svg') as native:
            with c.pymupdf.open('pdf', native.convert_to_pdf()) as pdf:
                pdf.save(pdf_path)
    editable = c.OUT / 'materials/chemistry' / f'chem_gray_{key}.svg'
    editable.write_text(svg, encoding='utf-8')
    entry.update({'pdf': str(pdf_path), 'normalized_svg': str(vector), 'sha256': hashlib.sha256(pdf_path.read_bytes()).hexdigest()})
    entry['editable_assets']['svg'] = str(editable)

def layout_review(fig, report):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:
        if not ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                hidden.update((id(t) for t in axis.get_ticklabels()))
                hidden.update([id(axis.label), id(axis.get_offset_text())])
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text() and (id(t) not in hidden)]
    overlaps = []
    for i, (a, aa) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if min(aa.x1, bb.x1) - max(aa.x0, bb.x0) > 1 and min(aa.y1, bb.y1) - max(aa.y0, bb.y0) > 1:
                overlaps.append([a.get_text(), b.get_text()])
    chemical_hits, point_hits, arrow_hits = ([], [], [])
    hpt = report['height_mm'] / 25.4 * 72
    for text, box in texts:
        for chem in report['chemistry']:
            x, y, w, h = chem['bounds_pt']
            tx, ty = (box.x0 * 72 / fig.dpi, hpt - box.y1 * 72 / fig.dpi)
            tw, th = (box.width * 72 / fig.dpi, box.height * 72 / fig.dpi)
            if min(tx + tw, x + w) > max(tx, x) and min(ty + th, y + h) > max(ty, y):
                chemical_hits.append([text.get_text(), chem['smiles']])
        for ax in fig.axes:
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    points = collection.get_offset_transform().transform(np.asarray(collection.get_offsets()))
                    if len(points) and ((points[:, 0] > box.x0) & (points[:, 0] < box.x1) & (points[:, 1] > box.y0) & (points[:, 1] < box.y1)).any():
                        point_hits.append(text.get_text())
        for patch in fig.findobj(FancyArrowPatch):
            path = patch.get_transform().transform_path(patch.get_path())
            if path.get_extents().overlaps(box) and path.intersects_bbox(box, filled=False):
                arrow_hits.append(text.get_text())
    return {'text_text_intersections': overlaps, 'text_chemical_box_intersections': chemical_hits, 'text_scatter_center_intersections': sorted(set(point_hits)), 'text_arrow_intersections': sorted(set(arrow_hits)), 'visible_text_count': len(texts)}

def save(fig, stem, heading, extra=None):
    report = c.save(fig, stem, heading)
    report['layout_review'] = layout_review(fig, report)
    report['redesign_checks'] = extra or {}
    for values in report['layout_review'].values():
        if isinstance(values, list) and values:
            raise ValueError(f"Layout intersection in {stem}: {report['layout_review']}")
    (c.OUT / f'{stem}_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

def compact_card(fig, ax, witness, smiles, number, bounds):
    x0, x1, y0, y1 = bounds
    xy = ax.transAxes.inverted().transform(ax.transData.transform([[x0, y0], [x1, y1]]))
    pos = ax.get_position()
    fx, fy = (pos.x0 + xy[0, 0] * pos.width, pos.y0 + xy[0, 1] * pos.height)
    fw, fh = ((xy[1, 0] - xy[0, 0]) * pos.width, (xy[1, 1] - xy[0, 1]) * pos.height)
    rows = witness.loc[witness.sub_1_smiles.eq(smiles)].copy()
    rows['order'] = rows.background_fingerprint.str.contains('CMP0334@3').astype(int)
    fig.text(fx + 1 / (fig.get_figwidth() * 72), fy + fh * 0.88 + 1 / (fig.get_figheight() * 72), str(number), weight='bold', fontsize=6)
    for i, row in enumerate(rows.sort_values('order').itertuples()):
        yy = fy + fh * (0.51 - i * 0.43)
        fig.text(fx, yy + fh * 0.12, 'Cs2CO3' if i == 0 else 'K2CO3', fontsize=5)
        for j, value in enumerate([row.low_yield_pp, row.high_yield_pp]):
            xx = fx + fw * (0.4 + j * 0.32)
            fig.add_artist(Rectangle((xx, yy), fw * 0.27, fh * 0.3, transform=fig.transFigure, facecolor='white', edgecolor=GRAY, lw=0.5))
            fig.text(xx + fw * 0.135, yy + fh * 0.15, f'{value:.0f}', ha='center', va='center', fontsize=5.3)
        fig.add_artist(FancyArrowPatch((fx + fw * 0.674, yy + fh * 0.15), (fx + fw * 0.714, yy + fh * 0.15), transform=fig.transFigure, arrowstyle='->', mutation_scale=4, lw=0.6, color=BLUE if row.delta > 0 else RED))

def figure2():
    wide, witness, _ = old23.context_data()
    swaps, responses = old23.six_swaps()
    signs = old23.sign_data()
    fig = figure(170)
    ax = panel(fig, [0.075, 0.24, 0.65, 0.7], 'a', 'Catalyst responses across base backgrounds')
    ax.scatter(wide.Cs2CO3_delta_pp, wide.K2CO3_delta_pp, color=GRAY, s=5, alpha=0.45, lw=0)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set(xlim=(-100, 100), ylim=(-100, 100), xticks=[-100, -50, 0, 50, 100], yticks=[-100, -50, 0, 50, 100])
    c.clean(ax, 'Cs2CO3 response (pp)', 'K2CO3 response (pp)')
    correlation = wide[['Cs2CO3_delta_pp', 'K2CO3_delta_pp']].corr().iloc[0, 1]
    ax.text(0.98, 0.98, f'Opposite: 145/412\nr = {correlation:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=6, linespacing=1.8)
    bounds = [(-99, -50, -99, -80), (-99, -50, 77, 96)]
    for i, (smiles, rect) in enumerate(zip(['Brc1ccc2[nH]ncc2c1', 'COc1cccnc1Br'], bounds)):
        x0, x1, y0, y1 = rect
        if ((wide.Cs2CO3_delta_pp > x0) & (wide.Cs2CO3_delta_pp < x1) & (wide.K2CO3_delta_pp > y0) & (wide.K2CO3_delta_pp < y1)).any():
            raise ValueError('Response card would hide a measured point')
        compact_card(fig, ax, witness, smiles, i + 1, rect)
        rows = witness.loc[witness.sub_1_smiles.eq(smiles)].copy()
        rows['order'] = rows.background_fingerprint.str.contains('CMP0334@3').astype(int)
        x, y = rows.sort_values('order').delta.to_numpy()
        ax.scatter(x, y, s=25, facecolors='white', edgecolors=INK, lw=1.1, zorder=5)
        ax.annotate(str(i + 1), (x, y), xytext=(6, 3), textcoords='offset points', fontsize=6, weight='bold')
    ax.text(-99, 67, 'No pair × background split', fontsize=5.5)
    fig.text(0.78, 0.955, 'Example substrates', fontsize=6, weight='bold')
    for i, (smiles, y) in enumerate([('Brc1ccc2[nH]ncc2c1', 0.835), ('COc1cccnc1Br', 0.72)]):
        molecule(fig, [0.79, y, 0.15, 0.09], smiles)
        fig.text(0.755, y + 0.04, str(i + 1), fontsize=6, weight='bold')
    conflict = panel(fig, [0.868, 0.24, 0.056, 0.425], 'c', 'Conflicts across 19 arrays', title_x=0.77, title_y=0.7)
    order = signs.loc[signs.threshold_pp.eq(20)].sort_values(['conflict_percent', 'stratum'], ascending=[False, True], na_position='last').stratum.tolist()
    for i, name in enumerate(order):
        rows = signs.loc[signs.stratum.eq(name)].set_index('threshold_pp')
        values = rows.loc[[0, 20], 'conflict_percent'].to_numpy()
        if np.isfinite(values).all():
            conflict.plot(values, [i, i], color=GRAY, lw=0.5)
        for value, marker in zip(values, ['o', 'D']):
            if np.isfinite(value):
                conflict.scatter(value, i, s=8, marker=marker, facecolors='white' if marker == 'D' else INK, edgecolors=INK, lw=0.5, zorder=3, clip_on=False)
        counts = ' | '.join((f"{int(rows.loc[t, 'n_conflicting_edges'])}/{int(rows.loc[t, 'n_cycle_edges'])}" for t in [0, 20]))
        conflict.text(1.13, i, counts, transform=conflict.get_yaxis_transform(), va='center', fontsize=4.4)
        if not np.isfinite(values).any():
            conflict.text(3, i, 'No cycles', fontsize=4.2, va='center', color=GRAY)
    names = [old23.CLASSES.get(x, x) for x in order]
    conflict.set_yticks(range(len(order)), names, fontsize=5)
    conflict.set(xlim=(-3, 103), ylim=(len(order) - 0.5, -0.5), xticks=[0, 100])
    c.clean(conflict, 'Conflicts (%)')
    conflict.legend(handles=[Line2D([], [], marker='o', ls='', color=INK, label='0 pp', markersize=3), Line2D([], [], marker='D', ls='', color=INK, markerfacecolor='white', label='20 pp', markersize=3)], loc='lower right', bbox_to_anchor=(1.12, 1.0), frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.6, fontsize=5)
    fig.text(0.966, 0.667, '0 | 20 pp', ha='center', fontsize=4.5)
    strips = panel(fig, [0.075, 0.038, 0.855, 0.101], 'b', 'Response directions across transformations', title_y=0.157)
    strips.axis('off')
    equations = {'suzuki': 'ArBr + ArB(OH)₂ → ArAr', 'acylation': 'RCO₂H + RNH₂ → RCONHR', 'buchwald-hartwig': 'ArBr + RNH₂ → ArNHR', 'snar': 'ArF + HNu → ArNu', 'ullmann': 'ArBr + ROH → ArOR', 'kumada': 'ArBr + RMgBr → ArR'}
    substitutions = {'suzuki': 'PdCl2(dtbpf) → PdCl2(PPh3)2', 'acylation': 'TFFH → DCMT', 'buchwald-hartwig': 'RuPhos Pd G3 → BrettPhos Pd G4', 'snar': 'Triethylamine → 2,6-lutidine', 'ullmann': 'N-Methyl-L-proline → L-proline', 'kumada': 'Tetramethylurea → DMI'}
    for i, row in enumerate(swaps.itertuples()):
        x, y = (0.075 + i % 3 * 0.3, 0.103 - i // 3 * 0.065)
        axis = fig.add_axes([x, y, 0.255, 0.018])
        values = responses[row.reaction_type].display_delta_pp.to_numpy()
        jitter = (np.arange(len(values)) * 0.61803398875 % 1 - 0.5) * 0.6
        axis.scatter(values, jitter, s=1.5, alpha=0.5, color=np.where(values > 0, BLUE, RED), lw=0)
        axis.axvline(0, color=GRAY, lw=0.45)
        axis.set(xlim=(-100, 100), ylim=(-0.5, 0.5), yticks=[], xticks=[-100, 0, 100])
        axis.spines[['left', 'top', 'right']].set_visible(False)
        axis.tick_params(length=2, pad=1, labelsize=5)
        fig.text(x, y + 0.036, equations[row.reaction_type], fontsize=5)
        fig.text(x, y + 0.023, substitutions[row.reaction_type], fontsize=4.8)
        fig.text(x + 0.255, y + 0.036, f'n={len(values):,}', fontsize=4.5, ha='right')
    fig.text(0.5, 0.003, 'Yield change (pp)', ha='center', fontsize=6)
    fig._m1_panels.sort(key=lambda item: item[0])
    report = save(fig, 'Figure_2', 'Substrate pairs and recipe backgrounds jointly determine substitution responses')
    extent = ax.get_window_extent(fig.canvas.get_renderer())
    ratio = extent.width * extent.height * (300 / fig.dpi) ** 2 / np.prod(report['png_pixels'])
    if ratio < 0.45:
        raise ValueError('The saved Figure 2 canvas fails the 45% scatter-area requirement')
    report['redesign_checks'].update(scatter_actual_saved_png_area_fraction=ratio, card_bounds_data=bounds, points_inside_card_bounds=[0, 0])
    (c.OUT / 'Figure_2_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

def reference_motif(fig):
    ax = fig.add_axes([0.075, 0.858, 0.85, 0.095])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(0, 0.72, 'Same substrate pair', fontsize=6, weight='bold')
    for xx, label, target in [(0.23, 'Reference background', False), (0.74, 'Target background', True)]:
        ax.text(xx + 0.084, 1.03, label, ha='center', va='bottom', fontsize=5.5)
        for i, endpoint in enumerate(['X', 'Y']):
            x = xx + i * 0.085
            ax.add_patch(Rectangle((x, 0.44), 0.07, 0.37, facecolor=LIGHT if target else 'white', edgecolor=GRAY, lw=0.5))
            ax.text(x + 0.035, 0.625, '?' if target else 'y$_{' + endpoint + '}$', ha='center', va='center', fontsize=6)
            ax.text(x + 0.035, 0.18, endpoint, ha='center', fontsize=5)
        arrow(ax, (xx + 0.071, 0.625), (xx + 0.083, 0.625), INK)
    for y, label, color, rad in [(1.18, 'Copy', ORANGE, -0.18), (0.16, 'Learn + history', TEAL, 0.18)]:
        arrow(ax, (0.405, 0.625), (0.745, 0.625), color, connectionstyle=f'arc3,rad={rad}', shrinkB=0)
        ax.text(0.56, y, label, color=color, ha='center', va='center', fontsize=5.5)

def figure3():
    with panel_source_names():
        shown, errors = (old23.main_screen_data(), old23.direction_errors())
        points, coefficient = old23.training_data()
        identities = old23.reagent_identities()
        local, macro = old23.specificity_data()
    fig = figure(170)
    reference_motif(fig)
    ax = panel(fig, [0.275, 0.425, 0.65, 0.398], 'a', 'Same reference, different decisions', title_x=0.075, title_y=0.831)
    for i, row in enumerate(shown.itertuples()):
        learn, copy = (row.semantic_full_raw_relative_pct, row.direct_raw_relative_pct)
        ax.plot([learn, min(copy, 78)], [i, i], color=GRAY, lw=0.65)
        ax.scatter(learn, i, color=TEAL, s=18 if i == 0 else 9, zorder=3)
        if copy >= 80:
            ax.plot(78, i, '>', color=ORANGE, ms=4)
            ax.text(75, i - 0.4, f'{copy:.0f}%', ha='right', color=ORANGE, fontsize=5)
        else:
            ax.scatter(copy, i, color=ORANGE, s=18 if i == 0 else 9, zorder=3)
    ax.axvline(0, color=GRAY, ls='--', lw=0.5)
    ax.set(xlim=(-45, 80), ylim=(len(shown) - 0.5, -0.8), xticks=[-40, 0, 40, 80])
    ax.set_yticks(range(len(shown)), [old23.screen_label(s) for s in shown.screen], fontsize=5.5)
    ax.get_yticklabels()[0].set_weight('bold')
    c.clean(ax, 'Error change vs history alone\n(no reference) (%)')
    for value, color, align in [(shown.iloc[0].semantic_full_raw_relative_pct, TEAL, 'right'), (shown.iloc[0].direct_raw_relative_pct, ORANGE, 'left')]:
        ax.annotate(f'{value:+.1f}%', (value, 0), xytext=(-5 if align == 'right' else 5, 4), textcoords='offset points', color=color, ha=align, fontsize=6)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=ORANGE, label='Copy', markersize=3), Line2D([], [], marker='o', ls='', color=TEAL, label='Learn', markersize=3)], frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.text(78, 0.8, f'missed or wrong directions {errors.wrong_per_100.iloc[0]:.1f} → {errors.wrong_per_100.iloc[1]:.1f} per 100', ha='right', va='center', fontsize=5)
    ax = panel(fig, [0.095, 0.075, 0.265, 0.25], 'b', 'Reference-to-target relationship', title_y=0.35)
    ax.scatter(points.reference_contrast_pp, points.target_contrast_pp, color=GRAY, s=3, alpha=0.4, lw=0)
    xx = np.array([-100, 100])
    ax.plot(xx, xx, color=ORANGE, lw=0.9)
    ax.plot(xx, coefficient.centered_S_zero_slice_intercept_pp + coefficient.centered_S_zero_slice_slope * xx, color=TEAL, lw=1)
    ax.text(27, 91, 'Copy (slope 1)', color=ORANGE, fontsize=5.5)
    ax.text(70, 23.5, 'Learned', color=TEAL, fontsize=5.5)
    ax.axhline(0, color=LIGHT, lw=0.4)
    ax.axvline(0, color=LIGHT, lw=0.4)
    ax.set(xlim=(-100, 100), ylim=(-100, 100), xticks=[-100, 0, 100], yticks=[-100, 0, 100])
    c.clean(ax, 'Reference response (pp)', 'Target response (pp)')
    ax.text(0.035, 0.93, '452 training pairs', transform=ax.transAxes, fontsize=5)
    for y, row in zip([0.235, 0.1], identities):
        molecule(fig, [0.38, y, 0.125, 0.08], row['smiles'])
    fig.add_artist(FancyArrowPatch((0.443, 0.225), (0.443, 0.19), transform=fig.transFigure, arrowstyle='->', mutation_scale=5, color=INK, lw=0.55))
    fig.text(0.443, 0.34, 'reagent swap', ha='center', fontsize=5)
    ax = panel(fig, [0.79, 0.075, 0.16, 0.225], 'c', 'Same-substitution reference', title_x=0.535, title_y=0.35)
    settings = [(s, f) for s in ['pair', 'role1', 'role2'] for f in [0.25, 1.0]]
    names = {'pair': 'Substrate pair', 'role1': 'First reactant', 'role2': 'Second reactant'}
    for i, (split, fraction) in enumerate(settings):
        rows = local.loc[local.split.eq(split) & local.fraction.eq(fraction)].sort_values('screen')
        value = macro.loc[macro.split.eq(split) & macro.fraction.eq(fraction)]
        if len(rows) != 17 or len(value) != 1:
            raise ValueError('Matched-reference population changed')
        ax.scatter(100 * rows.other_recovery_vs_control, i + np.linspace(-0.15, 0.15, len(rows)), color=GRAY, s=5, alpha=0.6, lw=0)
        ax.scatter(100 * value.other_recovery_vs_control, [i], color=TEAL, s=16, edgecolors='white', lw=0.4)
        fig.text(0.53, 0.075 + 0.225 * (1 - (i + 0.6) / 6.4), names[split], va='center', fontsize=5.5)
        fig.text(0.75, 0.075 + 0.225 * (1 - (i + 0.6) / 6.4), f'{int(100 * fraction)}%', ha='right', va='center', fontsize=5.5)
    ax.set(ylim=(5.8, -0.6), xlim=(-10, 70), yticks=[], xticks=[0, 30, 60])
    ax.axvline(0, color=GRAY, lw=0.5, ls='--')
    c.clean(ax, 'Error removed versus\nmatched control (%)')
    fig.text(0.53, 0.317, 'Held out', fontsize=5.5, weight='bold')
    fig.text(0.75, 0.317, 'History', ha='right', fontsize=5.5, weight='bold')
    return save(fig, 'Figure_3', 'Historical relationships make the same reference measurements useful', {'direction_annotation_population': '21997 targets; measured absolute change at least20 pp; definition in caption', 'panels': {'a': 'same-reference screen comparison and existing direction counts', 'b': 'stored training relationship and actual reagent identities', 'c': 'same-substitution specificity'}, 'new_scientific_counts_or_fits': False})

def amide_example(fig, cells, rect):
    ymap, hmap = rect
    start = cells.loc[cells.ternary_level.eq(0) & cells.binary_level.eq(0)].set_index('molecular_side')
    for x, side in [(0.08, 'before'), (0.29, 'after')]:
        molecule(fig, [x, ymap(0.827), 0.125, hmap(0.1)], start.loc[side, 'reactant1_smiles'], halo=True)
    fig.add_artist(FancyArrowPatch((0.22, ymap(0.858)), (0.273, ymap(0.858)), transform=fig.transFigure, arrowstyle='->', mutation_scale=6, lw=0.6, color=RED))
    for x, value in [(0.142, 'meta'), (0.352, 'para')]:
        fig.text(x, ymap(0.805), value, ha='center', fontsize=5.5)
    fig.text(0.142, ymap(0.778), f"{c.display_decimal(100 * start.loc['before', 'yield_numeric'])}%", ha='center', fontsize=6)
    fig.text(0.352, ymap(0.778), f"{c.display_decimal(100 * start.loc['after', 'yield_numeric'])}%", ha='center', fontsize=6, color=RED)
    fig.text(0.075, ymap(0.748), 'HBTU; phenoxyacetic acid', fontsize=5.5)
    ax = fig.add_axes([0.14, ymap(0.548), 0.3, hmap(0.167)])
    ax.set(xlim=(-0.6, 2.4), ylim=(-0.5, 1.5))
    ax.axis('off')
    after = cells.loc[cells.molecular_side.eq('after')].sort_values('panel_position')
    half_width, half_height = (0.27, 0.27)
    edges = []
    for i, row in after.iterrows():
        for j, other in after.iterrows():
            if j <= i or int(row.ternary_level != other.ternary_level) + int(row.binary_level != other.binary_level) != 1:
                continue
            if row.binary_level == other.binary_level:
                low, high = sorted((int(row.ternary_level), int(other.ternary_level)))
                for column in range(low, high):
                    ax.plot([column + half_width, column + 1 - half_width], [row.binary_level, row.binary_level], color='#CBD0D3', lw=0.5, zorder=0)
            else:
                low, high = sorted((row.binary_level, other.binary_level))
                ax.plot([row.ternary_level, row.ternary_level], [low + half_height, high - half_height], color='#CBD0D3', lw=0.5, zorder=0)
            rescue = row.binary_level == other.binary_level == 0 and {row.ternary_level, other.ternary_level} == {0, 2}
            edges.append(dict(recipe0=row.recipe_id, recipe1=other.recipe_id, differing_slots=1, selected_recovery=bool(rescue)))
    c.source('Fig4_a_recipe_edges', pd.DataFrame(edges), old14.PAIRING / 'selected_pairing_repair_all_12_yields_and_conditions.csv', 'Prescribed amide after-edit six-node grid; edge iff one measured recipe slot differs', 'edges')
    arrow(ax, (half_width + 0.05, -0.03), (2 - half_width - 0.05, -0.03), BLUE, connectionstyle='arc3,rad=-.32')
    for row in after.itertuples():
        failure = row.binary_level == 0 and row.ternary_level == 0
        recovery = row.binary_level == 0 and row.ternary_level == 2
        color = RED if failure else BLUE if recovery else GRAY
        ax.add_patch(Rectangle((row.ternary_level - half_width, row.binary_level - half_height), 2 * half_width, 2 * half_height, facecolor='white', edgecolor=color, lw=1 if failure or recovery else 0.5, zorder=2))
        digits = 2 if failure or recovery else 0
        shown = c.display_decimal(100 * row.yield_numeric) if digits == 2 else f'{100 * row.yield_numeric:.0f}'
        ax.text(row.ternary_level, row.binary_level, shown + '%', ha='center', va='center', fontsize=6, zorder=3)
    for i, label in enumerate(['HBTU', 'PyBOP', 'TPTU']):
        ax.text(i, 1.45, label, ha='center', fontsize=5.5)
    ax.text(-0.36, 0, 'No HOBt', ha='right', va='center', fontsize=5.5)
    ax.text(-0.36, 1, '+ HOBt', ha='right', va='center', fontsize=5.5)
    fig.text(0.075, ymap(0.519), 'DIPEA, DMF; 25 °C, 2 h', fontsize=5.5)

def suzuki_example(fig, cells, rect):
    ymap, hmap = rect
    images = []
    frame_location = None
    for j, side in enumerate(['before', 'after']):
        rows = cells.loc[cells.molecular_side.eq(side)].sort_values('panel_position')
        values = rows.yield_numeric.to_numpy().reshape(3, 2) * 100
        x = 0.68 + j * 0.15
        molecule(fig, [x - 0.01, ymap(0.827), 0.13, hmap(0.1)], rows.reactant2_smiles.iloc[0], highlight_methyl=True, template='OB(O)c1ccccc1', halo=True)
        fig.text(x + 0.055, ymap(0.805), '3-methyl' if j == 0 else '2-methyl', ha='center', fontsize=5.5)
        ax = fig.add_axes([x, ymap(0.575), 0.115, hmap(0.19)])
        image = ax.pcolormesh(np.arange(3) - 0.5, np.arange(4) - 0.5, values, cmap=__import__('matplotlib.colors', fromlist=['ListedColormap']).ListedColormap(plt.get_cmap('Greys')(np.linspace(15 / 150, 115 / 150, 256))), vmin=0, vmax=100, rasterized=False)
        images.append(image)
        ax.set(xlim=(-0.5, 1.5), ylim=(2.5, -0.5))
        ax.set_xticks([0, 1], ['Cs2CO3', 'K3PO4'], fontsize=5)
        ax.set_yticks([0, 1, 2], ['PdCl2(dtbpf)', 'PdCl2(PCy3)2', 'SPhos Pd G3'] if j == 0 else [], fontsize=5.5)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0, pad=2)
        for (r, k), value in np.ndenumerate(values):
            cited = k == 1 and r in (1, 2)
            shown = c.display_decimal(value) if cited else f'{value:.0f}'
            ax.text(k, r, shown, ha='center', va='center', fontsize=6, color='white' if value >= 65 else INK)
        if j:
            for rr, color in [(1, RED), (2, BLUE)]:
                ax.add_patch(Rectangle((0.52, rr - 0.48), 0.96, 0.96, fill=False, edgecolor='white', lw=3, zorder=4))
                ax.add_patch(Rectangle((0.52, rr - 0.48), 0.96, 0.96, fill=False, edgecolor=color, lw=1, zorder=5))
            arrow(ax, (1.6, 1.02), (1.6, 1.96), BLUE, connectionstyle='arc3,rad=-.40', clip_on=False)
            frame_location = (x + 0.115 * 1.02 / 2, ymap(0.575) + hmap(0.19) * 0.5 / 3)
    fig.add_artist(FancyArrowPatch((0.81, ymap(0.858)), (0.83, ymap(0.858)), transform=fig.transFigure, arrowstyle='->', mutation_scale=6, lw=0.6, color=RED))
    bar = fig.colorbar(images[-1], cax=fig.add_axes([0.68, ymap(0.533), 0.265, hmap(0.007)]), orientation='horizontal', ticks=[0, 50, 100])
    bar.solids.set_rasterized(False)
    bar.ax.tick_params(length=2, pad=1, labelsize=5)
    bar.set_label('Yield (%)', labelpad=1, fontsize=5.5)
    fig.text(0.55, ymap(0.47), 'Same bromoindazole; 100 °C, 12 h', fontsize=5.5)
    fig.text(0.55, ymap(0.452), '1,4-dioxane/H₂O', fontsize=5.5)
    return frame_location

def lstar(color):
    rgb = np.array(to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = np.dot(linear, [0.2126729, 0.7151522, 0.072175])
    f = luminance ** (1 / 3) if luminance > (6 / 29) ** 3 else luminance / (3 * (6 / 29) ** 2) + 4 / 29
    return float(116 * f - 16)

def blank_square(png, fraction_box):
    x, y, w, h = fraction_box
    width, height = png.size
    bounds = (int(round(x * width)), int(round((1 - y - h) * height)), int(round((x + w) * width)), int(round((1 - y) * height)))
    crop = np.array(png.crop(bounds))
    white = np.pad((crop[:, :, :3] > 245).all(axis=2), 1, constant_values=False)
    distance = distance_transform_cdt(white, metric='chessboard')
    row, col = np.unravel_index(np.argmax(distance), distance.shape)
    side = int(2 * distance[row, col] - 1)
    return {'panel_crop_pixels': list(bounds), 'largest_empty_square_side_pixels': side, 'largest_empty_square_fraction_panel_width': side / crop.shape[1], 'largest_empty_square_center_pixels': [int(bounds[0] + col - 1), int(bounds[1] + row - 1)], 'white_definition': 'All RGB channels >245; maximal axis-aligned empty square; panel boundary treated as occupied', 'passes_panel_width_quarter': side <= crop.shape[1] / 4}

def figure4():
    amide = old14.repair_data(old14.AMIDE_ACTION, 'Fig4_a_amide_12_cells')
    methyl = old14.repair_data(old14.METHYL_ACTION, 'Fig4_b_methyl_12_cells')
    repair_path = old14.PAIRING / 'all_actual_yield_repairs_with_12cells.csv'
    repairs = c.frame(repair_path)
    chosen = repairs.loc[repairs.action_id.eq(old14.AMIDE_ACTION)]
    if len(chosen) != 1 or not bool(chosen.actual_yield_repair_5pp_gain10.iloc[0]):
        raise ValueError('The prescribed amide rescue is outside the117 measured recovery contrasts')
    counts = repairs.groupby('dataset_id').measured_cell_repair_id.nunique().rename('n_measured_repairs').reset_index()
    if counts.n_measured_repairs.sum() != 117:
        raise ValueError('The measured rescue-contrast count changed')
    c.source('Fig4_a_recovery_classes', counts, repair_path, 'All eligible repairs; distinct measured_cell_repair_id within each reaction array', 'measured contrasts')
    with panel_source_names():
        values, costs = old14.learning_data()
        partition = old14.scope_partition()
    height = 112
    fig = figure(height)
    ymap = lambda y: 1 - (1 - y) * 130 / height
    hmap = lambda h: h * 130 / height
    ax = panel(fig, [0.075, ymap(0.52), 0.4, hmap(0.39)], 'a', 'Amide alternatives', title_y=ymap(0.94))
    ax.axis('off')
    ax = panel(fig, [0.55, ymap(0.52), 0.4, hmap(0.39)], 'b', 'Suzuki alternatives', title_y=ymap(0.94))
    ax.axis('off')
    amide_example(fig, amide, (ymap, hmap))
    frame = suzuki_example(fig, methyl, (ymap, hmap))
    row_bottom, row_height = (2 / height, 35 / height)
    heading_y = 34.5 / height
    c_region, d_region = ([0.075, row_bottom, 0.525, row_height], [0.6, row_bottom, 0.35, row_height])
    scope = panel(fig, [0.075, 19 / height, 0.525, 2.4 / height], 'c', 'Shared-scope conflicts', title_y=heading_y)
    left = 0
    for count, color in zip(partition.n, ['#718089', '#A3ADB3', '#C6CDD1', '#E3E6E8']):
        scope.barh(0, count, left=left, color=color, height=0.75, edgecolor='white', lw=0.3)
        left += count
    scope.set(xlim=(0, int(partition.n.sum())), ylim=(-0.5, 0.5))
    scope.axis('off')
    fig.text(0.075, 25 / height, '285/542 resolved', fontsize=6)
    fig.text(0.6, 25 / height, '257/542 unresolved', ha='right', fontsize=6)
    from matplotlib.patches import Patch
    keys = [f'{k} extra option{"s" if k > 1 else ""} ({n})' for k, n in zip((1, 2, 3), partition.n.iloc[:3])] + [f'unresolved ({partition.n.iloc[3]})']
    fig.legend(handles=[Patch(facecolor=col, edgecolor='none', label=lab) for col, lab in zip(['#718089', '#A3ADB3', '#C6CDD1', '#E3E6E8'], keys)],
               loc='upper left', bbox_to_anchor=(0.07, 16.5 / height), ncol=4, frameon=False, fontsize=5.5,
               handlelength=1.0, handleheight=0.8, handletextpad=0.4, columnspacing=1.2)
    curve = panel(fig, [0.72, 12 / height, 0.23, 17.5 / height], 'd', 'History adds to local profiles', title_x=0.65, title_y=heading_y)
    x = np.arange(len(values))
    curve.plot(x, values.error_change_pct, 'o-', color=TEAL, lw=1.1, ms=3)
    curve.axhline(0, color=GRAY, lw=0.6)
    labels = []
    for n in values.n_vectors:
        measured = costs.loc[costs.n_vectors.eq(n), 'local_condition_cells']
        labels.append(f'{n}\n{int(measured.min())}–{int(measured.max())}')
    curve.set_xticks(x, labels, fontsize=5.3)
    curve.set(xlim=(-0.1, 3.2), ylim=(-52, 5), yticks=[-50, -25, 0])
    curve.text(3, values.error_change_pct.iloc[-1] - 7, f'{values.error_change_pct.iloc[-1]:.1f}%', ha='center', va='top', fontsize=6, color=TEAL)
    curve.text(0.98, 1.0, 'Local-only baseline', transform=curve.transAxes, ha='right', fontsize=5, color=GRAY)
    c.clean(curve, 'Local profiles\n(endpoint wells)', 'Error change after\nadding history (%)')
    report = save(fig, 'Figure_4', 'Measured alternatives resolve part of the shared-scope conflict', {'panel_order': 'a amide; b Suzuki; c scope band; d local curve', 'lower_row_height_mm': 35, 'lower_row_width_fractions': {'c': 0.6, 'd': 0.4}, 'panel_source_mapping': {'c': 'Fig4_c_scope_partition', 'd': 'Fig4_d_local_learning/measurement_cost'}})
    png = Image.open(c.FIG / 'Figure_4.png').convert('RGB')
    width, pixels_high = png.size
    border_x, middle_y = (int(round(frame[0] * width)), int(round((1 - frame[1]) * pixels_high)))
    blue_rgb = tuple((round(255 * value) for value in to_rgb(BLUE)))
    blue_pixels = [(x, y) for y in range(middle_y - 3, middle_y + 4) for x in range(border_x - 6, border_x + 7) if png.getpixel((x, y)) == blue_rgb]
    white_pixels = [(x, y) for y in range(middle_y - 3, middle_y + 4) for x in range(border_x + 3, border_x + 7) if png.getpixel((x, y)) == (255, 255, 255)]
    if not blue_pixels or not white_pixels:
        raise ValueError('The rendered 92% cell lacks a blue frame or inner white lining')
    blue_at, white_at = (blue_pixels[len(blue_pixels) // 2], white_pixels[len(white_pixels) // 2])
    blue_sample, white_sample = (png.getpixel(blue_at), png.getpixel(white_at))
    delta = abs(lstar(np.array(white_sample) / 255) - lstar(np.array(blue_sample) / 255))
    if delta < 30:
        raise ValueError('The rendered blue frame/white lining fails CIELAB contrast')
    blanks = {name: blank_square(png, region) for name, region in [('c', c_region), ('d', d_region)]}
    if not all((item['passes_panel_width_quarter'] for item in blanks.values())):
        raise ValueError(f'The compact Figure4 lower row fails the empty-square threshold: {blanks}')
    report['redesign_checks'].update(rendered_92_cell_contrast={'blue_sample_xy_px': blue_at, 'blue_sample_rgb': blue_sample, 'inner_white_sample_xy_px': white_at, 'inner_white_sample_rgb': white_sample, 'delta_Lstar': delta, 'passes_30_Lstar': delta >= 30}, lower_row_pixel_blanks=blanks)
    (c.OUT / 'Figure_4_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

class AcceptedChemistry:
    """Use the accepted native vectors during a scoped layout review."""

    def __init__(self):
        from rdkit import Chem
        self.Chem = Chem
        self.assets = json.loads((c.ROOT / 'Submission/02_Figures/ChemDraw/chemistry_assets.json').read_text(encoding='utf-8'))['assets']

    def get(self, smiles, marked=False, template=None):
        key = self.Chem.MolToSmiles(self.Chem.MolFromSmiles(smiles))
        rows = [row for row in self.assets if row['canonical_smiles'] == key and bool(row['marked']) == marked]
        if len(rows) != 1:
            raise ValueError(f'Accepted native vector is not unique: {smiles}, marked={marked}')
        return rows[0]

def main():
    c.OUT = c.A / 'out/redesign_step2_20261022/figures234'
    c.FIG, c.DATA = (c.OUT, c.OUT / 'source_tables')
    c.OUT.mkdir(parents=True, exist_ok=True)
    c.CHEM = AcceptedChemistry()
    c.QA.clear()
    for builder in [figure2, figure3, figure4]:
        c.TABLES.clear()
        builder()
    reverse_names = {new: old for old, new in SOURCE_RENAMES.items()}
    comparisons = []
    for report in c.QA:
        for table in report['source_tables']:
            name = table['name']
            original_name = reverse_names.get(name, name)
            original = c.A / 'out/final_figures/source_tables' / f'{original_name}.csv'
            current = pd.read_csv(table['csv'])
            accepted = pd.read_csv(original)
            metadata = {'provenance_source_file', 'source_filter', 'definition', 'unit'}
            columns = [column for column in current if column not in metadata]
            if set(columns) != {column for column in accepted if column not in metadata}:
                raise ValueError(f'Scientific source columns differ: {name}')
            pd.testing.assert_frame_equal(current[columns], accepted[columns], check_dtype=False, check_exact=False, rtol=0, atol=1e-09)
            differences = []
            for column in current[columns].select_dtypes(include='number'):
                difference = np.abs(current[column].to_numpy() - accepted[column].to_numpy())
                finite = difference[np.isfinite(difference)]
                if len(finite):
                    differences.append(float(finite.max()))
            comparisons.append({'current_table': name, 'accepted_table': original_name, 'rows': len(current), 'scientific_columns': len(columns), 'maximum_absolute_numeric_difference': max(differences, default=0), 'passed': True})
    (c.OUT / 'scientific_source_comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    (c.OUT / 'figure_checks.json').write_text(json.dumps(c.QA, indent=2), encoding='utf-8')
    print(json.dumps({'figures': [item['figure'] for item in c.QA], 'output': str(c.OUT)}, indent=2))
if __name__ == '__main__':
    main()
