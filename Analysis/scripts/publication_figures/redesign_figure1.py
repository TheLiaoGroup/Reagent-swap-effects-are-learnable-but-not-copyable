"""Approved reagent-swap map, using measured endpoints and native vectors."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pymupdf
from scipy.ndimage import distance_transform_cdt
from publication_figures import common as c, figures14, figures23
SVG = 'http://www.w3.org/2000/svg'
INKSCAPE = 'http://www.inkscape.org/namespaces/inkscape'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
ET.register_namespace('inkscape', INKSCAPE)
W, H = (183, 100)
PT = 25.4 / 72
BLUE, RED, ORANGE, TEAL = ('#2F6F9F', '#B8434E', '#D9822B', '#0E7C74')
INK, GRAY = ('#25292C', '#7D858B')
PYRIDINE = 'COc1cccnc1Br'

def node(parent, kind, **attributes):
    return ET.SubElement(parent, f'{{{SVG}}}{kind}', {key.replace('_', '-'): str(value) for key, value in attributes.items()})

class Map:

    def __init__(self):
        self.root = ET.Element(f'{{{SVG}}}svg', width=f'{W}mm', height=f'{H}mm', viewBox=f'0 0 {W} {H}', version='1.1')
        self.labels, self.chemistry, self.arrows = ([], [], [])

    def layer(self, name):
        group = node(self.root, 'g', id=name)
        group.set(f'{{{INKSCAPE}}}groupmode', 'layer')
        group.set(f'{{{INKSCAPE}}}label', name)
        return group

    def text(self, group, x, y, text, size=6, color=INK, anchor='start', bold=False):
        text = c.numeric_minus(text)
        if len(text.split()) > 6:
            raise ValueError(f'Figure 1 label exceeds six words: {text}')
        parts = re.split('([\\u2080-\\u2089]+)', text)
        runs = []
        for part in parts:
            if not part:
                continue
            subscript = all(('₀' <= char <= '₉' for char in part))
            value = part.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
            run_size = size * (0.7 if subscript else 1)
            runs.append((value, subscript, run_size))
        if not any((run[1] for run in runs)):
            width = pymupdf.get_text_length(text, fontname='hebo' if bold else 'helv', fontsize=size) * PT
            left = x - {'start': 0, 'middle': width / 2, 'end': width}[anchor]
            node(group, 'text', x=x, y=y, fill=color, font_family='Arial', font_size=size * PT, text_anchor=anchor, font_weight='bold' if bold else 'normal').text = text
            self.labels.append({'text': text, 'bounds_mm': [left, y - size * PT * 0.8, width, size * PT], 'font_pt': size})
            return
        label = node(group, 'text', x=0, y=y, fill=color, font_family='Arial', font_size=size * PT, text_anchor='start', font_weight='bold' if bold else 'normal')
        previous_drop = 0
        for value, subscript, run_size in runs:
            drop = size * PT * 0.22 if subscript else 0
            span = node(label, 'tspan', dy=drop - previous_drop, font_size=run_size * PT)
            if subscript:
                span.set('data-script', 'sub')
            span.text = value
            previous_drop = drop
        probe = ET.Element(f'{{{SVG}}}svg', width=f'{W}mm', height=f'{H}mm', viewBox=f'0 0 {W} {H}')
        measured_label = copy.deepcopy(label)
        measured_label.set('y', str(size * PT * 2))
        probe.append(measured_label)
        with pymupdf.open(stream=ET.tostring(probe), filetype='svg') as source:
            with pymupdf.open('pdf', source.convert_to_pdf()) as document:
                spans = [span for block in document[0].get_text('dict')['blocks'] for line in block.get('lines', []) for span in line['spans']]
        x0, x1 = (min((span['bbox'][0] for span in spans)), max((span['bbox'][2] for span in spans)))
        width = (x1 - x0) * PT
        left = x - {'start': 0, 'middle': width / 2, 'end': width}[anchor]
        label.set('x', str(left - x0 * PT))
        self.labels.append({'text': text, 'bounds_mm': [left, y - size * PT * 0.8, width, size * PT], 'font_pt': size})

    def line(self, group, start, end, color=INK, width=0.2, arrow=False, identity=None):
        attributes = dict(d=f'M {start[0]} {start[1]} L {end[0]} {end[1]}', fill='none', stroke=color, stroke_width=width)
        if identity:
            attributes['id'] = identity
        node(group, 'path', **attributes)
        if arrow:
            delta = np.array(end, float) - start
            unit = delta / np.linalg.norm(delta)
            back = np.array(end) - 1.2 * unit
            perp = 0.43 * np.array([-unit[1], unit[0]])
            points = [end, back + perp, back - perp]
            node(group, 'polygon', points=' '.join((f'{x},{y}' for x, y in points)), fill=color)
            self.arrows.append({'id': identity, 'start_mm': list(start), 'end_mm': list(end), 'length_mm': float(np.linalg.norm(delta)), 'color': color})

    def chemical(self, group, smiles, cx, cy, name, left=None):
        entry = c.CHEM.get(smiles)
        if not entry['native'] or entry['inchi_key'] != entry['roundtrip_inchi_key']:
            raise ValueError('Figure 1 requires an identity-checked native chemical vector')
        raw = Path(entry['svg'])
        root = ET.fromstring(raw.read_text(encoding='utf-8-sig'))
        if root.get('data-m1-native-scale') != '1':
            raise ValueError('Native chemical scale was not normalized')
        box = list(map(float, root.get('viewBox').split()))
        scale = 25.4 / 96
        width, height = (box[2] * scale, box[3] * scale)
        if left is not None:
            cx = left + width / 2
        fragment = node(group, 'g', id=f'chem-{name}', transform=f'translate({cx - width / 2} {cy - height / 2}) scale({scale})')
        for child in root:
            fragment.append(copy.deepcopy(child))
        assets = c.OUT / 'materials/figure1'
        assets.mkdir(parents=True, exist_ok=True)
        for extension in ('svg', 'cdxml', 'cdx'):
            shutil.copy2(entry[extension], assets / f'{name}.{extension}')
        self.chemistry.append({'smiles': smiles, 'inchi_key': entry['inchi_key'], 'bounds_mm': [cx - width / 2, cy - height / 2, width, height], 'bond_length_pt': 10.8, 'label_size_pt': 6, 'native_scale': json.loads(Path(entry['scale_metadata']).read_text()), 'source_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()})

    def pair(self, group, x, y, values):
        delta = values[1] - values[0]
        color = BLUE if delta > 0 else RED
        for offset, label, value in ((0, 'X', values[0]), (26, 'Y', values[1])):
            node(group, 'rect', x=x + offset, y=y, width=12, height=8, fill='white', stroke=GRAY, stroke_width=0.18)
            self.text(group, x + offset + 6, y + 5, f'{value:.2f}%', anchor='middle')
            self.text(group, x + offset + 6, y - 1.6, label, color=GRAY, anchor='middle', size=5.5)
        self.line(group, (x + 13, y + 4), (x + 25, y + 4), color, 0.28, True)
        self.text(group, x + 19, y + 1.1, f'Δ {delta:+.0f} pp', size=6.5, color=color, bold=True, anchor='middle')

def references():
    return {'recommendation': [16, 17], 'general': [22, 23], 'transfer': [19], 'prediction': [20]}

def measurements():
    original_source = c.source

    def source(name, *args, **kwargs):
        return original_source(name.replace('Fig1_a_', 'Fig1_', 1), *args, **kwargs)
    c.source = source
    try:
        records, corpus = figures14.flagship_data()
    finally:
        c.source = original_source
    path = figures23.EDM / 'matched_response_profiles.parquet'
    raw = c.frame(path)
    witness = raw.loc[raw.reaction_type.eq('suzuki') & raw.swap_slot.eq('reagent1') & raw.component_pair.eq('CMP0092|CMP0097') & raw.sub_1_smiles.eq(PYRIDINE) & raw.sub_2_smiles.eq(figures14.BORONIC)].copy()
    if len(witness) != 2:
        raise ValueError('Pyridine illustration must have precisely two measured backgrounds')
    backgrounds = {}
    for name, code, expected in (('Cs', 'CMP0333@3', -28.3798), ('K', 'CMP0334@3', 33.8708)):
        rows = witness.loc[witness.background_fingerprint.str.contains(code, regex=False)]
        if len(rows) != 1 or not np.isclose(rows.delta.iloc[0], expected, rtol=0, atol=1e-07):
            raise ValueError('Pyridine measured illustration changed')
        backgrounds[name] = [float(rows[column].iloc[0]) for column in ('low_yield_pp', 'high_yield_pp')]
    c.source('Fig1_measured_reference', witness, path, 'Suzuki catalyst CMP0092|CMP0097; methoxybromopyridine and common anisylboronic acid; both carbonate backgrounds', 'yield percent and change pp')
    return (records, corpus, backgrounds)

def curve(m, group, start, control, end, color, width=0.35, identity=None):
    """Quadratic route with an arrowhead aligned to the final tangent."""
    node(group, 'path', d=f'M {start[0]} {start[1]} Q {control[0]} {control[1]} {end[0]} {end[1]}', fill='none', stroke=color, stroke_width=width, id=identity or '')
    tangent = np.array(end, float) - np.array(control, float)
    unit = tangent / np.linalg.norm(tangent)
    back = np.array(end) - 1.6 * unit
    perp = 0.6 * np.array([-unit[1], unit[0]])
    node(group, 'polygon', points=' '.join(f'{x},{y}' for x, y in [end, back + perp, back - perp]), fill=color)
    m.arrows.append({'id': identity, 'start_mm': list(start), 'end_mm': list(end), 'color': color})


def plate(m, group, x, y, w=21, h=13.5, shade=1.0):
    """Outline of a well plate: rounded frame and a 4 x 6 grid of wells."""
    stroke = GRAY if shade == 1.0 else '#B8BEC2'
    node(group, 'rect', x=x, y=y, width=w, height=h, rx=1.2, fill='white', stroke=stroke, stroke_width=0.22)
    for r in range(4):
        for k in range(6):
            node(group, 'circle', cx=x + 2.4 + k * (w - 4.8) / 5, cy=y + 2.3 + r * (h - 4.6) / 3, r=0.75, fill='none', stroke=stroke, stroke_width=0.15)


def headline_numbers():
    """Corpus-wide numbers shown on the map, read from the same outputs as Figs. 2C and 3A."""
    original = c.source
    renames = {'Fig2_c_array_conflicts': 'Fig1_array_conflicts', 'Fig3_a_screen_comparison': 'Fig1_overall_error_change'}

    def source(name, *args, **kwargs):
        return original(renames.get(name, name), *args, **kwargs)
    c.source = source
    try:
        signs = figures23.sign_data()
        shown = figures23.main_screen_data()
    finally:
        c.source = original
    zero = signs.loc[signs.threshold_pp.eq(0)]
    conflicts, testable = int(zero.n_conflicting_edges.sum()), int(zero.n_cycle_edges.sum())
    overall = shown.iloc[0]
    if overall.screen != 'Overall':
        raise ValueError('The first comparison row must be the overall group average')
    return conflicts, testable, float(overall.direct_raw_relative_pct), float(overall.semantic_full_raw_relative_pct)


def figure1():
    original_source = c.source

    def source(name, *args, **kwargs):
        return original_source(name.replace('Fig1_a_', 'Fig1_', 1), *args, **kwargs)
    c.source = source
    try:
        records, corpus = figures14.flagship_data()
    finally:
        c.source = original_source
    conflicts, testable, copy_pct, learn_pct = headline_numbers()
    refs = references()
    global H
    H = 94
    m = Map()
    m.root.set('height', f'{H}mm')
    m.root.set('viewBox', f'0 0 {W} {H}')
    title = 'A measured reagent swap must be learned, not copied'
    node(m.root, 'title').text = title
    yields = {s: records.loc[records.substrate_smiles.eq(s)].sort_values('endpoint').yield_percent.tolist() for s in (figures14.QUINOLINE, figures14.INDAZOLE)}

    top = m.layer('01-reaction')
    m.text(top, 2, 12, 'Ar–Br', size=7)
    m.text(top, 13.5, 12, '+', size=7)
    m.chemical(top, figures14.BORONIC, 33, 11, 'anisylboronic-acid')
    m.line(top, (52, 12), (120, 12), width=0.25, arrow=True)
    m.text(top, 86, 9.6, 'X: PdCl₂(dtbpf) → Y: PdCl₂(PPh₃)₂', size=6.5, bold=True, anchor='middle')
    m.text(top, 86, 15.6, 'Cs₂CO₃, 1,4-dioxane/H₂O, 100 °C, 12 h', anchor='middle', size=5.5, color=GRAY)
    m.text(top, 123, 12.6, 'Ar–Ar′', size=7)
    m.line(top, (0, 22.5), (W, 22.5), '#D7DADC', 0.15)

    source_layer = m.layer('02-measured-swap')
    m.text(source_layer, 1, 29, 'Measured swap', size=6.5, bold=True)
    m.chemical(source_layer, figures14.QUINOLINE, 15, 41, '6-bromoquinoline', left=2)
    m.text(source_layer, 2, 53, '6-bromoquinoline', size=6)
    m.pair(source_layer, 2, 62, yields[figures14.QUINOLINE])

    routes = m.layer('03-two-uses')

    def tag(group, x, y, label, numbers, color=GRAY):
        m.text(group, x, y, label, size=5, color=color)
        width = pymupdf.get_text_length(label, fontname='helv', fontsize=5) * PT
        m.text(group, x + width + 0.4, y - 1, ','.join(str(n) for n in numbers), size=4.5, color=color)

    curve(m, routes, (42, 63.5), (48, 35), (124, 36.5), ORANGE, 0.45, 'copy-route')
    m.text(routes, 76, 34.5, 'copy Δ', size=6.5, color=ORANGE, bold=True)
    tag(routes, 58, 28.5, 'condition recommendation', refs['recommendation'])
    tag(routes, 92, 28.5, 'general conditions', refs['general'])

    strip = m.layer('03b-all-pairs')
    deltas = corpus.display_delta_pp.to_numpy(float)
    sx0, sx1, sy = 62, 118, 52
    sp = lambda value: sx0 + (value + 100) * (sx1 - sx0) / 200
    jitter = (np.arange(len(deltas)) * 0.61803398875 % 1 - 0.5) * 5.0
    for value, dy in zip(deltas, jitter):
        node(strip, 'circle', cx=sp(value), cy=sy + dy, r=0.42, fill=BLUE if value > 0 else RED, fill_opacity=0.55)
    for smiles in (figures14.QUINOLINE, figures14.INDAZOLE):
        lo_y, hi_y = yields[smiles]
        node(strip, 'circle', cx=sp(hi_y - lo_y), cy=sy, r=1.0, fill='none', stroke=INK, stroke_width=0.3)
    m.line(strip, (sx0, sy + 4), (sx1, sy + 4), INK, 0.2)
    for tick in (-100, 0, 100):
        m.line(strip, (sp(tick), sy + 4), (sp(tick), sy + 4.9), INK, 0.2)
        m.text(strip, sp(tick), sy + 7.4, f'{tick:+d}' if tick else '0', size=5, anchor='middle')
    node(strip, 'path', d=f'M {sp(0)} {sy - 3.4} L {sp(0)} {sy + 4}', stroke=GRAY, stroke_width=0.2, fill='none')
    m.text(strip, (sx0 + sx1) / 2, sy - 4.6, f'same swap, {len(deltas)} substrate pairs', size=5.5, color=INK, anchor='middle')
    m.text(strip, (sx0 + sx1) / 2, sy + 10.4, 'yield change (pp)', size=5, color=GRAY, anchor='middle')
    curve(m, routes, (42, 67), (50, 75.5), (60, 75.5), TEAL, 0.45, 'learn-route-in')
    history = m.layer('04-history')
    for k, shade in ((2, 0.6), (1, 0.8), (0, 1.0)):
        plate(m, history, 62 + 1.6 * k, 69 - 1.6 * k, shade=shade)
    m.text(history, 74, 88.5, '19 arrays · 17 transformations', size=5.2, color=GRAY, anchor='middle')
    m.text(history, 74, 91.7, '406,909 reactions', size=5.2, color=GRAY, anchor='middle')
    curve(m, routes, (88, 76), (106, 76), (124, 76), TEAL, 0.45, 'learn-route-out')
    m.text(routes, 90, 72.5, 'learn from history', size=6.5, color=TEAL, bold=True)
    tag(routes, 90, 81.5, 'transfer learning', refs['transfer'])
    tag(routes, 90, 86, 'yield-change prediction', refs['prediction'])

    related = m.layer('05-related-substrate')
    m.text(related, 127, 29, 'Related substrate', size=6.5, bold=True)
    m.chemical(related, figures14.INDAZOLE, 140, 39, '5-bromo-1H-indazole', left=127)
    m.text(related, 127, 50.5, '5-bromo-1H-indazole', size=6)
    m.pair(related, 127, 58, yields[figures14.INDAZOLE])
    m.text(related, 171, 36, 'copy predicts +88 pp', size=5.5, color=ORANGE, bold=True, anchor='middle')
    m.line(related, (169, 39), (173, 43), ORANGE, 0.45)
    m.line(related, (169, 43), (173, 39), ORANGE, 0.45)
    m.text(related, 127, 71.5, f'Direction set jointly: {conflicts}/{testable} swaps', size=5.5, color=GRAY)

    result = m.layer('06-held-out-result')
    m.text(result, 127, 78.5, 'Same pair, new recipe', size=6.5, bold=True)
    lo, hi, x0, x1, axis_y = -20, 50, 130, 180, 87.5
    scale = (x1 - x0) / (hi - lo)
    px = lambda value: x0 + (value - lo) * scale
    m.line(result, (x0, axis_y), (x1, axis_y), INK, 0.2)
    for tick in (-20, 0, 20, 40):
        m.line(result, (px(tick), axis_y), (px(tick), axis_y + 0.9), INK, 0.2)
        m.text(result, px(tick), axis_y + 3.4, f'{tick:+d}' if tick else '0', size=5, anchor='middle')
    node(result, 'path', d=f'M {px(0)} 84 L {px(0)} {axis_y}', stroke=GRAY, stroke_width=0.2, stroke_dasharray='0.8 0.6', fill='none')
    m.line(result, (px(learn_pct), 84), (px(copy_pct), 84), GRAY, 0.3)
    node(result, 'circle', cx=px(learn_pct), cy=84, r=1.1, fill=TEAL)
    node(result, 'circle', cx=px(copy_pct), cy=84, r=1.1, fill=ORANGE)
    m.text(result, px(learn_pct) - 1.5, 81.6, f'learn {c.numeric_minus(f"{learn_pct:.1f}")}%', size=5.5, color=TEAL, bold=True)
    m.text(result, px(copy_pct) + 1.5, 81.6, f'copy +{copy_pct:.1f}%', size=5.5, color=ORANGE, bold=True, anchor='end')
    m.text(result, (x0 + x1) / 2, 93.6, 'error vs history alone (%)', size=5, color=GRAY, anchor='middle')

    shown_numbers = pd.DataFrame([
        dict(quantity='conflicting swaps', value=conflicts, denominator=testable, definition='Swaps whose observed signs contradict one sign per substrate pair times one sign per background; all nonzero changes; nineteen arrays'),
        dict(quantity='copy error change vs history alone (%)', value=copy_pct, denominator=None, definition='Overall hierarchical group average, held-out targets, two-well reference copied'),
        dict(quantity='learned error change vs history alone (%)', value=learn_pct, denominator=None, definition='Overall hierarchical group average, held-out targets, learned reference relationship'),
    ])
    c.source('Fig1_map_numbers', shown_numbers, figures23.EDM / 'sparse_sign_feasibility_summary.csv', 'Numbers printed on the map; populations as in Figs. 2C and 3A', 'count or percent')

    for i, a in enumerate(m.labels):
        for b in m.labels[i + 1:]:
            ax0, ay0, aw, ah = a['bounds_mm']
            bx0, by0, bw, bh = b['bounds_mm']
            if min(ax0 + aw, bx0 + bw) - max(ax0, bx0) > 0.2 and min(ay0 + ah, by0 + bh) - max(ay0, by0) > 0.2:
                raise ValueError(f"Figure 1 labels overlap: {a['text']} / {b['text']}")
    c.FIG.mkdir(parents=True, exist_ok=True)
    svg_path = c.FIG / 'Figure_1.svg'
    ET.ElementTree(m.root).write(svg_path, encoding='utf-8', xml_declaration=True)
    with pymupdf.open(stream=ET.tostring(m.root), filetype='svg') as svg:
        with pymupdf.open('pdf', svg.convert_to_pdf()) as pdf:
            pdf.set_metadata({'title': title, 'creator': 'M1'})
            pdf.save(c.FIG / 'Figure_1.pdf')
            page = pdf[0]
            pix = page.get_pixmap(dpi=300, alpha=False)
            pix.save(c.FIG / 'Figure_1.png')
            pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            ink = np.min(pixels, axis=2) < 225
            distance = distance_transform_cdt(np.pad(~ink, 1, constant_values=False), metric='chessboard')
            blank_mm = (2 * int(distance.max()) - 1) * 25.4 / 300
            if blank_mm > W / 6:
                raise ValueError(f'Figure 1 has an excessive blank region ({blank_mm:.1f} mm)')
            for label in m.labels:
                x, y, width, height = label['bounds_mm']
                if min(x, y) < -0.1 or x + width > W + 0.1 or y + height > H + 0.1:
                    raise ValueError(f"Figure 1 text outside canvas: {label['text']}")
            chemical_boxes = [dict(zip(('x_pt', 'y_pt', 'width_pt', 'height_pt'), [v / PT for v in item['bounds_mm']])) for item in m.chemistry]
            report = {'figure': 'Figure_1', 'title': title, 'title_visible': False, 'draft': False, 'width_mm': W, 'height_mm': H, 'panels': [], 'single_composite': True, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300, 'pixel_bounds_pass': True, 'raster_images': len(page.get_images()), 'vector_paths': len(page.get_drawings()), 'editable_svg': str(svg_path), 'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(), 'chemical_boxes': chemical_boxes, 'chemistry': m.chemistry, 'text_bounds': m.labels, 'arrows': m.arrows, 'largest_empty_square_mm': blank_mm, 'source_tables': list(c.TABLES)}
    (c.OUT / 'Figure_1_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    c.QA.append(report)
    return report
