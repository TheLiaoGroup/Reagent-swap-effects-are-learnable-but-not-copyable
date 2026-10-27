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

def figure1():
    records, _corpus, backgrounds = measurements()
    refs = references()
    m = Map()
    title = 'A measured reagent swap can reverse with substrate or background'
    node(m.root, 'title').text = title
    node(m.root, 'desc').text = 'Copying uses measured reference and target changes. Learning is shown only as an aggregate held-out result, without an individual learned prediction.'
    top = m.layer('01-reaction-and-condition-slot')
    m.text(top, 2, 11, 'Ar–Br', size=7)
    m.text(top, 13, 11, '+', size=7)
    m.chemical(top, figures14.BORONIC, 33, 10, 'anisylboronic-acid')
    m.line(top, (49, 11), (125, 11), width=0.25, arrow=True)
    m.text(top, 87, 8.7, 'X: PdCl₂(dtbpf) → Y: PdCl₂(PPh₃)₂', size=6.5, bold=True, anchor='middle')
    m.text(top, 87, 14.5, 'Cs₂CO₃ or K₂CO₃, 1,4-dioxane/H₂O,', anchor='middle', size=5.5)
    m.text(top, 87, 17, '100 °C, 12 h', size=5.5, anchor='middle')
    m.text(top, 130, 11, 'Ar–Ar′', size=7)
    m.text(top, 180, 4, '19 arrays', anchor='end', color=GRAY)
    m.text(top, 180, 9, '17 transformations', anchor='end', color=GRAY)
    m.text(top, 180, 14, '406,909 reactions', anchor='end', color=GRAY)

    def tag(group, x, y, label, numbers):
        m.text(group, x, y, label, size=5, color=GRAY)
        width = pymupdf.get_text_length(label, fontname='helv', fontsize=5) * PT
        m.text(group, x + width + 0.4, y - 1, ','.join((str(number) for number in numbers)), size=4.5, color=GRAY)
    tag(top, 49, 3.1, 'condition recommendation', refs['recommendation'])
    tag(top, 91, 3.1, 'general conditions', refs['general'])
    left = m.layer('02-related-substrates')
    middle = m.layer('03-reference-and-target')
    rules = m.layer('04-measured-well-grid')
    m.text(left, 1, 20.5, 'Related substrates, same base', size=6.5, bold=True)
    m.text(middle, 65, 20.5, 'Same pair, other base', size=6.5, bold=True)
    for lo, hi in ((0, 62), (64, 125)):
        m.line(rules, (lo, 23.5), (hi, 23.5), GRAY, 0.18)
        m.line(rules, (lo, 99.5), (hi, 99.5), GRAY, 0.18)
    m.line(rules, (63, 23.5), (63, 99.5), '#D7DADC', 0.15)
    m.line(rules, (1, 63), (61, 63), '#D7DADC', 0.15)
    for smiles, name, cy in ((figures14.QUINOLINE, '6-bromoquinoline', 41), (figures14.INDAZOLE, '5-bromo-1H-indazole', 78)):
        m.chemical(left, smiles, 11, cy - 1, name, left=0.5)
        m.text(left, 1, cy + 11, name)
        m.text(left, 1, cy + 16, 'Cs₂CO₃', color=GRAY)
        values = records.loc[records.substrate_smiles.eq(smiles)].sort_values('endpoint').yield_percent.tolist()
        m.pair(left, 23, cy - 4, values)
    m.chemical(middle, PYRIDINE, 94, 35, 'methoxybromopyridine')
    m.text(middle, 94, 46, '2-bromo-3-methoxypyridine', anchor='middle')
    m.text(middle, 66, 51, 'Cs₂CO₃ reference', color=GRAY, size=5.5)
    m.pair(middle, 73, 55, backgrounds['Cs'])
    m.text(middle, 66, 84.5, 'K₂CO₃ target', color=GRAY, size=5.5)
    m.pair(middle, 73, 88.5, backgrounds['K'])
    routes = m.layer('05-short-reference-to-target-routes')
    m.line(routes, (84, 65), (84, 80.5), ORANGE, 0.3, True, 'copy-reference-to-target')
    m.text(routes, 66, 72, 'copy: −28', size=5.5, color=ORANGE, bold=True)
    m.line(routes, (78, 75), (80, 77), ORANGE, 0.4)
    m.line(routes, (78, 77), (80, 75), ORANGE, 0.4)
    m.text(routes, 66, 79.5, 'measured +34', size=5, color=GRAY)
    m.line(routes, (96, 65), (96, 80.5), TEAL, 0.3, True, 'learn-reference-to-target')
    m.text(routes, 99, 71, 'learn from history', size=5.5, color=TEAL, bold=True)
    tag(routes, 99, 76, 'transfer learning', refs['transfer'])
    tag(routes, 99, 80, 'yield-change prediction', refs['prediction'])
    results = m.layer('06-held-out-aggregate')
    m.text(results, 130, 21, 'Using the same reference', size=6.5, bold=True)
    m.text(results, 130, 40, '17 screens, held-out pairs', color=GRAY)
    m.text(results, 130, 48, 'Error vs no reference', size=6)
    m.text(results, 130, 57, 'copy', color=ORANGE)
    m.text(results, 151, 57, '+42.4%', color=ORANGE, bold=True, size=7)
    m.text(results, 130, 65, 'learn', color=TEAL)
    m.text(results, 151, 65, '−13.7%', color=TEAL, bold=True, size=7)
    m.text(results, 130, 79, 'Missed or wrong directions', size=6)
    m.text(results, 130, 87, '18.6 → 13.6 per 100', size=7)
    c.FIG.mkdir(parents=True, exist_ok=True)
    svg_path = c.FIG / 'Figure_1.svg'
    ET.ElementTree(m.root).write(svg_path, encoding='utf-8', xml_declaration=True)
    grid_root = copy.deepcopy(m.root)
    for child in list(grid_root):
        if child.tag.endswith('}g') and child.get('id', '')[:2] not in {'02', '03', '04'}:
            grid_root.remove(child)
    with pymupdf.open(stream=ET.tostring(grid_root), filetype='svg') as vector:
        with pymupdf.open('pdf', vector.convert_to_pdf()) as pdf:
            raster = pdf[0].get_pixmap(dpi=300, alpha=False)
            marks = np.min(np.frombuffer(raster.samples, np.uint8).reshape(raster.height, raster.width, 3), axis=2) < 225
            yy, xx = np.where(marks)
            grid_fraction = (int(xx.max()) - int(xx.min()) + 1) * (int(yy.max()) - int(yy.min()) + 1) / marks.size
    if not 0.55 <= grid_fraction <= 0.65:
        raise ValueError(f'Measured well-grid bounds occupy {grid_fraction:.3%} of Figure 1')
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
                raise ValueError('Figure 1 has an excessive blank region')
            for label in m.labels:
                x, y, width, height = label['bounds_mm']
                if min(x, y) < -0.1 or x + width > W + 0.1 or y + height > H + 0.1:
                    raise ValueError(f"Figure 1 text outside canvas: {label['text']}")
            chemical_boxes = [dict(zip(('x_pt', 'y_pt', 'width_pt', 'height_pt'), [v / PT for v in item['bounds_mm']])) for item in m.chemistry]
            report = {'figure': 'Figure_1', 'title': title, 'title_visible': False, 'draft': False, 'width_mm': W, 'height_mm': H, 'panels': [], 'single_composite': True, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300, 'pixel_bounds_pass': True, 'raster_images': len(page.get_images()), 'vector_paths': len(page.get_drawings()), 'editable_svg': str(svg_path), 'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(), 'chemical_boxes': chemical_boxes, 'chemistry': m.chemistry, 'text_bounds': m.labels, 'arrows': m.arrows, 'largest_empty_square_mm': blank_mm, 'grid_bounds_mm': [0, 18.5, 125, 99.5], 'grid_allocation_fraction': 125 * 81 / (W * H), 'grid_measured_bounds_fraction': grid_fraction, 'source_tables': list(c.TABLES)}
    (c.OUT / 'Figure_1_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    c.QA.append(report)
    return report
