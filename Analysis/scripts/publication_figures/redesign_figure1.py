"""Reagent-swap map with distinct chemical and matched-reference comparisons."""
from __future__ import annotations

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
W, H, PT = 183, 110, 25.4 / 72
BLUE, RED, ORANGE, TEAL = '#2F6F9F', '#B8434E', '#D9822B', '#0E7C74'
INK, GRAY = '#25292C', '#7D858B'


def node(parent, kind, **attributes):
    return ET.SubElement(parent, f'{{{SVG}}}{kind}', {key.replace('_', '-'): str(value) for key, value in attributes.items()})


class Map:
    def __init__(self):
        self.root = ET.Element(f'{{{SVG}}}svg', width=f'{W}mm', height=f'{H}mm', viewBox=f'0 0 {W} {H}', version='1.1')
        self.labels, self.chemistry, self.arrows = [], [], []

    def layer(self, name):
        group = node(self.root, 'g', id=name)
        group.set(f'{{{INKSCAPE}}}groupmode', 'layer')
        group.set(f'{{{INKSCAPE}}}label', name)
        return group

    def text(self, group, x, y, text, size=7, color=INK, anchor='start', bold=False, citation=False):
        text = c.numeric_minus(text)
        if len(text.split()) > 6:
            raise ValueError(f'Figure 1 label exceeds six words: {text}')
        if size < (6 if citation else 7):
            raise ValueError(f'Figure 1 label too small: {text}')
        runs = []
        for part in re.split('([\u2080-\u2089]+)', text):
            if not part:
                continue
            subscript = all('\u2080' <= char <= '\u2089' for char in part)
            drop = size * PT * 0.22 if subscript else 0
            value = part.translate(str.maketrans('\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089', '0123456789'))
            run_size = size * (0.75 if subscript else 1)
            run_width = pymupdf.get_text_length(value, fontname='hebo' if bold else 'helv', fontsize=run_size) * PT
            runs.append((value, subscript, drop, run_size, run_width))
        width = sum(run[4] for run in runs)
        left = x - {'start': 0, 'middle': width / 2, 'end': width}[anchor]
        label = node(group, 'text', fill=color, font_family='Arial', text_anchor='start', font_weight='bold' if bold else 'normal')
        cursor = left
        for value, subscript, drop, run_size, run_width in runs:
            span = node(label, 'tspan', x=cursor, y=y + drop, font_size=run_size * PT)
            if subscript:
                span.set('data-script', 'sub')
            span.text = value
            cursor += run_width
        self.labels.append({'text': text, 'bounds_mm': [left, y - size * PT * 0.8, width, size * PT], 'font_pt': size})

    def line(self, group, start, end, color=INK, width=0.22, arrow=False, identity=None):
        node(group, 'path', d=f'M {start[0]} {start[1]} L {end[0]} {end[1]}', fill='none', stroke=color,
             stroke_width=width, id=identity or '')
        if arrow:
            delta = np.array(end, float) - start
            unit = delta / np.linalg.norm(delta)
            back = np.array(end) - 1.35 * unit
            perp = 0.48 * np.array([-unit[1], unit[0]])
            node(group, 'polygon', points=' '.join(f'{x},{y}' for x, y in [end, back + perp, back - perp]), fill=color)
            self.arrows.append({'id': identity, 'start_mm': list(start), 'end_mm': list(end), 'color': color})

    def chemical(self, group, smiles, left, cy, name):
        entry = c.CHEM.get(smiles)
        if not entry['native'] or entry['inchi_key'] != entry['roundtrip_inchi_key']:
            raise ValueError('Figure 1 requires an identity-checked native vector')
        raw = Path(entry['svg'])
        root = ET.fromstring(raw.read_text(encoding='utf-8-sig'))
        if root.get('data-m1-native-scale') != '1':
            raise ValueError('Native chemical scale was not normalized')
        box = list(map(float, root.get('viewBox').split()))
        scale = 25.4 / 96
        width, height = box[2] * scale, box[3] * scale
        fragment = node(group, 'g', id=f'chem-{name}', transform=f'translate({left} {cy - height / 2}) scale({scale})')
        for child in root:
            fragment.append(copy.deepcopy(child))
        assets = c.OUT / 'materials/figure1'
        assets.mkdir(parents=True, exist_ok=True)
        for extension in ('svg', 'cdxml', 'cdx'):
            shutil.copy2(entry[extension], assets / f'{name}.{extension}')
        self.chemistry.append({'smiles': smiles, 'inchi_key': entry['inchi_key'], 'bounds_mm': [left, cy - height / 2, width, height],
                               'bond_length_pt': 10.8, 'label_size_pt': 6, 'native_scale': json.loads(Path(entry['scale_metadata']).read_text()),
                               'source_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()})

    def tag(self, group, x, y, label, numbers):
        self.text(group, x, y, label, color=GRAY)
        width = pymupdf.get_text_length(label, fontname='helv', fontsize=7) * PT
        self.text(group, x + width + 0.5, y - 1.2, ','.join(str(number) for number in numbers), size=6, color=GRAY, citation=True)


def references():
    return {'recommendation': [16, 17], 'general': [22, 23], 'transfer': [19], 'prediction': [20]}


def suzuki_training_data():
    analysis = c.A / 'scripts/reference_learning/Analysis'
    relation_path = figures23.METRICS / 'learned_training_relationships.csv'
    registry_path = analysis / 'tables/chemical_observation_channels_target_registry.csv'
    sample_path = analysis / 'tables/chemical_mode_training_sample_registry.csv'
    matrix_path = analysis / 'data/matrix_Suzuki_1.npz'
    relationships = c.frame(relation_path)
    selected = relationships.loc[
        relationships.screen.eq('Suzuki|1') & relationships.edge_id.eq('0e47d3fbb415d7a8')
        & relationships.split.eq('pair') & relationships['repeat'].eq(0) & relationships.fraction.eq(1.0)
    ].sort_values('fold')
    if len(selected) != 5 or selected.fold.tolist() != list(range(5)):
        raise ValueError('Stored flagship Suzuki relationships changed')
    coefficient = selected.iloc[0]
    if json.loads(coefficient.swap_key)[-2:] != ['CMP0092', 'CMP0097']:
        raise ValueError('Figure 1 learned relationship uses a different catalyst swap')
    registry = c.frame(registry_path)
    target = registry.loc[registry.screen.eq(coefficient.screen) & registry.edge_id.eq(coefficient.edge_id)]
    sample = c.frame(sample_path)
    for field in ('screen', 'split', 'repeat', 'fold', 'fraction'):
        sample = sample.loc[sample[field].eq(coefficient[field])]
    if len(target) != 1 or len(sample) != 1:
        raise ValueError('Stored Suzuki illustration is not unique')
    target = target.iloc[0]
    anchors = [int(target.reference_condition_a_index), int(target.reference_condition_b_index)]
    endpoints = [int(target.condition_a_index), int(target.condition_b_index)]
    indices = np.asarray(json.loads(sample.iloc[0].training_indices_json), int)
    tests = np.asarray(json.loads(sample.iloc[0].test_indices_json), int)
    if np.intersect1d(indices, tests).size:
        raise ValueError('Illustration training and target folds overlap')
    with np.load(matrix_path, allow_pickle=False) as saved:
        observed = saved['M'][indices]
        identities = saved['sps'].astype(str)[indices]
        conditions = saved['conds'].astype(str)
    if conditions[anchors + endpoints].tolist() != ['C27', 'C39', 'C23', 'C26']:
        raise ValueError('Stored Suzuki matrix condition identities changed')
    finite = np.isfinite(observed[:, anchors + endpoints]).all(axis=1)
    points = pd.DataFrame({
        'pair_index': indices[finite], 'substrate_pair': identities[finite],
        'reference_contrast_pp': 100 * int(target.reference_orientation) * (observed[finite, anchors[0]] - observed[finite, anchors[1]]),
        'target_contrast_pp': 100 * int(target.orientation) * (observed[finite, endpoints[0]] - observed[finite, endpoints[1]]),
    })
    c.source('Fig1_history_pairs', points, [matrix_path, registry_path, sample_path],
             'Flagship Suzuki catalyst swap; frozen pair split, repeat 0, lowest fold; training indices only; no fit', 'yield change pp')
    c.source('Fig1_history_relation', selected.iloc[:1], relation_path,
             'Stored full S+D relationship for C27/C39 reference to C23/C26 target; centred reference S=0 section; no fit', 'slope and intercept pp')
    return points, coefficient


def source_data():
    original = c.source
    names = {'Fig2_c_array_conflicts': 'Fig1_array_conflicts', 'Fig3_a_screen_comparison': 'Fig1_overall_error_change',
             'Fig3_c_training_pairs': 'Fig1_history_pairs', 'Fig3_c_training_relation': 'Fig1_history_relation'}

    def source(name, *args, **kwargs):
        return original(names.get(name, name.replace('Fig1_a_', 'Fig1_', 1)), *args, **kwargs)
    c.source = source
    try:
        records, _ = figures14.flagship_data()
        signs = figures23.sign_data()
        shown = figures23.main_screen_data()
        points, coefficient = None, None
    finally:
        c.source = original
    zero = signs.loc[signs.threshold_pp.eq(0)]
    if shown.iloc[0].screen != 'Overall':
        raise ValueError('Overall comparison must be the first row')
    return records, int(zero.n_conflicting_edges.sum()), int(zero.n_cycle_edges.sum()), shown.iloc[0], points, coefficient


def selected_example():
    from publication_figures.select_fig1_example import MANIFEST_PATH, POPULATION_PATH, REGISTRY_PATH, select_example
    original = c.source

    def source(name, *args, **kwargs):
        return original('Fig1_example_wells' if name == 'S5_source_well_audit' else name, *args, **kwargs)

    c.source = source
    try:
        row, cells, selection = select_example(write_source=False)
    finally:
        c.source = original
    if (row.screen, row.edge_id, int(row.pair_index)) != ('Suzuki|1', '8f3d681cf95d064e', 10):
        raise ValueError('The predeclared S4 example changed')
    reference = cells.loc[cells.background.eq('Reference')].sort_values('endpoint_position')
    target = cells.loc[cells.background.eq('Target')].sort_values('endpoint_position')
    mean = float(reference.yield_percent.mean())
    values = row.to_frame().T.copy()
    values['reference_mean_yield_percent'] = mean
    paths = [Path(entry['path']) for entry in selection['source_well_audit']['sources']]
    matrix = c.A / 'scripts/reference_learning/Analysis/data/matrix_Suzuki_1.npz'
    prediction_paths = list(dict.fromkeys([POPULATION_PATH, MANIFEST_PATH, REGISTRY_PATH, matrix, *paths]))
    original('Fig1_example_prediction', values, prediction_paths,
             'The unchanged predeclared median-error selection shared with Figure S4', 'yield percentage points')
    original('Fig1_example_inputs', cells, paths,
             'The four measured endpoint wells of the unchanged Figure S4 example', 'measured yield %')
    return row, reference, target, mean


def figure1():
    records, conflicts, testable, overall, _, _ = source_data()
    example, reference, target, mean = selected_example()
    copy_pct, learn_pct = float(overall.direct_raw_relative_pct), float(overall.semantic_full_raw_relative_pct)
    refs, m = references(), Map()
    title = 'The same measured reagent swap becomes useful through learning'
    node(m.root, 'title').text = title

    left = m.layer('01-chemical-reversal')
    m.text(left, 1, 7, 'A', size=8, bold=True)
    m.text(left, 7, 7, 'One swap, opposite responses', size=8, bold=True)
    m.text(left, 2, 17, 'Ar\u2013Br')
    m.text(left, 15, 17, '+')
    m.chemical(left, figures14.BORONIC, 20, 16, 'anisylboronic-acid')
    m.line(left, (44, 17), (50, 17), arrow=True)
    m.text(left, 51.5, 17, 'Ar\u2013Ar\u2032')
    m.text(left, 2, 27, 'X: PdCl\u2082(dtbpf)')
    m.text(left, 2, 32, 'Y: PdCl\u2082(PPh\u2083)\u2082')
    m.text(left, 2, 38, 'Cs\u2082CO\u2083, dioxane/H\u2082O', color=GRAY)
    m.text(left, 38, 38, '100 \u00b0C, 12 h', color=GRAY)
    for smiles, name, y in [(figures14.QUINOLINE, '6-bromoquinoline', 52), (figures14.INDAZOLE, '5-bromo-1H-indazole', 78)]:
        values = records.loc[records.substrate_smiles.eq(smiles)].sort_values('endpoint').yield_percent.to_numpy(float)
        m.chemical(left, smiles, 2, y, name)
        m.text(left, 2, y + 12, name)
        color = BLUE if values[1] > values[0] else RED
        for x, tag, value in [(34, 'X', values[0]), (48, 'Y', values[1])]:
            m.text(left, x + 5.75, y - 6.7, tag, color=GRAY, anchor='middle')
            node(left, 'rect', x=x, y=y - 5, width=11.5, height=7, fill='white', stroke=GRAY, stroke_width=0.2)
            m.text(left, x + 5.75, y - 0.3, c.display_decimal(value, 2) + '%', anchor='middle')
        m.text(left, 48, y + 7.5, f'{values[1] - values[0]:+.0f} pp', size=8, color=color, bold=True, anchor='middle')
    m.text(left, 2, 102, f'{conflicts}/{testable} sign conflicts \u00b7 19 arrays', bold=True)
    m.line(left, (64, 4), (64, 107), '#D7DADC', 0.2)

    shared = m.layer('02-identical-measured-reference')
    m.text(shared, 68, 7, 'B', size=8, bold=True)
    m.text(shared, 74, 7, 'Same wells, different use', size=8, bold=True)
    m.text(shared, 68, 14, 'Choose the target recipe', bold=True)
    m.tag(shared, 68, 19.5, 'condition recommendation', refs['recommendation'])
    m.tag(shared, 110, 19.5, 'general conditions', refs['general'])
    first, second = [value.strip() for value in example.substrate_pair.split('>>')]
    m.chemical(shared, first, 70, 33, 'example-heteroaryl-bromide')
    m.text(shared, 94, 34, '+')
    m.chemical(shared, second, 102, 33, 'example-boronic-acid')
    m.text(shared, 68, 44, 'K\u2083PO\u2084 reference \u00b7 same substrate pair', color=GRAY)
    for x, label, value in [(69, 'PdCl\u2082(dtbpf)', reference.yield_percent.iloc[0]),
                            (110, 'XPhos Pd G3', reference.yield_percent.iloc[1])]:
        m.text(shared, x + 14, 50, label, anchor='middle')
        node(shared, 'rect', x=x, y=52, width=28, height=8, fill='white', stroke=INK, stroke_width=0.22)
        m.text(shared, x + 14, 57.5, c.display_decimal(value, 2) + '%', size=8, anchor='middle')
    m.text(shared, 69, 69, f'Change {float(example.reference_contrast_pp):+.2f} pp')
    m.text(shared, 110, 69, f'Mean {mean:.2f}%')
    m.line(shared, (83, 60.5), (83, 62), GRAY, 0.2)
    m.line(shared, (124, 60.5), (124, 62), GRAY, 0.2)
    m.line(shared, (83, 62), (124, 62), GRAY, 0.2)
    m.line(shared, (103.5, 62), (83, 65), GRAY, 0.2, True)
    m.line(shared, (103.5, 62), (124, 65), GRAY, 0.2, True)

    routes = m.layer('03-interpret-the-same-measurement')
    m.line(routes, (82, 72), (82, 86), ORANGE, 0.4, True, 'copy-example')
    m.line(routes, (102, 73), (102, 88), TEAL, 0.35)
    m.line(routes, (102, 88), (110, 88), TEAL, 0.35, True, 'learn-example')
    m.line(routes, (125, 71), (125, 72.5), TEAL, 0.35, True)
    m.text(routes, 106, 76, 'Learn from history', color=TEAL, bold=True)
    m.tag(routes, 106, 81, 'transfer learning', refs['transfer'])
    m.tag(routes, 106, 85.5, 'yield-change prediction', refs['prediction'])
    for x, label, value, color in [
        (81, 'Copy', example.direct_raw_prediction_pp, ORANGE),
        (110, 'Learn', example.semantic_full_raw_prediction_pp, TEAL),
        (138, 'Measured', example.target_contrast_pp, INK),
    ]:
        m.text(routes, x, 93, label, color=color, bold=True, anchor='middle')
        m.text(routes, x, 101, f'{float(value):+.2f}', size=9, color=color, bold=True, anchor='middle')
    m.text(routes, 108, 107, 'Cs\u2082CO\u2083 target change (pp)', anchor='middle', color=GRAY)

    summary = m.layer('04-all-held-out-targets')
    m.line(summary, (150, 4), (150, 107), '#D7DADC', 0.2)
    m.text(summary, 153, 14, 'All held-out targets', bold=True)
    m.text(summary, 153, 20, '17 screens (Fig. 3A)', color=GRAY)
    m.text(summary, 153, 34, 'Error vs history alone', color=GRAY)
    m.text(summary, 153, 47, 'Copy', color=ORANGE, bold=True)
    m.text(summary, 153, 56, f'+{copy_pct:.1f}%', size=10, color=ORANGE, bold=True)
    m.text(summary, 153, 75, 'Learn', color=TEAL, bold=True)
    m.text(summary, 153, 84, f'{learn_pct:.1f}%', size=10, color=TEAL, bold=True)
    m.text(summary, 153, 98, 'MSE change (%)', color=GRAY)

    shown = pd.DataFrame([
        dict(quantity='conflicting swaps', value=conflicts, denominator=testable),
        dict(quantity='copy error change vs history alone (%)', value=copy_pct, denominator=None),
        dict(quantity='learned error change vs history alone (%)', value=learn_pct, denominator=None),
    ])
    c.source('Fig1_map_numbers', shown,
             [figures23.EDM / 'sparse_sign_feasibility_summary.csv',
              figures23.REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv'],
             'Sign-conflict counts from Figure 2 and held-out hierarchical error changes from Figure 3', 'count or percent')
    return save(m, title)


def save(m, title):
    for i, a in enumerate(m.labels):
        for b in m.labels[i + 1:]:
            ax, ay, aw, ah = a['bounds_mm']
            bx, by, bw, bh = b['bounds_mm']
            if min(ax + aw, bx + bw) - max(ax, bx) > 0.2 and min(ay + ah, by + bh) - max(ay, by) > 0.2:
                raise ValueError(f"Figure 1 labels overlap: {a['text']} / {b['text']}")
    c.FIG.mkdir(parents=True, exist_ok=True)
    path = c.FIG / 'Figure_1.svg'
    ET.ElementTree(m.root).write(path, encoding='utf-8', xml_declaration=True)
    with pymupdf.open(stream=ET.tostring(m.root), filetype='svg') as source:
        with pymupdf.open('pdf', source.convert_to_pdf()) as pdf:
            pdf.set_metadata({'title': title, 'creator': 'M1'})
            pdf.save(c.FIG / 'Figure_1.pdf')
            page = pdf[0]
            pix = page.get_pixmap(dpi=300, alpha=False)
            pix.save(c.FIG / 'Figure_1.png')
            pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            distance = distance_transform_cdt(np.pad(np.min(pixels, axis=2) >= 225, 1, constant_values=False), metric='chessboard')
            blank_mm = (2 * int(distance.max()) - 1) * 25.4 / 300
            if blank_mm > W / 6:
                raise ValueError(f'Figure 1 excessive blank region: {blank_mm:.1f} mm')
            for label in m.labels:
                x, y, width, height = label['bounds_mm']
                if min(x, y) < -0.1 or x + width > W + 0.1 or y + height > H + 0.1:
                    raise ValueError(f"Figure 1 text outside canvas: {label['text']}")
            chemical_boxes = [dict(zip(('x_pt', 'y_pt', 'width_pt', 'height_pt'), [v / PT for v in item['bounds_mm']])) for item in m.chemistry]
            report = {'figure': 'Figure_1', 'title': title, 'title_visible': False, 'draft': False, 'width_mm': W, 'height_mm': H,
                      'panels': [], 'single_composite': True, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300,
                      'pixel_bounds_pass': True, 'raster_images': len(page.get_images()), 'vector_paths': len(page.get_drawings()),
                      'editable_svg': str(path), 'svg_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'chemical_boxes': chemical_boxes,
                      'chemistry': m.chemistry, 'text_bounds': m.labels, 'arrows': m.arrows, 'largest_empty_square_mm': blank_mm,
                      'source_tables': list(c.TABLES)}
    intersections = c.structure_box_intersections(report)
    report['layout_review'] = {'structure_structure_box_intersections': intersections}
    if intersections:
        raise ValueError(f'Figure 1 chemical structures overlap: {intersections}')
    (c.OUT / 'Figure_1_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    c.QA.append(report)
    return report
