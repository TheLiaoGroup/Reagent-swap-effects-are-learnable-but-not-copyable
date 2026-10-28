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
W, H, PT = 183, 100, 25.4 / 72
BLUE, RED, ORANGE, TEAL = '#2F6F9F', '#B8434E', '#D9822B', '#0E7C74'
INK, GRAY = '#25292C', '#7D858B'


def node(parent, kind, **attributes):
    return ET.SubElement(parent, f'{{{SVG}}}{kind}', {key.replace('_', '-'): str(value) for key, value in attributes.items()})


class Map:
    def __init__(self, wireframe=False):
        self.root = ET.Element(f'{{{SVG}}}svg', width=f'{W}mm', height=f'{H}mm', viewBox=f'0 0 {W} {H}', version='1.1')
        self.labels, self.chemistry, self.arrows = [], [], []
        self.wireframe = bool(wireframe)

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
            # Font.text_length measures Unicode glyphs (not the Latin-1-only helper).
            run_width = pymupdf.Font('hebo' if bold else 'helv').text_length(value, fontsize=run_size) * PT
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
        if self.wireframe:
            node(group, 'rect', x=left, y=cy - height / 2, width=width, height=height,
                 fill='none', stroke=GRAY, stroke_width=0.18, stroke_dasharray='1 1')
        else:
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
        width = pymupdf.Font('helv').text_length(label, fontsize=7) * PT
        self.text(group, x + width + 0.5, y - 1.2, ','.join(str(number) for number in numbers), size=6, color=GRAY, citation=True)

    def curve(self, group, start, controls, end, color, identity):
        a, b = controls
        node(group, 'path', d=f'M {start[0]} {start[1]} C {a[0]} {a[1]} {b[0]} {b[1]} {end[0]} {end[1]}',
             fill='none', stroke=color, stroke_width=0.32, id=identity)
        delta = np.asarray(end, float) - b
        unit = delta / np.linalg.norm(delta)
        back = np.asarray(end) - 1.35 * unit
        perp = 0.48 * np.array([-unit[1], unit[0]])
        node(group, 'polygon', points=' '.join(f'{x},{y}' for x, y in [end, back + perp, back - perp]), fill=color)
        self.arrows.append({'id': identity, 'start_mm': list(start), 'end_mm': list(end),
                            'controls_mm': [list(a), list(b)], 'color': color})

    def wells(self, group, left, top, target=False):
        for i, label in enumerate(('X', 'Y')):
            x = left + i * 16
            node(group, 'rect', x=x, y=top, width=12, height=7, fill='#E9EDEF' if target else 'white',
                 stroke=GRAY, stroke_width=0.2)
            self.text(group, x + 6, top + 4.7, '?' if target else 'measured', anchor='middle')
            self.text(group, x + 6, top + 11, label, anchor='middle', color=GRAY)
        self.line(group, (left + 12.8, top + 3.5), (left + 15.2, top + 3.5), arrow=True)

    def plate(self, group, left, top):
        for offset in (2.2, 1.1, 0):
            node(group, 'rect', x=left + offset, y=top - offset, width=13, height=8,
                 rx=0.8, fill='white', stroke=TEAL, stroke_width=0.18,
                 stroke_opacity=0.45 if offset else 1)
        for r in range(3):
            for col in range(5):
                node(group, 'circle', cx=left + 1.8 + col * 2.35, cy=top + 1.7 + r * 2.3,
                     r=0.65, fill='none', stroke=TEAL, stroke_width=0.18)


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
        if name == 'Fig1_a_448_responses':
            # The complete flagship distribution belongs to Figure 2, not this map.
            return args[0]
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


def map_rescue_data():
    path = figures14.PAIRING / 'selected_pairing_repair_all_12_yields_and_conditions.csv'
    cells = c.frame(path).loc[lambda d: d.action_id.eq(figures14.AMIDE_ACTION)].copy()
    if len(cells) != 12 or cells.groupby('molecular_side').size().tolist() != [6, 6]:
        raise ValueError('The prescribed amide rescue requires all twelve measured wells')
    selected = cells.loc[cells.binary_level.eq(0) &
                         ((cells.molecular_side.eq('before') & cells.ternary_level.eq(0)) |
                          (cells.molecular_side.eq('after') & cells.ternary_level.isin([0, 2])))].copy()
    if len(selected) != 3:
        raise ValueError('Figure 1 amide relocation/rescue wells are not unique')
    c.source('Fig1_rescue_wells', selected, path,
             'Unchanged Figure 4A action; meta HBTU/no HOBt, para HBTU/no HOBt and para TPTU/no HOBt', 'yield fraction')
    before = selected.loc[selected.molecular_side.eq('before')].iloc[0]
    failure = selected.loc[selected.molecular_side.eq('after') & selected.ternary_level.eq(0)].iloc[0]
    rescue = selected.loc[selected.molecular_side.eq('after') & selected.ternary_level.eq(2)].iloc[0]
    if failure.yield_numeric != 0 or c.display_decimal(100 * rescue.yield_numeric, 0) != '93':
        raise ValueError('The prescribed measured rescue changed')
    views_path = figures14.RECIPE_OPTIONS / 'round29/option_scope_domains/all_collective_conflict_domains.csv'
    views = c.frame(views_path)
    resolved, conflicts = int(views.full_graph_rescues20.sum()), len(views)
    if (resolved, conflicts) != (285, 542):
        raise ValueError('The complete measured-alternative rescue population changed')
    return before, failure, rescue, resolved, conflicts, views_path


def map_inventory():
    path = figures23.E59 / 'full_edge_corpus_class_support_20.csv'
    inventory = c.frame(path).loc[lambda d: d.reaction_type.ne('finkelstein')].copy()
    if len(inventory) != 19 or int(inventory.raw_observation_rows.sum()) != 406909:
        raise ValueError('The nineteen-array historical inventory changed')
    inventory['transformation'] = inventory.reaction_type.replace({
        'buchwald-hartwig-cmc': 'buchwald-hartwig', 'suzuki-bpin': 'suzuki'})
    transformations = inventory.transformation.nunique()
    if transformations != 17:
        raise ValueError('The nineteen arrays no longer represent seventeen transformations')
    c.source('Fig1_history_scope', inventory, path,
             'All nineteen non-Finkelstein arrays; BH CMC and Suzuki BPin count within their parent transformations; retained measured raw records', 'counts')
    return len(inventory), transformations, int(inventory.raw_observation_rows.sum())


def figure1(wireframe=False):
    records, conflicts, testable, overall, _, _ = source_data()
    before, failure, rescue, resolved, rescue_total, rescue_path = map_rescue_data()
    n_arrays, n_transformations, n_records = map_inventory()
    copy_pct, learn_pct = float(overall.direct_raw_relative_pct), float(overall.semantic_full_raw_relative_pct)
    refs, m = references(), Map(wireframe=wireframe)
    title = 'Learning makes measured reagent swaps useful beyond their original recipe'
    node(m.root, 'title').text = title
    gap = m.layer('layer_gap')
    routes = m.layer('layer_routes')
    prior = m.layer('layer_prior_work')
    result = m.layer('layer_results')
    chemistry = m.layer('layer_chemistry')

    # The approved physical grid: A 0–44 mm, B 50–144 mm, C 150–183 mm.
    for x in (47, 147):
        m.line(gap, (x, 2), (x, 98), '#D7DADC', 0.3 * PT)
    for x, letter, label in [(0.5, 'A', 'One swap, opposite responses'),
                              (50.5, 'B', 'Copying hurts; learning helps')]:
        m.text(gap, x, 4, letter, size=8, bold=True)
        m.text(gap, x + 6, 4, label, size=7, bold=True)
    m.text(gap, 150.5, 4, 'C', size=8, bold=True)
    m.text(gap, 156, 4, 'Measured alternatives', size=7, bold=True)
    m.text(gap, 156, 8, 'rescue failures', size=7, bold=True)

    # A: a compact genuine Suzuki scheme and two native chemical witnesses.
    m.text(gap, 1, 13, 'Ar–Br + 4-MeOC₆H₄B(OH)₂')
    m.line(gap, (4, 19), (20, 19), arrow=True)
    m.text(gap, 23, 20, 'Ar–Ar′')
    m.text(gap, 1, 27, 'X: PdCl₂(dtbpf)')
    m.text(gap, 1, 31, 'Y: PdCl₂(PPh₃)₂')
    m.text(gap, 1, 36, 'Cs₂CO₃, dioxane/H₂O', color=GRAY)
    m.text(gap, 1, 40, '100 °C, 12 h', color=GRAY)
    for smiles, name, cy in [(figures14.QUINOLINE, '6-bromoquinoline', 52),
                              (figures14.INDAZOLE, '5-bromo-1H-indazole', 77)]:
        values = records.loc[records.substrate_smiles.eq(smiles)].sort_values('endpoint').yield_percent.to_numpy(float)
        if len(values) != 2:
            raise ValueError('Each chemical witness requires its two unchanged measured wells')
        m.chemical(chemistry, smiles, 0.5, cy, name)
        m.text(gap, 0.5, cy + 11, name)
        color = BLUE if values[1] > values[0] else RED
        for x, tag, value in [(23, 'X', values[0]), (34.5, 'Y', values[1])]:
            m.text(gap, x + 4.5, cy - 6.5, tag, color=GRAY, anchor='middle')
            node(gap, 'rect', x=x, y=cy - 4.5, width=9, height=7, fill='white', stroke=GRAY, stroke_width=0.2)
            m.text(result, x + 4.5, cy + 0.2, c.display_decimal(value, 0) + '%', anchor='middle')
        m.line(gap, (32.4, cy - 1), (34.0, cy - 1), arrow=True)
        m.text(result, 33, cy + 7.5, f'{values[1] - values[0]:+.0f} pp', size=8,
               color=color, bold=True, anchor='middle')
    m.text(result, 1, 92, f'{conflicts}/{testable} swaps reverse', bold=True)
    m.text(result, 1, 96, f'{n_arrays} arrays · Fig. 2', color=GRAY)

    # B: identical reference wells branch into two genuinely different uses.
    m.text(routes, 68, 29, 'Reference recipe', anchor='middle', bold=True)
    m.text(routes, 128, 29, 'Target recipe', anchor='middle', bold=True)
    m.wells(routes, 54, 33)
    m.wells(routes, 114, 33, target=True)
    m.text(routes, 128, 49, 'same substrate pair', anchor='middle', color=GRAY)
    m.line(routes, (82, 36.5), (86, 36.5), INK, 0.25)
    node(routes, 'circle', cx=86, cy=36.5, r=0.45, fill=INK)
    m.curve(routes, (86, 36.5), ((92, 18), (107, 18)), (113, 36.5), ORANGE, 'copy-route')
    m.text(routes, 98.5, 17, 'copy Δ', color=ORANGE, anchor='middle', bold=True)
    m.text(result, 98.5, 35, f'{copy_pct:+.1f}% error', size=8, color=ORANGE, anchor='middle', bold=True)
    m.line(routes, (86, 36.5), (86, 58), TEAL, 0.28)
    m.line(routes, (86, 58), (80, 58), TEAL, 0.28)
    m.line(routes, (80, 58), (80, 61), TEAL, 0.28, True, 'history-input')
    m.plate(routes, 74, 63)
    m.line(routes, (90.5, 67), (142.5, 67), TEAL, 0.28)
    m.line(routes, (142.5, 67), (142.5, 41), TEAL, 0.28, True, 'learn-route')
    m.text(result, 123, 73, f'{learn_pct:+.1f}% error', size=8, color=TEAL, anchor='middle', bold=True)
    m.text(routes, 81, 79, 'same swap, other pairs', anchor='middle', color=TEAL, bold=True)
    m.text(routes, 81, 85, f'{n_arrays} arrays · {n_transformations} transformations', anchor='middle', color=GRAY)
    m.text(routes, 81, 90, f'{n_records:,} records', anchor='middle', color=GRAY)
    m.text(result, 98, 96, 'vs history alone (Fig. 3A)', anchor='middle', color=GRAY)
    m.tag(prior, 109, 13, 'condition recommendation', refs['recommendation'])
    m.tag(prior, 57, 51, 'transfer learning', refs['transfer'])
    m.tag(prior, 57, 56, 'yield-change prediction', refs['prediction'])

    # C: the original Figure 4A edit, failure and directly measured alternative.
    m.tag(prior, 151, 16, 'general conditions', refs['general'])
    m.chemical(chemistry, before.reactant1_smiles, 155, 30, 'meta-methylthioaniline')
    m.text(gap, 166, 40, 'meta', anchor='middle', color=GRAY)
    m.line(gap, (166, 43), (166, 46), arrow=True)
    m.chemical(chemistry, failure.reactant1_smiles, 155, 55, 'para-methylthioaniline')
    m.text(gap, 166, 65, 'para', anchor='middle', color=GRAY)
    for x, recipe, value, color in [(150.5, 'HBTU', failure.yield_numeric, RED),
                                    (168, 'TPTU', rescue.yield_numeric, BLUE)]:
        m.text(gap, x + 6.75, 72, recipe, anchor='middle', bold=True)
        node(gap, 'rect', x=x, y=75, width=13.5, height=8, fill='white', stroke=color, stroke_width=0.3)
        m.text(result, x + 6.75, 80.3, c.display_decimal(100 * value, 0) + '%', size=8,
               anchor='middle', bold=True, color=color)
    m.line(gap, (164.8, 79), (167.2, 79), BLUE, 0.25, True)
    m.text(result, 150.5, 92, f'{resolved}/{rescue_total} conflicts resolved', bold=True)
    m.text(result, 150.5, 96, 'Fig. 4', color=GRAY)

    shown = pd.DataFrame([
        dict(quantity='conflicting swaps', value=conflicts, denominator=testable),
        dict(quantity='copy error change vs history alone (%)', value=copy_pct, denominator=None),
        dict(quantity='learned error change vs history alone (%)', value=learn_pct, denominator=None),
        dict(quantity='conflicts resolved by measured alternatives', value=resolved, denominator=rescue_total),
    ])
    c.source('Fig1_map_numbers', shown,
             [figures23.EDM / 'sparse_sign_feasibility_summary.csv',
              figures23.REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv', rescue_path],
             'Unchanged counts from Figures 2 and 4 and held-out hierarchical error changes from Figure 3', 'count or percent')
    return save(m, title, 'Figure_1_wireframe' if wireframe else 'Figure_1')


def save(m, title, stem='Figure_1'):
    for i, a in enumerate(m.labels):
        for b in m.labels[i + 1:]:
            ax, ay, aw, ah = a['bounds_mm']
            bx, by, bw, bh = b['bounds_mm']
            if min(ax + aw, bx + bw) - max(ax, bx) > 0.2 and min(ay + ah, by + bh) - max(ay, by) > 0.2:
                raise ValueError(f"Figure 1 labels overlap: {a['text']} / {b['text']}")
    c.FIG.mkdir(parents=True, exist_ok=True)
    path = c.FIG / f'{stem}.svg'
    ET.ElementTree(m.root).write(path, encoding='utf-8', xml_declaration=True)
    with pymupdf.open(stream=ET.tostring(m.root), filetype='svg') as source:
        with pymupdf.open('pdf', source.convert_to_pdf()) as pdf:
            pdf.set_metadata({'title': title, 'creator': 'M1'})
            pdf.save(c.FIG / f'{stem}.pdf')
            page = pdf[0]
            pix = page.get_pixmap(dpi=300, alpha=False)
            pix.save(c.FIG / f'{stem}.png')
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
            report = {'figure': 'Figure_1', 'title': title, 'title_visible': False, 'draft': m.wireframe, 'width_mm': W, 'height_mm': H,
                      'panels': [], 'single_composite': True, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300,
                      'pixel_bounds_pass': True, 'raster_images': len(page.get_images()), 'vector_paths': len(page.get_drawings()),
                      'editable_svg': str(path), 'svg_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'chemical_boxes': chemical_boxes,
                      'chemistry': m.chemistry, 'text_bounds': m.labels, 'arrows': m.arrows, 'largest_empty_square_mm': blank_mm,
                      'source_tables': list(c.TABLES)}
    intersections = c.structure_box_intersections(report)
    report['layout_review'] = {'structure_structure_box_intersections': intersections}
    if intersections:
        raise ValueError(f'Figure 1 chemical structures overlap: {intersections}')
    (c.OUT / f'{stem}_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    c.QA.append(report)
    return report
