"""Final physical layout, source tables and native chemical vectors."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd
import pymupdf
A = Path(__file__).resolve().parents[2]
ROOT = A.parent
OUT = A / 'out/final_figures'
DATA = OUT / 'source_tables'
FIG = ROOT / 'Submission/02_Figures'
RED, TEAL, BLUE = ('#B8434E', '#0E7C74', '#2F6F9F')
GRAY, INK, LIGHT = ('#89939B', '#25292C', '#E9EDEF')
GOLD = '#D9822B'
TABLES = []
QA = []
CHEM = None
sys.path.insert(0, str(A / 'scripts'))
from m1_chemistry import normalize_native_svg
from m1_figure_text import numeric_minus
from m1_reader_labels import reader_table, reader_text
CLASS_NAMES = {'acylation': 'Amide coupling', 'amide coupling': 'Amide coupling', 'buchwald-hartwig': 'Buchwald–Hartwig', 'bh': 'Buchwald–Hartwig', 'buchwald-hartwig-cmc': 'Buchwald–Hartwig (CMC)', 'bh cmc': 'Buchwald–Hartwig (CMC)', 'buchwald-hartwig(cmc)': 'Buchwald–Hartwig (CMC)', 'buchwald–hartwig(cmc)': 'Buchwald–Hartwig (CMC)', 'allylic-substitution': 'Allylic substitution', 'c-o-competitive': 'C–O competition', 'c-o competitive reactions': 'C–O competition', 'chan-lam': 'Chan–Lam', 'pd-carbonylation': 'Pd carbonylation', 'reductive-amination': 'Reductive amination', 'snar': 'SNAr', 'suzuki-bpin': 'Suzuki BPin'}

def class_label(raw):
    value = str(raw).strip()
    return CLASS_NAMES.get(value.lower(), value[:1].upper() + value[1:])

def screen_label(raw):
    value = str(raw).strip()
    if '|' in value:
        name, number = value.rsplit('|', 1)
        return f'{class_label(name)} {number}'
    return class_label(value)

def setup():
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 6, 'axes.titlesize': 7, 'axes.titleweight': 'bold', 'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5, 'axes.linewidth': 0.5, 'lines.linewidth': 0.7, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True, 'savefig.facecolor': 'white'})

def frame(path):
    path = Path(path)
    data = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    if data.empty:
        raise ValueError(f'Empty source: {path}')
    return data

def newfig(height_mm=160, width_mm=180):
    setup()
    if not 0 < width_mm <= 183 or not 0 < height_mm <= 247:
        raise ValueError('Figure exceeds approved physical dimensions')
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=180)
    fig._m1_chemistry, fig._m1_panels = ([], [])
    return fig

def panel(fig, rect, letter, label='', title_x=None):
    ax = fig.add_axes(rect)
    x, y, w, h = rect
    tx = x if title_x is None else title_x
    title = fig.text(tx, y + h + 0.018, label, fontsize=6, weight='bold', va='bottom')
    letter_text = fig.text(tx - 0.026, y + h + 0.018, letter.upper(), fontsize=8, weight='bold', va='bottom')
    ax._draft_panel_title = title
    ax._draft_panel_letter = letter_text
    fig._m1_panels.append((letter, ax, title))
    return ax

def clean(ax, xlabel='', ylabel=''):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax

def source_pointer(path):
    original_path, path = (Path(path), Path(path).resolve())
    if path.is_relative_to(A.resolve()):
        return path.relative_to(A.resolve()).as_posix()
    from m1_exclusions import load_exclusions
    if path == load_exclusions().source_path.resolve():
        return 'external-input:M1_EXCLUDE_CSV'
    elif original_path.is_absolute() and path.parent == reader_tables.data_dir().resolve() and reader_tables.is_source(path) and reader_tables.source_is_file(path):
        return (Path('data/raw_zero_rule/01_原始数据') / path.name).as_posix()
    raise ValueError(f'Unregistered external Source Data input: {path}')

def source(name, data, path, selector, unit='', note=''):
    if len(data) == 0:
        raise ValueError(f'No rows selected: {name}')
    data = reader_table(data)
    selector, note = (reader_text(selector), reader_text(note))
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    if 'unit' not in data:
        data['unit'] = unit
    data['provenance_source_file'] = '; '.join((source_pointer(p) for p in paths))
    data['source_filter'] = selector
    if note:
        data['definition'] = note
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / f'{name}.csv'
    data.to_csv(target, index=False)
    TABLES.append({'name': name, 'csv': str(target), 'rows': len(data), 'source': str(paths[0]), 'selector': selector, 'numeric_columns': list(data.select_dtypes(include='number').columns), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'inputs': [{'path': str(p), 'sha256': reader_tables.source_evidence(p)['sha256'] if reader_tables.is_source(p) else hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths], 'selection': selector, 'unit': unit, 'note': note})
    return data

def molecule(fig, rect, smiles, highlight_methyl=False, template=None, halo=False):
    if CHEM is None:
        raise RuntimeError('Native chemistry has not been initialized')
    entry = CHEM.get(smiles, bool(highlight_methyl), template)
    if not entry['native'] or entry['inchi_key'] != entry['roundtrip_inchi_key']:
        raise ValueError('Unverified chemical identity')
    raw = Path(entry.get('native_svg', entry['svg']))
    cdxml = Path(entry['cdxml'])
    svg, metadata = normalize_native_svg(raw.read_text(encoding='utf-8-sig'), cdxml.read_text(encoding='utf-8-sig'))
    if halo or highlight_methyl:
        root = ET.fromstring(svg)
        for elem in root.iter():
            for attr in ('fill', 'stroke'):
                if elem.get(attr) not in (None, 'none', '#000000', 'black'):
                    elem.set(attr, '#000000')
        doc = ET.fromstring(cdxml.read_text(encoding='utf-8-sig'))
        nodes = {a.get('id'): a for a in doc.iter('n') if a.get('p')}
        bonds = list(doc.iter('b'))
        degree = {k: sum((k in (b.get('B'), b.get('E')) for b in bonds)) for k in nodes}
        eligible = [a for k, a in nodes.items() if a.get('Element', '6') == '6' and degree[k] == 1]
        if not eligible:
            raise ValueError('Halo has no terminal carbon')
        marked_bonds = [b for b in bonds if b.get('color') not in (None, '0', '1')]
        marked_ids = {b.get(k) for b in marked_bonds for k in ('B', 'E')}
        choices = [a for a in eligible if a.get('id') in marked_ids] or eligible
        atom = choices[0]
        x, y = map(float, atom.get('p').split())
        rawroot = ET.fromstring(raw.read_text(encoding='utf-8-sig'))
        path = next((a for a in rawroot if a.tag.split('}')[-1] == 'path'))
        matrix = [float(v) for v in path.get('transform')[7:-1].split()]
        f = metadata['normalization_factor']
        cx, cy = ((20 * x * matrix[0] + matrix[4]) * f, (20 * y * matrix[3] + matrix[5]) * f)
        circle = ET.Element('{http://www.w3.org/2000/svg}circle', {'cx': str(cx), 'cy': str(cy), 'r': '4.2', 'fill': '#BA4852', 'fill-opacity': '.24'})
        if halo:
            root.insert(0, circle)
            vx, vy, vw, vh = map(float, root.get('viewBox').split())
            r = 4.2 + 0.6
            nx, ny = min(vx, cx - r), min(vy, cy - r)
            nw, nh = max(vx + vw, cx + r) - nx, max(vy + vh, cy + r) - ny
            if (nx, ny, nw, nh) != (vx, vy, vw, vh):
                for attr, old_size, new_size in (('width', vw, nw), ('height', vh, nh)):
                    value = root.get(attr)
                    if value:
                        number = float(''.join(ch for ch in value if ch.isdigit() or ch in '.-'))
                        unit = ''.join(ch for ch in value if ch.isalpha())
                        root.set(attr, f'{number * new_size / old_size:.4f}{unit}')
                root.set('viewBox', f'{nx} {ny} {nw} {nh}')
        svg = ET.tostring(root, encoding='unicode')
    work = OUT / 'render_work'
    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((svg + str(halo)).encode()).hexdigest()[:14]
    target = work / f'chem_{key}.pdf'
    normalized_svg = work / f'chem_{key}.svg'
    normalized_svg.write_text(svg, encoding='utf-8')
    if not target.is_file():
        with pymupdf.open(stream=svg.encode(), filetype='svg') as vector:
            with pymupdf.open('pdf', vector.convert_to_pdf()) as pdf:
                pdf.save(target)
    materials = OUT / 'materials' / 'chemistry'
    materials.mkdir(parents=True, exist_ok=True)
    asset_stem = materials / f'chem_{key}'
    exported = {}
    for kind in ('cdx', 'cdxml'):
        copied = asset_stem.with_suffix(f'.{kind}')
        shutil.copy2(entry[kind], copied)
        exported[kind] = str(copied)
    copied_svg = asset_stem.with_suffix('.svg')
    copied_svg.write_text(svg, encoding='utf-8')
    exported['svg'] = str(copied_svg)
    fig._m1_chemistry.append({'rect': list(rect), 'pdf': str(target), 'smiles': smiles, 'normalized_svg': str(normalized_svg), 'editable_assets': exported, 'inchi_key': entry['inchi_key'], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'raw_svg_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'cdxml_sha256': hashlib.sha256(cdxml.read_bytes()).hexdigest(), 'halo': bool(halo), 'native_scale': metadata})

def display_decimal(value, precision=2):
    """Match the pinned-number ledger's decimal half-up display policy."""
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP))

def formatted(value):
    value = re.sub('\\bSNAr\\b', lambda _: 'S$_\\mathrm{N}$Ar', value)
    for old, new in (('Buchwald–Hartwig CMC', 'Buchwald–Hartwig (CMC)'), ('BH CMC', 'Buchwald–Hartwig (CMC)'), ('Acylation', 'Amide coupling')):
        value = value.replace(old, new)
    for old, new in (('PdCl2(PPh3)2', 'PdCl₂(PPh₃)₂'), ('PdCl2(PCy3)2', 'PdCl₂(PCy₃)₂'), ('PdCl2', 'PdCl₂'), ('Cs2CO3', 'Cs₂CO₃'), ('K2CO3', 'K₂CO₃'), ('K3PO4', 'K₃PO₄'), ('Na2CO3', 'Na₂CO₃')):
        value = value.replace(old, new)
    for i, digit in enumerate('₀₁₂₃₄₅₆₇₈₉'):
        value = value.replace(digit, f'$_{i}$')
    return numeric_minus(value)

def normalize_snar_svg_reading_order(root):
    """Keep fully positioned SNAr mathtext glyphs in their visual reading order."""
    changed = 0
    for label in root.iter('{http://www.w3.org/2000/svg}text'):
        spans = list(label)
        if not spans or (label.text or '').strip() or any((span.tag != '{http://www.w3.org/2000/svg}tspan' or len(span) or span.get('dx') is not None or (span.get('dy') is not None) for span in spans)):
            continue
        try:
            positions = {span: (float(span.get('x')), float(span.get('y'))) for span in spans}
        except (TypeError, ValueError):
            continue
        ordered = sorted(spans, key=lambda span: positions[span][0])
        value = re.sub('\\s', '', ''.join((span.text or '' for span in ordered)))
        if not re.fullmatch('SNAr\\d*', value) or ordered == spans:
            continue
        label[:] = ordered
        changed += 1
    return changed

def save(fig, stem, title=''):
    caption_path = A / 'scripts/templates/FIGURE_TEXT.json'
    if caption_path.is_file():
        approved = json.loads(caption_path.read_text(encoding='utf-8'))
        title = approved.get(stem, {}).get('title', title)
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / 'render_work'
    work.mkdir(exist_ok=True)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            positions = axis.get_majorticklocs()
            labels = [formatted(t.get_text()) for t in axis.get_ticklabels()]
            low, high = sorted(axis.get_view_interval())
            retained = [(position, label) for position, label in zip(positions, labels) if low <= position <= high]
            axis.set_major_locator(FixedLocator([item[0] for item in retained]))
            axis.set_major_formatter(FixedFormatter([item[1] for item in retained]))
    for t in fig.findobj(Text):
        t.set_text(formatted(t.get_text()))
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)
        ax.tick_params(axis='both', labelsize=6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi
    boxes, outside = ([], [])
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        rec = {'text': t.get_text(), 'bounds_px': list(b.bounds), 'font_pt': t.get_fontsize()}
        boxes.append(rec)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or (b.y1 > fh + 1)):
            outside.append(rec)
    if outside:
        raise ValueError(f'Text outside {stem}: {outside}')
    base = work / f'{stem}_base.pdf'
    fig.savefig(base)
    svg_base = work / f'{stem}_base.svg'
    fig.savefig(svg_base)
    editable = ET.fromstring(svg_base.read_text(encoding='utf-8'))
    normalize_snar_svg_reading_order(editable)
    pdf = pymupdf.open(base)
    pw, ph = (pdf[0].rect.width, pdf[0].rect.height)
    chemicals = []
    for e in fig._m1_chemistry:
        native = pymupdf.open(e['pdf'])
        ew, eh = (native[0].rect.width * 0.75, native[0].rect.height * 0.75)
        x, y, w, h = e['rect']
        px, py = (x * pw + (w * pw - ew) / 2, (1 - y - h) * ph + (h * ph - eh) / 2)
        if min(px, py) < 0 or px + ew > pw or py + eh > ph:
            raise ValueError(f'Chemical outside {stem}')
        pdf[0].show_pdf_page(pymupdf.Rect(px, py, px + ew, py + eh), native, 0)
        fragment = ET.fromstring(Path(e['normalized_svg']).read_text(encoding='utf-8'))
        vx, vy, vw, vh = map(float, fragment.get('viewBox').split())
        group = ET.SubElement(editable, '{http://www.w3.org/2000/svg}g', {'id': f'native_chemistry_{len(chemicals) + 1}', 'data-inchi-key': e['inchi_key'], 'data-smiles': e['smiles'], 'transform': f'translate({px} {py}) scale({ew / vw} {eh / vh}) translate({-vx} {-vy})'})
        for child in fragment:
            group.append(child)
        chemicals.append({**e, 'bounds_pt': [px, py, ew, eh], 'placement_scale': 0.75})
        native.close()
    FIG.mkdir(parents=True, exist_ok=True)
    target = FIG / f'{stem}.pdf'
    if target.exists():
        target.unlink()
    pdf.set_metadata({'title': title, 'creator': 'M1; verified native ChemDraw vectors'})
    pdf.save(target)
    editable_path = FIG / f'{stem}.svg'
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    ET.ElementTree(editable).write(editable_path, encoding='utf-8', xml_declaration=True)
    pix = pdf[0].get_pixmap(dpi=300, alpha=False)
    pix.save(FIG / f'{stem}.png')
    pdf[0].get_pixmap(dpi=96, alpha=False).save(work / f'{stem}_180mm_96dpi.png')
    grid = []
    for letter, ax, panel_title in fig._m1_panels:
        grid.append({'panel': letter, 'axes_bounds_px': list(ax.get_window_extent(renderer).bounds), 'title_bounds_px': list(panel_title.get_window_extent(renderer).bounds), 'title': panel_title.get_text(), 'letter_font_pt': ax._draft_panel_letter.get_fontsize(), 'letter_weight': ax._draft_panel_letter.get_fontweight()})
    result = {'figure': stem, 'title': title, 'title_visible': False, 'draft': False, 'width_mm': pw / 72 * 25.4, 'height_mm': ph / 72 * 25.4, 'png_pixels': [pix.width, pix.height], 'text_bounds': boxes, 'text_outside': outside, 'png_width': pix.width, 'png_height': pix.height, 'png_dpi': 300, 'pixel_bounds_pass': True, 'chemical_boxes': [{'x_pt': e['bounds_pt'][0], 'y_pt': e['bounds_pt'][1], 'width_pt': e['bounds_pt'][2], 'height_pt': e['bounds_pt'][3]} for e in chemicals], 'panels': grid, 'chemistry': chemicals, 'source_tables': list(TABLES), 'raster_images': len(pdf[0].get_images()), 'vector_paths': len(pdf[0].get_drawings()), 'editable_svg': str(editable_path), 'svg_sha256': hashlib.sha256(editable_path.read_bytes()).hexdigest(), 'axes_fonts_pt': [{'xlabel': ax.xaxis.label.get_fontsize(), 'ylabel': ax.yaxis.label.get_fontsize(), 'ticks': sorted({t.get_fontsize() for t in ax.get_xticklabels() + ax.get_yticklabels()})} for ax in fig.axes]}
    (OUT / f'{stem}_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    QA.append(result)
    pdf.close()
    plt.close(fig)
    return result
setup()
