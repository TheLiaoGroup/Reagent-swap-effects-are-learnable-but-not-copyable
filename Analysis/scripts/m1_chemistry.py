"""Fresh, identity-checked ChemDraw vector structures from M1 dataset SMILES."""
from __future__ import annotations
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import pymupdf
TEMPLATE = Path(__file__).resolve().parents[1] / 'figure_assets/Nature Research Document.cds'
TEMPLATE_SHA256 = 'afb36333c2d79d08c23330926678bad13b9bcb25cac56f5d6bb7df30fac1514e'
SETTINGS = ('BondLength', 'BondSpacing', 'BoldWidth', 'LineWidth', 'MarginWidth', 'HashSpacing', 'ChainAngle', 'LabelFont', 'LabelSize', 'LabelFace', 'CaptionFont', 'CaptionSize', 'CaptionFace')
CSS_PX_PER_PT = 96 / 72
NATIVE_UNITS_PER_PT = 20
SCALE_VERSION = 1

def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite native vector coordinate or scale')
    return number

def normalize_native_svg(svg: str, cdxml: str):
    """Recover document units before the existing CSS-pixel-to-point placement."""
    document = ET.fromstring(cdxml)
    style = {'BondLength': 10.8, 'LabelSize': 6, 'CaptionSize': 6, 'LineWidth': 0.6}
    for name, expected in style.items():
        if not math.isclose(_number(document.get(name, 'nan')), expected, abs_tol=1e-06):
            raise ValueError(f'Native CDXML style differs: {name}')
    labels = list(document.iter('s'))
    if not labels or any((not math.isclose(_number(item.get('size', 'nan')), 6, abs_tol=1e-06) for item in labels)):
        raise ValueError('Native CDXML label style differs')
    fonts = {item.get('id'): item.get('name') for item in document.iter('font')}
    if not fonts or any((fonts.get(item.get('font', document.get('LabelFont'))) != 'Arial' for item in labels)):
        raise ValueError('Native CDXML label font differs from Arial')
    nodes = {item.get('id'): item for item in document.iter('n') if item.get('p')}
    coordinates = {key: tuple((_number(v) for v in item.get('p').split())) for key, item in nodes.items()}
    if any((len(point) != 2 for point in coordinates.values())):
        raise ValueError('Invalid native atom coordinate')
    lengths = []
    for bond in document.iter('b'):
        begin, end = (coordinates.get(bond.get('B')), coordinates.get(bond.get('E')))
        if begin is None or end is None:
            raise ValueError('Missing native bond coordinate')
        lengths.append(math.dist(begin, end))
    if not lengths or any((abs(length - style['BondLength']) > 0.03 for length in lengths)):
        raise ValueError('Actual CDXML bond geometry differs from the document bond length')
    root = ET.fromstring(svg)
    if any((item.tag.split('}')[-1] == 'image' for item in root.iter())):
        raise ValueError('Raster content in native chemical SVG')
    if root.tag.split('}')[-1] != 'svg' or root.get('transform'):
        raise ValueError('Unsupported native SVG root transform')
    if root.get('style', '').replace('background-color: #ffffff00', '').strip(' ;'):
        raise ValueError('Unsupported native SVG CSS scale or style')
    view = [_number(value) for value in root.get('viewBox', '').replace(',', ' ').split()]
    if len(view) != 4 or view[2] <= 0 or view[3] <= 0:
        raise ValueError('Invalid native SVG viewport')
    dimensions = []
    for name, span in (('width', view[2]), ('height', view[3])):
        text = root.get(name, '')
        match = re.fullmatch('([-+0-9.eE]+)(?:px)?', text)
        if not match or not math.isclose(_number(match[1]), span, rel_tol=1e-06):
            raise ValueError('Native SVG viewport must use one CSS pixel per viewBox unit')
        dimensions.append(_number(match[1]))
    matrices, text_sizes, text_characters, path_vertices, path_offsets, short_edges = ([], [], [], [], [], [])
    for item in root:
        tag = item.tag.split('}')[-1]
        if tag not in ('text', 'path') or len(item) or item.get('style'):
            raise ValueError('Unsupported native SVG element, nested transform or CSS')
        match = re.fullmatch('matrix\\(([^()]*)\\)', item.get('transform', ''))
        if not match:
            raise ValueError('Unsupported native SVG transform')
        matrix = [_number(value) for value in match[1].replace(',', ' ').split()]
        if len(matrix) != 6 or matrix[0] <= 0 or matrix[1:3] != [0, 0] or (not math.isclose(matrix[0], matrix[3], rel_tol=1e-08)):
            raise ValueError('Native SVG transform must have positive isotropic scale')
        matrices.append(matrix)
        if tag == 'text':
            text = ''.join(item.itertext()).strip()
            font = re.fullmatch('([-+0-9.eE]+)px', item.get('font-size', ''))
            if not font or item.get('font-family') != 'Arial':
                raise ValueError('Native SVG label font or size differs')
            if item.get('font-style', 'normal') != 'normal' or item.get('font-weight', 'normal') != 'normal':
                raise ValueError('Native SVG label font style differs')
            size = _number(font[1])
            if not (math.isclose(size, 120, abs_tol=1e-06) or (math.isclose(size, 90, abs_tol=1e-06) and text.isdecimal())):
                raise ValueError('Native SVG label size is not the CDXML label or chemical subscript size')
            text_sizes.append(size)
            text_characters.extend((character for character in text if not character.isspace()))
        else:
            data = item.get('d', '')
            if re.sub('[MLZ0-9.eE+,\\-\\s]', '', data):
                raise ValueError('Unsupported native bond path coordinates')
            values = [_number(value) for value in re.findall('[-+]?(?:\\d*\\.\\d+|\\d+\\.?)(?:[eE][-+]?\\d+)?', data)]
            if len(values) % 2 or not values:
                raise ValueError('Invalid native bond path geometry')
            vertices = list(zip(values[::2], values[1::2]))
            path_vertices.extend(vertices)
            path_offsets.append(matrix[4:])
            short_edges.extend((math.dist(a, b) for a, b in zip(vertices, vertices[1:])))
    if not matrices or not text_sizes or (not path_vertices):
        raise ValueError('Native vector needs both label and bond scale evidence')
    document_characters = [character for item in labels for character in ''.join(item.itertext()) if not character.isspace()]
    if Counter(text_characters) != Counter(document_characters):
        raise ValueError('Native SVG element labels differ from CDXML labels')
    export_scale = matrices[0][0]
    if any((not math.isclose(item[0], export_scale, rel_tol=1e-07) for item in matrices)):
        raise ValueError('Native text and path scales disagree')
    if any((offset != path_offsets[0] for offset in path_offsets)):
        raise ValueError('Native bond path coordinate frames disagree')
    tolerance = max(style['LineWidth'] * 1.25, _number(document.get('BoldWidth', '1.56')) * 0.6)
    distances = []
    for key, node in nodes.items():
        if node.get('Element', '6') == '6' and node.find('t') is None:
            point = tuple((value * NATIVE_UNITS_PER_PT for value in coordinates[key]))
            distances.append(min((math.dist(point, vertex) for vertex in path_vertices)) / NATIVE_UNITS_PER_PT)
    if len(distances) < 2 or max(distances) > tolerance:
        raise ValueError('Native SVG bond geometry does not match CDXML atom coordinates')
    expected_line = style['LineWidth'] * NATIVE_UNITS_PER_PT
    widths = [edge for edge in short_edges if abs(edge - expected_line) <= 0.03 * expected_line]
    if not widths:
        raise ValueError('Native SVG line width does not match the CDXML style')
    factor = CSS_PX_PER_PT / (NATIVE_UNITS_PER_PT * export_scale)
    root.set('width', f'{dimensions[0] * factor:.12g}px')
    root.set('height', f'{dimensions[1] * factor:.12g}px')
    root.set('viewBox', ' '.join((f'{value * factor:.12g}' for value in view)))
    root.set('data-m1-native-scale', str(SCALE_VERSION))
    group = ET.Element('{http://www.w3.org/2000/svg}g', {'transform': f'scale({factor:.15g})'})
    for item in list(root):
        root.remove(item)
        group.append(item)
    root.append(group)
    normalized = ET.tostring(root, encoding='unicode')
    with pymupdf.open(stream=normalized.encode('utf-8'), filetype='svg') as vector:
        with pymupdf.open('pdf', vector.convert_to_pdf()) as pdf:
            page = pdf[0]
            content = page.get_bboxlog()
            if not content or page.get_images() or any((bounds[0] < -0.0001 or bounds[1] < -0.0001 or bounds[2] > page.rect.width + 0.0001 or (bounds[3] > page.rect.height + 0.0001) for _, bounds, *_ in content)):
                raise ValueError('Native SVG content is outside its viewport or contains raster content')
            spans = [span for block in page.get_text('dict')['blocks'] for line in block.get('lines', []) for span in line['spans']]
            expected_sizes = {size * export_scale * factor * 0.75 for size in text_sizes}
            if not spans or any((not any((abs(span['size'] * 0.75 - size) < 0.0001 for size in expected_sizes)) for span in spans)):
                raise ValueError('Rendered native SVG label size differs from document units')
            rendered_fonts = sorted({span['font'] for span in spans})
    metadata = {'version': SCALE_VERSION, 'native_units_per_cdxml_point': NATIVE_UNITS_PER_PT, 'css_pixels_per_point': CSS_PX_PER_PT, 'export_matrix_scale': export_scale, 'normalization_factor': factor, 'existing_layout_scale': 0.75, 'raw_viewbox': view, 'normalized_viewbox': [v * factor for v in view], 'cdxml_bond_lengths_pt': lengths, 'max_carbon_endpoint_error_pt': max(distances), 'verified_line_cross_sections': len(widths), 'final_bond_length_pt': NATIVE_UNITS_PER_PT * export_scale * factor * 0.75 * style['BondLength'], 'final_label_size_pt': max(text_sizes) * export_scale * factor * 0.75, 'final_text_sizes_pt': sorted({size * export_scale * factor * 0.75 for size in text_sizes}), 'final_line_width_pt': expected_line * export_scale * factor * 0.75, 'rendered_pdf_fonts': rendered_fonts, 'content_within_viewport': True, 'raw_svg_sha256': hashlib.sha256(svg.encode('utf-8')).hexdigest(), 'normalized_svg_sha256': hashlib.sha256(normalized.encode('utf-8')).hexdigest(), 'cdxml_sha256': hashlib.sha256(cdxml.encode('utf-8')).hexdigest()}
    return (normalized, metadata)

def verified_template():
    if not TEMPLATE.is_file():
        raise FileNotFoundError('The packaged Nature Research Document.cds style input is required')
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('Packaged ChemDraw style digest differs from the frozen style input')
    return TEMPLATE

def verified_cached_native_pdf(stem, svg):
    cached = pymupdf.open(stem.with_suffix('.pdf'))
    try:
        with pymupdf.open(stream=svg.encode(), filetype='svg') as vector:
            with pymupdf.open('pdf', vector.convert_to_pdf()) as expected:
                if len(cached) != 1 or len(expected) != 1:
                    raise ValueError('Native PDF cache must contain exactly one page')
                if cached[0].rect != expected[0].rect or cached[0].get_images():
                    raise ValueError('Native PDF cache geometry differs or contains raster content')
                matrix = pymupdf.Matrix(3, 3)
                if cached[0].get_pixmap(matrix=matrix).samples != expected[0].get_pixmap(matrix=matrix).samples:
                    raise ValueError('Native PDF cache content differs from its verified native SVG')
        return cached
    except BaseException:
        cached.close()
        raise

ORIENTATION_VERSION = 'lattice-v1'


def _orient_to_lattice(mol, free=True):
    """Rotate 2D coordinates so six-membered ring edges lie on the 30-degree lattice.

    Every hexagon then has vertical or horizontal sides. With free=True the molecule is
    turned (in 30-degree steps on that lattice) to put its long axis horizontal; with
    free=False only the smallest lattice correction is applied, so template-matched
    series keep a common orientation.
    """
    import math
    conf = mol.GetConformer()
    rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) == 6]
    pts = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(mol.GetNumAtoms())]
    if not pts:
        return
    cx, cy = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)

    def rotated(deg):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        return [((x - cx) * c - (y - cy) * s, (x - cx) * s + (y - cy) * c) for x, y in pts]

    base = 0.0
    if rings:
        ring = max(rings, key=lambda r: (all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r),
                                         sum(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in r)))
        sx = sy = 0.0
        for a, b in zip(ring, ring[1:] + ring[:1]):
            ang = math.atan2(pts[b][1] - pts[a][1], pts[b][0] - pts[a][0]) * 6.0  # period 60 deg -> 360
            sx += math.cos(ang)
            sy += math.sin(ang)
        e = (math.degrees(math.atan2(sy, sx)) / 6.0) % 30.0  # edge offset from the 30-degree lattice
        base = -e if e < 15.0 else 30.0 - e
    candidates = [base] if (not free or not rings) else [base + 30.0 * k for k in range(12)]
    if not free and not rings:
        return

    def score(deg):
        q = rotated(deg)
        w = max(x for x, _ in q) - min(x for x, _ in q)
        h = max(y for _, y in q) - min(y for _, y in q)
        return (round(w - h, 3), -abs(((deg + 180) % 360) - 180))

    best = max(candidates, key=score)
    for i, (x, y) in enumerate(rotated(best)):
        conf.SetAtomPosition(i, (x + cx, y + cy, 0.0))


class Chemistry:

    def __init__(self, directory: Path, work: Path):
        self.template = verified_template()
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.work = work
        work.mkdir(parents=True, exist_ok=True)
        self.assets = {}
        self.failures = []
        self.app = None
        self.settings = {}
        try:
            import win32com.client
            self.app = win32com.client.Dispatch('ChemDraw.Application.22.0')
            doc = self.app.Documents.Open(str(self.template))
            self.settings = {k: getattr(doc.Settings, k) for k in SETTINGS}
            doc.Close(False)
        except Exception as exc:
            self.failures.append(f'ChemDraw initialization: {exc}')

    def _native_svg(self, stem):
        svg_path = stem.with_suffix('.svg')
        raw_path = stem.with_suffix('.native.svg')
        metadata_path = stem.with_suffix('.scale.json')
        current = svg_path.read_bytes()
        cdxml_bytes = stem.with_suffix('.cdxml').read_bytes()
        cdxml = cdxml_bytes.decode('utf-8-sig')
        if raw_path.exists() or metadata_path.exists():
            if not (raw_path.is_file() and metadata_path.is_file()):
                raise ValueError('Incomplete native vector scale cache')
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if hashlib.sha256(current).hexdigest() != metadata.get('normalized_svg_sha256'):
                raise ValueError('Normalized native SVG cache hash differs')
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata.get('raw_svg_sha256'):
                raise ValueError('Original native SVG cache hash differs')
            if hashlib.sha256(cdxml_bytes).hexdigest() != metadata.get('cdxml_sha256'):
                raise ValueError('Native CDXML cache hash differs')
        else:
            if ET.fromstring(current).get('data-m1-native-scale'):
                raise ValueError('Normalized native SVG cache lacks original scale evidence')
            raw = current
        normalized, metadata = normalize_native_svg(raw.decode('utf-8-sig'), cdxml)
        metadata['raw_svg_sha256'] = hashlib.sha256(raw).hexdigest()
        metadata['cdxml_sha256'] = hashlib.sha256(cdxml_bytes).hexdigest()
        if current != raw and hashlib.sha256(current).hexdigest() != metadata['normalized_svg_sha256']:
            raise ValueError('Native SVG cache does not match its CDXML style or normalization version')
        if not raw_path.exists():
            raw_path.write_bytes(raw)
        svg_path.write_bytes(normalized.encode('utf-8'))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return (normalized, str(metadata_path))

    def _export_native(self, source, stem):
        script = Path(__file__).with_name('m1_native_export.py')
        result = subprocess.run([sys.executable, str(script), str(self.template.resolve()), str(source.resolve()), str(stem.resolve()), str((self.work / (stem.name + '_readback.mol')).resolve())], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())

    def get(self, smiles: str, marked: bool=False, template: str | None=None):
        key = (smiles, marked, template)
        if key in self.assets:
            return self.assets[key]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid dataset SMILES: {smiles}')
        rdDepictor.Compute2DCoords(mol)
        if template:
            core = Chem.MolFromSmiles(template)
            rdDepictor.Compute2DCoords(core)
            rdDepictor.GenerateDepictionMatching2DStructure(mol, core)
        _orient_to_lattice(mol, free=not template)
        stem = self.directory / ('M_' + hashlib.sha256(repr((key, ORIENTATION_VERSION)).encode()).hexdigest()[:12])
        readback_path = self.work / (stem.name + '_readback.mol')
        expected_key = Chem.MolToInchiKey(mol)
        if all((stem.with_name(stem.name + suffix).is_file() for suffix in ('.cdx', '.cdxml', '.svg'))) and readback_path.is_file():
            observed = Chem.MolFromMolBlock(readback_path.read_text(encoding='utf-8-sig'))
            if observed is not None and Chem.MolToInchiKey(observed) == expected_key:
                svg, scale_metadata = self._native_svg(stem)
                cached_pdf = stem.with_suffix('.pdf').is_file()
                if cached_pdf:
                    pdf = verified_cached_native_pdf(stem, svg)
                else:
                    vector = pymupdf.open(stream=svg.encode(), filetype='svg')
                    pdf = pymupdf.open('pdf', vector.convert_to_pdf())
                    pdf.save(stem.with_suffix('.pdf'))
                entry = {'smiles': smiles, 'canonical_smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 'inchi_key': expected_key, 'roundtrip_inchi_key': Chem.MolToInchiKey(observed), 'svg': str(stem.with_suffix('.svg')), 'pdf': str(stem.with_suffix('.pdf')), 'cdxml': str(stem.with_suffix('.cdxml')), 'cdx': str(stem.with_suffix('.cdx')), 'width_pt': pdf[0].rect.width, 'height_pt': pdf[0].rect.height, 'native': True, 'marked': marked, 'alignment_template': template, 'fallback_reason': None, 'scale_metadata': scale_metadata, 'native_svg': str(stem.with_suffix('.native.svg'))}
                pdf.close()
                if not cached_pdf:
                    vector.close()
                self.assets[key] = entry
                return entry
        mol_path = self.work / (stem.name + '.mol')
        mol_path.write_text(Chem.MolToMolBlock(mol), encoding='ascii')
        native = False
        error = None
        try:
            if self.app is None:
                raise RuntimeError('ChemDraw COM unavailable')
            self._export_native(mol_path, stem)
            xml = stem.with_suffix('.cdxml').read_text(encoding='utf-8-sig')
            root = ET.fromstring(xml)
            root.set('BondLength', '10.8')
            root.set('LabelSize', '6')
            root.set('CaptionSize', '6')
            for elem in root.iter():
                if elem.tag == 's':
                    elem.set('size', '6')
            if marked:
                table = root.find('colortable')
                if table is None:
                    table = ET.SubElement(root, 'colortable')
                ET.SubElement(table, 'color', {'r': '0.73', 'g': '0.28', 'b': '0.32'})
                color_id = str(len(list(table)) + 1)
                nodes = {n.get('id'): n for n in root.iter('n')}
                degree = {nid: 0 for nid in nodes}
                for bond in root.iter('b'):
                    for edge in (bond.get('B'), bond.get('E')):
                        if edge in degree:
                            degree[edge] += 1
                for nid, node in nodes.items():
                    if node.get('Element', '6') == '16' or (node.get('Element', '6') == '6' and degree[nid] == 1):
                        node.set('color', color_id)
                        for text in node.iter('t'):
                            text.set('color', color_id)
                        for bond in root.iter('b'):
                            if nid in (bond.get('B'), bond.get('E')):
                                bond.set('color', color_id)
            stem.with_suffix('.cdxml').write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')
            self._export_native(stem.with_suffix('.cdxml'), stem)
            svg = stem.with_suffix('.svg').read_text(encoding='utf-8-sig')
            readback = readback_path.read_text(encoding='utf-8-sig')
            readback_path.write_text(readback, encoding='utf-8')
            observed = Chem.MolFromMolBlock(readback, removeHs=True)
            if observed is None or Chem.MolToInchiKey(observed) != expected_key:
                raise ValueError(f'ChemDraw identity round-trip mismatch: {smiles}')
            native = True
        except Exception as exc:
            error = str(exc)
            if 'identity round-trip mismatch' in error:
                raise
            self.failures.append(f'{smiles}: {error}')
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 150)
            opts = drawer.drawOptions()
            opts.fixedBondLength = 25
            opts.useBWAtomPalette()
            atoms = [a.GetIdx() for a in mol.GetAtoms() if marked and (a.GetAtomicNum() == 16 or (a.GetAtomicNum() == 6 and a.GetDegree() == 1 and (a.GetTotalNumHs() == 3)))]
            drawer.DrawMolecule(mol, highlightAtoms=atoms)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            stem.with_suffix('.svg').write_text(svg, encoding='utf-8')
            observed = Chem.MolFromMolBlock(mol_path.read_text(encoding='ascii'))
            if Chem.MolToInchiKey(observed) != expected_key:
                raise ValueError('RDKit MOL identity mismatch')
        scale_metadata = None
        if native:
            svg, scale_metadata = self._native_svg(stem)
        element = ET.fromstring(svg)
        if any((n.tag.split('}')[-1] == 'image' for n in element.iter())):
            raise ValueError('Raster content in chemical SVG')
        doc = pymupdf.open(stream=svg.encode(), filetype='svg')
        pdf = pymupdf.open('pdf', doc.convert_to_pdf())
        width, height = (pdf[0].rect.width, pdf[0].rect.height)
        pdf.save(stem.with_suffix('.pdf'))
        pdf.close()
        doc.close()
        entry = {'smiles': smiles, 'canonical_smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 'inchi_key': expected_key, 'roundtrip_inchi_key': Chem.MolToInchiKey(observed), 'svg': str(stem.with_suffix('.svg')), 'pdf': str(stem.with_suffix('.pdf')), 'cdxml': str(stem.with_suffix('.cdxml')) if native else None, 'cdx': str(stem.with_suffix('.cdx')) if native else None, 'width_pt': width, 'height_pt': height, 'native': native, 'marked': marked, 'alignment_template': template, 'fallback_reason': error, 'scale_metadata': scale_metadata, 'native_svg': str(stem.with_suffix('.native.svg')) if native else None}
        self.assets[key] = entry
        return entry

    def finalize(self):
        reader_root = Path(__file__).resolve().parents[2]
        path_fields = ('svg', 'pdf', 'cdxml', 'cdx', 'scale_metadata', 'native_svg')
        assets = []
        for runtime_asset in self.assets.values():
            asset = dict(runtime_asset)
            for field in path_fields:
                if asset.get(field) is not None:
                    asset[field] = Path(asset[field]).resolve().relative_to(reader_root).as_posix()
            assets.append(asset)
        result = {'stylesheet': 'Nature Research Document', 'bond_length_pt': 10.8, 'stylesheet_source': self.template.resolve().relative_to(reader_root).as_posix(), 'stylesheet_sha256': TEMPLATE_SHA256, 'label_size_pt': 6, 'assets': assets, 'fallbacks': self.failures}
        (self.directory / 'chemistry_assets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
