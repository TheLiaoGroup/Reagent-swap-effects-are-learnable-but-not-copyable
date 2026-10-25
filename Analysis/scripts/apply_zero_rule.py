"""Write the raw corpus after exclusions and the zero-yield rule.

Rule (author decision, 2026-10-18): a well with a blank yield whose DESI call
``yield_clf`` is 0 (no product detected) is recorded as yield 0. Blank yields with
``yield_clf`` 1, or with no DESI call, stay missing.

The authoritative EXCLUDE_wells.csv is applied by name0 before this rule. CSV
files are rewritten field by field and every retained non-yield field keeps its
original text; workbook rows and yield cells are edited in their XML. The
condition and compound registries are copied byte for byte. The result is
verified by reading both versions back. The manifest is written beside OUT_DIR so that
OUT_DIR holds only the 22 corpus files.

Usage: python scripts/apply_zero_rule.py OUT_DIR"""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import re
import zipfile
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_raw import RAW_DIR
from m1_exclusions import Exclusions, assert_exclusions_reconciled, exclusion_mask, exclusion_summary, load_exclusions

def blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == '') or (isinstance(value, float) and np.isnan(value))

def is_zero_call(value) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False

def _require_measurement_id(header, label: str) -> None:
    if 'yield' in header and 'name0' not in header:
        raise ValueError(f'{label}: reaction table missing measurement ID column name0')

def rewrite_csv(src: Path, dst: Path, exclusions: Exclusions | None=None, deleted_ids: list[str] | None=None) -> int:
    raw = src.read_bytes()
    text = raw.decode('utf-8-sig') if raw.startswith(b'\xef\xbb\xbf') else raw.decode('utf-8')
    newline = '\r\n' if '\r\n' in text[:10000] else '\n'
    rows = list(csv.reader(io.StringIO(text, newline='')))
    header = rows[0]
    if exclusions is not None:
        _require_measurement_id(header, src.name)
        if 'yield' in header:
            key = header.index('name0')
            keep = [header]
            for row in rows[1:]:
                if len(row) > key and row[key] in exclusions.ids:
                    if deleted_ids is not None:
                        deleted_ids.append(row[key])
                else:
                    keep.append(row)
            rows = keep
    if 'yield' not in header:
        shutil.copy2(src, dst)
        return 0
    changed = 0
    if 'yield_clf' in header:
        y, c = (header.index('yield'), header.index('yield_clf'))
        for row in rows[1:]:
            if len(row) > max(y, c) and blank(row[y]) and is_zero_call(row[c]):
                row[y] = '0'
                changed += 1
    out = io.StringIO(newline='')
    csv.writer(out, lineterminator=newline).writerows(rows)
    body = out.getvalue().encode('utf-8')
    dst.write_bytes((b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'') + body)
    return changed
_NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main', 'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships', 'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}
_CELL = re.compile('<c r="([A-Z]+)(\\d+)"[^>]*?(?:/>|>.*?</c>)', re.S)
_ROW = re.compile('(<row(?:\\s[^>]*)?>)(.*?)(</row>)', re.S)

def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n

def _cell_value(cell_xml: str, shared: list[str]):
    v = re.search('<v>(.*?)</v>', cell_xml, re.S)
    if v is None:
        t = re.search('<t[^>]*>(.*?)</t>', cell_xml, re.S)
        return t.group(1) if t else None
    if 't="s"' in cell_xml.split('>', 1)[0]:
        return shared[int(v.group(1))]
    return v.group(1)

def _sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    from xml.etree import ElementTree as ET
    book = ET.fromstring(zf.read('xl/workbook.xml'))
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    target = {r.get('Id'): r.get('Target') for r in rels.findall('rel:Relationship', _NS)}
    out = {}
    for sheet in book.find('m:sheets', _NS):
        path = target[sheet.get(f"{{{_NS['r']}}}id")].lstrip('/')
        out[sheet.get('name')] = path if path.startswith('xl/') else 'xl/' + path
    return out

def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    from xml.etree import ElementTree as ET
    if 'xl/sharedStrings.xml' not in zf.namelist():
        return []
    root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    return [''.join((t.text or '' for t in si.iter(f"{{{_NS['m']}}}t"))) for si in root.findall('m:si', _NS)]

def _zero_sheet(xml: str, shared: list[str], exclusions: Exclusions | None=None, deleted_ids: list[str] | None=None) -> tuple[str, int]:
    rows = list(_ROW.finditer(xml))
    if not rows:
        return (xml, 0)
    header = {_cell_value(m.group(0), shared): m.group(1) for m in _CELL.finditer(rows[0].group(2))}
    if exclusions is not None:
        _require_measurement_id(header, 'workbook sheet')
    if 'yield' not in header:
        return (xml, 0)
    ycol, ccol = (header['yield'], header.get('yield_clf'))
    idcol = header.get('name0')
    header_number = re.search('\\sr="(\\d+)"', rows[0].group(1)).group(1)
    changed = 0
    removed = 0

    def fix(row: re.Match) -> str:
        nonlocal changed, removed
        cells = list(_CELL.finditer(row.group(2)))
        by_col = {m.group(1): m for m in cells}
        number = re.search('\\sr="(\\d+)"', row.group(1)).group(1)
        if number == header_number:
            return row.group(0)
        identifier = _cell_value(by_col[idcol].group(0), shared) if idcol in by_col else None
        if exclusions is not None and identifier in exclusions.ids:
            removed += 1
            if deleted_ids is not None:
                deleted_ids.append(identifier)
            return ''
        body = row.group(2)
        existing = by_col.get(ycol)
        if ccol in by_col and is_zero_call(_cell_value(by_col[ccol].group(0), shared)) and (existing is None or blank(_cell_value(existing.group(0), shared))):
            style = re.search('\\ss="(\\d+)"', existing.group(0).split('>', 1)[0]) if existing else None
            new = f'<c r="{ycol}{number}"' + (f' s="{style.group(1)}"' if style else '') + '><v>0</v></c>'
            if existing is not None:
                body = body[:existing.start()] + new + body[existing.end():]
            else:
                after = [m for m in cells if _col_index(m.group(1)) > _col_index(ycol)]
                cut = after[0].start() if after else len(body)
                body = body[:cut] + new + body[cut:]
            changed += 1
        opening = row.group(1)
        if removed:
            new_number = str(int(number) - removed)
            opening = re.sub('(?<= r=")\\d+', new_number, opening, count=1)
            body = re.sub('(<c r="[A-Z]+)\\d+(?=")', lambda m: m.group(1) + new_number, body)
        return opening + body + row.group(3)
    result = _ROW.sub(fix, xml)
    if removed:
        result = re.sub('(<dimension ref="[A-Z]+\\d+:[A-Z]+)(\\d+)("/>)', lambda m: m.group(1) + str(int(m.group(2)) - removed) + m.group(3), result)
    return (result, changed)

def rewrite_xlsx(src: Path, dst: Path, exclusions: Exclusions | None=None, deleted_ids: list[str] | None=None) -> int:
    """Remove excluded rows and edit yield cells; copy other zip members unchanged."""
    changed = 0
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, 'w') as zout:
        shared = _shared_strings(zin)
        sheets = set(_sheet_paths(zin).values())
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in sheets:
                original = data.decode('utf-8')
                xml, n = _zero_sheet(original, shared, exclusions, deleted_ids)
                if xml != original:
                    data = xml.encode('utf-8')
                changed += n
            zout.writestr(info, data, compress_type=info.compress_type)
    return changed

def read_all(path: Path) -> dict[str, pd.DataFrame]:
    if path.suffix.lower() == '.xlsx':
        return reader_tables.read_excel(path, sheet_name=None, dtype=object, keep_default_na=False)
    return {'CSV': reader_tables.read_csv(path, low_memory=False, dtype=object, keep_default_na=False)}

def verify(src: Path, dst: Path, expected: int, exclusions: Exclusions | None=None) -> tuple[int, int]:
    a, b = (read_all(src), read_all(dst))
    if list(a) != list(b):
        raise AssertionError(f'{src.name}: sheet list changed')
    seen = 0
    source_rows = output_rows = 0
    for name in a:
        x, z = (a[name], b[name])
        source_rows += len(x)
        output_rows += len(z)
        if exclusions is not None and 'yield' in x:
            x = x.loc[~exclusion_mask(x, exclusions)].reset_index(drop=True)
        if list(x.columns) != list(z.columns) or len(x) != len(z):
            raise AssertionError(f'{src.name}/{name}: shape or columns changed')
        for column in x.columns:
            if column == 'yield' and 'yield_clf' in x:
                rule = x['yield'].map(blank) & pd.to_numeric(x['yield_clf'], errors='coerce').eq(0)
                seen += int(rule.sum())
                if not (z.loc[rule, 'yield'].astype(float) == 0).all():
                    raise AssertionError(f'{src.name}/{name}: rule cells not zero')
                pd.testing.assert_series_equal(x.loc[~rule, 'yield'], z.loc[~rule, 'yield'], check_dtype=False)
            else:
                pd.testing.assert_series_equal(x[column], z[column], check_dtype=False)
    if seen != expected:
        raise AssertionError(f'{src.name}: changed {expected} cells, rule predicts {seen}')
    return (source_rows, output_rows)

def matching_ids(src: Path, exclusions: Exclusions) -> list[str]:
    """Scan IDs without loading the raw corpus into pandas during preflight."""
    if src.name.startswith('0_'):
        return []
    matched = []
    if src.suffix.lower() == '.csv':
        with src.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            _require_measurement_id(reader.fieldnames or [], src.name)
            if 'yield' in (reader.fieldnames or []):
                matched.extend((row['name0'] for row in reader if row['name0'] in exclusions.ids))
    else:
        with zipfile.ZipFile(src) as zin:
            shared = _shared_strings(zin)
            for path in _sheet_paths(zin).values():
                rows = _ROW.finditer(zin.read(path).decode('utf-8'))
                first = next(rows, None)
                if first is None:
                    continue
                header = {_cell_value(m.group(0), shared): m.group(1) for m in _CELL.finditer(first.group(2))}
                _require_measurement_id(header, f'{src.name}/{path}')
                if 'yield' not in header:
                    continue
                key = header['name0']
                for row in rows:
                    for cell in _CELL.finditer(row.group(2)):
                        if cell.group(1) == key:
                            identifier = _cell_value(cell.group(0), shared)
                            if identifier in exclusions.ids:
                                matched.append(identifier)
                            break
    return matched

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if len(sys.argv) not in (1, 2):
        raise SystemExit('Usage: python apply_zero_rule.py [OUTPUT_DIRECTORY]')
    out = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parents[1] / 'out/curated_inputs'
    if out == RAW_DIR.resolve() or RAW_DIR.resolve() in out.parents:
        raise ValueError('Output must be outside the original CSV source directory')
    exclusions = load_exclusions()
    manifest = reader_tables.apply_zero_rule_manifest(RAW_DIR)
    deleted = manifest['exclusions']['matched_ids']
    assert_exclusions_reconciled(deleted, exclusions)
    manifest['exclusions'] = exclusion_summary(deleted, exclusions)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f'{out.name}_zero_rule_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', zeroed_cells=manifest['zeroed_cells'], sources=len(manifest['files'])), indent=2))
if __name__ == '__main__':
    main()
