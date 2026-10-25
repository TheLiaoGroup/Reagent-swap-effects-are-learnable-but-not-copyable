"""Read full original CSV support; apply exclusions and zero rule at runtime."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.parsers import TextParser


SCHEMA = "m1-full-ingestion-support-v2"


def data_dir():
    configured = os.environ.get("M1_TABLES_DIR") or os.environ.get("M1_DATA_DIR")
    if configured:
        root = Path(configured).resolve()
        _manifest(root)
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "data/tables.json").is_file():
            return parent / "data"
    raise FileNotFoundError("Full support data not found; set M1_DATA_DIR")


def _root(path=None):
    if isinstance(path, ExcelFile):
        return path.root
    if path is not None:
        path = Path(path)
        if (path / "tables.json").is_file():
            return path.resolve()
        if (path.parent / "tables.json").is_file():
            return path.parent.resolve()
        if path.is_absolute() or path.parent != Path("."):
            raise ValueError("Source must be a logical name inside the distributed data directory")
    return data_dir()


def _manifest(root):
    value = json.loads((Path(root) / "tables.json").read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("schema_version") != 2:
        raise ValueError("Unknown full ingestion support schema")
    return value


def _member(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest member is outside the distributed data directory")
    return path


def _source(path):
    root = _root(path)
    name = path.path.name if isinstance(path, ExcelFile) else Path(path).name
    found = [s for s in _manifest(root)["sources"] if s["logical_source"] == name]
    if len(found) != 1:
        raise FileNotFoundError(f"No unique logical source: {name}")
    return root, found[0]


def source_paths(root=None, pattern=None):
    root = _root(root)
    paths = [root / s["logical_source"] for s in _manifest(root)["sources"]]
    return [p for p in paths if pattern is None or fnmatch.fnmatch(p.name, pattern)]


def is_source(path):
    try:
        _source(path)
        return True
    except (ValueError, FileNotFoundError):
        return False


def source_is_file(path):
    try:
        root, source = _source(path)
    except (ValueError, FileNotFoundError):
        return Path(path).is_file()
    names = [s[k] for s in source["sheets"] for k in ("csv", "types", "row_map") if s.get(k)]
    return bool(names) and all(_member(root, name).is_file() for name in names)


def _sheet(source, sheet):
    if isinstance(sheet, int):
        return source["sheets"][sheet]
    return next(s for s in source["sheets"] if s["name"] == sheet)


def _decode(value, code):
    if code == "i":
        return int(value)
    if code == "f":
        return float(value)
    if code == "b":
        return value == "True"
    if code == "d":
        return datetime.fromisoformat(value)
    if code == "n":
        return np.nan
    if code == "s":
        return value
    raise ValueError(f"Unknown original cell type: {code}")


def _profile(value):
    result = value or {"recorded": "original", "processed": "analysis"}.get(
        os.environ.get("M1_INPUT_PROFILE", "processed"), "analysis")
    if result not in {"original", "analysis"}:
        raise ValueError(f"Unsupported source profile: {result}")
    return result


def _original_rows(root, sheet):
    with _member(root, sheet["csv"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if sheet.get("types"):
        with _member(root, sheet["types"]).open(encoding="utf-8", newline="") as stream:
            types = list(csv.reader(stream))
        if not rows or not types or types[0] != rows[0] or len(types) != len(rows):
            raise ValueError("Invalid type table dimensions or header")
        if any(len(values) != len(rows[0]) or len(codes) != len(rows[0])
               for values, codes in zip(rows[1:], types[1:])):
            raise ValueError("Invalid type table row dimensions")
        rows[1:] = [[_decode(value, code) for value, code in zip(values, codes)]
                    for values, codes in zip(rows[1:], types[1:])]
    if len(rows) - 1 != sheet["original_rows"] or rows[0] != sheet["columns"]:
        raise ValueError("Original source shape differs")
    return rows


def _blank(value):
    return (isinstance(value, str) and not value.strip()) or pd.isna(value)


def _zero_call(value):
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


def _process(root, source, rows):
    if source["kind"] != "reaction" or "yield" not in rows[0]:
        return rows, [], 0
    header = rows[0]
    if "name0" not in header:
        raise ValueError("Reaction source lacks exclusion identity name0")
    ids = frozenset(_manifest(root)["exclusions"]["matched_ids"])
    identifier, y = header.index("name0"), header.index("yield")
    call = header.index("yield_clf") if "yield_clf" in header else None
    result, deleted, zeroed = [header], [], 0
    for values in rows[1:]:
        if values[identifier] in ids:
            deleted.append(values[identifier])
            continue
        values = list(values)
        if call is not None and _blank(values[y]) and _zero_call(values[call]):
            values[y] = "0" if source["format"] == "csv" else 0
            zeroed += 1
        result.append(values)
    return result, deleted, zeroed


def _rows(root, sheet, profile, source=None):
    rows = _original_rows(root, sheet)
    if _profile(profile) == "analysis":
        if source is None:
            source = next(s for s in _manifest(root)["sources"] if sheet in s["sheets"])
        rows = _process(root, source, rows)[0]
    return rows


def read_excel(path, sheet_name=0, *, profile=None, **kwargs):
    root, source = _source(path)
    kwargs.pop("engine", None)
    kwargs.pop("engine_kwargs", None)
    sheets = [s["name"] for s in source["sheets"]] if sheet_name is None else (
        sheet_name if isinstance(sheet_name, list) else [sheet_name])
    result = {}
    for name in sheets:
        options = dict(kwargs)
        nrows = options.pop("nrows", None)
        options.setdefault("skip_blank_lines", False)
        with TextParser(_rows(root, _sheet(source, name), profile, source), **options) as parser:
            result[name] = parser.read(nrows=nrows)
    return result if sheet_name is None or isinstance(sheet_name, list) else result[sheet_name]


def read_csv(path, *, profile=None, **kwargs):
    root, source = _source(path)
    if source["format"] != "csv":
        raise ValueError("read_csv requires a logical CSV source")
    sheet = source["sheets"][0]
    if _profile(profile) == "original":
        return pd.read_csv(_member(root, sheet["csv"]), **kwargs)
    rows = _rows(root, sheet, "analysis", source)
    body = io.StringIO(newline="")
    csv.writer(body, lineterminator="\n").writerows(rows)
    body.seek(0)
    return pd.read_csv(body, **kwargs)


class ExcelFile:
    def __init__(self, path, engine=None, **kwargs):
        self.root, self.source = _source(path)
        self.path = self.root / self.source["logical_source"]
        self.sheet_names = [s["name"] for s in self.source["sheets"]]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        pass

    def parse(self, sheet_name=0, **kwargs):
        return read_excel(self, sheet_name=sheet_name, **kwargs)


def source_rows(path, sheet=0, *, original=False):
    root, source = _source(path)
    item = _sheet(source, sheet)
    if original or source["kind"] != "reaction":
        return np.arange(2, item["original_rows"] + 2, dtype=np.int64)
    rows = _rows(root, item, "analysis", source)
    return np.arange(2, len(rows) + 1, dtype=np.int64)


def read_row(path, sheet, source_row, *, original=False, **kwargs):
    coordinates = source_rows(path, sheet, original=original)
    found = np.flatnonzero(coordinates == int(source_row))
    if len(found) != 1:
        raise KeyError(f"Source row absent: {Path(path).name}/{sheet}/{source_row}")
    _, source = _source(path)
    kwargs.setdefault("profile", "original" if original else "analysis")
    frame = read_csv(path, **kwargs) if source["format"] == "csv" else read_excel(path, sheet_name=sheet, **kwargs)
    return frame.iloc[int(found[0])]


def recipe_number_map(recipes, root=None):
    values = list(recipes)
    if values != sorted(set(values)):
        raise ValueError("Full recipe support must be unique and sorted before enumeration")
    return {recipe: number for number, recipe in enumerate(values)}


def condition_aliases(dataset, recipe, condition_ids, root=None):
    values = list(condition_ids)
    if any(not isinstance(value, str) or not value or ";" in value for value in values):
        raise ValueError("Invalid original condition identity")
    return ";".join(values)


def source_evidence(path):
    root, source = _source(path)
    names = sorted({s[k] for s in source["sheets"] for k in ("csv", "types", "row_map") if s.get(k)})
    files = [{"path": name, "sha256": hashlib.sha256(_member(root, name).read_bytes()).hexdigest(),
              "bytes": _member(root, name).stat().st_size} for name in names]
    binding = [{"path": f["path"], "sha256": f["sha256"]} for f in files]
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"logical_source": source["logical_source"], "files": files, "sha256": digest,
            "bytes": sum(f["bytes"] for f in files),
            "mtime_ns": max(_member(root, name).stat().st_mtime_ns for name in names),
            "hash_definition": "sha256 of UTF-8 canonical JSON [{path,sha256}], keys sorted, compact separators; members sorted by path"}


def verify_manifest(root=None):
    root = _root(root)
    manifest = _manifest(root)
    members = manifest["members"]
    if len({m["path"] for m in members}) != len(members):
        raise ValueError("Duplicate support member")
    for member in members:
        path = _member(root, member["path"])
        if path.stat().st_size != member["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Support member digest differs: {member['path']}")
    expected = {s[k] for source in manifest["sources"] for s in source["sheets"]
                for k in ("csv", "types", "row_map") if s.get(k)} | {manifest["exclusions"]["csv"]}
    if expected != {m["path"] for m in members}:
        raise ValueError("Support member closure differs")
    return True


def original_source_count(root=None):
    return sum(sheet["original_rows"] for source in _manifest(_root(root))["sources"]
               if source["kind"] == "reaction" for sheet in source["sheets"]
               if {"condition_id", "sub_1_smiles", "yield"}.issubset(sheet["columns"]))


def pre_applied_exclusion_ids(root=None):
    return frozenset()


def apply_zero_rule_manifest(root=None):
    root = _root(root)
    verify_manifest(root)
    log, deleted = [], []
    for source in _manifest(root)["sources"]:
        source_rows_count = output_rows = zeroed = 0
        matched = []
        for sheet in source["sheets"]:
            rows = _original_rows(root, sheet)
            processed, removed, changed = _process(root, source, rows)
            source_rows_count += len(rows) - 1
            output_rows += len(processed) - 1
            zeroed += changed
            matched.extend(removed)
        deleted.extend(matched)
        log.append({"file": source["logical_source"], "source_rows": source_rows_count,
                    "output_rows": output_rows, "zeroed_cells": zeroed,
                    "deleted_wells": len(matched), "matched_ids": sorted(matched),
                    "csv_source_evidence": source_evidence(root / source["logical_source"])})
    expected = _manifest(root)["exclusions"]["matched_ids"]
    if Counter(deleted) != Counter({value: 1 for value in expected}):
        raise ValueError("Every exclusion must occur exactly once in full original support")
    count = original_source_count(root)
    return {"rule": "blank yield and yield_clf == 0 -> yield 0; blank yield with yield_clf == 1 or no call stays missing",
            "source": str(root), "zeroed_cells": sum(s["zeroed_cells"] for s in log),
            "exclusions": {"count": len(deleted), "matched_ids": sorted(deleted)},
            "original_source_records": count, "retained_source_records": count - len(deleted),
            "files": log, "input_mode": "full original CSV; exclusions and zero rule applied at runtime"}


def validate_registry_scope(recipes, compounds):
    sources = {_source(p)[1]["kind"]: p for p in source_paths()}
    for kind, frame in (("conditions", recipes), ("compounds", compounds)):
        expected = read_excel(sources[kind], dtype=object, keep_default_na=False, profile="original")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected, check_exact=True)
    if recipes.shape != (2364, 31) or recipes.duplicated(["reaction_type", "condition_id"]).any():
        raise ValueError("Full original condition registry differs")
    return True
