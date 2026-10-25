"""Locate the bundled raw reaction tables and registries."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import os
from pathlib import Path
M1_ROOT = Path(__file__).resolve().parents[2]

def raw_dir():
    return reader_tables.data_dir()
RAW_DIR = raw_dir()
