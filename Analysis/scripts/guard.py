"""Run an analysis within the reader root. M1_GUARD_ROOT selects the guarded parent; M1_FORBIDDEN_TREES adds forbidden directories."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_raw import M1_ROOT, RAW_DIR
TREE = Path(os.environ.get('M1_GUARD_ROOT', M1_ROOT.parent)).resolve()
EXCLUSION_CSV = os.path.normcase(str(Path(os.environ.get('M1_EXCLUDE_CSV', M1_ROOT / 'Input/EXCLUDE_wells.csv')).resolve()))
ALLOWED = tuple((os.path.normcase(str(p)) + os.sep for p in (M1_ROOT, RAW_DIR)))
TREE_LOWER = os.path.normcase(str(TREE)) + os.sep
FORBIDDEN = tuple((os.path.normcase(os.path.abspath(p)) + os.sep for p in os.environ.get('M1_FORBIDDEN_TREES', '').split(os.pathsep) if p))
EVENTS = {'open', 'os.listdir', 'os.scandir', 'os.stat', 'shutil.copyfile', 'os.rename', 'os.remove'}

def _norm(value) -> str | None:
    if isinstance(value, int):
        return None
    try:
        text = os.fsdecode(os.fspath(value))
    except TypeError:
        return None
    return os.path.normcase(os.path.abspath(text))

def _hook(event, args):
    if event == 'subprocess.Popen':
        print(f'[guard] child process not guarded: {args[1]!r}', file=sys.stderr, flush=True)
        return
    if event not in EVENTS or not args:
        return
    if event in {'os.rename', 'shutil.copyfile'} and _norm(args[1]) == EXCLUSION_CSV:
        raise PermissionError('[guard] cannot overwrite the exclusion input')
    path = _norm(args[0])
    if path == EXCLUSION_CSV:
        if event == 'open':
            mode, flags = (args[1], args[2])
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if (mode is None or not any((c in mode for c in 'wax+'))) and (not flags & write_flags):
                return
        elif event == 'os.stat':
            return
        raise PermissionError(f'[guard] exclusion input is read-only: {path}')
    if path and (path + os.sep).startswith(FORBIDDEN) and (not (path + os.sep).startswith(ALLOWED)):
        raise PermissionError(f'[guard] clean reproduction may not touch {path}')
    if path and (path + os.sep).startswith(TREE_LOWER) and (not (path + os.sep).startswith(ALLOWED)):
        raise PermissionError(f'[guard] M1 pipeline may not touch {path}')

def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    sys.addaudithook(_hook)
    runpy.run_path(str(script), run_name='__main__')
if __name__ == '__main__':
    main()
