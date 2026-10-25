"""Export native ChemDraw vectors in a disposable Python COM client."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import sys
from m1_chemistry import SETTINGS

@contextmanager
def _opened_document(documents, path):
    document = documents.Open(str(Path(path).resolve()))
    if document is None:
        raise RuntimeError('ChemDraw could not open structure')
    try:
        yield document
    finally:
        failure = sys.exception()
        try:
            document.Close(False)
        except Exception as close_error:
            if failure is None:
                raise
            failure.add_note(f'ChemDraw document Close also failed: {close_error}')
        finally:
            document = None

def _write_native(path, value):
    Path(path).write_bytes(value.encode('utf-8') if isinstance(value, str) else bytes(value))

def export_native(template, source, stem, readback, dispatch):
    app = documents = template_document = template_settings = None
    document = document_settings = objects = object_settings = None
    try:
        app = dispatch('ChemDraw.Application.22.0')
        documents = app.Documents
        with _opened_document(documents, template) as template_document:
            template_settings = template_document.Settings
            settings = {field: getattr(template_settings, field) for field in SETTINGS}
        template_document = template_settings = None
        with _opened_document(documents, source) as document:
            document_settings, objects = (document.Settings, document.Objects)
            object_settings = objects.Settings
            for field in SETTINGS:
                setattr(document_settings, field, settings[field])
                setattr(object_settings, field, settings[field])
            base = str(Path(stem).resolve())
            _write_native(base + '.cdxml', objects.GetData('text/xml'))
            Path(base + '.cdx').write_bytes(bytes(objects.GetData('chemical/x-cdx')))
            _write_native(base + '.svg', objects.GetData('image/svg+xml'))
            _write_native(readback, objects.GetData('chemical/x-mdl-molfile'))
    finally:
        object_settings = objects = document_settings = document = None
        template_settings = template_document = documents = app = None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('stem', type=Path)
    parser.add_argument('readback', type=Path)
    args = parser.parse_args()
    import pythoncom
    from win32com.client.gencache import EnsureDispatch
    pythoncom.CoInitialize()
    try:
        export_native(args.template, args.source, args.stem, args.readback, EnsureDispatch)
    finally:
        pythoncom.CoUninitialize()
if __name__ == '__main__':
    main()
