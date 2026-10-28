"""Validate current supplementary figure, caption and Source Data numbering."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

SM_NUMBER = {1: 1, 2: 8, 3: 3, 4: 6, 5: 4, 6: 2, 7: 5, 8: 7}


def validate(manifest, captions):
    figures = manifest['extended_data_figures']
    if sorted(entry['number'] for entry in figures) != list(range(1, 9)):
        raise ValueError('Supplementary figures must contain current numbers S1-S8 exactly once')
    tables = {entry['name']: entry for entry in manifest['source_data']}
    if len(tables) != len(manifest['source_data']):
        raise ValueError('Duplicate Source Data table names')
    ownership = {}
    for figure in figures:
        number = figure['number']
        key = f'Figure_S{number}'
        if SM_NUMBER[figure['legacy_number']] != number:
            raise ValueError(f'Legacy/current numbering mismatch: {key}')
        if figure['caption'] != captions[key]['legend'] or figure['title'] != captions[key]['title']:
            raise ValueError(f'SM caption mismatch: {key}')
        if Path(figure['pdf']).name != f'{key}.pdf':
            raise ValueError(f'Figure filename mismatch: {key}')
        names = figure['source_tables']
        if not names:
            raise ValueError(f'No Source Data table for SM Fig. S{number}')
        for name in names:
            if name not in tables or not name.startswith(f'S{number}_'):
                raise ValueError(f'Source table {name} does not belong to SM Fig. S{number}')
            if name in ownership:
                raise ValueError(f'Source table has multiple figure owners: {name}')
            ownership[name] = number
    for name, table in tables.items():
        match = re.match(r'^S(\d+)_', name)
        if not match:
            continue
        number = int(match.group(1))
        if ownership.get(name) != number or table.get('supplementary_figure') != number:
            raise ValueError(f'Source table/SM caption mapping mismatch: {name}')
        if table.get('caption_key') != f'Figure_S{number}':
            raise ValueError(f'Source table caption key mismatch: {name}')
    return {'figures': len(figures), 'mapped_tables': len(ownership), 'all_figures_have_tables': True}


def reject_wrong_table(manifest, captions):
    wrong = copy.deepcopy(manifest)
    table = next(entry for entry in wrong['source_data'] if re.match(r'^S\d+_', entry['name']))
    old_name = table['name']
    number = int(re.match(r'^S(\d+)_', old_name).group(1))
    table['name'] = re.sub(r'^S\d+_', f'S{number % 8 + 1}_', old_name)
    try:
        validate(wrong, captions)
    except ValueError as error:
        return {'wrong_table_rejected': True, 'original_name': old_name, 'wrong_name': table['name'], 'error': str(error)}
    raise AssertionError('A deliberately wrong supplementary table number was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('captions', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    captions = json.loads(args.captions.read_text(encoding='utf-8'))
    report = validate(manifest, captions)
    if args.self_test:
        report['negative_test'] = reject_wrong_table(manifest, captions)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
