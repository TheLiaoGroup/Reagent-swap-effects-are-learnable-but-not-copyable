"""Check structure bounding boxes in all four main and eight supplement figures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from publication_figures import common as c

EXPECTED_FIGURES = tuple([f'Figure_{i}' for i in range(1, 5)] +
                         [f'Figure_S{i}' for i in range(1, 9)])


def check_reports(report_dir, negative_control=None):
    results = []
    for stem in EXPECTED_FIGURES:
        path = Path(report_dir) / f'{stem}_manifest.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('figure') != stem:
            raise ValueError(f'Unexpected figure identity in {path.name}')
        boxes = c.structure_boxes_pt(report)
        hits = c.structure_box_intersections(report)
        results.append({'figure': stem, 'structures_checked': len(boxes),
                        'structure_pairs_checked': len(boxes) * (len(boxes) - 1) // 2,
                        'structure_structure_box_intersections': hits,
                        'passed': not hits})
    result = {'figures_checked': len(results), 'figures': results,
              'passed': all(item['passed'] for item in results)}
    if negative_control is not None:
        path = Path(negative_control)
        report = json.loads(path.read_text(encoding='utf-8'))
        hits = c.structure_box_intersections(report)
        rejected = bool(hits)
        result['negative_control'] = {'figure': report.get('figure'),
                                      'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'structure_structure_box_intersections': hits,
                                      'rejected': rejected}
        result['passed'] = result['passed'] and rejected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=c.OUT)
    parser.add_argument('--negative-control', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_reports(args.report_dir, args.negative_control)
    value = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value, encoding='utf-8')
    print(value)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
