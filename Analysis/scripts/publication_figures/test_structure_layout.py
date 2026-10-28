"""Regression coverage for collisions between independently placed native structures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt

from publication_figures import common as c
from publication_figures.redesign_figures234 import layout_review
from publication_figures.check_structure_layout import EXPECTED_FIGURES, check_reports


# Measured native bounds from the overlapping pre-v16B Figure 2 manifest.
OVERLAPPING_FIGURE2 = {
    'figure': 'Figure_2',
    'height_mm': 164.99999533759222,
    'chemistry': [
        {'smiles': 'Brc1ccc2[nH]ncc2c1',
         'bounds_pt': [50.24845733642578, 303.3232748031616,
                       55.12513732910156, 34.12508583068848]},
        {'smiles': 'COc1cccnc1Br',
         'bounds_pt': [56.248471641540526, 331.0112661361694,
                       43.12510871887207, 34.87508583068848]},
    ],
}


class StructureLayoutTests(unittest.TestCase):
    def test_layout_review_rejects_actual_overlapping_figure2_structures(self):
        fig = c.newfig(165, width_mm=183)
        try:
            review = layout_review(fig, deepcopy(OVERLAPPING_FIGURE2))
            collisions = review.get('structure_structure_box_intersections', [])
            self.assertEqual(len(collisions), 1)
            self.assertAlmostEqual(collisions[0]['intersection_pt'][2], 43.12510871887207)
            self.assertAlmostEqual(collisions[0]['intersection_pt'][3], 6.437094497680685)
        finally:
            plt.close(fig)

    def test_native_bounds_not_nominal_placement_slots_determine_intersections(self):
        report = deepcopy(OVERLAPPING_FIGURE2)
        for item in report['chemistry']:
            item['rect'] = [0, 0, 0.01, 0.01]
        self.assertEqual(len(c.structure_box_intersections(report)), 1)
        report['chemistry'][1]['bounds_pt'][1] = 340.3655965805053
        self.assertEqual(c.structure_box_intersections(report), [])

    def test_custom_svg_figure1_mm_bounds_and_standard_point_boxes_agree(self):
        report = deepcopy(OVERLAPPING_FIGURE2)
        report['figure'] = 'Figure_1'
        for item in report['chemistry']:
            item['bounds_mm'] = [value * 25.4 / 72 for value in item.pop('bounds_pt')]
        hits = c.structure_box_intersections(report)
        self.assertEqual(len(hits), 1)
        report['chemical_boxes'] = [dict(zip(('x_pt', 'y_pt', 'width_pt', 'height_pt'),
                                            [value * 72 / 25.4 for value in item['bounds_mm']]))
                                    for item in report['chemistry']]
        self.assertEqual(c.structure_box_intersections(report), hits)

    def test_touching_edges_have_no_positive_area_but_tiny_overlap_is_rejected(self):
        report = {'chemical_boxes': [{'x_pt': 0, 'y_pt': 0, 'width_pt': 10, 'height_pt': 10},
                                     {'x_pt': 10, 'y_pt': 0, 'width_pt': 10, 'height_pt': 10}]}
        self.assertEqual(c.structure_box_intersections(report), [])
        report['chemical_boxes'][1]['x_pt'] -= 0.00001
        self.assertEqual(len(c.structure_box_intersections(report)), 1)

    def test_missing_malformed_or_nonfinite_inventory_fails_closed(self):
        reports = [{}, {'chemistry': [{}]}, {'chemistry': [{}], 'chemical_boxes': []},
                   {'chemistry': [{'bounds_pt': [0, 0, float('nan'), 10]}]},
                   {'chemistry': [{'bounds_pt': [0, 0, -1, 10]}]}]
        for report in reports:
            with self.subTest(report=report), self.assertRaises(ValueError):
                c.structure_box_intersections(report)

    def test_all_twelve_reports_and_actual_old_negative_control_are_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for stem in EXPECTED_FIGURES:
                report = {'figure': stem, 'chemistry': []}
                (root / f'{stem}_manifest.json').write_text(json.dumps(report), encoding='utf-8')
            negative = root / 'before_manifest.json'
            negative.write_text(json.dumps(OVERLAPPING_FIGURE2), encoding='utf-8')
            result = check_reports(root, negative)
            self.assertTrue(result['passed'])
            self.assertEqual(result['figures_checked'], 12)
            self.assertTrue(result['negative_control']['rejected'])
            (root / 'Figure_S8_manifest.json').write_text(
                json.dumps({**OVERLAPPING_FIGURE2, 'figure': 'Figure_S8'}), encoding='utf-8')
            result = check_reports(root, negative)
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['figures'][-1]['structure_structure_box_intersections']), 1)
            (root / 'Figure_S1_manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):
                check_reports(root)


if __name__ == '__main__':
    unittest.main()
