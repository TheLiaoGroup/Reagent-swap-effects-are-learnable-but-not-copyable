"""Small delivery-manifest checks without running the RECIPE_OPTIONS analysis chain."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
SPEC = importlib.util.spec_from_file_location('finalize_partner_scope', Path(__file__).with_name('finalize_partner_scope.py'))
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)
NO_CERTIFICATE = 'no_pairwise_feasible_full_scope_failed_certificate20'

def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class DeliveryManifestTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='m1_recipe_options_finalizer_')
        self.out = Path(self.temporary.name)
        self.original_out = FINALIZER.OUT
        FINALIZER.OUT = self.out
        for name in ('round27_partner_scope.py', 'verify_partner_scope.py', 'finalize_partner_scope.py'):
            (self.out / name).write_text('# fixture\n', encoding='utf-8')
        for dest in (self.out, self.out / 'triple_scope'):
            dest.mkdir(exist_ok=True)
            self.frame(dest)

    def tearDown(self):
        FINALIZER.OUT = self.original_out
        self.temporary.cleanup()

    def frame(self, dest):
        write_json(dest / 'independent_verification.json', {'status': 'passed'})
        write_json(dest / 'source_manifest.json', {'generator': {'path': 'initial.py'}})
        write_json(dest / 'chemical_witness.json', {'group_id': 1, 'n_partners': 3, 'n_acquisition_contexts': 3, 'partners': ['partner_a', 'partner_b', 'partner_c']})
        write_json(dest / 'chemical_witness_source_verification.json', {'status': 'passed', 'original_workbook_sha256': {'raw.xlsx': 'fixture_hash'}})
        write_csv(dest / 'complete_threshold_crossclassification.csv', [{'threshold': threshold, 'n_endpoint_views': 4} for threshold in (0.1, 0.2, 0.3)])
        write_csv(dest / 'summary.csv', [{'stratum': 'groups_ge3', 'dimension': 'all', 'threshold': threshold, 'n_connected_endpoint_pairs': 4} for threshold in (0.1, 0.2, 0.3)])

    def no_certificate(self):
        write_json(self.out / 'chemical_witness.json', {'status': NO_CERTIFICATE})
        (self.out / 'chemical_witness_source_verification.json').unlink()

    def test_certificate_frame_retains_checks_and_workbook_hashes(self):
        FINALIZER.delivery_manifests()
        for dest in (self.out, self.out / 'triple_scope'):
            result = json.loads((dest / 'delivery_manifest.json').read_text(encoding='utf-8'))
            source = json.loads((dest / 'source_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(result['scope_metadata_check'], 'passed')
            self.assertEqual(result['crossclassification_sum_check'], 'passed')
            self.assertEqual(source['original_witness_workbooks'], {'raw.xlsx': 'fixture_hash'})
            self.assertTrue(all((file['sha256'] for file in result['files'])))

    def test_no_certificate_frame_records_status_without_raw_witness_file(self):
        self.no_certificate()
        FINALIZER.delivery_manifests()
        result = json.loads((self.out / 'delivery_manifest.json').read_text(encoding='utf-8'))
        source = json.loads((self.out / 'source_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['chemical_witness_status'], NO_CERTIFICATE)
        self.assertEqual(source['chemical_witness_status'], NO_CERTIFICATE)
        self.assertEqual(source['original_witness_workbooks'], {})
        self.assertEqual(result['scope_metadata_check'], 'skipped_no_witness')
        self.assertEqual(result['witness_source_verification_check'], 'skipped_no_witness')
        self.assertEqual(result['crossclassification_sum_check'], 'passed')
        triple = json.loads((self.out / 'triple_scope/delivery_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(triple['scope_metadata_check'], 'passed')

    def test_unknown_witness_status_remains_fatal(self):
        write_json(self.out / 'chemical_witness.json', {'status': 'unexpected_failure'})
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()

    def test_no_certificate_with_group_id_remains_fatal(self):
        write_json(self.out / 'chemical_witness.json', {'status': NO_CERTIFICATE, 'group_id': 1})
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()

    def test_no_certificate_independent_verification_remains_fatal(self):
        self.no_certificate()
        write_json(self.out / 'independent_verification.json', {'status': 'failed'})
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()

    def test_no_certificate_crossclassification_remains_fatal(self):
        self.no_certificate()
        write_csv(self.out / 'complete_threshold_crossclassification.csv', [{'threshold': threshold, 'n_endpoint_views': 3} for threshold in (0.1, 0.2, 0.3)])
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()

    def test_certificate_raw_verification_remains_fatal(self):
        write_json(self.out / 'chemical_witness_source_verification.json', {'status': 'failed'})
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()

    def test_certificate_scope_metadata_remains_fatal(self):
        write_json(self.out / 'chemical_witness.json', {'group_id': 1, 'n_partners': 3, 'n_acquisition_contexts': 2, 'partners': ['partner_a', 'partner_b', 'partner_c']})
        with self.assertRaises(AssertionError):
            FINALIZER.delivery_manifests()
if __name__ == '__main__':
    unittest.main()
