"""Run the REFERENCE_LEARNING dependency closure in order, each step under scripts/guard.py.

Usage: python run_reference_learning.py [--from STEP] [--only STEP ...] [--list]
Logs go to logs/<nn>_<step>.log; per-step wall time is appended to logs/runtimes.csv."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / 'Analysis'
GUARD = HERE.parent / 'guard.py'
LOGS = HERE / 'logs'
STEPS = [('raw_audit', ['raw_audit.py']), ('condition_audit', ['condition_audit.py']), ('run_existing_evidence', ['run_existing_evidence.py', '--stage', 'all']), ('chemical_response_structure', ['chemical_response_structure.py']), ('chemical_context_replication', ['chemical_context_replication.py']), ('chemical_mode_recovery', ['chemical_mode_recovery.py']), ('chemically_aligned_measurements', ['chemically_aligned_measurements.py']), ('chemical_observation_specificity', ['chemical_observation_specificity.py']), ('chemical_observation_channels', ['chemical_observation_channels.py']), ('compute_matched_copy_comparison', ['Nature_Manuscript/compute_matched_copy_comparison.py']), ('supplementary_table9', ['Nature_Manuscript/supplementary_table9.py']), ('cross_library_component_structure', ['cross_library_component_structure.py']), ('controlled_cross_library_bridges', ['controlled_cross_library_bridges.py']), ('transport_lattice_composition', ['transport_lattice_composition.py']), ('shared_latent_dimension', ['shared_latent_dimension.py']), ('transport_domain_atlas', ['transport_domain_atlas.py']), ('response_estimand_lineage', ['response_estimand_lineage.py']), ('verify_response_estimand_lineage', ['verify_response_estimand_lineage.py']), ('budgeted_target_calibration', ['budgeted_target_calibration.py']), ('verify_budgeted_target_calibration', ['verify_budgeted_target_calibration.py']), ('source_identity_alignment', ['source_identity_alignment.py']), ('verify_source_identity_alignment', ['verify_source_identity_alignment.py']), ('cross_library_source_channels', ['cross_library_source_channels.py'])]

def run(index, name, argv):
    script = ANALYSIS / argv[0]
    log = LOGS / f'{index:02d}_{name}.log'
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.call([sys.executable, '-B', str(GUARD), str(script), *argv[1:]], cwd=script.parent, env=env, stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - start
    with (LOGS / 'runtimes.csv').open('a', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerow([index, name, code, f'{seconds:.1f}', time.strftime('%Y-%m-%d %H:%M:%S')])
    print(f'{index:02d} {name}: exit {code}, {seconds:.1f}s', flush=True)
    if code:
        print(log.read_text(encoding='utf-8')[-4000:], flush=True)
        raise SystemExit(code)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='start')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    names = [name for name, _ in STEPS]
    if args.list:
        print('\n'.join((f'{i:02d} {n}' for i, n in enumerate(names, 1))))
        return
    for folder in ('data', 'tables', 'figures'):
        (ANALYSIS / folder).mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    first = names.index(args.start) if args.start else 0
    for index, (name, argv) in enumerate(STEPS, 1):
        if index - 1 < first or (args.only and name not in args.only):
            continue
        run(index, name, argv)
if __name__ == '__main__':
    main()
