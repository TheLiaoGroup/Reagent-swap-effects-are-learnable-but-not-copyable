"""Run the ported RESPONSE_LEARNING chain from the raw corpus, every step under scripts/guard.py.

Usage:
    python scripts/response_learning/run_response_learning.py              # whole chain
    python scripts/response_learning/run_response_learning.py --from e62   # resume at a step (name or 1-based number)
    python scripts/response_learning/run_response_learning.py --only e63   # one step
    python scripts/response_learning/run_response_learning.py --list"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
GUARD = HERE.parent / 'guard.py'
EDM = 'Existing_Data_Mining_2026-10-16'
STEPS = [('prepare_data', f'{EDM}/code/prepare_data.py', []), ('chemical_features', f'{EDM}/code/chemical_features.py', []), ('mine_swaps', f'{EDM}/code/mine_swaps.py', []), ('condition_network_meta', f'{EDM}/code/condition_network_meta.py', []), ('context_portability', f'{EDM}/code/context_portability_audit.py', []), ('direction_magnitude', f'{EDM}/code/direction_magnitude_decomposition.py', []), ('background_portability', f'{EDM}/code/background_portability_overlap.py', []), ('matched_response_transport', f'{EDM}/code/matched_response_transport.py', []), ('matched_sign_information', f'{EDM}/code/matched_sign_information.py', []), ('monotone_sign_separability', f'{EDM}/code/monotone_sign_separability.py', []), ('sparse_sign_tests', f'{EDM}/audit/sparse_sign_feasibility_tests.py', []), ('sparse_sign_feasibility', f'{EDM}/code/sparse_sign_feasibility.py', []), ('focused_chemical_evidence', f'{EDM}/code/focused_chemical_evidence.py', []), ('e59', 'Full_Edge_Corpus_E59/code/build_full_edge_corpus.py', []), ('e62', 'Matched_Representation_E62/code/finalize_e62.py', ['--allow-fitting']), ('e62_summary', 'Matched_Representation_E62/code/summarize_representation.py', []), ('e63', 'Response_Information_Decomposition_E63/code/decompose_response_information.py', []), ('e64', 'Target_Geometry_E64/code/target_geometry.py', []), ('e65_plan', 'Structure_Family_Transport_E65/code/build_e65_plan.py', []), ('e65_fit', 'Structure_Family_Transport_E65/code/fit_family_transport.py', []), ('e66_scan', 'Family_Information_Recovery_E66/code/scan_e66_feasibility.py', []), ('e66_plan', 'Family_Information_Recovery_E66/code/build_e66_plan.py', []), ('e66_fit', 'Family_Information_Recovery_E66/code/fit_family_recovery.py', []), ('e66_aggregate', 'Family_Information_Recovery_E66/code/aggregate_family_recovery.py', []), ('e67_fit', 'Local_Information_Content_E67/code/fit_e67_content.py', []), ('e67_aggregate', 'Local_Information_Content_E67/code/aggregate_e67_content.py', [])]

def index_of(token: str) -> int:
    if token.isdigit():
        return int(token) - 1
    names = [name for name, _, _ in STEPS]
    if token not in names:
        raise SystemExit(f'Unknown step {token!r}; use --list')
    return names.index(token)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='start')
    parser.add_argument('--only')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        for i, (name, script, extra) in enumerate(STEPS, 1):
            print(f"{i:2d} {name:28s} {script} {' '.join(extra)}")
        return
    if args.only:
        selected = [STEPS[index_of(args.only)]]
    else:
        selected = STEPS[index_of(args.start) if args.start else 0:]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    total = time.perf_counter()
    for name, script, extra in selected:
        started = time.perf_counter()
        print(f"=== {name}: {script} {' '.join(extra)}", flush=True)
        result = subprocess.run([sys.executable, str(GUARD), str(HERE / script), *extra], cwd=HERE, env=env)
        print(f'=== {name}: exit {result.returncode}, {time.perf_counter() - started:.1f}s', flush=True)
        if result.returncode:
            raise SystemExit(f'Step {name} failed')
    print(f'=== chain complete: {time.perf_counter() - total:.1f}s', flush=True)
if __name__ == '__main__':
    main()
