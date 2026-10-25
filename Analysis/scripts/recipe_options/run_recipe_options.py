"""Run the ported RECIPE_OPTIONS chain (raw corpus -> Fig. 2 / repair / scope numbers) in order, each step under guard.py.

Usage: python scripts/recipe_options/run_recipe_options.py [--from STEP] [--to STEP] [--list]
Steps are numbered; runtimes are appended to scripts/recipe_options/Analysis/run_times.json."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
GUARD = HERE.parent / 'guard.py'
A = HERE / 'Analysis'
STEPS = [('code/ingest_raw.py', []), ('code/structure_features.py', []), ('code/mine_components.py', []), ('code/recompute_flagship_cases.py', []), ('code/analyze_structure_rules.py', []), ('code/synthesize_evidence.py', []), ('code/round3_global_scope.py', []), ('code/round3_context_rectangles.py', []), ('code/round4_context_transport.py', []), ('code/round4_structure_context.py', []), ('code/round5_joint_accessibility.py', []), ('code/round6_molecular_rewiring.py', []), ('code/round6_molecular_connectivity.py', []), ('code/round6_path_organization.py', []), ('code/round7_partner_conditioned_edits.py', []), ('code/round7_joint_reactant_connectivity.py', []), ('code/round7_bridge_domains.py', []), ('code/round8_context_decomposition.py', []), ('code/round8_partner_neighborhoods.py', []), ('code/round9_shared_recipe_paths.py', []), ('code/round9_pairing_repair.py', []), ('round22/molecular_relation_geometry/round22_molecular_relation_geometry.py', []), ('round26/support_fixed_routes/round26_support_fixed_routes.py', []), ('round27/partner_scope/round27_partner_scope.py', []), ('round27/partner_scope/round27_partner_scope.py', ['--triple-frame']), ('round27/partner_scope/verify_partner_scope.py', []), ('round27/partner_scope/verify_partner_scope.py', ['--triple-frame']), ('round27/partner_scope/finalize_partner_scope.py', []), ('round27/route_compromise/analyze_route_compromise.py', []), ('round27/route_compromise/export_route_frontiers.py', []), ('round28/shared_bridges/analyze_shared_bridges.py', []), ('round28/shared_bridges/verify_raw_witness.py', []), ('round28/shared_bridges/export_large_gain_witness.py', []), ('round28/shared_bridges/verify_raw_witness.py', ['--witness-dir', 'large_gain_witness']), ('round28/heldout_routes/analyze_heldout_routes.py', ['--limit', '500']), ('round28/heldout_routes/analyze_heldout_routes.py', []), ('round28/heldout_routes/export_measured_witness.py', []), ('round28/heldout_routes/finalize_metadata.py', []), ('round28/heldout_routes/summarize_diagnostics.py', []), ('round28/heldout_routes/verify_with_self_test.py', []), ('round28/heldout_routes/build_report.py', []), ('round29/source_compromise/analyze_source_compromise.py', []), ('round29/source_compromise/export_measured_cases.py', []), ('round29/component_options/analyze_component_options.py', []), ('round29/component_options/verify_and_summarize.py', []), ('round29/option_scope_domains/analyze_option_scope_domains.py', [])]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='start', type=int, default=1)
    parser.add_argument('--to', dest='stop', type=int, default=len(STEPS))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        for i, (script, extra) in enumerate(STEPS, 1):
            print(i, script, *extra)
        return
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    log_path = A / 'run_times.json'
    for i, (script, extra) in enumerate(STEPS, 1):
        if not args.start <= i <= args.stop:
            continue
        print(f"[{i}/{len(STEPS)}] {script} {' '.join(extra)}", flush=True)
        started = time.time()
        subprocess.run([sys.executable, '-u', str(GUARD), str(A / script), *extra], check=True, env=env, cwd=str((A / script).parent))
        seconds = round(time.time() - started, 1)
        log = json.loads(log_path.read_text(encoding='utf-8')) if log_path.is_file() else {}
        log[f"{i:02d} {script} {' '.join(extra)}".strip()] = seconds
        log_path.write_text(json.dumps(log, indent=2) + '\n', encoding='utf-8')
        print(f'    {seconds} s', flush=True)
if __name__ == '__main__':
    main()
