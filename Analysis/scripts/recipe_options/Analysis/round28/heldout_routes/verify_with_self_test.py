"""M1 port: execute the independent heldout QA and original three-group self-test."""
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
from pathlib import Path
import sys
SCRIPTS = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPTS / 'recipe_options/Analysis/code'))
from run_qa_commands import run_stage

def configuration(analysis=None):
    base = 'scripts/recipe_options/Analysis/round28/heldout_routes/'
    source = 'scripts/recipe_options/Analysis/round27/partner_scope/triple_scope/'
    independent = base + 'independent_verification.json'
    self_test = base + 'qa/self_test_verification.json'
    return {'analysis': Path(analysis) if analysis is not None else SCRIPTS.parent, 'stage': 'heldout_routes', 'entrypoint': base + 'verify_with_self_test.py', 'code_inputs': ['scripts/guard.py', 'scripts/m1_raw.py', 'scripts/m1_exclusions.py', 'scripts/recipe_options/Analysis/code/run_qa_commands.py', base + 'verify_heldout_routes.py', base + 'analyze_heldout_routes.py'], 'inputs': [source + name for name in ['complete_group_registry_before_outcomes.csv.gz', 'complete_focal_key_registry.csv.gz', 'complete_context_registry.csv.gz', 'full_source_block_node_yields.csv.gz', 'complete_support_registry.csv.gz', 'full_support_edges.csv.gz', 'full_support_nodes.csv.gz', 'full_condition_registry.csv.gz']] + [base + name for name in ['complete_heldout_statistics.csv.gz', 'complete_heldout_registry_before_target_outcomes.csv.gz', 'complete_endpoint_results.csv.gz', 'measured_witness.json', 'design_census.json', 'complete_470_focal_key_statistics.csv.gz', 'complete_20_dataset_statistics.csv', 'complete_18_class_statistics.csv', 'continuous_and_threshold_weighting_summary.csv']] + ['out/curated_inputs_zero_rule_manifest.json'], 'commands': [{'script': base + 'verify_heldout_routes.py', 'arguments': [], 'receipts': {independent: 'passed'}}, {'script': base + 'verify_heldout_routes.py', 'arguments': ['--self-test'], 'receipts': {self_test: 'passed'}, 'self_test': True}], 'outputs': [independent, self_test]}

def main():
    run_stage(configuration())
if __name__ == '__main__':
    main()
