"""M1 port: require fresh independent/raw QA before the component summary."""
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
    base = 'scripts/recipe_options/Analysis/round29/component_options/'
    source = 'scripts/recipe_options/Analysis/round27/partner_scope/triple_scope/'
    bridges = 'scripts/recipe_options/Analysis/round28/shared_bridges/'
    independent = base + 'independent_verification.json'
    raw = base + 'chemical_witness_raw_verification.json'
    status = base + 'witness_source_status.json'
    delivery = base + 'delivery_manifest.json'
    return {'analysis': Path(analysis) if analysis is not None else SCRIPTS.parent, 'stage': 'component_options', 'entrypoint': base + 'verify_and_summarize.py', 'code_inputs': ['scripts/guard.py', 'scripts/m1_raw.py', 'scripts/m1_exclusions.py', 'scripts/recipe_options/Analysis/code/run_qa_commands.py', base + 'verify_component_options.py', base + 'verify_raw_component_witnesses.py', base + 'summarize_component_options.py', base + 'analyze_component_options.py'], 'inputs': [base + name for name in ['all_single_options.csv.gz', 'all_pair_options.csv.gz', 'all_triple_options.csv.gz', 'certificate_option_summary.csv', 'chemical_witness_routes.csv', 'complete_class_option_summary.csv', 'chemical_witness_source_records.csv.gz', 'chemical_witness_conditions.csv', 'chemical_witness_full_node_yields.csv', 'named_option_registry.csv', 'named_option_descriptive_scope.csv', 'run_summary.json']] + [source + name for name in ['complete_group_registry.csv.gz', 'complete_context_registry.csv.gz', 'complete_support_registry.csv.gz', 'full_condition_registry.csv.gz', 'full_source_block_node_yields.csv.gz', 'complete_20_dataset_18_class_support.csv']] + [bridges + name for name in ['all_strong_scope_reconciliation20.csv.gz', 'chemical_witness_source_records.csv.gz', 'chemical_witness_source_verification.json', 'large_gain_witness/chemical_witness_source_records.csv.gz', 'large_gain_witness/chemical_witness_source_verification.json']] + ['scripts/recipe_options/Analysis/round7/joint_reactant_connectivity/all_strict_source_records.csv.gz', 'out/curated_inputs_zero_rule_manifest.json'], 'commands': [{'script': base + 'verify_component_options.py', 'arguments': [], 'receipts': {independent: 'passed'}}, {'script': base + 'verify_raw_component_witnesses.py', 'arguments': [], 'receipts': {raw: 'passed'}}, {'script': base + 'summarize_component_options.py', 'arguments': [], 'receipts': {status: 'all_five_witnesses_raw_readback_passed', delivery: None}}], 'outputs': [independent, raw, status, delivery]}

def main():
    run_stage(configuration())
if __name__ == '__main__':
    main()
