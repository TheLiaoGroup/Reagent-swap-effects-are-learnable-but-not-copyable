"""Supplementary high-clearance witness, explicitly retaining support truncation."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import pandas as pd
import analyze_shared_bridges as a
if __name__ == '__main__':
    candidates = pd.read_csv(a.OUT / 'all_strong_scope_reconciliation20.csv.gz', float_precision='round_trip')
    candidate = candidates.sort_values(['complete_endpoint_corners', 'two_sided_clearance', 'full_minus_endpoint_gain'], ascending=False).iloc[0].to_dict()
    groups, contexts, conditions, supports, values = a.load()
    a.OUT = a.OUT / 'large_gain_witness'
    a.OUT.mkdir(exist_ok=True)
    a.witness(candidate, groups, contexts, conditions, supports, values)
    selected = a.OUT / 'chemical_witness.json'
    import json
    data = json.loads(selected.read_text(encoding='utf-8'))
    data['selection'] = 'Supplementary outcome-selected witness: complete endpoint corners, largest two-sided20% clearance, then largest continuous gain; native pair support equality not required.'
    data['support_truncation_limitation'] = 'All paths and token-deletion claims are conditional on this fixed six-target common measured graph; native pair supports include additional recipes.'
    selected.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(data, ensure_ascii=True))
