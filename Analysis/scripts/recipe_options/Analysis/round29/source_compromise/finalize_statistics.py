"""Retain complete inventories and define only scientifically meaningful ratios."""
from __future__ import annotations
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import analyze_source_compromise as a

def main():
    heldout = a.read(a.OUT / 'complete_heldout_compromise_statistics.csv.gz')
    inventory = a.read(a.R27 / 'complete_20_dataset_18_class_support.csv')
    focal = a.read(a.R27 / 'complete_focal_key_registry.csv.gz')
    topologies = pd.DataFrame({'endpoint_topology': ['all', 'adjacent', 'nonadjacent']})
    units = [('focal_key', 'focal_key_id', focal[['focal_key_id', 'dataset_id', 'rxn_class', 'molecular_pair_id']]), ('dataset', 'dataset_id', inventory[['dataset_id', 'rxn_class']].drop_duplicates()), ('reaction_class', 'rxn_class', inventory[['rxn_class']].drop_duplicates())]
    expected = {'focal_key': 512, 'dataset': 20, 'reaction_class': 18}
    for unit, key, base in units:
        assert len(base) == expected[unit]
        pooled = heldout.groupby([key, 'endpoint_topology'], as_index=False)[a.FIELDS].sum()
        frame = base.merge(topologies, how='cross').merge(pooled, on=[key, 'endpoint_topology'], how='left', validate='one_to_one')
        frame[a.FIELDS] = frame[a.FIELDS].fillna(0.0)
        frame['conditional_opportunity'] = frame.n_endpoints > 0
        a.save(frame, 'complete_' + unit + '_compromise_statistics.csv' + ('.gz' if unit == 'focal_key' else ''))
    weight_rows = []
    pairs = [(num, den) for num in ['n_frozen_failure_tie_rescue_only20', 'n_frozen_failure_positive_source_compromise20'] for den in ['n_endpoints', 'n_union_feasible20', 'n_frozen_failure20']]
    pairs += [('source_sacrifice_union20_sum', 'n_union_feasible20'), ('source_sacrifice_positive_union20_sum', 'n_positive_sacrifice_union20')]
    for topology in ['all', 'nonadjacent', 'adjacent']:
        for native_scope in ['all', 'native_support_equal', 'native_support_has_extra']:
            frame = heldout[heldout.endpoint_topology == topology]
            if native_scope != 'all':
                frame = frame[frame.all_native_pair_supports_identical_to_full_support == (native_scope == 'native_support_equal')]
            for unit, key in [('endpoint', None), ('heldout', None), ('focal_key', 'focal_key_id'), ('dataset', 'dataset_id'), ('reaction_class', 'rxn_class')]:
                mm = frame[a.FIELDS] if key is None else frame.groupby(key)[a.FIELDS].sum()
                for num, den in pairs:
                    active = mm[den] > 0
                    value = (float(mm.loc[active, num].sum() / mm.loc[active, den].sum()) if unit == 'endpoint' else float((mm.loc[active, num] / mm.loc[active, den]).mean())) if active.any() else None
                    weight_rows.append({'endpoint_topology': topology, 'native_scope': native_scope, 'weighting_unit': unit, 'numerator': num, 'denominator': den, 'value': value, 'n_active_units': int(active.sum()) if unit != 'endpoint' else int(mm.loc[active, den].sum()), 'denominator_endpoint_views': int(mm.loc[active, den].sum()), 'unit_rule': 'endpoint pooled; otherwise endpoint ratio within unit then equal mean across units with positive denominator'})
    a.save(pd.DataFrame(weight_rows), 'compromise_weighting_summary.csv')
    summary = json.loads((a.OUT / 'run_summary.json').read_text(encoding='utf-8'))
    summary['n_focal_keys_inventory'] = 512
    summary['n_datasets_inventory'] = 20
    summary['n_reaction_classes_inventory'] = 18
    summary['n_focal_keys_with_coherent_triples'] = heldout.focal_key_id.nunique()
    summary['n_datasets_with_coherent_triples'] = heldout.dataset_id.nunique()
    summary['n_reaction_classes_with_coherent_triples'] = heldout.rxn_class.nunique()
    a.write_json(summary, 'run_summary.json')
    counts = a.read(a.OUT / 'class_and_hierarchy_coverage.csv')
    assert counts.query("endpoint_topology == 'all'").n_endpoint_views.iloc[:8].sum() == 9631618
    print('INVENTORIES', {k: v for k, v in expected.items()}, flush=True)
if __name__ == '__main__':
    main()
