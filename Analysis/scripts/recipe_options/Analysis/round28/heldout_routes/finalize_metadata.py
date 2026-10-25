"""Complete source-feasibility statuses, reuse census and exact-pair weights."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
POLICIES = ['source_maximin', 'source_mean_bottleneck', 'full_graph_shortest']

def read(name):
    return pd.read_csv(OUT / name, keep_default_na=False, float_precision='round_trip', low_memory=False)

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'compresslevel': 2} if name.endswith('.gz') else None)

def main():
    ag = read('complete_heldout_statistics.csv.gz')
    reg = ag[['group_id', 'heldout_position', 'focal_key_id', 'dataset_id', 'rxn_class', 'molecular_pair_id', 'heldout_context_id', 'source_context_ids', 'support_id', 'n_nodes', 'n_edges', 'support_status', 'n_source_high_nodes', 'n_source_endpoint_pairs', 'n_connected_source_endpoint_pairs', 'n_source_route_feasible20', 'n_source_route_failed20']].copy()
    reg['source_route_status'] = np.select([reg.n_connected_source_endpoint_pairs == 0, reg.n_source_route_feasible20 == 0, reg.n_source_route_failed20 == 0], ['no_connected_source_selected_endpoint_pairs', 'all_connected_source_selected_pairs_source_route_failed20', 'all_connected_source_selected_pairs_source_route_feasible20'], default='mixed_source_route_feasible_and_failed_pairs20')
    save(reg, 'complete_heldout_source_feasibility_registry.csv.gz')
    rows = []
    strata = {'all_holdouts': None, 'has_source_selected_endpoints': 'n_source_endpoint_pairs', 'connected': 'n_connected_source_endpoint_pairs', 'source_feasible': 'n_source_route_feasible20', 'source_feasible_target_endpoint_failure': 'n_source_feasible_target_endpoint_failures20', 'source_feasible_target_own_internal_failure': 'n_source_feasible_target_own_internal_failures20', 'separately_feasible_shared_impossible': 'n_separately_feasible_shared_impossible20', 'union_feasible': 'n_union_route_feasible20'}
    for policy in POLICIES:
        strata[policy + '_avoidable_selection_failure'] = policy + '_n_avoidable_selection_failure20'
    for stratum, field in strata.items():
        pp = ag if field is None else ag[ag[field] > 0]
        rows.append({'stratum': stratum, 'n_endpoint_views': int(pp[field].sum()) if field else int(pp.n_source_endpoint_pairs.sum()), 'n_heldout_assignments': len(pp), 'n_triples': pp.group_id.nunique(), 'n_focal_keys': pp.focal_key_id.nunique(), 'n_global_exact_pairs': pp.molecular_pair_id.nunique(), 'n_heldout_acquisition_contexts': pp.heldout_context_id.nunique(), 'n_datasets': pp.dataset_id.nunique(), 'n_reaction_classes': pp.rxn_class.nunique(), 'interpretation': 'reused measured units; endpoint counts are not independent experiments'})
    save(pd.DataFrame(rows), 'coverage_and_reuse.csv')
    sumcols = [c for c in ag if c.endswith('_sum') or c.startswith('n_') or '_n_' in c]
    sumcols = [c for c in sumcols if c not in {'n_nodes', 'n_edges', 'n_source_high_nodes'}]
    pairs = ag.groupby('molecular_pair_id', as_index=False)[sumcols].sum()
    parent_pairs = read('complete_470_focal_key_statistics.csv.gz')[['molecular_pair_id']].drop_duplicates()
    pairs = parent_pairs.merge(pairs, on='molecular_pair_id', how='left')
    pairs[sumcols] = pairs[sumcols].fillna(0)
    save(pairs, 'complete_global_exact_pair_statistics.csv.gz')
    macro = []
    for stratum in ['connected', 'source_feasible', 'source_feasible_target_endpoints_high', 'source_and_target_separately_feasible', 'union_feasible']:
        prefix = stratum + '__'
        denom = pairs[prefix + 'denominator_sum']
        active = denom > 0
        for policy in POLICIES:
            for metric in ['joint', 'union_gap', 'success20', 'source_failure20', 'avoidable_selection_failure20', 'target_internal_failure_despite_own_route20']:
                macro.append({'stratum': stratum, 'weighting_unit': 'global_exact_molecular_pair', 'policy': policy, 'metric': metric, 'value': float((pairs.loc[active, prefix + policy + '_' + metric + '_sum'] / denom[active]).mean()), 'n_active_units': int(active.sum()), 'n_all_units': len(pairs), 'endpoint_denominator': int(denom.sum()), 'missing_unit_rule': 'endpoint mean within exact pair, equal mean across active pairs; zero-denominator units undefined'})
    old = read('continuous_and_threshold_weighting_summary.csv')
    old = old[old.weighting_unit != 'global_exact_molecular_pair']
    save(pd.concat([old, pd.DataFrame(macro)], ignore_index=True), 'continuous_and_threshold_weighting_summary.csv')
    result = {'status': 'passed', 'n_heldouts': len(ag), 'source_route_status_counts': reg.source_route_status.value_counts().to_dict(), 'n_parent_focal_keys': len(read('complete_470_focal_key_statistics.csv.gz')), 'n_datasets_including_zeros': len(read('complete_20_dataset_statistics.csv')), 'n_reaction_classes_including_zeros': len(read('complete_18_class_statistics.csv'))}
    (OUT / 'metadata_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
