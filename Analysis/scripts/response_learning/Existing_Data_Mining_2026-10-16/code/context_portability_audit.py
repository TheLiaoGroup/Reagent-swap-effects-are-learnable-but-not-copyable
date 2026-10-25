"""Summarize cross-background portability of repeated strict component swaps.

This is a descriptive aggregation of the already generated condition-network
table.  It never treats substrate pairs as independent replicates and does not
fit a new outcome model."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'condition_network_meta.csv'
OUT = ROOT / 'results' / 'context_portability_summary.csv'

def _summarize(group: pd.DataFrame) -> pd.Series:
    n = len(group)
    consistency = group['pair_context_majority_consistency'].astype(float)
    mixing = group['pair_context_tail_mixing_rate20'].astype(float)
    balance = group['pair_mean_balanced_support20'].astype(float)
    jaccard = group['pair_mean_context_jaccard'].astype(float)
    n_context = group['pair_n_unique_backgrounds'].astype(float)
    n_union = group['pair_n_substrate_union'].astype(float)
    n_shared = group['pair_n_substrate_in_multiple_contexts'].astype(float)
    return pd.Series({'n_repeated_component_edges': n, 'n_contexts_total': int(n_context.sum()), 'median_contexts_per_edge': float(n_context.median()), 'mean_contexts_per_edge': float(n_context.mean()), 'median_majority_consistency': float(consistency.median()), 'mean_majority_consistency': float(consistency.mean()), 'fraction_consistency_ge_0_75': float((consistency >= 0.75).mean()), 'fraction_consistency_ge_0_90': float((consistency >= 0.9).mean()), 'median_tail_mixing_rate20': float(mixing.median()), 'mean_tail_mixing_rate20': float(mixing.mean()), 'fraction_tail_mixing_eq_1': float(np.isclose(mixing, 1.0).mean()), 'median_mean_balanced_support20': float(balance.median()), 'mean_mean_balanced_support20': float(balance.mean()), 'median_context_jaccard': float(jaccard.median()), 'median_substrate_union': float(n_union.median()), 'median_substrates_in_multiple_contexts': float(n_shared.median())})

def main() -> None:
    df = pd.read_csv(INPUT)
    if 'row_level' not in df or 'reaction_type' not in df:
        raise ValueError('condition_network_meta.csv is missing required columns')
    pair = df.loc[df['row_level'].eq('component_pair')].copy()
    repeated = pair.loc[pair['pair_n_unique_backgrounds'].ge(2)].copy()
    if repeated.empty:
        raise ValueError('no repeated component edges')
    rows = []
    rows.append({'reaction_type': 'ALL', 'swap_slot': 'ALL', **_summarize(repeated).to_dict()})
    for (reaction_type, swap_slot), g in repeated.groupby(['reaction_type', 'swap_slot'], sort=True):
        rows.append({'reaction_type': reaction_type, 'swap_slot': swap_slot, **_summarize(g).to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT} rows={len(out)} repeated_edges={len(repeated)}')
if __name__ == '__main__':
    main()
