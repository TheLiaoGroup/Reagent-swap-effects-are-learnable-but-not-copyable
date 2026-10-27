"""Separate background portability from substrate-panel overlap.

The unit of the detailed table is an unordered pair of distinct background
contexts for one strict component edge.  A substrate pair reused across two
contexts is not an independent replication; all summaries therefore report
both pairwise context counts and edge-balanced averages."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
META = BASE / 'results' / 'condition_network_meta.csv'
BG = BASE / 'results' / 'direction_magnitude_background.csv'
EDGE = BASE / 'results' / 'direction_magnitude_edge.csv'
SUMMARY = BASE / 'results' / 'direction_magnitude_summary.csv'
PORT = BASE / 'results' / 'context_portability_summary.csv'
DELTAS = BASE / 'results' / 'swap_deltas.parquet'
OUT_PAIRS = BASE / 'results' / 'background_portability_overlap.csv'
OUT_SUMMARY = BASE / 'results' / 'background_portability_overlap_summary.csv'
OUT_AUDIT = BASE / 'audit' / 'background_portability_overlap_metadata.json'
OUT_REPORT = BASE / 'reports' / 'BACKGROUND_PORTABILITY_OVERLAP.md'
EDGE_KEYS = ['reaction_type', 'swap_slot', 'component_pair']
JAC_EDGES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.000000001]
JAC_LABELS = ['J<0.25', 'J0.25-0.50', 'J0.50-0.75', 'J0.75-0.90', 'J>=0.90']
OVERLAP_EDGES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.000000001]
OVERLAP_LABELS = ['Omin<0.25', 'Omin0.25-0.50', 'Omin0.50-0.75', 'Omin0.75-0.90', 'Omin>=0.90']

def _safe_float(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def _direction_agreement(a: object, b: object) -> float:
    a, b = (str(a), str(b))
    if a == 'tie' or b == 'tie' or a == 'nan' or (b == 'nan'):
        return np.nan
    return float(a == b)

def _weighted_mean(g: pd.DataFrame, value: str, weight: str='n_pairs') -> float:
    v = pd.to_numeric(g[value], errors='coerce').to_numpy(float)
    w = pd.to_numeric(g[weight], errors='coerce').to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

def _summarize_bin(g: pd.DataFrame, row_level: str, stratum: str, label: str) -> dict[str, object]:
    """Summarize a fixed overlap bin, retaining pairwise and edge counts."""
    eligible = g.loc[g['direction_agreement'].notna()].copy()
    tail = g.loc[g['tail_agreement'].notna()].copy()
    edge_agree = eligible.groupby(EDGE_KEYS, sort=False)['direction_agreement'].mean() if not eligible.empty else pd.Series(dtype=float)
    edge_tail = tail.groupby(EDGE_KEYS, sort=False)['tail_agreement'].mean() if not tail.empty else pd.Series(dtype=float)
    return {'row_level': row_level, 'stratum_type': stratum, 'stratum': label, 'n_context_pairs': int(len(g)), 'n_direction_pairs': int(len(eligible)), 'n_tail_pairs': int(len(tail)), 'n_edges': int(len(edge_agree)), 'n_edges_tail': int(len(edge_tail)), 'pairwise_direction_agreement': float(eligible['direction_agreement'].mean()) if not eligible.empty else np.nan, 'edge_balanced_direction_agreement': float(edge_agree.mean()) if not edge_agree.empty else np.nan, 'pairwise_tail_agreement': float(tail['tail_agreement'].mean()) if not tail.empty else np.nan, 'edge_balanced_tail_agreement': float(edge_tail.mean()) if not edge_tail.empty else np.nan, 'pairwise_median_abs_delta_difference_pp': float(g['abs_mean_delta_difference_pp'].median()) if len(g) else np.nan, 'pairwise_mean_abs_delta_difference_pp': float(g['abs_mean_delta_difference_pp'].mean()) if len(g) else np.nan, 'edge_balanced_median_abs_delta_difference_pp': float(g.groupby(EDGE_KEYS)['abs_mean_delta_difference_pp'].median().median()) if len(g) else np.nan, 'median_jaccard': float(g['substrate_jaccard'].median()) if len(g) else np.nan, 'median_overlap_min_fraction': float(g['overlap_min_fraction'].median()) if len(g) else np.nan, 'median_shared_substrate_pairs': float(g['n_shared_substrate_pairs'].median()) if len(g) else np.nan, 'reaction_type': 'ALL', 'swap_slot': 'ALL'}

def main() -> None:
    meta = pd.read_csv(META)
    if 'row_level' not in meta or 'swap_id' not in meta:
        raise ValueError('condition_network_meta.csv missing row_level or swap_id')
    swap = meta.loc[meta['row_level'].eq('swap')].copy()
    swap['swap_id'] = swap['swap_id'].astype(str)
    strict_ids = set(swap['swap_id'])
    bg = pd.read_csv(BG)
    edge = pd.read_csv(EDGE)
    if not set(EDGE_KEYS).issubset(bg.columns) or not set(EDGE_KEYS).issubset(edge.columns):
        raise ValueError('direction magnitude inputs missing edge keys')
    repeated_edges = edge.loc[edge['n_unique_backgrounds'].ge(2), EDGE_KEYS].drop_duplicates()
    bg = bg.merge(repeated_edges.assign(_repeated=True), on=EDGE_KEYS, how='inner', validate='many_to_one')
    if bg.empty:
        raise ValueError('no repeated-background edges')
    deltas = pd.read_parquet(DELTAS)
    deltas['swap_id'] = deltas['swap_id'].astype(str)
    deltas = deltas.loc[deltas['swap_id'].isin(strict_ids)].copy()
    sets = deltas.groupby('swap_id')['substrate_pair_key'].agg(lambda s: set(s.astype(str))).to_dict()
    if not sets:
        raise ValueError('no substrate sets for strict swaps')
    context_records: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for _, row in bg.iterrows():
        ids = [x for x in str(row['swap_ids']).split('||') if x and x != 'nan']
        missing = [x for x in ids if x not in sets]
        missing_ids.extend(missing)
        panel = set().union(*(sets.get(x, set()) for x in ids))
        context_records.append({**{k: row[k] for k in EDGE_KEYS}, 'background_fingerprint': row['background_fingerprint'], 'swap_ids': row['swap_ids'], 'n_swap_rows_collapsed': int(row['n_swap_rows_collapsed']), 'n_pairs': _safe_float(row['n_pairs']), 'n_substrate_pairs': int(len(panel)), 'substrate_set': panel, 'mean_delta_norm_pp': _safe_float(row['mean_delta_norm_pp']), 'abs_mean_delta_norm_pp': _safe_float(row['abs_mean_delta_norm_pp']), 'mean_abs_delta_pp': _safe_float(row['mean_abs_delta_pp']), 'direction_sign': row['direction_sign'], 'tail_direction20': row['tail_direction20'], 'balanced_support_20': _safe_float(row['balanced_support_20']), 'tail_mixed20': bool(row['tail_mixed20']), 'frac_either_zero': _safe_float(row['frac_either_zero']), 'n_check_false_pairs': _safe_float(row['n_check_false_pairs'])})
    if missing_ids:
        raise ValueError(f'missing substrate sets for {len(missing_ids)} swap IDs')
    context = pd.DataFrame(context_records)
    pair_rows: list[dict[str, object]] = []
    for key, g in context.groupby(EDGE_KEYS, sort=True):
        if g['background_fingerprint'].duplicated().any():
            raise AssertionError(f'duplicate background context in {key}')
        recs = list(g.to_dict('records'))
        for a, b in combinations(recs, 2):
            sa, sb = (a['substrate_set'], b['substrate_set'])
            shared = len(sa & sb)
            union = len(sa | sb)
            jac = float(shared / union) if union else np.nan
            min_size = min(len(sa), len(sb))
            min_overlap = float(shared / min_size) if min_size else np.nan
            mean_overlap = float(2 * shared / (len(sa) + len(sb))) if len(sa) + len(sb) else np.nan
            pair_rows.append({**{k: a[k] for k in EDGE_KEYS}, 'background_a': a['background_fingerprint'], 'background_b': b['background_fingerprint'], 'swap_ids_a': a['swap_ids'], 'swap_ids_b': b['swap_ids'], 'n_pairs_a': a['n_pairs'], 'n_pairs_b': b['n_pairs'], 'n_substrate_pairs_a': len(sa), 'n_substrate_pairs_b': len(sb), 'n_swap_rows_collapsed_a': a['n_swap_rows_collapsed'], 'n_swap_rows_collapsed_b': b['n_swap_rows_collapsed'], 'panel_repeat_factor_a': float(a['n_pairs'] / len(sa)) if len(sa) else np.nan, 'panel_repeat_factor_b': float(b['n_pairs'] / len(sb)) if len(sb) else np.nan, 'panel_duplicate_context_a': bool(a['n_swap_rows_collapsed'] > 1), 'panel_duplicate_context_b': bool(b['n_swap_rows_collapsed'] > 1), 'n_shared_substrate_pairs': shared, 'n_union_substrate_pairs': union, 'substrate_jaccard': jac, 'overlap_min_fraction': min_overlap, 'overlap_mean_fraction': mean_overlap, 'mean_delta_norm_pp_a': a['mean_delta_norm_pp'], 'mean_delta_norm_pp_b': b['mean_delta_norm_pp'], 'abs_mean_delta_norm_pp_a': a['abs_mean_delta_norm_pp'], 'abs_mean_delta_norm_pp_b': b['abs_mean_delta_norm_pp'], 'abs_mean_delta_difference_pp': abs(a['abs_mean_delta_norm_pp'] - b['abs_mean_delta_norm_pp']), 'signed_mean_delta_difference_pp': a['mean_delta_norm_pp'] - b['mean_delta_norm_pp'], 'mean_abs_delta_pp_a': a['mean_abs_delta_pp'], 'mean_abs_delta_pp_b': b['mean_abs_delta_pp'], 'direction_a': a['direction_sign'], 'direction_b': b['direction_sign'], 'direction_agreement': _direction_agreement(a['direction_sign'], b['direction_sign']), 'tail_direction_a': a['tail_direction20'], 'tail_direction_b': b['tail_direction20'], 'tail_agreement': _direction_agreement(a['tail_direction20'], b['tail_direction20']), 'balanced_support20_a': a['balanced_support_20'], 'balanced_support20_b': b['balanced_support_20'], 'balanced_support20_mean': np.nanmean([a['balanced_support_20'], b['balanced_support_20']]), 'tail_mixed20_a': a['tail_mixed20'], 'tail_mixed20_b': b['tail_mixed20'], 'frac_either_zero_a': a['frac_either_zero'], 'frac_either_zero_b': b['frac_either_zero'], 'n_check_false_pairs_a': a['n_check_false_pairs'], 'n_check_false_pairs_b': b['n_check_false_pairs'], 'n_unique_backgrounds_edge': len(recs)})
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise ValueError('no context pairs generated')
    pairs['jaccard_stratum'] = pd.cut(pairs['substrate_jaccard'], JAC_EDGES, labels=JAC_LABELS, include_lowest=True, right=False).astype(str)
    pairs['overlap_min_stratum'] = pd.cut(pairs['overlap_min_fraction'], OVERLAP_EDGES, labels=OVERLAP_LABELS, include_lowest=True, right=False).astype(str)
    carry_cols = [*EDGE_KEYS, 'direction_sign_consistency', 'direction_class', 'median_abs_mean_delta_pp', 'magnitude_drift_range_pp', 'state_label', 'median_balanced_support20']
    pairs = pairs.merge(edge[carry_cols], on=EDGE_KEYS, how='left', validate='many_to_one')
    association_rows = []
    for overlap_col, overlap_label in [('substrate_jaccard', 'jaccard'), ('overlap_min_fraction', 'overlap_min')]:
        for outcome_col, outcome_label in [('direction_agreement', 'direction'), ('tail_agreement', 'tail')]:
            q = pairs[[overlap_col, outcome_col]].dropna()
            association_rows.append({'level': 'context_pair', 'overlap_metric': overlap_label, 'outcome_metric': outcome_label, 'n': int(len(q)), 'spearman': float(q.corr(method='spearman').iloc[0, 1]) if len(q) >= 3 else np.nan})
            edge_mean = pairs.dropna(subset=[overlap_col, outcome_col]).groupby(EDGE_KEYS, sort=False).agg(overlap=(overlap_col, 'mean'), outcome=(outcome_col, 'mean'))
            association_rows.append({'level': 'edge_balanced', 'overlap_metric': overlap_label, 'outcome_metric': outcome_label, 'n': int(len(edge_mean)), 'spearman': float(edge_mean.corr(method='spearman').iloc[0, 1]) if len(edge_mean) >= 3 else np.nan})
    association = pd.DataFrame(association_rows)
    summary_rows: list[dict[str, object]] = []
    for stratum, col in [('jaccard', 'jaccard_stratum'), ('overlap_min', 'overlap_min_stratum')]:
        for label in JAC_LABELS if stratum == 'jaccard' else OVERLAP_LABELS:
            g = pairs.loc[pairs[col].eq(label)]
            if g.empty:
                continue
            summary_rows.append(_summarize_bin(g, 'overall', stratum, label))
            for (reaction_type, slot), gg in g.groupby(EDGE_KEYS[:2], sort=True):
                rec = _summarize_bin(gg, 'reaction_slot', stratum, label)
                rec['reaction_type'], rec['swap_slot'] = (reaction_type, slot)
                summary_rows.append(rec)
            rec = _summarize_bin(g, 'edge_balanced', stratum, label)
            rec['reaction_type'], rec['swap_slot'] = ('ALL', 'ALL')
            summary_rows.append(rec)
            for (reaction_type, slot), gg in g.groupby(EDGE_KEYS[:2], sort=True):
                rec = _summarize_bin(gg, 'edge_balanced_reaction_slot', stratum, label)
                rec['reaction_type'], rec['swap_slot'] = (reaction_type, slot)
                summary_rows.append(rec)
    contrast_rows = []
    for key, g in pairs.groupby(EDGE_KEYS, sort=True):
        low = g.loc[g['substrate_jaccard'] < 0.75, 'direction_agreement'].dropna()
        high = g.loc[g['substrate_jaccard'] >= 0.75, 'direction_agreement'].dropna()
        if len(low) and len(high):
            contrast_rows.append({'row_level': 'within_edge_jaccard_contrast', **{k: key[i] for i, k in enumerate(EDGE_KEYS)}, 'n_low_jaccard_pairs': int(len(low)), 'n_high_jaccard_pairs': int(len(high)), 'low_jaccard_agreement': float(low.mean()), 'high_jaccard_agreement': float(high.mean()), 'high_minus_low_agreement': float(high.mean() - low.mean())})
    contrast = pd.DataFrame(contrast_rows)
    pairs.to_csv(OUT_PAIRS, index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    contrast_path = BASE / 'results' / 'background_portability_overlap_within_edge.csv'
    contrast.to_csv(contrast_path, index=False, encoding='utf-8-sig')
    port = pd.read_csv(PORT)
    dm_summary = pd.read_csv(SUMMARY)
    audit = {'input_meta': str(META), 'input_background_rows': int(len(bg)), 'input_repeated_edges': int(len(repeated_edges)), 'strict_swap_ids': int(len(strict_ids)), 'strict_swap_ids_with_panel_sets': int(len(sets)), 'context_pair_rows': int(len(pairs)), 'context_pair_edges': int(pairs.groupby(EDGE_KEYS).ngroups), 'within_edge_contrast_edges': int(len(contrast)), 'jaccard_bins': JAC_LABELS, 'overlap_min_bins': OVERLAP_LABELS, 'background_rows_reconciled_to_direction_magnitude': int(len(bg)), 'context_portability_rows_available': int(len(port)), 'direction_magnitude_summary_rows_available': int(len(dm_summary)), 'missing_swap_ids': sorted(set(missing_ids)), 'association_checks': association.to_dict('records'), 'collapsed_contexts_with_multiple_swap_rows': int((context['n_swap_rows_collapsed'] > 1).sum()), 'max_context_panel_repeat_factor': float(np.nanmax(context['n_pairs'] / context['n_substrate_pairs']))}
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

    def f(x: object, n: int=3) -> str:
        try:
            if not np.isfinite(float(x)):
                return 'NA'
            return f'{float(x):.{n}f}'
        except (TypeError, ValueError):
            return 'NA'
    overall = summary.loc[summary['row_level'].eq('overall')].copy()
    jac_overall = overall.loc[overall['stratum_type'].eq('jaccard')]
    overlap_overall = overall.loc[overall['stratum_type'].eq('overlap_min')]
    assoc = association.copy()
    lines = ['# Background portability versus substrate-panel overlap', '', 'This report splits cross-background consistency into two sources: whether a condition swap keeps its direction in different backgrounds, and whether those backgrounds in fact reuse similar substrate panels. The analysis uses only the existing strict identity-only network, the direction/magnitude background table and the corresponding `swap_deltas` substrate keys; no experiments were added and original results were not changed.', '', '## Unit of analysis and guarding against pseudo-replication', '', f'There are {len(repeated_edges)} component edges with at least two unique `background_fingerprint`, expanded into {len(pairs):,} pairwise background comparisons within the same component edge. Each background first merges identical fingerprints by `direction_magnitude_background.csv`; the substrate panel of each background is the union of substrate pairs over all its swap IDs. When a substrate pair recurs in several backgrounds, `n_shared_substrate_pairs` is used only to compute panel overlap and is not counted as independent replication.', f"In total, {int((context['n_swap_rows_collapsed'] > 1).sum())} background fingerprints are merged from several swap rows; the maximum panel duplication factor is {f(np.nanmax(context['n_pairs'] / context['n_substrate_pairs']))}; these duplicates are kept as a single background context and are flagged in the CSV by `n_swap_rows_collapsed` and `panel_duplicate_context_*`.", '', 'Main metrics:', '- `substrate_jaccard = |A∩B|/|A∪B|`; `overlap_min_fraction = |A∩B|/min(|A|,|B|)` is also reported, so that one large panel alone does not dilute Jaccard.', '- `direction_agreement` compares the signs of the canonical background mean Δ of the two backgrounds; ties are excluded from the denominator. `tail_agreement` compares the majority directions of the +20/-20 pp tails.', '- Results are given with both context-pair weighting and edge-balanced weighting; the latter first averages within each component edge and then gives every edge equal weight.', '- Jaccard and minimum overlap fraction both use fixed bins, not quantiles tuned to the results; empty bins are not shown.', '', '## Full-corpus stratified results (one row per background pair)', '', '### Shared substrate Jaccard', '', '| Jaccard stratum | context pairs | edges | pairwise direction agreement | edge-balanced direction agreement | pairwise tail agreement | median abs-Δ difference (pp) | median shared pairs |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in jac_overall.iterrows():
        lines.append(f"| {x['stratum']} | {int(x['n_context_pairs'])} | {int(x['n_edges'])} | {f(x['pairwise_direction_agreement'])} | {f(x['edge_balanced_direction_agreement'])} | {f(x['pairwise_tail_agreement'])} | {f(x['pairwise_median_abs_delta_difference_pp'], 2)} | {f(x['median_shared_substrate_pairs'], 1)} |")
    lines += ['', '### Minimum panel-overlap fraction', '', '| overlap-min stratum | context pairs | edges | pairwise direction agreement | edge-balanced direction agreement | pairwise tail agreement | median abs-Δ difference (pp) | median Jaccard |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in overlap_overall.iterrows():
        lines.append(f"| {x['stratum']} | {int(x['n_context_pairs'])} | {int(x['n_edges'])} | {f(x['pairwise_direction_agreement'])} | {f(x['edge_balanced_direction_agreement'])} | {f(x['pairwise_tail_agreement'])} | {f(x['pairwise_median_abs_delta_difference_pp'], 2)} | {f(x['median_jaccard'])} |")
    lines += ['', '## Within-edge portability test', '', f'There are {len(contrast)} component edges containing both background pairs with Jaccard < 0.75 and background pairs with Jaccard ≥ 0.75. `background_portability_overlap_within_edge.csv` gives, for each edge, the high-overlap minus low-overlap difference in direction agreement rate; a positive value means high-overlap background pairs agree more, a negative value means low-overlap background pairs agree more. This comparison is still a historical description, and background pairs within each edge share contexts, so it cannot be tested as independent samples.', '', '| quantity | value |', '|---|---:|', f'| within-edge contrasts | {len(contrast)} |', f"| median (high-low agreement) | {f(contrast['high_minus_low_agreement'].median() if len(contrast) else np.nan)} |", f"| mean (high-low agreement) | {f(contrast['high_minus_low_agreement'].mean() if len(contrast) else np.nan)} |", f"| fraction high > low | {f((contrast['high_minus_low_agreement'] > 0).mean() if len(contrast) else np.nan)} |", f"| fraction high < low | {f((contrast['high_minus_low_agreement'] < 0).mean() if len(contrast) else np.nan)} |", '', '## Overlap-association audit', '', 'The Spearman values only check the monotonic relation between panel overlap and consistency and are not significance tests on independent samples. `edge_balanced` first averages within component edges and then computes the correlation across edges.', '', '| level | overlap metric | outcome | n | Spearman |', '|---|---|---|---:|---:|']
    for _, x in assoc.iterrows():
        lines.append(f"| {x['level']} | {x['overlap_metric']} | {x['outcome_metric']} | {int(x['n'])} | {f(x['spearman'])} |")
    lines += ['', '## Reaction class and changed-slot detail', '', 'The full reaction_type × changed-slot stratification is also saved in `background_portability_overlap_summary.csv`. These rows locate which networks show evidence of keeping direction under low overlap; sparse strata should not be ranked as stable effects.', '', '## Interpretation', '', 'Direction agreement under high Jaccard only shows that the condition ranking of one component edge is fairly stable on similar substrate panels; it may come from repeated substrates, plate/batch collinearity or the same historical screening network. Direction agreement that persists under low Jaccard is closer to descriptive evidence of background portability, but low overlap here may still share substrate families, product-analysis workflows or other unrecorded factors. Conversely, direction disagreement under high overlap shows directly that "the same panel repeated" does not guarantee a fixed ranking either.', '', 'This analysis does not treat Jaccard-bin results as independent significance tests, does not treat context-pair counts as experimental replicate counts, and does not promote them to a hidden mechanistic regime. Its use is to give two dimensions for prioritising the next round of validation: whether direction is kept after a background change, and whether that judgement goes beyond reuse of substrate panels.', '', '## Reproduction', '', '```powershell', 'python code/background_portability_overlap.py', '```', '', 'Outputs: `results/background_portability_overlap.csv`, `results/background_portability_overlap_summary.csv`, `results/background_portability_overlap_within_edge.csv`.']
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {OUT_PAIRS} rows={len(pairs)}; summary={len(summary)}; within_edge={len(contrast)}')
if __name__ == '__main__':
    main()
