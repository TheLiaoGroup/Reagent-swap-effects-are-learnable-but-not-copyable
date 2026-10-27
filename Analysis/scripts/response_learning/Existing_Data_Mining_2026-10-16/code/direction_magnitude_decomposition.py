"""Decompose cross-background component-edge effects into direction and magnitude.

The input is the existing strict identity-only condition-network table.  The
unit of cross-background comparison is a component edge within a reaction type
and changed slot.  Multiple strict swaps sharing the same background
fingerprint are pooled before edge-level summaries, so they are not counted as
independent backgrounds."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
META = BASE / 'results' / 'condition_network_meta.csv'
REGISTRY = BASE / 'results' / 'swap_registry.csv'
OUT_BACKGROUND = BASE / 'results' / 'direction_magnitude_background.csv'
OUT_EDGE = BASE / 'results' / 'direction_magnitude_edge.csv'
OUT_SUMMARY = BASE / 'results' / 'direction_magnitude_summary.csv'
OUT_AUDIT = BASE / 'audit' / 'direction_magnitude_metadata.json'
OUT_REPORT = BASE / 'reports' / 'DIRECTION_MAGNITUDE_DECOMPOSITION.md'
MAGNITUDE_HIGH_PP = 10.0
MAGNITUDE_LOW_PP = 5.0
DRIFT_HIGH_PP = 10.0
CONSISTENCY_HIGH = 0.75
CONSISTENCY_LOW = 0.5
EDGE_KEYS = ['reaction_type', 'swap_slot', 'component_pair']

def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors='coerce').to_numpy(float)
    w = pd.to_numeric(weights, errors='coerce').to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

def _pairwise_median(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    diffs = [abs(float(a) - float(b)) for a, b in combinations(values, 2)]
    return float(np.median(diffs)) if diffs else np.nan

def _background_direction(delta: float) -> str:
    if not np.isfinite(delta) or abs(delta) < 1e-09:
        return 'tie'
    return 'positive' if delta > 0 else 'negative'

def _direction_class(consistency: float, n_non_tie: int) -> str:
    if n_non_tie == 0 or not np.isfinite(consistency):
        return 'all_tied'
    if consistency >= CONSISTENCY_HIGH:
        return 'direction_preserved'
    if consistency <= CONSISTENCY_LOW:
        return 'direction_unstable'
    return 'direction_mixed'

def _magnitude_class(median_magnitude: float) -> str:
    if not np.isfinite(median_magnitude):
        return 'unquantified'
    if median_magnitude >= MAGNITUDE_HIGH_PP:
        return 'high_magnitude'
    if median_magnitude < MAGNITUDE_LOW_PP:
        return 'low_magnitude'
    return 'intermediate_magnitude'

def _state_label(direction_class: str, magnitude_class: str, drift_range: float, max_magnitude: float) -> str:
    if direction_class == 'all_tied':
        return 'all_tied'
    drift = np.isfinite(drift_range) and drift_range >= DRIFT_HIGH_PP
    if direction_class == 'direction_preserved' and drift:
        return 'direction_preserved_magnitude_drift'
    if direction_class == 'direction_preserved':
        return 'direction_preserved_magnitude_stable'
    if direction_class == 'direction_unstable' and (magnitude_class == 'high_magnitude' or (np.isfinite(max_magnitude) and max_magnitude >= MAGNITUDE_HIGH_PP)):
        return 'high_magnitude_direction_unstable'
    if direction_class == 'direction_unstable':
        return 'direction_unstable_low_or_moderate_magnitude'
    return 'direction_mixed'

def _collapse_backgrounds(g: pd.DataFrame) -> pd.DataFrame:
    """Pool duplicate rows with the same background fingerprint."""
    rows: list[dict[str, object]] = []
    for bg, b in g.groupby('background_fingerprint', sort=True, dropna=False):
        n_pairs = pd.to_numeric(b['n_pairs'], errors='coerce').fillna(0.0)
        pos = pd.to_numeric(b['n_pos_20_norm'], errors='coerce').fillna(0.0).sum()
        neg = pd.to_numeric(b['n_neg_20_norm'], errors='coerce').fillna(0.0).sum()
        total_pairs = float(n_pairs.sum())
        mean_delta = _weighted_mean(b['mean_delta_norm_pp'], n_pairs)
        mean_abs_pair_delta = _weighted_mean(b['mean_abs_delta_pp'], n_pairs)
        balanced = float(2.0 * min(pos, neg) / total_pairs) if total_pairs else np.nan
        direction = _background_direction(mean_delta)
        tail_direction = 'positive' if pos > neg else 'negative' if neg > pos else 'tie'
        rows.append({'reaction_type': b['reaction_type'].iloc[0], 'swap_slot': b['swap_slot'].iloc[0], 'component_pair': b['component_pair'].iloc[0], 'component_pair_orientation': b['component_pair_orientation'].iloc[0], 'background_fingerprint': bg, 'n_swap_rows_collapsed': int(len(b)), 'n_pairs': total_pairs, 'mean_delta_norm_pp': mean_delta, 'abs_mean_delta_norm_pp': abs(mean_delta) if np.isfinite(mean_delta) else np.nan, 'mean_abs_delta_pp': mean_abs_pair_delta, 'n_pos_20_norm': int(pos), 'n_neg_20_norm': int(neg), 'balanced_support_20': balanced, 'direction_sign': direction, 'tail_direction20': tail_direction, 'tail_mixed20': bool(pos > 0 and neg > 0), 'frac_either_zero': _weighted_mean(b['frac_either_zero'], n_pairs), 'n_check_false_pairs': int(pd.to_numeric(b['n_check_false_pairs'], errors='coerce').fillna(0).sum()), 'swap_ids': '||'.join(sorted(b['swap_id'].astype(str)))})
    return pd.DataFrame(rows)

def _summarize_edge(bg: pd.DataFrame) -> dict[str, object]:
    edge_key = {k: bg[k].iloc[0] for k in EDGE_KEYS}
    deltas = pd.to_numeric(bg['mean_delta_norm_pp'], errors='coerce').to_numpy(float)
    magnitudes = np.abs(deltas)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    directions = bg['direction_sign'].astype(str)
    pos = int((directions == 'positive').sum())
    neg = int((directions == 'negative').sum())
    ties = int((directions == 'tie').sum())
    non_tie = pos + neg
    consistency = float(max(pos, neg) / non_tie) if non_tie else np.nan
    tail = bg['tail_direction20'].astype(str)
    tail_pos = int((tail == 'positive').sum())
    tail_neg = int((tail == 'negative').sum())
    tail_tie = int((tail == 'tie').sum())
    tail_non_tie = tail_pos + tail_neg
    tail_consistency = float(max(tail_pos, tail_neg) / tail_non_tie) if tail_non_tie else np.nan
    median_mag = float(np.median(magnitudes)) if len(magnitudes) else np.nan
    mean_mag = float(np.mean(magnitudes)) if len(magnitudes) else np.nan
    signed_sd = float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else np.nan
    mag_sd = float(np.std(magnitudes, ddof=1)) if len(magnitudes) >= 2 else np.nan
    mag_range = float(np.max(magnitudes) - np.min(magnitudes)) if len(magnitudes) else np.nan
    q25, q75 = np.percentile(magnitudes, [25, 75]) if len(magnitudes) else (np.nan, np.nan)
    magnitude_class = _magnitude_class(median_mag)
    direction_class = _direction_class(consistency, non_tie)
    rec: dict[str, object] = {**edge_key, 'n_unique_backgrounds': int(len(bg)), 'n_swap_rows': int(bg['n_swap_rows_collapsed'].sum()), 'n_backgrounds_positive_sign': pos, 'n_backgrounds_negative_sign': neg, 'n_backgrounds_tied_sign': ties, 'n_backgrounds_non_tied_sign': non_tie, 'direction_sign_consistency': consistency, 'direction_fraction_positive': float(pos / non_tie) if non_tie else np.nan, 'direction_fraction_negative': float(neg / non_tie) if non_tie else np.nan, 'direction_class': direction_class, 'n_backgrounds_positive_tail20': tail_pos, 'n_backgrounds_negative_tail20': tail_neg, 'n_backgrounds_tied_tail20': tail_tie, 'tail_direction20_consistency': tail_consistency, 'n_backgrounds_tail_mixed20': int(bg['tail_mixed20'].sum()), 'fraction_backgrounds_tail_mixed20': float(bg['tail_mixed20'].mean()), 'median_abs_mean_delta_pp': median_mag, 'mean_abs_mean_delta_pp': mean_mag, 'min_abs_mean_delta_pp': float(np.min(magnitudes)) if len(magnitudes) else np.nan, 'max_abs_mean_delta_pp': float(np.max(magnitudes)) if len(magnitudes) else np.nan, 'signed_delta_mean_pp': float(np.mean(deltas)) if len(deltas) else np.nan, 'signed_delta_median_pp': float(np.median(deltas)) if len(deltas) else np.nan, 'signed_delta_sd_pp': signed_sd, 'signed_delta_range_pp': float(np.max(deltas) - np.min(deltas)) if len(deltas) else np.nan, 'magnitude_drift_sd_pp': mag_sd, 'magnitude_drift_range_pp': mag_range, 'magnitude_drift_iqr_pp': float(q75 - q25) if np.isfinite(q25) else np.nan, 'magnitude_pairwise_median_abs_diff_pp': _pairwise_median(magnitudes), 'magnitude_drift_range_over_median': float(mag_range / median_mag) if median_mag > 0 else np.nan, 'median_mean_abs_pair_delta_pp': float(np.nanmedian(bg['mean_abs_delta_pp'])), 'mean_mean_abs_pair_delta_pp': float(np.nanmean(bg['mean_abs_delta_pp'])), 'within_to_mean_abs_magnitude_ratio': float(np.nanmedian(bg['mean_abs_delta_pp']) / median_mag) if median_mag > 0 else np.nan, 'median_balanced_support20': float(np.nanmedian(bg['balanced_support_20'])), 'mean_balanced_support20': float(np.nanmean(bg['balanced_support_20'])), 'min_balanced_support20': float(np.nanmin(bg['balanced_support_20'])), 'max_balanced_support20': float(np.nanmax(bg['balanced_support_20'])), 'total_n_pairs': float(bg['n_pairs'].sum()), 'magnitude_class': magnitude_class}
    rec['high_magnitude_any_background'] = bool(np.isfinite(np.max(magnitudes)) and np.max(magnitudes) >= MAGNITUDE_HIGH_PP) if len(magnitudes) else False
    rec['state_label'] = _state_label(direction_class, magnitude_class, mag_range, rec['max_abs_mean_delta_pp'])
    return rec

def _summary_row(label: str, g: pd.DataFrame) -> dict[str, object]:

    def median(col: str) -> float:
        return float(pd.to_numeric(g[col], errors='coerce').median())

    def mean(col: str) -> float:
        return float(pd.to_numeric(g[col], errors='coerce').mean())
    n = len(g)
    return {'reaction_type': label, 'swap_slot': 'ALL' if label == 'ALL' else str(g['swap_slot'].iloc[0]), 'n_edges': n, 'n_unique_backgrounds_total': int(g['n_unique_backgrounds'].sum()), 'median_backgrounds_per_edge': median('n_unique_backgrounds'), 'mean_backgrounds_per_edge': mean('n_unique_backgrounds'), 'median_direction_consistency': median('direction_sign_consistency'), 'mean_direction_consistency': mean('direction_sign_consistency'), 'fraction_direction_consistency_ge_0_75': float((g['direction_sign_consistency'] >= CONSISTENCY_HIGH).mean()), 'fraction_direction_consistency_le_0_50': float((g['direction_sign_consistency'] <= CONSISTENCY_LOW).mean()), 'median_abs_mean_delta_pp': median('median_abs_mean_delta_pp'), 'mean_abs_mean_delta_pp': mean('mean_abs_mean_delta_pp'), 'median_mean_abs_pair_delta_pp': median('median_mean_abs_pair_delta_pp'), 'mean_mean_abs_pair_delta_pp': mean('mean_mean_abs_pair_delta_pp'), 'median_magnitude_drift_range_pp': median('magnitude_drift_range_pp'), 'mean_magnitude_drift_range_pp': mean('magnitude_drift_range_pp'), 'median_magnitude_drift_sd_pp': median('magnitude_drift_sd_pp'), 'median_magnitude_pairwise_diff_pp': median('magnitude_pairwise_median_abs_diff_pp'), 'median_balanced_support20': median('median_balanced_support20'), 'mean_balanced_support20': mean('mean_balanced_support20'), 'median_fraction_backgrounds_tail_mixed20': median('fraction_backgrounds_tail_mixed20'), 'mean_fraction_backgrounds_tail_mixed20': mean('fraction_backgrounds_tail_mixed20'), 'median_within_to_mean_abs_magnitude_ratio': median('within_to_mean_abs_magnitude_ratio'), 'mean_within_to_mean_abs_magnitude_ratio': mean('within_to_mean_abs_magnitude_ratio'), 'fraction_direction_preserved_magnitude_drift': float((g['state_label'] == 'direction_preserved_magnitude_drift').mean()), 'fraction_high_magnitude_direction_unstable': float((g['state_label'] == 'high_magnitude_direction_unstable').mean()), 'fraction_direction_preserved_magnitude_stable': float((g['state_label'] == 'direction_preserved_magnitude_stable').mean()), 'fraction_direction_unstable_low_or_moderate_magnitude': float((g['state_label'] == 'direction_unstable_low_or_moderate_magnitude').mean()), 'n_direction_preserved_magnitude_drift': int((g['state_label'] == 'direction_preserved_magnitude_drift').sum()), 'n_high_magnitude_direction_unstable': int((g['state_label'] == 'high_magnitude_direction_unstable').sum()), 'n_direction_preserved_magnitude_stable': int((g['state_label'] == 'direction_preserved_magnitude_stable').sum()), 'n_direction_unstable_low_or_moderate_magnitude': int((g['state_label'] == 'direction_unstable_low_or_moderate_magnitude').sum())}

def _fmt(x: object, digits: int=2) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return 'NA'
    return f'{float(x):.{digits}f}'

def main() -> None:
    meta = pd.read_csv(META)
    registry = pd.read_csv(REGISTRY)
    strict = registry.loc[registry['match_class'].eq('strict_identity_only'), ['swap_id', 'mean_abs_delta_pp']].copy()
    strict['swap_id'] = strict['swap_id'].astype(str)
    if strict['swap_id'].duplicated().any():
        raise ValueError('duplicate strict swap_id in registry')
    swap = meta.loc[meta['row_level'].eq('swap')].copy()
    swap['swap_id'] = swap['swap_id'].astype(str)
    swap = swap.merge(strict, on='swap_id', how='left', validate='one_to_one')
    if swap['mean_abs_delta_pp'].isna().any():
        raise ValueError('strict swap rows missing registry mean_abs_delta_pp')
    pair = meta.loc[meta['row_level'].eq('component_pair')].copy()
    repeated = pair.loc[pair['pair_n_unique_backgrounds'].ge(2), EDGE_KEYS].drop_duplicates()
    if repeated.empty:
        raise ValueError('no component edges repeated across at least two backgrounds')
    swap = swap.merge(repeated.assign(_repeated=True), on=EDGE_KEYS, how='inner', validate='many_to_one')
    if swap.empty:
        raise ValueError('repeated edge selection produced no swap rows')
    swap['n_pairs'] = pd.to_numeric(swap['n_pairs'], errors='coerce').fillna(0.0)
    for col in ['mean_delta_norm_pp', 'n_pos_20_norm', 'n_neg_20_norm', 'balanced_support_20', 'frac_either_zero', 'n_check_false_pairs']:
        swap[col] = pd.to_numeric(swap[col], errors='coerce')
    collapsed = []
    for _, group in swap.groupby(EDGE_KEYS, sort=True):
        collapsed.append(_collapse_backgrounds(group))
    bg = pd.concat(collapsed, ignore_index=True) if collapsed else pd.DataFrame()
    if bg.empty:
        raise ValueError('background collapse produced no rows')
    edge_rows = [_summarize_edge(g) for _, g in bg.groupby(EDGE_KEYS, sort=True)]
    edge = pd.DataFrame(edge_rows).sort_values(EDGE_KEYS).reset_index(drop=True)
    if (edge['n_unique_backgrounds'] < 2).any():
        raise AssertionError('edge output contains a single-background edge')
    summary_rows = [_summary_row('ALL', edge)]
    for (reaction_type, slot), g in edge.groupby(['reaction_type', 'swap_slot'], sort=True):
        rec = _summary_row(str(reaction_type), g)
        rec['reaction_type'] = reaction_type
        rec['swap_slot'] = slot
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    OUT_BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    OUT_BACKGROUND.write_text(bg.to_csv(index=False), encoding='utf-8-sig')
    OUT_EDGE.write_text(edge.to_csv(index=False), encoding='utf-8-sig')
    OUT_SUMMARY.write_text(summary.to_csv(index=False), encoding='utf-8-sig')
    audit = {'input_meta': str(META), 'input_registry': str(REGISTRY), 'strict_swap_rows_in_meta': int(meta['row_level'].eq('swap').sum()), 'strict_registry_rows': int(len(strict)), 'repeated_edges_at_least_two_unique_backgrounds': int(len(repeated)), 'retained_swap_rows_before_background_collapse': int(len(swap)), 'retained_unique_background_rows': int(len(bg)), 'retained_edge_rows': int(len(edge)), 'reaction_slot_strata': int(edge.groupby(['reaction_type', 'swap_slot']).ngroups), 'same_background_duplicate_rows_collapsed': int(len(swap) - len(bg)), 'thresholds': {'magnitude_high_pp': MAGNITUDE_HIGH_PP, 'magnitude_low_pp': MAGNITUDE_LOW_PP, 'drift_high_pp': DRIFT_HIGH_PP, 'direction_consistency_high': CONSISTENCY_HIGH, 'direction_consistency_low': CONSISTENCY_LOW}, 'state_counts': edge['state_label'].value_counts().to_dict(), 'edge_unique_key_assertion': bool(not edge.duplicated(EDGE_KEYS).any())}
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    all_row = summary.iloc[0]
    top_drift = edge.sort_values(['magnitude_drift_range_pp', 'median_abs_mean_delta_pp'], ascending=False).head(12)
    top_unstable = edge.loc[edge['state_label'].eq('high_magnitude_direction_unstable')].sort_values('median_abs_mean_delta_pp', ascending=False).head(12)
    lines = ['# Strict component edges: direction versus magnitude', '', 'This report splits the cross-background behaviour of strict identity swap edges into two dimensions: whether a component edge keeps its direction in the canonical orientation, and whether the effect magnitude (absolute value of the within-background mean Δ) drifts. The analysis uses only the existing `condition_network_meta.csv` and `swap_registry.csv`; original data were not re-read or modified.', '', '## Definitions', '', f"Retained: {int(all_row['n_edges']):,} component edges; each edge appears in at least two unique `background_fingerprint`. Repeated strict swaps within one background are first merged with weights equal to the number of shared substrate pairs, then direction and magnitude are computed across backgrounds, so several condition records of one background are not counted as several backgrounds. The canonical direction runs from the lower to the higher component ID; `mean_delta_norm_pp` and direction signs are taken from the existing network table.", '', '- `direction_sign_consistency`: after removing zero ties of the within-background mean Δ, the fraction of backgrounds in the majority direction; 0.75 or above is labelled `direction_preserved`, 0.50 or below `direction_unstable`.', '- `median_abs_mean_delta_pp` / `mean_abs_mean_delta_pp`: absolute value of the mean Δ across backgrounds, in yield percentage points (pp).', '- `magnitude_drift_range_pp`: maximum minus minimum of the absolute mean Δ across backgrounds; 10 pp or more is labelled magnitude drift. This threshold is a descriptive partition, not a statistical significance boundary.', '- `median_balanced_support20`: median of `balanced_support_20` over backgrounds; it reflects how balanced the positive and negative 20 pp tails are within a background and is not a replication probability.', '', '## Full-corpus results', '', f"Median cross-background direction consistency: {_fmt(all_row['median_direction_consistency'])}, mean {_fmt(all_row['mean_direction_consistency'])}; summarising within the backgrounds of each edge and then taking the median across edges, the absolute mean effect is {_fmt(all_row['median_abs_mean_delta_pp'])} pp, and the within-background mean absolute Δ over substrate pairs is {_fmt(all_row['median_mean_abs_pair_delta_pp'])} pp. The median of the per-edge ratio is {_fmt(all_row['median_within_to_mean_abs_magnitude_ratio'])}; it is not the ratio of the two full-corpus medians above. The triangle inequality guarantees that this ratio is at least 1; it should be read as a descriptive measure of positive/negative cancellation and cannot serve as independent proof of distribution width or hidden regions. The median magnitude drift range is {_fmt(all_row['median_magnitude_drift_range_pp'])} pp, and the median within-background balanced support20 is {_fmt(all_row['median_balanced_support20'])}.", '', '| State | Edges |', '|---|---:|']
    for state, count in edge['state_label'].value_counts().items():
        lines.append(f'| {state} | {int(count)} |')
    lines += ['', '## Reaction type × changed slot', '', '| reaction type | slot | edges | median direction consistency | median abs mean Δ (pp) | median within/mean magnitude ratio | median magnitude drift range (pp) | median balanced support20 | preserved + drift | high magnitude + unstable |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, x in summary.iloc[1:].sort_values('n_edges', ascending=False).iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {int(x.n_edges)} | {_fmt(x.median_direction_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.median_within_to_mean_abs_magnitude_ratio)} | {_fmt(x.median_magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} | {int(x.n_direction_preserved_magnitude_drift)} | {int(x.n_high_magnitude_direction_unstable)} |')
    lines += ['', '## Direction preserved but magnitude drifted', '', 'These edges keep the same canonical direction in most backgrounds, but effect magnitude differs by at least 10 pp between backgrounds. They are historical candidates whose "direction carries over, magnitude needs recalibration", and should not be read directly as transferable fixed gains.', '', '| reaction | slot | component pair | backgrounds | direction consistency | median abs mean Δ (pp) | drift range (pp) | median balanced support20 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    preserved_drift = edge.loc[edge['state_label'].eq('direction_preserved_magnitude_drift')].sort_values(['magnitude_drift_range_pp', 'median_abs_mean_delta_pp'], ascending=False).head(25)
    for _, x in preserved_drift.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {_fmt(x.direction_sign_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} |')
    if preserved_drift.empty:
        lines.append('| none | | | | | | | |')
    lines += ['', '## High magnitude but direction unstable', '', 'These edges have an absolute mean effect of at least 10 pp in at least one background, but direction consistency of at most 0.50; they directly reveal background dependence and cannot be compressed into a single component ranking. Cross-background medians and means are still given separately in the CSV.', '', '| reaction | slot | component pair | backgrounds | direction consistency | median abs mean Δ (pp) | drift range (pp) | median balanced support20 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for _, x in top_unstable.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {_fmt(x.direction_sign_consistency)} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.median_balanced_support20)} |')
    if top_unstable.empty:
        lines.append('| none | | | | | | | |')
    lines += ['', '## Edges with the largest magnitude drift (audit entry point)', '', '| reaction | slot | component pair | backgrounds | direction class | median abs mean Δ (pp) | drift range (pp) | signed range (pp) |', '|---|---:|---|---:|---|---:|---:|---:|']
    for _, x in top_drift.iterrows():
        lines.append(f'| {x.reaction_type} | {x.swap_slot} | {x.component_pair} | {int(x.n_unique_backgrounds)} | {x.direction_class} | {_fmt(x.median_abs_mean_delta_pp)} | {_fmt(x.magnitude_drift_range_pp)} | {_fmt(x.signed_delta_range_pp)} |')
    lines += ['', '## Scope of interpretation', '', 'This is a descriptive decomposition of historical HTE condition swaps. Backgrounds may share substrate panels, reference conditions, plate/batch labels and measurement workflows; cross-background consistency of one component edge is not independent experimental replication. Preserved direction does not imply the same mechanism, and magnitude drift alone does not locate its source. All magnitudes are matched yield percentage points; zeros, missing measurements and product-identity restrictions follow the main pipeline definitions.', '', '## Reproduction', '', '```powershell', 'python code/direction_magnitude_decomposition.py', '```', '', 'Outputs: `results/direction_magnitude_background.csv`, `results/direction_magnitude_edge.csv`, `results/direction_magnitude_summary.csv`, and `audit/direction_magnitude_metadata.json`.']
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote background={OUT_BACKGROUND} rows={len(bg)} edge={len(edge)} summary={len(summary)}')
if __name__ == '__main__':
    main()
