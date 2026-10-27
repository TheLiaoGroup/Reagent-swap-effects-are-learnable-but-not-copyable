"""Next-level transportability lattice and compositional bridge analysis.

This module reads only already-produced controlled bridge registry/score tables.
It keeps support, transfer direction, matching hierarchy, and aggregation level
separate.  Rows are retrospective conditional means; molecular units, screens,
and families are not treated as independent experimental replicates."""
from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
DATA = HERE / 'data'
REGISTRY_FILE = TABLES / 'controlled_bridge_directed_bridge_registry.csv'
SCORE_FILE = TABLES / 'controlled_bridge_scores.csv'
FRONTIER_FILE = TABLES / 'transport_support_frontier_summary.csv'
MODEL_COLUMNS = ['zero', 'source_pair_marginal', 'source_position_marginal', 'source_marginal_excluding_unit', 'source_conditioned', 'source_conditioned_shrunk5', 'source_all_available_marginal']
BASES = ['position_1', 'position_2', 'full_pair']
MODES = ['strict', 'dose_relaxed']
BACKGROUNDS = ['background_free', 'exact_background']
RELATIONS = ['within_family', 'cross_family']
THRESHOLDS = [1, 3, 5]
OVERLAPS = ['all', 'no_shared_partner']
CONFIG_KEYS = ['scheme', 'mode', 'background_control', 'basis', 'relation', 'min_partners_each', 'partner_overlap']
PAIR_KEYS = ['source_screen', 'target_screen', 'source_family', 'target_family']
NUMERIC_REGISTRY = ['observed', *MODEL_COLUMNS, 'source_yield_sd', 'source_n_partners', 'target_n_partners', 'shared_partner_count', 'n_source_screens', 'n_source_families']
EPS = 1e-12

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False

def classify_transfer_state(status: str, model_rmse: object, zero_rmse: object, tol: float=1e-12) -> str:
    """Classify a model relative to zero without turning missing support into failure."""
    status = str(status)
    if status != 'scored':
        return status
    if not (_finite(model_rmse) and _finite(zero_rmse)):
        return 'supported_unresolved_vs_zero'
    difference = float(model_rmse) - float(zero_rmse)
    if difference < -tol:
        return 'supported_positive_vs_zero'
    if difference > tol:
        return 'supported_negative_vs_zero'
    return 'supported_neutral_vs_zero'

def _decode_dose_levels(value: object) -> set[str] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, list):
        return None
    return {str(item).strip() for item in decoded}

def dose_overlap_class(source_levels: object, target_levels: object) -> str:
    """Return exact/partial/none/unknown overlap for serialized dose sets."""
    source = _decode_dose_levels(source_levels)
    target = _decode_dose_levels(target_levels)
    if not source or not target:
        return 'unknown'
    if source == target:
        return 'exact'
    if source.intersection(target):
        return 'partial'
    return 'none'

def filter_registry_support(frame: pd.DataFrame, threshold: int, overlap: str) -> pd.DataFrame:
    """Apply the declared partner support filter to registry rows."""
    data = frame.copy()
    source_n = pd.to_numeric(data['source_n_partners'], errors='coerce')
    target_n = pd.to_numeric(data['target_n_partners'], errors='coerce')
    mask = source_n.ge(int(threshold)) & target_n.ge(int(threshold))
    if overlap == 'no_shared_partner':
        mask &= pd.to_numeric(data['shared_partner_count'], errors='coerce').eq(0)
    elif overlap != 'all':
        raise ValueError(f'unknown partner overlap: {overlap}')
    return data.loc[mask].copy()

def paired_mse_delta(observed: pd.Series, left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    """Compare two predictions on exactly the same finite observed units."""
    obs = pd.to_numeric(observed, errors='coerce').to_numpy(dtype=float)
    left_values = pd.to_numeric(left, errors='coerce').to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(mask):
        return {'n_common': 0, 'mse_left': np.nan, 'mse_right': np.nan, 'mae_left_pp': np.nan, 'mae_right_pp': np.nan, 'delta_left_minus_right': np.nan, 'delta_mae_left_minus_right_pp': np.nan}
    left_error = left_values[mask] - obs[mask]
    right_error = right_values[mask] - obs[mask]
    mse_left = float(np.mean(left_error ** 2))
    mse_right = float(np.mean(right_error ** 2))
    return {'n_common': int(mask.sum()), 'mse_left': mse_left, 'mse_right': mse_right, 'mae_left_pp': float(100.0 * np.mean(np.abs(left_error))), 'mae_right_pp': float(100.0 * np.mean(np.abs(right_error))), 'delta_left_minus_right': mse_left - mse_right, 'delta_mae_left_minus_right_pp': float(100.0 * (np.mean(np.abs(left_error)) - np.mean(np.abs(right_error))))}

def _delta_state(delta: object, n_common: object) -> str:
    if not _finite(n_common) or int(n_common) <= 0 or (not _finite(delta)):
        return 'support_threshold_unavailable'
    if float(delta) > EPS:
        return 'positive_increment'
    if float(delta) < -EPS:
        return 'negative_increment'
    return 'neutral_increment'

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)
    scores = pd.read_csv(SCORE_FILE, keep_default_na=False)
    for column in NUMERIC_REGISTRY:
        if column in registry:
            registry[column] = pd.to_numeric(registry[column], errors='coerce')
    for column in ['min_partners_each', 'family_macro_rmse_pp', 'family_macro_mse', 'family_macro_mae_pp', 'n_prediction_units', 'n_target_screens', 'n_family_pairs']:
        if column in scores:
            scores[column] = pd.to_numeric(scores[column], errors='coerce')
    scores = scores.rename(columns={'family_macro_mse': 'mse', 'family_macro_rmse_pp': 'rmse_pp', 'family_macro_mae_pp': 'mae_pp'})
    hashes = {str(path.relative_to(HERE)): digest(path) for path in [REGISTRY_FILE, SCORE_FILE, FRONTIER_FILE] if path.exists()}
    return (registry, scores, hashes)

def applicable_settings() -> list[tuple[str, str, str, int, str]]:
    settings: list[tuple[str, str, str, int, str]] = []
    for mode in MODES:
        for background in BACKGROUNDS:
            for basis in BASES:
                thresholds = [1] if basis == 'full_pair' else THRESHOLDS
                overlaps = ['all'] if basis == 'full_pair' else OVERLAPS
                settings.extend(((mode, background, basis, threshold, overlap) for threshold in thresholds for overlap in overlaps))
    return settings

def _config(scheme: str, mode: str, background: str, basis: str, relation: str, threshold: int, overlap: str) -> dict[str, object]:
    return {'scheme': scheme, 'mode': mode, 'background_control': background, 'basis': basis, 'relation': relation, 'min_partners_each': threshold, 'partner_overlap': overlap}

def _registry_subset(registry: pd.DataFrame, mode: str, background: str, basis: str, relation: str, threshold: int, overlap: str) -> pd.DataFrame:
    base = registry[(registry['mode'] == mode) & (registry['background_control'] == background) & (registry['basis'] == basis) & (registry['relation'] == relation)]
    return filter_registry_support(base, threshold, overlap)

def _collapse_unit_predictions(frame: pd.DataFrame, model_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    """Average repeated perturbation/background records before scoring."""
    model_columns = list(dict.fromkeys(model_columns))
    columns = list(dict.fromkeys([*group_columns, 'observed', *model_columns]))
    if frame.empty or not set(columns).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    data = frame[columns].copy()
    for column in ['observed', *model_columns]:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=['observed'])
    if data.empty:
        return data
    return data.groupby(group_columns, as_index=False, sort=True)[['observed', *model_columns]].mean()

def _prediction_metrics(frame: pd.DataFrame, model: str, group_columns: list[str] | None=None) -> dict[str, object]:
    if frame.empty or model not in frame:
        return {'status': 'zero_eligible', 'n_directed_rows': 0, 'n_units': 0, 'mse': np.nan, 'rmse_pp': np.nan, 'mae_pp': np.nan, 'sign_agreement': np.nan, 'observed_mean': np.nan}
    group_columns = group_columns or ['unit_identity']
    collapsed = _collapse_unit_predictions(frame, [model], group_columns)
    observed = pd.to_numeric(collapsed['observed'], errors='coerce')
    predicted = pd.to_numeric(collapsed[model], errors='coerce')
    mask = np.isfinite(observed.to_numpy()) & np.isfinite(predicted.to_numpy())
    if not np.any(mask):
        return {'status': 'support_threshold_unavailable', 'n_directed_rows': int(len(frame)), 'n_units': 0, 'mse': np.nan, 'rmse_pp': np.nan, 'mae_pp': np.nan, 'sign_agreement': np.nan, 'observed_mean': np.nan}
    error = predicted.to_numpy(dtype=float)[mask] - observed.to_numpy(dtype=float)[mask]
    obs = observed.to_numpy(dtype=float)[mask]
    pred = predicted.to_numpy(dtype=float)[mask]
    mse = float(np.mean(error ** 2))
    return {'status': 'scored', 'n_directed_rows': int(len(frame)), 'n_units': int(mask.sum()), 'mse': mse, 'rmse_pp': float(100.0 * math.sqrt(mse)), 'mae_pp': float(100.0 * np.mean(np.abs(error))), 'sign_agreement': float(np.mean(np.sign(obs) == np.sign(pred))), 'observed_mean': float(np.mean(obs))}

def build_screen_pair_scores(registry: pd.DataFrame) -> pd.DataFrame:
    """Build the same-model, same-support screen-pair ledger used by B and D."""
    rows: list[dict[str, object]] = []
    for mode, background, basis, threshold, overlap in applicable_settings():
        for relation in RELATIONS:
            raw = registry[(registry['mode'] == mode) & (registry['background_control'] == background) & (registry['basis'] == basis) & (registry['relation'] == relation)]
            universe = raw[PAIR_KEYS].drop_duplicates().sort_values(PAIR_KEYS)
            selected = filter_registry_support(raw, threshold, overlap)
            for pair in universe.to_dict('records'):
                local = selected.copy()
                for key in PAIR_KEYS:
                    local = local[local[key].eq(pair[key])]
                for model in MODEL_COLUMNS:
                    metrics = _prediction_metrics(local, model, ['unit_identity'])
                    rows.append({**_config('directed_screen', mode, background, basis, relation, threshold, overlap), **pair, 'model': model, **metrics})
    columns = CONFIG_KEYS + PAIR_KEYS + ['model', 'status', 'n_directed_rows', 'n_units', 'mse', 'rmse_pp', 'mae_pp', 'sign_agreement', 'observed_mean']
    return pd.DataFrame(rows, columns=columns).sort_values(CONFIG_KEYS + PAIR_KEYS + ['model']).reset_index(drop=True)

def build_lattice_paths(scores: pd.DataFrame) -> pd.DataFrame:
    """Join zero baselines and label every declared lattice point."""
    pair_keys = [column for column in PAIR_KEYS if column in scores.columns]
    key = CONFIG_KEYS + pair_keys
    zero_columns = key + ['status', 'mse', 'rmse_pp']
    if 'n_prediction_units' in scores.columns:
        zero_columns.append('n_prediction_units')
    zero = scores[scores['model'].eq('zero')][zero_columns].rename(columns={'status': 'zero_status', 'mse': 'zero_mse', 'rmse_pp': 'zero_rmse_pp', 'n_prediction_units': 'zero_n_prediction_units'})
    result = scores.merge(zero, on=key, how='left', validate='many_to_one')
    result['transfer_state'] = ['baseline_zero' if model == 'zero' else classify_transfer_state(status, rmse, zero_rmse) for model, status, rmse, zero_rmse in zip(result['model'], result['status'], result['rmse_pp'], result['zero_rmse_pp'])]
    result['rmse_delta_vs_zero_pp'] = result['rmse_pp'] - result['zero_rmse_pp']
    result['mse_delta_vs_zero'] = result['mse'] - result['zero_mse']
    result['support_state'] = result['status']
    sort_keys = CONFIG_KEYS + [column for column in PAIR_KEYS if column in result.columns] + ['model']
    return result.sort_values(sort_keys).reset_index(drop=True)

def build_lattice_monotonicity(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize support and transfer paths across nested partner thresholds."""
    rows: list[dict[str, object]] = []
    grouping = ['scheme', 'mode', 'background_control', 'basis', 'relation', 'partner_overlap', 'model'] + PAIR_KEYS
    for keys, part in paths[paths['model'].ne('zero')].groupby(grouping, sort=True, dropna=False):
        values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        part = part.sort_values('min_partners_each')
        scored_flags = part['status'].eq('scored').to_numpy(dtype=bool)
        if 'n_prediction_units' in part.columns:
            units = pd.to_numeric(part['n_prediction_units'], errors='coerce').to_numpy(dtype=float)
        else:
            units = pd.to_numeric(part['n_units'], errors='coerce').to_numpy(dtype=float)
        support_monotone = bool(all((not scored_flags[i + 1] or scored_flags[i] for i in range(len(scored_flags) - 1))))
        unit_monotone = bool(all((not np.isfinite(units[i + 1]) or not np.isfinite(units[i]) or units[i + 1] <= units[i] for i in range(len(units) - 1))))
        states = [str(value) for value in part['transfer_state']]
        signs = [state for state in states if state in {'supported_positive_vs_zero', 'supported_negative_vs_zero', 'supported_neutral_vs_zero'}]
        first_failure = np.nan
        for threshold, state in zip(part['min_partners_each'], states):
            if state in {'supported_negative_vs_zero', 'zero_eligible', 'support_threshold_unavailable', 'supported_unresolved_vs_zero'}:
                first_failure = int(threshold)
                break
        rows.append({**values, 'n_thresholds': int(len(part)), 'threshold_path': '|'.join((str(int(value)) for value in part['min_partners_each'])), 'status_path': '|'.join((str(value) for value in part['status'])), 'transfer_state_path': '|'.join(states), 'support_monotone': support_monotone, 'unit_support_monotone': unit_monotone, 'n_positive_states': int(sum((state == 'supported_positive_vs_zero' for state in states))), 'n_negative_states': int(sum((state == 'supported_negative_vs_zero' for state in states))), 'n_zero_eligible_states': int(sum((state == 'zero_eligible' for state in states))), 'sign_persistent': bool(len(signs) > 0 and len(set(signs)) == 1), 'first_failure_threshold': first_failure})
    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)

def _composition_rows_for_pair(local: pd.DataFrame, config: dict[str, object], pair: dict[str, object]) -> list[dict[str, object]]:
    definitions = [('pair_given_position', 'source_position_marginal', 'source_pair_marginal'), ('condition_given_pair', 'source_pair_marginal', 'source_conditioned'), ('source_given_zero', 'zero', 'source_conditioned')]
    rows: list[dict[str, object]] = []
    collapsed = _collapse_unit_predictions(local, [left for _, left, _ in definitions] + [right for _, _, right in definitions], ['unit_identity'])
    for delta_name, left_model, right_model in definitions:
        metrics = paired_mse_delta(collapsed['observed'], collapsed[left_model], collapsed[right_model])
        rows.append({**config, **pair, 'aggregation_level': 'screen_pair', 'delta_name': delta_name, 'model_left': left_model, 'model_right': right_model, **metrics, 'delta_state': _delta_state(metrics['delta_left_minus_right'], metrics['n_common']), 'n_source_rows': int(len(local)), 'n_unit_rows': int(len(collapsed))})
    return rows

def build_composition_deltas(registry: pd.DataFrame) -> pd.DataFrame:
    """Compare marginal, full-pair, conditioned, and zero predictions on common units."""
    rows: list[dict[str, object]] = []
    for mode, background, basis, threshold, overlap in applicable_settings():
        for relation in RELATIONS:
            selected = _registry_subset(registry, mode, background, basis, relation, threshold, overlap)
            raw = registry[(registry['mode'] == mode) & (registry['background_control'] == background) & (registry['basis'] == basis) & (registry['relation'] == relation)]
            universe = raw[PAIR_KEYS].drop_duplicates().sort_values(PAIR_KEYS)
            config = _config('directed_screen', mode, background, basis, relation, threshold, overlap)
            for pair in universe.to_dict('records'):
                local = selected.copy()
                for key in PAIR_KEYS:
                    local = local[local[key].eq(pair[key])]
                if local.empty:
                    for delta_name, left_model, right_model in [('pair_given_position', 'source_position_marginal', 'source_pair_marginal'), ('condition_given_pair', 'source_pair_marginal', 'source_conditioned'), ('source_given_zero', 'zero', 'source_conditioned')]:
                        rows.append({**config, **pair, 'aggregation_level': 'screen_pair', 'delta_name': delta_name, 'model_left': left_model, 'model_right': right_model, 'n_common': 0, 'mse_left': np.nan, 'mse_right': np.nan, 'mae_left_pp': np.nan, 'mae_right_pp': np.nan, 'delta_left_minus_right': np.nan, 'delta_mae_left_minus_right_pp': np.nan, 'delta_state': 'zero_eligible', 'n_source_rows': 0, 'n_unit_rows': 0})
                else:
                    rows.extend(_composition_rows_for_pair(local, config, pair))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    group = CONFIG_KEYS + ['source_family', 'target_family', 'delta_name', 'model_left', 'model_right']
    scored = result[result['n_common'].gt(0)].copy()
    family_rows: list[dict[str, object]] = []
    for keys, part in scored.groupby(group, sort=True, dropna=False):
        values = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        family_rows.append({**values, 'aggregation_level': 'family_pair', 'n_common': int(part['n_common'].sum()), 'n_screen_pairs': int(len(part)), 'mse_left': float(part['mse_left'].mean()), 'mse_right': float(part['mse_right'].mean()), 'mae_left_pp': float(part['mae_left_pp'].mean()), 'mae_right_pp': float(part['mae_right_pp'].mean()), 'delta_left_minus_right': float(part['delta_left_minus_right'].mean()), 'delta_mae_left_minus_right_pp': float(part['delta_mae_left_minus_right_pp'].mean()), 'delta_state': _delta_state(part['delta_left_minus_right'].mean(), part['n_common'].sum()), 'n_source_rows': int(part['n_source_rows'].sum()), 'n_unit_rows': int(part['n_unit_rows'].sum())})
    if family_rows:
        result = pd.concat([result, pd.DataFrame(family_rows)], ignore_index=True, sort=False)
    return result.sort_values(CONFIG_KEYS + ['aggregation_level', 'source_family', 'target_family', 'delta_name']).reset_index(drop=True)

def _partner_bin(value: object) -> str:
    if not _finite(value):
        return 'unknown'
    value = int(value)
    if value <= 1:
        return '1'
    if value == 2:
        return '2'
    if value <= 4:
        return '3-4'
    return '5+'

def _shared_bin(value: object) -> str:
    if not _finite(value):
        return 'unknown'
    value = int(value)
    return '0' if value == 0 else '1' if value == 1 else '2+'

def build_structure_boundary(registry: pd.DataFrame) -> pd.DataFrame:
    """Describe model error across support redundancy and dose-overlap strata."""
    data = registry.copy()
    data['partner_redundancy'] = np.minimum(data['source_n_partners'], data['target_n_partners'])
    data['partner_redundancy_bin'] = data['partner_redundancy'].map(_partner_bin)
    data['shared_partner_bin'] = data['shared_partner_count'].map(_shared_bin)
    data['dose_overlap_class'] = [dose_overlap_class(a, b) for a, b in zip(data['source_dose_levels_json'], data['target_dose_levels_json'])]
    rows: list[dict[str, object]] = []
    stratum_keys = CONFIG_KEYS + ['partner_redundancy_bin', 'shared_partner_bin', 'dose_overlap_class', 'model']
    for mode, background, basis, threshold, overlap in applicable_settings():
        for relation in RELATIONS:
            selected = _registry_subset(data, mode, background, basis, relation, threshold, overlap)
            if selected.empty:
                continue
            selected = selected.copy()
            selected['scheme'] = 'directed_screen'
            selected['relation'] = relation
            selected['min_partners_each'] = threshold
            selected['partner_overlap'] = overlap
            for keys, part in selected.groupby(stratum_keys[:-1], sort=True, dropna=False):
                values = dict(zip(stratum_keys[:-1], keys if isinstance(keys, tuple) else (keys,)))
                for model in MODEL_COLUMNS:
                    metrics = _prediction_metrics(part, model, PAIR_KEYS + ['unit_identity'])
                    rows.append({**values, 'model': model, **metrics, 'n_screen_pairs': int(part[PAIR_KEYS].drop_duplicates().shape[0]), 'n_target_screens': int(part['target_screen'].nunique())})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    key = CONFIG_KEYS + ['partner_redundancy_bin', 'shared_partner_bin', 'dose_overlap_class']
    zero = result[result['model'].eq('zero')][key + ['status', 'mse', 'rmse_pp']].rename(columns={'status': 'zero_status', 'mse': 'zero_mse', 'rmse_pp': 'zero_rmse_pp'})
    result = result.merge(zero, on=key, how='left', validate='many_to_one')
    result['transfer_state'] = ['baseline_zero' if model == 'zero' else classify_transfer_state(status, rmse, zero_rmse) for model, status, rmse, zero_rmse in zip(result['model'], result['status'], result['rmse_pp'], result['zero_rmse_pp'])]
    result['rmse_delta_vs_zero_pp'] = result['rmse_pp'] - result['zero_rmse_pp']
    return result.sort_values(key + ['model']).reset_index(drop=True)

def _aggregate_level(screen_scores: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == 'screen_pair':
        result = screen_scores.copy()
        result['aggregation_level'] = level
        return result
    if level == 'family_pair':
        grouping = CONFIG_KEYS + ['source_family', 'target_family', 'model']
    elif level == 'target_screen':
        grouping = CONFIG_KEYS + ['target_screen', 'model']
    else:
        raise ValueError(level)
    rows: list[dict[str, object]] = []
    for keys, part in screen_scores.groupby(grouping, sort=True, dropna=False):
        values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        scored = part[part['status'].eq('scored')]
        if scored.empty:
            metrics = {'status': 'zero_eligible', 'n_units': 0, 'mse': np.nan, 'rmse_pp': np.nan, 'mae_pp': np.nan, 'n_groups': 0}
        else:
            mse = float(scored['mse'].mean())
            metrics = {'status': 'scored', 'n_units': int(scored['n_units'].sum()), 'mse': mse, 'rmse_pp': float(100.0 * math.sqrt(mse)), 'mae_pp': float(scored['mae_pp'].mean()), 'n_groups': int(len(scored))}
        rows.append({**values, 'aggregation_level': level, **metrics})
    return pd.DataFrame(rows)

def build_hierarchy_invariance(screen_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare source-vs-zero states under screen-pair, family-pair, and target aggregation."""
    level_frames: list[pd.DataFrame] = []
    level_join_dimensions = {'screen_pair': PAIR_KEYS, 'family_pair': ['source_family', 'target_family'], 'target_screen': ['target_screen']}
    for level in ['screen_pair', 'family_pair', 'target_screen']:
        frame = _aggregate_level(screen_scores, level)
        grouping = CONFIG_KEYS + ['model']
        macro_rows: list[dict[str, object]] = []
        for keys, part in frame.groupby(grouping, sort=True, dropna=False):
            values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
            scored = part[part['status'].eq('scored')]
            if scored.empty:
                macro_rows.append({**values, 'aggregation_level': level, 'status': 'zero_eligible', 'n_units': 0, 'mse': np.nan, 'rmse_pp': np.nan, 'mae_pp': np.nan, 'n_groups': 0})
            else:
                mse = float(scored['mse'].mean())
                macro_rows.append({**values, 'aggregation_level': level, 'status': 'scored', 'n_units': int(scored['n_units'].sum()), 'mse': mse, 'rmse_pp': float(100.0 * math.sqrt(mse)), 'mae_pp': float(scored['mae_pp'].mean()), 'n_groups': int(len(scored))})
        macro = pd.DataFrame(macro_rows)
        join_key = CONFIG_KEYS + ['aggregation_level']
        zero = macro[macro['model'].eq('zero')][join_key + ['mse', 'rmse_pp']].rename(columns={'mse': 'zero_mse', 'rmse_pp': 'zero_rmse_pp'})
        macro = macro.merge(zero, on=join_key, how='left', validate='many_to_one')
        level_frames.append(macro)
    levels = pd.concat(level_frames, ignore_index=True, sort=False)
    levels['transfer_state'] = ['baseline_zero' if model == 'zero' else classify_transfer_state(status, rmse, zero_rmse) for model, status, rmse, zero_rmse in zip(levels['model'], levels['status'], levels['rmse_pp'], levels['zero_rmse_pp'])]
    levels['mse_delta_vs_zero'] = levels['mse'] - levels['zero_mse']
    summary_rows: list[dict[str, object]] = []
    for keys, part in levels[levels['model'].ne('zero')].groupby(CONFIG_KEYS + ['model'], sort=True, dropna=False):
        grouping = CONFIG_KEYS + ['model']
        values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        state_map = {row['aggregation_level']: row['transfer_state'] for _, row in part.iterrows()}
        substantive = [state for state in state_map.values() if state in {'supported_positive_vs_zero', 'supported_negative_vs_zero', 'supported_neutral_vs_zero'}]
        same = len(substantive) > 0 and len(set(substantive)) == 1 and (len(substantive) == len(state_map))
        if len(substantive) == 0:
            hierarchy_state = 'support_limited'
        elif same:
            hierarchy_state = 'hierarchy_invariant'
        else:
            hierarchy_state = 'aggregation_fragile'
        row: dict[str, object] = {**values, 'hierarchy_state': hierarchy_state, 'n_levels': int(len(state_map)), 'n_scored_levels': int(len(substantive)), 'state_screen_pair': state_map.get('screen_pair', 'missing'), 'state_family_pair': state_map.get('family_pair', 'missing'), 'state_target_screen': state_map.get('target_screen', 'missing')}
        for level in ['screen_pair', 'family_pair', 'target_screen']:
            local = part[part['aggregation_level'].eq(level)]
            if local.empty:
                row[f'mse_delta_{level}'] = np.nan
                row[f'rmse_pp_{level}'] = np.nan
            else:
                row[f'mse_delta_{level}'] = float(local['mse_delta_vs_zero'].iloc[0])
                row[f'rmse_pp_{level}'] = float(local['rmse_pp'].iloc[0])
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(CONFIG_KEYS + ['model']).reset_index(drop=True)

def _central_filter(frame: pd.DataFrame, model: str | None=None, threshold: int | None=1) -> pd.DataFrame:
    mask = (frame['mode'] == 'strict') & (frame['background_control'] == 'background_free') & (frame['basis'] == 'position_2') & (frame['relation'] == 'cross_family') & (frame['partner_overlap'] == 'all')
    if threshold is not None:
        mask &= frame['min_partners_each'].eq(threshold)
    if model is not None and 'model' in frame:
        mask &= frame['model'].eq(model)
    return frame.loc[mask].copy()

def plot_results(paths: pd.DataFrame, composition: pd.DataFrame, hierarchy: pd.DataFrame, screen_scores: pd.DataFrame | None=None) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    plot_paths = paths if screen_scores is None else build_lattice_paths(screen_scores.assign(n_prediction_units=screen_scores['n_units']))
    central = _central_filter(plot_paths, 'source_conditioned', threshold=None)
    if not central.empty:
        counts = central.groupby(['min_partners_each', 'transfer_state']).size().unstack(fill_value=0)
        counts = counts.reindex(columns=['supported_positive_vs_zero', 'supported_negative_vs_zero', 'supported_neutral_vs_zero', 'supported_unresolved_vs_zero', 'zero_eligible'], fill_value=0)
        counts.plot(kind='bar', stacked=True, ax=axes[0], color=['#2a9d8f', '#e76f51', '#e9c46a', '#8d99ae', '#6c757d'])
        axes[0].set_title('A  Central transport lattice\nsource-conditioned, cross-family')
        axes[0].set_xlabel('minimum partners per side')
        axes[0].set_ylabel('score rows')
        axes[0].legend(frameon=False, fontsize=7)
    else:
        axes[0].text(0.5, 0.5, 'No central lattice rows', ha='center', va='center')
        axes[0].set_axis_off()
    comp = composition[(composition['aggregation_level'] == 'family_pair') & (composition['mode'] == 'strict') & (composition['background_control'] == 'background_free') & (composition['basis'] == 'position_2') & (composition['relation'] == 'cross_family') & (composition['min_partners_each'] == 1) & (composition['partner_overlap'] == 'all') & composition['delta_left_minus_right'].notna()]
    if not comp.empty:
        summary = comp.groupby('delta_name')['delta_left_minus_right'].agg(['median', 'count'])
        axes[1].bar(summary.index, summary['median'], color=['#457b9d', '#bc6c25', '#6a994e'][:len(summary)])
        axes[1].axhline(0, color='#444444', linewidth=0.8)
        axes[1].set_title('B  Composition increments\nfamily-pair macro MSE')
        axes[1].set_ylabel('left MSE - right MSE')
        axes[1].tick_params(axis='x', rotation=25)
        for i, (label, row) in enumerate(summary.iterrows()):
            axes[1].text(i, row['median'], f"n={int(row['count'])}", ha='center', va='bottom' if row['median'] >= 0 else 'top', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'No central composition rows', ha='center', va='center')
        axes[1].set_axis_off()
    path = FIGURES / '24_transport_lattice_composition.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def _state_counts(paths: pd.DataFrame, mode: str, background: str, basis: str, relation: str, model: str, threshold: int, overlap: str) -> dict[str, int]:
    local = paths[(paths['mode'] == mode) & (paths['background_control'] == background) & (paths['basis'] == basis) & (paths['relation'] == relation) & (paths['model'] == model) & (paths['min_partners_each'] == threshold) & (paths['partner_overlap'] == overlap)]
    return {str(key): int(value) for key, value in local['transfer_state'].value_counts(dropna=False).to_dict().items()}

def report_text(paths: pd.DataFrame, monotonicity: pd.DataFrame, composition: pd.DataFrame, structure: pd.DataFrame, hierarchy: pd.DataFrame, screen_scores: pd.DataFrame | None=None) -> str:
    screen_paths = paths if screen_scores is None else build_lattice_paths(screen_scores.assign(n_prediction_units=screen_scores['n_units']))
    central_counts = {threshold: _state_counts(screen_paths, 'strict', 'background_free', 'position_2', 'cross_family', 'source_conditioned', threshold, 'all') for threshold in [1, 3, 5]}
    central_comp = composition[(composition['aggregation_level'] == 'family_pair') & (composition['mode'] == 'strict') & (composition['background_control'] == 'background_free') & (composition['basis'] == 'position_2') & (composition['relation'] == 'cross_family') & (composition['min_partners_each'] == 1) & (composition['partner_overlap'] == 'all')]
    comp_lines = []
    for name in ['pair_given_position', 'condition_given_pair', 'source_given_zero']:
        part = central_comp[(central_comp['delta_name'] == name) & central_comp['delta_left_minus_right'].notna()]
        pos = int((part['delta_left_minus_right'] > EPS).sum())
        neg = int((part['delta_left_minus_right'] < -EPS).sum())
        comp_lines.append(f"| `{name}` | {len(part)} | {pos} | {neg} | {(float(part['delta_left_minus_right'].median()) if len(part) else float('nan')):.4g} |")
    central_h = _central_filter(hierarchy)
    hierarchy_counts = central_h['hierarchy_state'].value_counts().to_dict() if not central_h.empty else {}
    central_structure = structure[(structure['mode'] == 'strict') & (structure['background_control'] == 'background_free') & (structure['basis'] == 'position_2') & (structure['relation'] == 'cross_family') & (structure['min_partners_each'] == 1) & (structure['partner_overlap'] == 'all') & (structure['model'] == 'source_conditioned')]
    structure_positive = int((central_structure['transfer_state'] == 'supported_positive_vs_zero').sum())
    structure_negative = int((central_structure['transfer_state'] == 'supported_negative_vs_zero').sum())
    mono_c = monotonicity[(monotonicity['mode'] == 'strict') & (monotonicity['background_control'] == 'background_free') & (monotonicity['basis'] == 'position_2') & (monotonicity['relation'] == 'cross_family') & (monotonicity['partner_overlap'] == 'all') & (monotonicity['model'] == 'source_conditioned')]
    mono_rate = float(mono_c['support_monotone'].mean()) if len(mono_c) else np.nan
    return f'# 34_TRANSPORTABILITY_LATTICE_AND_COMPOSITION_2026-10-16\n\nDate: 2026-10-16\n\n## Question and scope\n\nThis round follows reports 32–33 and moves cross-library reuse from "support and direction" to matching-level paths, joint conditional increments and aggregation-level invariance. The analysis reads only the already generated `controlled_bridge_directed_bridge_registry.csv`, `controlled_bridge_scores.csv` and the support frontier table; raw data are not modified, and unit, screen or family are not treated as independent experimental replicates.\n\n## A. Transportability lattice\n\nWithin the same scheme, background control, basis, relation, partner threshold and overlap, `zero_eligible` and `support_threshold_unavailable` are kept first, and the error of nonzero models relative to zero is then labelled `supported_positive_vs_zero`, `supported_negative_vs_zero` or `supported_neutral_vs_zero`. `position_1` and `position_2` are two independent registered positions and are not nested paths of each other; `full_pair` is used only as a stricter joint level of common substrate pairs. State counts for strict/background-free/position-2/cross-family/source-conditioned are as follows (based on directed screen-pair summaries within the same support level; the family-level score table is saved separately, and screen-pairs are not treated as independent experimental replicates):\n\n| minimum partners | state counts |\n|---:|---|\n| 1 | `{json.dumps(central_counts[1], ensure_ascii=False)}` |\n| 3 | `{json.dumps(central_counts[3], ensure_ascii=False)}` |\n| 5 | `{json.dumps(central_counts[5], ensure_ascii=False)}` |\n\nThe support monotonicity rate along partner-threshold paths (within the same mode/background/basis/relation/overlap/model) is `{mono_rate:.3f}`; only `min1→min3→min5` within a fixed basis is compared here, and `no_shared_partner` is a separate sensitivity axis independent of the partner threshold. This is a descriptive check of lattice states, not a continuous cost law. Amount strict/dose-relaxed is not treated as a strict inclusion relation, and exact/partial/none amount overlap is kept separately.\n\n## B. Joint conditional information increments\n\nComparisons are made on the same target unit, screen pair, support threshold and denominator:\n\n```text\npair_given_position  = MSE(position_marginal) - MSE(pair_marginal)\ncondition_given_pair = MSE(pair_marginal) - MSE(source_conditioned)\nsource_given_zero    = MSE(zero) - MSE(source_conditioned)\n```\n\nBelow are finite comparisons for central strict/background-free/position-2/cross-family, family-pair macro-averaged; a positive value means the left model has higher error and adding the right-hand information lowers the error.\n\n| increment | finite family-pairs | positive | negative | median delta MSE |\n|---|---:|---:|---:|---:|\n{chr(10).join(comp_lines)}\n\nThis decomposition answers whether joint identity adds an increment over existing position support; a positive increment is not described as chemical interaction or causal synergy. In the bridge producer, `source_conditioned` directly takes the observed conditional mean on the source side (code semantics `source_conditioned=source["observed"]`) and is not a separately fitted asymmetric model; `condition_given_pair` is therefore read as "increment of the source observed mean over the pair marginal" and is not mechanistic evidence from an independent predictor. Common support for full-pair and exact-background is very sparse; all missing values, undefined zero baselines and small denominators are kept.\n\n## C. Cross-screen structural boundary\n\nAfter stratification by partner redundancy, shared-partner bin, amount-set overlap and matching level, the structural strata of central source-conditioned have `{structure_positive}` positive and `{structure_negative}` negative relative to zero; the main criterion is the RMSE/MAE difference, gain is auxiliary only, and these are conditional means within structural bins, not universal thresholds. Structural features come from source/target registry coverage; target observations are not used to choose bins.\n\n## D. Hierarchy invariance\n\nUnder the same directed-screen setting, three aggregations are compared: screen-pair, family-pair and target-screen. Hierarchy state counts for central source-conditioned are `{json.dumps({str(k): int(v) for k, v in hierarchy_counts.items()}, ensure_ascii=False)}`. The target-screen level here is the target-screen macro-average within the directed registry; `pooled family-excluded` is a parallel estimand and is not treated as a fourth level of unit→screen→family. `hierarchy_invariant` means the state relative to zero agrees across the three levels; `aggregation_fragile` means the state changes with aggregation level; `support_limited` means there is no comparable nonzero level.\n\n## Interpretation boundary\n\nThese results are a retrospective structural analysis of frozen historical conditional means. `zero_eligible` does not mean negative transfer; `support_threshold_unavailable` does not mean failure; added coverage from dose relaxation cannot by itself be read as chemical transportability; a registered position across families is not a mechanistic role. Composition increments show only differences in prediction error on a common denominator, not reaction mechanism, causal synergy or transferability of individual reactions. Structural bins, hierarchy states and cluster bootstrap (if added later) serve only as boundary diagnostics and do not replace prospective whole-family hold-out or new common-background experiments.\n\n## Outputs\n\n- `tables/transport_lattice_paths.csv`\n- `tables/transport_lattice_monotonicity.csv`\n- `tables/transport_lattice_screen_pair_scores.csv`\n- `tables/transport_composition_deltas.csv`\n- `tables/transport_structure_boundary.csv`\n- `tables/transport_hierarchy_invariance.csv`\n- `figures/24_transport_lattice_composition.png`\n- `data/transport_lattice_composition_manifest.json`\n'

def main() -> None:
    with threadpool_limits(limits=1):
        registry, score_input, input_hashes = load_inputs()
        screen_scores = build_screen_pair_scores(registry)
        paths = build_lattice_paths(screen_scores)
        monotonicity = build_lattice_monotonicity(paths)
        composition = build_composition_deltas(registry)
        structure = build_structure_boundary(registry)
        hierarchy = build_hierarchy_invariance(screen_scores)
        TABLES.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, pd.DataFrame] = {'transport_lattice_paths.csv': paths, 'transport_lattice_monotonicity.csv': monotonicity, 'transport_lattice_screen_pair_scores.csv': screen_scores, 'transport_composition_deltas.csv': composition, 'transport_structure_boundary.csv': structure, 'transport_hierarchy_invariance.csv': hierarchy}
        output_paths: list[Path] = []
        for name, frame in outputs.items():
            path = TABLES / name
            frame.to_csv(path, index=False, encoding='utf-8-sig')
            output_paths.append(path)
        figure = plot_results(paths, composition, hierarchy, screen_scores)
        output_paths.append(figure)
        report = HERE / '34_TRANSPORTABILITY_LATTICE_AND_COMPOSITION_2026-10-16.md'
        report.write_text(report_text(paths, monotonicity, composition, structure, hierarchy, screen_scores), encoding='utf-8')
        output_paths.append(report)
        manifest = {'date': '2026-10-16', 'scope': 'Existing controlled bridge transportability lattice, composition increments, structure strata, and aggregation invariance', 'input_hashes': input_hashes, 'code_hashes': {name: digest(HERE / name) for name in ['transport_lattice_composition.py', 'test_transport_lattice_composition.py']}, 'output_hashes': {str(path.relative_to(HERE)): digest(path) for path in output_paths}, 'counts': {'registry_rows': int(len(registry)), 'score_input_rows': int(len(score_input)), 'lattice_path_rows': int(len(paths)), 'lattice_zero_eligible_rows': int(paths['status'].eq('zero_eligible').sum()), 'lattice_support_threshold_unavailable_rows': int(paths['status'].eq('support_threshold_unavailable').sum()), 'lattice_monotonicity_rows': int(len(monotonicity)), 'composition_rows': int(len(composition)), 'composition_screen_pair_rows': int((composition.get('aggregation_level', pd.Series(dtype=object)) == 'screen_pair').sum()), 'composition_family_pair_rows': int((composition.get('aggregation_level', pd.Series(dtype=object)) == 'family_pair').sum()), 'structure_rows': int(len(structure)), 'hierarchy_rows': int(len(hierarchy)), 'hierarchy_invariant_rows': int((hierarchy.get('hierarchy_state', pd.Series(dtype=object)) == 'hierarchy_invariant').sum()), 'hierarchy_fragile_rows': int((hierarchy.get('hierarchy_state', pd.Series(dtype=object)) == 'aggregation_fragile').sum()), 'hierarchy_support_limited_rows': int((hierarchy.get('hierarchy_state', pd.Series(dtype=object)) == 'support_limited').sum())}, 'definitions': {'positive_vs_zero': 'RMSE(model) < RMSE(zero) on the same score stratum; no gain ratio is required.', 'zero_eligible': 'No selected bridge rows or no scored target units at the declared support threshold; never converted to negative transfer.', 'support_threshold_unavailable': 'A declared row exists but a finite common prediction/error is unavailable; kept distinct from zero_eligible.', 'composition': 'Pairwise MSE/MAE differences use exactly the same finite observed units for both predictions.', 'source_conditioned': 'The producer defines source_conditioned as the source-side observed conditional mean; it is not an independently fitted asymmetric predictor.', 'hierarchy': 'Screen-pair, family-pair, and target-screen summaries average MSE equally within the named aggregation level.', 'pooled_family_excluded': 'Pooled family-excluded summaries remain a parallel estimand and are not inserted as a fourth hierarchy level.'}, 'verification': {'required_models_present_in_registry': set(MODEL_COLUMNS).issubset(set(registry.columns)), 'required_score_axes_present': all((column in score_input.columns for column in ['scheme', 'mode', 'background_control', 'basis', 'relation', 'min_partners_each', 'partner_overlap', 'status'])), 'zero_eligible_rmse_missing': bool(paths.loc[paths['status'].eq('zero_eligible'), 'rmse_pp'].isna().all()), 'raw_inputs_unchanged_during_run': input_hashes == {str(path.relative_to(HERE)): digest(path) for path in [REGISTRY_FILE, SCORE_FILE, FRONTIER_FILE] if path.exists()}, 'composition_common_units_positive_only': bool(composition.loc[composition['delta_state'].isin(['positive_increment', 'negative_increment', 'neutral_increment']), 'n_common'].gt(0).all()) if len(composition) else True, 'no_zero_eligible_negative_transfer': bool(not paths.loc[paths['status'].eq('zero_eligible'), 'transfer_state'].isin(['supported_negative_vs_zero', 'supported_positive_vs_zero']).any()), 'position_1_position_2_kept_separate': bool(set(paths['basis'].unique()).issuperset({'position_1', 'position_2'})), 'source_conditioned_column_present': 'source_conditioned' in registry.columns}, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'blas_threads': 'library_default_unrestricted', 'invoked_with': 'python -B'}}
        (DATA / 'transport_lattice_composition_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(manifest['counts'], ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
