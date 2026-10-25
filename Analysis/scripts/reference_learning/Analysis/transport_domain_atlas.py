"""Complete registered directed-pair domain, with registry support provenance.

The domain is the union of observed directed pairs crossed with admissible
settings, not an inferred complete screen network. Empty support and unknown
outcomes remain distinct. Repeated cells are descriptive, not replicates."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PATHS_FILE = TABLES / 'transport_lattice_paths.csv'
REGISTRY_FILE = TABLES / 'controlled_bridge_directed_bridge_registry.csv'
SPECTRA_FILE = TABLES / 'shared_latent_dimension_screen.csv'
PAIR_KEYS = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
CONFIG_KEYS = ['mode', 'background_control', 'basis', 'min_partners_each', 'partner_overlap']
BASE_KEYS = ['mode', 'background_control', 'basis']
MODEL = 'source_conditioned'
STATES = ['geometry_absent', 'filter_empty', 'prediction_unavailable', 'scored_negative', 'scored_neutral', 'scored_positive']
ANCHOR = dict(mode='strict', background_control='background_free', basis='position_2', min_partners_each=1, partner_overlap='all')
EPS = 1e-12

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def admissible_settings() -> list[dict[str, object]]:
    settings = []
    for mode, background, basis in itertools.product(('strict', 'dose_relaxed'), ('background_free', 'exact_background'), ('position_1', 'position_2', 'full_pair')):
        thresholds = (1,) if basis == 'full_pair' else (1, 3, 5)
        overlaps = ('all',) if basis == 'full_pair' else ('all', 'no_shared_partner')
        for threshold, overlap in itertools.product(thresholds, overlaps):
            settings.append(dict(zip(CONFIG_KEYS, (mode, background, basis, threshold, overlap))))
    return settings

def validate_configuration(config: dict[str, object]) -> None:
    if {key: config[key] for key in CONFIG_KEYS} not in admissible_settings():
        raise ValueError(f'structurally inapplicable configuration: {config}')

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    files = (PATHS_FILE, REGISTRY_FILE, SPECTRA_FILE)
    paths, registry, spectra = [pd.read_csv(path, keep_default_na=False) for path in files]
    paths['min_partners_each'] = pd.to_numeric(paths.min_partners_each)
    for column in ('source_n_partners', 'target_n_partners', 'shared_partner_count', 'observed', MODEL, 'zero'):
        registry[column] = pd.to_numeric(registry[column], errors='coerce')
    return (paths, registry, spectra, {str(path.relative_to(HERE)): digest(path) for path in files})

def select_rows(registry: pd.DataFrame, pair: dict[str, object], config: dict[str, object], filtered: bool=True) -> pd.DataFrame:
    validate_configuration(config)
    mask = pd.Series(True, index=registry.index)
    for key in PAIR_KEYS + BASE_KEYS:
        mask &= registry[key].eq(pair[key] if key in PAIR_KEYS else config[key])
    if filtered:
        mask &= registry.source_n_partners.ge(config['min_partners_each']) & registry.target_n_partners.ge(config['min_partners_each'])
        if config['partner_overlap'] == 'no_shared_partner':
            mask &= registry.shared_partner_count.eq(0)
    return registry.loc[mask]

def registry_metrics(raw: pd.DataFrame) -> dict[str, object]:
    result = dict(n_selected_rows=len(raw), n_selected_identities=raw.unit_identity.nunique(), n_source_finite=0, n_zero_finite=0, n_common_finite=0, common_source_rmse_pp=np.nan, common_zero_rmse_pp=np.nan)
    if raw.empty:
        return result
    values = raw[['unit_identity', 'observed', MODEL, 'zero']].copy()
    for column in ('observed', MODEL, 'zero'):
        values[column] = pd.to_numeric(values[column], errors='coerce')
    collapsed = values.dropna(subset=['observed']).groupby('unit_identity', sort=True)[['observed', MODEL, 'zero']].mean()
    source = np.isfinite(collapsed.observed) & np.isfinite(collapsed[MODEL])
    zero = np.isfinite(collapsed.observed) & np.isfinite(collapsed.zero)
    common = source & zero
    result.update(n_source_finite=int(source.sum()), n_zero_finite=int(zero.sum()), n_common_finite=int(common.sum()))
    if common.any():
        result['common_source_rmse_pp'] = float(100 * np.sqrt(np.mean((collapsed.loc[common, MODEL] - collapsed.loc[common, 'observed']) ** 2)))
        result['common_zero_rmse_pp'] = float(100 * np.sqrt(np.mean((collapsed.loc[common, 'zero'] - collapsed.loc[common, 'observed']) ** 2)))
    return result

def collapsed_finite_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw[['unit_identity', 'observed', MODEL, 'zero']].copy()
    for column in ('observed', MODEL, 'zero'):
        values[column] = pd.to_numeric(values[column], errors='coerce')
    values = values.dropna(subset=['observed']).groupby('unit_identity')[['observed', MODEL, 'zero']].mean()
    return values[np.isfinite(values).all(axis=1)]

def cell_state(n_base_rows: int, metrics: dict[str, object], path_status: str, zero_status: str) -> str:
    if n_base_rows == 0:
        return 'geometry_absent'
    if metrics['n_selected_rows'] == 0:
        return 'filter_empty'
    if path_status != 'scored' or zero_status != 'scored' or metrics['n_common_finite'] == 0:
        return 'prediction_unavailable'
    delta = float(metrics['common_source_rmse_pp']) - float(metrics['common_zero_rmse_pp'])
    return 'scored_positive' if delta < -EPS else 'scored_negative' if delta > EPS else 'scored_neutral'

def build_domain_atlas(paths: pd.DataFrame, registry: pd.DataFrame, spectra: pd.DataFrame) -> pd.DataFrame:
    local = paths[paths.model.isin([MODEL, 'zero'])].copy()
    key = PAIR_KEYS + CONFIG_KEYS + ['model']
    if local.duplicated(key).any():
        raise ValueError('duplicate pair/config/model lattice row')
    if spectra.screen.duplicated().any():
        raise ValueError('duplicate spectrum screen')
    pairs = registry[PAIR_KEYS].drop_duplicates().sort_values(PAIR_KEYS)
    lookup = {tuple((row[k] for k in key)): row for row in local.to_dict('records')}
    anchor_pairs = {tuple((row[k] for k in PAIR_KEYS)) for row in local[(local.model == MODEL) & (local.relation == 'cross_family') & np.logical_and.reduce([local[k].eq(v) for k, v in ANCHOR.items()])].to_dict('records')}
    rows = []
    for pair in pairs.to_dict('records'):
        cohort = 'within_family' if pair['relation'] == 'within_family' else 'central42' if tuple((pair[k] for k in PAIR_KEYS)) in anchor_pairs else 'off_anchor22'
        for config in admissible_settings():
            combined = {**pair, **config}
            source = lookup.get(tuple((combined[k] for k in PAIR_KEYS + CONFIG_KEYS)) + (MODEL,))
            zero = lookup.get(tuple((combined[k] for k in PAIR_KEYS + CONFIG_KEYS)) + ('zero',))
            raw = select_rows(registry, pair, config, filtered=False)
            selected = select_rows(registry, pair, config)
            if (len(raw) > 0) != (source is not None):
                raise ValueError('atomic geometry and lattice registration disagree')
            metrics = registry_metrics(selected)
            source_status, zero_status = (str(source['status']) if source else 'absent_configuration', str(zero['status']) if zero else 'absent_configuration')
            state = cell_state(len(raw), metrics, source_status, zero_status)
            rows.append({**combined, 'cohort': cohort, 'n_base_rows': len(raw), 'path_present': source is not None, 'path_status': source_status, 'zero_status': zero_status, **metrics, 'domain_state': state, 'outcome_known': state.startswith('scored_'), 'zero_baseline_state': 'scored_zero_baseline' if metrics['n_zero_finite'] > 0 and zero_status == 'scored' else 'unknown_zero_baseline', 'saved_transfer_state': source['transfer_state'] if source else 'absent_configuration', 'rmse_delta_vs_zero_pp': metrics['common_source_rmse_pp'] - metrics['common_zero_rmse_pp']})
    result = pd.DataFrame(rows)
    spectrum_columns = ['screen', 'k_var90', 'participation_rank', 'observed_rank']
    for role in ('source', 'target'):
        rename = {column: role + '_' + column for column in spectrum_columns}
        result = result.merge(spectra[spectrum_columns].rename(columns=rename), on=role + '_screen', how='left', validate='many_to_one')
    result['absolute_participation_rank_gap'] = (pd.to_numeric(result.source_participation_rank) - pd.to_numeric(result.target_participation_rank)).abs()
    return result.sort_values(PAIR_KEYS + CONFIG_KEYS).reset_index(drop=True)

def build_configuration_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    grouping = ['relation', 'cohort'] + CONFIG_KEYS
    result = atlas.groupby(grouping + ['domain_state'], sort=True).size().unstack(fill_value=0).reindex(columns=STATES, fill_value=0).reset_index()
    result['n_pair_denominator'] = result[STATES].sum(axis=1)
    result['n_finite_scored'] = result[['scored_positive', 'scored_negative', 'scored_neutral']].sum(axis=1)
    return result

def build_pair_inventory(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in atlas.groupby(PAIR_KEYS + ['cohort'], sort=True):
        rows.append({**dict(zip(PAIR_KEYS + ['cohort'], keys)), 'n_admissible': len(part), 'n_materialized': int(part.path_present.sum()), **{state: int(part.domain_state.eq(state).sum()) for state in STATES}, 'n_scored_strict': int((part.outcome_known & part['mode'].eq('strict')).sum()), 'n_scored_dose_relaxed': int((part.outcome_known & part['mode'].eq('dose_relaxed')).sum()), 'source_participation_rank': part.source_participation_rank.iloc[0], 'target_participation_rank': part.target_participation_rank.iloc[0], 'source_k_var90': part.source_k_var90.iloc[0], 'target_k_var90': part.target_k_var90.iloc[0]})
    return pd.DataFrame(rows)

def sign_transition(a: str, b: str) -> str:
    known = {'scored_positive', 'scored_negative', 'scored_neutral'}
    if a not in known or b not in known:
        return 'unknown_comparison'
    if a == b:
        return 'same_outcome'
    if 'scored_neutral' in (a, b):
        return 'to_or_from_neutral'
    return 'positive_negative_reversal'

def _identity_sets(rows: pd.DataFrame) -> tuple[set[str], set[tuple[str, str]]]:
    return (set(rows.unit_identity), set(map(tuple, rows[['source_unit_id', 'target_unit_id']].to_numpy())))

def build_dose_contrasts(atlas: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    key = PAIR_KEYS + [k for k in CONFIG_KEYS if k != 'mode']
    strict = atlas[atlas['mode'].eq('strict')].set_index(key)
    relaxed = atlas[atlas['mode'].eq('dose_relaxed')].set_index(key)
    rows = []
    for keys, left in strict.iterrows():
        right = relaxed.loc[keys]
        pair_config = dict(zip(key, keys))
        pair = {k: pair_config[k] for k in PAIR_KEYS}
        left_raw = select_rows(registry, pair, {**pair_config, 'mode': 'strict'})
        right_raw = select_rows(registry, pair, {**pair_config, 'mode': 'dose_relaxed'})
        li, la = _identity_sets(left_raw)
        ri, ra = _identity_sets(right_raw)
        common_identities = li & ri
        left_values, right_values = (collapsed_finite_predictions(left_raw), collapsed_finite_predictions(right_raw))
        common_finite = sorted(set(left_values.index) & set(right_values.index))
        common_left = registry_metrics(left_raw[left_raw.unit_identity.isin(common_finite)])
        common_right = registry_metrics(right_raw[right_raw.unit_identity.isin(common_finite)])
        observed_change = right_values.loc[common_finite, 'observed'] - left_values.loc[common_finite, 'observed']
        common_metrics = {'common_identity_' + mode + '_' + metric: values[metric] for mode, values in (('strict', common_left), ('dose_relaxed', common_right)) for metric in ('n_common_finite', 'common_source_rmse_pp', 'common_zero_rmse_pp')}
        rows.append({**pair_config, 'cohort': left.cohort, 'strict_state': left.domain_state, 'dose_relaxed_state': right.domain_state, 'strict_known': bool(left.outcome_known), 'dose_relaxed_known': bool(right.outcome_known), 'dose_discovery': bool(right.outcome_known and (not left.outcome_known)), 'strict_only': bool(left.outcome_known and (not right.outcome_known)), 'outcome_transition': sign_transition(left.domain_state, right.domain_state), 'n_strict_identities': len(li), 'n_relaxed_identities': len(ri), 'n_common_identities': len(common_identities), 'n_strict_registered_estimand_pairs': len(la), 'n_relaxed_registered_estimand_pairs': len(ra), 'n_common_registered_estimand_pairs': len(la & ra), 'strict_rmse_delta_pp': left.rmse_delta_vs_zero_pp, 'dose_relaxed_rmse_delta_pp': right.rmse_delta_vs_zero_pp, 'common_identity_observed_mean_absolute_change_pp': float(100 * observed_change.abs().mean()) if len(observed_change) else np.nan, 'common_identity_observed_max_absolute_change_pp': float(100 * observed_change.abs().max()) if len(observed_change) else np.nan, **common_metrics})
    return pd.DataFrame(rows)

def joint_definitions() -> list[tuple[str, dict[str, object], dict[str, object]]]:
    definitions = []
    perturbations = [('dose', {'mode': 'dose_relaxed'}), ('exact', {'background_control': 'exact_background'}), ('threshold3', {'min_partners_each': 3}), ('threshold5', {'min_partners_each': 5}), ('no_shared', {'partner_overlap': 'no_shared_partner'}), ('position1', {'basis': 'position_1'}), ('full_pair', {'basis': 'full_pair'})]
    for (a, ca), (b, cb) in itertools.combinations(perturbations, 2):
        if set(ca).intersection(cb):
            continue
        joint = {**ANCHOR, **ca, **cb}
        if joint not in admissible_settings():
            continue
        definitions.append((a + '+' + b, ca, cb))
    return definitions

def build_joint_support(atlas: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    lookup = atlas.set_index(PAIR_KEYS + CONFIG_KEYS)
    rows = []
    for pair in atlas[PAIR_KEYS].drop_duplicates().to_dict('records'):
        for name, ca, cb in joint_definitions():
            configs = ({**ANCHOR, **ca}, {**ANCHOR, **cb}, {**ANCHOR, **ca, **cb})
            cells = [lookup.loc[tuple((pair[k] for k in PAIR_KEYS)) + tuple((config[k] for k in CONFIG_KEYS))] for config in configs]
            raw = [select_rows(registry, pair, config) for config in configs]
            same_basis = len({config['basis'] for config in configs}) == 1
            identities = [_identity_sets(frame)[0] for frame in raw]
            estimands = [_identity_sets(frame)[1] for frame in raw]
            marginal_known = bool(cells[0].outcome_known and cells[1].outcome_known)
            rows.append({**pair, 'cohort': cells[0].cohort, 'axis_combination': name, 'marginal_a_state': cells[0].domain_state, 'marginal_b_state': cells[1].domain_state, 'joint_state': cells[2].domain_state, 'both_marginals_known': marginal_known, 'joint_known': bool(cells[2].outcome_known), 'marginal_coverage_without_joint': bool(marginal_known and (not cells[2].outcome_known)), 'same_basis_identity_comparison': same_basis, 'n_identity_intersection_marginals': len(identities[0] & identities[1]) if same_basis else np.nan, 'n_identity_intersection_all_three': len(set.intersection(*identities)) if same_basis else np.nan, 'n_registered_estimand_intersection_all_three': len(set.intersection(*estimands)), 'joint_config': json.dumps(configs[2], sort_keys=True)})
    return pd.DataFrame(rows)

def build_location_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pair_inventory(atlas)
    return pairs.groupby(['relation', 'cohort'], sort=True).agg(n_pairs=('source_screen', 'size'), source_participation_rank_median=('source_participation_rank', 'median'), target_participation_rank_median=('target_participation_rank', 'median'), source_k_var90_median=('source_k_var90', 'median'), target_k_var90_median=('target_k_var90', 'median'), finite_dose_relaxed_configs_median=('n_scored_dose_relaxed', 'median')).reset_index()

def build_screen_inventory(atlas: pd.DataFrame, spectra: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pair_inventory(atlas)
    rows = []
    for screen in spectra.to_dict('records'):
        outgoing = pairs[pairs.source_screen.eq(screen['screen'])]
        incoming = pairs[pairs.target_screen.eq(screen['screen'])]
        rows.append({'screen': screen['screen'], 'spectrum_family': screen['family'], 'participation_rank': screen['participation_rank'], 'k_var90': screen['k_var90'], 'n_outgoing_registered_pairs': len(outgoing), 'n_incoming_registered_pairs': len(incoming), 'n_outgoing_cross_family_pairs': int(outgoing.relation.eq('cross_family').sum()), 'n_incoming_cross_family_pairs': int(incoming.relation.eq('cross_family').sum()), 'registered_domain_status': 'registered_pair_present' if len(outgoing) + len(incoming) else 'no_registered_pair_unknown_transfer_outcome'})
    return pd.DataFrame(rows).sort_values('screen').reset_index(drop=True)

def build_mode_semantics(registry: pd.DataFrame) -> pd.DataFrame:
    """Audit matching declared metadata; this does not establish atom weights."""
    data = registry.copy()

    def chemical_key(value: str) -> str:
        decoded = json.loads(value)
        if not isinstance(decoded, list) or len(decoded) not in (3, 4):
            raise ValueError('unexpected registered perturbation key schema')
        return json.dumps(decoded[:3], separators=(',', ':'))
    data['chemical_key'] = data.perturbation_key.map(chemical_key)
    signature = PAIR_KEYS + ['background_control', 'background_key', 'basis', 'unit_identity', 'chemical_key', 'source_dose_levels_json', 'target_dose_levels_json', 'source_partner_union_json', 'target_partner_union_json', 'source_contexts', 'target_contexts']
    values = ['source_unit_id', 'target_unit_id', 'observed', MODEL]
    strict = data[data['mode'].eq('strict')][signature + values]
    relaxed = data[data['mode'].eq('dose_relaxed')][signature + values]
    matched = strict.merge(relaxed, on=signature, validate='one_to_one', suffixes=('_strict', '_dose_relaxed'))
    matched['observed_difference_pp'] = 100 * (matched.observed_dose_relaxed - matched.observed_strict)
    matched['source_conditioned_difference_pp'] = 100 * (matched.source_conditioned_dose_relaxed - matched.source_conditioned_strict)
    matched['both_mode_qualified_ids_different'] = matched.source_unit_id_strict.ne(matched.source_unit_id_dose_relaxed) & matched.target_unit_id_strict.ne(matched.target_unit_id_dose_relaxed)
    return matched.sort_values(signature).reset_index(drop=True)

def build_outputs(paths: pd.DataFrame, registry: pd.DataFrame, spectra: pd.DataFrame) -> dict[str, pd.DataFrame]:
    atlas = build_domain_atlas(paths, registry, spectra)
    return {'transport_domain_atlas_cells.csv': atlas, 'transport_domain_atlas_configurations.csv': build_configuration_summary(atlas), 'transport_domain_atlas_pairs.csv': build_pair_inventory(atlas), 'transport_domain_atlas_dose_contrasts.csv': build_dose_contrasts(atlas, registry), 'transport_domain_atlas_joint_support.csv': build_joint_support(atlas, registry), 'transport_domain_atlas_location_summary.csv': build_location_summary(atlas), 'transport_domain_atlas_screens.csv': build_screen_inventory(atlas, spectra), 'transport_domain_atlas_mode_semantics.csv': build_mode_semantics(registry)}

def make_figure(outputs: dict[str, pd.DataFrame]) -> Path:
    atlas = outputs['transport_domain_atlas_cells.csv']
    pairs = outputs['transport_domain_atlas_pairs.csv']
    joint = outputs['transport_domain_atlas_joint_support.csv']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    colors = ['#c7c7c7', '#e4ba69', '#9680ae', '#bf5c62', '#6f8790', '#438e73']
    counts = atlas.groupby(['cohort', 'domain_state']).size().unstack(fill_value=0).reindex(columns=STATES, fill_value=0)
    counts.plot.bar(stacked=True, color=colors, ax=axes[0, 0], rot=0)
    axes[0, 0].set(title='Registered pair domain: support and outcome', ylabel='Pair-configuration cells')
    axes[0, 0].legend(fontsize=8)
    for cohort, part in pairs.groupby('cohort', sort=True):
        axes[0, 1].scatter(part.n_scored_strict, part.n_scored_dose_relaxed, label=cohort, alpha=0.65, s=35)
    axes[0, 1].plot([0, 26], [0, 26], color='gray', linewidth=0.7)
    axes[0, 1].set(xlabel='Finite scored strict configurations / 26', ylabel='Finite scored dose-relaxed configurations / 26', title='Dose registration expands the observed domain')
    axes[0, 1].legend(fontsize=8)
    cross = joint[joint.relation.eq('cross_family')]
    rows = []
    for name, group in cross.groupby('axis_combination', sort=True):
        rows.append((name, int(group.both_marginals_known.sum()), int((group.both_marginals_known & group.joint_known).sum())))
    names, marginal, observed = zip(*rows)
    y = np.arange(len(rows))
    axes[1, 0].barh(y, marginal, color='#b5b5b5', label='Both one-axis outcomes known')
    axes[1, 0].barh(y, observed, color='#428b76', label='Joint outcome also known')
    axes[1, 0].set(yticks=y, yticklabels=names, xlabel='Directed cross-family pairs / 64', title='Marginal coverage and actual joint coverage')
    axes[1, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].legend(fontsize=7)
    for cohort, part in pairs.groupby('cohort', sort=True):
        axes[1, 1].scatter(part.target_participation_rank, part.scored_positive + part.scored_negative + part.scored_neutral, label=cohort, alpha=0.6, s=35)
    axes[1, 1].set(xlabel='Target participation rank', ylabel='Finite scored configurations / 52', title='Spectrum complexity locates coverage descriptively')
    axes[1, 1].legend(fontsize=8)
    path = FIGURES / '38_TRANSPORT_DOMAIN_ATLAS.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def main() -> None:
    paths, registry, spectra, hashes = load_inputs()
    outputs = build_outputs(paths, registry, spectra)
    files = []
    for name, frame in outputs.items():
        path = TABLES / name
        frame.to_csv(path, index=False)
        files.append(path)
    files.append(make_figure(outputs))
    files.append(HERE / 'transport_domain_atlas.py')
    manifest = {'input_sha256': hashes, 'code_sha256': {name: digest(HERE / name) for name in ('transport_domain_atlas.py', 'test_transport_domain_atlas.py', 'verify_transport_domain_atlas.py')}, 'output_sha256': {str(path.relative_to(HERE)): digest(path) for path in files}, 'inventory': {name: len(frame) for name, frame in outputs.items()}, 'domain': 'Union registered directed pairs x 52 admissible configurations; full_pair only min1/all', 'interpretation': 'Retrospective conditional means; missing geometry and empty filters have unknown transfer outcome; no independent molecular or cell replicates; spectrum summaries descriptive'}
    (DATA / 'transport_domain_atlas_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['inventory'], indent=2))
if __name__ == '__main__':
    main()
