"""Controlled molecular bridges across existing reaction-condition libraries.

All predictions use source outcomes only. Targets are descriptive conditional
population means, not prospective predictions for individual reactions."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import itertools
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from cross_library_component_structure import canonical_pair, transformation
HERE = Path(__file__).resolve().parent
PREFIX = 'controlled_bridge_'
MODES = {'strict': 'perturbation_key', 'dose_relaxed': 'no_cross_screen_dose_key'}
BASES = ['position_1', 'position_2', 'full_pair']
MODELS = ['zero', 'source_pair_marginal', 'source_position_marginal', 'source_marginal_excluding_unit', 'source_conditioned', 'source_conditioned_shrunk5', 'source_all_available_marginal']
GROUP = ['mode', 'background_control', 'screen', 'family', 'perturbation_key', 'background_key']
MATCH = ['mode', 'background_control', 'perturbation_key', 'background_key', 'basis', 'unit_identity']

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()

def source_relation_allowed(source_family, target_family, relation):
    return source_family == target_family if relation == 'within_family' else source_family != target_family

def balance_pairs(cells):
    result = cells.groupby(['sub_1', 'sub_2'], as_index=False).agg(delta=('delta', 'mean'), n_contexts=('contrast_id', 'nunique'))
    assert not result.duplicated(['sub_1', 'sub_2']).any()
    return result

def leave_one_unit_out(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return (values.copy(), np.zeros(len(values), bool))
    return ((values.sum() - values) / (len(values) - 1), np.ones(len(values), bool))

def full_pair_units(pairs):
    result = pairs.copy()
    result['basis'] = 'full_pair'
    result['unit_identity'] = [dumps([a, b]) for a, b in zip(result.sub_1, result.sub_2)]
    return result

def family_source_mean(frame, column):
    return float(frame.groupby(['source_family', 'source_screen'])[column].mean().groupby('source_family').mean().mean())

def pooled_source_scale(frame):
    values = frame[['source_family', 'source_screen', 'source_yield_sd']].drop_duplicates().copy()
    values['variance'] = values.source_yield_sd ** 2
    return float(np.sqrt(family_source_mean(values, 'variance')))

def load_atomic_effects():
    old_manifest = HERE / 'data/cross_library_components_manifest.json'
    old = json.loads(old_manifest.read_text(encoding='utf-8'))
    for path, expected in old['input_hashes'].items():
        assert digest(path) == expected, path
    paths = [Path(path) for path in old['input_hashes']]
    paths += [old_manifest, HERE / 'tables/cross_library_components_perturbation_contexts.csv', HERE / 'tables/cross_library_components_matching_substrate_comparisons.csv', HERE / 'tables/cross_library_components_screen_inventory.csv', HERE / 'cross_library_component_structure.py']
    paths = list(dict.fromkeys(paths))
    hashes = {str(path): digest(path) for path in paths}
    contrasts = pd.read_csv(HERE / 'tables/cross_library_components_perturbation_contexts.csv', keep_default_na=False)
    inventory = pd.read_csv(HERE / 'tables/cross_library_components_screen_inventory.csv', keep_default_na=False)
    frames, screen_scales = ([], {})
    for screen in inventory.screen:
        path = HERE / 'data' / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            matrix, conditions = (saved['M'], saved['conds'].astype(str))
            pairs = [canonical_pair(value) for value in saved['sps']]
        clean = np.array([not (screen.split('|')[0] == 'SNAr' and value == 'C2') and (not (screen.split('|')[0] == 'Chan-Lam' and value == 'C68')) for value in conditions])
        screen_scales[screen] = float(np.nanstd(matrix[:, clean]))
        lookup = {condition: index for index, condition in enumerate(conditions)}
        subset = contrasts[contrasts.screen.eq(screen)]
        for row in subset.itertuples(index=False):
            low, high = (matrix[:, lookup[row.condition_low]], matrix[:, lookup[row.condition_high]])
            keep = np.flatnonzero(np.isfinite(low) & np.isfinite(high))
            assert len(keep) == row.n_overlap
            delta = high[keep] - low[keep]
            assert abs(delta.mean() - row.mean_delta) < 1e-12
            frames.append(pd.DataFrame(dict(contrast_id=row.contrast_id, pair_index=keep, sub_1=[pairs[index][0] for index in keep], sub_2=[pairs[index][1] for index in keep], yield_low=low[keep], yield_high=high[keep], delta=delta)))
    atomic = pd.concat(frames, ignore_index=True)
    assert len(atomic) == int(contrasts.n_overlap.sum()) == 502945
    assert not atomic.duplicated(['contrast_id', 'pair_index']).any()
    assert set(inventory.screen) == set(contrasts.screen) and len(inventory) == 19
    return (contrasts, inventory, atomic, screen_scales, hashes)

def build_estimands(contrasts, atomic, scales):
    unit_rows, pair_frames, context_rows = ([], [], [])
    for mode, key_column in MODES.items():
        key_screens = contrasts.groupby(key_column).screen.nunique()
        recurrent_keys = set(key_screens[key_screens >= 2].index)
        selected = contrasts[contrasts[key_column].isin(recurrent_keys)]
        metadata = selected[['contrast_id', 'screen', 'family', key_column, 'background_key', 'fixed_equiv']].rename(columns={key_column: 'bridge_key'})
        joined = atomic.merge(metadata, on='contrast_id', validate='many_to_one', how='inner')
        for control in ('background_free', 'exact_background'):
            values = joined.copy()
            values['matched_background'] = '*' if control == 'background_free' else values.background_key
            group_columns = ['screen', 'family', 'bridge_key', 'matched_background']
            for keys, cells in values.groupby(group_columns, sort=True):
                screen, family, key, background = keys
                prefix = dict(mode=mode, background_control=control, screen=screen, family=family, perturbation_key=key, background_key=background)
                paired = balance_pairs(cells)
                paired['mode'], paired['background_control'] = (mode, control)
                paired['screen'], paired['family'] = (screen, family)
                paired['perturbation_key'], paired['background_key'] = (key, background)
                pair_frames.append(paired)
                context_rows.append(dict(**prefix, n_condition_contexts=cells.contrast_id.nunique(), n_unique_full_pairs=len(paired), n_atomic_cells=len(cells), dose_levels_json=dumps(sorted(set(cells.fixed_equiv.astype(str)))), n_dose_levels=cells.fixed_equiv.nunique(), pair_marginal=paired.delta.mean()))
                base = dict(**prefix, source_yield_sd=scales[screen], pair_marginal=float(paired.delta.mean()), n_condition_contexts=int(cells.contrast_id.nunique()), dose_levels_json=dumps(sorted(set(cells.fixed_equiv.astype(str)))), n_dose_levels=int(cells.fixed_equiv.nunique()))
                for basis, molecule_column, partner_column in [('position_1', 'sub_1', 'sub_2'), ('position_2', 'sub_2', 'sub_1')]:
                    grouped = list(paired.groupby(molecule_column, sort=True))
                    estimates = np.array([group.delta.mean() for _, group in grouped])
                    excluded, available = leave_one_unit_out(estimates)
                    for index, (molecule, group) in enumerate(grouped):
                        partner_effects = dict(zip(group[partner_column], group.delta))
                        row = dict(**base, basis=basis, unit_identity=molecule, observed=float(estimates[index]), position_marginal=float(estimates.mean()), marginal_excluding_unit=float(excluded[index]), exclusion_available=bool(available[index]), n_units_in_position=len(grouped), n_partners=len(group), n_atomic_cells=int(group.n_contexts.sum()), min_contexts_per_partner=int(group.n_contexts.min()), max_contexts_per_partner=int(group.n_contexts.max()), partner_effects_json=dumps(partner_effects))
                        row['unit_id'] = fingerprint([row[field] for field in GROUP + ['basis', 'unit_identity']])
                        unit_rows.append(row)
                full = full_pair_units(paired)
                excluded, available = leave_one_unit_out(full.delta.to_numpy())
                for index, row in enumerate(full.itertuples(index=False)):
                    item = dict(**base, basis='full_pair', unit_identity=row.unit_identity, observed=float(row.delta), position_marginal=base['pair_marginal'], marginal_excluding_unit=float(excluded[index]), exclusion_available=bool(available[index]), n_units_in_position=len(full), n_partners=1, n_atomic_cells=int(row.n_contexts), min_contexts_per_partner=int(row.n_contexts), max_contexts_per_partner=int(row.n_contexts), partner_effects_json=dumps({row.unit_identity: row.delta}))
                    item['unit_id'] = fingerprint([item[field] for field in GROUP + ['basis', 'unit_identity']])
                    unit_rows.append(item)
        print(f'{mode}: {len(recurrent_keys)} recurrent perturbation identities; {len(selected)} source contexts', flush=True)
    units = pd.DataFrame(unit_rows)
    assert not units.unit_id.duplicated().any()
    return (units, pd.concat(pair_frames, ignore_index=True), pd.DataFrame(context_rows))

def directed_bridges(units):
    rows = []
    for _, group in units.groupby(MATCH, sort=True):
        if group.screen.nunique() < 2:
            continue
        for source, target in itertools.permutations(group.to_dict('records'), 2):
            if source['screen'] == target['screen']:
                continue
            source_partners = set(json.loads(source['partner_effects_json']))
            target_partners = set(json.loads(target['partner_effects_json']))
            relation = 'within_family' if source['family'] == target['family'] else 'cross_family'
            assert source['family'] == transformation(source['screen'])
            assert target['family'] == transformation(target['screen'])
            n = source['n_partners']
            row = {field: target[field] for field in MATCH}
            row.update(scheme='directed_screen', relation=relation, target_unit_id=target['unit_id'], source_unit_id=source['unit_id'], source_screen=source['screen'], source_family=source['family'], target_screen=target['screen'], target_family=target['family'], observed=target['observed'], zero=0.0, source_pair_marginal=source['pair_marginal'], source_position_marginal=source['position_marginal'], source_marginal_excluding_unit=source['marginal_excluding_unit'], source_conditioned=source['observed'], source_conditioned_shrunk5=(n * source['observed'] + 5 * source['marginal_excluding_unit']) / (n + 5), source_all_available_marginal=source['pair_marginal'], source_yield_sd=source['source_yield_sd'], source_n_partners=source['n_partners'], target_n_partners=target['n_partners'], shared_partner_count=len(source_partners & target_partners), source_partner_union_json=dumps(sorted(source_partners)), target_partner_union_json=dumps(sorted(target_partners)), source_contexts=source['n_condition_contexts'], target_contexts=target['n_condition_contexts'], source_exclusion_available=source['exclusion_available'], source_dose_levels_json=source['dose_levels_json'], target_dose_levels_json=target['dose_levels_json'], n_source_screens=1, n_source_families=1, source_screens_json=dumps([source['screen']]), source_families_json=dumps([source['family']]))
            rows.append(row)
    bridges = pd.DataFrame(rows)
    assert not bridges.duplicated(['source_unit_id', 'target_unit_id']).any()
    return bridges

def pool_other_families(bridges, units):
    rows = []
    catalog_columns = ['mode', 'background_control', 'perturbation_key', 'background_key', 'basis']
    population = units.drop_duplicates(catalog_columns + ['screen'])
    catalog = {key: frame for key, frame in population.groupby(catalog_columns, sort=False)}
    cross = bridges[bridges.relation.eq('cross_family')]
    for identifier, group in cross.groupby('target_unit_id', sort=True):
        target = group.iloc[0]
        source_families = sorted(group.source_family.unique())
        assert target.target_family not in source_families
        partners = set((partner for text in group.source_partner_union_json for partner in json.loads(text)))
        target_partners = set(json.loads(target.target_partner_union_json))
        row = {field: target[field] for field in MATCH}
        row.update(scheme='pooled_other_families', relation='cross_family', target_unit_id=identifier, source_unit_id='pooled', source_screen='pooled', source_family='ALL_OTHER_FAMILIES', target_screen=target.target_screen, target_family=target.target_family, observed=target.observed, source_yield_sd=pooled_source_scale(group), source_n_partners=len(partners), target_n_partners=target.target_n_partners, shared_partner_count=len(partners & target_partners), source_partner_union_json=dumps(sorted(partners)), target_partner_union_json=target.target_partner_union_json, source_contexts=int(group.source_contexts.sum()), target_contexts=target.target_contexts, source_exclusion_available=bool(group.source_exclusion_available.all()), source_dose_levels_json=dumps(sorted({dose for text in group.source_dose_levels_json for dose in json.loads(text)})), target_dose_levels_json=target.target_dose_levels_json, n_source_screens=group.source_screen.nunique(), n_source_families=len(source_families), source_screens_json=dumps(sorted(group.source_screen.unique())), source_families_json=dumps(source_families))
        for model in MODELS:
            row[model] = family_source_mean(group, model)
        source_pool = catalog[tuple((target[field] for field in catalog_columns))]
        source_pool = source_pool[~source_pool.family.eq(target.target_family)]
        assert len(source_pool) and (not source_pool.family.eq(target.target_family).any())
        row['source_all_available_marginal'] = float(source_pool.groupby(['family', 'screen']).pair_marginal.mean().groupby('family').mean().mean())
        rows.append(row)
    pooled = pd.DataFrame(rows)
    assert not pooled.target_unit_id.duplicated().any()
    return pooled

def molecular_errors(frame, model):
    keys = ['source_screen', 'target_screen', 'source_family', 'target_family', 'unit_identity']
    data = frame[keys].copy()
    error = frame[model] - frame.observed
    data['mse'] = error ** 2
    data['mae'] = abs(error)
    data['standardized_mse'] = (error / frame.source_yield_sd) ** 2
    data['sign_agreement'] = (np.sign(frame[model]) == np.sign(frame.observed)).astype(float)
    return data.groupby(keys, as_index=False)[['mse', 'mae', 'standardized_mse', 'sign_agreement']].mean()

def error_summary(frame, model):
    if frame.empty:
        return dict(status='zero_eligible', n_prediction_units=0, n_molecular_units=0, n_unique_molecules=0, n_screen_pairs=0, n_family_pairs=0, n_target_screens=0, n_target_families=0, family_macro_rmse_pp=np.nan, family_macro_mse=np.nan, family_macro_mae_pp=np.nan, source_standardized_rmse=np.nan, family_macro_sign_agreement=np.nan)
    molecule = molecular_errors(frame, model)
    numeric = ['mse', 'mae', 'standardized_mse', 'sign_agreement']
    screen = molecule.groupby(['source_screen', 'target_screen', 'source_family', 'target_family'], as_index=False)[numeric].mean()
    family = screen.groupby(['source_family', 'target_family'], as_index=False)[numeric].mean()
    return dict(status='scored', n_prediction_units=len(frame), n_molecular_units=len(molecule), n_unique_molecules=frame.unit_identity.nunique(), n_screen_pairs=len(screen), n_family_pairs=len(family), n_target_screens=frame.target_screen.nunique(), n_target_families=frame.target_family.nunique(), family_macro_rmse_pp=100 * np.sqrt(family.mse.mean()), family_macro_mse=family.mse.mean(), family_macro_mae_pp=100 * family.mae.mean(), source_standardized_rmse=np.sqrt(family.standardized_mse.mean()), family_macro_sign_agreement=family.sign_agreement.mean())

def score_all(predictions):
    rows, screen_rows, unit_rows = ([], [], [])
    for scheme in ('directed_screen', 'pooled_other_families'):
        relations = ['within_family', 'cross_family'] if scheme == 'directed_screen' else ['cross_family']
        for mode, control, basis, relation in itertools.product(MODES, ('background_free', 'exact_background'), BASES, relations):
            initial = predictions[predictions.scheme.eq(scheme) & predictions['mode'].eq(mode) & predictions.background_control.eq(control) & predictions.basis.eq(basis) & predictions.relation.eq(relation)]
            for threshold in [1] if basis == 'full_pair' else [1, 3, 5]:
                for overlap in ['all'] if basis == 'full_pair' else ['all', 'no_shared_partner']:
                    keep = (initial.source_n_partners >= threshold) & (initial.target_n_partners >= threshold)
                    if overlap == 'no_shared_partner':
                        keep &= initial.shared_partner_count.eq(0)
                    selected = initial[keep]
                    prefix = dict(scheme=scheme, mode=mode, background_control=control, basis=basis, relation=relation, min_partners_each=threshold, partner_overlap=overlap)
                    for model in MODELS:
                        rows.append(dict(**prefix, model=model, **error_summary(selected, model)))
                        if len(selected):
                            units = molecular_errors(selected, model)
                            for row in units.to_dict('records'):
                                unit_rows.append(dict(**prefix, model=model, **row))
                            numeric = ['mse', 'mae', 'standardized_mse', 'sign_agreement']
                            by_screen = units.groupby(['source_screen', 'target_screen', 'source_family', 'target_family'], as_index=False)[numeric].mean()
                            counts = units.groupby(['source_screen', 'target_screen']).size()
                            for row in by_screen.to_dict('records'):
                                row['n_molecular_units'] = int(counts.loc[row['source_screen'], row['target_screen']])
                                screen_rows.append(dict(**prefix, model=model, **row))
    scores = pd.DataFrame(rows)
    reference = scores[scores.model.eq('source_marginal_excluding_unit')].drop(columns='model')
    reference = reference[[column for column in scores.columns if column in ['scheme', 'mode', 'background_control', 'basis', 'relation', 'min_partners_each', 'partner_overlap', 'family_macro_mse']]]
    reference = reference.rename(columns={'family_macro_mse': 'exclusion_marginal_mse'})
    keys = [column for column in reference if column != 'exclusion_marginal_mse']
    scores = scores.merge(reference, on=keys, validate='many_to_one')
    scores['gain_vs_exclusion_marginal'] = 1 - scores.family_macro_mse / scores.exclusion_marginal_mse
    return (scores, pd.DataFrame(screen_rows), pd.DataFrame(unit_rows))

def support_registry(contrasts, units, bridges, inventory):
    rows = []
    for mode, key_column in MODES.items():
        for control, relation in itertools.product(('background_free', 'exact_background'), ('within_family', 'cross_family')):
            metadata = contrasts.copy()
            metadata['bridge_key'] = metadata[key_column]
            metadata['matched_background'] = '*' if control == 'background_free' else metadata.background_key
            context_sources = {key: group[['screen', 'family']].drop_duplicates().to_dict('records') for key, group in metadata.groupby(['bridge_key', 'matched_background'], sort=False)}
            for inv in inventory.to_dict('records'):
                screen = inv['screen']
                local = metadata[metadata.screen.eq(screen)]
                supported_context_ids = []
                for row in local.itertuples(index=False):
                    possible = context_sources[row.bridge_key, row.matched_background]
                    if any((source['screen'] != screen and source_relation_allowed(source['family'], row.family, relation) for source in possible)):
                        supported_context_ids.append(row.contrast_id)
                for basis in BASES:
                    candidate = units[units['mode'].eq(mode) & units.background_control.eq(control) & units.screen.eq(screen) & units.basis.eq(basis)]
                    eligible = bridges[bridges['mode'].eq(mode) & bridges.background_control.eq(control) & bridges.relation.eq(relation) & bridges.target_screen.eq(screen) & bridges.basis.eq(basis)]
                    rows.append(dict(screen=screen, family=inv['family'], mode=mode, background_control=control, relation=relation, basis=basis, analysis_substrate_pairs=inv['n_substrate_pairs'], clean_conditions=inv['clean_conditions'], registered_perturbation_contexts=len(local), contexts_with_allowed_source_perturbation=len(supported_context_ids), candidate_units_in_recurrent_perturbations=len(candidate), supported_target_units=eligible.target_unit_id.nunique(), directed_source_target_units=len(eligible), supported_target_units_min3=eligible[(eligible.source_n_partners >= 3) & (eligible.target_n_partners >= 3)].target_unit_id.nunique() if basis != 'full_pair' else np.nan, supported_target_units_min5=eligible[(eligible.source_n_partners >= 5) & (eligible.target_n_partners >= 5)].target_unit_id.nunique() if basis != 'full_pair' else np.nan, supported_with_no_shared_partner=eligible[eligible.shared_partner_count.eq(0)].target_unit_id.nunique() if basis != 'full_pair' else np.nan, status='supported' if len(eligible) else 'zero_eligible'))
    return pd.DataFrame(rows)

def plot_results(scores, support, predictions, screen_scores, inventory):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
    a, b, c, d = axes.flat
    labels = ['Shared perturbation', 'Same-position molecule 1', 'Same-position molecule 2', 'Full substrate pair', 'Molecule + fixed background', 'Full pair + fixed background']
    counts = {}
    for mode in MODES:
        part = support[support['mode'].eq(mode) & support.relation.eq('cross_family')]
        free, fixed = (part[part.background_control.eq('background_free')], part[part.background_control.eq('exact_background')])
        counts[mode] = [free[free.contexts_with_allowed_source_perturbation.gt(0)].screen.nunique(), free[free.basis.eq('position_1') & free.supported_target_units.gt(0)].screen.nunique(), free[free.basis.eq('position_2') & free.supported_target_units.gt(0)].screen.nunique(), free[free.basis.eq('full_pair') & free.supported_target_units.gt(0)].screen.nunique(), fixed[fixed.basis.ne('full_pair') & fixed.supported_target_units.gt(0)].screen.nunique(), fixed[fixed.basis.eq('full_pair') & fixed.supported_target_units.gt(0)].screen.nunique()]
    for offset, (mode, color) in zip([-0.18, 0.18], [('strict', '#287e75'), ('dose_relaxed', '#bc7556')]):
        bars = a.barh(np.arange(len(labels)) + offset, counts[mode], height=0.34, color=color, label=mode.replace('_', ' '))
        for bar, value in zip(bars, counts[mode]):
            a.text(value + 0.15, bar.get_y() + bar.get_height() / 2, str(value), va='center', fontsize=8)
    a.set_yticks(range(len(labels)), labels, fontsize=9)
    a.invert_yaxis()
    a.set(xlim=(0, 20), xlabel='Target screens with cross-family support / 18', title='A  Progressively controlled support, including zero coverage')
    a.legend(frameon=False, fontsize=9, loc='lower right')
    display_models = ['zero', 'source_all_available_marginal', 'source_marginal_excluding_unit', 'source_conditioned', 'source_conditioned_shrunk5']
    column_keys = list(itertools.product(MODES, ['position_1', 'position_2'], [1, 3]))
    heat = np.full((len(display_models), len(column_keys)), np.nan)
    pool = scores[scores.scheme.eq('pooled_other_families') & scores.background_control.eq('background_free') & scores.partner_overlap.eq('all')]
    for i, model in enumerate(display_models):
        for j, (mode, basis, threshold) in enumerate(column_keys):
            row = pool[pool.model.eq(model) & pool['mode'].eq(mode) & pool.basis.eq(basis) & pool.min_partners_each.eq(threshold)]
            if len(row):
                heat[i, j] = row.family_macro_rmse_pp.iloc[0]
    im = b.imshow(heat, cmap='YlOrRd', aspect='auto')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                b.text(j, i, f'{heat[i, j]:.1f}', ha='center', va='center', fontsize=8)
    b.set_yticks(range(len(display_models)), ['Zero', 'All-source marginal', 'Marginal excluding molecule', 'Same molecule', 'Same molecule, shrink 5'], fontsize=8)
    column_labels = []
    for mode, basis, threshold in column_keys:
        part = pool[pool['mode'].eq(mode) & pool.basis.eq(basis) & pool.min_partners_each.eq(threshold) & pool.model.eq('zero')]
        count = int(part.n_prediction_units.iloc[0])
        molecules = int(part.n_unique_molecules.iloc[0])
        column_labels.append(f"{('Strict' if mode == 'strict' else 'Dose relaxed')}\nPos {basis[-1]}, n>={threshold}\nN={count}; molecules={molecules}")
    b.set_xticks(range(len(column_keys)), column_labels, rotation=35, ha='right', fontsize=8)
    b.set_title('B  Whole-family exclusion: target conditional-mean RMSE')
    fig.colorbar(im, ax=b, shrink=0.65, label='Family-macro RMSE (yield pp)')
    chosen = screen_scores[screen_scores.scheme.eq('pooled_other_families') & screen_scores['mode'].eq('strict') & screen_scores.background_control.eq('background_free') & screen_scores.min_partners_each.eq(1) & screen_scores.partner_overlap.eq('all')]
    order = list(inventory.screen)
    gains = np.full((len(order), 2), np.nan)
    for i, screen in enumerate(order):
        for j, basis in enumerate(['position_1', 'position_2']):
            part = chosen[chosen.target_screen.eq(screen) & chosen.basis.eq(basis)].set_index('model')
            if 'source_conditioned' in part.index and part.loc['source_marginal_excluding_unit', 'mse'] > 1e-16:
                gains[i, j] = 1 - part.loc['source_conditioned', 'mse'] / part.loc['source_marginal_excluding_unit', 'mse']
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#dddddd')
    im2 = c.imshow(np.clip(gains, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    c.set_yticks(range(len(order)), [s.replace('Buchwald-Hartwig(CMC)', 'BH(CMC)').replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O competitive').replace('|', ' #') for s in order], fontsize=8)
    c.set_xticks([0, 1], ['Position 1', 'Position 2'])
    for i in range(len(order)):
        for j in range(2):
            if np.isfinite(gains[i, j]):
                c.text(j, i, f'{100 * gains[i, j]:.0f}%', ha='center', va='center', fontsize=7)
    c.set_title('C  Same-molecule gain over source marginal excluding it')
    fig.colorbar(im2, ax=c, shrink=0.7, label='1 - conditional SSE / source marginal SSE; color clipped')
    for mode, color in [('strict', '#287e75'), ('dose_relaxed', '#bc7556')]:
        for basis, marker in [('position_1', 'o'), ('position_2', 's')]:
            counts_n = []
            for threshold in [1, 3, 5]:
                part = pool[pool['mode'].eq(mode) & pool.basis.eq(basis) & pool.min_partners_each.eq(threshold) & pool.model.eq('zero')]
                counts_n.append(int(part.n_prediction_units.iloc[0]))
            d.plot([1, 3, 5], counts_n, marker=marker, color=color, linestyle='-' if basis == 'position_1' else '--', label=f"{mode.replace('_', ' ')}; position {basis[-1]}")
            for threshold, value in zip([1, 3, 5], counts_n):
                d.annotate(str(value), (threshold, value), xytext=(3, 4), textcoords='offset points', fontsize=8)
    d.set_yscale('symlog', linthresh=1)
    d.set(xlabel='Minimum distinct partners in source union and target', ylabel='Scored target molecular summaries (symlog; linear at 0-1)', title='D  Molecular support remains sparse under stronger partner control', xticks=[1, 3, 5], ylim=(-0.2, 2000))
    d.legend(frameon=False, fontsize=8)
    path = HERE / 'figures/16_controlled_molecular_component_bridges.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def main():
    contrasts, inventory, atomic, scales, hashes = load_atomic_effects()
    units, pairs, contexts = build_estimands(contrasts, atomic, scales)
    directed = directed_bridges(units)
    pooled = pool_other_families(directed, units)
    predictions = pd.concat([directed, pooled], ignore_index=True)
    scores, screen_scores, molecular_scores = score_all(predictions)
    support = support_registry(contrasts, units, directed, inventory)
    assert len(support) == 19 * 2 * 2 * 2 * 3
    assert not predictions[predictions.scheme.eq('pooled_other_families')].apply(lambda row: row.target_family in json.loads(row.source_families_json), axis=1).any()
    assert predictions.source_yield_sd.gt(0).all()
    joint_full_pair_units = len(directed[directed.background_control.eq('exact_background') & directed.basis.eq('full_pair')])
    legacy = pd.read_csv(HERE / 'tables/cross_library_components_matching_substrate_comparisons.csv', keep_default_na=False)
    assert legacy[legacy.matching_basis.eq('ordered_pair')].substrate_identity.nunique() == directed[directed['mode'].eq('strict') & directed.basis.eq('full_pair')].unit_identity.nunique()
    outputs = {'atomic_effects.csv.gz': atomic, 'partner_balanced_pairs.csv.gz': pairs, 'conditional_estimands.csv.gz': units, 'perturbation_background_registry.csv': contexts, 'directed_bridge_registry.csv': directed, 'family_excluded_predictions.csv': pooled, 'support_hierarchy.csv': support, 'scores.csv': scores, 'screen_pair_scores.csv': screen_scores, 'molecular_unit_scores.csv.gz': molecular_scores}
    output_paths = []
    for name, frame in outputs.items():
        path = HERE / 'tables' / (PREFIX + name)
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        output_paths.append(path)
    figure = plot_results(scores, support, predictions, screen_scores, inventory)
    assert hashes == {path: digest(path) for path in hashes}, 'An input changed'
    report = HERE / 'controlled_cross_library_bridges.py'
    if report.exists():
        output_paths.append(report)
    manifest = dict(date='2026-10-16', scope=f'All{len(inventory)} existing analysis screens, no raw mutation, controlled molecular/component bridge exploration', input_hashes=hashes, code_hashes={str(HERE / name): digest(HERE / name) for name in ['controlled_cross_library_bridges.py', 'test_controlled_cross_library_bridges.py']}, output_hashes={str(path.relative_to(HERE)): digest(path) for path in output_paths + [figure]}, counts=dict(screens=len(inventory), families=14, analysis_substrate_pairs=int(inventory.n_substrate_pairs.sum()), atomic_paired_cells=len(atomic), conditional_estimands=len(units), directed_bridge_units=len(directed), family_excluded_target_units=len(pooled), support_inventory_rows=len(support), joint_full_pair_background_units=joint_full_pair_units, unsupported_inventory_rows=int(support.status.eq('zero_eligible').sum())), definitions=dict(perturbation='Strict uses same exact oriented identities and equivalent across screens. Dose-relaxed keeps each within-screen intervention same-dose but allows different equivalent across screens; mixed doses within a screen are recorded. Solvent replacement is separate in registered key.', background="background_free averages observed backgrounds. exact_background requires identical remaining ingredient/dose multiset, solvent when fixed, temperature and time; in dose-relaxed mode the replaced reagent's dose can still differ.", molecular_positions='Canonical isomeric SMILES and same registry sub_1 or sub_2 position. Cross-transformation positions are operational and are not asserted to be equivalent reaction roles. Position bases are scored separately; full pairs have one basis only.', label='Within screen/key/background: average delta over observed contexts for each unique full substrate pair, then equal weight distinct partners for each fixed molecule. Fixed conditional population summary; target outcomes define labels/scoring only.', source_marginals='Pair marginal gives full pairs equal weight. Position marginal gives source molecules equal weight after partner balancing. Exclusion marginal removes the current shared molecule (or full pair) from the position mean; a single-unit fallback is flagged.', shrinkage='Fixed source n_partners/(n_partners+5) weight on conditional source mean, remainder on source marginal excluding unit. No target tuning or fitting.', source_pooling='For family-excluded predictions all target-family screens are excluded. First source screen means, then equal source families. Shared partners may recur across screens; unique partner union and screen/family counts are retained. All-available marginal can additionally use other-family source screens sharing perturbation/background without requiring the molecule.', scoring='Average squared errors across perturbation/background predictions for each molecule within directed screen pair; average molecular units within screen pair; average screen pairs within directed family pair; equal directed family pairs. Pooled-other-family scores reduce to equal target families after equal screens and molecules. No independent-context significance or confidence interval.', support=f'All{len(inventory)} screens retained in support inventory. At least1,3,5 distinct partners each side are prespecified descriptive sensitivities; pooled source uses unique partner union. Full-pair basis only uses threshold1. Zero eligible scores are NaN with explicit status, never zero error.', partner_overlap='All main comparisons retained. no_shared_partner sensitivity excludes rows with any common other partner between source union and target; full pairs are not duplicated into both position bases.', standardization='Error divided by source-screen overall clean-matrix yield SD; pooled sources use square root of equal-family/equal-screen mean source variance. No target scale, target activity or target calibration enters prediction or error scale.'), limits=['Observational population summaries confound remaining partner, background, dose and library sampling; no mechanistic or causal transport claim.', 'Within-family comparisons occur only where multiple screens exist, chiefly BH/CMC; these screens are not independent chemical mechanisms.', 'Exact joint full-substrate/background support is absent; unsupported is distinct from a negative predictive result.', 'Source outcomes are historical supervised information; no target-family outcomes enter source predictions. Targets are retrospective population labels, not prospective individual reaction forecasts.', 'No replicate noise floor, batch adjustment, new laboratory or prospective validation is supplied.'], verification=dict(all19_accounted=True, source_target_screens_disjoint=True, pooled_entire_target_family_excluded=True, paired_finite_only=True, previous_pair_match_inventory_agrees=True, full_pair_not_role_duplicated=True, joint_full_pair_background_support_explicit=True, all_inputs_unchanged=True), environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, blas_threads='library_default_unrestricted', invoked_with='python -B'))
    (HERE / 'data/controlled_bridge_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts'], ensure_ascii=False), flush=True)
    show = scores[scores.scheme.eq('pooled_other_families') & scores.background_control.eq('background_free') & scores.min_partners_each.eq(1) & scores.partner_overlap.eq('all') & scores.model.isin(['zero', 'source_all_available_marginal', 'source_marginal_excluding_unit', 'source_conditioned', 'source_conditioned_shrunk5'])]
    print(show[['mode', 'basis', 'model', 'n_prediction_units', 'n_target_screens', 'family_macro_rmse_pp', 'gain_vs_exclusion_marginal']].to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
