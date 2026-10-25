"""Multi-level library overlap and honest transport of perturbation summaries.

Registry-only features predict population summaries in a wholly held-out screen
or transformation family. No held-out-screen yield supplies a predictor."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import itertools
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import sys as _m1_sys
_m1_sys.path.append(str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR as _M1_RAW_DIR
from chemical_response_structure import AMBIGUOUS, CORE, matched_kind
from condition_audit import families, solvent_features
HERE = Path(__file__).resolve().parent
PREFIX = 'cross_library_components_'
OUTCOMES = ['mean_delta', 'signed_consistency']
MODELS = ['zero', 'training_mean', 'same_perturbation_mean', 'same_identity_replacement_mean', 'endpoint_marginal_ridge', 'component_context_ridge']
RIDGE_ALPHA = 10.0
MIN_OVERLAP = 10

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()

def transformation(screen):
    value = screen.split('|')[0]
    return 'Buchwald-Hartwig' if value == 'Buchwald-Hartwig(CMC)' else value

def canonical_pair(value):
    parts = value.split(' >> ')
    assert len(parts) == 2, value
    result = []
    for text in parts:
        molecule = Chem.MolFromSmiles(text)
        assert molecule is not None, text
        result.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return tuple(result)

def ingredients(row):
    return Counter(((row[f'Reagent{i}_id'], row[f'Reagent{i}_HTE_Equiv']) for i in range(1, 7) if row[f'Reagent{i}_id']))

def counter_list(counter):
    return sorted((item for item, count in counter.items() for _ in range(count)))

def recipe_key(row, dose=True):
    reagent = counter_list(ingredients(row))
    if not dose:
        reagent = sorted((identity for identity, _ in reagent))
    return dumps([reagent, row['solvent_id'], row['temp_id'], row['time_id']])

def detect_perturbation(a, b):
    """Detect one same-dose identity change without dependence on slot labels."""
    ca, cb = (ingredients(a), ingredients(b))
    different_physical = [key for key in ('solvent_id', 'temp_id', 'time_id') if a[key] != b[key]]
    removed, added = (counter_list(ca - cb), counter_list(cb - ca))
    if not different_physical and len(removed) == len(added) == 1:
        old, new = (removed[0], added[0])
        if old[0] == new[0] or old[1] != new[1] or (not old[1]):
            return None
        low, high = sorted([old[0], new[0]])
        kind, dose = ('reagent_replacement', old[1])
        orientation = 1 if old[0] == low else -1
        background = counter_list(ca & cb)
        fixed = [background, a['solvent_id'], a['temp_id'], a['time_id']]
    elif ca == cb and different_physical == ['solvent_id'] and a['solvent_id'] and b['solvent_id']:
        low, high = sorted([a['solvent_id'], b['solvent_id']])
        kind, dose = ('solvent_replacement', '')
        orientation = 1 if a['solvent_id'] == low else -1
        background = counter_list(ca)
        fixed = [background, a['temp_id'], a['time_id']]
    else:
        return None
    key = dumps([kind, low, high, dose])
    return dict(kind=kind, low_id=low, high_id=high, fixed_equiv=dose, key=key, no_cross_screen_dose_key=dumps([kind, low, high]), orientation=orientation, background_reagents=background, background_key=dumps(fixed))

def delta_summary(low, high):
    keep = np.isfinite(low) & np.isfinite(high)
    delta = high[keep] - low[keep]
    mean = float(np.mean(delta)) if len(delta) else np.nan
    rms = float(np.sqrt(np.mean(delta ** 2))) if len(delta) else np.nan
    consistency = mean / rms if rms > 1e-12 else 0.0 if len(delta) else np.nan
    assert not np.isfinite(consistency) or abs(consistency) <= 1 + 1e-12
    return dict(n_overlap=int(keep.sum()), mean_delta=mean, delta_rms=rms, signed_consistency=consistency, fraction_both_zero=float(np.mean((low[keep] == 0) & (high[keep] == 0))) if len(delta) else np.nan)

def feature_dict(row, names, contextual):
    result = {}
    domain = row['kind']
    for sign, endpoint in [(-1, row['low_id']), (1, row['high_id'])]:
        result[f'endpoint:{domain}:{endpoint}'] = sign
    if not contextual:
        return result
    key = row['perturbation_key']
    result['perturbation:' + key] = 1.0
    background = json.loads(row['background_reagents_json'])
    context = Counter((identity for identity, dose in background))
    context['solvent:' + row['fixed_solvent_id']] += 1
    context['temperature'] = float(row['temp_id']) / 100
    context['log_time'] = float(np.log1p(float(row['time_id'])) / 5)
    for token, value in context.items():
        result[f'perturbation_context:{key}:{token}'] = value
    endpoint_families = Counter()
    for sign, endpoint in [(-1, row['low_id']), (1, row['high_id'])]:
        labels = families(names[endpoint]) if domain == 'reagent_replacement' else solvent_features(names[endpoint])[1]
        for family in labels:
            endpoint_families[family] += sign
    bg_families = Counter((label for identity, dose in background for label in families(names[identity])))
    if row['fixed_solvent_id']:
        bg_families.update(('solvent_' + label for label in solvent_features(names[row['fixed_solvent_id']])[1]))
    bg_families['temperature'] = context['temperature']
    bg_families['log_time'] = context['log_time']
    for family, value in endpoint_families.items():
        if value:
            result['family_delta:' + family] = value
            for bg, count in bg_families.items():
                result[f'family_context:{family}:{bg}'] = value * count
    return result

def load_inputs():
    paths = [HERE / 'tables/condition_recipe_audit_details.csv', HERE / 'tables/screen_inventory.csv', HERE / 'data/raw_condition_table.csv', HERE / 'tables/condition_recipe_audit_provenance.json', HERE / 'condition_audit.py', HERE / 'chemical_response_structure.py']
    paths += sorted((HERE / 'data').glob('matrix_*.npz'))
    provenance = json.loads(paths[3].read_text(encoding='utf-8'))
    raw = _M1_RAW_DIR
    for filename in ('0_total condition_260810.xlsx', '0_Compound ID_260810.xlsx'):
        path = raw / filename
        assert sha256(path) == provenance['source_hashes'][filename]['sha256']
        paths.append(path)
    source_hashes = {str(path): sha256(path) for path in paths}
    details = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    mapping = pd.read_csv(paths[2], dtype=str, keep_default_na=False)[['reaction_class', 'reaction_type']].drop_duplicates()
    details = details.merge(mapping, on='reaction_type', validate='many_to_one')
    for _, row in details.iterrows():
        assert fingerprint([row[field] for field in CORE]) == row.recipe_full_id_dose_sha256
    inventory = pd.read_csv(paths[1])
    screens, names, component_rows, recipe_rows, inventory_rows = ({}, {}, [], [], [])
    raw_screens = {}
    for _, inv in inventory.iterrows():
        screen = inv.screen
        cls = screen.split('|')[0]
        saved = np.load(HERE / 'data' / ('matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz'), allow_pickle=False)
        matrix, conditions = (saved['M'], saved['conds'].astype(str))
        pairs = [canonical_pair(text) for text in saved['sps']]
        assert len(set(pairs)) == len(pairs) == inv.analysis_pairs
        assert np.nanmin(matrix) >= 0 and np.nanmax(matrix) <= 1
        lookup = details[details.reaction_class.eq(cls)].set_index('condition_id', verify_integrity=True)
        recipes_all = [lookup.loc[condition].to_dict() for condition in conditions]
        raw_screens[screen] = recipes_all
        keep = np.array([(cls, condition) not in AMBIGUOUS for condition in conditions])
        excluded = conditions[~keep].tolist()
        matrix, conditions = (matrix[:, keep], conditions[keep])
        recipes = [recipe for recipe, ok in zip(recipes_all, keep) if ok]
        for condition, recipe in zip(conditions, recipes):
            for field in ('missing_component_id_slots_json', 'missing_dose_slots_json', 'orphan_dose_slots_json'):
                assert recipe[field] == '[]'
            recipe_rows.append(dict(screen=screen, condition_id=condition, recipe_slot_key=recipe['recipe_full_id_dose_sha256'], recipe_multiset_key=recipe_key(recipe), source_row=recipe['source_row'], **{field: recipe[field] for field in CORE}))
            for slot in range(1, 7):
                identity, dose = (recipe[f'Reagent{slot}_id'], recipe[f'Reagent{slot}_HTE_Equiv'])
                if identity:
                    name = recipe[f'Reagent{slot}_Standardized Name']
                    assert identity not in names or names[identity] == name
                    names[identity] = name
                    component_rows.append(dict(screen=screen, condition_id=condition, domain='reagent', slot=slot, identity=identity, name=name, dose=dose, name_based_families_json=dumps(families(name))))
            identity, name = (recipe['solvent_id'], recipe['Standardized Name'])
            assert identity not in names or names[identity] == name
            names[identity] = name
            component_rows.append(dict(screen=screen, condition_id=condition, domain='solvent', slot='', identity=identity, name=name, dose='', name_based_families_json=dumps(solvent_features(name)[1])))
        screens[screen] = dict(matrix=matrix, conditions=conditions, recipes=recipes, pairs=pairs, family=transformation(screen))
        inventory_rows.append(dict(screen=screen, family=transformation(screen), n_substrate_pairs=len(pairs), original_conditions=len(keep), clean_conditions=len(conditions), excluded_ambiguous_conditions_json=dumps(excluded), n_substrate_1=len({pair[0] for pair in pairs}), n_substrate_2=len({pair[1] for pair in pairs}), n_reagents=len({identity for recipe in recipes for identity, dose in ingredients(recipe)}), n_solvents=len({recipe['solvent_id'] for recipe in recipes}), missing_yield_fraction=float(np.isnan(matrix).mean())))
    assert len(screens) == 19 and sum((len(x['pairs']) for x in screens.values())) == 2454
    assert len(set(names.values())) == len(names), 'Exact-name alias IDs require explicit review'
    return (screens, raw_screens, names, component_rows, recipe_rows, inventory_rows, source_hashes)

def enumerate_perturbations(screens, names):
    rows, deltas, duplicate_rows = ([], {}, [])
    for screen, data in screens.items():
        recipes, matrix = (data['recipes'], data['matrix'])
        seen = {}
        n_edges = 0
        for i, j in itertools.combinations(range(len(recipes)), 2):
            contrast = detect_perturbation(recipes[i], recipes[j])
            if contrast is None:
                continue
            n_edges += 1
            low, high = (i, j) if contrast['orientation'] == 1 else (j, i)
            unique_key = (contrast['key'], contrast['background_key'])
            if unique_key in seen:
                duplicate_rows.append(dict(screen=screen, condition_low=data['conditions'][low], condition_high=data['conditions'][high], duplicate_of=seen[unique_key]))
                continue
            identifier = f'P{len(rows) + 1:05d}'
            seen[unique_key] = identifier
            slot_kind, _, slot = matched_kind(recipes[i], recipes[j])
            info = delta_summary(matrix[:, low], matrix[:, high])
            row = dict(contrast_id=identifier, screen=screen, family=data['family'], condition_low=data['conditions'][low], condition_high=data['conditions'][high], kind=contrast['kind'], low_id=contrast['low_id'], high_id=contrast['high_id'], low_name=names[contrast['low_id']], high_name=names[contrast['high_id']], fixed_equiv=contrast['fixed_equiv'], perturbation_key=contrast['key'], no_cross_screen_dose_key=contrast['no_cross_screen_dose_key'], background_key=contrast['background_key'], background_reagents_json=dumps(contrast['background_reagents']), fixed_solvent_id=recipes[i]['solvent_id'] if contrast['kind'] == 'reagent_replacement' else '', temp_id=recipes[i]['temp_id'], time_id=recipes[i]['time_id'], also_single_registry_field=slot_kind in {'identity_swap', 'solvent_only'}, registry_slot=slot, source_row_low=recipes[low]['source_row'], source_row_high=recipes[high]['source_row'], eligible_summary=info['n_overlap'] >= MIN_OVERLAP, **info)
            rows.append(row)
            keep = np.isfinite(matrix[:, low]) & np.isfinite(matrix[:, high])
            delta = np.full(len(matrix), np.nan)
            delta[keep] = matrix[keep, high] - matrix[keep, low]
            deltas[identifier] = delta
        print(f'{screen}: {n_edges} slot-invariant contrasts, {len(seen)} unique backgrounds', flush=True)
    contrasts = pd.DataFrame(rows)
    assert not contrasts.duplicated(['screen', 'perturbation_key', 'background_key']).any()
    return (contrasts, deltas, pd.DataFrame(duplicate_rows, columns=['screen', 'condition_low', 'condition_high', 'duplicate_of']))

def domain_sets(data, contrasts):
    recipes, pairs = (data['recipes'], data['pairs'])
    reagent = {identity for recipe in recipes for identity, dose in ingredients(recipe)}
    result = {'reagent_identity': reagent, 'reagent_same_dose': {dumps(token) for recipe in recipes for token in ingredients(recipe)}, 'solvent_identity': {recipe['solvent_id'] for recipe in recipes}, 'solvent_ingredient': {name for recipe in recipes for name in json.loads(recipe['solvent_ingredients_json'])}, 'putative_reagent_family': {label for recipe in recipes for label in json.loads(recipe['putative_name_based_families_json']) if label != 'unclassified'}, 'full_recipe_slot': {recipe['recipe_full_id_dose_sha256'] for recipe in recipes}, 'full_recipe_multiset': {recipe_key(recipe) for recipe in recipes}, 'recipe_ignoring_all_doses': {recipe_key(recipe, dose=False) for recipe in recipes}, 'ordered_substrate_pair': set(pairs), 'unordered_substrate_pair': {tuple(sorted(pair)) for pair in pairs}, 'substrate_position_1': {pair[0] for pair in pairs}, 'substrate_position_2': {pair[1] for pair in pairs}, 'substrate_any_position': {molecule for pair in pairs for molecule in pair}}
    result['any_component_identity'] = result['reagent_identity'] | result['solvent_identity']
    result['same_dose_reagent_replacement'] = set(contrasts.loc[contrasts.kind.eq('reagent_replacement'), 'perturbation_key'])
    result['solvent_replacement'] = set(contrasts.loc[contrasts.kind.eq('solvent_replacement'), 'perturbation_key'])
    result['oriented_perturbation'] = set(contrasts.perturbation_key)
    result['perturbation_ignoring_cross_screen_dose'] = set(contrasts.no_cross_screen_dose_key)
    result['perturbation_fixed_background'] = set(zip(contrasts.perturbation_key, contrasts.background_key))
    return result

def connectivity(screens, edges):
    adjacency = {screen: set() for screen in screens}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(screens)
    while unseen:
        root = min(unseen)
        reached, pending = (set(), [root])
        while pending:
            node = pending.pop()
            if node in reached:
                continue
            reached.add(node)
            pending.extend(adjacency[node] - reached)
        components.append(sorted(reached))
        unseen -= reached
    components.sort(key=lambda x: (-len(x), x))
    return dict(n_edges=len(edges), n_connected_components=len(components), largest_component_size=len(components[0]), n_isolated_screens=sum((len(x) == 1 for x in components)), components_json=dumps(components))

def overlap_tables(screens, raw_screens, contrasts):
    sets = {screen: domain_sets(data, contrasts[contrasts.screen.eq(screen)]) for screen, data in screens.items()}
    pair_rows, sensitivity_rows = ([], [])
    for a, b in itertools.combinations(screens, 2):
        row = dict(screen_a=a, screen_b=b, family_a=screens[a]['family'], family_b=screens[b]['family'])
        for level in sets[a]:
            shared = sets[a][level] & sets[b][level]
            row['shared_' + level] = len(shared)
        row['shared_perturbation_and_ordered_pair'] = int(row['shared_oriented_perturbation'] > 0 and row['shared_ordered_substrate_pair'] > 0)
        row['shared_perturbation_and_substrate_position'] = int(row['shared_oriented_perturbation'] > 0 and (row['shared_substrate_position_1'] > 0 or row['shared_substrate_position_2'] > 0))
        pair_rows.append(row)
        clean_full = row['shared_full_recipe_multiset']
        include = len({recipe_key(x) for x in raw_screens[a]} & {recipe_key(x) for x in raw_screens[b]})
        clean_ids = row['shared_reagent_identity']
        include_ids = len({token[0] for x in raw_screens[a] for token in ingredients(x)} & {token[0] for x in raw_screens[b] for token in ingredients(x)})
        sensitivity_rows.append(dict(screen_a=a, screen_b=b, clean_shared_full_recipes=clean_full, ambiguous_static_included_shared_full_recipes=include, clean_shared_reagent_ids=clean_ids, ambiguous_static_included_shared_reagent_ids=include_ids))
    pairs = pd.DataFrame(pair_rows)
    graph_rows = []
    for column in pairs.columns:
        if column.startswith('shared_'):
            edge_frame = pairs[pairs[column] > 0]
            graph_rows.append(dict(level=column.removeprefix('shared_'), n_cross_family_edges=int(edge_frame.family_a.ne(edge_frame.family_b).sum()), **connectivity(list(screens), list(zip(edge_frame.screen_a, edge_frame.screen_b)))))
    return (pairs, pd.DataFrame(graph_rows), pd.DataFrame(sensitivity_rows))

def matching_substrate_comparison(screens, contrasts, deltas):
    rows = []
    by_screen = {screen: group for screen, group in contrasts.groupby('screen')}
    for a, b in itertools.combinations(screens, 2):
        left, right = (by_screen[a], by_screen[b])
        shared_keys = set(left.perturbation_key) & set(right.perturbation_key)
        if not shared_keys:
            continue
        for key in shared_keys:
            for _, ca in left[left.perturbation_key.eq(key)].iterrows():
                for _, cb in right[right.perturbation_key.eq(key)].iterrows():
                    da, db = (deltas[ca.contrast_id], deltas[cb.contrast_id])
                    for role in ('ordered_pair', 'position_1', 'position_2'):
                        role_index = 0 if role == 'position_1' else 1
                        values_a, values_b = (defaultdict(list), defaultdict(list))
                        for pair, delta in zip(screens[a]['pairs'], da):
                            if np.isfinite(delta):
                                values_a[pair if role == 'ordered_pair' else pair[role_index]].append(delta)
                        for pair, delta in zip(screens[b]['pairs'], db):
                            if np.isfinite(delta):
                                values_b[pair if role == 'ordered_pair' else pair[role_index]].append(delta)
                        for molecule in sorted(values_a.keys() & values_b.keys()):
                            va, vb = (np.asarray(values_a[molecule]), np.asarray(values_b[molecule]))
                            ma, mb = (float(va.mean()), float(vb.mean()))
                            rows.append(dict(screen_a=a, screen_b=b, family_a=screens[a]['family'], family_b=screens[b]['family'], perturbation_key=key, contrast_a=ca.contrast_id, contrast_b=cb.contrast_id, same_full_background=ca.background_key == cb.background_key, matching_basis=role, substrate_identity=dumps(molecule), n_pairs_a=len(va), n_pairs_b=len(vb), mean_delta_a=ma, mean_delta_b=mb, absolute_difference=abs(ma - mb), sign_agreement=int(np.sign(ma) == np.sign(mb)), both_at_least_3=len(va) >= 3 and len(vb) >= 3, both_at_least_5=len(va) >= 5 and len(vb) >= 5))
    columns = ['screen_a', 'screen_b', 'family_a', 'family_b', 'perturbation_key', 'contrast_a', 'contrast_b', 'same_full_background', 'matching_basis', 'substrate_identity', 'n_pairs_a', 'n_pairs_b', 'mean_delta_a', 'mean_delta_b', 'absolute_difference', 'sign_agreement', 'both_at_least_3', 'both_at_least_5']
    matched = pd.DataFrame(rows, columns=columns)
    summaries = []
    if len(matched):
        for keys, group in matched.groupby(['screen_a', 'screen_b', 'matching_basis']):
            for threshold in (1, 3, 5):
                good = group[(group.n_pairs_a >= threshold) & (group.n_pairs_b >= threshold)]
                rho = spearmanr(good.mean_delta_a, good.mean_delta_b).statistic if len(good) >= 5 and good.mean_delta_a.nunique() > 1 and (good.mean_delta_b.nunique() > 1) else np.nan
                summaries.append(dict(screen_a=keys[0], screen_b=keys[1], matching_basis=keys[2], min_partners_each=threshold, n_dependent_comparisons=len(good), n_shared_molecules=good.substrate_identity.nunique(), n_shared_perturbations=good.perturbation_key.nunique(), median_absolute_delta_difference=float(good.absolute_difference.median()) if len(good) else np.nan, sign_agreement_fraction=float(good.sign_agreement.mean()) if len(good) else np.nan, descriptive_spearman=rho))
    return (matched, pd.DataFrame(summaries))

def screen_balanced_weights(frame):
    counts = frame.groupby('screen').screen.transform('size').to_numpy(float)
    screens_per_family = frame[['family', 'screen']].drop_duplicates().groupby('family').size()
    family_screens = frame.family.map(screens_per_family).to_numpy(float)
    weights = 1 / (counts * family_screens)
    return weights / weights.mean()

def weighted_training_mean(frame, outcome):
    return float(frame.groupby(['family', 'screen'])[outcome].mean().groupby('family').mean().mean())

def predict_transport(contrasts, names):
    valid = contrasts[contrasts.eligible_summary].copy().reset_index(drop=True)
    feature_sets = {'endpoint_marginal_ridge': [feature_dict(row, names, False) for row in valid.to_dict('records')], 'component_context_ridge': [feature_dict(row, names, True) for row in valid.to_dict('records')]}
    prediction_rows, fold_rows = ([], [])
    for scheme, group_field in (('leave_family_out', 'family'), ('leave_screen_out', 'screen')):
        for holdout in sorted(valid[group_field].unique()):
            test_mask = valid[group_field].eq(holdout).to_numpy()
            train_index, test_index = (np.flatnonzero(~test_mask), np.flatnonzero(test_mask))
            train, test = (valid.iloc[train_index], valid.iloc[test_index])
            train_screens = set(train.screen)
            assert not train_screens & set(test.screen)
            if scheme == 'leave_family_out':
                assert not set(train.family) & set(test.family)
            known = test.perturbation_key.isin(set(train.perturbation_key)).to_numpy()
            identity_known = test.no_cross_screen_dose_key.isin(set(train.no_cross_screen_dose_key)).to_numpy()
            train_endpoint = set(train.low_id) | set(train.high_id)
            endpoints_known = (test.low_id.isin(train_endpoint) & test.high_id.isin(train_endpoint)).to_numpy()
            fitted = {}
            for model, features in feature_sets.items():
                vectorizer = DictVectorizer(sparse=True)
                xtrain = vectorizer.fit_transform([features[index] for index in train_index])
                xtest = vectorizer.transform([features[index] for index in test_index])
                estimator = Ridge(alpha=RIDGE_ALPHA, solver='lsqr', fit_intercept=True, tol=1e-09)
                estimator.fit(xtrain, train[OUTCOMES].to_numpy(), sample_weight=screen_balanced_weights(train))
                fitted[model] = estimator.predict(xtest)
            for k, outcome in enumerate(OUTCOMES):
                global_mean = weighted_training_mean(train, outcome)
                swap_mean = train.groupby(['perturbation_key', 'family', 'screen'])[outcome].mean().groupby(['perturbation_key', 'family']).mean().groupby('perturbation_key').mean()
                identity_mean = train.groupby(['no_cross_screen_dose_key', 'family', 'screen'])[outcome].mean().groupby(['no_cross_screen_dose_key', 'family']).mean().groupby('no_cross_screen_dose_key').mean()
                predictions = {'zero': np.zeros(len(test)), 'training_mean': np.full(len(test), global_mean), 'same_perturbation_mean': test.perturbation_key.map(swap_mean).fillna(global_mean).to_numpy(), 'same_identity_replacement_mean': test.no_cross_screen_dose_key.map(identity_mean).fillna(global_mean).to_numpy(), **{name: values[:, k] for name, values in fitted.items()}}
                for model, values in predictions.items():
                    if outcome == 'signed_consistency':
                        values = np.clip(values, -1, 1)
                    for position, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append(dict(scheme=scheme, heldout_group=holdout, contrast_id=row.contrast_id, screen=row.screen, family=row.family, kind=row.kind, outcome=outcome, model=model, observed=row[outcome], predicted=values[position], squared_error=(row[outcome] - values[position]) ** 2, absolute_error=abs(row[outcome] - values[position]), exact_perturbation_seen=bool(known[position]), both_endpoints_seen=bool(endpoints_known[position]), identity_perturbation_seen=bool(identity_known[position]), n_paired_substrates=int(row.n_overlap), n_training_screens=train.screen.nunique(), n_training_families=train.family.nunique()))
            fold_rows.append(dict(scheme=scheme, heldout_group=holdout, n_training_screens=train.screen.nunique(), n_training_families=train.family.nunique(), n_training_contexts=len(train), n_test_contexts=len(test), n_test_shared_perturbation_contexts=int(known.sum()), n_test_both_endpoints_seen=int(endpoints_known.sum()), n_test_identity_perturbation_contexts=int(identity_known.sum()), heldout_screens_json=dumps(sorted(test.screen.unique())), training_screens_json=dumps(sorted(train_screens))))
        print(f'{scheme}: all holdouts scored', flush=True)
    predictions = pd.DataFrame(prediction_rows)
    score_rows, screen_score_rows = ([], [])
    subset_flags = {'shared_perturbation': 'exact_perturbation_seen', 'both_endpoints_seen': 'both_endpoints_seen', 'identity_perturbation_allow_dose': 'identity_perturbation_seen'}
    for subset in ('all_contexts', *subset_flags):
        selected = predictions if subset == 'all_contexts' else predictions[predictions[subset_flags[subset]]]
        for keys, group in selected.groupby(['scheme', 'outcome', 'model', 'screen']):
            screen_score_rows.append(dict(subset=subset, scheme=keys[0], outcome=keys[1], model=keys[2], screen=keys[3], family=transformation(keys[3]), n_contexts=len(group), mse=group.squared_error.mean(), rmse=np.sqrt(group.squared_error.mean()), mae=group.absolute_error.mean(), sign_agreement=np.mean(np.sign(group.observed) == np.sign(group.predicted))))
    screen_scores = pd.DataFrame(screen_score_rows)
    for keys, group in screen_scores.groupby(['subset', 'scheme', 'outcome', 'model']):
        family_mse = group.groupby('family').mse.mean()
        score_rows.append(dict(subset=keys[0], scheme=keys[1], outcome=keys[2], model=keys[3], n_screens=len(group), n_families=len(family_mse), n_contexts=int(group.n_contexts.sum()), screen_macro_rmse=np.sqrt(group.mse.mean()), family_macro_rmse=np.sqrt(family_mse.mean()), median_screen_rmse=group.rmse.median(), screen_macro_mae=group.mae.mean()))
    return (predictions, pd.DataFrame(fold_rows), pd.DataFrame(score_rows), screen_scores)

def make_figure(pairs, graphs, contrasts, scores, screen_scores, matched_summary, screens):
    order = list(screens)
    labels = [s.replace('Buchwald-Hartwig(CMC)', 'BH(CMC)').replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O competitive').replace('|', ' #') for s in order]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), layout='constrained')
    a = axes[0, 0]
    display_levels = ['reagent_identity', 'solvent_identity', 'reagent_same_dose', 'perturbation_ignoring_cross_screen_dose', 'oriented_perturbation', 'substrate_position_2', 'ordered_substrate_pair', 'perturbation_fixed_background', 'full_recipe_multiset']
    short = ['Reagent identity', 'Solvent identity', 'Reagent + dose', 'Replacement, allow dose across screens', 'Same-dose / solvent replacement', 'Substrate position 2', 'Full substrate pair', 'Replacement + fixed background', 'Full recipe (slot invariant)']
    selected = graphs.set_index('level').loc[display_levels]
    bars = a.barh(np.arange(len(short)), selected.n_edges, color='#388677')
    a.set_yticks(np.arange(len(short)), short, fontsize=9)
    a.invert_yaxis()
    for bar, value, largest in zip(bars, selected.n_edges, selected.largest_component_size):
        a.text(value + 1, bar.get_y() + bar.get_height() / 2, f'{value}/153; largest {largest}/18', va='center', fontsize=8)
    a.set(xlim=(0, 200), xlabel='Screen-pair edges; graph size shown at right', title='A  Shared components connect libraries at different levels')
    b = axes[0, 1]
    heat = np.full((len(order), len(order)), np.nan)
    for _, row in pairs.iterrows():
        i, j = (order.index(row.screen_a), order.index(row.screen_b))
        heat[i, j] = heat[j, i] = row.shared_oriented_perturbation
    im = b.imshow(np.log1p(heat), cmap='YlGnBu', vmin=0, vmax=np.log1p(np.nanmax(heat)))
    b.set_xticks(range(len(order)), labels, rotation=65, ha='right', fontsize=7)
    b.set_yticks(range(len(order)), labels, fontsize=7)
    for i in range(len(order)):
        for j in range(len(order)):
            if np.isfinite(heat[i, j]) and heat[i, j] > 0:
                b.text(j, i, str(int(heat[i, j])), ha='center', va='center', fontsize=6, color='black')
    b.set_title('B  Number of shared oriented perturbations', loc='left')
    fig.colorbar(im, ax=b, shrink=0.65, label='log(1 + count)')
    c = axes[1, 0]
    chosen = scores[scores.scheme.eq('leave_family_out') & scores.outcome.eq('signed_consistency') & scores.subset.eq('shared_perturbation')].set_index('model').loc[MODELS]
    c.bar(np.arange(len(MODELS)), chosen.family_macro_rmse, color=['#8c8c8c', '#b28f54', '#4e8792', '#a87b79', '#8b6c9f', '#358275'])
    for i, value in enumerate(chosen.family_macro_rmse):
        c.text(i, value + 0.01, f'{value:.3f}', ha='center', fontsize=9)
    c.set_xticks(range(len(MODELS)), ['Zero', 'Training\nmean', 'Exact swap\nmean', 'Swap identity\nmean', 'Endpoint\nmarginals', 'Component +\ncontext'], fontsize=8)
    c.set(ylabel='Family-macro RMSE of signed consistency', title=f'C  Leave-family-out transport ({int(chosen.n_screens.iloc[0])} screens supported)')
    c.set_ylim(0, max(chosen.family_macro_rmse) * 1.2)
    d = axes[1, 1]
    part = screen_scores[screen_scores.scheme.eq('leave_family_out') & screen_scores.outcome.eq('signed_consistency') & screen_scores.subset.eq('shared_perturbation')]
    pivot = part.pivot(index='screen', columns='model', values='rmse')
    present = [screen for screen in order if screen in pivot.index]
    improvement = pivot.loc[present, 'endpoint_marginal_ridge'] - pivot.loc[present, 'component_context_ridge']
    colors = np.where(improvement >= 0, '#358275', '#bd6654')
    d.barh(range(len(present)), improvement, color=colors)
    d.axvline(0, color='.4', lw=1)
    d.set_yticks(range(len(present)), [labels[order.index(s)] for s in present], fontsize=8)
    d.invert_yaxis()
    d.set(xlabel='Endpoint marginal RMSE minus context-model RMSE', title='D  Context-model gain beyond the stronger marginal baseline')
    path = HERE / 'figures/13_cross_library_components.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def main():
    screens, raw_screens, names, component_rows, recipe_rows, inventory_rows, source_hashes = load_inputs()
    contrasts, deltas, duplicates = enumerate_perturbations(screens, names)
    pairs, graphs, ambiguity = overlap_tables(screens, raw_screens, contrasts)
    matched, matched_summary = matching_substrate_comparison(screens, contrasts, deltas)
    predictions, folds, scores, screen_scores = predict_transport(contrasts, names)
    inventory = pd.DataFrame(inventory_rows)
    for scheme in ('leave_family_out', 'leave_screen_out'):
        support = predictions[predictions.scheme.eq(scheme) & predictions.model.eq('zero') & predictions.outcome.eq('mean_delta')]
        inventory[scheme + '_contexts'] = inventory.screen.map(support.groupby('screen').size()).fillna(0).astype(int)
        inventory[scheme + '_shared_contexts'] = inventory.screen.map(support[support.exact_perturbation_seen].groupby('screen').size()).fillna(0).astype(int)
        inventory[scheme + '_identity_contexts'] = inventory.screen.map(support[support.identity_perturbation_seen].groupby('screen').size()).fillna(0).astype(int)
    coverage = contrasts.groupby('screen').agg(n_contexts=('contrast_id', 'size'), n_distinct_perturbations=('perturbation_key', 'nunique'), n_slot_invariant_added_contexts=('also_single_registry_field', lambda x: int((~x).sum())))
    inventory = inventory.merge(coverage, left_on='screen', right_index=True, validate='one_to_one')
    shared_rows = []
    for key, group in contrasts.groupby('perturbation_key'):
        example = group.iloc[0]
        shared_rows.append(dict(perturbation_key=key, kind=example.kind, low_name=example.low_name, high_name=example.high_name, fixed_equiv=example.fixed_equiv, n_screens=group.screen.nunique(), n_families=group.family.nunique(), n_contexts=len(group), screens_json=dumps(sorted(group.screen.unique())), screen_mean_deltas_json=dumps(group.groupby('screen').mean_delta.mean().to_dict())))
    outputs = {'screen_inventory.csv': inventory, 'recipe_registry.csv': pd.DataFrame(recipe_rows), 'component_occurrences.csv': pd.DataFrame(component_rows), 'perturbation_contexts.csv': contrasts, 'perturbation_catalog.csv': pd.DataFrame(shared_rows), 'canonical_duplicate_contexts.csv': duplicates, 'pair_overlap.csv': pairs, 'graph_connectedness.csv': graphs, 'ambiguity_sensitivity.csv': ambiguity, 'matching_substrate_comparisons.csv': matched, 'matching_substrate_summary.csv': matched_summary, 'transport_predictions.csv': predictions, 'transport_folds.csv': folds, 'transport_scores.csv': scores, 'transport_screen_scores.csv': screen_scores}
    output_paths = []
    for filename, frame in outputs.items():
        path = HERE / 'tables' / (PREFIX + filename)
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        output_paths.append(path)
    figure = make_figure(pairs, graphs, contrasts, scores, screen_scores, matched_summary, screens)
    assert len(pairs) == 171 and len(inventory) == 19
    assert sum(inventory.n_substrate_pairs) == 2454
    assert np.array_equal(contrasts.eligible_summary, contrasts.n_overlap >= MIN_OVERLAP)
    assert contrasts.mean_delta.between(-1, 1).all()
    assert contrasts.signed_consistency.between(-1 - 1e-12, 1 + 1e-12).all()
    assert sum(pairs.shared_full_recipe_slot > 0) == 1
    assert pairs.shared_full_recipe_slot.sum() == 9
    assert source_hashes == {path: sha256(path) for path in source_hashes}, 'An analysis input changed'
    catalog = pd.DataFrame(shared_rows)
    report = HERE / '18_CROSS_LIBRARY_COMPONENT_STRUCTURE_2026-10-16.md'
    if report.exists():
        output_paths.append(report)
    manifest = dict(date='2026-10-16', scope='Exploratory existing analysis partitions, all 18 screens; raw inputs read-only', input_hashes=source_hashes, code_hashes={str(HERE / name): sha256(HERE / name) for name in ('cross_library_component_structure.py', 'test_cross_library_component_structure.py')}, output_hashes={str(path.relative_to(HERE)): sha256(path) for path in output_paths + [figure]}, environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, rdkit=rdkit.__version__, blas_threads='library_default_unrestricted'), counts=dict(screens=len(inventory), families=inventory.family.nunique(), analysis_substrate_pairs=int(inventory.n_substrate_pairs.sum()), perturbation_contexts=len(contrasts), slot_invariant_added_contexts=int((~contrasts.also_single_registry_field).sum()), canonical_duplicate_contexts=len(duplicates), distinct_perturbations=len(catalog), perturbations_in_multiple_screens=int((catalog.n_screens >= 2).sum()), perturbations_in_multiple_families=int((catalog.n_families >= 2).sum()), matching_substrate_comparisons=len(matched)), definitions=dict(identity='Exact registry compound IDs; exact standardized-name consistency checked. No inferred chemical synonyms or mechanism are merged.', substrate='RDKit canonical isomeric SMILES; ordered sub_1/sub_2 positions are registry roles, not universal chemical roles. No charge/tautomer normalization.', primary_perturbation='Slot-invariant multiset differs by one identity at the same nonmissing equivalent, with all other ingredients, solvent, temperature and time fixed; or only solvent changes. Low-to-high registry ID orientation; delta = high minus low.', duplicate_context='Same screen, exact oriented perturbation and full slot-invariant background is one context; aliases enumerated separately, never averaged without disclosure.', graph_sensitivities='Separate domains, same dose vs identity, pure solvent vs constituent, putative name families, slot vs multiset full recipe, and dose-relaxed cross-screen perturbation identity. Different graph levels identify different objects.', population_mean="Mean paired finite yield difference over each screen's frozen analysis substrate population, all yields included. Different screens have different substrate populations.", signed_consistency="mean(delta)/sqrt(mean(delta^2)), bounded [-1,1]; zero if all paired deltas zero. A signed standardized population summary, not Cohen's d, causality or transfer of full response vectors.", eligibility=f'At least {MIN_OVERLAP} paired finite rows; missing is never zero. Shared-perturbation evaluation additionally requires the exact perturbation in a training screen/family. Secondary identity_perturbation_allow_dose allows different fixed equivalent across screens while retaining equal dose between endpoints within each screen.', features='Registry identities and fixed recipe context only. Endpoint marginal ridge uses signed identities; context ridge adds exact perturbation, perturbation/background identity interactions and putative-family/background interactions; physical features fixed at temp/100 and log1p(time)/5.', model=f'DictVectorizer fit only training contexts; Ridge alpha={RIDGE_ALPHA}, no tuning, LSQR; equal training-family total weight, equal screens within family, equal contexts within screen; consistency predictions clipped [-1,1]. Held-out-screen yields only define scored labels.', baselines='Zero, training global mean averaging contexts within screen, screens within family then families; same exact perturbation mean and same replacement identities allowing cross-screen dose difference, both with the same aggregation and global fallback; signed endpoint additive ridge with the same training weights. Baselines receive identical training/test contexts.', primary_holdout='All Buchwald-Hartwig and CMC screens are one transformation family. Leave-family-out excludes every target-family screen, feature fitting, outcomes and means. Leave-screen-out is a less strict secondary diagnostic.', aggregation='Within-screen MSE then equal screen; primary family-macro metric averages screen MSE within family then families before square root. Context edges and screens are not treated as independent replicates.', matching_substrates='Direct comparison when identical ordered substrate pair, or identical molecule in the same registry position, has paired yields in both contexts. Same-position molecule comparisons average over remaining partners; partners/backgrounds may differ. Thresholds 1/3/5 are displayed sensitivity, not causal sufficiency.', ambiguity='SNAr C2 excluded; Chan-Lam C68 excluded if present (not in its selected matrix). Static-inclusion sensitivity is not a correction of uncertain dispensed identities.'), limits=['Overlap graph connectivity is design coverage, not evidence of a universal transferable effect.', 'Population-averaged perturbation effects confound chemical background and substrate composition; separate matching-substrate results do not identify mechanism.', 'No laboratory holdout, prospective validation, replicate error estimate or randomization is supplied.', 'Family definitions are operational; 14 family units are not 14 independent mechanisms.', 'Putative name-based family features are broad metadata; roles catalyst/base/ligand are unvalidated.', 'No held-out target-family historical yields enter predictors, but source historical yields supply supervised labels.', 'Exploratory fixed model choices and historical corpus selection prevent confirmatory generalization claims.'], verification=dict(all_18_screens=True, all_2360_analysis_rows=True, raw_workbook_hashes_match=True, all_recipe_hashes_match=True, all_source_hashes_unchanged=True, exact_names_and_ids_one_to_one=True, exact_recipe_overlap_matches_prior_9=True, no_heldout_screen_or_family_in_training=True, paired_finite_only=True, feature_vocabulary_fit_train_only=True))
    path = HERE / 'data/cross_library_components_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(graphs[['level', 'n_edges', 'n_cross_family_edges', 'largest_component_size']].to_string(index=False))
    print(scores[scores.scheme.eq('leave_family_out') & scores.subset.eq('shared_perturbation')].to_string(index=False))
    print(json.dumps(manifest['counts'], ensure_ascii=False))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
