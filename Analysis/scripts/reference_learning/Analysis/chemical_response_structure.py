"""Exploratory recipe/response geometry and strictly matched recipe contrasts.

Only frozen analysis matrices are used. Recipe identities are registry IDs, not
name-based mechanistic families. All original inputs are opened read-only."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits
import sys as _m1_sys
_m1_sys.path.append(str(next((p for p in Path(__file__).resolve().parents if (p / 'm1_raw.py').is_file()))))
from m1_raw import RAW_DIR as _M1_RAW_DIR
HERE = Path(__file__).resolve().parent
CORE = [field for i in range(1, 7) for field in (f'Reagent{i}_id', f'Reagent{i}_HTE_Equiv')]
CORE += ['solvent_id', 'temp_id', 'time_id']
SEED = 2026101608
AMBIGUOUS = {('SNAr', 'C2'), ('Chan-Lam', 'C68')}
VIEWS = {'overall_rms': 'All rows', 'productive_rms': 'Productive rows', 'productive_interaction_rms': 'Additive residual', 'productive_generic_adjusted_rms': 'Generic activity removed', 'productive_row_z_rms': 'Row standardized'}

def sha256(path):
    if reader_tables.is_source(path):
        return reader_tables.source_evidence(path)['sha256']
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def finite_median(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan

def correlation(a, b, rank=False):
    if len(a) < 5 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return np.nan
    if rank:
        a, b = (rankdata(a), rankdata(b))
    return float(np.corrcoef(a, b)[0, 1])

def additive_residual(m):
    mask = np.isfinite(m)
    row = np.nanmean(m, axis=1, keepdims=True)
    col = np.zeros((1, m.shape[1]))
    for _ in range(100):
        previous = col.copy()
        row = np.nanmean(m - col, axis=1, keepdims=True)
        col = np.nanmean(m - row, axis=0, keepdims=True)
        if np.max(np.abs(col - previous)) < 1e-12:
            break
    residual = np.where(mask, m - row - col, np.nan)
    assert np.nanmax(np.abs(np.nanmean(residual, axis=0))) < 1e-09
    assert np.nanmax(np.abs(np.nanmean(residual, axis=1))) < 1e-09
    return residual

def matched_kind(a, b):
    changed = [field for field in CORE if a[field] != b[field]]
    if len(changed) != 1:
        return ('', changed, '')
    field = changed[0]
    if field.startswith('Reagent'):
        slot = field.split('_')[0]
        if field.endswith('_id') and a[field] and b[field]:
            assert a[slot + '_HTE_Equiv'] == b[slot + '_HTE_Equiv']
            assert a[slot + '_HTE_Equiv'] != ''
            return ('identity_swap', changed, slot)
        if field.endswith('_HTE_Equiv') and a[slot + '_id'] == b[slot + '_id']:
            return ('dose_only', changed, slot)
        return ('addition_removal_or_incomplete', changed, slot)
    return ({'solvent_id': 'solvent_only', 'temp_id': 'temperature_only', 'time_id': 'time_only'}[field], changed, field)

def block_values(row):
    return [(row[f'Reagent{i}_id'], row[f'Reagent{i}_HTE_Equiv']) for i in range(1, 7)] + [row[x] for x in ('solvent_id', 'temp_id', 'time_id')]

def recipe_distances(recipes):
    blocks = [block_values(row) for row in recipes]
    varying = [i for i in range(9) if len(set((row[i] for row in blocks))) > 1]
    tokens = [Counter(((row[f'Reagent{i}_id'], row[f'Reagent{i}_HTE_Equiv']) for i in range(1, 7) if row[f'Reagent{i}_id'])) for row in recipes]
    token_varying = len({tuple(sorted(x.items())) for x in tokens}) > 1
    physical = [field for field in ('solvent_id', 'temp_id', 'time_id') if len({row[field] for row in recipes}) > 1]
    n = len(recipes)
    primary = np.zeros((n, n))
    invariant = np.zeros((n, n))
    invariant_edit = np.zeros((n, n))
    changed_count = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            changed_count[i, j] = changed_count[j, i] = sum((blocks[i][k] != blocks[j][k] for k in varying))
            primary[i, j] = primary[j, i] = changed_count[i, j] / len(varying) if varying else 0.0
            values = [float(recipes[i][field] != recipes[j][field]) for field in physical]
            keys = tokens[i].keys() | tokens[j].keys()
            intersection = sum((min(tokens[i][k], tokens[j][k]) for k in keys))
            edit = max(sum(tokens[i].values()), sum(tokens[j].values())) - intersection
            edit += sum((recipes[i][field] != recipes[j][field] for field in physical))
            invariant_edit[i, j] = invariant_edit[j, i] = edit / len(varying) if varying else 0.0
            assert invariant_edit[i, j] <= primary[i, j] + 1e-12
            if token_varying:
                union = sum((max(tokens[i][k], tokens[j][k]) for k in keys))
                values.append(1 - intersection / union if union else 0.0)
            invariant[i, j] = invariant[j, i] = np.mean(values) if values else 0.0
    names = [f'Reagent{i}' for i in range(1, 7)] + ['solvent', 'temperature', 'time']
    return ({'slot_block_hamming': primary, 'slot_invariant_edit': invariant_edit, 'domain_multiset_jaccard': invariant}, changed_count, [names[i] for i in varying])

def pair_metrics(m, residual, z, productive, i, j):
    out = {}
    for label, subset in (('overall', np.ones(len(m), dtype=bool)), ('productive', productive), ('pair_active', np.maximum(m[:, i], m[:, j]) >= 0.05)):
        keep = subset & np.isfinite(m[:, i]) & np.isfinite(m[:, j])
        a, b = (m[keep, i], m[keep, j])
        n = len(a)
        out[label + '_n_overlap'] = n
        for suffix in ('rms', 'mad', 'pearson', 'spearman', 'mean_delta', 'centered_rms', 'centered_fraction'):
            out[label + '_' + suffix] = np.nan
        if n < 10:
            continue
        delta = a - b
        total = np.mean(delta ** 2)
        centered = np.mean((delta - delta.mean()) ** 2)
        assert np.isclose(total, centered + delta.mean() ** 2, atol=1e-12)
        out.update({label + '_rms': np.sqrt(total), label + '_mad': np.mean(np.abs(delta)), label + '_pearson': correlation(a, b), label + '_spearman': correlation(a, b, rank=True), label + '_mean_delta': np.mean(delta), label + '_centered_rms': np.sqrt(centered), label + '_centered_fraction': centered / total if total > 1e-16 else np.nan})
    keep = productive & np.isfinite(m[:, i]) & np.isfinite(m[:, j])
    out['productive_interaction_rms'] = np.sqrt(np.mean((residual[keep, i] - residual[keep, j]) ** 2)) if keep.sum() >= 10 else np.nan
    valid_z = np.isfinite(z[:, i]) & np.isfinite(z[:, j])
    out['productive_row_z_n_overlap'] = int(valid_z.sum())
    out['productive_row_z_rms'] = np.sqrt(np.mean((z[valid_z, i] - z[valid_z, j]) ** 2)) if valid_z.sum() >= 10 else np.nan
    count = np.isfinite(m).sum(axis=1) - np.isfinite(m[:, i]) - np.isfinite(m[:, j])
    total = np.nansum(m, axis=1) - np.nan_to_num(m[:, i]) - np.nan_to_num(m[:, j])
    propensity = np.divide(total, count, out=np.full(len(m), np.nan), where=count > 0)
    keep &= np.isfinite(propensity)
    out['productive_generic_adjusted_rms'] = np.nan
    out['productive_generic_residual_fraction'] = np.nan
    out['productive_generic_slope'] = np.nan
    if keep.sum() >= 10:
        delta = m[keep, i] - m[keep, j]
        g = propensity[keep] - np.mean(propensity[keep])
        design = np.column_stack([np.ones(len(g)), g])
        coef = np.linalg.lstsq(design, delta, rcond=None)[0]
        error = delta - design @ coef
        rms2 = np.mean(error ** 2)
        total2 = np.mean(delta ** 2)
        centered2 = np.mean((delta - delta.mean()) ** 2)
        assert rms2 <= centered2 + 1e-12
        out['productive_generic_adjusted_rms'] = np.sqrt(rms2)
        out['productive_generic_residual_fraction'] = rms2 / total2 if total2 > 1e-16 else np.nan
        out['productive_generic_slope'] = coef[1]
    return out

def ranked_correlation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 20:
        return np.nan
    a, b = (rankdata(x[good]), rankdata(y[good]))
    a -= a.mean()
    b -= b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 else np.nan

def mantel(recipe, response, permutations):
    n = len(recipe)
    tri = np.triu_indices(n, 1)
    x, y = (recipe[tri], response[tri])
    observed = ranked_correlation(x, y)
    valid = np.isfinite(y)
    n_pairs = int(valid.sum())
    if not np.isfinite(observed):
        return dict(rho=np.nan, n_distance_pairs=n_pairs, permutation_p_two_sided=np.nan, permutation_p_positive=np.nan, null_q025=np.nan, null_q975=np.nan, null_mean=np.nan, n_null_extreme=np.nan)
    reference = spearmanr(x[valid], y[valid]).statistic
    assert abs(reference - observed) < 1e-12
    ranked_y = np.full_like(response, np.nan)
    ranks = rankdata(y[valid])
    ranked_y[tri[0][valid], tri[1][valid]] = ranks
    ranked_y[tri[1][valid], tri[0][valid]] = ranks
    ranks -= ranks.mean()
    if valid.all():
        rx = rankdata(x)
        rx -= rx.mean()
        denominator = np.linalg.norm(rx) * np.linalg.norm(ranks)
        null = np.array([rx @ (ranked_y[np.ix_(p, p)][tri] - np.mean(ranked_y[np.ix_(p, p)][tri])) / denominator for p in permutations])
    else:
        null = np.array([ranked_correlation(x, ranked_y[np.ix_(p, p)][tri]) for p in permutations])
    assert np.isfinite(null).all()
    two = int((np.abs(null) >= abs(observed) - 1e-14).sum())
    positive = int((null >= observed - 1e-14).sum())
    return dict(rho=observed, n_distance_pairs=n_pairs, permutation_p_two_sided=(1 + two) / (len(null) + 1), permutation_p_positive=(1 + positive) / (len(null) + 1), null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)), null_mean=float(null.mean()), n_null_extreme=two)

def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    ok = np.flatnonzero(np.isfinite(values))
    order = ok[np.argsort(values[ok])]
    if len(order):
        q = values[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return result

def make_figure(associations, matched, coverage, summary):
    order = coverage.screen.tolist()
    primary = associations[associations.recipe_distance.eq('slot_block_hamming')]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')
    a = axes[0, 0]
    heat = primary.pivot(index='screen', columns='response_view', values='rho').reindex(index=order, columns=VIEWS)
    im = a.imshow(heat, vmin=-0.4, vmax=0.6, cmap='RdBu_r', aspect='auto')
    a.set_yticks(range(len(order)), [x.replace('|', ' #') for x in order], fontsize=8)
    a.set_xticks(range(len(VIEWS)), VIEWS.values(), rotation=35, ha='right', fontsize=8)
    a.set_title('A  Recipe distance / response distance (Spearman)', loc='left', fontsize=11)
    for i, screen in enumerate(order):
        for j, view in enumerate(VIEWS):
            row = primary[primary.screen.eq(screen) & primary.response_view.eq(view)].iloc[0]
            if np.isfinite(row.rho):
                a.text(j, i, f'{row.rho:.2f}' + ('*' if row.bh_q_two_sided < 0.05 else ''), ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=a, label='Mantel rho', shrink=0.6)
    b = axes[0, 1]
    strict = matched[matched.contrast_type.isin(['identity_swap', 'solvent_only'])]
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(order)))
    for screen, color in zip(order, colors):
        g = strict[strict.screen.eq(screen)]
        if g.productive_rms.notna().any():
            b.scatter(100 * finite_median(g.productive_rms), 100 * finite_median(g.productive_generic_adjusted_rms), color=color, s=40)
    upper = max(b.get_xlim()[1], b.get_ylim()[1])
    b.plot([0, upper], [0, upper], color='.5', lw=1, linestyle='--')
    b.set(xlabel='Matched composition swap: total RMS (pp)', ylabel='After generic activity removal (pp)', title='B  Screen medians on productive overlap', xlim=(0, upper), ylim=(0, upper))
    c = axes[1, 0]
    counts = matched.groupby(['contrast_type', 'contrast_category']).size().sort_values()
    labels = [f'{kind}: {category}' for kind, category in counts.index]
    c.barh(labels, counts, color='#2e8b77')
    c.set(xlabel='Exact matched condition pairs (dependent edges)', title='C  Coverage of strict registry contrasts')
    c.tick_params(axis='y', labelsize=8)
    d = axes[1, 1]
    for screen, color in zip(order, colors):
        p = primary[primary.screen.eq(screen) & primary.response_view.eq('productive_generic_adjusted_rms')]
        s = associations[associations.screen.eq(screen) & associations.recipe_distance.eq('domain_multiset_jaccard') & associations.response_view.eq('productive_generic_adjusted_rms')]
        if len(p) and len(s):
            d.scatter(p.rho, s.rho, color=color, s=40, label=screen.replace('|', ' #'))
    d.axhline(0, color='.7', lw=1)
    d.axvline(0, color='.7', lw=1)
    d.set(xlabel='Slot-block recipe distance: Mantel rho', ylabel='Domain-weighted multiset distance: Mantel rho', title='D  Generic-activity residual: weighting sensitivity')
    d.legend(fontsize=7, ncol=2, loc='best', frameon=False)
    fig.savefig(HERE / 'figures/08_chemical_response_structure.png', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permutations', type=int, default=499)
    args = parser.parse_args()
    assert args.permutations >= 19
    source_paths = [HERE / 'tables/condition_recipe_audit_details.csv', HERE / 'tables/condition_recipe_audit_provenance.json', HERE / 'data/raw_condition_table.csv', HERE / 'tables/screen_inventory.csv', HERE / 'condition_audit.py']
    matrices = sorted((HERE / 'data').glob('matrix_*.npz'))
    source_paths += matrices
    provenance = json.loads((HERE / 'tables/condition_recipe_audit_provenance.json').read_text(encoding='utf-8'))
    raw_dir = _M1_RAW_DIR
    for name in ('0_total condition_260810.xlsx', '0_Compound ID_260810.xlsx'):
        path = raw_dir / name
        assert sha256(path) == provenance['source_hashes'][name]['sha256']
        source_paths.append(path)
    hashes_before = {str(path): sha256(path) for path in source_paths}
    details = pd.read_csv(source_paths[0], dtype=str, keep_default_na=False)
    registry = pd.read_csv(HERE / 'data/raw_condition_table.csv', dtype=str, keep_default_na=False)
    mapping = registry[['reaction_class', 'reaction_type']].drop_duplicates()
    assert not mapping.reaction_type.duplicated().any()
    details = details.merge(mapping, on='reaction_type', validate='many_to_one')
    inventory = pd.read_csv(HERE / 'tables/screen_inventory.csv')
    filename_to_screen = {'matrix_' + screen.replace('|', '_').replace(' ', '_') + '.npz': screen for screen in inventory.screen}
    for _, row in details.iterrows():
        expected = hashlib.sha256(dumps([row[field] for field in CORE]).encode('utf-8')).hexdigest()
        assert expected == row.recipe_full_id_dose_sha256
    rng = np.random.default_rng(SEED)
    pair_rows, matched_rows, association_rows, coverage_rows = ([], [], [], [])
    for path in matrices:
        screen = filename_to_screen[path.name]
        cls = screen.split('|')[0]
        saved = np.load(path, allow_pickle=False)
        m, conds = (saved['M'], saved['conds'].astype(str))
        assert len(saved['sps']) == len(m)
        assert np.nanmin(m) >= 0 and np.nanmax(m) <= 1
        original_n = len(conds)
        excluded = [c for c in conds if (cls, c) in AMBIGUOUS]
        keep_columns = np.array([(cls, c) not in AMBIGUOUS for c in conds])
        m, conds = (m[:, keep_columns], conds[keep_columns])
        lookup = details[details.reaction_class.eq(cls)].set_index('condition_id', verify_integrity=True)
        assert set(conds).issubset(lookup.index)
        recipes = [lookup.loc[c].to_dict() for c in conds]
        for recipe in recipes:
            assert recipe['missing_component_id_slots_json'] == '[]'
            assert recipe['missing_dose_slots_json'] == '[]'
            assert recipe['orphan_dose_slots_json'] == '[]'
        productive = np.nanmax(m, axis=1) >= 0.2
        sd = np.nanstd(m, axis=1)
        z = np.full_like(m, np.nan)
        eligible = productive & (sd > 0.01)
        z[eligible] = (m[eligible] - np.nanmean(m[eligible], axis=1, keepdims=True)) / sd[eligible, None]
        residual = additive_residual(m)
        distances, changes, varying = recipe_distances(recipes)
        response = {view: np.full((len(conds), len(conds)), np.nan) for view in VIEWS}
        n_matched = 0
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                a, b = (recipes[i], recipes[j])
                kind, changed, category = matched_kind(a, b)
                metrics = pair_metrics(m, residual, z, productive, i, j)
                row = dict(screen=screen, reaction_class=cls, condition_a=conds[i], condition_b=conds[j], changed_core_field_count=len(changed), changed_block_count=int(changes[i, j]), slot_block_recipe_distance=distances['slot_block_hamming'][i, j], slot_invariant_edit_distance=distances['slot_invariant_edit'][i, j], domain_multiset_jaccard_distance=distances['domain_multiset_jaccard'][i, j], **metrics)
                pair_rows.append(row)
                for view in VIEWS:
                    response[view][i, j] = response[view][j, i] = metrics[view]
                if kind:
                    n_matched += 1
                    field = changed[0]
                    slot = category if category.startswith('Reagent') else ''
                    matched_rows.append(dict(**row, contrast_type=kind, contrast_category=category, changed_fields_json=dumps(changed), value_a=a[field], value_b=b[field], name_a=a[slot + '_Standardized Name'] if slot else a['Standardized Name'] if category == 'solvent_id' else a[field], name_b=b[slot + '_Standardized Name'] if slot else b['Standardized Name'] if category == 'solvent_id' else b[field], fixed_equiv=a[slot + '_HTE_Equiv'] if slot else '', fixed_core_fields_json=dumps({key: a[key] for key in CORE if key != field}), recipe_a_sha256=a['recipe_full_id_dose_sha256'], recipe_b_sha256=b['recipe_full_id_dose_sha256'], source_row_a=int(a['source_row']), source_row_b=int(b['source_row'])))
        permutations = [rng.permutation(len(conds)) for _ in range(args.permutations)]
        for definition, distance in distances.items():
            for view, matrix in response.items():
                association_rows.append(dict(screen=screen, reaction_class=cls, recipe_distance=definition, response_view=view, n_conditions=len(conds), permutations=args.permutations, **mantel(distance, matrix, permutations)))
        coverage_rows.append(dict(screen=screen, reaction_class=cls, n_rows=len(m), original_conditions=original_n, n_conditions=len(conds), excluded_ambiguous_conditions_json=dumps(excluded), productive_rows=int(productive.sum()), productive_row_z_rows=int(eligible.sum()), exact_single_field_pairs=n_matched, all_condition_pairs=len(conds) * (len(conds) - 1) // 2, varying_recipe_blocks_json=dumps(varying), missing_yield_fraction=float(np.isnan(m).mean()), zero_yield_fraction=float((m[np.isfinite(m)] == 0).mean())))
        print(f'{screen}: {len(conds)} conditions, {productive.sum()} productive rows, {n_matched} matched edges', flush=True)
    pairs = pd.DataFrame(pair_rows)
    matched = pd.DataFrame(matched_rows)
    associations = pd.DataFrame(association_rows)
    coverage = pd.DataFrame(coverage_rows)
    associations['bh_q_two_sided'] = np.nan
    for _, indices in associations.groupby(['recipe_distance', 'response_view']).groups.items():
        associations.loc[indices, 'bh_q_two_sided'] = bh_adjust(associations.loc[indices, 'permutation_p_two_sided'])
    summaries = []
    for (screen, kind, category), group in matched.groupby(['screen', 'contrast_type', 'contrast_category'], sort=True):
        summaries.append(dict(screen=screen, reaction_class=screen.split('|')[0], contrast_type=kind, contrast_category=category, n_pairs=len(group), n_supported_productive_pairs=int(group.productive_rms.notna().sum()), n_supported_pair_active_pairs=int(group.pair_active_rms.notna().sum()), median_productive_overlap=finite_median(group.productive_n_overlap), **{'median_' + column: finite_median(group[column]) for column in ('overall_rms', 'productive_rms', 'productive_mad', 'productive_pearson', 'productive_spearman', 'productive_centered_rms', 'productive_centered_fraction', 'productive_interaction_rms', 'productive_generic_adjusted_rms', 'productive_generic_residual_fraction', 'productive_row_z_rms')}))
    summary = pd.DataFrame(summaries)
    comparisons = []
    for screen, group in pairs.groupby('screen', sort=True):
        swaps = matched[matched.screen.eq(screen) & matched.contrast_type.isin(['identity_swap', 'solvent_only'])]
        multi = group[group.changed_block_count >= 2]
        for view in VIEWS:
            near, far = (finite_median(swaps[view]), finite_median(multi[view]))
            comparisons.append(dict(screen=screen, response_view=view, n_composition_swap_pairs=int(swaps[view].notna().sum()), n_multiblock_pairs=int(multi[view].notna().sum()), composition_swap_median=near, multiblock_median=far, near_far_ratio=near / far if far > 0 else np.nan))
    comparisons = pd.DataFrame(comparisons)
    outputs = {'chemical_response_pair_geometry.csv': pairs, 'chemical_response_matched_pairs.csv': matched, 'chemical_response_matched_summary.csv': summary, 'chemical_response_distance_associations.csv': associations, 'chemical_response_coverage.csv': coverage, 'chemical_response_near_far.csv': comparisons}
    for name, frame in outputs.items():
        frame.to_csv(HERE / 'tables' / name, index=False, encoding='utf-8-sig')
    make_figure(associations, matched, coverage, summary)
    hashes_after = {str(path): sha256(path) for path in source_paths}
    assert hashes_before == hashes_after, 'An input changed during analysis'
    assert len(coverage) == 19 and coverage.n_rows.sum() == 2454
    assert matched.changed_core_field_count.eq(1).all()
    assert not matched.duplicated(['screen', 'condition_a', 'condition_b']).any()
    manifest = dict(date='2026-10-16', scope='Exploratory existing-data extension; analysis partitions only', seed=SEED, permutations=args.permutations, source_hashes=hashes_before, inputs_unchanged=True, code_sha256=sha256(Path(__file__)), output_hashes={name: sha256(HERE / 'tables' / name) for name in outputs}, figure_sha256=sha256(HERE / 'figures/08_chemical_response_structure.png'), rows=int(coverage.n_rows.sum()), screens=len(coverage), reaction_classes=int(coverage.reaction_class.nunique()), exact_matched_pairs=len(matched), supported_productive_matched_pairs=int(matched.productive_rms.notna().sum()), strict_composition_pairs=int(matched.contrast_type.isin(['identity_swap', 'solvent_only']).sum()), exclusions=[{'reaction_class': c, 'condition_id': k} for c, k in sorted(AMBIGUOUS)], definitions={'exact_match': 'Exactly one of 15 ID/dose/solvent/temperature/time core fields differs; identity swaps require both identities and unchanged nonmissing dose. Dose-only and physical-field changes are separated.', 'recipe_primary': 'Fraction of within-screen varying blocks differing: six (ID,equiv) reagent slot blocks plus solvent ID, temperature, time. No name-based chemistry attribution.', 'recipe_slot_invariance': 'Slot-invariant minimum multiset edit count: max(total ingredient count A,B) minus ID/equiv multiset intersection, plus physical-field mismatches; divided by the SAME number of variable blocks as primary. Holds overall block weighting fixed.', 'recipe_domain_weight_sensitivity': 'Mean over varying domains: slot-invariant multiset Jaccard of (ID,equiv) ingredients, categorical solvent, categorical temperature, categorical time. Reweights whole reagent composition as one domain; distinct from pure slot invariance. Physical fields are categorical, not interval scaled.', 'productive': 'Observed screen row maximum >= 0.20 after ambiguity exclusions; descriptive response-selected subset.', 'pair_active': 'At least one of the paired yields >= 0.05; descriptive response-selected sensitivity.', 'overlap': 'Pairwise finite overlap only; >=10 rows required for distances and decomposition, >=5 for Pearson/Spearman and nonconstant vectors.', 'interaction': 'Alternating observed-cell row and column mean additive fit; original missing values remain missing.', 'row_z': 'Productive rows with within-screen population SD > 0.01; subtract observed row mean and divide by observed row SD.', 'generic_adjustment': 'For each productive overlap, regress paired yield difference on intercept and that substrate row mean excluding BOTH tested conditions. In-sample descriptive residual, not predictive accuracy.', 'remaining_fraction': 'Residual mean squared paired difference divided by original mean squared paired difference; undefined for zero original difference.', 'mantel': 'Spearman distance correlation; permute complete condition labels jointly across response rows/columns; +1 two-sided Monte Carlo p; no independent-edge t test. Missing distance pattern is permuted too.', 'multiple_testing': 'BH across screens separately within each prespecified response view and recipe encoding; exploratory, not correction across every analysis.'}, limits=['Registry slots are operational fields, not validated catalyst/base/ligand roles.', 'No atom-level descriptors, chemical similarity model, mechanistic or causal attribution.', 'Conditions, matched edges and repeated CMC screens are dependent; report screen-level tests and descriptive reaction-class summaries.', 'Permutation exchangeability is an association reference, not randomization of experiments or batch adjustment.', 'Missing yields, response-selected productive subsets and screened library design can affect distances.', 'Does not predict unmeasured conditions, unseen conditions, or cross-laboratory behavior.'], verification={'raw_workbook_hashes_match_audit': True, 'all_recipe_core_fingerprints_match': True, 'all_matches_single_core_field': True, 'pairwise_sse_decomposition': True, 'residual_sse_not_increased': True, 'mantel_observed_matches_scipy': True, 'all_sources_hash_unchanged': True})
    (HERE / 'data/chemical_response_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(associations[associations.recipe_distance.eq('slot_block_hamming')].groupby('response_view').agg(median_rho=('rho', 'median'), nominal_positive=('permutation_p_positive', lambda x: int((x < 0.05).sum())), bh_significant=('bh_q_two_sided', lambda x: int((x < 0.05).sum()))).to_string())
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
