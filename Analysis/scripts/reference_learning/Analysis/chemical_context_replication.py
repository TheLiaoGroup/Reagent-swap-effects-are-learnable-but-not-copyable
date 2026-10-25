"""Substrate-split and cross-context replication of strict recipe contrasts.

Existing analysis data only. Contrasts are descriptive: no replicate-well error
model and no randomized component intervention is available in these matrices."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from chemical_response_structure import AMBIGUOUS, dumps, sha256
HERE = Path(__file__).resolve().parent
SEED = 2026101617
MIN_HALF = 10
SIGN_FLOOR = 0.01

def finite_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan

def finite_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan

def corr(x, y, minimum=MIN_HALF):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < minimum:
        return np.nan
    a, b = (x[good], y[good])
    a, b = (a - a.mean(), b - b.mean())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-12 else np.nan

def oriented_ids(row):
    low, high = sorted((str(row.value_a), str(row.value_b)))
    assert low != high
    return (low, high, 1 if str(row.value_a) == high else -1)

def enrich_edges(frame):
    frame = frame[frame.contrast_type.isin(['identity_swap', 'solvent_only'])].copy()
    rows = []
    for row in frame.itertuples(index=False):
        low, high, sign = oriented_ids(row)
        fixed = json.loads(row.fixed_core_fields_json)
        changed = json.loads(row.changed_fields_json)
        assert len(changed) == 1 and len(fixed) == 14
        context = dumps([row.contrast_category, fixed])
        swap = dumps([row.reaction_class, row.contrast_category, str(row.fixed_equiv), low, high])
        domain = 'solvent' if row.contrast_type == 'solvent_only' else 'reagent'
        domain_swap = dumps([row.reaction_class, domain, str(row.fixed_equiv), low, high])
        rows.append(dict(screen=row.screen, reaction_class=row.reaction_class, condition_a=row.condition_a, condition_b=row.condition_b, category=row.contrast_category, contrast_type=row.contrast_type, dose=str(row.fixed_equiv), low_id=low, high_id=high, low_name=row.name_b if sign == 1 else row.name_a, high_name=row.name_a if sign == 1 else row.name_b, orientation=sign, context_id=hashlib.sha256(context.encode()).hexdigest()[:16], context_json=context, swap_key=swap, domain_swap_key=domain_swap))
    out = pd.DataFrame(rows)
    assert not out.duplicated(['screen', 'condition_a', 'condition_b']).any()
    assert not out.duplicated(['screen', 'swap_key', 'context_id']).any()
    return out

def split_half(labels, rng):
    """Shuffle intact groups, then greedily balance substrate-row counts."""
    labels = np.asarray(labels).astype(str)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return None
    order = rng.permutation(len(unique))
    selected, loads = ([[], []], [0, 0])
    for index in order:
        side = int(loads[1] < loads[0])
        selected[side].append(unique[index])
        loads[side] += int(counts[index])
    a = np.isin(labels, selected[0])
    b = ~a
    assert a.any() and b.any() and (not np.any(a & b))
    assert not set(labels[a]) & set(labels[b])
    return (a, b)

def other_activity(m, excluded):
    keep = np.ones(m.shape[1], dtype=bool)
    keep[np.unique(excluded)] = False
    if not keep.any():
        return (np.full(len(m), np.nan), np.full(len(m), np.nan))
    other = m[:, keep]
    count = np.isfinite(other).sum(axis=1)
    g = np.divide(np.nansum(other, axis=1), count, out=np.full(len(m), np.nan), where=count > 0)
    maximum = np.max(np.where(np.isfinite(other), other, -np.inf), axis=1)
    maximum[count == 0] = np.nan
    return (g, maximum)

def crossfit_residual(delta, activity, halves, eligible):
    """Each row's nuisance regression is fit exclusively on the opposite half."""
    out = np.full(len(delta), np.nan)
    for train, test in (halves, halves[::-1]):
        tr = train & eligible & np.isfinite(delta) & np.isfinite(activity)
        te = test & eligible & np.isfinite(delta) & np.isfinite(activity)
        if tr.sum() < MIN_HALF or te.sum() < MIN_HALF:
            continue
        design = np.column_stack((np.ones(tr.sum()), activity[tr]))
        coefficient = np.linalg.lstsq(design, delta[tr], rcond=None)[0]
        out[te] = delta[te] - coefficient[0] - coefficient[1] * activity[te]
    return out

def center_halves(vector, halves):
    out = vector.copy()
    for half in halves:
        use = half & np.isfinite(out)
        if use.any():
            out[use] -= out[use].mean()
    return out

def context_mean_metrics(edges, delta, halves):
    a, b = halves
    means = []
    for k, edge in enumerate(edges.itertuples(index=False)):
        va, vb = (delta[a, k], delta[b, k])
        na, nb = (np.isfinite(va).sum(), np.isfinite(vb).sum())
        if min(na, nb) < MIN_HALF:
            continue
        ma, mb = (float(np.nanmean(va)), float(np.nanmean(vb)))
        informative = abs(ma) >= SIGN_FLOOR and abs(mb) >= SIGN_FLOOR
        mse_zero = 0.5 * (np.nanmean(va ** 2) + np.nanmean(vb ** 2))
        mse_transferred = 0.5 * (np.nanmean((va - mb) ** 2) + np.nanmean((vb - ma) ** 2))
        means.append(dict(context_id=edge.context_id, mean_a=ma, mean_b=mb, mean_half_abs_gap=abs(ma - mb), mse_zero=mse_zero, mse_transferred_mean=mse_transferred, sign_agreement=float(ma * mb > 0) if informative else np.nan, sign_eligible=int(informative), n_a=int(na), n_b=int(nb)))
    if not means:
        return None
    estimates = pd.DataFrame(means)
    contexts = estimates.groupby('context_id').agg(mean_half_abs_gap=('mean_half_abs_gap', 'median'), mse_zero=('mse_zero', 'mean'), mse_transferred_mean=('mse_transferred_mean', 'mean'), sign_agreement=('sign_agreement', 'mean'), sign_eligible=('sign_eligible', 'sum'), edges=('mean_a', 'size'))
    return dict(n_eligible_edges=len(estimates), n_eligible_contexts=len(contexts), n_sign_eligible_edges=int(estimates.sign_eligible.sum()), n_sign_eligible_contexts=int(contexts.sign_agreement.notna().sum()), context_mean_abs_half_gap=float(contexts.mean_half_abs_gap.mean()), context_mean_sign_agreement=finite_mean(contexts.sign_agreement), context_equal_transferred_mean_mse=float(contexts.mse_transferred_mean.mean()), context_equal_zero_mse=float(contexts.mse_zero.mean()), context_equal_transferred_mean_gain=1 - contexts.mse_transferred_mean.mean() / contexts.mse_zero.mean() if contexts.mse_zero.mean() > 1e-16 else np.nan, edge_mean_effect_correlation=corr(estimates.mean_a.to_numpy(), estimates.mean_b.to_numpy(), minimum=5))

def summarize_transport(frame):
    out = []
    keys = ['screen', 'scheme', 'repeat', 'subset', 'response']
    for group_key, group in frame.groupby(keys, sort=True):
        supported = group[group.eligible.eq(1)]
        valid = supported[supported.rho_a.notna() & supported.rho_b.notna()]
        swaps = valid.groupby('swap_key').agg(rho_a=('rho_a', 'mean'), rho_b=('rho_b', 'mean'), both_positive=('both_positive', 'mean'), half_abs_gap=('half_abs_gap', 'mean'))
        out.append(dict(zip(keys, group_key), n_candidate_context_pairs=len(group), n_overlap_eligible_context_pairs=len(supported), n_valid_context_pairs=len(valid), n_valid_swaps=len(swaps), screen_swap_mean_rho_a=finite_mean(swaps.rho_a), screen_swap_mean_rho_b=finite_mean(swaps.rho_b), screen_swap_mean_rho=finite_mean(np.r_[swaps.rho_a, swaps.rho_b]), screen_swap_fraction_both_positive=finite_mean(swaps.both_positive), screen_swap_mean_abs_half_rho_gap=finite_mean(swaps.half_abs_gap), context_pair_split_reliability=corr(valid.rho_a.to_numpy(), valid.rho_b.to_numpy(), minimum=5), swap_split_reliability=corr(swaps.rho_a.to_numpy(), swaps.rho_b.to_numpy(), minimum=5)))
    return pd.DataFrame(out)

def alignment_reference(pairs, vectors, permutations, rng):
    """Common row permutation of every right-hand vector, not independent edges."""
    if not pairs:
        return ([], [])
    n = len(vectors[0][0])
    null = {'raw': [], 'generic_crossfit': []}
    observed = {'raw': [], 'generic_crossfit': []}
    for index, response in enumerate(null):
        for pair, four in zip(pairs, vectors):
            observed[response].append((pair['swap_key'], corr(four[2 * index], four[2 * index + 1])))

    def statistic(items):
        table = pd.DataFrame(items, columns=['swap_key', 'rho'])
        return finite_mean(table.groupby('swap_key').rho.mean())
    for _ in range(permutations):
        permutation = rng.permutation(n)
        for index, response in enumerate(null):
            values = [(pair['swap_key'], corr(four[2 * index], four[2 * index + 1][permutation])) for pair, four in zip(pairs, vectors)]
            null[response].append(statistic(values))
    summary, draws = ([], [])
    for response in null:
        values = np.array(null[response])
        good = values[np.isfinite(values)]
        obs = statistic(observed[response])
        summary.append(dict(response=response, observed_swap_equal_mean_rho=obs, null_mean=finite_mean(values), null_q025=float(np.quantile(good, 0.025)) if len(good) else np.nan, null_q975=float(np.quantile(good, 0.975)) if len(good) else np.nan, n_valid_reference_draws=len(good), n_candidate_pairs=len(pairs), above_reference_975=int(obs > np.quantile(good, 0.975)) if len(good) and np.isfinite(obs) else 0))
        draws.extend((dict(response=response, draw=i, swap_equal_mean_rho=value) for i, value in enumerate(values)))
    return (summary, draws)

def make_figure(means, transport, reference, coverage):
    screens = coverage.screen.tolist()
    labels = [s.replace('Buchwald-Hartwig', 'BH').replace('C-O competitive reactions', 'C-O').replace('Reductive-amination', 'Red. amination') for s in screens]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), sharey=True)
    colors = {'pair': '#426b8a', 'role1': '#23836d', 'role2': '#b14e4b'}
    y = np.arange(len(screens))
    for scheme, offset in (('pair', -0.18), ('role1', 0), ('role2', 0.18)):
        one = means[means.scheme.eq(scheme)].groupby('screen').context_mean_sign_agreement.mean().reindex(screens)
        axes[0].scatter(one, y + offset, s=25, color=colors[scheme], label=scheme)
        for response, marker, fill in (('raw', 'o', True), ('generic_crossfit', 's', False)):
            subset = transport[transport.scheme.eq(scheme) & transport.subset.eq('all') & transport.response.eq(response)]
            value = subset.groupby('screen').screen_swap_mean_rho.mean().reindex(screens)
            axes[1].scatter(value, y + offset, s=26, marker=marker, facecolors=colors[scheme] if fill else 'none', edgecolors=colors[scheme])
    for response, offset, color in (('raw', -0.11, '#426b8a'), ('generic_crossfit', 0.11, '#b14e4b')):
        ref = reference[reference.response.eq(response)].set_index('screen').reindex(screens)
        axes[2].hlines(y + offset, ref.null_q025, ref.null_q975, color=color, alpha=0.55, lw=2)
        axes[2].scatter(ref.observed_swap_equal_mean_rho, y + offset, color=color, s=27, marker='o' if response == 'raw' else 's', label=response.replace('generic_crossfit', 'Activity adjusted'))
    for ax in axes:
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(0 if ax is not axes[0] else 0.5, color='#888888', lw=0.8, ls=':')
    axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 1.03), xlabel='Context-equal mean sign agreement')
    axes[0].invert_yaxis()
    axes[1].set(xlim=(-0.35, 1.03), xlabel='Swap-equal centered pattern correlation')
    axes[2].set(xlim=(-0.35, 1.03), xlabel='Observed correlation / alignment reference')
    axes[0].set_title('A  Mean-effect direction across substrate halves', fontsize=11, loc='left')
    axes[1].set_title('B  Same swap in different recipe contexts', fontsize=11, loc='left')
    axes[2].set_title('C  Common-row permutation reference', fontsize=11, loc='left')
    axes[0].legend(loc='lower left', fontsize=8, frameon=False)
    axes[2].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle('Chemical contrast replication depends on substrate grouping and recipe context', fontsize=15, y=0.98)
    fig.text(0.21, 0.035, 'A: both half means >= 1 pp in magnitude. B: filled = raw; open = cross-fitted activity adjustment.\nC: bars are 95% permutation reference ranges, not confidence intervals. Missing points indicate insufficient support.', fontsize=9)
    fig.subplots_adjust(left=0.215, right=0.985, bottom=0.12, top=0.91, wspace=0.1)
    fig.savefig(HERE / 'figures/12_chemical_context_replication.png', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--permutations', type=int, default=199)
    args = parser.parse_args()
    assert args.repeats >= 2 and args.permutations >= 19
    sources = [HERE / 'tables/chemical_response_matched_pairs.csv', HERE / 'tables/screen_inventory.csv', HERE / 'data/chemical_response_manifest.json', HERE / 'chemical_response_structure.py']
    sources += sorted((HERE / 'data').glob('matrix_*.npz'))
    sources.append(HERE / 'tables/condition_recipe_audit_details.csv')
    before = {str(path): sha256(path) for path in sources}
    inherited = json.loads((HERE / 'data/chemical_response_manifest.json').read_text(encoding='utf-8'))
    assert sha256(sources[0]) == inherited['output_hashes']['chemical_response_matched_pairs.csv']
    for path in (p for p in sources if p.suffix == '.npz'):
        assert sha256(path) == inherited['source_hashes'][str(path)]
    edges = enrich_edges(pd.read_csv(sources[0], dtype=str, keep_default_na=False))
    inventory = pd.read_csv(sources[1])
    paths = {'matrix_' + s.replace('|', '_').replace(' ', '_') + '.npz': s for s in inventory.screen}
    mean_rows, transport_rows, coverage_rows, reference_rows, null_rows, split_rows, pair_rows = ([], [], [], [], [], [], [])
    for path in sorted((HERE / 'data').glob('matrix_*.npz')):
        screen = paths[path.name]
        cls = screen.split('|')[0]
        seed = SEED + int(hashlib.sha256(screen.encode()).hexdigest()[:6], 16)
        rng = np.random.default_rng(seed)
        saved = np.load(path, allow_pickle=False)
        m, conds, sps = (saved['M'], saved['conds'].astype(str), saved['sps'].astype(str))
        assert len(sps) == len(m) and len(set(sps)) == len(m)
        excluded = [c for c in conds if (cls, c) in AMBIGUOUS]
        keep = np.array([(cls, c) not in AMBIGUOUS for c in conds])
        m, conds = (m[:, keep], conds[keep])
        lookup = {c: i for i, c in enumerate(conds)}
        one = edges[edges.screen.eq(screen)].reset_index(drop=True)
        indices = [(lookup[r.condition_a], lookup[r.condition_b]) for r in one.itertuples(index=False)]
        delta = np.column_stack([r.orientation * (m[:, i] - m[:, j]) for r, (i, j) in zip(one.itertuples(index=False), indices)])
        roles = np.array([sp.split(' >> ') for sp in sps])
        assert roles.shape == (len(m), 2)
        labels = {'pair': np.array([str(i) for i in range(len(m))]), 'role1': roles[:, 0], 'role2': roles[:, 1]}
        pairs, cache = ([], [])
        for swap_key, group in one.groupby('swap_key', sort=True):
            for i, j in itertools.combinations(group.index, 2):
                ei, ej = (one.iloc[i], one.iloc[j])
                assert ei.context_id != ej.context_id
                union = np.unique(np.r_[indices[i], indices[j]])
                assert len(union) == 4
                g, productive = other_activity(m, union)
                common = np.isfinite(delta[:, i]) & np.isfinite(delta[:, j]) & np.isfinite(g)
                pair = dict(screen=screen, swap_key=swap_key, category=ei.category, context_a=ei.context_id, context_b=ej.context_id, conditions_a=dumps(list(np.array(conds)[list(indices[i])])), conditions_b=dumps(list(np.array(conds)[list(indices[j])])), low_name=ei.low_name, high_name=ei.high_name, dose=ei.dose, n_all_common=int(common.sum()), n_productive_other_common=int((common & (productive >= 0.2)).sum()), mean_a=finite_mean(delta[common, i]), mean_b=finite_mean(delta[common, j]))
                ma, mb = (pair['mean_a'], pair['mean_b'])
                pair['mean_direction_agreement_1pp'] = float(ma * mb > 0) if min(abs(ma), abs(mb)) >= SIGN_FLOOR else np.nan
                pair_rows.append(pair)
                pairs.append(pair)
                cache.append((i, j, g, productive, common))
        canonical_vectors = None
        for scheme, group_labels in labels.items():
            for repeat in range(args.repeats):
                halves = split_half(group_labels, rng)
                if halves is None:
                    split_rows.append(dict(screen=screen, scheme=scheme, repeat=repeat, eligible=0, reason='fewer_than_two_unique_groups'))
                    continue
                a, b = halves
                split_rows.append(dict(screen=screen, scheme=scheme, repeat=repeat, eligible=1, reason='', n_a=int(a.sum()), n_b=int(b.sum()), n_groups_a=len(set(group_labels[a])), n_groups_b=len(set(group_labels[b])), groups_a_json=dumps(sorted(set(group_labels[a]))), groups_b_json=dumps(sorted(set(group_labels[b])))))
                summary = context_mean_metrics(one, delta, halves)
                if summary:
                    mean_rows.append(dict(screen=screen, scheme=scheme, repeat=repeat, **summary))
                vectors = []
                for pair, (i, j, g, productive, common) in zip(pairs, cache):
                    for subset, eligible in (('all', common), ('productive_other', common & (productive >= 0.2))):
                        na, nb = (int((eligible & a).sum()), int((eligible & b).sum()))
                        nga, ngb = (len(set(group_labels[eligible & a])), len(set(group_labels[eligible & b])))
                        supported = min(na, nb) >= MIN_HALF
                        raw_i, raw_j = (np.where(eligible, delta[:, i], np.nan), np.where(eligible, delta[:, j], np.nan))
                        if supported:
                            ri = crossfit_residual(delta[:, i], g, halves, eligible)
                            rj = crossfit_residual(delta[:, j], g, halves, eligible)
                        else:
                            ri, rj = (np.full(len(m), np.nan), np.full(len(m), np.nan))
                        for response, vi, vj in (('raw', raw_i, raw_j), ('generic_crossfit', ri, rj)):
                            ra, rb = (corr(vi[a], vj[a]), corr(vi[b], vj[b])) if supported else (np.nan, np.nan)
                            valid = np.isfinite(ra) and np.isfinite(rb)
                            transport_rows.append(dict(screen=screen, swap_key=pair['swap_key'], context_a=pair['context_a'], context_b=pair['context_b'], scheme=scheme, repeat=repeat, subset=subset, response=response, n_a=na, n_b=nb, n_groups_a=nga, n_groups_b=ngb, eligible=int(supported), reason='' if valid else 'low_overlap' if not supported else 'constant_response', rho_a=ra, rho_b=rb, both_positive=float(ra > 0 and rb > 0) if valid else np.nan, half_abs_gap=abs(ra - rb) if valid else np.nan))
                        if scheme == 'role1' and repeat == 0 and (subset == 'all'):
                            if not supported:
                                raw_i, raw_j = (np.full(len(m), np.nan), np.full(len(m), np.nan))
                            vectors.append(tuple((center_halves(v, halves) for v in (raw_i, raw_j, ri, rj))))
                if scheme == 'role1' and repeat == 0:
                    canonical_vectors = vectors
        if canonical_vectors is not None and pairs:
            reference, null = alignment_reference(pairs, canonical_vectors, args.permutations, rng)
            reference_rows.extend((dict(screen=screen, **row) for row in reference))
            null_rows.extend((dict(screen=screen, **row) for row in null))
        slot_groups = one.groupby('swap_key').size()
        domain_groups = one.groupby('domain_swap_key').size()
        slot_pairs = sum((n * (n - 1) // 2 for n in slot_groups))
        domain_pairs = sum((n * (n - 1) // 2 for n in domain_groups))
        coverage_rows.append(dict(screen=screen, reaction_class=cls, n_rows=len(m), n_conditions=len(conds), excluded_ambiguous_conditions_json=dumps(excluded), n_edges=len(one), n_contexts=one.context_id.nunique(), n_distinct_swaps=len(slot_groups), n_repeated_swaps=int((slot_groups >= 2).sum()), n_candidate_context_pairs=slot_pairs, n_domain_candidate_context_pairs=domain_pairs, n_added_cross_slot_pairs=domain_pairs - slot_pairs, unique_role1=len(set(labels['role1'])), unique_role2=len(set(labels['role2'])), transport_reason='eligible_candidates' if slot_pairs else 'no_same_swap_across_contexts'))
        print(f'{screen}: {len(one)} edges; {one.context_id.nunique()} contexts; {len(pairs)} repeated-swap context pairs', flush=True)
    means, transport, coverage = (pd.DataFrame(mean_rows), pd.DataFrame(transport_rows), pd.DataFrame(coverage_rows))
    reference, null, splits, pairs = (pd.DataFrame(reference_rows), pd.DataFrame(null_rows), pd.DataFrame(split_rows), pd.DataFrame(pair_rows))
    summary = summarize_transport(transport)
    mean_aggregate = means.groupby(['screen', 'scheme'], as_index=False).mean(numeric_only=True)
    transport_aggregate = summary.groupby(['screen', 'scheme', 'subset', 'response'], as_index=False).mean(numeric_only=True)
    family_source = transport_aggregate.copy()
    family_source['family'] = family_source.screen.str.split('|', regex=False).str[0].replace({'Buchwald-Hartwig(CMC)': 'Buchwald-Hartwig'})
    family = family_source.groupby(['family', 'scheme', 'subset', 'response'], as_index=False).mean(numeric_only=True)
    family['n_screens'] = family_source.groupby(['family', 'scheme', 'subset', 'response']).screen.nunique().to_numpy()
    outputs = {'chemical_context_replication_edges.csv': edges, 'chemical_context_replication_coverage.csv': coverage, 'chemical_context_replication_splits.csv': splits, 'chemical_context_replication_mean_stability.csv': means, 'chemical_context_replication_mean_screen_aggregate.csv': mean_aggregate, 'chemical_context_replication_context_pairs.csv': pairs, 'chemical_context_replication_transport.csv': transport, 'chemical_context_replication_screen_summary.csv': summary, 'chemical_context_replication_screen_aggregate.csv': transport_aggregate, 'chemical_context_replication_family_aggregate.csv': family, 'chemical_context_replication_alignment_reference.csv': reference, 'chemical_context_replication_alignment_draws.csv': null}
    for name, frame in outputs.items():
        frame.to_csv(HERE / 'tables' / name, index=False, encoding='utf-8-sig')
    make_figure(means, summary, reference, coverage)
    assert len(coverage) == 19 and coverage.n_rows.sum() == 2454
    assert len(edges) == 1846 and coverage.n_contexts.sum() == 400
    assert coverage.n_distinct_swaps.sum() == 1425
    assert len(pairs) == 902 and coverage.n_candidate_context_pairs.sum() == len(pairs)
    assert all((set(json.loads(row.groups_a_json)).isdisjoint(json.loads(row.groups_b_json)) for row in splits[splits.eligible.eq(1)].itertuples(index=False)))
    after = {str(path): sha256(path) for path in sources}
    assert before == after, 'An input changed during analysis'
    manifest = dict(date='2026-10-16', seed=SEED, repeats=args.repeats, permutations=args.permutations, source_hashes=before, code_sha256=sha256(Path(__file__)), test_code_sha256=sha256(HERE / 'test_chemical_context_replication.py'), output_hashes={name: sha256(HERE / 'tables' / name) for name in outputs}, figure_sha256=sha256(HERE / 'figures/12_chemical_context_replication.png'), report_sha256=sha256(HERE / '17_CHEMICAL_CONTEXT_REPLICATION_2026-10-16.md') if (HERE / '17_CHEMICAL_CONTEXT_REPLICATION_2026-10-16.md').exists() else None, inputs_unchanged=True, screens=18, analysis_rows=2360, edges=1808, contexts=384, screen_specific_swaps=1394, candidate_context_pairs=895, definitions={'orientation': 'Registry lexically higher ID minus lower ID; a bookkeeping direction with no chemical ordering.', 'same_swap': 'Within reaction_class, same registry slot, same unchanged dose, same unordered registry identity pair; solvent is its own category.', 'mean_stability': 'Train/test halves disjoint by pair or the selected substrate role; compare raw contrast means, sign counted only when both means have magnitude >=1 pp; edge->fixed context mean->screen mean. Transfer each training-half mean to every opposite-half contrast, average the two half MSEs per edge, then edge->equal context; gain=1-MSE_transferred_mean/MSE_zero. This predicts only the average effect and does not predict substrate-specific residuals.', 'transport': 'Same swap in different fixed-other-field recipe contexts; Pearson correlation centers both response contrasts across substrates, and therefore does not count a shared mean sign as pattern agreement.', 'generic_crossfit': 'Other-condition mean excludes the union of all four contrasted columns. Fit delta=alpha+beta*g on opposite substrate half, and calculate residual on each held-out half. Both half nuisance fits are independent of their evaluated contrast outcomes.', 'productive_other': 'Other-condition maximum >=20%, excluding all four contrasted columns; a response-selected descriptive sensitivity, not prospective anchor availability.', 'aggregation': 'Context pairs->equal swap averages->screen; repeated random/group splits averaged within screen. Cross-screen conclusions use screen and merged-family summaries. Never 895 or1808 independent replicates.', 'alignment_reference': 'Canonical first role1 split, >=10 common rows in each half. Each vector centered separately within each half before pooling; avoids counting half-level nuisance-fit mean shifts. Common row permutations of every right-hand context vector per screen; all left vectors unchanged. Original residual vector values and missingness move together. Summary is equal-swap mean correlation. A structured alignment reference, not a randomized-experiment p-value or sampling confidence interval.', 'minimum_overlap': MIN_HALF}, limitations=['No repeated-well technical-error estimate; finite-substrate variability, context-dependent chemistry, measurement noise and batch cannot be uniquely separated.', 'Role-wise halves exclude overlap in one substrate role only; the other role can overlap. No claim of both-reactants-disjoint validation.', 'Missingness and constant contrasts cause extra metric-specific exclusions.', 'Cross-fitted activity adjustment uses many other observed columns and is not a prospective limited-anchor prediction.', 'Registry slots are operational, not validated mechanistic roles; cross-slot/domain coverage is audited separately.', 'The same recipe conditions and substrate rows support many edges; splits are repeated dependent reuses, not independent experiments.', 'No causal or mechanistic attribution, no new-library or new-laboratory generalization.', "Does not repair report13's negative result for prediction of extra condition-specific gain."], verification={'all_original_input_hashes_unchanged': True, 'inherited_contrast_and_matrix_hashes_match': True, 'all_half_groups_disjoint': True, 'all_context_pairs_four_distinct_conditions': True, 'strict_composition_denominator_reconciled': True})
    (HERE / 'data/chemical_context_replication_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.groupby(['scheme', 'subset', 'response']).screen_swap_mean_rho.median().to_string(), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
