"""Report60: frozen-test identity alignment and secondary source reassignment."""
from __future__ import annotations
import sys as _reader_sys
from pathlib import Path as _ReaderPath
_reader_sys.path.insert(0, str(next((p for p in _ReaderPath(__file__).resolve().parents if (p / 'reader_tables.py').is_file()))))
import reader_tables
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
TABLES, DATA, FIGURES = (HERE / 'tables', HERE / 'data', HERE / 'figures')
PREFIX = 'source_identity_alignment'
VERSION = 'source-identity-alignment-v1'
REPORT = HERE / '60_SOURCE_IDENTITY_ALIGNMENT_2026-10-17.md'
FIGURE = FIGURES / '47_SOURCE_IDENTITY_ALIGNMENT.png'
PAIR = ['source_screen', 'target_screen', 'source_family', 'target_family', 'relation']
KEYS = PAIR + ['design', 'budget', 'model']
MODELS = ('source_offset', 'source_affine_ridge')
BUDGETS = (24, 48, 96, 192)
N_DRAWS, ALPHA, EPS = (128, 3.0, 1e-12)
PRIMARY_METRICS = ('real_mse', 'target_mean_mse', 'test_permuted_exact_mse', 'delta_test', 'net_gain')
SECONDARY_METRICS = ('real_mse', 'target_mean_mse', 'mean_permuted_mse', 'delta_pair', 'net_gain')

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixed_mappings(source, target, identities):
    identities = list(identities)
    if len(identities) < 6:
        raise ValueError('at least six identities required for 128 unique nonidentity mappings')
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError('canonical identities must be unique and sorted')
    result, seen, candidate = ([], {tuple(range(len(identities)))}, 0)
    while len(result) < N_DRAWS:
        order = sorted(range(len(identities)), key=lambda index: (hashlib.sha256(dumps([VERSION, source, target, candidate, identities[index]]).encode('utf-8')).hexdigest(), identities[index]))
        permutation = tuple(order)
        if permutation not in seen:
            assigned = [identities[index] for index in order]
            signature = hashlib.sha256(dumps([VERSION, source, target, assigned]).encode('utf-8')).hexdigest()
            result.append({'draw': len(result), 'candidate': candidate, 'mapping_sha256': signature, 'permutation': order})
            seen.add(permutation)
        candidate += 1
    return result

def fit_batch(source_all, source_calibration, target_calibration, source_test):
    """Closed form of the frozen standardized alpha=3 Ridge; no TEST y input."""
    source_all, xc, yc, xt = [np.asarray(value, dtype=float) for value in (source_all, source_calibration, target_calibration, source_test)]
    if xc.ndim != 2 or xt.ndim != 2 or len(xc) != len(xt) or (xc.shape[1] < 3) or (len(yc) != xc.shape[1]):
        raise ValueError('matched batches and at least three calibration identities required')
    xmean, ymean = (xc.mean(axis=1), float(yc.mean()))
    sd = float(source_all.std())
    if sd <= EPS:
        slope = np.zeros(len(xc))
    else:
        centered = xc - xmean[:, None]
        slope = centered @ (yc - ymean) / (np.sum(centered ** 2, axis=1) + ALPHA * sd ** 2)
    intercept = ymean - slope * xmean
    offset = ymean - xmean
    return {'source_affine_ridge': {'prediction': intercept[:, None] + slope[:, None] * xt, 'intercept': intercept, 'slope': slope}, 'source_offset': {'prediction': offset[:, None] + xt, 'intercept': offset, 'slope': np.ones(len(xc))}}

def exact_test_reference(target_test, source_test, intercept, slope):
    y, x = (np.asarray(target_test, dtype=float), np.asarray(source_test, dtype=float))
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    actual = float(np.mean((y - intercept - slope * x) ** 2))
    reference = float(np.mean((y - intercept - slope * x.mean()) ** 2) + slope ** 2 * x.var())
    advantage = reference - actual
    return {'real_mse': actual, 'test_permuted_exact_mse': reference, 'delta_test': advantage, 'test_source_mean': float(x.mean()), 'test_source_variance': float(x.var()), 'test_target_mean': float(y.mean()), 'test_covariance': covariance, 'delta_test_covariance_identity_error': advantage - 2 * slope * covariance, 'n_distinct_test_source_values': len(set(x))}

def build_mappings(support, designs):
    registry, mappings, cache, audits = ([], [], {}, [])
    pair_rows = designs[PAIR].drop_duplicates().sort_values(PAIR)
    groups = {key: group.sort_values('unit_identity').reset_index(drop=True) for key, group in support.groupby(PAIR, sort=True)}
    for pair in pair_rows.itertuples(index=False, name=None):
        key = dict(zip(PAIR, pair))
        group = groups.get(pair, support.iloc[:0])
        identities = group.unit_identity.tolist()
        source = group.source_value.to_numpy(float)
        n = len(identities)
        eligible = n >= 6
        status = 'mapped' if eligible else 'geometry_absent' if n == 0 else 'insufficient_identity_support'
        local = fixed_mappings(pair[0], pair[1], identities) if eligible else []
        assigned_values = {tuple(source[row['permutation']]) for row in local}
        registry.append({**key, 'mapping_status': status, 'n_identities': n, 'n_mappings': len(local), 'n_distinct_assigned_source_vectors': len(assigned_values), 'all_source_zero': bool(n and np.max(np.abs(source)) <= EPS), 'all_source_constant': bool(n and np.ptp(source) <= EPS)})
        if eligible:
            permutations = np.array([row['permutation'] for row in local], dtype=int)
            cache[pair] = (group, local, source[permutations])
            for draw in local:
                for identity, assigned in zip(identities, [identities[index] for index in draw['permutation']]):
                    mappings.append({**key, **{name: draw[name] for name in ('draw', 'candidate', 'mapping_sha256')}, 'unit_identity': identity, 'assigned_source_identity': assigned})
        if n:
            coordinates = {}
            records = {}
            for role in ('source', 'target'):
                coordinates[role] = set((tuple(value) for text in group[role + '_coordinates_json'] for value in json.loads(text)))
                records[role] = set((value for text in group[role + '_record_ids_json'] for value in json.loads(text)))
            audits.append({**key, 'n_identities': n, 'source_coordinate_count': len(coordinates['source']), 'target_coordinate_count': len(coordinates['target']), 'all_source_target_coordinate_overlap': len(coordinates['source'] & coordinates['target']), 'all_source_target_record_overlap': len(records['source'] & records['target'])})
    audit = pd.DataFrame(audits)
    if audit[['all_source_target_coordinate_overlap', 'all_source_target_record_overlap']].to_numpy().any():
        raise ValueError('whole source response support intersects target labels')
    return (pd.DataFrame(registry), pd.DataFrame(mappings), cache, audit)

def build_scores(support, budgets, original_scores, cache):
    primary, secondary, draws = ([], [], [])
    score_lookup = original_scores.set_index(KEYS)
    fixed_lookup = budgets[budgets.budget.eq(24)].set_index(PAIR + ['design']).status.eq('scored')
    lookup = {key: group.set_index('unit_identity').sort_index() for key, group in support.groupby(PAIR, sort=True)}
    for budget in budgets.to_dict('records'):
        pair = tuple((budget[name] for name in PAIR))
        base = {name: budget[name] for name in PAIR + ['design', 'budget', 'status', 'n_primary_identities', 'n_test', 'n_calibration', 'actual_spent', 'unused_budget']}
        base['b24_scored_domain'] = bool(fixed_lookup.loc[pair + (budget['design'],)])
        if budget['status'] == 'scored':
            units = lookup[pair]
            calibration, test = (json.loads(budget['calibration_identities_json']), json.loads(budget['test_identities_json']))
            all_x = units.source_value.to_numpy(float)
            xc, yc = (units.loc[calibration, 'source_value'].to_numpy(float), units.loc[calibration, 'target_value'].to_numpy(float))
            xt, yt = (units.loc[test, 'source_value'].to_numpy(float), units.loc[test, 'target_value'].to_numpy(float))
            real_fit = fit_batch(all_x, xc[None, :], yc, xt[None, :])
            target_mean_mse = float(np.mean((yt - yc.mean()) ** 2))
            group, local, assigned = cache[pair]
            identity_index = {identity: index for index, identity in enumerate(group.unit_identity)}
            ci, ti = ([identity_index[identity] for identity in calibration], [identity_index[identity] for identity in test])
            permuted_fit = fit_batch(all_x, assigned[:, ci], yc, assigned[:, ti])
        for model in MODELS:
            key = pair + (budget['design'], budget['budget'], model)
            original = score_lookup.loc[key]
            item = {**base, 'model': model}
            if budget['status'] != 'scored':
                primary.append({**item, **{name: np.nan for name in ('intercept', 'slope', 'source_only_mean', 'source_only_sd', *PRIMARY_METRICS, 'test_source_mean', 'test_source_variance', 'test_target_mean', 'test_covariance', 'delta_test_covariance_identity_error', 'n_distinct_test_source_values')}})
                secondary.append({**item, **{name: np.nan for name in SECONDARY_METRICS}, 'n_draws': 0})
                continue
            parameters = real_fit[model]
            a, b = (float(parameters['intercept'][0]), float(parameters['slope'][0]))
            if not np.allclose([a, b], [original.intercept, original.slope], rtol=1e-09, atol=1e-12):
                raise AssertionError('equivalent ridge/offset does not match frozen report57')
            reference = exact_test_reference(yt, xt, float(original.intercept), float(original.slope))
            if not np.isclose(reference['real_mse'], original.mse, rtol=1e-09, atol=1e-12):
                raise AssertionError('frozen real score changed')
            primary.append({**item, 'intercept': float(original.intercept), 'slope': float(original.slope), 'source_only_mean': float(all_x.mean()), 'source_only_sd': float(all_x.std()), **reference, 'target_mean_mse': target_mean_mse, 'net_gain': target_mean_mse - reference['real_mse']})
            fit = permuted_fit[model]
            mse = np.mean((fit['prediction'] - yt[None, :]) ** 2, axis=1)
            for index, draw in enumerate(local):
                draws.append({**{name: item[name] for name in PAIR + ['design', 'budget', 'model', 'b24_scored_domain', 'n_test', 'n_calibration', 'actual_spent']}, **{name: draw[name] for name in ('draw', 'candidate', 'mapping_sha256')}, 'permuted_mse': float(mse[index]), 'real_mse': reference['real_mse'], 'target_mean_mse': target_mean_mse, 'pair_advantage': float(mse[index] - reference['real_mse']), 'permuted_intercept': float(fit['intercept'][index]), 'permuted_slope': float(fit['slope'][index])})
            secondary.append({**item, 'real_mse': reference['real_mse'], 'target_mean_mse': target_mean_mse, 'mean_permuted_mse': float(mse.mean()), 'delta_pair': float(mse.mean() - reference['real_mse']), 'net_gain': target_mean_mse - reference['real_mse'], 'n_draws': N_DRAWS})
    return (pd.DataFrame(primary), pd.DataFrame(secondary), pd.DataFrame(draws))

def aggregate_designs(scores, secondary=False):
    metrics = SECONDARY_METRICS if secondary else PRIMARY_METRICS
    delta = 'delta_pair' if secondary else 'delta_test'
    direction_rows = []
    for domain in ('all_scored', 'b24_fixed'):
        for key, group in scores.groupby(PAIR + ['budget', 'model'], sort=True):
            in_domain = group if domain == 'all_scored' else group[group.b24_scored_domain]
            finite = in_domain[in_domain.status.eq('scored')]
            status = 'scored' if len(finite) else 'outside_b24_scored_designs' if domain == 'b24_fixed' and (not len(in_domain)) else str(group.status.iloc[0])
            item = {**dict(zip(PAIR + ['budget', 'model'], key)), 'domain': domain, 'n_registered_designs': len(group), 'n_domain_designs': len(in_domain), 'n_scored_designs': len(finite), 'status': status}
            item.update({name: float(finite[name].mean()) if len(finite) else np.nan for name in (*metrics, 'actual_spent', 'n_calibration', 'n_test')})
            direction_rows.append(item)
    directions = pd.DataFrame(direction_rows)
    rows = []
    for key, group in directions.groupby(['domain', 'relation', 'budget', 'model'], sort=True):
        finite = group[group.status.eq('scored')]
        item = {**dict(zip(('domain', 'relation', 'budget', 'model'), key)), 'n_registered_directions': len(group), 'n_scored_directions': len(finite), 'n_scored_designs': int(finite.n_scored_designs.sum()), 'positive_' + delta + '_directions': int(finite[delta].gt(EPS).sum()), 'positive_net_directions': int(finite.net_gain.gt(EPS).sum()), 'positive_delta_without_net_directions': int((finite[delta].gt(EPS) & finite.net_gain.le(EPS)).sum()), 'neutral_' + delta + '_directions': int(finite[delta].abs().le(EPS).sum())}
        item.update({name: float(finite[name].mean()) for name in (*metrics, 'actual_spent', 'n_calibration', 'n_test')})
        for name in metrics:
            if name.endswith('_mse'):
                item[name[:-4] + '_rmse_pp'] = 100 * np.sqrt(item[name])
        rows.append(item)
    return (directions, pd.DataFrame(rows))

def build_coverage(budgets):
    rows = []
    for (relation, budget), group in budgets.groupby(['relation', 'budget'], sort=True):
        scored = group[group.status.eq('scored')]
        item = {'relation': relation, 'budget': budget, 'n_registered_directions': len(group[PAIR].drop_duplicates()), 'n_registered_designs': len(group), 'n_scored_directions': len(scored[PAIR].drop_duplicates()), 'n_scored_designs': len(scored)}
        item.update({'n_' + status: int(group.status.eq(status).sum()) for status in sorted(budgets.status.unique())})
        rows.append(item)
    return pd.DataFrame(rows)

def plot_figure(primary, secondary, directions):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {'source_affine_ridge': '#286f8a', 'source_offset': '#bb4b59'}
    for model in MODELS:
        group = primary[primary.domain.eq('b24_fixed') & primary.relation.eq('cross_family') & primary.model.eq(model)].sort_values('budget')
        label = 'affine ridge' if model == 'source_affine_ridge' else 'offset'
        axes[0, 0].plot(np.arange(4), 10000.0 * group.delta_test, marker='o', color=colors[model], label=label)
        axes[0, 1].plot(np.arange(4), 10000.0 * group.net_gain, marker='s', color=colors[model], label=label)
    axes[0, 0].set_title('Exact TEST identity alignment')
    axes[0, 0].set_ylabel('R exact permutation - R real (yield pp squared)')
    axes[0, 1].set_title('Same-cost net reuse')
    axes[0, 1].set_ylabel('R target mean - R real (yield pp squared)')
    for ax in axes[0]:
        ax.set_xticks(np.arange(4), BUDGETS)
        ax.set_xlabel('Target calibration coordinate budget')
        ax.axhline(0, color='#888888', linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    local = directions[directions.domain.eq('b24_fixed') & directions.relation.eq('cross_family') & directions.budget.eq(192) & directions.model.eq('source_affine_ridge') & directions.status.eq('scored')]
    axes[1, 0].scatter(10000.0 * local.delta_test, 10000.0 * local.net_gain, c='#286f8a', s=32)
    axes[1, 0].axhline(0, color='#888888', linewidth=0.8)
    axes[1, 0].axvline(0, color='#888888', linewidth=0.8)
    axes[1, 0].set_xlabel('Exact TEST alignment advantage (yield pp squared)')
    axes[1, 0].set_ylabel('Net gain vs target mean (yield pp squared)')
    axes[1, 0].set_title('All 21 fixed-domain directions; affine, budget192')
    for model in MODELS:
        group = secondary[secondary.domain.eq('b24_fixed') & secondary.relation.eq('cross_family') & secondary.model.eq(model)].sort_values('budget')
        axes[1, 1].plot(np.arange(4), 10000.0 * group.delta_pair, marker='o', color=colors[model], label='affine ridge' if model == 'source_affine_ridge' else 'offset')
    axes[1, 1].set_title('Secondary: all-source reassignment and refit')
    axes[1, 1].set_ylabel('Mean R reassigned - R real (yield pp squared)')
    axes[1, 1].set_xlabel('Target calibration coordinate budget')
    axes[1, 1].set_xticks(np.arange(4), BUDGETS)
    axes[1, 1].axhline(0, color='#888888', linewidth=0.8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

def render_report(primary, secondary, mapping_registry, coverage):
    fixed = primary[primary.domain.eq('b24_fixed') & primary.relation.eq('cross_family') & primary.model.eq('source_affine_ridge')].sort_values('budget')
    top = fixed[fixed.budget.eq(192)].iloc[0]
    lines = ['# 60 Source-identity correspondence and net reuse under equal-cost calibration', '', 'Date: 2026-10-17. Retrospective analysis added after the results of report 57 were seen; not preregistered, not an independent confirmation and not a new experiment.', '', f'**For the 21 cross-family directions and 182 designs scorable at fixed budget 24, the exact test-correspondence advantage of affine ridge at budget 192 is {10000.0 * top.delta_test:+.3f} yield pp², but the net gain relative to the equal-cost target mean is {10000.0 * top.net_gain:+.3f} yield pp².** Correspondence advantage and net reuse are different propositions; this round scores both on the same targets, costs and test support.', '', '## Brief design record and fixed scientific objects', '', 'This round keeps in full the 78 registered directions of report 57, the original target response operators for strict/background_free/position2/min1/all, identity-level hold-out, target calibration readouts, the 10 hash designs and the 24/48/96/192 coordinate budgets. The question is whether, at equal cost, the true source-identity correspondence gives a smaller held-out error than shuffled correspondence, and whether this correspondence value translates into a net gain relative to the target-mean control. Response endpoints, models, alpha and acquisition order were not changed, and no mapping was chosen using test y.', '', 'The primary test freezes the existing a, b and all calibration readouts of 57 within each design/budget and permutes source x only among held-out test identities; the secondary test applies 128 fixed unique non-identity mappings over all source identities and refits by the same rule. The two reference tests address different objects and are not subtracted into a mechanistic contribution. Scorable design MSE is first averaged with equal weight within each direction, then with equal weight across directions; designs, mappings and test identities are not treated as independent experimental replicates.', '', 'The cross-budget primary domain is the 21 cross-family directions and 182 designs already scorable at budget 24 in 57; later budgets use exactly the same directions and test identities. All results for the all-scored domain and the within-family domain at each budget are also kept. The denominator of the all-scored domain changes with budget, so its change across budgets cannot be called a pure budget effect.', '', '## Primary test: all TEST permutations after the frozen fit', '', 'Let the original frozen prediction be `a+b*x_i`, with x/y the source/target responses of the same set of n_test test identities. The equal-weight average over all n_test! identity permutations (including the original one) is computed exactly without enumeration:', '', '`R_perm_exact = mean[(y-a-b*mean(x))²] + b²*Var_n(x)`', '', '`Delta_test = R_perm_exact-R_real = 2*b*Cov_n(x,y)`', '', '`G_net = R_target_calibration_mean-R_real`', '', 'Variances and covariances use denominator n_test; there is no randomisation p value, and the permutation SD is not used as experimental error. Real and permuted predictions share the same test source marginal, calibration fit, target calibration cost and target operator; only the test-identity correspondence differs. All covariances and errors that use test y are retrospective oracle scoring diagnostics and cannot serve as criteria for choosing directions, models or budgets in advance.', '', '| Fixed cross-domain budget | target mean RMSE pp | real affine RMSE pp | exact TEST reference RMSE pp | Delta_test pp² | G_net pp² | Directions with Delta>0 | Directions with Gnet>0 | Directions with Delta>0 and no net gain |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in fixed.itertuples():
        lines.append(f'| {row.budget} | {row.target_mean_rmse_pp:.6f} | {row.real_rmse_pp:.6f} | {row.test_permuted_exact_rmse_pp:.6f} | {10000.0 * row.delta_test:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.positive_delta_test_directions}/{row.n_scored_directions} | {row.positive_net_directions}/{row.n_scored_directions} | {row.positive_delta_without_net_directions}/{row.n_scored_directions} |')
    lines += ['', 'Macro RMSE is the square root of MSE averaged with equal weight first over designs, then over directions; Delta and Gnet are paired differences on the same MSE scale, not differences of RMSE in percentage points and not means of per-direction relative risk ratios. A positive Delta means that this fitted slope is better aligned on these test identities than the average over all test permutations; it does not guarantee doing better than the target mean. All positive, negative and neutral directions are kept.', '', 'For offset, b is fixed at 1, and the test x/y of each design in the fixed domain do not change across budgets, so Delta_test=2*Cov_n(x,y) is exactly constant across the four budgets. This identity is not reported as a four-fold robustness finding; budget can change only the offset intercept and the corresponding absolute error and net gain.', '', '## Coverage and complete direction ledger', '', '| relation | budget | Scorable directions | Scorable designs | Registered directions |', '|---|---:|---:|---:|---:|']
    for row in coverage.itertuples():
        lines.append(f'| {row.relation} | {row.budget} | {row.n_scored_directions} | {row.n_scored_designs} | {row.n_registered_directions} |')
    lines += ['', 'Results of both models on all other scored domains are kept below; the fixed domain is the subset of designs originally scorable at B24, and all-scored is the originally scorable domain at each budget. Direction and design counts are pairing denominators, not counts of independent replicates.', '', '| domain | relation | budget | model | Delta_test pp² | G_net pp² | Directions/designs |', '|---|---|---:|---|---:|---:|---:|']
    other = primary[~(primary.domain.eq('b24_fixed') & primary.relation.eq('cross_family') & primary.model.eq('source_affine_ridge'))]
    for row in other.itertuples():
        lines.append(f'| {row.domain} | {row.relation} | {row.budget} | {row.model} | {10000.0 * row.delta_test:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.n_scored_directions}/{row.n_scored_designs} |')
    lines += ['', 'The remaining geometry_absent, insufficient_identity_support, budget_insufficient and original statuses are kept in full with risk NA; unknown geometry or under-budget calibration is not recoded as negative transport. For the original 54 finite directions and 500 collapsed response identities, all underlying source/target coordinates and the union of imported records were rechecked to be disjoint, so that full-source reassignment does not bring in target labels already read. Disjointness in this computation does not show that different identities, screens or families are statistically independent.', '', '## Secondary reference: reassignment of all source identities with recalibration', '', '128 unique non-identity identity permutations are generated only for the 38 eligible directions with at least 6 identities. The other 16 finite directions with fewer than 6 identities have at most 5!−1=119 non-identity permutations; they and the 24 directions without geometry keep the mapping guard, and 128 unique mappings are not fabricated. A mapping acts on all source x of a direction and is frozen across all designs and budgets; all target calibration labels, test identities, target operators and costs are unchanged.', '', f"hash version is `{VERSION}`. Complete identities are first sorted in Python string order; candidate increases from 0, identities are sorted by SHA256(JSON[version,source,target,candidate,identity]) to give the assigned-source index vector, the identity and already accepted mappings are skipped, and the first 128 are taken. JSON is UTF-8, ensure_ascii=False, sort_keys=True, separators=(',',':'). All mapping/candidate/hash values are saved; the analysis can be rebuilt exactly from the original 57 tables without storing tens of millions of unit predictions.", '', 'Each permutation keeps the distribution of all source x and the full-source mean/SD; real and permuted fits use the same alpha=3 standardized affine ridge, unpenalised intercept and offset. The code uses the one-dimensional closed form `b=Sxy/(Sxx+3*SD_all²)`, equivalent to the SVD Ridge of 57; when the full-source SD<=1e-12, affine falls back to the target mean. Real coefficients and MSE are checked term by term against 57. Test y never enters the fit or the mapping choice, and alpha is not tuned.', '', 'All 128 draw-level scores are saved; numerical duplicates, duplicate predictions and all-zero sources are not deduplicated. mapping_registry also records the number of distinct assigned-x vectors; 128 identity permutations do not represent 128 numerically distinct experiments. The secondary reference changes both the coverage/correspondence of calibration x and the test correspondence, so the real fit outperforming full-domain reassignment cannot be attributed to test correspondence alone.', '', '`Delta_pair = mean_128(R_reassigned_refit)-R_real`; `G_net=R_targetmean-R_real`. These two quantities are reported separately from the primary Delta_test and are not subtracted into independent mechanistic contributions.', '', '| Fixed cross-domain budget | model | real RMSE pp | reassigned/refit RMSE pp | Delta_pair pp² | G_net pp² | Directions with Delta_pair>0 |', '|---:|---|---:|---:|---:|---:|---:|']
    for row in secondary[secondary.domain.eq('b24_fixed') & secondary.relation.eq('cross_family')].sort_values(['model', 'budget']).itertuples():
        lines.append(f'| {row.budget} | {row.model} | {row.real_rmse_pp:.6f} | {row.mean_permuted_rmse_pp:.6f} | {10000.0 * row.delta_pair:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.positive_delta_pair_directions}/{row.n_scored_directions} |')
    lines += ['', '## Degenerate, local and scope boundaries', '', '| source | target | Identities | All source zero | Distinct assigned-x vectors among the 128 mappings |', '|---|---|---:|---|---:|']
    for row in mapping_registry[mapping_registry.mapping_status.eq('mapped') & mapping_registry.all_source_constant].itertuples():
        source, target = (row.source_screen.replace('|', '\\|'), row.target_screen.replace('|', '\\|'))
        lines.append(f'| {source} | {target} | {row.n_identities} | {row.all_source_zero} | {row.n_distinct_assigned_source_vectors} |')
    lines += ['', 'The all-zero source of SNAr→Heck has no usable identity differences, and keeping 128 mappings does not create new information; the all-zero target risk of the reverse Heck→SNAr is handled by the original eligibility and budget. All direction tables keep the states positive Delta with non-positive Gnet, negative Delta and neutral; a positive average correspondence is not presented as realised net reuse.', '', 'This test is still the correspondence test within the full question of "whether learning, measurement, chemical response and transport can work together on the same objects and at the same cost". It constrains only the original one-dimensional affine/offset rules, the finite historical identities and the fixed acquisition strategy of 57; it is not a refutation or proof for all possible models, nor a cross-library physicochemical intervention, an experimental causal effect, a universal transport law or a source-structure mechanism. Source and target populations still have their own partner/background composition; they have not been made into common substrates or the same underlying response operator.', '', 'All directions and hash designs share sources, targets and substrates, so no independent-sample p values or confidence intervals are computed; the distribution over 128 draws is a fixed set of structural references, not an experimental randomisation distribution. New data, independent acquisition, additional models or fully matched populations may lead to different conclusions; this round does not pick directions or rewrite response objects based on the results.', '', '## Outputs and verification', '', 'producer `source_identity_alignment.py`; focused tests `test_source_identity_alignment.py`; independent verification `verify_source_identity_alignment.py`. The same-prefix mapping_registry, mappings, source_target_support_audit, design_scores, draw_scores, secondary_scores, direction_scores, headline, secondary_direction_scores, secondary_headline and coverage tables keep all mappings, statuses, draws and hierarchical denominators. figure47 and the manifest record input, code and output SHA256; raw data, the 57 tables, README and other reports are not modified.', '']
    return '\n'.join(lines)

def main():
    names = ['tables/budgeted_target_calibration_unit_support.csv.gz', 'tables/budgeted_target_calibration_designs.csv', 'tables/budgeted_target_calibration_budget_designs.csv.gz', 'tables/budgeted_target_calibration_scores.csv', 'data/budgeted_target_calibration_manifest.json', 'data/budgeted_target_calibration_verification.json', 'budgeted_target_calibration.py', 'budgeted_target_calibration.py']
    hashes = {name: digest(HERE / name) for name in names}
    support, designs, budgets, old_scores = [pd.read_csv(HERE / name, keep_default_na=False if 'unit_support' in name else True) for name in names[:4]]
    if not json.loads((DATA / 'budgeted_target_calibration_verification.json').read_text())['passed']:
        raise ValueError('report57 independent verification is not passing')
    mapping_registry, mappings, cache, audit = build_mappings(support, designs)
    primary, secondary, draws = build_scores(support, budgets, old_scores, cache)
    directions, headline = aggregate_designs(primary)
    secondary_directions, secondary_headline = aggregate_designs(secondary, secondary=True)
    coverage = build_coverage(budgets)
    outputs = []
    for name, frame in (('mapping_registry', mapping_registry), ('mappings', mappings), ('source_target_support_audit', audit), ('design_scores', primary), ('draw_scores', draws), ('secondary_scores', secondary), ('direction_scores', directions), ('headline', headline), ('secondary_direction_scores', secondary_directions), ('secondary_headline', secondary_headline), ('coverage', coverage)):
        path = TABLES / (PREFIX + '_' + name + ('.csv.gz' if name in {'mappings', 'draw_scores'} else '.csv'))
        frame.to_csv(path, index=False)
        outputs.append(path)
    plot_figure(headline, secondary_headline, directions)
    REPORT.write_text(render_report(headline, secondary_headline, mapping_registry, coverage), encoding='utf-8')
    outputs += [REPORT, FIGURE]
    if hashes != {name: digest(HERE / name) for name in names}:
        raise RuntimeError('an inherited input changed')
    code = ['source_identity_alignment.py', 'test_source_identity_alignment.py', 'verify_source_identity_alignment.py']
    manifest = {'date': '2026-10-17', 'status': 'completed_source_identity_alignment', 'input_sha256': hashes, 'code_sha256': {name: digest(HERE / name) for name in code}, 'output_sha256': {str(path.relative_to(HERE)): digest(path) for path in outputs}, 'counts': {'registered_directions': len(mapping_registry), 'finite_directions': len(audit), 'mapped_directions': int(mapping_registry.mapping_status.eq('mapped').sum()), 'mapping_identity_rows': len(mappings), 'design_model_rows': len(primary), 'scored_design_model_rows': int(primary.status.eq('scored').sum()), 'draw_model_rows': len(draws), 'secondary_rows': len(secondary), 'primary_direction_rows': len(directions), 'primary_headline_rows': len(headline), 'secondary_direction_rows': len(secondary_directions), 'secondary_headline_rows': len(secondary_headline), 'n_draws': N_DRAWS}, 'design': {'version': VERSION, 'alpha': ALPHA, 'eps': EPS, 'primary': 'frozen calibration fit; exact mean over all TEST identity permutations including identity', 'secondary': '128 unique nonidentity all-source identity permutations; same mappings across designs/budgets; refit', 'primary_budget_domain': 'budget24 scored cross-family designs fixed across24/48/96/192; all-scored and within retained', 'aggregation': 'equal draws within design for secondary; equal scored designs within direction; equal directions', 'mapping': 'hash-rank candidate identities; reject identity and duplicate index permutations; retain duplicate values', 'diagnostics': 'TEST-y metrics are retrospective scoring oracles; no tuning or mapping choice using TEST y'}, 'checks': {'max_exact_covariance_identity_error': float(primary.delta_test_covariance_identity_error.abs().max()), 'all_full_source_target_coordinates_disjoint': bool(audit.all_source_target_coordinate_overlap.eq(0).all()), 'all_full_source_target_records_disjoint': bool(audit.all_source_target_record_overlap.eq(0).all())}, 'limitations': ['TEST-only fixed-fit permutations and all-source refit permutations test different objects; no causal contribution subtraction.', 'Positive identity alignment without positive net gain does not establish net reuse.', 'Mappings/designs are fixed references, not experimental replication, physical interventions or randomization p-values.', 'Unknown geometry and insufficient identity/budget support stay guarded; all degenerate sources retained.', 'Restricted affine/offset models, existing response operators and cost-first designs; not all potential models.']}
    (DATA / f'{PREFIX}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': manifest['counts'], 'checks': manifest['checks']}, indent=2))
if __name__ == '__main__':
    main()
