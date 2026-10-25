"""Audit observed decision headroom and saved policy regret without refitting.

All original strict swaps and both outcome populations remain in the outputs.
The observed per-cell oracle is retrospective and not a deployable policy."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
BASE = Path(__file__).resolve().parents[1]
MODELS = ('unpenalized', 'ridge_1', 'ridge_10', 'ridge_100')
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
TIE_TOL_PP = 1e-08
STABLE_HEADROOM_PP = 1.0
STABLE_BASELINE_RMSE_PP = 1.0
SEED = 20261017

def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def original_folds(swap_id: str, n: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f'{SEED}|{swap_id}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    folds = np.empty(n, dtype=int)
    folds[rng.permutation(n)] = np.arange(n) % 5
    return folds

def reconstruct(frame: pd.DataFrame, folds: np.ndarray) -> tuple[dict, list[dict]]:
    i, l1 = pd.factorize(frame.sub_1_smiles.fillna('<MISSING>'), sort=True)
    j, l2 = pd.factorize(frame.sub_2_smiles.fillna('<SINGLE>'), sort=True)
    n1, n2 = (len(l1), len(l2))
    y = frame.delta_pp.to_numpy(float)
    eligible_values = []
    rows = []
    for fold in range(5):
        train, test = (folds != fold, folds == fold)
        row = dict(fold=fold, n_train=int(train.sum()), n_test=int(test.sum()), n_identifiable=0, training_delta_mean_pp=np.nan, training_fixed_chooses_high=False, target_delta_sum_pp=0.0, oracle_increment_sum_pp=0.0, lower_increment_sum_pp=0.0, training_fixed_increment_sum_pp=0.0, constant_sse=0.0)
        if train.any() and test.any():
            left, right = (i[train], j[train] + n1)
            graph = coo_matrix((np.ones(2 * train.sum()), (np.r_[left, right], np.r_[right, left])), shape=(n1 + n2, n1 + n2)).tocsr()
            _, labels = connected_components(graph, directed=False)
            degree1 = np.bincount(i[train], minlength=n1)
            degree2 = np.bincount(j[train], minlength=n2)
            it, jt = (i[test], j[test])
            valid = (degree1[it] > 0) & (degree2[jt] > 0) & (labels[it] == labels[jt + n1])
            target = y[test][valid]
            mean = float(y[train].mean())
            choose_high = mean > TIE_TOL_PP
            row.update(n_identifiable=len(target), training_delta_mean_pp=mean, training_fixed_chooses_high=choose_high, target_delta_sum_pp=float(target.sum()), oracle_increment_sum_pp=float(np.maximum(target, 0).sum()), lower_increment_sum_pp=float(np.minimum(target, 0).sum()), training_fixed_increment_sum_pp=float(choose_high * target.sum()), constant_sse=float(np.square(target - mean).sum()))
            eligible_values.append(target)
        rows.append(row)
    eligible = np.concatenate(eligible_values) if eligible_values else np.array([])
    n = len(eligible)
    out = dict(n_pairs=len(y), n_identifiable=n, cv_coverage=n / len(y) if len(y) else np.nan, reconstructed_constant_sse=float(sum((row['constant_sse'] for row in rows))), n_positive_eligible=int((eligible > TIE_TOL_PP).sum()), n_negative_eligible=int((eligible < -TIE_TOL_PP).sum()), n_numerical_zero_eligible=int((np.abs(eligible) <= TIE_TOL_PP).sum()))
    if n:
        oracle = float(np.maximum(eligible, 0).mean())
        hindsight = max(0.0, float(eligible.mean()))
        headroom = oracle - hindsight
        out.update(target_mean_delta_pp=float(eligible.mean()), target_mean_abs_delta_pp=float(np.abs(eligible).mean()), oracle_increment_pp=oracle, lower_possible_increment_pp=float(np.minimum(eligible, 0).mean()), hindsight_fixed_increment_pp=hindsight, oracle_headroom_pp=headroom, headroom_from_mass_identity_pp=float((np.abs(eligible).mean() - abs(eligible.mean())) / 2), training_fixed_increment_pp=sum((row['training_fixed_increment_sum_pp'] for row in rows)) / n, reconstructed_baseline_rmse_pp=float(np.sqrt(out['reconstructed_constant_sse'] / n)), headroom_bin='zero' if headroom <= TIE_TOL_PP else 'positive_lt1pp' if headroom < 1 else 'ge1_lt5pp' if headroom < 5 else 'ge5pp')
    else:
        out.update({name: np.nan for name in ('target_mean_delta_pp', 'target_mean_abs_delta_pp', 'oracle_increment_pp', 'lower_possible_increment_pp', 'hindsight_fixed_increment_pp', 'oracle_headroom_pp', 'headroom_from_mass_identity_pp', 'training_fixed_increment_pp', 'reconstructed_baseline_rmse_pp')})
        out['headroom_bin'] = 'no_scoreable_cells'
    return (out, rows)

def prepare_summary(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    scoreable = g.n_identifiable.gt(0)
    utility = g.gain_vs_hindsight_fixed_pp
    flags = dict(scoreable=scoreable, no_scoreable=~scoreable, zero_variance=g.zero_sst.eq(True), low_df=g.residual_df.lt(20), zero_headroom=scoreable & g.oracle_headroom_pp.le(TIE_TOL_PP), low_headroom=g.oracle_headroom_pp.gt(TIE_TOL_PP) & g.oracle_headroom_pp.lt(1), headroom1=g.oracle_headroom_pp.ge(1), headroom5=g.oracle_headroom_pp.ge(5), utility_positive=utility.gt(TIE_TOL_PP), utility1=utility.ge(1), utility5=utility.ge(5), utility_nonpositive=scoreable & utility.le(TIE_TOL_PP), stable_error=g.stable_error_ratio.notna(), error90=g.stable_error_ratio.le(0.9), error90_no_utility=g.stable_error_ratio.le(0.9) & utility.le(TIE_TOL_PP), error_better=g.mse_improvement_pp2.gt(1e-08), error_better_no_utility=g.mse_improvement_pp2.gt(1e-08) & utility.le(TIE_TOL_PP))
    for name, values in flags.items():
        g['_' + name] = values
    return g

def summarize_many(g: pd.DataFrame, keys: list[str], base_index: pd.MultiIndex) -> pd.DataFrame:
    fields = dict(n_swaps=('swap_id', 'size'), n_scoreable=('_scoreable', 'sum'), n_no_scoreable_cells=('_no_scoreable', 'sum'), n_zero_full_panel_variance=('_zero_variance', 'sum'), n_residual_df_below20=('_low_df', 'sum'), n_zero_headroom=('_zero_headroom', 'sum'), n_headroom_positive_lt1pp=('_low_headroom', 'sum'), n_headroom_ge1pp=('_headroom1', 'sum'), n_headroom_ge5pp=('_headroom5', 'sum'), median_headroom_pp=('oracle_headroom_pp', 'median'), mean_headroom_pp=('oracle_headroom_pp', 'mean'), median_gain_vs_training_fixed_pp=('saved_gain_vs_training_fixed_pp', 'median'), median_gain_vs_hindsight_fixed_pp=('gain_vs_hindsight_fixed_pp', 'median'), mean_gain_vs_hindsight_fixed_pp=('gain_vs_hindsight_fixed_pp', 'mean'), n_gain_vs_hindsight_positive=('_utility_positive', 'sum'), n_gain_vs_hindsight_ge1pp=('_utility1', 'sum'), n_gain_vs_hindsight_ge5pp=('_utility5', 'sum'), n_nonpositive_gain=('_utility_nonpositive', 'sum'), median_regret_vs_oracle_pp=('regret_vs_oracle_pp', 'median'), mean_regret_vs_oracle_pp=('regret_vs_oracle_pp', 'mean'), median_headroom_capture_fraction=('headroom_capture_fraction', 'median'), median_regret_to_headroom_ratio=('regret_to_headroom_ratio', 'median'), n_stable_error_ratios=('_stable_error', 'sum'), median_stable_error_ratio=('stable_error_ratio', 'median'), n_stable_error_ratio_le90=('_error90', 'sum'), n_mse_improved=('_error_better', 'sum'), n_stable_error_ratio_le90_but_nonpositive_gain=('_error90_no_utility', 'sum'), n_mse_improved_but_nonpositive_decision_gain=('_error_better_no_utility', 'sum'), median_rmse_improvement_pp=('rmse_improvement_pp', 'median'), mean_mse_improvement_pp2=('mse_improvement_pp2', 'mean'), total_identifiable_cells=('n_identifiable', 'sum'))
    out = g.groupby(keys, sort=True).agg(**fields).reindex(base_index)
    count_columns = [name for name in fields if name.startswith('n_') or name == 'total_identifiable_cells']
    out[count_columns] = out[count_columns].fillna(0).astype(int)
    return out.reset_index()

def gates(g: pd.DataFrame, edge: bool=False):
    yield ('all_retained', g)
    if not edge:
        yield ('scoreable', g.loc[g.n_identifiable.gt(0)])
        yield ('headroom_positive', g.loc[g.oracle_headroom_pp.gt(TIE_TOL_PP)])
    yield ('headroom_ge1pp', g.loc[g.oracle_headroom_pp.ge(1)])
    yield ('headroom_ge5pp', g.loc[g.oracle_headroom_pp.ge(5)])

def markdown_table(g: pd.DataFrame) -> str:

    def fmt(v):
        if pd.isna(v):
            return 'NA'
        return f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(g.columns) + ' |', '| ' + ' | '.join(['---'] * len(g.columns)) + ' |', *['| ' + ' | '.join((fmt(v) for v in row)) + ' |' for row in g.itertuples(index=False, name=None)]])

def main() -> None:
    inputs = ['results/partner_shrinkage_swaps.csv', 'results/partner_shrinkage_folds.csv', 'results/swap_deltas.parquet']
    hashes = {name: file_hash(BASE / name) for name in inputs}
    reference = pd.read_csv(BASE / inputs[0])
    saved_folds = pd.read_csv(BASE / inputs[1])
    check_rows = []

    def check(name, ok, detail=''):
        check_rows.append(dict(check=name, passed=bool(ok), detail=str(detail)))
        if not ok:
            raise AssertionError(f'{name}: {detail}')
    toy = pd.DataFrame(dict(sub_1_smiles=['a', 'a', 'b', 'b'], sub_2_smiles=['x', 'y', 'x', 'y'], delta_pp=[10.0, 10.0, -9.0, -9.0]))
    toy_result, _ = reconstruct(toy, np.arange(4))
    check('independent_global_hindsight_and_training_fixed_example', all((np.isclose(toy_result[name], value) for name, value in {'n_identifiable': 4, 'oracle_increment_pp': 5.0, 'hindsight_fixed_increment_pp': 0.5, 'oracle_headroom_pp': 4.5, 'training_fixed_increment_pp': -4.5}.items())))
    empty_result, _ = reconstruct(toy.iloc[:0], np.array([], dtype=int))
    check('empty_panel_retains_undefined_opportunity', empty_result['n_identifiable'] == 0 and np.isnan(empty_result['oracle_headroom_pp']))
    zero_result, _ = reconstruct(toy.assign(delta_pp=0.0), np.arange(4))
    check('constant_zero_panel_has_zero_opportunity', zero_result['oracle_headroom_pp'] == 0 and zero_result['n_numerical_zero_eligible'] == 4)
    check('all_strict_swap_outcomes_retained', len(reference) == 4264 and reference.groupby('outcome').size().eq(2132).all())
    check('reference_keys_unique', not reference.duplicated(['swap_id', 'outcome']).any())
    cols = ['swap_id', 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles', 'delta_pp', 'y0', 'y1']
    delta = pd.read_parquet(BASE / inputs[2], columns=cols)
    delta = delta.loc[delta.swap_id.isin(reference.swap_id)].copy()
    ref = reference.set_index(['swap_id', 'outcome'])
    opportunities, folds = ([], [])
    for count, (sid, g) in enumerate(delta.groupby('swap_id', sort=True), 1):
        g = g.sort_values('substrate_pair_key').reset_index(drop=True)
        full_folds = original_folds(sid, len(g))
        for outcome in ('all', 'both_positive'):
            mask = np.ones(len(g), dtype=bool) if outcome == 'all' else (g.y0.gt(0) & g.y1.gt(0)).to_numpy()
            r = ref.loc[sid, outcome]
            result, fold_rows = reconstruct(g.loc[mask], full_folds[mask])
            result.update({name: r[name] for name in ['reaction_type', 'swap_slot', 'component_pair', 'condition0', 'condition1', 'residual_df', 'zero_sst']})
            result.update(swap_id=sid, outcome=outcome)
            opportunities.append(result)
            folds.extend((dict(swap_id=sid, outcome=outcome, **row) for row in fold_rows))
        if count % 500 == 0:
            print(f'reconstructed {count}/2132 swaps', flush=True)
    opportunity = pd.DataFrame(opportunities)
    fold_table = pd.DataFrame(folds)
    check('headroom_population_partition', bool((opportunity.n_identifiable.eq(0).astype(int) + opportunity.oracle_headroom_pp.le(TIE_TOL_PP).astype(int) + (opportunity.oracle_headroom_pp.gt(TIE_TOL_PP) & opportunity.oracle_headroom_pp.lt(1)).astype(int) + opportunity.oracle_headroom_pp.ge(1).astype(int)).eq(1).all()))
    joined = opportunity.merge(reference, on=['swap_id', 'outcome'], validate='one_to_one', suffixes=('', '_saved'))
    check('all_eligible_cell_counts_match', joined.n_identifiable.eq(joined.cv_n_identifiable).all())
    check('all_outcome_panel_counts_match', joined.n_pairs.eq(joined.n_pairs_saved).all())
    check('all_reconstructed_baseline_sse_match', np.allclose(joined.reconstructed_constant_sse, joined.cv_constant_sse.fillna(0), atol=1e-06, rtol=1e-10))
    valid = opportunity.n_identifiable.gt(0)
    check('headroom_mass_identity', np.allclose(opportunity.loc[valid, 'oracle_headroom_pp'], opportunity.loc[valid, 'headroom_from_mass_identity_pp'], atol=1e-10))
    fold_join = saved_folds.merge(fold_table, on=['swap_id', 'outcome', 'fold'], suffixes=('_saved', ''), validate='many_to_one')
    check('all_saved_ridge_fold_counts_match', len(fold_join) == len(saved_folds) and all((fold_join[name].eq(fold_join[name + '_saved']).all() for name in ['n_train', 'n_test', 'n_identifiable'])))
    check('all_saved_fold_training_means_match', np.allclose(fold_join.training_delta_mean_pp, fold_join.training_delta_mean_pp_saved, atol=1e-10))
    check('all_saved_fold_baseline_sse_match', np.allclose(fold_join.constant_sse, fold_join.constant_sse_saved, atol=1e-06, rtol=1e-10))
    records = []
    for model in MODELS:
        g = opportunity.copy()
        values = reference.set_index(['swap_id', 'outcome']).reindex(pd.MultiIndex.from_frame(g[['swap_id', 'outcome']]))
        gain = values[f'cv_{model}_policy_gain_pp'].to_numpy(float)
        sse = values[f'cv_{model}_sse'].to_numpy(float)
        saved_ratio = values[f'cv_{model}_error_ratio'].to_numpy(float)
        g['model'] = model
        g['saved_gain_vs_training_fixed_pp'] = gain
        g['saved_error_ratio'] = saved_ratio
        g['model_sse'] = sse
        g['policy_increment_pp'] = gain + g.training_fixed_increment_pp
        g['gain_vs_hindsight_fixed_pp'] = g.policy_increment_pp - g.hindsight_fixed_increment_pp
        g['regret_vs_oracle_pp'] = g.oracle_increment_pp - g.policy_increment_pp
        g['headroom_minus_regret_minus_gain_pp'] = g.oracle_headroom_pp - g.regret_vs_oracle_pp - g.gain_vs_hindsight_fixed_pp
        g['model_mse_pp2'] = sse / g.n_identifiable.replace(0, np.nan)
        g['baseline_mse_pp2'] = g.reconstructed_constant_sse / g.n_identifiable.replace(0, np.nan)
        g['mse_improvement_pp2'] = g.baseline_mse_pp2 - g.model_mse_pp2
        g['rmse_improvement_pp'] = np.sqrt(g.baseline_mse_pp2) - np.sqrt(g.model_mse_pp2)
        stable_error = g.reconstructed_baseline_rmse_pp.ge(STABLE_BASELINE_RMSE_PP)
        g['stable_error_ratio'] = (sse / g.reconstructed_constant_sse.replace(0, np.nan)).where(stable_error)
        stable_headroom = g.oracle_headroom_pp.ge(STABLE_HEADROOM_PP)
        g['headroom_capture_fraction'] = (g.gain_vs_hindsight_fixed_pp / g.oracle_headroom_pp).where(stable_headroom)
        g['regret_to_headroom_ratio'] = (g.regret_vs_oracle_pp / g.oracle_headroom_pp).where(stable_headroom)
        records.append(g)
        if model.startswith('ridge_'):
            alpha = float(model.split('_')[1])
            fs = saved_folds.loc[saved_folds.alpha.eq(alpha)].groupby(['swap_id', 'outcome']).agg(gain_sum=('policy_gain_sum_pp', 'sum'), n=('n_identifiable', 'sum'), sse=('ridge_sse', 'sum'))
            j = g.merge(fs, on=['swap_id', 'outcome'], validate='one_to_one')
            check(f'{model}_fold_policy_gain_match', np.allclose(j.saved_gain_vs_training_fixed_pp, j.gain_sum / j.n, atol=1e-10))
            check(f'{model}_fold_sse_match', np.allclose(j.model_sse, j.sse, atol=1e-06, rtol=1e-10))
    policies = pd.concat(records, ignore_index=True)
    scoreable = policies.n_identifiable.gt(0)
    check('all_scoreable_policies_present', policies.loc[scoreable, 'policy_increment_pp'].notna().all())
    check('policy_increment_oracle_bounds', bool((policies.loc[scoreable, 'policy_increment_pp'] <= policies.loc[scoreable, 'oracle_increment_pp'] + 1e-08).all() and (policies.loc[scoreable, 'policy_increment_pp'] >= policies.loc[scoreable, 'lower_possible_increment_pp'] - 1e-08).all()))
    check('regret_headroom_gain_identity', bool(policies.headroom_minus_regret_minus_gain_pp.abs().dropna().lt(1e-09).all()))
    check('saved_stable_error_ratios_match', np.allclose(policies.loc[policies.stable_error_ratio.notna(), 'stable_error_ratio'], policies.loc[policies.stable_error_ratio.notna(), 'saved_error_ratio'], atol=1e-09))
    check('full_long_population_retained', len(policies) == 4264 * 4)
    prepared = prepare_summary(policies)
    summaries = []
    for level, strata in [('corpus', []), ('reaction', ['reaction_type']), ('reaction_slot', ['reaction_type', 'swap_slot'])]:
        keys = ['outcome', 'model', *strata]
        base_index = prepared.groupby(keys, sort=True).size().index
        for gate, subset in gates(prepared):
            row = summarize_many(subset, keys, base_index)
            row['row_level'], row['gate'] = (level, gate)
            if 'reaction_type' not in row:
                row['reaction_type'] = 'ALL'
            if 'swap_slot' not in row:
                row['swap_slot'] = 'ALL'
            summaries.append(row)
    summary = pd.concat(summaries, ignore_index=True)
    edge_keys = ['outcome', 'model', *EDGE]
    base_index = prepared.groupby(edge_keys, sort=True).size().index
    edge_rows = []
    for gate, subset in gates(prepared, edge=True):
        row = summarize_many(subset, edge_keys, base_index)
        row['gate'] = gate
        edge_rows.append(row)
    edges = pd.concat(edge_rows, ignore_index=True)
    corpus_counts = summary.loc[summary.row_level.eq('corpus')].set_index(['outcome', 'model', 'gate'])['n_swaps'].sort_index()
    for level in ('reaction', 'reaction_slot'):
        counts = summary.loc[summary.row_level.eq(level)].groupby(['outcome', 'model', 'gate']).n_swaps.sum().sort_index()
        check(f'{level}_denominators_sum_to_corpus', counts.equals(corpus_counts))
    edge_counts = edges.groupby(['outcome', 'model', 'gate']).n_swaps.sum().sort_index()
    check('typed_edge_denominators_sum_to_corpus', edge_counts.equals(corpus_counts.reindex(edge_counts.index)))
    check('all_typed_edges_retained_in_all_strata', edges.groupby(['outcome', 'model', 'gate']).size().eq(1548).all())
    equal_edges = []
    original_no_cv = edges.loc[edges.gate.eq('all_retained')].assign(no_original_cv=lambda g: g.n_scoreable.eq(0)).groupby(['outcome', 'model']).no_original_cv.sum().to_dict()
    for key, g in edges.groupby(['outcome', 'model', 'gate'], sort=True):
        usable = g.loc[g.n_scoreable.gt(0)]
        equal_edges.append(dict(zip(['outcome', 'model', 'gate'], key), n_typed_edges_total=len(g), n_typed_edges_scoreable=len(usable), n_typed_edges_no_selected_scoreable=int(g.n_scoreable.eq(0).sum()), n_typed_edges_no_cv_in_original_population=int(original_no_cv[key[:2]]), n_typed_edges_with_original_cv_but_no_selected_scoreable=int(g.n_scoreable.eq(0).sum() - original_no_cv[key[:2]]), median_edge_median_headroom_pp=usable.median_headroom_pp.median(), median_edge_median_gain_vs_hindsight_pp=usable.median_gain_vs_hindsight_fixed_pp.median(), mean_edge_mean_gain_vs_hindsight_pp=usable.mean_gain_vs_hindsight_fixed_pp.mean(), median_edge_median_regret_pp=usable.median_regret_vs_oracle_pp.median(), n_edges_positive_median_gain=int(usable.median_gain_vs_hindsight_fixed_pp.gt(TIE_TOL_PP).sum())))
    equal_edges = pd.DataFrame(equal_edges)
    check('all_inputs_unchanged', hashes == {name: file_hash(BASE / name) for name in inputs})
    outputs = {'swaps': opportunity, 'folds': fold_table, 'policies': policies, 'summary': summary, 'typed_edges': edges, 'equal_edge_summary': equal_edges, 'checks': pd.DataFrame(check_rows)}
    for name, table in outputs.items():
        table.to_csv(BASE / 'results' / f'decision_opportunity_{name}.csv', index=False)
    metadata = dict(input_sha256=hashes, source_sha256=file_hash(Path(__file__)), seed=SEED, policy_tie_tolerance_pp=TIE_TOL_PP, stable_headroom_ratio_denominator_pp=STABLE_HEADROOM_PP, stable_error_ratio_baseline_rmse_pp=STABLE_BASELINE_RMSE_PP, models=MODELS, refits=0, checks_passed=len(check_rows), rows={name: len(table) for name, table in outputs.items()}, scope='All 2132 strict swaps, both outcomes, all four saved policies; no-CV/constant/sparse rows retained.', oracle='Observed per-pair max(delta,0); no noise separation, causal identification or prospective attainability.', identity='headroom=oracle-hindsight_fixed; regret=oracle-policy; gain=headroom-regret; identity is not independent evidence.')
    (BASE / 'audit' / 'decision_opportunity_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    write_report(opportunity, summary, equal_edges, len(check_rows))
    print(summary.loc[summary.row_level.eq('corpus') & summary.gate.isin(['scoreable', 'headroom_ge1pp', 'headroom_ge5pp']), ['outcome', 'model', 'gate', 'n_swaps', 'median_headroom_pp', 'median_gain_vs_hindsight_fixed_pp', 'median_regret_vs_oracle_pp', 'median_stable_error_ratio', 'n_mse_improved_but_nonpositive_decision_gain']].to_string(index=False), flush=True)

def write_report(opportunity, summary, edges, n_checks):
    populations = []
    for outcome, g in opportunity.groupby('outcome', sort=True):
        scoreable = g.n_identifiable.gt(0)
        populations.append(dict(outcome=outcome, n_retained=len(g), n_scoreable=int(scoreable.sum()), n_no_cv=int((~scoreable).sum()), n_zero_headroom=int(g.oracle_headroom_pp.le(TIE_TOL_PP).sum()), n_positive_headroom_lt1pp=int((g.oracle_headroom_pp.gt(TIE_TOL_PP) & g.oracle_headroom_pp.lt(1)).sum()), n_headroom_ge1pp=int(g.oracle_headroom_pp.ge(1).sum()), n_headroom_ge5pp=int(g.oracle_headroom_pp.ge(5).sum()), median_headroom_pp=g.oracle_headroom_pp.median()))
    populations = pd.DataFrame(populations)
    corpus = summary.loc[summary.row_level.eq('corpus')]
    full = corpus.loc[corpus.outcome.eq('all') & corpus.gate.eq('scoreable')]
    meaningful = corpus.loc[corpus.outcome.eq('all') & corpus.gate.eq('headroom_ge1pp')]
    pop = populations.set_index('outcome').loc['all']
    meaningful_by_model = meaningful.set_index('model')
    gain_text = '; '.join((f"{model}: {meaningful_by_model.loc[model, 'median_gain_vs_hindsight_fixed_pp']:.3f} pp gain, {100 * meaningful_by_model.loc[model, 'median_headroom_capture_fraction']:.1f}% median headroom capture" for model in MODELS))
    low_headroom_n = int(pop.n_zero_headroom + pop.n_positive_headroom_lt1pp)
    lines = ['# Decision Opportunity and Saved Policy Regret', '', 'This no-refit audit retains every one of the 2,132 strict swaps under both the full and both-positive populations, and all four saved policies. It asks whether weak condition-choice gains arise because observed cells offer little advantage over one fixed arm, or because models fail to capture available signed-response opportunity.', '', f'In the full population, {int(pop.n_scoreable):,} swaps have graph-identifiable held-out cells. Among them, {int(pop.n_zero_headroom):,} have zero observed oracle headroom, {int(pop.n_positive_headroom_lt1pp):,} have positive headroom below 1 pp, and {int(pop.n_headroom_ge1pp):,} have at least 1 pp ({int(pop.n_headroom_ge5pp):,} at least 5 pp). The complete scoreable population has median headroom {pop.median_headroom_pp:.3f} pp. Zero or small headroom limits possible adaptive improvements, but policy regret on the positive-headroom strata separately quantifies unexploited observed opportunity.', '', f"Low headroom accounts for {low_headroom_n:,}/{int(pop.n_scoreable):,} scoreable swaps ({100 * low_headroom_n / pop.n_scoreable:.1f}%). The other {int(pop.n_headroom_ge1pp):,} swaps have median observed headroom {meaningful_by_model.loc['unpenalized', 'median_headroom_pp']:.3f} pp. Within this >=1 pp stratum, the four fixed policies give {gain_text}. Their median captured fractions are calculated per swap before summarizing, rather than dividing a median gain by a median headroom. Thus limited opportunity contributes to the aggregate result, while substantial observed opportunity also remains unexploited. This is retrospective oracle regret, not proof of an attainable prospective yield improvement.", '', '## Estimands and Arithmetic', '', 'All increments are relative to choosing original condition0 on the same eligible cells, in yield percentage points. On the union of the five eligible held-out folds: oracle increment = mean(max(delta,0)); hindsight fixed increment = max(0,mean(delta)); headroom = oracle minus hindsight fixed. The hindsight comparator chooses one arm over the complete eligible evaluation population, not a different arm per fold. Headroom is also (mean(abs(delta))-abs(mean(delta)))/2, the smaller positive/negative signed-effect mass. It is not the percentage of substrates in a regime.', '', "The source training-fixed increment is reconstructed from each fold's training-only mean sign, choosing condition1 only above 1e-8 pp. Saved gain versus that comparator plus its reconstructed increment recovers each model's policy increment without refitting. Gain versus hindsight fixed = policy minus hindsight fixed; regret versus oracle = oracle minus policy. Consequently headroom minus regret equals hindsight gain exactly. This identity and the analogous mass formula are consistency checks, not additional scientific evidence.", '', 'The per-cell oracle selects the higher observed yield after observing both outcomes. It is an optimistic retrospective bound containing assay error, not a deployable policy, a causal effect estimate or a noise-corrected chemical opportunity. A model can have negative hindsight gain and regret greater than headroom. Ratios are not clipped to conceal this.', '', '## Complete Population Accounting', '', markdown_table(populations), '', 'No-CV, constant-response, saturated and sparse panels remain in the swap and policy tables. Unavailable utility/error quantities are missing, not imputed as zero. Both-positive conditions on both measured yields >0 and changes the observed population; it is not independent replication.', '', '## Error Improvement and Utility', '', markdown_table(corpus.loc[corpus.gate.isin(['scoreable', 'headroom_ge1pp', 'headroom_ge5pp']), ['outcome', 'model', 'gate', 'n_swaps', 'median_headroom_pp', 'median_stable_error_ratio', 'median_gain_vs_hindsight_fixed_pp', 'n_gain_vs_hindsight_positive', 'n_gain_vs_hindsight_ge1pp', 'median_regret_vs_oracle_pp', 'n_mse_improved', 'n_mse_improved_but_nonpositive_decision_gain']]), '', 'Each table row states its selected denominator. The full corpus, zero/low-headroom counts and all failed or unavailable cases remain explicit. Error improvement compares saved held-out SSE with the reconstructed training-mean SSE on identical cells. A stable error ratio additionally requires baseline RMSE >=1 pp; absolute MSE and RMSE improvements remain available for all scoreable cells. Headroom-capture and regret/headroom ratios are reported only for headroom >=1 pp. These descriptive denominator thresholds are not significance or discovery criteria. All fixed penalties are retained without selecting a winner.', '', 'Zero headroom denotes values <=1e-8 pp to avoid assigning scientific meaning to arithmetic roundoff. Values between that numerical tolerance and 1 pp are retained as a separate low-headroom group. Raw measured deltas and oracle increments are not rounded or clipped to those descriptive thresholds.', '', 'Squared-error reduction can improve prediction magnitudes while leaving predicted signs unchanged, and therefore need not improve arm selection. Conversely, hindsight policy gain requires correctly exploiting both response signs. The saved summary counts panels with improved MSE but nonpositive hindsight gain, both across the entire scoreable population and after requiring meaningful headroom. The observed regret distinguishes unexploited opportunity from an absence of opposite-sign opportunity; neither establishes why a model failed.', '', '## Typed Edges and Class-Slot Strata', '', markdown_table(edges), '', 'Typed edges are reaction class x changed slot x unordered component identity pair. The edge file retains all 1,548 typed edges per model/outcome, including edges with no scoreable cells or no rows in a headroom stratum. The equal-edge summary first computes within-edge means or medians over eligible exact swaps, then summarizes those edge values equally as named. Class and class x slot rows are saved for all 20 reaction classes. These comparisons reuse molecule panels and condition cells, so edges, folds and swaps are not independent experiments.', '', 'An edge with no selected scoreable swap in a headroom stratum can still have CV-eligible swaps elsewhere. The equal-edge table therefore reports both no-selected-scoreable edges and original no-CV edges, with a separate count for original CV support excluded by the headroom stratum. Empty opportunity strata are not labeled as missing original CV data.', '', '## Verification and Reproduction', '', f'Passed {n_checks} checks. An independent 2x2 example with deltas [10,10,-9,-9] verifies the global hindsight fixed increment .5 pp, oracle 5 pp, headroom 4.5 pp and training-fold fixed increment -4.5 pp; empty and zero panels have separate verified behavior. Every original fold is reconstructed from the saved seed and swap ID after sorting substrate-pair keys. Both raw partner identities must be present in the same training graph component. Eligible counts, baseline SSE, saved ridge-fold counts, training means, fold policy-gain sums and model SSE sums reconcile. Policy increments obey observed per-cell oracle bounds. Class, class-slot and typed-edge denominator sums match the corpus. All input SHA-256 hashes are unchanged; no model was fitted.', '', 'Run `python code/decision_opportunity_audit.py`. Outputs use `results/decision_opportunity_`: swaps (4,264 original swap/outcome rows), policies (four models per row), reconstructed folds, corpus/class/class-slot summary, typed edges, equal-edge summary and checks. `audit/decision_opportunity_metadata.json` records inputs, thresholds and row counts. The analysis identifies observed decision opportunity and regret, without a mechanism, significance or prospective-generalization claim.']
    (BASE / 'reports' / 'DECISION_OPPORTUNITY_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
