"""Read final-layout display quantities from the original scientific outputs.

The approved expectations are a separate frozen manifest. No quantity is read
from rendered figures, final Source Data or the previous numerical-pin export."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import numpy as np
import pandas as pd
A = Path(__file__).resolve().parents[2]
REFERENCE_LEARNING = A / 'scripts/reference_learning/Analysis'
RECIPE_OPTIONS = A / 'scripts/recipe_options/Analysis'
RESPONSE_LEARNING = A / 'scripts/response_learning'
EDM = RESPONSE_LEARNING / 'Existing_Data_Mining_2026-10-16/results'
E59 = RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/results'
E63 = RESPONSE_LEARNING / 'Response_Information_Decomposition_E63/results'
E66 = RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results'
METRICS = A / 'out/direction_metrics'
BASELINE = A / 'figure_assets/approved_display_pins.json'

def collect():
    records = []

    def add(item, value, precision, source, selector):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite final display quantity: {item}')
        records.append(dict(item=item, value=value, precision=precision, source=str(Path(source).relative_to(A)), selector=selector))
    path = EDM / 'matched_response_profiles.parquet'
    profiles = pd.read_parquet(path)
    selected = profiles.loc[profiles.reaction_type.eq('suzuki') & profiles.swap_slot.eq('reagent1') & profiles.component_pair.eq('CMP0092|CMP0097')]
    if selected.empty or selected.duplicated(['substrate_pair_key', 'background_fingerprint']).any():
        raise ValueError('Flagship background responses are empty or duplicated')
    cs = next((x for x in selected.background_fingerprint.unique() if 'CMP0333@3' in x))
    kk = next((x for x in selected.background_fingerprint.unique() if 'CMP0334@3' in x))
    wide = selected.pivot(index='substrate_pair_key', columns='background_fingerprint', values='delta')[[cs, kk]].dropna()
    add('Figure 2a shared substrate pairs', len(wide), 0, path, 'Suzuki/CMP0092|CMP0097; finite both backgrounds')
    add('Figure 2a opposite response signs', int((wide[cs] * wide[kk] < 0).sum()), 0, path, 'product of shared responses < 0')
    add('Figure 2a Pearson correlation', wide.corr().iloc[0, 1], 2, path, 'Pearson over shared pairs')
    for smiles in ['Brc1ccc2[nH]ncc2c1', 'COc1cccnc1Br']:
        witness = selected.loc[selected.sub_1_smiles.eq(smiles) & selected.sub_2_smiles.eq('COc1ccc(B(O)O)cc1')]
        if len(witness) != 2:
            raise ValueError('The sign witness requires both backgrounds')
        for label, background in [('Cs2CO3', cs), ('K2CO3', kk)]:
            row = witness.loc[witness.background_fingerprint.eq(background)]
            if len(row) != 1:
                raise ValueError('A sign-witness response is not unique')
            add(f'Figure 2a {smiles} {label} response pp', row.delta.iloc[0], 0, path, 'original two-pair sign witness')
    path = E59 / 'full_edge_strict_swap_registry.csv'
    registry = pd.read_csv(path)
    for cls in ['suzuki', 'acylation', 'buchwald-hartwig', 'snar', 'ullmann', 'kumada']:
        pool = registry.loc[registry.reaction_type.eq(cls)].sort_values(['n_pairs', 'swap_id'], ascending=[False, True])
        row = registry.loc[registry.swap_id.eq('suzuki__C23__C26')].iloc[0] if cls == 'suzuki' else pool.iloc[0]
        add(f'Figure 2b {cls} substrate pairs', row.n_pairs, 0, path, str(row.swap_id))
    path = EDM / 'sparse_sign_feasibility_summary.csv'
    raw = pd.read_csv(path)
    signs = raw.loc[raw.population.eq('all') & raw.panel.eq('observed_sparse') & raw.objective.eq('count') & raw.threshold_pp.isin([0, 20]) & ~raw.stratum.isin(['ALL', 'finkelstein'])]
    if len(signs) != 38 or signs.stratum.nunique() != 19:
        raise ValueError('Final sign panel is not the prescribed nineteen arrays')
    for row in signs.sort_values(['stratum', 'threshold_pp']).itertuples():
        key = f'Figure 2c {row.stratum} threshold {int(row.threshold_pp)}'
        selector = 'all/observed_sparse/count; ' + key
        add(key + ' cyclic swaps', row.n_cycle_edges, 0, path, selector)
        add(key + ' conflicting swaps', row.n_conflicting_edges, 0, path, selector)
        if row.n_cycle_edges:
            add(key + ' conflict percent', 100 * row.n_conflicting_edges / row.n_cycle_edges, 2, path, selector)
    path = REFERENCE_LEARNING / 'Nature_Manuscript/source_data/Matched_copy_comparison.csv'
    raw = pd.read_csv(path)
    for level in ['headline', 'screen']:
        part = raw.loc[raw.level.eq(level)].copy()
        part['display_group'] = 'Overall' if level == 'headline' else part.screen
        table = part.pivot(index='display_group', columns='model', values='mse_pp2')
        if len(table) != (1 if level == 'headline' else 17):
            raise ValueError('Matched-copy displayed population changed')
        for screen, row in table.sort_index().iterrows():
            for model in ['direct_raw', 'semantic_full_raw']:
                add(f'Figure 3a {screen} {model} error change percent', 100 * (row[model] / row.marginal - 1), 1, path, f'level={level}; model={model}; group={screen}')
    path = METRICS / 'direction_accuracy_summary.csv'
    summaries = pd.read_csv(path)
    selected = summaries.loc[summaries.threshold_pp.eq(20) & summaries.aggregation.eq('pooled') & summaries.model.isin(['direct_raw', 'semantic_full_raw'])].set_index('model')
    if len(selected) != 2:
        raise ValueError('Direction comparison requires both unchanged predictors')
    for model, row in selected.iterrows():
        add(f'Figure 3b {model} errors per 100', 100 * (1 - row.accuracy), 1, path, 'pooled; threshold20')
        for field in ['n_targets', 'n_correct', 'n_zero_predictions', 'n_screens', 'n_families']:
            add(f'Figure 3b {model} {field}', row[field], 0, path, 'pooled; threshold20')
        add(f'Figure 3b {model} errors', row.n_targets - row.n_correct, 0, path, 'pooled; threshold20')
    add('direction errors copy per 100', 100 * (1 - selected.loc['direct_raw', 'accuracy']), 1, path, 'pooled; threshold20')
    add('direction errors learned per 100', 100 * (1 - selected.loc['semantic_full_raw', 'accuracy']), 1, path, 'pooled; threshold20')
    add('direction-error targets', selected.loc['direct_raw', 'n_targets'], 0, path, 'pooled; threshold20')
    add('direction errors copy count', selected.loc['direct_raw', 'n_targets'] - selected.loc['direct_raw', 'n_correct'], 0, path, 'pooled; threshold20')
    add('direction errors learned count', selected.loc['semantic_full_raw', 'n_targets'] - selected.loc['semantic_full_raw', 'n_correct'], 0, path, 'pooled; threshold20')
    add('additional correct direction decisions', selected.loc['semantic_full_raw', 'n_correct'] - selected.loc['direct_raw', 'n_correct'], 0, path, 'pooled; threshold20')
    path = METRICS / 'selected_training_pairs.csv'
    training = pd.read_csv(path)
    add('Figure 3c training pairs', len(training), 0, path, 'original support-selected training fold')
    coefficients = pd.read_csv(METRICS / 'learned_training_relationships.csv')
    for key in ['screen', 'edge_id', 'repeat', 'fold']:
        unique = training[key].unique()
        if len(unique) != 1:
            raise ValueError('Training illustration spans multiple frozen relationships')
        coefficients = coefficients.loc[coefficients[key].eq(unique[0])]
    if len(coefficients) != 1:
        raise ValueError('Training illustration lacks its original fitted relationship')
    path = REFERENCE_LEARNING / 'tables/chemical_observation_specificity_screen_scores.csv'
    specificity = pd.read_csv(path)
    specificity = specificity.loc[specificity.domain.eq('strict_641') & specificity.scope.eq('all_panels_complete') & specificity.weighting.eq('context_balanced')]
    if len(specificity) != 102:
        raise ValueError('Same-substitution controls do not have six settings in seventeen screens')
    for row in specificity.sort_values(['screen', 'split', 'fraction']).itertuples():
        add(f'Figure 3d {row.screen} {row.split} fraction {row.fraction:g} error removed percent', 100 * row.other_recovery_vs_control, 2, path, 'strict_641/all_panels_complete/context_balanced')
    path = E66 / 'e66_summary.csv'
    raw = pd.read_csv(path)
    local = raw.loc[raw.population.eq('common8') & raw.target_geometry.eq('raw_delta') & raw.stratum.eq('all') & raw.weighting.eq('equal_class_edge_balanced') & raw.learner.isin(['joint', 'calibration_only']) & raw.budget.isin(['1', '2', '4', '8'])]
    if len(local) != 8 or not local.n_test_vectors.eq(24221).all():
        raise ValueError('Common-eight model comparison changed')
    table = local.pivot(index='budget', columns='learner', values='native_vector_mse')
    scope = pd.read_parquet(E66 / 'e66_scope_vectors.parquet', columns=['common8', 'n_backgrounds'])
    scope = scope.loc[scope.common8]
    for budget in ['1', '2', '4', '8']:
        percent = 100 * (table.loc[budget, 'joint'] / table.loc[budget, 'calibration_only'] - 1)
        add(f'Figure 4c profiles {budget} error change percent', percent, 1, path, 'common8/raw_delta/all/class-edge-balanced')
        for label, backgrounds in [('minimum', scope.n_backgrounds.min()), ('maximum', scope.n_backgrounds.max())]:
            add(f'Figure 4c profiles {budget} {label} condition cells', 2 * int(budget) * backgrounds, 0, E66 / 'e66_scope_vectors.parquet', 'common8; 2*profiles*backgrounds')
    add('history plus 8 profiles: error reduction vs local only %', 100 * (1 - table.loc['8', 'joint'] / table.loc['8', 'calibration_only']), 1, path, 'common8/raw_delta/all/class-edge-balanced; budget8')
    larger = raw.loc[raw.population.eq('common16') & raw.target_geometry.eq('raw_delta') & raw.stratum.eq('all') & raw.weighting.eq('equal_class_edge_balanced') & raw.learner.isin(['joint', 'calibration_only', 'frozen_family']) & raw.budget.isin(['0', '1', '2', '4', '8', '16'])]
    historical = larger.loc[larger.learner.eq('frozen_family'), 'native_vector_mse']
    if historical.empty or not np.allclose(historical, historical.iloc[0], rtol=0, atol=1e-09):
        raise ValueError('Common-sixteen historical baseline is not fixed')
    if not larger.n_test_vectors.eq(22963).all():
        raise ValueError('Common-sixteen query support changed')
    add('Extended Data 2 query vectors', larger.n_test_vectors.iloc[0], 0, path, 'common16/raw_delta/all/class-edge-balanced')
    for row in larger.loc[larger.learner.ne('frozen_family')].sort_values(['learner', 'budget']).itertuples():
        add(f'Extended Data 2 {row.learner} profiles {row.budget} error change percent', 100 * (row.native_vector_mse / historical.iloc[0] - 1), 2, path, 'common16/raw_delta/all/class-edge-balanced')
    path = REFERENCE_LEARNING / 'tables/cross_library_source_channels_direction_scores.csv'
    raw = pd.read_csv(path)
    cross = raw.loc[raw.domain.eq('b24_fixed') & raw.relation.eq('cross_family') & raw.channel.eq('d') & raw.budget.eq(192) & raw.status.eq('scored')]
    if len(cross) != 17:
        raise ValueError('Cross-library direction support changed')
    add('Extended Data 3 calibrated transfers worse than target only', int(cross.real_mse.gt(cross.target_mean_mse).sum()), 0, path, 'b24_fixed/cross_family/d/budget192/scored')
    for row in cross.sort_values(['source_screen', 'target_screen']).itertuples():
        for field in ['target_mean_mse', 'real_mse', 'test_permuted_exact_mse']:
            add(f'Extended Data 3 {row.source_screen} to {row.target_screen} {field} percent', 100 * getattr(row, field) / row.target_mean_mse, 2, path, 'b24_fixed/cross_family/d/budget192/scored')
    path = E63 / 'e63_paired_component_summary.csv'
    gains = pd.read_csv(path)
    gains = gains.loc[gains.comparison.eq('centroid_minus_codebook_continuous')]
    if len(gains) != 6:
        raise ValueError('Structure-only decomposition support changed')
    for row in gains.sort_values(['nominal_k', 'weighting']).itertuples():
        add(f'Extended Data 4 k{int(row.nominal_k)} {row.weighting} own-level share percent', 100 * row.offset_difference_pp2 / row.mse_difference_pp2, 1, path, 'centroid_minus_codebook_continuous')
    base = pd.read_csv(E63 / 'e63_summary.csv')
    base = base.loc[base.model.eq('centroid') & base.nominal_k.eq(3) & base.weighting.eq('equal_class_edge_balanced')]
    gain = gains.loc[gains.nominal_k.eq(3) & gains.weighting.eq('equal_class_edge_balanced')]
    if len(base) != 1 or len(gain) != 1:
        raise ValueError('Structure-only class-balanced baseline is not unique')
    base, gain = (base.iloc[0], gain.iloc[0])
    add('Extended Data 4 within-level error removed percent', 100 * gain.offset_difference_pp2 / base.offset_mse_pp2, 2, path, 'k3/class-edge-balanced; removed level error divided by baseline level error')
    add('Extended Data 4 within-fluctuation error removed percent', 100 * (gain.amplitude_difference_pp2 + gain.alignment_difference_pp2) / (base.amplitude_mse_pp2 + base.alignment_mse_pp2), 2, path, 'k3/class-edge-balanced; same-layer denominator')
    path = METRICS / 'direction_predictions_targets.csv.gz'
    predictions = pd.read_csv(path)
    if len(predictions) != 121383 or predictions.duplicated(['screen', 'edge_id', 'pair_index']).any():
        raise ValueError('Unique direction-prediction support changed')
    add('direction replay unique targets', len(predictions), 0, path, 'unique screen/edge/pair; repeat0')
    example = predictions.loc[predictions.screen.eq('Suzuki|1') & predictions.edge_id.eq('8f3d681cf95d064e') & predictions.pair_index.eq(10)]
    if len(example) != 1:
        raise ValueError('The declared median example is not unique')
    for field in ['reference_contrast_pp', 'direct_raw_prediction_pp', 'semantic_full_raw_prediction_pp', 'target_contrast_pp']:
        add('Extended Data 5 ' + field, example[field].iloc[0], 1, path, 'Suzuki|1; edge8f3d681cf95d064e; pair10')
    from publication_figures.select_fig1_example import select_example
    chosen, cells, selection = select_example(write_source=False)
    if chosen.screen != 'Suzuki|1' or chosen.edge_id != '8f3d681cf95d064e' or chosen.pair_index != 10:
        raise ValueError('Predeclared median-error selection changed')
    for pool, count in selection['eligible_pool_counts'].items():
        add('Extended Data 5 eligible pool ' + pool, count, 0, path, 'complete predeclared rule; retained four-well check')
    for row in cells.itertuples():
        add(f'Extended Data 5 {row.background} endpoint {row.endpoint_position} measured yield percent', row.yield_percent, 1, REFERENCE_LEARNING / 'data/matrix_Suzuki_1.npz', 'selected pair10; oriented endpoint indices from target registry')
    support_path = E59 / 'full_edge_corpus_class_support_20.csv'
    corpus_path = E59 / 'full_edge_response_corpus.parquet'
    support = pd.read_csv(support_path)
    pair_counts = pd.read_parquet(corpus_path, columns=['reaction_type', 'substrate_pair_key'])
    pair_counts = pair_counts.groupby('reaction_type').substrate_pair_key.nunique().rename('substrate_pairs')
    inventory = support.loc[support.reaction_type.ne('finkelstein')].merge(pair_counts, on='reaction_type', validate='one_to_one')
    if len(inventory) != 19 or int(inventory.raw_observation_rows.sum()) != 406909:
        raise ValueError('Extended Data 1 no longer contains the nineteen coupling arrays')
    for row in inventory.sort_values('reaction_type').itertuples():
        add(f'Extended Data 1 {row.reaction_type} exact substitutions', row.strict_swaps, 0, support_path, 'non-Finkelstein; array=' + row.reaction_type)
        add(f'Extended Data 1 {row.reaction_type} substrate pairs', row.substrate_pairs, 0, corpus_path, 'unique substrate_pair_key within reaction_type=' + row.reaction_type)
    if len({record['item'] for record in records}) != len(records):
        raise ValueError('Final display pin identifiers are duplicated')
    return records

def displayed(record):
    quantum = Decimal(1).scaleb(-record['precision'])
    return str(Decimal(str(record['value'])).quantize(quantum, rounding=ROUND_HALF_UP))

def apply(rounded_pin):
    actual = collect()
    approved = json.loads(BASELINE.read_text(encoding='utf-8'))
    expected = approved['pins']
    if approved['baseline'] != 'M1 final integration 2026-10-22' or set(expected) != {r['item'] for r in actual}:
        raise ValueError('Final display pin manifest does not match the scientific selectors')
    for record in actual:
        rounded_pin('final', record['item'], record['value'], expected[record['item']], record['source'], record['selector'])
if __name__ == '__main__':
    records = collect()
    print(json.dumps({'baseline': 'M1 final integration 2026-10-22', 'provenance': 'Read from original REFERENCE_LEARNING/RECIPE_OPTIONS/RESPONSE_LEARNING outputs and SSE-verified unchanged prediction replay, never final exports', 'pins': {r['item']: displayed(r) for r in records}}, indent=2))
