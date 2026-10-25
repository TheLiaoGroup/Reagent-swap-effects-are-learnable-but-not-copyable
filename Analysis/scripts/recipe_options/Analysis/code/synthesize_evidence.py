"""Summarize named chemical contrasts and render exploratory scientific figures."""
from __future__ import annotations
import itertools
import json
import platform
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
COLORS = {'PyAOP': '#B63849', 'TFFH': '#177E89', 'BTFFH': '#435A9D', 'DCMT': '#727272'}
ORDER = list(COLORS)

def bootstrap_mean(values, seed=20261016):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(20000, len(x)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])

def save_figure(fig, name):
    fig.savefig(ROOT / 'figures' / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)

def make_named_evidence():
    cases = ROOT / 'cases'
    neighbors = pd.read_csv(cases / 'near_neighbor_common_partners.csv')
    common = pd.read_csv(cases / 'near_neighbor_common_all_activators.csv')
    shared = neighbors.loc[neighbors.case.eq('PyAOP_NMM_minus_DIPEA')].copy()
    shared['base_effect_4OMe_pp'] = -100 * shared.delta_target
    shared['base_effect_benzyl_pp'] = -100 * shared.delta_control
    shared['structural_double_difference_pp'] = -100 * shared.delta_delta
    shared = shared.sort_values('structural_double_difference_pp', ascending=False)
    shared['acid_label'] = [f'A{i:02}' for i in range(1, len(shared) + 1)]
    labels = shared.set_index('sub_2_smiles').acid_label
    common['activator'] = common.case.str.replace('_NMM_minus_DIPEA', '', regex=False)
    common['structural_double_difference_pp'] = -100 * common.delta_delta
    common['acid_label'] = common.sub_2_smiles.map(labels)
    shared.to_csv(cases / 'figure_PyAOP_shared10_acids.csv', index=False)
    common.to_csv(cases / 'figure_common8_activators.csv', index=False)
    all_pairs = pd.read_csv(cases / 'substrate_pair_means.csv')
    methoxy = all_pairs[all_pairs.case.eq('PyAOP_NMM_minus_DIPEA') & all_pairs.sub_1_smiles.eq('COc1ccc(CN)cc1')].copy()
    all_labels = labels.to_dict()
    remaining = sorted(set(methoxy.sub_2_smiles) - set(all_labels))
    all_labels.update({smiles: f'A{i:02}' for i, smiles in enumerate(remaining, start=len(labels) + 1)})
    methoxy['acid_label'] = methoxy.sub_2_smiles.map(all_labels)
    methoxy = methoxy.sort_values('acid_label')
    methoxy.to_csv(cases / 'figure_PyAOP_4OMe_all18_yields.csv', index=False)
    crossing = []
    for threshold in [0.2, 0.5]:
        crossing.append({'case': 'PyAOP_4OMe_NMM_to_DIPEA', 'n_acid_partners': len(methoxy), 'yield_threshold': threshold, 'NMM_above_threshold': int(methoxy.yield_B.ge(threshold).sum()), 'DIPEA_above_threshold': int(methoxy.yield_A.ge(threshold).sum()), 'gained': int((methoxy.yield_A.ge(threshold) & methoxy.yield_B.lt(threshold)).sum()), 'lost': int((methoxy.yield_A.lt(threshold) & methoxy.yield_B.ge(threshold)).sum()), 'scope': 'observed endpoint yields; descriptive thresholds'})
    pd.DataFrame(crossing).to_csv(cases / 'yield_threshold_crossing_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8.3, 7.2), layout='constrained')
    y = np.arange(len(methoxy))
    ax.hlines(y, methoxy.yield_B * 100, methoxy.yield_A * 100, color='#CBCBCB', lw=1.8)
    ax.scatter(methoxy.yield_B * 100, y, s=34, color='#177E89', label='NMM: 5/18 yields >=50%', zorder=3)
    ax.scatter(methoxy.yield_A * 100, y, s=34, color='#B63849', label='DIPEA: 14/18 yields >=50%', zorder=3)
    ax.axvline(50, color='#7B7B7B', lw=0.9, ls='--')
    ax.set(yticks=y, yticklabels=methoxy.acid_label, xlim=(0, 100), xlabel='Observed yield (%)', ylabel='Acid partner', title='PyAOP / 4-methoxybenzylamine: NMM to DIPEA')
    ax.invert_yaxis()
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='outside lower center', ncol=2, fontsize=9)
    save_figure(fig, '05_PyAOP_18acid_actual_yields')
    estimates = []

    def estimate(label, values, unit, scope):
        lo, hi = bootstrap_mean(values)
        estimates.append({'contrast': label, 'n_units': len(values), 'unit': unit, 'mean_pp': np.mean(values), 'bootstrap_95_low_pp': lo, 'bootstrap_95_high_pp': hi, 'resampling_scope': scope})
    estimate('PyAOP: base effect 4OMe minus benzyl', shared.structural_double_difference_pp, 'shared acid partner', 'conditional on these two selected amines and observed archives')
    pivot = common.pivot(index='sub_2_smiles', columns='activator', values='structural_double_difference_pp')
    for activator in ORDER:
        estimate(f'{activator}: base effect 4OMe minus benzyl (same8)', pivot[activator], 'shared acid partner', 'same eight acids, selected amine pair')
    for activator in ORDER[1:]:
        estimate(f'PyAOP minus {activator}: structural double difference', pivot.PyAOP - pivot[activator], 'shared acid partner', 'paired activator-background contrast over same eight acids')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.1), layout='constrained')
    y = np.arange(len(shared))
    ax.hlines(y, shared.base_effect_benzyl_pp, shared.base_effect_4OMe_pp, color='#CBCBCB', lw=2)
    ax.scatter(shared.base_effect_benzyl_pp, y, color='#177E89', s=42, label='Benzylamine', zorder=3)
    ax.scatter(shared.base_effect_4OMe_pp, y, color='#B63849', s=42, label='4-Methoxybenzylamine', zorder=3)
    ax.axvline(0, color='0.4', lw=0.8)
    ax.set(yticks=y, yticklabels=shared.acid_label, xlabel='DIPEA - NMM yield effect (percentage points)', ylabel='Shared acid partner', title='A  PyAOP background: same 10 acids')
    ax.invert_yaxis()
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    wide = common.pivot(index='acid_label', columns='activator', values='structural_double_difference_pp')[ORDER]
    for _, row in wide.iterrows():
        bx.plot(range(4), row, color='#C6C6C6', marker='o', markersize=3, lw=0.9)
    for i, activator in enumerate(ORDER):
        lo, hi = bootstrap_mean(wide[activator])
        mean = wide[activator].mean()
        bx.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='o', color=COLORS[activator], ms=8, capsize=5, lw=2, zorder=5)
    bx.axhline(0, color='0.4', lw=0.8)
    bx.set(xticks=range(4), xticklabels=ORDER, ylabel='Base-effect difference: 4-OMe minus H (pp)', title='B  Activator context: same 8 acids')
    fig.supxlabel('Points represent observed acid partners; bars are acid-bootstrap intervals, conditional on the selected amine pair.', fontsize=9)
    save_figure(fig, '01_context_gated_benzylamine')
    heat = []
    for row in common.itertuples():
        for amine, suffix in [('4-OMe', 'target'), ('H', 'control')]:
            for base, prefix in [('DIPEA', 'yield_A'), ('NMM', 'yield_B')]:
                heat.append({'acid_label': row.acid_label, 'acid_smiles': row.sub_2_smiles, 'activator': row.activator, 'amine': amine, 'base': base, 'yield_percent': getattr(row, prefix + '_' + suffix) * 100})
    heat = pd.DataFrame(heat)
    heat.to_csv(cases / 'figure_common8_actual_yields.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.5), sharey=True, layout='constrained')
    combinations = [('H', 'DIPEA'), ('H', 'NMM'), ('4-OMe', 'DIPEA'), ('4-OMe', 'NMM')]
    for ax, activator in zip(axes, ORDER):
        matrix = heat[heat.activator.eq(activator)].pivot(index='acid_label', columns=['amine', 'base'], values='yield_percent')
        matrix = matrix.reindex(columns=pd.MultiIndex.from_tuples(combinations)).sort_index()
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
        for (i, j), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=8, color='white' if value > 57 else '#202020')
        ax.set(xticks=range(4), xticklabels=[f'{a}\n{b}' for a, b in combinations], yticks=range(len(matrix)), yticklabels=matrix.index, title=activator)
        ax.tick_params(length=0)
    axes[0].set_ylabel('Same eight acid partners')
    fig.colorbar(im, ax=axes, label='Observed yield (%)', shrink=0.8, pad=0.02)
    fig.supxlabel('H = benzylamine; 4-OMe = 4-methoxybenzylamine. Cells are matched-pair means, not independent repeats.', fontsize=9)
    save_figure(fig, '02_common8_actual_yields')
    profiles = pd.read_csv(cases / 'acylation_acid_class_profiles.csv')
    pair_means = pd.read_csv(cases / 'substrate_pair_means.csv')
    acid_pairs = pair_means.merge(profiles[['case', 'sub_2_smiles', 'acid_class']], on=['case', 'sub_2_smiles'], validate='many_to_one')
    partner_groups = acid_pairs.groupby(['case', 'sub_1_smiles', 'acid_class']).agg(mean_delta=('delta', 'mean'), n_acids=('sub_2_smiles', 'nunique')).reset_index()
    hetero = partner_groups[partner_groups.acid_class.eq('heteroaromatic')].drop(columns='acid_class')
    nonarom = partner_groups[partner_groups.acid_class.eq('nonaromatic')].drop(columns='acid_class')
    adjusted = hetero.merge(nonarom, on=['case', 'sub_1_smiles'], suffixes=('_hetero', '_nonarom'), validate='one_to_one')
    adjusted['contrast_pp'] = 100 * (adjusted.mean_delta_hetero - adjusted.mean_delta_nonarom)
    adjusted['at_least_two_acids_each_group'] = adjusted.n_acids_hetero.ge(2) & adjusted.n_acids_nonarom.ge(2)
    adjusted.to_csv(cases / 'acylation_common_amine_adjustment.csv', index=False)
    for case, part in adjusted.groupby('case'):
        for subset_name, subset in [('all shared amines', part), ('>=2 acids per group', part[part.at_least_two_acids_each_group])]:
            estimate(f'{case}: hetero-nonarom, {subset_name}', subset.contrast_pp, 'common amine partner', 'partner-distribution sensitivity; acid groups share the same amine, equal amine weights')
    class_order = ['heteroaromatic', 'carbocyclic_aryl', 'nonaromatic']
    class_colors = ['#177E89', '#7D7396', '#B63849']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout='constrained')
    for ax, case in zip(axes, ['EEDQ_minus_BTFFH', 'IIDQ_minus_BTFFH']):
        part = profiles[profiles.case.eq(case)]
        for i, (cls, color) in enumerate(zip(class_order, class_colors)):
            values = 100 * part.loc[part.acid_class.eq(cls), 'delta'].to_numpy()
            jitter = np.random.default_rng(5 + i).uniform(-0.15, 0.15, len(values))
            ax.scatter(i + jitter, values, color=color, s=22, alpha=0.7)
            lo, hi = bootstrap_mean(values)
            mean = values.mean()
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='_', color='black', ms=23, capsize=5, lw=1.4)
            ax.text(i, -51, f'n={len(values)}', ha='center', fontsize=9)
            estimate(f'{case}: {cls}', values, 'acid molecule', 'each acid averages its observed matched amines')
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set(xticks=range(3), xticklabels=['Heteroaromatic\nacid', 'Carbocyclic aryl\nacid', 'Nonaromatic\nacid'], title=case.replace('_minus_', ' - '), ylim=(-56, 50))
        values1 = 100 * part.loc[part.acid_class.eq('heteroaromatic'), 'delta'].to_numpy()
        values0 = 100 * part.loc[part.acid_class.eq('nonaromatic'), 'delta'].to_numpy()
        rng = np.random.default_rng(20261016)
        boot = values1[rng.integers(len(values1), size=(20000, len(values1)))].mean(1) - values0[rng.integers(len(values0), size=(20000, len(values0)))].mean(1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        estimates.append({'contrast': case + ': heteroaromatic minus nonaromatic', 'n_units': len(values1) + len(values0), 'unit': 'acid molecule', 'mean_pp': values1.mean() - values0.mean(), 'bootstrap_95_low_pp': lo, 'bootstrap_95_high_pp': hi, 'resampling_scope': 'separate acid bootstrap within observed structural groups'})
    axes[0].set_ylabel('Activator yield effect (percentage points)')
    fig.supxlabel('Each dot is one acid after averaging its matched amine partners. Bars: acid-bootstrap 95% intervals.', fontsize=9)
    save_figure(fig, '03_activator_acid_structure')
    pd.DataFrame(estimates).to_csv(ROOT / 'results' / 'named_contrast_uncertainty.csv', index=False)

def make_atlas():
    swaps = pd.read_csv(ROOT / 'results' / 'matched_swap_summary.csv')
    coverage = pd.read_csv(ROOT / 'results' / 'dataset_matching_coverage.csv').sort_values('pairs_with_at_least_20_targets')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 7.2), layout='constrained', gridspec_kw={'width_ratios': [1, 1.3]})
    ax.barh(coverage.dataset_id, coverage.pairs_with_at_least_20_targets, color='#177E89')
    ax.set(xlabel='Matched single-slot recipe comparisons', title='A  Coverage of existing data')
    for i, value in enumerate(coverage.pairs_with_at_least_20_targets):
        ax.text(value + 5, i, str(value), va='center', fontsize=7.5)
    ax.set_xlim(0, 675)
    for kind, color, label in [('identity_same_dose', '#177E89', 'Identity, same dose'), ('solvent_identity', '#B63849', 'Solvent identity'), ('addition_removal', '#807090', 'Addition / removal')]:
        part = swaps[swaps.swap_kind.eq(kind)]
        bx.scatter(abs(part.mean_delta) * 100, part.sd_delta * 100, s=9, alpha=0.25, color=color, label=f'{label} (n={len(part):,})')
    bx.set(xlabel='Absolute average yield effect (pp)', ylabel='Between-target SD of matched yield effects (pp)', title='B  Average response and target heterogeneity')
    bx.legend(frameon=False, fontsize=8, loc='lower right')
    save_figure(fig, '04_matched_component_atlas')
    stats = {'n_swaps': len(swaps), 'swap_kind_counts': swaps.swap_kind.value_counts().to_dict(), 'bidirectional_both_5pp_at_least_10pct_targets': int(((swaps.p_delta_gt_5pp >= 0.1) & (swaps.p_delta_lt_minus5pp >= 0.1)).sum()), 'near_zero_mean_large_sd': int((swaps.mean_delta.abs().lt(0.02) & swaps.sd_delta.ge(0.15)).sum()), 'all_true_flag_same_mean_count': int(np.isclose(swaps.mean_delta, swaps.mean_delta_source_check_true, equal_nan=False).sum()), 'true_flag_missing_swaps': int(swaps.mean_delta_source_check_true.isna().sum())}
    (ROOT / 'results' / 'atlas_descriptive_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

def make_additive_rule():
    profiles = pd.read_csv(ROOT / 'results' / 'structure_named_rule_profiles.csv')
    rule = profiles[profiles.rule_name.eq('acylation_DCC_PFP_to_NHS_aniline')].copy()
    assert len(rule) == 86 and rule.smiles.is_unique
    strata = pd.read_csv(ROOT / 'results' / 'structure_shared_partner_strata.csv')
    strata = strata[strata.pair_id.eq(rule.pair_id.iloc[0]) & strata.feature.eq('aniline')].copy()
    assert len(strata) == 62
    rule.to_csv(ROOT / 'cases' / 'figure_DCC_additive_molecule_profiles.csv', index=False)
    strata.to_csv(ROOT / 'cases' / 'figure_DCC_additive_common_acids.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout='constrained')
    ax, bx, cx = axes
    for group, color in [(0, '#177E89'), (1, '#B63849')]:
        part = rule[rule.feature_group.eq(group)]
        offset = 3 * group
        for row in part.itertuples():
            ax.plot([offset, offset + 1], [100 * row.y0, 100 * row.y1], color=color, alpha=0.17, lw=0.8)
        ax.plot([offset, offset + 1], [100 * part.y0.mean(), 100 * part.y1.mean()], color=color, marker='o', ms=7, lw=3)
        rng = np.random.default_rng(20261016 + group)
        bx.scatter(group + rng.uniform(-0.17, 0.17, len(part)), part.delta * 100, s=23, alpha=0.65, color=color)
        bx.plot([group - 0.2, group + 0.2], [part.delta.mean() * 100] * 2, color='black', lw=2.2)
    ax.set(xticks=[0, 1, 3, 4], xticklabels=['PFP', 'NHS', 'PFP', 'NHS'], ylim=(0, 100), ylabel='Mean matched-partner yield per amine (%)', title='A  Actual yields for each amine')
    ax.text(0.2, 0.94, 'No aniline\nn=40', ha='center', va='top', transform=ax.transAxes, fontsize=9)
    ax.text(0.8, 0.94, 'Aniline\nn=45', ha='center', va='top', transform=ax.transAxes, fontsize=9)
    bx.axhline(0, lw=0.8, color='0.45')
    bx.set(xticks=[0, 1], xticklabels=['No aniline\nn=40', 'Aniline\nn=45'], ylabel='NHS - PFP yield effect (pp)', title='B  Opposing structure-group responses')
    cx.scatter(100 * strata.mean_delta_group0, 100 * strata.mean_delta_group1, s=24, alpha=0.6, color='#565656')
    cx.scatter(100 * strata.mean_delta_group0.mean(), 100 * strata.mean_delta_group1.mean(), s=85, marker='D', color='#B63849', edgecolors='white', zorder=4)
    bounds = (-45, 45)
    cx.plot(bounds, bounds, ls='--', color='#BEBEBE', lw=0.8)
    cx.axhline(0, color='0.6', lw=0.7)
    cx.axvline(0, color='0.6', lw=0.7)
    cx.set(xlim=bounds, ylim=bounds, xlabel='No-aniline mean effect per acid (pp)', ylabel='Aniline mean effect per acid (pp)', title='C  Same 60 acid partners')
    fig.supxlabel('DCC 1.5 eq / additive 1.5 eq / DIPEA 3 eq / DMF / 25 C / 2 h. PFP = pentafluorophenol.', fontsize=9)
    save_figure(fig, '06_DCC_additive_amine_structure')

def main():
    (ROOT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    make_named_evidence()
    make_atlas()
    make_additive_rule()
    import importlib.metadata
    versions = {key: importlib.metadata.version(key) for key in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'rdkit', 'matplotlib', 'openpyxl']}
    (ROOT / 'results' / 'runtime_versions.json').write_text(json.dumps({'python': platform.python_version(), 'packages': versions}, indent=2), encoding='utf-8')
    print(pd.read_csv(ROOT / 'results' / 'named_contrast_uncertainty.csv').to_string(index=False))
    print((ROOT / 'results' / 'atlas_descriptive_summary.json').read_text())
if __name__ == '__main__':
    main()
