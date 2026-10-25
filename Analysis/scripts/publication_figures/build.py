"""Build the approved four-figure narrative and eight Extended Data figures."""
from __future__ import annotations
import argparse
import json
import re
import numpy as np
import pandas as pd
from matplotlib.text import Text
from publication_figures import common as c, figures14, figures23, redesign_figure1, redesign_figures234
from m1_chemistry import Chemistry
RESPONSE_LEARNING = c.A / 'scripts/response_learning'
E66 = RESPONSE_LEARNING / 'Family_Information_Recovery_E66/results'

def extended_inventory():
    support_path = RESPONSE_LEARNING / 'Full_Edge_Corpus_E59/results/full_edge_corpus_class_support_20.csv'
    corpus_path = support_path.parent / 'full_edge_response_corpus.parquet'
    support = c.frame(support_path)
    pairs = pd.read_parquet(corpus_path, columns=['reaction_type', 'substrate_pair_key'])
    pairs = pairs.groupby('reaction_type').substrate_pair_key.nunique().rename('substrate_pairs')
    data = support.loc[support.reaction_type.ne('finkelstein')].merge(pairs, on='reaction_type', validate='one_to_one')
    data = data.rename(columns={'corpus_rows': 'observed_responses', 'raw_observation_rows': 'raw_records'})
    data = data.sort_values(['strict_swaps', 'reaction_type'], ascending=[False, True])
    if len(data) != 19 or int(data.raw_records.sum()) != 406909:
        raise ValueError('The nineteen-array inventory changed')
    c.source('S1_array_inventory', data, [support_path, corpus_path], 'All non-Finkelstein arrays; unique within-array substrate pairs supporting exact responses', 'counts')
    fig = c.newfig(140)
    ax = c.panel(fig, [0.33, 0.13, 0.59, 0.78], 'a', 'Reaction arrays')
    y = np.arange(len(data))
    ax.barh(y, data.strict_swaps, height=0.65, color=c.BLUE)
    ax.set_yticks(y, data.reaction_type.map(c.class_label))
    ax.invert_yaxis()
    ax.set_xlim(0, 660)
    c.clean(ax, 'Exact substitutions')
    for i, row in enumerate(data.itertuples()):
        ax.text(row.strict_swaps + 6, i, f'{row.strict_swaps:,}', va='center', fontsize=5.5)
        ax.text(650, i, f'{row.substrate_pairs:,}', ha='right', va='center', fontsize=5.5)
    ax.text(650, -1.2, 'Substrate pairs', ha='right', fontsize=5.5)
    return c.save(fig, 'Extended_Data_Figure_1', 'Exact substitutions span nineteen reaction arrays')

def extended_budgets():
    path = E66 / 'e66_summary.csv'
    raw = c.frame(path)
    budgets = [0, 1, 2, 4, 8, 16]
    data = raw.loc[raw.population.eq('common16') & raw.target_geometry.eq('raw_delta') & raw.stratum.eq('all') & raw.weighting.eq('equal_class_edge_balanced') & raw.learner.isin(['joint', 'calibration_only', 'frozen_family']) & raw.budget.isin([str(n) for n in budgets])].copy()
    baseline = data.loc[data.learner.eq('frozen_family'), 'native_vector_mse']
    if baseline.empty or not np.allclose(baseline, baseline.iloc[0], rtol=0, atol=1e-09):
        raise ValueError('The fixed historical reference changed')
    if not data.n_test_vectors.eq(22963).all():
        raise ValueError('The sixteen-profile population changed')
    data['n_vectors'] = data.budget.astype(int)
    data['error_change_pct'] = 100 * (data.native_vector_mse / baseline.iloc[0] - 1)
    scope_path = E66 / 'e66_scope_vectors.parquet'
    scope = c.frame(scope_path)
    scope = scope.loc[scope.common16]
    if len(scope) != 22963:
        raise ValueError('Curve and measurement-cost denominators differ')
    costs = pd.DataFrame([dict(n_vectors=n, n_backgrounds=int(b), local_wells=2 * int(b) * n, n_query_vectors=int(count)) for n in budgets for b, count in scope.n_backgrounds.value_counts().sort_index().items()])
    c.source('S2_local_learning', data, path, 'common16/raw_delta/all/equal_class_edge_balanced; original budgets and learners', 'percent MSE change vs history')
    c.source('S2_measurement_cost', costs, scope_path, 'common16; distinct condition cells=2*backgrounds*profiles', 'condition cells')
    fig = c.newfig(95)
    ax = c.panel(fig, [0.14, 0.22, 0.78, 0.65], 'a', 'Local measurement budgets')
    for learner, label, color in [('calibration_only', 'Local only', c.RED), ('joint', 'History + local', c.TEAL)]:
        rows = data.loc[data.learner.eq(learner)].sort_values('n_vectors')
        ax.plot([budgets.index(n) for n in rows.n_vectors], rows.error_change_pct, 'o-', color=color, ms=3, label=label)
    ax.axhline(0, color=c.BLUE, lw=0.7, label='History only')
    labels = ['0' if n == 0 else f"{costs.loc[costs.n_vectors.eq(n), 'local_wells'].min()}–{costs.loc[costs.n_vectors.eq(n), 'local_wells'].max()}" for n in budgets]
    ax.set_xticks(range(len(budgets)), labels)
    ax.set_xlim(-0.1, 5.45)
    ax.set_ylim(-8, 90)
    ax.set_yticks([0, 40, 80])
    c.clean(ax, 'Local condition cells (ordered budget categories)', 'Error change relative to history (%)')
    ax.legend(frameon=False, loc='upper right')
    ax.text(0.02, 0.96, '22,963 query vectors', transform=ax.transAxes, va='top', fontsize=5.5)
    return c.save(fig, 'Extended_Data_Figure_2', 'The larger local budget retains a fixed sixteen-profile population')

def legacy_extended(number, legacy):
    """Keep accepted SI data selections while using the final physical artwork style."""
    labels = {6: {'a': 'Acid fluorination', 'b': 'Amine-class responses', 'c': 'Activator-dependent base responses'}, 7: {'a': 'Reference difference beyond mean activity', 'b': 'Local information components'}, 8: {'a': 'Shared Suzuki scope', 'b': 'Amide scope compromise'}}
    original_add = legacy.add_table
    marker = []

    def add(name, data, source, selector, unit, note=''):
        prefix = {6: 'S1', 7: 'S2', 8: 'S3'}[number]
        name = name.replace(prefix + '_', f'S{number}_', 1)
        value = original_add(name, data, source, selector, unit, note)
        c.source(name, value, source, selector, unit, note)
        return value

    def new(_):
        fig = c.newfig({6: 120, 8: 105}.get(number, 145))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        return fig

    def label(fig, x, y, letter, _):
        if number == 6 and letter == 'a':
            y = 0.85
        title = fig.text(x + 0.018, y, labels[number][letter], fontsize=6, weight='bold', va='bottom')
        text = fig.text(x - 0.007, y, letter.upper(), fontsize=8, weight='bold', va='bottom')
        marker.append((letter, title, text, x, y))
    legacy.new_figure = new
    legacy.molecule = c.molecule
    legacy.add_table = add
    legacy.sb.label = label
    legacy.sb.table_heatmap = legacy.table_heatmap
    try:
        fig, title = {6: lambda: legacy.si_chemical(None), 7: legacy.si_information, 8: legacy.si_routes}[number]()
    finally:
        legacy.add_table = original_add
    fig.axes[0].remove()
    for ax in fig.axes:
        ax.set_title('', loc='left')
        ax.set_title('')
        ticks = ax.get_yticks()
        tick_labels = [item.get_text() for item in ax.get_yticklabels()]
        if any(('|' in value for value in tick_labels)):
            ax.set_yticks(ticks, [c.screen_label(value) for value in tick_labels])
    for text in fig.findobj(Text):
        if text not in [m[2] for m in marker]:
            text.set_fontsize(max(5.2, min(7, text.get_fontsize() * 0.65)))
    for letter, title_text, letter_text, x, y in marker:
        candidates = [ax for ax in fig.axes if ax.axison]
        ax = min(candidates, key=lambda ax: abs(ax.get_position().x0 - x) + abs(ax.get_position().y1 - y))
        ax._draft_panel_letter = letter_text
        fig._m1_panels.append((letter, ax, title_text))
    return c.save(fig, f'Extended_Data_Figure_{number}', title)
CAPTIONS = {1: 'Bars give exact substitution counts for nineteen non-Finkelstein arrays. Bar-end numbers count substitutions; the right column counts distinct substrate pairs within each array. Counts of pairs are not additive across arrays. The retained source corpus contains 410,587 records in twenty tables after 396 exclusions; the nineteen arrays shown contain 406,909 records.', 2: 'Original local-update curves for the separate fixed population of 22,963 queries supporting sixteen local profiles. Budgets are 0, 1, 2, 4, 8 and 16 profiles; points are ordered categories labelled with the minimum–maximum number of distinct condition cells. The original equal-group, edge-balanced weighting and frozen historical reference are retained. One complete profile contains two reagent endpoints in each included recipe background. Values show changes relative to history alone, without refitting the models.', 6: '**a**, All sixteen measured yields for acid fluorination and two anilines. Activator and additive are each 1.5 equiv in DMF at 25 °C for 2 h, without base. **b**, FDPP-to-TOTU response contrasts between primary aryl and primary alkyl amines within 42 common acid partners; the red line marks their mean of 33.87 pp. **c**, NMM-to-DIPEA difference-in-differences: the response for 4-methoxybenzylamine minus that for benzylamine over the same eight acids. Points are acids and red ticks their means.', 7: '**a**, Information added by the observed reference difference D beyond mean reference activity S, expressed as the intended-swap improvement minus the matched different-swap improvement in each screen. The pair-held-out, full-history, context-balanced complete-panel population is retained. **b**, Fixed-operator label-content contrasts for Buchwald–Hartwig and Ullmann reactions separate local level and fluctuation information. Negative changes reduce error; the Ullmann fluctuation contrast is near zero. These are predictive accounting contrasts, rather than randomized experimental effects.', 8: '**a**, Best measured route bottlenecks for two Suzuki scopes separately and after combining them on identical native support. The bracket marks the 13.48 pp yield sacrifice. **b**, Exact Pareto frontier between amide source-scope sacrifice and target-scope bottleneck. The dashed line marks 20% target yield; a source-scope sacrifice of at least 5.00 pp is required to reach it. Displayed values use two decimals; sacrifices are calculated before rounding. The routes use measured recipes. Molecular identities, measured route nodes and raw-record links accompany the source tables. These comparisons search the measured recipes and do not establish recovery in unmeasured conditions.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-workbooks', action='store_true')
    parser.add_argument('--source-data-only', action='store_true')
    args = parser.parse_args()
    c.OUT.mkdir(parents=True, exist_ok=True)
    if args.source_data_only:
        from m1_source_data import build_workbooks
        build_workbooks(c.OUT / 'figure_manifest.json', c.ROOT / 'Submission/04_Source_Data', c.OUT)
        return
    from publication_figures.direction_accuracy import ensure
    ensure()
    import m1_figures as legacy
    c.CHEM = Chemistry(c.FIG / 'ChemDraw', c.OUT / 'chemistry_work')
    c.QA.clear()
    all_tables, main, extended = ([], [], [])
    builders = [(1, redesign_figure1.figure1), (2, redesign_figures234.figure2), (3, redesign_figures234.figure3), (4, redesign_figures234.figure4)]
    builders += [(n, func) for n, func in [(1, extended_inventory), (2, extended_budgets), (3, figures23.extended_cross), (4, figures23.extended_structure), (5, figures23.extended_reference)]]
    builders += [(n, lambda n=n: legacy_extended(n, legacy)) for n in [6, 7, 8]]
    approved_text = json.loads((c.A / 'scripts/templates/FIGURE_TEXT.json').read_text(encoding='utf-8'))
    for index, (number, builder) in enumerate(builders):
        c.TABLES.clear()
        report = builder() or c.QA[-1]
        is_main = index < 4
        stem = f'Figure_{number}' if is_main else f'Extended_Data_Figure_{number}'
        full_caption = {**figures14.CAPTIONS, **figures23.CAPTIONS}.get(stem)
        if full_caption:
            match = re.match('\\*\\*[^|]+\\| (.+?)\\.\\*\\*\\s*(.*)', full_caption, re.S)
            if not match:
                raise ValueError(f'Unrecognized caption: {stem}')
            title, caption = match.groups()
        else:
            title, caption = (report['title'], CAPTIONS[number])
        if stem in approved_text:
            title, caption = (approved_text[stem]['title'], approved_text[stem]['legend'])
        entry = {'number': number, 'title': title, 'caption': caption, 'legend': caption, 'pdf': str(c.FIG / f'{stem}.pdf'), 'png': str(c.FIG / f'{stem}.png'), 'files': {ext: str(c.FIG / f'{stem}.{ext}') for ext in ['pdf', 'svg', 'png']}, 'panels': [p['panel'] for p in report['panels']], 'single_composite': report.get('single_composite', False)}
        if is_main:
            main.append(entry)
        else:
            entry['si_notes'] = {1: [1], 2: [7], 3: [5], 4: [6], 5: [4], 6: [3], 7: [4, 7], 8: [8]}[number]
            extended.append(entry)
        all_tables.extend(c.TABLES)
    c.TABLES.clear()
    original_add = legacy.add_table

    def technical(name, data, source, selector, unit, note=''):
        name = re.sub('^S\\d+_', 'SI_', name)
        value = original_add(name, data, source, selector, unit, note)
        return c.source(name, value, source, selector, unit, note)
    legacy.add_table = technical
    try:
        legacy.technical_tables()
    finally:
        legacy.add_table = original_add
    all_tables.extend(c.TABLES)
    if len({x['name'] for x in all_tables}) != len(all_tables):
        raise ValueError('Duplicate Source Data table names')
    chemistry = c.CHEM.finalize()
    manifest = {'central_statement': 'Reagent-swap effects are learnable but not copyable', 'figure_width_mm': 183, 'figures': main, 'main_figures': main, 'extended_data_figures': extended, 'supplementary_figures': [], 'source_data': all_tables, 'supplementary_tables': [{'name': e['name'], 'file': e['csv'], 'source': e['source'], 'filter': e['selector']} for e in all_tables if e['name'].startswith('S')], 'chemistry': str(c.FIG / 'ChemDraw/chemistry_assets.json')}
    (c.OUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (c.OUT / 'figure_checks.json').write_text(json.dumps({'figures': c.QA, 'native_structures': sum((e['native'] for e in chemistry['assets'])), 'structure_count': len(chemistry['assets']), 'fallbacks': chemistry['fallbacks']}, indent=2), encoding='utf-8')
    if not args.skip_workbooks:
        from m1_source_data import build_workbooks
        build_workbooks(c.OUT / 'figure_manifest.json', c.ROOT / 'Submission/04_Source_Data', c.OUT)
    print(f'Generated {len(c.QA)} vector figures and {len(all_tables)} Source Data tables', flush=True)
if __name__ == '__main__':
    main()
