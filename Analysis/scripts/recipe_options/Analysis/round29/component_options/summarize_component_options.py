"""Produce chemical route tables and the scientific account from fixed outputs."""
from pathlib import Path
import hashlib
import json
import pandas as pd
OUT = Path(__file__).resolve().parent

def read(name):
    return pd.read_csv(OUT / name, keep_default_na=False, float_precision='round_trip')

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g')

def main():
    cert = read('certificate_option_summary.csv')
    single = read('all_single_options.csv.gz')
    pair = read('all_pair_options.csv.gz')
    triple = read('all_triple_options.csv.gz')
    class_summary = read('complete_class_option_summary.csv')
    run_summary = json.loads((OUT / 'run_summary.json').read_text(encoding='utf-8'))
    registry = read('named_option_registry.csv').drop_duplicates('option_id').set_index('option_id')
    route = read('chemical_witness_routes.csv')
    conditions = read('chemical_witness_conditions.csv')
    nodes = read('chemical_witness_full_node_yields.csv')
    source = read('chemical_witness_source_records.csv.gz')
    verification_path = OUT / 'independent_verification.json'
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    raw_path = OUT / 'chemical_witness_raw_verification.json'
    raw_verification = json.loads(raw_path.read_text(encoding='utf-8'))
    assert verification.get('status') == 'passed' and verification.get('fresh_independent_verification') is True
    assert raw_verification.get('status') == 'passed' and raw_verification.get('fresh_raw_readback') is True
    fresh_witnesses = sorted(raw_verification['witness_certificates'])
    raw_witnesses = {row['certificate_id']: row for row in raw_verification['new_witnesses']}
    assert len(fresh_witnesses) == 5 and len(set(fresh_witnesses)) == 5
    assert len(raw_verification['new_witnesses']) == len(fresh_witnesses)
    assert set(fresh_witnesses) == set(raw_witnesses) == set(source.certificate_id) == set(nodes.certificate_id) == set(route.certificate_id)
    assert not raw_verification['reused_R28_verifications']
    witness_ids = {'alternatives': 'g110_n5_n38', 'necessary': 'g22733_n4_n74', 'two_option': 'g35658_n6_n29', 'three_option': 'g43931_n12_n51', 'substitutable': 'g11926_n29_n34'}
    assert set(witness_ids.values()) == set(fresh_witnesses), 'Witness role IDs require fresh identity proof'
    complete = cert[cert.complete_endpoint_corners]
    incomplete = cert[~cert.complete_endpoint_corners]
    distribution = []
    for name, z in [('all', cert), ('complete', complete), ('incomplete', incomplete)]:
        for minimum in [1, 2, 3]:
            part = z[z.minimum_extra_options_exact == minimum]
            distribution.append({'endpoint_corner_stratum': name, 'minimum_extra_options': minimum, 'n_certificates': len(part), 'n_triples': part.group_id.nunique(), 'n_exact_molecular_pairs': part.molecular_pair_id.nunique(), 'n_classes': part.rxn_class.nunique()})
    save(pd.DataFrame(distribution), 'minimum_option_distribution.csv')
    unique = cert[(cert.n_single_sufficient == 1) & (cert.n_necessary_options_in_full_graph == 0)]
    save(single[single.certificate_id.isin(unique.certificate_id) & single.single_sufficient20], 'unique_singleton_but_substitutable.csv')
    sufficient_combinations = []
    for number, filename in [(2, 'all_pair_options.csv.gz'), (3, 'all_triple_options.csv.gz')]:
        combos = read(filename)
        good = combos[combos['pair_sufficient20' if number == 2 else 'triple_sufficient20']]
        for row in good.to_dict('records'):
            for i in range(1, number + 1):
                description = registry.loc[row[f'option{i}_id']]
                for col in ['option_name', 'dose', 'dose_unit', 'component_id']:
                    row[f'option{i}_{col}'] = description[col]
            row['minimum_extra_options'] = number
            sufficient_combinations.append(row)
    save(pd.DataFrame(sufficient_combinations), 'all_sufficient_minimum_multi_option_combinations.csv')
    yy = nodes.pivot(index=['certificate_id', 'recipe_id'], columns='partner_position', values=['yield_a', 'yield_b'])
    yy.columns = [f'partner{p}_{side}' for side, p in yy.columns]
    yy = yy.reset_index()
    rows = route.merge(conditions, on=['certificate_id', 'recipe_id'], validate='many_to_one').merge(yy, on=['certificate_id', 'recipe_id'], validate='many_to_one')
    for i in [1, 2, 3]:
        rows[f'option{i}_name'] = rows[f'option{i}_id'].map(registry.option_name).fillna('')
        rows[f'option{i}_dose'] = rows[f'option{i}_id'].map(registry.dose).fillna('')
    pointers = source.groupby(['certificate_id', 'recipe_id']).record_id.agg(lambda x: json.dumps(sorted(x.tolist())))
    rows = rows.merge(pointers.rename('all_six_target_source_record_ids').reset_index(), on=['certificate_id', 'recipe_id'], validate='many_to_one')
    save(rows, 'chemical_witness_routes_with_six_yields_and_sources.csv')
    route_options = []
    slots = [f'Reagent{i}_token' for i in range(1, 7)] + ['Solvent_token', 'Temperature_token', 'Time_token']
    for key, rr in rows.groupby(['certificate_id', 'route_kind', 'option1_id', 'option2_id', 'option3_id'], dropna=False):
        rr = rr.sort_values('step')
        outside = []
        for slot in slots:
            for token in sorted(set(rr[slot]) - {rr.iloc[0][slot], rr.iloc[-1][slot]}):
                description = registry[(registry.slot == slot) & (registry.token == token)]
                assert len(description) == 1
                d = description.iloc[0]
                outside.append({'slot': slot, 'token': token, 'option_name': d.option_name, 'dose': d.dose, 'option_id': description.index[0]})
        route_options.append(dict(zip(['certificate_id', 'route_kind', 'option1_id', 'option2_id', 'option3_id'], key)) | {'n_distinct_extra_options_on_route': len(outside), 'extra_options': json.dumps(outside)})
    save(pd.DataFrame(route_options), 'chemical_witness_distinct_route_options.csv')
    evidence = []
    for cid, part in source.groupby('certificate_id'):
        nn = nodes[nodes.certificate_id == cid]
        means = part.groupby(['block_id', 'recipe_id']).yield_numeric.mean()
        error = 0.0
        for r in nn.to_dict('records'):
            for side in ['a', 'b']:
                error = max(error, abs(means.loc[r['block_' + side], r['recipe_id']] - r['yield_' + side]))
        evidence.append({'certificate_id': cid, 'n_reaction_source_records': len(part), 'n_source_blocks': part.block_id.nunique(), 'n_conditions': nn.recipe_id.nunique(), 'n_target_recipe_values': 2 * len(nn), 'source_mean_maximum_error': error, 'all_source_pointers_present': bool(part.record_id.ne('').all()), 'raw_readback_reused_from_R28': False, 'fresh_raw_workbook_readback': cid in fresh_witnesses})
    save(pd.DataFrame(evidence), 'witness_source_pointer_and_mean_reconciliation.csv')
    fixed = single[single.single_sufficient20 & ~single.endpoint_changing_slot]
    need = single[single.necessary_in_full_graph20]
    nmultiple = int((cert.n_single_sufficient >= 2).sum())
    scope = read('named_option_descriptive_scope.csv')
    top = scope[scope.n_sufficient_molecular_pairs >= 4].sort_values(['n_sufficient_molecular_pairs', 'n_single_sufficient_certificates'], ascending=False)
    top_table = '\n'.join((f'| {r.option_name} | {r.dose} | {r.n_single_sufficient_certificates} | {r.n_sufficient_triples} | {r.n_sufficient_molecular_pairs} | {r.n_strong_endpoint_certificates_tested} |' for r in top.itertuples()))
    (OUT / 'witness_source_status.json').write_text(json.dumps({'status': 'all_five_witnesses_raw_readback_passed', 'fresh_raw_readback': True, 'reused_raw_verified_witnesses': [], 'freshly_raw_verified_witnesses': fresh_witnesses, 'newly_raw_verified_witnesses': fresh_witnesses, 'verification_file': str(raw_path)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def witness(role):
        selected = cert[cert.certificate_id == witness_ids[role]]
        assert len(selected) == 1, (role, 'missing or repeated witness certificate')
        return selected.iloc[0]

    def option(role, oid):
        selected = single[(single.certificate_id == witness_ids[role]) & (single.option_id == oid)]
        assert len(selected) == 1, (role, oid, 'missing or repeated option')
        return selected.iloc[0]

    def named_option(role, name, dose):
        ids = registry[(registry.option_name.str.casefold() == name.casefold()) & (pd.to_numeric(registry.dose, errors='coerce') == dose)].index
        selected = single[(single.certificate_id == witness_ids[role]) & single.option_id.isin(ids)]
        assert len(selected) == 1, (role, name, dose, 'missing or repeated named option')
        return selected.iloc[0]

    def witness_route(role, kind, oid=''):
        rr = route[(route.certificate_id == witness_ids[role]) & (route.route_kind == kind) & (route.option1_id == oid)].sort_values('step')
        assert len(rr), (role, kind, oid, 'missing witness route')
        return rr

    def sequence(role, kind, oid=''):
        rr = witness_route(role, kind, oid)
        return ', '.join((f'{v * 100:.5f}%' for v in rr.shared_node_yield))

    def percent(value):
        return f'{float(value) * 100:.5f}%'

    def route_deletions(role, kind, oid):
        first = witness_route(role, kind, oid).iloc[0]
        return '; '.join((f"{registry.loc[first[f'option{i}_id']].option_name} {registry.loc[first[f'option{i}_id']].dose} eq: {percent(option(role, first[f'option{i}_id']).K_after_full_deletion)}" for i in range(1, 4) if first[f'option{i}_id']))

    def scope_pairs(name, dose):
        ids = scope[(scope.option_name.str.casefold() == name.casefold()) & (pd.to_numeric(scope.dose, errors='coerce') == dose)].option_id
        assert len(ids), (name, dose, 'missing named scope')
        return single[single.option_id.isin(ids) & single.single_sufficient20].molecular_pair_id.nunique()
    minimum_counts = cert.minimum_extra_options_exact.value_counts()
    complete_counts = complete.minimum_extra_options_exact.value_counts()
    incomplete_counts = incomplete.minimum_extra_options_exact.value_counts()
    n_opportunities = int(class_summary[class_summary.endpoint_corner_stratum == 'all'].n_original_opportunity_triples.sum())
    n_datasets, n_classes = (class_summary.dataset_id.nunique(), class_summary.rxn_class.nunique())
    singleton = cert[cert.n_single_sufficient == 1]
    minimum_two = cert[cert.minimum_extra_options_exact == 2]
    hoat = option('alternatives', '487f9246f61897c6')
    edc = option('alternatives', 'bed09ab4473a6ee4')
    hobt = named_option('alternatives', 'HOBt', 1.5)
    pybop = option('necessary', '095d36086b9b2e13')
    substitute_single = single[(single.certificate_id == witness_ids['substitutable']) & single.single_sufficient20]
    assert len(substitute_single) == 1
    substitute_single = substitute_single.iloc[0]
    related_three = cert[(cert.minimum_extra_options_exact == 3) & (cert.molecular_pair_id == witness('three_option').molecular_pair_id)]
    other_three = cert[(cert.minimum_extra_options_exact == 3) & (cert.molecular_pair_id != witness('three_option').molecular_pair_id)]
    other_three_classes = ', '.join((f'{name} {len(part)} certificates' for name, part in other_three.groupby('rxn_class'))) or 'none'
    report = f"# From shared bridges to specific chemical options: substitutable, indispensable and multi-step relays\n\nDate: 2026-10-17. Round 28 showed that component substitution orders that each work for one of three partners sometimes cannot be merged, while additional measured conditions can restore a route shared by all six targets. This round asks further: which reagent and dose does the restoration need? Is it just one usable choice, or a choice that cannot be omitted within the measured condition range? When one option is not enough, how many additional choices are needed at minimum?\n\n## New findings\n\n**Restoring shared scope has different chemical degrees of freedom: sometimes activators and additives provide alternative routes for each other, sometimes one specified activator is indispensable, and sometimes the same substitution can only be carried across three partners via two catalyst choices plus another base.** This distinction comes from testing all candidates on fixed endpoints and fixed six-target measured support.\n\n{len(cert)} strong coordination certificates have their minimum number of extra slot options fully determined: {int(minimum_counts.get(1, 0))} need 1, {int(minimum_counts.get(2, 0))} need 2, {int(minimum_counts.get(3, 0))} need 3. For the complete endpoint-corner stratum the counts are **{int(complete_counts.get(1, 0))}, {int(complete_counts.get(2, 0))}, {int(complete_counts.get(3, 0))}**, from {complete.group_id.nunique()} three-partner combinations and {complete.molecular_pair_id.nunique()} exact A/B molecule pairs; the other {len(incomplete)} incomplete-corner certificates have minimum-option distribution: 1 option {int(incomplete_counts.get(1, 0))}, 2 options {int(incomplete_counts.get(2, 0))}, 3 options {int(incomplete_counts.get(3, 0))}, from {incomplete.group_id.nunique()} three-partner combinations and {incomplete.molecular_pair_id.nunique()} exact molecule pairs. Different strata and minimum-count groups reuse combinations and molecule pairs and cannot be summed as independent experiments.\n\n| Endpoint-corner stratum | Minimum 1 option | Minimum 2 options | Minimum 3 options | All strong certificates |\n|---|---:|---:|---:|---:|\n| Complete | {int(complete_counts.get(1, 0))} | {int(complete_counts.get(2, 0))} | {int(complete_counts.get(3, 0))} | {len(complete)} |\n| Incomplete | {int(incomplete_counts.get(1, 0))} | {int(incomplete_counts.get(2, 0))} | {int(incomplete_counts.get(3, 0))} | {len(incomplete)} |\n\nAn option here is **slot × identity × dose**, not how many reagents are added together in one vial. Three extra options can include two different values of the same catalyst slot. Route arrows denote replacing one condition slot at a time; each node is a separate measured reaction condition. The six-target yield takes the minimum under each condition, and the optimal bottleneck is then taken over all routes.\n\n## Substitutable activators and additives: group {int(witness('alternatives').group_id)}\n\nA/B are `C#Cc1cccc(N)c1` and `CSc1cccc(N)c1`; the three acid partners are `COc1ccc(CC(=O)O)cc1`, `Cc1cc(C(=O)O)cc(Cl)n1`, `O=C(O)COc1ccccc1`; source acquisitions are in `chemical_witness_acquisition_contexts.csv`. Endpoints are DCC/Cl-HOBT and DIC/ethyl cyanoglyoxylate-2-oxime; all four corners measured, DMF, 25 C, 2 h, activator and both endpoint additives at 1.5 eq, other slots fixed. The best common six-target bottleneck at endpoint levels is {percent(witness('alternatives').K_endpoint_all)}; each partner's A/B bundle still has a high-yield path separately.\n\nWith only one extra option allowed, there are {int(witness('alternatives').n_single_sufficient)} mutually substitutable solutions; three chemical routes among them:\n\n| Extra slot option | Best six-target bottleneck | Common route |\n|---|---:|---|\n| HOAT, additive slot, 1.5 eq | {percent(hoat.K_single)} | DCC/Cl-HOBT → DCC/HOAT → DIC/HOAT → DIC/oxime additive |\n| EDC-HCl, activator slot, 1.5 eq | {percent(edc.K_single)} | DCC/Cl-HOBT → EDC-HCl/Cl-HOBT → EDC-HCl/oxime additive → DIC/oxime additive |\n| HOBt, additive slot, 1.5 eq | {percent(hobt.K_single)} | DCC/Cl-HOBT → DCC/HOBt → DIC/HOBt → DIC/oxime additive |\n\nThe HOAT single-option route has {len(witness_route('alternatives', 'single_option', '487f9246f61897c6'))} nodes, with six-target minimum yields in order {sequence('alternatives', 'single_option', '487f9246f61897c6')}; for the EDC-HCl route they are {sequence('alternatives', 'single_option', 'bed09ab4473a6ee4')}. After deleting the HOAT 1.5 eq conditions from the whole common support, the best full-graph bottleneck is still {percent(hoat.K_after_full_deletion)}. HOAT 1.5 eq is therefore a high-yield choice, but not a necessary choice for reaching 20%. HOAT 1 eq and 3 eq are two other dose tokens; neither restores this case in the single-option test. The native partner-pair supports differ; the conclusion of this case applies to the fixed six-target common support.\n\n## An indispensable activator: group {int(witness('necessary').group_id)}\n\nA is `O=C(O)c1ccc(C(F)(F)F)cn1`, B is `O=C(O)c1ccc(F)cn1`; the three partners are `CNOC`, `COc1ccc(N)cc1`, `Cc1ccc(Oc2ccc(N)cc2)cc1`. The native supports of all three partner pairs equal the same common support of {nodes[nodes.certificate_id == witness_ids['necessary']].recipe_id.nunique()} conditions.\n\nThe best bottleneck at endpoint levels from TPTU/no additive to HBTU/HOBt is {percent(witness('necessary').K_endpoint_all)}. Among all {int(witness('necessary').n_extra_options)} extra slot options, only PyBOP 1.5 eq is sufficient alone:\n\n`TPTU/no additive → PyBOP/no additive → PyBOP/HOBt → HBTU/HOBt`.\n\nThe route has {len(witness_route('necessary', 'single_option', '095d36086b9b2e13'))} nodes, with six-target minimum yields in order {sequence('necessary', 'single_option', '095d36086b9b2e13')}. After deleting all PyBOP 1.5 eq conditions, even with the rest of the complete measured graph allowed, the best bottleneck is {percent(pybop.K_after_full_deletion)}. This token is therefore both sufficient alone and indispensable in the measured graph of this case. Fixed: DIPEA 3 eq, DMF, 25 C, 2 h; endpoint activator/HOBt at 1.5 eq.\n\n## Two options must act together: group {int(witness('two_option').group_id)}\n\nA/B are para- and meta-methoxyphenylboronic acid: `COc1ccc(B(O)O)cc1`, `COc1cccc(B(O)O)c1`. Partners in order are `Brc1cn2ccnc2cn1`, `COc1cccnc1Br`, `Fc1ccnc(Br)c1`. Endpoints are PdCl2(dtbpf)/Cs2CO3 and PdCl2(PPh3)2/K2CO3, catalyst 0.05 eq, base 3 eq, fixed 1,4-dioxane/H2O (4:1), 100 C, 12 h.\n\nAll four endpoint corners are complete; the six-target common bottleneck is {percent(witness('two_option').K_endpoint_all)}, and each partner bundle is still feasible separately. {int(witness('two_option').n_extra_options)} single options all fail; of {len(pair[pair.certificate_id == witness_ids['two_option']]):,} double options, {int(witness('two_option').n_sufficient_pairs)} succeed; the displayed successful combination is **Pd(dppe)Cl2 0.05 eq + K3PO4 3 eq**. Its route:\n\n`PdCl2(dtbpf)/Cs2CO3 → PdCl2(dtbpf)/K3PO4 → Pd(dppe)Cl2/K3PO4 → Pd(dppe)Cl2/K2CO3 → PdCl2(PPh3)2/K2CO3`.\n\nSix-target minimum yields at the nodes, in order: {sequence('two_option', 'two_option', 'd4be0010fc039e7b')}. This gives exactly the minimum of two extra choices. The complete graph can detour further via PdCl2(PCy3)2, raising the best bottleneck to {percent(witness('two_option').K_all)}. After deleting Pd(dppe)Cl2 and K3PO4 separately, the best full-graph bottleneck is {route_deletions('two_option', 'two_option', 'd4be0010fc039e7b')}. Both options are necessary in this complete graph. The native partner supports differ.\n\n## Three-option relay: group {int(witness('three_option').group_id)}\n\nA/B are meta- and ortho-methoxyphenylboronic acid: `COc1cccc(B(O)O)c1`, `COc1ccccc1B(O)O`; the three partners are `COc1cc(C)c(Br)cn1`, `Cc1ccc(Br)nn1`, `Cc1ccnc(Br)c1`. The four corners of endpoints PCy3 Pd G3/K3PO4 and Pd(DPEPhos)Cl2/Cs2CO3 are complete; the endpoint common bottleneck is {percent(witness('three_option').K_endpoint_all)}.\n\n{int(witness('three_option').n_extra_options)} single options and {len(pair[pair.certificate_id == witness_ids['three_option']]):,} double options all fail; of {len(triple[triple.certificate_id == witness_ids['three_option']]):,} triple options, {int(witness('three_option').n_sufficient_triples)} succeed; the displayed combination is **PdCl2(dtbpf) 0.05 eq, Pd(dppe)Cl2 0.05 eq, K2CO3 3 eq**. Common route:\n\n`PCy3 Pd G3/K3PO4 → PdCl2(dtbpf)/K3PO4 → PdCl2(dtbpf)/K2CO3 → Pd(dppe)Cl2/K2CO3 → Pd(dppe)Cl2/Cs2CO3 → Pd(DPEPhos)Cl2/Cs2CO3`.\n\nThe route has {len(witness_route('three_option', 'three_option', 'd4be0010fc039e7b'))} nodes, with minimum yields in order {sequence('three_option', 'three_option', 'd4be0010fc039e7b')}. After deleting each of the above tokens in turn, the best full-graph bottlenecks are {route_deletions('three_option', 'three_option', 'd4be0010fc039e7b')}. Other slots are fixed; solvent, temperature, time and catalyst/base amounts are as in the previous Suzuki case. The three options here are a sequential relay of two catalyst levels and one base level. For the same A/B relation, {related_three.group_id.nunique()} three-partner combinations have a minimum of three options; the reaction classes of the other minimum-three-option certificates are {other_three_classes}.\n\n## A unique single option can still be replaced by a longer route\n\nThere are {len(unique)} certificates with only one single-option solution where the complete graph still reaches 20% after that option is deleted. Representative case: group {int(witness('substitutable').group_id)}, endpoints {int(witness('substitutable').node0)}→{int(witness('substitutable').node1)}, whose unique single-option solution is pentafluorophenol 1.5 eq; A/B are `COc1ccc(N)cc1` and `COc1ccc(N)cn1`, the three acid partners are `O=C(O)c1cc(Cl)cc(Cl)n1`, `O=C(O)c1ccc(C(F)(F)F)cn1`, `O=C(O)c1ccc(F)cn1`. Endpoints are NHPI/DIPEA and HOBt/TEA with EDC-HCl 1.5 eq fixed; additive 1.5 eq, base 3 eq, DMF, 25 C, 2 h.\n\nThe pentafluorophenol single option reaches {percent(substitute_single.K_single)}. After deleting it, the following measured route via three extra choices, DCC 1.5 eq, NHS 1.5 eq and NMM 3 eq, still reaches {percent(substitute_single.K_after_full_deletion)}:\n\n`EDC-HCl/NHPI/DIPEA → DCC/NHPI/DIPEA → DCC/NHS/DIPEA → DCC/NHS/NMM → EDC-HCl/NHS/NMM → EDC-HCl/HOBt/NMM → EDC-HCl/HOBt/TEA`.\n\nThis displayed route shows that three extra choices can replace pentafluorophenol; the minimum number of choices after deleting pentafluorophenol was not separately enumerated. A 'unique singleton' therefore cannot be rewritten directly as a 'necessary reagent'. These {len(unique)} complete records are in `unique_singleton_but_substitutable.csv`; the two routes, actual six-target yields and sources of the representative case are listed separately.\n\n## Actual option scope across molecular relations\n\nListed below is the descriptive scope over which the same identity, dose and slot succeed on the fixed certificates of this round. Each support is kept under the original {n_opportunities:,} opportunity registrations; different endpoints or three-partner combinations are not merged into one independent validation, and these results are not interpreted as predictions for new molecules.\n\n| Option | eq | Sufficient endpoint certificates | Three-partner combinations | Exact A/B pairs | Strong certificates tested |\n|---|---:|---:|---:|---:|---:|\n{top_table}\n\nThe single-option scope of PdCl2(dtbpf) 0.05 eq covers {scope_pairs('PdCl2(dtbpf)', 0.05)} exact molecule pairs; XPhos Pd G3 and PCy3 Pd G3 cover {scope_pairs('XPhos Pd G3', 0.05)} and {scope_pairs('PCy3 Pd G3', 0.05)} respectively; in Acylation, pentafluorophenol 1.5 eq covers {scope_pairs('pentafluorophenol', 1.5)} pairs, and HOAT 1.5 eq and NHS 1.5 eq cover {scope_pairs('HOAT', 1.5)} and {scope_pairs('NHS', 1.5)} pairs respectively. 'Strong certificates tested' here counts only candidate certificates where the token is not an endpoint value and actually occurs in that common support; the 'token measured' column over complete opportunities is in `named_option_descriptive_scope.csv`. It is not the same set as the success-rate denominator of this round.\n\n{nmultiple} certificates have at least two different single-option solutions; {len(singleton)} have only one single-option solution, of which {int((singleton.n_necessary_options_in_full_graph > 0).sum())} have that token necessary in the complete graph and {len(unique)} can be replaced by multiple options. In total, {int((cert.n_necessary_options_in_full_graph > 0).sum())} certificates have at least one necessary token, with {len(need)} necessary token × certificate states. Among the {len(minimum_two)} certificates whose minimum is two options, {int((minimum_two.n_necessary_options_in_full_graph == 2).sum())} have two necessary tokens and {int((minimum_two.n_necessary_options_in_full_graph == 1).sum())} have only one necessary token; the minimum number of choices and the number of necessary choices are therefore not the same.\n\nAll {len(single):,} single-option candidates are kept, of which {int((~single.single_sufficient20).sum()):,} fail. All {len(pair):,} double options and {len(triple):,} triple options are also kept (expanded only on certificates where all options of the previous level fail). Among single-option successes, {len(fixed)} token × certificate states vary a slot that is unchanged between endpoints, from {fixed.certificate_id.nunique()} certificates; the rest are in slots that change between endpoints. Both are saved separately by actual slot.\n\n## Denominators, algorithms and sources\n\nThe condition layer of this round follows the R28 **{int(run_summary['R28_complete_corner_bundle_feasible_endpoint_denominator20']):,} 20% endpoint views with complete corners where each partner's A/B bundle is feasible alone**; {len(complete)} complete-corner strong coordination certificates are the restorations within this condition denominator. {len(incomplete)} incomplete-corner certificates are listed separately. All {n_opportunities:,} compatible three-partner opportunities are kept; opportunities without common conditions, without high endpoints and from other reaction classes are not deleted. `complete_class_option_summary.csv` includes the zero results for all {n_datasets} datasets/{n_classes} reaction classes, and `complete_opportunity_registry.csv.gz` is the corresponding complete opportunity registry.\n\nThe complete graph contains only conditions measured for all six targets. Edges differ in exactly one complete slot token. The baseline allows only the two endpoint values per slot, with unchanged slots fixed; the sufficiency test for one extra token opens only that identity/dose in that slot, with other slots kept at endpoint values. Necessity is optimised after deleting all nodes with that slot token from the complete measured graph. All capacities are computed over all paths with a heap max-bottleneck algorithm, without restriction to shortest paths or fewest steps. Double- and triple-option enumeration includes different tokens in the same slot, so the minimum is the exact number of distinct added choices.\n\nIndependent verification status: `{verification['status']}`. The independent verifier also rebuilds all candidates, supports and six-target yields from the R27 tables, checks 20% feasibility with BFS, and computes continuous optimal bottlenecks with descending node activation/union-find. Verification covers all {verification['n_single_options']:,} single options, {verification['n_single_deletions']:,} complete-graph deletions, {verification['n_pair_options']:,} double options, {verification['n_triple_options']:,} triple options and the complete graph and baseline of each certificate, in total **{verification['n_continuous_states_checked']:,} continuous capacities and {verification['n_threshold_states_checked']:,} threshold states**; it checks {verification['n_saved_paths_checked']:,} saved paths, exhaustive-enumeration eligibility of all candidate combinations, exact minima and all {verification['n_dataset_corner_summary_rows_checked']} dataset × corner-stratum summaries. The maximum capacity/path difference is {verification['maximum_absolute_capacity_or_path_error']:.17g}. See `independent_verification.json`.\n\nAll {len(fresh_witnesses)} cases were re-read record by record from the original workbooks in this round, with status `{raw_verification['status']}`. In total, {raw_verification['n_deduplicated_raw_reaction_records']:,} deduplicated original reaction records and {raw_verification['n_deduplicated_raw_conditions']:,} conditions were read back, keeping {raw_verification['n_deduplicated_exact_zero_raw_records']:,} true zero values, and all {raw_verification['n_target_recipe_values_reconstructed']:,} target × recipe values were rebuilt, with maximum difference {max((row['maximum_yield_error'] for row in raw_witnesses.values())):.17g}. The condition check covers all reagent identities, doses, standard names and fixed solvent/temperature/time. `chemical_witness_raw_verification.json` records per-case counts for the five actual certificates, SHA256 of the original workbooks and the read-back proof of this round; no earlier read-back receipt is reused, and the row-level read-back table and rebuilt values are stored separately. Full stepwise conditions and six-target yields are in `chemical_witness_routes_with_six_yields_and_sources.csv`. Source tables and original files were not modified.\n\nThe synthetic-chemistry insight from these results: choosing one high-yield display route does not yet answer how to widen the common scope. The usable degree of freedom can belong to another additive, another activator, or a relay of several catalyst/base levels; these correspond to alternative routes, indispensable choices and multi-option coordination respectively.\n"
    (OUT / 'ROUND29_COMPONENT_OPTION_REPORT.md').write_text(report, encoding='utf-8')
    manifest = {}
    for path in OUT.iterdir():
        if path.is_file() and path.name != 'delivery_manifest.json':
            with path.open('rb') as stream:
                manifest[path.name] = {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    (OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'summarized', 'witness_mean_checks': evidence, 'independent_status': verification.get('status')}))
if __name__ == '__main__':
    main()
