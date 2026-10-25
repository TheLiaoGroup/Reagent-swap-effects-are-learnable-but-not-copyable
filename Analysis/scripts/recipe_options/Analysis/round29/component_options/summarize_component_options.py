"""Produce chemical route tables and the Chinese scientific account from fixed outputs."""
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
        return '、'.join((f'{v * 100:.5f}%' for v in rr.shared_node_yield))

    def percent(value):
        return f'{float(value) * 100:.5f}%'

    def route_deletions(role, kind, oid):
        first = witness_route(role, kind, oid).iloc[0]
        return '；'.join((f"{registry.loc[first[f'option{i}_id']].option_name} {registry.loc[first[f'option{i}_id']].dose} eq：{percent(option(role, first[f'option{i}_id']).K_after_full_deletion)}" for i in range(1, 4) if first[f'option{i}_id']))

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
    other_three_classes = '、'.join((f'{name} {len(part)} 个' for name, part in other_three.groupby('rxn_class'))) or '无'
    report = f"# 从共享桥接到具体化学选项：可替代、不可缺少与多级中继\n\n日期：2026-10-17。Round 28 已证明，分别适用于三个伙伴的组分替换顺序有时无法合并，而额外的已测条件可恢复六目标共享路线。本轮进一步问：恢复具体需要哪一种试剂及剂量？它只是一个可用选择，还是该已测条件范围内不可缺少的选择？一个选项不够时，最少需要多少种额外选择？\n\n## 新发现\n\n**共享适用范围的恢复具有不同的化学自由度：有时活化剂和添加剂可以互相提供替代路线，有时一个指定活化剂不可缺少，有时必须经过两个催化剂选择和另一种碱，才能把同一替换带过三个伙伴。** 这一区分来自固定端点、固定六目标已测支持上的全部候选检验。\n\n{len(cert)} 个强协调证书的最少额外槽位选项数已全部求得：{int(minimum_counts.get(1, 0))} 个需要 1 种，{int(minimum_counts.get(2, 0))} 个需要 2 种，{int(minimum_counts.get(3, 0))} 个需要 3 种。完整端点角点层分别为 **{int(complete_counts.get(1, 0))}、{int(complete_counts.get(2, 0))}、{int(complete_counts.get(3, 0))}**，来自 {complete.group_id.nunique()} 个三伙伴组合和 {complete.molecular_pair_id.nunique()} 个精确 A/B 分子对；另 {len(incomplete)} 个不完整角点证书的最少选项分布为 1 种 {int(incomplete_counts.get(1, 0))} 个、2 种 {int(incomplete_counts.get(2, 0))} 个、3 种 {int(incomplete_counts.get(3, 0))} 个，来自 {incomplete.group_id.nunique()} 个三伙伴组合和 {incomplete.molecular_pair_id.nunique()} 个精确分子对。不同层和最少数分组会复用组合与分子对，不能相加为独立实验数。\n\n| 端点角点层 | 最少 1 选项 | 最少 2 选项 | 最少 3 选项 | 全部强证书 |\n|---|---:|---:|---:|---:|\n| 完整 | {int(complete_counts.get(1, 0))} | {int(complete_counts.get(2, 0))} | {int(complete_counts.get(3, 0))} | {len(complete)} |\n| 不完整 | {int(incomplete_counts.get(1, 0))} | {int(incomplete_counts.get(2, 0))} | {int(incomplete_counts.get(3, 0))} | {len(incomplete)} |\n\n这里的选项是**槽位 × 身份 × 剂量**，不是同一瓶内同时加入多少试剂。三个额外选项可以包含同一催化剂槽位的两个不同取值。路线箭头表示一次只替换一个条件槽位，每个节点各自是一项已测反应条件。六目标产率是在每个条件下取最低值，再在全部路线中求最优瓶颈。\n\n## 可替代的活化剂与添加剂：group {int(witness('alternatives').group_id)}\n\nA/B 为 `C#Cc1cccc(N)c1` 与 `CSc1cccc(N)c1`，三种酸伙伴为 `COc1ccc(CC(=O)O)cc1`、`Cc1cc(C(=O)O)cc(Cl)n1`、`O=C(O)COc1ccccc1`；来源 acquisition 见 `chemical_witness_acquisition_contexts.csv`。端点为 DCC/Cl-HOBT 和 DIC/ethyl cyanoglyoxylate-2-oxime；四角全部实测，DMF、25 C、2 h，活化剂和两端添加剂均 1.5 eq，其余槽位固定。端点水平六目标最佳共同瓶颈是 {percent(witness('alternatives').K_endpoint_all)}，各伙伴 A/B bundle 分别仍有高产路径。\n\n只允许一个额外选项时，存在 {int(witness('alternatives').n_single_sufficient)} 种彼此可替代的解；其中三条化学路线如下：\n\n| 额外槽位选项 | 最佳六目标瓶颈 | 共同路线 |\n|---|---:|---|\n| HOAT，添加剂槽，1.5 eq | {percent(hoat.K_single)} | DCC/Cl-HOBT → DCC/HOAT → DIC/HOAT → DIC/肟添加剂 |\n| EDC-HCl，活化剂槽，1.5 eq | {percent(edc.K_single)} | DCC/Cl-HOBT → EDC-HCl/Cl-HOBT → EDC-HCl/肟添加剂 → DIC/肟添加剂 |\n| HOBt，添加剂槽，1.5 eq | {percent(hobt.K_single)} | DCC/Cl-HOBT → DCC/HOBt → DIC/HOBt → DIC/肟添加剂 |\n\nHOAT 单选项路线 {len(witness_route('alternatives', 'single_option', '487f9246f61897c6'))} 个节点的六目标最低产率依次为 {sequence('alternatives', 'single_option', '487f9246f61897c6')}；EDC-HCl 路线则为 {sequence('alternatives', 'single_option', 'bed09ab4473a6ee4')}。删除整个共同支持中的 HOAT 1.5 eq 条件后，完整图最佳瓶颈仍为 {percent(hoat.K_after_full_deletion)}。因此 HOAT 1.5 eq 是一个高产选择，但不是达到 20% 的必要选择。HOAT 1 eq 和 3 eq 是另外两个剂量 token；它们在单选项检验中均不能恢复本例。原生伙伴对支持不相同，本例的结论属于固定六目标共同支持。\n\n## 不可缺少的活化剂：group {int(witness('necessary').group_id)}\n\nA 为 `O=C(O)c1ccc(C(F)(F)F)cn1`，B 为 `O=C(O)c1ccc(F)cn1`；三伙伴为 `CNOC`、`COc1ccc(N)cc1`、`Cc1ccc(Oc2ccc(N)cc2)cc1`。三对原生伙伴支持均等于同一 {nodes[nodes.certificate_id == witness_ids['necessary']].recipe_id.nunique()} 条件共同支持。\n\nTPTU/无添加剂到 HBTU/HOBt 的端点水平最佳瓶颈为 {percent(witness('necessary').K_endpoint_all)}。在全部 {int(witness('necessary').n_extra_options)} 个额外槽位选项中，只有 PyBOP 1.5 eq 单独足够：\n\n`TPTU/无添加剂 → PyBOP/无添加剂 → PyBOP/HOBt → HBTU/HOBt`。\n\n路线 {len(witness_route('necessary', 'single_option', '095d36086b9b2e13'))} 个节点的六目标最低产率依次为 {sequence('necessary', 'single_option', '095d36086b9b2e13')}。删除全部 PyBOP 1.5 eq 条件，即使允许其余完整已测图，最佳瓶颈仍为 {percent(pybop.K_after_full_deletion)}。所以这个 token 在本例的已测图中既单独足够，也不可缺少。固定 DIPEA 3 eq、DMF、25 C、2 h，端点活化剂/HOBt 为 1.5 eq。\n\n## 双选项必须配合：group {int(witness('two_option').group_id)}\n\nA/B 是对位与间位甲氧基苯硼酸：`COc1ccc(B(O)O)cc1`、`COc1cccc(B(O)O)c1`。伙伴依次为 `Brc1cn2ccnc2cn1`、`COc1cccnc1Br`、`Fc1ccnc(Br)c1`。两端为 PdCl2(dtbpf)/Cs2CO3 与 PdCl2(PPh3)2/K2CO3，催化剂 0.05 eq、碱 3 eq，固定 1,4-dioxane/H2O (4:1)、100 C、12 h。\n\n端点四角完整，六目标共同瓶颈为 {percent(witness('two_option').K_endpoint_all)}，各伙伴 bundle 仍分别可行。{int(witness('two_option').n_extra_options)} 个单选项全部失败；{len(pair[pair.certificate_id == witness_ids['two_option']]):,} 个双选项中 {int(witness('two_option').n_sufficient_pairs)} 个成功，展示的成功组合为 **Pd(dppe)Cl2 0.05 eq + K3PO4 3 eq**。其路线为：\n\n`PdCl2(dtbpf)/Cs2CO3 → PdCl2(dtbpf)/K3PO4 → Pd(dppe)Cl2/K3PO4 → Pd(dppe)Cl2/K2CO3 → PdCl2(PPh3)2/K2CO3`。\n\n节点六目标最低产率依次为 {sequence('two_option', 'two_option', 'd4be0010fc039e7b')}。这精确给出最少两种额外选择。完整图可再经 PdCl2(PCy3)2 绕行，将最佳瓶颈提高到 {percent(witness('two_option').K_all)}。分别删除 Pd(dppe)Cl2 和 K3PO4 后，完整图最佳瓶颈为 {route_deletions('two_option', 'two_option', 'd4be0010fc039e7b')}。两种选项在此完整图中均必要。原生伙伴支持不相同。\n\n## 三选项接力：group {int(witness('three_option').group_id)}\n\nA/B 为间位与邻位甲氧基苯硼酸：`COc1cccc(B(O)O)c1`、`COc1ccccc1B(O)O`；三伙伴为 `COc1cc(C)c(Br)cn1`、`Cc1ccc(Br)nn1`、`Cc1ccnc(Br)c1`。两端 PCy3 Pd G3/K3PO4 与 Pd(DPEPhos)Cl2/Cs2CO3 的四角完整，端点共同瓶颈为 {percent(witness('three_option').K_endpoint_all)}。\n\n{int(witness('three_option').n_extra_options)} 个单选项、{len(pair[pair.certificate_id == witness_ids['three_option']]):,} 个双选项全部失败；{len(triple[triple.certificate_id == witness_ids['three_option']]):,} 个三选项中 {int(witness('three_option').n_sufficient_triples)} 组成功，展示组合为 **PdCl2(dtbpf) 0.05 eq、Pd(dppe)Cl2 0.05 eq、K2CO3 3 eq**。共同路线为：\n\n`PCy3 Pd G3/K3PO4 → PdCl2(dtbpf)/K3PO4 → PdCl2(dtbpf)/K2CO3 → Pd(dppe)Cl2/K2CO3 → Pd(dppe)Cl2/Cs2CO3 → Pd(DPEPhos)Cl2/Cs2CO3`。\n\n路线 {len(witness_route('three_option', 'three_option', 'd4be0010fc039e7b'))} 个节点的最低产率依次为 {sequence('three_option', 'three_option', 'd4be0010fc039e7b')}。逐一删除上述 token 后，完整图最佳瓶颈分别为 {route_deletions('three_option', 'three_option', 'd4be0010fc039e7b')}。其余槽位固定；溶剂、温度、时间及催化剂/碱用量同上一 Suzuki 例。这里的三选项是两个催化剂水平与一个碱水平的顺序接力。同一 A/B 关系共有 {related_three.group_id.nunique()} 个三伙伴组合具有最低三选项事实；其余最少三选项证书的反应类为 {other_three_classes}。\n\n## 唯一单选项仍可被复杂路线替代\n\n共有 {len(unique)} 个证书只有一个单选项解，但删除该选项后完整图仍可达到 20%。代表例 group {int(witness('substitutable').group_id)}、端点 {int(witness('substitutable').node0)}→{int(witness('substitutable').node1)} 的唯一单选项解为五氟苯酚 1.5 eq；A/B 为 `COc1ccc(N)cc1` 与 `COc1ccc(N)cn1`，三种酸伙伴为 `O=C(O)c1cc(Cl)cc(Cl)n1`、`O=C(O)c1ccc(C(F)(F)F)cn1`、`O=C(O)c1ccc(F)cn1`。两端为固定 EDC-HCl 1.5 eq 的 NHPI/DIPEA 与 HOBt/TEA；添加剂为 1.5 eq，碱为 3 eq，DMF、25 C、2 h。\n\n五氟苯酚单选项达到 {percent(substitute_single.K_single)}。删除它后，下列已测路线经 DCC 1.5 eq、NHS 1.5 eq、NMM 3 eq 三种额外选择仍可达到 {percent(substitute_single.K_after_full_deletion)}：\n\n`EDC-HCl/NHPI/DIPEA → DCC/NHPI/DIPEA → DCC/NHS/DIPEA → DCC/NHS/NMM → EDC-HCl/NHS/NMM → EDC-HCl/HOBt/NMM → EDC-HCl/HOBt/TEA`。\n\n这条展示路线证明三种额外选择可替代五氟苯酚，未额外穷举删除五氟苯酚后的最少选择数。因此不能把“唯一 singleton”直接改写为“必要试剂”。这 {len(unique)} 个完整记录见 `unique_singleton_but_substitutable.csv`；代表例的两条路线、实际六目标产率与来源均单列。\n\n## 跨分子关系的实际选项范围\n\n以下列的是相同身份、相同剂量、相同槽位在本轮固定证书上成功的描述性范围。每个支持按原 {n_opportunities:,} 机会注册保留，不把不同端点或三伙伴组合混成一次独立验证，也不把这些结果解释为新分子上的预测。\n\n| 选项 | eq | 足够的端点证书 | 三伙伴组合 | 精确 A/B 对 | 已检验强证书 |\n|---|---:|---:|---:|---:|---:|\n{top_table}\n\nPdCl2(dtbpf) 0.05 eq 的单选项范围覆盖 {scope_pairs('PdCl2(dtbpf)', 0.05)} 个精确分子对，XPhos Pd G3 与 PCy3 Pd G3 分别覆盖 {scope_pairs('XPhos Pd G3', 0.05)}、{scope_pairs('PCy3 Pd G3', 0.05)} 个；Acylation 中五氟苯酚 1.5 eq 覆盖 {scope_pairs('pentafluorophenol', 1.5)} 对，HOAT 1.5 eq 与 NHS 1.5 eq 分别覆盖 {scope_pairs('HOAT', 1.5)}、{scope_pairs('NHS', 1.5)} 对。这里的“已检验强证书”仅是该 token 不属于端点值、且实际在该共同支持中出现的候选证书数；完整机会上的“token被测得”列另见 `named_option_descriptive_scope.csv`。它与本轮成功率分母不是同一个集合。\n\n{nmultiple} 个证书至少有两种不同的单选项解；{len(singleton)} 个只有一种单选项解，其中 {int((singleton.n_necessary_options_in_full_graph > 0).sum())} 个的该 token 在完整图内必要、{len(unique)} 个可由多选项替代。总计 {int((cert.n_necessary_options_in_full_graph > 0).sum())} 个证书至少有一个必要 token，共 {len(need)} 个必要 token×证书状态。两选项最低的 {len(minimum_two)} 个证书中，{int((minimum_two.n_necessary_options_in_full_graph == 2).sum())} 个有两个必要 token，{int((minimum_two.n_necessary_options_in_full_graph == 1).sum())} 个只有一个必要 token；因而最少选择数与必要选择数并不相同。\n\n所有 {len(single):,} 个单选项候选保留，其中 {int((~single.single_sufficient20).sum()):,} 个失败。另保留全部 {len(pair):,} 个双选项和 {len(triple):,} 个三选项（只在前一级全部失败的证书上展开）。单选项成功中 {len(fixed)} 个 token×证书状态游走的是端点未变槽位，来自 {fixed.certificate_id.nunique()} 个证书；其余在端点变化槽位。二者均按实际槽位区分保存。\n\n## 分母、算法与来源\n\n本轮条件层沿用 R28 的 **{int(run_summary['R28_complete_corner_bundle_feasible_endpoint_denominator20']):,} 个完整角点且每个伙伴 A/B bundle 单独可行的 20% 端点视图**；{len(complete)} 个完整角点强协调证书是该条件分母中的恢复项。{len(incomplete)} 个不完整角点证书单列。全部 {n_opportunities:,} 个相容三伙伴机会仍保留，其中无共同条件、无高端点及其他反应类的机会没有删除。`complete_class_option_summary.csv` 含全部 {n_datasets} 数据集/{n_classes} 反应类的零结果，`complete_opportunity_registry.csv.gz` 对应完整机会注册。\n\n完整图只含六个目标均测得的条件。边严格为一个完整槽位 token 不同。基线每槽只允许两端取值，未变槽位固定；一个额外 token 的充分性检验只放开该槽位这一身份/剂量，其他槽仍取端点值。必要性则从完整已测图删除该槽位 token 的所有节点后优化。所有容量由 heap 最大瓶颈算法求全部路径，不限制最短路径或最少步数。双选项、三选项穷举包括同槽不同 token，故最少数是不同新增选择数的精确值。\n\n独立核验状态：`{verification['status']}`。独立核验器另从 R27 表重建全部候选、支持、六目标产率，以 BFS 检查 20% 可行性，以降序节点激活/union-find 求连续最优瓶颈。核验覆盖全部 {verification['n_single_options']:,} 个单选项、{verification['n_single_deletions']:,} 个完整图删除、{verification['n_pair_options']:,} 个双选项、{verification['n_triple_options']:,} 个三选项及每证书的完整图和基线，共 **{verification['n_continuous_states_checked']:,} 个连续容量及 {verification['n_threshold_states_checked']:,} 个阈值状态**；检查 {verification['n_saved_paths_checked']:,} 条保存路径、全部候选组合穷举资格、精确最少数及全部 {verification['n_dataset_corner_summary_rows_checked']} 个数据集×角点层汇总。最大容量/路径差为 {verification['maximum_absolute_capacity_or_path_error']:.17g}。详见 `independent_verification.json`。\n\n全部 {len(fresh_witnesses)} 例均在本轮逐条重新回读原始工作簿，状态为 `{raw_verification['status']}`。共去重回读 {raw_verification['n_deduplicated_raw_reaction_records']:,} 条原始反应和 {raw_verification['n_deduplicated_raw_conditions']:,} 条条件，保留 {raw_verification['n_deduplicated_exact_zero_raw_records']:,} 条真实零值，重建全部 {raw_verification['n_target_recipe_values_reconstructed']:,} 个目标×配方值，最大差为 {max((row['maximum_yield_error'] for row in raw_witnesses.values())):.17g}。条件检查覆盖全部试剂身份、剂量、标准名称及固定溶剂/温度/时间。`chemical_witness_raw_verification.json` 记录实际五个证书的逐例计数、原始工作簿 SHA256 和本轮回读证明；没有复用早期回读回执，逐行回读表与重建值另存。完整逐步条件和六目标产率见 `chemical_witness_routes_with_six_yields_and_sources.csv`。源表和原始文件均未修改。\n\n这组结果给出的合成认识是：选择一个高产展示路线还没有回答如何扩大共同适用范围。实际可用的自由度可以属于另一种添加剂、另一种活化剂，或者数个催化剂/碱水平的接力；它们分别对应替代路线、不可缺少的选择和多选项协调。\n"
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
