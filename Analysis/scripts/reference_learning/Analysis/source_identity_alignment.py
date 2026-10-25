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
    lines = ['# 60 同成本校准下的来源身份对应与净复用', '', '日期：2026-10-17。看到报告57结果后新增的回顾性分析；不是预注册、独立确认或新增实验。', '', f'**固定预算24可评分的21个跨家族方向、182个设计后，affine ridge在预算192的精确测试对应优势为{10000.0 * top.delta_test:+.3f} yield pp²，但相对同成本target mean的净收益为{10000.0 * top.net_gain:+.3f} yield pp²。** 对应优势与净复用是不同命题；本轮将二者在同一目标、成本和测试支持上同时评分。', '', '## 简短设计记录与固定科学对象', '', '本轮完整保留报告57的78个已登记方向、strict/background_free/position2/min1/all的原目标响应算子、身份级留出、目标校准读数、10个hash设计和24/48/96/192坐标预算。问题是同成本下，真实来源身份对应是否比打乱对应提供更小的held-out误差，以及这种对应价值是否转化为相对target-mean控制的净收益。未改变响应端点、模型、alpha、采集顺序或根据测试y选择映射。', '', '主检验在每个设计/预算内冻结57已有a,b和全部校准读数，只在held-out测试身份之间排列来源x；次级检验在全来源身份上作128个固定唯一非identity映射，并按同一规则重新拟合。两种参照检验不同对象，不相减为机制贡献。先在每方向等权平均可评分设计MSE，再等方向平均；不将设计、映射或测试身份当成独立实验复制。', '', '跨预算主域为57预算24已可评分的21个cross-family方向、182个设计，后续预算使用完全相同的方向与测试身份。另保留每预算all-scored域及within-family域的全部结果。all-scored域的分母随预算改变，不能把其纵向变化直接称纯预算效应。', '', '## 主检验：冻结拟合后的全部TEST排列', '', '设原冻结预测为`a+b*x_i`，x/y为同一组n_test个测试身份的来源/目标响应。对全部n_test!身份排列（含原排列）等权平均，不枚举即可精确计算：', '', '`R_perm_exact = mean[(y-a-b*mean(x))²] + b²*Var_n(x)`', '', '`Delta_test = R_perm_exact-R_real = 2*b*Cov_n(x,y)`', '', '`G_net = R_target_calibration_mean-R_real`', '', '方差、协方差均用分母n_test；不存在随机化p值或以排列标准差当实验误差。真实/排列具有相同测试source边际、校准拟合、目标校准成本和目标算子，仅测试身份对应不同。所有使用测试y的协方差与误差都是retrospective oracle评分诊断，不能作为预先选择方向、模型或预算的指标。', '', '| 固定cross域预算 | target mean RMSE pp | real affine RMSE pp | exact TEST参照RMSE pp | Delta_test pp² | G_net pp² | Delta>0方向 | Gnet>0方向 | Delta>0且无净收益方向 |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in fixed.itertuples():
        lines.append(f'| {row.budget} | {row.target_mean_rmse_pp:.6f} | {row.real_rmse_pp:.6f} | {row.test_permuted_exact_rmse_pp:.6f} | {10000.0 * row.delta_test:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.positive_delta_test_directions}/{row.n_scored_directions} | {row.positive_net_directions}/{row.n_scored_directions} | {row.positive_delta_without_net_directions}/{row.n_scored_directions} |')
    lines += ['', '宏RMSE是先等设计、等方向平均MSE后开根号；Delta与Gnet是在同一MSE尺度的配对差，不是RMSE百分点差，也不是各方向相对风险比的均值。正Delta说明这条已拟合斜率在这些测试身份中比全部测试排列的平均更对准，并不保证比target mean更好。所有正、负和neutral方向保留。', '', 'offset的b固定为1，而固定域各设计的测试x/y跨预算不变，因此Delta_test=2*Cov_n(x,y)跨四预算完全不变。这个恒等结果不作为四次重复稳健性发现；预算只可改变offset截距和相应绝对误差及净收益。', '', '## 覆盖与完整方向账本', '', '| relation | budget | 可评分方向 | 可评分设计 | 登记方向 |', '|---|---:|---:|---:|---:|']
    for row in coverage.itertuples():
        lines.append(f'| {row.relation} | {row.budget} | {row.n_scored_directions} | {row.n_scored_designs} | {row.n_registered_directions} |')
    lines += ['', '以下保留两模型在其余全部评分域上的结果；fixed域是B24原可评分设计子集，all-scored是各预算原可评分域。方向数及设计数是配对分母，不是独立复制数。', '', '| domain | relation | budget | model | Delta_test pp² | G_net pp² | 方向/设计数 |', '|---|---|---:|---|---:|---:|---:|']
    other = primary[~(primary.domain.eq('b24_fixed') & primary.relation.eq('cross_family') & primary.model.eq('source_affine_ridge'))]
    for row in other.itertuples():
        lines.append(f'| {row.domain} | {row.relation} | {row.budget} | {row.model} | {10000.0 * row.delta_test:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.n_scored_directions}/{row.n_scored_designs} |')
    lines += ['', '其余geometry_absent、insufficient_identity_support、budget_insufficient及原状态完整保留，风险为NA，不将未知几何或未足额校准改成负运输。原54个有限方向、500个collapsed响应身份的来源/目标全部底层坐标和导入record并集再核对不相交，避免全来源重配引入已读目标标签。该计算不相交不证明不同身份、screen或family统计独立。', '', '## 次级参照：全部来源身份重配并重新校准', '', '仅对38个至少6身份的eligible方向生成128个唯一、非identity的身份排列。另16个有限但少于6身份的方向最多只有5!−1=119种非identity排列，及24个geometry缺失方向均保留mapping guard，不伪造128个唯一映射。映射作用于每方向全部source x，跨全部设计和预算冻结；所有目标校准标签、测试身份、目标算子及成本不变。', '', f"hash版本为`{VERSION}`。先按Python字符串序排序完整身份；candidate从0递增，以SHA256(JSON[version,source,target,candidate,identity])排序身份得到assigned-source索引向量，跳过identity及已接受映射，取首128个。JSON为UTF-8、ensure_ascii=False、sort_keys=True、separators=(',',':')。完整mapping/candidate/hash均保存；无需保存数千万单位预测，也能从原57表精确重建。", '', '每个置换保留来源全部x的分布和全来源均值/SD；真实与置换均使用相同alpha=3 standardized affine ridge、未惩罚截距及offset。代码使用与57 SVD Ridge等价的一维闭式`b=Sxy/(Sxx+3*SD_all²)`；全来源SD<=1e-12时affine回退target mean。真实系数及MSE逐项核对57。测试y从不进入拟合或映射选择，不调alpha。', '', '全部128 draw级分数保存；数值重复、预测重复、全零来源均不去重。mapping_registry另登记不同assigned-x向量数；128个身份排列不代表128个不同数值实验。次级参照同时改变校准x的覆盖/对应和测试对应，因此真实优于全域重配不能单独归因于测试对应。', '', '`Delta_pair = mean_128(R_reassigned_refit)-R_real`；`G_net=R_targetmean-R_real`。这两个量与主检验Delta_test分开报告，不相减成独立机制贡献。', '', '| 固定cross域预算 | model | real RMSE pp | reassigned/refit RMSE pp | Delta_pair pp² | G_net pp² | Delta_pair>0方向 |', '|---:|---|---:|---:|---:|---:|---:|']
    for row in secondary[secondary.domain.eq('b24_fixed') & secondary.relation.eq('cross_family')].sort_values(['model', 'budget']).itertuples():
        lines.append(f'| {row.budget} | {row.model} | {row.real_rmse_pp:.6f} | {row.mean_permuted_rmse_pp:.6f} | {10000.0 * row.delta_pair:+.6f} | {10000.0 * row.net_gain:+.6f} | {row.positive_delta_pair_directions}/{row.n_scored_directions} |')
    lines += ['', '## 退化、局部与范围边界', '', '| source | target | 身份数 | 全source为零 | 128映射的distinct assigned-x向量数 |', '|---|---|---:|---|---:|']
    for row in mapping_registry[mapping_registry.mapping_status.eq('mapped') & mapping_registry.all_source_constant].itertuples():
        source, target = (row.source_screen.replace('|', '\\|'), row.target_screen.replace('|', '\\|'))
        lines.append(f'| {source} | {target} | {row.n_identities} | {row.all_source_zero} | {row.n_distinct_assigned_source_vectors} |')
    lines += ['', 'SNAr→Heck的全零source没有可用身份差异，不能因为保留128个映射就产生新信息；反向Heck→SNAr的全零target风险亦按原资格和预算处理。所有direction表保留正Delta但Gnet非正、负Delta及neutral状态，不以平均正对应冒充实现净复用。', '', '这项检验仍属于完整问题中“学习、测量、化学响应与运输能否在相同对象和成本下配合”的对应检验。它只约束原一维affine/offset规则、有限历史身份及57固定采集策略，不是所有潜在模型的否定或证明，也不是跨库物理化学干预、实验因果效应、普适运输律或来源结构机制。来源和目标总体仍有各自伙伴/背景组成，未把它们变成共同底物或同一底层响应算子。', '', '所有方向与hash设计共享来源、目标和底物，不作独立样本p值或置信区间；128 draw的分布只是一组固定结构参照，不是实验随机化分布。新增数据、独立采集、额外模型或完全匹配总体可能有不同结论，本轮不据结果挑方向或重写响应对象。', '', '## 产物与复核', '', 'producer `source_identity_alignment.py`；focused tests `test_source_identity_alignment.py`；独立核验 `verify_source_identity_alignment.py`。同前缀mapping_registry、mappings、source_target_support_audit、design_scores、draw_scores、secondary_scores、direction_scores、headline、secondary_direction_scores、secondary_headline和coverage保存全部映射、状态、draw及层级分母。figure47及manifest记录输入、代码和输出SHA256；原始数据、57表、README和其他报告不改。', '']
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
