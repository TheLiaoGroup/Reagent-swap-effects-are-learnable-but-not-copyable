"""Shared versus family-specific versus support-limited latent dimensions.

The 18 screen matrices do not share a response object or a cross-reaction
condition coordinate system.  This analysis therefore never concatenates raw
condition columns across reaction classes.  Each screen is decomposed on its
own interaction-centred response matrix; the normalized singular spectrum is
then treated as the comparable object.  A pooled spectrum describes a shared
rank profile, repeated families provide the only identifiable family effect,
and complete-row/support sensitivity is reported separately."""
from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dimension_core import centred
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
MAX_PROFILE_RANK = 32
PROFILE_TOL = 1e-12

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def family_of(screen: str) -> str:
    return str(screen).split('|', 1)[0]

def spectrum_profile(matrix: np.ndarray, max_rank: int=MAX_PROFILE_RANK) -> tuple[np.ndarray, dict[str, float]]:
    """Return normalized interaction spectrum and stable descriptive metrics."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or not x.size:
        raise ValueError('matrix must be a non-empty 2D array')
    r = centred(x, view='interaction')
    singular = np.linalg.svd(r, compute_uv=False, full_matrices=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= PROFILE_TOL:
        profile = np.zeros(max_rank, dtype=float)
        metrics = {'k_var90': 0, 'k_var95': 0, 'participation_rank': 0.0, 'ev6': 0.0, 'ev12': 0.0, 'energy': total, 'observed_rank': 0}
        return (profile, metrics)
    p = energy / total
    cumulative = np.cumsum(p)
    profile = np.zeros(max_rank, dtype=float)
    profile[:min(max_rank, len(p))] = p[:max_rank]
    metrics = {'k_var90': int(np.searchsorted(cumulative, 0.9) + 1), 'k_var95': int(np.searchsorted(cumulative, 0.95) + 1), 'participation_rank': float(1.0 / np.sum(p * p)), 'ev6': float(cumulative[min(5, len(cumulative) - 1)]), 'ev12': float(cumulative[min(11, len(cumulative) - 1)]), 'energy': total, 'observed_rank': int(np.sum(p > PROFILE_TOL))}
    return (profile, metrics)

def support_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Quantify missingness and zero support without conflating the two."""
    x = np.asarray(matrix, dtype=float)
    finite = np.isfinite(x)
    n_rows, n_cols = x.shape
    observed = x[finite]
    complete = finite.all(axis=1)
    productive = np.nanmax(np.where(finite, x, np.nan), axis=1) >= 0.2
    return {'n_pairs': int(n_rows), 'n_conditions': int(n_cols), 'finite_cells': int(finite.sum()), 'support_fraction': float(finite.mean()), 'missing_fraction': float((~finite).mean()), 'zero_fraction_observed': float(np.mean(observed == 0.0)) if observed.size else np.nan, 'all_zero_pair_fraction': float(np.mean(np.nanmax(np.where(finite, x, np.nan), axis=1) == 0.0)), 'complete_row_fraction': float(complete.mean()), 'productive_row_fraction': float(productive.mean()), 'n_complete_rows': int(complete.sum()), 'n_productive_rows': int(productive.sum())}

def _profile_frame(screen: str, family: str, profile: np.ndarray, label: str) -> pd.DataFrame:
    cumulative = np.cumsum(profile)
    return pd.DataFrame({'screen': screen, 'family': family, 'profile': label, 'rank': np.arange(1, len(profile) + 1), 'variance_fraction': profile, 'cumulative_fraction': cumulative})

def build_screen_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    records: list[dict[str, object]] = []
    spectra: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    for path in sorted(DATA.glob('matrix_*.npz')):
        input_hashes[str(path.relative_to(HERE))] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            matrix = np.asarray(saved['M'], dtype=float)
        screen = path.name[len('matrix_'):-len('.npz')].replace('_', '|', 1)
        inventory = TABLES / 'screen_inventory.csv'
        if inventory.exists():
            keys = pd.read_csv(inventory, usecols=['screen'], keep_default_na=False)['screen'].tolist()
            encoded = path.stem[len('matrix_'):]
            matches = [k for k in keys if k.replace('|', '_').replace(' ', '_') == encoded]
            if len(matches) == 1:
                screen = matches[0]
        family = family_of(screen)
        profile, metrics = spectrum_profile(matrix)
        records.append({'screen': screen, 'family': family, **metrics})
        support = support_metrics(matrix)
        support_rows.append({'screen': screen, 'family': family, **support})
        spectra.append(_profile_frame(screen, family, profile, 'observed_all'))
        finite_rows = np.isfinite(matrix).all(axis=1)
        if finite_rows.any():
            complete_profile, complete_metrics = spectrum_profile(matrix[finite_rows])
            records[-1].update({f'complete_{k}': v for k, v in complete_metrics.items()})
            spectra.append(_profile_frame(screen, family, complete_profile, 'complete_rows'))
        else:
            records[-1].update({'complete_k_var90': np.nan, 'complete_k_var95': np.nan, 'complete_participation_rank': np.nan, 'complete_ev6': np.nan, 'complete_ev12': np.nan, 'complete_energy': np.nan, 'complete_observed_rank': 0})
        productive = np.nanmax(np.where(np.isfinite(matrix), matrix, np.nan), axis=1) >= 0.2
        if productive.any():
            productive_profile, productive_metrics = spectrum_profile(matrix[productive])
            records[-1].update({f'productive_{k}': v for k, v in productive_metrics.items()})
            spectra.append(_profile_frame(screen, family, productive_profile, 'productive_rows'))
        else:
            records[-1].update({'productive_k_var90': np.nan, 'productive_k_var95': np.nan, 'productive_participation_rank': np.nan, 'productive_ev6': np.nan, 'productive_ev12': np.nan, 'productive_energy': np.nan, 'productive_observed_rank': 0})
        arrays[screen] = matrix
        profiles[screen] = profile
    screens = sorted(profiles)
    if len(screens) != 19:
        raise AssertionError(f'expected 19 screen matrices, found {len(screens)}')
    screen_df = pd.DataFrame(records).sort_values('screen').reset_index(drop=True)
    support_df = pd.DataFrame(support_rows).sort_values('screen').reset_index(drop=True)
    return (screen_df, pd.concat(spectra, ignore_index=True), support_df, input_hashes)

def build_pooled_tables(screen_df: pd.DataFrame, spectrum_df: pd.DataFrame, support_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    all_profile = spectrum_df[spectrum_df.profile.eq('observed_all')]
    pivot = all_profile.pivot(index='screen', columns='rank', values='variance_fraction').sort_index()
    p = pivot.to_numpy(dtype=float)
    screens = pivot.index.tolist()
    global_mean = p.mean(axis=0)
    centered = p - global_mean
    pooled_s = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    pooled_e = pooled_s * pooled_s
    pooled_cum = np.cumsum(pooled_e) / pooled_e.sum() if pooled_e.sum() > 0 else np.zeros_like(pooled_e)
    pooled_rank90 = int(np.searchsorted(pooled_cum, 0.9) + 1) if pooled_e.sum() > 0 else 0
    pooled_rows = pd.DataFrame({'rank': np.arange(1, len(global_mean) + 1), 'global_mean_variance_fraction': global_mean, 'global_mean_cumulative_fraction': np.cumsum(global_mean), 'pooled_profile_singular_value': np.pad(pooled_s, (0, len(global_mean) - len(pooled_s))), 'pooled_profile_component_fraction': np.pad(pooled_e / pooled_e.sum() if pooled_e.sum() > 0 else pooled_e, (0, len(global_mean) - len(pooled_e))), 'pooled_profile_component_cumulative': np.pad(pooled_cum, (0, len(global_mean) - len(pooled_cum)))})
    loo_rows: list[dict[str, object]] = []
    family_map = screen_df.set_index('screen')['family'].to_dict()
    repeated = {f for f, n in screen_df.family.value_counts().items() if n >= 2}
    for i, screen in enumerate(screens):
        others = np.delete(p, i, axis=0)
        pred = others.mean(axis=0)
        actual = p[i]
        global_rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        global_l1 = float(np.mean(np.abs(actual - pred)))
        global_corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(actual) > 0 and np.std(pred) > 0 else np.nan
        fam = family_map[screen]
        fam_others = p[[j for j, s in enumerate(screens) if j != i and family_map[s] == fam]]
        fam_supported = fam in repeated and len(fam_others) > 0
        fam_pred = fam_others.mean(axis=0) if fam_supported else np.full(p.shape[1], np.nan)
        fam_rmse = float(np.sqrt(np.mean((actual - fam_pred) ** 2))) if fam_supported else np.nan
        loo_rows.append({'screen': screen, 'family': fam, 'shared_profile_rmse': global_rmse, 'shared_profile_mae': global_l1, 'shared_profile_correlation': global_corr, 'family_reconstruction_identifiable': bool(fam_supported), 'family_profile_rmse': fam_rmse, 'family_profile_gain_rmse': global_rmse - fam_rmse if fam_supported else np.nan, 'family_n_other_screens': int(len(fam_others)) if fam_supported else 0, 'family_effect_identifiability': 'repeated_family' if fam_supported else 'singleton_family'})
    loo_df = pd.DataFrame(loo_rows).sort_values('screen').reset_index(drop=True)
    repeated_screens = [s for s in screens if family_map[s] in repeated]
    rows: list[dict[str, object]] = []
    if repeated_screens:
        idx = [screens.index(s) for s in repeated_screens]
        pr = p[idx]
        fam_labels = [family_map[s] for s in repeated_screens]
        for rank_idx in range(p.shape[1]):
            vals = pr[:, rank_idx]
            grand = float(vals.mean())
            between = 0.0
            within = 0.0
            n_total = len(vals)
            for fam in sorted(set(fam_labels)):
                local = vals[np.array(fam_labels) == fam]
                fm = float(local.mean())
                between += len(local) * (fm - grand) ** 2
                within += float(np.sum((local - fm) ** 2))
            total = between + within
            rows.append({'rank': rank_idx + 1, 'repeated_family_screens': n_total, 'repeated_family_count': len(set(fam_labels)), 'between_family_ss': between, 'within_family_ss': within, 'total_ss': total, 'between_fraction': between / total if total > 0 else np.nan, 'within_fraction': within / total if total > 0 else np.nan})
    variance_df = pd.DataFrame(rows)
    summary = {'n_screens': len(screens), 'n_families': int(screen_df.family.nunique()), 'repeated_families': sorted(repeated), 'repeated_family_screens': len(repeated_screens), 'profile_rank': int(len(global_mean)), 'pooled_profile_rank90': pooled_rank90, 'global_mean_k_var90': int(np.searchsorted(np.cumsum(global_mean), 0.9) + 1), 'global_mean_k_var95': int(np.searchsorted(np.cumsum(global_mean), 0.95) + 1), 'mean_shared_profile_loo_rmse': float(loo_df.shared_profile_rmse.mean()), 'median_shared_profile_loo_rmse': float(loo_df.shared_profile_rmse.median()), 'n_family_reconstructions_identifiable': int(loo_df.family_reconstruction_identifiable.sum())}
    return (pooled_rows, loo_df, {'summary': summary, 'variance': variance_df})

def plot_figure(screen_df: pd.DataFrame, spectrum_df: pd.DataFrame, loo_df: pd.DataFrame, variance_df: pd.DataFrame, support_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    observed = spectrum_df[spectrum_df.profile.eq('observed_all')]
    for _, group in observed.groupby('screen', sort=True):
        axes[0, 0].plot(group['rank'], group['cumulative_fraction'], color='#9aa0a6', alpha=0.55, lw=1)
    global_curve = observed.groupby('rank', sort=True).variance_fraction.mean().cumsum()
    axes[0, 0].plot(global_curve.index, global_curve.values, color='#1f4e79', lw=2.5, label='pooled mean')
    axes[0, 0].axhline(0.9, color='#d1495b', ls='--', lw=1)
    axes[0, 0].set(title='Independent screen spectra and pooled profile', xlabel='abstract latent rank', ylabel='cumulative response energy')
    axes[0, 0].legend(frameon=False)
    s = screen_df.merge(support_df[['screen', 'support_fraction', 'zero_fraction_observed']], on='screen', validate='one_to_one').sort_values('participation_rank')
    axes[0, 1].scatter(s.participation_rank, s.support_fraction, c=s.zero_fraction_observed, cmap='viridis', s=55, edgecolor='white', linewidth=0.4)
    for _, row in s.iterrows():
        axes[0, 1].annotate(row.screen.split('|')[0][:12], (row.participation_rank, row.support_fraction), fontsize=7, xytext=(3, 3), textcoords='offset points')
    axes[0, 1].set(title='Effective rank versus observed support', xlabel='participation rank', ylabel='finite-cell fraction')
    axes[1, 0].hist(loo_df.shared_profile_rmse, bins=7, color='#2a9d8f', edgecolor='white')
    axes[1, 0].set(title='Leave-one-screen-out shared-profile error', xlabel='spectrum RMSE', ylabel='screens')
    if not variance_df.empty:
        axes[1, 1].plot(variance_df['rank'], variance_df.between_fraction, color='#d1495b', lw=2, label='between repeated families')
        axes[1, 1].plot(variance_df['rank'], variance_df.within_fraction, color='#457b9d', lw=2, label='within repeated families')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].set(title='Repeated-family variance decomposition', xlabel='abstract latent rank', ylabel='fraction of profile variance')
    path = FIGURES / '29_shared_latent_dimension.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def render_report(screen_df: pd.DataFrame, support_df: pd.DataFrame, pooled_df: pd.DataFrame, loo_df: pd.DataFrame, variance_df: pd.DataFrame, summary: dict[str, object]) -> str:
    top = screen_df.sort_values('participation_rank', ascending=False).head(5)
    lines = ['# 39_SHARED_LATENT_DIMENSION_2026-10-17', '', '## 研究问题', '', '本轮检验 18 个 screen matrix 是否表现出可共享的潜在维度轮廓，同时把 family-specific deviation 与支持/缺测造成的维度变化分开。由于原始登记明确禁止跨 reaction_type 连接 condition_id，本轮不把不同反应类的条件列拼接成一个共同化学坐标，也不跨 response object 数值相减。共享只定义为归一化 interaction-centered singular spectrum 的稳定形状。', '', '## 设计与识别边界', '', '每个屏幕在自己的底物对×条件矩阵内完成缺测保留的 interaction centering 和 SVD，谱能量归一化后截取前 32 个抽象 rank 位置。pooled mean profile 与 leave-one-screen-out 误差用于描述跨屏幕共享；重复 family 才能估计 family-specific effect。当前只有 Buchwald-Hartwig(CMC) 的 3 个 screen 和 Buchwald-Hartwig 的 2 个 screen 可提供 family 内重复，singleton family 的 family effect 与 screen effect 不可分辨。', '', '## A. 共享谱形', '', f"18 个屏幕、{summary['n_families']} 个 reaction family 的 pooled mean spectrum 在前 {summary['global_mean_k_var90']} 个抽象 rank 达到 90% 能量，在前 {summary['global_mean_k_var95']} 个 rank 达到 95%。pooled profile 自身的 90% 变化由前 {summary['pooled_profile_rank90']} 个 profile component 描述。leave-one-screen-out 的平均 spectrum RMSE 为 {summary['mean_shared_profile_loo_rmse']:.4f}，中位数为 {summary['median_shared_profile_loo_rmse']:.4f}；该误差是 rank-profile 重构误差，不是 yield 预测误差。", '', '## B. family-specific 维度', '', f"重复 family 子集包含 {summary['repeated_family_screens']} 个 screen。按抽象 rank 的 profile variance 分解，between-family 与 within-family 项分别保留在表中；它们只表示谱形差异的登记层级来源，不能推断化学机理。{summary['n_family_reconstructions_identifiable']} 个 screen 的 leave-one-out family profile 可识别，其余 singleton family 保持不可识别。", '', '## C. 支持/缺测维度', '', '缺测被记录为 missing_fraction，数值零被单独记录为 zero_fraction_observed。complete-row SVD 作为支持敏感性分析，报告 all-row 与 complete-row 的 k_var90 和 participation rank 差异；complete rows 不是无偏总体替代，尤其在完整行比例较低的屏幕。', '', '| screen | family | support | missing | zero observed | participation rank | k90 | complete-row fraction |', '|---|---|---:|---:|---:|---:|---:|---:|']
    joined = screen_df.merge(support_df[['screen', 'support_fraction', 'missing_fraction', 'zero_fraction_observed', 'complete_row_fraction']], on='screen', suffixes=('', '_support'))
    for _, row in joined.sort_values('screen').iterrows():
        display_screen = str(row.screen).replace('|', '\\|')
        lines.append(f'| {display_screen} | {row.family} | {row.support_fraction:.3f} | {row.missing_fraction:.3f} | {row.zero_fraction_observed:.3f} | {row.participation_rank:.2f} | {int(row.k_var90)} | {row.complete_row_fraction:.3f} |')
    lines += ['', '## D. 高维度屏幕与下一步可证伪方向', '', 'interaction spectrum 的复杂度跨屏幕差异很大。当前最高 participation-rank 屏幕为 ' + ', '.join((f'{r.screen} ({r.participation_rank:.1f})' for _, r in top.iterrows())) + '。这说明一个统一的固定 rank 不能代表全部 family；更可行的下一步是预先登记一个跨屏幕谱形基线，再以新增 family 内重复和共同 response object 的实验检验 shared component 是否能预测新屏幕的残差维度。', '', '## 结论边界', '', '本报告是已有数据的谱形级探索。它不声称不同反应类共享相同化学变量，不把 family-specific profile deviation 当作机制，也不把 support-limited dimension 当作真实潜在维度。新增实验若要把共享维度提升为可迁移科学命题，必须在同一 response object、明确共同条件语义和预注册支持阈值下复测。']
    return '\n'.join(lines) + '\n'

def main() -> None:
    started = time.time()
    screen_df, spectrum_df, support_df, input_hashes = build_screen_tables()
    pooled_df, loo_df, extras = build_pooled_tables(screen_df, spectrum_df, support_df)
    variance_df = extras['variance']
    summary = extras['summary']
    outputs = {'screen': TABLES / 'shared_latent_dimension_screen.csv', 'spectrum': TABLES / 'shared_latent_dimension_spectrum.csv', 'loo': TABLES / 'shared_latent_dimension_loo.csv', 'pooled': TABLES / 'shared_latent_dimension_pooled_profile.csv', 'variance': TABLES / 'shared_latent_dimension_variance_partition.csv', 'support': TABLES / 'shared_latent_dimension_support.csv', 'figure': FIGURES / '29_shared_latent_dimension.png', 'report': HERE / '39_SHARED_LATENT_DIMENSION_2026-10-17.md'}
    screen_df.to_csv(outputs['screen'], index=False)
    spectrum_df.to_csv(outputs['spectrum'], index=False)
    loo_df.to_csv(outputs['loo'], index=False)
    pooled_df.to_csv(outputs['pooled'], index=False)
    variance_df.to_csv(outputs['variance'], index=False)
    support_df.to_csv(outputs['support'], index=False)
    figure = plot_figure(screen_df, spectrum_df, loo_df, variance_df, support_df)
    report = render_report(screen_df, support_df, pooled_df, loo_df, variance_df, summary)
    outputs['report'].write_text(report, encoding='utf-8')
    manifest = {'date': '2026-10-17', 'status': 'completed_exploratory_spectrum_level_analysis', 'method': 'per-screen interaction-centered SVD; normalized singular-spectrum pooled profile; leave-one-screen-out profile reconstruction', 'response_object_rule': 'No raw condition columns are joined across reaction types; no cross-response-object numerical comparison', 'family_effect_rule': 'Family-specific decomposition restricted to repeated families; singleton family effect is not identifiable', 'support_rule': 'Missing cells remain missing in centering; complete-row spectrum is sensitivity only; zeros are counted separately', 'max_profile_rank': MAX_PROFILE_RANK, 'summary': summary, 'inputs_sha256': input_hashes, 'code_sha256': {'shared_latent_dimension.py': digest(Path(__file__)), 'dimension_core.py': digest(HERE / 'dimension_core.py'), 'test_shared_latent_dimension.py': digest(HERE / 'test_shared_latent_dimension.py')}, 'outputs_sha256': {str(path.relative_to(HERE)): digest(path) for path in outputs.values()}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'seconds': time.time() - started}
    (DATA / 'shared_latent_dimension_manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='utf-8')
    print('FINISHED_SHARED_LATENT_DIMENSION', json.dumps(summary, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
