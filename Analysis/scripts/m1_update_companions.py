"""Reconstruct fixed E68 update steps from M1's saved E67 predictions.

The original E68 protocol fixed t = 0, .25, .5, .75, 1 before its path
coefficients were computed. This downstream arithmetic reproduces its primary
content paths; no model is fitted, and no
optimal step or nonlinear action threshold is selected."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
S = Path(__file__).resolve().parent
RESPONSE_LEARNING = S / 'response_learning'
OUT = S.parent / 'out'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
QUERY = ['family_request_id', 'allocation', 'vector_id']
COEFFICIENTS = ['R0', 'A', 'B']
GROUP = ['target_geometry', 'path', 'path_family', 'risk_component', 'allocation']
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PATHS = [('raw_delta', 'zero_local', 'level_only', 'primary_content'), ('raw_delta', 'zero_local', 'modulation_only', 'primary_content'), ('centered_unit_rms_zeroembed', 'zero_local', 'full', 'primary_unit')]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def centered_unit(raw):
    constant = np.all(raw == raw[:, :1], axis=1)
    mean = raw[:, 0] + (raw - raw[:, :1]).mean(axis=1)
    centered = raw - mean[:, None]
    centered[constant] = 0
    rms = np.sqrt(np.mean(centered ** 2, axis=1))
    unit = np.zeros_like(centered)
    np.divide(centered, rms[:, None], out=unit, where=~constant[:, None])
    return (unit, constant)

def components(error):
    mean = error[:, 0] + (error - error[:, :1]).mean(axis=1)
    return np.stack([np.mean(error ** 2, axis=1), mean ** 2, np.mean((error - mean[:, None]) ** 2, axis=1)], axis=1)

def aggregate(frame):
    edge_rows, class_rows, summaries = ([], [], [])
    for stratum in ('all', 'true_nonconstant', 'true_constant'):
        subset = frame if stratum == 'all' else frame[frame.true_constant.eq(stratum == 'true_constant')]
        edge = subset.groupby([*EDGE, *GROUP], observed=True, sort=True).agg(n_test_vectors=('vector_id', 'size'), **{c: (c, 'mean') for c in COEFFICIENTS}).reset_index()
        classes = edge.groupby(['reaction_type', *GROUP], observed=True, sort=True).agg(n_test_vectors=('n_test_vectors', 'sum'), n_test_edges=('component_pair', 'size'), **{c: (c, 'mean') for c in COEFFICIENTS}).reset_index()
        support = edge.groupby(GROUP, observed=True, sort=True).agg(n_test_vectors=('n_test_vectors', 'sum'), n_test_edges=('component_pair', 'size'), n_test_classes=('reaction_type', 'nunique')).reset_index()
        for weighting, source in [('pooled', subset), ('equal_edge', edge), ('equal_class_edge_balanced', classes)]:
            means = source.groupby(GROUP, observed=True, sort=True)[COEFFICIENTS].mean().reset_index()
            summaries.append(support.merge(means, on=GROUP, validate='one_to_one').assign(stratum=stratum, weighting=weighting, weight_basis='conditional_stratum_population'))
        edge_rows.append(edge.assign(stratum=stratum))
        class_rows.append(classes.assign(stratum=stratum, weighting='equal_edge'))
    return (pd.concat(edge_rows, ignore_index=True), pd.concat(class_rows, ignore_index=True), pd.concat(summaries, ignore_index=True))

def add_risks(frame):
    frame = frame.copy()
    for t, name in zip(GRID, ['t0', 't025', 't05', 't075', 't1']):
        frame['risk_' + name] = frame.R0 - 2 * frame.A * t + frame.B * t * t
        frame['change_' + name] = frame['risk_' + name] - frame.R0
    return frame

def build():
    e67 = RESPONSE_LEARNING / 'Local_Information_Content_E67'
    target_path = RESPONSE_LEARNING / 'Matched_Representation_E62/results/e62_vectors.parquet'
    vectors = pd.read_parquet(target_path).set_index('vector_id')
    losses = pd.read_parquet(e67 / 'results/e67_vector_losses.parquet')
    loss_lookup = losses.set_index([*QUERY, 'target_geometry', 'variant'])
    frames, inputs = ([], [target_path, e67 / 'results/e67_vector_losses.parquet'])
    max_endpoint_error = max_offgrid_error = max_orthogonality_error = 0.0
    for path in sorted((e67 / 'parts').glob('edge_*_predictions.parquet')):
        inputs.append(path)
        saved = pd.read_parquet(path)
        key = saved.iloc[0][EDGE].to_dict()
        for geometry, group in saved.groupby('target_geometry', sort=True):
            table = group.pivot(index=QUERY, columns='variant', values='prediction')
            raw = np.stack(vectors.loc[table.index.get_level_values('vector_id'), 'target_vector'])
            unit, constant = centered_unit(raw)
            target = raw if geometry == 'raw_delta' else unit
            predictions = {name: np.stack(table[name]) for name in table.columns}
            if geometry == 'raw_delta':
                dl = predictions['level_only'] - predictions['zero_local']
                dm = predictions['modulation_only'] - predictions['zero_local']
                max_orthogonality_error = max(max_orthogonality_error, float(np.max(np.abs(np.mean(dl * dm, axis=1)))), float(np.max(np.abs(dm.mean(axis=1)))), float(np.max(np.abs(dl - dl.mean(axis=1)[:, None]))))
            for g, anchor, destination, family in PATHS:
                if g != geometry:
                    continue
                p, q = (predictions[anchor], predictions[destination])
                residual, displacement = (target - p, q - p)
                rm, dm = (residual.mean(axis=1), displacement.mean(axis=1))
                r0, b = (components(residual), components(displacement))
                a = np.stack([np.mean(residual * displacement, axis=1), rm * dm, np.mean((residual - rm[:, None]) * (displacement - dm[:, None]), axis=1)], axis=1)
                for variant, risk in [(anchor, r0), (destination, r0 - 2 * a + b)]:
                    idx = pd.MultiIndex.from_tuples([(*i, geometry, variant) for i in table.index], names=[*QUERY, 'target_geometry', 'variant'])
                    old = loss_lookup.loc[idx, ['native_mse', 'level_mse', 'modulation_mse']].to_numpy(float)
                    max_endpoint_error = max(max_endpoint_error, float(np.max(np.abs(risk - old))))
                t = 0.137
                direct = components(target - (p + t * displacement))
                max_offgrid_error = max(max_offgrid_error, float(np.max(np.abs(direct - (r0 - 2 * a * t + b * t * t)))))
                for index, component in enumerate(('native', 'level', 'modulation')):
                    frame = table.reset_index()[QUERY].copy()
                    for name, value in key.items():
                        frame[name] = value
                    frame = frame.assign(target_geometry=geometry, path=anchor + '_to_' + destination, path_family=family, risk_component=component, true_constant=constant, R0=r0[:, index], A=a[:, index], B=b[:, index])
                    frames.append(frame)
    if max(max_endpoint_error, max_offgrid_error, max_orthogonality_error) > 1e-08:
        raise ValueError('Fixed update path arithmetic failed endpoint, off-grid, or orthogonality verification')
    data = pd.concat(frames, ignore_index=True)
    mean_keys = [*EDGE, 'family_request_id', 'vector_id', 'target_geometry', 'path', 'path_family', 'risk_component', 'true_constant']
    mean = data.groupby(mean_keys, observed=True, sort=True)[COEFFICIENTS].mean().reset_index()
    mean['allocation'] = 'mean_of_orders'
    data = pd.concat([data, mean], ignore_index=True)
    edges, classes, summary = aggregate(data)
    edge_combinations = edges[[*GROUP, 'stratum']].drop_duplicates()
    edge_registry = pd.read_csv(e67 / 'results/e67_support_bridge_1548.csv')[EDGE]
    edges = edge_registry.merge(edge_combinations, how='cross').merge(edges, on=[*EDGE, *GROUP, 'stratum'], how='left', validate='one_to_one')
    edges['supported'] = edges.n_test_vectors.notna()
    edges['n_test_vectors'] = edges.n_test_vectors.fillna(0).astype(int)
    combinations = classes[[*GROUP, 'stratum', 'weighting']].drop_duplicates()
    class_registry = pd.read_csv(e67 / 'results/e67_support_classes_20.csv')[['reaction_type']]
    classes = class_registry.merge(combinations, how='cross').merge(classes, on=['reaction_type', *GROUP, 'stratum', 'weighting'], how='left', validate='one_to_one')
    classes['supported'] = classes.n_test_vectors.notna()
    for name in ('n_test_vectors', 'n_test_edges'):
        classes[name] = classes[name].fillna(0).astype(int)
    OUT.mkdir(exist_ok=True)
    for frame, name in [(summary, 'summary'), (classes, 'classes'), (edges, 'edges')]:
        add_risks(frame).to_csv(OUT / ('m1_update_step_' + name + '.csv'), index=False)
    metadata = dict(protocol='Fixed E68 primary paths; no fitting or selected step', grid=GRID, source_sha256={str(p.relative_to(S)): sha(p) for p in inputs}, n_query_vectors=int(mean.vector_id.nunique()), n_allocation_orders=int(losses.allocation.nunique()), maximum_endpoint_error=max_endpoint_error, maximum_offgrid_error=max_offgrid_error, maximum_orthogonality_error=max_orthogonality_error, status='passed')
    (OUT / 'm1_update_step_checks.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return add_risks(summary)
if __name__ == '__main__':
    result = build()
    print(result[result.allocation.eq('mean_of_orders') & result.stratum.eq('all') & result.risk_component.eq('native') & result.weighting.eq('equal_class_edge_balanced')].to_string(index=False))
