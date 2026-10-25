"""Count E66 calibration support using frozen identities, never outcomes."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE.parent
E65 = ANALYSIS / 'Structure_Family_Transport_E65'
E62 = ANALYSIS / 'Matched_Representation_E62'
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
FOLD = [*EDGE, 'fold_role1', 'fold_role2']
BUDGETS = [0, 1, 2, 4, 8, 16, 32, 'full']
READS = {E65 / 'results/e65_family_requests.parquet': [*FOLD, 'family_request_id', 'family_role1', 'family_role2', 'query_vector_ids', 'n_query', 'original_train_vector_ids', 'n_original_train', 'family_train_vector_ids', 'n_family_train'], E65 / 'results/e65_scope_vectors.parquet': [*FOLD, 'vector_id', 'canonical_role1', 'canonical_role2', 'family_role1', 'family_role2', 'family_request_id', 'n_original_train', 'n_family_train'], E65 / 'results/e65_molecule_families.parquet': ['canonical_smiles', 'family', 'family_id'], E62 / 'results/e62_vectors.parquet': [*EDGE, 'vector_id', 'canonical_role1', 'canonical_role2', 'backgrounds', 'response_row_ids', 'condition_cell_ids', 'n_backgrounds'], E62 / 'results/e62_support_bridge_1548.csv': [*EDGE, 'edge_number', 'n_corpus_response_rows', 'n_backgrounds', 'n_substrate_keys', 'n_complete_vectors', 'n_incomplete_vectors', 'n_complete_single_role', 'n_complete_invalid_structure', 'n_candidate_vectors', 'panel_status', 'n_scored_vectors', 'complete_coverage'], E62 / 'results/e62_support_classes_20.csv': ['reaction_type', 'n_registered_edges', 'n_corpus_response_rows', 'n_eligible_edges', 'n_candidate_vectors', 'n_scored_vectors']}
FORBIDDEN = {'target_vector', 'allzero_vector', 'n_complete_allzero_vectors'}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def pins():
    return {str(path.relative_to(ANALYSIS)): sha(path) for path in READS}

def load(path):
    columns = READS[path]
    assert not FORBIDDEN.intersection(columns)
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)

def union(items, mapping):
    result = set()
    for item in items:
        result.update(mapping[item])
    return result

def main():
    started = time.perf_counter()
    before = pins()
    fs, scope, molecules, vectors, bridge, classes = [load(path) for path in READS]
    assert len(bridge) == 1550 and len(classes) == 20
    assert not fs.family_request_id.duplicated().any()
    assert not scope.vector_id.duplicated().any()
    assert not vectors.vector_id.duplicated().any()
    assert len(scope) == len(vectors) == 25956
    assert set(scope.vector_id) == set(vectors.vector_id)
    assert not molecules.canonical_smiles.duplicated().any()
    family = molecules.set_index('canonical_smiles').family.to_dict()
    vi = vectors.set_index('vector_id')
    molecule_sets = {int(r.vector_id): {r.canonical_role1, r.canonical_role2} for r in vectors.itertuples(index=False)}
    family_sets = {int(r.vector_id): {family[r.canonical_role1], family[r.canonical_role2]} for r in vectors.itertuples(index=False)}
    cell_sets = {int(r.vector_id): set(map(int, r.condition_cell_ids)) for r in vectors.itertuples(index=False)}
    coordinates = vectors.set_index('vector_id').n_backgrounds.astype(int).to_dict()
    response_sets = {int(r.vector_id): set(map(int, r.response_row_ids)) for r in vectors.itertuples(index=False)}
    for r in vectors.itertuples(index=False):
        assert len(r.backgrounds) == len(r.response_row_ids) == r.n_backgrounds
        assert len(r.condition_cell_ids) == 2 * r.n_backgrounds
    for r in scope.itertuples(index=False):
        s = vi.loc[r.vector_id]
        assert (r.canonical_role1, r.canonical_role2) == (s.canonical_role1, s.canonical_role2)
        assert (r.family_role1, r.family_role2) == (family[s.canonical_role1], family[s.canonical_role2])
    scope_requests = {key: set(map(int, group.vector_id)) for key, group in scope.groupby('family_request_id', sort=False)}
    fold_queries = {key: set(map(int, group.vector_id)) for key, group in scope.groupby(FOLD, sort=False)}
    original_folds = {}
    rows = []
    for r in fs.itertuples(index=False):
        key = tuple((getattr(r, name) for name in FOLD))
        original = set(map(int, r.original_train_vector_ids))
        retained = set(map(int, r.family_train_vector_ids))
        query = set(map(int, r.query_vector_ids))
        assert len(original) == r.n_original_train == len(r.original_train_vector_ids)
        assert len(retained) == r.n_family_train == len(r.family_train_vector_ids)
        assert len(query) == r.n_query == len(r.query_vector_ids)
        assert query == scope_requests[r.family_request_id]
        assert retained <= original
        calibration = original - retained
        assert not calibration.intersection(retained)
        assert calibration.union(retained) == original
        held = {r.family_role1, r.family_role2}
        assert all((not family_sets[v].intersection(held) for v in retained))
        assert all((family_sets[v].intersection(held) for v in calibration))
        assert calibration == {v for v in original if family_sets[v].intersection(held)}
        assert all((family_sets[v] == held for v in query))
        if key not in original_folds:
            all_query = fold_queries[key]
            assert not original.intersection(all_query)
            assert not union(original, molecule_sets).intersection(union(all_query, molecule_sets))
            train_cells = union(original, cell_sets)
            assert not train_cells.intersection(union(all_query, cell_sets))
            assert len(train_cells) == sum((len(cell_sets[v]) for v in original))
            assert not union(original, response_sets).intersection(union(all_query, response_sets))
            original_folds[key] = original
        assert original == original_folds[key]
        assert not calibration.intersection(query) and (not retained.intersection(query))
        rows.append({**{name: getattr(r, name) for name in EDGE}, 'family_request_id': r.family_request_id, 'n_query': len(query), 'n_original_train': len(original), 'n_family_train': len(retained), 'n_calibration_available': len(calibration), 'calibration_pool_scalar_coordinates': sum((coordinates[v] for v in calibration)), 'calibration_pool_condition_cells': sum((len(cell_sets[v]) for v in calibration)), 'query_scalar_coordinates': sum((coordinates[v] for v in query))})
    counts = pd.DataFrame(rows)
    assert len(original_folds) == 1984
    assert counts.n_query.sum() == len(vectors)
    assert (counts.n_original_train == counts.n_family_train + counts.n_calibration_available).all()
    summaries = []
    for floor in [0, 1]:
        for budget in BUDGETS:
            available = counts.n_family_train.ge(floor)
            if budget != 'full':
                available &= counts.n_calibration_available.ge(budget)
            selected = counts.loc[available]
            summaries.append({'minimum_family_training': floor, 'budget': budget, 'n_requests': len(selected), 'n_query_vectors': int(selected.n_query.sum()), 'n_edges': len(selected[EDGE].drop_duplicates()), 'n_classes': selected.reaction_type.nunique(), 'n_query_scalar_coordinates': int(selected.query_scalar_coordinates.sum()), 'n_query_vectors_no_family_training': int(selected.loc[selected.n_family_train.eq(0), 'n_query'].sum()), 'n_query_vectors_no_calibration': int(selected.loc[selected.n_calibration_available.eq(0), 'n_query'].sum()), 'n_query_vectors_not_in_cohort': int(counts.n_query.sum() - selected.n_query.sum())})
            label = f'tf{floor}_b{budget}'
            for output, groupcols in [(bridge, EDGE), (classes, ['reaction_type'])]:
                grouped = selected.groupby(groupcols, sort=False).agg(requests=('family_request_id', 'size'), query_vectors=('n_query', 'sum'))
                lookup = grouped.to_dict('index')
                keys = output[groupcols].itertuples(index=False, name=None)
                entries = [lookup.get(key if len(groupcols) > 1 else key[0], {}) for key in keys]
                output[f'{label}_requests'] = [int(value.get('requests', 0)) for value in entries]
                output[f'{label}_query_vectors'] = [int(value.get('query_vectors', 0)) for value in entries]
    for output, groupcols in [(bridge, EDGE), (classes, ['reaction_type'])]:
        total = output['tf0_bfull_query_vectors']
        output['e66_scope_status'] = np.where(total.gt(0), 'in_E65_scope', 'outside_E65_vector_scope')
        for value, label in [(counts.n_family_train.eq(0), 'no_family_training'), (counts.n_calibration_available.eq(0), 'no_calibration_available')]:
            group = counts.loc[value].groupby(groupcols).n_query.sum().to_dict()
            output[f'{label}_query_vectors'] = [int(group.get(key if len(groupcols) > 1 else key[0], 0)) for key in output[groupcols].itertuples(index=False, name=None)]
    after = pins()
    assert before == after
    for name in ['results', 'metadata']:
        (BASE / name).mkdir(parents=True, exist_ok=True)
    outputs = {'summary': BASE / 'results/e66_feasibility_counts.csv', 'requests': BASE / 'results/e66_feasibility_requests.parquet', 'bridge': BASE / 'results/e66_feasibility_bridge_1548.csv', 'classes': BASE / 'results/e66_feasibility_classes_20.csv'}
    pd.DataFrame(summaries).to_csv(outputs['summary'], index=False)
    counts.to_parquet(outputs['requests'], index=False)
    bridge.to_csv(outputs['bridge'], index=False)
    classes.to_csv(outputs['classes'], index=False)
    metadata = {'schema_version': 'E66_outcome_blind_feasibility_v1', 'purpose': 'Support counts only; no E66 protocol, selection order, fit, or prediction.', 'source_sha256_before': before, 'source_sha256_after': after, 'code_sha256': sha(Path(__file__)), 'columns_read': {str(path.relative_to(ANALYSIS)): columns for path, columns in READS.items()}, 'outcome_columns_read': [], 'excluded_outcome_columns': sorted(FORBIDDEN), 'counts': {'requests': len(counts), 'query_vectors': len(vectors), 'scalar_coordinates': int(vectors.n_backgrounds.sum()), 'original_folds': len(original_folds), 'bridge_edges': len(bridge), 'bridge_classes': len(classes), 'no_family_training_requests': int(counts.n_family_train.eq(0).sum()), 'no_family_training_query_vectors': int(counts.loc[counts.n_family_train.eq(0), 'n_query'].sum()), 'no_calibration_requests': int(counts.n_calibration_available.eq(0).sum()), 'no_calibration_query_vectors': int(counts.loc[counts.n_calibration_available.eq(0), 'n_query'].sum())}, 'support': summaries, 'checks': {'calibration_and_retained_partition_original': True, 'each_nonempty_calibration_vector_contains_a_held_family': True, 'retained_vectors_contain_no_held_family': True, 'calibration_is_exact_original_held_family_subset': True, 'cross_role_canonical_identity_train_query_disjoint': True, 'condition_cells_train_query_disjoint': True, 'original_training_cells_have_unique_vector_owner': True, 'calibration_retained_condition_cells_disjoint_by_unique_owner': True, 'response_rows_train_query_disjoint': True, 'scope_and_request_query_memberships_equal': True, 'source_files_unchanged': before == after}, 'budget_interpretation': {'integer': 'Exact-B availability requires |C|>=B and the indicated TF minimum.', 'full': 'All available C; includes C empty and every indicated TF minimum.', 'common_cohort': 'Choose a maximum integer budget before outcomes; the exact-B common cohort is the row for that maximum.', 'pool_costs': 'Unordered full candidate-pool counts only; no selected-budget raw-assay costs computed.'}, 'output_sha256': {str(path.relative_to(BASE)): sha(path) for path in outputs.values()}, 'seconds': time.perf_counter() - started}
    (BASE / 'metadata/e66_feasibility_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    print(json.dumps({'counts': metadata['counts'], 'support': summaries, 'seconds': metadata['seconds']}), flush=True)
if __name__ == '__main__':
    main()
