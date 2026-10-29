"""Cross-background sign reversals for Figure 2B.

This publication step reads the packaged, previously used response corpus and
recipe registries. It does not read a precomputed reversal summary or fit a
model. Each comparison fixes a typed component swap, its component-specific
doses and a substrate pair, while exactly one other recipe component changes.
The changed background component may have a different dose; its entire slot
is the single changed field. Dose-only background changes are not included.

An unordered background pair is counted once. Shared substrates and wells
make comparisons dependent; these counts are descriptive, not independent
experimental replicates. Unsupported arrays are undefined, never zero rates.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARRAYS = [
    'acylation', 'allylic-substitution', 'buchwald-hartwig',
    'buchwald-hartwig-cmc', 'c-o-competitive', 'chan-lam', 'esterification',
    'grignard', 'heck', 'hiyama', 'kumada', 'minisci', 'pd-carbonylation',
    'reductive-amination', 'snar', 'sonogashira', 'suzuki', 'suzuki-bpin',
    'ullmann',
]
EDGE = ['reaction_type', 'swap_slot', 'component_pair']
CONTEXT = EDGE + ['background_fingerprint']
COMPONENT_SLOTS = {f'reagent{i}' for i in range(1, 7)} | {'solvent'}
THRESHOLD_PP = 20.0
_LAST_ANCHOR = None


def source_paths() -> list[Path]:
    """Return the three packaged inputs used by this step, for Source Data."""
    scripts = Path(__file__).resolve().parents[1]
    response = scripts / 'response_learning'
    e59 = response / 'Full_Edge_Corpus_E59/results'
    mining = response / 'Existing_Data_Mining_2026-10-16/results'
    return [e59 / 'full_edge_response_corpus.parquet',
            e59 / 'full_edge_strict_swap_registry.csv',
            mining / 'swap_registry.csv']


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _parse_background(value: str) -> dict[str, str]:
    fields = [field.split('=', 1) for field in value.split(';')]
    _require(all(len(pair) == 2 for pair in fields), 'Malformed recipe field')
    parsed = dict(fields)
    _require(len(parsed) == len(fields), 'Duplicate background slot')
    _require({'T', 't', 'solvent'} <= parsed.keys(), 'Incomplete background recipe')
    return parsed


def _component_id(value: str) -> str:
    if value in ('NA', 'None', 'nan', ''):
        return 'ABSENT'
    identity = value.split('@', 1)[0]
    _require(identity.startswith('CMP'), f'Unrecognized component: {value}')
    return identity


def _dose_signature(record) -> str:
    a, b = json.loads(record.recipe0), json.loads(record.recipe1)
    low, high = record.component_pair.split('|')
    slot = record.swap_slot
    doses = {a[f'{slot}_id']: a[f'{slot}_equiv'],
             b[f'{slot}_id']: b[f'{slot}_equiv']}
    _require(set(doses) == {low, high}, 'Recipe arms disagree with component pair')
    return json.dumps([doses[low], doses[high]], separators=(',', ':'))


def _context_metadata(strict_path: Path, recipes_path: Path) -> pd.DataFrame:
    strict = pd.read_csv(strict_path)
    strict = strict.loc[strict.reaction_type.isin(ARRAYS)].copy()
    recipes = pd.read_csv(recipes_path, usecols=['swap_id', 'recipe0', 'recipe1'])
    strict = strict.merge(recipes, on='swap_id', how='left', validate='one_to_one')
    _require(strict[['recipe0', 'recipe1']].notna().all().all(),
             'Strict swap lacks full recipe definitions')
    strict['swapped_doses'] = [_dose_signature(r) for r in strict.itertuples()]
    metadata = strict.groupby(CONTEXT, sort=True).agg(
        swapped_dose_variants=('swapped_doses', 'nunique'),
        swapped_doses=('swapped_doses', 'first'),
    ).reset_index()
    metadata['dose_unambiguous'] = metadata.swapped_dose_variants.eq(1)
    return metadata


def _one_component_difference(a: dict, b: dict):
    """Return the changed slot, or None if more than one field/identity differs."""
    _require(a.keys() == b.keys(), 'Background slots differ within the same swap')
    changes = [slot for slot in a if a[slot] != b[slot]]
    if len(changes) != 1 or changes[0] not in COMPONENT_SLOTS:
        return None
    slot = changes[0]
    if _component_id(a[slot]) == _component_id(b[slot]):
        return None
    return slot


def _neighbor_pairs(metadata: pd.DataFrame):
    rows, excluded = [], Counter()
    for edge, group in metadata.groupby(EDGE, sort=True):
        for a, b in combinations(group.to_dict('records'), 2):
            aa = _parse_background(a['background_fingerprint'])
            bb = _parse_background(b['background_fingerprint'])
            slot = _one_component_difference(aa, bb)
            if slot is None:
                excluded['not_one_other_component_identity_change'] += 1
                continue
            if not a['dose_unambiguous'] or not b['dose_unambiguous']:
                excluded['ambiguous_swapped_dose_in_collapsed_context'] += 1
                continue
            if a['swapped_doses'] != b['swapped_doses']:
                excluded['different_swapped_component_dose'] += 1
                continue
            rows.append({**dict(zip(EDGE, edge)),
                         'background_a': a['background_fingerprint'],
                         'background_b': b['background_fingerprint'],
                         'changed_slot': slot,
                         'changed_component_a': _component_id(aa[slot]),
                         'changed_component_b': _component_id(bb[slot])})
    _require(bool(rows), 'No eligible one-component background pairs')
    return pd.DataFrame(rows), dict(excluded)


def _match_response_pair(contexts: dict, background_pair) -> pd.DataFrame:
    edge = (background_pair.reaction_type, background_pair.swap_slot,
            background_pair.component_pair)
    a = contexts[edge + (background_pair.background_a,)]
    b = contexts[edge + (background_pair.background_b,)]
    paired = a.merge(b, on='substrate_pair_key', suffixes=('_a', '_b'),
                     validate='one_to_one')
    for role in ('sub_1_smiles', 'sub_2_smiles'):
        _require(paired[f'{role}_a'].eq(paired[f'{role}_b']).all(),
                 'Substrate identities differ across matched backgrounds')
    return paired


def _counts(pair: pd.DataFrame) -> dict[str, int]:
    x, y = pair.delta_a.to_numpy(float), pair.delta_b.to_numpy(float)
    _require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite response')
    reverse = x * y < 0
    nonzero = (x != 0) & (y != 0)
    decisive = (abs(x) >= THRESHOLD_PP) & (abs(y) >= THRESHOLD_PP)
    return {'n_shared_comparisons': len(pair),
            'n_nonzero_comparisons': int(nonzero.sum()),
            'n_nonzero_reversed': int((nonzero & reverse).sum()),
            'n_ge20_comparisons': int(decisive.sum()),
            'n_ge20_reversed': int((decisive & reverse).sum())}


def _with_rates(row: dict) -> dict:
    for threshold in ('nonzero', 'ge20'):
        n = row[f'n_{threshold}_comparisons']
        k = row[f'n_{threshold}_reversed']
        _require(0 <= k <= n, 'Invalid reversal numerator or denominator')
        row[f'{threshold}_reversal_pct'] = 100 * k / n if n else np.nan
        row[f'{threshold}_status'] = (
            'defined' if n else 'undefined_no_eligible_comparisons')
    return row


def verify_anchor(paired: pd.DataFrame) -> dict:
    """Fail closed unless the general matching engine recovers the flagship."""
    stats = _counts(paired)
    passed = stats['n_shared_comparisons'] == 412 and stats['n_nonzero_reversed'] == 145
    _require(passed, 'Flagship 145/412 anchor failed; reversal summary stopped')
    return {'shared_substrate_pairs': stats['n_shared_comparisons'],
            'opposite_sign_pairs': stats['n_nonzero_reversed'],
            'both_nonzero_pairs': stats['n_nonzero_comparisons'],
            'both_abs_ge20_pairs': stats['n_ge20_comparisons'],
            'opposite_both_abs_ge20': stats['n_ge20_reversed'],
            'same_general_neighbor_engine': True, 'passed': True}


def last_anchor_result() -> dict | None:
    """Return a copy of the most recent anchor check, without writing a file."""
    return None if _LAST_ANCHOR is None else dict(_LAST_ANCHOR)


def _self_check():
    # Boundary and sign checks: exact zero is not a sign, 20 is inclusive.
    synthetic = pd.DataFrame({'delta_a': [0, 20, 20, -20, 19.999],
                              'delta_b': [50, -20, 20, 20, -50]})
    _require(_counts(synthetic) == {
        'n_shared_comparisons': 5, 'n_nonzero_comparisons': 4,
        'n_nonzero_reversed': 3, 'n_ge20_comparisons': 3, 'n_ge20_reversed': 2},
        'Threshold/sign self-check failed')
    base = _parse_background('reagent2=CMP0200@0.2;reagent3=CMP0333@3;'
                             'reagent4=NA;reagent5=NA;reagent6=NA;'
                             'solvent=CMP0552@DMF;T=100;t=12')
    changed = dict(base, reagent3='CMP0334@3')
    _require(_one_component_difference(base, changed) == 'reagent3',
             'Single-component self-check failed')
    _require(_one_component_difference(base, dict(changed, T='120')) is None,
             'Two-field corruption was accepted')
    _require(_one_component_difference(base, dict(base, reagent3='CMP0333@2')) is None,
             'Dose-only background change was accepted')
    # The same IDs at different swapped doses must not become eligible neighbors.
    records = []
    for i, (background, doses, unambiguous) in enumerate([
        (base, '[0.05,0.05]', True),
        (changed, '[0.10,0.10]', True),
        (changed, '[0.05,0.05]', False),
    ]):
        records.append({'reaction_type': 'suzuki', 'swap_slot': 'reagent1',
                        'component_pair': 'CMP0092|CMP0097',
                        'background_fingerprint': ';'.join(f'{k}={v}' for k, v in background.items()),
                        'swapped_doses': doses, 'dose_unambiguous': unambiguous})
    try:
        _neighbor_pairs(pd.DataFrame(records))
    except ValueError as err:
        _require(str(err) == 'No eligible one-component background pairs',
                 'Unexpected dose self-check failure')
    else:
        raise ValueError('Swapped-dose/ambiguous-context corruption was accepted')
    try:
        verify_anchor(synthetic)
    except ValueError:
        pass
    else:
        raise ValueError('Corrupted anchor was accepted')


def build_decisive_reversals() -> pd.DataFrame:
    """Rebuild the 19-array and pooled summary from packaged inputs.

    The output columns and values match the separately reviewed V21 descriptive
    calculation. The function has no output-file side effects; the caller
    registers it with the ordinary publication Source Data writer.
    """
    global _LAST_ANCHOR
    _self_check()
    corpus_path, strict_path, recipes_path = source_paths()
    metadata = _context_metadata(strict_path, recipes_path)
    neighbors, excluded = _neighbor_pairs(metadata)
    corpus = pd.read_parquet(corpus_path, columns=[
        *CONTEXT, 'substrate_pair_key', 'sub_1_smiles', 'sub_2_smiles', 'delta'])
    corpus = corpus.loc[corpus.reaction_type.isin(ARRAYS)].copy()
    _require(len(corpus) == 824473 and corpus.reaction_type.nunique() == 19,
             'Unexpected previously used 19-array response population')
    _require(not corpus.duplicated(CONTEXT + ['substrate_pair_key']).any(),
             'Duplicate response identity in corpus')
    _require(np.isfinite(corpus.delta).all(), 'Nonfinite corpus response')
    contexts = {key: frame for key, frame in corpus.groupby(CONTEXT, sort=False)}
    flagship = neighbors.loc[
        neighbors.reaction_type.eq('suzuki')
        & neighbors.swap_slot.eq('reagent1')
        & neighbors.component_pair.eq('CMP0092|CMP0097')
        & neighbors.changed_component_a.isin(['CMP0333', 'CMP0334'])
        & neighbors.changed_component_b.isin(['CMP0333', 'CMP0334'])]
    _require(len(flagship) == 1, 'Flagship backgrounds are not uniquely identified')
    _LAST_ANCHOR = verify_anchor(_match_response_pair(
        contexts, next(flagship.itertuples(index=False))))
    counts = {array: {key: 0 for key in _counts(pd.DataFrame(
        {'delta_a': pd.Series(dtype=float), 'delta_b': pd.Series(dtype=float)}))}
        for array in ARRAYS}
    for pair in neighbors.itertuples(index=False):
        pair_counts = _counts(_match_response_pair(contexts, pair))
        for name, value in pair_counts.items():
            counts[pair.reaction_type][name] += value
    rows = [_with_rates({'level': 'array', 'reaction_type': array,
                         'component_group': 'ALL', **counts[array]}) for array in ARRAYS]
    pooled_counts = {name: sum(counts[array][name] for array in ARRAYS)
                     for name in next(iter(counts.values()))}
    pooled = _with_rates({'level': 'pooled', 'reaction_type': 'ALL',
                          'component_group': 'ALL', **pooled_counts})
    arrays = pd.DataFrame(rows)
    for threshold in ('nonzero', 'ge20'):
        # Preserve the reviewed 19-array reduction order as well as the
        # scientific denominator. NaNs stay undefined and pandas excludes them;
        # removing them before reduction can change the final float by one ULP.
        pooled[f'n_arrays_with_{threshold}_support'] = int(
            arrays[f'n_{threshold}_comparisons'].gt(0).sum())
        pooled[f'{threshold}_equal_array_mean_pct'] = arrays[f'{threshold}_reversal_pct'].mean()
        pooled[f'{threshold}_equal_array_median_pct'] = arrays[f'{threshold}_reversal_pct'].median()
    result = pd.DataFrame(rows + [pooled])
    result.attrs.update(anchor_result=dict(_LAST_ANCHOR),
                        n_background_pairs=len(neighbors),
                        excluded_background_pairs=excluded,
                        threshold_pp=THRESHOLD_PP)
    return result


if __name__ == '__main__':
    print(build_decisive_reversals().to_string(index=False))
