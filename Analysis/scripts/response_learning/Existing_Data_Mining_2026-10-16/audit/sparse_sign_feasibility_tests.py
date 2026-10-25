"""Independent exhaustive row/background sign enumeration for E47."""
from itertools import product
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from sparse_sign_feasibility import solve_graph

def exhaustive(rows, bgs, signs, mass, n_rows, n_bgs):
    count_best = len(rows) + 1
    mass_best = float('inf')
    for rs in product([1, -1], repeat=n_rows):
        for cs in product([1, -1], repeat=n_bgs):
            wrong = np.array(rs)[rows] * np.array(cs)[bgs] != signs
            count_best = min(count_best, int(wrong.sum()))
            mass_best = min(mass_best, float(np.sum(mass[wrong])))
    return (count_best, mass_best)

def run():
    rows = np.repeat(np.arange(3), 2)
    bgs = np.tile(np.arange(2), 3)
    signs = np.array([1, 1, 1, 1, 1, -1])
    mass = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0])
    result = solve_graph(rows, bgs, signs, mass, 3, 2)
    assert result['count']['n_mismatched'] == 1, 'count objective must detect frustrated cycle'
    assert result['mass']['n_mismatched'] == 2, 'weighted optimum differs from minimum count'
    assert result['mass']['mismatched_mass_pp'] == 2
    assert result['count']['background_signs'].tolist() == [1, 1], 'first lex count tie'
    tiny = np.array([0.1, 0.1, 0.2, 0.2, 0.1 + 0.2, 0.1 + 0.2])
    precision = solve_graph(rows, bgs, signs, tiny, 3, 2)
    assert precision['mass']['background_signs'].tolist() == [1, -1], 'exact binary costs must resolve sub-ulp objective difference'
    exact_cost = sum((Fraction.from_float(float(w)) for w, bad in zip(tiny, precision['mass']['mismatched']) if bad))
    count_cost = sum((Fraction.from_float(float(w)) for w, bad in zip(tiny, precision['count']['mismatched']) if bad))
    assert exact_cost < count_cost and float(exact_cost) == float(count_cost)
    rng = np.random.default_rng(20261017)
    checked = 1
    for n_rows, n_bgs in [(1, 1), (2, 2), (3, 2), (2, 3), (3, 3)]:
        for trial in range(12):
            mask = rng.random((n_rows, n_bgs)) > 0.35
            rr, bb = np.nonzero(mask)
            ss = rng.choice([-1, 1], size=len(rr))
            ww = rng.integers(1, 8, size=len(rr)).astype(float)
            result = solve_graph(rr, bb, ss, ww, n_rows, n_bgs)
            cb, wb = exhaustive(rr, bb, ss, ww, n_rows, n_bgs)
            assert result['count']['n_mismatched'] == cb
            assert result['mass']['mismatched_mass_pp'] == wb
            for objective in ('count', 'mass'):
                x = result[objective]
                wrong = x['row_signs'][rr] * x['background_signs'][bb] != ss
                assert np.array_equal(wrong, x['mismatched'])
            checked += 1
    rr, bb = (np.array([0, 1, 1]), np.array([0, 0, 1]))
    tree = solve_graph(rr, bb, np.array([1, 1, -1]), np.ones(3), 3, 3)
    assert tree['topology']['cycle_rank'] == 0 and tree['count']['n_mismatched'] == 0
    assert tree['topology']['n_isolated_rows'] == 1 and tree['topology']['n_isolated_backgrounds'] == 1
    assert tree['topology']['n_active_cells'] == 3
    print(f'Passed {checked} exhaustive graphs plus exact sub-ulp weighted tie and sparse-mask/isolate checks')
    return checked
if __name__ == '__main__':
    run()
