"""Bayesian negative-update correctness vs. the proposal formulas."""
from _common import small_area
import numpy as np

from sar_uav.belief.update import bayes_negative_update, combine_q


def test_matches_proposal_formula():
    p = np.zeros(10) + 0.05
    p[3] = 0.35
    p /= p.sum()
    i, q = 3, 0.4
    pn = bayes_negative_update(p, [i], [q])
    denom = 1 - p[i] * q
    assert abs(pn[i] - p[i] * (1 - q) / denom) < 1e-12
    j = 7
    assert abs(pn[j] - p[j] / denom) < 1e-12
    assert abs(pn.sum() - 1.0) < 1e-9


def test_multi_cell_disjoint():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    p /= p.sum()
    cells = [5, 20, 33]
    qs = [0.3, 0.7, 0.1]
    pn = bayes_negative_update(p, cells, qs)
    D = 1 - sum(p[c] * q for c, q in zip(cells, qs))
    assert abs(pn.sum() - 1) < 1e-9
    for c, q in zip(cells, qs):
        assert abs(pn[c] - p[c] * (1 - q) / D) < 1e-12
    # unsearched cell scaled by 1/D
    assert abs(pn[0] - p[0] / D) < 1e-12


def test_duplicate_cells_merged():
    p = np.full(8, 1 / 8)
    pn = bayes_negative_update(p, [2, 2], [0.5, 0.5])
    combined = combine_q([0.5, 0.5])          # 0.75
    D = 1 - p[2] * combined
    assert abs(pn[2] - p[2] * (1 - combined) / D) < 1e-12


def test_input_not_mutated():
    p = np.full(6, 1 / 6)
    before = p.copy()
    bayes_negative_update(p, [1], [0.9])
    assert np.allclose(p, before)


def test_combine_q_bounds():
    assert combine_q([0.2, 0.3]) == 1 - (0.8 * 0.7)   # 1 - prod(1-q)
    assert combine_q([1.0, 0.5]) == 1.0
    assert 0 <= combine_q([0.99, 0.99, 0.99]) <= 1
