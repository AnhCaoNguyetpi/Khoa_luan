"""Unit test for JointMassEngine, hypothesis set, and exact equivalence between
Full Evaluator and Forward-Backward Fast Evaluator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.belief.joint_mass import JointMassEngine


def test_hypothesis_set_creation():
    nominal = np.array([0.1, 0.2, 0.3])
    hset = HypothesisSet.create_nominal(nominal)
    assert hset.S == 1
    assert np.allclose(hset.weights, [1.0])

    veg_groups = np.array([0, 1, 0])
    ensemble = HypothesisSet.create_structured_ensemble(nominal, veg_groups, num_hypotheses=4, seed=42)
    assert ensemble.S == 4
    assert np.allclose(ensemble.weights, np.full(4, 0.25))
    assert np.allclose(ensemble.hypotheses[0].lambda_rates, nominal)


def test_algebraic_equivalence_fast_and_full_evaluator():
    """Verify that Fast Evaluator gives the exact same result as Full Evaluator
    up to floating point precision.
    """
    num_cells = 8
    H = 15
    delta_t = 60.0

    b0 = np.array([0.05, 0.15, 0.2, 0.3, 0.1, 0.05, 0.1, 0.05])
    b0 /= np.sum(b0)

    # Synthetic Markov transition matrix (random walk on 1D graph)
    M = np.eye(num_cells) * 0.6
    for i in range(num_cells):
        if i > 0:
            M[i, i - 1] += 0.2
        else:
            M[i, i] += 0.2
        if i < num_cells - 1:
            M[i, i + 1] += 0.2
        else:
            M[i, i] += 0.2
    assert np.allclose(np.sum(M, axis=1), 1.0)

    nominal = np.full(num_cells, 0.12)
    veg_groups = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    hset = HypothesisSet.create_structured_ensemble(nominal, veg_groups, num_hypotheses=3, seed=123)

    engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)

    # Base schedule: search cell 2 at ticks 3..5, cell 4 at ticks 8..10
    base_fp = np.zeros((H, num_cells))
    base_fp[3:6, 2] = 1.0
    base_fp[8:11, 4] = 1.0

    u_minus, u_plus, J_base = engine.compute_forward_trajectory(base_fp)
    V_base = engine.compute_backward_continuation(base_fp)

    # Case 1: Internal modification at [t_a, t_b] = [3, 5] (change cell 2 to cell 3)
    mod_fp = base_fp.copy()
    mod_fp[3:6, 2] = 0.0
    mod_fp[3:6, 3] = 1.0

    # Full evaluation of modified schedule
    _, _, J_mod_full = engine.compute_forward_trajectory(mod_fp)

    # Fast evaluation using segment [3, 5]
    dJ_fast, J_mod_fast = engine.fast_evaluate_segment(
        mod_fp[3:6],
        t_a=3,
        t_b=5,
        u_ta_minus=u_minus[3],
        V_tb_plus_1=V_base[6],
        current_J_star=J_base
    )

    assert np.isclose(J_mod_fast, J_mod_full, atol=1e-12, rtol=0), f"Diff: {abs(J_mod_fast - J_mod_full)}"
    assert np.isclose(dJ_fast, J_mod_full - J_base, atol=1e-12, rtol=0)

    # Case 2: Modification extending to horizon end [8, 14]
    mod_tail_fp = base_fp.copy()
    mod_tail_fp[8:12, 4] = 0.0
    mod_tail_fp[8:12, 5] = 1.0

    _, _, J_tail_full = engine.compute_forward_trajectory(mod_tail_fp)
    dJ_tail_fast, J_tail_fast = engine.fast_evaluate_segment(
        mod_tail_fp[8:15],
        t_a=8,
        t_b=14,
        u_ta_minus=u_minus[8],
        V_tb_plus_1=V_base[15],
        current_J_star=J_base
    )
    assert np.isclose(J_tail_fast, J_tail_full, atol=1e-12, rtol=0)


if __name__ == "__main__":
    test_hypothesis_set_creation()
    test_algebraic_equivalence_fast_and_full_evaluator()
    print("All joint belief & evaluator tests passed successfully!")
