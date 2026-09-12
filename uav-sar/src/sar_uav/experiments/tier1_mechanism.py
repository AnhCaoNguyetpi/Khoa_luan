"""Tier 1 Experiment: Mechanism Analysis on small 3x3 grid.

Corresponds to proposal Tier 1:
    - Small 3x3 grid, 1-2 UAVs
    - Demonstrates 3 key mechanism scenarios:
        1. Ensemble provides clear advantage over nominal.
        2. Nominal is accurate and ensemble provides no additional gain.
        3. High uncertainty but optimal action remains unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.baselines import ExactBruteForcePlanner

log = logging.getLogger(__name__)


def run_tier1_mechanism_analysis(seed: int = 42) -> Dict:
    """Runs the 3 mechanism scenarios on a 3x3 grid."""
    num_cells = 9
    H = 10
    delta_t = 60.0
    depot_idx = 0

    # Distance matrix for 3x3 grid with 300m cells
    coords = [(r, c) for r in range(3) for c in range(3)]
    dist_m = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            dr = coords[i][0] - coords[j][0]
            dc = coords[i][1] - coords[j][1]
            dist_m[i, j] = np.hypot(dr, dc) * 300.0

    uav_spec = UAVSpec(speed_ms=10.0, battery_joules=180000.0, reserve_joules=20000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [uav_spec], H, delta_t=delta_t)

    # Stationary target in this mechanism test
    M = np.eye(num_cells)
    candidate_cells = [1, 2, 3, 4, 5, 6, 7, 8]

    results = {}

    # -------------------------------------------------------------
    # Scenario 1: Ensemble has clear advantage over nominal
    # -------------------------------------------------------------
    # Cell 1 is near depot, high prior (0.35), reliable rate (0.15).
    # Cell 3 has prior 0.33, but STRONGLY bimodal rate: 0.001 under s1, 0.35 under s2 (mean 0.1755).
    # Cell 4 has prior 0.30, reliable rate (0.15) under all hypotheses.
    # Nominal evaluator optimizes using mean rate and chooses cell 3 over cell 4.
    # Ensemble evaluator accounts for Jensen's inequality / downside risk and hedges
    # by visiting reliable cell 4 instead, achieving higher true expected detection.
    b0_scen1 = np.full(num_cells, 0.01)
    b0_scen1[1] = 0.35
    b0_scen1[3] = 0.33
    b0_scen1[4] = 0.30
    b0_scen1 /= np.sum(b0_scen1)

    h1_rates = np.full(num_cells, 0.05)
    h1_rates[1] = 0.15
    h1_rates[3] = 0.001   # nearly undetectable under s1
    h1_rates[4] = 0.15

    h2_rates = np.full(num_cells, 0.05)
    h2_rates[1] = 0.15
    h2_rates[3] = 0.35    # high under s2
    h2_rates[4] = 0.15

    h1 = DetectionHypothesis(0, "s1_cell3_weak", 0.5, h1_rates)
    h2 = DetectionHypothesis(1, "s2_cell3_strong", 0.5, h2_rates)
    hset_scen1 = HypothesisSet([h1, h2])
    hset_nom1 = HypothesisSet.create_nominal(np.mean([h1.lambda_rates, h2.lambda_rates], axis=0))

    eng_ens1 = JointMassEngine(b0_scen1, hset_scen1, M, H, delta_t)
    eng_nom1 = JointMassEngine(b0_scen1, hset_nom1, M, H, delta_t)

    candidate_subset = [1, 3, 4]
    exact_ens1 = ExactBruteForcePlanner(checker, eng_ens1, candidate_subset, max_visits_per_uav=2, allowed_dwells=(2, 3))
    exact_nom1 = ExactBruteForcePlanner(checker, eng_nom1, candidate_subset, max_visits_per_uav=2, allowed_dwells=(2, 3))

    best_sched_ens1, J_ens1, _ = exact_ens1.solve(num_uavs=1)
    best_sched_nom1, _, _ = exact_nom1.solve(num_uavs=1)

    # Evaluate nominal schedule under true ensemble
    _, _, J_nom1_under_ens = eng_ens1.compute_forward_trajectory(best_sched_nom1.generate_footprint_array())

    ens1_cells = [v.cell_idx for v in best_sched_ens1.uav_schedules[0].visits]
    nom1_cells = [v.cell_idx for v in best_sched_nom1.uav_schedules[0].visits]

    results["scenario_1"] = {
        "description": "Ensemble hedges against bimodal cell 3 detection rate by selecting cell 4 instead",
        "ensemble_optimal_J": float(J_ens1),
        "nominal_choice_evaluated_on_ensemble": float(J_nom1_under_ens),
        "gain": float(J_ens1 - J_nom1_under_ens),
        "different_decisions": bool(ens1_cells != nom1_cells),
        "ensemble_cells": ens1_cells,
        "nominal_cells": nom1_cells
    }

    # -------------------------------------------------------------
    # Scenario 2: Nominal is accurate, ensemble provides no gain
    # -------------------------------------------------------------
    h_true = DetectionHypothesis(0, "accurate_nominal", 1.0, np.full(num_cells, 0.15))
    hset_scen2 = HypothesisSet([h_true])
    eng_scen2 = JointMassEngine(b0_scen1, hset_scen2, M, H, delta_t)

    exact2 = ExactBruteForcePlanner(checker, eng_scen2, candidate_subset, max_visits_per_uav=2, allowed_dwells=(2, 3))
    _, J2, _ = exact2.solve(num_uavs=1)

    results["scenario_2"] = {
        "description": "Nominal is exact ground truth (S=1), ensemble equals nominal",
        "J_optimal": float(J2),
        "gain": 0.0
    }

    # -------------------------------------------------------------
    # Scenario 3: High uncertainty but optimal decision unchanged
    # -------------------------------------------------------------
    # Cell 1 dominates massively (prior 0.90) so both nominal and ensemble visit cell 1
    b0_scen3 = np.full(num_cells, 0.01)
    b0_scen3[1] = 0.90
    b0_scen3 /= np.sum(b0_scen3)

    eng_ens3 = JointMassEngine(b0_scen3, hset_scen1, M, H, delta_t)
    eng_nom3 = JointMassEngine(b0_scen3, hset_nom1, M, H, delta_t)

    exact_ens3 = ExactBruteForcePlanner(checker, eng_ens3, candidate_subset, max_visits_per_uav=2, allowed_dwells=(2, 3))
    exact_nom3 = ExactBruteForcePlanner(checker, eng_nom3, candidate_subset, max_visits_per_uav=2, allowed_dwells=(2, 3))

    sched_ens3, J_ens3, _ = exact_ens3.solve(num_uavs=1)
    sched_nom3, _, _ = exact_nom3.solve(num_uavs=1)
    _, _, J_nom3_on_ens = eng_ens3.compute_forward_trajectory(sched_nom3.generate_footprint_array())

    ens3_cells = [v.cell_idx for v in sched_ens3.uav_schedules[0].visits]
    nom3_cells = [v.cell_idx for v in sched_nom3.uav_schedules[0].visits]

    results["scenario_3"] = {
        "description": "High uncertainty but optimal route identical due to high prior concentration",
        "ensemble_optimal_J": float(J_ens3),
        "nominal_on_ensemble_J": float(J_nom3_on_ens),
        "identical_performance": bool(abs(J_ens3 - J_nom3_on_ens) < 1e-4),
        "ensemble_cells": ens3_cells,
        "nominal_cells": nom3_cells
    }

    # -------------------------------------------------------------
    # Scenario 4: When ensemble is disadvantageous
    # -------------------------------------------------------------
    # If the ground truth is actually the optimistic nominal model, the ensemble
    # hedges conservatively against downside risk, incurring an opportunity penalty.
    eng_gt = JointMassEngine(b0_scen1, hset_nom1, M, H, delta_t)

    _, _, J_ens4_under_gt = eng_gt.compute_forward_trajectory(best_sched_ens1.generate_footprint_array())
    _, _, J_nom4_under_gt = eng_gt.compute_forward_trajectory(best_sched_nom1.generate_footprint_array())

    results["scenario_4"] = {
        "description": "Ensemble conservative hedge underperforms nominal when ground truth is the optimistic nominal",
        "ensemble_J_under_gt": float(J_ens4_under_gt),
        "nominal_J_under_gt": float(J_nom4_under_gt),
        "ensemble_penalty": float(J_nom4_under_gt - J_ens4_under_gt)
    }

    return results


if __name__ == "__main__":
    res = run_tier1_mechanism_analysis()
    print("Tier 1 Mechanism Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")

