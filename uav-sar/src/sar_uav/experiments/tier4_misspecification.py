"""Tier 4 Experiment: Model Misspecification and Robustness Boundary Analysis.

Corresponds to proposal Tier 4:
    - Tests planner under out-of-distribution ground truth q* not in S
    - Sweeps systematic attenuation and canopy mismatch multipliers
    - Identifies boundary where multi-hypothesis planning degrades
"""

from __future__ import annotations

import logging
from typing import Dict, List
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import HypothesisSet
from sar_uav.experiments.rng import IndexedRNGStream
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.sim.mission import ScheduleSimulator

log = logging.getLogger(__name__)


def run_tier4_misspecification(
    attenuation_factors: Sequence[float] = (0.3, 0.6, 1.0, 1.5, 2.0),
    num_missions: int = 8,
    seed: int = 42
) -> Dict:
    """Evaluates planner robustness against misspecified ground truth."""
    num_cells = 16
    H = 12
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(4) for c in range(4)]
    dist_m = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 350.0

    uav_spec = UAVSpec(speed_ms=12.0, battery_joules=300000.0, reserve_joules=35000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [uav_spec], H, delta_t=delta_t)

    M = np.eye(num_cells) * 0.8 + 0.2 / num_cells
    b0 = np.full(num_cells, 1.0 / num_cells)

    nominal_rates = np.full(num_cells, 0.15)
    veg_groups = np.array([0, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1])
    hset = HypothesisSet.create_structured_ensemble(nominal_rates, veg_groups, uncertainty_scale=0.5, num_hypotheses=3, seed=seed)
    hset_nom = HypothesisSet.create_nominal(nominal_rates)

    engine_ens = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    engine_nom = JointMassEngine(b0, hset_nom, M, H, delta_t=delta_t)

    candidate_cells = list(range(1, num_cells))
    explorer = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3))

    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=5, dwell_ticks=2)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    solver_ens = JointRouteEffortLocalSearch(explorer, ForwardBackwardFastEvaluator(engine_ens), max_evals=400, time_limit_sec=10.0)
    solver_nom = JointRouteEffortLocalSearch(explorer, ForwardBackwardFastEvaluator(engine_nom), max_evals=400, time_limit_sec=10.0)

    sched_ens, J_ens, _ = solver_ens.solve(init_sched)
    sched_nom, _, _ = solver_nom.solve(init_sched)

    rng_master = IndexedRNGStream(master_seed=seed)

    # -----------------------------------------------------------------
    # Protocol 1: In-Distribution Calibration Test
    # Ground truth hypothesis h_s is sampled per mission according to w_s
    # -----------------------------------------------------------------
    hyp_weights = np.array([h.weight for h in hset.hypotheses])
    hyp_weights /= hyp_weights.sum()

    calib_ens_detects = []
    calib_nom_detects = []

    for m_idx in range(num_missions):
        init_rng = rng_master.get_rng(m_idx, 0, 0, "calib_gt_sample")
        chosen_h_idx = int(init_rng.choice(len(hset.hypotheses), p=hyp_weights))
        mission_true_rates = hset.hypotheses[chosen_h_idx].lambda_rates

        sim_calib = ScheduleSimulator(checker, mission_true_rates, delta_t=delta_t)

        target_path = np.zeros(H, dtype=int)
        target_path[0] = int(init_rng.choice(num_cells, p=b0))
        for t in range(1, H):
            step_rng = rng_master.get_rng(m_idx, t, 0, "calib_target_step")
            target_path[t] = int(step_rng.choice(num_cells, p=M[target_path[t - 1]]))

        res_ens = sim_calib.run_simulation(sched_ens, target_path, mission_idx=m_idx, rng_stream=rng_master, predicted_J=J_ens)
        res_nom = sim_calib.run_simulation(sched_nom, target_path, mission_idx=m_idx, rng_stream=rng_master, predicted_J=0.0)

        calib_ens_detects.append(1 if res_ens.detected else 0)
        calib_nom_detects.append(1 if res_nom.detected else 0)

    protocol_1_calibration = {
        "J_predicted_ensemble": float(J_ens),
        "empirical_DSR_ensemble": float(np.mean(calib_ens_detects)),
        "empirical_DSR_nominal": float(np.mean(calib_nom_detects)),
        "ensemble_calibration_gap": float(J_ens - np.mean(calib_ens_detects))
    }

    # -----------------------------------------------------------------
    # Protocol 2: Out-of-Distribution Robustness Boundary Analysis
    # Sweeps systematic attenuation and adversarial canopy mismatch
    # -----------------------------------------------------------------
    results_by_attenuation = {}

    for alpha in attenuation_factors:
        true_rates = nominal_rates * alpha
        # Distort vegetation group 2 heavily (adversarial mismatch)
        for i in range(num_cells):
            if veg_groups[i] == 2:
                true_rates[i] *= 0.4

        sim = ScheduleSimulator(checker, true_rates, delta_t=delta_t)
        ens_detects = []
        nom_detects = []

        for m_idx in range(num_missions):
            target_path = np.zeros(H, dtype=int)
            init_rng = rng_master.get_rng(m_idx, 0, 0, f"tgt_init_{alpha}")
            target_path[0] = int(init_rng.choice(num_cells, p=b0))
            for t in range(1, H):
                step_rng = rng_master.get_rng(m_idx, t, 0, f"tgt_step_{alpha}")
                target_path[t] = int(step_rng.choice(num_cells, p=M[target_path[t - 1]]))

            res_ens = sim.run_simulation(sched_ens, target_path, mission_idx=m_idx, rng_stream=rng_master, predicted_J=J_ens)
            res_nom = sim.run_simulation(sched_nom, target_path, mission_idx=m_idx, rng_stream=rng_master, predicted_J=0.0)

            ens_detects.append(1 if res_ens.detected else 0)
            nom_detects.append(1 if res_nom.detected else 0)

        results_by_attenuation[float(alpha)] = {
            "DSR_Ensemble": float(np.mean(ens_detects)),
            "DSR_Nominal": float(np.mean(nom_detects)),
            "DSR_Delta": float(np.mean(ens_detects) - np.mean(nom_detects))
        }

    return {
        "protocol_1_calibration": protocol_1_calibration,
        "protocol_2_robustness_boundary": results_by_attenuation
    }


if __name__ == "__main__":
    res = run_tier4_misspecification(num_missions=4)
    print("Tier 4 Misspecification Results:")
    print("Protocol 1 (Calibration):", res["protocol_1_calibration"])
    print("Protocol 2 (Robustness Boundary):")
    for alpha, v in res["protocol_2_robustness_boundary"].items():
        print(f"  alpha={alpha}: {v}")
