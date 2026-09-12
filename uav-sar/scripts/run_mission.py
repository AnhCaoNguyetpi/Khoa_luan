"""Run a single demo mission under persistent detection uncertainty."""

from _bootstrap import *  # noqa: F401,F403
import argparse
import json
from pathlib import Path
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, FullEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.sim.mission import ScheduleSimulator

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a single demo SAR mission")
    ap.add_argument("--evaluator", default="fast", choices=["fast", "prefix", "full"])
    ap.add_argument("--uavs", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    num_cells = 16  # 4x4
    H = a.horizon
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(4) for c in range(4)]
    dist_m = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 350.0

    specs = [
        UAVSpec(name=f"uav_{k}", speed_ms=12.0, battery_joules=300000.0, reserve_joules=35000.0, delta_t=delta_t)
        for k in range(a.uavs)
    ]
    checker = ConstraintChecker(dist_m, depot_idx, specs, H, delta_t=delta_t)

    # Diffusion motion model
    M = np.eye(num_cells) * 0.75 + 0.25 / num_cells
    b0 = np.full(num_cells, 1.0 / num_cells)
    b0[5] = 0.3
    b0 /= np.sum(b0)

    nominal_rates = np.full(num_cells, 0.16)
    veg_groups = np.arange(num_cells) % 3
    hset = HypothesisSet.create_structured_ensemble(nominal_rates, veg_groups, uncertainty_scale=0.5, num_hypotheses=3, seed=a.seed)

    engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    candidate_cells = list(range(1, num_cells))

    # Select evaluator tier
    if a.evaluator == "fast":
        evaluator = ForwardBackwardFastEvaluator(engine)
    elif a.evaluator == "prefix":
        evaluator = PrefixOnlyEvaluator(engine)
    else:
        evaluator = FullEvaluator(engine)

    explorer = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3))
    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=1000, time_limit_sec=10.0, verbose=True)

    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=k, visits=[Visit(cell_idx=5 + k*2, dwell_ticks=2)]) for k in range(a.uavs)],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    print("Optimizing schedule with JointRouteEffortLocalSearch...")
    best_sched, best_J, stats = solver.solve(init_sched)
    print(f"\nOptimization finished: J* = {best_J:.5f} in {stats['runtime_sec']:.2f}s ({stats['n_evals']} evals, reason: {stats['stopped_reason']})")

    # Simulate against hidden target
    true_lambda = hset.hypotheses[1].lambda_rates
    sim = ScheduleSimulator(checker, true_lambda, delta_t=delta_t)

    target_path = np.zeros(H, dtype=int)
    rng = np.random.RandomState(a.seed)
    target_path[0] = int(rng.choice(num_cells, p=b0))
    for t in range(1, H):
        target_path[t] = int(rng.choice(num_cells, p=M[target_path[t - 1]]))

    res = sim.run_simulation(best_sched, target_path, mission_idx=0, predicted_J=best_J)
    print(f"Simulation outcome: detected={res.detected}, tick={res.detection_tick}, energy={res.actual_energy_joules/1000.0:.1f} kJ, calib_gap={res.calibration_gap:.4f}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "demo_mission_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "predicted_J": best_J,
            "detected": res.detected,
            "detection_tick": res.detection_tick,
            "actual_energy_kJ": res.actual_energy_joules / 1000.0,
            "calibration_gap": res.calibration_gap,
            "solver_stats": stats
        }, f, indent=2)
    print(f"Result saved -> {out_file}")
