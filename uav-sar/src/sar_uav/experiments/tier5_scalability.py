"""Tier 5 Experiment: Scalability Analysis and Quality-vs-Runtime Curves.

Corresponds to proposal Tier 5:
    - Scales problem instances across grid sizes (|G| in {16, 36, 64, 100}) and UAV fleet sizes (m in {1, 2, 4}).
    - Measures Quality-vs-Runtime curves for Fast Evaluator vs Full and Prefix-only.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Sequence
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, FullEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch

log = logging.getLogger(__name__)


def run_tier5_scalability(
    grid_sizes: Sequence[int] = (16, 36, 64),
    fleet_sizes: Sequence[int] = (1, 2),
    time_limit_sec: float = 5.0,
    seed: int = 42
) -> Dict:
    """Evaluates scalability and convergence speed across dimensions."""
    results = {}
    delta_t = 60.0
    H = 15

    for N_cells in grid_sizes:
        side = int(np.sqrt(N_cells))
        coords = [(r, c) for r in range(side) for c in range(side)]
        dist_m = np.zeros((N_cells, N_cells))
        for i in range(N_cells):
            for j in range(N_cells):
                dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 300.0

        # Scale horizon with grid dimension so UAVs can navigate across larger boards
        H = 10 + 2 * side

        M = np.eye(N_cells) * 0.8 + 0.2 / N_cells
        b0 = np.full(N_cells, 1.0 / N_cells)
        nominal_rates = np.full(N_cells, 0.15)
        veg_groups = np.arange(N_cells) % 3
        hset = HypothesisSet.create_structured_ensemble(nominal_rates, veg_groups, num_hypotheses=3, seed=seed)

        engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
        # Full candidate pool: all search cells excluding depot 0
        candidate_cells = list(range(1, N_cells))

        for m in fleet_sizes:
            specs = [UAVSpec(name=f"uav_{k}", speed_ms=12.0, battery_joules=300000.0, reserve_joules=35000.0, delta_t=delta_t)
                     for k in range(m)]
            checker = ConstraintChecker(dist_m, 0, specs, H, delta_t=delta_t)
            explorer = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3))

            init_sched = JointSchedule(
                uav_schedules=[UAVSchedule(uav_idx=k, visits=[Visit(cell_idx=min(N_cells-1, 1 + k * max(1, (N_cells-1)//m)), dwell_ticks=2)])
                               for k in range(m)],
                num_cells=N_cells,
                H=H,
                delta_t=delta_t
            )

            # 1. Fast Evaluator solver
            eval_fast = ForwardBackwardFastEvaluator(engine)
            solver_fast = JointRouteEffortLocalSearch(explorer, eval_fast, max_evals=400, time_limit_sec=time_limit_sec)
            _, J_fast, stats_fast = solver_fast.solve(init_sched)

            # 2. Prefix-only Evaluator solver
            eval_prefix = PrefixOnlyEvaluator(engine)
            solver_prefix = JointRouteEffortLocalSearch(explorer, eval_prefix, max_evals=400, time_limit_sec=time_limit_sec)
            _, J_prefix, stats_prefix = solver_prefix.solve(init_sched)

            # 3. Full Evaluator solver
            eval_full = FullEvaluator(engine)
            solver_full = JointRouteEffortLocalSearch(explorer, eval_full, max_evals=400, time_limit_sec=time_limit_sec)
            _, J_full, stats_full = solver_full.solve(init_sched)

            key = f"grid_{N_cells}_uav_{m}"
            results[key] = {
                "grid_size": N_cells,
                "num_uavs": m,
                "horizon_H": H,
                "candidate_pool_size": len(candidate_cells),
                "fast_J": float(J_fast),
                "prefix_J": float(J_prefix),
                "full_J": float(J_full),
                "fast_evals": stats_fast["n_evals"],
                "prefix_evals": stats_prefix["n_evals"],
                "full_evals": stats_full["n_evals"],
                "fast_avg_ms": stats_fast["evaluator_stats"]["avg_time_ms"],
                "prefix_avg_ms": stats_prefix["evaluator_stats"]["avg_time_ms"],
                "full_avg_ms": stats_full["evaluator_stats"]["avg_time_ms"],
                "speedup_vs_prefix": float(stats_prefix["evaluator_stats"]["avg_time_ms"] /
                                           max(1e-6, stats_fast["evaluator_stats"]["avg_time_ms"])),
                "speedup_vs_full": float(stats_full["evaluator_stats"]["avg_time_ms"] /
                                         max(1e-6, stats_fast["evaluator_stats"]["avg_time_ms"])),
                "fast_runtime_sec": float(stats_fast["runtime_sec"]),
                "prefix_runtime_sec": float(stats_prefix["runtime_sec"]),
                "full_runtime_sec": float(stats_full["runtime_sec"]),
                "total_speedup_vs_full": float(stats_full["runtime_sec"] /
                                               max(1e-6, stats_fast["runtime_sec"]))
            }

    return results


if __name__ == "__main__":
    res = run_tier5_scalability(grid_sizes=[16], fleet_sizes=[1], time_limit_sec=2.0)
    print("Tier 5 Scalability Sample Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
