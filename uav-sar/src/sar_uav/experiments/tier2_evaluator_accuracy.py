"""Tier 2 Experiment: Evaluator Accuracy and Optimality Gap Benchmark.

Corresponds to proposal Tier 2:
    - Verifies numerical equivalence between Fast and Full evaluators.
    - Benchmarks 3 Evaluator Tiers: Full vs. Prefix-only vs. Forward-Backward Fast.
    - Measures speedup factors and optimality gap Delta_opt = J* - J_alg against ExactBruteForce.
"""

from __future__ import annotations

import logging
import time
from typing import Dict
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, FullEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.planning.baselines import ExactBruteForcePlanner

log = logging.getLogger(__name__)


def run_tier2_accuracy_benchmark(seed: int = 42) -> Dict:
    """Runs accuracy and speedup evaluation."""
    num_cells = 9
    H = 12
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(3) for c in range(3)]
    dist_m = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 300.0

    uav_spec = UAVSpec(speed_ms=12.0, battery_joules=250000.0, reserve_joules=30000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [uav_spec], H, delta_t=delta_t)

    # Random walk motion matrix
    rng = np.random.RandomState(seed)
    M = np.eye(num_cells) * 0.7
    for i in range(num_cells):
        r, c = coords[i]
        neighbors = []
        for j in range(num_cells):
            if i != j and abs(coords[j][0] - r) + abs(coords[j][1] - c) == 1:
                neighbors.append(j)
        if neighbors:
            prob = 0.3 / len(neighbors)
            for n in neighbors:
                M[i, n] = prob

    b0 = np.full(num_cells, 1.0 / num_cells)
    nominal = np.full(num_cells, 0.15)
    veg_groups = np.array([0, 1, 0, 1, 2, 0, 2, 1, 0])
    hset = HypothesisSet.create_structured_ensemble(nominal, veg_groups, uncertainty_scale=0.6, num_hypotheses=4, seed=seed)

    engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    candidate_cells = [1, 2, 3, 4, 5, 6, 7, 8]

    # Initial schedule
    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    # CRITICAL: Validate and compute action chaining for initial schedule BEFORE micro-benchmarks
    feasible, reason = checker.check_joint_schedule(init_sched)
    if not feasible:
        raise ValueError(f"Initial schedule is infeasible ({reason})")

    # 1. Exact Optimal J*
    exact_solver = ExactBruteForcePlanner(checker, engine, candidate_cells, max_visits_per_uav=3, allowed_dwells=(1, 2, 3))
    exact_sched, J_star, exact_stats = exact_solver.solve(num_uavs=1)

    # 2. Evaluator Micro-Benchmark on IDENTICAL candidate pool generated from chained schedule
    explorer_base = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3), max_wait_ticks=0, max_visits=3)
    candidate_pool = list(explorer_base.generate_all_neighbors(init_sched))
    n_candidates = len(candidate_pool)

    # Partition candidates by affected interval structure:
    # 1. short: L_aff <= 3 (localized perturbations)
    # 2. medium: 3 < L_aff < H (moderate interval)
    # 3. shifted: t_b == H - 1 (time shifts extending to horizon end)
    pool_short = []
    pool_medium = []
    pool_shifted = []

    for item in candidate_pool:
        _, _, t_a, t_b = item
        l_aff = t_b - t_a + 1
        if t_b == H - 1:
            pool_shifted.append(item)
        elif l_aff <= 3:
            pool_short.append(item)
        else:
            pool_medium.append(item)

    # Instantiate evaluators for micro-benchmark
    micro_full = FullEvaluator(engine)
    micro_prefix = PrefixOnlyEvaluator(engine)
    micro_fast = ForwardBackwardFastEvaluator(engine)

    micro_full.initialize_base_schedule(init_sched)
    micro_prefix.initialize_base_schedule(init_sched)
    micro_fast.initialize_base_schedule(init_sched)

    repeats = 10
    max_abs_err_fast = 0.0
    max_abs_err_prefix = 0.0

    # Warmup
    for _, cand, t_a, t_b in candidate_pool[:10]:
        micro_full.evaluate_candidate(cand, t_a, t_b)
        micro_prefix.evaluate_candidate(cand, t_a, t_b)
        micro_fast.evaluate_candidate(cand, t_a, t_b)

    # Measure on identical candidate pool with randomized evaluator ordering across repeats
    # to avoid CPU cache and thermal biasing.
    # Collect per-repeat timings to report variance.
    eval_funcs = {
        "full": lambda cand, ta, tb: micro_full.evaluate_candidate(cand, ta, tb),
        "prefix": lambda cand, ta, tb: micro_prefix.evaluate_candidate(cand, ta, tb),
        "fast": lambda cand, ta, tb: micro_fast.evaluate_candidate(cand, ta, tb)
    }
    per_repeat = {"full": [], "prefix": [], "fast": []}

    order_rng = np.random.RandomState(seed)
    eval_names = ["full", "prefix", "fast"]

    for _ in range(repeats):
        order_rng.shuffle(eval_names)
        for name in eval_names:
            fn = eval_funcs[name]
            t0 = time.perf_counter()
            for _, cand, t_a, t_b in candidate_pool:
                fn(cand, t_a, t_b)
            per_repeat[name].append(time.perf_counter() - t0)

    # Per-candidate mean and std (in seconds)
    def _stats(name):
        arr = np.array(per_repeat[name]) / n_candidates  # per-candidate time per repeat
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    t_full, t_full_std = _stats("full")
    t_prefix, t_prefix_std = _stats("prefix")
    t_fast, t_fast_std = _stats("fast")

    # Grouped micro-benchmarks by affected interval length
    def measure_subset(subset):
        if not subset:
            return {
                "count": 0,
                "full_ms": None, "prefix_ms": None, "fast_ms": None,
                "full_ms_std": None, "prefix_ms_std": None, "fast_ms_std": None,
                "speedup_vs_full": None, "speedup_vs_prefix": None,
            }
        n_sub = len(subset)
        sub_funcs = {
            "full": lambda: [micro_full.evaluate_candidate(cand, ta, tb) for _, cand, ta, tb in subset],
            "prefix": lambda: [micro_prefix.evaluate_candidate(cand, ta, tb) for _, cand, ta, tb in subset],
            "fast": lambda: [micro_fast.evaluate_candidate(cand, ta, tb) for _, cand, ta, tb in subset],
        }
        sub_per_repeat = {"full": [], "prefix": [], "fast": []}
        sub_names = ["full", "prefix", "fast"]
        sub_rng = np.random.RandomState(seed + 1)

        for _ in range(repeats):
            sub_rng.shuffle(sub_names)
            for name in sub_names:
                t0 = time.perf_counter()
                sub_funcs[name]()
                sub_per_repeat[name].append((time.perf_counter() - t0) / n_sub * 1000.0)

        avg_f = float(np.mean(sub_per_repeat["full"]))
        avg_p = float(np.mean(sub_per_repeat["prefix"]))
        avg_fa = float(np.mean(sub_per_repeat["fast"]))
        std_f = float(np.std(sub_per_repeat["full"], ddof=1))
        std_p = float(np.std(sub_per_repeat["prefix"], ddof=1))
        std_fa = float(np.std(sub_per_repeat["fast"], ddof=1))
        return {
            "count": n_sub,
            "full_ms": avg_f, "prefix_ms": avg_p, "fast_ms": avg_fa,
            "full_ms_std": std_f, "prefix_ms_std": std_p, "fast_ms_std": std_fa,
            "speedup_vs_full": avg_f / max(1e-6, avg_fa),
            "speedup_vs_prefix": avg_p / max(1e-6, avg_fa),
        }

    breakdown_short = measure_subset(pool_short)
    breakdown_medium = measure_subset(pool_medium)
    breakdown_shifted = measure_subset(pool_shifted)

    # Precision check across entire pool
    for _, cand, t_a, t_b in candidate_pool:
        dJ_full, _ = micro_full.evaluate_candidate(cand, t_a, t_b)
        dJ_prefix, _ = micro_prefix.evaluate_candidate(cand, t_a, t_b)
        dJ_fast, _ = micro_fast.evaluate_candidate(cand, t_a, t_b)
        err_fast = abs(dJ_fast - dJ_full)
        err_prefix = abs(dJ_prefix - dJ_full)
        if err_fast > max_abs_err_fast:
            max_abs_err_fast = err_fast
        if err_prefix > max_abs_err_prefix:
            max_abs_err_prefix = err_prefix

    micro_speedup_vs_full = t_full / max(1e-9, t_fast)
    micro_speedup_vs_prefix = t_prefix / max(1e-9, t_fast)

    # 3. Solver Macro-Benchmark (End-to-End Local Search)
    eval_fast = ForwardBackwardFastEvaluator(engine)
    explorer_fast = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3), max_wait_ticks=0, max_visits=3)
    solver_fast = JointRouteEffortLocalSearch(explorer_fast, eval_fast, max_evals=1000, time_limit_sec=10.0)
    sched_fast, J_fast, stats_fast = solver_fast.solve(init_sched)

    eval_prefix = PrefixOnlyEvaluator(engine)
    explorer_prefix = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3), max_wait_ticks=0, max_visits=3)
    solver_prefix = JointRouteEffortLocalSearch(explorer_prefix, eval_prefix, max_evals=1000, time_limit_sec=10.0)
    sched_prefix, J_prefix, stats_prefix = solver_prefix.solve(init_sched)

    eval_full = FullEvaluator(engine)
    explorer_full = NeighborhoodExplorer(checker, candidate_cells, allowed_dwells=(1, 2, 3), max_wait_ticks=0, max_visits=3)
    solver_full = JointRouteEffortLocalSearch(explorer_full, eval_full, max_evals=1000, time_limit_sec=10.0)
    sched_full, J_full, stats_full = solver_full.solve(init_sched)

    optimality_gap = float(J_star - J_fast)
    rel_gap = float(optimality_gap / max(1e-6, J_star))

    speedup_vs_full = (stats_full["evaluator_stats"]["avg_time_ms"] /
                       max(1e-6, stats_fast["evaluator_stats"]["avg_time_ms"]))
    speedup_vs_prefix = (stats_prefix["evaluator_stats"]["avg_time_ms"] /
                         max(1e-6, stats_fast["evaluator_stats"]["avg_time_ms"]))

    total_time_speedup_vs_full = (stats_full["runtime_sec"] /
                                  max(1e-6, stats_fast["runtime_sec"]))

    return {
        "exact_J_star": float(J_star),
        "fast_evaluator_J": float(J_fast),
        "prefix_only_J": float(J_prefix),
        "full_evaluator_J": float(J_full),
        "optimality_gap_absolute": optimality_gap,
        "optimality_gap_relative": rel_gap,
        "micro_benchmark": {
            "identical_candidate_count": n_candidates,
            "repeats": repeats,
            "latency_full_ms": float(t_full * 1000.0),
            "latency_full_ms_std": float(t_full_std * 1000.0),
            "latency_prefix_ms": float(t_prefix * 1000.0),
            "latency_prefix_ms_std": float(t_prefix_std * 1000.0),
            "latency_fast_ms": float(t_fast * 1000.0),
            "latency_fast_ms_std": float(t_fast_std * 1000.0),
            "speedup_vs_full": float(micro_speedup_vs_full),
            "speedup_vs_prefix": float(micro_speedup_vs_prefix),
            "max_abs_err_fast": float(max_abs_err_fast),
            "max_abs_err_prefix": float(max_abs_err_prefix),
            "breakdown_by_interval": {
                "short_L_le_3": breakdown_short,
                "medium_L_4_to_6": breakdown_medium,
                "long_or_shifted_to_H": breakdown_shifted
            }
        },
        "speedup_vs_full": float(speedup_vs_full),
        "speedup_vs_prefix": float(speedup_vs_prefix),
        "total_time_speedup_vs_full": float(total_time_speedup_vs_full),
        "solver_runtime_fast_sec": float(stats_fast["runtime_sec"]),
        "solver_runtime_prefix_sec": float(stats_prefix["runtime_sec"]),
        "solver_runtime_full_sec": float(stats_full["runtime_sec"]),
        "cache_update_time_fast_sec": float(stats_fast["cache_update_time_sec"]),
        "operator_stats_fast": stats_fast["operator_stats"],
        "evaluator_stats_fast": stats_fast["evaluator_stats"],
        "evaluator_stats_prefix": stats_prefix["evaluator_stats"],
        "evaluator_stats_full": stats_full["evaluator_stats"]
    }


if __name__ == "__main__":
    res = run_tier2_accuracy_benchmark()
    print("Tier 2 Accuracy & Speedup Benchmark Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
