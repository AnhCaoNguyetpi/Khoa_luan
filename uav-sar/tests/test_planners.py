"""Tests for planning engine, local search, and baseline solvers."""

import numpy as np
import pytest

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.planning.baselines import AdaptiveGAPlanner, ExactBruteForcePlanner, GreedyLookaheadPlanner


@pytest.fixture
def test_setup():
    num_cells = 9
    H = 10
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(3) for c in range(3)]
    dist_m = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 200.0

    specs = [
        UAVSpec(name="uav_0", speed_ms=12.0, battery_joules=180000.0, reserve_joules=20000.0, delta_t=delta_t),
        UAVSpec(name="uav_1", speed_ms=12.0, battery_joules=180000.0, reserve_joules=20000.0, delta_t=delta_t)
    ]
    checker = ConstraintChecker(dist_m, depot_idx, specs, H, delta_t=delta_t)

    M = np.eye(num_cells)
    b0 = np.full(num_cells, 1.0 / num_cells)
    nominal = np.full(num_cells, 0.15)
    veg = np.zeros(num_cells, dtype=int)
    hset = HypothesisSet.create_structured_ensemble(nominal, veg, num_hypotheses=2, seed=42)

    engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
    return checker, engine, candidates, H, delta_t, num_cells


def test_greedy_planner_feasibility(test_setup):
    checker, engine, candidates, H, delta_t, num_cells = test_setup
    greedy = GreedyLookaheadPlanner(checker, engine, candidates, default_dwell=2)
    sched, J, stats = greedy.solve(num_uavs=2)

    assert J >= 0.0
    feasible, reason = checker.check_joint_schedule(sched)
    assert feasible, f"Greedy schedule infeasible: {reason}"
    assert stats["n_evals"] >= 1


def test_adaptive_ga_planner_feasibility(test_setup):
    checker, engine, candidates, H, delta_t, num_cells = test_setup
    ga = AdaptiveGAPlanner(checker, engine, candidates, population_size=10, max_generations=5, seed=42)
    sched, J, stats = ga.solve(num_uavs=1)

    assert J >= 0.0
    feasible, reason = checker.check_joint_schedule(sched)
    assert feasible, f"GA schedule infeasible: {reason}"


def test_exact_brute_force(test_setup):
    checker, engine, candidates, H, delta_t, num_cells = test_setup
    exact = ExactBruteForcePlanner(checker, engine, candidates[:4], max_visits_per_uav=2, allowed_dwells=(1, 2))
    sched, J_star, stats = exact.solve(num_uavs=1)

    assert J_star > 0.0
    feasible, reason = checker.check_joint_schedule(sched)
    assert feasible, f"Exact schedule infeasible: {reason}"


def test_local_search_improvement(test_setup):
    checker, engine, candidates, H, delta_t, num_cells = test_setup
    explorer = NeighborhoodExplorer(checker, candidates, allowed_dwells=(1, 2))
    evaluator = ForwardBackwardFastEvaluator(engine)

    # Suboptimal initial schedule: visit cell 1 with 1 tick
    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=1)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=200, time_limit_sec=5.0)
    best_sched, best_J, stats = solver.solve(init_sched)

    assert best_J >= stats["initial_J"]
    feasible, reason = checker.check_joint_schedule(best_sched)
    assert feasible, f"Local search result infeasible: {reason}"
