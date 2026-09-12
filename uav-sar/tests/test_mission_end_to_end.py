"""End-to-end mission execution and simulation tests."""

import numpy as np
import pytest

from sar_uav.experiments.rng import IndexedRNGStream
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.sim.mission import ScheduleSimulator


def test_simulation_execution_and_early_termination():
    num_cells = 9
    H = 10
    delta_t = 60.0
    depot_idx = 0

    dist_m = np.zeros((9, 9))
    coords = [(r, c) for r in range(3) for c in range(3)]
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 250.0

    uav_spec = UAVSpec(speed_ms=12.0, battery_joules=200000.0, reserve_joules=25000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [uav_spec], H, delta_t=delta_t)

    # Schedule visits cell 4 from tick 2 to 4
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=3)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )
    feasible, msg = checker.check_joint_schedule(sched)
    assert feasible, f"Infeasible: {msg}"

    # Target path stays in cell 4
    target_path = np.full(H, 4)

    # High detection rate to guarantee detection
    true_lambda = np.full(num_cells, 5.0)
    sim = ScheduleSimulator(checker, true_lambda, delta_t=delta_t)

    rng_stream = IndexedRNGStream(master_seed=123)
    res = sim.run_simulation(sched, target_path, mission_idx=0, rng_stream=rng_stream, predicted_J=0.9)

    assert res.detected is True
    assert res.detection_tick is not None
    assert 1 <= res.detection_tick <= 4
    assert res.rmst_time == float(res.detection_tick)
    assert res.actual_energy_joules > 0.0
    assert abs(res.calibration_gap) <= 1.0


def test_simulation_non_detection():
    num_cells = 9
    H = 10
    delta_t = 60.0
    depot_idx = 0

    dist_m = np.zeros((9, 9))
    coords = [(r, c) for r in range(3) for c in range(3)]
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 250.0

    uav_spec = UAVSpec(speed_ms=12.0, battery_joules=200000.0, reserve_joules=25000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [uav_spec], H, delta_t=delta_t)

    # Schedule visits cell 1
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=2)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    # Target is in cell 8 (disjoint from search)
    target_path = np.full(H, 8)
    true_lambda = np.full(num_cells, 0.2)
    sim = ScheduleSimulator(checker, true_lambda, delta_t=delta_t)

    res = sim.run_simulation(sched, target_path, mission_idx=1, predicted_J=0.3)
    assert res.detected is False
    assert res.detection_tick is None
    assert res.rmst_time == float(H)
