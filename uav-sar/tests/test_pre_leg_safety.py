"""Unit test for pre-leg departure safety checks and energy constraints."""

import numpy as np
import pytest

from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit


def test_pre_leg_departure_energy_and_deadline_check():
    num_cells = 5
    H = 12
    delta_t = 60.0
    depot_idx = 0

    # Distance matrix where cell 4 is very far away
    dist_m = np.zeros((5, 5))
    dist_m[0, 1] = 1000.0
    dist_m[1, 0] = 1000.0
    dist_m[1, 4] = 8000.0  # very far
    dist_m[4, 1] = 8000.0
    dist_m[4, 0] = 9000.0
    dist_m[0, 4] = 9000.0

    # Low battery UAV
    spec_tight = UAVSpec(speed_ms=10.0, battery_joules=100000.0, reserve_joules=20000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [spec_tight], H, delta_t=delta_t)

    # Schedule: depot -> 1 -> 4 -> depot
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[
            Visit(cell_idx=1, dwell_ticks=2),
            Visit(cell_idx=4, dwell_ticks=2)
        ])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    feasible, reason = checker.check_joint_schedule(sched)
    # Must be infeasible because flight to node 4 exceeds pre-leg energy or deadline
    assert feasible is False
    assert "deadline" in reason or "energy" in reason


def test_feasible_schedule_passes_pre_leg_check():
    num_cells = 3
    H = 15
    delta_t = 60.0
    depot_idx = 0

    dist_m = np.full((3, 3), 300.0)
    np.fill_diagonal(dist_m, 0.0)

    spec = UAVSpec(speed_ms=15.0, battery_joules=360000.0, reserve_joules=30000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [spec], H, delta_t=delta_t)

    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[
            Visit(cell_idx=1, dwell_ticks=2, wait_ticks=1),
            Visit(cell_idx=2, dwell_ticks=2, wait_ticks=0)
        ])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )

    feasible, reason = checker.check_joint_schedule(sched)
    assert feasible is True
    assert reason == "all_feasible"
