"""Unit tests for the 8 neighborhood operators."""

import numpy as np
import pytest

from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.neighborhood import NeighborhoodExplorer, detect_affected_interval


@pytest.fixture
def neighborhood_fixture():
    num_cells = 9
    H = 15
    delta_t = 60.0
    depot_idx = 0

    dist_m = np.zeros((9, 9))
    coords = [(r, c) for r in range(3) for c in range(3)]
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 200.0

    specs = [
        UAVSpec(name="uav_0", speed_ms=12.0, battery_joules=300000.0, reserve_joules=25000.0, delta_t=delta_t),
        UAVSpec(name="uav_1", speed_ms=12.0, battery_joules=300000.0, reserve_joules=25000.0, delta_t=delta_t)
    ]
    checker = ConstraintChecker(dist_m, depot_idx, specs, H, delta_t=delta_t)
    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
    explorer = NeighborhoodExplorer(checker, candidates, allowed_dwells=(1, 2, 3, 4))
    return checker, explorer, H, num_cells, delta_t


def test_replace_visit_operator(neighborhood_fixture):
    checker, explorer, H, num_cells, delta_t = neighborhood_fixture
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=2)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )
    replaces = list(explorer.gen_replace_visits(sched))
    assert len(replaces) > 0
    # Every replacement must have changed cell_idx from 1
    for cand in replaces:
        assert cand.uav_schedules[0].visits[0].cell_idx != 1


def test_dwell_rebalance_operator(neighborhood_fixture):
    checker, explorer, H, num_cells, delta_t = neighborhood_fixture
    sched = JointSchedule(
        uav_schedules=[
            UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=3), Visit(cell_idx=2, dwell_ticks=2)])
        ],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )
    rebalances = list(explorer.gen_dwell_rebalance(sched))
    assert len(rebalances) > 0
    # Sum of dwell ticks across visits must be conserved
    orig_total_dwell = sum(v.dwell_ticks for v in sched.uav_schedules[0].visits)
    for cand in rebalances:
        cand_total_dwell = sum(v.dwell_ticks for v in cand.uav_schedules[0].visits)
        assert cand_total_dwell == orig_total_dwell


def test_insert_visit_supports_revisit(neighborhood_fixture):
    checker, explorer, H, num_cells, delta_t = neighborhood_fixture
    # Initial schedule already has cell 4
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2)])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )
    inserts = list(explorer.gen_insert_visits(sched))
    # Check that cell 4 can be inserted again (revisit)
    revisit_cands = [c for c in inserts if sum(1 for v in c.uav_schedules[0].visits if v.cell_idx == 4) >= 2]
    assert len(revisit_cands) > 0, "Insert visit must support revisit of already-searched cells"


def test_detect_affected_interval(neighborhood_fixture):
    checker, explorer, H, num_cells, delta_t = neighborhood_fixture
    # Schedule with 3 visits: changing dwell of visit 1 shifts visit 2
    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[
            Visit(cell_idx=1, dwell_ticks=2),
            Visit(cell_idx=2, dwell_ticks=2),
            Visit(cell_idx=3, dwell_ticks=2)
        ])],
        num_cells=num_cells,
        H=H,
        delta_t=delta_t
    )
    checker.compute_and_update_chaining(sched.uav_schedules[0])

    # Modify visit 1 dwell (shifts visit 2 start time and subsequent schedule)
    cand = sched.clone()
    cand.uav_schedules[0].visits[1].dwell_ticks = 3
    checker.compute_and_update_chaining(cand.uav_schedules[0])

    t_a, t_b = detect_affected_interval(sched, cand)
    assert t_a > 0
    assert t_b == H - 1  # trailing visits or end times shifted
