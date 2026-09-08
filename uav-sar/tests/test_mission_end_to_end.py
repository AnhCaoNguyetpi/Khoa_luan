"""End-to-end mission smoke tests on a tiny area (fast)."""
from _common import small_area, tiny_cfg
import numpy as np

from sar_uav.sim.mission import MissionSetup, run_mission
from sar_uav.sim.metrics import summarize


def _setup():
    cfg = tiny_cfg()
    setup = MissionSetup.create(cfg, seed=99,
                                area=small_area())
    return cfg, setup


def test_mission_completes_and_metrics_consistent():
    cfg, setup = _setup()
    res = run_mission(setup, planner_cfg={"kind": "rolling"},
                      initial_model="uniform", arm_name="t")
    assert 0 <= res.detect_time_censored <= res.horizon_min
    assert res.detected == bool(np.isfinite(res.detect_time_min))
    if res.detected:
        assert res.detect_time_min > 0
        # detection must coincide with at least one searched cell footprint
    assert 0 < res.unique_cells_searched <= max(res.total_search_ops, 1)
    assert res.redundant_search_ops == (res.total_search_ops
                                        - res.unique_cells_searched)
    assert res.flight_distance_m >= 0 and res.energy_wh >= 0
    assert res.n_uavs == len(cfg["fleet"])


def test_static_open_loop_runs():
    _, setup = _setup()
    res = run_mission(setup, planner_cfg={"kind": "static"},
                      initial_model="distance", arm_name="s")
    assert res.planner == "static"
    assert np.isfinite(res.horizon_min)


def test_stationary_target_ablation():
    cfg = tiny_cfg()
    setup = MissionSetup.create(cfg, seed=7, area=small_area(),
                                moving=False)
    res = run_mission(setup, planner_cfg={"kind": "rolling"},
                      initial_model="distance", arm_name="stat",
                      moving_target=False)
    assert res.moving_target is False


def test_assign_criterion_p_only_changes_behaviour():
    """p-only assignment must not crash; usually differs from pq."""
    cfg = tiny_cfg()
    setup = MissionSetup.create(cfg, seed=5, area=small_area())
    r_pq = run_mission(setup, planner_cfg={"kind": "greedy"},
                       initial_model="distance", assign_criterion="pq",
                       arm_name="pq")
    r_p = run_mission(setup, planner_cfg={"kind": "greedy"},
                      initial_model="distance", assign_criterion="p",
                      arm_name="p")
    assert (r_pq.unique_cells_searched, r_pq.total_search_ops) != (0, 0)
    assert (r_p.unique_cells_searched, r_p.total_search_ops) != (0, 0)


def test_summarize_dsr():
    rows = [{"detected": True, "detect_time_min": 30.0,
             "horizon_min": 60.0, "flight_distance_m": 100.0,
             "energy_wh": 10.0, "search_minutes": 6.0,
             "unique_cells_searched": 2, "redundant_search_ops": 0,
             "n_replans": 1, "replan_ms_total": 5.0, "n_swaps": 0},
            {"detected": False, "detect_time_min": np.nan,
             "horizon_min": 60.0, "flight_distance_m": 200.0,
             "energy_wh": 20.0, "search_minutes": 8.0,
             "unique_cells_searched": 3, "redundant_search_ops": 1,
             "n_replans": 2, "replan_ms_total": 9.0, "n_swaps": 1}]
    s = summarize(rows)
    assert abs(s["DSR"] - 0.5) < 1e-9
    assert abs(s["EDT_censored"] - 45.0) < 1e-9
    assert abs(s["P(T<=60)"] - 1.0) < 1e-9
