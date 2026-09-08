"""Planner interface behaviour and feasibility of produced assignments."""
from _common import small_area
import numpy as np

from sar_uav.belief.motion import MotionModel, PROFILES
from sar_uav.planning import PlanningProblem, make_planner
from sar_uav.uav.platform import UAVSpec
from sar_uav.uav.state import UAVState


def _problem(area):
    fleet = [UAVSpec(name="u-rgb", sensor="rgb"),
             UAVSpec(name="u-th", sensor="thermal")]
    mm = MotionModel(area, PROFILES["hiker"])
    pb = PlanningProblem(area, fleet, mm, dwell_ticks=2,
                         weather_fn=lambda t: {"cloud": .2, "rain": 0.,
                                               "wind_ms": 3., "temp_c": 20.})
    return pb


def _states(pb, area):
    land = area.land_idx()
    out = {}
    for i, spec in enumerate(pb.fleet):
        out[spec.name] = UAVState(spec, area,
                                  area.centers_xy()[int(land[i * 7])])
    return out


def _check_assignments(routes, pb, states, horizon_left):
    for name, cycles in routes.items():
        st = states[name]
        assert st.is_idle
        assert len(cycles) >= 1
        for cy in cycles:
            assert all(not pb.water_flat[c] for c, _ in cy)


def test_greedy_dispatches_idle_uavs():
    area = small_area()
    pb = _problem(area)
    states = _states(pb, area)
    belief = np.full(area.n_cells, 1.0 / area.n_cells)
    pl = make_planner({"kind": "greedy"}, pb)
    routes = pl.decide(states, belief, t=0.0, horizon_left=90.0)
    assert set(routes) == {"u-rgb", "u-th"}
    _check_assignments(routes, pb, states, 90.0)


def test_rolling_builds_multi_cell_cycles():
    area = small_area()
    pb = _problem(area)
    states = _states(pb, area)
    belief = np.full(area.n_cells, 1.0 / area.n_cells)
    pl = make_planner({"kind": "rolling", "lookahead_cells": 3}, pb)
    routes = pl.decide(states, belief, t=0.0, horizon_left=90.0)
    assert routes
    total_cells = sum(len(cy) for cys in routes.values() for cy in cys)
    assert 1 <= total_cells <= 6


def test_static_plans_only_once():
    area = small_area()
    pb = _problem(area)
    states = _states(pb, area)
    belief = np.full(area.n_cells, 1.0 / area.n_cells)
    pl = make_planner({"kind": "static"}, pb)
    first = pl.decide(states, belief, 0.0, 120.0)
    second = pl.decide(states, belief, 10.0, 110.0)
    assert first and second == {}


def test_milp_fallback_without_pulp():
    area = small_area()
    pb = _problem(area)
    try:
        import pulp          # noqa: F401
        have_pulp = True
    except ImportError:
        have_pulp = False
    states = _states(pb, area)
    belief = np.full(area.n_cells, 1.0 / area.n_cells)
    pl = make_planner({"kind": "milp"}, pb)
    routes = pl.decide(states, belief, 0.0, 60.0)
    if have_pulp:
        assert isinstance(routes, dict)
    else:
        # falls back to rolling behaviour: dispatches idle uavs
        assert set(routes) >= {"u-rgb"}
