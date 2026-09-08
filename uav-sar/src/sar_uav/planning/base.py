"""Shared planning utilities and the unified planner interface.

Every planner implements::

    decide(states, belief, t, horizon_left) -> {uav_name: [cycle, ...]}

where ``cycle = [(cell_idx, dwell_ticks), ...]`` is one battery sortie.
Planners only ever see the *belief* ``P^t`` -- never the hidden true target
position (proposal box: "Vi tri thuc != belief cua planner").
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

Cycle = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# geometry / feasibility helpers
# ---------------------------------------------------------------------------

def cell_xy(area, idx: int) -> np.ndarray:
    r, c = divmod(int(idx), area.W)
    return np.array([(c + 0.5) * area.cell, (r + 0.5) * area.cell])


def leg_metrics(spec, area, xy_from, xy_to) -> Tuple[float, float]:
    """(distance_m, ascent_m) of a straight leg."""
    d = float(np.hypot(*(np.asarray(xy_to, float) - np.asarray(xy_from, float))))
    asc = max(0.0, area.elev_at_xy(xy_to) - area.elev_at_xy(xy_from))
    return d, asc


def cycle_feasibility(spec, area, start_xy, base_xy, cycle: Cycle,
                      dwell_scale_min: float = 1.0,
                      horizon_left: Optional[float] = None):
    """Evaluate one sortie from ``start_xy`` back to ``base_xy``.

    Returns ``(feasible, energy_needed, minutes)`` where ``energy_needed``
    includes the mandatory return leg and the battery reserve.
    """
    e = spec.battery_wh
    minutes = 0.0
    pos = np.asarray(start_xy, float)
    for (cell, dwell_ticks) in cycle:
        wp = cell_xy(area, cell)
        d, asc = leg_metrics(spec, area, pos, wp)
        e -= spec.fly_cost(d, asc)
        e -= spec.hover_cost(dwell_ticks * dwell_scale_min)
        minutes += d / spec.speed_mpm + dwell_ticks * dwell_scale_min
        if e < spec.reserve_frac * spec.battery_wh:
            return False, e, minutes
        pos = wp
    d, asc = leg_metrics(spec, area, pos, base_xy)
    e -= spec.fly_cost(d, asc)
    minutes += d / spec.speed_mpm
    ok = e >= spec.reserve_frac * spec.battery_wh
    if horizon_left is not None and minutes > horizon_left:
        ok = False
    return ok, e, minutes


def best_value_cell(problem, k: int, st, belief: np.ndarray,
                    top_k: int = 48, overlap_kappa: float = 0.7):
    """Highest-value feasible single cell for UAV ``k`` (fallback dispatch).

    Used when the planner's net-gain threshold rejects everything: in real
    SAR an idle UAV is re-tasked to the best available area rather than
    parked.  Returns ``None`` when nothing feasible remains.
    """
    V = problem.value_matrix(belief, 0.0)[k]
    visits = getattr(st, "visits", {})
    decay = np.array([overlap_kappa ** visits.get(i, 0)
                      for i in range(problem.n)])
    gain = V * decay
    gain[problem.water_flat] = -np.inf
    order = np.argsort(-gain)[: max(top_k, 8)]
    for c in order:
        c = int(c)
        if not np.isfinite(gain[c]):
            continue
        feas, _, _ = cycle_feasibility(
            problem.fleet[k], problem.area, st.pos, st.base_xy,
            [(c, problem.dwell_ticks)],
            dwell_scale_min=problem.dt_min)
        if feas:
            return c
    return None


# ---------------------------------------------------------------------------
# planner interface
# ---------------------------------------------------------------------------

class BasePlanner:
    """Unified decision interface used by MissionRunner."""

    kind = "base"

    def __init__(self, problem: "PlanningProblem"):
        self.pb = problem

    def reset(self) -> None:
        pass

    def decide(self, states: Dict[str, "UAVState"], belief: np.ndarray,
               t: float, horizon_left: float) -> Dict[str, List[Cycle]]:
        raise NotImplementedError


PLANNER_REGISTRY = {}


def register_planner(cls):
    PLANNER_REGISTRY[cls.kind] = cls
    return cls


def make_planner(cfg: Dict, problem: "PlanningProblem") -> BasePlanner:
    kind = cfg.get("kind", "rolling")
    if kind not in PLANNER_REGISTRY:
        raise ValueError(f"unknown planner '{kind}'; "
                         f"available: {sorted(PLANNER_REGISTRY)}")
    return PLANNER_REGISTRY[kind](problem, **{k: v for k, v in cfg.items()
                                              if k != "kind"})
