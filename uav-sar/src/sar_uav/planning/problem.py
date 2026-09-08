"""PlanningProblem: everything a planner is allowed to know.

Bundles the area, fleet specs, the planner's *estimated* motion model,
detection configuration and a weather accessor -- i.e. exactly the
information set I_t of the proposal, never the hidden target state.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..detection.sensors import q_map_for_sensor


class PlanningProblem:
    def __init__(self, area, fleet: List, motion_est,
                 dwell_ticks: int = 3,
                 weather_fn: Callable[[float], Dict] = lambda t: {},
                 dt_min: float = 1.0):
        self.area = area
        self.fleet = fleet                      # list[UAVSpec]
        self.motion_est = motion_est            # belief/motion.MotionModel
        self.dwell_ticks = int(dwell_ticks)
        self.weather_fn = weather_fn            # mission-tick -> weather dict
        self.dt_min = float(dt_min)
        self.n = area.n_cells

        self.xy = area.centers_xy()
        self.water_flat = area.water.reshape(-1)
        self.land = area.land_idx()

    # ------------------------------------------------------------------
    def q_matrix(self, t: float) -> np.ndarray:
        """(K, n) per-UAV per-cell *dwell* detection probabilities at tick t."""
        wmean = self.weather_fn(t)
        K = len(self.fleet)
        Q = np.zeros((K, self.n))
        for k, spec in enumerate(self.fleet):
            Q[k] = q_map_for_sensor(spec.sensor, self.area, wmean,
                                    self.dwell_ticks)
        return Q

    def value_matrix(self, belief: np.ndarray, t: float) -> np.ndarray:
        """V_ikt = p_i^t * q_ikt (proposal objective integrand)."""
        return self.q_matrix(t) * belief[None, :]

    def travel_min(self, k: int, from_xy, cell: int) -> float:
        d = float(np.hypot(*(self.xy[cell] - np.asarray(from_xy, float))))
        return d / self.fleet[k].speed_mpm
