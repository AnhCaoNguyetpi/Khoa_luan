"""Greedy dispatch planners (single-cell assignment on current belief).

* ``greedy``       -- argmax_i [ p_i q_ki * decay - travel/energy penalties ]
* ``greedy_ratio`` -- argmax_i [ expected detections per sortie-minute ]

Both are myopic baselines; they dispatch idle UAVs every decision tick.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import (BasePlanner, Cycle, cell_xy, cycle_feasibility,
                   register_planner, best_value_cell)


class _GreedyBase(BasePlanner):
    def __init__(self, problem, top_k: int = 48,
                 lambda_travel_km: float = 0.002, lambda_energy: float = 0.0,
                 overlap_kappa: float = 0.7, force_dispatch: bool = True,
                 **_ignored):
        super().__init__(problem)
        self.top_k = int(top_k)
        self.l_travel = float(lambda_travel_km)
        self.l_energy = float(lambda_energy)
        self.kappa = float(overlap_kappa)
        self.force_dispatch = bool(force_dispatch)

    def _score(self, g, dist_km, energy_wh_norm, sortie_min):
        raise NotImplementedError

    def decide(self, states, belief, t, horizon_left):
        pb = self.pb
        out: Dict[str, List[Cycle]] = {}
        if horizon_left <= 0:
            return out
        V = pb.value_matrix(belief, t)          # (K, n) = p_i^t q_ikt
        for k, spec in enumerate(pb.fleet):
            st = states.get(spec.name)
            if st is None or not st.is_idle:
                continue
            visits = st.visits
            decay = np.array([self.kappa ** visits.get(i, 0)
                              for i in range(pb.n)])
            gain = V[k] * decay
            gain[pb.water_flat] = -np.inf
            cand = np.argsort(-gain)[: self.top_k]
            best_cell, best_net = None, -np.inf
            for c in cand:
                if not np.isfinite(gain[c]) or gain[c] <= 1e-12:
                    continue
                xy_c = cell_xy(pb.area, c)
                d = float(np.hypot(float(xy_c[0] - st.pos[0]),
                                   float(xy_c[1] - st.pos[1])))
                e_add = spec.fly_cost(2.0 * d) \
                    + spec.hover_cost(pb.dwell_ticks * pb.dt_min)
                sortie_min = 2.0 * d / spec.speed_mpm \
                    + pb.dwell_ticks * pb.dt_min
                feas, _, _ = cycle_feasibility(
                    spec, pb.area, st.pos, st.base_xy,
                    [(int(c), pb.dwell_ticks)],
                    dwell_scale_min=pb.dt_min, horizon_left=horizon_left)
                if not feas:
                    continue
                # penalties RANK candidates but do not gate: an idle UAV
                # searches the best available area regardless
                net = self._score(float(gain[c]), d / 1000.0,
                                  e_add / spec.battery_wh, sortie_min)
                if net > best_net:
                    best_net, best_cell = net, int(c)
            if best_cell is None and self.force_dispatch and horizon_left > 10.0:
                # never park a capable UAV while valuable cells remain
                c_fb = best_value_cell(pb, k, st, belief,
                                       top_k=self.top_k, overlap_kappa=self.kappa)
                if c_fb is not None:
                    best_cell = c_fb
            if best_cell is not None:
                out[spec.name] = [[(best_cell, pb.dwell_ticks)]]
        return out


@register_planner
class GreedyPlanner(_GreedyBase):
    kind = "greedy"

    def _score(self, g, dist_km, energy_wh_norm, sortie_min):
        return g - self.l_travel * dist_km - self.l_energy * energy_wh_norm


@register_planner
class GreedyRatioPlanner(_GreedyBase):
    kind = "greedy_ratio"

    def _score(self, g, dist_km, energy_wh_norm, sortie_min):
        return g / max(sortie_min, 1e-9)
