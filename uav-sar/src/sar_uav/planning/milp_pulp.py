"""Optional MILP allocation planner (requires ``pulp``).

Formulates one replanning period as a team orienteering *allocation* problem::

    max  sum_{k,i} p_i^t q_ikt z_ki  -  lambda_1 sum_{k,i} d(base_k,i) z_ki
    s.t. sum_k z_ki <= 1                       (no duplicated assignment)
         sum_i tau_ki z_ki <= T_budget_k       (sortie time budget)
         z_ki in {0,1}

and orders each UAV's cells nearest-neighbour from its base, splitting them
into feasible sortie cycles.  Ordering is heuristic -- this MILP is a
compact, honest baseline for small instances, not a full VRP formulation
(full routing MILP / column generation is future work).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from .base import BasePlanner, Cycle, cell_xy, cycle_feasibility, register_planner

log = logging.getLogger(__name__)


@register_planner
class MilpPlanner(BasePlanner):
    kind = "milp"

    def __init__(self, problem, top_k: int = 48, milp_time_limit_s: float = 10.0,
                 lambda_travel_km: float = 0.002, lookahead_cells: int = 6,
                 overlap_kappa: float = 0.7, **_ignored):
        super().__init__(problem)
        self.top_k = int(top_k)
        self.time_limit = float(milp_time_limit_s)
        self.l_travel = float(lambda_travel_km)
        self.lookahead = int(lookahead_cells)
        self.kappa = float(overlap_kappa)

    def decide(self, states, belief, t, horizon_left):
        try:
            import pulp
        except ImportError:
            log.warning("pulp not installed -> MilpPlanner falls back to "
                        "'rolling'. pip install pulp")
            from .cycle_planners import RollingPlanner
            if not hasattr(self, "_fallback"):
                self._fallback = RollingPlanner(
                    self.pb, top_k=self.top_k,
                    lambda_travel_km=self.l_travel,
                    lookahead_cells=self.lookahead,
                    overlap_kappa=self.kappa)
            return self._fallback.decide(states, belief, t, horizon_left)

        pb = self.pb
        out: Dict[str, List[Cycle]] = {}
        Q = pb.q_matrix(t)                                   # (K, n)

        for k, spec in enumerate(pb.fleet):
            st = states.get(spec.name)
            if st is None or not st.is_idle:
                continue

            visits = st.visits
            decay = np.array([self.kappa ** visits.get(i, 0) for i in range(pb.n)])
            val = Q[k] * np.asarray(belief) * decay
            val[pb.water_flat] = -np.inf
            cand = [int(c) for c in np.argsort(-val)[: self.top_k]
                    if np.isfinite(val[c])]
            if not cand:
                continue

            # time budget: outbound+return+dwell must fit horizon & battery
            budget = min(horizon_left,
                         spec.endurance_min() * (1 - spec.reserve_frac))
            prob = pulp.LpProblem("alloc", pulp.LpMaximize)
            z = {c: pulp.LpVariable(f"z_{c}", cat="Binary") for c in cand}
            d_km = {c: float(np.hypot(*(cell_xy(pb.area, c) - st.pos))) / 1000.0
                    for c in cand}
            tau = {c: 2.0 * d_km[c] * 1000.0 / spec.speed_mpm
                   + pb.dwell_ticks * pb.dt_min for c in cand}
            prob += pulp.lpSum((val[c] - self.l_travel * d_km[c]) * z[c]
                               for c in cand)
            prob += pulp.lpSum(tau[c] * z[c] for c in cand) <= budget
            prob += pulp.lpSum(z.values()) <= self.lookahead
            try:
                prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit))
            except Exception as e:                           # pragma: no cover
                log.warning("MILP solve failed (%s); skipping dispatch", e)
                continue
            chosen = [c for c in cand if z[c].value() and z[c].value() > 0.5]
            if not chosen:
                continue

            # nearest-neighbour ordering from current position
            order, pos = [], st.pos.copy()
            pool = list(chosen)
            while pool:
                dists = [float(np.hypot(*(cell_xy(pb.area, c) - pos))) for c in pool]
                nxt = pool[int(np.argmin(dists))]
                order.append(nxt)
                pos = cell_xy(pb.area, nxt)
                pool.remove(nxt)

            # split into feasible cycles (greedy append)
            cycles: List[Cycle] = []
            cur: Cycle = []
            for c in order:
                trial = cur + [(c, pb.dwell_ticks)]
                feas, _, mins = cycle_feasibility(
                    spec, pb.area, st.pos, st.base_xy, trial,
                    dwell_scale_min=pb.dt_min, horizon_left=horizon_left)
                if feas:
                    cur = trial
                else:
                    if cur:
                        cycles.append(cur)
                    cur = [(c, pb.dwell_ticks)]
                    feas2, _, _ = cycle_feasibility(
                        spec, pb.area, st.pos, st.base_xy, cur,
                        dwell_scale_min=pb.dt_min, horizon_left=horizon_left)
                    if not feas2:
                        cur = []
            if cur:
                cycles.append(cur)
            if cycles:
                out[spec.name] = cycles
        return out
