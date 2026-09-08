"""Cycle-based planners: static multi-sortie routing and rolling-horizon.

Both build **sortie cycles** ``[(cell, dwell), ...]`` with a greedy
insertion heuristic whose marginal gain accounts for

* target-motion uncertainty -- the belief is diffused forward by the
  estimated motion model, so visiting a cell later yields
  ``p_hat(arrival_time) * q`` rather than today's ``p * q``
  (arrival-time discounted value);
* travel / energy cost (lambda penalties);
* search overlap (kappa decay on repeated visits).

``StaticPlanner``  plans the whole remaining horizon once at mission start
                   and executes it open-loop (across battery swaps).
``RollingPlanner`` rebuilds short plans (``lookahead_cells``) every time a
                   UAV is idle -- i.e. replans on the *current* posterior.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import (BasePlanner, Cycle, cell_xy, leg_metrics, register_planner,
                   best_value_cell)


class CyclePlanner(BasePlanner):
    def __init__(self, problem, top_k: int = 48,
                 lambda_travel_km: float = 0.002,
                 lambda_energy: float = 0.0,
                 overlap_kappa: float = 0.7,
                 lookahead_cells: int = 3,
                 gain_bucket_min: float = 5.0,
                 force_dispatch: bool = True,
                 **_ignored):
        super().__init__(problem)
        self.top_k = int(top_k)
        self.l_travel = float(lambda_travel_km)
        self.l_energy = float(lambda_energy)
        self.kappa = float(overlap_kappa)
        self.lookahead = int(lookahead_cells)
        self.bucket = float(gain_bucket_min)
        self.force_dispatch = bool(force_dispatch)

    # ------------------------------------------------------------ helpers
    def _buckets(self, p0: np.ndarray, t_now: float, horizon_left: float):
        """Diffused beliefs p_hat at each arrival-time bucket."""
        B = max(1, int(math.ceil(max(horizon_left, self.bucket) / self.bucket)))
        ps = [np.asarray(p0, float)]
        for b in range(B):
            rain = self.pb.weather_fn(t_now + b * self.bucket).get("rain", 0.0)
            ps.append(self.pb.motion_est.forecast(ps[-1], rain))
        return ps

    def _profile(self, spec, start_xy, base_xy, cycle: Cycle,
                 battery: float, horizon_left: float):
        """Full sortie metrics: feasibility, energy margin, minutes, etas."""
        e = battery
        minutes = 0.0
        pos = np.asarray(start_xy, float)
        etas: List[float] = []
        dist_total = 0.0
        for (cell, dwell_ticks) in cycle:
            wp = cell_xy(self.pb.area, cell)
            d, asc = leg_metrics(spec, self.pb.area, pos, wp)
            leg_min = d / spec.speed_mpm
            e -= spec.fly_cost(d, asc)
            dist_total += d
            minutes += leg_min
            if e < 0:
                return {"ok": False}
            etas.append(minutes)
            e -= spec.hover_cost(dwell_ticks * self.pb.dt_min)
            minutes += dwell_ticks * self.pb.dt_min
            pos = wp
        d, asc = leg_metrics(spec, self.pb.area, pos, base_xy)
        e -= spec.fly_cost(d, asc)
        dist_total += d
        minutes += d / spec.speed_mpm
        ok = e >= spec.reserve_frac * spec.battery_wh \
            and minutes <= horizon_left
        return {"ok": ok, "margin": e, "minutes": minutes,
                "etas": etas, "dist": dist_total}

    def _gain_at(self, k, cell, eta, buckets, qmaps, visits, plan_visits):
        b = min(int(eta / self.bucket), len(buckets) - 1)
        v = float(buckets[b][cell]) * float(qmaps[b][k][cell])
        n_visit = visits.get(cell, 0) + plan_visits.get(cell, 0)
        return v * (self.kappa ** n_visit)

    # ---------------------------------------------------------- planning
    def plan_for_uav(self, k: int, st, belief: np.ndarray, t_now: float,
                     horizon_left: float, max_cells: int) -> List[Cycle]:
        pb = self.pb
        spec = pb.fleet[k]
        buckets = self._buckets(belief, t_now, horizon_left)
        qmaps = []
        for b in range(len(buckets)):
            tb = t_now + min(b * self.bucket, horizon_left)
            qmaps.append(pb.q_matrix(tb))

        # ---- candidate shortlist: mean early-horizon value, overlap-decayed
        nb_early = max(1, len(buckets) // 3)
        visits = getattr(st, "visits", {})
        early = np.zeros(pb.n)
        for b in range(nb_early):
            early += qmaps[b][k] * buckets[b]
        early /= nb_early
        early[pb.water_flat] = -np.inf
        cand = [int(c) for c in np.argsort(-early)[: self.top_k]
                if np.isfinite(early[c])]

        cycles: List[List[Tuple[int, int]]] = []
        plan_visits: Dict[int, int] = {}

        def try_place(c: int):
            """Best feasible insertion of cell c anywhere; None if none."""
            best = None
            options = []
            for ci, cyc in enumerate(cycles):
                for j in range(len(cyc) + 1):
                    options.append((ci, j))
            options.append((len(cycles), 0))          # brand-new sortie
            for (ci, j) in options:
                new_cyc = ([list(x) for x in cycles[ci]] if ci < len(cycles) else [])
                new_cyc = new_cyc[:j] + [[c, pb.dwell_ticks]] + new_cyc[j:]
                new_cyc = [tuple(x) for x in new_cyc]
                prof_new = self._profile(spec, st.pos, st.base_xy, new_cyc,
                                         st.battery, horizon_left)
                if not prof_new["ok"]:
                    continue
                old = self._profile(spec, st.pos, st.base_xy,
                                    [tuple(x) for x in cycles[ci]],
                                    st.battery, horizon_left) \
                    if ci < len(cycles) else {"dist": 0.0, "ok": True}
                eta = prof_new["etas"][j]
                g = self._gain_at(k, c, eta, buckets, qmaps,
                                  visits, plan_visits)
                d_km = (prof_new["dist"] - old.get("dist", 0.0)) / 1000.0
                d_e_norm = (old.get("margin", spec.battery_wh)
                            - prof_new["margin"]) / spec.battery_wh
                # penalties rank placements; the gate is on raw gain only,
                # so diffuse priors still produce full search plans
                if g <= 1e-12:
                    continue
                net = g - self.l_travel * d_km - self.l_energy * d_e_norm
                if best is None or net > best[0]:
                    best = (net, ci, j)
            if best is None:
                return False
            _, ci, j = best
            if ci == len(cycles):
                cycles.append([(c, pb.dwell_ticks)])
            else:
                cycles[ci].insert(j, (c, pb.dwell_ticks))
            plan_visits[c] = plan_visits.get(c, 0) + 1
            return True

        placed = 0
        progress = True
        while placed < max_cells and progress and cand:
            progress = False
            # one sweep tries every remaining candidate once
            for c in list(cand):
                if placed >= max_cells:
                    break
                if try_place(c):
                    cand.remove(c)
                    placed += 1
                    progress = True
        if not cycles and self.force_dispatch and horizon_left > 10.0:
            c_fb = best_value_cell(pb, k, st, belief,
                                   top_k=self.top_k, overlap_kappa=self.kappa)
            if c_fb is not None:
                cycles = [[(c_fb, pb.dwell_ticks)]]
        return [cy for cy in cycles if cy]


@register_planner
class StaticPlanner(CyclePlanner):
    """One-shot full-horizon plan at mission start; executed open-loop."""
    kind = "static"

    def __init__(self, problem, **kw):
        super().__init__(problem, **kw)
        self._dispatched = set()

    def reset(self):
        self._dispatched = set()

    def decide(self, states, belief, t, horizon_left):
        out: Dict[str, List[Cycle]] = {}
        for k, spec in enumerate(self.pb.fleet):
            st = states.get(spec.name)
            if st is None or not st.is_idle or spec.name in self._dispatched:
                continue
            cyc = self.plan_for_uav(k, st, belief, t, horizon_left,
                                    max_cells=10_000)
            self._dispatched.add(spec.name)
            if cyc:
                out[spec.name] = cyc
        return out


@register_planner
class RollingPlanner(CyclePlanner):
    """Rolling horizon: rebuild short plans on current posterior whenever
    a UAV is idle (post-battery-swap or between sorties)."""
    kind = "rolling"

    def __init__(self, problem, **kw):
        super().__init__(problem, **kw)

    def decide(self, states, belief, t, horizon_left):
        out: Dict[str, List[Cycle]] = {}
        for k, spec in enumerate(self.pb.fleet):
            st = states.get(spec.name)
            if st is None or not st.is_idle:
                continue
            cyc = self.plan_for_uav(k, st, belief, t, horizon_left,
                                    max_cells=self.lookahead)
            if cyc:
                out[spec.name] = cyc
        return out
