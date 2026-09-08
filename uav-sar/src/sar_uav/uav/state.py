"""Per-UAV state machine and tick dynamics.

Modes: ``idle`` -> (assigned a list of *cycles*) -> ``fly`` / ``search``
alternation -> automatic RTB when the battery reserve rule would be
violated -> ``swap`` at base -> next cycle or ``idle``.

A **cycle** is ``[(cell, dwell_ticks), ...]``: one battery sortie that is
guaranteed feasible (energy + return leg) by the planner.  Executing cycles
sequentially keeps open-loop static plans valid across battery swaps.

Actions per proposal: {move, search, wait, return}; ``wait`` appears as an
explicit idle/hold and ``return`` as the reserve-triggered RTB.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class UAVState:
    def __init__(self, spec, area, base_xy):
        self.spec = spec
        self.area = area
        self.base_xy = np.asarray(base_xy, dtype=float)
        self.pos = self.base_xy.copy()

        self.battery = float(spec.battery_wh)
        self.mode = "idle"          # idle|fly|search|rtb|swap|done
        self.target_cell: Optional[int] = None
        self.dwell_left = 0
        self.swap_left = 0.0

        self._cycles: Deque[List[Tuple[int, int]]] = deque()
        self._cur_cycle: Optional[List[Tuple[int, int]]] = None
        self._cycle_idx = 0
        self._leg_from = self.pos.copy()
        self._leg_progress = 0.0    # metres flown on current leg
        self._leg_len = 0.0

        # statistics
        self.dist_flown = 0.0
        self.energy_spent = 0.0
        self.search_minutes = 0.0
        self.n_swaps = 0
        self.visits: Dict[int, int] = {}   # cell -> times searched

    # ------------------------------------------------------------ helpers
    @property
    def is_active(self) -> bool:
        return self.mode != "done"

    @property
    def is_idle(self) -> bool:
        return self.mode == "idle"

    @property
    def at_base(self) -> bool:
        return bool(np.allclose(self.pos, self.base_xy, atol=1e-6))

    def return_cost(self, xy=None) -> float:
        p = self.pos if xy is None else np.asarray(xy, dtype=float)
        d = float(np.hypot(*(p - self.base_xy)))
        asc = max(0.0, self.area.elev_at_xy(self.base_xy) - self.area.elev_at_xy(p))
        return self.spec.fly_cost(d, asc)

    def reserve_needed(self) -> float:
        return self.spec.reserve_frac * self.spec.battery_wh + self.return_cost()

    # ---------------------------------------------------------- assignment
    def assign_cycles(self, cycles: List[List[Tuple[int, int]]]) -> None:
        """Queue new sorties; only allowed while idle at base."""
        if not self.is_idle:
            raise RuntimeError(f"{self.spec.name}: cannot assign in mode '{self.mode}'")
        for cy in cycles:
            if cy:
                self._cycles.append(list(cy))

    def has_pending_work(self) -> bool:
        return bool(self._cycles) or self.mode in ("fly", "search", "rtb", "swap")

    # --------------------------------------------------------------- tick
    def _start_next_cycle_or_idle(self) -> Dict:
        """Called idle at base: pop next queued cycle or remain idle/done."""
        while self._cycles:
            cyc = self._cycles[0]
            if self._cycle_feasible_now(cyc):
                self._cycles.popleft()
                self._cur_cycle = cyc
                self._cycle_idx = 0
                self._begin_leg_to_cell(cyc[0][0])
                return {}
            else:
                self._cycles.popleft()      # stale/infeasible -> drop
                continue
        self._cur_cycle = None
        return {}

    def _cycle_feasible_now(self, cyc) -> bool:
        """Battery+time feasibility of executing cycle from current position."""
        s = self.spec
        pos = self.pos
        e = self.battery
        for (cell, dwell) in cyc:
            wp = self._cell_xy(cell)
            d = float(np.hypot(*(wp - pos)))
            asc = max(0.0, self.area.elev_at_xy(wp) - self.area.elev_at_xy(pos))
            e -= s.fly_cost(d, asc)
            e -= s.hover_cost(dwell * 1.0)   # dt=1 min ticks
            if e < s.reserve_frac * s.battery_wh:
                return False
            pos = wp
        home_d = float(np.hypot(*(self.base_xy - pos)))
        asc = max(0.0, self.area.elev_at_xy(self.base_xy) - self.area.elev_at_xy(pos))
        e -= s.fly_cost(home_d, asc)
        return e >= s.reserve_frac * s.battery_wh

    def _cell_xy(self, idx: int) -> np.ndarray:
        r, c = divmod(int(idx), self.area.W)
        return np.array([(c + 0.5) * self.area.cell, (r + 0.5) * self.area.cell])

    def _begin_leg_to_cell(self, cell: int) -> None:
        self.target_cell = int(cell)
        self._leg_from = self.pos.copy()
        self._leg_len = float(np.hypot(*(self._cell_xy(cell) - self.pos)))
        self._leg_progress = 0.0
        self.mode = "fly"

    def _fly_step(self, dt_min: float, toward_xy: np.ndarray,
                  guard: bool = True) -> Dict:
        """Advance one cruise tick toward ``toward_xy`` with energy accounting.
        Returns {} normally, {'forced_rtb': True} if reserve would break.
        ``guard=False`` while already returning to base (never self-trigger)."""
        s = self.spec
        seg_len = s.speed_mpm * dt_min
        remaining = float(np.hypot(*(toward_xy - self.pos)))

        # reserve guard: after this leg must still fly home from the waypoint
        if guard and remaining > seg_len:
            nxt = self.pos + (toward_xy - self.pos) / remaining * seg_len
            if self.battery - s.fly_cost(seg_len, 0.0) < (
                    s.reserve_frac * s.battery_wh + self.return_cost(nxt)):
                self.mode = "rtb"
                self.target_cell = None
                self._leg_from = self.pos.copy()
                self._leg_len = float(np.hypot(*(self.base_xy - self.pos)))
                self._leg_progress = 0.0
                return {"forced_rtb": True}

        step = min(seg_len, remaining)
        frac = step / max(remaining, 1e-9)
        x0, x1 = self.pos, self.pos + (toward_xy - self.pos) * frac
        ascent = max(0.0, self.area.elev_at_xy(x1) - self.area.elev_at_xy(x0))
        cost = s.fly_cost(step, ascent)
        self.battery -= cost
        self.energy_spent += cost
        self.dist_flown += step
        self.pos = x1
        self._leg_progress += step

        ev = {}
        if step >= remaining - 1e-9:                 # arrived at waypoint
            self.pos = toward_xy.copy() if isinstance(toward_xy, np.ndarray) \
                else np.asarray(toward_xy, dtype=float)
            ev["arrived"] = True
        return ev

    def tick(self, dt_min: float) -> Dict:
        """Advance one decision tick. Events::

            arrived        reached the assigned search cell
            search_done    finished dwell at target_cell (cell id attached)
            cycle_done     completed all cells of current cycle (at base)
            forced_rtb     reserve rule triggered an early return
            swapped        battery replaced at base
        """
        s = self.spec
        ev: Dict = {}

        if self.mode == "done":
            return ev

        if self.mode == "idle":
            ev.update(self._start_next_cycle_or_idle())
            if self.mode == "idle":
                return ev                            # wait (no work queued)

        if self.mode == "fly":
            tgt = self._cell_xy(self.target_cell)
            ev.update(self._fly_step(dt_min, tgt))
            if ev.get("arrived"):
                self.dwell_left = self._pending_dwell()
                self.mode = "search"

        elif self.mode == "search":
            cost = s.hover_cost(dt_min)
            if self.battery - cost < s.reserve_frac * s.battery_wh:
                self.mode = "rtb"                    # abort dwell -> go home
                self.target_cell = None
                self._leg_from = self.pos.copy()
                self._leg_len = float(np.hypot(*(self.base_xy - self.pos)))
                self._leg_progress = 0.0
                ev["forced_rtb"] = True
            else:
                self.battery -= cost
                self.energy_spent += cost
                self.search_minutes += dt_min
                ev["searching"] = self.target_cell      # observation opportunity
                self.dwell_left -= dt_min
                if self.dwell_left <= 0:
                    cell = self.target_cell
                    self.visits[cell] = self.visits.get(cell, 0) + 1
                    ev["search_done"] = cell
                    self._advance_within_cycle()

        elif self.mode == "rtb":
            ev.update(self._fly_step(dt_min, self.base_xy, guard=False))
            if ev.get("arrived"):
                if s.allow_recharge:
                    self.mode = "swap"
                    self.swap_left = s.swap_minutes
                else:
                    self.mode = "idle"

        elif self.mode == "swap":
            self.swap_left -= dt_min
            if self.swap_left <= 0:
                self.battery = float(s.battery_wh)
                self.n_swaps += 1
                ev["swapped"] = True
                self.mode = "idle"
                ev.update(self._start_next_cycle_or_idle())

        return ev

    # ------------------------------------------------------------------
    def _pending_dwell(self) -> int:
        if getattr(self, "_cur_cycle", None) \
                and self._cycle_idx < len(self._cur_cycle):
            return int(self._cur_cycle[self._cycle_idx][1])
        return 0

    def _advance_within_cycle(self) -> None:
        self._cycle_idx += 1
        cyc = self._cur_cycle
        if cyc is not None and self._cycle_idx < len(cyc):
            self._begin_leg_to_cell(cyc[self._cycle_idx][0])
        else:
            # end of cycle -> head home
            self._cur_cycle = None
            self.mode = "rtb"
            self.target_cell = None
            self._leg_from = self.pos.copy()
            self._leg_len = float(np.hypot(*(self.base_xy - self.pos)))
            self._leg_progress = 0.0
