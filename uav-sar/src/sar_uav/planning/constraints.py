"""Routing and energy constraints with discrete action chaining and mid-flight safety.

Corresponds to proposal section "Rang buoc dinh tuyen va nang luong":
    - Equalities for action chaining (cons1 - cons2):
        s_{k1} = w_{k1} + tau_{o, v_{k1}}
        s_{k,l+1} = s_{kl} + d_{kl} + tau_{v_{kl}, v_{k,l+1}} + w_{k,l+1}
    - Deadline constraint (cons3):
        s_{kr_k} + d_{kr_k} + tau_{v_{kr_k}, o} <= H
    - Energy budget with discrete round-off (cons4):
        sum_{l=0}^{r_k} e_{v_{kl}, v_{k,l+1}} + P_search * sum d_{kl} * dt + P_idle * sum w_{kl} * dt <= B_k - R_k
    - Pre-leg departure safety check:
        B_k(t) >= e_{uv} + e_{vo} + R_k,   t + tau_{uv} + tau_{vo} <= H
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class UAVSpec:
    """Aircraft flight, search, and battery parameters."""
    name: str = "uav1"
    speed_ms: float = 15.0               # cruise velocity m/s
    p_flight: float = 180.0              # Watts during straight flight
    p_search: float = 200.0              # Watts during hover/search
    p_idle: float = 60.0                 # Watts during active wait (ground or hover-wait)
    battery_joules: float = 360000.0     # nominal battery capacity B_k (e.g. 100 Wh = 360 kJ)
    reserve_joules: float = 54000.0      # safety reserve R_k (e.g. 15%)
    delta_t: float = 60.0                # seconds per tick


@dataclass
class Visit:
    """A single visit by a UAV to a search cell."""
    cell_idx: int
    dwell_ticks: int                     # d_{kl} in D
    wait_ticks: int = 0                  # w_{kl}: delay at depot if l=1, or arrival wait if l >= 2
    start_tick: int = 0                  # s_{kl}: calculated from chaining equations

    def clone(self) -> "Visit":
        return Visit(
            cell_idx=self.cell_idx,
            dwell_ticks=self.dwell_ticks,
            wait_ticks=self.wait_ticks,
            start_tick=self.start_tick
        )

    def to_dict(self) -> Dict:
        return {
            "cell_idx": int(self.cell_idx),
            "dwell_ticks": int(self.dwell_ticks),
            "wait_ticks": int(self.wait_ticks),
            "start_tick": int(self.start_tick)
        }


@dataclass
class UAVSchedule:
    """Schedule for one UAV: chain of visits starting and ending at depot."""
    uav_idx: int
    visits: List[Visit] = field(default_factory=list)

    @property
    def r_k(self) -> int:
        return len(self.visits)

    def clone(self) -> "UAVSchedule":
        return UAVSchedule(
            uav_idx=self.uav_idx,
            visits=[Visit(cell_idx=v.cell_idx, dwell_ticks=v.dwell_ticks,
                          wait_ticks=v.wait_ticks, start_tick=v.start_tick)
                    for v in self.visits]
        )

    def to_dict(self) -> Dict:
        return {
            "uav_idx": int(self.uav_idx),
            "r_k": self.r_k,
            "visits": [v.to_dict() for v in self.visits]
        }


class JointSchedule:
    """Coordinated multi-UAV schedule a = (a_0, ..., a_{H-1})."""

    def __init__(
        self,
        uav_schedules: Sequence[UAVSchedule],
        num_cells: int,
        H: int,
        delta_t: float = 60.0
    ):
        self.uav_schedules = list(uav_schedules)
        self.num_cells = num_cells
        self.H = H
        self.delta_t = delta_t

    def clone(self) -> "JointSchedule":
        return JointSchedule(
            uav_schedules=[s.clone() for s in self.uav_schedules],
            num_cells=self.num_cells,
            H=self.H,
            delta_t=self.delta_t
        )

    def to_dict(self) -> Dict:
        return {
            "num_cells": int(self.num_cells),
            "H": int(self.H),
            "delta_t": float(self.delta_t),
            "uav_schedules": [s.to_dict() for s in self.uav_schedules]
        }

    def generate_footprint_array(self) -> np.ndarray:
        """Convert schedule into optical coverage array of shape [H, num_cells].
        Each UAV searching cell v during [s_{kl}, s_{kl} + d_{kl} - 1] contributes 1.0.
        Multiple UAVs searching the same cell at the same tick accumulate
        (e.g., 2 UAVs → coverage = 2.0) so that the likelihood model
        exp(-lambda * total_coverage * dt/60) correctly captures independent sensing.
        Always returns a fresh array reflecting current visit state.
        """
        footprints = np.zeros((self.H, self.num_cells), dtype=np.float64)
        for sched in self.uav_schedules:
            for v in sched.visits:
                for tick in range(v.start_tick, min(self.H, v.start_tick + v.dwell_ticks)):
                    footprints[tick, v.cell_idx] += 1.0
        return footprints


class ConstraintChecker:
    """Validates feasibility of flight timings, energy, and mid-flight abort checks."""

    def __init__(
        self,
        dist_matrix_m: np.ndarray,
        depot_idx: int,
        uav_specs: Sequence[UAVSpec],
        H: int,
        delta_t: float = 60.0
    ):
        """
        dist_matrix_m: distance in meters between all cells (including depot).
        depot_idx: cell index of depot.
        uav_specs: list of UAVSpec for each UAV k.
        H: horizon in ticks.
        """
        self.dist_matrix = np.asarray(dist_matrix_m, dtype=np.float64)
        self.depot_idx = depot_idx
        self.uav_specs = list(uav_specs)
        self.m = len(self.uav_specs)
        self.H = H
        self.delta_t = delta_t

        # Precompute discrete flight ticks and energy for each UAV
        # tau_bar_ticks[k, i, j], e_bar_joules[k, i, j]
        num_nodes = self.dist_matrix.shape[0]
        self.tau_bar = np.zeros((self.m, num_nodes, num_nodes), dtype=np.int64)
        self.e_bar = np.zeros((self.m, num_nodes, num_nodes), dtype=np.float64)

        for k, spec in enumerate(self.uav_specs):
            raw_time_sec = self.dist_matrix / max(1.0, spec.speed_ms)
            discrete_ticks = np.ceil(raw_time_sec / self.delta_t).astype(np.int64)
            self.tau_bar[k] = discrete_ticks

            # Discrete flight energy including idle roundoff:
            # e_bar = P_flight * raw_time + P_idle * (discrete_ticks * dt - raw_time)
            roundoff_idle_sec = np.maximum(0.0, discrete_ticks * self.delta_t - raw_time_sec)
            self.e_bar[k] = (spec.p_flight * raw_time_sec) + (spec.p_idle * roundoff_idle_sec)

    def compute_and_update_chaining(self, sched: UAVSchedule) -> bool:
        """Compute start_tick for all visits using chaining equations (cons1 - cons2).
        Returns False if deadline H is exceeded at any point, True otherwise.
        """
        k = sched.uav_idx
        if sched.r_k == 0:
            return True

        current_node = self.depot_idx
        current_tick = 0

        for l_idx, visit in enumerate(sched.visits):
            if visit.dwell_ticks <= 0 or visit.wait_ticks < 0:
                return False
            dest_node = visit.cell_idx
            tau_leg = self.tau_bar[k, current_node, dest_node]

            if l_idx == 0:
                # cons1: s_{k1} = w_{k1} + tau_{o, v_{k1}}
                visit.start_tick = visit.wait_ticks + tau_leg
            else:
                # cons2: s_{k,l+1} = s_{kl} + d_{kl} + tau_{v_{kl}, v_{k,l+1}} + w_{k,l+1}
                prev_visit = sched.visits[l_idx - 1]
                visit.start_tick = (prev_visit.start_tick + prev_visit.dwell_ticks +
                                    tau_leg + visit.wait_ticks)

            current_tick = visit.start_tick + visit.dwell_ticks
            current_node = dest_node

            if current_tick > self.H:
                return False

        # cons3: return to depot before H
        tau_return = self.tau_bar[k, current_node, self.depot_idx]
        if current_tick + tau_return > self.H:
            return False

        return True

    def check_uav_feasibility(self, sched: UAVSchedule) -> Tuple[bool, str]:
        """Check all constraints (cons1 - cons4) and mid-flight return feasibility."""
        k = sched.uav_idx
        spec = self.uav_specs[k]

        if sched.r_k == 0:
            return True, "ok_idle"

        # Validate non-negative wait and strictly positive dwell
        for l_idx, visit in enumerate(sched.visits):
            if visit.dwell_ticks <= 0:
                return False, f"invalid_dwell_{visit.dwell_ticks}_at_visit_{l_idx}"
            if visit.wait_ticks < 0:
                return False, f"invalid_wait_{visit.wait_ticks}_at_visit_{l_idx}"

        # 1. Update chaining & check deadline
        if not self.compute_and_update_chaining(sched):
            return False, "deadline_or_timing_infeasible"

        # 2. Energy consumption calculation
        total_energy = 0.0
        current_node = self.depot_idx
        current_tick = 0

        for l_idx, visit in enumerate(sched.visits):
            dest_node = visit.cell_idx
            tau_leg = self.tau_bar[k, current_node, dest_node]
            e_leg = self.e_bar[k, current_node, dest_node]

            # Departure tick for this leg
            if l_idx == 0:
                departure_tick = visit.wait_ticks
                total_energy += spec.p_idle * visit.wait_ticks * self.delta_t
            else:
                prev_visit = sched.visits[l_idx - 1]
                departure_tick = prev_visit.start_tick + prev_visit.dwell_ticks

            # Check pre-leg departure mid-flight safety:
            # B_k(t_dep) >= e_{uv} + e_{vo} + R_k
            # and t_dep + tau_{uv} + tau_{vo} <= H
            tau_return_from_dest = self.tau_bar[k, dest_node, self.depot_idx]
            e_return_from_dest = self.e_bar[k, dest_node, self.depot_idx]

            if departure_tick + tau_leg + tau_return_from_dest > self.H:
                return False, f"pre_leg_deadline_exceeded_leg_{l_idx}"

            remaining_budget_at_dep = spec.battery_joules - total_energy
            required_safety_energy = e_leg + e_return_from_dest + spec.reserve_joules
            if remaining_budget_at_dep < required_safety_energy:
                return False, f"pre_leg_energy_deficit_leg_{l_idx}"

            # Add flight energy
            total_energy += e_leg

            # Add on-station wait energy (for l >= 2)
            if l_idx > 0 and visit.wait_ticks > 0:
                total_energy += spec.p_idle * visit.wait_ticks * self.delta_t

            # Add search dwell energy
            total_energy += spec.p_search * visit.dwell_ticks * self.delta_t
            current_node = dest_node

        # Return leg to depot
        e_final_return = self.e_bar[k, current_node, self.depot_idx]
        total_energy += e_final_return

        # Check total energy vs budget minus reserve (cons4)
        if total_energy > spec.battery_joules - spec.reserve_joules:
            return False, "total_energy_exceeded"

        return True, "feasible"

    def check_joint_schedule(self, joint_sched: JointSchedule) -> Tuple[bool, str]:
        """Check feasibility across all UAVs in joint schedule.

        Performs structural validation (unique IDs, index ranges) before
        per-UAV constraint checks, so that downstream code never sees
        an inconsistent schedule (e.g. duplicate UAV IDs that would cause
        footprint/simulator divergence).
        """
        num_nodes = self.dist_matrix.shape[0]

        # Dimension consistency checks between joint_sched and checker
        if joint_sched.H != self.H:
            return False, f"horizon_mismatch_sched_{joint_sched.H}_checker_{self.H}"
        if abs(joint_sched.delta_t - self.delta_t) > 1e-6:
            return False, f"delta_t_mismatch_sched_{joint_sched.delta_t}_checker_{self.delta_t}"
        if joint_sched.num_cells != num_nodes:
            return False, f"num_cells_mismatch_sched_{joint_sched.num_cells}_checker_{num_nodes}"

        seen_ids: set = set()

        for sched in joint_sched.uav_schedules:
            # Duplicate UAV ID check
            if sched.uav_idx in seen_ids:
                return False, f"duplicate_uav_idx_{sched.uav_idx}"
            seen_ids.add(sched.uav_idx)

            # UAV ID range check
            if sched.uav_idx < 0 or sched.uav_idx >= self.m:
                return False, f"uav_idx_{sched.uav_idx}_out_of_range_0_to_{self.m - 1}"

            # Cell index range check
            for v_idx, v in enumerate(sched.visits):
                if v.cell_idx < 0 or v.cell_idx >= num_nodes:
                    return False, (
                        f"UAV_{sched.uav_idx}_visit_{v_idx}_"
                        f"cell_{v.cell_idx}_out_of_range_0_to_{num_nodes - 1}"
                    )

        # Per-UAV constraint checks (chaining, energy, deadline, mid-flight safety)
        for sched in joint_sched.uav_schedules:
            feasible, reason = self.check_uav_feasibility(sched)
            if not feasible:
                return False, f"UAV_{sched.uav_idx}_{reason}"
        return True, "all_feasible"
