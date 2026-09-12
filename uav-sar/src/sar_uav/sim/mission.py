"""Simulation execution of joint multi-UAV schedules.

Corresponds to proposal section:
    - Safe return-to-depot protocol upon target detection (including mid-flight completion to endpoint v)
    - Statistical metrics collection (DSR, RMST, actual energy, Calibration Gap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sar_uav.experiments.rng import IndexedRNGStream
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule


@dataclass
class MissionExecutionResult:
    """Outcomes of executing a schedule in simulation under ground truth."""
    detected: bool
    detection_tick: Optional[int]
    rmst_time: float               # min(T_r, T_max) in ticks
    actual_energy_joules: float    # total energy consumed across all UAVs
    predicted_J: float             # planner expected J_S(a)
    calibration_gap: float         # predicted_J - (1.0 if detected else 0.0)
    details: Dict


class ScheduleSimulator:
    """Executes a JointSchedule against hidden true target path."""

    def __init__(
        self,
        checker: ConstraintChecker,
        true_lambda_rates: np.ndarray,
        delta_t: float = 60.0
    ):
        self.checker = checker
        self.true_lambda = np.asarray(true_lambda_rates, dtype=np.float64)
        if self.true_lambda.ndim != 1:
            raise ValueError(f"true_lambda_rates must be a 1D array, got ndim={self.true_lambda.ndim}")
        num_nodes = self.checker.dist_matrix.shape[0]
        if len(self.true_lambda) != num_nodes:
            raise ValueError(
                f"true_lambda_rates length ({len(self.true_lambda)}) must match checker cell count ({num_nodes})"
            )
        if not np.all(np.isfinite(self.true_lambda)):
            raise ValueError("true_lambda_rates must be finite (no NaN or Inf)")
        if np.any(self.true_lambda < 0.0):
            raise ValueError(f"true_lambda_rates must be non-negative, got min={float(np.min(self.true_lambda))}")
        if not (np.isfinite(delta_t) and delta_t > 0.0):
            raise ValueError(f"delta_t must be finite and positive, got {delta_t}")
        self.delta_t = delta_t

    def run_simulation(
        self,
        joint_sched: JointSchedule,
        target_path: np.ndarray,
        mission_idx: int = 0,
        rng_stream: Optional["IndexedRNGStream"] = None,
        predicted_J: float = 0.0
    ) -> MissionExecutionResult:
        """Simulates schedule execution tick by tick.

        Enforces input validation: schedules must be valid according to the checker
        and consistent in horizon, delta_t, and cell dimensions before simulation.
        Also validates target_path length, dtype, coordinate ranges, and 1D shape.
        """
        if abs(self.delta_t - self.checker.delta_t) > 1e-6:
            raise ValueError(
                f"Simulator delta_t ({self.delta_t}) != checker delta_t ({self.checker.delta_t})"
            )

        feasible, reason = self.checker.check_joint_schedule(joint_sched)
        if not feasible:
            raise ValueError(f"Infeasible schedule rejected by simulator: {reason}")

        if rng_stream is None:
            from sar_uav.experiments.rng import IndexedRNGStream
            rng_stream = IndexedRNGStream(master_seed=42)

        H = joint_sched.H
        target_path_arr = np.asarray(target_path)
        if target_path_arr.ndim != 1:
            raise ValueError(f"target_path must be a 1D array of cell indices, got ndim={target_path_arr.ndim}")
        if len(target_path_arr) < H:
            raise ValueError(
                f"target_path length ({len(target_path_arr)}) is less than schedule horizon H ({H})"
            )
        if not np.issubdtype(target_path_arr.dtype, np.integer):
            raise ValueError(f"target_path must contain integer cell indices, got dtype={target_path_arr.dtype}")
        if not np.all(np.isfinite(target_path_arr)):
            raise ValueError("target_path must contain finite cell indices")
        num_nodes = self.checker.dist_matrix.shape[0]
        min_cell = int(np.min(target_path_arr[:H]))
        max_cell = int(np.max(target_path_arr[:H]))
        if min_cell < 0 or max_cell >= num_nodes:
            raise ValueError(
                f"target_path cell indices out of range [0, {num_nodes - 1}]: min={min_cell}, max={max_cell}"
            )

        # Pre-calculate active action for each UAV at each tick:
        # action[k, t] = ('search', cell_idx) or ('fly', from_node, to_node) or ('wait', node) or ('depot', 0)
        uav_actions: Dict[int, List[Tuple]] = {}

        for u_sched in joint_sched.uav_schedules:
            k = u_sched.uav_idx
            actions = [('depot', self.checker.depot_idx) for _ in range(H)]
            if u_sched.r_k > 0:
                current_node = self.checker.depot_idx
                for l_idx, v in enumerate(u_sched.visits):
                    dest_node = v.cell_idx
                    tau_leg = self.checker.tau_bar[k, current_node, dest_node]

                    if l_idx == 0:
                        # Departure delay at depot
                        for t in range(0, min(H, v.wait_ticks)):
                            actions[t] = ('wait_depot', self.checker.depot_idx)
                        flight_start = v.wait_ticks
                    else:
                        prev_v = u_sched.visits[l_idx - 1]
                        flight_start = prev_v.start_tick + prev_v.dwell_ticks

                    # Flight leg
                    for t in range(flight_start, min(H, flight_start + tau_leg)):
                        actions[t] = ('fly', current_node, dest_node)

                    # Arrival wait (l >= 2)
                    if l_idx > 0 and v.wait_ticks > 0:
                        wait_start = flight_start + tau_leg
                        for t in range(wait_start, min(H, wait_start + v.wait_ticks)):
                            actions[t] = ('wait_station', dest_node)

                    # Search dwell
                    for t in range(v.start_tick, min(H, v.start_tick + v.dwell_ticks)):
                        actions[t] = ('search', dest_node)

                    current_node = dest_node

                # Return leg to depot
                last_v = u_sched.visits[-1]
                ret_start = last_v.start_tick + last_v.dwell_ticks
                tau_ret = self.checker.tau_bar[k, current_node, self.checker.depot_idx]
                for t in range(ret_start, min(H, ret_start + tau_ret)):
                    actions[t] = ('fly_return', current_node, self.checker.depot_idx)

            uav_actions[k] = actions

        # Tick-by-tick simulation loop
        detected = False
        detection_tick = None
        detecting_uav = None

        for t in range(H):
            target_cell = int(target_path[t])

            for k, actions in uav_actions.items():
                act = actions[t]
                if act[0] == 'search':
                    search_cell = act[1]
                    if search_cell == target_cell:
                        # True detection probability
                        rate = self.true_lambda[search_cell]
                        q_true = 1.0 - np.exp(-rate * (self.delta_t / 60.0))
                        u_rand = rng_stream.uniform(mission_idx, t, k, "detection")
                        if u_rand < q_true:
                            detected = True
                            detection_tick = t
                            detecting_uav = k
                            break
            if detected:
                break

        # Sortie battery energy consumption calculation with early termination.
        # Evaluates actual energy drawn during departure wait, flight legs, on-station wait,
        # and search dwell, plus safe return to depot upon early detection or mission completion.
        # Aligns strictly with ConstraintChecker.check_uav_feasibility().
        total_energy = 0.0
        end_tick = detection_tick if detected else H - 1

        for u_sched in joint_sched.uav_schedules:
            k = u_sched.uav_idx
            spec = self.checker.uav_specs[k]

            if u_sched.r_k == 0:
                # Inactive UAV at depot: zero sortie energy consumed
                continue

            current_node = self.checker.depot_idx
            mission_ended = False

            for l_idx, v in enumerate(u_sched.visits):
                dest_node = v.cell_idx
                tau_leg = self.checker.tau_bar[k, current_node, dest_node]

                # Departure tick and wait before flight
                if l_idx == 0:
                    departure_tick = v.wait_ticks
                    if end_tick < departure_tick:
                        # Detection during initial depot delay: abort before takeoff
                        total_energy += spec.p_idle * (end_tick + 1) * self.delta_t
                        mission_ended = True
                        break
                    else:
                        total_energy += spec.p_idle * v.wait_ticks * self.delta_t
                else:
                    prev_v = u_sched.visits[l_idx - 1]
                    departure_tick = prev_v.start_tick + prev_v.dwell_ticks
                    if end_tick < departure_tick:
                        # Detection occurred at or before departure tick
                        mission_ended = True
                        break

                # Flight leg from current_node to dest_node
                flight_end_tick = departure_tick + tau_leg - 1

                if end_tick < departure_tick:
                    mission_ended = True
                    break

                if end_tick < flight_end_tick:
                    # Detection mid-flight: complete current leg to dest_node, then return
                    total_energy += self.checker.e_bar[k, current_node, dest_node]
                    total_energy += self.checker.e_bar[k, dest_node, self.checker.depot_idx]
                    mission_ended = True
                    break
                else:
                    # Full flight leg completed
                    total_energy += self.checker.e_bar[k, current_node, dest_node]

                # On-station wait at dest_node (for l_idx >= 1)
                dwell_start = v.start_tick
                if l_idx > 0 and v.wait_ticks > 0:
                    wait_start = departure_tick + tau_leg
                    if end_tick < dwell_start:
                        # Detection during on-station wait
                        completed_wait = max(0, end_tick - wait_start + 1)
                        total_energy += spec.p_idle * completed_wait * self.delta_t
                        total_energy += self.checker.e_bar[k, dest_node, self.checker.depot_idx]
                        mission_ended = True
                        break
                    else:
                        total_energy += spec.p_idle * v.wait_ticks * self.delta_t

                # Search dwell at dest_node
                dwell_end = dwell_start + v.dwell_ticks - 1

                if end_tick <= dwell_end:
                    # Detection occurred during dwell (including first, middle, or exact last tick)
                    dwell_done = max(0, end_tick - dwell_start + 1)
                    total_energy += spec.p_search * dwell_done * self.delta_t
                    # Abort further schedule and return to depot from dest_node
                    total_energy += self.checker.e_bar[k, dest_node, self.checker.depot_idx]
                    mission_ended = True
                    break
                else:
                    # Full dwell completed
                    total_energy += spec.p_search * v.dwell_ticks * self.delta_t

                current_node = dest_node

            if not mission_ended:
                # All visits completed without detection: UAV returns to depot
                total_energy += self.checker.e_bar[k, current_node, self.checker.depot_idx]

        rmst = float(detection_tick) if detected else float(H)
        indicator = 1.0 if detected else 0.0
        calib_gap = float(predicted_J - indicator)

        return MissionExecutionResult(
            detected=detected,
            detection_tick=detection_tick,
            rmst_time=rmst,
            actual_energy_joules=total_energy,
            predicted_J=predicted_J,
            calibration_gap=calib_gap,
            details={"detecting_uav": detecting_uav, "end_tick": end_tick}
        )
