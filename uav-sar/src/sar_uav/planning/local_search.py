"""Algorithm 1: Joint Route-Effort Local Search with Fast Evaluator.

Corresponds to proposal section "Thuat toan toi uu hoa cuc bo phoi hop Route--Effort" (Algorithm 1).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple
import numpy as np

from sar_uav.planning.constraints import ConstraintChecker, JointSchedule
from sar_uav.planning.evaluator import BaseEvaluator, ForwardBackwardFastEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer

log = logging.getLogger(__name__)


class JointRouteEffortLocalSearch:
    """Algorithm 1: Joint Route-Effort Local Search with Fast Evaluator."""

    def __init__(
        self,
        explorer: NeighborhoodExplorer,
        evaluator: BaseEvaluator,
        max_evals: int = 5000,
        time_limit_sec: float = 30.0,
        verbose: bool = False
    ):
        self.explorer = explorer
        self.evaluator = evaluator
        self.max_evals = max_evals
        self.time_limit_sec = time_limit_sec
        self.verbose = verbose

    def solve(
        self,
        initial_schedule: JointSchedule,
        delta_J_threshold: float = 1e-12
    ) -> Tuple[JointSchedule, float, Dict]:
        """Runs the hill-climbing search loop."""
        start_time = time.perf_counter()
        current_sched = initial_schedule.clone()

        # P0.1: Validate and chain initial schedule BEFORE scoring.
        # This ensures start_tick values are computed from chaining equations
        # rather than staying at the default 0, which would imply the UAV
        # searches at tick 0 even if the cell is far from the depot.
        feasible, reason = self.explorer.checker.check_joint_schedule(current_sched)
        if not feasible:
            raise ValueError(
                f"Initial schedule is infeasible ({reason}). "
                f"The solver requires a feasible starting point with valid chaining."
            )

        # Line 1-2: Initialize baseline schedule and cache (now with correct start_ticks)
        current_J = self.evaluator.initialize_base_schedule(current_sched)
        if not np.isfinite(current_J):
            raise RuntimeError(f"Non-finite initial objective encountered: current_J={current_J}")
        if not (-1e-8 <= current_J <= 1.0 + 1e-8):
            raise RuntimeError(f"Initial objective out of probabilistic bounds [0, 1]: current_J={current_J}")
        n_eval = 1
        iteration = 0
        accepted_moves = 0
        cache_update_time_sec = 0.0
        history_J = [current_J]
        history_evals = [n_eval]
        history_times = [0.0]

        operator_stats: Dict[str, Dict[str, float]] = {}

        stopped_reason = "local_optimum"

        while True:
            iteration += 1
            delta_J_max = delta_J_threshold
            best_cand: Optional[JointSchedule] = None
            best_cand_J = current_J
            best_op_name: Optional[str] = None

            # Check budgets at start of outer loop
            elapsed = time.perf_counter() - start_time
            if n_eval >= self.max_evals:
                stopped_reason = "max_evals_exceeded"
                break
            if elapsed >= self.time_limit_sec:
                stopped_reason = "time_limit_exceeded"
                break

            # Line 5-6: Lazy generation over neighborhood operators
            for op_name, cand, t_a, t_b in self.explorer.generate_all_neighbors(current_sched):
                elapsed = time.perf_counter() - start_time
                if n_eval >= self.max_evals or elapsed >= self.time_limit_sec:
                    # Line 7-8: break inner search and jump to accept step
                    break

                if op_name not in operator_stats:
                    operator_stats[op_name] = {"evaluated": 0, "accepted": 0, "improvement": 0.0}
                operator_stats[op_name]["evaluated"] += 1

                # Line 13-14: Evaluate candidate
                delta_J, cand_J = self.evaluator.evaluate_candidate(cand, t_a, t_b)
                n_eval += 1

                # Line 15-16: Track best improving candidate in current neighborhood
                if delta_J > delta_J_max:
                    delta_J_max = delta_J
                    best_cand = cand
                    best_cand_J = cand_J
                    best_op_name = op_name

            # Line 17-20: Accept step
            if delta_J_max > 0.0 and best_cand is not None:
                current_sched = best_cand
                current_J = best_cand_J
                t_cache_start = time.perf_counter()
                self.evaluator.accept_candidate(current_sched, current_J)
                cache_update_time_sec += time.perf_counter() - t_cache_start

                accepted_moves += 1
                if best_op_name in operator_stats:
                    operator_stats[best_op_name]["accepted"] += 1
                    operator_stats[best_op_name]["improvement"] += delta_J_max

                now_elapsed = time.perf_counter() - start_time
                history_J.append(current_J)
                history_evals.append(n_eval)
                history_times.append(now_elapsed)

                if self.verbose:
                    log.info(f"Iter {iteration}: Accepted {best_op_name} -> J={current_J:.5f} (dJ={delta_J_max:.5f}) at eval {n_eval}")
            else:
                # No improving candidate found across the full neighborhood
                if n_eval >= self.max_evals:
                    stopped_reason = "max_evals_exceeded"
                elif time.perf_counter() - start_time >= self.time_limit_sec:
                    stopped_reason = "time_limit_exceeded"
                else:
                    stopped_reason = "local_optimum_reached"
                break

        # Re-verify final schedule feasibility and re-compute objective independently
        # to guarantee mathematical consistency of the returned solution.
        feasible, reason = self.explorer.checker.check_joint_schedule(current_sched)
        if not feasible:
            raise RuntimeError(f"Final schedule became infeasible ({reason})")

        _, _, verified_J = self.evaluator.engine.compute_forward_trajectory(
            current_sched.generate_footprint_array()
        )
        if not (np.isfinite(current_J) and np.isfinite(verified_J)):
            raise RuntimeError(
                f"Non-finite objective encountered in final schedule: accumulated J={current_J}, "
                f"verified J={verified_J}"
            )
        if not (-1e-8 <= current_J <= 1.0 + 1e-8 and -1e-8 <= verified_J <= 1.0 + 1e-8):
            raise RuntimeError(
                f"Objective out of probabilistic bounds [0, 1]: accumulated J={current_J}, "
                f"verified J={verified_J}"
            )
        abs_diff = abs(current_J - verified_J)
        if abs_diff > 1e-8:
            raise RuntimeError(
                f"Objective mismatch in final schedule: accumulated J={current_J:.8f} vs "
                f"verified J={verified_J:.8f} (abs_diff={abs_diff:.2e} > 1e-8)"
            )
        current_J = float(verified_J)

        total_runtime = time.perf_counter() - start_time
        stats = {
            "initial_J": history_J[0],
            "best_J": current_J,
            "improvement": float(current_J - history_J[0]),
            "n_evals": n_eval,
            "iterations": iteration,
            "accepted_moves": accepted_moves,
            "runtime_sec": total_runtime,
            "cache_update_time_sec": cache_update_time_sec,
            "operator_stats": operator_stats,
            "stopped_reason": stopped_reason,
            "history_J": history_J,
            "history_evals": history_evals,
            "history_times": history_times,
            "evaluator_stats": self.evaluator.stats.summary()
        }

        return current_sched, current_J, stats
