"""Neighborhood structure and 8 lazy perturbation operators.

Corresponds to proposal section "Cau truc khong gian lang gieng N(a)":
    1. Reorder Visits (Swap / 2-opt)
    2. Insert Visit (supports revisit of previously searched cells)
    3. Delete Visit
    4. Replace Visit (essential for budget-saturated schedules)
    5. Reassign Visit (Relocate between UAVs)
    6. Change Dwell (stepwise adjustment in D)
    7. Dwell Rebalance / Transfer (preserves combined dwell, re-checks individual budgets)
    8. Adjust Wait (departure delay w_{k1} or arrival wait w_{kl})
"""

from __future__ import annotations

from typing import Generator, Iterator, List, Optional, Sequence, Tuple
import numpy as np

from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, Visit


def detect_affected_interval(
    orig_sched: JointSchedule,
    cand_sched: JointSchedule
) -> Tuple[int, int]:
    """Identify the affected time interval [t_a, t_b] where actions or timings changed.
    If subsequent visits are time-shifted, t_b is set to H - 1.
    """
    H = orig_sched.H
    orig_fp = orig_sched.generate_footprint_array()
    cand_fp = cand_sched.generate_footprint_array()

    diff_ticks = np.where(np.any(orig_fp != cand_fp, axis=1))[0]
    if len(diff_ticks) == 0:
        return 0, 0

    t_a = int(diff_ticks[0])
    t_b_diff = int(diff_ticks[-1])

    # Check if any visit start times or durations shifted the schedule
    shifted = False
    for k in range(len(orig_sched.uav_schedules)):
        s_orig = orig_sched.uav_schedules[k]
        s_cand = cand_sched.uav_schedules[k]
        if s_orig.r_k != s_cand.r_k:
            shifted = True
            break
        for v_o, v_c in zip(s_orig.visits, s_cand.visits):
            if v_o.start_tick != v_c.start_tick or v_o.dwell_ticks != v_c.dwell_ticks:
                shifted = True
                break
        if shifted:
            break

    t_b = (H - 1) if shifted else t_b_diff
    return t_a, t_b


class NeighborhoodExplorer:
    """Generates feasible neighbor schedules lazily using 8 operators."""

    def __init__(
        self,
        checker: ConstraintChecker,
        candidate_cells: Sequence[int],
        allowed_dwells: Sequence[int] = (1, 2, 3, 4, 5),
        max_wait_ticks: int = 10,
        max_visits: Optional[int] = None,
        enable_replace: bool = True,
        enable_dwell_rebalance: bool = True,
        order_strategy: str = "fixed",
        seed: Optional[int] = None,
    ):
        self.checker = checker
        self.candidate_cells = list(candidate_cells)
        self.allowed_dwells = list(allowed_dwells)
        self.max_wait_ticks = max_wait_ticks
        self.max_visits = max_visits
        self.enable_replace = bool(enable_replace)
        self.enable_dwell_rebalance = bool(enable_dwell_rebalance)
        self.order_strategy = str(order_strategy).lower()
        self.seed = seed
        self._iteration = 0

    def reset_iteration(self) -> None:
        """Reset iteration counter for deterministic behavior."""
        self._iteration = 0

    def step_iteration(self) -> None:
        """Advance iteration counter."""
        self._iteration += 1

    def get_operator_order(self) -> List[str]:
        """Returns the ordered list of active operator names for the current iteration."""
        operators = []
        if self.enable_replace:
            operators.append("Replace")
        if self.enable_dwell_rebalance:
            operators.append("DwellRebalance")
        operators.extend([
            "Swap",
            "ChangeDwell",
        ])
        if self.max_wait_ticks > 0:
            operators.append("AdjustWait")
        operators.extend([
            "Insert",
            "Delete",
            "Relocate",
        ])

        if len(operators) > 1:
            if self.order_strategy == "round_robin":
                shift = self._iteration % len(operators)
                operators = operators[shift:] + operators[:shift]
            elif self.order_strategy == "shuffled":
                rng = np.random.default_rng(
                    (self.seed if self.seed is not None else 42) + self._iteration * 10007
                )
                perm = rng.permutation(len(operators))
                operators = [operators[i] for i in perm]
        return operators

    def generate_all_neighbors(
        self,
        current_sched: JointSchedule
    ) -> Generator[Tuple[str, JointSchedule, int, int], None, None]:
        """Yields (operator_name, candidate_sched, t_a, t_b) lazily."""
        op_map = {
            "Replace": self.gen_replace_visits,
            "DwellRebalance": self.gen_dwell_rebalance,
            "Swap": self.gen_reorder_visits,
            "ChangeDwell": self.gen_change_dwell,
            "AdjustWait": self.gen_adjust_wait,
            "Insert": self.gen_insert_visits,
            "Delete": self.gen_delete_visits,
            "Relocate": self.gen_relocate_visits,
        }

        for op_name in self.get_operator_order():
            op_func = op_map[op_name]
            for cand in op_func(current_sched):
                feasible, _ = self.checker.check_joint_schedule(cand)
                if feasible:
                    t_a, t_b = detect_affected_interval(current_sched, cand)
                    yield op_name, cand, t_a, t_b

    def gen_reorder_visits(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 1: Swap or 2-opt subsegment within single UAV."""
        for k_idx, u_sched in enumerate(current.uav_schedules):
            r = u_sched.r_k
            if r < 2:
                continue
            # Swap pairs
            for i in range(r):
                for j in range(i + 1, r):
                    cand = current.clone()
                    target_uav = cand.uav_schedules[k_idx]
                    target_uav.visits[i], target_uav.visits[j] = target_uav.visits[j], target_uav.visits[i]
                    yield cand

            # 2-opt inversion for segments >= 3
            if r >= 3:
                for i in range(r - 2):
                    for j in range(i + 2, r):
                        cand = current.clone()
                        target_uav = cand.uav_schedules[k_idx]
                        target_uav.visits[i : j + 1] = list(reversed(target_uav.visits[i : j + 1]))
                        yield cand

    def gen_insert_visits(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 2: Insert visit (supports revisit of previously searched cells)."""
        for k_idx, u_sched in enumerate(current.uav_schedules):
            if self.max_visits is not None and u_sched.r_k >= self.max_visits:
                continue
            r = u_sched.r_k
            for pos in range(r + 1):
                for cell in self.candidate_cells:
                    for d in self.allowed_dwells[:3]:  # small dwell insertion
                        cand = current.clone()
                        new_visit = Visit(cell_idx=cell, dwell_ticks=d, wait_ticks=0)
                        cand.uav_schedules[k_idx].visits.insert(pos, new_visit)
                        yield cand

    def gen_delete_visits(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 3: Remove a visit."""
        for k_idx, u_sched in enumerate(current.uav_schedules):
            for pos in range(u_sched.r_k):
                cand = current.clone()
                cand.uav_schedules[k_idx].visits.pop(pos)
                yield cand

    def gen_replace_visits(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 4: Replace visit v_{kl} with alternative cell c in G (anti-budget-trap move)."""
        for k_idx, u_sched in enumerate(current.uav_schedules):
            for pos, orig_v in enumerate(u_sched.visits):
                for alt_cell in self.candidate_cells:
                    if alt_cell == orig_v.cell_idx:
                        continue
                    cand = current.clone()
                    cand.uav_schedules[k_idx].visits[pos].cell_idx = alt_cell
                    yield cand

    def gen_relocate_visits(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 5: Transfer visit from UAV k1 to UAV k2."""
        m = len(current.uav_schedules)
        if m < 2:
            return
        for k1 in range(m):
            if current.uav_schedules[k1].r_k == 0:
                continue
            for pos in range(current.uav_schedules[k1].r_k):
                for k2 in range(m):
                    if k1 == k2:
                        continue
                    if self.max_visits is not None and current.uav_schedules[k2].r_k >= self.max_visits:
                        continue
                    for ins_pos in range(current.uav_schedules[k2].r_k + 1):
                        cand = current.clone()
                        moved_v = cand.uav_schedules[k1].visits.pop(pos)
                        cand.uav_schedules[k2].visits.insert(ins_pos, moved_v)
                        yield cand

    def gen_change_dwell(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 6: Stepwise change dwell duration d_{kl} in D."""
        for k_idx, u_sched in enumerate(current.uav_schedules):
            for pos, v in enumerate(u_sched.visits):
                for new_d in self.allowed_dwells:
                    if new_d != v.dwell_ticks:
                        cand = current.clone()
                        cand.uav_schedules[k_idx].visits[pos].dwell_ticks = new_d
                        yield cand

    def gen_dwell_rebalance(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 7: Transfer dwell ticks between two visits without changing total dwell."""
        all_visits = []
        for k, u_sched in enumerate(current.uav_schedules):
            for pos in range(u_sched.r_k):
                all_visits.append((k, pos))

        if len(all_visits) < 2:
            return

        min_d = min(self.allowed_dwells)
        max_d = max(self.allowed_dwells)

        for i in range(len(all_visits)):
            for j in range(len(all_visits)):
                if i == j:
                    continue
                k1, pos1 = all_visits[i]
                k2, pos2 = all_visits[j]

                d1 = current.uav_schedules[k1].visits[pos1].dwell_ticks
                d2 = current.uav_schedules[k2].visits[pos2].dwell_ticks

                for step in (1, 2):
                    if d1 - step >= min_d and d2 + step <= max_d:
                        cand = current.clone()
                        cand.uav_schedules[k1].visits[pos1].dwell_ticks -= step
                        cand.uav_schedules[k2].visits[pos2].dwell_ticks += step
                        yield cand

    def gen_adjust_wait(self, current: JointSchedule) -> Generator[JointSchedule, None, None]:
        """Operator 8: Adjust wait time w_{kl} (delay at depot or on-station wait)."""
        if self.max_wait_ticks <= 0:
            return
        for k_idx, u_sched in enumerate(current.uav_schedules):
            for pos, v in enumerate(u_sched.visits):
                for delta_w in (-2, -1, 1, 2):
                    new_w = v.wait_ticks + delta_w
                    if 0 <= new_w <= self.max_wait_ticks:
                        cand = current.clone()
                        cand.uav_schedules[k_idx].visits[pos].wait_ticks = new_w
                        yield cand
