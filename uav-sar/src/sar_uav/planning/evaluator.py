"""3-Tier Evaluator framework for schedule candidate evaluation.

Corresponds to proposal section "Evaluator nhanh: Khai thac Backward Continuation va Ba cap do doi chung":
    1. FullEvaluator: Recomputes forward unnormalized mass trajectory from t=0 to H-1.
       Complexity: O(|S| * nnz(M) * H).
    2. PrefixOnlyEvaluator: Reuses prefix u_{t_a}^-, propagates forward from t_a to H-1.
       Complexity: O(|S| * nnz(M) * (H - t_a)).
    3. ForwardBackwardFastEvaluator: Reuses prefix u_{t_a}^-, propagates through [t_a, t_b],
       and joins with backward continuation V_{t_b+1}.
       Complexity: O(|S| * nnz(M) * (t_b - t_a + 1)).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.planning.constraints import JointSchedule


class EvaluatorStats:
    """Tracks evaluation metrics, timing, and affected segment lengths."""

    def __init__(self, name: str):
        self.name = name
        self.eval_count: int = 0
        self.total_time_sec: float = 0.0
        self.l_affected_records: List[int] = []

    def record(self, elapsed: float, l_affected: int = 0):
        self.eval_count += 1
        self.total_time_sec += elapsed
        if l_affected > 0:
            self.l_affected_records.append(l_affected)

    @property
    def avg_time_ms(self) -> float:
        if self.eval_count == 0:
            return 0.0
        return (self.total_time_sec / self.eval_count) * 1000.0

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "eval_count": self.eval_count,
            "total_time_sec": self.total_time_sec,
            "avg_time_ms": self.avg_time_ms,
            "mean_l_affected": float(np.mean(self.l_affected_records)) if self.l_affected_records else 0.0,
            "median_l_affected": float(np.median(self.l_affected_records)) if self.l_affected_records else 0.0
        }


class BaseEvaluator:
    """Abstract base evaluator interface."""

    def __init__(self, engine: JointMassEngine):
        self.engine = engine
        self.H = engine.H
        self.stats = EvaluatorStats("base")

    def initialize_base_schedule(self, joint_sched: JointSchedule) -> float:
        raise NotImplementedError

    def evaluate_candidate(
        self,
        candidate_sched: JointSchedule,
        t_a: int = 0,
        t_b: Optional[int] = None
    ) -> Tuple[float, float]:
        """Returns (delta_J, new_J)."""
        raise NotImplementedError

    def accept_candidate(self, candidate_sched: JointSchedule, new_J: float):
        raise NotImplementedError


class FullEvaluator(BaseEvaluator):
    """Full trajectory recomputation from t=0 to H-1."""

    def __init__(self, engine: JointMassEngine):
        super().__init__(engine)
        self.stats = EvaluatorStats("FullEvaluator")
        self.current_J: float = 0.0
        self.current_footprints: Optional[np.ndarray] = None

    def initialize_base_schedule(self, joint_sched: JointSchedule) -> float:
        self.current_footprints = joint_sched.generate_footprint_array()
        _, _, J = self.engine.compute_forward_trajectory(self.current_footprints)
        self.current_J = J
        return J

    def evaluate_candidate(
        self,
        candidate_sched: JointSchedule,
        t_a: int = 0,
        t_b: Optional[int] = None
    ) -> Tuple[float, float]:
        t0 = time.perf_counter()
        cand_fp = candidate_sched.generate_footprint_array()
        _, _, new_J = self.engine.compute_forward_trajectory(cand_fp)
        elapsed = time.perf_counter() - t0

        l_aff = self.H if t_b is None else (t_b - t_a + 1)
        self.stats.record(elapsed, l_aff)
        delta_J = float(new_J - self.current_J)
        return delta_J, new_J

    def accept_candidate(self, candidate_sched: JointSchedule, new_J: float):
        self.current_footprints = candidate_sched.generate_footprint_array()
        self.current_J = new_J


class PrefixOnlyEvaluator(BaseEvaluator):
    """Reuses forward mass prefix up to t_a, propagates forward t_a to H-1."""

    def __init__(self, engine: JointMassEngine):
        super().__init__(engine)
        self.stats = EvaluatorStats("PrefixOnlyEvaluator")
        self.current_J: float = 0.0
        self.current_footprints: Optional[np.ndarray] = None
        self.u_minus_cache: Optional[np.ndarray] = None
        self.u_plus_cache: Optional[np.ndarray] = None

    def initialize_base_schedule(self, joint_sched: JointSchedule) -> float:
        self.current_footprints = joint_sched.generate_footprint_array()
        u_m, u_p, J = self.engine.compute_forward_trajectory(self.current_footprints)
        self.u_minus_cache = u_m
        self.u_plus_cache = u_p
        self.current_J = J
        return J

    def evaluate_candidate(
        self,
        candidate_sched: JointSchedule,
        t_a: int = 0,
        t_b: Optional[int] = None
    ) -> Tuple[float, float]:
        t0 = time.perf_counter()
        cand_fp = candidate_sched.generate_footprint_array()

        # Prefix reuse at t_a
        u_curr_minus = self.u_minus_cache[t_a].copy() if t_a > 0 else self.engine.u0_minus.copy()

        # Propagate from t_a to H - 1
        for t in range(t_a, self.H):
            fp_t = cand_fp[t]
            ell_t = self.engine.hypothesis_set.compute_all_negative_likelihood(fp_t, delta_t=self.engine.delta_t)
            u_curr_plus = u_curr_minus * ell_t
            if t + 1 < self.H:
                u_curr_minus = np.dot(u_curr_plus, self.engine.M)

        total_fail = float(np.sum(u_curr_plus))
        new_J = float(1.0 - total_fail)
        delta_J = float(new_J - self.current_J)
        elapsed = time.perf_counter() - t0

        l_aff = (self.H - t_a)
        self.stats.record(elapsed, l_aff)
        return delta_J, new_J

    def accept_candidate(self, candidate_sched: JointSchedule, new_J: float):
        self.current_footprints = candidate_sched.generate_footprint_array()
        u_m, u_p, J = self.engine.compute_forward_trajectory(self.current_footprints)
        self.u_minus_cache = u_m
        self.u_plus_cache = u_p
        self.current_J = new_J


class ForwardBackwardFastEvaluator(BaseEvaluator):
    """Reuses forward mass prefix up to t_a, propagates through [t_a, t_b],
    and stitches with backward continuation V_{t_b+1}.
    """

    def __init__(self, engine: JointMassEngine):
        super().__init__(engine)
        self.stats = EvaluatorStats("ForwardBackwardFastEvaluator")
        self.current_J: float = 0.0
        self.current_footprints: Optional[np.ndarray] = None
        self.u_minus_cache: Optional[np.ndarray] = None
        self.u_plus_cache: Optional[np.ndarray] = None
        self.V_cache: Optional[np.ndarray] = None

    def initialize_base_schedule(self, joint_sched: JointSchedule) -> float:
        self.current_footprints = joint_sched.generate_footprint_array()
        u_m, u_p, J = self.engine.compute_forward_trajectory(self.current_footprints)
        V = self.engine.compute_backward_continuation(self.current_footprints)
        self.u_minus_cache = u_m
        self.u_plus_cache = u_p
        self.V_cache = V
        self.current_J = J
        return J

    def evaluate_candidate(
        self,
        candidate_sched: JointSchedule,
        t_a: int = 0,
        t_b: Optional[int] = None
    ) -> Tuple[float, float]:
        if t_b is None or t_b >= self.H:
            t_b = self.H - 1

        t0 = time.perf_counter()
        cand_fp = candidate_sched.generate_footprint_array()
        u_ta_minus = self.u_minus_cache[t_a] if t_a > 0 else self.engine.u0_minus
        V_tb_plus_1 = self.V_cache[t_b + 1]

        delta_J, new_J = self.engine.fast_evaluate_segment(
            cand_fp[t_a : t_b + 1],
            t_a=t_a,
            t_b=t_b,
            u_ta_minus=u_ta_minus,
            V_tb_plus_1=V_tb_plus_1,
            current_J_star=self.current_J
        )
        elapsed = time.perf_counter() - t0

        l_aff = t_b - t_a + 1
        self.stats.record(elapsed, l_aff)
        return delta_J, new_J

    def accept_candidate(self, candidate_sched: JointSchedule, new_J: float):
        self.current_footprints = candidate_sched.generate_footprint_array()
        u_m, u_p, J = self.engine.compute_forward_trajectory(self.current_footprints)
        V = self.engine.compute_backward_continuation(self.current_footprints)
        self.u_minus_cache = u_m
        self.u_plus_cache = u_p
        self.V_cache = V
        self.current_J = new_J
