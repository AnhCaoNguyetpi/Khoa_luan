"""Joint belief and unnormalized mass engine for multi-hypothesis search.

Corresponds to proposal section "Cap nhat Joint Belief va Danh gia Lich hanh dong":
    - Initial joint belief / unnormalized mass: u_0^-(i, s) = b_0(i) * w_s
    - Non-detection update: u_t^+(i, s) = u_t^-(i, s) * ell_t^s(i, a_t)
    - Motion propagation: u_{t+1}^-(j, s) = sum_i u_t^+(i, s) * M(i, j)
    - Backward continuation V_t(i, s) with boundary V_H(i, s) = 1.0:
      V_t(i, s) = ell_t^s(i, a_t) * sum_j M(i, j) * V_{t+1}(j, s)
    - Objective function: J_S(a) = 1 - sum_{i, s} u_{H-1}^+(i, s)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from sar_uav.detection.hypothesis import HypothesisSet


class JointMassEngine:
    """Engine computing forward unnormalized mass and backward continuation."""

    def __init__(
        self,
        b0: np.ndarray,
        hypothesis_set: HypothesisSet,
        motion_matrix: np.ndarray,
        H: int,
        delta_t: float = 60.0
    ):
        """
        b0: initial target location belief, shape [num_cells], sum(b0) == 1.
        hypothesis_set: HypothesisSet with S hypotheses.
        motion_matrix: Markov transition matrix M, shape [num_cells, num_cells], row stochastic.
        H: planning horizon in ticks.
        delta_t: seconds per tick.
        """
        b0_arr = np.asarray(b0, dtype=np.float64)
        if b0_arr.ndim != 1:
            raise ValueError(f"Prior b0 must be a 1D array, got ndim={b0_arr.ndim}")
        if len(b0_arr) == 0:
            raise ValueError("Prior b0 cannot be empty.")
        if not np.all(np.isfinite(b0_arr)):
            raise ValueError("Prior b0 must contain finite values (no NaN or Inf).")
        if np.any(b0_arr < 0.0):
            raise ValueError("Prior b0 must be non-negative.")
        sum_b0 = np.sum(b0_arr)
        if sum_b0 <= 0.0 or not np.isfinite(sum_b0):
            raise ValueError("Prior b0 must have positive, finite sum.")
        self.num_cells = len(b0_arr)
        self.b0 = b0_arr / sum_b0

        if len(hypothesis_set.hypotheses) == 0:
            raise ValueError("hypothesis_set must contain at least one hypothesis.")
        hypo_num_cells = hypothesis_set.hypotheses[0].lambda_rates.shape[0]
        if self.num_cells != hypo_num_cells:
            raise ValueError(
                f"Prior b0 length ({self.num_cells}) does not match hypothesis cell count ({hypo_num_cells})"
            )

        if not isinstance(H, (int, np.integer)) or H <= 0:
            raise ValueError(f"Planning horizon H must be a positive integer, got {H}")
        if not (np.isfinite(delta_t) and delta_t > 0.0):
            raise ValueError(f"delta_t must be finite and positive, got {delta_t}")

        self.hypothesis_set = hypothesis_set
        self.S = hypothesis_set.S
        self.H = int(H)
        self.delta_t = float(delta_t)

        self.M = np.asarray(motion_matrix, dtype=np.float64)
        if self.M.shape != (self.num_cells, self.num_cells):
            raise ValueError(f"Motion matrix shape {self.M.shape} does not match ({self.num_cells}, {self.num_cells})")
        if np.any(self.M < 0.0):
            raise ValueError("Motion matrix M entries must be non-negative.")
        row_sums = np.sum(self.M, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-5):
            raise ValueError(f"Motion matrix M must be row-stochastic (row sums must equal 1.0, got {row_sums})")
        # Enforce exact row normalization so probability is strictly conserved across long horizons
        self.M = self.M / row_sums[:, np.newaxis]

        # Transpose of M for backward continuation: (V * M^T)
        self.M_T = self.M.T.copy()

        # Initial forward mass u0^-(s, i) = w_s * b0(i), shape [S, num_cells]
        self.u0_minus = np.outer(self.hypothesis_set.weights, self.b0)

    def compute_forward_trajectory(
        self,
        footprints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute full forward unnormalized mass trajectory for a schedule.
        footprints: shape [H, num_cells] or [H, m, num_cells]
        
        Returns:
            u_minus: shape [H, S, num_cells]
            u_plus: shape [H, S, num_cells]
            J_S: cumulative probability of detection
        """
        u_minus = np.zeros((self.H, self.S, self.num_cells), dtype=np.float64)
        u_plus = np.zeros((self.H, self.S, self.num_cells), dtype=np.float64)

        u_current_minus = self.u0_minus.copy()

        for t in range(self.H):
            u_minus[t] = u_current_minus
            # Non-detection likelihood ell_t^s(i, a_t): shape [S, num_cells]
            fp_t = footprints[t]
            ell_t = self.hypothesis_set.compute_all_negative_likelihood(fp_t, delta_t=self.delta_t)
            
            # u_t^+(i, s) = u_t^-(i, s) * ell_t^s(i, a_t)
            u_current_plus = u_current_minus * ell_t
            u_plus[t] = u_current_plus

            # Motion propagation: u_{t+1}^- = u_t^+ * M
            if t + 1 < self.H:
                # u_current_plus: [S, num_cells], M: [num_cells, num_cells] -> [S, num_cells]
                u_current_minus = np.dot(u_current_plus, self.M)

        # Objective value J_S = 1 - sum_{i, s} u_{H-1}^+(i, s)
        total_fail_mass = float(np.sum(u_plus[self.H - 1]))
        if np.all(footprints == 0):
            J_S = 0.0
        else:
            J_S = float(np.clip(1.0 - total_fail_mass, 0.0, 1.0))
        return u_minus, u_plus, J_S

    def compute_backward_continuation(
        self,
        footprints: np.ndarray
    ) -> np.ndarray:
        """Compute backward continuation V_t(i, s) for t = 0..H.
        V_H(i, s) = 1.0
        V_t(i, s) = ell_t^s(i, a_t) * sum_j M(i, j) * V_{t+1}(j, s)
        
        Returns:
            V: shape [H + 1, S, num_cells]
        """
        V = np.zeros((self.H + 1, self.S, self.num_cells), dtype=np.float64)
        # Boundary condition V_H = 1.0
        V[self.H, :, :] = 1.0

        for t in range(self.H - 1, -1, -1):
            fp_t = footprints[t]
            ell_t = self.hypothesis_set.compute_all_negative_likelihood(fp_t, delta_t=self.delta_t)
            # sum_j M(i, j) * V_{t+1}(j, s):
            # V_{t+1}: [S, num_cells], M_T: [num_cells, num_cells]
            # V_{t+1} * M_T -> [S, num_cells]
            future_expected_V = np.dot(V[t + 1], self.M_T)
            V[t] = ell_t * future_expected_V

        return V

    def fast_evaluate_segment(
        self,
        new_footprints_segment: np.ndarray,
        t_a: int,
        t_b: int,
        u_ta_minus: np.ndarray,
        V_tb_plus_1: np.ndarray,
        current_J_star: float
    ) -> Tuple[float, float]:
        """Fast evaluation for schedule modification localized to [t_a, t_b].
        
        new_footprints_segment: shape [t_b - t_a + 1, num_cells] (or with m dim)
        u_ta_minus: forward unnormalized mass at t_a, shape [S, num_cells]
        V_tb_plus_1: backward continuation at t_b + 1, shape [S, num_cells]
        
        Returns:
            delta_J: improvement over current_J_star
            new_J: new objective value
        """
        u_curr_minus = u_ta_minus.copy()
        seg_len = t_b - t_a + 1

        for step in range(seg_len):
            t = t_a + step
            fp_t = new_footprints_segment[step]
            ell_t = self.hypothesis_set.compute_all_negative_likelihood(fp_t, delta_t=self.delta_t)
            u_curr_plus = u_curr_minus * ell_t

            if step + 1 < seg_len:
                u_curr_minus = np.dot(u_curr_plus, self.M)
            else:
                # Reached t_b, propagate forward one step to t_b + 1 to meet V_{t_b + 1}
                u_tb_plus_1_minus = np.dot(u_curr_plus, self.M)

        # Stitching formula: P_fail(a_tilde) = sum_{j, s} u_{t_b + 1}^-(j, s) * V_{t_b + 1}(j, s)
        # If t_b == H - 1, V_H is identically 1.0, so sum(u_H^-) == sum(u_{H-1}^+)
        if t_b == self.H - 1:
            P_fail = float(np.sum(u_curr_plus))
        else:
            P_fail = float(np.sum(u_tb_plus_1_minus * V_tb_plus_1))

        new_J = float(1.0 - P_fail)
        delta_J = float(new_J - current_J_star)
        return delta_J, new_J

    def get_normalized_joint_belief(self, u_mass: np.ndarray) -> np.ndarray:
        """Compute normalized joint belief beta(i, s) from unnormalized mass.
        u_mass: shape [S, num_cells]
        Returns: beta of shape [S, num_cells], sum(beta) == 1.
        """
        total = np.sum(u_mass)
        if total <= 1e-15:
            return np.ones_like(u_mass) / u_mass.size
        return u_mass / total
