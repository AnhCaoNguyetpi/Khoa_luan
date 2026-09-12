"""Detection hypothesis model and ensemble under persistent uncertainty.

Corresponds to proposal section "Bat dinh mo hinh phat hien va tap gia thuyet":
    S = {q^s, w_s}_{s=1}^S
with log-linear vegetation-structured error:
    log lambda_i^s = log hat_lambda_i + eta_{g(i)}^s
and exponential exposure detection probability per tick:
    q_{ki}^s(a_{kt}) = 1 - exp(-lambda_i^s * F_{ki}(a_{kt}) * delta_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import numpy as np


@dataclass
class DetectionHypothesis:
    """A single persistent detection hypothesis s in S."""
    id: int
    name: str
    weight: float
    # Per-cell detection rates lambda_i^s (shape: [num_cells])
    lambda_rates: np.ndarray
    delta_t: float = 60.0  # seconds per tick

    def __post_init__(self):
        self.lambda_rates = np.asarray(self.lambda_rates, dtype=np.float64)
        if self.lambda_rates.ndim != 1:
            raise ValueError(
                f"DetectionHypothesis {self.id} ('{self.name}'): lambda_rates must be a 1D array, "
                f"got ndim={self.lambda_rates.ndim}."
            )
        if len(self.lambda_rates) == 0:
            raise ValueError(
                f"DetectionHypothesis {self.id} ('{self.name}'): lambda_rates cannot be empty."
            )
        if not np.all(np.isfinite(self.lambda_rates)):
            raise ValueError(
                f"DetectionHypothesis {self.id} ('{self.name}'): lambda_rates must be finite (no NaN or Inf)."
            )
        if np.any(self.lambda_rates < 0.0):
            min_rate = float(np.min(self.lambda_rates))
            raise ValueError(
                f"DetectionHypothesis {self.id} ('{self.name}'): lambda_rates must be non-negative, got min={min_rate}."
            )
        if self.delta_t <= 0.0 or not np.isfinite(self.delta_t):
            raise ValueError(
                f"DetectionHypothesis {self.id} ('{self.name}'): delta_t must be strictly positive and finite, got {self.delta_t}."
            )

    def q_single_tick(self, cell_idx: int, footprint_coverage: float = 1.0) -> float:
        """Single-tick detection probability for a specific cell."""
        rate = float(self.lambda_rates[cell_idx])
        if rate <= 0.0 or footprint_coverage <= 0.0:
            return 0.0
        exponent = -rate * footprint_coverage * (self.delta_t / 60.0)
        return float(1.0 - np.exp(exponent))

    def q_all_cells(self, footprint_coverages: np.ndarray) -> np.ndarray:
        """Vectorized single-tick detection probability across all cells.
        footprint_coverages: shape [num_cells], in [0, 1].
        """
        rates = self.lambda_rates * footprint_coverages * (self.delta_t / 60.0)
        return 1.0 - np.exp(-rates)


class HypothesisSet:
    """Ensemble of detection hypotheses S = {q^s, w_s}_{s=1}^S."""

    def __init__(self, hypotheses: Sequence[DetectionHypothesis]):
        if not hypotheses:
            raise ValueError("HypothesisSet must contain at least one hypothesis.")
        self.hypotheses: List[DetectionHypothesis] = list(hypotheses)
        self.S: int = len(self.hypotheses)
        self.num_cells: int = len(self.hypotheses[0].lambda_rates)
        
        # Verify consistent cell counts
        for h in self.hypotheses:
            if len(h.lambda_rates) != self.num_cells:
                raise ValueError(f"Hypothesis {h.id} has {len(h.lambda_rates)} cells, expected {self.num_cells}")

        # Verify non-negative weights and positive sum
        raw_weights = np.array([h.weight for h in self.hypotheses], dtype=np.float64)
        if np.any(raw_weights < 0.0):
            raise ValueError("Hypothesis weights must be non-negative.")
        total_w = np.sum(raw_weights)
        if total_w <= 0.0 or not np.isfinite(total_w):
            raise ValueError("Total hypothesis weight must be strictly positive and finite.")
        self.weights = raw_weights / total_w
        for i, h in enumerate(self.hypotheses):
            h.weight = float(self.weights[i])

        # Cache lambda rates matrix of shape [S, num_cells]
        self.rates_matrix = np.zeros((self.S, self.num_cells), dtype=np.float64)
        for s_idx, h in enumerate(self.hypotheses):
            self.rates_matrix[s_idx, :] = h.lambda_rates

    @classmethod
    def create_nominal(cls, nominal_rates: np.ndarray, delta_t: float = 60.0) -> "HypothesisSet":
        """Create a single-hypothesis set representing the nominal model (S=1)."""
        h0 = DetectionHypothesis(
            id=0,
            name="nominal",
            weight=1.0,
            lambda_rates=np.asarray(nominal_rates, dtype=np.float64),
            delta_t=delta_t
        )
        return cls([h0])

    @classmethod
    def create_structured_ensemble(
        cls,
        nominal_rates: np.ndarray,
        vegetation_groups: np.ndarray,
        uncertainty_scale: float = 0.5,
        num_hypotheses: int = 5,
        seed: int = 42,
        delta_t: float = 60.0
    ) -> "HypothesisSet":
        """Create an ensemble with spatial structured deviations per vegetation group:
            log lambda_i^s = log hat_lambda_i + eta_{g(i)}^s
        """
        nominal = np.asarray(nominal_rates, dtype=np.float64)
        nominal = np.maximum(nominal, 1e-6)
        log_nominal = np.log(nominal)
        veg_groups = np.asarray(vegetation_groups, dtype=np.int64)
        unique_groups = np.unique(veg_groups)

        rng = np.random.RandomState(seed)
        hypotheses = []

        # Hypothesis 0 is nominal centered
        h0 = DetectionHypothesis(
            id=0,
            name="nominal_baseline",
            weight=1.0 / num_hypotheses,
            lambda_rates=nominal.copy(),
            delta_t=delta_t
        )
        hypotheses.append(h0)

        for s in range(1, num_hypotheses):
            # Sample group perturbations
            group_offsets = {}
            for g in unique_groups:
                group_offsets[g] = rng.normal(0.0, uncertainty_scale)

            # Apply to each cell
            eta = np.array([group_offsets[g] for g in veg_groups], dtype=np.float64)
            perturbed_rates = np.exp(log_nominal + eta)
            
            hypotheses.append(DetectionHypothesis(
                id=s,
                name=f"hypothesis_{s}",
                weight=1.0 / num_hypotheses,
                lambda_rates=perturbed_rates,
                delta_t=delta_t
            ))

        return cls(hypotheses)

    def compute_all_negative_likelihood(
        self,
        footprints: np.ndarray,
        delta_t: float = 60.0
    ) -> np.ndarray:
        """Compute likelihood of non-detection for all cells and all hypotheses at one tick.
        footprints: shape [m, num_cells] or [num_cells] indicating optical coverage of UAVs.
        Returns: ell^s(i, a_t) of shape [S, num_cells].
        
        ell^s(i, a_t) = prod_{k=1}^m [1 - q_{ki}^s(a_{kt})]
                      = prod_{k=1}^m exp(-lambda_i^s * F_{ki}(a_{kt}) * delta_t/60)
                      = exp( -lambda_i^s * (sum_k F_{ki}(a_{kt})) * delta_t/60 )
        """
        # Sum coverages across UAVs
        if footprints.ndim == 2:
            total_coverage = np.sum(footprints, axis=0)  # [num_cells]
        else:
            total_coverage = footprints

        # rates_matrix: [S, num_cells]
        factor = (delta_t / 60.0)
        # exponent: [S, num_cells]
        exponent = -self.rates_matrix * total_coverage[None, :] * factor
        return np.exp(exponent)
