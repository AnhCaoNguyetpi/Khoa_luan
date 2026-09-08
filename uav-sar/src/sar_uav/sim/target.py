"""Hidden ground-truth lost-person simulation.

Generates the true trajectory ``L_0, L_1, ..., L_T`` from a behaviour
profile.  The trajectory is *never* exposed to the planner -- the planner
only receives the belief ``P^t`` (proposal box: ground truth != belief).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..belief.motion import PROFILES, MotionBetas, MotionModel


def sample_profile(rng: np.random.Generator,
                   probs: Dict[str, float]) -> str:
    names = list(probs)
    p = np.array([probs[n] for n in names], dtype=float)
    p = p / p.sum()
    return str(rng.choice(names, p=p))


class LostPersonSim:
    def __init__(self, area, profile: str, ipp_idx: int,
                 rng: np.random.Generator, moving: bool = True,
                 betas: Optional[MotionBetas] = None):
        self.area = area
        self.profile = profile
        self.betas = betas if betas is not None else PROFILES[profile]
        self.model = MotionModel(area, self.betas)
        self.ipp_idx = int(ipp_idx)
        self.rng = rng
        self.moving = moving and profile != "stationary"

    def full_path(self, burn_min: int, horizon_min: float,
                  rain_fn: Callable[[int], float]) -> np.ndarray:
        """Absolute-tick trajectory: index 0 = moment of disappearance.

        The first ``burn_min`` ticks elapse before the mission starts; the
        position at tick ``burn_min + t`` is the truth during mission tick t.
        """
        total = int(burn_min + horizon_min) + 2
        path = np.empty(total, dtype=np.int64)
        cur = self.ipp_idx
        path[0] = cur
        for i in range(1, total):
            if self.moving:
                cur = self.model.sample_step(self.rng, cur, rain_fn(i))
            path[i] = cur
        return path
