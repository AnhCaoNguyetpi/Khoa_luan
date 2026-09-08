"""Lost-person movement model (Markov random walk on the grid).

Proposal section "Mo rong sang nguoi mat tich co kha nang di chuyen"::

    M_ij = b_trail * TrailProx_j - b_slope * Slope_j - b_veg * Veg_j
           - b_dist * Dist_ij + b_view * Rugged_j - b_rain * Rain_t
    P(j | i) = softmax over {stay} U neighbours(i)

The same class serves two roles with different beta vectors:

* ground truth -- samples the real trajectory (hidden from the planner);
* planner estimate -- used for belief forecasting and route gains
  (nominally identical; perturbed in the RQ5 robustness study).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MotionBetas:
    trail: float = 1.0        # attraction to proximity of trails
    slope: float = 0.8        # aversion to steep cells (per ~45deg normalised)
    veg: float = 0.3          # aversion to thick vegetation
    dist: float = 1.5         # distance decay per 0.5 km step
    rain: float = 1.0         # rain discourages moving
    stay: float = 0.8         # logit weight of staying put
    view: float = 0.0         # attraction to viewpoints (rugged high ground)


#: behaviour profiles in the spirit of lost-person taxonomy (ISRID categories)
PROFILES: Dict[str, MotionBetas] = {
    # walks far along trails, tolerates slope
    "hiker": MotionBetas(trail=2.0, slope=0.8, veg=0.3, dist=1.5,
                         rain=1.0, stay=0.8, view=0.0),
    # wanders off-trail, low directionality
    "desoriented": MotionBetas(trail=0.3, slope=0.5, veg=0.1, dist=0.9,
                               rain=0.6, stay=0.4, view=0.0),
    # barely moves (immobilised / exhausted)
    "injured": MotionBetas(trail=0.6, slope=1.4, veg=0.4, dist=2.4,
                           rain=1.5, stay=2.4, view=0.0),
    # seeks scenic high points, moderate movement
    "photographer": MotionBetas(trail=0.9, slope=0.35, veg=0.25, dist=1.2,
                                rain=0.8, stay=1.1, view=1.6),
}


def perturb_betas(b: MotionBetas, bias: float) -> MotionBetas:
    """Multiplicative model-error perturbation used in RQ5 (bias in [-0.5, 0.5])."""
    if bias == 0.0:
        return b
    kw = {}
    sign = 1.0
    for i, f in enumerate(fields(b)):
        v = getattr(b, f.name)
        if f.name in ("stay",):
            sign = -sign                      # alternate signs for variety
        kw[f.name] = max(0.0, v * (1.0 + bias * sign))
        sign = 1.0
    return MotionBetas(**kw)


class MotionModel:
    """Markov transition model on non-water cells (land-local indexing)."""

    def __init__(self, area, betas: MotionBetas):
        self.area = area
        self.b = betas
        H, W = area.shape

        self.land = area.land_idx()                       # flat idx of land cells
        self.n_land = len(self.land)
        pos = -np.ones(area.n_cells, dtype=np.int64)
        pos[self.land] = np.arange(self.n_land)
        self.local_of_full = pos                          # full->local (-1 water)

        rr, cc = np.divmod(self.land, W)
        # 8-neighbourhood, row-major order then stay appended last
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
        nb = np.full((self.n_land, len(offs)), -1, dtype=np.int64)
        dstep = np.zeros((self.n_land, len(offs)))
        for m, (dr, dc) in enumerate(offs):
            nr, nc = rr + dr, cc + dc
            ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            flat = np.where(ok, nr * W + nc, -1)
            valid = ok & (pos[np.clip(flat, 0, None)] >= 0)
            nb[valid, m] = pos[np.clip(flat, 0, None)[valid]]
            dstep[:, m] = np.hypot(dr, dc) * area.cell

        self.nb = nb                                       # -1 padded
        self.dstep = dstep                                 # metres (0 for pads)

        xy = area.centers_xy()[self.land]
        self.xy = xy
        # per-cell static score components (moves only)
        self.trail_prox = np.exp(-area.dist_trail.reshape(-1)[self.land] / 300.0)
        self.slope_n = np.clip(area.slope.reshape(-1)[self.land] / 45.0, 0, 2)
        self.veg_n = area.veg.reshape(-1)[self.land]
        self.view_n = np.clip(area.rugged.reshape(-1)[self.land]
                              / max(area.rugged.max(), 1e-6), 0, 1)
        self._T_cache: Dict[float, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _move_scores(self, rain: float) -> np.ndarray:
        b = self.b
        base = (b.trail * self.trail_prox - b.slope * self.slope_n
                - b.veg * self.veg_n + b.view * self.view_n)
        s = np.empty((self.n_land, self.nb.shape[1] + 1))
        s[:, :-1] = base[:, None] \
            - b.dist * (self.dstep / 500.0) \
            - b.rain * rain
        s[:, -1] = b.stay                                  # stay column
        pad = np.zeros(s.shape, dtype=bool)
        pad[:, : self.nb.shape[1]] = self.nb < 0           # pads -> impossible
        s[pad] = -np.inf
        return s

    def _softmax_rows(self, s: np.ndarray) -> np.ndarray:
        z = s - s.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def transition_matrix(self, rain: float = 0.0) -> np.ndarray:
        """Dense (n_land, n_land) transition matrix, rows sum to 1."""
        key = round(float(rain), 2)
        T = self._T_cache.get(key)
        if T is None:
            P = self._softmax_rows(self._move_scores(key))
            T = np.zeros((self.n_land, self.n_land))
            rows = np.arange(self.n_land)
            nb_safe = np.where(self.nb < 0, rows[:, None], self.nb)  # pads->own
            np.add.at(T,
                      (np.repeat(rows, self.nb.shape[1]), nb_safe.ravel()),
                      P[:, :self.nb.shape[1]].ravel())
            T[rows, rows] += P[:, -1]                        # stay
            self._T_cache[key] = T
        return T

    # ------------------------------------------------------------------
    def to_local(self, p_full: np.ndarray) -> np.ndarray:
        return np.asarray(p_full)[self.land]

    def to_full(self, p_local: np.ndarray) -> np.ndarray:
        out = np.zeros(self.area.n_cells)
        out[self.land] = p_local
        return out

    def forecast(self, p_full: np.ndarray, rain: float = 0.0) -> np.ndarray:
        """One-step belief propagation: p^{t+1}_j = sum_i P(j|i) p_i^t."""
        pl = self.to_local(p_full)
        pl = self.transition_matrix(rain).T @ pl
        pl = np.maximum(pl, 0)
        s = pl.sum()
        if s <= 0:
            return self.to_full(np.full(self.n_land, 1.0 / self.n_land))
        return self.to_full(pl / s)

    def step_probs(self, idx_full: int, rain: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """(options_full_idx, probs) for the cell ``idx_full`` (stay first)."""
        k = self.local_of_full[idx_full]
        if k < 0:
            raise ValueError("cannot move on a water cell")
        P = self._softmax_rows(self._move_scores(rain))[k]
        valid = self.nb[k] >= 0
        # nb stores land-LOCAL indices -> map back to full grid indices
        opts = np.concatenate([[idx_full], self.land[self.nb[k][valid]]])
        pr = np.concatenate([[P[-1]], P[:-1][valid]])
        return opts, pr / pr.sum()

    def sample_step(self, rng: np.random.Generator, idx_full: int,
                    rain: float = 0.0) -> int:
        opts, pr = self.step_probs(idx_full, rain)
        return int(rng.choice(opts, p=pr))
