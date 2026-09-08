"""Train the data-driven initial-belief model (PMR logistic regression).

Approach (mirrors Potential-Maximum-Response practice in lost-person
behaviour research, cf. Hashimoto et al. 2022 / Seric et al. 2021):

1. generate synthetic SAR incidents: random IPP + behaviour profile +
   elapsed time -> simulated trajectory -> find location;
2. build the per-cell feature table relative to each IPP
   (:func:`sar_uav.belief.models.cell_features`);
3. fit a class-weighted binary logistic regression
   ("does the subject end up in cell i?") with L2 regularisation using
   plain numpy gradient descent -- no extra dependencies;
4. the learned weights define :func:`models.data_driven_belief`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .models import FEATURE_NAMES, cell_features


def fit_pmr_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                     iters: int = 4000, lr: float = 0.5,
                     batch: int = 4096, rng: np.random.Generator | None = None,
                     verbose: bool = False) -> np.ndarray:
    """Weighted BCE logistic regression.

    ``X``: (N, F) already standardised; ``y`` in {0,1} with rare positives.
    Positive class is weighted by n_neg/n_pos so that the model matches the
    probability mass rather than the raw counts.
    """
    rng = rng or np.random.default_rng(0)
    N, F = X.shape
    w = np.zeros(F)
    b0 = 0.0
    pos_w = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    sample_w = np.where(y == 1, pos_w, 1.0)

    def loss():
        z = np.clip(X @ w + b0, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        return -np.mean(sample_w * (y * np.log(p + eps)
                                    + (1 - y) * np.log(1 - p + eps))) \
            + l2 * np.dot(w, w)

    prev = np.inf
    for it in range(iters):
        idx = rng.integers(0, N, size=min(batch, N))
        xb, yb, sb = X[idx], y[idx], sample_w[idx]
        z = np.clip(xb @ w + b0, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = sb * (p - yb)
        gw = xb.T @ g / len(idx) + 2 * l2 * w
        gb = g.mean()
        w -= lr * gw
        b0 -= lr * gb
        if verbose and it % 500 == 0:
            cur = loss()
            print(f"  iter {it:5d}  loss={cur:.6f}")
            if prev - cur < 1e-7:
                break
            prev = cur
    return w, b0


def train_from_incidents(incidents: List, area, rng: np.random.Generator,
                         l2: float = 1e-3, iters: int = 3000,
                         verbose: bool = False):
    """Fit the PMR model on a list of incidents.

    Returns a dict ``{"w", "mean", "std", "feature_names"}`` ready for
    :func:`belief.models.data_driven_belief` and saving to .npz.
    """
    Xs, ys = [], []
    for inc in incidents:
        X = cell_features(area, inc.ipp, inc.elapsed_min)
        y = np.zeros(len(X), dtype=np.float64)
        y[inc.find_cell] = 1.0
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.maximum(std, 1e-6)
    Xs_std = (X - mean) / std
    w, b0 = fit_pmr_logistic(Xs_std, y, l2=l2, iters=iters, rng=rng,
                             verbose=verbose)
    return {"w": w.astype(np.float64),
            "mean": mean.astype(np.float64),
            "std": std.astype(np.float64),
            "intercept": np.float64(b0),
            "feature_names": np.array(FEATURE_NAMES)}
