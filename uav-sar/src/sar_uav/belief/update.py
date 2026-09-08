"""Bayesian belief update after an unsuccessful search (negative observation).

Proposal formula (single cell)::

    p_i^{t+} = p_i^t (1 - q) / (1 - p_i^t q),      p_j^{t+} = p_j^t / (1 - p_i^t q)

generalised to a set of simultaneously searched cells (disjoint), which is
the product-form update with denominator ``D = 1 - sum_i p_i q_i``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def combine_q(qs) -> float:
    """Independent-sensor combination of per-UAV detection probabilities."""
    qs = np.asarray(qs, dtype=float)
    return float(1.0 - np.prod(1.0 - np.clip(qs, 0.0, 1.0)))


def bayes_negative_update(p: np.ndarray, cells, qs) -> np.ndarray:
    """Update belief after searching ``cells`` with detection probs ``qs``
    and finding nothing.

    Duplicate cells are merged first (combined-q rule).  Returns a new
    normalised vector; the input is not modified.
    """
    p = np.asarray(p, dtype=float)
    out = p.copy()
    cells = np.asarray(cells, dtype=np.int64).reshape(-1)
    qs = np.clip(np.asarray(qs, dtype=float).reshape(-1), 0.0, 1.0)

    # merge duplicates
    uniq, inv = np.unique(cells, return_inverse=True)
    qm = np.zeros(len(uniq))
    for k, i in enumerate(inv):
        qm[i] = combine_q([qm[i], qs[k]])

    denom = 1.0 - float(np.dot(out[uniq], qm))
    denom = max(denom, _EPS)
    out[uniq] *= (1.0 - qm)
    out /= denom
    s = out.sum()
    if abs(s - 1.0) > 1e-6:      # numerical safety renormalisation
        out /= s
    return out
