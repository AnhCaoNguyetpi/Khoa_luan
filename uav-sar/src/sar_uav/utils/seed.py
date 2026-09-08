"""Reproducible seeding helpers.

Experiments use *common random numbers*: for replicate ``r`` the environment
(weather + target trajectory) is generated from seeds that do not depend on
the strategy under test, so arm comparisons are variance-reduced.
"""

from __future__ import annotations

import numpy as np


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent generators from one master seed."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(s) for s in children]


def derive_seed(seed: int, *tags) -> int:
    """Deterministically derive a sub-seed from a master seed plus tags."""
    ss = np.random.SeedSequence([seed] + [int(t) & 0xFFFFFFFF for t in tags])
    return int(ss.generate_state(1)[0])
