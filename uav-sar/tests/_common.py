"""Shared fixtures for the test suite (plain functions, no pytest needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sar_uav.data.area import build_synthetic_area

_CACHE = {}


def small_area():
    """Deterministic 24x24 synthetic area shared across test modules."""
    if "area" not in _CACHE:
        _CACHE["area"] = build_synthetic_area("test_area", W=24, H=24,
                                              cell=100.0, seed=11)
    return _CACHE["area"]


def tiny_cfg():
    from copy import deepcopy
    from sar_uav.config import DEFAULT_CONFIG
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["mission"]["horizon_min"] = 60.0
    cfg["detection"]["dwell_ticks"] = 2
    cfg["target"]["burn_min_range"] = [20, 30]
    return cfg
