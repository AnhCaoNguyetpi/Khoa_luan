"""Batch mission runner with common random numbers.

One ``MissionSetup`` per replicate is shared across all arms, so strategy
comparisons see identical weather/target realisations (variance reduction).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.area import get_or_build_area
from ..sim.mission import MissionSetup, run_mission
from pathlib import Path

log = logging.getLogger(__name__)


def make_setups(cfg: Dict | None, seeds: List[int],
                area_mod: Optional[Callable[[object], object]] = None,
                profile: Optional[str] = None,
                moving: Optional[bool] = None):
    """Build one shared environment per replicate."""
    cfg = load_config(None, cfg or {})
    root = Path(__file__).resolve().parents[3] / "data"
    setups = []
    for s in seeds:
        area = get_or_build_area(cfg["area"]["name"], root, cfg)
        if area_mod is not None:
            area = area_mod(area)
        setups.append(MissionSetup.create(cfg, seed=s, area=area,
                                          profile=profile, moving=moving))
    return setups


def resolve_pmr_file(area_name: str) -> Optional[str]:
    mf = Path(__file__).resolve().parents[3] / "data" / "models" \
        / f"{area_name}_pmr.npz"
    return str(mf) if mf.exists() else None


def initial_map(setup: MissionSetup, model: str = "datadriven",
                model_file: Optional[str] = None) -> np.ndarray:
    """Initial probability map for a given setup's IPP + elapsed time."""
    from ..belief.models import build_initial_belief
    if model == "datadriven":
        model_file = model_file or resolve_pmr_file(setup.area.name)
        if model_file is None:
            raise ValueError("train the PMR model first (RQ1)")
        return build_initial_belief("datadriven", setup.area, setup.ipp_idx,
                                    float(setup.burn_min),
                                    {"model_file": model_file})
    return build_initial_belief(model, setup.area, setup.ipp_idx,
                                float(setup.burn_min))


def run_arms(arms: Dict[str, Dict], setups: List[MissionSetup]) -> pd.DataFrame:
    """Run every arm against every setup; returns mission-level rows.

    An arm's kwargs dict may contain callables ``fn(setup)`` which are
    resolved per replicate -- needed when e.g. ``belief_override`` depends
    on the replicate's IPP.
    """
    rows = []
    for arm_name, kwargs in arms.items():
        for rep, setup in enumerate(setups):
            solved = {k: (v(setup) if callable(v) else v)
                      for k, v in kwargs.items()}
            res = run_mission(setup, arm_name=arm_name, replicate=rep,
                              **solved)
            row = res.to_row()
            rows.append(row)
            log.info("arm %-26s rep %2d/%d det=%s t=%.0f",
                     arm_name, rep + 1, len(setups), res.detected,
                     res.detect_time_censored)
    return pd.DataFrame(rows)


def veg_scaled_area(factor: float):
    """Area modifier scaling vegetation density (RQ3 sweeps)."""
    def _mod(area):
        import copy
        a = copy.copy(area)
        a.veg = np.clip(area.veg * factor, 0.0, 1.0)
        return a
    return _mod
