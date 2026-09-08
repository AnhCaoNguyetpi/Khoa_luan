"""Mission result record and aggregate SAR metrics (proposal section
"Chi so danh gia"): DSR, P(T_detect <= T_max), EDT plus operational costs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MissionResult:
    detected: bool
    detect_time_min: float          # mission-relative; np.nan if not detected
    horizon_min: float

    # operational cost metrics
    flight_distance_m: float = 0.0
    energy_wh: float = 0.0
    search_minutes: float = 0.0
    unique_cells_searched: int = 0
    total_search_ops: int = 0
    redundant_search_ops: int = 0
    n_swaps: int = 0
    n_replans: int = 0
    replan_ms_total: float = 0.0

    # context / metadata (filled by caller)
    planner: str = ""
    initial_model: str = ""
    area_name: str = ""
    seed: int = -1
    replicate: int = -1
    arm: str = ""
    target_profile: str = ""
    elapsed_before_min: float = 0.0
    n_uavs: int = 0
    moving_target: bool = True
    per_uav: Dict = field(default_factory=dict)
    trace: Optional[List[Dict]] = None      # diagnostic only (not in rows)

    @property
    def detect_time_censored(self) -> float:
        """T_detect with right-censoring at the horizon (undetected -> T_max)."""
        return self.detect_time_min if self.detected else self.horizon_min

    def to_row(self) -> Dict:
        d = asdict(self)
        d.pop("per_uav")
        d.pop("trace", None)
        return d


def summarize(rows: List[Dict] or pd.DataFrame) -> Dict[str, float]:
    """Aggregate SAR performance over a batch of missions."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    out: Dict[str, float] = {}
    if len(df) == 0:
        return out
    horizon = df["horizon_min"].max() if "horizon_min" in df else np.nan
    det = df["detected"].astype(bool)
    t = df["detect_time_min"].astype(float).where(det, horizon)
    out["n_missions"] = int(len(df))
    # DSR = N_detected / N_missions
    out["DSR"] = float(det.mean())
    # EDT (censored at horizon for undetected missions)
    out["EDT_censored"] = float(t.mean())
    out["EDT_median"] = float(t.median())
    out["EDT_conditional"] = float(df.loc[det, "detect_time_min"].mean()) \
        if det.any() else float("nan")
    for deadline in (30, 60, 90, 120, 180):
        if np.isfinite(horizon) and deadline <= horizon:
            out[f"P(T<={deadline})"] = float((t <= deadline).mean())
    for col in ("flight_distance_m", "energy_wh", "search_minutes",
                "unique_cells_searched", "redundant_search_ops",
                "n_replans", "replan_ms_total", "n_swaps"):
        if col in df:
            out[f"mean_{col}"] = float(df[col].mean())
    return out


def survival_curve(rows: List[Dict] or pd.DataFrame, step: float = 5.0):
    """(times, fraction_undetected) for detection-time CDF plots."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    horizon = float(df["horizon_min"].max())
    det = df["detected"].astype(bool)
    times = np.arange(0.0, horizon + step, step)
    frac = np.array([1.0 - float(((df.loc[det, "detect_time_min"]) <= x).mean())
                    for x in times])
    base = 1.0 - float(det.mean())      # never-detected mass stays at 1
    frac = np.maximum(frac, 0.0)
    # add censored mass: fraction undetected = 1 - CDF among all
    cdf = np.array([float(((df["detect_time_min"].where(det, horizon)) <= x).mean())
                    for x in times])
    return times, 1.0 - cdf
