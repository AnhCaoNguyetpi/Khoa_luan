"""Central configuration.

Configuration is a plain nested dict (JSON-serialisable) merged over
``DEFAULT_CONFIG``.  Every experiment / mission script accepts a config file
plus CLI overrides; see ``scripts/`` for usage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # ------------------------------------------------------------------ area
    "area": {
        "name": "tay_nguyen_real",
        "grid_w": 40,
        "grid_h": 40,
        "cell_size_m": 100.0,
        "seed": 7,
        # data directory root holding data/areas/<name>/
        "data_dir": None,          # None -> <project_root>/data
    },
    # --------------------------------------------------------------- weather
    "weather": {
        "cloud_mean": 0.35,        # mean cloud cover in [0, 1]
        "rain_episode_prob": 0.02, # per-tick prob of a rain episode starting
        "rain_episode_len": (20, 60),   # minutes
        "rain_peak_range": (0.3, 0.9),
        "wind_base_ms": 3.0,
        "temp_base_c": 18.0,
        "diurnal_amplitude_c": 6.0,
    },
    # ----------------------------------------------------------------- fleet
    # Two UAVs: one RGB mapper + one thermal -- the heterogeneous baseline of RQ3.
    "fleet": [
        {
            "name": "uav-rgb",
            "sensor": "rgb",
            "speed_mps": 13.0,
            "battery_wh": 88.0,
            "fly_power_w": 170.0,
            "hover_power_w": 190.0,
            "climb_wh_per_m": 0.05,
            "reserve_frac": 0.15,
            "swap_minutes": 4.0,
            "allow_recharge": True,
            "base_cell": None,     # None -> auto (trailhead near a map corner)
        },
        {
            "name": "uav-thermal",
            "sensor": "thermal",
            "speed_mps": 12.0,
            "battery_wh": 92.0,
            "fly_power_w": 175.0,
            "hover_power_w": 205.0,
            "climb_wh_per_m": 0.05,
            "reserve_frac": 0.15,
            "swap_minutes": 4.0,
            "allow_recharge": True,
            "base_cell": None,
        },
    ],
    # ---------------------------------------------------------------- belief
    "belief": {
        # uniform | distance | gis | datadriven | oracle
        "initial_model": "gis",
        "sigma_per_min": 25.0,         # DistanceGaussian spread (m per elapsed min)
        "model_file": None,            # path to trained datadriven weights (.npz)
        "forecast_enabled": True,      # propagate belief with motion model each tick
    },
    # ---------------------------------------------------------------- target
    "target": {
        # true behaviour profile sampling weights
        "profile_probs": {"hiker": 0.40, "desoriented": 0.25,
                          "injured": 0.15, "photographer": 0.20},
        "moving": True,                # False -> stationary-target ablation
        "burn_min_range": [30, 90],    # elapsed time before mission start
        # planner's *estimated* motion betas are perturbed by +-beta_bias (RQ5)
        "beta_bias": 0.0,
    },
    # -------------------------------------------------------------- detection
    "detection": {
        "dwell_ticks": 3,              # search time per cell (dt=1 min ticks)
        # sensor footprint wider than one grid cell: neighbours are covered
        # at reduced probability (realistic FOV vs. grid resolution)
        "footprint_side": 0.45,
    },
    # --------------------------------------------------------------- planner
    "planner": {
        # greedy | greedy_ratio | static | rolling | milp
        "kind": "rolling",
        "top_k": 48,                   # candidate cells considered per replan
        "lambda_travel_km": 0.010,     # travel penalty per km leg added
        "lambda_energy": 0.0,          # optional battery-fraction penalty
        "overlap_kappa": 0.7,          # gain decay per previous visit of a cell
        "lookahead_cells": 3,          # cells per cycle for rolling dispatch
        "replan_interval_min": 15.0,   # reserved for interval-based variants
        "gain_bucket_min": 5.0,        # diffusion bucket for arrival-time gains
        "milp_time_limit_s": 10.0,
        "force_dispatch": True,        # never leave an idle UAV parked while
                                       # positive-value cells remain reachable
    },
    # ---------------------------------------------------------------- mission
    "mission": {
        "horizon_min": 180.0,          # mission time limit
        "dt_min": 1.0,                 # decision tick length
        "objective_lambdas": {"travel_km": 0.002, "energy_norm": 0.05},
    },
    # ------------------------------------------------------------- simulation
    "sim": {
        "seed": 1234,                  # master seed for env/target/observations
        "render_dir": None,            # if set, save belief/path snapshots here
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns new dict)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None,
                overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a JSON config over :data:`DEFAULT_CONFIG`, apply CLI overrides."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, json.load(f))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def project_root() -> Path:
    """Return the uav-sar project root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def data_root(cfg: Dict[str, Any] | None = None) -> Path:
    """Root directory holding ``areas/`` and ``models/``."""
    if cfg and cfg.get("area", {}).get("data_dir"):
        return Path(cfg["area"]["data_dir"])
    return project_root() / "data"


def area_dir(name: str, cfg: Dict[str, Any] | None = None) -> Path:
    return data_root(cfg) / "areas" / name


def models_dir() -> Path:
    return data_root() / "models"
