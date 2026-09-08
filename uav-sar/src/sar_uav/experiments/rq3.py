"""RQ3 -- Gia tri cua dinh tuyen co xet kha nang phat hien.

Assignment criterion ``p_i^t q_ikt`` (detection-aware) versus plain
``p_i^t`` (detection-blind), across fleet compositions (heterogeneous
RGB+thermal vs homogeneous RGB) and vegetation-density levels.
Observations and Bayesian updates always use the real detection model --
only the assignment rule changes.
"""

from __future__ import annotations

import logging
import copy as _copy
from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import DEFAULT_CONFIG, deep_update, load_config
from ..sim.metrics import summarize
from ..uav.platform import fleet_from_config
from ..viz.plots import grouped_bars
from .runner import initial_map, make_setups, run_arms, veg_scaled_area
from .stats import arm_proportion_table, compare_arms

log = logging.getLogger(__name__)


HOMOGENEOUS_RGB_FLEET = [
    {"name": "uav-rgb-1", "sensor": "rgb", "speed_mps": 13.0,
     "battery_wh": 88.0, "fly_power_w": 170.0, "hover_power_w": 190.0,
     "climb_wh_per_m": 0.05, "reserve_frac": 0.15, "swap_minutes": 4.0,
     "allow_recharge": True},
    {"name": "uav-rgb-2", "sensor": "rgb", "speed_mps": 13.0,
     "battery_wh": 88.0, "fly_power_w": 170.0, "hover_power_w": 190.0,
     "climb_wh_per_m": 0.05, "reserve_frac": 0.15, "swap_minutes": 4.0,
     "allow_recharge": True},
]


def run(cfg: Dict | None = None, replicates: int = 24,
        veg_scales=(1.0,), out_dir: str | Path = "results",
        quick: bool = False) -> Dict:
    cfg = load_config(None, cfg or {})
    if quick:
        replicates = max(6, replicates // 3)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for veg_scale in veg_scales:
        area_mod = None if abs(veg_scale - 1.0) < 1e-9 \
            else veg_scaled_area(veg_scale)
        setups = make_setups(cfg, seeds=list(range(4000, 4000 + replicates)),
                             area_mod=area_mod)
        dd = {"initial_model": "datadriven",
              "planner_cfg": {"kind": "rolling"}}
        arms = {
            f"pq|mixed|veg{veg_scale}": dict(dd),
            f"p |mixed|veg{veg_scale}": dict(dd, assign_criterion="p"),
            f"pq|homRGB|veg{veg_scale}": dict(dd,
                                              fleet_cfg=HOMOGENEOUS_RGB_FLEET),
            f"p |homRGB|veg{veg_scale}": dict(dd, assign_criterion="p",
                                              fleet_cfg=HOMOGENEOUS_RGB_FLEET),
        }
        all_rows.append(run_arms(arms, setups))

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(out / "rq3_rows.csv", index=False)
    summ = rows.groupby("arm").apply(
        lambda g: pd.Series(summarize(g)), include_groups=False)
    prop = arm_proportion_table(rows).set_index("arm")
    summ = summ.join(prop[["k", "n", "detected_ci_lo", "detected_ci_hi"]])
    print("\n== RQ3: detection-aware (pq) vs detection-blind (p) assignment ==")
    cols = ["DSR", "detected_ci_lo", "detected_ci_hi", "EDT_censored",
            "mean_flight_distance_m", "mean_energy_wh"]
    cols = [c for c in cols if c in summ.columns]
    print(summ[cols].round(3).to_string())
    summ.to_csv(out / "rq3_summary.csv")

    # paired pq-vs-p comparisons on identical environments per fleet/veg level
    arm_names = set(rows["arm"])
    pairs = [(a.replace("pq|", "p |", 1), a) for a in sorted(arm_names)
             if a.startswith("pq|")
             and a.replace("pq|", "p |", 1) in arm_names]
    comp = compare_arms(rows, pairs)
    comp.to_csv(out / "rq3_stats.csv", index=False)
    det = comp[comp.metric == "detected"]
    print("\n-- RQ3 paired stats pq - p (positive diff favours pq) --")
    print(det[["control", "treatment", "diff", "ci_lo", "ci_hi",
               "p_mcnemar"]].round(4).to_string(index=False))

    plot_df = summ.reset_index()
    plot_df[["criterion", "fleet", "veg"]] = plot_df["arm"].str.split(
        "|", expand=True)
    grouped_bars(plot_df, x="veg", y="DSR", hue="criterion",
                 title="RQ3: DSR of pq- vs p-assignment (per vegetation level)",
                 save=out / "rq3_dsr_by_veg.png")
    grouped_bars(plot_df[plot_df.fleet == "mixed"], x="veg", y="mean_energy_wh",
                 hue="criterion",
                 title="RQ3: energy cost of pq- vs p-assignment (mixed fleet)",
                 save=out / "rq3_energy_by_veg.png")
    return {"rows": rows, "summary": summ}
