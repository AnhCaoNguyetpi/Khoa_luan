"""RQ4 -- Gia tri cua tai dinh tuyen dong: thiet ke 2x2.

                   Static Routing   Dynamic Routing
    Baseline Prob        A                 B
    Data-driven Prob     C                 D

Contrasts:  C-A = value of data-driven belief (static)
            B-A = value of dynamic replanning (baseline belief)
            D-A = combined framework benefit
Plus a stationary-target ablation (E/F) probing when mobility matters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import load_config
from ..sim.metrics import summarize, survival_curve
from ..viz.plots import factorial_chart, survival_curves
from .runner import make_setups, run_arms
from .stats import arm_proportion_table, compare_arms

log = logging.getLogger(__name__)


def run(cfg: Dict | None = None, replicates: int = 24,
        out_dir: str | Path = "results",
        with_stationary_ablation: bool = True, quick: bool = False) -> Dict:
    cfg = load_config(None, cfg or {})
    if quick:
        replicates = max(6, replicates // 3)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    setups = make_setups(cfg, seeds=list(range(5000, 5000 + replicates)))
    arms = {
        "A": {"initial_model": "distance", "planner_cfg": {"kind": "static"}},
        "B": {"initial_model": "distance", "planner_cfg": {"kind": "rolling"}},
        "C": {"initial_model": "datadriven", "planner_cfg": {"kind": "static"}},
        "D": {"initial_model": "datadriven", "planner_cfg": {"kind": "rolling"}},
    }
    rows = run_arms(arms, setups)

    # stationary-target ablation (E/F): does mobility drive the dynamics gain?
    if with_stationary_ablation:
        stat_setups = make_setups(cfg,
                                  seeds=list(range(5000, 5000 + replicates)),
                                  moving=False)
        rows_stat = run_arms({
            "E_static_target": arms["D"] | {"moving_target": False},
            # E re-uses arm D's planner but the target never moves
        }, stat_setups)
        rows_stat["arm"] = "E|stationary-target"
        rows = pd.concat([rows, rows_stat], ignore_index=True)

    rows.to_csv(out / "rq4_rows.csv", index=False)
    summ = rows.groupby("arm").apply(
        lambda g: pd.Series(summarize(g)), include_groups=False)
    print("\n== RQ4: static x dynamic factorial ==")
    cols = [c for c in ["DSR", "EDT_censored", "P(T<=60)", "P(T<=120)",
                        "mean_n_replans", "mean_replan_ms_total",
                        "mean_flight_distance_m", "mean_energy_wh"]
            if c in summ.columns]
    print(summ[cols].round(3).to_string())
    summ.to_csv(out / "rq4_summary.csv")

    s = summ["DSR"].to_dict()
    factorial_chart({k: s.get(k, np.nan) for k in ["A", "B", "C", "D"]},
                    title=f"RQ4 factorial (DSR, n={replicates}/cell)",
                    save=out / "rq4_factorial_dsr.png")
    if all(k in s for k in "ABCD"):
        print(f"\ncontrasts: C-A={s['C']-s['A']:+.3f} "
              f"(data-driven belief, static) | "
              f"B-A={s['B']-s['A']:+.3f} (dynamic, baseline belief) | "
              f"D-A={s['D']-s['A']:+.3f} (full framework)")

    # paired significance tests for the contrasts; E is excluded because its
    # target process differs (stationary), so pairing with A..D is invalid.
    comp = compare_arms(rows, [("A", "B"), ("A", "C"), ("A", "D"),
                               ("B", "D")])
    comp.to_csv(out / "rq4_stats.csv", index=False)
    print("\n-- RQ4 paired statistics (CRN-paired, bootstrap 95% CI) --")
    print(comp.round(4).to_string(index=False))
    prop = arm_proportion_table(rows[rows.arm.isin(["A", "B", "C", "D"])])
    prop.to_csv(out / "rq4_arm_stats.csv", index=False)

    # detection-time survival curves for A-D
    curves = {}
    for a in ["A", "B", "C", "D"]:
        sub = rows[rows.arm == a]
        if len(sub):
            curves[a] = survival_curve(sub)
    survival_curves(curves, title="RQ4: P(not yet detected) vs time",
                    save=out / "rq4_survival.png")
    return {"rows": rows, "summary": summ}
