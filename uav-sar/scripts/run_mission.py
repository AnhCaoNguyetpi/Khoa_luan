"""Run one demo mission and save belief/path snapshots + summary CSV row."""
from _bootstrap import *  # noqa: F401,F403
import argparse

import pandas as pd

from sar_uav.config import load_config
from sar_uav.sim.mission import MissionSetup, run_mission

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--planner", default="rolling",
                    choices=["greedy", "greedy_ratio", "static", "rolling", "milp"])
    ap.add_argument("--belief", default="gis",
                    choices=["uniform", "distance", "gis", "datadriven"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render-dir", default=str(ROOT / "results" / "figures" / "demo"))
    a = ap.parse_args()
    cfg = load_config(a.config, {"belief": {"initial_model": a.belief}})
    setup = MissionSetup.create(cfg, seed=a.seed)
    res = run_mission(setup, planner_cfg={"kind": a.planner},
                      initial_model=a.belief,
                      render_dir=a.render_dir,
                      arm_name=f"{a.planner}-{a.belief}")
    row = pd.DataFrame([res.to_row()])
    out = ROOT / "results" / "data" / "demo_mission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(out, index=False)
    print(res.to_row())
    print(f"\nsaved row -> {out}")
    print(f"snapshots -> {a.render_dir}")
