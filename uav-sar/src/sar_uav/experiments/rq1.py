"""RQ1 -- Gia tri cua du bao vi tri dua tren du lieu.

    Uniform -> Distance-based -> GIS-based -> Data-driven

Trains the PMR logistic model on synthetic incidents, evaluates all four
initial-belief models on held-out incidents (prediction metrics), then runs
an operational preview (same rolling planner fed by each map).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..belief.models import (build_initial_belief, data_driven_belief,
                             distance_gaussian_belief, gis_expert_belief,
                             uniform_belief)
from ..belief.train import train_from_incidents
from ..config import load_config
from ..viz.maps import compare_initial_beliefs
from ..viz.plots import grouped_bars, sweep_lines
from .belief_eval import evaluate_model, summarize_prediction
from .datasets import generate_incidents
from .runner import make_setups, run_arms
from .stats import arm_proportion_table, compare_arms

log = logging.getLogger(__name__)


def run(cfg: Dict | None = None, replicates: int = 24,
        n_train: int = 250, n_test: int = 80,
        out_dir: str | Path = "results", quick: bool = False) -> Dict:
    cfg = load_config(None, cfg or {})
    out = Path(out_dir)
    from ..data.area import get_or_build_area
    area = get_or_build_area(cfg["area"]["name"],
                             Path(__file__).resolve().parents[3] / "data", cfg)
    if quick:
        n_train, n_test, replicates = 60, 30, 8

    # ---------------------------------------------------------- train PMR
    log.info("generating %d training incidents ...", n_train)
    train = generate_incidents(cfg, area, n_train, seed=9000 + 1)
    pmr = train_from_incidents(train, area,
                               rng=np.random.default_rng(77))
    models_dir = Path(__file__).resolve().parents[3] / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{area.name}_pmr.npz"
    np.savez(model_file, **pmr)
    log.info("saved %s", model_file)

    # ------------------------------------------------------------- predict
    log.info("evaluating on %d held-out incidents ...", n_test)
    test = generate_incidents(cfg, area, n_test, seed=9000 + 2)

    def b_uniform(inc):
        return uniform_belief(area)

    def b_distance(inc):
        return distance_gaussian_belief(area, inc.ipp, inc.elapsed_min,
                sigma_per_min=cfg["belief"]["sigma_per_min"])

    def b_gis(inc):
        return gis_expert_belief(area, inc.ipp, inc.elapsed_min)

    def b_dd(inc):
        return data_driven_belief(area, inc.ipp, inc.elapsed_min,
                                  {"w": pmr["w"], "mean": pmr["mean"],
                                   "std": pmr["std"]})

    builders = [("uniform", b_uniform), ("distance", b_distance),
                ("gis", b_gis), ("datadriven", b_dd)]

    frames = [evaluate_model(nm, bd, area, test) for nm, bd in builders]
    pred = pd.concat(frames, ignore_index=True)
    summary = summarize_prediction(pred)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "rq1_prediction_rows.csv", index=False)
    summary.to_csv(out / "rq1_prediction_summary.csv")
    print("\n== RQ1 prediction quality (held-out incidents) ==")
    print(summary.round(4).to_string())

    # example heatmaps for one incident
    inc0 = test[0]
    maps = {nm: bd(inc0) for nm, bd in builders}
    compare_initial_beliefs(area, maps, ipp_cell=inc0.ipp,
                            find_cell=inc0.find_cell,
                            elapsed_min=inc0.elapsed_min,
                            save=out / "rq1_initial_maps_example.png")

    mass_cols = [c for c in summary.columns if c.startswith("mass_")]
    fig = sweep_lines(summary.reset_index().melt(id_vars="model",
                      value_vars=mass_cols, var_name="radius",
                      value_name="mass"),
                      x="radius", y="mass", hue="model",
                      title="RQ1: probability mass near the true location",
                      xlabel="radius around truth",
                      save=out / "rq1_mass_vs_radius.png")
    rec_cols = [c for c in summary.columns if c.startswith("recall_")]
    grouped_bars(summary.reset_index().melt(id_vars="model",
                 value_vars=rec_cols, var_name="topk", value_name="recall"),
                 x="topk", y="recall", hue="model",
                 title="RQ1: Top-k spatial recall",
                 save=out / "rq1_topk_recall.png")

    # ------------------------------------------- operational preview (->RQ2)
    setups = make_setups(cfg, seeds=list(range(2000, 2000 + replicates)))
    rows = run_arms({nm: {"planner_cfg": {"kind": "rolling"},
                          "initial_model": nm} for nm, _ in builders},
                    setups)
    rows.to_csv(out / "rq1_operational_rows.csv", index=False)

    from ..sim.metrics import summarize
    op_by_arm = rows.groupby("arm").apply(
        lambda g: pd.Series(summarize(g)),
        include_groups=False,
    ).reset_index()
    print("\n== RQ1 operational preview (rolling planner, same envs) ==")
    print(op_by_arm[["arm", "DSR", "EDT_censored", "mean_unique_cells_searched"]]
          .round(3).to_string(index=False))
    op_by_arm.to_csv(out / "rq1_operational_summary.csv", index=False)
    grouped_bars(op_by_arm, x="arm", y="DSR", hue="arm",
                 title="RQ1/2 bridge: DSR by initial-belief model",
                 save=out / "rq1_operational_dsr.png")

    # ------------------------------------------------- operational statistics
    prop = arm_proportion_table(rows)
    comp = compare_arms(rows, [("uniform", nm) for nm in
                               ("distance", "gis", "datadriven")])
    print("\n-- RQ1 operational stats: Wilson 95% CI per arm --")
    print(prop.round(3).to_string(index=False))
    print("\n-- RQ1 paired comparisons vs uniform "
          "(CRN-paired, bootstrap 95% CI) --")
    print(comp.round(3).to_string(index=False))
    prop.to_csv(out / "rq1_operational_stats.csv", index=False)
    comp.to_csv(out / "rq1_operational_comparisons.csv", index=False)
    return {"summary": summary, "rows": rows, "model_file": model_file}
