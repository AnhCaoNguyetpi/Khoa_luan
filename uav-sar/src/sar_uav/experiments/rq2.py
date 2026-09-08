"""RQ2 -- Chat luong du bao co chuyen thanh loi ich van hanh?

    Prediction Quality -> Routing Decisions -> SAR Performance

Degrades the reference (data-driven) initial map by blending it toward a
featureless uniform map::

    P_est = (1 - eps) * P_reference + eps * P_noise,   eps in [0, 1]

and measures DSR / EDT of the same planner on the same environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import load_config
from ..sim.metrics import summarize
from ..viz.plots import sweep_lines
from .runner import initial_map, make_setups, run_arms
from .stats import compare_arms, grouped_proportion_ci

log = logging.getLogger(__name__)

DEFAULT_EPS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def run(cfg: Dict | None = None, replicates: int = 24,
        eps_list=None, out_dir: str | Path = "results",
        quick: bool = False) -> Dict:
    cfg = load_config(None, cfg or {})
    if quick:
        replicates = max(6, replicates // 3)
    eps_list = list(eps_list or DEFAULT_EPS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    setups = make_setups(cfg, seeds=list(range(3000, 3000 + replicates)))
    land = ~setups[0].area.water.reshape(-1)

    def blend(eps):
        def _belief_override(setup):
            p_ref = initial_map(setup, "datadriven")
            p_noise = np.where(land, 1.0, 0.0)
            p_noise /= p_noise.sum()
            return (1 - eps) * p_ref + eps * p_noise

        return {"planner_cfg": {"kind": "rolling"},
                "belief_override": _belief_override}

    arms: Dict[str, Dict] = {f"eps={e}": blend(e) for e in eps_list}
    rows = run_arms(arms, setups)
    rows.to_csv(out / "rq2_rows.csv", index=False)

    by_eps = rows.groupby("arm").apply(
        lambda g: pd.Series(summarize(g)), include_groups=False)
    by_eps["eps"] = [float(a.split("=")[1]) for a in by_eps.index]
    by_eps = by_eps.sort_values("eps")
    ci = grouped_proportion_ci(rows, ["arm"]).set_index("arm")
    by_eps["DSR_ci_lo"], by_eps["DSR_ci_hi"] = \
        ci["detected_ci_lo"], ci["detected_ci_hi"]
    print("\n== RQ2: SAR performance vs initial-map quality ==")
    print(by_eps[["eps", "DSR", "DSR_ci_lo", "DSR_ci_hi", "EDT_censored",
                  "P(T<=60)", "P(T<=120)",
                  "mean_unique_cells_searched"]].round(3).to_string(index=False))
    by_eps.to_csv(out / "rq2_summary.csv", index=False)

    base = f"eps={eps_list[0]}"
    comp = compare_arms(rows, [(base, a) for a in arms if a != base])
    comp.to_csv(out / "rq2_comparisons.csv", index=False)
    det = comp[comp.metric == "detected"]
    print("\n-- RQ2 paired stats vs eps=0 (CRN-paired) --")
    print(det[["treatment", "diff", "ci_lo", "ci_hi",
               "p_mcnemar"]].round(4).to_string(index=False))

    long = by_eps.reset_index().melt(id_vars="eps",
                                     value_vars=["DSR", "EDT_censored"],
                                     var_name="metric", value_name="value")
    fig = sweep_lines(long, x="eps", y="value", hue="metric",
                      title="RQ2: operational gain degrades with map quality",
                      xlabel="eps (map distortion toward uniform)",
                      save=out / "rq2_performance_vs_eps.png")
    return {"rows": rows, "summary": by_eps}
