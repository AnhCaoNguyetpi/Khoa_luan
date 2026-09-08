"""RQ5 -- Do ben vung truoc sai so mo hinh.

The planner's world model is deliberately wrong while the environment and
the true target process are unchanged:

* ``eps``       -- initial map distorted toward uniform noise;
* ``q_bias``    -- detection probabilities over/under-estimated by the
                   planner (used in both assignment values and Bayes updates);
* ``beta_bias`` -- motion-model betas perturbed for belief forecasting.
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
from .stats import grouped_proportion_ci

log = logging.getLogger(__name__)

EPS_LIST = [0.0, 0.2, 0.4, 0.6]
Q_BIAS_LIST = [0.6, 0.8, 1.0, 1.2, 1.4]
BETA_BIAS_LIST = [-0.3, -0.15, 0.0, 0.15, 0.3]


def run(cfg: Dict | None = None, replicates: int = 24,
        eps_list=None, q_bias_list=None, beta_bias_list=None,
        out_dir: str | Path = "results", quick: bool = False) -> Dict:
    cfg = load_config(None, cfg or {})
    if quick:
        replicates = max(6, replicates // 3)
    eps_list = list(eps_list or EPS_LIST)
    q_bias_list = list(q_bias_list or Q_BIAS_LIST)
    beta_bias_list = list(beta_bias_list or BETA_BIAS_LIST)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    land = None

    setups = make_setups(cfg, seeds=list(range(6000, 6000 + replicates)))
    land = ~setups[0].area.water.reshape(-1)

    def noisy_map(eps):
        def _belief_override(setup):
            p_ref = initial_map(setup, "datadriven")
            noise = np.where(land, 1.0, 0.0)
            noise /= noise.sum()
            return (1 - eps) * p_ref + eps * noise

        return {"belief_override": _belief_override,
                "initial_model": "datadriven"}

    base_arm = {"initial_model": "datadriven",
                "planner_cfg": {"kind": "rolling"}}

    arms: Dict[str, Dict] = {}
    for e in eps_list:
        arms[f"map_eps={e}"] = {**noisy_map(e),
                                "planner_cfg": {"kind": "rolling"}}
    for qb in q_bias_list:
        arms[f"q_bias={qb}"] = {**base_arm, "q_est_bias": qb}
    for bb in beta_bias_list:
        arms[f"beta_bias={bb}"] = {**base_arm, "beta_est_bias": bb}

    rows = run_arms(arms, setups)
    rows["sweep_group"] = rows["arm"].str.split("=").str[0]
    try:
        rows["sweep_value"] = rows["arm"].str.split("=").str[1].astype(float)
    except (ValueError, IndexError):
        rows["sweep_value"] = np.nan
    rows.to_csv(out / "rq5_rows.csv", index=False)

    summ = rows.groupby(["sweep_group", "sweep_value"]).apply(
        lambda g: pd.Series(summarize(g)),
        include_groups=False).reset_index()
    ci = grouped_proportion_ci(rows,
                               ["sweep_group", "sweep_value"]).rename(
        columns={"detected_ci_lo": "DSR_ci_lo",
                 "detected_ci_hi": "DSR_ci_hi"})
    summ = summ.merge(ci, on=["sweep_group", "sweep_value"], how="left")
    print("\n== RQ5: robustness to model error ==")
    cols = ["sweep_group", "sweep_value"] \
        + [c for c in ["DSR", "DSR_ci_lo", "DSR_ci_hi", "EDT_censored",
                       "P(T<=120)", "mean_flight_distance_m"]
           if c in summ.columns]
    print(summ[cols].round(3).to_string(index=False))
    summ.to_csv(out / "rq5_summary.csv", index=False)

    sweep_lines(summ, x="sweep_value", y="DSR", hue="sweep_group",
                title="RQ5: DSR under planner model error",
                save=out / "rq5_robustness_dsr.png")
    return {"rows": rows, "summary": summ}
