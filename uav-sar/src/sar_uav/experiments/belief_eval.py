"""Prediction-quality evaluation of initial-belief models (RQ1 metrics).

Metrics per the proposal ("Thi nghiem ve probability model"):

* log-likelihood of the true find location under the model pmf;
* probability mass within radius r of the truth;
* rank of the true cell;
* Top-k spatial recall (k as a fraction of the grid).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


def evaluate_model(name: str, builder: Callable[[object], np.ndarray],
                   area, incidents: Sequence,
                   radii_m: Sequence[float] = (200.0, 400.0, 800.0),
                   topk_fracs: Sequence[float] = (0.05, 0.10, 0.20)) -> pd.DataFrame:
    xy = area.centers_xy()
    rows: List[Dict] = []
    n = area.n_cells
    topk = {f: max(1, int(round(f * n))) for f in topk_fracs}
    for inc in incidents:
        p = np.asarray(builder(inc), dtype=float)
        p = p / p.sum()
        txy = xy[inc.find_cell]
        d = np.hypot(xy[:, 0] - txy[0], xy[:, 1] - txy[1])
        order = np.argsort(-p)
        rank = int(np.where(order == inc.find_cell)[0][0])
        row = {
            "model": name, "incident": id(inc), "profile": inc.profile,
            "loglik": float(np.log(max(p[inc.find_cell], 1e-300))),
            "rank": rank,
        }
        for r in radii_m:
            row[f"mass_r{int(r)}m"] = float(p[d <= r].sum())
        for f, k in topk.items():
            row[f"recall_top{int(f*100)}pct"] = bool(rank < k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_prediction(df: pd.DataFrame) -> pd.DataFrame:
    from .stats import wilson_interval
    g = df.drop(columns=["incident"]).groupby("model")
    out = g.mean(numeric_only=True)
    for col in df.columns:
        if col.startswith("recall_"):
            out[col] = g[col].mean()          # fraction == accuracy
            ks, ns = g[col].sum(), g[col].size()
            cis = [wilson_interval(int(k), int(n))
                   for k, n in zip(ks, ns)]
            out[f"{col}_ci_lo"] = [c[0] for c in cis]
            out[f"{col}_ci_hi"] = [c[1] for c in cis]
    return out.sort_values("loglik", ascending=False)
