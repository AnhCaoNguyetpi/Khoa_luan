"""Initial belief (probability map) models -- proposal section RQ1.

Comparison ladder::

    Uniform  ->  DistanceGaussian  ->  GisExpert  ->  DataDriven

All models return a normalised probability vector over the ``n`` grid cells
with zero mass on water cells, conditioned on the Initial Planning Point
(IPP) and the elapsed time since the person went missing.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FEATURE_NAMES = [
    "dist_ipp_km",          # distance from IPP
    "dist_ipp_x_elapsed",   # interaction: dist_ipp_km * elapsed_hours
    "dist_trail_km",        # distance to nearest trail
    "slope_deg",
    "veg",                  # vegetation fraction
    "elev_rel_m",           # elevation relative to IPP
    "rugged_m",
]

GIS_EXPERT_WEIGHTS = {
    # hand-set from lost-person-behaviour literature (Koester; Lin & Goodrich):
    # subjects concentrate near trails and near the IPP, avoid steep slopes,
    # and drift downhill.
    "dist_ipp_km": -0.55,
    "dist_ipp_x_elapsed": -0.35,
    "dist_trail_km": -0.90,
    "slope_deg": -0.045,
    "veg": 0.30,            # thick vegetation: people shelter / slow down there
    "elev_rel_m": -0.004,
    "rugged_m": -0.02,
}


def cell_features(area, ipp_idx: int, elapsed_min: float) -> np.ndarray:
    """(n, F) feature matrix used by both the expert and the learned model."""
    xy = area.centers_xy()
    ipp_xy = xy[ipp_idx]
    d_ipp_km = np.hypot(xy[:, 0] - ipp_xy[0], xy[:, 1] - ipp_xy[1]) / 1000.0
    elapsed_h = max(elapsed_min, 1.0) / 60.0
    feats = np.stack([
        d_ipp_km,
        d_ipp_km * elapsed_h,
        area.dist_trail.reshape(-1) / 1000.0,
        area.slope.reshape(-1),
        area.veg.reshape(-1),
        area.elev.reshape(-1) - float(area.elev.reshape(-1)[ipp_idx]),
        area.rugged.reshape(-1),
    ], axis=1)
    return feats


def _normalise(p: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    p = np.where(valid_mask, np.clip(p, 0.0, None), 0.0)
    s = p.sum()
    if s <= 0:                       # degenerate fallback: uniform on land
        p = valid_mask.astype(float)
        s = p.sum()
    return p / s


# --------------------------------------------------------------------- models

def uniform_belief(area) -> np.ndarray:
    p = (~area.water.reshape(-1)).astype(float)
    return p / p.sum()


def distance_gaussian_belief(area, ipp_idx: int, elapsed_min: float,
                             sigma_per_min: float = 34.0) -> np.ndarray:
    """Isotropic Gaussian around the IPP with elapsed-time scaling.

    Baseline that ignores terrain/trails/land-cover entirely.
    """
    xy = area.centers_xy()
    ipp_xy = xy[ipp_idx]
    sigma = max(sigma_per_min * elapsed_min, 120.0)
    d2 = (xy[:, 0] - ipp_xy[0]) ** 2 + (xy[:, 1] - ipp_xy[1]) ** 2
    p = np.exp(-d2 / (2.0 * sigma ** 2))
    return _normalise(p, ~area.water.reshape(-1))


def gis_expert_belief(area, ipp_idx: int, elapsed_min: float,
                      weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Log-linear expert model on GIS features (literature-informed weights)."""
    w = dict(GIS_EXPERT_WEIGHTS)
    if weights:
        w.update(weights)
    wv = np.array([w[k] for k in FEATURE_NAMES])
    X = cell_features(area, ipp_idx, elapsed_min)
    z = X @ wv
    z -= z.max()
    p = np.exp(z)
    return _normalise(p, ~area.water.reshape(-1))


def data_driven_belief(area, ipp_idx: int, elapsed_min: float,
                       weights_npz: str | Dict) -> np.ndarray:
    """Learned PMR logistic model (see belief/train.py).

    ``weights_npz`` is either a path to the .npz produced by training or a
    dict with keys ``w``, ``mean``, ``std``.
    """
    if isinstance(weights_npz, (str,)):
        z = np.load(weights_npz, allow_pickle=True)
        w, mean, std = z["w"].astype(float), z["mean"].astype(float), z["std"].astype(float)
    else:
        w, mean, std = (weights_npz["w"], weights_npz["mean"], weights_npz["std"])
    X = cell_features(area, ipp_idx, elapsed_min)
    Xs = (X - mean) / np.maximum(std, 1e-9)
    logit = Xs @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    return _normalise(p, ~area.water.reshape(-1))


def oracle_belief(area, true_cell: int, spread_cells: float = 1.5) -> np.ndarray:
    """Diagnostic-only 'perfect prediction': tight kernel on the true cell."""
    xy = area.centers_xy()
    t = xy[true_cell]
    sig = spread_cells * area.cell
    d2 = (xy[:, 0] - t[0]) ** 2 + (xy[:, 1] - t[1]) ** 2
    p = np.exp(-d2 / (2 * sig ** 2))
    return _normalise(p, ~area.water.reshape(-1))


def build_initial_belief(model: str, area, ipp_idx: int, elapsed_min: float,
                         cfg: Dict | None = None) -> np.ndarray:
    """Dispatch initial-belief model by name (config key ``belief.initial_model``)."""
    c = cfg or {}
    if model == "uniform":
        return uniform_belief(area)
    if model == "distance":
        return distance_gaussian_belief(area, ipp_idx, elapsed_min,
                                        sigma_per_min=c.get("sigma_per_min", 34.0))
    if model == "gis":
        return gis_expert_belief(area, ipp_idx, elapsed_min)
    if model == "datadriven":
        mf = c.get("model_file")
        if not mf:
            raise ValueError("belief.initial_model='datadriven' requires "
                             "belief.model_file (train via scripts/train_belief_model.py)")
        return data_driven_belief(area, ipp_idx, elapsed_min, mf)
    if model == "oracle":
        raise ValueError("oracle belief requires the true cell; "
                         "use experiments.rq2 to blend maps instead")
    raise ValueError(f"unknown initial belief model '{model}'")
