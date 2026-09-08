"""Train the data-driven initial-belief (PMR logistic) model on synthetic incidents."""
from _bootstrap import *  # noqa: F401,F403
import argparse

import numpy as np

from sar_uav.belief.train import train_from_incidents
from sar_uav.config import load_config
from sar_uav.data.area import get_or_build_area
from sar_uav.experiments.datasets import generate_incidents

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--incidents", type=int, default=250)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    area = get_or_build_area(cfg["area"]["name"], ROOT / "data", cfg)
    inc = generate_incidents(cfg, area, a.incidents, seed=a.seed)
    print(f"training PMR model on {len(inc)} incidents "
          f"(area={area.name}, grid={area.W}x{area.H}) ...")
    pmr = train_from_incidents(inc, area,
                               rng=np.random.default_rng(a.seed),
                               verbose=a.verbose)
    out = ROOT / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{area.name}_pmr.npz"
    np.savez(f, **pmr)
    names = pmr.get("feature_names")
    w = pmr["w"]
    print("saved ->", f)
    if names is not None:
        for nm, wi in sorted(zip(names, w), key=lambda t: -abs(t[1])):
            print(f"  {nm:20s} {wi:+.3f}")
