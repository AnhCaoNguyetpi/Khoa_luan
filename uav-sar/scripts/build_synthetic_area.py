"""Build the synthetic wilderness area used by all offline experiments."""
from _bootstrap import *  # noqa: F401,F403
import argparse

from sar_uav.data.area import build_synthetic_area

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo_valley")
    ap.add_argument("--grid", default="40x40", help="WxH cells")
    ap.add_argument("--cell", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    W, H = (int(x) for x in a.grid.lower().split("x"))
    area = build_synthetic_area(a.name, W=W, H=H, cell=a.cell, seed=a.seed)
    out = ROOT / "data" / "areas" / a.name
    area.save(out)
    print(f"saved -> {out}")
    print(f"  grid {area.W}x{area.H} @ {area.cell:.0f} m | water {int(area.water.sum())} | "
          f"trail cells {int(area.trail_mask.sum())} | max slope {area.slope.max():.1f} deg")
