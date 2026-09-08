"""Command-line interface.

    python -m sar_uav build-area  --name demo_valley
    python -m sar_uav train-belief [--incidents 250]
    python -m sar_uav mission     [--planner rolling --belief datadriven]
    python -m sar_uav exp         --rq 1..5 [--replicates 24] [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_common(p):
    p.add_argument("--config", default=None, help="JSON config file")
    p.add_argument("--out", default="results", help="output directory")


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(prog="sar-uav",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-area", help="build the synthetic area data")
    p.add_argument("--name", default="demo_valley")
    p.add_argument("--grid", default="40x40")
    p.add_argument("--cell", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("train-belief", help="fit the data-driven PMR model")
    p.add_argument("--area", default="demo_valley")
    p.add_argument("--incidents", type=int, default=250)
    p.add_argument("--seed", type=int, default=9001)

    p = sub.add_parser("mission", help="run one demo mission")
    _add_common(p)
    p.add_argument("--planner", default="rolling",
                   choices=["greedy", "greedy_ratio", "static", "rolling", "milp"])
    p.add_argument("--belief", default="gis",
                   choices=["uniform", "distance", "gis", "datadriven"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render-dir", default=None)

    p = sub.add_parser("exp", help="run an experiment (RQ1..RQ5)")
    _add_common(p)
    p.add_argument("--rq", required=True, type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--replicates", type=int, default=24)
    p.add_argument("--quick", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "build-area":
        from .data.area import build_synthetic_area
        w, h = (int(x) for x in args.grid.lower().split("x"))
        area = build_synthetic_area(args.name, W=w, H=h, cell=args.cell,
                                    seed=args.seed)
        d = Path("data/areas") / args.name
        area.save(d)
        print(f"saved area to {d} ({area.W}x{area.H}, "
              f"{int(area.water.sum())} water cells)")

    elif args.cmd == "train-belief":
        import numpy as np
        from .belief.train import train_from_incidents
        from .config import load_config
        from .data.area import get_or_build_area
        from .experiments.datasets import generate_incidents
        cfg = load_config(None, {"area": {"name": args.area}})
        area = get_or_build_area(args.area, _data_root(), cfg)
        inc = generate_incidents(cfg, area, args.incidents, seed=args.seed)
        pmr = train_from_incidents(inc, area,
                                   rng=np.random.default_rng(args.seed))
        out = _data_root() / "models"
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{args.area}_pmr.npz"
        np.savez(f, **pmr)
        print(f"saved PMR weights -> {f}")

    elif args.cmd == "mission":
        from .config import load_config
        from .sim.mission import MissionSetup, run_mission
        cfg = load_config(args.config, {
            "belief": {"initial_model": args.belief}})
        setup = MissionSetup.create(cfg, seed=args.seed)
        res = run_mission(setup,
                          planner_cfg={"kind": args.planner},
                          initial_model=args.belief,
                          render_dir=args.render_dir,
                          arm_name=f"{args.planner}-{args.belief}")
        print("\n--- mission result "
              f"({args.planner}/{args.belief}, profile={setup.profile}) ---")
        for k, v in res.to_row().items():
            if v not in ("", None) or k in ("detected",):
                print(f"  {k:26s} {v}")

    elif args.cmd == "exp":
        from .config import load_config
        from .experiments import rq1, rq2, rq3, rq4, rq5
        mods = {1: rq1, 2: rq2, 3: rq3, 4: rq4, 5: rq5}
        mod = mods[args.rq]
        kwargs = dict(replicates=args.replicates, quick=args.quick,
                      out_dir=args.out)
        mod.run(load_config(args.config), **kwargs)
        print(f"\nRQ{args.rq} finished; outputs in '{args.out}/'.")


def _data_root():
    # cli.py sits in <root>/src/sar_uav/ -> project root is parents[2]
    return Path(__file__).resolve().parents[2] / "data"


if __name__ == "__main__":
    main()
