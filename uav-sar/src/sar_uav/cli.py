"""Command-line interface for Multi-UAV SAR System.

Usage:
    python -m sar_uav exp         --tier 1..5,all [--quick] [--seed 42]
    python -m sar_uav plot        [--out results/figures]
    python -m sar_uav build-area  --name demo_valley [--grid 40x40]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_common(p):
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    ap = argparse.ArgumentParser(
        prog="sar-uav",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1. Experiment runner (Tier 1..5)
    p_exp = sub.add_parser("exp", help="Run 5-tier experimental benchmark suite")
    _add_common(p_exp)
    p_exp.add_argument("--tier", default="all", help="1..5, comma list (e.g. 1,2,3), or 'all'")
    p_exp.add_argument("--quick", action="store_true", help="Quick smoke test mode with reduced runs")

    # 2. Plot figures
    p_plot = sub.add_parser("plot", help="Generate research paper figures (Figures 1, 2, 3)")
    p_plot.add_argument("--out", default="results/figures", help="Output directory for figures")

    # 3. Build area
    p_area = sub.add_parser("build-area", help="Build synthetic terrain and GIS area data")
    p_area.add_argument("--name", default="demo_valley")
    p_area.add_argument("--grid", default="40x40")
    p_area.add_argument("--cell", type=float, default=100.0)
    p_area.add_argument("--seed", type=int, default=7)

    args = ap.parse_args(argv)

    if args.cmd == "exp":
        from .experiments.experiment_runner import run_experiment_suite

        out_dir = Path(args.out)
        tier_selection = (
            list(range(1, 6)) if args.tier == "all"
            else [int(x.strip()) for x in args.tier.split(",")]
        )
        run_experiment_suite(
            tier_selection=tier_selection,
            out_dir=out_dir,
            seed=args.seed,
            quick=args.quick,
        )
        logging.info("All requested tiers completed.")

    elif args.cmd == "plot":
        # Locate project root dynamically
        proj_root = _find_project_root()
        import subprocess
        script_path = proj_root / "scripts" / "plot_paper_figures.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Plotting script not found at {script_path}")
        subprocess.run([sys.executable, str(script_path), "--out", args.out], check=True)

    elif args.cmd == "build-area":
        from .data.area import build_synthetic_area
        w, h = (int(x) for x in args.grid.lower().split("x"))
        area = build_synthetic_area(args.name, W=w, H=h, cell=args.cell, seed=args.seed)
        d = Path("data/areas") / args.name
        area.save(d)
        logging.info(f"Saved area to {d} ({area.W}x{area.H}, {int(area.water.sum())} water cells)")


def _find_project_root() -> Path:
    """Finds the root directory containing the scripts folder."""
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (curr / "scripts" / "plot_paper_figures.py").exists():
            return curr
        curr = curr.parent
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
