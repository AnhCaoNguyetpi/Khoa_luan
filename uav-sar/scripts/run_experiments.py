"""Run the 5-tier experiment suite: --tier 1..5 or --tier all."""

from _bootstrap import *  # noqa: F401,F403
import argparse
import logging
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    ap = argparse.ArgumentParser(description="Multi-UAV SAR 5-Tier Experiment Runner")
    ap.add_argument("--tier", default="1", help="1..5, comma list, or 'all'")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true", help="reduced size for smoke testing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sar_uav.experiments.experiment_runner import run_experiment_suite

    tier_selection = (
        list(range(1, 6)) if args.tier == "all"
        else [int(x) for x in args.tier.split(",")]
    )
    run_experiment_suite(
        tier_selection=tier_selection,
        out_dir=Path(args.out),
        seed=args.seed,
        quick=args.quick,
    )
    print(f"\nAll requested tiers completed. Results saved to {args.out}")
