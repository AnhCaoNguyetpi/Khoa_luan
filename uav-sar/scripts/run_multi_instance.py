"""Run Multi-Instance Comparative Benchmark (10 instances x 100 missions)."""

import json
from pathlib import Path
import sys

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sar_uav.experiments.tier3_multi_instance import run_multi_instance_benchmark


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--num-missions", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--operator-order", type=str, default="fixed", choices=["fixed", "round_robin", "shuffled"])
    parser.add_argument("--out", type=str, default="results/multi_instance_tier3_results.json")
    args = parser.parse_args()

    results = run_multi_instance_benchmark(
        num_instances=args.num_instances,
        num_missions_per_instance=args.num_missions,
        master_seed=args.master_seed,
        order_strategy=args.operator_order,
    )

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    run_id = results["metadata"]["run_id"]
    runs_dir = out_file.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_archival_file = runs_dir / f"{run_id}_multi_instance.json"
    with open(run_archival_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark finished successfully!")
    print(f"Canonical results: {out_file}")
    print(f"Archival run file: {run_archival_file}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
