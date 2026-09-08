"""Run the RQ experiment suite: --rq 1..5 or --rq all."""
from _bootstrap import *  # noqa: F401,F403
import argparse
import logging
import time

from sar_uav.config import load_config
from sar_uav.experiments import rq1, rq2, rq3, rq4, rq5

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rq", default="4", help="1..5, comma list, or 'all'")
    ap.add_argument("--replicates", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true",
                    help="reduced sizes for smoke-testing the pipeline")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rqs = list(range(1, 6)) if a.rq == "all" else \
        [int(x) for x in a.rq.split(",")]
    mods = {1: rq1, 2: rq2, 3: rq3, 4: rq4, 5: rq5}

    for n in rqs:
        t0 = time.time()
        print(f"\n================ RQ{n} ================")
        mods[n].run(cfg, replicates=a.replicates, out_dir=a.out,
                    quick=a.quick)
        print(f"RQ{n} done in {time.time()-t0:.0f}s")
