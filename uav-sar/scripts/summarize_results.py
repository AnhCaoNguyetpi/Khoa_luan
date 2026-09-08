"""Aggregate all results/*.csv into one console report."""
from _bootstrap import *  # noqa: F401,F403
from pathlib import Path

import pandas as pd

RESULTS = ROOT / "results"

if __name__ == "__main__":
    files = sorted(RESULTS.rglob("*summary*.csv")) + \
        sorted(RESULTS.rglob("*_rows.csv"))
    if not files:
        print("no result csvs found under", RESULTS)
    for f in files:
        print("\n" + "=" * 72)
        print(f)
        df = pd.read_csv(f)
        with pd.option_context("display.max_columns", None,
                               "display.width", 160):
            print(df.head(30).round(3).to_string())
