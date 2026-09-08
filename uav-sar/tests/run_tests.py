"""Plain test runner: `python tests/run_tests.py` (no pytest required).

Also compatible with pytest if it is installed.
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = [
    "test_belief_update",
    "test_motion",
    "test_detection",
    "test_initial_belief",
    "test_energy_state",
    "test_planners",
    "test_mission_end_to_end",
    "test_stats",
]


def main() -> int:
    passed, failed = [], []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
                passed.append(f"{mod_name}::{name}")
                print(f"PASS {mod_name}::{name} ({time.time()-t0:.2f}s)")
            except Exception:
                failed.append((f"{mod_name}::{name}", traceback.format_exc()))
                print(f"FAIL {mod_name}::{name}")
                traceback.print_exc()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
