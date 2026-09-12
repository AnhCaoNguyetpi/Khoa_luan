"""Test runner for uav-sar test suite."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    exit_code = pytest.main(["-v", str(ROOT / "tests")])
    sys.exit(exit_code)
