"""Read-only reproductions supporting research_improvement_plan.md.

Run from uav-sar with:
    .venv/Scripts/python.exe -B docs/proposal/reproduce_research_audit.py

These examples report current behavior; they are not fixes or acceptance tests.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sar_uav.belief.motion import MotionBetas, MotionModel
from sar_uav.sim.metrics import summarize, survival_curve


def audit():
    rows = [
        {"detected": True, "detect_time_min": 30.0, "horizon_min": 180.0},
        {"detected": False, "detect_time_min": float("nan"), "horizon_min": 180.0},
    ]
    summary = summarize(rows)
    _, survival = survival_curve(rows)
    area = SimpleNamespace(
        shape=(1, 3), n_cells=3, cell=100.0,
        land_idx=lambda: np.arange(3),
        centers_xy=lambda: np.array([[0., 0.], [100., 0.], [200., 0.]]),
        dist_trail=np.array([[0., 100., 1000.]]),
        slope=np.zeros((1, 3)), veg=np.zeros((1, 3)), rugged=np.zeros((1, 3)),
    )
    model = MotionModel(area, MotionBetas(
        trail=2., slope=0., veg=0., dist=1., rain=0., stay=0., view=0.,
    ))
    options, probs = model.step_probs(1)
    q, f, dwell = 0.3, 0.45, 3
    return {
        "metrics": {
            "DSR_actual": summary["DSR"],
            "deadline_probability_actual": summary["P(T<=180)"],
            "deadline_probability_expected": 0.5,
            "survival_at_horizon_actual": float(survival[-1]),
            "survival_at_horizon_expected": 0.5,
            "restricted_mean_actual": summary["EDT_censored"],
            "restricted_mean_expected": 105.0,
        },
        "movement": {
            "start_cell": 1,
            "distance_to_trail_m": [0, 100, 1000],
            "probability_by_destination": dict(zip(options.tolist(), probs.tolist())),
            "expected_under_proposal_destination_score": "P(0) > P(2)",
        },
        "neighbor_footprint_likelihood": {
            "q_tick": q, "attenuation": f, "dwell_ticks": dwell,
            "probability_from_tick_sampling": 1 - (1 - f * q) ** dwell,
            "probability_used_by_current_block_update": f * (1 - (1 - q) ** dwell),
            "assumptions": "Stationary target; constant q; independent ticks.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
