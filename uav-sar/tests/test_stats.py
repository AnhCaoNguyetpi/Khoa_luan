"""Correctness of the statistical helpers (Wilson CI, McNemar, bootstrap)."""
import _common  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import pandas as pd

from sar_uav.experiments.stats import (arm_proportion_table, bootstrap_mean_ci,
                                       compare_arms, mcnemar_exact_p,
                                       wilson_interval, _norm_ppf)


def test_norm_ppf_matches_tables():
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-5
    assert abs(_norm_ppf(0.995) - 2.575829) < 1e-5
    assert abs(_norm_ppf(0.5)) < 1e-9


def test_wilson_known_value():
    # k=5, n=10 -> Wilson 95% interval (0.2366, 0.7634); the wider
    # (0.187, 0.813) belongs to Clopper-Pearson, not Wilson.
    lo, hi = wilson_interval(5, 10)
    assert abs(lo - 0.23659) < 0.001 and abs(hi - 0.76341) < 0.001


def test_wilson_edges_and_narrowing():
    lo, hi = wilson_interval(0, 12)
    assert lo == 0.0 and hi < 0.3
    lo, hi = wilson_interval(12, 12)
    assert hi == 1.0 and lo > 0.7
    w6 = wilson_interval(3, 6)[1] - wilson_interval(3, 6)[0]
    w60 = wilson_interval(30, 60)[1] - wilson_interval(30, 60)[0]
    assert w60 < w6                       # CI narrows as n grows
    assert np.isnan(wilson_interval(0, 0)[0])


def test_mcnemar_exact():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0            # perfectly split discordance
    p = mcnemar_exact_p(12, 0)                     # one-sided all the way
    assert p < 0.001
    assert mcnemar_exact_p(3, 1) == mcnemar_exact_p(1, 3)   # symmetric
    assert abs(mcnemar_exact_p(8, 2) - 0.1094) < 0.001      # 2*(56/1024)
    assert mcnemar_exact_p(9, 1) < 0.03


def test_bootstrap_mean_ci():
    rng = np.random.default_rng(42)
    x = rng.normal(10.0, 2.0, size=400)
    lo, hi = bootstrap_mean_ci(x, n_boot=4000, seed=7)
    assert lo < 10.0 < hi and (hi - lo) < 0.6      # covers truth, reasonably tight
    lo1, hi1 = bootstrap_mean_ci(np.arange(50.0), n_boot=2000, seed=3)
    lo2, hi2 = bootstrap_mean_ci(np.arange(50.0), n_boot=2000, seed=3)
    assert (lo1, hi1) == (lo2, hi2)                # deterministic under seed


def _synthetic_rows(n_rep=40, trt_gain=True):
    """CRN-style rows: same replicates for both arms; treatment detects more."""
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(n_rep):
        base = rng.random() < 0.25                  # control detection ~25%
        if trt_gain:
            trt = base or (rng.random() < 0.55)     # superset: CRN-like
        else:
            trt = rng.random() < 0.25               # independent, null case
        t_c = rng.uniform(20, 170) if base else np.nan
        t_t = rng.uniform(20, 170) if trt else np.nan
        rows.append({"arm": "ctrl", "replicate": rep, "detected": base,
                     "detect_time_min": t_c, "horizon_min": 180.0})
        rows.append({"arm": "trt", "replicate": rep, "detected": trt,
                     "detect_time_min": t_t, "horizon_min": 180.0})
    return pd.DataFrame(rows)


def test_compare_arms_detects_real_effect():
    rows = _synthetic_rows(trt_gain=True)
    comp = compare_arms(rows, [("ctrl", "trt")], metrics=("detected",),
                        n_boot=3000, seed=1)
    r = comp.iloc[0]
    assert r["n_pairs"] == 40 and r["diff"] > 0
    assert r["ci_lo"] > 0                           # effect clear of zero
    assert r["p_mcnemar"] < 0.01


def test_compare_arms_null_stays_nonsignificant():
    rows = _synthetic_rows(trt_gain=False)
    comp = compare_arms(rows, [("ctrl", "trt")], metrics=("detected",),
                        n_boot=3000, seed=1)
    r = comp.iloc[0]
    assert r["p_mcnemar"] > 0.05


def test_compare_arms_edt_derived_and_paired():
    rows = _synthetic_rows(trt_gain=True)
    comp = compare_arms(rows, [("ctrl", "trt")],
                        metrics=("EDT_censored",), n_boot=3000, seed=2)
    r = comp.iloc[0]
    assert r["metric"] == "EDT_censored"
    assert r["diff"] < 0                            # treatment finds sooner
    assert r["ci_hi"] < 0


def test_arm_proportion_table():
    rows = _synthetic_rows()
    tbl = arm_proportion_table(rows).set_index("arm")
    assert int(tbl.loc["ctrl", "n"]) == 40
    assert abs(tbl.loc["ctrl", "detected_est"] -
               rows[rows.arm == "ctrl"].detected.mean()) < 1e-12
    assert (tbl.loc["ctrl", "detected_ci_lo"]
            < tbl.loc["ctrl", "detected_est"]
            < tbl.loc["ctrl", "detected_ci_hi"])
