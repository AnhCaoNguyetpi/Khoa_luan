"""Correctness of the statistical helpers (Wilson CI, McNemar, bootstrap)."""
import _common  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import pandas as pd

from sar_uav.experiments.stats import (arm_proportion_table, bootstrap_mean_ci,
                                       compare_arms, mcnemar_exact_p,
                                       student_t_ci, wilson_interval, _norm_ppf)


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


def test_student_t_ci_properties():
    # 1. Edge cases
    mean, se, ci = student_t_ci([])
    assert np.isnan(mean) and np.isnan(se) and np.isnan(ci[0]) and np.isnan(ci[1])

    # n = 1: Sample mean is preserved, but SE and CI must be NaN because variance cannot be estimated
    mean1, se1, ci1 = student_t_ci([5.0])
    assert mean1 == 5.0
    assert np.isnan(se1)
    assert np.isnan(ci1[0]) and np.isnan(ci1[1])

    # 2. Sample size 15 (e.g. 15 missions)
    vals = [10.0, 12.0, 11.0, 9.0, 14.0, 10.0, 13.0, 11.0, 12.0, 8.0, 10.0, 15.0, 11.0, 12.0, 10.0]
    mean, se, (lo, hi) = student_t_ci(vals, alpha=0.05)
    assert abs(mean - np.mean(vals)) < 1e-12
    # At df=14, t_crit is approx 2.145, which is strictly wider than 1.96
    margin = hi - mean
    assert margin > 1.96 * se
    assert abs(margin - 2.1448 * se) < 0.01

    # 3. Domain bounds clipping — tested separately for lower and upper bounds
    # 3a. Lower bound clipping: positive values with high variance whose raw lower CI penetrates below 0
    vals_low = [0.1, 0.2, 0.0, 0.3, 0.1, 0.0, 0.2, 0.1, 5.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1]
    _, _, (lo_raw, _) = student_t_ci(vals_low, alpha=0.05, bounds=None)
    assert lo_raw < 0.0, f"Precondition failed: unclipped lo={lo_raw} must be negative to test clipping"
    _, _, (lo_clipped, _) = student_t_ci(vals_low, alpha=0.05, bounds=(0.0, 24.0))
    assert lo_clipped == 0.0, f"Expected lo_clipped == 0.0, got {lo_clipped}"

    # 3b. Upper bound clipping: values near 24.0 with high variance whose raw upper CI penetrates above 24.0
    vals_high = [23.9, 23.8, 24.0, 23.7, 23.9, 24.0, 23.8, 23.9, 15.0, 23.9, 23.8, 24.0, 23.9, 23.8, 23.9]
    _, _, (_, hi_raw) = student_t_ci(vals_high, alpha=0.05, bounds=None)
    assert hi_raw > 24.0, f"Precondition failed: unclipped hi={hi_raw} must be > 24.0 to test clipping"
    _, _, (_, hi_clipped) = student_t_ci(vals_high, alpha=0.05, bounds=(0.0, 24.0))
    assert hi_clipped == 24.0, f"Expected hi_clipped == 24.0, got {hi_clipped}"
