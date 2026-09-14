"""Statistical helpers for the RQ summaries: confidence intervals and
hypothesis tests, implemented without scipy.

The batch runner shares environments across arms (common random numbers),
so every between-arm comparison pairs missions on ``replicate`` and uses
paired methods: exact McNemar for binary outcomes, paired bootstrap for
continuous ones. Sign convention: pair = (control, treatment) and
``diff = treatment - control``, so a positive diff means the treatment arm
scores higher on the metric (for EDT_censored lower is better).
"""

from __future__ import annotations

import logging
import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ intervals

def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3])
                                * r + b[4]) * r + 1)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for the binomial proportion ``k / n``."""
    if n == 0:
        return float("nan"), float("nan")
    z = _norm_ppf(1.0 - alpha / 2.0)
    ph = k / n
    denom = 1.0 + z * z / n
    centre = ph + z * z / (2.0 * n)
    spread = z * math.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n))
    return ((centre - spread) / denom, (centre + spread) / denom)


def bootstrap_mean_ci(x: Sequence[float], alpha: float = 0.05,
                      n_boot: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def student_t_ci(
    values: Sequence[float],
    alpha: float = 0.05,
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
) -> Tuple[float, float, Tuple[float, float]]:
    """Computes sample mean, standard error, and Student's t CI.

    Appropriate for small sample sizes (e.g. n=15 missions, df=14, t_crit ≈ 2.145).
    Applies uniformly to both per-arm metrics (e.g. RMST) and paired difference
    metrics between arms (e.g. delta_RMST, delta_energy).

    Parameters:
      values: 1D sequence of observed values or paired differences.
      alpha: Significance level (default 0.05 for 95% CI).
      bounds: Optional (min_bound, max_bound) to clip CI to physical domains
              (e.g. [0, H] for RMST, [-H, H] for delta RMST).

    Returns:
      (mean, se, (ci_lower, ci_upper))
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    mean_val = float(np.mean(arr))
    if n < 2:
        return mean_val, float("nan"), (float("nan"), float("nan"))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    try:
        import scipy.stats as st
        t_crit = float(st.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    except ImportError:
        t_crit = 2.1448 if n == 15 else 1.96
    lower = mean_val - t_crit * se
    upper = mean_val + t_crit * se
    if bounds is not None:
        b_min, b_max = bounds
        if b_min is not None:
            lower = max(lower, b_min)
            upper = max(upper, b_min)
        if b_max is not None:
            lower = min(lower, b_max)
            upper = min(upper, b_max)
    return mean_val, se, (float(lower), float(upper))


# ----------------------------------------------------------------- p-values

def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts.

    ``b``/``c`` count pairs where exactly one of control/treatment
    succeeded; the statistic is symmetric in ``b, c``.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def _paired_boot(diff: np.ndarray, alpha: float, n_boot: int,
                 rng: np.random.Generator) -> Tuple[float, float, float]:
    """Bootstrap CI of the paired-mean difference plus an approximate
    two-sided p-value (2 x min one-sided bootstrap tail)."""
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    p = min(1.0, max(p, 2.0 / n_boot))     # resolution floor of this method
    return float(lo), float(hi), p


# ------------------------------------------------------------ row-level APIs

def arm_proportion_table(rows: pd.DataFrame, value: str = "detected",
                         alpha: float = 0.05) -> pd.DataFrame:
    """Per-arm count, proportion and Wilson CI from mission-level rows."""
    recs = []
    for arm, g in rows.groupby("arm"):
        v = g[value].astype(bool)
        k, n = int(v.sum()), int(len(v))
        lo, hi = wilson_interval(k, n, alpha)
        recs.append({"arm": arm, "k": k, "n": n,
                     f"{value}_est": k / n if n else np.nan,
                     f"{value}_ci_lo": lo, f"{value}_ci_hi": hi})
    return pd.DataFrame(recs)


def grouped_proportion_ci(rows: pd.DataFrame, group_cols: Sequence[str],
                          value: str = "detected",
                          alpha: float = 0.05) -> pd.DataFrame:
    """Wilson CI per group (e.g. sweep points) from mission-level rows."""
    recs = []
    for key, g in rows.groupby(list(group_cols)):
        keys = key if isinstance(key, tuple) else (key,)
        v = g[value].astype(bool)
        k, n = int(v.sum()), int(len(v))
        lo, hi = wilson_interval(k, n, alpha)
        rec = dict(zip(group_cols, keys))
        rec.update({f"{value}_ci_lo": lo, f"{value}_ci_hi": hi})
        recs.append(rec)
    return pd.DataFrame(recs)


def compare_arms(rows: pd.DataFrame, pairs: Sequence[Tuple[str, str]],
                 metrics: Sequence[str] = ("detected", "EDT_censored"),
                 alpha: float = 0.05, n_boot: int = 5000,
                 seed: int = 0) -> pd.DataFrame:
    """Paired comparisons of arms sharing replicates (CRN design).

    ``pairs`` lists ``(control, treatment)`` arm names. For each metric the
    result carries the mean difference (treatment - control), its bootstrap
    CI, a bootstrap p-value and -- for binary metrics -- the exact McNemar
    p-value on the discordant pairs.
    """
    rows = rows.copy()
    if "EDT_censored" not in rows.columns:      # derived, like sim.metrics
        det = rows["detected"].astype(bool)
        rows["EDT_censored"] = rows["detect_time_min"].where(
            det, rows["horizon_min"])

    joined_all = []
    for ctrl, trt in pairs:
        a = rows.loc[rows["arm"] == ctrl].set_index("replicate")
        b = rows.loc[rows["arm"] == trt].set_index("replicate")
        common = [m for m in metrics if m in a and m in b]
        j = a[common].join(b[common], how="inner",
                           lsuffix="_ctrl", rsuffix="_trt").dropna()
        j["control"], j["treatment"] = ctrl, trt
        joined_all.append(j.reset_index())
    joined = pd.concat(joined_all, ignore_index=True)

    rng = np.random.default_rng(seed)
    recs = []
    for (ctrl, trt), j in zip(pairs, joined_all):
        for m in metrics:
            va = j[f"{m}_ctrl"].astype(float).to_numpy()
            vb = j[f"{m}_trt"].astype(float).to_numpy()
            diff = vb - va
            lo, hi, p_boot = _paired_boot(diff, alpha, n_boot, rng)
            rec = {"control": ctrl, "treatment": trt, "metric": m,
                   "n_pairs": int(diff.size),
                   "mean_ctrl": float(va.mean()),
                   "mean_trt": float(vb.mean()),
                   "diff": float(diff.mean()),
                   "ci_lo": lo, "ci_hi": hi, "p_boot": p_boot,
                   "p_mcnemar": np.nan}
            if m == "detected":
                n10 = int(((vb > 0) & (va <= 0)).sum())   # only trt found
                n01 = int(((va > 0) & (vb <= 0)).sum())   # only ctrl found
                rec["p_mcnemar"] = mcnemar_exact_p(n10, n01)
            recs.append(rec)
    out = pd.DataFrame(recs)
    log.info("compared %d arm pairs over %d metrics", len(pairs), len(metrics))
    return out


def stratified_bootstrap_ci(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
    equal_group_weight: bool = True,
) -> Tuple[float, float, Tuple[float, float]]:
    """Performs cluster/stratified bootstrap resampled at the group level.

    Resamples groups with replacement.
    When ``equal_group_weight=True`` (default), each group (e.g. scenario) receives equal
    weight 1/N_groups regardless of group size or replicate imbalance.
    Returns (point_estimate, se, (ci_lo, ci_hi)).
    """
    groups = list(data[group_col].unique())
    n_groups = len(groups)
    if n_groups == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))

    group_means = {g: float(data.loc[data[group_col] == g, metric_col].mean()) for g in groups}
    group_sizes = {g: int(len(data.loc[data[group_col] == g])) for g in groups}

    if equal_group_weight:
        point_est = float(np.mean([group_means[g] for g in groups]))
    else:
        point_est = float(data[metric_col].mean())

    if n_groups < 2:
        return point_est, float("nan"), (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_g = rng.choice(groups, size=n_groups, replace=True)
        if equal_group_weight:
            boot_means[b] = float(np.mean([group_means[g] for g in sample_g]))
        else:
            total_sum = sum(group_means[g] * group_sizes[g] for g in sample_g)
            total_n = sum(group_sizes[g] for g in sample_g)
            boot_means[b] = total_sum / total_n if total_n > 0 else np.nan

    lo, hi = np.nanquantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(np.nanstd(boot_means, ddof=1))
    return point_est, se, (float(lo), float(hi))


def stratified_paired_bootstrap_ci(
    data: pd.DataFrame,
    group_col: str,
    col_a: str,
    col_b: str,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
    equal_group_weight: bool = True,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Performs cluster/stratified bootstrap on paired differences (col_a - col_b).

    Resamples groups with replacement, preserving within-group and within-mission pairing.
    When ``equal_group_weight=True`` (default), each group (e.g. scenario) receives equal
    weight 1/N_groups regardless of group size or replicate imbalance.
    Returns (mean_diff, se, (ci_lo, ci_hi), p_boot).
    """
    groups = list(data[group_col].unique())
    n_groups = len(groups)
    if n_groups == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan")), float("nan")

    diff_series = data[col_a].astype(float) - data[col_b].astype(float)
    group_diffs = {g: float(diff_series.loc[data[group_col] == g].mean()) for g in groups}
    group_sizes = {g: int(len(data.loc[data[group_col] == g])) for g in groups}

    if equal_group_weight:
        point_diff = float(np.mean([group_diffs[g] for g in groups]))
    else:
        point_diff = float(diff_series.mean())

    if n_groups < 2:
        return point_diff, float("nan"), (float("nan"), float("nan")), float("nan")

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_g = rng.choice(groups, size=n_groups, replace=True)
        if equal_group_weight:
            boot_diffs[b] = float(np.mean([group_diffs[g] for g in sample_g]))
        else:
            total_sum = sum(group_diffs[g] * group_sizes[g] for g in sample_g)
            total_n = sum(group_sizes[g] for g in sample_g)
            boot_diffs[b] = total_sum / total_n if total_n > 0 else np.nan

    lo, hi = np.nanquantile(boot_diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(np.nanstd(boot_diffs, ddof=1))
    p_boot = 2.0 * min(float(np.nanmean(boot_diffs <= 0)), float(np.nanmean(boot_diffs >= 0)))
    p_boot = min(1.0, max(p_boot, 2.0 / n_boot))
    return point_diff, se, (float(lo), float(hi)), float(p_boot)


