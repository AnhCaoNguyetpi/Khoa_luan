"""Factorial Operator Order & Ablation Study.

Runs 4 ablation arms x 3 ordering strategies across 10 benchmark scenarios:
- Arms: Proposed_Fast (Full), Ablation_NoReplace, Ablation_NoRebalance, Ablation_NoReplaceRebalance
- Ordering strategies: fixed, round_robin, shuffled
- Evaluates search metrics (J_ensemble, evals, runtime, accepted moves) and mission performance (DSR, RMST)
- Calculates scenario-paired differences (Full vs NoReplace, Full vs NoRebalance, Full vs Neither) and cross-order differences
- Stores full raw records, environment metadata, provenance, and run archival file

Answers research question: Does NoRebalance yield higher J due to inherent operator dynamics or due to evaluation budget allocation under fixed ordering?
"""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    pass
import argparse
import json
import logging
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sar_uav.experiments.experiment_runner import _get_git_info
from sar_uav.experiments.stats import (
    student_t_ci,
    stratified_bootstrap_ci,
    stratified_paired_bootstrap_ci,
)
from sar_uav.experiments.tier3_multi_instance import (
    create_instance,
    solve_instance_strategies,
    evaluate_scenario_multi_regimes,
    get_regime_seed_offset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def aggregate_operator_study_results(
    raw_records: List[Dict],
    order_strategies: List[str],
    ablation_arms: List[str],
    H: int,
    alpha: float = 0.05,
    n_boot: int = 5000,
    master_seed: int = 42,
) -> Tuple[Dict, List[Dict], Dict, List[Dict], Dict]:
    """
    Aggregate raw study records with strictly equal scenario weighting (Equal Scenario Weighting).
    All averages, paired differences, and confidence intervals are computed at the scenario level.
    """
    results_matrix: Dict[str, Dict[str, Dict]] = {order: {} for order in order_strategies}
    df = pd.DataFrame(raw_records)

    # Summary table and individual CIs per (order, arm)
    summary_table = []
    for order in order_strategies:
        df_order = df[df["order_strategy"] == order]
        for arm in ablation_arms:
            df_arm = df_order[df_order["arm"] == arm]

            # Scenario-level J_ensemble (from in_ensemble solves)
            df_arm_in = df_arm[df_arm["gt_regime"] == "in_ensemble"]
            scen_j = df_arm_in.groupby("scenario_id")["J_ensemble"].mean().tolist()
            m_j, se_j, ci_j = student_t_ci(scen_j, alpha=alpha)

            # Scenario-level DSR and RMST across all evaluated regimes
            scen_dsr = df_arm.groupby("scenario_id")["DSR"].mean().tolist()
            m_dsr, se_dsr, ci_dsr = student_t_ci(scen_dsr, alpha=alpha, bounds=(0.0, 1.0))
            _, boot_se_dsr, boot_ci_dsr = stratified_bootstrap_ci(
                df_arm, group_col="scenario_id", metric_col="DSR", alpha=alpha, n_boot=n_boot, seed=master_seed, equal_group_weight=True
            )

            scen_rmst = df_arm.groupby("scenario_id")["RMST"].mean().tolist()
            m_rmst, se_rmst, ci_rmst = student_t_ci(scen_rmst, alpha=alpha, bounds=(0.0, float(H)))

            scen_evals = df_arm_in.groupby("scenario_id")["total_evals"].mean().tolist()
            scen_rt = df_arm_in.groupby("scenario_id")["runtime_sec"].mean().tolist()
            scen_moves = df_arm_in.groupby("scenario_id")["accepted_moves"].mean().tolist()

            results_matrix[order][arm] = {
                "mean_J_ensemble": float(m_j),
                "ci95_J_ensemble": list(ci_j),
                "mean_DSR": float(m_dsr),
                "ci95_DSR_student_t": list(ci_dsr),
                "ci95_DSR_bootstrap": list(boot_ci_dsr),
                "mean_RMST": float(m_rmst),
                "ci95_RMST": list(ci_rmst),
                "mean_evals": float(np.mean(scen_evals)),
                "mean_runtime_sec": float(np.mean(scen_rt)),
                "mean_accepted_moves": float(np.mean(scen_moves)),
                "per_scenario_J_ensemble": scen_j,
                "per_scenario_DSR": scen_dsr,
                "per_scenario_RMST": scen_rmst,
            }

            summary_table.append({
                "Order": order,
                "Arm": arm,
                "Mean_J": round(float(m_j), 5),
                "CI95_J": f"[{ci_j[0]:.5f}, {ci_j[1]:.5f}]",
                "Mean_DSR": round(float(m_dsr) * 100, 2),
                "CI95_DSR": f"[{boot_ci_dsr[0]*100:.2f}%, {boot_ci_dsr[1]*100:.2f}%]",
                "Mean_RMST": round(float(m_rmst), 3),
                "Mean_Evals": round(float(np.mean(scen_evals)), 1),
                "Mean_Moves": round(float(np.mean(scen_moves)), 1),
                "Mean_Runtime_s": round(float(np.mean(scen_rt)), 3),
            })

    # Paired comparisons within each ordering strategy (Proposed_Fast minus each ablation arm)
    paired_comparisons_within_order: Dict[str, Dict[str, Dict]] = {}
    paired_summary_table = []

    for order in order_strategies:
        paired_comparisons_within_order[order] = {}
        df_order = df[df["order_strategy"] == order]
        df_prop = df_order[df_order["arm"] == "Proposed_Fast"]

        for arm in ["Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"]:
            df_arm = df_order[df_order["arm"] == arm]

            # Merge matched pairs on (scenario_id, replicate_id, gt_regime)
            merged = pd.merge(df_prop, df_arm, on=["scenario_id", "replicate_id", "gt_regime"], suffixes=("_prop", "_arm"))
            merged["delta_dsr"] = merged["DSR_prop"] - merged["DSR_arm"]
            merged["delta_rmst"] = merged["RMST_prop"] - merged["RMST_arm"]
            merged["delta_j"] = merged["J_ensemble_prop"] - merged["J_ensemble_arm"]

            # Scenario-paired Delta J (filter to in_ensemble to avoid repeating replicates across GT regimes)
            merged_j = merged[merged["gt_regime"] == "in_ensemble"]
            scen_delta_j = merged_j.groupby("scenario_id")["delta_j"].mean().tolist()
            m_dj, se_dj, ci_dj = student_t_ci(scen_delta_j, alpha=alpha)

            # Scenario-paired Delta DSR
            scen_delta_dsr = merged.groupby("scenario_id")["delta_dsr"].mean().tolist()
            m_ddsr, se_ddsr, ci_ddsr = student_t_ci(scen_delta_dsr, alpha=alpha, bounds=(-1.0, 1.0))
            _, _, boot_ci_ddsr, p_boot_ddsr = stratified_paired_bootstrap_ci(
                merged, group_col="scenario_id", col_a="DSR_prop", col_b="DSR_arm", alpha=alpha, n_boot=n_boot, seed=master_seed, equal_group_weight=True
            )

            scen_delta_rmst = merged.groupby("scenario_id")["delta_rmst"].mean().tolist()
            m_drmst, se_drmst, ci_drmst = student_t_ci(scen_delta_rmst, alpha=alpha, bounds=(-float(H), float(H)))

            pair_key = f"Proposed_vs_{arm}"
            paired_comparisons_within_order[order][pair_key] = {
                "mean_delta_J": float(m_dj),
                "ci95_delta_J_student_t": list(ci_dj),
                "mean_delta_DSR": float(m_ddsr),
                "ci95_delta_DSR_student_t": list(ci_ddsr),
                "ci95_delta_DSR_bootstrap": list(boot_ci_ddsr),
                "p_boot_delta_DSR": float(p_boot_ddsr),
                "mean_delta_RMST": float(m_drmst),
                "ci95_delta_RMST_student_t": list(ci_drmst),
                "per_scenario_delta_J": scen_delta_j,
                "per_scenario_delta_DSR": scen_delta_dsr,
            }

            paired_summary_table.append({
                "Order": order,
                "Comparison": f"Proposed vs {arm.replace('Ablation_', '')}",
                "Mean_dJ": round(float(m_dj), 5),
                "CI95_dJ": f"[{ci_dj[0]:.5f}, {ci_dj[1]:.5f}]",
                "Mean_dDSR": round(float(m_ddsr) * 100, 2),
                "CI95_dDSR": f"[{boot_ci_ddsr[0]*100:.2f}%, {boot_ci_ddsr[1]*100:.2f}%]",
                "p_boot": round(float(p_boot_ddsr), 4),
                "Mean_dRMST": round(float(m_drmst), 3),
            })

    # Cross-ordering paired differences for each arm (Order minus Fixed on same scenarios)
    paired_comparisons_across_orders: Dict[str, Dict] = {}
    df_fixed = df[df["order_strategy"] == "fixed"]

    for arm in ablation_arms:
        df_arm_fixed = df_fixed[df_fixed["arm"] == arm]

        for alt_order in ["round_robin", "shuffled"]:
            df_alt = df[(df["order_strategy"] == alt_order) & (df["arm"] == arm)]

            merged_order = pd.merge(df_alt, df_arm_fixed, on=["scenario_id", "replicate_id", "gt_regime"], suffixes=("_alt", "_fixed"))
            merged_order["delta_dsr"] = merged_order["DSR_alt"] - merged_order["DSR_fixed"]
            merged_order["delta_j"] = merged_order["J_ensemble_alt"] - merged_order["J_ensemble_fixed"]

            merged_order_j = merged_order[merged_order["gt_regime"] == "in_ensemble"]
            scen_diff_j = merged_order_j.groupby("scenario_id")["delta_j"].mean().tolist()
            m_dj_order, _, ci_dj_order = student_t_ci(scen_diff_j, alpha=alpha)

            scen_diff_dsr = merged_order.groupby("scenario_id")["delta_dsr"].mean().tolist()
            m_ddsr_order, _, ci_ddsr_order = student_t_ci(scen_diff_dsr, alpha=alpha, bounds=(-1.0, 1.0))

            paired_comparisons_across_orders[f"{arm}_{alt_order}_vs_fixed"] = {
                "arm": arm,
                "alt_order": alt_order,
                "mean_delta_J_vs_fixed": float(m_dj_order),
                "ci95_delta_J_vs_fixed": list(ci_dj_order),
                "mean_delta_DSR_vs_fixed": float(m_ddsr_order),
                "ci95_delta_DSR_vs_fixed": list(ci_ddsr_order),
                "per_scenario_delta_J": scen_diff_j,
                "per_scenario_delta_DSR": scen_diff_dsr,
            }

    return results_matrix, summary_table, paired_comparisons_within_order, paired_summary_table, paired_comparisons_across_orders


def run_operator_order_study(
    num_instances: int = 10,
    num_missions: int = 100,
    master_seed: int = 42,
    H: int = 15,
    num_uavs: int = 2,
    max_evals_total: int = 600,
    time_limit_sec: float = 15.0,
    n_boot: int = 5000,
    alpha: float = 0.05,
    out_path: str = "results/operator_order_study_results.json"
) -> Dict:
    t0 = time.perf_counter()
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    git_info = _get_git_info()

    order_strategies = ["fixed", "round_robin", "shuffled"]
    ablation_arms = [
        "Proposed_Fast",
        "Ablation_NoReplace",
        "Ablation_NoRebalance",
        "Ablation_NoReplaceRebalance"
    ]
    gt_regimes = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]

    raw_records: List[Dict] = []

    for order in order_strategies:
        log.info(f"=== Running Operator Order Strategy: {order} ===")
        for inst_id in range(num_instances):
            inst = create_instance(inst_id, master_seed=master_seed, H=H, num_uavs=num_uavs)
            scenario_id = inst_id % 10
            replicate_id = inst_id // 10

            # Solve strategies with specific order strategy
            strategies, solver_stats = solve_instance_strategies(
                inst,
                max_evals_total=max_evals_total,
                time_limit_sec=time_limit_sec,
                order_strategy=order
            )

            # Evaluate across all 3 ground truth regimes
            regime_results = evaluate_scenario_multi_regimes(
                inst,
                strategies,
                num_missions=num_missions,
                gt_regimes=gt_regimes
            )

            for regime, (summary, paired, _) in regime_results.items():
                for arm in ablation_arms:
                    raw_records.append({
                        "order_strategy": order,
                        "instance_id": inst_id,
                        "scenario_id": scenario_id,
                        "replicate_id": replicate_id,
                        "gt_regime": regime,
                        "arm": arm,
                        "J_ensemble": summary[arm]["J_ensemble"],
                        "J_true_gt": summary[arm]["J_true_gt"],
                        "DSR": summary[arm]["DSR"],
                        "RMST": summary[arm]["mean_RMST_ticks"],
                        "energy_kJ": summary[arm]["mean_energy_kJ"],
                        "runtime_sec": solver_stats[arm]["total_runtime_sec"],
                        "total_evals": solver_stats[arm]["total_evals"],
                        "accepted_moves": solver_stats[arm]["accepted_moves"],
                        "operator_stats": solver_stats[arm].get("operator_stats", {})
                    })

    df = pd.DataFrame(raw_records)
    unique_scenarios = sorted(list(df["scenario_id"].unique()))

    (
        results_matrix,
        summary_table,
        paired_comparisons_within_order,
        paired_summary_table,
        paired_comparisons_across_orders
    ) = aggregate_operator_study_results(
        raw_records=raw_records,
        order_strategies=order_strategies,
        ablation_arms=ablation_arms,
        H=H,
        alpha=alpha,
        n_boot=n_boot,
        master_seed=master_seed,
    )

    total_time = time.perf_counter() - t0

    metadata_dict = {
        "tier": "study",
        "subtier": "operator_order_factorial",
        "run_id": run_id,
        "seed": master_seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": round(total_time, 4),
        "ci_methods": {
            "n_boot": n_boot,
            "alpha": alpha,
            "primary_ci": "stratified_scenario_cluster_bootstrap_95",
            "parametric_scenario_ci": "scenario_aggregated_student_t_95",
            "sampling_level": "scenario_id",
            "aggregation_weighting": "equal_scenario_weight"
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **git_info
        },
        "experiment_config": {
            "num_instances": num_instances,
            "num_unique_scenarios": len(unique_scenarios),
            "num_designs": num_instances,
            "num_missions_per_instance": num_missions,
            "total_missions_evaluated": len(raw_records) * num_missions,
            "master_seed": master_seed,
            "H": H,
            "num_uavs": num_uavs,
            "max_evals_total": max_evals_total,
            "time_limit_sec": time_limit_sec,
            "order_strategies": order_strategies,
            "ablation_arms": ablation_arms,
            "gt_regimes": gt_regimes,
            "aggregation_weighting": "equal_scenario_weight"
        }
    }

    output_data = {
        "metadata": metadata_dict,
        "results_matrix": results_matrix,
        "paired_comparisons_within_order": paired_comparisons_within_order,
        "paired_comparisons_across_orders": paired_comparisons_across_orders,
        "summary_table": summary_table,
        "paired_summary_table": paired_summary_table,
        "raw_records": raw_records
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Archive run file to results/runs/{run_id}_operator_order_study.json
    runs_dir = out_file.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_archival_file = runs_dir / f"{run_id}_operator_order_study.json"
    with open(run_archival_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Operator order study complete in {total_time:.2f}s. Saved to {out_path} and {run_archival_file}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Factorial Operator Ordering & Ablation Study")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--num-missions", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--H", type=int, default=15, help="Planning horizon (default: 15 to match main benchmark)")
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--max-evals", type=int, default=600)
    parser.add_argument("--time-limit", type=float, default=15.0)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--out", type=str, default="results/operator_order_study_results.json")
    args = parser.parse_args()

    res = run_operator_order_study(
        num_instances=args.num_instances,
        num_missions=args.num_missions,
        master_seed=args.master_seed,
        H=args.H,
        num_uavs=args.num_uavs,
        max_evals_total=args.max_evals,
        time_limit_sec=args.time_limit,
        n_boot=args.n_boot,
        out_path=args.out
    )

    print("\n" + "="*85)
    print("OPERATOR ORDERING & ABLATION FACTORIAL STUDY RESULTS (Individual Arms)")
    print("="*85)
    df_res = pd.DataFrame(res["summary_table"])
    print(df_res.to_string(index=False))

    print("\n" + "="*85)
    print("PAIRED COMPARISONS (Proposed_Fast vs Each Ablation Arm per Ordering Strategy)")
    print("="*85)
    df_paired = pd.DataFrame(res["paired_summary_table"])
    print(df_paired.to_string(index=False))
