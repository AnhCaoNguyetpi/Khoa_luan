"""Generate publication-ready figures for Multi-Instance Benchmark & Operator Ablation.

Figures generated:
1. fig4_multi_instance_benchmark.png / .pdf:
   - Panel A: Per-instance DSR across 10 problem instances stratified by Ground Truth regime.
   - Panel B: Paired DSR Difference (Proposed vs Baselines) with 95% Student's t CIs.
   - Panel C: Detection Rate (DSR) vs. Mean Detection Time (RMST) Pareto frontier.
2. fig5_operator_ablation.png / .pdf:
   - Panel A: Objective J_ens vs. Empirical DSR across 4 ablation configurations.
   - Panel B: Evaluated vs. Accepted candidates per neighborhood operator.
   - Panel C: Cumulative Objective Improvement contribution (Delta J) per operator.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def set_paper_style():
    """Applies clean IEEE/ACM conference style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })


def plot_fig4_multi_instance(data: dict, out_dir: Path):
    """Figure 4: Multi-instance benchmark generalization & regime stratification."""
    print("Generating Figure 4: Multi-Instance Benchmark Generalization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    instances = data["instances"]
    n_inst = len(instances)
    regime_abbr = {
        "in_ensemble": "ens",
        "nominal_matched": "nom",
        "misspecified_out_of_ensemble": "mis"
    }
    names = [
        f"S{inst.get('scenario_id', i % 10)}.{regime_abbr.get(inst.get('gt_category', ''), str(i))}"
        for i, inst in enumerate(instances)
    ]
    x = np.arange(n_inst)

    n_missions = data.get("metadata", {}).get(
        "experiment_config", {}
    ).get("num_missions_per_instance", data.get("benchmark_metadata", {}).get("num_missions_per_instance", 100))

    # Panel A: DSR per instance with dynamic regime background shading
    prop_dsr = [inst["summary"]["Proposed_Fast"]["DSR"] for inst in instances]
    nom_dsr = [inst["summary"]["Nominal_Planning"]["DSR"] for inst in instances]
    greedy_dsr = [inst["summary"]["Greedy_Lookahead"]["DSR"] for inst in instances]
    ga_dsr = [inst["summary"]["Adaptive_GA"]["DSR"] for inst in instances]

    # Dynamic background shading for contiguous regimes
    regime_configs = {
        "in_ensemble": ("#e6f2ff", "In-Ensemble GT"),
        "nominal_matched": ("#fff2e6", "Nominal-Matched GT"),
        "misspecified_out_of_ensemble": ("#f0e6ff", "Misspecified GT"),
    }
    blocks = []
    if instances:
        curr_cat = instances[0].get("gt_category", "in_ensemble")
        start = 0
        for i in range(1, len(instances)):
            cat = instances[i].get("gt_category", "in_ensemble")
            if cat != curr_cat:
                blocks.append((curr_cat, start, i - 1))
                curr_cat = cat
                start = i
        blocks.append((curr_cat, start, len(instances) - 1))

    seen_labels = set()
    for cat, start_idx, end_idx in blocks:
        color, lbl = regime_configs.get(cat, ("#f9f9f9", cat))
        label_to_show = lbl if lbl not in seen_labels else None
        if label_to_show:
            seen_labels.add(lbl)
        ax1.axvspan(start_idx - 0.5, end_idx + 0.5, color=color, alpha=0.6, label=label_to_show)

    width = 0.20
    ax1.bar(x - 1.5 * width, prop_dsr, width, label="Proposed_Fast", color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax1.bar(x - 0.5 * width, nom_dsr, width, label="Nominal_Planning", color="#ff7f0e", edgecolor="black", linewidth=0.8)
    ax1.bar(x + 0.5 * width, greedy_dsr, width, label="Greedy_Lookahead", color="#2ca02c", edgecolor="black", linewidth=0.8)
    ax1.bar(x + 1.5 * width, ga_dsr, width, label="GA baseline", color="#d62728", edgecolor="black", linewidth=0.8)

    meta = data.get("metadata", {}).get("experiment_config", {}) or data.get("benchmark_metadata", {})
    unique_scenarios = set(inst.get("scenario_id", i % 10) for i, inst in enumerate(instances))
    n_scenarios = meta.get("num_unique_scenarios", len(unique_scenarios))
    unique_regimes = set(inst.get("gt_category", "in_ensemble") for inst in instances)
    n_regimes = len(unique_regimes)

    ax1.set_xlabel("Evaluated Instance (Scenario ID . GT Regime)")
    ax1.set_ylabel(f"Empirical DSR ({n_missions} missions/eval)")
    ax1.set_title(f"(a) Performance Across {n_inst} Evaluations ({n_scenarios} Scenarios × {n_regimes} GT Regimes)", fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=6.5)
    all_dsr = prop_dsr + nom_dsr + greedy_dsr + ga_dsr
    max_dsr = max(all_dsr) if all_dsr else 0.5
    ax1.set_ylim(0.0, min(1.0, max_dsr * 1.25))
    ax1.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=7.5)

    # Panel B: Paired differences (Proposed vs Baselines) across instances
    candidate_paired_keys = [
        ("Nominal_2Start", "#17becf"),
        ("Nominal_Planning", "#ff7f0e"),
        ("SingleStart_Fast_FixedLS", "#8c564b"),
        ("SingleStart_Fast_MatchedTotal", "#e377c2"),
        ("Greedy_Lookahead", "#2ca02c"),
        ("Adaptive_GA", "#d62728")
    ]
    paired_keys = [
        pk for pk in candidate_paired_keys
        if f"Proposed_vs_{pk[0]}" in data.get("cross_instance_paired", {})
    ]
    b_names = [k.replace("_Planning", " (1-start)").replace("_2Start", " (2-start)").replace("_Fast", "").replace("SingleStart_", "SS_").replace("Adaptive_GA", "GA baseline") for k, _ in paired_keys]
    means = [data["cross_instance_paired"][f"Proposed_vs_{k}"]["mean_delta_DSR"] for k, _ in paired_keys]
    ci_lowers = [data["cross_instance_paired"][f"Proposed_vs_{k}"]["ci95_delta_DSR"][0] for k, _ in paired_keys]
    ci_uppers = [data["cross_instance_paired"][f"Proposed_vs_{k}"]["ci95_delta_DSR"][1] for k, _ in paired_keys]
    colors = [c for _, c in paired_keys]

    y_pos = np.arange(len(b_names))
    xerr = [
        [m - lo for m, lo in zip(means, ci_lowers)],
        [hi - m for m, hi in zip(means, ci_uppers)]
    ]

    ax2.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.errorbar(means, y_pos, xerr=xerr, fmt="o", color="#1f77b4", ecolor="gray", elinewidth=2, capsize=5, markersize=7)
    for i, (m, col) in enumerate(zip(means, colors)):
        ax2.scatter(m, i, color=col, s=70, zorder=5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b_names)
    ax2.set_xlabel(r"Mean Paired $\Delta$DSR (Proposed $-$ Baseline)")
    ax2.set_title(r"(b) Cross-Scenario Paired $\Delta$DSR (95% Scenario Bootstrap CI)", fontsize=9.5)
    ax2.invert_yaxis()

    # Panel C: DSR vs RMST Pareto Frontier
    all_strats = ["Proposed_Fast", "Nominal_2Start", "SingleStart_Fast_FixedLS", "SingleStart_Fast_MatchedTotal", "Greedy_Lookahead", "Adaptive_GA", "Nominal_Planning"]
    strats = [s for s in all_strats if s in data.get("cross_instance_summary", {})]
    markers = {"Proposed_Fast": "*", "Nominal_2Start": "P", "SingleStart_Fast_FixedLS": "s", "SingleStart_Fast_MatchedTotal": "D", "Greedy_Lookahead": "^", "Adaptive_GA": "v", "Nominal_Planning": "o"}
    strat_colors = {"Proposed_Fast": "#1f77b4", "Nominal_2Start": "#17becf", "SingleStart_Fast_FixedLS": "#8c564b", "SingleStart_Fast_MatchedTotal": "#e377c2", "Greedy_Lookahead": "#2ca02c", "Adaptive_GA": "#d62728", "Nominal_Planning": "#ff7f0e"}

    for s in strats:
        mean_dsr = data["cross_instance_summary"][s]["mean_DSR"]
        mean_rmst = data["cross_instance_summary"][s]["mean_RMST_ticks"]
        msize = 120 if s == "Proposed_Fast" else 80
        lbl = s.replace("_Planning", " (1-start)").replace("_2Start", " (2-start)").replace("_Fast", "").replace("SingleStart_", "SS_").replace("Adaptive_GA", "GA baseline")
        ax3.scatter(mean_rmst, mean_dsr, marker=markers[s], color=strat_colors[s], s=msize, label=lbl, zorder=5)

    ax3.set_xlabel("Mean Detection Time RMST (ticks, lower is better)")
    ax3.set_ylabel("Mean Empirical DSR (higher is better)")
    ax3.set_title("(c) Multi-Objective Quality (DSR vs. RMST)", fontsize=9.5)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_multi_instance_benchmark.png", dpi=300)
    fig.savefig(out_dir / "fig4_multi_instance_benchmark.pdf")
    plt.close(fig)
    print("Figure 4 saved successfully.")


def plot_fig5_operator_ablation(data: dict, out_dir: Path):
    """Figure 5: 4-Arm Operator Ablation & Operator-Level Contribution."""
    print("Generating Figure 5: Operator Ablation & Usage Breakdown...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    ablation_data = data["ablation_summary"]
    arms = ["Proposed_Fast", "Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"]
    arm_labels = ["Full\n(Repl+Rebal)", "No Replace\n(Rebal only)", "No Rebalance\n(Repl only)", "Neither\n(Base 6 ops)"]
    x = np.arange(len(arms))

    # Panel A: Expected Objective J_ens vs Empirical DSR
    j_vals = [ablation_data[a]["mean_J_ensemble"] for a in arms]
    dsr_vals = [ablation_data[a]["mean_DSR"] for a in arms]

    width = 0.35
    bars1 = ax1.bar(x - width / 2, j_vals, width, label=r"Predicted $\mathcal{J}_{\mathcal{S}}$", color="#4a90e2", edgecolor="black", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, dsr_vals, width, label="Empirical DSR", color="#50e3c2", edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_labels)
    ax1.set_ylabel("Metric Value")
    ax1.set_title(r"(a) Expected Objective $\mathcal{J}_{\mathcal{S}}$ vs. Empirical DSR")
    
    # Ground bar chart at 0.0 with data-driven upper limit and exact value annotations
    all_vals = j_vals + dsr_vals
    max_val = max(all_vals) if all_vals else 0.35
    ax1.set_ylim(0.0, min(1.0, max_val * 1.25))
    
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel B: Operator Evaluation and Acceptance Counts in Proposed_Fast
    prop_ops = ablation_data["Proposed_Fast"]["aggregated_operator_stats"]
    op_names = ["Replace", "DwellRebal", "Swap", "ChangeDwell", "Insert", "Relocate"]
    op_keys = ["Replace", "DwellRebalance", "Swap", "ChangeDwell", "Insert", "Relocate"]

    eval_counts = [prop_ops.get(k, {}).get("evaluated", 0) for k in op_keys]
    accept_counts = [prop_ops.get(k, {}).get("accepted", 0) for k in op_keys]

    x_ops = np.arange(len(op_names))
    ax2.bar(x_ops, eval_counts, color="#d3d3d3", edgecolor="black", label="Candidates Evaluated", linewidth=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_ops, accept_counts, color="#d9534f", marker="o", linewidth=2.0, label="Moves Accepted")

    ax2.set_xticks(x_ops)
    ax2.set_xticklabels(op_names, rotation=25, ha="right")
    ax2.set_ylabel("Candidates Evaluated (Breadth)")
    ax2_twin.set_ylabel("Moves Accepted (Productivity)", color="#d9534f")
    ax2.set_title("(b) Neighborhood Search Breadth vs. Acceptance")

    # Panel C: Cumulative Objective Improvement per Operator
    improvements = [prop_ops.get(k, {}).get("total_improvement", 0.0) for k in op_keys]
    bar_colors = ["#2b5c8f", "#417dc1", "#84a9c0", "#e27d60", "#c38d9e", "#41b3a3"]
    ax3.bar(op_names, improvements, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax3.set_xticks(x_ops)
    ax3.set_xticklabels(op_names, rotation=25, ha="right")
    ax3.set_ylabel(r"Cumulative Objective Improvement $\sum \Delta J$")
    ax3.set_title("(c) Total Objective Gain by Operator")

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_operator_ablation.png", dpi=300)
    fig.savefig(out_dir / "fig5_operator_ablation.pdf")
    plt.close(fig)
    print("Figure 5 saved successfully.")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_fig4_compatibility(data: dict) -> tuple[bool, dict[str, bool]]:
    """Check explicit verification conditions for multi-instance benchmark data."""
    meta = data.get("metadata", data.get("benchmark_metadata", {}))
    cfg = meta.get("experiment_config", {})
    instances = data.get("instances", [])
    summary = data.get("cross_instance_summary", {})
    paired = data.get("cross_instance_paired", {})

    required_strats = [
        "Proposed_Fast", "Nominal_Planning", "SingleStart_Fast_FixedLS",
        "SingleStart_Fast_MatchedTotal", "Greedy_Lookahead", "Adaptive_GA"
    ]
    required_paired = [
        "Proposed_vs_Nominal_Planning",
        "Proposed_vs_SingleStart_Fast_FixedLS",
        "Proposed_vs_SingleStart_Fast_MatchedTotal",
        "Proposed_vs_Greedy_Lookahead",
        "Proposed_vs_Adaptive_GA"
    ]

    expected_budget = cfg.get("max_evals_total")
    budget_consistent = False
    if expected_budget is not None and expected_budget > 0 and len(instances) > 0:
        consistent = True
        for inst in instances:
            fixed_ls_stats = inst.get("solver_stats", {}).get("SingleStart_Fast_FixedLS")
            if not fixed_ls_stats:
                consistent = False
                break
            # FixedLS must be allocated max_evals_total as its local search budget ceiling
            inst_budget = fixed_ls_stats.get("max_evals_budget")
            if inst_budget != expected_budget:
                consistent = False
                break
            # Explicitly require local_search_evals; do not guess using total_evals (which includes greedy)
            if "local_search_evals" not in fixed_ls_stats:
                consistent = False
                break
            actual_evals = fixed_ls_stats["local_search_evals"]
            if not isinstance(actual_evals, (int, float)) or actual_evals < 0:
                consistent = False
                break
            # Actual evaluation count used must not exceed the allocated budget ceiling
            if actual_evals > inst_budget:
                consistent = False
                break
        budget_consistent = consistent

    expected_evals_count = cfg.get("num_evaluations_total", cfg.get("num_instances", len(instances)))

    checks = {
        "run_id_present": meta.get("run_id") not in (None, "unknown", ""),
        "git_hash_present": meta.get("environment", {}).get("git_hash") not in (None, "unknown", ""),
        "instances_non_empty": len(instances) > 0,
        "instances_count_matches_config": len(instances) == expected_evals_count,
        "strategies_complete": all(s in summary for s in required_strats),
        "paired_stats_complete": all(p in paired for p in required_paired),
        "budget_consistent_fixed_ls": budget_consistent,
    }
    all_passed = all(checks.values())
    return all_passed, checks


def verify_fig5_compatibility(data: dict) -> tuple[bool, dict[str, bool]]:
    """Check explicit verification conditions for ablation benchmark data."""
    meta = data.get("metadata", data.get("benchmark_metadata", {}))
    ablation = data.get("ablation_summary", {})
    required_arms = [
        "Proposed_Fast", "Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"
    ]
    prop_stats = ablation.get("Proposed_Fast", {}).get("aggregated_operator_stats")

    # Verify operator stats schema is valid (dict with numeric counts, or empty if no moves evaluated)
    stats_valid = isinstance(prop_stats, dict)
    if stats_valid and prop_stats:
        for op, counts in prop_stats.items():
            if not (isinstance(counts, dict) and counts.get("evaluated", 0) >= 0 and counts.get("accepted", 0) >= 0):
                stats_valid = False
                break

    checks = {
        "run_id_present": meta.get("run_id") not in (None, "unknown", ""),
        "git_hash_present": meta.get("environment", {}).get("git_hash") not in (None, "unknown", ""),
        "ablation_arms_complete": all(a in ablation for a in required_arms),
        "operator_stats_valid": stats_valid,
    }
    all_passed = all(checks.values())
    return all_passed, checks


if __name__ == "__main__":
    set_paper_style()
    results_path = Path("results/multi_instance_tier3_results.json")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_fig4_multi_instance(data, out_dir)
    plot_fig5_operator_ablation(data, out_dir)

    # Compute provenance metadata & perform explicit condition checks
    input_file_hash = compute_file_sha256(results_path)
    meta = data.get("metadata", data.get("benchmark_metadata", {}))
    env = meta.get("environment", {})

    run_id = meta.get("run_id", "unknown")
    git_hash = env.get("git_hash", "unknown")
    git_hash_full = env.get("git_hash_full", "unknown")
    git_branch = env.get("git_branch", "unknown")
    git_is_dirty = env.get("git_is_dirty", None)
    source_sha256 = env.get("source_sha256", "unknown")
    project_tree_sha256 = env.get("project_tree_sha256", "unknown")
    timestamp = meta.get("timestamp", "unknown")

    fig4_verified, fig4_checks = verify_fig4_compatibility(data)
    fig5_verified, fig5_checks = verify_fig5_compatibility(data)

    # Record provenance into figures_provenance.json
    prov_path = out_dir / "figures_provenance.json"
    prov_data = {}
    if prov_path.exists():
        try:
            with open(prov_path, "r", encoding="utf-8") as pf:
                prov_data = json.load(pf)
        except Exception:
            prov_data = {}

    if "figures" not in prov_data:
        prov_data["figures"] = {}

    prov_data["figures"]["fig4_multi_instance_benchmark"] = {
        "source": str(results_path),
        "input_file_sha256": input_file_hash,
        "run_id": run_id,
        "git_hash": git_hash,
        "git_hash_full": git_hash_full,
        "git_branch": git_branch,
        "git_is_dirty": git_is_dirty,
        "source_sha256": source_sha256,
        "project_tree_sha256": project_tree_sha256,
        "timestamp": timestamp,
        "num_instances": len(data.get("instances", [])),
        "num_missions_per_instance": meta.get("experiment_config", {}).get("num_missions_per_instance", 100),
        "provenance_status": "source_recorded",
        "verification_status": "verified" if fig4_verified else "unverified",
        "compatibility_status": "source_recorded_and_verified" if fig4_verified else "source_recorded_unverified",
        "verification_checks": fig4_checks
    }

    prov_data["figures"]["fig5_operator_ablation"] = {
        "source": str(results_path),
        "input_file_sha256": input_file_hash,
        "run_id": run_id,
        "git_hash": git_hash,
        "git_hash_full": git_hash_full,
        "git_branch": git_branch,
        "git_is_dirty": git_is_dirty,
        "source_sha256": source_sha256,
        "project_tree_sha256": project_tree_sha256,
        "timestamp": timestamp,
        "ablation_arms": list(data.get("ablation_summary", {}).keys()),
        "provenance_status": "source_recorded",
        "verification_status": "verified" if fig5_verified else "unverified",
        "compatibility_status": "source_recorded_and_verified" if fig5_verified else "source_recorded_unverified",
        "verification_checks": fig5_checks
    }

    with open(prov_path, "w", encoding="utf-8") as pf:
        json.dump(prov_data, pf, indent=2)

    print("All multi-instance figures and provenance generated successfully!")
