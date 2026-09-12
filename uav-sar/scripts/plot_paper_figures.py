"""Generate publication-ready figures for the conference paper.

Figures generated:
1. fig1_jensen_mechanism.png / .pdf:
   - Detection probability decay across hypotheses vs. vegetation cover
   - Marginal gain curves comparing true ensemble expectation vs. Jensen nominal upper bound.
2. fig2_evaluator_benchmark.png / .pdf:
   - Average evaluation latency (ms) across Full, Prefix-only, and Fast Evaluator.
   - Evaluator speedup and affected horizon length distribution (L_aff vs H).
3. fig3_tay_nguyen_case_study.png / .pdf:
   - Multi-UAV (3 UAVs) planned flight trajectories overlaid on real-world GIS
     terrain (Copernicus DEM elevation & ESA WorldCover vegetation) at Tay Nguyen.
"""

from __future__ import annotations

from _bootstrap import *  # noqa: F401,F403
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, FullEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch


def set_paper_style():
    """Applies a clean, modern IEEE/ACM conference style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })


def generate_figure1_jensen(out_dir: Path):
    """Figure 1: Mechanism & Jensen Bound on Detection Effort."""
    print("Generating Figure 1: Jensen Mechanism & Hypothesis Decay...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Subplot 1: Vegetation vs q^s(x) for different hypotheses
    veg = np.linspace(0.0, 0.8, 100)
    hypotheses = [
        {"name": r"$s_1$ (Optimistic, $\lambda=0.4$)", "q0": 0.85, "lam": 0.4, "color": "#2ca02c", "ls": "-"},
        {"name": r"$s_2$ (Nominal, $\lambda=1.0$)", "q0": 0.70, "lam": 1.0, "color": "#1f77b4", "ls": "-."},
        {"name": r"$s_3$ (Pessimistic, $\lambda=1.8$)", "q0": 0.50, "lam": 1.8, "color": "#d62728", "ls": ":"},
    ]

    q_vals = []
    weights = [0.25, 0.50, 0.25]
    for h in hypotheses:
        q = h["q0"] * np.exp(-h["lam"] * veg)
        q_vals.append(q)
        ax1.plot(veg, q, label=h["name"], color=h["color"], linestyle=h["ls"])

    # Mean rate (Nominal surrogate)
    q_mean = sum(w * q for w, q in zip(weights, q_vals))
    ax1.plot(veg, q_mean, label=r"Ensemble Mean $\bar{q}(x)$", color="#333333", linestyle="--", linewidth=2.2)

    ax1.set_xlabel("Vegetation Canopy Density $v(x)$")
    ax1.set_ylabel("Per-Tick Detection Rate $q^s(x)$")
    ax1.set_title("(a) Persistent Sensor Hypotheses")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")

    # Subplot 2: Cumulative detection probability P_det(tau) vs dwell time tau (Jensen Gap)
    tau = np.arange(1, 15)
    # Pick a moderate vegetation cell v = 0.45
    v_target = 0.45
    q_s_target = [h["q0"] * np.exp(-h["lam"] * v_target) for h in hypotheses]
    q_bar_target = sum(w * q for w, q in zip(weights, q_s_target))

    # True ensemble expectation: 1 - E[(1-q)^tau]
    p_true = 1.0 - sum(w * ((1.0 - q) ** tau) for w, q in zip(weights, q_s_target))
    # Jensen nominal bound (convexity of (1-q)^tau): 1 - (1 - E[q])^tau
    p_jensen = 1.0 - ((1.0 - q_bar_target) ** tau)

    ax2.plot(tau, p_jensen, label=r"Jensen Nominal Bound $1 - (1-\bar{q})^\tau$", color="#d62728", linestyle="--")
    ax2.plot(tau, p_true, label=r"True Ensemble $\mathbb{E}_\mathcal{S}[1 - (1-q^s)^\tau]$", color="#1f77b4", marker="o", markersize=4)
    ax2.fill_between(tau, p_true, p_jensen, color="#ff7f0e", alpha=0.25, label=r"Jensen Optimism Gap ($\Delta > 0$)")

    ax2.set_xlabel(r"Cumulative Dwell Search Effort $\tau$ (ticks)")
    ax2.set_ylabel(r"Cumulative Detection Probability $P_{\mathrm{det}}$")
    ax2.set_title("(b) Jensen Discrepancy & Over-optimism")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_jensen_mechanism.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_jensen_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved fig1_jensen_mechanism.png & .pdf")


def generate_figure2_evaluator(out_dir: Path, run_id: Optional[str] = None, results_dir: Optional[Path] = None):
    """Figure 2: Evaluator latency and speedup analysis from real experimental benchmarks."""
    print("Generating Figure 2: Evaluator Benchmark & Speedup...")
    if results_dir is None:
        results_dir = out_dir.parent

    if run_id:
        runs_dir = results_dir / "runs"
        tier2_path = runs_dir / f"{run_id}_tier_2.json"
        tier5_path = runs_dir / f"{run_id}_tier_5.json"
        print(f"Loading data from specific run: {run_id}")
    else:
        tier2_path = results_dir / "tier_2_results.json"
        tier5_path = results_dir / "tier_5_results.json"

    if not tier2_path.exists():
        raise FileNotFoundError(
            f"Required experimental results file not found: {tier2_path}. "
            f"Please run 'python scripts/run_experiments.py --tier 2,5' before generating paper figures."
        )
    if not tier5_path.exists():
        raise FileNotFoundError(
            f"Required experimental results file not found: {tier5_path}. "
            f"Please run 'python scripts/run_experiments.py --tier 2,5' before generating paper figures."
        )

    with open(tier2_path, "r", encoding="utf-8") as f:
        tier2_raw = json.load(f)
        tier2_data = tier2_raw["results"]
    with open(tier5_path, "r", encoding="utf-8") as f:
        tier5_raw = json.load(f)
        tier5_data = tier5_raw["results"]

    # Verify provenance and compatibility of result files
    meta_2 = tier2_raw.get("metadata", {})
    meta_5 = tier5_raw.get("metadata", {})
    run_id_2 = meta_2.get("run_id", "unknown")
    run_id_5 = meta_5.get("run_id", "unknown")
    env_2 = meta_2.get("environment", {})
    env_5 = meta_5.get("environment", {})

    is_same_run = (run_id_2 == run_id_5 and run_id_2 != "unknown")
    hash_2 = env_2.get("project_tree_sha256") or env_2.get("source_sha256") or env_2.get("git_hash")
    hash_5 = env_5.get("project_tree_sha256") or env_5.get("source_sha256") or env_5.get("git_hash")

    if is_same_run:
        if hash_2:
            compat_status = "single_run_verified"
            print(f"✓ Provenance verified: Both Tier 2 and Tier 5 originate from common run '{run_id_2}' (code hash: {hash_2})")
        else:
            compat_status = "single_run_missing_hash"
            print(f"ℹ Provenance notice: Both Tier 2 and Tier 5 originate from run '{run_id_2}', but code version hash is missing.")
    else:
        print(f"ℹ Notice: Tier 2 (run {run_id_2}) and Tier 5 (run {run_id_5}) were recorded in distinct runs.\n"
              f"  Tier 2 timestamp: {meta_2.get('timestamp')}, seed: {meta_2.get('seed')}\n"
              f"  Tier 5 timestamp: {meta_5.get('timestamp')}, seed: {meta_5.get('seed')}")
        # Validate algorithmic and environment compatibility
        if meta_2.get("seed") != meta_5.get("seed"):
            print(f"  ⚠ Caution: Master random seeds differ ({meta_2.get('seed')} vs {meta_5.get('seed')}).")
        
        # Explicit 3-state verification: matching, mismatch, or insufficient info
        if hash_2 and hash_5:
            if hash_2 == hash_5:
                compat_status = "distinct_runs_hash_match"
                print(f"  ✓ Code version compatibility verified (hash: {hash_2})")
            else:
                compat_status = "distinct_runs_hash_mismatch"
                print(f"  ⚠ Caution: Source version hashes differ ({hash_2} vs {hash_5}).")
        else:
            compat_status = "distinct_runs_insufficient_metadata"
            print("  ℹ Notice: Incomplete provenance metadata (source/git hash missing in one or both runs), compatibility cannot be verified.")

    fig2_provenance = {
        "is_single_run": is_same_run,
        "compatibility_status": compat_status,
        "run_id_tier2": run_id_2,
        "run_id_tier5": run_id_5,
        "timestamp_tier2": meta_2.get("timestamp"),
        "timestamp_tier5": meta_5.get("timestamp"),
        "seed_tier2": meta_2.get("seed"),
        "seed_tier5": meta_5.get("seed"),
        "git_hash_tier2": env_2.get("git_hash"),
        "git_hash_tier5": env_5.get("git_hash"),
        "source_sha256_tier2": env_2.get("source_sha256"),
        "source_sha256_tier5": env_5.get("source_sha256"),
        "project_tree_sha256_tier2": env_2.get("project_tree_sha256"),
        "project_tree_sha256_tier5": env_5.get("project_tree_sha256"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Subplot (a): Measured computational latency per evaluation move from Tier 2
    evaluators = ["Full Evaluator\n(Naive)", "Prefix-Only\nEvaluator", "Fast Evaluator\n(Proposed)"]
    if "micro_benchmark" in tier2_data:
        lat_full = tier2_data["micro_benchmark"]["latency_full_ms"]
        lat_prefix = tier2_data["micro_benchmark"]["latency_prefix_ms"]
        lat_fast = tier2_data["micro_benchmark"]["latency_fast_ms"]
    else:
        lat_full = tier2_data["evaluator_stats_full"]["avg_time_ms"]
        lat_prefix = tier2_data["evaluator_stats_prefix"]["avg_time_ms"]
        lat_fast = tier2_data["evaluator_stats_fast"]["avg_time_ms"]

    latencies = [lat_full, lat_prefix, lat_fast]
    colors = ["#7f7f7f", "#3b528b", "#21918c"]

    bars = ax1.bar(evaluators, latencies, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean Latency per Move (ms)")
    ax1.set_title("(a) Computational Latency per Evaluation (Tier 2)")
    ax1.set_ylim(0.0, max(latencies) * 1.35)

    base_lat = latencies[0]
    for b in bars:
        h = b.get_height()
        speedup = base_lat / max(1e-6, h)
        ax1.annotate(f"{h*1000.0:.1f} µs\n({speedup:.2f}×)",
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot (b): Speedup scaling across problem dimensions from Tier 5
    # Extract keys sorted by grid_size and num_uavs
    grid_sizes = sorted(list({v["grid_size"] for v in tier5_data.values()}))
    uav_counts = sorted(list({v["num_uavs"] for v in tier5_data.values()}))

    palette = {1: ("#1f77b4", "o"), 2: ("#2ca02c", "s"), 3: ("#d62728", "^")}

    for u in uav_counts:
        x_vals = []
        speedups_eval = []
        speedups_total = []
        for g in grid_sizes:
            key = f"grid_{g}_uav_{u}"
            if key in tier5_data:
                rec = tier5_data[key]
                x_vals.append(g)
                speedups_eval.append(rec["speedup_vs_full"])
                speedups_total.append(rec["total_speedup_vs_full"])

        col, marker = palette.get(u, ("#333333", "d"))
        if x_vals:
            ax2.plot(x_vals, speedups_eval, marker=marker, color=col, linestyle="-",
                     label=f"Eval Speedup ($K={u}$ UAVs)")
            ax2.plot(x_vals, speedups_total, marker=marker, color=col, linestyle="--", alpha=0.7,
                     label=f"Solver Speedup ($K={u}$ UAVs)")

    ax2.axhline(1.0, color="gray", linestyle=":", label="Baseline (Full Evaluator = 1.0×)")
    ax2.set_xlabel("Grid Size $N$ (Cells)")
    ax2.set_ylabel("Measured Speedup Factor vs Full")
    ax2.set_title("(b) Speedup across Problem Scales (Tier 5)")
    ax2.set_xticks(grid_sizes)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_evaluator_benchmark.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_evaluator_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved fig2_evaluator_benchmark.png & .pdf from real experimental data")
    return fig2_provenance


def generate_figure3_tay_nguyen(out_dir: Path):
    """Figure 3: Multi-UAV Search trajectories on Tay Nguyen real GIS terrain."""
    print("Generating Figure 3: Tay Nguyen Real GIS Case Study...")
    gis_dir = Path("data/areas/tay_nguyen_real")
    layers = np.load(gis_dir / "layers.npz")
    with open(gis_dir / "area.json", "r", encoding="utf-8") as f:
        area_meta = json.load(f)

    elev = layers["elev"]      # 40x40 Copernicus DEM
    veg = layers["veg"]        # 40x40 ESA WorldCover
    trail_dist = layers["dist_trail"]

    H_grid, W_grid = elev.shape
    num_cells = H_grid * W_grid
    delta_t = 60.0
    depot_idx = 0  # (0, 0)

    # Calculate distance matrix (Euclidean meters between cells, cell=100m)
    coords = [(r, c) for r in range(H_grid) for c in range(W_grid)]
    
    # Fast sub-region for clear visualization: 20x20 window around search area
    sub_r, sub_c = 20, 20
    sub_coords = coords[:sub_r*sub_c]
    sub_elev = elev[:sub_r, :sub_c]
    sub_veg = veg[:sub_r, :sub_c]
    sub_dist_trail = trail_dist[:sub_r, :sub_c]
    N_sub = sub_r * sub_c

    dist_m = np.zeros((N_sub, N_sub))
    for i, (r1, c1) in enumerate(sub_coords):
        for j, (r2, c2) in enumerate(sub_coords):
            dist_m[i, j] = np.hypot((r1 - r2) * 100.0, (c1 - c2) * 100.0)

    # Fleet of 3 UAVs
    uav_specs = [
        UAVSpec(name="UAV 1", speed_ms=15.0, battery_joules=500000.0, reserve_joules=50000.0, delta_t=delta_t),
        UAVSpec(name="UAV 2", speed_ms=15.0, battery_joules=500000.0, reserve_joules=50000.0, delta_t=delta_t),
        UAVSpec(name="UAV 3", speed_ms=15.0, battery_joules=500000.0, reserve_joules=50000.0, delta_t=delta_t),
    ]
    H_mission = 24
    checker = ConstraintChecker(dist_m, depot_idx, uav_specs, H_mission, delta_t=delta_t)

    # Prior belief based on trail proximity
    b0 = np.exp(-sub_dist_trail.flatten() / 300.0)
    b0 /= np.sum(b0)

    # Sensor hypothesis set based on real vegetation
    veg_flat = sub_veg.flatten()
    q0_mean = 0.8
    nom_rates = q0_mean * np.exp(-1.0 * veg_flat)
    hset = HypothesisSet([
        DetectionHypothesis(id=0, name="Optimistic", lambda_rates=0.9 * np.exp(-0.6 * veg_flat), weight=0.25),
        DetectionHypothesis(id=1, name="Nominal", lambda_rates=0.75 * np.exp(-1.0 * veg_flat), weight=0.50),
        DetectionHypothesis(id=2, name="Pessimistic", lambda_rates=0.6 * np.exp(-1.5 * veg_flat), weight=0.25),
    ])

    # Transition matrix (stationary diffusion)
    M = np.eye(N_sub) * 0.85 + 0.15 / N_sub
    engine = JointMassEngine(b0, hset, M, H_mission, delta_t=delta_t)

    # Initial Schedule for 3 UAVs targeting high prior regions
    # Sort cells by prior
    top_prior_cells = np.argsort(b0)[::-1]
    
    # 3 diverse initial schedules
    sched1 = UAVSchedule(uav_idx=0, visits=[
        Visit(cell_idx=int(top_prior_cells[2]), dwell_ticks=3),
        Visit(cell_idx=int(top_prior_cells[8]), dwell_ticks=2)
    ])
    sched2 = UAVSchedule(uav_idx=1, visits=[
        Visit(cell_idx=int(top_prior_cells[4]), dwell_ticks=3),
        Visit(cell_idx=int(top_prior_cells[12]), dwell_ticks=2)
    ])
    sched3 = UAVSchedule(uav_idx=2, visits=[
        Visit(cell_idx=int(top_prior_cells[6]), dwell_ticks=3),
        Visit(cell_idx=int(top_prior_cells[15]), dwell_ticks=2)
    ])

    joint_init = JointSchedule(uav_schedules=[sched1, sched2, sched3], num_cells=N_sub, H=H_mission, delta_t=delta_t)

    # Solve with Local Search
    evaluator = ForwardBackwardFastEvaluator(engine)
    explorer = NeighborhoodExplorer(checker, candidate_cells=list(top_prior_cells[:40]), allowed_dwells=(1, 2, 3, 4))
    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=600, time_limit_sec=15.0, verbose=False)
    best_sched, best_J, stats = solver.solve(joint_init)

    # Plot GIS map with routes
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Background: Elevation contour + Vegetation transparency
    im = ax.imshow(sub_elev, cmap="terrain", origin="upper", extent=[0, sub_c*100, sub_r*100, 0])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Copernicus DEM Elevation (m)")

    # Overlay Vegetation as stipple/alpha
    ax.imshow(sub_veg, cmap="Greens", alpha=0.35, origin="upper", extent=[0, sub_c*100, sub_r*100, 0])

    # Overlay high prior contours
    ax.contour(np.linspace(0, sub_c*100, sub_c), np.linspace(0, sub_r*100, sub_r),
               b0.reshape(sub_r, sub_c), levels=4, colors="#d62728", alpha=0.6, linewidths=1.0)

    # Plot Depot
    depot_r, depot_c = sub_coords[depot_idx]
    ax.scatter([depot_c * 100], [depot_r * 100], color="black", marker="s", s=120, zorder=5, label="Depot Base")

    # Plot routes for each UAV
    uav_colors = ["#e41a1c", "#377eb8", "#984ea3"]
    for uav_idx, u_sched in enumerate(best_sched.uav_schedules):
        color = uav_colors[uav_idx % len(uav_colors)]
        pts_x = [depot_c * 100]
        pts_y = [depot_r * 100]

        for v in u_sched.visits:
            vr, vc = sub_coords[v.cell_idx]
            pts_x.append(vc * 100)
            pts_y.append(vr * 100)
            # Draw dwell circle
            dwell_circle = plt.Circle((vc * 100, vr * 100), radius=25 + v.dwell_ticks * 15,
                                      color=color, fill=False, linewidth=2.0, linestyle="--")
            ax.add_patch(dwell_circle)
            ax.annotate(f"$\\tau={v.dwell_ticks}$", (vc * 100 + 10, vr * 100 - 10),
                        fontsize=8, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

        # Return to depot
        pts_x.append(depot_c * 100)
        pts_y.append(depot_r * 100)

        # Plot flight path line
        ax.plot(pts_x, pts_y, color=color, linewidth=2.4, marker="o", markersize=6,
                label=f"UAV {uav_idx+1} Route")

    ax.set_title(f"Multi-UAV Coordinated Search Plan (Tay Nguyen Real GIS)\n"
                 f"Horizon $H={H_mission}$ ticks (24 min) | Predicted Detection Prob $J^*={best_J:.3f}$",
                 fontsize=11)
    ax.set_xlabel("Easting Distance (m)")
    ax.set_ylabel("Northing Distance (m)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_tay_nguyen_case_study.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_tay_nguyen_case_study.pdf", bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved fig3_tay_nguyen_case_study.png & .pdf")


def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Generate publication-ready figures for UAV SAR paper.")
    parser.add_argument("--run-id", type=str, default=None, help="Target specific archived experiment run ID")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experimental results")
    parser.add_argument("--out-dir", type=str, default="results/figures", help="Directory to save generated figures")
    args = parser.parse_args()

    set_paper_style()
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir.resolve()}")
    if args.run_id:
        print(f"Targeting experiment run ID: {args.run_id}")
    print()

    generate_figure1_jensen(out_dir)
    fig2_provenance = generate_figure2_evaluator(out_dir, run_id=args.run_id, results_dir=results_dir)
    generate_figure3_tay_nguyen(out_dir)

    # Save comprehensive figure provenance manifest
    provenance_doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_run_id": args.run_id,
        "results_directory": str(results_dir.resolve()),
        "figures": {
            "fig1_jensen_mechanism": {
                "source": "analytical_model",
                "description": "Vegetation-conditioned detection decay curves & Jensen bound"
            },
            "fig2_evaluator_benchmark": fig2_provenance,
            "fig3_tay_nguyen_case_study": {
                "source": "gis_optimization",
                "elevation_dataset": "Copernicus DEM 100m",
                "vegetation_dataset": "ESA WorldCover 100m",
                "H": 24,
                "num_uavs": 3
            }
        }
    }
    provenance_file = out_dir / "figures_provenance.json"
    with open(provenance_file, "w", encoding="utf-8") as f:
        json.dump(provenance_doc, f, indent=2)
    print(f"✓ Saved figures_provenance.json in {out_dir}")
    print("\nAll 3 figures and provenance metadata successfully created!")


if __name__ == "__main__":
    main()
