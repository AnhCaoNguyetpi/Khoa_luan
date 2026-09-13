"""Tier 3 Experiment: Main Comparative Benchmark across Baselines.

Corresponds to proposal Tier 3 & Table 2:
    - Benchmarks Proposed Search vs. Prefix-only, No-Replace/Rebalance Ablation,
      Greedy Lookahead, Adaptive GA, and Nominal Planning.
    - Evaluates on simulation missions using indexed CRN stream.
    - Computes DSR, RMST, actual energy, and Calibration Gap.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.experiments.rng import IndexedRNGStream
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.planning.baselines import AdaptiveGAPlanner, GreedyLookaheadPlanner
from sar_uav.sim.mission import ScheduleSimulator

log = logging.getLogger(__name__)


def run_tier3_benchmark(
    num_missions: int = 10,
    H: int = 15,
    num_uavs: int = 2,
    seed: int = 42
) -> Dict:
    """Runs the main comparative benchmark."""
    num_cells = 16  # 4x4 grid
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(4) for c in range(4)]
    dist_m = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 400.0

    uav_specs = [
        UAVSpec(name=f"uav_{k}", speed_ms=12.0, battery_joules=300000.0, reserve_joules=35000.0, delta_t=delta_t)
        for k in range(num_uavs)
    ]
    checker = ConstraintChecker(dist_m, depot_idx, uav_specs, H, delta_t=delta_t)

    # Markov motion model (diffusion on grid)
    M = np.eye(num_cells) * 0.7
    for i in range(num_cells):
        r, c = coords[i]
        nbrs = [j for j in range(num_cells) if i != j and abs(coords[j][0] - r) + abs(coords[j][1] - c) == 1]
        if nbrs:
            p_step = 0.3 / len(nbrs)
            for n in nbrs:
                M[i, n] = p_step

    b0 = np.full(num_cells, 1.0 / num_cells)
    b0[5] = 0.25
    b0[10] = 0.25
    b0 /= np.sum(b0)

    nominal_rates = np.full(num_cells, 0.18)
    veg_groups = np.array([0, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1])
    hset = HypothesisSet.create_structured_ensemble(nominal_rates, veg_groups, uncertainty_scale=0.5, num_hypotheses=3, seed=seed)
    hset_nom = HypothesisSet.create_nominal(nominal_rates)

    engine_ens = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    engine_nom = JointMassEngine(b0, hset_nom, M, H, delta_t=delta_t)

    candidate_cells = list(range(1, num_cells))

    # Planners
    eval_fast = ForwardBackwardFastEvaluator(engine_ens)
    eval_prefix = PrefixOnlyEvaluator(engine_ens)
    explorer_full = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=True, enable_dwell_rebalance=True
    )
    explorer_no_replace = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=False, enable_dwell_rebalance=True
    )
    explorer_no_rebalance = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=True, enable_dwell_rebalance=False
    )
    explorer_no_both = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=False, enable_dwell_rebalance=False
    )

    # Initial greedy construction (Ensemble & Nominal)
    greedy_planner_ens = GreedyLookaheadPlanner(checker, engine_ens, candidate_cells, default_dwell=2)
    sched_greedy_ens, J_greedy_ens, stats_greedy_ens = greedy_planner_ens.solve(num_uavs)

    greedy_planner_nom = GreedyLookaheadPlanner(checker, engine_nom, candidate_cells, default_dwell=2)
    sched_greedy_nom, J_greedy_nom, stats_greedy_nom = greedy_planner_nom.solve(num_uavs)

    # Score of initial nominal schedule under ensemble BEFORE any local search
    _, _, J_nom_under_ens = engine_ens.compute_forward_trajectory(
        sched_greedy_nom.generate_footprint_array()
    )
    J_after_greedy_nom_under_ensemble = float(J_nom_under_ens)

    # Standardized 2-start local search helper:
    # Divides total budget equally between start 1 (greedy_ensemble) and start 2 (nominal_greedy)
    def run_two_start_local_search(
        explorer,
        eval_factory,
        sched_start1: JointSchedule,
        sched_start2: JointSchedule,
        budget_total: int,
        time_limit: float,
        start1_name: str = "greedy_ensemble",
        start2_name: str = "nominal_greedy"
    ):
        budget_per_start = budget_total // 2
        time_per_start = time_limit / 2.0

        # Start 1: from greedy ensemble
        s1 = JointRouteEffortLocalSearch(
            explorer, eval_factory(), max_evals=budget_per_start, time_limit_sec=time_per_start
        )
        s1_sched, s1_J, s1_stats = s1.solve(sched_start1)

        # Start 2: from nominal greedy
        s2 = JointRouteEffortLocalSearch(
            explorer, eval_factory(), max_evals=budget_per_start, time_limit_sec=time_per_start
        )
        s2_sched, s2_J, s2_stats = s2.solve(sched_start2)

        starts = [
            {
                "start_idx": 1,
                "initial_name": start1_name,
                "J_init": float(s1_stats.get("initial_J", 0.0)),
                "J_final": float(s1_J),
                "n_evals": int(s1_stats["n_evals"]),
                "runtime_sec": float(s1_stats["runtime_sec"]),
                "accepted_moves": int(s1_stats["accepted_moves"]),
                "stopped_reason": str(s1_stats.get("stopped_reason", "unknown")),
                "operator_stats": s1_stats.get("operator_stats", {})
            },
            {
                "start_idx": 2,
                "initial_name": start2_name,
                "J_init": float(s2_stats.get("initial_J", 0.0)),
                "J_final": float(s2_J),
                "n_evals": int(s2_stats["n_evals"]),
                "runtime_sec": float(s2_stats["runtime_sec"]),
                "accepted_moves": int(s2_stats["accepted_moves"]),
                "stopped_reason": str(s2_stats.get("stopped_reason", "unknown")),
                "operator_stats": s2_stats.get("operator_stats", {})
            }
        ]

        if s1_J >= s2_J:
            best_sched, best_J, selected = s1_sched, s1_J, start1_name
            best_stats = s1_stats
        else:
            best_sched, best_J, selected = s2_sched, s2_J, start2_name
            best_stats = s2_stats

        ls_runtime = float(s1_stats["runtime_sec"] + s2_stats["runtime_sec"])
        ls_evals = int(s1_stats["n_evals"] + s2_stats["n_evals"])
        total_accepted = int(s1_stats["accepted_moves"] + s2_stats["accepted_moves"])

        meta = {
            "starts": starts,
            "selected_start": selected,
            "local_search_runtime_sec": ls_runtime,
            "local_search_evals": ls_evals,
            "accepted_moves": total_accepted,
            "stopped_reason": best_stats.get("stopped_reason", "unknown"),
            "operator_stats": best_stats.get("operator_stats", {})
        }
        return best_sched, best_J, meta

    # 1. Proposed Fast (2-start, fast evaluator, full explorer)
    sched_fast, J_fast, meta_fast = run_two_start_local_search(
        explorer_full,
        lambda: ForwardBackwardFastEvaluator(engine_ens),
        sched_greedy_ens, sched_greedy_nom,
        max_evals_total, time_limit
    )

    # 2a. Single-Start Fast (Fixed Local Search Budget = 600 evals)
    # Compares multi-start vs single-start under the exact same local search evaluation limit
    solver_single_fixed = JointRouteEffortLocalSearch(
        explorer_full, ForwardBackwardFastEvaluator(engine_ens),
        max_evals=max_evals_total, time_limit_sec=time_limit
    )
    sched_single_fixed, J_single_fixed, stats_single_fixed = solver_single_fixed.solve(sched_greedy_ens)

    # Accounting for greedy initialization costs
    rt_greedy_ens = float(stats_greedy_ens["runtime_sec"])
    ev_greedy_ens = int(stats_greedy_ens.get("n_evals", 0))
    rt_greedy_nom = float(stats_greedy_nom["runtime_sec"])
    ev_greedy_nom = int(stats_greedy_nom.get("n_evals", 0))
    rt_both_greedy = rt_greedy_ens + rt_greedy_nom
    ev_both_greedy = ev_greedy_ens + ev_greedy_nom

    # 2b. Single-Start Fast (Matched Total Evaluation Budget)
    # Total method evaluation budget of Proposed_Fast:
    #   N_total(Proposed) = ev_greedy_ens + ev_greedy_nom + max_evals_total (e.g., 190 + 228 + 600 = 1018 evals).
    # Single-start only performs one greedy initialization (ev_greedy_ens).
    # To match total evaluations across the entire method, we transfer the evaluation budget
    # that multi-start spent on the second greedy initialization (ev_greedy_nom)
    # directly into single-start's local search budget:
    #   matched_ls_budget = max_evals_total + ev_greedy_nom (e.g., 600 + 228 = 828 evals).
    # Resulting in exact equality of total method evaluations:
    #   N_total(SingleStart_Matched) = ev_greedy_ens + matched_ls_budget = 190 + 828 = 1018 evals.
    # NOTE:
    # 1. This explicitly matches total objective evaluations (evaluation count ceiling),
    #    NOT wall-clock runtime. Wall-clock runtime reflects actual operator execution costs.
    # 2. time_limit_sec=time_limit serves as the timeout ceiling for the local search phase
    #    (not total method elapsed wall-clock time including greedy initialization).
    matched_ls_budget = max_evals_total + ev_greedy_nom
    solver_single_matched = JointRouteEffortLocalSearch(
        explorer_full, ForwardBackwardFastEvaluator(engine_ens),
        max_evals=matched_ls_budget, time_limit_sec=time_limit
    )
    sched_single_matched, J_single_matched, stats_single_matched = solver_single_matched.solve(sched_greedy_ens)

    # 3. Proposed Prefix-Only (2-start, prefix evaluator, full explorer)
    # Fair comparison against Proposed Fast: identical multi-start, identical 300+300 budget
    sched_prefix, J_prefix, meta_prefix = run_two_start_local_search(
        explorer_full,
        lambda: PrefixOnlyEvaluator(engine_ens),
        sched_greedy_ens, sched_greedy_nom,
        max_evals_total, time_limit
    )

    # 4a. Operator Ablation: No Replace (keeps Dwell Rebalance)
    sched_no_repl, J_no_repl, meta_no_repl = run_two_start_local_search(
        explorer_no_replace,
        lambda: ForwardBackwardFastEvaluator(engine_ens),
        sched_greedy_ens, sched_greedy_nom,
        max_evals_total, time_limit
    )

    # 4b. Operator Ablation: No Dwell Rebalance (keeps Replace)
    sched_no_rebal, J_no_rebal, meta_no_rebal = run_two_start_local_search(
        explorer_no_rebalance,
        lambda: ForwardBackwardFastEvaluator(engine_ens),
        sched_greedy_ens, sched_greedy_nom,
        max_evals_total, time_limit
    )

    # 4c. Operator Ablation: Neither Replace nor Dwell Rebalance
    sched_no_both, J_no_both, meta_no_both = run_two_start_local_search(
        explorer_no_both,
        lambda: ForwardBackwardFastEvaluator(engine_ens),
        sched_greedy_ens, sched_greedy_nom,
        max_evals_total, time_limit
    )

    # 5. Metaheuristic Baseline (GA)
    ga_planner = AdaptiveGAPlanner(checker, engine_ens, candidate_cells, population_size=20, max_generations=25, seed=seed)
    sched_ga, J_ga, stats_ga = ga_planner.solve(num_uavs)

    # 6. Nominal Baseline (strictly decoupled from ensemble information)
    nominal_solver = JointRouteEffortLocalSearch(
        explorer_full, ForwardBackwardFastEvaluator(engine_nom),
        max_evals=max_evals_total, time_limit_sec=time_limit
    )
    sched_nom, J_nom_self, stats_nom = nominal_solver.solve(sched_greedy_nom)

    strategies = {
        "Proposed_Fast": (sched_fast, J_fast),
        "SingleStart_Fast_FixedLS": (sched_single_fixed, J_single_fixed),
        "SingleStart_Fast_MatchedTotal": (sched_single_matched, J_single_matched),
        "Proposed_PrefixOnly": (sched_prefix, J_prefix),
        "Ablation_NoReplace": (sched_no_repl, J_no_repl),
        "Ablation_NoRebalance": (sched_no_rebal, J_no_rebal),
        "Ablation_NoReplaceRebalance": (sched_no_both, J_no_both),
        "Greedy_Lookahead": (sched_greedy_ens, J_greedy_ens),
        "Adaptive_GA": (sched_ga, J_ga),
        "Nominal_Planning": (sched_nom, J_nom_self)
    }

    solver_stats = {
        "Proposed_Fast": {
            "total_runtime_sec": float(rt_both_greedy + meta_fast["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_fast["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_fast["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_fast["local_search_evals"]),
            "n_evals": int(meta_fast["local_search_evals"]),
            "accepted_moves": int(meta_fast["accepted_moves"]),
            "stopped_reason": meta_fast["stopped_reason"],
            "selected_start": meta_fast["selected_start"],
            "J_after_greedy_ens": float(J_greedy_ens),
            "J_after_greedy_nom_under_ensemble": float(J_after_greedy_nom_under_ensemble),
            "starts": meta_fast["starts"],
            "operator_stats": meta_fast.get("operator_stats", {})
        },
        "SingleStart_Fast_FixedLS": {
            "comparison_mode": "fixed_local_search_budget",
            "total_runtime_sec": float(rt_greedy_ens + stats_single_fixed["runtime_sec"]),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": float(stats_single_fixed["runtime_sec"]),
            "total_evals": int(ev_greedy_ens + stats_single_fixed["n_evals"]),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": int(stats_single_fixed["n_evals"]),
            "n_evals": int(stats_single_fixed["n_evals"]),
            "accepted_moves": int(stats_single_fixed["accepted_moves"]),
            "stopped_reason": stats_single_fixed.get("stopped_reason", "unknown"),
            "selected_start": "greedy_ensemble",
            "J_after_greedy_ens": float(J_greedy_ens),
            "starts": [
                {
                    "start_idx": 1,
                    "initial_name": "greedy_ensemble",
                    "J_init": float(stats_single_fixed.get("initial_J", J_greedy_ens)),
                    "J_final": float(J_single_fixed),
                    "n_evals": int(stats_single_fixed["n_evals"]),
                    "runtime_sec": float(stats_single_fixed["runtime_sec"]),
                    "accepted_moves": int(stats_single_fixed["accepted_moves"]),
                    "stopped_reason": str(stats_single_fixed.get("stopped_reason", "unknown")),
                    "operator_stats": stats_single_fixed.get("operator_stats", {})
                }
            ],
            "operator_stats": stats_single_fixed.get("operator_stats", {})
        },
        "SingleStart_Fast_MatchedTotal": {
            "comparison_mode": "matched_total_evaluation_budget",
            "total_runtime_sec": float(rt_greedy_ens + stats_single_matched["runtime_sec"]),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": float(stats_single_matched["runtime_sec"]),
            "total_evals": int(ev_greedy_ens + stats_single_matched["n_evals"]),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": int(stats_single_matched["n_evals"]),
            "n_evals": int(stats_single_matched["n_evals"]),
            "accepted_moves": int(stats_single_matched["accepted_moves"]),
            "stopped_reason": stats_single_matched.get("stopped_reason", "unknown"),
            "selected_start": "greedy_ensemble",
            "J_after_greedy_ens": float(J_greedy_ens),
            "starts": [
                {
                    "start_idx": 1,
                    "initial_name": "greedy_ensemble",
                    "J_init": float(stats_single_matched.get("initial_J", J_greedy_ens)),
                    "J_final": float(J_single_matched),
                    "n_evals": int(stats_single_matched["n_evals"]),
                    "runtime_sec": float(stats_single_matched["runtime_sec"]),
                    "accepted_moves": int(stats_single_matched["accepted_moves"]),
                    "stopped_reason": str(stats_single_matched.get("stopped_reason", "unknown")),
                    "operator_stats": stats_single_matched.get("operator_stats", {})
                }
            ],
            "operator_stats": stats_single_matched.get("operator_stats", {})
        },
        "Proposed_PrefixOnly": {
            "total_runtime_sec": float(rt_both_greedy + meta_prefix["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_prefix["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_prefix["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_prefix["local_search_evals"]),
            "n_evals": int(meta_prefix["local_search_evals"]),
            "accepted_moves": int(meta_prefix["accepted_moves"]),
            "stopped_reason": meta_prefix["stopped_reason"],
            "selected_start": meta_prefix["selected_start"],
            "J_after_greedy_ens": float(J_greedy_ens),
            "J_after_greedy_nom_under_ensemble": float(J_after_greedy_nom_under_ensemble),
            "starts": meta_prefix["starts"],
            "operator_stats": meta_prefix.get("operator_stats", {})
        },
        "Ablation_NoReplace": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_repl["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_repl["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_repl["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_repl["local_search_evals"]),
            "n_evals": int(meta_no_repl["local_search_evals"]),
            "accepted_moves": int(meta_no_repl["accepted_moves"]),
            "stopped_reason": meta_no_repl["stopped_reason"],
            "selected_start": meta_no_repl["selected_start"],
            "J_after_greedy_ens": float(J_greedy_ens),
            "J_after_greedy_nom_under_ensemble": float(J_after_greedy_nom_under_ensemble),
            "starts": meta_no_repl["starts"],
            "operator_stats": meta_no_repl.get("operator_stats", {})
        },
        "Ablation_NoRebalance": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_rebal["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_rebal["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_rebal["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_rebal["local_search_evals"]),
            "n_evals": int(meta_no_rebal["local_search_evals"]),
            "accepted_moves": int(meta_no_rebal["accepted_moves"]),
            "stopped_reason": meta_no_rebal["stopped_reason"],
            "selected_start": meta_no_rebal["selected_start"],
            "J_after_greedy_ens": float(J_greedy_ens),
            "J_after_greedy_nom_under_ensemble": float(J_after_greedy_nom_under_ensemble),
            "starts": meta_no_rebal["starts"],
            "operator_stats": meta_no_rebal.get("operator_stats", {})
        },
        "Ablation_NoReplaceRebalance": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_both["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_both["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_both["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_both["local_search_evals"]),
            "n_evals": int(meta_no_both["local_search_evals"]),
            "accepted_moves": int(meta_no_both["accepted_moves"]),
            "stopped_reason": meta_no_both["stopped_reason"],
            "selected_start": meta_no_both["selected_start"],
            "J_after_greedy_ens": float(J_greedy_ens),
            "J_after_greedy_nom_under_ensemble": float(J_after_greedy_nom_under_ensemble),
            "starts": meta_no_both["starts"],
            "operator_stats": meta_no_both.get("operator_stats", {})
        },
        "Greedy_Lookahead": {
            "total_runtime_sec": float(rt_greedy_ens),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": 0.0,
            "total_evals": int(ev_greedy_ens),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": 0,
            "n_evals": int(ev_greedy_ens),
            "J_after_greedy_ens": float(J_greedy_ens)
        },
        "Adaptive_GA": {
            "algorithm": "SimpleEvolutionaryPlanner",
            "description": "Standard genetic algorithm baseline with tournament selection (size 2), 1-point crossover, fixed uniform mutation (cell/dwell), and tail-repair elitism",
            "total_runtime_sec": float(stats_ga.get("runtime_sec", 0.0)),
            "local_search_runtime_sec": float(stats_ga.get("runtime_sec", 0.0)),
            "total_evals": int(stats_ga.get("n_evals", 0)),
            "local_search_evals": int(stats_ga.get("n_evals", 0)),
            "n_evals": int(stats_ga.get("n_evals", 0))
        },
        "Nominal_Planning": {
            "total_runtime_sec": float(rt_greedy_nom + stats_nom["runtime_sec"]),
            "greedy_runtime_sec": float(rt_greedy_nom),
            "local_search_runtime_sec": float(stats_nom["runtime_sec"]),
            "total_evals": int(ev_greedy_nom + stats_nom["n_evals"]),
            "greedy_evals": int(ev_greedy_nom),
            "local_search_evals": int(stats_nom["n_evals"]),
            "n_evals": int(stats_nom["n_evals"]),
            "accepted_moves": int(stats_nom["accepted_moves"]),
            "stopped_reason": stats_nom.get("stopped_reason", "unknown"),
            "selected_start": "nominal_greedy",
            "J_after_greedy": float(J_greedy_nom),
            "starts": [
                {
                    "start_idx": 1,
                    "initial_name": "nominal_greedy",
                    "J_init": float(stats_nom.get("initial_J", J_greedy_nom)),
                    "J_final": float(J_nom_self),
                    "n_evals": int(stats_nom["n_evals"]),
                    "runtime_sec": float(stats_nom["runtime_sec"]),
                    "accepted_moves": int(stats_nom["accepted_moves"]),
                    "stopped_reason": str(stats_nom.get("stopped_reason", "unknown")),
                    "operator_stats": stats_nom.get("operator_stats", {})
                }
            ],
            "operator_stats": stats_nom.get("operator_stats", {})
        }
    }

    # Simulation under ground truth
    rng_master = IndexedRNGStream(master_seed=seed)
    true_lambda = hset.hypotheses[1].lambda_rates
    engine_gt = JointMassEngine(b0, HypothesisSet.create_nominal(true_lambda), M, H, delta_t=delta_t)

    sim = ScheduleSimulator(checker, true_lambda, delta_t=delta_t)
    metrics_by_strat: Dict[str, List[Dict]] = {s: [] for s in strategies}
    mission_records: List[Dict] = []

    for m_idx in range(num_missions):
        # Generate true target path using Markov model
        target_path = np.zeros(H, dtype=int)
        init_rng = rng_master.get_rng(m_idx, 0, 0, "target_init")
        target_path[0] = int(init_rng.choice(num_cells, p=b0))
        for t in range(1, H):
            step_rng = rng_master.get_rng(m_idx, t, 0, "target_step")
            target_path[t] = int(step_rng.choice(num_cells, p=M[target_path[t - 1]]))

        m_entry = {
            "mission_idx": m_idx,
            "target_path": target_path.tolist(),
            "arms": {}
        }

        for s_name, (sched, J_self) in strategies.items():
            res = sim.run_simulation(
                sched,
                target_path,
                mission_idx=m_idx,
                rng_stream=rng_master,
                predicted_J=J_self
            )
            rec = {
                "detected": 1 if res.detected else 0,
                "rmst": res.rmst_time,
                "energy": res.actual_energy_joules,
                "calib_gap": res.calibration_gap
            }
            metrics_by_strat[s_name].append(rec)
            m_entry["arms"][s_name] = rec

        mission_records.append(m_entry)

    from sar_uav.experiments.stats import mcnemar_exact_p, student_t_ci, wilson_interval

    # Summarize per-arm metrics with Wilson CIs (proportion) and Student's t CIs (continuous)
    summary = {}
    for s_name, (sched, J_self) in strategies.items():
        records = metrics_by_strat[s_name]
        n_m = len(records)
        fp = sched.generate_footprint_array()
        _, _, J_ens = engine_ens.compute_forward_trajectory(fp)
        _, _, J_gt = engine_gt.compute_forward_trajectory(fp)

        k_det = sum(r["detected"] for r in records)
        dsr = float(k_det / max(1, n_m))
        ci_dsr = list(wilson_interval(k_det, n_m, alpha=0.05))

        rmst_vals = [r["rmst"] for r in records]
        # Student's t CI for small samples (df=n_m-1), clipped to physical domain [0, H]
        mean_rmst, se_rmst, ci_rmst = student_t_ci(rmst_vals, alpha=0.05, bounds=(0.0, float(H)))

        mean_energy = float(np.mean([r["energy"] for r in records]))

        summary[s_name] = {
            "J_self": float(J_self),
            "J_ensemble": float(J_ens),
            "J_true_gt": float(J_gt),
            "DSR": dsr,
            "DSR_ci95_wilson": ci_dsr,
            "mean_RMST_ticks": mean_rmst,
            "RMST_ci95": list(ci_rmst),
            "mean_energy_kJ": mean_energy / 1000.0,
            "calib_gap_self_vs_dsr": float(J_self - dsr),
            "calib_gap_gt_vs_dsr": float(J_gt - dsr)
        }

    # Compute paired comparisons (Proposed_Fast vs Each Baseline) using exact McNemar & Student's t CI
    paired_comparisons = {}
    prop_records = metrics_by_strat["Proposed_Fast"]
    n_m = len(prop_records)

    for b_name in strategies:
        if b_name == "Proposed_Fast":
            continue
        base_records = metrics_by_strat[b_name]

        # Paired DSR differences
        prop_det = np.array([r["detected"] for r in prop_records])
        base_det = np.array([r["detected"] for r in base_records])
        diff_dsr = prop_det - base_det
        mean_diff_dsr = float(np.mean(diff_dsr))

        # Discordant counts for McNemar test
        b_disc = int(np.sum((prop_det == 1) & (base_det == 0)))
        c_disc = int(np.sum((prop_det == 0) & (base_det == 1)))
        p_mcnemar = mcnemar_exact_p(b_disc, c_disc)

        # Paired RMST differences with Student's t CI (bounded in [-H, H])
        prop_rmst = np.array([r["rmst"] for r in prop_records])
        base_rmst = np.array([r["rmst"] for r in base_records])
        diff_rmst = prop_rmst - base_rmst
        mean_diff_rmst, se_diff_rmst, ci_diff_rmst = student_t_ci(
            diff_rmst, alpha=0.05, bounds=(-float(H), float(H))
        )

        # Paired Energy differences with Student's t CI
        prop_energy = np.array([r["energy"] for r in prop_records]) / 1000.0
        base_energy = np.array([r["energy"] for r in base_records]) / 1000.0
        diff_energy = prop_energy - base_energy
        mean_diff_energy, se_diff_energy, ci_diff_energy = student_t_ci(diff_energy, alpha=0.05)

        paired_comparisons[f"Proposed_vs_{b_name}"] = {
            "mean_delta_DSR": mean_diff_dsr,
            "discordant_pairs": {"proposed_wins": b_disc, "baseline_wins": c_disc},
            "mcnemar_p_value": p_mcnemar,
            "mean_delta_RMST_ticks": mean_diff_rmst,
            "ci95_delta_RMST": list(ci_diff_rmst),
            "mean_delta_energy_kJ": mean_diff_energy,
            "ci95_delta_energy_kJ": list(ci_diff_energy)
        }

    final_schedules = {
        s_name: sched.to_dict()
        for s_name, (sched, _) in strategies.items()
    }

    return {
        "experiment_config": {
            "H": H,
            "num_uavs": num_uavs,
            "num_missions": num_missions,
            "num_cells": num_cells,
            "max_evals_total": max_evals_total,
            "max_evals_per_start": max_evals_per_start,
            "time_limit_sec": time_limit,
            "seed": seed,
        },
        "summary": summary,
        "paired_comparisons": paired_comparisons,
        "solver_stats": solver_stats,
        "final_schedules": final_schedules,
        "mission_records": mission_records
    }


if __name__ == "__main__":
    res = run_tier3_benchmark(num_missions=5, H=12, num_uavs=2)
    print("Tier 3 Benchmark Summary:")
    for k, v in res["summary"].items():
        print(f"  {k}: {v}")
    print("\nPaired Comparisons:")
    for k, v in res["paired_comparisons"].items():
        print(f"  {k}: {v}")
