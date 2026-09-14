"""Multi-Instance Comparative Benchmark for UAV Search and Rescue.

Addresses the two-level experimental hierarchy:
1. Problem Instance Level: Prior distribution (b0), Markov motion drift (M),
   sensor detection rates, vegetation clutter, and ground truth alignment categories.
2. Mission Level: Realized target motion trajectories and stochastic detection
   events under Common Random Numbers (CRN).

Evaluates:
- Generalization of Proposed_Fast across diverse search topologies.
- Standardized evaluation budget vs wall-clock comparison (Matched Total vs Fixed LS).
- 4-arm Operator Ablation: Full, No-Replace, No-Rebalance, Neither.
- Ensemble vs Nominal contributions under 3 Ground Truth regimes:
  (a) In-Ensemble: ground truth corresponds to an ensemble hypothesis different from nominal.
  (b) Nominal-Matched: ground truth matches the nominal model (evaluates ensemble conservatism).
  (c) Misspecified: ground truth features environmental degradation / unmodeled drift.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as st

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.planning.constraints import (
    ConstraintChecker,
    JointSchedule,
    UAVSchedule,
    UAVSpec,
    Visit,
)
from sar_uav.planning.evaluator import (
    ForwardBackwardFastEvaluator,
    PrefixOnlyEvaluator,
)
from sar_uav.planning.neighborhood import NeighborhoodExplorer
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.planning.baselines import AdaptiveGAPlanner, GreedyLookaheadPlanner
from sar_uav.sim.mission import ScheduleSimulator
from sar_uav.experiments.rng import IndexedRNGStream
from sar_uav.experiments.stats import (
    mcnemar_exact_p,
    student_t_ci,
    wilson_interval,
)

log = logging.getLogger(__name__)


@dataclass
class ProblemInstance:
    """Specification of an independent SAR problem instance."""
    instance_id: int
    name: str
    gt_category: str  # "in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"
    description: str
    seed: int
    num_cells: int
    H: int
    delta_t: float
    depot_idx: int
    coords: List[Tuple[int, int]]
    dist_m: np.ndarray
    uav_specs: List[UAVSpec]
    checker: ConstraintChecker
    b0: np.ndarray
    M: np.ndarray
    nominal_rates: np.ndarray
    veg_groups: np.ndarray
    hset: HypothesisSet
    hset_nom: HypothesisSet
    true_lambda: np.ndarray
    true_M: np.ndarray
    candidate_cells: List[int]


def get_gt_parameters(
    scenario_id: int,
    gt_category: str,
    nominal_rates: np.ndarray,
    base_veg: np.ndarray,
    hset: HypothesisSet,
    coords: List[Tuple[int, int]],
    num_cells: int,
    M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs true sensor rates and true transition matrix for a given GT regime."""
    true_M = np.copy(M)
    cat = gt_category.lower()
    if cat in ("nominal_matched", "nominal"):
        true_lambda = np.copy(nominal_rates)
    elif cat in ("in_ensemble", "ensemble"):
        h_idx = 1 if (scenario_id % 2 == 0) else 2
        true_lambda = np.copy(hset.hypotheses[h_idx].lambda_rates)
    elif cat in ("misspecified_out_of_ensemble", "misspecified", "out_of_ensemble"):
        if scenario_id == 7:
            true_lambda = np.copy(nominal_rates)
            true_lambda[base_veg == 0] *= 0.55
        elif scenario_id == 8:
            true_lambda = np.copy(nominal_rates)
            true_p_stay = 0.40
            true_M = np.eye(num_cells) * true_p_stay
            for i in range(num_cells):
                r, c = coords[i]
                nbrs = [j for j in range(num_cells) if i != j and abs(coords[j][0] - r) + abs(coords[j][1] - c) == 1]
                if nbrs:
                    for n in nbrs:
                        true_M[i, n] = (1.0 - true_p_stay) / len(nbrs)
                true_M[i] /= np.sum(true_M[i])
        elif scenario_id == 9:
            true_lambda = np.copy(hset.hypotheses[1].lambda_rates)
            target_cells = [c for c in [8, 9, 10] if c < num_cells]
            true_lambda[target_cells] *= 0.45
        else:
            # Generic misspecification for scenarios 0-6: 45% reduction in dense vegetation
            true_lambda = np.copy(nominal_rates)
            true_lambda[base_veg == 2] *= 0.55
    else:
        raise ValueError(f"Unknown gt_category/gt_regime: {gt_category}")
    return true_lambda, true_M


def create_instance(
    instance_id: int,
    master_seed: int = 42,
    H: int = 15,
    num_uavs: int = 2,
    delta_t: float = 60.0,
    gt_category: Optional[str] = None,
) -> ProblemInstance:
    """Constructs a distinct problem instance with specified topology and GT regime."""
    num_cells = 16  # 4x4 grid
    depot_idx = 0
    coords = [(r, c) for r in range(4) for c in range(4)]
    dist_m = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) * 400.0

    uav_specs = [
        UAVSpec(name=f"uav_{k}", speed_ms=12.0, battery_joules=300000.0, reserve_joules=35000.0, delta_t=delta_t)
        for k in range(num_uavs)
    ]
    checker = ConstraintChecker(dist_m, depot_idx, uav_specs, H, delta_t=delta_t)
    candidate_cells = list(range(1, num_cells))

    # Base vegetation groups (3 types: savanna, open, dense canopy)
    base_veg = np.array([0, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1])

    # Topology & Prior variations (Decouple scenario_id and replicate_id to support any num_instances >= 1)
    scenario_id = instance_id % 10
    replicate_id = instance_id // 10

    b0 = np.full(num_cells, 1.0 / num_cells)
    p_stay = 0.70
    drift_dir: Optional[Tuple[int, int]] = None
    base_nominal_rate = 0.18
    uncertainty_scale = 0.50

    default_gt_category = "in_ensemble"
    if scenario_id == 0:
        base_name = "TayNguyen_DualHotspot"
        default_gt_category = "in_ensemble"
        description = "Dual hotspot prior at cells 5 and 10 with standard isotropic diffusion."
        b0 = np.full(num_cells, 1.0 / num_cells)
        b0[5] = 0.25
        b0[10] = 0.25
        p_stay = 0.70

    elif scenario_id == 1:
        base_name = "MountainRidge_Trail"
        default_gt_category = "in_ensemble"
        description = "Target concentrated along mountain ridge line (cells 4, 5, 6, 7)."
        b0 = np.full(num_cells, 0.05)
        b0[4:8] = 0.20
        p_stay = 0.65
        drift_dir = (0, 1)  # Eastward drift along ridge

    elif scenario_id == 2:
        base_name = "Valley_DownhillDrift"
        default_gt_category = "in_ensemble"
        description = "Diagonal valley path with downhill South-East drift."
        b0 = np.full(num_cells, 0.05)
        for c_idx in [0, 5, 10, 15]:
            b0[c_idx] = 0.20
        p_stay = 0.60
        drift_dir = (1, 1)  # SE drift

    elif scenario_id == 3:
        base_name = "Bimodal_DispersedSearch"
        default_gt_category = "in_ensemble"
        description = "Separated search zones: NW quadrant (cells 0, 1) and SE quadrant (cells 14, 15)."
        b0 = np.full(num_cells, 0.04)
        b0[0] = 0.22
        b0[1] = 0.14
        b0[14] = 0.14
        b0[15] = 0.22
        p_stay = 0.75

    elif scenario_id == 4:
        base_name = "Concentrated_RavinePeak"
        default_gt_category = "nominal_matched"
        description = "High prior confidence centered at cell 12; true environment strictly matches nominal model."
        b0 = np.full(num_cells, 0.03)
        b0[12] = 0.55
        p_stay = 0.80

    elif scenario_id == 5:
        base_name = "WindCorridor_Eastward"
        default_gt_category = "nominal_matched"
        description = "Strong eastward wind corridor across cells 1, 2, 5, 6; ground truth aligns with nominal."
        b0 = np.full(num_cells, 0.05)
        b0[[1, 2, 5, 6]] = 0.20
        p_stay = 0.65
        drift_dir = (0, 1)

    elif scenario_id == 6:
        base_name = "Central_BroadDiffusion"
        default_gt_category = "nominal_matched"
        description = "Broad diffuse prior across central 8 cells under high uncertainty; nominal matches GT."
        b0 = np.full(num_cells, 0.02)
        b0[[5, 6, 9, 10, 4, 7, 8, 11]] = 0.105
        p_stay = 0.55

    elif scenario_id == 7:
        base_name = "WeatherDegradation_LowlandFog"
        default_gt_category = "misspecified_out_of_ensemble"
        description = "Ground truth experiences 45% sensor degradation in lowlands unmodeled by planners."
        b0 = np.full(num_cells, 1.0 / num_cells)
        b0[3] = 0.20
        b0[6] = 0.20
        b0[9] = 0.20
        b0[12] = 0.20
        p_stay = 0.70

    elif scenario_id == 8:
        base_name = "RiverCorridor_FastCurrent"
        default_gt_category = "misspecified_out_of_ensemble"
        description = "Target carried downstream by unmodeled fast current (p_stay drops to 0.40 in GT)."
        b0 = np.full(num_cells, 0.05)
        b0[[0, 5, 10, 15]] = 0.20
        p_stay = 0.70  # Modeled p_stay is 0.70, but GT true_p_stay will be 0.40

    elif scenario_id == 9:
        base_name = "CanopyOcclusion_Inversion"
        default_gt_category = "misspecified_out_of_ensemble"
        description = "Dense forest in cells 8, 9, 10 cuts true detection rate by half compared to all models."
        b0 = np.full(num_cells, 0.05)
        b0[[2, 8, 13]] = 0.25
        p_stay = 0.68

    else:
        raise ValueError(f"Invalid scenario_id {scenario_id} for instance_id {instance_id}")

    actual_gt_category = gt_category if gt_category is not None else default_gt_category

    if replicate_id == 0:
        name = f"Instance_{instance_id}_{base_name}"
    else:
        name = f"Instance_{instance_id}_Scen{scenario_id}_Rep{replicate_id}_{base_name}"

    b0 = b0 / np.sum(b0)

    # Build Modeled Transition Matrix M
    M = np.eye(num_cells) * p_stay
    for i in range(num_cells):
        r, c = coords[i]
        nbrs = [j for j in range(num_cells) if i != j and abs(coords[j][0] - r) + abs(coords[j][1] - c) == 1]
        if nbrs:
            rem_p = 1.0 - p_stay
            if drift_dir is not None:
                dr, dc = drift_dir
                weights = []
                for n in nbrs:
                    nr, nc = coords[n]
                    dot = (nr - r) * dr + (nc - c) * dc
                    weights.append(max(0.1, 1.0 + 0.8 * dot))
                weights = np.array(weights) / sum(weights)
                for n, w in zip(nbrs, weights):
                    M[i, n] = rem_p * w
            else:
                p_step = rem_p / len(nbrs)
                for n in nbrs:
                    M[i, n] = p_step

    # Normalize M rows
    for i in range(num_cells):
        M[i] /= np.sum(M[i])

    # Sensor rate configurations
    nominal_rates = np.full(num_cells, base_nominal_rate)
    instance_seed = master_seed + instance_id * 1000
    hset = HypothesisSet.create_structured_ensemble(
        nominal_rates, base_veg, uncertainty_scale=uncertainty_scale, num_hypotheses=3, seed=instance_seed
    )
    hset_nom = HypothesisSet.create_nominal(nominal_rates)

    # Build Ground Truth: true_lambda and true_M
    true_lambda, true_M = get_gt_parameters(
        scenario_id=scenario_id,
        gt_category=actual_gt_category,
        nominal_rates=nominal_rates,
        base_veg=base_veg,
        hset=hset,
        coords=coords,
        num_cells=num_cells,
        M=M
    )

    return ProblemInstance(
        instance_id=instance_id,
        name=name,
        gt_category=actual_gt_category,
        description=description,
        seed=instance_seed,
        num_cells=num_cells,
        H=H,
        delta_t=delta_t,
        depot_idx=depot_idx,
        coords=coords,
        dist_m=dist_m,
        uav_specs=uav_specs,
        checker=checker,
        b0=b0,
        M=M,
        nominal_rates=nominal_rates,
        veg_groups=base_veg,
        hset=hset,
        hset_nom=hset_nom,
        true_lambda=true_lambda,
        true_M=true_M,
        candidate_cells=candidate_cells,
    )


def solve_instance_strategies(
    instance: ProblemInstance,
    max_evals_total: int = 600,
    time_limit_sec: float = 15.0,
    order_strategy: str = "fixed",
) -> Tuple[Dict[str, Tuple[JointSchedule, float]], Dict[str, Dict]]:
    """Runs all planner strategies on a single problem instance.

    Tracks explicit initialization, local search, and total evaluations/runtimes.
    """
    checker = instance.checker
    candidate_cells = instance.candidate_cells
    num_uavs = len(instance.uav_specs)
    H = instance.H
    delta_t = instance.delta_t

    engine_ens = JointMassEngine(instance.b0, instance.hset, instance.M, H, delta_t=delta_t)
    engine_nom = JointMassEngine(instance.b0, instance.hset_nom, instance.M, H, delta_t=delta_t)

    # Explorers for 4-arm Ablation
    explorer_full = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=True, enable_dwell_rebalance=True,
        order_strategy=order_strategy, seed=instance.seed
    )
    explorer_no_replace = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=False, enable_dwell_rebalance=True,
        order_strategy=order_strategy, seed=instance.seed
    )
    explorer_no_rebalance = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=True, enable_dwell_rebalance=False,
        order_strategy=order_strategy, seed=instance.seed
    )
    explorer_no_both = NeighborhoodExplorer(
        checker, candidate_cells, allowed_dwells=(1, 2, 3),
        enable_replace=False, enable_dwell_rebalance=False,
        order_strategy=order_strategy, seed=instance.seed
    )

    # Greedy Initializations
    greedy_planner_ens = GreedyLookaheadPlanner(checker, engine_ens, candidate_cells, default_dwell=2)
    t0_ge = time.perf_counter()
    sched_greedy_ens, J_greedy_ens, stats_greedy_ens = greedy_planner_ens.solve(num_uavs)
    rt_greedy_ens = time.perf_counter() - t0_ge
    ev_greedy_ens = stats_greedy_ens.get("n_evals", 0)

    greedy_planner_nom = GreedyLookaheadPlanner(checker, engine_nom, candidate_cells, default_dwell=2)
    t0_gn = time.perf_counter()
    sched_greedy_nom, J_greedy_nom, stats_greedy_nom = greedy_planner_nom.solve(num_uavs)
    rt_greedy_nom = time.perf_counter() - t0_gn
    ev_greedy_nom = stats_greedy_nom.get("n_evals", 0)

    rt_both_greedy = rt_greedy_ens + rt_greedy_nom
    ev_both_greedy = ev_greedy_ens + ev_greedy_nom

    # Multi-start helper
    max_evals_per_start = max_evals_total // 2

    def run_two_start(explorer_obj, evaluator_builder):
        evaluator1 = evaluator_builder()
        solver1 = JointRouteEffortLocalSearch(
            explorer_obj, evaluator1,
            max_evals=max_evals_per_start, time_limit_sec=time_limit_sec / 2.0
        )
        t0_s1 = time.perf_counter()
        sched1, J1, st1 = solver1.solve(sched_greedy_ens)
        rt1 = time.perf_counter() - t0_s1

        evaluator2 = evaluator_builder()
        solver2 = JointRouteEffortLocalSearch(
            explorer_obj, evaluator2,
            max_evals=max_evals_per_start, time_limit_sec=time_limit_sec / 2.0
        )
        t0_s2 = time.perf_counter()
        sched2, J2, st2 = solver2.solve(sched_greedy_nom)
        rt2 = time.perf_counter() - t0_s2

        if J1 >= J2:
            best_sched = sched1
            best_J = J1
            sel = "greedy_ensemble"
        else:
            best_sched = sched2
            best_J = J2
            sel = "greedy_nominal"

        # Merge operator stats
        combined_op_stats = {}
        for st in (st1, st2):
            for op, op_data in st.get("operator_stats", {}).items():
                if op not in combined_op_stats:
                    combined_op_stats[op] = {"evaluated": 0, "accepted": 0, "improvement": 0.0}
                combined_op_stats[op]["evaluated"] += op_data.get("evaluated", 0)
                combined_op_stats[op]["accepted"] += op_data.get("accepted", 0)
                combined_op_stats[op]["improvement"] += op_data.get("improvement", 0.0)

        meta = {
            "local_search_runtime_sec": float(rt1 + rt2),
            "local_search_evals": int(st1["n_evals"] + st2["n_evals"]),
            "accepted_moves": int(st1["accepted_moves"] + st2["accepted_moves"]),
            "stopped_reason": f"start1:{st1.get('stopped_reason', 'unknown')};start2:{st2.get('stopped_reason', 'unknown')}",
            "selected_start": sel,
            "operator_stats": combined_op_stats,
            "starts": [
                {"start_idx": 1, "initial": "greedy_ensemble", "J_init": float(st1.get("initial_J", J_greedy_ens)), "J_final": float(J1), "n_evals": int(st1["n_evals"])},
                {"start_idx": 2, "initial": "greedy_nominal", "J_init": float(st2.get("initial_J", 0.0)), "J_final": float(J2), "n_evals": int(st2["n_evals"])},
            ]
        }
        return best_sched, best_J, meta

    # 1. Proposed_Fast (Full Explorer, Fast Evaluator, 2-start)
    sched_fast, J_fast, meta_fast = run_two_start(explorer_full, lambda: ForwardBackwardFastEvaluator(engine_ens))

    # 2. SingleStart Fixed LS (full local search budget: max_evals_total = 600 evals)
    eval_single_fixed = ForwardBackwardFastEvaluator(engine_ens)
    solver_single_fixed = JointRouteEffortLocalSearch(
        explorer_full, eval_single_fixed,
        max_evals=max_evals_total, time_limit_sec=time_limit_sec
    )
    t0_sf = time.perf_counter()
    sched_s_fixed, J_s_fixed, st_s_fixed = solver_single_fixed.solve(sched_greedy_ens)
    rt_s_fixed = time.perf_counter() - t0_sf

    # 3. SingleStart Matched Total (300 + 300 + ev_greedy_nom evals)
    matched_ls_budget = max_evals_total + ev_greedy_nom
    eval_single_matched = ForwardBackwardFastEvaluator(engine_ens)
    solver_single_matched = JointRouteEffortLocalSearch(
        explorer_full, eval_single_matched,
        max_evals=matched_ls_budget, time_limit_sec=time_limit_sec
    )
    t0_sm = time.perf_counter()
    sched_s_matched, J_s_matched, st_s_matched = solver_single_matched.solve(sched_greedy_ens)
    rt_s_matched = time.perf_counter() - t0_sm

    # 4. Proposed_PrefixOnly
    sched_prefix, J_prefix, meta_prefix = run_two_start(explorer_full, lambda: PrefixOnlyEvaluator(engine_ens))

    # 5. 4-Arm Operator Ablations
    sched_no_repl, J_no_repl, meta_no_repl = run_two_start(explorer_no_replace, lambda: ForwardBackwardFastEvaluator(engine_ens))
    sched_no_rebal, J_no_rebal, meta_no_rebal = run_two_start(explorer_no_rebalance, lambda: ForwardBackwardFastEvaluator(engine_ens))
    sched_no_both, J_no_both, meta_no_both = run_two_start(explorer_no_both, lambda: ForwardBackwardFastEvaluator(engine_ens))

    # 6. Adaptive_GA Baseline
    ga_planner = AdaptiveGAPlanner(checker, engine_ens, candidate_cells, population_size=20, max_generations=25, seed=instance.seed)
    sched_ga, J_ga, stats_ga = ga_planner.solve(num_uavs)

    # 7. Nominal Planning 2-Start Baseline (Pure Model Comparison: matching 2-start budget & solver)
    sched_nom_2s, J_nom_2s, meta_nom_2s = run_two_start(explorer_full, lambda: ForwardBackwardFastEvaluator(engine_nom))

    # 8. Nominal Planning 1-Start Baseline (solved under nominal model)
    eval_nom_solver = ForwardBackwardFastEvaluator(engine_nom)
    nom_solver = JointRouteEffortLocalSearch(
        explorer_full, eval_nom_solver,
        max_evals=max_evals_total, time_limit_sec=time_limit_sec
    )
    t0_nom = time.perf_counter()
    sched_nom, J_nom_self, stats_nom = nom_solver.solve(sched_greedy_nom)
    rt_nom = time.perf_counter() - t0_nom

    strategies: Dict[str, Tuple[JointSchedule, float]] = {
        "Proposed_Fast": (sched_fast, J_fast),
        "Nominal_2Start": (sched_nom_2s, J_nom_2s),
        "SingleStart_Fast_FixedLS": (sched_s_fixed, J_s_fixed),
        "SingleStart_Fast_MatchedTotal": (sched_s_matched, J_s_matched),
        "Proposed_PrefixOnly": (sched_prefix, J_prefix),
        "Ablation_NoReplace": (sched_no_repl, J_no_repl),
        "Ablation_NoRebalance": (sched_no_rebal, J_no_rebal),
        "Ablation_NoReplaceRebalance": (sched_no_both, J_no_both),
        "Greedy_Lookahead": (sched_greedy_ens, J_greedy_ens),
        "Adaptive_GA": (sched_ga, J_ga),
        "Nominal_Planning": (sched_nom, J_nom_self),
    }

    solver_stats: Dict[str, Dict] = {
        "Proposed_Fast": {
            "total_runtime_sec": float(rt_both_greedy + meta_fast["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_fast["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_fast["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_fast["local_search_evals"]),
            "accepted_moves": int(meta_fast["accepted_moves"]),
            "stopped_reason": meta_fast["stopped_reason"],
            "selected_start": meta_fast["selected_start"],
            "operator_stats": meta_fast.get("operator_stats", {})
        },
        "Nominal_2Start": {
            "comparison_mode": "pure_model_comparison_2start",
            "total_runtime_sec": float(rt_both_greedy + meta_nom_2s["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_nom_2s["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_nom_2s["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_nom_2s["local_search_evals"]),
            "accepted_moves": int(meta_nom_2s["accepted_moves"]),
            "stopped_reason": meta_nom_2s["stopped_reason"],
            "selected_start": meta_nom_2s["selected_start"],
            "operator_stats": meta_nom_2s.get("operator_stats", {})
        },
        "SingleStart_Fast_FixedLS": {
            "comparison_mode": "fixed_local_search_budget",
            "max_evals_budget": int(max_evals_total),
            "total_runtime_sec": float(rt_greedy_ens + rt_s_fixed),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": float(rt_s_fixed),
            "total_evals": int(ev_greedy_ens + st_s_fixed["n_evals"]),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": int(st_s_fixed["n_evals"]),
            "accepted_moves": int(st_s_fixed["accepted_moves"]),
            "stopped_reason": st_s_fixed.get("stopped_reason", "unknown"),
            "selected_start": "greedy_ensemble",
            "operator_stats": st_s_fixed.get("operator_stats", {})
        },
        "SingleStart_Fast_MatchedTotal": {
            "comparison_mode": "matched_total_evaluation_budget",
            "max_evals_budget": int(matched_ls_budget),
            "total_runtime_sec": float(rt_greedy_ens + rt_s_matched),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": float(rt_s_matched),
            "total_evals": int(ev_greedy_ens + st_s_matched["n_evals"]),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": int(st_s_matched["n_evals"]),
            "accepted_moves": int(st_s_matched["accepted_moves"]),
            "stopped_reason": st_s_matched.get("stopped_reason", "unknown"),
            "selected_start": "greedy_ensemble",
            "operator_stats": st_s_matched.get("operator_stats", {})
        },
        "Proposed_PrefixOnly": {
            "total_runtime_sec": float(rt_both_greedy + meta_prefix["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_prefix["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_prefix["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_prefix["local_search_evals"]),
            "accepted_moves": int(meta_prefix["accepted_moves"]),
            "stopped_reason": meta_prefix["stopped_reason"],
            "selected_start": meta_prefix["selected_start"],
            "operator_stats": meta_prefix.get("operator_stats", {})
        },
        "Ablation_NoReplace": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_repl["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_repl["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_repl["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_repl["local_search_evals"]),
            "accepted_moves": int(meta_no_repl["accepted_moves"]),
            "stopped_reason": meta_no_repl["stopped_reason"],
            "selected_start": meta_no_repl["selected_start"],
            "operator_stats": meta_no_repl.get("operator_stats", {})
        },
        "Ablation_NoRebalance": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_rebal["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_rebal["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_rebal["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_rebal["local_search_evals"]),
            "accepted_moves": int(meta_no_rebal["accepted_moves"]),
            "stopped_reason": meta_no_rebal["stopped_reason"],
            "selected_start": meta_no_rebal["selected_start"],
            "operator_stats": meta_no_rebal.get("operator_stats", {})
        },
        "Ablation_NoReplaceRebalance": {
            "total_runtime_sec": float(rt_both_greedy + meta_no_both["local_search_runtime_sec"]),
            "greedy_runtime_sec": float(rt_both_greedy),
            "local_search_runtime_sec": float(meta_no_both["local_search_runtime_sec"]),
            "total_evals": int(ev_both_greedy + meta_no_both["local_search_evals"]),
            "greedy_evals": int(ev_both_greedy),
            "local_search_evals": int(meta_no_both["local_search_evals"]),
            "accepted_moves": int(meta_no_both["accepted_moves"]),
            "stopped_reason": meta_no_both["stopped_reason"],
            "selected_start": meta_no_both["selected_start"],
            "operator_stats": meta_no_both.get("operator_stats", {})
        },
        "Greedy_Lookahead": {
            "total_runtime_sec": float(rt_greedy_ens),
            "greedy_runtime_sec": float(rt_greedy_ens),
            "local_search_runtime_sec": 0.0,
            "total_evals": int(ev_greedy_ens),
            "greedy_evals": int(ev_greedy_ens),
            "local_search_evals": 0,
        },
        "Adaptive_GA": {
            "total_runtime_sec": float(stats_ga.get("runtime_sec", 0.0)),
            "greedy_runtime_sec": 0.0,
            "local_search_runtime_sec": float(stats_ga.get("runtime_sec", 0.0)),
            "total_evals": int(stats_ga.get("n_evals", 0)),
            "greedy_evals": 0,
            "local_search_evals": int(stats_ga.get("n_evals", 0)),
        },
        "Nominal_Planning": {
            "total_runtime_sec": float(rt_greedy_nom + rt_nom),
            "greedy_runtime_sec": float(rt_greedy_nom),
            "local_search_runtime_sec": float(rt_nom),
            "total_evals": int(ev_greedy_nom + stats_nom["n_evals"]),
            "greedy_evals": int(ev_greedy_nom),
            "local_search_evals": int(stats_nom["n_evals"]),
            "accepted_moves": int(stats_nom["accepted_moves"]),
            "stopped_reason": stats_nom.get("stopped_reason", "unknown"),
            "selected_start": "nominal_greedy",
            "operator_stats": stats_nom.get("operator_stats", {})
        },
    }

    return strategies, solver_stats


def evaluate_instance_missions(
    instance: ProblemInstance,
    strategies: Dict[str, Tuple[JointSchedule, float]],
    num_missions: int = 100,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[Dict]]:
    """Evaluates all planned strategies over N simulation missions under ground truth.

    Uses Common Random Numbers (CRN) across all strategies within the instance.
    """
    H = instance.H
    delta_t = instance.delta_t
    checker = instance.checker
    num_cells = instance.num_cells

    # Simulator with True Environment Parameters
    sim = ScheduleSimulator(checker, instance.true_lambda, delta_t=delta_t)
    rng_master = IndexedRNGStream(master_seed=instance.seed)

    engine_ens = JointMassEngine(instance.b0, instance.hset, instance.M, H, delta_t=delta_t)
    engine_gt = JointMassEngine(instance.b0, HypothesisSet.create_nominal(instance.true_lambda), instance.true_M, H, delta_t=delta_t)

    metrics_by_strat: Dict[str, List[Dict]] = {s: [] for s in strategies}
    mission_records: List[Dict] = []

    for m_idx in range(num_missions):
        # 1. Target trajectory generation using true_b0 and true_M
        target_path = np.zeros(H, dtype=int)
        init_rng = rng_master.get_rng(m_idx, 0, 0, "target_init")
        target_path[0] = int(init_rng.choice(num_cells, p=instance.b0))
        for t in range(1, H):
            step_rng = rng_master.get_rng(m_idx, t, 0, "target_step")
            target_path[t] = int(step_rng.choice(num_cells, p=instance.true_M[target_path[t - 1]]))

        m_record = {
            "instance_id": instance.instance_id,
            "mission_idx": m_idx,
            "target_path": target_path.tolist(),
            "results": {}
        }

        # 2. Simulate each strategy on the exact same mission (CRN)
        for s_name, (sched, _) in strategies.items():
            res = sim.run_simulation(sched, target_path, mission_idx=m_idx, rng_stream=rng_master)
            rec = {
                "detected": 1 if res.detected else 0,
                "detection_tick": res.detection_tick,
                "rmst": res.rmst_time,
                "energy": res.actual_energy_joules,
            }
            metrics_by_strat[s_name].append(rec)
            m_record["results"][s_name] = rec

        mission_records.append(m_record)

    # 3. Aggregate Instance Summary per strategy
    summary: Dict[str, Dict] = {}
    for s_name, (sched, J_self) in strategies.items():
        records = metrics_by_strat[s_name]
        det_vals = [r["detected"] for r in records]
        dsr = float(np.mean(det_vals))
        ci_dsr = list(wilson_interval(sum(det_vals), len(det_vals)))

        rmst_vals = [r["rmst"] for r in records]
        mean_rmst, se_rmst, ci_rmst = student_t_ci(rmst_vals, alpha=0.05, bounds=(0.0, float(H)))

        energy_vals = [r["energy"] for r in records]
        mean_energy, se_energy, ci_energy = student_t_ci(energy_vals, alpha=0.05)

        fp = sched.generate_footprint_array()
        _, _, J_ens = engine_ens.compute_forward_trajectory(fp)
        _, _, J_gt = engine_gt.compute_forward_trajectory(fp)

        summary[s_name] = {
            "J_self": float(J_self),
            "J_ensemble": float(J_ens),
            "J_true_gt": float(J_gt),
            "DSR": dsr,
            "DSR_ci95_wilson": ci_dsr,
            "mean_RMST_ticks": mean_rmst,
            "RMST_ci95": list(ci_rmst),
            "mean_energy_kJ": mean_energy / 1000.0,
            "energy_ci95_kJ": [e / 1000.0 for e in ci_energy],
            "calib_gap_self_vs_dsr": float(J_self - dsr),
            "calib_gap_gt_vs_dsr": float(J_gt - dsr)
        }

    # 4. Paired Comparisons within this Instance (Proposed_Fast vs Baselines)
    prop_records = metrics_by_strat["Proposed_Fast"]
    prop_det = np.array([r["detected"] for r in prop_records])
    prop_rmst = np.array([r["rmst"] for r in prop_records])
    prop_energy = np.array([r["energy"] for r in prop_records]) / 1000.0

    paired_comparisons: Dict[str, Dict] = {}
    for b_name in strategies:
        if b_name == "Proposed_Fast":
            continue
        base_records = metrics_by_strat[b_name]
        base_det = np.array([r["detected"] for r in base_records])
        base_rmst = np.array([r["rmst"] for r in base_records])
        base_energy = np.array([r["energy"] for r in base_records]) / 1000.0

        diff_dsr = prop_det - base_det
        mean_diff_dsr = float(np.mean(diff_dsr))

        b_disc = int(np.sum((prop_det == 1) & (base_det == 0)))
        c_disc = int(np.sum((prop_det == 0) & (base_det == 1)))
        p_mcnemar = mcnemar_exact_p(b_disc, c_disc)

        diff_rmst = prop_rmst - base_rmst
        mean_diff_rmst, se_diff_rmst, ci_diff_rmst = student_t_ci(
            diff_rmst, alpha=0.05, bounds=(-float(H), float(H))
        )

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

    return summary, paired_comparisons, mission_records


def evaluate_scenario_multi_regimes(
    inst: ProblemInstance,
    strategies: Dict[str, Tuple[JointSchedule, float]],
    num_missions: int = 100,
    gt_regimes: Optional[List[str]] = None,
) -> Dict[str, Tuple[Dict[str, Dict], Dict[str, Dict], List[Dict]]]:
    """Evaluates planned strategies across multiple Ground Truth regimes for the same scenario.

    Returns a dict mapping regime_name -> (summary, paired_comparisons, mission_records).
    """
    if gt_regimes is None:
        gt_regimes = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]

    H = inst.H
    delta_t = inst.delta_t
    checker = inst.checker
    num_cells = inst.num_cells
    scenario_id = inst.instance_id % 10

    regime_results = {}

    for regime in gt_regimes:
        true_lambda, true_M = get_gt_parameters(
            scenario_id=scenario_id,
            gt_category=regime,
            nominal_rates=inst.nominal_rates,
            base_veg=inst.veg_groups,
            hset=inst.hset,
            coords=inst.coords,
            num_cells=num_cells,
            M=inst.M,
        )

REGIME_SEED_OFFSETS: Dict[str, int] = {
    "in_ensemble": 10007,
    "nominal_matched": 20011,
    "misspecified_out_of_ensemble": 30013,
}


def get_regime_seed_offset(regime: str) -> int:
    """Deterministic, process-safe seed offset for ground truth regimes (independent of PYTHONHASHSEED)."""
    if regime in REGIME_SEED_OFFSETS:
        return REGIME_SEED_OFFSETS[regime]
    return int(hashlib.sha256(regime.encode("utf-8")).hexdigest()[:8], 16) % 100000


def evaluate_scenario_multi_regimes(
    inst: ProblemInstance,
    strategies: Dict[str, Tuple[Schedule, float]],
    num_missions: int = 100,
    gt_regimes: Optional[List[str]] = None,
) -> Dict[str, Tuple[Dict[str, Dict], Dict[str, Dict], List[Dict]]]:
    """
    Evaluates pre-computed strategies for a single scenario design across multiple Ground Truth regimes.
    Uses Common Random Numbers (CRN) across strategies within each regime.
    
    Returns a dict mapping regime_name -> (summary, paired_comparisons, mission_records).
    """
    if gt_regimes is None:
        gt_regimes = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]

    H = inst.H
    delta_t = inst.delta_t
    checker = inst.checker
    num_cells = inst.num_cells
    scenario_id = inst.instance_id % 10

    regime_results = {}

    for regime in gt_regimes:
        true_lambda, true_M = get_gt_parameters(
            scenario_id=scenario_id,
            gt_category=regime,
            nominal_rates=inst.nominal_rates,
            base_veg=inst.veg_groups,
            hset=inst.hset,
            coords=inst.coords,
            num_cells=num_cells,
            M=inst.M,
        )

        sim = ScheduleSimulator(checker, true_lambda, delta_t=delta_t)
        regime_offset = get_regime_seed_offset(regime)
        sim_seed = inst.seed + regime_offset
        rng_master = IndexedRNGStream(master_seed=sim_seed)

        engine_ens = JointMassEngine(inst.b0, inst.hset, inst.M, H, delta_t=delta_t)
        engine_gt = JointMassEngine(inst.b0, HypothesisSet.create_nominal(true_lambda), true_M, H, delta_t=delta_t)

        metrics_by_strat: Dict[str, List[Dict]] = {s: [] for s in strategies}
        mission_records: List[Dict] = []

        for m_idx in range(num_missions):
            target_path = np.zeros(H, dtype=int)
            init_rng = rng_master.get_rng(m_idx, 0, 0, "target_init")
            target_path[0] = int(init_rng.choice(num_cells, p=inst.b0))
            for t in range(1, H):
                step_rng = rng_master.get_rng(m_idx, t, 0, "target_step")
                target_path[t] = int(step_rng.choice(num_cells, p=true_M[target_path[t - 1]]))

            m_record = {
                "instance_id": inst.instance_id,
                "scenario_id": scenario_id,
                "replicate_id": inst.instance_id // 10,
                "gt_regime": regime,
                "simulation_seed": sim_seed,
                "mission_idx": m_idx,
                "target_path": target_path.tolist(),
                "results": {}
            }

            for s_name, (sched, _) in strategies.items():
                res = sim.run_simulation(sched, target_path, mission_idx=m_idx, rng_stream=rng_master)
                rec = {
                    "detected": 1 if res.detected else 0,
                    "detection_tick": res.detection_tick,
                    "rmst": res.rmst_time,
                    "energy": res.actual_energy_joules,
                }
                metrics_by_strat[s_name].append(rec)
                m_record["results"][s_name] = rec

            mission_records.append(m_record)

        summary: Dict[str, Dict] = {}
        for s_name, (sched, J_self) in strategies.items():
            records = metrics_by_strat[s_name]
            det_vals = [r["detected"] for r in records]
            dsr = float(np.mean(det_vals))
            ci_dsr = list(wilson_interval(sum(det_vals), len(det_vals)))

            rmst_vals = [r["rmst"] for r in records]
            mean_rmst, se_rmst, ci_rmst = student_t_ci(rmst_vals, alpha=0.05, bounds=(0.0, float(H)))

            energy_vals = [r["energy"] for r in records]
            mean_energy, se_energy, ci_energy = student_t_ci(energy_vals, alpha=0.05)

            fp = sched.generate_footprint_array()
            _, _, J_ens = engine_ens.compute_forward_trajectory(fp)
            _, _, J_gt = engine_gt.compute_forward_trajectory(fp)

            summary[s_name] = {
                "J_self": float(J_self),
                "J_ensemble": float(J_ens),
                "J_true_gt": float(J_gt),
                "DSR": dsr,
                "DSR_ci95_wilson": ci_dsr,
                "mean_RMST_ticks": mean_rmst,
                "RMST_ci95": list(ci_rmst),
                "mean_energy_kJ": mean_energy / 1000.0,
                "energy_ci95_kJ": [e / 1000.0 for e in ci_energy],
                "calib_gap_self_vs_dsr": float(J_self - dsr),
                "calib_gap_gt_vs_dsr": float(J_gt - dsr)
            }

        prop_records = metrics_by_strat["Proposed_Fast"]
        prop_det = np.array([r["detected"] for r in prop_records])
        prop_rmst = np.array([r["rmst"] for r in prop_records])
        prop_energy = np.array([r["energy"] for r in prop_records]) / 1000.0

        paired_comparisons: Dict[str, Dict] = {}
        for b_name in strategies:
            if b_name == "Proposed_Fast":
                continue
            base_records = metrics_by_strat[b_name]
            base_det = np.array([r["detected"] for r in base_records])
            base_rmst = np.array([r["rmst"] for r in base_records])
            base_energy = np.array([r["energy"] for r in base_records]) / 1000.0

            diff_dsr = prop_det - base_det
            mean_diff_dsr = float(np.mean(diff_dsr))

            b_disc = int(np.sum((prop_det == 1) & (base_det == 0)))
            c_disc = int(np.sum((prop_det == 0) & (base_det == 1)))
            p_mcnemar = mcnemar_exact_p(b_disc, c_disc)

            diff_rmst = prop_rmst - base_rmst
            mean_diff_rmst, se_diff_rmst, ci_diff_rmst = student_t_ci(
                diff_rmst, alpha=0.05, bounds=(-float(H), float(H))
            )

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

        regime_results[regime] = (summary, paired_comparisons, mission_records)

    return regime_results


def run_multi_instance_benchmark(
    num_instances: int = 10,
    num_missions_per_instance: int = 100,
    master_seed: int = 42,
    H: int = 15,
    num_uavs: int = 2,
    max_evals_total: int = 600,
    time_limit_sec: float = 15.0,
    order_strategy: str = "fixed",
    gt_regimes: Optional[List[str]] = None,
) -> Dict:
    """Executes the complete multi-instance benchmark suite across 3 decoupled GT regimes."""
    if num_instances < 1:
        raise ValueError(f"num_instances must be >= 1, got {num_instances}")

    if gt_regimes is None:
        gt_regimes = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]

    log.info(f"Starting Multi-Instance Benchmark: {num_instances} problem designs x {len(gt_regimes)} GT regimes x {num_missions_per_instance} missions (order_strategy={order_strategy})")
    t0_all = time.perf_counter()

    import platform
    import sys
    import uuid
    import pandas as pd
    from sar_uav.experiments.experiment_runner import _get_git_info
    from sar_uav.experiments.stats import stratified_bootstrap_ci, stratified_paired_bootstrap_ci

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    git_info = _get_git_info()

    base_instances: List[ProblemInstance] = [
        create_instance(i, master_seed=master_seed, H=H, num_uavs=num_uavs)
        for i in range(num_instances)
    ]

    instance_results: List[Dict] = []
    evaluations_by_scenario: Dict[int, Dict] = {}

    # Flat records list for dataframe-level hierarchical bootstrap
    flat_rows = []

    for inst in base_instances:
        scenario_id = inst.instance_id % 10
        replicate_id = inst.instance_id // 10
        log.info(f"--- Solving Scenario {scenario_id} Rep {replicate_id}: {inst.name} ---")
        t0_inst = time.perf_counter()

        # Step 1: Solve strategies ONCE for this problem design
        strategies, solver_stats = solve_instance_strategies(
            inst, max_evals_total=max_evals_total, time_limit_sec=time_limit_sec,
            order_strategy=order_strategy
        )
        solve_time = time.perf_counter() - t0_inst

        # Step 2: Evaluate schedules under all 3 Ground Truth regimes
        regime_evals = evaluate_scenario_multi_regimes(
            inst, strategies, num_missions=num_missions_per_instance,
            gt_regimes=gt_regimes
        )

        scenario_summary_map = {}

        for regime, (summary, paired, missions) in regime_evals.items():
            eval_name = f"{inst.name}_{regime}"
            inst_record = {
                "instance_id": inst.instance_id,
                "scenario_id": scenario_id,
                "replicate_id": replicate_id,
                "name": eval_name,
                "base_name": inst.name,
                "gt_category": regime,
                "description": inst.description,
                "seed": inst.seed,
                "simulation_seed": inst.seed + get_regime_seed_offset(regime),
                "solve_runtime_sec": solve_time,
                "solver_stats": solver_stats,
                "summary": summary,
                "paired_comparisons": paired,
                "mission_records": missions
            }
            instance_results.append(inst_record)
            scenario_summary_map[regime] = summary

            # Collect for flat dataframe
            for s_name, s_data in summary.items():
                flat_rows.append({
                    "scenario_id": scenario_id,
                    "replicate_id": replicate_id,
                    "gt_regime": regime,
                    "strategy": s_name,
                    "dsr": s_data["DSR"],
                    "rmst": s_data["mean_RMST_ticks"],
                    "energy": s_data["mean_energy_kJ"],
                    "j_gt": s_data["J_true_gt"],
                })

        evaluations_by_scenario[inst.instance_id] = {
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "name": inst.name,
            "solve_runtime_sec": solve_time,
            "solver_stats": solver_stats,
            "regime_summaries": scenario_summary_map
        }

    df_flat = pd.DataFrame(flat_rows)
    strategy_names = list(instance_results[0]["summary"].keys())
    cross_instance_summary: Dict[str, Dict] = {}

    for s_name in strategy_names:
        df_s = df_flat[df_flat["strategy"] == s_name]
        dsr_list = df_s["dsr"].tolist()
        rmst_list = df_s["rmst"].tolist()
        energy_list = df_s["energy"].tolist()
        j_gt_list = df_s["j_gt"].tolist()

        # Scenario-aggregated means (N_scenarios independent problem designs)
        # Equal weighting across unique scenarios: mean of scenario means
        scenario_dsr = df_s.groupby("scenario_id")["dsr"].mean().tolist()
        scenario_rmst = df_s.groupby("scenario_id")["rmst"].mean().tolist()
        scenario_energy = df_s.groupby("scenario_id")["energy"].mean().tolist()
        scenario_j_gt = df_s.groupby("scenario_id")["j_gt"].mean().tolist()

        # Scenario-level parametric Student's t CI (on the scenario means)
        s_mean_dsr, s_se_dsr, s_ci_dsr = student_t_ci(scenario_dsr, alpha=0.05, bounds=(0.0, 1.0))
        s_mean_rmst, s_se_rmst, s_ci_rmst = student_t_ci(scenario_rmst, alpha=0.05, bounds=(0.0, float(H)))
        s_mean_energy, s_se_energy, s_ci_energy = student_t_ci(scenario_energy, alpha=0.05)
        s_mean_j_gt, s_se_j_gt, s_ci_j_gt = student_t_ci(scenario_j_gt, alpha=0.05)

        # Hierarchical / Cluster Bootstrap by Scenario (equal scenario weighting)
        boot_dsr_mean, boot_dsr_se, boot_dsr_ci = stratified_bootstrap_ci(
            df_s, group_col="scenario_id", metric_col="dsr", seed=master_seed, equal_group_weight=True
        )
        boot_rmst_mean, boot_rmst_se, boot_rmst_ci = stratified_bootstrap_ci(
            df_s, group_col="scenario_id", metric_col="rmst", seed=master_seed, equal_group_weight=True
        )
        boot_energy_mean, boot_energy_se, boot_energy_ci = stratified_bootstrap_ci(
            df_s, group_col="scenario_id", metric_col="energy", seed=master_seed, equal_group_weight=True
        )

        cross_instance_summary[s_name] = {
            "mean_DSR": float(np.mean(scenario_dsr)),
            "ci95_DSR": list(boot_dsr_ci),  # Primary scenario-level bootstrap CI
            "ci95_DSR_stratified_bootstrap": list(boot_dsr_ci),
            "ci95_DSR_scenario_level_t": list(s_ci_dsr),
            "se_DSR_stratified_bootstrap": boot_dsr_se,
            "mean_RMST_ticks": float(np.mean(scenario_rmst)),
            "ci95_RMST": list(boot_rmst_ci),  # Primary scenario-level bootstrap CI
            "ci95_RMST_stratified_bootstrap": list(boot_rmst_ci),
            "ci95_RMST_scenario_level_t": list(s_ci_rmst),
            "mean_energy_kJ": float(np.mean(scenario_energy)),
            "ci95_energy_kJ": list(s_ci_energy),
            "ci95_energy_kJ_bootstrap": list(boot_energy_ci),
            "mean_J_true_gt": float(np.mean(scenario_j_gt)),
            "ci95_J_true_gt_scenario_level_t": list(s_ci_j_gt),
            "per_instance_DSR": dsr_list,
            "per_instance_RMST": rmst_list,
            "per_scenario_mean_DSR": scenario_dsr,
            "per_scenario_mean_RMST": scenario_rmst,
        }

    # Cross-Instance Paired Comparison (Proposed_Fast vs Each Baseline)
    cross_instance_paired: Dict[str, Dict] = {}
    df_prop = df_flat[df_flat["strategy"] == "Proposed_Fast"].copy()

    for b_name in strategy_names:
        if b_name == "Proposed_Fast":
            continue
        df_base = df_flat[df_flat["strategy"] == b_name].copy()
        merged = pd.merge(
            df_prop, df_base,
            on=["scenario_id", "replicate_id", "gt_regime"],
            suffixes=("_prop", "_base")
        )
        merged["delta_dsr"] = merged["dsr_prop"] - merged["dsr_base"]
        merged["delta_rmst"] = merged["rmst_prop"] - merged["rmst_base"]
        merged["delta_energy"] = merged["energy_prop"] - merged["energy_base"]

        delta_dsr_list = [
            ir["paired_comparisons"][f"Proposed_vs_{b_name}"]["mean_delta_DSR"]
            for ir in instance_results
        ]
        delta_rmst_list = [
            ir["paired_comparisons"][f"Proposed_vs_{b_name}"]["mean_delta_RMST_ticks"]
            for ir in instance_results
        ]
        delta_energy_list = [
            ir["paired_comparisons"][f"Proposed_vs_{b_name}"]["mean_delta_energy_kJ"]
            for ir in instance_results
        ]

        scenario_delta_dsr = merged.groupby("scenario_id")["delta_dsr"].mean().tolist()
        scenario_delta_rmst = merged.groupby("scenario_id")["delta_rmst"].mean().tolist()
        scenario_delta_energy = merged.groupby("scenario_id")["delta_energy"].mean().tolist()

        wins = 0
        losses = 0
        ties = 0
        for d_dsr, d_rmst in zip(delta_dsr_list, delta_rmst_list):
            if d_dsr > 0.001:
                wins += 1
            elif d_dsr < -0.001:
                losses += 1
            else:
                if d_rmst < -0.05:
                    wins += 1
                elif d_rmst > 0.05:
                    losses += 1
                else:
                    ties += 1

        s_m_d_dsr, s_se_d_dsr, s_ci_d_dsr = student_t_ci(scenario_delta_dsr, alpha=0.05, bounds=(-1.0, 1.0))
        s_m_d_rmst, s_se_d_rmst, s_ci_d_rmst = student_t_ci(scenario_delta_rmst, alpha=0.05, bounds=(-float(H), float(H)))
        s_m_d_energy, s_se_d_energy, s_ci_d_energy = student_t_ci(scenario_delta_energy, alpha=0.05)

        # Cluster paired bootstrap (equal scenario weighting)
        boot_diff_dsr, boot_se_d_dsr, boot_ci_d_dsr, p_boot = stratified_paired_bootstrap_ci(
            merged, group_col="scenario_id", col_a="dsr_prop", col_b="dsr_base", seed=master_seed, equal_group_weight=True
        )
        boot_diff_rmst, boot_se_d_rmst, boot_ci_d_rmst, _ = stratified_paired_bootstrap_ci(
            merged, group_col="scenario_id", col_a="rmst_prop", col_b="rmst_base", seed=master_seed, equal_group_weight=True
        )

        nonzero_diffs = [d for d in scenario_delta_dsr if abs(d) > 1e-6]
        if len(nonzero_diffs) >= 5:
            try:
                import scipy.stats as st
                w_stat, w_p = st.wilcoxon(scenario_delta_dsr)
                wilcoxon_p = float(w_p)
            except Exception:
                wilcoxon_p = float("nan")
        else:
            wilcoxon_p = float("nan")

        cross_instance_paired[f"Proposed_vs_{b_name}"] = {
            "instance_record": {"wins": wins, "losses": losses, "ties": ties},
            "mean_delta_DSR": float(np.mean(scenario_delta_dsr)),
            "ci95_delta_DSR": list(boot_ci_d_dsr),  # Primary scenario-level bootstrap CI
            "ci95_delta_DSR_stratified_bootstrap": list(boot_ci_d_dsr),
            "ci95_delta_DSR_scenario_level_t": list(s_ci_d_dsr),
            "p_boot_stratified": p_boot,
            "mean_delta_RMST_ticks": float(np.mean(scenario_delta_rmst)),
            "ci95_delta_RMST": list(s_ci_d_rmst),
            "ci95_delta_RMST_bootstrap": list(boot_ci_d_rmst),
            "mean_delta_energy_kJ": float(np.mean(scenario_delta_energy)),
            "ci95_delta_energy_kJ": list(s_ci_d_energy),
            "wilcoxon_p_value": wilcoxon_p,
            "per_instance_delta_DSR": delta_dsr_list,
            "per_instance_delta_RMST": delta_rmst_list,
            "per_scenario_delta_DSR": scenario_delta_dsr,
            "per_scenario_delta_RMST": scenario_delta_rmst
        }

    # Regime Stratification: Complete factor across all scenarios!
    categories = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]
    regime_breakdown: Dict[str, Dict] = {}
    for cat in categories:
        cat_insts = [ir for ir in instance_results if ir["gt_category"] == cat]
        df_regime = df_flat[df_flat["gt_regime"] == cat]
        n_cat = len(cat_insts)
        if n_cat > 0:
            reg_scenarios = sorted(df_regime["scenario_id"].unique())
            num_reg_scenarios = len(reg_scenarios)

            # Per-strategy scenario means in this regime
            df_reg_prop = df_regime[df_regime["strategy"] == "Proposed_Fast"]
            df_reg_nom2 = df_regime[df_regime["strategy"] == "Nominal_2Start"]
            df_reg_nom1 = df_regime[df_regime["strategy"] == "Nominal_Planning"]

            scen_prop_dsr = df_reg_prop.groupby("scenario_id")["dsr"].mean().tolist()
            scen_prop_rmst = df_reg_prop.groupby("scenario_id")["rmst"].mean().tolist()
            _, _, ci_prop_dsr = student_t_ci(scen_prop_dsr, bounds=(0.0, 1.0))
            _, _, ci_prop_rmst = student_t_ci(scen_prop_rmst, bounds=(0.0, float(H)))

            scen_nom2_dsr = df_reg_nom2.groupby("scenario_id")["dsr"].mean().tolist() if not df_reg_nom2.empty else []
            scen_nom2_rmst = df_reg_nom2.groupby("scenario_id")["rmst"].mean().tolist() if not df_reg_nom2.empty else []
            _, _, ci_nom2_dsr = student_t_ci(scen_nom2_dsr, bounds=(0.0, 1.0)) if scen_nom2_dsr else (0, 0, (float("nan"), float("nan")))
            _, _, ci_nom2_rmst = student_t_ci(scen_nom2_rmst, bounds=(0.0, float(H))) if scen_nom2_rmst else (0, 0, (float("nan"), float("nan")))

            scen_nom1_dsr = df_reg_nom1.groupby("scenario_id")["dsr"].mean().tolist() if not df_reg_nom1.empty else []
            scen_nom1_rmst = df_reg_nom1.groupby("scenario_id")["rmst"].mean().tolist() if not df_reg_nom1.empty else []
            _, _, ci_nom1_dsr = student_t_ci(scen_nom1_dsr, bounds=(0.0, 1.0)) if scen_nom1_dsr else (0, 0, (float("nan"), float("nan")))
            _, _, ci_nom1_rmst = student_t_ci(scen_nom1_rmst, bounds=(0.0, float(H))) if scen_nom1_rmst else (0, 0, (float("nan"), float("nan")))

            # Paired differences within regime
            merged_nom2 = pd.merge(df_reg_prop, df_reg_nom2, on=["scenario_id", "replicate_id"], suffixes=("_prop", "_base"))
            merged_nom2["delta_dsr"] = merged_nom2["dsr_prop"] - merged_nom2["dsr_base"]
            merged_nom2["delta_rmst"] = merged_nom2["rmst_prop"] - merged_nom2["rmst_base"]
            scen_delta_nom2_dsr = merged_nom2.groupby("scenario_id")["delta_dsr"].mean().tolist()
            scen_delta_nom2_rmst = merged_nom2.groupby("scenario_id")["delta_rmst"].mean().tolist()
            _, _, ci_d_nom2_dsr = student_t_ci(scen_delta_nom2_dsr, bounds=(-1.0, 1.0))
            _, _, ci_d_nom2_rmst = student_t_ci(scen_delta_nom2_rmst, bounds=(-float(H), float(H)))
            _, _, boot_ci_d_nom2_dsr, p_boot_nom2 = stratified_paired_bootstrap_ci(
                merged_nom2, group_col="scenario_id", col_a="dsr_prop", col_b="dsr_base", seed=master_seed, equal_group_weight=True
            )

            merged_nom1 = pd.merge(df_reg_prop, df_reg_nom1, on=["scenario_id", "replicate_id"], suffixes=("_prop", "_base"))
            merged_nom1["delta_dsr"] = merged_nom1["dsr_prop"] - merged_nom1["dsr_base"]
            merged_nom1["delta_rmst"] = merged_nom1["rmst_prop"] - merged_nom1["rmst_base"]
            scen_delta_nom1_dsr = merged_nom1.groupby("scenario_id")["delta_dsr"].mean().tolist()
            scen_delta_nom1_rmst = merged_nom1.groupby("scenario_id")["delta_rmst"].mean().tolist()
            _, _, ci_d_nom1_dsr = student_t_ci(scen_delta_nom1_dsr, bounds=(-1.0, 1.0))
            _, _, ci_d_nom1_rmst = student_t_ci(scen_delta_nom1_rmst, bounds=(-float(H), float(H)))
            _, _, boot_ci_d_nom1_dsr, p_boot_nom1 = stratified_paired_bootstrap_ci(
                merged_nom1, group_col="scenario_id", col_a="dsr_prop", col_b="dsr_base", seed=master_seed, equal_group_weight=True
            )

            regime_breakdown[cat] = {
                "num_instances": n_cat,
                "num_scenarios": num_reg_scenarios,
                "instances": [ci["name"] for ci in cat_insts],
                "Proposed_Fast": {
                    "mean_DSR": float(np.mean(scen_prop_dsr)),
                    "ci95_DSR": list(ci_prop_dsr),
                    "mean_RMST": float(np.mean(scen_prop_rmst)),
                    "ci95_RMST": list(ci_prop_rmst),
                },
                "Nominal_2Start": {
                    "mean_DSR": float(np.mean(scen_nom2_dsr)),
                    "ci95_DSR": list(ci_nom2_dsr),
                    "mean_RMST": float(np.mean(scen_nom2_rmst)),
                    "ci95_RMST": list(ci_nom2_rmst),
                },
                "Nominal_Planning": {
                    "mean_DSR": float(np.mean(scen_nom1_dsr)),
                    "ci95_DSR": list(ci_nom1_dsr),
                    "mean_RMST": float(np.mean(scen_nom1_rmst)),
                    "ci95_RMST": list(ci_nom1_rmst),
                },
                "delta_Proposed_vs_Nominal_2Start": {
                    "mean_delta_DSR": float(np.mean(scen_delta_nom2_dsr)),
                    "ci95_delta_DSR_student_t": list(ci_d_nom2_dsr),
                    "ci95_delta_DSR_bootstrap": list(boot_ci_d_nom2_dsr),
                    "p_boot": p_boot_nom2,
                    "mean_delta_RMST": float(np.mean(scen_delta_nom2_rmst)),
                    "ci95_delta_RMST_student_t": list(ci_d_nom2_rmst),
                },
                "delta_Proposed_vs_Nominal": {
                    "mean_delta_DSR": float(np.mean(scen_delta_nom1_dsr)),
                    "ci95_delta_DSR_student_t": list(ci_d_nom1_dsr),
                    "ci95_delta_DSR_bootstrap": list(boot_ci_d_nom1_dsr),
                    "p_boot": p_boot_nom1,
                    "mean_delta_RMST": float(np.mean(scen_delta_nom1_rmst)),
                    "ci95_delta_RMST_student_t": list(ci_d_nom1_rmst),
                }
            }
        else:
            regime_breakdown[cat] = {
                "num_instances": 0,
                "num_scenarios": 0,
                "instances": [],
                "Proposed_Fast": {"mean_DSR": float("nan"), "ci95_DSR": [float("nan"), float("nan")], "mean_RMST": float("nan"), "ci95_RMST": [float("nan"), float("nan")]},
                "Nominal_2Start": {"mean_DSR": float("nan"), "ci95_DSR": [float("nan"), float("nan")], "mean_RMST": float("nan"), "ci95_RMST": [float("nan"), float("nan")]},
                "Nominal_Planning": {"mean_DSR": float("nan"), "ci95_DSR": [float("nan"), float("nan")], "mean_RMST": float("nan"), "ci95_RMST": [float("nan"), float("nan")]},
                "delta_Proposed_vs_Nominal_2Start": {"mean_delta_DSR": float("nan"), "ci95_delta_DSR_student_t": [float("nan"), float("nan")], "ci95_delta_DSR_bootstrap": [float("nan"), float("nan")], "p_boot": float("nan"), "mean_delta_RMST": float("nan"), "ci95_delta_RMST_student_t": [float("nan"), float("nan")]},
                "delta_Proposed_vs_Nominal": {"mean_delta_DSR": float("nan"), "ci95_delta_DSR_student_t": [float("nan"), float("nan")], "ci95_delta_DSR_bootstrap": [float("nan"), float("nan")], "p_boot": float("nan"), "mean_delta_RMST": float("nan"), "ci95_delta_RMST_student_t": [float("nan"), float("nan")]}
            }

    # Operator Ablation Aggregation strictly across unique scenario solutions (not multiplied by GT regimes)
    ablation_arms = ["Proposed_Fast", "Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"]
    ablation_summary: Dict[str, Dict] = {}
    scenario_records = list(evaluations_by_scenario.values())

    prop_scenario_j = [rec["regime_summaries"]["in_ensemble"]["Proposed_Fast"]["J_ensemble"] for rec in scenario_records]

    for arm in ablation_arms:
        # Search & optimization metrics: aggregated across unique scenario solves
        arm_j = [rec["regime_summaries"]["in_ensemble"][arm]["J_ensemble"] for rec in scenario_records]
        arm_rt = [rec["solver_stats"][arm]["total_runtime_sec"] for rec in scenario_records]
        arm_evals = [rec["solver_stats"][arm]["total_evals"] for rec in scenario_records]
        arm_moves = [rec["solver_stats"][arm]["accepted_moves"] for rec in scenario_records]

        _, _, ci_arm_j = student_t_ci(arm_j)

        # Delta J vs Proposed_Fast (paired per scenario)
        delta_j = [p_j - a_j for p_j, a_j in zip(prop_scenario_j, arm_j)]
        _, _, ci_delta_j = student_t_ci(delta_j)

        # Mission outcome metrics: aggregated across scenarios
        df_arm = df_flat[df_flat["strategy"] == arm]
        scen_arm_dsr = df_arm.groupby("scenario_id")["dsr"].mean().tolist()
        scen_arm_rmst = df_arm.groupby("scenario_id")["rmst"].mean().tolist()

        _, _, ci_arm_dsr = student_t_ci(scen_arm_dsr, bounds=(0.0, 1.0))
        _, _, ci_arm_rmst = student_t_ci(scen_arm_rmst, bounds=(0.0, float(H)))

        # Paired DSR difference vs Proposed_Fast
        merged_ablation = pd.merge(df_prop, df_arm, on=["scenario_id", "replicate_id", "gt_regime"], suffixes=("_prop", "_arm"))
        merged_ablation["delta_dsr"] = merged_ablation["dsr_prop"] - merged_ablation["dsr_arm"]
        scen_delta_abl_dsr = merged_ablation.groupby("scenario_id")["delta_dsr"].mean().tolist()
        _, _, ci_d_abl_dsr = student_t_ci(scen_delta_abl_dsr, bounds=(-1.0, 1.0))
        _, _, boot_ci_d_abl_dsr, p_boot_abl = stratified_paired_bootstrap_ci(
            merged_ablation, group_col="scenario_id", col_a="dsr_prop", col_b="dsr_arm", seed=master_seed, equal_group_weight=True
        )

        ALL_OPERATORS = [
            "Replace", "DwellRebalance", "Swap", "ChangeDwell",
            "AdjustWait", "Insert", "Delete", "Relocate"
        ]
        op_counts: Dict[str, Dict] = {
            op: {"evaluated": 0, "accepted": 0, "total_improvement": 0.0}
            for op in ALL_OPERATORS
        }
        for rec in scenario_records:
            op_stats = rec["solver_stats"][arm].get("operator_stats", {})
            for op_k, op_v in op_stats.items():
                if op_k not in op_counts:
                    op_counts[op_k] = {"evaluated": 0, "accepted": 0, "total_improvement": 0.0}
                op_counts[op_k]["evaluated"] += op_v.get("evaluated", 0)
                op_counts[op_k]["accepted"] += op_v.get("accepted", 0)
                op_counts[op_k]["total_improvement"] += op_v.get("improvement", 0.0)

        ablation_summary[arm] = {
            "mean_J_ensemble": float(np.mean(arm_j)),
            "ci95_J_ensemble": list(ci_arm_j),
            "mean_delta_J_vs_proposed": float(np.mean(delta_j)),
            "ci95_delta_J_vs_proposed": list(ci_delta_j),
            "mean_DSR": float(np.mean(scen_arm_dsr)),
            "ci95_DSR": list(ci_arm_dsr),
            "mean_delta_DSR_vs_proposed": float(np.mean(scen_delta_abl_dsr)),
            "ci95_delta_DSR_student_t": list(ci_d_abl_dsr),
            "ci95_delta_DSR_bootstrap": list(boot_ci_d_abl_dsr),
            "p_boot_delta_DSR": p_boot_abl,
            "mean_RMST_ticks": float(np.mean(scen_arm_rmst)),
            "ci95_RMST": list(ci_arm_rmst),
            "mean_runtime_sec": float(np.mean(arm_rt)),
            "mean_total_evals": float(np.mean(arm_evals)),
            "mean_accepted_moves": float(np.mean(arm_moves)),
            "aggregated_operator_stats": op_counts
        }

    total_time = time.perf_counter() - t0_all
    log.info(f"Multi-Instance Benchmark completed in {total_time:.2f}s")

    simulation_seeds = {
        f"scenario_{inst.instance_id % 10}_rep_{inst.instance_id // 10}_{regime}": inst.seed + get_regime_seed_offset(regime)
        for inst in base_instances
        for regime in gt_regimes
    }

    unique_scenarios = sorted(list(set(inst.instance_id % 10 for inst in base_instances)))
    replicates_map = {
        int(s): sum(1 for inst in base_instances if (inst.instance_id % 10) == s)
        for s in unique_scenarios
    }

    metadata_dict = {
        "tier": 3,
        "subtier": "multi_instance",
        "run_id": run_id,
        "seed": master_seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": round(total_time, 4),
        "ci_methods": {
            "primary_ci": "stratified_scenario_cluster_bootstrap_95",
            "parametric_scenario_ci": "scenario_aggregated_student_t_95",
            "sampling_level": "scenario_id",
            "aggregation_weighting": "equal_scenario_weight"
        },
        "simulation_seeds": simulation_seeds,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **git_info
        },
        "experiment_config": {
            "num_instances": num_instances,
            "num_unique_scenarios": len(unique_scenarios),
            "num_designs": num_instances,
            "replicates_per_scenario": replicates_map,
            "aggregation_weighting": "equal_scenario_weight",
            "gt_regimes": gt_regimes,
            "num_evaluations_total": len(instance_results),
            "num_missions_per_instance": num_missions_per_instance,
            "total_missions_evaluated": len(instance_results) * num_missions_per_instance,
            "master_seed": master_seed,
            "H": H,
            "num_uavs": num_uavs,
            "max_evals_total": max_evals_total,
            "max_evals_per_start": max_evals_total // 2,
            "time_limit_sec": time_limit_sec,
            "order_strategy": order_strategy,
        }
    }

    return {
        "metadata": metadata_dict,
        "benchmark_metadata": {
            **metadata_dict,
            "num_instances": num_instances,
            "num_unique_scenarios": len(unique_scenarios),
            "num_designs": num_instances,
            "replicates_per_scenario": replicates_map,
            "aggregation_weighting": "equal_scenario_weight",
            "num_missions_per_instance": num_missions_per_instance,
            "total_missions_evaluated": len(instance_results) * num_missions_per_instance,
            "master_seed": master_seed,
            "H": H,
            "num_uavs": num_uavs,
            "max_evals_total": max_evals_total,
            "time_limit_sec": time_limit_sec,
            "total_benchmark_runtime_sec": total_time,
            "order_strategy": order_strategy,
        },
        "cross_instance_summary": cross_instance_summary,
        "cross_instance_paired": cross_instance_paired,
        "regime_breakdown": regime_breakdown,
        "ablation_summary": ablation_summary,
        "evaluations_by_scenario": evaluations_by_scenario,
        "instances": instance_results
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Multi-Instance Comparative Benchmark")
    parser.add_argument("--num-instances", type=int, default=10, help="Number of distinct problem instances")
    parser.add_argument("--num-missions", type=int, default=100, help="Missions per instance")
    parser.add_argument("--master-seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--operator-order", type=str, default="fixed", choices=["fixed", "round_robin", "shuffled"], help="Operator ordering strategy")
    parser.add_argument("--out", type=str, default="results/multi_instance_tier3_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = run_multi_instance_benchmark(
        num_instances=args.num_instances,
        num_missions_per_instance=args.num_missions,
        master_seed=args.master_seed,
        order_strategy=args.operator_order
    )

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Archive run file to results/runs/{run_id}_multi_instance.json
    runs_dir = out_file.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_archival_file = runs_dir / f"{results['metadata']['run_id']}_multi_instance.json"
    with open(run_archival_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark finished and saved to {args.out}")
    print("\nCross-Instance Paired Comparison (vs Proposed_Fast):")
    for k, v in results["cross_instance_paired"].items():
        rec = v["instance_record"]
        print(f"  {k}: W/L/T={rec['wins']}/{rec['losses']}/{rec['ties']}, Mean dDSR={v['mean_delta_DSR']:+.4f}, Mean dRMST={v['mean_delta_RMST_ticks']:+.3f}")
