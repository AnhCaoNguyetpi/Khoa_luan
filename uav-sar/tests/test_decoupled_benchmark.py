from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

from sar_uav.experiments.stats import (
    stratified_bootstrap_ci,
    stratified_paired_bootstrap_ci,
    student_t_ci
)
from sar_uav.experiments.tier3_multi_instance import (
    create_instance,
    get_gt_parameters,
    solve_instance_strategies,
    evaluate_instance_missions,
    evaluate_scenario_multi_regimes,
    run_multi_instance_benchmark,
)
from sar_uav.planning.neighborhood import NeighborhoodExplorer


def test_gt_category_planner_independence():
    """Verify that changing gt_category does NOT modify the planner's perceived environment."""
    inst_in_ens = create_instance(0, master_seed=42, H=10, num_uavs=2, gt_category="in_ensemble")
    inst_nom = create_instance(0, master_seed=42, H=10, num_uavs=2, gt_category="nominal_matched")
    inst_misspec = create_instance(0, master_seed=42, H=10, num_uavs=2, gt_category="misspecified")

    # Planner inputs must be bitwise identical across all 3 GT regimes
    np.testing.assert_array_equal(inst_in_ens.b0, inst_nom.b0)
    np.testing.assert_array_equal(inst_in_ens.b0, inst_misspec.b0)
    np.testing.assert_array_equal(inst_in_ens.M, inst_nom.M)
    np.testing.assert_array_equal(inst_in_ens.M, inst_misspec.M)
    np.testing.assert_array_equal(inst_in_ens.nominal_rates, inst_nom.nominal_rates)
    np.testing.assert_array_equal(inst_in_ens.nominal_rates, inst_misspec.nominal_rates)
    assert inst_in_ens.seed == inst_nom.seed == inst_misspec.seed

    # Ground truth parameters MUST reflect the respective regimes
    np.testing.assert_array_equal(inst_nom.true_lambda, inst_nom.nominal_rates)


def test_nominal_2start_baseline_budget_parity():
    """Verify that Nominal_2Start has matched 2-start structure with Proposed_Fast."""
    inst = create_instance(1, master_seed=100, H=8, num_uavs=2)
    strategies, solver_stats = solve_instance_strategies(
        inst, max_evals_total=40, time_limit_sec=5.0
    )

    assert "Nominal_2Start" in strategies
    assert "Proposed_Fast" in strategies
    assert "Nominal_Planning" in strategies

    stats_prop = solver_stats["Proposed_Fast"]
    stats_nom_2s = solver_stats["Nominal_2Start"]

    # Both must report 2-start initialization and local search evals
    assert "greedy_evals" in stats_nom_2s
    assert "local_search_evals" in stats_nom_2s
    assert stats_nom_2s["greedy_evals"] == stats_prop["greedy_evals"]
    assert stats_nom_2s["comparison_mode"] == "pure_model_comparison_2start"


def test_operator_ordering_direct_sequence():
    """Directly verifies ordering sequence, shift on iteration, seed determinism, reset, and disabled operators."""
    inst = create_instance(0, master_seed=42, H=8, num_uavs=2)
    strat, _ = solve_instance_strategies(inst, max_evals_total=10, time_limit_sec=2.0)
    sched = strat["Greedy_Lookahead"][0]

    # 1. Fixed mode: operator sequence is deterministic and unchanging across steps
    exp_fixed = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        order_strategy="fixed", seed=42
    )
    order_iter0 = exp_fixed.get_operator_order()
    exp_fixed.step_iteration()
    order_iter1 = exp_fixed.get_operator_order()
    assert order_iter0 == order_iter1
    assert "Replace" in order_iter0
    assert order_iter0[0] == "Replace"

    # 2. Round-Robin mode: order rotates cyclically across iterations
    exp_rr = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        order_strategy="round_robin", seed=42
    )
    exp_rr.reset_iteration()
    order_rr0 = exp_rr.get_operator_order()
    exp_rr.step_iteration()
    order_rr1 = exp_rr.get_operator_order()
    exp_rr.step_iteration()
    order_rr2 = exp_rr.get_operator_order()

    assert order_rr0[0] != order_rr1[0]
    assert order_rr1[0] != order_rr2[0]
    assert order_rr1 == order_rr0[1:] + order_rr0[:1]

    # 3. Shuffled mode: deterministic with same seed, resets properly
    exp_shuf1 = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        order_strategy="shuffled", seed=123
    )
    order_s1 = exp_shuf1.get_operator_order()

    exp_shuf2 = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        order_strategy="shuffled", seed=123
    )
    order_s2 = exp_shuf2.get_operator_order()
    assert order_s1 == order_s2

    # Reset restores original stream
    exp_shuf1.step_iteration()
    order_s1_step = exp_shuf1.get_operator_order()
    assert order_s1_step != order_s1
    exp_shuf1.reset_iteration()
    order_s1_reset = exp_shuf1.get_operator_order()
    assert order_s1_reset == order_s1

    # 4. Disabled operators: disabled operators never appear in order or in candidate generation
    exp_no_replace = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        enable_replace=False, enable_dwell_rebalance=True,
        order_strategy="fixed"
    )
    assert "Replace" not in exp_no_replace.get_operator_order()
    assert "DwellRebalance" in exp_no_replace.get_operator_order()
    ops_no_repl = [op for op, _, _, _ in exp_no_replace.generate_all_neighbors(sched)]
    assert "Replace" not in ops_no_repl

    exp_no_both = NeighborhoodExplorer(
        inst.checker, inst.candidate_cells, allowed_dwells=(1, 2),
        enable_replace=False, enable_dwell_rebalance=False,
        order_strategy="fixed"
    )
    assert "Replace" not in exp_no_both.get_operator_order()
    assert "DwellRebalance" not in exp_no_both.get_operator_order()
    ops_no_both = [op for op, _, _, _ in exp_no_both.generate_all_neighbors(sched)]
    assert "Replace" not in ops_no_both
    assert "DwellRebalance" not in ops_no_both


def test_scenario_multi_regime_evaluation():
    """Verify that a single scenario design produces 3 evaluation regimes with shared planned schedules."""
    inst = create_instance(0, master_seed=42, H=8, num_uavs=2)
    strategies, solver_stats = solve_instance_strategies(
        inst, max_evals_total=30, time_limit_sec=3.0
    )

    regimes_result = evaluate_scenario_multi_regimes(
        inst, strategies, num_missions=10,
        gt_regimes=["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]
    )

    assert "in_ensemble" in regimes_result
    assert "nominal_matched" in regimes_result
    assert "misspecified_out_of_ensemble" in regimes_result

    # Check each regime has summary and paired comparisons
    for regime, (summary, paired, missions) in regimes_result.items():
        assert "Proposed_Fast" in summary
        assert "Nominal_2Start" in summary
        assert "Nominal_Planning" in summary
        assert "Proposed_vs_Nominal_2Start" in paired
        assert len(missions) == 10


def test_stratified_bootstrap_ci_single_group_nan():
    """Verify that 1 single group returns NaN CI instead of a misleading 0-width interval."""
    df_single = pd.DataFrame({
        "scenario_id": [0, 0, 0],
        "dsr": [0.2, 0.3, 0.25]
    })

    mean_est, se, (ci_lo, ci_hi) = stratified_bootstrap_ci(
        df_single, group_col="scenario_id", metric_col="dsr", alpha=0.05, n_boot=200, seed=42
    )

    assert np.isclose(mean_est, df_single["dsr"].mean(), atol=1e-4)
    assert np.isnan(se)
    assert np.isnan(ci_lo) and np.isnan(ci_hi)


def test_stratified_paired_bootstrap_multi_group():
    """Verify cluster/stratified paired bootstrap on multi-group dataset."""
    df_multi = pd.DataFrame({
        "scenario_id": [0, 0, 1, 1, 2, 2],
        "dsr_prop": [0.30, 0.32, 0.40, 0.42, 0.50, 0.52],
        "dsr_nom": [0.25, 0.28, 0.35, 0.36, 0.42, 0.45],
    })

    mean_diff, se, (ci_lo, ci_hi), p_boot = stratified_paired_bootstrap_ci(
        df_multi, group_col="scenario_id", col_a="dsr_prop", col_b="dsr_nom", alpha=0.05, n_boot=500, seed=42
    )

    expected_diff = float((df_multi["dsr_prop"] - df_multi["dsr_nom"]).mean())
    assert np.isclose(mean_diff, expected_diff, atol=1e-4)
    assert ci_lo <= mean_diff <= ci_hi
    assert se > 0.0
    assert 0.0 <= p_boot <= 1.0


def test_cross_process_reproducibility():
    """Verify that execution across separate processes with different PYTHONHASHSEED is bitwise deterministic."""
    import subprocess
    import sys
    import json

    code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src") if "__file__" in locals() else "src")
import json
from sar_uav.experiments.tier3_multi_instance import create_instance, solve_instance_strategies, evaluate_scenario_multi_regimes

inst = create_instance(0, master_seed=42, H=6, num_uavs=2)
strategies, _ = solve_instance_strategies(inst, max_evals_total=10, time_limit_sec=2.0)
results = evaluate_scenario_multi_regimes(inst, strategies, num_missions=5)

out = {}
for reg, (summ, paired, missions) in results.items():
    out[reg] = {
        "DSR_prop": summ["Proposed_Fast"]["DSR"],
        "RMST_prop": summ["Proposed_Fast"]["mean_RMST_ticks"],
        "paths": [m["target_path"] for m in missions]
    }
print("JSON_OUT:" + json.dumps(out))
"""
    env1 = {"PYTHONHASHSEED": "0"}
    env2 = {"PYTHONHASHSEED": "99999"}

    import os
    full_env1 = os.environ.copy()
    full_env1.update(env1)
    full_env1["PYTHONPATH"] = str(ROOT / "src")
    full_env2 = os.environ.copy()
    full_env2.update(env2)
    full_env2["PYTHONPATH"] = str(ROOT / "src")

    proc1 = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=full_env1, cwd=str(ROOT)
    )
    proc2 = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=full_env2, cwd=str(ROOT)
    )

    line1 = [l for l in proc1.stdout.splitlines() if l.startswith("JSON_OUT:")][0][9:]
    line2 = [l for l in proc2.stdout.splitlines() if l.startswith("JSON_OUT:")][0][9:]

    assert line1 == line2, "Results differed across processes with different PYTHONHASHSEED!"


def test_operator_stats_not_tripled_by_multi_regime():
    """Verify that evaluating across 3 GT regimes does NOT multiply search evaluation and operator counts."""
    res_3reg = run_multi_instance_benchmark(
        num_instances=1, num_missions_per_instance=5, master_seed=42, H=6, max_evals_total=20,
        gt_regimes=["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]
    )
    res_1reg = run_multi_instance_benchmark(
        num_instances=1, num_missions_per_instance=5, master_seed=42, H=6, max_evals_total=20,
        gt_regimes=["in_ensemble"]
    )

    ablation_3 = res_3reg["ablation_summary"]["Proposed_Fast"]
    ablation_1 = res_1reg["ablation_summary"]["Proposed_Fast"]

    # Mean total solver evals and operator evaluated counts must be identical between 1-regime and 3-regime benchmark
    assert ablation_3["mean_total_evals"] == ablation_1["mean_total_evals"]
    assert ablation_3["mean_accepted_moves"] == ablation_1["mean_accepted_moves"]

    op_3 = ablation_3["aggregated_operator_stats"]
    op_1 = ablation_1["aggregated_operator_stats"]

    for op_name in op_1:
        assert op_3[op_name]["evaluated"] == op_1[op_name]["evaluated"], (
            f"Operator {op_name} evaluated count differed! 3-regime={op_3[op_name]['evaluated']}, 1-regime={op_1[op_name]['evaluated']}"
        )
        assert op_3[op_name]["accepted"] == op_1[op_name]["accepted"]

