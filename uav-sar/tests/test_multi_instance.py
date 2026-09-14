"""Unit tests for the Multi-Instance benchmark module."""

import numpy as np
import pytest

from sar_uav.experiments.tier3_multi_instance import (
    create_instance,
    solve_instance_strategies,
    evaluate_instance_missions,
    run_multi_instance_benchmark,
)


def test_create_instance_structure():
    """Verify problem instances are created with valid specifications across all 10 designs."""
    for inst_id in range(10):
        inst = create_instance(inst_id, master_seed=42, H=12, num_uavs=2)
        assert inst.instance_id == inst_id
        assert inst.num_cells == 16
        assert inst.b0.shape == (16,)
        assert np.isclose(np.sum(inst.b0), 1.0)
        assert inst.M.shape == (16, 16)
        assert np.allclose(np.sum(inst.M, axis=1), 1.0)
        assert inst.true_M.shape == (16, 16)
        assert np.allclose(np.sum(inst.true_M, axis=1), 1.0)
        assert inst.true_lambda.shape == (16,)
        assert np.all(inst.true_lambda >= 0.0)
        assert inst.gt_category in ("in_ensemble", "nominal_matched", "misspecified_out_of_ensemble")


def test_solve_and_evaluate_single_instance():
    """Verify strategy solving and mission evaluation on a fast test instance."""
    inst = create_instance(0, master_seed=42, H=8, num_uavs=2)
    strategies, solver_stats = solve_instance_strategies(inst, max_evals_total=50, time_limit_sec=5.0)

    assert "Proposed_Fast" in strategies
    assert "SingleStart_Fast_FixedLS" in strategies
    assert "SingleStart_Fast_MatchedTotal" in strategies
    assert "Ablation_NoReplace" in strategies
    assert "Ablation_NoRebalance" in strategies
    assert "Ablation_NoReplaceRebalance" in strategies
    assert "Nominal_Planning" in strategies

    # Check solver stats
    for s_name, stats in solver_stats.items():
        assert "total_evals" in stats
        assert "total_runtime_sec" in stats
        assert stats["total_evals"] >= 0

    # Evaluate on 5 missions
    summary, paired, missions = evaluate_instance_missions(inst, strategies, num_missions=5)

    assert len(missions) == 5
    for s_name in strategies:
        assert s_name in summary
        assert 0.0 <= summary[s_name]["DSR"] <= 1.0
        assert 0.0 <= summary[s_name]["mean_RMST_ticks"] <= inst.H

    assert "Proposed_vs_Nominal_Planning" in paired
    assert "Proposed_vs_Ablation_NoReplace" in paired
    assert "mean_delta_DSR" in paired["Proposed_vs_Nominal_Planning"]


def test_multi_instance_benchmark_small_run():
    """Verify the multi-instance benchmark returns structured meta-analysis."""
    res = run_multi_instance_benchmark(
        num_instances=3,
        num_missions_per_instance=5,
        master_seed=123,
        H=8,
        max_evals_total=40,
        time_limit_sec=3.0
    )
    assert len(res["instances"]) in (3, 9)
    assert len(res["evaluations_by_scenario"]) == 3
    assert "cross_instance_summary" in res
    assert "cross_instance_paired" in res
    assert "regime_breakdown" in res
    assert "ablation_summary" in res

    # Verify regime breakdown has all 3 categories
    for cat in ("in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"):
        assert cat in res["regime_breakdown"]

    # Verify 4 ablation arms present
    for arm in ("Proposed_Fast", "Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"):
        assert arm in res["ablation_summary"]

    # Verify metadata and provenance
    assert "metadata" in res
    assert "run_id" in res["metadata"]
    assert res["metadata"]["run_id"].startswith("run_")
    assert "environment" in res["metadata"]
    assert "experiment_config" in res["metadata"]
    assert res["metadata"]["experiment_config"]["time_limit_sec"] == 3.0


def test_fixed_ls_budget_matches_total():
    """Verify FixedLS receives max_evals_total, not half the budget."""
    inst = create_instance(0, master_seed=42, H=8, num_uavs=2)
    max_evals = 100
    _, solver_stats = solve_instance_strategies(inst, max_evals_total=max_evals, time_limit_sec=5.0)

    # FixedLS must receive full max_evals_total for local search
    assert solver_stats["SingleStart_Fast_FixedLS"]["max_evals_budget"] == max_evals
    # MatchedTotal receives max_evals_total + greedy_nom_evals
    assert solver_stats["SingleStart_Fast_MatchedTotal"]["max_evals_budget"] >= max_evals


def test_scaling_beyond_10_instances():
    """Verify num_instances > 10 cleanly wraps scenario_id with replicate_id and does not duplicate instance 9."""
    for inst_id in range(25):
        inst = create_instance(inst_id, master_seed=42, H=8, num_uavs=2)
        scen_id = inst_id % 10
        rep_id = inst_id // 10
        if rep_id == 0:
            assert f"Instance_{inst_id}_" in inst.name
        else:
            assert f"Instance_{inst_id}_Scen{scen_id}_Rep{rep_id}_" in inst.name
        if scen_id in (0, 1, 2, 3):
            assert inst.gt_category == "in_ensemble"
        elif scen_id in (4, 5, 6):
            assert inst.gt_category == "nominal_matched"
        else:
            assert inst.gt_category == "misspecified_out_of_ensemble"


def test_num_instances_invalid_raises():
    """Verify non-positive num_instances raises ValueError."""
    with pytest.raises(ValueError, match="num_instances must be >= 1"):
        run_multi_instance_benchmark(num_instances=0)


def test_multi_instance_figure_provenance_and_verification():
    """Verify verification checks enforce conditions and provenance metadata format."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from plot_multi_instance_figures import (
        verify_fig4_compatibility,
        verify_fig5_compatibility,
        compute_file_sha256,
    )

    # Incomplete/empty data should fail verification
    empty_data = {}
    v4_pass, v4_checks = verify_fig4_compatibility(empty_data)
    assert not v4_pass
    assert not v4_checks["run_id_present"]
    assert not v4_checks["instances_non_empty"]

    v5_pass, v5_checks = verify_fig5_compatibility(empty_data)
    assert not v5_pass
    assert not v5_checks["run_id_present"]
    assert not v5_checks["ablation_arms_complete"]

    # Verify sha256 computation
    test_file = Path(__file__)
    h = compute_file_sha256(test_file)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_fixed_ls_budget_consistency_verification():
    """Verify FixedLS budget verification checks instance allocation against config, not a hardcoded 600."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from plot_multi_instance_figures import verify_fig4_compatibility

    base_dummy = {
        "metadata": {
            "run_id": "run_test_budget",
            "environment": {"git_hash": "a1b2c3d4e5"},
            "experiment_config": {
                "num_instances": 1,
                "max_evals_total": 100  # Budget is 100, NOT 600
            }
        },
        "cross_instance_summary": {
            s: {"mean_DSR": 0.5, "mean_RMST_ticks": 10.0}
            for s in [
                "Proposed_Fast", "Nominal_Planning", "SingleStart_Fast_FixedLS",
                "SingleStart_Fast_MatchedTotal", "Greedy_Lookahead", "Adaptive_GA"
            ]
        },
        "cross_instance_paired": {
            p: {"mean_delta_DSR": 0.05, "ci95_delta_DSR": [0.01, 0.09]}
            for p in [
                "Proposed_vs_Nominal_Planning",
                "Proposed_vs_SingleStart_Fast_FixedLS",
                "Proposed_vs_SingleStart_Fast_MatchedTotal",
                "Proposed_vs_Greedy_Lookahead",
                "Proposed_vs_Adaptive_GA"
            ]
        }
    }

    # Case 1: Budget is 100 (different from 600), and FixedLS receives 100 -> MUST PASS
    data_100 = dict(base_dummy)
    data_100["instances"] = [{
        "instance_id": 0,
        "solver_stats": {
            "SingleStart_Fast_FixedLS": {
                "max_evals_budget": 100,
                "local_search_evals": 75  # stopped early, <= 100
            }
        }
    }]
    passed, checks = verify_fig4_compatibility(data_100)
    assert checks["budget_consistent_fixed_ls"] is True
    assert passed is True

    # Case 2: Config specifies 600, but FixedLS in instance was allocated 300 -> MUST FAIL
    data_mismatch = dict(base_dummy)
    data_mismatch["metadata"] = {
        "run_id": "run_test_mismatch",
        "environment": {"git_hash": "a1b2c3d4e5"},
        "experiment_config": {"num_instances": 1, "max_evals_total": 600}
    }
    data_mismatch["instances"] = [{
        "instance_id": 0,
        "solver_stats": {
            "SingleStart_Fast_FixedLS": {
                "max_evals_budget": 300,  # 300 != 600
                "local_search_evals": 300
            }
        }
    }]
    passed_mismatch, checks_mismatch = verify_fig4_compatibility(data_mismatch)
    assert checks_mismatch["budget_consistent_fixed_ls"] is False
    assert passed_mismatch is False

    # Case 3: Config specifies 600, budget is 600, but actual evals exceed budget (650 > 600) -> MUST FAIL
    data_exceeded = dict(base_dummy)
    data_exceeded["metadata"] = {
        "run_id": "run_test_exceeded",
        "environment": {"git_hash": "a1b2c3d4e5"},
        "experiment_config": {"num_instances": 1, "max_evals_total": 600}
    }
    data_exceeded["instances"] = [{
        "instance_id": 0,
        "solver_stats": {
            "SingleStart_Fast_FixedLS": {
                "max_evals_budget": 600,
                "local_search_evals": 650  # Over budget!
            }
        }
    }]
    passed_exceeded, checks_exceeded = verify_fig4_compatibility(data_exceeded)
    assert checks_exceeded["budget_consistent_fixed_ls"] is False
    assert passed_exceeded is False

    # Case 4: Missing local_search_evals (only total_evals or nothing) -> MUST FAIL (no guesswork)
    data_missing_field = dict(base_dummy)
    data_missing_field["metadata"] = {
        "run_id": "run_test_missing_field",
        "environment": {"git_hash": "a1b2c3d4e5"},
        "experiment_config": {"num_instances": 1, "max_evals_total": 600}
    }
    data_missing_field["instances"] = [{
        "instance_id": 0,
        "solver_stats": {
            "SingleStart_Fast_FixedLS": {
                "max_evals_budget": 600,
                "total_evals": 790  # Has total_evals, but missing local_search_evals
            }
        }
    }]
    passed_missing, checks_missing = verify_fig4_compatibility(data_missing_field)
    assert checks_missing["budget_consistent_fixed_ls"] is False
    assert passed_missing is False


def test_operator_stats_verification_distinguishes_unevaluated():
    """Verify operator stats schema distinguishes unevaluated operators from missing/malformed stats."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from plot_multi_instance_figures import verify_fig5_compatibility

    base_ablation = {
        "metadata": {
            "run_id": "run_ablation_test",
            "environment": {"git_hash": "a1b2c3d4e5"}
        },
        "ablation_summary": {
            "Proposed_Fast": {},
            "Ablation_NoReplace": {},
            "Ablation_NoRebalance": {},
            "Ablation_NoReplaceRebalance": {}
        }
    }

    # Case 1: Valid stats with some operators unevaluated (0 evals or key omitted) -> PASS
    data_valid = dict(base_ablation)
    data_valid["ablation_summary"]["Proposed_Fast"] = {
        "aggregated_operator_stats": {
            "Replace": {"evaluated": 0, "accepted": 0, "total_improvement": 0.0},
            "DwellRebalance": {"evaluated": 15, "accepted": 2, "total_improvement": 0.05}
        }
    }
    passed_valid, checks_valid = verify_fig5_compatibility(data_valid)
    assert checks_valid["operator_stats_valid"] is True
    assert passed_valid is True

    # Case 2: Empty dict (small budget run with no neighborhood candidates evaluated) -> PASS
    data_empty_ops = dict(base_ablation)
    data_empty_ops["ablation_summary"]["Proposed_Fast"] = {
        "aggregated_operator_stats": {}
    }
    passed_empty, checks_empty = verify_fig5_compatibility(data_empty_ops)
    assert checks_empty["operator_stats_valid"] is True
    assert passed_empty is True

    # Case 3: Missing/malformed operator stats (None instead of dict) -> FAIL
    data_malformed = dict(base_ablation)
    data_malformed["ablation_summary"]["Proposed_Fast"] = {
        "aggregated_operator_stats": None
    }
    passed_malformed, checks_malformed = verify_fig5_compatibility(data_malformed)
    assert checks_malformed["operator_stats_valid"] is False
    assert passed_malformed is False


def test_unequal_replicates_benchmark_aggregation_and_metadata():
    """Verify benchmark correctly handles unequal replicates per scenario with equal weighting."""
    res = run_multi_instance_benchmark(
        num_instances=15,
        num_missions_per_instance=2,
        master_seed=42,
        H=6,
        max_evals_total=20,
        time_limit_sec=2.0
    )

    # Check metadata separation
    exp_cfg = res["metadata"]["experiment_config"]
    assert exp_cfg["num_unique_scenarios"] == 10
    assert exp_cfg["num_designs"] == 15
    assert exp_cfg["aggregation_weighting"] == "equal_scenario_weight"
    assert exp_cfg["replicates_per_scenario"][0] == 2
    assert exp_cfg["replicates_per_scenario"][5] == 1

    # Check equal scenario weighting in summary
    for s_name, s_summary in res["cross_instance_summary"].items():
        scen_means = s_summary["per_scenario_mean_DSR"]
        assert len(scen_means) == 10
        assert abs(s_summary["mean_DSR"] - float(np.mean(scen_means))) < 1e-9

    # Check paired comparisons have scenario-level weighting
    for p_name, p_data in res["cross_instance_paired"].items():
        scen_deltas = p_data["per_scenario_delta_DSR"]
        assert len(scen_deltas) == 10
        assert abs(p_data["mean_delta_DSR"] - float(np.mean(scen_deltas))) < 1e-9

    # Check regime breakdown has scenario-level CIs
    for cat, reg_data in res["regime_breakdown"].items():
        assert "ci95_delta_DSR_student_t" in reg_data["delta_Proposed_vs_Nominal_2Start"]
        assert "ci95_delta_DSR_bootstrap" in reg_data["delta_Proposed_vs_Nominal_2Start"]

    # Check ablation summary has scenario-level CIs
    for arm, arm_data in res["ablation_summary"].items():
        assert "ci95_J_ensemble" in arm_data
        assert "ci95_DSR" in arm_data
        assert "ci95_delta_DSR_student_t" in arm_data


def test_plot_figures_supports_variable_configurations(tmp_path):
    """Verify plot scripts handle 3, 10, and 15 design results seamlessly."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from plot_multi_instance_figures import plot_fig4_multi_instance, plot_fig5_operator_ablation

    for n_inst in (3, 10, 15):
        res = run_multi_instance_benchmark(
            num_instances=n_inst,
            num_missions_per_instance=2,
            master_seed=42,
            H=6,
            max_evals_total=20,
            time_limit_sec=2.0
        )
        out_dir = tmp_path / f"figs_n{n_inst}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_fig4_multi_instance(res, out_dir)
        plot_fig5_operator_ablation(res, out_dir)
        assert (out_dir / "fig4_multi_instance_benchmark.png").exists()
        assert (out_dir / "fig5_operator_ablation.png").exists()


