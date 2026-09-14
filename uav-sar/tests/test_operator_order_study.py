import pytest
import numpy as np
import pandas as pd
from sar_uav.experiments.stats import student_t_ci
from scripts.run_operator_order_study import aggregate_operator_study_results


def test_operator_order_study_unequal_replicates_equal_scenario_weight():
    """
    Test that aggregate_operator_study_results correctly computes paired differences
    with equal scenario weighting when scenarios have unequal replicate counts.
    """
    # Create synthetic raw_records with 3 scenarios having unequal replicates:
    # Scenario 0: 3 replicates (rep 0, 1, 2)
    # Scenario 1: 1 replicate  (rep 0)
    # Scenario 2: 2 replicates (rep 0, 1)
    # Total designs = 6
    scenario_replicates = {0: [0, 1, 2], 1: [0], 2: [0, 1]}
    order_strategies = ["fixed", "round_robin"]
    ablation_arms = ["Proposed_Fast", "Ablation_NoReplace", "Ablation_NoRebalance", "Ablation_NoReplaceRebalance"]
    gt_regimes = ["in_ensemble", "nominal_matched", "misspecified_out_of_ensemble"]

    raw_records = []
    # Seeded synthetic J values to verify manual vs automated calculation
    # Scenario 0:
    #   rep 0: Proposed J=0.30, NoReplace J=0.20 -> delta=0.10
    #   rep 1: Proposed J=0.32, NoReplace J=0.20 -> delta=0.12
    #   rep 2: Proposed J=0.34, NoReplace J=0.20 -> delta=0.14
    #   Mean delta for scenario 0 = (0.10 + 0.12 + 0.14) / 3 = 0.12
    # Scenario 1:
    #   rep 0: Proposed J=0.40, NoReplace J=0.10 -> delta=0.30
    #   Mean delta for scenario 1 = 0.30
    # Scenario 2:
    #   rep 0: Proposed J=0.25, NoReplace J=0.20 -> delta=0.05
    #   rep 1: Proposed J=0.27, NoReplace J=0.20 -> delta=0.07
    #   Mean delta for scenario 2 = (0.05 + 0.07) / 2 = 0.06
    #
    # Equal scenario mean delta J = (0.12 + 0.30 + 0.06) / 3 = 0.48 / 3 = 0.16
    # Note: Unweighted row-average would be (0.10+0.12+0.14+0.30+0.05+0.07)/6 = 0.78 / 6 = 0.13 (different!)

    for order in order_strategies:
        for sc_id, reps in scenario_replicates.items():
            for rep_id in reps:
                inst_id = sc_id * 10 + rep_id
                for regime in gt_regimes:
                    for arm in ablation_arms:
                        # Base J_ensemble
                        if arm == "Proposed_Fast":
                            if order == "fixed":
                                if sc_id == 0:
                                    j_val = 0.30 + 0.02 * rep_id
                                elif sc_id == 1:
                                    j_val = 0.40
                                else:
                                    j_val = 0.25 + 0.02 * rep_id
                            else:  # round_robin
                                # Give a different J for order comparison testing
                                j_val = 0.35 + 0.01 * sc_id
                        elif arm == "Ablation_NoReplace":
                            if sc_id == 1:
                                j_val = 0.10
                            else:
                                j_val = 0.20
                        else:
                            j_val = 0.22

                        raw_records.append({
                            "order_strategy": order,
                            "instance_id": inst_id,
                            "scenario_id": sc_id,
                            "replicate_id": rep_id,
                            "gt_regime": regime,
                            "arm": arm,
                            "J_ensemble": j_val,
                            "J_true_gt": j_val,
                            "DSR": 0.30 if arm == "Proposed_Fast" else 0.25,
                            "RMST": 12.0,
                            "energy_kJ": 250.0,
                            "runtime_sec": 0.15,
                            "total_evals": 600,
                            "accepted_moves": 4,
                            "operator_stats": {}
                        })

    (
        results_matrix,
        summary_table,
        paired_within,
        paired_summary,
        paired_across
    ) = aggregate_operator_study_results(
        raw_records=raw_records,
        order_strategies=order_strategies,
        ablation_arms=ablation_arms,
        H=15,
        alpha=0.05,
        n_boot=100,
        master_seed=42
    )

    # 1. Verify within-order paired comparison (fixed, Proposed vs NoReplace)
    comp = paired_within["fixed"]["Proposed_vs_Ablation_NoReplace"]
    scen_deltas = comp["per_scenario_delta_J"]

    assert len(scen_deltas) == 3, f"Expected 3 scenario deltas, got {len(scen_deltas)}"
    assert np.isclose(scen_deltas[0], 0.12), f"Scenario 0 mean delta should be 0.12, got {scen_deltas[0]}"
    assert np.isclose(scen_deltas[1], 0.30), f"Scenario 1 mean delta should be 0.30, got {scen_deltas[1]}"
    assert np.isclose(scen_deltas[2], 0.06), f"Scenario 2 mean delta should be 0.06, got {scen_deltas[2]}"

    expected_mean_delta = 0.16
    assert np.isclose(comp["mean_delta_J"], expected_mean_delta), (
        f"Mean delta J must equal {expected_mean_delta} (Equal Scenario Weighting), got {comp['mean_delta_J']}"
    )

    # Verify student_t CI on the 3 scenario means
    _, _, expected_ci = student_t_ci([0.12, 0.30, 0.06], alpha=0.05)
    assert np.isclose(comp["ci95_delta_J_student_t"][0], expected_ci[0])
    assert np.isclose(comp["ci95_delta_J_student_t"][1], expected_ci[1])

    # 2. Verify cross-order comparisons also group by scenario_id first
    cross_key = "Proposed_Fast_round_robin_vs_fixed"
    assert cross_key in paired_across
    cross_comp = paired_across[cross_key]
    assert len(cross_comp["per_scenario_delta_J"]) == 3
