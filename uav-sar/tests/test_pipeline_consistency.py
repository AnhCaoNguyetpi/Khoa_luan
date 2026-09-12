"""Pipeline consistency tests — verifies end-to-end invariants across
solver, evaluator, checker, and simulator.

Each test targets a specific cross-component contract identified in the
code review.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

from sar_uav.detection.hypothesis import DetectionHypothesis, HypothesisSet
from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, UAVSpec, Visit
from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator, FullEvaluator, PrefixOnlyEvaluator
from sar_uav.planning.neighborhood import NeighborhoodExplorer, detect_affected_interval
from sar_uav.planning.local_search import JointRouteEffortLocalSearch
from sar_uav.planning.baselines import ExactBruteForcePlanner
from sar_uav.sim.mission import ScheduleSimulator
from sar_uav.experiments.rng import IndexedRNGStream


@pytest.fixture
def small_setup():
    """3x3 grid, 1 UAV, diffusive target, 3 hypotheses."""
    num_cells = 9
    H = 12
    delta_t = 60.0
    depot_idx = 0

    coords = [(r, c) for r in range(3) for c in range(3)]
    dist_m = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            dist_m[i, j] = np.hypot(coords[i][0] - coords[j][0],
                                     coords[i][1] - coords[j][1]) * 300.0

    spec = UAVSpec(speed_ms=10.0, battery_joules=200000.0,
                   reserve_joules=25000.0, delta_t=delta_t)
    checker = ConstraintChecker(dist_m, depot_idx, [spec], H, delta_t=delta_t)

    # Stochastic motion matrix (diffusion)
    M = np.eye(num_cells) * 0.7
    for i in range(num_cells):
        r, c = coords[i]
        nbrs = [j for j in range(num_cells) if i != j and abs(coords[j][0] - r) + abs(coords[j][1] - c) == 1]
        if nbrs:
            for n in nbrs:
                M[i, n] = 0.3 / len(nbrs)
        else:
            M[i, i] = 1.0

    b0 = np.full(num_cells, 1.0 / num_cells)
    b0[4] = 0.5
    b0 /= b0.sum()

    hset = HypothesisSet([
        DetectionHypothesis(id=0, name="low", weight=0.3,
                            lambda_rates=np.full(num_cells, 0.08)),
        DetectionHypothesis(id=1, name="mid", weight=0.4,
                            lambda_rates=np.full(num_cells, 0.15)),
        DetectionHypothesis(id=2, name="high", weight=0.3,
                            lambda_rates=np.full(num_cells, 0.25)),
    ])

    engine = JointMassEngine(b0, hset, M, H, delta_t=delta_t)
    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
    return checker, engine, candidates, H, delta_t, num_cells, hset


# ---------------------------------------------------------------
# Test 1: Solver returns a schedule with correct chaining and
# the reported J matches a fresh full-trajectory evaluation.
# ---------------------------------------------------------------
def test_solver_returns_chained_schedule(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    explorer = NeighborhoodExplorer(checker, candidates, allowed_dwells=(1, 2))
    evaluator = ForwardBackwardFastEvaluator(engine)

    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2)])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )

    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=100, time_limit_sec=5.0)
    best_sched, best_J, stats = solver.solve(init_sched)

    # 1a. Final schedule must be feasible
    feasible, reason = checker.check_joint_schedule(best_sched)
    assert feasible, f"Final schedule infeasible: {reason}"

    # 1b. Reported J must match fresh full-trajectory evaluation
    fp = best_sched.generate_footprint_array()
    _, _, J_recomputed = engine.compute_forward_trajectory(fp)
    assert np.isclose(best_J, J_recomputed, atol=1e-12, rtol=0), \
        f"Reported J={best_J} != recomputed J={J_recomputed}"


# ---------------------------------------------------------------
# Test 2: All 3 evaluators agree after multiple accepted moves,
# including localized affected intervals with moving target.
# ---------------------------------------------------------------
def test_evaluator_equivalence_after_multiple_accepts(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    eval_full = FullEvaluator(engine)
    eval_prefix = PrefixOnlyEvaluator(engine)
    eval_fast = ForwardBackwardFastEvaluator(engine)

    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[
            Visit(cell_idx=1, dwell_ticks=2, wait_ticks=0),
            Visit(cell_idx=4, dwell_ticks=2, wait_ticks=0),
        ])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    checker.check_joint_schedule(init_sched)

    J_full = eval_full.initialize_base_schedule(init_sched)
    J_prefix = eval_prefix.initialize_base_schedule(init_sched)
    J_fast = eval_fast.initialize_base_schedule(init_sched)
    assert np.isclose(J_full, J_prefix, atol=1e-12, rtol=0)
    assert np.isclose(J_full, J_fast, atol=1e-12, rtol=0)

    # Candidate 1: Replace cell of visit 0
    cand1 = init_sched.clone()
    cand1.uav_schedules[0].visits[0].cell_idx = 3
    checker.check_joint_schedule(cand1)
    t_a1, t_b1 = detect_affected_interval(init_sched, cand1)

    dJ_full1, J1_f = eval_full.evaluate_candidate(cand1, t_a1, t_b1)
    dJ_pref1, J1_p = eval_prefix.evaluate_candidate(cand1, t_a1, t_b1)
    dJ_fast1, J1_b = eval_fast.evaluate_candidate(cand1, t_a1, t_b1)

    assert np.isclose(J1_f, J1_p, atol=1e-12, rtol=0)
    assert np.isclose(J1_f, J1_b, atol=1e-12, rtol=0)

    eval_full.accept_candidate(cand1, J1_f)
    eval_prefix.accept_candidate(cand1, J1_p)
    eval_fast.accept_candidate(cand1, J1_b)

    # Candidate 2: Change dwell of visit 1
    cand2 = cand1.clone()
    cand2.uav_schedules[0].visits[1].dwell_ticks = 1
    checker.check_joint_schedule(cand2)
    t_a2, t_b2 = detect_affected_interval(cand1, cand2)

    dJ_full2, J2_f = eval_full.evaluate_candidate(cand2, t_a2, t_b2)
    dJ_pref2, J2_p = eval_prefix.evaluate_candidate(cand2, t_a2, t_b2)
    dJ_fast2, J2_b = eval_fast.evaluate_candidate(cand2, t_a2, t_b2)

    assert np.isclose(J2_f, J2_p, atol=1e-12, rtol=0)
    assert np.isclose(J2_f, J2_b, atol=1e-12, rtol=0)


# ---------------------------------------------------------------
# Test 3: Two UAVs at the same cell produce higher detection
# probability than one UAV (coverage > 1.0 accumulates).
# ---------------------------------------------------------------
def test_two_uavs_same_cell_detection(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    sched_1 = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=3)])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    checker.check_joint_schedule(sched_1)
    fp1 = sched_1.generate_footprint_array()
    _, _, J_one = engine.compute_forward_trajectory(fp1)

    spec2 = UAVSpec(speed_ms=10.0, battery_joules=200000.0,
                    reserve_joules=25000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(
        checker.dist_matrix, checker.depot_idx, [checker.uav_specs[0], spec2],
        H, delta_t=delta_t
    )

    sched_2 = JointSchedule(
        uav_schedules=[
            UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=3)]),
            UAVSchedule(uav_idx=1, visits=[Visit(cell_idx=4, dwell_ticks=3)]),
        ],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    checker2.check_joint_schedule(sched_2)
    fp2 = sched_2.generate_footprint_array()

    start_t = sched_2.uav_schedules[0].visits[0].start_tick
    for t in range(start_t, start_t + 3):
        if t < H:
            assert fp2[t, 4] == 2.0, f"Expected coverage=2.0 at tick {t}, got {fp2[t, 4]}"

    _, _, J_two = engine.compute_forward_trajectory(fp2)
    assert J_two > J_one, f"Two UAVs (J={J_two}) must detect better than one (J={J_one})"


# ---------------------------------------------------------------
# Test 4: Simulator sortie energy strictly matches ConstraintChecker
# on full mission without detection (exact match at atol=1e-6).
# ---------------------------------------------------------------
def test_energy_simulator_matches_checker(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[
            Visit(cell_idx=1, dwell_ticks=2, wait_ticks=1),
            Visit(cell_idx=4, dwell_ticks=3, wait_ticks=0)
        ])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    checker.check_joint_schedule(sched)

    spec = checker.uav_specs[0]
    expected_energy = (
        spec.p_idle * 1 * delta_t +
        checker.e_bar[0, 0, 1] +
        spec.p_search * 2 * delta_t +
        checker.e_bar[0, 1, 4] +
        spec.p_search * 3 * delta_t +
        checker.e_bar[0, 4, 0]
    )

    target_path = np.full(H, 8, dtype=int)
    true_rates = np.full(num_cells, 0.15)
    sim = ScheduleSimulator(checker, true_rates, delta_t=delta_t)
    rng = IndexedRNGStream(master_seed=999)

    result = sim.run_simulation(sched, target_path, mission_idx=0, rng_stream=rng)
    assert not result.detected

    assert np.isclose(result.actual_energy_joules, expected_energy, rtol=0, atol=1e-6), \
        f"Sim energy={result.actual_energy_joules} != checker energy={expected_energy}"


# ---------------------------------------------------------------
# Test 4b: Simulator energy boundary cases for early detection
# (First tick of dwell, middle tick, exact last tick).
# ---------------------------------------------------------------
def test_energy_simulator_early_detection_boundaries(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    spec = checker.uav_specs[0]
    sched = UAVSchedule(0, [
        Visit(cell_idx=1, dwell_ticks=3, wait_ticks=1),
        Visit(cell_idx=4, dwell_ticks=2, wait_ticks=0),
    ])
    checker.compute_and_update_chaining(sched)
    joint_sched = JointSchedule([sched], num_cells, H, delta_t)

    v0 = sched.visits[0]
    t_dwell_first = v0.start_tick
    t_dwell_mid = v0.start_tick + 1
    t_dwell_last = v0.start_tick + v0.dwell_ticks - 1

    def check_detection_at_tick(tick, expected_e):
        target_path = np.full(H, 8, dtype=int)
        target_path[tick] = 1
        rates = np.zeros(num_cells)
        rates[1] = 100.0
        sim_forced = ScheduleSimulator(checker, rates, delta_t=delta_t)
        res = sim_forced.run_simulation(joint_sched, target_path, mission_idx=0,
                                        rng_stream=IndexedRNGStream(master_seed=42))
        assert res.detected
        assert res.detection_tick == tick
        assert np.isclose(res.actual_energy_joules, expected_e, rtol=0, atol=1e-6), \
            f"At tick {tick}: got {res.actual_energy_joules}, expected {expected_e}"

    # 1. Detection at FIRST tick of dwell 1
    e_first = spec.p_idle * 1 * delta_t + checker.e_bar[0, 0, 1] + spec.p_search * 1 * delta_t + checker.e_bar[0, 1, 0]
    check_detection_at_tick(t_dwell_first, e_first)

    # 2. Detection at MIDDLE tick of dwell 1
    e_mid = spec.p_idle * 1 * delta_t + checker.e_bar[0, 0, 1] + spec.p_search * 2 * delta_t + checker.e_bar[0, 1, 0]
    check_detection_at_tick(t_dwell_mid, e_mid)

    # 3. Detection at EXACT LAST TICK of dwell 1 (Critical Boundary)
    e_last = spec.p_idle * 1 * delta_t + checker.e_bar[0, 0, 1] + spec.p_search * 3 * delta_t + checker.e_bar[0, 1, 0]
    check_detection_at_tick(t_dwell_last, e_last)


# ---------------------------------------------------------------
# Test 4c-1: Multi-UAV Early Detection - Companion delayed at depot
# ---------------------------------------------------------------
def test_companion_state_delayed_at_depot(small_setup):
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec0 = checker1.uav_specs[0]
    spec1 = UAVSpec(name="uav_comp", speed_ms=10.0, battery_joules=300000.0,
                    p_flight=190.0, p_search=210.0, p_idle=50.0,
                    reserve_joules=30000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx, [spec0, spec1], H, delta_t=delta_t)

    # UAV 0 detects at tick 1 (first tick of search at cell 1)
    sched0 = UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=3, wait_ticks=0)])
    checker2.compute_and_update_chaining(sched0)
    t_det = sched0.visits[0].start_tick
    assert t_det == 1

    # Companion delayed at depot for 3 ticks (ticks 0, 1, 2)
    # Entire schedule: wait 3 + fly 1 + dwell 2 + return 1 = 7 ticks <= H (10)
    sched1 = UAVSchedule(1, [Visit(cell_idx=2, dwell_ticks=2, wait_ticks=3)])
    joint_sched = JointSchedule([sched0.clone(), sched1], num_cells, H, delta_t)

    feasible, reason = checker2.check_joint_schedule(joint_sched)
    assert feasible, f"Precondition: schedule must be fully feasible: {reason}"
    assert sched1.visits[0].wait_ticks > t_det, "Precondition: companion still waiting at depot at t_det"

    target_path = np.full(H, 7, dtype=int)
    target_path[t_det] = 1
    rates = np.zeros(num_cells)
    rates[1] = 100.0
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res = sim.run_simulation(joint_sched, target_path, mission_idx=0,
                             rng_stream=IndexedRNGStream(master_seed=42))
    assert res.detected and res.detection_tick == t_det

    e_uav0 = checker2.e_bar[0, 0, 1] + spec0.p_search * 1 * delta_t + checker2.e_bar[0, 1, 0]
    e_uav1_delayed = spec1.p_idle * (t_det + 1) * delta_t
    assert np.isclose(res.actual_energy_joules, e_uav0 + e_uav1_delayed, rtol=0, atol=1e-6)


# ---------------------------------------------------------------
# Test 4c-2: Multi-UAV Early Detection - Companion strictly mid-flight
# ---------------------------------------------------------------
def test_companion_state_midflight(small_setup):
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec0 = checker1.uav_specs[0]
    # Speed 4 m/s: dist(0,8)=848.5m -> raw 212s -> tau=4 ticks (ticks 0,1,2,3)
    # Total: fly 4 + dwell 1 + return 4 = 9 ticks <= H (10), 100% feasible!
    spec1 = UAVSpec(name="uav_comp", speed_ms=4.0, battery_joules=500000.0,
                    p_flight=190.0, p_search=210.0, p_idle=50.0,
                    reserve_joules=30000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx, [spec0, spec1], H, delta_t=delta_t)

    sched0 = UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=3, wait_ticks=0)])
    checker2.compute_and_update_chaining(sched0)
    t_det = sched0.visits[0].start_tick
    assert t_det == 1

    sched1 = UAVSchedule(1, [Visit(cell_idx=8, dwell_ticks=1, wait_ticks=0)])
    joint_sched = JointSchedule([sched0.clone(), sched1], num_cells, H, delta_t)

    feasible, reason = checker2.check_joint_schedule(joint_sched)
    assert feasible, f"Precondition: schedule must be fully feasible: {reason}"

    tau_out = int(checker2.tau_bar[1, 0, 8])
    flight_end_tick = tau_out - 1  # ticks 0..tau_out-1
    assert t_det < flight_end_tick, f"Precondition: t_det={t_det} must be before flight ends at {flight_end_tick}"

    target_path = np.full(H, 7, dtype=int)
    target_path[t_det] = 1
    rates = np.zeros(num_cells)
    rates[1] = 100.0
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res = sim.run_simulation(joint_sched, target_path, mission_idx=0,
                             rng_stream=IndexedRNGStream(master_seed=42))
    assert res.detected and res.detection_tick == t_det

    e_uav0 = checker2.e_bar[0, 0, 1] + spec0.p_search * 1 * delta_t + checker2.e_bar[0, 1, 0]
    e_uav1_midflight = checker2.e_bar[1, 0, 8] + checker2.e_bar[1, 8, 0]
    assert np.isclose(res.actual_energy_joules, e_uav0 + e_uav1_midflight, rtol=0, atol=1e-6)


# ---------------------------------------------------------------
# Test 4c-3: Multi-UAV Early Detection - Companion waiting on-station
# ---------------------------------------------------------------
def test_companion_state_waiting_on_station(small_setup):
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec0 = checker1.uav_specs[0]
    spec1 = UAVSpec(name="uav_comp", speed_ms=10.0, battery_joules=300000.0,
                    p_flight=190.0, p_search=210.0, p_idle=50.0,
                    reserve_joules=30000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx, [spec0, spec1], H, delta_t=delta_t)

    # UAV 0 waits 4 ticks at depot, starts search at cell 1 at tick 5
    sched0 = UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=3, wait_ticks=4)])
    checker2.compute_and_update_chaining(sched0)
    t_det = sched0.visits[0].start_tick
    assert t_det == 5

    # Companion: visit cell 2 (dwell 1), then visit cell 3 with wait_ticks=2.
    # Total schedule fits within H (8 ticks <= 10).
    sched1 = UAVSchedule(1, [
        Visit(cell_idx=2, dwell_ticks=1, wait_ticks=0),
        Visit(cell_idx=3, dwell_ticks=1, wait_ticks=2),
    ])
    joint_sched = JointSchedule([sched0.clone(), sched1], num_cells, H, delta_t)

    feasible, reason = checker2.check_joint_schedule(joint_sched)
    assert feasible, f"Precondition: schedule must be fully feasible: {reason}"

    v0_comp = sched1.visits[0]
    v1_comp = sched1.visits[1]
    tau_2_3 = int(checker2.tau_bar[1, 2, 3])
    wait_start = v0_comp.start_tick + v0_comp.dwell_ticks + tau_2_3
    wait_end = wait_start + v1_comp.wait_ticks - 1
    assert wait_start <= t_det <= wait_end, (
        f"Precondition: companion must be waiting on-station [{wait_start}, {wait_end}] at t_det={t_det}"
    )

    target_path = np.full(H, 7, dtype=int)
    target_path[t_det] = 1
    rates = np.zeros(num_cells)
    rates[1] = 100.0
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res = sim.run_simulation(joint_sched, target_path, mission_idx=0,
                             rng_stream=IndexedRNGStream(master_seed=42))
    assert res.detected and res.detection_tick == t_det

    e_uav0 = (spec0.p_idle * 4 * delta_t
              + checker2.e_bar[0, 0, 1]
              + spec0.p_search * 1 * delta_t
              + checker2.e_bar[0, 1, 0])
    completed_wait = t_det - wait_start + 1
    e_uav1 = (checker2.e_bar[1, 0, 2]
              + spec1.p_search * v0_comp.dwell_ticks * delta_t
              + checker2.e_bar[1, 2, 3]
              + spec1.p_idle * completed_wait * delta_t
              + checker2.e_bar[1, 3, 0])
    assert np.isclose(res.actual_energy_joules, e_uav0 + e_uav1, rtol=0, atol=1e-6)


# ---------------------------------------------------------------
# Test 4c-4: Multi-UAV Early Detection - Companion returning to depot
# ---------------------------------------------------------------
def test_companion_state_returning_to_depot(small_setup):
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec0 = checker1.uav_specs[0]
    # Speed 4 m/s: fly 0->8 is 4 ticks (0..3), dwell 1 tick (4), return 8->0 is 4 ticks (5..8), depot at 9 <= 10
    spec1 = UAVSpec(name="uav_comp", speed_ms=4.0, battery_joules=500000.0,
                    p_flight=190.0, p_search=210.0, p_idle=50.0,
                    reserve_joules=30000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx, [spec0, spec1], H, delta_t=delta_t)

    # UAV 0 waits 4 ticks, searches cell 1 at tick 5
    sched0 = UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=3, wait_ticks=4)])
    checker2.compute_and_update_chaining(sched0)
    t_det = sched0.visits[0].start_tick
    assert t_det == 5

    sched1 = UAVSchedule(1, [Visit(cell_idx=8, dwell_ticks=1, wait_ticks=0)])
    joint_sched = JointSchedule([sched0.clone(), sched1], num_cells, H, delta_t)

    feasible, reason = checker2.check_joint_schedule(joint_sched)
    assert feasible, f"Precondition: schedule must be fully feasible: {reason}"

    v1 = sched1.visits[0]
    ret_start = v1.start_tick + v1.dwell_ticks  # 4 + 1 = 5
    tau_ret = int(checker2.tau_bar[1, 8, 0])    # 4
    ret_end = ret_start + tau_ret - 1           # 5 + 4 - 1 = 8

    # Strict assertion precondition: no if-branching bypass!
    assert ret_start <= t_det <= ret_end, (
        f"Precondition: detection at {t_det} must fall strictly within return interval [{ret_start}, {ret_end}]"
    )

    target_path = np.full(H, 7, dtype=int)
    target_path[t_det] = 1
    rates = np.zeros(num_cells)
    rates[1] = 100.0
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res = sim.run_simulation(joint_sched, target_path, mission_idx=0,
                             rng_stream=IndexedRNGStream(master_seed=42))
    assert res.detected and res.detection_tick == t_det

    e_uav0 = (spec0.p_idle * 4 * delta_t
              + checker2.e_bar[0, 0, 1]
              + spec0.p_search * 1 * delta_t
              + checker2.e_bar[0, 1, 0])
    e_uav1_return = (checker2.e_bar[1, 0, 8]
                     + spec1.p_search * 1 * delta_t
                     + checker2.e_bar[1, 8, 0])
    assert np.isclose(res.actual_energy_joules, e_uav0 + e_uav1_return, rtol=0, atol=1e-6)


# ---------------------------------------------------------------
# Test 4c-5: Multi-UAV Early Detection - Companion already returned
# ---------------------------------------------------------------
def test_companion_state_already_returned(small_setup):
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec0 = checker1.uav_specs[0]
    spec1 = UAVSpec(name="uav_comp", speed_ms=10.0, battery_joules=300000.0,
                    p_flight=190.0, p_search=210.0, p_idle=50.0,
                    reserve_joules=30000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx, [spec0, spec1], H, delta_t=delta_t)

    # UAV 0 waits 4 ticks, detects at tick 5
    sched0 = UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=3, wait_ticks=4)])
    checker2.compute_and_update_chaining(sched0)
    t_det = sched0.visits[0].start_tick
    assert t_det == 5

    # Companion: tiny mission to cell 1 (fly 1 + dwell 1 + return 1 = done at tick 3)
    sched1 = UAVSchedule(1, [Visit(cell_idx=1, dwell_ticks=1, wait_ticks=0)])
    joint_sched = JointSchedule([sched0.clone(), sched1], num_cells, H, delta_t)

    feasible, reason = checker2.check_joint_schedule(joint_sched)
    assert feasible, f"Precondition: schedule must be fully feasible: {reason}"

    v1 = sched1.visits[0]
    done_tick = v1.start_tick + v1.dwell_ticks + int(checker2.tau_bar[1, 1, 0])
    assert done_tick < t_det, f"Precondition: companion done at tick {done_tick} < detection at {t_det}"

    target_path = np.full(H, 7, dtype=int)
    target_path[t_det] = 1
    rates = np.zeros(num_cells)
    rates[1] = 100.0
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res = sim.run_simulation(joint_sched, target_path, mission_idx=0,
                             rng_stream=IndexedRNGStream(master_seed=42))
    assert res.detected and res.detection_tick == t_det

    e_uav0 = (spec0.p_idle * 4 * delta_t
              + checker2.e_bar[0, 0, 1]
              + spec0.p_search * 1 * delta_t
              + checker2.e_bar[0, 1, 0])
    e_uav1_done = (checker2.e_bar[1, 0, 1]
                   + spec1.p_search * 1 * delta_t
                   + checker2.e_bar[1, 1, 0])
    assert np.isclose(res.actual_energy_joules, e_uav0 + e_uav1_done, rtol=0, atol=1e-6)


# ---------------------------------------------------------------
# Test 4c-6: ScheduleSimulator input validation contract
# ---------------------------------------------------------------
def test_schedule_simulator_rejects_infeasible_or_mismatched(small_setup):
    import pytest
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
    rates = np.zeros(num_cells)
    sim = ScheduleSimulator(checker, rates, delta_t=delta_t)
    target_path = np.zeros(H, dtype=int)

    # 1. Infeasible schedule (excessive dwell exceeding deadline)
    bad_sched = JointSchedule([UAVSchedule(0, [Visit(cell_idx=1, dwell_ticks=H + 5)])],
                              num_cells, H, delta_t)
    with pytest.raises(ValueError, match="Infeasible schedule rejected"):
        sim.run_simulation(bad_sched, target_path)

    # 2. Horizon mismatch
    mismatch_h = JointSchedule([UAVSchedule(0, [])], num_cells, H + 2, delta_t)
    with pytest.raises(ValueError, match="horizon_mismatch"):
        sim.run_simulation(mismatch_h, target_path)

    # 3. Delta_t mismatch
    mismatch_dt = JointSchedule([UAVSchedule(0, [])], num_cells, H, delta_t + 10.0)
    with pytest.raises(ValueError, match="delta_t_mismatch"):
        sim.run_simulation(mismatch_dt, target_path)

    # 4. Cell count mismatch
    mismatch_cells = JointSchedule([UAVSchedule(0, [])], num_cells + 5, H, delta_t)
    with pytest.raises(ValueError, match="num_cells_mismatch"):
        sim.run_simulation(mismatch_cells, target_path)

    # 5. Target path too short
    valid_sched = JointSchedule([UAVSchedule(0, [])], num_cells, H, delta_t)
    short_target = np.zeros(H - 2, dtype=int)
    with pytest.raises(ValueError, match="target_path length"):
        sim.run_simulation(valid_sched, short_target)

    # 6. Target path with cell indices out of range
    out_of_bounds_target = np.zeros(H, dtype=int)
    out_of_bounds_target[2] = num_cells + 3  # cell index out of range
    with pytest.raises(ValueError, match="target_path cell indices out of range"):
        sim.run_simulation(valid_sched, out_of_bounds_target)

    neg_target = np.zeros(H, dtype=int)
    neg_target[1] = -1
    with pytest.raises(ValueError, match="target_path cell indices out of range"):
        sim.run_simulation(valid_sched, neg_target)

    # 7. Target path with non-integer dtype
    float_target = np.zeros(H, dtype=float)
    with pytest.raises(ValueError, match="target_path must contain integer"):
        sim.run_simulation(valid_sched, float_target)

    # 8. Target path with 2D shape (e.g. coordinates instead of 1D cell indices)
    shape_2d_target = np.zeros((H, 2), dtype=int)
    with pytest.raises(ValueError, match="target_path must be a 1D array"):
        sim.run_simulation(valid_sched, shape_2d_target)

    # 9. Simulator with invalid delta_t
    with pytest.raises(ValueError, match="delta_t must be finite and positive"):
        ScheduleSimulator(checker, rates, delta_t=-5.0)


# ---------------------------------------------------------------
# Test 4c-7: Sensor detection rate input validation contract
# ---------------------------------------------------------------
def test_hypothesis_and_simulator_sensor_rate_validation(small_setup):
    import pytest
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    # 1. DetectionHypothesis with negative rate
    bad_rates_neg = np.ones(num_cells)
    bad_rates_neg[3] = -0.5
    with pytest.raises(ValueError, match="lambda_rates must be non-negative"):
        DetectionHypothesis(id=0, name="bad_neg", weight=1.0, lambda_rates=bad_rates_neg)

    # 2. DetectionHypothesis with NaN
    bad_rates_nan = np.ones(num_cells)
    bad_rates_nan[2] = np.nan
    with pytest.raises(ValueError, match="lambda_rates must be finite"):
        DetectionHypothesis(id=0, name="bad_nan", weight=1.0, lambda_rates=bad_rates_nan)

    # 3. DetectionHypothesis with Inf
    bad_rates_inf = np.ones(num_cells)
    bad_rates_inf[2] = np.inf
    with pytest.raises(ValueError, match="lambda_rates must be finite"):
        DetectionHypothesis(id=0, name="bad_inf", weight=1.0, lambda_rates=bad_rates_inf)

    # 4. DetectionHypothesis with 2D array
    with pytest.raises(ValueError, match="must be a 1D array"):
        DetectionHypothesis(id=0, name="bad_2d", weight=1.0, lambda_rates=np.ones((num_cells, 2)))

    # 5. ScheduleSimulator rejects invalid true_lambda_rates
    with pytest.raises(ValueError, match="true_lambda_rates must be non-negative"):
        ScheduleSimulator(checker, bad_rates_neg, delta_t=delta_t)

    with pytest.raises(ValueError, match="true_lambda_rates must be finite"):
        ScheduleSimulator(checker, bad_rates_nan, delta_t=delta_t)

    with pytest.raises(ValueError, match="length .* must match checker cell count"):
        ScheduleSimulator(checker, np.ones(num_cells + 2), delta_t=delta_t)


# ---------------------------------------------------------------
# Test 4c-8: JointMassEngine input validation contract
# ---------------------------------------------------------------
def test_joint_mass_engine_input_validation(small_setup):
    import pytest
    from sar_uav.belief.joint_mass import JointMassEngine
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    valid_b0 = np.ones(num_cells) / num_cells
    valid_M = np.eye(num_cells)

    # 1. Prior b0 2D array
    with pytest.raises(ValueError, match="Prior b0 must be a 1D array"):
        JointMassEngine(np.ones((num_cells, 2)), hset, valid_M, H, delta_t)

    # 2. Prior b0 empty
    with pytest.raises(ValueError, match="Prior b0 cannot be empty"):
        JointMassEngine(np.array([]), hset, valid_M, H, delta_t)

    # 3. Prior b0 NaN or Inf
    bad_b0_nan = valid_b0.copy()
    bad_b0_nan[0] = np.nan
    with pytest.raises(ValueError, match="Prior b0 must contain finite values"):
        JointMassEngine(bad_b0_nan, hset, valid_M, H, delta_t)

    # 4. Prior b0 length mismatch with hypothesis cell count
    with pytest.raises(ValueError, match="Prior b0 length .* does not match hypothesis cell count"):
        JointMassEngine(np.ones(num_cells + 4), hset, np.eye(num_cells + 4), H, delta_t)

    # 5. Horizon H <= 0 or not integer
    with pytest.raises(ValueError, match="Planning horizon H must be a positive integer"):
        JointMassEngine(valid_b0, hset, valid_M, 0, delta_t)

    with pytest.raises(ValueError, match="Planning horizon H must be a positive integer"):
        JointMassEngine(valid_b0, hset, valid_M, -3, delta_t)

    # 6. Delta_t <= 0 or non-finite
    with pytest.raises(ValueError, match="delta_t must be finite and positive"):
        JointMassEngine(valid_b0, hset, valid_M, H, delta_t=-10.0)

    with pytest.raises(ValueError, match="delta_t must be finite and positive"):
        JointMassEngine(valid_b0, hset, valid_M, H, delta_t=np.nan)


# ---------------------------------------------------------------
# Test 4c-9: Solver rejects NaN or out-of-bounds objective
# ---------------------------------------------------------------
def test_solver_rejects_nan_or_out_of_bounds_objective(small_setup, monkeypatch):
    import pytest
    from sar_uav.planning.local_search import JointRouteEffortLocalSearch
    from sar_uav.planning.evaluator import ForwardBackwardFastEvaluator
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    explorer = NeighborhoodExplorer(checker, candidates[:3], allowed_dwells=(1,), max_visits=2)
    evaluator = ForwardBackwardFastEvaluator(engine)
    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=10)
    init_sched = JointSchedule([UAVSchedule(0, [Visit(1, 1, 0)])], num_cells, H, delta_t)

    real_compute = engine.compute_forward_trajectory

    # 1. Final verification returns NaN
    call_count = [0]
    def mock_trajectory_nan(footprint):
        call_count[0] += 1
        res = real_compute(footprint)
        if call_count[0] > 1:
            return res[0], res[1], float('nan')
        return res

    monkeypatch.setattr(engine, "compute_forward_trajectory", mock_trajectory_nan)
    with pytest.raises(RuntimeError, match="Non-finite objective encountered in final schedule"):
        solver.solve(init_sched)

    # 2. Final verification returns out of bounds (1.5)
    call_count[0] = 0
    def mock_trajectory_oob(footprint):
        call_count[0] += 1
        res = real_compute(footprint)
        if call_count[0] > 1:
            return res[0], res[1], 1.5
        return res

    monkeypatch.setattr(engine, "compute_forward_trajectory", mock_trajectory_oob)
    with pytest.raises(RuntimeError, match="Objective out of probabilistic bounds"):
        solver.solve(init_sched)

    # 3. Initial schedule yields non-finite objective
    def mock_initial_nan(footprint):
        res = real_compute(footprint)
        return res[0], res[1], float('nan')

    monkeypatch.setattr(engine, "compute_forward_trajectory", mock_initial_nan)
    with pytest.raises(RuntimeError, match="Non-finite initial objective encountered"):
        solver.solve(init_sched)


# ---------------------------------------------------------------
# Test 5: Heuristic ≤ exact brute-force on the exact same domain.
# ---------------------------------------------------------------
def test_heuristic_not_exceeds_exact_on_same_domain(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    restricted_cells = candidates[:4]  # cells 1,2,3,4
    allowed_dwells = (1, 2)
    max_visits = 2

    exact = ExactBruteForcePlanner(checker, engine, restricted_cells,
                                    max_visits_per_uav=max_visits,
                                    allowed_dwells=allowed_dwells)
    exact_sched, J_exact, _ = exact.solve(num_uavs=1)
    assert J_exact > 0, "Exact solver found no feasible solution"

    # Execute Heuristic Local Search on the EXACT same restricted domain
    explorer = NeighborhoodExplorer(checker, restricted_cells, allowed_dwells=allowed_dwells,
                                    max_wait_ticks=0, max_visits=max_visits)
    evaluator = ForwardBackwardFastEvaluator(engine)
    init_sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=restricted_cells[0], dwell_ticks=allowed_dwells[0])])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    solver = JointRouteEffortLocalSearch(explorer, evaluator, max_evals=300, time_limit_sec=5.0)
    best_sched, J_heuristic, _ = solver.solve(init_sched)

    # Heuristic CANNOT exceed globally optimal J_exact on the same domain
    assert J_heuristic <= J_exact + 1e-9, \
        f"Heuristic J={J_heuristic:.8f} exceeded exact optimal J*={J_exact:.8f}"


# ---------------------------------------------------------------
# Test 6: Footprint generation reflects schedule mutations
# ---------------------------------------------------------------
def test_footprint_reflects_mutations(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup

    sched = JointSchedule(
        uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=2, dwell_ticks=2)])],
        num_cells=num_cells, H=H, delta_t=delta_t
    )
    checker.check_joint_schedule(sched)
    fp1 = sched.generate_footprint_array()
    start_t = sched.uav_schedules[0].visits[0].start_tick
    assert fp1[start_t, 2] == 1.0
    assert fp1[start_t, 5] == 0.0

    # Mutate cell directly on the same schedule object
    sched.uav_schedules[0].visits[0].cell_idx = 5
    checker.check_joint_schedule(sched)
    fp2 = sched.generate_footprint_array()
    new_start_t = sched.uav_schedules[0].visits[0].start_tick
    assert fp2[new_start_t, 5] == 1.0
    assert fp2[new_start_t, 2] == 0.0


# ---------------------------------------------------------------
# Test 7: Invariants and input validation
# ---------------------------------------------------------------
class TestInvariantsAndInputs:

    def test_zero_coverage_yields_exact_zero_detection(self, small_setup):
        """Invariant: Zero optical coverage implies exactly zero detection probability."""
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        empty_fp = np.zeros((H, num_cells), dtype=np.float64)
        _, u_plus, J_empty = engine.compute_forward_trajectory(empty_fp)
        assert J_empty == 0.0, f"Zero coverage must yield J=0.0, got {J_empty}"
        assert np.isclose(np.sum(u_plus[H - 1]), 1.0, rtol=0, atol=1e-12)

    def test_negative_dwell_rejected(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=-1)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Checker must reject negative dwell, got feasible: {reason}"

    def test_negative_wait_rejected(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2, wait_ticks=-2)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Checker must reject negative wait, got feasible: {reason}"

    def test_cell_index_out_of_range(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=999, dwell_ticks=2)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Checker must reject out-of-range cell, got feasible: {reason}"
        assert "cell_999_out_of_range" in reason

    def test_negative_weight_hypothesis_raises(self):
        with pytest.raises(ValueError, match="Hypothesis weights must be non-negative"):
            HypothesisSet([
                DetectionHypothesis(id=0, name="neg", weight=-1.0,
                                    lambda_rates=np.array([0.1, 0.2]))
            ])

    def test_zero_prior_sum_raises(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        b0_zero = np.zeros(num_cells)
        with pytest.raises(ValueError, match="Prior b0 must have positive, finite sum"):
            JointMassEngine(b0_zero, hset, np.eye(num_cells), H, delta_t=delta_t)

    def test_non_stochastic_motion_matrix_raises(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        M_bad = np.ones((num_cells, num_cells))  # rows sum to num_cells != 1
        b0 = np.full(num_cells, 1.0 / num_cells)
        with pytest.raises(ValueError, match="Motion matrix M must be row-stochastic"):
            JointMassEngine(b0, hset, M_bad, H, delta_t=delta_t)

    def test_motion_matrix_row_stochastic_tolerance_strict(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        b0 = np.full(num_cells, 1.0 / num_cells)
        M_slightly_off = np.eye(num_cells)
        M_slightly_off[0, 0] = 1.0005  # row 0 sum = 1.0005, beyond 1e-5
        with pytest.raises(ValueError, match="Motion matrix M must be row-stochastic"):
            JointMassEngine(b0, hset, M_slightly_off, H, delta_t=delta_t)

    def test_duplicate_uav_id_rejected(self, small_setup):
        """Two UAVSchedules with the same uav_idx must be rejected."""
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        spec2 = UAVSpec(speed_ms=10.0, battery_joules=200000.0, reserve_joules=25000.0, delta_t=delta_t)
        checker2 = ConstraintChecker(checker.dist_matrix, checker.depot_idx,
                                     [checker.uav_specs[0], spec2], H, delta_t=delta_t)
        sched = JointSchedule(
            uav_schedules=[
                UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=2)]),
                UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=2, dwell_ticks=2)]),
            ],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker2.check_joint_schedule(sched)
        assert not feasible, f"Must reject duplicate UAV ID, got: {reason}"
        assert "duplicate_uav_idx_0" in reason

    def test_uav_id_negative_rejected(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=-1, visits=[Visit(cell_idx=1, dwell_ticks=2)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Must reject negative UAV ID, got: {reason}"
        assert "out_of_range" in reason

    def test_uav_id_out_of_range_rejected(self, small_setup):
        """UAV idx=5 when only 1 UAV spec exists."""
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=5, visits=[Visit(cell_idx=1, dwell_ticks=2)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Must reject UAV idx=5 with 1 spec, got: {reason}"
        assert "out_of_range" in reason

    def test_negative_cell_index_rejected(self, small_setup):
        checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
        sched = JointSchedule(
            uav_schedules=[UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=-1, dwell_ticks=2)])],
            num_cells=num_cells, H=H, delta_t=delta_t
        )
        feasible, reason = checker.check_joint_schedule(sched)
        assert not feasible, f"Must reject negative cell index, got: {reason}"
        assert "cell_-1_out_of_range" in reason


# ---------------------------------------------------------------
# Test 8: Reordering UAV schedules yields same footprint and simulation result
# ---------------------------------------------------------------
def test_reorder_uav_schedules_same_result(small_setup):
    """Footprint and simulation results must not depend on UAV ordering in the list."""
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec2 = UAVSpec(speed_ms=10.0, battery_joules=200000.0, reserve_joules=25000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx,
                                 [checker1.uav_specs[0], spec2], H, delta_t=delta_t)

    s0 = UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=2)])
    s1 = UAVSchedule(uav_idx=1, visits=[Visit(cell_idx=4, dwell_ticks=3)])

    joint_ab = JointSchedule([s0.clone(), s1.clone()], num_cells, H, delta_t)
    joint_ba = JointSchedule([s1.clone(), s0.clone()], num_cells, H, delta_t)

    checker2.check_joint_schedule(joint_ab)
    checker2.check_joint_schedule(joint_ba)

    fp_ab = joint_ab.generate_footprint_array()
    fp_ba = joint_ba.generate_footprint_array()
    np.testing.assert_array_equal(fp_ab, fp_ba, err_msg="Footprints must be equal regardless of UAV order")

    # Simulation
    target_path = np.full(H, 7, dtype=int)
    rates = np.full(num_cells, 0.15)
    sim = ScheduleSimulator(checker2, rates, delta_t=delta_t)

    res_ab = sim.run_simulation(joint_ab, target_path, mission_idx=0, rng_stream=IndexedRNGStream(master_seed=42))
    res_ba = sim.run_simulation(joint_ba, target_path, mission_idx=0, rng_stream=IndexedRNGStream(master_seed=42))
    assert np.isclose(res_ab.actual_energy_joules, res_ba.actual_energy_joules, rtol=0, atol=1e-6), \
        f"Energy must match: {res_ab.actual_energy_joules} vs {res_ba.actual_energy_joules}"


# ---------------------------------------------------------------
# Test 9: Partial fleet — only one of two UAVs has visits
# ---------------------------------------------------------------
def test_partial_fleet_simulation(small_setup):
    """UAV with r_k=0 contributes zero energy; active UAV works normally."""
    checker1, engine, candidates, H, delta_t, num_cells, hset = small_setup
    spec2 = UAVSpec(speed_ms=10.0, battery_joules=200000.0, reserve_joules=25000.0, delta_t=delta_t)
    checker2 = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx,
                                 [checker1.uav_specs[0], spec2], H, delta_t=delta_t)

    s_active = UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2)])
    s_idle = UAVSchedule(uav_idx=1, visits=[])  # r_k = 0

    joint = JointSchedule([s_active, s_idle], num_cells, H, delta_t)
    feasible, reason = checker2.check_joint_schedule(joint)
    assert feasible, f"Partial fleet should be feasible: {reason}"

    # Energy should equal single-UAV energy (idle UAV contributes 0)
    checker_solo = ConstraintChecker(checker1.dist_matrix, checker1.depot_idx,
                                     [checker1.uav_specs[0]], H, delta_t=delta_t)
    s_solo = UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=4, dwell_ticks=2)])
    joint_solo = JointSchedule([s_solo], num_cells, H, delta_t)
    checker_solo.check_joint_schedule(joint_solo)

    target_path = np.full(H, 7, dtype=int)
    rates = np.full(num_cells, 0.15)

    sim_duo = ScheduleSimulator(checker2, rates, delta_t=delta_t)
    sim_solo = ScheduleSimulator(checker_solo, rates, delta_t=delta_t)

    res_duo = sim_duo.run_simulation(joint, target_path, mission_idx=0, rng_stream=IndexedRNGStream(master_seed=42))
    res_solo = sim_solo.run_simulation(joint_solo, target_path, mission_idx=0, rng_stream=IndexedRNGStream(master_seed=42))

    assert np.isclose(res_duo.actual_energy_joules, res_solo.actual_energy_joules, rtol=0, atol=1e-6), \
        f"Idle UAV must add zero energy: {res_duo.actual_energy_joules} vs {res_solo.actual_energy_joules}"


# ---------------------------------------------------------------
# Test 10: Target path must be 1D array of integers
# ---------------------------------------------------------------
def test_target_path_2d_raises_error(small_setup):
    """Simulator must reject 2D target path (e.g. shape (H, 2)) immediately with ValueError."""
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
    rates = np.full(num_cells, 0.15)
    sim = ScheduleSimulator(checker, rates, delta_t=delta_t)

    s0 = UAVSchedule(uav_idx=0, visits=[Visit(cell_idx=1, dwell_ticks=2)])
    joint = JointSchedule([s0], num_cells, H, delta_t)

    target_path_2d = np.zeros((H, 2), dtype=int)
    with pytest.raises(ValueError, match="target_path must be a 1D array"):
        sim.run_simulation(joint, target_path_2d)


# ---------------------------------------------------------------
# Test 11: JointMassEngine constructor parameter validation
# ---------------------------------------------------------------
def test_joint_mass_engine_comprehensive_validation(small_setup):
    checker, engine, candidates, H, delta_t, num_cells, hset = small_setup
    M = np.eye(num_cells)

    # 1. 2D b0 must raise ValueError
    b0_2d = np.ones((num_cells, 1)) / num_cells
    with pytest.raises(ValueError, match="Prior b0 must be a 1D array"):
        JointMassEngine(b0_2d, hset, M, H, delta_t=delta_t)

    # 2. b0 length mismatch with hypothesis
    b0_wrong_len = np.ones(num_cells + 2) / (num_cells + 2)
    with pytest.raises(ValueError, match="does not match hypothesis cell count"):
        JointMassEngine(b0_wrong_len, hset, np.eye(num_cells + 2), H, delta_t=delta_t)

    # 3. Horizon H must be positive integer
    b0_valid = np.ones(num_cells) / num_cells
    with pytest.raises(ValueError, match="Planning horizon H must be a positive integer"):
        JointMassEngine(b0_valid, hset, M, H=0, delta_t=delta_t)

    with pytest.raises(ValueError, match="Planning horizon H must be a positive integer"):
        JointMassEngine(b0_valid, hset, M, H=-5, delta_t=delta_t)

    with pytest.raises(ValueError, match="Planning horizon H must be a positive integer"):
        JointMassEngine(b0_valid, hset, M, H=12.5, delta_t=delta_t)  # float

    # 4. delta_t must be finite positive float
    with pytest.raises(ValueError, match="delta_t must be finite and positive"):
        JointMassEngine(b0_valid, hset, M, H, delta_t=-10.0)

    with pytest.raises(ValueError, match="delta_t must be finite and positive"):
        JointMassEngine(b0_valid, hset, M, H, delta_t=float("nan"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
