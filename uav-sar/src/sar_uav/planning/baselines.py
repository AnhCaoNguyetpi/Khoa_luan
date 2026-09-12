"""Baseline algorithms for algorithmic ablation and comparative benchmarking.

Corresponds to proposal Table 2:
    - Greedy / Myopic Lookahead Heuristic (instant probability maximization)
    - Adaptive GA for UAV SAR (metaheuristic benchmark)
    - Exact Solver / Brute-force (for small 3x3 grids to compute J*)
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from sar_uav.belief.joint_mass import JointMassEngine
from sar_uav.planning.constraints import ConstraintChecker, JointSchedule, UAVSchedule, Visit
from sar_uav.planning.evaluator import FullEvaluator


class GreedyLookaheadPlanner:
    """Myopic / Greedy Search SAR baseline.
    Greedily inserts the visit yielding maximum immediate marginal detection gain.
    """

    def __init__(
        self,
        checker: ConstraintChecker,
        engine: JointMassEngine,
        candidate_cells: Sequence[int],
        default_dwell: int = 2
    ):
        self.checker = checker
        self.engine = engine
        self.candidate_cells = list(candidate_cells)
        self.default_dwell = default_dwell

    def solve(self, num_uavs: int) -> Tuple[JointSchedule, float, Dict]:
        t0 = time.perf_counter()
        uav_scheds = [UAVSchedule(uav_idx=k, visits=[]) for k in range(num_uavs)]
        current_sched = JointSchedule(
            uav_schedules=uav_scheds,
            num_cells=self.engine.num_cells,
            H=self.engine.H,
            delta_t=self.engine.delta_t
        )

        _, _, current_J = self.engine.compute_forward_trajectory(current_sched.generate_footprint_array())
        n_evals = 1

        # Iteratively append best feasible visit across all UAVs
        while True:
            best_gain = 0.0
            best_cand: Optional[JointSchedule] = None

            for k in range(num_uavs):
                for cell in self.candidate_cells:
                    cand = current_sched.clone()
                    cand.uav_schedules[k].visits.append(Visit(cell_idx=cell, dwell_ticks=self.default_dwell))
                    
                    feasible, _ = self.checker.check_joint_schedule(cand)
                    if feasible:
                        cand_fp = cand.generate_footprint_array()
                        _, _, new_J = self.engine.compute_forward_trajectory(cand_fp)
                        n_evals += 1
                        gain = new_J - current_J
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand

            if best_gain > 1e-6 and best_cand is not None:
                current_sched = best_cand
                current_J += best_gain
            else:
                break

        elapsed = time.perf_counter() - t0
        return current_sched, current_J, {"n_evals": n_evals, "runtime_sec": elapsed, "method": "Greedy"}


class SimpleEvolutionaryPlanner:
    """Standard Genetic Algorithm (GA) baseline for UAV SAR routing and dwell allocation.

    Implements:
      - Tournament selection (tournament size 2)
      - 1-point crossover on per-UAV visit sequences (p_cross = 0.7)
      - Fixed-probability uniform mutation:
          * Cell replacement (p_mut = 0.3)
          * Dwell duration adjustment (p_mut = 0.3)
      - Feasibility repair: truncates visits from schedule tail until all safety/energy constraints pass
      - 1-individual elitist preservation across generations
    Note: Crossover and mutation probabilities are constant (non-adaptive).
    """

    def __init__(
        self,
        checker: ConstraintChecker,
        engine: JointMassEngine,
        candidate_cells: Sequence[int],
        population_size: int = 30,
        max_generations: int = 50,
        allowed_dwells: Sequence[int] = (1, 2, 3, 4),
        seed: int = 42
    ):
        self.checker = checker
        self.engine = engine
        self.candidate_cells = list(candidate_cells)
        self.pop_size = population_size
        self.max_gen = max_generations
        self.allowed_dwells = list(allowed_dwells)
        self.rng = np.random.RandomState(seed)

    def _random_schedule(self, num_uavs: int) -> JointSchedule:
        uav_scheds = []
        for k in range(num_uavs):
            num_v = self.rng.randint(1, 4)
            chosen_cells = self.rng.choice(self.candidate_cells, size=num_v, replace=True)
            visits = [Visit(cell_idx=int(c), dwell_ticks=int(self.rng.choice(self.allowed_dwells)))
                      for c in chosen_cells]
            uav_scheds.append(UAVSchedule(uav_idx=k, visits=visits))
        
        sched = JointSchedule(uav_schedules=uav_scheds, num_cells=self.engine.num_cells,
                              H=self.engine.H, delta_t=self.engine.delta_t)
        return self._repair(sched)

    def _repair(self, sched: JointSchedule) -> JointSchedule:
        """Trims visits from tail if infeasible until feasible."""
        for u_sched in sched.uav_schedules:
            while u_sched.r_k > 0:
                feasible, _ = self.checker.check_uav_feasibility(u_sched)
                if feasible:
                    break
                u_sched.visits.pop()
        # Compute proper chaining across the joint schedule
        self.checker.check_joint_schedule(sched)
        return sched

    def _crossover(self, p1: JointSchedule, p2: JointSchedule) -> JointSchedule:
        """1-point crossover on per-UAV visit sequences."""
        num_uavs = len(p1.uav_schedules)
        child_uavs = []
        for k in range(num_uavs):
            v1 = p1.uav_schedules[k].visits
            v2 = p2.uav_schedules[k].visits
            if len(v1) > 1 and len(v2) > 1 and self.rng.rand() < 0.7:
                cut1 = self.rng.randint(1, len(v1))
                cut2 = self.rng.randint(1, len(v2))
                crossed = [v.clone() for v in v1[:cut1]] + [v.clone() for v in v2[cut2:]]
                child_uavs.append(UAVSchedule(uav_idx=k, visits=crossed))
            else:
                chosen = p1 if self.rng.rand() < 0.5 else p2
                child_uavs.append(chosen.uav_schedules[k].clone())
        child = JointSchedule(uav_schedules=child_uavs, num_cells=self.engine.num_cells,
                              H=self.engine.H, delta_t=self.engine.delta_t)
        return self._repair(child)

    def solve(self, num_uavs: int) -> Tuple[JointSchedule, float, Dict]:
        t0 = time.perf_counter()
        population = [self._random_schedule(num_uavs) for _ in range(self.pop_size)]
        fitnesses = []
        n_evals = 0

        for ind in population:
            fp = ind.generate_footprint_array()
            _, _, J = self.engine.compute_forward_trajectory(fp)
            fitnesses.append(J)
            n_evals += 1

        best_idx = int(np.argmax(fitnesses))
        best_sched = population[best_idx].clone()
        best_J = fitnesses[best_idx]

        for gen in range(self.max_gen):
            # Elitism: retain best individual
            new_pop = [best_sched.clone()]
            while len(new_pop) < self.pop_size:
                # Tournament selection for parent 1 and 2
                cand1 = self.rng.choice(self.pop_size, size=2, replace=False)
                p1_idx = cand1[0] if fitnesses[cand1[0]] >= fitnesses[cand1[1]] else cand1[1]
                cand2 = self.rng.choice(self.pop_size, size=2, replace=False)
                p2_idx = cand2[0] if fitnesses[cand2[0]] >= fitnesses[cand2[1]] else cand2[1]

                child = self._crossover(population[p1_idx], population[p2_idx])

                # Mutation
                for u_sched in child.uav_schedules:
                    if u_sched.r_k > 0 and self.rng.rand() < 0.3:
                        idx = self.rng.randint(0, u_sched.r_k)
                        u_sched.visits[idx].cell_idx = int(self.rng.choice(self.candidate_cells))
                    if u_sched.r_k > 0 and self.rng.rand() < 0.3:
                        idx = self.rng.randint(0, u_sched.r_k)
                        u_sched.visits[idx].dwell_ticks = int(self.rng.choice(self.allowed_dwells))

                child = self._repair(child)
                new_pop.append(child)

            population = new_pop
            fitnesses = []
            for ind in population:
                fp = ind.generate_footprint_array()
                _, _, J = self.engine.compute_forward_trajectory(fp)
                fitnesses.append(J)
                n_evals += 1

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_J:
                best_J = fitnesses[gen_best_idx]
                best_sched = population[gen_best_idx].clone()

        elapsed = time.perf_counter() - t0
        return best_sched, best_J, {"n_evals": n_evals, "runtime_sec": elapsed, "method": "SimpleEvolutionary"}


# Aliases
GABaselinePlanner = SimpleEvolutionaryPlanner
AdaptiveGAPlanner = SimpleEvolutionaryPlanner  # backward compatibility alias


class ExactBruteForcePlanner:
    """Exact enumeration for small problem instances to find globally optimal J*.

    Note: This enumerates within a *restricted domain* (max_visits_per_uav,
    allowed_dwells, no wait). The result is optimal within that domain but
    not necessarily globally optimal if the heuristic searches a larger space.
    """

    def __init__(
        self,
        checker: ConstraintChecker,
        engine: JointMassEngine,
        candidate_cells: Sequence[int],
        max_visits_per_uav: int = 3,
        allowed_dwells: Sequence[int] = (1, 2)
    ):
        self.checker = checker
        self.engine = engine
        self.candidate_cells = list(candidate_cells)
        self.max_visits = max_visits_per_uav
        self.allowed_dwells = list(allowed_dwells)

    def _enumerate_feasible_chains(self, uav_idx: int) -> List[List[Visit]]:
        """Enumerate all feasible visit chains for a specific UAV."""
        chains = [[]]  # empty chain is always feasible
        for length in range(1, self.max_visits + 1):
            for cell_tuple in itertools.product(self.candidate_cells, repeat=length):
                for dwell_tuple in itertools.product(self.allowed_dwells, repeat=length):
                    chain = [Visit(cell_idx=c, dwell_ticks=d) for c, d in zip(cell_tuple, dwell_tuple)]
                    test_uav = UAVSchedule(uav_idx=uav_idx, visits=chain)
                    feasible, _ = self.checker.check_uav_feasibility(test_uav)
                    if feasible:
                        chains.append(chain)
        return chains

    def solve(self, num_uavs: int = 1) -> Tuple[JointSchedule, float, Dict]:
        t0 = time.perf_counter()
        best_J = -1.0
        best_sched: Optional[JointSchedule] = None
        n_evals = 0

        # P0.2: Generate feasible chains PER UAV (using correct uav_idx)
        per_uav_chains = [self._enumerate_feasible_chains(k) for k in range(num_uavs)]

        # Combine across UAVs
        for combo in itertools.product(*per_uav_chains):
            uav_scheds = [
                UAVSchedule(
                    uav_idx=k,
                    visits=[Visit(cell_idx=v.cell_idx, dwell_ticks=v.dwell_ticks,
                                  wait_ticks=v.wait_ticks) for v in combo[k]]
                )
                for k in range(num_uavs)
            ]
            joint = JointSchedule(
                uav_schedules=uav_scheds,
                num_cells=self.engine.num_cells,
                H=self.engine.H,
                delta_t=self.engine.delta_t
            )

            # P0.2: Chain and validate the assembled joint schedule
            # This computes correct start_tick values from chaining equations
            # before generating footprints for scoring.
            feasible, _ = self.checker.check_joint_schedule(joint)
            if not feasible:
                continue

            fp = joint.generate_footprint_array()
            _, _, J = self.engine.compute_forward_trajectory(fp)
            n_evals += 1
            if J > best_J:
                best_J = J
                best_sched = joint.clone()

        elapsed = time.perf_counter() - t0
        return best_sched, best_J, {"n_evals": n_evals, "runtime_sec": elapsed, "method": "ExactBruteForce"}
