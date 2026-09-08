"""Mission simulation loop (proposal section "Quy trinh mo phong tong the").

    P^t -> assignment -> routing -> search -> observation -> Bayesian update
        -> P^{t+1} -> re-optimise

``MissionSetup`` bundles everything that must be *shared across strategy
arms* of one replicate (common random numbers): area, weather, IPP, hidden
target trajectory.  ``run_mission`` then executes one strategy against that
shared environment.
"""

from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..belief.models import build_initial_belief
from ..belief.motion import MotionModel, perturb_betas, PROFILES
from ..belief.update import bayes_negative_update
from ..config import DEFAULT_CONFIG, deep_update
from ..data.area import AreaData, get_or_build_area
from ..data.weather import WeatherSeries, load_weather, make_weather
from ..detection.sensors import q_tick, q_dwell
from ..planning import make_planner, PlanningProblem
from ..uav.platform import UAVSpec
from ..uav.state import UAVState
from ..utils.seed import spawn_rngs
from .metrics import MissionResult
from .target import LostPersonSim, sample_profile

log = logging.getLogger(__name__)


def resolve_belief_model_file(belief_cfg: Dict, area_name: str) -> Optional[str]:
    mf = belief_cfg.get("model_file")
    if mf:
        return mf
    default = Path(__file__).resolve().parents[3] / "data" / "models" \
        / f"{area_name}_pmr.npz"
    return str(default) if default.exists() else None


class MissionSetup:
    """Environment shared by all arms of one replicate (CRN)."""

    def __init__(self, cfg: Dict, area: AreaData, weather: WeatherSeries,
                 ipp_idx: int, profile: str, burn_min: int,
                 target_path: np.ndarray, seed: int):
        self.cfg = cfg
        self.area = area
        self.weather = weather
        self.ipp_idx = int(ipp_idx)
        self.profile = profile
        self.burn_min = int(burn_min)
        self.target_path = target_path          # absolute-tick indexed
        self.seed = int(seed)

    @classmethod
    def create(cls, cfg: Dict, seed: int, area: Optional[AreaData] = None,
               ipp_idx: Optional[int] = None,
               profile: Optional[str] = None,
               moving: Optional[bool] = None) -> "MissionSetup":
        mcfg = deep_update(DEFAULT_CONFIG, cfg)
        rngs = spawn_rngs(seed, 2)
        rng_env, rng_target = rngs

        if area is None:
            area = get_or_build_area(mcfg["area"]["name"],
                                     Path(__file__).resolve().parents[3] / "data",
                                     mcfg)

        horizon = float(mcfg["mission"]["horizon_min"])
        dt = float(mcfg["mission"]["dt_min"])
        burn_lo, burn_hi = mcfg["target"].get("burn_min_range", [45, 120])
        burn_max = int(max(burn_lo, burn_hi))
        total_min = horizon + burn_max + 2 * dt
        weather_file = Path(__file__).resolve().parents[3] / "data" / "areas" \
            / area.name / "weather.npz"
        if weather_file.exists():
            weather = load_weather(weather_file)
            if len(weather) * weather.dt_min < total_min:
                raise ValueError(f"weather series too short for mission: {weather_file}")
        else:
            weather = make_weather(total_min, dt, rng_env, mcfg["weather"])
        rain_fn = lambda tick: float(weather.rain[min(int(tick), len(weather) - 1)])

        burn = int(rng_env.integers(int(burn_lo), int(burn_hi) + 1))
        prof = profile or sample_profile(rng_target,
                                         mcfg["target"]["profile_probs"])

        if ipp_idx is None:
            trail_cells = np.flatnonzero(area.trail_mask.reshape(-1))
            pool = trail_cells if len(trail_cells) > 4 else area.land_idx()
            ipp = int(rng_env.choice(pool))
        else:
            ipp = int(ipp_idx)

        sim = LostPersonSim(area, prof, ipp, rng_target,
                            moving=bool(mcfg["target"]["moving"]) if moving is None
                            else bool(moving))
        path = sim.full_path(burn, horizon, rain_fn)

        return cls(mcfg, area, weather, ipp, prof, burn, path, seed)

    # ------------------------------------------------------------------
    def rain_at(self, abs_tick: int) -> float:
        return float(self.weather.rain[min(int(abs_tick), len(self.weather) - 1)])


# ---------------------------------------------------------------------------

def _auto_bases(area, fleet: List[UAVSpec]) -> List[np.ndarray]:
    """Place bases near distinct map corners (trailhead depots)."""
    H, W = area.shape
    fracs = [(0.12, 0.12), (0.88, 0.88), (0.12, 0.88),
             (0.88, 0.12), (0.5, 0.5)]
    bases = []
    land = area.land_idx()
    xy = area.centers_xy()
    for k in range(len(fleet)):
        fx, fy = fracs[k % len(fracs)]
        target = np.array([fx * W * area.cell, fy * H * area.cell])
        d = np.hypot(xy[land][:, 0] - target[0], xy[land][:, 1] - target[1])
        bases.append(xy[land[int(np.argmin(d))]])
    return bases


def run_mission(setup: MissionSetup,
                planner_cfg: Optional[Dict] = None,
                initial_model: Optional[str] = None,
                belief_override: Optional[np.ndarray] = None,
                q_est_bias: float = 1.0,
                beta_est_bias: float = 0.0,
                forecast_enabled: Optional[bool] = None,
                arm_name: str = "",
                replicate: int = -1,
                render_dir: Optional[str] = None,
                render_every: int = 25,
                obs_seed: Optional[int] = None,
                collect_trace: bool = False,
                assign_criterion: str = "pq",
                fleet_cfg: Optional[List[Dict]] = None,
                moving_target: Optional[bool] = None) -> MissionResult:
    """Execute one SAR mission of a given strategy against ``setup``."""
    cfg = setup.cfg
    area = setup.area
    dt = float(cfg["mission"]["dt_min"])
    horizon = float(cfg["mission"]["horizon_min"])
    dwell = int(cfg["detection"]["dwell_ticks"])
    fp_side = float(cfg["detection"].get("footprint_side", 0.45))
    moving = bool(cfg["target"]["moving"]) if moving_target is None \
        else bool(moving_target)
    if forecast_enabled is None:
        forecast_enabled = bool(cfg["belief"]["forecast_enabled"]) and moving

    # ---- fleet & states -------------------------------------------------
    fleet_src = fleet_cfg if fleet_cfg is not None else cfg["fleet"]
    fleet = [UAVSpec(**{k: v for k, v in c.items()
                        if k in UAVSpec.__dataclass_fields__})
             for c in fleet_src]
    bases_xy = _auto_bases(area, fleet)
    states: Dict[str, UAVState] = {}
    for spec, bxy in zip(fleet, bases_xy):
        if spec.base_cell is not None:
            bxy = area.centers_xy()[spec.base_cell]
        states[spec.name] = UAVState(spec, area, bxy)

    # ---- planner's information world ------------------------------------
    true_betas = PROFILES[setup.profile]
    est_betas = perturb_betas(true_betas, float(beta_est_bias))
    motion_est = MotionModel(area, est_betas)

    def weather_fn(t_rel: float) -> Dict[str, float]:
        return setup.weather.window_mean(int(setup.burn_min + t_rel), dwell)

    problem = PlanningProblem(area, fleet, motion_est, dwell_ticks=dwell,
                              weather_fn=weather_fn, dt_min=dt)
    if abs(q_est_bias - 1.0) > 1e-9:
        # wrap q_matrix to apply the RQ5 estimation bias
        base_qm = problem.q_matrix

        def biased_qm(t, _b=float(q_est_bias)):
            return np.clip(_b * base_qm(t), 0.0, 0.999)

        problem.q_matrix = biased_qm            # type: ignore[method-assign]

    if assign_criterion == "p":
        # RQ3 ablation: assignment ignores detectability (uses p_i^t only);
        # observations/Bayes updates still use the real detection model
        K = len(fleet)

        def p_only_qm(t):
            return np.ones((K, problem.n))

        problem.q_matrix = p_only_qm            # type: ignore[method-assign]

    p_cfg = deep_update(cfg["planner"], planner_cfg or {})
    planner = make_planner(p_cfg, problem)

    # ---- initial belief --------------------------------------------------
    if belief_override is not None:
        belief = np.asarray(belief_override, dtype=float).reshape(-1).copy()
    else:
        model_name = initial_model or cfg["belief"]["initial_model"]
        bcfg = dict(cfg["belief"])
        if model_name == "datadriven":
            mf = resolve_belief_model_file(bcfg, area.name)
            if mf is None:
                raise ValueError(
                    "No trained PMR weights found -- run "
                    "scripts/train_belief_model.py first, or pick another "
                    "initial_model.")
            bcfg["model_file"] = mf
        belief = build_initial_belief(model_name, area, setup.ipp_idx,
                                      float(setup.burn_min), bcfg)

    # ---- observation RNG -------------------------------------------------
    if obs_seed is None:
        obs_seed = setup.seed * 1000 + zlib.crc32(arm_name.encode()) % 1000
    obs_rng = np.random.default_rng(obs_seed)

    veg_flat = np.clip(area.veg.reshape(-1), 0, 1)

    # ---- main loop -------------------------------------------------------
    detected = False
    detect_t = float("nan")
    n_replans, replan_ms = 0, 0.0
    idle_streak = 0
    frames: List[Dict] = []
    trace: List[Dict] = [] if collect_trace else None
    t = 0.0

    while t < horizon - 1e-9:
        tabs = int(round(setup.burn_min + t))

        # 1) belief forecast under the planner's motion model
        if forecast_enabled and t > 0:
            belief = motion_est.forecast(belief, setup.rain_at(tabs))

        # 2) planning (only ever sees the belief)
        t0 = time.perf_counter()
        routes = planner.decide(states, belief, t, horizon_left=horizon - t)
        replan_ms += (time.perf_counter() - t0) * 1000.0
        if routes:
            n_replans += 1
            idle_streak = 0
            for name, cycles in routes.items():
                try:
                    states[name].assign_cycles(cycles)
                except RuntimeError:
                    pass                       # became busy mid-decision: skip

        # 3) tick dynamics + observations
        any_searching = False
        for st in states.values():
            ev = st.tick(dt)
            if "searching" in ev:
                any_searching = True
                cell = int(ev["searching"])
                # sensor footprint: centre cell at full q, neighbours attenuated
                fp_cells = [cell] + area.neighbor_idx(cell)
                true_cell = int(setup.target_path[tabs])
                q_true_tick = q_tick(st.spec.sensor, veg_flat[cell],
                                     setup.weather.cloud[tabs],
                                     setup.weather.rain[tabs],
                                     setup.weather.wind_ms[tabs])
                p_hit = 0.0
                for fc in fp_cells:
                    if fc == true_cell:
                        p_hit += q_true_tick * (1.0 if fc == cell else fp_side)
                if p_hit > 0 and obs_rng.random() < min(p_hit, 0.999):
                    detected = True
                    detect_t = min(t + dt, horizon)
                    break
            if "search_done" in ev and not detected:
                cell = int(ev["search_done"])
                wmean = setup.weather.window_mean(tabs - dwell + 1, dwell)
                q_cen = q_dwell(st.spec.sensor, veg_flat[cell], wmean, dwell)
                q_cen *= float(q_est_bias)
                fp_cells = [cell] + area.neighbor_idx(cell)
                qs = [q_cen] + [q_cen * fp_side] * (len(fp_cells) - 1)
                belief = bayes_negative_update(belief, fp_cells, qs)
                if collect_trace:
                    present = sum(int(setup.target_path[max(tabs - dt_i, 0)]
                                      in set(fp_cells))
                                  for dt_i in range(int(dwell)))
                    order = np.argsort(-belief)
                    trace.append({
                        "t": t, "uav": st.spec.name, "cell": cell,
                        "q_est": q_cen, "target_present_ticks": int(present),
                        "belief_at_cell": float(belief[cell]),
                        "true_rank_after": int(np.where(order == cell)[0][0]),
                    })
        if detected:
            break

        # 4) early stop when nothing is happening anymore
        all_idle = all((not st.is_active) or
                       (st.is_idle and not st.has_pending_work())
                       for st in states.values())
        if all_idle and not any_searching:
            idle_streak += 1
            if idle_streak >= 3:
                break
        else:
            idle_streak = 0

        if render_dir and int(t) % max(render_every, 1) == 0:
            frames.append({
                "t": t, "belief": belief.copy(),
                "uav_xy": {n: s.pos.copy() for n, s in states.items()},
                "target_cell": int(setup.target_path[tabs]),
            })
        t += dt

    # ---- result ----------------------------------------------------------
    res = MissionResult(detected=detected, detect_time_min=detect_t,
                        horizon_min=horizon)
    dist = sum(s.dist_flown for s in states.values())
    energy = sum(s.energy_spent for s in states.values())
    searched = {}
    for s in states.values():
        for c, nvis in s.visits.items():
            searched[c] = searched.get(c, 0) + nvis
    ops = int(sum(searched.values()))
    uniq = len(searched)
    res.flight_distance_m = float(dist)
    res.energy_wh = float(energy)
    res.search_minutes = float(sum(s.search_minutes for s in states.values()))
    res.unique_cells_searched = uniq
    res.total_search_ops = ops
    res.redundant_search_ops = max(ops - uniq, 0)
    res.n_swaps = int(sum(s.n_swaps for s in states.values()))
    res.n_replans = n_replans
    res.replan_ms_total = float(replan_ms)
    res.planner = p_cfg.get("kind", "")
    res.initial_model = initial_model or cfg["belief"]["initial_model"] \
        if belief_override is None else "custom"
    res.area_name = area.name
    res.seed = setup.seed
    res.replicate = replicate
    res.arm = arm_name
    res.target_profile = setup.profile
    res.elapsed_before_min = float(setup.burn_min)
    res.n_uavs = len(fleet)
    res.moving_target = moving
    res.per_uav = {n: {"dist": s.dist_flown, "energy": s.energy_spent,
                       "swaps": s.n_swaps, "visits": dict(s.visits)}
                   for n, s in states.items()}
    res.trace = trace

    if render_dir:
        frames.append({"t": t, "belief": belief.copy(),
                       "uav_xy": {n: s.pos.copy() for n, s in states.items()},
                       "target_cell": int(setup.target_path[tabs])})
        _render_frames(setup, states, res, frames, render_dir)

    return res


# ---------------------------------------------------------------------------
def _render_frames(setup, states, res, frames, render_dir):
    try:
        from ..viz.maps import render_mission_frame
    except ImportError:                                  # pragma: no cover
        return
    out = Path(render_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{res.arm or res.planner}_seed{setup.seed}"
    for i, fr in enumerate(frames):
        render_mission_frame(
            setup.area, fr["belief"],
            uav_xy=fr["uav_xy"],
            target_cells=[int(c) for c in
                          setup.target_path[:int(setup.burn_min + fr["t"]) + 1]],
            ipp_cell=setup.ipp_idx,
            bases={n: s.base_xy for n, s in states.items()},
            title=f"{res.arm} t={fr['t']:.0f} min detected={res.detected}",
            save=out / f"{tag}_t{int(fr['t']):04d}.png")
