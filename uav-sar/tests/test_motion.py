"""Motion model: stochastic matrix, forecasting, sampling geometry."""
from _common import small_area
import numpy as np

from sar_uav.belief.motion import PROFILES, MotionModel
from sar_uav.belief import models as B


def _model(profile="hiker"):
    return MotionModel(small_area(), PROFILES[profile])


def test_transition_rows_sum_to_one():
    mm = _model()
    T = mm.transition_matrix(0.25)
    assert T.shape == (mm.n_land, mm.n_land)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert (T >= -1e-15).all()


def test_forecast_conserves_mass_and_stays_nonneg():
    area = small_area()
    mm = _model()
    p = B.uniform_belief(area)
    pf = mm.forecast(p, rain=0.4)
    assert abs(pf.sum() - 1) < 1e-9
    assert (pf >= 0).all()


def test_sample_steps_are_adjacent_land_cells():
    area = small_area()
    mm = _model("desoriented")
    rng = np.random.default_rng(3)
    land_set = set(area.land_idx().tolist())
    xy = area.centers_xy()
    cur = int(area.land_idx()[len(area.land_idx()) // 2])
    for _ in range(200):
        nxt = mm.sample_step(rng, cur, 0.1)
        assert nxt in land_set
        d = np.hypot(*(xy[nxt] - xy[cur]))
        assert d <= area.cell * np.sqrt(2) + 1e-9
        cur = nxt


def test_hiker_prefers_trail_proximity():
    area = small_area()
    mm = _model("hiker")
    rng = np.random.default_rng(5)
    trails = np.flatnonzero(area.trail_mask.reshape(-1))
    chosen = []
    for t in trails[:150]:
        o, p = mm.step_probs(int(t), 0.2)
        chosen.append(int(rng.choice(o, p=p)))
    mean_chosen = float(area.dist_trail.reshape(-1)[chosen].mean())
    mean_all = float(area.dist_trail.reshape(-1)[area.land_idx()].mean())
    assert mean_chosen < 0.5 * mean_all


def test_injured_stays_more_than_hiker():
    area = small_area()
    k = int(area.land_idx()[100])

    def stay_prob(profile):
        m = MotionModel(area, PROFILES[profile])
        _, p = m.step_probs(k, 0.0)
        return float(p[0])

    assert stay_prob("injured") > 2.0 * stay_prob("hiker")


def test_rain_reduces_movement():
    area = small_area()
    mm = _model()

    def move_mass(rain):
        T = mm.transition_matrix(rain)
        return float(np.mean(1.0 - np.diag(T)))

    assert move_mass(0.8) < move_mass(0.0)
