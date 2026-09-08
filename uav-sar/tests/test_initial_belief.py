"""Initial-belief models: normalisation, water exclusion, concentration."""
from _common import small_area
import numpy as np

from sar_uav.belief import models as B


def _check_valid(p, area):
    assert abs(p.sum() - 1) < 1e-9
    assert (p >= 0).all()
    assert p[area.water.reshape(-1)].sum() == 0


def test_uniform():
    area = small_area()
    p = B.uniform_belief(area)
    _check_valid(p, area)


def test_distance_peaks_near_ipp():
    area = small_area()
    ipp = int(area.land_idx()[len(area.land_idx()) // 2])
    p = B.distance_gaussian_belief(area, ipp, elapsed_min=45.0)
    _check_valid(p, area)
    xy = area.centers_xy()
    top = int(np.argmax(p))
    d = np.hypot(*(xy[top] - xy[ipp]))
    assert d < 600.0          # mode within ~6 cells of the IPP


def test_gis_expert_normalised_and_concentrated():
    area = small_area()
    trails = np.flatnonzero(area.trail_mask.reshape(-1))
    p = B.gis_expert_belief(area, int(trails[0]), 60.0)
    _check_valid(p, area)
    # mass near trails should exceed its share of cells
    near_trail = area.dist_trail.reshape(-1)[area.land_idx()] < 300
    share_mass = p[area.land_idx()][near_trail].sum()
    share_cells = near_trail.mean()
    assert share_mass > share_cells


def test_datadriven_from_synthetic_weights(tmp_path=None):
    area = small_area()
    rng = np.random.default_rng(0)
    X = B.cell_features(area, int(area.land_idx()[5]), 40.0)
    w = rng.normal(size=X.shape[1])
    pack = {"w": w, "mean": X.mean(0), "std": np.maximum(X.std(0), 1e-6)}
    p = B.data_driven_belief(area, int(area.land_idx()[5]), 40.0, pack)
    _check_valid(p, area)


def test_build_dispatch_unknown_model_raises():
    try:
        B.build_initial_belief("nope", small_area(), 0, 30.0, {})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
