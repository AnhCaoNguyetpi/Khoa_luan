"""Detection model sanity: monotonicity, sensor asymmetry, dwell bounds."""
from _common import small_area
import numpy as np

from sar_uav.detection.sensors import (q_dwell, q_map_for_sensor, q_tick,
                                       sensor_for)


def test_veg_monotonic_decreases_q():
    qs = [q_tick("rgb", veg=v) for v in np.linspace(0, 1, 11)]
    assert all(a >= b - 1e-12 for a, b in zip(qs, qs[1:]))


def test_rgb_degrades_faster_than_thermal_under_canopy():
    open_q = [q_tick("rgb", 0.0), q_tick("thermal", 0.0)]
    forest = [q_tick("rgb", 0.9), q_tick("thermal", 0.9)]
    rgb_ratio = forest[0] / open_q[0]
    th_ratio = forest[1] / open_q[1]
    assert rgb_ratio < th_ratio


def test_weather_reduces_detection():
    clear = q_tick("rgb", 0.3)
    stormy = q_tick("rgb", 0.3, cloud=0.9, rain=0.8, wind_ms=12.0)
    assert stormy < clear
    # thermal barely cares about clouds
    th_clear = q_tick("thermal", 0.3, cloud=0.0)
    th_cloud = q_tick("thermal", 0.3, cloud=0.9)
    assert th_cloud > 0.7 * th_clear


def test_dwell_accumulates_and_is_bounded():
    w = {"cloud": 0.2, "rain": 0.0, "wind_ms": 3.0}
    for d in (1, 3, 10):
        q = q_dwell("rgb", 0.4, w, d)
        assert 0 < q < 1
    assert q_dwell("rgb", 0.4, w, 6) > q_dwell("rgb", 0.4, w, 2)


def test_qmap_zero_on_water():
    area = small_area()
    qm = q_map_for_sensor("thermal", area, {"cloud": .3}, 3)
    assert qm[area.water.reshape(-1)].sum() == 0.0
    assert (qm[area.land_idx()] > 0).all()
