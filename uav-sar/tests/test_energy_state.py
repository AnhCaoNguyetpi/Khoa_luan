"""UAV platform energy model and state machine behaviour."""
from _common import small_area
import numpy as np

from sar_uav.uav.platform import UAVSpec
from sar_uav.uav.state import UAVState


def test_energy_formulas():
    s = UAVSpec(name="t", sensor="rgb", speed_mps=10.0, fly_power_w=100.0,
                hover_power_w=120.0, battery_wh=60.0, climb_wh_per_m=0.05)
    assert abs(s.wh_per_m - 100 / 36000) < 1e-12       # W/(m/s)/3600
    assert abs(s.fly_cost(36000.0) - 100.0) < 1e-9     # 1 h at 10 m/s
    assert abs(s.hover_cost(30.0) - 60.0) < 1e-9
    assert s.fly_cost(1000, ascent_m=50) == s.fly_cost(1000) + 2.5


def test_fly_consumes_battery_and_counts_distance():
    area = small_area()
    spec = UAVSpec(name="u", sensor="rgb", speed_mps=13.0)
    base = area.centers_xy()[int(area.land_idx()[10])]
    st = UAVState(spec, area, base)
    st.assign_cycles([[(int(area.land_idx()[200]), 1)]])
    for _ in range(50):
        st.tick(1.0)
        if st.at_base:
            break
    assert st.dist_flown > 0
    assert st.energy_spent > 0
    assert st.battery < spec.battery_wh


def test_assign_only_when_idle():
    area = small_area()
    spec = UAVSpec(name="u", sensor="rgb")
    st = UAVState(spec, area, np.zeros(2))
    st.assign_cycles([[(int(area.land_idx()[3]), 1)]])
    st.tick(1.0)                       # now flying
    try:
        st.assign_cycles([[((int(area.land_idx()[4])), 1)]])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when busy")


def test_low_battery_forces_rtb_then_swap():
    area = small_area()
    spec = UAVSpec(name="u", sensor="rgb", allow_recharge=True,
                   swap_minutes=2.0, speed_mps=5.0)   # slow enough to see flight
    land = area.land_idx()
    base = area.centers_xy()[int(land[3])]
    c1 = int(land[-30])                     # well away from base
    st = UAVState(spec, area, base)
    st.assign_cycles([[(c1, 2)]])
    for _ in range(2):                      # get airborne
        st.tick(1.0)
    assert st.mode == "fly"
    # leave less battery than reserve + flight-home cost -> guard must fire
    st.battery = spec.reserve_frac * spec.battery_wh \
        + st.return_cost() - 1e-6
    swapped = False
    for _ in range(600):
        ev = st.tick(1.0)
        if ev.get("swapped"):
            swapped = True
            assert abs(st.battery - spec.battery_wh) < 1e-9
            break
    assert swapped, "UAV must RTB and swap when reserve would be violated"


def test_visits_counted_per_searched_cell():
    area = small_area()
    spec = UAVSpec(name="u", sensor="rgb", allow_recharge=False)
    land = area.land_idx()
    base = area.centers_xy()[int(land[3])]
    c1 = int(land[40])
    st = UAVState(spec, area, base)
    st.assign_cycles([[(c1, 2)]])
    done = False
    for _ in range(200):
        ev = st.tick(1.0)
        if ev.get("search_done") == c1:
            done = True
            break
    assert done and st.visits.get(c1, 0) == 1
