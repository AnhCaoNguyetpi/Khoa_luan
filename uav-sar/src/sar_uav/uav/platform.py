"""UAV platform specification and fleet construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UAVSpec:
    name: str
    sensor: str                     # 'rgb' | 'thermal'
    speed_mps: float = 13.0         # cruise ground speed
    battery_wh: float = 88.0
    fly_power_w: float = 170.0      # mean electrical power while cruising
    hover_power_w: float = 190.0    # mean power while hovering & searching
    climb_wh_per_m: float = 0.05    # extra energy per metre of ascent
    reserve_frac: float = 0.15      # keep this fraction to guarantee RTB
    swap_minutes: float = 4.0       # battery swap time at base
    allow_recharge: bool = True
    base_cell: Optional[int] = None

    @property
    def speed_mpm(self) -> float:                 # metres per minute
        return self.speed_mps * 60.0

    @property
    def wh_per_m(self) -> float:
        """Cruise energy per metre: W / (m/s) -> Wh per m."""
        return self.fly_power_w / (3600.0 * self.speed_mps)

    def fly_cost(self, dist_m: float, ascent_m: float = 0.0) -> float:
        """Energy model (proposal section "Mo phong hoat dong cua UAV")::

            E_fly = gamma * d (+ eta * max(0, e_j - e_i))
        """
        return self.wh_per_m * max(dist_m, 0.0) \
            + self.climb_wh_per_m * max(ascent_m, 0.0)

    def hover_cost(self, minutes: float) -> float:
        """E_search = alpha * tau."""
        return self.hover_power_w / 60.0 * max(minutes, 0.0)

    def endurance_min(self) -> float:
        return self.battery_wh / self.hover_power_w * 60.0

    def range_m(self) -> float:
        return self.battery_wh / self.wh_per_m


def fleet_from_config(cfg_list: List[Dict]) -> List[UAVSpec]:
    return [UAVSpec(**{k: v for k, v in c.items() if k in UAVSpec.__dataclass_fields__})
            for c in cfg_list]
