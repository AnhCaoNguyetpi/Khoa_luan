"""Environment- and UAV-dependent detection probability (proposal section
"Xac suat vi tri va xac suat phat hien").

    q_ikt = f(sensor, altitude(folded into q_base), vegetation,
              visibility/cloud, weather, search time)

Per-minute tick probability::

    q_tick = q_base * exp(-a_veg*veg) * exp(-a_cloud*cloud) * exp(-a_rain*rain)

Dwell accumulation over ``d`` ticks at a fixed cell::

    q_dwell = 1 - (1 - q_tick)^d

RGB degrades sharply under canopy and clouds; thermal is much less affected
by vegetation/light but more power-hungry -- this asymmetry is exactly what
the RQ3 detection-aware assignment exploits.  Parameters are placeholders
calibratable against WiSARD / Dumencic et al. (2025) field data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

SENSOR_PARAMS: Dict[str, Dict] = {
    "rgb": {
        "q_base": 0.30,
        "atten_veg": 2.2,
        "atten_cloud": 1.8,
        "atten_rain": 1.2,
        "atten_wind": 0.01,      # per m/s above 8 m/s (image blur)
    },
    "thermal": {
        "q_base": 0.26,
        "atten_veg": 0.7,
        "atten_cloud": 0.15,
        "atten_rain": 0.8,
        "atten_wind": 0.005,
    },
}


def sensor_for(name: str) -> Dict:
    try:
        return SENSOR_PARAMS[name]
    except KeyError:
        raise ValueError(f"unknown sensor '{name}', available: "
                         f"{list(SENSOR_PARAMS)}")


def q_tick(sensor: str, veg: float, cloud: float = 0.0, rain: float = 0.0,
           wind_ms: float = 0.0) -> float:
    """Single-tick detection probability for one cell."""
    p = SENSOR_PARAMS[sensor] if sensor in SENSOR_PARAMS \
        else dict(q_base=sensor, atten_veg=0, atten_cloud=0, atten_rain=0,
                  atten_wind=0)
    wind_excess = max(0.0, float(wind_ms) - 8.0)
    q = p["q_base"] \
        * np.exp(-p["atten_veg"] * float(np.clip(veg, 0, 1))) \
        * np.exp(-p["atten_cloud"] * float(np.clip(cloud, 0, 1))) \
        * np.exp(-p["atten_rain"] * float(np.clip(rain, 0, 1))) \
        * np.exp(-p["atten_wind"] * wind_excess)
    return float(np.clip(q, 1e-6, 0.999))


def q_dwell(sensor: str, veg: float, weather_mean: Dict[str, float],
            dwell_ticks: int) -> float:
    """Detection probability accumulated over a dwell of ``dwell_ticks``
    ticks using mean weather over the window."""
    q1 = q_tick(sensor, veg, weather_mean.get("cloud", 0.0),
                weather_mean.get("rain", 0.0), weather_mean.get("wind_ms", 0.0))
    d = max(int(dwell_ticks), 0)
    return float(1.0 - (1.0 - q1) ** d)


def q_map_for_sensor(sensor: str, area, weather_mean: Dict[str, float],
                     dwell_ticks: int) -> np.ndarray:
    """(n,) vector of per-cell dwell detection probabilities (0 on water).

    Vectorised equivalent of calling :func:`q_dwell` per cell.
    """
    p = sensor_for(sensor)
    veg = np.clip(area.veg.reshape(-1), 0, 1)
    q1 = p["q_base"] \
        * np.exp(-p["atten_veg"] * veg) \
        * np.exp(-p["atten_cloud"] * float(weather_mean.get("cloud", 0.0))) \
        * np.exp(-p["atten_rain"] * float(weather_mean.get("rain", 0.0))) \
        * np.exp(-p["atten_wind"] * max(0.0, float(weather_mean.get("wind_ms", 0.0)) - 8.0))
    q1 = np.clip(q1, 1e-6, 0.999)
    q = 1.0 - (1.0 - q1) ** max(int(dwell_ticks), 0)
    q[area.water.reshape(-1)] = 0.0
    return q
