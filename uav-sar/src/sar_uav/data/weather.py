"""Weather series over the mission timeline (ERA5-Land stand-in).

The synthetic generator produces diurnal temperature, slowly-varying cloud
cover and episodic rain -- enough to drive both the movement model
(``beta_rain``) and the detection model (cloud / rain attenuation).
Real ERA5-Land data can be imported via ``scripts/import_gis_layers.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from pathlib import Path


@dataclass
class WeatherSeries:
    dt_min: float
    cloud: np.ndarray      # (T,) in [0, 1]
    rain: np.ndarray       # (T,) in [0, 1] (normalised intensity)
    wind_ms: np.ndarray    # (T,)
    temp_c: np.ndarray     # (T,)

    def __len__(self):
        return len(self.cloud)

    def at(self, tick: int) -> Dict[str, float]:
        i = int(np.clip(tick, 0, len(self) - 1))
        return {"cloud": float(self.cloud[i]), "rain": float(self.rain[i]),
                "wind_ms": float(self.wind_ms[i]), "temp_c": float(self.temp_c[i])}

    def window_mean(self, t0: int, ticks: int) -> Dict[str, float]:
        sl = slice(int(t0), int(t0) + max(int(ticks), 1))
        return {"cloud": float(self.cloud[sl].mean()) if len(self.cloud[sl]) else float(self.cloud[-1]),
                "rain": float(self.rain[sl].mean()) if len(self.rain[sl]) else float(self.rain[-1]),
                "wind_ms": float(self.wind_ms[sl].mean()) if len(self.wind_ms[sl]) else float(self.wind_ms[-1]),
                "temp_c": float(self.temp_c[sl].mean()) if len(self.temp_c[sl]) else float(self.temp_c[-1])}


def load_weather(path: str | Path) -> WeatherSeries:
    """Load a provenance-tracked weather series prepared for an area."""
    z = np.load(path)
    return WeatherSeries(dt_min=float(z["dt_min"]), cloud=z["cloud"],
                         rain=z["rain"], wind_ms=z["wind_ms"],
                         temp_c=z["temp_c"])


def make_weather(total_min: float, dt_min: float, rng: np.random.Generator,
                 params: Dict | None = None) -> WeatherSeries:
    p = params or {}
    T = int(np.ceil(total_min / dt_min)) + 1
    t = np.arange(T) * dt_min

    # cloud cover: smooth OU-like random walk clipped to [0, 1]
    cloud = np.zeros(T)
    cloud[0] = p.get("cloud_mean", 0.35)
    for i in range(1, T):
        cloud[i] = np.clip(cloud[i - 1] + 0.02 * rng.standard_normal()
                           + 0.004 * (p.get("cloud_mean", 0.35) - cloud[i - 1]),
                           0.0, 1.0)

    # rain episodes
    rain = np.zeros(T)
    i = 1
    ep_lo, ep_hi = p.get("rain_episode_len", (20, 60))
    while i < T:
        if rng.random() < p.get("rain_episode_prob", 0.02):
            dur = int(rng.integers(ep_lo, ep_hi)) // int(dt_min) + 1
            peak = rng.uniform(*p.get("rain_peak_range", (0.3, 0.9)))
            shape = np.sin(np.linspace(0.2, np.pi - 0.2, dur))  # ramp up/down
            rain[i:i + dur] = np.maximum(rain[i:i + dur], peak * shape[:max(0, T - i)])
            i += dur
        else:
            i += 1

    wind_base = p.get("wind_base_ms", 3.0)
    wind = wind_base + 1.5 * np.abs(np.cumsum(0.03 * rng.standard_normal(T)))
    wind = np.minimum(wind, 14.0)

    temp_base = p.get("temp_base_c", 18.0)
    amp = p.get("diurnal_amplitude_c", 6.0)
    temp = temp_base + amp * np.sin(2 * np.pi * (t / (24 * 60.0)) - np.pi / 2) \
        - 3.0 * rain

    return WeatherSeries(dt_min=dt_min, cloud=cloud, rain=rain,
                         wind_ms=wind, temp_c=temp)
