"""Spatial search area: grid + GIS layers.

The area is a regular grid of ``W x H`` cells (row-major indexing:
``idx = row * W + col``).  Static per-cell layers follow the integrated
spatial database described in the proposal (section "Xay dung bo du lieu
tich hop"):

    elevation, slope, ruggedness, vegetation, land cover id,
    distance-to-trail, distance-to-road, trail / road / water masks.

Two ways to obtain an area:

* :func:`build_synthetic_area` -- fully offline synthetic wilderness
  (default; keeps every experiment reproducible without internet);
* real GIS import -- see ``data/adapters.py`` and
  ``scripts/import_gis_layers.py`` (SRTM / OSM / ESA WorldCover / ERA5-Land).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LAND_NAMES = {0: "grass", 1: "shrub", 2: "tree", 3: "bare", 4: "water"}


@dataclass
class AreaData:
    name: str
    W: int                       # columns
    H: int                       # rows
    cell: float                  # cell size [m]
    elev: np.ndarray             # (H, W) metres
    slope: np.ndarray            # (H, W) degrees
    rugged: np.ndarray           # (H, W) metres (local std of elevation)
    veg: np.ndarray              # (H, W) vegetation fraction in [0, 1]
    land_id: np.ndarray          # (H, W) int codes, see LAND_NAMES
    water: np.ndarray            # (H, W) bool
    trail_mask: np.ndarray       # (H, W) bool
    road_mask: np.ndarray        # (H, W) bool
    dist_trail: np.ndarray       # (H, W) metres to nearest trail cell
    dist_road: np.ndarray        # (H, W) metres to nearest road cell
    origin_lonlat: Optional[Tuple[float, float]] = None
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------- basics
    @property
    def n_cells(self) -> int:
        return self.W * self.H

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.H, self.W)

    def idx_grid(self) -> np.ndarray:
        """(H, W) array of flat cell indices."""
        return np.arange(self.n_cells, dtype=np.int64).reshape(self.H, self.W)

    def centers_xy(self) -> np.ndarray:
        """(n, 2) world coordinates [m] of cell centres (x=col, y=row)."""
        ii = self.idx_grid()
        rr, cc = np.divmod(ii, self.W)
        x = ((cc + 0.5) * self.cell).reshape(-1)
        y = ((rr + 0.5) * self.cell).reshape(-1)
        return np.stack([x, y], axis=1)

    def idx_of_xy(self, xy) -> int:
        """Nearest cell index for world coordinates (clipped to bounds)."""
        x, y = float(xy[0]), float(xy[1])
        c = int(np.clip(np.floor(x / self.cell), 0, self.W - 1))
        r = int(np.clip(np.floor(y / self.cell), 0, self.H - 1))
        return r * self.W + c

    def rc_of(self, idx):
        r, c = np.divmod(np.asarray(idx, dtype=np.int64), self.W)
        return r, c

    def land_idx(self) -> np.ndarray:
        """Flat indices of searchable (non-water) cells."""
        return np.flatnonzero(~self.water.reshape(-1))

    def neighbor_idx(self, idx: int) -> list:
        """Valid (in-bounds, non-water) 8-neighbour flat indices."""
        r, c = divmod(int(idx), self.W)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.H and 0 <= cc < self.W \
                        and not self.water[rr, cc]:
                    out.append(rr * self.W + cc)
        return out

    def is_water(self, idx) -> np.ndarray:
        flat = self.water.reshape(-1)
        return flat[np.asarray(idx, dtype=np.int64)]

    def elev_at_xy(self, xy) -> float:
        """Bilinear elevation interpolation at world coordinates."""
        x = np.clip(float(xy[0]) / self.cell - 0.5, 0.0, self.W - 1.001)
        y = np.clip(float(xy[1]) / self.cell - 0.5, 0.0, self.H - 1.001)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        e = self.elev
        return float((1 - fy) * ((1 - fx) * e[y0, x0] + fx * e[y0, x0 + 1])
                     + fy * ((1 - fx) * e[y0 + 1, x0] + fx * e[y0 + 1, x0 + 1]))

    # ------------------------------------------------------------ persistence
    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "layers.npz",
            elev=self.elev.astype(np.float32),
            slope=self.slope.astype(np.float32),
            rugged=self.rugged.astype(np.float32),
            veg=self.veg.astype(np.float32),
            land_id=self.land_id.astype(np.int16),
            water=self.water.astype(np.uint8),
            trail_mask=self.trail_mask.astype(np.uint8),
            road_mask=self.road_mask.astype(np.uint8),
            dist_trail=self.dist_trail.astype(np.float32),
            dist_road=self.dist_road.astype(np.float32),
        )
        meta = {
            "name": self.name, "W": self.W, "H": self.H,
            "cell": self.cell, "origin_lonlat": self.origin_lonlat,
            "meta": self.meta,
        }
        with open(d / "area.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, d: str | Path) -> "AreaData":
        d = Path(d)
        with open(d / "area.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        z = np.load(d / "layers.npz")
        return cls(
            name=meta["name"], W=int(meta["W"]), H=int(meta["H"]),
            cell=float(meta["cell"]),
            elev=z["elev"].astype(np.float64),
            slope=z["slope"].astype(np.float64),
            rugged=z["rugged"].astype(np.float64),
            veg=z["veg"].astype(np.float64),
            land_id=z["land_id"],
            water=z["water"].astype(bool),
            trail_mask=z["trail_mask"].astype(bool),
            road_mask=z["road_mask"].astype(bool),
            dist_trail=z["dist_trail"].astype(np.float64),
            dist_road=z["dist_road"].astype(np.float64),
            origin_lonlat=tuple(meta["origin_lonlat"]) if meta.get("origin_lonlat") else None,
            meta=meta.get("meta", {}),
        )

    @classmethod
    def from_arrays(cls, name, cell, layers: Dict[str, np.ndarray],
                    origin_lonlat=None, meta=None) -> "AreaData":
        """Build from pre-computed (H, W) layers -- used by the GIS importer."""
        elev = layers["elev"]
        H, W = elev.shape
        required = ["slope", "rugged", "veg", "land_id", "water",
                    "trail_mask", "road_mask", "dist_trail", "dist_road"]
        for k in required:
            if k not in layers:
                raise ValueError(f"missing layer '{k}'")
        return cls(name=name, W=W, H=H, cell=float(cell), elev=elev,
                   **{k: layers[k] for k in required},
                   origin_lonlat=origin_lonlat, meta=meta or {})


# --------------------------------------------------------------------------
# synthetic terrain generation (offline default)
# --------------------------------------------------------------------------

def _bump_field(rng, H, W, cell, n_bumps, amp_range, sigma_range, positive=False):
    """Smooth random scalar field = sum of Gaussian bumps, scaled to [0, 1]."""
    yy, xx = np.mgrid[0:H, 0:W]
    xx = (xx + 0.5) * cell
    yy = (yy + 0.5) * cell
    out = np.zeros((H, W))
    for _ in range(n_bumps):
        cx, cy = rng.uniform(0, W * cell), rng.uniform(0, H * cell)
        amp = rng.uniform(*amp_range)
        if positive:
            amp = abs(amp)
        sig = rng.uniform(*sigma_range)
        out += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-9)


def _carve_path(rng, elev, water, start, goal, slope_pen=2.0, slack=0.35):
    """Stochastic greedy least-cost walk from ``start`` to ``goal`` on the grid.

    Used for rivers (pure downhill + noise) and trails (penalise slope).
    Marks visited cells in-place on ``mask``.
    """
    H, W = elev.shape
    r, c = start
    gr, gc = goal
    mask = set()
    for _ in range(4 * H * W):
        mask.add((r, c))
        if (r, c) == (gr, gc):
            break
        best, best_score = None, np.inf
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if water[nr, nc]:
                    continue
                step_len = np.hypot(dr, dc)
                d_goal = np.hypot(gr - nr, gc - nc) / step_len
                rise = max(0.0, elev[nr, nc] - elev[r, c]) / 10.0
                score = d_goal + slope_pen * rise + slack * step_len * rng.random()
                if score < best_score:
                    best_score, best = score, (nr, nc)
        if best is None:
            break
        r, c = best
    return mask


def build_synthetic_area(name: str = "demo_valley", W: int = 40, H: int = 40,
                         cell: float = 100.0, seed: int = 7) -> AreaData:
    """Generate a synthetic wilderness area (mountainous, trailed, with a river)."""
    rng = np.random.default_rng(seed)

    # --- elevation: tilted plain + Gaussian mounds (metres)
    yy, xx = np.mgrid[0:H, 0:W]
    xm = (xx + 0.5) * cell
    ym = (yy + 0.5) * cell
    elev = 380.0 + 0.06 * xm + 0.03 * ym
    for _ in range(9):
        cx, cy = rng.uniform(0, W * cell), rng.uniform(0, H * cell)
        amp = rng.uniform(-70, 130)
        sig = rng.uniform(350, 950)
        elev += amp * np.exp(-(((xm - cx) ** 2 + (ym - cy) ** 2) / (2 * sig ** 2)))
    # sharper ridge detail -> realistic slope distribution
    for _ in range(14):
        cx, cy = rng.uniform(0, W * cell), rng.uniform(0, H * cell)
        amp = rng.uniform(-35, 70)
        sig = rng.uniform(120, 320)
        elev += amp * np.exp(-(((xm - cx) ** 2 + (ym - cy) ** 2) / (2 * sig ** 2)))

    # --- slope [deg] and ruggedness (local std of elevation, 3x3)
    gy, gx = np.gradient(elev, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    padded = np.pad(elev, 1, mode="edge")
    stack = np.stack([padded[dy:dy + H, dx:dx + W]
                      for dy in range(3) for dx in range(3)])
    rugged = stack.std(axis=0)

    # --- river: noisy downhill walk from north edge to south edge
    water = np.zeros((H, W), dtype=bool)
    c0 = int(rng.integers(W // 4, 3 * W // 4))
    riv = _carve_path(rng, elev + 25.0 * rng.random((H, W)), water,
                      start=(0, c0), goal=(H - 1, int(rng.integers(0, W))),
                      slope_pen=-0.6, slack=1.2)
    for (r, c) in riv:
        water[r, c] = True
        if c + 1 < W:
            water[r, c + 1] = True     # ~2-cell wide channel

    # --- trails: connect three edge trailheads through the interior
    trail_mask = np.zeros((H, W), dtype=bool)
    heads = [(int(rng.integers(0, H)), 0),
             (H - 1, int(rng.integers(W // 2, W))),
             (0, int(rng.integers(W // 2, W)))]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        seg = _carve_path(rng, elev, water, start=heads[a], goal=heads[b])
        for (r, c) in seg:
            if not water[r, c]:
                trail_mask[r, c] = True

    # --- a rough road along the southern edge
    road_mask = np.zeros((H, W), dtype=bool)
    road_mask[H - 2, :] = True
    road_mask &= ~water

    # --- vegetation: smooth field, denser at low elevation / gentle slopes
    vegf = _bump_field(rng, H, W, cell, n_bumps=12, amp_range=(0.4, 1.0),
                       sigma_range=(250, 700), positive=True)
    veg = 0.18 + 0.72 * vegf + 0.0025 * (elev.mean() - elev) \
        - 0.012 * np.maximum(0.0, slope - 22.0)
    veg = np.clip(veg, 0.02, 0.95)
    veg[water] = 0.0

    # --- land cover classes
    land_id = np.zeros((H, W), dtype=np.int16)          # grass
    land_id[veg > 0.35] = 1                             # shrub
    land_id[veg > 0.58] = 2                             # tree
    land_id[(veg < 0.15) | (slope > 34.0)] = 3          # bare / rock
    land_id[water] = 4

    # --- distance transforms (brute force: fine for these grid sizes)
    def dist_to(mask):
        pts = np.argwhere(mask)
        if len(pts) == 0:
            return np.full((H, W), np.hypot(H, W) * cell)
        pc = (pts[:, ::-1] + 0.5) * cell              # (m, 2) xy of masked cells
        cc = np.stack([xm.reshape(-1), ym.reshape(-1)], axis=1)
        d = np.sqrt(((cc[:, None, :] - pc[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return d.reshape(H, W)

    dist_trail = dist_to(trail_mask)
    dist_road = dist_to(road_mask)

    return AreaData(
        name=name, W=W, H=H, cell=cell, elev=elev, slope=slope,
        rugged=rugged, veg=veg, land_id=land_id, water=water,
        trail_mask=trail_mask, road_mask=road_mask,
        dist_trail=dist_trail, dist_road=dist_road,
        origin_lonlat=None,
        meta={"source": "synthetic", "seed": seed},
    )


def get_or_build_area(name: str, data_dir_parent: str | Path,
                      cfg: Dict | None = None) -> AreaData:
    """Load ``<data_dir_parent>/areas/<name>`` or build the synthetic default."""
    d = Path(data_dir_parent) / "areas" / name
    if (d / "area.json").exists():
        return AreaData.load(d)
    c = (cfg or {}).get("area", {})
    area = build_synthetic_area(name=name, W=c.get("grid_w", 40),
                                H=c.get("grid_h", 40),
                                cell=c.get("cell_size_m", 100.0),
                                seed=c.get("seed", 7))
    area.save(d)
    return area
