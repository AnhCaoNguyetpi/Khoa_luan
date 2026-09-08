"""Adapters for real GIS data (optional; require extra packages).

The core simulator runs fully offline on synthetic areas.  To use real
wilderness data, download the layers manually (see README "Du lieu that")
and convert them with :func:`raster_to_grid` / :func:`vector_to_distance`.
Every helper degrades gracefully: missing optional packages raise an
informative ``ImportError`` instead of breaking the pipeline.

Datasets referenced by the proposal:

* DEM / slope      -- NASA SRTM 1 arc-second (~30 m)          [doi:10.5067/MEASURES/SRTM/SRTMGL1.003]
* trails / roads   -- OpenStreetMap (highway=path|track|road, waterway=river)
* land cover       -- ESA WorldCover 10 m 2021 v200           [doi:10.5281/zenodo.7254221]
* weather          -- ERA5-Land hourly                        [doi:10.24381/cds.e2161bac]
* UAV imagery      -- WiSARD RGB+thermal dataset              [Broyles et al., IROS 2022]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .area import AreaData


def _need(pkg: str, extra: str) -> None:
    try:
        __import__(pkg)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"Package '{pkg}' is required for real-GIS import. "
            f"Install with:  pip install {extra}"
        ) from e


def raster_to_grid(tif_path: str | Path, target_cell_m: float,
                   band: int = 1) -> np.ndarray:
    """Read a GeoTIFF and resample to the target cell size (nearest)."""
    _need("rasterio", "rasterio")
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        scale = src.res[0] / target_cell_m
        out_h = max(1, int(src.height / scale))
        out_w = max(1, int(src.width / scale))
        data = src.read(band, out_shape=(out_h, out_w),
                        resampling=Resampling.nearest).astype(np.float64)
    return data


def vector_to_distance(geojson_path: str | Path, template_area) -> np.ndarray:
    """Distance [m] from every cell centre to the nearest vector feature."""
    _need("geopandas", "geopandas shapely")
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    H, W = template_area.shape
    xs, ys = template_area.centers_xy().T
    # assume same CRS in metres (reproject beforehand, e.g. UTM)
    union = gdf.geometry.union_all()
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=gdf.crs)
    d = pts.distance(union).to_numpy()
    return d.reshape(H, W)


def worldcover_to_land_id(tif_path: str | Path) -> np.ndarray:
    """ESA WorldCover classes -> internal land_id codes."""
    _need("rasterio", "rasterio")
    import rasterio
    mapping = {10: 2,   # tree cover   -> tree
               20: 1,   # shrubland    -> shrub
               30: 0,   # grassland    -> grass
               40: 0,   # cropland     -> grass (approx)
               50: 1,   # built-up     -> shrub (avoid)
               60: 3,   # bare         -> bare
               70: 3,   # snow/ice     -> bare
               80: 4,   # water        -> water
               90: 2,   # herbaceous wetland -> tree
               95: 2,   # mangroves    -> tree
               100: 3}  # moss/lichen  -> bare
    with rasterio.open(tif_path) as src:
        raw = src.read(1)
    out = np.zeros_like(raw, dtype=np.int16)
    for k, v in mapping.items():
        out[raw == k] = v
    return out


def era5_csv_to_series(csv_path: str | Path, dt_min: float):
    """Convert an ERA5-Land CSV export (time, t2m, tp, u10...) to WeatherSeries."""
    import pandas as pd
    from .weather import WeatherSeries

    df = pd.read_csv(csv_path)
    n = len(df)
    return WeatherSeries(
        dt_min=dt_min,
        cloud=np.clip(df.get("tcc", pd.Series(np.full(n, 0.35))).to_numpy(), 0, 1),
        rain=np.clip(df.get("tp", pd.Series(np.zeros(n))).to_numpy() * 10.0, 0, 1),
        wind_ms=df.get("wind", pd.Series(np.full(n, 3.0))).to_numpy(),
        temp_c=df.get("t2m_c", pd.Series(np.full(n, 18.0))).to_numpy(),
    )


def assemble_area(name: str, cell_m: float, dem_tif: str,
                  trails_geojson: Optional[str] = None,
                  roads_geojson: Optional[str] = None,
                  landcover_tif: Optional[str] = None,
                  origin_lonlat=None) -> "AreaData":  # noqa: F821
    """Assemble an :class:`AreaData` from downloaded files.

    All inputs must already share a projected CRS in metres (e.g. EPSG:32648
    for Vietnam's longitudes ~105-108E).
    """
    elev = raster_to_grid(dem_tif, cell_m)
    H, W = elev.shape
    gy, gx = np.gradient(elev, cell_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    padded = np.pad(elev, 1, mode="edge")
    stack = np.stack([padded[dy:dy + H, dx:dx + W]
                      for dy in range(3) for dx in range(3)])
    rugged = stack.std(axis=0)

    tmp = AreaData.__new__(AreaData)          # geometry-only template
    tmp.W, tmp.H, tmp.cell = W, H, float(cell_m)
    ii = np.arange(W * H).reshape(H, W)
    rr, cc = np.divmod(ii, W)
    tmp.centers_xy = lambda: np.stack([((cc + 0.5) * cell_m).reshape(-1),
                                       ((rr + 0.5) * cell_m).reshape(-1)], axis=1)

    if landcover_tif:
        land = worldcover_to_land_id(landcover_tif)
        if land.shape != (H, W):
            raise ValueError("land cover grid must match DEM grid after resampling")
    else:
        veg_guess = np.clip(0.5 - 0.01 * (slope - 20.0), 0.05, 0.9)
        land = np.where(slope > 34, 3, np.where(veg_guess > 0.55, 2,
                        np.where(veg_guess > 0.35, 1, 0))).astype(np.int16)

    water = (land == 4).astype(bool)
    veg = np.select([land == 2, land == 1, land == 0, land == 3],
                    [0.75, 0.45, 0.15, 0.05], default=0.05).astype(np.float64)
    veg[water] = 0.0

    def dist_mask(mask):
        pts = np.argwhere(mask)
        cc = tmp.centers_xy()
        if len(pts) == 0 or len(cc) == 0:
            return np.full((H, W), np.hypot(H, W) * cell_m)
        pc = (pts[:, ::-1] + 0.5) * cell_m
        d = np.sqrt(((cc[:, None, :] - pc[None, :, :]) ** 2).sum(-1)).min(1)
        return d.reshape(H, W)

    trail_mask = np.zeros((H, W), bool)
    road_mask = np.zeros((H, W), bool)
    dist_trail = dist_mask(trail_mask)
    dist_road = dist_mask(road_mask)
    if trails_geojson:
        dist_trail = vector_to_distance(trails_geojson, tmp)
        trail_mask = dist_trail <= cell_m
    if roads_geojson:
        dist_road = vector_to_distance(roads_geojson, tmp)
        road_mask = dist_road <= cell_m

    from .area import AreaData as AD
    return AD.from_arrays(name=name, cell=cell_m, layers={
        "elev": elev, "slope": slope, "rugged": rugged, "veg": veg,
        "land_id": land.astype(np.int16), "water": water,
        "trail_mask": trail_mask, "road_mask": road_mask,
        "dist_trail": dist_trail, "dist_road": dist_road,
    }, origin_lonlat=origin_lonlat, meta={"source": "gis-import"})
