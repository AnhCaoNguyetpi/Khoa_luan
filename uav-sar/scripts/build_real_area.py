"""Build a provenance-tracked 100 m SAR grid from downloaded official data."""
from _bootstrap import *  # noqa: F401,F403

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString

from sar_uav.data.area import AreaData


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regrid(path: Path, dst_shape, dst_transform, dst_crs, resampling):
    out = np.zeros(dst_shape, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            rasterio.band(src, 1), out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            dst_nodata=0, resampling=resampling,
        )
    return out


def osm_lines(path: Path, transformer, wanted):
    root = ET.parse(path).getroot()
    nodes = {n.attrib["id"]: (float(n.attrib["lon"]), float(n.attrib["lat"]))
             for n in root.findall("node")}
    lines = []
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if not wanted(tags):
            continue
        coords = [nodes[n.attrib["ref"]] for n in way.findall("nd")
                  if n.attrib["ref"] in nodes]
        if len(coords) >= 2:
            lines.append(LineString([transformer.transform(*p) for p in coords]))
    return lines


def distance_to(mask, cell):
    h, w = mask.shape
    pts = np.argwhere(mask)
    if not len(pts):
        return np.full((h, w), np.hypot(h, w) * cell)
    yy, xx = np.mgrid[0:h, 0:w]
    cells = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    return (np.sqrt(((cells[:, None] - pts[None, :]) ** 2).sum(-1)).min(1)
            * cell).reshape(h, w)


def build_weather(power_json: Path, out: Path, start_key: str):
    raw = json.loads(power_json.read_text(encoding="utf-8"))
    p = raw["properties"]["parameter"]
    keys = sorted(p["T2M"])
    start = keys.index(start_key)
    keys = keys[start:start + 8]  # enough for burn-in + 3-hour mission
    hourly = lambda name: np.array([p[name][k] for k in keys], dtype=float)
    xh = np.arange(len(keys)) * 60.0
    xm = np.arange((len(keys) - 1) * 60 + 1, dtype=float)
    temp = np.interp(xm, xh, hourly("T2M"))
    wind = np.interp(xm, xh, hourly("WS10M"))
    cloud = np.clip(np.interp(xm, xh, hourly("CLOUD_AMT")) / 100.0, 0, 1)
    precip = np.maximum(np.interp(xm, xh, hourly("PRECTOTCORR")), 0)
    # Detection model expects a dimensionless intensity. 10 mm/h maps to 1.
    rain = np.clip(precip / 10.0, 0, 1)
    np.savez_compressed(out, dt_min=1.0, cloud=cloud, rain=rain,
                        wind_ms=wind, temp_c=temp)
    return {"start_local_standard_time": start_key,
            "rain_normalisation_mm_h": 10.0,
            "source_header": raw.get("header", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tay_nguyen_real")
    ap.add_argument("--lon", type=float, default=108.25)
    ap.add_argument("--lat", type=float, default=12.60)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--cell", type=float, default=100.0)
    ap.add_argument("--weather-start", default="2025011512")
    a = ap.parse_args()

    raw = ROOT / "data" / "raw" / "tay_nguyen_10825_1260"
    dem = raw / "copernicus_dem_n12e108.tif"
    cover = raw / "esa_worldcover_2021_n12e108.tif"
    osm = raw / "openstreetmap_map.osm"
    power = raw / "nasa_power_hourly_2025.json"
    for p in (dem, cover, osm, power):
        if not p.exists():
            raise FileNotFoundError(p)

    crs = "EPSG:32649"  # UTM zone 49N
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = to_utm.transform(a.lon, a.lat)
    extent = a.size * a.cell
    left, top = cx - extent / 2, cy + extent / 2
    transform = from_origin(left, top, a.cell, a.cell)
    shape = (a.size, a.size)

    elev = regrid(dem, shape, transform, crs, Resampling.bilinear)
    wc = regrid(cover, shape, transform, crs, Resampling.nearest).astype(np.int16)
    mapping = {10: 2, 20: 1, 30: 0, 40: 0, 50: 3, 60: 3,
               70: 3, 80: 4, 90: 1, 95: 2, 100: 1}
    land = np.full(shape, 3, dtype=np.int16)
    for src, dst in mapping.items():
        land[wc == src] = dst
    water = land == 4
    veg = np.select([land == 2, land == 1, land == 0, land == 3],
                    [0.75, 0.45, 0.15, 0.05], default=0.05).astype(float)
    veg[water] = 0

    trail_types = {"path", "track", "footway", "bridleway", "steps"}
    trails = osm_lines(osm, to_utm,
                      lambda t: t.get("highway") in trail_types)
    roads = osm_lines(osm, to_utm,
                     lambda t: "highway" in t and t.get("highway") not in trail_types)
    rivers = osm_lines(osm, to_utm, lambda t: "waterway" in t)
    burn = lambda lines: rasterize([(g, 1) for g in lines], out_shape=shape,
                                   transform=transform, all_touched=True,
                                   dtype="uint8").astype(bool) if lines else np.zeros(shape, bool)
    trail_mask, road_mask, river_mask = burn(trails), burn(roads), burn(rivers)
    water |= river_mask
    land[water] = 4
    veg[water] = 0

    gy, gx = np.gradient(elev, a.cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    padded = np.pad(elev, 1, mode="edge")
    rugged = np.stack([padded[y:y+a.size, x:x+a.size]
                       for y in range(3) for x in range(3)]).std(0)

    sources = {
        "dem": {"dataset": "Copernicus DEM GLO-30", "provider": "Copernicus",
                "url": "https://copernicus-dem-30m.s3.amazonaws.com/", "sha256": sha256(dem)},
        "landcover": {"dataset": "ESA WorldCover 2021 v200", "provider": "ESA",
                      "doi": "10.5281/zenodo.7254221", "sha256": sha256(cover)},
        "transport": {"dataset": "OpenStreetMap", "provider": "OSM contributors",
                      "license": "ODbL 1.0", "sha256": sha256(osm)},
        "weather": {"dataset": "NASA POWER Hourly 2025", "provider": "NASA Langley",
                    "url": "https://power.larc.nasa.gov/", "sha256": sha256(power)},
    }
    area = AreaData(a.name, a.size, a.size, a.cell, elev, slope, rugged,
                    veg, land, water, trail_mask, road_mask,
                    distance_to(trail_mask, a.cell), distance_to(road_mask, a.cell),
                    origin_lonlat=(a.lon, a.lat),
                    meta={"source": "real-gis", "crs": crs,
                          "bbox_size_m": extent, "sources": sources,
                          "built_utc": datetime.now(timezone.utc).isoformat()})
    out = ROOT / "data" / "areas" / a.name
    area.save(out)
    weather_meta = build_weather(power, out / "weather.npz", a.weather_start)
    meta_path = out / "area.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["meta"]["weather_processing"] = weather_meta
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}: {shape}, elev={elev.min():.1f}..{elev.max():.1f} m, "
          f"trails={trail_mask.sum()}, roads={road_mask.sum()}, water={water.sum()}")


if __name__ == "__main__":
    main()
