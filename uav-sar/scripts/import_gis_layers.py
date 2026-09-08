"""Import real GIS layers into an AreaData.

Workflow (manual download; see README section 'Du lieu that'):

  1. SRTM 1-arc-second DEM for your region  -> GeoTIFF in metres CRS
  2. OpenStreetMap trails/roads             -> GeoJSON (same CRS)
  3. ESA WorldCover 2021                    -> GeoTIFF matching DEM grid
  4. ERA5-Land weather                      -> CSV export

then::

    python scripts/import_gis_layers.py --name taynguyen \
        --cell 100 --dem path/to/dem.tif \
        --trails path/to/trails.geojson --roads path/to/roads.geojson \
        [--landcover path/to/worldcover.tif] [--origin-lonlat 108.2 12.6]

Requires: pip install rasterio geopandas shapely
"""
from _bootstrap import *  # noqa: F401,F403
import argparse

from sar_uav.data.adapters import assemble_area

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell", type=float, default=100.0)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--trails", default=None)
    ap.add_argument("--roads", default=None)
    ap.add_argument("--landcover", default=None)
    ap.add_argument("--origin-lonlat", nargs=2, type=float, default=None)
    a = ap.parse_args()

    area = assemble_area(a.name, a.cell, a.dem, trails_geojson=a.trails,
                         roads_geojson=a.roads, landcover_tif=a.landcover,
                         origin_lonlat=tuple(a.origin_lonlat) if a.origin_lonlat else None)
    out = ROOT / "data" / "areas" / a.name
    area.save(out)
    print(f"saved -> {out} ({area.W}x{area.H})")
