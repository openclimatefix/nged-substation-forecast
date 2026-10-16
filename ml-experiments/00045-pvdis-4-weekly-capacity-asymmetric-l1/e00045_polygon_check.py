"""Validate the Primary service-area polygons in `TimeSeriesMetadata.area_wkt`.

Reports, for the 16 Primary series: validity, area, whether the substation point lies inside its
own polygon, pairwise overlaps (excluding the two Wrangle transformers, which share one polygon),
gaps inside the union's hull, and the union area. Writes `polygon_check.parquet` and a map.
"""

import itertools
from pathlib import Path
from typing import Final

import polars as pl
import shapely
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

METADATA_PATH: Final[Path] = Path("data/NGED/metadata.parquet")
OUT_DIR: Final[Path] = Path("data/ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1")
TO_BNG: Final = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform
"""WGS84 longitude/latitude to British National Grid metres, so areas come out in km²."""


def main() -> None:
    meta = (
        pl.read_parquet(METADATA_PATH)
        .filter(pl.col("time_series_type") == "Disaggregated Demand")
        .select("time_series_id", "time_series_name", "units", "latitude", "longitude", "area_wkt")
        .sort("time_series_id")
    )
    rows = []
    polys = {}
    for r in meta.iter_rows(named=True):
        g = wkt.loads(r["area_wkt"])
        gm = transform(TO_BNG, g)
        pt = transform(TO_BNG, Point(r["longitude"], r["latitude"]))
        polys[r["time_series_id"]] = gm
        rows.append(
            {
                "time_series_id": r["time_series_id"],
                "name": r["time_series_name"],
                "units": str(r["units"]),
                "geom_type": g.geom_type,
                "is_valid": bool(gm.is_valid),
                "invalid_reason": None if gm.is_valid else shapely.is_valid_reason(gm),
                "area_km2": gm.area / 1e6,
                "point_inside": bool(gm.contains(pt)),
                "point_distance_to_polygon_m": float(gm.distance(pt)),
            }
        )
    out = pl.DataFrame(rows)
    # Pairwise overlaps; ids 12 and 13 are the two Wrangle transformers and share one polygon.
    overlaps = []
    for a, b in itertools.combinations(polys, 2):
        inter = polys[a].intersection(polys[b]).area / 1e6
        if inter > 1e-6:
            overlaps.append(
                {
                    "a": a,
                    "b": b,
                    "overlap_km2": inter,
                    "share_of_smaller": inter / min(polys[a].area, polys[b].area) * 1e6,
                    "identical": polys[a].equals(polys[b]),
                }
            )
    ov = pl.DataFrame(
        overlaps,
        schema={
            "a": pl.Int64,
            "b": pl.Int64,
            "overlap_km2": pl.Float64,
            "share_of_smaller": pl.Float64,
            "identical": pl.Boolean,
        },
    )
    union = shapely.union_all(list(polys.values()))
    hull = union.convex_hull
    sum_area = sum(p.area for i, p in polys.items() if i != 13) / 1e6
    gaps = hull.difference(union)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_DIR / "polygon_check.parquet")
    ov.write_parquet(OUT_DIR / "polygon_overlaps.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(30)
    print(out.drop("invalid_reason"))
    print(ov.sort("share_of_smaller", descending=True))
    print(f"Union area {union.area / 1e6:.1f} km2; sum of areas (Wrangle once) {sum_area:.1f} km2")
    print(f"Union convex hull {hull.area / 1e6:.1f} km2; hull minus union {gaps.area / 1e6:.1f} km2")
    print(f"Union parts: {len(getattr(union, 'geoms', [union]))}")


if __name__ == "__main__":
    main()
