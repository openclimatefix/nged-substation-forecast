"""Build `shapes.parquet`: plane-of-array fractions at 25° tilt at every Primary and farm location.

For each site the table holds `gE`, `gS`, and `gW`: the plane-of-array irradiance over 1000 W/m²
for panels facing east, south, and west. The fleet shape is `sum_o mix_o * min(r * g_o, 1)`.

    POLARS_MAX_THREADS=4 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_shapes.py
"""

import e00079_common as common
import polars as pl
from studies.pv_physics import half_hour_sun_and_sky, plane_of_array_w_m2


def site_shapes(
    *, cams: pl.DataFrame, site: str, latitude: float, longitude: float
) -> pl.DataFrame:
    """Return one site's plane-of-array fractions, per half-hour of each reliable hour."""
    reliable = cams.filter(
        (pl.col("site") == site) & (pl.col("reliability") >= common.MIN_RELIABILITY)
    ).sort("time")
    sky = half_hour_sun_and_sky(
        hourly=reliable.select("time", pl.col("ghi_w_m2").cast(pl.Float64)),
        latitude=latitude,
        longitude=longitude,
    )
    columns: dict[str, object] = {"time": sky.half_hour_end_time}
    for letter, azimuth in common.AZIMUTHS.items():
        columns[f"g{letter}"] = (
            plane_of_array_w_m2(sky=sky, tilt_deg=common.TILT, azimuth_deg=azimuth, tracker=False)
            / 1000.0
        )
    return pl.DataFrame(columns).with_columns(
        pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"), site=pl.lit(site)
    )


def main() -> None:
    """Write the table for the 16 Primaries (demand points) and 6 farms (farm points)."""
    sites = pl.read_parquet(common.SITES_PATH)
    demand = pl.read_parquet(common.CAMS_DEMAND_POINTS_PATH)
    farm = pl.read_parquet(common.CAMS_FARM_POINTS_PATH)
    primaries = set(common.MW_PRIMARIES) | set(common.MVA_PRIMARIES)
    tables = []
    for row in sites.iter_rows(named=True):
        if row["site"] in primaries:
            source = demand
        elif row["site"] in common.FARMS:
            source = farm
        else:
            continue
        tables.append(
            site_shapes(
                cams=source, site=row["site"], latitude=row["latitude"], longitude=row["longitude"]
            )
        )
    out = pl.concat(tables)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(common.SHAPES_PATH)
    print(
        out.group_by("site").agg(pl.len(), pl.col("gS").quantile(0.99).alias("p99_gS")).sort("site")
    )


if __name__ == "__main__":
    main()
