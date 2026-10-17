"""Build `shapes.parquet`: unclipped plane-of-array fractions at each Primary and farm location.

For each site and tilt the table holds `g<E|S|W><tilt>`, the plane-of-array irradiance over
1000 W/m² for panels facing east, south, and west. A shape clipped at DC:AC ratio `r` is
`min(r * g, 1)`: the AC output per MW of AC capacity.

    POLARS_MAX_THREADS=6 uv run --with holidays python \
        ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_shapes.py
"""

import e00073_common as common
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
    for tilt in common.TILTS:
        for letter, azimuth in common.AZIMUTHS.items():
            columns[f"g{letter}{int(tilt)}"] = (
                plane_of_array_w_m2(sky=sky, tilt_deg=tilt, azimuth_deg=azimuth, tracker=False)
                / 1000.0
            )
    return pl.DataFrame(columns).with_columns(
        pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"), site=pl.lit(site)
    )


def main() -> None:
    """Write the table for the 8 Primaries (demand points) and 6 farms (farm points)."""
    sites = pl.read_parquet(common.SITES_PATH)
    demand = pl.read_parquet(common.CAMS_DEMAND_POINTS_PATH)
    farm = pl.read_parquet(common.CAMS_FARM_POINTS_PATH)
    tables = []
    for row in sites.iter_rows(named=True):
        if row["site"] in common.DIRECTIONAL_PRIMARIES:
            tables.append(
                site_shapes(
                    cams=demand,
                    site=row["site"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                )
            )
        elif row["site"] in common.FARMS:
            tables.append(
                site_shapes(
                    cams=farm,
                    site=row["site"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                )
            )
    out = pl.concat(tables)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(common.SHAPES_PATH)
    print(out.group_by("site").agg(pl.len(), pl.col("gS25").quantile(0.99).alias("p99_gS25")))


if __name__ == "__main__":
    main()
