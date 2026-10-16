"""Extract the generators' ENS rows for the 73 runs of `fold_a_every5d_v1`, all 51 members.

The stage 1 pilot stored generator ENS rows for Tuesday runs only, so the best-guess generator
models cannot be applied to the 73 runs from its files. This script reads each run from the ENS
archive, cuts it to the generator cells, resamples to the half-hourly grid with production's
`_upsample_nwp_to_half_hourly` (as `demand_build_ens.py` does), and builds the columns of the
pilot's `ens_features` (the 9 solar columns and the 6 wind columns). `--check` compares the
2025-07-01 run with the pilot's file for the same run.

Output: `work/ens_generators/init=<date>.parquet`, keyed by series, run, valid time, and member.

    uv run python ml-experiments/00036-*/e00036_extract.py [--check]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00036_common as common
import e00036_gen_common as gen
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = common.WORK_DIR / "ens_generators"
ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
STRIDE_DAYS: Final[int] = 5
HORIZON_HOURS: Final[int] = 351
PUBLICATION_DELAY_HOURS: Final[int] = 9
SOURCE: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
}
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *common.KEYS,
    "ensemble_member",
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
    "solar_zenith",
    "solar_azimuth",
)


def validation_runs() -> list[datetime]:
    """Return the 73 validation runs: every fifth day from 2025-07-01 to 2026-06-30."""
    runs = []
    run = FIRST_RUN
    while run <= LAST_RUN:
        runs.append(run)
        run += timedelta(days=STRIDE_DAYS)
    return runs


def generator_cells() -> pl.DataFrame:
    """Return the generators' ids, coordinates, and H3 cells (the coordinates stay in process)."""
    sites = gen.site_table().select("time_series_id", "latitude", "longitude")
    cells = pl.read_parquet(
        gen.FRAMES_DIR / "metadata.parquet", columns=["time_series_id", "h3_res_5"]
    )
    return sites.join(cells, on="time_series_id", how="left", validate="1:1")


def _wind_columns(*, height: str) -> list[pl.Expr]:
    """Return the speed and the bearing's sine and cosine from components `u` and `v`."""
    u, v = pl.col(f"wind_u_{height}m"), pl.col(f"wind_v_{height}m")
    bearing = (pl.arctan2(u, v) * 180.0 / np.pi + 180.0) % 360.0 * np.pi / 180.0
    return [
        (u**2 + v**2).sqrt().alias(f"wind_speed_{height}m"),
        bearing.sin().alias(f"wind_dir_sin_{height}m"),
        bearing.cos().alias(f"wind_dir_cos_{height}m"),
    ]


def run_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's half-hourly generator weather for all 51 members."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    cells = series["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select(
            "h3_index",
            "ensemble_member",
            "valid_time",
            *SOURCE.values(),
            "wind_u_10m",
            "wind_v_10m",
            "wind_u_100m",
            "wind_v_100m",
        )
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            series.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
    init = run + timedelta(hours=PUBLICATION_DELAY_HOURS)
    rows = (
        half_hourly.with_columns(
            *_wind_columns(height="10"),
            *_wind_columns(height="100"),
            power_fcst_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
        )
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=HORIZON_HOURS + 9)
            )
        )
        .rename({v: k for k, v in SOURCE.items() if v != k})
    )
    sun = []
    for row in series.iter_rows(named=True):
        times = rows.filter(pl.col("time_series_id") == row["time_series_id"])["valid_time"]
        stamps = times.unique().sort()
        midpoint = stamps.dt.offset_by("-15m")
        sun.append(
            pl.DataFrame(
                {
                    "valid_time": stamps,
                    "solar_zenith": zenith(
                        stamps=midpoint, latitude=row["latitude"], longitude=row["longitude"]
                    ),
                    "solar_azimuth": azimuth(
                        stamps=midpoint, latitude=row["latitude"], longitude=row["longitude"]
                    ),
                }
            ).with_columns(time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32))
        )
    return (
        rows.join(pl.concat(sun), on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .select(OUTPUT_COLUMNS)
        .sort("time_series_id", "valid_time", "ensemble_member")
    )


def check(*, series: pl.DataFrame) -> None:
    """Compare the 2025-07-01 run with the pilot's file for the same run."""
    mine = run_rows(run=FIRST_RUN, series=series)
    theirs = pl.read_parquet(gen.ENS_VALIDATION_DIR / "init=2025-07-01.parquet")
    keys = ["time_series_id", "valid_time", "ensemble_member"]
    both = mine.join(theirs, on=keys, suffix="_pilot")
    print(f"rows: new {mine.height:,}, pilot {theirs.height:,}, matched {both.height:,}")
    for column in OUTPUT_COLUMNS[len(common.KEYS) + 1 :]:
        diff = (both[column].cast(pl.Float64) - both[f"{column}_pilot"].cast(pl.Float64)).abs()
        print(f"{column}: max abs difference {diff.max():.4g}, mean {diff.mean():.4g}")


def main() -> None:
    """Build every validation run, skipping files that exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    series = generator_cells()
    if args.check:
        check(series=series)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in validation_runs():
        target = OUT_DIR / f"init={run:%Y-%m-%d}.parquet"
        if target.exists():
            continue
        started = time.monotonic()
        rows = run_rows(run=run, series=series)
        rows.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(
            f"{target.name}: {rows.height:,} rows in {time.monotonic() - started:.0f} s", flush=True
        )


if __name__ == "__main__":
    main()
