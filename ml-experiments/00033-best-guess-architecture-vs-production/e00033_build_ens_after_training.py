"""Build ENS control-member rows for the daily runs after the training period, for monthly refits.

The monthly-retraining arm of experiment 00033 refits each Primary substation's N1 model on the
first day of each month from 2025-07 to 2026-06, using every daily run whose valid times end at
least 3.5 hours before the refit. The stage 1 pilot's training table stops at 2025-06-30, so this
script extracts the control member of every daily 00:00 UTC run from 2025-07-01 to 2026-06-29 from
the ENS archive in the way the pilot did (the archive's H3 cells for the 20 series, resampled to
the half-hourly grid by `ml_core`'s `_upsample_nwp_to_half_hourly`, valid times from the run plus 9
hours to 351 hours later). `--check` compares the 2025-07-01 control rows with the pilot's
validation file.

Output: `work/ens_after_training/init=<date>.parquet`, keyed by series, run, and valid time, with
the seven weather columns. Existing files are kept.

    uv run python ml-experiments/00033-*/e00033_build_ens_after_training.py [--check]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00033_common as common
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = common.WORK_DIR / "ens_after_training"
ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 29, tzinfo=UTC)
HORIZON_HOURS: Final[int] = 351
SOURCE: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
}
WEATHER: Final[tuple[str, ...]] = (*SOURCE, "wind_speed_10m")


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return ARCHIVE_DIR / f"init_time={stamp}"


def run_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's control-member half-hourly weather for every series."""
    cells = series["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(partition_path(run=run) / "*.parquet")
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
        .filter(pl.col("h3_index").is_in(cells) & (pl.col("ensemble_member") == 0))
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
    init = run + timedelta(hours=common.PUBLICATION_DELAY.total_seconds() // 3600)
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
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
        .select(*common.KEYS, "ensemble_member", *WEATHER)
    )


def check(*, series: pl.DataFrame) -> None:
    """Compare the 2025-07-01 control rows with the pilot's validation file."""
    mine = run_rows(run=FIRST_RUN, series=series)
    theirs = pl.read_parquet(common.VALIDATION_DIR / "init=2025-07-01.parquet").filter(
        pl.col("ensemble_member") == 0
    )
    both = mine.join(theirs, on=["time_series_id", "valid_time"], suffix="_pilot")
    print(f"rows: new {mine.height:,}, pilot control {theirs.height:,}, matched {both.height:,}")
    for column in WEATHER:
        diff = (both[column].cast(pl.Float64) - both[f"{column}_pilot"].cast(pl.Float64)).abs()
        print(f"{column}: max abs difference {diff.max():.4g}, mean {diff.mean():.4g}")


def main() -> None:
    """Build every daily run, skipping files that exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    series = common.series_table()
    if args.check:
        check(series=series)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = FIRST_RUN
    while run <= LAST_RUN:
        target = OUT_DIR / f"init={run:%Y-%m-%d}.parquet"
        if not target.exists():
            started = time.monotonic()
            rows = run_rows(run=run, series=series)
            rows.write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
            print(
                f"{target.name}: {rows.height:,} rows in {time.monotonic() - started:.0f} s",
                flush=True,
            )
        run += timedelta(days=1)


if __name__ == "__main__":
    main()
