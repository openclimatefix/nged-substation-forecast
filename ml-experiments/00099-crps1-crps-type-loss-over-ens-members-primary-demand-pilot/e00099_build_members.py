"""Extract a few ENS members per daily run for the Primary series over the training period.

The stage 1 pilot's training table holds only the control member (member 0). Experiment 99 trains
on `MEMBERS_PER_RUN` members, so this script reads the other members of each daily 00:00 UTC run
from 2024-04-01 to 2025-06-30 from the ENS archive, the way experiment 33's
`e00033_build_ens_after_training.py` did for the control member: the archive's H3 cells for the 16
Primary series, resampled to the half-hourly grid by `ml_core`'s `_upsample_nwp_to_half_hourly`,
valid times from the run plus 9 hours to 351 hours later.

Each run keeps the control member and `MEMBERS_PER_RUN - 1` other members drawn with a seed fixed by
the run's date, so the same members are used by every arm and every refit. Runs from 2025-07-01
onward are not built here: the 51-member frames of experiment 48 hold them.

Output: `work/members_train/init=<date>.parquet`, keyed by series, run, valid time, and member.
Existing files are kept. `--after` writes the same columns for the 365 runs of the validation year
to `work/members_after/`, taking the chosen members out of the 51-member frames that experiments 33
and 48 stored, so the training rows of the later refits exist for the same members of every run.

    uv run python ml-experiments/00099-*/e00099_build_members.py [--first 2024-04-01]
        [--last 2025-06-30] [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00099_common as common
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

OUT_DIR: Final[Path] = common.WORK_DIR / "members_train"
AFTER_DIR: Final[Path] = common.WORK_DIR / "members_after"
FIRST_TRAIN_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_TRAIN_RUN: Final[datetime] = datetime(2025, 6, 30, tzinfo=UTC)
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
    return common.ARCHIVE_DIR / f"init_time={stamp}"


def run_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's half-hourly weather for the chosen members of every series."""
    members = common.members_of_run(run=run)
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
        .filter(pl.col("h3_index").is_in(cells) & pl.col("ensemble_member").is_in(members))
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
    origin = common.origin_of(run=run)
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
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
        .cast({"ensemble_member": pl.Int8})
    )


def after_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return the chosen members' rows of a validation-year run from its stored 51-member frame."""
    ids = series["time_series_id"].to_list()
    return (
        pl.read_parquet(common.demand_frame_path(run=run))
        .filter(
            pl.col("time_series_id").is_in(ids)
            & pl.col("ensemble_member").is_in(common.members_of_run(run=run))
        )
        .select(*common.KEYS, "ensemble_member", *WEATHER)
    )


def build_one(*, run: datetime, after: bool = False) -> str:
    """Build one run's file unless it exists, and return a progress line."""
    target = (AFTER_DIR if after else OUT_DIR) / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{target.name}: exists"
    started = time.monotonic()
    series = common.primary_series()
    rows = after_rows(run=run, series=series) if after else run_rows(run=run, series=series)
    common.atomic_write(frame=rows, target=target)
    return f"{target.name}: {rows.height:,} rows in {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build every daily run of the training period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", default=f"{FIRST_TRAIN_RUN:%Y-%m-%d}")
    parser.add_argument("--last", default=f"{LAST_TRAIN_RUN:%Y-%m-%d}")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--after", action="store_true")
    args = parser.parse_args()
    if args.after:
        args.first, args.last = "2025-07-01", "2026-06-30"
    first = datetime.fromisoformat(args.first).replace(tzinfo=UTC)
    last = datetime.fromisoformat(args.last).replace(tzinfo=UTC)
    runs = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_build, runs, [args.after] * len(runs)):
            print(line, flush=True)


def _build(run: datetime, after: bool) -> str:
    return build_one(run=run, after=after)


if __name__ == "__main__":
    main()
