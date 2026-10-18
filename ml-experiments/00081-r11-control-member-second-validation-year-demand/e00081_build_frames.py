"""Build the control-member demand frames that experiment 00081 trains on and forecasts.

One `init=<date>.parquet` file per daily 00:00 UTC run of 2021-03-21 to 2024-03-31 (a date with no
backfill file is skipped), holding the 13 Primary substations and ensemble member 0. Every fold's
training rows and forecast runs come from these files. Each run is cut to the series' H3 cells by
`e00081_control_nwp.control_rows`, resampled to the half-hourly grid by production's
`_upsample_nwp_to_half_hourly`, and cut to valid times after `power_fcst_init_time` (the run plus 9
hours) up to 351 hours later, as experiment 00048 does. The frames carry the observed power and its
four lags, each lag nulled where it is no longer than the lead. A file is written atomically, so a
stopped build resumes.

`--check` builds three runs in memory and asserts: lags null where the lag is no longer than the
lead, no missing weather value, a diurnal short-wave profile that peaks near 12:00 UTC, and
temperatures in a plausible range. `--smoke` writes six runs into a scratch folder (three runs that
the smoke fit trains on and the three runs it forecasts).

    uv run python ml-experiments/00081-*/e00081_build_frames.py [--check] [--smoke] [--workers 5]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00081_common as common
import e00081_control_nwp as control
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

CORES: Final[int] = 10
ARCHIVE_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "pressure_msl": "pressure_reduced_to_mean_sea_level",
    "precipitation": "precipitation_surface",
}
"""Which archive column fills each weather column that the archive serves directly."""
WEATHER: Final[tuple[str, ...]] = (
    *ARCHIVE_COLUMNS,
    "wind_speed_10m",
    "wind_speed_100m",
)
"""The frame's weather columns; the backfill has no precipitation type, which no feature uses."""
RUN_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
)
SMOKE_RUNS: Final[tuple[str, ...]] = (
    "2023-06-28",
    "2023-06-29",
    "2023-06-30",
    "2023-07-01",
    "2023-07-02",
    "2023-07-03",
)
CHECK_RUNS: Final[tuple[str, ...]] = ("2022-01-15", "2023-07-01", "2024-01-01")
TEMPERATURE_RANGE_C: Final[tuple[float, float]] = (-20.0, 40.0)
PEAK_HOURS_UTC: Final[tuple[int, ...]] = (11, 12, 13)
SMOKE_FRAMES_DIR: Final[Path] = common.FRAMES_DIR.with_name("frames_smoke")


def _day(*, text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def half_hourly_rows(*, run: datetime, targets: pl.DataFrame) -> pl.DataFrame:
    """Return one run's half-hourly weather for the target series, from the first valid time on.

    Args:
        run: The run's 00:00 UTC initialisation time.
        targets: One row per series with `time_series_id`, `h3_res_5`, `latitude`, `longitude`.

    Returns:
        The rows keyed by `RUN_KEY_COLUMNS`, with the `WEATHER` columns and `lead_hours`, for valid
        times after `power_fcst_init_time` up to `common.HORIZON_HOURS` later.
    """
    cells = [int(cell) for cell in targets["h3_res_5"].unique().to_list()]
    native = (
        control.control_rows(run=run, cells=cells)
        .lazy()
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = targets.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates)
        .collect()
        .rename({source: name for name, source in ARCHIVE_COLUMNS.items()})
    )
    init = common.origin_of(run=run)
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind_speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
        )
        .filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
    )


def demand_rows(*, rows: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return the demand frame of one run: weather, observed power, and lags.

    Args:
        rows: `half_hourly_rows`' output.
        observed: The observed power with its four lags, by series and valid time.

    Returns:
        The key columns, `power`, the `WEATHER` columns, `hour_local`, `day_of_week_local`, the four
        power lags (null where the lead is at least the lag), and `lead_hours`, sorted by series,
        valid time, and member.
    """
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    framed = (
        rows.join(observed, on=["time_series_id", "valid_time"], how="left")
        .with_columns(
            *[
                pl.when(pl.col("lead_hours") >= hours)
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name, hours in common.LAG_HOURS.items()
            ],
            hour_local=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0,
            day_of_week_local=local.dt.weekday().cast(pl.Float64),
        )
        .select(
            *RUN_KEY_COLUMNS,
            "power",
            *WEATHER,
            "hour_local",
            "day_of_week_local",
            *common.LAG_HOURS,
            "lead_hours",
        )
    )
    return framed.sort("time_series_id", "valid_time", "ensemble_member")


def build_run(*, run: datetime, series: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return one run's demand frame for every series."""
    columns = ["time_series_id", "h3_res_5", "latitude", "longitude"]
    return demand_rows(
        rows=half_hourly_rows(run=run, targets=series.select(columns)), observed=observed
    )


def _write_sorted(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a frame atomically, one row group per series so a series read skips the rest."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    per_series = frame.height // frame["time_series_id"].n_unique()
    frame.write_parquet(temporary, row_group_size=per_series)
    temporary.rename(target)


_SHARED: dict[str, pl.DataFrame] = {}


def _initialise_worker() -> None:
    """Load the series table and the observed power once per worker process."""
    _SHARED["series"] = common.series_table()
    _SHARED["observed"] = pl.read_parquet(common.OBSERVED_PATH)


def _build_and_write(*, run: datetime, folder: Path) -> str:
    """Build one run's file in a worker process if it is missing, and return a progress line."""
    started = time.monotonic()
    target = common.frame_path(run=run, folder=folder)
    if target.exists():
        return f"{run:%Y-%m-%d}: file exists"
    frame = build_run(run=run, series=_SHARED["series"], observed=_SHARED["observed"])
    _write_sorted(frame=frame, target=target)
    return f"{run:%Y-%m-%d}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def check_frame(*, frame: pl.DataFrame, run: datetime) -> list[str]:
    """Assert the frame's invariants and return report lines.

    Args:
        frame: One run's frame from `build_run`.
        run: The run's initialisation time.

    Returns:
        Report lines with the numbers behind each assertion.

    Raises:
        ValueError: If a lag is present where the lead is at least the lag, a weather value is
            missing, the mean short-wave profile peaks outside 11 to 13 UTC, or a temperature is
            outside -20 to 40 degrees C.
    """
    lines = [
        f"run {run:%Y-%m-%d}: {frame.height:,} rows, {frame['time_series_id'].n_unique()} series"
    ]
    for name, hours in common.LAG_HOURS.items():
        leaks = frame.filter((pl.col("lead_hours") >= hours) & pl.col(name).is_not_null()).height
        shortest = frame.filter(pl.col(name).is_not_null())["lead_hours"].max()
        if leaks:
            msg = f"{name}: {leaks} rows hold the lag although the lead is at least {hours} h"
            raise ValueError(msg)
        lines.append(f"  {name}: no leak; largest lead holding the lag {shortest} h")
    missing = {
        column: int(frame[column].null_count() + frame[column].is_nan().sum()) for column in WEATHER
    }
    if any(missing.values()):
        msg = f"missing weather values: {missing}"
        raise ValueError(msg)
    profile = (
        frame.group_by(hour=pl.col("valid_time").dt.hour())
        .agg(pl.col("shortwave").mean())
        .sort("shortwave", descending=True)
    )
    peak = int(profile["hour"][0])
    if peak not in PEAK_HOURS_UTC:
        msg = f"short-wave profile peaks at {peak}:00 UTC, expected one of {PEAK_HOURS_UTC}"
        raise ValueError(msg)
    low = float(frame["temperature_2m"].min())  # ty: ignore[invalid-argument-type]
    high = float(frame["temperature_2m"].max())  # ty: ignore[invalid-argument-type]
    if not (TEMPERATURE_RANGE_C[0] < low and high < TEMPERATURE_RANGE_C[1]):
        msg = f"temperature range {low:.1f} to {high:.1f} C is implausible"
        raise ValueError(msg)
    negative = frame.filter(pl.col("shortwave") < 0).height
    if negative:
        msg = f"{negative} rows hold negative short-wave radiation"
        raise ValueError(msg)
    lines.append(
        f"  short-wave peaks at {peak}:00 UTC (mean {float(profile['shortwave'][0]):.0f} W/m2); "
        f"temperature {low:.1f} to {high:.1f} C; power nulls {frame['power'].null_count():,}"
    )
    return lines


def run_check() -> list[str]:
    """Build the three check runs in memory, assert their invariants, and return the report."""
    series = common.series_table()
    observed = pl.read_parquet(common.OBSERVED_PATH)
    raw = control.control_rows(
        run=_day(text=CHECK_RUNS[1]).date(), cells=[int(c) for c in series["h3_res_5"].unique()]
    )
    accumulated = [column for column, _factor in control.ACCUMULATED.values()]
    nulls_elsewhere = raw.filter(pl.col("valid_time") > pl.col("valid_time").min()).select(
        pl.any_horizontal(pl.col(*accumulated, *control.INSTANTANEOUS.values()).is_null()).sum()
    )
    if (
        int(nulls_elsewhere.item())
        or raw.select(pl.col(pl.Float64).is_nan().sum()).sum_horizontal().item()
    ):
        msg = "the raw control rows hold a null or NaN beyond the accumulated step-0 rates"
        raise ValueError(msg)
    lines = ["raw control rows: nulls only in the accumulated step-0 rates"]
    for text in CHECK_RUNS:
        run = _day(text=text)
        lines += check_frame(frame=build_run(run=run, series=series, observed=observed), run=run)
    return lines


def main() -> None:
    """Build the missing frames of every run, in parallel across runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="build three runs and assert")
    parser.add_argument("--smoke", action="store_true", help="six runs into a scratch folder")
    parser.add_argument("--workers", type=int, default=5, help="processes; each gets 10 // workers")
    args = parser.parse_args()
    if args.check:
        print("\n".join(run_check()))
        print("check passed")
        return
    folder = common.FRAMES_DIR
    runs = common.runs_between(
        first=common.BACKFILL_FIRST_RUN, last=common.BACKFILL_LAST_RUN, folder=control.BACKFILL_DIR
    )
    if args.smoke:
        folder = SMOKE_FRAMES_DIR
        runs = [_day(text=text) for text in SMOKE_RUNS]
    workers = min(args.workers, 5)
    os.environ["POLARS_MAX_THREADS"] = str(max(1, CORES // workers))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker) as pool:
        futures = [pool.submit(_build_and_write, run=run, folder=folder) for run in runs]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
