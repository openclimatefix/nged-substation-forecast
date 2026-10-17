"""Build the ENS frames that experiment 00048 forecasts, from the archive.

Two sets of frames, one `init=<date>.parquet` file per daily 00:00 UTC run, all 51 members:

- `work/frames_demand/`: the 20 demand series, for the 292 runs of `fold_a_daily_gap_v1` (the 365
  daily runs of 2025-07-01 to 2026-06-30 less the 73 runs of `fold_a_every5d_v1`, whose frames the
  stage 1 pilot stored and which stay where they are, read only). The columns are those of the
  stored frames.
- `work/frames_generators/`: the 9 generator series, for all 365 runs (no daily generator frames
  are stored). The columns are the technology's stage 1a inputs: the weather columns, the sine and
  cosine of the wind bearing at 10 m and 100 m, and the solar zenith and azimuth of the solar farms.

Each run is cut to the series' H3 cells, resampled to the half-hourly grid by production's
`_upsample_nwp_to_half_hourly` (short-wave energy-conserving, long-wave PCHIP, other variables
linear, the precipitation type forward-filled), and cut to valid times from `power_fcst_init_time`
(the run plus 9 hours) to 351 hours later, as `studies/stage1_pilot/demand_build_ens.py` does for
the demand series. The demand frames also carry the observed power and its four lags from the
stage 1 pilot's observed table, as the stored frames do. One archive read serves both sets for a
run. A run's files are written atomically, so a stopped build resumes.

`--check` builds the 2025-07-01 run and compares it with the stored demand frame and the stored
generator frame of the same run.

    uv run python ml-experiments/00048-*/e00048_build_frames.py [--check] [--smoke] [--workers 5]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00048_common as common
import e00048_gen_common as gen
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.solar import azimuth, zenith

HORIZON_HOURS: Final[float] = 351.0
CORES: Final[int] = 10
ROW_GROUP_ROWS: Final[int] = 50_000
ENS_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "pressure_msl": "pressure_reduced_to_mean_sea_level",
    "precipitation": "precipitation_surface",
    "precipitation_type": "categorical_precipitation_type_surface",
}
"""Which archive column fills each weather column that the archive serves directly."""
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    *ENS_COLUMNS.values(),
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)
LAG_HOURS: Final[dict[str, float]] = {
    "power_lag_24h": 24.0,
    "power_lag_48h": 48.0,
    "power_lag_168h": 168.0,
    "power_lag_336h": 336.0,
}
"""Each lag column and its lag; a lag is null where it is no longer than the lead."""
DEMAND_WEATHER: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "shortwave",
    "longwave",
    "pressure_surface",
    "pressure_msl",
    "precipitation",
    "precipitation_type",
    "wind_speed_10m",
    "wind_speed_100m",
)
RUN_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
)


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return common.ARCHIVE_DIR / f"init_time={stamp}"


def _bearing_columns(*, heights: tuple[str, ...]) -> list[pl.Expr]:
    """Sine and cosine of the wind bearing from components `wind_u_<h>m` and `wind_v_<h>m`.

    The bearing is the direction the wind blows from, in degrees clockwise from north, as the stage
    1 pilot computed it for the ERA5 analysis.
    """
    columns: list[pl.Expr] = []
    for height in heights:
        u, v = pl.col(f"wind_u_{height}m"), pl.col(f"wind_v_{height}m")
        bearing = (pl.arctan2(u, v) * 180.0 / np.pi + 180.0) % 360.0
        columns += [
            (bearing * np.pi / 180.0).sin().alias(f"wind_dir_sin_{height}m"),
            (bearing * np.pi / 180.0).cos().alias(f"wind_dir_cos_{height}m"),
        ]
    return columns


def _half_hourly_rows(*, run: datetime, targets: pl.DataFrame) -> tuple[pl.DataFrame, datetime]:
    """Return one run's half-hourly weather for the target series, from the first valid time on.

    Args:
        run: The run's 00:00 UTC initialisation time.
        targets: One row per series with `time_series_id`, `h3_res_5`, `latitude`, `longitude`.

    Returns:
        The rows keyed by `RUN_KEY_COLUMNS`, with `ENS_COLUMNS`' names, the wind speeds and the
        wind components, `lead_hours`, from `power_fcst_init_time` (excluded) to the horizon; and
        the run's `power_fcst_init_time`.
    """
    cells = targets["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(partition_path(run=run) / "*.parquet")
        .select(SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
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
        .rename({source: name for name, source in ENS_COLUMNS.items()})
    )
    init = common.origin_of(run=run)
    rows = (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind_speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
        )
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=int(HORIZON_HOURS) + 9)
            )
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .filter(pl.col("lead_hours") <= HORIZON_HOURS + 9.0)
    )
    return rows, init


def demand_rows(
    *, rows: pl.DataFrame, series: pl.DataFrame, observed: pl.DataFrame
) -> pl.DataFrame:
    """Return the demand series' frame of one run: weather, observed power, and lags.

    Args:
        rows: `_half_hourly_rows`' output.
        series: `common.series_table()`.
        observed: The stage 1 pilot's observed power with its four lags, by series and valid time.

    Returns:
        The columns of the stored demand frames, sorted by series, valid time, and member.
    """
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    framed = (
        rows.filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        .join(observed, on=["time_series_id", "valid_time"], how="left")
        .with_columns(
            *[
                pl.when(pl.col("lead_hours") >= hours)
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name, hours in LAG_HOURS.items()
            ],
            hour_local=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0,
            day_of_week_local=local.dt.weekday().cast(pl.Float64),
        )
    )
    return framed.select(
        *RUN_KEY_COLUMNS,
        "power",
        *DEMAND_WEATHER,
        "hour_local",
        "day_of_week_local",
        *LAG_HOURS,
        "lead_hours",
    ).sort("time_series_id", "valid_time", "ensemble_member")


def generator_rows(*, rows: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the generator series' frame of one run: stage 1a's inputs for every member.

    Args:
        rows: `_half_hourly_rows`' output.
        sites: `gen.site_table()`.

    Returns:
        The key columns and `gen.GENERATOR_FRAME_COLUMNS`, sorted by series, valid time, and
        member. The solar zenith and azimuth are null for the wind farms.
    """
    stamps = rows["valid_time"].unique().sort()
    midpoints = stamps.dt.offset_by("-15m")
    geometry = pl.concat(
        [
            pl.DataFrame(
                {
                    "time_series_id": pl.Series([row["time_series_id"]] * stamps.len()).cast(
                        pl.Int32
                    ),
                    "valid_time": stamps,
                    "solar_zenith": zenith(
                        stamps=midpoints, latitude=row["latitude"], longitude=row["longitude"]
                    ),
                    "solar_azimuth": azimuth(
                        stamps=midpoints, latitude=row["latitude"], longitude=row["longitude"]
                    ),
                }
            )
            for row in sites.filter(pl.col("technology") == "solar").iter_rows(named=True)
        ]
    )
    return (
        rows.filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .with_columns(*_bearing_columns(heights=("10", "100")))
        .join(geometry, on=["time_series_id", "valid_time"], how="left")
        .select(*RUN_KEY_COLUMNS, *gen.GENERATOR_FRAME_COLUMNS)
        .sort("time_series_id", "valid_time", "ensemble_member")
    )


def build_run(
    *,
    run: datetime,
    series: pl.DataFrame,
    sites: pl.DataFrame,
    observed: pl.DataFrame,
    want_demand: bool,
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    """Return one run's demand frame (if wanted) and generator frame from one archive read."""
    columns = ["time_series_id", "h3_res_5", "latitude", "longitude"]
    targets = pl.concat([sites.select(columns), *([series.select(columns)] if want_demand else [])])
    rows, _init = _half_hourly_rows(run=run, targets=targets)
    demand = demand_rows(rows=rows, series=series, observed=observed) if want_demand else None
    return demand, generator_rows(rows=rows, sites=sites)


def _write_sorted(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a frame atomically, in row groups small enough that a series read skips the rest."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    frame.write_parquet(temporary, row_group_size=ROW_GROUP_ROWS)
    temporary.rename(target)


_SHARED: dict[str, pl.DataFrame] = {}


def _initialise_worker() -> None:
    """Load the series tables and the observed power once per worker process."""
    _SHARED["series"] = common.series_table()
    _SHARED["sites"] = gen.site_table()
    _SHARED["observed"] = pl.read_parquet(common.OBSERVED_PATH)


def _build_and_write(*, run: datetime, demand_dir: Path, generators_dir: Path) -> str:
    """Build one run's missing files in a worker process, and return a progress line."""
    started = time.monotonic()
    demand_target = demand_dir / f"init={run:%Y-%m-%d}.parquet"
    generator_target = generators_dir / f"init={run:%Y-%m-%d}.parquet"
    want_demand = run in set(common.run_dates(subset="gap")) and not demand_target.exists()
    want_generators = not generator_target.exists()
    if not want_demand and not want_generators:
        return f"{run:%Y-%m-%d}: both files exist"
    demand, generators = build_run(
        run=run,
        series=_SHARED["series"],
        sites=_SHARED["sites"],
        observed=_SHARED["observed"],
        want_demand=want_demand,
    )
    if demand is not None:
        _write_sorted(frame=demand, target=demand_target)
    if want_generators:
        _write_sorted(frame=generators, target=generator_target)
    return (
        f"{run:%Y-%m-%d}: demand {0 if demand is None else demand.height:,} rows, "
        f"generators {generators.height:,} rows, {time.monotonic() - started:.0f} s"
    )


def _compare(*, mine: pl.DataFrame, stored: pl.DataFrame, columns: list[str]) -> list[str]:
    """Return report lines comparing two frames' columns on their shared cases."""
    keys = ["time_series_id", "ensemble_member", "valid_time"]
    joined = mine.join(stored, on=keys, how="full", suffix="_stored", coalesce=True)
    both = joined.filter(
        pl.col("power_fcst_init_time").is_not_null()
        & pl.col("power_fcst_init_time_stored").is_not_null()
    )
    lines = [
        f"- Rows: new {mine.height:,}, stored {stored.height:,}; in both {both.height:,}.",
        "",
        "| column | largest absolute difference | mean absolute difference | nulls only in one |",
        "| --- | --- | --- | --- |",
    ]
    for column in columns:
        new = both[column].cast(pl.Float64)
        old = both[f"{column}_stored"].cast(pl.Float64)
        difference = (new - old).abs()
        one_sided = int((new.is_null() != old.is_null()).sum())
        lines.append(
            f"| {column} | {difference.max():.4g} | {difference.mean():.4g} | {one_sided} |"
        )
    return [*lines, ""]


def check_against_stored() -> list[str]:
    """Rebuild the 2025-07-01 run and compare it with the stored demand and generator frames.

    Returns:
        Report lines: for each of the two frames, the row counts and, for each shared column, the
        largest and mean absolute differences.
    """
    run = common.FIRST_RUN
    series = common.series_table()
    sites = gen.site_table()
    observed = pl.read_parquet(common.OBSERVED_PATH)
    demand, generators = build_run(
        run=run, series=series, sites=sites, observed=observed, want_demand=True
    )
    if demand is None:
        msg = "the demand frame was not built"
        raise ValueError(msg)
    stored_demand = pl.read_parquet(common.VALIDATION_DIR / f"init={run:%Y-%m-%d}.parquet").filter(
        pl.col("time_series_id").is_in(series["time_series_id"].to_list())
    )
    stored_generators = pl.read_parquet(gen.ENS_VALIDATION_DIR / f"init={run:%Y-%m-%d}.parquet")
    return [
        f"## New demand frame against the stored frame, run of {run:%Y-%m-%d}",
        "",
        *_compare(
            mine=demand,
            stored=stored_demand,
            columns=[*DEMAND_WEATHER, "power", "hour_local", "day_of_week_local", *LAG_HOURS],
        ),
        f"## New generator frame against the stored frame, run of {run:%Y-%m-%d}",
        "",
        *_compare(
            mine=generators, stored=stored_generators, columns=list(gen.GENERATOR_FRAME_COLUMNS)
        ),
    ]


def main() -> None:
    """Build the missing frames of every run, in parallel across runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the 2025-07-01 run")
    parser.add_argument("--smoke", action="store_true", help="three runs into a scratch folder")
    parser.add_argument("--workers", type=int, default=5, help="processes; each gets 10 // workers")
    args = parser.parse_args()
    if args.check:
        print("\n".join(check_against_stored()))
        return
    demand_dir, generators_dir = common.FRAMES_DEMAND_DIR, common.FRAMES_GENERATORS_DIR
    runs = common.run_dates(subset="all")
    if args.smoke:
        demand_dir = demand_dir.with_name("frames_demand_smoke")
        generators_dir = generators_dir.with_name("frames_generators_smoke")
        runs = [common.FIRST_RUN + timedelta(days=d) for d in (1, 2, 3)]
    os.environ["POLARS_MAX_THREADS"] = str(max(1, CORES // args.workers))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialise_worker) as pool:
        futures = [
            pool.submit(
                _build_and_write, run=run, demand_dir=demand_dir, generators_dir=generators_dir
            )
            for run in runs
        ]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
