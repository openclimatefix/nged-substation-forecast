"""Build the 51-member ENS frames of the 7 MW sites for experiment 00111's early runs.

The 292 scored runs of 2025-07-01 to 2026-06-30 (`fold_a_daily_gap_v1`) already have frames from
experiment 00048, and the 73 runs on the every-fifth-day stride have the stage 1 pilot's. This
script builds the missing ones: every daily 00:00 UTC run from 2024-04-01 to 2025-06-30 (456 runs),
which the tuning and replicate months, the analysis proxy, and the XGBoost bars' training rows use.
The frames have the columns of experiment 00048's demand frames, cut to the 7 MW sites of
experiment 00103 (roles `mw_long` and `mw_short`), so one reader serves every run.

Each run is cut to the sites' H3 cells, resampled to the half-hourly grid by production's
`_upsample_nwp_to_half_hourly`, and cut to valid times from the run's `power_fcst_init_time` (the
run plus 9 hours) to 351 hours later, as experiment 00048 does. Observed power and its four lags
come from the stage 1 pilot's observed table, with a lag null where it is no longer than the lead.
A run's file is written atomically, so a stopped build resumes.

**Do not feed the `power_lag_*` columns to a model as they stand.** They are nulled only where the
lead is no longer than the lag, as experiment 00048 does, which ignores the 3.5-hour telemetry
delay: a lag at a lead within 3.5 hours of it reads a reading a live service would not yet have.
Experiment 00087 rebuilds its lags with the delay. The columns are kept so that the frames match
experiment 00048's stored frames exactly.

The script reads the archive's Delta table as raw parquet files, bypassing the Delta log, so the
validation fails on duplicate keys if a partition ever holds old and new files together.

    uv run python ml-experiments/00111-*/e00111_build_frames.py [--check] [--validate] [--smoke]
        [--workers 3]

`--check` rebuilds the 2025-07-02 run, which experiment 00048 stored, and prints the largest
difference per column at the 7 sites. `--validate` checks every written file and that every daily
run from 2024-04-01 to 2026-06-30 has a frame in one of the three folders that hold them.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import h3
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.dp_features import default_input_paths
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SLUG: Final[str] = "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
FRAMES_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work" / "frames_demand"
STORED_00048_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00048-weekly-retraining-confirmation-and-extension-demand-generators"
    / "work"
    / "frames_demand"
)
OBSERVED_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/observed.parquet"
)
ARCHIVE_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2025, 6, 30, tzinfo=UTC)
SCORED_LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
STAGE1_EVERY5D_DIR: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/ens_every5d/val"
)
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
HORIZON_HOURS: Final[int] = 351
H3_RESOLUTION: Final[int] = 5
SITE_ROLES: Final[tuple[str, ...]] = ("mw_long", "mw_short")
N_SITES: Final[int] = 7
ENSEMBLE_SIZE: Final[int] = 51
CORES: Final[int] = 6
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


def decision_sites() -> pl.DataFrame:
    """Return the 7 MW sites: `site`, `time_series_id`, `latitude`, `longitude`, and `h3_res_5`.

    The sites are the `mw_long` role (six sites) and the `mw_short` role (D18).

    The coordinates stay inside this process and the frames carry only `time_series_id`.
    """
    paths = default_input_paths()
    sites = pl.read_parquet(paths.sites, columns=["site", "time_series_id", "role"])
    table = pl.read_parquet(paths.series_table, columns=["site", "latitude", "longitude"])
    kept = sites.filter(pl.col("role").is_in(SITE_ROLES)).join(table, on="site", how="left")
    return kept.with_columns(
        h3_res_5=pl.struct("latitude", "longitude").map_elements(
            lambda row: h3.str_to_int(
                h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)
            ),
            return_dtype=pl.Int64,
        ),
        time_series_id=pl.col("time_series_id").cast(pl.Int32),
    ).select("site", "time_series_id", "latitude", "longitude", "h3_res_5")


def run_dates() -> list[datetime]:
    """Return the 456 daily 00:00 UTC runs from 2024-04-01 to 2025-06-30."""
    days = (LAST_RUN - FIRST_RUN).days + 1
    return [FIRST_RUN + timedelta(days=i) for i in range(days)]


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return ARCHIVE_DIR / f"init_time={stamp}"


def build_run(*, run: datetime, sites: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return one run's demand frame at the sites: weather, observed power, and lags.

    Args:
        run: The run's 00:00 UTC initialisation time.
        sites: `decision_sites()`.
        observed: The stage 1 pilot's observed power with its four lags.

    Returns:
        The columns of experiment 00048's demand frames, sorted by series, valid time, and member.
    """
    cells = sites["h3_res_5"].unique().to_list()
    targets = sites.select("time_series_id", "h3_res_5", "latitude", "longitude")
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
    origin = run + PUBLICATION_DELAY
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind_speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .filter((pl.col("lead_hours") > 0.0) & (pl.col("lead_hours") <= HORIZON_HOURS))
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
        .select(
            *RUN_KEY_COLUMNS,
            "power",
            *DEMAND_WEATHER,
            "hour_local",
            "day_of_week_local",
            *LAG_HOURS,
            "lead_hours",
        )
        .sort("time_series_id", "valid_time", "ensemble_member")
    )


def validate_frame(*, frame: pl.DataFrame, run: datetime) -> dict[str, int]:
    """Raise if a run's frame is short, duplicated, or has a missing or implausible weather value.

    Args:
        frame: One run's frame.
        run: The run's 00:00 UTC initialisation time.

    Returns:
        The number of null `shortwave` values, which the source holds in a few runs, by column.

    Raises:
        ValueError: If a site or member is missing, a key repeats, the leads are outside
            0 to 351 hours, a temperature is missing or implausible, or a shortwave value is
            negative or above 1,400 W/m2.
    """
    keys = ["time_series_id", "ensemble_member", "valid_time"]
    problems = []
    if frame["time_series_id"].n_unique() != N_SITES:
        problems.append(f"{frame['time_series_id'].n_unique()} sites, not {N_SITES}")
    if frame["ensemble_member"].n_unique() != ENSEMBLE_SIZE:
        problems.append(f"{frame['ensemble_member'].n_unique()} members, not {ENSEMBLE_SIZE}")
    if frame.select(keys).is_duplicated().any():
        problems.append("duplicate (series, member, valid time) keys")
    if frame.height % (N_SITES * ENSEMBLE_SIZE):
        problems.append(f"{frame.height} rows is not a whole number of half-hours per member")
    if frame["lead_hours"].min() != 0.5 or frame["lead_hours"].max() > HORIZON_HOURS:
        problems.append("the leads do not run from 0.5 hours to at most 351 hours")
    temperature = frame["temperature_2m"]
    if temperature.null_count() or not -30.0 < temperature.min() < temperature.max() < 45.0:
        problems.append("a temperature is missing or outside -30 to 45 degrees Celsius")
    shortwave = frame["shortwave"].drop_nulls()
    if shortwave.min() < 0.0 or shortwave.max() > 1400.0:
        problems.append("a shortwave value is negative or above 1,400 W/m2")
    if problems:
        msg = f"the {run:%Y-%m-%d} frame failed validation: " + "; ".join(problems)
        raise ValueError(msg)
    return {"shortwave_nulls": frame["shortwave"].null_count()}


def validate_directory() -> str:
    """Validate every written file and the coverage of daily runs 2024-04-01 to 2026-06-30.

    Returns:
        A report: the files checked, the runs with null shortwave values, and any run with no frame.

    Raises:
        ValueError: If a file fails `validate_frame` or a daily run has no frame in the three
            folders that hold them.
    """
    holes = []
    for path in sorted(FRAMES_DIR.glob("init=*.parquet")):
        run = datetime.strptime(path.stem.removeprefix("init="), "%Y-%m-%d").replace(tzinfo=UTC)
        outcome = validate_frame(frame=pl.read_parquet(path), run=run)
        if outcome["shortwave_nulls"]:
            holes.append(f"{run:%Y-%m-%d}: {outcome['shortwave_nulls']} null shortwave values")
    folders = [FRAMES_DIR, STORED_00048_DIR, STAGE1_EVERY5D_DIR]
    missing = []
    day = FIRST_RUN
    while day <= SCORED_LAST_RUN:
        if not any((folder / f"init={day:%Y-%m-%d}.parquet").exists() for folder in folders):
            missing.append(f"{day:%Y-%m-%d}")
        day += timedelta(days=1)
    if missing:
        msg = f"{len(missing)} daily runs have no frame in any folder, the first on {missing[0]}"
        raise ValueError(msg)
    return "\n".join(
        [f"all files valid; {len(holes)} runs with null shortwave values:", *holes[:20]]
    )


_SHARED: dict[str, pl.DataFrame] = {}


def _initialise_worker() -> None:
    """Load the site table and the observed power once per worker process."""
    _SHARED["sites"] = decision_sites()
    _SHARED["observed"] = pl.read_parquet(OBSERVED_PATH)


def _build_and_write(*, run: datetime, directory: Path) -> str:
    """Build one run's file in a worker process if it is missing, and return a progress line."""
    target = directory / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    frame = build_run(run=run, sites=_SHARED["sites"], observed=_SHARED["observed"])
    validate_frame(frame=frame, run=run)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary, row_group_size=ROW_GROUP_ROWS)
    temporary.rename(target)
    return f"{run:%Y-%m-%d}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def check_against_stored() -> str:
    """Rebuild the 2025-07-02 run and compare it with experiment 00048's stored frame.

    Returns:
        A report: row counts and, for each shared column, the largest absolute difference.
    """
    run = datetime(2025, 7, 2, tzinfo=UTC)
    sites = decision_sites()
    mine = build_run(run=run, sites=sites, observed=pl.read_parquet(OBSERVED_PATH))
    stored = pl.read_parquet(STORED_00048_DIR / f"init={run:%Y-%m-%d}.parquet").filter(
        pl.col("time_series_id").is_in(sites["time_series_id"].to_list())
    )
    keys = ["time_series_id", "ensemble_member", "valid_time"]
    joined = mine.join(stored, on=keys, how="inner", suffix="_stored")
    lines = [f"rows: new {mine.height:,}, stored {stored.height:,}, shared {joined.height:,}"]
    for column in [*DEMAND_WEATHER, "power", *LAG_HOURS, "lead_hours"]:
        new = joined[column].cast(pl.Float64)
        old = joined[f"{column}_stored"].cast(pl.Float64)
        difference = (new - old).abs().max()
        lines.append(f"{column}: largest absolute difference {difference}")
    return "\n".join(lines)


def main() -> None:
    """Build the missing frames of every run, in parallel across runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="three runs into a scratch folder")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.check:
        print(check_against_stored())
        return
    if args.validate:
        print(validate_directory())
        return
    directory = FRAMES_DIR
    runs = run_dates()
    if args.smoke:
        directory = directory.with_name("frames_demand_smoke")
        runs = runs[:3]
    os.environ["POLARS_MAX_THREADS"] = str(max(1, CORES // args.workers))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialise_worker) as pool:
        futures = [pool.submit(_build_and_write, run=run, directory=directory) for run in runs]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
