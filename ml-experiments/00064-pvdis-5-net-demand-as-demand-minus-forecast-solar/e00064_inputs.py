"""Build the solar inputs that experiment 00064 adds to experiment 00046's, under `work/inputs/`.

Experiment 00046's solar folder (CAMS `u`, climatologies, calibration slopes, validation
anomalies) is read, never written. This script writes:

- `train_shape.parquet`: for each training row (series, run, valid time; the ENS control member's
  rows of experiment 00003's `train.parquet`) the ensemble-mean calibrated anomaly
  (`anomaly_mean_calibrated`, the lead-band slope times the ensemble-mean ENS anomaly), the CAMS
  climatology (`clim_cams`) and their sum `shape_calibrated`, plus the CAMS `u` and anomaly at the
  valid time, the month, half-hour, lead band, and the daylight flag.
- `t_ref.parquet`: per series, the training-window mean over daylight rows of the cell temperature
  (the control member's 2 m temperature plus 25 K times `shape_calibrated`), the reference of the
  exploratory temperature derating `sf_temp`. Training files hold the control member only, so the
  reference uses the control member's temperature.
- `val_keys.parquet`: each validation run's distinct (series, run, valid time) with the
  ensemble-mean calibrated anomaly, read from experiment 00046's `ens_val` files.
- `lag_runs/run=*.parquet`: for each daily 00:00 UTC ENS run the arm `b` needs, the ensemble-mean
  `u` of every valid time in the first 36 hours, per series.
- `lag_shapes_train.parquet`, `lag_shapes_val.parquet`: the unit PV shape at each of the three
  lagged times of every training and validation row, for arm `b`'s gross lags, with the source of
  each value. The shape comes from CAMS when the lagged time is more than 24 hours before the
  origin; otherwise from the ensemble-mean `u` of the 00:00 UTC run of the origin's day when that
  run covers the lagged time, else of the previous day's run; otherwise it is missing. A row with a
  missing shape is dropped from `b`'s training and has a missing gross lag at inference (a missing
  value that XGBoost handles natively; a validation row is never dropped, so the arms share keys).
  The counts of each source per lag column are printed.
- `placebo_train.parquet`, `placebo_val.parquet`: the placebo twins' anomaly. Within a series,
  local calendar month, UTC half-hour, and lead band, the ensemble-mean calibrated anomaly of each
  forecast run is exchanged for that of another run by a seeded permutation (without replacement),
  separately in training and validation.

    uv run python ml-experiments/00064-*/e00064_inputs.py --part train_shape
    uv run python ml-experiments/00064-*/e00064_inputs.py --part lag_runs [--runs N]
    uv run python ml-experiments/00064-*/e00064_inputs.py --part lag_shapes
    uv run python ml-experiments/00064-*/e00064_inputs.py --part placebo
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00064_common as common
import numpy as np
import polars as pl
import pvlib
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.pv_physics import SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
SHORTWAVE_SOURCE: Final[str] = "downward_short_wave_radiation_flux_surface"
LAG_RUN_HOURS: Final[int] = 36
"""Hours of each run kept for the lag shapes.

An ENS-sourced lagged time is at most 24 hours before the origin, which is 9 hours after the run's
initialisation, so it lies within the first 9 hours of the origin's run and the 24 to 33 hours of
the previous day's run. Keeping 36 hours covers both."""
LAG_RUNS_DIR: Final[Path] = common.INPUTS_DIR / "lag_runs"
PUBLICATION_HOURS: Final[int] = 9
GRID_START_MICROSECONDS: Final[int] = int(common.GRID_START.timestamp() * 1e6)
HALF_HOUR_MICROSECONDS: Final[int] = 1_800_000_000


def sites_table() -> pl.DataFrame:
    """Return the verdict series' `time_series_id`, `site`, `h3_res_5`, and coordinates."""
    return common.series_table().filter(pl.col("site").is_in(list(common.SITES)))


def train_shape_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the calibrated ENS shape and the CAMS values of every training row of the series.

    Args:
        series: `sites_table()`.

    Returns:
        Columns `time_series_id` (Int32), `site`, `power_fcst_init_time`, `valid_time`, `month`
        (Int8), `hh` (Int8), `band` (Int64), `u_mean` (the raw ensemble-mean ENS `u`),
        `anomaly_mean_calibrated`, `clim_cams`, `shape_calibrated` (their sum), `anomaly_cams`,
        `u_cams` (null where CAMS is missing), and `daylight` (false where the cell has no CAMS
        climatology). A missing or NaN anomaly or climatology counts as zero, so the shift of that
        row is zero.

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    ens = pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((common.E46_SOLAR_DIR / "ens_train").glob("init=*.parquet"))
        ]
    ).filter(pl.col("site").is_in(list(common.SITES)))
    common.require_training_window(times=ens["valid_time"], what="ENS training rows")
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "clim_cams", "clim_ens", "daylight"
    )
    slopes = pl.read_parquet(common.E46_SOLAR_DIR / "calibration.parquet").select(
        "site", "band", "slope"
    )
    cams_anomaly = pl.read_parquet(common.E46_SOLAR_DIR / "cams_anomaly.parquet")
    cams_u = pl.read_parquet(common.E46_SOLAR_DIR / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    return (
        common.with_cell(frame=ens, time_column="valid_time")
        .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
        .join(climatology, on=["site", "month", "hh"], how="left", validate="m:1")
        .join(slopes, on=["site", "band"], how="left", validate="m:1")
        .join(cams_anomaly, on=["site", "valid_time"], how="left", validate="m:1")
        .join(cams_u, on=["site", "valid_time"], how="left", validate="m:1")
        .join(series.select("site", "time_series_id"), on="site", how="left", validate="m:1")
        .with_columns(
            anomaly_mean_calibrated=(pl.col("slope") * (pl.col("u_mean") - pl.col("clim_ens")))
            .fill_nan(0.0)
            .fill_null(0.0),
            clim_cams=pl.col("clim_cams").fill_null(0.0),
            daylight=pl.col("daylight").fill_null(value=False),
        )
        .with_columns(shape_calibrated=pl.col("clim_cams") + pl.col("anomaly_mean_calibrated"))
        .select(
            "time_series_id",
            "site",
            "power_fcst_init_time",
            "valid_time",
            "month",
            "hh",
            "band",
            "u_mean",
            "anomaly_mean_calibrated",
            "clim_cams",
            "shape_calibrated",
            "anomaly_cams",
            "u_cams",
            "daylight",
        )
    )


def reference_temperature(*, shape: pl.DataFrame) -> pl.DataFrame:
    """Return, per series, the training-window mean cell temperature over daylight rows.

    Args:
        shape: `train_shape_table()`.

    Returns:
        Columns `site` and `t_ref` (the mean of the control member's 2 m temperature plus 25 K
        times `shape_calibrated`).
    """
    temperature = pl.read_parquet(
        common.ENS_TRAIN_PATH,
        columns=["time_series_id", "power_fcst_init_time", "valid_time", "temperature_2m"],
    )
    common.require_training_window(times=temperature["valid_time"], what="training temperatures")
    return (
        shape.filter(pl.col("daylight"))
        .join(
            temperature,
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            cell_temperature=pl.col("temperature_2m")
            + common.CELL_TEMPERATURE_RISE_K * pl.col("shape_calibrated")
        )
        .group_by("site")
        .agg(t_ref=pl.col("cell_temperature").mean())
        .sort("site")
    )


def validation_keys(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every validation run's distinct keys with the ensemble-mean calibrated anomaly.

    Args:
        series: `sites_table()`.

    Returns:
        Columns `time_series_id`, `site`, `power_fcst_init_time`, `valid_time`, `month`, `hh`,
        `band`, and `anomaly_mean_calibrated`, from experiment 00046's `ens_val` files (member 0's
        rows; the ensemble-mean anomaly is the same for every member).
    """
    ens = pl.concat(
        [
            pl.read_parquet(
                path,
                columns=[
                    "site",
                    "power_fcst_init_time",
                    "valid_time",
                    "ensemble_member",
                    "anomaly_mean_calibrated",
                ],
            ).filter(pl.col("site").is_in(list(common.SITES)) & (pl.col("ensemble_member") == 0))
            for path in sorted((common.E46_SOLAR_DIR / "ens_val").glob("init=*.parquet"))
        ]
    ).drop("ensemble_member")
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    return (
        common.with_cell(frame=ens, time_column="valid_time")
        .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
        .join(series.select("site", "time_series_id"), on="site", validate="m:1")
        .with_columns(pl.col("anomaly_mean_calibrated").fill_nan(0.0).fill_null(0.0))
        .select(
            "time_series_id",
            "site",
            "power_fcst_init_time",
            "valid_time",
            "month",
            "hh",
            "band",
            "anomaly_mean_calibrated",
        )
    )


def u_from_half_hour_ghi(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape of half-hour mean irradiance, the chain CAMS goes through.

    The sun is at the midpoint of the half-hour ending at each stamp, the Erbs correlation splits
    the irradiance, and the generic fleet turns it into power per MW of AC capacity.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `u`, shape (k, n), NaN where `ghi` is NaN.
    """
    copies = ghi.shape[0]
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).ravel().astype(np.float64)
    split = pvlib.irradiance.erbs(
        ghi=flat, zenith=np.tile(zenith, copies), datetime_or_doy=np.tile(day, copies)
    )
    sky = SunAndSky(
        half_hour_end_time=np.tile(stamps.to_numpy(), copies),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=np.tile(zenith, copies),
        sun_azimuth_deg=np.tile(sun_azimuth, copies),
    )
    shape = plant_power_mw(sky=sky, parameters=common.FLEET).reshape(copies, -1)
    return np.where(unknown, np.nan, shape)


def read_early_shortwave(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return the first `LAG_RUN_HOURS` hours of one archive run's `shortwave`, every member.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `sites_table()`.

    Returns:
        Columns `time_series_id`, `valid_time`, `ensemble_member`, and `shortwave` (W/m2, half-hour
        mean ending at `valid_time`), for valid times after the run's initialisation time and at
        most `LAG_RUN_HOURS` hours later.
    """
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select("h3_index", "ensemble_member", "valid_time", SHORTWAVE_SOURCE)
        .filter(
            pl.col("h3_index").is_in(series["h3_res_5"].unique().to_list())
            & (pl.col("valid_time") <= run + timedelta(hours=LAG_RUN_HOURS + 12))
        )
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
    return (
        half_hourly.rename({SHORTWAVE_SOURCE: "shortwave"})
        .filter(
            (pl.col("valid_time") > run)
            & (pl.col("valid_time") <= run + timedelta(hours=LAG_RUN_HOURS))
        )
        .select("time_series_id", "valid_time", "ensemble_member", "shortwave")
    )


def build_lag_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's ensemble-mean `u` per series and valid time.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `sites_table()`.

    Returns:
        Columns `site`, `run_time`, `valid_time`, and `u_mean` (Float32).
    """
    rows = read_early_shortwave(run=run, series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        own = rows.filter(pl.col("time_series_id") == row["time_series_id"]).sort(
            "valid_time", "ensemble_member"
        )
        times = own["valid_time"].unique(maintain_order=True)
        members = own["ensemble_member"].unique().sort()
        if own.height != times.len() * members.len():
            msg = f"run {run}: every member must hold every valid time"
            raise ValueError(msg)
        ghi = own["shortwave"].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
        u = u_from_half_hour_ghi(
            stamps=times, ghi=ghi, latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(
            pl.DataFrame(
                {
                    "site": row["site"],
                    "run_time": run,
                    "valid_time": times,
                    "u_mean": u.mean(axis=0).astype(np.float32),
                }
            ).with_columns(pl.col("run_time").dt.replace_time_zone("UTC"))
        )
    return pl.concat(pieces)


def needed_runs(*, train_origins: pl.Series, validation_origins: pl.Series) -> list[datetime]:
    """Return the 00:00 UTC runs that arm `b`'s lag shapes can read.

    Args:
        train_origins: Distinct forecast origins of the training rows.
        validation_origins: Distinct forecast origins of the validation rows.

    Returns:
        Each origin's run (the origin minus 9 hours) and the run of the day before, sorted.
    """
    runs: set[datetime] = set()
    for origin in [*train_origins.to_list(), *validation_origins.to_list()]:
        today = origin - timedelta(hours=PUBLICATION_HOURS)
        runs |= {today, today - timedelta(days=1)}
    return sorted(runs)


def build_lag_runs(*, series: pl.DataFrame, runs: list[datetime], limit: int | None) -> None:
    """Build (or resume) the per-run ensemble-mean `u` files; skip runs missing from the archive.

    Args:
        series: `sites_table()`.
        runs: `needed_runs()`.
        limit: Only the first this many runs when given.
    """
    LAG_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for run in runs[:limit]:
        target = LAG_RUNS_DIR / f"run={run:%Y-%m-%d}.parquet"
        stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
        if target.exists():
            continue
        if not (ARCHIVE_DIR / f"init_time={stamp}").exists():
            print(f"{run:%Y-%m-%d}: not in the NWP archive", flush=True)
            continue
        started = time.monotonic()
        table = build_lag_run(run=run, series=series)
        common.write_atomically(frame=table, target=target)
        print(f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s")


def cams_grids(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each series' CAMS `u` on the dense grid, NaN where CAMS is missing."""
    cams_u = pl.read_parquet(common.E46_SOLAR_DIR / "cams_u.parquet")
    grids: dict[str, np.ndarray] = {}
    for site in series["site"]:
        own = cams_u.filter(pl.col("site") == site)
        grid = np.full(common.grid_times().len(), np.nan, dtype=np.float32)
        index = np.asarray(common.grid_index(times=own["time"]))
        inside = (index >= 0) & (index < grid.size)
        grid[index[inside]] = own["u"].to_numpy()[inside]
        grids[site] = grid
    return grids


def lagged_shape(
    *,
    keys: pl.DataFrame,
    lag_index: np.ndarray,
    cams_grid: np.ndarray,
    lag_runs: pl.DataFrame,
) -> pl.DataFrame:
    """Return one lag column's unit PV shape and its source for one series' rows.

    Args:
        keys: The series' rows with `site`, `power_fcst_init_time`, and `valid_time`.
        lag_index: Grid index of each row's lagged half-hour.
        cams_grid: The series' CAMS `u` on the grid.
        lag_runs: The `lag_runs` table (`site`, `run_time`, `valid_time`, `u_mean`).

    Returns:
        Columns `shape` (Float32, null where missing) and `source` (`cams`, `ens_today`,
        `ens_yesterday`, or `missing`), in the order of `keys`.
    """
    lagged_time = pl.Series(
        GRID_START_MICROSECONDS + lag_index.astype(np.int64) * HALF_HOUR_MICROSECONDS
    )
    lagged_time = lagged_time.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    origin_index = np.asarray(common.grid_index(times=keys["power_fcst_init_time"]))
    far = (origin_index - lag_index) > common.LAG_SHAPE_ENS_HOURS * 2
    cams_values = pl.Series(common.take(values=cams_grid, index=lag_index)).fill_nan(None)
    frame = pl.DataFrame(
        {
            "site": keys["site"],
            "lagged_time": lagged_time,
            "origin": keys["power_fcst_init_time"],
            "far": far,
            "cams": cams_values,
        }
    ).with_columns(run_today=pl.col("origin") - pl.duration(hours=PUBLICATION_HOURS))
    frame = frame.with_columns(run_previous=pl.col("run_today") - pl.duration(days=1)).with_columns(
        run_time=pl.when(pl.col("lagged_time") > pl.col("run_today"))
        .then(pl.col("run_today"))
        .when(pl.col("lagged_time") > pl.col("run_previous"))
        .then(pl.col("run_previous"))
    )
    joined = frame.join(
        lag_runs,
        left_on=["site", "run_time", "lagged_time"],
        right_on=["site", "run_time", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    return joined.select(
        shape=pl.when(pl.col("far")).then(pl.col("cams")).otherwise(pl.col("u_mean")),
        source=pl.when(pl.col("far"))
        .then(
            pl.when(pl.col("cams").is_not_null()).then(pl.lit("cams")).otherwise(pl.lit("missing"))
        )
        .when(pl.col("u_mean").is_null())
        .then(pl.lit("missing"))
        .when(pl.col("run_time") == pl.col("run_today"))
        .then(pl.lit("ens_today"))
        .otherwise(pl.lit("ens_yesterday")),
    ).with_columns(pl.col("shape").cast(pl.Float32))


def lag_shapes_table(
    *, keys: pl.DataFrame, series: pl.DataFrame, lag_runs: pl.DataFrame
) -> pl.DataFrame:
    """Return the three lag shapes and their sources for every row of `keys`.

    Args:
        keys: Rows with `time_series_id`, `site`, `power_fcst_init_time`, and `valid_time`.
        series: `sites_table()`.
        lag_runs: The concatenated `lag_runs` files.

    Returns:
        `time_series_id`, `power_fcst_init_time`, `valid_time`, then for each `LAG_COLUMNS` name
        `shape_<name>` (Float32) and `source_<name>` (String).
    """
    grids = cams_grids(series=series)
    pieces = []
    for site in common.SITES:
        own = keys.filter(pl.col("site") == site)
        indices = common.lag_indices(
            target=np.asarray(common.grid_index(times=own["valid_time"])),
            origin=np.asarray(common.grid_index(times=own["power_fcst_init_time"])),
        )
        columns: dict[str, pl.Series] = {}
        for k, name in enumerate(common.LAG_COLUMNS):
            result = lagged_shape(
                keys=own, lag_index=indices[:, k], cams_grid=grids[site], lag_runs=lag_runs
            )
            columns[f"shape_{name}"] = result["shape"]
            columns[f"source_{name}"] = result["source"]
        pieces.append(
            own.select("time_series_id", "power_fcst_init_time", "valid_time").with_columns(
                **columns
            )
        )
    return pl.concat(pieces)


def print_source_counts(*, table: pl.DataFrame, split: str) -> None:
    """Print how many rows take each lag column's shape from each source."""
    print(f"Source of the lag shape in the {split} rows ({table.height:,} rows)")
    counts = pl.concat(
        [
            table.group_by(source=pl.col(f"source_{name}"))
            .agg(rows=pl.len())
            .with_columns(lag=pl.lit(name))
            for name in common.LAG_COLUMNS
        ]
    )
    print(counts.pivot(on="source", index="lag", values="rows").fill_null(0))


def placebo_table(*, keys: pl.DataFrame, stream: int) -> pl.DataFrame:
    """Exchange each row's ensemble-mean calibrated anomaly with that of another row of its group.

    The group is the series, local calendar month, UTC half-hour, and lead band. Within a group the
    rows are ordered by valid time and each takes the anomaly of the row a random number of places
    along (a cyclic shift between a quarter and three quarters of the group, drawn by a generator
    seeded with `PLACEBO_SEED` and `stream`). Every anomaly is used exactly once, so the
    distribution is kept. A plain random permutation would sometimes pair two runs that forecast
    the same valid time, which would keep the timing of the sunshine; the shift pairs rows of
    different days.

    Args:
        keys: Rows with `site`, `month`, `hh`, `band`, `power_fcst_init_time`, `valid_time`, and
            `anomaly_mean_calibrated`, one row per run and valid time.
        stream: 0 for training and 1 for validation, so each has its own random stream.

    Returns:
        `time_series_id`, `power_fcst_init_time`, `valid_time`, and `anomaly_mean_calibrated_twin`,
        the donor row's anomaly.
    """
    ordered = keys.sort("site", "month", "hh", "band", "valid_time", "power_fcst_init_time")
    group = ordered.select(pl.struct("site", "month", "hh", "band").rank("dense").alias("group"))[
        "group"
    ].to_numpy()
    values = ordered["anomaly_mean_calibrated"].to_numpy()
    rng = np.random.default_rng([common.PLACEBO_SEED, stream])
    donor = np.arange(len(values))
    starts = np.flatnonzero(np.diff(group, prepend=-1))
    for begin, end in zip(starts, [*starts[1:], len(values)], strict=True):
        size = end - begin
        if size > 1:
            shift = int(rng.integers(max(size // 4, 1), max(3 * size // 4, 1) + 1))
            donor[begin:end] = begin + (np.arange(size) + shift) % size
    own_date = ordered["valid_time"].dt.date().to_numpy()
    same_date = float((own_date == own_date[donor]).mean())
    print(
        f"placebo stream {stream}: {len(values):,} rows, "
        f"{float((donor == np.arange(len(values))).mean()):.4f} keep their own value, "
        f"{same_date:.4f} take a value from the same valid date"
    )
    return ordered.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        anomaly_mean_calibrated_twin=pl.Series(values[donor]),
    )


def read_lag_runs() -> pl.DataFrame:
    """Return every built `lag_runs` file as one table."""
    files = sorted(LAG_RUNS_DIR.glob("run=*.parquet"))
    if not files:
        msg = f"no lag run files under {LAG_RUNS_DIR}; run --part lag_runs first"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(path) for path in files])


def main() -> None:
    """Build the requested part of the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("train_shape", "lag_runs", "lag_shapes", "placebo"),
        required=True,
    )
    parser.add_argument("--runs", type=int, default=None, help="lag_runs: only the first N runs")
    args = parser.parse_args()
    common.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    series = sites_table()
    train_path = common.INPUTS_DIR / "train_shape.parquet"
    val_path = common.INPUTS_DIR / "val_keys.parquet"
    if args.part == "train_shape":
        shape = train_shape_table(series=series)
        print(
            f"training rows {shape.height:,}; null shape: {shape['shape_calibrated'].null_count()}"
        )
        common.write_atomically(frame=shape, target=train_path)
        common.write_atomically(
            frame=reference_temperature(shape=shape), target=common.INPUTS_DIR / "t_ref.parquet"
        )
        common.write_atomically(frame=validation_keys(series=series), target=val_path)
        print(pl.read_parquet(common.INPUTS_DIR / "t_ref.parquet"))
    elif args.part == "lag_runs":
        train_origins = pl.read_parquet(train_path, columns=["power_fcst_init_time"])[
            "power_fcst_init_time"
        ].unique()
        validation_origins = pl.read_parquet(val_path, columns=["power_fcst_init_time"])[
            "power_fcst_init_time"
        ].unique()
        runs = needed_runs(train_origins=train_origins, validation_origins=validation_origins)
        print(f"{len(runs)} runs needed, {runs[0]:%Y-%m-%d} to {runs[-1]:%Y-%m-%d}")
        build_lag_runs(series=series, runs=runs, limit=args.runs)
    elif args.part == "lag_shapes":
        lag_runs = read_lag_runs()
        for split, path, target in (
            ("training", train_path, "lag_shapes_train.parquet"),
            ("validation", val_path, "lag_shapes_val.parquet"),
        ):
            keys = pl.read_parquet(path)
            table = lag_shapes_table(keys=keys, series=series, lag_runs=lag_runs)
            print_source_counts(table=table, split=split)
            common.write_atomically(frame=table, target=common.INPUTS_DIR / target)
    else:
        for stream, (path, target) in enumerate(
            ((train_path, "placebo_train.parquet"), (val_path, "placebo_val.parquet"))
        ):
            common.write_atomically(
                frame=placebo_table(keys=pl.read_parquet(path), stream=stream),
                target=common.INPUTS_DIR / target,
            )


if __name__ == "__main__":
    main()
