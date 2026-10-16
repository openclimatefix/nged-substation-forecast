"""Build the solar inputs of experiment 00046 under `work/solar/`.

For each of the six sites (`SITES`):

- `cams_u.parquet`: the CAMS unit PV shape `u` on the half-hour grid, from experiment 00039's frame
  (`kind == "substation"`), present only where CAMS reliability is at least 0.9. The 99th percentile
  of `u` over a solar zenith under 75 degrees is asserted to be 1.0.
- `climatology.parquet`: the month-by-half-hour mean of `u` over valid times up to 2025-06-30, for
  CAMS and for the ENS ensemble mean (local Europe/London month, UTC half-hour of day), and the
  daylight mask (CAMS climatology above 0.02). When any cell holds fewer than 30 days, every source
  pools the neighbouring local months.
- `ens_train/init=*.parquet`: for each daily ENS run of the training window, the mean over the 51
  members of the ENS `u` (and the control member's `u`), from the archive's `shortwave`. The served
  `shortwave` is already a half-hour mean ending at the valid time (the same convention as CAMS),
  so the chain is the CAMS chain with no step interpolation.
- `calibration.parquet`: per site and lead band, the slope through the origin of the CAMS anomaly on
  the ensemble-mean ENS anomaly, training window only.
- `ens_val/init=*.parquet`: for each validation run, the raw ENS anomaly of every member and the
  calibrated anomaly (the calibrated ensemble mean plus each member's unscaled deviation from the
  mean), keyed by run, valid time, and member.

    uv run python ml-experiments/00046-*/e00046_solar_inputs.py [--train-runs N] [--check]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00046_common as common
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
FIRST_TRAINING_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
HORIZON_HOURS: Final[int] = 351
MAX_ZENITH_DEG: Final[float] = 75.0
U_P99_TOLERANCE: Final[float] = 0.02
TRAIN_DIR: Final[Path] = common.SOLAR_DIR / "ens_train"
VAL_DIR: Final[Path] = common.SOLAR_DIR / "ens_val"


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


def cams_frame() -> pl.DataFrame:
    """Return the CAMS `u`, irradiance, and zenith of the six sites on the half-hour grid.

    Returns:
        Columns `site`, `time` (UTC, period-ending), `u`, `ghi_u`, `zenith`; reliable hours only.
    """
    return (
        pl.read_parquet(common.FRAME_00039_PATH)
        .filter((pl.col("kind") == "substation") & pl.col("site").is_in(common.SITES))
        .select("site", "time", "u", "ghi_u", "zenith")
    )


def climatology(*, frame: pl.DataFrame, value: str, time_column: str, pool: bool) -> pl.DataFrame:
    """Return the month-by-half-hour mean of a column over training-window valid times.

    Args:
        frame: Columns `site`, a UTC time column, and the value column.
        value: The value column.
        time_column: The time column.
        pool: Pool each local month with its two neighbours when true.

    Returns:
        Columns `site`, `month`, `hh`, `clim` (the mean), and `days` (distinct UTC dates behind it).

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    training = frame.filter(pl.col(value).is_not_null() & pl.col(value).is_not_nan())
    if common.latest_time(times=training[time_column]) > common.LAST_TRAINING_VALID:
        msg = "climatology rows must be training-window valid times only"
        raise ValueError(msg)
    cells = common.with_cell(frame=training, time_column=time_column).with_columns(
        date=pl.col(time_column).dt.date()
    )
    if pool:
        cells = pl.concat(
            [
                cells.with_columns(month=((pl.col("month") - 1 + shift) % 12 + 1).cast(pl.Int8))
                for shift in (-1, 0, 1)
            ]
        )
    return cells.group_by("site", "month", "hh").agg(
        clim=pl.col(value).mean(), days=pl.col("date").n_unique()
    )


def read_run_shortwave(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly `shortwave` for every member at every series' cell.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `series_table()` filtered to the sites, with `h3_res_5`, `latitude`, `longitude`.

    Returns:
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `ensemble_member`, and
        `shortwave` (W/m2, half-hour mean ending at `valid_time`), valid times after the forecast
        origin (the run plus 9 hours) and at most 351 hours later.
    """
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select("h3_index", "ensemble_member", "valid_time", SHORTWAVE_SOURCE)
        .filter(pl.col("h3_index").is_in(series["h3_res_5"].unique().to_list()))
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
    origin = run + timedelta(hours=9)
    return (
        half_hourly.rename({SHORTWAVE_SOURCE: "shortwave"})
        .with_columns(power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")))
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=HORIZON_HOURS)
            )
        )
        .select(*common.KEYS, "ensemble_member", "shortwave")
    )


def member_u(*, rows: pl.DataFrame, series_row: dict) -> pl.DataFrame:
    """Return the ENS `u` of every member and valid time of one series and one run.

    Args:
        rows: One series' rows of one run with `valid_time`, `ensemble_member`, and `shortwave`,
            every member holding every valid time.
        series_row: The series' `latitude` and `longitude`.

    Returns:
        Columns `valid_time`, `ensemble_member`, and `u`, sorted by valid time then member.

    Raises:
        ValueError: If a member lacks a valid time.
    """
    ordered = rows.sort("valid_time", "ensemble_member")
    times = ordered["valid_time"].unique(maintain_order=True)
    members = ordered["ensemble_member"].unique().sort()
    if ordered.height != times.len() * members.len():
        msg = "every member must hold every valid time"
        raise ValueError(msg)
    ghi = ordered["shortwave"].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
    u = u_from_half_hour_ghi(
        stamps=times,
        ghi=ghi,
        latitude=series_row["latitude"],
        longitude=series_row["longitude"],
    )
    return pl.DataFrame(
        {
            "valid_time": np.repeat(times.to_numpy(), members.len()),
            "ensemble_member": np.tile(members.to_numpy(), times.len()),
            "u": u.T.ravel().astype(np.float32),
        }
    ).with_columns(pl.col("valid_time").dt.replace_time_zone("UTC"))


def build_train_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one training run's ensemble-mean and control-member ENS `u` per site.

    Args:
        run: The 00:00 UTC initialisation time.
        series: The six sites' `series_table()` rows.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `u_mean`, and `u_control`, with valid
        times up to the last training valid time.
    """
    rows = read_run_shortwave(run=run, series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        own = rows.filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
        )
        if own.is_empty():
            continue
        u = member_u(rows=own, series_row=row)
        pieces.append(
            u.group_by("valid_time")
            .agg(
                u_mean=pl.col("u").mean(),
                u_control=pl.col("u").filter(pl.col("ensemble_member") == 0).first(),
            )
            .with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(run + timedelta(hours=9)).cast(
                    pl.Datetime("us", "UTC")
                ),
            )
            .select("site", "power_fcst_init_time", "valid_time", "u_mean", "u_control")
        )
    return pl.concat(pieces) if pieces else pl.DataFrame()


def build_train(*, series: pl.DataFrame, n_runs: int | None) -> pl.DataFrame:
    """Build (or resume) the training-window ENS `u` files and return them all concatenated.

    Args:
        series: The six sites' `series_table()` rows.
        n_runs: Only the first this many runs of the window when given.

    Returns:
        The concatenated `build_train_run` tables of the runs built or found on disk.
    """
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    run = FIRST_TRAINING_RUN
    last_run = common.LAST_TRAINING_VALID.replace(hour=0, minute=0)
    done = 0
    pieces = []
    while run <= last_run and (n_runs is None or done < n_runs):
        target = TRAIN_DIR / f"init={run:%Y-%m-%d}.parquet"
        if not target.exists():
            started = time.monotonic()
            table = build_train_run(run=run, series=series)
            if not table.is_empty():
                common.write_atomically(frame=table, target=target)
                print(
                    f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s",
                    flush=True,
                )
        if target.exists():
            pieces.append(pl.read_parquet(target))
        done += 1
        run += timedelta(days=1)
    return pl.concat(pieces)


def check_against_pilot(*, series: pl.DataFrame) -> None:
    """Compare a rebuilt control-member `shortwave` with the pilot's validation file.

    Args:
        series: The six sites' `series_table()` rows.
    """
    run = datetime(2025, 7, 1, tzinfo=UTC)
    mine = read_run_shortwave(run=run, series=series).filter(pl.col("ensemble_member") == 0)
    theirs = pl.read_parquet(
        common.VALIDATION_DIR / "init=2025-07-01.parquet",
        columns=["time_series_id", "valid_time", "ensemble_member", "shortwave"],
    ).filter(pl.col("ensemble_member") == 0)
    both = mine.join(theirs, on=["time_series_id", "valid_time"], suffix="_pilot")
    diff = (both["shortwave"].cast(pl.Float64) - both["shortwave_pilot"].cast(pl.Float64)).abs()
    print(
        f"control shortwave vs pilot: rows new {mine.height:,}, pilot {theirs.height:,}, "
        f"matched {both.height:,}, max abs difference {diff.max():.4g} W/m2"
    )


def print_u_checks(*, cams: pl.DataFrame) -> None:
    """Assert the design property that the 99th percentile of CAMS `u` is 1.0 in daylight.

    Args:
        cams: `cams_frame()`.

    Raises:
        AssertionError: If the percentile differs from 1.0 by more than `U_P99_TOLERANCE`.
    """
    p99 = (
        cams.filter(pl.col("zenith") < MAX_ZENITH_DEG)
        .group_by("site")
        .agg(p99_u=pl.col("u").quantile(0.99))
    )
    print("P99 of CAMS u over zenith < 75 degrees (design: 1.0)")
    print(p99.sort("site"))
    assert float(np.abs(p99["p99_u"].to_numpy() - 1.0).max()) < U_P99_TOLERANCE


def calibration_table(
    *, train: pl.DataFrame, cams_anomaly: pl.DataFrame, daylight: pl.DataFrame
) -> pl.DataFrame:
    """Fit, per site and lead band, the slope of the CAMS anomaly on the ensemble-mean ENS anomaly.

    Args:
        train: Columns `site`, `power_fcst_init_time`, `valid_time`, `anomaly_mean`, and
            `anomaly_control`, training-window valid times only.
        cams_anomaly: Columns `site`, `valid_time`, and `anomaly_cams`.
        daylight: Columns `site`, `month`, `hh`, and `daylight`.

    Returns:
        Columns `site`, `band` (an index into `LEAD_BANDS`), `slope` (through the origin),
        `correlation_mean`, `correlation_control`, and `n_rows`.

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    if common.latest_time(times=train["valid_time"]) > common.LAST_TRAINING_VALID:
        msg = "calibration rows must be training-window valid times only"
        raise ValueError(msg)
    lead_hours = (
        (train["valid_time"] - train["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
    joined = (
        train.with_columns(band=pl.Series(common.lead_band(lead_hours=lead_hours)))
        .join(cams_anomaly, on=["site", "valid_time"], how="inner")
        .pipe(lambda frame: common.with_cell(frame=frame, time_column="valid_time"))
        .join(daylight, on=["site", "month", "hh"], how="left")
        .filter(pl.col("daylight"))
    )
    return (
        joined.group_by("site", "band")
        .agg(
            slope=(pl.col("anomaly_cams") * pl.col("anomaly_mean")).sum()
            / (pl.col("anomaly_mean") ** 2).sum(),
            correlation_mean=pl.corr("anomaly_cams", "anomaly_mean"),
            correlation_control=pl.corr("anomaly_cams", "anomaly_control"),
            n_rows=pl.len(),
        )
        .sort("site", "band")
    )


def build_validation(
    *,
    series: pl.DataFrame,
    ens_climatology: pl.DataFrame,
    calibration: pl.DataFrame,
) -> None:
    """Write, for every validation run, the raw and calibrated ENS anomaly of every member.

    The calibrated anomaly of a member is the slope times the ensemble-mean anomaly plus the
    member's unscaled deviation from that mean. `anomaly_mean_calibrated` is the slope times the
    ensemble-mean anomaly alone, repeated for every member.

    Args:
        series: The six sites' `series_table()` rows.
        ens_climatology: `climatology()` of the ENS ensemble-mean `u`.
        calibration: `calibration_table()`.
    """
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    clim = ens_climatology.select("site", "month", "hh", "clim")
    slopes = calibration.select("site", "band", "slope")
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        target = VAL_DIR / path.name
        if target.exists():
            continue
        started = time.monotonic()
        run = pl.read_parquet(path, columns=[*common.KEYS, "ensemble_member", "shortwave"]).filter(
            pl.col("time_series_id").is_in(series["time_series_id"])
        )
        pieces = []
        for row in series.iter_rows(named=True):
            own = run.filter(pl.col("time_series_id") == row["time_series_id"])
            u = member_u(rows=own, series_row=row)
            origin = own["power_fcst_init_time"][0]
            frame = (
                u.with_columns(site=pl.lit(row["site"]), power_fcst_init_time=pl.lit(origin))
                .pipe(lambda f: common.with_cell(frame=f, time_column="valid_time"))
                .join(clim.filter(pl.col("site") == row["site"]), on=["site", "month", "hh"])
                .with_columns(
                    anomaly=pl.col("u") - pl.col("clim"),
                    band=pl.Series(
                        common.lead_band(
                            lead_hours=(
                                (u["valid_time"] - origin).dt.total_minutes() / 60.0
                            ).to_numpy()
                        )
                    ),
                )
                .sort("valid_time", "ensemble_member")
            )
            frame = frame.join(slopes, on=["site", "band"], how="left", maintain_order="left")
            mean_anomaly = pl.col("anomaly").mean().over("valid_time")
            pieces.append(
                frame.with_columns(
                    anomaly_mean_calibrated=pl.col("slope") * mean_anomaly,
                )
                .with_columns(
                    anomaly_calibrated=pl.col("anomaly_mean_calibrated")
                    + (pl.col("anomaly") - mean_anomaly)
                )
                .select(
                    "site",
                    "power_fcst_init_time",
                    "valid_time",
                    "ensemble_member",
                    "anomaly",
                    "anomaly_calibrated",
                    "anomaly_mean_calibrated",
                )
            )
        common.write_atomically(frame=pl.concat(pieces), target=target)
        print(f"{target.name}: {time.monotonic() - started:.1f} s", flush=True)


def main() -> None:
    """Build every solar input, printing the checks as it goes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-runs", type=int, default=None, help="only the first N runs")
    parser.add_argument("--check", action="store_true", help="compare one run with the pilot")
    args = parser.parse_args()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    if args.check:
        check_against_pilot(series=series)
        return
    common.SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    cams = cams_frame()
    print_u_checks(cams=cams)
    common.write_atomically(
        frame=cams.select("site", "time", "u"), target=common.SOLAR_DIR / "cams_u.parquet"
    )
    train = build_train(series=series, n_runs=args.train_runs)
    cams_training = cams.filter(pl.col("time") <= common.LAST_TRAINING_VALID)
    ens_training = train.select("site", time=pl.col("valid_time"), u=pl.col("u_mean"))
    unpooled = {
        name: climatology(frame=frame, value="u", time_column="time", pool=False)
        for name, frame in (("cams", cams_training), ("ens", ens_training))
    }
    sunny = unpooled["cams"].filter(pl.col("clim") > common.DAYLIGHT_THRESHOLD)
    counts = []
    for name, cells in unpooled.items():
        in_daylight = cells.join(sunny.select("site", "month", "hh"), on=["site", "month", "hh"])
        print(
            f"{name}: smallest number of days in any cell = {cells['days'].min()}, "
            f"in any daylight cell = {in_daylight['days'].min()}"
        )
        counts.append(int(in_daylight["days"].to_numpy().min()))
    pool = any(count < common.MIN_CLIMATOLOGY_DAYS for count in counts)
    print(f"pooling the neighbouring months for every source: {pool}")
    cams_clim = climatology(frame=cams_training, value="u", time_column="time", pool=pool)
    ens_clim = climatology(frame=ens_training, value="u", time_column="time", pool=pool)
    daylight = cams_clim.with_columns(daylight=pl.col("clim") > common.DAYLIGHT_THRESHOLD).select(
        "site", "month", "hh", "daylight"
    )
    common.write_atomically(
        frame=cams_clim.rename({"clim": "clim_cams", "days": "days_cams"})
        .join(
            ens_clim.rename({"clim": "clim_ens", "days": "days_ens"}),
            on=["site", "month", "hh"],
            how="full",
            coalesce=True,
        )
        .join(daylight, on=["site", "month", "hh"], how="left"),
        target=common.SOLAR_DIR / "climatology.parquet",
    )
    cams_anomaly = (
        common.with_cell(frame=cams.rename({"time": "valid_time"}), time_column="valid_time")
        .join(cams_clim.select("site", "month", "hh", "clim"), on=["site", "month", "hh"])
        .select("site", "valid_time", anomaly_cams=pl.col("u") - pl.col("clim"))
    )
    common.write_atomically(frame=cams_anomaly, target=common.SOLAR_DIR / "cams_anomaly.parquet")
    ens_anomaly = (
        common.with_cell(frame=train, time_column="valid_time")
        .join(ens_clim.select("site", "month", "hh", "clim"), on=["site", "month", "hh"])
        .select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            anomaly_mean=pl.col("u_mean") - pl.col("clim"),
            anomaly_control=pl.col("u_control") - pl.col("clim"),
        )
    )
    calibration = calibration_table(train=ens_anomaly, cams_anomaly=cams_anomaly, daylight=daylight)
    with pl.Config(tbl_rows=40):
        print("Calibration slope of the CAMS anomaly on the ensemble-mean ENS anomaly")
        print(calibration)
    common.write_atomically(frame=calibration, target=common.SOLAR_DIR / "calibration.parquet")
    if args.train_runs is not None:
        print("partial training window: validation anomalies not written")
        return
    build_validation(series=series, ens_climatology=ens_clim, calibration=calibration)


if __name__ == "__main__":
    main()
