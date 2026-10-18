"""Fit experiment 98's residual models: H0.5 with an optional solar and heating base margin.

Experiment 17's arm H0.5 (recency weights, half-life 0.5 years) gave the residual that experiment
82's detector reads. Every arm here refits that model on the 1st and on the 16th of each month
(experiment 71's layout), with experiment 17's inputs and parameters except 150 rounds at eta
0.15, and differs only in the base margin. The margin is the training median of the target plus:

- `solar`: minus the series' capacity over its effective capacity times the anomaly of a unit PV
  shape `u`, where `u` comes from ERA5 shortwave (`era5`) or from CAMS irradiance (`cams`) and the
  anomaly is `u` minus its mean in that month and half-hour of 2019-09 to 2021-12. Only the five
  series that have a capacity (experiment 64's) get a solar term.
- `heating`: experiment 94's hourly coefficient times the degrees below 15.5 C of the ERA5 2 m
  temperature, fitted on the refit's training rows with a month-by-hour intercept.

    uv run python ml-experiments/00098-*/e00098_fit.py fit --arm ctrl
    uv run python ml-experiments/00098-*/e00098_fit.py fit --arm era5_solar --solar era5
    uv run python ml-experiments/00098-*/e00098_fit.py fit --arm heat --solar era5 --heating
    uv run python ml-experiments/00098-*/e00098_fit.py diagnose
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
import xgboost as xgb
from e00098_common import BASE_DIR, FEATURES_PATH, FITS_ROOT, SERIES_IDS
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

SOLAR_INPUTS_DIR: Final[Path] = BASE_DIR / "solar_inputs"
SERIES_PATH: Final[Path] = (
    study_dir_for(study="stage1_pilot") / "demand" / "inputs" / "series.parquet"
)
E46_SOLAR_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00046-embedded-solar-share-of-primary-forecast-error"
    / "work"
    / "solar"
)
CAPACITY_HELD: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar"
    / "work"
    / "capacity.parquet"
)
CAPACITY_VINTAGES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00078-pvdis-5-current-capacity-vintages"
    / "work"
    / "capacity_vintages.parquet"
)
VINTAGE_FROM: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""From this date the capacity is experiment 78's season-year value of this date; before it, the
held capacity of experiment 64."""
CLIMATOLOGY_END: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
SCORE_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
N_ROUNDS: Final[int] = 150
HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
BALANCE_POINT_C: Final[float] = 15.5
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
N_WORKERS: Final[int] = 6
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.15,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "objective": "reg:absoluteerror",
    "seed": 0,
    "nthread": 1,
}
COLUMNS: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_christmas_window",
)
SOLAR_SERIES: Final[tuple[int, ...]] = (7, 8, 10, 16, 18)
"""Experiment 64's sites D9, D11, D10, D14, and D16."""


def _christmas_window() -> pl.Expr:
    """Return days to the nearest 25 December, signed, null outside the 14-day window."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    year = local.dt.year()
    offsets = [(pl.date(year + shift, 12, 25) - local).dt.total_days() for shift in (-1, 0, 1)]
    nearest = pl.min_horizontal([o.abs() for o in offsets])
    signed = (
        pl.when(offsets[0].abs() == nearest)
        .then(offsets[0])
        .when(offsets[1].abs() == nearest)
        .then(offsets[1])
        .otherwise(offsets[2])
    )
    return pl.when(nearest <= CHRISTMAS_WINDOW_DAYS).then(signed).otherwise(None)


def origins(*, first: datetime, day: int) -> list[datetime]:
    """Return the first instant of the given `day` of each month from `first` to mid-2026."""
    out, month = [], datetime(first.year, first.month, day, tzinfo=UTC)
    while month < datetime(2026, 7, 1, tzinfo=UTC):
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, day, tzinfo=UTC)
    return out


def unit_shape(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape `u` of half-hour mean irradiance, shape (n,), NaN where unknown."""
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).astype(np.float64)
    split = pvlib.irradiance.erbs(ghi=flat, zenith=zenith, datetime_or_doy=day)
    sky = SunAndSky(
        half_hour_end_time=stamps.to_numpy(),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=zenith,
        sun_azimuth_deg=sun_azimuth,
    )
    return np.where(unknown, np.nan, plant_power_mw(sky=sky, parameters=FLEET))


def anomaly_of(*, frame: pl.DataFrame, u: np.ndarray) -> np.ndarray:
    """`u` minus its mean in that UTC month and half-hour over 2019-09 to 2021-12; NaN is zero."""
    table = frame.select("valid_time").with_columns(
        u=pl.Series(np.where(np.isfinite(u), u, np.nan)).fill_nan(None),
        month=pl.col("valid_time").dt.month(),
        hh=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
    )
    climatology = (
        table.filter(pl.col("valid_time") < CLIMATOLOGY_END)
        .group_by("month", "hh")
        .agg(clim=pl.col("u").mean())
    )
    anomaly = (
        table.join(climatology, on=["month", "hh"], how="left")["u"]
        - table.join(climatology, on=["month", "hh"], how="left")["clim"]
    )
    return np.nan_to_num(anomaly.to_numpy(), nan=0.0)


def capacity_series(*, site: str, times: pl.Series, effective: float) -> np.ndarray:
    """Solar capacity over effective capacity: held before 2025-07, then the season-year value."""
    held = pl.read_parquet(CAPACITY_HELD).filter(pl.col("site") == site)["capacity_mw"].item()
    vintage = (
        pl.read_parquet(CAPACITY_VINTAGES)
        .filter(pl.col("site") == site, pl.col("run_date") == VINTAGE_FROM.date())["season_year_mw"]
        .item()
    )
    return (
        np.where(
            times.to_numpy() >= np.datetime64(VINTAGE_FROM.replace(tzinfo=None)), vintage, held
        )
        / effective
    )


def series_frame(*, time_series_id: int) -> pl.DataFrame:
    """The series' feature rows plus the unit-shape anomalies `anom_era5` and `anom_cams`."""
    cache = SOLAR_INPUTS_DIR / f"{time_series_id}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    frame = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id)
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("time_series_id", "valid_time", "target", "effective_capacity_mw", *COLUMNS)
        .collect()
        .sort("valid_time")
    )
    meta = pl.read_parquet(SERIES_PATH).filter(pl.col("time_series_id") == time_series_id)
    site = meta["site"].item()
    effective = float(frame["effective_capacity_mw"][0])
    anom_era5 = np.zeros(frame.height)
    anom_cams = np.zeros(frame.height)
    scale = np.zeros(frame.height)
    if time_series_id in SOLAR_SERIES:
        era5_u = unit_shape(
            stamps=frame["valid_time"],
            ghi=frame["shortwave"].to_numpy().astype(np.float64),
            latitude=float(meta["latitude"].item()),
            longitude=float(meta["longitude"].item()),
        )
        anom_era5 = anomaly_of(frame=frame, u=era5_u)
        cams = (
            pl.read_parquet(E46_SOLAR_DIR / "cams_u.parquet")
            .filter(pl.col("site") == site)
            .select(valid_time="time", cams_u="u")
        )
        cams_u = frame.join(cams, on="valid_time", how="left")["cams_u"].to_numpy()
        anom_cams = anomaly_of(frame=frame, u=cams_u)
        scale = capacity_series(
            site=site, times=frame["valid_time"].dt.replace_time_zone(None), effective=effective
        )
        frame = frame.with_columns(era5_u=pl.Series(era5_u), cams_u=pl.Series(cams_u))
    out = frame.with_columns(
        anom_era5=pl.Series(anom_era5),
        anom_cams=pl.Series(anom_cams),
        capacity_scale=pl.Series(scale),
    )
    SOLAR_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scratch = cache.with_suffix(f".{os.getpid()}.tmp")
    out.write_parquet(scratch)
    scratch.rename(cache)
    return out


def heating_degree_hours(*, temperature: np.ndarray) -> np.ndarray:
    """The degrees (C) each temperature is below the balance point, zero above it."""
    return np.maximum(0.0, BALANCE_POINT_C - temperature.astype(np.float64))


def heating_coefficients(*, rows: pl.DataFrame, train_end: datetime) -> np.ndarray:
    """The 24 hourly heating coefficients of one series (experiment 94), each at least zero."""
    age_days = (train_end - rows["valid_time"]).dt.total_seconds() / 86_400
    cells = rows.select(
        hour=pl.col("hour_local").cast(pl.Int32),
        month=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month(),
        y=pl.col("target"),
        h=pl.Series(heating_degree_hours(temperature=rows["temperature_2m"].to_numpy())),
        w=pl.Series(0.5 ** (age_days.to_numpy() / HALF_LIFE_DAYS)),
    )
    cells = cells.with_columns(
        y_bar=(pl.col("y") * pl.col("w")).sum().over("hour", "month")
        / pl.col("w").sum().over("hour", "month"),
        h_bar=(pl.col("h") * pl.col("w")).sum().over("hour", "month")
        / pl.col("w").sum().over("hour", "month"),
    ).with_columns(dy=pl.col("y") - pl.col("y_bar"), dh=pl.col("h") - pl.col("h_bar"))
    per_hour = cells.group_by("hour").agg(
        cov=(pl.col("w") * pl.col("dh") * pl.col("dy")).sum(),
        var=(pl.col("w") * pl.col("dh") ** 2).sum(),
    )
    coefficients = np.zeros(24)
    for hour, cov, var in per_hour.iter_rows():
        if var > 1e-9:
            coefficients[hour] = max(0.0, cov / var)
    return coefficients


def offset_margin(
    *, frame: pl.DataFrame, solar: str | None, coefficients: np.ndarray | None
) -> np.ndarray:
    """The solar plus heating margin of each row (zero for an arm without either term)."""
    margin = np.zeros(frame.height)
    if solar is not None:
        margin -= frame["capacity_scale"].to_numpy() * frame[f"anom_{solar}"].to_numpy()
    if coefficients is not None:
        hours = frame["hour_local"].to_numpy().astype(int)
        margin += coefficients[hours] * heating_degree_hours(
            temperature=frame["temperature_2m"].to_numpy()
        )
    return margin


def matrix(*, frame: pl.DataFrame, margin: np.ndarray) -> xgb.DMatrix:
    """The DMatrix of the input columns with the base margin set."""
    out = xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())
    out.set_base_margin(margin.astype(np.float32))
    return out


def predict_origin(
    *, frame: pl.DataFrame, origin: datetime, end: datetime, solar: str | None, heating: bool
) -> pl.DataFrame | None:
    """Predictions for `origin` up to `end`, or None if the series has too little history yet."""
    train_end = origin - PURGE
    rows = frame.filter(
        pl.col("valid_time") >= TRAIN_START,
        pl.col("valid_time") < train_end,
        pl.col("target").is_not_null(),
    )
    if rows.height < MIN_TRAIN_ROWS:
        return None
    valid = frame.filter(
        pl.col("valid_time") >= origin, pl.col("valid_time") < end, pl.col("target").is_not_null()
    )
    if valid.is_empty():
        return None
    coefficients = heating_coefficients(rows=rows, train_end=train_end) if heating else None
    start = float(np.median(rows["target"].to_numpy()))
    train_matrix = matrix(
        frame=rows,
        margin=start + offset_margin(frame=rows, solar=solar, coefficients=coefficients),
    )
    train_matrix.set_label(rows["target"].to_numpy())
    age_years = (train_end - rows["valid_time"]).dt.total_days().to_numpy() / 365.25
    train_matrix.set_weight(0.5 ** (age_years / (HALF_LIFE_DAYS / 365.25)))
    booster = xgb.train(PARAMETERS, train_matrix, num_boost_round=N_ROUNDS)
    prediction = booster.predict(
        matrix(
            frame=valid,
            margin=start + offset_margin(frame=valid, solar=solar, coefficients=coefficients),
        )
    ).astype(np.float32)
    return valid.select("valid_time", "target").with_columns(pred=pl.Series(prediction))


def fit_job(job: tuple[str, int, str, str | None, bool, bool]) -> str:
    """Fit every origin of one (arm, series, which) and write its predictions."""
    arm, series, which, solar, heating, smoke = job
    path = FITS_ROOT / arm / which / f"{series}.parquet"
    if path.exists():
        return f"{arm}/{which}/{series}: exists"
    frame = series_frame(time_series_id=series)
    day = 1 if which == "first" else 16
    starts = origins(first=datetime(2021, 12, 1, tzinfo=UTC), day=day)
    if smoke:
        starts = starts[-3:]
    ends = [*starts[1:], SCORE_END]
    pieces = []
    for origin, end in zip(starts, ends, strict=True):
        piece = predict_origin(
            frame=frame,
            origin=origin,
            end=end,
            solar=solar if series in SOLAR_SERIES else None,
            heating=heating,
        )
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return f"{arm}/{which}/{series}: no origin has a year of training data"
    out = pl.concat(pieces)
    if smoke:
        mae = np.abs(out["pred"] - out["target"]).mean()
        return f"{arm}/{which}/{series}: smoke {out.height} rows, MAE {mae:.4f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{arm}/{which}/{series}: {out.height:,} rows, {len(pieces)} refits"


def diagnose() -> None:
    """Print the ERA5 and CAMS unit shapes' agreement and the control's slope on each margin."""
    for series in SOLAR_SERIES:
        frame = series_frame(time_series_id=series).filter(pl.col("era5_u") > 0.02)
        both = frame.filter(pl.col("cams_u").is_not_nan())
        missing = frame.height - both.height
        print(
            f"series {series}: daylight rows {both.height}, corr(u_era5, u_cams) "
            f"{np.corrcoef(both['era5_u'], both['cams_u'])[0, 1]:.3f}, mean u era5 "
            f"{both['era5_u'].mean():.3f} cams {both['cams_u'].mean():.3f}, "
            f"CAMS missing {missing}, capacity scale {frame['capacity_scale'].min():.3f} to "
            f"{frame['capacity_scale'].max():.3f}"
        )


def main() -> None:
    """Fit an arm (`fit`) or print the solar-input diagnostics (`diagnose`)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "diagnose"))
    parser.add_argument("--arm")
    parser.add_argument("--solar", choices=("era5", "cams"))
    parser.add_argument("--heating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if args.stage == "diagnose":
        diagnose()
        return
    series = SOLAR_SERIES if args.solar and not args.heating else SERIES_IDS
    jobs = [
        (args.arm, s, which, args.solar, args.heating, args.smoke)
        for s in series
        for which in ("first", "second")
    ]
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for line in pool.map(fit_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
