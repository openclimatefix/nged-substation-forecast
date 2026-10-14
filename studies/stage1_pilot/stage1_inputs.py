"""Paths, series, folds, lead bands, and the canonical weather the stage 1 pilot shares.

Stage 1a is a weather-to-power XGBoost model trained on ERA5 (and CAMS irradiance) and applied to
ECMWF ENS weather. This module reads the ENS rows the fixed-pipeline fits were scored on, puts ERA5
and CAMS onto the same half-hourly, period-ending convention as those rows, and names the feature
columns of both sides, so that the two sides cannot drift apart.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.pv_dataset import nearest_era5_cell, pv_sites, wind_sites
from studies.solar import azimuth, cos_zenith, zenith
from studies.sources import PER_STUDY_DIR, REPO_DATA_DIR

TechnologyType = Literal["solar", "wind"]

OUTPUT_ROOT: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
"""Where this study writes every table, prediction file, and report."""

FIXED_ROOT: Final[Path] = PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156"
"""The fixed-pipeline study whose fold A frames and control fits this study reuses, read only."""

FRAMES_DIR: Final[Path] = FIXED_ROOT / "frames" / "combined" / "fold_a"
"""Fold A's engineered frames: `train.parquet` (control member) and `val/init=<date>.parquet`."""

CONTROL_FITS_DIR: Final[Path] = (
    FIXED_ROOT / "fits" / "combined" / "fold_a" / "production_cleaned" / "main"
)
"""The control arm C0: production features, colsample_bytree 1.0, the chosen cleaning."""

CLEANING_DROPS: Final[Path] = (
    PER_STUDY_DIR
    / "quick_wins"
    / "cleaning"
    / "mid_2025_to_mid_2026"
    / "new_meter_settling.parquet"
)
"""The `(time_series_id, time)` pairs the chosen cleaning rule drops from the training months."""

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""The XGBoost seeds of every fit, shared by stage 1a and the ENS-trained models."""

STAGE1_TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""The first ERA5 hour on disk."""

VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The first ENS run scored; fold A validates from here to 2026-06-30."""

PURGE_DAYS: Final[int] = 14
"""Days between the end of a stage 1a training window and the first valid time it is applied to."""

STAGE1_TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Exclusive end of the pre-validation stage 1a training window: `VALIDATION_START` less 14 days."""

LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)
"""Lead bands as (name, lower hour inclusive, upper hour exclusive), with lead the valid time minus
`power_fcst_init_time`. Day `d` of lead covers hours `24 d` to `24 d + 24`, so "1 to 2 days" is
hours 24 to 72."""

SOLAR_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
"""Every arm's columns for a solar farm. Solar zenith and azimuth carry the time of day and year."""

WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
"""Every arm's columns for a wind farm. ERA5 temperature and pressure are not on disk at the wind
farms' cells, so only the wind speeds and bearings are used."""

FEATURES: Final[dict[str, tuple[str, ...]]] = {"solar": SOLAR_FEATURES, "wind": WIND_FEATURES}
"""Stage 1a's feature columns by technology."""

ERA5_DIR: Final[Path] = REPO_DATA_DIR / "studies" / "weather" / "ERA5"
WIND_2019_2023_DIR: Final[Path] = REPO_DATA_DIR / "studies" / "weather" / "ERA5-WIND-2019-2023"
CAMS_PATH: Final[Path] = (
    REPO_DATA_DIR
    / "studies"
    / "downloads"
    / "reanalysis"
    / "CAMS"
    / "site_points"
    / "beam_diffuse_cams.parquet"
)

ENS_COLUMNS: Final[dict[str, str]] = {
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
    "wind_speed_10m": "wind_speed_10m",
    "wind_speed_100m": "wind_speed_100m",
}
"""Which ENS frame column fills each stage 1a input that ENS serves directly."""

MIN_HOUR_SHAPE: Final[float] = 1e-3
"""An hour whose mean clear-sky shape is below this has no sun, so its irradiance is set to zero."""

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
)


def site_table() -> pl.DataFrame:
    """Return the nine generator series with their anonymous label, technology, and capacity.

    The label is `studies.anonymise`'s fixed A to F or W1 to W3. The capacity is the one the control
    arm's scores were normalised by, read from the fold A frames.

    Returns:
        One row per series with `time_series_id`, `site`, `technology`, `latitude`, `longitude`,
        and `effective_capacity_mw`. The coordinates never leave this process.
    """
    capacity = pl.read_parquet(FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    solar = pv_sites().select("time_series_id", "site", "latitude", "longitude")
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if solar["site"].to_list().count("A") != 1 or LABEL_PERMUTATION_SEED != 784:
        msg = "unexpected solar labels: the anonymisation helper has changed"
        raise ValueError(msg)
    if set(solar["site"].to_list()) != set(SITE_LABELS):
        msg = "the solar site list is not the six labelled farms"
        raise ValueError(msg)
    return (
        pl.concat(
            [
                solar.with_columns(technology=pl.lit("solar")),
                wind.with_columns(technology=pl.lit("wind")),
            ]
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("technology", "site")
    )


def _read_era5_variable(*, name: str) -> pl.DataFrame:
    return pl.read_parquet(ERA5_DIR / "solar_variables" / f"{name}.parquet").select(
        "time", "latitude", "longitude", pl.col("value").alias(name)
    )


def solar_hourly(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return ERA5 hourly weather at each solar farm's nearest 0.25 degree cell.

    Radiation and precipitation are hour-ending means; the rest are instants at `time`.

    Args:
        sites: The solar rows of `site_table()`.

    Returns:
        One row per `(time_series_id, time)` with `ghi`, `longwave`, `precipitation`
        (period-ending), `temperature_2m`, `dew_point_temperature_2m`, `pressure_surface`,
        `u10`, `v10` (instants), in W m-2, kg m-2 s-1, degrees Celsius, Pa, and m s-1.
    """
    mirror = pl.read_parquet(ERA5_DIR / "site_points" / "beam_diffuse_open_meteo.parquet")
    cells = nearest_era5_cell(sites=sites, era5=mirror).select(
        "time_series_id",
        pl.col("cell_latitude").alias("latitude"),
        pl.col("cell_longitude").alias("longitude"),
    )
    frame = mirror.select(
        "time",
        "latitude",
        "longitude",
        pl.col("ghi_w_m2").alias("ghi"),
        pl.col("temp_c").alias("temperature_2m"),
    )
    for name in ("d2m", "sp", "strd", "tp", "u10", "v10"):
        frame = frame.join(
            _read_era5_variable(name=name),
            on=["time", "latitude", "longitude"],
            how="inner",
            validate="1:1",
        )
    return (
        cells.join(frame, on=["latitude", "longitude"], how="inner")
        .select(
            "time_series_id",
            "time",
            "ghi",
            (pl.col("strd") / 3600.0).alias("longwave"),
            (pl.col("tp").clip(lower_bound=0.0) * 1000.0 / 3600.0).alias("precipitation"),
            pl.col("temperature_2m").cast(pl.Float64),
            (pl.col("d2m") - 273.15).alias("dew_point_temperature_2m"),
            pl.col("sp").alias("pressure_surface"),
            "u10",
            "v10",
        )
        .sort("time_series_id", "time")
    )


def wind_hourly(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return ERA5 hourly 10 m and 100 m wind components at each wind farm's nearest cell.

    The 2019 to 2023 file is keyed by cell, the 2024 on file by site; both name the nearest cell by
    zero offsets.

    Args:
        sites: The wind rows of `site_table()`.

    Returns:
        One row per `(time_series_id, time)` with `u10`, `v10`, `u100`, `v100` in m s-1, instants.
    """
    groups = pl.read_parquet(WIND_2019_2023_DIR / "cell_groups.parquet").filter(
        (pl.col("kind") == "wind") & (pl.col("dy") == 0) & (pl.col("dx") == 0)
    )
    early = pl.read_parquet(WIND_2019_2023_DIR / "wind_era5_cds.parquet").join(
        groups.select("cell_id", pl.col("label").alias("site")), on="cell_id", how="inner"
    )
    late = (
        pl.read_parquet(ERA5_DIR / "wind_native_cds.parquet")
        .filter((pl.col("dy") == 0) & (pl.col("dx") == 0))
        .with_columns(pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"))
    )
    columns = ["site", "time", "u10", "v10", "u100", "v100"]
    joined = pl.concat([early.select(columns), late.select(columns)]).unique(
        subset=["site", "time"], keep="last"
    )
    return (
        sites.select("time_series_id", "site")
        .join(joined, on="site", how="inner")
        .drop("site")
        .sort("time_series_id", "time")
    )


def cams_hourly(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return CAMS hourly global irradiance at each solar farm, hour-ending, in W m-2.

    No reliability filter is applied: the row set is decided by the power observations, so every
    arm sees the same hours.

    Args:
        sites: The solar rows of `site_table()`.

    Returns:
        One row per `(time_series_id, time)` with `ghi_cams`.
    """
    cams = pl.read_parquet(CAMS_PATH)
    return (
        sites.select("time_series_id", "site")
        .join(
            cams.select("site", "time", pl.col("ghi_w_m2").alias("ghi_cams")),
            on="site",
            how="inner",
        )
        .drop("site")
        .sort("time_series_id", "time")
    )


def half_hourly(
    *,
    hourly: pl.DataFrame,
    instants: list[str],
    period: list[str],
    irradiance: list[str],
    latitude: float,
    longitude: float,
) -> pl.DataFrame:
    """Put one series' hourly ERA5 or CAMS table onto the half-hourly, period-ending grid.

    A period-ending value labelled `T` is the mean over `(T - 1 h, T]`. The half-hours ending at
    `T - 30 min` and at `T` both take it, so the hour's mean is conserved. An irradiance is spread
    over those two half-hours by the clear-sky shape instead: each half-hour gets the hour's
    clear-sky index times its own clear-sky irradiance, so the two values average back to the
    hour's value and the diurnal curve is not flat within the hour. An instant is interpolated
    linearly between its two neighbouring hours, so a `:30` stamp takes their mean.

    Args:
        hourly: Rows sorted by `time` with one series' hourly values.
        instants: Columns that are instants.
        period: Columns that are period-ending means, held constant within the hour.
        irradiance: Columns that are period-ending irradiances in W m-2.
        latitude: Degrees north of the series.
        longitude: Degrees east of the series.

    Returns:
        Rows at every `:00` and `:30` stamp from the first hour to the last, with `valid_time` and
        every column named above.
    """
    stamps = pl.datetime_range(hourly["time"].min(), hourly["time"].max(), "30m", eager=True)  # ty: ignore[no-matching-overload]
    on_hour = pl.col("valid_time").dt.minute() == 0
    grid = pl.DataFrame({"valid_time": stamps}).with_columns(
        hour_start=pl.col("valid_time").dt.truncate("1h"),
        hour_end=pl.when(on_hour)
        .then(pl.col("valid_time"))
        .otherwise(pl.col("valid_time") + pl.duration(minutes=30)),
        shape=pl.Series(
            cos_zenith(
                zenith_deg=zenith(
                    stamps=stamps.dt.offset_by("-15m"), latitude=latitude, longitude=longitude
                )
            )
        ),
    )
    grid = grid.join(
        grid.group_by("hour_end").agg(hour_shape=pl.col("shape").mean()), on="hour_end", how="left"
    )
    low = hourly.select(
        pl.col("time").alias("hour_start"), *[pl.col(c).alias(f"{c}_lo") for c in instants]
    )
    high = hourly.select(
        (pl.col("time") - pl.duration(hours=1)).alias("hour_start"),
        *[pl.col(c).alias(f"{c}_hi") for c in instants],
    )
    ending = hourly.select(pl.col("time").alias("hour_end"), *period, *irradiance)
    joined = (
        grid.join(low, on="hour_start", how="left")
        .join(high, on="hour_start", how="left")
        .join(ending, on="hour_end", how="left")
    )
    return joined.select(
        "valid_time",
        *[
            pl.when(on_hour)
            .then(pl.col(f"{c}_lo"))
            .otherwise((pl.col(f"{c}_lo") + pl.col(f"{c}_hi")) / 2.0)
            .alias(c)
            for c in instants
        ],
        *period,
        *[
            pl.when(pl.col("hour_shape") > MIN_HOUR_SHAPE)
            .then(pl.col(c) * pl.col("shape") / pl.col("hour_shape"))
            .otherwise(0.0)
            .alias(c)
            for c in irradiance
        ],
    )


def calendar_columns() -> list[pl.Expr]:
    """Return sine and cosine of the UTC hour of day and of the day of year of `valid_time`."""
    hour = (
        pl.col("valid_time").dt.hour().cast(pl.Float64)
        + pl.col("valid_time").dt.minute().cast(pl.Float64) / 60.0
    )
    day = pl.col("valid_time").dt.ordinal_day().cast(pl.Float64)
    return [
        (np.sin(2 * np.pi * hour / 24.0)).alias("hour_sin"),
        (np.cos(2 * np.pi * hour / 24.0)).alias("hour_cos"),
        (np.sin(2 * np.pi * day / 365.25)).alias("doy_sin"),
        (np.cos(2 * np.pi * day / 365.25)).alias("doy_cos"),
    ]


def solar_geometry(*, stamps: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the solar zenith and azimuth at the midpoint of the half-hour ending at each stamp."""
    midpoints = stamps.dt.offset_by("-15m")
    return pl.DataFrame(
        {
            "valid_time": stamps,
            "solar_zenith": zenith(stamps=midpoints, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoints, latitude=latitude, longitude=longitude),
        }
    )
