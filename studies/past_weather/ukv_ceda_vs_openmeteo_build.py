"""Build the rows of the UKV-from-CEDA against UKV-from-Open-Meteo study: wind, solar, model-free.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. The question is how
different the two archives of the Met Office's UKV are, and whether the difference matters for power
forecasts and for training history. This script builds the three frames the other scripts read: the
wind rows and the solar rows that `ukv_ceda_vs_openmeteo_fit.py` fits, and the model-free hours that
`ukv_ceda_vs_openmeteo_compare.py` describes.

**Most CEDA columns are read from the built frames of the CEDA-against-ERA5 study** (`wind_rows`,
`solar_rows` in `data/studies/per_study/ukv_ceda_vs_era5/`), which hold the power targets, the
capacities, the export-cap flag, the solar geometry, CEDA's 10 m wind, and CEDA's two-instant
temperature. The build replaces that study's folds, eras, and shuffled columns. It reads CEDA's
instants again at the nine generator sites for the model-free hours and for CEDA's irradiance, with
the store readers of `ukv_ceda_vs_era5_build.py`, which sits in the same folder.

**Open-Meteo's values come from `previous_runs/combined.parquet`**, and three constructions are
made to match CEDA's before any arm sees them. Wind is converted from km/h to m/s. Temperature is
the mean of the instants at the hour's two ends, as CEDA's is. CEDA's irradiance snapshot at the
label is turned into a backward hourly mean with the zenith-cosine ratio Open-Meteo applies
(`studies.solar.hourly_mean_from_snapshot`), because Open-Meteo's value is not a snapshot. A guard
for each construction compares the two archives at the lead-0 hours, where both are the same UKV
analysis, and stops the build if it fails.

**Rows are the intersection across every arm, decided from the target and from availability.**
Open-Meteo's backfill before 2024-08-12, the 94 hours in November 2024 that only
`combined.parquet` fills from an unchecked lineage, and the hours with a null wind direction are
dropped for every arm. The study's months are 2024-09 to 2025-12 and 2026-02 to 2026-08.

**Privacy.** Nothing here prints a generator's name, identifier or coordinates.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_build.py`. `--check-only`
reads the real data, runs every guard, and writes nothing. `--dry-run` builds one month of the
model-free hours and writes nothing. A fresh run stops (`refuse_to_overwrite`) while an output
exists.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from cerra_past_solar import check_column_counts, with_covering_folds
from studies.blending import climatology_permutation
from studies.cross_validation import (
    calendar_month_coverage,
    cut_eras,
    raise_on_uncovered_months,
    search_fold_offsets,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.neighbouring_hours import with_neighbouring_hours
from studies.pv_dataset import pv_sites, wind_sites
from studies.solar import cos_zenith, cos_zenith_hour_mean, hourly_mean_from_snapshot, zenith
from studies.sources import (
    UKV_CEDA_VS_OPEN_METEO_DIR,
    UKV_VS_ERA5_DIR,
    previous_runs_product_dir_for,
)
from studies.ukv_ceda_stores import STRADDLING_MONTHS
from ukv_ceda_vs_era5_build import (
    KELVIN,
    MAX_MONTH_LOSS_SHARE,
    UKV_TEMPERATURE_VARIABLE,
    UkvStores,
    hours_lost_lines,
    nearest_ukv_cells,
    open_ukv_stores,
    ukv_at,
)

_LOG: Final[logging.Logger] = logging.getLogger("ukv_ceda_vs_openmeteo_build")

OUTPUT_DIR: Final[Path] = UKV_CEDA_VS_OPEN_METEO_DIR
"""The write-once folder every script of the study writes to."""

WIND_ROWS_NAME: Final[str] = "wind_rows.parquet"
SOLAR_ROWS_NAME: Final[str] = "solar_rows.parquet"
SOLAR_ERA_0_ROWS_NAME: Final[str] = "solar_rows_era0.parquet"
"""The solar rows of era 0 alone, with folds cut inside era 0, for the planned-scope sensitivity."""
MODEL_FREE_NAME: Final[str] = "model_free_hours.parquet"
STAMP_NAME: Final[str] = "build.json"
README_NAME: Final[str] = "README.md"

OPEN_METEO_PATH: Final[Path] = (
    previous_runs_product_dir_for(product="UKV") / "previous_runs" / "combined.parquet"
)
"""Open-Meteo's UKV at the nine generator sites: temperature, wind, and irradiance."""

CEDA_FRAME_NAMES: Final[tuple[str, str]] = ("wind_rows.parquet", "solar_rows.parquet")
"""The built frames of the CEDA-against-ERA5 study that this build reads."""

CEDA_BUILD_STAMP_NAME: Final[str] = "build.json"

FIRST_MONTH: Final[str] = "2024-09"
LAST_MONTH: Final[str] = "2026-08"
"""The first and last whole months both archives cover after the dropped partial months."""

OPEN_METEO_FIRST_HOUR: Final[datetime] = datetime(2024, 8, 12, tzinfo=UTC)
"""The first hour Open-Meteo's own UKV downloader covers. Earlier rows are a backfill."""

UNVERIFIED_HOURS: Final[tuple[datetime, datetime]] = (
    datetime(2024, 11, 9, 16, tzinfo=UTC),
    datetime(2024, 11, 13, 13, tzinfo=UTC),
)
"""The 94 hours (inclusive) that `combined.parquet` fills and the `site_points/` extract lacks."""

OPEN_METEO_WIND_STEP_DAYS: Final[tuple[tuple[date, date], ...]] = (
    (date(2024, 11, 7), date(2024, 11, 30)),
    (date(2025, 1, 16), date(2025, 2, 18)),
)
"""The UTC days (inclusive) on which Open-Meteo's served 10 m wind speed is built differently.

**In these two spans Open-Meteo's 10 m speed is about 6% higher against CEDA's (median ratio 1.064
against 0.968 elsewhere at lead 0) and about 5% higher against ERA5's, while CEDA's speed against
ERA5's is steady, and Open-Meteo's temperature, direction, and irradiance do not step.** Its 100 m
to 10 m speed ratio falls from 1.95 to 1.78. The spans start and end within a day of the dates
here, which are the whole days that contain each change. The steps are a property of the served
series, so every wind arm drops the spans. A month that loses more than `MAX_MONTH_LOSS_SHARE` of
its days to them is dropped whole.
"""

KM_PER_HOUR_PER_M_PER_S: Final[float] = 3.6
"""Open-Meteo serves wind in km/h and CEDA in m/s, so Open-Meteo's is divided by this."""

SPEED_RATIO_BOUNDS: Final[tuple[float, float]] = (0.9, 1.1)
"""The median ratio of the archives' 10 m speeds at lead-0 hours must lie inside these bounds."""

TEMPERATURE_OFFSET_LIMIT_C: Final[float] = 2.0
"""The median difference of the archives' temperatures at lead-0 hours must be below this."""

IRRADIANCE_RATIO_BOUNDS: Final[tuple[float, float]] = (0.97, 1.03)
"""At each lead-0 hour of day, the median of CEDA's rebuilt hourly irradiance over Open-Meteo's."""

MIN_GUARD_IRRADIANCE_W_M2: Final[float] = 50.0
"""Rows below this irradiance are left out of the irradiance guard, where ratios are unstable."""

SHUFFLE_SEED: Final[int] = 1051
"""Seeds the control shuffles. Archive `i` is shuffled under `SHUFFLE_SEED + i`."""

SHUFFLE_GROUPS: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""A shuffled value stays within one site, one year-month and one hour of day."""

SHUFFLED_SUFFIX: Final[str] = "_shuffled"

SUN_ELEVATION_BINS_DEG: Final[tuple[float, ...]] = (2.0, 5.0, 10.0, 20.0, 40.0)
"""The upper edges of the sun-elevation bins of the irradiance ratio, in degrees."""

LOW_SUN_ZENITH_DEG: Final[float] = 85.0
"""Rows with the sun more than 5 degrees above the horizon have a zenith below this."""

ARCHIVES: Final[tuple[str, str]] = ("ceda", "om")
"""The two archives, in the order a contrast reads: the treatment, then the reference."""

MARGIN_WIND_PP: Final[float] = 0.16
"""The wind margin, in percentage points of capacity, inherited from the CEDA-against-ERA5 study."""

MARGIN_SOLAR_PP: Final[float] = 0.06
"""The solar margin, in percentage points of capacity, inherited from the same study."""

WIND_SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year", "era_code")
SOLAR_SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "hour_of_day",
    "day_of_year",
    "era_code",
)

LEAD_CYCLE_HOURS: Final[int] = 6
"""CEDA's runs start every 6 hours, so an hour's lead is its UTC hour modulo this."""

CEDA_INSTANT_VARIABLES: Final[tuple[str, str, str, str]] = (
    UKV_TEMPERATURE_VARIABLE,
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_down",
)
"""The CEDA fields the model-free hours read at each hour's label."""

CEDA_DROPPED_COLUMNS: Final[tuple[str, ...]] = ("fold", "era", "era_code")
"""Columns of the CEDA-against-ERA5 frames that this build replaces."""


# --- Months ---------------------------------------------------------------------------------------


def study_months() -> tuple[str, ...]:
    """List the study's whole months: `FIRST_MONTH` to `LAST_MONTH` without the PS47 month.

    Returns:
        The months as `%Y-%m` labels, in order.
    """
    months: list[str] = []
    year, month = (int(part) for part in FIRST_MONTH.split("-"))
    while f"{year}-{month:02d}" <= LAST_MONTH:
        label = f"{year}-{month:02d}"
        if label not in STRADDLING_MONTHS:
            months.append(label)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(months)


STUDY_MONTHS: Final[tuple[str, ...]] = study_months()
"""The 23 whole months both archives cover."""


# --- Arm columns ----------------------------------------------------------------------------------


def wind_columns(*, archive: str) -> tuple[str, str, str]:
    """Return an archive's three matched 10 m wind columns: speed, sine and cosine of direction.

    Args:
        archive: `ceda` or `om`.

    Returns:
        Three frame column names.

    Raises:
        ValueError: If the archive is not one of `ARCHIVES`.
    """
    if archive == "ceda":
        return ("ukv_ceda_speed_10m", "ukv_ceda_sin_10m", "ukv_ceda_cos_10m")
    if archive == "om":
        return ("om_speed_10m", "om_sin_10m", "om_cos_10m")
    msg = f"unknown archive {archive!r}"
    raise ValueError(msg)


def solar_columns(*, archive: str) -> tuple[str, str]:
    """Return an archive's two solar weather columns: global irradiance and temperature.

    Args:
        archive: `ceda` or `om`.

    Returns:
        Two frame column names.

    Raises:
        ValueError: If the archive is not one of `ARCHIVES`.
    """
    if archive not in ARCHIVES:
        msg = f"unknown archive {archive!r}"
        raise ValueError(msg)
    return (f"{archive}_ghi", f"{archive}_temp")


def _suffixed(*, columns: Sequence[str], shuffled: bool) -> tuple[str, ...]:
    return tuple(f"{column}{SHUFFLED_SUFFIX}" for column in columns) if shuffled else tuple(columns)


def wind_arm_columns(*, archive: str, shuffled: bool = False) -> tuple[str, ...]:
    """Return a wind arm's six feature columns.

    Args:
        archive: `ceda` or `om`.
        shuffled: Whether the arm reads the archive's shuffled copy, for the negative control.

    Returns:
        The three shared columns, then the archive's 10 m speed, sine and cosine.
    """
    return (
        *WIND_SHARED_FEATURES,
        *_suffixed(columns=wind_columns(archive=archive), shuffled=shuffled),
    )


def solar_arm_columns(*, archive: str, shuffled: bool = False) -> tuple[str, ...]:
    """Return a solar arm's eight feature columns.

    Args:
        archive: `ceda` or `om`.
        shuffled: Whether the arm reads the archive's shuffled copy, for the negative control.

    Returns:
        The six shared columns, then the archive's global irradiance and temperature.
    """
    return (
        *SOLAR_SHARED_FEATURES,
        *_suffixed(columns=solar_columns(archive=archive), shuffled=shuffled),
    )


def check_arm_widths() -> None:
    """Raise unless the two arms of every contrast carry the same number of distinct columns."""
    for shuffled in (False, True):
        check_column_counts(
            features={a: wind_arm_columns(archive=a, shuffled=shuffled) for a in ARCHIVES},
            contrasts=[("ceda", "om")],
        )
        check_column_counts(
            features={a: solar_arm_columns(archive=a, shuffled=shuffled) for a in ARCHIVES},
            contrasts=[("ceda", "om")],
        )


# --- Reading Open-Meteo ---------------------------------------------------------------------------


def read_open_meteo(*, path: Path = OPEN_METEO_PATH) -> pl.DataFrame:
    """Read Open-Meteo's UKV, keep the hours from its own downloader, and convert wind to m/s.

    The rows before `OPEN_METEO_FIRST_HOUR` (a backfill from an unnamed source) and the
    `UNVERIFIED_HOURS` are dropped here, so no arm can read them.

    Args:
        path: Open-Meteo's `combined.parquet`.

    Returns:
        One row per (site, hour) with `om_temp_c`, `om_speed_10m_m_s`, `om_direction_10m_deg`
        (null in the hours the archive left null) and `om_ghi`.
    """
    low, high = UNVERIFIED_HOURS
    return (
        pl.read_parquet(
            path,
            columns=[
                "site",
                "time",
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "shortwave_radiation",
            ],
        )
        .filter(pl.col("time") >= OPEN_METEO_FIRST_HOUR)
        .filter(~pl.col("time").is_between(low, high, closed="both"))
        .select(
            "site",
            "time",
            om_temp_c=pl.col("temperature_2m"),
            om_speed_10m_m_s=pl.col("wind_speed_10m") / KM_PER_HOUR_PER_M_PER_S,
            om_direction_10m_deg=pl.col("wind_direction_10m"),
            om_ghi=pl.col("shortwave_radiation"),
        )
        .sort("site", "time")
    )


def check_no_backfill(*, frame: pl.DataFrame) -> None:
    """Raise if any row falls before Open-Meteo's own downloader started.

    Args:
        frame: Rows carrying `time`.

    Raises:
        ValueError: If a row is earlier than `OPEN_METEO_FIRST_HOUR`.
    """
    early = frame.filter(pl.col("time") < OPEN_METEO_FIRST_HOUR).height
    if early:
        msg = f"{early} rows fall before {OPEN_METEO_FIRST_HOUR:%Y-%m-%d}, Open-Meteo's backfill"
        raise ValueError(msg)


# --- Reading CEDA ---------------------------------------------------------------------------------


def generator_roster() -> pl.DataFrame:
    """Return the nine generator sites with their coordinates. Private: never printed.

    Returns:
        One row per site with `site`, `latitude` and `longitude`.
    """
    return pl.concat(
        [frame.select("site", "latitude", "longitude") for frame in (pv_sites(), wind_sites())]
    )


def ceda_hourly_irradiance(
    *, snapshot: np.ndarray, hours: pl.Series, latitude: float, longitude: float
) -> np.ndarray:
    """Rebuild CEDA's irradiance snapshot at each label as Open-Meteo builds its hourly value.

    Args:
        snapshot: CEDA's downward shortwave at each label instant, in W m⁻².
        hours: The labels, as a UTC datetime series.
        latitude: The site's latitude in degrees. Private: never printed.
        longitude: The site's longitude in degrees. Private: never printed.

    Returns:
        The backward hourly mean, in W m⁻².
    """
    return hourly_mean_from_snapshot(
        snapshot_w_m2=snapshot,
        cos_zenith_instant=cos_zenith(
            zenith_deg=zenith(stamps=hours, latitude=latitude, longitude=longitude)
        ),
        cos_zenith_hour_mean=cos_zenith_hour_mean(
            stamps=hours, latitude=latitude, longitude=longitude
        ),
    )


def read_ceda_instants(
    *, ukv: UkvStores, sites: pl.DataFrame, hours: pl.Series, read_values: bool
) -> pl.DataFrame:
    """Read CEDA's instants at each site's nearest cell, at the labels of the given hours.

    Args:
        ukv: The opened stores.
        sites: Rows with `site`, `latitude` and `longitude`.
        hours: The labels, as a sorted UTC datetime series.
        read_values: Whether to read values (see `ukv_at`).

    Returns:
        One row per (site, usable hour) with `lead_hours`, `ceda_temp_c`, `ceda_speed_10m_m_s`,
        `ceda_direction_10m_deg`, `sun_elevation_deg` (at the label), `ceda_ghi_snapshot` and
        `ceda_ghi` (the rebuilt hourly value).
    """
    cells = nearest_ukv_cells(ukv=ukv, points=sites)
    usable, means, leads = ukv_at(
        ukv=ukv,
        hours=hours,
        offsets_hours=(0,),
        variables=CEDA_INSTANT_VARIABLES,
        cells=cells["cell_id"].to_numpy(),
        read_values=read_values,
    )
    kept_hours = hours.filter(pl.Series(usable))
    coordinates = {row["site"]: (row["latitude"], row["longitude"]) for row in sites.to_dicts()}
    frames: list[pl.DataFrame] = []
    for column, site in enumerate(cells["site"]):
        snapshot = means["shortwave_down"][usable, column]
        latitude, longitude = coordinates[site]
        frames.append(
            pl.DataFrame(
                {
                    "site": site,
                    "time": kept_hours,
                    "lead_hours": leads[usable],
                    "ceda_temp_c": means[UKV_TEMPERATURE_VARIABLE][usable, column] - KELVIN,
                    "ceda_speed_10m_m_s": means["wind_speed_10m"][usable, column],
                    "ceda_direction_10m_deg": means["wind_direction_10m"][usable, column],
                    "sun_elevation_deg": 90.0
                    - zenith(
                        stamps=kept_hours, latitude=float(latitude), longitude=float(longitude)
                    ),
                    "ceda_ghi_snapshot": snapshot,
                    "ceda_ghi": ceda_hourly_irradiance(
                        snapshot=snapshot,
                        hours=kept_hours,
                        latitude=float(latitude),
                        longitude=float(longitude),
                    ),
                }
            )
        )
    return pl.concat(frames)


def in_wind_step_days() -> pl.Expr:
    """Return whether `time` falls on a day of `OPEN_METEO_WIND_STEP_DAYS`.

    Returns:
        A Boolean expression over the `time` column.
    """
    day = pl.col("time").dt.date()
    return pl.any_horizontal(
        (day >= pl.lit(first)) & (day <= pl.lit(last)) for first, last in OPEN_METEO_WIND_STEP_DAYS
    )


def model_free_hours(
    *, ukv: UkvStores, open_meteo: pl.DataFrame, hours: pl.Series, read_values: bool
) -> pl.DataFrame:
    """Join both archives at the nine generator sites on every hour they both cover.

    Args:
        ukv: The opened stores.
        open_meteo: `read_open_meteo`'s result.
        hours: The labels to read, as a sorted UTC datetime series.
        read_values: Whether to read CEDA values.

    Returns:
        One row per (site, hour) with both archives' temperature, 10 m speed (m/s), 10 m
        direction, and global irradiance, the CEDA lead, `month`, `hour_of_day`, and `era_code`
        (0 before the PS47 upgrade and 1 after). `om_wind_step` says whether the hour lies in a
        span in which Open-Meteo's 10 m speed is built differently.
    """
    ceda = read_ceda_instants(
        ukv=ukv, sites=generator_roster(), hours=hours, read_values=read_values
    )
    return (
        ceda.join(open_meteo, on=["site", "time"], how="inner")
        .with_columns(
            month=pl.col("time").dt.strftime("%Y-%m"),
            hour_of_day=pl.col("time").dt.hour(),
            om_wind_step=in_wind_step_days(),
        )
        .with_columns(era_code=(pl.col("month") >= STRADDLING_MONTHS[1]).cast(pl.Int8))
        .filter(pl.col("month").is_in(list(STUDY_MONTHS)))
        .sort("site", "time")
    )


# --- Guards ---------------------------------------------------------------------------------------


def lead_zero(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows at CEDA's lead 0, where both archives are the same UKV analysis.

    Args:
        frame: Rows carrying `lead_hours`.

    Returns:
        The lead-0 rows.
    """
    return frame.filter(pl.col("lead_hours") == 0)


def unit_guard_failures(*, frame: pl.DataFrame) -> list[str]:
    """List the unit guards the model-free hours fail.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One message per failed guard, empty when both pass. Wind speeds must agree in ratio and
        temperatures in offset at the lead-0 hours, which a factor of 3.6 or a Kelvin offset breaks.
    """
    zero = lead_zero(frame=frame).drop_nulls(
        ["ceda_speed_10m_m_s", "om_speed_10m_m_s", "ceda_temp_c", "om_temp_c"]
    )
    failures: list[str] = []
    windy = zero.filter(pl.col("ceda_speed_10m_m_s") > 1.0)
    ratio = float(np.median((windy["om_speed_10m_m_s"] / windy["ceda_speed_10m_m_s"]).to_numpy()))
    low, high = SPEED_RATIO_BOUNDS
    if not low <= ratio <= high:
        failures.append(
            f"median Open-Meteo to CEDA 10 m speed ratio is {ratio:.3f}, not in {low}-{high}"
        )
    offset = float(np.median((zero["om_temp_c"] - zero["ceda_temp_c"]).to_numpy()))
    if not abs(offset) < TEMPERATURE_OFFSET_LIMIT_C:
        failures.append(f"median temperature offset is {offset:+.2f} C, not within the limit")
    return failures


def irradiance_ratio_failures(*, frame: pl.DataFrame, era_code: int) -> list[str]:
    """List the hours of day at which CEDA's rebuilt irradiance does not match Open-Meteo's.

    Args:
        frame: `model_free_hours`'s result.
        era_code: The UKV era to check, 0 before the PS47 upgrade and 1 after.

    Returns:
        One message per failed hour of day, and one if the era has no lead-0 daylight hour. At
        lead-0 hours the two archives are the same snapshot, so the median ratio of the rebuilt
        CEDA value to Open-Meteo's must lie within `IRRADIANCE_RATIO_BOUNDS` at every hour of day.
    """
    zero = lead_zero(frame=frame).filter(
        (pl.col("om_ghi") > MIN_GUARD_IRRADIANCE_W_M2) & (pl.col("era_code") == era_code)
    )
    if zero.is_empty():
        return [f"era {era_code} has no lead-0 daylight hour to check the irradiance against"]
    by_hour = (
        zero.group_by("hour_of_day")
        .agg(ratio=(pl.col("ceda_ghi") / pl.col("om_ghi")).median(), n=pl.len())
        .sort("hour_of_day")
    )
    low, high = IRRADIANCE_RATIO_BOUNDS
    return [
        f"era {era_code}: at {row['hour_of_day']:02d} UTC the median rebuilt-to-served irradiance "
        f"ratio is {row['ratio']:.3f}, not in {low}-{high}"
        for row in by_hour.iter_rows(named=True)
        if not low <= row["ratio"] <= high
    ]


def irradiance_guard_failures(*, frame: pl.DataFrame) -> list[str]:
    """List the irradiance guard failures that stop the build: those of the era before PS47.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        The messages of `irradiance_ratio_failures` for the era before the upgrade. The era after
        it is reported by `irradiance_mismatch_notes` and does not stop the build.
    """
    return irradiance_ratio_failures(frame=frame, era_code=0)


def irradiance_mismatch_notes(*, frame: pl.DataFrame) -> list[str]:
    """List the irradiance mismatches of the era after PS47, which the page must state.

    Open-Meteo's hourly irradiance after the 2026-01-21 upgrade is not the snapshot scaled by the
    zenith-cosine ratio that it was before, so CEDA's rebuilt value does not match it at the lead-0
    hours. The solar rows of that era therefore compare two differently built irradiances.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        The messages of `irradiance_ratio_failures` for the era after the upgrade.
    """
    return irradiance_ratio_failures(frame=frame, era_code=1)


def irradiance_ratios_by_era_hour(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the median ratio of CEDA's irradiance to Open-Meteo's at lead 0, by era and hour.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per (era, UTC hour of day) with `rebuilt_ratio` (CEDA's snapshot rebuilt as
        Open-Meteo built its value before PS47), `raw_ratio` (the unscaled snapshot), and `n`, over
        the lead-0 rows where Open-Meteo's irradiance exceeds `MIN_GUARD_IRRADIANCE_W_M2`.
    """
    return (
        lead_zero(frame=frame)
        .filter(pl.col("om_ghi") > MIN_GUARD_IRRADIANCE_W_M2)
        .group_by("era_code", "hour_of_day")
        .agg(
            rebuilt_ratio=(pl.col("ceda_ghi") / pl.col("om_ghi")).median(),
            raw_ratio=(pl.col("ceda_ghi_snapshot") / pl.col("om_ghi")).median(),
            n=pl.len(),
        )
        .sort("era_code", "hour_of_day")
    )


def elevation_bin_label(*, upper: float, lower: float | None) -> str:
    """Name one sun-elevation bin.

    Args:
        upper: The bin's upper edge in degrees, or infinity for the last bin.
        lower: The bin's lower edge, or `None` for the first.

    Returns:
        A label such as `2 to 5 degrees`.
    """
    if lower is None:
        return f"up to {upper:g} degrees"
    if upper == float("inf"):
        return f"above {lower:g} degrees"
    return f"{lower:g} to {upper:g} degrees"


def irradiance_ratios_by_elevation(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the median ratio of CEDA's rebuilt irradiance to Open-Meteo's, by sun elevation.

    The 50 W/m2 cut of `irradiance_ratios_by_era_hour` leaves out the low-sun rows, where a rebuild
    from a snapshot at the label does not reproduce Open-Meteo's value.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per (era, elevation bin) with `bin`, `rebuilt_ratio` (the median), `p10`, `p90`,
        `mean_abs_diff` in W/m2, and `n`, over the lead-0 rows where Open-Meteo's irradiance is
        above zero.
    """
    edges = (None, *SUN_ELEVATION_BINS_DEG)
    uppers = (*SUN_ELEVATION_BINS_DEG, float("inf"))
    labelled = lead_zero(frame=frame).filter(pl.col("om_ghi") > 0.0)
    parts = [
        labelled.filter(
            (pl.col("sun_elevation_deg") > (lower if lower is not None else -90.0))
            & (pl.col("sun_elevation_deg") <= upper)
        )
        .group_by("era_code")
        .agg(
            rebuilt_ratio=(pl.col("ceda_ghi") / pl.col("om_ghi")).median(),
            p10=(pl.col("ceda_ghi") / pl.col("om_ghi")).quantile(0.1),
            p90=(pl.col("ceda_ghi") / pl.col("om_ghi")).quantile(0.9),
            mean_abs_diff=(pl.col("ceda_ghi") - pl.col("om_ghi")).abs().mean(),
            n=pl.len(),
        )
        .with_columns(
            bin=pl.lit(elevation_bin_label(upper=upper, lower=lower)),
            order=pl.lit(index, dtype=pl.Int64),
        )
        for index, (lower, upper) in enumerate(zip(edges, uppers, strict=True))
    ]
    return pl.concat(parts).sort("era_code", "order").drop("order")


def era_1_irradiance_note(*, frame: pl.DataFrame) -> str:
    """Name the post-PS47 irradiance mismatch for the labels of the exploratory solar rows.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        A phrase such as `irradiance construction differs after PS47 (ratio 1.11 at 06 UTC, 0.86 at
        18 UTC)`, listing the hours of day whose rebuilt ratio misses `IRRADIANCE_RATIO_BOUNDS`, or
        an empty string where every hour matches.
    """
    low, high = IRRADIANCE_RATIO_BOUNDS
    missed = [
        f"{row['rebuilt_ratio']:.2f} at {row['hour_of_day']:02d} UTC"
        for row in irradiance_ratios_by_era_hour(frame=frame)
        .filter(pl.col("era_code") == 1)
        .iter_rows(named=True)
        if not low <= row["rebuilt_ratio"] <= high
    ]
    if not missed:
        return ""
    return f"irradiance construction differs after PS47 (ratio {', '.join(missed)})"


# --- Row builders ---------------------------------------------------------------------------------


def read_ceda_frame(*, name: str, directory: Path = UKV_VS_ERA5_DIR) -> pl.DataFrame:
    """Read one built frame of the CEDA-against-ERA5 study, in the study's months.

    Args:
        name: The frame's file name.
        directory: The study's folder.

    Returns:
        The rows of `STUDY_MONTHS`, without the study's folds, eras, and shuffled columns.
    """
    frame = pl.read_parquet(directory / name)
    drop = [
        column
        for column in frame.columns
        if column in CEDA_DROPPED_COLUMNS or column.endswith(SHUFFLED_SUFFIX)
    ]
    return frame.drop(drop).filter(pl.col("month").is_in(list(STUDY_MONTHS))).sort("site", "time")


def _finish(*, frame: pl.DataFrame, shuffle_groups: Sequence[Sequence[str]]) -> pl.DataFrame:
    """Label eras and folds, and add the shuffled control columns.

    Args:
        frame: Common rows with every arm column.
        shuffle_groups: One column group per archive, each shuffled under its own permutation.

    Returns:
        The rows sorted by site and time, because the shuffles and XGBoost's row subsampling both
        depend on row order.
    """
    cut, _ = with_covering_folds(frame=frame.sort("site", "time"))
    return climatology_permutation(
        frame=cut,
        column_groups=shuffle_groups,
        by=SHUFFLE_GROUPS,
        seed=SHUFFLE_SEED,
        suffix=SHUFFLED_SUFFIX,
    )


def wind_step_months(*, base: pl.DataFrame) -> dict[str, float]:
    """Find the months that `OPEN_METEO_WIND_STEP_DAYS` removes more than the loss limit from.

    Args:
        base: The CEDA-against-ERA5 study's wind rows in the study's months.

    Returns:
        The share of each such month's rows that lie in the spans, for the months over
        `MAX_MONTH_LOSS_SHARE`. Those months are dropped from every wind arm, and the other
        months lose only the hours inside the spans.
    """
    kept = base.filter(~in_wind_step_days())
    return {
        month: share
        for month, share in loss_by_month(base=base, kept=kept).items()
        if share > MAX_MONTH_LOSS_SHARE
    }


def with_rescaled_speed(*, frame: pl.DataFrame, model_free: pl.DataFrame) -> pl.DataFrame:
    """Add Open-Meteo's 10 m speed rescaled to CEDA's level, learned on the training folds.

    The scale is each site's median ratio of CEDA's to Open-Meteo's 10 m speed at the instants where
    CEDA's lead is 0, over the months of the site's other folds, so a row's scale never reads the
    fold the row is scored in. It is the simplest calibrator, and the transfer scoring uses it to
    test whether a rescale absorbs the transfer penalty.

    Args:
        frame: Wind rows carrying `site`, `month`, `fold` and `om_speed_10m`.
        model_free: `model_free_hours`'s result.

    Returns:
        `frame` with `om_speed_10m_rescaled`, in `frame`'s row order.
    """
    ratios = (
        lead_zero(frame=model_free)
        .filter(~pl.col("om_wind_step") & (pl.col("ceda_speed_10m_m_s") > 1.0))
        .select(
            "site",
            train_month="month",
            ratio=pl.col("ceda_speed_10m_m_s") / pl.col("om_speed_10m_m_s"),
        )
    )
    folds = frame.select("site", "month", "fold").unique()
    per_fold = (
        folds.rename({"fold": "scored_fold"})
        .join(folds.rename({"month": "train_month", "fold": "train_fold"}), on="site")
        .filter(pl.col("train_fold") != pl.col("scored_fold"))
        .join(ratios, on=["site", "train_month"])
        .group_by("site", "scored_fold")
        .agg(scale=pl.col("ratio").median())
        .rename({"scored_fold": "fold"})
    )
    return (
        frame.join(per_fold, on=["site", "fold"], how="left", maintain_order="left")
        .with_columns(om_speed_10m_rescaled=pl.col("om_speed_10m") * pl.col("scale"))
        .drop("scale")
    )


def wind_rows(
    *, base: pl.DataFrame, open_meteo: pl.DataFrame, model_free: pl.DataFrame
) -> pl.DataFrame:
    """Build the wind farms' fit rows: CEDA's matched 10 m wind beside Open-Meteo's.

    The hours in `OPEN_METEO_WIND_STEP_DAYS` are dropped from every arm, and so are the months that
    lose more than `MAX_MONTH_LOSS_SHARE` of their rows to them.

    Args:
        base: The CEDA-against-ERA5 study's wind rows in the study's months.
        open_meteo: `read_open_meteo`'s result.
        model_free: `model_free_hours`'s result, which the speed rescale is learned from.

    Returns:
        One row per (site, hour) with the centred power, both archives' three wind columns,
        `constrained`, `cap_mw`, `month`, `era_code`, `fold`, the shuffled columns, and
        `om_speed_10m_rescaled`.
    """
    ceda_speed, ceda_sin, ceda_cos = wind_columns(archive="ceda")
    om_speed, om_sin, om_cos = wind_columns(archive="om")
    om_wind = open_meteo.select(
        "site",
        "time",
        pl.col("om_speed_10m_m_s").alias(om_speed),
        pl.col("om_direction_10m_deg").radians().sin().alias(om_sin),
        pl.col("om_direction_10m_deg").radians().cos().alias(om_cos),
    ).drop_nulls()
    kept = base.select(
        "site",
        "time",
        "power_mw",
        "effective_capacity_mw",
        "constrained",
        "cap_mw",
        "hour_of_day",
        "day_of_year",
        "month",
        ceda_speed,
        ceda_sin,
        ceda_cos,
    ).join(om_wind, on=["site", "time"], how="inner")
    dropped = list(wind_step_months(base=base))
    kept = kept.filter(~in_wind_step_days() & ~pl.col("month").is_in(dropped))
    finished = _finish(
        frame=kept,
        shuffle_groups=[wind_columns(archive=a) for a in ARCHIVES],
    )
    return with_rescaled_speed(frame=finished, model_free=model_free)


def open_meteo_temperature_two_end_mean(
    *, frame: pl.DataFrame, open_meteo: pl.DataFrame
) -> pl.DataFrame:
    """Add Open-Meteo's temperature as the mean of the instants at each hour's two ends.

    The solar power hour ends at its label, and CEDA's temperature is the mean of the instants at
    the label and the hour before it, so Open-Meteo's is built the same way.

    Args:
        frame: Rows carrying `site` and `time`.
        open_meteo: `read_open_meteo`'s result.

    Returns:
        `frame` with `om_temp`, null where either instant is missing from Open-Meteo.
    """
    return with_neighbouring_hours(
        frame=frame,
        source=open_meteo.select("site", "time", temperature="om_temp_c"),
        columns={"om_temp_previous": ("temperature", -1), "om_temp_label": ("temperature", 0)},
    ).with_columns(om_temp=(pl.col("om_temp_previous") + pl.col("om_temp_label")) / 2.0)


def ceda_irradiance_for(*, ceda: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Add CEDA's rebuilt hourly irradiance to rows, dropping the rows CEDA cannot serve.

    Args:
        ceda: `read_ceda_instants`'s result for the solar sites.
        frame: Rows carrying `site` and `time`.

    Returns:
        `frame` with `ceda_ghi`.
    """
    return frame.join(ceda.select("site", "time", "ceda_ghi"), on=["site", "time"], how="inner")


def with_offset_removed_temperature(
    *, frame: pl.DataFrame, model_free: pl.DataFrame
) -> pl.DataFrame:
    """Add Open-Meteo's temperature minus its lead-0 offset from CEDA, learned on training folds.

    The offset is each site's mean of Open-Meteo's minus CEDA's temperature at the instants where
    CEDA's lead is 0 (instants, not the two-end means of the arms), over the months of the site's
    other folds. A row's offset therefore never reads the fold the row is scored in.

    Args:
        frame: Solar rows carrying `site`, `month`, `fold` and `om_temp`.
        model_free: `model_free_hours`'s result.

    Returns:
        `frame` with `om_temp_offset_removed`, in `frame`'s row order.
    """
    offsets = (
        lead_zero(frame=model_free)
        .drop_nulls(["om_temp_c", "ceda_temp_c"])
        .select("site", train_month="month", offset=pl.col("om_temp_c") - pl.col("ceda_temp_c"))
    )
    folds = frame.select("site", "month", "fold").unique()
    per_fold = (
        folds.rename({"fold": "scored_fold"})
        .join(folds.rename({"month": "train_month", "fold": "train_fold"}), on="site")
        .filter(pl.col("train_fold") != pl.col("scored_fold"))
        .join(offsets, on=["site", "train_month"])
        .group_by("site", "scored_fold")
        .agg(offset=pl.col("offset").mean())
        .rename({"scored_fold": "fold"})
    )
    return (
        frame.join(per_fold, on=["site", "fold"], how="left", maintain_order="left")
        .with_columns(om_temp_offset_removed=pl.col("om_temp") - pl.col("offset"))
        .drop("offset")
    )


def solar_rows(
    *,
    base: pl.DataFrame,
    open_meteo: pl.DataFrame,
    ceda_irradiance: pl.DataFrame,
    model_free: pl.DataFrame,
) -> pl.DataFrame:
    """Build the solar farms' fit rows: each archive's global irradiance and temperature.

    Args:
        base: The CEDA-against-ERA5 study's solar rows in the study's months.
        open_meteo: `read_open_meteo`'s result.
        ceda_irradiance: `read_ceda_instants`'s result at the solar sites.
        model_free: `model_free_hours`'s result, which the temperature offset is learned from.

    Returns:
        One row per (site, daytime hour) with the power, the geometry, both archives' irradiance
        and temperature, `constrained`, `cap_mw`, `month`, `era_code`, `fold`, the shuffled
        columns, and `om_temp_offset_removed`.
    """
    _, ceda_temp = solar_columns(archive="ceda")
    om_ghi, om_temp = solar_columns(archive="om")
    kept = base.select(
        "site",
        "time",
        "power_mw",
        "effective_capacity_mw",
        *SOLAR_SHARED_FEATURES[:3],
        "hour_of_day",
        "day_of_year",
        "cap_mw",
        "constrained",
        "month",
        pl.col("ukv_ceda_temp").alias(ceda_temp),
    )
    joined = ceda_irradiance_for(ceda=ceda_irradiance, frame=kept).join(
        open_meteo.select("site", "time", "om_ghi"), on=["site", "time"]
    )
    with_temperature = open_meteo_temperature_two_end_mean(frame=joined, open_meteo=open_meteo)
    complete = with_temperature.drop("om_temp_previous", "om_temp_label").drop_nulls(
        [om_ghi, om_temp]
    )
    finished = _finish(
        frame=complete,
        shuffle_groups=[solar_columns(archive=a) for a in ARCHIVES],
    )
    return with_offset_removed_temperature(frame=finished, model_free=model_free)


def era_0_solar_rows(*, solar: pl.DataFrame, model_free: pl.DataFrame) -> pl.DataFrame:
    """Cut the solar rows of era 0 alone, with folds cut inside era 0.

    **Training and scoring on era 0 only means the irradiance construction that differs after PS47
    cannot reach the planned solar fits.** The shuffled columns stay as built, because a shuffle
    stays inside one site, month and hour. The offset is relearned on the new folds.

    Args:
        solar: `solar_rows`' result.
        model_free: `model_free_hours`'s result.

    Returns:
        The rows before the PS47 month with `era_code` 0 and folds that cover every calendar month
        occurring in more than one year.

    Raises:
        ValueError: If no rotation of the folds covers every calendar month.
    """
    era_0 = solar.filter(pl.col("month") < STRADDLING_MONTHS[1]).drop(
        "fold", "era", "era_code", "om_temp_offset_removed"
    )
    designs = search_fold_offsets(frame=era_0, first_months=())
    if not designs:
        msg = "no fold rotation leaves every calendar month of era 0 with a training row"
        raise ValueError(msg)
    cut = cut_eras(frame=era_0, first_months=(), fold_offsets=dict(designs[0]))
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=cut))
    return with_offset_removed_temperature(frame=cut, model_free=model_free)


def check_sorted_and_unique(*, frame: pl.DataFrame, name: str) -> None:
    """Raise unless the rows are sorted by site and time and no (site, time) repeats.

    Row order is part of the fit: XGBoost's row subsampling and the control shuffles both depend on
    it, and a repeated key would duplicate a row in every arm.

    Args:
        frame: Rows carrying `site` and `time`.
        name: The frame's name, for the message.

    Raises:
        ValueError: If the rows are out of order or a key repeats.
    """
    if frame.select("site", "time").is_duplicated().any():
        msg = f"{name} holds a repeated (site, time)"
        raise ValueError(msg)
    if not frame.select("site", "time").equals(frame.sort("site", "time").select("site", "time")):
        msg = f"{name} is not sorted by site and time"
        raise ValueError(msg)


def check_rows(
    *, wind: pl.DataFrame, solar: pl.DataFrame, solar_era_0: pl.DataFrame | None = None
) -> None:
    """Raise unless every arm's columns are present on every row and the widths match.

    Args:
        wind: The wind rows.
        solar: The solar rows.
        solar_era_0: The solar rows of era 0 alone, checked like the solar rows when given.
    """
    check_arm_widths()
    checked = [("wind", wind, wind_arm_columns), ("solar", solar, solar_arm_columns)]
    if solar_era_0 is not None:
        checked.append(("solar era 0", solar_era_0, solar_arm_columns))
    for name, frame, function in checked:
        check_sorted_and_unique(frame=frame, name=name)
        columns = [
            column
            for archive in ARCHIVES
            for shuffled in (False, True)
            for column in function(archive=archive, shuffled=shuffled)
        ]
        extra = (
            ["om_temp_offset_removed"]
            if function is solar_arm_columns
            else ["om_speed_10m_rescaled"]
        )
        check_no_missing(
            frame=frame, columns=["power_mw", "effective_capacity_mw", *columns, *extra]
        )
        check_no_backfill(frame=frame)


def transfer_frame(*, frame: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Return the Open-Meteo values under the CEDA arm's column names, for the transfer scoring.

    Args:
        frame: The wind or solar rows.
        domain: `wind` or `solar`.

    Returns:
        `time` and `fold` with every feature column of the CEDA arm holding Open-Meteo's value, and
        the shared features unchanged.

    Raises:
        ValueError: If the domain is neither `wind` nor `solar`.
    """
    if domain == "wind":
        names = dict(zip(wind_columns(archive="ceda"), wind_columns(archive="om"), strict=True))
        shared = WIND_SHARED_FEATURES
    elif domain == "solar":
        names = dict(zip(solar_columns(archive="ceda"), solar_columns(archive="om"), strict=True))
        shared = SOLAR_SHARED_FEATURES
    else:
        msg = f"unknown domain {domain!r}"
        raise ValueError(msg)
    return frame.select("time", "fold", *shared, **{ceda: pl.col(om) for ceda, om in names.items()})


def loss_by_month(*, base: pl.DataFrame, kept: pl.DataFrame) -> dict[str, float]:
    """Return each study month's share of the base rows that Open-Meteo's gaps removed.

    Args:
        base: The rows before the Open-Meteo join.
        kept: The rows after it.

    Returns:
        The share lost, by month, for every month.
    """
    before = base.group_by("month").agg(n=pl.len())
    after = kept.group_by("month").agg(m=pl.len())
    joined = before.join(after, on="month", how="left").with_columns(pl.col("m").fill_null(0))
    return {
        row["month"]: 1.0 - row["m"] / row["n"]
        for row in joined.sort("month").iter_rows(named=True)
    }


def lossy_months(*, shares: Mapping[str, float]) -> list[str]:
    """List the months that lose more than `MAX_MONTH_LOSS_SHARE` of their hours.

    Args:
        shares: The share lost, by month.

    Returns:
        The months over the limit, in order.
    """
    return [month for month, share in sorted(shares.items()) if share > MAX_MONTH_LOSS_SHARE]


# --- The coverage check ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frames:
    """Every frame the build produces."""

    wind: pl.DataFrame
    solar: pl.DataFrame
    solar_era_0: pl.DataFrame
    model_free: pl.DataFrame
    wind_loss: dict[str, float]
    wind_step_months: dict[str, float]
    solar_loss: dict[str, float]


def month_start(*, month: str) -> datetime:
    """Return the first instant of a `%Y-%m` month.

    Args:
        month: The month label.

    Returns:
        The instant, in UTC.
    """
    return datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)


def month_hours(*, first: str, last: str) -> pl.Series:
    """Return every hour from the start of one month to the end of another, in UTC.

    Args:
        first: The first month, such as `2025-06`.
        last: The last month, included.

    Returns:
        The hours, as a UTC datetime series.
    """
    start = month_start(month=first)
    final = month_start(month=last)
    end = final.replace(year=final.year + (final.month == 12), month=final.month % 12 + 1)
    return pl.datetime_range(start, end, interval="1h", time_zone="UTC", eager=True, closed="left")


def build_frames(*, ukv: UkvStores, read_values: bool) -> Frames:
    """Build the wind rows, the solar rows, and the model-free hours.

    Args:
        ukv: The opened stores.
        read_values: Whether to read CEDA values (the check without them still decides rows).

    Returns:
        The three frames and each row set's monthly loss to Open-Meteo's gaps.
    """
    open_meteo = read_open_meteo()
    wind_base = read_ceda_frame(name=CEDA_FRAME_NAMES[0])
    solar_base = read_ceda_frame(name=CEDA_FRAME_NAMES[1])
    hours = month_hours(first=FIRST_MONTH, last=LAST_MONTH)
    free = model_free_hours(ukv=ukv, open_meteo=open_meteo, hours=hours, read_values=read_values)
    solar_hours = solar_base["time"].unique().sort()
    ceda_solar = read_ceda_instants(
        ukv=ukv,
        sites=pv_sites().select("site", "latitude", "longitude"),
        hours=solar_hours,
        read_values=read_values,
    )
    wind = wind_rows(base=wind_base, open_meteo=open_meteo, model_free=free)
    step_months = wind_step_months(base=wind_base)
    wind_base = wind_base.filter(~pl.col("month").is_in(list(step_months)))
    solar = solar_rows(
        base=solar_base, open_meteo=open_meteo, ceda_irradiance=ceda_solar, model_free=free
    )
    return Frames(
        wind=wind,
        solar=solar,
        solar_era_0=era_0_solar_rows(solar=solar, model_free=free),
        model_free=free,
        wind_loss=loss_by_month(base=wind_base, kept=wind),
        wind_step_months=step_months,
        solar_loss=loss_by_month(base=solar_base, kept=solar),
    )


def coverage_lines(*, ukv: UkvStores, frames: Frames) -> tuple[list[str], list[str]]:
    """Describe the row sets, and list the guards that failed.

    Args:
        ukv: The opened stores.
        frames: `build_frames`'s result.

    Returns:
        Markdown lines, and the failed guards (empty when the check passes).
    """
    failures: list[str] = []
    _, ceda_lossy = hours_lost_lines(ukv=ukv)
    over = sorted(set(ceda_lossy) & set(STUDY_MONTHS))
    if over:
        failures.append(f"CEDA loses more than 25% of the hours of {over}")
    for name, shares in (("wind", frames.wind_loss), ("solar", frames.solar_loss)):
        failures += [
            f"{name} rows lose over 25% of {month} to Open-Meteo gaps ({shares[month]:.0%})"
            for month in lossy_months(shares=shares)
        ]
    failures += unit_guard_failures(frame=frames.model_free)
    failures += irradiance_guard_failures(frame=frames.model_free)
    notes = irradiance_mismatch_notes(frame=frames.model_free)
    lines = [
        "### Coverage check",
        "",
        *(f"- KNOWN MISMATCH, not a failure: {note}" for note in notes),
        (
            f"- Study months: {len(STUDY_MONTHS)} ({STUDY_MONTHS[0]} to {STUDY_MONTHS[-1]}, "
            f"without {', '.join(STRADDLING_MONTHS)})."
        ),
    ]
    for name, frame, shares in (
        ("wind", frames.wind, frames.wind_loss),
        ("solar", frames.solar, frames.solar_loss),
    ):
        per_site = frame.group_by("site").agg(n=pl.len()).sort("site")
        per_era = frame.group_by("era_code").agg(n=pl.len(), months=pl.col("month").n_unique())
        lines += [
            (
                f"- {name}: {frame.height:,} rows; per site {dict(per_site.iter_rows())}; "
                f"per era {sorted(per_era.iter_rows())}."
            ),
            f"- {name}: largest monthly loss to Open-Meteo gaps {max(shares.values()):.1%}.",
        ]
    lines.append(
        "- Wind: the spans in which Open-Meteo's 10 m speed is built differently ("
        + ", ".join(f"{first} to {last}" for first, last in OPEN_METEO_WIND_STEP_DAYS)
        + ") are dropped from every wind arm, and so are the months they empty: "
        + (
            ", ".join(f"{m} ({share:.0%})" for m, share in sorted(frames.wind_step_months.items()))
            or "none"
        )
        + "."
    )
    lines.append(
        f"- Solar rows of era 0 alone (planned-scope sensitivity): {frames.solar_era_0.height:,}."
    )
    lines += [
        f"- Irradiance ratio, era {row['era_code']}, sun {row['bin']}: {row['rebuilt_ratio']:.3f} "
        f"({row['n']:,} lead-0 rows)."
        for row in irradiance_ratios_by_elevation(frame=frames.model_free).iter_rows(named=True)
    ]
    lines.append(
        f"- Model-free hours: {frames.model_free.height:,} rows, "
        f"{lead_zero(frame=frames.model_free).height:,} at lead 0."
    )
    lines.append(
        "- Arm feature columns: wind "
        f"{[len(wind_arm_columns(archive=a)) for a in ARCHIVES]}, solar "
        f"{[len(solar_arm_columns(archive=a)) for a in ARCHIVES]}."
    )
    check_rows(wind=frames.wind, solar=frames.solar, solar_era_0=frames.solar_era_0)
    return lines, failures


# --- Writing --------------------------------------------------------------------------------------


def _file_hash(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


README_TEXT: Final[str] = """# UKV from CEDA against UKV from Open-Meteo, rows and results

Private: this folder holds per-generator values and is never published.

- `wind_rows.parquet`, `solar_rows.parquet`, `solar_rows_era0.parquet`, `model_free_hours.parquet`,
  `build.json`
  (`ukv_ceda_vs_openmeteo_build.py`): the fit rows of each domain with every arm's columns and the
  shuffled controls, the hours both archives cover at the nine generator sites with each archive's
  temperature, 10 m wind, and global irradiance, and the hashes of the inputs.
- `direct_report.md`, `direct_differences.parquet` (`ukv_ceda_vs_openmeteo_compare.py`): the
  model-free comparison and its diagnostics.
- `losses_<domain>.parquet`, `losses_<domain>.fingerprint` (`ukv_ceda_vs_openmeteo_fit.py`): every
  fit's per-row signed and absolute errors with the site, time, fold, seed, arm, setting, and
  scoring archive. The prediction is the actual power plus the signed error, so the actual power
  is joined from the rows.
- `intervals.parquet`, `report.md`, `decision.md` (`ukv_ceda_vs_openmeteo_fit.py`): every interval,
  every table the page quotes, and the decision rule applied.
- `superseded/`: earlier outputs, moved here before a re-run, because each script refuses to
  overwrite.
"""


ROW_FILE_NAMES: Final[tuple[str, ...]] = (WIND_ROWS_NAME, SOLAR_ROWS_NAME, SOLAR_ERA_0_ROWS_NAME)
"""The row files the fit reads, whose hashes the stamp records."""


def write_outputs(
    *, output_dir: Path, ukv: UkvStores, frames: Frames, guards_failed: Sequence[str] = ()
) -> None:
    """Write the frames, the stamp and the README into a new write-once folder.

    Args:
        output_dir: The folder to write.
        ukv: The opened stores, for the status hashes and snapshots.
        frames: `build_frames`'s result.
        guards_failed: The guards that failed, recorded in the stamp as `guards_passed`. The fit
            refuses to run unless the stamp says every guard passed and the row files still hash
            to the stamped values.

    Raises:
        FileExistsError: If an output already exists, before anything is written.
    """
    paths = {
        name: output_dir / name
        for name in (*ROW_FILE_NAMES, MODEL_FREE_NAME, STAMP_NAME, README_NAME)
    }
    refuse_to_overwrite(paths=paths.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    frames.wind.write_parquet(paths[WIND_ROWS_NAME])
    frames.solar.write_parquet(paths[SOLAR_ROWS_NAME])
    frames.solar_era_0.write_parquet(paths[SOLAR_ERA_0_ROWS_NAME])
    frames.model_free.write_parquet(paths[MODEL_FREE_NAME])
    ceda_stamp = json.loads((UKV_VS_ERA5_DIR / CEDA_BUILD_STAMP_NAME).read_text())
    stamp = {
        "status_hashes": [
            hashlib.sha256(store.statuses.tobytes()).hexdigest() for store in ukv.stores
        ],
        "snapshot_ids": list(ukv.snapshot_ids),
        "input_hashes": {
            **{name: _file_hash(path=UKV_VS_ERA5_DIR / name) for name in CEDA_FRAME_NAMES},
            OPEN_METEO_PATH.name: _file_hash(path=OPEN_METEO_PATH),
        },
        "effective_capacity_source": {
            "study": "ukv_ceda_vs_era5",
            "delta_versions": ceda_stamp["delta_versions"],
        },
        "guards_passed": not guards_failed,
        "row_file_hashes": {name: _file_hash(path=paths[name]) for name in ROW_FILE_NAMES},
        "rows": {
            "wind": frames.wind.height,
            "solar": frames.solar.height,
            "solar_era_0": frames.solar_era_0.height,
            "model_free": frames.model_free.height,
        },
        "study_months": list(STUDY_MONTHS),
        "wind_step_days": [
            [first.isoformat(), last.isoformat()] for first, last in OPEN_METEO_WIND_STEP_DAYS
        ],
        "wind_step_months_dropped": frames.wind_step_months,
        "era_1_irradiance_matched": not irradiance_mismatch_notes(frame=frames.model_free),
        "era_1_irradiance_note": era_1_irradiance_note(frame=frames.model_free),
        "margins_pp": {"wind": MARGIN_WIND_PP, "solar": MARGIN_SOLAR_PP},
    }
    paths[STAMP_NAME].write_text(json.dumps(stamp, indent=2, sort_keys=True))
    paths[README_NAME].write_text(README_TEXT)


def dry_run(*, month: str) -> list[str]:
    """Build one month of the model-free hours with real CEDA values, and write nothing.

    Args:
        month: The month, such as `2025-06`.

    Returns:
        Markdown lines about the rows built.
    """
    hours = month_hours(first=month, last=month)
    ukv = open_ukv_stores()
    free = model_free_hours(ukv=ukv, open_meteo=read_open_meteo(), hours=hours, read_values=True)
    zero = lead_zero(frame=free)
    return [
        f"--dry-run {month}: {free.height:,} site-hours, {zero.height:,} at lead 0.",
        f"- Unit guard failures: {unit_guard_failures(frame=free) or 'none'}.",
        f"- Irradiance guard failures: {irradiance_guard_failures(frame=free) or 'none'}.",
        f"- Era 1 irradiance mismatches: {irradiance_mismatch_notes(frame=free) or 'none'}.",
        f"- Nulls per column: {dict(zip(free.columns, free.null_count().row(0), strict=True))}.",
    ]


def main() -> int:
    """Run the coverage check, a one-month dry run, or the full build."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Coverage check; write nothing.")
    mode.add_argument("--dry-run", action="store_true", help="Build one month; write nothing.")
    parser.add_argument("--dry-run-month", default="2025-06")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    if arguments.dry_run:
        sys.stdout.write("\n".join(dry_run(month=arguments.dry_run_month)) + "\n")
        return 0
    ukv = open_ukv_stores()
    frames = build_frames(ukv=ukv, read_values=True)
    lines, failures = coverage_lines(ukv=ukv, frames=frames)
    sys.stdout.write("\n".join(lines) + "\n")
    if failures:
        sys.stdout.write("\nGUARDS FAILED:\n" + "\n".join(f"- {f}" for f in failures) + "\n")
        return 1
    if arguments.check_only:
        sys.stdout.write("\nAll guards passed.\n")
        return 0
    write_outputs(output_dir=arguments.output_dir, ukv=ukv, frames=frames, guards_failed=failures)
    sys.stdout.write(
        f"Wrote {frames.wind.height:,} wind rows, {frames.solar.height:,} solar rows and "
        f"{frames.model_free.height:,} model-free hours to {arguments.output_dir}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
