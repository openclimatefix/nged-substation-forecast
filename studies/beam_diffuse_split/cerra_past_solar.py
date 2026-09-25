"""Score the Copernicus regional reanalysis CERRA in the past-solar weather-products study.

One-off throwaway script for <https://github.com/openclimatefix/nged-substation-forecast/issues/938>,
extending `weather_products.py`'s comparison with CERRA, a 5.5 km regional reanalysis of Europe. The
plan was committed before the first fit, and stays in the git history after `plans/` is emptied at
merge.

**Data.** `data/studies/weather/CERRA/`: two parquet files of 3-hour accumulations in J m⁻², at the
190 grid cells around the generators. `surface_solar_radiation_downwards` is global horizontal
irradiance, and `time_integrated_surface_direct_short_wave_radiation_flux` is the direct beam on a
horizontal surface. CERRA has no analysis product for either field, so each value is the energy over
the 3 hours ending at its label (00, 03, ..., 21 UTC), one to three hours into a short forecast from
the preceding 3-hourly analysis. Dividing by 10,800 s gives the mean flux in W m⁻².

**No hourly values, so hourly means are rebuilt.** `windows_from_accumulation` turns the
accumulations into window means, and `rebuild_hourly_from_windows` rebuilds the mean over each hour
ending at its label from them, through the clear-sky index (`studies.resample.
clear_sky_index_resample`, as `ens_past_solar.py` does for ENS's steps). Each rebuilt hour is a
model, so CERRA's hourly values inside a window are not CERRA's own. No rescaling to the window's
mean is applied; the report prints how far each rebuilt window's mean lies from the window mean.

**Two reference arms given the same 3-hour treatment.** `era5_3h` and `cams_3h` average ERA5's and
CAMS's hourly means over CERRA's own windows (a plain mean where all three hours are present, and
dropped where not) and rebuild them with the same function. `era5_global` minus `era5_3h` is then
the effect of the step width alone, and `cerra_global` minus `era5_3h` is CERRA against ERA5 with
the step width matched. The step width is not matched in `cerra_global` minus `era5_global` or
`cerra_global` minus `cams_global`, and CERRA differs from ERA5 in grid spacing and radiation scheme
as well, so no contrast here separates CERRA's physics from the rest.

**Arms, refit on this row set.** Every arm carries the study's seven shared features (the geometry
and calendar features, `temp_c` and `era_code`) and the arm's own irradiance columns, and every fit
uses `colsample_bytree=1`. CERRA has no temperature field in the download, so every arm reads
ERA5's `temp_c`. `era5_global` and `cams_global` are refit here because the row set is shorter than
the published panel's, and `check_column_counts` raises if a contrast's two arms differ in width.

- `cerra_global`: CERRA's rebuilt global irradiance, read at each generator's nearest cell.
- `cerra_split`: that column, CERRA's rebuilt direct beam, and diffuse as global minus direct,
  clipped at zero.
- `cerra_erbs`: that column, with the Erbs separation model's beam and diffuse.
- `era5_global`, `cams_global`, `era5_3h`, `cams_3h`: the reference arms.

**Row set.** The main study's site-hours (`blend_products._solar_frame`) up to the last CERRA
label, 2026-07-01 00:00 UTC (the window that ends there is 21 to 24 UTC on 30 June), and only the
hours where CERRA, ERA5 and CAMS all have a value, so every arm scores the same rows. The row set
is a near-subset of the main rows and ends about 10 weeks before them.

**Folds cover every calendar month.** The rows are cut into eras at `UKV_UPGRADE_MONTH` (the one
era boundary in this window), and `search_fold_offsets` picks the fold rotation that leaves no
calendar month without a training row. `raise_on_uncovered_months` runs before any fit, and the
report prints the uncovered share this row set would have under the main rows' own fold design.

**Nearest cells.** `derive_nearest_cells` finds each generator's nearest CERRA cell from the grid's
latitude and longitude (`GRID_PATH`), and `check_cells_match` stops the run unless the cells equal
the ones in `generator_cells.parquet`, from which the values are read. The report prints the cell
count and the pooled range of distances, and never a coordinate, a cell index or a per-generator
distance.

**Planned contrasts** (`PLANNED_CONTRASTS`, written into the plan before the first fit, each also
run at the second hyperparameter setting):

- `cerra_global − era5_global`: CERRA against the reanalysis that forces it.
- `cerra_global − cams_global`: CERRA against the best product on the main leaderboard.
- `cerra_global − era5_3h`: CERRA against ERA5 with the step width matched.
- `cerra_split − cerra_erbs`: whether CERRA's own direct beam adds information over Erbs separation.

**Exploratory, labelled so in the report:** `EXPLORATORY_CONTRASTS` (CERRA against `cams_3h`, and
the effect of the step width on ERA5 and CAMS), the four planned contrasts per generator, the
checks on the rebuild, and every arm at the second hyperparameter setting.

Run it with `uv run python studies/beam_diffuse_split/cerra_past_solar.py`, after
`weather_products.py` has built its datasets and `fetch_cerra.py` has written the CERRA files.
`--report-only` rebuilds `report.md` from the saved `losses.parquet` alone, still checking the saved
fingerprint. A re-run first moves `losses.parquet`, `losses.fingerprint` and `report.md` to a
`superseded/` subfolder (`refuse_to_overwrite`). Only one agent may run it at a time, because every
worktree shares one data folder.
"""

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple, cast

import numpy as np
import polars as pl
from blend_products import SOLAR, _solar_frame
from build_dataset import _add_separation_models, _pv_sites
from ens_past_solar import (
    _absolute_table_lines,
    _arm_columns_lines,
    _era5_and_cams_hourly,
    _fingerprint,
)
from run_experiment import MAX_CONCURRENT_FITS, Job, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from station_past_solar import _main_panel_lines
from studies.baselines import hourly_clear_sky
from studies.charts import ProductFamily
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    UKV_UPGRADE_MONTH,
    calendar_month_coverage,
    cut_eras,
    raise_on_uncovered_months,
    search_fold_offsets,
    uncovered_months,
)
from studies.grid_sampling import nearest_cells
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.resample import DEFAULT_DAYLIGHT_FLOOR_W_M2, clear_sky_index_resample
from weather_products import CONTRAST_HEADER, _contrast_line, geometry_lines, with_eras

_LOG: Final[logging.Logger] = logging.getLogger("cerra_past_solar")

CERRA_DIR: Final[Path] = WEATHER_DATA_DIR / "CERRA"
GHI_PATH: Final[Path] = CERRA_DIR / "surface_solar_radiation_downwards_surface.parquet"
BHI_PATH: Final[Path] = CERRA_DIR / (
    "time_integrated_surface_direct_short_wave_radiation_flux_surface.parquet"
)
GHI_VALUE_COLUMN: Final[str] = "surface_solar_radiation_downwards_value"
BHI_VALUE_COLUMN: Final[str] = "time_integrated_surface_direct_short_wave_radiation_flux_value"
GENERATOR_CELLS_PATH: Final[Path] = CERRA_DIR / "generator_cells.parquet"
"""One row per generator with `site`, `y_index`, `x_index` and `distance_km`, which the values are
read through once `check_cells_match` has confirmed them against the grid."""

GRID_PATH: Final[Path] = CERRA_DIR / "grid_latlon.parquet"
"""The grid's `y_index`, `x_index`, `latitude` and `longitude`, one row per cell. The path is a
placeholder until `fetch_cerra.py` writes the grid."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / "cerra_past_solar"
"""Where the script writes `losses.parquet`, `losses.fingerprint`, `report.md`, and a `superseded/`
folder for re-runs."""

WINDOW_HOURS: Final[int] = 3
"""The width of every CERRA accumulation window, and the step between labels."""

WINDOW_SECONDS: Final[float] = WINDOW_HOURS * 3600.0
"""The seconds an accumulation in J m⁻² is divided by to give a mean flux in W m⁻²."""

WINDOW_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The last CERRA label. The row set holds only hours ending at or before it."""

NEGATIVE_FLOOR_W_M2: Final[float] = -1.0
"""A rebuilt hour below this is counted in the report's check on the rebuild."""

CLEAR_DAY_SHARE: Final[float] = 0.1
"""The share of each generator's days, those with the highest daily ERA5 irradiation, whose mean
profile sets the peak hour in `check_peak_hours_agree`."""

MIN_PROFILE_ROWS: Final[int] = 500
"""Hours of day with fewer scored rows than this are left out of the hour-of-day profile."""

GHI_COLUMN: Final[str] = "ghi_cerra"
BHI_COLUMN: Final[str] = "bhi_cerra"
DHI_COLUMN: Final[str] = "dhi_cerra"
ERBS_BHI_COLUMN: Final[str] = "erbs_bhi_cerra"
ERBS_DHI_COLUMN: Final[str] = "erbs_dhi_cerra"
ERA5_3H_COLUMN: Final[str] = "ghi_era5_3h"
CAMS_3H_COLUMN: Final[str] = "ghi_cams_3h"

CERRA_ARM: Final[str] = "cerra_global"
SPLIT_ARM: Final[str] = "cerra_split"
ERBS_ARM: Final[str] = "cerra_erbs"
ERA5_3H_ARM: Final[str] = "era5_3h"
CAMS_3H_ARM: Final[str] = "cams_3h"

ARM_ORDER: Final[tuple[str, ...]] = (
    CERRA_ARM,
    SPLIT_ARM,
    ERBS_ARM,
    "era5_global",
    "cams_global",
    ERA5_3H_ARM,
    CAMS_3H_ARM,
)
"""Every fitted arm, in the order the report and the leaderboard print them."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (CERRA_ARM, "era5_global"),
    (CERRA_ARM, "cams_global"),
    (CERRA_ARM, ERA5_3H_ARM),
    (SPLIT_ARM, ERBS_ARM),
)
"""The four contrasts written into the plan before the first fit: (treatment, reference)."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (CERRA_ARM, CAMS_3H_ARM),
    ("era5_global", ERA5_3H_ARM),
    ("cams_global", CAMS_3H_ARM),
)
"""CERRA against CAMS with the step width matched, and the effect of the step width on ERA5 and
CAMS."""

NAMES: Final[dict[str, str]] = {
    CERRA_ARM: "CERRA",
    SPLIT_ARM: "CERRA with its own direct beam",
    ERBS_ARM: "CERRA with Erbs separation",
    "era5_global": "ERA5",
    "cams_global": "CAMS",
    ERA5_3H_ARM: "ERA5 averaged to 3-hour steps",
    CAMS_3H_ARM: "CAMS averaged to 3-hour steps",
}
"""Every arm's public name, as the page writes it."""

FAMILIES: Final[dict[str, ProductFamily]] = {
    CERRA_ARM: "reanalysis",
    SPLIT_ARM: "reanalysis",
    ERBS_ARM: "reanalysis",
    "era5_global": "reanalysis",
    "cams_global": "satellite",
    ERA5_3H_ARM: "reanalysis",
    CAMS_3H_ARM: "satellite",
}
"""Every arm's family, which sets its colour in `studies.charts`."""

FIRST_MONTHS: Final[tuple[str, ...]] = (UKV_UPGRADE_MONTH,)
"""The first month of every era after the first: the one era boundary in this row set's window."""

SITE_HOURS_HEADING: Final[str] = "CERRA's 3-hour accumulations"
"""The report heading's subject, before its row count and dates."""


class Built(NamedTuple):
    """The row set and the counts the report prints about how it was built.

    Attributes:
        frame: One row per (site, time), with every arm's columns, `era_code` and `fold`.
        candidates: The main study's site-hours before any cut.
        within_window: The main study's site-hours ending at or before `WINDOW_END`.
        clipped_diffuse_hours: The scored hours where CERRA's rebuilt direct beam exceeds its
            rebuilt global irradiance, so diffuse is clipped to zero.
        fold_offsets: The rotation of each era's folds.
        uncovered_main_folds: The share of scored hours in calendar months with no training row
            under the main rows' own fold design.
        rebuilt_negative_shares: Each rebuilt column's share of hours below `NEGATIVE_FLOOR_W_M2`.
        window_gaps: Each rebuilt column's relative gap between the mean of its rebuilt hours and
            the window mean, per daylight window.
        grid_cells: The number of CERRA grid cells the file holds.
        distance_range_km: The pooled range of the nearest cells' distances.
    """

    frame: pl.DataFrame
    candidates: int
    within_window: int
    clipped_diffuse_hours: int
    fold_offsets: Mapping[int, int]
    uncovered_main_folds: float
    rebuilt_negative_shares: dict[str, float]
    window_gaps: dict[str, pl.Series]
    grid_cells: int
    distance_range_km: tuple[float, float]


def derive_nearest_cells(*, grid: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Find each generator's nearest CERRA cell from the grid's latitude and longitude.

    Args:
        grid: One row per cell, with `y_index`, `x_index`, `latitude` and `longitude` in degrees.
            A longitude on 0 to 360 degrees is wrapped to -180 to 180.
        sites: The roster, with `site`, `latitude` and `longitude`.

    Returns:
        One row per generator with `site`, `y_index`, `x_index` and `distance_km`.
    """
    cells = grid.with_row_index("cell_id").with_columns(
        longitude=(pl.col("longitude") + 180.0) % 360.0 - 180.0
    )
    nearest = nearest_cells(sites=sites, cells=cells)
    return nearest.join(
        cells.select("cell_id", "y_index", "x_index"), on="cell_id", how="left"
    ).select("site", "y_index", "x_index", "distance_km")


def check_cells_match(*, derived: pl.DataFrame, saved: pl.DataFrame) -> None:
    """Stop unless the derived nearest cells are the saved ones, generator by generator.

    Args:
        derived: `derive_nearest_cells`'s result.
        saved: The rows of `generator_cells.parquet`, with `site`, `y_index` and `x_index`.

    Raises:
        ValueError: Naming how many generators differ, and never which cell any of them has.
    """
    joined = derived.select("site", "y_index", "x_index").join(
        saved.select("site", "y_index", "x_index"), on="site", how="full", suffix="_saved"
    )
    differs = joined.filter(
        pl.col("site").is_null()
        | pl.col("site_saved").is_null()
        | (pl.col("y_index") != pl.col("y_index_saved"))
        | (pl.col("x_index") != pl.col("x_index_saved"))
    )
    if differs.height:
        msg = (
            f"{differs.height} of {joined.height} generators' nearest CERRA cell differs between "
            "the grid's latitude and longitude and generator_cells.parquet"
        )
        raise ValueError(msg)


def _check_window_labels(*, times: pl.Series) -> None:
    """Raise unless every label is a UTC microsecond timestamp on a whole multiple of 3 hours.

    Args:
        times: A `time` column of window ends.

    Raises:
        ValueError: If the dtype is not `Datetime("us", "UTC")` or a label is off the 3-hour grid.
    """
    if times.dtype != pl.Datetime("us", "UTC"):
        msg = f"window labels must be Datetime('us', 'UTC'), got {times.dtype}"
        raise ValueError(msg)
    off_grid = (
        (times.dt.hour() % WINDOW_HOURS != 0) | (times.dt.minute() != 0) | (times.dt.second() != 0)
    ).sum()
    if off_grid:
        msg = f"{off_grid} window labels are not whole multiples of {WINDOW_HOURS} hours UTC"
        raise ValueError(msg)


def windows_from_accumulation(
    *, accumulation: pl.DataFrame, value_column: str, name: str
) -> pl.DataFrame:
    """Turn CERRA's 3-hour accumulations in J m⁻² into window means in W m⁻².

    Each `valid_time` is the end of the window it accumulates over, so the returned `time` is the
    window's end too. The label is not moved.

    Args:
        accumulation: `site`, `valid_time` (a naive nanosecond timestamp in UTC) and the value.
        value_column: The accumulation's column, in J m⁻².
        name: The name of the returned mean flux column.

    Returns:
        `site`, `time` (`Datetime("us", "UTC")`, the window's end) and `name` in W m⁻².

    Raises:
        ValueError: If `valid_time` is not a naive nanosecond timestamp, or a label is off the
            3-hour grid.
    """
    if accumulation["valid_time"].dtype != pl.Datetime("ns"):
        msg = f"valid_time must be a naive Datetime('ns'), got {accumulation['valid_time'].dtype}"
        raise ValueError(msg)
    windows = accumulation.select(
        "site",
        time=pl.col("valid_time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
        **{name: pl.col(value_column).cast(pl.Float64) / WINDOW_SECONDS},
    )
    _check_window_labels(times=windows["time"])
    return windows


def rebuild_hourly_from_windows(
    *, windows: pl.DataFrame, clear_sky: pl.DataFrame, column: str
) -> pl.DataFrame:
    """Rebuild the mean over each hour from 3-hour window means, through the clear-sky index.

    Each window ending at `T` covers the hours ending at `T - 2`, `T - 1` and `T`. The rebuilt hours
    are those of every window present, and a window missing from the series leaves its three hours
    out. No rescaling to the window's mean is applied.

    Args:
        windows: `site`, `time` (the window's end, on a multiple of 3 hours UTC) and `column`, the
            mean flux over the window in W m⁻².
        clear_sky: `site`, `time` (the hour's end) and `clear_sky_w_m2`, covering every hour from
            2 hours before the first window's end to the last window's end.
        column: The flux column in `windows`, and the name of the rebuilt column.

    Returns:
        `site`, `time` (the hour's end) and `column`, one row per rebuilt hour.

    Raises:
        ValueError: If a label is off the grid, a (site, window) is duplicated, or `clear_sky`
            misses an hour.
    """
    _check_window_labels(times=windows["time"])
    if windows.select("site", "time").is_duplicated().any():
        msg = "windows holds a (site, time) more than once"
        raise ValueError(msg)
    sites = sorted(windows["site"].unique().to_list())
    first_end = cast("datetime", windows["time"].min())
    last_end = cast("datetime", windows["time"].max())
    ends = pl.datetime_range(first_end, last_end, interval="3h", time_zone="UTC", eager=True)
    hours = pl.datetime_range(
        first_end - timedelta(hours=WINDOW_HOURS - 1),
        last_end,
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    roster = pl.DataFrame({"site": sites})
    step_grid = roster.join(pl.DataFrame({"time": ends}), how="cross").sort("site", "time")
    values = (
        step_grid.join(windows.select("site", "time", column), on=["site", "time"], how="left")
        .sort("site", "time")[column]
        .cast(pl.Float64)
        .to_numpy()
        .reshape(len(sites), len(ends))
    )
    hour_grid = (
        roster.join(pl.DataFrame({"time": hours}), how="cross")
        .sort("site", "time")
        .join(clear_sky.select("site", "time", "clear_sky_w_m2"), on=["site", "time"], how="left")
        .sort("site", "time")
    )
    if hour_grid["clear_sky_w_m2"].null_count():
        msg = "clear_sky misses an hour a window covers"
        raise ValueError(msg)
    hourly_clear = hour_grid["clear_sky_w_m2"].to_numpy().reshape(len(sites), len(hours))
    step_clear = hourly_clear.reshape(len(sites), len(ends), WINDOW_HOURS).mean(axis=2)
    step_midpoints = ends.dt.epoch("s").to_numpy() / 3600.0 - WINDOW_HOURS / 2.0
    target_midpoints = hours.dt.epoch("s").to_numpy() / 3600.0 - 0.5
    rebuilt = clear_sky_index_resample(
        values=values,
        step_clear_sky=step_clear,
        step_midpoints=step_midpoints,
        morning=np.mod(step_midpoints, 24.0) < 12.0,
        target_clear_sky=hourly_clear,
        target_midpoints=target_midpoints,
        daylight_floor_w_m2=DEFAULT_DAYLIGHT_FLOOR_W_M2,
    )
    present = np.repeat(~np.isnan(values), WINDOW_HOURS, axis=1)
    return (
        hour_grid.select("site", "time")
        .with_columns(
            **{column: pl.Series(rebuilt.reshape(-1)), "present": pl.Series(present.reshape(-1))}
        )
        .filter(pl.col("present"), pl.col(column).is_not_nan())
        .select("site", "time", column)
    )


def windowed_mean(*, hourly: pl.DataFrame, column: str) -> pl.DataFrame:
    """Average hourly means over CERRA's windows: the three hours ending at 00, 03, ..., 21 UTC.

    Args:
        hourly: `site`, `time` (the end of the hour each mean covers) and `column`.
        column: The hourly mean flux column.

    Returns:
        `site`, `time` (the window's end) and `column`: the plain mean of the three hours, and no
        row for a window that lacks one of them.
    """
    window_end = (pl.col("time") - pl.duration(hours=1)).dt.truncate(
        f"{WINDOW_HOURS}h"
    ) + pl.duration(hours=WINDOW_HOURS)
    return (
        hourly.select("site", "time", column)
        .filter(pl.col(column).is_not_null(), pl.col(column).is_not_nan())
        .group_by("site", window_end.alias("time"))
        .agg(pl.col(column).mean(), hours=pl.len())
        .filter(pl.col("hours") == WINDOW_HOURS)
        .select("site", "time", column)
        .sort("site", "time")
    )


def window_mean_gaps(*, windows: pl.DataFrame, rebuilt: pl.DataFrame, column: str) -> pl.Series:
    """Measure how far each rebuilt window's mean lies from the window mean it was rebuilt from.

    Args:
        windows: `site`, `time` (the window's end) and `column`, the window means.
        rebuilt: `rebuild_hourly_from_windows`'s result for the same windows.
        column: The flux column in both frames.

    Returns:
        The relative gap, `|mean of the three rebuilt hours - window mean| / window mean`, of every
        window whose mean is at least `DEFAULT_DAYLIGHT_FLOOR_W_M2`, and which has all three hours.
    """
    window_end = (pl.col("time") - pl.duration(hours=1)).dt.truncate(
        f"{WINDOW_HOURS}h"
    ) + pl.duration(hours=WINDOW_HOURS)
    rebuilt_means = (
        rebuilt.group_by("site", window_end.alias("time"))
        .agg(rebuilt=pl.col(column).mean(), hours=pl.len())
        .filter(pl.col("hours") == WINDOW_HOURS)
    )
    joined = windows.join(rebuilt_means, on=["site", "time"], how="inner").filter(
        pl.col(column) >= DEFAULT_DAYLIGHT_FLOOR_W_M2
    )
    return ((joined["rebuilt"] - joined[column]).abs() / joined[column]).alias("gap")


def _check_utc_microseconds(*, frame: pl.DataFrame, name: str) -> None:
    """Raise unless a frame's `time` column is `Datetime("us", "UTC")`."""
    if frame["time"].dtype != pl.Datetime("us", "UTC"):
        msg = f"{name}: time must be Datetime('us', 'UTC'), got {frame['time'].dtype}"
        raise ValueError(msg)


def join_rows(
    *,
    base: pl.DataFrame,
    cerra: pl.DataFrame,
    era5_3h: pl.DataFrame,
    cams_3h: pl.DataFrame,
    window_end: datetime = WINDOW_END,
) -> pl.DataFrame:
    """Cut the main rows to the hours every arm covers.

    Keeps the main study's site-hours that end at or before `window_end` and have a value in CERRA,
    ERA5, CAMS and the two 3-hour reference columns, so every arm scores the same rows.

    Args:
        base: The main study's site-hours, with `ghi_era5` and `ghi_cams`, and the columns to keep.
        cerra: `site`, `time`, `ghi_cerra` and `bhi_cerra`, the rebuilt hourly means.
        era5_3h: `site`, `time` and `ghi_era5_3h`.
        cams_3h: `site`, `time` and `ghi_cams_3h`.
        window_end: The last hour's end.

    Returns:
        The joined rows, sorted by site and time, with `era`, `era_code` and `fold` dropped.

    Raises:
        ValueError: If any `time` column is not `Datetime("us", "UTC")`, or a (site, time) repeats.
    """
    for name, frame in (
        ("base", base),
        ("cerra", cerra),
        ("era5_3h", era5_3h),
        ("cams_3h", cams_3h),
    ):
        _check_utc_microseconds(frame=frame, name=name)
    present = [
        pl.col(column).is_not_null() & pl.col(column).is_not_nan()
        for column in (
            GHI_COLUMN,
            BHI_COLUMN,
            ERA5_3H_COLUMN,
            CAMS_3H_COLUMN,
            "ghi_era5",
            "ghi_cams",
        )
    ]
    joined = (
        base.filter(pl.col("time") <= window_end)
        .join(cerra, on=["site", "time"], how="inner")
        .join(era5_3h, on=["site", "time"], how="inner")
        .join(cams_3h, on=["site", "time"], how="inner")
        .filter(*present)
        .drop("era", "era_code", "fold", strict=False)
        .sort("site", "time")
    )
    if joined.select("site", "time").is_duplicated().any():
        msg = "join_rows: a (site, time) is duplicated"
        raise ValueError(msg)
    return joined


def with_diffuse_and_erbs(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Add CERRA's diffuse (global minus direct, clipped at zero) and its Erbs beam and diffuse.

    Args:
        frame: Rows with `time`, `solar_zenith_deg`, `ghi_cerra` and `bhi_cerra`.

    Returns:
        The frame with `dhi_cerra`, `erbs_bhi_cerra` and `erbs_dhi_cerra`, and the number of hours
        whose direct beam exceeds the global irradiance, so that diffuse was clipped to zero.
    """
    clipped = int((frame[BHI_COLUMN] > frame[GHI_COLUMN]).sum())
    separated = _add_separation_models(
        frame=frame.select(
            "time",
            "solar_zenith_deg",
            ghi_w_m2=pl.col(GHI_COLUMN),
            bhi_w_m2=pl.col(BHI_COLUMN),
        )
    )
    return (
        frame.with_columns(
            **{
                DHI_COLUMN: (pl.col(GHI_COLUMN) - pl.col(BHI_COLUMN)).clip(lower_bound=0.0),
                ERBS_BHI_COLUMN: separated["erbs_bhi_w_m2"],
                ERBS_DHI_COLUMN: separated["erbs_dhi_w_m2"],
            }
        ),
        clipped,
    )


def clear_day_peak_hour(*, frame: pl.DataFrame, column: str, ranking_column: str) -> int:
    """Return the hour, UTC, at which a column peaks on each generator's clearest days.

    A day is clear when its total of `ranking_column` is in the top `CLEAR_DAY_SHARE` of the
    generator's days. A row labelled hour 0 belongs to the day before.

    Args:
        frame: `site`, `time` (the hour's end), `column` and `ranking_column`.
        column: The column whose mean by hour on those days is compared.
        ranking_column: The column whose daily total picks the clear days.

    Returns:
        The hour of day, UTC, at the end of the hour where the mean of `column` is highest.
    """
    days = frame.with_columns(day=(pl.col("time") - pl.duration(hours=1)).dt.date())
    totals = days.group_by("site", "day").agg(total=pl.col(ranking_column).sum())
    clear = totals.filter(
        pl.col("total") >= pl.col("total").quantile(1.0 - CLEAR_DAY_SHARE).over("site")
    ).select("site", "day")
    profile = (
        days.join(clear, on=["site", "day"], how="inner")
        .group_by(hour=pl.col("time").dt.hour())
        .agg(mean=pl.col(column).mean())
        .sort("mean", descending=True)
    )
    return int(profile["hour"][0])


def check_peak_hours_agree(*, frame: pl.DataFrame) -> tuple[int, int]:
    """Stop unless CERRA's rebuilt hours peak at the same hour as ERA5's on the clearest days.

    A window convention read wrongly (a label taken as the window's start) moves every rebuilt hour
    by 3 hours without any value looking wrong, and moves the peak with it.

    Args:
        frame: The row set, with `ghi_cerra` and `ghi_era5`.

    Returns:
        CERRA's peak hour and ERA5's, UTC.

    Raises:
        ValueError: If the two peak hours differ.
    """
    peaks = tuple(
        clear_day_peak_hour(frame=frame, column=column, ranking_column="ghi_era5")
        for column in (GHI_COLUMN, "ghi_era5")
    )
    if peaks[0] != peaks[1]:
        msg = f"CERRA's clear-day peak is at hour {peaks[0]} UTC and ERA5's at hour {peaks[1]}"
        raise ValueError(msg)
    return peaks[0], peaks[1]


def uncovered_share(*, frame: pl.DataFrame) -> float:
    """Return the share of scored hours whose calendar month has no training row.

    Args:
        frame: Rows with `site`, `fold` and `time`.

    Returns:
        The rows in the (site, fold, calendar month) cells that `uncovered_months` lists, over all
        rows.
    """
    failures = uncovered_months(coverage=calendar_month_coverage(frame=frame))
    return float(failures["n_scored"].sum()) / frame.height


def choose_fold_offsets(*, frame: pl.DataFrame) -> Mapping[int, int]:
    """Pick the fold rotation that leaves no calendar month without a training row.

    Args:
        frame: Rows with `site`, `month` and `time`, before any fold is cut.

    Returns:
        The first offsets `search_fold_offsets` returns: the design with the fewest rotations.

    Raises:
        ValueError: If no rotation covers every calendar month.
    """
    designs = search_fold_offsets(frame=frame, first_months=FIRST_MONTHS)
    if not designs:
        msg = "no fold rotation leaves every calendar month with a training row"
        raise ValueError(msg)
    return MappingProxyType(dict(designs[0]))


def with_covering_folds(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, Mapping[int, int]]:
    """Cut the era-by-era folds that cover every calendar month, and check that they do.

    Args:
        frame: Rows with `site`, `month` and `time`, before any fold is cut.

    Returns:
        The frame with `era_code`, `era` and `fold`, and the rotation of each era's folds.
    """
    offsets = choose_fold_offsets(frame=frame)
    cut = cut_eras(frame=frame, first_months=FIRST_MONTHS, fold_offsets=offsets)
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=cut))
    return cut, offsets


def _arm_features() -> dict[str, tuple[str, ...]]:
    """Return every arm's feature columns, from one place, in the order the model sees them.

    Returns:
        Arm name to feature columns.
    """
    shared = SOLAR.shared_features
    return {
        CERRA_ARM: (*shared, GHI_COLUMN),
        SPLIT_ARM: (*shared, GHI_COLUMN, BHI_COLUMN, DHI_COLUMN),
        ERBS_ARM: (*shared, GHI_COLUMN, ERBS_BHI_COLUMN, ERBS_DHI_COLUMN),
        "era5_global": (*shared, "ghi_era5"),
        "cams_global": (*shared, "ghi_cams"),
        ERA5_3H_ARM: (*shared, ERA5_3H_COLUMN),
        CAMS_3H_ARM: (*shared, CAMS_3H_COLUMN),
    }


def check_column_counts(
    *, features: Mapping[str, Sequence[str]], contrasts: Sequence[tuple[str, str]]
) -> None:
    """Raise unless every contrast's two arms carry the same number of distinct feature columns.

    Args:
        features: Arm name to feature columns.
        contrasts: The (treatment, reference) pairs to check.

    Raises:
        ValueError: If an arm repeats a column, a contrast names an arm without columns, or a
            contrast pairs two arms of different widths.
    """
    for arm, columns in features.items():
        if len(set(columns)) != len(columns):
            msg = f"{arm} repeats a feature column: {tuple(columns)}"
            raise ValueError(msg)
    for treatment, reference in contrasts:
        for arm in (treatment, reference):
            if arm not in features:
                msg = f"{arm} is in a contrast but has no feature columns"
                raise ValueError(msg)
        if len(features[treatment]) != len(features[reference]):
            msg = (
                f"{treatment} has {len(features[treatment])} columns and {reference} has "
                f"{len(features[reference])}: equal counts are required"
            )
            raise ValueError(msg)


def jobs() -> list[Job]:
    """Return every arm at `pooled` and at `sensitivity`.

    Every arm is refit at the second hyperparameter setting, so that the four planned contrasts,
    and every exploratory contrast that lands near the 5% line, have a second setting.

    Returns:
        One job per arm at each setting.
    """
    features = _arm_features()
    check_column_counts(features=features, contrasts=(*PLANNED_CONTRASTS, *EXPLORATORY_CONTRASTS))
    return [
        (arm, setting, "power_mw", features[arm], hyper_parameters, False)
        for setting, hyper_parameters in (
            ("pooled", PRIMARY_HYPER_PARAMETERS),
            ("sensitivity", SENSITIVITY_HYPER_PARAMETERS),
        )
        for arm in ARM_ORDER
    ]


def _read_windows(*, path: Path, value_column: str, cells: pl.DataFrame, name: str) -> pl.DataFrame:
    """Read one CERRA file at each generator's cell and convert it to window means.

    Args:
        path: The parquet file of accumulations.
        value_column: Its value column, in J m⁻².
        cells: `site`, `y_index` and `x_index`.
        name: The name of the returned mean flux column.

    Returns:
        `windows_from_accumulation`'s result, one row per generator and window.
    """
    accumulation = (
        pl.scan_parquet(path)
        .join(cells.lazy(), on=["y_index", "x_index"], how="inner")
        .select("site", "valid_time", value_column)
        .collect()
    )
    return windows_from_accumulation(
        accumulation=accumulation, value_column=value_column, name=name
    )


def _negative_share(*, rebuilt: pl.DataFrame, column: str) -> float:
    """Return the share of rebuilt hours below `NEGATIVE_FLOOR_W_M2`."""
    return float((rebuilt[column] < NEGATIVE_FLOOR_W_M2).sum()) / rebuilt.height


def build_rows() -> Built:
    """Build this section's row set from the CERRA files and the study's own columns.

    Returns:
        The rows and the counts the report prints.

    Raises:
        ValueError: If the derived nearest cells differ from `generator_cells.parquet`, no fold
            design covers every calendar month, an arm's column holds a missing value, or CERRA's
            and ERA5's peak hours differ.
    """
    sites = _pv_sites()
    saved = pl.read_parquet(GENERATOR_CELLS_PATH)
    check_cells_match(
        derived=derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites), saved=saved
    )
    cells = saved.select("site", "y_index", "x_index")

    base = _solar_frame()
    first_day = cast("datetime", base["time"].min()).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    crop_start = first_day - timedelta(days=1)
    clear_sky = hourly_clear_sky(
        sites=sites, first=crop_start - timedelta(hours=WINDOW_HOURS - 1), last=WINDOW_END
    )

    def cropped(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("time").is_between(crop_start, WINDOW_END))

    ghi_windows = cropped(
        _read_windows(path=GHI_PATH, value_column=GHI_VALUE_COLUMN, cells=cells, name=GHI_COLUMN)
    )
    bhi_windows = cropped(
        _read_windows(path=BHI_PATH, value_column=BHI_VALUE_COLUMN, cells=cells, name=BHI_COLUMN)
    )
    hourly = _era5_and_cams_hourly()
    era5_windows = cropped(windowed_mean(hourly=hourly["era5"], column="ghi_w_m2")).rename(
        {"ghi_w_m2": ERA5_3H_COLUMN}
    )
    cams_windows = cropped(windowed_mean(hourly=hourly["cams"], column="ghi_w_m2")).rename(
        {"ghi_w_m2": CAMS_3H_COLUMN}
    )
    rebuilt = {
        column: rebuild_hourly_from_windows(windows=windows, clear_sky=clear_sky, column=column)
        for column, windows in (
            (GHI_COLUMN, ghi_windows),
            (BHI_COLUMN, bhi_windows),
            (ERA5_3H_COLUMN, era5_windows),
            (CAMS_3H_COLUMN, cams_windows),
        )
    }
    window_frames = {
        GHI_COLUMN: ghi_windows,
        BHI_COLUMN: bhi_windows,
        ERA5_3H_COLUMN: era5_windows,
        CAMS_3H_COLUMN: cams_windows,
    }
    joined = join_rows(
        base=base,
        cerra=rebuilt[GHI_COLUMN].join(rebuilt[BHI_COLUMN], on=["site", "time"], how="inner"),
        era5_3h=rebuilt[ERA5_3H_COLUMN],
        cams_3h=rebuilt[CAMS_3H_COLUMN],
    )
    with_columns, clipped = with_diffuse_and_erbs(frame=joined)
    check_no_missing(
        frame=with_columns,
        columns=[column for columns in _arm_features().values() for column in columns],
    )
    check_peak_hours_agree(frame=with_columns)
    uncovered_main = uncovered_share(
        frame=with_eras(frame=with_columns.drop("era", "era_code", "fold", strict=False))
    )
    frame, offsets = with_covering_folds(frame=with_columns)
    _LOG.info("%d rows in this section's own row set", frame.height)
    grid_cells = int(
        pl.scan_parquet(GHI_PATH)
        .select(pl.struct("y_index", "x_index").n_unique())
        .collect()
        .item()
    )
    distances = saved["distance_km"]
    return Built(
        frame=frame,
        candidates=base.height,
        within_window=base.filter(pl.col("time") <= WINDOW_END).height,
        clipped_diffuse_hours=clipped,
        fold_offsets=offsets,
        uncovered_main_folds=uncovered_main,
        rebuilt_negative_shares={
            column: _negative_share(rebuilt=frame_, column=column)
            for column, frame_ in rebuilt.items()
        },
        window_gaps={
            column: window_mean_gaps(
                windows=window_frames[column], rebuilt=rebuilt[column], column=column
            )
            for column in rebuilt
        },
        grid_cells=grid_cells,
        distance_range_km=(cast("float", distances.min()), cast("float", distances.max())),
    )


def _row_lines(*, built: Built) -> list[str]:
    """Render the row counts, the months dropped, and the fold design.

    Args:
        built: `build_rows`'s result.

    Returns:
        Markdown lines.
    """
    frame = built.frame
    months = sorted(frame["month"].unique().to_list())
    per_site = frame.group_by("site").agg(n=pl.len()).sort("site")
    coverage = calendar_month_coverage(frame=frame)
    smallest_training = cast("int", coverage["n_train"].min())
    return [
        "#### The row set and its folds",
        "",
        (
            f"The main study's common rows number {built.candidates:,}. Cutting them to hours "
            f"ending at or before {WINDOW_END:%Y-%m-%d %H:%M} UTC leaves {built.within_window:,}, "
            f"and keeping the hours where CERRA, ERA5 and CAMS all have a value leaves "
            f"{frame.height:,}, spanning {len(months)} calendar months ({months[0]} to "
            f"{months[-1]})."
        ),
        "",
        "| Generator | Site-hours |",
        "|---|---|",
        *(f"| {row['site']} | {row['n']:,} |" for row in per_site.iter_rows(named=True)),
        "",
        (
            f"The folds are cut inside the two eras that begin at the first month and at "
            f"{FIRST_MONTHS[0]}, rotated by {dict(built.fold_offsets)} (era code to rotation). "
            f"Under these folds {uncovered_share(frame=frame):.1%} of the scored hours fall in a "
            f"calendar month with no training row, and under the main rows' own fold design "
            f"{built.uncovered_main_folds:.1%} would."
        ),
        "",
        (
            f"The smallest number of training rows for any held-out (generator, fold, calendar "
            f"month) is {smallest_training:,}."
        ),
    ]


def _cell_lines(*, built: Built) -> list[str]:
    """Render the grid's cell count and the nearest cells' pooled distance range.

    Args:
        built: `build_rows`'s result.

    Returns:
        Markdown lines.
    """
    low, high = built.distance_range_km
    return [
        "#### The CERRA grid cells",
        "",
        (
            f"The CERRA files hold {built.grid_cells} grid cells. Each generator is read at its "
            f"nearest cell, {low:.1f} to {high:.1f} km from the generator (pooled range)."
        ),
    ]


def _rebuild_lines(*, built: Built) -> list[str]:
    """Render the checks on the rebuilt hourly values.

    Args:
        built: `build_rows`'s result.

    Returns:
        Markdown lines.
    """
    cerra_peak, era5_peak = check_peak_hours_agree(frame=built.frame)
    lines = [
        "#### Checks on the rebuild from 3-hour windows (exploratory)",
        "",
        (
            f"The hour, UTC, at which the mean over each generator's clearest days peaks: CERRA "
            f"{cerra_peak:02d}:00, ERA5 {era5_peak:02d}:00. The script stops unless the two are "
            "equal."
        ),
        "",
        (
            "| Rebuilt column | Share of hours below -1 W m⁻² | Median gap "
            "| Share of windows with gap over 10% |"
        ),
        "|---|---|---|---|",
    ]
    for column, share in built.rebuilt_negative_shares.items():
        gaps = built.window_gaps[column]
        lines.append(
            f"| {column} | {share:.4%} | {gaps.median():.2%} | {(gaps > 0.1).mean():.2%} |"
        )
    lines += [
        "",
        (
            "The gap is the difference between the mean of a window's three rebuilt hours and the "
            "window mean, over the window mean, for windows whose mean is at least "
            f"{DEFAULT_DAYLIGHT_FLOOR_W_M2:.0f} W m⁻²."
        ),
        "",
        (
            f"CERRA's direct beam exceeds its global irradiance in {built.clipped_diffuse_hours:,} "
            "scored hours, and diffuse is clipped to zero there."
        ),
    ]
    return lines


def _hour_profile_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render mean irradiance by hour of day for CERRA, ERA5 and CAMS.

    Args:
        frame: The row set.

    Returns:
        Markdown lines.
    """
    profile = (
        frame.group_by(hour=pl.col("time").dt.hour())
        .agg(
            n=pl.len(),
            cerra=pl.col(GHI_COLUMN).mean(),
            era5=pl.col("ghi_era5").mean(),
            cams=pl.col("ghi_cams").mean(),
        )
        .filter(pl.col("n") >= MIN_PROFILE_ROWS)
        .sort("hour")
    )
    return [
        "#### Hour-of-day profile of CERRA, ERA5 and CAMS irradiance",
        "",
        "| Hour ending (UTC) | Rows | CERRA (W m⁻²) | ERA5 (W m⁻²) | CAMS (W m⁻²) |",
        "|---|---|---|---|---|",
        *(
            f"| {row['hour']:02d}:00 | {row['n']:,} | {row['cerra']:.1f} | {row['era5']:.1f} "
            f"| {row['cams']:.1f} |"
            for row in profile.iter_rows(named=True)
        ),
    ]


def _report(*, built: Built, losses: pl.DataFrame, sites: pl.DataFrame, job_list: list[Job]) -> str:
    """Assemble the markdown report.

    Args:
        built: `build_rows`'s result, whose frame is this section's own row set.
        losses: Every arm's losses, at both `pooled` and `sensitivity` settings.
        sites: The solar roster, for the geometry lines.
        job_list: Every job `jobs()` returns, for the feature-column section.

    Returns:
        The report.
    """
    frame = built.frame
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    site_labels = sorted(frame["site"].unique().to_list())
    lines = [
        (
            f"### {SITE_HOURS_HEADING} on {frame.height:,} common site-hours of solar "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        *_absolute_table_lines(pooled=pooled, arms=ARM_ORDER),
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. The interval is a 95% "
            "bound from resampling whole months and a fitting seed. The four contrasts under "
            "`Planned contrasts` were written into the plan before the first fit; every other "
            "contrast is exploratory."
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Planned contrasts",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
            for t, r in PLANNED_CONTRASTS
        ),
        "",
        "#### Planned contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=sensitivity, treatment=t, reference=r, label="sensitivity")
            for t, r in PLANNED_CONTRASTS
        ),
        "",
        "#### The four planned contrasts, per generator (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in PLANNED_CONTRASTS:
        lines += [
            _contrast_line(
                losses=pooled.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in site_labels
        ]
    lines += [
        "",
        "#### What the 3-hour step does to ERA5 and CAMS, and CERRA against CAMS (exploratory)",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
            for t, r in EXPLORATORY_CONTRASTS
        ),
        "",
        "#### The same contrasts at the second hyperparameter setting (exploratory)",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=sensitivity, treatment=t, reference=r, label="sensitivity")
            for t, r in EXPLORATORY_CONTRASTS
        ),
        "",
        *_row_lines(built=built),
        "",
        *_cell_lines(built=built),
        "",
        *_rebuild_lines(built=built),
        "",
        *_hour_profile_lines(frame=frame),
        "",
        *_main_panel_lines(pooled=pooled, frame=frame),
        "",
    ]
    lines += geometry_lines(sites=sites, noun="solar farms")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build the row set, fit every arm, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    arguments = parser.parse_args()

    sites = _pv_sites()
    built = build_rows()
    frame = built.frame
    _LOG.info("%d rows, %s to %s", frame.height, frame["time"].min(), frame["time"].max())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "losses.parquet"
    fingerprint_path = OUTPUT_DIR / "losses.fingerprint"
    report_path = OUTPUT_DIR / "report.md"

    all_jobs = jobs()
    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)

    if arguments.report_only:
        saved_fingerprint = (
            fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
        )
        if saved_fingerprint != fingerprint:
            msg = (
                f"--report-only: {path} was fitted on a different row set, column set, seed set, "
                "or hyperparameter setting than this code now produces; re-run without "
                "--report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
    else:
        refuse_to_overwrite(paths=[path, fingerprint_path, report_path])
        losses = run_all(dataset=frame, jobs=all_jobs, max_workers=MAX_CONCURRENT_FITS)
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)

    report = _report(built=built, losses=losses, sites=sites, job_list=all_jobs)
    report_path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
