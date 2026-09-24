"""Score ICON-DREAM-EU as a description of past wind, refitting every arm on its own row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, extending
`wind_products.py`'s five-product comparison (ERA5, UKV, ICON-D2, ICON-EU, ICON global) with the
German Weather Service's ICON-DREAM reanalysis. ICON-DREAM-EU is not on Open-Meteo, so its wind is
read from a gridded download (`data/studies/weather/ICON-DREAM-EU/`) rather than fetched at each
generator's coordinates: `WS`, `U`, `V` at ten model levels (65-74) and `WS_10M`, `U_10M`, `V_10M`
at the surface, over 126 cells in a box around the trial area, September 2019 to August 2026.

**This design is pre-registered and fixed before any fit runs, per the `study` skill.** The row set,
the arms, the contrasts, the folds, the seeds and the hyperparameters below are written down before
any result exists, and are not to be changed after seeing one.

**Row set.** `common_rows(joined(sites=sites))` from `wind_products.py` — the same five products'
wind, the same power hour (centred on the label, since wind is instantaneous), the same
zero-half-hour and post-upgrade-tail drops — inner-joined to ICON-DREAM-EU's own columns, at each
generator's nearest cell. **Every arm below, including the five original products, is refit on this
row set**: it differs from `wind_products.py`'s own row set because it stops where ICON-DREAM-EU's
record does (31 August 2026, 10 days short of the other five products' 10 September 2026 end), so a
saved loss from `wind_products.py`'s run cannot be reused without silently comparing two different
row sets. The folds, eras, seeds, `SHARED_FEATURES` and hyperparameter settings are exactly
`wind_products.py`'s.

**Served lead.** ICON-DREAM-EU is not an hourly analysis: DWD assembles its hourly series from short
forecast steps run every 3 hours, so a served hour is a 1, 2 or 3 hour forecast, never a T+0 value.
`STEP_HOURS` and the pre-fit padding-hour evidence below establish which step each hour is. At every
third hour (`h % 3 == 0`, the hour ICON-EU itself is served as a T+0 analysis), ICON-DREAM-EU's
served value is the *longest*-lead step, 3 hours, because DWD's short forecasts start from the
*previous* 3-hourly run.

**Primary arm — `icon_dream_eu_wind`, the same four columns every product gets in the wind study**
(`_wind_columns` from `wind_products.py`): ICON-DREAM-EU's level-72 speed (about 96 m — DWD's own
`generalVerticalLayer` numbering, not a 0-based index; see `LEVEL_HEIGHTS_M`), that level's
direction as sine and cosine from `U` and `V`, and the 10 m speed from `WS_10M`. ERA5 and UKV are
shown their 100 m wind and the ICON products their 80 m wind, as in `wind_products.py`;
ICON-DREAM-EU is shown its native level 72 because DWD does not serve an 80 m or 100 m
interpolation of it.

**Planned contrasts, at both hyperparameter settings — the only ones a recommendation may rest on:**

- `icon_dream_eu_wind − era5_wind`: the two reanalyses.
- `icon_dream_eu_wind − icon_eu_wind`: the reanalysis against ICON-EU, DWD's operational ICON model
  over Europe at the same 6.5 km grid spacing. ICON-DREAM-EU does not use ICON-EU's output; it is
  DWD's own reanalysis run of ICON, with its own data assimilation, nested inside a 13 km global
  run.

**Exploratory arms and contrasts, each labelled so in the report:**

- ICON-DREAM-EU's wind against UKV, ICON-D2 and ICON global.
- `icon_dream_eu_levels`: the four `icon_dream_eu_wind` columns plus the speeds at levels 73 (about
  42 m) and 71 (about 167 m), against `icon_dream_eu_wind` — whether shear across three heights adds
  skill. This arm carries two more columns than its reference, so its comparison is read with the
  column-count caveat the `study` skill states: an arm with more columns can win without carrying
  more information, and this repository's own measurement puts that effect at up to 10% of mean
  absolute error at `colsample_bytree` below 1 and about 0.4% even at 1 (this study's setting).
- A speed-only arm per product (`SHARED_FEATURES` plus one hub-height speed column), showing what
  the speed alone, with no direction, carries.
- The two planned contrasts, by generator (W1-W3) and by calendar year (the same months in every
  year, the "too few months" rule, `bootstrap_year_change` between 2025 and 2026).

**Before any fit runs**, `run_checks` and `_raise_on_failed_checks` establish, and raise if any
fails: that ICON-DREAM-EU's `U`, `V` and `WS` agree (`check_component_speed`), that direction from
`U`/`V` agrees with ERA5's 100 m direction (`check_direction_against_era5`), that the two products'
hour-to-hour changes correlate most at zero offset (`check_timestamp_offset`), and each generator's
nearest cell and its distance (`icon_dream_site_frame`, via `extract_site_series._log_distances`).
**The duplicate-key gate is a hard stop, not a warning**: DWD assembles ICON-DREAM-EU's hourly
series from overlapping short-range forecast steps with no overlap resolved, so a `(valid_time,
model_level, cell_id)` key held by more than one row means a choice between duplicates that needs a
design decision, not code silently picking one (`raise_on_duplicate_keys`).

**The pre-fit checks raise, not just report**, if the offset scan peaks anywhere but zero, if
either component-speed check's median disagreement exceeds `MAX_COMPONENT_SPEED_MEDIAN_DIFF_M_S`,
or if the mean absolute direction disagreement against ERA5 exceeds
`MAX_DIRECTION_MEAN_ABS_DIFF_DEG`. They run before any arm is fitted, in both a fresh run and
`--report-only`.

Run it with `uv run python studies/beam_diffuse_split/wind_icon_dream.py`, after
`fetch_wind_point.py`. `refuse_to_overwrite` on a fresh run means `losses.parquet`,
`losses.fingerprint` and `report.md` each have to move to a `superseded/` subfolder before a
re-run. `--report-only` rebuilds `report.md` from the saved `losses.parquet` alone, fitting
nothing, but still raises if the saved fingerprint (the row set, every job's columns, the seeds,
and the hyperparameters) no longer matches what this code would fit.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl
from build_dataset import _wind_sites
from extract_site_series import ICON_DREAM_CELL_CENTRES, _icon_dream_cell_centres, _log_distances
from fetch_wind_point import output_path_for
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    YearChangeInterval,
    YearInterval,
    bootstrap_absolute,
    bootstrap_difference_by_year,
    bootstrap_year_change,
)
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS, SENSITIVITY_HYPER_PARAMETERS
from studies.grid_sampling import nearest_cells
from studies.guards import refuse_to_overwrite
from weather_products import (
    CONTRAST_HEADER,
    ERA5_BY_YEAR_MONTHS,
    METRIC,
    PERCENTAGE_POINTS,
    _contrast_line,
    _mae,
)
from wind_products import (
    SHARED_FEATURES,
    _hub_height_m,
    _wind_columns,
    common_rows,
    geometry_lines,
    joined,
    with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR_NAME: Final[str] = "wind_icon_dream"
"""The results directory under `sources.STUDY_DATA_DIR / 'past_weather_v2'`."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / OUTPUT_DIR_NAME
"""Where this study writes `losses.parquet`, `report.md`, and a `superseded/` folder for re-runs."""

ICON_DREAM_DIR: Final[Path] = WEATHER_DATA_DIR / "ICON-DREAM-EU"
"""Holds the wind download this script reads: `WS`, `U`, `V`, and their `_10M` siblings."""

WS_FILE: Final[str] = "WS_201909_202608.parquet"
U_FILE: Final[str] = "U_201909_202608.parquet"
V_FILE: Final[str] = "V_201909_202608.parquet"
WS_10M_FILE: Final[str] = "WS_10M_201909_202608.parquet"
U_10M_FILE: Final[str] = "U_10M_201909_202608.parquet"
V_10M_FILE: Final[str] = "V_10M_201909_202608.parquet"

LEVEL_HEIGHTS_M: Final[dict[int, int]] = {74: 10, 73: 42, 72: 96, 71: 167, 70: 253}
"""DWD's `generalVerticalLayer` value to its nominal full-level height in metres.

DWD numbers levels top-down, so 74 (the largest value) is nearest the ground. Read from the
"Height of the full and half levels" table in DWD's ICON-DREAM parameter table
(`.claude/worktrees/scratch/icon-dream-check/param_table.txt`, EU column, full levels): level 74 at
10.000 m, 73 at 42.083 m, 72 at 95.582 m, 71 at 166.626 m, 70 at 253.409 m. The table prints these
with a comma as the decimal separator, not as a thousands separator.
"""

HUB_LEVEL: Final[int] = 72
"""ICON-DREAM-EU's level shown as the study's hub-height wind, about 96 m."""

LEVELS_ARM_EXTRA_LEVELS: Final[tuple[int, int]] = (73, 71)
"""The two extra levels `icon_dream_eu_levels` is shown, about 42 m and 167 m."""

MAX_PLAUSIBLE_CELL_DISTANCE_KM: Final[float] = 10.0
"""A generator's nearest ICON-DREAM-EU cell should sit within about 1.5 cell widths.

ICON-DREAM-EU's triangles are about 6.5 km across (`extract_site_series.py`), so a generator well
inside the download's 126-cell box should read a cell within a few kilometres. A distance beyond
this points at a generator sitting near or outside the box the download covers.
"""

PRODUCT: Final[str] = "icon_dream_eu"
"""This study's own product key, matching `wind_products._wind_columns`' naming."""

PRODUCTS: Final[tuple[str, ...]] = ("era5", "ukv", "icon_d2", "icon_eu", "icon_global", PRODUCT)
"""Every product this study refits, the five original products plus ICON-DREAM-EU."""

HUB_HEIGHT_M: Final[dict[str, int]] = {
    "era5": _hub_height_m(product="era5"),
    "ukv": _hub_height_m(product="ukv"),
    "icon_d2": _hub_height_m(product="icon_d2"),
    "icon_eu": _hub_height_m(product="icon_eu"),
    "icon_global": _hub_height_m(product="icon_global"),
    PRODUCT: LEVEL_HEIGHTS_M[HUB_LEVEL],
}
"""Every product's hub height, for the report. `wind_products._hub_height_m` reads `icon_dream_eu`
as an ICON product (80 m) by its name prefix, which is wrong for this study's own product, so its
entry is set explicitly from `LEVEL_HEIGHTS_M` instead.
"""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (f"{PRODUCT}_wind", "era5_wind"),
    (f"{PRODUCT}_wind", "icon_eu_wind"),
)
"""The two contrasts the recommendations rest on, named before the run.

Whether the reanalysis beats ERA5, the other reanalysis; and whether it beats ICON-EU, DWD's
operational ICON model over Europe at the same 6.5 km grid spacing. Every other contrast in the
report is exploratory.
"""

EXPLORATORY_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (f"{PRODUCT}_wind", "ukv_wind"),
    (f"{PRODUCT}_wind", "icon_d2_wind"),
    (f"{PRODUCT}_wind", "icon_global_wind"),
)
"""ICON-DREAM-EU's wind against the three remaining products, exploratory."""

LEVELS_CONTRAST: Final[tuple[str, str]] = (f"{PRODUCT}_levels", f"{PRODUCT}_wind")
"""Whether the two extra levels (shear) add skill over the hub-height-only arm, exploratory.

`{PRODUCT}_levels` carries two more feature columns than `{PRODUCT}_wind`; read this contrast with
the column-count caveat in this module's docstring and the `study` skill.
"""

LEVELS_VS_OTHERS_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (f"{PRODUCT}_levels", "era5_wind"),
    (f"{PRODUCT}_levels", "icon_eu_wind"),
)
"""Whether the three-level shear arm settles the height question, against ERA5 and ICON-EU,
exploratory. Added after the first run, so a reviewer can check whether a different ICON-DREAM-EU
height would change either planned answer."""

STEP_HOURS: Final[tuple[int, ...]] = (1, 2, 3)
"""ICON-DREAM-EU's served lead in hours: `h % 3 == 0` is step 3, `h % 3 == 1` is step 1, `h % 3 ==
2` is step 2 -- see this module's docstring, "Served lead"."""

SPEED_ONLY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (f"{product}_wind", f"{product}_speed_only") for product in PRODUCTS
)
"""Each product's full arm against its own speed-only arm: what direction adds, exploratory."""

ERA5_YEAR_CHANGE_YEARS: Final[tuple[int, int]] = (2025, 2026)
"""The two years `_by_year_lines` tests for a change in each planned contrast, on matched months."""

MAX_COMPONENT_SPEED_MEDIAN_DIFF_M_S: Final[float] = 0.01
"""`_raise_on_failed_checks` fails if `sqrt(u^2 + v^2)` disagrees with the served WS by more than
this, at the median, at either the hub level or the surface."""

MAX_DIRECTION_MEAN_ABS_DIFF_DEG: Final[float] = 30.0
"""`_raise_on_failed_checks` fails if the direction from `U`/`V` disagrees with ERA5's 100 m
direction by more than this, mean absolute, pooled over every site."""


def filter_nan_padding(*, frame: pl.DataFrame, value_columns: list[str]) -> pl.DataFrame:
    """Drop cfgrib's NaN-padded rows, holding not-a-number in any of `value_columns`.

    cfgrib pads the (time, step) grid to a rectangle at each month boundary, leaving three
    not-a-number rows per (cell[, level]) per month. These are `Float32` `NaN`, not a Polars null,
    so `is_not_nan()` is what removes them; a naive `unique()` on the raw rows can keep one instead
    of the real value.

    Args:
        frame: The raw download, one row per (valid_time[, model_level], cell_id).
        value_columns: The float columns cfgrib pads with not-a-number.

    Returns:
        `frame` without any row holding not-a-number in any of `value_columns`.
    """
    condition = pl.all_horizontal([pl.col(column).is_not_nan() for column in value_columns])
    return frame.filter(condition)


def duplicate_key_counts(*, frame: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    """Return every key in `frame` held by more than one row.

    Args:
        frame: The rows to check, after `filter_nan_padding`.
        key_columns: The columns that should together identify one row.

    Returns:
        One row per duplicated key with `n_rows`, sorted by `n_rows` descending; empty if every key
        is unique.
    """
    return (
        frame.group_by(key_columns)
        .agg(n_rows=pl.len())
        .filter(pl.col("n_rows") > 1)
        .sort("n_rows", descending=True)
    )


def raise_on_duplicate_keys(*, frame: pl.DataFrame, key_columns: list[str], label: str) -> None:
    """Raise if any key in `frame` is held by more than one row, after the NaN filter.

    A hard gate, not a warning: DWD assembles ICON-DREAM-EU's hourly series from overlapping
    short-range forecast steps and keeps every (time, step) value as reported, with no overlap
    resolved, so a duplicated key means a real choice between two published values. Which value to
    keep is a design decision this script does not make silently.

    Args:
        frame: The rows to check, after `filter_nan_padding`.
        key_columns: The columns that should together identify one row.
        label: What to call `frame` in the error message, such as a filename.

    Raises:
        ValueError: Naming the duplicate count and a few example keys.
    """
    duplicates = duplicate_key_counts(frame=frame, key_columns=key_columns)
    if duplicates.height:
        examples = duplicates.head(5).to_dicts()
        msg = (
            f"{label}: {duplicates.height} keys of {key_columns} hold more than one row after the "
            f"NaN filter; examples: {examples}. The choice between duplicates needs a design "
            "decision, so this is a hard gate rather than a silent dedupe."
        )
        raise ValueError(msg)


def speed_at_level(*, frame: pl.DataFrame, level: int) -> pl.DataFrame:
    """Return one model level's rows only, with `model_level` dropped.

    Args:
        frame: Rows carrying `model_level`, DWD's `generalVerticalLayer` numbering.
        level: The level to keep, a key of `LEVEL_HEIGHTS_M`.

    Returns:
        `frame` filtered to `level`, with `model_level` dropped.

    Raises:
        ValueError: If `level` is not one of `LEVEL_HEIGHTS_M`'s keys.
    """
    if level not in LEVEL_HEIGHTS_M:
        msg = f"level {level} is not one of the served levels {sorted(LEVEL_HEIGHTS_M)}"
        raise ValueError(msg)
    return frame.filter(pl.col("model_level") == level).drop("model_level")


def wind_direction_degrees(*, u: pl.Expr, v: pl.Expr) -> pl.Expr:
    """Return the meteorological wind direction, in degrees, that `u` and `v` blow from.

    Meteorological convention: 0 degrees is a wind from the north, 90 from the east, measured
    clockwise — the direction the wind blows *from*, opposite its own velocity vector. For example,
    `u=0, v=-5` (blowing due south) is a wind from the north, 0 degrees; `u=5, v=0` (blowing due
    east) is a wind from the west, 270 degrees.

    Args:
        u: Eastward wind component, m/s (positive eastward).
        v: Northward wind component, m/s (positive northward).

    Returns:
        Direction in degrees, wrapped to [0, 360).
    """
    return (pl.arctan2(-u, -v).degrees() + 360.0) % 360.0


def mean_absolute_angle_difference_deg(*, a_deg: pl.Series, b_deg: pl.Series) -> float:
    """Return the mean absolute difference between two series of circular angles.

    Args:
        a_deg: One series of angles in degrees, 0-360.
        b_deg: The other series of angles in degrees, 0-360, the same length and row order.

    Returns:
        The mean of `abs(((a - b + 180) % 360) - 180)`, in degrees, which never exceeds 180.
    """
    wrapped = ((a_deg - b_deg + 180.0) % 360.0) - 180.0
    return float(pl.select(wrapped.abs().mean()).item())


def _read_level_variable(*, filename: str, value_column: str) -> pl.DataFrame:
    """Read one multi-level ICON-DREAM-EU variable, NaN-filtered and duplicate-key-checked.

    Args:
        filename: One of `WS_FILE`, `U_FILE`, `V_FILE`.
        value_column: The variable's own value column.

    Returns:
        `valid_time`, `model_level`, `cell_id`, `value_column`.

    Raises:
        ValueError: If any `(valid_time, model_level, cell_id)` key holds more than one row.
    """
    frame = filter_nan_padding(
        frame=pl.read_parquet(ICON_DREAM_DIR / filename), value_columns=[value_column]
    )
    raise_on_duplicate_keys(
        frame=frame, key_columns=["valid_time", "model_level", "cell_id"], label=filename
    )
    return frame


def _read_surface_variable(*, filename: str, value_column: str) -> pl.DataFrame:
    """Read one single-level (10 m) ICON-DREAM-EU variable, NaN-filtered and duplicate-checked.

    Args:
        filename: One of `WS_10M_FILE`, `U_10M_FILE`, `V_10M_FILE`.
        value_column: The variable's own value column.

    Returns:
        `valid_time`, `cell_id`, `value_column`.

    Raises:
        ValueError: If any `(valid_time, cell_id)` key holds more than one row.
    """
    frame = filter_nan_padding(
        frame=pl.read_parquet(ICON_DREAM_DIR / filename), value_columns=[value_column]
    )
    raise_on_duplicate_keys(frame=frame, key_columns=["valid_time", "cell_id"], label=filename)
    return frame


def _as_time(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast `valid_time` (naive, implicitly UTC) to a tz-aware `time` column, dropping the old one.

    Args:
        frame: Rows carrying `valid_time`.

    Returns:
        `frame` with `valid_time` replaced by `time`, cast to `Datetime("us", "UTC")`.
    """
    return frame.with_columns(
        time=pl.col("valid_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    ).drop("valid_time")


def icon_dream_cells(*, sites: pl.DataFrame, cell_ids: list[int]) -> pl.DataFrame:
    """Return each site's nearest ICON-DREAM-EU cell, logging and checking its distance.

    Reuses `extract_site_series._icon_dream_cell_centres` (cached at `ICON_DREAM_CELL_CENTRES`,
    already written by the solar round's download) and `studies.grid_sampling.nearest_cells`, the
    same nearest-cell code the solar study's `extract_icon_dream` uses.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.
        cell_ids: The cell ids the download holds.

    Returns:
        One row per site with `site`, `cell_id`, `distance_km`.

    Raises:
        ValueError: If any site's nearest cell sits further than `MAX_PLAUSIBLE_CELL_DISTANCE_KM`
            away, which would mean the site sits near or outside the download's box.
    """
    centres = _icon_dream_cell_centres(cache=ICON_DREAM_CELL_CENTRES, cell_ids=cell_ids)
    nearest = nearest_cells(sites=sites, cells=centres)
    _log_distances(product="ICON-DREAM-EU", nearest=nearest)
    too_far = nearest.filter(pl.col("distance_km") > MAX_PLAUSIBLE_CELL_DISTANCE_KM)
    if too_far.height:
        msg = (
            f"{too_far.height} site(s) sit more than {MAX_PLAUSIBLE_CELL_DISTANCE_KM} km from "
            f"their nearest ICON-DREAM-EU cell: {too_far.select('site', 'distance_km').to_dicts()}"
        )
        raise ValueError(msg)
    return nearest


def icon_dream_site_frame(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return each wind generator's nearest-cell ICON-DREAM-EU wind, hourly.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.

    Returns:
        One row per (site, time) with `speed_hub_icon_dream_eu` (level `HUB_LEVEL`, ~96 m),
        `direction_sin_icon_dream_eu` and `direction_cos_icon_dream_eu` (from `U`/`V` at that
        level), `speed_10m_icon_dream_eu`, and `speed_73_icon_dream_eu` / `speed_71_icon_dream_eu`
        for `LEVELS_ARM_EXTRA_LEVELS`.
    """
    ws = _read_level_variable(filename=WS_FILE, value_column="ws_m_s")
    u = _read_level_variable(filename=U_FILE, value_column="u_m_s")
    v = _read_level_variable(filename=V_FILE, value_column="v_m_s")
    ws_10m = _as_time(_read_surface_variable(filename=WS_10M_FILE, value_column="ws_10m_m_s"))

    cell_ids = sorted(ws["cell_id"].unique().to_list())
    nearest = icon_dream_cells(sites=sites, cell_ids=cell_ids)
    wanted = nearest["cell_id"].unique().to_list()

    speed_name, sine_name, cosine_name, surface_name = _wind_columns(product=PRODUCT)

    hub_u = speed_at_level(frame=u, level=HUB_LEVEL).rename({"u_m_s": "u_hub"})
    hub_v = speed_at_level(frame=v, level=HUB_LEVEL).rename({"v_m_s": "v_hub"})
    hub_ws = speed_at_level(frame=ws, level=HUB_LEVEL).rename({"ws_m_s": speed_name})
    direction_deg = wind_direction_degrees(u=pl.col("u_hub"), v=pl.col("v_hub"))
    hub = _as_time(
        hub_u.join(hub_v, on=["valid_time", "cell_id"])
        .join(hub_ws, on=["valid_time", "cell_id"])
        .with_columns(
            **{
                sine_name: direction_deg.radians().sin(),
                cosine_name: direction_deg.radians().cos(),
            }
        )
        .drop("u_hub", "v_hub")
    )

    extra_levels = [
        _as_time(
            speed_at_level(frame=ws, level=level).rename({"ws_m_s": f"speed_{level}_{PRODUCT}"})
        )
        for level in LEVELS_ARM_EXTRA_LEVELS
    ]

    frame = hub.join(ws_10m.rename({"ws_10m_m_s": surface_name}), on=["time", "cell_id"])
    for extra in extra_levels:
        frame = frame.join(extra, on=["time", "cell_id"])

    return (
        nearest.select("site", "cell_id")
        .join(frame.filter(pl.col("cell_id").is_in(wanted)), on="cell_id")
        .drop("cell_id")
        .sort("site", "time")
    )


def check_component_speed(*, level: int) -> dict[str, float]:
    """Compare `sqrt(u^2 + v^2)` with the served wind speed, at one level, over the whole box.

    Args:
        level: A key of `LEVEL_HEIGHTS_M`.

    Returns:
        `median_abs_diff_m_s` and `p99_abs_diff_m_s`.
    """
    u = speed_at_level(
        frame=_read_level_variable(filename=U_FILE, value_column="u_m_s"), level=level
    )
    v = speed_at_level(
        frame=_read_level_variable(filename=V_FILE, value_column="v_m_s"), level=level
    )
    ws = speed_at_level(
        frame=_read_level_variable(filename=WS_FILE, value_column="ws_m_s"), level=level
    )
    joined_frame = u.join(v, on=["valid_time", "cell_id"]).join(ws, on=["valid_time", "cell_id"])
    computed = (pl.col("u_m_s") ** 2 + pl.col("v_m_s") ** 2).sqrt()
    diff = (computed - pl.col("ws_m_s")).abs()
    result = joined_frame.select(
        median_abs_diff_m_s=diff.median(), p99_abs_diff_m_s=diff.quantile(0.99)
    ).row(0, named=True)
    return {key: float(value) for key, value in result.items()}


def check_component_speed_10m() -> dict[str, float]:
    """Compare `sqrt(u^2 + v^2)` with the served 10 m wind speed, over the whole box.

    Returns:
        `median_abs_diff_m_s` and `p99_abs_diff_m_s`.
    """
    u = _read_surface_variable(filename=U_10M_FILE, value_column="u_10m_m_s")
    v = _read_surface_variable(filename=V_10M_FILE, value_column="v_10m_m_s")
    ws = _read_surface_variable(filename=WS_10M_FILE, value_column="ws_10m_m_s")
    joined_frame = u.join(v, on=["valid_time", "cell_id"]).join(ws, on=["valid_time", "cell_id"])
    computed = (pl.col("u_10m_m_s") ** 2 + pl.col("v_10m_m_s") ** 2).sqrt()
    diff = (computed - pl.col("ws_10m_m_s")).abs()
    result = joined_frame.select(
        median_abs_diff_m_s=diff.median(), p99_abs_diff_m_s=diff.quantile(0.99)
    ).row(0, named=True)
    return {key: float(value) for key, value in result.items()}


def check_direction_against_era5(*, sites: pl.DataFrame) -> dict[str, float]:
    """Return the mean absolute angle difference between ICON-DREAM-EU's and ERA5's directions.

    ICON-DREAM-EU's direction is read at `HUB_LEVEL` (~96 m) from `U` and `V`; ERA5's is its served
    100 m direction, from `fetch_wind_point.py`'s download. Only the direction columns are used;
    speed plays no part in this check.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.

    Returns:
        One entry per site label, plus `"all"`, of the mean absolute angle difference in degrees.
    """
    u = speed_at_level(
        frame=_read_level_variable(filename=U_FILE, value_column="u_m_s"), level=HUB_LEVEL
    ).rename({"u_m_s": "u_hub"})
    v = speed_at_level(
        frame=_read_level_variable(filename=V_FILE, value_column="v_m_s"), level=HUB_LEVEL
    ).rename({"v_m_s": "v_hub"})
    cell_ids = sorted(u["cell_id"].unique().to_list())
    nearest = icon_dream_cells(sites=sites, cell_ids=cell_ids)
    directions = _as_time(
        u.join(v, on=["valid_time", "cell_id"]).with_columns(
            icon_dream_direction=wind_direction_degrees(u=pl.col("u_hub"), v=pl.col("v_hub"))
        )
    )
    per_site_icon_dream = (
        nearest.select("site", "cell_id")
        .join(directions.select("cell_id", "time", "icon_dream_direction"), on="cell_id")
        .drop("cell_id")
    )
    era5 = pl.read_parquet(output_path_for(product="era5")).select(
        "site", "time", era5_direction=pl.col("wind_direction_100m")
    )
    joined_frame = per_site_icon_dream.join(era5, on=["site", "time"], how="inner")
    results = {
        row["site"]: mean_absolute_angle_difference_deg(
            a_deg=joined_frame.filter(pl.col("site") == row["site"])["icon_dream_direction"],
            b_deg=joined_frame.filter(pl.col("site") == row["site"])["era5_direction"],
        )
        for row in sites.select("site").unique().iter_rows(named=True)
    }
    results["all"] = mean_absolute_angle_difference_deg(
        a_deg=joined_frame["icon_dream_direction"], b_deg=joined_frame["era5_direction"]
    )
    return results


OTHER_DIRECTION_PRODUCTS: Final[dict[str, str]] = {
    "ukv": "wind_direction_100m",
    "icon_eu": "wind_direction_80m",
    "icon_d2": "wind_direction_80m",
}
"""Each already-scored product's own served direction column, keyed for
`other_products_direction_vs_era5`."""


def other_products_direction_vs_era5(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return UKV's, ICON-EU's and ICON-D2's mean absolute direction disagreement with ERA5.

    Same method as `check_direction_against_era5`, over exactly `frame`'s row set, so
    ICON-DREAM-EU's own direction disagreement can be read against a same-method baseline from
    three products already trusted, rather than judged on its own as "expected" with no comparison.

    Args:
        frame: The common row set (`icon_dream_common_rows`'s result), for `site` and `time`.

    Returns:
        One mean absolute angle difference in degrees, keyed by `OTHER_DIRECTION_PRODUCTS`.
    """
    rows = frame.select("site", "time")
    era5 = pl.read_parquet(output_path_for(product="era5")).select(
        "site", "time", era5_direction=pl.col("wind_direction_100m")
    )
    results: dict[str, float] = {}
    for product, column in OTHER_DIRECTION_PRODUCTS.items():
        other = pl.read_parquet(output_path_for(product=product)).select(
            "site", "time", other_direction=pl.col(column)
        )
        joined_frame = rows.join(other, on=["site", "time"], how="inner").join(
            era5, on=["site", "time"], how="inner"
        )
        results[product] = mean_absolute_angle_difference_deg(
            a_deg=joined_frame["other_direction"], b_deg=joined_frame["era5_direction"]
        )
    return results


OFFSET_SCAN_HOURS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)
"""The offsets `check_timestamp_offset` scans, in hours."""


def check_timestamp_offset(*, sites: pl.DataFrame) -> dict[int, float]:
    """Correlate ICON-DREAM-EU's and ERA5's hour-to-hour speed changes, at each offset scanned.

    Uses no power data. Each series' hour-to-hour first difference is more sensitive to a timestamp
    error than the levels themselves, which both products keep inside a narrow physical range
    regardless of any shift.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.

    Returns:
        One Pearson correlation per offset in `OFFSET_SCAN_HOURS`, pooled over every site, of
        ICON-DREAM-EU's speed change against ERA5's shifted by that many hours.
    """
    icon_dream = icon_dream_site_frame(sites=sites).select(
        "site", "time", speed=pl.col(f"speed_hub_{PRODUCT}")
    )
    era5 = pl.read_parquet(output_path_for(product="era5")).select(
        "site", "time", speed=pl.col("wind_speed_100m")
    )
    results: dict[int, float] = {}
    for offset in OFFSET_SCAN_HOURS:
        shifted = era5.with_columns(time=pl.col("time").dt.offset_by(f"{-offset}h"))
        joined_frame = (
            icon_dream.sort("site", "time")
            .join(shifted.sort("site", "time"), on=["site", "time"], suffix="_era5")
            .sort("site", "time")
            .with_columns(
                icon_dream_diff=pl.col("speed").diff().over("site"),
                era5_diff=pl.col("speed_era5").diff().over("site"),
            )
            .drop_nulls(["icon_dream_diff", "era5_diff"])
        )
        results[offset] = float(
            np.corrcoef(
                joined_frame["icon_dream_diff"].to_numpy(), joined_frame["era5_diff"].to_numpy()
            )[0, 1]
        )
    return results


def icon_dream_common_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the pre-registered row set: `common_rows(joined(...))` inner-joined to ICON-DREAM-EU.

    Args:
        sites: The wind roster.

    Returns:
        One row per common site-hour, carrying the five original products' wind, ICON-DREAM-EU's
        wind, the power, the capacity, `constrained`, `cap_mw`, `hour_of_day`, `day_of_year`,
        `month`, `era`, `era_code` and `fold`.
    """
    base = common_rows(frame=joined(sites=sites))
    icon_dream = icon_dream_site_frame(sites=sites)
    frame = base.join(icon_dream, on=["site", "time"], how="inner")
    return with_eras(frame=_add_time_features(dataset=frame))


def jobs() -> list[Job]:
    """Return every arm's job: the five products, ICON-DREAM-EU, its levels arm and speed-only arms.

    Returns:
        One job per arm; the two deciding contrasts' arms are duplicated at
        `SENSITIVITY_HYPER_PARAMETERS`.
    """
    job_list: list[Job] = []
    for product in PRODUCTS:
        columns = (*SHARED_FEATURES, *_wind_columns(product=product))
        job_list.append(
            (f"{product}_wind", "pooled", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)
        )
    for product in ("era5", "icon_eu", PRODUCT):
        columns = (*SHARED_FEATURES, *_wind_columns(product=product))
        job_list.append(
            (
                f"{product}_wind",
                "sensitivity",
                "power_mw",
                columns,
                SENSITIVITY_HYPER_PARAMETERS,
                False,
            )
        )
    levels_columns = (
        *SHARED_FEATURES,
        *_wind_columns(product=PRODUCT),
        *(f"speed_{level}_{PRODUCT}" for level in LEVELS_ARM_EXTRA_LEVELS),
    )
    job_list.append(
        (f"{PRODUCT}_levels", "pooled", "power_mw", levels_columns, PRIMARY_HYPER_PARAMETERS, False)
    )
    for product in PRODUCTS:
        hub_speed = _wind_columns(product=product)[0]
        job_list.append(
            (
                f"{product}_speed_only",
                "pooled",
                "power_mw",
                (*SHARED_FEATURES, hub_speed),
                PRIMARY_HYPER_PARAMETERS,
                False,
            )
        )
    return job_list


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    `--report-only` refuses to reuse a saved `losses.parquet` when this does not match, so a code
    change that moves the row set, a feature's value, a column, a seed, or a hyperparameter setting
    cannot silently mix its fits with a previous run's. Hashing every column of `frame`, not only
    `site` and `time`, is what catches a code change that keeps the same rows and the same column
    names but recomputes a value differently -- a flipped direction convention, a shifted model
    level, or a different fold assignment, for example.

    Args:
        frame: The row set every job is fitted on, including `power_mw` and `fold`.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    row_hashes = ordered.hash_rows(seed=0).to_list()
    payload = repr(
        (
            row_hashes,
            [
                (arm, setting, target, tuple(features), tuple(sorted(hyper_parameters.items())))
                for arm, setting, target, features, hyper_parameters, _ in job_list
            ],
            SEEDS,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _by_year_lines(*, losses: pl.DataFrame, treatment: str, reference: str) -> list[str]:
    """Report one contrast's error, year by year on matched months, and its 2025-to-2026 change.

    Args:
        losses: The pooled setting's losses, holding both arms.
        treatment: The arm named first in the contrast.
        reference: The arm it is compared against.

    Returns:
        Markdown lines: a table of year-by-year differences, then the year-on-year change.
    """
    intervals: list[YearInterval] = bootstrap_difference_by_year(
        losses=losses,
        treatment=treatment,
        references=(reference,),
        metric=METRIC,
        months=ERA5_BY_YEAR_MONTHS,
    )
    lines = [
        f"#### {treatment} − {reference}, year by year, January to August of each year",
        "",
        f"A year of fewer than {MIN_MONTHS_FOR_INTERVAL} months gets no interval.",
        "",
        "| Year | Difference (pp of capacity) | 95% interval | Excludes zero? | Months | Rows |",
        "|---|---|---|---|---|---|",
    ]
    for interval in intervals:
        difference, lower, upper = (
            interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
        )
        if interval["enough_months"]:
            interval_text = f"[{lower:+.3f}, {upper:+.3f}]"
            verdict = (
                "**yes**" if (interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0) else "no"
            )
        else:
            interval_text, verdict = "too few months", "—"
        lines.append(
            f"| {interval['year']} | {difference:+.3f} | {interval_text} | {verdict} "
            f"| {interval['n_months']} | {interval['n_rows']:,} |"
        )
    year0, year1 = ERA5_YEAR_CHANGE_YEARS
    change: YearChangeInterval = bootstrap_year_change(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=METRIC,
        year0=year0,
        year1=year1,
        months=ERA5_BY_YEAR_MONTHS,
    )
    change_diff, change_lower, change_upper = (
        change[key] * PERCENTAGE_POINTS for key in ("change", "lower_95", "upper_95")
    )
    lines += [
        "",
        (
            f"Change in the difference, {year0} to {year1} (positive: {treatment}'s lead over "
            f"{reference} shrank): {change_diff:+.3f} [{change_lower:+.3f}, {change_upper:+.3f}] "
            f"pp, on {change['n_rows_year0']:,} and {change['n_rows_year1']:,} rows."
        ),
    ]
    return lines


def _step_of_hour() -> pl.Expr:
    """Return ICON-DREAM-EU's served lead in hours (a value in `STEP_HOURS`) for each row's `time`.

    Returns:
        `3` where `h % 3 == 0`, otherwise `h % 3`.
    """
    residue = pl.col("time").dt.hour() % 3
    return pl.when(residue == 0).then(pl.lit(3)).otherwise(residue)


def _by_step_lines(*, losses: pl.DataFrame) -> list[str]:
    """Report the two deciding contrasts split by ICON-DREAM-EU's served lead, exploratory.

    Added after the first run, once the padding-hour evidence in this module's docstring's "Served
    lead" section showed ICON-DREAM-EU's hourly value is a short forecast at every hour, never an
    analysis.

    Args:
        losses: The pooled setting's losses, holding every arm.

    Returns:
        Markdown lines: one table per deciding contrast, one row per step.
    """
    lines = ["#### The two deciding contrasts, by ICON-DREAM-EU's served lead (exploratory)", ""]
    for treatment, reference in DECIDING_CONTRASTS:
        lines += [f"`{treatment} − {reference}`", "", *CONTRAST_HEADER]
        lines += [
            _contrast_line(
                losses=losses.filter(_step_of_hour() == step),
                treatment=treatment,
                reference=reference,
                label=f"step {step}",
            )
            for step in STEP_HOURS
        ]
        lines.append("")
    return lines


def _equal_lead_lines(*, losses: pl.DataFrame) -> list[str]:
    """Report the two deciding contrasts on equal-lead hours only, at both settings, exploratory.

    ICON-EU is served at 0 to 2 hours; restricting to `h % 3 != 0` keeps the two hours in three
    where ICON-DREAM-EU (1 to 3 hours) and ICON-EU sit at the same served lead, 1 or 2 hours.

    Args:
        losses: Every arm's losses, both settings.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### The two deciding contrasts on equal-lead hours only (`h % 3 != 0`, exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    equal_lead = losses.filter(pl.col("time").dt.hour() % 3 != 0)
    settings: tuple[tuple[str, str], ...] = (
        ("primary setting", "pooled"),
        ("second setting", "sensitivity"),
    )
    for setting_label, setting in settings:
        subset = equal_lead.filter(pl.col("setting") == setting)
        lines += [
            _contrast_line(
                losses=subset, treatment=treatment, reference=reference, label=setting_label
            )
            for treatment, reference in DECIDING_CONTRASTS
        ]
    return lines


class ChecksResult(TypedDict):
    """Every pre-fit check's raw result, computed once by `run_checks`.

    `_raise_on_failed_checks` validates this and `_checks_lines` renders it, so no check runs
    twice and a reviewer reading the report sees exactly what the gate before `run_all` judged.
    """

    hub_component_speed: dict[str, float]
    surface_component_speed: dict[str, float]
    direction_vs_era5: dict[str, float]
    other_direction_vs_era5: dict[str, float]
    offset_correlations: dict[int, float]
    cell_distances: pl.DataFrame


def _nearest_cells(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return every generator's nearest ICON-DREAM-EU cell and its distance.

    Args:
        sites: The wind roster.

    Returns:
        One row per site with `site`, `cell_id`, `distance_km`.
    """
    ws = _read_level_variable(filename=WS_FILE, value_column="ws_m_s")
    cell_ids = sorted(ws["cell_id"].unique().to_list())
    return icon_dream_cells(sites=sites, cell_ids=cell_ids)


def run_checks(*, sites: pl.DataFrame, frame: pl.DataFrame) -> ChecksResult:
    """Run every pre-fit check once, before any arm is fitted.

    Args:
        sites: The wind roster.
        frame: The common row set (`icon_dream_common_rows`'s result), for
            `other_products_direction_vs_era5`'s same-method baseline.

    Returns:
        Every check's raw result.
    """
    return {
        "hub_component_speed": check_component_speed(level=HUB_LEVEL),
        "surface_component_speed": check_component_speed_10m(),
        "direction_vs_era5": check_direction_against_era5(sites=sites),
        "other_direction_vs_era5": other_products_direction_vs_era5(frame=frame),
        "offset_correlations": check_timestamp_offset(sites=sites),
        "cell_distances": _nearest_cells(sites=sites),
    }


def _raise_on_failed_checks(*, checks: ChecksResult) -> None:
    """Raise if any pre-fit check fails, before any arm is fitted on data the checks distrust.

    Args:
        checks: `run_checks`'s result.

    Raises:
        ValueError: Naming every failed check and by how much it missed its threshold.
    """
    problems: list[str] = []
    offsets = checks["offset_correlations"]
    best_offset = max(offsets, key=offsets.__getitem__)
    if best_offset != 0:
        problems.append(f"the offset scan peaks at {best_offset:+d} h, not 0: {offsets}")
    for label, result in (
        (f"level {HUB_LEVEL}", checks["hub_component_speed"]),
        ("10 m", checks["surface_component_speed"]),
    ):
        diff = result["median_abs_diff_m_s"]
        if diff > MAX_COMPONENT_SPEED_MEDIAN_DIFF_M_S:
            problems.append(
                f"{label} component speed disagrees with served WS by {diff:.4f} m/s median, "
                f"more than {MAX_COMPONENT_SPEED_MEDIAN_DIFF_M_S} m/s"
            )
    direction_all = checks["direction_vs_era5"]["all"]
    if direction_all > MAX_DIRECTION_MEAN_ABS_DIFF_DEG:
        problems.append(
            f"direction disagrees with ERA5 by {direction_all:.1f} degrees mean absolute, more "
            f"than {MAX_DIRECTION_MEAN_ABS_DIFF_DEG}"
        )
    if problems:
        msg = "; ".join(problems)
        raise ValueError(msg)


def _checks_lines(*, checks: ChecksResult) -> list[str]:
    """Render every pre-fit check's result as markdown.

    Args:
        checks: `run_checks`'s result.

    Returns:
        Markdown lines.
    """
    hub = checks["hub_component_speed"]
    surface = checks["surface_component_speed"]
    direction = checks["direction_vs_era5"]
    other_direction = checks["other_direction_vs_era5"]
    offsets = checks["offset_correlations"]
    lines = [
        "#### Checks run before any fit",
        "",
        (
            f"- Component speed vs served WS, level {HUB_LEVEL} "
            f"(~{LEVEL_HEIGHTS_M[HUB_LEVEL]} m): median absolute difference "
            f"{hub['median_abs_diff_m_s']:.4f} m/s, 99th percentile "
            f"{hub['p99_abs_diff_m_s']:.4f} m/s."
        ),
        (
            "- Component speed vs served WS, 10 m: median absolute difference "
            f"{surface['median_abs_diff_m_s']:.4f} m/s, 99th percentile "
            f"{surface['p99_abs_diff_m_s']:.4f} m/s."
        ),
    ]
    lines += [
        (
            f"- Direction vs ERA5 100 m, site {site}: mean absolute angle difference "
            f"{direction[site]:.1f} degrees."
        )
        for site in sorted(direction)
    ]
    lines += [
        (
            f"- Direction vs ERA5 100 m, {product} (same-method baseline, not ICON-DREAM-EU): "
            f"mean absolute angle difference {other_direction[product]:.1f} degrees."
        )
        for product in sorted(other_direction)
    ]
    lines += [
        "",
        "| Offset (h) | Correlation of hour-to-hour speed change with ERA5 |",
        "|---|---|",
    ]
    lines += [f"| {offset:+d} | {offsets[offset]:.3f} |" for offset in OFFSET_SCAN_HOURS]
    return lines


def _cell_distance_lines(*, cell_distances: pl.DataFrame) -> list[str]:
    """Render each generator's distance to the ICON-DREAM-EU cell it reads, as markdown.

    Args:
        cell_distances: `run_checks`'s `cell_distances`, one row per site.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Each generator's ICON-DREAM-EU cell",
        "",
        "| Site | Distance (km) |",
        "|---|---|",
    ]
    lines += [
        f"| {row['site']} | {row['distance_km']:.1f} |"
        for row in cell_distances.sort("site").iter_rows(named=True)
    ]
    return lines


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns, once per arm, as markdown.

    A reviewer checks this against the plan, since an arm can silently lose a column (see this
    module's docstring and the `study` skill).

    Args:
        job_list: Every job `jobs()` returns.

    Returns:
        Markdown lines.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for arm, _, _, columns, _, _ in job_list:
        seen.setdefault(arm, columns)
    lines = ["#### Every arm's feature columns", ""]
    lines += [
        f"- `{arm}`: {', '.join(f'`{column}`' for column in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _report(
    *,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
    sites: pl.DataFrame,
    job_list: list[Job],
    checks: ChecksResult,
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        losses: Every arm's losses, at every setting.
        sites: The wind roster, for the geometry lines.
        job_list: Every job `jobs()` returns, for the feature-column section.
        checks: `run_checks`'s result, for the checks and cell-distance sections.

    Returns:
        The report.
    """
    site_labels = sorted(frame["site"].unique().to_list())
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    lines = [
        (
            f"### Six weather products on {frame.height:,} common site-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        "| Product | Hub height shown | All sites | 95% interval | Speed only |",
        "|---|---|---|---|---|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        speed_only = f"{product}_speed_only"
        interval = bootstrap_absolute(losses=pooled, arm=arm, metric=METRIC)
        lower, upper = (interval[key] * PERCENTAGE_POINTS for key in ("lower_95", "upper_95"))
        lines.append(
            f"| {product} | {HUB_HEIGHT_M[product]} m | {_mae(losses=pooled, arm=arm):.3f} "
            f"| [{lower:.3f}, {upper:.3f}] | {_mae(losses=pooled, arm=speed_only):.3f} |"
        )
    lines.append(f"| {PRODUCT}_levels | | {_mae(losses=pooled, arm=f'{PRODUCT}_levels'):.3f} | | |")
    lines += [
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. The interval is a "
            "95% bound from resampling whole months and a fitting seed."
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        *_cell_distance_lines(cell_distances=checks["cell_distances"]),
        "",
        *_checks_lines(checks=checks),
        "",
        "#### Deciding contrasts, named before the run",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        lines.append(
            _contrast_line(losses=pooled, treatment=treatment, reference=reference, label="all")
        )
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
        "#### The same two contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=sensitivity, treatment=t, reference=r, label="all")
        for t, r in DECIDING_CONTRASTS
    ]
    lines += ["", *_by_step_lines(losses=pooled)]
    lines += ["", *_equal_lead_lines(losses=losses)]
    lines += ["", "#### Exploratory contrasts", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
        for t, r in (
            *EXPLORATORY_PRODUCT_CONTRASTS,
            LEVELS_CONTRAST,
            *LEVELS_VS_OTHERS_CONTRASTS,
            *SPEED_ONLY_CONTRASTS,
        )
    ]
    lines.append("")
    for treatment, reference in DECIDING_CONTRASTS:
        lines += _by_year_lines(losses=pooled, treatment=treatment, reference=reference)
        lines.append("")
    lines += geometry_lines(sites=sites, noun="wind farms")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build the row set, run every check, fit every arm, and write the report.

    Every check in `run_checks` runs before any arm is fitted, in both a fresh run and
    `--report-only`, and `_raise_on_failed_checks` raises rather than letting a bad row set reach
    `run_all`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    arguments = parser.parse_args()

    sites = _wind_sites()
    frame = icon_dream_common_rows(sites=sites)
    _LOG.info(
        "common rows: %d, %s to %s",
        frame.height,
        frame["time"].min(),
        frame["time"].max(),
    )

    checks = run_checks(sites=sites, frame=frame)
    _raise_on_failed_checks(checks=checks)

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
                "feature values or hyperparameter setting than this code now produces; re-run "
                "without --report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
    else:
        refuse_to_overwrite(paths=[path, fingerprint_path, report_path])
        losses = run_all(dataset=frame, jobs=all_jobs)
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)

    report = _report(frame=frame, losses=losses, sites=sites, job_list=all_jobs, checks=checks)
    report_path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
