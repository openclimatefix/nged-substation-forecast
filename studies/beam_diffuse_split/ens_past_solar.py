"""Score ECMWF ENS's shortest-available-lead forecast in the past-solar weather-products study.

One-off throwaway script for the addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, extending
`weather_products.py`'s comparison with ECMWF ENS, the live service's own forecast product.

**This section scores a longer-lead forecast than any other product on this page.** ERA5's
radiation is a forecast 1 to 12 hours ahead, UKV's archive holds the analysis, ICON-D2 and ICON-EU
read 1 to 3 hours ahead, and CAMS is a satellite retrieval with no forecast step. ENS's shortest
available band in the download used here, `T+3`, spans leads 3 to 21 hours from each day's own 00
UTC run; after this section's row set is joined to the rest of the page's hours, it scores leads 5
to 20. A live service reading Dynamical.org's archive gets the 00 UTC run from about 09:00 UTC
(ECMWF itself disseminates steps 0 to 90 by about 06:55 UTC), so the report counts the scored hours
that end at or before each time; for those hours ENS reaches the service after the hour has
ended, so it is past weather delivered late. The planned contrasts do not separate ENS's lead,
its 3-hourly steps, its grid and its model version from one another; the exploratory arms
`era5_3h`, `cams_3h` and `era5_3x3` separate two of them.

**Data.** `data/studies/weather/ENS/beam_diffuse_ens.parquet`
(`data/studies/weather/ENS/README.md`), filtered to `horizon == "T+3"`: seven 3-hour radiation and
temperature steps per generator per run, leads 3, 6, ..., 21, all 51 members. ENS's coverage starts
2024-04-01, well after the main row set's December 2022 start, so this section's row set is shorter
and later than the rest of the page, the same shape of caveat the "four extra Open-Meteo models" and
ICON-DREAM-EU sections carry. This section's row set also ends earlier than ENS's own runs allow: it
is trimmed to `era5_grid.LAST_DATE` (2026-09-10), the date `build_dataset.py` trims every product
to, even though ENS's own runs go on to 2026-09-22 and the join drops none of the rows inside that
window. The free open-data subset of ECMWF's ENS, which Dynamical.org's archive is built from,
carries no direct-beam field (ECMWF's full ENS catalogue does), so this section, like the page's
other global-only products, carries a global-irradiance arm only.

**Upsampling to hourly, reusing the study's own tested machinery.** The `T+3` band holds seven
3-hour steps per run, so each run is rebuilt to 19 hourly values by the clear-sky-index
reconstruction `ens_forecast_horizons.py` picked as the best technique for ENS's radiation
(`COMBINATIONS["solar"]["clear_sky"]` there): `ens_forecast_horizons.Steps`,
`ens_forecast_horizons._clear_sky_arrays`, `studies.resample.clear_sky_index_resample`, and
`studies.resample.interpolate_linear` are reused unchanged; only the code that arranges this file's
own seven fixed leads into a `Steps` object is new, because `ens_forecast_horizons.py`'s own
`band_steps` assumes the wider, day-numbered grid its own extract holds, which this file does not.

**Arms, refit on this section's own shorter row set.** `ens_mean_t3` is an XGBoost model shown the
mean of the 51 members' hourly global irradiance and temperature. `era5_global` and `cams_global`
(the page's ERA5 and CAMS arms) are refit here because the row set is shorter than the published
panel's — the `study` skill's shared-rows rule. Every arm carries the same eight feature columns:
the shared geometry and calendar features, `era_code`, and either the arm's own global irradiance,
or the ensemble's own global irradiance and temperature. `era5_global`'s and `cams_global`'s
temperature column is ERA5's `temp_c`, one of the shared features every arm on this page reads,
not CAMS's own temperature, because the CAMS radiation service publishes none. Every fit uses
`colsample_bytree=1` (XGBoost's default, never overridden here), so no arm wins on column count
alone.

**The two planned contrasts.** The brief for the first run named both contrasts, the `T+3` band,
the member mean and the clear-sky-index reconstruction before the first model was fitted. The brief
is a working file outside the repository, and this docstring was committed after the first fit, so
the repository itself does not record that ordering:

- `ens_mean_t3 − era5_global`: ENS's own mean-of-members forecast against ERA5, the other
  reanalysis-adjacent product a reader might reach for first.
- `ens_mean_t3 − cams_global`: ENS against CAMS, the best product on the main leaderboard.

**Exploratory, labelled so in the report:** the two planned contrasts per generator;
`ens_mean_t3` against `ens_control_t3` (the control member alone, through its own XGBoost model),
which shows what averaging all 51 members over the control member is worth; and three arms added
after the first science review. `era5_3h` and `cams_3h` average ERA5 and CAMS over ENS's seven
3-hour steps and rebuild hourly values with the code that rebuilds ENS's own. `era5_3x3` averages
ERA5 over the 3 by 3 block of 0.25-degree cells around each generator's nearest cell. Each carries
eight feature columns and is scored on the same rows as every other arm.

This section carries no member-by-member arm and no year-by-year panel: the 51-way member fit day-1
the horizon study runs costs 51 times a normal fit, out of proportion to this section's two
contrasts, and ENS's own coverage here is under 2.5 years, too short for the "too few months"
year-by-year rule to add much.

**This is solar only.** A wind addition is a separate, future study, not started in this PR.

Run it with `uv run python studies/beam_diffuse_split/ens_past_solar.py`, after
`weather_products.py` has built its datasets (`build_dataset.py` for `open-meteo` and `cams
--min-cams-reliability 0 --suffix _allhours`). `--report-only` rebuilds `report.md` from the saved
`losses.parquet` alone, still checking the saved fingerprint against what this code would now fit.
`refuse_to_overwrite` means a re-run first moves `losses.parquet`, `losses.fingerprint` and
`report.md` to a `superseded/` subfolder.
"""

import argparse
import hashlib
import logging
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import h3
import h3.api.basic_int as h3_api
import numpy as np
import polars as pl
from blend_products import SOLAR, _solar_frame
from build_dataset import _pv_sites, _read_cams, nearest_era5_cell, read_era5
from ens_forecast_horizons import (
    Steps,
    _clear_sky_arrays,
    _long,
    _prefixed,
    ens_columns,
    reduce_members,
    shared_features,
)
from fetch_ens_forecast_horizons import ENSEMBLE_SIZE
from geo.h3 import compute_h3_grid_weights
from run_experiment import MAX_CONCURRENT_FITS, Job, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.baselines import hourly_clear_sky
from studies.bootstrap import bootstrap_absolute
from studies.charts import report_errors
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS, SENSITIVITY_HYPER_PARAMETERS
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.resample import (
    DEFAULT_DAYLIGHT_FLOOR_W_M2,
    clear_sky_index_resample,
    interpolate_linear,
)
from weather_products import (
    CONTRAST_HEADER,
    METRIC,
    PERCENTAGE_POINTS,
    _contrast_line,
    _mae,
    geometry_lines,
    with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger("ens_past_solar")

T3_PATH: Final[Path] = WEATHER_DATA_DIR / "ENS" / "beam_diffuse_ens.parquet"
"""The download this section reads, filtered to `horizon == "T+3"`."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / "ens_past_solar"
"""Where this section writes `losses.parquet`, `losses.fingerprint`, `report.md`, and a
`superseded/` folder for re-runs."""

ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""ENS's own coverage start (`data/studies/weather/ENS/README.md`), which this section's row set
never reaches before."""

STEP_LEADS: Final[tuple[int, ...]] = (3, 6, 9, 12, 15, 18, 21)
"""The `T+3` band's seven leads, in hours, each a 3-hour mean ending at the lead."""

STEP_WIDTH_HOURS: Final[int] = 3
"""Every `T+3` step's width."""

TARGET_LEADS: Final[np.ndarray] = np.arange(STEP_LEADS[0], STEP_LEADS[-1] + 1, dtype=np.float64)
"""Every hour this section scores, labelled by its end: leads 3 to 21, 19 hourly values."""

MEAN_ARM: Final[str] = "ens_mean_t3"
"""The planned arm: an XGBoost model shown the mean of the 51 members' hourly fields."""

CONTROL_ARM: Final[str] = "ens_control_t3"
"""The exploratory arm: an XGBoost model shown the control member's own hourly fields."""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (MEAN_ARM, "era5_global"),
    (MEAN_ARM, "cams_global"),
)
"""The two planned contrasts: ENS against ERA5, and ENS against CAMS."""

ERA5_3H_ARM: Final[str] = "era5_3h"
CAMS_3H_ARM: Final[str] = "cams_3h"
ERA5_3X3_ARM: Final[str] = "era5_3x3"
EXPLORATORY_ARMS: Final[tuple[str, ...]] = (ERA5_3H_ARM, CAMS_3H_ARM, ERA5_3X3_ARM)
"""Arms added after the first review, so exploratory: `era5_3h` and `cams_3h` are ERA5 and CAMS
averaged over ENS's 3-hour steps and rebuilt to hourly values by the same code as ENS's own;
`era5_3x3` is ERA5 averaged over the 3 by 3 block of 0.25-degree cells around each generator's
nearest cell. Each carries the same eight feature columns as every other arm."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (MEAN_ARM, CONTROL_ARM),
    (CONTROL_ARM, "era5_global"),
)
"""What averaging all 51 members over the control member alone is worth, and how the control member
alone compares with ERA5 (`docs/studies/ens-forecast-horizons.md` reports the same comparison at
its day 0, on its own rows)."""

CONFOUND_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (ERA5_3H_ARM, "era5_global"),
    (CAMS_3H_ARM, "cams_global"),
    (ERA5_3X3_ARM, "era5_global"),
    (MEAN_ARM, ERA5_3H_ARM),
    (MEAN_ARM, CAMS_3H_ARM),
    (MEAN_ARM, ERA5_3X3_ARM),
)
"""What the 3-hourly step and the wider area do to ERA5 and CAMS, and what is left of ENS's gap."""

ERA5_3H_COLUMN: Final[str] = "ghi_era5_3h"
CAMS_3H_COLUMN: Final[str] = "ghi_cams_3h"
ERA5_3X3_COLUMN: Final[str] = "ghi_era5_3x3"
EXPLORATORY_COLUMNS: Final[dict[str, str]] = {
    ERA5_3H_ARM: ERA5_3H_COLUMN,
    CAMS_3H_ARM: CAMS_3H_COLUMN,
    ERA5_3X3_ARM: ERA5_3X3_COLUMN,
}
"""Each exploratory arm's own irradiance column."""

CONFOUND_HEADING: Final[str] = (
    "What a 3-hourly step and a wider ERA5 area do to the gap (exploratory)"
)
"""The report heading above `CONFOUND_CONTRASTS`, which the chart script reads."""

FIRST_SERVABLE_HOUR_UTC: Final[int] = 9
"""The hour of day, UTC, from which a live service reading Dynamical.org's archive can read the
00 UTC run: about 09:00 UTC (`docs/studies/ens-forecast-horizons.md`, "Horizons and issue time")."""

FIRST_DISSEMINATED_HOUR_UTC: Final[int] = 6
"""The last hour label, UTC, whose hour ends before ECMWF disseminates the 00 UTC run's steps 0 to
90, at about 06:55 UTC ([ECMWF's dissemination
schedule](https://confluence.ecmwf.int/display/DAC/Dissemination+schedule)): an hour labelled 06
ends at 06:00, and an hour labelled 07 ends after 06:55."""

ERA5_RUN_INTERVAL_HOURS: Final[int] = 12
"""ERA5's radiation is a forecast at steps of 1 to 12 hours from the 06 and 18 UTC runs, so
its lead at an hour labelled `h` (the hour ending at `h` UTC) is `((h - 7) % 12) + 1`: lead 1
at 07 and 19 UTC, lead 12 at 18 and 06 UTC."""

H3_RESOLUTION: Final[int] = 5
"""The H3 resolution ENS's values are averaged over at each generator."""

ERA5_CELL_DEGREES: Final[float] = 0.25
"""ERA5's grid spacing, in degrees."""

KM_PER_DEGREE_LATITUDE: Final[float] = 111.19
"""One degree of latitude, in kilometres, on a sphere of mean Earth radius."""

SERVABLE_HEADING: Final[str] = (
    "The planned contrasts, split by when the 00 UTC run becomes readable (exploratory, post hoc)"
)
"""The report heading above the split by when the run becomes readable."""

HORIZONS_RISE_POINTS: Final[float] = 0.61
"""The ENS horizons study's rise in solar error, in points of capacity, from day 0 to day 1
(`docs/studies/ens-forecast-horizons.md`: 0.61 points [0.45, 0.79]), which is 24 hours of extra
lead. Scaling it to this section's extra hours gives a rough guide with no interval, on that
page's own rows."""

HOURS_PER_DAY: Final[int] = 24

ERA5_FIRST_RUN_HOUR_UTC: Final[int] = 6
"""The hour of day, UTC, of the earlier of ERA5's two daily forecast runs (the other is 12 hours
later)."""


def _t3_members() -> pl.DataFrame:
    """Read the download and keep the `T+3` band only.

    Returns:
        One row per generator, run, valid time and member, at leads 3 to 21.
    """
    return pl.read_parquet(T3_PATH).filter(pl.col("horizon") == "T+3")


def _t3_steps(*, members: pl.DataFrame) -> Steps:
    """Arrange the `T+3` band's members as arrays over its seven fixed steps.

    Mirrors `ens_forecast_horizons.band_steps`'s pivot-and-complete-ensemble logic, but over this
    file's own fixed leads (`STEP_LEADS`) rather than a day-numbered window, because
    `beam_diffuse_ens.parquet` holds the `T+3` band alone, not the continuous grid `band_steps`
    assumes. A (site, run) is dropped whole where any member lacks any of the seven leads, so every
    run kept holds all 51 members, in member order.

    Args:
        members: `_t3_members`'s output.

    Returns:
        The arrays.

    Raises:
        ValueError: If a kept run does not hold every member.
    """
    columns = ("ghi_w_m2", "temp_c")
    wide = members.pivot(
        on="lead_hours",
        index=["site", "init_time", "ensemble_member"],
        values=list(columns),
        sort_columns=True,
    )
    names = {column: [f"{column}_{lead}" for lead in STEP_LEADS] for column in columns}
    complete = wide.drop_nulls([name for group in names.values() for name in group])
    runs = (
        complete.group_by("site", "init_time")
        .len()
        .filter(pl.col("len") == ENSEMBLE_SIZE)
        .select("site", "init_time")
    )
    kept = complete.join(runs, on=["site", "init_time"]).sort(
        "site", "init_time", "ensemble_member"
    )
    expected_members = np.tile(np.arange(ENSEMBLE_SIZE), kept.height // ENSEMBLE_SIZE)
    if not np.array_equal(kept["ensemble_member"].to_numpy(), expected_members):
        msg = f"a kept run does not hold members 0 to {ENSEMBLE_SIZE - 1} in order"
        raise ValueError(msg)
    values = {column: kept.select(names[column]).to_numpy() for column in columns}
    return Steps(
        keys=kept.select("site", "init_time", "ensemble_member"),
        leads=np.array(STEP_LEADS, dtype=np.float64),
        widths=np.full(len(STEP_LEADS), STEP_WIDTH_HOURS, dtype=np.float64),
        values=values,
    )


def _t3_upsampled(*, steps: Steps, clear_sky: pl.DataFrame) -> dict[str, np.ndarray]:
    """Upsample every member's `T+3` steps to hourly: `clear_sky` for radiation, `linear` for temp.

    The technique `ens_forecast_horizons.py`'s pre-registered choosing rule picked as best for
    solar's radiation (`COMBINATIONS["solar"]["clear_sky"]`).

    Args:
        steps: The `T+3` band's steps.
        clear_sky: `hourly_clear_sky`'s table, wide enough for every run's leads 1 to 21.

    Returns:
        `ghi` and `temp`, each shape (n_series, 19), matching `TARGET_LEADS`.
    """
    x = steps.leads
    step_clear_sky, target_clear_sky = _clear_sky_arrays(
        steps=steps, targets=TARGET_LEADS, clear_sky=clear_sky
    )
    midpoints = x - steps.widths / 2.0
    morning = np.mod(midpoints, 24.0) < 12.0
    ghi = clear_sky_index_resample(
        values=steps.values["ghi_w_m2"],
        step_clear_sky=step_clear_sky,
        step_midpoints=midpoints,
        morning=morning,
        target_clear_sky=target_clear_sky,
        target_midpoints=TARGET_LEADS - 0.5,
        daylight_floor_w_m2=DEFAULT_DAYLIGHT_FLOOR_W_M2,
    )
    temp = interpolate_linear(values=steps.values["temp_c"], x=x, targets=TARGET_LEADS - 0.5)
    return {"ghi": ghi, "temp": temp}


def _era5_and_cams_hourly() -> dict[str, pl.DataFrame]:
    """Read the hourly global irradiance ERA5 and CAMS give each generator.

    Both are read the way `build_dataset.py` reads them for the page's main arms: ERA5 at each
    generator's nearest cell, and CAMS with no reliability filter.

    Returns:
        `era5` and `cams`, each with `site`, `time` and `ghi_w_m2`.
    """
    era5 = read_era5(source="open-meteo")
    cells = nearest_era5_cell(sites=_pv_sites(), era5=era5).select(
        "site", latitude="cell_latitude", longitude="cell_longitude"
    )
    return {
        "era5": cells.join(era5, on=["latitude", "longitude"]).select("site", "time", "ghi_w_m2"),
        "cams": _read_cams(min_reliability=0.0).select("site", "time", "ghi_w_m2"),
    }


def _three_hourly_rebuilt(
    *, hourly: pl.DataFrame, runs: pl.DataFrame, clear_sky: pl.DataFrame, column: str
) -> pl.DataFrame:
    """Give a product ENS's 3-hourly treatment: average it over ENS's steps, then rebuild hourly.

    For each ENS run, the product's hourly values are averaged over the seven 3-hour steps ENS
    publishes, and the seven steps are turned back into 19 hourly values by the same
    clear-sky-index code `_t3_upsampled` applies to ENS. Where every daylight hour of a step is
    present, the step mean is the plain mean; where a daylight hour is missing, the step mean is
    the mean clear-sky index of the hours present times the step's mean clear-sky irradiance.
    A run with an unusable step is dropped whole.

    Args:
        hourly: The product's `site`, `time` and `ghi_w_m2`, the mean over the hour ending at
            `time`.
        runs: One row per (site, run), with `site` and `init_time`, sorted.
        clear_sky: `hourly_clear_sky`'s table, wide enough for every run's leads 1 to 21.
        column: The name to give the rebuilt hourly irradiance.

    Returns:
        `site`, `time` and `column`, one row per hour of each kept run.
    """
    grid = (
        runs.join(pl.DataFrame({"lead": list(range(1, STEP_LEADS[-1] + 1))}), how="cross")
        .with_columns(time=pl.col("init_time") + pl.duration(hours=pl.col("lead")))
        .join(clear_sky, on=["site", "time"], how="left")
        .join(hourly, on=["site", "time"], how="left")
        .with_columns(
            ghi=pl.when(pl.col("clear_sky_w_m2") <= 0.0).then(0.0).otherwise(pl.col("ghi_w_m2"))
        )
        .with_columns(step=((pl.col("lead") + STEP_WIDTH_HOURS - 1) // STEP_WIDTH_HOURS) * 3)
    )
    daylight = pl.col("clear_sky_w_m2") > 0
    step_means = grid.group_by("site", "init_time", "step").agg(
        clear_sky_index=(pl.col("ghi") / pl.col("clear_sky_w_m2")).filter(daylight).mean(),
        missing=(pl.col("ghi").is_null() & daylight).sum(),
        clear_sky=pl.col("clear_sky_w_m2").mean(),
        mean=pl.col("ghi").mean(),
    )
    step_means = step_means.with_columns(
        value=pl.when(pl.col("missing") == 0)
        .then(pl.col("mean"))
        .otherwise(pl.col("clear_sky_index") * pl.col("clear_sky"))
    )
    names = [str(lead) for lead in STEP_LEADS]
    wide = (
        step_means.pivot(on="step", index=["site", "init_time"], values="value", sort_columns=True)
        .select("site", "init_time", *names)
        .sort("site", "init_time")
    )
    unusable = pl.any_horizontal(pl.col(names).is_null() | pl.col(names).is_nan())
    wide = wide.filter(~unusable)
    keys = wide.select("site", "init_time", ensemble_member=pl.lit(0).cast(pl.Int8))
    steps = Steps(
        keys=keys,
        leads=np.array(STEP_LEADS, dtype=np.float64),
        widths=np.full(len(STEP_LEADS), STEP_WIDTH_HOURS, dtype=np.float64),
        values={"ghi_w_m2": wide.select(names).to_numpy().astype(np.float64)},
    )
    step_clear_sky, target_clear_sky = _clear_sky_arrays(
        steps=steps, targets=TARGET_LEADS, clear_sky=clear_sky
    )
    # `_clear_sky_arrays` repeats each run once per ENS member; this frame has one series per run.
    step_clear_sky, target_clear_sky = (
        step_clear_sky[::ENSEMBLE_SIZE],
        target_clear_sky[::ENSEMBLE_SIZE],
    )
    midpoints = steps.leads - steps.widths / 2.0
    rebuilt = clear_sky_index_resample(
        values=steps.values["ghi_w_m2"],
        step_clear_sky=step_clear_sky,
        step_midpoints=midpoints,
        morning=np.mod(midpoints, 24.0) < 12.0,
        target_clear_sky=target_clear_sky,
        target_midpoints=TARGET_LEADS - 0.5,
        daylight_floor_w_m2=DEFAULT_DAYLIGHT_FLOOR_W_M2,
    )
    return _long(steps=steps, targets=TARGET_LEADS, values={column: rebuilt}).select(
        "site", "time", column
    )


def _era5_three_by_three() -> pl.DataFrame:
    """Average ERA5's global irradiance over the 3 by 3 block of cells around each nearest cell.

    Returns:
        `site`, `time` and `ghi_era5_3x3`.

    Raises:
        ValueError: If a generator's block leaves the grid or holds fewer than nine cells.
    """
    era5 = read_era5(source="open-meteo")
    latitudes = np.sort(era5["latitude"].unique().to_numpy())
    longitudes = np.sort(era5["longitude"].unique().to_numpy())
    cells = nearest_era5_cell(sites=_pv_sites(), era5=era5)
    blocks = []
    for site, latitude, longitude in cells.select(
        "site", "cell_latitude", "cell_longitude"
    ).iter_rows():
        i = int(np.argmin(np.abs(latitudes - latitude)))
        j = int(np.argmin(np.abs(longitudes - longitude)))
        if not (1 <= i < len(latitudes) - 1 and 1 <= j < len(longitudes) - 1):
            msg = f"site {site}: the 3 by 3 block leaves the ERA5 grid"
            raise ValueError(msg)
        block = era5.filter(
            pl.col("latitude").is_in(latitudes[i - 1 : i + 2].tolist()),
            pl.col("longitude").is_in(longitudes[j - 1 : j + 2].tolist()),
        )
        blocks.append(
            block.group_by("time")
            .agg(pl.col("ghi_w_m2").mean().alias(ERA5_3X3_COLUMN), cells=pl.len())
            .with_columns(site=pl.lit(site))
        )
    result = pl.concat(blocks)
    if result["cells"].min() != 9 or result["cells"].max() != 9:
        msg = "an ERA5 3 by 3 block holds fewer than nine cells"
        raise ValueError(msg)
    return result.select("site", "time", ERA5_3X3_COLUMN)


def build_rows() -> pl.DataFrame:
    """Build this section's row set.

    `weather_products.py`'s common rows, ENS's own window, and every hour the `T+3` band supplies.

    Returns:
        One row per (site, time), with every arm's feature columns, `power_mw`, `cap_mw`,
        `constrained`, `era`, `era_code` and `fold` recomputed on this restricted row set.

    Raises:
        ValueError: If a (site, time) is duplicated, or an ENS column holds a missing value.
    """
    sites = _pv_sites()
    base = _solar_frame().filter(pl.col("time") >= ENS_START)
    used_sites = sorted(base["site"].unique().to_list())
    members = _t3_members().filter(pl.col("site").is_in(used_sites))
    steps = _t3_steps(members=members)
    clear_sky = hourly_clear_sky(
        sites=sites.filter(pl.col("site").is_in(used_sites)),
        first=cast("datetime", steps.keys["init_time"].min()) + timedelta(hours=1),
        last=cast("datetime", steps.keys["init_time"].max()) + timedelta(hours=STEP_LEADS[-1]),
    )
    upsampled = _t3_upsampled(steps=steps, clear_sky=clear_sky)
    hourly = _long(steps=steps, targets=TARGET_LEADS, values=upsampled)
    mean_frame = _prefixed(
        frame=reduce_members(hourly=hourly, domain="solar", way="mean"),
        arm=MEAN_ARM,
        domain="solar",
    )
    control_frame = _prefixed(
        frame=reduce_members(hourly=hourly, domain="solar", way="control"),
        arm=CONTROL_ARM,
        domain="solar",
    )
    runs = steps.keys.select("site", "init_time").unique(maintain_order=True)
    hourly_products = _era5_and_cams_hourly()
    rebuilt_era5 = _three_hourly_rebuilt(
        hourly=hourly_products["era5"], runs=runs, clear_sky=clear_sky, column=ERA5_3H_COLUMN
    )
    rebuilt_cams = _three_hourly_rebuilt(
        hourly=hourly_products["cams"], runs=runs, clear_sky=clear_sky, column=CAMS_3H_COLUMN
    )
    joined = (
        base.join(mean_frame, on=["site", "time"], how="inner")
        .join(control_frame, on=["site", "time"], how="inner")
        .join(rebuilt_era5, on=["site", "time"], how="inner")
        .join(rebuilt_cams, on=["site", "time"], how="inner")
        .join(_era5_three_by_three(), on=["site", "time"], how="inner")
        .drop("era", "era_code", "fold")
        .sort("site", "time")
    )
    if joined.select("site", "time").is_duplicated().any():
        msg = "build_rows: a (site, time) is duplicated"
        raise ValueError(msg)
    ens_columns_all = (
        *ens_columns(arm=MEAN_ARM, domain="solar"),
        *ens_columns(arm=CONTROL_ARM, domain="solar"),
    )
    check_no_missing(frame=joined, columns=(*ens_columns_all, *EXPLORATORY_COLUMNS.values()))
    _LOG.info("%d rows in this section's own row set", joined.height)
    return with_eras(frame=joined)


def jobs() -> list[Job]:
    """Return every arm's job at `pooled`, plus the two planned contrasts' arms at `sensitivity`.

    The `pooled` jobs are the two ENS arms, and ERA5 and CAMS refit on this row set. The
    `sensitivity` jobs refit `ens_mean_t3`, `era5_global` and `cams_global` — the arms in
    `DECIDING_CONTRASTS` — at `SENSITIVITY_HYPER_PARAMETERS`, the `study` skill's rule that every
    deciding contrast gets a second hyperparameter setting.

    Returns:
        One job per arm at `pooled`, plus one job per planned-contrast arm at `sensitivity`, every
        arm shown eight feature columns.

    Raises:
        ValueError: Unless every arm's feature-column count matches.
    """
    ens_features = (*shared_features(domain=SOLAR), *ens_columns(arm=MEAN_ARM, domain="solar"))
    control_features = (
        *shared_features(domain=SOLAR),
        *ens_columns(arm=CONTROL_ARM, domain="solar"),
    )
    era5_features = (*SOLAR.shared_features, *SOLAR.columns("era5"))
    cams_features = (*SOLAR.shared_features, *SOLAR.columns("cams"))
    plain_features = {
        "era5_global": era5_features,
        "cams_global": cams_features,
        **{arm: (*SOLAR.shared_features, column) for arm, column in EXPLORATORY_COLUMNS.items()},
    }
    job_list: list[Job] = [
        (MEAN_ARM, "pooled", "power_mw", ens_features, PRIMARY_HYPER_PARAMETERS, False),
        (CONTROL_ARM, "pooled", "power_mw", control_features, PRIMARY_HYPER_PARAMETERS, False),
        *(
            (arm, "pooled", "power_mw", features, PRIMARY_HYPER_PARAMETERS, False)
            for arm, features in plain_features.items()
        ),
        (MEAN_ARM, "sensitivity", "power_mw", ens_features, SENSITIVITY_HYPER_PARAMETERS, False),
        *(
            (arm, "sensitivity", "power_mw", features, SENSITIVITY_HYPER_PARAMETERS, False)
            for arm, features in plain_features.items()
        ),
    ]
    counts = {len(features) for _, _, _, features, _, _ in job_list}
    if len(counts) != 1:
        msg = f"every arm should carry the same number of feature columns, found counts {counts}"
        raise ValueError(msg)
    return job_list


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    `--report-only` refuses to reuse a saved `losses.parquet` when this does not match, so a code
    change to the row set, a feature, a column, a seed, or a hyperparameter setting cannot silently
    mix its fits with a previous run's. `reduce_members`'s `group_by().agg(mean())` sums the 51
    members in a parallel, non-fixed order, so two builds of the same row set differ by up to
    4.5e-13 in `ens_mean_t3_ghi` and `ens_mean_t3_temp` — real, but far below any meaningful
    precision. Every float column is cast to `Float32` before hashing so that noise cannot flip the
    fingerprint; the saved `losses.parquet` itself keeps full precision, because only the
    fingerprint is affected.

    Args:
        frame: The row set every job is fitted on.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    float_columns = [name for name, dtype in ordered.schema.items() if dtype.is_float()]
    stable = ordered.cast(dict.fromkeys(float_columns, pl.Float32))
    row_hashes = stable.hash_rows(seed=0).to_list()
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


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns, once per arm, as markdown.

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
        f"- `{arm}` ({len(columns)} columns): {', '.join(f'`{c}`' for c in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _absolute_table_lines(*, pooled: pl.DataFrame, arms: tuple[str, ...]) -> list[str]:
    """Render each arm's mean absolute error and 95% interval as a markdown table.

    Args:
        pooled: Every arm's losses at the `pooled` setting.
        arms: The arms, one row each.

    Returns:
        Markdown lines.
    """
    lines = ["| Arm | All sites | 95% interval |", "|---|---|---|"]
    for arm in arms:
        interval = bootstrap_absolute(losses=pooled, arm=arm, metric=METRIC)
        lower, upper = (interval[key] * PERCENTAGE_POINTS for key in ("lower_95", "upper_95"))
        lines.append(f"| {arm} | {_mae(losses=pooled, arm=arm):.3f} | [{lower:.3f}, {upper:.3f}] |")
    return lines


def _servable_lines(*, pooled: pl.DataFrame) -> list[str]:
    """Render the two planned contrasts on hours split by when the 00 UTC run becomes readable.

    This is a post hoc, exploratory split. Every consumer on this page reads a value after its hour
    has passed, so for an hour ending at or before `FIRST_SERVABLE_HOUR_UTC` ENS is readable 0 to 4
    hours after the hour ends: past weather, delivered late, not a forecast. Each contrast is paired
    within the subset by the same month-and-seed resampling as every other interval.

    Args:
        pooled: Every arm's losses at the `pooled` setting.

    Returns:
        Markdown lines.
    """
    hour = pl.col("time").dt.hour()
    subsets = (
        (f"ends after {FIRST_SERVABLE_HOUR_UTC:02d}:00 UTC", hour > FIRST_SERVABLE_HOUR_UTC),
        (
            f"ends at or before {FIRST_SERVABLE_HOUR_UTC:02d}:00 UTC",
            hour <= FIRST_SERVABLE_HOUR_UTC,
        ),
    )
    lines = [
        f"#### {SERVABLE_HEADING}",
        "",
        *CONTRAST_HEADER,
    ]
    gains = []
    for name, condition in subsets:
        subset = pooled.filter(condition)
        for treatment, reference in DECIDING_CONTRASTS:
            lines.append(
                _contrast_line(losses=subset, treatment=treatment, reference=reference, label=name)
            )
        era5_error = _mae(losses=subset, arm="era5_global")
        gain = era5_error - _mae(losses=subset, arm=MEAN_ARM)
        gains.append(
            f"- Hours that end {name.removeprefix('ends ')}: ENS's gain over ERA5 is {gain:.3f} "
            f"points, {gain / era5_error:.1%} of ERA5's error of {era5_error:.3f}."
        )
    return [*lines, "", *gains]


def _lead_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render the scored hours' leads, and how many hours end before the 00 UTC run is readable.

    ENS's lead at an hour is the hour of day, because every run starts at 00 UTC. ERA5's radiation
    lead follows `ERA5_RUN_INTERVAL_HOURS`. A live service reads the 00 UTC run from about
    `FIRST_SERVABLE_HOUR_UTC`, so for an hour labelled at or before then (an hour that ends at or
    before the run is readable) ENS is past weather delivered up to 4 hours late, not a forecast.

    Args:
        frame: This section's row set.

    Returns:
        Markdown lines.
    """
    hour = frame["time"].dt.hour().to_numpy().astype(np.float64)
    ens_lead = hour
    era5_lead = (hour - ERA5_FIRST_RUN_HOUR_UTC - 1) % ERA5_RUN_INTERVAL_HOURS + 1
    too_early = int((hour <= FIRST_SERVABLE_HOUR_UTC).sum())
    before_dissemination = int((hour <= FIRST_DISSEMINATED_HOUR_UTC).sum())
    extra = float(ens_lead.mean() - era5_lead.mean())
    return [
        "#### Leads of the scored hours",
        "",
        (
            f"- ENS's lead is {ens_lead.min():.0f} to {ens_lead.max():.0f} hours, "
            f"mean {ens_lead.mean():.2f}."
        ),
        (
            f"- ERA5's radiation lead is {era5_lead.min():.0f} to {era5_lead.max():.0f} hours, "
            f"mean {era5_lead.mean():.2f}."
        ),
        f"- ENS's mean lead exceeds ERA5's by {ens_lead.mean() - era5_lead.mean():.2f} hours.",
        (
            f"- A live service reads the 00 UTC run from about {FIRST_SERVABLE_HOUR_UTC:02d}:00 "
            f"UTC, so {too_early:,} of {frame.height:,} scored hours "
            f"({too_early / frame.height:.1%}) end at or before that time."
        ),
        (
            "- For those hours the run becomes readable 0 to "
            f"{int(FIRST_SERVABLE_HOUR_UTC - hour[hour <= FIRST_SERVABLE_HOUR_UTC].min())} hours "
            "after the hour ends."
        ),
        (
            f"- Scaling the ENS horizons study's rise of {HORIZONS_RISE_POINTS:.2f} points over "
            f"{HOURS_PER_DAY} hours of extra lead to {extra:.2f} hours gives about "
            f"{HORIZONS_RISE_POINTS * extra / HOURS_PER_DAY:.2f} points, with no interval."
        ),
        (
            "- ECMWF itself disseminates the run's steps 0 to 90 by about 06:55 UTC, so "
            f"{before_dissemination:,} of {frame.height:,} scored hours "
            f"({before_dissemination / frame.height:.1%}) end at or before that time."
        ),
    ]


def _support_lines(*, sites: pl.DataFrame) -> list[str]:
    """Render the area each product's value at a generator averages over.

    Both products are served on the same 0.25-degree grid. ENS's value at a generator is the
    area-weighted mean of the grid points whose boxes overlap the generator's H3 resolution-5 cell
    (`geo.h3.compute_h3_grid_weights`, as `delta_store`'s ENS table stores it). ERA5's value is
    the one grid cell nearest the generator, and `era5_3x3` averages the nine cells around it. One
    grid cell is 0.25 degrees of latitude tall and 0.25 degrees of longitude times the cosine of the
    latitude wide, at the generators' mean latitude.

    Args:
        sites: The solar roster, with `latitude` and `longitude`.

    Returns:
        Markdown lines.
    """
    latitude = cast("float", sites["latitude"].mean())
    cell_km2 = h3.average_hexagon_area(H3_RESOLUTION, unit="km^2")
    era5_cell_km2 = (
        ERA5_CELL_DEGREES
        * KM_PER_DEGREE_LATITUDE
        * ERA5_CELL_DEGREES
        * KM_PER_DEGREE_LATITUDE
        * float(np.cos(np.radians(latitude)))
    )
    block_km2 = 9 * era5_cell_km2
    h3_cells = sorted(
        {
            h3_api.latlng_to_cell(lat, lng, H3_RESOLUTION)
            for lat, lng in zip(
                sites["latitude"].to_list(), sites["longitude"].to_list(), strict=True
            )
        }
    )
    weights = compute_h3_grid_weights(nwp_grid_size_degrees=ERA5_CELL_DEGREES, h3_index=h3_cells)
    per_cell = weights.group_by("h3_index").len()["len"]
    ens_points = set(zip(weights["nwp_lat"].to_list(), weights["nwp_lon"].to_list(), strict=True))
    era5_cells = nearest_era5_cell(sites=sites, era5=read_era5(source="open-meteo"))
    era5_points = set(
        zip(
            era5_cells["cell_latitude"].to_list(),
            era5_cells["cell_longitude"].to_list(),
            strict=True,
        )
    )
    return [
        "#### Spatial support",
        "",
        (
            f"- One 0.25° grid cell, the grid both products are served on, covers "
            f"{era5_cell_km2:,.0f} km² at latitude {latitude:.1f}°."
        ),
        (
            f"- ENS's H3 resolution-{H3_RESOLUTION} cell has a mean area of {cell_km2:,.0f} km², "
            "and ENS's value is the mean of the grid cells it overlaps, weighted by overlap."
        ),
        (
            f"- The {len(h3_cells)} H3 cells the generators sit in each overlap {per_cell.min()} "
            f"to {per_cell.max()} grid cells, {len(ens_points)} distinct grid cells in all."
        ),
        (
            f"- ERA5's nearest cells number {len(era5_points)}, and "
            f"{len(era5_points & ens_points)} of them are among ENS's grid cells."
        ),
        (
            f"- The 3 by 3 ERA5 block is 9 grid cells and covers {block_km2:,.0f} km² "
            f"at latitude {latitude:.1f}°."
        ),
        f"- The block is {block_km2 / cell_km2:.1f} times the area of an H3 cell.",
    ]


def _generator_lines(*, frame: pl.DataFrame, members: pl.DataFrame) -> list[str]:
    """Render each generator's rows and months, and which generators share one ENS input.

    Two generators in the same H3 resolution-5 cell read the same ENS values, so the six
    generators carry fewer distinct ENS inputs than six.

    Args:
        frame: This section's row set.
        members: `_t3_members`'s output.

    Returns:
        Markdown lines.
    """
    per_site = (
        frame.group_by("site")
        .agg(rows=pl.len(), months=pl.col("time").dt.strftime("%Y-%m").n_unique())
        .sort("site")
    )
    control = (
        members.filter(pl.col("ensemble_member") == 0)
        .sort("init_time", "lead_hours")
        .group_by("site", maintain_order=True)
        .agg(pl.col("ghi_w_m2"))
    )
    inputs = dict(zip(control["site"].to_list(), control["ghi_w_m2"].to_list(), strict=True))
    groups: dict[tuple[float, ...], list[str]] = {}
    for site in sorted(inputs):
        groups.setdefault(tuple(inputs[site]), []).append(site)
    lines = [
        "#### Rows, months, and ENS inputs per generator",
        "",
        "| Site | Rows | Months |",
        "|---|---|---|",
    ]
    lines += [f"| {site} | {rows:,} | {months} |" for site, rows, months in per_site.iter_rows()]
    months = frame["time"].dt.strftime("%Y-%m").n_unique()
    lines += ["", f"The row set spans {months} calendar months.", ""]
    lines += [f"The six generators carry {len(groups)} distinct ENS inputs:", ""]
    lines += [f"- {', '.join(sites)}" for sites in groups.values()]
    return lines


def _main_panel_lines(*, pooled: pl.DataFrame) -> list[str]:
    """Render how far ERA5 and CAMS refit here differ from the page's main row set.

    Args:
        pooled: Every arm's losses at the `pooled` setting.

    Returns:
        Markdown lines.
    """
    path = OUTPUT_DIR.parent / "solar_long" / "report.md"
    heading = path.read_text().splitlines()[0]
    match = re.search(r"on ([\d,]+) common site-hours \(([\d-]+) to ([\d-]+)\)", heading)
    if match is None:
        msg = f"{path}: cannot read the row count from {heading!r}"
        raise ValueError(msg)
    main = report_errors(report_path=path, column="Global only")
    first, last = (datetime.strptime(match[n], "%Y-%m-%d") for n in (2, 3))  # noqa: DTZ007
    main_months = (last.year - first.year) * 12 + last.month - first.month + 1
    lines = [
        "#### Against the page's main row set",
        "",
        (
            f"The main row set holds {match[1]} common site-hours ({match[2]} to {match[3]}), "
            f"spanning {main_months} calendar months; this section's row set is shorter."
        ),
        "",
        "| Arm | This section | Main row set | Difference |",
        "|---|---|---|---|",
    ]
    for arm, name in (("era5_global", "era5"), ("cams_global", "cams")):
        here = _mae(losses=pooled, arm=arm)
        lines.append(f"| {arm} | {here:.3f} | {main[name]:.3f} | {here - main[name]:+.3f} |")
    return lines


def _report(
    *, frame: pl.DataFrame, losses: pl.DataFrame, sites: pl.DataFrame, job_list: list[Job]
) -> str:
    """Assemble the markdown report.

    Args:
        frame: This section's own row set.
        losses: Every arm's losses, at both `pooled` and `sensitivity` settings.
        sites: The solar roster, for the geometry lines.
        job_list: Every job `jobs()` returns, for the feature-column section.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    site_labels = sorted(frame["site"].unique().to_list())
    lines = [
        (
            f"### ECMWF ENS's `T+3` band on {frame.height:,} common site-hours of solar "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        *_absolute_table_lines(
            pooled=pooled, arms=(MEAN_ARM, CONTROL_ARM, "era5_global", "cams_global")
        ),
    ]
    lines += [
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. The interval is a 95% "
            "bound from resampling whole months and a fitting seed."
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Planned contrasts",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        lines.append(
            _contrast_line(losses=pooled, treatment=treatment, reference=reference, label="all")
        )
    lines += [
        "",
        "#### Planned contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        lines.append(
            _contrast_line(
                losses=sensitivity, treatment=treatment, reference=reference, label="sensitivity"
            )
        )
    lines += [
        "",
        "#### The same two contrasts, per generator (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
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
        "#### The control member alone, against the members' mean and against ERA5 (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    lines += [
        "",
        "#### Exploratory arms added after the first science review",
        "",
        *_absolute_table_lines(pooled=pooled, arms=EXPLORATORY_ARMS),
        "",
        f"#### {CONFOUND_HEADING}",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
            for t, r in CONFOUND_CONTRASTS
        ),
        "",
        "#### The same contrasts at the second hyperparameter setting (exploratory)",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=sensitivity, treatment=t, reference=r, label="sensitivity")
            for t, r in CONFOUND_CONTRASTS
        ),
        "",
        *_servable_lines(pooled=pooled),
        "",
        *_lead_lines(frame=frame),
        "",
        *_support_lines(sites=sites),
        "",
        *_generator_lines(frame=frame, members=_t3_members()),
        "",
        *_main_panel_lines(pooled=pooled),
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
    frame = build_rows()
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

    report = _report(frame=frame, losses=losses, sites=sites, job_list=all_jobs)
    report_path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
