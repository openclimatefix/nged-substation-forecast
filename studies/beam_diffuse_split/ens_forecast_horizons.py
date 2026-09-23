"""Score ECMWF ENS-driven power forecasts at each forecast horizon, for solar and for wind.

One-off throwaway script for the first step of
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. It reads the ENS members
`fetch_ens_forecast_horizons.py` extracted, and the solar and wind rows the past-weather studies
built (`weather_products.py`, `wind_products.py`, `blend_products.py`).

**The question is how accurate a power forecast driven by ECMWF ENS is at each horizon.** The answer
is an absolute mean-absolute-error leaderboard per technology: three ways of using ENS at each
horizon, four baselines that read no weather forecast, and two reference rows that are not
forecasts.

**A horizon is a whole UTC day after the ENS run's own day.** Band `day d` forecasts calendar day
`D` from the 00 UTC run of day `D - d`. The bands are days 0, 1, 2, 3, 5, 7, 10, and 14: every
day to day 3, where the change from one day to the next is steepest, then days 5, 7, and 10, which
span the 3-to-10-day band the live service's users care most about, and day 14, near the end of the
run.
Every band covers the same hours of the day. The live service reads a 00 UTC run from about 09:00
UTC (`ml_core.features._nwp.NWP_PUBLICATION_DELAY_HOURS` is 9), so most of day 0 is over before
the run can be used, and the day-0 row is a best case rather than a product. The day-ahead product
is day 1, which a forecast issued at 09:00 UTC reaches 15 to 39 hours after issue.

**Every row is scored at hourly resolution**, on the past-weather studies' own hourly rows: every
daylight hour for solar, and every hour for wind. The commissioning ramp, the zero half-hours, and
the outages are dropped exactly as there. A row is also dropped from every arm where any band's ENS
input or any baseline's input is missing, so every arm and baseline scores every row, and each
panel's rows are shared; `check_shared_rows` stops the run otherwise.

**ENS steps every 3 hours to lead 144 and every 6 hours beyond, so its values are upsampled to
hourly first, and the first part of the study measures how.** Every upsampling is applied to each
member before any member is averaged. A combination names one technique per field:

- `native`: not a combination but a reference, ENS on its own steps inside the band's day. For
  solar each step's power is the mean over every hour of the step, as ENS's step radiation is, with
  zero for each hour the sun is down, and a step missing any scored daylight hour is dropped; for
  wind the power is read at the step. It is not hourly, so it is scored on rows of its own.
- `linear` for every field: radiation read as instantaneous at its stamp, direction interpolated
  as an angle. This is what the production resample does today.
- `clear_sky` for radiation: rebuilt through the clear-sky index by
  `studies.resample.clear_sky_index_resample`.
- `clear_sky_conserving` for radiation: as `clear_sky`, each step's hours then scaled so they
  average to the step's own mean (`studies.resample.rescale_to_step_means`), which conserves every
  step's energy exactly.
- `pchip` for temperature: a shape-preserving cubic (`studies.resample.interpolate_pchip`). Solar
  temperature is read at each hour's midpoint.
- `components` for wind direction, or for wind speed: both winds interpolated as eastward and
  northward components, the direction, or the speed, then taken from them. The two are separate
  changes because the magnitude of interpolated components dips between stamps where the direction
  turns, which changes the speed as well as the direction.

**The rule that chooses the combination every other comparison uses was written before any result
existed**: start from `linear`, and try each change in `CHANGES` in order, each candidate being the
combination chosen so far with that one change made. A candidate replaces the combination chosen
so far only if its ensemble-mean arm's error is lower at day 1, on the 3-hour part of the grid, and
at day 7, on the 6-hour part. Every combination is fitted at every band, and the page shows each
candidate against the combination it was judged against, and against `linear`.

**Three ways of using ENS, each trained on the input it is scored with:**

- `ens_control`: the control member, through an XGBoost model trained on the control member.
- `ens_mean`: the mean of the 51 members' fields, through an XGBoost model trained on that mean.
  The mean wind direction is the direction of the mean wind vector.
- `ens_members`: each member through one XGBoost model, then the 51 power forecasts averaged. The
  model is trained on every member's rows stacked, all 51 carrying the same measured power, because
  that is the model the scored input needs: one mapping from a member's weather to power, applied to
  any member. Each member's row carries a weight of 1/51, so an hour weighs as much as it does in
  the other arms' fits (`studies.cross_validation.out_of_fold_member_forecasts`). The folds are
  whole months, so every member of a scored month is left out of training together.
  `ens_members_median` takes the median of the same 51 forecasts instead, because the median
  minimises the absolute error.

Two checks on the member-by-member arm. Row subsampling drops member rows rather than hours, so
day 1's member-by-member and ensemble-mean arms are also fitted without row subsampling
(`no_subsample`). And an exploratory arm, `ens_mean_applied`, is an XGBoost model trained on the
ensemble mean and applied to each member, the 51 forecasts averaged: it breaks the rule that each
arm is trained on the input it is scored with, and answers the question as first put literally.

**Each arm is fitted per generator, out of fold**, by `studies.cross_validation.out_of_fold_losses`:
five folds of whole months cut inside each era of the UKV record, three fitting seeds, the
absolute-error objective, curtailed hours dropped from training, and the prediction held to the
export cap at scoring. A solar arm is shown the hour's sun position, the hour of day, the day of
the year, the UKV era, and the ENS radiation and temperature; a wind arm the hour of day, the day
of the year, the UKV era, and the wind study's four columns from ENS: the 100 m speed, the 100 m
direction as sine and cosine, and the 10 m speed.

**The baselines read no weather forecast** and are set out in `studies.baselines`:
persistence, diurnal persistence, smart persistence, and climatology. Each is issued at 09:00 UTC
on the run's own day, when the live service can first read the run, except for day 0, whose
baselines are cut off at the run's 00 UTC init time: an issue at 09:00 on the target day would hand
them the target's own morning, which the day-0 run never saw.

**The reference rows are not forecasts: each is scored on the same rows to show how good the weather
input could be.** `era5` is ERA5, the reanalysis, as the past-weather studies gave it, available a
few days late. `live_gb_rich_xgb` is the blending study's best row that a live service anywhere in
Great Britain can read: UKV and ICON-EU, each with its neighbouring hours. Both are shown ERA5's
temperature, as in the past-weather studies.

**An exploratory arm, `ens_mean6_day1`, measures what the 6-hour step costs at a short lead**: day
1's ensemble mean given only every other stamp, its radiation the 6-hour mean of two 3-hour means,
upsampled by the chosen technique.

**The planned contrasts, named before any result existed**: `ens_mean_day1 - ens_mean_day0`,
`ens_mean_day7 - ens_mean_day0`, `ens_members_day1 - ens_mean_day1`,
`ens_mean_day1 - ens_control_day1`, and `ens_mean_day7 - climatology`. Every other number is
exploratory, and the upsampling contrasts too. The best baseline at each band is chosen after the
run, so a contrast against it is exploratory as well.

Run it with `uv run python studies/beam_diffuse_split/ens_forecast_horizons.py`, after
`fetch_ens_forecast_horizons.py` and the three past-weather studies. With `--report-only` it
rebuilds the report from the losses a full run saved.
"""

import argparse
import concurrent.futures
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from blend_products import (
    CONTRAST_HEADER,
    SOLAR,
    WIND,
    Domain,
    IntervalRecord,
    _interval,
    _line,
    _solar_frame,
    _wind_frame,
)
from build_dataset import _hourly_power as solar_hourly_power
from build_dataset import _pv_sites, _wind_sites
from fetch_ens_forecast_horizons import BAND_DAYS, ENSEMBLE_SIZE, OUTPUT_DIR, OUTPUT_PATH
from run_experiment import MAX_CONCURRENT_FITS, Job, run_all
from run_experiment import SHARED_FEATURES as SOLAR_SHARED_FEATURES
from studies.baselines import (
    clear_sky_index,
    climatology,
    diurnal_persistence,
    hourly_clear_sky,
    hourly_grid,
    issue_time,
    persistence,
    shrunk_persistence,
)
from studies.bootstrap import bootstrap_absolute
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    assign_folds,
    out_of_fold_forecasts_for_members,
    out_of_fold_member_forecasts,
    score_prediction,
    summarise_member_forecasts,
)
from studies.ensemble import check_one_run_per_hour
from studies.resample import (
    DEFAULT_DAYLIGHT_FLOOR_W_M2,
    clear_sky_index_resample,
    coarsen_to_six_hourly,
    interpolate_linear,
    interpolate_pchip,
    rescale_to_step_means,
    step_means,
    wind_components,
    wind_polar,
)
from studies.solar import zenith
from weather_products import with_eras
from wind_products import _hourly_power as wind_hourly_power

_LOG: Final[logging.Logger] = logging.getLogger("ens_forecast_horizons")

DomainType = Literal["solar", "wind"]

MethodType = str
"""An upsampling combination's name, a key of `COMBINATIONS`."""

TechniqueType = Literal["linear", "clear_sky", "clear_sky_conserving", "pchip", "components"]
"""How one field is upsampled."""

SettingType = Literal["pooled", "sensitivity", "no_subsample"]
"""The hyperparameter setting: the past-weather studies' two, and the primary setting with row
subsampling off, for the member-by-member sensitivity check."""

SETTINGS: Final[dict[SettingType, HyperParameters]] = {
    "pooled": PRIMARY_HYPER_PARAMETERS,
    "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
    "no_subsample": {**PRIMARY_HYPER_PARAMETERS, "subsample": 1.0},
}
"""`no_subsample` exists because row subsampling on the stacked member rows drops member rows
rather than hours, which regularises the member-by-member arm differently from an arm with one row
per hour."""

PERCENTAGE_POINTS: Final[float] = 100.0

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

CONTROL_MEMBER: Final[int] = 0
"""The ensemble member ECMWF runs from the unperturbed analysis."""

FINE_STEP_LAST_LEAD: Final[int] = 144
"""The last lead ENS publishes on 3-hour steps; beyond it the steps are 6 hours wide."""

COMBINATIONS: Final[dict[DomainType, dict[str, dict[str, TechniqueType]]]] = {
    "solar": {
        "linear": {"ghi": "linear", "temp": "linear"},
        "clear_sky": {"ghi": "clear_sky", "temp": "linear"},
        "clear_sky_conserving": {"ghi": "clear_sky_conserving", "temp": "linear"},
        "linear_pchip": {"ghi": "linear", "temp": "pchip"},
        "clear_sky_pchip": {"ghi": "clear_sky", "temp": "pchip"},
        "clear_sky_conserving_pchip": {"ghi": "clear_sky_conserving", "temp": "pchip"},
    },
    "wind": {
        "linear": {"speed": "linear", "direction": "linear"},
        "direction_components": {"speed": "linear", "direction": "components"},
        "speed_components": {"speed": "components", "direction": "linear"},
        "components": {"speed": "components", "direction": "components"},
    },
}
"""Every combination of techniques fitted, per field, by name.

For solar, `ghi` is the radiation and `temp` the temperature. For wind, `speed` and `direction`
apply to both heights: `components` for the speed takes the magnitude of the interpolated
eastward and northward components, and for the direction their bearing."""

CHANGES: Final[dict[DomainType, tuple[tuple[str, TechniqueType], ...]]] = {
    "solar": (("ghi", "clear_sky"), ("ghi", "clear_sky_conserving"), ("temp", "pchip")),
    "wind": (("direction", "components"), ("speed", "components")),
}
"""The changes the choosing rule tries, in order, each one field's technique from `linear`."""

DECIDING_DAYS: Final[tuple[int, int]] = (1, 7)
"""The bands the choice of technique rests on: one on the 3-hour grid, one on the 6-hour grid."""

WAYS: Final[tuple[str, ...]] = ("control", "mean", "members", "members_median")
"""The ways of using the members, as each arm's name spells them."""

APPLIED: Final[str] = "mean_applied"
"""The exploratory way: an XGBoost model trained on the ensemble mean, applied to each member, the
51 forecasts averaged. It breaks the train-on-what-you-score rule the three ways keep."""

BASELINES: Final[tuple[str, ...]] = ("persistence", "diurnal_persistence", "smart_persistence")
"""The baselines that depend on the band; climatology does not, and is one arm."""

REFERENCES: Final[tuple[str, ...]] = ("era5", "live_gb_rich_xgb")
"""The two reference rows, which are not forecasts."""

PLANNED: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day1", "ens_mean_day0"),
    ("ens_mean_day7", "ens_mean_day0"),
    ("ens_members_day1", "ens_mean_day1"),
    ("ens_mean_day1", "ens_control_day1"),
    ("ens_mean_day7", "climatology"),
)
"""The planned contrasts, named before any result existed."""

MEMBER_SENSITIVITY_DAYS: Final[tuple[int, ...]] = (1, 7)
"""The bands whose member-by-member arm is fitted at the second setting too."""

NO_SUBSAMPLE_DAY: Final[int] = 1
"""The band whose member-by-member and ensemble-mean arms are fitted without row subsampling too."""

EMULATED_DAY: Final[int] = 1
"""The band whose ensemble mean is also given 6-hourly stamps only, to cost the step width."""

SPAN: Final[tuple[datetime, datetime]] = (
    datetime(2024, 3, 25, tzinfo=UTC),
    datetime(2026, 10, 15, tzinfo=UTC),
)
"""The hours the clear-sky table covers, enough for every band of every run in the extract."""


def ens_arm(*, way: str, day: int) -> str:
    """Return the name of one way's arm at one band.

    Args:
        way: One of `WAYS`, `APPLIED`, or `mean6` for the 6-hourly emulation.
        day: The band's day.

    Returns:
        The arm name, such as `ens_mean_day1`.
    """
    return f"ens_{way}_day{day}"


def upsampling_arm(*, method: str, day: int) -> str:
    """Return the name of one upsampling combination's ensemble-mean arm at one band.

    Args:
        method: A key of `COMBINATIONS`, or `native`.
        day: The band's day.

    Returns:
        The arm name, such as `up_linear_day1`.
    """
    return f"up_{method}_day{day}"


def baseline_arm(*, name: str, day: int) -> str:
    """Return the name of one baseline at one band.

    Args:
        name: One of `BASELINES`.
        day: The band's day.

    Returns:
        The arm name, such as `persistence_day1`.
    """
    return f"{name}_day{day}"


def fields(*, domain: DomainType) -> tuple[str, ...]:
    """Return the weather fields an ENS arm is shown, in the order it is shown them.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The field names every ENS arm's columns end in.
    """
    if domain == "solar":
        return ("ghi", "temp")
    return ("speed_100m", "sin_100m", "cos_100m", "speed_10m")


def ens_columns(*, arm: str, domain: DomainType) -> tuple[str, ...]:
    """Return one ENS arm's weather columns.

    Args:
        arm: The arm's name.
        domain: `solar` or `wind`.

    Returns:
        `<arm>_<field>` for each of `fields`.
    """
    return tuple(f"{arm}_{field}" for field in fields(domain=domain))


# --- ENS on its native steps ------------------------------------------------------------------


@dataclass(frozen=True)
class Steps:
    """One band's ENS members on their native steps, one row per (site, run, member)."""

    keys: pl.DataFrame
    """`site`, `init_time`, and `ensemble_member`, sorted, every (site, run) holding all members."""
    leads: np.ndarray
    """Each step's lead in hours."""
    widths: np.ndarray
    """How many hours each step's radiation averages over."""
    values: dict[str, np.ndarray]
    """Each field's values, shape (n_series, n_steps)."""


def _step_width(lead: int) -> int:
    """Return the width in hours of the radiation step ending at `lead`.

    Args:
        lead: The step's lead.

    Returns:
        3 to lead 144 and 6 beyond it.
    """
    return 3 if lead <= FINE_STEP_LAST_LEAD else 6


def _members(*, sites: list[str]) -> pl.DataFrame:
    """Read the extract for some generators.

    Args:
        sites: The generator labels.

    Returns:
        One row per generator, run, valid time and member.
    """
    return pl.read_parquet(OUTPUT_PATH).filter(pl.col("site").is_in(sites))


def band_steps(
    *, members: pl.DataFrame, day: int, domain: DomainType, six_hourly: bool = False
) -> Steps:
    """Arrange one band's members as arrays over the band's native steps and a margin either side.

    A (site, run) is dropped whole where any member lacks any step, so every run kept has all 51
    members, in member order. With `six_hourly`, `studies.resample.coarsen_to_six_hourly` keeps
    only the steps at multiples of 6 hours, each pair of 3-hour radiation steps averaged into the
    6-hour mean it makes up.

    Args:
        members: The extract, for one technology's generators.
        day: The band's day.
        domain: `solar` or `wind`, which decides the fields.
        six_hourly: Whether to emulate 6-hourly steps.

    Returns:
        The arrays.

    Raises:
        ValueError: If a kept run does not hold every member.
    """
    first, last = max(24 * day - 6, 0), 24 * day + 30
    columns = (
        ("ghi_w_m2", "temp_c")
        if domain == "solar"
        else ("speed_100m", "direction_100m", "speed_10m", "direction_10m")
    )
    rows = members.filter(pl.col("lead_hours").is_between(first, last))
    if domain == "solar":
        # Radiation is null at lead 0, where a period mean has no period to cover.
        rows = rows.filter(pl.col("lead_hours") > 0)
    leads = np.sort(rows["lead_hours"].unique().to_numpy())
    wide = rows.pivot(
        on="lead_hours",
        index=["site", "init_time", "ensemble_member"],
        values=list(columns),
        sort_columns=True,
    )
    names = {column: [f"{column}_{lead}" for lead in leads] for column in columns}
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
        msg = f"day {day}: a kept run does not hold members 0 to {ENSEMBLE_SIZE - 1} in order"
        raise ValueError(msg)
    values = {column: kept.select(names[column]).to_numpy() for column in columns}
    step_leads = leads.astype(np.float64)
    widths = np.array([_step_width(int(lead)) for lead in leads])
    if six_hourly:
        step_leads, values = coarsen_to_six_hourly(
            leads=step_leads,
            values=values,
            period_means=frozenset({"ghi_w_m2"}),
            last_three_hourly_lead=FINE_STEP_LAST_LEAD,
        )
        widths = np.full(len(step_leads), 6)
    return Steps(
        keys=kept.select("site", "init_time", "ensemble_member"),
        leads=step_leads,
        widths=widths,
        values=values,
    )


# --- Upsampling to hourly -----------------------------------------------------------------------


def target_leads(*, day: int, domain: DomainType) -> np.ndarray:
    """Return the leads of the hours one band scores.

    Args:
        day: The band's day.
        domain: `solar` or `wind`.

    Returns:
        For solar, the end of every hour of the band's day, since a solar hour is labelled by its
        end; for wind, the start of every hour, since a wind value is read at its label.
    """
    offset = 1 if domain == "solar" else 0
    return np.arange(24 * day + offset, 24 * day + 24 + offset, dtype=np.float64)


def _clear_sky_arrays(
    *, steps: Steps, targets: np.ndarray, clear_sky: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clear-sky mean over each step and over each target hour, per series.

    Args:
        steps: The band's steps.
        targets: The target hours' leads (each hour's end).
        clear_sky: The hourly clear-sky table from `hourly_clear_sky`.

    Returns:
        Shapes (n_series, n_steps) and (n_series, n_targets).

    Raises:
        ValueError: If a run's clear-sky hours are missing from the table.
    """
    runs = steps.keys.select("site", "init_time").unique(maintain_order=True)
    first_hour = int(min(steps.leads[0] - steps.widths[0] + 1, targets[0]))
    last_hour = int(max(steps.leads[-1], targets[-1]))
    hours = np.arange(first_hour, last_hour + 1)
    table = (
        runs.with_row_index("run")
        .join(pl.DataFrame({"lead": hours}), how="cross")
        .with_columns(time=pl.col("init_time") + pl.duration(hours=pl.col("lead")))
        .join(clear_sky, on=["site", "time"], how="left")
        .sort("run", "lead")
    )
    if table["clear_sky_w_m2"].null_count():
        msg = "a run's clear-sky hours are missing from the table"
        raise ValueError(msg)
    per_hour = table["clear_sky_w_m2"].to_numpy().reshape(runs.height, len(hours))
    run_of_series = np.repeat(np.arange(runs.height), ENSEMBLE_SIZE)
    return (
        step_means(
            hourly=per_hour, first_hour=first_hour, step_leads=steps.leads, step_widths=steps.widths
        )[run_of_series],
        per_hour[:, (targets - first_hour).astype(int)][run_of_series],
    )


def _upsampled_fields(
    *, steps: Steps, day: int, domain: DomainType, clear_sky: pl.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    """Upsample every member of one band to hourly, every technique for every field.

    A solar temperature is read at each hour's midpoint, where the hour's power is centred; a
    solar radiation is a mean over the hour ending at its target; a wind is read at its target.

    Args:
        steps: The band's steps.
        day: The band's day.
        domain: `solar` or `wind`.
        clear_sky: The hourly clear-sky table, read by the clear-sky-index techniques.

    Returns:
        Per field of `COMBINATIONS`, per technique, the hourly values, shape (n_series, n_targets);
        wind's `direction` holds degrees.
    """
    targets = target_leads(day=day, domain=domain)
    x = steps.leads
    if domain == "solar":
        temp, ghi = steps.values["temp_c"], steps.values["ghi_w_m2"]
        step_clear_sky, target_clear_sky = _clear_sky_arrays(
            steps=steps, targets=targets, clear_sky=clear_sky
        )
        midpoints = x - steps.widths / 2.0
        clear = clear_sky_index_resample(
            values=ghi,
            step_clear_sky=step_clear_sky,
            step_midpoints=midpoints,
            morning=np.mod(midpoints, 24.0) < 12.0,
            target_clear_sky=target_clear_sky,
            target_midpoints=targets - 0.5,
            daylight_floor_w_m2=DEFAULT_DAYLIGHT_FLOOR_W_M2,
        )
        inside = (x > targets[0] - 1) & (x <= targets[-1])
        return {
            "ghi": {
                "linear": interpolate_linear(values=ghi, x=x, targets=targets),
                "clear_sky": clear,
                "clear_sky_conserving": rescale_to_step_means(
                    values=clear,
                    target_leads=targets,
                    step_values=ghi[:, inside],
                    step_leads=x[inside],
                    step_widths=steps.widths[inside],
                ),
            },
            "temp": {
                "linear": interpolate_linear(values=temp, x=x, targets=targets - 0.5),
                "pchip": interpolate_pchip(values=temp, x=x, targets=targets - 0.5),
            },
        }
    out: dict[str, dict[str, np.ndarray]] = {}
    for height in ("100m", "10m"):
        speed = steps.values[f"speed_{height}"]
        direction = steps.values[f"direction_{height}"]
        u, v = wind_components(speed=speed, direction_deg=direction)
        vector_speed, vector_direction = wind_polar(
            u=interpolate_linear(values=u, x=x, targets=targets),
            v=interpolate_linear(values=v, x=x, targets=targets),
        )
        out[f"speed_{height}"] = {
            "linear": interpolate_linear(values=speed, x=x, targets=targets),
            "components": vector_speed,
        }
        out[f"direction_{height}"] = {
            "linear": interpolate_linear(values=direction, x=x, targets=targets),
            "components": vector_direction,
        }
    return out


def combine(
    *,
    steps: Steps,
    upsampled: dict[str, dict[str, np.ndarray]],
    day: int,
    domain: DomainType,
    method: MethodType,
) -> pl.DataFrame:
    """Assemble one combination's hourly fields for every member of one band.

    Args:
        steps: The band's steps.
        upsampled: The output of `_upsampled_fields`.
        day: The band's day.
        domain: `solar` or `wind`.
        method: A key of `COMBINATIONS[domain]`.

    Returns:
        One row per (site, time, member), with `init_time` and `fields(domain=domain)`.
    """
    choice = COMBINATIONS[domain][method]
    if domain == "solar":
        out = {"ghi": upsampled["ghi"][choice["ghi"]], "temp": upsampled["temp"][choice["temp"]]}
    else:
        direction = np.radians(upsampled["direction_100m"][choice["direction"]])
        out = {
            "speed_100m": upsampled["speed_100m"][choice["speed"]],
            "sin_100m": np.sin(direction),
            "cos_100m": np.cos(direction),
            "speed_10m": upsampled["speed_10m"][choice["speed"]],
        }
    return _long(steps=steps, targets=target_leads(day=day, domain=domain), values=out)


def native(*, steps: Steps, day: int, domain: DomainType) -> pl.DataFrame:
    """Return every member at its own steps inside the band's day, as the native technique reads.

    Only the steps the band's own day holds are kept: for solar the steps ending after lead 24d and
    at or before 24d + 24, for wind the stamps at 24d up to but not including 24d + 24. The margins
    either side belong to the neighbouring bands' days and would put two runs on one hour.

    Args:
        steps: The band's steps.
        day: The band's day.
        domain: `solar` or `wind`.

    Returns:
        One row per (site, step end, member), with `init_time` and `fields(domain=domain)`.
    """
    if domain == "solar":
        own = (steps.leads > 24 * day) & (steps.leads <= 24 * day + 24)
        out = {"ghi": steps.values["ghi_w_m2"][:, own], "temp": steps.values["temp_c"][:, own]}
    else:
        own = (steps.leads >= 24 * day) & (steps.leads < 24 * day + 24)
        radians = np.radians(steps.values["direction_100m"][:, own])
        out = {
            "speed_100m": steps.values["speed_100m"][:, own],
            "sin_100m": np.sin(radians),
            "cos_100m": np.cos(radians),
            "speed_10m": steps.values["speed_10m"][:, own],
        }
    return _long(steps=steps, targets=steps.leads[own], values=out)


def _long(*, steps: Steps, targets: np.ndarray, values: dict[str, np.ndarray]) -> pl.DataFrame:
    """Turn arrays over (series, target) into rows keyed by site, run, time, and member.

    Args:
        steps: The band's steps, whose keys name each series.
        targets: Each column's lead.
        values: Each field's array, shape (n_series, n_targets).

    Returns:
        One row per (site, time, member), with the run's `init_time`.
    """
    repeat = len(targets)
    keys = steps.keys.select(
        pl.col("site", "init_time", "ensemble_member").repeat_by(repeat).explode()
    )
    return keys.with_columns(
        lead=pl.Series(np.tile(targets, steps.keys.height)),
        **{name: pl.Series(array.reshape(-1).astype(np.float64)) for name, array in values.items()},
    ).select(
        "site",
        *values,
        init_time=pl.col("init_time").dt.replace_time_zone("UTC", non_existent="raise"),
        time=pl.col("init_time").dt.replace_time_zone("UTC", non_existent="raise")
        + pl.duration(hours=pl.col("lead").cast(pl.Int64)),
        member=pl.col("ensemble_member").cast(pl.Int32),
    )


def reduce_members(*, hourly: pl.DataFrame, domain: DomainType, way: str) -> pl.DataFrame:
    """Reduce every member's hourly fields to one value per (site, time).

    Args:
        hourly: One row per (site, time, member), with `init_time`.
        domain: `solar` or `wind`.
        way: `control` for the control member's own values, `mean` for the ensemble mean.

    Returns:
        One row per (site, time), with `fields(domain=domain)`.

    Raises:
        ValueError: Unless every (site, time) holds one run and all its members.
    """
    check_one_run_per_hour(hourly=hourly, members=ENSEMBLE_SIZE)
    if way == "control":
        return hourly.filter(pl.col("member") == CONTROL_MEMBER).drop("member", "init_time")
    if domain == "solar":
        return hourly.group_by("site", "time").agg(pl.col("ghi", "temp").mean())
    # The mean direction is the mean wind vector's: each member's direction weighted by its speed.
    return (
        hourly.group_by("site", "time")
        .agg(
            pl.col("speed_100m", "speed_10m").mean(),
            east=(pl.col("speed_100m") * pl.col("sin_100m")).mean(),
            north=(pl.col("speed_100m") * pl.col("cos_100m")).mean(),
        )
        .with_columns(norm=(pl.col("east") ** 2 + pl.col("north") ** 2).sqrt())
        .select(
            "site",
            "time",
            "speed_100m",
            sin_100m=pl.col("east") / pl.col("norm"),
            cos_100m=pl.col("north") / pl.col("norm"),
            speed_10m="speed_10m",
        )
    )


def _prefixed(*, frame: pl.DataFrame, arm: str, domain: DomainType) -> pl.DataFrame:
    """Rename a reduced frame's fields to one arm's column names.

    Args:
        frame: One row per (site, time), with `fields(domain=domain)`.
        arm: The arm.
        domain: `solar` or `wind`.

    Returns:
        `site`, `time`, and the arm's columns.
    """
    return frame.select(
        "site",
        "time",
        *(
            pl.col(field).alias(column)
            for field, column in zip(
                fields(domain=domain), ens_columns(arm=arm, domain=domain), strict=True
            )
        ),
    )


# --- The rows -----------------------------------------------------------------------------------


@dataclass
class Inputs:
    """Everything one technology's fits read."""

    frame: pl.DataFrame
    """The hourly rows every arm scores, with every non-member arm's columns and every baseline."""
    members: dict[int, pl.DataFrame]
    """Per band, one row per (site, time, member) with the chosen technique's fields."""
    native: dict[int, pl.DataFrame]
    """Per deciding band, the native technique's own rows with its columns."""
    inputs: pl.DataFrame
    """The ensemble mean at every band under every technique, for the charts."""


def _hourly_power(*, domain: DomainType) -> pl.DataFrame:
    """Return NGED's hourly power as published, on each technology's hour convention.

    Args:
        domain: `solar` or `wind`.

    Returns:
        One row per (site, time) with `power_mw`.
    """
    if domain == "solar":
        return solar_hourly_power(sites=_pv_sites()).select("site", "time", "power_mw")
    return wind_hourly_power(sites=_wind_sites()).select("site", "time", "power_mw")


def _base_frame(*, domain: DomainType) -> pl.DataFrame:
    """Return the past-weather study's hourly rows, with the references' columns.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The rows, from the first run's first scored day onwards.
    """
    frame = _solar_frame() if domain == "solar" else _wind_frame()
    return frame.filter(pl.col("time") > SPAN[0] + timedelta(days=1))


def _day_start(*, domain: DomainType) -> pl.Expr:
    """Return midnight UTC at the start of each row's day.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The day a solar hour's midpoint falls on, or the day a wind instant falls on.
    """
    time = pl.col("time") - pl.duration(minutes=30) if domain == "solar" else pl.col("time")
    return time.dt.truncate("1d")


def _with_baselines(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Add every band-dependent baseline's forecast input.

    Args:
        frame: The rows.
        domain: `solar` or `wind`.

    Returns:
        `frame` with `persistence_day<d>`, `diurnal_persistence_day<d>`, and for solar
        `clear_sky_index_day<d>` for every band, and `clear_sky_w_m2` for solar.
    """
    hourly = _hourly_power(domain=domain).filter(pl.col("time") >= SPAN[0] - timedelta(days=8))
    lag = timedelta(0) if domain == "solar" else timedelta(minutes=30)
    columns = []
    grid = None
    if domain == "solar":
        clear_sky = hourly_clear_sky(
            sites=_pv_sites(), first=SPAN[0] - timedelta(days=2), last=SPAN[1]
        )
        grid = hourly_grid(hourly=hourly).join(clear_sky, on=["site", "time"], how="inner")
        frame = frame.join(clear_sky, on=["site", "time"], how="left")
    for day in BAND_DAYS:
        keys = frame.select(
            "site", "time", issue_time=issue_time(day_start=_day_start(domain=domain), day=day)
        )
        columns.append(
            persistence(keys=keys, hourly=hourly, observed_lag=lag).alias(
                baseline_arm(name="persistence", day=day)
            )
        )
        columns.append(
            diurnal_persistence(keys=keys, hourly=hourly, day=day).alias(
                baseline_arm(name="diurnal_persistence", day=day)
            )
        )
        if grid is not None:
            columns.append(
                clear_sky_index(keys=keys, hourly=grid).alias(f"clear_sky_index_day{day}")
            )
    return frame.with_columns(columns)


def _clear_sky_table(*, domain: DomainType) -> pl.DataFrame:
    """Return the hourly clear-sky table the solar resample reads, or an empty frame for wind.

    Args:
        domain: `solar` or `wind`.

    Returns:
        `hourly_clear_sky` over `SPAN` at the solar generators.
    """
    if domain == "wind":
        return pl.DataFrame()
    return hourly_clear_sky(sites=_pv_sites(), first=SPAN[0], last=SPAN[1])


def build_inputs(*, domain: DomainType) -> Inputs:
    """Build one technology's rows, and every combination's ensemble mean at every band.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The inputs, with every combination's ensemble-mean columns on the rows every arm and
        baseline can score, the native technique's own rows, and no member rows yet.
    """
    frame = _with_baselines(frame=_base_frame(domain=domain), domain=domain)
    extract = _members(sites=sorted(frame["site"].unique().to_list()))
    clear_sky = _clear_sky_table(domain=domain)
    native_rows: dict[int, pl.DataFrame] = {}
    chart_inputs = []
    for day in BAND_DAYS:
        steps = band_steps(members=extract, day=day, domain=domain)
        upsampled = _upsampled_fields(steps=steps, day=day, domain=domain, clear_sky=clear_sky)
        for method in COMBINATIONS[domain]:
            mean = reduce_members(
                hourly=combine(
                    steps=steps, upsampled=upsampled, day=day, domain=domain, method=method
                ),
                domain=domain,
                way="mean",
            )
            frame = frame.join(
                _prefixed(frame=mean, arm=upsampling_arm(method=method, day=day), domain=domain),
                on=["site", "time"],
                how="left",
            )
            chart_inputs.append(mean.with_columns(day=pl.lit(day), method=pl.lit(method)))
        native_mean = reduce_members(
            hourly=native(steps=steps, day=day, domain=domain), domain=domain, way="mean"
        )
        chart_inputs.append(native_mean.with_columns(day=pl.lit(day), method=pl.lit("native")))
        native_rows[day] = native_mean
        _LOG.info("%s day %d: %d runs upsampled", domain, day, steps.keys.height // ENSEMBLE_SIZE)
    required = [
        column
        for day in BAND_DAYS
        for method in COMBINATIONS[domain]
        for column in ens_columns(arm=upsampling_arm(method=method, day=day), domain=domain)
    ]
    required += [baseline_arm(name=name, day=day) for day in BAND_DAYS for name in BASELINES[:2]]
    if domain == "solar":
        required += [f"clear_sky_index_day{day}" for day in BAND_DAYS] + ["clear_sky_w_m2"]
    return Inputs(
        frame=_complete(frame=frame, columns=required),
        members={},
        native=native_rows,
        inputs=pl.concat(chart_inputs, how="diagonal"),
    )


def _complete(*, frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Keep the rows every arm can score, and cut the folds afresh on them.

    Args:
        frame: The rows, with every arm's columns.
        columns: Every arm's and baseline's own input columns.

    Returns:
        The rows with no null in `columns`, with `era`, `era_code` and `fold` recomputed.
    """
    kept = frame.drop_nulls(columns).drop("era", "era_code", "fold").sort("site", "time")
    _LOG.info("kept %d of %d rows with every arm's input", kept.height, frame.height)
    return with_eras(frame=kept)


# --- Fitting -----------------------------------------------------------------------------------


def shared_features(*, domain: Domain) -> tuple[str, ...]:
    """Return the columns every ENS arm is shown besides its weather.

    Args:
        domain: `blend_products.SOLAR` or `blend_products.WIND`.

    Returns:
        For solar, the past-weather studies' shared columns without ERA5's temperature, which each
        ENS arm replaces with its own; for wind, the wind study's shared columns.
    """
    if domain.name == "solar":
        return (*(c for c in SOLAR_SHARED_FEATURES if c != "temp_c"), "era_code")
    return domain.shared_features


def reference_features(*, domain: Domain) -> dict[str, tuple[str, ...]]:
    """Return each reference arm's feature columns, as its past-weather study showed them.

    Args:
        domain: The domain.

    Returns:
        Arm name to feature columns.
    """
    return {
        "era5": (*domain.shared_features, *domain.columns("era5")),
        "live_gb_rich_xgb": (
            *domain.shared_features,
            *domain.rich_columns("ukv"),
            *domain.rich_columns("icon_eu"),
        ),
    }


def _stacked(
    *, frame: pl.DataFrame, members: pl.DataFrame, arm: str, domain: Domain
) -> pl.DataFrame:
    """Stack every member's fields under the row they belong to, renamed to one arm's columns.

    Args:
        frame: The rows.
        members: One row per (site, time, member) with the fields.
        arm: The arm whose column names the fields take.
        domain: The domain.

    Returns:
        One row per (site, time, member), with the fit loop's columns.
    """
    columns = ens_columns(arm=arm, domain=domain.name)
    carried = ["site", "time", "month", "fold", "cap_mw", "constrained", "effective_capacity_mw"]
    carried += ["power_mw", *shared_features(domain=domain)]
    return (
        frame.select(carried)
        .join(
            members.drop("init_time").rename(
                dict(zip(fields(domain=domain.name), columns, strict=True))
            ),
            on=["site", "time"],
        )
        .sort("site", "time", "member")
    )


def _member_losses(
    *, frame: pl.DataFrame, summary: pl.DataFrame, day: int, setting: str, ways: dict[str, str]
) -> pl.DataFrame:
    """Score the reductions of the member forecasts as arms.

    Args:
        frame: The rows.
        summary: The output of `studies.cross_validation.summarise_member_forecasts`.
        day: The band's day.
        setting: The setting's name.
        ways: Each arm's way, mapped to the summary column it is scored on.

    Returns:
        The arms' losses.

    Raises:
        ValueError: If a row's forecast does not rest on every member.
    """
    if not (summary["members"] == ENSEMBLE_SIZE).all():
        msg = f"day {day}: a row's forecast does not rest on {ENSEMBLE_SIZE} members"
        raise ValueError(msg)
    return pl.concat(
        [
            losses_from_prediction(
                frame=frame,
                prediction=summary.select("site", "time", "seed", prediction=reduction),
                arm=ens_arm(way=way, day=day),
                setting=setting,
            )
            for way, reduction in ways.items()
        ]
    )


def _fit_members(
    *, frame: pl.DataFrame, members: pl.DataFrame, day: int, domain: Domain, setting: SettingType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit one band's member-by-member arm and score the mean and the median of its forecasts.

    `studies.cross_validation.out_of_fold_member_forecasts` fits one model per generator on every
    member's rows, each weighted 1/51, with the folds cut by time so that no member of a scored
    month is trained on.

    Args:
        frame: The rows.
        members: One row per (site, time, member) with the chosen combination's fields.
        day: The band's day.
        domain: The domain.
        setting: The hyperparameter setting.

    Returns:
        The losses of the mean and median arms, and the summary of the 51 forecasts per row and
        seed.
    """
    arm = ens_arm(way="members", day=day)
    stacked = _stacked(frame=frame, members=members, arm=arm, domain=domain)
    features = [*shared_features(domain=domain), *ens_columns(arm=arm, domain=domain.name)]

    def _one(site: str) -> pl.DataFrame:
        return summarise_member_forecasts(
            forecasts=out_of_fold_member_forecasts(
                site_rows=stacked.filter(pl.col("site") == site),
                features=features,
                target="power_mw",
                hyper_parameters=SETTINGS[setting],
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        summary = pl.concat(list(pool.map(_one, sorted(frame["site"].unique().to_list()))))
    losses = _member_losses(
        frame=frame,
        summary=summary,
        day=day,
        setting=setting,
        ways={"members": "mean", "members_median": "median"},
    )
    return losses, summary.with_columns(arm=pl.lit(arm), setting=pl.lit(setting))


def _fit_mean_applied(
    *, frame: pl.DataFrame, members: pl.DataFrame, day: int, domain: Domain
) -> pl.DataFrame:
    """Fit the exploratory arm: a model trained on the ensemble mean, applied to each member.

    Args:
        frame: The rows, with the band's ensemble-mean columns.
        members: One row per (site, time, member) with the chosen combination's fields.
        day: The band's day.
        domain: The domain.

    Returns:
        The arm's losses at the primary setting, scored on the mean of the 51 forecasts.
    """
    mean_arm = ens_arm(way="mean", day=day)
    stacked = _stacked(frame=frame, members=members, arm=mean_arm, domain=domain)
    features = [*shared_features(domain=domain), *ens_columns(arm=mean_arm, domain=domain.name)]

    def _one(site: str) -> pl.DataFrame:
        return summarise_member_forecasts(
            forecasts=out_of_fold_forecasts_for_members(
                site_rows=frame.filter(pl.col("site") == site),
                member_rows=stacked.filter(pl.col("site") == site),
                features=features,
                target="power_mw",
                hyper_parameters=SETTINGS["pooled"],
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        summary = pl.concat(list(pool.map(_one, sorted(frame["site"].unique().to_list()))))
    return _member_losses(
        frame=frame, summary=summary, day=day, setting="pooled", ways={APPLIED: "mean"}
    )


def losses_from_prediction(
    *, frame: pl.DataFrame, prediction: pl.DataFrame, arm: str, setting: str
) -> pl.DataFrame:
    """Score a prediction per (site, time, seed) with `studies.cross_validation.score_prediction`.

    Args:
        frame: The rows, with `power_mw`, `cap_mw`, and the fold and month labels.
        prediction: `site`, `time`, `seed`, and `prediction` in MW.
        arm: The arm's name.
        setting: The setting's name.

    Returns:
        One row per (site, time, seed), labelled with the arm and the setting.
    """
    return score_prediction(rows=frame, prediction=prediction, target="power_mw").with_columns(
        arm=pl.lit(arm), setting=pl.lit(setting), target=pl.lit("power_mw")
    )


def _baseline_losses(
    *, frame: pl.DataFrame, domain: DomainType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score every baseline at both settings, the same forecast for every seed.

    A baseline has no fitting seed, so its forecast is repeated for each of `SEEDS`, which lets the
    paired bootstrap pair it with a fitted arm seed by seed. It has no hyperparameters either, so
    the same losses stand under the primary and the second setting.

    Args:
        frame: The rows, with each baseline's input columns.
        domain: `solar` or `wind`.

    Returns:
        Every baseline's losses, and the wind smart persistence's fitted weights.
    """
    frame = frame.with_columns(climatology=climatology(frame=frame))
    forecasts = {"climatology": pl.col("climatology")}
    weights = []
    for day in BAND_DAYS:
        for name in ("persistence", "diurnal_persistence"):
            forecasts[baseline_arm(name=name, day=day)] = pl.col(baseline_arm(name=name, day=day))
        smart = baseline_arm(name="smart_persistence", day=day)
        if domain == "solar":
            forecasts[smart] = pl.col(f"clear_sky_index_day{day}") * pl.col("clear_sky_w_m2")
        else:
            shrunk = shrunk_persistence(
                frame=frame, persisted=baseline_arm(name="persistence", day=day)
            )
            frame = frame.with_columns(shrunk["shrunk"].alias(smart))
            forecasts[smart] = pl.col(smart)
            weights.append(
                frame.select("site", "fold")
                .with_columns(weight=shrunk["weight"], day=pl.lit(day))
                .unique()
            )
    seeds = pl.DataFrame({"seed": list(SEEDS)}, schema={"seed": pl.Int32})
    losses = [
        losses_from_prediction(
            frame=frame,
            prediction=frame.select("site", "time", prediction=expression).join(seeds, how="cross"),
            arm=arm,
            setting=setting,
        )
        for arm, expression in forecasts.items()
        for setting in ("pooled", "sensitivity")
    ]
    return pl.concat(losses), (pl.concat(weights) if weights else pl.DataFrame())


def _daylight(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return, for every hour of every solar generator's span, whether its sun is up.

    The rule is `build_dataset`'s: the sun above the horizon at the hour's midpoint.

    Args:
        frame: The solar rows, for the generators and their span.

    Returns:
        One row per (site, time), with `daylight`.
    """
    sites = _pv_sites().filter(pl.col("site").is_in(frame["site"].unique().to_list()))
    hours = pl.datetime_range(SPAN[0], SPAN[1], interval="1h", eager=True, time_zone="UTC")
    parts = []
    for site, latitude, longitude in sites.select("site", "latitude", "longitude").iter_rows():
        angle = zenith(stamps=hours.dt.offset_by("-30m"), latitude=latitude, longitude=longitude)
        parts.append(pl.DataFrame({"site": site, "time": hours, "daylight": angle < 90.0}))
    return pl.concat(parts)


def _native_solar_rows(*, frame: pl.DataFrame, day: int) -> pl.DataFrame:
    """Average the solar rows up to ENS's own steps, as the native technique scores them.

    A step's power is the mean over every hour of the step, as ENS's step radiation is: each
    daylight hour's measured power and zero for each hour with the sun below the horizon. A step
    with any daylight hour missing from the scored rows, or with no daylight hour, is dropped. The
    sun position, the hour of day, and the extraterrestrial flux are the means over the step's
    daylight hours.

    Args:
        frame: The solar rows.
        day: The band's day.

    Returns:
        One row per (site, step end), with the fit loop's columns.
    """
    width = _step_width(24 * day + 24)
    step_end = pl.col("time").dt.truncate(f"{width}h")
    step_end = (
        pl.when(step_end == pl.col("time"))
        .then(step_end)
        .otherwise(step_end + pl.duration(hours=width))
    )
    hours = _daylight(frame=frame).join(
        frame.select(
            "site",
            "time",
            "power_mw",
            "solar_zenith_deg",
            "solar_azimuth_deg",
            "extraterrestrial_horizontal_w_m2",
            "cap_mw",
            "constrained",
            "effective_capacity_mw",
            "era_code",
            scored=pl.lit(value=True),
        ),
        on=["site", "time"],
        how="left",
    )
    daylit = pl.col("daylight")
    return (
        hours.with_columns(step=step_end)
        .group_by("site", "step")
        .agg(
            hours=pl.len(),
            daylight_hours=daylit.sum(),
            scored_daylight=(daylit & pl.col("scored").fill_null(value=False)).sum(),
            power_mw=pl.when(daylit).then(pl.col("power_mw")).otherwise(0.0).sum() / width,
            solar_zenith_deg=pl.col("solar_zenith_deg").mean(),
            solar_azimuth_deg=pl.col("solar_azimuth_deg").mean(),
            extraterrestrial_horizontal_w_m2=pl.col("extraterrestrial_horizontal_w_m2").mean(),
            cap_mw=pl.col("cap_mw").mean(),
            constrained=pl.col("constrained").any(),
            effective_capacity_mw=pl.col("effective_capacity_mw").drop_nulls().first(),
            era_code=pl.col("era_code").drop_nulls().first(),
        )
        .filter(
            (pl.col("hours") == width)
            & (pl.col("daylight_hours") > 0)
            & (pl.col("scored_daylight") == pl.col("daylight_hours"))
        )
        .rename({"step": "time"})
        .with_columns(
            hour_of_day=pl.col("time").dt.hour(),
            day_of_year=pl.col("time").dt.ordinal_day(),
            month=pl.col("time").dt.strftime("%Y-%m"),
        )
        .drop("hours", "daylight_hours", "scored_daylight")
    )


def _native_losses(*, inputs: Inputs, domain: Domain) -> pl.DataFrame:
    """Fit the native technique's ensemble-mean arm at every band, on its own rows.

    For solar, each row is one ENS step at one generator (`_native_solar_rows`). For wind, each row
    is a scored hour at one of the band's ENS stamps.

    Args:
        inputs: The technology's inputs.
        domain: The domain.

    Returns:
        The losses.
    """
    outputs = []
    for day, mean in inputs.native.items():
        arm = upsampling_arm(method="native", day=day)
        columns = ens_columns(arm=arm, domain=domain.name)
        if domain.name == "solar":
            rows = _native_solar_rows(frame=inputs.frame, day=day)
        else:
            width = _step_width(24 * day + 24)
            rows = inputs.frame.filter(pl.col("time").dt.hour() % width == 0)
        rows = (
            rows.join(_prefixed(frame=mean, arm=arm, domain=domain.name), on=["site", "time"])
            .drop("fold", "era", strict=False)
            .sort("site", "time")
        )
        rows = assign_folds(
            dataset=rows.with_columns(
                era=pl.when(pl.col("era_code") == 1).then(pl.lit("post")).otherwise(pl.lit("pre"))
            ),
            by=("site", "era"),
        )
        job: Job = (
            arm,
            "pooled",
            "power_mw",
            (*shared_features(domain=domain), *columns),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        outputs.append(run_all(dataset=rows, jobs=[job]))
    return pl.concat(outputs, how="diagonal")


class Decision(NamedTuple):
    """One step of the choosing rule: a candidate combination and the one it was judged against."""

    candidate: str
    against: str
    changes: dict[int, float]
    adopted: bool


def _combination_name(*, domain: DomainType, choice: dict[str, TechniqueType]) -> str:
    """Return the name of the combination that makes these choices.

    Args:
        domain: `solar` or `wind`.
        choice: Each field's technique.

    Returns:
        The key of `COMBINATIONS[domain]` whose techniques equal `choice`.
    """
    return next(name for name, techniques in COMBINATIONS[domain].items() if techniques == choice)


def choose_method(*, losses: pl.DataFrame, domain: DomainType) -> tuple[MethodType, list[Decision]]:
    """Apply the rule the module docstring states, one change at a time.

    Starting from `linear`, each candidate is the combination chosen so far with exactly one of
    `CHANGES[domain]` made, and it replaces the combination chosen so far only if its ensemble-mean
    arm's error is lower at both of `DECIDING_DAYS`. A technique rejected once is never tried again
    as part of a later candidate, because each later candidate starts from the chosen combination.

    Args:
        losses: The upsampling arms' losses at the primary setting.
        domain: `solar` or `wind`.

    Returns:
        The chosen combination, and one decision per change tried.
    """
    chosen = "linear"
    decisions = []
    for field, technique in CHANGES[domain]:
        candidate = _combination_name(
            domain=domain, choice={**COMBINATIONS[domain][chosen], field: technique}
        )
        changes = {
            day: _mae(losses=losses, arm=upsampling_arm(method=candidate, day=day))
            - _mae(losses=losses, arm=upsampling_arm(method=chosen, day=day))
            for day in DECIDING_DAYS
        }
        adopted = all(change < 0.0 for change in changes.values())
        decisions.append(
            Decision(candidate=candidate, against=chosen, changes=changes, adopted=adopted)
        )
        if adopted:
            chosen = candidate
    return chosen, decisions


def _decision_lines(*, decisions: list[Decision]) -> list[str]:
    """Render the choosing rule's decisions.

    Args:
        decisions: The output of `choose_method`.

    Returns:
        Markdown lines.
    """
    return [
        f"- {decision.candidate} against {decision.against}: "
        + ", ".join(f"day {day} {change:+.3f} pp" for day, change in decision.changes.items())
        + (" — adopted." if decision.adopted else " — not adopted.")
        for decision in decisions
    ]


def _mae(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean error at the primary setting, in percentage points of capacity.

    Args:
        losses: Per-row losses.
        arm: The arm.

    Returns:
        The mean of each row's capped error over its generator's capacity.
    """
    rows = losses.filter((pl.col("arm") == arm) & (pl.col("setting") == "pooled"))
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def main_frame(*, inputs: Inputs, method: MethodType, domain: DomainType) -> Inputs:
    """Add the control, ensemble-mean, and 6-hourly-emulation arms under the chosen combination.

    Args:
        inputs: The technology's inputs.
        method: The chosen combination.
        domain: `solar` or `wind`.

    Returns:
        The inputs with every main arm's columns and, per band, every member's hourly fields under
        the chosen combination, on the same rows.

    Raises:
        ValueError: If a main arm lacks an input on a row every combination covered.
    """
    frame = inputs.frame
    keys = frame.select("site", "time")
    extract = _members(sites=sorted(frame["site"].unique().to_list()))
    clear_sky = _clear_sky_table(domain=domain)
    members: dict[int, pl.DataFrame] = {}
    for day in BAND_DAYS:
        steps = band_steps(members=extract, day=day, domain=domain)
        hourly = combine(
            steps=steps,
            upsampled=_upsampled_fields(steps=steps, day=day, domain=domain, clear_sky=clear_sky),
            day=day,
            domain=domain,
            method=method,
        )
        for way in ("control", "mean"):
            reduced = reduce_members(hourly=hourly, domain=domain, way=way)
            frame = frame.join(
                _prefixed(frame=reduced, arm=ens_arm(way=way, day=day), domain=domain),
                on=["site", "time"],
                how="left",
            )
        members[day] = hourly.join(keys, on=["site", "time"], how="semi")
    emulated = band_steps(members=extract, day=EMULATED_DAY, domain=domain, six_hourly=True)
    emulated_mean = reduce_members(
        hourly=combine(
            steps=emulated,
            upsampled=_upsampled_fields(
                steps=emulated, day=EMULATED_DAY, domain=domain, clear_sky=clear_sky
            ),
            day=EMULATED_DAY,
            domain=domain,
            method=method,
        ),
        domain=domain,
        way="mean",
    )
    frame = frame.join(
        _prefixed(frame=emulated_mean, arm=ens_arm(way="mean6", day=EMULATED_DAY), domain=domain),
        on=["site", "time"],
        how="left",
    )
    added = [
        column
        for day in BAND_DAYS
        for way in ("control", "mean")
        for column in ens_columns(arm=ens_arm(way=way, day=day), domain=domain)
    ]
    added += list(ens_columns(arm=ens_arm(way="mean6", day=EMULATED_DAY), domain=domain))
    missing = frame.select(pl.any_horizontal(pl.col(added).is_null()).sum()).item()
    if missing:
        msg = f"{domain}: {missing} rows lack a main arm's input every combination covered"
        raise ValueError(msg)
    return Inputs(frame=frame, members=members, native=inputs.native, inputs=inputs.inputs)


def fitted_features(*, domain: Domain) -> dict[str, tuple[str, ...]]:
    """Return every arm fitted by `run_all`, with its feature columns.

    Args:
        domain: The domain.

    Returns:
        Arm name to feature columns: each combination's, the control, and the ensemble-mean arm
        at every band, the 6-hourly emulation, and the two references.
    """
    shared = shared_features(domain=domain)
    arms = {}
    for day in BAND_DAYS:
        for method in COMBINATIONS[domain.name]:
            arm = upsampling_arm(method=method, day=day)
            arms[arm] = (*shared, *ens_columns(arm=arm, domain=domain.name))
        for way in ("control", "mean"):
            arm = ens_arm(way=way, day=day)
            arms[arm] = (*shared, *ens_columns(arm=arm, domain=domain.name))
    emulated = ens_arm(way="mean6", day=EMULATED_DAY)
    arms[emulated] = (*shared, *ens_columns(arm=emulated, domain=domain.name))
    return arms | reference_features(domain=domain)


def jobs(*, domain: Domain) -> list[Job]:
    """Return every `run_all` fit and the setting it runs at.

    Every arm runs at the primary setting. The control, the ensemble mean, and the references run
    at the second setting too, and the day-1 ensemble mean without row subsampling.

    Args:
        domain: The domain.

    Returns:
        The jobs.
    """
    features = fitted_features(domain=domain)
    second = {ens_arm(way=way, day=day) for day in BAND_DAYS for way in ("control", "mean")}
    second |= set(REFERENCES)
    fits: list[Job] = [
        (arm, "pooled", "power_mw", columns, SETTINGS["pooled"], False)
        for arm, columns in features.items()
    ]
    fits += [
        (arm, "sensitivity", "power_mw", features[arm], SETTINGS["sensitivity"], False)
        for arm in features
        if arm in second
    ]
    no_subsample = ens_arm(way="mean", day=NO_SUBSAMPLE_DAY)
    fits.append(
        (
            no_subsample,
            "no_subsample",
            "power_mw",
            features[no_subsample],
            SETTINGS["no_subsample"],
            False,
        )
    )
    return fits


# --- Intervals and the report ----------------------------------------------------------------


def best_baselines(*, losses: pl.DataFrame) -> dict[int, str]:
    """Return the baseline with the lowest error at each band, at the primary setting.

    Args:
        losses: Every arm's losses.

    Returns:
        Band to the best baseline's arm name, climatology included at every band.
    """
    return {
        day: min(
            ("climatology", *(baseline_arm(name=name, day=day) for name in BASELINES)),
            key=lambda arm: _mae(losses=losses, arm=arm),
        )
        for day in BAND_DAYS
    }


Contrast = tuple[str, SettingType, str, str]
"""(section, setting, treatment, reference)."""


def _upsampling_contrasts(*, decisions: list[Decision]) -> list[Contrast]:
    """Return each candidate against what it was judged against, and against linear, every band.

    Args:
        decisions: The output of `choose_method`.

    Returns:
        The contrasts.
    """
    wanted: list[Contrast] = []
    for day in BAND_DAYS:
        for decision in decisions:
            arm = upsampling_arm(method=decision.candidate, day=day)
            wanted += [
                ("upsampling", "pooled", arm, upsampling_arm(method=reference, day=day))
                for reference in dict.fromkeys((decision.against, "linear"))
            ]
    return wanted


def _way_contrasts() -> list[Contrast]:
    """Return every way against day 0, and the ways against each other at every band.

    Returns:
        The contrasts.
    """
    wanted: list[Contrast] = [
        ("horizon", "pooled", ens_arm(way=way, day=day), ens_arm(way=way, day=0))
        for way in WAYS
        for day in BAND_DAYS[1:]
    ]
    wanted += [
        ("horizon", "sensitivity", ens_arm(way="mean", day=day), ens_arm(way="mean", day=0))
        for day in BAND_DAYS[1:]
    ]
    pairs = (("members", "mean"), ("mean", "control"), ("members", "control"))
    pairs += (("members_median", "members"),)
    wanted += [
        ("ways", "pooled", ens_arm(way=treatment, day=day), ens_arm(way=reference, day=day))
        for day in BAND_DAYS
        for treatment, reference in pairs
    ]
    wanted += [
        ("ways", "sensitivity", ens_arm(way="members", day=day), ens_arm(way="mean", day=day))
        for day in MEMBER_SENSITIVITY_DAYS
    ]
    wanted.append(
        (
            "ways",
            "no_subsample",
            ens_arm(way="members", day=NO_SUBSAMPLE_DAY),
            ens_arm(way="mean", day=NO_SUBSAMPLE_DAY),
        )
    )
    wanted += [
        ("trained on the mean", "pooled", ens_arm(way=APPLIED, day=day), ens_arm(way=way, day=day))
        for day in BAND_DAYS
        for way in ("mean", "members")
    ]
    return wanted


def contrasts(*, decisions: list[Decision], best: dict[int, str]) -> list[Contrast]:
    """Return every contrast the report prints, in the order it prints them.

    Args:
        decisions: The upsampling rule's decisions.
        best: Each band's best baseline.

    Returns:
        The contrasts, each once, a planned contrast only in the planned section.
    """
    wanted: list[Contrast] = [
        ("planned", setting, treatment, reference)
        for setting in ("pooled", "sensitivity")
        for treatment, reference in PLANNED
    ]
    wanted += _upsampling_contrasts(decisions=decisions) + _way_contrasts()
    wanted += [
        ("baselines", "pooled", ens_arm(way=way, day=day), reference)
        for day in BAND_DAYS
        for way in WAYS
        for reference in dict.fromkeys((best[day], "climatology"))
    ]
    wanted += [
        ("references", "pooled", ens_arm(way="mean", day=day), reference)
        for day in (0, 1)
        for reference in REFERENCES
    ]
    wanted += [
        ("references", "pooled", "live_gb_rich_xgb", "era5"),
        (
            "step width",
            "pooled",
            ens_arm(way="mean6", day=EMULATED_DAY),
            ens_arm(way="mean", day=EMULATED_DAY),
        ),
    ]
    planned = {contrast[1:] for contrast in wanted if contrast[0] == "planned"}
    unique: list[Contrast] = []
    for contrast in wanted:
        if contrast not in unique and (contrast[0] == "planned" or contrast[1:] not in planned):
            unique.append(contrast)
    return unique


def _intervals(
    *, losses: pl.DataFrame, domain: Domain, wanted: list[Contrast]
) -> list[IntervalRecord]:
    """Bootstrap every contrast.

    Args:
        losses: Every arm's losses, both settings.
        domain: The domain.
        wanted: The contrasts.

    Returns:
        One record per contrast.
    """
    by_setting = {setting: losses.filter(pl.col("setting") == setting) for setting in SETTINGS}

    def _one(contrast: Contrast) -> IntervalRecord:
        section, setting, treatment, reference = contrast
        return _interval(
            losses=by_setting[setting].filter(pl.col("arm").is_in([treatment, reference])),
            contrast=(treatment, reference),
            domain=domain,
            setting=setting,
            section=section,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        return list(pool.map(_one, wanted))


def _leaderboard(*, losses: pl.DataFrame, domain: DomainType) -> list[dict[str, object]]:
    """Bootstrap every arm's own error, at every setting it was scored at.

    Args:
        losses: Every arm's losses.
        domain: `solar` or `wind`.

    Returns:
        One record per arm and setting.
    """
    groups = sorted(losses.partition_by("setting", "arm", as_dict=True).items())

    def _one(item: tuple[tuple[object, ...], pl.DataFrame]) -> dict[str, object]:
        (setting, arm), scoped = item
        interval = bootstrap_absolute(losses=scoped, arm=str(arm), metric=METRIC)
        return {
            "domain": domain,
            "setting": setting,
            "arm": arm,
            "mae_pp": interval["value"] * PERCENTAGE_POINTS,
            "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
            "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
            "seed_spread_pp": interval["seed_spread"] * PERCENTAGE_POINTS,
            "n_rows": interval["n_rows"],
            "n_months": interval["n_months"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        return list(pool.map(_one, groups))


def arm_order(*, domain: DomainType) -> list[str]:
    """Return every arm in the order the flat leaderboard prints it.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The native and upsampling arms, then each band's ways, the exploratory arm trained on the
        mean, and the baselines, then climatology, the emulation, and the references.
    """
    order = [
        upsampling_arm(method=method, day=day)
        for day in BAND_DAYS
        for method in ("native", *COMBINATIONS[domain])
    ]
    for day in BAND_DAYS:
        order += [ens_arm(way=way, day=day) for way in (*WAYS, APPLIED)]
        order += [baseline_arm(name=name, day=day) for name in BASELINES]
    order += ["climatology", ens_arm(way="mean6", day=EMULATED_DAY), *REFERENCES]
    return order


def _leaderboard_lines(*, board: list[dict[str, object]], domain: DomainType) -> list[str]:
    """Render one technology's leaderboard, flat and as a horizon table.

    Args:
        board: The records from `_leaderboard`.
        domain: `solar` or `wind`.

    Returns:
        Markdown lines.
    """
    lines = []
    for setting in SETTINGS:
        by_arm = {row["arm"]: row for row in board if row["setting"] == setting}
        lines += [
            f"#### {domain.capitalize()}: every arm's error, {setting} setting",
            "",
            (
                "| Arm | MAE (pp of capacity) | 95% interval, months and seed | Seed spread | Rows "
                "| Months |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for arm in arm_order(domain=domain):
            if arm in by_arm:
                row = by_arm[arm]
                lines.append(
                    f"| {arm} | {row['mae_pp']:.3f} "
                    f"| [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] "
                    f"| {row['seed_spread_pp']:.4f} | {row['n_rows']:,} | {row['n_months']} |"
                )
        lines.append("")
    pooled = {row["arm"]: row["mae_pp"] for row in board if row["setting"] == "pooled"}
    columns = [*WAYS, *BASELINES]
    lines += [
        f"#### {domain.capitalize()}: MAE by band (pp of capacity, main setting)",
        "",
        "| Band | " + " | ".join(columns) + " | climatology |",
        "|---" * (len(columns) + 2) + "|",
    ]
    for day in BAND_DAYS:
        cells = [
            f"{pooled[ens_arm(way=c, day=day) if c in WAYS else baseline_arm(name=c, day=day)]:.3f}"
            for c in columns
        ]
        lines.append(f"| day {day} | " + " | ".join(cells) + f" | {pooled['climatology']:.3f} |")
    return [*lines, ""]


def _contrast_lines(*, records: list[IntervalRecord]) -> list[str]:
    """Render the contrasts, one table per section and setting.

    Args:
        records: Every interval.

    Returns:
        Markdown lines.
    """
    lines = []
    sections = list(dict.fromkeys(record["section"] for record in records))
    for section in sections:
        for setting in SETTINGS:
            chosen = [r for r in records if r["section"] == section and r["setting"] == setting]
            if chosen:
                lines += [
                    f"#### Contrasts: {section}, {setting} setting",
                    "",
                    *CONTRAST_HEADER,
                    *(_line(record) for record in chosen),
                    "",
                ]
    return lines


def _row_lines(*, frame: pl.DataFrame, domain: DomainType) -> list[str]:
    """Describe the rows: how many, which span, and how they split by generator and era.

    Args:
        frame: The rows every arm is scored on.
        domain: `solar` or `wind`.

    Returns:
        Markdown lines.
    """
    by_site = (
        frame.group_by("site", "era")
        .agg(rows=pl.len(), months=pl.col("month").n_unique())
        .sort("site", "era")
    )
    lines = [
        (
            f"### {domain.capitalize()}: {frame.height:,} generator-hours "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d}), "
            f"{frame['month'].n_unique()} months, {int(frame['constrained'].sum()):,} curtailed"
        ),
        "",
        "| Generator | UKV era | Rows | Months |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['site']} | {row['era']} | {row['rows']:,} | {row['months']} |"
        for row in by_site.iter_rows(named=True)
    ]
    return [*lines, ""]


def _feature_lines(*, domain: Domain) -> list[str]:
    """Print every fitted arm's feature columns, so a reviewer can check them against the plan.

    Args:
        domain: The domain.

    Returns:
        Markdown lines.
    """
    features = fitted_features(domain=domain)
    for day in BAND_DAYS:
        arm = ens_arm(way="members", day=day)
        features[arm] = (*shared_features(domain=domain), *ens_columns(arm=arm, domain=domain.name))
    lines = [
        f"#### {domain.name.capitalize()}: every fitted arm's feature columns",
        "",
        "| Arm | Columns | Features |",
        "|---|---|---|",
    ]
    lines += [
        f"| {arm} | {len(columns)} | {', '.join(f'`{column}`' for column in columns)} |"
        for arm, columns in features.items()
    ]
    return [*lines, ""]


def _per_site_lines(*, losses: pl.DataFrame, domain: DomainType) -> list[str]:
    """Render the main arms' and the best-known baselines' error at each generator.

    Args:
        losses: Every arm's losses.
        domain: `solar` or `wind`.

    Returns:
        Markdown lines.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sites = sorted(pooled["site"].unique().to_list())
    table = (
        pooled.group_by("arm", "site")
        .agg(mae=pl.col(METRIC).mean() * PERCENTAGE_POINTS)
        .pivot(on="site", index="arm", values="mae")
    )
    by_arm = {row["arm"]: row for row in table.iter_rows(named=True)}
    arms = [a for a in arm_order(domain=domain) if a in by_arm and not a.startswith("up_")]
    lines = [
        f"#### {domain.capitalize()}: MAE by generator (pp of capacity, main setting)",
        "",
        "| Arm | " + " | ".join(sites) + " |",
        "|---" * (len(sites) + 1) + "|",
    ]
    lines += [
        f"| {arm} | " + " | ".join(f"{by_arm[arm][site]:.3f}" for site in sites) + " |"
        for arm in arms
    ]
    return [*lines, ""]


def _spread_lines(*, summary: pl.DataFrame, frame: pl.DataFrame) -> list[str]:
    """Describe the 51 member forecasts: their spread, and how often their 10–90% range holds truth.

    Args:
        summary: The member summaries from `_fit_members`, primary setting.
        frame: The rows, with `power_mw` and `effective_capacity_mw`.

    Returns:
        Markdown lines.
    """
    joined = summary.filter(pl.col("setting") == "pooled").join(
        frame.select("site", "time", "power_mw", "effective_capacity_mw"), on=["site", "time"]
    )
    table = (
        joined.group_by("arm")
        .agg(
            spread_pp=(pl.col("spread") / pl.col("effective_capacity_mw")).mean()
            * PERCENTAGE_POINTS,
            inside=(
                (pl.col("power_mw") >= pl.col("p10")) & (pl.col("power_mw") <= pl.col("p90"))
            ).mean(),
        )
        .with_columns(day=pl.col("arm").str.extract(r"day(\d+)$").cast(pl.Int32))
        .sort("day")
    )
    lines = [
        "#### Member-by-member forecasts: spread, and the 10th-to-90th-percentile range",
        "",
        "| Band | Mean standard deviation of the 51 forecasts (pp) | Share of hours inside |",
        "|---|---|---|",
    ]
    lines += [
        f"| day {row['day']} | {row['spread_pp']:.3f} | {row['inside']:.3f} |"
        for row in table.iter_rows(named=True)
    ]
    return [*lines, ""]


def _weight_lines(*, weights: pl.DataFrame) -> list[str]:
    """Describe the wind smart persistence's fitted weight on persistence, per band.

    Args:
        weights: One row per generator, fold, and band.

    Returns:
        Markdown lines.
    """
    table = (
        weights.group_by("day")
        .agg(
            mean=pl.col("weight").mean(),
            lowest=pl.col("weight").min(),
            highest=pl.col("weight").max(),
        )
        .sort("day")
    )
    lines = [
        "#### Wind smart persistence: the weight on persistence, over generators and folds",
        "",
        "| Band | Mean | Lowest | Highest |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| day {row['day']} | {row['mean']:.2f} | {row['lowest']:.2f} | {row['highest']:.2f} |"
        for row in table.iter_rows(named=True)
    ]
    return [*lines, ""]


def check_shared_rows(*, losses: pl.DataFrame) -> None:
    """Stop unless every arm but the native ones scores the same (site, time, seed) rows.

    A contrast pairs its two arms by an inner join on those keys, so an arm missing rows would
    silently shrink every contrast it enters rather than fail.

    Args:
        losses: Every arm's losses.

    Raises:
        ValueError: If two arms at one setting score different rows.
    """
    hourly = losses.filter(~pl.col("arm").str.starts_with("up_native_"))
    reference = hourly.filter(
        (pl.col("arm") == "climatology") & (pl.col("setting") == "pooled")
    ).select("site", "time", "seed")
    for (arm, setting), rows in hourly.partition_by("arm", "setting", as_dict=True).items():
        keys = rows.select("site", "time", "seed")
        if keys.height != reference.height or not keys.sort(pl.all()).equals(
            reference.sort(pl.all())
        ):
            msg = f"{arm} at {setting} scores {keys.height:,} rows, not {reference.height:,}"
            raise ValueError(msg)


# --- Running ------------------------------------------------------------------------------------


@dataclass
class Outputs:
    """Everything one technology's run produced."""

    frame: pl.DataFrame
    losses: pl.DataFrame
    summary: pl.DataFrame
    weights: pl.DataFrame
    method: MethodType
    decisions: list[Decision]


def _paths(*, domain: DomainType) -> dict[str, Path]:
    """Return where one technology's outputs are written.

    Args:
        domain: `solar` or `wind`.

    Returns:
        Output name to path.
    """
    return {
        name: OUTPUT_DIR / f"{domain}_{name}.parquet"
        for name in ("rows", "losses", "member_summary", "weights", "inputs", "predictions")
    }


def run_domain(*, domain: Domain, report_only: bool) -> Outputs:
    """Build one technology's inputs, choose the upsampling, fit every arm, and score the baselines.

    Args:
        domain: The domain.
        report_only: Whether to read the saved outputs instead of fitting.

    Returns:
        The outputs.
    """
    paths = _paths(domain=domain.name)
    if report_only:
        losses = pl.read_parquet(paths["losses"])
        method, decisions = choose_method(losses=losses, domain=domain.name)
        return Outputs(
            frame=pl.read_parquet(paths["rows"]),
            losses=losses,
            summary=pl.read_parquet(paths["member_summary"]),
            weights=pl.read_parquet(paths["weights"]),
            method=method,
            decisions=decisions,
        )
    inputs = build_inputs(domain=domain.name)
    inputs.inputs.join(
        inputs.frame.select("site", "time", "power_mw", "effective_capacity_mw"),
        on=["site", "time"],
        how="left",
    ).write_parquet(paths["inputs"])
    upsampling_jobs: list[Job] = [
        job for job in jobs(domain=domain) if job[0].startswith("up_") and job[1] == "pooled"
    ]
    upsampling_losses = pl.concat(
        [
            run_all(dataset=inputs.frame, jobs=upsampling_jobs),
            _native_losses(inputs=inputs, domain=domain),
        ],
        how="diagonal",
    )
    method, decisions = choose_method(losses=upsampling_losses, domain=domain.name)
    _LOG.info(
        "%s: chose %s\n%s", domain.name, method, "\n".join(_decision_lines(decisions=decisions))
    )
    inputs = main_frame(inputs=inputs, method=method, domain=domain.name)
    frame = inputs.frame
    main_losses = run_all(
        dataset=frame, jobs=[job for job in jobs(domain=domain) if not job[0].startswith("up_")]
    )
    member_losses, summaries = [], []
    for day in BAND_DAYS:
        settings: list[SettingType] = ["pooled"]
        settings += ["sensitivity"] if day in MEMBER_SENSITIVITY_DAYS else []
        settings += ["no_subsample"] if day == NO_SUBSAMPLE_DAY else []
        for setting in settings:
            losses, summary = _fit_members(
                frame=frame, members=inputs.members[day], day=day, domain=domain, setting=setting
            )
            member_losses.append(losses)
            summaries.append(summary)
            _LOG.info("%s day %d %s: member-by-member arm fitted", domain.name, day, setting)
        member_losses.append(
            _fit_mean_applied(frame=frame, members=inputs.members[day], day=day, domain=domain)
        )
    baselines, weights = _baseline_losses(frame=frame, domain=domain.name)
    keep = [
        "site",
        "time",
        "month",
        "fold",
        "seed",
        "arm",
        "setting",
        METRIC,
        "signed_error_capped_mw",
    ]
    losses = pl.concat(
        [
            frame_losses.select(keep)
            for frame_losses in (upsampling_losses, main_losses, *member_losses, baselines)
        ]
    )
    summary = pl.concat(summaries)
    check_shared_rows(losses=losses)
    frame.select(
        "site",
        "time",
        "month",
        "fold",
        "era",
        "power_mw",
        "effective_capacity_mw",
        "cap_mw",
        "constrained",
    ).write_parquet(paths["rows"])
    losses.write_parquet(paths["losses"])
    summary.write_parquet(paths["member_summary"])
    weights.write_parquet(paths["weights"])
    losses.join(frame.select("site", "time", "power_mw"), on=["site", "time"], how="left").select(
        "site",
        "time",
        "arm",
        "setting",
        "seed",
        "power_mw",
        prediction_mw=pl.col("signed_error_capped_mw") + pl.col("power_mw"),
    ).write_parquet(paths["predictions"])
    return Outputs(
        frame=frame,
        losses=losses,
        summary=summary,
        weights=weights,
        method=method,
        decisions=decisions,
    )


def main() -> int:
    """Run both technologies, bootstrap everything, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from the outputs already on disk instead of refitting.",
    )
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# ENS forecast error by horizon", ""]
    records: list[IntervalRecord] = []
    boards: list[dict[str, object]] = []
    for domain in (SOLAR, WIND):
        outputs = run_domain(domain=domain, report_only=arguments.report_only)
        best = best_baselines(losses=outputs.losses)
        domain_records = _intervals(
            losses=outputs.losses,
            domain=domain,
            wanted=contrasts(decisions=outputs.decisions, best=best),
        )
        board = _leaderboard(losses=outputs.losses, domain=domain.name)
        records += domain_records
        boards += board
        lines += [
            *_row_lines(frame=outputs.frame, domain=domain.name),
            f"#### {domain.name.capitalize()}: the upsampling technique chosen: {outputs.method}",
            "",
            *_decision_lines(decisions=outputs.decisions),
            "",
            "Best baseline at each band: "
            + ", ".join(f"day {day} {arm}" for day, arm in best.items())
            + ".",
            "",
            *_feature_lines(domain=domain),
            *_leaderboard_lines(board=board, domain=domain.name),
            *_contrast_lines(records=domain_records),
            *_spread_lines(summary=outputs.summary, frame=outputs.frame),
            *(_weight_lines(weights=outputs.weights) if domain.name == "wind" else []),
            *_per_site_lines(losses=outputs.losses, domain=domain.name),
        ]
    pl.DataFrame(records).write_parquet(OUTPUT_DIR / "intervals.parquet")
    pl.DataFrame(boards).write_parquet(OUTPUT_DIR / "leaderboard.parquet")
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
