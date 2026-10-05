"""Draw the UKV-CEDA blends study's figures from the saved intervals, losses, and predictions.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. It fits nothing. It reads
`intervals.parquet` and the saved per-row losses and predictions that `fit_ukv_ceda_blends.py`
wrote, and writes SVG files into a new `--figures-dir`, each written once.

- `<domain>_headline.svg`: one panel per lead day, with P1 (the blend minus the padded ENS arm) and
  P2 (the blend minus each shuffled control) as dots with 95% intervals, at both hyperparameter
  settings. The filled dot is the primary setting and the lighter hollow mark is the sensitivity
  setting. The title and each panel's title state the reading that
  `fit_ukv_ceda_blends.reading` gives from the saved intervals.
- `<domain>_generators.svg`: P1 at the primary setting, for each generator alone, one panel per
  lead day. This figure is exploratory.
- `<domain>_errors.svg`: each XGBoost model's own mean absolute error with its 95% interval, one
  panel per lead day, at both settings, read from the `error` rows of `intervals.parquet`.
- `<domain>_week<k>.svg`: measured output and the day-1 blend's out-of-fold forecast for each
  generator over one week of era `k - 1`. The week is chosen by `nwp_forecast_charts.choose_week`
  from measured output alone, and the axis counts days 1 to 7, so no figure carries a calendar date.
  The months and years of the weeks are printed for the page's text.

Two post hoc figures read the rows a later report adds (`--post-hoc-only` draws only these):

- `solar_permutation.svg`: for each solar lead day, the 17 shuffled controls' differences from
  padded ENS as grey ticks and the planned blend's P1 as a marker, with its rank and permutation
  p-value.
- `<domain>_older_run.svg`: for each lead day 1 to 3, the planned blend's P1 on the older run's
  rows, the older-run blend's P1, and its P2 against its shuffled control, at both settings.

Every chart refuses a site label that is not one of the technology's anonymised labels, and every
mark has accessibility text turned off, because Vega would otherwise write each point's value into
the SVG.

Run it with `uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py --figures-dir DIR`.
"""

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

_STUDIES_DIR: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES_DIR / "ukv_ceda_blends"))
sys.path.insert(0, str(_STUDIES_DIR / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDIES_DIR / "beam_diffuse_split"))
sys.path.insert(0, str(_STUDIES_DIR / "weather_downloads"))

import build_ukv_ceda_inputs as build  # noqa: E402
import fit_aifs  # noqa: E402
import fit_ukv_ceda_blends as fit  # noqa: E402
from nwp_forecast_charts import (  # noqa: E402
    CAPACITY_NOTE,
    DIFFERENCE_TITLE,
    FORECAST_COLOUR,
    MEASURED_COLOUR,
    SITES,
    TECHNOLOGY_NAMES,
    TIME_PANEL_HEIGHT_PX,
    check_anonymised,
    choose_week,
    line_key,
    measured_and_forecast,
    padded_domain,
)
from nwp_forecast_comparison import (  # noqa: E402
    NWP_ERA_START_MONTHS,
    PERCENTAGE_POINTS,
    DomainType,
)
from paths import REPO_DATA_DIR  # noqa: E402
from studies.bootstrap import BootstrapInterval  # noqa: E402
from studies.charts import (  # noqa: E402
    ABSOLUTE_ERROR_X_TITLE,
    CONTENT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
)
from studies.guards import refuse_to_overwrite  # noqa: E402

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

FIGURE_NUMBERS: Final[dict[tuple[DomainType, str], int]] = {
    ("solar", "headline"): 1,
    ("wind", "headline"): 2,
    ("solar", "generators"): 3,
    ("wind", "generators"): 4,
    ("solar", "errors"): 5,
    ("wind", "errors"): 6,
    ("solar", "weeks"): 7,
    ("wind", "weeks"): 8,
}
"""The figure's number on the page. A week figure is captioned with its number and a letter."""

PRIMARY_LABEL: Final[str] = "Primary setting"
SENSITIVITY_LABEL: Final[str] = "Sensitivity setting"
SETTING_LABELS: Final[dict[str, str]] = {
    fit.PRIMARY: PRIMARY_LABEL,
    fit.SENSITIVITY: SENSITIVITY_LABEL,
}
"""Each hyperparameter setting's name in the key, by its code in `intervals.parquet`."""

READING_LABELS: Final[dict[str, str]] = {
    "lowers": "planned rule met",
    fit.UNRESOLVED_LOWER: "unresolved (lower than padded ENS, control test not passed)",
    fit.NO_DETECTABLE_DIFFERENCE: "inconclusive (a gain is not excluded)",
    "raises": "blend raises the error",
}
"""A reading's panel-title label, where the key `lowers` or `raises` stands for the verdict
`lowers the error at day N` or `raises the error at day N`."""

TITLES: Final[dict[DomainType, str]] = {
    "solar": (
        "For six solar farms, adding UKV-CEDA lowered the ENS mean's error by about 0.1 points of "
        "capacity at lead days 1 to 3, and day 4 is inconclusive"
    ),
    "wind": (
        "For three wind farms, adding UKV-CEDA's winds lowered the ENS mean's error at lead days 1 "
        "and 2 under every check, day 3 rests on February 2026, and day 4 is inconclusive"
    ),
}
"""Each headline figure's title, written by hand after reading the intervals, the Bonferroni
correction, the leave-one-month-out table, and the control gap. A title generated from the planned
rule alone would state a verdict per lead day that those checks do not support. The numbers behind
a title are machine-printed in the subtitle."""

ARM_LABELS: Final[dict[fit_aifs.BlendRoleType, str]] = {
    "_pad": "ENS mean, padded to the same column count",
    "": "ENS mean + UKV-CEDA",
    "_control": "ENS mean + shuffled UKV-CEDA, seed 0",
    "_control_b": "ENS mean + shuffled UKV-CEDA, seed 1000",
}
"""The error figure's row label of each arm, by its role."""

P1_LABEL: Final[str] = "Blend minus padded ENS"
P2_LABEL: Final[str] = "Blend minus shuffled UKV-CEDA"
P2_SECOND_LABEL: Final[str] = "Blend minus shuffled UKV-CEDA, second seed"

CONTRAST_LABELS: Final[dict[str, str]] = {
    "p1": P1_LABEL,
    "p2": P2_LABEL,
    "p2b": P2_SECOND_LABEL,
}
"""Each planned contrast's row label, by its code in `intervals.parquet`."""

AXIS_TITLE: Final[str] = "Error minus padded ENS's (points of capacity; more negative is better)"

POST_HOC_FIGURE_NUMBERS: Final[dict[tuple[DomainType, str], int]] = {
    ("solar", "permutation"): 9,
    ("solar", "older"): 10,
    ("wind", "older"): 11,
}
"""The post hoc figures' numbers, after the eight of `FIGURE_NUMBERS`."""

OLDER_LABELS: Final[dict[str, str]] = {
    "fresh_p1_same_rows": "Planned blend minus padded ENS, same rows",
    "older_p1": "Older-run blend minus its padded ENS",
    "older_p2": "Older-run blend minus its shuffled control",
}
"""The older-run figure's row labels, by contrast code in `intervals.parquet`."""

FAMILY: Final[str] = "weather model"
"""Every row is the same kind of comparison, so every row takes the one family's colour."""

DAYS_ON_AXIS: Final[int] = 7
WEEKS_PER_TECHNOLOGY: Final[int] = len(NWP_ERA_START_MONTHS) + 1

BLEND_ARM_DAY1: Final[str] = fit.arm_name(day=1, role="")
"""The arm whose out-of-fold forecast the week figures draw."""

SCALE_NOTE: Final[str] = (
    "UKV-CEDA's lead is 3 hours fresher than ENS's at every hour, which favours the blend."
)


def scope_note(*, intervals: pl.DataFrame, domain: DomainType) -> str:
    """Name the generators and the span of rows an interval figure rests on.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        A sentence naming the technology's generators and the largest month count of any interval.
    """
    months = int(np.max(intervals.filter(pl.col("domain") == domain)["n_months"].to_numpy()))
    return (
        f"{TECHNOLOGY_NAMES[domain].capitalize()}, up to {months} calendar months of "
        "out-of-fold forecasts."
    )


def contrast_interval(
    *,
    intervals: pl.DataFrame,
    domain: DomainType,
    day: int,
    setting: str,
    contrast: str,
    scope: str = "all rows",
) -> BootstrapInterval:
    """Return one saved interval as the `BootstrapInterval` the reading rule takes.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.
        day: The lead day.
        setting: `primary` or `sensitivity`.
        contrast: The contrast's code: `p1`, `p2`, or `p2b`.
        scope: The row set the interval is of.

    Returns:
        The interval, with the fitting-seed spread, which the reading rule does not use, as zero.
    """
    row = intervals.filter(
        pl.col("domain") == domain,
        pl.col("day") == day,
        pl.col("setting") == setting,
        pl.col("contrast") == contrast,
        pl.col("scope") == scope,
    ).row(0, named=True)
    return {
        "difference": row["difference"],
        "lower_95": row["lower"],
        "upper_95": row["upper"],
        "seed_spread": 0.0,
        "n_rows": row["n_rows"],
        "n_months": row["n_months"],
    }


def day_reading(*, intervals: pl.DataFrame, domain: DomainType, day: int) -> str:
    """Return the reading the planned rule gives one lead day, from the saved intervals.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        `fit_ukv_ceda_blends.reading`'s verdict.
    """
    settings = (fit.PRIMARY, fit.SENSITIVITY)

    def of(setting: str, contrast: str) -> BootstrapInterval:
        return contrast_interval(
            intervals=intervals, domain=domain, day=day, setting=setting, contrast=contrast
        )

    return fit.reading(
        day=day,
        p1={s: of(s, "p1") for s in settings},
        p2={s: [of(s, "p2"), of(s, "p2b")] for s in settings},
    )


def reading_kind(*, reading: str) -> str:
    """Return the key of `READING_LABELS` that a reading falls under."""
    if reading.startswith("lowers"):
        return "lowers"
    if reading.startswith("raises"):
        return "raises"
    return reading


def days_text(*, days: Sequence[int]) -> str:
    """Name lead days in words: `day 4`, `days 1 and 2`, `days 1, 2, and 3`."""
    if len(days) == 1:
        return f"day {days[0]}"
    names = [str(day) for day in days]
    joined = " and ".join(names) if len(names) == 2 else f"{', '.join(names[:-1])}, and {names[-1]}"
    return f"days {joined}"


def open_gain_note(*, intervals: pl.DataFrame, domain: DomainType) -> str:
    """State the gain P1 leaves open at each lead day whose reading is inconclusive.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        A sentence giving, per such day and setting, the largest gain P1's interval does not
        exclude, in points of capacity, or an empty string if no day is inconclusive.
    """
    parts = []
    for day in build.LEAD_DAYS:
        if day_reading(intervals=intervals, domain=domain, day=day) != fit.NO_DETECTABLE_DIFFERENCE:
            continue
        p1 = {
            setting: contrast_interval(
                intervals=intervals, domain=domain, day=day, setting=setting, contrast="p1"
            )
            for setting in (fit.PRIMARY, fit.SENSITIVITY)
        }
        parts.append(f"day {day}: {fit.left_open_text(p1=p1)}")
    if not parts:
        return ""
    return (
        "Largest gain P1's interval does not exclude where the reading is inconclusive, in points "
        "of capacity: " + "; ".join(parts) + "."
    )


def bonferroni_note(*, intervals: pl.DataFrame, domain: DomainType) -> str:
    """Say at which lead days P1 stays below zero after the Bonferroni correction at both settings.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        A sentence naming the lead days, or saying there are none.
    """
    surviving = []
    for day in build.LEAD_DAYS:
        wide = {}
        for setting in (fit.PRIMARY, fit.SENSITIVITY):
            interval = contrast_interval(
                intervals=intervals,
                domain=domain,
                day=day,
                setting=setting,
                contrast="p1",
                scope="Bonferroni",
            )
            wide[setting] = (interval["lower_95"], interval["upper_95"])
        if fit.survives_bonferroni(wide=wide):
            surviving.append(day)
    where = days_text(days=surviving) if surviving else "no lead day"
    return (
        f"After the Bonferroni correction across the {fit.N_P1_INTERVALS} P1 intervals per "
        f"setting, P1 stays below zero at both settings at {where}."
    )


def scale_of(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Convert saved `difference`, `lower`, and `upper` fractions of capacity to points."""
    return frame.with_columns(pl.col("difference", "lower", "upper") * PERCENTAGE_POINTS)


def headline_rows(*, intervals: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Shape one lead day's planned contrasts for `interval_panel`.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        One row per planned contrast and setting, P1 first, in points of capacity. The row's
        `condition` is the setting's name, so the panel draws the sensitivity setting's own
        interval beside the primary setting's.
    """
    scope = scale_of(
        frame=intervals.filter(
            pl.col("domain") == domain, pl.col("day") == day, pl.col("scope") == "all rows"
        )
    ).filter(pl.col("contrast").is_in(list(CONTRAST_LABELS)))
    order = {code: index for index, code in enumerate(CONTRAST_LABELS)}
    setting_order = {setting: index for index, setting in enumerate(SETTING_LABELS)}
    return (
        scope.with_columns(
            order=pl.col("contrast").replace_strict(order, return_dtype=pl.Int8),
            setting_order=pl.col("setting").replace_strict(setting_order, return_dtype=pl.Int8),
        )
        .sort("order", "setting_order")
        .select(
            label=pl.col("contrast").replace_strict(CONTRAST_LABELS, return_dtype=pl.String),
            family=pl.lit(FAMILY),
            planned=pl.lit(value=True),
            condition=pl.col("setting").replace_strict(SETTING_LABELS, return_dtype=pl.String),
            difference=pl.col("difference"),
            lower_95=pl.col("lower"),
            upper_95=pl.col("upper"),
        )
    )


def x_domain_of(*, rows: Sequence[pl.DataFrame]) -> tuple[float, float]:
    """Return one padded x range, including zero, that holds every mark and interval of `rows`.

    Args:
        rows: Frames carrying `lower_95` and `upper_95`, and optionally `second_difference`.

    Returns:
        The range.
    """
    columns = [
        name for name in ("lower_95", "upper_95", "second_difference") if name in rows[0].columns
    ]
    values = np.concatenate([frame.select(columns).to_numpy().ravel() for frame in rows])
    return padded_domain(
        low=float(np.nanmin(values)), high=float(np.nanmax(values)), include_zero=True
    )


def headline(*, intervals: pl.DataFrame, domain: DomainType) -> alt.VConcatChart:
    """Draw one technology's planned contrasts: one panel per lead day, on one shared axis.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        The figure, titled with the reading the planned rule gives each lead day.
    """
    rows = {
        day: headline_rows(intervals=intervals, domain=domain, day=day) for day in build.LEAD_DAYS
    }
    shared = x_domain_of(rows=list(rows.values()))
    kinds = {
        day: reading_kind(reading=day_reading(intervals=intervals, domain=domain, day=day))
        for day in build.LEAD_DAYS
    }
    panels = [
        interval_panel(
            rows=day_rows,
            x_domain=shared,
            x_title=DIFFERENCE_TITLE if day == build.LEAD_DAYS[-1] else "",
            zero_label="same error",
            better_label="blend better",
            panel_title=f"Lead day {day}: {READING_LABELS[kinds[day]]}",
            reference_labels=index == 0,
            family_key=False,
            conditions=list(SETTING_LABELS.values()),
            condition_title="Hyperparameter setting",
            condition_key=index == 0,
            figure_planning="planned",
            colour_by_family=True,
        )
        for index, (day, day_rows) in enumerate(rows.items())
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS[(domain, "headline")],
        title=TITLES[domain],
        subtitle=[
            (
                "Difference in mean absolute error between two XGBoost models, first arm minus "
                "second, in points of capacity. Negative means the blend forecasts better. "
                "The blend lowers the error only if its interval and both shuffled-control "
                "intervals are below zero at both settings. Filled dot: primary setting. Lighter "
                "hollow mark: sensitivity setting. Line: 95% interval from resampling whole "
                "months and a fitting seed, at each setting."
            ),
            bonferroni_note(intervals=intervals, domain=domain),
            *filter(None, [open_gain_note(intervals=intervals, domain=domain)]),
            f"{scope_note(intervals=intervals, domain=domain)} {CAPACITY_NOTE} {SCALE_NOTE}",
        ],
        figure_planning="planned",
    )


def arm_error_rows(*, intervals: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Shape one lead day's arms, each with its own absolute error, for `leaderboard_panel`.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        One row per arm and setting, in points of capacity, best primary-setting error first. A
        sensitivity-setting row is a `reference` row, which the panel draws as a lighter hollow
        mark.
    """
    arms = {fit.arm_name(day=day, role=role): label for role, label in ARM_LABELS.items()}
    errors = scale_of(
        frame=intervals.filter(
            pl.col("domain") == domain,
            pl.col("day") == day,
            pl.col("contrast") == "error",
            pl.col("scope").is_in(list(arms)),
        )
    )
    best_first = (
        errors.filter(pl.col("setting") == fit.PRIMARY).sort("difference")["scope"].to_list()
    )
    rank = {arm: index for index, arm in enumerate(best_first)}
    setting_order = {setting: index for index, setting in enumerate(SETTING_LABELS)}
    return (
        errors.with_columns(
            rank=pl.col("scope").replace_strict(rank, return_dtype=pl.Int8),
            setting_order=pl.col("setting").replace_strict(setting_order, return_dtype=pl.Int8),
        )
        .sort("rank", "setting_order")
        .select(
            label=pl.concat_str(
                pl.col("scope").replace_strict(arms, return_dtype=pl.String),
                pl.col("setting").replace_strict(
                    {fit.PRIMARY: ", primary setting", fit.SENSITIVITY: ", sensitivity setting"},
                    return_dtype=pl.String,
                ),
            ),
            family=pl.lit(FAMILY),
            reference=pl.col("setting") == fit.SENSITIVITY,
            value=pl.col("difference"),
            lower_95=pl.col("lower"),
            upper_95=pl.col("upper"),
        )
    )


def errors_title(*, intervals: pl.DataFrame, domain: DomainType) -> str:
    """State the padded ENS model's error at lead day 1 and lead day 4, from the saved intervals."""
    padded = {
        day: float(
            arm_error_rows(intervals=intervals, domain=domain, day=day).filter(
                pl.col("label") == f"{ARM_LABELS['_pad']}, primary setting"
            )["value"][0]
        )
        for day in (build.LEAD_DAYS[0], build.LEAD_DAYS[-1])
    }
    return (
        f"For {TECHNOLOGY_NAMES[domain]}, an XGBoost model given ENS's mean alone has a mean "
        f"absolute error of {padded[build.LEAD_DAYS[0]]:.1f}% of capacity at lead day "
        f"{build.LEAD_DAYS[0]} and {padded[build.LEAD_DAYS[-1]]:.1f}% at lead day "
        f"{build.LEAD_DAYS[-1]}"
    )


def errors(*, intervals: pl.DataFrame, domain: DomainType) -> alt.VConcatChart:
    """Draw every arm's own absolute error at every lead day: one panel per lead day.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        The figure.
    """
    rows = {
        day: arm_error_rows(intervals=intervals, domain=domain, day=day) for day in build.LEAD_DAYS
    }
    shared = x_domain_of(rows=list(rows.values()))
    panels = [
        leaderboard_panel(
            rows=day_rows,
            x_domain=(0.0, shared[1]),
            x_title=ABSOLUTE_ERROR_X_TITLE if day == build.LEAD_DAYS[-1] else "",
            panel_title=f"Lead day {day}",
            keys=False,
        )
        for index, (day, day_rows) in enumerate(rows.items())
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS[(domain, "errors")],
        title=errors_title(intervals=intervals, domain=domain),
        subtitle=[
            (
                "Mean absolute error of each XGBoost model, in percent of capacity, with its 95% "
                "interval from resampling whole months and a fitting seed. The arms share their "
                "rows, so these intervals are wider than the paired differences in the headline "
                "figure, which cancel the month-to-month swing every arm shares. Each arm has two "
                "rows: the primary setting, and the sensitivity setting as a lighter hollow row. "
                "Arms are sorted best first at the primary setting."
            ),
            f"{scope_note(intervals=intervals, domain=domain)} {CAPACITY_NOTE}",
        ],
        figure_planning=None,
    )


def generator_rows(*, intervals: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Shape P1 at each generator alone, at one lead day, for `interval_panel`.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        One row per generator in label order, in points of capacity.
    """
    prefix = "E5 generator "
    scope = intervals.filter(
        pl.col("domain") == domain,
        pl.col("day") == day,
        pl.col("setting") == fit.PRIMARY,
        pl.col("scope").str.starts_with(prefix),
    )
    check_anonymised(
        frame=scope.select(site=pl.col("scope").str.replace(prefix, "", literal=True)),
        domain=domain,
    )
    return scope.sort("scope").select(
        label=pl.col("scope").str.replace("E5 generator", "Generator", literal=True),
        family=pl.lit(FAMILY),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower_95=pl.col("lower") * PERCENTAGE_POINTS,
        upper_95=pl.col("upper") * PERCENTAGE_POINTS,
    )


def generators(*, intervals: pl.DataFrame, domain: DomainType) -> alt.VConcatChart:
    """Draw P1 at each generator alone, one panel per lead day, on one shared axis.

    Args:
        intervals: `intervals.parquet`'s rows.
        domain: `solar` or `wind`.

    Returns:
        The figure.
    """
    rows = {
        day: generator_rows(intervals=intervals, domain=domain, day=day) for day in build.LEAD_DAYS
    }
    shared = x_domain_of(rows=list(rows.values()))
    panels = [
        interval_panel(
            rows=day_rows,
            x_domain=shared,
            x_title=DIFFERENCE_TITLE if day == build.LEAD_DAYS[-1] else "",
            zero_label="same error",
            better_label="blend better",
            panel_title=f"Lead day {day}",
            reference_labels=index == 0,
            family_key=False,
            figure_planning="exploratory",
            colour_by_family=True,
        )
        for index, (day, day_rows) in enumerate(rows.items())
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS[(domain, "generators")],
        title=(
            f"At each of {TECHNOLOGY_NAMES[domain]}, adding UKV-CEDA to the ENS mean "
            "changes the error by a different amount"
        ),
        subtitle=[
            (
                "Difference in mean absolute error between the blend and ENS's mean padded to the "
                "blend's column count, blend minus padded ENS, in points of capacity, at each "
                "generator alone and the primary setting. Negative means the blend forecasts "
                "better. The 95% interval resamples whole months and a fitting seed within one "
                "generator, so it does not cover differences between generators."
            ),
            f"{scope_note(intervals=intervals, domain=domain)} {CAPACITY_NOTE} {SCALE_NOTE}",
        ],
        figure_planning="exploratory",
    )


def permutation_values(
    *, intervals: pl.DataFrame, day: int
) -> tuple[list[float], dict[str, float]]:
    """Return one solar lead day's shuffled-control differences and its P1, rank, and p-value.

    Args:
        intervals: A report's `*_intervals.parquet` rows, holding the permutation test.
        day: The lead day.

    Returns:
        The 17 controls' differences from padded ENS in points of capacity, and `p1`, `rank`, and
        `p_value` for the planned blend.
    """
    rows = intervals.filter(
        pl.col("domain") == "solar",
        pl.col("day") == day,
        pl.col("scope").str.starts_with(fit.PERMUTATION_SCOPE),
    )
    draws = rows.filter(pl.col("contrast") == "permutation_draw")["difference"].to_list()
    single = {
        row["contrast"]: row["difference"]
        for row in rows.filter(pl.col("contrast") != "permutation_draw").iter_rows(named=True)
    }
    return [value * PERCENTAGE_POINTS for value in draws], {
        "p1": single["permutation_p1"] * PERCENTAGE_POINTS,
        "rank": single["permutation_rank"],
        "p_value": single["permutation_p"],
    }


def permutation_figure(*, intervals: pl.DataFrame) -> alt.VConcatChart:
    """Draw the solar permutation test: one panel per lead day on one shared axis.

    Args:
        intervals: A report's `*_intervals.parquet` rows, holding the permutation test.

    Returns:
        The figure, titled from the ranks the intervals hold.
    """
    days = [
        day
        for day in build.LEAD_DAYS
        if not intervals.filter(
            pl.col("day") == day, pl.col("contrast") == "permutation_p", pl.col("domain") == "solar"
        ).is_empty()
    ]
    values = {day: permutation_values(intervals=intervals, day=day) for day in days}
    low, high = padded_domain(
        low=min(min([*draws, single["p1"]]) for draws, single in values.values()),
        high=max(max([*draws, single["p1"]]) for draws, single in values.values()),
        include_zero=True,
    )
    panels = []
    for index, day in enumerate(days):
        draws, single = values[day]
        last = index == len(days) - 1
        title = (
            f"Lead day {day}: P1 {single['p1']:+.3f} points, rank {int(single['rank'])} of "
            f"{len(draws) + 1} from the lowest, permutation p-value {single['p_value']:.3f}"
        )
        controls = pl.DataFrame({"x": draws, "kind": ["Shuffled control"] * len(draws)})
        planned = pl.DataFrame({"x": [single["p1"]], "kind": ["Planned blend (P1)"]})
        scale = alt.Scale(
            domain=["Shuffled control", "Planned blend (P1)"],
            range=[ocf.ENSEMBLE_LINE, ocf.DATA_BLUE],
        )
        x = alt.X(
            "x:Q",
            scale=alt.Scale(domain=[low, high], nice=False),
            title=AXIS_TITLE if last else None,
        )
        colour = alt.Color("kind:N", scale=scale, legend=None)
        ticks = (
            alt.Chart(controls)
            .mark_tick(thickness=2.5, size=34, aria=False)
            .encode(x=x, color=colour)  # ty: ignore[unresolved-attribute]
        )
        marker = (
            alt.Chart(planned)
            .mark_point(shape="diamond", size=220, filled=True, aria=False)
            .encode(x=x, color=colour)  # ty: ignore[unresolved-attribute]
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panel = (ticks + marker + zero).properties(
            width=CONTENT_WIDTH_PX - 120,
            height=44,
            title=alt.TitleParams(title, anchor="start", frame="group"),
        )
        panels.append(panel)
    rank_one = [day for day in days if int(values[day][1]["rank"]) == 1]
    where = days_text(days=rank_one) if rank_one else "no lead day"
    return figure(
        panels=[
            line_key(
                labels=["Shuffled control", "Planned blend (P1)"],
                colours=[ocf.ENSEMBLE_LINE, ocf.DATA_BLUE],
            ),
            *panels,
        ],
        number=POST_HOC_FIGURE_NUMBERS[("solar", "permutation")],
        title=(
            "For six solar farms, the planned blend's gain over padded ENS is larger than all "
            f"17 shuffled controls' at {where}"
        ),
        subtitle=[
            (
                "Each grey tick is one shuffled control: UKV-CEDA's columns shuffled within "
                "generator, year-month, and hour of day under its own seed, minus padded ENS, as a "
                "difference in mean absolute error in points of capacity at the primary setting. "
                "The blue diamond is the planned blend minus padded ENS. The dashed line is zero: "
                "no difference from padded ENS."
            ),
            (
                "The p-value is the share of the 18 values (17 controls and P1) at or below P1, so "
                "with 17 controls it cannot go below 0.056. Six solar farms, "
                f"{CAPACITY_NOTE}"
            ),
        ],
        figure_planning="exploratory",
    )


def older_rows(*, intervals: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Shape one lead day's older-run contrasts for `interval_panel`, at both settings.

    Args:
        intervals: A report's `*_intervals.parquet` rows, holding the older-run section.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        One row per contrast and setting, in points of capacity.
    """
    scope = scale_of(
        frame=intervals.filter(
            pl.col("domain") == domain,
            pl.col("day") == day,
            pl.col("scope") == fit.OLDER_SCOPE,
            pl.col("contrast").is_in(list(OLDER_LABELS)),
        )
    )
    order = {code: index for index, code in enumerate(OLDER_LABELS)}
    setting_order = {setting: index for index, setting in enumerate(SETTING_LABELS)}
    return (
        scope.with_columns(
            order=pl.col("contrast").replace_strict(order, return_dtype=pl.Int8),
            setting_order=pl.col("setting").replace_strict(setting_order, return_dtype=pl.Int8),
        )
        .sort("order", "setting_order")
        .select(
            label=pl.col("contrast").replace_strict(OLDER_LABELS, return_dtype=pl.String),
            family=pl.lit(FAMILY),
            planned=pl.lit(value=False),
            condition=pl.col("setting").replace_strict(SETTING_LABELS, return_dtype=pl.String),
            difference=pl.col("difference"),
            lower_95=pl.col("lower"),
            upper_95=pl.col("upper"),
        )
    )


def older_figure(*, intervals: pl.DataFrame, domain: DomainType) -> alt.VConcatChart:
    """Draw the older-run contrasts of one technology: one panel per lead day, one shared axis.

    Args:
        intervals: A report's `*_intervals.parquet` rows, holding the older-run section.
        domain: `solar` or `wind`.

    Returns:
        The figure.
    """
    days = fit.OLDER_DAYS
    rows = {day: older_rows(intervals=intervals, domain=domain, day=day) for day in days}
    shared = x_domain_of(rows=list(rows.values()))
    panels = [
        interval_panel(
            rows=day_rows,
            x_domain=shared,
            x_title=DIFFERENCE_TITLE if day == days[-1] else "",
            zero_label="same error",
            better_label="blend better",
            panel_title=f"Lead day {day}",
            reference_labels=index == 0,
            family_key=False,
            conditions=list(SETTING_LABELS.values()),
            condition_title="Hyperparameter setting",
            condition_key=index == 0,
            figure_planning="exploratory",
            colour_by_family=True,
        )
        for index, (day, day_rows) in enumerate(rows.items())
    ]
    return figure(
        panels=panels,
        number=POST_HOC_FIGURE_NUMBERS[(domain, "older")],
        title=(
            f"For {TECHNOLOGY_NAMES[domain]}, adding UKV-CEDA's 15 UTC run of the day before "
            "ENS's run, which starts 9 hours before ENS's run, against the planned blend"
        ),
        subtitle=[
            (
                "Post hoc. Difference in mean absolute error between two XGBoost models, first "
                "arm minus second, in points of capacity; negative means the first forecasts "
                "better. All rows are the rows where the older run is present. The older run "
                "leads 12 hours longer than the planned run, so the figure cannot separate the "
                "longer lead from the earlier start. Filled dot: primary setting. Lighter hollow "
                "mark: sensitivity setting. Line: 95% interval from resampling whole months and a "
                "fitting seed."
            ),
            f"{scope_note(intervals=intervals, domain=domain)} {CAPACITY_NOTE}",
        ],
        figure_planning="exploratory",
    )


def era_of(*, month: pl.Expr) -> pl.Expr:
    """Return the era code of a `%Y-%m` month label, by the shared design's era start months."""
    return pl.lit(0, dtype=pl.Int8) + sum(
        (month >= first).cast(pl.Int8) for first in NWP_ERA_START_MONTHS
    )


def era_weeks(*, series: pl.DataFrame, domain: DomainType) -> dict[int, datetime]:
    """Choose one week by rule within each era, from measured output alone.

    Args:
        series: `nwp_forecast_charts.measured_and_forecast`'s result: one row per (site, time).
        domain: `solar` or `wind`.

    Returns:
        Each era code that has a week covered on all seven days by every generator, to that week's
        Monday. An era with no such week is left out.
    """
    labelled = series.with_columns(era=era_of(month=pl.col("time").dt.strftime("%Y-%m")))
    weeks: dict[int, datetime] = {}
    for era in range(WEEKS_PER_TECHNOLOGY):
        subset = labelled.filter(pl.col("era") == era).drop("era")
        if subset.is_empty():
            continue
        try:
            weeks[era] = choose_week(series=subset, domain=domain)
        except ValueError:
            continue
    return weeks


def week_figure(
    *, series: pl.DataFrame, week: datetime, domain: DomainType, letter: str
) -> alt.VConcatChart:
    """Draw one week of measured output and the day-1 blend's forecast, one panel per generator.

    Args:
        series: One row per (site, time) with `measured` and `forecast`, as fractions of capacity.
        week: The week's Monday.
        domain: `solar` or `wind`.
        letter: The figure's sub-letter on its page.

    Returns:
        The figure. Its axis counts days 1 to 7 and carries no calendar date.
    """
    names = ["Measured output", "Day-1 blend forecast"]
    colours = [MEASURED_COLOUR, FORECAST_COLOUR]
    long = pl.concat(
        [
            series.select(
                "site", "time", series=pl.lit(names[0]), value=pl.col("measured") * 100.0
            ),
            series.select(
                "site", "time", series=pl.lit(names[1]), value=pl.col("forecast") * 100.0
            ),
        ]
    )
    hours = pl.datetime_range(
        week, week + timedelta(days=DAYS_ON_AXIS), interval="1h", closed="left", eager=True
    )
    grid = pl.DataFrame({"time": hours}).join(pl.DataFrame({"series": names}), how="cross")
    panels = []
    for index, site in enumerate(SITES[domain]):
        last = index == len(SITES[domain]) - 1
        drawn = (
            grid.join(
                long.filter(pl.col("site") == site).drop("site"), on=["time", "series"], how="left"
            )
            .with_columns(elapsed=(pl.col("time") - week).dt.total_minutes() / (60 * 24))
            .sort("series", "time")
            .drop("time")
        )
        panels.append(
            alt.Chart(drawn)
            .mark_line(strokeWidth=1.3, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "elapsed:Q",
                    scale=alt.Scale(domain=[0, DAYS_ON_AXIS], nice=False),
                    axis=alt.Axis(
                        values=[0.5 + day for day in range(DAYS_ON_AXIS)],
                        labelExpr="'Day ' + (datum.value + 0.5)",
                        labels=last,
                        ticks=False,
                        grid=False,
                        title="Day of the chosen week" if last else None,
                    ),
                ),
                y=alt.Y(
                    "value:Q",
                    title=site,
                    scale=alt.Scale(domain=[0, 110], nice=False),
                    axis=alt.Axis(values=[0, 50, 100], titleAngle=0),
                ),
                color=alt.Color(
                    "series:N", scale=alt.Scale(domain=names, range=colours), legend=None
                ),
            )
            .properties(width=CONTENT_WIDTH_PX - 120, height=TIME_PANEL_HEIGHT_PX)
        )
    rule = (
        "the week whose daily mean output varies most from day to day"
        if domain == "solar"
        else "the week with the largest mean hour-to-hour change in output"
    )
    return figure(
        panels=[line_key(labels=names, colours=colours), alt.vconcat(*panels, spacing=4)],
        number=letter,
        title=(
            f"Measured output of {TECHNOLOGY_NAMES[domain]} and the XGBoost model's out-of-fold "
            "forecast over one week"
        ),
        subtitle=[
            (
                "Hourly output as a percentage of capacity, measured and as forecast out of fold "
                "by the XGBoost model given ENS's mean and UKV-CEDA at lead day 1, averaged over "
                "its fitting seeds. The XGBoost model uses the primary setting. A gap in a line is "
                "an hour outside the scored rows."
            ),
            (
                f"The week is chosen by rule from measured output alone, within one era: {rule}. "
                "Each vertical axis runs from 0 to 110% of the generator's own capacity."
            ),
        ],
        figure_planning=None,
    )


def weeks(
    *, losses: pl.DataFrame, predictions: pl.DataFrame, domain: DomainType
) -> dict[int, tuple[alt.VConcatChart, str]]:
    """Draw one week figure per era, and name each week's month and year.

    Args:
        losses: The day-1 stage's saved losses at both settings.
        predictions: The day-1 stage's saved predictions.
        domain: `solar` or `wind`.

    Returns:
        Each era code to its figure and the week's month and year, for the page's text.
    """
    check_anonymised(frame=losses, domain=domain)
    series = measured_and_forecast(losses=losses, predictions=predictions, arm=BLEND_ARM_DAY1)
    figures: dict[int, tuple[alt.VConcatChart, str]] = {}
    for era, week in era_weeks(series=series, domain=domain).items():
        letter = f"{FIGURE_NUMBERS[(domain, 'weeks')]}{'abc'[era]}"
        figures[era] = (
            week_figure(series=series, week=week, domain=domain, letter=letter),
            f"{week:%B %Y}",
        )
    return figures


def optimise(*, path: Path) -> None:
    """Optimise one SVG in place with `svgo`, as `CLAUDE.md` requires before committing a chart."""
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )


def write_figure(*, chart: alt.VConcatChart, path: Path, svgo: bool) -> None:
    """Write one chart as SVG, once.

    Args:
        chart: The figure.
        path: Where to write.
        svgo: Whether to optimise the SVG afterwards.
    """
    refuse_to_overwrite(paths=[path])
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    if svgo:
        optimise(path=path)


DATE_IN_SVG: Final[re.Pattern[str]] = re.compile(r"\b20\d\d-\d\d-\d\d\b")
"""A calendar date, which no figure of a generator's time series may carry."""


def check_no_dates(*, path: Path) -> None:
    """Raise if a written SVG carries a calendar date.

    Args:
        path: The SVG.

    Raises:
        ValueError: If the file holds a `YYYY-MM-DD` date.
    """
    if DATE_IN_SVG.search(path.read_text()):
        msg = f"{path} carries a calendar date, which would identify a generator"
        raise ValueError(msg)


def main() -> int:
    """Draw every figure and write the SVGs, once."""
    studies_dir = REPO_DATA_DIR / "studies"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=studies_dir / build.OUTPUT_DIR_NAME)
    parser.add_argument("--intervals-name", default="intervals.parquet")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Where SVGs are written.")
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    parser.add_argument(
        "--post-hoc-only",
        action="store_true",
        help="Draw only the permutation and older-run figures.",
    )
    args = parser.parse_args()
    intervals = pl.read_parquet(args.results_dir / args.intervals_name)
    if args.post_hoc_only:
        post_hoc = {
            "solar_permutation.svg": permutation_figure(intervals=intervals),
            **{
                f"{domain}_older_run.svg": older_figure(intervals=intervals, domain=domain)
                for domain in DOMAINS
            },
        }
        for name, chart in post_hoc.items():
            write_figure(chart=chart, path=args.figures_dir / name, svgo=not args.no_svgo)
        return 0
    for domain in DOMAINS:
        figures = {
            f"{domain}_headline.svg": headline(intervals=intervals, domain=domain),
            f"{domain}_generators.svg": generators(intervals=intervals, domain=domain),
            f"{domain}_errors.svg": errors(intervals=intervals, domain=domain),
        }
        stage = fit.Stage(domain=domain, day=1)
        losses = fit.saved_losses(output_dir=args.results_dir, stage=stage)
        predictions = pl.concat(
            [
                pl.read_parquet(path.with_name(path.name.replace("_losses", "_predictions")))
                for path in fit.group_files(output_dir=args.results_dir, stage=stage)
            ]
        )
        for era, (chart, month) in weeks(
            losses=losses, predictions=predictions, domain=domain
        ).items():
            figures[f"{domain}_week{era + 1}.svg"] = chart
            sys.stdout.write(f"{domain} week {era + 1}: {month}\n")
        for name, chart in figures.items():
            path = args.figures_dir / name
            write_figure(chart=chart, path=path, svgo=not args.no_svgo)
            check_no_dates(path=path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
