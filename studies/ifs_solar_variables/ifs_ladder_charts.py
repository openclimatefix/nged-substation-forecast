"""Draw the figures of the IFS variable ladder page from the report's tables and the built frames.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. It reads the tables that
`ifs_ladder_report.py` wrote (`leaderboard`, `contrasts`, `splits`, `farms`, and
`planned_verdicts`),
the frames `ifs_ladder_build_dataset.py` wrote, and, for the figure that shows the models working,
the saved per-row losses of lead day 1. It refits nothing. It writes one SVG per figure into
`docs/studies/assets/`, named `ifs_solar_variables_<name>.svg`.

**A figure whose inputs do not exist yet is skipped and logged.**

**Every chart is anonymised.** Farms are labelled A to F, outputs are fractions of each farm's own
capacity, no coordinate appears, and no calendar date appears on an axis. The weather figure draws
public forecast and satellite series on three days chosen on one farm, and draws no farm's output.
The models-work figure draws every farm's output on days chosen separately for each farm, which are
never the weather figure's days, and draws no weather series. The month and year of each chosen day
are printed to the log for the page's text, and the day of the month never is.

**The days are chosen by the CAMS clear-sky index of a farm's daylight hours.** The clearest day has
the highest mean index, the dullest day the lowest, and the most variable day the highest standard
deviation of the hourly index.

Run it with `uv run --with vl-convert-python python
studies/ifs_solar_variables/ifs_ladder_charts.py`. The SVGs then go through `svgo` before they are
committed.
"""

import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf
import polars as pl
from ifs_ladder_arms import (
    ADJUSTED_LEVEL_PERCENT,
    ARM_LABELS,
    BLEND_ARM,
    BLEND_FULL_ARM,
    CONTROL_CLOUD_COVER_ARM,
    CONTROL_CLOUD_LAYERS_ARM,
    CONTROL_LATER_GROUPS_ARM,
    CONTROL_PARTNER_FULL_ARM,
    CONTROL_PARTNER_MINIMAL_ARM,
    DROP_PREFIX,
    ERA5_COMPARISON_LEAD_DAY,
    F6_ARM,
    PLANNED_CONTRASTS,
    POSITIVE_CONTROL_ARM,
    PRIMARY_SETTING,
    PRODUCTION_ARM,
    SENSITIVITY_SETTING,
    SIGNED_ERROR,
    SMALLEST_EFFECT,
    FitKey,
    lead_day_dataset_path,
    report_paths,
    results_path,
)
from studies.charts import LABEL_WIDTH_PX, PLOT_WIDTH_PX, Panel, figure
from studies.era5_ladder import (
    SEASONS,
    SKY_REGIMES,
    TOTAL_CLOUD_COVER_THRESHOLDS,
)
from studies.ifs_ladder import GROUP_NAMES, RUNGS
from studies.ifs_lead_days import LEAD_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("ifs_ladder_charts")

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the page's SVGs live."""

FILE_PREFIX: Final[str] = "ifs_solar_variables_"
"""The start of every SVG's file name."""

MIN_DAYLIGHT_HOURS: Final[int] = 5
"""A day counts for the chosen-day figures only if the farm has this many kept hours on it."""

PERCENTAGE_POINTS: Final[float] = 100.0
"""Turns a fraction of capacity into percentage points."""

HOUR_TICKS: Final[tuple[int, ...]] = (6, 12, 18)
"""Where an hour-of-day axis is labelled: the hours a reader finds at once."""

DAY_HOUR_DOMAIN: Final[tuple[int, int]] = (6, 18)
"""The hours a chosen-day figure draws, which hold the chosen days' daylight hours."""

ROW_HEIGHT_PX: Final[int] = 26
"""The height of one row of a dot-and-interval panel."""

DAY_PANEL_WIDTH_PX: Final[int] = 130
"""The width of one day in a small multiple that also has a farm label column."""

WIDE_DAY_PANEL_WIDTH_PX: Final[int] = 190
"""The width of one day in a three-day small multiple, so the three days fill the text column."""

DAY_PANEL_HEIGHT_PX: Final[int] = 80
"""The height of one variable's row in a three-day small multiple."""

DOMAIN_PADDING: Final[float] = 0.1
"""How far beyond the widest interval a dot-and-interval panel's axis extends."""

WEATHER_LEAD_DAYS: Final[tuple[int, int]] = (1, 7)
"""The two lead days the weather figure draws."""

LEAD_DAY_COLOURS: Final[dict[int, str]] = {
    1: ocf.BRAND_ORANGE,
    2: ocf.DATA_BLUE,
    3: ocf.DATA_GREEN,
    5: ocf.DATA_PURPLE,
    7: ocf.DATA_SKY,
    9: ocf.TEXT,
}
"""One colour per lead day, the same in every figure that names one."""

ARM_COLOURS: Final[dict[str, str]] = {
    "f0": ocf.DATA_BLUE,
    F6_ARM: ocf.BRAND_ORANGE,
    PRODUCTION_ARM: ocf.DATA_PURPLE,
    "f2": ocf.DATA_GREEN,
    "f1": ocf.DATA_SKY,
}
"""One colour per arm that a line chart names."""

ARM_SHORT: Final[dict[str, str]] = {
    **{rung: rung.upper() for rung in RUNGS},
    PRODUCTION_ARM: "FP",
    BLEND_ARM: "FB",
    BLEND_FULL_ARM: "F6B",
}
"""How an arm is named inside a row label."""

ARM_KEY_LINES: Final[tuple[str, ...]] = (
    "A variable set is the list of inputs the XGBoost model is given, beside sun position and",
    "calendar. F0: IFS radiation and air temperature. Each set adds to the one before. F1: total",
    "cloud cover. F2: low, mid, and high cloud cover. F3: direct radiation. F4: dew point, column",
    "water vapour, boundary-layer height. F5: convective energy (CAPE), convective inhibition,",
    "visibility. F6 (all 20 IFS variables): surface temperature, snow, precipitation, pressure,",
    "wind, gusts. IFS: ECMWF's 9 km forecast, 00 UTC run, read at a lead day.",
)
"""Subtitle lines saying what every variable set named in a chart contains."""

LEAD_DAY_KEY: Final[str] = (
    "Lead day L: the forecast was issued L days before the day it describes, so lead day 1 is "
    "the morning before."
)
"""The subtitle line that defines a lead day."""

PRODUCTION_KEY: Final[str] = (
    "FP: F0 plus dew point, pressure, precipitation, and wind speed, the variables the "
    "production forecast already receives."
)
"""The subtitle line that defines the production reference."""

BLEND_KEY: Final[str] = (
    "FB: F0 plus AIFS Single (ECMWF's machine-learned forecast) radiation and temperature. "
    "F6B: F6 plus the same two."
)
"""The subtitle line that defines the blend arms."""

CONTROL_KEY: Final[tuple[str, ...]] = (
    "Negative control: the arm's columns, with the new variables replaced by shuffled copies that",
    "carry no new information. Positive control: F2 plus CAMS's satellite-derived irradiance, a",
    "column that must help.",
)
"""Subtitle lines that define the two kinds of control."""

EXPLORATORY_KEY: Final[str] = (
    "Exploratory: not named in the study plan before any result existed, and not corrected "
    "for multiple comparisons."
)
"""The subtitle line that says what an exploratory row is."""

PLANNED_KEY: Final[str] = (
    "Planned: written into the study plan before any result existed. Lead days 1 to 3 are pooled."
)
"""The subtitle line that says what a planned row is."""

SETTINGS_PHRASE: Final[str] = "main XGBoost tuning setting"
"""The one name of the first hyperparameter setting, used in every figure."""

CLEAR_SKY_INDEX_KEY: Final[str] = (
    "Clear-sky index: CAMS irradiance divided by CAMS's irradiance for a cloudless sky."
)
"""The subtitle line that defines the clear-sky index."""


Y_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
"""The error axis title."""


def _domain(*, values: Sequence[float]) -> list[float]:
    """Return an x domain that holds every value, with `DOMAIN_PADDING` of room each side."""
    low, high = min(values), max(values)
    pad = (high - low) * DOMAIN_PADDING
    return [low - pad, high + pad]


def _title(text: str) -> alt.TitleParams:
    """Return the title of a panel, left-aligned."""
    size = ocf.font_size(style="Body Large", body_px=11)
    return alt.TitleParams(text, anchor="start", fontSize=size)


def dot_interval_panel(
    *,
    rows: pl.DataFrame,
    x_title: str,
    panel_title: str,
    colour: str,
    reference_rules: Sequence[float] = (0.0,),
    order: Sequence[str] | None = None,
) -> Panel:
    """Draw one row per label: a dot at its estimate, a thin 95% line, and an optional thick line.

    Args:
        rows: One row per mark, with `label`, `value`, `lower`, `upper`, and optionally
            `thick_lower` and `thick_upper` (an adjusted interval).
        x_title: The axis title, naming the quantity, the unit, and the better direction.
        panel_title: The title above the panel.
        colour: The colour of every mark.
        reference_rules: The x values that get a dashed vertical rule, such as zero.
        order: The labels from top to bottom, or `None` for the order of `rows`.

    Returns:
        The layered panel.
    """
    labels = list(order) if order is not None else rows["label"].to_list()
    thick = "thick_lower" in rows.columns
    domain = _domain(
        values=[
            *rows["lower"].to_list(),
            *rows["upper"].to_list(),
            *(rows["thick_lower"].to_list() if thick else []),
            *(rows["thick_upper"].to_list() if thick else []),
            *reference_rules,
        ]
    )
    scale = alt.Scale(domain=domain, nice=False, zero=False)
    y = alt.Y(
        "label:N",
        sort=labels,
        title=None,
        axis=alt.Axis(labelLimit=LABEL_WIDTH_PX, labelPadding=6, domain=False, ticks=False),
    )
    layers: list[alt.Chart] = []
    if reference_rules:
        layers.append(
            alt.Chart(pl.DataFrame({"x": list(reference_rules)}))
            .mark_rule(color=ocf.TEXT, strokeDash=[4, 3], aria=False)
            .encode(x=alt.X("x:Q", scale=scale))  # ty: ignore[unresolved-attribute]
        )
    if thick:
        layers.append(
            alt.Chart(rows)
            .mark_rule(strokeWidth=6, opacity=0.35, color=colour, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X("thick_lower:Q", scale=scale, title=x_title),
                x2="thick_upper:Q",
            )
        )
    layers.append(
        alt.Chart(rows)
        .mark_rule(strokeWidth=1.5, color=colour, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, x=alt.X("lower:Q", scale=scale, title=x_title), x2="upper:Q"
        )
    )
    layers.append(
        alt.Chart(rows)
        .mark_point(filled=True, size=70, opacity=1, color=colour, aria=False)
        .encode(y=y, x=alt.X("value:Q", scale=scale, title=x_title))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(*layers).properties(
        width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * len(labels), title=_title(panel_title)
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart to `docs/studies/assets/` as an SVG."""
    path = ASSETS_DIR / f"{FILE_PREFIX}{name}.svg"
    chart.save(path)
    _LOG.info("wrote %s", path)


def independent_colours(*, chart: alt.VConcatChart) -> alt.VConcatChart:
    """Give each panel of a figure its own colour scale, so different series work."""
    return chart.resolve_scale(color="independent")


def pick(
    *,
    contrasts: pl.DataFrame,
    family: str,
    pairs: Sequence[tuple[str, str]],
    scope: str,
    setting: str = PRIMARY_SETTING,
) -> pl.DataFrame:
    """Return the contrast rows of some pairs, in the order of `pairs`.

    Args:
        contrasts: The report's contrast table.
        family: The contrast family.
        pairs: The (treatment, reference) pairs wanted.
        scope: The scope label.
        setting: The hyperparameter setting.

    Returns:
        The rows found, in the pairs' order, one per pair.

    Raises:
        ValueError: If a pair has no row or more than one.
    """
    parts = []
    for treatment, reference in pairs:
        found = contrasts.filter(
            (pl.col("family") == family)
            & (pl.col("treatment") == treatment)
            & (pl.col("reference") == reference)
            & (pl.col("scope") == scope)
            & (pl.col("setting") == setting)
            & (pl.col("target") == "pv")
        )
        if found.height != 1:
            msg = f"{found.height} rows for {family} {treatment} - {reference} ({scope}, {setting})"
            raise ValueError(msg)
        parts.append(found)
    return pl.concat(parts)


def contrast_rows(
    *, rows: pl.DataFrame, label: pl.Expr | pl.Series, adjusted: bool = False
) -> pl.DataFrame:
    """Turn contrast rows into the columns `dot_interval_panel` draws, in points of capacity."""
    columns: dict[str, pl.Expr | pl.Series] = {
        "label": label,
        "value": pl.col("difference") * PERCENTAGE_POINTS,
        "lower": pl.col("lower_95") * PERCENTAGE_POINTS,
        "upper": pl.col("upper_95") * PERCENTAGE_POINTS,
    }
    if adjusted:
        columns |= {
            "thick_lower": pl.col("lower_adjusted") * PERCENTAGE_POINTS,
            "thick_upper": pl.col("upper_adjusted") * PERCENTAGE_POINTS,
        }
    return rows.select(**columns)


def figure_headline(
    *, contrasts: pl.DataFrame, verdicts: pl.DataFrame, scope: str
) -> alt.TopLevelMixin:
    """Draw the five planned contrasts, pooled over lead days 1 to 3, at both settings.

    Args:
        contrasts: The report's contrast table.
        verdicts: The report's planned verdict table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: the main setting above the second setting.
    """
    smallest = SMALLEST_EFFECT["pv"] * PERCENTAGE_POINTS
    final = dict(zip(verdicts["label"].to_list(), verdicts["final"].to_list(), strict=True))
    labels = {
        planned.label: (
            f"{planned.label}: {ARM_SHORT[planned.treatment]} minus "
            f"{ARM_SHORT[planned.reference]} ({final.get(planned.label, 'not fitted')})"
        )
        for planned in PLANNED_CONTRASTS
    }
    order = list(labels.values())
    panels: list[Panel] = []
    for setting, title in (
        (PRIMARY_SETTING, "Main XGBoost tuning setting"),
        (SENSITIVITY_SETTING, "Second XGBoost tuning setting"),
    ):
        selected = contrasts.filter(
            (pl.col("family") == "planned") & (pl.col("setting") == setting)
        ).with_columns(key=pl.col("label"))
        rows = contrast_rows(
            rows=selected,
            label=pl.col("label").replace_strict(labels, default=pl.col("label")),
            adjusted=True,
        )
        if rows.is_empty():
            continue
        panels.append(
            dot_interval_panel(
                rows=rows,
                x_title=(
                    "Mean absolute error minus the reference's (points of capacity; "
                    "more negative is better)"
                ),
                panel_title=title,
                colour=ocf.DATA_PURPLE,
                reference_rules=(0.0, -smallest),
                order=[label for label in order if label in rows["label"].to_list()],
            )
        )
    return figure(
        panels=panels,
        number=1,
        title="Change in solar farm error from adding IFS variables, five planned contrasts",
        subtitle=[
            scope,
            "Dot: estimate. Thin line: 95% interval from resampling whole months.",
            f"Thick line: {ADJUSTED_LEVEL_PERCENT:g}% interval, adjusted for the five planned ",
            "contrasts.",
            "Dashed rules: no difference, and the smallest improvement worth acting on (0.1 ",
            "points of",
            "capacity). Each row is the error with the first variable set minus the error with the",
            "second. The bracket is the verdict from both settings and the negative control: gain,",
            "no gain, or unresolved. Error: mean absolute error, pooled over lead days 1 to 3.",
            "P4 and P5 use only the months that AIFS Single also covers.",
            *ARM_KEY_LINES,
            BLEND_KEY,
            LEAD_DAY_KEY,
            "All rows are planned: written into the study plan before any result existed.",
        ],
        figure_planning=None,
    )


def choose_days(
    *, dataset: pl.DataFrame, site: str, exclude: Sequence[date] = (), log_site: bool = True
) -> pl.DataFrame:
    """Choose the clearest, most variable, and dullest day on one farm by the stated rule.

    Args:
        dataset: The kept rows.
        site: The farm.
        exclude: Dates that may not be chosen.
        log_site: Whether the log names the farm. The weather figure leaves it out, because a page
            that named the farm beside public weather would let a reader place the farm.

    Returns:
        One row per chosen day with `day_label`, `site`, and `date`.
    """
    daily = (
        dataset.filter((pl.col("site") == site) & pl.col("cams_clear_sky_index").is_not_null())
        .with_columns(date=pl.col("time").dt.date())
        .filter(~pl.col("date").is_in(list(exclude)))
        .group_by("date")
        .agg(
            mean_index=pl.col("cams_clear_sky_index").mean(),
            spread=pl.col("cams_clear_sky_index").std(),
            hours=pl.len(),
        )
        .filter(pl.col("hours") >= MIN_DAYLIGHT_HOURS)
    )
    chosen = [
        ("Clearest day", daily.sort("mean_index", descending=True).row(0, named=True)),
        ("Most variable day", daily.sort("spread", descending=True).row(0, named=True)),
        ("Dullest day", daily.sort("mean_index").row(0, named=True)),
    ]
    for label, row in chosen:
        month = row["date"].strftime("%B %Y")
        if log_site:
            _LOG.info("%s: farm %s, %s", label, site, month)
        else:
            _LOG.info("%s: %s", label, month)
    return pl.DataFrame(
        {
            "day_label": [label for label, _ in chosen],
            "site": [site] * len(chosen),
            "date": [row["date"] for _, row in chosen],
        }
    )


def figure_weather(
    *, near: pl.DataFrame, far: pl.DataFrame, days: pl.DataFrame, scope: str
) -> alt.TopLevelMixin:
    """Draw the IFS forecasts at two lead days and CAMS irradiance on three days at one farm.

    No farm's output is drawn, so the series cannot be matched to a farm's output on a date.

    Args:
        near: The lead-day-1 frame.
        far: The lead-day-7 frame.
        days: The chosen days.
        scope: The line naming the farm count, hours, and span.

    Returns:
        The figure, with one panel per group of series.
    """
    lead_near, lead_far = WEATHER_LEAD_DAYS
    names = {
        "cams": "CAMS irradiance",
        "ifs_near": f"IFS, lead day {lead_near}",
        "ifs_far": f"IFS, lead day {lead_far}",
    }
    base = (
        near.filter(pl.col("site") == days["site"][0])
        .with_columns(date=pl.col("time").dt.date())
        .join(days.select("date", "day_label"), on="date")
        .join(
            far.select(
                "site",
                "time",
                far_radiation=pl.col("shortwave_radiation"),
                far_cloud=pl.col("cloud_cover"),
            ),
            on=["site", "time"],
        )
    )
    panels_spec: list[tuple[str, str, dict[str, str]]] = [
        (
            "Radiation",
            "W m⁻²",
            {
                names["cams"]: "cams_ghi_w_m2",
                names["ifs_near"]: "shortwave_radiation",
                names["ifs_far"]: "far_radiation",
            },
        ),
        (
            "Total cloud cover",
            "%",
            {names["ifs_near"]: "cloud_cover", names["ifs_far"]: "far_cloud"},
        ),
        (
            f"Cloud layers, lead day {lead_near}",
            "%",
            {
                "Low": "cloud_cover_low",
                "Medium": "cloud_cover_mid",
                "High": "cloud_cover_high",
            },
        ),
    ]
    colours = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
    panels: list[Panel] = []
    for title, unit, series in panels_spec:
        long = pl.concat(
            base.select(
                "day_label",
                "hour_of_day",
                series=pl.lit(name),
                value=pl.col(column).cast(pl.Float64),
            ).drop_nulls("value")
            for name, column in series.items()
        )
        chart = (
            alt.Chart(long)
            .mark_line(strokeWidth=1.5, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour_of_day:Q",
                    title="Hour of day (UTC)",
                    scale=alt.Scale(domain=DAY_HOUR_DOMAIN),
                    axis=alt.Axis(values=HOUR_TICKS),
                ),
                y=alt.Y("value:Q", title=f"{title} ({unit})"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=list(series), range=colours[: len(series)]),
                    legend=alt.Legend(title=title, orient="top", columns=3, labelLimit=300),
                ),
            )
            .properties(width=WIDE_DAY_PANEL_WIDTH_PX, height=DAY_PANEL_HEIGHT_PX)
            .facet(column=alt.Column("day_label:N", sort=days["day_label"].to_list(), title=None))
        )
        panels.append(chart)
    return independent_colours(
        chart=figure(
            panels=panels,
            number=2,
            title="IFS forecasts a day and a week ahead, against CAMS, on three days",
            subtitle=[
                scope,
                (
                    "Three days at one farm, chosen by the CAMS clear-sky index of its daylight "
                    "hours: the clearest (highest mean), the most variable (highest spread), and "
                    "the dullest (lowest mean). The same hours are drawn at both lead days."
                ),
                CLEAR_SKY_INDEX_KEY,
                LEAD_DAY_KEY,
            ],
            figure_planning=None,
        )
    )


def figure_lead_day_skill(*, board: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw the error of F0, F6, and FP at every lead day, on the rows every lead day holds.

    Args:
        board: The report's leaderboard table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure.
    """
    arms = ("f0", PRODUCTION_ARM, F6_ARM)
    rows = board.filter(
        (pl.col("view") == "ladder")
        & (pl.col("target") == "pv")
        & (pl.col("rows") == "common")
        & pl.col("arm").is_in(list(arms))
    ).select(
        "lead_day",
        arm=pl.col("arm"),
        value=pl.col("mae") * PERCENTAGE_POINTS,
        lower=pl.col("mae_lower") * PERCENTAGE_POINTS,
        upper=pl.col("mae_upper") * PERCENTAGE_POINTS,
    )
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(arms), range=[ARM_COLOURS[arm] for arm in arms]),
        legend=alt.Legend(
            title=None,
            labelExpr=" : ".join(f"datum.value == '{arm}' ? '{ARM_LABELS[arm]}'" for arm in arms)
            + " : datum.value",
        ),
    )
    x = alt.X(
        "lead_day:Q",
        title="Lead day (days between the forecast's issue and the day it describes)",
        scale=alt.Scale(domain=[0.5, max(LEAD_DAYS) + 0.5]),
        axis=alt.Axis(values=list(LEAD_DAYS)),
    )
    y = alt.Y("value:Q", title=Y_TITLE)
    chart = alt.layer(
        alt.Chart(rows)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(x=x, y=alt.Y("lower:Q", title=Y_TITLE), y2="upper:Q", color=colour),  # ty: ignore[unresolved-attribute]
        alt.Chart(rows).mark_line(strokeWidth=1.5, aria=False).encode(x=x, y=y, color=colour),  # ty: ignore[unresolved-attribute]
        alt.Chart(rows)
        .mark_point(filled=True, size=60, opacity=1, aria=False)
        .encode(x=x, y=y, color=colour),  # ty: ignore[unresolved-attribute]
    ).properties(width=PLOT_WIDTH_PX, height=200)
    return figure(
        panels=[chart],
        number=3,
        title="Forecast error rises with lead day, and the extra IFS variables do not change that",
        subtitle=[
            scope,
            "Dot: estimate at the main XGBoost tuning setting. Line: 95% interval from resampling",
            "whole months. Every lead day is scored on the same farm-hours, those that all six ",
            "lead",
            "days hold, so the lead days compare.",
            *ARM_KEY_LINES,
            PRODUCTION_KEY,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def predictions_for_days(
    *, dataset: pl.DataFrame, losses: pl.DataFrame, days: pl.DataFrame
) -> pl.DataFrame:
    """Return measured and out-of-fold output as a percentage of capacity on each farm's days.

    Args:
        dataset: The lead-day-1 rows.
        losses: The output target's lead-day-1 losses.
        days: The chosen days of every farm, with `site`, `date`, and `day_label`.

    Returns:
        Rows of `site`, `day_label`, `hour_of_day`, `series`, and `value`, in percent of capacity,
        for the measured output and the first seed's `f0` and `f6` predictions at the main setting.
    """
    base = (
        dataset.with_columns(date=pl.col("time").dt.date())
        .join(days.select("site", "date", "day_label"), on=["site", "date"])
        .select("site", "time", "day_label", "hour_of_day", "power_mw", "effective_capacity_mw")
    )
    measured = base.select(
        "site",
        "day_label",
        "hour_of_day",
        series=pl.lit("Measured"),
        value=(pl.col("power_mw") / pl.col("effective_capacity_mw") * 100.0).cast(pl.Float64),
    )
    predicted = [
        losses.filter(
            (pl.col("arm") == arm) & (pl.col("setting") == PRIMARY_SETTING) & (pl.col("seed") == 0)
        )
        .join(base, on=["site", "time"])
        .select(
            "site",
            "day_label",
            "hour_of_day",
            series=pl.lit(ARM_LABELS[arm]),
            value=(
                (pl.col("power_mw") + pl.col(SIGNED_ERROR))
                / pl.col("effective_capacity_mw")
                * 100.0
            ).cast(pl.Float64),
        )
        for arm in ("f0", F6_ARM)
    ]
    long = pl.concat([measured, *predicted]).sort("site", "day_label", "series", "hour_of_day")
    # A run of hours is one line, so a gap in the kept hours breaks the line instead of joining it.
    return long.with_columns(
        segment=(pl.col("hour_of_day").diff().over("site", "day_label", "series") > 1)
        .fill_null(False)
        .cum_sum()
        .over("site", "day_label", "series")
    )


def figure_models_work(
    *,
    dataset: pl.DataFrame,
    losses: pl.DataFrame,
    days: pl.DataFrame,
    farms: pl.DataFrame,
    scope: str,
) -> alt.TopLevelMixin:
    """Draw measured and predicted output on each farm's chosen days, and each farm's error.

    Args:
        dataset: The lead-day-1 rows.
        losses: The output target's lead-day-1 losses.
        days: Each farm's chosen days, which exclude the weather figure's days.
        farms: The report's per-farm error table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure.
    """
    series = predictions_for_days(dataset=dataset, losses=losses, days=days)
    day_order = days["day_label"].unique(maintain_order=True).to_list()
    names = ["Measured", ARM_LABELS["f0"], ARM_LABELS[F6_ARM]]
    timeline = (
        alt.Chart(series)
        .mark_line(strokeWidth=1.25, aria=False, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour_of_day:Q",
                title="Hour of day (UTC)",
                scale=alt.Scale(domain=DAY_HOUR_DOMAIN),
                axis=alt.Axis(values=HOUR_TICKS),
            ),
            y=alt.Y("value:Q", title="% of capacity"),
            detail="segment:N",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=names, range=[ocf.TEXT, ARM_COLOURS["f0"], ARM_COLOURS[F6_ARM]]
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=300),
            ),
        )
        .properties(width=DAY_PANEL_WIDTH_PX, height=60)
        .facet(
            row=alt.Row("site:N", title="Farm"),
            column=alt.Column("day_label:N", sort=day_order, title=None),
        )
    )
    per_farm = farms.filter(
        (pl.col("lead_day") == 1) & pl.col("arm").is_in(["f0", "f1", "f2", F6_ARM])
    ).select("site", "arm", value=pl.col("value") * 100.0)
    arms = ("f0", "f1", "f2", F6_ARM)
    error = (
        alt.Chart(per_farm)
        .mark_point(filled=True, size=70, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            y=alt.Y("site:N", title="Farm"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(arms), range=[ARM_COLOURS[arm] for arm in arms]),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelExpr=" : ".join(
                        f"datum.value == '{arm}' ? '{arm.upper()}'" for arm in arms
                    )
                    + " : datum.value",
                ),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=160)
    )
    return independent_colours(
        chart=figure(
            panels=[timeline, error],
            number=4,
            title="Day-ahead predictions on months the models never saw, three days at every farm",
            subtitle=[
                scope,
                "Top: measured output and the lead-day-1 predictions of F0 and F6, on three days",
                "chosen for each farm (the clearest, most variable, and dullest).",
                "Bottom: each farm's mean absolute error for F0, F1, F2, and F6, lead day 1.",
                CLEAR_SKY_INDEX_KEY,
                *ARM_KEY_LINES,
                "Output is a percentage of each farm's own capacity. Hours before 06 and after",
                "18 UTC are not drawn.",
            ],
            figure_planning=None,
        )
    )


def figure_controls(*, board: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw the controls' errors beside F0 and F6 at lead days 1 and 3.

    Args:
        board: The report's leaderboard table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: one panel per lead day.
    """
    arms = [
        "f0",
        CONTROL_CLOUD_COVER_ARM,
        "f1",
        CONTROL_CLOUD_LAYERS_ARM,
        "f2",
        POSITIVE_CONTROL_ARM,
        CONTROL_LATER_GROUPS_ARM,
        F6_ARM,
    ]
    panels: list[Panel] = []
    for lead_day in (1, 3):
        rows = board.filter(
            (pl.col("view") == "ladder")
            & (pl.col("target") == "pv")
            & (pl.col("rows") == "all")
            & (pl.col("lead_day") == lead_day)
            & pl.col("arm").is_in(arms)
        ).select(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            value=pl.col("mae") * PERCENTAGE_POINTS,
            lower=pl.col("mae_lower") * PERCENTAGE_POINTS,
            upper=pl.col("mae_upper") * PERCENTAGE_POINTS,
            arm=pl.col("arm"),
        )
        order = [ARM_LABELS[arm] for arm in arms if arm in rows["arm"].to_list()]
        panels.append(
            dot_interval_panel(
                rows=rows.drop("arm"),
                x_title="Mean absolute error (% of capacity; smaller is better)",
                panel_title=f"Lead day {lead_day}",
                colour=LEAD_DAY_COLOURS[lead_day],
                reference_rules=(),
                order=order,
            )
        )
    return figure(
        panels=panels,
        number=5,
        title="Shuffled copies of extra variables do not lower the error; CAMS irradiance does",
        subtitle=[
            scope,
            f"Dot: estimate at the {SETTINGS_PHRASE}. Line: 95% interval from resampling whole ",
            "months.",
            "Each row is one variable set, scored on the same hours, so these intervals overlap ",
            "more",
            "than the paired differences would.",
            *ARM_KEY_LINES,
            *CONTROL_KEY,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def figure_leaderboard(*, board: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw every rung's error at every lead day, with intervals.

    Args:
        board: The report's leaderboard table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: one panel with a row per variable set and a coloured mark per lead day.
    """
    arms = [*RUNGS, PRODUCTION_ARM]
    rows = board.filter(
        (pl.col("view") == "ladder")
        & (pl.col("target") == "pv")
        & (pl.col("rows") == "common")
        & pl.col("arm").is_in(arms)
    ).select(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        lead_day=pl.col("lead_day").cast(pl.String),
        value=pl.col("mae") * PERCENTAGE_POINTS,
        lower=pl.col("mae_lower") * PERCENTAGE_POINTS,
        upper=pl.col("mae_upper") * PERCENTAGE_POINTS,
    )
    order = [ARM_LABELS[arm] for arm in arms]
    domain = _domain(values=[*rows["lower"].to_list(), *rows["upper"].to_list()])
    scale = alt.Scale(domain=domain, nice=False, zero=False)
    y = alt.Y(
        "label:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=LABEL_WIDTH_PX, labelPadding=6, domain=False, ticks=False),
    )
    offset = alt.YOffset("lead_day:N", scale=alt.Scale(domain=[str(d) for d in LEAD_DAYS]))
    colour = alt.Color(
        "lead_day:N",
        title="Lead day",
        scale=alt.Scale(
            domain=[str(d) for d in LEAD_DAYS], range=[LEAD_DAY_COLOURS[d] for d in LEAD_DAYS]
        ),
    )
    title = "Mean absolute error (% of capacity; smaller is better)"
    chart = alt.layer(
        alt.Chart(rows)
        .mark_rule(strokeWidth=1.2, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X("lower:Q", scale=scale, title=title),
            x2="upper:Q",
            color=colour,
        ),
        alt.Chart(rows)
        .mark_point(filled=True, size=40, opacity=1, aria=False)
        .encode(y=y, yOffset=offset, x=alt.X("value:Q", scale=scale, title=title), color=colour),  # ty: ignore[unresolved-attribute]
    ).properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * len(arms) * 2.4)
    return figure(
        panels=[chart],
        number=6,
        title="Error of every variable set at every lead day",
        subtitle=[
            scope,
            f"Dot: estimate at the {SETTINGS_PHRASE}. Line: 95% interval from resampling whole ",
            "months.",
            "Every lead day is scored on the same farm-hours, those that all six lead days hold. ",
            "The",
            "intervals of two rows overlap more than the paired differences between rows would.",
            *ARM_KEY_LINES,
            PRODUCTION_KEY,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def figure_era5_against_ifs(*, contrasts: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw ERA5's gain from each group beside IFS's gain, at lead day 1, on the same rows.

    Args:
        contrasts: The report's contrast table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: one panel for IFS and one for ERA5.
    """
    scope_label = f"lead day {ERA5_COMPARISON_LEAD_DAY}"
    ifs = pick(
        contrasts=contrasts,
        family="against_f0",
        pairs=[("f1", "f0"), ("f2", "f0"), (F6_ARM, "f0")],
        scope=scope_label,
    )
    era5 = pick(
        contrasts=contrasts,
        family="era5",
        pairs=[("era5_g1", "era5_g0"), ("era5_g2", "era5_g0"), ("era5_g9", "era5_g0")],
        scope=scope_label,
    )
    ifs_labels = ["Total cloud (F1)", "Cloud layers (F2)", "All 20 IFS variables (F6)"]
    era5_labels = ["Total cloud (G1)", "Cloud layers (G2)", "All 34 ERA5 variables (G9)"]
    x_title = (
        "Mean absolute error minus the minimal set's (points of capacity; more negative is better)"
    )
    panels = [
        dot_interval_panel(
            rows=contrast_rows(rows=ifs, label=pl.Series(ifs_labels)),
            x_title=x_title,
            panel_title=f"IFS forecast, lead day {ERA5_COMPARISON_LEAD_DAY}: change from F0",
            colour=ocf.BRAND_ORANGE,
        ),
        dot_interval_panel(
            rows=contrast_rows(rows=era5, label=pl.Series(era5_labels)),
            x_title=x_title,
            panel_title="ERA5 reanalysis of the same hours: change from G0",
            colour=ocf.DATA_BLUE,
        ),
    ]
    return figure(
        panels=panels,
        number=7,
        title="ERA5 and IFS agree on which variables help, on the same farm-hours",
        subtitle=[
            scope,
            "Dot: estimate at the main XGBoost tuning setting. Line: 95% interval from resampling",
            "whole months. Dashed rule: no difference. Both panels are scored on the same ",
            "farm-hours",
            "and months. ERA5 is a reanalysis of past weather, not a forecast; its radiation comes",
            "from short forecasts and its cloud fields from the analysis, so its cloud gain can",
            "include a timing advantage that IFS lacks. G0: ERA5 radiation and air temperature. ",
            "G1 and",
            "G2 add the cloud variables as F1 and F2 do. G9 adds every other ERA5 variable ",
            "studied.",
            *ARM_KEY_LINES,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def _split_panel(*, splits: pl.DataFrame, kind: str, order: Sequence[str]) -> Panel:
    """Draw one split's contrasts against F0: a dot per contrast and group."""
    contrasts = [("f2", "F2 (cloud layers)"), (F6_ARM, "F6 (all 20)")]
    selected = (
        splits.filter(
            (pl.col("split") == kind)
            & (pl.col("reference") == "f0")
            & pl.col("difference").is_not_null()
        )
        .with_columns(contrast=pl.col("treatment"))
        .select(
            label=pl.col("group"),
            contrast=pl.col("contrast"),
            value=pl.col("difference") * PERCENTAGE_POINTS,
            lower=pl.col("lower_95") * PERCENTAGE_POINTS,
            upper=pl.col("upper_95") * PERCENTAGE_POINTS,
        )
    )
    domain = _domain(values=[0.0, *selected["lower"].to_list(), *selected["upper"].to_list()])
    axis_scale = alt.Scale(domain=domain, nice=False, zero=False)
    title = "Mean absolute error minus F0's (points of capacity; more negative is better)"
    y = alt.Y("label:N", sort=list(order), title=None, axis=alt.Axis(labelLimit=LABEL_WIDTH_PX))
    colour = alt.Color(
        "contrast:N",
        scale=alt.Scale(
            domain=[arm for arm, _ in contrasts],
            range=[ARM_COLOURS[arm] for arm, _ in contrasts],
        ),
        legend=alt.Legend(
            title=None,
            labelExpr=" : ".join(f"datum.value == '{arm}' ? '{name}'" for arm, name in contrasts)
            + " : datum.value",
        ),
    )
    offset = alt.YOffset("contrast:N", scale=alt.Scale(domain=[arm for arm, _ in contrasts]))
    return alt.layer(
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.TEXT, strokeDash=[4, 3], aria=False)
        .encode(x=alt.X("x:Q", scale=axis_scale)),  # ty: ignore[unresolved-attribute]
        alt.Chart(selected)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X("lower:Q", scale=axis_scale, title=title),
            x2="upper:Q",
            color=colour,
        ),
        alt.Chart(selected)
        .mark_point(filled=True, size=60, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=offset, x=alt.X("value:Q", scale=axis_scale, title=title), color=colour
        ),
    ).properties(width=PLOT_WIDTH_PX, height=int(ROW_HEIGHT_PX * len(order) * 1.8))


def figure_regimes(*, splits: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw F2's and F6's change from F0 by ERA5 cloud regime and by season.

    Args:
        splits: The report's split table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: a regime panel above a season panel.
    """
    clear_below, overcast_from = TOTAL_CLOUD_COVER_THRESHOLDS
    return figure(
        panels=[
            _split_panel(splits=splits, kind="regime_era5", order=list(SKY_REGIMES)),
            _split_panel(splits=splits, kind="season", order=list(SEASONS)),
        ],
        number=8,
        title="Change in error by sky regime and by season",
        subtitle=[
            scope,
            "Dot: estimate. Line: 95% interval from resampling whole months. Dashed rule: no",
            "difference. Each row is one group of hours, pooled over lead days 1 to 3. Each ",
            "colour is",
            "the error with a variable set minus the error with F0, the minimal set. Error: mean",
            "absolute error.",
            f"Sky regime from ERA5's total cloud cover: clear below {clear_below}, overcast from",
            f"{overcast_from}, broken between. The thresholds were fixed before any result.",
            *ARM_KEY_LINES,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def figure_hour_of_day(*, splits: pl.DataFrame, scope: str) -> alt.TopLevelMixin | None:
    """Draw how much all 20 IFS variables lower the error at each UTC hour of the day.

    Args:
        splits: The report's split table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the splits hold no hour-of-day contrasts.
    """
    hours = splits.filter(
        (pl.col("split") == "hour_of_day")
        & (pl.col("treatment") == F6_ARM)
        & pl.col("difference").is_not_null()
    ).select(
        hour=pl.col("group").cast(pl.Int64),
        relative=pl.col("difference") / pl.col("reference_value") * 100.0,
        lower=pl.col("lower_95") / pl.col("reference_value") * 100.0,
        upper=pl.col("upper_95") / pl.col("reference_value") * 100.0,
    )
    if hours.is_empty():
        return None
    base = alt.Chart(hours).encode(
        x=alt.X(
            "hour:Q",
            title="Hour of day (UTC; the label of the hour that ends at that time)",
            axis=alt.Axis(values=HOUR_TICKS),
            scale=alt.Scale(domain=[4, 21]),
        )
    )
    band = base.mark_area(opacity=0.2, color=ocf.BRAND_ORANGE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lower:Q", title="Change in mean absolute error (% of F0's)"), y2="upper:Q"
    )
    line = base.mark_line(point=True, strokeWidth=1.5, color=ocf.BRAND_ORANGE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y="relative:Q"
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color=ocf.TEXT, strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return figure(
        panels=[(band + line + zero).properties(width=PLOT_WIDTH_PX, height=150)],
        number=9,
        title="The gain from all 20 IFS variables by hour of day",
        subtitle=[
            scope,
            "Line: the mean absolute error with F6 minus the error with F0, as a percentage of ",
            "F0's",
            "error, pooled over lead days 1 to 3. Band: 95% interval from resampling whole months.",
            "Below the dashed line is a gain.",
            *ARM_KEY_LINES,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def figure_drop_one(*, contrasts: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw the error added when each group is removed from all 20 IFS variables.

    Args:
        contrasts: The report's contrast table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure.
    """
    pairs = [(f"{DROP_PREFIX}{rung}", F6_ARM) for rung in RUNGS[1:]]
    selected = pick(contrasts=contrasts, family="drop_one", pairs=pairs, scope="pooled 1-3")
    labels = [f"Without {GROUP_NAMES[rung]} ({rung.upper()})" for rung in RUNGS[1:]]
    rows = contrast_rows(rows=selected, label=pl.Series(labels))
    panel = dot_interval_panel(
        rows=rows,
        x_title=(
            "Mean absolute error minus the full set's (points of capacity; positive means the "
            "group helped)"
        ),
        panel_title="Lead days 1 to 3, pooled",
        colour=ocf.BRAND_ORANGE,
    )
    return figure(
        panels=[panel],
        number=10,
        title="Change in error when one group of variables is removed from all 20 IFS variables",
        subtitle=[
            scope,
            "Dot: estimate. Line: 95% interval from resampling whole months. Dashed rule: no",
            "difference. Each row is the error of F6 without the variables that one group adds,",
            "minus the error of F6. Error: mean absolute error. Positive means the group helped.",
            *ARM_KEY_LINES,
            LEAD_DAY_KEY,
            EXPLORATORY_KEY,
        ],
        figure_planning=None,
    )


def figure_variables_or_forecast(
    *, contrasts: pl.DataFrame, board: pl.DataFrame, scope: str
) -> alt.TopLevelMixin:
    """Draw the third decision's three gains, then the blend arms' errors and contrasts.

    Args:
        contrasts: The report's contrast table.
        board: The report's leaderboard table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure: the three gains, the blend errors at lead day 1, and the blend contrasts.
    """
    smallest = SMALLEST_EFFECT["pv"] * PERCENTAGE_POINTS
    gains = pl.concat(
        [
            pick(
                contrasts=contrasts, family="planned", pairs=[(BLEND_ARM, "f0")], scope="pooled 1-3"
            ),
            pick(
                contrasts=contrasts,
                family="planned",
                pairs=[(BLEND_FULL_ARM, F6_ARM)],
                scope="pooled 1-3",
            ),
            contrasts.filter(pl.col("family") == "era5_p4")
            .filter(pl.col("setting") == PRIMARY_SETTING)
            .select(pl.all()),
        ],
        how="diagonal_relaxed",
    )
    gain_labels = [
        "P4: second forecast added to the minimal IFS set",
        "P5: second forecast added to all 20 IFS variables",
        "ERA5 study's P4: 12 MARS-only variables added to the rest",
    ]
    gain_rows = contrast_rows(rows=gains, label=pl.Series(gain_labels), adjusted=True)
    arms = [
        "f0",
        BLEND_ARM,
        F6_ARM,
        BLEND_FULL_ARM,
        CONTROL_PARTNER_MINIMAL_ARM,
        CONTROL_PARTNER_FULL_ARM,
    ]
    errors = board.filter(
        (pl.col("view") == "blend") & (pl.col("lead_day") == 1) & pl.col("arm").is_in(arms)
    ).select(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        arm=pl.col("arm"),
        value=pl.col("mae") * PERCENTAGE_POINTS,
        lower=pl.col("mae_lower") * PERCENTAGE_POINTS,
        upper=pl.col("mae_upper") * PERCENTAGE_POINTS,
    )
    order = [ARM_LABELS[arm] for arm in arms if arm in errors["arm"].to_list()]
    blend_pairs = [(BLEND_ARM, "f0"), (F6_ARM, "f0"), (F6_ARM, BLEND_ARM)]
    blend = pick(contrasts=contrasts, family="blend", pairs=blend_pairs, scope="pooled 1-3")
    blend_rows = contrast_rows(
        rows=blend,
        label=pl.Series(["FB minus F0", "F6 minus F0", "F6 minus FB"]),
    )
    x_title = (
        "Mean absolute error minus the reference's (points of capacity; more negative is better)"
    )
    panels = [
        dot_interval_panel(
            rows=gain_rows,
            x_title=x_title,
            panel_title="The three gains the third decision compares (planned)",
            colour=ocf.DATA_PURPLE,
            reference_rules=(0.0, -smallest),
        ),
        dot_interval_panel(
            rows=errors.drop("arm"),
            x_title="Mean absolute error (% of capacity; smaller is better)",
            panel_title="Error at lead day 1 on the blend rows",
            colour=LEAD_DAY_COLOURS[1],
            reference_rules=(),
            order=order,
        ),
        dot_interval_panel(
            rows=blend_rows,
            x_title=x_title,
            panel_title="Blend contrasts, lead days 1 to 3 pooled (exploratory)",
            colour=ocf.DATA_GREEN,
        ),
    ]
    return figure(
        panels=panels,
        number=11,
        title="A second forecast and the MARS-only variables lower the error by different amounts",
        subtitle=[
            scope,
            "Dot: estimate at the main XGBoost tuning setting. Thin line: 95% interval from",
            f"resampling whole months. Thick line: {ADJUSTED_LEVEL_PERCENT:g}% interval (the ",
            "ERA5 study's",
            "own is 99.5%). Dashed rules: no difference, and the smallest improvement worth ",
            "acting on",
            "(0.1 points of capacity). Panels 1 and 3 are differences; panel 2 is each set's ",
            "error.",
            "The IFS rows are only the months AIFS Single also covers, March 2025 to April 2026.",
            "MARS-only: 12 ERA5 variables that ECMWF's MARS archive serves and that Open-Meteo ",
            "does",
            "not. The ERA5 study fitted those on a different row set, so the two gains are not on",
            "equal rows.",
            *ARM_KEY_LINES,
            BLEND_KEY,
            CONTROL_KEY[0],
            LEAD_DAY_KEY,
            "Panel 1 is planned; panels 2 and 3 are exploratory.",
        ],
        figure_planning=None,
    )


def main() -> int:
    """Draw every figure whose inputs exist."""
    paths = report_paths()
    board = pl.read_parquet(paths.leaderboard)
    contrasts = pl.read_parquet(paths.contrasts)
    verdicts = pl.read_parquet(paths.verdicts)
    splits = pl.read_parquet(paths.splits) if paths.splits.exists() else pl.DataFrame()
    farms = pl.read_parquet(paths.farms)
    near = pl.read_parquet(lead_day_dataset_path(lead_day=1))
    far = pl.read_parquet(lead_day_dataset_path(lead_day=WEATHER_LEAD_DAYS[1]))
    first, last = near["time"].min(), near["time"].max()
    span = f"{first.strftime('%B %Y')} to {last.strftime('%B %Y')}"  # ty: ignore[unresolved-attribute]
    scope = (
        f"Six NGED solar farms, daylight hours, {span}. IFS: ECMWF's forecast via Open-Meteo. "
        "CAMS: satellite irradiance service."
    )

    shared = near.join(far.select("site", "time"), on=["site", "time"])
    weather_days = choose_days(
        dataset=shared, site=min(near["site"].unique().to_list()), log_site=False
    )
    farm_days = pl.concat(
        choose_days(dataset=near, site=site, exclude=weather_days["date"].to_list())
        for site in sorted(near["site"].unique().to_list())
    )
    losses_1 = pl.read_parquet(results_path(key=FitKey(view="ladder", lead_day=1, target="pv")))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    save(
        chart=figure_headline(contrasts=contrasts, verdicts=verdicts, scope=scope), name="headline"
    )
    save(
        chart=figure_weather(
            near=near,
            far=far,
            days=weather_days,
            scope=scope.replace("Six NGED solar farms", "One NGED solar farm"),
        ),
        name="weather",
    )
    save(chart=figure_lead_day_skill(board=board, scope=scope), name="lead_day_skill")
    save(
        chart=figure_models_work(
            dataset=near, losses=losses_1, days=farm_days, farms=farms, scope=scope
        ),
        name="models_work",
    )
    save(chart=figure_controls(board=board, scope=scope), name="controls")
    save(chart=figure_leaderboard(board=board, scope=scope), name="leaderboard")
    save(chart=figure_era5_against_ifs(contrasts=contrasts, scope=scope), name="era5_against_ifs")
    if not splits.is_empty():
        save(chart=figure_regimes(splits=splits, scope=scope), name="regimes")
        hour_chart = figure_hour_of_day(splits=splits, scope=scope)
        if hour_chart is not None:
            save(chart=hour_chart, name="hour_of_day")
    save(chart=figure_drop_one(contrasts=contrasts, scope=scope), name="drop_one")
    save(
        chart=figure_variables_or_forecast(contrasts=contrasts, board=board, scope=scope),
        name="variables_or_forecast",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
