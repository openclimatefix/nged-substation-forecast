"""Draw the figures of the ERA5 variable ladder page from the report's tables and the built frame.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It reads the tables that
`era5_ladder_report.py` wrote (`leaderboard`, `contrasts`, `splits`, and `worst_days`, each ending
in the variant and the highest rung), the frame `era5_ladder_build_dataset.py` wrote, and, for the
figure that shows the models working, the saved per-row losses. It refits nothing. It writes one SVG
per figure into `docs/studies/assets/`, named `era5_solar_variables_<name>.svg`.

**A figure whose inputs have not been built yet is skipped and logged**, so the script runs on the
first rungs while the later variables are still downloading.

**Every chart is anonymised.** Farms are labelled A to F, outputs are fractions of each farm's own
capacity, no coordinate appears, and no calendar date appears on an axis. Figure 3 and figure 5
draw public weather series on three days chosen on one farm, and draw no farm's output, so that no
farm's output can be matched to a date through those series. Figure 7 draws every farm's output on
days chosen separately for each farm, which are never the days of figures 3 and 5, and draws no
weather series. The month and year of each chosen day are printed to the log for the page's text,
and the day of the month never is.

**The days are chosen by the CAMS clear-sky index of a farm's daylight hours.** The clearest day has
the highest mean index, the dullest day the lowest, and the most variable day the highest standard
deviation of the hourly index. Only days with at least `MIN_DAYLIGHT_HOURS` kept hours count.

Run it with `uv run --with vl-convert-python python
studies/era5_solar_variables/era5_ladder_charts.py`. The SVGs then go through `svgo` before they
are committed.
"""

import argparse
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
from era5_ladder_arms import (
    ADJUSTED_LEVEL_PERCENT,
    ARM_LABELS,
    DROP_PREFIX,
    PRIMARY_SETTING,
    SIGNED_ERROR,
    SMALLEST_EFFECT,
    TARGETS,
    FitKey,
    TargetType,
    dataset_path,
    report_paths,
    results_path,
)
from studies.charts import LABEL_WIDTH_PX, PLOT_WIDTH_PX, Panel, figure
from studies.era5_ladder import (
    CLEAR_SKY_INDEX_THRESHOLDS,
    RUNGS,
    SEASONS,
    SKY_REGIMES,
    TOTAL_CLOUD_COVER_THRESHOLDS,
    RungType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_charts")

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the page's SVGs live."""

FILE_PREFIX: Final[str] = "era5_solar_variables_"
"""The start of every SVG's file name."""

MIN_DAYLIGHT_HOURS: Final[int] = 5
"""A day counts for the chosen-day figures only if the farm has this many kept hours on it."""

PANEL_TITLES: Final[dict[TargetType, str]] = {
    "pv": "Solar farms' output",
    "cams": "CAMS clearness index",
}
"""The title of a target's panel."""

TARGET_NAMES: Final[dict[TargetType, str]] = {
    "pv": "the solar farms' output",
    "cams": "CAMS clearness index",
}
"""How a target is named in a sentence."""

TARGET_COLOURS: Final[dict[TargetType, str]] = {"pv": ocf.DATA_PURPLE, "cams": ocf.DATA_GREEN}
"""One colour per target, so a reader tracks the target across figures."""

ARM_COLOURS: Final[dict[str, str]] = {
    "g0": ocf.DATA_BLUE,
    "g1": ocf.DATA_SKY,
    "g2": ocf.DATA_GREEN,
    "g9": ocf.BRAND_ORANGE,
}
"""One colour per arm named in a chart, the same in every figure that names it."""

CONTRAST_ARMS: Final[tuple[str, ...]] = ("g1", "g2", "g9")
"""The arms whose contrast against the minimal set the regime and season figures draw."""

CONTRAST_NAMES: Final[dict[str, str]] = {
    "g1": "total cloud (G1)",
    "g2": "three cloud layers (G2)",
    "g9": "every ERA5 variable (G9)",
}
"""What each contrast against the minimal set adds, in words."""

SERIES_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
)
"""The colours of the series within one panel of the chosen-day figures."""

SERIES_PANELS: Final[tuple[tuple[str, tuple[str, ...], str], ...]] = (
    ("Irradiance", ("ssrd", "ssrdc", "cams_ghi_w_m2", "fdir"), "W m⁻²"),
    ("Cloud cover", ("tcc", "lcc", "mcc", "hcc"), "fraction"),
    ("Cloud water", ("tclw", "tciw"), "kg m⁻²"),
    ("Cloud base", ("cbh",), "m"),
)
"""The variable groups of figure 3 as (panel title, columns, unit)."""

DISPLAY_NAMES: Final[dict[str, str]] = {
    "ssrd": "ERA5 ssrd",
    "ssrdc": "ERA5 ssrdc (clear sky)",
    "cams_ghi_w_m2": "CAMS GHI",
    "fdir": "ERA5 fdir (direct)",
    "tcc": "Total",
    "lcc": "Low",
    "mcc": "Medium",
    "hcc": "High",
    "tclw": "Liquid",
    "tciw": "Ice",
    "cbh": "Cloud base",
}
"""What each column is called in a legend."""

CANDIDATES: Final[tuple[tuple[str, str], ...]] = (
    ("tclw", "Cloud liquid water (kg m⁻²)"),
    ("tciw", "Cloud ice water (kg m⁻²)"),
    ("cbh", "Cloud base height (m)"),
    ("tcwv", "Water vapour column (kg m⁻²)"),
    ("d2m", "Dew point (K)"),
    ("blh", "Boundary layer height (m)"),
    ("sd", "Snow depth (m of water)"),
)
"""The candidate variables figure 5 bins the ERA5-minus-CAMS gap against, with their titles."""

ROW_HEIGHT_PX: Final[int] = 26
"""The height of one row of a dot-and-interval panel."""

DAY_PANEL_WIDTH_PX: Final[int] = 130
"""The width of one day in a three-day small multiple."""

DAY_PANEL_HEIGHT_PX: Final[int] = 80
"""The height of one variable's row in a three-day small multiple."""

DOMAIN_PADDING: Final[float] = 0.1
"""How far beyond the widest interval a dot-and-interval panel's axis extends."""

MIN_BAND_ROWS: Final[int] = 30
"""A binned mean is drawn only if the band holds at least this many rows."""

ARM_LABEL_EXPRESSION: Final[str] = (
    "datum.value == 'g0' ? 'G0 minimal' : datum.value == 'g1' ? 'G1 total cloud' : "
    "datum.value == 'g2' ? 'G2 cloud layers' : 'G9 everything'"
)
"""The Vega expression that turns the arm names `g0`, `g1`, `g2`, and `g9` into words."""


def _scale(*, target: TargetType) -> float:
    """Return what turns the metric into the chart's unit: percentage points, or clearness index."""
    return 100.0 if target == "pv" else 1.0


def _error_unit(*, target: TargetType) -> str:
    """Return the unit of a mean absolute error on the chart."""
    return "% of capacity" if target == "pv" else "clearness index"


def _difference_unit(*, target: TargetType) -> str:
    """Return the unit of a difference in the mean absolute error."""
    return "points of capacity" if target == "pv" else "clearness index"


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
    """Give each panel of a figure its own colour scale, so that panels with different series work.

    `studies.charts.figure` shares one colour scale across its panels, which merges the domains of
    panels that colour different things.
    """
    return chart.resolve_scale(color="independent")


def figure_1_headline(*, contrasts: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw the planned contrasts on both targets, at the adjusted level and at 95%.

    Args:
        contrasts: The report's contrast table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure.
    """
    panels: list[Panel] = []
    for target in TARGETS:
        scale = _scale(target=target)
        smallest = SMALLEST_EFFECT[target] * scale
        selected = contrasts.filter(
            (pl.col("target") == target)
            & pl.col("planned")
            & (pl.col("setting") == PRIMARY_SETTING)
        )
        rows = selected.select(
            label=pl.format(
                "{}: {} minus {}",
                pl.col("label"),
                pl.col("treatment").str.to_uppercase(),
                pl.col("reference").str.to_uppercase(),
            ),
            value=pl.col("difference") * scale,
            lower=pl.col("lower_95") * scale,
            upper=pl.col("upper_95") * scale,
            thick_lower=pl.col("lower_adjusted") * scale,
            thick_upper=pl.col("upper_adjusted") * scale,
        )
        panels.append(
            dot_interval_panel(
                rows=rows,
                x_title=(
                    "Mean absolute error minus the reference's "
                    f"({_difference_unit(target=target)}; more negative is better)"
                ),
                panel_title=PANEL_TITLES[target],
                colour=TARGET_COLOURS[target],
                reference_rules=(0.0, -smallest),
            )
        )
    return figure(
        panels=panels,
        number=1,
        title="Change in error from adding ERA5 variables, the eight planned contrasts",
        subtitle=[
            scope,
            "Dot: estimate. Thin line: 95% interval from resampling whole months.",
            (
                f"Thick line: {ADJUSTED_LEVEL_PERCENT:.3f}% interval, adjusted for the eight "
                "planned contrasts."
            ),
            "Dashed rules: no difference, and the smallest improvement worth acting on.",
            "All rows are planned: written into the study plan before any result existed.",
        ],
        figure_planning=None,
    )


def figure_2_leaderboard(
    *, board: pl.DataFrame, target: TargetType, scope: str
) -> alt.TopLevelMixin:
    """Draw every arm's mean absolute error and correlation, best first.

    Args:
        board: The report's leaderboard table.
        target: The target.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, with the error panel above the correlation panel.
    """
    scale = _scale(target=target)
    rows = board.filter((pl.col("target") == target) & (pl.col("view") == "ladder")).sort("mae")
    order = rows["label"].to_list()
    error = rows.select(
        "label",
        value=pl.col("mae") * scale,
        lower=pl.col("mae_lower") * scale,
        upper=pl.col("mae_upper") * scale,
    )
    correlation = rows.select(
        "label",
        value=pl.col("correlation"),
        lower=pl.col("correlation_lower"),
        upper=pl.col("correlation_upper"),
    )
    return figure(
        panels=[
            dot_interval_panel(
                rows=error,
                x_title=f"Mean absolute error ({_error_unit(target=target)}; smaller is better)",
                panel_title="Error",
                colour=TARGET_COLOURS[target],
                reference_rules=(),
                order=order,
            ),
            dot_interval_panel(
                rows=correlation,
                x_title="Correlation of prediction with measured value (larger is better)",
                panel_title="Correlation",
                colour=TARGET_COLOURS[target],
                reference_rules=(),
                order=order,
            ),
        ],
        number=2 if target == "pv" else "2b",
        title=f"Error and correlation of every arm for {TARGET_NAMES[target]}",
        subtitle=[
            scope,
            "Dot: estimate at the first hyperparameter setting. Line: 95% interval.",
            (
                "The arms share their rows, so these intervals overlap more than the paired "
                "differences of figure 1 do."
            ),
            "All rows are exploratory.",
        ],
        figure_planning=None,
    )


def _legend_expression() -> str:
    """Return the Vega expression that turns a contrast's arm name into its words."""
    cases = " : ".join(f"datum.value == '{arm}' ? '{name}'" for arm, name in CONTRAST_NAMES.items())
    return f"{cases} : datum.value"


def _split_panel(
    *, splits: pl.DataFrame, target: TargetType, kind: str, order: Sequence[str]
) -> Panel:
    """Draw one split's contrasts against the minimal set, one dot per added group and group."""
    scale = _scale(target=target)
    selected = splits.filter(
        (pl.col("target") == target)
        & (pl.col("split") == kind)
        & (pl.col("reference") == "g0")
        & pl.col("difference").is_not_null()
    ).select(
        label=pl.col("group"),
        contrast=pl.col("treatment"),
        value=pl.col("difference") * scale,
        lower=pl.col("lower_95") * scale,
        upper=pl.col("upper_95") * scale,
    )
    domain = _domain(values=[0.0, *selected["lower"].to_list(), *selected["upper"].to_list()])
    axis_scale = alt.Scale(domain=domain, nice=False, zero=False)
    title = (
        f"Mean absolute error minus the minimal set's ({_difference_unit(target=target)}; "
        "more negative is better)"
    )
    y = alt.Y("label:N", sort=list(order), title=None, axis=alt.Axis(labelLimit=LABEL_WIDTH_PX))
    colour = alt.Color(
        "contrast:N",
        scale=alt.Scale(
            domain=list(CONTRAST_ARMS), range=[ARM_COLOURS[arm] for arm in CONTRAST_ARMS]
        ),
        legend=alt.Legend(title=None, labelExpr=_legend_expression()),
    )
    offset = alt.YOffset("contrast:N", scale=alt.Scale(domain=list(CONTRAST_ARMS)))
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
    ).properties(
        width=PLOT_WIDTH_PX,
        height=int(ROW_HEIGHT_PX * len(order) * 1.6),
        title=_title(PANEL_TITLES[target]),
    )


def figure_regimes(
    *,
    splits: pl.DataFrame,
    number: int | str,
    kind: str,
    groups: Sequence[str],
    title: str,
    regime_note: str | None,
    scope: str,
) -> alt.TopLevelMixin:
    """Draw the contrasts against the minimal set within each weather regime or season.

    Args:
        splits: The report's split table.
        number: The figure's number.
        kind: The split, such as `regime_cams` or `season`.
        groups: The groups from top to bottom.
        title: The figure's title, a descriptor of what is split.
        regime_note: The line defining the regimes the split uses, or `None` for a split that uses
            none.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, one panel per target.
    """
    panels = [
        _split_panel(splits=splits, target=target, kind=kind, order=groups)
        for target in TARGETS
        if not splits.filter((pl.col("target") == target) & (pl.col("split") == kind)).is_empty()
    ]
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[
            scope,
            "Dot: estimate. Line: 95% interval from resampling whole months.",
            "Dashed rule: no difference.",
            *([] if regime_note is None else [regime_note]),
            "All rows are exploratory and are not corrected for multiple comparisons.",
        ],
        figure_planning=None,
    )


def figure_drop_one(*, contrasts: pl.DataFrame, scope: str) -> alt.TopLevelMixin:
    """Draw the error added when each group is removed from the full set.

    Args:
        contrasts: The report's contrast table.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, one panel per target.
    """
    panels: list[Panel] = []
    for target in TARGETS:
        scale = _scale(target=target)
        selected = contrasts.filter(
            (pl.col("target") == target)
            & (pl.col("family") == "drop_one")
            & (pl.col("setting") == PRIMARY_SETTING)
        )
        if selected.is_empty():
            continue
        rows = selected.select(
            label=pl.col("treatment").str.replace(DROP_PREFIX, "").str.to_uppercase()
            + pl.lit("'s additions removed"),
            value=pl.col("difference") * scale,
            lower=pl.col("lower_95") * scale,
            upper=pl.col("upper_95") * scale,
        )
        panels.append(
            dot_interval_panel(
                rows=rows,
                x_title=(
                    f"Mean absolute error minus the full set's ({_difference_unit(target=target)}; "
                    "positive means the group helped)"
                ),
                panel_title=PANEL_TITLES[target],
                colour=TARGET_COLOURS[target],
            )
        )
    return figure(
        panels=panels,
        number=10,
        title="Change in error when one group of variables is removed from the full set",
        subtitle=[
            scope,
            "Dot: estimate. Line: 95% interval from resampling whole months.",
            "Dashed rule: no difference.",
            "A group counts as useful only if the ladder (figure 2) and this figure agree.",
            "All rows are exploratory and are not corrected for multiple comparisons.",
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
        log_site: Whether the log names the farm. The weather figures leave it out, because a page
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


def _day_facets(*, chart: alt.Chart, days: pl.DataFrame) -> alt.FacetChart:
    """Facet a line chart into one column per chosen day."""
    return chart.properties(width=DAY_PANEL_WIDTH_PX, height=DAY_PANEL_HEIGHT_PX).facet(
        column=alt.Column("day_label:N", sort=days["day_label"].to_list(), title=None)
    )


def figure_3_days(
    *, dataset: pl.DataFrame, days: pl.DataFrame, scope: str
) -> alt.TopLevelMixin | None:
    """Draw the public weather variables for the three chosen days, one panel per group.

    No farm's output is drawn, so the series cannot be matched to a farm's output on a date.

    Args:
        dataset: The kept rows.
        days: The chosen days.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the frame lacks the cloud variables.
    """
    needed = {name for _, columns, _ in SERIES_PANELS for name in columns}
    if not needed <= set(dataset.columns):
        _LOG.info("figure 3 skipped: the frame lacks %s", sorted(needed - set(dataset.columns)))
        return None
    rows = dataset.filter(pl.col("site") == days["site"][0]).with_columns(
        date=pl.col("time").dt.date()
    )
    panels: list[Panel] = []
    for title, columns, unit in SERIES_PANELS:
        long = pl.concat(
            rows.join(days.select("date", "day_label"), on="date")
            .select(
                "day_label",
                "hour_of_day",
                series=pl.lit(DISPLAY_NAMES[column]),
                value=pl.col(column),
            )
            .filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
            for column in columns
        )
        chart = (
            alt.Chart(long)
            .mark_line(strokeWidth=1.5, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour_of_day:Q", title="Hour of day (UTC)", scale=alt.Scale(domain=[0, 24])
                ),
                y=alt.Y("value:Q", title=unit),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[DISPLAY_NAMES[column] for column in columns],
                        range=list(SERIES_COLOURS)[: len(columns)],
                    ),
                    legend=alt.Legend(title=title, orient="top"),
                ),
            )
        )
        panels.append(_day_facets(chart=chart, days=days))
    return independent_colours(
        chart=figure(
            panels=panels,
            number=3,
            title="ERA5 variables and CAMS irradiance on three days",
            subtitle=[
                scope,
                (
                    "Three days at one farm, chosen by the CAMS clear-sky index of its daylight "
                    "hours: the highest mean, the highest spread, and the lowest mean."
                ),
                f"Only days with at least {MIN_DAYLIGHT_HOURS} kept hours count.",
            ],
            figure_planning=None,
        )
    )


def _cover_label_expression() -> str:
    """Return the Vega expression that names each cloud cover in words."""
    names = {"tcc": "Total cloud cover", "lcc": "Low", "mcc": "Medium", "hcc": "High"}
    cases = " : ".join(f"datum.value == '{key}' ? '{name}'" for key, name in names.items())
    return f"{cases} : datum.value"


def figure_4_cloud_against_clearness(
    *, dataset: pl.DataFrame, scope: str
) -> alt.TopLevelMixin | None:
    """Draw CAMS clearness index against each ERA5 cloud cover, as binned means with a spread band.

    Args:
        dataset: The kept rows.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the frame lacks the cloud covers.
    """
    covers = ("tcc", "lcc", "mcc", "hcc")
    if not set(covers) <= set(dataset.columns):
        _LOG.info("figure 4 skipped: the frame lacks the cloud covers")
        return None
    bins = 20
    long = pl.concat(
        dataset.filter(pl.col("cams_clearness_index").is_not_null()).select(
            cover=pl.lit(name),
            bin=(pl.col(name).clip(0.0, 1.0) * bins).floor().clip(0, bins - 1) / bins + 0.5 / bins,
            index=pl.col("cams_clearness_index"),
        )
        for name in covers
    )
    summary = (
        long.group_by("cover", "bin")
        .agg(
            mean=pl.col("index").mean(),
            lower=pl.col("index").quantile(0.1),
            upper=pl.col("index").quantile(0.9),
            n=pl.len(),
        )
        .filter(pl.col("n") >= MIN_BAND_ROWS)
    )
    x = alt.X("bin:Q", title="ERA5 cloud cover (fraction)", scale=alt.Scale(domain=[0, 1]))
    chart = alt.layer(
        alt.Chart(summary)
        .mark_area(opacity=0.25, color=ocf.DATA_GREEN, aria=False)
        .encode(x=x, y=alt.Y("lower:Q", title="CAMS clearness index"), y2="upper:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(summary)
        .mark_line(color=ocf.DATA_GREEN, strokeWidth=2, aria=False)
        .encode(x=x, y="mean:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=PLOT_WIDTH_PX // 2 - 40, height=130)
    faceted = chart.facet(
        facet=alt.Facet(
            "cover:N",
            sort=list(covers),
            title=None,
            header=alt.Header(labelExpr=_cover_label_expression()),
        ),
        columns=2,
    )
    return figure(
        panels=[faceted],
        number=4,
        title="CAMS clearness index against each ERA5 cloud cover",
        subtitle=[
            scope,
            "Line: mean clearness index in each 0.05-wide band of cloud cover.",
            "Band: 10th to 90th percentile.",
        ],
        figure_planning=None,
    )


def figure_5_where_ssrd_misses(
    *, dataset: pl.DataFrame, days: pl.DataFrame, scope: str
) -> alt.TopLevelMixin | None:
    """Draw how far ERA5's `ssrd` is from CAMS global irradiance, over time and by each variable.

    Args:
        dataset: The kept rows.
        days: The chosen days of figure 3.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the frame holds none of the candidate variables.
    """
    present = [(name, title) for name, title in CANDIDATES if name in dataset.columns]
    if not present:
        _LOG.info("figure 5 skipped: the frame holds none of the candidate variables")
        return None
    gap = dataset.with_columns(gap=pl.col("ssrd") - pl.col("cams_ghi_w_m2"))
    rows = gap.filter(pl.col("site") == days["site"][0]).with_columns(date=pl.col("time").dt.date())
    on_days = rows.join(days.select("date", "day_label"), on="date").select(
        "day_label", "hour_of_day", "gap"
    )
    timeline = _day_facets(
        chart=alt.Chart(on_days)
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour_of_day:Q", title="Hour of day (UTC)", scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("gap:Q", title="ERA5 ssrd minus CAMS (W m⁻²)"),
        ),
        days=days,
    )
    summaries = []
    for name, title in present:
        valid = gap.filter(pl.col(name).is_not_null() & pl.col(name).is_not_nan())
        if valid.is_empty() or valid[name].min() == valid[name].max():
            continue
        banded = valid.with_columns(
            decile=(pl.col(name).rank(method="average") / valid.height * 10).floor().clip(0, 9)
        )
        summaries.append(
            banded.group_by("decile")
            .agg(x=pl.col(name).mean(), mean_gap=pl.col("gap").mean())
            .select(variable=pl.lit(title), x="x", mean_gap="mean_gap")
        )
    panels: list[Panel] = [timeline]
    if summaries:
        binned = (
            alt.Chart(pl.concat(summaries))
            .mark_line(
                point=alt.OverlayMarkDef(color=ocf.BRAND_ORANGE),
                color=ocf.BRAND_ORANGE,
                strokeWidth=1.5,
                aria=False,
            )
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("x:Q", title=None, scale=alt.Scale(zero=False)),
                y=alt.Y("mean_gap:Q", title="ERA5 ssrd minus CAMS (W m⁻²)"),
            )
            .properties(width=PLOT_WIDTH_PX // 2 - 40, height=110)
            .facet(
                facet=alt.Facet("variable:N", title=None, sort=[title for _, title in present]),
                columns=2,
            )
            .resolve_scale(x="independent")
        )
        panels.append(binned)
    return figure(
        panels=panels,
        number=5,
        title="Where ERA5's downward solar radiation differs from CAMS",
        subtitle=[
            scope,
            "Top: the gap on the three days of figure 3. Positive means ERA5 is brighter.",
            (
                "Bottom: each dot is the mean gap over one tenth of the kept hours, ranked by the "
                "variable named above the panel."
            ),
        ],
        figure_planning=None,
    )


def figure_6_cloud_water(*, dataset: pl.DataFrame, scope: str) -> alt.TopLevelMixin | None:
    """Draw clearness index against total cloud water, for hours of low, medium, and high low-cloud.

    Args:
        dataset: The kept rows.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the frame lacks cloud water or low cloud cover.
    """
    needed = {"tclw", "tciw", "lcc"}
    if not needed <= set(dataset.columns):
        _LOG.info("figure 6 skipped: the frame lacks %s", sorted(needed - set(dataset.columns)))
        return None
    names = ["Low cloud under 0.2", "Low cloud 0.2 to 0.6", "Low cloud 0.6 or more"]
    rows = dataset.filter(pl.col("cams_clearness_index").is_not_null()).with_columns(
        water=pl.col("tclw") + pl.col("tciw"),
        low_cloud=pl.when(pl.col("lcc") < 0.2)
        .then(pl.lit(names[0]))
        .when(pl.col("lcc") < 0.6)
        .then(pl.lit(names[1]))
        .otherwise(pl.lit(names[2])),
    )
    bands = 15
    rank_share = pl.col("water").rank(method="average").over("low_cloud") / pl.len().over(
        "low_cloud"
    )
    banded = rows.with_columns(band=(rank_share * bands).floor().clip(0, bands - 1))
    summary = (
        banded.group_by("low_cloud", "band")
        .agg(
            water=pl.col("water").mean(),
            mean_index=pl.col("cams_clearness_index").mean(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= MIN_BAND_ROWS)
    )
    chart = (
        alt.Chart(summary)
        .mark_line(point=True, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "water:Q",
                title="ERA5 cloud liquid plus ice water (kg m⁻²)",
                scale=alt.Scale(type="symlog", constant=0.01),
                axis=alt.Axis(values=[0, 0.01, 0.03, 0.1, 0.3, 1]),
            ),
            y=alt.Y("mean_index:Q", title="CAMS clearness index"),
            color=alt.Color(
                "low_cloud:N",
                scale=alt.Scale(domain=names, range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE]),
                legend=alt.Legend(title=None),
            ),
            shape=alt.Shape("low_cloud:N", scale=alt.Scale(domain=names), legend=None),
        )
        .properties(width=PLOT_WIDTH_PX + LABEL_WIDTH_PX - 60, height=200)
    )
    return figure(
        panels=[chart],
        number=6,
        title="CAMS clearness index against ERA5 cloud water, by low-cloud cover",
        subtitle=[
            scope,
            (
                "Each dot: the mean over one fifteenth of the hours in its low-cloud group, "
                "ranked by cloud water."
            ),
        ],
        figure_planning=None,
    )


def predictions_for_days(
    *, dataset: pl.DataFrame, losses: pl.DataFrame, days: pl.DataFrame
) -> pl.DataFrame:
    """Return measured and out-of-fold output as a percentage of capacity on each farm's days.

    Args:
        dataset: The kept rows.
        losses: The output target's losses.
        days: The chosen days of every farm, with `site`, `date`, and `day_label`.

    Returns:
        Rows of `site`, `day_label`, `hour_of_day`, `series`, and `value`, in percent of capacity,
        for the measured output and the first seed's `g0` and `g9` predictions.
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
        for arm in ("g0", "g9")
    ]
    return pl.concat([measured, *predicted])


def figure_7_models_work(
    *,
    dataset: pl.DataFrame,
    losses: pl.DataFrame,
    days: pl.DataFrame,
    splits: pl.DataFrame,
    scope: str,
) -> alt.TopLevelMixin:
    """Draw measured and predicted output on each farm's chosen days, and each farm's error.

    Args:
        dataset: The kept rows.
        losses: The output target's losses.
        days: Each farm's chosen days, which exclude the days of figures 3 and 5.
        splits: The report's split table, for each farm's error.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure.
    """
    series = predictions_for_days(dataset=dataset, losses=losses, days=days)
    day_order = days["day_label"].unique(maintain_order=True).to_list()
    timeline = (
        alt.Chart(series)
        .mark_line(strokeWidth=1.25, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour_of_day:Q", title="Hour of day (UTC)", scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("value:Q", title="% of capacity"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Measured", ARM_LABELS["g0"], ARM_LABELS["g9"]],
                    range=[ocf.TEXT, ARM_COLOURS["g0"], ARM_COLOURS["g9"]],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=DAY_PANEL_WIDTH_PX, height=60)
        .facet(
            row=alt.Row("site:N", title="Farm"),
            column=alt.Column("day_label:N", sort=day_order, title=None),
        )
    )
    arms = ("g0", "g1", "g2", "g9")
    per_farm = splits.filter(
        (pl.col("target") == "pv") & (pl.col("split") == "farm") & pl.col("treatment").is_in(arms)
    ).select("group", arm=pl.col("treatment"), value=pl.col("value") * 100.0)
    error = (
        alt.Chart(per_farm)
        .mark_point(filled=True, size=70, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            y=alt.Y("group:N", title="Farm"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(arms), range=[ARM_COLOURS[arm] for arm in arms]),
                legend=alt.Legend(title=None, labelExpr=ARM_LABEL_EXPRESSION, orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=160)
    )
    return independent_colours(
        chart=figure(
            panels=[timeline, error],
            number=7,
            title="Out-of-fold output on three days at every farm, and each farm's error",
            subtitle=[
                scope,
                (
                    "Top: measured output and the predictions of the minimal and the full set, on "
                    "three days chosen for each farm by the rule of figure 3."
                ),
                "Those days are never the days of figures 3 and 5.",
                "Bottom: each farm's mean absolute error for four arms.",
                "Output is a percentage of each farm's own capacity.",
            ],
            figure_planning=None,
        )
    )


def figure_11_hour_and_worst_days(
    *, splits: pl.DataFrame, worst: pl.DataFrame | None, scope: str
) -> alt.TopLevelMixin | None:
    """Draw error by hour of day for the minimal and full arms, and the minimal arm's worst days.

    Args:
        splits: The report's split table.
        worst: The worst farm-days table, or `None`.
        scope: The line naming the farms, hours, and span.

    Returns:
        The figure, or `None` if the splits hold no hour-of-day rows for both arms.
    """
    if splits.is_empty():
        return None
    hours = splits.filter(
        (pl.col("split") == "hour_of_day") & pl.col("treatment").is_in(["g0", "g9"])
    ).with_columns(hour=pl.col("group").cast(pl.Int64))
    if hours.is_empty() or "g9" not in hours["treatment"].to_list():
        return None
    arm_scale = alt.Scale(domain=["g0", "g9"], range=[ARM_COLOURS["g0"], ARM_COLOURS["g9"]])
    legend = alt.Legend(
        title=None,
        labelExpr="datum.value == 'g0' ? 'G0 minimal' : 'G9 everything'",
        orient="top",
    )
    panels: list[Panel] = []
    for target in TARGETS:
        scale = _scale(target=target)
        rows = hours.filter(pl.col("target") == target).with_columns(value=pl.col("value") * scale)
        panels.append(
            alt.Chart(rows)
            .mark_line(point=True, strokeWidth=1.5, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour:Q", title="Hour of day (UTC)", axis=alt.Axis(format="d", tickMinStep=1)
                ),
                y=alt.Y("value:Q", title=f"Mean absolute error ({_error_unit(target=target)})"),
                color=alt.Color("treatment:N", scale=arm_scale, legend=legend),
            )
            .properties(width=PLOT_WIDTH_PX, height=120, title=_title(PANEL_TITLES[target]))
        )
    if worst is not None and not worst.is_empty():
        pv_worst = worst.filter(pl.col("target") == "pv")
        rows = pv_worst.with_columns(
            label=pl.format(
                "{} {}: farm {}",
                pl.int_range(1, pv_worst.height + 1),
                pl.col("month"),
                pl.col("site"),
            ),
            g0=pl.col("g0") * 100.0,
            g9=pl.col("g9") * 100.0,
        )
        order = rows["label"].to_list()
        long = rows.unpivot(on=["g0", "g9"], index="label", variable_name="arm", value_name="value")
        panels.append(
            alt.Chart(long)
            .mark_point(filled=True, size=60, opacity=1, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("value:Q", title="Mean absolute error on the day (% of capacity)"),
                y=alt.Y("label:N", sort=order, title=None),
                color=alt.Color("arm:N", scale=arm_scale, legend=legend),
            )
            .properties(
                width=PLOT_WIDTH_PX,
                height=ROW_HEIGHT_PX * len(order),
                title=_title("The minimal set's worst 20 farm-days"),
            )
        )
    return independent_colours(
        chart=figure(
            panels=panels,
            number=11,
            title="Error by hour of day, and the minimal set's worst farm-days",
            subtitle=[
                scope,
                "Top: mean absolute error by hour of day, all farms.",
                (
                    "Bottom: the 20 farm-days with the largest error for the minimal set, "
                    "labelled by rank, month, and farm."
                ),
                "All rows are exploratory.",
            ],
            figure_planning=None,
        )
    )


def main() -> int:
    """Draw every figure whose inputs exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--variant", choices=("main", "snow_zero_hours"), default="main")
    arguments = parser.parse_args()
    through_rung: RungType = arguments.through_rung
    variant: str = arguments.variant

    dataset = pl.read_parquet(dataset_path(through_rung=through_rung, variant=variant))
    paths = report_paths(variant=variant, through_rung=through_rung)
    board = pl.read_parquet(paths.leaderboard)
    contrasts = pl.read_parquet(paths.contrasts)
    splits = pl.read_parquet(paths.splits) if paths.splits.exists() else pl.DataFrame()
    worst = pl.read_parquet(paths.worst_days) if paths.worst_days.exists() else None
    pv_losses = pl.read_parquet(
        results_path(
            key=FitKey(variant=variant, through_rung=through_rung, target="pv", view="ladder")
        )
    )
    first = dataset["time"].min()
    last = dataset["time"].max()
    scope = (
        "Six NGED solar farms, daylight hours, "
        f"{first.strftime('%B %Y')} to {last.strftime('%B %Y')}."  # ty: ignore[unresolved-attribute]
    )

    weather_days = choose_days(
        dataset=dataset, site=min(dataset["site"].unique().to_list()), log_site=False
    )
    farm_days = pl.concat(
        choose_days(dataset=dataset, site=site, exclude=weather_days["date"].to_list())
        for site in sorted(dataset["site"].unique().to_list())
    )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    save(chart=figure_1_headline(contrasts=contrasts, scope=scope), name="headline")
    for target in TARGETS:
        save(
            chart=figure_2_leaderboard(board=board, target=target, scope=scope),
            name=f"leaderboard_{target}",
        )
    for name, chart in (
        ("days", figure_3_days(dataset=dataset, days=weather_days, scope=scope)),
        ("cloud_against_clearness", figure_4_cloud_against_clearness(dataset=dataset, scope=scope)),
        (
            "where_ssrd_misses",
            figure_5_where_ssrd_misses(dataset=dataset, days=weather_days, scope=scope),
        ),
        ("cloud_water", figure_6_cloud_water(dataset=dataset, scope=scope)),
    ):
        if chart is not None:
            save(chart=chart, name=name)
    if not splits.is_empty():
        save(
            chart=figure_7_models_work(
                dataset=dataset, losses=pv_losses, days=farm_days, splits=splits, scope=scope
            ),
            name="models_work",
        )
        cams_note = (
            f"Clear is a CAMS clear-sky index of {CLEAR_SKY_INDEX_THRESHOLDS[1]} or more, and "
            f"overcast is below {CLEAR_SKY_INDEX_THRESHOLDS[0]}. The thresholds were fixed before "
            "any result."
        )
        era5_note = (
            f"Clear is ERA5 total cloud cover below {TOTAL_CLOUD_COVER_THRESHOLDS[0]}, and "
            f"overcast is {TOTAL_CLOUD_COVER_THRESHOLDS[1]} or more. The thresholds were fixed "
            "before any result."
        )
        seasons_by_regime = [f"{season} / {regime}" for season in SEASONS for regime in SKY_REGIMES]
        for number, kind, groups, title, note, name in (
            (8, "regime_cams", SKY_REGIMES, "Contrasts by sky regime", cams_note, "regimes"),
            (
                "8b",
                "regime_era5",
                SKY_REGIMES,
                "Contrasts by sky regime, from ERA5 total cloud cover",
                era5_note,
                "regimes_era5",
            ),
            (9, "season", SEASONS, "Contrasts by season", None, "seasons"),
            (
                "9b",
                "regime_cams_by_season",
                seasons_by_regime,
                "Contrasts by season and sky regime",
                cams_note,
                "regimes_by_season",
            ),
        ):
            save(
                chart=figure_regimes(
                    splits=splits,
                    number=number,
                    kind=kind,
                    groups=groups,
                    title=title,
                    regime_note=note,
                    scope=scope,
                ),
                name=name,
            )
    if (contrasts["family"] == "drop_one").any():
        save(chart=figure_drop_one(contrasts=contrasts, scope=scope), name="drop_one")
    figure_11 = figure_11_hour_and_worst_days(splits=splits, worst=worst, scope=scope)
    if figure_11 is not None:
        save(chart=figure_11, name="hour_and_worst_days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
