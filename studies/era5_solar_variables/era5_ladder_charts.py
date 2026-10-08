"""Draw the figures of the ERA5 variable ladder page from the report's tables and the built frame.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It reads the tables that
`era5_ladder_report.py` wrote (`leaderboard.parquet`, `contrasts.parquet`, `splits.parquet`), the
frame `era5_ladder_build_dataset.py` wrote, and, for the figure that shows the models working, the
saved per-row losses. It refits nothing. It writes one SVG per figure into `docs/studies/assets/`,
named `era5_solar_variables_<name>.svg`.

**A figure whose inputs have not been built yet is skipped and logged**, so the script runs on the
first rungs while the later variables are still downloading.

**Every chart is anonymised.** Farms are labelled A to F, outputs are fractions of each farm's own
capacity, no coordinate appears, and no calendar date appears on an axis of a farm's time series:
the days are chosen by a stated rule and drawn against the hour of the day. The month and year of
each chosen day are printed to the log for the page's text, and the day of the month never is.

**The three days are chosen on one farm by the CAMS clear-sky index of its daylight hours.** The
clearest day has the highest mean index, the dullest day the lowest, and the most variable day the
highest standard deviation of the hourly index. Only days with at least `MIN_DAYLIGHT_HOURS`
daylight hours on the farm count.

Run it with `uv run --with vl-convert-python python
studies/era5_solar_variables/era5_ladder_charts.py`. The SVGs then go through `svgo` before they
are committed.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
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
    RESULTS_DIR,
    SIGNED_ERROR,
    SMALLEST_EFFECT,
    TARGETS,
    FitKey,
    TargetType,
    dataset_path,
    results_path,
)
from studies.charts import LABEL_WIDTH_PX, PLOT_WIDTH_PX, Panel, figure
from studies.era5_ladder import (
    CLEAR_SKY_INDEX_THRESHOLDS,
    RUNGS,
    SEASONS,
    SKY_REGIMES,
    RungType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_charts")

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the page's SVGs live."""

FILE_PREFIX: Final[str] = "era5_solar_variables_"
"""The start of every SVG's file name."""

MIN_DAYLIGHT_HOURS: Final[int] = 8
"""A day counts for the three-day figures only if the chosen farm has this many daylight hours."""

TARGET_NAMES: Final[dict[TargetType, str]] = {
    "pv": "solar farms' output",
    "cams": "CAMS clearness index",
}
"""How a target is named in a chart."""

TARGET_COLOURS: Final[dict[TargetType, str]] = {"pv": ocf.BRAND_ORANGE, "cams": ocf.DATA_BLUE}
"""One colour per target, so a reader tracks the target across figures."""

CONTRAST_COLOURS: Final[dict[str, str]] = {
    "g1": ocf.DATA_SKY,
    "g2": ocf.DATA_BLUE,
    "g9": ocf.BRAND_ORANGE,
}
"""The colour of each contrast against the minimal set in the regime and season figures."""

CONTRAST_NAMES: Final[dict[str, str]] = {
    "g1": "total cloud (G1)",
    "g2": "three cloud layers (G2)",
    "g9": "every ERA5 variable (G9)",
}
"""What each contrast against the minimal set adds, in words."""

SERIES_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
)
"""The first colours of the brand palette, which a series takes in the order it is listed."""

ROW_HEIGHT_PX: Final[int] = 26
"""The height of one row of a dot-and-interval panel."""

DAY_PANEL_WIDTH_PX: Final[int] = 160
"""The width of one day in a three-day small multiple."""

DAY_PANEL_HEIGHT_PX: Final[int] = 90
"""The height of one variable's row in a three-day small multiple."""

REGIME_DOMAIN_PADDING: Final[float] = 0.1
"""How far beyond the widest interval a dot-and-interval panel's axis extends."""


def _scale(*, target: TargetType) -> float:
    """Return what turns the metric into the chart's unit: percentage points, or clearness index."""
    return 100.0 if target == "pv" else 1.0


ARM_LABEL_EXPRESSION: Final[str] = (
    "datum.value == 'g0' ? 'G0 minimal' : datum.value == 'g1' ? 'G1 total cloud' : "
    "datum.value == 'g2' ? 'G2 cloud layers' : 'G9 everything'"
)
"""The Vega expression that turns the arm names `g0`, `g1`, `g2`, and `g9` into words."""


def _error_unit(*, target: TargetType) -> str:
    """Return the unit of a mean absolute error on the chart."""
    return "% of capacity" if target == "pv" else "clearness index"


def _unit(*, target: TargetType) -> str:
    """Return the unit of a difference in the metric."""
    return "points of capacity" if target == "pv" else "clearness index"


def _domain(*, values: Sequence[float]) -> list[float]:
    """Return an x domain that holds every value, with `REGIME_DOMAIN_PADDING` of room each side."""
    low, high = min(values), max(values)
    pad = (high - low) * REGIME_DOMAIN_PADDING
    return [low - pad, high + pad]


def dot_interval_panel(
    *,
    rows: pl.DataFrame,
    x_title: str,
    panel_title: str,
    colour: str | None = None,
    reference_rules: Sequence[float] = (0.0,),
    order: Sequence[str] | None = None,
) -> Panel:
    """Draw one row per label: a dot at its estimate, a thin 95% line, and an optional thick line.

    Args:
        rows: One row per mark, with `label`, `value`, `lower`, `upper`, and optionally
            `thick_lower` and `thick_upper` (an adjusted interval), and `colour` if `colour` is
            not given.
        x_title: The axis title, naming the quantity, the unit, and the better direction.
        panel_title: The title above the panel.
        colour: One colour for every row, or `None` to read each row's `colour` column.
        reference_rules: The x values that get a dashed vertical rule, such as zero.
        order: The labels from top to bottom, or `None` for the order of `rows`.

    Returns:
        The layered panel.
    """
    labels = list(order) if order is not None else rows["label"].to_list()
    extremes = [
        *rows["lower"].to_list(),
        *rows["upper"].to_list(),
        *(rows["thick_lower"].to_list() if "thick_lower" in rows.columns else []),
        *(rows["thick_upper"].to_list() if "thick_upper" in rows.columns else []),
        *reference_rules,
    ]
    domain = _domain(values=extremes)
    x = alt.X("value:Q", title=x_title, scale=alt.Scale(domain=domain, nice=False))
    y = alt.Y(
        "label:N",
        sort=labels,
        title=None,
        axis=alt.Axis(labelLimit=LABEL_WIDTH_PX, labelPadding=6, domain=False, ticks=False),
    )
    colour_encoding = (
        alt.value(colour) if colour is not None else alt.Color("colour:N", scale=None, legend=None)
    )
    layers: list[alt.Chart] = [
        alt.Chart(pl.DataFrame({"x": list(reference_rules)}))
        .mark_rule(color=ocf.TEXT, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ]
    if "thick_lower" in rows.columns:
        layers.append(
            alt.Chart(rows)
            .mark_rule(strokeWidth=6, opacity=0.35, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y, x="thick_lower:Q", x2="thick_upper:Q", color=colour_encoding
            )
        )
    layers.append(
        alt.Chart(rows)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(y=y, x="lower:Q", x2="upper:Q", color=colour_encoding)  # ty: ignore[unresolved-attribute]
    )
    layers.append(
        alt.Chart(rows)
        .mark_point(filled=True, size=70, opacity=1, aria=False)
        .encode(y=y, x=x, color=colour_encoding)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(*layers).properties(
        width=PLOT_WIDTH_PX,
        height=ROW_HEIGHT_PX * len(labels),
        title=alt.TitleParams(
            panel_title, anchor="start", fontSize=ocf.font_size(style="Body Large", body_px=11)
        ),
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart to `docs/studies/assets/` as an SVG."""
    path = ASSETS_DIR / f"{FILE_PREFIX}{name}.svg"
    chart.save(path)
    _LOG.info("wrote %s", path)


def figure_1_headline(*, contrasts: pl.DataFrame) -> alt.TopLevelMixin:
    """Draw the planned contrasts on both targets, at the adjusted level and at 95%.

    Args:
        contrasts: The report's contrast table.

    Returns:
        The figure.
    """
    panels = []
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
                "{}: {} minus {}", "label", "treatment", "reference"
            ).str.to_uppercase(),
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
                    f"Mean absolute error minus the reference's ({_unit(target=target)}; "
                    "more negative is better)"
                ),
                panel_title=TARGET_NAMES[target].capitalize(),
                colour=TARGET_COLOURS[target],
                reference_rules=(0.0, -smallest),
            )
        )
    return figure(
        panels=panels,
        number=1,
        title="Adding ERA5 variables to the minimal set changes the error by the amounts shown",
        subtitle=[
            "Dot: estimate. Thin line: 95% interval from resampling whole months.",
            (
                f"Thick line: {ADJUSTED_LEVEL_PERCENT:.3f}% interval, adjusted for the eight "
                "planned contrasts."
            ),
            "Solid rule: no difference. Dashed rule: the smallest improvement worth acting on.",
            "All rows are planned: written into the study plan before any result existed.",
        ],
        figure_planning=None,
    )


def figure_2_leaderboard(*, board: pl.DataFrame, target: TargetType) -> alt.TopLevelMixin:
    """Draw every arm's mean absolute error and correlation, best first.

    Args:
        board: The report's leaderboard table.
        target: The target.

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
    unit = "% of capacity" if target == "pv" else "clearness index"
    return figure(
        panels=[
            dot_interval_panel(
                rows=error,
                x_title=f"Mean absolute error ({unit}; smaller is better)",
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
        title=f"Every arm's error and correlation for the {TARGET_NAMES[target]}",
        subtitle=[
            (
                "Dot: estimate at the first hyperparameter setting. Line: 95% interval from "
                "resampling whole months."
            ),
            (
                "The arms share their rows, so these intervals overlap more than the paired "
                "differences of figure 1 do."
            ),
            "All rows are exploratory.",
        ],
        figure_planning=None,
    )


def _split_panel(
    *, splits: pl.DataFrame, target: TargetType, kind: str, order: Sequence[str]
) -> Panel:
    """Draw one split's contrasts against the minimal set, three dots per group."""
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
    y = alt.Y("label:N", sort=list(order), title=None, axis=alt.Axis(labelLimit=LABEL_WIDTH_PX))
    colour = alt.Color(
        "contrast:N",
        scale=alt.Scale(domain=list(CONTRAST_COLOURS), range=list(CONTRAST_COLOURS.values())),
        legend=alt.Legend(title=None, labelExpr=_legend_expression()),
    )
    offset = alt.YOffset("contrast:N", scale=alt.Scale(domain=list(CONTRAST_COLOURS)))
    x = alt.X(
        "value:Q",
        title=(
            f"Mean absolute error minus the minimal set's ({_unit(target=target)}; "
            "more negative is better)"
        ),
        scale=alt.Scale(domain=domain, nice=False),
    )
    return alt.layer(
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.TEXT, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(selected)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(y=y, yOffset=offset, x="lower:Q", x2="upper:Q", color=colour),  # ty: ignore[unresolved-attribute]
        alt.Chart(selected)
        .mark_point(filled=True, size=60, opacity=1, aria=False)
        .encode(y=y, yOffset=offset, x=x, color=colour),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=PLOT_WIDTH_PX,
        height=int(ROW_HEIGHT_PX * len(order) * 1.6),
        title=alt.TitleParams(TARGET_NAMES[target].capitalize(), anchor="start"),
    )


def _legend_expression() -> str:
    """Return the Vega expression that turns a contrast's arm name into its words."""
    cases = " : ".join(f"datum.value == '{arm}' ? '{name}'" for arm, name in CONTRAST_NAMES.items())
    return f"{cases} : datum.value"


def figure_regimes(
    *, splits: pl.DataFrame, number: int | str, kind: str, groups: Sequence[str], title: str
) -> alt.TopLevelMixin:
    """Draw the contrasts against the minimal set within each weather regime or season.

    Args:
        splits: The report's split table.
        number: The figure's number.
        kind: The split, such as `regime_cams` or `season`.
        groups: The groups from top to bottom.
        title: The finding the figure shows.

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
            "Dot: estimate. Line: 95% interval from resampling whole months.",
            "Dashed rule: no difference.",
            (
                f"Clear is a CAMS clear-sky index of {CLEAR_SKY_INDEX_THRESHOLDS[1]} or more, "
                f"and overcast is below {CLEAR_SKY_INDEX_THRESHOLDS[0]}. The thresholds were "
                "fixed before any result."
            ),
            "All rows are exploratory and are not corrected for multiple comparisons.",
        ],
        figure_planning=None,
    )


def figure_drop_one(*, contrasts: pl.DataFrame) -> alt.TopLevelMixin:
    """Draw the error added when each group is removed from the full set.

    Args:
        contrasts: The report's contrast table.

    Returns:
        The figure, one panel per target.
    """
    panels = []
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
            + pl.lit(" removed"),
            value=pl.col("difference") * scale,
            lower=pl.col("lower_95") * scale,
            upper=pl.col("upper_95") * scale,
        )
        panels.append(
            dot_interval_panel(
                rows=rows,
                x_title=(
                    f"Mean absolute error minus the full set's ({_unit(target=target)}; "
                    "positive means the group helped)"
                ),
                panel_title=TARGET_NAMES[target].capitalize(),
                colour=TARGET_COLOURS[target],
            )
        )
    return figure(
        panels=panels,
        number=10,
        title="Removing one group of variables from the full set changes the error as shown",
        subtitle=[
            "Dot: estimate. Line: 95% interval from resampling whole months.",
            "Dashed rule: no difference.",
            "A group counts as useful only if the ladder (figure 2) and this figure agree.",
            "All rows are exploratory and are not corrected for multiple comparisons.",
        ],
        figure_planning=None,
    )


def choose_days(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Choose the clearest, most variable, and dullest day on the first farm by the stated rule.

    Args:
        dataset: The kept rows.

    Returns:
        One row per chosen day with `day_label`, `site`, and `date`.
    """
    site = min(dataset["site"].unique().to_list())
    daily = (
        dataset.filter((pl.col("site") == site) & pl.col("cams_clear_sky_index").is_not_null())
        .with_columns(date=pl.col("time").dt.date())
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
        _LOG.info("%s: farm %s, %s", label, site, row["date"].strftime("%B %Y"))
    return pl.DataFrame(
        {
            "day_label": [label for label, _ in chosen],
            "site": [site] * 3,
            "date": [row["date"] for _, row in chosen],
        }
    )


SERIES_PANELS: Final[tuple[tuple[str, tuple[str, ...], str], ...]] = (
    ("Irradiance", ("ssrd", "ssrdc", "cams_ghi_w_m2", "fdir"), "W m⁻²"),
    ("Cloud cover", ("tcc", "lcc", "mcc", "hcc"), "fraction"),
    ("Cloud water", ("tclw", "tciw"), "kg m⁻²"),
    ("Cloud base", ("cbh",), "m"),
)
"""The variable groups of the three-day figure as (panel title, columns, unit)."""


def figure_3_days(*, dataset: pl.DataFrame, days: pl.DataFrame) -> alt.TopLevelMixin | None:
    """Draw the variables for the three chosen days, one panel per variable group.

    Args:
        dataset: The kept rows.
        days: The chosen days.

    Returns:
        The figure, or `None` if the frame lacks the cloud variables.
    """
    needed = {name for _, columns, _ in SERIES_PANELS for name in columns}
    if not needed <= set(dataset.columns):
        _LOG.info("figure 3 skipped: the frame lacks %s", sorted(needed - set(dataset.columns)))
        return None
    site = days["site"][0]
    rows = dataset.filter(pl.col("site") == site).with_columns(date=pl.col("time").dt.date())
    panels = []
    farm_power = rows.with_columns(
        pv_percent=pl.col("power_mw") / pl.col("effective_capacity_mw") * 100.0
    )
    groups = [*SERIES_PANELS, ("Solar farm output", ("pv_percent",), "% of capacity")]
    for title, columns, unit in groups:
        long = pl.concat(
            farm_power.join(days.select("date", "day_label"), on="date").select(
                "day_label", "hour_of_day", series=pl.lit(column), value=pl.col(column)
            )
            for column in columns
        )
        panels.append(
            alt.Chart(long)
            .mark_line(strokeWidth=1.5, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour_of_day:Q", title="Hour of day (UTC)", scale=alt.Scale(domain=[0, 24])
                ),
                y=alt.Y("value:Q", title=unit),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(range=list(SERIES_COLOURS)),
                    legend=alt.Legend(title=title),
                ),
            )
            .properties(width=DAY_PANEL_WIDTH_PX, height=DAY_PANEL_HEIGHT_PX)
            .facet(column=alt.Column("day_label:N", sort=list(days["day_label"]), title=None))
        )
    return figure(
        panels=panels,
        number=3,
        title="What the ERA5 variables, CAMS, and one farm's output look like on three days",
        subtitle=[
            (
                "Three days at one farm, chosen by the CAMS clear-sky index of the daylight "
                "hours: the highest mean, the highest spread, and the lowest mean."
            ),
            "Lines: the quantity named in each panel's legend. Output is a percentage of capacity.",
        ],
        figure_planning=None,
    )


def figure_4_cloud_against_clearness(*, dataset: pl.DataFrame) -> alt.TopLevelMixin | None:
    """Draw CAMS clearness index against each ERA5 cloud cover, as binned means with a spread band.

    Args:
        dataset: The kept rows.

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
        .filter(pl.col("n") >= 30)
    )
    chart = (
        alt.layer(
            alt.Chart(summary)
            .mark_area(opacity=0.25, color=ocf.DATA_BLUE, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "bin:Q", title="ERA5 cloud cover (fraction)", scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y("lower:Q", title="CAMS clearness index"),
                y2="upper:Q",
            ),
            alt.Chart(summary)
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=2, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x="bin:Q", y="mean:Q"
            ),
        )
        .properties(width=PLOT_WIDTH_PX // 2, height=130)
        .facet(
            facet=alt.Facet(
                "cover:N",
                sort=list(covers),
                title=None,
                header=alt.Header(labelExpr=_cover_label_expression()),
            ),
            columns=2,
        )
    )
    return figure(
        panels=[chart],
        number=4,
        title="CAMS clearness index falls as each ERA5 cloud cover rises",
        subtitle=[
            "Line: mean clearness index in each 0.05-wide band of cloud cover.",
            "Band: 10th to 90th percentile.",
            "All farms, all kept daylight hours.",
        ],
        figure_planning=None,
    )


def _cover_label_expression() -> str:
    """Return the Vega expression that names each cloud cover in words."""
    names = {"tcc": "Total cloud cover", "lcc": "Low", "mcc": "Medium", "hcc": "High"}
    cases = " : ".join(f"datum.value == '{key}' ? '{name}'" for key, name in names.items())
    return f"{cases} : datum.value"


CANDIDATES: Final[tuple[tuple[str, str], ...]] = (
    ("tclw", "Cloud liquid water (kg m⁻²)"),
    ("tciw", "Cloud ice water (kg m⁻²)"),
    ("cbh", "Cloud base height (m)"),
    ("tcwv", "Water vapour column (kg m⁻²)"),
    ("d2m", "Dew point (K)"),
    ("blh", "Boundary layer height (m)"),
    ("sd", "Snow depth (m of water)"),
)
"""The candidate variables figure 5 bins the ERA5-minus-CAMS gap against, with their axis titles."""


def figure_5_where_ssrd_misses(*, dataset: pl.DataFrame) -> alt.TopLevelMixin | None:
    """Draw how far ERA5's `ssrd` is from CAMS global irradiance, binned by each candidate variable.

    Args:
        dataset: The kept rows.

    Returns:
        The figure, or `None` if the frame lacks `ssrd` or any candidate variable.
    """
    present = [(name, title) for name, title in CANDIDATES if name in dataset.columns]
    if not present:
        _LOG.info("figure 5 skipped: the frame holds none of the candidate variables")
        return None
    gap = dataset.with_columns(gap=pl.col("ssrd") - pl.col("cams_ghi_w_m2"))
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
            .agg(x=pl.col(name).mean(), mean_gap=pl.col("gap").mean(), n=pl.len())
            .select(variable=pl.lit(title), x="x", mean_gap="mean_gap")
        )
    summary = pl.concat(summaries)
    chart = (
        alt.Chart(summary)
        .mark_line(point=True, color=ocf.BRAND_ORANGE, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", title=None, scale=alt.Scale(zero=False)),
            y=alt.Y("mean_gap:Q", title="ERA5 ssrd minus CAMS (W m⁻²)"),
        )
        .properties(width=PLOT_WIDTH_PX // 2, height=110)
        .resolve_scale(x="independent")
        .facet(
            facet=alt.Facet("variable:N", title=None, sort=[title for _, title in present]),
            columns=2,
        )
        .resolve_scale(x="independent")
    )
    return figure(
        panels=[chart],
        number=5,
        title="Where ERA5's downward solar radiation differs from CAMS",
        subtitle=[
            (
                "Each dot: the mean of ERA5 ssrd minus CAMS global irradiance over one tenth "
                "of the kept hours, ranked by the variable named above the panel."
            ),
            "Positive means ERA5 is brighter than CAMS. Daylight hours at all farms.",
        ],
        figure_planning=None,
    )


def figure_6_cloud_water(*, dataset: pl.DataFrame) -> alt.TopLevelMixin | None:
    """Draw clearness index against total cloud water, for hours of low, medium, and high low-cloud.

    Args:
        dataset: The kept rows.

    Returns:
        The figure, or `None` if the frame lacks cloud water or low cloud cover.
    """
    needed = {"tclw", "tciw", "lcc"}
    if not needed <= set(dataset.columns):
        _LOG.info("figure 6 skipped: the frame lacks %s", sorted(needed - set(dataset.columns)))
        return None
    rows = dataset.filter(pl.col("cams_clearness_index").is_not_null()).with_columns(
        water=pl.col("tclw") + pl.col("tciw"),
        low_cloud=pl.when(pl.col("lcc") < 0.2)
        .then(pl.lit("Low cloud under 0.2"))
        .when(pl.col("lcc") < 0.6)
        .then(pl.lit("Low cloud 0.2 to 0.6"))
        .otherwise(pl.lit("Low cloud 0.6 or more")),
    )
    banded = rows.with_columns(
        band=(pl.col("water").rank(method="average") / rows.height * 15).floor().clip(0, 14)
    )
    summary = banded.group_by("low_cloud", "band").agg(
        water=pl.col("water").mean(), mean_index=pl.col("cams_clearness_index").mean()
    )
    chart = (
        alt.Chart(summary)
        .mark_line(point=True, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "water:Q",
                title="ERA5 cloud liquid plus ice water (kg m⁻²)",
                scale=alt.Scale(type="symlog"),
            ),
            y=alt.Y("mean_index:Q", title="CAMS clearness index"),
            color=alt.Color(
                "low_cloud:N",
                scale=alt.Scale(
                    domain=["Low cloud under 0.2", "Low cloud 0.2 to 0.6", "Low cloud 0.6 or more"],
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=PLOT_WIDTH_PX + LABEL_WIDTH_PX - 60, height=200)
    )
    return figure(
        panels=[chart],
        number=6,
        title="More cloud water means less sunlight reaches the ground, at any low-cloud cover",
        subtitle=[
            "Each dot: the mean over one fifteenth of the kept hours, ranked by cloud water.",
            "Daylight hours at all farms.",
        ],
        figure_planning=None,
    )


def predictions_for_days(
    *, dataset: pl.DataFrame, losses: pl.DataFrame, days: pl.DataFrame
) -> pl.DataFrame:
    """Return measured and out-of-fold output as a fraction of capacity for the chosen days.

    Args:
        dataset: The kept rows.
        losses: The output target's losses.
        days: The chosen days.

    Returns:
        Rows of `site`, `day_label`, `hour_of_day`, `series`, and `value`, in percent of capacity,
        for the measured output and the first seed's `g0` and `g9` predictions.
    """
    arms = ["g0", "g9"]
    base = (
        dataset.with_columns(date=pl.col("time").dt.date())
        .join(days.select("date", "day_label"), on="date")
        .select("site", "time", "day_label", "hour_of_day", "power_mw", "effective_capacity_mw")
    )
    measured = base.select(
        "site",
        "day_label",
        "hour_of_day",
        series=pl.lit("Measured"),
        value=pl.col("power_mw") / pl.col("effective_capacity_mw") * 100.0,
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
            value=(pl.col("power_mw") + pl.col(SIGNED_ERROR))
            / pl.col("effective_capacity_mw")
            * 100.0,
        )
        for arm in arms
    ]
    return pl.concat([measured, *predicted])


def figure_7_models_work(
    *, dataset: pl.DataFrame, losses: pl.DataFrame, days: pl.DataFrame, splits: pl.DataFrame
) -> alt.TopLevelMixin:
    """Draw measured and predicted output on the chosen days for every farm, and each farm's error.

    Args:
        dataset: The kept rows.
        losses: The output target's losses.
        days: The chosen days.
        splits: The report's split table, for each farm's error.

    Returns:
        The figure.
    """
    series = predictions_for_days(dataset=dataset, losses=losses, days=days)
    measured_label = "Measured"
    timeline = (
        alt.Chart(series)
        .mark_line(strokeWidth=1.25, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour_of_day:Q", title="Hour of day (UTC)", scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("value:Q", title="% of capacity"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[measured_label, ARM_LABELS["g0"], ARM_LABELS["g9"]],
                    range=[ocf.TEXT, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=DAY_PANEL_WIDTH_PX, height=70)
        .facet(
            row=alt.Row("site:N", title="Farm"),
            column=alt.Column("day_label:N", sort=list(days["day_label"]), title=None),
        )
    )
    arms = ("g0", "g1", "g2", "g9")
    per_farm = splits.filter(
        (pl.col("target") == "pv") & (pl.col("split") == "farm") & pl.col("treatment").is_in(arms)
    ).select(
        "group",
        arm=pl.col("treatment"),
        value=pl.col("value") * 100.0,
    )
    error = (
        alt.Chart(per_farm)
        .mark_point(filled=True, size=70, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            y=alt.Y("group:N", title="Farm"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(arms),
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(
                    title=None,
                    labelExpr=ARM_LABEL_EXPRESSION,
                ),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=160)
    )
    return figure(
        panels=[timeline, error],
        number=7,
        title="The XGBoost models track each farm's output, and farms differ more than arms do",
        subtitle=[
            "Top: out-of-fold predictions on the three days of figure 3, at every farm.",
            "Bottom: each farm's mean absolute error for four arms.",
            "Output is a percentage of each farm's own capacity.",
        ],
        figure_planning=None,
    )


def figure_11_hour_and_worst_days(
    *, splits: pl.DataFrame, worst: pl.DataFrame | None
) -> alt.TopLevelMixin | None:
    """Draw error by hour of day for the minimal and full arms, and the minimal arm's worst days.

    Args:
        splits: The report's split table.
        worst: The worst farm-days table, or `None`.

    Returns:
        The figure, or `None` if the splits hold no hour-of-day rows.
    """
    hours = splits.filter(
        (pl.col("split") == "hour_of_day") & pl.col("treatment").is_in(["g0", "g9"])
    ).with_columns(hour=pl.col("group").cast(pl.Int64), value=pl.col("value"))
    if hours.is_empty():
        return None
    panels = []
    for target in TARGETS:
        scale = _scale(target=target)
        rows = hours.filter(pl.col("target") == target).with_columns(value=pl.col("value") * scale)
        panels.append(
            alt.Chart(rows)
            .mark_line(point=True, strokeWidth=1.5, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("hour:Q", title="Hour of day (UTC)"),
                y=alt.Y(
                    "value:Q",
                    title=f"Mean absolute error ({_error_unit(target=target)})",
                ),
                color=alt.Color(
                    "treatment:N",
                    scale=alt.Scale(domain=["g0", "g9"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(
                        title=None, labelExpr="datum.value == 'g0' ? 'G0 minimal' : 'G9 everything'"
                    ),
                ),
            )
            .properties(
                width=PLOT_WIDTH_PX,
                height=120,
                title=alt.TitleParams(TARGET_NAMES[target].capitalize(), anchor="start"),
            )
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
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=["g0", "g9"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(
                        title=None, labelExpr="datum.value == 'g0' ? 'G0 minimal' : 'G9 everything'"
                    ),
                ),
            )
            .properties(
                width=PLOT_WIDTH_PX,
                height=ROW_HEIGHT_PX * len(order),
                title=alt.TitleParams("The minimal set's worst 20 farm-days", anchor="start"),
            )
        )
    return figure(
        panels=panels,
        number=11,
        title="The error is largest around midday, and the worst days are not the snow days",
        subtitle=[
            "Top: mean absolute error by hour of day, all farms.",
            (
                "Bottom: the 20 farm-days with the largest error for the minimal set, labelled by "
                "rank, month, and farm."
            ),
            "All rows are exploratory.",
        ],
        figure_planning=None,
    )


def main() -> int:
    """Draw every figure whose inputs exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--variant", default="main")
    arguments = parser.parse_args()
    through_rung: RungType = arguments.through_rung

    dataset = pl.read_parquet(dataset_path(through_rung=through_rung, variant=arguments.variant))
    board = pl.read_parquet(RESULTS_DIR / "leaderboard.parquet")
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
    splits_path = RESULTS_DIR / "splits.parquet"
    splits = pl.read_parquet(splits_path) if splits_path.exists() else pl.DataFrame()
    pv_losses = pl.read_parquet(
        results_path(
            key=FitKey(
                variant=arguments.variant, through_rung=through_rung, target="pv", view="ladder"
            )
        )
    )
    days = choose_days(dataset=dataset)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    save(chart=figure_1_headline(contrasts=contrasts), name="headline")
    for target in TARGETS:
        save(chart=figure_2_leaderboard(board=board, target=target), name=f"leaderboard_{target}")
    for name, chart in (
        ("days", figure_3_days(dataset=dataset, days=days)),
        ("cloud_against_clearness", figure_4_cloud_against_clearness(dataset=dataset)),
        ("where_ssrd_misses", figure_5_where_ssrd_misses(dataset=dataset)),
        ("cloud_water", figure_6_cloud_water(dataset=dataset)),
    ):
        if chart is not None:
            save(chart=chart, name=name)
    if not splits.is_empty():
        save(
            chart=figure_7_models_work(dataset=dataset, losses=pv_losses, days=days, splits=splits),
            name="models_work",
        )
        save(
            chart=figure_regimes(
                splits=splits,
                number=8,
                kind="regime_cams",
                groups=list(SKY_REGIMES),
                title="Extra variables help most, if at all, under broken cloud",
            ),
            name="regimes",
        )
        save(
            chart=figure_regimes(
                splits=splits,
                number="8b",
                kind="regime_era5",
                groups=list(SKY_REGIMES),
                title="The same split by ERA5 total cloud cover",
            ),
            name="regimes_era5",
        )
        save(
            chart=figure_regimes(
                splits=splits,
                number=9,
                kind="season",
                groups=list(SEASONS),
                title="Extra variables help differently in each season",
            ),
            name="seasons",
        )
        save(
            chart=figure_regimes(
                splits=splits,
                number="9b",
                kind="regime_cams_by_season",
                groups=[f"{season} / {regime}" for season in SEASONS for regime in SKY_REGIMES],
                title="The regime split within each season",
            ),
            name="regimes_by_season",
        )
    if (contrasts["family"] == "drop_one").any():
        save(chart=figure_drop_one(contrasts=contrasts), name="drop_one")
    worst = None
    report_worst = RESULTS_DIR / "worst_days.parquet"
    if report_worst.exists():
        worst = pl.read_parquet(report_worst)
    figure_11 = figure_11_hour_and_worst_days(splits=splits, worst=worst)
    if figure_11 is not None:
        save(chart=figure_11, name="hour_and_worst_days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
