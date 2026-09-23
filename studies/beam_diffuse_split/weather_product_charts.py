"""Draw the eleven anonymised charts for the write-up on which product best describes sunshine.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-solar/>.

**Every number a chart shares with the report is read from the report `weather_products.py`
wrote**, so a chart cannot disagree with the page. Three charts also draw numbers the report does
not print, computed from `losses.parquet` without refitting any model: the leaderboard's intervals,
and the two "models work" charts' out-of-fold predictions and per-generator errors.

Generators appear only as `A` to `F`. No chart plots output in megawatts or carries a calendar date
beside a generator's output.

Run it with `uv run python studies/beam_diffuse_split/weather_product_charts.py`, after
`weather_products.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import calendar
import logging
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from sources import STUDY_DATA_DIR
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    CONTENT_WIDTH_PX,
    FAMILY_COLOURS,
    LABEL_WIDTH_PX,
    PLOT_WIDTH_PX,
    ContrastKey,
    ProductFamily,
    figure,
    flip_contrast,
    interval_panel,
    leaderboard_panel,
    planning,
    report_contrasts,
    report_errors,
    select_contrasts,
    ticks,
)
from weather_products import (
    METRIC,
    OUTPUT_DIR_NAME,
    PERCENTAGE_POINTS,
    common_rows,
    joined,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = STUDY_DATA_DIR / OUTPUT_DIR_NAME
"""Where `weather_products.py` wrote its report and losses."""

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "cams": "CAMS",
    "era5": "ERA5",
    "ukv": "UKV",
    "icon_d2": "ICON-D2",
    "icon_eu": "ICON-EU",
    "icon_global": "ICON global",
}
"""Each product's name as the page writes it."""

FAMILIES: Final[dict[str, ProductFamily]] = {
    "cams": "satellite",
    "era5": "reanalysis",
    "ukv": "weather model",
    "icon_d2": "weather model",
    "icon_eu": "weather model",
    "icon_global": "weather model",
}
"""Each product's family, which sets its colour."""

PANEL_WIDTH_PX: Final[int] = (CONTENT_WIDTH_PX - 32) // 3
"""One "models work" time-series panel's width: 3 weeks side by side, 16 px apart."""

PANEL_HEIGHT_PX: Final[int] = 110
"""One "models work" time-series panel's height."""

MINUTES_PER_DAY: Final[int] = 24 * 60

MODELS_WORK_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""Solar week selection is restricted to these months, as `make_figures.py` does for the
beam/diffuse study: a midwinter week has too few daylight hours to tell a clear day from a dull
one.
"""

SOLAR_WEEK_CRITERIA: Final[tuple[tuple[str, str, bool], ...]] = (
    ("Clearest week", "mean_output", True),
    ("Dullest week", "mean_output", False),
    ("Most variable week", "spread", True),
)
"""Solar weeks are picked in this order, each excluding the weeks already picked."""

SOLAR_WEEK_DISPLAY_ORDER: Final[tuple[str, ...]] = (
    "Clearest week",
    "Most variable week",
    "Dullest week",
)
"""Solar weeks are drawn easiest to hardest, left to right."""


def _reconstruct_predicted(
    *, losses: pl.DataFrame, measured: pl.DataFrame, arm: str
) -> pl.DataFrame:
    """Reconstruct one arm's scored out-of-fold prediction, averaged over its three fitting seeds.

    `signed_error_capped_mw` is the prediction, held to the export cap as every score on the page
    holds it, minus the measured value (`studies.cross_validation._losses`), so adding it back to
    the measured value recovers the scored prediction. `losses` carries no measured value of its
    own, so it is joined from the rows the study actually scored.

    Args:
        losses: Rows from `losses.parquet`, holding `arm`.
        measured: One row per (site, time) with `power_mw`, from the rows the study scored.
        arm: The arm being reconstructed.

    Returns:
        One row per (site, time) with `predicted_mw`.

    Raises:
        ValueError: If a loss row has no matching measured row.
    """
    arm_losses = losses.filter(pl.col("arm") == arm)
    joined = arm_losses.select("site", "time", "signed_error_capped_mw").join(
        measured.select("site", "time", "power_mw"), on=["site", "time"], how="inner"
    )
    if joined.height != arm_losses.height:
        msg = (
            f"{arm}: {arm_losses.height} loss rows but only {joined.height} matched a measured row"
        )
        raise ValueError(msg)
    return joined.group_by("site", "time").agg(
        predicted_mw=(pl.col("power_mw") + pl.col("signed_error_capped_mw")).mean()
    )


def _models_work_long_frame(
    *, measured: pl.DataFrame, predicted: tuple[tuple[pl.DataFrame, str], ...]
) -> pl.DataFrame:
    """Join measured and predicted power into one long frame, as a percentage of capacity.

    Args:
        measured: One row per (site, time) with `power_mw` and `effective_capacity_mw`.
        predicted: (predictions, series label) pairs, each from `_reconstruct_predicted`.

    Returns:
        One row per (site, time, series) with `percent`.
    """
    capacity = measured.select("site", "time", "effective_capacity_mw")
    rows = [
        measured.select(
            "site",
            "time",
            percent=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw") * 100,
        ).with_columns(series=pl.lit("Measured"))
    ]
    rows.extend(
        frame.join(capacity, on=["site", "time"], how="inner")
        .select(
            "site", "time", percent=pl.col("predicted_mw") / pl.col("effective_capacity_mw") * 100
        )
        .with_columns(series=pl.lit(label))
        for frame, label in predicted
    )
    return pl.concat(rows)


def _pick_weeks(
    *,
    hourly: pl.DataFrame,
    min_hours: int,
    agg: str,
    criteria: tuple[tuple[str, str, bool], ...],
) -> pl.DataFrame:
    """Pick distinct weeks from an hourly output-fraction frame, pooled across every generator.

    No weather product's own values enter this choice, so it cannot favour one: `output_frac` is
    measured power over the generator's own capacity. A week is a candidate only if every
    generator has a full 7 days of eligible hours in it.

    Args:
        hourly: One row per (site, time) with `output_frac`, already restricted to the hours and
            months eligible for selection.
        min_hours: The minimum eligible hours a (site, day) needs to count towards that day.
        agg: `"sum"` for a daily total (solar) or `"mean"` for a daily average (wind).
        criteria: (name, column, descending) triples, each picked in turn, excluding weeks already
            picked, so every pick is distinct. `column` is `mean_output` or `spread`.

    Returns:
        One row per chosen week, with `week` and `label`, the criterion's name. The label carries
        no date, so a published chart cannot be matched against public generation data.

    Raises:
        ValueError: If fewer candidate weeks exist than `criteria` has entries.
    """
    n_sites = hourly["site"].n_unique()
    daily_agg = pl.col("output_frac").sum() if agg == "sum" else pl.col("output_frac").mean()
    daily = (
        hourly.with_columns(day=pl.col("time").dt.date())
        .group_by("site", "day")
        .agg(daily_output=daily_agg, hours=pl.len())
        .filter(pl.col("hours") >= min_hours)
    )
    pooled = (
        daily.group_by("day")
        .agg(mean_daily=pl.col("daily_output").mean(), sites=pl.col("site").n_unique())
        .filter(pl.col("sites") == n_sites)
        .with_columns(week=pl.col("day").cast(pl.Datetime("us", "UTC")).dt.truncate("1w"))
    )
    complete = (
        pooled.group_by("week")
        .agg(
            days=pl.len(),
            mean_output=pl.col("mean_daily").mean(),
            spread=pl.col("mean_daily").std(),
        )
        .filter(pl.col("days") == 7)
    )
    if complete.height < len(criteria):
        msg = f"only {complete.height} candidate weeks for {len(criteria)} criteria"
        raise ValueError(msg)
    chosen: list[object] = []
    picks: list[pl.DataFrame] = []
    for name, column, descending in criteria:
        row = (
            complete.filter(~pl.col("week").is_in(chosen))
            .sort(column, descending=descending)
            .head(1)
        )
        chosen.append(row["week"][0])
        picks.append(row.select("week", label=pl.lit(name)))
    return pl.concat(picks)


def _models_work_panel(
    *,
    long_frame: pl.DataFrame,
    site: str,
    week_label: str,
    order: tuple[str, ...],
    colours: tuple[str, ...],
    show_legend: bool,
    show_x_title: bool,
) -> alt.Chart:
    """Draw one generator's measured and predicted power across one chosen week.

    The x axis counts days 1 to 7 of the week rather than showing calendar dates, and the lines
    carry no ARIA text, so the published SVG holds no date and no per-point value that could match
    a generator's series against public generation data.

    Args:
        long_frame: The output of `_models_work_long_frame`, joined to each row's chosen `week`
            and its `label`.
        site: The anonymised generator label to draw.
        week_label: Which chosen week to draw.
        order: The series in legend order, `Measured` first.
        colours: One colour per entry of `order`.
        show_legend: Whether this panel carries the shared legend.
        show_x_title: Whether this panel names its x axis, which only the bottom row does.

    Returns:
        One panel.
    """
    rows = (
        long_frame.filter((pl.col("site") == site) & (pl.col("label") == week_label))
        .with_columns(
            day_of_week=(pl.col("time") - pl.col("week")).dt.total_minutes() / MINUTES_PER_DAY
        )
        .with_columns(day=pl.col("day_of_week").floor())
    )
    return (
        alt.Chart(rows)
        .mark_line(strokeWidth=1.3, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day_of_week:Q",
                title="Day of the week" if show_x_title else None,
                scale=alt.Scale(domain=(0, 7), nice=False, zero=False),
                # One label at the middle of each day, numbered 1 to 7.
                axis=alt.Axis(
                    values=[day + 0.5 for day in range(7)],
                    labelExpr="datum.value + 0.5",
                    grid=False,
                ),
            ),
            # One line per day, so a night the model was never scored on does not join two days.
            detail=alt.Detail("day:Q"),
            y=alt.Y(
                "percent:Q", title=None, scale=alt.Scale(domain=(0, 120), clamp=True, nice=False)
            ),
            color=alt.Color(
                "series:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colours),
                legend=(
                    alt.Legend(orient="bottom", direction="horizontal", labelLimit=0, title=None)
                    if show_legend
                    else None
                ),
            ),
            strokeDash=alt.StrokeDash("series:N", sort=order, legend=None),
        )
        .properties(
            width=PANEL_WIDTH_PX,
            height=PANEL_HEIGHT_PX,
            title=alt.TitleParams(f"Generator {site}: {week_label}", anchor="start", fontSize=11),
        )
    )


def _models_work_timeseries(
    *,
    long_frame: pl.DataFrame,
    sites: tuple[str, ...],
    week_order: tuple[str, ...],
    order: tuple[str, ...],
    colours: tuple[str, ...],
    number: int,
    title: str,
    subtitle: list[str],
) -> alt.VConcatChart:
    """Draw predicted against measured power, one row per generator, one column per chosen week.

    Args:
        long_frame: The output of `_models_work_long_frame`, joined to each row's chosen `week`
            and its `label`.
        sites: The anonymised generator labels, in row order.
        week_order: The chosen weeks' labels, in column order.
        order: The series in legend order, `Measured` first.
        colours: One colour per entry of `order`.
        number: The figure's number on the page.
        title: The finding the figure shows.
        subtitle: Short lines naming the quantity, its scope, and what a line means.

    Returns:
        The figure.
    """
    rows = [
        alt.hconcat(
            *(
                _models_work_panel(
                    long_frame=long_frame,
                    site=site,
                    week_label=week_label,
                    order=order,
                    colours=colours,
                    show_legend=row_index == 0 and week_index == 0,
                    show_x_title=row_index == len(sites) - 1,
                )
                for week_index, week_label in enumerate(week_order)
            ),
            spacing=16,
        )
        for row_index, site in enumerate(sites)
    ]
    return figure(panels=rows, number=number, figure_planning=None, title=title, subtitle=subtitle)


def _models_work_error_chart(
    *,
    losses: pl.DataFrame,
    arm_suffix: str,
    sites: tuple[str, ...],
    names: dict[str, str],
    errors: dict[str, float],
    x_domain: tuple[float, float],
    number: int,
    title: str,
    subtitle: list[str],
) -> alt.VConcatChart:
    """Draw each product's mean absolute error at each generator, one dot per (product, generator).

    The dots carry no ARIA text, so the SVG does not say which dot is which generator.

    Args:
        losses: Every arm's rows from `losses.parquet`, already restricted to the setting that
            `errors` was computed on.
        arm_suffix: The suffix that turns a product key into its arm name, such as `_global`.
        sites: The anonymised generator labels, in offset order.
        names: Each product's name as the page writes it.
        errors: The pooled mean absolute error of each product to draw, which sets the row order.
        x_domain: The x axis's range.
        number: The figure's number on the page.
        title: The finding the figure shows.
        subtitle: Short lines naming the quantity and what a dot means.

    Returns:
        The figure.
    """
    order = sorted(errors, key=errors.__getitem__)
    per_site = (
        losses.filter(pl.col("arm").is_in([f"{product}{arm_suffix}" for product in order]))
        .group_by("arm", "site")
        .agg(mae_percent=pl.col(METRIC).mean() * PERCENTAGE_POINTS)
        .with_columns(product=pl.col("arm").str.strip_suffix(arm_suffix))
        .with_columns(
            name=pl.col("product").replace_strict(names),
            family=pl.col("product").replace_strict(FAMILIES),
        )
    )
    families = [family for family in FAMILY_COLOURS if family in set(per_site["family"])]
    panel = (
        alt.Chart(per_site)
        .mark_point(filled=True, size=70, opacity=0.85, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "name:N",
                sort=[names[product] for product in order],
                title=None,
                axis=alt.Axis(
                    labelLimit=LABEL_WIDTH_PX, minExtent=LABEL_WIDTH_PX, maxExtent=LABEL_WIDTH_PX
                ),
            ),
            x=alt.X(
                "mae_percent:Q",
                title="Mean absolute error (% of capacity; smaller is better)",
                scale=alt.Scale(domain=list(x_domain), nice=False, zero=False),
                axis=alt.Axis(values=ticks(x_domain=x_domain), format=".1f"),
            ),
            yOffset=alt.YOffset("site:N", sort=list(sites)),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(
                    domain=families, range=[FAMILY_COLOURS[family] for family in families]
                ),
                legend=alt.Legend(title="Product type", orient="bottom"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=len(order) * (6 * len(sites) + 14))
    )
    return figure(
        panels=[panel], number=number, figure_planning=None, title=title, subtitle=subtitle
    )


SECTION_DECIDING: Final[str] = "Planned contrasts, named before the run"
SECTION_AGAINST_ERA5: Final[str] = "Every product against ERA5, by scope (exploratory)"
SECTION_POST_ONLY: Final[str] = "The post scope, fitted on post-upgrade rows alone"
SECTION_SPLIT: Final[str] = "A product's own split against Erbs on its own global"
SECTION_TRANSFER: Final[str] = (
    "Leave one site out, the scored months withheld everywhere: the planned contrasts"
)
SECTION_SNAPSHOTS: Final[str] = (
    "UKV's served hour against the mean of its own two snapshots (post hoc)"
)
SECTION_MATCHED_LEAD: Final[str] = "ICON-D2 against ICON-EU at matched served leads"
SECTION_GLOBAL_LEAD: Final[str] = "ICON global against ICON-EU, split by ICON global's lead"
SECTION_BREAKDOWN: Final[str] = "CAMS against ICON-D2, broken down"
SECTION_HOURLY: Final[str] = "ICON-D2 against ICON-EU at each hour, with intervals (post hoc)"
SECTION_MONTHLY: Final[str] = "Implied capacity by calendar month against the annual mean (%)"

DECIDING: Final[tuple[tuple[str, str], ...]] = (
    ("cams_global", "icon_d2_global"),
    ("icon_eu_global", "icon_d2_global"),
    ("icon_eu_global", "ukv_global"),
    ("icon_global_global", "icon_eu_global"),
)
"""The four planned contrasts, as the report writes them."""

HEADLINE_DOMAIN: Final[tuple[float, float]] = (-4.5, 1.0)
"""The x range of the headline's left panel."""

LEADERBOARD_DOMAIN: Final[tuple[float, float]] = (4.0, 9.5)
"""The x range of the leaderboard, covering every product's 95% interval with a small margin."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, December 2022 to September 2026."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
LEADERBOARD_WIDTH: Final[str] = (
    "The intervals are wide mainly because every product's error swings together from month to "
    "month, a swing that Figure 2's paired contrasts cancel."
)


def _product(arm: str) -> str:
    """Return the product an arm belongs to, such as `icon_eu` for `icon_eu_ctx_global`."""
    return max((product for product in NAMES if arm.startswith(f"{product}_")), key=len)


def _two_places(value: float) -> str:
    """Round a report's three-decimal number to two places, half up, as the page does."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _rows(
    *, contrasts: pl.DataFrame, labels: list[str], planned: list[bool] | None = None
) -> pl.DataFrame:
    """Give selected report rows the label, family, and planning a panel draws them with.

    Args:
        contrasts: Rows from `select_contrasts`, in drawing order.
        labels: One label per row.
        planned: Whether each row's contrast was written into the study plan; every row is
            exploratory where this is `None`.

    Returns:
        The rows with `label`, `family` (the first product's), and `planned`.
    """
    return contrasts.with_columns(
        label=pl.Series(labels),
        family=pl.Series([FAMILIES[_product(arm)] for arm in contrasts["treatment"]]),
        planned=pl.Series(planned or [False] * contrasts.height, dtype=pl.Boolean),
    )


def _contrast_name(*, treatment: str, reference: str) -> str:
    """Return a contrast as the page writes it, such as `CAMS − ICON-D2`."""
    return f"{NAMES[_product(treatment)]} − {NAMES[_product(reference)]}"


def _served_name(product: str) -> str:
    """Return a product's name, saying where UKV's value is Open-Meteo's hourly construction."""
    return "UKV, Open-Meteo's hourly value" if product == "ukv" else NAMES[product]


def _served_contrast_name(*, treatment: str, reference: str) -> str:
    """Return a contrast as `_contrast_name` does, naming Open-Meteo's hourly UKV as such."""
    name = _contrast_name(treatment=treatment, reference=reference)
    return name.replace("UKV", "Open-Meteo's hourly UKV")


def _leaderboard(*, losses: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product's own mean absolute error, best first, with its 95% interval.

    Bootstraps each product's absolute error from `losses.parquet` directly, the same
    month-and-seed resampling `weather_products.py` uses for every contrast, because the report
    prints only each product's point estimate, not its interval. No model is refitted.

    Args:
        losses: Every arm's rows from `losses.parquet`.
        errors: Each product's pooled mean absolute error, read from the report.

    Returns:
        Figure 1.
    """
    order = sorted(errors, key=errors.__getitem__)
    records = []
    for product in order:
        arm = f"{product}_global"
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value = interval["value"] * PERCENTAGE_POINTS
        if round(value, 3) != errors[product]:
            msg = f"{product}: bootstrapped {value:.3f} but the report says {errors[product]}"
            raise ValueError(msg)
        records.append(
            {
                "label": _served_name(product),
                "family": FAMILIES[product],
                "value": value,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    rows = pl.DataFrame(records)
    panel = leaderboard_panel(rows=rows, x_domain=LEADERBOARD_DOMAIN, x_title=LEADERBOARD_X_TITLE)
    return figure(
        panels=[panel],
        number=1,
        figure_planning=None,
        title="CAMS has the lowest error of the six products tested, and ERA5 the highest",
        subtitle=[
            "Each product's own mean absolute error, sorted best first.",
            DOTS,
            LEADERBOARD_WIDTH,
            CAPACITY,
            SCOPE,
        ],
    )


def _headline(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product against ERA5 above the four planned contrasts.

    Args:
        contrasts: Every contrast row in the report.
        errors: Each product's mean absolute error.

    Returns:
        Figure 2.
    """
    order = sorted(errors, key=errors.__getitem__)
    against_era5 = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_AGAINST_ERA5, "all", f"{product}_global", "era5_global")
            for product in order
            if product != "era5"
        ],
    ).select("treatment", "difference", "lower_95", "upper_95")
    era5 = pl.DataFrame(
        {"treatment": ["era5_global"], "difference": [0.0], "lower_95": [0.0], "upper_95": [0.0]}
    )
    by_product = pl.concat([against_era5, era5]).sort(
        pl.col("treatment").replace_strict({f"{p}_global": i for i, p in enumerate(order)})
    )
    left_rows = _rows(
        contrasts=by_product,
        labels=[f"{_served_name(product)} · {_two_places(errors[product])}%" for product in order],
    )
    named = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_DECIDING, "all", t, r) for t, r in DECIDING],
    )
    right_rows = _rows(
        contrasts=named,
        labels=[_served_contrast_name(treatment=t, reference=r) for t, r in DECIDING],
        planned=[True] * len(DECIDING),
    )
    figure_planning = planning(rows=[left_rows, right_rows])
    left = interval_panel(
        rows=left_rows,
        x_domain=HEADLINE_DOMAIN,
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        panel_title="Every product against ERA5 (exploratory)",
        figure_planning=figure_planning,
    )
    right = interval_panel(
        rows=right_rows,
        x_domain=(-3.0, 1.0),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        panel_title="The four planned contrasts",
        family_key=False,
        figure_planning=figure_planning,
    )
    return figure(
        panels=[left, right],
        number=2,
        figure_planning=figure_planning,
        title="CAMS beats the next best product, ICON-D2, by more than 2 points of capacity",
        subtitle=[
            (
                "Top: each product against ERA5 (exploratory); each label gives the product's own "
                "error. Bottom: the four planned contrasts."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _cams_breakdown(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw CAMS − ICON-D2 across the record, and by generator, season and calendar year.

    The report's 2022 row is one month in one fold, and is left out, as it is on the page.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 6.
    """
    groups = {
        "Whole record": [("all", "All hours", SECTION_DECIDING)],
        "By generator": [(f"site {s}", f"Generator {s}", SECTION_BREAKDOWN) for s in "ABCDEF"],
        "By season": [
            (f"season {s}", s.capitalize(), SECTION_BREAKDOWN)
            for s in ("winter", "spring", "summer", "autumn")
        ],
        "By calendar year": [
            (f"year {y}", f"{y} (to 10 September)" if y == 2026 else str(y), SECTION_BREAKDOWN)
            for y in range(2023, 2027)
        ],
    }
    group_rows = {
        name: _rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(section, scope, "cams_global", "icon_d2_global")
                    for scope, _, section in rows
                ],
            ),
            labels=[label for _, label, _ in rows],
            planned=[section == SECTION_DECIDING for _, _, section in rows],
        )
        for name, rows in groups.items()
    }
    figure_planning = planning(rows=list(group_rows.values()))
    panels = [
        interval_panel(
            rows=rows,
            x_domain=(-4.0, 0.5),
            x_title=X_TITLE if name == "By calendar year" else "",
            zero_label="same as ICON-D2",
            better_label="CAMS better",
            panel_title=name,
            reference_labels=name == "Whole record",
            figure_planning=figure_planning,
        )
        for name, rows in group_rows.items()
    ]
    return figure(
        panels=panels,
        number=6,
        figure_planning=figure_planning,
        title=(
            "CAMS's margin over ICON-D2 holds at every generator, in every season, and every year"
        ),
        subtitle=[
            (
                "CAMS's mean absolute error minus ICON-D2's. The breakdowns are exploratory. "
                "Winter is December to February, spring March to May, summer June to August, and "
                "autumn September to November. December 2022, one month, is left out of the years."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _hourly_rows(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Return ICON-D2 − ICON-EU at each hour from 09 to 16 UTC, as the report prints them.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        One row per hour with `label`, `family`, `condition` (the served lead), `difference`,
        `lower_95`, and `upper_95`.
    """
    hours = range(9, 17)
    rows = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_HOURLY, f"hour {hour:02d} UTC", "icon_d2_global", "icon_eu_global")
            for hour in hours
        ],
    )
    return rows.select("difference", "lower_95", "upper_95").with_columns(
        label=pl.Series([f"{hour:02d} UTC" for hour in hours]),
        family=pl.lit("weather model"),
        condition=pl.Series([f"{(hour - 1) % 3 + 1} h" for hour in hours]),
    )


def _icon_d2_leads(*, contrasts: pl.DataFrame, report_text: str) -> alt.VConcatChart:
    """Draw ICON-D2 − ICON-EU hour by hour, above the whole record and the matched-lead rows.

    Args:
        contrasts: Every contrast row in the report.
        report_text: The report.

    Returns:
        Figure 7.
    """
    domain = (-2.0, 0.5)
    hourly_rows = _hourly_rows(contrasts=contrasts)
    whole = flip_contrast(
        contrasts=select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(SECTION_DECIDING, "all", "icon_eu_global", "icon_d2_global")],
        )
    )
    matched = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(
                SECTION_MATCHED_LEAD,
                f"both at lead {lead} h, 07–19 UTC",
                "icon_d2_global",
                "icon_eu_global",
            )
            for lead in (1, 2, 3)
        ],
    )
    summary_rows = _rows(
        contrasts=pl.concat([whole, matched]),
        labels=["All hours", *(f"Both at lead {lead} h, 07–19 UTC" for lead in (1, 2, 3))],
        planned=[True, False, False, False],
    )
    figure_planning = planning(rows=[hourly_rows, summary_rows])
    hourly = interval_panel(
        rows=hourly_rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="same as ICON-EU",
        better_label="ICON-D2 better",
        conditions=("1 h", "2 h", "3 h"),
        condition_title="Served lead of both",
        panel_title="Hour by hour (UTC hour ending)",
        figure_planning=figure_planning,
    )
    summary = interval_panel(
        rows=summary_rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="same as ICON-EU",
        better_label="ICON-D2 better",
        panel_title="Whole record, and by served lead",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[hourly, summary],
        number=7,
        figure_planning=figure_planning,
        title="ICON-D2's advantage over ICON-EU shrinks within hours of each run",
        subtitle=[
            (
                "ICON-D2's mean absolute error minus ICON-EU's. Both run every 3 hours, so at each"
                " hour both are served at the same lead."
            ),
            (
                "Rows other than all hours were added after the first run. Filled: 1 h lead; pale "
                "and hollow: 2 h and 3 h."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _icon_eu_rivals(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw ICON global and each UKV construction against ICON-EU, rival minus ICON-EU.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 8.
    """
    icon_global = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_DECIDING, "all", "icon_global_global", "icon_eu_global"),
            *(
                ContrastKey(SECTION_GLOBAL_LEAD, scope, "icon_global_global", "icon_eu_global")
                for scope in (
                    "ICON global lead 1 to 3 h, equal to ICON-EU's, 07–19 UTC",
                    "ICON global lead 4 to 6 h, 07–19 UTC",
                )
            ),
        ],
    )
    ukv = flip_contrast(
        contrasts=select_contrasts(
            contrasts=contrasts,
            wanted=[
                ContrastKey(SECTION_DECIDING, "all", "icon_eu_global", "ukv_global"),
                ContrastKey(SECTION_SNAPSHOTS, "all", "icon_eu_global", "ukv_pair_global"),
                ContrastKey(SECTION_SNAPSHOTS, "all", "icon_eu_global", "ukv_trap_global"),
                ContrastKey(SECTION_SNAPSHOTS, "all", "icon_eu_ctx_global", "ukv_trap_ctx_global"),
            ],
        )
    )
    icon_global_rows = _rows(
        contrasts=icon_global,
        labels=[
            "All hours",
            "Hours its lead equals ICON-EU's, 07–19 UTC",
            "Hours it is served 4 to 6 h ahead, 07–19 UTC",
        ],
        planned=[True, False, False],
    )
    ukv_rows = _rows(
        contrasts=ukv,
        labels=[
            "Open-Meteo's hourly value for UKV",
            "XGBoost model given UKV's two snapshots as a pair",
            "UKV rebuilt as the mean of its two snapshots",
            "Rebuilt UKV with neighbouring hours, against ICON-EU with them",
        ],
        planned=[True, False, False, False],
    )
    figure_planning = planning(rows=[icon_global_rows, ukv_rows])
    shared = {
        "x_domain": (-0.6, 0.8),
        "zero_label": "same as ICON-EU",
        "better_label": "ICON-EU better",
        "better_direction": "positive",
        "width": 420,
        "figure_planning": figure_planning,
    }
    panels = [
        interval_panel(
            rows=icon_global_rows,
            x_title="",
            panel_title="ICON global − ICON-EU",
            **shared,  # ty: ignore[invalid-argument-type]
        ),
        interval_panel(
            rows=ukv_rows,
            x_title="Rival's mean absolute error minus ICON-EU's (points of capacity)",
            panel_title="UKV − ICON-EU",
            reference_labels=False,
            **shared,  # ty: ignore[invalid-argument-type]
        ),
    ]
    return figure(
        panels=panels,
        number=8,
        figure_planning=figure_planning,
        title=(
            "ICON-EU does not beat UKV rebuilt from its snapshots, but beats ICON global and "
            "Open-Meteo's hourly UKV"
        ),
        subtitle=[
            (
                "Each rival's mean absolute error minus ICON-EU's. Rows other than the two "
                "all-hours rows were added after the first run."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _ukv_against_era5(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV − ERA5 by scope, for Open-Meteo's hourly value and for UKV rebuilt from snapshots.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 9.
    """
    conditions = ("Open-Meteo's hourly value", "Rebuilt from its snapshots")
    scopes = {
        "all": "All hours",
        "ukv_live": "Since August 2024",
        "post": "After the upgrade: XGBoost models trained on both sides of the upgrade",
    }
    wanted = []
    for scope in scopes:
        wanted += [
            ContrastKey(SECTION_AGAINST_ERA5, scope, "ukv_global", "era5_global"),
            ContrastKey(SECTION_SNAPSHOTS, scope, "ukv_trap_global", "era5_global"),
        ]
    wanted.append(ContrastKey(SECTION_POST_ONLY, "post", "ukv_global", "era5_global"))
    labels = [label for label in scopes.values() for _ in conditions]
    labels.append("After the upgrade: XGBoost models trained on those 8 months")
    rows = _rows(
        contrasts=select_contrasts(contrasts=contrasts, wanted=wanted), labels=labels
    ).with_columns(condition=pl.Series([*conditions * len(scopes), conditions[0]]))
    figure_planning = planning(rows=[rows])
    panel = interval_panel(
        rows=rows,
        x_domain=(-1.2, 0.6),
        x_title="UKV's mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="UKV better",
        conditions=conditions,
        condition_title="UKV's hourly value",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[panel],
        number=9,
        figure_planning=figure_planning,
        title="UKV rebuilt from its snapshots beats ERA5; Open-Meteo's hourly UKV against ERA5 is "
        "unresolved",
        subtitle=[
            (f"{DOTS} Rebuilt: the mean of UKV's two snapshots, added after the first run."),
            (
                "Since August 2024: Open-Meteo's own UKV download. The upgrade: January 2026; its "
                "rows rest on 8 months, so their intervals are likely too narrow."
            ),
            f"{CAPACITY} {SCOPE}",
        ],
    )


def _own_beam(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw each product's own beam/diffuse split against the Erbs split of its own global.

    Args:
        contrasts: Every contrast row in the report.
        errors: Each product's mean absolute error, which sets the row order.

    Returns:
        Figure 10.
    """
    order = sorted(errors, key=errors.__getitem__)
    rows = _rows(
        contrasts=select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(SECTION_SPLIT, "all", f"{p}_split", f"{p}_erbs") for p in order],
        ),
        labels=[NAMES[product] for product in order],
    )
    figure_planning = planning(rows=[rows])
    panel = interval_panel(
        rows=rows,
        x_domain=(-0.2, 0.1),
        x_title="Mean absolute error with its own beam minus with the Erbs split "
        "(points of capacity)",
        zero_label="no gain",
        better_label="own beam better",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[panel],
        number=10,
        figure_planning=figure_planning,
        title="Every product except ERA5 gains 0.03 to 0.12 points from its own direct beam",
        subtitle=[
            (
                "Each product with its own published beam and diffuse, minus with the Erbs split "
                "of its own global irradiance."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _neighbours(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the four named contrasts for per-generator models and for leave-one-site-out models.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 11.
    """
    conditions = (
        "Trained on the generator itself",
        "Trained on the other five generators",
    )
    sections = (SECTION_DECIDING, SECTION_TRANSFER)
    frames = [
        select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(section, "all", t, r) for t, r in DECIDING],
        ).with_columns(condition=pl.lit(condition))
        for section, condition in zip(sections, conditions, strict=True)
    ]
    labels = [_served_contrast_name(treatment=t, reference=r) for t, r in DECIDING]
    rows = pl.concat(
        [
            _rows(
                contrasts=frame,
                labels=labels,
                planned=[section == SECTION_DECIDING] * len(DECIDING),
            )
            for frame, section in zip(frames, sections, strict=True)
        ]
    ).sort(pl.col("label").replace_strict({label: i for i, label in enumerate(labels)}))
    figure_planning = planning(rows=[rows])
    panel = interval_panel(
        rows=rows,
        x_domain=(-3.0, 1.0),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        conditions=conditions,
        condition_title="XGBoost model",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[panel],
        number=11,
        figure_planning=figure_planning,
        title="The ranking holds for a generator predicted from its neighbours",
        subtitle=[
            (
                "Hollow, and exploratory: the same contrast from an XGBoost model trained on the "
                "other five generators, with the scored months withheld everywhere."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _implied_capacity_rows(*, report_text: str) -> pl.DataFrame:
    """Read each product's implied capacity by calendar month from the report's monthly table.

    Args:
        report_text: The report.

    Returns:
        One row per (product, calendar month) with `percent`, the month against the annual mean.

    Raises:
        ValueError: If the report has no monthly table, or a product is missing from it.
    """
    lines = report_text.splitlines()
    try:
        start = lines.index(f"#### {SECTION_MONTHLY}")
    except ValueError as error:
        msg = "the report has no implied-capacity-by-month table"
        raise ValueError(msg) from error
    records = []
    for line in lines[start + 4 : start + 4 + len(NAMES)]:
        product, *cells = (cell.strip() for cell in line.strip().strip("|").split("|"))
        records += [
            {"product": product, "month": month, "percent": float(cell)}
            for month, cell in enumerate(cells, start=1)
        ]
    if {record["product"] for record in records} != set(NAMES):
        msg = f"the monthly table does not list every product: {records}"
        raise ValueError(msg)
    return pl.DataFrame(records)


def _implied_capacity_chart(*, report_text: str) -> alt.VConcatChart:
    """Draw each product's implied capacity by calendar month, against its annual mean.

    Args:
        report_text: The report.

    Returns:
        Figure 12.
    """
    rows = _implied_capacity_rows(report_text=report_text).with_columns(
        family=pl.col("product").replace_strict(FAMILIES),
        month_name=pl.col("month").replace_strict(
            {m: calendar.month_abbr[m] for m in range(1, 13)}
        ),
    )
    months = [calendar.month_abbr[m] for m in range(1, 13)]
    families = list(FAMILY_COLOURS)
    colour = alt.Color(
        "family:N",
        scale=alt.Scale(domain=families, range=list(FAMILY_COLOURS.values())),
        legend=alt.Legend(title="Product type", values=families),
    )
    x = alt.X(
        "month_name:N",
        sort=months,
        title=None,
        axis=alt.Axis(labelAngle=0, values=["Jan", "Apr", "Jul", "Oct", "Dec"]),
    )
    panels = []
    y = alt.Y("percent:Q", title="Against annual mean (%)", scale=alt.Scale(domain=[-30, 30]))
    for product in NAMES:
        data = rows.filter(pl.col("product") == product).sort("month")
        base = alt.Chart(data)
        line = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
        points = base.mark_point(filled=True, size=40, opacity=1).encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(zero, line, points).properties(
                width=(CONTENT_WIDTH_PX - 2 * 64 - 24) // 2,
                height=170,
                title=alt.TitleParams(NAMES[product], anchor="start", frame="group", fontSize=14),
            )
        )
    grid = [alt.hconcat(*panels[i : i + 2], spacing=24) for i in (0, 2, 4)]
    return figure(
        panels=grid,
        number=12,
        figure_planning=None,
        title=(
            "Of the six products tested, CAMS's implied capacity swings the most with the seasons"
        ),
        subtitle=[
            (
                "Implied capacity: metered output over what a south-facing panel at 30° tilt "
                "predicts per megawatt."
            ),
            (
                "Each point: that calendar month against the generator's annual mean. Closer to "
                "zero means steadier; this study cannot say which product is right."
            ),
            (
                "Exploratory; no interval is drawn. Hours with no curtailment cap and the sun "
                "above 10°."
            ),
            SCOPE,
        ],
    )


MODELS_WORK_SITES: Final[tuple[str, ...]] = tuple("ABCDEF")
"""The six anonymised solar generator labels, in the order every "models work" panel lists them."""

MODELS_WORK_PRODUCTS: Final[tuple[str, str]] = ("cams", "era5")
"""The products whose XGBoost models the "models work" time series draws: CAMS, which has the lowest
pooled mean absolute error in the report, and ERA5, the reference every product is measured
against.
"""

MODELS_WORK_MIN_DAYLIGHT_HOURS: Final[int] = 8
"""A (generator, day) needs at least this many daylight hours to count towards week selection."""


def _models_work_frame() -> pl.DataFrame:
    """Rebuild the exact rows and measured power `weather_products.py` scored, without refitting.

    Calls the row-building functions `weather_products.py`'s own `main` calls, so the rows match
    the study exactly; nothing here fits a model.

    Returns:
        One row per (site, time) the pooled run scored, carrying `power_mw`,
        `effective_capacity_mw`, and `extraterrestrial_horizontal_w_m2`.
    """
    return common_rows(frame=joined()).select(
        "site", "time", "power_mw", "effective_capacity_mw", "extraterrestrial_horizontal_w_m2"
    )


def _solar_models_work(
    *, losses: pl.DataFrame, errors: dict[str, float]
) -> tuple[alt.VConcatChart, alt.VConcatChart]:
    """Draw the solar "models work" figures: the out-of-fold time series, and error per generator.

    Args:
        losses: Every arm's rows from `losses.parquet`.
        errors: Each product's pooled mean absolute error.

    Returns:
        Figures 4 and 5.
    """
    measured = _models_work_frame()
    order = ("Measured", *(f"XGBoost model given {NAMES[p]}" for p in MODELS_WORK_PRODUCTS))
    predicted = tuple(
        (_reconstruct_predicted(losses=losses, measured=measured, arm=f"{product}_global"), label)
        for product, label in zip(MODELS_WORK_PRODUCTS, order[1:], strict=True)
    )
    hourly = measured.filter(
        (pl.col("extraterrestrial_horizontal_w_m2") > 0)
        & pl.col("time").dt.month().is_in(MODELS_WORK_MONTHS)
    ).with_columns(
        output_frac=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw")
    )
    weeks = _pick_weeks(
        hourly=hourly,
        min_hours=MODELS_WORK_MIN_DAYLIGHT_HOURS,
        agg="sum",
        criteria=SOLAR_WEEK_CRITERIA,
    )
    long_frame = (
        _models_work_long_frame(measured=measured, predicted=predicted)
        .with_columns(week=pl.col("time").dt.truncate("1w"))
        .join(weeks, on="week", how="inner")
    )
    timeseries = _models_work_timeseries(
        long_frame=long_frame,
        sites=MODELS_WORK_SITES,
        week_order=SOLAR_WEEK_DISPLAY_ORDER,
        order=order,
        colours=(ocf.TEXT, *(FAMILY_COLOURS[FAMILIES[p]] for p in MODELS_WORK_PRODUCTS)),
        number=4,
        title=(
            "An XGBoost model given CAMS tracks measured power at every generator, across a "
            "clear, a variable, and a dull week"
        ),
        subtitle=[
            (
                "Out-of-fold power as a percentage of the generator's own capacity, each "
                "prediction held to the export cap as the scores are."
            ),
            (
                "Weeks are picked from measured power alone, pooled over the six generators, "
                "April to September: the clearest has the most output, the dullest the least, and "
                "the most variable the largest swing in daily output."
            ),
            CAPACITY,
            SCOPE,
        ],
    )
    error = _models_work_error_chart(
        losses=losses,
        arm_suffix="_global",
        sites=MODELS_WORK_SITES,
        names={product: _served_name(product) for product in NAMES},
        errors=errors,
        x_domain=(4.0, 11.5),
        number=5,
        title=(
            "CAMS has the lowest error at each of the six generators, and ICON-D2 the second lowest"
        ),
        subtitle=[
            (
                "Each dot is one generator's mean absolute error given one product. The other "
                "four products reorder between generators."
            ),
            CAPACITY,
            SCOPE,
        ],
    )
    return timeseries, error


def main() -> int:
    """Read the report, compute the new numbers, and write the eleven SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = RESULTS_DIR / "report.md"
    report_text = report_path.read_text()
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="Global only")
    losses = pl.read_parquet(RESULTS_DIR / "losses.parquet")
    models_work_timeseries, models_work_error = _solar_models_work(losses=losses, errors=errors)
    charts = {
        "sunshine_leaderboard": _leaderboard(losses=losses, errors=errors),
        "sunshine_headline": _headline(contrasts=contrasts, errors=errors),
        "sunshine_models_work_timeseries": models_work_timeseries,
        "sunshine_models_work_error": models_work_error,
        "sunshine_cams_breakdown": _cams_breakdown(contrasts=contrasts),
        "sunshine_icon_d2_leads": _icon_d2_leads(contrasts=contrasts, report_text=report_text),
        "sunshine_icon_eu_rivals": _icon_eu_rivals(contrasts=contrasts),
        "sunshine_ukv_against_era5": _ukv_against_era5(contrasts=contrasts),
        "sunshine_own_beam": _own_beam(contrasts=contrasts, errors=errors),
        "sunshine_neighbours": _neighbours(contrasts=contrasts),
        "sunshine_implied_capacity": _implied_capacity_chart(report_text=report_text),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
