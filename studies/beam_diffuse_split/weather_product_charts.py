"""Draw the anonymised charts of the past-solar page that read the panel reports.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/solar/>.
`figure_numbers.FIGURE_NUMBERS` gives each chart's number, and `past_solar_leaderboard_charts.py`
draws the page's leaderboard and contrasts.

**Every number a chart shares with the report is read from the report `weather_products.py`
wrote for the `long` panel**, so a chart cannot disagree with the page. The ERA5-by-year chart and
the satellite rows of the SARAH-3 chart read the `record` panel's report and table instead,
because only that panel reaches back to 2021. The own-beam chart's extra-rows panel reads the `all`
panel's own report, because that panel's row set is shorter and starts later, from November 2024.
The two "models work" charts also draw numbers the report does not print, computed from
`losses.parquet` without refitting any model: the out-of-fold predictions and per-generator errors.

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
from figure_numbers import FIGURE_NUMBERS
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
    planning,
    report_contrasts,
    report_errors,
    select_contrasts,
    ticks,
)
from weather_products import (
    METRIC,
    NEW_PLANNED_CONTRASTS,
    PANELS,
    PERCENTAGE_POINTS,
    SARAH_SATELLITE_ERAS,
    UNUSABLE_SPLITS,
    common_rows,
    joined,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = PANELS["long"].output_dir
"""Where `weather_products.py` wrote the `long` panel's report and losses, which the page uses."""

RECORD_DIR: Final[Path] = PANELS["record"].output_dir
"""Where `weather_products.py` wrote the `record` panel's ERA5-by-year table."""

ALL_DIR: Final[Path] = PANELS["all"].output_dir
"""Where `weather_products.py` wrote the `all` panel's report and losses, from November 2024."""

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "cams": "CAMS",
    "era5": "ERA5",
    "ukv": "UKV",
    "icon_d2": "ICON-D2",
    "icon_eu": "ICON-EU",
    "icon_global": "ICON global",
    "sarah3": "SARAH-3",
    "icon_dream": "ICON-DREAM-EU",
}
"""Each `long`-panel product's name as the page writes it.

Exactly the `long` panel's eight products, no more: `_implied_capacity_rows` reads exactly
`len(NAMES)` lines from that panel's own report table, and `_implied_capacity_chart` draws one
panel per key. `ALL_PANEL_NAMES` is the `all` panel's own, separate registry.
"""

FAMILIES: Final[dict[str, ProductFamily]] = {
    "cams": "satellite",
    "era5": "reanalysis",
    "ukv": "weather model",
    "icon_d2": "weather model",
    "icon_eu": "weather model",
    "icon_global": "weather model",
    "sarah3": "satellite",
    "icon_dream": "reanalysis",
}
"""Each `long`-panel product's family, which sets its colour. See `NAMES`."""

ALL_PANEL_NAMES: Final[dict[str, str]] = NAMES | {
    "ifs_hres": "ECMWF-IFS-HRES",
    "arpege": "ARPEGE Europe",
    "dmi_harmonie": "DMI HARMONIE-AROME",
    "knmi_harmonie": "KNMI HARMONIE-AROME",
}
"""Every `all`-panel product's name: `NAMES` plus the four Open-Meteo models only that panel
scores."""

ALL_PANEL_FAMILIES: Final[dict[str, ProductFamily]] = FAMILIES | {
    "ifs_hres": "weather model",
    "arpege": "weather model",
    "dmi_harmonie": "weather model",
    "knmi_harmonie": "weather model",
}
"""Every `all`-panel product's family. See `ALL_PANEL_NAMES`."""

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
    panel_width: int = PANEL_WIDTH_PX,
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
        panel_width: The plot area's width in pixels.

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
            width=panel_width,
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
    panel_width: int = PANEL_WIDTH_PX,
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
        panel_width: Each panel's plot-area width in pixels.

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
                    panel_width=panel_width,
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

PLANNED: Final[tuple[tuple[str, str], ...]] = PANELS["long"].planned
"""Every planned contrast the page reports: the first round's four, then the second round's two."""

NEW_PLANNED: Final[tuple[tuple[str, str], ...]] = NEW_PLANNED_CONTRASTS["long"]
"""The second round's planned contrasts: SARAH-3 against CAMS, ICON-DREAM-EU against ERA5."""

SECTION_SENSITIVITY: Final[str] = "Planned contrasts at the second hyperparameter setting"
SECTION_SATELLITE: Final[str] = (
    "SARAH-3 against CAMS, by the satellite behind SARAH-3 (exploratory)"
)

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, December 2022 to August 2026."
RECORD_SCOPE: Final[str] = "Six solar farms in Lincolnshire, January 2021 to August 2026."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
NEW_PRODUCTS_TITLE: Final[str] = (
    "CAMS beats SARAH-3 by about 0.4 points under every satellite, and ICON-DREAM-EU beats ERA5 by "
    "0.3 points"
)
"""The title of the SARAH-3 and ICON-DREAM-EU figure, which states the finding."""

ERA5_BY_YEAR_TITLE: Final[str] = (
    "On matched months, ERA5 trails both satellite retrievals by more than 3 points in every year "
    "from 2021 to 2026"
)
"""The title of the ERA5-by-year figure, which states the finding."""

ERA5_BY_YEAR_DOMAIN: Final[tuple[float, float]] = (-5.5, 0.5)
"""The ERA5-by-year figure's x range, covering every year's interval with a small margin."""

LEADERBOARD_WIDTH: Final[str] = (
    "The intervals are wide mainly because every product's error swings together from month to "
    "month, a swing that Figure 2's paired contrasts cancel."
)


def _product(arm: str) -> str:
    """Return the product an arm belongs to, such as `icon_eu` for `icon_eu_ctx_global`."""
    return max((product for product in ALL_PANEL_NAMES if arm.startswith(f"{product}_")), key=len)


def era5_by_year_rows(
    *, by_year: pl.DataFrame, products: tuple[str, ...], suffix: str
) -> pl.DataFrame:
    """Turn `weather_products.era5_difference_by_year`'s table into rows of product minus ERA5.

    The table holds ERA5 minus each product; the charts draw each product minus ERA5, as every
    other chart against ERA5 does, so the sign and the interval's ends are flipped. A year of too
    few months to carry an interval is left out.

    Args:
        by_year: The saved `era5_by_year.parquet`.
        products: The products to draw, in drawing order.
        suffix: The arm suffix, `_global` for solar and `_wind` for wind.

    Returns:
        One row per (product, year) with `label`, `family`, `condition` (the year), `difference`,
        `lower_95` and `upper_95`, in points of capacity.
    """
    frames = [
        by_year.filter(pl.col("arm") == f"{product}{suffix}", pl.col("enough_months"))
        .sort("year")
        .select(
            label=pl.lit(NAMES[product]),
            family=pl.lit(FAMILIES[product]),
            condition=pl.col("year").cast(pl.Utf8),
            difference=-pl.col("difference") * PERCENTAGE_POINTS,
            lower_95=-pl.col("upper_95") * PERCENTAGE_POINTS,
            upper_95=-pl.col("lower_95") * PERCENTAGE_POINTS,
        )
        for product in products
    ]
    return pl.concat(frames)


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
        family=pl.Series([ALL_PANEL_FAMILIES[_product(arm)] for arm in contrasts["treatment"]]),
        planned=pl.Series(planned or [False] * contrasts.height, dtype=pl.Boolean),
    )


def _contrast_name(*, treatment: str, reference: str) -> str:
    """Return a contrast as the page writes it, such as `CAMS − ICON-D2`."""
    return f"{NAMES[_product(treatment)]} − {NAMES[_product(reference)]}"


def served_name(product: str) -> str:
    """Return a product's name, saying where UKV's value is Open-Meteo's hourly construction."""
    return "UKV, Open-Meteo's hourly value" if product == "ukv" else NAMES[product]


def _served_contrast_name(*, treatment: str, reference: str) -> str:
    """Return a contrast as `_contrast_name` does, naming Open-Meteo's hourly UKV as such."""
    name = _contrast_name(treatment=treatment, reference=reference)
    return name.replace("UKV", "Open-Meteo's hourly UKV")


def _cams_breakdown(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw CAMS − ICON-D2 across the record, and by generator, season and calendar year.

    The report's 2022 row is one month in one fold, and is left out, as it is on the page.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The figure.
    """
    groups = {
        "Whole record": [("all", "All hours", SECTION_DECIDING)],
        "By generator": [(f"site {s}", f"Generator {s}", SECTION_BREAKDOWN) for s in "ABCDEF"],
        "By season": [
            (f"season {s}", s.capitalize(), SECTION_BREAKDOWN)
            for s in ("winter", "spring", "summer", "autumn")
        ],
        "By calendar year": [
            (f"year {y}", f"{y} (to August)" if y == 2026 else str(y), SECTION_BREAKDOWN)
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
        number=FIGURE_NUMBERS["cams_breakdown"],
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


def _new_products(*, contrasts: pl.DataFrame, record_contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw SARAH-3 against CAMS and ICON-DREAM-EU against ERA5, the second round's two contrasts.

    Each is drawn at the main hyperparameter setting and the second. SARAH-3 against CAMS is also
    drawn by the satellite behind SARAH-3, from the `record` panel, whose rows reach back to 2021.

    Args:
        contrasts: Every contrast row in the `long` panel's report.
        record_contrasts: Every contrast row in the `record` panel's report.

    Returns:
        The figure.
    """
    sarah, dream = NEW_PLANNED
    settings = ["All hours", "All hours, second XGBoost setting"]
    present = set(record_contrasts.filter(pl.col("section") == SECTION_SATELLITE)["scope"])
    eras = [label for label, _, _ in SARAH_SATELLITE_ERAS if label in present]
    panels_rows = [
        _rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(SECTION_DECIDING, "all", *pair),
                    ContrastKey(SECTION_SENSITIVITY, "sensitivity", *pair),
                ],
            ),
            labels=settings,
            planned=[True, False],
        )
        for pair in (sarah, dream)
    ]
    panels_rows.insert(
        1,
        _rows(
            contrasts=select_contrasts(
                contrasts=record_contrasts,
                wanted=[ContrastKey(SECTION_SATELLITE, era, *sarah) for era in eras],
            ),
            labels=eras,
        ),
    )
    titles = (
        "SARAH-3 − CAMS, December 2022 to August 2026",
        "SARAH-3 − CAMS by satellite, 2021 to August 2026",
        "ICON-DREAM-EU − ERA5, December 2022 to August 2026",
    )
    pairs = (sarah, sarah, dream)
    figure_planning = planning(rows=panels_rows)
    panels = [
        interval_panel(
            rows=rows,
            x_domain=(-1.0, 1.0),
            x_title=X_TITLE if index == len(panels_rows) - 1 else "",
            zero_label=f"same as {NAMES[_product(reference)]}",
            better_label=f"{NAMES[_product(treatment)]} better",
            panel_title=title,
            reference_labels=index != 1,
            figure_planning=figure_planning,
        )
        for index, (rows, title, (treatment, reference)) in enumerate(
            zip(panels_rows, titles, pairs, strict=True)
        )
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS["new_products"],
        figure_planning=figure_planning,
        title=NEW_PRODUCTS_TITLE,
        subtitle=[
            (
                "Top two: SARAH-3's mean absolute error minus CAMS's. Bottom: ICON-DREAM-EU's "
                "minus ERA5's. The satellite rows are on the hours ERA5, CAMS, SARAH-3 and "
                "ICON-DREAM-EU all cover; January 2022, one month with a fortnight from "
                "Meteosat-9, is too short for an interval and is left out."
            ),
            f"{DOTS} {CAPACITY}",
            "Six solar farms in Lincolnshire.",
        ],
    )


def _era5_by_year() -> alt.VConcatChart:
    """Draw CAMS, SARAH-3 and ICON-DREAM-EU against ERA5 in each calendar year from 2021.

    Reads the `record` panel's `era5_by_year.parquet`, which `weather_products.py` wrote on the
    four products whose records start by 2021, restricted to January to August of each year so a
    partial 2026 compares against the same months of the complete years before it. A year of too
    few months carries no interval and is left out.

    Returns:
        The figure.
    """
    by_year = pl.read_parquet(RECORD_DIR / "era5_by_year.parquet")
    products = ("cams", "sarah3", "icon_dream")
    product_rows = [
        era5_by_year_rows(by_year=by_year, products=(product,), suffix="_global").with_columns(
            label=pl.col("condition")
        )
        for product in products
    ]
    figure_planning = planning(rows=product_rows)
    panels = [
        interval_panel(
            rows=rows,
            x_domain=ERA5_BY_YEAR_DOMAIN,
            x_title=(
                "Mean absolute error minus ERA5's (points of capacity)"
                if index == len(products) - 1
                else ""
            ),
            zero_label="same as ERA5",
            better_label="better than ERA5",
            panel_title=f"{NAMES[product]} − ERA5",
            reference_labels=index == 0,
            family_key=False,
            figure_planning=figure_planning,
        )
        for index, (product, rows) in enumerate(zip(products, product_rows, strict=True))
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS["era5_by_year"],
        figure_planning=figure_planning,
        title=ERA5_BY_YEAR_TITLE,
        subtitle=[
            (
                "Each product's mean absolute error minus ERA5's, on January to August of one "
                "calendar year, on the hours all four products cover. Every year is restricted to "
                "the same months so a partial 2026 compares against the same months of the "
                "complete years before it."
            ),
            f"{DOTS} {CAPACITY}",
            RECORD_SCOPE,
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
        The figure.
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
        number=FIGURE_NUMBERS["icon_d2_leads"],
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


def _icon_eu_rivals_rows(*, contrasts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ICON global's and each UKV construction's rows against ICON-EU, rival minus ICON-EU.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The ICON global rows, then the UKV rows.
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
    return icon_global_rows, ukv_rows


UKV_ERA5_CONDITIONS: Final[tuple[str, str]] = (
    "Open-Meteo's hourly value",
    "Rebuilt from its snapshots",
)
"""The two ways UKV's hourly value is built, as the UKV-against-ERA5 panel's conditions."""


def _ukv_against_era5_rows(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Return UKV minus ERA5 by scope, for Open-Meteo's hourly value and for UKV rebuilt.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The rows, with `condition` from `UKV_ERA5_CONDITIONS`.
    """
    conditions = UKV_ERA5_CONDITIONS
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
    labels.append("After the upgrade: XGBoost models trained on those 7 months")
    return _rows(
        contrasts=select_contrasts(contrasts=contrasts, wanted=wanted), labels=labels
    ).with_columns(condition=pl.Series([*conditions * len(scopes), conditions[0]]))


def _weather_model_rivals(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw ICON-EU against ICON global and UKV, and UKV against ERA5, in one figure.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The figure.
    """
    icon_global_rows, ukv_rows = _icon_eu_rivals_rows(contrasts=contrasts)
    era5_rows = _ukv_against_era5_rows(contrasts=contrasts)
    figure_planning = planning(rows=[icon_global_rows, ukv_rows, era5_rows])
    shared = {"width": 420, "figure_planning": figure_planning}
    against_icon_eu = {
        "x_domain": (-0.6, 0.8),
        "zero_label": "same as ICON-EU",
        "better_label": "ICON-EU better",
        "better_direction": "positive",
    }
    panels = [
        interval_panel(
            rows=icon_global_rows,
            x_title="",
            panel_title="ICON global − ICON-EU",
            **against_icon_eu,  # ty: ignore[invalid-argument-type]
            **shared,
        ),
        interval_panel(
            rows=ukv_rows,
            x_title="Rival's mean absolute error minus ICON-EU's (points of capacity)",
            panel_title="UKV − ICON-EU",
            reference_labels=False,
            **against_icon_eu,  # ty: ignore[invalid-argument-type]
            **shared,
        ),
        interval_panel(
            rows=era5_rows,
            x_domain=(-1.2, 0.6),
            x_title="UKV's mean absolute error minus ERA5's (points of capacity)",
            zero_label="same as ERA5",
            better_label="UKV better",
            conditions=UKV_ERA5_CONDITIONS,
            condition_title="UKV's hourly value",
            panel_title="UKV − ERA5",
            family_key=False,
            **shared,  # ty: ignore[invalid-argument-type]
        ),
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS["weather_model_rivals"],
        figure_planning=figure_planning,
        title=(
            "UKV rebuilt from its snapshots beats ICON-EU, which beats ICON global and "
            "Open-Meteo's hourly UKV; rebuilt UKV also beats ERA5 in every period"
        ),
        subtitle=[
            (
                "Top two panels: each rival's mean absolute error minus ICON-EU's. Bottom panel: "
                "UKV's minus ERA5's, by period; UKV rebuilt is the mean of its two snapshots. Rows "
                "other than the all-hours rows of the top two panels were added after the first "
                "run."
            ),
            (
                "Since August 2024: Open-Meteo's own UKV download. The upgrade: January 2026; its "
                "rows rest on 7 months, so their intervals are likely too narrow."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


OWN_BEAM_DOMAIN: Final[tuple[float, float]] = (-0.25, 0.2)
"""The x range of both own-beam panels, covering every interval with a small margin."""


def _own_beam_rows(
    *, contrasts: pl.DataFrame, errors: dict[str, float], names: dict[str, str]
) -> pl.DataFrame:
    """Return each product's own beam/diffuse split minus the Erbs split of its own global.

    Args:
        contrasts: Every contrast row in one panel's report.
        errors: Each product's mean absolute error, which sets the row order.
        names: Each product's name as the page writes it.

    Returns:
        One row per product the report holds a split contrast for, best first, with `label`,
        `family`, and `planned`.
    """
    held = set(contrasts["treatment"])
    order = [
        p
        for p in sorted(errors, key=errors.__getitem__)
        if p not in UNUSABLE_SPLITS and f"{p}_split" in held
    ]
    return _rows(
        contrasts=select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(SECTION_SPLIT, "all", f"{p}_split", f"{p}_erbs") for p in order],
        ),
        labels=[names[product] for product in order],
    )


def _own_beam(
    *,
    contrasts: pl.DataFrame,
    errors: dict[str, float],
    all_contrasts: pl.DataFrame,
    all_errors: dict[str, float],
) -> alt.VConcatChart:
    """Draw each product's own beam/diffuse split against the Erbs split, on two row sets.

    The main row set's panel holds seven products. The extra row set's panel holds the same
    products and ECMWF-IFS-HRES, scored on that row set's own rows.

    Args:
        contrasts: Every contrast row in the `long` panel's report.
        errors: Each product's mean absolute error on the `long` panel, which sets the row order.
        all_contrasts: Every contrast row in the `all` panel's report.
        all_errors: Each product's mean absolute error on the `all` panel.

    Returns:
        The figure.
    """
    row_sets = (
        ("Main rows: December 2022 to August 2026", contrasts, errors, NAMES),
        ("Extra rows: November 2024 to August 2026", all_contrasts, all_errors, ALL_PANEL_NAMES),
    )
    frames = [_own_beam_rows(contrasts=c, errors=e, names=names) for _, c, e, names in row_sets]
    figure_planning = planning(rows=frames)
    panels = [
        interval_panel(
            rows=rows,
            x_domain=OWN_BEAM_DOMAIN,
            x_title=(
                "Mean absolute error with its own beam minus with the Erbs split "
                "(points of capacity)"
                if index == len(frames) - 1
                else ""
            ),
            zero_label="no gain",
            better_label="own beam better",
            panel_title=title,
            family_key=index == 0,
            reference_labels=index == 0,
            figure_planning=figure_planning,
        )
        for index, ((title, *_), rows) in enumerate(zip(row_sets, frames, strict=True))
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS["own_beam"],
        figure_planning=figure_planning,
        title=(
            "On the main rows every product with its own direct beam, except ERA5, gains 0.03 to "
            "0.10 points from it; on the extra rows UKV and ICON-D2 gain and ECMWF-IFS-HRES loses"
        ),
        subtitle=[
            (
                "Each product with its own published beam and diffuse, minus with the Erbs split "
                "of its own global irradiance. SARAH-3 is left out: its direct beam is modelled "
                "from its own global irradiance. The two row sets are scored on their own rows, "
                "so compare products within a panel."
            ),
            f"{DOTS} {CAPACITY}",
            "Six solar farms in Lincolnshire.",
        ],
    )


def _neighbours(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the six planned contrasts for per-generator models and for leave-one-site-out models.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The figure.
    """
    conditions = (
        "Trained on the generator itself",
        "Trained on the other five generators",
    )
    sections = (SECTION_DECIDING, SECTION_TRANSFER)
    frames = [
        select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(section, "all", t, r) for t, r in PLANNED],
        ).with_columns(condition=pl.lit(condition))
        for section, condition in zip(sections, conditions, strict=True)
    ]
    labels = [_served_contrast_name(treatment=t, reference=r) for t, r in PLANNED]
    rows = pl.concat(
        [
            _rows(
                contrasts=frame,
                labels=labels,
                planned=[section == SECTION_DECIDING] * len(PLANNED),
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
        number=FIGURE_NUMBERS["neighbours"],
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
        The figure.
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
    grid = [alt.hconcat(*panels[i : i + 2], spacing=24) for i in range(0, len(panels), 2)]
    return figure(
        panels=grid,
        number=FIGURE_NUMBERS["implied_capacity"],
        figure_planning=None,
        title=(
            "Of the eight products tested, CAMS and SARAH-3 imply the steadiest capacity from "
            "month to month but swing the most with the seasons"
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


def all_served_name(product: str) -> str:
    """Return an `all`-panel product's name, as `served_name` does, against `ALL_PANEL_NAMES`."""
    return "UKV, Open-Meteo's hourly value" if product == "ukv" else ALL_PANEL_NAMES[product]


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
    return common_rows(frame=joined(products=PANELS["long"].products)).select(
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
        The time-series figure, then the error-by-generator figure.
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
        number=FIGURE_NUMBERS["models_work_timeseries"],
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
        names={product: served_name(product) for product in NAMES},
        errors=errors,
        x_domain=(4.0, 11.5),
        number=FIGURE_NUMBERS["models_work_error"],
        title=(
            "CAMS has the lowest error at each of the six generators, SARAH-3 the second lowest, "
            "and ICON-D2 the third"
        ),
        subtitle=[
            (
                "Each dot is one generator's mean absolute error given one product. The other "
                "five products reorder between generators."
            ),
            CAPACITY,
            SCOPE,
        ],
    )
    return timeseries, error


def main() -> int:
    """Read the reports and write the SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = RESULTS_DIR / "report.md"
    report_text = report_path.read_text()
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="Global only")
    # The saved losses hold the second hyperparameter setting's arms too, under the same names.
    losses = pl.read_parquet(RESULTS_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")
    models_work_timeseries, models_work_error = _solar_models_work(losses=losses, errors=errors)
    all_report_path = ALL_DIR / "report.md"
    charts = {
        "sunshine_models_work_timeseries": models_work_timeseries,
        "sunshine_models_work_error": models_work_error,
        "sunshine_cams_breakdown": _cams_breakdown(contrasts=contrasts),
        "sunshine_new_products": _new_products(
            contrasts=contrasts,
            record_contrasts=report_contrasts(report_path=RECORD_DIR / "report.md"),
        ),
        "sunshine_era5_by_year": _era5_by_year(),
        "sunshine_icon_d2_leads": _icon_d2_leads(contrasts=contrasts, report_text=report_text),
        "sunshine_weather_model_rivals": _weather_model_rivals(contrasts=contrasts),
        "sunshine_own_beam": _own_beam(
            contrasts=contrasts,
            errors=errors,
            all_contrasts=report_contrasts(report_path=all_report_path),
            all_errors=report_errors(report_path=all_report_path, column="Global only"),
        ),
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
