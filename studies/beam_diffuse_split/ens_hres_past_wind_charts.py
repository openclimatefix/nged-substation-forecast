"""Draw the charts for the ECMWF HRES and ENS section of the past-wind page.

One-off throwaway script for the charts of the ECMWF addition to
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>,
in `wind_icon_dream_charts.py`'s style. **Every number a chart shares with the page is read from
`intervals.parquet` or `report.md`, both written by `ens_hres_past_wind.py`**, so a chart cannot
disagree with the page. Before any chart is saved, `Source.verify` requires every interval a chart
draws, printed the way the report prints it, to be in `report.md`. Each title that states a sign or
a significance checks it against the saved intervals before the chart is saved.

Generators appear only as `W1` to `W3`. The only chart that plots a generator's output is the
"models work" time series, which counts days 1 to 7 of a week and carries no calendar date. Every
mark is drawn with `aria=False`, so Vega does not write a point's value into the SVG. The monthly
ratio chart has calendar months on its axis because it pools all three farms and names none.

**Do not run this script until `ens_hres_past_wind.py` has fitted every arm, run `--extra-fits`,
and written its report.**

Run it with `uv run python studies/beam_diffuse_split/ens_hres_past_wind_charts.py`, after
`ens_hres_past_wind.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from ens_hres_past_wind import OUTPUT_DIR
from figure_numbers import WIND_FIGURE_NUMBERS, wind_figure_number, wind_figure_title
from studies.charts import (
    CONTENT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    wrapped,
)
from weather_product_charts import (
    ASSETS_DIR,
    CAPACITY,
    DOTS,
    LEADERBOARD_X_TITLE,
    X_TITLE,
    _models_work_long_frame,
    _models_work_timeseries,
    _pick_weeks,
    _reconstruct_predicted,
)
from wind_product_charts import (
    MODELS_WORK_MIN_HOURS,
    WIND_WEEK_CRITERIA,
    WIND_WEEK_DISPLAY_ORDER,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RATIO_PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 74
"""The plot width of a monthly-ratio panel, whose y axis title and tick labels take 74 px, so
that the panel and its axes fill the text column as an interval figure does."""

_CAPTION_CHARACTERS: Final[int] = 100
"""The characters a caption line holds before wrapping, which keeps it inside a 680 px figure."""

SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous wind farm labels."""

NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "era5": "ERA5",
        "ukv": "UKV",
        "icon_d2": "ICON-D2",
        "icon_eu": "ICON-EU",
        "icon_global": "ICON global",
        "hres": "ECMWF HRES",
        "ens_mean_day0": "ECMWF ENS day-0 mean",
    }
)
"""Every product's public name, as the page writes it."""

FAMILIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "era5": "reanalysis",
        "ukv": "weather model",
        "icon_d2": "weather model",
        "icon_eu": "weather model",
        "icon_global": "weather model",
        "hres": "weather model",
        "ens_mean_day0": "weather model",
    }
)
"""Every product's family, which sets its colour in `studies.charts`."""

HRES_COLOUR: Final[str] = ocf.DATA_BLUE
ENS_COLOUR: Final[str] = ocf.BRAND_ORANGE
"""The colours of the two ECMWF products where one chart draws both. Data Blue and Brand Orange pass
`validate_palette.py` (32.5 ΔE apart under protanopia, the worst case; both above 3:1 contrast
against white). A third colour fails the check against one of the two, so a third series is drawn
in `ocf.BLACK_1`, dashed."""

LEADERBOARD_ORDER: Final[tuple[str, ...]] = tuple(NAMES)
"""Every product the leaderboard draws."""

PER_FARM_PRODUCTS: Final[tuple[str, ...]] = tuple(
    product for product in NAMES if product != "icon_global"
)
"""The products drawn at each farm. ICON global is left out, because its served wind steps at one
farm and its per-farm error would show which."""

PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("P1", "hres", "ukv"),
    ("P2", "ens_mean_day0", "ukv"),
    ("P3", "hres", "era5"),
)
"""The three planned contrasts: label, treatment product, reference product."""

EXPLORATORY: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0", "hres"),
    ("ens_mean_day0", "era5"),
)
"""The two exploratory contrasts between the ECMWF products and the others."""

MODELS_WORK_AXIS_PX: Final[int] = 88
"""The width of a week panel's y axis and gutter beyond its plot area, measured on the saved SVG."""

MODELS_WORK_PANEL_WIDTH_PX: Final[int] = (CONTENT_WIDTH_PX - MODELS_WORK_AXIS_PX - 32) // 3
"""The plot width of each of the `models_work_timeseries` figure's three week columns, so the
figure fills the text column."""

DOMAIN_MARGIN: Final[float] = 0.15
"""How far past the lowest and highest value a difference chart's x domain extends."""

LEADERBOARD_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest interval end a leaderboard's x domain extends."""

RATIO_MARKERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "10 m": "Monthly ratio of each product's mean 10 m wind speed to ERA5's",
        "100 m": "Monthly ratio of each product's mean 100 m wind speed to ERA5's",
    }
)
"""The text that opens the sentence above each height's monthly-ratio table in `report.md`."""

RATIO_HEADER: Final[tuple[str, ...]] = ("Month", "Rows", "ENS / ERA5", "HRES / ERA5", "UKV / ERA5")
"""The header of the monthly-ratio table."""

IFS_49R1: Final[datetime] = datetime(2024, 11, 12, tzinfo=UTC)
"""The date IFS Cycle 49r1 went operational, drawn as a rule on the monthly-ratio chart."""

RUN_NOTES: Final[str] = (
    "ECMWF HRES is Open-Meteo's freshest-run forecast from ECMWF's high-resolution model. "
    "ECMWF ENS day-0 mean is the mean of ECMWF's 51 ensemble members for the hours 00 to 23 UTC "
    "of the 00 UTC run's own day."
)
"""What the two ECMWF products are, for a chart that names them."""


@dataclass
class Source:
    """The study's saved results, and every line a chart drew from them.

    Attributes:
        intervals: `intervals.parquet`, one row per interval `report.md` prints.
        report: `report.md`'s text.
        printed: The report line each interval a chart drew would have been printed as.
    """

    intervals: pl.DataFrame
    report: str
    printed: list[str] = field(default_factory=list)

    def row(
        self,
        *,
        section: str,
        scope: str,
        treatment: str,
        reference: str | None,
        setting: str = "pooled",
    ) -> dict[str, Any]:
        """Return the one interval that matches, and record the report line it must match.

        Args:
            section: The `intervals.parquet` section, such as `planned`.
            scope: The scope cell, such as `all` or `site W1`.
            treatment: The first arm, such as `hres_wind`.
            reference: The second arm of a contrast, or None for an arm's own error.
            setting: `pooled` or `sensitivity`.

        Returns:
            The row, as a dictionary.

        Raises:
            ValueError: If the key matches no row, or more than one.
        """
        matches = self.intervals.filter(
            pl.col("section") == section,
            pl.col("setting") == setting,
            pl.col("scope") == scope,
            pl.col("treatment") == treatment,
            (pl.col("reference") == reference)
            if reference is not None
            else pl.col("reference").is_null(),
        )
        if matches.height != 1:
            msg = f"{(section, setting, scope, treatment, reference)} matches {matches.height} rows"
            raise ValueError(msg)
        row = matches.row(0, named=True)
        self.printed.append(_printed_line(row=row, scope=scope))
        return row

    def verify(self) -> None:
        """Stop unless every recorded line appears in `report.md` exactly as printed.

        Raises:
            ValueError: Naming the lines the report does not hold.
        """
        missing = [line for line in self.printed if line not in self.report]
        if missing:
            msg = f"{len(missing)} intervals are not in report.md as printed, such as {missing[:3]}"
            raise ValueError(msg)


def _printed_line(*, row: dict[str, Any], scope: str) -> str:
    """Return the start of a report table row for one interval, as the report prints it.

    Args:
        row: An `intervals.parquet` row.
        scope: The scope cell of the row.

    Returns:
        A leaderboard row's arm, error, interval, rows and months, or a contrast row's scope,
        contrast, difference, and interval.
    """
    value, lower, upper = row["value"], row["lower"], row["upper"]
    if row["reference"] is None:
        return (
            f"| {row['treatment']} | {value:.4f} | [{lower:.4f}, {upper:.4f}] "
            f"| {row['n_rows']:,} | {row['n_months']} |"
        )
    return (
        f"| {scope} | {row['treatment']} − {row['reference']} | {value:+.4f} "
        f"| [{lower:+.4f}, {upper:+.4f}] |"
    )


def _difference_domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Return an x range holding every interval and zero, with a margin.

    Args:
        rows: Rows with `lower_95` and `upper_95`.

    Returns:
        The lowest and highest x value, rounded outwards to a tick step.
    """
    low = min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN
    high = max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN
    return (round(low * 10) / 10, round(high * 10) / 10)


def _difference_record(
    *,
    label: str,
    family: str,
    row: dict[str, Any],
    planned: bool,
    **extra: str,
) -> dict[str, Any]:
    """Return one difference row of an interval panel, from an `intervals.parquet` row.

    Args:
        label: The row's label on the panel.
        family: The family that sets the row's colour.
        row: The interval, from `Source.row`.
        planned: Whether the row is one of the study's three planned contrasts.
        **extra: Further columns, such as `condition`.

    Returns:
        The record, with `label`, `family`, `difference`, `lower_95`, `upper_95` and `planned`.
    """
    return {
        "label": label,
        "family": family,
        "difference": row["value"],
        "lower_95": row["lower"],
        "upper_95": row["upper"],
        "planned": planned,
        **extra,
    }


def _contrast_rows(
    *,
    source: Source,
    section: str,
    scope: str,
    contrasts: list[tuple[str, str, str]],
    planned: bool,
    setting: str = "pooled",
) -> pl.DataFrame:
    """Return one panel's rows: a label, the estimate and its interval, and the row's family.

    Args:
        source: The saved results.
        section: The `intervals.parquet` section.
        scope: The scope cell.
        contrasts: One (label, treatment product, reference product) per row.
        planned: Whether every row is a planned contrast.
        setting: `pooled` or `sensitivity`.

    Returns:
        The rows in the order of `contrasts`, with `label`, `family`, `difference`, `lower_95`,
        `upper_95` and `planned`.
    """
    return pl.DataFrame(
        [
            _difference_record(
                label=label,
                family=FAMILIES[treatment],
                row=source.row(
                    section=section,
                    scope=scope,
                    treatment=f"{treatment}_wind",
                    reference=f"{reference}_wind",
                    setting=setting,
                ),
                planned=planned,
            )
            for label, treatment, reference in contrasts
        ]
    )


def _name_pair(*, treatment: str, reference: str) -> str:
    """Return a contrast's row label, such as `ECMWF HRES − UKV`."""
    return f"{NAMES[treatment]} − {NAMES[reference]}"


def _row_counts(*, source: Source) -> tuple[int, int, datetime, datetime]:
    """Return the row and month counts every chart states, and the first and last day of the rows.

    Args:
        source: The saved results.

    Returns:
        The pooled leaderboard's row and month counts, and the first and last date, read from the
        report's heading.

    Raises:
        ValueError: If the report has no heading with a row count and dates, or the heading's row
            count disagrees with `intervals.parquet`.
    """
    match = re.search(
        r"on ([\d,]+) common farm-hours \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)",
        source.report,
    )
    if match is None:
        msg = "report.md has no 'on N common farm-hours (start to end)' heading"
        raise ValueError(msg)
    rows = int(match[1].replace(",", ""))
    board = source.intervals.filter(
        pl.col("section") == "leaderboard", pl.col("setting") == "pooled", pl.col("scope") == "all"
    )
    counts = board.select(rows=pl.col("n_rows").unique(), months=pl.col("n_months").unique())
    if counts["rows"].to_list() != [rows] or counts["months"].n_unique() != 1:
        msg = f"the report heading's {rows} rows disagree with intervals.parquet: {counts}"
        raise ValueError(msg)
    return (
        rows,
        counts["months"][0],
        datetime.fromisoformat(match[2]),
        datetime.fromisoformat(match[3]),
    )


def _scope_line(*, source: Source) -> str:
    """Return the scope every chart states: the farms, the months, and the row and month counts.

    Args:
        source: The saved results.

    Returns:
        A sentence read from the report's heading and the pooled leaderboard.
    """
    rows, months, first, last = _row_counts(source=source)
    return (
        f"Three wind farms in Lincolnshire, {first:%B %Y} to {last:%B %Y}: {rows:,} farm-hours in "
        f"{months} calendar months. Three farms are few independent sites."
    )


def _leaderboard_rows(*, source: Source, scope: str, products: tuple[str, ...]) -> pl.DataFrame:
    """Return each product's own error and 95% interval at one scope, best first.

    Args:
        source: The saved results.
        scope: `all` or `site W1`.
        products: The products to draw.

    Returns:
        Rows with `label`, `family`, `value`, `lower_95` and `upper_95`, sorted by error.
    """
    records = []
    for product in products:
        row = source.row(
            section="leaderboard",
            scope=scope,
            treatment=f"{product}_wind",
            reference=None,
        )
        records.append(
            {
                "label": NAMES[product],
                "product": product,
                "family": FAMILIES[product],
                "value": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
            }
        )
    return pl.DataFrame(records).sort("value")


def _require_headline_claims(*, planned: pl.DataFrame) -> None:
    """Stop unless the three planned contrasts support the leaderboard figure's title.

    Args:
        planned: The three planned contrasts' rows, P1 to P3 in order, with `difference`,
            `lower_95` and `upper_95`.

    Raises:
        ValueError: If HRES minus UKV or ENS day 0 minus UKV is not positive, HRES minus ERA5 is
            not negative, or any of the three intervals includes zero.
    """
    signs = [1 if difference > 0 else -1 for difference in planned["difference"].to_list()]
    significant = ((planned["lower_95"] > 0) | (planned["upper_95"] < 0)).to_list()
    if signs != [1, 1, -1] or not all(significant):
        msg = (
            f"Figure {WIND_FIGURE_NUMBERS['leaderboard']}'s title says UKV beats HRES and ENS "
            "day 0 and HRES beats ERA5, each "
            f"significantly; the signs are {signs} and significance is {significant}"
        )
        raise ValueError(msg)


def _headline(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw the leaderboard, the three planned contrasts, and the two exploratory contrasts.

    Args:
        source: The saved results.

    Returns:
        The `leaderboard` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    board = _leaderboard_rows(source=source, scope="all", products=LEADERBOARD_ORDER)
    board_domain = (
        min(board["lower_95"].to_list()) - LEADERBOARD_MARGIN,
        max(board["upper_95"].to_list()) + LEADERBOARD_MARGIN,
    )
    leaderboard = leaderboard_panel(
        rows=board.drop("product"),
        x_domain=board_domain,
        x_title=LEADERBOARD_X_TITLE,
        panel_title="Each product's own error",
        row_step_px=34,
    )
    planned = _contrast_rows(
        source=source,
        section="planned",
        scope="all",
        contrasts=[(_name_pair(treatment=t, reference=r), t, r) for _, t, r in PLANNED],
        planned=True,
    )
    exploratory = _contrast_rows(
        source=source,
        section="exploratory",
        scope="all",
        contrasts=[(_name_pair(treatment=t, reference=r), t, r) for t, r in EXPLORATORY],
        planned=False,
    )
    domain = _difference_domain(rows=pl.concat([planned, exploratory]))
    panels = [
        interval_panel(
            rows=rows,
            x_domain=domain,
            x_title=X_TITLE if last else "",
            zero_label="no difference",
            better_label="first-named product better",
            panel_title=panel_title,
            reference_labels=first,
            figure_planning="mixed",
        )
        for rows, panel_title, first, last in (
            (planned, "Paired differences: the three planned contrasts", True, False),
            (exploratory, "Paired differences: two exploratory contrasts", False, True),
        )
    ]
    _require_headline_claims(planned=planned)
    p1, p2, p3 = (planned["difference"][i] for i in range(3))
    title = (
        f"UKV beats ECMWF's HRES by {p1:.2f} points and ENS day 0 by {p2:.2f}, and HRES beats "
        f"ERA5 by {-p3:.2f}"
    )
    rows, months, _, _ = _row_counts(source=source)
    return (
        figure(
            panels=[leaderboard, *panels],
            number=WIND_FIGURE_NUMBERS["leaderboard"],
            figure_planning="mixed",
            title=title,
            subtitle=[
                (
                    f"Top: each product's own mean absolute error on the same {rows:,} farm-hours "
                    f"in {months} calendar months, best first. Below: paired differences."
                ),
                (
                    "Own intervals overlap more than paired ones, because every product's error "
                    "rises and falls with the month."
                ),
                RUN_NOTES,
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


def _pooled_week_frame(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (farm, hour) the study scored, with measured power and capacity.

    Args:
        losses: The pooled setting's losses, every arm.

    Returns:
        Rows with `site`, `time`, `power_mw` and `effective_capacity_mw`.

    Raises:
        ValueError: If two arms disagree about an hour's measured power.
    """
    per_arm = losses.group_by("arm", "site", "time").agg(
        power_mw=pl.col("actual_mw").first(),
        effective_capacity_mw=pl.col("effective_capacity_mw").first(),
    )
    measured = per_arm.group_by("site", "time").agg(
        power_mw=pl.col("power_mw").first(),
        spread=pl.col("power_mw").max() - pl.col("power_mw").min(),
        effective_capacity_mw=pl.col("effective_capacity_mw").first(),
    )
    if measured["spread"].max() != 0:
        msg = "the arms disagree about an hour's measured power"
        raise ValueError(msg)
    return measured.drop("spread")


def _models_work(*, losses: pl.DataFrame) -> tuple[alt.VConcatChart, str]:
    """Draw out-of-fold predictions against measured power for HRES and ENS, in three weeks.

    The prediction is the measured power plus the scored error, `signed_error_capped_mw`, which
    `studies.cross_validation` defines as the prediction held to the export cap minus the measured
    value, averaged over the three fitting seeds. The weeks are chosen by the rule the main row
    set's time-series figure uses, from measured power alone, so the choice cannot favour either
    product.

    Args:
        losses: The pooled setting's losses, every arm.

    Returns:
        The `models_work_timeseries` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    measured = _pooled_week_frame(losses=losses)
    products = ("hres", "ens_mean_day0")
    order = (
        "Measured",
        "XGBoost model given ECMWF HRES's wind",
        "XGBoost model given ECMWF ENS's day-0 wind",
    )
    predicted = tuple(
        (_reconstruct_predicted(losses=losses, measured=measured, arm=f"{product}_wind"), label)
        for product, label in zip(products, order[1:], strict=True)
    )
    hourly = measured.with_columns(
        output_frac=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw")
    )
    weeks = _pick_weeks(
        hourly=hourly, min_hours=MODELS_WORK_MIN_HOURS, agg="mean", criteria=WIND_WEEK_CRITERIA
    )
    for row in weeks.iter_rows(named=True):
        _LOG.info(
            "%s runs from %s to %s",
            row["label"],
            f"{row['week']:%B %Y}",
            f"{row['week'] + timedelta(days=6):%B %Y}",
        )
    long_frame = (
        _models_work_long_frame(measured=measured, predicted=predicted)
        .with_columns(week=pl.col("time").dt.truncate("1w"))
        .join(weeks, on="week", how="inner")
    )
    title = (
        "XGBoost models given HRES's or ENS day 0's wind follow the shape of measured power at W1 "
        "and W2, and at W3 outside its windiest week"
    )
    return (
        _models_work_timeseries(
            long_frame=long_frame,
            sites=SITES,
            week_order=WIND_WEEK_DISPLAY_ORDER,
            order=order,
            colours=(ocf.TEXT, HRES_COLOUR, ENS_COLOUR),
            number=wind_figure_number(key="models_work_timeseries", row_set="ecmwf"),
            title=wind_figure_title(row_set="ecmwf", title=title),
            panel_width=MODELS_WORK_PANEL_WIDTH_PX,
            subtitle=[
                "Out-of-fold power as a percentage of the generator's own capacity, days 1 to 7.",
                (
                    "Weeks are picked from measured power alone, pooled over the three farms: the "
                    "windiest has the highest mean output, the calmest the lowest, and the most "
                    "variable the largest swing in daily mean output."
                ),
                CAPACITY,
                "Three wind farms in Lincolnshire. Weeks carry no calendar date.",
            ],
        ),
        title,
    )


def _per_farm_error(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw each product's own error and 95% interval at each of the three farms.

    Args:
        source: The saved results.

    Returns:
        The `models_work_error` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.

    Raises:
        ValueError: If the title's claim does not hold in the saved results: ICON-D2's error is the
            lowest, and HRES's error is below ENS day 0's, at every farm.
    """
    per_farm = {
        site: _leaderboard_rows(source=source, scope=f"site {site}", products=PER_FARM_PRODUCTS)
        for site in SITES
    }
    for site, rows in per_farm.items():
        products = rows["product"].to_list()
        if products[0] != "icon_d2" or products.index("hres") > products.index("ens_mean_day0"):
            msg = f"farm {site}: the order {products} contradicts the title"
            raise ValueError(msg)
    every = pl.concat(per_farm.values())
    domain = (
        min(every["lower_95"].to_list()) - LEADERBOARD_MARGIN,
        max(every["upper_95"].to_list()) + LEADERBOARD_MARGIN,
    )
    panels = [
        leaderboard_panel(
            rows=rows.drop("product"),
            x_domain=domain,
            x_title=LEADERBOARD_X_TITLE if site == SITES[-1] else "",
            panel_title=f"Farm {site}",
            keys=site == SITES[0],
            row_step_px=26,
        )
        for site, rows in per_farm.items()
    ]
    title = (
        "ICON-D2 has the lowest error at each of the three farms, and ECMWF HRES's error is lower "
        "than ENS day 0's at each"
    )
    return (
        figure(
            panels=panels,
            number=wind_figure_number(key="models_work_error", row_set="ecmwf"),
            figure_planning=None,
            title=wind_figure_title(row_set="ecmwf", title=title),
            subtitle=[
                (
                    "Each product's own mean absolute error at one farm, best first. ICON global "
                    "is left out, because its served wind steps at one farm and its error there "
                    "would show which."
                ),
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


DESIGN_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "study design": "The study's design: three eras, third era's folds rotated",
        "study design, May 2026 rows removed": "The study's design, May 2026 rows removed",
        "three eras, no fold rotation": "Three eras, no fold rotation",
        "two UKV eras (the page's design)": "Two UKV eras, as the rest of the page",
        "study folds, two-valued era_code": "The study's folds, era code with two values",
        "extra era cut at IFS 50r1, May 2026 dropped": "Extra era cut at IFS 50r1, May dropped",
    }
)
"""Each fold design in `losses_fold_designs.parquet`, as the chart labels it, in the order drawn.

The first row is the study's own design, so its rows are the planned contrasts.
"""

STUDY_DESIGN: Final[str] = "study design"
"""The key of `DESIGN_LABELS` whose rows are the planned contrasts themselves."""


def _robustness(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw P1 to P3 under the study's design, one row subset of it, and four other fold designs.

    Args:
        source: The saved results.

    Returns:
        The `robustness` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.

    Raises:
        ValueError: If a contrast is not statistically significant at the 5% level under every
            design, which the title says it is.
    """
    panels = []
    frames = []
    for label, treatment, reference in PLANNED:
        rows = pl.concat(
            _contrast_rows(
                source=source,
                section="fold designs",
                scope=design,
                contrasts=[(design_label, treatment, reference)],
                planned=design == STUDY_DESIGN,
            )
            for design, design_label in DESIGN_LABELS.items()
        )
        if not ((rows["lower_95"] > 0) | (rows["upper_95"] < 0)).all():
            msg = f"{label} is not significant under every fold design, so the title is wrong"
            raise ValueError(msg)
        frames.append((label, treatment, reference, rows))
    domain = _difference_domain(rows=pl.concat(rows for *_, rows in frames))
    for index, (label, treatment, reference, rows) in enumerate(frames):
        panels.append(
            interval_panel(
                rows=rows,
                x_domain=domain,
                x_title=X_TITLE if index == len(frames) - 1 else "",
                zero_label="no difference",
                better_label="first-named product better",
                panel_title=f"{label}: {_name_pair(treatment=treatment, reference=reference)}",
                reference_labels=index == 0,
                figure_planning="mixed",
            )
        )
    without_may = source.row(
        section="fold designs",
        scope="study design, May 2026 rows removed",
        treatment="hres_wind",
        reference="ukv_wind",
    )
    title = (
        "Each planned contrast keeps its sign and stays statistically significant at the 5% level "
        "under five fold designs and one row subset"
    )
    return (
        figure(
            panels=panels,
            number=WIND_FIGURE_NUMBERS["robustness"],
            figure_planning="mixed",
            title=title,
            subtitle=[
                (
                    "The three planned contrasts, refitted under each fold design. The first row "
                    "of each panel is the planned contrast itself. The other designs were added "
                    "after the first results, so those rows are exploratory."
                ),
                (
                    "A fold is a block of whole months held out for scoring. A design says where "
                    "the blocks are cut and what the XGBoost model is told about each hour's era. "
                    "The second row scores the study's own fits without the rows of May 2026, the "
                    "rows the last design drops."
                ),
                (
                    "The two rows that drop May 2026 (the row subset and the IFS Cycle 50r1 "
                    f"design) score {without_may['n_rows']:,} farm-hours, and every other row "
                    "scores the farm-hours counted in the last line."
                ),
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


LONG_DESIGNS: Final[tuple[tuple[str, str], ...]] = (
    ("long rows: two UKV eras, no cut at 49r1 (horizons design)", "Folds as on the horizons page"),
    ("long rows: two UKV eras, no cut at 49r1, folds rotated", "Same eras, folds rotated"),
    ("long rows: extra era cut at 2024-12-01", "Extra era cut at 1 December 2024"),
)
"""The `intervals.parquet` section of each long-row-set design, and its condition name."""

LONG_SCOPES: Final[tuple[tuple[str, str], ...]] = (
    ("all rows", "all rows"),
    ("rows from 2024-12-01", "rows from 1 December 2024"),
)
"""Each long-row-set scope's cell in the report, and its label."""

LONG_TREATMENTS: Final[tuple[str, ...]] = ("ens_mean_day0_wind", "hres_wind")
"""The two ECMWF arms the `reconciliation` figure draws against ERA5."""


def _horizons_published(*, source: Source) -> pl.DataFrame:
    """Return the horizons page's own ENS-against-ERA5 figure, as `report.md` quotes it.

    Args:
        source: The saved results.

    Returns:
        One row, with `label`, `family`, `difference`, `lower_95`, `upper_95` and `planned`.

    Raises:
        ValueError: If the report does not quote the figure.
    """
    match = re.search(
        r"gives `ens_mean_day0 − era5` for wind, pooled setting, as ([+-][\d.]+) "
        r"\[([+-][\d.]+), ([+-][\d.]+)\] pp of capacity on ([\d,]+) rows from (\d{4}-\d{2}-\d{2})",
        source.report,
    )
    if match is None:
        msg = "report.md does not quote the horizons study's ENS-against-ERA5 figure"
        raise ValueError(msg)
    return pl.DataFrame(
        {
            "label": [f"ENS − ERA5, the horizons page's {match[4]} rows"],
            "family": ["weather model"],
            "difference": [float(match[1])],
            "lower_95": [float(match[2])],
            "upper_95": [float(match[3])],
            "planned": [False],
        }
    )


def _uncovered_cells(*, source: Source, design: str) -> int:
    """Return how many cells a design leaves without training rows, from the report's table.

    Args:
        source: The saved results.
        design: The design's name in the report's coverage table, without the section prefix
            that begins "long rows".

    Returns:
        The count of (site, fold, calendar month) cells with no training row for a calendar month
        that occurs in two years.

    Raises:
        ValueError: If the report's coverage tables hold no row for the design.
    """
    for line in source.report.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == design and len(cells) == 4:
            return int(cells[2])
    msg = f"report.md has no coverage row for {design!r}"
    raise ValueError(msg)


def _long_rows_panel(
    *, source: Source, treatment: str, panel_title: str, x_domain: tuple[float, float], last: bool
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one long-row-set contrast under every fold design, on two row sets.

    Args:
        source: The saved results.
        treatment: The treatment arm, such as `ens_mean_day0_wind`.
        panel_title: The panel's title.
        x_domain: The x range every panel shares.
        last: Whether this panel carries the x axis title.

    Returns:
        The panel.
    """
    records = [
        _difference_record(
            label=scope_label[0].upper() + scope_label[1:],
            family="weather model",
            row=source.row(
                section=section, scope=scope, treatment=treatment, reference="era5_wind"
            ),
            planned=False,
            condition=condition,
        )
        for scope, scope_label in LONG_SCOPES
        for section, condition in LONG_DESIGNS
    ]
    return interval_panel(
        rows=pl.DataFrame(records),
        x_domain=x_domain,
        x_title=X_TITLE if last else "",
        zero_label="no difference",
        better_label="first-named product better",
        conditions=[condition for _, condition in LONG_DESIGNS],
        condition_colours=(HRES_COLOUR, ENS_COLOUR, ocf.BLACK_1),
        condition_title="Fold design",
        panel_title=panel_title,
        reference_labels=not last,
        figure_planning="exploratory",
    )


def _require_reconciliation_claims(*, source: Source) -> None:
    """Stop unless an era cut at IFS Cycle 49r1 moves the scores far more than rotating folds does.

    Args:
        source: The saved results.

    Raises:
        ValueError: If, for either ECMWF arm on either row set, rotating the folds moves the
            difference from ERA5 by as much as the smallest move the extra era cut makes from
            either design without it, or the cut moves it the wrong way.
    """
    horizons, rotated, cut = (section for section, _ in LONG_DESIGNS)
    for treatment in LONG_TREATMENTS:
        for scope, _ in LONG_SCOPES:
            value = {
                section: source.row(
                    section=section, scope=scope, treatment=treatment, reference="era5_wind"
                )["value"]
                for section in (horizons, rotated, cut)
            }
            rotation_move = abs(value[rotated] - value[horizons])
            cut_moves = [value[without] - value[cut] for without in (horizons, rotated)]
            if min(cut_moves) <= 0 or rotation_move >= min(cut_moves):
                msg = (
                    f"{treatment} on {scope}: rotating folds moves the difference by "
                    f"{rotation_move:.3f} and the era cut by {cut_moves}, so Figure "
                    f"{WIND_FIGURE_NUMBERS['reconciliation']}'s title is wrong"
                )
                raise ValueError(msg)


def _reconciliation(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw ENS and HRES against ERA5 under three fold designs.

    Args:
        source: The saved results.

    Returns:
        The `reconciliation` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    _require_reconciliation_claims(source=source)
    published = _horizons_published(source=source)
    every = pl.DataFrame(
        [
            {"lower_95": row["lower"], "upper_95": row["upper"]}
            for treatment in LONG_TREATMENTS
            for section, _ in LONG_DESIGNS
            for scope, _ in LONG_SCOPES
            for row in [
                source.row(section=section, scope=scope, treatment=treatment, reference="era5_wind")
            ]
        ]
    )
    domain = _difference_domain(rows=every)
    panels = [
        _long_rows_panel(
            source=source,
            treatment=treatment,
            panel_title=panel_title,
            x_domain=domain,
            last=last,
        )
        for treatment, panel_title, last in (
            ("ens_mean_day0_wind", "ENS day 0 − ERA5, refitted here", False),
            ("hres_wind", "ECMWF HRES − ERA5, refitted here", True),
        )
    ]
    title = (
        "An extra era cut at 1 December 2024, the first whole month after IFS Cycle 49r1, changes "
        "ENS day 0's and HRES's scores against ERA5 far more than rotating the folds does"
    )
    late = source.row(
        section=LONG_DESIGNS[0][0],
        scope=LONG_SCOPES[1][0],
        treatment=LONG_TREATMENTS[0],
        reference="era5_wind",
    )
    all_rows = source.row(
        section=LONG_DESIGNS[0][0],
        scope=LONG_SCOPES[0][0],
        treatment=LONG_TREATMENTS[0],
        reference="era5_wind",
    )
    uncovered = [
        _uncovered_cells(source=source, design=section.removeprefix("long rows: "))
        for section, _ in LONG_DESIGNS
    ]
    published_row = published.row(0, named=True)
    return (
        figure(
            panels=panels,
            number=WIND_FIGURE_NUMBERS["reconciliation"],
            figure_planning="exploratory",
            title=title,
            subtitle=[
                (
                    "ECMWF changed its forecast model, IFS Cycle 49r1, on 12 November 2024. The "
                    "ENS horizons page's folds do not treat that change as a break; the extra "
                    "cut, at 1 December 2024, does."
                ),
                (
                    f"All rows: {all_rows['n_rows']:,} farm-hours in {all_rows['n_months']} "
                    f"calendar months from 12 August 2024. Rows from 1 December 2024: "
                    f"{late['n_rows']:,} in {late['n_months']} months, the rows the rest of this "
                    "section scores."
                ),
                (
                    "Fold cells with no training row for a calendar month that occurs in two "
                    f"years: {uncovered[0]} under the horizons page's folds, {uncovered[1]} with "
                    f"those folds rotated, and {uncovered[2]} with the extra cut."
                ),
                (
                    "The ENS horizons page published ENS day 0 − ERA5 as "
                    f"{published_row['difference']:+.3f} points "
                    f"[{published_row['lower_95']:+.3f}, {published_row['upper_95']:+.3f}] on its "
                    "own rows and folds."
                ),
                f"{DOTS} {CAPACITY}",
                "Three wind farms in Lincolnshire. Three farms are few independent sites.",
            ],
        ),
        title,
    )


def _monthly_ratios(*, report: str, height: str) -> pl.DataFrame:
    """Read one height's monthly ratio table of wind speed to ERA5's from `report.md`.

    Args:
        report: `report.md`'s text.
        height: A key of `RATIO_MARKERS`.

    Returns:
        One row per month, with `month` (the middle of the month), `product` and `ratio`.

    Raises:
        ValueError: If the table is missing or its header changed.
    """
    marker = RATIO_MARKERS[height]
    start = report.find(marker)
    if start < 0:
        msg = f"report.md has no line starting {marker!r}"
        raise ValueError(msg)
    table = []
    for line in report[start:].splitlines():
        if line.startswith("|"):
            table.append(tuple(cell.strip() for cell in line.strip().strip("|").split("|")))
        elif table:
            break
    if not table or table[0] != RATIO_HEADER:
        msg = f"the monthly ratio table's header is {table[:1]}, not {RATIO_HEADER}"
        raise ValueError(msg)
    records = []
    for month, _, ens, hres, ukv in table[2:]:
        middle = datetime.strptime(month, "%Y-%m").replace(day=15, tzinfo=UTC)
        records += [
            {"month": middle, "product": "ECMWF ENS day-0 mean", "ratio": float(ens)},
            {"month": middle, "product": "ECMWF HRES", "ratio": float(hres)},
            {"month": middle, "product": "UKV", "ratio": float(ukv)},
        ]
    return pl.DataFrame(records)


def _ratio_panel(
    *,
    ratios: pl.DataFrame,
    height: str,
    domain: tuple[float, float],
    ticks: list[float],
    last: bool,
) -> alt.LayerChart:
    """Draw one height's monthly ratios, with IFS Cycle 49r1 marked.

    Args:
        ratios: `_monthly_ratios`'s result.
        height: `10 m` or `100 m`, for the axis title.
        domain: The y range.
        ticks: The y axis tick values.
        last: Whether this panel carries the x axis title and the legend.

    Returns:
        The panel.
    """
    order = ["ECMWF ENS day-0 mean", "ECMWF HRES", "UKV"]
    colours = [ENS_COLOUR, HRES_COLOUR, ocf.BLACK_1]
    dashes = [[1, 0], [6, 3], [2, 2]]
    line = (
        alt.Chart(ratios)
        .mark_line(
            strokeWidth=2, aria=False, point=alt.OverlayMarkDef(size=40, filled=True, aria=False)
        )
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:T",
                title="Month (points sit mid-month)" if last else None,
                axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount="month", grid=False),
            ),
            y=alt.Y(
                "ratio:Q",
                title=[f"Mean {height} wind speed over ERA5's", "(1 means equal)"],
                scale=alt.Scale(domain=list(domain), nice=False, zero=False),
                axis=alt.Axis(format=".2f", values=ticks),
            ),
            color=alt.Color(
                "product:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colours),
                legend=alt.Legend(orient="bottom", direction="horizontal", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "product:N", sort=order, scale=alt.Scale(domain=order, range=dashes), legend=None
            ),
        )
    )
    marker = pl.DataFrame({"date": [IFS_49R1], "text": ["IFS Cycle 49r1, 12 November 2024"]})
    rule = (
        alt.Chart(marker)
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 4], aria=False)
        .encode(x="date:T")  # ty: ignore[unresolved-attribute]
    )
    label = (
        alt.Chart(marker)
        .mark_text(align="left", baseline="top", dx=4, dy=2, color=ocf.BLACK_1, aria=False)
        .encode(x="date:T", y=alt.value(0), text="text:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=[rule, label, line], width=RATIO_PLOT_WIDTH_PX, height=200)


def _monthly_ratio_chart(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw each product's monthly mean 10 m and 100 m wind speed over ERA5's, with 49r1 marked.

    Args:
        source: The saved results.

    Returns:
        The `monthly_ratio` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    panels = [
        _ratio_panel(
            ratios=_monthly_ratios(report=source.report, height="10 m"),
            height="10 m",
            domain=(0.65, 1.0),
            ticks=[0.7, 0.8, 0.9, 1.0],
            last=False,
        ),
        _ratio_panel(
            ratios=_monthly_ratios(report=source.report, height="100 m"),
            height="100 m",
            domain=(0.85, 1.1),
            ticks=[0.9, 0.95, 1.0, 1.05],
            last=True,
        ),
    ]
    title = (
        "ENS's and HRES's 10 m wind speeds fall against ERA5's from October to November 2024, "
        "and UKV's does not"
    )
    return (
        figure(
            panels=panels,
            number=WIND_FIGURE_NUMBERS["monthly_ratio"],
            figure_planning=None,
            title=title,
            subtitle=[
                line
                for text in (
                    (
                        "Each month's mean wind speed of one product, over ERA5's mean for the "
                        "same hours, pooled over the three farms. The upper panel is 10 m. The "
                        "lower panel is 100 m, the height the XGBoost models are given, where the "
                        "fall is smaller and gradual."
                    ),
                    (
                        "Rows are the page's own from 12 August 2024, before any hour is dropped "
                        "for the fold design. Each month's row count is in the report."
                    ),
                    "Three wind farms in Lincolnshire. Three farms are few independent sites.",
                )
                for line in wrapped(text=text, width=_CAPTION_CHARACTERS)
            ],
        ),
        title,
    )


SPLIT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0", "ukv"),
    ("ens_mean_day0", "hres"),
    ("ens_mean_day0", "era5"),
    ("hres", "ukv"),
    ("ukv", "era5"),
)
"""The contrasts the `time_of_day` figure draws by label hour. The last two involve no ENS lead,
and ERA5, an analysis, has no lead at all, so they show how much of a change between the halves
is time of day."""

SPLIT_GROUPS: Final[tuple[tuple[str, str], ...]] = (
    ("labels 00-08 UTC", "Labels 00-08 UTC"),
    ("labels 10-23 UTC", "Labels 10-23 UTC"),
)
"""Each label-hour half's scope cell in the report, and its condition name in the figure."""


def _require_split_claims(*, source: Source) -> None:
    """Stop unless the saved results support the `time_of_day` figure's title.

    Args:
        source: The saved results.

    Raises:
        ValueError: If ENS day 0's error is not further above UKV's and HRES's in the later hours
            than in the early hours, or UKV's error is not further below ERA5's.
    """
    value = {
        (scope, treatment, reference): source.row(
            section="lead and time of day",
            scope=scope,
            treatment=f"{treatment}_wind",
            reference=f"{reference}_wind",
        )["value"]
        for scope, _ in SPLIT_GROUPS
        for treatment, reference in SPLIT_CONTRASTS
    }
    (early, _), (late, _) = SPLIT_GROUPS
    grows = [
        (treatment, reference)
        for treatment, reference in (("ens_mean_day0", "ukv"), ("ens_mean_day0", "hres"))
        if value[late, treatment, reference] > value[early, treatment, reference]
    ]
    ukv_lead_grows = value[late, "ukv", "era5"] < value[early, "ukv", "era5"]
    if len(grows) != 2 or not ukv_lead_grows:
        msg = (
            f"Figure {WIND_FIGURE_NUMBERS['time_of_day']}'s title is wrong: the ENS gaps that grow "
            f"in the later hours are {grows}, and UKV's lead over ERA5 grows: {ukv_lead_grows}"
        )
        raise ValueError(msg)


def _change_lines(*, source: Source) -> list[str]:
    """Return the subtitle lines giving the change of four contrasts between the two halves.

    Args:
        source: The saved results.

    Returns:
        One line, read from the report's `label-hour change` rows.
    """
    parts = []
    for treatment, reference in (
        ("ens_mean_day0", "ukv"),
        ("ukv", "era5"),
        ("ens_mean_day0", "era5"),
        ("ens_mean_day0", "hres"),
    ):
        row = source.row(
            section="label-hour change",
            scope="labels 10-23 UTC minus labels 00-08 UTC",
            treatment=f"{treatment}_wind",
            reference=f"{reference}_wind",
        )
        parts.append(
            f"{_name_pair(treatment=treatment, reference=reference)} "
            f"{row['value']:+.2f} [{row['lower']:+.2f}, {row['upper']:+.2f}]"
        )
    return [
        "Change from labels 00-08 UTC to labels 10-23 UTC, in points, exploratory: "
        + "; ".join(parts)
        + "."
    ]


def _split_figure(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw ENS's and the control contrasts' gaps by label hour.

    Args:
        source: The saved results.

    Returns:
        The `time_of_day` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    _require_split_claims(source=source)
    records = []
    for scope, condition in SPLIT_GROUPS:
        for treatment, reference in SPLIT_CONTRASTS:
            row = source.row(
                section="lead and time of day",
                scope=scope,
                treatment=f"{treatment}_wind",
                reference=f"{reference}_wind",
            )
            records.append(
                _difference_record(
                    label=_name_pair(treatment=treatment, reference=reference),
                    family="weather model",
                    row=row,
                    planned=False,
                    condition=f"{condition} ({row['n_rows']:,} rows)",
                )
            )
    split = pl.DataFrame(records)
    panel = interval_panel(
        rows=split,
        x_domain=_difference_domain(rows=split),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first-named product better",
        conditions=list(dict.fromkeys(split["condition"].to_list())),
        condition_title="Hours scored",
        panel_title="Each contrast by label hour",
        figure_planning="exploratory",
    )
    title = (
        "ENS day 0's gap to UKV and HRES is larger in the later hours of the day, as is ERA5's gap "
        "to UKV"
    )
    return (
        figure(
            panels=[panel],
            number=WIND_FIGURE_NUMBERS["time_of_day"],
            figure_planning="exploratory",
            title=title,
            subtitle=[
                (
                    "Labels 00-08 UTC are ENS leads 0 to 8 h from the 00 UTC run, labels 10-23 UTC "
                    "leads 10 to 23 h. The split mixes lead with time of day and does not show "
                    "whether ENS could be read in time."
                ),
                (
                    "UKV against ERA5 involves no forecast lead, but its change between the "
                    "halves is not time of day alone: HRES's served lead varies with the UTC "
                    "hour before 1 October 2025, and ERA5's assimilation windows change at 09 to "
                    "10 and 21 to 22 UTC. The split cannot apportion the widening between ENS "
                    "lead and time of day."
                ),
                *_change_lines(source=source),
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


def _farm_figure(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw the three planned contrasts at each farm and pooled.

    Args:
        source: The saved results.

    Returns:
        The `per_generator` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    farm_frames = []
    for label, treatment, reference in PLANNED:
        records = []
        for scope, scope_label in (("all", "All three farms"), *((f"site {s}", s) for s in SITES)):
            row = source.row(
                section="planned" if scope == "all" else "planned by farm",
                scope=scope,
                treatment=f"{treatment}_wind",
                reference=f"{reference}_wind",
            )
            records.append(
                _difference_record(
                    label=scope_label,
                    family="weather model",
                    row=row,
                    planned=scope == "all",
                )
            )
        farm_frames.append((label, treatment, reference, pl.DataFrame(records)))
    farm_domain = _difference_domain(rows=pl.concat(frame for *_, frame in farm_frames))
    panels = [
        interval_panel(
            rows=frame,
            x_domain=farm_domain,
            x_title=X_TITLE if index == len(farm_frames) - 1 else "",
            zero_label="no difference",
            better_label="first-named product better",
            panel_title=(
                f"{label} at each farm: {_name_pair(treatment=treatment, reference=reference)}"
            ),
            reference_labels=index == 0,
            figure_planning="mixed",
        )
        for index, (label, treatment, reference, frame) in enumerate(farm_frames)
    ]
    title = "The three planned contrasts pooled over the three farms and at each farm"
    between = source.row(
        section="between farms",
        scope="W2 minus W1",
        treatment="hres_wind",
        reference="era5_wind",
    )
    other_between = [
        source.row(
            section="between farms",
            scope=scope,
            treatment="hres_wind",
            reference="era5_wind",
        )
        for scope in ("W2 minus W3", "W3 minus W1")
    ]
    return (
        figure(
            panels=panels,
            number=wind_figure_number(key="per_generator", row_set="ecmwf"),
            figure_planning="mixed",
            title=wind_figure_title(row_set="ecmwf", title=title),
            subtitle=[
                (
                    "Each planned contrast pooled over the three farms, which is the planned "
                    "row, and at each farm. The per-farm rows were not planned, so they are "
                    "exploratory."
                ),
                (
                    "HRES − ERA5 differs between farms by "
                    f"{between['value']:+.2f} points [{between['lower']:+.2f}, "
                    f"{between['upper']:+.2f}] for W2 minus W1, "
                    f"{other_between[0]['value']:+.2f} [{other_between[0]['lower']:+.2f}, "
                    f"{other_between[0]['upper']:+.2f}] for W2 minus W3, and "
                    f"{other_between[1]['value']:+.2f} [{other_between[1]['lower']:+.2f}, "
                    f"{other_between[1]['upper']:+.2f}] for W3 minus W1 (exploratory)."
                ),
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


def main() -> int:
    """Read the saved results, check every number, and write the SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = Source(
        intervals=pl.read_parquet(OUTPUT_DIR / "intervals.parquet"),
        report=(OUTPUT_DIR / "report.md").read_text(),
    )
    losses = pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")
    charts: dict[str, tuple[alt.VConcatChart, str]] = {
        "ens_hres_wind_leaderboard": _headline(source=source),
        "ens_hres_wind_models_work": _models_work(losses=losses),
        "ens_hres_wind_per_farm_error": _per_farm_error(source=source),
        "ens_hres_wind_robustness": _robustness(source=source),
        "ens_hres_wind_reconciliation": _reconciliation(source=source),
        "ens_hres_wind_monthly_ratio": _monthly_ratio_chart(source=source),
        "ens_hres_wind_split": _split_figure(source=source),
        "ens_hres_wind_by_farm": _farm_figure(source=source),
    }
    source.verify()
    _LOG.info("%d report lines checked", len(source.printed))
    for name, (chart, title) in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s: %s", path, " ".join(wrapped(text=title)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
