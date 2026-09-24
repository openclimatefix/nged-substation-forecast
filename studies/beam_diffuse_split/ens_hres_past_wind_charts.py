"""Draw the charts for the ECMWF HRES and ENS section of the past-wind page.

One-off throwaway script for the charts of the ECMWF addition to
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>,
in `wind_icon_dream_charts.py`'s style. **Every number a chart shares with the page is read from
`intervals.parquet` or `report.md`, both written by `ens_hres_past_wind.py`**, so a chart cannot
disagree with the page. Before any chart is saved, `Source.verify` requires every interval a chart
draws, printed the way the report prints it, to be in `report.md`, and `_check_title_numbers`
requires every decimal number in a chart's title to be a report number rounded to the title's
precision.

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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from check_page_numbers import _report_numbers
from ens_hres_past_wind import OUTPUT_DIR
from studies.charts import (
    PLOT_WIDTH_PX,
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

SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous wind farm labels."""

NAMES: Final[dict[str, str]] = {
    "era5": "ERA5",
    "ukv": "UKV",
    "icon_d2": "ICON-D2",
    "icon_eu": "ICON-EU",
    "icon_global": "ICON global",
    "hres": "ECMWF HRES",
    "ens_mean_day0": "ECMWF ENS day-0 mean",
}
"""Every product's public name, as the page writes it."""

FAMILIES: Final[dict[str, str]] = {
    "era5": "reanalysis",
    "ukv": "weather model",
    "icon_d2": "weather model",
    "icon_eu": "weather model",
    "icon_global": "weather model",
    "hres": "weather model",
    "ens_mean_day0": "weather model",
}
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

FIGURE_LEADERBOARD: Final[int] = 13
FIGURE_MODELS_WORK: Final[int] = 14
FIGURE_PER_FARM_ERROR: Final[int] = 15
FIGURE_ROBUSTNESS: Final[int] = 16
FIGURE_RECONCILIATION: Final[int] = 17
FIGURE_MONTHLY_RATIO: Final[int] = 18
FIGURE_SPLIT: Final[int] = 19
FIGURE_BY_FARM: Final[int] = 20

DOMAIN_MARGIN: Final[float] = 0.15
"""How far past the lowest and highest value a difference chart's x domain extends."""

LEADERBOARD_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest interval end a leaderboard's x domain extends."""

RATIO_MARKER: Final[str] = "Monthly ratio of each product's mean 10 m wind speed to ERA5's"
"""The text that opens the sentence above the monthly-ratio table in `report.md`."""

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
            f"| {row['treatment']} | {value:.3f} | [{lower:.3f}, {upper:.3f}] "
            f"| {row['n_rows']:,} | {row['n_months']} |"
        )
    return (
        f"| {scope} | {row['treatment']} − {row['reference']} | {value:+.3f} "
        f"| [{lower:+.3f}, {upper:+.3f}] |"
    )


def _check_title_numbers(*, title: str, report: str) -> None:
    """Stop unless every decimal number in a title is a report number rounded to the title's places.

    Args:
        title: A chart's title.
        report: `report.md`'s text.

    Raises:
        ValueError: If a decimal number in the title matches no report number.
    """
    reported = _report_numbers(report_text=report)
    for printed in re.findall(r"\d+\.\d+", title):
        places = len(printed.split(".")[1])
        step = Decimal(1).scaleb(-places)
        rounded = {
            abs(Decimal(number)).quantize(step, rounding=ROUND_HALF_UP) for number in reported
        }
        if Decimal(printed) not in rounded:
            msg = f"the title number {printed} is not a report number rounded to {places} places"
            raise ValueError(msg)


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
    records = []
    for label, treatment, reference in contrasts:
        row = source.row(
            section=section,
            scope=scope,
            treatment=f"{treatment}_wind",
            reference=f"{reference}_wind",
            setting=setting,
        )
        records.append(
            {
                "label": label,
                "family": FAMILIES[treatment],
                "difference": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
                "planned": planned,
            }
        )
    return pl.DataFrame(records)


def _name_pair(*, treatment: str, reference: str) -> str:
    """Return a contrast's row label, such as `ECMWF HRES − UKV`."""
    return f"{NAMES[treatment]} − {NAMES[reference]}"


def _scope_line(*, source: Source) -> str:
    """Return the scope every chart states: the farms, the months, and the row and month counts.

    Args:
        source: The saved results.

    Returns:
        A sentence read from the report's heading and the pooled leaderboard.

    Raises:
        ValueError: If the report has no heading with a row count and dates.
    """
    match = re.search(
        r"on ([\d,]+) common farm-hours \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)",
        source.report,
    )
    if match is None:
        msg = "report.md has no 'on N common farm-hours (start to end)' heading"
        raise ValueError(msg)
    rows = int(match[1].replace(",", ""))
    first = datetime.fromisoformat(match[2])
    last = datetime.fromisoformat(match[3])
    board = source.intervals.filter(
        pl.col("section") == "leaderboard", pl.col("setting") == "pooled", pl.col("scope") == "all"
    )
    counts = board.select(rows=pl.col("n_rows").unique(), months=pl.col("n_months").unique())
    if counts["rows"].to_list() != [rows] or counts["months"].n_unique() != 1:
        msg = f"the report heading's {rows} rows disagree with intervals.parquet: {counts}"
        raise ValueError(msg)
    return (
        f"Three wind farms in Lincolnshire, {first:%B %Y} to {last:%B %Y}: {rows:,} farm-hours in "
        f"{counts['months'][0]} calendar months. Three farms are few independent sites."
    )


def _row_counts(*, source: Source) -> tuple[int, int]:
    """Return the pooled leaderboard's row and month counts, as `_scope_line` checked them."""
    board = source.intervals.filter(
        pl.col("section") == "leaderboard", pl.col("scope") == "all", pl.col("setting") == "pooled"
    )
    return board["n_rows"][0], board["n_months"][0]


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


def _headline(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw the leaderboard, the three planned contrasts, and the two exploratory contrasts.

    Args:
        source: The saved results.

    Returns:
        Figure 13, and its title.
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
    p1, p2, p3 = (planned["difference"][i] for i in range(3))
    title = (
        f"UKV beats ECMWF's HRES by {p1:.2f} points and ENS day 0 by {p2:.2f}, and HRES beats "
        f"ERA5 by {-p3:.2f}"
    )
    rows, months = _row_counts(source=source)
    return (
        figure(
            panels=[leaderboard, *panels],
            number=FIGURE_LEADERBOARD,
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
    value, averaged over the three fitting seeds. The weeks are chosen by the rule the page's
    Figure 4 uses, from measured power alone, so the choice cannot favour either product.

    Args:
        losses: The pooled setting's losses, every arm.

    Returns:
        Figure 14, and its title.
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
        "XGBoost models given HRES's or ENS day 0's wind follow the shape of measured power at "
        "every farm, in a windy, a variable, and a calm week"
    )
    return (
        _models_work_timeseries(
            long_frame=long_frame,
            sites=SITES,
            week_order=WIND_WEEK_DISPLAY_ORDER,
            order=order,
            colours=(ocf.TEXT, HRES_COLOUR, ENS_COLOUR),
            number=FIGURE_MODELS_WORK,
            title=title,
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
        Figure 15, and its title.

    Raises:
        ValueError: If the title's claim (ICON-D2 lowest and HRES ahead of ENS at every farm)
            does not hold in the saved results.
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
        "ICON-D2 has the lowest error at each of the three farms, and ECMWF HRES is ahead of ENS "
        "day 0 at each"
    )
    return (
        figure(
            panels=panels,
            number=FIGURE_PER_FARM_ERROR,
            figure_planning=None,
            title=title,
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


DESIGN_LABELS: Final[dict[str, str]] = {
    "study design": "The study's design: three eras, third era's folds rotated",
    "study design, May 2026 rows removed": "The study's design, May 2026 rows removed",
    "three eras, no fold rotation": "Three eras, no fold rotation",
    "two UKV eras (the page's design)": "Two UKV eras, as the rest of the page",
    "study folds, two-valued era_code": "The study's folds, era code with two values",
    "extra era cut at IFS 50r1, May 2026 dropped": "Extra era cut at IFS Cycle 50r1, May dropped",
}
"""Each fold design in `losses_fold_designs.parquet`, as the chart labels it, in the order drawn."""


def _robustness(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw P1 to P3 under each of six fold designs.

    Args:
        source: The saved results.

    Returns:
        Figure 16, and its title.

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
                planned=False,
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
                figure_planning="exploratory",
            )
        )
    title = (
        "Each planned contrast keeps its sign and stays statistically significant at the 5% level "
        "under all six fold designs"
    )
    return (
        figure(
            panels=panels,
            number=FIGURE_ROBUSTNESS,
            figure_planning="exploratory",
            title=title,
            subtitle=[
                (
                    "The three planned contrasts refitted under each fold design. The designs were "
                    "added after the first results, so every row here is exploratory."
                ),
                (
                    "A fold is a block of whole months held out for scoring. A design says where "
                    "the blocks are cut and what the XGBoost model is told about each hour's era."
                ),
                f"{DOTS} {CAPACITY}",
                _scope_line(source=source),
            ],
        ),
        title,
    )


LONG_DESIGNS: Final[tuple[tuple[str, str], ...]] = (
    ("long rows: two UKV eras, no cut at 49r1 (horizons design)", "Folds as on the horizons page"),
    ("long rows: extra era cut at 2024-12-01", "Extra era cut at 1 December 2024"),
)
"""The `intervals.parquet` section of each long-row-set design, and its condition name."""

LONG_SCOPES: Final[tuple[tuple[str, str], ...]] = (
    ("all rows", "all rows"),
    ("rows from 2024-12-01", "rows from 1 December 2024"),
)
"""Each long-row-set scope's cell in the report, and its label."""


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


def _long_rows_panel(
    *, source: Source, treatment: str, panel_title: str, x_domain: tuple[float, float], last: bool
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one long-row-set contrast under both fold designs, on two row sets.

    Args:
        source: The saved results.
        treatment: The treatment arm, such as `ens_mean_day0_wind`.
        panel_title: The panel's title.
        x_domain: The x range every panel shares.
        last: Whether this panel carries the x axis title.

    Returns:
        The panel.
    """
    records = []
    for scope, scope_label in LONG_SCOPES:
        for section, condition in LONG_DESIGNS:
            row = source.row(
                section=section,
                scope=scope,
                treatment=treatment,
                reference="era5_wind",
            )
            records.append(
                {
                    "label": scope_label[0].upper() + scope_label[1:],
                    "condition": condition,
                    "family": "weather model",
                    "difference": row["value"],
                    "lower_95": row["lower"],
                    "upper_95": row["upper"],
                    "planned": False,
                }
            )
    return interval_panel(
        rows=pl.DataFrame(records),
        x_domain=x_domain,
        x_title=X_TITLE if last else "",
        zero_label="no difference",
        better_label="first-named product better",
        conditions=[condition for _, condition in LONG_DESIGNS],
        condition_title="Fold design",
        panel_title=panel_title,
        reference_labels=not last,
        figure_planning="exploratory",
    )


def _reconciliation(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw ENS and HRES against ERA5 under the horizons page's folds and under an extra era cut.

    Args:
        source: The saved results.

    Returns:
        Figure 17, and its title.
    """
    published = _horizons_published(source=source)
    every = []
    counts: dict[str, tuple[int, int]] = {}
    for treatment in ("ens_mean_day0_wind", "hres_wind"):
        for section, _ in LONG_DESIGNS:
            for scope, _ in LONG_SCOPES:
                row = source.row(
                    section=section, scope=scope, treatment=treatment, reference="era5_wind"
                )
                counts[scope] = (row["n_rows"], row["n_months"])
                every.append({"lower_95": row["lower"], "upper_95": row["upper"]})
    domain = _difference_domain(rows=pl.DataFrame(every))
    panels = [
        _long_rows_panel(
            source=source,
            treatment="ens_mean_day0_wind",
            panel_title="ENS day 0 − ERA5, refitted here",
            x_domain=domain,
            last=False,
        ),
        _long_rows_panel(
            source=source,
            treatment="hres_wind",
            panel_title="ECMWF HRES − ERA5, refitted here",
            x_domain=domain,
            last=True,
        ),
    ]
    title = (
        "How the folds treat IFS Cycle 49r1 decides whether ENS day 0 trails ERA5 and whether "
        "HRES beats ERA5"
    )
    (all_rows, all_months), (late_rows, late_months) = (counts[scope] for scope, _ in LONG_SCOPES)
    published_row = published.row(0, named=True)
    return (
        figure(
            panels=panels,
            number=FIGURE_RECONCILIATION,
            figure_planning="exploratory",
            title=title,
            subtitle=[
                (
                    "ECMWF changed its forecast model, IFS Cycle 49r1, on 12 November 2024. The "
                    "ENS horizons page's folds do not treat that date as a break; the extra cut "
                    "does."
                ),
                (
                    f"All rows: {all_rows:,} farm-hours in {all_months} calendar months from 12 "
                    f"August 2024. Rows from 1 December 2024: {late_rows:,} in {late_months} "
                    "months, the rows the rest of this section scores."
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


def _monthly_ratios(*, report: str) -> pl.DataFrame:
    """Read the monthly ratio table of 10 m wind speed to ERA5's from `report.md`.

    Args:
        report: `report.md`'s text.

    Returns:
        One row per month, with `month` (the middle of the month), `product` and `ratio`.

    Raises:
        ValueError: If the table is missing or its header changed.
    """
    start = report.find(RATIO_MARKER)
    if start < 0:
        msg = f"report.md has no line starting {RATIO_MARKER!r}"
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


def _monthly_ratio_chart(*, source: Source) -> tuple[alt.VConcatChart, str]:
    """Draw each product's monthly mean 10 m wind speed over ERA5's, with IFS Cycle 49r1 marked.

    Args:
        source: The saved results.

    Returns:
        Figure 18, and its title.
    """
    ratios = _monthly_ratios(report=source.report)
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
                title="Month (points sit mid-month)",
                axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount="month", grid=False),
            ),
            y=alt.Y(
                "ratio:Q",
                title=["Mean 10 m wind speed over ERA5's", "(1 means equal)"],
                scale=alt.Scale(domain=[0.65, 1.0], nice=False, zero=False),
                axis=alt.Axis(format=".2f", values=[0.7, 0.8, 0.9, 1.0]),
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
    panel = alt.LayerChart(layer=[rule, label, line], width=PLOT_WIDTH_PX, height=230)
    title = (
        "ENS's and HRES's 10 m wind speeds fall against ERA5's between September and December "
        "2024, and UKV's does not"
    )
    return (
        figure(
            panels=[panel],
            number=FIGURE_MONTHLY_RATIO,
            figure_planning=None,
            title=title,
            subtitle=[
                (
                    "Each month's mean 10 m wind speed of one product, over ERA5's mean for the "
                    "same hours, pooled over the three farms."
                ),
                (
                    "Rows are the page's own from 12 August 2024, before any hour is dropped for "
                    "the fold design. Each month's row count is in the report."
                ),
                "Three wind farms in Lincolnshire. Three farms are few independent sites.",
            ],
        ),
        title,
    )


def _require_split_and_farm_claims(
    *, split: pl.DataFrame, farm_frames: list[tuple[str, str, str, pl.DataFrame]]
) -> None:
    """Stop unless the saved results support the title of the label-hour and by-farm figure.

    Args:
        split: The label-hour panel's rows, with `condition` and the interval.
        farm_frames: Each planned contrast's per-farm rows, pooled row first.

    Raises:
        ValueError: If the early hours are significant, the later hours are not, or P3 is
            significant at other than exactly one farm.
    """
    significant = (split["lower_95"] > 0) | (split["upper_95"] < 0)
    early = split.filter(pl.col("condition").str.starts_with("Labels 00-08"))
    if significant.filter(split["condition"].str.starts_with("Labels 00-08")).any():
        msg = f"an early-hours contrast is significant: {early}"
        raise ValueError(msg)
    if not significant.filter(split["condition"].str.starts_with("Labels 10-23")).all():
        msg = "a later-hours contrast is not significant"
        raise ValueError(msg)
    label, _, _, frame = farm_frames[-1]
    farms = frame.filter(pl.col("label") != "All three farms")
    at_farms = ((farms["lower_95"] > 0) | (farms["upper_95"] < 0)).sum()
    if label != "P3" or at_farms != 1:
        msg = f"{label} is significant at {at_farms} farms, not one"
        raise ValueError(msg)


def _split_and_farms(*, source: Source) -> dict[str, tuple[alt.VConcatChart, str]]:
    """Draw ENS's gaps by label hour, and the three planned contrasts at each farm, as two figures.

    Args:
        source: The saved results.

    Returns:
        Figures 19 and 20, each with its title, keyed by file name.
    """
    groups = (
        ("labels 00-08 UTC", "Labels 00-08 UTC"),
        ("labels 10-23 UTC", "Labels 10-23 UTC"),
    )
    split_records = []
    for scope, condition in groups:
        for reference in ("ukv", "hres"):
            row = source.row(
                section="lead and time of day",
                scope=scope,
                treatment="ens_mean_day0_wind",
                reference=f"{reference}_wind",
            )
            split_records.append(
                {
                    "label": _name_pair(treatment="ens_mean_day0", reference=reference),
                    "condition": f"{condition} ({row['n_rows']:,} rows)",
                    "family": "weather model",
                    "difference": row["value"],
                    "lower_95": row["lower"],
                    "upper_95": row["upper"],
                    "planned": False,
                }
            )
    split = pl.DataFrame(split_records)
    conditions = list(dict.fromkeys(split["condition"].to_list()))
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
                {
                    "label": scope_label,
                    "family": "weather model",
                    "difference": row["value"],
                    "lower_95": row["lower"],
                    "upper_95": row["upper"],
                    "planned": False,
                }
            )
        farm_frames.append((label, treatment, reference, pl.DataFrame(records)))
    _require_split_and_farm_claims(split=split, farm_frames=farm_frames)
    split_domain = _difference_domain(rows=split)
    split_panel = interval_panel(
        rows=split,
        x_domain=split_domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first-named product better",
        conditions=conditions,
        condition_title="Hours scored",
        panel_title="ENS by label hour",
        figure_planning="exploratory",
    )
    split_title = (
        "ENS day 0 trails UKV and HRES by a statistically significant margin only in the later "
        "hours of the day"
    )
    split_figure = figure(
        panels=[split_panel],
        number=FIGURE_SPLIT,
        figure_planning="exploratory",
        title=split_title,
        subtitle=[
            (
                "Labels 00-08 UTC are ENS leads 0 to 8 h from the 00 UTC run, labels 10-23 UTC "
                "leads 10 to 23 h. The split mixes lead with time of day and does not show "
                "whether ENS could be read in time."
            ),
            f"{DOTS} {CAPACITY}",
            _scope_line(source=source),
        ],
    )
    farm_domain = _difference_domain(rows=pl.concat(frame for *_, frame in farm_frames))
    farm_panels = [
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
            figure_planning="exploratory",
        )
        for index, (label, treatment, reference, frame) in enumerate(farm_frames)
    ]
    farm_title = "HRES beats ERA5 by a statistically significant margin at one farm of three"
    farm_figure = figure(
        panels=farm_panels,
        number=FIGURE_BY_FARM,
        figure_planning="exploratory",
        title=farm_title,
        subtitle=[
            (
                "Each planned contrast at each farm, and pooled over the three farms. The "
                "per-farm rows were not planned, so every row is exploratory."
            ),
            f"{DOTS} {CAPACITY}",
            _scope_line(source=source),
        ],
    )
    return {
        "ens_hres_wind_split": (split_figure, split_title),
        "ens_hres_wind_by_farm": (farm_figure, farm_title),
    }


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
    }
    charts.update(_split_and_farms(source=source))
    source.verify()
    _LOG.info("%d report lines checked", len(source.printed))
    for name, (chart, title) in charts.items():
        _check_title_numbers(title=title, report=source.report)
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s: %s", path, " ".join(wrapped(text=title)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
