"""Draw the eight anonymised charts for the write-up on which product best describes sunshine.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-solar/>.

**Every number is read from the report `weather_products.py` wrote**, so a chart cannot disagree
with the page.

Generators appear only as `A` to `F`, and no chart plots output.

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
from studies.charts import (
    CONTENT_WIDTH_PX,
    FAMILY_COLOURS,
    ContrastKey,
    ProductFamily,
    figure,
    flip_contrast,
    interval_panel,
    planning,
    report_contrasts,
    report_errors,
    select_contrasts,
)
from weather_products import (
    OUTPUT_DIR_NAME,
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

SECTION_DECIDING: Final[str] = "Deciding contrasts, named before the run"
SECTION_AGAINST_ERA5: Final[str] = "Every product against ERA5, by scope (exploratory)"
SECTION_POST_ONLY: Final[str] = "The post scope, fitted on post-upgrade rows alone"
SECTION_SPLIT: Final[str] = "A product's own split against Erbs on its own global"
SECTION_TRANSFER: Final[str] = (
    "Leave one site out, the scored months withheld everywhere: the deciding contrasts"
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

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, December 2022 to September 2026."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"


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


def _headline(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product against ERA5 above the four planned contrasts.

    Args:
        contrasts: Every contrast row in the report.
        errors: Each product's mean absolute error.

    Returns:
        Figure 1.
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
        number=1,
        figure_planning=figure_planning,
        title="CAMS describes past sunshine best of the six products tested, by a wide margin",
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
        Figure 2.
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
        number=2,
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
        Figure 3.
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
        number=3,
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
        Figure 4.
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
            "UKV's two snapshots given as a pair",
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
        number=4,
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
        Figure 5.
    """
    conditions = ("Open-Meteo's hourly value", "Rebuilt from its snapshots")
    scopes = {
        "all": "All hours",
        "ukv_live": "Since August 2024",
        "post": "After the upgrade: XGBoost models trained on both sides",
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
        number=5,
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
        Figure 6.
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
        number=6,
        figure_planning=figure_planning,
        title="Every product except ERA5 gains 0.03 to 0.11 points from its own direct beam",
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
        Figure 7.
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
        number=7,
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
        Figure 8.
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
        number=8,
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


def main() -> int:
    """Read the report, compute the new numbers, and write the eight SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = RESULTS_DIR / "report.md"
    report_text = report_path.read_text()
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="Global only")
    charts = {
        "sunshine_headline": _headline(contrasts=contrasts, errors=errors),
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
