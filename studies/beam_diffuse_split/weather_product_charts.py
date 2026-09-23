"""Draw the eight anonymised charts for the write-up on which product best describes sunshine.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-solar/>.

**Almost every number is read from the report `weather_products.py` wrote**, so a chart cannot
disagree with the page. Two charts need numbers the report lacks. The hour-by-hour chart needs an
interval at each hour, bootstrapped from `losses.parquet` after the report's "both at lead 1 h" row
has been reproduced through the same code path. The implied-capacity chart needs all 12 calendar
months, computed from the frame `weather_products.main` builds after that frame has reproduced the
report's implied-capacity table line for line. The script raises rather than draw a chart whose
anchor does not match.

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
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from export_cap import with_export_cap
from run_experiment import _add_time_features
from sources import STUDY_DATA_DIR
from studies.bootstrap import bootstrap_difference
from studies.charts import (
    FAMILY_COLOURS,
    NAMED_SUFFIX,
    ContrastKey,
    ProductFamily,
    figure,
    flip_contrast,
    interval_panel,
    report_contrasts,
    report_errors,
    select_contrasts,
)
from weather_products import (
    LEAD_TABLE_HOURS,
    METRIC,
    OUTPUT_DIR_NAME,
    PERCENTAGE_POINTS,
    _common_rows,
    _contrast_line,
    _implied_capacity,
    _joined,
    _log_capacity_by_month,
    _served_lead,
    _with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = STUDY_DATA_DIR / OUTPUT_DIR_NAME
"""Where `weather_products.py` wrote its report and losses."""

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

COMMON_ROWS: Final[int] = 79_384
"""The common site-hours the report states, which the rebuilt frame must hold."""

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

DECIDING: Final[tuple[tuple[str, str], ...]] = (
    ("cams_global", "icon_d2_global"),
    ("icon_eu_global", "icon_d2_global"),
    ("icon_eu_global", "ukv_global"),
    ("icon_global_global", "icon_eu_global"),
)
"""The four contrasts named before the run, as the report writes them."""

HEADLINE_DOMAIN: Final[tuple[float, float]] = (-4.5, 1.0)
"""The x range of the headline's left panel."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "6 solar farms in Lincolnshire, December 2022 to September 2026."
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
WIDTH: Final[int] = 900


def _product(arm: str) -> str:
    """Return the product an arm belongs to, such as `icon_eu` for `icon_eu_ctx_global`."""
    return max((product for product in NAMES if arm.startswith(f"{product}_")), key=len)


def _two_places(value: float) -> str:
    """Round a report's three-decimal number to two places, half up, as the page does."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _rows(*, contrasts: pl.DataFrame, labels: list[str]) -> pl.DataFrame:
    """Give selected report rows the label and family a panel draws them with.

    Args:
        contrasts: Rows from `select_contrasts`, in drawing order.
        labels: One label per row.

    Returns:
        The rows with `label` and `family`, the family being the first product's.
    """
    return contrasts.with_columns(
        label=pl.Series(labels),
        family=pl.Series([FAMILIES[_product(arm)] for arm in contrasts["treatment"]]),
    )


def _contrast_name(*, treatment: str, reference: str) -> str:
    """Return a contrast as the page writes it, such as `CAMS − ICON-D2`."""
    return f"{NAMES[_product(treatment)]} − {NAMES[_product(reference)]}"


def _headline(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product against ERA5 beside the four contrasts named before the run.

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
    left = interval_panel(
        rows=_rows(
            contrasts=by_product,
            labels=[f"{NAMES[product]} · {_two_places(errors[product])}%" for product in order],
        ),
        x_domain=HEADLINE_DOMAIN,
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        panel_title="Every product against ERA5 (exploratory)",
        width=300,
    )
    named = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_DECIDING, "all", t, r) for t, r in DECIDING],
    )
    right = interval_panel(
        rows=_rows(
            contrasts=named,
            labels=[_contrast_name(treatment=t, reference=r) + NAMED_SUFFIX for t, r in DECIDING],
        ),
        x_domain=(-3.0, 1.0),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        panel_title="The four contrasts named before the run",
        width=300,
    )
    return figure(
        panels=[left, right],
        number=1,
        title="CAMS describes past sunshine best, by a wide margin",
        subtitle=[
            (
                "Left: each product's mean absolute error minus ERA5's. Each label gives the "
                "product's own error, as a percentage of capacity."
            ),
            f"Right: the four contrasts named before the run. {CAPACITY}",
            f"{DOTS} {SCOPE}",
            (
                "Intervals on the left are against ERA5. Overlapping intervals do not mean two "
                "products are indistinguishable; the right-hand panel compares them directly."
            ),
        ],
        width=WIDTH,
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
        "Whole record": [("all", "All hours" + NAMED_SUFFIX, SECTION_DECIDING)],
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
    panels = [
        interval_panel(
            rows=_rows(
                contrasts=select_contrasts(
                    contrasts=contrasts,
                    wanted=[
                        ContrastKey(section, scope, "cams_global", "icon_d2_global")
                        for scope, _, section in rows
                    ],
                ),
                labels=[label for _, label, _ in rows],
            ),
            x_domain=(-4.0, 0.5),
            x_title=X_TITLE if name == "By calendar year" else "",
            zero_label="same as ICON-D2",
            better_label="CAMS better",
            panel_title=name,
            reference_labels=name == "Whole record",
            width=460,
        )
        for name, rows in groups.items()
    ]
    return figure(
        panels=panels,
        number=2,
        title="CAMS's lead over ICON-D2 holds at every generator, in every season, and every year",
        subtitle=[
            f"CAMS's mean absolute error minus ICON-D2's, in points of capacity. {CAPACITY}",
            f"{DOTS} The breakdowns are exploratory.",
            SCOPE,
        ],
        width=700,
        direction="vertical",
    )


def _hourly_rows(*, report_text: str) -> pl.DataFrame:
    """Bootstrap ICON-D2 − ICON-EU at each hour from 09 to 16 UTC, after reproducing a report row.

    Args:
        report_text: The report, which must hold the "both at lead 1 h" row this function
            recomputes.

    Returns:
        One row per hour with `label`, `family`, `condition` (the served lead), `difference`,
        `lower_95`, and `upper_95`.

    Raises:
        ValueError: If the recomputed row does not match the report's to the printed digit.
    """
    first, last = LEAD_TABLE_HOURS
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    daytime = (
        pl.scan_parquet(RESULTS_DIR / "losses.parquet")
        .filter(pl.col("arm").is_in(["icon_d2_global", "icon_eu_global"]))
        .collect()
        .filter(hour.is_between(first, last))
        .with_columns(lead_3h=_served_lead(product="icon_eu"))
    )
    reproduced = _contrast_line(
        losses=daytime.filter(pl.col("lead_3h") == 1),
        treatment="icon_d2_global",
        reference="icon_eu_global",
        label=f"both at lead 1 h, {first:02d}–{last:02d} UTC",
    )
    if reproduced not in report_text.splitlines():
        msg = f"the recomputed row does not match the report: {reproduced}"
        raise ValueError(msg)
    records = []
    for label_hour in range(9, 17):
        interval = bootstrap_difference(
            losses=daytime.filter(hour == label_hour),
            treatment="icon_d2_global",
            reference="icon_eu_global",
            metric=METRIC,
        )
        records.append(
            {
                "label": f"{label_hour:02d} UTC",
                "family": "weather model",
                "condition": f"{(label_hour - 1) % 3 + 1} h",
                **{
                    key: interval[key] * PERCENTAGE_POINTS
                    for key in ("difference", "lower_95", "upper_95")
                },
            }
        )
    return pl.DataFrame(records)


def _icon_d2_leads(*, contrasts: pl.DataFrame, report_text: str) -> alt.VConcatChart:
    """Draw ICON-D2 − ICON-EU hour by hour, beside the whole record and the matched-lead rows.

    Args:
        contrasts: Every contrast row in the report.
        report_text: The report.

    Returns:
        Figure 3.
    """
    domain = (-2.0, 0.5)
    hourly = interval_panel(
        rows=_hourly_rows(report_text=report_text),
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="same as ICON-EU",
        better_label="ICON-D2 better",
        conditions=("1 h", "2 h", "3 h"),
        condition_title="Served lead of both",
        panel_title="Hour by hour (UTC hour ending)",
        width=300,
    )
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
    summary = interval_panel(
        rows=_rows(
            contrasts=pl.concat([whole, matched]),
            labels=[
                "All hours" + NAMED_SUFFIX,
                *(f"Both at lead {lead} h, 07–19 UTC" for lead in (1, 2, 3)),
            ],
        ),
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="same as ICON-EU",
        better_label="ICON-D2 better",
        panel_title="Whole record, and by served lead",
        width=300,
    )
    return figure(
        panels=[hourly, summary],
        number=3,
        title="ICON-D2's advantage over ICON-EU fades within hours of each run",
        subtitle=[
            f"ICON-D2's mean absolute error minus ICON-EU's, in points of capacity. {CAPACITY}",
            (
                "Both models run every 3 hours, so at each hour both are served at the same lead. "
                "Post hoc: the hourly and by-lead rows were added after the first run."
            ),
            f"{DOTS} {SCOPE}",
        ],
        width=WIDTH,
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
    shared = {
        "x_domain": (-0.6, 0.8),
        "zero_label": "same as ICON-EU",
        "better_label": "ICON-EU better",
        "better_direction": "positive",
        "width": 420,
    }
    panels = [
        interval_panel(
            rows=_rows(
                contrasts=icon_global,
                labels=[
                    "All hours" + NAMED_SUFFIX,
                    "Hours its lead equals ICON-EU's, 07–19 UTC",
                    "Hours it is served 4 to 6 h ahead, 07–19 UTC",
                ],
            ),
            x_title="",
            panel_title="ICON global − ICON-EU",
            **shared,  # ty: ignore[invalid-argument-type]
        ),
        interval_panel(
            rows=_rows(
                contrasts=ukv,
                labels=[
                    "UKV as served" + NAMED_SUFFIX,
                    "UKV's two snapshots given as a pair",
                    "UKV rebuilt as the mean of its two snapshots",
                    "Rebuilt UKV with neighbouring hours, against ICON-EU with them",
                ],
            ),
            x_title="Rival's mean absolute error minus ICON-EU's (points of capacity)",
            panel_title="UKV − ICON-EU",
            reference_labels=False,
            **shared,  # ty: ignore[invalid-argument-type]
        ),
    ]
    return figure(
        panels=panels,
        number=4,
        title="ICON-EU beats ICON global and UKV as served, but not UKV rebuilt from its snapshots",
        subtitle=[
            f"Each rival's mean absolute error minus ICON-EU's, in points of capacity. {CAPACITY}",
            (
                "UKV rows other than UKV as served, and the ICON global rows split by lead, are "
                "post hoc."
            ),
            f"{DOTS} {SCOPE}",
        ],
        width=760,
        direction="vertical",
    )


def _ukv_against_era5(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV − ERA5 across the record, since the live downloader, and after the upgrade.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 5.
    """
    rows = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_AGAINST_ERA5, "all", "ukv_global", "era5_global"),
            ContrastKey(SECTION_AGAINST_ERA5, "ukv_live", "ukv_global", "era5_global"),
            ContrastKey(SECTION_AGAINST_ERA5, "post", "ukv_global", "era5_global"),
            ContrastKey(SECTION_POST_ONLY, "post", "ukv_global", "era5_global"),
        ],
    )
    panel = interval_panel(
        rows=_rows(
            contrasts=rows,
            labels=[
                "All hours, December 2022 to September 2026",
                "Since Open-Meteo's own UKV downloader started, August 2024",
                "After the January 2026 upgrade, one model for both eras",
                "After the upgrade, models fitted on those 8 months alone",
            ],
        ),
        x_domain=(-0.6, 0.6),
        x_title="UKV's mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="UKV better",
        width=360,
    )
    return figure(
        panels=[panel],
        number=5,
        title="UKV against ERA5 is unresolved",
        subtitle=[
            f"UKV's mean absolute error minus ERA5's, in points of capacity. {CAPACITY}",
            f"{DOTS} Exploratory.",
            "The two post-upgrade rows rest on 8 monthly clusters, so their intervals under-cover.",
            SCOPE,
        ],
        width=760,
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
    panel = interval_panel(
        rows=_rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(SECTION_SPLIT, "all", f"{p}_split", f"{p}_erbs") for p in order
                ],
            ),
            labels=[NAMES[product] for product in order],
        ),
        x_domain=(-0.2, 0.1),
        x_title="Mean absolute error with its own beam minus with the Erbs split "
        "(points of capacity)",
        zero_label="no gain",
        better_label="own beam better",
        width=460,
    )
    return figure(
        panels=[panel],
        number=6,
        title="Every product except ERA5 gains 0.03 to 0.11 points from its own direct beam",
        subtitle=[
            (
                "Each product shown its own published beam and diffuse, minus the same product "
                "with the Erbs split of its own global irradiance."
            ),
            (
                f"Points of capacity. {CAPACITY} For scale, CAMS leads ERA5 by 3.92 points across "
                "the whole record (Figure 1)."
            ),
            f"{DOTS} Exploratory. {SCOPE}",
        ],
        width=760,
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
    frames = [
        select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(section, "all", t, r) for t, r in DECIDING],
        ).with_columns(condition=pl.lit(condition))
        for section, condition in zip((SECTION_DECIDING, SECTION_TRANSFER), conditions, strict=True)
    ]
    labels = [_contrast_name(treatment=t, reference=r) + NAMED_SUFFIX for t, r in DECIDING]
    rows = pl.concat([_rows(contrasts=frame, labels=labels) for frame in frames]).sort(
        pl.col("label").replace_strict({label: i for i, label in enumerate(labels)})
    )
    panel = interval_panel(
        rows=rows,
        x_domain=(-3.0, 1.0),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        conditions=conditions,
        condition_title="Model",
        width=380,
    )
    return figure(
        panels=[panel],
        number=7,
        title="The ranking holds for a generator trained on its neighbours",
        subtitle=[
            f"The four contrasts named before the run, in points of capacity. {CAPACITY}",
            (
                "Hollow: a model trained on the other five generators, with the scored months "
                "withheld everywhere."
            ),
            f"{DOTS} {SCOPE}",
        ],
        width=820,
    )


def _implied_capacity_rows(*, report_text: str) -> pl.DataFrame:
    """Return each product's seasonal term at each calendar month, after reproducing the report.

    Args:
        report_text: The report, whose implied-capacity table the rebuilt frame must reproduce.

    Returns:
        One row per (product, calendar month) with `percent`: the exponential, minus one, of the
        mean over that month's site-months of the site's calendar-month mean log implied
        capacity minus the site's overall mean, as a percentage.

    Raises:
        ValueError: If the rebuilt frame's row count or implied-capacity table differs from the
            report's.
    """
    frame = with_export_cap(
        dataset=_with_eras(frame=_add_time_features(dataset=_common_rows(frame=_joined())))
    )
    if frame.height != COMMON_ROWS:
        msg = f"the rebuilt frame holds {frame.height} rows, not {COMMON_ROWS}"
        raise ValueError(msg)
    log_by_product = _log_capacity_by_month(frame=frame)
    table = _implied_capacity(log_by_product=log_by_product)
    if "\n".join(table) not in report_text:
        msg = "the rebuilt implied-capacity table differs from the report's:\n" + "\n".join(table)
        raise ValueError(msg)
    return pl.concat(
        log_capacity.group_by("calendar")
        .agg(pl.col("seasonal").mean())
        .select(
            product=pl.lit(product),
            month=pl.col("calendar").cast(pl.Int32),
            percent=pl.col("seasonal").map_batches(np.expm1) * PERCENTAGE_POINTS,
        )
        for product, log_capacity in log_by_product.items()
    )


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
                width=220,
                height=190,
                title=alt.TitleParams(NAMES[product], anchor="start", frame="group", fontSize=14),
            )
        )
    grid = [alt.hconcat(*panels[i : i + 3], spacing=24) for i in (0, 3)]
    return figure(
        panels=grid,
        number=8,
        title="CAMS's implied capacity swings the most with the seasons",
        subtitle=[
            (
                "Implied capacity: metered output over what a fixed south-facing panel at 30° tilt "
                "predicts per megawatt from the product."
            ),
            (
                "Each point: the mean over generators and years of that calendar month's log "
                "implied capacity minus the generator's own mean,"
            ),
            (
                "shown as a percentage. Zero is the annual mean. Unconstrained hours with the sun "
                "above 10°. No interval is drawn."
            ),
            (
                "With the seasonal cycle removed, CAMS's month-to-month spread is the smallest: "
                "6.5%, against 7.9% to 10.0%."
            ),
            SCOPE,
        ],
        width=800,
        direction="vertical",
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
