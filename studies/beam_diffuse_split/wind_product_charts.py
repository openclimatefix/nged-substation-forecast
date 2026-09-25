"""Draw the eight anonymised charts for the write-up on which weather product best describes wind.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.

**Every number a chart shares with the page is read from the report `wind_products.py` wrote**,
so a chart cannot disagree with the page. The step chart's fortnightly wind-speed ratios are read
from the downloads `fetch_wind_point.py` wrote, and its period means from the report. Two charts
also draw numbers the report does not print, computed from `losses.parquet` without refitting any
model: the two "models work" charts' out-of-fold predictions and per-generator errors.
The ERA5-by-year chart reads the table `wind_products.py --era5-by-year` wrote.

Generators appear only as `W1` to `W3`. Only the "models work" time series plots output, as a
percentage of capacity on days 1 to 7 of a week, with no calendar date. The ratio of two products'
wind speeds at the generator with the steps carries no generator label, and the per-generator error
chart leaves ICON global out, because the steps are a fingerprint a public ICON archive could match
to a grid cell.

Run it with `uv run python studies/beam_diffuse_split/wind_product_charts.py`, after
`wind_products.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from build_dataset import _wind_sites
from fetch_wind_point import output_path_for
from figure_numbers import WIND_FIGURE_NUMBERS
from sources import STUDY_DATA_DIR
from studies.charts import (
    FAMILY_COLOURS,
    PLOT_WIDTH_PX,
    ContrastKey,
    figure,
    interval_panel,
    planning,
    report_contrasts,
    report_errors,
    select_contrasts,
)
from weather_product_charts import (
    ASSETS_DIR,
    CAPACITY,
    DOTS,
    FAMILIES,
    NAMES,
    X_TITLE,
    _contrast_name,
    _models_work_error_chart,
    _models_work_long_frame,
    _models_work_timeseries,
    _pick_weeks,
    _reconstruct_predicted,
    _rows,
    _two_places,
    era5_by_year_rows,
)
from weather_products import _contrast_line
from wind_products import (
    ERA5_BY_YEAR_DIR,
    OUTPUT_DIR_NAME,
    STEP_DATES,
    STEP_SITE,
    _renamed,
    _scoped,
    common_rows,
    joined,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = STUDY_DATA_DIR / OUTPUT_DIR_NAME
"""Where `wind_products.py` wrote its report and losses."""

SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous wind generator labels."""

SECTION_DECIDING: Final[str] = "Deciding contrasts, named before the run"
SECTION_ERA: Final[str] = "By era and by half of the year (exploratory)"
SECTION_SENSITIVITY: Final[str] = (
    "Sensitivity: the served 100 m arm the plan specified, and the second setting"
)
SECTION_CHECKS: Final[str] = "Checks added after the first run (exploratory)"
SECTION_OTHER: Final[str] = "Other contrasts (exploratory)"

DECIDING: Final[tuple[tuple[str, str], ...]] = (
    ("icon_eu_wind", "era5_wind"),
    ("ukv_wind", "era5_wind"),
    ("icon_eu_wind", "ukv_wind"),
    ("icon_d2_wind", "icon_eu_wind"),
)
"""The four planned contrasts, as the report writes them."""

SCOPE: Final[str] = "Three wind farms in Lincolnshire, August 2024 to September 2026."
HALVES: Final[tuple[str, str]] = ("April to September", "October to March")


def _wind_losses() -> pl.DataFrame:
    """Return the pooled setting's losses, every arm.

    Both settings share arm names, so a join across them would multiply rows.

    Returns:
        The losses whose `setting` is `pooled`.
    """
    return pl.read_parquet(RESULTS_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")


def _reproduce(*, pooled: pl.DataFrame, report_text: str) -> None:
    """Recompute two printed rows through the report's own code path, and check both match.

    Args:
        pooled: The pooled setting's losses.
        report_text: The report.

    Raises:
        ValueError: If either recomputed row differs from the report's.
    """
    wind = _renamed(losses=pooled, suffix="_wind")
    lines = report_text.splitlines()
    for line in (
        _contrast_line(
            losses=_scoped(losses=wind, scope="summer"),
            treatment="ukv_wind",
            reference="era5_wind",
            label="summer",
        ),
        _contrast_line(
            losses=wind.filter(pl.col("site") == "W1"),
            treatment="icon_eu_wind",
            reference="era5_wind",
            label="site W1",
        ),
    ):
        if line not in lines:
            msg = f"the recomputed row does not match the report: {line}"
            raise ValueError(msg)


AFTER_FIRST_RUN_SUFFIX: Final[str] = " (planned; 80 m chosen after the first run)"
"""Ends the label of a named contrast whose ICON arm was switched to 80 m after the first run.

A row carrying this suffix is not marked `planned`, so `NAMED_SUFFIX` does not also bold its label
in a mixed figure: only a row whose arm is exactly as the plan specified gets that.
"""


def _changed_after_first_run(*arms: str) -> bool:
    """Say whether a named contrast has an ICON arm, each switched to 80 m after the first run."""
    return any(arm.startswith("icon") for arm in arms)


def _deciding_label(*, treatment: str, reference: str) -> str:
    """Label a named contrast, marking it as changed where either arm is an ICON product."""
    suffix = AFTER_FIRST_RUN_SUFFIX if _changed_after_first_run(treatment, reference) else ""
    return _contrast_name(treatment=treatment, reference=reference) + suffix


def _headline(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product against ERA5 above the four planned contrasts.

    Args:
        contrasts: Every contrast row in the report.
        errors: Each product's mean absolute error.

    Returns:
        The `contrasts` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    order = sorted(errors, key=errors.__getitem__)
    named = {treatment for treatment, reference in DECIDING if reference == "era5_wind"}
    keys = {
        f"{p}_wind": ContrastKey(
            SECTION_DECIDING if f"{p}_wind" in named else SECTION_OTHER,
            "all",
            f"{p}_wind",
            "era5_wind",
        )
        for p in order
        if p != "era5"
    }
    against_era5 = select_contrasts(contrasts=contrasts, wanted=list(keys.values())).select(
        "treatment", "difference", "lower_95", "upper_95"
    )
    era5 = pl.DataFrame(
        {"treatment": ["era5_wind"], "difference": [0.0], "lower_95": [0.0], "upper_95": [0.0]}
    )
    by_product = pl.concat([against_era5, era5]).sort(
        pl.col("treatment").replace_strict({f"{p}_wind": i for i, p in enumerate(order)})
    )
    labels = [
        f"{NAMES[p]} · {_two_places(errors[p])}%"
        + (AFTER_FIRST_RUN_SUFFIX if f"{p}_wind" in named and _changed_after_first_run(p) else "")
        for p in order
    ]
    left_rows = _rows(
        contrasts=by_product,
        labels=labels,
        planned=[f"{p}_wind" in named and not _changed_after_first_run(p) for p in order],
    )
    right_rows = _rows(
        contrasts=select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(SECTION_DECIDING, "all", t, r) for t, r in DECIDING],
        ),
        labels=[_deciding_label(treatment=t, reference=r) for t, r in DECIDING],
        planned=[not _changed_after_first_run(t, r) for t, r in DECIDING],
    )
    figure_planning = planning(rows=[left_rows, right_rows])
    domain = (-1.0, 1.0)
    left = interval_panel(
        rows=left_rows,
        x_domain=domain,
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        panel_title="Every product against ERA5",
        figure_planning=figure_planning,
    )
    right = interval_panel(
        rows=right_rows,
        x_domain=(-1.0, 0.6),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        panel_title="The four planned contrasts",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[left, right],
        number=WIND_FIGURE_NUMBERS["contrasts"],
        figure_planning=figure_planning,
        title=(
            "UKV, ICON-D2, and ICON-EU each beat ERA5 by a margin statistically significant at "
            "the 5% level"
        ),
        subtitle=[
            (
                "Top: each product against ERA5; each label gives the product's own error. The "
                "ICON-D2 and ICON global rows are exploratory. Bottom: the four planned contrasts, "
                "planned with the served 100 m wind; each ICON product was switched "
                "to its 80 m wind after the first run."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _half_years(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV, ICON-D2 and ICON-EU against ERA5 in each half of the year.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The `half_years` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    products = ("ukv", "icon_d2", "icon_eu")
    frames = [
        _rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(SECTION_ERA, scope, f"{p}_wind", "era5_wind") for p in products
                ],
            ),
            labels=[_contrast_name(treatment=f"{p}_wind", reference="era5_wind") for p in products],
        ).with_columns(condition=pl.lit(half))
        for scope, half in zip(("summer", "winter"), HALVES, strict=True)
    ]
    rows = pl.concat(frames).sort(
        pl.col("treatment").replace_strict({f"{p}_wind": i for i, p in enumerate(products)}),
        maintain_order=True,
    )
    figure_planning = planning(rows=[rows])
    panel = interval_panel(
        rows=rows,
        x_domain=(-1.0, 0.6),
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        conditions=HALVES,
        condition_title="Months",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[panel],
        number=WIND_FIGURE_NUMBERS["half_years"],
        figure_planning=figure_planning,
        title="UKV's and ICON-D2's advantage over ERA5 is larger from April to September",
        subtitle=[DOTS, f"{CAPACITY} {SCOPE}"],
    )


def _icon_d2_against_ukv(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw ICON-D2 − UKV across the window, by era and half-year, by setting, and by lead.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The `icon_d2_leads` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    groups = {
        "Whole window, and either side of the January 2026 UKV upgrade": [
            (SECTION_DECIDING, "all", "All hours"),
            (SECTION_ERA, "pre", "Before the upgrade"),
            (SECTION_ERA, "post", "After the upgrade (8 months)"),
        ],
        "Half of the year": [
            (SECTION_ERA, "summer", HALVES[0]),
            (SECTION_ERA, "winter", HALVES[1]),
        ],
        "Other settings": [
            (SECTION_SENSITIVITY, "served 100 m and 10 m", "Served 100 m wind and 10 m speed"),
            (SECTION_SENSITIVITY, "second setting", "Second hyperparameter setting"),
            (SECTION_CHECKS, "UKV at 80 m", "XGBoost model given UKV's 80 m wind"),
        ],
        "Hours into ICON-D2's run (UKV is always at T+0)": [
            (SECTION_CHECKS, f"ICON lead {lead} h", f"{lead} h into the run") for lead in (0, 1, 2)
        ],
    }
    group_rows = {
        name: _rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(section, scope, "icon_d2_wind", "ukv_wind")
                    for section, scope, _ in rows
                ],
            ),
            labels=[label for _, _, label in rows],
        )
        for name, rows in groups.items()
    }
    figure_planning = planning(rows=list(group_rows.values()))
    panels = [
        interval_panel(
            rows=rows,
            x_domain=(-0.6, 0.4),
            x_title=X_TITLE if index == len(groups) - 1 else "",
            zero_label="same as UKV",
            better_label="ICON-D2 better",
            panel_title=name,
            reference_labels=index == 0,
            figure_planning=figure_planning,
        )
        for index, (name, rows) in enumerate(group_rows.items())
    ]
    return figure(
        panels=panels,
        number=WIND_FIGURE_NUMBERS["icon_d2_leads"],
        figure_planning=figure_planning,
        title="ICON-D2 leads UKV across the window, but not since UKV's upgrade",
        subtitle=[
            (
                "ICON-D2's mean absolute error minus UKV's. The UKV-at-80-m and "
                "hours-into-the-run rows were added after the first run."
            ),
            "The post-upgrade row rests on 8 months, so its interval is likely too narrow.",
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _fortnightly_ratios() -> pl.DataFrame:
    """Return ICON global's wind speed over ICON-EU's, as a ratio of fortnightly means.

    Returns:
        One row per (fortnight, height, group) with `ratio`, where `group` is the generator with
        the steps or the other two generators pooled.
    """
    heights = (10, 80)
    speeds = [f"wind_speed_{height}m" for height in heights]
    joined = (
        pl.read_parquet(output_path_for(product="icon_global"))
        .select("site", "time", *speeds)
        .join(
            pl.read_parquet(output_path_for(product="icon_eu")).select("site", "time", *speeds),
            on=["site", "time"],
            suffix="_eu",
        )
        .with_columns(
            group=pl.when(pl.col("site") == STEP_SITE)
            .then(pl.lit("The generator with the steps"))
            .otherwise(pl.lit("The other two generators")),
            period=sum(pl.col("time") >= date for date in STEP_DATES),
        )
    )
    return pl.concat(
        joined.group_by(pl.col("time").dt.truncate("2w").alias("fortnight"), "group")
        .agg(
            ratio=pl.col(f"wind_speed_{height}m").mean() / pl.col(f"wind_speed_{height}m_eu").mean()
        )
        .with_columns(height=pl.lit(f"{height} m"))
        for height in heights
    ).sort("height", "group", "fortnight")


def _step_ratios(*, report_text: str) -> dict[int, tuple[float, ...]]:
    """Read the step generator's period-mean speed ratios from the report, per height.

    Args:
        report_text: The report.

    Returns:
        For 10 m and 80 m, ICON global's mean speed over ICON-EU's before, between and after the
        steps.
    """
    ratios: dict[int, tuple[float, ...]] = {}
    for line in report_text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] == STEP_SITE and cells[1].endswith(" m"):
            ratios[int(cells[1].removesuffix(" m"))] = tuple(float(cell) for cell in cells[2:])
    return ratios


def _ratio_panel(
    *, ratios: pl.DataFrame, height: str, step_ratios: dict[int, tuple[float, ...]]
) -> alt.LayerChart:
    """Draw one height's fortnightly ratio, the step dates, and each period's mean at the step site.

    Args:
        ratios: The output of `_fortnightly_ratios`.
        height: `10 m` or `80 m`.
        step_ratios: The output of `_step_ratios`.

    Returns:
        One panel.
    """
    groups = ["The generator with the steps", "The other two generators"]
    data = ratios.filter(pl.col("height") == height)
    x = alt.X(
        "fortnight:T",
        title=None,
        axis=alt.Axis(format="%b %Y", tickCount=alt.TimeIntervalStep(interval="month", step=6)),
    )
    y = alt.Y(
        "ratio:Q",
        title="ICON global over ICON-EU",
        scale=alt.Scale(domain=[0.8, 1.3], nice=False, zero=False),
    )
    # Stroke rather than colour, so this scale stays apart from the family colour scale the
    # interval panel below it uses.
    colour = alt.Stroke(
        "group:N",
        scale=alt.Scale(domain=groups, range=[ocf.DATA_BLUE, ocf.ENSEMBLE_LINE]),
        legend=None,
    )
    line = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, stroke=colour)  # ty: ignore[unresolved-attribute]
    steps = (
        alt.Chart(pl.DataFrame({"date": list(STEP_DATES)}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(x="date:T")  # ty: ignore[unresolved-attribute]
    )
    # The second step falls near the right edge, so its label sits to the left of its rule.
    step_labels = [
        alt.Chart(pl.DataFrame({"date": [date], "text": [text]}))
        .mark_text(align=align, baseline="top", dx=dx, dy=2, color=ocf.BLACK_1)
        .encode(x="date:T", y=alt.value(0), text="text:N")  # ty: ignore[unresolved-attribute]
        for date, text, align, dx in zip(
            STEP_DATES, ("2 June 2025", "2 June 2026"), ("left", "right"), (4, -4), strict=True
        )
    ]
    edges = [data["fortnight"].min(), *STEP_DATES, data["fortnight"].max()]
    periods = pl.DataFrame(
        {
            "start": edges[:3],
            "end": edges[1:],
            "ratio": list(step_ratios[int(height.removesuffix(" m"))]),
        }
    )
    period_means = (
        alt.Chart(periods)
        .mark_rule(color=ocf.DATA_BLUE, strokeWidth=3, opacity=0.5)
        .encode(x="start:T", x2="end:T", y="ratio:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[steps, *step_labels, period_means, line],
        width=PLOT_WIDTH_PX,
        height=160,
        title=alt.TitleParams(
            f"Wind speed at {height}", anchor="start", frame="group", fontSize=14
        ),
    )


def _steps(*, contrasts: pl.DataFrame, report_text: str) -> alt.VConcatChart:
    """Draw the steps in ICON global's served wind above how much they add to its error.

    Args:
        contrasts: Every contrast row in the report.
        report_text: The report, for the step generator's period-mean ratios.

    Returns:
        The `icon_global_steps` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    conditions = ("Not told", "Told when the steps fall")
    at_step_site = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_CHECKS, f"site {STEP_SITE}", "icon_global_wind", "icon_eu_wind"),
            ContrastKey(
                SECTION_CHECKS,
                f"told the step period, site {STEP_SITE}",
                "icon_global_step",
                "icon_eu_step",
            ),
        ],
    ).select("difference", "lower_95", "upper_95")
    others = " and ".join(site for site in SITES if site != STEP_SITE)
    elsewhere = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_CHECKS, f"sites {others}", "icon_global_wind", "icon_eu_wind"),
            ContrastKey(
                SECTION_CHECKS,
                f"told the step period, sites {others}",
                "icon_global_step",
                "icon_eu_step",
            ),
        ],
    ).select("difference", "lower_95", "upper_95")
    all_sites = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(SECTION_OTHER, "all", "icon_global_wind", "icon_eu_wind"),
            ContrastKey(
                SECTION_CHECKS,
                "told the step period, all sites",
                "icon_global_step",
                "icon_eu_step",
            ),
        ],
    ).select("difference", "lower_95", "upper_95")
    rows = pl.concat([at_step_site, elsewhere, all_sites]).with_columns(
        label=pl.Series(
            ["The generator with the steps"] * 2
            + ["The other two generators, pooled"] * 2
            + ["All three generators"] * 2
        ),
        family=pl.lit("weather model"),
        condition=pl.Series([*conditions, *conditions, *conditions]),
    )
    ratios = _fortnightly_ratios()
    step_ratios = _step_ratios(report_text=report_text)
    figure_planning = planning(rows=[rows])
    right = interval_panel(
        rows=rows,
        x_domain=(-0.5, 2.5),
        x_title="ICON global's mean absolute error minus ICON-EU's (points of capacity)",
        zero_label="same as ICON-EU",
        better_label="ICON-EU better",
        better_direction="positive",
        conditions=conditions,
        condition_title="The XGBoost model is",
        panel_title="How much the steps add to ICON global's error",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[
            *(
                _ratio_panel(ratios=ratios, height=height, step_ratios=step_ratios)
                for height in ("10 m", "80 m")
            ),
            right,
        ],
        number=WIND_FIGURE_NUMBERS["icon_global_steps"],
        figure_planning=figure_planning,
        title="About half of ICON global's gap to ICON-EU is a pair of steps in its served wind at "
        "one generator",
        subtitle=[
            (
                "Top two panels: ICON global's wind speed over ICON-EU's, fortnightly, in blue at "
                "the generator with the steps and grey at the other two; 1 means the two agree."
            ),
            ("Pale bars: the mean of each period at the generator with the steps."),
            "Bottom: error difference, added after the first run.",
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _per_generator(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV, ICON-D2 and ICON-EU against ERA5 at each generator.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        The `per_generator` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    products = ("ukv", "icon_d2", "icon_eu")
    product_rows = [
        _rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(
                        SECTION_OTHER if product == "icon_d2" else SECTION_DECIDING,
                        f"site {site}",
                        f"{product}_wind",
                        "era5_wind",
                    )
                    for site in SITES
                ],
            ),
            labels=[f"Generator {site}" for site in SITES],
        )
        for product in products
    ]
    figure_planning = planning(rows=product_rows)
    panels = [
        interval_panel(
            rows=rows,
            x_domain=(-1.5, 0.5),
            x_title=X_TITLE if index == len(products) - 1 else "",
            zero_label="same as ERA5",
            better_label="better than ERA5",
            panel_title=_contrast_name(treatment=f"{product}_wind", reference="era5_wind"),
            reference_labels=index == 0,
            figure_planning=figure_planning,
        )
        for index, (product, rows) in enumerate(zip(products, product_rows, strict=True))
    ]
    return figure(
        panels=panels,
        number=WIND_FIGURE_NUMBERS["per_generator"],
        figure_planning=figure_planning,
        title=(
            "UKV's advantage over ERA5 is statistically significant at the 5% level at two of the "
            "three generators"
        ),
        subtitle=[
            f"{DOTS} The ICON-D2 rows are computed for this chart.",
            f"{CAPACITY} {SCOPE}",
        ],
    )


def _era5_by_year() -> alt.VConcatChart:
    """Draw each product's error minus ERA5's, on the same months of 2025 and of 2026.

    Reads `wind_products.py --era5-by-year`'s table, which restricts every year to January to
    September so a partial 2026 compares against the same months of the complete years before it.
    August to December 2024, five months of a partial year even under that restriction, is too few
    for an interval and is left out.

    Returns:
        The `era5_by_year` figure of `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    by_year = pl.read_parquet(ERA5_BY_YEAR_DIR / "era5_by_year.parquet")
    rows = era5_by_year_rows(
        by_year=by_year, products=("icon_d2", "ukv", "icon_eu", "icon_global"), suffix="_wind"
    )
    years = tuple(sorted(rows["condition"].unique().to_list()))
    figure_planning = planning(rows=[rows])
    panel = interval_panel(
        rows=rows,
        x_domain=(-1.2, 1.0),
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        conditions=years,
        condition_title="Calendar year",
        figure_planning=figure_planning,
    )
    return figure(
        panels=[panel],
        number=WIND_FIGURE_NUMBERS["era5_by_year"],
        figure_planning=figure_planning,
        title=(
            "On January to September of each year, UKV's lead over ERA5 grew in 2026; ICON-EU's "
            "and ICON-D2's did not"
        ),
        subtitle=[
            (
                "Each product's mean absolute error minus ERA5's, on January to September of one "
                "calendar year. August to December 2024, five months, is too short for an "
                "interval and is left out."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


MODELS_WORK_PRODUCTS: Final[tuple[str, str]] = ("icon_d2", "era5")
"""The products whose XGBoost models the "models work" time series draws: ICON-D2, which has the
lowest pooled mean absolute error in the report, and ERA5, the reference every product is measured
against.
"""

MODELS_WORK_MIN_HOURS: Final[int] = 20
"""A (generator, day) needs at least this many hours to count towards week selection."""

WIND_WEEK_CRITERIA: Final[tuple[tuple[str, str, bool], ...]] = (
    ("Windiest week", "mean_output", True),
    ("Calmest week", "mean_output", False),
    ("Most variable week", "spread", True),
)
"""Wind weeks are picked in this order, each excluding the weeks already picked."""

WIND_WEEK_DISPLAY_ORDER: Final[tuple[str, ...]] = (
    "Windiest week",
    "Most variable week",
    "Calmest week",
)
"""Wind weeks are drawn easiest to hardest, left to right."""


def _models_work_frame() -> pl.DataFrame:
    """Rebuild the exact rows and measured power `wind_products.py` scored, without refitting.

    Calls the same row-building functions `wind_products.py`'s own `main` calls, so the rows match
    the study exactly; nothing here fits a model.

    Returns:
        One row per (site, time) the pooled run scored, carrying `power_mw` and
        `effective_capacity_mw`.
    """
    return common_rows(frame=joined(sites=_wind_sites())).select(
        "site", "time", "power_mw", "effective_capacity_mw"
    )


def _wind_models_work(
    *, losses: pl.DataFrame, errors: dict[str, float]
) -> tuple[alt.VConcatChart, alt.VConcatChart]:
    """Draw the wind "models work" figures: the out-of-fold time series, and error per generator.

    Args:
        losses: The pooled setting's losses, every arm.
        errors: Each product's pooled mean absolute error.

    Returns:
        The `models_work_timeseries` and `models_work_error` figures of
        `figure_numbers.WIND_FIGURE_NUMBERS`.
    """
    measured = _models_work_frame()
    order = ("Measured", *(f"XGBoost model given {NAMES[p]}" for p in MODELS_WORK_PRODUCTS))
    predicted = tuple(
        (_reconstruct_predicted(losses=losses, measured=measured, arm=f"{product}_wind"), label)
        for product, label in zip(MODELS_WORK_PRODUCTS, order[1:], strict=True)
    )
    hourly = measured.with_columns(
        output_frac=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw")
    )
    weeks = _pick_weeks(
        hourly=hourly, min_hours=MODELS_WORK_MIN_HOURS, agg="mean", criteria=WIND_WEEK_CRITERIA
    )
    long_frame = (
        _models_work_long_frame(measured=measured, predicted=predicted)
        .with_columns(week=pl.col("time").dt.truncate("1w"))
        .join(weeks, on="week", how="inner")
    )
    timeseries = _models_work_timeseries(
        long_frame=long_frame,
        sites=SITES,
        week_order=WIND_WEEK_DISPLAY_ORDER,
        order=order,
        colours=(ocf.TEXT, *(FAMILY_COLOURS[FAMILIES[p]] for p in MODELS_WORK_PRODUCTS)),
        number=WIND_FIGURE_NUMBERS["models_work_timeseries"],
        title=(
            "An XGBoost model given ICON-D2 follows measured power at every generator, across a "
            "windy, a variable, and a calm week"
        ),
        subtitle=[
            "Out-of-fold power as a percentage of the generator's own capacity.",
            (
                "Weeks are picked from measured power alone, pooled over the three generators: "
                "the windiest has the highest mean output, the calmest the lowest, and the most "
                "variable the largest swing in daily mean output."
            ),
            CAPACITY,
            SCOPE,
        ],
    )
    error = _models_work_error_chart(
        losses=losses,
        arm_suffix="_wind",
        sites=SITES,
        names=NAMES,
        errors={product: error for product, error in errors.items() if product != "icon_global"},
        x_domain=(5.0, 9.0),
        number=WIND_FIGURE_NUMBERS["models_work_error"],
        title=(
            "ICON-D2, UKV, ICON-EU, and ERA5 rank in the same order at each of the three generators"
        ),
        subtitle=[
            (
                "Each dot is one generator's mean absolute error given one product. ICON global "
                "is left out, because its served wind steps at one generator "
                f"(Figure {WIND_FIGURE_NUMBERS['icon_global_steps']})."
            ),
            CAPACITY,
            SCOPE,
        ],
    )
    return timeseries, error


def main() -> int:
    """Read the report, compute the new numbers, and write the eight SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = RESULTS_DIR / "report.md"
    report_text = report_path.read_text()
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="All sites")
    pooled = _wind_losses()
    _reproduce(pooled=pooled, report_text=report_text)
    models_work_timeseries, models_work_error = _wind_models_work(losses=pooled, errors=errors)
    charts = {
        "wind_headline": _headline(contrasts=contrasts, errors=errors),
        "wind_models_work_timeseries": models_work_timeseries,
        "wind_models_work_error": models_work_error,
        "wind_half_years": _half_years(contrasts=contrasts),
        "wind_icon_d2_against_ukv": _icon_d2_against_ukv(contrasts=contrasts),
        "wind_icon_global_steps": _steps(contrasts=contrasts, report_text=report_text),
        "wind_per_generator": _per_generator(contrasts=contrasts),
        "wind_era5_by_year": _era5_by_year(),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
