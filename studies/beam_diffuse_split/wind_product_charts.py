"""Draw the five anonymised charts for the write-up on which weather product best describes wind.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/830>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.

**Almost every number is read from the report `wind_products.py` wrote**, so a chart cannot disagree
with the page. Two charts need numbers the report lacks: ICON-D2 against ERA5 at each generator, and
ICON global against ICON-EU pooled over the two generators without the steps in ICON global's
served wind. Both are bootstrapped from `losses.parquet`, reading only the pooled setting, after
two printed rows have been reproduced through the same code path. The step chart's wind-speed
ratios are read from the downloads `fetch_wind_point.py` wrote, and checked against the figures the
page quotes.

Generators appear only as `W1` to `W3`, and no chart plots output. The one per-generator time
series, the ratio of two products' wind speeds at the generator with the steps, carries no
generator label, because the steps are a fingerprint a public ICON archive could match to a grid
cell.

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
from fetch_wind_point import output_path_for
from sources import STUDY_DATA_DIR
from studies.bootstrap import bootstrap_difference
from studies.charts import (
    NAMED_SUFFIX,
    ContrastKey,
    figure,
    interval_panel,
    report_contrasts,
    report_errors,
    select_contrasts,
)
from weather_product_charts import (
    ASSETS_DIR,
    CAPACITY,
    DOTS,
    NAMES,
    X_TITLE,
    _contrast_name,
    _rows,
    _two_places,
)
from weather_products import METRIC, PERCENTAGE_POINTS, _contrast_line
from wind_products import OUTPUT_DIR_NAME, STEP_DATES, _renamed, _scoped

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = STUDY_DATA_DIR / OUTPUT_DIR_NAME
"""Where `wind_products.py` wrote its report and losses."""

SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous wind generator labels."""

STEP_SITE: Final[str] = "W3"
"""The generator at which ICON global's served wind steps, relative to ICON-EU's."""

STEP_RATIOS: Final[dict[int, tuple[float, float, float]]] = {
    10: (1.128, 0.962, 1.111),
    80: (1.028, 0.948, 1.020),
}
"""ICON global's mean wind speed over ICON-EU's at `STEP_SITE`, before, between and after the steps.

The page's account of the steps rests on these, at 10 m and 80 m, so the chart must reproduce them.
"""

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
"""The four contrasts named before the run, as the report writes them."""

SCOPE: Final[str] = "Three wind farms in Lincolnshire, August 2024 to September 2026."
HALVES: Final[tuple[str, str]] = ("April to September", "October to March")
WIDTH: Final[int] = 900


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


def _new_row(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Bootstrap one contrast not in the report, in points of capacity.

    Args:
        losses: Per-row losses for both arms, restricted to the scope wanted.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.

    Returns:
        `difference`, `lower_95`, and `upper_95`.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    return {
        key: interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    }


def _headline(*, contrasts: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product against ERA5 beside the four contrasts named before the run.

    Args:
        contrasts: Every contrast row in the report.
        errors: Each product's mean absolute error.

    Returns:
        Figure 1.
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
        f"{NAMES[p]} · {_two_places(errors[p])}%" + (NAMED_SUFFIX if f"{p}_wind" in named else "")
        for p in order
    ]
    domain = (-1.0, 1.0)
    left = interval_panel(
        rows=_rows(contrasts=by_product, labels=labels),
        x_domain=domain,
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        panel_title="Every product against ERA5",
        width=300,
    )
    right = interval_panel(
        rows=_rows(
            contrasts=select_contrasts(
                contrasts=contrasts,
                wanted=[ContrastKey(SECTION_DECIDING, "all", t, r) for t, r in DECIDING],
            ),
            labels=[_contrast_name(treatment=t, reference=r) + NAMED_SUFFIX for t, r in DECIDING],
        ),
        x_domain=(-1.0, 0.6),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first product better",
        panel_title="The four contrasts named before the run",
        width=420,
    )
    return figure(
        panels=[left, right],
        number=1,
        title="UKV and ICON-D2 describe past wind best",
        subtitle=[
            (
                "Left: each product's mean absolute error minus ERA5's. Each label gives the "
                "product's own error, as a percentage of capacity."
            ),
            (
                "The ICON-D2 and ICON global rows are exploratory. Right: the four contrasts named "
                "before the run."
            ),
            f"{CAPACITY} {DOTS}",
            SCOPE,
            (
                "Intervals on the left are against ERA5. Overlapping intervals do not mean two "
                "products are indistinguishable; the right-hand panel compares them directly."
            ),
        ],
        width=WIDTH,
    )


def _half_years(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV, ICON-D2 and ICON-EU against ERA5 in each half of the year.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 2.
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
    panel = interval_panel(
        rows=rows,
        x_domain=(-1.0, 0.6),
        x_title="Mean absolute error minus ERA5's (points of capacity)",
        zero_label="same as ERA5",
        better_label="better than ERA5",
        conditions=HALVES,
        condition_title="Months",
        width=380,
    )
    return figure(
        panels=[panel],
        number=2,
        title="UKV's and ICON-D2's advantage over ERA5 is larger from April to September",
        subtitle=[
            f"Each product's mean absolute error minus ERA5's, in points of capacity. {CAPACITY}",
            f"{DOTS} Exploratory.",
            SCOPE,
        ],
        width=760,
    )


def _icon_d2_against_ukv(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw ICON-D2 − UKV across the window, by era and half-year, by setting, and by lead.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 4.
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
            (SECTION_CHECKS, "UKV at 80 m", "UKV shown its 80 m wind"),
        ],
        "Hours into ICON-D2's run (UKV is always at T+0)": [
            (SECTION_CHECKS, f"ICON lead {lead} h", f"{lead} h into the run") for lead in (0, 1, 2)
        ],
    }
    panels = [
        interval_panel(
            rows=_rows(
                contrasts=select_contrasts(
                    contrasts=contrasts,
                    wanted=[
                        ContrastKey(section, scope, "icon_d2_wind", "ukv_wind")
                        for section, scope, _ in rows
                    ],
                ),
                labels=[label for _, _, label in rows],
            ),
            x_domain=(-0.6, 0.4),
            x_title=X_TITLE if index == len(groups) - 1 else "",
            zero_label="same as UKV",
            better_label="ICON-D2 better",
            panel_title=name,
            reference_labels=index == 0,
            width=420,
        )
        for index, (name, rows) in enumerate(groups.items())
    ]
    return figure(
        panels=panels,
        number=4,
        title="ICON-D2 leads UKV across the window, but not since UKV's upgrade",
        subtitle=[
            f"ICON-D2's mean absolute error minus UKV's, in points of capacity. {CAPACITY}",
            (
                f"{DOTS} Exploratory. Added after the first run: the UKV-at-80-m row "
                "and the hours-into-the-run rows."
            ),
            "The post-upgrade row rests on 8 months, so its interval is likely too narrow.",
            SCOPE,
        ],
        width=700,
        direction="vertical",
    )


def _fortnightly_ratios() -> pl.DataFrame:
    """Return ICON global's wind speed over ICON-EU's, as a ratio of fortnightly means.

    Returns:
        One row per (fortnight, height, group) with `ratio`, where `group` is the generator with
        the steps or the other two generators pooled.

    Raises:
        ValueError: If the ratio of each period's means at the generator with the steps differs
            from `STEP_RATIOS`.
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
    for height, expected in STEP_RATIOS.items():
        speed = f"wind_speed_{height}m"
        measured = (
            joined.filter(pl.col("site") == STEP_SITE)
            .group_by("period")
            .agg(ratio=pl.col(speed).mean() / pl.col(f"{speed}_eu").mean())
            .sort("period")["ratio"]
            .round(3)
            .to_list()
        )
        if tuple(measured) != expected:
            msg = f"the {height} m period ratios are {measured}, not {expected}"
            raise ValueError(msg)
    return pl.concat(
        joined.group_by(pl.col("time").dt.truncate("2w").alias("fortnight"), "group")
        .agg(
            ratio=pl.col(f"wind_speed_{height}m").mean() / pl.col(f"wind_speed_{height}m_eu").mean()
        )
        .with_columns(height=pl.lit(f"{height} m"))
        for height in heights
    ).sort("height", "group", "fortnight")


def _ratio_panel(*, ratios: pl.DataFrame, height: str) -> alt.LayerChart:
    """Draw one height's fortnightly ratio, the step dates, and each period's mean at the step site.

    Args:
        ratios: The output of `_fortnightly_ratios`.
        height: `10 m` or `80 m`.

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
    # interval panel beside it uses.
    colour = alt.Stroke(
        "group:N",
        scale=alt.Scale(domain=groups, range=[ocf.DATA_BLUE, ocf.ENSEMBLE_LINE]),
        legend=alt.Legend(title="Fortnightly mean speed ratio", labelLimit=400),
    )
    line = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, stroke=colour)  # ty: ignore[unresolved-attribute]
    steps = (
        alt.Chart(pl.DataFrame({"date": list(STEP_DATES)}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(x="date:T")  # ty: ignore[unresolved-attribute]
    )
    step_labels = (
        alt.Chart(pl.DataFrame({"date": list(STEP_DATES), "text": ["2 June 2025", "2 June 2026"]}))
        .mark_text(align="left", baseline="top", dx=4, dy=2, color=ocf.BLACK_1)
        .encode(x="date:T", y=alt.value(0), text="text:N")  # ty: ignore[unresolved-attribute]
    )
    edges = [data["fortnight"].min(), *STEP_DATES, data["fortnight"].max()]
    periods = pl.DataFrame(
        {
            "start": edges[:3],
            "end": edges[1:],
            "ratio": list(STEP_RATIOS[int(height.removesuffix(" m"))]),
        }
    )
    period_means = (
        alt.Chart(periods)
        .mark_rule(color=ocf.DATA_BLUE, strokeWidth=3, opacity=0.5)
        .encode(x="start:T", x2="end:T", y="ratio:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[steps, step_labels, period_means, line],
        width=420,
        height=180,
        title=alt.TitleParams(
            f"Wind speed at {height}", anchor="start", frame="group", fontSize=14
        ),
    )


def _steps(*, contrasts: pl.DataFrame, pooled: pl.DataFrame) -> alt.VConcatChart:
    """Draw the steps in ICON global's served wind beside how much they add to its error.

    Args:
        contrasts: Every contrast row in the report.
        pooled: The pooled setting's losses.

    Returns:
        Figure 5.
    """
    conditions = ("Not told", "Told which side of the steps each hour falls on")
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
    others = pooled.filter(pl.col("site") != STEP_SITE)
    elsewhere = pl.DataFrame(
        [
            _new_row(
                losses=_renamed(losses=others, suffix="_wind"),
                treatment="icon_global_wind",
                reference="icon_eu_wind",
            ),
            _new_row(losses=others, treatment="icon_global_step", reference="icon_eu_step"),
        ]
    )
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
    left = alt.vconcat(
        *(_ratio_panel(ratios=ratios, height=height) for height in ("10 m", "80 m")), spacing=20
    )
    right = interval_panel(
        rows=rows,
        x_domain=(-0.5, 2.5),
        x_title="ICON global's mean absolute error minus ICON-EU's (points of capacity)",
        zero_label="same as ICON-EU",
        better_label="ICON-EU better",
        better_direction="positive",
        conditions=conditions,
        condition_title="The model is",
        panel_title="How much the steps add to ICON global's error",
        width=280,
    )
    return figure(
        panels=[left, right],
        number=5,
        title="About half of ICON global's gap to ICON-EU is a pair of steps in its served wind at "
        "one generator",
        subtitle=[
            (
                "Left: ICON global's wind speed over ICON-EU's, fortnightly, at the generator with "
                "the steps and pooled over the other two."
            ),
            (
                "Pale bars: the ratio of the whole period's means at the generator with the steps, "
                "either side of each step."
            ),
            (
                "Right: ICON global's mean absolute error minus ICON-EU's, in points of capacity. "
                f"{CAPACITY}"
            ),
            (
                f"{DOTS} Added after the first run. The pooled row for the other two generators "
                "is computed for this chart; the rest are in the report."
            ),
            SCOPE,
        ],
        width=WIDTH,
    )


def _per_generator(*, contrasts: pl.DataFrame, pooled: pl.DataFrame) -> alt.VConcatChart:
    """Draw UKV, ICON-D2 and ICON-EU against ERA5 at each generator.

    Args:
        contrasts: Every contrast row in the report.
        pooled: The pooled setting's losses.

    Returns:
        Figure 3.
    """
    wind = _renamed(losses=pooled, suffix="_wind")
    panels = []
    for index, product in enumerate(("ukv", "icon_d2", "icon_eu")):
        treatment = f"{product}_wind"
        if product == "icon_d2":
            numbers = pl.DataFrame(
                [
                    _new_row(
                        losses=wind.filter(pl.col("site") == site),
                        treatment=treatment,
                        reference="era5_wind",
                    )
                    for site in SITES
                ]
            ).with_columns(treatment=pl.lit(treatment))
        else:
            numbers = select_contrasts(
                contrasts=contrasts,
                wanted=[
                    ContrastKey(SECTION_DECIDING, f"site {site}", treatment, "era5_wind")
                    for site in SITES
                ],
            )
        name = _contrast_name(treatment=treatment, reference="era5_wind")
        panels.append(
            interval_panel(
                rows=_rows(contrasts=numbers, labels=[f"Generator {site}" for site in SITES]),
                x_domain=(-1.5, 0.5),
                x_title=X_TITLE if index == 2 else "",
                zero_label="same as ERA5",
                better_label="better than ERA5",
                panel_title=name,
                reference_labels=index == 0,
                width=420,
            )
        )
    return figure(
        panels=panels,
        number=3,
        title="UKV's advantage over ERA5 excludes zero at two of the three generators",
        subtitle=[
            f"Each product's mean absolute error minus ERA5's, in points of capacity. {CAPACITY}",
            (
                f"{DOTS} Exploratory. The ICON-D2 rows are computed for this chart; the rest are "
                "in the report."
            ),
            SCOPE,
        ],
        width=700,
        direction="vertical",
    )


def main() -> int:
    """Read the report, compute the new numbers, and write the five SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = RESULTS_DIR / "report.md"
    report_text = report_path.read_text()
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="All sites")
    pooled = _wind_losses()
    _reproduce(pooled=pooled, report_text=report_text)
    charts = {
        "wind_headline": _headline(contrasts=contrasts, errors=errors),
        "wind_half_years": _half_years(contrasts=contrasts),
        "wind_icon_d2_against_ukv": _icon_d2_against_ukv(contrasts=contrasts),
        "wind_icon_global_steps": _steps(contrasts=contrasts, pooled=pooled),
        "wind_per_generator": _per_generator(contrasts=contrasts, pooled=pooled),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
