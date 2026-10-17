"""Draw experiment 00064's figures from the saved results, as SVG, headline contrasts first.

1. The headline: each arm's absolute quantile score with the reference lines, and the verdict
   contrasts N and T with their 98.75% intervals, the margin, and the gain threshold.
2. Contrast N by lead band for the four arms (exploratory).
3. The slope diagnostic: the oracle slope by lead band for the four `d1l` variants (exploratory).
4. Contrast N by valid month for the four arms, the sign check (exploratory).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00064-*/e00064_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00064_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 80
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, forecast runs one "
    "every 5 days: July to September 2025 and April to June 2026. Quantile score in points of "
    "capacity (1 point is 1% of a substation's capacity)."
)
GREY: Final[str] = "#8A8A8A"
ARM_ORDER: Final[tuple[str, ...]] = ("sf", "a", "am", "b")
ARM_LABELS: Final[dict[str, str]] = {
    "d0": "d0: best guess (reference)",
    "d1l": "d1l: best guess, no radiation",
    "sf": "sf: shift after the fit",
    "a": "a: solar as an input column",
    "am": "am: solar as a starting offset",
    "b": "b: gross demand minus solar",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "d0": GREY,
    "d1l": GREY,
    "sf": ocf.BRAND_ORANGE,
    "a": ocf.DATA_BLUE,
    "am": ocf.DATA_GREEN,
    "b": ocf.DATA_PURPLE,
}
MONTH_NAMES: Final[dict[int, str]] = {
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
}
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
VARIANT_LABELS: Final[dict[str, str]] = {
    "all": "All inputs",
    "no_lags": "Without lagged-power columns",
    "no_cloud_proxies": "Without cloud-proxy columns",
    "neither": "Without either",
}
VARIANT_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
)
MARGIN: Final[float] = 0.10
ABSOLUTE_DOMAIN: Final[tuple[float, float]] = (4.1, 5.5)
CONTRAST_DOMAIN: Final[tuple[float, float]] = (-1.5, 0.5)
BAND_DOMAIN: Final[tuple[float, float]] = (-0.8, 0.6)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(
    *,
    x: float,
    domain: tuple[float, float],
    dash: tuple[int, int] | None = None,
    colour: str = "#999999",
) -> alt.Chart:
    mark = alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(color=colour, strokeDash=dash or [])
    return mark.encode(x=alt.X("x:Q", scale=alt.Scale(domain=list(domain))))  # ty: ignore[unresolved-attribute]


def _arm_scale() -> alt.Scale:
    return alt.Scale(
        domain=[ARM_LABELS[arm] for arm in ARM_ORDER],
        range=[ARM_COLOURS[arm] for arm in ARM_ORDER],
    )


def _absolute_panel() -> alt.LayerChart | alt.FacetChart:
    """Each arm's absolute quantile score with its 95% interval, and the reference lines."""
    absolute = pl.read_parquet(common.RESULTS_DIR / "absolute.parquet")
    references = pl.read_parquet(common.RESULTS_DIR / "stored_reference.parquet")
    rows = ("d0", "d1l", *ARM_ORDER)
    data = absolute.filter(pl.col("arm").is_in(rows)).with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        colour=pl.col("arm").replace_strict(ARM_COLOURS),
    )
    order = [ARM_LABELS[arm] for arm in rows]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("colour:N", scale=None),
    )
    d0_score = absolute.filter(pl.col("arm") == "d0")["qs"].item()
    lines = [
        _rule(x=d0_score, domain=ABSOLUTE_DOMAIN, colour=GREY),
        *[
            _rule(x=row["qs"], domain=ABSOLUTE_DOMAIN, dash=(4, 3), colour="black")
            for row in references.iter_rows(named=True)
        ],
    ]
    labels = [
        alt.Chart(references.filter(pl.col("arm") == name))
        .mark_text(dy=-4, fontSize=10, baseline="bottom", align=align, color="black")
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("qs:Q", scale=alt.Scale(domain=list(ABSOLUTE_DOMAIN))),
            y=alt.value(0),
            text="arm:N",
        )
        for name, align in (("manual_heuristic", "right"), ("climatology", "left"))
    ]
    return alt.layer(
        *lines,
        *labels,
        base.mark_rule().encode(x="qs_lower:Q", x2="qs_upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=list(ABSOLUTE_DOMAIN)),
            )
        ),
    ).properties(width=PLOT_WIDTH, height=170)


def _contrast_panel() -> alt.LayerChart | alt.FacetChart:
    """The verdict contrasts N and T of the four arms, with the margin and the gain threshold."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "verdict_family.parquet")
        .filter(pl.col("contrast").is_in(["N", "T"]))
        .with_columns(
            label=pl.col("treatment").replace_strict(ARM_LABELS),
            kind=pl.col("contrast").replace_strict(
                {"N": "N: minus d0", "T": "T: minus placebo twin"}
            ),
        )
    )
    order = [ARM_LABELS[arm] for arm in ARM_ORDER]
    kinds = ["N: minus d0", "T: minus placebo twin"]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("kind:N", sort=kinds),
        color=alt.Color("label:N", scale=_arm_scale(), legend=None),
    )
    return alt.layer(
        _rule(x=0.0, domain=CONTRAST_DOMAIN),
        _rule(x=MARGIN, domain=CONTRAST_DOMAIN, dash=(4, 3), colour="black"),
        _rule(x=-MARGIN, domain=CONTRAST_DOMAIN, dash=(4, 3), colour="black"),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(CONTRAST_DOMAIN)),
            ),
            shape=alt.Shape(
                "kind:N",
                sort=kinds,
                scale=alt.Scale(domain=kinds, range=["circle", "diamond"]),
                legend=alt.Legend(title=None, orient="top", symbolFillColor=GREY),
            ),
        ),
    ).properties(width=PLOT_WIDTH, height=190)


def figure_headline() -> alt.TopLevelMixin:
    """Absolute scores above, verdict contrasts below."""
    return (
        alt.vconcat(_absolute_panel(), _contrast_panel())
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Starting the XGBoost fit from the forecast solar output beats the best guess",
                lines=[
                    (
                        "Top: each arm's quantile score (dot) with its 95% interval from "
                        "re-drawing whole weeks of forecast start and seeds. Grey line: d0, the "
                        "best guess. Dashed lines: climatology and manual_heuristic, scored on 40 "
                        "fewer rows. Bottom: planned contrasts with 98.75% intervals. Left dashed "
                        "line, -0.10: a gain. Right dashed line, +0.10: the margin an arm must "
                        "stay under to match d0. Grey line, 0: same score. Quantile score in "
                        "points of capacity (1 point is 1% of a substation's capacity), on "
                        "daylight half-hours of April to September at five Primary substations."
                    )
                ],
            )
        )
    )


def figure_by_lead_band() -> alt.TopLevelMixin:
    """Contrast N of the four arms in each lead band, at 95%."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "exploratory.parquet")
        .filter((pl.col("contrast") == "N_by_band") & pl.col("treatment").is_in(list(ARM_ORDER)))
        .with_columns(
            band=pl.col("scope").str.replace("band ", ""),
            label=pl.col("treatment").replace_strict(ARM_LABELS),
        )
    )
    labels = [ARM_LABELS[arm] for arm in ARM_ORDER]
    base = alt.Chart(data).encode(
        y=alt.Y(
            "band:N", sort=list(BAND_NAMES), title="Lead time", scale=alt.Scale(paddingOuter=0.1)
        ),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            sort=labels,
            scale=_arm_scale(),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
        ),
    )
    return alt.layer(
        _rule(x=0.0, domain=BAND_DOMAIN),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score minus d0's (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(BAND_DOMAIN)),
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="The gain sits mostly within 6 days of the forecast start",
            lines=[
                "Each arm's quantile score minus d0's in each lead band. Dot: estimate. Line: 95% "
                "interval from re-drawing whole weeks of forecast start and seeds. Line at 0: "
                "same score as d0. All rows are exploratory. In the 10+ day band the fitted "
                "weight is zero for sf and am. " + SCOPE
            ],
        ),
    )


def figure_slope_diagnostic() -> alt.TopLevelMixin:
    """The median oracle slope by lead band for the four variants of `d1l`."""
    data = (
        pl.read_parquet(common.WEIGHTS_FILE)
        .group_by("variant", "band")
        .agg(slope=pl.col("slope_oracle").median())
        .with_columns(
            lead_band=pl.col("band").replace_strict(dict(enumerate(BAND_NAMES))),
            label=pl.col("variant").replace_strict(VARIANT_LABELS),
        )
    )
    labels = list(VARIANT_LABELS.values())
    base = alt.Chart(data).encode(
        x=alt.X(
            "lead_band:N", sort=list(BAND_NAMES), title="Lead time", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "slope:Q",
            title="Slope of the residual on the CAMS shift",
            scale=alt.Scale(domain=[-1.0, 0.0]),
        ),
        color=alt.Color(
            "label:N",
            sort=labels,
            scale=alt.Scale(domain=labels, range=list(VARIANT_COLOURS)),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
        ),
    )
    return alt.layer(
        base.mark_line(),  # ty: ignore[invalid-argument-type]
        base.mark_point(filled=True, size=60),  # ty: ignore[invalid-argument-type]
    ).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Lagged power absorbs none of the anomaly; cloud proxies some",
            lines=[
                (
                    "Slope of the out-of-fold residual of the median forecast (actual minus "
                    "forecast) on the shift that CAMS sunshine would cause, median over five "
                    "series and two seasons, on the training window. A slope of -1 is what a "
                    "forecast "
                    "holding none of the anomaly would show if CAMS were exact; a slope nearer 0 "
                    "means the forecast already holds part of the anomaly, or CAMS is noisy. Each "
                    "line is d1l with the named columns removed. Exploratory. Training months: "
                    "April to September 2024 and 2025 (July to September rests on 2024)."
                )
            ],
        ),
    )


def figure_by_month() -> alt.TopLevelMixin:
    """Contrast N of the four arms in each valid month."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "by_month.parquet")
        .filter(pl.col("contrast").is_in([f"N {arm}" for arm in ARM_ORDER]))
        .with_columns(
            arm=pl.col("contrast").str.replace("N ", ""),
            month=pl.col("valid_month").replace_strict(MONTH_NAMES),
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    order = list(MONTH_NAMES.values())
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:N", sort=order, title="Valid month (local)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "difference:Q",
                title=None,
                scale=alt.Scale(domain=[-0.4, 0.5]),
                axis=alt.Axis(values=[-0.4, 0, 0.4]),
            ),
            color=alt.Color("label:N", scale=_arm_scale(), legend=None),
        )
        .properties(width=PLOT_WIDTH, height=70)
        .facet(
            row=alt.Row(
                "label:N",
                sort=[ARM_LABELS[arm] for arm in ARM_ORDER],
                title=None,
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",
                    labelOrient="top",
                    labelAnchor="start",
                    labelFontSize=11,
                ),
            )
        )
        .properties(
            title=_title(
                text="sf and am beat d0 in every month; b loses from April to July",
                lines=[
                    (
                        "Mean quantile score minus d0's in each local valid month, in points "
                        "of capacity (more negative is better). Bars below zero: the arm beat "
                        "d0 in that month. Exploratory. Five Primary substations, daylight "
                        "half-hours of April to September, one forecast run every 5 days."
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write every figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_headline": figure_headline,
        "figure_2_by_lead_band": figure_by_lead_band,
        "figure_3_slope_diagnostic": figure_slope_diagnostic,
        "figure_4_by_month": figure_by_month,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
