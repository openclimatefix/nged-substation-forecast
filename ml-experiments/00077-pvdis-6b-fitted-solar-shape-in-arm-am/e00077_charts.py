"""Draw experiment 00077's figures from the saved results, as SVG, headline contrasts first.

1. The headline: each arm's absolute quantile score, and the three verdict contrasts with their
   98.33% intervals and the 0.5% threshold.
2. The fitted solar shapes through a June day, for each Primary, beside the all-south shape.
3. The contrast of the fitted shape against `am` in each lead band (exploratory).
4. The evening-west check: each Primary's after-dark response to daytime cloud, beside its fitted
   west capacity (exploratory, descriptive).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00077_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.BASE_WORK_DIR / "results"
CHART_DIR: Final[Path] = common.BASE_WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 80
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, forecast runs one "
    "every 5 days: July to September 2025 and April to June 2026. Quantile score in points of "
    "capacity (1 point is 1% of a substation's capacity)."
)
GREY: Final[str] = "#8A8A8A"
ARM_LABELS: Final[dict[str, str]] = {
    "am": "am: all-south shape, tilt 35 (experiment 64)",
    "am_s25": "am_s25: all-south shape, tilt 25",
    "am_blend": "am_blend: fitted shape",
    "am_blend_twin": "am_blend_twin: fitted shape, sunshine of another date",
    "am_shuffled": "am_shuffled: another Primary's fitted shape",
    "am0": "am0: no solar term (experiment 76)",
    "d0": "d0: best guess (reference)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "am": GREY,
    "am_s25": ocf.DATA_BLUE,
    "am_blend": ocf.DATA_GREEN,
    "am_blend_twin": ocf.DATA_SKY,
    "am_shuffled": ocf.DATA_PURPLE,
    "am0": GREY,
    "d0": GREY,
}
ABSOLUTE_ARMS: Final[tuple[str, ...]] = (
    "d0",
    "am0",
    "am",
    "am_s25",
    "am_blend",
    "am_shuffled",
    "am_blend_twin",
)
SITE_COLOURS: Final[dict[str, str]] = {
    "D9": ocf.BRAND_ORANGE,
    "D10": ocf.DATA_BLUE,
    "D11": ocf.DATA_SKY,
    "D14": ocf.DATA_PURPLE,
    "D16": ocf.DATA_GREEN,
}
VERDICT_LABELS: Final[dict[str, str]] = {
    "C1": "C1: fitted shape minus am",
    "C2": "C2: fitted shape minus all-south at tilt 25",
    "C3": "C3: fitted shape minus its placebo twin",
}
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
ABSOLUTE_DOMAIN: Final[tuple[float, float]] = (4.0, 5.2)
CONTRAST_DOMAIN: Final[tuple[float, float]] = (-0.6, 0.4)
BAND_DOMAIN: Final[tuple[float, float]] = (-0.3, 0.3)


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
    colour: str = GREY,
) -> alt.Chart:
    mark = alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(color=colour, strokeDash=dash or [])
    return mark.encode(x=alt.X("x:Q", scale=alt.Scale(domain=list(domain))))  # ty: ignore[unresolved-attribute]


def _threshold() -> float:
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet")
    return 0.005 * float(absolute.filter(pl.col("arm") == "am")["qs"][0])


def _absolute_panel() -> alt.LayerChart | alt.FacetChart:
    """Each arm's absolute quantile score with its 95% interval."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(pl.col("arm").is_in(ABSOLUTE_ARMS))
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            colour=pl.col("arm").replace_strict(ARM_COLOURS),
        )
    )
    order = [ARM_LABELS[arm] for arm in ABSOLUTE_ARMS]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color("colour:N", scale=None),
    )
    return alt.layer(
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
    """The three verdict contrasts with the threshold."""
    data = pl.read_parquet(RESULTS_DIR / "verdict.parquet").with_columns(
        label=pl.col("contrast").replace_strict(VERDICT_LABELS)
    )
    order = list(VERDICT_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.value(ocf.DATA_GREEN),
    )
    return alt.layer(
        _rule(x=0.0, domain=CONTRAST_DOMAIN),
        _rule(x=-_threshold(), domain=CONTRAST_DOMAIN, dash=(4, 3), colour="black"),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(CONTRAST_DOMAIN)),
            )
        ),
    ).properties(width=PLOT_WIDTH, height=110)


def figure_headline() -> alt.TopLevelMixin:
    """Absolute scores above, verdict contrasts below."""
    return (
        alt.vconcat(_absolute_panel(), _contrast_panel())
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="The fitted solar shape against the all-south shape in arm am",
                lines=[
                    "Top: each arm's quantile score (dot) with its 95% interval from re-drawing "
                    "whole weeks of forecast start and seeds. Bottom: the three planned contrasts "
                    "with 98.33% intervals. Line at 0: same score. Dashed line: a gain of 0.5% of "
                    "am's score, the smallest gain worth a table of shares. " + SCOPE
                ],
            )
        )
    )


def figure_shapes() -> alt.TopLevelMixin:
    """The fitted shape of each Primary through a June day, beside the all-south shape."""
    pieces = []
    for shape in ("blend", "s25"):
        table = pl.read_parquet(common.BASE_WORK_DIR / "solar" / shape / "climatology.parquet")
        pieces.append(
            table.filter(pl.col("month") == 6).select(
                "site", "hh", shape=pl.lit(shape), value=pl.col("clim_cams")
            )
        )
    data = (
        pl.concat(pieces)
        .with_columns(hour=pl.col("hh") / 2.0)
        .filter((pl.col("hour") >= 4) & (pl.col("hour") <= 20))
    )
    sites = list(SITE_COLOURS)
    blend = data.filter(pl.col("shape") == "blend")
    south = data.filter(pl.col("shape") == "s25")
    south_mean = (
        south.group_by("hour")
        .agg(value=pl.col("value").mean())
        .with_columns(site=pl.lit("All-south, tilt 25 (mean of the Primaries)"))
    )
    x = alt.X(
        "hour:Q",
        title="Hour of day (UTC, period ending)",
        scale=alt.Scale(domain=[4, 20]),
        axis=alt.Axis(values=[4, 8, 12, 16, 20]),
    )
    lines = (
        alt.Chart(blend)
        .mark_line(clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("value:Q", title="Mean shape in June (output per MW of AC capacity)"),
            color=alt.Color(
                "site:N",
                sort=sites,
                scale=alt.Scale(domain=sites, range=[SITE_COLOURS[s] for s in sites]),
                legend=alt.Legend(title="Fitted shape of", orient="top", columns=5),
            ),
        )
    )
    reference = (
        alt.Chart(south_mean)
        .mark_line(clip=True, color=GREY, strokeDash=[5, 3])
        .encode(x=x, y="value:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(reference, lines).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="The fitted solar shape of each Primary through a June day",
            lines=[
                (
                    "CAMS output per MW of AC capacity, averaged over every June day of the "
                    "training window (to 30 June 2025), for the fitted east-south-west blend of "
                    "each Primary. Grey dashed line: all-south panels at a tilt of 25 degrees. "
                    "The shares are a flexible shape and not a measurement of panel "
                    "orientation. Daylight hours only."
                )
            ],
        ),
    )


def figure_by_lead_band() -> alt.TopLevelMixin:
    """The fitted shape minus `am` in each lead band, at 95%."""
    data = (
        pl.read_parquet(RESULTS_DIR / "exploratory.parquet")
        .filter(pl.col("contrast").str.starts_with("B "))
        .with_columns(band=pl.col("scope").str.replace("band ", ""))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "band:N", sort=list(BAND_NAMES), title="Lead time", scale=alt.Scale(paddingOuter=0.1)
        ),
        color=alt.value(ocf.DATA_GREEN),
    )
    return alt.layer(
        _rule(x=0.0, domain=BAND_DOMAIN),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Fitted shape minus am (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(BAND_DOMAIN)),
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="The gain of the fitted shape by lead time",
            lines=[
                "Quantile score of the fitted shape minus that of am in each lead band. Dot: "
                "estimate. Line: 95% interval from re-drawing whole weeks of forecast start and "
                "seeds. Line at 0: same score as am. All rows are exploratory. In the 10+ day "
                "band the weight of the solar term is zero. " + SCOPE
            ],
        ),
    )


def figure_evening() -> alt.TopLevelMixin:
    """After-dark response to daytime cloud against the fitted west capacity, per Primary."""
    data = pl.read_parquet(common.BASE_WORK_DIR / "evening.parquet")
    base = alt.Chart(data).encode(
        x=alt.X("west_capacity_mw:Q", title="Fitted west capacity (MW)"),
        color=alt.Color(
            "site:N",
            scale=alt.Scale(
                domain=list(SITE_COLOURS), range=[SITE_COLOURS[s] for s in SITE_COLOURS]
            ),
            legend=alt.Legend(title="Primary", orient="top", columns=5),
        ),
    )
    return alt.layer(
        _rule_y(),
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("cloud_low:Q"), y2="cloud_high:Q"
        ),
        base.mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "cloud_slope_mw:Q",
                title="After-dark demand change from clear to fully cloudy day (MW)",
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="After-dark demand response and fitted west capacity",
            lines=[
                (
                    "For each Primary: the change in half-hourly demand after dark (18:00 to "
                    "23:30 local time, April to September, calendar cell and heating degree-days "
                    "removed) between a clear and a fully cloudy day, against the Primary's "
                    "fitted west capacity. Dot: estimate. Line: 95% interval from re-drawing "
                    "7-day blocks. Five Primaries, training window only. Descriptive; no verdict "
                    "reads it."
                )
            ],
        ),
    )


def _rule_y() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )


def main() -> None:
    """Write every figure as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1_headline", figure_headline()),
        ("figure_2_shapes", figure_shapes()),
        ("figure_3_by_lead_band", figure_by_lead_band()),
        ("figure_4_evening", figure_evening()),
    ):
        chart.save(CHART_DIR / f"{name}.svg")
        print(f"wrote {CHART_DIR / f'{name}.svg'}")


if __name__ == "__main__":
    main()
