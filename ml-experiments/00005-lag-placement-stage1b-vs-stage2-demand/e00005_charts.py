"""Draw experiment 00005's figures from the saved arms and results, as SVG.

1. The quantile score by lead band for the Primary substations, one dot per arm with its 95%
   interval (the absolute scores).
2. The five planned contrasts for the Primary substations, by lead scope, with their 99% intervals.
3. Example forecasts: for three Primary substations, the run whose stage 2 quantile score over the
   first 72 hours is the median of that substation's runs, with the observed power and the 5th to
   95th and 25th to 75th percentile bands of three arms.
4. The same, for the run with the largest score over the first 72 hours (the worst case).

The x axis of the example figures is hours since the forecast's origin, never a calendar date.
The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00005_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
"""The plot area's width in pixels; every title and subtitle line is wrapped to it."""
WRAP_CHARACTERS: Final[int] = 92
"""How many characters of 11-pixel subtitle text fit across `PLOT_WIDTH`."""
HEADLINE: Final[str] = "Primary"
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_BANDS)
EXAMPLE_HOURS: Final[float] = 72.0
EXAMPLE_SEED: Final[int] = 0
EXAMPLE_SERIES: Final[int] = 3

ARM_LABELS: Final[dict[str, str]] = {
    "r2": "Climatology (month, time of day, weekday or weekend)",
    "r0_from2023": "Stage 1 only, no lagged power (R0)",
    "r0c_from2023": "Stage 1 plus calibrator, no lagged power (R0c)",
    "l3_from2023": "Lagged power in the calibrator (L3)",
    "l2c_from2023": "Lagged power in stage 1b, then the calibrator (L2c)",
    "l2l3_from2023": "Lagged power in stage 1b and the calibrator (L2+L3)",
}
ARM_COLOURS: Final[list[str]] = [
    "#999999",
    ocf.DATA_SKY,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    ocf.BRAND_ORANGE,
]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 (primary): L2c minus L3": "C1 (primary): lagged power in stage 1b minus in the calibrator",
    "C2: L3 minus R0c": "C2: lagged power in the calibrator minus none",
    "C3: L2c minus R0c": "C3: lagged power in stage 1b minus none",
    "C4: L2+L3 minus L2c": "C4: both places minus stage 1b only",
    "C5: L2+L3 minus L3": "C5: both places minus the calibrator only",
}
EXAMPLE_ARMS: Final[dict[str, str]] = {
    "r0c_from2023": "No lagged power (R0c)",
    "l3_from2023": "Lagged power in the calibrator (L3)",
    "l2c_from2023": "Lagged power in stage 1b (L2c)",
}
DEFINITIONS: Final[str] = (
    "Stage 1 is an XGBoost model that maps ECMWF ensemble weather (51 members) to demand; the "
    "calibrator is a second XGBoost model that turns the 51 forecasts into 9 quantiles. "
    "Lagged power means recent observed demand."
)
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026. "
    "Lagged power uses ERA5 weather at past times, so these arms are an upper bound on what "
    "production can do."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a title whose subtitle paragraphs are wrapped to the plot area's width."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The Primary substations' quantile score by lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & (pl.col("metric") == "quantile_score")
            & pl.col("scope").is_in(list(BANDS))
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    labels = list(ARM_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=labels, range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Quantile score (points of capacity; smaller is better)")
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text=(
                "Lagged power in the calibrator scores as low as any arm at every lead "
                "for the Primary substations"
            ),
            lines=[
                (
                    "Quantile score of the 9-quantile forecast, an estimate of the continuous "
                    "ranked probability score, in points of the substation's capacity "
                    "(smaller is better), by how many days ahead the forecast is."
                ),
                DEFINITIONS,
                SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts, by lead scope, with their 99% intervals."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & pl.col("label").is_in(list(CONTRAST_LABELS))
            & pl.col("scope").is_in(["up to 48 h", *BANDS])
        )
        .with_columns(label=pl.col("label").replace_strict(CONTRAST_LABELS))
    )
    labels = list(CONTRAST_LABELS.values())
    order = ["up to 48 h", *BANDS]
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=order, title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=labels,
                range=[
                    ocf.BRAND_ORANGE,
                    ocf.DATA_BLUE,
                    ocf.DATA_PURPLE,
                    ocf.DATA_GREEN,
                    ocf.DATA_SKY,
                ],
            ),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Change in quantile score (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=320,
        title=_title(
            text="Lagged power in the calibrator beats lagged power in stage 1b at every lead",
            lines=[
                (
                    "Change in the quantile score between two arms (points of the substation's "
                    "capacity; more negative means the first arm is better), by how many days "
                    "ahead the forecast is."
                ),
                DEFINITIONS,
                SCOPE,
                (
                    "Dot: estimate. Line: 99% interval, widened for the five planned contrasts. "
                    "Solid rule: no difference."
                ),
            ],
        ),
    )


def _examples(*, worst: bool) -> pl.DataFrame:
    """Return the example cases: three Primary substations, one run each, three arms.

    The substations are the three Primary substations with the lowest labels. A substation's
    example run is the one whose stage 2 (L3) quantile score over the first 72 hours is the median
    of its runs, or the largest when `worst` is set. The rule, not the eye, picks the run.

    Args:
        worst: Pick the worst run instead of the median run.

    Returns:
        Rows keyed by series label, arm label, and lead in hours, with the 9 quantiles and the
        observed power over capacity.
    """
    series = common.series_table().select("time_series_id", "site", "substation_type")
    primary = (
        series.filter(pl.col("substation_type") == HEADLINE)
        .sort(pl.col("site").str.slice(1).cast(pl.Int32))
        .head(EXAMPLE_SERIES)
    )
    reference = (
        pl.read_parquet(ARMS_DIR / "l3_from2023.parquet")
        .filter((pl.col("seed") == EXAMPLE_SEED) & (pl.col("lead_hours") <= EXAMPLE_HOURS))
        .join(primary, on="time_series_id")
        .group_by("site", "time_series_id", "power_fcst_init_time")
        .agg(pl.col("quantile_score").mean().alias("score"))
        .sort("site", "score")
    )
    chosen = []
    for _, part in reference.group_by("site", maintain_order=True):
        rows = part.sort("score")
        chosen.append(rows.tail(1) if worst else rows[rows.height // 2 : rows.height // 2 + 1])
    picked = pl.concat(chosen)
    pieces = []
    for arm, label in EXAMPLE_ARMS.items():
        frame = pl.read_parquet(ARMS_DIR / f"{arm}.parquet").filter(
            (pl.col("seed") == EXAMPLE_SEED) & (pl.col("lead_hours") <= EXAMPLE_HOURS)
        )
        pieces.append(
            frame.join(
                picked.select("site", "time_series_id", "power_fcst_init_time"),
                on=["time_series_id", "power_fcst_init_time"],
            ).with_columns(arm=pl.lit(label))
        )
    return pl.concat(pieces).with_columns(
        actual=pl.col("actual") * 100.0,
        **{c: pl.col(c) * 100.0 for c in common.LEVEL_COLUMNS},
    )


def _example_figure(*, worst: bool) -> alt.TopLevelMixin:
    data = _examples(worst=worst)
    arms = list(EXAMPLE_ARMS.values())
    sites = sorted(data["site"].unique().to_list(), key=lambda s: int(s[1:]))
    rows = []
    for site in sites:
        panels = []
        for column, (arm, colour) in enumerate(
            zip(arms, [ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN], strict=True)
        ):
            part = data.filter((pl.col("site") == site) & (pl.col("arm") == arm))
            x = alt.X(
                "lead_hours:Q",
                title="Hours since the forecast's origin",
                scale=alt.Scale(domain=[0, EXAMPLE_HOURS], nice=False),
                axis=alt.Axis(values=[0, 24, 48, 72]),
            )
            outer = (
                alt.Chart(part)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.2, color=colour)
                .encode(
                    x=x,
                    y=alt.Y(
                        "q05:Q",
                        title="Demand (% of capacity)" if column == 0 else None,
                        scale=alt.Scale(zero=False),
                    ),
                    y2="q95:Q",
                )
            )
            inner = (
                alt.Chart(part)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.35, color=colour)
                .encode(x=x, y="q25:Q", y2="q75:Q")
            )
            median = alt.Chart(part).mark_line(color=colour, strokeWidth=2).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
            observed = (
                alt.Chart(part)  # ty: ignore[unresolved-attribute]
                .mark_line(color="#222222", strokeWidth=1.5, strokeDash=[4, 2])
                .encode(x=x, y="actual:Q")
            )
            panels.append(
                alt.layer(outer, inner, median, observed).properties(
                    width=(PLOT_WIDTH - 60) // 3,
                    height=120,
                    title=alt.TitleParams(
                        arm.split("(")[-1].rstrip(")") + f", {site}", anchor="start", fontSize=12
                    ),
                )
            )
        rows.append(alt.hconcat(*panels).resolve_scale(y="shared"))
    kind = "worst" if worst else "typical"
    headline = (
        "In the worst runs, all three arms miss the level by the same amount"
        if worst
        else (
            "In typical runs, the three arms follow the daily shape "
            "and differ mainly in the first day's level"
        )
    )
    return alt.vconcat(*rows).properties(
        title=_title(
            text=headline,
            lines=[
                "The three Primary substations with the lowest labels, one forecast run each.",
                f"Each run is picked by rule: the {kind} run by L3's quantile score over 72 hours."
                + (" The rule is unfavourable to L3." if worst else ""),
                "Dashed line: observed demand. Solid line: median forecast.",
                "Dark band: 25th to 75th percentile. Light band: 5th to 95th percentile.",
                "R0c: no lagged power. L3: lagged power in the calibrator. L2c: in stage 1b.",
                "Seed 0 only. Lagged power uses ERA5 weather at past times (an upper bound).",
            ],
        )
    )


def main() -> None:
    """Draw the four figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "1": figure_1,
        "2": figure_2,
        "3": lambda: _example_figure(worst=False),
        "4": lambda: _example_figure(worst=True),
    }
    for key, build in figures.items():
        chart = build()
        chart.save(CHART_DIR / f"figure_{key}.svg")
        chart.save(CHART_DIR / f"figure_{key}.png", scale_factor=2)
        print(f"figure {key} written")


if __name__ == "__main__":
    main()
