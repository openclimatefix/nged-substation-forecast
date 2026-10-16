"""Draw experiment 00063's four figures from the saved results, as SVG.

1. One summer week at one Primary: the residual of the baseline without radiation, the fitted
   solar term, and the contribution of the radiation inputs to the frozen v1 baseline.
2. C1 by Primary, with 95% intervals, and pooled two ways (post hoc for the shared-month interval).
3. S1: the recovered share of an injected solar farm's output, by farm, beside the positive control
   and the agreement between each farm's anomaly and the CAMS anomaly (post hoc).
4. S2: the recovered share on placebo intervals, by donor, sorted by the donor's mean power.

The x axis of figure 1 counts days of the week; the calendar month and year go in the page text.
Run `e00063_report.py` and `e00063_inject.py` first. The SVGs are afterwards optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00063-*/e00063_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00063_common as common
import e00063_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
"""The plot area's width in pixels; every subtitle line is wrapped to it."""
WRAP_CHARACTERS: Final[int] = 92
"""How many characters of 11-pixel subtitle text fit across `PLOT_WIDTH`."""
WEEK_SERIES: Final[int] = 7
"""The Primary drawn in figure 1."""
MIN_WEEK_ROWS: Final[int] = 150
"""A week is drawn only if it holds this many scored half-hours."""
DAY_START_HOUR: Final[int] = 4
DAY_END_HOUR: Final[int] = 20
"""Each day of figure 1 is drawn from 04:00 to 20:00 UTC, the hours that hold daylight."""
X_SCALE: Final[alt.Scale] = alt.Scale(domain=[0.5, 1.2])
"""The shared x scale of figure 3."""
JITTER_SEED: Final[int] = 63
DURATION_BANDS: Final[dict[str, tuple[int, int]]] = {
    "6 to 12 hours": (6, 12),
    "24 to 72 hours": (24, 72),
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a title whose subtitle paragraphs are wrapped to the plot area's width."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=wrapped,
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _rule(*, x: float, dash: list[int] | None = None) -> alt.Chart:
    """Return a vertical reference rule at `x`."""
    return (
        alt.Chart(pl.DataFrame({"x": [x]}))
        .mark_rule(color="#333333", strokeDash=dash or [])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )


def _clearest_week(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the April to September week of `WEEK_SERIES` with the largest mean solar term.

    A week runs Monday to Sunday and must hold `MIN_WEEK_ROWS` scored half-hours on all 7 days.
    """
    own = rows.filter(pl.col("series") == WEEK_SERIES, pl.col("summer")).with_columns(
        week=pl.col("valid_time").dt.truncate("1w"),
        day=pl.col("valid_time").dt.weekday() - 1,
        solar_term=(pl.col("median_c1") - pl.col("median_control")),
    )
    weeks = (
        own.group_by("week")
        .agg(
            n=pl.len(),
            days=pl.col("day").n_unique(),
            mean_abs_term=pl.col("solar_term").abs().mean(),
        )
        .filter(pl.col("n") >= MIN_WEEK_ROWS, pl.col("days") == 7)
        .sort("mean_abs_term", descending=True)
    )
    best = weeks["week"][0]
    print(f"figure 1: week starting {best.date()}, {weeks['n'][0]} half-hours")
    return own.filter(pl.col("week") == best)


def _day_position() -> pl.Expr:
    """Return the x position: the day (0 to 6) plus the share of 04:00 to 20:00 UTC elapsed."""
    hours = pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60
    return pl.col("day") + ((hours - DAY_START_HOUR) / (DAY_END_HOUR - DAY_START_HOUR)).clip(0, 1)


def figure_1(*, rows: pl.DataFrame, norms: dict[int, float]) -> alt.TopLevelMixin:
    """The raw series: residual, fitted solar term, and the radiation inputs' contribution."""
    week = _clearest_week(rows=rows)
    scale = 100.0 / norms[WEEK_SERIES]
    long = pl.concat(
        [
            week.select(
                "day",
                days=_day_position(),
                value=(pl.col("observed_mw") - pl.col("median_control")) * scale,
                line=pl.lit("Residual: metered power minus the baseline without radiation"),
            ),
            week.select(
                "day",
                days=_day_position(),
                value=pl.col("solar_term") * scale,
                line=pl.lit("Fitted solar term of the joint fit"),
            ),
            week.select(
                "day",
                days=_day_position(),
                value=(pl.col("median_v1") - pl.col("median_control")) * scale,
                line=pl.lit("What the radiation inputs add to the v1 baseline"),
            ),
        ]
    )
    labels = long["line"].unique(maintain_order=True).to_list()
    colours = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY]
    chart = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(clip=True, strokeWidth=1.6)
        .encode(
            x=alt.X(
                "days:Q",
                title="Day of the week (Monday is day 1; each day runs from 04:00 to 20:00 UTC)",
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(
                    values=[day + 0.5 for day in range(7)],
                    labelExpr="'Day ' + floor(datum.value + 0.5)",
                    grid=False,
                ),
            ),
            y=alt.Y("value:Q", title="Power (points of capacity)"),
            color=alt.Color(
                "line:N",
                scale=alt.Scale(domain=labels, range=colours),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
            ),
            detail="day:N",
        )
        .properties(width=PLOT_WIDTH, height=260)
    )
    separators = (
        alt.Chart(pl.DataFrame({"x": [float(day) for day in range(1, 7)]}))
        .mark_rule(color="#CCCCCC")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#333333").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(separators, zero, chart).properties(
        title=_title(
            text="The fitted solar term moves with the residual, and explains only part of it",
            lines=[
                (
                    "One substation (Trusthorpe), the April to September week with the largest "
                    "fitted solar term, from the end of April 2026. Each line is power in points "
                    "of the substation's capacity, for daylight half-hours outside logged "
                    "switching. Night is not drawn. Zero marks no difference from the baseline "
                    "without radiation."
                ),
            ],
        )
    )


def _series_values(*, rows: pl.DataFrame, norms: dict[int, float]) -> pl.DataFrame:
    """Return `series`, `month_id`, and the C1 difference in points of capacity per row."""
    return rows.select(
        "series",
        "month_id",
        value=(pl.col("abs_error_c1") - pl.col("abs_error_v1"))
        / pl.col("series").replace_strict(norms, return_dtype=pl.Float64)
        * 100.0,
    )


def figure_2(*, rows: pl.DataFrame, norms: dict[int, float]) -> alt.TopLevelMixin:
    """C1 by Primary with 95% intervals, and pooled two ways."""
    values = _series_values(rows=rows, norms=norms)
    records: list[dict[str, object]] = []
    for series in sorted(values["series"].unique().to_list()):
        sub = values.filter(pl.col("series") == series)
        mean, lo, hi, n, _ = report.pooled_bootstrap(values=sub)
        records.append(
            {
                "label": f"{common.SERIES_NAMES[series]} ({n:,} half-hours)",
                "difference": mean,
                "lower": lo,
                "upper": hi,
                "kind": "Single Primary",
            }
        )
    records.sort(key=lambda record: float(record["difference"]))  # ty: ignore[invalid-argument-type]
    mean, lo, hi, n, _ = report.pooled_bootstrap(values=values)
    pooled = [
        {
            "label": "All 5, months drawn per Primary",
            "difference": mean,
            "lower": lo,
            "upper": hi,
            "kind": "Pooled (planned)",
        }
    ]
    mean, lo, hi, _ = report.shared_month_bootstrap(values=values)
    pooled.append(
        {
            "label": "All 5, same months for all (post hoc)",
            "difference": mean,
            "lower": lo,
            "upper": hi,
            "kind": "Pooled (post hoc)",
        }
    )
    data = pl.DataFrame([*records, *pooled])
    print("figure 2 rows (C1, points of capacity; difference, 95% interval):")
    for row in data.iter_rows(named=True):
        print(
            f"  {row['label']:40s} {row['difference']:+.3f} "
            f"({row['lower']:+.3f} to {row['upper']:+.3f})"
        )
    order = [str(record["label"]) for record in [*records, *pooled]]
    kinds = ["Single Primary", "Pooled (planned)", "Pooled (post hoc)"]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=kinds, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Change in error (points of capacity; more negative is better)",
            scale=alt.Scale(domain=[-0.9, 0.2]),
        )
    )
    bars = base.mark_rule(strokeWidth=2).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_rule(x=0.0), _rule(x=-0.35, dash=[4, 3]), bars, points).properties(
        width=PLOT_WIDTH,
        height=210,
        title=_title(
            text="The joint fit has 0.2 to 0.4 points less error than v1 at each substation",
            lines=[
                (
                    "C1: error of the median forecast with the fitted solar term, minus the error "
                    "of the frozen v1 baseline with radiation inputs. September 2023 to June 2026, "
                    "all daylight half-hours outside logged switching, with oracle sunshine and "
                    "in-sample capacity, so a ceiling. Dot: estimate. Line: 95% interval from "
                    "re-drawing whole calendar months. Solid rule: no difference. Dashed rule: "
                    "the 0.35-point threshold for porting."
                ),
            ],
        ),
    )


def figure_3(*, norms: dict[int, float]) -> alt.TopLevelMixin:
    """S1: recovery of injected farm output, with the positive control and the CAMS agreement."""
    del norms
    s1 = pl.read_parquet(common.RESULTS_DIR / "s1.parquet").filter(pl.col("season") == "all")
    control = pl.read_parquet(common.RESULTS_DIR / "s1_positive_control.parquet").filter(
        pl.col("season") == "all"
    )
    agreement = pl.read_parquet(common.RESULTS_DIR / "s1_farm_agreement.parquet")
    farm_rows = (
        s1.group_by("farm")
        .agg(
            median=pl.col("ratio").median(),
            low=pl.col("ratio").quantile(0.1),
            high=pl.col("ratio").quantile(0.9),
        )
        .join(
            agreement.group_by("farm").agg(slope=pl.col("slope").median()),
            on="farm",
        )
        .with_columns(label="Solar farm " + pl.col("farm"))
        .sort("farm")
    )
    control_label = "Positive control"
    control_row = pl.DataFrame(
        {
            "farm": ["control"],
            "median": [control["ratio"].median()],
            "low": [control["ratio"].min()],
            "high": [control["ratio"].max()],
            "slope": [None],
            "label": [control_label],
        },
        schema_overrides={"slope": pl.Float64},
    )
    data = pl.concat([farm_rows.select(control_row.columns), control_row])
    order = [*farm_rows["label"].to_list(), control_label]
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=order, title=None))
    x_title = "Recovered capacity / injected capacity (1 is exact)"
    gate = (
        alt.Chart(pl.DataFrame({"lo": [0.85], "hi": [1.15]}))
        .mark_rect(color="#EAEAEA", opacity=0.8)
        .encode(x=alt.X("lo:Q", scale=X_SCALE), x2="hi:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            scale=X_SCALE,
            title=x_title,
            axis=alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        ),
        x2="high:Q",
    )
    dots = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("median:Q", scale=X_SCALE)
    )
    slopes = base.mark_point(  # ty: ignore[unresolved-attribute]
        shape="diamond", size=90, color=ocf.BRAND_ORANGE, filled=False, strokeWidth=2
    ).encode(x=alt.X("slope:Q", scale=X_SCALE))
    return alt.layer(
        gate, _rule(x=1.0).encode(x=alt.X("x:Q", scale=X_SCALE)), bars, dots, slopes
    ).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text=(
                "A real farm's output is recovered at 0.7 to 0.8 of its size, the slope of the "
                "farm's anomaly on the CAMS anomaly"
            ),
            lines=[
                (
                    "S1: output of a real solar farm added to a substation's power at 10%, 25%, "
                    "and 50% of the substation's capacity, and the joint fit's recovered "
                    "capacity as a share of the injected capacity. Blue dot: median over 6 "
                    "substations and 3 injection sizes. Blue line: 10th to 90th percentile. "
                    "Orange diamond (post hoc): slope of the farm's anomaly on the CAMS "
                    "anomaly. Solid rule: exact recovery. Grey band: the pre-registered "
                    "0.85 to 1.15. Positive control (post hoc): the fit's own regressor "
                    "shape injected in place of a farm, which must be recovered exactly."
                ),
            ],
        ),
    )


def figure_4() -> alt.TopLevelMixin:
    """S2: the recovered share on placebo intervals, by donor."""
    mean_power = report.donor_mean_power_mw()
    placebo = (
        pl.read_parquet(common.RESULTS_DIR / "s2.parquet")
        .filter(pl.col("kind") == "placebo", pl.col("beta_hat").is_not_nan())
        .with_columns(abs_share=pl.col("beta_hat").abs())
    )
    generator = np.random.default_rng(JITTER_SEED)
    placebo = placebo.with_columns(
        label=pl.col("donor").map_elements(
            lambda donor: f"{common.SERIES_NAMES[donor]} ({mean_power[donor]:.1f} MW)",
            return_dtype=pl.String,
        ),
        band=pl.col("duration_hours").map_elements(
            lambda hours: next(
                name for name, (lo, hi) in DURATION_BANDS.items() if lo <= hours <= hi
            ),
            return_dtype=pl.String,
        ),
        jitter=pl.Series(generator.uniform(-1.0, 1.0, placebo.height)),
    )
    order = [
        f"{common.SERIES_NAMES[donor]} ({mean_power[donor]:.1f} MW)"
        for donor in sorted(mean_power, key=lambda donor: mean_power[donor])
        if donor in placebo["donor"].to_list()
    ]
    p95 = (
        placebo.group_by("label")
        .agg(p95=pl.col("abs_share").quantile(0.95, interpolation="linear"))
        .sort("label")
    )
    base_y = alt.Y("label:N", sort=order, title="Donor (mean absolute power)")
    points = (
        alt.Chart(placebo)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=22, opacity=0.7, aria=False)
        .encode(
            x=alt.X(
                "abs_share:Q",
                title="Recovered share when none was injected (absolute value)",
                scale=alt.Scale(domain=[0, 1.1]),
            ),
            y=base_y,
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1, 1], range=[-9, 9])),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(
                    domain=list(DURATION_BANDS), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title="Interval length", orient="top"),
            ),
        )
    )
    ticks = (
        alt.Chart(p95).mark_tick(color="#111111", thickness=3, size=26).encode(x="p95:Q", y=base_y)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(_rule(x=0.10, dash=[4, 3]), points, ticks).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="Small donors give recovered shares far above 0.10 when none was injected",
            lines=[
                (
                    "S2 placebo: 240 intervals of 6 to 72 hours at six substations, each given "
                    "a donor's real power with a true share of zero. Dot: one interval. Black "
                    "tick: 95th percentile of the absolute recovered share for the donor. "
                    "Dashed rule: the pre-registered limit of 0.10 for that percentile. "
                    "Donors are sorted by mean absolute power over September 2023 to June 2026."
                ),
            ],
        ),
    )


def main() -> None:
    """Draw the four figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    rows = pl.read_parquet(common.RESULTS_DIR / "c_rows.parquet")
    norms = common.effective_capacity_mw()
    figures = {
        "1": lambda: figure_1(rows=rows, norms=norms),
        "2": lambda: figure_2(rows=rows, norms=norms),
        "3": lambda: figure_3(norms=norms),
        "4": figure_4,
    }
    for key, build in figures.items():
        chart = build()
        chart.save(CHART_DIR / f"figure_{key}.svg")
        chart.save(CHART_DIR / f"figure_{key}.png", scale_factor=2)
        print(f"figure {key} written")


if __name__ == "__main__":
    main()
