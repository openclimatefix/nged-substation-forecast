"""Draw experiment 72's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The farms are anonymised (A to F) and the signal chart has no calendar dates on its axis.

    uv run python ml-experiments/00072-*/e00072_charts.py
"""

import json
import textwrap
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00072_detectors import fitted_level
from e00072_inject import METHODS
from e00072_report import DOCS_DIR
from e00072_score import OUTPUT_DIR
from e00072_signal import POOL_SITES, clean_baseline, load_record
from e00072_tune import FROZEN_PATH
from studies.charts import CONTENT_WIDTH_PX

WRAP: Final[int] = 105
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 60
LABELS: Final[dict[str, str]] = {
    "changepoint": "Change-point fit (14 valid days per segment)",
    "changepoint_short_segment": "Change-point fit, 7 valid days (post hoc)",
    "rolling_median": "Rolling median",
    "hyst": "Hysteresis rule (HYST)",
}
COLOURS: Final[dict[str, str]] = {
    "changepoint": ocf.BRAND_ORANGE,
    "changepoint_short_segment": ocf.BRAND_ORANGE_LIGHT,
    "rolling_median": ocf.DATA_BLUE,
    "hyst": ocf.DATA_PURPLE,
}
RESAMPLES: Final[int] = 2000
SEED: Final[int] = 5


def _title(*, text: str, body: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(body, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _interval(*, hits: dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    """Return the pooled rate and a 95% interval from resampling the five farms."""
    rng = np.random.default_rng(SEED)
    sites = list(hits)
    pooled = sum(h for h, _ in hits.values()) / sum(n for _, n in hits.values())
    draws = []
    for _ in range(RESAMPLES):
        pick = [sites[i] for i in rng.integers(0, len(sites), len(sites))]
        draws.append(sum(hits[s][0] for s in pick) / sum(hits[s][1] for s in pick))
    low, high = np.percentile(draws, [2.5, 97.5])
    return pooled, float(low), float(high)


def detection_chart() -> alt.TopLevelMixin:
    """Detection rate against loss size, by duration, April to September starts."""
    events = pl.concat(
        [
            pl.read_parquet(OUTPUT_DIR / "events.parquet"),
            pl.read_parquet(OUTPUT_DIR / "posthoc_events.parquet").with_columns(
                detected_early_window=pl.col("detected_early_window")
            ),
        ]
    ).filter(pl.col("season") == "apr_sep")
    rows = []
    names = {
        30: "Loss lasts 30 days",
        90: "Loss lasts 90 days",
        -1: "Loss lasts to the end of the record",
    }
    for method in (*METHODS, "changepoint_short_segment"):
        for days in (30, 90, -1):
            for size in (0.0, 0.03, 0.10, 0.25):
                sub = events.filter(
                    (pl.col("method") == method)
                    & (pl.col("days") == days)
                    & (pl.col("size") == size)
                )
                if sub.height == 0:
                    continue
                hits = {
                    site: (int(s["detected"].sum()), s.height)
                    for site in POOL_SITES
                    if (s := sub.filter(pl.col("site") == site)).height
                }
                rate, low, high = _interval(hits=hits)
                rows.append(
                    {
                        "method": LABELS[method],
                        "panel": names[days],
                        "size": 100 * size,
                        "rate": rate,
                        "low": low,
                        "high": high,
                    }
                )
    frame = pl.DataFrame(rows)
    order = [LABELS[m] for m in (*METHODS, "changepoint_short_segment")]
    colours = [COLOURS[m] for m in (*METHODS, "changepoint_short_segment")]
    base = alt.Chart().encode(
        x=alt.X(
            "size:Q",
            title="Share of the farm's power lost (%; 0 is the no-loss control)",
            axis=alt.Axis(values=[0, 3, 10, 25]),
        ),
        color=alt.Color(
            "method:N",
            scale=alt.Scale(domain=order, range=colours),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=400),
        ),
    )
    line = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "rate:Q",
            title="Share of losses detected (higher is better)",
            scale=alt.Scale(domain=[0, 1]),
        )
    )
    band = base.mark_errorbar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title=None), y2="high:Q"
    )
    points = base.mark_point(filled=True, size=40, aria=False).encode(y="rate:Q")  # ty: ignore[unresolved-attribute]
    rule = alt.Chart().mark_rule(strokeDash=[4, 3], color="#666").encode(y=alt.datum(0.8))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(rule, band, line, points)
        .properties(width=PLOT_WIDTH, height=130)
        .facet(
            row=alt.Row(
                "panel:N",
                sort=list(names.values()),
                title=None,
                header=alt.Header(
                    labelAnchor="start", labelFontSize=12, labelOrient="top", labelPadding=2
                ),
            ),
            data=frame,
        )
    )
    return chart.properties(
        title=_title(
            text="Only a 25% loss is found reliably, and a 10% loss mostly is not",
            body=(
                "Each point is the share of losses planted in the real power of 5 solar "
                "farms, with onsets from April to September, that a monitor flagged within 28 "
                "days. "
                "Bars: 95% interval from resampling the 5 farms. Dashed line: the 80% the "
                "plan required. Each monitor is tuned to about 0.35 false alarms per "
                "farm-year."
            ),
        )
    )


def false_alarm_chart() -> alt.Chart:
    """False alarms per farm-year on the placebo and the unmodified record."""
    placebo = pl.read_parquet(OUTPUT_DIR / "placebo.parquet")
    short = pl.read_parquet(OUTPUT_DIR / "posthoc_placebo.parquet")
    unmodified = pl.read_parquet(OUTPUT_DIR / "unmodified.parquet").filter(
        (pl.col("site") != "E") & ~pl.col("near_farm_a_low_months")
    )
    years = float(
        placebo.filter(pl.col("draw") == 0)
        .filter(pl.col("method") == "changepoint")["farm_years"]
        .sum()
    )
    rows = []
    for method in METHODS:
        sub = placebo.filter(pl.col("method") == method)
        rows.append(
            {
                "method": LABELS[method],
                "record": "Week-shuffled placebo (20 shuffles)",
                "rate": float(sub["alarms"].sum()) / float(sub["farm_years"].sum()),
            }
        )
        rows.append(
            {
                "method": LABELS[method],
                "record": "Unmodified record",
                "rate": unmodified.filter(pl.col("method") == method).height / years,
            }
        )
    rows.append(
        {
            "method": LABELS["changepoint_short_segment"],
            "record": "Week-shuffled placebo (20 shuffles)",
            "rate": float(short["alarms"].sum()) / float(short["farm_years"].sum()),
        }
    )
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y(
                "method:N",
                sort=[LABELS[m] for m in (*METHODS, "changepoint_short_segment")],
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            x=alt.X("rate:Q", title="False alarms per farm-year (lower is better)"),
            yOffset="record:N",
            color=alt.Color(
                "record:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
        .properties(width=PLOT_WIDTH - 250, height=160)
    )
    return chart.properties(
        title=_title(
            text=("False alarms: 0.29 a year on the placebo, 0.36 to 0.54 on the real record"),
            body=(
                "Alarms of at least 5% on the 5 pool farms. The unmodified record excludes "
                "alarms within 30 days of farm A's three low months. The placebo shuffles "
                "whole weeks among years."
            ),
        )
    )


def signal_chart() -> alt.TopLevelMixin:
    """The signal and the fitted level of each farm, in years since the start of the record."""
    record = load_record()
    baseline = clean_baseline(record=record)
    scale = float(json.loads(FROZEN_PATH.read_text())["changepoint"]["parameter"])
    years = (record.dates - record.dates[0]).astype(float) / 365.25
    frames = []
    for site in record.sites:
        farm = record.sites.index(site)
        signal = baseline.signal[farm]
        level = fitted_level(signal=signal, penalty_scale=scale)
        keep = np.isfinite(signal)
        frames.append(
            pl.DataFrame(
                {
                    "farm": f"Farm {site}",
                    "year": years[keep],
                    "signal": np.expm1(signal[keep]),
                    "level": np.expm1(level[keep]),
                    "low": baseline.low_months[keep]
                    if site == "A"
                    else np.zeros(keep.sum(), dtype=bool),
                }
            )
        )
    frame = pl.concat(frames)
    base = alt.Chart().encode(
        x=alt.X(
            "year:Q",
            title="Years since the start of the record",
            axis=alt.Axis(values=list(range(8))),
        ),
    )
    dots = base.mark_circle(size=6, opacity=0.5, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "signal:Q",
            title="Power relative to normal",
            scale=alt.Scale(domain=[-0.6, 0.6]),
            axis=alt.Axis(format="+.0%"),
        )
    )
    highlighted = (
        base.transform_filter("datum.low")  # ty: ignore[unresolved-attribute]
        .mark_circle(size=14, color=ocf.BRAND_ORANGE, aria=False)
        .encode(y="signal:Q")
    )
    line = base.mark_line(color="#222", interpolate="step-after", aria=False).encode(y="level:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(dots, highlighted, line)
        .properties(width=PLOT_WIDTH, height=75)
        .facet(
            row=alt.Row(
                "farm:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            ),
            data=frame,
        )
        .properties(
            title=_title(
                text="The daily signal wanders by 10% or more, so only a large step stands out",
                body=(
                    "Dots: one valid day's power as a share above or below the farm's normal, "
                    "after dividing by the other farms. "
                    "Orange dots: farm A's three low months. Black line: the change-point "
                    "fit's level. Farm E is not in the pool of other farms."
                ),
            )
        )
    )


def main() -> None:
    """Write the three figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1_detection_by_loss_size.svg", detection_chart()),
        ("figure_2_false_alarms.svg", false_alarm_chart()),
        ("figure_3_signal_and_fit.svg", signal_chart()),
    ):
        chart.save(DOCS_DIR / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
