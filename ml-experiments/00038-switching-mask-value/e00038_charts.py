"""Draw experiment 00038's figures as SVG from the saved tables.

    uv run python ml-experiments/00038-switching-mask-value/e00038_charts.py

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.
Substations may be named; there are no generator time series here.
"""

import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00038_base import GRID_START, SOURCE_DIR, WORK_DIR
from e00038_corrections import NEWEST_AGE, correction_series, detected_mask
from e00038_proxy import FIRST_ORIGIN, frame_to_intervals, load_series

CHART_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / "00038-switching-mask-value"
)
WIDTH: Final[int] = 600
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
ARM_COLOURS: Final[dict[str, str]] = {
    "d1_mask7": ocf.DATA_PURPLE,
    "d1fx_mask7": ocf.BRAND_ORANGE,
    "oracle_conf14": ocf.DATA_BLUE,
    "randomfx14": GREY,
}
ARM_LABELS: Final[dict[str, str]] = {
    "d1_mask7": "D1 mask, 7 days (as specified)",
    "d1fx_mask7": "D1 mask, 7 days (post hoc builder)",
    "oracle_conf14": "Logs mask, 14 days (ceiling)",
    "randomfx14": "Random mask, 14 days (placebo)",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _time(stamp: int) -> datetime:
    return (GRID_START + timedelta(minutes=30 * int(stamp))).replace(tzinfo=None)


def _save(chart: alt.TopLevelMixin, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(CHART_DIR / name)


def headline() -> None:
    """Per-event change in the threshold-weighted error, with the aggregate and its interval."""
    events = pl.read_parquet(WORK_DIR / "tier_a_events.parquet")
    arms = list(ARM_COLOURS)
    refs = {a: ("plain7" if a.endswith("7") else "plain14") for a in arms}
    rows = [
        {
            "event": f"{rank + 1:02d}: {row['sheet']} ({row['hours']:.0f} h)",
            "arm": ARM_LABELS[arm],
            "change": 100 * (row[f"tw_{arm}"] / row[f"tw_{refs[arm]}"] - 1),
        }
        for rank, row in enumerate(events.sort("share", descending=True).iter_rows(named=True))
        for arm in arms
        if row[f"tw_{refs[arm]}"] > 0
    ]
    arm_scale = alt.Scale(
        domain=[ARM_LABELS[a] for a in arms], range=[ARM_COLOURS[a] for a in arms]
    )
    colour = alt.Color(
        "arm:N",
        scale=arm_scale,
        legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400),
    )
    dots = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=60, opacity=0.9)
        .encode(
            x=alt.X(
                "change:Q",
                scale=alt.Scale(type="symlog", domain=[-100, 1000]),
                axis=alt.Axis(values=[-100, -10, 0, 10, 100, 1000]),
                title="Change in error, event window (%, log-like axis; negative is better)",
            ),
            y=alt.Y("event:N", title=None, sort=None),
            color=colour,
        )
        .properties(width=WIDTH - 200, height=300)
    )
    zero = alt.Chart(pl.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    contrasts = pl.read_parquet(WORK_DIR / "tier_a_contrasts.parquet").filter(
        pl.col("band") == "lead 4 to 48 h",
        pl.col("metric") == "tw",
        pl.col("view") == "out_of_event",
    )
    wanted = {
        "primary: d1_mask7 vs plain7": "d1_mask7",
        "post hoc: d1fx_mask7 vs plain7": "d1fx_mask7",
        "oracle_conf14 vs plain14": "oracle_conf14",
        "post hoc: randomfx14 vs plain14": "randomfx14",
    }
    agg = contrasts.filter(pl.col("contrast").is_in(list(wanted))).with_columns(
        arm=pl.col("contrast").replace_strict(wanted).replace_strict(ARM_LABELS)
    )
    base = alt.Chart(agg).encode(
        y=alt.Y(
            "arm:N", title=None, sort=[ARM_LABELS[a] for a in arms], axis=alt.Axis(labelLimit=300)
        ),
        color=alt.Color("arm:N", scale=arm_scale, legend=None),
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="All out-of-event half-hours: change in error (%)"), x2="hi:Q"
    ) + base.mark_circle(size=90).encode(x="change_pct:Q")  # ty: ignore[unresolved-attribute]
    chart = alt.vconcat(dots + zero, bars + zero, spacing=20).properties(
        title=_title(
            text="Masking detected events helped only after the interval builder was revised",
            subtitle=(
                "Threshold-weighted absolute error at each series' P99, proxy stack, forecasts 3.5 "
                "to 48 hours ahead, half-hours outside logged switching. Top: one dot per "
                "confirmed transfer of 24 hours or more (window: onset to 16 days after the end). "
                "Bottom: all rows; dot is the estimate, line the 95% interval from resampling "
                "whole months."
            ),
        )
    )
    _save(chart, "figure_1_headline.svg")


def _corrections_for_series(series: int) -> pl.DataFrame:
    """Daily corrections of series `series` under four arms, with the origin time."""
    y, f = load_series()[series]
    origins = np.arange(FIRST_ORIGIN, 70_800 + 48 * 40, 48)
    newest = origins - NEWEST_AGE
    arms = {
        "plain 14-day": (14, "none"),
        "D1 mask 14-day (as specified)": (14, "d1"),
        "D1 mask 14-day (post hoc builder)": (14, "d1fx"),
    }
    d1 = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1.parquet"))
    d1fx = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1fx.parquet"))
    out = []
    for name, (days, kind) in arms.items():
        mask = {
            "none": None,
            "d1": detected_mask(intervals=d1, series=series, window=8),
            "d1fx": detected_mask(intervals=d1fx, series=series, window=8),
        }[kind]
        values = correction_series(residual=y - f, newest=newest, window=days * 48, masked=mask)
        out.extend(
            {"time": _time(int(o)), "arm": name, "correction": 100 * v}
            for o, v in zip(origins, values, strict=True)
        )
    return pl.DataFrame(out)


def raw_and_corrections() -> None:
    """Series 17 and 18 around the 2025-12-30 closure: load, baseline, and the correction."""
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter(
        "primary", pl.col("sheet") == "17-18", pl.col("first_stamp") > 70_000
    )
    first, last = int(oracle["first_stamp"][0]), int(oracle["last_stamp_exclusive"][0])
    lo, hi = first - 15 * 48, last + 21 * 48
    panel = pl.read_parquet(SOURCE_DIR / "panel.parquet").filter(
        pl.col("series").is_in([17, 18]), pl.col("stamp").is_between(lo, hi)
    )
    long = panel.select(
        "series",
        pl.col("valid_time").dt.replace_time_zone(None).alias("time"),
        pl.col("obs_mw").alias("Observed"),
        pl.col("expected_mw").alias("Baseline (ERA5 weather)"),
    ).unpivot(index=["series", "time"], variable_name="line", value_name="mw")
    span = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            alt.InlineData(values=[{"a": _time(first).isoformat(), "b": _time(last).isoformat()}])
        )
        .mark_rect(color=GREY, opacity=0.25)
        .encode(x="a:T", x2="b:T")
    )
    panels = []
    for series in (17, 18):
        lines = (
            alt.Chart(long.filter(pl.col("series") == series))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b %Y", tickCount=5)),
                y=alt.Y("mw:Q", title=f"Primary {series} (MW)"),
                color=alt.Color(
                    "line:N",
                    scale=alt.Scale(
                        domain=["Observed", "Baseline (ERA5 weather)"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        panels.append((span + lines).properties(width=WIDTH, height=120))
    raw = alt.vconcat(*panels).properties(
        title=_title(
            text="Raw data: Primary substations 17 and 18 across the logged closure",
            subtitle=(
                "Half-hourly power (MW). The grey band is the logged closure of 2025-12-30 to "
                "2026-01-14; load moves from one substation to the other for its whole length."
            ),
        )
    )
    _save(raw, "figure_2_raw_17_18.svg")
    corrections = _corrections_for_series(18)
    corrections = corrections.filter(pl.col("time").is_between(_time(lo), _time(hi)))
    line = (
        alt.Chart(corrections)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b %Y", tickCount=5)),
            y=alt.Y("correction:Q", title="Correction (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[
                        "plain 14-day",
                        "D1 mask 14-day (as specified)",
                        "D1 mask 14-day (post hoc builder)",
                    ],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", columns=1, title=None),
            ),
        )
    )
    _save(
        (span + line).properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="The correction on substation 18 under three rules",
                subtitle=(
                    "Added to the baseline forecast at each 09:00 origin. The plain rule absorbs "
                    "the closure's shift and keeps it for two weeks after the closure ends; a mask "
                    "that "
                    "follows the closure avoids that, and a mask whose interval never ends holds "
                    "the pre-closure level too long. Grey: logged closure."
                ),
            ),
        ),
        "figure_3_corrections_18.svg",
    )


def monthly_gain() -> None:
    """Monthly gain in total threshold-weighted error: where the gain sits."""
    months = pl.read_parquet(WORK_DIR / "tier_a_months.parquet")
    gain = months.select(
        "month",
        (pl.col("plain7") - pl.col("d1fx_mask7")).alias("Post hoc D1 mask, 7-day"),
        (pl.col("plain14") - pl.col("oracle_conf14")).alias("Logs mask, 14-day"),
    ).unpivot(index="month", variable_name="arm", value_name="gain")
    chart = (
        alt.Chart(gain)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=8)),
            y=alt.Y("gain:Q", title="Error removed (sum of half-hour errors; above 0 is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["Post hoc D1 mask, 7-day", "Logs mask, 14-day"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="arm:N",
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="The gain sits in one month",
                subtitle=(
                    "Total threshold-weighted error removed per calendar month against the plain "
                    "correction of the same window, in points of capacity summed over half-hours, "
                    "outside logged switching. January 2026 holds the 2025-12-30 closure."
                ),
            ),
        )
    )
    _save(chart, "figure_4_monthly_gain.svg")


def interval_lengths() -> None:
    """Detected interval lengths, split by whether they overlap a logged closure."""
    table = pl.read_parquet(WORK_DIR / "q3_table_d1fx.parquet").with_columns(
        overlap=pl.when("overlaps_logged")
        .then(pl.lit("overlaps a logged closure"))
        .otherwise(pl.lit("overlaps no logged closure"))
    )
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "hours:Q",
                bin=alt.Bin(extent=[0, 1500], step=100),
                title="Detected interval length (hours)",
            ),
            y=alt.Y("count():Q", title="Intervals"),
            color=alt.Color(
                "overlap:N",
                scale=alt.Scale(
                    domain=["overlaps a logged closure", "overlaps no logged closure"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Most long detected intervals overlap no logged closure",
                subtitle=(
                    "D1 intervals from the post hoc builder, alarm and partner series together, "
                    "2022-01 to 2026-06. The longest bin holds intervals that reached the 60-day "
                    "cap."
                ),
            ),
        )
    )
    _save(chart, "figure_5_interval_lengths.svg")


def injection() -> None:
    """Recall and end error of injected transfers."""
    frame = pl.read_parquet(WORK_DIR / "injection_results.parquet").with_columns(
        builder=pl.when(pl.col("variant") == "spec")
        .then(pl.lit("as specified"))
        .otherwise(pl.lit("post hoc"))
    )
    recall = (
        frame.group_by("builder", "phi", "n_receivers")
        .agg(recall=pl.col("hit").mean())
        .with_columns(phi_label=(pl.col("phi") * 100).cast(pl.Int64).cast(pl.String) + "%")
    )
    colours = alt.Scale(
        domain=["as specified", "post hoc"], range=[ocf.DATA_PURPLE, ocf.BRAND_ORANGE]
    )
    recall_chart = (
        alt.Chart(recall.filter(pl.col("builder") == "as specified"))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "phi_label:N", title=None, sort=["5%", "15%", "30%"], axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("recall:Q", title="Recall (share found)", scale=alt.Scale(domain=[0, 1])),
            column=alt.Column(
                "n_receivers:N", title="Receivers; x: moved share of the donor's own power"
            ),
        )
        .properties(
            width=150,
            height=140,
            title=_title(
                text="Injected transfers: recall rises with the moved share",
                subtitle=(
                    "600 synthetic transfers in quiet periods; a transfer is found when a D1 "
                    "alarm on the donor or a receiver lies within 2 hours of the onset."
                ),
            ),
        )
    )
    _save(recall_chart, "figure_6_injection_recall.svg")
    hits = frame.filter("hit").drop_nulls("end_error_hours")
    ends = (
        alt.Chart(hits)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X("hours:Q", scale=alt.Scale(type="log"), title="True event length (hours)"),
            y=alt.Y(
                "end_error_hours:Q",
                scale=alt.Scale(type="symlog"),
                title="Estimated minus true end (hours)",
            ),
            color=alt.Color(
                "builder:N", scale=colours, legend=alt.Legend(orient="top", title="Builder")
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Injected transfers: error in the estimated end",
                subtitle="One dot per found transfer; zero is a perfect end.",
            ),
        )
    )
    _save(ends, "figure_7_injection_end_error.svg")


def tier_b() -> None:
    """The real stack's contrasts with intervals."""
    table = pl.read_parquet(WORK_DIR / "tier_b_contrasts.parquet").filter(
        pl.col("view") == "out_of_event", pl.col("metric") == "tw"
    )
    chart_base = alt.Chart(table).encode(y=alt.Y("contrast:N", title=None, sort=None))
    chart = (
        chart_base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Change in threshold-weighted score (%; negative is better)"),
            x2="hi:Q",
        )
        + chart_base.mark_circle(size=80, color=ocf.BRAND_ORANGE).encode(x="change_pct:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH - 220,
        height=220,
        title=_title(
            text="The real stack agrees with the proxy stack",
            subtitle=(
                "Experiment 00019's stack, three seeds pooled, half-hours outside logged "
                "switching, "
                "July 2025 to March 2026. Dot: estimate. Line: 95% interval from resampling whole "
                "months."
            ),
        ),
    )
    _save(chart, "figure_8_tier_b.svg")


def event_panels() -> None:
    """Residuals of the donor and receiver around four long confirmed events, with D1 intervals."""
    oracle = (
        pl.read_parquet(SOURCE_DIR / "oracle_real.parquet")
        .filter("primary", "confirmed", pl.col("hours") >= 24)
        .sort("share", descending=True)
        .head(6)
    )
    panel = pl.read_parquet(SOURCE_DIR / "panel.parquet")
    intervals = pl.read_parquet(WORK_DIR / "intervals_d1fx.parquet")
    charts = []
    for row in oracle.iter_rows(named=True):
        sides = [s for s in (row["series_a"], row["series_b"]) if s is not None]
        lo = row["first_stamp"] - 4 * 48
        hi = min(row["last_stamp_exclusive"], row["first_stamp"] + 40 * 48) + 4 * 48
        frame = panel.filter(pl.col("series").is_in(sides), pl.col("stamp").is_between(lo, hi))
        frame = frame.with_columns(
            time=pl.col("valid_time").dt.replace_time_zone(None),
            label=pl.format("series {}", pl.col("series")),
        )
        span = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                alt.InlineData(
                    values=[
                        {
                            "a": _time(row["first_stamp"]).isoformat(),
                            "b": _time(min(row["last_stamp_exclusive"], hi)).isoformat(),
                        }
                    ]
                )
            )
            .mark_rect(color=GREY, opacity=0.25)
            .encode(x="a:T", x2="b:T")
        )
        lines = (
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b %Y", tickCount=5)),
                y=alt.Y("res_mw:Q", title="Residual (MW)"),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        domain=[f"series {s}" for s in sides],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE][: len(sides)],
                    ),
                    legend=None,
                ),
            )
        )
        detected = intervals.filter(
            pl.col("series").is_in(sides), pl.col("onset") < hi, pl.col("end") > lo
        )
        layers = [span, lines]
        if detected.height:
            bars = (
                alt.Chart(  # ty: ignore[unresolved-attribute]
                    alt.InlineData(
                        values=[
                            {
                                "a": _time(r["onset"]).isoformat(),
                                "b": _time(min(r["end"], hi)).isoformat(),
                            }
                            for r in detected.iter_rows(named=True)
                        ]
                    )
                )
                .mark_rect(color=ocf.BRAND_ORANGE, opacity=0.18, height=8, y=0)
                .encode(x="a:T", x2="b:T")
            )
            layers.append(bars)
        title = (
            f"{row['sheet']}, {row['hours']:.0f} h, "
            f"{100 * row['share']:.0f}% of the receiver's load "
            f"(blue: series {sides[0]}"
            + (f", orange: series {sides[1]})" if len(sides) > 1 else ")")
        )
        charts.append(alt.layer(*layers).properties(width=WIDTH - 80, height=105, title=title))
    chart = alt.vconcat(*charts, spacing=14).properties(
        title=_title(
            text="Raw data: the six largest confirmed transfers of 24 hours or more",
            subtitle=(
                "Residual of the baseline forecast (MW). Grey band: logged closure. Orange strip "
                "at the top: intervals the post hoc builder detected on these substations."
            ),
        )
    )
    _save(chart, "figure_9_raw_events.svg")


def main() -> None:
    """Draw every figure."""
    headline()
    raw_and_corrections()
    monthly_gain()
    interval_lengths()
    injection()
    tier_b()
    event_panels()


if __name__ == "__main__":
    main()
