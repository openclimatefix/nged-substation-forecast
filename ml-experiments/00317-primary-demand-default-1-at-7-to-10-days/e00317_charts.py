# ruff: noqa: E501
"""Draw the figures of experiment 317 from `work/report/*.json` and `scored_rows.parquet`.

uv run --no-sync python ml-experiments/00317-*/e00317_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00317_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
GT: Final = "Lower quantile score than the benchmark (%; larger is better)"
YT: Final = "Mean quantile score (points of capacity; smaller is better)"
BANDS: Final[tuple[str, ...]] = tuple(f"{a}-{a + 24} h" for a in range(0, 240, 24))
CENTRES: Final[dict[str, int]] = {b: i * 24 + 12 for i, b in enumerate(BANDS)}
COLOURS: Final[dict[str, str]] = {
    "Convex chain (A)": ocf.DATA_BLUE,
    "A + network head + solar summary (to 168 h)": ocf.DATA_PURPLE,
    "Manual heuristic": ocf.BRAND_ORANGE,
    "Climatology": "#7A7A7A",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _rule168() -> Any:
    frame = pl.DataFrame({"x": [168], "t": ["Earlier experiments stop at 168 h"]})
    line = _chart(frame).mark_rule(strokeDash=[4, 3], color="#444444").encode(x="x:Q")
    text = (
        _chart(frame)
        .mark_text(align="left", dx=4, dy=-4, baseline="bottom", fontSize=10, color="#444444")
        .encode(x="x:Q", y=alt.value(0), text="t:N")
    )
    return line + text


def headline(*, report: dict) -> None:
    """Quantile score against lead, three arms with intervals."""
    rows = []
    for b in BANDS:
        e = report["by_band"][b]
        for label, key in (
            ("Convex chain (A)", "A"),
            ("Manual heuristic", "manual_heuristic"),
            ("Climatology", "climatology"),
        ):
            rows.append(
                {"arm": label, "lead": CENTRES[b], **{k: e[key][k] for k in ("est", "lo", "hi")}}
            )
        if "A+H+S_on_251_rows" in e:
            v = e["A+H+S_on_251_rows"]
            rows.append(
                {
                    "arm": "A + network head + solar summary (to 168 h)",
                    "lead": CENTRES[b],
                    "est": v["est"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                }
            )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; each point is a 24-hour band, plotted at its centre)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    y = alt.Y("est:Q", title=YT, scale=alt.Scale(zero=False))
    base = _chart(data)
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo:Q", title=YT), y2="hi:Q", color=colour
        )
        + base.mark_line(strokeWidth=2).encode(x=x, y=y, color=colour)
        + base.mark_point(filled=True, size=40).encode(x=x, y=y, color=colour)
        + _rule168()
    ).properties(
        width=WIDTH,
        height=300,
        title=_title(
            title="The convex model stays better than climatology to 240 hours; against the manual heuristic the gap narrows to within the interval",
            subtitle="Dot: mean quantile score over the six Primary substations, scored year 2025-07-02 to 2026-06-30. Bar: 95% interval from resampling the 12 months of forecast date. Rows with a reading outside the hand-filtered events.",
        ),
    )
    _save(chart, name="headline_score_by_lead")


def gain(*, report: dict) -> None:
    """Relative gain of A over each benchmark by lead band."""
    rows = []
    for b in BANDS:
        e = report["by_band"][b]
        for label, key in (
            ("Manual heuristic", "A_vs_manual_heuristic"),
            ("Climatology", "A_vs_climatology"),
        ):
            g = e[key]
            rows.append(
                {
                    "benchmark": label,
                    "lead": CENTRES[b],
                    "gain": 100 * g["gain"],
                    "lo": 100 * g["gain_lo"],
                    "hi": 100 * g["gain_hi"],
                }
            )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "benchmark:N",
        scale=alt.Scale(
            domain=["Manual heuristic", "Climatology"], range=[ocf.BRAND_ORANGE, "#7A7A7A"]
        ),
        legend=alt.Legend(title="Gain over", orient="top"),
    )
    x = alt.X(
        "lead:Q",
        title="Lead time (hours)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    y = alt.Y("gain:Q", title=GT)
    zero = (
        _chart(pl.DataFrame({"y": [0.0, 5.0]}))
        .mark_rule(color="#000000", strokeDash=[2, 2])
        .encode(y="y:Q")
    )
    base = _chart(data)
    chart = (
        zero
        + base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo:Q", title=GT), y2="hi:Q", color=colour
        )
        + base.mark_line().encode(x=x, y=y, color=colour)
        + base.mark_point(filled=True, size=40).encode(x=x, y=y, color=colour)
        + _rule168()
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="Gain over climatology stays above zero to 240 hours; gain over the manual heuristic reaches zero inside its interval from 168 hours",
            subtitle="Dot: 1 minus the convex chain's quantile score divided by the benchmark's. Bar: 95% interval, 12 monthly blocks. Dashed rules: 0% (same as the benchmark) and 5% (the planned threshold).",
        ),
    )
    _save(chart, name="gain_by_lead")


def tails(*, report: dict) -> None:
    """Coverage of the 90% interval and pinball loss at q01 and q99 by lead."""
    rows = []
    for b in BANDS:
        t = report["by_band"][b]["tails"]["A"]
        rows += [
            {
                "measure": "Share of observations inside the 5% to 95% band (nominal 90%)",
                "lead": CENTRES[b],
                "value": 100 * t["interval90_cov"],
            },
            {
                "measure": "Share below the 1% quantile (nominal 1%)",
                "lead": CENTRES[b],
                "value": 100 * t["q01_cov"],
            },
            {
                "measure": "Share below the 99% quantile (nominal 99%)",
                "lead": CENTRES[b],
                "value": 100 * t["q99_cov"],
            },
        ]
    data = pl.DataFrame(rows)
    chart = (
        _chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "lead:Q",
                title="Lead time (hours)",
                scale=alt.Scale(domain=[0, 240]),
                axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
            ),
            y=alt.Y("value:Q", title="Share of observations (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top", columns=1),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=_title(
                title="Band coverage barely changes from day 1 to day 10",
                subtitle="Convex chain (A), six Primary substations, scored year. The 5% to 95% band holds about 87 to 89% of observations at every lead; the nominal share is 90%.",
            ),
        )
    )
    _save(chart, name="tail_calibration_by_lead")


def groups(*, report: dict) -> None:
    """Gain over each benchmark at 168 to 240 hours by protected group."""
    rows = []
    for g in ("all rows", "winter (Nov-Mar)", "cold winter nights", "daylight", "sunny daylight"):
        e = report["by_group"][g]["168-240 h"]
        for label, key in (
            ("Manual heuristic", "A_vs_manual_heuristic"),
            ("Climatology", "A_vs_climatology"),
        ):
            v = e[key]
            rows.append(
                {
                    "group": f"{g} (n={e['n']:,})",
                    "benchmark": label,
                    "gain": 100 * v["gain"],
                    "lo": 100 * v["gain_lo"],
                    "hi": 100 * v["gain_hi"],
                }
            )
    data = pl.DataFrame(rows)
    order = [r["group"] for r in rows[::2]]
    colour = alt.Color(
        "benchmark:N",
        scale=alt.Scale(
            domain=["Manual heuristic", "Climatology"], range=[ocf.BRAND_ORANGE, "#7A7A7A"]
        ),
        legend=alt.Legend(title="Gain over", orient="top"),
    )
    y = alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    base = _chart(data)
    zero = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#000000", strokeDash=[2, 2])
        .encode(x="x:Q")
    )
    chart = (
        zero
        + base.mark_errorbar(thickness=1.5).encode(
            y=y,
            x=alt.X(
                "lo:Q",
                title="Lower quantile score than the benchmark, 168 to 240 hours (%; larger is better)",
            ),
            x2="hi:Q",
            color=colour,
            yOffset="benchmark:N",
        )
        + base.mark_point(filled=True, size=50).encode(
            y=y, x="gain:Q", color=colour, yOffset="benchmark:N"
        )
    ).properties(
        width=WIDTH - 250,
        height=230,
        title=_title(
            title="The gain is largest on cold winter nights and negative on sunny daylight half-hours",
            subtitle="Dot: gain of the convex chain (A). Bar: 95% interval, 12 monthly blocks. Groups are defined from realised temperature and sunshine, so they overstate the gain that a forecaster could know in advance.",
        ),
    )
    _save(chart, name="gain_by_group_168_240")


def example() -> None:
    """One week of day-8 to day-10 forecasts at one Primary substation in winter."""
    rows = pl.read_parquet(common.REPORT_DIR / "scored_rows.parquet").filter(
        (pl.col("site") == "D10") & (pl.col("lead_hours") > 192) & (pl.col("lead_hours") <= 216)
    )
    start = pl.datetime(2025, 12, 8, time_zone="UTC")
    week = rows.filter(
        (pl.col("valid_time") >= start) & (pl.col("valid_time") < start + pl.duration(days=7))
    ).sort("valid_time")
    week = week.with_columns(hours=(pl.col("valid_time") - start).dt.total_minutes() / 60.0)
    base = _chart(week)
    x = alt.X(
        "hours:Q",
        title="Time since Monday 8 December 2025, 00:00 UTC (hours; each forecast was issued 8 to 9 days earlier)",
        axis=alt.Axis(values=list(range(0, 169, 24))),
    )
    band = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(
        x=x, y=alt.Y("A_q05:Q", title="Demand (points of capacity)"), y2="A_q95:Q"
    )
    lines = (
        base.mark_line(color="#000000", strokeWidth=1.2).encode(x=x, y="actual:Q")
        + base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5).encode(x=x, y="A_q50:Q")
        + base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.5).encode(x=x, y="M_q50:Q")
        + base.mark_line(color="#7A7A7A", strokeWidth=1.5).encode(x=x, y="C_q50:Q")
    )
    legend = (
        _chart(
            pl.DataFrame(
                {
                    "x": [0.0, 0.0, 0.0, 0.0],
                    "y": [0.0, 0.0, 0.0, 0.0],
                    "k": [
                        "Observed (black)",
                        "Convex chain median and 5% to 95% band (blue)",
                        "Manual heuristic median (orange)",
                        "Climatology median (grey)",
                    ],
                }
            )
        )
        .mark_point(opacity=0)
        .encode(
            color=alt.Color(
                "k:N",
                scale=alt.Scale(
                    domain=[
                        "Observed (black)",
                        "Convex chain median and 5% to 95% band (blue)",
                        "Manual heuristic median (orange)",
                        "Climatology median (grey)",
                    ],
                    range=["#000000", ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#7A7A7A"],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420),
            )
        )
    )
    chart = (band + lines + legend).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="One winter week of forecasts issued 8 to 9 days ahead",
            subtitle="One Primary substation (label D10), forecasts of lead 192 to 216 hours, in the week of 8 December 2025. Straight lines cross a gap in the readings. The week was moved once to avoid a longer gap, not chosen for its fit.",
        ),
    )
    _save(chart, name="example_week_day_8_to_9")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    headline(report=report)
    gain(report=report)
    tails(report=report)
    groups(report=report)
    example()


if __name__ == "__main__":
    main()
