# ruff: noqa: D103, E501
"""Draw experiment 296's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations (labels D5 to D16) and simulated shifts, not NGED generators.

    uv run --no-sync python e00296_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00296_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00296-lsc-benefit-by-event-class-long-deep-versus-small-shifts"
)
RESULTS: Final[Path] = WORK_DIR / "results"
WIDTH: Final[int] = 520
METRIC_LABELS: Final[dict[str, str]] = {
    "MASK|tw95_nra": "Upper-tail score (twCRPS) against the normal arrangement, readings set aside",
    "ADD|cost0.5": "Curtailment cost against observed power, megawatts added back",
}
CLASS_LABELS: Final[dict[str, str]] = {
    "LD": "Long and deep: at least 2 days and 10% of load",
    "SHORT_OR_SMALL": "Short or small: under 12 hours or under 10% of load",
}
SCENARIOS: Final[tuple[str, ...]] = ("low", "central", "high")
SCENARIO_LABELS: Final[dict[str, str]] = {
    "low": "Few small shifts",
    "central": "Central",
    "high": "Many small shifts",
}
BIN_ORDER: Final[list[str]] = ["2-6 h", "6-24 h", "1-3 d", "3-7 d", "7-30 d", "over 30 d"]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(OUT_DIR / f"{name}.svg")


def headline(*, numbers: dict[str, Any]) -> None:
    rows = []
    for scenario in SCENARIOS:
        for metric, label in METRIC_LABELS.items():
            arms, name = metric.split("|")
            res = numbers["shares"][f"{scenario}|{arms}|G5 near|{name}"]
            for cls, cls_label in CLASS_LABELS.items():
                p, lo, hi = res[cls]
                rows.append(
                    {
                        "scenario": SCENARIO_LABELS[scenario],
                        "metric": label,
                        "class": cls_label,
                        "est": 100 * p,
                        "lo": 100 * lo,
                        "hi": 100 * hi,
                    }
                )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "class:N",
        scale=alt.Scale(
            domain=list(CLASS_LABELS.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", direction="vertical", title=None, labelLimit=600),
    )
    panels = []
    for label in METRIC_LABELS.values():
        sub = frame.filter(pl.col("metric") == label)
        base = alt.Chart(sub).encode(
            y=alt.Y(
                "scenario:N",
                sort=[SCENARIO_LABELS[s] for s in SCENARIOS],
                title=None,
                axis=alt.Axis(labelLimit=200),
            ),
            yOffset=alt.YOffset("class:N"),
            color=colour,
        )
        rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Share of the benefit of knowing every shift (%)",
                scale=alt.Scale(domain=[0, 105]),
            ),
            x2="hi:Q",
        )
        dots = base.mark_point(filled=True, size=70).encode(x="est:Q")  # ty: ignore[unresolved-attribute]
        ref = (
            alt.Chart(pl.DataFrame({"x": [70.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="grey")
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(ref, rules, dots).properties(
                width=WIDTH, height=110, title=textwrap.wrap(label, 80)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text="Long and deep shifts carry most of the benefit of knowing every shift",
            line="Dot: estimate. Line: 95% interval from resampling whole shifts within each class of shift. Rows within 7 days of a shift. Dashed rule: 70%. The three scenarios differ in how often small and short shifts occur.",
        )
    )
    save(chart=chart, name="figure_1_share_of_benefit_by_class")


def ladder(*, numbers: dict[str, Any]) -> None:
    rows = []
    for metric, label in METRIC_LABELS.items():
        arms, name = metric.split("|")
        res = numbers["ladder"][f"central|{arms}|{name}"]
        for hours in (2, 6, 12, 24, 48, 168, 720):
            p, lo, hi = res[f"min {hours:g} h, 10%"]
            rows.append(
                {
                    "metric": label,
                    "axis": "Minimum duration (hours; shifts of at least 10% of load)",
                    "x": float(hours),
                    "est": 100 * p,
                    "lo": 100 * lo,
                    "hi": 100 * hi,
                }
            )
        for depth in (5, 10, 25, 50):
            p, lo, hi = res[f"min 48 h, {depth}%"]
            rows.append(
                {
                    "metric": label,
                    "axis": "Minimum depth (% of load; shifts of at least 2 days)",
                    "x": float(depth),
                    "est": 100 * p,
                    "lo": 100 * lo,
                    "hi": 100 * hi,
                }
            )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "metric:N",
        scale=alt.Scale(
            domain=list(METRIC_LABELS.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", direction="vertical", title=None, labelLimit=600),
    )

    def one(axis: str, log: bool) -> alt.LayerChart:
        sub = frame.filter(pl.col("axis") == axis)
        base = alt.Chart(sub).encode(
            x=alt.X(
                "x:Q",
                title=axis,
                scale=alt.Scale(type="log" if log else "linear"),
                axis=alt.Axis(values=[2, 6, 12, 24, 48, 168, 720] if log else [5, 10, 25, 50]),
            ),
            color=colour,
        )
        band = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "lo:Q", title="Share of benefit captured (%)", scale=alt.Scale(domain=[0, 100])
            ),
            y2="hi:Q",
        )
        line = base.mark_line(point=True).encode(y="est:Q")  # ty: ignore[unresolved-attribute]
        return (band + line).properties(width=WIDTH, height=170)

    chart = alt.vconcat(
        one("Minimum duration (hours; shifts of at least 10% of load)", True),
        one("Minimum depth (% of load; shifts of at least 2 days)", False),
    ).properties(
        title=_title(
            text="Lowering the minimum duration adds more than lowering the minimum depth",
            line="Share of the benefit of knowing every shift that an oracle knowing only shifts above the minimum captures, central scenario, rows within 7 days of a shift. Line: estimate. Band: 95% interval from resampling whole shifts.",
        )
    )
    save(chart=chart.resolve_scale(color="shared"), name="figure_2_ladder")


def heat(*, cells: pl.DataFrame) -> None:
    sub = cells.filter(pl.col("shape") == "all")
    panels = []
    for metric, label in METRIC_LABELS.items():
        arms, name = metric.split("|")
        data = sub.filter((pl.col("arms") == arms) & (pl.col("metric") == name)).with_columns(
            depth_label=(pl.col("depth") * 100).cast(pl.Int64).cast(pl.String) + "%"
        )
        base = alt.Chart(data).encode(
            x=alt.X("duration_bin:N", sort=BIN_ORDER, title="Duration of the shift"),
            y=alt.Y(
                "depth_label:N", sort=["50%", "25%", "10%", "5%"], title="Depth (share of load)"
            ),
        )
        rect = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "share_of_ceiling_pct:Q",
                title="Share of benefit (%)",
                scale=alt.Scale(scheme="oranges"),
                legend=alt.Legend(orient="top"),
            )
        )
        text = base.mark_text(fontSize=10).encode(  # ty: ignore[unresolved-attribute]
            text=alt.Text("share_of_ceiling_pct:Q", format=".0f")
        )
        panels.append((rect + text).properties(width=WIDTH - 60, height=120, title=label))
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text="Most of the benefit sits in shifts of 3 to 30 days at 25% of load or more",
            line="Each cell is the share (%) of the total benefit of knowing every shift that comes from shifts in that duration and depth, central scenario, rows within 7 days of a shift.",
        )
    )
    save(chart=chart, name="figure_3_benefit_by_cell")


def frequency(*, numbers: dict[str, Any]) -> None:
    rows = []
    for scenario in SCENARIOS:
        for key, value in numbers[f"frequency_{scenario}"].items():
            b, d = key.rsplit("|", 1)
            rows.append(
                {
                    "scenario": SCENARIO_LABELS[scenario],
                    "duration_bin": b,
                    "depth": f"{round(float(d) * 100)}%",
                    "per_year": 1000 * value,
                }
            )
    frame = pl.DataFrame(rows)
    panels = []
    for scenario in SCENARIOS:
        sub = frame.filter(pl.col("scenario") == SCENARIO_LABELS[scenario])
        base = alt.Chart(sub).encode(
            x=alt.X("duration_bin:N", sort=BIN_ORDER, title="Duration of the shift"),
            y=alt.Y("depth:N", sort=["50%", "25%", "10%", "5%"], title="Depth (share of load)"),
        )
        rect = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "per_year:Q",
                title="Shifts per 1,000 series-years",
                scale=alt.Scale(scheme="blues", domain=[0, 8000], clamp=True),
                legend=alt.Legend(orient="top"),
            )
        )
        text = base.mark_text(fontSize=10).encode(text=alt.Text("per_year:Q", format=".0f"))  # ty: ignore[unresolved-attribute]
        panels.append(
            (rect + text).properties(width=WIDTH - 60, height=100, title=SCENARIO_LABELS[scenario])
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text="How often shifts of each class occur in the three scenarios",
            line="Shifts per 1,000 series-years. Durations come from the 537 logged operations; the depth mix and the extra count of short shifts are assumptions of each scenario.",
        )
    )
    save(chart=chart, name="figure_4_frequency")


def by_group(*, numbers: dict[str, Any]) -> None:
    rows = []
    for group in ("G2 before alarm", "G3 during", "G4 after", "G5 near"):
        for metric, label in METRIC_LABELS.items():
            arms, name = metric.split("|")
            key = f"central|{arms}|{group}|{name}"
            if key not in numbers["shares"]:
                continue
            p, lo, hi = numbers["shares"][key]["LD"]
            rows.append(
                {
                    "group": group[3:].capitalize(),
                    "metric": label,
                    "est": 100 * p,
                    "lo": 100 * max(lo, -0.5),
                    "hi": 100 * min(hi, 2.0),
                }
            )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "metric:N",
        scale=alt.Scale(
            domain=list(METRIC_LABELS.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", direction="vertical", title=None, labelLimit=600),
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("group:N", title=None, sort=["Before alarm", "During", "After", "Near"]),
        yOffset="metric:N",
        color=colour,
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Share captured by the long-and-deep oracle (%)"), x2="hi:Q"
    )
    dots = base.mark_point(filled=True, size=70).encode(x="est:Q")  # ty: ignore[unresolved-attribute]
    chart = (rules + dots).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The long-and-deep share by phase of the shift",
            line="Central scenario. Dot: estimate. Line: 95% interval (clipped to -50% to 200%). Near: within 7 days of a shift.",
        ),
    )
    save(chart=chart, name="figure_5_share_by_phase")


def main() -> None:
    numbers = json.loads((RESULTS / "report_numbers.json").read_text())
    cells = pl.read_parquet(RESULTS / "cells.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headline(numbers=numbers)
    ladder(numbers=numbers)
    heat(cells=cells)
    frequency(numbers=numbers)
    by_group(numbers=numbers)


if __name__ == "__main__":
    main()
