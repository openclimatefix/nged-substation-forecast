"""Draw the figures of experiment 123 as SVG into the docs folder.

Only Primary substations appear, under their D-labels. The numbers come from
`work/numbers_<stage>.json` (written by `e00123_report.py`) and the gate's stored draws, so a
figure and the page quote the same numbers. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00123-*/e00123_charts.py headline
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00123_common as common
import e00123_estimate as estimate
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCS_DIR: Final[Path] = (
    REPO_ROOT / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = REPO_ROOT / ".claude" / "worktrees" / "e7_png"
WIDTH: Final[int] = 460
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 92
GREY: Final[str] = "#808080"
DECISION_COLOUR: Final[str] = ocf.DATA_BLUE
EXTENSION_COLOUR: Final[str] = ocf.BRAND_ORANGE
POOLED_COLOUR: Final[str] = ocf.DATA_PURPLE
WINTER_COLOURS: Final[dict[int, str]] = {
    2021: ocf.DATA_BLUE_LIGHT,
    2022: ocf.DATA_SKY,
    2023: ocf.DATA_BLUE,
    2024: ocf.DATA_PURPLE,
    2025: ocf.BRAND_ORANGE,
}
SITE_ORDER: Final[list[str]] = [*common.DECISION_SERIES, *common.EXTENSION_SERIES]
CLOCK_LABEL: Final[str] = (
    "(floor(((1320 + 30 * datum.value) % 1440) / 60) < 10 ? '0' : '') + "
    "floor(((1320 + 30 * datum.value) % 1440) / 60) + ':' + "
    "(((1320 + 30 * datum.value) % 60) == 0 ? '00' : '30')"
)
SET_NAMES: Final[dict[str, str]] = {
    "six decision series (headline)": "Six decision series (headline)",
    "D5, D10, D16 (selected for showing the step)": "D5, D10, D16 (selected)",
    "D9, D11, D14 (not selected)": "D9, D11, D14 (not selected)",
    "E7-X (eight further Primaries)": "E7-X (eight series)",
    "all 14 series": "All 14 series",
}
STEP_AXIS: Final[str] = "Night step at a daily mean of 3 C (points; larger is a bigger rise)"


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _group(site: str) -> str:
    return "decision series" if site in common.DECISION_SERIES else "E7-X series"


def _kind_colour(*, pooled: bool = True) -> alt.Color:
    domain = ["pooled", "decision series", "E7-X series"]
    colours = [POOLED_COLOUR, DECISION_COLOUR, EXTENSION_COLOUR]
    return alt.Color(
        "kind:N",
        scale=alt.Scale(domain=domain[0 if pooled else 1 :], range=colours[0 if pooled else 1 :]),
        legend=alt.Legend(title=None, orient="top"),
    )


def _zero_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[3, 3])
        .encode(x="x:Q")
    )


def figure_1_headline(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """The pooled steps, then every series' step with its interval."""
    pooled = numbers["pooled"]
    pooled_rows = [
        {
            "row": f"Pooled: {name}",
            "kind": "pooled",
            "value": p["mean"],
            "low": p["low_hk"],
            "high": p["high_hk"],
        }
        for name, p in (
            ("six decision series", pooled[0]),
            ("E7-X (eight series)", pooled[3]),
            ("all 14 series", pooled[4]),
        )
    ]
    series_rows = [
        {
            "row": r["site"],
            "kind": _group(r["site"]),
            "value": r["step"],
            "low": r["step_low"],
            "high": r["step_high"],
        }
        for r in numbers["unpooled"]
    ]
    frame = pl.DataFrame([*pooled_rows, *series_rows])
    order = [r["row"] for r in pooled_rows] + SITE_ORDER
    base = alt.Chart(frame).encode(y=alt.Y("row:N", sort=order, title=None), color=_kind_colour())
    rule = base.mark_rule(strokeWidth=3).encode(x=alt.X("low:Q", title=STEP_AXIS), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70, opacity=1).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    return (_zero_rule() + rule + dot).properties(
        width=WIDTH,
        height=22 * len(order),
        title=_title(
            text="Demand steps up at about 01:00 on cold winter nights; pooled over the six "
            "decision "
            f"series the step is {pooled[0]['mean']:.0f} points",
            subtitle="Dot: estimate for winters 2021-22 to 2024-25. Line: 90% interval. The six "
            "decision substations were set before any fit as the ones the conclusion rests on; "
            "the eight further substations (E7-X) were added as a check. Pooled rows average the "
            "substations' steps, weighted by how precisely each is known, with a Hartung-Knapp "
            "interval. Other rows: resampling 14-day blocks of nights. One point is 1% of the "
            "substation's 99th percentile of demand. The step is the rise from 22:00-23:30 to "
            "00:30-03:00 on a night with daily mean 3 C, minus the same rise in summer.",
        ),
    )


def _nights_for(*, site: str, numbers: dict[str, Any]) -> estimate.Nights:
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    usable = [
        w
        for w in numbers["winters"]
        if w
        in {
            int(x)
            for x in table.filter(
                (pl.col("site") == site) & pl.col("d").is_not_null() & (pl.col("season") == "gmt")
            )["winter"]
            .unique()
            .to_list()
            if x is not None
        }
    ]
    return estimate.load_nights(
        table=table,
        site=site,
        capacity_mw=infos[site].effective_capacity_mw,
        winters=tuple(usable),
        summer_years=common.SUMMER_YEARS,
    )


def figure_2_raw(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """Night difference against cold, for the three series that prompted the study."""
    panels = []
    winters = numbers["winters"]
    for site in common.SELECTED_SERIES:
        nights = _nights_for(site=site, numbers=numbers)
        fitted = next(r for r in numbers["unpooled"] if r["site"] == site)
        summer = float(nights.summer_d.mean())
        grid = np.linspace(0.0, float(nights.cold.max()), 20)
        points = pl.DataFrame(
            {
                "cold": nights.cold,
                "d": nights.d,
                "winter": [f"{w}-{str(w + 1)[2:]}" for w in nights.winter],
            }
        )
        line = pl.DataFrame(
            {"cold": grid, "line": fitted["flat"] + summer + fitted["slope"] * grid}
        )
        dots = (
            alt.Chart(points)  # ty: ignore[unresolved-attribute]
            .mark_circle(size=14, opacity=0.55)
            .encode(
                x=alt.X(
                    "cold:Q", title="Cold: degrees below 15.5 C of the day's mean (3 C is 12.5)"
                ),
                y=alt.Y("d:Q", title="Night difference (points)"),
                color=alt.Color(
                    "winter:N",
                    scale=alt.Scale(
                        domain=[f"{w}-{str(w + 1)[2:]}" for w in winters],
                        range=[WINTER_COLOURS[w] for w in winters],
                    ),
                    legend=alt.Legend(title="Winter", orient="top")
                    if site == common.SELECTED_SERIES[0]
                    else None,
                ),
            )
        )
        fit_line = (
            alt.Chart(line)  # ty: ignore[unresolved-attribute]
            .mark_line(color="#333333", strokeWidth=2.5)
            .encode(x="cold:Q", y="line:Q")
        )
        summer_rule = (
            alt.Chart(pl.DataFrame({"s": [summer]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(y="s:Q")
        )
        panels.append(
            (dots + fit_line + summer_rule).properties(width=WIDTH, height=140, title=site)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="On winter nights demand rises after midnight where in summer it falls, and "
            "the rise "
            "grows with cold",
            subtitle="Each dot is one winter night: mean demand over the half-hours starting 00:30 "
            "to 02:30 minus the mean over those starting 22:00 to 23:30, in points (1% of the "
            "substation's 99th percentile of demand). Black line: the fitted line, set on the "
            "average winter. Dashed grey line: the same difference averaged over June to "
            "August nights.",
        )
    )


def figure_3_winters(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """The step per winter for the six decision series."""
    rows = []
    for r in numbers["per_winter"]:
        if r["site"] not in common.DECISION_SERIES:
            continue
        winter = int(r["winters"])
        rows.append(
            {
                "site": r["site"],
                "winter": f"{winter}-{str(winter + 1)[2:]}",
                "value": r["step"],
                "low": r["step_low"],
                "high": r["step_high"],
            }
        )
    frame = pl.DataFrame(rows)
    winters = sorted({int(r["winters"]) for r in numbers["per_winter"]})
    names = [f"{w}-{str(w + 1)[2:]}" for w in winters]
    base = alt.Chart(frame).encode(
        y=alt.Y("site:N", sort=list(common.DECISION_SERIES), title=None),
        yOffset=alt.YOffset("winter:N", sort=names),
        color=alt.Color(
            "winter:N",
            scale=alt.Scale(domain=names, range=[WINTER_COLOURS[w] for w in winters]),
            legend=alt.Legend(title="Winter", orient="top"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=STEP_AXIS), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=45, opacity=1).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    return (_zero_rule() + rule + dot).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The step has the same sign in every winter at the six decision substations, "
            "and falls "
            "over the four winters at D10",
            subtitle="Dot: step at a daily mean of 3 C, in points, fitted to that winter alone. "
            "Line: "
            "90% interval from resampling 14-day blocks of nights. The six decision substations "
            "were set before any fit as the ones the conclusion rests on.",
        ),
    )


def figure_4_gate() -> alt.TopLevelMixin:
    """Known-answer recovery: coverage of the true step by the 90% interval, per series and cell."""
    summary = pl.read_parquet(common.WORK_DIR / "gate_summary.parquet")
    names = {
        "matched": "matched shape",
        "ramp_plateau": "ramp then plateau",
        "flat": "flat",
        "night_temperature": "night temperature",
        "placebo": "no step (placebo)",
    }
    frame = summary.with_columns(
        cell=pl.col("shape").replace_strict(names)
        + pl.when(pl.col("shape") == "placebo")
        .then(pl.lit(""))
        .otherwise(pl.format(", {} points", pl.col("S").cast(pl.Int64))),
        coverage_pct=pl.col("coverage") * 100.0,
    )
    order = [
        "matched shape, 4 points",
        "matched shape, 7 points",
        "ramp then plateau, 4 points",
        "ramp then plateau, 7 points",
        "flat, 4 points",
        "flat, 7 points",
        "no step (placebo)",
        "night temperature, 4 points",
        "night temperature, 7 points",
    ]
    band = (
        alt.Chart(pl.DataFrame({"low": [80.0], "high": [98.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN_LIGHT, opacity=0.6)
        .encode(x="low:Q", x2="high:Q")
    )
    dots = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=36, opacity=0.8, color=DECISION_COLOUR)
        .encode(
            x=alt.X(
                "coverage_pct:Q",
                title="Share of 200 draws whose 90% interval holds the true step (%)",
                scale=alt.Scale(domain=[40, 100]),
            ),
            y=alt.Y("cell:N", sort=order, title=None),
        )
    )
    target = alt.Chart(pl.DataFrame({"x": [90.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (band + target + dots).properties(
        width=WIDTH,
        height=22 * len(order),
        title=_title(
            text="Known-answer test: the 90% interval holds the injected step in 80-98% of "
            "draws at "
            "every substation, for the four gated shapes",
            subtitle="We added a known step to the real nights 200 times per substation and "
            "counted "
            "how often the interval held it. Each dot is one of the 14 substations. Green band: "
            "the per-substation target of 80 to 98%. Black line: 90%. Shapes: matched (grows with "
            "cold), ramp then plateau, flat (no cold dependence), and no step. The bottom two rows "
            "are descriptive and have no target.",
        ),
    )


def figure_5_pooled(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """The five pooled sets, with both intervals where there are at least five series."""
    rows = [
        {
            "set": f"{SET_NAMES[p['set']]} (k={p['k']})",
            "kind": "Hartung-Knapp 90% interval" if p["k"] >= 5 else "range of the series' steps",
            "value": p["mean"],
            "low": p["low_hk"] if p["k"] >= 5 else p["minimum"],
            "high": p["high_hk"] if p["k"] >= 5 else p["maximum"],
            "boot_low": p["low_boot"],
            "boot_high": p["high_boot"],
        }
        for p in numbers["pooled"]
    ]
    frame = pl.DataFrame(rows)
    order = [r["set"] for r in rows]
    base = alt.Chart(frame).encode(
        y=alt.Y("set:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    )
    rule = base.mark_rule(strokeWidth=4, color=POOLED_COLOUR).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=STEP_AXIS), x2="high:Q"
    )
    boot = base.mark_rule(strokeWidth=1.5, color="#333333", strokeDash=[2, 2]).encode(  # ty: ignore[unresolved-attribute]
        x="boot_low:Q", x2="boot_high:Q", yOffset=alt.value(8)
    )
    dot = base.mark_point(filled=True, size=80, color="#333333", opacity=1).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    return (_zero_rule() + rule + boot + dot).properties(
        width=WIDTH,
        height=30 * len(order),
        title=_title(
            text="The three substations that prompted the study average a larger step than the "
            "other "
            "three decision substations, but the eight further substations (E7-X) average as "
            "large",
            subtitle="Dot: pooled mean (plain mean for sets of fewer than 5). Thick purple line: "
            "Hartung-Knapp 90% interval, or the smallest to largest step for a set of three. Thin "
            "dashed line: 90% interval from resampling whole substations. k is the number of "
            "substations.",
        ),
    )


def figure_6_absorbed(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """The full step against the part experiment 114's baseline model leaves, same winters."""
    rows = []
    for r in numbers["cross_checks"]:
        label = {
            "b0": "left by the baseline model",
            "b0_k": "left after its recent-error filter",
            "t_all_k": "left after the interaction model and filter",
        }[r["arm"]]
        rows.append(
            {
                "site": r["site"],
                "what": label,
                "value": r["step"],
                "low": r["step_low"],
                "high": r["step_high"],
            }
        )
        if r["arm"] == "b0":
            rows.append(
                {
                    "site": r["site"],
                    "what": "full step (this experiment)",
                    "value": r["d_step_same_winters"],
                    "low": r["d_step_same_winters"],
                    "high": r["d_step_same_winters"],
                }
            )
    frame = pl.DataFrame(rows)
    names = [
        "full step (this experiment)",
        "left by the baseline model",
        "left after its recent-error filter",
        "left after the interaction model and filter",
    ]
    base = alt.Chart(frame).encode(
        y=alt.Y("site:N", sort=list(common.DECISION_SERIES), title=None),
        yOffset=alt.YOffset("what:N", sort=names),
        color=alt.Color(
            "what:N",
            scale=alt.Scale(
                domain=names, range=[POOLED_COLOUR, DECISION_COLOUR, ocf.DATA_SKY, ocf.DATA_GREEN]
            ),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=STEP_AXIS), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=40, opacity=1).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    return (_zero_rule() + rule + dot).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="Experiment 114's day-ahead forecast errors still contain a large part of the "
            "step",
            subtitle="Winters 2023-24 and 2024-25 only (the winters with stored forecasts). "
            "Purple dot: "
            "the full step over the same two winters. Other rows: the same regression on the "
            "difference between actual demand and experiment 114's day-ahead median forecast, for "
            "the baseline model, then with a recent-error filter, then with an interaction model "
            "and the filter. Line: 90% interval.",
        ),
    )


def figure_7_timing(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """Where the largest jump falls on cold British Summer Time nights, against winter nights."""
    rows = [
        {
            "site": r["site"],
            "shift": r["shift_half_hours"],
            "nights": r["bst_nights"],
            "kind": _group(r["site"]),
        }
        for r in numbers["bst_jump"]
        if "shift_half_hours" in r
    ]
    frame = pl.DataFrame(rows)
    dots = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1)
        .encode(
            x=alt.X(
                "shift:Q",
                title="Shift of the largest jump on cold BST nights (half-hours later; 0 = "
                "fixed to the local clock, 2 = fixed to UTC)",
                scale=alt.Scale(domain=[-1, 10]),
            ),
            y=alt.Y("site:N", sort=SITE_ORDER, title=None),
            color=_kind_colour(pooled=False),
        )
    )
    local = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[3, 3])
        .encode(x="x:Q")
    )
    utc = alt.Chart(pl.DataFrame({"x": [2.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (local + utc + dots).properties(
        width=WIDTH,
        height=20 * len(rows),
        title=_title(
            text="On cold British Summer Time nights the jump arrives one hour later on the "
            "clock, so "
            "the step follows UTC at 12 of 14 substations",
            subtitle="Chosen after seeing the data, and its error rate has not been tested. "
            "Each dot "
            "is the local-clock half-hour of the largest mean rise on nights with a daily mean "
            "below 8 C in British Summer Time, minus the same on winter nights, in half-hours "
            "(positive is later). Nights counted: 18 at D9, 68 to 90 at the others.",
        ),
    )


def figure_8_profile(*, numbers: dict[str, Any]) -> alt.TopLevelMixin:
    """The mean demand profile on cold winter nights minus summer nights, per series."""
    rows = []
    labels = [f"{(m // 60) % 24:02d}:{m % 60:02d}" for m in (22 * 60 + 30 * i for i in range(20))]
    for r in numbers["start_times"]:
        profile = r.get("profile_cold_minus_summer")
        if not profile:
            continue
        rows.extend(
            {
                "site": r["site"],
                "slot": index,
                "excess": value,
                "kind": "selected" if r["site"] in common.SELECTED_SERIES else _group(r["site"]),
            }
            for index, value in enumerate(profile)
        )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["selected", "decision series", "E7-X series"],
            range=[POOLED_COLOUR, DECISION_COLOUR, EXTENSION_COLOUR],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    axis_values = [0, 4, 8, 12, 16]
    axis = alt.Axis(values=axis_values, labelExpr=f"{labels[:1]}[0]")
    del axis
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X(
                "slot:Q",
                title="Half-hour start (clock time on winter nights, when local time equals UTC)",
                axis=alt.Axis(values=[0, 4, 8, 12, 16], labelExpr=CLOCK_LABEL),
            ),
            y=alt.Y("excess:Q", title="Demand minus 22:00-23:30 level, cold minus summer (points)"),
            detail="site:N",
            color=colour,
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="On cold winter nights demand rises within one half-hour around 01:00",
                subtitle="Each line is one substation: mean demand over the night relative to "
                "the mean of "
                "the half-hours starting 22:00 to 23:30, in points, on winter nights with a daily "
                "mean below 8 C. Summer nights' value is subtracted.",
            ),
        )
    )


def _cold_night_panel(
    *, entry: dict[str, Any], labels: list[str], title: str, show_forecast: bool
) -> alt.LayerChart:
    rows = []
    for name, key in (
        ("Actual demand", "actual"),
        ("Baseline forecast (experiment 114)", "forecast"),
    ):
        if key not in entry or (key == "forecast" and not show_forecast):
            continue
        block = entry[key]
        rows.extend(
            {
                "slot": index,
                "series": name,
                "median": block["median"][index],
                "low": block["low"][index],
                "high": block["high"][index],
            }
            for index in range(len(labels))
        )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=["Actual demand", "Baseline forecast (experiment 114)"],
            range=[ocf.BRAND_ORANGE, DECISION_COLOUR],
        ),
        legend=alt.Legend(title=None, orient="top", labelLimit=360) if show_forecast else None,
    )
    x = alt.X(
        "slot:Q",
        title="Half-hour start (clock time)",
        axis=alt.Axis(values=[0, 4, 8, 12, 16], labelExpr=CLOCK_LABEL),
        scale=alt.Scale(domain=[0, 19]),
    )
    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("low:Q", title="Demand (points)", scale=alt.Scale(zero=False)),
            y2="high:Q",
            color=colour,
        )
    )
    line = alt.Chart(frame).mark_line(strokeWidth=2).encode(x=x, y="median:Q", color=colour)  # ty: ignore[unresolved-attribute]
    return (band + line).properties(width=WIDTH, height=120, title=title)


def figure_9_cold_nights(*, cold: dict[str, Any]) -> alt.TopLevelMixin:
    """Actual demand against the baseline forecast on cold nights, decision series."""
    sites = [s for s in common.DECISION_SERIES if "forecast" in cold["series"][s]]
    panels = [
        _cold_night_panel(
            entry=cold["series"][s],
            labels=cold["labels"],
            title=f"{s}: {cold['series'][s]['nights']} cold nights",
            show_forecast=True,
        )
        for s in sites
    ]
    return alt.vconcat(*panels).properties(
        title=_title(
            text="On cold nights demand at D5 and D16 runs above the baseline forecast from 01:00, "
            "and D10 runs below it before midnight",
            subtitle="Line: median over the cold winter nights of winters 2023-24 and 2024-25; "
            "band: "
            "25th to 75th percentile. Cold nights are the coldest 10% of days by daily mean "
            "temperature. The baseline forecast is experiment 114's day-ahead forecast; D9 has no "
            "stored forecast for these winters. Points are 1% of each substation's 99th percentile "
            "of demand.",
        )
    )


def figure_10_cold_nights_e7x(*, cold: dict[str, Any]) -> alt.TopLevelMixin:
    """Actual demand on cold nights at the E7-X series (no stored forecast)."""
    panels = [
        _cold_night_panel(
            entry=cold["series"][s],
            labels=cold["labels"],
            title=f"{s}: {cold['series'][s]['nights']} cold nights",
            show_forecast=False,
        )
        for s in common.EXTENSION_SERIES
    ]
    pairs = [alt.hconcat(*panels[i : i + 2]) for i in range(0, len(panels), 2)]
    return alt.vconcat(*pairs).properties(
        title=_title(
            text="Seven of the eight further substations (E7-X) show a rise around 01:00 on cold "
            "nights; D8 shows none",
            subtitle="Line: median demand over the cold winter nights of winters 2023-24 and "
            "2024-25; band: 25th to 75th percentile (orange). No baseline forecast is stored "
            "for these substations.",
        )
    )


def main() -> None:
    """Draw the figures of a stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["headline", "all"])
    stage = parser.parse_args().stage
    numbers = json.loads((common.WORK_DIR / f"numbers_{stage}.json").read_text())
    _save(figure_1_headline(numbers=numbers), name="figure_1_headline")
    _save(figure_2_raw(numbers=numbers), name="figure_2_raw")
    _save(figure_3_winters(numbers=numbers), name="figure_3_winters")
    _save(figure_4_gate(), name="figure_4_gate")
    _save(figure_5_pooled(numbers=numbers), name="figure_5_pooled")
    _save(figure_6_absorbed(numbers=numbers), name="figure_6_absorbed")
    _save(figure_7_timing(numbers=numbers), name="figure_7_timing")
    _save(figure_8_profile(numbers=numbers), name="figure_8_profile")
    cold = json.loads((common.WORK_DIR / f"cold_nights_{stage}.json").read_text())
    _save(figure_9_cold_nights(cold=cold), name="figure_9_cold_nights")
    _save(figure_10_cold_nights_e7x(cold=cold), name="figure_10_cold_nights_e7x")


if __name__ == "__main__":
    main()
