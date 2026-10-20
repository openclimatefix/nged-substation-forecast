"""Draw the figures of experiment 112 as SVG into `draft/`, from the stored results.

Only Primary substations appear, under their D-labels, so dates may be shown. Every number comes
from `work/report_contrasts.json`, `work/objectives.json`, `work/gate.json`,
`work/gate_diagnostics.json`, `work/profile.json`, and the stored predictions. Afterwards optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00112-*/e00112_charts.py
"""

import json
import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00112_common as common
import e00112_report as report
import plotting.ocf_theme as ocf
import polars as pl

FIGURE_DIR: Final[Path] = Path(__file__).parent / "draft"
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16", "D9"]
GATE_LABEL: Final[str] = "The known-answer gate failed, so the verdict is capped at inconclusive."
ARM_LABELS: Final[dict[str, str]] = {
    "vp_p": "VP-P (lower-objective start)",
    "vp_p_s0": "VP-P, start 0",
    "vp_p_s1": "VP-P, start 1",
    "vp_p_nonet": "VP-P without the network",
    "vp_s": "VP-S",
    "p_map_same": "P-map-same",
    "p_map_polish": "P-map-polish",
    "p_map": "P-map (105)",
    "p": "P (105)",
    "s": "S (105)",
    "g_2y": "G-2y",
    "g_all": "G-all",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "VP-P (lower-objective start)": ocf.BRAND_ORANGE,
    "VP-P, start 0": ocf.BRAND_ORANGE,
    "VP-P, start 1": "#FF9A6B",
    "VP-P without the network": ocf.DATA_PURPLE,
    "VP-S": ocf.DATA_SKY,
    "P-map-same": ocf.DATA_BLUE,
    "P-map-polish": "#7FA1FF",
    "P-map (105)": "#5C5C5C",
    "P (105)": "#8A8A8A",
    "S (105)": "#B0B0B0",
    "G-2y": ocf.DATA_GREEN,
    "G-all": "#8EF2C7",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(f"{subtitle} {GATE_LABEL}", WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / f"{name}.svg")
    chart.save(FIGURE_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _read(name: str) -> dict:
    return json.loads((common.WORK_DIR / name).read_text())


def _arms(year: str) -> dict[str, pl.DataFrame]:
    ids = report.site_of_id()
    return {n: report.headline(f) for n, f in report.load_year(year=year, ids=ids).items()}


def figure_1_headline() -> alt.TopLevelMixin:
    """The four planned contrasts, scored year with the replicate year beside it."""
    contrasts = _read("report_contrasts.json")
    rows = []
    for label, *_ in report.PLANNED:
        scored = contrasts["scored"].get(label)
        replicate = contrasts["replicate"].get(label)
        if scored is None:
            continue
        rows.append(
            {
                "contrast": label,
                "estimate": scored["estimate"],
                "low": scored["low"],
                "high": scored["high"],
                "replicate": replicate["estimate"] if replicate else None,
                "sites": f"{scored['better']} better, {scored['worse']} worse",
            }
        )
    table = pl.DataFrame(rows)
    order = table["contrast"].to_list()
    y = alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    rule = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2, color=ocf.BRAND_ORANGE)
        .encode(y=y, x="low:Q", x2="high:Q")
    )
    dot = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE)
        .encode(
            y=y,
            x=alt.X("estimate:Q", title="Difference in quantile score (points; below 0 is better)"),
        )
    )
    hollow = (
        alt.Chart(table.drop_nulls("replicate"))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=False, size=70, color=ocf.DATA_BLUE, strokeWidth=2)
        .encode(y=y, x="replicate:Q")
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=8, fontSize=10)
        .encode(y=y, x="high:Q", text="sites:N")
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + hollow + text).properties(
        width=WIDTH - 150,
        height=48 * len(order),
        title=_title(
            text="Variable projection did not beat the gradient fit, pooled or per-site",
            subtitle=(
                "Orange dot: scored year (2025-07 to 2026-06), difference in mean quantile score "
                "(points of capacity; lower is better) over the five decision sites; orange line: "
                "98.75% interval from resampling whole calendar months (Bonferroni for four "
                "planned contrasts). Blue ring: the replicate year (2024-07 to 2025-06). "
                "C2v and C3v are over six substations."
            ),
        ),
    )


def figure_2_objectives() -> alt.TopLevelMixin:
    """The final training objective of each fit, in nats, at each cut."""
    objectives = _read("objectives.json")
    labels = {
        "vp_p_s0": "VP-P, start 0",
        "vp_p_s1": "VP-P, start 1",
        "p_map_polish": "P-map-polish",
        "p_map_same": "P-map-same",
    }
    rows = [
        {"cut": cut, "fit": labels[name], "nats": value}
        for cut, values in objectives.items()
        for name, value in values.items()
        if name in labels
    ]
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("fit:N", sort=list(labels.values()), title=None),
            x=alt.X(
                "nats:Q",
                title="Training objective (nats; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "fit:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
            row=alt.Row("cut:N", title="Training cut"),
        )
        .properties(
            width=WIDTH - 150,
            height=70,
            title=_title(
                text="Variable projection reaches a far lower training objective",
                subtitle=(
                    "All fits are evaluated by one function (the variable-projection objective), "
                    "on "
                    "the same training rows and noise scale at each cut."
                ),
            ),
        )
    )


def figure_3_scores() -> alt.TopLevelMixin:
    """Five-site mean quantile score of the main arms, both years."""
    rows = []
    for year, title in (("scored", "Scored year"), ("rep", "Replicate year")):
        for name, frame in _arms(year).items():
            if name not in ARM_LABELS or name in {"vp_p_s0", "vp_p_s1"}:
                continue
            five = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
            rows.append(
                {
                    "year": title,
                    "arm": ARM_LABELS[name],
                    "score": float(five["score"].to_numpy().mean()),
                }
            )
    table = pl.DataFrame(rows)
    order = [label for label in ARM_LABELS.values() if label in set(table["arm"])]
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X(
                "score:Q",
                title="Mean quantile score (points; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
            row=alt.Row("year:N", title=None),
        )
        .properties(
            width=WIDTH - 200,
            height=16 * len(order),
            title=_title(
                text="With the network, the exact fit scores worse than the gradient fit",
                subtitle=(
                    "Mean quantile score over the five decision substations (D5, D10, D11, D14, "
                    "D16) on rows outside the switching events, in points of effective capacity."
                ),
            ),
        )
    )


def figure_4_weeks() -> alt.TopLevelMixin:
    """Actual power and four forecasts in a winter week and a summer week at each site."""
    arms = {
        name: report.load_year(year="scored", ids=report.site_of_id())[name]
        for name in ("vp_p", "vp_p_nonet", "p_map_same", "g_2y")
    }
    labels = {name: ARM_LABELS[name] for name in arms}
    weeks = {
        "Winter: 12 to 18 January 2026": date(2026, 1, 12),
        "Spring: 16 to 22 March 2026": date(2026, 3, 16),
    }
    frames = []
    for week, start in weeks.items():
        for name, frame in arms.items():
            part = frame.filter(
                pl.col("valid_time").dt.date() >= start,
                pl.col("valid_time").dt.date() < pl.lit(start) + pl.duration(days=7),
                pl.col("site").is_in(SITE_ORDER[:5]),
            ).with_columns(
                week=pl.lit(week),
                series=pl.lit(labels[name]),
                value=pl.col("q50"),
                day=(
                    pl.col("valid_time")
                    - pl.datetime(start.year, start.month, start.day, time_zone="UTC")
                ).dt.total_hours()
                / 24.0,
            )
            frames.append(part.select("site", "week", "series", "day", "value"))
            if name == "vp_p":
                frames.append(
                    part.select(
                        "site",
                        "week",
                        pl.lit("Actual").alias("series"),
                        "day",
                        pl.col("actual").alias("value"),
                    )
                )
    table = pl.concat(frames)
    domain = ["Actual", *labels.values()]
    colours = ["#222222", ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.DATA_GREEN]
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X(
                "day:Q",
                title="Day of the week (0 = Monday 00:00 UTC)",
                axis=alt.Axis(values=[0, 2, 4, 6]),
            ),
            y=alt.Y(
                "value:Q", title="Power (% of effective capacity)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(orient="top", title=None),
            ),
            row=alt.Row("site:N", sort=SITE_ORDER[:5], title=None),
            column=alt.Column("week:N", title=None),
        )
        .properties(
            width=(WIDTH - 90) / 2,
            height=70,
            title=_title(
                text="The network-on fit misses in spring",
                subtitle=(
                    "Median forecast and actual power, scored year, at the five decision "
                    "substations."
                ),
            ),
        )
    )


def figure_5_profile() -> alt.TopLevelMixin | None:
    """The capacity profile: objective against the multiplier on the convex capacity path."""
    path = common.WORK_DIR / "profile.json"
    if not path.exists():
        return None
    out = json.loads(path.read_text())
    rows, marks = [], []
    for config, result in out.items():
        lowest = min(v["objective"] for v in result["profile"].values())
        rows += [
            {"config": config, "multiplier": float(m), "excess": v["objective"] - lowest}
            for m, v in result["profile"].items()
        ]
        convex = result["convex_capacity_points"]
        for start, entry in result["free"].items():
            for site, points in entry["capacity_points"].items():
                marks.append(
                    {
                        "config": config,
                        "start": start,
                        "multiplier": points / convex[site],
                        "excess": entry["objective_without_capacity_prior"] - lowest,
                    }
                )
    line = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(
            x=alt.X("multiplier:Q", title="Multiplier on the convex capacity path"),
            y=alt.Y("excess:Q", title="Objective above the lowest profile point (nats)"),
            row=alt.Row("config:N", title=None),
        )
    )
    points = (
        alt.Chart(pl.DataFrame(marks))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40)
        .encode(
            x="multiplier:Q",
            y="excess:Q",
            color=alt.Color(
                "start:N",
                title="Free start",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE]),
            ),
            row=alt.Row("config:N", title=None),
        )
    )
    return (line + points).properties(
        width=WIDTH - 90,
        height=110,
        title=_title(
            text="Free fits land away from the profile minimum",
            subtitle=(
                "Line: refit objective with the capacity pinned at the multiplier times the "
                "convex path. Dots: each scored substation's free-fit capacity over the convex "
                "capacity, at that fit's objective, from three starts."
            ),
        ),
    )


def figure_6_gate() -> alt.TopLevelMixin | None:
    """Worst-case recovery errors of the gate and of the post hoc variants, against the limits."""
    path = common.WORK_DIR / "gate_diagnostics.json"
    if not path.exists():
        return None
    diagnostics = json.loads(path.read_text())
    gate = _read("gate.json")["test_2"]["recovery"]["s0"]
    limits = {
        "threshold_error_c": 1.0,
        "slope_relative_error": 0.15,
        "capacity_relative_error": 0.10,
        "network_midday_points": 0.1,
    }
    names = {
        "threshold_error_c": "Threshold error (degrees C; limit 1)",
        "slope_relative_error": "Slope error (share; limit 0.15)",
        "capacity_relative_error": "Capacity error (share; limit 0.10)",
        "network_midday_points": "Network midday output (points; limit 0.1)",
    }
    rows = []
    for key, label in names.items():
        rows.append(
            {
                "run": "Gate: noise sd 5, network on",
                "measure": label,
                "value": float(max(gate[key]) if isinstance(gate[key], list) else gate[key]),
                "limit": limits[key],
            }
        )
        for run, entry in diagnostics.items():
            if isinstance(entry, dict) and "worst" in entry and run != "continue_default_start":
                rows.append(
                    {
                        "run": run.replace("_", " "),
                        "measure": label,
                        "value": entry["worst"][key],
                        "limit": limits[key],
                    }
                )
    table = pl.DataFrame(rows)
    bars = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("run:N", title=None, sort=None),
            x=alt.X("value:Q", title="Worst error over the eight substations"),
            facet=alt.Facet("measure:N", columns=2, title=None),
        )
        .properties(width=(WIDTH - 200) / 2, height=130)
    )
    rule = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="limit:Q", facet=alt.Facet("measure:N", columns=2, title=None))
    )
    return (bars + rule).properties(
        title=_title(
            text="The synthetic fleet is recovered exactly without noise, not at realistic noise",
            subtitle=(
                "Worst-case error over eight synthetic substations for the pre-registered gate and "
                "the post hoc diagnostics; the orange line is the gate's limit."
            ),
        )
    )


def main() -> None:
    """Draw every figure whose inputs exist."""
    for name, function in (
        ("figure_01_headline", figure_1_headline),
        ("figure_02_objectives", figure_2_objectives),
        ("figure_03_scores", figure_3_scores),
        ("figure_04_weeks", figure_4_weeks),
        ("figure_05_profile", figure_5_profile),
        ("figure_06_gate", figure_6_gate),
    ):
        chart = function()
        if chart is not None:
            _save(chart, name=name)


if __name__ == "__main__":
    main()
