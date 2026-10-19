"""Draw the figures of experiment 104 as SVG into `draft/`, from the stored predictions.

Only Primary substations appear, under their D-labels, so dates may be shown. Afterwards optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00104-*/e00104_charts.py
"""

import json
import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00104_common as common
import e00104_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

FIGURE_DIR: Final[Path] = Path(__file__).parent / "draft"
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16", "D9"]
SITE_COLOURS: Final[list[str]] = [
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    "#808080",
]
ARM_LABELS: Final[dict[str, str]] = {
    "g1": "G1",
    "g_all": "G-all",
    "g_all_d1": "G-all+D1",
    "g_all_d1_steps": "G-all+D1+steps",
    "x_12": "X-12",
    "x_12z": "X-12z",
    "x_12am": "X-12am",
    "x_12amq": "X-12amq",
    "x_all": "X-all",
    "x_allz": "X-allz",
    "x_allam": "X-allam",
    "x_allamq": "X-allamq",
    "x_12z_ng": "X-12z (no irradiance)",
    "x_12am_ng": "X-12am (no irradiance)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
SHADE: Final[str] = "#EAEAEA"


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / f"{name}.svg")
    chart.save(FIGURE_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _arms(*, names: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    return {name: report.load_arm(name=name) for name in names}


def figure_1_raw_weeks() -> alt.TopLevelMixin:
    """Actual power and three forecasts in a summer week and a winter week at each site."""
    weeks = {
        "Summer: 14 to 20 July 2025": date(2025, 7, 14),
        "Winter: 12 to 18 January 2026": date(2026, 1, 12),
    }
    arms = {
        name: report.load_arm(name=name, headline=False) for name in ("g1", "g_all_d1", "x_allamq")
    }
    labels = {"g1": "G1", "g_all_d1": "G-all+D1", "x_allamq": "X-allamq"}
    frames = []
    for week, start in weeks.items():
        for name, frame in arms.items():
            part = frame.filter(
                pl.col("valid_time").dt.date() >= start,
                pl.col("valid_time").dt.date() < pl.lit(start) + pl.duration(days=7),
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
            if name == "g1":
                frames.append(
                    part.select(
                        "site",
                        "week",
                        pl.lit("Actual").alias("series"),
                        "day",
                        pl.col("actual").alias("value"),
                    )
                )
    table = pl.concat(frames).filter(pl.col("site").is_in(SITE_ORDER))
    order = ["Actual", "G1", "G-all+D1", "X-allamq"]

    def week_chart(week: str, *, legend: bool) -> alt.FacetChart:
        return (
            alt.Chart(table.filter(pl.col("week") == week))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X(
                    "day:Q",
                    title="Days since Monday 00:00 UTC",
                    axis=alt.Axis(values=list(range(8))),
                    scale=alt.Scale(domain=[0, 7]),
                ),
                y=alt.Y("value:Q", title=None),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=order,
                        range=["#333333", ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                    ),
                    legend=alt.Legend(title="Line", orient="top", direction="horizontal")
                    if legend
                    else None,
                ),
            )
            .properties(width=WIDTH - 40, height=60)
            .facet(
                row=alt.Row(
                    "site:N",
                    sort=SITE_ORDER,
                    title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left"),
                )
            )
            .properties(title=week)
        )

    return alt.vconcat(
        *[week_chart(week, legend=position == 0) for position, week in enumerate(weeks)]
    ).properties(
        title=_title(
            text="Net power and three forecasts in a summer and a winter week",
            subtitle=(
                "Half-hourly net power as a percentage of each site's effective capacity (negative "
                "is export). Forecast lines are the median of each frozen fit's forecast, made "
                "with "
                "data to 30 June 2025. Black: measured. Orange: convex model, 12 months. Blue: "
                "convex model, all years with drift. Purple: XGBoost, all years, with the solar "
                "offset."
            ),
        )
    )


def figure_2_headline() -> alt.TopLevelMixin:
    """The three planned contrasts, and the contrasts against the stronger XGBoost arms."""
    arms = _arms(names=report.MAIN_ARMS)
    rows = []
    for label, treatment, reference, level, kind in (
        ("C1 years: G-all+D1 minus G1", "g_all_d1", "g1", report.BONFERRONI_LEVEL, "planned"),
        ("C2: G1 minus X-12am", "g1", "x_12am", report.BONFERRONI_LEVEL, "planned"),
        ("C3: G-all+D1 minus X-allam", "g_all_d1", "x_allam", report.BONFERRONI_LEVEL, "planned"),
        ("G-all minus G1", "g_all", "g1", None, "exploratory"),
        ("G-all+D1+steps minus G-all+D1", "g_all_d1_steps", "g_all_d1", None, "exploratory"),
        ("G1 minus X-12 (default start)", "g1", "x_12", None, "exploratory"),
        ("G-all minus X-all (default start)", "g_all", "x_all", None, "exploratory"),
        ("G-all+D1 minus X-all (default start)", "g_all_d1", "x_all", None, "exploratory"),
        ("X-12amq minus X-12 (offset)", "x_12amq", "x_12", None, "exploratory"),
        ("X-12z minus X-12 (zero start)", "x_12z", "x_12", None, "exploratory"),
    ):
        result = report.contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=common.ELIGIBLE_SITES,
            level=level,
        )
        rows.append(
            {
                "contrast": label,
                "estimate": result["estimate"],
                "low": result["low"],
                "high": result["high"],
                "kind": kind,
                "sites": f"{result['better']} of 5 sites better",
            }
        )
    uncentred_arms = {
        name: report.load_arm(name=name, uncentred=name in common.GAM_ARMS)
        for name in ("g1", "g_all", "g_all_d1", "x_12am", "x_allam")
    }
    for label, treatment, reference in (
        ("C1 years: G-all+D1 minus G1", "g_all_d1", "g1"),
        ("C2: G1 minus X-12am", "g1", "x_12am"),
        ("C3: G-all+D1 minus X-allam", "g_all_d1", "x_allam"),
        ("G-all minus G1", "g_all", "g1"),
    ):
        result = report.contrast(
            arms=uncentred_arms,
            treatment=treatment,
            reference=reference,
            sites=common.ELIGIBLE_SITES,
            level=None,
        )
        rows.append(
            {
                "contrast": label,
                "estimate": result["estimate"],
                "low": result["low"],
                "high": result["high"],
                "kind": "uncentred spread (first run, 95%)",
                "sites": f"{result['better']} of 5 sites better",
            }
        )
    table = pl.DataFrame(rows)
    order = list(dict.fromkeys(table["contrast"].to_list()))
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("kind:N"),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["planned", "exploratory", "uncentred spread (first run, 95%)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, "#888888"],
            ),
            legend=alt.Legend(
                title="Interval", orient="top", direction="horizontal", labelLimit=300
            ),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=6, fontSize=10)
        .encode(
            y=alt.Y("contrast:N", sort=order),
            yOffset=alt.YOffset("kind:N"),
            x="high:Q",
            text="sites:N",
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=WIDTH - 150,
        height=34 * len(order),
        title=_title(
            text="Years of history, and the convex model against XGBoost",
            subtitle=(
                "Dot: difference in mean quantile score over the five decision sites (D5, D10, "
                "D11, D14, D16), scored 1 July 2025 to 30 June 2026 on rows outside the switching"
                " events. Line: 98.33% interval (planned, Bonferroni for three) or 95% interval "
                "(exploratory) from resampling whole calendar months. Grey: the first run's "
                "forecasts, "
                "whose spread carried the replica year's mean bias (95%). Dashed line: no "
                "difference."
            ),
        ),
    )


def figure_3_absolute() -> alt.TopLevelMixin:
    """Every arm's mean quantile score at each site."""
    arms = _arms(names=report.MAIN_ARMS)
    rows = [
        {"arm": ARM_LABELS[name], "site": site, "score": float(score)}
        for name, frame in arms.items()
        for site, score in frame.group_by("site").agg(pl.col("score").mean()).iter_rows()
    ]
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[name] for name in report.MAIN_ARMS]
    base = alt.Chart(table).encode(
        x=alt.X("site:N", sort=SITE_ORDER, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("arm:N", sort=order, title=None),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[2.0, 6.0], range=["#FFFFFF", ocf.DATA_BLUE], clamp=True),
            legend=alt.Legend(title="Score (points)", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text=alt.Text("score:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    return (heat + text).properties(
        width=WIDTH - 130,
        height=18 * len(order),
        title=_title(
            text="Quantile score of every arm at every site",
            subtitle=(
                "Twice the mean pinball loss over 11 quantile levels, in points of effective "
                "capacity (smaller is better), 1 July 2025 to 30 June 2026, on rows outside the "
                "switching events. D9 has 9.0 months of history at 1 July 2024 and is not in the "
                "decision set."
            ),
        ),
    )


def figure_4_drift() -> alt.TopLevelMixin:
    """The fitted drift path per hour band for the arm with drift."""
    drift = pl.read_parquet(common.WORK_DIR / "gam_drift_scored.parquet").filter(
        pl.col("arm") == "g_all_d1"
    )
    band_names = ["Night", "Morning", "Midday", "Evening"]
    table = drift.with_columns(
        date=(pl.col("day") * 86400.0 * 1000.0).cast(pl.Datetime("ms")),
        band_name=pl.col("band").replace_strict(
            dict(enumerate(band_names)), return_dtype=pl.String
        ),
    )
    table = table.with_columns(level=pl.col("drift") - pl.col("drift").mean().over("site", "band"))
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.3)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y")),
            y=alt.Y("level:Q", title=None),
            color=alt.Color(
                "site:N",
                sort=SITE_ORDER,
                scale=alt.Scale(domain=SITE_ORDER, range=SITE_COLOURS),
                legend=alt.Legend(title="Site", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 40, height=70)
        .facet(
            row=alt.Row(
                "band_name:N",
                sort=band_names,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=_title(
                text="The fitted drift of the level of demand, by hour band",
                subtitle=(
                    "Smooth drift path of the convex model with drift (G-all+D1) at the 2025-07-01 "
                    "cut, less each path's own mean, in points of effective capacity. Night is "
                    "00:00 to 05:59, morning 06:00 to 10:59, midday 11:00 to 15:59, evening 16:00 "
                    "to 23:59 "
                    "local time. A forecast holds the last value."
                ),
            )
        )
    )


def figure_5_capacity() -> alt.TopLevelMixin:
    """Fitted capacity per season-year against experiment 45's vintage and hindsight constant."""
    fitted = pl.read_parquet(common.WORK_DIR / "gam_capacities_scored.parquet")
    rows = []
    for site in common.SCORED_SITES:
        both = (
            report.vintage_and_hindsight(site=site)
            .group_by("season_year")
            .agg(pl.col("vintage_mw").mean(), pl.col("hindsight_mw").mean())
        )
        for year, vintage, hindsight in both.iter_rows():
            rows.append(
                {"site": site, "year": year, "source": "Vintage (real time)", "mw": vintage}
            )
            rows.append(
                {"site": site, "year": year, "source": "Hindsight constant", "mw": hindsight}
            )
        for arm, label in (("g1", "G1"), ("g_all", "G-all"), ("g_all_d1", "G-all+D1")):
            for row in fitted.filter(pl.col("site") == site, pl.col("arm") == arm).iter_rows(
                named=True
            ):
                if row["capacity_mw"] > 0.05:
                    rows.append(  # noqa: PERF401 (one dict per fitted year)
                        {
                            "site": site,
                            "year": row["season_year"],
                            "source": label,
                            "mw": row["capacity_mw"],
                        }
                    )
    table = pl.DataFrame(rows).filter(pl.col("year") >= 2020)
    domain = ["Vintage (real time)", "Hindsight constant", "G1", "G-all", "G-all+D1"]
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.2)
        .encode(
            x=alt.X("year:O", title="Season-year (starts in July)"),
            y=alt.Y("mw:Q", title="MW"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=domain,
                    range=["#333333", "#999999", ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(
                    title="Capacity", orient="top", direction="horizontal", columns=5
                ),
            ),
        )
        .properties(width=(WIDTH - 60) // 2, height=100)
        .facet(facet=alt.Facet("site:N", sort=SITE_ORDER, title=None), columns=2)
        .properties(
            title=_title(
                text="Solar capacity per season-year: fitted against experiment 45",
                subtitle=(
                    "Capacity of embedded solar in MW. The convex model fits one non-decreasing "
                    "capacity per season-year; experiment 45's real-time vintage is the weekly "
                    "estimate made with data to June 2025, and its hindsight constant is fitted on "
                    "whole years."
                ),
            )
        )
    )


def figure_6_tails() -> alt.TopLevelMixin:
    """Coverage at the four tail levels for every arm, with the nominal level marked."""
    arms = _arms(names=report.MAIN_ARMS)
    rows = []
    for name, frame in arms.items():
        kept = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        actual = kept["actual"].to_numpy().astype(np.float64)
        for column, level in zip(report.TAIL_COLUMNS, report.TAIL_LEVELS, strict=True):
            predicted = kept[column].to_numpy().astype(np.float64)
            rows.append(
                {
                    "arm": ARM_LABELS[name],
                    "level": f"q{round(level * 100):02d} (nominal {level:.0%})",
                    "nominal": level,
                    "coverage": float(np.mean(actual <= predicted)),
                }
            )
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[name] for name in report.MAIN_ARMS]
    levels = [f"q{round(v * 100):02d} (nominal {v:.0%})" for v in report.TAIL_LEVELS]
    base = alt.Chart(table).encode(y=alt.Y("arm:N", sort=order, title=None))
    dots = base.mark_point(filled=True, size=45, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "coverage:Q",
            title="Share of actual values at or below the quantile",
            axis=alt.Axis(format=".0%"),
        )
    )
    nominal = base.mark_tick(color=ocf.BRAND_ORANGE, thickness=2, size=14).encode(x="nominal:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(nominal, dots)
        .properties(width=(WIDTH - 130) // 2, height=14 * len(order))
        .facet(column=alt.Column("level:N", sort=levels, title=None), columns=2)
        .properties(
            title=_title(
                text="Tail calibration of every arm",
                subtitle=(
                    "Share of measured values at or below each tail quantile, on the five decision "
                    "sites' scored rows. Orange tick: the nominal level; a dot on the tick is "
                    "perfectly calibrated, a dot to its right means the quantile sits too low."
                ),
            )
        )
    )


def figure_7_staleness() -> alt.TopLevelMixin:
    """Score and upper-tail coverage by month of the scored year."""
    names = ("g1", "g_all", "g_all_d1", "x_12amq", "x_allamq")
    rows = []
    for name in names:
        kept = report.load_arm(name=name).filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        kept = kept.with_columns(
            month=(pl.col("valid_time").dt.year() - 2025) * 12 + pl.col("valid_time").dt.month() - 7
        )
        stats = kept.group_by("month").agg(
            pl.col("score").mean(), (pl.col("actual") <= pl.col("q95")).mean().alias("coverage")
        )
        rows += [
            {"arm": ARM_LABELS[name], "month": m, "score": s, "coverage": c}
            for m, s, c in stats.iter_rows()
        ]
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[n] for n in names]
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_SKY, ocf.DATA_GREEN]

    def panel(column: str, title: str, *, nominal: float | None) -> alt.LayerChart | alt.Chart:
        line = (
            alt.Chart(table)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, strokeWidth=1.3)
            .encode(
                x=alt.X(
                    "month:Q",
                    title="Months since the cut (July 2025 is month 0)",
                    axis=alt.Axis(values=list(range(12))),
                ),
                y=alt.Y(f"{column}:Q", title=title),
                color=alt.Color(
                    "arm:N",
                    sort=order,
                    scale=alt.Scale(domain=order, range=colours),
                    legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
                ),
            )
            .properties(width=WIDTH - 40, height=110)
        )
        if nominal is None:
            return line
        rule = (
            alt.Chart(pl.DataFrame({"y": [nominal]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(y="y:Q")
        )
        return line + rule

    return alt.vconcat(
        panel("score", "Quantile score (points)", nominal=None),
        panel("coverage", "Share at or below q95", nominal=0.95),
    ).properties(
        title=_title(
            text="Score and upper-tail coverage by month since the fit was frozen",
            subtitle=(
                "Mean quantile score (top; smaller is better) and the share of measured values at"
                " or below the 95% quantile (bottom; the dashed line is the nominal 95%), by "
                "month of the scored year, on the five decision sites. Every fit is frozen at 30 "
                "June 2025."
            ),
        )
    )


def figure_8_benchmark() -> alt.TopLevelMixin | None:
    """Wall-clock seconds of the three workloads on the CPU and the GPU."""
    path = common.WORK_DIR / "benchmark.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    names = {
        "single_fit": "1. Single fit, six sites",
        "stacked_bootstrap": "2. 20 bootstrap fits, stacked",
        "pooled_proxy": "3. Part 3 proxy, 200 steps",
    }
    rows = [
        {
            "workload": names[w],
            "device": d.upper() if d == "cpu" else "GPU",
            "seconds": v["seconds"],
        }
        for d, workloads in summary["devices"].items()
        for w, v in workloads.items()
    ]
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=140)
        .encode(
            y=alt.Y("workload:N", sort=list(names.values()), title=None),
            yOffset="device:N",
            x=alt.X(
                "seconds:Q",
                title="Wall-clock seconds, from cached tensors to results (log scale)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color(
                "device:N",
                scale=alt.Scale(domain=["CPU", "GPU"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title="Device", orient="top", direction="horizontal"),
            ),
        )
        .properties(
            width=WIDTH - 150,
            height=120,
            title=_title(
                text="CPU against GPU for the framework module",
                subtitle=(
                    "Workloads 1 and 2 run in float64 on both devices; workload 3 runs in float32. "
                    "The CPU uses 8 threads. A dot further left is faster."
                ),
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    for name, function in (
        ("figure_01_raw_weeks", figure_1_raw_weeks),
        ("figure_02_headline", figure_2_headline),
        ("figure_03_absolute", figure_3_absolute),
        ("figure_04_drift", figure_4_drift),
        ("figure_05_capacity", figure_5_capacity),
        ("figure_06_tails", figure_6_tails),
        ("figure_07_staleness", figure_7_staleness),
        ("figure_08_benchmark", figure_8_benchmark),
    ):
        chart = function()
        if chart is not None:
            _save(chart, name=name)


if __name__ == "__main__":
    main()
