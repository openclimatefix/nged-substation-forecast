"""Draw the figures of experiment 106 as SVG into the docs folder, from the report and predictions.

Only Primary substations appear, under their D-labels, so dates may be shown. The tables come from
`report.txt` (`e00106_report.py`), so a figure and the page quote the same numbers. Afterwards
optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00106-*/e00106_charts.py
"""

import io
import json
import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00106_common as common
import e00106_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT_PATH: Final[Path] = DOCS_DIR / "report.txt"
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
GREY: Final[str] = "#808080"
ARM_LABELS: Final[dict[str, str]] = {
    "a0": "A0 no drift",
    "a1": "A1 drift",
    "a2": "A2 no drift + Kalman",
    "a3": "A3 drift + Kalman",
    "a4": "A4 drift + EWMA",
    "a4_0": "A4-0 no drift + EWMA",
    "w": "W best window",
    "wk": "WK best window + Kalman",
    "x_best": "X-best (XGBoost)",
    "x_best_k": "X-best + Kalman",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
YEAR_LABELS: Final[dict[str, str]] = {"scored": "Scored year", "replicate": "Replicate year"}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(Path("/tmp/e00106_png") / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def read_sections() -> dict[str, pl.DataFrame]:
    """Return each table of `report.txt` by the text of its `==` heading."""
    sections: dict[str, pl.DataFrame] = {}
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if heading is not None and lines:
            sections[heading] = pl.read_csv(io.StringIO("\n".join(lines)), infer_schema_length=None)

    for line in REPORT_PATH.read_text().splitlines():
        if line.startswith("== "):
            flush()
            heading, lines = line.strip("= ").strip(), []
        elif line.strip():
            lines.append(line)
    flush()
    return sections


def _section(*, sections: dict[str, pl.DataFrame], start: str) -> pl.DataFrame:
    return next(frame for name, frame in sections.items() if name.startswith(start))


def figure_1_weeks() -> alt.TopLevelMixin:
    """Actual power and three forecasts in a summer week and a winter week at each site."""
    weeks = {
        "Summer: 14 to 20 July 2025": date(2025, 7, 14),
        "Winter: 12 to 18 January 2026": date(2026, 1, 12),
    }
    labels = {"a1": "A1", "a3": "A3", "w": "W"}
    frames = []
    for week, start in weeks.items():
        for name, label in labels.items():
            frame = report.load_arm(label="scored", arm=name, drop_switching=False)
            part = frame.filter(
                pl.col("valid_time").dt.date() >= start,
                pl.col("valid_time").dt.date() < pl.lit(start) + pl.duration(days=7),
                pl.col("site_name").is_in(SITE_ORDER),
            ).with_columns(
                week=pl.lit(week),
                site=pl.col("site_name"),
                series=pl.lit(label),
                value=pl.col("q50"),
                day=(
                    pl.col("valid_time")
                    - pl.datetime(start.year, start.month, start.day, time_zone="UTC")
                ).dt.total_hours()
                / 24.0,
            )
            frames.append(part.select("site", "week", "series", "day", "value"))
            if name == "a1":
                frames.append(
                    part.select(
                        "site",
                        "week",
                        pl.lit("Measured").alias("series"),
                        "day",
                        pl.col("actual").alias("value"),
                    )
                )
    table = pl.concat(frames)
    order = ["Measured", "A1", "A3", "W"]

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
            text="Net power and three day-ahead forecasts in a summer and a winter week",
            subtitle=(
                "Half-hourly net power as a percentage of each substation's effective capacity "
                "(negative is export). Lines are the median of each day-ahead forecast. Black: "
                "measured. Orange: A1, the model with drift (frozen at 1 July 2025). Blue: A3, "
                "A1 plus the Kalman filter on recent residuals. Purple: W, the best history "
                "window."
            ),
        )
    )


def figure_2_headline(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The six planned contrasts with intervals, and the replicate year beside each."""
    planned = _section(sections=sections, start="Planned contrasts (")
    rows = []
    for row in planned.iter_rows(named=True):
        name = str(row["contrast"])
        short = name.split(":", 1)[0] + ": " + name.split(":", 1)[1].strip()
        rows.append(
            {
                "contrast": short,
                "estimate": row["scored QS"],
                "low": row["scored low"],
                "high": row["scored high"],
                "kind": "scored year (interval)",
                "label": f"{row['scored sites better']} of 5 sites better",
            }
        )
        rows.append(
            {
                "contrast": short,
                "estimate": row["replicate QS"],
                "low": None,
                "high": None,
                "kind": "replicate year",
                "label": f"{row['replicate sites better']} of 5",
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        label_x=pl.max_horizontal("estimate", "high")
    )
    order = list(dict.fromkeys(table["contrast"].to_list()))
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("kind:N"),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["scored year (interval)", "replicate year"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title="Year", orient="top", direction="horizontal"),
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
            x=alt.X("label_x:Q"),
            text="label:N",
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=WIDTH - 200,
        height=46 * len(order),
        title=_title(
            text="The six planned contrasts, in two scored years",
            subtitle=(
                "Dot: difference in mean quantile score over the five decision substations "
                "(D5, D10, D11, D14, D16), rows outside the switching events. Orange: scored "
                "year, 1 July 2025 to 30 June 2026, with a 99.17% interval from resampling whole "
                "calendar months. Blue: replicate year, 1 July 2024 to 30 June 2025, estimate "
                "only. Dashed line: no difference. The weather is an estimate of past weather "
                "(ERA5 temperature, CAMS irradiance)."
            ),
        ),
    )


def figure_3_absolute(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Every headline arm's mean quantile score at each site, both years."""
    absolute = _section(sections=sections, start="Absolute quantile score")
    arms = list(ARM_LABELS)
    rows = []
    for row in absolute.iter_rows(named=True):
        if row["arm"] not in arms:
            continue
        rows.extend(
            {
                "year": YEAR_LABELS[row["year"]],
                "arm": ARM_LABELS[row["arm"]],
                "site": site,
                "score": row[site],
            }
            for site in SITE_ORDER
            if row[site] is not None
        )
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[name] for name in arms]
    base = alt.Chart(table).encode(
        x=alt.X("site:N", sort=SITE_ORDER, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("arm:N", sort=order, title=None),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[1.7, 5.5], range=["#FFFFFF", ocf.DATA_BLUE], clamp=True),
            legend=alt.Legend(title="Score (points)", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text=alt.Text("score:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    return (
        (heat + text)
        .properties(width=WIDTH - 160, height=17 * len(order))
        .facet(column=alt.Column("year:N", sort=list(YEAR_LABELS.values()), title=None))
        .properties(
            title=_title(
                text="Quantile score of every headline arm at every substation",
                subtitle=(
                    "Twice the mean pinball loss over 11 quantile levels, in points of effective "
                    "capacity (smaller is better), day-ahead forecasts, on rows outside the "
                    "switching events. Scored year: 1 July 2025 to 30 June 2026. Replicate year: "
                    "1 July 2024 to 30 June 2025."
                ),
            )
        )
    )


def figure_4_lead(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The filter's gain and its two states by days ahead."""
    curve = _section(sections=sections, start="Quantile score by days ahead")
    absolute = _section(sections=sections, start="Absolute quantile score")
    a1 = {
        r["year"]: r["five sites QS"]
        for r in absolute.filter(pl.col("arm") == "a1").iter_rows(named=True)
    }
    table = curve.with_columns(
        arm_label=pl.col("arm").replace_strict({k: ARM_LABELS[k] for k in ("a2", "a3", "a4")}),
        year_label=pl.col("year").replace_strict(YEAR_LABELS),
    )
    table = table.with_columns(baseline=pl.col("year").replace_strict(a1, return_dtype=pl.Float64))
    days_axis = alt.Axis(values=list(range(1, 8)))
    lines = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("days ahead:Q", title="Days ahead of the forecast", axis=days_axis),
            y=alt.Y("QS:Q", title="Quantile score (points)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[k] for k in ("a2", "a3", "a4")],
                    range=[ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
        )
    )
    rule = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#333333", strokeDash=[4, 3])
        .encode(y="baseline:Q")
    )
    states = (
        alt.Chart(table.filter(pl.col("arm") == "a3"))  # ty: ignore[unresolved-attribute]
        .transform_fold(["mean |level|", "mean |transient|"], as_=["state", "points"])
        .mark_line(point=True)
        .encode(
            x=alt.X("days ahead:Q", title="Days ahead of the forecast", axis=days_axis),
            y=alt.Y("points:Q", title="Mean absolute correction (points)"),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["mean |level|", "mean |transient|"],
                    range=[ocf.DATA_GREEN, ocf.DATA_SKY],
                ),
                legend=alt.Legend(
                    title="Part of A3's correction", orient="bottom", direction="horizontal"
                ),
            ),
        )
    )
    year_column = alt.Column("year_label:N", sort=list(YEAR_LABELS.values()), title=None)
    top = (lines + rule).properties(width=WIDTH // 2 - 40, height=140).facet(column=year_column)
    bottom = states.properties(width=WIDTH // 2 - 40, height=110).facet(column=year_column)
    return alt.vconcat(top, bottom).properties(
        title=_title(
            text="The Kalman filter's gain by days ahead, and what it adds",
            subtitle=(
                "Top: quantile score over the five decision substations (smaller is better); the "
                "dashed line is A1, the same model without a filter. Bottom: mean absolute size "
                "of A3's correction, split into the persistent level and the transient. Weather "
                "is an estimate of past weather."
            ),
        )
    )


def figure_5_cadence(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Quantile score of frozen, 4-weekly and weekly refits, with and without the filter."""
    absolute = _section(sections=sections, start="Absolute quantile score")
    arms = {
        "a1": ("Frozen", "A1 (no filter)"),
        "a1_4weekly": ("4-weekly refits", "A1 (no filter)"),
        "a1_weekly": ("Weekly refits", "A1 (no filter)"),
        "a3": ("Frozen", "A3 (Kalman filter)"),
        "a3_4weekly": ("4-weekly refits", "A3 (Kalman filter)"),
        "a3_weekly": ("Weekly refits", "A3 (Kalman filter)"),
    }
    rows = [
        {
            "year": YEAR_LABELS[row["year"]],
            "cadence": arms[row["arm"]][0],
            "family": arms[row["arm"]][1],
            "score": row["five sites QS"],
        }
        for row in absolute.iter_rows(named=True)
        if row["arm"] in arms
    ]
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("cadence:N", sort=["Frozen", "4-weekly refits", "Weekly refits"], title=None),
            y=alt.Y("score:Q", title="Quantile score (points)"),
            xOffset=alt.XOffset("family:N"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(
                    domain=["A1 (no filter)", "A3 (Kalman filter)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH // 2 - 60, height=170)
        .facet(column=alt.Column("year:N", sort=list(YEAR_LABELS.values()), title=None))
    )
    return chart.properties(
        title=_title(
            text="Refitting helps the model without a filter, and does not help the filter",
            subtitle=(
                "Quantile score over the five decision substations (smaller is better) for the "
                "model fitted once at the start of the year, refitted every 4 weeks, and refitted "
                "every week."
            ),
        )
    )


def figure_6_tails(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Coverage at the four tail levels for the headline arms."""
    tails = _section(sections=sections, start="Tail calibration")
    nominal = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
    rows = [
        {
            "year": YEAR_LABELS[row["year"]],
            "arm": ARM_LABELS[row["arm"]],
            "level": level,
            "error": 100.0 * (row[f"{level} coverage"] - target),
        }
        for row in tails.iter_rows(named=True)
        for level, target in nominal.items()
    ]
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[k] for k in ("a0", "a1", "w", "a2", "a3", "a4")]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X("error:Q", title="Coverage minus nominal (percentage points; 0 is calibrated)"),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(
                    domain=list(nominal),
                    range=[ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title="Quantile level", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 200, height=130)
    )
    zero = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x=alt.datum(0))
    )
    return (
        (zero + chart)
        .facet(row=alt.Row("year:N", sort=list(YEAR_LABELS.values()), title=None))
        .properties(
            title=_title(
                text="Tail calibration of the headline arms",
                subtitle=(
                    "Share of observations at or below each tail quantile, minus the nominal "
                    "share, over the five decision substations. A point to the right of the "
                    "dashed line means the quantile sits too high."
                ),
            )
        )
    )


def figure_7_months() -> alt.TopLevelMixin:
    """Quantile score by month since the cut for the headline arms, scored year."""
    rows = []
    for name in ("a1", "a3", "w", "x_best"):
        frame = report.load_arm(label="scored", arm=name).filter(
            pl.col("site_name").is_in(SITE_ORDER)
        )
        monthly = frame.group_by(pl.col("valid_time").dt.truncate("1mo").alias("month")).agg(
            pl.col("qs").mean()
        )
        rows.extend(
            {"arm": ARM_LABELS[name], "month": month, "score": score}
            for month, score in monthly.iter_rows()
        )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month of the scored year", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("score:Q", title="Quantile score (points)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[k] for k in ("a1", "a3", "w", "x_best")],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, "#333333"],
                ),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 40, height=190)
    )
    return chart.properties(
        title=_title(
            text="Score by month through the scored year",
            subtitle=(
                "Quantile score over the five decision substations in each month from July 2025 "
                "to June 2026 (smaller is better). The models are frozen at 1 July 2025, so the "
                "months count the time since the last fit."
            ),
        )
    )


def figure_8_drift() -> alt.TopLevelMixin:
    """The fitted midday drift of A1 at each site, with the held values of two variants."""
    drift = pl.read_parquet(common.WORK_DIR / "grid_drift.parquet").filter(
        pl.col("cut") == "scored",
        pl.col("band") == 2,
        pl.col("arm").is_in(["a1", "a1_knot_pinned", "a1_smooth"]),
        pl.col("site").is_in(SITE_ORDER),
    )
    labels = {
        "a1": "A1 drift (free last knot)",
        "a1_knot_pinned": "Last knot tied to the one before",
        "a1_smooth": "A1 with smooth drift",
    }
    lines = drift.with_columns(
        date=(pl.col("day") * 86400.0 * 1000.0).cast(pl.Datetime("ms")),
        series=pl.col("arm").replace_strict(labels, return_dtype=pl.String),
        level=pl.col("drift") + pl.col("step_level"),
    ).select("site", "date", "series", "level", pl.lit(False).alias("held"))
    free = lines.filter(pl.col("series") == labels["a1"]).sort("date")
    held_rows = []
    for site in SITE_ORDER:
        path = free.filter(pl.col("site") == site)["level"].to_numpy()
        last = float(path[-1])
        shrunk = last + 0.5 * (float(path[-12:].mean()) - last)
        date_end = free.filter(pl.col("site") == site)["date"][-1]
        held_rows.extend(
            {"site": site, "date": date_end, "series": name, "level": value, "held": True}
            for name, value in (
                ("Held value: free last knot", last),
                ("Held value: shrunk by 0.5", shrunk),
            )
        )
    table = pl.concat([lines, pl.DataFrame(held_rows, schema=lines.schema)])
    domain = [
        *labels.values(),
        "Held value: free last knot",
        "Held value: shrunk by 0.5",
    ]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=domain,
            range=[
                ocf.BRAND_ORANGE,
                ocf.DATA_PURPLE,
                ocf.DATA_GREEN,
                ocf.BRAND_ORANGE,
                ocf.DATA_BLUE,
            ],
        ),
        legend=alt.Legend(title=None, orient="top", direction="vertical", columns=1),
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y"))
    y = alt.Y("level:Q", title=None)
    line = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .transform_filter(~alt.datum.held)
        .encode(x=x, y=y, color=colour)
    )
    points = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80, shape="diamond")
        .transform_filter(alt.datum.held)
        .encode(x=x, y=y, color=colour)
    )
    chart = (
        alt.layer(line, points, data=table)
        .properties(width=WIDTH - 40, height=70)
        .facet(
            row=alt.Row(
                "site:N",
                sort=SITE_ORDER,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
    )
    return chart.properties(
        title=_title(
            text="Fitted drift in the midday hours at each substation",
            subtitle=(
                "Monthly drift level (points of effective capacity) of fits made with data to "
                "30 June 2025, in the 11:00 to 15:59 hour band. A forecast holds the last value "
                "(diamonds, at the cut). Each fit's level is centred on its own profile, so the "
                "shape is comparable and the height is not."
            ),
        )
    )


def figure_9_gate() -> alt.TopLevelMixin:
    """The injected-farm movement of the drift in each hour band."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    rows = []
    for farm in gate["farm"]:
        rows.extend(
            {
                "site": farm["site"],
                "band": band.capitalize(),
                "movement": farm[f"{band} drift change (points)"],
            }
            for band in ("night", "morning", "midday", "evening")
        )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=["Night", "Morning", "Midday", "Evening"], title=None),
            y=alt.Y("movement:Q", title="Change in fitted drift (points)"),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(
                    domain=["Night", "Morning", "Midday", "Evening"],
                    range=[GREY, ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=None,
            ),
        )
        .properties(width=WIDTH // 4 - 20, height=150)
        .facet(column=alt.Column("site:N", title=None))
    )
    return chart.properties(
        title=_title(
            text="Known-answer gate: an injected solar farm moves the midday drift, not the night",
            subtitle=(
                "A synthetic farm of 30% of the substation's pinned capacity is switched on at "
                "1 July 2023 and subtracted from the real power. With the capacity pinned, the "
                "drift absorbs it in the hours when the farm produces (morning, midday, evening) "
                "and leaves the night band alone."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    sections = read_sections()
    _save(figure_1_weeks(), name="figure_01_weeks")
    _save(figure_2_headline(sections=sections), name="figure_02_headline")
    _save(figure_3_absolute(sections=sections), name="figure_03_absolute")
    _save(figure_4_lead(sections=sections), name="figure_04_lead")
    _save(figure_5_cadence(sections=sections), name="figure_05_cadence")
    _save(figure_6_tails(sections=sections), name="figure_06_tails")
    _save(figure_7_months(), name="figure_07_months")
    _save(figure_8_drift(), name="figure_08_drift")
    _save(figure_9_gate(), name="figure_09_gate")


if __name__ == "__main__":
    main()
