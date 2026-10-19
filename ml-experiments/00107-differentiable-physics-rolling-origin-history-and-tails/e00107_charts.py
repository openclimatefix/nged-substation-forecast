"""Draw the figures of experiment 107 as SVG (and a PNG to look at) beside its `result.md`.

Only Primary substations appear, under their D-labels, so dates may be shown. Afterwards optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00107-*/e00107_charts.py
"""

import json
import textwrap
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00107_common as common
import plotting.ocf_theme as ocf
import polars as pl

FIGURE_DIR: Final[Path] = (
    Path(__file__).parents[2]
    / "docs"
    / "ml-experiments"
    / (f"{common.EXPERIMENT_ID:05d}-{common.SLUG}")
)
TABLE_DIR: Final[Path] = common.WORK_DIR / "tables"
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#808080"
QUARTER_COLOURS: Final[dict[str, str]] = {"winter": ocf.DATA_BLUE, "summer": ocf.BRAND_ORANGE}
GROUP_SHAPES: Final[dict[str, str]] = {
    "pre-tuning": "triangle-up",
    "in-tuning": "circle",
    "post-tuning": "square",
}
ARM_LABELS: Final[dict[str, str]] = {
    "g1__s0": "G1",
    "g_all__s0": "G-all",
    "g_all_d1__s0": "G-all+D1",
    "x_12_nodoy__own": "XGBoost bar",
    "climatology__own": "Climatology",
    "manual_heuristic__own": "Manual heuristic",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    GREY,
    "#333333",
]
CONTRAST_TITLES: Final[dict[str, str]] = {
    "C1": "C1: all years without drift minus 12 months (G-all minus G1)",
    "C2": "C2: drift minus no drift on all years (G-all+D1 minus G-all)",
    "C3": "C3: all years with drift minus 12 months (G-all+D1 minus G1)",
    "C4": "C4: 12-month convex model minus the XGBoost bar (G1 minus bar)",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / f"{name}.svg")
    png_dir = common.WORK_DIR / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    chart.save(png_dir / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _origin_meta() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "origin": common.cut_label(cut=cut),
                "quarter": common.quarter_type(cut=cut),
                "group": common.causality_group(cut=cut),
            }
            for cut in common.SCORED_ORIGINS
        ]
    )


def figure_1_headline() -> alt.TopLevelMixin:
    """One dot per origin for C1 to C4, with 95% interval, tie band, and the pooled estimate."""
    labels = json.loads((TABLE_DIR / "labels.json").read_text())
    pooled = json.loads((TABLE_DIR / "pooled.json").read_text())
    meta = _origin_meta()
    panels = []
    for name, title in CONTRAST_TITLES.items():
        table = pl.read_parquet(TABLE_DIR / f"origin_score_{name}.parquet").join(meta, on="origin")
        pool = pooled[name]["pooled"]["all 12"]
        rows = [{**row, "x": row["origin"][2:7]} for row in table.iter_rows(named=True)] + [
            {
                "x": "all 12",
                "estimate": pool["estimate"],
                "lower_95": pool["lower"],
                "upper_95": pool["upper"],
                "band": 0.0,
                "quarter": "pooled",
                "group": "pooled",
            }
        ]
        frame = pl.DataFrame(rows)
        order = [row["x"] for row in rows]
        base = alt.Chart(frame).encode(
            x=alt.X(
                "x:N",
                sort=order,
                title="Origin (forecast cut, year-month)" if name == "C4" else None,
            )
        )
        band = (
            base.mark_bar(color="#EAEAEA", width=22)  # ty: ignore[unresolved-attribute]
            .encode(
                y=alt.Y("lower_band:Q", title="Quantile-score difference (points)"),
                y2="upper_band:Q",
            )
            .transform_calculate(lower_band="-datum.band", upper_band="datum.band")
        )
        interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            y="lower_95:Q",
            y2="upper_95:Q",
            color=alt.Color(
                "quarter:N",
                scale=alt.Scale(
                    domain=["winter", "summer", "pooled"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#333333"],
                ),
                legend=alt.Legend(
                    title="Origin's scored months", orient="top", direction="horizontal"
                )
                if name == "C1"
                else None,
            ),
        )
        dot = base.mark_point(filled=True, size=70, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            y="estimate:Q",
            color=alt.Color("quarter:N", legend=None),
            shape=alt.Shape(
                "group:N",
                scale=alt.Scale(
                    domain=["pre-tuning", "in-tuning", "post-tuning", "pooled"],
                    range=["triangle-up", "circle", "square", "diamond"],
                ),
                legend=alt.Legend(title="Penalties tuned", orient="top", direction="horizontal")
                if name == "C1"
                else None,
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[3, 3])
            .encode(y="y:Q")
        )
        label = labels[name]["label"]
        panels.append(
            (band + interval + dot + zero).properties(
                width=WIDTH - 60,
                height=95,
                title=alt.TitleParams(f"{title}: {label}", anchor="start", fontSize=11),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Drift lowered the score at 10 of 12 origins; old years alone raised it at 8",
            subtitle=(
                "Difference in quantile score (twice the mean pinball loss over 11 quantiles, "
                "points of effective capacity) between two convex-model forecasts, at each of 12 "
                "quarterly origins and pooled over all 12 at the right. Below zero favours the "
                "first-named forecast. Dot: estimate. Line: 95% interval from resampling one-week "
                "blocks (pooled: 99% interval from resampling months). Grey bar: tie band, one "
                "standard error. Five substations, rows outside switching events."
            ),
        )
    )


def _scored(*, names: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    folder = common.WORK_DIR / "scored"
    return {name: pl.read_parquet(folder / f"{name}.parquet") for name in names}


def figure_2_absolute() -> alt.TopLevelMixin:
    """Each arm's quantile score per origin, one row of the chart per substation."""
    table = pl.read_parquet(TABLE_DIR / "absolute.parquet").filter(
        pl.col("arm").is_in(list(ARM_LABELS))
    )
    table = table.with_columns(
        arm_label=pl.col("arm").replace(ARM_LABELS), x=pl.col("origin").str.slice(2, 5)
    )
    order = [o[2:7] for o in [common.cut_label(cut=c) for c in common.SCORED_ORIGINS]]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("x:N", sort=order, title="Origin (forecast cut, year-month)"),
            y=alt.Y("score:Q", title=None),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=3),
            ),
        )
        .properties(width=WIDTH - 60, height=80)
        .facet(row=alt.Row("site:N", sort=list(common.DECISION_SITES), title=None))
    )
    return chart.properties(
        title=_title(
            text="Forecast rankings move from origin to origin",
            subtitle=(
                "Mean quantile score (points of effective capacity; smaller is better) in the 3 "
                "months after each origin, one row per substation. Convex-model spreads are the "
                "rolling-pool spread S0. The XGBoost bar has no day-of-year column."
            ),
        )
    )


def figure_3_leave_one_out() -> alt.TopLevelMixin:
    """How many origins fall on each side of zero for each contrast, leaving each site out."""
    pooled = json.loads((TABLE_DIR / "pooled.json").read_text())
    rows = []
    for name in CONTRAST_TITLES:
        full = pl.read_parquet(TABLE_DIR / f"origin_score_{name}.parquet")
        counts = {
            "none left out": (
                int((full["side"] == -1).sum()),
                int((full["side"] == 1).sum()),
                int((full["side"] == 0).sum()),
            )
        } | {f"without {site}": tuple(c) for site, c in pooled[name]["loso_counts"].items()}
        for left_out, (negative, positive, tied) in counts.items():
            for side, count in (("negative", negative), ("tied", tied), ("positive", positive)):
                rows.append(
                    {"contrast": name, "left_out": left_out, "side": side, "origins": count}
                )
    frame = pl.DataFrame(rows)
    order = ["none left out"] + [f"without {s}" for s in common.DECISION_SITES]
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("left_out:N", sort=order, title=None),
            x=alt.X("origins:Q", title="Origins (of 12)", scale=alt.Scale(domain=[0, 12])),
            color=alt.Color(
                "side:N",
                sort=["negative", "tied", "positive"],
                scale=alt.Scale(
                    domain=["negative", "tied", "positive"],
                    range=[ocf.DATA_BLUE, "#CCCCCC", ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(
                    title="Side of zero (outside the tie band)",
                    orient="top",
                    direction="horizontal",
                ),
            ),
            order=alt.Order("side:N", sort="ascending"),
        )
        .properties(width=WIDTH - 130, height=110)
        .facet(row=alt.Row("contrast:N", title=None))
    )
    return chart.properties(
        title=_title(
            text="Leaving out one substation changes how many origins favour each side",
            subtitle=(
                "For each planned contrast, the number of the 12 origins whose estimate is below "
                "zero (first-named forecast better), tied, or above zero, with all five decision "
                "substations and with each left out in turn."
            ),
        )
    )


def _coverage_by_origin(*, name: str, level: float) -> pl.DataFrame:
    frame = _scored(names=(name,))[name].filter(
        pl.col("site").is_in(common.DECISION_SITES), ~pl.col("switching_mask")
    )
    column = f"q{round(level * 100):02d}"
    return (
        frame.group_by("origin")
        .agg(coverage=(pl.col("actual") <= pl.col(column)).cast(pl.Float64).mean())
        .with_columns(level=pl.lit(f"q{round(level * 100)}"), arm=pl.lit(name))
        .sort("origin")
    )


def figure_4_tails() -> alt.TopLevelMixin:
    """Upper-quantile coverage per origin under the three spreads, with the nominal level."""
    names = {
        "g_all_d1__s0": "S0 (hour band and quarter)",
        "g_all_d1__s1": "S1 (adds level tercile)",
    }
    frames = [
        _coverage_by_origin(name=name, level=level).with_columns(
            spread=pl.lit(label), nominal=pl.lit(level)
        )
        for name, label in names.items()
        for level in (0.95, 0.99)
    ]
    table = pl.concat(frames).with_columns(
        x=pl.col("origin").str.slice(2, 5),
        coverage=pl.col("coverage") * 100.0,
        nominal=pl.col("nominal") * 100.0,
    )
    order = [o[2:7] for o in [common.cut_label(cut=c) for c in common.SCORED_ORIGINS]]
    base = alt.Chart(table).encode(
        x=alt.X("x:N", sort=order, title="Origin (forecast cut, year-month)")
    )
    lines = base.mark_line(point=True, strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "coverage:Q",
            title="Coverage (%)",
            scale=alt.Scale(domain=[70, 100]),
        ),
        color=alt.Color(
            "spread:N",
            scale=alt.Scale(
                domain=list(names.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, GREY]
            ),
            legend=alt.Legend(
                title=None, orient="top", direction="horizontal", columns=3, labelLimit=250
            ),
        ),
    )
    nominal = alt.Chart(table).mark_rule(color="black", strokeDash=[3, 3]).encode(y="nominal:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(lines, nominal)
        .properties(width=WIDTH - 60, height=110)
        .facet(row=alt.Row("level:N", sort=["q95", "q99"], title=None))
        .properties(
            title=_title(
                text="Neither spread brings the 95% and 99% quantiles to their nominal coverage",
                subtitle=(
                    "Coverage of the upper quantiles of the all-years convex model with drift "
                    "(G-all+D1) per origin, under two spreads, five substations. The dashed line "
                    "is the nominal level (95% and 99%); below it, outcomes exceed the quantile "
                    "too often."
                ),
            )
        )
    )


def figure_5_weeks() -> alt.TopLevelMixin:
    """Actual power and three forecasts in a summer and a winter week at three substations."""
    weeks = {
        "Summer: 14 to 20 July 2025": date(2025, 7, 14),
        "Winter: 12 to 18 January 2026": date(2026, 1, 12),
    }
    names = {
        "g1__s0": "G1",
        "g_all_d1__s0": "G-all+D1",
        "x_12_nodoy__own": "XGBoost bar",
    }
    arms = _scored(names=tuple(names))
    frames = []
    for week, start in weeks.items():
        for name, label in names.items():
            part = arms[name].filter(
                pl.col("site").is_in(["D5", "D14", "D16"]),
                pl.col("valid_time").dt.date() >= start,
                pl.col("valid_time").dt.date() < pl.lit(start) + pl.duration(days=7),
            )
            day = (
                part["valid_time"]
                - pl.Series(
                    [datetime(start.year, start.month, start.day, tzinfo=UTC)] * part.height
                )
            ).dt.total_hours() / 24.0
            frames.append(
                part.select("site", value=pl.col("mean").cast(pl.Float64)).with_columns(
                    week=pl.lit(week), series=pl.lit(label), day=day
                )
            )
            if name == "g1__s0":
                frames.append(
                    part.select("site", value=pl.col("actual").cast(pl.Float64)).with_columns(
                        week=pl.lit(week), series=pl.lit("Measured"), day=day
                    )
                )
    table = pl.concat(frames)
    order = ["Measured", "G1", "G-all+D1", "XGBoost bar"]

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
                        range=["#333333", ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
                    ),
                    legend=alt.Legend(title="Line", orient="top", direction="horizontal")
                    if legend
                    else None,
                ),
            )
            .properties(width=WIDTH - 40, height=60)
            .facet(row=alt.Row("site:N", sort=["D5", "D14", "D16"], title=None))
            .properties(title=week)
        )

    return alt.vconcat(
        *[week_chart(week, legend=position == 0) for position, week in enumerate(weeks)]
    ).properties(
        title=_title(
            text="Net power and three forecasts in a summer and a winter week",
            subtitle=(
                "Half-hourly net power as a percentage of each substation's effective capacity "
                "(negative is export). Forecast lines are each fit's mean forecast (the median for "
                "XGBoost), fitted on data up to 1 July 2025 (summer) and 1 January 2026 (winter)."
            ),
        )
    )


def figure_6_exploratory() -> alt.TopLevelMixin:
    """Pooled intervals of the controls, the replication column, and the tail-model contrasts."""
    pooled = json.loads((TABLE_DIR / "pooled.json").read_text())
    descriptions = {
        "E1": "Fitted capacity: G-all minus G1",
        "E2": "Fitted capacity: G-all+D1 minus G-all",
        "E3": "Fitted capacity: G-all+D1 minus G1",
        "E4": "Drift step 10,000 minus 3,000",
        "K1": "Control: G1 without day-of-year harmonics",
        "K2": "Control: G-all+D1 without harmonics",
        "K3": "Control: XGBoost with minus without day of year",
        "E5": "XGBoost: solar offset minus plain (X-12amq minus X-12)",
        "T2": "Tails: S1 minus S0 on G1",
    }
    rows = [
        {
            "contrast": label,
            "estimate": pooled[name]["pooled"]["all 12"]["estimate"],
            "low": pooled[name]["pooled"]["all 12"]["lower"],
            "high": pooled[name]["pooled"]["all 12"]["upper"],
        }
        for name, label in descriptions.items()
    ]
    frame = pl.DataFrame(rows)
    order = [row["contrast"] for row in rows]
    base = alt.Chart(frame).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    return (
        base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q",
                title="Quantile-score difference (points; below zero favours the first-named)",
            )
        )
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[3, 3])
        .encode(x="x:Q")
    ).properties(
        width=WIDTH - 200,
        height=260,
        title=_title(
            text="Day-of-year terms help the convex model with drift and hurt XGBoost",
            subtitle=(
                "Pooled over 12 origins and five substations, all exploratory (95% interval from "
                "resampling whole months). Dot: estimate. Line: 95% interval."
            ),
        ),
    )


def figure_7_d16_bias() -> alt.TopLevelMixin:
    """Mean forecast error at D16 per origin for G1, G-all, and G-all+D1."""
    names = {"g1__s0": "G1", "g_all__s0": "G-all", "g_all_d1__s0": "G-all+D1"}
    arms = _scored(names=tuple(names))
    parts = []
    for name, label in names.items():
        parts.append(
            arms[name]
            .filter(pl.col("site") == "D16", ~pl.col("switching_mask"))
            .group_by("origin")
            .agg(bias=(pl.col("actual") - pl.col("mean")).mean())
            .with_columns(arm=pl.lit(label), x=pl.col("origin").str.slice(2, 5))
        )
    table = pl.concat(parts)
    order = [o[2:7] for o in [common.cut_label(cut=c) for c in common.SCORED_ORIGINS]]
    base = alt.Chart(table).encode(
        x=alt.X("x:N", sort=order, title="Origin (forecast cut, year-month)")
    )
    return (
        base.mark_line(point=True, strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("bias:Q", title="Measured minus mean forecast (points)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(names.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        + alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[3, 3])
        .encode(y="y:Q")
    ).properties(
        width=WIDTH - 60,
        height=180,
        title=_title(
            text="At D16 forecasts without drift miss low by 3 to 4 points on average",
            subtitle=(
                "Mean of measured minus forecast power at D16 in the 3 months after each "
                "origin, in "
                "points of effective capacity. Above zero means demand was higher than forecast."
            ),
        ),
    )


def figure_8_capacity() -> alt.TopLevelMixin:
    """Pinned and fitted PV capacity at each cut."""
    table = pl.read_parquet(common.WORK_DIR / "capacity_sources.parquet").filter(
        pl.col("site").is_in(common.DECISION_SITES)
    )
    long = pl.concat(
        [
            table.select(
                "site", "origin", "pinned_source", value=pl.col("pinned_held_points")
            ).with_columns(kind=pl.lit("Pinned (real-time vintage)")),
            table.select(
                "site", "origin", "pinned_source", value=pl.col("fitted_last_capacity_points")
            ).with_columns(kind=pl.lit("Fitted (G-all+D1, fitted capacity)")),
        ]
    ).with_columns(x=pl.col("origin").str.slice(2, 5))
    order = [c.date().isoformat()[2:7] for c in common.CUTS]
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("x:N", sort=order, title="Cut (year-month)"),
            y=alt.Y("value:Q", title=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=[
                        "Pinned (real-time vintage)",
                        "Fitted (G-all+D1, fitted capacity)",
                    ],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
        .properties(width=WIDTH - 60, height=80)
        .facet(row=alt.Row("site:N", sort=list(common.DECISION_SITES), title=None))
        .properties(
            title=_title(
                text="Pinned PV capacity is above the fitted capacity at four of five substations",
                subtitle=(
                    "PV capacity at each cut, points of effective capacity, one row per "
                    "substation; the gap reaches 17 points at D5. "
                    "The first four cuts use a growth-prior level scaled back from a later vintage."
                ),
            )
        )
    )


def main() -> None:
    """Draw every figure."""
    for name, function in (
        ("figure_01_headline", figure_1_headline),
        ("figure_02_absolute", figure_2_absolute),
        ("figure_03_leave_one_out", figure_3_leave_one_out),
        ("figure_04_tails", figure_4_tails),
        ("figure_05_weeks", figure_5_weeks),
        ("figure_06_exploratory", figure_6_exploratory),
        ("figure_07_d16_bias", figure_7_d16_bias),
        ("figure_08_capacity", figure_8_capacity),
    ):
        _save(function(), name=name)


if __name__ == "__main__":
    main()
