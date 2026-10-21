"""Draw the figures of experiment 149 as SVG (and a PNG to look at) beside its `result.md`.

Only Primary substations appear, under their D-labels. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00149-*/e00149_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00149_common as common
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
QUARTER_COLOURS: Final[dict[str, str]] = {
    "winter": ocf.DATA_BLUE,
    "summer": ocf.BRAND_ORANGE,
    "pooled": "#333333",
}
CONTRAST_TITLES: Final[dict[str, str]] = {
    "C2": "C2 (the verdict): all history minus 24 months, both with drift",
    "C1": "C1: all history minus 24 months, no drift",
    "C3": "C3: drift minus no drift, all history",
    "C4": "C4: drift minus no drift, 24 months",
}
ARM_LABELS: Final[dict[str, str]] = {
    "a24__s0": "24 months",
    "a24_d1__s0": "24 months + drift",
    "aall__s0": "All history",
    "aall_d1__s0": "All history + drift",
    "climatology__own": "Climatology",
    "manual_heuristic__own": "Manual heuristic",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
    GREY,
    "#333333",
]


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


def _quarter(*, origin: str) -> str:
    return "winter" if int(origin[5:7]) in (10, 1) else "summer"


def figure_1_headline() -> alt.TopLevelMixin:
    """One dot per origin for C1 to C4, with 95% interval and the pooled estimate."""
    panels = []
    for key, title in CONTRAST_TITLES.items():
        table = pl.read_parquet(TABLE_DIR / f"origin_{key}.parquet")
        pooled = pl.read_parquet(TABLE_DIR / f"pooled_{key}.parquet").filter(
            pl.col("metric") == "score"
        )
        rows = [
            {**row, "x": row["origin"][2:7], "quarter": _quarter(origin=row["origin"])}
            for row in table.iter_rows(named=True)
        ]
        rows.append(
            {
                "x": "pooled",
                "estimate": pooled["estimate"][0],
                "lower_95": pooled["lower"][0],
                "upper_95": pooled["upper"][0],
                "se": 0.0,
                "quarter": "pooled",
            }
        )
        frame = pl.DataFrame(rows, infer_schema_length=None)
        order = [row["x"] for row in rows]
        base = alt.Chart(frame).encode(
            x=alt.X("x:N", sort=order, title="Origin (year-month)" if key == "C4" else None)
        )
        band = (
            base.mark_bar(color="#EAEAEA", width=22)  # ty: ignore[unresolved-attribute]
            .encode(
                y=alt.Y("lower_band:Q", title="Score difference (points)"),
                y2="upper_band:Q",
            )
            .transform_calculate(lower_band="-datum.se", upper_band="datum.se")
        )
        colour = alt.Color(
            "quarter:N",
            scale=alt.Scale(domain=list(QUARTER_COLOURS), range=list(QUARTER_COLOURS.values())),
            legend=alt.Legend(title="Scored months", orient="top", direction="horizontal")
            if key == "C2"
            else None,
        )
        interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            y="lower_95:Q", y2="upper_95:Q", color=colour
        )
        dot = base.mark_point(filled=True, size=70, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            y="estimate:Q", color=colour
        )
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[3, 3])
            .encode(y="y:Q")
        )
        panels.append(
            (band + interval + dot + zero).properties(
                width=WIDTH - 60,
                height=95,
                title=alt.TitleParams(title, anchor="start", fontSize=11),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="All history beat 24 months at 7 of 12 origins, by 0.08 points, not reliably",
            subtitle=(
                "Difference in quantile score (twice the mean pinball loss over 11 quantiles, "
                "points of effective capacity) between two convex-model forecasts, at each of 12 "
                "quarterly origins and pooled at the right. Below zero favours the first-named "
                "forecast. Dot: estimate. Line: 95% interval from resampling one-week blocks "
                "(pooled: 98.75% interval from resampling months). Grey bar: tie band, one "
                "standard error. D5, D10, D11, D14, and D16 once each has 36 months of history."
            ),
        )
    )


def figure_2_absolute() -> alt.TopLevelMixin:
    """Each arm's quantile score per origin, one row of the chart per substation."""
    wide = pl.read_parquet(TABLE_DIR / "wide.parquet")
    columns = [f"score__{name}" for name in ARM_LABELS]
    table = (
        wide.group_by("site", "origin")
        .agg([pl.col(column).mean() for column in columns])
        .unpivot(index=["site", "origin"], variable_name="arm", value_name="score")
        .with_columns(
            arm=pl.col("arm").str.replace("score__", "").replace(ARM_LABELS),
            x=pl.col("origin").str.slice(2, 5),
        )
    )
    order = [common.cut_label(cut=c)[2:7] for c in common.SCORED_ORIGINS]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("x:N", sort=order, title="Origin (year-month)"),
            y=alt.Y("score:Q", title=None),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=3),
            ),
        )
        .properties(width=WIDTH - 60, height=80)
        .facet(row=alt.Row("site:N", sort=list(common.DECISION_SITES), title=None))
    )
    return chart.properties(
        title=_title(
            text="Drift lowers the score at every substation but D14; the window matters less",
            subtitle=(
                "Mean quantile score (points of effective capacity; smaller is better) in the 3 "
                "months after each origin, one row per substation. Convex-model spreads are the "
                "rolling-pool spread S0."
            ),
        )
    )


def figure_3_splits() -> alt.TopLevelMixin:
    """C2 and C1 by substation, with and without each substation, and by origin group."""
    panels = []
    for key in ("C2", "C1"):
        splits = pl.read_parquet(TABLE_DIR / f"splits_{key}.parquet").with_columns(
            kind=pl.when(pl.col("group").str.starts_with("D"))
            .then(pl.lit("one substation alone"))
            .when(pl.col("group").str.starts_with("20"))
            .then(pl.lit("origin year"))
            .when(pl.col("group").is_in(["winter", "summer"]))
            .then(pl.lit("origin quarter"))
            .when(pl.col("group").str.contains("rows"))
            .then(pl.lit("row season"))
            .otherwise(pl.lit("lead band"))
        )
        loso = (
            pl.read_parquet(TABLE_DIR / f"loso_{key}.parquet")
            .with_columns(
                group=pl.lit("without ") + pl.col("left_out"),
                kind=pl.lit("one substation left out"),
                n_rows=pl.col("n_rows"),
            )
            .select(splits.columns)
        )
        frame = pl.concat([splits, loso])
        order = frame["group"].to_list()
        base = alt.Chart(frame).encode(
            y=alt.Y("group:N", sort=order, title=None),
        )
        rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower:Q", title="Score difference (points)"),
            x2="upper:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    range=[
                        ocf.DATA_BLUE,
                        ocf.BRAND_ORANGE,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        GREY,
                        "#333333",
                    ]
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=3)
                if key == "C2"
                else None,
            ),
        )
        dot = base.mark_point(filled=True, size=50, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x="estimate:Q", color="kind:N"
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[3, 3])
            .encode(x="x:Q")
        )
        panels.append(
            (rule + dot + zero).properties(
                width=WIDTH - 130,
                height=15 * len(order),
                title=alt.TitleParams(CONTRAST_TITLES[key], anchor="start", fontSize=11),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="C2 is below zero in every split, but most intervals span zero",
            subtitle=(
                "Pooled difference in quantile score with a 95% interval from resampling months "
                "(98.75% for the leave-one-out rows). Below zero favours all history. C1, which "
                "has no drift term, changes sign with the substation left out and with the season."
            ),
        )
    )


def figure_4_tails() -> alt.TopLevelMixin:
    """Coverage of the four tail quantiles per arm, against the nominal level."""
    tails = pl.read_parquet(TABLE_DIR / "tails.parquet").filter(
        pl.col("arm").is_in(list(ARM_LABELS))
    )
    tails = tails.with_columns(arm_label=pl.col("arm").replace(ARM_LABELS))
    base = alt.Chart(tails).encode(
        x=alt.X("level:O", title="Quantile level"),
        color=alt.Color(
            "arm_label:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=3),
        ),
        xOffset="arm_label:N",
    )
    bars = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("coverage:Q", title="Coverage")
    )
    ranges = base.mark_rule().encode(y="cov_lo:Q", y2="cov_hi:Q")  # ty: ignore[unresolved-attribute]
    nominal = (
        alt.Chart(pl.DataFrame({"level": [0.01, 0.05, 0.95, 0.99], "v": [0.01, 0.05, 0.95, 0.99]}))
        .mark_tick(color="black", thickness=2, size=70)
        .encode(x="level:O", y="v:Q")  # ty: ignore[unresolved-attribute]
    )
    return (ranges + bars + nominal).properties(
        width=WIDTH - 60,
        height=200,
        title=_title(
            text="The convex arms cover less than nominal at the 95% and 99% quantiles",
            subtitle=(
                "Share of outcomes at or below each quantile, pooled over the scored rows, with a "
                "95% interval from resampling months. Black tick: nominal level. Closer to the "
                "tick is better."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    for name, function in (
        ("figure_01_headline", figure_1_headline),
        ("figure_02_absolute", figure_2_absolute),
        ("figure_03_splits", figure_3_splits),
        ("figure_04_tails", figure_4_tails),
    ):
        _save(function(), name=name)


if __name__ == "__main__":
    main()
