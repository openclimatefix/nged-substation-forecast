"""Print the numbers of experiment 157 and draw its figures as SVG into the docs folder.

Only Primary substations appear, under their D-labels; the metered farms appear only as ratios.
Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00157-*/e00157_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00157_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00157-differentiable-physics-joint-pv-capacity-known-answer"
)
PNG_DIR: Final[Path] = Path("/tmp/e00157_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
GATE_LOW: Final[float] = 0.95
GATE_HIGH: Final[float] = 1.05
BENCHMARK: Final[float] = 0.88
PRE_REGISTERED_SHARE: Final[float] = 0.1
GREY: Final[str] = "#808080"
Y_SCALE: Final[alt.Scale] = alt.Scale(domain=[0.5, 1.1])
READOUTS: Final[dict[str, str]] = {
    "two_stage": "Two-stage (experiment 129's arm 2)",
    "joint_year": "Joint, mean over the season-year (the gate)",
    "joint_last": "Joint, last month (what a forecast holds)",
    "joint_weighted": "Joint, mean weighted by squared solar shape (post hoc)",
}
READOUT_COLOURS: Final[list[str]] = [
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
]


def load_recovery() -> pl.DataFrame:
    """Return every Primary's recovery table."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"recovery_{s}.parquet") for s in common.ALL_SITES]
    )


def readouts(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Primary, farm, and read-out, with the recovery."""
    two = table.filter(pl.col("arm") == "two_stage").select(
        "site", "farm", readout=pl.lit("two_stage"), recovery="recovery_year"
    )
    joint = table.filter(pl.col("arm") == "joint", pl.col("share") == PRE_REGISTERED_SHARE)
    parts = [two]
    for name, column in (
        ("joint_year", "recovery_year"),
        ("joint_last", "recovery_last_month"),
        ("joint_weighted", "recovery_weighted"),
    ):
        parts.append(joint.select("site", "farm", readout=pl.lit(name), recovery=column))
    return pl.concat(parts)


def summary(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the median, minimum, and maximum over the five farms per Primary and read-out."""
    return (
        readouts(table=table)
        .group_by("site", "readout")
        .agg(
            median=pl.col("recovery").median(),
            low=pl.col("recovery").min(),
            high=pl.col("recovery").max(),
        )
        .sort("site", "readout")
    )


def print_numbers(*, table: pl.DataFrame) -> None:
    """Print the numbers the write-up quotes."""
    ceiling = pl.read_parquet(common.WORK_DIR / "ceiling.parquet")
    with pl.Config(tbl_rows=60, float_precision=3, tbl_cols=12):
        print(summary(table=table).pivot(on="readout", index="site", values="median"))
        print(summary(table=table).pivot(on="readout", index="site", values="low"))
        print(summary(table=table).pivot(on="readout", index="site", values="high"))
        print(
            ceiling.group_by("site")
            .agg(
                median=pl.col("ceiling").median(),
                low=pl.col("ceiling").min(),
                high=pl.col("ceiling").max(),
            )
            .sort("site")
        )
        sens = (
            table.filter(pl.col("arm") == "joint")
            .group_by("share", "site")
            .agg(
                year=pl.col("recovery_year").median(),
                last=pl.col("recovery_last_month").median(),
                weighted=pl.col("recovery_weighted").median(),
            )
            .sort("share", "site")
        )
        print(sens)
        decision = summary(table=table).filter(pl.col("site").is_in(common.DECISION_SITES))
        for readout in READOUTS:
            sub = decision.filter(pl.col("readout") == readout)
            in_band = sub.filter(pl.col("median").is_between(GATE_LOW, GATE_HIGH)).height
            above = sub.filter(pl.col("median") > BENCHMARK).height
            print(f"{readout}: in 0.95-1.05 at {in_band} of 5; above 0.88 at {above} of 5")
        print(
            table.filter(pl.col("arm") == "two_stage")
            .group_by("site")
            .agg(
                base_mw=(pl.col("year_mean_base") * pl.col("scale_mw_per_point")).median(),
                pinned_mw=(pl.col("pinned_points") * pl.col("scale_mw_per_point")).first(),
                injected_mw=(pl.col("injected_points") * pl.col("scale_mw_per_point")).first(),
                n_rows=pl.col("n_rows").median(),
            )
            .sort("site")
        )


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def figure_recovery(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery of the injected farm per Primary and read-out, with the gate band."""
    data = summary(table=table).with_columns(
        label=pl.col("readout").replace(READOUTS), site_order=pl.col("site")
    )
    ceiling = (
        pl.read_parquet(common.WORK_DIR / "ceiling.parquet")
        .group_by("site")
        .agg(ceiling=pl.col("ceiling").median())
    )
    sites = list(common.ALL_SITES)
    band = (
        alt.Chart(pl.DataFrame({"low": [GATE_LOW], "high": [GATE_HIGH]}))
        .mark_rect(color=ocf.DATA_GREEN, opacity=0.2)
        .encode(y=alt.Y("low:Q", scale=Y_SCALE), y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    bench = (
        alt.Chart(pl.DataFrame({"y": [BENCHMARK]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    scale = alt.Scale(domain=list(READOUTS.values()), range=READOUT_COLOURS)
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=list(READOUTS.values())),
        color=alt.Color(
            "label:N",
            scale=scale,
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    bars = base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "low:Q",
            title="Recovery (share of injected capacity)",
            scale=Y_SCALE,
        ),
        y2="high:Q",
    )
    points = base.mark_point(filled=True, size=45).encode(y=alt.Y("median:Q", scale=Y_SCALE))  # ty: ignore[unresolved-attribute]
    ticks = (
        alt.Chart(ceiling)
        .mark_tick(color="black", thickness=2, size=70)
        .encode(x=alt.X("site:N", sort=sites), y=alt.Y("ceiling:Q", scale=Y_SCALE))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(band, bench, bars, points, ticks).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="Fitting the capacity monthly does not push the recovery to 1",
            subtitle=(
                "Dot: median over five real farms injected at 50% of the Primary's pinned"
                " capacity. Line: range over the farms. Green band: pre-registered gate"
                " (0.95 to 1.05). Dashed line: 0.88, the best two-stage figure. Black tick:"
                " the share of the farm that one scale of the fixed solar shape fits"
                " (shape ceiling)."
                " Fits use rows before 2025-07-01."
            ),
        ),
    )


def figure_months(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the joint path's recovery by month of season-year 2024 at the decision Primaries."""
    capacity = table.group_by("site").agg(injected_points=pl.col("injected_points").first())
    paths = pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"paths_{s}.parquet") for s in common.DECISION_SITES]
    ).filter(pl.col("share") == PRE_REGISTERED_SHARE)
    first = (common.LATEST_SEASON_YEAR - 1970) * 12 + 6
    wide = paths.pivot(on="fit", index=["site", "farm", "month"], values="level").join(
        capacity, on="site"
    )
    data = (
        wide.filter(pl.col("month").is_between(first, first + 11))
        .with_columns(
            recovery=(pl.col("injected") - pl.col("base")) / pl.col("injected_points"),
            position=pl.col("month") - first,
        )
        .group_by("site", "position")
        .agg(recovery=pl.col("recovery").median())
        .sort("site", "position")
    )
    names = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    scale = alt.Scale(
        domain=list(common.DECISION_SITES),
        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.DATA_GREEN],
    )
    line = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "position:O",
                title="Month of season-year 2024 (July 2024 to June 2025)",
                axis=alt.Axis(labelAngle=0, labelExpr=f"{names}[datum.value]"),
            ),
            y=alt.Y("recovery:Q", title="Recovery (share of injected capacity)"),
            color=alt.Color("site:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
        )
    )
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(one, line).properties(
        width=WIDTH,
        height=220,
        title=chart_title(
            text="The joint path finds the farm in summer and misses it in winter",
            subtitle=(
                "Line: median over five real farms of the joint fit's monthly capacity with the"
                " farm injected, minus without it, as a share of the injected capacity."
                " Dashed line: full recovery."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write both figures."""
    table = load_recovery()
    print_numbers(table=table)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_recovery(table=table), name="figure_01_recovery")
    save_chart(figure_months(table=table), name="figure_02_months")


if __name__ == "__main__":
    main()
