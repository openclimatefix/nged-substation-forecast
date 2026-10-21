"""Draw the figures of experiment 129 as SVG into the docs folder, from the `work/` tables.

Only Primary substations appear, under their D-labels; the metered farms appear only as ratios.
Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00129-*/e00129_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00129_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00129-differentiable-physics-pv-capacity-known-answer-test"
)
PNG_DIR: Final[Path] = Path("/tmp/e00129_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
ARM_COLOURS: Final[dict[str, str]] = {
    "arm2": ocf.DATA_BLUE,
    "arm4@0.1": ocf.BRAND_ORANGE,
    "arm3": ocf.DATA_GREEN,
}
ARM_LABELS: Final[dict[str, str]] = {
    "arm2": "Arm 2: capacity steps up each season-year",
    "arm4@0.1": "Arm 4: steps penalised",
    "arm3": "Arm 3: arm 2 plus sun-blind demand terms",
}
GREY: Final[str] = "#808080"


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def figure_capacity() -> alt.TopLevelMixin:
    """Return the unmetered solar capacity per Primary and season-year, with 80% ranges."""
    table = pl.concat(
        [
            pl.read_parquet(common.WORK_DIR / f"real_{arm}_to_july.parquet")
            for arm in ("arm2", "arm4@0.1")
        ]
    ).with_columns(
        year=pl.format("{}/{}", pl.col("season_year"), (pl.col("season_year") + 1) % 100)
    )
    arm_scale = alt.Scale(domain=["arm2", "arm4@0.1"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
    base = alt.Chart().encode(
        x=alt.X(
            "year:O",
            title="Season-year (July to June); 2025/26 also holds July 2026",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "arm:N",
            scale=arm_scale,
            legend=alt.Legend(
                title=None,
                orient="top",
                labelExpr="datum.value == 'arm2' ? 'Arm 2: steps up each year'"
                " : 'Arm 4: steps penalised'",
            ),
        ),
        xOffset=alt.XOffset("arm:N"),
    )
    bars = base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("range_low_mw:Q", title="AC capacity (MW)"), y2="range_high_mw:Q"
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        y="estimate_mw:Q"
    )
    pinned = (
        alt.Chart().mark_rule(strokeDash=[4, 3], color=GREY).encode(y="pinned_mw:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(bars, points, pinned)
        .properties(width=WIDTH, height=110)
        .facet(
            row=alt.Row(
                "site:N",
                sort=list(common.DECISION_SITES),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            data=table,
        )
        .properties(
            title=chart_title(
                text="Unmetered solar behind each Primary has grown; ranges are 1.2 to 1.6 MW wide",
                subtitle=(
                    "Dot: estimate of AC capacity in MW. Line: range from resampling whole months,"
                    " divided by the recoveries measured on known farms and plants at other"
                    " Primaries, so a spread over test scenarios and not a calibrated 80%"
                    " interval. Grey dashed line: experiment 45's pinned capacity, a control and"
                    " not a truth."
                ),
            )
        )
    )


def figure_recovery() -> alt.Chart:
    """Return arm 2's recovery of injected solar at 100% of pinned capacity, one dot per Primary."""
    kat = pl.read_parquet(common.WORK_DIR / "kat_arm2.parquet").filter(
        pl.col("season_year") == common.COMPLETE_SEASON_YEARS[-1], pl.col("fraction") == 1.0
    )
    plants = {
        "matched": "Real farm",
        "east_west": "East-west plant",
        "south_west": "South-west plant",
    }
    table = (
        kat.group_by("site", "plant", "cloud_load")
        .agg(recovery=pl.col("recovery").median())
        .with_columns(
            plant=pl.col("plant").replace(plants),
            load=pl.when(pl.col("cloud_load"))
            .then(pl.lit("With cloud-driven load"))
            .otherwise(pl.lit("No load")),
        )
    )
    return (
        alt.Chart(table)
        .mark_point(filled=True, size=70, opacity=0.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "recovery:Q",
                title="Share of the injected capacity recovered (1 is exact; no divisor applied)",
                scale=alt.Scale(domain=[0.6, 1.3]),
            ),
            y=alt.Y("plant:N", sort=list(plants.values()), title=None),
            color=alt.Color(
                "plant:N",
                scale=alt.Scale(
                    domain=list(plants.values()),
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=None,
            ),
            row=alt.Row("load:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 150,
            height=80,
            title=chart_title(
                text="Arm 2 recovers about 80% of injected solar; a cloud-driven load adds 20%",
                subtitle=(
                    "One dot per Primary, season-year 2024/25, injection at 100% of the pinned"
                    " capacity. Recovery is the same at 20% and 50%. The cloud-driven load is a"
                    " day-long demand that rises on overcast days, of 10% of the injected"
                    " capacity, which the fit partly reads as solar."
                ),
            ),
        )
    )


def figure_step() -> alt.Chart:
    """Return the step-up injection: recovery per season-year by arm."""
    step = pl.read_parquet(common.WORK_DIR / "step.parquet")
    table = (
        step.group_by("arm", "season_year")
        .agg(recovery=pl.col("recovery").median())
        .with_columns(
            year=pl.format("{}/{}", pl.col("season_year"), (pl.col("season_year") + 1) % 100)
        )
        .filter(pl.col("arm").is_in(["arm2", "arm3", "arm4@0.1", "arm4@1.0"]))
    )
    return (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "year:O",
                title="Season-year (July to June); the step is on 1 March 2024, in 2023/24",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("recovery:Q", title="Share of the added capacity recovered"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["arm2", "arm3", "arm4@0.1", "arm4@1.0"],
                    range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelExpr="datum.value == 'arm2' ? 'Arm 2' : datum.value == 'arm3' ? 'Arm 3'"
                    " : datum.value == 'arm4@0.1' ? 'Arm 4, penalty 0.1' : 'Arm 4, penalty 1.0'",
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=chart_title(
                text="A step penalty makes arm 4 follow a sudden rise in capacity more slowly",
                subtitle=(
                    "Median over five Primaries. Half of the pinned capacity was added from "
                    "March 2024. Arm 4 spreads that rise over earlier years, more so with a "
                    "larger penalty."
                ),
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    save_chart(figure_capacity(), name="figure_01_capacity")
    save_chart(figure_recovery(), name="figure_02_recovery")
    save_chart(figure_step(), name="figure_03_step")


if __name__ == "__main__":
    main()
