"""Print every number of experiment 231's write-up and draw its figures.

uv run --no-sync python ml-experiments/00231-*/e00231_report.py
"""

import statistics
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00231_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00231-pv-capacity-audit-where-recovery-goes"
)
PNG_DIR: Final[Path] = Path("/tmp/e00231_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
RUNGS: Final[dict[str, str]] = {
    "r1": "1 Farm only",
    "r2": "2 Farm only, clear half-hours",
    "r3": "3 Demand frozen",
    "r4": "4 Full fit (reported)",
}
CAUSES: Final[dict[str, str]] = {
    "shape": "Plant shape (1 - rung 2)",
    "irradiance": "Irradiance input (rung 2 - rung 1)",
    "noise": "Estimator noise (rung 1 - rung 3)",
    "demand": "Demand absorption (rung 3 - rung 4)",
}
CAUSE_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN, ocf.BRAND_ORANGE]
ROWS: Final[dict[str, str]] = {
    "r4": "All rows",
    "r5_summer": "April to August rows only",
    "r5_winter": "November to February rows only",
}
ROW_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
SEASON_ORDER: Final[list[str]] = ["year", "spring", "summer", "autumn", "winter"]
SEASON_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
]
PREDICTION: Final[dict[str, float]] = {
    "demand": 0.40,
    "irradiance": 0.25,
    "season": 0.15,
    "divisor": 0.15,
}
"""The pre-registered thresholds: demand and irradiance at least, season and divisor under."""


def load_ladder() -> pl.DataFrame:
    """Return every Primary's ladder table in long form."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"ladder_{s}.parquet") for s in common.ALL_SITES]
    )


def site_values(*, ladder: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each rung, and the shares.

    Resample 0 is the point fit. Components and the season gain are differences of the medians.
    """
    rungs = ["r1", "r2", "r3", "r4", "r5_summer", "r5_winter"]
    medians = (
        ladder.filter(pl.col("rung").is_in(rungs))
        .group_by("site", "rung", "b")
        .agg(pl.col("value").median())
        .pivot(on="rung", index=["site", "b"], values="value")
    )
    return medians.with_columns(
        shape=1 - pl.col("r2"),
        irradiance=pl.col("r2") - pl.col("r1"),
        noise=pl.col("r1") - pl.col("r3"),
        demand=pl.col("r3") - pl.col("r4"),
        total=1 - pl.col("r4"),
        season=pl.col("r5_summer") - pl.col("r4"),
    ).with_columns(
        **{f"{name}_share": pl.col(name) / pl.col("total") for name in CAUSES},
        season_share=pl.col("season") / pl.col("total"),
    )


def pooled(*, values: pl.DataFrame) -> pl.DataFrame:
    """Return the median over the five decision Primaries, per resample, of every column."""
    columns = [c for c in values.columns if c not in {"site", "b"}]
    return (
        values.filter(pl.col("site").is_in(common.DECISION_SITES))
        .group_by("b")
        .agg(pl.col(c).median() for c in columns)
        .with_columns(site=pl.lit("Pooled"))
    )


def intervals(*, values: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Return the point value and the 10th to 90th percentile over resamples, per site."""
    point = values.filter(pl.col("b") == 0).select("site", *columns)
    resampled = values.filter(pl.col("b") > 0)
    return point.join(
        resampled.group_by("site").agg(
            [pl.col(c).quantile(LOW_QUANTILE).alias(f"{c}_lo") for c in columns]
            + [pl.col(c).quantile(HIGH_QUANTILE).alias(f"{c}_hi") for c in columns]
        ),
        on="site",
    )


def divisor_table() -> pl.DataFrame:
    """Return the p99 ratio per Primary and window: median over farms and the range."""
    divisor = pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"divisor_{s}.parquet") for s in common.ALL_SITES]
    )
    return (
        divisor.group_by("site", "window")
        .agg(
            pl.col("ratio").median().alias("ratio"),
            pl.col("ratio").min().alias("ratio_lo"),
            pl.col("ratio").max().alias("ratio_hi"),
        )
        .sort("site", "window")
    )


def reproduction(*, ladder: pl.DataFrame) -> pl.DataFrame:
    """Compare the injected-fit capacity with experiment 129's, per Primary and farm."""
    mine = (
        ladder.filter(pl.col("rung").is_in(["cap_base", "cap_injected", "injected_points"]))
        .pivot(on="rung", index=["site", "farm"], values="value")
        .rename({"cap_base": "base_here", "cap_injected": "injected_here"})
    )
    reference = (
        pl.read_parquet(common.KNOWN_ANSWER_PATH)
        .filter(
            pl.col("arm") == "arm2",
            pl.col("plant") == "matched",
            ~pl.col("cloud_load"),
            pl.col("fraction") == common.INJECTED_SHARE,
            pl.col("season_year") == common.READ_OUT_SEASON_YEAR,
            pl.col("source").is_in(list(common.FARMS)),
        )
        .unique(subset=["site", "source"])
        .select(
            "site",
            farm="source",
            base_129="capacity_base",
            injected_129="capacity_injected",
            recovery_129="recovery",
        )
    )
    return (
        mine.join(reference, on=["site", "farm"])
        .with_columns(
            injected_relative_difference=(pl.col("injected_here") - pl.col("injected_129")).abs()
            / pl.col("injected_129"),
            recovery_here=(pl.col("injected_here") - pl.col("base_here"))
            / pl.col("injected_points"),
            recovery_with_129_base=(pl.col("injected_here") - pl.col("base_129"))
            / pl.col("injected_points"),
        )
        .with_columns(
            recovery_gap=(pl.col("recovery_here") - pl.col("recovery_129")).abs(),
            recovery_gap_with_129_base=(
                pl.col("recovery_with_129_base") - pl.col("recovery_129")
            ).abs(),
        )
    )


def print_numbers(*, ladder: pl.DataFrame, values: pl.DataFrame) -> None:
    """Print the tables the write-up quotes."""
    with pl.Config(tbl_rows=60, tbl_cols=30, float_precision=3, tbl_width_chars=200):
        all_values = pl.concat([values, pooled(values=values).select(values.columns)])
        rung_columns = ["r1", "r2", "r3", "r4", "r5_summer", "r5_winter"]
        print("Rungs (median over farms; 10th and 90th percentile over resamples)")
        print(intervals(values=all_values, columns=rung_columns).sort("site"))
        share_columns = [*(f"{name}_share" for name in CAUSES), "season_share", "total"]
        print("Shares of the shortfall 1 - rung 4")
        print(intervals(values=all_values, columns=share_columns).sort("site"))
        print("Components in points of the injected capacity")
        print(intervals(values=all_values, columns=[*CAUSES, "season"]).sort("site"))
        divisor = divisor_table()
        print("p99 ratios (median over farms, range over farms)")
        print(divisor)
        year = (
            divisor.filter(pl.col("window") == "year")
            .select("site", year_ratio="ratio")
            .join(
                values.filter(pl.col("b") == 0).select("site", "r4", "total"),
                on="site",
            )
            .with_columns(
                divisor_shift=pl.col("year_ratio") - pl.col("r4"),
                divisor_share=(pl.col("year_ratio") - pl.col("r4")).abs() / pl.col("total"),
            )
        )
        print("Divisor shift")
        print(year)
        check = reproduction(ladder=ladder)
        print("Reproduction of experiment 129 (max over farms)")
        print(
            check.group_by("site")
            .agg(
                pl.col("injected_relative_difference").max(),
                pl.col("recovery_gap").max(),
                pl.col("recovery_gap_with_129_base").max(),
            )
            .sort("site")
        )
        print_prediction(values=values, divisor_year=year)


def print_prediction(*, values: pl.DataFrame, divisor_year: pl.DataFrame) -> None:
    """Print the pre-registered prediction, pooled and per Primary."""
    point = values.filter(pl.col("b") == 0, pl.col("site").is_in(common.DECISION_SITES))
    pooled_point = pooled(values=values).filter(pl.col("b") == 0)
    divisor_pooled = statistics.median(
        divisor_year.filter(pl.col("site").is_in(common.DECISION_SITES))["divisor_share"].to_list()
    )
    checks = {
        "demand >= 0.40": ("demand_share", lambda v: v >= PREDICTION["demand"]),
        "irradiance >= 0.25": ("irradiance_share", lambda v: v >= PREDICTION["irradiance"]),
        "season < 0.15": ("season_share", lambda v: v < PREDICTION["season"]),
    }
    for label, (column, test) in checks.items():
        n_sites = int(sum(test(v) for v in point[column].to_list()))
        print(f"{label}: pooled {pooled_point[column][0]:.3f}, {n_sites} of 5 Primaries")
    n_divisor = int(
        (
            divisor_year.filter(pl.col("site").is_in(common.DECISION_SITES))["divisor_share"]
            < PREDICTION["divisor"]
        ).sum()
    )
    print(f"divisor < 0.15: pooled {divisor_pooled:.3f}, {n_divisor} of 5 Primaries")


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


def site_order() -> list[str]:
    """Return the x order: the five decision Primaries, D9, then the pooled median."""
    return [*common.DECISION_SITES, "D9", "Pooled"]


def dot_chart(
    *,
    data: pl.DataFrame,
    series: dict[str, str],
    colours: list[str],
    title: alt.TitleParams,
    y_title: str,
    y_domain: list[float],
    reference: float | None,
) -> alt.TopLevelMixin:
    """Draw one dot per series and Primary with an interval, from `site`, `label`, value columns."""
    sites = site_order()
    scale = alt.Scale(domain=list(series.values()), range=colours)
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=list(series.values())),
        color=alt.Color(
            "label:N",
            scale=scale,
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    y_scale = alt.Scale(domain=y_domain)
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", scale=y_scale, title=y_title),
        y2="hi:Q",
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", scale=y_scale)
    )
    layers = [bars, points]
    if reference is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"y": [reference]}))
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(y=alt.Y("y:Q", scale=y_scale)),  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(width=WIDTH, height=280, title=title)


def long_form(*, values: pl.DataFrame, columns: dict[str, str]) -> pl.DataFrame:
    """Return `site`, `label`, `value`, `lo`, `hi` rows for the named columns of the values."""
    all_values = pl.concat([values, pooled(values=values).select(values.columns)])
    table = intervals(values=all_values, columns=list(columns))
    return pl.concat(
        [
            table.select(
                "site",
                label=pl.lit(label),
                value=pl.col(column),
                lo=pl.col(f"{column}_lo"),
                hi=pl.col(f"{column}_hi"),
            )
            for column, label in columns.items()
        ]
    )


def figure_rungs(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery at each rung of the ladder, per Primary."""
    return dot_chart(
        data=long_form(values=values, columns=RUNGS),
        series=RUNGS,
        colours=[ocf.DATA_SKY, ocf.DATA_GREEN, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        title=chart_title(
            text="Most of the missing capacity appears only when demand is fitted too",
            subtitle=(
                "Share of an injected real solar farm's capacity that each fit finds (1 is all of"
                " it; farm injected at 50% of the Primary's pinned capacity, read at season-year"
                " 2024). Dot: median over five farms. Line: 80% interval from resampling whole"
                " months. Dashed line: a perfect recovery. Pooled: median over the five Primaries"
                " D5, D10, D11, D14, and D16; D9 is shown beside them. Fits use rows before"
                " 2026-07-01."
            ),
        ),
        y_title="Recovery (share of injected capacity; 1 is best)",
        y_domain=[0.5, 1.1],
        reference=1.0,
    )


def figure_causes(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the shortfall split by cause, as stacked bars per Primary."""
    point = pl.concat([values, pooled(values=values).select(values.columns)]).filter(
        pl.col("b") == 0
    )
    data = pl.concat(
        [
            point.select("site", cause=pl.lit(label), loss=pl.col(column))
            for column, label in CAUSES.items()
        ]
    )
    sites = site_order()
    order = list(CAUSES.values())
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("loss:Q", title="Share of injected capacity missing (0 is best)"),
            color=alt.Color(
                "cause:N",
                scale=alt.Scale(domain=order, range=CAUSE_COLOURS),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
                sort=order,
            ),
            order=alt.Order("cause:N", sort="ascending"),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=chart_title(
                text="Where the missing share of the injected farm goes",
                subtitle=(
                    "Each bar adds up to 1 minus the recovery of the full fit. Medians over five"
                    " farms, point fit. A segment below zero means that rung found more than the"
                    " rung before it. Pooled: median over the five decision Primaries."
                ),
            ),
        )
    )


def figure_rows(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery of the full fit with all, summer-only, and winter-only rows."""
    return dot_chart(
        data=long_form(values=values, columns=ROWS),
        series=ROWS,
        colours=ROW_COLOURS,
        title=chart_title(
            text="Winter rows find about half of the farm and summer rows find more than all rows",
            subtitle=(
                "Share of the injected farm's capacity that the full fit finds, using only the"
                " named months of every year. Dot: median over five farms. Line: 80% interval"
                " from resampling whole months. Dashed line: a perfect recovery."
            ),
        ),
        y_title="Recovery (share of injected capacity; 1 is best)",
        y_domain=[0.0, 1.1],
        reference=1.0,
    )


def figure_divisor() -> alt.TopLevelMixin:
    """Return the 99th-percentile ratio of recovered to injected output, by season."""
    names = {"year": "Whole year", **{k: k.title() for k in SEASON_ORDER[1:]}}
    table = divisor_table().with_columns(label=pl.col("window").replace(names))
    data = table.select(
        "site", "label", value=pl.col("ratio"), lo=pl.col("ratio_lo"), hi=pl.col("ratio_hi")
    )
    return dot_chart(
        data=data,
        series=names,
        colours=SEASON_COLOURS,
        title=chart_title(
            text=(
                "The recovered output's flat top is a fifth to a quarter below the farm's"
                " in spring and summer"
            ),
            subtitle=(
                "99th percentile of the recovered solar output divided by the 99th percentile of"
                " the injected farm's output, over daylight half-hours of season-year 2024."
                " Dot: median over five farms. Line: range over the five farms."
                " Dashed line: equal flat tops."
            ),
        ),
        y_title="Recovered flat top / farm flat top (1 is best)",
        y_domain=[0.5, 1.2],
        reference=1.0,
    )


def main() -> None:
    """Print the numbers and write the figures."""
    ladder = load_ladder()
    values = site_values(ladder=ladder)
    print_numbers(ladder=ladder, values=values)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_rungs(values=values), name="figure_01_rungs")
    save_chart(figure_causes(values=values), name="figure_02_causes")
    save_chart(figure_rows(values=values), name="figure_03_rows")
    save_chart(figure_divisor(), name="figure_04_divisor")


if __name__ == "__main__":
    main()
