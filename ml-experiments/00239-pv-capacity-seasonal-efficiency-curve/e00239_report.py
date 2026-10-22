"""Print every number of experiment 239's write-up and draw its figures.

uv run --no-sync python ml-experiments/00239-*/e00239_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00239_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00239-pv-capacity-seasonal-efficiency-curve"
)
PNG_DIR: Final[Path] = Path("/tmp/e00239_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
RISE: Final[float] = 0.03
SERIES: Final[dict[str, str]] = {
    "orig_plain": "Satellite shape, farm output zeroed above 75 degrees of zenith",
    "orig_curve": "With the curve, same injection",
    "full_plain": "Satellite shape, farm output kept at every sun height",
    "full_curve": "With the curve, same injection",
}
LABELS: Final[dict[str, str]] = {
    "orig_plain": "A: satellite shape, injection as in experiment 231",
    "orig_curve": "B: with the curve, injection as in experiment 231",
    "full_plain": "C: satellite shape, full-sun injection",
    "full_curve": "D: with the curve, full-sun injection",
}
COLOURS: Final[list[str]] = [ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.BRAND_ORANGE]
MONTHS: Final[list[str]] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def load() -> pl.DataFrame:
    """Return every Primary's table in long form."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"result_{s}.parquet") for s in common.ALL_SITES]
    )


def farm_arm_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary, farm, and resample with a column per arm."""
    return table.filter(pl.col("stat") == stat).pivot(
        on="arm", index=["site", "farm", "b"], values="value"
    )


def site_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each arm and the rises."""
    return (
        farm_arm_values(table=table, stat=stat)
        .with_columns(
            rise_orig=pl.col("orig_curve") - pl.col("orig_plain"),
            rise_full=pl.col("full_curve") - pl.col("full_plain"),
            zeroed_cost=pl.col("full_plain") - pl.col("orig_plain"),
        )
        .group_by("site", "b")
        .agg(pl.col(c).median() for c in [*SERIES, "rise_orig", "rise_full", "zeroed_cost"])
    )


def with_pooled(*, values: pl.DataFrame) -> pl.DataFrame:
    """Return the per-Primary values with the median over the five decision Primaries appended."""
    columns = [c for c in values.columns if c not in {"site", "b"}]
    pooled = (
        values.filter(pl.col("site").is_in(common.DECISION_SITES))
        .group_by("b")
        .agg(pl.col(c).median() for c in columns)
        .with_columns(site=pl.lit("Pooled"))
    )
    return pl.concat([values, pooled.select(values.columns)])


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


def print_numbers(*, table: pl.DataFrame) -> None:
    """Print the tables the write-up quotes."""
    columns = [*SERIES, "rise_orig", "rise_full", "zeroed_cost"]
    with pl.Config(tbl_rows=80, tbl_cols=30, float_precision=3, tbl_width_chars=250):
        for stat in (
            "recovery",
            "capacity_ratio",
            "recovery_frozen",
            "explained",
            "farm_only_scale",
        ):
            values = with_pooled(values=site_values(table=table, stat=stat))
            result = intervals(values=values, columns=columns).sort("site")
            print(stat, "(median over farms; 10th and 90th percentile over resamples)")
            print(result.select("site", *columns))
            print(
                result.select("site", *[f"{c}_lo" for c in columns], *[f"{c}_hi" for c in columns])
            )
            if stat == "recovery":
                point = result.filter(pl.col("site").is_in(common.DECISION_SITES))
                for name in ("rise_orig", "rise_full"):
                    n = int((point[name] >= RISE).sum())
                    print(f"{name} >= {RISE} at {n} of 5 decision Primaries")
        print("p99_factor (median over farms)")
        print(
            table.filter(pl.col("stat") == "p99_factor")
            .group_by("site", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="site", values="value")
            .sort("site")
        )
        print("Training rows (median over farms)")
        print(
            table.filter(pl.col("stat") == "n_rows", pl.col("b") == 0)
            .group_by("site", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="site", values="value")
            .sort("site")
        )
        reproduction = (
            table.filter(
                pl.col("arm") == "orig_plain", pl.col("stat") == "capacity_ratio", pl.col("b") == 0
            )
            .select("site", "farm", here="value")
            .join(
                pl.concat(
                    [
                        pl.read_parquet(common.AUDIT_LADDER_DIR / f"ladder_{s}.parquet")
                        for s in common.ALL_SITES
                    ]
                )
                .filter(pl.col("rung") == "r4", pl.col("b") == 0)
                .select("site", "farm", audit="value"),
                on=["site", "farm"],
            )
            .select((pl.col("here") - pl.col("audit")).abs().max().alias("max_abs_difference"))
        )
        print("Reproduction of experiment 231's rung 4 (max abs difference)")
        print(
            reproduction.with_columns(
                pl.col("max_abs_difference").map_elements(lambda v: f"{v:.2e}")
            )
        )
        curves = pl.read_parquet(common.WORK_DIR / "curves.parquet")
        scales = curves.filter(pl.col("kind") == "farm_scale")
        mean_curve = (
            scales.group_by("month")
            .agg(mean=pl.col("value").mean(), lo=pl.col("value").min(), hi=pl.col("value").max())
            .sort("month")
        )
        mean_curve = mean_curve.with_columns(
            mean=pl.col("mean") / pl.col("mean").mean(),
            lo=pl.col("lo") / pl.col("mean").mean(),
            hi=pl.col("hi") / pl.col("mean").mean(),
        )
        print("Curve of all five farms (mean 1) and the range of the farms' own monthly scales")
        print(mean_curve)
        print(
            "largest over smallest factor:",
            float(mean_curve["mean"].max() / mean_curve["mean"].min()),
            "; April to September only:",
            float(
                mean_curve.filter(pl.col("month").is_between(4, 9))["mean"].max()
                / mean_curve.filter(pl.col("month").is_between(4, 9))["mean"].min()
            ),
        )
        own = pl.read_parquet(common.WORK_DIR / "own_site.parquet")
        print("Own-site share of squared output explained, per farm")
        print(own.pivot(on="arm", index=["farm", "stat"], values="value").sort("stat", "farm"))
        print(
            own.group_by("stat", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="stat", values="value")
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


def figure_curve() -> alt.TopLevelMixin:
    """Return the seasonal efficiency curve with the range of the farms' own monthly scales."""
    curves = pl.read_parquet(common.WORK_DIR / "curves.parquet")
    scales = curves.filter(pl.col("kind") == "farm_scale")
    grand = scales.group_by("month").agg(pl.col("value").mean()).get_column("value").mean()
    table = (
        scales.group_by("month")
        .agg(
            value=pl.col("value").mean() / grand,
            lo=pl.col("value").min() / grand,
            hi=pl.col("value").max() / grand,
        )
        .with_columns(label=pl.col("month").map_elements(lambda m: MONTHS[m - 1]))
        .sort("month")
    )
    scale = alt.Scale(domain=[0.4, 1.4])
    base = alt.Chart(table).encode(
        x=alt.X("label:N", sort=MONTHS, title="Calendar month", axis=alt.Axis(labelAngle=0))
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.DATA_SKY).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", scale=scale, title="Efficiency factor (mean over the year is 1)"), y2="hi:Q"
    )
    points = base.mark_point(filled=True, size=60, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", scale=scale)
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rule, bars, points).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="The metered farms' output falls below the satellite shape in winter",
            subtitle=(
                "Orange dot: mean over farms A, B, C, D, and F of the monthly least-squares scale"
                " of a farm's output on its own satellite-based shape, scaled so that the mean of"
                " the 12 months is 1. Blue line: range over the five farms. Dashed line: no"
                " seasonal effect. Rows before 2026-07-01."
            ),
        ),
    )


def figure_recovery(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery under the four arms, per Primary."""
    sites = [*common.DECISION_SITES, "D9", "Pooled"]
    values = with_pooled(values=site_values(table=table, stat="recovery"))
    result = intervals(values=values, columns=list(SERIES))
    data = pl.concat(
        [
            result.select(
                "site",
                label=pl.lit(LABELS[column]),
                value=pl.col(column),
                lo=pl.col(f"{column}_lo"),
                hi=pl.col(f"{column}_hi"),
            )
            for column in SERIES
        ]
    )
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=list(LABELS.values())),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    scale = alt.Scale(domain=[0.5, 1.1])
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", scale=scale, title="Recovery (share of injected output; 1 is best)"),
        y2="hi:Q",
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", scale=scale)
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rule, bars, points).properties(
        width=WIDTH,
        height=300,
        title=chart_title(
            text="The seasonal curve and the injection's low-sun cut change the recovery",
            subtitle=(
                "Share of an injected real solar farm's output that each fit finds (1 is all of"
                " it; farm injected at 50% of the Primary's pinned capacity, read at season-year"
                " 2024, as the ratio of 99th percentiles of output over daylight half-hours)."
                " Dot: median over the five injected farms. Line: 80% interval from resampling"
                " whole months. Pooled: median over D5, D10, D11, D14, and D16; D9 is shown"
                " beside them. Fits use rows before 2026-07-01."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load()
    print_numbers(table=table)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_curve(), name="figure_01_curve")
    save_chart(figure_recovery(table=table), name="figure_02_recovery")


if __name__ == "__main__":
    main()
