"""Print every number of experiment 235's write-up and draw its figures.

uv run --no-sync python ml-experiments/00235-*/e00235_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00235_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00235-pv-capacity-matched-day-pairs"
)
PNG_DIR: Final[Path] = Path("/tmp/e00235_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
SERIES: Final[dict[str, str]] = {
    "recovery": "Matched pairs, farm added",
    "recovery_with_load": "Matched pairs, farm and a cloud-driven load added",
    "placebo_recovery": "Matched pairs, December and January only (placebo)",
}
COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
LOW: Final[float] = 0.90
HIGH: Final[float] = 1.10
DROP: Final[float] = 0.88
PLACEBO_LIMIT: Final[float] = 0.15
LOAD_LIMIT: Final[float] = 0.10


def load(*, suffix: str = "") -> pl.DataFrame:
    """Return every Primary's table in long form."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"pairs_{s}{suffix}.parquet") for s in common.ALL_SITES]
    )


def site_values(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each series."""
    ladder = pl.concat(
        [pl.read_parquet(common.AUDIT_LADDER_DIR / f"ladder_{s}.parquet") for s in common.ALL_SITES]
    )
    convex = ladder.filter(pl.col("rung") == "cap_base", pl.col("b") == 0).select(
        "site", "farm", convex="value"
    )
    wanted = (
        table.filter(pl.col("stat").is_in(["recovery", "recovery_with_load", "capacity"]))
        .with_columns(
            series=pl.when(pl.col("set") == "placebo")
            .then(pl.lit("placebo_") + pl.col("stat"))
            .otherwise(pl.col("stat"))
        )
        .join(convex, on=["site", "farm"])
        .with_columns(
            value=pl.when(pl.col("stat") == "capacity")
            .then(pl.col("value") / pl.col("convex"))
            .otherwise(pl.col("value")),
        )
        .with_columns(
            series=pl.when(pl.col("series") == "capacity")
            .then(pl.lit("capacity_ratio"))
            .otherwise(pl.col("series"))
        )
        .filter(pl.col("series") != "placebo_capacity")
    )
    wide = (
        wanted.group_by("site", "series", "b")
        .agg(pl.col("value").median())
        .pivot(on="series", index=["site", "b"], values="value")
    )
    if "placebo_recovery" not in wide.columns:
        wide = wide.with_columns(placebo_recovery=pl.lit(None, dtype=pl.Float64))
    return wide.with_columns(load_bias=pl.col("recovery_with_load") - pl.col("recovery"))


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


def print_numbers(*, table: pl.DataFrame, values: pl.DataFrame, label: str) -> None:
    """Print the tables the write-up quotes."""
    with pl.Config(tbl_rows=60, tbl_cols=30, float_precision=3, tbl_width_chars=220):
        everything = pl.concat([values, pooled(values=values).select(values.columns)])
        columns = [*SERIES, "load_bias", "capacity_ratio"]
        result = intervals(values=everything, columns=columns).sort("site")
        print(f"[{label}] Recovery (median over farms; 10th and 90th percentile over resamples)")
        print(result)
        point = result.filter(pl.col("site").is_in(common.DECISION_SITES))
        in_range = int(((point["recovery"] >= LOW) & (point["recovery"] <= HIGH)).sum())
        print(
            f"[{label}] recovery in {LOW} to {HIGH} at {in_range} of 5;"
            f" below {DROP} at {int((point['recovery'] < DROP).sum())} of 5;"
            f" placebo under {PLACEBO_LIMIT} at"
            f" {int((point['placebo_recovery'] < PLACEBO_LIMIT).sum())} of 5;"
            f" load bias within {LOAD_LIMIT} at"
            f" {int((point['load_bias'].abs() <= LOAD_LIMIT).sum())} of 5"
        )
        counts = (
            table.filter(pl.col("stat").is_in(["n_pairs", "n_half_hours"]))
            .group_by("site", "set", "stat")
            .agg(pl.col("value").median())
            .pivot(on=["set", "stat"], index="site", values="value")
            .sort("site")
        )
        print(f"[{label}] Pairs and usable half-hours (median over farms)")
        print(counts)
        spread = (
            table.filter(pl.col("stat") == "recovery", pl.col("b") == 0, pl.col("set") == "main")
            .group_by("site")
            .agg(pl.col("value").min().alias("min_farm"), pl.col("value").max().alias("max_farm"))
            .sort("site")
        )
        print(f"[{label}] Recovery range over farms")
        print(spread)


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


def _unused(*, values: pl.DataFrame, columns: dict[str, str]) -> pl.DataFrame:
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


def long_form(*, values: pl.DataFrame, series: dict[str, str]) -> pl.DataFrame:
    """Return `site`, `label`, `value`, `lo`, `hi` rows for the series."""
    everything = pl.concat([values, pooled(values=values).select(values.columns)])
    table = intervals(values=everything, columns=list(series))
    return pl.concat(
        [
            table.select(
                "site",
                label=pl.lit(label),
                value=pl.col(column),
                lo=pl.col(f"{column}_lo"),
                hi=pl.col(f"{column}_hi"),
            )
            for column, label in series.items()
        ]
    )


def figure_recovery(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery with and without the load, and the placebo, per Primary."""
    return dot_chart(
        data=long_form(values=values, series=SERIES),
        series=SERIES,
        colours=COLOURS,
        title=chart_title(
            text="Matched pairs find 0.76 of the farm, and a cloud-driven load adds 0.3 to that",
            subtitle=(
                "Share of an injected real solar farm's capacity that the matched-day-pair"
                " estimator finds (1 is all of it; farm injected at 50% of the Primary's pinned"
                " capacity, read at season-year 2024). Dot: median over five farms. Line: 80%"
                " interval from resampling whole months. Dashed line: a perfect recovery."
                " Pooled: median over the five Primaries D5, D10, D11, D14, and D16; D9 is shown"
                " beside them. The placebo has one month per resampling stratum and so no"
                " interval. Fits use rows before 2026-07-01."
            ),
        ),
        y_title="Recovery (share of injected capacity; 1 is best)",
        y_domain=[0.0, 1.5],
        reference=1.0,
    )


def figure_capacity(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the matched-pair capacity of the real power as a share of the convex fit's."""
    series = {"capacity_ratio": "Matched-pair capacity / convex-fit capacity (no farm added)"}
    return dot_chart(
        data=long_form(values=values, series=series),
        series=series,
        colours=[ocf.BRAND_ORANGE],
        title=chart_title(
            text="The matched-pair capacity of the real power against the convex fit's",
            subtitle=(
                "Capacity found in each Primary's own power (no farm added) by the matched-pair"
                " estimator, divided by the convex fit's capacity at season-year 2024. Dot: median"
                " over five farms' row sets. Line: 80% interval from resampling whole months."
                " Dashed line: equal capacities."
            ),
        ),
        y_title="Matched-pair capacity / convex-fit capacity",
        y_domain=[0.4, 1.4],
        reference=1.0,
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load()
    values = site_values(table=table)
    print_numbers(table=table, values=values, label="min 10 daylight rows")
    sensitivity = load(suffix="_min20")
    print_numbers(
        table=sensitivity,
        values=site_values(table=sensitivity),
        label="min 20 daylight rows, post hoc",
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_recovery(values=values), name="figure_01_recovery")
    save_chart(figure_capacity(values=values), name="figure_02_capacity")


if __name__ == "__main__":
    main()
