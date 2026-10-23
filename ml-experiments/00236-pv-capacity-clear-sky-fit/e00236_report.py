"""Print every number of experiment 236's write-up and draw its figures.

uv run --no-sync python ml-experiments/00236-*/e00236_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00236_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00236-pv-capacity-clear-sky-fit"
)
PNG_DIR: Final[Path] = Path("/tmp/e00236_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
SERIES: Final[dict[str, str]] = {
    "all_recovery": "Full fit on all rows (experiment 231's rung 4)",
    "clear_recovery": "Full fit on clear, high-sun half-hours only",
    "clear_farm_only": "Farm alone on the clear half-hours, no demand",
}
COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
RISE: Final[float] = 0.05
WIDTH_LIMIT: Final[float] = 1.5


def load() -> pl.DataFrame:
    """Return every Primary's table in long form."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"clear_{s}.parquet") for s in common.ALL_SITES]
    )


def site_values(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each series."""
    wanted = table.filter(
        pl.col("stat").is_in(["recovery", "farm_only"]),
    ).with_columns(
        series=pl.col("row_set") + "_" + pl.col("stat").replace({"farm_only": "farm_only"})
    )
    wide = (
        wanted.group_by("site", "series", "b")
        .agg(pl.col("value").median())
        .pivot(on="series", index=["site", "b"], values="value")
    )
    return wide.with_columns(rise=pl.col("clear_recovery") - pl.col("all_recovery"))


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


def print_numbers(*, table: pl.DataFrame, values: pl.DataFrame) -> None:
    """Print the tables the write-up quotes."""
    with pl.Config(tbl_rows=60, tbl_cols=30, float_precision=3, tbl_width_chars=220):
        everything = pl.concat([values, pooled(values=values).select(values.columns)])
        columns = [*SERIES, "rise"]
        result = intervals(values=everything, columns=columns).sort("site")
        print("Recovery (median over farms; 10th and 90th percentile over resamples)")
        print(result)
        widths = result.select(
            "site",
            all_width=pl.col("all_recovery_hi") - pl.col("all_recovery_lo"),
            clear_width=pl.col("clear_recovery_hi") - pl.col("clear_recovery_lo"),
        ).with_columns(ratio=pl.col("clear_width") / pl.col("all_width"))
        print("Interval widths")
        print(widths)
        point = result.filter(pl.col("site").is_in(common.DECISION_SITES))
        decision_widths = widths.filter(pl.col("site").is_in(common.DECISION_SITES))
        print(
            f"rise >= {RISE} at {int((point['rise'] >= RISE).sum())} of 5 Primaries;"
            f" interval ratio <= {WIDTH_LIMIT} at"
            f" {int((decision_widths['ratio'] <= WIDTH_LIMIT).sum())} of 5"
        )
        rows = (
            table.filter(pl.col("stat").is_in(["n_rows", "n_rows_read_out_year"]), pl.col("b") == 0)
            .group_by("site", "row_set", "stat")
            .agg(pl.col("value").median())
            .pivot(on=["row_set", "stat"], index="site", values="value")
            .sort("site")
        )
        print("Training rows (median over farms)")
        print(rows)
        reference = (
            table.filter(pl.col("row_set") == "all", pl.col("stat") == "recovery", pl.col("b") == 0)
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
        print("Reproduction of experiment 231's rung 4 (max abs difference over all farms)")
        print(reference)


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


def long_form(*, values: pl.DataFrame) -> pl.DataFrame:
    """Return `site`, `label`, `value`, `lo`, `hi` rows for the series."""
    everything = pl.concat([values, pooled(values=values).select(values.columns)])
    table = intervals(values=everything, columns=list(SERIES))
    return pl.concat(
        [
            table.select(
                "site",
                label=pl.lit(label),
                value=pl.col(column),
                lo=pl.col(f"{column}_lo"),
                hi=pl.col(f"{column}_hi"),
            )
            for column, label in SERIES.items()
        ]
    )


def figure_recovery(*, values: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery on all rows and on clear rows, per Primary."""
    return dot_chart(
        data=long_form(values=values),
        series=SERIES,
        colours=COLOURS,
        title=chart_title(
            text="Fitting only on clear half-hours does not recover more of the farm",
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
        y_domain=[0.3, 1.1],
        reference=1.0,
    )


def main() -> None:
    """Print the numbers and write the figure."""
    table = load()
    values = site_values(table=table)
    print_numbers(table=table, values=values)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_recovery(values=values), name="figure_01_recovery")


if __name__ == "__main__":
    main()
