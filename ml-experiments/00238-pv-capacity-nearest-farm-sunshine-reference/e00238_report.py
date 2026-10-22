"""Print every number of experiment 238's write-up and draw its figures.

uv run --no-sync python ml-experiments/00238-*/e00238_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00238_common as common
import plotting.ocf_theme as ocf
import polars as pl
from studies.pv_dataset import pv_sites

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00238-pv-capacity-nearest-farm-sunshine-reference"
)
PNG_DIR: Final[Path] = Path("/tmp/e00238_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
RISE: Final[float] = 0.05
CLOSE_PAIR_KM: Final[float] = 5.0
"""Injected and reference farms closer than this are near-duplicates (farms A and C, 2 km apart)."""
SERIES: Final[dict[str, str]] = {
    "cams_all": "Satellite shape, all rows (experiment 231's rung 4)",
    "cams_same_rows": "Satellite shape, rows where the reference farm has a reading",
    "reference": "Nearby farm's output as the shape, same rows",
}
COLOURS: Final[list[str]] = [ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
DISTANCE_BINS: Final[list[tuple[str, float, float]]] = [
    ("under 5 km", 0.0, 5.0),
    ("5 to 12 km", 5.0, 12.0),
    ("12 to 30 km", 12.0, 30.0),
]


def load() -> pl.DataFrame:
    """Return every Primary's table in long form."""
    return pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"result_{s}.parquet") for s in common.ALL_SITES]
    ).rename({"reference": "ref_farm"})


def farm_distances() -> pl.DataFrame:
    """Return the distance in km between every pair of farms."""
    rows = pv_sites().to_dicts()
    return pl.DataFrame(
        [
            {
                "farm": a["site"],
                "ref_farm": b["site"],
                "pair_km": common.distance_km(
                    lat_a=a["latitude"],
                    lon_a=a["longitude"],
                    lat_b=b["latitude"],
                    lon_b=b["longitude"],
                ),
            }
            for a in rows
            for b in rows
            if a["site"] != b["site"]
        ]
    )


def farm_arm_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary, farm, and resample with a column per arm."""
    return table.filter(pl.col("stat") == stat).pivot(
        on="arm", index=["site", "farm", "ref_farm", "b"], values="value"
    )


def site_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each arm and the rise."""
    return (
        farm_arm_values(table=table, stat=stat)
        .with_columns(rise=pl.col("reference") - pl.col("cams_same_rows"))
        .group_by("site", "b")
        .agg(pl.col(c).median() for c in [*SERIES, "rise"])
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


def with_pooled(*, values: pl.DataFrame) -> pl.DataFrame:
    """Return the per-Primary values with the pooled rows appended."""
    return pl.concat([values, pooled(values=values).select(values.columns)])


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


def print_numbers(*, table: pl.DataFrame) -> pl.DataFrame:
    """Print the tables the write-up quotes and return the recovery values."""
    with pl.Config(tbl_rows=80, tbl_cols=30, float_precision=3, tbl_width_chars=220):
        recovery = site_values(table=table, stat="recovery")
        columns = [*SERIES, "rise"]
        result = intervals(values=with_pooled(values=recovery), columns=columns).sort("site")
        print("Recovery (99th-percentile ratio; median over farms; 10th and 90th percentile)")
        print(result)
        point = result.filter(pl.col("site").is_in(common.DECISION_SITES))
        print(f"rise >= {RISE} at {int((point['rise'] >= RISE).sum())} of 5 decision Primaries")
        for stat in (
            "p99_factor",
            "recovery_frozen",
            "capacity_ratio",
            "explained",
            "farm_only_scale",
        ):
            values = site_values(table=table, stat=stat)
            print(stat)
            print(intervals(values=with_pooled(values=values), columns=columns).sort("site"))
        factor = (
            table.filter(pl.col("stat") == "p99_factor")
            .group_by("site", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="site", values="value")
            .sort("site")
        )
        print(
            "p99_factor: p99 of the arm shape over p99 of the injected output (median over farms)"
        )
        print(factor)
        pairs = (
            farm_arm_values(table=table, stat="recovery")
            .filter(pl.col("b") == 0)
            .with_columns(rise=pl.col("reference") - pl.col("cams_same_rows"))
            .join(farm_distances(), on=["farm", "ref_farm"])
        )
        print("Pairs: rise by distance between the injected farm and the reference farm")
        for label, low, high in DISTANCE_BINS:
            chosen = pairs.filter(pl.col("pair_km") >= low, pl.col("pair_km") < high)
            print(
                f"{label}: {chosen.height} pairs, median rise {chosen['rise'].median():.3f},"
                f" range {chosen['rise'].min():.3f} to {chosen['rise'].max():.3f}"
            )
        far = pairs.filter(pl.col("pair_km") >= CLOSE_PAIR_KM)
        far_by_site = (
            far.group_by("site").agg(n=pl.len(), rise=pl.col("rise").median()).sort("site")
        )
        print(f"Median rise by Primary, pairs {CLOSE_PAIR_KM:.0f} km or more apart")
        print(far_by_site)
        decision = far_by_site.filter(pl.col("site").is_in(common.DECISION_SITES))
        n_decision = int((decision["rise"] >= RISE).sum())
        print(f"decision Primaries with rise >= {RISE} on those pairs: {n_decision}")
        by_site = pairs.group_by("site").agg(
            n=pl.len(), km=pl.col("pair_km").median(), rise=pl.col("rise").median()
        )
        print(by_site.sort("site"))
        distances = (
            table.filter(pl.col("stat") == "distance_km")
            .group_by("site")
            .agg(
                to_primary_min=pl.col("value").min(),
                to_primary_max=pl.col("value").max(),
                n_pairs=pl.len(),
            )
            .sort("site")
        )
        print("Distance from the Primary to the reference farm (km)")
        print(distances)
        rows = (
            table.filter(pl.col("stat") == "n_rows", pl.col("b") == 0)
            .group_by("site", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="site", values="value")
            .sort("site")
        )
        print("Training rows (median over farms)")
        print(rows)
        reproduction = (
            table.filter(pl.col("arm") == "cams_all", pl.col("stat") == "capacity_ratio")
            .filter(pl.col("b") == 0)
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
        print(reproduction)
        return pairs


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


def figure_recovery(*, recovery: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the recovery under the three solar shapes, per Primary."""
    sites = [*common.DECISION_SITES, "D9", "Pooled"]
    table = intervals(values=with_pooled(values=recovery), columns=list(SERIES))
    data = pl.concat(
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
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=list(SERIES.values())),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(SERIES.values()), range=COLOURS),
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
        height=280,
        title=chart_title(
            text="A nearby farm's output as the shape finds more of an injected farm",
            subtitle=(
                "Share of an injected real solar farm's output that each fit finds (1 is all of"
                " it; farm injected at 50% of the Primary's pinned capacity, read at season-year"
                " 2024, as the ratio of 99th percentiles of output over daylight half-hours)."
                " Dot: median over the eligible injected farms. Line: 80% interval from"
                " resampling whole months. Dashed line: a perfect recovery. Pooled: median over"
                " D5, D10, D11, D14, and D16; D9 is shown beside them. Fits use rows before"
                " 2026-07-01."
            ),
        ),
    )


def figure_distance(*, pairs: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the rise of each injected farm and Primary against the farm-to-reference distance."""
    scale = alt.Scale(domain=[-0.05, 0.3])
    points = (
        alt.Chart(pairs.select("site", "pair_km", "rise"))
        .mark_point(filled=True, size=50, color=ocf.DATA_BLUE, opacity=0.8, aria=False)
        .encode(
            x=alt.X(
                "pair_km:Q",
                title="Distance between the injected farm and the reference farm (km)",
                scale=alt.Scale(domain=[0, 28]),
            ),
            y=alt.Y("rise:Q", scale=scale, title="Recovery gain over the satellite shape"),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [0.0, RISE]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rules, points).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="The gain is largest for farms 2 km apart and stays positive out to 25 km",
            subtitle=(
                "Each dot is one Primary and one injected farm. Gain: recovery with the nearby"
                " farm's output as the shape, minus recovery with the satellite shape on the same"
                " rows (point fit; more is better). Lower dashed line: no gain. Upper dashed"
                " line: the 0.05 threshold."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load()
    pairs = print_numbers(table=table)
    recovery = site_values(table=table, stat="recovery")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_recovery(recovery=recovery), name="figure_01_recovery")
    save_chart(figure_distance(pairs=pairs), name="figure_02_distance")


if __name__ == "__main__":
    main()
