"""Print every number of experiment 269's write-up and draw its figures.

uv run --no-sync python ml-experiments/00269-*/e00269_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00269_common as common
import plotting.ocf_theme as ocf
import polars as pl
from studies.pv_dataset import pv_sites

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00269-pv-capacity-gain-decomposition"
)
PNG_DIR: Final[Path] = Path("/tmp/e00269_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
LOW_QUANTILE: Final[float] = 0.1
HIGH_QUANTILE: Final[float] = 0.9
CLOSE_PAIR_KM: Final[float] = 5.0
"""Injected and reference farms closer than this are near-duplicates (farms A and C, 2 km apart)."""
ARM_LABELS: Final[dict[str, str]] = {
    "cams_same_rows": "1. Satellite fleet shape",
    "reference": "2. Nearby farm's output",
    "reference_plant": "3. Nearby farm's plant, satellite sunshine",
    "pooled_plant": "4. Pooled empirical shape",
}
"""The four arms of the plan; the satellite shape on all rows is a reproduction check only."""
UNITS: Final[dict[str, str]] = {
    "recovery": "Ratio of 99th percentiles",
    "capacity_ratio": "Capacity ratio (experiment 231)",
    "recovery_p95": "Ratio of 95th percentiles",
    "recovery_energy": "Ratio of clear-sky daylight energy",
}
COLOURS: Final[list[str]] = [ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
DISTANCE_BINS: Final[list[tuple[str, float, float]]] = [
    ("under 5 km", 0.0, 5.0),
    ("5 to 12 km", 5.0, 12.0),
    ("12 to 30 km", 12.0, 30.0),
]
BAND_LOW: Final[float] = 0.90
BAND_HIGH: Final[float] = 1.10
GAIN_SHARE_PREDICTED: Final[float] = 0.60
SUNSHINE_PREDICTED_MAX: Final[float] = 0.04
SUNSHINE_LIMIT: Final[float] = 0.05


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


def farm_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary, farm, and resample with a column per arm and per contrast."""
    wide = table.filter(pl.col("stat") == stat).pivot(
        on="arm", index=["site", "farm", "ref_farm", "b"], values="value"
    )
    return wide.with_columns(
        total=pl.col("reference") - pl.col("cams_same_rows"),
        sunshine=pl.col("reference") - pl.col("reference_plant"),
        plant_nearby=pl.col("reference_plant") - pl.col("cams_same_rows"),
        plant_pooled=pl.col("pooled_plant") - pl.col("cams_same_rows"),
    )


def site_values(*, table: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return one row per Primary and resample: the median over farms of each column."""
    wide = farm_values(table=table, stat=stat)
    columns = [c for c in wide.columns if c not in {"site", "farm", "ref_farm", "b"}]
    return wide.group_by("site", "b").agg(pl.col(c).median() for c in columns)


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


def print_numbers(*, table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Print the tables the write-up quotes; return the intervals and the pair table."""
    columns = [*ARM_LABELS, "total", "sunshine", "plant_nearby", "plant_pooled"]
    long = []
    with pl.Config(tbl_rows=80, tbl_cols=30, float_precision=3, tbl_width_chars=220):
        for stat, unit in UNITS.items():
            result = intervals(
                values=with_pooled(values=site_values(table=table, stat=stat)), columns=columns
            ).sort("site")
            print(f"{unit} ({stat}): median over farms; 10th and 90th percentile over resamples")
            print(result)
            long.append(result.with_columns(stat=pl.lit(stat)))
            decision = result.filter(pl.col("site").is_in(common.DECISION_SITES))
            n_band = int(
                (
                    (decision["pooled_plant"] >= BAND_LOW) & (decision["pooled_plant"] <= BAND_HIGH)
                ).sum()
            )
            n_ninety = int((decision["pooled_plant"] >= BAND_LOW).sum())
            print(
                f"{stat}: arm 4 in {BAND_LOW} to {BAND_HIGH} at {n_band} of 5;"
                f" >= {BAND_LOW} at {n_ninety} of 5"
            )
            print(f"{stat}: arm 4 under 0.80 at {int((decision['pooled_plant'] < 0.8).sum())} of 5")
        pooled_p99 = long[0].filter(pl.col("site") == "Pooled").row(0, named=True)
        share = (pooled_p99["total"] - pooled_p99["sunshine"]) / pooled_p99["total"]
        print(
            f"Share of the p99-unit gain not due to sunshine: {share:.3f}"
            f" (predicted >= {GAIN_SHARE_PREDICTED})"
        )
        for stat in UNITS:
            row = (
                intervals(
                    values=with_pooled(values=site_values(table=table, stat=stat)), columns=columns
                )
                .filter(pl.col("site") == "Pooled")
                .row(0, named=True)
            )
            print(f"{stat}: sunshine share {row['sunshine']:.3f}, total {row['total']:.3f}")
        pairs = (
            farm_values(table=table, stat="recovery")
            .filter(pl.col("b") == 0)
            .join(farm_distances(), on=["farm", "ref_farm"])
        )
        print("Pairs (p99 unit): contrasts by distance between the injected and reference farms")
        for label, low, high in DISTANCE_BINS:
            chosen = pairs.filter(pl.col("pair_km") >= low, pl.col("pair_km") < high)
            print(
                f"{label}: {chosen.height} pairs, median total {chosen['total'].median():.3f},"
                f" sunshine {chosen['sunshine'].median():.3f}"
                f" (range {chosen['sunshine'].min():.3f} to {chosen['sunshine'].max():.3f}),"
                f" plant pooled {chosen['plant_pooled'].median():.3f}"
            )
        far = pairs.filter(pl.col("pair_km") >= CLOSE_PAIR_KM)
        print(f"Sunshine share by Primary, pairs {CLOSE_PAIR_KM:.0f} km or more apart (p99 unit)")
        print(
            far.group_by("site")
            .agg(n=pl.len(), sunshine=pl.col("sunshine").median(), total=pl.col("total").median())
            .sort("site")
        )
        print("Without the pairs under 5 km: arm recovery by Primary and unit (point fit)")
        for stat, unit in UNITS.items():
            farm_wide = (
                farm_values(table=table, stat=stat)
                .filter(pl.col("b") == 0)
                .join(farm_distances(), on=["farm", "ref_farm"])
            )
            kept = (
                farm_wide.filter(pl.col("pair_km") >= CLOSE_PAIR_KM)
                .group_by("site")
                .agg(pl.col(c).median() for c in [*ARM_LABELS, "total", "sunshine"])
                .sort("site")
            )
            print(unit)
            print(kept)
        rows = (
            table.filter(pl.col("stat") == "n_rows", pl.col("b") == 0)
            .group_by("site", "arm")
            .agg(pl.col("value").median())
            .pivot(on="arm", index="site", values="value")
            .sort("site")
        )
        print("Training rows (median over farms)")
        print(rows)
        reference = pl.read_parquet(
            common.ML_EXPERIMENTS_DATA_DIR
            / "00238-pv-capacity-nearest-farm-sunshine-reference"
            / "work"
            / "result_D5.parquet"
        )
        print(
            f"(reproduction below compares with experiment 238's tables; {reference.height} rows)"
        )
        old = pl.concat(
            [
                pl.read_parquet(
                    common.ML_EXPERIMENTS_DATA_DIR
                    / "00238-pv-capacity-nearest-farm-sunshine-reference"
                    / "work"
                    / f"result_{s}.parquet"
                )
                for s in common.ALL_SITES
            ]
        ).rename({"reference": "ref_farm"})
        merged = (
            table.filter(pl.col("arm").is_in(["cams_all", "cams_same_rows", "reference"]))
            .join(
                old,
                on=["site", "farm", "ref_farm", "arm", "stat", "b"],
                suffix="_old",
            )
            .select((pl.col("value") - pl.col("value_old")).abs().max().alias("max_abs_difference"))
        )
        print("Reproduction of experiment 238 (arms 1 and 2, every statistic): max abs difference")
        print(merged)
    return pl.concat(long), pairs


MOVE_LIMIT_MW: Final[float] = 0.3
"""Stage 2's predicted largest move in a Primary's capacity when the shape changes."""


def real_capacities() -> pl.DataFrame:
    """Print stage 2a's capacities and return one row per Primary and season-year."""
    table = pl.read_parquet(common.WORK_DIR / "real_capacities.parquet")
    wide = table.pivot(
        on="shape",
        index=[
            "site",
            "season_year",
            "pinned_mw",
            "published_mw",
            "published_low_mw",
            "published_high_mw",
        ],
        values=["raw_mw", "estimate_mw", "divisor"],
    ).with_columns(
        move_mw=pl.col("estimate_mw_empirical") - pl.col("estimate_mw_physical"),
        raw_move_mw=pl.col("raw_mw_empirical") - pl.col("raw_mw_physical"),
    )
    with pl.Config(tbl_rows=80, tbl_cols=30, float_precision=2, tbl_width_chars=220):
        print("Stage 2a: capacity in MW by shape (divided by that shape's own recovery)")
        print(wide.filter(pl.col("season_year") >= 2024).sort("site", "season_year"))
        for year in (2024, 2025):
            chosen = wide.filter(
                pl.col("season_year") == year, pl.col("site").is_in(common.DECISION_SITES)
            )
            n_small = int((chosen["move_mw"].abs() < MOVE_LIMIT_MW).sum())
            print(
                f"season-year {year}: capacity moves by under {MOVE_LIMIT_MW} MW at {n_small} of 5"
            )
            print(
                "  moves (MW): "
                f"{dict(zip(chosen['site'], chosen['move_mw'].round(2), strict=True))}"
            )
            inside = chosen.filter(
                pl.col("estimate_mw_empirical").is_between(
                    pl.col("published_low_mw"), pl.col("published_high_mw")
                )
            )
            print(f"  empirical estimate inside experiment 129's 80% range: {inside.height} of 5")
        physical = wide.filter(
            pl.col("site").is_in(common.DECISION_SITES), pl.col("season_year") == 2024
        )
        gap = (physical["estimate_mw_physical"] - physical["published_mw"]).abs().max()
        print(
            "physical shape vs experiment 129's published estimate, season-year 2024:"
            f" max abs {gap:.2f} MW"
        )
    return wide


def figure_real(*, wide: pl.DataFrame) -> alt.TopLevelMixin:
    """Return each Primary's season-year 2025 capacity under the two shapes and experiment 129's."""
    year = wide.filter(pl.col("season_year") == 2025)
    sites = [*common.DECISION_SITES, "D9"]
    data = pl.concat(
        [
            year.select("site", label=pl.lit("Physical fleet shape"), value="estimate_mw_physical"),
            year.select(
                "site", label=pl.lit("Pooled empirical shape"), value="estimate_mw_empirical"
            ),
            year.select("site", label=pl.lit("Pinned capacity"), value="pinned_mw"),
        ]
    )
    labels = ["Physical fleet shape", "Pooled empirical shape", "Pinned capacity"]
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=labels, range=[ocf.DATA_SKY, ocf.BRAND_ORANGE, "#808080"]),
            legend=alt.Legend(title=None, orient="top", columns=1),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", title="Capacity of unmetered solar (MW AC)")
    )
    ranges = (
        alt.Chart(year.filter(pl.col("published_low_mw").is_not_null()))
        .mark_rule(color="#b0b0b0", strokeWidth=6, opacity=0.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=sites), y="published_low_mw:Q", y2="published_high_mw:Q"
        )
    )
    return alt.layer(ranges, points).properties(
        width=WIDTH,
        height=280,
        title=chart_title(
            text="The empirical shape lowers each substation's solar capacity by 0.2 to 0.4 MW",
            subtitle=(
                "Fitted capacity of unmetered solar at season-year 2025 (July 2025 to June 2026),"
                " each divided by its own shape's recovery of an injected farm. Grey bar: the 80%"
                " range of experiment 129 (not available at D9). Fits use rows before 2026-07-01."
            ),
        ),
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


def figure_pooled(*, long: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the pooled recovery of each arm in each capacity unit, with 80% intervals."""
    pooled = long.filter(pl.col("site") == "Pooled")
    data = pl.concat(
        [
            pooled.select(
                "stat",
                unit=pl.col("stat").replace(UNITS),
                label=pl.lit(label),
                value=pl.col(arm),
                lo=pl.col(f"{arm}_lo"),
                hi=pl.col(f"{arm}_hi"),
            )
            for arm, label in ARM_LABELS.items()
        ]
    )
    scale = alt.Scale(domain=[0.6, 1.2])
    base = alt.Chart(data).encode(
        x=alt.X(
            "label:N",
            sort=list(ARM_LABELS.values()),
            title=None,
            axis=alt.Axis(labels=False, ticks=False),
        ),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", scale=scale, title="Recovery (share of injected output; 1 is best)"),
        y2="hi:Q",
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", scale=scale)
    )
    return (
        alt.layer(bars, points)
        .properties(width=130, height=260)
        .facet(
            column=alt.Column("unit:N", sort=list(UNITS.values()), title=None),
        )
        .properties(
            title=chart_title(
                text="A learned plant shape finds most of what the nearby farm's output finds",
                subtitle=(
                    "Share of an injected real solar farm's output that the fit finds (1 is all of"
                    " it), in four capacity units, median over the five decision Primaries."
                    " Dot: median over the injected farms. Line: 80% interval from resampling"
                    " whole months. Fits use rows before 2026-07-01."
                ),
            )
        )
    )


def figure_arm4(*, long: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the pooled empirical shape's recovery per Primary and unit, with a 0.9 to 1.1 band."""
    sites = [*common.DECISION_SITES, "D9", "Pooled"]
    data = long.select(
        "site",
        unit=pl.col("stat").replace(UNITS),
        value=pl.col("pooled_plant"),
        lo=pl.col("pooled_plant_lo"),
        hi=pl.col("pooled_plant_hi"),
    )
    band = (
        alt.Chart(pl.DataFrame({"low": [BAND_LOW], "high": [BAND_HIGH]}))
        .mark_rect(color="#d9d9d9", opacity=0.5)
        .encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
        color=alt.Color(
            "unit:N",
            scale=alt.Scale(domain=list(UNITS.values()), range=COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=500),
        ),
        xOffset=alt.XOffset("unit:N", sort=list(UNITS.values())),
    )
    scale = alt.Scale(domain=[0.6, 1.2])
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", scale=scale, title="Recovery (1 is best)"), y2="hi:Q"
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", scale=scale)
    )
    return alt.layer(band, rules, points).properties(
        width=WIDTH,
        height=280,
        title=chart_title(
            text="Recovery with the pooled empirical shape, by Primary and capacity unit",
            subtitle=(
                "Recovery with the mean learned response of the five other farms as the solar"
                " shape, per Primary and capacity unit (median over injected farms; line: 80%"
                " interval from resampling whole months). Grey band: 0.90 to 1.10."
            ),
        ),
    )


def figure_distance(*, pairs: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the sunshine share of each pair against the distance between the two farms."""
    scale = alt.Scale(domain=[-0.1, 0.25])
    points = (
        alt.Chart(pairs.select("site", "pair_km", "sunshine"))
        .mark_point(filled=True, size=50, color=ocf.DATA_BLUE, opacity=0.8, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "pair_km:Q",
                title="Distance between the injected farm and the reference farm (km)",
                scale=alt.Scale(domain=[0, 28]),
            ),
            y=alt.Y("sunshine:Q", scale=scale, title="Sunshine share of the gain"),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [0.0, SUNSHINE_LIMIT]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rules, points).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="The nearby farm's measured sunshine adds little beyond 5 km",
            subtitle=(
                "Each dot is one Primary and one injected farm. Sunshine share: recovery with the"
                " nearby farm's output as the shape, minus recovery with the same farm's learned"
                " response driven by satellite irradiance (ratio of 99th percentiles; point fit)."
                " Lower dashed line: no sunshine share. Upper: 0.05."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load()
    long, pairs = print_numbers(table=table)
    wide = real_capacities()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(figure_pooled(long=long), name="figure_01_pooled")
    save_chart(figure_arm4(long=long), name="figure_02_pooled_shape")
    save_chart(figure_distance(pairs=pairs), name="figure_03_distance")
    save_chart(figure_real(wide=wide), name="figure_04_real_capacities")


if __name__ == "__main__":
    main()
