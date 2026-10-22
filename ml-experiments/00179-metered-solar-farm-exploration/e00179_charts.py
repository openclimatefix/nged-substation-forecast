# ruff: noqa: E501
"""Draw the anonymised figures for experiment 179 from the tables `e00179_explore.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00179-metered-solar-farm-exploration"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
HEIGHT: Final[int] = 130


def main() -> None:  # noqa: D103
    sites = pv_sites().select("time_series_id", "site", "effective_capacity_mw")
    cleaned = (
        pl.read_parquet(DATA_DIR / "cleaned_solar_farms.parquet")
        .join(sites, on="time_series_id")
        .filter(pl.col("clear_sky_ghi_w_m2") > 20, pl.col("drop_reason").is_null())
        .with_columns(ratio=pl.col("power") / pl.col("effective_capacity_mw"))
    )
    sample = cleaned
    bins = pl.read_csv(DOCS_DIR / "envelope_bins.csv")
    panels = []
    for site in sorted(sites["site"].to_list()):
        pts = (
            alt.Chart(sample.filter(pl.col("site") == site).sample(n=150, seed=1))  # ty: ignore[unresolved-attribute]
            .mark_circle(size=3, opacity=0.25, color=ocf.DATA_BLUE_LIGHT)
            .encode(
                x=alt.X(
                    "clear_sky_ghi_w_m2:Q",
                    title="Clear-sky irradiance (W/m²)",
                    scale=alt.Scale(domain=[0, 1000]),
                ),
                y=alt.Y("ratio:Q", title="Power / capacity", scale=alt.Scale(domain=[0, 1.3])),
            )
        )
        env = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                bins.filter(pl.col("site") == site).unpivot(
                    ["p50", "p95", "p99"], index=["site", "ghi_cs"]
                )
            )
            .mark_line(strokeWidth=2)
            .encode(
                x="ghi_cs:Q",
                y="value:Q",
                color=alt.Color(
                    "variable:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["p50", "p95", "p99"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                    ),
                ),
            )
        )
        panels.append(
            (pts + env).properties(width=CONTENT_WIDTH_PX - 80, height=HEIGHT, title=f"Farm {site}")
        )
    fig1 = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=[
                "Power against clear-sky irradiance: a flat top at high irradiance means AC clipping.",
                "Lines are the median, 95th, and 99th percentiles per irradiance bin.",
            ]
        )
    )
    fig1.save(DOCS_DIR / "figure_1.svg")
    monthly = pl.read_csv(DOCS_DIR / "monthly_yield.csv", try_parse_dates=True)
    fig2 = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("yield_p90:Q", title="Yield index (90th percentile)"),
            color=alt.Color(
                "site:N",
                title="Farm",
                scale=alt.Scale(
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_AMBER,
                    ]
                ),
            ),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 80,
            height=260,
            title=[
                "Monthly yield index: power over capacity per kW/m² of clear-sky irradiance.",
                "Steps and slow slopes in the line are outages and drift.",
            ],
        )
    )
    fig2.save(DOCS_DIR / "figure_2.svg")
    # Daily peak, as a share of capacity, per farm.
    daily = (
        pl.read_parquet(DATA_DIR / "cleaned_solar_farms.parquet")
        .join(sites, on="time_series_id")
        .group_by("site", pl.col("time").dt.truncate("1d").alias("day"))
        .agg((pl.col("power").max() / pl.col("effective_capacity_mw").first()).alias("peak"))
    )
    fig3 = (
        alt.Chart(daily)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("yearmonth(day):T", title=None),
            y=alt.Y("site:N", title="Farm"),
            color=alt.Color(
                "mean(peak):Q",
                title="Mean daily peak / capacity",
                scale=alt.Scale(scheme="oranges"),
            ),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 80,
            height=160,
            title="Mean daily peak power over capacity, by month: gaps and dark columns are outages.",
        )
    )
    fig3.save(DOCS_DIR / "figure_3.svg")


if __name__ == "__main__":
    main()
