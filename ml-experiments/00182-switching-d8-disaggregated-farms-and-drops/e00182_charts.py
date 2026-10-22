# ruff: noqa: E501
"""Draw the figures for experiment 182 from the tables `e00182_check.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00182-switching-d8-disaggregated-farms-and-drops"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    monthly = (
        pl.read_parquet(DATA_DIR / "farm_availability_monthly.parquet")
        .unpivot(["farm_1", "farm_2"], index="time", variable_name="farm", value_name="readings")
        .with_columns(farm=pl.col("farm").str.replace("_", " "))
    )
    fig1 = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("time:T", title="Month"),
            y=alt.Y(
                "readings:Q", title="Half-hours with a reading", scale=alt.Scale(domain=[0, 1500])
            ),
            color=alt.Color(
                "farm:N",
                title=None,
                scale=alt.Scale(
                    domain=["farm 1", "farm 2"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
            ),
            xOffset="farm:N",
        )
        .properties(
            width=WIDTH,
            height=160,
            title="Readings per month in the raw power table (a month holds about 1,440)",
        )
    )
    fig1.save(DOCS_DIR / "figure_1.svg")

    rnd = pl.read_parquet(DATA_DIR / "random_counts.parquet")
    obs = pl.read_parquet(DATA_DIR / "drops_vs_farms.parquet")["any_anomaly"].sum()
    fig2 = (
        alt.Chart(rnd.group_by("anomaly_count").len())  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE_LIGHT)
        .encode(
            x=alt.X(
                "anomaly_count:Q",
                title="Windows (of 29) with a farm anomaly",
                scale=alt.Scale(domain=[0, 30]),
            ),
            y=alt.Y("len:Q", title="Draws (of 1,000)"),
        )
    ) + alt.Chart(pl.DataFrame({"x": [int(obs)]})).mark_rule(
        color=ocf.BRAND_ORANGE, strokeWidth=3
    ).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    fig2.properties(
        width=WIDTH,
        height=130,
        title="Random windows (blue bars) and the 29 drops (orange rule) both show a farm anomaly in every window",
    ).save(DOCS_DIR / "figure_2.svg")

    prof = pl.read_parquet(DATA_DIR / "hour_profile.parquet")
    base = alt.Chart(prof).encode(
        x=alt.X("hour:Q", title="Local hour of day", scale=alt.Scale(domain=[0, 23]))
    )
    depth = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("excess:Q", title="Extra shortfall in drops (MW)")
    )
    sky = base.mark_line(color=ocf.DATA_BLUE, strokeDash=[4, 3]).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("clear_sky_ghi:Q", title="Clear-sky irradiance (W/m²)")
    )
    alt.layer(depth, sky).resolve_scale(y="independent").properties(
        width=WIDTH,
        height=160,
        title="D8's extra shortfall inside drops (orange) and clear-sky irradiance (blue dashed) by hour",
    ).save(DOCS_DIR / "figure_3.svg")


if __name__ == "__main__":
    main()
