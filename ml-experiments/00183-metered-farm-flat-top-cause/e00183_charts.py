"""Draw the anonymised figure for experiment 183 from the tables `e00183_flat_top.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00183-metered-farm-flat-top-cause"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MONTH_NAMES: Final[dict[int, str]] = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}


def main() -> None:  # noqa: D103
    levels = (
        pl.read_csv(DATA_DIR / "monthly_levels_by_quantile.csv")
        .filter(pl.col("quantile") != 0.95)
        .with_columns(
            month_name=pl.col("month").replace_strict(MONTH_NAMES),
            percentile=pl.when(pl.col("quantile") == 0.90)
            .then(pl.lit("90th percentile (planned)"))
            .otherwise(pl.lit("99th percentile")),
        )
    )
    chart = (
        alt.Chart(levels)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X(
                "month_name:N",
                sort=list(MONTH_NAMES.values()),
                title="Month",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "level:Q",
                title="Power over capacity",
                scale=alt.Scale(domain=[0.9, 1.12]),
            ),
            color=alt.Color(
                "percentile:N",
                title=None,
                scale=alt.Scale(
                    domain=["90th percentile (planned)", "99th percentile"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 70, height=110)
        .facet(
            row=alt.Row("site:N", title="Farm"),
            title=alt.TitleParams(
                [
                    "Flat level of power over capacity by month at high clear-sky irradiance",
                    "(above 650 W/m²), farms C to F, April to August.",
                    "A constant line would indicate an export cap.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_1.svg")


if __name__ == "__main__":
    main()
