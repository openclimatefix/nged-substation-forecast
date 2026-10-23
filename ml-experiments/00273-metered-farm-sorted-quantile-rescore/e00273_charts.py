# ruff: noqa: E501
"""Draw experiment 273's figure from the crossing table (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00273_rescore as rescore
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / rescore.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
REPORTED_CROSSING_ROWS: Final[int] = 26_344
REPORTED_ROWS: Final[int] = 77_897
"""Experiment 268's farm-forecast rows with crossing quantiles and all its stored rows, as its session reported them."""


def crossing() -> None:
    """Figure 1: share of stored rows with crossing quantiles, per experiment."""
    table = pl.read_csv(rescore.DATA_DIR / "crossing.csv")
    per_experiment = (
        table.group_by("experiment")
        .agg(
            rate=(pl.col("any_crossing") * pl.col("n_rows")).sum() / pl.col("n_rows").sum(),
            arms=pl.col("arm").n_unique(),
            rows=pl.col("n_rows").sum(),
        )
        .with_columns(
            label=pl.format(
                "Experiment {}: {} arms, {} stored rows",
                pl.col("experiment"),
                pl.col("arms"),
                pl.col("rows").cast(pl.String),
            ),
            source=pl.lit("This page, from the stored predictions"),
        )
    )
    reference = pl.DataFrame(
        {
            "experiment": [268],
            "rate": [REPORTED_CROSSING_ROWS / REPORTED_ROWS],
            "label": ["Experiment 268: farm-forecast head, as reported by its session"],
            "source": ["Reported, not re-counted here"],
        }
    )
    data = pl.concat([per_experiment.select(reference.columns), reference]).with_columns(
        percent=pl.col("rate") * 100.0
    )
    order = data.sort("experiment")["label"].to_list()
    chart = (
        alt.Chart(data)
        .mark_bar(size=18)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "percent:Q",
                title="Stored rows with at least one crossing pair of quantiles (%)",
                scale=alt.Scale(domain=[0, 40]),
            ),
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=[
                        "This page, from the stored predictions",
                        "Reported, not re-counted here",
                    ],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=210,
            title=alt.TitleParams(
                [
                    "Share of stored quantile forecasts with crossing quantiles (a lower level above a higher one),",
                    "every arm of experiments 224, 243, 248, 257, and 260 pooled, 12 to 36 h leads, farms A to F.",
                ],
                anchor="start",
                fontSize=12,
                fontWeight="normal",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_1_crossing.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    crossing()
