"""Draw the figures of experiment 351 from `work/contrasts_scored.parquet` and `work/asof/`.

uv run --no-sync python ml-experiments/00351-*/e00351_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import e00351_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
SCOPES: Final[tuple[str, ...]] = (
    "sunniest third (headline)",
    "all daylight",
    "darkest daylight third",
    "night",
    "winter nights (Dec to Feb)",
    "all half-hours",
)
NAMES: Final[dict[str, str]] = {
    "p - pin": "P (physical-shape read-out, divided by injection recovery) minus default",
    "r - pin": "R (physical-shape read-out, raw) minus default",
    "b - pin": "B (experiment 349's empirical-shape read-out) minus default",
}
AXIS_TITLE: Final[str] = (
    "Quantile score minus the default's (points of capacity; below 0 is better)"
)
POINTS_TITLE: Final[str] = "Solar capacity (points of effective capacity)"


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def contrasts() -> None:
    """Draw the quantile-score contrasts with 95% month-resampled intervals."""
    table = pl.read_parquet(common.WORK_DIR / "contrasts_scored.parquet").filter(
        pl.col("metric") == "qs",
        pl.col("contrast").is_in(list(NAMES)),
        pl.col("scope").is_in(SCOPES),
    )
    for lead, name in (
        ("0.5 to 168 h", "figure_1_contrasts_0p5_168"),
        ("24 to 168 h", "figure_2_contrasts_24_168"),
    ):
        data = table.filter(pl.col("lead") == lead).with_columns(
            arm=pl.col("contrast").replace(NAMES)
        )
        base = _chart(data.to_pandas()).encode(
            y=alt.Y("scope:N", sort=list(SCOPES), title=None),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(NAMES.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="bottom", columns=1),
            ),
        )
        zero = _chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="black")
        chart = (
            base.mark_rule().encode(x=alt.X("lower:Q"), x2="upper:Q")
            + base.mark_point(filled=True, size=60).encode(x=alt.X("estimate:Q", title=AXIS_TITLE))
            + zero.encode(x="x:Q")
        ).properties(
            width=WIDTH,
            title=alt.TitleParams(
                text=f"The physical-shape capacity does not lower the sunny score ({lead})",
                anchor="start",
                fontSize=13,
            ),
        )
        chart.save(DOCS_DIR / f"{name}.svg")


def capacities() -> None:
    """Draw the pinned capacity and the arm B and arm P estimates of each Primary by origin."""
    table = pl.read_parquet(common.WORK_DIR / "capacity_comparison.parquet")
    pinned = "pinned (experiment 45)"
    arm_b = "B (experiment 349, empirical shape)"
    arm_p = "P (physical shape, this experiment)"
    long = pl.concat(
        [
            table.select("site", "origin", series=pl.lit(pinned), points=pl.col("pinned_points")),
            table.select(
                "site", "origin", series=pl.lit(arm_b), points=pl.col("estimate_points_b")
            ),
            table.select("site", "origin", series=pl.lit(arm_p), points=pl.col("estimate_points")),
        ]
    ).with_columns(origin=pl.col("origin").dt.replace_time_zone(None))
    chart = (
        _chart(long.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("origin:T", title="Forecast fold origin"),
            y=alt.Y("points:Q", title=POINTS_TITLE),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[pinned, arm_b, arm_p],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="bottom", columns=1),
            ),
        )
        .properties(width=170, height=120)
        .facet(
            facet=alt.Facet("site:N", title=None, sort=["D5", "D9", "D10", "D11", "D14", "D16"]),
            columns=3,
        )
        .properties(
            title=alt.TitleParams(
                text="Arm P's capacity is 5 to 13% above arm B's, above the pin at some Primaries",
                anchor="start",
                fontSize=13,
            )
        )
    )
    chart.save(DOCS_DIR / "figure_3_capacities.svg")


if __name__ == "__main__":
    contrasts()
    capacities()
