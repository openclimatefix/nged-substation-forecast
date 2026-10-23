# ruff: noqa: E501
"""Draw the anonymised figures for experiment 286 from the tables `e00286_injection.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00286-metered-farm-drift-soiling-injection"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / "00286-metered-farm-drift-soiling-injection"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
METHOD_NAMES: Final[dict[str, str]] = {
    "A": "A: monthly yield index",
    "B": "B: farm over other farms",
}
METHOD_SCALE: Final[alt.Scale] = alt.Scale(
    domain=list(METHOD_NAMES.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
)
REFERENCE_LINE: Final[str] = "#555555"
"""Dark grey for the zero line and the injected-size line."""
WIDTH: Final[int] = CONTENT_WIDTH_PX - 120


def named(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace the method letter by its long name."""
    return frame.with_columns(pl.col("method").replace(METHOD_NAMES))


def main() -> None:  # noqa: D103
    base = named(pl.read_csv(DATA_DIR / "baseline_trends.csv"))
    drift = named(pl.read_csv(DATA_DIR / "drift_summary.csv"))
    soil = named(pl.read_csv(DATA_DIR / "soiling_summary.csv"))
    drift_raw = named(pl.read_csv(DATA_DIR / "drift_injection.csv"))
    soil_raw = named(pl.read_csv(DATA_DIR / "soiling_injection.csv"))

    dodge = alt.Y("method:N", axis=None)
    bars = (
        alt.Chart(base)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=3)
        .encode(
            x=alt.X("lo:Q", title="Yearly trend of the real series (% per year), 90% interval"),
            x2="hi:Q",
            y=dodge,
            color=alt.Color("method:N", title=None, scale=METHOD_SCALE),
        )
    )
    points = (
        alt.Chart(base)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50)
        .encode(x="trend:Q", y=dodge, color=alt.Color("method:N", scale=METHOD_SCALE))
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=REFERENCE_LINE).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    fig1 = (
        (bars + points + zero)
        .properties(width=WIDTH, height=36)
        .facet(row=alt.Row("farm:N", title="Farm", header=alt.Header(labelAngle=0)))
        .properties(
            title=[
                "Real yearly trend per farm: the farm-over-farms index is not narrower overall.",
                "Farm E has 2 years of data, so its intervals are wide or rest on few months.",
            ]
        )
    )
    fig1.save(DOCS_DIR / "figure_1.svg")

    shift = (
        drift_raw.with_columns(
            shift=(pl.col("trend") - pl.col("base_trend")).abs(),
            half_width=(pl.col("base_hi") - pl.col("base_lo")) / 2,
        )
        .group_by("method", "farm", "rate")
        .agg(
            (pl.col("shift") / pl.col("half_width")).median().alias("ratio"),
        )
    )
    lines = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("rate:Q", title="Injected loss (% per year)", scale=alt.Scale(domain=[0, 2.1])),
            y=alt.Y("ratio:Q", title="Shift ÷ interval half-width"),
            color=alt.Color("method:N", title=None, scale=METHOD_SCALE),
        )
    )
    one = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color=REFERENCE_LINE, strokeDash=[4, 3])
        .encode(y=alt.datum(1))
    )
    fig2 = (
        alt.layer(lines, one, data=shift)
        .properties(width=WIDTH // 2 - 20, height=90)
        .facet(facet=alt.Facet("farm:N", title="Farm"), columns=3)
        .properties(
            title=[
                "Injected drift is seen when the shift in the fitted trend exceeds the interval half-width.",
                "Above the dashed line at 1 the injected loss moves the trend by more than the real noise.",
            ]
        )
    )
    fig2.save(DOCS_DIR / "figure_2.svg")

    soil_shift = (
        soil_raw.with_columns(
            shift=(pl.col("beta") - pl.col("base_beta")).abs(),
            half_width=(pl.col("base_hi") - pl.col("base_lo")) / 2,
        )
        .group_by("method", "farm")
        .agg(
            pl.col("shift").median().alias("shift"),
            pl.col("half_width").first().alias("half_width"),
        )
    )
    del soil
    fig3 = (
        alt.Chart(soil_shift)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=90)
        .encode(
            x=alt.X("farm:N", title="Farm", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("half_width:Q", title="90% interval half-width (% per day)"),
            color=alt.Color("method:N", title=None, scale=METHOD_SCALE),
            xOffset="method:N",
        )
        .properties(width=WIDTH, height=200)
    )
    line = (
        alt.Chart(pl.DataFrame({"y": [0.1]}))
        .mark_rule(color=REFERENCE_LINE, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    fig3 = (fig3 + line).properties(
        title=[
            "Soiling slope noise per farm and index, against the injected 0.1% per day (dashed line).",
            "A point below the line means the injected slope is larger than the interval's half-width.",
        ]
    )
    fig3.save(DOCS_DIR / "figure_3.svg")
    del drift


if __name__ == "__main__":
    main()
