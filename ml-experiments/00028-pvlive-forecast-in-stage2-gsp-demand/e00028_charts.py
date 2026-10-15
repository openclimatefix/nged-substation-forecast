"""Draw experiment 00028's figures from the saved results, as SVG.

1. The contrasts on the GSP quantile score, with 95% intervals.
2. The solar forecast's mean absolute error by lead band, against a climatology.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00028-pvlive-forecast-in-stage2-gsp-demand/e00028_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00028_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
SITE_NAMES: Final[dict[str, str]] = {"D1": "Bicker Fen", "D6": "Walpole"}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The three contrasts, over all leads and up to 48 hours."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("scope").is_in(["all leads", "up to 48 h"])
    )
    labels = data["label"].unique(maintain_order=True).to_list()
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset="scope:N",
        color=alt.Color(
            "scope:N",
            title=None,
            scale=alt.Scale(
                domain=["all leads", "up to 48 h"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="First arm minus second arm (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=170,
        title=_title(
            text="A forecast of the GSP's solar generation and the true value, against none",
            lines=[
                (
                    "Quantile score difference of stage 2 arms (smaller score is better; below "
                    "zero favours the first arm). Walpole and Bicker Fen, 73 forecast runs, July "
                    "2025 to June 2026, ENS weather. Dot: estimate. Line: 95% interval from "
                    "re-drawing whole months and seeds."
                )
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The solar forecast's and the climatology's daytime error by lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "solar.parquet")
        .with_columns(site=pl.col("site").replace(SITE_NAMES))
        .unpivot(
            ["mae", "climatology_mae"],
            index=["site", "band"],
            variable_name="source",
            value_name="error",
        )
        .with_columns(
            source=pl.col("source").replace(
                {"mae": "ENS member mean", "climatology_mae": "Climatology"}
            )
        )
    )
    bands = ["0-24 h", "1-2 d", "3-6 d", "7-15 d"]
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", sort=bands, title=None),
            xOffset="source:N",
            y=alt.Y("error:Q", title="Mean absolute error (points of PV_Live capacity)"),
            color=alt.Color(
                "source:N",
                title=None,
                scale=alt.Scale(
                    domain=["ENS member mean", "Climatology"], range=[ocf.DATA_BLUE, "#999999"]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH // 2 - 40, height=160)
        .facet(column=alt.Column("site:N", title=None))
        .properties(
            title=_title(
                text="The solar forecast against a month-by-half-hour climatology",
                lines=[
                    (
                        "Daytime half-hours (PV_Live above 1% of capacity) of the validation runs, "
                        "by lead; smaller is better. The solar model is trained on ERA5 weather "
                        "and applied to ENS members."
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
