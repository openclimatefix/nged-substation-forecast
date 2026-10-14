"""Draw experiment 00014's figures and print every number the write-up quotes.

Reads `work/results/*.parquet` from `e00014_score.py`. Substations appear by their anonymous labels
only. Writes `figure_1.svg` and `figure_2.svg` (and 2x PNGs for checking) to the experiment's page
folder under `docs/ml-experiments/`.

    uv run python ml-experiments/00014-*/e00014_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00014_common as common
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

FIGURES_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
"""This checkout's page folder, which is not the main checkout's `ML_EXPERIMENTS_DOCS_DIR`."""
ROW_HEIGHT_PX: Final[int] = 26
LABEL_PX: Final[int] = 250
THRESHOLD: Final[float] = -0.03


def figure_1(*, mismatch: pl.DataFrame, post_hoc: pl.DataFrame) -> None:
    """Draw the monthly ratio of ENS daylight shortwave to CAMS, before and after stage 0."""
    months = mismatch.filter(pl.col("month") != "all")
    long = pl.concat(
        [
            months.select("month", ratio="raw_to_cams_ratio").with_columns(
                source=pl.lit("ENS as published")
            ),
            months.select("month", ratio="stage0_to_cams_ratio").with_columns(
                source=pl.lit("Stage 0")
            ),
            post_hoc.select("month", ratio="monthly_stage0_to_cams_ratio").with_columns(
                source=pl.lit("Post hoc: monthly stage 0")
            ),
        ]
    )
    colours = {
        "ENS as published": ocf.DATA_BLUE,
        "Stage 0": ocf.BRAND_ORANGE,
        "Post hoc: monthly stage 0": ocf.DATA_PURPLE,
    }
    scale = alt.Scale(domain=[0.75, 1.3])
    base = alt.Chart(long).encode(
        x=alt.X("month:N", title="Valid-time month", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("ratio:Q", title="Mean daylight shortwave over CAMS", scale=scale),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=1),
        ),
    )
    reference = alt.Chart(pl.DataFrame({"y": [1.0]}))
    rule = reference.mark_rule(color="#444444", strokeDash=[4, 3]).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("y:Q", scale=scale)
    )
    chart = (
        rule + base.mark_line(point=True, strokeWidth=2).encode()  # ty: ignore[unresolved-attribute]
    ).properties(
        width=CONTENT_WIDTH_PX - 70,
        height=240,
        title=alt.TitleParams(
            [
                "Stage 0 removes the ENS excess over CAMS for the year but not month by month.",
                "Dashed line: same as CAMS. 16 Primary series, daylight, all leads, 73 runs.",
            ],
            anchor="start",
            fontSize=12,
        ),
    )
    _save(chart=chart, name="figure_1")


def figure_2(*, planned: pl.DataFrame, exploratory: pl.DataFrame) -> None:
    """Draw the contrasts: the planned pair, the lead bands, and the exploratory variants."""
    wanted = {
        "C2 by lead band: 0-24 h": "C2, 0 to 24 h ahead",
        "C2 by lead band: 1-2 d": "C2, 1 to 2 days ahead",
        "C2 by lead band: 3-6 d": "C2, 3 to 6 days ahead",
        "C2 by lead band: 7-10 d": "C2, 7 to 10 days ahead",
        "C2 by lead band: 11-15 d": "C2, 11 to 15 days ahead",
        "C2, daylight valid times": "C2, daylight valid times",
        "C2, BSP and GSP together": "C2, BSP and GSP (4 series)",
        "X1: era5-stage0 minus era5-trained, Primary": "X1: stage 0 on the ERA5-trained arm",
        "X3 (post hoc): cams-stage0-month minus cams-trained, Primary": "X3 (post hoc): monthly",
    }
    rows = [
        {
            "label": "C1: CAMS-trained vs ERA5-trained"
            if row["label"].startswith("C1")
            else "C2: stage 0 vs raw ENS (CAMS-trained)",
            "kind": "Planned (97.5% interval)",
            **{k: row[k] for k in ("difference", "lower", "upper")},
        }
        for row in planned.iter_rows(named=True)
    ]
    rows += [
        {
            "label": wanted[row["label"]],
            "kind": "Exploratory (95% interval)",
            **{k: row[k] for k in ("difference", "lower", "upper")},
        }
        for row in exploratory.iter_rows(named=True)
        if row["label"] in wanted
    ]
    frame = pl.DataFrame(rows)
    colours = {
        "Planned (97.5% interval)": ocf.BRAND_ORANGE,
        "Exploratory (95% interval)": ocf.DATA_BLUE,
    }
    scale = alt.Scale(domain=[-0.25, 0.06])
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "label:N",
            sort=frame["label"].to_list(),
            title=None,
            axis=alt.Axis(labelLimit=LABEL_PX),
        ),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    title = "Quantile score minus its reference (points of capacity; more negative is better)"
    zero = alt.Chart(pl.DataFrame({"x": [0.0], "text": ["same as reference"]}))
    threshold = alt.Chart(pl.DataFrame({"x": [THRESHOLD], "text": ["planned: -0.03"]}))
    chart = (
        zero.mark_rule(color="#444444", strokeDash=[4, 3]).encode(x=alt.X("x:Q", scale=scale))  # ty: ignore[unresolved-attribute]
        + threshold.mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[2, 2]).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=scale)
        )
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower:Q", title=title, scale=scale), x2="upper:Q"
        )
        + base.mark_circle(size=70, opacity=1.0).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("difference:Q", title=title, scale=scale)
        )
    ).properties(
        width=CONTENT_WIDTH_PX - LABEL_PX - 20,
        height=ROW_HEIGHT_PX * frame.height,
        title=alt.TitleParams(
            [
                "Stage 0 lowers the Primary quantile score by 0.067 points. Dot: estimate. Line:",
                "interval from resampling whole months and seeds. Orange rule: planned -0.03.",
            ],
            anchor="start",
            fontSize=12,
        ),
    )
    _save(chart=chart, name="figure_2")


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURES_DIR / f"{name}.svg")
    chart.save(common.WORK_DIR / "results" / f"{name}.png", scale_factor=2)


def main() -> None:
    """Print the quoted numbers and draw the figures."""
    results = common.WORK_DIR / "results"
    planned = pl.read_parquet(results / "planned.parquet")
    exploratory = pl.read_parquet(results / "exploratory.parquet")
    mismatch = pl.read_parquet(results / "mismatch.parquet")
    post_hoc = pl.read_parquet(results / "post_hoc_mismatch.parquet")
    absolute = pl.read_parquet(results / "absolute.parquet")
    pl.Config.set_tbl_rows(100).set_tbl_cols(20).set_fmt_str_lengths(60)
    print(planned)
    print(exploratory.filter(~pl.col("label").str.ends_with(" only")))
    print(mismatch)
    print(post_hoc)
    print(
        absolute.filter((pl.col("scope") == "all leads") & (pl.col("metric") == "quantile_score"))
    )
    figure_1(mismatch=mismatch, post_hoc=post_hoc)
    figure_2(planned=planned, exploratory=exploratory)


if __name__ == "__main__":
    main()
