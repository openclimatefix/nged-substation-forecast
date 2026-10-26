"""Draw the figures of experiment 352 from `work/scores.parquet` and `work/part1_cells.parquet`.

uv run --no-sync python ml-experiments/00352-*/e00352_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import e00352_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
SITES: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
FILTER_NAMES: Final[dict[str, str]] = {"kf": "with the Kalman filter", "nokf": "without the filter"}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
X_TITLE: Final[str] = "Solar capacity as a multiple of the pinned capacity (1.0 = default)"
Y_TITLE: Final[str] = "Quantile score (points of capacity; smaller is better)"


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _title(lines: list[str]) -> Any:
    return alt.TitleParams(text=lines, anchor="start", fontSize=13)


def _curves(*, scope: str, lead: str) -> pl.DataFrame:
    table = pl.read_parquet(common.WORK_DIR / "scores.parquet").filter(
        pl.col("metric") == "qs",
        pl.col("lead") == lead,
        pl.col("scope") == scope,
        pl.col("statistic").str.starts_with("score "),
    )
    parts = table["statistic"].str.extract_groups(r"score s(?P<scale>[\d.]+) (?P<filter>\w+)")
    return table.with_columns(
        scale=parts.struct.field("scale").cast(pl.Float64),
        filter=parts.struct.field("filter").replace(FILTER_NAMES),
    )


def headline(*, lead: str, name: str, scope: str, title: list[str]) -> None:
    """Draw score against scale, with and without the filter, with 95% intervals."""
    data = _curves(scope=scope, lead=lead).to_pandas()
    base = _chart(data).encode(
        x=alt.X("scale:Q", title=X_TITLE, axis=alt.Axis(values=list(common.SCALES))),
        color=alt.Color(
            "filter:N",
            scale=alt.Scale(domain=list(FILTER_NAMES.values()), range=COLOURS),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    chart = (
        base.mark_errorband(opacity=0.15).encode(
            y=alt.Y("lower:Q", title=Y_TITLE, scale=alt.Scale(zero=False)), y2="upper:Q"
        )
        + base.mark_line(point=True).encode(y="estimate:Q")
    ).properties(width=WIDTH, height=260, title=_title(title))
    chart.save(DOCS_DIR / f"{name}.svg")


def per_primary(*, lead: str, name: str) -> None:
    """Draw the sunny score against scale for each Primary."""
    frames = []
    for site in SITES:
        frame = _curves(scope=f"sunny {site}", lead=lead).with_columns(site=pl.lit(site))
        frames.append(frame)
    data = pl.concat(frames).to_pandas()
    chart = (
        _chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("scale:Q", title=None, axis=alt.Axis(values=list(common.SCALES))),
            y=alt.Y("estimate:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "filter:N",
                scale=alt.Scale(domain=list(FILTER_NAMES.values()), range=COLOURS),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=170, height=130)
        .facet(facet=alt.Facet("site:N", title=None, sort=SITES), columns=3)
        .properties(
            title=_title(
                [
                    "Each Primary's sunny-half-hour score against the capacity scale",
                    f"Sunniest third, daylight, {lead}, scored year 2025-07 to 2026-06.",
                    "x: capacity as a multiple of the pin. y: quantile score (smaller is better)",
                ]
            )
        )
    )
    chart.save(DOCS_DIR / f"{name}.svg")


def part1() -> None:
    """Draw experiment 349's sunny contrast by Primary, lead band, and month."""
    cells = pl.read_parquet(common.WORK_DIR / "part1_cells.parquet")
    by_band = (
        cells.group_by("label", "lead_band")
        .agg(
            difference=(pl.col("difference") * pl.col("n")).sum() / pl.col("n").sum(),
        )
        .to_pandas()
    )
    bands = ["0 to 24 h", "24 to 72 h", "72 to 168 h"]
    bars = (
        _chart(by_band)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort=SITES, title=None),
            yOffset=alt.YOffset("lead_band:N", sort=bands),
            x=alt.X(
                "difference:Q",
                title="Experiment 349 estimate minus pin, quantile score (below 0 is better)",
            ),
            color=alt.Color(
                "lead_band:N",
                sort=bands,
                scale=alt.Scale(
                    domain=bands, range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Lead", orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                [
                    "The estimate scored worse at D16 and D10 and better at D9, D14 and D5",
                    "Sunniest third of daylight half-hours, scored year 2025-07 to 2026-06",
                ]
            ),
        )
    )
    bars.save(DOCS_DIR / "figure_1a_experiment_349_by_primary_and_lead.svg")
    by_month = (
        cells.group_by("label", "calendar_month")
        .agg(difference=(pl.col("difference") * pl.col("n")).sum() / pl.col("n").sum())
        .to_pandas()
    )
    heat = (
        _chart(by_month)
        .mark_rect()
        .encode(
            x=alt.X("calendar_month:O", title="Calendar month of the valid time"),
            y=alt.Y("label:N", sort=SITES, title=None),
            color=alt.Color(
                "difference:Q",
                scale=alt.Scale(
                    domain=[-0.5, 0, 1.0], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Score change", orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                [
                    "The pooled sign comes mostly from D5 and D16 in January to March",
                    "Experiment 349 estimate minus pin on sunny half-hours (orange is worse)",
                ]
            ),
        )
    )
    heat.save(DOCS_DIR / "figure_1b_experiment_349_by_primary_and_month.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    part1()
    headline(
        lead="0.5 to 168 h",
        name="figure_2_sunny_score_against_scale",
        scope="sunny",
        title=[
            "Lowering the capacity by 20% costs nothing on sunny half-hours; raising it does",
            "Six Primaries pooled, sunniest third, 0.5 to 168 h. Band: 95% interval, whole months",
        ],
    )
    per_primary(lead="0.5 to 168 h", name="figure_3_sunny_score_per_primary")
    headline(
        lead="0.5 to 168 h",
        name="figure_4_all_half_hours_score_against_scale",
        scope="all half-hours",
        title=[
            "On all half-hours the filter helps and the best capacity is the pin",
            "Six Primaries pooled, all half-hours, 0.5 to 168 h. Band: 95% interval, whole months",
        ],
    )
