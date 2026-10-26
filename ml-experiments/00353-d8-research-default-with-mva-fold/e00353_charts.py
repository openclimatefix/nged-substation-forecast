# ruff: noqa: E501
"""Draw the figures of experiment 353 from `work/scores.json` and `work/wide_scored_rows.parquet`.

uv run --no-sync python ml-experiments/00353-*/e00353_charts.py
"""

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00353-d8-research-default-with-mva-fold"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
PNG_DIR: Final[Path] = WORK / "png"
WIDTH: Final[int] = 640
SLICE_LABELS: Final[dict[str, str]] = {
    "all": "All rows",
    "sunny": "Sunny daylight",
    "daylight": "All daylight",
    "winter_night": "Winter nights",
    "cold_night": "Cold nights",
    "high_block": "High block (6 to 8 MW level)",
    "low_block": "Low block (1 to 3 MW level)",
}
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "c1": ("B minus C: fold-aware fit versus reading as signed demand", ocf.BRAND_ORANGE),
    "c1_fold": ("B-fold minus C: the same, with quantiles folded at zero", ocf.DATA_BLUE),
}
GAPS: Final[dict[str, tuple[str, str]]] = {
    "a_bg2": ("A: the card's forecast (experiment 252)", ocf.DATA_BLUE),
    "c_bg2": ("C: fitted to the reading as signed demand", ocf.DATA_SKY),
    "c3": ("B: fold-aware fit", ocf.BRAND_ORANGE),
    "c7": ("B with a band from D8's own earlier months (post hoc)", ocf.DATA_GREEN),
}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def forest(
    *,
    scores: dict,
    series: dict[str, tuple[str, str]],
    x_title: str,
    title: str,
    subtitle: str,
    name: str,
    groups: list[str],
    height: int,
) -> None:
    """Draw each contrast per group of rows with its 95% interval."""
    rows = []
    for key, (label, _) in series.items():
        for group in groups:
            estimate, low, high = scores["contrast"][key][group]["pooled"]
            rows.append(
                {
                    "group": SLICE_LABELS[group],
                    "arm": label,
                    "estimate": estimate,
                    "low": low,
                    "high": high,
                }
            )
    domain = [v[0] for v in series.values()]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=[SLICE_LABELS[g] for g in groups], title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color("arm:N", scale=alt.Scale(domain=domain, range=[v[1] for v in series.values()]),
                        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500)),
    )  # fmt: skip
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=x_title), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=height, title=_title(title, subtitle))
    _save(chart, name=name)


def absolute(*, scores: dict) -> None:
    """Draw every arm's and reference's absolute score on all rows and on sunny daylight rows."""
    names = {
        "bg2": ("BG2 (production style)", ocf.DATA_GREEN),
        "manual": ("Manual heuristic", ocf.DATA_PURPLE),
        "climatology": ("Climatology", "#8C8C8C"),
        "a": ("A: card (252)", ocf.DATA_BLUE),
        "b": ("B: fold-aware + stage 2", ocf.BRAND_ORANGE),
        "c": ("C: signed-demand + stage 2", ocf.DATA_SKY),
        "b_causal": ("B, own earlier-month band (post hoc)", "#5CEDAD"),
        "b_own": ("B, own-residual band (post hoc)", "#8CF2C7"),
        "raw_a": ("Fold-aware alone (152 A)", "#FF9A70"),
        "raw_n": ("Signed-demand alone (152 N)", "#9DBBFF"),
    }
    rows = [
        {"slice": SLICE_LABELS[s], "arm": label, "score": scores["absolute"][s][key]}
        for s in ("all", "sunny")
        for key, (label, _) in names.items()
    ]
    domain = [v[0] for v in names.values()]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=domain, title=None),
            x=alt.X("score:Q", title="Quantile score (points of capacity; lower is better)"),
            color=alt.Color("arm:N", scale=alt.Scale(domain=domain, range=[v[1] for v in names.values()]), legend=None),
            row=alt.Row("slice:N", title=None, sort=[SLICE_LABELS["all"], SLICE_LABELS["sunny"]]),
        )
        .properties(width=WIDTH - 150, height=130, title=_title(
            "Every D8 forecast scores worse than BG2, the manual heuristic, and (on all rows) climatology",
            "Mean quantile score at D8 over leads 0.5 to 168 hours, scored year 2025-07-02 to 2026-06-30. Bars: absolute score, no interval."))
    )  # fmt: skip
    _save(chart, name="figure_3_absolute")


def weeks(*, capacity_mw: float) -> None:
    """Draw three example weeks of day-ahead forecasts against the reading."""
    wide = pl.read_parquet(WORK / "wide_scored_rows.parquet").filter(
        pl.col("lead_hours").is_between(15.5, 39.0)
    )
    pieces = []
    for label, start in (
        ("July 2025", datetime(2025, 7, 7, tzinfo=UTC)),
        ("January 2026", datetime(2026, 1, 12, tzinfo=UTC)),
        ("April 2026", datetime(2026, 4, 13, tzinfo=UTC)),
    ):
        part = wide.filter(
            (pl.col("valid_time") >= start) & (pl.col("valid_time") < start + timedelta(days=7))
        )
        pieces.append(
            part.select(
                week=pl.lit(label),
                day=(pl.col("valid_time") - pl.lit(start)).dt.total_seconds() / 86400.0,
                actual=pl.col("actual") * capacity_mw / 100.0,
                b=pl.col("b__q50") * capacity_mw / 100.0,
                a=pl.col("a__q50") * capacity_mw / 100.0,
                low=pl.col("b__q05") * capacity_mw / 100.0,
                high=pl.col("b__q95") * capacity_mw / 100.0,
            )
        )
    data = (
        pl.concat(pieces)
        .sort("week", "day")
        .with_columns(
            segment=(pl.col("day").diff().over("week") > 0.05)
            .fill_null(False)
            .cum_sum()
            .over("week")
        )
    )
    x = alt.X(
        "day:Q",
        title="Days since the start of the week (day 0 is a Monday, 00:00 UTC)",
        scale=alt.Scale(domain=[0, 7]),
        axis=alt.Axis(values=list(range(8))),
    )
    band = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.2, color=ocf.BRAND_ORANGE, clip=True)
        .encode(x=x, y=alt.Y("low:Q", title="Apparent power (MVA)"), y2="high:Q")
    )
    lines = [
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(color=colour, strokeWidth=width, clip=True)
        .encode(x=x, y="v:Q", detail="segment:N")
        .transform_calculate(v=f"datum.{column}")
        for column, colour, width in (
            ("actual", "#222222", 1.5),
            ("a", ocf.DATA_BLUE, 1.2),
            ("b", ocf.BRAND_ORANGE, 1.5),
        )
    ]
    chart = (
        alt.layer(band, *lines)
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("week:N", title=None, sort=["July 2025", "January 2026", "April 2026"]))
        .properties(title=_title(
            "D8's readings move in blocks that the day-ahead forecast smooths over",
            "Black: reading. Orange: arm B median, shaded 5% to 95%. Blue: arm A median (the card's forecast). Forecasts made 15.5 to 39 hours ahead."))
    )  # fmt: skip
    _save(chart, name="figure_4_weeks")


def reverse_flow(*, scores: dict) -> None:
    """Draw the half-hours where the fitted flow is negative, by month and by hour of day."""
    flow = scores["reverse_flow"]
    months = pl.DataFrame(
        {
            "x": sorted(flow["negative_by_month"]),
            "n": [flow["negative_by_month"][m] for m in sorted(flow["negative_by_month"])],
        }
    )
    hours = pl.DataFrame(
        {
            "x": [int(h) for h in flow["negative_by_utc_hour"]],
            "n": list(flow["negative_by_utc_hour"].values()),
        }
    )
    left = (
        alt.Chart(months)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(
            x=alt.X(
                "x:N",
                title="Month of the valid time (months with none are not shown)",
                sort=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("n:Q", title="Half-hours"),
        )
    )
    right = (
        alt.Chart(hours)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "x:O", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18], labelAngle=0)
            ),
            y=alt.Y("n:Q", title="Half-hours"),
        )
    )
    chart = alt.vconcat(left.properties(width=WIDTH, height=110), right.properties(width=WIDTH, height=110)).properties(
        title=_title(
            "The fitted flow goes negative in August and September 2025, and at night as well as at midday",
            "Count of valid times (half-hours) where the fold-aware fit's signed flow is below zero, scored year, all leads. Not solar export alone."))  # fmt: skip
    _save(chart, name="figure_5_reverse_flow")


def main() -> None:
    """Draw every figure."""
    scores = json.loads((WORK / "scores.json").read_text())
    forest(
        scores=scores, series=CONTRASTS,
        x_title="Quantile score difference (points of capacity; below zero favours the fold-aware fit)",
        title="Modelling the fold moves D8's score by a few tenths of a point, and over all rows the interval includes zero",
        subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Both arms include the stack's night offset and recipe R. Scored year, leads 0.5 to 168 hours.",
        name="figure_1_fold_effect", groups=["all", "sunny", "winter_night", "cold_night", "high_block", "low_block"], height=300,
    )  # fmt: skip
    forest(
        scores=scores, series=GAPS,
        x_title="Quantile score minus BG2 (points of capacity; below zero is better)",
        title="The fold does not close D8's gap to BG2, but a band fitted on D8's own earlier months closes most of it",
        subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Scored year, leads 0.5 to 168 hours. The first three arms use a band borrowed from six other Primaries.",
        name="figure_2_gap_to_bg2", groups=["all", "sunny", "winter_night", "cold_night", "high_block", "low_block"], height=300,
    )  # fmt: skip
    absolute(scores=scores)
    weeks(capacity_mw=scores["effective_capacity_mw"])
    reverse_flow(scores=scores)


if __name__ == "__main__":
    main()
