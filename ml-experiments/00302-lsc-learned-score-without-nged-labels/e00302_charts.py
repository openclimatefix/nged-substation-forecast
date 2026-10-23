# ruff: noqa: E501
"""Draw the figures of experiment 302 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python e00302_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00302_common import WORK_DIR, stamp_months
from e00302_score import shifts_table
from studies.charts import CONTENT_WIDTH_PX

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00302-lsc-learned-score-without-nged-labels"
)
W: Final[int] = CONTENT_WIDTH_PX - 150
NAMES: Final[dict[str, str]] = {
    "L0": "L0 labelled, even months",
    "L1": "L1 labelled, odd months",
    "L2": "L2 label-free negatives",
    "L3": "L3 + per-Primary rate",
    "L3_2": "L3 at 2 per Primary-year",
    "L3_5": "L3 at 5 per Primary-year",
}
ORDER: Final[list[str]] = list(NAMES.values())


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def fig_matches() -> None:
    """Load shifts matched, split by whether the shift starts in a month the model trained on."""
    months = stamp_months()
    sh = shifts_table()
    odd = {
        r["card_id"]: int(months[min(r["onset"], len(months) - 1)]) % 2 == 1
        for r in sh.iter_rows(named=True)
    }
    hits = pl.read_parquet(WORK_DIR / "shift_hits.parquet").filter(
        (pl.col("variant") == "all") & pl.col("found")
    )
    hits = hits.with_columns(
        arm_name=pl.col("arm").replace(NAMES),
        start=pl.col("card_id").map_elements(
            lambda c: "odd month" if odd[c] else "even month", return_dtype=pl.String
        ),
    )
    t = hits.group_by("arm_name", "start").agg(found=pl.len())
    chart = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm_name:N", sort=ORDER, title=None),
            x=alt.X("found:Q", title="Logged load shifts matched, out of 35"),
            color=alt.Color(
                "start:N",
                scale=alt.Scale(
                    domain=["odd month", "even month"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Month the load shift starts in", orient="top"),
            ),
        )
        .properties(width=W, height=200)
    )
    save(chart, "figure_1_matches")


def fig_harm() -> None:
    """Ordinary-day harm on held-out months with 95% intervals."""
    summary = json.loads((WORK_DIR / "harm_summary.json").read_text())
    t = pl.DataFrame(
        [
            {"arm_name": NAMES[a], "point": v["point"], "lo": v["lo"], "hi": v["hi"]}
            for a, v in summary.items()
        ]
    )
    base = alt.Chart(t).encode(y=alt.Y("arm_name:N", sort=ORDER, title=None))
    rule = (
        alt.Chart(pl.DataFrame({"x": [-0.05, 0.05]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ocf.DATA_BLUE)
        .encode(x="x:Q")
    )
    chart = (
        base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q", title="Ordinary-day harm, points of effective capacity (lower is better)"
            ),
            x2="hi:Q",
        )
        + base.mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
        + rule
    ).properties(width=W, height=200)
    save(chart, "figure_2_ordinary_day_harm")


def fig_rates() -> None:
    """Alarms per Primary-year by Primary for L1, L2, L3."""
    c = (
        pl.read_parquet(WORK_DIR / "alarm_counts.parquet")
        .filter((pl.col("variant") == "all") & pl.col("arm").is_in(["L1", "L2", "L3"]))
        .group_by("arm", "site", "years")
        .agg(alarms=pl.col("alarms").mean())
        .with_columns(
            rate=pl.col("alarms") / pl.col("years"), arm_name=pl.col("arm").replace(NAMES)
        )
    )
    chart = (
        alt.Chart(c)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "site:N",
                title="Primary substation",
                sort=alt.EncodingSortField("rate", op="max", order="descending"),
            ),
            y=alt.Y("rate:Q", title="Merged alarms per Primary-year"),
            color=alt.Color(
                "arm_name:N",
                scale=alt.Scale(
                    domain=[NAMES[a] for a in ("L1", "L2", "L3")],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=W, height=240)
    )
    save(chart, "figure_3_alarm_rate_by_primary")


def fig_contamination() -> None:
    """Real training negatives of L2 inside the spans L1 removes."""
    t = pl.DataFrame(
        {
            "kind": [
                "Inside a logged or labelled span",
                "Inside a D8 multi-day drop",
                "On one of the 17 block days",
                "None of the three",
            ],
            "count": [364, 93, 2, 497],
        }
    )
    chart = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("kind:N", sort=None, title=None),
            x=alt.X(
                "count:Q",
                title="Odd-month RO-NC candidates trained as negatives by L2 (of 913; the first three overlap)",
            ),
        )
        .properties(width=W, height=140)
    )
    save(chart, "figure_4_label_contamination")


if __name__ == "__main__":
    fig_matches()
    fig_harm()
    fig_rates()
    fig_contamination()
