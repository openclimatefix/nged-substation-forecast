# ruff: noqa: E501
"""Draw the figures of experiment 298 as SVG beside the write-up.

uv run --no-sync python e00298_charts.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import altair as alt
import e00298_common as common
import e00298_report_cleaning as rc
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00298-lsc-long-deep-detector-and-refit-cleaning"
)
ARMS: Final[dict[str, str]] = {
    "none": "No cleaning",
    "oracle": "Oracle (true days excluded)",
    "C": "Cleaned with detector C",
    "E3": "Cleaned with detector E3 (pair test)",
    "F-C": "Cleaned with detector F-C (neighbours)",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
]
WIDTH: Final[int] = CONTENT_WIDTH_PX - 120


def fig_share() -> None:
    """Share learned as normal after the end, by duration and arm (25% of load)."""
    worlds = rc.world_table(rc.load())
    rows = []
    for arm in ARMS:
        for duration in (1, 3, 6, 7, 14, 30, 90):
            sel = worlds.filter(
                (pl.col("mode") == arm)
                & (pl.col("size") == 0.25)
                & (pl.col("duration") == duration)
                & ~pl.col("metered")
                & (pl.col("shape") == "share")
            )
            mean, lo, hi = rc.boot(sel["share_0_2"].to_numpy())
            rows.append(
                {
                    "arm": ARMS[arm],
                    "days": duration,
                    "share": mean * 100,
                    "lo": lo * 100,
                    "hi": hi * 100,
                }
            )
    data = pl.DataFrame(rows).to_pandas()
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(ARMS.values()), range=ARM_COLOURS),
        legend=alt.Legend(title=None, orient="top", columns=2),
    )
    x = alt.X(
        "days:Q",
        scale=alt.Scale(type="log", domain=[0.8, 110]),
        axis=alt.Axis(values=[1, 3, 7, 14, 30, 90], title="Length of the transfer (days)"),
    )
    line = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=x,
            y=alt.Y("share:Q", title="Share learned as normal (%; lower is better)"),
            color=colour,
        )
    )
    band = alt.Chart(data).mark_rule(strokeWidth=1.5).encode(x=x, y="lo:Q", y2="hi:Q", color=colour)  # ty: ignore[unresolved-attribute]
    chart = (band + line).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            "Share of a transfer of 25% of load that the no-lag forecast learns as normal",
            subtitle=[
                "In the 48 hours after the transfer ends. Dot: mean of 40 worlds (10 substations, 4 onset dates).",
                "Line: 95% interval from resampling the 40 worlds.",
            ],
            anchor="start",
        ),
    )
    chart.configure_view(stroke=None).save(OUT / "share_learned_by_duration.svg")


def fig_recall() -> None:
    """Injected recall by duration for the main detector variants."""
    d = pl.read_parquet(common.EXPERIMENT_DIR / "injected_detection.parquet").filter(
        (pl.col("run") == "true")
        & ~pl.col("metered")
        & (pl.col("shape") == "share")
        & (pl.col("size") == 0.25)
        & (pl.col("onset").dt.month() % 2 == 0)
        & (pl.col("duration") > 0)
        & pl.col("variant").is_in(["A3", "B", "C", "E3", "F-C"])
    )
    data = d.group_by("variant", "duration").agg(recall=pl.col("found").mean() * 100).to_pandas()
    domain = ["A3", "B", "C", "E3", "F-C"]
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "duration:Q",
                scale=alt.Scale(type="log", domain=[0.8, 110]),
                axis=alt.Axis(values=[1, 3, 7, 14, 30, 90], title="Length of the transfer (days)"),
            ),
            y=alt.Y(
                "recall:Q",
                title="Transfers found (%; higher is better)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(domain=domain, range=ARM_COLOURS),
                legend=alt.Legend(title="Detector", orient="top", direction="horizontal"),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                "Share of injected transfers (25% of load, no metered partner) that each detector finds",
                subtitle=[
                    "Thresholds give 2 false alarms per Primary-year on the real record. 40 worlds per point."
                ],
                anchor="start",
            ),
        )
    )
    chart.configure_view(stroke=None).save(OUT / "injected_recall_by_duration.svg")


def fig_stratum() -> None:
    """Matches in the 19-shift long-and-deep stratum."""
    rows = []
    for text in (Path(__file__).parent / "printed_stratum.txt").read_text().splitlines():
        m = re.match(
            r"(.+?)\s+alarms=\s*([\d.]+)\s+stratum: (\d+)/19 apm=\s*([\d.]+) mw=([\d.]+)\s+rest: (\d+)/16",
            text,
        )
        if m:
            rows.append({"detector": m[1].strip(), "stratum": int(m[3]), "rest": int(m[6])})
    keep = [
        "LR_b20 (302 L0)",
        "RO-NC R2",
        "RO-NC current",
        "learned re-ranker",
        "A3",
        "B",
        "C",
        "E3",
        "F-C",
    ]
    data = (
        pl.DataFrame(rows)
        .filter(pl.col("detector").is_in(keep))
        .unpivot(index="detector", variable_name="group", value_name="matches")
        .with_columns(
            group=pl.when(pl.col("group") == "stratum")
            .then(pl.lit("Long and deep (19 shifts)"))
            .otherwise(pl.lit("Other (16 shifts)"))
        )
        .to_pandas()
    )
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("detector:N", sort=keep, title=None),
            x=alt.X("matches:Q", title="Real load shifts matched"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_SKY]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            yOffset="group:N",
        )
        .properties(
            width=WIDTH,
            height=300,
            title=alt.TitleParams(
                "Real load shifts of more than 1 day that each detector matches",
                subtitle=[
                    "Long and deep: at least 2 days and at least 10% of load. Detectors A3 to F-C are this experiment's, at 2 false alarms per Primary-year."
                ],
                anchor="start",
            ),
        )
    )
    chart.configure_view(stroke=None).save(OUT / "real_matches_by_stratum.svg")


if __name__ == "__main__":
    fig_share()
    fig_recall()
    fig_stratum()
