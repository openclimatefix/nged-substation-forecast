"""Draw the figures of experiment 189's write-up from the stored summaries.

uv run --no-sync python ml-experiments/00189-*/e00189_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00189_common import WORK_DIR, real_transfers

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00189-switching-neural-onset-end-block-detector"
)
WIDTH: Final[int] = 600
PAIR_ALARM_LONG: Final[int] = 5
"""Experiment 156's pair alarm on the 35 transfers, as quoted in the brief."""


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def main() -> None:
    """Draw the three figures."""
    s0 = json.loads((WORK_DIR / "summary_seed0.json").read_text())
    s1 = json.loads((WORK_DIR / "summary_seed1.json").read_text())
    realism = json.loads((WORK_DIR / "realism_default.json").read_text())
    long = real_transfers().filter(pl.col("hours") > 24)
    ronc_found, ronc_onset = int(long["found_ronc"].sum()), int(long["onset_found_ronc"].sum())

    rows = []
    for label, s in (
        ("Network, 14 epochs (pre-registered)", s0),
        ("Network, 40 epochs (post hoc)", s1),
    ):
        for budget, n in s["found_by_budget_long"].items():
            rows.append({"run": label, "alarms": int(budget), "found": n})
    curve = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "alarms:Q", scale=alt.Scale(type="log"), title="Unexplained odd-month onset alarms"
            ),
            y=alt.Y(
                "found:Q",
                scale=alt.Scale(domain=[0, 20]),
                title="Transfers over 1 day found (of 35)",
            ),
            color=alt.Color(
                "run:N", scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), title=None
            ),
        )
    )
    ronc = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame({"alarms": [80], "found": [ronc_found], "run": ["RO-NC (80 alarms)"]})
        )
        .mark_point(shape="diamond", size=200, filled=True, color=ocf.DATA_GREEN)
        .encode(x="alarms:Q", y="found:Q")
    )
    _save((curve + ronc).properties(width=WIDTH, height=300), name="figure_1_found_against_alarms")

    bars = pl.DataFrame(
        [
            {
                "detector": "Pair alarm (experiment 156)",
                "rule": "Onset and end",
                "found": PAIR_ALARM_LONG,
            },
            {"detector": "RO-NC", "rule": "Onset and end", "found": ronc_found},
            {"detector": "RO-NC", "rule": "Onset only", "found": ronc_onset},
            {"detector": "Network, 14 epochs", "rule": "Onset and end", "found": s0["long_found"]},
            {
                "detector": "Network, 14 epochs",
                "rule": "Onset only",
                "found": s0["long_onset_found"],
            },
            {"detector": "Network, 40 epochs", "rule": "Onset and end", "found": s1["long_found"]},
            {
                "detector": "Network, 40 epochs",
                "rule": "Onset only",
                "found": s1["long_onset_found"],
            },
        ]
    )
    chart2 = (
        alt.Chart(bars)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("detector:N", title=None, sort=None),
            x=alt.X("found:Q", title="Transfers over 1 day found (of 35) at 80 alarms"),
            yOffset="rule:N",
            color=alt.Color(
                "rule:N", scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_SKY]), title=None
            ),
        )
        .properties(width=WIDTH, height=260)
    )
    _save(chart2, name="figure_2_real_transfers_found")

    recall = pl.DataFrame(
        [
            {
                "set": "Real transfers",
                "group": "Over 1 day",
                "recall": realism["standin_recall_real_long"],
            },
            {
                "set": "Injected",
                "group": "Over 1 day",
                "recall": realism["standin_recall_injected_long"],
            },
            {
                "set": "Real transfers",
                "group": "All durations",
                "recall": realism["standin_recall_real_all"],
            },
            {
                "set": "Injected",
                "group": "All durations",
                "recall": realism["standin_recall_injected_all"],
            },
        ]
    )
    chart3 = (
        alt.Chart(recall)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("group:N", title=None),
            xOffset="set:N",
            y=alt.Y(
                "recall:Q",
                title="Share with a 24-hour step alarm at the onset",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "set:N", scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), title=None
            ),
        )
        .properties(width=WIDTH, height=260)
    )
    _save(chart3, name="figure_3_injector_realism")


if __name__ == "__main__":
    main()
