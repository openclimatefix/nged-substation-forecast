"""Draw experiment 181's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00181_block import ONSET_TOLERANCE
from e00181_blocks17 import block_days
from e00181_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00181-switching-block-detector-and-like-for-like-table"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
ARMS: Final[dict[str, str]] = {
    "S0": "Block detector, odd-month noise unit (headline)",
    "S2": "Block detector, previous-7-day noise unit (post hoc)",
    "S1": "Block detector plus neighbour support",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
THRESHOLDS: Final[list[float]] = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_curve() -> alt.TopLevelMixin:
    """Block days found against unexplained odd-month alarms, by threshold and arm."""
    days = block_days()
    rows = []
    for arm in ARMS:
        candidates = pl.read_parquet(WORK_DIR / f"block_candidates_{arm}.parquet")
        for tau in THRESHOLDS:
            kept = candidates.filter(pl.col("z") > tau)
            found = sum(
                kept.filter(
                    pl.col("series") == d["series"],
                    pl.col("stamp").is_between(
                        d["day_first_stamp"] - ONSET_TOLERANCE,
                        d["day_last_stamp"] + ONSET_TOLERANCE,
                    ),
                ).height
                > 0
                for d in days.iter_rows(named=True)
            )
            rows.append(
                {
                    "arm": ARMS[arm],
                    "alarms": kept.filter(pl.col("odd"), ~pl.col("explained")).height,
                    "found": found,
                }
            )
    data = pl.DataFrame(rows).filter(pl.col("alarms") > 0)
    budget = (
        alt.Chart(pl.DataFrame({"alarms": [80]})).mark_rule(strokeDash=[4, 3]).encode(x="alarms:Q")  # ty: ignore[unresolved-attribute]
    )
    lines = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "alarms:Q",
                scale=alt.Scale(type="log", domain=[40, 3000]),
                title="Unexplained alarms in odd-numbered months (log scale; fewer is better)",
            ),
            y=alt.Y("found:Q", scale=alt.Scale(domain=[0, 17]), title="Block days found of 17"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARMS.values()), range=COLOURS),
                legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=400),
            ),
        )
    )
    return (budget + lines).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="At RO-NC's budget of 80 alarms the headline block detector finds 0 of 17 days",
            line="Dashed rule: 80 alarms, the budget of RO-NC. RO-NC and the pair alarm find 0 "
            "of 17 at their own thresholds. Each dot is one threshold.",
        ),
    )


def figure_scatter() -> alt.TopLevelMixin:
    """The best block score of each block day against the threshold at the budget."""
    summary = json.loads((WORK_DIR / "block_summary.json").read_text())
    data = pl.read_parquet(WORK_DIR / "why_missed.parquet").with_columns(
        label=pl.col("site") + " " + pl.col("local_date").cast(pl.String)
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=90, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "block_hours:Q",
                scale=alt.Scale(domain=[0, 30]),
                title="Length of the best-fitting block (hours)",
            ),
            y=alt.Y(
                "block_z:Q", scale=alt.Scale(domain=[0, 6]), title="Best block score (noise units)"
            ),
            tooltip=["label", "block_hours", "block_z"],
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [summary["S0"]["threshold"]]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return (points + rule).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="The block days sit at 1 to 4 noise units, under the alarm threshold of 4.4",
            line="Each dot is one of the 17 block days, on the convex-model residual. Dashed rule: "
            "the threshold that leaves 80 unexplained odd-month alarms.",
        ),
    )


def figure_control() -> alt.TopLevelMixin:
    """Recall of injected blocks by height, one line per length."""
    data = pl.read_parquet(WORK_DIR / "block_injection_control.parquet").with_columns(
        length=pl.col("length_h").cast(pl.Int64).cast(pl.String) + " hours"
    )
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("height:Q", title="Injected block height (noise units)"),
            y=alt.Y(
                "recall:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of injected blocks alarmed",
            ),
            color=alt.Color(
                "length:N",
                sort=["6 hours", "12 hours", "24 hours", "48 hours"],
                scale=alt.Scale(
                    domain=["6 hours", "12 hours", "24 hours", "48 hours"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The detector finds injected blocks of 5 noise units or more",
                line="150 blocks per length, added one at a time to the real residual in "
                "even-numbered months, at the budgeted threshold. Under 4 noise units it misses "
                "most.",
            ),
        )
    )


def figure_like_for_like() -> alt.TopLevelMixin:
    """Real transfers found by each residual, by whether the XGBoost panel has readings."""
    table = pl.read_parquet(WORK_DIR / "like_for_like.parquet")
    groups = {
        "All 35": table,
        "Readings in both panels (29)": table.filter(pl.col("xgboost_has_readings")),
        "No readings in XGBoost panel (6)": table.filter(~pl.col("xgboost_has_readings")),
    }
    rows = []
    for name, part in groups.items():
        rows.append(
            {"group": name, "residual": "XGBoost", "found": int(part["found_xgboost"].sum())}
        )
        rows.append(
            {"group": name, "residual": "Convex model", "found": int(part["found_convex"].sum())}
        )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("group:N", sort=list(groups), title=None, axis=alt.Axis(labelLimit=320)),
            yOffset="residual:N",
            x=alt.X("found:Q", title="Real transfers found (RO-NC)"),
            color=alt.Color(
                "residual:N",
                scale=alt.Scale(
                    domain=["XGBoost", "Convex model"], range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Four of 6 extra finds are transfers the XGBoost panel cannot see",
                line="RO-NC on the 35 labelled real transfers lasting over 1 day, same matching "
                "rule for both residuals.",
            ),
        )
    )


def main() -> None:
    """Draw the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_like_for_like().save(OUT_DIR / "figure_1_like_for_like.svg")
    figure_curve().save(OUT_DIR / "figure_2_block_days_found.svg")
    figure_scatter().save(OUT_DIR / "figure_3_block_scores.svg")
    figure_control().save(OUT_DIR / "figure_4_injected_blocks.svg")
    print("wrote 4 figures")


if __name__ == "__main__":
    main()
