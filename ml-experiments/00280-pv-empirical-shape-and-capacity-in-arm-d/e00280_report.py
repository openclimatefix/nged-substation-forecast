"""Draw experiment 280's figure and print the scores the write-up quotes.

Reads `work/scores.json` (from `e00280_score.py`) and writes `result_effect.svg` beside the
write-up: each arm's change in mean absolute error against the control, by slice and group.

    uv run --no-sync python ml-experiments/00280-*/e00280_report.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00280_common as common
import polars as pl
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN, DATA_PURPLE, DATA_SKY
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00280-pv-empirical-shape-and-capacity-in-arm-d"
)
ARMS: Final[dict[str, str]] = {
    "e0": "Empirical shape, pinned capacity (e0)",
    "e1": "Empirical shape, swapped level (e1)",
    "e2": "Empirical shape, fitted capacity (e2)",
    "f1": "Fleet shape, fitted capacity (f1)",
    "s1": "April to June factor from 2025 (s1)",
    "s2": "Oracle April to June factor (s2)",
}
PANELS: Final[tuple[tuple[str, str, str], ...]] = (
    ("primary", "aj_day", "D9 and D16, April to June daylight"),
    ("d5", "aj_day", "D5, April to June daylight (60% clip)"),
    ("others", "aj_day", "D10, D11, and D14, April to June daylight"),
    ("decision", "sunny", "Five decision substations, sunny half-hours"),
)


def main() -> None:
    """Print the assembly-configuration table and write the figure."""
    scores = json.loads((common.WORK_DIR / "scores.json").read_text())
    rows = []
    for group, slice_name, title in PANELS:
        for arm, label in ARMS.items():
            entry = scores.get(f"asm|{arm}|{group}|{slice_name}")
            if entry is None:
                continue
            d = entry["d_mae"]
            rows.append(
                {
                    "panel": title,
                    "arm": label,
                    "mean": d["mean"],
                    "low": d["low"],
                    "high": d["high"],
                }
            )
            print(f"{title} | {arm} | {d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]")
    frame = pl.DataFrame(rows)
    colours = [DATA_BLUE, DATA_SKY, BRAND_ORANGE, DATA_GREEN, DATA_PURPLE, "#808080"]
    colour = alt.Color(
        "arm:N", scale=alt.Scale(domain=list(ARMS.values()), range=colours), legend=None
    )
    y = alt.Y("arm:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=260))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    panels = []
    for _, _, title in PANELS:
        base = alt.Chart(frame.filter(pl.col("panel") == title))
        bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Change in mean absolute error (MW; negative is better)"),
            x2="high:Q",
            y=y,
            color=colour,
        )
        dots = base.mark_circle(size=70, opacity=1).encode(x="mean:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(bars, dots, zero).properties(
                width=CONTENT_WIDTH_PX - 280, height=110, title=title
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Only the oracle cut at D5 moves April to June error clearly, and it hurts",
            subtitle=[
                "Dot: change in median-forecast absolute error against the control",
                "(fleet shape, pinned capacity). Line: 99% interval, resampling whole weeks.",
                "Each panel has its own horizontal scale.",
                "Scored year 2025-07 to 2026-06, leads to 168 hours, equal substation weights.",
            ],
        )
    )
    chart.save(str(DOCS_DIR / "result_effect.svg"))
    chart.save(str(common.WORK_DIR / "result_effect.png"), scale_factor=2)


if __name__ == "__main__":
    main()
