"""Draw the figure of experiment 177's write-up from the two stored reports.

uv run --no-sync python ml-experiments/00177-*/e00177_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BASE: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00177-dp-worst-days-other-data-sources"
)
LABELS: Final[dict[str, str]] = {
    "s1": "Logged switching (confirmed)",
    "s3": "Reading quality",
    "s4": "Own bad weekday",
    "union": "Union of the three above",
    "p1": "DFS event on the day",
    "p2": "High system price (top 5%)",
}
REPORTS: Final[tuple[str, ...]] = ("report.json", "report_stage2.json")


def main() -> None:
    """Write the figure."""
    rows = []
    for name in REPORTS:
        report = json.loads((BASE / "work" / name).read_text())
        for key, label in LABELS.items():
            if key not in report or (name == "report_stage2.json" and key == "union"):
                continue
            result = report[key]
            rows.append(
                {
                    "source": "Union of stage 1 sources" if key == "union" else label,
                    "observed": result["observed_of_37"],
                    "ordinary_mean": result["ordinary_mean_of_37"],
                    "low": result["ordinary_95pct_range"][0],
                    "high": result["ordinary_95pct_range"][1],
                }
            )
    frame = pl.DataFrame(rows)
    order = frame["source"].to_list()
    base = alt.Chart(frame).encode(y=alt.Y("source:N", sort=order, title=None))
    band = base.mark_bar(height=14, opacity=0.35, color="#9A9A9A").encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Days explained, out of 37"), x2="high:Q"
    )
    mean = base.mark_tick(color="#555555", thickness=2, size=18).encode(x="ordinary_mean:Q")  # ty: ignore[unresolved-attribute]
    obs = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(x="observed:Q")  # ty: ignore[unresolved-attribute]
    chart = (band + mean + obs).properties(
        width=480,
        height=200,
        title="Days explained: dot = 37 worst days, band = 37 ordinary days",
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "fig1_sources.svg")


if __name__ == "__main__":
    main()
