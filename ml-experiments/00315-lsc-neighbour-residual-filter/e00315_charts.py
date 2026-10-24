"""Draw experiment 315's figure: alarms per matched shift against matched shifts.

Afterwards optimise it with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00315-lsc-neighbour-residual-filter/e00315_charts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00315-lsc-neighbour-residual-filter"
)
LABELS: Final[dict[tuple[str, str], str]] = {
    ("r24", "plain"): "24 h mean residual, plain",
    ("r24", "filtered"): "24 h mean residual, neighbour filter",
    ("pit24", "plain"): "band position, plain",
    ("pit24", "filtered"): "band position, neighbour filter",
}
COLOURS: Final[list[str]] = ["#306BFF", "#FF4901", "#10C5F7", "#B701FF", "#17E58F", "#7f7f7f"]
CARRIED: Final[list[dict]] = [
    {"arm": "learned score LR_b20", "matched": 10, "apm": 4.0},
    {"arm": "RO-NC", "matched": 20, "apm": 7.3},
]


def main() -> None:
    """Write the figure."""
    curves = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR / "00315-lsc-neighbour-residual-filter" / "curves.parquet"
    )
    rows = []
    for (signal, variant), label in LABELS.items():
        sub = (
            curves.filter((pl.col("signal") == signal) & (pl.col("variant") == variant))
            .filter(pl.col("matched") >= 5)
            .group_by("matched")
            .agg(pl.col("apm").min())
            .sort("matched")
        )
        rows += [
            {"arm": label, "matched": r["matched"], "apm": r["apm"]}
            for r in sub.iter_rows(named=True)
        ]
    data = pl.DataFrame(rows)
    domain = [*LABELS.values(), "learned score LR_b20", "RO-NC"]
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=domain, range=COLOURS),
        legend=alt.Legend(orient="top", columns=2, title=None),
    )
    lines = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("matched:Q", title="Real load shifts found (of 35)"),
            y=alt.Y("apm:Q", title="Alarms per found shift", scale=alt.Scale(domain=[0, 12])),
            color=colour,
        )
    )
    points = (
        alt.Chart(pl.DataFrame(CARRIED))
        .mark_point(size=120, filled=True, shape="diamond")
        .encode(x="matched:Q", y="apm:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    chart = (lines + points).properties(width=420, height=260)
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / "figure_1_alarms_per_match_against_recall.svg")


if __name__ == "__main__":
    main()
