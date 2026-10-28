# ruff: noqa: E501
"""Draw the figures of experiment 360 from `work/results.json`."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00360-bg2-member-drop-one-at-d8"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
GAM_STACK_SCORE: Final[float] = 8.338
"""Experiment 355's weekly-refit GAM stack at D8, the same 96,986 rows."""
LABELS: Final[dict[str, str]] = {
    "full": "All five members (BG2)",
    "drop_W": "Without weekly XGBoost",
    "drop_M": "Without monthly XGBoost",
    "drop_AN": "Without analogue",
    "drop_PT": "Without pooled by type",
    "drop_ST": "Without XGBoost stack",
    "only_W": "Weekly XGBoost alone",
    "only_M": "Monthly XGBoost alone",
    "only_AN": "Analogue alone",
    "only_PT": "Pooled by type alone",
    "only_ST": "XGBoost stack alone",
}


def main() -> None:
    """Write the D8 figure."""
    arms = json.loads((WORK / "results.json").read_text())["D8"]["arms"]
    table = pl.DataFrame(
        [
            {"arm": LABELS[n], "score": v["score"], "low": v["ci"][0], "high": v["ci"][1]}
            for n, v in arms.items()
        ]
    )
    order = list(LABELS.values())
    base = alt.Chart(table).encode(
        y=alt.Y("arm:N", sort=order, title=None),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "score:Q",
            title="Quantile score at D8 (points of capacity; lower is better)",
            scale=alt.Scale(zero=False),
        ),
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    rule = (
        alt.Chart(pl.DataFrame({"x": [GAM_STACK_SCORE]}))
        .mark_rule(strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    label = (
        alt.Chart(pl.DataFrame({"x": [GAM_STACK_SCORE], "t": ["GAM stack, weekly (8.338)"]}))
        .mark_text(align="left", dx=4, dy=-120, fontSize=11)
        .encode(x="x:Q", text="t:N")  # ty: ignore[unresolved-attribute]
    )
    chart = (bars + points + rule + label).properties(
        width=420,
        height=300,
        title=alt.TitleParams(
            "Four of BG2's five members score 7.5 to 7.7 alone at D8, against 8.34 for the GAM stack",
            subtitle="Points with 95% intervals (whole valid months resampled), 96,986 rows",
            anchor="start",
        ),
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "figure_1_members_at_d8.svg")
    print("wrote figure_1_members_at_d8")


if __name__ == "__main__":
    main()
