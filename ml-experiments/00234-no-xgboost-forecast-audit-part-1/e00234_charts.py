"""Draw the figures of experiment 234 from `scores.json`.

uv run --no-sync python ml-experiments/00234-no-xgboost-forecast-audit-part-1/e00234_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00234-no-xgboost-forecast-audit-part-1/scores.json"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00234-no-xgboost-forecast-audit-part-1"
)
KINDS: Final[dict[str, str]] = {
    "demand": "Primary demand",
    "solar": "Metered solar farms",
    "wind": "Metered wind farms",
}
TOLERANCE_PERCENT: Final[float] = 2.0
WIDTH: Final[int] = 560


def save(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write one chart as SVG beside the write-up."""
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def rules() -> alt.LayerChart:
    """Return the zero line and the 2% tolerance line."""
    frame = pl.DataFrame({"x": [0.0, TOLERANCE_PERCENT], "k": ["equal", "2% tolerance"]})
    return alt.Chart(frame).mark_rule(strokeDash=[4, 3], color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def main() -> None:
    """Draw the headline gaps and the tail gaps."""
    scores = json.loads(SCORES.read_text())
    rows = [
        {
            "type": KINDS[kind],
            "leads": band,
            "gap": e["qs"]["gap_percent"],
            "low": e["qs"]["gap_lo_percent"],
            "high": e["qs"]["gap_hi_percent"],
        }
        for kind, bands in scores.items()
        for band, e in bands.items()
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("type:N", title=None, sort=list(KINDS.values())),
        yOffset="leads:N",
        color=alt.Color(
            "leads:N",
            scale=alt.Scale(
                domain=["12-36 h", "all leads"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title="Lead times", orient="top"),
        ),
    )
    chart = (
        rules()
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="XGBoost-free quantile score minus XGBoost's, % of XGBoost's"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=50).encode(x="gap:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH, height=170, title="Gap to XGBoost (dashed lines: equal, and the 2% tolerance)"
    )
    save(chart, name="figure_1_gap")

    tail_rows = [
        {
            "type": KINDS[kind],
            "level": level,
            "gap": bands["12-36 h"][f"pinball_{level}"]["gap_percent"],
        }
        for kind, bands in scores.items()
        for level in ("q01", "q05", "q95", "q99")
    ]
    tails = (
        alt.Chart(pl.DataFrame(tail_rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("level:N", title="Quantile level", sort=["q01", "q05", "q95", "q99"]),
            y=alt.Y("gap:Q", title="Pinball loss gap to XGBoost, %"),
            color=alt.value(ocf.DATA_BLUE),
            column=alt.Column("type:N", title=None, sort=list(KINDS.values())),
        )
        .properties(width=150, height=170, title="Tail pinball loss gap, 12 to 36 hour leads")
    )
    save(tails, name="figure_2_tails")


if __name__ == "__main__":
    main()
