# ruff: noqa: E501
"""Draw the figures of experiment 355 from `work/scores.json`.

PYTHONPATH=ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        uv run --no-sync python ml-experiments/00355-*/e00355_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00354_charts as c354  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00355-d8-weekly-refit-and-median-band-swap"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
c354.DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
c354.PNG_DIR = WORK / "png"


def _rows(*, scores: dict, spec: list[tuple[str, str, str]]) -> list[dict]:
    return [
        {"label": label, "estimate": c[0], "low": c[1], "high": c[2], "group": group}
        for label, column, group in spec
        for c in [scores["contrast"][column]["all"]["ci99"]]
    ]


def main() -> None:
    """Write the three figures."""
    scores = json.loads((WORK / "scores.json").read_text())
    c354.forest(
        rows=_rows(
            scores=scores,
            spec=[
                ("Monthly refit (M), own-residual band", "g_M", "Gap to BG2"),
                ("Weekly refit (W)", "g_W", "Gap to BG2"),
                ("M median with BG2's spread (S1)", "g_S1", "Gap to BG2"),
                ("BG2 median with M's own-residual band (S2)", "g_S2", "Gap to BG2"),
            ],
        ),
        x_title="Quantile score minus BG2's (points of capacity; smaller is better)",
        title="Figure 1: neither the median nor the band alone explains D8's gap to BG2",
        subtitle="Dot: estimate over the 96,986 scored rows. Line: 99% interval from resampling whole months. Zero is BG2.",
        name="figure_1_gap_to_bg2",
        height=200,
    )
    c354.forest(
        rows=_rows(
            scores=scores,
            spec=[
                ("Weekly minus monthly (W - M)", "c_W__M", "Refit cadence"),
                ("Quarterly minus monthly (Q - M)", "c_Q__M", "Refit cadence"),
                ("M median with BG2's spread minus M (S1 - M)", "c_S1__M", "Swap"),
                ("BG2 median with M's band minus M (S2 - M)", "c_S2__M", "Swap"),
            ],
        ),
        x_title="Change in quantile score (points of capacity; negative is better)",
        title="Figure 2: weekly refits change D8's score by under 0.12 points",
        subtitle="Dot: estimate over the 96,986 scored rows. Line: 99% interval from resampling whole months.",
        name="figure_2_cadence_and_swaps",
        height=200,
    )
    rows = [
        {"month": m, "arm": label, "score": values[key]}
        for m, values in scores["monthly"].items()
        for label, key in (("Monthly refit (M)", "M"), ("Weekly refit (W)", "W"), ("BG2", "bg2"))
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Month of valid time", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "score:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["Monthly refit (M)", "Weekly refit (W)", "BG2"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=c354.WIDTH,
            height=240,
            title=c354._title(
                "Figure 3: weekly refits help only in the first months",
                "Mean quantile score of each valid-time month at D8.",
            ),
        )
    )
    c354._save(chart, name="figure_3_monthly")


if __name__ == "__main__":
    main()
