# ruff: noqa: E501
"""Draw the figures of experiment 249's write-up.

uv run --no-sync python ml-experiments/00249-*/e00249_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00249_report import ARMS, N_RESAMPLES, load, pair_bootstrap

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00249-lsc-self-training-on-consistent-events"
)
WIDTH: Final[int] = 560
ROUND_ARMS: Final[tuple[str, ...]] = tuple(a for a in ARMS if a.startswith("R"))


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name, N_RESAMPLES)


def main() -> None:
    """Draw the two figures."""
    real, injected, _, used = load()
    long = real.filter(pl.col("hours") > 24)
    n = long.filter(pl.col("arm") == "R1").height
    rows = []
    for arm in ROUND_ARMS[1:]:
        d, lo, hi = pair_bootstrap(table=long, arm=arm, base="R1", weighted=False)
        rows.append({"arm": arm, "point": d * n, "lo": lo * n, "hi": hi * n})
    frame = pl.DataFrame(rows)
    base: Any = alt.Chart(frame)
    y = alt.Y("arm:N", sort=list(ROUND_ARMS[1:]), title=None)
    zero_base: Any = alt.Chart(pl.DataFrame({"x": [0.0]}))
    zero: Any = zero_base.mark_rule(strokeDash=[4, 3], color="#555555").encode(x="x:Q")
    chart = alt.layer(
        base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(
            y=y,
            x=alt.X(
                "lo:Q", title="Transfers found minus round 1's re-ranker (of 35; higher is better)"
            ),
            x2="hi:Q",
        ),
        base.mark_point(filled=True, size=70, opacity=1, color=ocf.DATA_BLUE).encode(
            y=y, x="point:Q"
        ),
        zero,
    ).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            [
                "Re-ranker after self-training minus the re-ranker before, 35 real transfers over 1 day, 80 unexplained odd-month alarms.",
                "Dot: estimate. Line: 95% interval from resampling whole Primary pairs. The prediction was +3 or more.",
            ],
            anchor="start",
            fontSize=12,
        ),
    )
    _save(chart, name="figure_1_found_minus_round_1")
    head = used.filter(pl.col("variant") == "2 w0.25").unique(["series", "stamp"])
    mix = (
        head.with_columns(
            size=pl.when(pl.col("step_mw24") < 1)
            .then(pl.lit("under 1 MW"))
            .otherwise(pl.lit("1 to 3 MW")),
            month=pl.when(pl.col("odd")).then(pl.lit("odd month")).otherwise(pl.lit("even month")),
        )
        .group_by("size", "month")
        .agg(n=pl.len())
    )
    mix_chart: Any = (
        alt.Chart(mix)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("size:N", sort=["under 1 MW", "1 to 3 MW"], title=None),
            x=alt.X("n:Q", title="Distinct pseudo-positives (56 in all)"),
            color=alt.Color(
                "month:N",
                scale=alt.Scale(
                    domain=["even month", "odd month"], range=[ocf.DATA_BLUE, ocf.DATA_SKY_LIGHT]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=120,
            title=alt.TitleParams(
                [
                    "The real events that pass the rule, by size of the 24-hour step in the forecast error.",
                    "Round 2's whole training addition is these 56 events.",
                ],
                anchor="start",
                fontSize=12,
            ),
        )
    )
    _save(mix_chart, name="figure_2_pseudo_positive_mix")
    del injected


if __name__ == "__main__":
    main()
