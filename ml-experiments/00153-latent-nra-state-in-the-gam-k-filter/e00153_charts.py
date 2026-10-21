# type: ignore
# ruff: noqa: PGH003, E501
"""Charts of experiment 153: each arm's quantile score minus K0's, with 95% intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import e00153_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00153-latent-nra-state-in-the-gam-k-filter"
)
RATIO: Final[int] = 10000
ARMS: Final[dict[str, tuple[str, str]]] = {
    "GAM-ONLY (ignore power)": ("GAM_ONLY", ocf.DATA_SKY),
    "MASK (drop readings in alarm)": ("MASK", ocf.DATA_GREEN),
    "S-KF (experiment 141)": ("SKF", ocf.BRAND_ORANGE),
    "NEW-DET (latent NRA, detected)": (f"NEW_DET_{RATIO}", ocf.DATA_BLUE),
    "NEW-ORACLE (true intervals)": (f"NEW_ORACLE_{RATIO}", ocf.DATA_PURPLE),
}


def main() -> None:
    """Write the headline chart."""
    df = report.load(report.SCORES)
    scored = df.filter(pl.col("period") != "tuning")
    post = scored.filter(pl.col("origin") == "post")
    sets = {
        "Targets inside the transfer": post.filter(pl.col("inside")),
        "Targets in the 48 hours after the true end": post.filter(pl.col("after48")),
        "Forecast origin before the alarm is known": scored.filter(
            (pl.col("origin") == "lag") & (pl.col("inside") | pl.col("after48"))
        ),
    }
    rows = []
    for title, frame in sets.items():
        for label, (arm, _) in ARMS.items():
            d = report.month_mean_difference(frame=frame, a=arm, b="K0")
            rows.append({"rows": title, "arm": label, **d})
    data = pl.DataFrame(rows)
    order = list(ARMS)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=order, range=[c for _, c in ARMS.values()]),
            legend=None,
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X(
                    "lo:Q",
                    title="Quantile score minus K0's (points of capacity; negative is better)",
                ),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="mean:Q")
            + zero
        )
        .properties(width=520, height=110)
        .facet(row=alt.Row("rows:N", title=None, sort=list(sets)))
        .properties(
            title={
                "text": ["Score against the true normal running arrangement, sites D11 and D16"],
                "subtitle": [
                    "Dot: mean difference from K0. Line: 95% interval from resampling whole months.",
                    "Vertical rule: same as K0. Injected transfers, 8 simulated years.",
                ],
            }
        )
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS / "headline.svg")
    data.write_csv(
        DOCS.parent.parent.parent / "data/ml-experiments/_e153_chart.csv"
    ) if False else None
    print(data)


if __name__ == "__main__":
    main()
