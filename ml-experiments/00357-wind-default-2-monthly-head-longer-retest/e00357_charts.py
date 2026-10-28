# ruff: noqa: E501
"""Draw experiment 00357's figures as SVG from `score.json`.

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`. Farms are
W1 to W3 and no chart shows a calendar date.

    PYTHONPATH=$PYTHONPATH_FOLDERS uv run --no-sync python e00357_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00357_common as common
import plotting.ocf_theme as ocf
import polars as pl

OUT: Final[Path] = Path("docs/ml-experiments/00357-wind-default-2-monthly-head-longer-retest")
WIDTH: Final[int] = 430
SCOPES: Final[dict[str, str]] = {
    "tue_year": "53 Tuesday runs, scored year",
    "all_year": "every run, scored year (365)",
    "rep_all": "every run, replicate (74)",
    "rep_tue": "Tuesday runs, replicate (10)",
    "all_days": "every run (439)",
}
ARMS: Final[dict[str, str]] = {
    "u3": "u3 head retrained monthly",
    "u3p": "placebo: head from an earlier month",
}


def title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title."""
    return alt.TitleParams(
        text, subtitle=subtitle, anchor="start", fontSize=14, subtitleFontSize=11
    )


def main() -> None:
    """Write the three figures."""
    result = json.loads((common.DATA_DIR / "score.json").read_text())
    rows = [
        {"scope": SCOPES[s], "arm": ARMS[a], "mean": v[0], "lo": v[1], "hi": v[2]}
        for a in ARMS
        for s in SCOPES
        if (v := result["contrasts"][a].get(f"{s}/headline"))
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("scope:N", sort=list(SCOPES.values()), title=None),
        yOffset="arm:N",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARMS.values()), range=[ocf.DATA_PURPLE, ocf.DATA_SKY]),
            title=None,
        ),
    )
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo:Q", title="Quantile score minus default 2 (points of capacity; negative is better)"
        ),
        x2="hi:Q",
    )
    dots = base.mark_point(filled=True, size=60).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    fig1 = alt.layer(bars, dots, zero).properties(
        width=WIDTH,
        title=title(
            text="Retraining the head monthly gains 0.04 to 0.10 points; a stale head gains less",
            subtitle="Wind farms W1 to W3, kept half-hours, true days 1 to 10. Intervals: 99% by month, 95% by run for the replicate.",
        ),
    )
    fig1.save(OUT / "figure_1_headline.svg")

    monthly = pl.DataFrame(
        {
            "month": list(range(1, len(result["monthly"]) + 1)),
            "gain": [m[1] for m in result["monthly"]],
        }
    )
    fig2 = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_PURPLE)
        .encode(
            x=alt.X(
                "month:O",
                title="Month of the run (1 is the first scored month, 15 the last)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("gain:Q", title="u3 minus default 2 (points)"),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=title(
                text="u3 beat default 2 in 11 of 15 months",
                subtitle="Headline scope, all runs; the last month has 12 runs.",
            ),
        )
    )
    fig2.save(OUT / "figure_2_monthly.svg")

    order = ["d1", "d2", "u3", "u3p"]
    tails = [
        {"arm": a, "scope": s, "share": result["tails"][f"{k}/{a}"][3]}
        for k, s in (("tue_year", "53 Tuesday runs"), ("all_days", "every run"))
        for a in order
    ]
    fig3 = (
        alt.Chart(pl.DataFrame(tails))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("arm:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Observations above q99 (%)"),
            color=alt.Color(
                "arm:N",
                legend=None,
                scale=alt.Scale(
                    domain=order, range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, "#555555", ocf.DATA_SKY]
                ),
            ),
            column=alt.Column("scope:N", title=None),
        )
        .properties(
            width=180,
            height=200,
            title=title(
                text="No arm brings the upper tail near the intended 1%",
                subtitle="d1 default 1, d2 default 2, u3 monthly head, u3p stale-head placebo. True days 1 to 10.",
            ),
        )
    )
    fig3.save(OUT / "figure_3_upper_tail.svg")


if __name__ == "__main__":
    main()
