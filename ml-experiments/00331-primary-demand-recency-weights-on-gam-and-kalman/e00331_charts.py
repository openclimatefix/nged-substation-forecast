# ruff: noqa: E501
"""Draw the figures of experiment 331 from the stored scores.

PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-*:ml-experiments/00331-* \
    uv run --no-sync python ml-experiments/00331-*/e00331_charts.py
"""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00331_common as c
import e00331_report as r
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{c.EXPERIMENT_ID:05d}-{c.SLUG}"
)
WIDTH: Final[int] = 640
ORANGE, BLUE, GREY = ocf.BRAND_ORANGE, ocf.DATA_BLUE, "#7F7F7F"
NAMES: Final[dict[str, str]] = {
    "W24": "last 24 months", "H3": "half-life 3 months", "H6": "half-life 6 months", "H12": "half-life 12 months",
    "H24": "half-life 24 months", "bestS": "half-life 12 months plus seasonal bonus",
}  # fmt: skip


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _intervals(
    *,
    rows: list[dict],
    x_title: str,
    title: str,
    subtitle: str,
    colour: str | None = None,
    height: int = 220,
) -> alt.Chart:
    df = pl.DataFrame(rows)
    order = list(dict.fromkeys(df["label"].to_list()))
    enc_colour = (
        alt.Color(
            "series:N",
            scale=alt.Scale(
                domain=["with Kalman filter", "without Kalman filter"], range=[ORANGE, BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
        if colour is None
        else alt.value(colour)
    )
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title=x_title),
        x2="hi:Q",
        color=enc_colour,
        yOffset="series:N" if colour is None else alt.value(0),
    )
    dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="mean:Q",
        color=enc_colour,
        yOffset="series:N" if colour is None else alt.value(0),
        tooltip=[],
    )
    return (rule + bars + dots).properties(
        width=WIDTH, height=height, title=_title(title=title, subtitle=subtitle)
    )


def headline() -> None:
    """Quantile score of each recency arm minus equal weights on the full history."""
    frame = r.load_real()
    sites = sorted(frame["site"].unique().to_list())
    months = sorted(frame["month"].unique().to_list())
    cf = r.cross_fit(frame=frame, sites=sites, months=months)
    rows = []
    for arm in ("W24", "H3", "H6", "H12", "H24", "bestS"):
        d = r.pooled(cf["score"][arm]) - r.pooled(cf["score"]["R0"])
        mean, lo, hi, _ = r.month_bootstrap(diff=d)
        rows.append({"label": NAMES[arm], "mean": mean, "lo": lo, "hi": hi, "series": "all"})
    chart = _intervals(rows=rows, colour=ORANGE, x_title="Quantile score minus equal weights on the full history (points of capacity; more negative is better)",
        title="Giving recent rows more weight improved the next-month forecast, and a 24-month window did as well",
        subtitle="Dot: estimate over 15 Primaries and 58 monthly forecast origins (2021-09 to 2026-06). Line: 95% interval from resampling whole calendar months. Dashed rule: same as equal weights on the full history. Past weather, with Kalman filter.")  # fmt: skip
    _save(chart, name="headline_contrasts")


def known_answer() -> None:
    """Injected-step worlds: first 90 days, with and without the Kalman filter."""
    rows = []
    for series, directory, gain in (
        ("with Kalman filter", c.SCORE_DIR, 0.10),
        ("without Kalman filter", c.WORK_DIR / "scores_nofilter", 0.0),
    ):
        inj = pl.concat(
            [pl.read_parquet(f) for f in sorted(directory.glob("inj_*.parquet"))]
        ).filter((pl.col("gain") == gain) & pl.col("group").is_in(["d0_30", "d30_90"]))
        g = (
            inj.filter(pl.col("arm") == "R0")
            .group_by("world", "site", "month")
            .agg(pb0=pl.col("pinball_sum").sum(), n0=pl.col("n").sum())
        )
        for arm in ("W24", "H3", "H6", "H12", "H24"):
            f = (
                inj.filter(pl.col("arm") == arm)
                .group_by("world", "site", "month")
                .agg(pb=pl.col("pinball_sum").sum(), n=pl.col("n").sum())
            )
            j = (
                f.join(g, on=["world", "site", "month"])
                .group_by("world", "site")
                .agg(
                    d=pl.col("pb").sum() / pl.col("n").sum()
                    - pl.col("pb0").sum() / pl.col("n0").sum()
                )
            )
            onset = np.array([w.split("_")[2] for w in j["world"]])
            dv = j["d"].to_numpy()
            bm = np.array([dv[onset == b].mean() for b in np.unique(onset)])
            mean, lo, hi, _ = r.month_bootstrap(diff=bm)
            rows.append({"label": NAMES[arm], "mean": mean, "lo": lo, "hi": hi, "series": series})
    chart = _intervals(rows=rows, height=300, x_title="Quantile score minus equal weights on the full history (points of capacity; more negative is better)",
        title="After an injected step in demand, short half-lives helped only without the Kalman filter",
        subtitle="Known-answer test: a permanent step of 10% or 25% of a Primary's load was added to 14 Primaries at four onsets (200 worlds); scored over the first 90 days after the step. Line: 95% interval from resampling the four onsets. Dashed rule: same as equal weights.")  # fmt: skip
    _save(chart, name="known_answer_with_and_without_filter")


def d8() -> None:
    """D8's quantile score, over all months and after the 2023-08-15 break."""
    frame = r.load_real()
    sites = sorted(frame["site"].unique().to_list())
    months = sorted(frame["month"].unique().to_list())
    cf = r.cross_fit(frame=frame, sites=sites, months=months)
    i8 = sites.index("D8")
    after = np.array([r.D8_AFTER[0] <= m <= r.D8_AFTER[1] for m in months])
    rows = []
    for arm in ("R0", "W24", "H3", "H6", "H12", "H24"):
        sc = cf["score"][arm][:, i8]
        rows.append(
            {
                "arm": arm if arm in ("R0", "W24") else f"half-life {arm[1:]} months",
                "period": "all 58 months",
                "score": float(np.nanmean(sc)),
            }
        )
        rows.append(
            {
                "arm": arm if arm in ("R0", "W24") else f"half-life {arm[1:]} months",
                "period": "3 months after the break",
                "score": float(np.nanmean(sc[after])),
            }
        )
    for row in rows:
        row["arm"] = {"R0": "equal weights, full history", "W24": "last 24 months"}.get(
            row["arm"], row["arm"]
        )
    order = list(dict.fromkeys(row["arm"] for row in rows))
    chart = (
        alt.Chart(pl.DataFrame(rows)).mark_bar().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None),  # ty: ignore[invalid-argument-type] x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)"),
            yOffset="period:N", color=alt.Color("period:N", scale=alt.Scale(domain=["all 58 months", "3 months after the break"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
        ).properties(width=WIDTH, height=260, title=_title(title="At D8, short half-lives scored best, and every arm was far off after the break", subtitle="D8 on its own, whose demand shifted on 2023-08-15. Bars: mean quantile score of the arm chosen by cross-fitting. Past weather, with Kalman filter."))
    )  # fmt: skip
    _save(chart, name="d8_by_arm")


if __name__ == "__main__":
    headline()
    known_answer()
    d8()
