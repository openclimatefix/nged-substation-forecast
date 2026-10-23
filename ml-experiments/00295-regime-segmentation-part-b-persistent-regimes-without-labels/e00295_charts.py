# ruff: noqa: E501, PERF401, PLW2901
"""Draw the figures of experiment 295 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python e00295_charts.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Final

import altair as alt
import e00295_common as c
import plotting.ocf_theme as ocf
import polars as pl
from e00295_report import engine_spells
from e00295_tune import load, site_years
from studies.charts import CONTENT_WIDTH_PX

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00295-regime-segmentation-part-b-persistent-regimes-without-labels"
)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 120


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def fig_d8() -> None:
    """D8's daily features with the boundaries of two penalties and the known break."""
    states = pickle.loads((c.EXPERIMENT_DIR / "clean_states.pkl").read_bytes())
    series = states["D8"]["series"]
    days = series.days.astype("datetime64[D]").astype(str)
    engine_days, share, _ = engine_spells()
    frame = pl.DataFrame(
        {
            "day": days,
            "Share of half-hours in a dip": series.values[:, 0],
            "5% quantile of the residual (points of capacity)": series.values[:, 1],
        }
    )
    engine = pl.DataFrame(
        {"day": engine_days.astype(str), "Share of half-hours the engine detector flags": share}
    )
    boundaries = []
    for beta, name in ((16.0, "penalty 16"), (64.0, "penalty 64 (chosen)")):
        for b in states["D8"]["bounds"][(beta, 7)]:
            boundaries.append({"day": str(b.date), "Boundaries of": name})
    marks = pl.DataFrame(boundaries)
    known = pl.DataFrame({"day": ["2023-08-15"], "Boundaries of": ["known break, 2023-08-15"]})
    colour = alt.Color(
        "Boundaries of:N",
        scale=alt.Scale(
            domain=["known break, 2023-08-15", "penalty 16", "penalty 64 (chosen)"],
            range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    panels = []
    for column, source in (
        ("Share of half-hours in a dip", frame),
        ("5% quantile of the residual (points of capacity)", frame),
        ("Share of half-hours the engine detector flags", engine),
    ):
        base = (
            alt.Chart(source.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color="#555555", strokeWidth=0.8)
            .encode(
                x=alt.X("day:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
                y=alt.Y(f"{column}:Q", title=column),
            )
        )
        rules = (
            alt.Chart(marks.to_pandas()).mark_rule(strokeWidth=1.2).encode(x="day:T", color=colour)  # ty: ignore[unresolved-attribute]
        )
        known_rule = (
            alt.Chart(known.to_pandas()).mark_rule(strokeWidth=2.5).encode(x="day:T", color=colour)  # ty: ignore[unresolved-attribute]
        )
        panels.append((base + rules + known_rule).properties(width=WIDTH, height=110))
    save(
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                "D8's daily residual distribution, with the boundaries the segmentation finds",
                subtitle=[
                    "Each line is one feature of D8's residual from a model fitted to the whole history; the rules are",
                    "segment boundaries (7-day minimum segment). Smaller penalties find more boundaries.",
                ],
            )
        ),
        "figure_2",
    )


def fig_recall() -> None:
    """Recall by step size and the regime changes per Primary-year, against the penalty."""
    chosen = json.loads((c.EXPERIMENT_DIR / "chosen_config.json").read_text())
    hold = chosen["hold_sites"]
    inj = load("injection").filter(
        pl.col("site").is_in(hold)
        & (pl.col("parity") == "even")
        & (pl.col("absorption") == "fit")
        & (pl.col("min_length") == 7)
    )
    rows = []
    for beta, group in inj.group_by("beta"):
        beta = beta[0]
        for name, sub, expected in (
            (
                "persistent step, 25% of load",
                group.filter((pl.col("family") == "persistent") & (pl.col("size") == 0.25)),
                "persistent",
            ),
            (
                "persistent step, 10% of load",
                group.filter((pl.col("family") == "persistent") & (pl.col("size") == 0.10)),
                "persistent",
            ),
            (
                "temporary shift, 14 to 60 days",
                group.filter(pl.col("family") == "temporary"),
                "temporary",
            ),
        ):
            rows.append(
                {
                    "beta": beta,
                    "series": name,
                    "recall": float((sub["label"] == expected).to_numpy().mean()),
                }
            )
    recall = pl.DataFrame(rows).to_pandas()
    clean = load("clean_break").filter(
        pl.col("site").is_in(hold) & (pl.col("min_length") == 7) & (pl.col("label") != "temporary")
    )
    years = sum(site_years()[s] for s in hold)
    rate = clean.group_by("beta").agg(n=pl.len()).with_columns(rate=pl.col("n") / years).to_pandas()
    domain = [
        "persistent step, 25% of load",
        "persistent step, 10% of load",
        "temporary shift, 14 to 60 days",
    ]
    top = (
        alt.Chart(recall)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "beta:O",
                title="Penalty per boundary (larger finds fewer boundaries)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("recall:Q", title="Share labelled correctly", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=400, columns=1),
            ),
        )
        .properties(width=WIDTH, height=170)
    )
    target = (
        alt.Chart(pl.DataFrame({"y": [0.8]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(y="y:Q")
    )
    bottom = (
        alt.Chart(rate)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("beta:O", title="Penalty per boundary", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title="Regime changes per Primary-year"),
        )
        .properties(width=WIDTH, height=130)
    )
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(y="y:Q")
    )
    save(
        alt.vconcat(top + target, bottom + one).properties(
            title=alt.TitleParams(
                "Small steps are missed, and a small penalty finds many false changes",
                subtitle=[
                    "Held-out Primaries, injections starting in even months, 7-day minimum segment. Dashed lines: 80% target (top)",
                    "and 1 regime change per Primary-year (bottom). Higher is better on top; lower is better below.",
                ],
            )
        ),
        "figure_1",
    )


def fig_gain() -> None:
    """Forecast error with no offset, the trailing offset, and the latest-regime offset."""
    files = sorted((c.EXPERIMENT_DIR / "gain").glob("*.parquet"))
    gain = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    injected = gain.filter(pl.col("scenario") != "real_d8")
    rows = []
    for k, group in injected.group_by("month_after"):
        for arm, column in (
            ("no offset", "mae_none"),
            ("trailing 60-day offset", "mae_trailing"),
            ("latest-boundary offset", "mae_segment"),
            ("latest-regime offset", "mae_regime"),
        ):
            rows.append({"month": k[0], "arm": arm, "mae": float(group[column].to_numpy().mean())})
    frame = pl.DataFrame(rows).to_pandas()
    save(
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:O",
                title="Months after the injected step (forecast month)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("mae:Q", title="Mean absolute error (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[
                        "no offset",
                        "trailing 60-day offset",
                        "latest-boundary offset",
                        "latest-regime offset",
                    ],
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=190,
            title=alt.TitleParams(
                "Offsets help after a step, and a plain 60-day offset helps as much as any",
                subtitle=[
                    "Injected persistent steps (even-month onsets) at 14 Primaries, 24-month window, monthly refit. Smaller is better."
                ],
            ),
        ),
        "figure_3",
    )


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["d8", "recall", "gain"]
    if "d8" in which:
        fig_d8()
    if "recall" in which:
        fig_recall()
    if "gain" in which:
        fig_gain()
