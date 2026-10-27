# ruff: noqa: E501
"""Draw experiment 00356's figures as SVG from the stored arms.

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`. Farms are
W1 to W3 and no chart shows a calendar date.

    PYTHONPATH=$PYTHONPATH_FOLDERS uv run --no-sync python e00356_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00356_arms as arms
import e00356_common as common
import e00356_score as sc
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

OUT: Final[Path] = Path("docs/ml-experiments/00356-wind-default-2-blend-and-monthly-refit-retest")
WIDTH: Final[int] = 430
WRAP: Final[int] = 100
SCOPE: Final[str] = (
    "Three metered wind farms W1 to W3; kept half-hours, true days 1 to 10 (lead 24 to 240 hours)."
)
LABELS: Final[dict[str, str]] = {
    "d1": "default 1 (XGBoost + offset table)",
    "d2": "default 2 (plant + mixture head)",
    "u1": "u1 plant band + default 1 (experiment 348's z5)",
    "u2": "u2 default 2 blended with default 1",
    "u2p": "u2 placebo (weights from another month)",
    "u3": "u3 head retrained monthly",
    "u4": "u4 default 2 + monthly offsets",
}
COLOURS: Final[dict[str, str]] = {
    "d1": ocf.DATA_BLUE,
    "d2": ocf.DATA_PURPLE,
    "u1": ocf.DATA_GREEN,
    "u2": ocf.BRAND_ORANGE,
    "u2p": ocf.DATA_SKY,
    "u3": "#555555",
    "u4": ocf.DATA_BLUE,
}


def title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a wrapped title."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def colour(*, names: list[str]) -> alt.Color:
    """Return the arm colour encoding."""
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[LABELS[n] for n in names], range=[COLOURS[n] for n in names]),
        legend=None,
    )


def figure_1(*, metrics: dict, rows: pl.DataFrame) -> None:
    """Each arm minus default 2: scored year with two intervals, and the out-of-time runs."""
    kept = rows["kept"].to_numpy()
    hours = rows["lead_hours"].to_numpy()
    scored = rows["scored"].to_numpy()
    scope = kept & (hours >= 24) & (hours <= 240)
    by_month = sc.Blocks(labels=rows["month"].to_numpy())
    by_run = sc.Blocks(labels=rows["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy())
    names = ["u2", "u1", "u4", "u3", "u2p", "d1"]
    out = []
    for period, blocks, keep, levels in (
        ("Scored year (53 runs)", by_month, scope & scored, (0.95, sc.LEVEL)),
        ("Later runs (about 16 runs)", by_run, scope & ~scored, (0.95, 0.95)),
    ):
        for n in names:
            diff = metrics[n]["qs"] - metrics["d2"]["qs"]
            p, lo95, hi95 = blocks.interval(values=diff, keep=keep, level=levels[0])
            _, lo, hi = blocks.interval(values=diff, keep=keep, level=levels[1])
            out.append(
                {
                    "period": period,
                    "arm": LABELS[n],
                    "mean": p,
                    "lo95": lo95,
                    "hi95": hi95,
                    "lo": lo,
                    "hi": hi,
                }
            )
    frame = pl.DataFrame(out)
    order = [LABELS[n] for n in names]
    panels = []
    for period in ("Scored year (53 runs)", "Later runs (about 16 runs)"):
        sub = frame.filter(pl.col("period") == period)
        y = alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
        wide = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeWidth=1.5, opacity=0.5)
            .encode(
                y=y,
                x=alt.X(
                    "lo:Q",
                    title="Quantile score minus default 2's (points of capacity; more negative is better)",
                ),
                x2="hi:Q",
                color=colour(names=names),
            )
        )
        narrow = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeWidth=4)
            .encode(y=y, x="lo95:Q", x2="hi95:Q", color=colour(names=names))
        )
        dots = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=70)
            .encode(y=y, x="mean:Q", color=colour(names=names))
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(wide, narrow, dots, zero).properties(
                width=WIDTH, height=170, title=alt.TitleParams(period, anchor="start", fontSize=12)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=title(
            text="No arm beats Wind farms research default 2 by more than its interval",
            lines=[
                SCOPE,
                "Dot: estimate. Thick line: 95% interval. Thin line: 99.75% interval for the scored year (the second period shows 95% only). Intervals resample whole calendar months (scored year) or whole runs (later runs). Dashed line: same as default 2.",
            ],
        )
    )
    chart.save(OUT / "figure_1_headline.svg")


def figure_2(*, metrics: dict, rows: pl.DataFrame) -> None:
    """Share above q99 and 0.99 pinball loss for each arm."""
    kept = rows["kept"].to_numpy()
    hours = rows["lead_hours"].to_numpy()
    keep = kept & (hours >= 24) & (hours <= 240) & rows["scored"].to_numpy()
    names = ["d1", "d2", "u1", "u2", "u3", "u4"]
    frame = pl.DataFrame(
        {
            "arm": [LABELS[n] for n in names],
            "share": [100 * metrics[n]["above_q99"][keep].mean() for n in names],
            "loss": [metrics[n]["p99"][keep].mean() for n in names],
        }
    )
    order = [LABELS[n] for n in names]
    panels = []
    for column, label in (
        ("share", "Observations above q99 (%; target 1)"),
        ("loss", "0.99 pinball loss (points of capacity; smaller is better)"),
    ):
        bars = (
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
                x=alt.X(f"{column}:Q", title=label),
                color=colour(names=names),
            )
        )
        panels.append(bars.properties(width=WIDTH, height=150))
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    panels[0] = alt.layer(panels[0], rule).properties(width=WIDTH, height=150)
    chart = alt.vconcat(*panels).properties(
        title=title(
            text="No arm brings the share above q99 near 1%",
            lines=[
                SCOPE
                + " Scored year (53 Tuesday runs). Dashed line: the share a calibrated q99 would give."
            ],
        )
    )
    chart.save(OUT / "figure_2_upper_tail.svg")


def figure_3(*, arm: str, rows: pl.DataFrame, q: np.ndarray, summary: np.ndarray) -> None:
    """Forecast q50, q95, and q99 and the observed quantiles, by bins of the plant's mean power."""
    keep = (
        (rows["kept"].to_numpy())
        & (rows["lead_hours"].to_numpy() >= 24)
        & (rows["lead_hours"].to_numpy() <= 240)
        & rows["scored"].to_numpy()
    )
    actual = rows["actual_norm"].to_numpy() * 100
    site = rows["site"].to_numpy()
    out = []
    for farm in common.FARMS:
        sel = keep & (site == farm)
        x = summary[sel, 1] * 100
        edges = np.quantile(x, np.linspace(0, 1, 11))
        ids = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, 9)
        for b in range(10):
            m = ids == b
            centre = float(x[m].mean())
            for level, idx in (("q50", 5), ("q95", 9), ("q99", 10)):
                tau = {"q50": 0.5, "q95": 0.95, "q99": 0.99}[level]
                out.append(
                    {
                        "farm": farm,
                        "x": centre,
                        "level": level,
                        "kind": "forecast",
                        "y": float(q[sel][m][:, idx].mean() * 100),
                    }
                )
                out.append(
                    {
                        "farm": farm,
                        "x": centre,
                        "level": level,
                        "kind": "observed",
                        "y": float(np.quantile(actual[sel][m], tau)),
                    }
                )
    frame = pl.DataFrame(out)
    colours = {"q50": ocf.DATA_BLUE, "q95": ocf.BRAND_ORANGE, "q99": ocf.DATA_PURPLE}
    panels = []
    for farm in common.FARMS:
        sub = frame.filter(pl.col("farm") == farm)
        base = alt.Chart(sub).encode(
            x=alt.X("x:Q", title="Plant's mean power over the 51 members (% of capacity)"),
            y=alt.Y("y:Q", title="Power (% of capacity)"),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        line = base.transform_filter(alt.datum.kind == "forecast").mark_line(strokeWidth=2)
        dots = base.transform_filter(alt.datum.kind == "observed").mark_point(filled=True, size=45)
        panels.append(
            alt.layer(line, dots).properties(  # ty: ignore[invalid-argument-type]
                width=WIDTH, height=130, title=alt.TitleParams(farm, anchor="start", fontSize=12)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=title(
            text=f"Forecast and observed q50, q95, and q99 by plant power: {LABELS[arm]}",
            lines=[
                SCOPE
                + " Scored year. Line: mean forecast quantile in 10 equal-count bins of the plant's mean power. Dot: observed quantile of the same level in the bin; a dot above its line means the forecast quantile is too low."
            ],
        )
    )
    chart.save(OUT / f"figure_3_power_curve_{arm}.svg")


def main() -> None:
    """Draw every figure."""
    rows = pl.read_parquet(arms.ARMS_DIR / "rows.parquet")
    actual = rows["actual_norm"].to_numpy().astype(np.float64)
    metrics = {
        n: sc.row_scores(q_all=np.load(arms.ARMS_DIR / f"{n}.npy"), actual=actual)
        for n in sc.ARM_NAMES
    }
    frame = pl.read_parquet(common.DATA_DIR / "frame.parquet")
    t_idx = np.flatnonzero(
        (
            frame["scored"]
            | (frame["tuesday"] & (frame["power_fcst_init_time"] >= common.OUT_OF_TIME_FIRST_RUN))
        ).to_numpy()
    )
    summary = np.load(common.DATA_DIR / "summary.npy")[t_idx]
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1(metrics=metrics, rows=rows)
    figure_2(metrics=metrics, rows=rows)
    for arm in ("d2", "u2"):
        q = np.load(arms.ARMS_DIR / f"{arm}.npy")[0]
        figure_3(arm=arm, rows=rows, q=q, summary=summary)


if __name__ == "__main__":
    main()
