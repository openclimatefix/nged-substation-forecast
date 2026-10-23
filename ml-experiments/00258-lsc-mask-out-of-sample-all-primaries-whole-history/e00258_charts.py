# ruff: noqa: E501
"""Draw the figures of experiment 258 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00258-*/e00258_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00258_data as dat
import e00258_harness as h
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, wrapped

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00258-lsc-mask-out-of-sample-all-primaries-whole-history"
)
W: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "LR_b20": "LR_b20 mask (learned, 20 candidates)",
    "LR_b80": "LR_b80 mask (learned, 80 candidates)",
    "RO_b80": "RO_b80 mask (RO-NC, 80 candidates)",
    "HALF": "HALF (average of K0 and LR_b80 mask)",
    "GAM_ONLY": "GAM_ONLY (no filter at all)",
    "ORACLE": "ORACLE (logged load shifts)",
}
ARM_ORDER: Final[list[str]] = list(ARMS.values())
YEAR_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def report() -> dict:
    """The stored report tables."""
    return json.loads((h.HARNESS_DIR / "report.json").read_text())


def fig_per_shift() -> None:
    """Gain over K0 of ORACLE and LR_b20 after each load shift, ordered by ORACLE's gain."""
    t = pl.read_parquet(h.HARNESS_DIR / "after_per_shift.parquet").sort(
        "gain_ORACLE", descending=True
    )
    t = t.with_columns(rank=pl.int_range(1, t.height + 1))
    long = pl.concat(
        [
            t.select(
                "rank", "hours", arm=pl.lit("ORACLE"), gain=pl.col("gain_ORACLE") / pl.col("n")
            ),
            t.select(
                "rank", "hours", arm=pl.lit("LR_b20 mask"), gain=pl.col("gain_LR_b20") / pl.col("n")
            ),
        ]
    )
    colours = ["#9AA0A6", ocf.DATA_BLUE]
    domain = ["ORACLE", "LR_b20 mask"]
    chart = (
        alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, opacity=0.9)
        .encode(
            x=alt.X(
                "rank:O",
                title="Load shift, ordered by the logged-oracle mask's gain (largest first)",
                axis=alt.Axis(labelAngle=0, values=[1, 5, 10, 15, 20, 25, 30, 34]),
            ),
            y=alt.Y("gain:Q", title="Gain over K0 (points of capacity; higher is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["rank", "arm", "gain", "hours"],
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(color="#5F6368").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    save(
        (zero + chart).properties(
            width=W + 140,
            height=280,
            title=alt.TitleParams(
                wrapped(
                    text="The LR_b20 mask recovers at least half of the oracle's gain after 6 of 34 load shifts and nothing after 22",
                    width=62,
                ),
                subtitle=wrapped(
                    text="One dot per labelled real load shift lasting over 1 day (34 of the 35 have readings after them). Gain is the K0 score minus the arm's score, averaged over the rows in the 72 hours after the load shift ends, in points (1 point is 1% of the substation's effective capacity). ORACLE is the mask over the logged load shifts."
                ),
            ),
        ),
        "per_shift_gain",
    )


def fig_harm_by_primary() -> None:
    """Ordinary-day score difference of LR_b20 by Primary, odd and even months."""
    rep = report()
    rows = []
    for s in h.SERIES:
        for parity, tag in (
            ("odd", "odd months (out of sample)"),
            ("even", "even months (in sample)"),
        ):
            r = rep["harm_by_primary"][f"{s}|{parity}"]["LR_b20"]
            rows.append(
                {
                    "primary": dat.label_of(series=s),
                    "months": tag,
                    "harm": r["harm"],
                    "lo": r["interval"][0],
                    "hi": r["interval"][1],
                }
            )
    t = pl.DataFrame(rows)
    order = [dat.label_of(series=s) for s in h.SERIES]
    colours = [ocf.DATA_BLUE, "#9AA0A6"]
    domain = ["odd months (out of sample)", "even months (in sample)"]
    base = alt.Chart(t.to_pandas())  # ty: ignore[unresolved-attribute]
    y = alt.Y("primary:N", sort=order, title=None)
    color = alt.Color(
        "months:N",
        scale=alt.Scale(domain=domain, range=colours),
        legend=alt.Legend(title=None, orient="top"),
    )
    off = alt.YOffset("months:N", scale=alt.Scale(domain=domain))
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "lo:Q",
            title="LR_b20 score minus K0 score on ordinary rows (points; negative is better)",
            axis=alt.Axis(values=[-0.4, -0.2, 0, 0.2, 0.4]),
        ),
        x2="hi:Q",
        color=color,
        yOffset=off,
    )
    dot = base.mark_point(filled=True, size=60).encode(y=y, x="harm:Q", color=color, yOffset=off)  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#5F6368").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    save(
        (zero + rule + dot).properties(
            width=W,
            height=420,
            title=alt.TitleParams(
                wrapped(
                    text="Out of sample, the LR_b20 mask changes ordinary-day scores by under 0.05 points at 13 of 15 Primaries and improves them at the other two",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Dot: estimate. Line: 95% interval from resampling whole calendar months. Rows are outside every logged closure that names the Primary and the 72 hours after it. D12 and D2 have no LR_b20 alarms at all."
                ),
            ),
        ),
        "harm_by_primary",
    )


def fig_arms() -> None:
    """Share of the oracle's gain and ordinary-day score difference for each arm."""
    rep = report()
    share = []
    harm = []
    for base, name in ARMS.items():
        if base != "ORACLE":
            r = rep["after_long"][base]
            share.append(
                {
                    "arm": name,
                    "v": 100 * r["share"],
                    "lo": 100 * r["share_interval"][0],
                    "hi": 100 * r["share_interval"][1],
                }
            )
        r = rep["harm"]["ordinary|odd"][base]
        harm.append({"arm": name, "v": r["harm"], "lo": r["interval"][0], "hi": r["interval"][1]})
    y = alt.Y("arm:N", sort=ARM_ORDER, title=None, axis=alt.Axis(labelLimit=300))

    def panel(
        rows: list[dict],
        title: str,
        domain: list[float] | None,
        ref: float,
        ref_label: str,
        ticks: list[float],
    ) -> alt.Chart:
        base = alt.Chart(pl.DataFrame(rows).to_pandas())  # ty: ignore[unresolved-attribute]
        scale = alt.Scale(domain=domain) if domain else alt.Scale()
        xe = alt.X("lo:Q", title=title, scale=scale, axis=alt.Axis(values=ticks))
        rule = base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(y=y, x=xe, x2="hi:Q")  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=80, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y=y, x=alt.X("v:Q", scale=scale, axis=alt.Axis(values=ticks))
        )
        line = (
            alt.Chart(pl.DataFrame({"x": [ref]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#5F6368", strokeDash=[4, 3])
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        return (line + rule + dot).properties(width=W, height=170, title=ref_label)

    top = panel(
        share,
        "Share of the logged-oracle mask's gain recovered, 34 load shifts (%; higher is better)",
        [-20, 110],
        50,
        "After a load shift: dashed line is the 50% bar",
        [0, 25, 50, 75, 100],
    )
    bottom = panel(
        harm,
        "Score minus K0's on ordinary rows, odd months (points; negative is better)",
        [-0.1, 1.0],
        0.0,
        "Ordinary days: 0 is K0's score",
        [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    save(
        alt.vconcat(top, bottom).properties(
            title=alt.TitleParams(
                wrapped(
                    text="No filter at all recovers the most after a load shift and loses the most on ordinary days",
                    width=58,
                ),
                subtitle=wrapped(
                    text="Dot: estimate. Line: 95% interval from resampling whole Primary pairs (top) or whole calendar months (bottom). ORACLE is the denominator of the top panel, so it has no bar there."
                ),
            )
        ),
        "arms",
    )


def fig_net_annual() -> None:
    """LR_b20 and GAM_ONLY minus K0 over every row of each series-year."""
    t = pl.read_parquet(h.HARNESS_DIR / "net_annual.parquet").with_columns(
        primary=pl.col("series").replace_strict({s: dat.label_of(series=s) for s in h.SERIES}),
        year=pl.col("year").cast(pl.String),
    )
    order = [dat.label_of(series=s) for s in h.SERIES]
    years = sorted(t["year"].unique())

    def panel(column: str, title: str, ticks: list[float]) -> alt.Chart:
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#5F6368")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        dots = (
            alt.Chart(t.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60, opacity=0.9)
            .encode(
                y=alt.Y("primary:N", sort=order, title=None),
                x=alt.X(f"{column}:Q", title=title, axis=alt.Axis(values=ticks)),
                color=alt.Color(
                    "year:N",
                    scale=alt.Scale(domain=years, range=YEAR_COLOURS),
                    legend=alt.Legend(title="Year", orient="top"),
                ),
                tooltip=["primary", "year", column],
            )
        )
        return (zero + dots).properties(width=W, height=300)

    top = panel(
        "d_LR_b20",
        "LR_b20 mask score minus K0's, every row of the year (points; negative is better)",
        [-0.5, 0, 0.5, 1.0, 1.5],
    )
    bottom = panel(
        "d_GAM_ONLY",
        "GAM_ONLY score minus K0's, every row of the year (points; negative is better)",
        [-1, 0, 1, 2, 3],
    )
    save(
        alt.vconcat(top, bottom).properties(
            title=alt.TitleParams(
                wrapped(
                    text="Over a whole year the LR_b20 mask loses where it covers long, unlabelled closures",
                    width=58,
                ),
                subtitle=wrapped(
                    text="One dot per Primary and calendar year; 2022 starts in February and 2026 ends in June. Rows are every forecast with a reading, inside and outside closures."
                ),
            )
        ),
        "net_annual",
    )


def fig_net_loss_classes() -> None:
    """Excess score of LR_b20 alarms by whether a logged closure or labelled load shift overlaps."""
    t = pl.read_parquet(h.HARNESS_DIR / "net_loss_by_alarm.parquet").with_columns(
        cls=pl.when(pl.col("overlaps_labelled_load_shift"))
        .then(pl.lit("overlaps a labelled load shift"))
        .when(pl.col("overlaps_logged_closure"))
        .then(pl.lit("overlaps another logged closure"))
        .otherwise(pl.lit("overlaps no logged closure"))
    )
    g = t.group_by("cls").agg(excess=pl.col("excess_score_sum").sum(), spans=pl.len())
    total = g["excess"].sum()
    g = g.with_columns(share=pl.col("excess") / total)
    order = [
        "overlaps a labelled load shift",
        "overlaps another logged closure",
        "overlaps no logged closure",
    ]
    chart = (
        alt.Chart(g.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("cls:N", sort=order, title=None, axis=alt.Axis(labelLimit=380)),
            x=alt.X(
                "share:Q",
                title="Share of the LR_b20 mask's total excess score over K0",
                axis=alt.Axis(format="%"),
            ),
            tooltip=["cls", "spans", "excess"],
        )
        .properties(
            width=W - 60,
            height=130,
            title=alt.TitleParams(
                wrapped(
                    text="Three quarters of the LR_b20 mask's excess score sits in logged closures that are not labelled load shifts",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Each alarm span of LR_b20 (fold-averaged) is summed over the rows whose target lies from its step to 72 hours after its end is known. Excess is the arm's quantile score minus K0's."
                ),
            ),
        )
    )
    save(chart, "net_loss_classes")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_per_shift()
    fig_harm_by_primary()
    fig_arms()
    fig_net_annual()
    fig_net_loss_classes()
