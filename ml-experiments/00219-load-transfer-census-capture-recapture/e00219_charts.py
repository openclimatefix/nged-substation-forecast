# ruff: noqa: E501, D103
"""Draw experiment 219's figures as SVG from the stored tables.

The series are Primary substations and injected transfers, not NGED generators. Afterwards optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00219-*/e00219_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00219_common import WORK_DIR

OUT_DIR: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00219-load-transfer-census-capture-recapture"
)
WIDTH: Final = 600
DETECTORS: Final = ("RO-NC", "pair", "learned")
LABELS: Final = {"RO-NC": "RO-NC", "pair": "Pair alarm", "learned": "Learned"}
YEARS: Final = 61.83


def _chart(data: pl.DataFrame) -> Any:
    return alt.Chart(data)


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(line, 100),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_overlaps() -> None:
    events = pl.read_parquet(WORK_DIR / "events_L6_M48_10pct_f0.parquet")
    big = events.filter(pl.col("ratio") >= 0.4)
    rows = []
    for name, frame in (("All large events", events), ("Size 40% of p99 or more", big)):
        counts = frame.group_by(*DETECTORS).len()
        for r in counts.iter_rows(named=True):
            found = [LABELS[d] for d in DETECTORS if r[d]]
            rows.append(
                {
                    "panel": name,
                    "found_by": " + ".join(found),
                    "events": r["len"],
                    "detectors": len(found),
                }
            )
    data = pl.DataFrame(rows)
    order = (
        data.group_by("found_by")
        .agg(pl.col("events").sum(), pl.col("detectors").first())
        .sort(["detectors", "events"], descending=[False, True])["found_by"]
        .to_list()
    )
    base = _chart(data)
    bars = (
        base.mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("events:Q", title="Events (count)"),
            y=alt.Y("found_by:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        )
        .properties(width=WIDTH - 260, height=170)
    )
    text = base.mark_text(align="left", dx=3).encode(
        x="events:Q", y=alt.Y("found_by:N", sort=order), text="events:Q"
    )
    (
        (bars + text)
        .facet(
            row=alt.Row("panel:N", title=None, sort=["All large events", "Size 40% of p99 or more"])
        )
        .properties(
            title=_title(
                text="Most large events are found by one detector only",
                line="Count of matched events by the detectors that found them, over 15 Primaries and 61.8 Primary-years (one learned-score fold). An event is a 3-hour-or-longer interval of at least 10% of a Primary's 99th percentile of demand.",
            )
        )
        .save(OUT_DIR / "figure_1_overlaps.svg")
    )


def figure_events_per_year() -> None:
    t = pl.read_parquet(WORK_DIR / "census.parquet").filter(
        (pl.col("variant") == "L6_M48") & (pl.col("frac") == 0.10)
    )
    strata = ["all", "size 10-20% of p99", "size 20-40% of p99", "size 40%+ of p99"]
    rows = []
    for s in strata:
        sub = t.filter(pl.col("stratum") == s)
        rows.append(
            {
                "stratum": s,
                "seen": float(np.nanmedian(sub["n_seen"].to_numpy())) / YEARS,
                "estimate": float(np.nanmedian(sub["primary"].to_numpy())) / YEARS,
                "lo": float(np.nanmedian(sub["primary_lo"].to_numpy())) / YEARS,
                "hi": float(np.nanmedian(sub["primary_hi"].to_numpy())) / YEARS,
                "chao": float(np.nanmedian(sub["chao"].to_numpy())) / YEARS,
            }
        )
    data = pl.DataFrame(rows)
    base = _chart(data)
    y = alt.Y("stratum:N", sort=strata, title=None)
    x_scale = alt.Scale(type="sqrt", domain=[0, 160], nice=False)
    ticks = alt.Axis(values=[0, 2, 5, 10, 20, 50, 100, 150])
    chart = alt.layer(
        base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(
            x=alt.X(
                "lo:Q",
                scale=x_scale,
                axis=ticks,
                title="Events per Primary-year (square-root scale)",
            ),
            x2="hi:Q",
            y=y,
        ),
        base.mark_point(filled=True, size=80, color=ocf.DATA_BLUE).encode(
            x=alt.X("estimate:Q", scale=x_scale), y=y
        ),
        base.mark_point(filled=True, size=80, shape="diamond", color=ocf.BRAND_ORANGE).encode(
            x=alt.X("seen:Q", scale=x_scale), y=y
        ),
        base.mark_point(size=80, shape="square", color=ocf.DATA_GREEN).encode(
            x=alt.X("chao:Q", scale=x_scale), y=y
        ),
    ).properties(
        width=WIDTH - 150,
        height=170,
        title=_title(
            text="The estimated number of large events is far above the number seen, and its interval is very wide",
            line="Orange diamond: events seen by any detector. Blue dot and line: three-list estimate and 95% interval from resampling whole calendar months. Green square: Chao lower bound. Median over three learned-score folds.",
        ),
    )
    chart.save(OUT_DIR / "figure_2_events_per_primary_year.svg")


def figure_injected() -> None:
    inj = pl.read_parquet(WORK_DIR / "injected_census.parquet")
    rows = []
    for r in inj.iter_rows(named=True):
        known = r["known_n"]
        for name, key in (
            ("Chao lower bound", "chao"),
            ("RO-NC and pair", "lp_RO-NC_pair"),
            ("RO-NC and learned", "lp_RO-NC_learned"),
            ("Pair and learned", "lp_pair_learned"),
            ("Events seen by any detector", "n_seen"),
        ):
            rows.append({"stratum": r["stratum"], "estimator": name, "ratio": r[key] / known})
    data = pl.DataFrame(rows)
    order = ["all", "small third", "middle third", "large third"]
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.DATA_GREEN]
    names = [
        "Events seen by any detector",
        "Chao lower bound",
        "RO-NC and pair",
        "RO-NC and learned",
        "Pair and learned",
    ]
    band = (
        _chart(pl.DataFrame({"lo": [0.75], "hi": [1.25]}))
        .mark_rect(opacity=0.12, color="grey")
        .encode(x="lo:Q", x2="hi:Q")
    )
    rule = _chart(pl.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    pts = (
        _chart(data)
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X(
                "ratio:Q",
                title="Estimated count divided by the known injected count (1 = exact; smaller means undercount)",
                scale=alt.Scale(domain=[0.3, 1.3]),
                axis=alt.Axis(values=[0.4, 0.6, 0.75, 1.0, 1.25], format=".2f"),
            ),
            y=alt.Y("stratum:N", sort=order, title=None),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=names, range=colours),
                legend=alt.Legend(
                    orient="top", title=None, columns=2, labelLimit=400, symbolLimit=10
                ),
            ),
        )
    )
    (band + rule + pts).properties(
        width=WIDTH - 100,
        height=190,
        title=_title(
            text="Every estimator undercounts the known injected transfers; the grey band is the pre-registered 25%",
            line="Experiment 195's held-out injected transfers of at least 24 hours (419 events), split into thirds by size. Each detector's found flag comes from matching to the known event.",
        ),
    ).save(OUT_DIR / "figure_3_injected_known_count.svg")


def figure_recall() -> None:
    t = pl.read_parquet(WORK_DIR / "census.parquet").filter(
        (pl.col("variant") == "L6_M48") & (pl.col("frac") == 0.10)
    )
    logged = {"RO-NC": 0.51, "pair": 5 / 35, "learned": 23 / 35}
    rows = []
    for d in DETECTORS:
        for stratum, name in (
            ("all", "Label-free estimate, all large events"),
            ("size 40%+ of p99", "Label-free estimate, size 40% of p99 or more"),
        ):
            v = float(
                np.nanmedian(t.filter(pl.col("stratum") == stratum)[f"recall_{d}"].to_numpy())
            )
            rows.append({"detector": LABELS[d], "source": name, "recall": v})
        rows.append(
            {
                "detector": LABELS[d],
                "source": "Measured on the 35 logged transfers",
                "recall": logged[d],
            }
        )
    data = pl.DataFrame(rows)
    names = [
        "Measured on the 35 logged transfers",
        "Label-free estimate, all large events",
        "Label-free estimate, size 40% of p99 or more",
    ]
    _chart(data).mark_point(filled=True, size=110).encode(
        x=alt.X(
            "recall:Q",
            title="Share of transfers found (higher is better)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y("detector:N", title=None, sort=list(LABELS.values())),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=names, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY]),
            legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
        ),
    ).properties(
        width=WIDTH - 130,
        height=170,
        title=_title(
            text="The label-free recall estimate is far below the measured recall for all large events",
            line="Learned re-ranker: 23 of 35 (midpoint of 22 to 24). Three-list estimate, median over three learned-score folds. The 40%-or-more estimate rests on 10 events.",
        ),
    ).save(OUT_DIR / "figure_4_recall_against_logged.svg")


if __name__ == "__main__":
    figure_overlaps()
    figure_events_per_year()
    figure_injected()
    figure_recall()
