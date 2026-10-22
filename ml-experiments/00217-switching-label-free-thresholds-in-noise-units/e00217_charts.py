# ruff: noqa: E501, D103
"""Draw experiment 217's figures as SVG from the stored tables.

The series are Primary substations, not NGED generators. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00217-*/e00217_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00217_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

WORK_DIR: Final = score.WORK_DIR
OUT_DIR: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00217-switching-label-free-thresholds-in-noise-units"
)
WIDTH: Final = 640
SPECIAL: Final = ("D8", "D9")
NAMES: Final = {
    "RO-NC/current": "RO-NC, 4.35 noise units",
    "pair/current": "Pair alarm, 3.75 noise units",
    "RO-NC XL stored": "RO-NC XL (label-tuned)",
    "R re-ranker": "Learned re-ranker (label-tuned)",
}


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


def _colour() -> alt.Color:
    return alt.Color(
        "group:N", title=None,
        scale=alt.Scale(domain=["D8 and D9", "Other Primaries"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(orient="top"),
    )  # fmt: skip


def figure_rates() -> None:
    rates = pl.read_parquet(WORK_DIR / "rates.parquet").filter(
        (pl.col("group") == "all") & pl.col("detector").is_in(list(NAMES))
    )
    data = rates.with_columns(
        panel=pl.col("detector").replace(NAMES),
        group=pl.when(pl.col("site").is_in(SPECIAL))
        .then(pl.lit("D8 and D9"))
        .otherwise(pl.lit("Other Primaries")),
    )
    order = (
        data.filter(pl.col("detector") == "RO-NC/current")
        .sort("rate", descending=True)["site"]
        .to_list()
    )
    base = _chart(data)
    chart = (
        alt.layer(
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X(
                    "lo:Q",
                    title="Alarms per Primary-year",
                    scale=alt.Scale(type="sqrt", domain=[0, 40], nice=False),
                    axis=alt.Axis(values=[0, 2, 5, 10, 20, 40]),
                ),
                x2="hi:Q",
                y=alt.Y("site:N", sort=order, title=None),
                color=_colour(),
            ),
            base.mark_point(filled=True, size=50).encode(
                x="rate:Q", y=alt.Y("site:N", sort=order), color=_colour()
            ),
        )
        .properties(width=120, height=250)
        .facet(
            column=alt.Column(
                "panel:N",
                title=None,
                sort=list(NAMES.values()),
                header=alt.Header(labelFontSize=10, labelLimit=140),
            )
        )
    )
    chart = chart.properties(
        title=_title(
            text="Alarm rates differ by a factor of 20 between Primaries at a fixed threshold",
            line="Dot: alarms per Primary-year over the whole residual history. Line: 95% interval from resampling whole calendar months. Square-root axis. D18 has only 1 year of residual, so its interval is wide. the first two panels use one threshold in noise units for every Primary, the last two use a threshold tuned on switching labels.",
        )
    )
    chart.save(OUT_DIR / "figure_1_alarm_rate_by_primary.svg")


def figure_predicted() -> None:
    rates = pl.read_parquet(WORK_DIR / "rates.parquet").filter(
        (pl.col("group") == "all") & (pl.col("detector") == "RO-NC/current")
    )
    features = score.feature_table(stats=pl.read_parquet(WORK_DIR / "stats.parquet"))
    r = rates.join(features, on="site")
    y = np.log2(np.maximum(r["rate"].to_numpy(), score.RATE_FLOOR))
    pred = score.lopo(x=r["log_misfit_share"].to_numpy(), y=y)
    data = pl.DataFrame({
        "site": r["site"], "observed": 2**y, "predicted": 2**pred,
        "group": ["D8 and D9" if s in SPECIAL else "Other Primaries" for s in r["site"]],
    })  # fmt: skip
    line = pl.DataFrame({"a": [0.5, 25.0], "b": [0.5, 25.0], "lo": [0.25, 12.5], "hi": [1.0, 50.0]})
    scale = alt.Scale(type="log", domain=[0.5, 50])
    axis = alt.Axis(values=[0.5, 1, 2, 5, 10, 20, 50])
    chart = alt.layer(
        _chart(line)
        .mark_area(opacity=0.12, color="#888")
        .encode(x=alt.X("a:Q", scale=scale, axis=axis), y=alt.Y("lo:Q", scale=scale), y2="hi:Q"),
        _chart(line).mark_line(strokeDash=[4, 3], color="#555").encode(x="a:Q", y="b:Q"),
        _chart(data)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "predicted:Q",
                title="Predicted alarms per Primary-year (leave one Primary out)",
                scale=scale,
                axis=axis,
            ),
            y=alt.Y("observed:Q", title="Observed alarms per Primary-year", scale=scale, axis=axis),
            color=_colour(),
        ),
        _chart(data)
        .mark_text(align="left", dx=7, fontSize=10)
        .encode(x="predicted:Q", y="observed:Q", text="site:N"),
    ).properties(
        width=WIDTH - 90,
        height=340,
        title=_title(
            text="The share of days the model is badly off predicts a Primary's alarm rate",
            line="RO-NC at 4.35 noise units. Predictor: share of days on which the day's mean residual exceeds 3 noise units, fitted on the other 14 Primaries. Dashed line: predicted equals observed. Grey band: within a factor of 2. A zero rate is drawn at 0.25.",
        ),
    )
    chart.save(OUT_DIR / "figure_2_predicted_against_observed.svg")


def figure_threshold() -> None:
    curves = pl.read_parquet(WORK_DIR / "curves.parquet").with_columns(
        group=pl.when(pl.col("site").is_in(SPECIAL))
        .then(pl.lit("D8 and D9"))
        .otherwise(pl.lit("Other Primaries")),
        rate=pl.max_horizontal(pl.col("rate"), pl.lit(0.1)),
    )
    med = curves.group_by("threshold").agg(rate=pl.col("rate").median())
    scale = alt.Scale(type="log", domain=[0.1, 60])
    axis = alt.Axis(values=[0.1, 0.5, 1, 2, 5, 10, 20, 50])
    lines = (
        _chart(curves)
        .mark_line(opacity=0.8)
        .encode(
            x=alt.X(
                "threshold:Q",
                title="Threshold (noise units of the 24-hour step statistic)",
                axis=alt.Axis(values=[2.5, 3, 3.5, 4.35, 5, 6, 8], format=".3~g"),
            ),
            y=alt.Y("rate:Q", title="Alarms per Primary-year", scale=scale, axis=axis),
            detail="site:N",
            color=_colour(),
            strokeWidth=alt.condition(
                alt.FieldOneOfPredicate(field="site", oneOf=list(SPECIAL)),
                alt.value(3),
                alt.value(1.2),
            ),
        )
    )
    median = (
        _chart(med).mark_line(color="black", strokeDash=[4, 3]).encode(x="threshold:Q", y="rate:Q")
    )
    (lines + median).properties(
        width=WIDTH - 90,
        height=300,
        title=_title(
            text="The lower the threshold, the more alike the Primaries' alarm rates",
            line="RO-NC alarms per Primary-year against the threshold, one line per Primary, 15 Primaries. Dashed black line: median over Primaries. A zero rate is drawn at 0.1. At 2.5 noise units all 13 Primaries other than D8 and D9 are within a factor of 2 of the median; at 4.35, 9 of 13 are.",
        ),
    ).save(OUT_DIR / "figure_3_rate_against_threshold.svg")


def figure_d8_d9() -> None:
    rates = pl.read_parquet(WORK_DIR / "rates.parquet").filter(pl.col("group") == "all")
    detectors = [
        "RO-NC/current",
        "RO-NC/R1",
        "RO-NC/R2",
        "pair/current",
        "RO-NC XL stored",
        "R re-ranker",
    ]
    labels = {
        "RO-NC/current": "RO-NC, current noise", "RO-NC/R1": "RO-NC, MAD of all months", "RO-NC/R2": "RO-NC, 95th percentile",
        "pair/current": "Pair alarm, current noise", "RO-NC XL stored": "RO-NC XL (label-tuned)", "R re-ranker": "Learned re-ranker (label-tuned)",
    }  # fmt: skip
    rows = []
    for det in detectors:
        r = rates.filter(pl.col("detector") == det)
        med = float(np.median(r["rate"].to_numpy()))
        rows += [
            {"detector": labels[det], "site": s, "ratio": max(v / med, 0.05)}
            for s, v in zip(r["site"], r["rate"], strict=True)
            if s in SPECIAL
        ]
    data = pl.DataFrame(rows)
    bands = pl.DataFrame({"lo": [0.5, 1 / 3], "hi": [2.0, 3.0], "level": ["factor 2", "factor 3"]})
    scale = alt.Scale(type="log", domain=[0.05, 6])
    axis = alt.Axis(values=[0.05, 0.1, 0.33, 0.5, 1, 2, 3, 6], format="g")
    chart = (
        alt.layer(
            _chart(bands)
            .mark_rect(opacity=0.12)
            .encode(
                x=alt.X("lo:Q", scale=scale, axis=axis),
                x2="hi:Q",
                color=alt.Color("level:N", legend=None, scale=alt.Scale(range=["#666", "#bbb"])),
            ),
            _chart(pl.DataFrame({"v": [1.0]}))
            .mark_rule(strokeDash=[4, 3], color="#555")
            .encode(x="v:Q"),
            _chart(data)
            .mark_point(filled=True, size=90)
            .encode(
                x=alt.X(
                    "ratio:Q",
                    title="Alarm rate divided by the median over Primaries",
                    scale=scale,
                    axis=axis,
                ),
                y=alt.Y(
                    "detector:N",
                    sort=list(labels.values()),
                    title=None,
                    axis=alt.Axis(labelLimit=260),
                ),
                color=alt.Color(
                    "site:N",
                    title=None,
                    scale=alt.Scale(domain=list(SPECIAL), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                    legend=alt.Legend(orient="top"),
                ),
                shape=alt.Shape("site:N", legend=None),
            ),
        )
        .resolve_scale(color="independent")
        .properties(
            width=WIDTH - 280,
            height=200,
            title=_title(
                text="D8 and D9 sit within a factor of 3 of the median at a fixed noise-unit threshold; only RO-NC XL puts D9 outside",
                line="Alarm rate of D8 and D9 as a multiple of the median over the 15 Primaries. Dark grey band: within a factor of 2; light grey band: within a factor of 3. A zero rate is drawn at 0.05.",
            ),
        )
    )
    chart.save(OUT_DIR / "figure_4_d8_and_d9.svg")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_rates()
    figure_predicted()
    figure_threshold()
    figure_d8_d9()
