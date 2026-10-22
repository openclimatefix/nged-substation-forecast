# ruff: noqa: E501
"""Draw the figures of experiment 254 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00254-*/e00254_charts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import e00254_data as d
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, wrapped

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00254-lsc-alarm-precision-against-full-log-with-chance-base-rate"
)
W: Final[int] = CONTENT_WIDTH_PX - 300
NAMES: Final[dict[str, str]] = {
    "LR_b20": "LR_b20 (learned, 20 candidates)",
    "LR_b80": "LR_b80 (learned, 80 candidates)",
    "RO_b80": "RO_b80 (RO-NC, 80 candidates)",
    "PG_g70": "PG_g70 (pair alarm, balance 0.7)",
}
ORDER: Final[list[str]] = list(NAMES.values())
GROUP: Final[dict[str, str]] = {
    "strict match": "strict match",
    "log-near (24 h)": "log-near, not a strict match",
    "balanced-opposite candidate": "balanced-opposite candidate",
    "D8 drop edge": "D8 edge, block day, holiday, or clock change",
    "block day": "D8 edge, block day, holiday, or clock change",
    "bank holiday or clock change": "D8 edge, block day, holiday, or clock change",
    "none": "none of the classes",
}
GROUPS: Final[list[str]] = list(dict.fromkeys(GROUP.values()))
GROUP_COLOURS: Final[list[str]] = [
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    "#9AA0A6",
]


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def fig_classes() -> None:
    """Share of each source's alarms in each class."""
    t = pl.read_parquet(d.WORK_DIR / "classes.parquet").with_columns(
        group=pl.col("class").replace_strict(GROUP), source=pl.col("source").replace_strict(NAMES)
    )
    t = t.group_by("source", "group").agg(alarms=pl.col("alarms").sum())
    t = t.with_columns(share=pl.col("alarms") / pl.col("alarms").sum().over("source"))
    chart = (
        alt.Chart(t.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("source:N", sort=ORDER, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "share:Q",
                title="Share of the source's alarms",
                axis=alt.Axis(format="%"),
                stack="normalize",
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=GROUPS, range=GROUP_COLOURS),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=420),
            ),
            tooltip=["source", "group", "alarms"],
        )
        .properties(
            width=W,
            height=150,
            title=alt.TitleParams(
                wrapped(
                    text="Almost half of LR_b20's alarms are not strict matches but sit near a logged closure",
                    width=58,
                ),
                subtitle=wrapped(
                    text="Each bar is one alarm source over the whole record and all 16 series; every alarm falls in the first class that fits, from strict match down to none. Alarms are counted once per stepped series and stamp, averaged over folds where a learned model was held out."
                ),
            ),
        )
    )
    save(chart, "classes")


def fig_chance() -> None:
    """Observed log-near share against the week-shifted distribution."""
    rows = []
    summary = pl.read_parquet(d.WORK_DIR / "summary.parquet")
    for r in summary.iter_rows(named=True):
        shifted = np.load(d.WORK_DIR / f"shifted_shares_{r['source']}.npy")
        rows.append(
            {
                "source": NAMES[r["source"]],
                "obs": r["obs_share"],
                "obs_lo": r["obs_share_lo"],
                "obs_hi": r["obs_share_hi"],
                "lo": float(np.quantile(shifted, 0.025)),
                "hi": float(np.quantile(shifted, 0.975)),
                "mean": float(shifted.mean()),
            }
        )
    t = pl.DataFrame(rows).to_pandas()
    y = alt.Y("source:N", sort=ORDER, title=None, axis=alt.Axis(labelLimit=260))
    x = alt.X(
        "mean:Q",
        title="Share of non-matching alarms with a logged closure edge within 24 hours",
        scale=alt.Scale(domain=[0, 0.8]),
        axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8]),
    )
    band = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=10, color="#BFC5CC", opacity=0.9)
        .encode(y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 0.8])), x2="hi:Q")
    )
    mean = alt.Chart(t).mark_tick(color="#5F6368", thickness=2, size=22).encode(y=y, x=x)  # ty: ignore[unresolved-attribute]
    obs_rule = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.DATA_BLUE, strokeWidth=2)
        .encode(y=y, x="obs_lo:Q", x2="obs_hi:Q")
    )
    obs = alt.Chart(t).mark_point(filled=True, color=ocf.DATA_BLUE, size=110).encode(y=y, x="obs:Q")  # ty: ignore[unresolved-attribute]
    chart = (band + mean + obs_rule + obs).properties(
        width=W,
        height=150,
        title=alt.TitleParams(
            wrapped(
                text="Every source sits above the share expected by chance, by the widest margin for LR_b20",
                width=58,
            ),
            subtitle=wrapped(
                text="Blue dot: share observed, with the 95% interval from resampling whole months. Grey bar: 2.5% to 97.5% range of 200 shifts of every alarm by whole weeks (3 to 100 weeks either way); tick: mean of the shifts. A dot right of the grey bar is more logged closures than chance gives."
            ),
        ),
    )
    save(chart, "observed_against_shifted")


def fig_precision() -> None:
    """Corrected false alarms per matched load shift by source, and corrected precision by Primary."""
    s = pl.read_parquet(d.WORK_DIR / "summary.parquet").with_columns(
        source=pl.col("source").replace_strict(NAMES)
    )
    y = alt.Y("source:N", sort=ORDER, title=None, axis=alt.Axis(labelLimit=260))
    dom = alt.Scale(domain=[0, 30], clamp=True)
    lr = s.filter(pl.col("source") == NAMES["LR_b20"]).row(0, named=True)
    t = s.to_pandas()
    rule = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.DATA_BLUE, strokeWidth=2)
        .encode(
            y=y,
            x=alt.X(
                "fa_per_match_lo:Q",
                scale=dom,
                title="False alarms per matched load shift (smaller is better)",
            ),
            x2="fa_per_match_hi:Q",
        )
    )
    dot = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, color=ocf.DATA_BLUE, size=110)
        .encode(y=y, x=alt.X("fa_per_match:Q", scale=dom))
    )
    naive = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_point(shape="diamond", filled=False, color=ocf.BRAND_ORANGE, size=110, strokeWidth=2)
        .encode(y=y, x=alt.X("fa_per_match_all_false:Q", scale=dom))
    )
    top = (rule + dot + naive).properties(
        width=W,
        height=150,
        title=alt.TitleParams(
            wrapped(
                text=f"Removing chance coincidences cuts LR_b20's false alarms per matched load shift from {lr['fa_per_match_all_false']:.1f} to {lr['fa_per_match']:.1f}",
                width=58,
            ),
            subtitle=wrapped(
                text="Blue dot: false alarms per matched load shift after counting the excess near-closure alarms as real, with the 95% interval from resampling whole months. Orange diamond: the same count if every alarm that is not a strict match were false. The pair alarm's upper end runs past the axis (79)."
            ),
        ),
    )
    p = (
        pl.read_parquet(d.WORK_DIR / "per_primary.parquet")
        .filter(pl.col("source") == "LR_b20")
        .with_columns(
            name=pl.col("label")
            + " ("
            + pl.col("alarms").round(0).cast(pl.Int64).cast(pl.String)
            + " alarms)"
        )
        .sort("precision", descending=True)
    )
    order = p["name"].to_list()
    py = alt.Y("name:N", sort=order, title=None)
    px = alt.X(
        "precision:Q",
        title="Corrected precision: estimated share of alarms that are real (larger is better)",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%"),
    )
    pt = p.to_pandas()
    bottom = (
        alt.Chart(pt)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.DATA_BLUE, strokeWidth=2)
        .encode(
            y=py, x=alt.X("precision_lo:Q", scale=alt.Scale(domain=[0, 1])), x2="precision_hi:Q"
        )
        + alt.Chart(pt).mark_point(filled=True, color=ocf.DATA_BLUE, size=90).encode(y=py, x=px)  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W,
        height=240,
        title=alt.TitleParams(
            wrapped(
                text="Corrected precision of LR_b20 varies from 0% to 85% across Primary substations",
                width=58,
            ),
            subtitle=wrapped(
                text="One row per Primary substation with at least 5 alarms; D-labels are the published anonymised labels, and 'series N' is an unlabelled Primary substation. Dot: estimate; line: 95% interval from resampling whole months."
            ),
        ),
    )
    save(alt.vconcat(top, bottom).resolve_scale(color="independent"), "corrected_precision")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_classes()
    fig_chance()
    fig_precision()
