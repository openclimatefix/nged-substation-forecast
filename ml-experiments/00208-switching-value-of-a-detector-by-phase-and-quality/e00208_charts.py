# ruff: noqa: E501
"""Draw experiment 208's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations and injected transfers, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python e00208_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00208_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00208-switching-value-of-a-detector-by-phase-and-quality"
)
REPORT: Final[Path] = WORK_DIR / "report"
WIDTH: Final[int] = 520
HEADLINE: Final[str] = "headline (25% or more, over 1 d)"
PHASE_NAMES: Final[dict[str, str]] = {
    "BEFORE_ALARM": "Before the alarm",
    "DURING": "During the transfer",
    "AFTER": "48 hours after the end",
}
PHASE_COLOURS: Final[list[str]] = [ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_BLUE]
SOURCE_NAMES: Final[dict[str, str]] = {
    "PERFECT|MASK": "Perfect knowledge",
    "PERFECT_LAG|MASK": "Perfect, live delays",
    "DET|MASK": "RO-NC, as stored",
    "DET_NOFALSE|MASK": "RO-NC without its false alarms",
    "GAM_ONLY|NONE": "No recent power at all",
    "CLEAN|NONE": "Recent power without the transfer",
}
SOURCE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_SKY, ocf.DATA_GREEN, "#888888"
]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(OUT_DIR / f"{name}.svg")


def _rule(value: float, axis: str = "x") -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"v": [value]}))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(**{axis: "v:Q"})  # ty: ignore[unresolved-attribute]
    )


def main() -> None:
    """Draw the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strata = pl.read_parquet(REPORT / "injected_strata.parquet")
    head = strata.filter(pl.col("stratum") == HEADLINE)

    # ---- figure 1: share of the perfect gain recovered, 48 hours after the end ----
    arms = ["DET|MASK", "DET_NOFALSE|MASK", "PERFECT_LAG|MASK", "GAM_ONLY|NONE", "CLEAN|NONE"]
    d1 = (
        head.filter((pl.col("phase") == "AFTER") & pl.col("arm").is_in(arms))
        .with_columns(
            source=pl.col("arm").replace_strict(SOURCE_NAMES),
            lo=pl.col("share_lo").clip(-0.5, 2.0),
            hi=pl.col("share_hi").clip(-0.5, 2.0),
        )
    )
    order = [SOURCE_NAMES[a] for a in arms]
    f1 = alt.layer(
        alt.Chart(d1).mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Share of the perfect-knowledge gain recovered (larger is better)", scale=alt.Scale(domain=[-0.5, 2.0])),
            x2="hi:Q",
            y=alt.Y("source:N", title=None, sort=order, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("source:N", legend=None, scale=alt.Scale(domain=order, range=SOURCE_COLOURS[: len(order)])),
        ),
        alt.Chart(d1).mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
            x="share:Q", y=alt.Y("source:N", sort=order),
            color=alt.Color("source:N", legend=None),
        ),
        _rule(0.5),
        _rule(1.0),
    ).properties(
        width=WIDTH, height=190,
        title=_title(
            text="RO-NC recovers about half of the perfect-knowledge gain in the 48 hours after a transfer ends",
            line="Injected transfers of 25% of load or more lasting over 1 day, 6 substations, 44 simulated worlds. Dot: estimate. Line: 95% interval from resampling whole transfers. Dashed rules: 50% (the bar) and 100%.",
        ),
    )
    _save(f1, "figure_1_share_recovered")

    # ---- figure 2: gain by phase ----
    arms2 = ["PERFECT|MASK", "PERFECT_LAG|MASK", "DET|MASK", "GAM_ONLY|NONE"]
    d2 = head.filter(pl.col("arm").is_in(arms2)).with_columns(
        source=pl.col("arm").replace_strict(SOURCE_NAMES), phase_name=pl.col("phase").replace_strict(PHASE_NAMES)
    )
    names2 = [SOURCE_NAMES[a] for a in arms2]
    f2 = alt.layer(
        alt.Chart(d2).mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("gain_lo:Q", title="Quantile score gain over no mask (points of effective capacity; larger is better)"),
            x2="gain_hi:Q",
            y=alt.Y("phase_name:N", title=None, sort=list(PHASE_NAMES.values())),
            yOffset=alt.YOffset("source:N", sort=names2),
            color=alt.Color("source:N", sort=names2, legend=alt.Legend(orient="top", columns=2, title=None), scale=alt.Scale(domain=names2, range=SOURCE_COLOURS[:4])),
        ),
        alt.Chart(d2).mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x="gain:Q", y=alt.Y("phase_name:N", sort=list(PHASE_NAMES.values())),
            yOffset=alt.YOffset("source:N", sort=names2), color=alt.Color("source:N", legend=None),
        ),
        _rule(0.0),
    ).properties(
        width=WIDTH, height=260,
        title=_title(
            text="Knowing about a transfer helps during it and in the 48 hours after it ends, and less before the alarm would normally arrive",
            line="Injected transfers of 25% of load or more lasting over 1 day, scored against the true normal-arrangement series. Dot: estimate. Line: 95% interval from resampling whole transfers. Dashed rule: no gain.",
        ),
    )
    _save(f2, "figure_2_gain_by_phase")

    # ---- figure 3: value curves ----
    rows = []
    axes = {
        "Recall (share of transfers known)": {"RECALL_0.25|MASK": 0.25, "RECALL_0.5|MASK": 0.5, "RECALL_0.75|MASK": 0.75, "PERFECT|MASK": 1.0},
        "Onset delay (hours; the end is known at once; 48 hours and over added post hoc)": {"PERFECT|MASK": 0.0, "DELAY_6|MASK": 3.0, "DELAY_12|MASK": 6.0, "DELAY_24|MASK": 12.0, "DELAY_48|MASK": 24.0, "DELAYX_96|MASK": 48.0, "DELAYX_144|MASK": 72.0, "DELAYX_192|MASK": 96.0},
        "Share of RO-NC's missed transfers repaired": {"DET|MASK": 0.0, "REPAIR_0.25|MASK": 0.25, "REPAIR_0.5|MASK": 0.5, "REPAIR_0.75|MASK": 0.75, "REPAIR_1.0|MASK": 1.0},
        "MW error (subtracting the MW; plus or minus)": {"PERFECT|SUB": 0.0, "BLUR_0.25|SUB": 0.25, "BLUR_0.5|SUB": 0.5},
    }
    for axis, mapping in axes.items():
        for arm, x in mapping.items():
            for r in head.filter(pl.col("arm") == arm).iter_rows(named=True):
                rows.append({"axis": axis, "x": x, "phase": PHASE_NAMES[r["phase"]], "share": r["share"], "lo": r["share_lo"], "hi": r["share_hi"]})
    d3 = pl.DataFrame(rows).with_columns(
        share=pl.col("share").clip(-0.5, 1.5), lo=pl.col("lo").clip(-0.5, 1.5), hi=pl.col("hi").clip(-0.5, 1.5)
    )
    def panel(axis: str, x_title: str) -> alt.LayerChart:
        sub = d3.filter(pl.col("axis") == axis)
        base = sub.pipe(alt.Chart).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", title=x_title),
            color=alt.Color("phase:N", legend=alt.Legend(orient="top", title=None), scale=alt.Scale(domain=list(PHASE_NAMES.values()), range=PHASE_COLOURS)),
        )
        return alt.layer(
            base.mark_area(opacity=0.15).encode(y=alt.Y("lo:Q", title="Share of perfect gain", scale=alt.Scale(domain=[-0.5, 1.5])), y2="hi:Q"),
            base.mark_line(point=True).encode(y="share:Q"),
            alt.Chart(pl.DataFrame({"v": [0.5]})).mark_rule(strokeDash=[4, 3], color="#444").encode(y="v:Q"),  # ty: ignore[unresolved-attribute]
        ).properties(width=WIDTH, height=120, title=alt.TitleParams(axis, anchor="start", fontSize=11))

    f3 = alt.vconcat(*[panel(axis, axis) for axis in axes]).properties(
        title=_title(
            text="The value of a detector falls as it misses transfers or arrives late",
            line="Perfect knowledge degraded in controlled steps, injected transfers of 25% of load or more lasting over 1 day. Line: estimate. Band: 95% interval from resampling whole transfers. Dashed rule: 50%.",
        )
    ).resolve_scale(color="shared")
    d3.write_parquet(REPORT / "value_curves.parquet")
    _save(f3, "figure_3_value_curves")

    # ---- figure 4: strata ----
    s4 = strata.filter(pl.col("arm") == "PERFECT|MASK").with_columns(phase_name=pl.col("phase").replace_strict(PHASE_NAMES))
    keep = [s for s in s4["stratum"].unique().to_list() if s != "all transfers" and not s.startswith("headline")]
    s4 = s4.filter(pl.col("stratum").is_in(keep))
    order4 = [s for s in (
        "size under 10%", "size 10% to 25%", "size 25% or more", "duration 2 to 6 h", "duration 6 to 24 h", "duration over 1 d",
        "season winter (Nov to Mar)", "season summer (Apr to Oct)", "lead 0 to 12 h", "lead 12 to 36 h", "lead 36 h to 7 d",
        "substation D5", "substation D9", "substation D10", "substation D11", "substation D14", "substation D16",
    ) if s in keep]
    f4 = alt.layer(
        alt.Chart(s4).mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("gain_lo:Q", title="Gain over no mask from perfect knowledge (points; larger is better)"),
            x2="gain_hi:Q", y=alt.Y("stratum:N", title=None, sort=order4), yOffset=alt.YOffset("phase_name:N", sort=list(PHASE_NAMES.values())),
            color=alt.Color("phase_name:N", sort=list(PHASE_NAMES.values()), legend=alt.Legend(orient="top", title=None), scale=alt.Scale(domain=list(PHASE_NAMES.values()), range=PHASE_COLOURS)),
        ),
        alt.Chart(s4).mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            x="gain:Q", y=alt.Y("stratum:N", sort=order4), yOffset=alt.YOffset("phase_name:N", sort=list(PHASE_NAMES.values())), color=alt.Color("phase_name:N", legend=None),
        ),
        _rule(0.0),
    ).properties(
        width=WIDTH, height=520,
        title=_title(
            text="The benefit by size, duration, season, forecast lead, and substation",
            line="Perfect knowledge, all injected transfers in each stratum. Dot: estimate. Line: 95% interval from resampling whole transfers. Dashed rule: no gain.",
        ),
    )
    _save(f4, "figure_4_strata")

    # ---- figure 5: harm on ordinary days ----
    prec = pl.read_parquet(REPORT / "precision_harm.parquet").with_columns(
        source=pl.format("Logged transfers plus false alarms, precision {}", pl.col("precision").round(2))
    ).select("source", "harm", "lo", "hi")
    oh = pl.read_parquet(REPORT / "ordinary_harm.parquet").with_columns(
        source=pl.col("source").replace({"PAIR_GATED": "Gated pair alarm", "PAIR_NEW": "Pair alarm", "LEARNED_R": "Re-ranker (experiment 195)", "LEARNED_F": "Logistic regression (experiment 195)", "RONC_195": "RO-NC (experiment 195 re-run)"})
    ).rename({"lo": "lo", "hi": "hi"}).select("source", "harm", "lo", "hi")
    d5 = pl.concat([prec, oh])
    order5 = d5["source"].to_list()
    f5 = alt.layer(
        alt.Chart(d5).mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Mean score minus no mask, ordinary days (points; smaller is better)"), x2="hi:Q", y=alt.Y("source:N", title=None, sort=order5, axis=alt.Axis(labelLimit=330)),
        ),
        alt.Chart(d5).mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(x="harm:Q", y=alt.Y("source:N", sort=order5)),  # ty: ignore[unresolved-attribute]
        _rule(0.0),
        alt.Chart(pl.DataFrame({"v": [-0.05, 0.05]})).mark_rule(strokeDash=[2, 2], color="#444").encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH, height=220,
        title=_title(
            text="Random false alarms cost little on ordinary days, but the alarms of every real detector cost more than the limit allows",
            line="Six substations, 27 months of forecast runs outside logged transfers. Dot: estimate. Line: 95% interval from resampling whole calendar months. Dotted rules: the limit of plus or minus 0.05.",
        ),
    )
    _save(f5, "figure_5_harm_on_ordinary_days")

    # ---- figure 6: real record ----
    real = pl.read_parquet(REPORT / "real_gain.parquet").filter((pl.col("scope") == "all") & (pl.col("phase") == "AFTER") & pl.col("arm").is_in(["LOGGED", "PAIR_NEW", "PAIR_GATED", "RONC", "LEARNED_R"]))
    names6 = {"LOGGED": "NGED's switching table", "PAIR_NEW": "Pair alarm", "PAIR_GATED": "Gated pair alarm", "RONC": "RO-NC", "LEARNED_R": "Re-ranker (experiment 195)"}
    order6 = list(names6.values())
    d6 = real.with_columns(source=pl.col("arm").replace_strict(names6))
    f6 = alt.layer(
        alt.Chart(d6).mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("gain_lo:Q", title="Gain over no mask, 48 hours after a logged transfer (points; larger is better)"), x2="gain_hi:Q", y=alt.Y("source:N", title=None, sort=order6, axis=alt.Axis(labelLimit=300)),
        ),
        alt.Chart(d6).mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(x="gain:Q", y=alt.Y("source:N", sort=order6)),  # ty: ignore[unresolved-attribute]
        _rule(0.0),
    ).properties(
        width=WIDTH, height=170,
        title=_title(
            text="On NGED's logged transfers, knowing the transfer lowers the error after it ends",
            line="Six substations, transfers logged in the scored and replicate months, scored against the metered power. Dot: estimate. Line: 95% interval from resampling whole transfers. Dashed rule: no gain.",
        ),
    )
    _save(f6, "figure_6_real_transfers")


if __name__ == "__main__":
    main()
