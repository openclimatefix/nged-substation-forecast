# ruff: noqa: D103, E501
"""Draw experiment 297's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python e00297_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00297_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00297-lsc-need-a-detector-no-detector-arms-against-the-mask"
)
WIDTH: Final[int] = 520
LABELS: Final[dict[str, str]] = {
    "GAM_ONLY": "No correction (GAM only)",
    "GAM_ONLY_K0SPREAD": "No correction, K0 bands",
    "BAND_TABLE_PLAIN": "Band table",
    "BAND_TABLE_DISAGREE": "Band table by disagreement",
    "BAND_HEAD": "Band quantile head",
    "BAND_HEAD_HELDOUT": "Band head, held-out worlds",
    "ASOF_WIDEN": "As-of widening",
    "LEAD_DECAY": "Lead-decay filter",
    "K0": "K0 (filter, no knowledge)",
    "DET": "RO-NC mask",
    "DET_ADD": "RO-NC, megawatts added back",
    "PERFECT": "Perfect mask",
    "PERFECT_ADD": "Perfect, megawatts added back",
}
KIND: Final[dict[str, str]] = {
    **dict.fromkeys(
        [
            "GAM_ONLY",
            "GAM_ONLY_K0SPREAD",
            "BAND_TABLE_PLAIN",
            "BAND_TABLE_DISAGREE",
            "BAND_HEAD",
            "ASOF_WIDEN",
            "LEAD_DECAY",
        ],
        "No detector",
    ),
    **dict.fromkeys(["DET", "DET_ADD", "PERFECT", "PERFECT_ADD"], "Detector or perfect"),
}
COLOURS: Final[dict[str, str]] = {
    "No detector": ocf.DATA_BLUE,
    "Detector or perfect": ocf.BRAND_ORANGE,
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def gain_chart(
    *,
    rows: list[dict[str, Any]],
    target: str,
    metric: str,
    groups: list[str],
    x_title: str,
    title: str,
    line: str,
) -> alt.VConcatChart:
    frame = pl.DataFrame(
        [
            {
                "group": r["group"],
                "arm": LABELS[r["arm"]],
                "kind": KIND[r["arm"]],
                "est": r["pct"],
                "lo": r["pct_lo"],
                "hi": r["pct_hi"],
            }
            for r in rows
            if r["target"] == target
            and r["metric"] == metric
            and r["group"] in groups
            and r["arm"] in KIND
        ]
    )
    order = [LABELS[a] for a in LABELS if a in KIND]
    panels = []
    for group in groups:
        sub = frame.filter(pl.col("group") == group)
        colour = alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        )
        y = alt.Y("arm:N", sort=order, title=None)
        rule = (
            alt.Chart(sub)
            .mark_rule(strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title=x_title), x2="hi:Q", y=y, color=colour
            )
        )
        dot = (
            alt.Chart(sub)
            .mark_point(filled=True, size=45, opacity=1)
            .encode(x="est:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(zero, rule, dot).properties(width=WIDTH, height=22 * len(order), title=group)
        )
    return alt.vconcat(*panels).properties(title=_title(text=title, line=line))


def coverage_chart(*, rows: list[dict[str, Any]]) -> alt.VConcatChart:
    frame = pl.DataFrame(
        [
            {
                "side": r["side"],
                "arm": LABELS[r["arm"]],
                "kind": KIND[r["arm"]],
                "est": 100 * r["mean"],
                "lo": 100 * r["mean_lo"],
                "hi": 100 * r["mean_hi"],
            }
            for r in rows
            if r["target"] == "nra" and r["arm"] in KIND
        ]
    )
    order = [LABELS[a] for a in LABELS if a in KIND]
    panels = []
    for side in ("below q01", "above q99"):
        sub = frame.filter(pl.col("side") == side)
        colour = alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        )
        y = alt.Y("arm:N", sort=order, title=None)
        rule = (
            alt.Chart(sub)
            .mark_rule(strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title="Readings outside the band (% of readings)"),
                x2="hi:Q",
                y=y,
                color=colour,
            )
        )
        dot = (
            alt.Chart(sub)
            .mark_point(filled=True, size=45, opacity=1)
            .encode(x="est:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        )
        nominal = (
            alt.Chart(pl.DataFrame({"x": [1.0]}))
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(nominal, rule, dot).properties(
                width=WIDTH, height=22 * len(order), title=f"Reading {side} (nominal 1%)"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="After a shift, the lead-decay filter and the RO-NC mask miss the upper tail most",
            line="Share of true normal-arrangement readings in the 48 hours after a shift that fall outside the 1% to 99% band. Dot: estimate. Line: 95% interval from resampling whole shifts. Dashed line: nominal 1%. Closer to the dashed line is better.",
        )
    )


def main() -> None:
    report = json.loads((WORK_DIR / "results" / "report.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = report["gain_over_k0"]
    gain_chart(
        rows=rows,
        target="nra",
        metric="tw95",
        groups=["G4 after", "G5 near", "G1 ordinary"],
        x_title="Gain in upper-tail score over K0 (% of K0's score; positive is better)",
        title="Forecasts with no detector recover much of perfect knowledge's upper-tail gain after a shift",
        line="Threshold-weighted CRPS against the true normal-arrangement series. G4: 48 hours after a shift ends. G5: within 7 days of a shift. G1: ordinary rows. Dot: estimate. Line: 95% interval from resampling whole shifts (G1: whole months).",
    ).save(OUT_DIR / "figure_1_upper_tail_gain_over_k0.svg")
    gain_chart(
        rows=rows,
        target="obs",
        metric="cost0.5",
        groups=["G5 near", "G4 after"],
        x_title="Gain in curtailment cost over K0 (% of K0's cost; positive is better)",
        title="Only perfect knowledge with the megawatts added back lowers curtailment cost clearly",
        line="Node-only curtailment cost against observed power, scaled limit 0.5, leads 12 to 36 hours, GBP 2,000 per MWh. A Primary substation has no published limit, so this compares arms and states no money saved. Dot: estimate. Line: 95% interval from resampling whole shifts.",
    ).save(OUT_DIR / "figure_2_curtailment_cost_gain_over_k0.svg")
    coverage_chart(rows=report["coverage_g4"]).save(
        OUT_DIR / "figure_3_tail_coverage_after_a_shift.svg"
    )


if __name__ == "__main__":
    main()
