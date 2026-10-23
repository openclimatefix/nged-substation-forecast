# ruff: noqa: D103, E501, E731
"""Draw experiment 283's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations (labels D5 to D16) and simulated shifts, not NGED generators.

    uv run --no-sync python e00283_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00283_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment"
)
RESULTS: Final[Path] = WORK_DIR / "results"
WIDTH: Final[int] = 520
GROUPS: Final[list[str]] = [
    "G2 before alarm",
    "G3 during",
    "G4 after",
    "G5 near",
    "G1 ordinary",
    "G0 all rows",
]
GROUP_LABELS: Final[dict[str, str]] = {
    "G2 before alarm": "Before the alarm",
    "G3 during": "During the shift",
    "G4 after": "48 h after the end",
    "G5 near": "Within 7 days of a shift",
    "G1 ordinary": "Ordinary rows",
    "G0 all rows": "All rows, year pooled",
}
ARM_LABELS: Final[dict[str, str]] = {
    "PERFECT": "Perfect, readings set aside",
    "PERFECT_LAG": "Live delays, readings set aside",
    "PERFECT_ADD": "Perfect, megawatts added back (post hoc)",
    "PERFECT_LAG_ADD": "Live delays, megawatts added back (post hoc)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "PERFECT": ocf.DATA_BLUE,
    "PERFECT_LAG": ocf.DATA_SKY,
    "PERFECT_ADD": ocf.BRAND_ORANGE,
    "PERFECT_LAG_ADD": ocf.DATA_PURPLE,
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def panel(
    *, rows: list[dict[str, Any]], title: str, line: str, arms: list[str], x_title: str, height: int
) -> alt.LayerChart:
    """Dots with 95% interval rules, one row per row group, coloured by arm."""
    frame = pl.DataFrame(
        [
            {
                "group": GROUP_LABELS[r["group"]],
                "arm": ARM_LABELS[r["contrast"]],
                "est": r["pct"],
                "lo": r["pct_lo"],
                "hi": r["pct_hi"],
            }
            for r in rows
            if r["contrast"] in arms and r["group"] in GROUP_LABELS and r["pct"] == r["pct"]
        ]
    )
    order = [GROUP_LABELS[g] for g in GROUPS]
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[ARM_LABELS[a] for a in arms], range=[ARM_COLOURS[a] for a in arms]),
        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
    )
    y = alt.Y("group:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.4))
    offset = alt.YOffset("arm:N", scale=alt.Scale(domain=[ARM_LABELS[a] for a in arms]))
    rule = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(x=alt.X("lo:Q", title=x_title), x2="hi:Q", y=y, yOffset=offset, color=colour)
    )
    dot = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45, opacity=1)
        .encode(x="est:Q", y=y, yOffset=offset, color=colour)
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return alt.layer(zero, rule, dot).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH, height=height, title=_title(text=title, line=line)
    )


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.configure(font="sans-serif").save(OUT_DIR / name)


def main() -> None:
    """Draw every figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alt.themes.enable("default")
    report = json.loads((RESULTS / "report.json").read_text())
    groups = report["groups"]
    pick = lambda target, metric: [
        g for g in groups if g["target"] == target and g["metric"] == metric
    ]
    note = "Dot: estimate. Line: 95% interval from resampling whole simulated shifts (and whole calendar months for ordinary rows). Zero: same as no knowledge."
    fig1 = alt.vconcat(
        panel(
            rows=pick("nra", "tw95"),
            title="Perfect knowledge lowers the upper-tail score of the normal-arrangement forecast after a shift",
            line="Gain in twCRPS at the 95th percentile of earlier power, % of the no-knowledge score, against the true normal-arrangement series (higher is better). "
            + note,
            arms=["PERFECT", "PERFECT_LAG"],
            x_title="Gain, % of no-knowledge twCRPS (higher is better)",
            height=190,
        ),
        panel(
            rows=pick("nra", "qs"),
            title="Perfect knowledge lowers CRPS in every phase of a shift",
            line="Gain in CRPS (twice the mean pinball loss over 11 quantiles), % of the no-knowledge score, against the true normal-arrangement series (higher is better).",
            arms=["PERFECT", "PERFECT_LAG"],
            x_title="Gain, % of no-knowledge CRPS (higher is better)",
            height=190,
        ),
    ).resolve_scale(color="shared")
    save(fig1, "figure_1_gain_against_true_normal_arrangement.svg")
    fig2 = panel(
        rows=pick("obs", "cost0.5"),
        title="Setting readings aside does not lower curtailment cost; adding the shift back does",
        line="Gain in node-only curtailment cost at the 0.5 scaled limit, 12 to 36 hour leads, % of the no-knowledge cost, against observed power (higher is better). Relative comparison between forecasts at Primary substations; no money is saved. "
        + note,
        arms=["PERFECT", "PERFECT_LAG", "PERFECT_ADD", "PERFECT_LAG_ADD"],
        x_title="Gain, % of no-knowledge curtailment cost (higher is better)",
        height=260,
    )
    save(fig2, "figure_2_curtailment_cost_gain_observed_power.svg")
    sweep = [
        {**g, "group": g["group"]}
        for g in groups
        if g["target"] == "obs"
        and g["metric"] in {"cost0.5", "cost0.75", "cost0.9"}
        and g["contrast"] == "PERFECT_ADD"
        and g["group"] in {"G4 after", "G5 near"}
    ]
    frame = pl.DataFrame(
        [
            {
                "limit": f"{float(g['metric'][4:]):g} of the largest earlier flow",
                "group": GROUP_LABELS[g["group"]],
                "est": g["pct"],
                "lo": g["pct_lo"],
                "hi": g["pct_hi"],
            }
            for g in sweep
        ]
    )
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "limit:N",
            sort=[
                "0.5 of the largest earlier flow",
                "0.75 of the largest earlier flow",
                "0.9 of the largest earlier flow",
            ],
            title=None,
        ),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        yOffset="group:N",
    )
    fig3 = alt.layer(
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q"),
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Gain, % of no-knowledge curtailment cost (higher is better)"),
            x2="hi:Q",
        ),
        base.mark_point(filled=True, size=45, opacity=1).encode(x="est:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=130,
        title=_title(
            text="The curtailment gain stays positive as the limit rises",
            line="Gain of the megawatts-added-back arm (post hoc) in node-only curtailment cost against observed power, by scaled limit. Dot: estimate. Line: 95% interval.",
        ),
    )
    save(fig3, "figure_3_limit_sweep.svg")
    strata = [
        s
        for s in report["strata"]
        if s["contrast"] == "PERFECT"
        and s["metric"] == "tw95"
        and s["target"] == "nra"
        and s["group"] == "G5 near"
        and s["kind"] in {"lead", "substation", "season"}
        and s["pct"] == s["pct"]
    ]
    frame = pl.DataFrame(
        [
            {
                "stratum": s["stratum"],
                "kind": s["kind"],
                "est": s["pct"],
                "lo": s["pct_lo"],
                "hi": s["pct_hi"],
            }
            for s in strata
        ]
    )
    kinds = {
        "lead": "Forecast lead",
        "season": "Season of the onset",
        "substation": "Primary substation",
    }
    layers = []
    for kind, label in kinds.items():
        sub = frame.filter(pl.col("kind") == kind)
        y = alt.Y("stratum:N", sort=sub["stratum"].to_list(), title=None)
        layers.append(
            alt.layer(
                alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color="#888888", strokeDash=[4, 3])
                .encode(x="x:Q"),
                alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeWidth=2, color=ocf.DATA_BLUE)
                .encode(
                    x=alt.X("lo:Q", title="Gain, % of no-knowledge twCRPS (higher is better)"),
                    x2="hi:Q",
                    y=y,
                ),
                alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                .mark_point(filled=True, size=45, opacity=1, color=ocf.DATA_BLUE)
                .encode(x="est:Q", y=y),
            ).properties(width=WIDTH, height=22 * max(len(sub), 2) + 20, title=label)
        )
    fig4 = alt.vconcat(*layers).properties(
        title=_title(
            text="The upper-tail gain is largest at short leads and varies across substations",
            line="Gain of perfect knowledge in twCRPS at the 95th percentile of earlier power against the true normal-arrangement series, rows within 7 days of a shift. All simulated shifts are 25% of load or more and last over 1 day. Dot: estimate. Line: 95% interval.",
        )
    )
    save(fig4, "figure_4_strata.svg")
    real = json.loads((RESULTS / "real.json").read_text())
    frame = pl.DataFrame(
        [
            {
                "phase": {
                    "BEFORE_ALARM": "Before the alarm",
                    "DURING": "During the shift",
                    "AFTER": "48 h after the end",
                    "TAIL": "48 to 72 h after the end",
                }[r["group"]],
                "metric": {
                    "qs": "CRPS",
                    "ae_median": "MAE of the median",
                    "pin95": "Pinball loss at 0.95",
                }[r["metric"]],
                "est": r["pct"],
                "lo": r["pct_lo"],
                "hi": r["pct_hi"],
            }
            for r in real
            if r["group"] in {"BEFORE_ALARM", "DURING", "AFTER", "TAIL"}
            and r["metric"] in {"qs", "ae_median", "pin95"}
        ]
    )
    y = alt.Y(
        "phase:N",
        sort=[
            "Before the alarm",
            "During the shift",
            "48 h after the end",
            "48 to 72 h after the end",
        ],
        title=None,
    )
    colour = alt.Color(
        "metric:N",
        scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    fig5 = alt.layer(
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q"),
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("lo:Q", title="Gain, % of no-knowledge score (higher is better)"),
            x2="hi:Q",
            y=y,
            yOffset="metric:N",
            color=colour,
        ),
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45, opacity=1)
        .encode(x="est:Q", y=y, yOffset="metric:N", color=colour),
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="On the 13 logged real shifts, the logged mask helps after a shift and hurts during it",
            line="Gain of setting readings aside from the logged onset, against observed power, four Primary substations (descriptive only, not part of any decision). Dot: estimate. Line: 95% interval from resampling whole logged shifts.",
        ),
    )
    save(fig5, "figure_5_real_record.svg")


if __name__ == "__main__":
    main()
