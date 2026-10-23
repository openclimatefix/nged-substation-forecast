# ruff: noqa: D103, E501, PERF401
"""Draw experiment 312's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python e00312_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00312_common import RESULTS_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00312-lsc-how-good-must-a-detector-be-against-the-band-head"
)
WIDTH: Final[int] = 420
AXES: Final[dict[str, list[tuple[str, str]]]] = {
    "Share of shifts found (recall)": [
        ("ADD_R25", "25% found"),
        ("ADD_R50", "50% found"),
        ("ADD_R75", "75% found"),
        ("ADD_PERFECT", "100% found (perfect)"),
    ],
    "Delay before a shift is known": [
        ("ADD_D6", "3 hours"),
        ("ADD_D12", "6 hours"),
        ("ADD_D24", "12 hours"),
        ("ADD_D48", "24 hours"),
        ("ADD_D96", "2 days"),
        ("ADD_D144", "3 days"),
        ("ADD_D192", "4 days"),
    ],
    "Error in the megawatts": [
        ("ADD_B25", "random +/-25%"),
        ("ADD_B50", "random +/-50%"),
        ("ADD_P25", "always 25% too high"),
        ("ADD_M25", "always 25% too low"),
    ],
    "False alarms added": [
        ("ADD_P75", "precision 0.75"),
        ("ADD_P50", "precision 0.5"),
    ],
    "All four together (post hoc)": [("ADD_MIX", "all four, mildly")],
}
VARIANTS: Final[dict[str, str]] = {
    "PIT": "Band position",
    "IND": "Indicators",
    "PIT_IND": "Band position + indicators",
    "IND_MW": "Indicators + added-back MW",
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 75),
        subtitle=textwrap.wrap(line, 95),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def dot_panel(
    *,
    frame: pl.DataFrame,
    order: list[str],
    colour: str,
    x_title: str,
    title: str,
    zero: float = 0.0,
    height: int | None = None,
    domain: tuple[float, float] = (-2.0, 10.0),
) -> alt.LayerChart:
    y = alt.Y("level:N", sort=order, title=None, axis=alt.Axis(labelLimit=190))
    rule = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2, color=colour)
        .encode(x=alt.X("lo:Q", title=x_title, scale=alt.Scale(domain=domain)), x2="hi:Q", y=y)  # ty: ignore[unresolved-attribute]
    )
    dot = (
        alt.Chart(frame)
        .mark_point(filled=True, size=45, opacity=1, color=colour)
        .encode(x=alt.X("est:Q", scale=alt.Scale(domain=domain)), y=y)  # ty: ignore[unresolved-attribute]
    )
    base = (
        alt.Chart(pl.DataFrame({"x": [zero]}))
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain)))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(base, rule, dot).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH, height=height or 22 * len(order), title=title
    )


def sweep_chart(
    *,
    rows: list[dict[str, Any]],
    cell: str,
    domain: tuple[float, float],
    x_title: str,
    title: str,
    line: str,
) -> alt.VConcatChart:
    panels = []
    for axis, levels in AXES.items():
        data = []
        for arm, label in levels:
            hit = next(r for r in rows if r["arm"] == arm and r["cell"] == cell)
            data.append(
                {"level": label, "est": hit["pct"], "lo": hit["pct_lo"], "hi": hit["pct_hi"]}
            )
        panels.append(
            dot_panel(
                frame=pl.DataFrame(data),
                order=[label for _, label in levels],
                colour=ocf.BRAND_ORANGE,
                x_title=x_title,
                title=axis,
                domain=domain,
            )
        )
    return alt.vconcat(*panels).properties(title=_title(text=title, line=line))


def heads_chart(*, rows: list[dict[str, Any]]) -> alt.VConcatChart:
    panels = []
    for cell, label in (
        ("cost0.5 G5", "Curtailment cost, within 7 days of a shift"),
        ("tw95 G5", "Upper-tail score, within 7 days of a shift"),
    ):
        data = []
        for variant, name in VARIANTS.items():
            for mode, mode_label in (("cross", "all worlds"), ("held", "held out")):
                hit = next(
                    (
                        r
                        for r in rows
                        if r["variant"] == variant
                        and r["mode"] == mode
                        and r["read"] == "even origin months"
                        and r["cell"] == cell
                    ),
                    None,
                )
                if hit:
                    data.append(
                        {
                            "level": f"{name}, {mode_label}",
                            "est": hit["pct"],
                            "lo": hit["pct_lo"],
                            "hi": hit["pct_hi"],
                        }
                    )
        order = [d["level"] for d in data]
        panels.append(
            dot_panel(
                frame=pl.DataFrame(data),
                order=order,
                colour=ocf.DATA_BLUE,
                x_title="Gain over the head without the new inputs (% of its score; positive is better)",
                title=label,
                height=18 * len(order),
                domain=(-4.0, 6.0),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Band position adds little to the quantile head, and even perfect knowledge adds under 3%",
            line="Even-month origins only. Each head is compared with the band quantile head fitted the same way. Dot: estimate. Line: 95% interval from resampling whole shifts. Zero: no change.",
        )
    )


def detector_chart(*, band: dict[str, Any]) -> alt.LayerChart:
    data = []
    for score, label in (
        ("pit24", "Band position, mean score"),
        ("ext24", "Band position, absolute"),
    ):
        for r in band[f"{score}_curve"]:
            if r["odd_alarms_per_match"] > 0 and r["odd_alarms_per_match"] < 40:
                data.append(
                    {
                        "signal": label,
                        "alarms": r["odd_alarms_per_match"],
                        "recall": 100 * r["odd_recall"],
                        "k": r["k"],
                    }
                )
    ro = band["RO_NC_odd"]
    data.append(
        {
            "signal": "RO-NC",
            "alarms": ro["alarms_per_match"],
            "recall": 100 * ro["recall"],
            "k": 0.0,
        }
    )
    frame = pl.DataFrame(data)
    colour = alt.Color(
        "signal:N",
        scale=alt.Scale(
            domain=["Band position, mean score", "Band position, absolute", "RO-NC"],
            range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    lines = (
        alt.Chart(frame.filter(pl.col("signal") != "RO-NC"))
        .mark_point(filled=True, size=40, opacity=0.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "alarms:Q",
                title="Alarms per matched shift (fewer is better)",
                scale=alt.Scale(domain=[0, 40]),
            ),
            y=alt.Y(
                "recall:Q",
                title="Shifts found (% of injected shifts; higher is better)",
                scale=alt.Scale(domain=[0, 40]),
            ),
            color=colour,
        )
    )
    point = (
        alt.Chart(frame.filter(pl.col("signal") == "RO-NC"))
        .mark_point(filled=True, size=120)
        .encode(x="alarms:Q", y="recall:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(lines, point).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH,
        height=260,
        title=_title(
            text="Band position finds fewer injected shifts than RO-NC at the alarm rates tested",
            line="Odd-month events and alarms, 44 simulated worlds, six Primary substations. A shift is found when an alarm opens between its onset and its end. Each blue dot is one threshold.",
        ),
    )


def real_chart(*, real: dict[str, Any]) -> alt.VConcatChart:
    labels = {
        "K0": "K0 bands",
        "GAM_ONLY": "No correction (GAM only)",
        "HEAD_BASE": "Band quantile head",
        "HEAD_PIT": "Head + band position",
    }
    colours = {
        "K0 bands": ocf.DATA_BLUE,
        "No correction (GAM only)": ocf.DATA_PURPLE,
        "Band quantile head": ocf.BRAND_ORANGE,
        "Head + band position": ocf.DATA_GREEN,
    }
    panels = []
    for side, nominal in (("below_q01", 1.0), ("above_q99", 1.0)):
        data = [
            {
                "level": f"{labels[r['arm']]}: {r['group'].lower()}",
                "arm": labels[r["arm"]],
                "est": 100 * r[side],
            }
            for r in real["table"]
            if r["group"] in ("AFTER", "ORDINARY")
        ]
        order = [f"{labels[a]}: {g}" for a in labels for g in ("after", "ordinary")]
        y = alt.Y("level:N", sort=order, title=None)
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top"),
        )
        dots = (
            alt.Chart(pl.DataFrame(data))  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60, opacity=1)
            .encode(
                x=alt.X("est:Q", title="Readings outside the band (% of readings)"),
                y=y,
                color=colour,
            )
        )
        base = (
            alt.Chart(pl.DataFrame({"x": [nominal]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(base, dots).properties(
                width=WIDTH,
                height=20 * len(order),
                title=f"Reading {side.replace('_', ' ')} (nominal 1%)",
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="On the real record the band quantile head covers the lower tail better than K0 and the upper tail as well",
            line="Measured power against each forecast's 1% and 99% quantiles, six Primary substations, replicate and scored months. After: 48 hours after a logged transfer. Dashed line: nominal 1%. Closer is better.",
        )
    )


def main() -> None:
    report = json.loads((RESULTS_DIR / "report.json").read_text())
    band = json.loads((RESULTS_DIR / "bandpos.json").read_text())
    real = json.loads((RESULTS_DIR / "real.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_chart(
        rows=report["sweep"],
        cell="cost0.5 G5",
        domain=(-2.0, 10.0),
        x_title="Gain in curtailment cost over the band quantile head (% of its cost; positive is better)",
        title="The added-back-megawatts arm beats the band quantile head even when degraded in most ways tested",
        line="Curtailment cost against observed power, scaled limit 0.5, leads 12 to 36 hours, within 7 days of a shift. Only the shifts' own rows count here, so false alarms away from shifts do not show. Dot: estimate. Line: 95% interval from resampling whole shifts. Zero: same as the band quantile head.",
    ).save(OUT_DIR / "figure_1_break_even_curtailment_cost.svg")
    sweep_chart(
        rows=report["sweep"],
        cell="tw95 Gs",
        domain=(-12.0, 46.0),
        x_title="Gain in upper-tail score over the band quantile head (% of its score; positive is better)",
        title="On the upper-tail score near a shift, the arm tolerates less: up to a day of delay and 25% megawatt error",
        line="Threshold-weighted CRPS at the 95th percentile of earlier power against observed power, rows before the alarm, during, and 48 hours after a shift. Dot: estimate. Line: 95% interval from resampling whole shifts. Zero: same as the band quantile head.",
    ).save(OUT_DIR / "figure_2_break_even_upper_tail_shift_rows.svg")
    heads_chart(rows=report["heads"]).save(OUT_DIR / "figure_3_head_inputs.svg")
    detector_chart(band=band).save(OUT_DIR / "figure_4_band_position_detector.svg")
    real_chart(real=real).save(OUT_DIR / "figure_5_real_record_coverage.svg")


if __name__ == "__main__":
    main()
