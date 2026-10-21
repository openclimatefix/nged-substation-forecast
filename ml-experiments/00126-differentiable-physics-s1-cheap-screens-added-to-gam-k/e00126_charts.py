"""Draw the figures of experiment 126 as SVG into the docs folder, from the report files.

The numbers come from `work/numbers.json` (`e00126_report.py`), `work/gate3.json`
(`e00126_gate3.py`), and `work/skf_gate_<cut>.parquet`, so a figure and the page quote the same
values. Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00126-*/e00126_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00126_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
ARM_LABELS: Final[dict[str, str]] = {
    "a_mono": "X1 A-mono: monotone curve",
    "a_free": "X2 A-free: free curve",
    "s_ewma": "X3 S-ewma: smoothed temp.",
    "s_mm": "X4 S-mm: daily extremes",
    "c_wind": "X5 C-wind: wind chill",
    "c_hw": "X6 C-hw: apparent temperature",
    "d_sch": "X7 D-sch: school holidays",
    "e_sol": "X8 E-sol: solar elevation",
    "f_mix": "X9 F-mix: regime noise",
    "g_skf": "X10 G-skf: switching filter",
    "w_basis": "X11 W-basis: learned cycle",
    "w_mix": "X12 W-mix: cold/mild slopes",
    "w_stn": "X13 W-stn: station gap",
}
LABEL_COLOURS: Final[dict[str, str]] = {
    "worse": ocf.BRAND_ORANGE,
    "undecidable": "#8A8A8A",
    "tie": ocf.DATA_BLUE,
    "small gain": ocf.DATA_GREEN,
    "gain": ocf.DATA_GREEN,
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", fontSize=14
    )


def headline() -> alt.LayerChart:
    """Return the forest chart of X1 to X13: estimate and interval on each arm's primary rows."""
    rows = json.loads((common.WORK_DIR / "numbers.json").read_text())
    frame = pl.DataFrame(
        [
            {
                "arm": ARM_LABELS[r["arm"]],
                "estimate": r["scored"],
                "low": r["low"],
                "high": r["high"],
                "label": r["label before gates"],
            }
            for r in rows
        ]
    )
    order = list(frame["arm"])
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(LABEL_COLOURS), range=list(LABEL_COLOURS.values())),
            legend=alt.Legend(title="Label before gates", orient="top"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("low:Q", title="Arm minus T-all (points of capacity; more negative is better)"),
        x2="high:Q",
    )
    dot = base.mark_point(filled=True, size=70).encode(x="estimate:Q")
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    return (
        alt.layer(zero, rule, dot)
        .properties(
            width=WIDTH - 230,
            height=26 * len(order),
            title=_title(
                text="No cheap addition lowered the day-ahead winter score",
                subtitle=(
                    "Scored year 2025-07 to 2026-06, five decision substations, quantile score "
                    "on each arm's primary rows after the Kalman filter. Dot: estimate. Line: "
                    "interval from resampling whole calendar months (99.44% for X1 to X9, "
                    "98.75% for X10 to X13). Dashed line: same as T-all-K."
                ),
            ),
        )
        .configure_view(stroke=None)
    )


def gate3() -> alt.LayerChart:
    """Return the injected-effect gate: what each arm recovers of an oracle gain of 0.03 points."""
    report = json.loads((common.WORK_DIR / "gate3.json").read_text())
    rows = [
        {
            "arm": ARM_LABELS[a],
            "estimate": v["mean_difference"],
            "low": v["low"],
            "high": v["high"],
            "powered": v["powered"],
        }
        for a, v in report.items()
        if "mean_difference" in v
    ]
    frame = pl.DataFrame(rows)
    order = list(frame["arm"])
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None),
        color=alt.Color(
            "powered:N",
            scale=alt.Scale(domain=[True, False], range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE]),
            legend=alt.Legend(
                title="Gate passed (recovers at least half of the oracle gain)", orient="top"
            ),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "low:Q", title="Arm minus T-all on injected data (points; more negative is better)"
        ),
        x2="high:Q",
    )
    dot = base.mark_point(filled=True, size=70).encode(x="estimate:Q")
    target = alt.Chart(pl.DataFrame({"x": [-0.015]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    return alt.layer(target, rule, dot).properties(
        width=WIDTH - 230,
        height=26 * len(order),
        title=_title(
            text="Five arms can show half of a 0.03-point effect",
            subtitle=(
                "Synthetic substations on the tuning year, 100 draws per arm. An effect sized "
                "to an oracle gain of 0.03 points is injected into T-all's real fit. Dashed "
                "line: half the oracle gain, the pass line. Line: 99.44% interval over draws."
            ),
        ),
    )


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline().save(DOCS_DIR / "figure_1_contrasts.svg")
    gate3().save(DOCS_DIR / "figure_2_gate3.svg")


if __name__ == "__main__":
    main()
