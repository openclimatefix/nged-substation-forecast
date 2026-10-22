"""Draw the figures of experiment 205's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00205-*/e00205_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00205_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
GAPS: Final[tuple[str, ...]] = ("3", "6", "24", "72")
GAP_AXIS: Final[alt.Axis] = alt.Axis(labelAngle=0)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _zero() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )


def figure_headline(*, report: dict) -> alt.TopLevelMixin:
    """Each rule's score minus the no-gap score, by gap length (gap ends at the cutoff)."""
    rows = []
    for gap in GAPS:
        h = report["headline"][gap]
        for key, label in (
            ("a_minus_n", "Coast (a)"),
            ("b_minus_n", "Coast, wider bands (b)"),
            ("c2_minus_n", "Drop filter, keep offset and R (c2)"),
            ("c_minus_n", "Frozen model alone (c)"),
        ):
            c = h[key]
            rows.append(
                {
                    "gap": f"{gap} h",
                    "rule": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    colours = {
        "Coast (a)": ocf.BRAND_ORANGE,
        "Coast, wider bands (b)": ocf.DATA_PURPLE,
        "Drop filter, keep offset and R (c2)": ocf.DATA_BLUE,
        "Frozen model alone (c)": GREY,
    }
    base = alt.Chart(pl.DataFrame(rows)).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("gap:N", sort=[f"{g} h" for g in GAPS], title="Gap length", axis=GAP_AXIS),
        xOffset=alt.XOffset("rule:N", sort=list(colours)),
        color=alt.Color(
            "rule:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320),
        ),
    )
    line = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title="Quantile score minus the no-gap score (points)"), y2="high:Q"
    )
    dot = base.mark_point(filled=True, size=60).encode(y="estimate:Q")  # ty: ignore[unresolved-attribute]
    return (_zero() + line + dot).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="Coasting costs 0.01 to 0.21 points as the gap grows from 3 to 72 hours",
            subtitle=(
                "Dot: pooled quantile score of the rule minus the no-gap score, over leads 0.5 to "
                "168 hours, 56 event runs at each of 5 substations, gap ending at the data "
                "cutoff. Line: 99.4% interval from resampling calendar months. Lower is better."
            ),
        ),
    )


def figure_contrast(*, report: dict) -> alt.TopLevelMixin:
    """Coast minus the two frozen alternatives, by gap length."""
    rows = []
    for gap in GAPS:
        h = report["headline"][gap]
        for key, label in (
            ("C_a_minus_c2", "Coast minus drop-the-filter (c2)"),
            ("a_minus_c", "Coast minus frozen model alone (c)"),
        ):
            c = h[key]
            rows.append(
                {
                    "gap": f"{gap} h",
                    "contrast": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    colours = {
        "Coast minus drop-the-filter (c2)": ocf.BRAND_ORANGE,
        "Coast minus frozen model alone (c)": ocf.DATA_BLUE,
    }
    base = alt.Chart(pl.DataFrame(rows)).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("gap:N", sort=[f"{g} h" for g in GAPS], title="Gap length", axis=GAP_AXIS),
        xOffset=alt.XOffset("contrast:N", sort=list(colours)),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=1),
        ),
    )
    line = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "low:Q", title="Difference in quantile score (points; below zero favours coasting)"
        ),
        y2="high:Q",
    )
    dot = base.mark_point(filled=True, size=60).encode(y="estimate:Q")  # ty: ignore[unresolved-attribute]
    return (_zero() + line + dot).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="Coasting beats the frozen model alone at every gap length tested",
            subtitle=(
                "Against drop-the-filter (c2) the interval spans zero at every length, and the "
                "point estimate crosses zero between 24 and 72 hours. Line: 99.4% interval from "
                "resampling calendar months."
            ),
        ),
    )


def figure_recovery(*, report: dict) -> alt.TopLevelMixin:
    """Score of coasting minus the no-gap score against hours between the gap's end and cutoff."""
    rows = []
    for gap in GAPS:
        for k, value in report["recovery"][gap]["all"]["a_score_by_distance"].items():
            rows.append(
                {
                    "gap": f"{gap}-hour gap",
                    "k": int(k),
                    "difference": value - report["recovery"][gap]["all"]["no_gap"],
                }
            )
    colours = [ocf.DATA_GREEN, ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
    domain = [f"{g}-hour gap" for g in GAPS]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="Hours between the end of the gap and the cutoff"),
            y=alt.Y("difference:Q", title="Quantile score minus the no-gap score (points)"),
            color=alt.Color(
                "gap:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top", columns=4),
            ),
        )
    )
    margin = (
        alt.Chart(pl.DataFrame({"y": [0.02]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return (_zero() + margin + chart).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="Recovery after a 24-hour gap is slow and uneven",
            subtitle=(
                "Rule (a), all leads. Dotted line: the 0.02-point recovery margin. The score is "
                "still above the margin 48 hours after a 24-hour gap."
            ),
        ),
    )


def figure_coverage(*, report: dict) -> alt.TopLevelMixin:
    """Coverage of q05 and q95 by gap length for rules a and b."""
    rows = []
    for gap in GAPS:
        for q in ("q05", "q95"):
            for rule, label in (("a", "Coast (a)"), ("b", "Coast, wider bands (b)")):
                rows.append(
                    {
                        "gap": f"{gap} h",
                        "quantile": q,
                        "rule": label,
                        "coverage": report["scores"][gap]["coverage"][q][rule],
                    }
                )
    colours = {"Coast (a)": ocf.BRAND_ORANGE, "Coast, wider bands (b)": ocf.DATA_PURPLE}
    panels = []
    for q, level in (("q05", 0.05), ("q95", 0.95)):
        points = (
            alt.Chart(pl.DataFrame(rows).filter(pl.col("quantile") == q))  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("gap:N", sort=[f"{g} h" for g in GAPS], title="Gap length", axis=GAP_AXIS),
                y=alt.Y(
                    "coverage:Q",
                    title=f"Share of actuals below {q}",
                    scale=alt.Scale(domain=[level - 0.04, level + 0.04]),
                ),
                color=alt.Color(
                    "rule:N",
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        line = (
            alt.Chart(pl.DataFrame({"y": [level]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[level - 0.04, level + 0.04])))
        )
        panels.append((line + points).properties(width=250, height=220))
    return alt.hconcat(*panels).properties(
        title=_title(
            text="Wider bands bring the tail coverage back towards the nominal 5% and 95%",
            subtitle=(
                "Share of actuals below the quantile, over leads 0.5 to 168 hours, gap ending at "
                "the cutoff. Dotted line: nominal."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_headline(report=report), name="figure_1_headline")
    _save(figure_contrast(report=report), name="figure_2_contrast")
    _save(figure_recovery(report=report), name="figure_3_recovery")
    _save(figure_coverage(report=report), name="figure_4_coverage")


if __name__ == "__main__":
    main()
