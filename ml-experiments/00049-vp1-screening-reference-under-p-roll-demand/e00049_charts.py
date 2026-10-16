"""Draw experiment 00049's figures from the saved results, as SVG into the docs folder.

1. The six pre-registered contrasts, each with its 99.17% interval.
2. The quantile score against the spread factor for three retrained arms.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00049-vp1-screening-reference-under-p-roll-demand/e00049_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00049_stage0 as stage0
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00049-vp1-screening-reference-under-p-roll-demand"
)
STAGE1_DIR: Final[Path] = stage0.WORK_DIR.parent / "stage1"
PLOT_WIDTH: Final[int] = 380
WRAP_CHARACTERS: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "D0": "D0 chosen spread factor, minus 1.15",
    "B1": "B1 five-model blend, minus best model",
    "C1": "C1 no Christmas column",
    "C2": "C2 no COVID indicator",
    "C3": "C3 half-life 2 years, not 0.5",
    "F1": "F1 fingerprints in pooled model",
}
SCOPE: Final[str] = (
    "16 Primary substations, 73 runs started 5 days apart, July 2025 to June 2026, seed 0, "
    "spread factor included."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The six pre-registered contrasts with their intervals."""
    stage0_rows = pl.read_parquet(stage0.WORK_DIR / "contrasts.parquet").filter(
        pl.col("family").is_in(["D0", "B1"])
    )
    stage1_rows = pl.read_parquet(STAGE1_DIR / "contrasts.parquet").filter(
        pl.col("id").is_in(["C1", "C2", "C3", "F1"])
    )
    data = pl.concat(
        [
            stage0_rows.select(id=pl.col("family"), d="difference", lo="lower", hi="upper"),
            stage1_rows.select("id", d="difference", lo="lower", hi="upper"),
        ]
    ).with_columns(label=pl.col("id").replace_strict(LABELS))
    order = [LABELS[k] for k in LABELS]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=11))
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo:Q",
            title="Score difference (points of capacity; more negative is better)",
        ),
        x2="hi:Q",
    )
    dots = base.mark_point(color=ocf.BRAND_ORANGE, filled=True, size=70, aria=False).encode(x="d:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(rule, bars, dots)
        .properties(
            width=PLOT_WIDTH,
            height=190,
            title=_title(
                text="Only the model blend resolves a gain",
                lines=[
                    "Dot: estimate. Line: 99.17% interval from resampling whole months. "
                    "Dashed rule: no difference. "
                    "C1 to C3: against P-roll. F1: against pooled P-roll. " + SCOPE
                ],
            ),
        )
        .configure_view(stroke=None)
    )


def figure_2() -> alt.TopLevelMixin:
    """The quantile score against the spread factor for the three retrained arms."""
    curve = pl.read_parquet(stage0.WORK_DIR / "curve.parquet").with_columns(
        arm_label=pl.col("arm").replace_strict(
            {
                "monthly": "Monthly, all history",
                "weekly": "Weekly, all history",
                "roll12": "Monthly, latest 12 months (P-roll)",
            }
        )
    )
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
    lines = (
        alt.Chart(curve)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("factor:Q", title="Spread factor", scale=alt.Scale(domain=[1.0, 1.3])),
            y=alt.Y(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(range=colours),
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [stage0.FIXED_FACTOR]}))
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(lines, rule)
        .properties(
            width=PLOT_WIDTH,
            height=190,
            title=_title(
                text="The score is flat across spread factors from 1.05 to 1.15",
                lines=["Dashed rule: the fixed factor 1.15. No 14-day correction. " + SCOPE],
            ),
        )
        .configure_view(stroke=None)
    )


def main() -> None:
    """Write both figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(str(DOCS_DIR / f"{name}.svg"))
        print("wrote", DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
