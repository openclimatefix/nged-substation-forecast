"""Draw experiment 99's figures from the saved results, as SVG.

Run `e00099_report.py` first. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00099-*/e00099_charts.py
"""

import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import altair as alt
import e00099_common as common
import e00099_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 480
WRAP: Final[int] = 100
ARM_LABELS: Final[dict[str, str]] = {
    "c1": "c1 CRPS loss, 6 members",
    "p3": "p3 Pinball (custom), 6 members",
    "p2": "p2 Pinball (built in), 6 members",
    "p1": "p1 Pinball, control member only",
    "bg2_weekly": "Best guess (weekly refit, 500 rounds)",
    "manual_heuristic": "NGED manual heuristic",
}
ARM_ORDER: Final[list[str]] = list(ARM_LABELS.values())
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE_LIGHT,
]
CONTRAST_ORDER: Final[list[str]] = [
    "C1 c1 - p3 (loss)",
    "C5 c1 - p3m (loss, same pooling)",
    "C2 p2 - p1 (6 members)",
    "C3 p3 - p2 (optimiser)",
    "C4 c1 - p1 (whole change)",
    "c1 - bg2_weekly (reference)",
]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 c1 - p3 (loss)": "C1 CRPS minus pinball (custom), the verdict",
    "C5 c1 - p3m (loss, same pooling)": "C5 as C1, pinball pooled with the CRPS masses",
    "C2 p2 - p1 (6 members)": "C2 six members minus control member (pinball)",
    "C3 p3 - p2 (optimiser)": "C3 custom minus built-in optimiser (pinball)",
    "C4 c1 - p1 (whole change)": "C4 CRPS, 6 members, minus pinball, control only",
    "c1 - bg2_weekly (reference)": "Reference: c1 minus best guess (schedule differs)",
}
RESULTS: Final[Path] = report.RESULTS_DIR


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _colour(*, domain: Sequence[str], range_: Sequence[str]) -> alt.Color:
    return alt.Color("label:N", scale=alt.Scale(domain=domain, range=range_), legend=None)


def figure_headline() -> alt.TopLevelMixin:
    """Absolute quantile score per arm with its 95% interval."""
    table = (
        pl.read_csv(RESULTS / "absolute_qs.csv")
        .filter(pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(table).encode(
        y=alt.Y("label:N", sort=ARM_ORDER, title=None, axis=alt.Axis(labelLimit=260)),
        color=_colour(domain=ARM_ORDER, range_=ARM_COLOURS),
    )
    title = "Quantile score (points of capacity; smaller is better)"
    interval = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower_95:Q", title=title, scale=alt.Scale(zero=False)), x2="upper_95:Q"
    )
    dots = base.mark_point(filled=True, size=70, aria=False).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(interval, dots).properties(
        width=WIDTH,
        height=190,
        title=_title(
            text="Absolute quantile score, 16 Primary substations, July 2025 to June 2026",
            lines=[
                (
                    "Dot: mean over 292 daily runs. Line: 95% interval from resampling whole "
                    "months. "
                    "The arms are close; the paired differences in the next figure are much "
                    "narrower than these intervals because every arm's error rises and falls "
                    "together from month to month."
                )
            ],
        ),
    )


def figure_contrasts() -> alt.TopLevelMixin:
    """Paired quantile-score differences with 95% intervals."""
    table = (
        pl.read_csv(RESULTS / "contrasts.csv")
        .filter((pl.col("metric") == "qs") & pl.col("contrast").is_in(CONTRAST_ORDER))
        .with_columns(label=pl.col("contrast").replace_strict(CONTRAST_LABELS), zero=pl.lit(0.0))
    )
    order = [CONTRAST_LABELS[c] for c in CONTRAST_ORDER]
    colours = [ocf.BRAND_ORANGE, ocf.BRAND_ORANGE_LIGHT, ocf.DATA_BLUE, ocf.DATA_SKY]
    colours += [ocf.DATA_PURPLE, ocf.DATA_GREEN]
    base = alt.Chart(table).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=_colour(domain=order, range_=colours),
    )
    title = "Quantile score difference (points of capacity; below 0 is better)"
    interval = base.mark_rule().encode(x=alt.X("lower_95:Q", title=title), x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=70, aria=False).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(table).mark_rule(color="black").encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-report.VERDICT_THRESHOLD]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, threshold, interval, dots).properties(
        width=WIDTH - 90,
        height=200,
        title=_title(
            text="Paired differences between arms, same 292 runs and 16 substations",
            lines=[
                (
                    "Dot: mean difference, first arm minus second. Line: 95% interval from "
                    "resampling whole months. Solid rule: no difference. Dashed rule: the "
                    "pre-registered gain of 0.05 points that the verdict needs."
                )
            ],
        ),
    )


def figure_lead_bands() -> alt.TopLevelMixin:
    """Quantile score by lead band for four arms."""
    arms = {k: v for k, v in ARM_LABELS.items() if k in ("c1", "p3", "p1", "bg2_weekly")}
    table = (
        pl.read_csv(RESULTS / "by_lead_band.csv")
        .filter(pl.col("arm").is_in(list(arms)))
        .with_columns(label=pl.col("arm").replace_strict(arms))
    )
    domain = list(arms.values())
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN]
    return (
        alt.Chart(table)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band:N",
                sort=[b[0] for b in report.LEAD_BANDS],
                title="Lead time",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "value:Q",
                title="Quantile score (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH + 100,
            height=220,
            title=_title(
                text="Quantile score by lead time",
                lines=[
                    (
                        "Mean quantile score of 16 Primary substations over 292 daily runs; "
                        "smaller is better. The best guess refits weekly with 500 rounds, the "
                        "other arms monthly with 150."
                    )
                ],
            ),
        )
    )


def figure_tails() -> alt.TopLevelMixin:
    """Coverage minus nominal at the four tail levels, 12 to 36 hour lead."""
    arms = {k: v for k, v in ARM_LABELS.items() if k in ("c1", "p3", "p1", "bg2_weekly")}
    table = (
        pl.read_csv(RESULTS / "tails_12-36_h.csv")
        .filter(pl.col("arm").is_in(list(arms)))
        .with_columns(
            label=pl.col("arm").replace_strict(arms),
            panel=pl.col("level").map_elements(
                lambda p: f"q{round(p * 100):02d} (nominal {p})", return_dtype=pl.String
            ),
            gap_low=pl.col("coverage_lower_95") - pl.col("level"),
            gap_high=pl.col("coverage_upper_95") - pl.col("level"),
            zero=pl.lit(0.0),
        )
    )
    domain = list(arms.values())
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN]
    base = alt.Chart().encode(
        y=alt.Y("label:N", sort=domain, title=None, axis=alt.Axis(labelLimit=300)),
        color=_colour(domain=domain, range_=colours),
    )
    title = "Coverage minus nominal level (0 is perfect)"
    interval = base.mark_rule().encode(x=alt.X("gap_low:Q", title=title), x2="gap_high:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=60, aria=False).encode(x=alt.X("gap:Q", title=title))  # ty: ignore[unresolved-attribute]
    zero = alt.Chart().mark_rule(color="black").encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(zero, interval, dots)
        .properties(width=WIDTH, height=90)
        .facet(
            row=alt.Row(
                "panel:N",
                sort=[
                    "q1 (nominal 0.01)",
                    "q5 (nominal 0.05)",
                    "q95 (nominal 0.95)",
                    "q99 (nominal 0.99)",
                ],
                title=None,
                header=alt.Header(labelAnchor="start", labelFontSize=12),
            ),
            data=table,
        )
        .properties(
            title=_title(
                text="Tail coverage at 12 to 36 hours lead",
                lines=[
                    (
                        "Dot: share of measurements at or below the quantile, minus its nominal "
                        "level. Line: 95% interval from resampling whole months. 16 Primary "
                        "substations, 292 daily runs."
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write the SVGs next to the page."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_absolute_score": figure_headline(),
        "figure_2_paired_differences": figure_contrasts(),
        "figure_3_lead_bands": figure_lead_bands(),
        "figure_4_tail_coverage": figure_tails(),
    }
    for name, chart in figures.items():
        chart.save(DOCS_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
