"""Draw the figures of experiment 135 from `work/results.json` and `work/controls.json`.

uv run --no-sync python ml-experiments/00135-*/e00135_figures.py
"""

import json
import statistics
from pathlib import Path
from typing import Final

import altair as alt
import e00135_common as common
import e00135_report as report
import plotting.ocf_theme  # noqa: F401  (registers the OCF Altair theme)
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00135-differentiable-physics-n10-joint-kalman-and-demand-fit"
)
BLUE: Final[str] = "#306BFF"
ORANGE: Final[str] = "#FF4901"
LEAD_BANDS: Final[list[str]] = ["0-24 h", "12-36 h", "24-72 h (days 2-3)", "72-168 h (days 4-7)"]
PERIOD_LABELS: Final[dict[str, str]] = {
    "scored": "Scored year (2025-07 to 2026-06)",
    "replicate": "Replicate months (even months, 2024-07 to 2025-06)",
}


def _contrast_frame(*, results: dict, contrast: str) -> pl.DataFrame:
    rows = [
        {
            "period": PERIOD_LABELS[period],
            "lead": band,
            "estimate": c[contrast]["estimate"],
            "lower": c[contrast]["lower"],
            "upper": c[contrast]["upper"],
        }
        for period in PERIOD_LABELS
        for band, c in results[period]["contrasts"].items()
        if band in LEAD_BANDS
    ]
    return pl.DataFrame(rows)


def _interval_chart(
    *, frame: pl.DataFrame, title: list[str], x_title: str, colour: str
) -> alt.Chart:
    base = alt.Chart(frame).encode(
        y=alt.Y("lead:N", sort=LEAD_BANDS, title=None),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_LABELS.values())),
    )
    rule = base.mark_rule(strokeWidth=2, color=colour).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q"
    )
    dot = base.mark_point(filled=True, size=70, color=colour).encode(  # ty: ignore[unresolved-attribute]
        x="estimate:Q",
        opacity=alt.Opacity(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_LABELS.values()), range=[1.0, 0.45]),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + rule + dot).properties(width=CONTENT_WIDTH_PX - 150, height=170, title=title)


def headline(*, results: dict) -> alt.VConcatChart:
    """Return the two contrasts' intervals by lead band, one panel per contrast."""
    c1 = _interval_chart(
        frame=_contrast_frame(results=results, contrast="C1"),
        title=[
            "C1: joint fit with no tie (J1) minus two-stage fit (TS).",
            "Positive is worse for J1. Dot: estimate. Line: 97.5% interval from resampling months.",
        ],
        x_title="Quantile score, J1 minus TS (points of capacity; more negative is better)",
        colour=ORANGE,
    )
    c2 = _interval_chart(
        frame=_contrast_frame(results=results, contrast="C2"),
        title=[
            "C2: joint fit tied to the two-stage shapes (J2) minus two-stage fit (TS).",
            "Zero (dashed) is no difference. Same five decision substations, same rows.",
        ],
        x_title="Quantile score, J2 minus TS (points of capacity; more negative is better)",
        colour=BLUE,
    )
    return alt.vconcat(c1, c2).resolve_scale(x="independent")


def site_labels() -> dict[str, str]:
    """Return each decision substation's series identifier (as text) mapped to its D-label."""
    frame = report.load_common(periods={"scored"}, sites=common.DECISION_SITES)["ts"]
    pairs = frame.select("time_series_id", "site").unique()
    return {str(i): label for i, label in zip(pairs["time_series_id"], pairs["site"], strict=True)}


def per_site(*, results: dict) -> alt.Chart:
    """Return C2 at 12 to 36 hours per substation, scored year and replicate months."""
    labels = site_labels()
    rows = [
        {"site": labels[str(site)], "period": PERIOD_LABELS[period], "difference": value}
        for period in PERIOD_LABELS
        for site, value in results[period]["contrasts"]["12-36 h"]["C2"]["site_differences"].items()
    ]
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(y=alt.Y("site:N", title=None, sort=list(common.DECISION_SITES)))
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    dots = base.mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Quantile score, J2 minus TS (points of capacity; more negative is better)",
        ),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_LABELS.values()), range=[BLUE, "#10C5F7"]),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
        ),
    )
    return (zero + dots).properties(
        width=CONTENT_WIDTH_PX - 120,
        height=150,
        title=[
            "C2 at 12 to 36 hours for each decision substation.",
            "Site labels are the anonymised D-labels.",
        ],
    )


def shapes(*, controls: dict, arm: str) -> alt.Chart:
    """Return the median share of each term's curve that the joint fit moves, J1 beside J2."""
    rows = []
    for label, key in (("J1 (no tie)", "j1"), ("J2 (ridge weight 3)", arm)):
        for term in ("profile", "tent", "harmonics", "heating", "holiday", "lighting"):
            values = [
                cut[term]["share_of_reference"]
                for site in controls["shape_distance"].values()
                for cut in site[key]
            ]
            rows.append({"arm": label, "term": term, "share": 100 * statistics.median(values)})
    frame = pl.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "term:N",
                title=None,
                sort=["profile", "tent", "harmonics", "heating", "holiday", "lighting"],
            ),
            x=alt.X(
                "share:Q",
                title=[
                    "Root-mean-square difference from TS's curve",
                    "(% of TS's curve; median over cuts and substations)",
                ],
                scale=alt.Scale(type="symlog", domain=[0, 500]),
                axis=alt.Axis(values=[0, 10, 30, 100, 300]),
            ),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["J1 (no tie)", "J2 (ridge weight 3)"], range=[ORANGE, BLUE]
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 100,
            height=220,
            title=[
                "How far the joint fits move the fitted curves from the two-stage curves.",
                "Symmetric log axis.",
            ],
        )
    )


def main() -> None:
    """Write the SVG figures beside the write-up."""
    results = json.loads((common.WORK_DIR / "results.json").read_text())
    controls = json.loads((common.WORK_DIR / "controls.json").read_text())
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    headline(results=results).save(FIGURE_DIR / "headline_contrasts.svg")
    per_site(results=results).save(FIGURE_DIR / "c2_per_site.svg")
    shapes(controls=controls, arm="j2_ridge_3").save(FIGURE_DIR / "shape_distance.svg")


if __name__ == "__main__":
    main()
