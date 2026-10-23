# ruff: noqa: E501
"""Draw the figures of experiment 265 from the score file.

uv run --no-sync python ml-experiments/00265-*/e00265_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00265_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path(
    "docs/ml-experiments/00265-dp-gas-engine-bands-prior-and-visibility-switch"
)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
QUARTERS: Final[tuple[str, ...]] = ("2025Q3", "2025Q4", "2026Q1", "2026Q2")
LABELS: Final[dict[str, str]] = {
    "arm1": "Unwidened reference",
    "control": "Control: own history only",
    "arm2": "Arm 2: outside prior (K = 2,000)",
    "arm3": "Arm 3: prior and visibility switch",
    "arm2p": "Post hoc 2p: faster shrinkage, light pool",
    "arm3p": "Post hoc 3p: 2p and switch",
}
COLOURS: Final[dict[str, str]] = {
    "Unwidened reference": ocf.DATA_SKY,
    "Control: own history only": ocf.BRAND_ORANGE,
    "Arm 2: outside prior (K = 2,000)": ocf.DATA_BLUE,
    "Arm 3: prior and visibility switch": ocf.DATA_PURPLE,
    "Post hoc 2p: faster shrinkage, light pool": ocf.DATA_GREEN,
    "Post hoc 3p: 2p and switch": ocf.DATA_AMBER,
}


def load_report() -> dict:
    """Return the score file written by `e00265_score.py`."""
    return json.loads((common.WORK_DIR / "score.json").read_text())


def coverage_by_quarter(*, report: dict) -> None:
    """Figure 1: D8's q05-to-q95 coverage by quarter for each arm."""
    rows = [
        {"quarter": q, "arm": LABELS[a], "coverage": report["sites"]["D8"][a][f"cov90_{q}"]}
        for a in LABELS
        for q in QUARTERS
    ]
    frame = pl.DataFrame(rows)
    band = (
        alt.Chart(pl.DataFrame({"lo": [0.85], "hi": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN, opacity=0.25)
        .encode(y="lo:Q", y2="hi:Q")
    )
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("quarter:N", sort=list(QUARTERS), title=None),
            y=alt.Y(
                "coverage:Q",
                title="Share of readings inside q05 to q95",
                scale=alt.Scale(domain=[0.2, 1.0]),
            ),
            color=alt.Color(
                "arm:N",
                sort=list(COLOURS),
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="bottom", columns=2),
            ),
        )
    )
    chart = (band + lines).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "No arm holds D8's coverage inside the green band in every quarter",
            subtitle=[
                "Green band: nominal 0.90 plus or minus 5 points. 2025Q3 includes July, which has no residual history."
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    chart.save(DOCS_DIR / "figure_1_d8_coverage_by_quarter.svg")


def forest(*, report: dict) -> None:
    """Figure 2: each arm's quantile score minus the control's at every Primary."""
    rows = []
    for site in common.SITES:
        base = report["sites"][site]["control"]["qs"]
        for arm in ("arm2", "arm3", "arm2p", "arm3p"):
            c = report["sites"][site][f"{arm}_minus_control"]
            rows.append(
                {
                    "site": f"{site} (control {base:.2f} points)",
                    "label": LABELS[arm],
                    "diff": c["pct"],
                    "lo": c["pct_low95"],
                    "hi": c["pct_high95"],
                }
            )
    frame = pl.DataFrame(rows)
    sites = list(dict.fromkeys(r["site"] for r in rows))
    base_chart = alt.Chart(frame).encode(
        y=alt.Y(
            "label:N",
            sort=[LABELS[a] for a in ("arm2", "arm3", "arm2p", "arm3p")],
            title=None,
            axis=alt.Axis(labelLimit=300),
        )
    )
    layer = (
        base_chart.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q", title="Quantile score minus the control's (% of control; below 0 is better)"
            ),
            x2="hi:Q",
        )
        + base_chart.mark_point(color=ocf.DATA_BLUE, filled=True, size=80).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        layer.properties(width=WIDTH - 320, height=110)
        .facet(
            row=alt.Row(
                "site:N",
                title=None,
                sort=sites,
                header=alt.Header(labelAnchor="start", labelFontSize=11),
            )
        )
        .properties(
            title=alt.TitleParams(
                "The outside prior costs the Primaries without an engine 1% to 11% in quantile score",
                subtitle=[
                    "Dot: estimate. Line: 95% interval from resampling the 12 months of the scored year.",
                    "Orange dashed line: the same score as the control.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            )
        )
    )
    chart.save(DOCS_DIR / "figure_2_score_against_control.svg")


def census(*, report: dict) -> None:
    """Figure 3: the reference's share of readings below q05, by quarter, at each Primary."""
    rows = [
        {"site": s, "quarter": q, "share": report["census"][s][q]["below_q05"]}
        for s in common.SITES
        for q in QUARTERS
    ]
    frame = pl.DataFrame(rows)
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("quarter:N", sort=list(QUARTERS), title=None),
            y=alt.Y("share:Q", title="Share of readings below the reference's q05"),
            color=alt.Color(
                "site:N",
                sort=list(common.SITES),
                legend=alt.Legend(title=None, orient="bottom", columns=6),
            ),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [0.05, 0.15]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ocf.BRAND_ORANGE)
        .encode(y="y:Q")
    )
    chart = (lines + rules).properties(
        width=WIDTH,
        height=240,
        title=alt.TitleParams(
            "Only D8 has a heavy lower tail: its share below q05 is 0.26 to 0.38 in every quarter",
            subtitle=[
                "Lower dashed line: nominal 0.05. Upper dashed line: the 0.15 'heavy' threshold, needed in two quarters."
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    chart.save(DOCS_DIR / "figure_3_census_lower_tail.svg")


def main() -> None:
    """Draw every figure."""
    report = load_report()
    coverage_by_quarter(report=report)
    forest(report=report)
    census(report=report)


if __name__ == "__main__":
    main()
