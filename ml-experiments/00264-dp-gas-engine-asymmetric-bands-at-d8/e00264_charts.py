# ruff: noqa: E501
"""Draw the figures of experiment 264 from the score file.

uv run --no-sync python ml-experiments/00264-*/e00264_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00264_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8")
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
ARM_LABELS: Final[dict[str, str]] = {
    "arm2": "Arm 2: lower quantiles widened",
    "arm2b": "Arm 2b: both sides widened",
    "arm3": "Arm 3: mixture",
    "arm4": "Arm 4: arm 3 plus hindsight price",
    "arm3_forced": "Arm 3 with the switch forced on",
    "arm_emp": "Post hoc: residual quantiles",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "Arm 1: reference": ocf.DATA_SKY,
    "Arm 2b: both sides widened": ocf.BRAND_ORANGE,
    "Arm 3: mixture": ocf.DATA_BLUE,
    "Post hoc: residual quantiles": ocf.DATA_PURPLE,
}


def load_report() -> dict:
    """Return the score file written by `e00264_score.py`."""
    return json.loads((common.WORK_DIR / "score.json").read_text())


def forest(*, report: dict) -> None:
    """Figure 1: each arm's quantile score minus arm 1's, at D8 and at the two placebo sites."""
    rows = []
    for site in ("D8", "D10", "D16"):
        base = report[site]["arm1"]["qs"]
        for arm, label in ARM_LABELS.items():
            key = f"{arm}_minus_arm1"
            if key not in report[site]:
                continue
            c = report[site][key]
            rows.append(
                {
                    "site": f"{site} (arm 1 score {base:.2f} points)",
                    "label": label,
                    "diff": c["pct"],
                    "lo": c["low95"] / base * 100,
                    "hi": c["high95"] / base * 100,
                }
            )
    frame = pl.DataFrame(rows)
    sorted_sites = list(dict.fromkeys(row["site"] for row in rows))
    base_chart = alt.Chart(frame).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
        )
    )
    layer = (
        base_chart.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Quantile score minus arm 1's (% of arm 1; below 0 is better)"),
            x2="hi:Q",
        )
        + base_chart.mark_point(color=ocf.DATA_BLUE, filled=True, size=90).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        layer.properties(width=WIDTH - 310, height=150)
        .facet(
            row=alt.Row(
                "site:N",
                title=None,
                sort=sorted_sites,
                header=alt.Header(labelAnchor="start", labelFontSize=11),
            )
        )
        .properties(
            title=alt.TitleParams(
                "As-of widening lowers D8's quantile score by 9% to 19%; a forced-on mixture costs 3% at D16",
                subtitle=[
                    "Dot: estimate. Line: 95% interval from resampling the 12 months of the scored year.",
                    "Orange dashed line: same score as arm 1, experiment 191's reference forecast.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_score_against_reference.svg")


def monthly_coverage(*, report: dict) -> None:
    """Figure 2: q05-to-q95 coverage by month at D8 for arms 1, 2b, and 3."""
    names = {
        "arm1": "Arm 1: reference",
        "arm2b": "Arm 2b: both sides widened",
        "arm3": "Arm 3: mixture",
        "arm_emp": "Post hoc: residual quantiles",
    }
    rows = [
        {"month": f"{int(m) // 100}-{int(m) % 100:02d}", "arm": names[arm], "coverage": cov}
        for arm, by_month in report["D8"]["cov90_by_month"].items()
        for m, cov in by_month.items()
    ]
    frame = pl.DataFrame(rows)
    band = (
        alt.Chart(pl.DataFrame({"lo": [0.85], "hi": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN, opacity=0.25)
        .encode(y="lo:Q", y2="hi:Q")
    )
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month:O", title="Valid month (first row of the scored year is July 2025)"),
            y=alt.Y(
                "coverage:Q",
                title="Share of readings inside q05 to q95",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(orient="top", columns=1),
            ),
        )
    )
    chart = (band + lines).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            "D8's q05 to q95 band covers 42% of readings; widened as-of it covers 70% to 99% from August",
            subtitle=[
                "Green band: coverage within 5 points of the nominal 90%. July 2025 has no resolved residuals yet.",
                "Shortest-lead forecast of each valid time, outside hand-filtered events; D8, scored year.",
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    chart.save(DOCS_DIR / "figure_2_coverage_by_month.svg")


def tails(*, report: dict) -> None:
    """Figure 3: share of D8 readings below q01 and q05 and above q95 and q99, by arm."""
    arms = {
        "arm1": "Arm 1: reference",
        "arm2b": "Arm 2b: both sides widened",
        "arm3": "Arm 3: mixture",
    }
    nominal = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
    rows = [
        {
            "quantile": q,
            "arm": label,
            "share": report["D8"][arm][f"below_{q}"],
            "nominal": nominal[q],
        }
        for arm, label in arms.items()
        for q in nominal
    ]
    frame = pl.DataFrame(rows)
    ticks = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_tick(color="#333333", thickness=2, size=40)
        .encode(x=alt.X("quantile:N", sort=list(nominal)), y="nominal:Q")
    )
    points = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X("quantile:N", sort=list(nominal), title="Quantile"),
            y=alt.Y(
                "share:Q",
                title="Share of readings below the quantile",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(orient="top", columns=1),
            ),
            xOffset="arm:N",
        )
    )
    chart = (ticks + points).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            "Widening brings D8's tail shares near nominal but not onto it",
            subtitle=[
                "Black tick: nominal share. Dot: observed share, D8, scored year, all leads.",
                "Both tails of arm 1 miss: 31% of readings fall below q05 and 27% rise above q95.",
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    chart.save(DOCS_DIR / "figure_3_tail_shares.svg")


def main() -> None:
    """Draw all three figures."""
    report = load_report()
    forest(report=report)
    monthly_coverage(report=report)
    tails(report=report)


if __name__ == "__main__":
    main()
