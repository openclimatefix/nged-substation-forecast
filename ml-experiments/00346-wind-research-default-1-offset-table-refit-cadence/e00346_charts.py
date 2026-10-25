"""Draw experiment 00346's figures as SVG. Reads `work/score.json` and `work/final/`.

Figure 1: each arm's kept-row quantile score minus the default's, in the scored year and in the
out-of-time months. Figure 2: the same difference month by month. Figure 3: the offsets that the
trailing 3-month table (x3) fitted on the first day of each month, per farm.
Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00346-*/e00346_charts.py   (PYTHONPATH: see common)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00346_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00346-wind-research-default-1-offset-table-refit-cadence"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
GREY: Final[str] = "#777777"
LEVEL: Final[str] = "0.9967"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
ARM_LABELS: Final[dict[str, str]] = {
    "x2": "x2: monthly, trailing 6 months",
    "x1": "x1: monthly, expanding window",
    "x3": "x3: monthly, trailing 3 months",
    "refq": "reference: quarterly, expanding window",
    "plc": "placebo: table of a random earlier month",
}
KINDS: Final[dict[str, str]] = {
    "x2": "arm",
    "x1": "arm",
    "x3": "arm",
    "refq": "reference",
    "plc": "placebo",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each arm minus the default: scored year and out-of-time months, kept rows."""
    panels = []
    for key, text, arms in (
        ("scored", "Scored year (53 Tuesday runs, 2025-07 to 2026-06)", list(ARM_LABELS)),
        (
            "out_of_time",
            "Out-of-time months (Tuesday runs, 2026-07 to 2026-10)",
            ["x2", "x1", "x3", "refq"],
        ),
    ):
        rows = []
        for arm in arms:
            found = score[key]["contrasts"][arm]["kept | qs"]
            low, high = found[LEVEL]
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
                    "estimate": found["mean"],
                    "lower": low,
                    "upper": high,
                    "kind": KINDS[arm],
                }
            )
        data = pl.DataFrame(rows)
        base = alt.Chart(data).encode(
            y=alt.Y("arm:N", sort=list(data["arm"]), title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["arm", "reference", "placebo"],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, GREY],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q",
                title="Quantile score minus the default's (points of capacity; lower is better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=WIDTH,
                height=24 * data.height,
                title=_title(
                    text=text,
                    lines=[
                        "Kept rows, difference from the default's table (fitted once on the "
                        "hindcast). Dot: estimate. Line: 99.67% interval from re-drawing whole "
                        "months and seeds; zero is the same score as the default. " + SCOPE
                    ],
                ),
            )
        )
    return alt.vconcat(*panels).resolve_scale(x="independent")


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Quantile-score gain by month of the run, kept rows, for the three arms."""
    rows = []
    for key in ("scored", "out_of_time"):
        for arm in ("x2", "x1", "x3"):
            for month, value in score[key]["contrasts"][arm]["by_month | qs"].items():
                rows.append({"month": month, "arm": ARM_LABELS[arm], "gain": value})
    data = pl.DataFrame(rows)
    line = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", title="Month of the forecast run", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("gain:Q", title="Quantile score minus the default's (points; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in ("x2", "x1", "x3")],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, line).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="The refitted tables gain in most months",
            lines=[
                (
                    "Kept rows, mean over the 3 farms and 3 seeds. Zero is the same score as the "
                    "default. The last three months (2026-07 to 2026-09) are out of time, with "
                    "fewer runs per month."
                )
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """The x3 offsets (mean over lead days 1 to 10) at four quantile levels, per fit month."""
    tables = pl.read_parquet(common.FINAL_DIR / "tables_seed0.parquet").filter(
        (pl.col("arm") == "x3")
        & pl.col("lead_day").is_between(1, 10)
        & pl.col("level").is_in(["q05", "q50", "q95", "q99"])
    )
    data = (
        tables.group_by("site", "month", "level")
        .agg(offset=pl.col("offset").mean() * 100.0)
        .sort("site", "month")
    )
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:N", title="Month of the table's first day", axis=alt.Axis(labelAngle=-60)
            ),
            y=alt.Y("offset:Q", title="Offset (points of capacity)"),
            color=alt.Color(
                "level:N",
                sort=["q05", "q50", "q95", "q99"],
                scale=alt.Scale(
                    domain=["q05", "q50", "q95", "q99"],
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH, height=130)
    )
    return (
        alt.layer(chart, data=data)
        .facet(row=alt.Row("site:N", title=None))
        .properties(
            title=_title(
                text="The trailing 3-month table's offsets change through time",
                lines=[
                    (
                        "Offset added to the forecast quantile at four levels, mean over lead "
                        "days 1 to 10, fitted on the first day of each month from the earlier 3 "
                        "months of daily runs. Seed 0. Farms W1 to W3 are anonymised."
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_2(score=score).save(DOCS_DIR / "figure_2.svg")
    figure_3().save(DOCS_DIR / "figure_3.svg")


if __name__ == "__main__":
    main()
