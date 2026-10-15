"""Draw experiment 00024's figures from the saved results, as SVG.

1. The nine planned contrasts on the quantile score over all leads, with 99.44% intervals.
2. The quantile score of each grouping, for the 16 Primary substations and for the 4 BSP and GSP.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00024-pooling-groupings-demand-xgboost/e00024_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00024_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
GREY: Final[str] = "#999999"
ARM_LABELS: Final[dict[str, str]] = {
    "own_stack": "One model per series (20 models)",
    "size_id_stack": "Per size band, with ID (3)",
    "type_id_stack": "Per substation type, with ID (4)",
    "random_id_stack": "Per random group, with ID (4)",
    "all_id_stack": "All series, with ID (1)",
    "type_noid_stack": "Per substation type, no ID (4)",
    "all_noid_stack": "All series, no ID (1)",
    "manual_heuristic": "manual_heuristic",
    "climatology": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "own_stack": ocf.DATA_BLUE,
    "size_id_stack": ocf.DATA_SKY,
    "type_id_stack": ocf.DATA_SKY,
    "random_id_stack": ocf.DATA_SKY,
    "all_id_stack": ocf.DATA_SKY,
    "type_noid_stack": ocf.BRAND_ORANGE,
    "all_noid_stack": ocf.BRAND_ORANGE,
    "manual_heuristic": GREY,
    "climatology": GREY,
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1 (primary): by type, ID, minus per series",
    "C2": "C2: all series, ID, minus per series",
    "C3": "C3: size bands, ID, minus per series",
    "C4": "C4: by type, ID, minus all series, ID (20 series)",
    "C5": "C5: by type, no ID, minus per series",
    "C6": "C6: all series, no ID, minus per series",
    "C7": "C7: by type, no ID, minus by type, ID",
    "C8": "C8: all, no ID, minus all, ID (20 series)",
    "C9": "C9: by type, ID, minus random groups, ID (20 series)",
}
SCOPE: Final[str] = "73 forecast runs, one every 5 days, July 2025 to June 2026, ENS weather."


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The nine planned contrasts on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("planned")
        & (pl.col("metric") == "quantile_score")
        & (pl.col("scope") == "all leads")
    )
    data = data.with_columns(label=pl.col("label").str.slice(0, 2).replace_strict(CONTRAST_LABELS))
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=data["label"].to_list(), title=None, axis=alt.Axis(labelLimit=330))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    dashed = alt.Chart(pl.DataFrame({"x": [0.3]})).mark_rule(
        color=ocf.BRAND_ORANGE, strokeDash=[4, 3]
    )
    margin = dashed.encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, margin, bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="Pooling costs 0.15 to 0.29 points with the series ID and 0.8 to 1.0 without it",
            lines=[
                (
                    "Each row is the first arm minus the second on the quantile score over all "
                    "leads; below zero favours the first arm. Grey line: no difference. Orange "
                    "dashed line: the 0.30-point margin. C1 to C3 and C5 to C7 use the 16 Primary "
                    "substations; the others use all 20 series. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 99.44% interval (Bonferroni across the nine planned "
                    "contrasts) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The quantile score of each grouping, for Primary and for BSP and GSP."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") == "all leads")
            & pl.col("arm").is_in(list(ARM_LABELS))
            & pl.col("group").is_in(["Primary (16)", "BSP and GSP (4)"])
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=240)
        ),
        color=alt.Color(
            "arm:N",
            legend=None,
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False))
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, points)
        .properties(width=PLOT_WIDTH - 230, height=190)
        .facet(
            row=alt.Row(
                "group:N",
                sort=["Primary (16)", "BSP and GSP (4)"],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="One model per substation type scores close to one model per series",
                lines=[
                    (
                        "Quantile score over all leads (smaller is better), after the 14-day "
                        "error correction and spread factor of 1.15. Blue: one model per series. "
                        "Light blue: pooled with the series ID. Orange: pooled without it. Grey: "
                        "references. Dot: estimate. Line: 95% interval from re-drawing whole "
                        "months and seeds. " + SCOPE
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
