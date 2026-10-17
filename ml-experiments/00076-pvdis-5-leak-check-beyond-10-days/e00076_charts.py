"""Draw experiment 00076's figure from the saved results, as SVG.

Figure 1: the contrasts at leads of 10 days and beyond, with their intervals.

The SVG is afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00076-*/e00076_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00076_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 80
LABELS: Final[dict[tuple[str, str], str]] = {
    ("am", "d0"): "am minus d0 (experiment 64's gain)",
    ("am", "d1l"): "am minus d1l (whole effect)",
    ("am0", "d1l"): "am0 minus d1l (start at zero)",
    ("am", "am0"): "am minus am0 (offset in training)",
    ("am_raw", "d1l_raw"): "am minus d1l, uncorrected",
    ("am0_raw", "d1l_raw"): "am0 minus d1l, uncorrected",
    ("am_raw", "am0_raw"): "am minus am0, uncorrected",
}
GROUPS: Final[dict[str, str]] = {
    "decision": "After the 14-day correction (98.33% interval)",
    "raw": "Without the correction (95% interval)",
    "context": "Experiment 64's own contrast (95% interval)",
}
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, leads of 10 days and "
    "beyond. Dot: difference in mean quantile score (points of capacity; negative favours the "
    "first arm). Line: interval from resampling whole weeks of forecast start."
)


def main() -> None:
    """Draw figure 1 from the decision and exploratory tables."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    decision = pl.read_parquet(common.RESULTS_DIR / "decision.parquet").with_columns(
        group=pl.lit(GROUPS["decision"])
    )
    exploratory = pl.read_parquet(common.RESULTS_DIR / "exploratory.parquet")
    raw = exploratory.filter(pl.col("scope") == "10+ d, no correction").with_columns(
        group=pl.lit(GROUPS["raw"])
    )
    context = exploratory.filter(
        (pl.col("scope") == "10+ d") & (pl.col("treatment") == "am") & (pl.col("reference") == "d0")
    ).with_columns(group=pl.lit(GROUPS["context"]))
    table = pl.concat([context, decision, raw], how="diagonal_relaxed").with_columns(
        label=pl.struct("treatment", "reference").map_elements(
            lambda row: LABELS[(row["treatment"], row["reference"])], return_dtype=pl.String
        )
    )
    order = [LABELS[key] for key in LABELS]
    base = alt.Chart(table.to_pandas()).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(
                domain=list(GROUPS.values()), range=[ocf.DATA_BLUE, ocf.DATA_GREEN, "#8A8A8A"]
            ),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
        ),
    )
    rules = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Difference in quantile score (points of capacity)",
            scale=alt.Scale(domain=[-0.2, 0.1]),
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=90, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q"
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    title = alt.TitleParams(
        text="A refit with no solar term reproduces the gain at 10 days and beyond",
        subtitle=textwrap.wrap(SCOPE, WRAP_CHARACTERS),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )
    chart = (rules + dots + zero).properties(width=PLOT_WIDTH, height=230, title=title)
    chart.save(CHART_DIR / "figure_1_contrasts.svg")
    print(f"written {CHART_DIR / 'figure_1_contrasts.svg'}")


if __name__ == "__main__":
    main()
