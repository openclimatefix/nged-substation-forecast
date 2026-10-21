"""Draw the figures of experiment 152's write-up.

uv run --no-sync python ml-experiments/00152-*/e00152_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00152_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
PERIOD_COLOURS: Final[dict[str, str]] = {
    "even months (replicate)": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}


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


def figure_step0() -> alt.TopLevelMixin:
    """Step 0: predicted reverse flow at sunny middays, by site."""
    table = pl.read_parquet(common.WORK_DIR / "step0_table.parquet").filter(
        pl.col("cut") == "scored"
    )
    order = table.sort("pv_capacity_mw", descending=True)["site"].to_list()
    base = alt.Chart(table.with_columns(pl.col("share_reverse") * 100)).encode(
        y=alt.Y("site:N", sort=order, title=None)
    )
    share = (
        base.mark_bar(color=ocf.BRAND_ORANGE)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X(
                "share_reverse:Q",
                title="Sunny-midday half-hours with predicted reverse flow (%)",
                scale=alt.Scale(domain=[0, 100]),
            )
        )
        .properties(width=WIDTH - 120, height=150)
    )
    size = (
        base.mark_bar(color=ocf.DATA_BLUE)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("largest_reverse_mw:Q", title="Largest predicted reverse flow (MW)"),
        )
        .properties(width=WIDTH - 120, height=150)
    )
    return alt.vconcat(share, size).properties(
        title=_title(
            text="Only D12 has clear predicted reverse flow at sunny middays",
            subtitle=(
                "Scored-year fit of the convex model through the meter, past weather, the 8 "
                "Primaries read in MVA only. D22 reaches 4% of sunny-midday half-hours and D8 "
                "none. D8's series changes meaning in April 2025, so its zero is weak evidence."
            ),
        )
    )


def figure_headline() -> alt.TopLevelMixin:
    """The pre-registered contrasts with 98.75% intervals, both periods."""
    data = json.loads((common.WORK_DIR / "score.json").read_text())
    names = {
        "R1_F_minus_N_D12": "R1 F minus N, D12, all rows",
        "R2_F_minus_A_D12_sunny": "R2 F minus A, D12, sunny middays",
        "R2_all_rows": "R2 F minus A, D12, all rows",
        "R3_F_minus_N_pooled": "R3 F minus N, other seven sites",
    }
    rows = [
        {
            "contrast": name,
            "period": label,
            "estimate": data[period][key]["difference"],
            "low": data[period][key]["lower"],
            "high": data[period][key]["upper"],
        }
        for period, label in (("replicate", "even months (replicate)"), ("scored", "scored year"))
        for key, name in names.items()
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points of capacity; below zero favours F)",
        )
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 260,
        height=220,
        title=_title(
            text="Folded quantiles help at D12's sunny middays and do no harm elsewhere",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours. Line: 98.75% "
                "interval from resampling whole calendar months. Zero: same score. Arm F is "
                "arm A plus folded quantiles; arm N reads the apparent power as signed power."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    _save(figure_step0(), name="figure_0_step0")
    _save(figure_headline(), name="figure_1_headline")


if __name__ == "__main__":
    main()
