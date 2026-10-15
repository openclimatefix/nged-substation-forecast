"""Draw experiment 00035's figures from the saved results, as SVG.

1. The blends minus the best single member, over all leads, per technology.
2. The rolling blend minus the best single member, per lead band.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00035-*/e00035_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00035_blend as blend
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = blend.WORK_DIR / "results"
CHART_DIR: Final[Path] = blend.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 380
WRAP_CHARACTERS: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "Primary demand: 73 runs, one every 5 days, seed 0. Solar and wind: 53 Tuesday runs, three "
    "seeds. July 2025 to June 2026."
)
GROUP_LABELS: Final[dict[str, str]] = {
    "Primary": "Primary demand",
    "solar": "Solar",
    "wind": "Wind",
    "BSP_GSP": "BSP and GSP (descriptive)",
}
ARM_LABELS: Final[dict[str, str]] = {
    "rolling": "Weights per lead band, chosen on earlier months",
    "global": "One weight for all leads, chosen on earlier months",
    "equal": "Equal weights",
}
SHORT_LABELS: Final[dict[str, str]] = {
    "rolling": "lead bands",
    "global": "one weight",
    "equal": "equal weights",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "rolling": ocf.BRAND_ORANGE,
    "global": ocf.DATA_BLUE,
    "equal": ocf.DATA_GREEN,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Each blend minus the best single member, quantile score over all leads."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        (pl.col("band") == "all")
        & pl.col("treatment").is_in(list(ARM_LABELS))
        & pl.col("reference").str.starts_with("member_")
        & pl.col("family").is_in(["decisive", "member"])
    )
    data = contrasts.with_columns(
        group_label=pl.col("group").replace_strict(GROUP_LABELS),
        arm_label=pl.col("treatment").replace_strict(ARM_LABELS),
        row=pl.col("group").replace_strict(GROUP_LABELS)
        + ", "
        + pl.col("treatment").replace_strict(SHORT_LABELS),
    )
    order = [
        f"{GROUP_LABELS[g]}, {SHORT_LABELS[t]}"
        for g in GROUP_LABELS
        for t in ("rolling", "global", "equal")
    ]
    base = alt.Chart(data).encode(
        y=alt.Y(
            "row:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=330),
        ),
        color=alt.Color(
            "arm_label:N",
            scale=alt.Scale(
                domain=list(ARM_LABELS.values()),
                range=[ARM_COLOURS[k] for k in ARM_LABELS],
            ),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Blend minus best member (points of capacity; negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="Blending helps Primary demand; solar and wind gains are unresolved",
            lines=[
                (
                    "Quantile score over all leads of each blend minus the best single member "
                    "of its technology (smaller is better). Grey line: same as the best member. "
                    "Dot: estimate. Line: interval from re-drawing whole months, 98.33% for the "
                    "lead-band blend and 99.17% for the others. " + SCOPE
                )
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The lead-band blend minus the best single member, by lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(pl.col("family") == "band")
        .with_columns(group_label=pl.col("group").replace_strict(GROUP_LABELS))
    )
    bands = [name for name, _, _ in blend.LEAD_BANDS]
    colours = {
        "Primary demand": ocf.DATA_BLUE,
        "Solar": ocf.BRAND_ORANGE,
        "Wind": ocf.DATA_GREEN,
        "BSP and GSP (descriptive)": ocf.DATA_PURPLE,
    }
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=bands, title="Lead band"),
        yOffset=alt.YOffset("group_label:N"),
        color=alt.Color(
            "group_label:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Blend minus best member (points of capacity; negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="Wind gains most under 2 days; Primary demand gains evenly",
            lines=[
                (
                    "Quantile score of the lead-band blend minus the best single member of its "
                    "technology, per lead band (smaller is better). Grey line: same as the best "
                    "member. Dot: estimate. Line: 99.67% interval from re-drawing whole months. "
                    + SCOPE
                )
            ],
        ),
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
