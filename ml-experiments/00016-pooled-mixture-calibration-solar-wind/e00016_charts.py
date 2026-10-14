"""Draw experiment 00016's figures from the saved results, as SVG.

1. Coverage of the 5% to 95% band by lead band, per arm and technology, with 95% intervals.
2. The CRPS of each correction minus that of the uncorrected mixture, with 99.4% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00016-*/e00016_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00016_common import LEAD_BANDS, WORK_DIR

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
ARM_LABELS: Final[dict[str, str]] = {
    "none": "No correction",
    "spread": "One spread factor",
    "spread_band": "Spread factor per lead band",
    "locscale": "Location and scale per lead band",
    "free_quantile": "Free quantile offsets per lead band",
    "spread_two_groups": "Post hoc: spread factor for 0 to 2 and 3 to 15 days",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "none": "#5a5a5a",
    "spread": ocf.DATA_BLUE,
    "spread_band": ocf.DATA_SKY,
    "locscale": ocf.DATA_PURPLE,
    "free_quantile": ocf.DATA_GREEN,
    "spread_two_groups": ocf.BRAND_ORANGE,
}
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; "
    "pooled mixture of 51 ENS members, 3 seeds."
)
TOLERANCE: Final[float] = 2.5
Y_SCALE: Final[alt.Scale] = alt.Scale(domain=[80, 100])


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Coverage by lead band, one panel per technology."""
    data = (
        pl.read_parquet(RESULTS_DIR / "coverage.parquet")
        .filter((pl.col("months") == "all") & pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    order = [name for name, _, _ in LEAD_BANDS]
    labels = list(ARM_LABELS.values())
    panels = []
    for technology in ("wind", "solar"):
        part = data.filter(pl.col("technology") == technology)
        target = alt.Chart(
            pl.DataFrame({"low": [90.0 - TOLERANCE], "high": [90.0 + TOLERANCE]})
        ).mark_rect(color=ocf.DATA_GREEN, opacity=0.15)
        target = target.encode(y=alt.Y("low:Q", scale=Y_SCALE), y2="high:Q")  # ty: ignore[unresolved-attribute]
        line = (
            alt.Chart(pl.DataFrame({"y": [90.0]}))
            .mark_rule(color="#292b2b", strokeDash=[4, 3])
            .encode(y=alt.Y("y:Q", scale=Y_SCALE))  # ty: ignore[unresolved-attribute]
        )
        base = alt.Chart(part).encode(
            x=alt.X(
                "band:N", sort=order, title="Lead time of the forecast", axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("label:N", sort=labels),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in ARM_LABELS]),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        points = base.mark_point(filled=True, size=36).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "coverage:Q",
                title="Cases inside the 5% to 95% band (%)",
                scale=Y_SCALE,
            )
        )
        bars = base.mark_rule().encode(y=alt.Y("lower:Q", scale=Y_SCALE), y2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(target, line, bars, points).properties(
                width=PLOT_WIDTH, height=190, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The 90% band is too wide for solar, and for wind within two days",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months. Dashed "
                    "line: the 90% the band should cover. Green band: within 2.5 points of 90. "
                    "Cases with a forecast 95% quantile below 1% of capacity (night-time solar) "
                    "are left out."
                ),
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """The CRPS contrasts against the uncorrected mixture, one panel per technology."""
    contrasts = pl.concat(
        [
            pl.read_parquet(RESULTS_DIR / "contrasts.parquet"),
            pl.read_parquet(RESULTS_DIR / "post_hoc.parquet"),
        ]
    ).with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    labels = [ARM_LABELS[arm] for arm in ARM_LABELS if arm != "none"]
    panels = []
    for technology in ("wind", "solar"):
        part = contrasts.filter(pl.col("technology") == technology)
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#292b2b", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="CRPS minus uncorrected mixture's (points of capacity; negative = better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=PLOT_WIDTH, height=22 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="No correction changed the CRPS by more than 0.2 points of capacity",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 99.4% interval from re-drawing whole months and seeds "
                    "(a 5% level corrected for 8 planned contrasts; the post hoc arm uses the "
                    "same level). Dashed line: same CRPS as the uncorrected mixture."
                ),
            ],
        )
    )


def main() -> None:
    """Write both figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in ((1, figure_1()), (2, figure_2())):
        chart.save(CHART_DIR / f"figure_{number}.svg")


if __name__ == "__main__":
    main()
