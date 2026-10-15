"""Draw experiment 00023's figures from the saved results, as SVG.

1. The planned CRPS contrasts (and the post hoc P1) with intervals, one panel per technology.
2. Coverage of the 5% to 95% band by lead band, for the mixture and its two corrected versions.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00023-*/e00023_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00023_common import LEAD_BANDS, WORK_DIR

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; "
    "pooled mixture of 51 ENS members, 3 seeds."
)
ARM_LABELS: Final[dict[str, str]] = {
    "mixture": "Production recipe (pooled mixture)",
    "mix_c14": "Mixture + 14-day correction (mean error)",
    "mix_c14_median": "Mixture + 14-day correction (median error, post hoc)",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Lagged power added to calibrator",
    "C2": "Summaries added to calibrator",
    "C3": "Summaries added to lagged calibrator",
    "C4": "Calibrator with both minus recipe",
    "C5": "14-day correction (mean) minus recipe",
    "C6": "14-day correction added to calibrator with both",
    "P1": "14-day correction (median, post hoc) minus recipe",
}
COVERAGE_ARMS: Final[tuple[str, ...]] = ("mixture", "mix_c14", "mix_c14_median")
COVERAGE_COLOURS: Final[tuple[str, ...]] = ("#5a5a5a", ocf.BRAND_ORANGE, ocf.DATA_BLUE)
Y_SCALE: Final[alt.Scale] = alt.Scale(domain=[50, 100])


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The contrasts, one panel per technology."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("contrast") + ": " + pl.col("contrast").replace_strict(CONTRAST_LABELS)
    )
    panels = []
    for technology in ("wind", "solar"):
        part = contrasts.filter(pl.col("technology") == technology).sort("contrast")
        labels = part["label"].to_list()
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#292b2b", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.condition(
                "datum.contrast == 'C1' || datum.contrast == 'C5'",
                alt.value(ocf.BRAND_ORANGE),
                alt.value(ocf.DATA_BLUE),
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="CRPS difference (points of capacity; negative = better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=PLOT_WIDTH, height=24 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Recent power does not lower the CRPS of the calibrator",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: interval from re-drawing whole months and seeds "
                    "(99.58% for C1 to C6, 95% for the post hoc P1). Orange: the two contrasts "
                    "that decide the verdict. Black dashed line: no change."
                ),
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Coverage by lead band for the mixture and its corrected versions."""
    data = (
        pl.read_parquet(RESULTS_DIR / "coverage.parquet")
        .filter(pl.col("arm").is_in(list(COVERAGE_ARMS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    order = [name for name, _, _ in LEAD_BANDS]
    labels = [ARM_LABELS[a] for a in COVERAGE_ARMS]
    panels = []
    for technology in ("wind", "solar"):
        part = data.filter(pl.col("technology") == technology)
        target = alt.Chart(pl.DataFrame({"y": [90.0]})).mark_rule(
            color=ocf.DATA_GREEN, strokeDash=[2, 3]
        )
        target = target.encode(y=alt.Y("y:Q", scale=Y_SCALE))  # ty: ignore[unresolved-attribute]
        base = alt.Chart(part).encode(
            x=alt.X(
                "band:N", sort=order, title="Lead time of the forecast", axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("label:N", sort=labels),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=list(COVERAGE_COLOURS)),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=330),
            ),
        )
        points = base.mark_point(filled=True, size=36).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("coverage:Q", title="Cases inside the 5% to 95% band (%)", scale=Y_SCALE)
        )
        bars = base.mark_rule().encode(y=alt.Y("lower:Q", scale=Y_SCALE), y2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(target, bars, points).properties(
                width=PLOT_WIDTH, height=190, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The mean-error correction cuts coverage of the 5% to 95% band to 53% to 77%",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months. "
                    "Green dotted line: "
                    "the 90% the band should cover. Cases with a 95% "
                    "quantile below 1% of capacity (night-time solar) are left out."
                ),
            ],
        )
    )


def main() -> None:
    """Write the figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in ((1, figure_1()), (2, figure_2())):
        chart.save(CHART_DIR / f"figure_{number}.svg")


if __name__ == "__main__":
    main()
