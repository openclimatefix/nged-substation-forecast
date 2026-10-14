"""Draw experiment 00012's figures from the saved results, as SVG.

1. Each arm's absolute error with its 95% interval, for solar and for wind.
2. The ten planned contrasts with their 99.5% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00012-*/e00012_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00012_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology (month and time of day)",
    "w": "w: benchmark (ERA5-trained, from 2019-09)",
    "start_2023": "Training from 2023-09",
    "start_2024": "Training from 2024-04 (ENS-era months)",
    "small_block": "Four weather columns only",
    "speed_block": "100 m wind speed and bearing only",
    "csr": "Clear-sky-ratio target",
    "base_margin": "Power-curve base margin",
    "cams": "Trained on CAMS irradiance",
    "cams_stage0": "CAMS-trained, ENS irradiance calibrated (stage 0)",
    "cams_control": "CAMS-trained, control member only",
    "cams_stage0_control": "CAMS-trained with stage 0, control member only",
    "era5_ceiling": "Given ERA5 weather (not a forecast)",
}
GROUP_COLOURS: Final[dict[str, str]] = {
    "Benchmark": ocf.BRAND_ORANGE,
    "Tested arm": ocf.DATA_BLUE,
    "Reference": "#5a5a5a",
}
VERDICT_COLOURS: Final[dict[str, str]] = {
    "gain": ocf.DATA_GREEN,
    "equivalent": ocf.DATA_BLUE,
    "unresolved": ocf.BRAND_ORANGE,
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1 solar: CAMS-trained minus ERA5-trained",
    "C2": "C2 solar: stage 0 minus plain CAMS-trained",
    "C3": "C3 solar: clear-sky-ratio target",
    "C4": "C4 wind: power-curve base margin",
    "C5": "C5 solar: training from 2023-09",
    "C6": "C6 wind: training from 2023-09",
    "C7": "C7 solar: training from 2024-04",
    "C8": "C8 wind: training from 2024-04",
    "C9": "C9 solar: four weather columns only",
    "C10": "C10 wind: 100 m speed and bearing only",
}
MIN_GAIN: Final[float] = 0.15
"""The smallest point estimate, in points of capacity, that counts as a gain."""
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; "
    "mean absolute error of the average of 51 ENS members, 3 seeds."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _group(arm: str) -> str:
    if arm == "w":
        return "Benchmark"
    return "Reference" if arm in ("climatology", "era5_ceiling") else "Tested arm"


def figure_1() -> alt.TopLevelMixin:
    """Each arm's absolute error with its 95% interval, one panel per technology."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            group=pl.col("arm").map_elements(_group, return_dtype=pl.String),
        )
    )
    order = list(ARM_LABELS.values())
    panels = []
    for technology in ("solar", "wind"):
        part = data.filter(pl.col("technology") == technology)
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "value:Q",
                title="Mean absolute error (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            )
        )
        bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(bars, points).properties(
                width=PLOT_WIDTH, height=22 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="No input change lowered the error of the ERA5-trained model by more than "
            "0.15 points",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds. "
                    "The intervals are wide because every arm's error rises and falls together "
                    "with the weather; the contrasts in figure 2 pair the arms and are far "
                    "narrower."
                ),
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """The ten planned contrasts with their 99.5% intervals."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS)
    )
    order = list(CONTRAST_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color(
            "verdict:N",
            scale=alt.Scale(domain=list(VERDICT_COLOURS), range=list(VERDICT_COLOURS.values())),
            legend=alt.Legend(title="Verdict", orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title=[
                "Error of the tested arm minus the reference",
                "(points of capacity; more negative is better)",
            ],
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    gain = (
        alt.Chart(pl.DataFrame({"x": [-MIN_GAIN]}))
        .mark_rule(color="#5a5a5a", strokeDash=[2, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(bars, points, zero, gain).properties(
        width=PLOT_WIDTH,
        height=26 * data.height,
        title=_title(
            text="Only CAMS training came close to the 0.15-point gain that was planned",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 99.5% interval (Bonferroni across the 10 contrasts) "
                    "from re-drawing whole months. Dashed rule: same as the reference. Dotted "
                    "rule: the smallest gain that counts (-0.15)."
                ),
            ],
        ),
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, figure in ((1, figure_1()), (2, figure_2())):
        figure.save(CHART_DIR / f"figure_{number}.svg")


if __name__ == "__main__":
    main()
