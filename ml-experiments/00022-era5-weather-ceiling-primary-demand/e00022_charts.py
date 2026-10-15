"""Draw experiment 00022's figures from the saved results, as SVG.

1. Each ERA5 arm minus the ENS arm on the quantile score over all leads, with 99.38% intervals.
2. The all-variable ERA5 arm minus the ENS arm by lead band, with 95% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00022-era5-weather-ceiling-primary-demand/e00022_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00022_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "era5_all": "All seven weather columns",
    "era5_shortwave": "Shortwave radiation",
    "era5_longwave": "Longwave radiation",
    "era5_temperature_2m": "Temperature at 2 m",
    "era5_dew_point_temperature_2m": "Dew point at 2 m",
    "era5_pressure_surface": "Surface pressure",
    "era5_precipitation": "Precipitation",
    "era5_wind_speed_10m": "Wind speed at 10 m",
}
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def figure_1() -> alt.TopLevelMixin:
    """Each ERA5 arm minus the ENS arm over all leads."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter((pl.col("scope") == "all leads") & pl.col("treatment").is_in(list(ARM_LABELS)))
        .with_columns(label=pl.col("treatment").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None))
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="ERA5 shortwave radiation carries nearly all of the gain",
            lines=[
                (
                    "Each row is ERA5 weather minus ENS weather at serving time, on the quantile "
                    "score over all leads; below zero favours ERA5. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 99.38% interval (Bonferroni across the eight planned "
                    "contrasts) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The all-variable ERA5 arm minus the ENS arm by lead band."""
    bands = [band[0] for band in common.LEAD_BANDS]
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        (pl.col("treatment") == "era5_all") & pl.col("scope").is_in(bands)
    )
    base = alt.Chart(data).encode(y=alt.Y("scope:N", sort=bands, title="Lead"))
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="ERA5 weather helps from 3 days ahead and slightly hurts under 24 hours",
            lines=[
                (
                    "All seven weather columns from ERA5 minus ENS, on the quantile score by lead "
                    "band; below zero favours ERA5. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds "
                    "(not corrected for the five bands)."
                ),
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
