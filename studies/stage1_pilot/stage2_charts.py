"""Draw the stage 2 pilot's charts from the saved results tables, as SVG.

Reads `results/*.parquet` (written by `stage2_report.py`) and writes two charts under
`data/studies/per_study/stage2_pilot/charts/`:

1. the CRPS by lead band for the benchmark, the pooled mixture, stage 2 with the blend, and
   climatology, with 95% intervals;
2. the three planned contrasts, Q1 to Q3, by lead band, with 95% intervals.

The SVGs are then optimised with `npx svgo@4 --multipass --precision=1 --final-newline`. No
calendar date appears on any chart.

    uv run python studies/stage1_pilot/stage2_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import stage1_inputs as inputs
from stage2_common import stage2_subfolder

RESULT_DIR: Final[Path] = stage2_subfolder(name="results")
CHART_DIR: Final[Path] = stage2_subfolder(name="charts")
WIDTH: Final[int] = 640
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in inputs.LEAD_BANDS)
TECHNOLOGY_LABELS: Final[dict[str, str]] = {"solar": "6 solar farms", "wind": "3 wind farms"}

ARM_LABELS: Final[dict[str, str]] = {
    "climatology_q": "Climatology (month and time of day)",
    "a_ens_quantile": "Trained on ENS weather (benchmark)",
    "b_mixture_full": "Trained on ERA5, pooled over the 51 ENS members",
    "c_stage2_full_blp": "Trained on ERA5, then calibrator and blend (stage 2)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "climatology_q": "#999999",
    "a_ens_quantile": ocf.BRAND_ORANGE,
    "b_mixture_full": ocf.DATA_BLUE,
    "c_stage2_full_blp": ocf.DATA_GREEN,
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "Q1": "Q1: stage 2 minus the pooled ERA5-trained forecast",
    "Q2": "Q2: stage 2 minus the ENS-trained benchmark",
    "Q3": "Q3: pooled ERA5-trained forecast minus the ENS-trained benchmark",
}
CONTRAST_COLOURS: Final[list[str]] = [ocf.DATA_GREEN, ocf.BRAND_ORANGE, ocf.DATA_BLUE]

DEFINITIONS: Final[str] = (
    "CRPS: continuous ranked probability score, estimated from nine quantile levels. ERA5: "
    "ECMWF's estimate of past weather. ENS: ECMWF's 51-member ensemble weather forecast. Every "
    "XGBoost model gets the same weather columns and no calendar or power-history columns."
)
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 ENS runs, one every 7 days, July 2025 to June 2026. "
    "Intervals come from re-drawing whole months at random and fitting with different random seeds."
)
FIGURE_TEXT: Final[dict[str, list[str]]] = {
    "1": [
        (
            "Both ERA5-trained forecasts have a lower probabilistic error than the ENS-trained "
            "benchmark up to day 6; beyond that no arm clearly beats climatology"
        ),
        (
            "Continuous ranked probability score, points of the generator's capacity (smaller is "
            "better), by how many days ahead the forecast is."
        ),
        DEFINITIONS,
        SCOPE,
        "Dot: estimate. Line: 95% interval.",
    ],
    "2": [
        (
            "The ERA5-trained pooled forecast beats the ENS-trained benchmark up to day 6; the "
            "second model (stage 2) does not improve on the pooled forecast"
        ),
        (
            "Change in continuous ranked probability score (points of capacity; more negative "
            "means the first forecast is better), by how many days ahead the forecast is."
        ),
        DEFINITIONS,
        SCOPE,
        (
            "Dot: estimate. Line: 95% interval. Solid rule: no difference. Dashed rule: the "
            "smallest difference counted as material, -0.15 points."
        ),
    ],
}


def _titles(*, key: str) -> alt.TitleParams:
    lines = FIGURE_TEXT[key]
    return alt.TitleParams(
        text=lines[0], subtitle=lines[1:], anchor="start", fontSize=14, subtitleFontSize=11
    )


def _legend(*, domain: list[str], range_: list[str]) -> alt.Color:
    return alt.Color(
        "label:N",
        scale=alt.Scale(domain=domain, range=range_),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
    )


def _panel(
    *, data: pl.DataFrame, labels: list[str], colours: list[str], x_title: str, title: str
) -> alt.LayerChart:
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=_legend(domain=labels, range_=colours),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title=x_title)
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=WIDTH - 200, height=260, title=alt.TitleParams(title, anchor="start")
    )


def figure_1() -> alt.TopLevelMixin:
    """CRPS by lead band, one panel per technology."""
    levels = pl.read_parquet(RESULT_DIR / "levels.parquet").filter(
        (pl.col("metric") == "crps")
        & pl.col("arm").is_in(list(ARM_LABELS))
        & (pl.col("band") != "all")
    )
    data = levels.with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    panels = [
        _panel(
            data=data.filter(pl.col("technology") == technology),
            labels=list(ARM_LABELS.values()),
            colours=list(ARM_COLOURS.values()),
            x_title="CRPS (points of capacity; smaller is better)",
            title=name,
        )
        for technology, name in TECHNOLOGY_LABELS.items()
    ]
    return (
        alt.vconcat(*panels).properties(title=_titles(key="1")).resolve_scale(color="independent")
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts Q1 to Q3 by lead band, one panel per technology."""
    contrasts = pl.read_parquet(RESULT_DIR / "contrasts.parquet").filter(
        (pl.col("metric") == "crps") & pl.col("contrast").is_in(list(CONTRAST_LABELS))
    )
    data = contrasts.filter(pl.col("band") != "all").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS), value=pl.col("difference")
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    material = (
        alt.Chart(pl.DataFrame({"x": [-0.15]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#777777")
        .encode(x="x:Q")
    )
    panels = [
        alt.layer(
            zero,
            material,
            _panel(
                data=data.filter(pl.col("technology") == technology),
                labels=list(CONTRAST_LABELS.values()),
                colours=CONTRAST_COLOURS,
                x_title="Change in CRPS (points of capacity; more negative is better)",
                title=name,
            ),
        )
        for technology, name in TECHNOLOGY_LABELS.items()
    ]
    return (
        alt.vconcat(*panels).properties(title=_titles(key="2")).resolve_scale(color="independent")
    )


def main() -> None:
    """Draw both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for key, build in {"1": figure_1, "2": figure_2}.items():
        chart = build()
        chart.save(CHART_DIR / f"figure_{key}.svg")
        chart.save(CHART_DIR / f"figure_{key}.png", scale_factor=2)
        print(f"figure {key} written")


if __name__ == "__main__":
    main()
