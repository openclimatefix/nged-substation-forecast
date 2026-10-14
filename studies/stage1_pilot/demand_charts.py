"""Draw the demand experiment's charts from the saved results tables, as SVG.

Reads `demand/results/levels.parquet` and `contrasts.parquet` (written by `demand_score.py`) and
writes two charts under `data/studies/per_study/stage1_pilot/demand/charts/`:

1. the mean absolute error by lead band, one panel per variant (weather only, plus calendar, plus
   calendar and lags), for the climatology, the ENS-trained arm, and the two ERA5-trained arms;
2. the paired difference of the ERA5-trained arm against the ENS-trained arm, and of the short
   against the long ERA5 history, by lead band.

The SVGs are then optimised with `npx svgo@4 --multipass --precision=1 --final-newline`. No
calendar date appears on any chart.

    uv run python studies/stage1_pilot/demand_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import demand_inputs as demand
import plotting.ocf_theme as ocf
import polars as pl
import stage1_inputs as inputs

RESULT_DIR: Final[Path] = demand.demand_subfolder(name="results")
CHART_DIR: Final[Path] = demand.demand_subfolder(name="charts")
WIDTH: Final[int] = 640
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in inputs.LEAD_BANDS)

VARIANT_TITLES: Final[dict[str, str]] = {
    "V0": "Weather columns only",
    "V1": "Weather plus hour of day and day of week",
    "V2": "Weather, calendar, and lagged power",
}
ARM_LABELS: Final[dict[str, str]] = {
    "climatology_dow": "Climatology (month, time of day, weekday or weekend)",
    "ens_trained": "Trained on ENS weather",
    "s1_memberavg": "Trained on ERA5, 2019 onward",
    "s1_short": "Trained on ERA5, 2024-04 onward",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "climatology_dow": "#999999",
    "ens_trained": ocf.BRAND_ORANGE,
    "s1_memberavg": ocf.DATA_BLUE,
    "s1_short": ocf.DATA_GREEN,
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "D1": "ERA5-trained minus ENS-trained, weather only",
    "E1: ERA5-trained minus ENS-trained, V1": "ERA5-trained minus ENS-trained, with calendar",
    "E2: ERA5-trained minus ENS-trained, V2": "ERA5-trained minus ENS-trained, with calendar and lags",
    "E6: short ERA5 history minus long, V1": "Short ERA5 history minus long, with calendar",
}
CONTRAST_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_SKY, ocf.DATA_GREEN]

DEFINITIONS: Final[str] = (
    "ERA5: ECMWF's estimate of past weather. ENS: ECMWF's 51-member ensemble weather forecast; "
    "every forecast is the mean over the 51 members. Every model is an XGBoost model."
)
SCOPE: Final[str] = (
    "22 electricity series (16 primary-substation demand, 4 bulk or grid supply flows, 2 battery "
    "or biofuel); 73 ENS runs, one every 5 days, July 2025 to June 2026. Intervals come from "
    "re-drawing whole months at random and fitting with different random seeds."
)
FIGURE_TEXT: Final[dict[str, list[str]]] = {
    "1": [
        (
            "Without calendar columns no model clearly beats the climatology, and the ERA5-trained "
            "model is worse than the ENS-trained model"
        ),
        (
            "Mean absolute error of the ensemble-mean forecast, points of the series' capacity "
            "(smaller is better), by how many days ahead the forecast is."
        ),
        DEFINITIONS,
        SCOPE,
        "Dot: estimate. Line: 95% interval.",
    ],
    "2": [
        (
            "The ERA5-trained model loses to the ENS-trained model without lags, and training on "
            "recent years only closes most of the gap"
        ),
        (
            "Change in mean absolute error (points of capacity; more negative means the first "
            "model is better), by how many days ahead the forecast is."
        ),
        DEFINITIONS,
        SCOPE,
        "Dot: estimate. Line: 95% interval. Solid rule: no difference.",
    ],
}


def _titles(*, key: str) -> alt.TitleParams:
    lines = FIGURE_TEXT[key]
    return alt.TitleParams(
        text=lines[0], subtitle=lines[1:], anchor="start", fontSize=14, subtitleFontSize=11
    )


def _panel(
    *, data: pl.DataFrame, labels: list[str], colours: list[str], x_title: str, title: str
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=labels, range=colours),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title=x_title)
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=WIDTH - 200, height=240, title=alt.TitleParams(title, anchor="start")
    )


def figure_1() -> alt.TopLevelMixin:
    """The mean absolute error by lead band, one panel per variant."""
    levels = pl.read_parquet(RESULT_DIR / "levels.parquet").filter(
        (pl.col("group") == "all") & (pl.col("tail") == "all cases") & (pl.col("band") != "all")
    )
    panels = []
    for variant, title in VARIANT_TITLES.items():
        names = {
            "climatology_dow": "climatology_dow",
            "ens_trained": f"ens_trained__{variant}",
            "s1_memberavg": f"s1_memberavg__{variant}",
            "s1_short": f"s1_short__{variant}",
        }
        data = levels.filter(pl.col("arm").is_in(list(names.values()))).with_columns(
            label=pl.col("arm").replace_strict({v: ARM_LABELS[k] for k, v in names.items()})
        )
        panels.append(
            _panel(
                data=data,
                labels=list(ARM_LABELS.values()),
                colours=list(ARM_COLOURS.values()),
                x_title="Mean absolute error (points of capacity; smaller is better)",
                title=title,
            )
        )
    return (
        alt.vconcat(*panels).properties(title=_titles(key="1")).resolve_scale(color="independent")
    )


def figure_2() -> alt.TopLevelMixin:
    """The paired differences by lead band, in one panel."""
    contrasts = pl.read_parquet(RESULT_DIR / "contrasts.parquet").filter(
        (pl.col("group") == "all")
        & (pl.col("tail") == "all cases")
        & (pl.col("band") != "all")
        & pl.col("contrast").is_in(list(CONTRAST_LABELS))
    )
    data = contrasts.with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS), value=pl.col("difference")
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    panel = alt.layer(
        zero,
        _panel(
            data=data,
            labels=list(CONTRAST_LABELS.values()),
            colours=CONTRAST_COLOURS,
            x_title="Change in mean absolute error (points of capacity; more negative is better)",
            title="22 series",
        ),
    )
    return panel.properties(title=_titles(key="2"))


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
