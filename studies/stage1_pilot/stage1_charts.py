"""Draw the stage 1 pilot's charts from the saved results tables, as SVG.

Reads `results/*.parquet` (written by `stage1_report.py`) and the per-row scores, and writes five
SVG charts under `data/studies/per_study/stage1_pilot/charts/`:

1. mean absolute error by lead band for each arm, with 95% intervals (the headline chart);
2. the two planned paired contrasts, P1 and P2, with 95% intervals;
3. the slope of observed on predicted power (the over-reaction diagnostic);
4. the post hoc effect of the feature lists, against the first run;
5. an extreme day and a random day at one solar farm and one wind farm.

`--text-only` prints every figure's text for review without drawing. The SVGs are then optimised
with `npx svgo@4 --multipass --precision=1 --final-newline`. No calendar date appears on any chart:
the example days are labelled by hour of day only.

    uv run python studies/stage1_pilot/stage1_charts.py [--text-only]
"""

import argparse
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import stage1_inputs as inputs
from stage1_score_arms import SCORE_DIR

RESULT_DIR: Final[Path] = inputs.OUTPUT_ROOT / "results"
CHART_DIR: Final[Path] = inputs.OUTPUT_ROOT / "charts"
WIDTH: Final[int] = 640

ARM_LABELS: Final[dict[str, str]] = {
    "ens_trained": "Trained on ENS weather (control member)",
    "s1_memberavg": "Trained on ERA5, 51 ENS members averaged",
    "s1_control": "Trained on ERA5, ENS control member only",
    "ens_trained_plus_s1": "Trained on ENS weather + ERA5-trained prediction",
    "s1_cams": "Trained on CAMS, 51 ENS members averaged",
    "climatology": "Climatology (month and time of day)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "ens_trained": ocf.BRAND_ORANGE,
    "s1_memberavg": ocf.DATA_BLUE,
    "ens_trained_plus_s1": ocf.DATA_GREEN,
    "s1_control": ocf.DATA_GREEN,
    "s1_cams": ocf.DATA_PURPLE,
    "climatology": "#999999",
}
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in inputs.LEAD_BANDS)
TECHNOLOGY_LABELS: Final[dict[str, str]] = {"solar": "6 solar farms", "wind": "3 wind farms"}
RANDOM_DAY_SEED: Final[int] = 20261013

DEFINITIONS: Final[str] = (
    "ERA5: ECMWF's estimate of past weather. ENS: ECMWF's 51-member ensemble weather forecast. "
    "Every model on these charts is an XGBoost model given the same weather columns and no "
    "calendar or power-history columns."
)

SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 ENS runs, one every 7 days, July 2025 to June 2026. "
    "Intervals come from re-drawing whole months at random and fitting with different random seeds."
)

FIGURE_TEXT: Final[dict[str, list[str]]] = {
    "1": [
        (
            "A power model trained on ERA5 makes smaller errors than the same model trained on "
            "ENS weather, up to day 6 ahead (one held-out year, nine generators)"
        ),
        (
            "Mean absolute error of the forecast built from the average of the ENS members' "
            "weather, points of the generator's capacity (smaller is better), by how many days "
            "ahead the forecast is."
        ),
        DEFINITIONS,
        SCOPE,
        (
            "Dot: estimate. Line: 95% interval. Grey tick: the ERA5-trained model given ERA5 "
            "weather at the valid time instead of a forecast (not a forecast; it shows how "
            "much of the error comes from forecast weather). Grey dot: climatology."
        ),
    ],
    "2": [
        (
            "Planned comparisons: training on ERA5 lowers error up to day 6; adding the "
            "ERA5-trained prediction as an extra input changes little"
        ),
        (
            "Change in mean absolute error from the ENS-trained model with the same columns, "
            "points of capacity (more negative is better). Solid line at zero: same as that "
            "model."
        ),
        (
            "Blue: the ERA5-trained model alone. Green: the ENS-trained model given the "
            "ERA5-trained model's prediction as an extra input. Dot: estimate. Line: 95% "
            "interval."
        ),
        (
            "Dashed line at -0.15: a change counts as material only if it is below -0.15, its "
            "interval is wholly below zero, and at least 9 of 12 months improve."
        ),
        DEFINITIONS,
        SCOPE,
    ],
    "3": [
        (
            "Averaging the 51 ENS members brings the ERA5-trained forecast's slope close to that "
            "of a forecast with no weather error; a single member's slope falls with lead"
        ),
        (
            "Slope of the best-fit line through observed power (y) against forecast power (x), "
            "both as a fraction of capacity. Below 1: the forecast's swings are larger than "
            "the swings that come true. At 1: on average, observed power equals the forecast "
            "at every forecast level."
        ),
        DEFINITIONS,
        SCOPE,
        (
            "Dashed line at 1. Grey: the ERA5-trained model given ERA5 weather at the valid "
            "time. Slope above 1: the forecast varies less than the outcomes."
        ),
    ],
    "4": [
        (
            "Removing the hour and day-of-year columns helped the ENS-trained model a lot and the "
            "ERA5-trained model little"
        ),
        (
            "Change in mean absolute error when the hour and day-of-year columns are removed, "
            "points of capacity (more negative is better), and when the 24 production columns "
            "are replaced by the same columns as the ERA5-trained model. Solid line at zero: "
            "no change."
        ),
        (
            "Blue: ERA5-trained model, hour and day-of-year columns removed. Orange: "
            "ENS-trained model, same removal. Purple: ENS-trained model on the common columns "
            "against the production ENS-trained model with 24 columns, including power lags. "
            "Dot: estimate. Line: 95% interval. All three comparisons are post hoc."
        ),
        DEFINITIONS,
        SCOPE,
    ],
    "5": [
        "An extreme day and a random day at one solar farm and one wind farm in the held-out year",
        (
            "Power as a fraction of capacity against hour of day (UTC). Each forecast comes from "
            "the latest sampled ENS run before the day began."
        ),
        (
            "Black: observed. Orange: model trained on ENS weather. Blue: model trained on "
            "ERA5, applied to each of the 51 ENS members' weather and averaged."
        ),
        (
            "Extreme days: highest day-average ERA5 sunlight (solar farm A) and highest "
            "day-average ERA5 wind speed at 100 m (wind farm W1). Random days: drawn at "
            f"random from the held-out year with seed {RANDOM_DAY_SEED}. " + DEFINITIONS
        ),
    ],
}
"""Every figure's text, in one place so it can be reviewed before any figure is drawn."""


def _frame(*, name: str) -> pl.DataFrame:
    return pl.read_parquet(RESULT_DIR / f"{name}.parquet")


def _titles(*, key: str) -> alt.TitleParams:
    lines = FIGURE_TEXT[key]
    return alt.TitleParams(
        text=lines[0], subtitle=lines[1:], anchor="start", fontSize=14, subtitleFontSize=11
    )


def _arm_encoding(*, arms: list[str]) -> alt.Color:
    return alt.Color(
        "label:N",
        scale=alt.Scale(domain=[ARM_LABELS[a] for a in arms], range=[ARM_COLOURS[a] for a in arms]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
    )


def _legend(*, domain: list[str], range_: list[str]) -> alt.Color:
    return alt.Color(
        "label:N",
        scale=alt.Scale(domain=domain, range=range_),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
    )


def figure_1() -> alt.TopLevelMixin:
    """Mean absolute error by lead band, one panel per technology."""
    mae = _frame(name="mae")
    arms = ["ens_trained", "s1_memberavg", "s1_control", "s1_cams", "climatology"]
    panels = []
    for technology, label in TECHNOLOGY_LABELS.items():
        present = [
            a
            for a in arms
            if not mae.filter(
                (pl.col("arm") == a) & (pl.col("technology") == technology)
            ).is_empty()
        ]
        data = mae.filter(
            (pl.col("technology") == technology)
            & pl.col("arm").is_in(present)
            & (pl.col("band") != "all")
        ).with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
        ceiling = mae.filter(
            (pl.col("technology") == technology)
            & (pl.col("arm") == "era5_ceiling")
            & (pl.col("band") != "all")
        )
        base = alt.Chart(data).encode(
            y=alt.Y(
                "band:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)
            ),
            yOffset=alt.YOffset("label:N", sort=[ARM_LABELS[a] for a in present]),
            color=_legend(
                domain=[ARM_LABELS[a] for a in present], range_=[ARM_COLOURS[a] for a in present]
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Mean absolute error (points of capacity; smaller is better)")
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        ticks = (
            alt.Chart(ceiling)  # ty: ignore[unresolved-attribute]
            .mark_tick(color="#555555", thickness=2, size=36, aria=False)
            .encode(y=alt.Y("band:N", sort=list(BANDS)), x="value:Q")
        )
        panels.append(
            alt.layer(bars, points, ticks).properties(
                width=WIDTH - 200, height=230, title=alt.TitleParams(label, anchor="start")
            )
        )
    return (
        alt.vconcat(*panels).properties(title=_titles(key="1")).resolve_scale(color="independent")
    )


def _contrast_panels(*, data: pl.DataFrame, labels: dict[str, str], colours: list[str]) -> list:
    panels = []
    for technology, name in TECHNOLOGY_LABELS.items():
        part = data.filter(pl.col("technology") == technology)
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", sort=list(BANDS), title="Forecast lead"),
            yOffset=alt.YOffset("label:N", sort=list(labels.values())),
            color=_legend(domain=list(labels.values()), range_=colours),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Change in mean absolute error (points of capacity; more negative is better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=WIDTH - 200, height=210, title=alt.TitleParams(name, anchor="start")
            )
        )
    return panels


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts P1 and P2 by lead band."""
    contrasts = _frame(name="contrasts").filter(pl.col("planned") & (pl.col("band") != "all"))
    labels = {
        "P1 (revised)": ARM_LABELS["s1_memberavg"],
        "P2 (revised)": ARM_LABELS["ens_trained_plus_s1"],
    }
    data = contrasts.with_columns(label=pl.col("contrast").replace_strict(labels))
    panels = _contrast_panels(data=data, labels=labels, colours=[ocf.DATA_BLUE, ocf.DATA_GREEN])
    for index, panel in enumerate(panels):
        material = (
            alt.Chart(pl.DataFrame({"x": [-0.15]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#777777")
            .encode(x="x:Q")
        )
        panels[index] = alt.layer(panel, material)
    return (
        alt.vconcat(*panels).properties(title=_titles(key="2")).resolve_scale(color="independent")
    )


def figure_3() -> alt.TopLevelMixin:
    """Slope of observed on predicted power, by lead band."""
    slopes = _frame(name="slopes")
    arms = ["ens_trained", "s1_control", "s1_memberavg", "era5_ceiling"]
    labels = {**ARM_LABELS, "era5_ceiling": "Trained on ERA5, given ERA5 weather (not a forecast)"}
    colours = {**ARM_COLOURS, "era5_ceiling": "#777777"}
    panels = []
    for technology, name in TECHNOLOGY_LABELS.items():
        data = slopes.filter(
            (pl.col("technology") == technology)
            & pl.col("arm").is_in(arms)
            & (pl.col("band") != "all")
        ).with_columns(label=pl.col("arm").replace_strict(labels))
        base = alt.Chart(data).encode(
            x=alt.X("band:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("label:N", sort=[labels[a] for a in arms]),
            color=_legend(domain=[labels[a] for a in arms], range_=[colours[a] for a in arms]),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "slope:Q",
                title="Slope of observed on predicted power",
                scale=alt.Scale(zero=False),
            )
        )
        bars = base.mark_rule().encode(y="lower:Q", y2="upper:Q")  # ty: ignore[unresolved-attribute]
        one = (
            alt.Chart(pl.DataFrame({"y": [1.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(y="y:Q")
        )
        panels.append(
            alt.layer(one, bars, points).properties(
                width=WIDTH - 90, height=190, title=alt.TitleParams(name, anchor="start")
            )
        )
    return (
        alt.vconcat(*panels).properties(title=_titles(key="3")).resolve_scale(color="independent")
    )


def figure_4() -> alt.TopLevelMixin:
    """The post hoc effect of the feature lists, by lead band."""
    contrasts = _frame(name="contrasts").filter(pl.col("band") != "all")
    labels = {
        "F1 (feature lists): s1_memberavg minus s1_memberavg_calendar": (
            "ERA5-trained: calendar columns removed"
        ),
        "F2 (feature lists): ens_trained minus ens_trained_calendar": (
            "ENS-trained: calendar columns removed"
        ),
        "E13: ens_trained minus c0 (effect of dropping the production columns)": (
            "ENS-trained: 24 production columns replaced by the common columns"
        ),
    }
    data = contrasts.filter(pl.col("contrast").is_in(list(labels))).with_columns(
        label=pl.col("contrast").replace_strict(labels)
    )
    panels = _contrast_panels(
        data=data,
        labels=labels,
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
    )
    return (
        alt.vconcat(*panels).properties(title=_titles(key="4")).resolve_scale(color="independent")
    )


def _example_rows(*, technology: str, series_site: str, column: str, pick: str) -> pl.DataFrame:
    sites = inputs.site_table()
    series = sites.filter(pl.col("site") == series_site)["time_series_id"].item()
    analysis = pl.read_parquet(inputs.OUTPUT_ROOT / "inputs" / "analysis_features.parquet").filter(
        (pl.col("time_series_id") == series)
        & (pl.col("valid_time") >= inputs.VALIDATION_START)
        & (pl.col("valid_time") < pl.datetime(2026, 7, 1, time_zone="UTC"))
    )
    days = (
        analysis.with_columns(day=pl.col("valid_time").dt.date())
        .group_by("day")
        .agg(score=pl.col(column).mean(), count=pl.len())
        .filter(pl.col("count") == 48)
        .sort("score", descending=True)
    )
    day = (
        days.row(0, named=True)["day"]
        if pick == "extreme"
        else days.sample(n=1, seed=RANDOM_DAY_SEED).row(0, named=True)["day"]
    )
    rows = pl.read_parquet(SCORE_DIR / "rows_seed0.parquet").filter(
        (pl.col("time_series_id") == series)
        & (pl.col("valid_time").dt.date() == day)
        & pl.col("arm").is_in(["ens_trained", "s1_memberavg"])
    )
    latest = rows["power_fcst_init_time"].max()
    rows = rows.filter(pl.col("power_fcst_init_time") == latest)
    observed = rows.filter(pl.col("arm") == "ens_trained").select(
        "valid_time", pl.col("actual_norm").alias("value"), pl.lit("Observed").alias("series")
    )
    forecasts = rows.select(
        "valid_time",
        pl.col("point_norm").alias("value"),
        pl.col("arm").replace_strict(ARM_LABELS).alias("series"),
    )
    return pl.concat([observed, forecasts]).with_columns(
        hour=pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60.0,
        technology=pl.lit(technology),
        lead=pl.lit(float(rows["lead_hours"].min() or 0.0)),  # ty: ignore[invalid-argument-type]
    )


def figure_5() -> alt.TopLevelMixin:
    """An extreme day and a random day at the first solar farm and the first wind farm."""
    domain = ["Observed", ARM_LABELS["ens_trained"], ARM_LABELS["s1_memberavg"]]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=["#222222", ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
    )
    panels = []
    for technology, site, column, name, hours in (
        ("solar", "A", "shortwave", "solar farm A", [4, 20]),
        ("wind", "W1", "wind_speed_100m", "wind farm W1", [0, 24]),
    ):
        for pick, kind in (("extreme", "extreme day"), ("random", "random day")):
            data = _example_rows(technology=technology, series_site=site, column=column, pick=pick)
            lead = data["lead"][0]
            chart = (
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_line(clip=True, strokeWidth=2, point=False, aria=False)
                .encode(
                    x=alt.X(
                        "hour:Q",
                        title="Hour of day (UTC)",
                        scale=alt.Scale(domain=hours),
                        axis=alt.Axis(values=list(range(hours[0], hours[1] + 1, 4))),
                    ),
                    y=alt.Y("value:Q", title="Power (fraction of capacity)"),
                    color=colour,
                )
                .properties(
                    width=WIDTH - 90,
                    height=170,
                    title=alt.TitleParams(
                        f"{kind.capitalize()} at {name} (forecast made {lead:.0f} hours ahead at "
                        "the start of the day)",
                        anchor="start",
                    ),
                )
            )
            panels.append(chart)
    return alt.vconcat(*panels).properties(title=_titles(key="5")).resolve_scale(color="shared")


def main() -> None:
    """Print the figure text, or draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    if args.text_only:
        for key, lines in FIGURE_TEXT.items():
            print(f"Figure {key}")
            for line in lines:
                print(f"  {line}")
            print()
        return
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for key, build in {
        "1": figure_1,
        "2": figure_2,
        "3": figure_3,
        "4": figure_4,
        "5": figure_5,
    }.items():
        chart = build()
        chart.save(CHART_DIR / f"figure_{key}.svg")
        chart.save(CHART_DIR / f"figure_{key}.png", scale_factor=2)
        print(f"figure {key} written")


if __name__ == "__main__":
    main()
