"""Draw experiment 00006's figures from the results tables, as SVG.

Reads the tables `e00006_report.py` wrote and writes the figures to
`docs/ml-experiments/00006-stage1a-demand-input-feature-sweep/`. No series identifier, name, or
coordinate appears on any figure; series are never labelled individually.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00006_report as report
import plotting.ocf_theme as ocf
import polars as pl
from contracts.settings import PROJECT_ROOT

OUT_DIR: Final[Path] = (
    PROJECT_ROOT / "docs" / "ml-experiments" / "00006-stage1a-demand-input-feature-sweep"
)
WIDTH: Final[int] = 440
"""The plot area's width; with the labels, the figure fills the docs text column."""
FROZEN: Final[str] = "W+clock+covid_flag+xmas"

ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology (month, half-hour, weekend)",
    "W": "Weather block only",
    "W+clock": "Weather + hour and weekday",
    FROZEN: "Chosen set (+ COVID flag, Christmas days)",
    f"{FROZEN}|recency": "Chosen set, recent years weighted more",
    f"{FROZEN}|from2023-09": "Chosen set, trained from 2023-09",
}
ARM_ORDER: Final[list[str]] = list(ARM_LABELS.values())
GROUP_LABELS: Final[dict[str, str]] = {
    "covid_flag": "COVID indicator (0/1)",
    "hol_flags": "Holiday flags (day, before, after)",
    "xmas": "Days to Christmas",
    "hol_nearest": "Days to nearest holiday",
    "hol_name": "Holiday name",
    "bridge": "Bridge-day flag",
    "azimuth_sincos": "Sun azimuth sine and cosine",
    "sun_day": "Day length, hours since sunrise",
    "windchill": "Windchill",
    "wind_heat_loss": "Wind-driven heat loss",
    "utc_offset": "UTC offset",
    "hdh": "Heating degree-hours (24 h, 72 h)",
    "t24": "Mean temperature, previous 24 h",
    "t72": "Mean temperature, previous 72 h",
    "t7d": "Mean temperature, previous 7 days",
    "ewm_t": "Weighted temperature (12 h, 72 h, 7 d)",
    "doy1": "Time of year, 1 harmonic",
    "doy2": "Time of year, 2 harmonics",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "Training history": ocf.BRAND_ORANGE,
    "Calendar": ocf.DATA_BLUE,
    "Weather memory": ocf.DATA_PURPLE,
    "Sun and wind": ocf.DATA_GREEN,
}
KIND_OF_GROUP: Final[dict[str, str]] = {
    "covid_flag": "Training history",
    "hol_flags": "Calendar",
    "xmas": "Calendar",
    "hol_nearest": "Calendar",
    "hol_name": "Calendar",
    "bridge": "Calendar",
    "doy1": "Calendar",
    "doy2": "Calendar",
    "utc_offset": "Calendar",
    "azimuth_sincos": "Sun and wind",
    "sun_day": "Sun and wind",
    "windchill": "Sun and wind",
    "wind_heat_loss": "Sun and wind",
    "hdh": "Weather memory",
    "t24": "Weather memory",
    "t72": "Weather memory",
    "t7d": "Weather memory",
    "ewm_t": "Weather memory",
}
CHANGE_TITLE: Final[str] = (
    "Change in mean absolute error (points of capacity; more negative is better)"
)
SCOPE_SELECTION: Final[str] = (
    "13 primary substations, validation year 2024-07 to 2025-06, one XGBoost model per substation "
    "trained on ERA5 weather (ECMWF's estimate of past weather)."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(
        text=text, subtitle=lines, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _read(name: str) -> pl.DataFrame:
    return pl.read_parquet(report.RESULTS_DIR / f"{name}.parquet")


def _dots(
    *,
    data: pl.DataFrame,
    y: str,
    order: list[str],
    x_title: str,
    colour: str | None = None,
    colour_field: str | None = None,
    scale: alt.Scale | None = None,
    height: int | None = None,
) -> alt.LayerChart | alt.FacetChart:
    """Draw estimate dots with interval rules; `value`, `lower`, and `upper` are the columns."""
    x_scale = scale or alt.Scale(zero=False)
    colour_encoding = (
        alt.Color(
            f"{colour_field}:N",
            scale=alt.Scale(domain=list(KIND_COLOURS), range=list(KIND_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        )
        if colour_field
        else alt.value(colour or ocf.DATA_BLUE)
    )
    base = alt.Chart(data).encode(
        y=alt.Y(f"{y}:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=colour_encoding,
    )
    rules = base.mark_rule().encode(x=alt.X("lower:Q", title=x_title, scale=x_scale), x2="upper:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(x=alt.X("value:Q", scale=x_scale))  # ty: ignore[unresolved-attribute]
    return alt.layer(rules, points).properties(width=WIDTH, height=height or 24 * len(order))


def figure_1() -> alt.TopLevelMixin:
    """Headline: absolute error on the confirmation year, by arm, one panel per type."""
    levels = _read("finals_levels_confirm_2025").filter(pl.col("arm").is_in(list(ARM_LABELS)))
    panels = []
    titles = {
        "Primary": "Primary substations (the headline)",
        "BSP": "Bulk supply points (2 series)",
        "GSP": "Grid supply points (2 series)",
    }
    for scope, title in titles.items():
        data = levels.filter(pl.col("scope") == scope).with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS)
        )
        panel = _dots(
            data=data,
            y="label",
            order=ARM_ORDER,
            x_title="Mean absolute error of the median (points of capacity; smaller is better)",
        ).properties(title=alt.TitleParams(title, anchor="start", fontSize=12))
        panels.append(panel)
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text=(
                    "Hour, weekday, the COVID indicator, and days to Christmas lower the error "
                    "for primary substations and grid supply points; recent years help further"
                ),
                lines=[
                    (
                        "Error on the confirmation year (2025-07 to 2026-06), 1 point = 1% of a "
                        "substation's capacity, from one XGBoost model per substation."
                    ),
                    "The chosen set was fixed on 2024-07 to 2025-06 and not changed afterwards.",
                    (
                        "Dot: estimate over three fitting seeds. Line: 95% interval from "
                        "re-drawing whole months."
                    ),
                ],
            )
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Contrasts on the confirmation year, by substation type."""
    contrasts = _read("finals_contrasts_confirm_2025")
    labels = {
        "P1 clock": "Hour and weekday, against weather only",
        "P8 frozen vs clock": "Chosen set, against weather + hour and weekday",
        "recency": "Recent years weighted more, against chosen set",
        "P9 from 2023-09": "Trained from 2023-09, against chosen set",
    }
    data = contrasts.filter(pl.col("contrast").is_in(list(labels))).with_columns(
        label=pl.col("contrast").replace_strict(labels), value=pl.col("difference")
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    panels = []
    for scope in ("Primary", "BSP", "GSP"):
        sub = data.filter(pl.col("scope") == scope)
        panels.append(
            alt.layer(
                zero,
                _dots(
                    data=sub,
                    y="label",
                    order=list(labels.values()),
                    x_title=CHANGE_TITLE,
                    scale=alt.Scale(zero=True),
                ),
            ).properties(title=alt.TitleParams(scope, anchor="start", fontSize=12))
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text=(
                    "Hour and weekday, recent-year weighting, and a 2023 start lower the error for "
                    "primary substations; the chosen-set step is weaker"
                ),
                lines=[
                    "Paired change in error on the confirmation year, 2025-07 to 2026-06.",
                    (
                        "The chosen-set and 2023-09 contrasts use 99.4% intervals (nine planned "
                        "contrasts); the others use 95%."
                    ),
                    "Dot: estimate. Line: interval from re-drawing whole months. Rule: no change.",
                ],
            )
        )
    )


def figure_3() -> alt.TopLevelMixin:
    """Add-one-in effects against weather + hour and weekday, Primary, selection year."""
    table = _read("add_one_in_selection_fold").filter(
        pl.col("scope") == "Primary", pl.col("reference") == "W+clock"
    )
    rows = []
    for row in table.iter_rows(named=True):
        group = row["treatment"].removeprefix("W+clock+")
        if group in GROUP_LABELS:
            rows.append(
                {
                    "label": GROUP_LABELS[group],
                    "kind": KIND_OF_GROUP[group],
                    "value": row["difference"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                }
            )
    treatments = {
        "W+clock|recency": ("Recent years weighted more", "Training history"),
        "W+clock|nocovid": ("Lockdown months dropped from training", "Training history"),
    }
    for row in table.iter_rows(named=True):
        if row["treatment"] in treatments:
            label, kind = treatments[row["treatment"]]
            rows.append(
                {
                    "label": label,
                    "kind": kind,
                    "value": row["difference"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                }
            )
    data = pl.DataFrame(rows).sort("value")
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = alt.layer(
        zero,
        _dots(
            data=data,
            y="label",
            order=data["label"].to_list(),
            x_title="Change in mean absolute error (points of capacity; more negative is better)",
            colour_field="kind",
            scale=alt.Scale(zero=True),
            height=20 * data.height,
        ),
    )
    return chart.properties(
        title=_title(
            text=(
                "Only the training-history choices and some holiday columns help primary "
                "substations; the temperature-history and windchill columns do not"
            ),
            lines=[
                (
                    "Change in error from adding one set of columns (or one training choice) to "
                    "weather + hour and weekday."
                ),
                SCOPE_SELECTION,
                (
                    "Dot: estimate, one fitting seed. Line: 95% interval from re-drawing whole "
                    "months. Rule: no change."
                ),
            ],
        )
    )


def figure_4() -> alt.TopLevelMixin:
    """Change in error from adding time of year, with and without the season-carrying columns."""
    table = _read("masking_contrasts").filter(pl.col("scope") == "Primary")
    variants = {
        "": "Full weather block",
        "-solar_zenith-solar_azimuth": "Without sun position",
        "-temperature_2m-dew_point_temperature_2m": "Without temperature and dew point",
        "-solar_zenith-solar_azimuth-temperature_2m-dew_point_temperature_2m": (
            "Without sun position, temperature, and dew point"
        ),
    }
    encodings = {
        "+doy1": "Sine and cosine, 1 harmonic",
        "+doy2": "Sine and cosine, 2 harmonics",
        "+month_cat": "Month (categorical)",
    }
    rows = [
        {
            "variant": variant,
            "encoding": encoding,
            "value": row["difference"],
            "lower": row["lower"],
            "upper": row["upper"],
        }
        for row in table.iter_rows(named=True)
        for drop, variant in variants.items()
        for suffix, encoding in encodings.items()
        if row["treatment"] == f"W{drop}+clock{suffix}"
    ]
    data = pl.DataFrame(rows)
    colours = [ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN]
    base = alt.Chart(data).encode(
        y=alt.Y(
            "variant:N", sort=list(variants.values()), title=None, axis=alt.Axis(labelLimit=330)
        ),
        yOffset=alt.YOffset("encoding:N", sort=list(encodings.values())),
        color=alt.Color(
            "encoding:N",
            scale=alt.Scale(domain=list(encodings.values()), range=colours),
            legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=260),
        ),
    )
    rules = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Change in mean absolute error (points of capacity; more negative is better)",
            scale=alt.Scale(zero=True),
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=50).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, rules, points).properties(
        width=WIDTH,
        height=190,
        title=_title(
            text=(
                "Time of year helps only after the weather block loses both sun position and "
                "temperature"
            ),
            lines=[
                (
                    "Change in error from adding a time-of-year encoding, with hour and weekday "
                    "always included."
                ),
                SCOPE_SELECTION,
                (
                    "Dot: estimate over three fitting seeds. Line: 95% interval from re-drawing "
                    "whole months. Rule: no change."
                ),
            ],
        ),
    )


def figure_5() -> alt.TopLevelMixin:
    """Importance of each column for the chosen set, across the Primary series."""
    table = _read("importance_series_confirm_2025").filter(
        pl.col("substation_type") == "Primary", pl.col("arm") == f"{FROZEN}|recency"
    )
    order = table.sort("permutation_median", descending=True)["column"].to_list()
    base = alt.Chart(table).encode(
        y=alt.Y("column:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    )

    def panel(*, median: str, low: str, high: str, title: str) -> alt.LayerChart | alt.FacetChart:
        bars = base.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{low}:Q", title=title), x2=f"{high}:Q"
        )
        points = base.mark_point(filled=True, color=ocf.DATA_BLUE, size=50).encode(  # ty: ignore[unresolved-attribute]
            x=f"{median}:Q"
        )
        return alt.layer(bars, points).properties(width=WIDTH, height=240)

    gain = panel(
        median="gain_median",
        low="gain_q1",
        high="gain_q3",
        title="Share of the model's total split gain (larger means used more)",
    )
    permutation = panel(
        median="permutation_median",
        low="permutation_q1",
        high="permutation_q3",
        title="Rise in error when the column is shuffled (points of capacity)",
    )
    return alt.vconcat(gain, permutation).properties(
        title=_title(
            text=(
                "Hour of day, temperature, and sunlight carry most of the model; the added "
                "columns matter little"
            ),
            lines=[
                (
                    "Importance of each input in the chosen set with recency weights, on the "
                    "confirmation year (2025-07 to 2026-06)."
                ),
                (
                    "One XGBoost model per primary substation. Dot: median over substations. "
                    "Line: first to third quartile across substations."
                ),
            ],
        )
    )


def figure_6() -> alt.TopLevelMixin:
    """Mean absolute error by month of the year, with and without time of year."""
    table = _read("by_month_hour_planned_select_2024").filter(
        pl.col("scope") == "Primary", pl.col("by") == "month"
    )
    labels = {"W+clock": "No time of year", "W+clock+doy1": "Sine and cosine of time of year"}
    data = table.filter(pl.col("arm").is_in(list(labels))).with_columns(
        label=pl.col("arm").replace_strict(labels)
    )
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "bucket:O",
                title="Month of the year (1 is January)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("mae:Q", title="Mean absolute error (points of capacity; smaller is better)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(labels.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=320),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="Error peaks in autumn and early winter, with or without time-of-year columns",
                lines=[
                    (
                        "Mean absolute error of the median by month, with hour and weekday in "
                        "the model."
                    ),
                    SCOPE_SELECTION,
                ],
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, build in enumerate(
        (figure_1, figure_2, figure_3, figure_4, figure_5, figure_6), start=1
    ):
        chart = build()
        chart.save(OUT_DIR / f"figure_{number}.svg")
        chart.save(report.RESULTS_DIR / f"figure_{number}.png", scale_factor=2)
        print(f"figure {number} written")


if __name__ == "__main__":
    main()
