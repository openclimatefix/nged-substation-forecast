"""Draw Figures 15 to 18 of the quick-wins study: the input features of the production model.

Figures 15 to 17 draw the 24 production feature columns, plus the actual power, over the first 10
days of one ECMWF run for four anonymised series (a PV farm, a wind farm, a Disaggregated Demand
series, a Raw Flow series). Figure 15 holds the power lags, Figure 16 the calendar columns and six
weather columns, and Figure 17 the other six weather columns. Figure 18 summarises every column over
the whole training window and the whole validation year: the share of null or NaN values by lead
band, and the minimum, median, and maximum. All four read the cached frames `build_frames.py`
wrote, which hold the production feature engineer's output.

The run is chosen by the PV farm's actual power alone (the run with the median mean power over its
first 10 days), as in Figures 7 to 10. No calendar date is drawn: the x axis counts days since the
run's initialisation, power is a percentage of the series' effective capacity, and the two
time-of-year columns are drawn as a change from their first hour, because their values would date
the run.

The text of every figure is written to `studies/quick_wins/figure_text.txt` before any chart is
rendered.

    uv run python studies/quick_wins/charts_features.py [--text-only] [--png-dir DIR]
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import charts_examples as examples
import plotting.ocf_theme as ocf
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from studies.charts import wrapped

from studies import charts

FOLD: Final[str] = inputs.FOLD_A
PERCENT: Final[float] = 100.0
SHOWN_DAYS: Final[int] = 10
HOURS_PER_DAY: Final[int] = 24
MEMBER_STEP_HOURS: Final[int] = 6
"""The other 50 members are drawn at this spacing, which keeps the SVG small."""
MID_GREY: Final[str] = "#8C8C8C"

SERIES_TYPES: Final[tuple[str, ...]] = ("PV", "Wind", "Disaggregated Demand", "Raw Flow")
POWER_ROW: Final[str] = "power (actual)"
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "power_lag_24h",
    "power_lag_48h",
    "power_lag_168h",
    "power_lag_336h",
)
CALENDAR_COLUMNS: Final[tuple[str, ...]] = (
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "local_time_of_year_sin",
    "local_time_of_year_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    "local_day_of_week",
    "local_utc_offset_minutes",
)
WEATHER_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "windchill",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_short_wave_radiation_flux_surface",
    "categorical_precipitation_type_surface",
)
FEATURES: Final[tuple[str, ...]] = (*LAG_COLUMNS, *CALENDAR_COLUMNS, *WEATHER_COLUMNS)
"""The 24 production columns, in the order the figures draw them."""
HIDDEN_VALUE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"local_time_of_year_sin", "local_time_of_year_cos"}
)
"""Columns whose y-axis values would date the run, so the panel draws the shape without them."""
DAY_NAMES: Final[tuple[str, ...]] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

BAND_EDGES_HOURS: Final[tuple[int, int]] = (72, 264)
BAND_LABELS: Final[tuple[str, ...]] = ("days 0-2", "days 3-10", "days 11+")

CONTROL: Final[str] = "Control member (member 0)"
OTHER: Final[str] = "The other 50 members"
POWER: Final[str] = "Actual power, and power lagged by the column's own lag"
WITHHELD: Final[str] = "Hours when the lag column is null (withheld)"
ROLE_COLOURS: Final[dict[str, str]] = {
    POWER: ocf.BLACK_1,
    CONTROL: ocf.DATA_BLUE,
    OTHER: ocf.DATA_SKY,
    WITHHELD: MID_GREY,
}

PLOT_WIDTH_PX: Final[int] = 100
PLOT_HEIGHT_PX: Final[int] = 34


assert set(FEATURES) == set(inputs.production_config().selected_features), (
    "FEATURES must be the production configuration's 24 selected features"
)
assert len(FEATURES) == 24


@dataclass(frozen=True)
class Figure:
    """A figure's text and the chart drawn from it."""

    text: examples.FigureText
    chart: alt.TopLevelMixin


def _val_path(*, init_time: datetime) -> Path:
    return inputs.OUTPUT_ROOT / "frames" / FOLD / "val" / f"init={init_time:%Y-%m-%d}.parquet"


def choose_run(*, pv_series_id: int) -> datetime:
    """Return the run whose PV farm has the median mean actual power over its first 10 days."""
    rows = []
    for init_time in inputs.sample_init_times(window=inputs.fold_window()):
        mean_power = (
            pl.scan_parquet(_val_path(init_time=init_time))
            .filter(
                pl.col("time_series_id") == pv_series_id,
                pl.col("ensemble_member") == 0,
                (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_hours()
                < SHOWN_DAYS * HOURS_PER_DAY,
            )
            .select(pl.col("power").mean())
            .collect()
            .item()
        )
        rows.append((init_time, mean_power))
    ranked = sorted(rows, key=lambda row: (row[1], row[0]))
    return ranked[len(ranked) // 2][0]


def series_to_draw(*, context: examples.Context) -> dict[str, tuple[int, str]]:
    """Return, per series type, the series id and the anonymous label Figures 7 to 10 use."""
    chosen = {}
    for series_type in SERIES_TYPES:
        series_id = examples.choose_series(context=context, series_type=series_type)
        chosen[series_type] = (
            series_id,
            examples.series_label(context=context, series_id=series_id, series_type=series_type),
        )
    return chosen


def _to_plain_numbers(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Turn the day-of-week enum into 0 (Monday) to 6 (Sunday) and every column into Float64."""
    return frame.with_columns(
        pl.col("local_day_of_week")
        .cast(pl.String)
        .replace_strict(
            {name: float(i) for i, name in enumerate(DAY_NAMES)}, return_dtype=pl.Float64
        )
    ).with_columns(pl.col(*FEATURES, "power").cast(pl.Float64))


def run_rows(*, init_time: datetime, series_ids: list[int]) -> pl.DataFrame:
    """Read one run's rows for the chosen series, with power as a fraction of capacity.

    Args:
        init_time: The ECMWF run's initialisation time.
        series_ids: The series to keep.

    Returns:
        The cached frame's rows within 10 days of the run's initialisation time, with
        `lead_hours` (since the ECMWF initialisation time) and the power columns divided by
        each series' effective capacity.
    """
    capacity = pl.DataFrame(inputs.effective_capacity(settings=Settings())).select(
        "time_series_id", "effective_capacity_mw"
    )
    frame = (
        pl.scan_parquet(_val_path(init_time=init_time))
        .filter(pl.col("time_series_id").is_in(series_ids))
        .select(
            "time_series_id",
            "valid_time",
            "nwp_init_time",
            "power_fcst_init_time",
            "ensemble_member",
            "power",
            *FEATURES,
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_minutes() / 60
        )
        .filter(pl.col("lead_hours") <= SHOWN_DAYS * HOURS_PER_DAY)
        .collect()
    )
    frame = _to_plain_numbers(frame=frame).join(capacity, on="time_series_id", how="left")
    if frame["effective_capacity_mw"].null_count():
        msg = "a drawn series has no effective capacity"
        raise ValueError(msg)
    scale = PERCENT / pl.col("effective_capacity_mw")
    return frame.with_columns(pl.col("power", *LAG_COLUMNS) * scale).sort(
        "time_series_id", "ensemble_member", "valid_time"
    )


X_AXIS_TITLE: Final[str] = "Days since run start"
CHANGE_SUFFIX: Final[str] = " (change)"
POWER_ROWS: Final[tuple[str, ...]] = (POWER_ROW, *LAG_COLUMNS)
OTHER_ROWS: Final[tuple[str, ...]] = (*CALENDAR_COLUMNS, *WEATHER_COLUMNS)
SLOW_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "dew_point_temperature_2m",
        "windchill",
        "wind_speed_100m",
        "wind_direction_100m",
    }
)
SMOOTH_COLUMNS: Final[frozenset[str]] = frozenset(
    {"pressure_surface", "pressure_reduced_to_mean_sea_level", "geopotential_height_500hpa"}
)
"""Columns whose other 50 members are drawn at four times `MEMBER_STEP_HOURS`."""
"""Columns whose other 50 members are drawn at twice `MEMBER_STEP_HOURS`, to keep the SVG small.

Each is either smooth or close to another column that is drawn at the finer step."""
CONTROL_ONLY: Final[frozenset[str]] = frozenset({"categorical_precipitation_type_surface"})
"""A 0-or-1 column draws as noise when 51 members overlap, so only the control member is drawn."""


def member_step_hours(*, feature: str) -> int:
    """Return the spacing, in hours, at which a column's other 50 members are drawn."""
    if feature in SMOOTH_COLUMNS:
        return 4 * MEMBER_STEP_HOURS
    return MEMBER_STEP_HOURS * (2 if feature in SLOW_COLUMNS else 1)


def row_label(*, feature: str) -> str:
    """Return the label of a feature's row; the hidden-value columns say they show a change."""
    return feature + CHANGE_SUFFIX if feature in HIDDEN_VALUE_COLUMNS else feature


def withheld_start_days(*, rows: pl.DataFrame, lag_hours: int) -> float | None:
    """Return the lead, in days since the ECMWF run, from which a power lag is withheld.

    A lag is null once it is no longer than the lead since the forecast's initialisation time,
    which lies `power_fcst_init_time - nwp_init_time` after the ECMWF run's initialisation.

    Args:
        rows: One series' control-member rows, with `power_fcst_init_time` and `nwp_init_time`.
        lag_hours: The lag of the power-lag column, in hours.

    Returns:
        The day at which the lag column becomes null, or `None` if that day is past the last
        drawn day.
    """
    delay_hours = float(
        float((rows["power_fcst_init_time"] - rows["nwp_init_time"]).dt.total_minutes().min())  # ty: ignore[invalid-argument-type]
        / 60
    )
    start = (delay_hours + lag_hours) / HOURS_PER_DAY
    return start if start < SHOWN_DAYS else None


def _marks(*, one: pl.DataFrame, label: str, feature: str, role: str, member: int) -> pl.DataFrame:
    """Return one panel's line points for one member."""
    column = "power" if feature == POWER_ROW else feature
    values = pl.col(column)
    if feature in HIDDEN_VALUE_COLUMNS:
        values = values - values.drop_nulls().first()
    return one.select(
        pl.lit(label).alias("series"),
        pl.lit(row_label(feature=feature)).alias("feature"),
        pl.lit(role).alias("role"),
        pl.lit(member, dtype=pl.Int64).alias("member"),
        (pl.col("lead_hours") / HOURS_PER_DAY).alias("days"),
        values.alias("value"),
        pl.lit(None, dtype=pl.Float64).alias("days_end"),
    )


def long_form(
    *, rows: pl.DataFrame, labels: dict[int, str], features: tuple[str, ...], with_members: bool
) -> pl.DataFrame:
    """Return the points and withheld-lag spans every panel draws, one row per mark.

    The columns are `series`, `feature`, `role`, `member`, `days`, `value`, and `days_end` (set
    on the withheld spans only). The control member draws every feature, and the other 50 members
    draw the weather features when `with_members` is set.
    """
    frames = []
    for series_id, label in labels.items():
        mine = rows.filter(pl.col("time_series_id") == series_id)
        control = mine.filter(pl.col("ensemble_member") == 0)
        for feature in features:
            role = POWER if feature in POWER_ROWS else CONTROL
            frames.append(_marks(one=control, label=label, feature=feature, role=role, member=0))
            if feature in LAG_COLUMNS:
                start = withheld_start_days(
                    rows=control, lag_hours=int(feature.split("_")[-1][:-1])
                )
                if start is not None:
                    frames.append(
                        pl.DataFrame(
                            {
                                "series": label,
                                "feature": row_label(feature=feature),
                                "role": WITHHELD,
                                "member": 0,
                                "days": start,
                                "value": 0.0,
                                "days_end": float(SHOWN_DAYS),
                            }
                        )
                    )
        if not with_members:
            continue
        for feature in features:
            if feature not in WEATHER_COLUMNS or feature in CONTROL_ONLY:
                continue
            step = member_step_hours(feature=feature)
            sparse = mine.filter(pl.col("ensemble_member") > 0, (pl.col("lead_hours") % step) == 0)
            for member, one in sparse.group_by("ensemble_member", maintain_order=True):
                frames.append(
                    _marks(one=one, label=label, feature=feature, role=OTHER, member=int(member[0]))
                )
    return pl.concat(frames, how="vertical_relaxed")


def small_multiples(
    *, data: pl.DataFrame, features: tuple[str, ...], series_order: list[str], roles: list[str]
) -> charts.Panel:
    """Draw the small multiples: one row per feature, one panel column per series."""
    colour = alt.Color(
        "role:N",
        title=None,
        scale=alt.Scale(domain=roles, range=[ROLE_COLOURS[role] for role in roles]),
        legend=alt.Legend(columns=1, symbolStrokeWidth=3, symbolType="stroke", labelLimit=400),
    )
    x_scale = alt.Scale(domain=[0, SHOWN_DAYS], nice=False)
    x = alt.X(
        "days:Q",
        title=X_AXIS_TITLE,
        scale=x_scale,
        axis=alt.Axis(values=[0, 5, 10], grid=False, format="d", labelFontSize=8, titleFontSize=9),
    )
    y_scale = alt.Scale(zero=features[0] == POWER_ROW)
    y_axis = alt.Axis(title=None, tickCount=2, labelFontSize=8, grid=False, format=".3~g")
    y = alt.Y("value:Q", title=None, axis=y_axis, scale=y_scale)
    withheld = (
        alt.Chart()
        .mark_rect(opacity=0.35, aria=False)
        .encode(x=alt.X("days:Q", scale=x_scale), x2="days_end:Q", color=colour)  # ty: ignore[unresolved-attribute]
        .transform_filter(alt.datum.role == WITHHELD)
    )
    others = (
        alt.Chart()
        .mark_line(strokeWidth=0.4, opacity=0.5, aria=False, clip=True)
        .encode(x=x, y=y, detail="member:N", color=colour)  # ty: ignore[unresolved-attribute]
        .transform_filter(alt.datum.role == OTHER)
    )
    control = (
        alt.Chart()
        .mark_line(strokeWidth=1.2, aria=False, clip=True)
        .encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
        .transform_filter((alt.datum.role == CONTROL) | (alt.datum.role == POWER))
    )
    panel = alt.layer(withheld, others, control).properties(
        width=PLOT_WIDTH_PX, height=PLOT_HEIGHT_PX
    )
    return panel.facet(
        row=alt.Row(
            "feature:N",
            sort=[row_label(feature=f) for f in features],
            title=None,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
                labelOrient="left",
                labelFontSize=9,
                labelLimit=170,
                labelPadding=4,
            ),
        ),
        column=alt.Column(
            "series:N",
            sort=series_order,
            title=None,
            header=alt.Header(labelOrient="top", labelFontSize=10, labelAnchor="start"),
        ),
        data=data,
        spacing={"row": 4, "column": 6},
    ).resolve_scale(y="independent")


def _stack(*, chart: charts.Panel, text: examples.FigureText) -> alt.TopLevelMixin:
    """Put one chart of small multiples under the figure's caption."""
    return charts.figure(
        panels=[chart],
        number=text.number,
        title=text.title,
        subtitle=text.subtitle,
        figure_planning=None,
    )


def _run_text(*, init_time: datetime) -> str:
    return (
        "One ECMWF ensemble run from the validation year, the median run by the PV farm's mean "
        "actual power over its first 10 days."
    )


def figure_15(*, labels: dict[int, str], rows: pl.DataFrame, init_time: datetime) -> Figure:
    """Build Figure 15: the actual power and the four power-lag columns."""
    data = long_form(rows=rows, labels=labels, features=POWER_ROWS, with_members=False)
    title = "A power-lag column is the actual power shifted by its lag, and is null beyond its lag"
    subtitle = [
        (
            f"{_run_text(init_time=init_time)} Top row: the actual power. Rows below: the four "
            "power-lag columns the production XGBoost model is given, each the actual power "
            "from 24, 48, 168, or 336 hours before the valid time."
        ),
        (
            "Power is a percentage of each series' effective capacity. A grey band marks the "
            "hours when the column is null, because the lag is no longer than the lead from the "
            "forecast's initialisation time (9 hours after the ECMWF run's)."
        ),
    ]
    text = examples.FigureText(
        number=15,
        title=title,
        subtitle=subtitle,
        panel_titles=[f"Column: {label}" for label in labels.values()],
        x_title=X_AXIS_TITLE,
        y_title="(none: the row label names the column)",
        legend=[POWER, WITHHELD],
        notes=[f"Row labels: {', '.join(POWER_ROWS)}."],
    )
    chart = small_multiples(
        data=data,
        features=POWER_ROWS,
        series_order=list(labels.values()),
        roles=[POWER, WITHHELD],
    )
    return Figure(text=text, chart=_stack(chart=chart, text=text))


SHARED_CELL_LABEL: Final[str] = "Generator weather"
SHARED_CELL_NOTE: Final[str] = (
    "The solar and wind generators share a weather grid cell, so one column serves both."
)
FIRST_WEATHER_ROWS: Final[tuple[str, ...]] = (
    *CALENDAR_COLUMNS,
    "temperature_2m",
    "dew_point_temperature_2m",
    "windchill",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
)
SECOND_WEATHER_ROWS: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "downward_short_wave_radiation_flux_surface",
    "categorical_precipitation_type_surface",
)
assert set(FIRST_WEATHER_ROWS) | set(SECOND_WEATHER_ROWS) == set(OTHER_ROWS)


def _weather_figure(
    *,
    number: int,
    title: str,
    subtitle: list[str],
    features: tuple[str, ...],
    labels: dict[int, str],
    rows: pl.DataFrame,
    init_time: datetime,
) -> Figure:
    """Build a figure of calendar or weather rows, with all 51 ensemble members."""
    shown = {
        series_id: SHARED_CELL_LABEL if label == "PV farm A" else label
        for series_id, label in labels.items()
        if label != "Wind farm W3"
    }
    data = long_form(rows=rows, labels=shown, features=features, with_members=True)
    text = examples.FigureText(
        number=number,
        title=title,
        subtitle=[_run_text(init_time=init_time), *subtitle],
        panel_titles=[f"Column: {label}" for label in shown.values()],
        x_title=X_AXIS_TITLE,
        y_title="(none: the row label names the column)",
        legend=[CONTROL, OTHER],
        notes=[
            f"Row labels: {', '.join(row_label(feature=f) for f in features)}.",
            "The other 50 members are drawn every "
            + "; ".join(
                f"{member_step_hours(feature=f)} hours for {f}"
                for f in features
                if f in WEATHER_COLUMNS and f not in CONTROL_ONLY
            )
            + ".",
        ],
    )
    chart = small_multiples(
        data=data, features=features, series_order=list(shown.values()), roles=[CONTROL, OTHER]
    )
    return Figure(text=text, chart=_stack(chart=chart, text=text))


def figure_16(*, labels: dict[int, str], rows: pl.DataFrame, init_time: datetime) -> Figure:
    """Build Figure 16: the calendar columns, then temperature, pressure, and height."""
    return _weather_figure(
        number=16,
        title="Calendar columns follow local time; temperature and pressure columns vary by member",
        subtitle=[
            (
                "Rows, top to bottom: the eight calendar columns, then six weather columns, as "
                "the production XGBoost model is given them. Units: degrees Celsius, pascals, and "
                "metres (geopotential height). Each panel has its own y axis."
            ),
            (
                f"{SHARED_CELL_NOTE} The two time-of-year rows show the change from the first "
                "hour, because their values would date the run."
            ),
        ],
        features=FIRST_WEATHER_ROWS,
        labels=labels,
        rows=rows,
        init_time=init_time,
    )


def figure_17(*, labels: dict[int, str], rows: pl.DataFrame, init_time: datetime) -> Figure:
    """Build Figure 17: wind, radiation, and precipitation type."""
    return _weather_figure(
        number=17,
        title="Wind, radiation, and precipitation-type columns, for all 51 ensemble members",
        subtitle=[
            (
                "The six remaining weather columns: wind speed in metres per second, wind "
                "direction in degrees, downward short-wave radiation in watts per square metre, "
                "and precipitation type (a category code). Each panel has its own y axis."
            ),
            f"{SHARED_CELL_NOTE} Precipitation type draws the control member only.",
        ],
        features=SECOND_WEATHER_ROWS,
        labels=labels,
        rows=rows,
        init_time=init_time,
    )


def _as_numbers(*, columns: tuple[str, ...]) -> list[pl.Expr]:
    return [pl.col(c).to_physical().cast(pl.Float64).alias(c) for c in columns]


def _missing(*, column: str) -> pl.Expr:
    return pl.col(column).is_null() | pl.col(column).is_nan()


def _band() -> pl.Expr:
    lead = (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_minutes() / 60
    return (
        pl.when(lead < BAND_EDGES_HOURS[0])
        .then(0)
        .when(lead < BAND_EDGES_HOURS[1])
        .then(1)
        .otherwise(2)
        .alias("band")
    )


def summary_table() -> pl.DataFrame:
    """Return, per feature, the null share by lead band and the range, for training and validation.

    The training frame holds the control member only, as production trains. The validation frames
    hold every ensemble member of the 53 weekly runs. Null shares count null and NaN values.

    Returns:
        One row per feature and dataset (`train` or `validation`) with `null_band0` to
        `null_band2`, `min`, `median`, and `max`. The range ignores null and NaN.
    """
    frames = {
        "train": pl.scan_parquet(inputs.OUTPUT_ROOT / "frames" / FOLD / "train.parquet"),
        "validation": pl.scan_parquet(inputs.OUTPUT_ROOT / "frames" / FOLD / "val" / "*.parquet"),
    }
    out = []
    for name, lazy in frames.items():
        base = lazy.select(
            "valid_time", "nwp_init_time", *_as_numbers(columns=FEATURES)
        ).with_columns(_band())
        shares = (
            base.group_by("band")
            .agg(*[_missing(column=c).mean().alias(c) for c in FEATURES])
            .collect(engine="streaming")
            .sort("band")
        )
        clean = base.select(
            *[pl.col(c).fill_nan(None) for c in FEATURES],
        )
        ranges = clean.select(
            *[pl.col(c).min().alias(f"{c}|min") for c in FEATURES],
            *[pl.col(c).median().alias(f"{c}|median") for c in FEATURES],
            *[pl.col(c).max().alias(f"{c}|max") for c in FEATURES],
        ).collect(engine="streaming")
        for feature in FEATURES:
            row = {
                "dataset": name,
                "feature": feature,
                **{f"null_band{b}": float(shares[feature][b]) for b in range(3)},
                **{k: float(ranges[f"{feature}|{k}"][0]) for k in ("min", "median", "max")},
            }
            out.append(row)
    return pl.DataFrame(out)


SUMMARY_FILE: Final[Path] = inputs.OUTPUT_ROOT / "report" / "features_summary.parquet"


def load_summary() -> pl.DataFrame:
    """Read the cached summary table, building it first if it is not on disk."""
    if not SUMMARY_FILE.exists():
        SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        summary_table().write_parquet(SUMMARY_FILE)
    return pl.read_parquet(SUMMARY_FILE)


def _shade(*, share: float) -> str:
    """Return the hex colour between white (no nulls) and brand orange (all null)."""
    orange = [int(ocf.BRAND_ORANGE[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * share):02x}" for c in orange)


def _number(*, value: float) -> str:
    return f"{value:.4g}"


def figure_18(*, summary: pl.DataFrame) -> Figure:
    """Build Figure 18: the null share by lead band, and the range of every column."""
    heat = []
    for row in summary.iter_rows(named=True):
        for band in range(3):
            share = row[f"null_band{band}"]
            heat.append(
                {
                    "feature": row["feature"],
                    "column": f"{'Training' if row['dataset'] == 'train' else 'Validation'}, "
                    f"{BAND_LABELS[band]}",
                    "share": share,
                    "shade": _shade(share=share),
                    "label": "0"
                    if share == 0
                    else f"{share * PERCENT:.0f}%"
                    if share >= 0.005
                    else "<1%",
                }
            )
    heat_frame = pl.DataFrame(heat)
    heat_columns = [f"{d}, {b}" for d in ("Training", "Validation") for b in BAND_LABELS]
    ranges = []
    for row in summary.iter_rows(named=True):
        dataset = "Training" if row["dataset"] == "train" else "Validation"
        ranges.extend(
            {
                "feature": row["feature"],
                "column": f"{dataset} {key}",
                "label": "hidden"
                if row["feature"].startswith("power_lag")
                else _number(value=row[key]),
            }
            for key in ("min", "median", "max")
        )
    range_frame = pl.DataFrame(ranges)
    range_columns = [
        f"{d} {k}" for d in ("Training", "Validation") for k in ("min", "median", "max")
    ]
    y = alt.Y(
        "feature:N",
        sort=list(FEATURES),
        title=None,
        axis=alt.Axis(labelFontSize=9, labelLimit=190, domain=False, ticks=False),
    )
    x_axis = alt.Axis(labelAngle=-40, labelFontSize=9, orient="top", title=None, ticks=False)
    heat_base = alt.Chart(heat_frame).encode(
        x=alt.X("column:N", sort=heat_columns, axis=x_axis), y=y
    )
    heat_chart = alt.layer(
        heat_base.mark_rect(aria=False).encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color("shade:N", scale=None)
        ),
        heat_base.mark_text(fontSize=8, aria=False).encode(text="label:N"),  # ty: ignore[unresolved-attribute]
    ).properties(width=6 * 56, height=len(FEATURES) * 16)
    range_base = alt.Chart(range_frame).encode(
        x=alt.X("column:N", sort=range_columns, axis=x_axis),
        y=alt.Y(
            "feature:N",
            sort=list(FEATURES),
            title=None,
            axis=alt.Axis(labelFontSize=9, labelLimit=190, domain=False, ticks=False),
        ),
    )
    range_chart = (
        range_base.mark_text(fontSize=8, aria=False)
        .encode(text="label:N")  # ty: ignore[unresolved-attribute]
        .properties(width=6 * 56, height=len(FEATURES) * 16)
    )
    heat_title = (
        "Share of values that are null or NaN, by lead band (days 0-2: under 72 hours; days 3-10: "
        "72 to 263 hours; days 11+: 264 hours and more)"
    )
    range_title = "Smallest, median, and largest value, in the column's own unit"
    title = (
        "Half the training rows lack the precipitation type; the power lags are null past their lag"
    )
    subtitle = [
        (
            "All 31 series. Training: the control member of the 15-month training window. "
            "Validation: all 51 ensemble members of the 53 weekly runs, July 2025 to June 2026 "
            "(fold A). The upper panel shades each cell by its share of null or NaN values: "
            "white is none and orange is all."
        ),
        (
            "The precipitation type is null in the first 7 months of the training window, because "
            "the stored ECMWF runs of those months carry none. A power lag is null once its lag "
            "is no longer than the lead, and also wherever the power history has a gap (6% at "
            "336 hours, days 0-2)."
        ),
        (
            "The lower panel gives each column's minimum, median, and maximum. The power lags "
            "are not shown, because their range in megawatts would reveal single generators' sizes."
        ),
    ]
    text = examples.FigureText(
        number=18,
        title=title,
        subtitle=subtitle,
        panel_titles=[heat_title, range_title],
        x_title="(none: the column headings name the dataset and lead band or statistic)",
        y_title="(none: the row label names the column)",
        legend=["(none: the shade is the share, printed in each cell)"],
        notes=[],
    )
    heat_panel = heat_chart.properties(
        title=alt.TitleParams(wrapped(text=heat_title, width=95), anchor="start", fontSize=11)
    )
    range_panel = range_chart.properties(
        title=alt.TitleParams(range_title, anchor="start", fontSize=11)
    )
    chart = charts.figure(
        panels=[heat_panel, range_panel],
        number=18,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )
    return Figure(text=text, chart=chart)


def main() -> None:
    """Write the figure text, then render the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    context = examples.load_context(fold=FOLD, seed=0)
    chosen = series_to_draw(context=context)
    labels = dict(chosen.values())
    init_time = choose_run(pv_series_id=chosen["PV"][0])
    rows = run_rows(init_time=init_time, series_ids=list(labels))
    figures = {
        15: figure_15(labels=labels, rows=rows, init_time=init_time),
        16: figure_16(labels=labels, rows=rows, init_time=init_time),
        17: figure_17(labels=labels, rows=rows, init_time=init_time),
        18: figure_18(summary=load_summary()),
    }
    examples.write_text_sections(texts=[f.text for f in figures.values()])
    print(f"wrote {examples.TEXT_FILE}")
    if args.text_only:
        return
    for number, figure in figures.items():
        print(examples.save(chart=figure.chart, number=number))
        if args.png_dir is not None:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            figure.chart.save(args.png_dir / f"quick_wins_figure_{number}.png", scale_factor=2)


if __name__ == "__main__":
    main()
