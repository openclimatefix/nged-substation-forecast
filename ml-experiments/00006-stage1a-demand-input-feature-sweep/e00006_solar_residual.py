"""Check whether the daytime errors of the Primary substations follow solar irradiance.

The residual is the observed power minus the chosen set's median forecast (with recency weights), in
points of capacity, averaged per hour. It is compared with CAMS satellite-derived irradiance
downloaded at each substation's own location, and with the ERA5 shortwave there, for July 2025.
Figure 8 plots Primary substation P12's residual against each irradiance, figure 9 shows its
residual by hour of day in the summer months of the confirmation year, and figure 10 shows the
variance explained for every Primary series. Figure 11 repeats the July week of figure 7 for P12
with CAMS irradiance below it. The script prints the correlations and the variance explained, with
series labelled by their anonymous label only.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_solar_residual.py
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import altair as alt
import e00006_charts as charts
import e00006_examples as examples
import e00006_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import PER_STUDY_DIR, STUDIES_DATA_DIR

SUBSTATION: Final[str] = "P12"
CAMS_PATH: Final[str] = str(
    STUDIES_DATA_DIR
    / "downloads"
    / "reanalysis"
    / "CAMS"
    / "demand_points"
    / "cams_demand_points.parquet"
)
DEMAND_SERIES_PATH: Final[str] = str(
    PER_STUDY_DIR / "stage1_pilot" / "demand" / "inputs" / "series.parquet"
)
MAX_ZENITH_DEGREES: Final[float] = 80.0
"""Hours with the sun lower than 10 degrees above the horizon are left out of the July analysis."""
JULY_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
JULY_END: Final[datetime] = datetime(2025, 8, 1, tzinfo=UTC)


def demand_labels() -> pl.DataFrame:
    """Return each series' `time_series_id` with its anonymous label here and in experiment 00003.

    Returns:
        One row per series with `time_series_id`, `site` (P1 to P16, B1, B2, G1, G2), and
        `demand_site` (D1 to D22), the label the CAMS download at the demand locations uses.
    """
    mine = pl.read_parquet(report.inputs.INPUTS_DIR / "series.parquet").select(
        "time_series_id", "site", "substation_type"
    )
    demand = pl.read_parquet(DEMAND_SERIES_PATH).select("time_series_id", demand_site="site")
    return mine.join(demand, on="time_series_id", how="inner", validate="1:1")


def hourly_residuals() -> pl.DataFrame:
    """Return one row per series and hour with the residual, ERA5 shortwave, and CAMS irradiance.

    CAMS and the half-hourly valid times both use the period-ending stamp, so the half-hours
    stamped 12:30 and 13:00 are averaged into the hour stamped 13:00.
    """
    cams = pl.read_parquet(CAMS_PATH).select(demand_site="site", hour="time", cams_ghi="ghi_w_m2")
    point = report.read_seeds(stage="finals", fold_name=examples.FOLD).filter(pl.col("seed") == 0)
    features = pl.read_parquet(report.inputs.INPUTS_DIR / "features.parquet").select(
        "time_series_id", "valid_time", "shortwave", "solar_zenith"
    )
    return (
        point.select("time_series_id", "valid_time", "target", examples.CHOSEN)
        .join(features, on=["time_series_id", "valid_time"])
        .with_columns(
            residual=(pl.col("target") - pl.col(examples.CHOSEN)) * 100.0,
            hour=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1),
        )
        .group_by("time_series_id", "hour")
        .agg(
            residual=pl.col("residual").mean(),
            era5_shortwave=pl.col("shortwave").mean(),
            zenith=pl.col("solar_zenith").mean(),
        )
        .join(demand_labels(), on="time_series_id")
        .join(cams, on=["demand_site", "hour"], how="left")
    )


def variance_explained(*, frame: pl.DataFrame, columns: list[str]) -> float:
    """Return the R squared of a least-squares fit of the residual on the given columns."""
    design = np.column_stack([np.ones(frame.height), *[frame[c].to_numpy() for c in columns]])
    target = frame["residual"].to_numpy()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(1 - ((target - design @ beta) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def july_daylight(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Return the daylight hours of July 2025 that have a CAMS value."""
    return hourly.filter(
        pl.col("hour") > JULY_START,
        pl.col("hour") <= JULY_END,
        pl.col("zenith") < MAX_ZENITH_DEGREES,
        pl.col("cams_ghi").is_not_null(),
    )


def scores_by_series(*, july: pl.DataFrame) -> pl.DataFrame:
    """Return each Primary series' correlations and variance explained in July 2025."""
    rows = []
    for label, frame in july.filter(pl.col("substation_type") == "Primary").group_by("site"):
        rows.append(
            {
                "site": label[0],
                "hours": frame.height,
                "corr_cams": frame.select(pl.corr("residual", "cams_ghi")).item(),
                "corr_era5": frame.select(pl.corr("residual", "era5_shortwave")).item(),
                "r2_era5": variance_explained(frame=frame, columns=["era5_shortwave"]),
                "r2_cams": variance_explained(frame=frame, columns=["cams_ghi"]),
                "r2_both": variance_explained(frame=frame, columns=["era5_shortwave", "cams_ghi"]),
            }
        )
    return pl.DataFrame(rows).sort(pl.col("site").str.slice(1).cast(pl.Int32))


def scatter_panel(
    *, frame: pl.DataFrame, column: str, title: str
) -> alt.LayerChart | alt.FacetChart:
    """Draw the residual against one irradiance column, with a least-squares line."""
    points = (
        alt.Chart(frame)
        .mark_circle(size=22, opacity=0.5, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{column}:Q", title=title),
            y=alt.Y("residual:Q", title="Observed minus forecast (points of capacity)"),
        )
    )
    line = points.transform_regression(column, "residual").mark_line(
        color=ocf.BRAND_ORANGE, strokeWidth=2
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#222222").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(points, line, zero).properties(width=charts.LINE_WIDTH, height=190)


def figure_8(*, july: pl.DataFrame) -> alt.TopLevelMixin:
    """Return substation P12's residual against CAMS irradiance and against ERA5 shortwave."""
    frame = july.filter(pl.col("site") == SUBSTATION)
    cams_r = frame.select(pl.corr("residual", "cams_ghi")).item()
    era5_r = frame.select(pl.corr("residual", "era5_shortwave")).item()
    top = scatter_panel(
        frame=frame, column="cams_ghi", title="CAMS irradiance at the substation (W/m²)"
    ).properties(
        title=alt.TitleParams(f"CAMS: correlation {cams_r:.2f}", anchor="start", fontSize=12)
    )
    bottom = scatter_panel(
        frame=frame, column="era5_shortwave", title="ERA5 shortwave at the substation (W/m²)"
    ).properties(
        title=alt.TitleParams(f"ERA5: correlation {era5_r:.2f}", anchor="start", fontSize=12)
    )
    return alt.vconcat(top, bottom).properties(
        title=charts._title(
            text="On bright July days the forecast for a typical substation is too high",
            lines=[
                (
                    f"Hourly residual (observed minus forecast) of Primary substation "
                    f"{SUBSTATION} in July 2025, daylight hours only, against CAMS "
                    "satellite-derived irradiance and ERA5 shortwave at the substation's "
                    "location. Dots: hours. Line: least-squares fit. Below zero: the forecast "
                    "is too high."
                )
            ],
        )
    )


def figure_9(*, hourly: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the mean residual and the mean absolute residual by hour of day in summer."""
    by_hour = (
        hourly.filter(
            pl.col("site") == SUBSTATION,
            (pl.col("hour") - pl.duration(minutes=30)).dt.month().is_in([6, 7, 8]),
        )
        .with_columns(
            local_hour=(pl.col("hour") - pl.duration(minutes=30))
            .dt.convert_time_zone("Europe/London")
            .dt.hour()
        )
        .group_by("local_hour")
        .agg(bias=pl.col("residual").mean(), error=pl.col("residual").abs().mean())
        .sort("local_hour")
        .unpivot(on=["bias", "error"], index="local_hour", variable_name="measure")
        .with_columns(
            measure=pl.col("measure").replace_strict(
                {
                    "bias": "Mean residual (negative: forecast too high)",
                    "error": "Mean absolute residual",
                }
            )
        )
    )
    colours = {
        "Mean residual (negative: forecast too high)": ocf.BRAND_ORANGE,
        "Mean absolute residual": ocf.DATA_BLUE,
    }
    chart = (
        alt.Chart(by_hour)
        .mark_line(strokeWidth=2, point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "local_hour:Q",
                title="Hour of the day (local time, start of the hour)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("value:Q", title="Points of capacity"),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
        )
        .properties(width=charts.LINE_WIDTH, height=220)
    )
    return chart.properties(
        title=charts._title(
            text="The forecast error is largest in daylight, when solar generation is present",
            lines=[
                (
                    f"Primary substation {SUBSTATION}, every hour of June, July, and August in "
                    "the confirmation year (July and August 2025, June 2026). The residual is "
                    "observed power minus the chosen set's forecast, in points of capacity."
                )
            ],
        )
    )


def figure_10(*, scores: pl.DataFrame) -> alt.TopLevelMixin:
    """Return each Primary series' variance explained by ERA5, by CAMS, and by both."""
    long = scores.unpivot(
        on=["r2_era5", "r2_cams", "r2_both"], index="site", variable_name="inputs"
    ).with_columns(
        inputs=pl.col("inputs").replace_strict(
            {"r2_era5": "ERA5 only", "r2_cams": "CAMS only", "r2_both": "ERA5 and CAMS"}
        ),
        value=pl.col("value") * 100.0,
    )
    colours = {
        "ERA5 only": "#999999",
        "CAMS only": ocf.DATA_BLUE,
        "ERA5 and CAMS": ocf.BRAND_ORANGE,
    }
    order = scores.sort("r2_both", descending=True)["site"].to_list()
    chart = (
        alt.Chart(long)
        .mark_circle(size=70, opacity=0.9)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X("value:Q", title="Share of the residual's variance explained (%)"),
            color=alt.Color(
                "inputs:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        .properties(width=charts.LINE_WIDTH, height=300)
    )
    return chart.properties(
        title=charts._title(
            text="CAMS irradiance explains more of the daytime error than ERA5 shortwave does",
            lines=[
                (
                    "Share of the variance of the hourly residual (observed minus forecast) "
                    "explained by a least-squares fit on each irradiance, for each of 14 Primary "
                    "substations, in daylight hours of July 2025. Both irradiances are read at "
                    "the substation's location."
                )
            ],
        )
    )


def figure_11() -> alt.TopLevelMixin:
    """Return P12's week in July 2025 from figure 7, with CAMS irradiance in a panel below."""
    start = examples.WEEKS["a week in July 2025"]
    series_id = demand_labels().filter(pl.col("site") == SUBSTATION)["time_series_id"].item()
    demand_site = demand_labels().filter(pl.col("site") == SUBSTATION)["demand_site"].item()
    upper = examples.panel(
        data=examples.week_frame(series_id=series_id, start=start),
        title=f"Primary substation {SUBSTATION}: observed power and forecasts",
        show_legend=True,
    )
    local = pl.col("time").dt.convert_time_zone("Europe/London")
    irradiance = (
        pl.read_parquet(CAMS_PATH)
        .filter(
            pl.col("site") == demand_site,
            pl.col("time") > start,
            pl.col("time") <= start + timedelta(days=7),
        )
        .with_columns(
            day=(
                local.dt.weekday().cast(pl.Float64)
                - 1.0
                + (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0)
                / 24.0
            ),
        )
        .with_columns(
            day=pl.when(pl.col("day") == 0.0).then(7.0).otherwise(pl.col("day")),
        )
        .select(
            "day",
            **{
                "CAMS irradiance": pl.col("ghi_w_m2"),
                "CAMS irradiance if the sky were clear": pl.col("clear_sky_ghi_w_m2"),
            },
        )
        .unpivot(index="day", variable_name="series", value_name="irradiance")
    )
    colours = {
        "CAMS irradiance": ocf.BRAND_ORANGE,
        "CAMS irradiance if the sky were clear": "#999999",
    }
    lower = (
        alt.Chart(irradiance)
        .mark_line(strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Day of the week (local time)",
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(values=list(range(7)), labelExpr=f"{examples.WEEKDAYS}[datum.value]"),
            ),
            y=alt.Y("irradiance:Q", title="Irradiance (W/m²)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320),
            ),
        )
        .properties(width=charts.LINE_WIDTH, height=130)
    )
    return (
        alt.vconcat(upper, lower)
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text="The forecast is too high on the sunniest days of the week",
                lines=[
                    (
                        f"Primary substation {SUBSTATION} for a week in July 2025, as a "
                        "percentage of capacity, with satellite-derived irradiance at the "
                        "substation below. Compare the observed power with the forecasts on the "
                        "days when the irradiance comes close to the clear-sky line. Band: 5% to "
                        "95% quantiles of the chosen set's quantile model."
                    )
                ],
            )
        )
    )


def main() -> None:
    """Print the numbers the write-up quotes and draw figures 8 to 11."""
    hourly = hourly_residuals()
    july = july_daylight(hourly=hourly)
    print(f"{july.filter(pl.col('site') == SUBSTATION).height} daylight hours for {SUBSTATION}")
    scores = scores_by_series(july=july)
    print(scores.with_columns(pl.exclude("site", "hours").round(3)))
    print(
        "mean over Primary series: "
        + ", ".join(f"{c} {scores[c].mean():.3f}" for c in scores.columns[2:])
    )
    print(
        "median R squared: "
        + ", ".join(f"{c} {scores[c].median():.3f}" for c in scores.columns if c.startswith("r2"))
    )
    cams_better = int((scores["r2_cams"] > scores["r2_era5"]).sum())
    print(f"CAMS alone explains more than ERA5 alone in {cams_better} of {scores.height} series")
    frame = july.filter(pl.col("site") == SUBSTATION)
    print("CAMS with ERA5 for P12:", frame.select(pl.corr("cams_ghi", "era5_shortwave")).item())
    summer = hourly.filter(
        pl.col("site") == SUBSTATION,
        (pl.col("hour") - pl.duration(minutes=30)).dt.month().is_in([6, 7, 8]),
    ).with_columns(
        local=(pl.col("hour") - pl.duration(minutes=30))
        .dt.convert_time_zone("Europe/London")
        .dt.hour()
    )
    day = summer.filter(pl.col("local").is_between(8, 16))
    night = summer.filter(~pl.col("local").is_between(6, 20))
    print(
        f"{SUBSTATION} summer residual 08:00-16:59: {day['residual'].mean():.2f}, night: "
        f"{night['residual'].mean():.2f}; MAE day {day['residual'].abs().mean():.2f}, "
        f"night {night['residual'].abs().mean():.2f}"
    )
    for name, chart in {
        "figure_10": figure_10(scores=scores),
        "figure_11": figure_11(),
        "figure_8": figure_8(july=july),
        "figure_9": figure_9(hourly=hourly),
    }.items():
        chart.save(charts.OUT_DIR / f"{name}.svg")
        chart.save(report.RESULTS_DIR / f"{name}.png", scale_factor=2)
    print("figures 8 to 11 written")


if __name__ == "__main__":
    main()
