"""Draw example predictions of experiment 00006 for two Primary substations.

The two substations are the ones whose chosen-set error (with recency weights) is the median and the
highest among the Primary series, so the figure shows a typical case and the worst case. Each
panel is one week: the observed power, the three point forecasts (weather only, weather plus
clock, and the chosen set with recency weights), and the 5% to 95% band of the chosen set's
quantile model, all as a percentage of the substation's effective capacity. Substations are
labelled by their anonymous label only.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_examples.py
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import altair as alt
import e00006_charts as charts
import e00006_report as report
import plotting.ocf_theme as ocf
import polars as pl

FOLD: Final[str] = "confirm_2025"
CHOSEN: Final[str] = "W+clock+covid_flag+xmas|recency"
WEEKS: Final[dict[str, datetime]] = {
    "a week in July 2025": datetime(2025, 7, 14, tzinfo=UTC),
    "a week in January 2026": datetime(2026, 1, 12, tzinfo=UTC),
}
"""Each week starts on a Monday at 00:00 UTC, which is also 00:00 local time in January and 01:00
in July; the x axis counts local days from Monday."""
SERIES_COLOURS: Final[dict[str, str]] = {
    "Observed": "#222222",
    "Weather only": "#999999",
    "Weather + hour and weekday": ocf.DATA_SKY,
    "Chosen set, recent years weighted more": ocf.BRAND_ORANGE,
}
WEEKDAYS: Final[str] = "['Mon','Tue','Wed','Thu','Fri','Sat','Sun']"


def pick_series() -> dict[str, int]:
    """Return the Primary series with the median and the highest chosen-set error."""
    rows = report.read_seeds(stage="finals", fold_name=FOLD).filter(pl.col("seed") == 0)
    primary = report.of_type(rows=rows, substation_type="Primary")
    errors = (
        primary.group_by("time_series_id")
        .agg(error=(pl.col(CHOSEN) - pl.col("target")).abs().mean())
        .sort("error")
    )
    return {
        "typical": int(errors["time_series_id"][errors.height // 2]),
        "worst": int(errors["time_series_id"][-1]),
    }


def week_frame(*, series_id: int, start: datetime) -> pl.DataFrame:
    """Return one series' half-hours of a week with the forecasts and the quantile band."""
    point = report.read_seeds(stage="finals", fold_name=FOLD).filter(
        pl.col("seed") == 0, pl.col("time_series_id") == series_id
    )
    quantiles = report.read_seeds(stage="finals_quantiles", fold_name=FOLD).filter(
        pl.col("seed") == 0, pl.col("time_series_id") == series_id
    )
    joined = point.select("valid_time", "target", "W", "W+clock", CHOSEN).join(
        quantiles.select("valid_time", lower=f"{CHOSEN}@q0.05", upper=f"{CHOSEN}@q0.95"),
        on="valid_time",
        how="inner",
    )
    end = start + timedelta(days=7)
    week = joined.filter(pl.col("valid_time") >= start, pl.col("valid_time") < end)
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return week.with_columns(
        day=(
            local.dt.weekday().cast(pl.Float64)
            - 1.0
            + (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0) / 24.0
        )
    ).sort("valid_time")


def panel(*, data: pl.DataFrame, title: str, show_legend: bool) -> alt.LayerChart | alt.FacetChart:
    """Draw one week: the band, then the observed power and the three forecasts."""
    long = data.unpivot(
        on=["target", "W", "W+clock", CHOSEN],
        index=["day"],
        variable_name="series",
        value_name="power",
    ).with_columns(
        series=pl.col("series").replace_strict(
            {
                "target": "Observed",
                "W": "Weather only",
                "W+clock": "Weather + hour and weekday",
                CHOSEN: "Chosen set, recent years weighted more",
            }
        ),
        power=pl.col("power") * 100.0,
    )
    band_data = data.select("day", lower=pl.col("lower") * 100.0, upper=pl.col("upper") * 100.0)
    x = alt.X(
        "day:Q",
        title="Day of the week (local time)",
        scale=alt.Scale(domain=[0, 7]),
        axis=alt.Axis(values=list(range(7)), labelExpr=f"{WEEKDAYS}[datum.value]"),
    )
    band = (
        alt.Chart(band_data)
        .mark_area(opacity=0.25, color=ocf.BRAND_ORANGE)  # ty: ignore[unresolved-attribute]
        .encode(x=x, y=alt.Y("lower:Q", title="Power (% of capacity)"), y2="upper:Q")
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.5)  # ty: ignore[unresolved-attribute]
        .encode(
            x=x,
            y=alt.Y("power:Q", title="Power (% of capacity)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(SERIES_COLOURS), range=list(SERIES_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320)
                if show_legend
                else None,
            ),
        )
    )
    return alt.layer(band, lines).properties(
        width=charts.WIDTH, height=150, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def main() -> None:
    """Draw the example-prediction figure."""
    chosen = pick_series()
    labels = pl.read_parquet(report.inputs.INPUTS_DIR / "series.parquet").select(
        "time_series_id", "site"
    )
    panels = []
    for kind, series_id in chosen.items():
        label = labels.filter(pl.col("time_series_id") == series_id)["site"][0]
        for week, start in WEEKS.items():
            panels.append(
                panel(
                    data=week_frame(series_id=series_id, start=start),
                    title=f"Primary substation {label} ({kind} error), {week}",
                    show_legend=not panels,
                )
            )
    chart = alt.vconcat(*panels).properties(
        title=charts._title(
            text=(
                "The chosen set follows a typical substation's daily and weekly shape, and misses "
                "the large swings of the worst one"
            ),
            lines=[
                (
                    "Observed power and forecasts for two primary substations, as a percentage of "
                    "each substation's capacity, on the confirmation year (2025-07 to 2026-06)."
                ),
                (
                    "The typical substation has the median error of the Primary series and the "
                    "worst has the highest. One XGBoost model per substation, trained on ERA5 "
                    "weather."
                ),
                "Band: 5% to 95% quantiles of the chosen set's quantile model.",
            ],
        )
    )
    chart.save(charts.OUT_DIR / "figure_7.svg")
    chart.save(report.RESULTS_DIR / "figure_7.png", scale_factor=2)
    print("figure 7 written")


if __name__ == "__main__":
    main()
