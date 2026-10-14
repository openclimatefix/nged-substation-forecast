"""Draw example power predictions over the Christmas period with several input configurations.

The figure covers 20 December 2025 to 3 January 2026, in the confirmation fold, from the
Christmas-control predictions (one XGBoost model per substation, averaged over three fitting
seeds). The upper panel is Primary substation P12, the median-error series, and the lower panel
is the mean over the 14 Primary series, both as a percentage of capacity. A middle panel shows
the Primary series that gains most from the windowed Christmas column over the window. Every
configuration adds columns to the weather block plus the clock plus the COVID indicator.

    uv run python \
        ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_christmas_examples.py
"""

from datetime import UTC, datetime
from typing import Final

import altair as alt
import e00006_charts as charts
import e00006_report as report
import plotting.ocf_theme as ocf
import polars as pl

FOLD: Final[str] = "confirm_2025"
START: Final[datetime] = datetime(2025, 12, 20, tzinfo=UTC)
END: Final[datetime] = datetime(2026, 1, 4, tzinfo=UTC)
SUBSTATION: Final[str] = "P12"
BASE: Final[str] = "W+clock+covid_flag"
SERIES: Final[dict[str, tuple[str, str]]] = {
    "target": ("Observed", "#222222"),
    BASE: ("Base: weather, hour, weekday, COVID indicator", "#999999"),
    f"{BASE}+xmas": ("Base + days to Christmas", ocf.BRAND_ORANGE),
    f"{BASE}+xmas_window": ("Base + days to Christmas, 14-day window", ocf.DATA_SKY),
    f"{BASE}+hol_flags": ("Base + holiday flags", ocf.DATA_BLUE),
    f"{BASE}+season_half": ("Base + half-year flag", ocf.DATA_PURPLE),
}


PLOTTED: Final[tuple[str, ...]] = ("target", BASE, f"{BASE}+xmas_window")
"""The lines drawn. The other configurations are scored in the printed errors only."""


def christmas_frame() -> pl.DataFrame:
    """Return per-series half-hourly rows over the window with each configuration's forecast."""
    rows = report.read_seeds(stage="christmas_check", fold_name=FOLD)
    arms = [a for a in SERIES if a != "target"]
    averaged = rows.group_by("time_series_id", "valid_time").agg(
        pl.col("target").first(), pl.col(arms).mean()
    )
    keep = report.of_type(rows=averaged, substation_type="Primary")
    return keep.filter(pl.col("valid_time") > START, pl.col("valid_time") <= END)


def long_format(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame as one row per time, series name, and power in percent of capacity."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.replace_time_zone(None)
    return (
        frame.unpivot(
            on=list(PLOTTED), index="valid_time", variable_name="series", value_name="power"
        )
        .with_columns(
            time=local,
            power=pl.col("power") * 100.0,
            series=pl.col("series").replace_strict({k: v[0] for k, v in SERIES.items()}),
        )
        .select("time", "series", "power")
    )


def panel(*, data: pl.DataFrame, title: str, show_legend: bool) -> alt.LayerChart | alt.FacetChart:
    """Draw one panel of lines."""
    chart = (
        alt.Chart(data)
        .mark_line(strokeWidth=1.5, opacity=0.7)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                title="Local date (midnight at each tick)",
                axis=alt.Axis(format="%d %b", values=_ticks(), labelAngle=-45),
            ),
            y=alt.Y("power:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[SERIES[a][0] for a in PLOTTED], range=[SERIES[a][1] for a in PLOTTED]
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320)
                if show_legend
                else None,
            ),
        )
    )
    return chart.properties(
        width=charts.LINE_WIDTH,
        height=150,
        title=alt.TitleParams(title, anchor="start", fontSize=12),
    )


def _ticks() -> list[dict[str, int]]:
    """Return a tick at midnight of every second day from 20 December to 3 January."""
    days = [(12, d) for d in range(20, 32, 2)] + [(1, d) for d in (1, 3)]
    return [
        {"year": 2025 if month == 12 else 2026, "month": month, "date": day} for month, day in days
    ]


def mean_absolute_error(*, data: pl.DataFrame, arm: str) -> float:
    """Return one configuration's mean absolute error in points of capacity."""
    return abs(data[arm].to_numpy() - data["target"].to_numpy()).mean() * 100.0


def largest_gain_series(*, frame: pl.DataFrame) -> int:
    """Return the Primary series whose error the windowed Christmas column lowers most."""
    gains = {
        int(series_id): mean_absolute_error(data=part, arm=BASE)
        - mean_absolute_error(data=part, arm=f"{BASE}+xmas_window")
        for (series_id,), part in frame.group_by("time_series_id")
    }
    return max(gains, key=lambda k: gains[k])


def main() -> None:
    """Draw the Christmas example figure and print the windowed errors."""
    frame = christmas_frame()
    labels = pl.read_parquet(report.inputs.INPUTS_DIR / "series.parquet")
    typical = labels.filter(pl.col("site") == SUBSTATION)["time_series_id"].item()
    gain_id = largest_gain_series(frame=frame)
    gain_label = labels.filter(pl.col("time_series_id") == gain_id)["site"].item()
    mean = frame.group_by("valid_time").agg(pl.exclude("time_series_id").mean()).sort("valid_time")
    panels = {
        f"Primary substation {SUBSTATION} (the median-error series)": frame.filter(
            pl.col("time_series_id") == typical
        ).sort("valid_time"),
        f"Primary substation {gain_label} (the series the windowed Christmas column helps most)": (
            frame.filter(pl.col("time_series_id") == gain_id).sort("valid_time")
        ),
        "Mean of the 14 Primary substations": mean,
    }
    for name, data in panels.items():
        errors = {
            SERIES[a][0]: round(mean_absolute_error(data=data, arm=a), 2)
            for a in SERIES
            if a != "target"
        }
        print(name, "mean absolute error over the window:", errors)
    chart = alt.vconcat(
        *[
            panel(data=long_format(frame=data), title=name, show_legend=index == 0)
            for index, (name, data) in enumerate(panels.items())
        ]
    ).properties(
        title=charts._title(
            text=(
                "A column that knows about Christmas follows the substation "
                "that drops over Christmas"
            ),
            lines=[
                (
                    "Observed power and forecasts for 20 December 2025 to 3 January 2026, as a "
                    "percentage of capacity, from one XGBoost model per substation trained on "
                    "ERA5 weather. Mean of three fitting seeds. The lowest panel averages the "
                    "forecasts and the observed power over the 14 series. Lines are slightly "
                    "transparent, so overlaps show darker."
                )
            ],
        )
    )
    chart.save(charts.OUT_DIR / "figure_13.svg")
    chart.save(report.RESULTS_DIR / "figure_13.png", scale_factor=2)
    print("figure 13 written")


if __name__ == "__main__":
    main()
