"""Draw experiment 00094's figures from the saved results, as SVG.

1. The contrasts with intervals (headline): `ht` and `mono` against `ref`, by slice.
2. Each arm's quantile score and q95 exceedance share, cold against mild winter hours.
3. Mean signed error of the median by lead band in cold hours.
4. Block B: `amh` minus `am` by slice.
5. The fitted heating term over a cold week at one series, beside the temperature.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00094-*/e00094_charts.py
"""

from datetime import UTC, datetime
from typing import Final

import altair as alt
import e00094_common as common
import e00094_fit as fitting
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 560
SCOPE: Final[str] = (
    "16 Primary substations, 365 daily forecast runs, July 2025 to June 2026. Monthly refits on "
    "the latest 12 months."
)
ARM_COLOURS: Final[dict[str, str]] = {
    "ref": ocf.DATA_BLUE_LIGHT,
    "ht": ocf.BRAND_ORANGE,
    "mono": ocf.DATA_GREEN,
}
ARM_LABELS: Final[dict[str, str]] = {
    "ref": "Reference (no heating term)",
    "ht": "Heating term in the margin",
    "mono": "Monotone temperature constraint",
}
SLICE_LABELS: Final[dict[str, str]] = {
    "all": "All rows",
    "cold": "Cold hours, all leads",
    "cold_short": "Cold hours, up to 48 h",
    "cold_long": "Cold hours, over 48 h",
    "mild_winter": "Mild winter hours",
}
WEEK_START: Final[datetime] = datetime(2026, 1, 3, tzinfo=UTC)
WEEK_END: Final[datetime] = datetime(2026, 1, 10, tzinfo=UTC)


def figure_1() -> alt.TopLevelMixin:
    """Contrasts with intervals, the pre-registered four first."""
    data = pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet").with_columns(
        kind=pl.when(pl.col("registered"))
        .then(pl.lit("Pre-registered (98.75% interval)"))
        .otherwise(pl.lit("Exploratory (95% interval)"))
    )
    order = data["contrast"].to_list()
    base = alt.Chart(data).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Pre-registered (98.75% interval)", "Exploratory (95% interval)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None, labelLimit=300, columns=1 if False else 3),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Quantile score, arm minus reference (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            "Quantile score of each arm against the reference",
            subtitle=[
                SCOPE,
                "Dot: estimate. Line: interval from re-drawing whole ISO weeks of valid time.",
                "Rule at zero: same score. Below zero: the arm is better than the reference.",
            ],
            anchor="start",
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Quantile score and q95 exceedance by arm, cold against mild winter hours."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "slices.parquet")
        .filter(pl.col("slice").is_in(["cold_short", "cold_long", "mild_winter"]))
        .with_columns(
            slice_label=pl.col("slice").replace_strict(SLICE_LABELS),
            arm_label=pl.col("arm").replace_strict(ARM_LABELS),
        )
    )
    order = [SLICE_LABELS[s] for s in ("cold_short", "cold_long", "mild_winter")]
    panels = []
    for column, title in (
        ("qs", "Quantile score (points of capacity; smaller is better)"),
        ("above95", "Share of observations above the q95 forecast (target 0.05)"),
    ):
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("slice_label:N", sort=order, title=None),
                yOffset=alt.YOffset("arm_label:N", sort=list(ARM_LABELS.values())),
                x=alt.X(f"{column}:Q", title=title),
                color=alt.Color(
                    "arm_label:N",
                    scale=alt.Scale(
                        domain=list(ARM_LABELS.values()),
                        range=[ARM_COLOURS[a] for a in ARM_LABELS],
                    ),
                    legend=alt.Legend(
                        orient="top", title=None, labelLimit=300, columns=1 if False else 3
                    ),
                ),
            )
            .properties(width=WIDTH, height=130)
        )
        panels.append(chart)
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Cold hours score worse than mild winter hours, and q95 is exceeded too often",
            subtitle=[SCOPE, "Cold: ensemble-mean forecast 2 m temperature below 3 C."],
            anchor="start",
        )
    )


def figure_3() -> alt.TopLevelMixin:
    """Mean signed error of the median by lead band, cold hours."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "lead_bias.parquet")
        .filter(pl.col("slice") == "cold")
        .with_columns(arm_label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    bands = [label for label, _low, _high in common.LEAD_BANDS]
    lines = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", sort=bands, title="Lead (hours or days after the forecast origin)"),
            y=alt.Y(
                "bias:Q",
                title="Observed minus median forecast (points of capacity; 0 is unbiased)",
            ),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(
                    domain=list(ARM_LABELS.values()), range=[ARM_COLOURS[a] for a in ARM_LABELS]
                ),
                legend=alt.Legend(
                    orient="top", title=None, labelLimit=300, columns=1 if False else 3
                ),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#555555").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, lines).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            "In cold hours the median forecast is too low, most at long leads",
            subtitle=[SCOPE, "Positive means the observed demand was higher than the forecast."],
            anchor="start",
        ),
    )


def figure_4() -> alt.TopLevelMixin:
    """Block B: the solar-plus-heating arm minus the solar arm, by slice."""
    data = pl.read_parquet(common.RESULTS_DIR / "b_contrasts.parquet").with_columns(
        label=pl.col("slice").replace_strict(
            {
                "all": "All rows",
                "cold": "Cold hours, all leads",
                "cold_short": "Cold hours, up to 48 h",
                "cold_long": "Cold hours, over 48 h",
                "mild_winter": "Mild winter hours",
                "april_to_september": "April to September",
            }
        )
    )
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=data["label"].to_list(), title=None))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Quantile score, solar plus heating minus solar (points)")
    )
    bars = base.mark_rule(color=ocf.BRAND_ORANGE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "Block B: does the heating term add to the solar term?",
            subtitle=[
                "5 Primary substations with a solar term, 73 runs, 3 seeds, one fit to June 2025.",
                "Dot: estimate. Line: 95% interval from re-drawing whole ISO weeks.",
                "Below zero: the solar-plus-heating arm is better.",
            ],
            anchor="start",
        ),
    )


def figure_5(*, site: str = "D3") -> alt.TopLevelMixin:
    """The fitted heating term over a cold week, beside the forecast temperature."""
    series = common.series_table().filter(pl.col("site") == site)
    series_id = series["time_series_id"][0]
    coefficients = pl.read_parquet(
        common.WORK_DIR / "coefficients" / "ht" / f"{site}_20260101T0000.parquet"
    )["coefficient"].to_numpy()
    arrays = common.load_series_arrays(series=series)[site]
    temperature = (
        pl.read_parquet(
            common.ERA5_PATH, columns=["time_series_id", "valid_time", "temperature_2m"]
        )
        .filter(
            (pl.col("time_series_id") == series_id)
            & (pl.col("valid_time") >= WEEK_START)
            & (pl.col("valid_time") < WEEK_END)
        )
        .sort("valid_time")
    )
    hours = temperature["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
    degree_hours = fitting.heating_degree_hours(
        temperature=temperature["temperature_2m"].to_numpy()
    )
    index = np.asarray(common.grid_index(times=temperature["valid_time"]))
    observed = common.take(values=arrays.power, index=index)
    table = temperature.with_columns(
        term=pl.Series(coefficients[hours] * degree_hours * 100.0),
        observed=pl.Series(observed * 100.0),
    )
    temp_chart = (
        alt.Chart(table)
        .mark_line(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("valid_time:T", title=None),
            y=alt.Y("temperature_2m:Q", title="2 m temperature (C)"),
        )
        .properties(width=WIDTH, height=110)
    )
    power_chart = alt.layer(
        alt.Chart(table)
        .mark_line(color=ocf.DATA_BLUE_LIGHT)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("valid_time:T", title="Valid time (UTC)"), y="observed:Q"
        ),
        alt.Chart(table)
        .mark_line(color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x="valid_time:T",
            y=alt.Y("term:Q", title="Observed power and heating term (% of capacity)"),
        ),
    ).properties(width=WIDTH, height=160)
    return alt.vconcat(temp_chart, power_chart).properties(
        title=alt.TitleParams(
            "The fitted heating term follows the temperature and the daily shape",
            subtitle=[
                f"Substation {site}, 3 to 9 January 2026, coefficients from the 1 January refit.",
                "Light blue: observed power. Orange: the heating term added to the margin.",
            ],
            anchor="start",
        )
    )


def main() -> None:
    """Write the figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
        ("figure_4", figure_4()),
        ("figure_5", figure_5()),
    ):
        chart.save(common.CHART_DIR / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
