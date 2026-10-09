"""Draw the battery study's rung 5 figures (10 to 13).

Every figure is built from the tables the rung scripts saved, except Figure 10, which rebuilds the
two schedules from the day-ahead prices because the saved series hold only the fitted aggregate.
The batteries, prices, and B1610 are public, so the figures name each battery and show dates and
megawatts.

Run after `battery_rung5.py` and `battery_rung5_summary.py`:
`uv run python studies/battery_pv_separation/battery_charts_rung5.py`.
"""

from datetime import UTC, datetime, timedelta
from typing import Final, cast

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from battery_charts import (
    ASSETS_DIR,
    PANEL_HEIGHT_PX,
    PLOT_WIDTH_PX,
    SUMMER_WEEK_START,
    Layer,
    _arm_dot_panel,
    line_panel,
)
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR, battery_output
from battery_rung5 import LP_CYCLES_PER_DAY, schedule_for
from battery_rung5_summary import SKIES
from battery_synthetic import day_ahead_on_grid, window_half_hours
from studies.battery_dispatch import state_of_charge
from studies.charts import figure

EXAMPLE_DAY_START: Final[datetime] = datetime(2025, 12, 16, tzinfo=UTC)
"""The Tuesday of the study's winter week; the example covers it and the next day."""
EXAMPLE_ENERGY_HOURS: Final[float] = 2.0
LP_ETA_ONE_WAY: Final[float] = 0.92
LP_SOC_MIN: Final[float] = 0.05
CHART_LABELS: Final[dict[str, str]] = {
    "A0": "A0: solar only",
    "A1": "A1: price level and rank",
    "A5": "A5: rank-rule schedule",
    "A6": "A6: linear-programme schedule",
    "A3": "A3: true battery (oracle)",
    "A4": "A4: price level and rank, week-old",
    "A5c": "A5c: rank-rule, week-old prices",
    "A6c": "A6c: linear-programme, week-old prices",
}
FIGURE_ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A5", "A6", "A3", "A4", "A5c", "A6c")
CONTRIBUTION_ARMS: Final[dict[str, str]] = {
    "A1": "Price level and rank (A1)",
    "A5": "Rank-rule schedule (A5)",
    "A6": "Linear-programme schedule (A6)",
    "A3": "True battery as regressor (A3)",
}
SCHEDULE_COLOURS: Final[dict[str, str]] = {
    "Real battery (Lakeside)": ocf.BLACK_1,
    "Rank-rule schedule (2 h)": ocf.BRAND_ORANGE,
    "Linear-programme schedule (2 h)": ocf.DATA_BLUE,
    "Day-ahead price": ocf.DATA_GREEN,
    "State of charge of the linear-programme schedule": ocf.DATA_PURPLE,
}
CONTRIBUTION_COLOURS: Final[dict[str, str]] = {
    "True battery": ocf.BLACK_1,
    "Price level and rank (A1)": ocf.DATA_PURPLE,
    "Rank-rule schedule (A5)": ocf.BRAND_ORANGE,
    "Linear-programme schedule (A6)": ocf.DATA_BLUE,
    "True battery as regressor (A3)": ocf.DATA_GREEN,
}
CONTRIBUTION_DASHES: Final[dict[str, list[int]]] = {
    "True battery": [4, 3],
    "Price level and rank (A1)": [1, 0],
    "Rank-rule schedule (A5)": [1, 0],
    "Linear-programme schedule (A6)": [6, 3],
    "True battery as regressor (A3)": [2, 2],
}
SCHEDULES_TITLE: Final[str] = (
    "A real battery trades in many small steps, while price-only schedules use full power in "
    "a few blocks"
)
CONTRIBUTION_TITLE: Final[str] = (
    "Price-only schedules fit the real battery at about a fifth of its size and follow it no "
    "better than unconstrained price regressors"
)
ERROR_TITLE: Final[str] = (
    "One structured schedule recovers the solar about 7 points worse than unconstrained price "
    "regressors"
)
FALSE_ALARM_TITLE: Final[str] = (
    "Structured schedules raise fewer false alarms than unconstrained price regressors on the "
    "regional sky (3 against 6), but not always on the GB-mean sky"
)


def _window_frame(*, columns: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the columns on the window's half-hours, labelled by period start.

    Args:
        columns: One array per name, one value per window half-hour.

    Returns:
        A frame with `time` (UTC) and the columns.
    """
    start = window_half_hours() - timedelta(minutes=30)
    return pl.DataFrame({"time": start, **columns})


def schedule_figure(*, number: int) -> alt.VConcatChart:
    """Draw two days of Lakeside's prices, the two schedules, and the real output.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    prices = day_ahead_on_grid()
    rank = schedule_for(prices=prices, kind="rank", duration_hours=EXAMPLE_ENERGY_HOURS)
    lp = schedule_for(prices=prices, kind="lp", duration_hours=EXAMPLE_ENERGY_HOURS)
    soc = state_of_charge(
        schedule=lp,
        energy_hours=EXAMPLE_ENERGY_HOURS,
        eta_one_way=LP_ETA_ONE_WAY,
        initial_soc=LP_SOC_MIN,
    )
    real = (
        _window_frame(columns={"price": prices, "rank": rank, "lp": lp, "soc": soc})
        .join(
            battery_output(bmu_id=BATTERIES[0]).select("time", "output_mw"), on="time", how="left"
        )
        .filter(
            (pl.col("time") >= EXAMPLE_DAY_START)
            & (pl.col("time") < EXAMPLE_DAY_START + timedelta(days=2))
        )
    )
    p99 = float(
        np.quantile(
            battery_output(bmu_id=BATTERIES[0])["output_mw"].drop_nulls().abs().to_numpy(), 0.99
        )
    )
    real = real.with_columns(per_unit=pl.col("output_mw") / p99)
    palette = SCHEDULE_COLOURS
    names = list(palette)
    prices_panel = line_panel(
        series={names[3]: (real, "price", palette[names[3]])},
        title="Day-ahead price (hourly)",
        y_title="£ per MWh",
        last=False,
        interpolate="step-after",
        tick_count="day",
        time_format="%a %-d %b",
        palette=palette,
    )
    schedules_panel = line_panel(
        series={
            names[0]: (real, "per_unit", palette[names[0]]),
            names[1]: (real, "rank", palette[names[1]]),
            names[2]: (real, "lp", palette[names[2]]),
        },
        title="Power per megawatt of rating (positive: exporting; negative: charging)",
        y_title="MW per MW",
        last=False,
        interpolate="step-after",
        rules={"zero": 0.0},
        dashes={names[0]: [4, 3], names[2]: [6, 3]},
        palette=palette,
        height=140,
    )
    soc_panel = line_panel(
        series={names[4]: (real, "soc", palette[names[4]])},
        title="State of charge of the linear-programme schedule (fraction of a 2-hour battery)",
        y_title="Fraction",
        last=True,
        rules={"empty": LP_SOC_MIN, "full": 0.95},
        palette=palette,
    )
    return figure(
        panels=[prices_panel, schedules_panel, soc_panel],
        number=number,
        title=SCHEDULES_TITLE,
        subtitle=[
            (
                "Tuesday 16 and Wednesday 17 December 2025, UTC. Both schedules are built from "
                "the day-ahead price alone for a 2-hour battery. The linear programme has "
                f"one-way efficiency {LP_ETA_ONE_WAY}, state of charge 5% to 95%, at most "
                f"{LP_CYCLES_PER_DAY:g} cycle a day, and each day ends at the state of charge it "
                "started at. The real output is Lakeside's, divided by its 99th-percentile "
                "output."
            )
        ],
        figure_planning=None,
    )


def _series(*, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return the fitted series of the drawn aggregate for some arms, in the summer week.

    Args:
        arms: The arms, each in rung 4's or rung 5's series file.

    Returns:
        Rows with a UTC `time`, the arm, and the true and fitted series.
    """
    frames = []
    for name in ("rung4", "rung5"):
        frame = pl.read_parquet(OUTPUT_DIR / f"{name}_series.parquet").filter(
            pl.col("arm").is_in(arms)
        )
        frames.append(frame.drop("schedule_per_unit", strict=False))
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        .filter(
            (pl.col("time") >= SUMMER_WEEK_START)
            & (pl.col("time") < SUMMER_WEEK_START + timedelta(days=7))
        )
    )


def contribution_figure(*, number: int) -> alt.VConcatChart:
    """Draw each arm's fitted battery output against the true battery for the summer week.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    frames = _series(arms=tuple(CONTRIBUTION_ARMS))
    first = frames.filter(pl.col("arm") == "A1")
    series = {"True battery": (first, "battery_truth_mw", CONTRIBUTION_COLOURS["True battery"])}
    for arm, name in CONTRIBUTION_ARMS.items():
        series[name] = (
            frames.filter(pl.col("arm") == arm),
            "regressor_contribution_mw",
            CONTRIBUTION_COLOURS[name],
        )
    panel = line_panel(
        series=series,
        title="Battery output the fit attributes to the battery, against the true battery",
        y_title="MW",
        last=True,
        rules={"zero": 0.0},
        palette=CONTRIBUTION_COLOURS,
        dashes=CONTRIBUTION_DASHES,
        height=220,
    )
    return figure(
        panels=[panel],
        number=number,
        title=CONTRIBUTION_TITLE,
        subtitle=[
            (
                "One week from Monday 22 June 2026, UTC. The aggregate is 25% Burwell solar and "
                "75% Lakeside battery by 99th-percentile output, as in Figures 7 to 9. Each line "
                "is the arm's fitted coefficient times its regressor column or columns."
            )
        ],
        figure_planning=None,
    )


def _levels(*, group: str) -> pl.DataFrame:
    """Return the regional-sky solar-error intervals of one group, one row per arm and share.

    Args:
        group: `pooled` or a battery's BMU id.

    Returns:
        Columns `label`, `share`, `mean`, `lower_95`, and `upper_95`.
    """
    intervals = pl.read_parquet(OUTPUT_DIR / "rung5_intervals.parquet")
    return intervals.filter(
        (pl.col("sky") == "regional")
        & (pl.col("group") == group)
        & (pl.col("metric") == "nmae_of_solar_p99")
        & pl.col("quantity").is_in(list(FIGURE_ARMS))
        & (pl.col("share") > 0)
    ).with_columns(label=pl.col("quantity").replace_strict(CHART_LABELS, return_dtype=pl.String))


def error_figure(*, number: int) -> alt.VConcatChart:
    """Draw the solar series error of every main arm for each battery.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    panels = [
        _arm_dot_panel(
            table=_levels(group=bmu_id),
            title=f"With the {NAMES[bmu_id]} battery",
            x_title="Mean absolute error of the recovered solar (% of true solar p99; "
            "smaller is better)",
            last=index == len(BATTERIES) - 1,
            show_key=index == 0,
            arms=FIGURE_ARMS,
            labels=CHART_LABELS,
            height=190,
        )
        for index, bmu_id in enumerate(BATTERIES)
    ]
    return figure(
        panels=panels,
        number=number,
        title=ERROR_TITLE,
        subtitle=[
            (
                "Regional sky. Dot: mean over the 4 solar sets. Line: 95% interval from "
                "resampling the 4 solar sets, so it is wide and covers only these four. A5 and "
                "A6 are the structured schedules, A5c and A6c are their wrong-week controls, "
                "A1 and A4 are rung 4's unconstrained price regressors and their control, and A3 "
                "is the oracle. Week-old means prices from 7 days earlier. The post hoc contrast "
                "B5 is A6 minus A1."
            )
        ],
        figure_planning=None,
    )


def false_alarm_figure(*, number: int) -> alt.VConcatChart:
    """Draw how many no-solar aggregates each arm detects as having solar.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.concat(
        [
            pl.read_parquet(OUTPUT_DIR / "rung4_fits.parquet"),
            pl.read_parquet(OUTPUT_DIR / "rung5_fits.parquet"),
        ],
        how="diagonal_relaxed",
    )
    rows = []
    for sky in SKIES:
        zero = fits.filter((pl.col("sky") == sky) & (pl.col("share") == 0))
        threshold = float(zero.filter(pl.col("arm") == "A0")["variance_explained"].max())  # ty: ignore[invalid-argument-type]
        for arm in FIGURE_ARMS:
            sub = zero.filter(pl.col("arm") == arm)
            rows.append(
                {
                    "sky": sky,
                    "label": CHART_LABELS[arm],
                    "count": int((sub["variance_explained"] > threshold).sum()),
                    "total": sub.height,
                }
            )
    table = pl.DataFrame(rows)
    sky_names = {"regional": "Regional sky", "gb_mean": "GB-mean sky"}
    table = table.with_columns(sky_label=pl.col("sky").replace_strict(sky_names))
    order = [CHART_LABELS[arm] for arm in FIGURE_ARMS]
    colour = alt.Color(
        "sky_label:N",
        scale=alt.Scale(domain=list(sky_names.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(table)
    bars = base.mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=order, axis=alt.Axis(title=None, labelLimit=300)),
        yOffset=alt.YOffset("sky_label:N"),
        x=alt.X("count:Q", title="No-solar aggregates above the solar-only threshold (of 16)"),
        color=colour,
    )
    labels = base.mark_text(align="left", dx=3, fontSize=9, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=order),
        yOffset=alt.YOffset("sky_label:N"),
        x="count:Q",
        text="count:Q",
    )
    panel = cast(Layer, alt.layer(bars, labels)).properties(
        width=PLOT_WIDTH_PX - 150, height=PANEL_HEIGHT_PX * 2.4
    )
    return figure(
        panels=[panel],
        number=number,
        title=FALSE_ALARM_TITLE,
        subtitle=[
            (
                "A false alarm is a battery-only aggregate (solar share 0%) whose fitted plant "
                "explains more of the corrected four-hour changes than any of the solar-only "
                "fit's 16 battery-only aggregates does. The solar-only fit has none by "
                "construction."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write figures 10 to 13 as SVGs under `docs/studies/assets/` and PNG previews."""
    charts = {
        "rung5_schedules": schedule_figure(number=10),
        "rung5_contribution": contribution_figure(number=11),
        "rung5_error": error_figure(number=12),
        "rung5_false_alarms": false_alarm_figure(number=13),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        target = ASSETS_DIR / f"battery_pv_separation_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
