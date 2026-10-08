"""Draw the battery study's rung 6 figures (1, 14, 15, and 16).

Every figure is built from the tables the rung scripts saved. The batteries, prices, and B1610 are
public, so the figures name each battery and show dates and megawatts.

Run after `battery_rung6.py` and `battery_rung6_summary.py`:
`uv run python studies/battery_pv_separation/battery_charts_rung6.py`.
"""

from datetime import timedelta
from typing import Final, cast

import altair as alt
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
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR
from battery_rung6 import ARM_LABELS, ARMS, FALSE_ALARM_SHARE
from battery_synthetic import AGGREGATE_P99_MW
from studies.charts import figure

COLOURS: Final[dict[str, str]] = {
    "True solar": ocf.BRAND_ORANGE,
    "Solar recovered by the joint model (A8)": ocf.DATA_PURPLE,
    "Solar recovered with a 2-hour battery (A9)": ocf.DATA_GREEN,
    "True battery": ocf.DATA_BLUE,
    "Battery recovered by the joint model (A8)": ocf.DATA_SKY,
    "Aggregate (what the fit sees)": ocf.BLACK_1,
    "True state of charge (rung 3 path)": ocf.DATA_BLUE,
    "Joint model, true battery size (A8)": ocf.DATA_PURPLE,
    "Joint model, half the energy capacity (A8h)": ocf.BRAND_ORANGE,
    "Joint model, 2-hour battery (A9)": ocf.DATA_GREEN,
}
DASHES: Final[dict[str, list[int]]] = {
    "True solar": [4, 3],
    "True battery": [4, 3],
    "True state of charge (rung 3 path)": [4, 3],
}
STORY_NAMES: Final[tuple[str, ...]] = (
    "True solar",
    "True battery",
    "Aggregate (what the fit sees)",
    "Solar recovered by the joint model (A8)",
    "Solar recovered with a 2-hour battery (A9)",
    "Battery recovered by the joint model (A8)",
)
SOC_NAMES: Final[tuple[str, ...]] = (
    "True state of charge (rung 3 path)",
    "Joint model, true battery size (A8)",
    "Joint model, half the energy capacity (A8h)",
    "Joint model, 2-hour battery (A9)",
)
FIGURE_ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A3", "A8", "A8h", "A8d", "A9", "A0b", "A3b")
FALSE_ALARM_FIGURE_ARMS: Final[tuple[str, ...]] = tuple(arm for arm in ARMS if arm != "A1b")
"""The arms Figure 16 draws. `A1b` is scored in the report's false-alarm tables only."""
CHART_LABELS: Final[dict[str, str]] = {
    **ARM_LABELS,
    "A0": "A0: solar only (rung 4 plant fit)",
    "A1": "A1: price level and rank",
    "A3": "A3: true battery as regressor",
    "A8": "A8: joint model, true size",
    "A8h": "A8h: joint, half the energy",
    "A8d": "A8d: joint, double the energy",
    "A9": "A9: joint, 2-hour battery",
    "A0b": "A0b: fleet-curve solar, no battery",
    "A3b": "A3b: fleet-curve, battery removed",
}
SKY_NAMES: Final[dict[str, str]] = {"regional": "Regional sky", "gb_mean": "GB-mean sky"}
STORY_TITLE: Final[str] = (
    "Given the battery's true power, the joint model separates solar from the battery on clear "
    "days, but misses the cloud on the cloudy last day"
)
SOC_TITLE: Final[str] = (
    "The fitted state of charge follows the state of charge integrated from the true battery's "
    "output (rung 3), but its level drifts when the energy capacity is large"
)
ERROR_TITLE: Final[str] = (
    "The joint model recovers solar nearly as well as the oracle and better than price "
    "regressors, but most of the gain over rung 4 comes from its solar model"
)
FALSE_ALARM_TITLE: Final[str] = (
    "On battery-only aggregates the joint model fits no solar with rung 3's energy capacity or "
    "more, and under 2 MW with less"
)


def _week(*, arm: str) -> pl.DataFrame:
    """Return one arm's fitted series for the summer week.

    Args:
        arm: The arm.

    Returns:
        The rows of `rung6_series.parquet` for the arm in the week.
    """
    frame = (
        pl.read_parquet(OUTPUT_DIR / "rung6_series.parquet")
        .filter(pl.col("arm") == arm)
        .with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    )
    return frame.filter(
        (pl.col("time") >= SUMMER_WEEK_START)
        & (pl.col("time") < SUMMER_WEEK_START + timedelta(days=7))
    ).with_columns(
        true_soc_relative=pl.col("true_soc_mwh") - pl.col("true_soc_mwh").min(),
        recovered_soc_relative=pl.col("recovered_soc_mwh") - pl.col("recovered_soc_mwh").min(),
    )


def story_figure(*, number: int) -> alt.VConcatChart:
    """Draw a summer week: the aggregate, the solar, and the battery, true and recovered.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    full, small = _week(arm="A8"), _week(arm="A9")
    palette = {name: COLOURS[name] for name in STORY_NAMES}
    top = line_panel(
        series={
            "True solar": (full, "solar_truth_mw", COLOURS["True solar"]),
            "True battery": (full, "battery_truth_mw", COLOURS["True battery"]),
            "Aggregate (what the fit sees)": (full, "aggregate_mw", ocf.BLACK_1),
        },
        title="The aggregate: Burwell solar (25% of it) plus the Lakeside battery",
        y_title="MW",
        last=False,
        rules={"zero": 0.0},
        palette=palette,
        height=130,
        dashes=DASHES,
    )
    middle = line_panel(
        series={
            "True solar": (full, "solar_truth_mw", COLOURS["True solar"]),
            "Solar recovered by the joint model (A8)": (
                full,
                "recovered_solar_mw",
                COLOURS["Solar recovered by the joint model (A8)"],
            ),
            "Solar recovered with a 2-hour battery (A9)": (
                small,
                "recovered_solar_mw",
                COLOURS["Solar recovered with a 2-hour battery (A9)"],
            ),
        },
        title="Solar recovered by the fit, against the true solar",
        y_title="MW",
        last=False,
        rules={"zero": 0.0},
        palette=palette,
        height=130,
        dashes=DASHES,
    )
    bottom = line_panel(
        series={
            "True battery": (full, "battery_truth_mw", COLOURS["True battery"]),
            "Battery recovered by the joint model (A8)": (
                full,
                "recovered_battery_mw",
                COLOURS["Battery recovered by the joint model (A8)"],
            ),
        },
        title="Battery output recovered by the fit, against the true battery",
        y_title="MW",
        last=True,
        rules={"zero": 0.0},
        palette=palette,
        height=130,
        dashes=DASHES,
    )
    return figure(
        panels=[top, middle, bottom],
        number=number,
        title=STORY_TITLE,
        subtitle=[
            (
                "One week from Monday 22 June 2026, UTC; the aggregate is rung 4's (25% Burwell "
                "solar, 75% Lakeside battery by 99th-percentile output, 100 MW in total, with no "
                "demand-like part; rung 7c adds one), with "
                "the regional CAMS sky. A8 gives the fitted battery the true battery's power and "
                "energy capacity; A9 gives it the true power and 2 hours of energy. The fit used "
                "the whole year in windows of about 4 weeks."
            )
        ],
        figure_planning=None,
    )


def soc_figure(*, number: int) -> alt.VConcatChart:
    """Draw the recovered and the true state of charge for the summer week.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    arms = {
        "A8": "Joint model, true battery size (A8)",
        "A8h": "Joint model, half the energy capacity (A8h)",
        "A9": "Joint model, 2-hour battery (A9)",
    }
    first = _week(arm="A8")
    series = {
        "True state of charge (rung 3 path)": (
            first,
            "true_soc_relative",
            COLOURS["True state of charge (rung 3 path)"],
        )
    }
    for arm, name in arms.items():
        series[name] = (_week(arm=arm), "recovered_soc_relative", COLOURS[name])
    panel = line_panel(
        series=series,
        title="State of charge above the week's lowest value",
        y_title="MWh",
        last=True,
        palette={name: COLOURS[name] for name in SOC_NAMES},
        height=170,
        dashes=DASHES,
    )
    return figure(
        panels=[panel],
        number=number,
        title=SOC_TITLE,
        subtitle=[
            (
                "The same week and aggregate as Figure 1. The state of charge of the true battery "
                "is rung 3's integral of its output at the fitted efficiency, scaled to the "
                "aggregate. The fit's starting charge is free in each 4-week window, so every "
                "path is shown above its own lowest value in the week."
            )
        ],
        figure_planning=None,
    )


def _levels(*, group: str) -> pl.DataFrame:
    """Return the regional-sky intervals of the solar error for one group.

    Args:
        group: `pooled` or a battery's BMU id.

    Returns:
        Columns `label`, `share`, `mean`, `lower_95`, and `upper_95`.
    """
    intervals = pl.read_parquet(OUTPUT_DIR / "rung6_intervals.parquet")
    return intervals.filter(
        (pl.col("sky") == "regional")
        & (pl.col("group") == group)
        & (pl.col("metric") == "nmae_of_solar_p99")
        & pl.col("quantity").is_in(list(FIGURE_ARMS))
        & (pl.col("share") > 0)
    ).with_columns(label=pl.col("quantity").replace_strict(CHART_LABELS, return_dtype=pl.String))


def error_figure(*, number: int) -> alt.VConcatChart:
    """Draw the solar series error by arm and share for each battery.

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
            height=200,
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
                "resampling the 4 solar sets, so it is wide and covers only these four. A0, A1, "
                "and A3 are rung 4's fits. A8, A8h, A8d, and A9 are the joint model with the "
                "true battery size, half and double its energy capacity, and a 2-hour battery. "
                "A0b and A3b fit the joint model's solar model with no battery, to the "
                "aggregate and to the aggregate minus the true battery. The post hoc contrasts "
                "are A8 minus A0 (B7) and A9 minus A0 (B8)."
            )
        ],
        figure_planning=None,
    )


def false_alarm_figure(*, number: int) -> alt.VConcatChart:
    """Draw the fitted solar capacity of each no-solar aggregate and the false-alarm counts.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "rung6_fits.parquet").filter(
        pl.col("arm").is_in(FALSE_ALARM_FIGURE_ARMS)
    )
    zero = (
        fits.filter(pl.col("share") == 0)
        .with_columns(
            label=pl.col("arm").replace_strict(CHART_LABELS, return_dtype=pl.String),
            sky_label=pl.col("sky").replace_strict(SKY_NAMES, return_dtype=pl.String),
        )
        .select("label", "sky_label", "fitted_ac_mw")
    )
    threshold = FALSE_ALARM_SHARE * AGGREGATE_P99_MW
    order = [CHART_LABELS[arm] for arm in FALSE_ALARM_FIGURE_ARMS]
    colour = alt.Color(
        "sky_label:N",
        scale=alt.Scale(domain=list(SKY_NAMES.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("label:N", sort=order, axis=alt.Axis(title=None, labelLimit=300))
    dots = (
        alt.Chart(zero)
        .mark_point(filled=True, size=40, opacity=0.7, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=alt.YOffset("sky_label:N"),
            x=alt.X(
                "fitted_ac_mw:Q",
                title="Fitted solar capacity in the 16 battery-only aggregates (MW)",
                scale=alt.Scale(type="symlog", constant=2),
            ),
            color=colour,
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [threshold]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=0.8, aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = cast(Layer, alt.layer(rule, dots)).properties(
        width=PLOT_WIDTH_PX - 150, height=PANEL_HEIGHT_PX * 3.2
    )
    return figure(
        panels=[panel],
        number=number,
        title=FALSE_ALARM_TITLE,
        subtitle=[
            (
                "Each dot is one of the 16 aggregates of a battery alone (4 batteries by 4 "
                "solar sets, solar share 0%), so the true solar capacity is 0 MW. The dashed "
                f"line is the detection threshold, {FALSE_ALARM_SHARE:.0%} of the "
                f"aggregate's 99th percentile ({threshold:.0f} MW); a dot to its right is a "
                "false alarm. The horizontal axis is stretched near 0 so that small "
                "capacities are visible."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write figures 1, 14, 15, and 16 as SVGs under `docs/studies/assets/` and PNG previews."""
    charts = {
        "rung6_story": story_figure(number=1),
        "rung6_soc": soc_figure(number=14),
        "rung6_error": error_figure(number=15),
        "rung6_false_alarms": false_alarm_figure(number=16),
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
