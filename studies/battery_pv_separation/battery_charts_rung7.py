"""Draw the battery study's rung 7 figures (17 to 19).

Every figure is built from the tables `battery_rung7.py` saved. The BMUs are public supplier and
virtual BMUs, so the figures name them and show dates and megawatts.

Run after `battery_rung7.py`:
`uv run python studies/battery_pv_separation/battery_charts_rung7.py`.
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
    line_panel,
)
from battery_inputs import OUTPUT_DIR
from battery_rung7 import CLOUD_SIGNAL_BMUS, SERIES_BMU, SERIES_POWERS_MW
from studies.charts import figure

SOLAR_STUDY_PHYSICAL_MW: Final[float] = 1163.0
SOLAR_STUDY_DIFFERENCE_MW: Final[float] = 1089.0
"""The solar study's totals over the 25 BMUs: the physical fit and the regional difference
separation."""
REAL_LABEL: Final[str] = "Real output"
REPLICA_LABEL: Final[str] = "Calendar replica"
CAPACITY_TITLE: Final[str] = (
    "Without a calendar baseline, allowing a larger battery cuts the fitted solar capacity of the "
    "biggest BMUs, but their cloud-free replicas fall by a similar share"
)
WEEK_TITLE: Final[str] = (
    "Without a calendar baseline, allowing a 50 MW battery cuts the fitted solar peak of "
    "2__ATGPL000 from 62 MW to 46 MW, and the battery then runs at its limit"
)
TOTAL_TITLE: Final[str] = (
    "Without a calendar baseline, the fitted solar capacity of the 25 BMUs is 161 to 323 MW at "
    "every assumed battery size tried, far below the solar study's 1,089 to 1,163 MW"
)
SYMLOG_CONSTANT: Final[float] = 2.0


def _scale() -> alt.Scale:
    """Return the symlog scale of the battery power axis, which includes 0."""
    return alt.Scale(type="symlog", constant=SYMLOG_CONSTANT, domain=[0, 200])


def capacity_figure(*, number: int) -> alt.VConcatChart:
    """Draw the fitted solar capacity of the nine cloud-signal BMUs against assumed battery power.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "rung7_fits.parquet").filter(
        pl.col("bmu").is_in(CLOUD_SIGNAL_BMUS)
    )
    real = fits.filter(pl.col("source") == "real").with_columns(line=pl.col("bmu"))
    replica = fits.filter(pl.col("source") == "replica").with_columns(line=pl.lit(REPLICA_LABEL))
    order = [*CLOUD_SIGNAL_BMUS]
    colours = [
        ocf.BRAND_ORANGE,
        ocf.DATA_BLUE,
        ocf.DATA_PURPLE,
        ocf.DATA_GREEN,
        ocf.DATA_SKY,
        ocf.DATA_AMBER,
        ocf.DATA_DEEP_TEAL,
        ocf.DATA_MAGENTA,
        ocf.DATA_BURNT_ORANGE,
    ]
    x = alt.X(
        "power_mw:Q",
        title="Assumed battery power (MW; 2 hours of energy; 0 is solar only)",
        scale=_scale(),
        axis=alt.Axis(values=[0, 2, 5, 10, 20, 50, 100, 200]),
    )
    y = alt.Y(
        "fitted_ac_mw:Q",
        title="Fitted solar capacity (MW)",
        scale=alt.Scale(type="symlog", constant=1.0),
    )
    grey = (
        alt.Chart(replica)
        .mark_line(color=ocf.BLACK_1, opacity=0.35, strokeWidth=1.0, aria=False)
        .encode(x=x, y=y, detail="bmu:N")  # ty: ignore[unresolved-attribute]
    )
    coloured = (
        alt.Chart(real)
        .mark_line(point=True, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=y,
            color=alt.Color(
                "bmu:N",
                scale=alt.Scale(domain=order, range=colours),
                legend=alt.Legend(title=None, columns=3, symbolLimit=0),
            ),
        )
    )
    panel = cast(Layer, alt.layer(grey, coloured)).properties(
        width=PLOT_WIDTH_PX - 60, height=PANEL_HEIGHT_PX * 3
    )
    return figure(
        panels=[panel],
        number=number,
        title=CAPACITY_TITLE,
        subtitle=[
            (
                "The nine aggregate BMUs in which the solar study found a cloud-correlated "
                "component. Each coloured line is one BMU's real output, fitted with solar "
                "(four fleet curves, regional CAMS sky) plus a battery of the assumed power, "
                "2 hours of energy, and a one-way efficiency of 0.92. Each grey line is the "
                "same fit to that BMU's calendar replica, which has no cloud; the "
                "lines overlap near zero for 2__BTGPL000 and 2__KTGPL000, whose fitted solar "
                "capacity is 0 MW throughout. No ground truth exists for these BMUs. Both "
                "axes are stretched near 0."
            )
        ],
        figure_planning=None,
    )


def week_figure(*, number: int) -> alt.VConcatChart:
    """Draw a summer week of one supplier BMU, fitted with no battery and with a 50 MW battery.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    series = pl.read_parquet(OUTPUT_DIR / "rung7_series.parquet").with_columns(
        pl.col("time").dt.replace_time_zone("UTC")
    )
    palette: dict[str, str] = {
        "Aggregate output": ocf.BLACK_1,
        "Fitted solar": ocf.BRAND_ORANGE,
        "Fitted battery": ocf.DATA_BLUE,
    }
    panels = []
    for index, power in enumerate(SERIES_POWERS_MW):
        week = series.filter(
            (pl.col("power_mw") == power)
            & (pl.col("time") >= SUMMER_WEEK_START)
            & (pl.col("time") < SUMMER_WEEK_START + timedelta(days=7))
        )
        label = "no battery" if power == 0 else f"a {power:.0f} MW battery allowed"
        panels.append(
            line_panel(
                series={
                    "Aggregate output": (week, "aggregate_mw", ocf.BLACK_1),
                    "Fitted solar": (week, "fitted_solar_mw", ocf.BRAND_ORANGE),
                    "Fitted battery": (week, "fitted_battery_mw", ocf.DATA_BLUE),
                },
                title=f"{SERIES_BMU} with {label}",
                y_title="MW",
                last=index == len(SERIES_POWERS_MW) - 1,
                rules={"zero": 0.0},
                palette=palette,
                height=170,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=WEEK_TITLE,
        subtitle=[
            (
                f"One week from Monday 22 June 2026, UTC, for the aggregate BMU {SERIES_BMU}. "
                "The fit used the whole year in windows of about 4 weeks. Top: solar only, so "
                "the fitted battery is zero. Bottom: the battery is limited to 50 MW and 100 "
                "MWh. The true split of this BMU's output between solar, batteries, and other "
                "plant is not known."
            )
        ],
        figure_planning=None,
    )


def total_figure(*, number: int) -> alt.VConcatChart:
    """Draw the total fitted solar capacity of the 25 BMUs against assumed battery power.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    totals = (
        pl.read_parquet(OUTPUT_DIR / "rung7_fits.parquet")
        .group_by("source", "power_mw")
        .agg(pl.col("fitted_ac_mw").sum())
        .with_columns(
            line=pl.col("source").replace_strict(
                {"real": REAL_LABEL, "replica": REPLICA_LABEL}, return_dtype=pl.String
            )
        )
    )
    x = alt.X(
        "power_mw:Q",
        title="Assumed battery power (MW; 2 hours of energy; 0 is solar only)",
        scale=_scale(),
        axis=alt.Axis(values=[0, 2, 5, 10, 20, 50, 100, 200]),
    )
    lines = (
        alt.Chart(totals)
        .mark_line(point=True, strokeWidth=1.6, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("fitted_ac_mw:Q", title="Total fitted solar capacity of the 25 BMUs (MW)"),
            color=alt.Color(
                "line:N",
                scale=alt.Scale(
                    domain=[REAL_LABEL, REPLICA_LABEL], range=[ocf.BRAND_ORANGE, ocf.BLACK_1]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    references = pl.DataFrame(
        {
            "y": [SOLAR_STUDY_PHYSICAL_MW, SOLAR_STUDY_DIFFERENCE_MW],
            "label": [
                f"Solar study, physical fit: {SOLAR_STUDY_PHYSICAL_MW:,.0f} MW",
                f"Solar study, difference separation: {SOLAR_STUDY_DIFFERENCE_MW:,.0f} MW",
            ],
        }
    )
    rules = (
        alt.Chart(references)
        .mark_rule(color=ocf.DATA_BLUE, strokeDash=[4, 3], strokeWidth=1.0, aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    text = (
        alt.Chart(references)
        .mark_text(align="left", dx=4, dy=-5, fontSize=10, color=ocf.DATA_BLUE, aria=False)
        .encode(x=alt.value(4), y="y:Q", text="label:N")  # ty: ignore[unresolved-attribute]
    )
    panel = cast(Layer, alt.layer(rules, text, lines)).properties(
        width=PLOT_WIDTH_PX - 60, height=PANEL_HEIGHT_PX * 3
    )
    return figure(
        panels=[panel],
        number=number,
        title=TOTAL_TITLE,
        subtitle=[
            (
                "Sum over the 25 aggregate BMUs of the fitted AC capacity of the four fleet "
                "curves (regional CAMS sky), with a battery of the assumed power and 2 hours of "
                "energy. The dashed lines are the solar study's totals over the same 25 BMUs; "
                "its method also fitted a seasonal calendar baseline, so the comparison shows "
                "scale, not agreement. No ground truth exists. The horizontal axis is "
                "stretched near 0."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write figures 17 to 19 as SVGs under `docs/studies/assets/` and PNG previews."""
    charts = {
        "rung7_capacity": capacity_figure(number=17),
        "rung7_week": week_figure(number=18),
        "rung7_total": total_figure(number=19),
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
