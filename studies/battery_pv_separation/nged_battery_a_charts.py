"""Draw NGED battery A's three figures, anonymised.

NGED's battery is a private generator, so the figures show it only as "NGED battery A", count the
days of the week 1 to 7 rather than dating them, give output as a fraction of the series' own 99th
percentile absolute output rather than in megawatts, and turn off per-point accessibility text.
The week is not the public batteries' week, no figure shows a price (a price curve dates the
week), no subtitle names a month or year, and the caption says "NGED figure" rather than "Figure".

Run after `nged_battery_a_rungs.py`:
`uv run python studies/battery_pv_separation/nged_battery_a_charts.py`. Writes the SVGs and PNG
previews under `NGED_BATTERY_A_FIGURES_DIR`, in the study's data folder and outside `docs/`,
because the figures are not on the battery study page.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from battery_charts import (
    OUTPUT_COLOUR,
    PLOT_WIDTH_PX,
    PRICE_COLOUR,
    line_panel,
)
from battery_inputs import OUTPUT_DIR, TERCILE_LABELS
from battery_rung3 import fit_efficiency
from nged_battery_a import ALIAS, battery_a_frame
from studies.battery_capacity import smallest_capacity
from studies.charts import figure
from studies.sources import NGED_BATTERY_A_FIGURES_DIR

SEARCH_START: Final[datetime] = datetime(2026, 2, 2, tzinfo=UTC)
"""The first Monday the week search considers. It lies well after the public batteries' week, so
the two weeks cannot be matched against each other."""
SEARCH_WEEKS: Final[int] = 30
"""The number of Mondays the week search tries."""
MIN_ACTIVE_DAYS: Final[int] = 5
"""The week must hold a charge and an export on at least this many of its seven days."""
ACTIVE_FRACTION: Final[float] = 0.1
"""A charge or an export counts when it exceeds this fraction of the series' p99 output."""
CAPTION_LABEL: Final[str] = "NGED figure"
DAY_ONE: Final[datetime] = datetime(2001, 1, 1, tzinfo=UTC)
"""Plotted days are placed on 1 to 7 January of a placeholder year, and the axis shows the day of
the month."""
DAY_FORMAT: Final[str] = "%-d"
LIMIT: Final[float] = 0.4
"""The heat map's colour range, as a fraction of p99 output."""
PALETTE: Final[dict[str, str]] = {f"{ALIAS} output": OUTPUT_COLOUR}
SOC_PALETTE: Final[dict[str, str]] = {
    f"{ALIAS} output": OUTPUT_COLOUR,
    "State of charge": PRICE_COLOUR,
}


def choose_week_start(*, frame: pl.DataFrame) -> datetime:
    """Return the first Monday from `SEARCH_START` whose week is complete and visibly cycling.

    Args:
        frame: A frame with UTC `time` and `output_mw` columns.

    Returns:
        The start of the chosen week.

    Raises:
        ValueError: If no week among the candidates qualifies.
    """
    p99 = float(frame["output_mw"].abs().quantile(0.99))  # ty: ignore[invalid-argument-type]
    for week in range(SEARCH_WEEKS):
        start = SEARCH_START + timedelta(weeks=week)
        rows = frame.filter(
            (pl.col("time") >= start) & (pl.col("time") < start + timedelta(days=7))
        )
        if rows.height != 336:
            continue
        days = (
            rows.with_columns(day=(pl.col("time") - start).dt.total_days())
            .group_by("day")
            .agg(
                (pl.col("output_mw").min() < -ACTIVE_FRACTION * p99).alias("charges"),
                (pl.col("output_mw").max() > ACTIVE_FRACTION * p99).alias("exports"),
            )
        )
        if int((days["charges"] & days["exports"]).sum()) >= MIN_ACTIVE_DAYS:
            return start
    msg = "No complete, visibly cycling week was found."
    raise ValueError(msg)


def _day_count(*, frame: pl.DataFrame, week_start: datetime) -> pl.DataFrame:
    """Return the week's rows with `time` moved onto the placeholder days 1 to 7.

    Args:
        frame: A frame with a UTC `time` column.
        week_start: The start of the week to keep.

    Returns:
        The rows from `week_start` for seven days, with `time` shifted so day 1 is 1 January.
    """
    return frame.filter(
        (pl.col("time") >= week_start) & (pl.col("time") < week_start + timedelta(days=7))
    ).with_columns(time=pl.col("time") - week_start + DAY_ONE)


def week_figure(*, number: int) -> alt.VConcatChart:
    """Draw the week of output.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    frame, _ = battery_a_frame()
    output = _day_count(frame=frame, week_start=choose_week_start(frame=frame))
    panel = line_panel(
        series={f"{ALIAS} output": (output, "output_mw", OUTPUT_COLOUR)},
        title="Output as a fraction of its 99th percentile absolute value (positive: exporting)",
        y_title="Fraction of p99",
        last=True,
        interpolate="step-after",
        rules={"zero": 0.0},
        time_format=DAY_FORMAT,
        x_title="Day of the week (1 to 7)",
        palette=PALETTE,
    )
    return figure(
        panels=[panel],
        number=number,
        label=CAPTION_LABEL,
        title=f"{ALIAS} made a few large charges and exports a day",
        subtitle=[
            "One week, UTC, with the days counted 1 to 7.",
            "Output is metered half-hourly power, shown as a fraction of its own 99th percentile.",
        ],
        figure_planning=None,
    )


def heatmap_figure(*, number: int) -> alt.VConcatChart:
    """Draw mean output by half-hour of day and within-day price third.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    table = pl.read_parquet(OUTPUT_DIR / "nged_battery_a_tod_tercile_means.parquet").with_columns(
        third=pl.col("tercile").replace_strict(
            dict(enumerate(TERCILE_LABELS)), return_dtype=pl.String
        )
    )
    panel = (
        alt.Chart(table)
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "tod:O",
                axis=alt.Axis(
                    title="Hour of day (UTC, start of half-hour)",
                    values=list(range(0, 48, 6)),
                    labelExpr="datum.value / 2",
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "third:N", sort=list(TERCILE_LABELS), axis=alt.Axis(title=None, labelLimit=170)
            ),
            color=alt.Color(
                "mean_output:Q",
                scale=alt.Scale(
                    domain=[-LIMIT, 0, LIMIT],
                    range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE],
                    clamp=True,
                ),
                legend=alt.Legend(
                    title="Mean output (fraction of p99): blue is charging, orange is exporting",
                    titleLimit=500,
                    orient="bottom",
                    direction="horizontal",
                    gradientLength=300,
                ),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=66, title=alt.TitleParams(ALIAS, anchor="start"))
    )
    return figure(
        panels=[panel],
        number=number,
        label=CAPTION_LABEL,
        title=f"{ALIAS} charges in the middle of the day and exports in the early evening",
        subtitle=[
            (
                "Mean output for each half-hour of the UTC day (columns) and each third of the "
                "day's 48 half-hours when ranked by day-ahead price (rows), over a year of data. "
                f"Output is a fraction of its 99th percentile; colours are clipped at {LIMIT}."
            )
        ],
        figure_planning=None,
    )


def soc_week_figure(*, number: int) -> alt.VConcatChart:
    """Draw the week's output and the implied state of charge from the fold's own fit.

    The fit uses the three-month fold that holds the week, because the whole-year fit's path
    drifts (see the report).

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    frame, _ = battery_a_frame()
    week_start = choose_week_start(frame=frame)
    week_fold = int(frame.filter(pl.col("time") >= week_start)["fold"][0])
    fold_frame = frame.filter(pl.col("fold") == week_fold)
    eta, capacity, _, _ = fit_efficiency(output_mwh=fold_frame["output_mwh"].to_numpy())
    _, path = smallest_capacity(output_mwh=fold_frame["output_mwh"].to_numpy(), eta=eta)
    soc = fold_frame.select("time", "output_mw").with_columns(
        soc_fraction=pl.Series(path / capacity)
    )
    week = _day_count(frame=soc, week_start=week_start)
    top = line_panel(
        series={f"{ALIAS} output": (week, "output_mw", OUTPUT_COLOUR)},
        title="Output as a fraction of its 99th percentile absolute value (positive: exporting)",
        y_title="Fraction of p99",
        last=False,
        interpolate="step-after",
        rules={"zero": 0.0},
        time_format=DAY_FORMAT,
        palette=SOC_PALETTE,
    )
    bottom = line_panel(
        series={"State of charge": (week, "soc_fraction", PRICE_COLOUR)},
        title=(
            "Implied state of charge as a fraction of the fitted capacity "
            f"(one-way efficiency {eta:.3f})"
        ),
        y_title="Fraction of capacity",
        last=True,
        rules={"empty": 0.0},
        time_format=DAY_FORMAT,
        x_title="Day of the week (1 to 7)",
        palette=SOC_PALETTE,
    )
    return figure(
        panels=[top, bottom],
        number=number,
        label=CAPTION_LABEL,
        title="Adding up the output gives a state of charge that rises and falls with it",
        subtitle=[
            (
                f"{ALIAS}, one week, UTC, with the days counted 1 to 7. The path "
                "comes from a fit to the three-month block of the year that holds the week."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write the three figures as SVG and PNG preview."""
    charts = {
        "nged_week": week_figure(number=1),
        "nged_heatmap": heatmap_figure(number=2),
        "nged_soc_week": soc_week_figure(number=3),
    }
    NGED_BATTERY_A_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        target = NGED_BATTERY_A_FIGURES_DIR / f"battery_pv_separation_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
