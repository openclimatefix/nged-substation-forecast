"""Draw each capacity estimator's estimate over a plant's measured power, for issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
`estimate_figure` stacks one panel per estimator on the same axes: the daily maximum of the plant's
measured hourly power as grey dots, the estimator's capacity as a line on top, and the
planted-loss windows shaded. **Both are drawn as a share of the plant's static power, never in
megawatts**, because one metered generator's output in megawatts must not be published.

**The time axis counts days from the first day shown and never carries a calendar date.** A
metered generator's output on known dates can be matched to public generation data, so the axis
reads "day of the window" and the figure's text gives no date. The marks carry no accessibility text
(`aria=False`), which Vega would otherwise fill with each point's value.

**Every panel shares one vertical range**, from zero to 15% above the larger of 1 (the static
power) and the highest daily maximum in the window, so the estimators can be compared by eye.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

from studies.capacity_estimators import CapacityRecord
from studies.capacity_recovery import PlantedEvent
from studies.charts import CONTENT_WIDTH_PX, figure

PANEL_HEIGHT_PX: Final[int] = 105
"""The height of one estimator's panel."""

HEADROOM: Final[float] = 1.15
"""The shared vertical range reaches this multiple of the larger of the static power and data."""


@dataclass(frozen=True)
class EstimatorPanel:
    """One estimator's panel.

    Attributes:
        key: The key of the estimator's series in the capacity-series store.
        title: The panel's title, naming the estimator.
    """

    key: str
    title: str


ROUND2_PANELS: Final[tuple[EstimatorPanel, ...]] = (
    EstimatorPanel(key="S0", title="Static 99th-percentile power (no estimator)"),
    EstimatorPanel(key="HYST2", title="HYST2: hold, then follow a lasting drop"),
    EstimatorPanel(key="HM2", title="HM2: hidden Markov model over capacity levels"),
    EstimatorPanel(key="tools_only_v1", title="Round 1 tools-only rule (changepoint list)"),
    EstimatorPanel(key="tools_only_v2", title="Round 2 tools-only rule (local scan)"),
    EstimatorPanel(key="llm_v1", title="Round 1 language-model labeller, hindsight"),
    EstimatorPanel(key="llm_v2", title="Round 2 language-model labeller, hindsight"),
    EstimatorPanel(key="llm_causal_v2", title="Round 2 language-model labeller, causal (6 months)"),
)
"""The eight panels of round 2's figures (post hoc), top to bottom."""

ESTIMATOR_PANELS: Final[tuple[EstimatorPanel, ...]] = (
    EstimatorPanel(key="S0", title="Static 99th-percentile power (no estimator)"),
    EstimatorPanel(key="HYST2", title="HYST2: hold, then follow a lasting drop"),
    EstimatorPanel(key="HM2", title="HM2: hidden Markov model over capacity levels"),
    EstimatorPanel(key="CX2S", title="CX2S: convex program (CX2) on the corrected ratio"),
    EstimatorPanel(key="tools_only", title="Tools-only labeller (fixed rule on the six tools)"),
    EstimatorPanel(key="llm_hindsight", title="Language-model labeller, hindsight (whole record)"),
    EstimatorPanel(
        key="llm_causal", title="Language-model labeller, causal (monthly, last 6 months)"
    ),
)
"""The seven panels, top to bottom."""


def daily_maximum(*, record: CapacityRecord, plant: int) -> np.ndarray:
    """Return each day's highest measured hourly power, NaN on a day with no reading.

    Args:
        record: The record.
        plant: The plant's index.

    Returns:
        One value per day in megawatts.
    """
    power = record.power_mw[plant]
    seen = np.isfinite(power).any(axis=1)
    return np.where(seen, np.nanmax(np.where(np.isfinite(power), power, -np.inf), axis=1), np.nan)


def window_rows(
    *,
    maximum: np.ndarray,
    capacity: np.ndarray | None,
    window: tuple[int, int],
) -> pl.DataFrame:
    """Return one panel's rows for a window of days.

    Args:
        maximum: `daily_maximum`'s result.
        capacity: The estimator's capacity per day in megawatts, or `None` if it has none here.
        window: The first day index (inclusive) and the last (exclusive).

    Returns:
        Rows with `day` (1 for the first day shown), `maximum_share`, and `capacity_share` (null
            where the
        estimator has no value).
    """
    start, stop = window
    days = np.arange(start, stop)
    capacity_values = np.full(len(days), np.nan) if capacity is None else capacity[start:stop]
    return pl.DataFrame(
        {
            "day": days - start + 1,
            "maximum_share": maximum[start:stop],
            "capacity_share": capacity_values,
        }
    ).with_columns(pl.all().fill_nan(None))


def loss_windows(
    *, events: Sequence[PlantedEvent], plant: int, window: tuple[int, int]
) -> pl.DataFrame:
    """Return the planted-loss windows of one plant that overlap a window, in window days.

    Args:
        events: The planted events of the record.
        plant: The plant's index.
        window: The first day index (inclusive) and the last (exclusive).

    Returns:
        Rows with `start_day` and `end_day`, counted as `window_rows`' `day`.
    """
    start, stop = window
    rows = [
        {
            "start_day": max(event.first_day, start) - start + 1,
            "end_day": min(event.first_day + event.month_days, stop) - start + 1,
        }
        for event in events
        if event.plant == plant
        and event.first_day < stop
        and event.first_day + event.month_days > start
    ]
    return pl.DataFrame(rows, schema={"start_day": pl.Int64, "end_day": pl.Int64})


def panel_chart(
    *,
    title: str,
    rows: pl.DataFrame,
    losses: pl.DataFrame,
    top_mw: float,
    days: int,
    show_axis: bool,
    missing_note: str | None,
) -> alt.LayerChart:
    """Draw one estimator's panel.

    Args:
        title: The panel's title.
        rows: `window_rows`' result.
        losses: `loss_windows`' result.
        top_mw: The shared top of the vertical axis.
        days: The number of days in the window.
        show_axis: Whether this panel draws the day axis labels (the bottom panel does).
        missing_note: Text to draw when the estimator has no series in this window, else `None`.

    Returns:
        The layered panel.
    """
    x = alt.X(
        "day:Q",
        scale=alt.Scale(domain=[1, days], nice=False),
        axis=alt.Axis(
            title="Day of the window (day 1 is the first day shown)" if show_axis else None,
            labels=show_axis,
            ticks=show_axis,
        ),
    )
    y = alt.Y(
        "maximum_share:Q",
        scale=alt.Scale(domain=[0, top_mw]),
        axis=alt.Axis(title="share of static power", tickCount=3),
    )
    width = CONTENT_WIDTH_PX - 90
    layers: list[alt.Chart] = []
    if losses.height:
        layers.append(
            alt.Chart(losses)
            .mark_rect(color=ocf.DATA_SKY_LIGHT, opacity=0.45, aria=False)
            .encode(x="start_day:Q", x2="end_day:Q")  # ty: ignore[unresolved-attribute]
        )
    layers.append(
        alt.Chart(rows.drop_nulls("maximum_share"))
        .mark_point(size=9, color="#8a8a8a", filled=True, opacity=0.8, aria=False)
        .encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
    )
    line_rows = rows.drop_nulls("capacity_share")
    if line_rows.height:
        layers.append(
            alt.Chart(line_rows)
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2, clip=True, aria=False)
            .encode(x=x, y=alt.Y("capacity_share:Q", scale=alt.Scale(domain=[0, top_mw])))  # ty: ignore[unresolved-attribute]
        )
    if missing_note is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"x": [days / 2], "y": [top_mw * 0.5], "t": [missing_note]}))
            .mark_text(color=ocf.BLACK_1, fontSize=11, aria=False)
            .encode(x="x:Q", y="y:Q", text="t:N")  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(  # ty: ignore[invalid-return-type]
        width=width,
        height=PANEL_HEIGHT_PX,
        title=alt.TitleParams(title, anchor="start", fontSize=12),
    )


def estimate_figure(
    *,
    record: CapacityRecord,
    plant: int,
    window: tuple[int, int],
    capacities: Mapping[str, np.ndarray | None],
    events: Sequence[PlantedEvent],
    number: str,
    title: str,
    subtitle: Sequence[str],
    panel_set: Sequence[EstimatorPanel] = ESTIMATOR_PANELS,
) -> alt.VConcatChart:
    """Stack the seven estimator panels for one plant and window.

    Args:
        record: The record whose power is drawn (planted or original).
        plant: The plant's index.
        window: The first day index (inclusive) and the last (exclusive).
        capacities: Each estimator's capacity per day in megawatts, by `EstimatorPanel.key`, which
            the function divides by the plant's static power; a
            missing key or `None` draws the data alone with a note.
        events: The planted events of the record, whose windows are shaded; empty for the original.
        number: The figure's number on its page.
        title: The figure's title.
        subtitle: The subtitle lines.
        panel_set: The panels to draw, top to bottom; the default is `ESTIMATOR_PANELS`.

    Returns:
        The figure.
    """
    static = float(record.static_p99_mw[plant])
    maximum = daily_maximum(record=record, plant=plant) / static
    in_window = maximum[window[0] : window[1]]
    top = HEADROOM * max(1.0, float(np.nanmax(in_window)))
    losses = loss_windows(events=events, plant=plant, window=window)
    days = window[1] - window[0]
    panels = []
    for index, panel in enumerate(panel_set):
        raw = capacities.get(panel.key)
        series = None if raw is None else raw / static
        rows = window_rows(maximum=maximum, capacity=series, window=window)
        has_values = series is not None and bool(np.isfinite(series[window[0] : window[1]]).any())
        panels.append(
            panel_chart(
                title=panel.title,
                rows=rows,
                losses=losses,
                top_mw=top,
                days=days,
                show_axis=index == len(panel_set) - 1,
                missing_note=None if has_values else "no estimate for this window",
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=subtitle,
        figure_planning="exploratory",
    )
