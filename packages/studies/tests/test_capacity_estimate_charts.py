"""Tests of the estimator-over-power figure's data and its anonymity."""

import json
import re

import numpy as np
import pytest
from studies.capacity_estimate_charts import (
    ESTIMATOR_PANELS,
    daily_maximum,
    estimate_figure,
    loss_windows,
    window_rows,
)
from studies.capacity_estimators import CapacityRecord, synthetic_record
from studies.capacity_recovery import PlantedEvent

DAYS = 300
WINDOW = (100, 220)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _record() -> CapacityRecord:
    return synthetic_record(plants=2, days=DAYS, seed=1)


def _capacities(*, record: CapacityRecord, causal: bool = True) -> dict[str, np.ndarray | None]:
    static = np.full(DAYS, record.static_p99_mw[0])
    return {panel.key: static * 0.9 for panel in ESTIMATOR_PANELS} | (
        {} if causal else {"llm_causal": None}
    )


def _figure(*, events: list[PlantedEvent], causal: bool = True) -> dict:
    record = _record()
    figure = estimate_figure(
        record=record,
        plant=0,
        window=WINDOW,
        capacities=_capacities(record=record, causal=causal),
        events=events,
        number="1",
        title="A title",
        subtitle=["A subtitle."],
    )
    return figure.to_dict()


def test_the_daily_maximum_is_the_highest_hourly_reading_and_nan_without_one() -> None:
    record = _record()
    power = record.power_mw.copy()
    power[0, 5] = np.nan
    power[0, 6, 12] = 99.0
    maximum = daily_maximum(record=record.with_power(power_mw=power), plant=0)
    assert np.isnan(maximum[5])
    assert maximum[6] == 99.0
    assert maximum[7] == pytest.approx(np.max(record.power_mw[0, 7]))


def test_window_rows_count_days_from_one_and_leave_a_missing_estimator_null() -> None:
    maximum = np.arange(DAYS, dtype=float)
    rows = window_rows(maximum=maximum, capacity=None, window=(10, 14))
    assert rows["day"].to_list() == [1, 2, 3, 4]
    assert rows["maximum_share"].to_list() == [10.0, 11.0, 12.0, 13.0]
    assert rows["capacity_share"].null_count() == 4


def test_only_the_plants_own_overlapping_events_are_shaded_and_clipped_to_the_window() -> None:
    events = [
        PlantedEvent(plant=0, shape="step", size=0.2, first_day=90, month_days=30),
        PlantedEvent(plant=1, shape="step", size=0.2, first_day=110, month_days=30),
        PlantedEvent(plant=0, shape="step", size=0.2, first_day=250, month_days=30),
    ]
    shaded = loss_windows(events=events, plant=0, window=WINDOW)
    assert shaded.rows() == [(1, 21)]
    assert loss_windows(events=[], plant=0, window=WINDOW).height == 0


def test_the_figure_has_seven_panels_on_one_vertical_range_and_no_calendar_date() -> None:
    spec = _figure(
        events=[PlantedEvent(plant=0, shape="step", size=0.2, first_day=130, month_days=30)]
    )
    panels = spec["vconcat"]
    assert len(panels) == len(ESTIMATOR_PANELS) == 7
    domains = {
        json.dumps(layer["encoding"]["y"]["scale"]["domain"])
        for panel in panels
        for layer in panel["layer"]
        if "y" in layer.get("encoding", {}) and "scale" in layer["encoding"]["y"]
    }
    assert len(domains) == 1
    assert not ISO_DATE.search(json.dumps(spec))
    assert all(layer["mark"].get("aria") is False for panel in panels for layer in panel["layer"])


def test_a_panel_without_an_estimate_says_so_and_a_quiet_window_has_no_shading() -> None:
    note = "no estimate for this window"
    text = json.dumps(_figure(events=[], causal=False))
    assert text.count(note) == 1
    assert note not in json.dumps(_figure(events=[]))
    assert '"type": "rect"' not in text
    shaded = json.dumps(
        _figure(
            events=[PlantedEvent(plant=0, shape="step", size=0.2, first_day=130, month_days=30)]
        )
    )
    assert '"type": "rect"' in shaded


def test_a_figure_draws_the_panels_it_is_given_in_order() -> None:
    from studies.capacity_estimate_charts import ROUND2_PANELS

    record = _record()
    figure = estimate_figure(
        record=record,
        plant=0,
        window=WINDOW,
        capacities={panel.key: np.full(DAYS, record.static_p99_mw[0]) for panel in ROUND2_PANELS},
        events=[],
        number="1",
        title="t",
        subtitle=["s"],
        panel_set=ROUND2_PANELS,
    ).to_dict()
    titles = [panel["title"]["text"] for panel in figure["vconcat"]]
    assert titles == [panel.title for panel in ROUND2_PANELS]
    assert len(titles) == 8


def test_a_figure_is_drawn_as_a_share_of_static_power_and_never_in_megawatts() -> None:
    record = _record()
    static = float(record.static_p99_mw[0])
    spec = _figure(events=[])
    text = json.dumps(spec)
    assert "share of static power" in text
    assert " MW" not in text
    assert '"MW"' not in text
    top = max(
        layer["encoding"]["y"]["scale"]["domain"][1]
        for panel in spec["vconcat"]
        for layer in panel["layer"]
        if "scale" in layer.get("encoding", {}).get("y", {})
    )
    assert top < 3.0
    assert top > 1.0
    assert static > 3.0 or static > 0.0
    values = [
        row["maximum_share"]
        for dataset in spec["datasets"].values()
        for row in dataset
        if "maximum_share" in row and row["maximum_share"] is not None
    ]
    assert values
    assert max(values) < 3.0
