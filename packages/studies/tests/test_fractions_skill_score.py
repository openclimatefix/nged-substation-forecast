import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.fractions_skill_score import (
    fss_from,
    monthly_components,
    on_a_complete_hourly_grid,
)

# One site, 30 days, a 3-hour spike each day, scored against a threshold the spike clears. A
# centred rolling window is easy to get wrong by one step, so the implementation is driven with
# forecasts whose right answer is known.
N_DAYS = 30
SPIKE_HOURS = (12, 13, 14)
SPIKE_MW = 10.0
THRESHOLD_MW = 5.0
START = datetime(2025, 6, 1, tzinfo=UTC)

# The tolerances, as the window width each is computed at. A width of 1 is the point score.
WINDOW_FOR_TOLERANCE = {0: 1, 1: 3, 2: 5, 4: 9}


def _stamps() -> list[datetime]:
    return [START + timedelta(hours=hour) for hour in range(N_DAYS * 24)]


def _observed() -> list[float]:
    return [SPIKE_MW if stamp.hour in SPIKE_HOURS else 0.0 for stamp in _stamps()]


def _late_by(hours: int) -> list[float]:
    observed = _observed()
    return [0.0] * hours + observed[:-hours] if hours else observed


def _frame(stamps: list[datetime], observed: list[float], forecast: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A"] * len(stamps),
            "time": stamps,
            "power_mw": observed,
            "forecast_mw": forecast,
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )


def _gridded(frame: pl.DataFrame) -> pl.DataFrame:
    return on_a_complete_hourly_grid(
        frame=frame, thresholds=pl.DataFrame({"site": ["A"], "threshold_mw": [THRESHOLD_MW]})
    )


def _score(*, forecast: list[float], window_hours: int) -> float:
    gridded = _gridded(_frame(_stamps(), _observed(), forecast))
    components = monthly_components(gridded=gridded, window_hours=window_hours)
    return fss_from(
        squared_difference=float(components["squared_difference"].sum()),
        reference=float(components["reference"].sum()),
    )


@pytest.mark.parametrize(
    ("tolerance_hours", "identical", "one_hour_late", "three_hours_late"),
    [(0, 1.0, 0.667, 0.0), (1, 1.0, 0.842, 0.211), (2, 1.0, 0.919, 0.486), (4, 1.0, 0.959, 0.741)],
)
def test_the_score_credits_timing_at_each_tolerance(
    tolerance_hours: int, identical: float, one_hour_late: float, three_hours_late: float
):
    window_hours = WINDOW_FOR_TOLERANCE[tolerance_hours]

    assert _score(forecast=_late_by(0), window_hours=window_hours) == pytest.approx(
        identical, abs=5e-4
    )
    assert _score(forecast=_late_by(1), window_hours=window_hours) == pytest.approx(
        one_hour_late, abs=5e-4
    )
    assert _score(forecast=_late_by(3), window_hours=window_hours) == pytest.approx(
        three_hours_late, abs=5e-4
    )


@pytest.mark.parametrize("tolerance_hours", list(WINDOW_FOR_TOLERANCE))
def test_a_wider_window_never_rescues_a_forecast_twelve_hours_late(tolerance_hours: int):
    # The control the other three rows are read against: without it, every recovery along a row
    # could be the widening window inflating the score rather than the score crediting timing. A
    # spike 12 hours late lies beyond every tolerance, so only a mis-sized or mis-centred window
    # could give it credit.
    assert _score(
        forecast=_late_by(12), window_hours=WINDOW_FOR_TOLERANCE[tolerance_hours]
    ) == pytest.approx(0.0, abs=5e-4)


def test_a_window_spanning_a_gap_contributes_nothing():
    # A site whose rows stop for a night must not have the hours either side joined into one
    # window. The complete-hourly-grid reindex leaves a null across the gap, and the rolling
    # aggregation rejects any window holding one.
    stamps = _stamps()
    with_a_gap = [stamp for stamp in stamps if not (stamp.day == 15 and 0 <= stamp.hour < 12)]
    power = [SPIKE_MW if stamp.hour in SPIKE_HOURS else 0.0 for stamp in with_a_gap]

    complete = monthly_components(
        gridded=_gridded(_frame(with_a_gap, power, power)), window_hours=3
    )["windows"].sum()
    ungapped = monthly_components(
        gridded=_gridded(_frame(stamps, _observed(), _observed())), window_hours=3
    )["windows"].sum()

    # The 12 absent hours cost their own windows plus the one either side that would have spanned
    # the gap, so the loss exceeds the 12 rows removed.
    assert complete == ungapped - 14


def test_no_exceedance_in_either_series_scores_nan():
    assert math.isnan(fss_from(squared_difference=0.0, reference=0.0))
