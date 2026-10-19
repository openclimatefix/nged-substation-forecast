"""Tests of the pinned PV capacity (`studies.dp_pinned_capacity`) on a hand-made vintage table."""

from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.dp_features import FeatureCache, SiteTable
from studies.dp_pinned_capacity import (
    CAPACITY_GROWTH_PER_YEAR,
    HOLD_WEEKS,
    PinnedCapacity,
    pinned_capacity,
)

TABLE: Final[SiteTable] = SiteTable(
    site="S1",
    time_series_id=1,
    units="MW",
    role="mw_long",
    latitude=53.0,
    longitude=0.0,
    effective_capacity_mw=10.0,
    first_usable=datetime(2019, 9, 1, tzinfo=UTC),
)
FIRST_MONDAY: Final[date] = date(2019, 9, 2)


def _vintage(
    *, origin: date, last_week: date, level: float, unobserved_from: date | None = None
) -> pl.DataFrame:
    """One vintage: `level` MW rising 0.01 MW a week from the first Monday to `last_week`."""
    weeks = []
    week = FIRST_MONDAY
    while week <= last_week:
        weeks.append(week)
        week += timedelta(days=7)
    return pl.DataFrame(
        {
            "time_series_id": 1,
            "variant": "frozen",
            "origin_monday": origin,
            "week_start": weeks,
            "capacity_mw": [level + 0.01 * i for i in range(len(weeks))],
            "unobserved": [unobserved_from is not None and w >= unobserved_from for w in weeks],
        }
    )


def test_the_latest_origin_on_or_before_the_cut_is_used():
    vintages = pl.concat(
        [
            _vintage(origin=date(2023, 8, 7), last_week=date(2023, 7, 31), level=1.0),
            _vintage(origin=date(2024, 6, 3), last_week=date(2024, 5, 27), level=2.0),
            _vintage(origin=date(2024, 7, 29), last_week=date(2024, 7, 22), level=3.0),
        ]
    )
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=datetime(2024, 7, 1, tzinfo=UTC))
    assert pinned.source == "vintage"
    assert pinned.origin == date(2024, 6, 3)
    assert pinned.points[0] == pytest.approx(100.0 * 2.0 / 10.0)


def test_the_capacity_after_the_cut_is_the_median_of_the_last_weeks_up_to_the_cut():
    vintages = _vintage(origin=date(2024, 6, 3), last_week=date(2024, 5, 27), level=2.0)
    cut = datetime(2024, 7, 1, tzinfo=UTC)
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    weekly = pinned.points
    assert pinned.held_points == pytest.approx(float(np.median(weekly[-HOLD_WEEKS:])))
    assert pinned.held_points < weekly[-1]


def test_a_row_after_the_cut_or_after_the_last_week_gets_the_held_value(
    dp_synthetic_cache: FeatureCache,
):
    vintages = _vintage(origin=date(2023, 8, 7), last_week=date(2023, 4, 17), level=1.0)
    cut = datetime(2024, 7, 1, tzinfo=UTC)
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    ends = dp_synthetic_cache.end_times
    rows = np.array(
        [
            int(np.searchsorted(ends, np.datetime64("2022-01-05T12:00", "us"))),
            int(np.searchsorted(ends, np.datetime64("2024-03-05T12:00", "us"))),
            int(np.searchsorted(ends, np.datetime64("2025-03-05T12:00", "us"))),
        ]
    )
    values = pinned.at_rows(cache=dp_synthetic_cache, index=rows)
    assert values[0] != pinned.held_points
    assert values[1] == pinned.held_points
    assert values[2] == pinned.held_points


def test_unobserved_weeks_hold_the_last_observed_value_and_are_never_zero():
    vintages = _vintage(
        origin=date(2024, 6, 3),
        last_week=date(2024, 5, 27),
        level=2.0,
        unobserved_from=date(2023, 1, 2),
    )
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=datetime(2024, 7, 1, tzinfo=UTC))
    unobserved = pinned.weeks >= np.datetime64("2023-01-02")
    assert pinned.n_unobserved == int(unobserved.sum())
    assert np.unique(pinned.points[unobserved]).size == 1
    assert pinned.points.min() > 0.0


def test_a_vintage_a_few_weeks_after_the_cut_is_used_as_it_is_but_holds_the_value_at_the_cut():
    vintages = _vintage(origin=date(2023, 8, 7), last_week=date(2023, 7, 31), level=1.0)
    cut = datetime(2023, 7, 1, tzinfo=UTC)
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    assert pinned.source == "early_vintage"
    assert pinned.points[0] == pytest.approx(10.0)
    # The held value is the median of the 13 weeks up to the cut, not of the whole vintage, which
    # would read the five weeks after the cut.
    up_to_cut = pinned.points[pinned.weeks <= np.datetime64("2023-07-01")]
    assert pinned.held_points == pytest.approx(float(np.median(up_to_cut[-HOLD_WEEKS:])))
    assert pinned.held_points != pytest.approx(float(np.median(pinned.points[-HOLD_WEEKS:])))


def test_rows_after_the_cut_get_the_held_value_even_where_the_vintage_has_weeks(
    dp_synthetic_cache: FeatureCache,
):
    vintages = _vintage(origin=date(2023, 8, 7), last_week=date(2023, 7, 31), level=1.0)
    cut = datetime(2023, 7, 1, tzinfo=UTC)
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    ends = dp_synthetic_cache.end_times
    before = int(np.searchsorted(ends, np.datetime64("2023-06-14T12:00", "us")))
    after = int(np.searchsorted(ends, np.datetime64("2023-07-19T12:00", "us")))
    values = pinned.at_rows(cache=dp_synthetic_cache, index=np.array([before, after]))
    monday_before = np.flatnonzero(pinned.weeks == np.datetime64("2023-06-12"))[0]
    assert values[0] == pinned.points[monday_before]
    assert values[1] == pinned.held_points
    assert pinned.weeks[-1] > np.datetime64("2023-07-17")  # the vintage does have that week


def test_a_vintage_far_after_the_cut_scales_its_level_at_the_origin_back_at_the_growth_prior():
    vintages = _vintage(origin=date(2025, 4, 14), last_week=date(2025, 4, 7), level=1.0)
    cut = datetime(2024, 7, 1, tzinfo=UTC)
    pinned = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    assert pinned.source == "growth_prior"
    n_weeks = len(pinned.weeks)
    points = [100.0 * (1.0 + 0.01 * i) / TABLE.effective_capacity_mw for i in range(n_weeks)]
    origin_level = float(np.median(points[-HOLD_WEEKS:]))
    growth = 1.0 + CAPACITY_GROWTH_PER_YEAR
    anchor = pinned.weeks[-1]  # the last week the level refers to
    years_to_cut = float((anchor - np.datetime64("2024-07-01")).astype(float)) / 365.25
    assert pinned.held_points == pytest.approx(origin_level / growth**years_to_cut)
    week = 100
    years_back = float((anchor - pinned.weeks[week]).astype(float)) / 365.25
    assert pinned.points[week] == pytest.approx(origin_level / growth**years_back)
    # The vintage's own weekly shape plays no part: the path is the growth prior alone.
    ratios = pinned.points[1:] / pinned.points[:-1]
    assert np.allclose(ratios, growth ** (7.0 / 365.25))


def test_a_substation_without_a_vintage_or_without_a_week_before_the_cut_is_refused():
    vintages = _vintage(origin=date(2023, 8, 7), last_week=date(2023, 7, 31), level=1.0)
    with pytest.raises(ValueError, match="no `frozen` capacity vintage"):
        pinned_capacity(
            vintages=vintages.with_columns(time_series_id=pl.lit(2)),
            table=TABLE,
            cut=datetime(2024, 7, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="no week on or before"):
        pinned_capacity(
            vintages=_vintage(origin=date(2019, 9, 30), last_week=date(2019, 9, 23), level=1.0),
            table=TABLE,
            cut=datetime(2019, 8, 25, tzinfo=UTC),
        )


def test_the_pinned_capacity_records_its_cut_and_site():
    vintages = _vintage(origin=date(2023, 8, 7), last_week=date(2023, 7, 31), level=1.0)
    cut = datetime(2023, 7, 1, tzinfo=UTC)
    pinned: PinnedCapacity = pinned_capacity(vintages=vintages, table=TABLE, cut=cut)
    assert (pinned.site, pinned.cut) == ("S1", cut)
