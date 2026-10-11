"""Tests for the constants and helpers of the scripts under `studies/lag_features/`.

Each test is written to fail on the defect it names.
"""

import fit_lag_arms
import polars as pl
from build_lag_frame import (
    SMOKE_ROWS_PER_SITE_MONTH,
    scored_months,
    shifted_months,
    smoke_subsample,
)


def test_the_reproduction_tolerance_is_the_planned_three_hundredths_of_a_point():
    assert fit_lag_arms.REPRODUCTION_TOLERANCE_PERCENT == 0.03
    assert fit_lag_arms.REPRODUCTION_MAE_PERCENT == 8.771


def test_the_positive_control_shifts_exactly_nine_of_the_eighteen_scored_months():
    scored = scored_months()

    assert len(scored) == 18
    assert len(shifted_months() & set(scored)) == 9


def test_the_positive_control_is_reproducible_between_calls():
    assert shifted_months() == shifted_months()


def test_the_smoke_subsample_keeps_every_plant_month_with_at_most_the_stated_rows():
    rows = [
        {"site": site, "month": month, "time": index}
        for site in ("A", "B")
        for month in ("2025-01", "2025-02")
        for index in range(50)
    ]

    kept = smoke_subsample(frame=pl.DataFrame(rows))

    counts = kept.group_by("site", "month").len()
    assert counts.height == 4
    assert counts["len"].to_list() == [SMOKE_ROWS_PER_SITE_MONTH] * 4
