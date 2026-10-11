"""Tests for the round-3 post hoc follow-up scripts under `studies/lag_features/`.

Each test is written to fail on the defect it names.
"""

from datetime import UTC, datetime, timedelta

import fit_followups3
import followup3_plan
import numpy as np
import polars as pl
import report_followups3


def test_the_round_three_fit_count_is_75_plus_15_plus_360():
    fits = fit_followups3.followup3_fits()

    counts = [fit_followups3.fits_in(fit=fit) for fit in fits]

    assert sum(counts) == 450
    assert [c for f, c in zip(fits, counts, strict=True) if f.scope == "fu3_derated"] == [15] * 5
    assert [c for f, c in zip(fits, counts, strict=True) if f.scope == "fu3_plant_control"] == [15]


def test_the_derated_frame_marks_only_the_plant_s_derated_months_as_not_for_training():
    frame = pl.DataFrame(
        {
            "site": ["A", "A", "A", "B"],
            "month": ["2025-03", "2025-04", "2026-04", "2025-03"],
            "constrained": [False, False, True, False],
        }
    )
    plan = {"site": "A", "months": ["2025-03", "2026-04"]}

    result = fit_followups3.derated_frame(frame=frame, plan=plan)

    assert result["site"].to_list() == ["A", "A", "A"]
    assert result["constrained"].to_list() == [True, False, True]


def test_output_ratios_normalise_by_each_plant_s_own_capacity_and_use_only_full_hours():
    times = [datetime(2025, 3, 1, 10, tzinfo=UTC) + timedelta(hours=h) for h in range(2)]
    # Capacities differ, so the ratio is right only if power is divided by each plant's own
    # capacity: plant A produces half of its 2 MW, the others all of their own 1, 3, 4, 5, 10 MW.
    capacities = {"A": 2.0, "B": 1.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 10.0}
    rows = [
        {
            "site": site,
            "time": time,
            "month": "2025-03",
            "power_mw": capacity * (0.5 if site == "A" else 1.0),
            "effective_capacity_mw": capacity,
        }
        for site, capacity in capacities.items()
        for time in times
    ]
    # Plant A alone has a third hour, at which not all six are scored.
    rows.append(
        {
            "site": "A",
            "time": times[-1] + timedelta(hours=1),
            "month": "2025-03",
            "power_mw": 9.0,
            "effective_capacity_mw": 2.0,
        }
    )
    frame = pl.DataFrame(rows)

    ratios = followup3_plan.output_ratios(frame=frame).filter(pl.col("site") == "A")

    assert ratios["hours"].to_list() == [2]
    assert abs(ratios["ratio"][0] - 0.5) < 1e-12
    # The alternative ratio counts the plant's third hour too, with no other plant scored there.
    assert ratios["n_alt"].to_list() == [2]


def test_the_month_t_interval_has_months_minus_one_degrees_of_freedom():
    months = np.array(["m1", "m2", "m3"] * 2)
    difference = np.array([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]])

    centre, lower, upper, n = report_followups3.month_t_interval(
        difference=difference, months=months, level=95.0
    )

    # Monthly means 1, 2, 3: mean 2, sd 1, t(0.975, 2) = 4.3027, half width 4.3027 / sqrt(3).
    assert n == 3
    assert abs(centre - 2.0) < 1e-12
    assert abs((upper - lower) / 2 - 4.302652729911275 / np.sqrt(3)) < 1e-9
