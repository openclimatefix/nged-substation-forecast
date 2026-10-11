"""Tests for the round-2 post hoc follow-up scripts under `studies/lag_features/`.

Each test is written to fail on the defect it names.
"""

from datetime import UTC, datetime, timedelta

import fit_followups2
import fit_lag_arms
import numpy as np
import polars as pl
import report_followups2


def test_the_round_two_fit_count_is_the_brief_s_1080_in_four_blocks():
    fits = fit_followups2.followup2_fits()

    counts = [fit_followups2.fits_in(fit=fit) for fit in fits]

    assert sum(counts) == 1080
    lines = fit_followups2.fit_count_lines(fits=fits)
    assert [line.split(" fits")[0].split(", ")[-1] for line in lines] == [
        "180",
        "180",
        "360",
        "360",
        "1080",
    ]


def test_a_no_day_of_year_arm_drops_only_that_column():
    base = fit_lag_arms.features_of(arm="W7")
    without = fit_lag_arms.features_of(arm="W7-noDOY")

    assert "day_of_year" in base
    assert "day_of_year" not in without
    assert without == tuple(column for column in base if column != "day_of_year")


def test_the_width_matched_nulls_read_two_and_three_random_lags():
    assert len(fit_lag_arms.features_of(arm="N2-3")) == len(fit_lag_arms.features_of(arm="N2")) + 2
    assert len(fit_lag_arms.features_of(arm="N2-2")) == len(fit_lag_arms.features_of(arm="N2")) + 1


def _hourly_history() -> pl.DataFrame:
    rows = [
        {"site": "A", "time": datetime(year, 6, day, 12, tzinfo=UTC), "power_mw": value}
        for year, value in ((2023, 10.0), (2024, 20.0), (2025, 1000.0))
        for day in range(1, 29)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("power_mw").cast(pl.Float32))


def test_the_past_only_climatology_reads_earlier_years_only_and_is_null_without_any():
    hourly = _hourly_history()
    rows = pl.DataFrame(
        {
            "site": ["A", "A", "A", "A"],
            "time": [
                datetime(2023, 6, 10, 12, tzinfo=UTC),
                datetime(2024, 6, 10, 12, tzinfo=UTC),
                datetime(2025, 6, 10, 12, tzinfo=UTC),
                datetime(2025, 7, 10, 12, tzinfo=UTC),
            ],
        }
    )

    result = report_followups2.past_only_climatology(rows=rows, hourly=hourly)

    # 2023 has no earlier June; 2024 reads 2023 alone; 2025 reads the median of 2023 and 2024,
    # never 2025's own 1000; July of 2025 has no earlier July.
    assert result["past_climatology"].to_list() == [None, 10.0, 15.0, None]


def _aligned(
    *, values: dict[str, list[list[float]]], months: list[str]
) -> report_followups2.Aligned:
    seeds = len(next(iter(values.values())))
    rows = len(months)
    frame = pl.DataFrame(
        {
            "site": ["A"] * rows * seeds,
            "time": [
                datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
                for _ in range(seeds)
                for hour in range(rows)
            ],
            "seed": [seed for seed in range(seeds) for _ in range(rows)],
            "month": months * seeds,
            **{name: [v for seed in series for v in seed] for name, series in values.items()},
        }
    )
    return report_followups2.Aligned(frame=frame, names=list(values))


def test_a_mean_difference_interval_is_centred_on_the_paired_mean():
    months = ["2025-01"] * 3 + ["2025-02"] * 3 + ["2025-03"] * 3
    data = _aligned(
        values={
            "a": [[2.0] * 9, [2.0] * 9],
            "b": [[1.0] * 9, [1.0] * 9],
        },
        months=months,
    )

    result = report_followups2.mean_difference(data=data, treatment="a", reference="b")

    assert result["value"] == 1.0
    assert result["lower_95"] == 1.0
    assert result["upper_95"] == 1.0


def test_where_keeps_the_rows_of_every_series_and_every_seed():
    months = ["2025-01", "2025-01", "2025-02", "2025-02"]
    data = _aligned(values={"a": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]}, months=months)

    kept = data.in_months(months=["2025-02"])

    assert kept.values["a"].tolist() == [[3.0, 4.0], [7.0, 8.0]]
    assert kept.n_rows == 2
    assert np.all(kept.month == "2025-02")
