"""Tests for the pure functions of `studies/solar_bmu_census/report.py`."""

from datetime import UTC, datetime, timedelta

import polars as pl
import report


def _output(*, mwh: list[float]) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "half_hour_end_time": [start + timedelta(minutes=30 * i) for i in range(len(mwh))],
            "output_mwh": mwh,
        }
    )


def test_the_coincident_peak_is_below_the_sum_of_the_peaks_when_the_peaks_differ() -> None:
    # BMU A peaks at 4 MWh (8 MW) in the first half-hour, BMU B at 5 MWh (10 MW) in the third.
    a = _output(mwh=[4.0, 1.0, 0.0])
    b = _output(mwh=[0.0, 2.0, 5.0])
    # The sums by half-hour are 4, 3, 5 MWh, so the coincident peak is 10 MW; the peaks add to 18.
    assert report.coincident_peak_mw(outputs=[a, b]) == 10.0


def test_the_coincident_peak_equals_the_sum_of_the_peaks_when_the_peaks_coincide() -> None:
    a = _output(mwh=[1.0, 4.0, 0.0])
    b = _output(mwh=[2.0, 5.0, 0.0])
    assert report.coincident_peak_mw(outputs=[a, b]) == 18.0


def test_a_bmu_without_a_row_in_a_half_hour_adds_nothing_to_that_half_hour() -> None:
    a = _output(mwh=[1.0, 1.0, 1.0])
    b = _output(mwh=[2.0]).with_columns(
        half_hour_end_time=pl.col("half_hour_end_time") + timedelta(minutes=60)
    )
    # B's only row falls in the third half-hour: 1 + 2 = 3 MWh, which is 6 MW, above 2 and 4 MW.
    assert report.coincident_peak_mw(outputs=[a, b]) == 6.0


def test_the_coincident_peak_of_no_series_is_zero() -> None:
    assert report.coincident_peak_mw(outputs=[]) == 0.0


def test_the_aggregate_capacity_sums_count_only_bmus_with_a_value() -> None:
    aggregates = pl.DataFrame(
        {
            "generation_capacity_mw": [10.0, 5.0],
            "igcpu_installed_capacity_mw": [16.0, None],
            "tec_mw": [None, None],
            "largest_mel_mw": [0.0, 3.0],
            "repd_installed_capacity_mw": [None, None],
        }
    )
    table = report.aggregate_capacity_table(aggregates=aggregates)
    rows = {row["capacity column"]: row for row in table.iter_rows(named=True)}
    assert rows["generation_capacity_mw"]["sum (MW)"] == 15.0
    assert rows["igcpu_installed_capacity_mw"]["bmus with a value"] == 1
    assert rows["igcpu_installed_capacity_mw"]["sum (MW)"] == 16.0
    assert rows["tec_mw"]["bmus with a value"] == 0
    assert rows["largest_mel_mw"]["bmus with a value"] == 2
