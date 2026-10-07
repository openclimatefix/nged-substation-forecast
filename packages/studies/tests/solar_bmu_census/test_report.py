"""Tests for the pure functions of `studies/solar_bmu_census/report.py`."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
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


def test_shape_errors_split_by_whether_the_bmu_ran_at_generation_capacity() -> None:
    single = pl.DataFrame(
        {
            "elexon_bmu_id": ["A", "B", "C"],
            "generation_capacity_mw": [100.0, 100.0, 100.0],
            "p99_output_mw": [99.0, 98.0, 70.0],
        }
    )
    shape = pl.DataFrame(
        {"elexon_bmu_id": ["A", "B", "C"], "error_at_base_ratio": [0.10, -0.20, 0.30]}
    )
    table = report.shape_error_by_subset_table(by_shape={"s": shape}, single=single)
    rows = {row["subset"]: row for row in table.iter_rows(named=True)}
    ran = rows["P99 within 2% of Generation Capacity"]
    other = rows["other BMUs"]
    assert (ran["bmus"], ran["mean absolute error"], ran["largest absolute error"]) == (
        2,
        0.15,
        0.2,
    )
    assert (other["bmus"], other["mean absolute error"]) == (1, 0.3)


def _register_row(
    *, bmu_id: str, unit_type: str, gsp: str | None, demand: str
) -> dict[str, object]:
    return {
        "elexonBmUnit": bmu_id,
        "bmUnitType": unit_type,
        "fuelType": None,
        "gspGroupId": gsp,
        "demandCapacity": demand,
    }


def test_the_aggregate_register_table_counts_what_the_register_records() -> None:
    reference = [
        _register_row(bmu_id="2__ABGAS000", unit_type="G", gsp="_A", demand="-3.0"),
        _register_row(bmu_id="2__BBGAS001", unit_type="S", gsp="_B", demand="0.0"),
        _register_row(bmu_id="V__CFLEX001", unit_type="V", gsp="_D", demand="0.0"),
    ]
    aggregates = pl.DataFrame({"elexon_bmu_id": ["2__ABGAS000", "2__BBGAS001", "V__CFLEX001"]})
    table = report.aggregate_register_table(aggregates=aggregates, reference=reference)
    values = dict(table.iter_rows())
    assert values["Aggregate census BMUs `2__` with register type G"] == "1"
    assert values["Aggregate census BMUs `2__` with register type S"] == "1"
    assert values["Aggregate census BMUs `V__` with register type V"] == "1"
    assert values["Aggregate census BMUs with a fuel type in the register"] == "0"
    # The V__ BMU's fourth character is C, but its GSP group is _D.
    assert values[
        "`2__` and `V__` census BMUs whose fourth character is their GSP group letter"
    ] == ("2 of 3")
    assert values["`G` type census BMUs with a negative Demand Capacity"] == "1"
    assert values["Register `2__` BMUs whose identifier ends 000, of type G"] == "1 of 1"
    assert values["Register `2__` BMUs whose identifier does not end 000, of type S"] == "1 of 1"
    assert values["`S` type aggregate census BMUs with a Demand Capacity of zero"] == "1 of 1"


def test_the_lccc_table_counts_current_c_bmus_and_probes_the_unlisted_solar_ones() -> None:
    def mapping_row(cfd: str, bmu: str, ended: str = "") -> dict[str, object]:
        return {"CFD_Id": cfd, "BMU_Id": bmu, "Effective_date_to": ended}

    def unit(cfd: str, technology: str) -> dict[str, object]:
        return {
            "CFD_ID": cfd,
            "Name_of_CFD_Unit": f"Unit {cfd}",
            "Technology_Type": technology,
            "Transmission_or_Distribution_connection": "Distribution",
            "Status": "Live",
            "Maximum_Contract_Capacity_MW": "10",
        }

    seen: list[list[str]] = []

    def probe(ids: list[str]) -> int:
        seen.append(ids)
        return 0

    table = report.lccc_table(
        mapping=[
            mapping_row("A", "C__SOLAR1"),
            mapping_row("B", "C__SOLAR2"),
            mapping_row("C", "C__WIND01"),
            mapping_row("D", "C__OLD", ended="2025-01-01"),
        ],
        portfolio=[
            unit("A", "Solar PV"),
            unit("B", "Solar PV"),
            unit("C", "Onshore Wind"),
            unit("D", "Solar PV"),
        ],
        reference=[{"elexonBmUnit": "C__SOLAR1"}],
        census=pl.DataFrame({"elexon_bmu_id": ["C__SOLAR1"], "scope": ["single-site"]}),
        today=date(2026, 10, 7),
        probe=probe,
        probe_label="in the probe week",
    )
    values = dict(table.iter_rows())
    assert values["`C__` BMUs with a current CfD identifier in the mapping"] == "3"
    assert values["Of those, BMUs whose CfD unit is Solar PV"] == "2"
    assert values["`C__` BMUs in the mapping that the BMU register lists"] == "1"
    assert values["Positive control: C__SOLAR1 has B1610 rows in the probe week"] == "0"
    assert seen == [["C__SOLAR2"], ["C__SOLAR1"]]


def test_the_probe_week_starts_three_months_before_the_window_ends() -> None:
    window = report.Window(
        start=datetime(2025, 9, 1, tzinfo=UTC), end=datetime(2026, 9, 1, tzinfo=UTC)
    )
    assert report.probe_week(window=window) == (date(2026, 6, 1), date(2026, 6, 8))
    early = report.Window(
        start=datetime(2024, 2, 1, tzinfo=UTC), end=datetime(2025, 2, 1, tzinfo=UTC)
    )
    assert report.probe_week(window=early) == (date(2024, 11, 1), date(2024, 11, 8))


def test_runs_of_low_output_days_are_found_by_the_half_hour_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Day 1 has output, days 2 and 3 are below the threshold, day 4 has output, day 5 is low.
    # Each half-hour ends at 12:00 and so starts on the same UTC day.
    ends = [datetime(2026, 1, d, 12, tzinfo=UTC) for d in (1, 2, 3, 4, 5)]
    frame = pl.DataFrame({"half_hour_end_time": ends, "output_mwh": [10.0, 0.1, 0.2, 5.0, 0.0]})
    frame.write_parquet(tmp_path / "C__X_w.parquet")
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)
    runs = report.low_output_runs_table(bmu_id="C__X", window_label="w")
    assert runs["first_day"].to_list() == [date(2026, 1, 2), date(2026, 1, 5)]
    assert runs["days"].to_list() == [2, 1]
    assert runs["largest_daily_maximum_mw"].to_list() == [0.4, 0.0]


def test_the_capped_period_table_lists_the_days_above_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ends = [datetime(2026, 3, d, 12, tzinfo=UTC) for d in (1, 2, 3, 4)]
    frame = pl.DataFrame({"half_hour_end_time": ends, "output_mwh": [10.0, 12.0, 25.0, 0.1]})
    frame.write_parquet(tmp_path / "C__X_w.parquet")
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)
    table = report.capped_period_table(
        bmu_id="C__X", window_label="w", first=date(2026, 3, 1), last=date(2026, 3, 4), cap_mw=30.0
    )
    values = dict(table.iter_rows())
    assert values["Days from 2026-03-01 to 2026-03-04"] == "4"
    assert values["Days with a largest output below 1.5 MW"] == "1"
    assert values["Days with a largest output above 30.0 MW"] == "2026-03-03 (50.0 MW)"
    assert values["Highest daily maximum on the other days (MW)"] == "24.0"
    assert values["Lowest daily maximum on the other days (MW)"] == "20.0"
