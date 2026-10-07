from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from contracts.power_schemas import POWER_TIMESTAMPS_CORRECTED_BEFORE
from studies.power import hourly_from_half_hourly

from studies import power


def _half_hourly(stamps: list[datetime], powers: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A"] * len(stamps),
            "time": stamps,
            "power_mw": powers,
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )


# Four contiguous half-hours, all stamped before the instant NGED corrected the feed. Two is not
# enough: under the shift this test exists to rule out, two contiguous stamps land in *different*
# hours, the complete-hour filter drops both, and an assertion on an empty frame would pass whether
# or not the shift was applied.
BEFORE_THE_REPAIR: list[datetime] = [
    datetime(2026, 3, 20, hour, minute, tzinfo=UTC)
    for hour, minute in ((9, 0), (9, 30), (10, 0), (10, 30))
]


def test_hour_pools_the_two_half_hours_ending_within_it():
    hourly = hourly_from_half_hourly(
        half_hourly=_half_hourly(BEFORE_THE_REPAIR, [1.0, 2.0, 4.0, 8.0])
    )

    # The hour ending 10:00 is the window (09:00, 10:00], so the stamps 09:30 and 10:00. A second
    # 30-minute shift would pool the stamps 10:00 and 10:30 instead, scoring 6.0 — the window
    # (09:30, 10:30], half an hour of the wrong weather, and the fault this asserts against.
    assert hourly["time"].to_list() == [datetime(2026, 3, 20, 10, 0, tzinfo=UTC)]
    assert hourly["power_mw"].to_list() == [3.0]


def test_an_hour_missing_a_half_hour_is_dropped():
    stamps = [BEFORE_THE_REPAIR[0], *BEFORE_THE_REPAIR[2:]]

    hourly = hourly_from_half_hourly(half_hourly=_half_hourly(stamps, [1.0, 4.0, 8.0]))

    # Each stamp lands in a different hour, so no hour has two readings.
    assert hourly.is_empty()


def test_the_orphan_half_hour_at_the_repair_boundary_is_dropped():
    # A repaired series has no reading at POWER_TIMESTAMPS_CORRECTED_BEFORE - 30 min, because NGED
    # never published that half-hour. The repaired reading 60 minutes before the instant is
    # therefore alone in the hour ending at the missing stamp, and the complete-hour filter is what
    # removes it. The hours either side are complete.
    boundary = POWER_TIMESTAMPS_CORRECTED_BEFORE
    stamps = [boundary + timedelta(minutes=offset) for offset in (-120, -90, -60, 0, 30)]

    hourly = hourly_from_half_hourly(half_hourly=_half_hourly(stamps, [1.0, 2.0, 4.0, 8.0, 16.0]))

    assert hourly["time"].to_list() == [
        boundary - timedelta(minutes=90),
        boundary + timedelta(minutes=30),
    ]
    assert hourly["power_mw"].to_list() == [1.5, 12.0]


@pytest.mark.parametrize(
    ("powers", "expected"), [([0.0, 1.0], True), ([1.0, 2.0], False)], ids=["zero", "no_zero"]
)
def test_has_zero_half_hour_flags_an_exact_zero(powers: list[float], expected: bool):
    stamps = BEFORE_THE_REPAIR[1:3]

    hourly = hourly_from_half_hourly(half_hourly=_half_hourly(stamps, powers))

    assert hourly["has_zero_half_hour"].to_list() == [expected]


def test_scan_power_returns_only_the_rows_no_cleaning_rule_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    delta_path = tmp_path / "cleaned_power_time_series.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1], dtype=pl.Int32),
            "time": [
                datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            ],
            "power": pl.Series([1.0, 0.0], dtype=pl.Float32),
            "drop_reason": [None, "substation_zero"],
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    ).write_delta(delta_path)
    monkeypatch.setattr(power, "CLEANED_POWER_DELTA_URI", str(delta_path))

    kept = power.scan_power().collect()

    assert kept.columns == ["time_series_id", "time", "power"]
    assert kept["power"].to_list() == [1.0]


def test_scan_power_stops_before_midnight_on_final_test_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    delta_path = tmp_path / "cleaned_power_time_series.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 1], dtype=pl.Int32),
            "time": [
                datetime(2026, 6, 30, 23, 30, tzinfo=UTC),
                datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
                datetime(2026, 7, 1, 0, 30, tzinfo=UTC),
            ],
            "power": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
            "drop_reason": [None, None, None],
        },
        schema_overrides={"time": pl.Datetime("us", "UTC"), "drop_reason": pl.String},
    ).write_delta(delta_path)
    monkeypatch.setattr(power, "CLEANED_POWER_DELTA_URI", str(delta_path))
    monkeypatch.setattr(
        power, "load_cv_config", lambda _path: SimpleNamespace(final_test_start=date(2026, 7, 1))
    )

    kept = power.scan_power().collect()

    assert kept["power"].to_list() == [1.0]
