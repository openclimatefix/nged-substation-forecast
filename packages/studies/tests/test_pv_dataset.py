from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from studies import pv_dataset

MIN_ROWS = int(pv_dataset.MIN_YEARS_OF_READINGS * 365.25 * 48)
START = datetime(2024, 1, 1, tzinfo=UTC)


def _serve_tables(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, series: dict[int, tuple[str, int, float]]
) -> None:
    """Point the roster readers at synthetic tables.

    Args:
        monkeypatch: The pytest fixture, used to swap the metadata path and the Delta scan.
        tmp_path: Where the metadata parquet is written.
        series: For each `time_series_id`, its technology, its row count and its capacity.
    """
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {
            "time_series_id": list(series),
            "time_series_type": [kind for kind, _, _ in series.values()],
            "latitude": [52.0 + index for index in range(len(series))],
            "longitude": [-1.0] * len(series),
        }
    ).write_parquet(metadata_path)
    power = pl.DataFrame(
        {
            "time_series_id": [
                identifier for identifier, (_, rows, _) in series.items() for _ in range(rows)
            ],
        }
    )
    capacity = pl.DataFrame(
        {
            "time_series_id": list(series),
            "time": [START] * len(series),
            "effective_capacity_mw": [capacity for _, _, capacity in series.values()],
        }
    )
    tables = {
        pv_dataset.POWER_DELTA_URI: power.lazy(),
        pv_dataset.CAPACITY_DELTA_URI: capacity.lazy(),
    }
    monkeypatch.setattr(pv_dataset, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(pl, "scan_delta", lambda uri, **_: tables[str(uri)])


def test_the_pv_roster_labels_the_long_enough_pv_series_and_leaves_out_wind_and_short_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    series = {identifier: ("PV", MIN_ROWS + 1, 10.0 + identifier) for identifier in range(1, 7)}
    series[7] = ("PV", MIN_ROWS - 1, 5.0)
    series[8] = ("Wind", MIN_ROWS + 1, 7.0)
    _serve_tables(monkeypatch=monkeypatch, tmp_path=tmp_path, series=series)

    roster = pv_dataset.pv_sites()

    assert roster["time_series_id"].to_list() == [1, 2, 3, 4, 5, 6]
    assert sorted(roster["site"].to_list()) == ["A", "B", "C", "D", "E", "F"]
    assert roster["effective_capacity_mw"].to_list() == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def test_the_wind_roster_labels_only_the_wind_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    series = dict.fromkeys(range(1, 4), ("Wind", MIN_ROWS + 1, 3.0))
    series[4] = ("PV", MIN_ROWS + 1, 5.0)
    _serve_tables(monkeypatch=monkeypatch, tmp_path=tmp_path, series=series)

    roster = pv_dataset.wind_sites()

    assert roster["time_series_id"].to_list() == [1, 2, 3]
    assert sorted(roster["site"].to_list()) == ["W1", "W2", "W3"]


def test_hourly_power_is_period_ending_and_covers_only_the_roster(
    monkeypatch: pytest.MonkeyPatch,
):
    half_hours = [START + timedelta(minutes=30 * step) for step in range(1, 5)]
    power = pl.DataFrame(
        {
            "time_series_id": [1] * 4 + [2] * 4,
            "time": half_hours * 2,
            "power": [2.0, 4.0, 0.0, 6.0, 9.0, 9.0, 9.0, 9.0],
        }
    )
    monkeypatch.setattr(pl, "scan_delta", lambda uri, **_: power.lazy())
    roster = pl.DataFrame({"time_series_id": [1], "site": ["A"]})

    hourly = pv_dataset.solar_hourly_power(sites=roster)

    assert hourly.sort("time").to_dicts() == [
        {
            "site": "A",
            "time": START + timedelta(hours=1),
            "power_mw": 3.0,
            "has_zero_half_hour": False,
        },
        {
            "site": "A",
            "time": START + timedelta(hours=2),
            "power_mw": 3.0,
            "has_zero_half_hour": True,
        },
    ]
