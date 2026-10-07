from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from studies.anonymise import (
    LABEL_PERMUTATION_SEED,
    SITE_LABELS,
    WIND_LABEL_PERMUTATION_SEED,
    WIND_SITE_LABELS,
    site_labels_for,
)

from studies import pv_dataset

MIN_ROWS = int(pv_dataset.MIN_YEARS_OF_READINGS * 365.25 * 48)
START = datetime(2024, 1, 1, tzinfo=UTC)


def _serve_tables(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    series: dict[int, tuple[str, int, list[float]]],
) -> None:
    """Point the site list readers at synthetic tables.

    Args:
        monkeypatch: The pytest fixture, used to swap the metadata path and the Delta scan.
        tmp_path: Where the metadata parquet is written.
        series: For each `time_series_id`, its technology, its row count, and its capacities from
            the oldest to the newest, each one day after the last. An empty list means the
            capacity table holds nothing for the series.
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
    capacity_rows = [
        (identifier, START + timedelta(days=day), value)
        for identifier, (_, _, capacities) in series.items()
        for day, value in reversed(list(enumerate(capacities)))
    ]
    capacity = pl.DataFrame(
        capacity_rows,
        schema={
            "time_series_id": pl.Int64,
            "time": pl.Datetime("us", "UTC"),
            "effective_capacity_mw": pl.Float64,
        },
        orient="row",
    )
    tables = {
        pv_dataset.POWER_DELTA_URI: power.lazy(),
        pv_dataset.CAPACITY_DELTA_URI: capacity.lazy(),
    }
    monkeypatch.setattr(pv_dataset, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(pl, "scan_delta", lambda uri, **_: tables[str(uri)])


def test_the_pv_site_list_labels_the_long_enough_pv_series_with_their_latest_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    series = {
        identifier: ("PV", MIN_ROWS + 1, [99.0, 10.0 + identifier]) for identifier in range(1, 7)
    }
    series[7] = ("PV", MIN_ROWS - 1, [5.0])  # too little history
    series[8] = ("Wind", MIN_ROWS + 1, [7.0])  # wrong technology
    series[9] = ("PV", MIN_ROWS + 1, [])  # no capacity recorded
    _serve_tables(monkeypatch=monkeypatch, tmp_path=tmp_path, series=series)

    site_list = pv_dataset.pv_sites()

    assert site_list["time_series_id"].to_list() == [1, 2, 3, 4, 5, 6]
    assert site_list["effective_capacity_mw"].to_list() == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    expected_labels = site_labels_for(
        eligible_ids=[1, 2, 3, 4, 5, 6], labels=SITE_LABELS, seed=LABEL_PERMUTATION_SEED
    )
    assert dict(site_list.select("time_series_id", "site").iter_rows()) == expected_labels


def test_the_wind_site_list_labels_only_the_wind_series_with_the_wind_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    series = {identifier: ("Wind", MIN_ROWS + 1, [3.0]) for identifier in range(1, 4)}
    series[4] = ("PV", MIN_ROWS + 1, [5.0])
    _serve_tables(monkeypatch=monkeypatch, tmp_path=tmp_path, series=series)

    site_list = pv_dataset.wind_sites()

    expected_labels = site_labels_for(
        eligible_ids=[1, 2, 3], labels=WIND_SITE_LABELS, seed=WIND_LABEL_PERMUTATION_SEED
    )
    assert dict(site_list.select("time_series_id", "site").iter_rows()) == expected_labels


def test_hourly_power_is_period_ending_and_covers_only_the_site_list(
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
    site_list = pl.DataFrame({"time_series_id": [1], "site": ["A"]})

    hourly = pv_dataset.solar_hourly_power(sites=site_list)

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
