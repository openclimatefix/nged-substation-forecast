import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import obstore
import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data.storage import (
    SMALL_FILE_SIZE_THRESHOLD_BYTES,
    _process_file_listing,
    _ProcessedFileListing,
    _RawFileListItem,
    coverage_from_power,
    download_metadata_of_series_without_new_files,
    remove_small_files_from_listing,
    scan_cleaned_power,
    select_new_rows,
    time_series_coverage,
    upsert_metadata,
)


def _file_listing(filesize_bytes: list[int]) -> pt.DataFrame[_ProcessedFileListing]:
    """Build a minimal, valid `_ProcessedFileListing` frame with the given file sizes.

    `remove_small_files_from_listing` only reads `filesize_bytes`, so every other column is a
    fixed placeholder.
    """
    n = len(filesize_bytes)
    raw = pl.DataFrame(
        {
            "path": [f"timeseries/0_1/TimeSeries_1_{i}.json" for i in range(n)],
            "filesize_bytes": pl.Series(filesize_bytes, dtype=pl.Int64),
            "time_series_id": pl.Series([1] * n, dtype=pl.Int32),
            "start_time": pl.Series([datetime(2026, 1, 1, tzinfo=UTC)] * n).cast(
                UTC_DATETIME_DTYPE
            ),
            "end_time": pl.Series([datetime(2026, 1, 1, 6, tzinfo=UTC)] * n).cast(
                UTC_DATETIME_DTYPE
            ),
        }
    )
    return pt.DataFrame(raw).set_model(_ProcessedFileListing).validate()


def test_upsert_metadata_new_file(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # Create dummy metadata
    metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Test Substation",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(metadata, str(metadata_path))

    assert metadata_path.exists()

    # Read back and verify
    read_metadata = pl.read_parquet(metadata_path)
    assert read_metadata.height == 1
    assert read_metadata["time_series_id"].item() == 1


def test_upsert_metadata_creates_missing_parent_dir(tmp_path: Path):
    """A first-ever run writes into a data root whose subdirectory doesn't exist yet; the create
    branch must make the parent dir rather than raising FileNotFoundError from write_parquet."""
    metadata_path = tmp_path / "NGED" / "metadata.parquet"  # parent NGED/ does not exist
    metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Test Substation",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(metadata, str(metadata_path))

    assert metadata_path.exists()


def test_upsert_metadata_merge(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # Create initial metadata
    initial_metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Old Name",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    initial_metadata.write_parquet(metadata_path)

    # Create new metadata for same ID
    new_metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "New Name",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(new_metadata, str(metadata_path))

    # Read back and verify
    read_metadata = pl.read_parquet(metadata_path)
    assert read_metadata.height == 1
    assert read_metadata["time_series_name"].item() == "New Name"


def test_upsert_metadata_returns_diff(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # 1. Create initial metadata
    initial_data = [
        {
            "time_series_id": 1,
            "time_series_name": "ID 1 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 1,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        {
            "time_series_id": 2,
            "time_series_name": "ID 2 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 2,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
    ]
    initial_metadata = pt.DataFrame(initial_data).set_model(TimeSeriesMetadata).cast().validate()
    initial_metadata.write_parquet(metadata_path)

    # 2. Create new metadata
    new_data = [
        # Identical to ID 1
        {
            "time_series_id": 1,
            "time_series_name": "ID 1 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 1,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        # Updated ID 2
        {
            "time_series_id": 2,
            "time_series_name": "ID 2 - Updated",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 2,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        # New ID 3
        {
            "time_series_id": 3,
            "time_series_name": "ID 3 - New",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 3,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
    ]
    new_metadata = pt.DataFrame(new_data).set_model(TimeSeriesMetadata).cast().validate()

    # 3. Call upsert_metadata
    stats = upsert_metadata(new_metadata, str(metadata_path))

    # 4. Assertions
    assert stats["metadata_n_new_TimeSeriesIDs"] == 1
    assert stats["metadata_n_updated_TimeSeriesIDs"] == 1
    assert set(stats["metadata_updated_TimeSeriesIDs"]) == {2}

    # Verify file content
    final_metadata = pl.read_parquet(metadata_path)
    assert final_metadata.height == 3
    assert (
        final_metadata.filter(pl.col("time_series_id") == 1)["time_series_name"].item()
        == "ID 1 - Original"
    )
    assert (
        final_metadata.filter(pl.col("time_series_id") == 2)["time_series_name"].item()
        == "ID 2 - Updated"
    )
    assert (
        final_metadata.filter(pl.col("time_series_id") == 3)["time_series_name"].item()
        == "ID 3 - New"
    )


def _metadata_table(
    ids: list[int], name: str = "ID", **extra: object
) -> pt.DataFrame[TimeSeriesMetadata]:
    """A valid metadata table covering ``ids``, plus any ``extra`` columns applied to every row."""
    rows = [
        {
            "time_series_id": i,
            "time_series_name": f"{name} {i}",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": i,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
            **extra,
        }
        for i in ids
    ]
    return pt.DataFrame(rows).set_model(TimeSeriesMetadata).cast().validate()


def test_upsert_metadata_adds_a_new_id_when_the_stored_metadata_table_is_thinner(tmp_path: Path):
    """The diff is derived by slicing the concatenated frame, so it must split back into exactly
    the snapshot's rows and the stored metadata table's rows. Getting that boundary wrong loses a
    whole time series silently: it never enters the metadata table, the stats claim nothing was new,
    and `select_new_rows` never re-offers the file, so it never arrives at all."""
    metadata_path = tmp_path / "metadata.parquet"
    _metadata_table([1]).write_parquet(metadata_path)

    stats = upsert_metadata(new_metadata=_metadata_table([1, 2]), metadata_path=str(metadata_path))

    assert stats["metadata_n_new_TimeSeriesIDs"] == 1
    assert stats["metadata_n_updated_TimeSeriesIDs"] == 0
    assert set(pl.read_parquet(metadata_path)["time_series_id"]) == {1, 2}


def test_upsert_metadata_merges_a_snapshot_missing_the_optional_columns(tmp_path: Path):
    """`TimeSeriesMetadata` has four `allow_missing` fields, so a snapshot can be narrower than
    the stored metadata table and still validate. Merging the two must not raise: a field the
    snapshot no longer carries is *cleared* for the series the snapshot covers, while a series the
    snapshot omits keeps every value it already had."""
    metadata_path = tmp_path / "metadata.parquet"
    _metadata_table([1, 2], information="note").write_parquet(metadata_path)

    # This run's snapshot covers id 2 only, and carries no `information` column at all.
    snapshot = _metadata_table([2], name="Renamed")
    assert "information" not in snapshot.columns
    upsert_metadata(new_metadata=snapshot, metadata_path=str(metadata_path))

    final = pl.read_parquet(metadata_path)
    assert final.filter(pl.col("time_series_id") == 2)["information"].item() is None
    assert final.filter(pl.col("time_series_id") == 1)["information"].item() == "note"
    assert final.filter(pl.col("time_series_id") == 2)["time_series_name"].item() == "Renamed 2"
    TimeSeriesMetadata.validate(final)


def test_upsert_metadata_ignores_the_stored_column_order(tmp_path: Path):
    """`hash_rows` is column-order sensitive, so a stored metadata table whose columns happen to sit
    in a different order must not be reported as wholly changed and rewritten every run."""
    metadata_path = tmp_path / "metadata.parquet"
    metadata_table = _metadata_table([1, 2])
    metadata_table.select(sorted(metadata_table.columns)).write_parquet(metadata_path)
    mtime_before = metadata_path.stat().st_mtime_ns

    stats = upsert_metadata(new_metadata=metadata_table, metadata_path=str(metadata_path))

    assert stats["metadata_n_new_TimeSeriesIDs"] == 0
    assert stats["metadata_n_updated_TimeSeriesIDs"] == 0
    assert metadata_path.stat().st_mtime_ns == mtime_before  # not rewritten at all


_EXAMPLE_OBJECT_KEY = (
    "timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json"
)
"""One real NGED object key, whose embedded epoch-millisecond window and id the parser extracts."""


@pytest.mark.parametrize(
    ("object_key", "expected_time_series_id"),
    [
        pytest.param(_EXAMPLE_OBJECT_KEY, 23, id="two_digit_id"),
        pytest.param(
            "timeseries/1774512000000_1774533600000/"
            "TimeSeries_237_20260326T080000Z_20260326T140000Z.json",
            237,
            id="three_digit_id",
        ),
        pytest.param(
            "timeseries/1774512000000_1774533600000/"
            "TimeSeries_2372_20260326T080000Z_20260326T140000Z.json",
            2372,
            id="four_digit_id",
        ),
    ],
)
def test_parse_file_listing_valid(object_key: str, expected_time_series_id: int):
    """The id capture group must not truncate above 99 (regression for the unanchored regex)."""
    raw_file_listing: list[_RawFileListItem] = [
        {
            "path": object_key,
            "filesize_bytes": 1024,
        }
    ]

    result = _process_file_listing(raw_file_listing)

    assert result.height == 1
    assert result["time_series_id"][0] == expected_time_series_id
    assert result["path"][0] == object_key
    assert result["filesize_bytes"][0] == 1024
    assert result["start_time"][0] == datetime(2026, 3, 26, 8, 0, 0, tzinfo=UTC)
    assert result["end_time"][0] == datetime(2026, 3, 26, 14, 0, 0, tzinfo=UTC)


def test_select_new_rows_file_listing(tmp_path: Path):
    """Regression: trailing comma made filtered_df a tuple, causing superfluous column_0 error.

    Also covers the lookback margin: a file sitting exactly at a series' on-disk `last_time`
    falls inside the margin and is kept, because the margin extends the cutoff to
    `_LATE_FILE_LOOKBACK` before `last_time`, while a file whose `end_time` falls further before
    the watermark than `_LATE_FILE_LOOKBACK` is still dropped — the margin bounds re-download
    cost rather than removing the cutoff outright.
    """
    delta_path = tmp_path / "power.delta"

    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([datetime(2026, 1, 1, 12, 0, tzinfo=UTC)]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    raw = pl.DataFrame(
        {
            "path": ["old.json", "new_ts1.json", "new_ts2.json", "too_old.json"],
            "filesize_bytes": pl.Series([1000, 1000, 1000, 1000], dtype=pl.Int64),
            "time_series_id": pl.Series([1, 1, 2, 1], dtype=pl.Int32),
            "start_time": pl.Series(
                [
                    datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2025, 12, 27, 0, 0, tzinfo=UTC),
                ]
            ).cast(UTC_DATETIME_DTYPE),
            "end_time": pl.Series(
                [
                    datetime(
                        2026, 1, 1, 12, 0, tzinfo=UTC
                    ),  # equals last_time for ts_id=1, within lookback margin → included
                    datetime(2026, 1, 1, 18, 0, tzinfo=UTC),  # > last_time for ts_id=1 → included
                    datetime(2026, 1, 1, 6, 0, tzinfo=UTC),  # ts_id=2 not in delta → included
                    datetime(2025, 12, 27, 0, 0, tzinfo=UTC),  # 5 days before last_time → dropped
                ]
            ).cast(UTC_DATETIME_DTYPE),
        }
    )
    file_listing = pt.DataFrame(raw).set_model(_ProcessedFileListing).validate()

    result = select_new_rows(file_listing, str(delta_path))

    assert set(result["path"].to_list()) == {"old.json", "new_ts1.json", "new_ts2.json"}
    _ProcessedFileListing.validate(result)  # schema must survive filtering


def test_select_new_rows_power_time_series_keeps_only_genuinely_missing_readings(tmp_path: Path):
    """A reading missing from a series' own history is kept even when a later reading for that
    series is already on disk, and existence is checked per series, not across the whole table
    — the anti-join must not match a candidate row against another series' on-disk reading
    that happens to share the same `time`."""
    delta_path = tmp_path / "power.delta"
    T = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 2], dtype=pl.Int32),
            "time": pl.Series([T, T + timedelta(hours=1), T + timedelta(minutes=30)]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([1.0, 1.0, 1.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    input_power = PowerTimeSeries.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1, 1, 1, 2], dtype=pl.Int32),
                "time": pl.Series(
                    [
                        T,  # already on disk for series 1 → dropped
                        T + timedelta(minutes=30),  # gap for series 1 (on disk only for 2) → kept
                        T + timedelta(minutes=90),  # past series 1's watermark → kept
                        T + timedelta(minutes=30),  # already on disk for series 2 → dropped
                    ]
                ).cast(UTC_DATETIME_DTYPE),
                "power": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            }
        )
    )

    result = select_new_rows(input_power, str(delta_path))

    assert result.select("time_series_id", "time").rows() == [
        (1, T + timedelta(minutes=30)),
        (1, T + timedelta(minutes=90)),
    ]
    PowerTimeSeries.validate(result)  # schema must survive filtering


def test_time_series_coverage(tmp_path: Path):
    """Returns the earliest and latest ``time`` per ``time_series_id`` from the Delta table."""
    delta_path = tmp_path / "power.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 2], dtype=pl.Int32),
            "time": pl.Series(
                [
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
                    datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
                ]
            ).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    coverage = time_series_coverage(str(delta_path)).sort("time_series_id")

    assert coverage["time_series_id"].to_list() == [1, 2]
    assert coverage["first_time"].to_list() == [
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
    ]
    assert coverage["last_time"].to_list() == [
        datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
        datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
    ]


def test_time_series_coverage_absent_table(tmp_path: Path):
    """A missing Delta table yields an empty but correctly-typed frame, not an error."""
    coverage = time_series_coverage(str(tmp_path / "does_not_exist.delta"))
    assert coverage.is_empty()
    assert coverage.columns == ["time_series_id", "first_time", "last_time"]


def test_parse_file_listing_invalid():
    # Invalid path format
    raw_file_listing: list[_RawFileListItem] = [
        {
            "path": "invalid/path/format.json",
            "filesize_bytes": 1024,
        }
    ]

    # The function uses `_TimeSeriesJsonFileListing.validate(paths_df)`
    # If the regex fails, the columns will be null, and validation should fail.
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _process_file_listing(raw_file_listing)


def test_remove_small_files_from_listing_keeps_one_reading_file():
    """A one-reading, WKT-less file is 556 bytes (measured on real NGED S3 downloads) and must
    survive the default filter."""
    file_listing = _file_listing([556])

    result = remove_small_files_from_listing(file_listing)

    assert result.height == 1
    _ProcessedFileListing.validate(result)
    # An eager `filter` returns a plain frame, so the model is only still attached if the function
    # re-attaches it.
    assert getattr(result, "model", None) is _ProcessedFileListing


def test_remove_small_files_from_listing_drops_genuinely_empty_file():
    """A genuinely empty, WKT-less file is 430-488 bytes (measured on real NGED S3 downloads) and
    must still be dropped by the default filter."""
    file_listing = _file_listing([430, 488])

    result = remove_small_files_from_listing(file_listing)

    assert result.height == 0


def test_remove_small_files_from_listing_logs_dropped_count(
    caplog: pytest.LogCaptureFixture,
):
    """The number of files retained/dropped must be logged at INFO, or the loss stays invisible."""
    file_listing = _file_listing([430, 556])

    with caplog.at_level(logging.INFO, logger="nged_data.storage"):
        remove_small_files_from_listing(file_listing)

    assert any(
        "1 out of n_files_before_filter=2" in r.message and r.levelno == logging.INFO
        for r in caplog.records
    )


def test_scan_cleaned_power_drops_flagged_rows_and_the_drop_reason_column(tmp_path: Path):
    delta_path = tmp_path / "cleaned.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 2], dtype=pl.Int32),
            "time": pl.Series(
                [
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                ]
            ).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 0.0, 3.0], dtype=pl.Float32),
            "drop_reason": pl.Series([None, "substation_zero", None], dtype=pl.String),
        }
    ).write_delta(delta_path)

    result = scan_cleaned_power(str(delta_path)).collect().sort("time_series_id")

    assert result.columns == ["time_series_id", "time", "power"]
    assert result["time_series_id"].to_list() == [1, 2]
    assert result["power"].to_list() == [1.0, 3.0]


def test_coverage_from_power_matches_time_series_coverage(tmp_path: Path):
    delta_path = tmp_path / "power.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1], dtype=pl.Int32),
            "time": pl.Series(
                [datetime(2026, 1, 1, 12, 0, tzinfo=UTC), datetime(2026, 1, 1, 12, 30, tzinfo=UTC)]
            ).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)
    power = pt.LazyFrame.from_existing(pl.scan_delta(str(delta_path))).set_model(PowerTimeSeries)

    assert coverage_from_power(power).equals(time_series_coverage(str(delta_path)))


# --- download_metadata_of_series_without_new_files -----------------------------------------------


def _small_nged_json(*, time_series_id: int, note: str | None, area: object = None) -> bytes:
    """A data-less NGED file of the size NGED publishes for a series that has stopped reporting.

    `area` defaults to the all-null struct that real files carry for a non-Primary series.
    """
    null_area = {
        "WKT": None,
        "SRID": None,
        "IsValid": False,
        "CenterLat": None,
        "CenterLon": None,
        "GeometryType": None,
    }
    return json.dumps(
        {
            "Area": null_area if area is None else area,
            "Units": "MW",
            "Latitude": 52.9,
            "Longitude": -0.01,
            "Information": note,
            "LicenceArea": "EMids",
            "TimeSeriesID": time_series_id,
            "SubstationType": "HV Customer",
            "TimeSeriesName": f"Test Generation {time_series_id}",
            "TimeSeriesType": "Other (Generation)",
            "SubstationNumber": 900_000 + time_series_id,
            "data": None,
        }
    ).encode()


# The fake's `.bytes()` method (named to match obstore's API) shadows the `bytes` builtin inside
# its own class scope, so its annotations use this module-level alias.
_JsonBytes = bytes


class _FakeGetResult:
    def __init__(self, data: _JsonBytes) -> None:
        self._data = data

    def bytes(self) -> _JsonBytes:
        return self._data


class _FakeStore:
    """Serves fixed file contents through the one `obstore` method the function under test calls."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def get(self, path: str) -> _FakeGetResult:
        return _FakeGetResult(self._files[path])


def _fake_store(files: dict[str, bytes]) -> obstore.store.S3Store:
    """Type the fake as the store the function under test takes: it only calls `.get()`."""
    return cast(obstore.store.S3Store, _FakeStore(files))


def _as_listing(listing: pl.DataFrame) -> pt.DataFrame[_ProcessedFileListing]:
    """Re-attach the Patito model that an eager Polars method drops."""
    return pt.DataFrame(listing).set_model(_ProcessedFileListing)


def _listing_of(files: dict[str, bytes]) -> pt.DataFrame[_ProcessedFileListing]:
    return _process_file_listing(
        [_RawFileListItem(path=path, filesize_bytes=len(data)) for path, data in files.items()]
    )


def _key(time_series_id: int, end_ms: int) -> str:
    return (
        f"timeseries/{end_ms - 21_600_000}_{end_ms}/TimeSeries_{time_series_id}_20260326T080000Z_"
        "20260326T140000Z.json"
    )


def test_download_metadata_of_series_without_new_files_reads_the_newest_small_file_of_each_series():
    files = {
        _key(33, 1_774_533_600_000): _small_nged_json(time_series_id=33, note="older note"),
        _key(33, 1_774_555_200_000): _small_nged_json(time_series_id=33, note="newer note"),
        _key(32, 1_774_533_600_000): _small_nged_json(time_series_id=32, note=None),
    }
    all_files = _listing_of(files)

    result = download_metadata_of_series_without_new_files(
        store=_fake_store(files),
        all_files=all_files,
        downloaded_files=_as_listing(all_files.clear()),
    )

    assert result.metadata is not None
    assert dict(result.metadata.select("time_series_id", "information").rows()) == {
        32: None,
        33: "newer note",
    }
    assert (result.n_downloaded, result.n_failed) == (2, 0)


def test_download_metadata_of_series_without_new_files_skips_series_with_a_downloaded_file():
    files = {
        _key(33, 1_774_533_600_000): _small_nged_json(time_series_id=33, note="note"),
        _key(34, 1_774_533_600_000): _small_nged_json(time_series_id=34, note="note"),
    }
    all_files = _listing_of(files)

    result = download_metadata_of_series_without_new_files(
        store=_fake_store(files),
        all_files=all_files,
        downloaded_files=_as_listing(all_files.filter(pl.col("time_series_id") == 34)),
    )

    assert result.metadata is not None
    assert result.metadata["time_series_id"].to_list() == [33]


def test_download_metadata_of_series_without_new_files_skips_a_series_whose_newest_file_is_large():
    large_note = "x" * SMALL_FILE_SIZE_THRESHOLD_BYTES
    files = {_key(33, 1_774_533_600_000): _small_nged_json(time_series_id=33, note=large_note)}
    all_files = _listing_of(files)

    result = download_metadata_of_series_without_new_files(
        store=_fake_store(files),
        all_files=all_files,
        downloaded_files=_as_listing(all_files.clear()),
    )

    assert result.metadata is None
    assert result.n_downloaded == 0


def test_download_metadata_of_series_without_new_files_counts_a_malformed_file_and_keeps_the_rest():
    files = {
        _key(33, 1_774_533_600_000): _small_nged_json(time_series_id=33, note="note"),
        # `Area: null` is not a struct, so `_extract_time_series_metadata` raises on it.
        _key(34, 1_774_533_600_000): _small_nged_json(time_series_id=34, note="note").replace(
            b'"Area": {', b'"Area": null, "Unused": {'
        ),
    }
    all_files = _listing_of(files)

    result = download_metadata_of_series_without_new_files(
        store=_fake_store(files),
        all_files=all_files,
        downloaded_files=_as_listing(all_files.clear()),
    )

    assert result.metadata is not None
    assert result.metadata["time_series_id"].to_list() == [33]
    assert (result.n_downloaded, result.n_failed) == (2, 1)


def test_download_metadata_of_series_without_new_files_with_nothing_to_read():
    files = {_key(33, 1_774_533_600_000): _small_nged_json(time_series_id=33, note="note")}
    all_files = _listing_of(files)

    result = download_metadata_of_series_without_new_files(
        store=_fake_store(files), all_files=all_files, downloaded_files=all_files
    )

    assert result == (None, 0, 0)
