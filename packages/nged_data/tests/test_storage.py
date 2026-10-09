import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import obstore
import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data import storage
from nged_data.storage import (
    DownloadedFilesError,
    NgedFileParseError,
    _DownloadedFiles,
    _process_file_listing,
    _ProcessedFileListing,
    _RawFileListItem,
    coverage_from_power,
    download_and_parse_files,
    read_downloaded_files,
    scan_cleaned_power,
    select_files_not_yet_downloaded,
    select_new_rows,
    time_series_coverage,
    upsert_metadata,
    write_downloaded_files,
)

_LAST_MODIFIED = datetime(2026, 3, 26, 14, 5, 0, 123456, tzinfo=UTC)
"""A `LastModified` with microseconds, to prove the downloaded-files list keeps them."""


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
    and the downloaded-files list records the file, so the ingest never downloads it again."""
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
            "last_modified": _LAST_MODIFIED,
        }
    ]

    result = _process_file_listing(raw_file_listing)

    assert result.height == 1
    assert result["time_series_id"][0] == expected_time_series_id
    assert result["path"][0] == object_key
    assert result["filesize_bytes"][0] == 1024
    assert result["last_modified"][0] == _LAST_MODIFIED
    assert result["start_time"][0] == datetime(2026, 3, 26, 8, 0, 0, tzinfo=UTC)
    assert result["end_time"][0] == datetime(2026, 3, 26, 14, 0, 0, tzinfo=UTC)


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
            "last_modified": _LAST_MODIFIED,
        }
    ]

    # The function uses `_TimeSeriesJsonFileListing.validate(paths_df)`
    # If the regex fails, the columns will be null, and validation should fail.
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _process_file_listing(raw_file_listing)


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


def _file_without_readings(*, time_series_id: int, data_field: str) -> bytes:
    """A real NGED file's metadata fields, with the given `data` field instead of readings."""
    fixture = Path(__file__).parent / "data" / "TimeSeries_10.json"
    file_contents = json.loads(fixture.read_text())
    file_contents.update(TimeSeriesID=time_series_id, data=json.loads(data_field))
    return json.dumps(file_contents).encode()


def _key(time_series_id: int, end_ms: int) -> str:
    return (
        f"timeseries/{end_ms - 21_600_000}_{end_ms}/TimeSeries_{time_series_id}_20260326T080000Z_"
        "20260326T140000Z.json"
    )


def _listing_of(
    paths: list[str], last_modified: datetime = _LAST_MODIFIED
) -> pt.DataFrame[_ProcessedFileListing]:
    return _process_file_listing(
        [
            _RawFileListItem(path=path, filesize_bytes=10_000, last_modified=last_modified)
            for path in paths
        ]
    )


def _downloaded(*rows: tuple[str, datetime]) -> pt.DataFrame[_DownloadedFiles]:
    frame = pl.DataFrame(
        {
            "path": [path for path, _ in rows],
            "last_modified": pl.Series([last_modified for _, last_modified in rows]).cast(
                UTC_DATETIME_DTYPE
            ),
        },
        schema={"path": pl.String, "last_modified": UTC_DATETIME_DTYPE},
    )
    return pt.DataFrame(frame).set_model(_DownloadedFiles).validate()


def test_select_files_not_yet_downloaded_selects_new_and_rewritten_files():
    known_file = _key(1, 1_774_533_600_000)
    rewritten_file = _key(2, 1_774_555_200_000)
    new_file = _key(3, 1_774_520_000_000)
    listing = _listing_of([known_file, rewritten_file, new_file])
    downloaded = _downloaded(
        (known_file, _LAST_MODIFIED), (rewritten_file, _LAST_MODIFIED - timedelta(hours=1))
    )

    result = select_files_not_yet_downloaded(file_listing=listing, downloaded_files=downloaded)

    assert set(result["path"]) == {new_file, rewritten_file}
    _ProcessedFileListing.validate(result)


def test_select_files_not_yet_downloaded_selects_everything_for_an_empty_list():
    paths = [_key(1, 1_774_533_600_000), _key(2, 1_774_555_200_000)]

    result = select_files_not_yet_downloaded(
        file_listing=_listing_of(paths), downloaded_files=_downloaded()
    )

    assert set(result["path"]) == set(paths)


def _power_table_and_metadata(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal `power_time_series` Delta table and a metadata parquet file."""
    power_path = tmp_path / "power.delta"
    pl.DataFrame({"time_series_id": [1]}).write_delta(power_path)
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame({"time_series_id": [1]}).write_parquet(metadata_path)
    return power_path, metadata_path


def _read(
    downloaded_files_path: Path, power_path: Path, metadata_path: Path
) -> pt.DataFrame[_DownloadedFiles]:
    return read_downloaded_files(
        downloaded_files_path=str(downloaded_files_path),
        power_table_path=str(power_path),
        metadata_path=str(metadata_path),
    )


def test_downloaded_files_survive_a_write_and_a_read_with_microseconds(tmp_path: Path):
    power_path, metadata_path = _power_table_and_metadata(tmp_path)
    downloaded_files_path = tmp_path / "downloaded_files.parquet"
    listing = _listing_of([_key(1, 1_774_533_600_000)])

    write_downloaded_files(downloaded_files_path=str(downloaded_files_path), listing=listing)
    result = _read(downloaded_files_path, power_path, metadata_path)

    assert result.rows() == [(listing["path"][0], _LAST_MODIFIED)]
    assert select_files_not_yet_downloaded(listing, result).is_empty()


def test_write_downloaded_files_keeps_the_previous_list_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    power_path, metadata_path = _power_table_and_metadata(tmp_path)
    downloaded_files_path = tmp_path / "downloaded_files.parquet"
    first = _listing_of([_key(1, 1_774_533_600_000)])
    write_downloaded_files(downloaded_files_path=str(downloaded_files_path), listing=first)

    def fail(self: Path, target: str) -> Path:
        raise OSError("killed before the rename")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="killed"):
        write_downloaded_files(
            downloaded_files_path=str(downloaded_files_path),
            listing=_listing_of([_key(2, 1_774_555_200_000)]),
        )
    monkeypatch.undo()

    assert _read(downloaded_files_path, power_path, metadata_path)["path"].to_list() == (
        first["path"].to_list()
    )


def test_read_downloaded_files_is_empty_unless_the_list_the_table_and_the_metadata_all_exist(
    tmp_path: Path,
):
    power_path, metadata_path = _power_table_and_metadata(tmp_path)
    downloaded_files_path = tmp_path / "downloaded_files.parquet"
    write_downloaded_files(
        downloaded_files_path=str(downloaded_files_path),
        listing=_listing_of([_key(1, 1_774_533_600_000)]),
    )
    assert _read(downloaded_files_path, power_path, metadata_path).height == 1

    missing_list = _read(tmp_path / "absent.parquet", power_path, metadata_path)
    missing_table = _read(downloaded_files_path, tmp_path / "absent.delta", metadata_path)
    missing_metadata = _read(downloaded_files_path, power_path, tmp_path / "absent.pq")

    for result in (missing_list, missing_table, missing_metadata):
        assert result.is_empty()
        assert result.columns == ["path", "last_modified"]


def test_read_downloaded_files_raises_a_named_error_for_a_corrupt_file(tmp_path: Path):
    power_path, metadata_path = _power_table_and_metadata(tmp_path)
    downloaded_files_path = tmp_path / "downloaded_files.parquet"
    downloaded_files_path.write_bytes(b"not a parquet file")

    with pytest.raises(DownloadedFilesError, match=r"downloaded_files\.parquet"):
        _read(downloaded_files_path, power_path, metadata_path)


def test_read_downloaded_files_raises_a_named_error_for_a_file_off_contract(tmp_path: Path):
    power_path, metadata_path = _power_table_and_metadata(tmp_path)
    downloaded_files_path = tmp_path / "downloaded_files.parquet"
    pl.DataFrame({"path": ["a"], "unexpected": [1]}).write_parquet(downloaded_files_path)

    with pytest.raises(DownloadedFilesError, match=r"downloaded_files\.parquet"):
        _read(downloaded_files_path, power_path, metadata_path)


def test_read_downloaded_files_does_not_wrap_a_transient_error_from_an_existence_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail(uri: str, storage_options: object = None) -> bool:
        raise OSError("transient")

    monkeypatch.setattr(storage, "object_exists", fail)

    with pytest.raises(OSError, match="transient") as excinfo:
        read_downloaded_files(
            downloaded_files_path=str(tmp_path / "a.parquet"),
            power_table_path=str(tmp_path / "p.delta"),
            metadata_path=str(tmp_path / "m.parquet"),
        )
    assert not isinstance(excinfo.value, DownloadedFilesError)


def _file_with_readings(
    *, time_series_id: int, information: str | None, values: list[float]
) -> bytes:
    """A real NGED file's metadata fields, with half-hourly readings ending at 12:30 UTC."""
    fixture = Path(__file__).parent / "data" / "TimeSeries_11.json"
    file_contents = json.loads(fixture.read_text())
    start = datetime(2026, 3, 26, 12, 0, tzinfo=UTC)
    readings = [
        {
            "value": value,
            "startTime": (start + timedelta(minutes=30 * i)).strftime("%Y-%m-%d %H:%M:%S%z"),
            "endTime": (start + timedelta(minutes=30 * (i + 1))).strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        for i, value in enumerate(values)
    ]
    file_contents.update(TimeSeriesID=time_series_id, Information=information, data=readings)
    return json.dumps(file_contents).encode()


class _FakeAsyncResult:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def bytes_async(self) -> bytes:
        return self._data


class _FakeAsyncStore:
    """Serves bytes through `get_async` with a per-path delay, tracking requests in flight."""

    def __init__(self, files: dict[str, bytes], delays: dict[str, float] | None = None) -> None:
        self.files = files
        self.delays = delays or {}
        self.in_flight = 0
        self.peak_in_flight = 0
        self.failing_paths: set[str] = set()
        self.requested_paths: list[str] = []

    async def get_async(self, path: str) -> _FakeAsyncResult:
        self.requested_paths.append(path)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(path, 0.001))
            if path in self.failing_paths:
                raise OSError(f"request failed: {path}")
            return _FakeAsyncResult(self.files[path])
        finally:
            self.in_flight -= 1


def _as_store(store: _FakeAsyncStore) -> obstore.store.S3Store:
    return cast(obstore.store.S3Store, store)


@pytest.mark.parametrize("data_field", ["null", "[]"])
def test_download_and_parse_files_skips_a_file_without_readings_and_keeps_the_others(
    data_field: str,
):
    real_file = (Path(__file__).parent / "data" / "TimeSeries_11.json").read_bytes()
    real_path = "timeseries/1774512000000_1774533600000/TimeSeries_11_a_b.json"
    empty_path = "timeseries/1774512000000_1774533600000/TimeSeries_10_a_b.json"
    store = obstore.store.MemoryStore()
    obstore.put(store, real_path, real_file)
    obstore.put(store, empty_path, _file_without_readings(time_series_id=10, data_field=data_field))

    result = download_and_parse_files(
        store=cast(obstore.store.S3Store, store),
        paths_df=_listing_of([empty_path, real_path]),
    )

    assert set(result.power_time_series["time_series_id"]) == {11}
    assert set(result.metadata["time_series_id"]) == {10, 11}


def test_download_and_parse_files_returns_the_metadata_and_no_rows_when_every_file_is_data_less():
    path = "timeseries/1774512000000_1774533600000/TimeSeries_10_a_b.json"
    store = obstore.store.MemoryStore()
    obstore.put(store, path, _file_without_readings(time_series_id=10, data_field="null"))

    result = download_and_parse_files(
        store=cast(obstore.store.S3Store, store), paths_df=_listing_of([path])
    )

    assert result.metadata["time_series_id"].to_list() == [10]
    assert result.power_time_series.is_empty()
    assert result.power_time_series.columns == PowerTimeSeries.columns


def test_download_and_parse_files_names_the_path_of_a_malformed_file():
    good_path = _key(11, 1_774_533_600_000)
    bad_path = _key(12, 1_774_555_200_000)
    store = _FakeAsyncStore(
        {
            good_path: _file_with_readings(time_series_id=11, information=None, values=[1.0]),
            bad_path: b"{not json",
        }
    )

    with pytest.raises(NgedFileParseError, match=bad_path):
        download_and_parse_files(
            store=_as_store(store), paths_df=_listing_of([good_path, bad_path])
        )


def test_download_and_parse_files_keeps_the_later_window_whatever_order_requests_finish_in():
    """Of two overlapping files, the later window's reading and note win, even if it lands first."""
    earlier_path = _key(11, 1_774_533_600_000)
    later_path = _key(11, 1_774_555_200_000)
    store = _FakeAsyncStore(
        {
            earlier_path: _file_with_readings(
                time_series_id=11, information="earlier note", values=[1.0, 2.0]
            ),
            later_path: _file_with_readings(
                time_series_id=11, information="later note", values=[10.0, 20.0]
            ),
        },
        # The earlier window's file finishes last, and the listing is passed latest-first.
        delays={earlier_path: 0.05, later_path: 0.001},
    )
    latest_first = pt.DataFrame(_listing_of([earlier_path, later_path]).reverse()).set_model(
        _ProcessedFileListing
    )

    result = download_and_parse_files(store=_as_store(store), paths_df=latest_first)

    assert result.power_time_series["power"].to_list() == [10.0, 20.0]
    assert result.metadata["information"].to_list() == ["later note"]


def test_download_and_parse_files_is_concurrent_and_capped_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    """A sequential loop has a peak of 1, a missing semaphore a peak of 10, and a semaphore made at
    module level raises `RuntimeError` on the second chunk's event loop."""
    monkeypatch.setattr(storage, "_DOWNLOAD_CHUNK_FILES", 3)
    monkeypatch.setattr(storage, "_MAX_REQUESTS_IN_FLIGHT", 2)
    paths = [_key(11, 1_774_533_600_000 + i * 21_600_000) for i in range(10)]
    store = _FakeAsyncStore(
        {
            path: _file_with_readings(time_series_id=11, information=None, values=[1.0])
            for path in paths
        },
        delays=dict.fromkeys(paths, 0.01),
    )

    result = download_and_parse_files(store=_as_store(store), paths_df=_listing_of(paths))

    assert sorted(store.requested_paths) == sorted(paths)
    assert result.metadata.height == 1
    assert 1 < store.peak_in_flight <= 2


def test_download_and_parse_files_raises_when_a_request_fails():
    paths = [_key(11, 1_774_533_600_000 + i * 21_600_000) for i in range(5)]
    store = _FakeAsyncStore(
        {
            path: _file_with_readings(time_series_id=11, information=None, values=[1.0])
            for path in paths
        }
    )
    store.failing_paths = {paths[2]}

    with pytest.raises(OSError, match="request failed"):
        download_and_parse_files(store=_as_store(store), paths_df=_listing_of(paths))


def test_read_downloaded_files_does_not_wrap_a_polars_error_reading_a_remote_file(
    monkeypatch: pytest.MonkeyPatch,
):
    """An object-store write is atomic, so a Polars error reading a remote list is transient."""
    monkeypatch.setattr(storage, "object_exists", lambda uri, storage_options=None: True)
    monkeypatch.setattr(storage, "delta_table_exists", lambda uri, storage_options=None: True)

    def fail(*args: object, **kwargs: object) -> pl.DataFrame:
        raise pl.exceptions.ComputeError("connection reset")

    monkeypatch.setattr(pl, "read_parquet", fail)

    with pytest.raises(pl.exceptions.ComputeError):
        read_downloaded_files(
            downloaded_files_path="s3://bucket/downloaded_files.parquet",
            power_table_path="s3://bucket/power.delta",
            metadata_path="s3://bucket/metadata.parquet",
        )


def test_read_downloaded_files_passes_the_storage_options_to_the_read(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(storage, "object_exists", lambda uri, storage_options=None: True)
    monkeypatch.setattr(storage, "delta_table_exists", lambda uri, storage_options=None: True)
    seen: list[object] = []

    def read(path: str, storage_options: object = None) -> pl.DataFrame:
        seen.append(storage_options)
        return pl.DataFrame(schema={"path": pl.String, "last_modified": UTC_DATETIME_DTYPE})

    monkeypatch.setattr(pl, "read_parquet", read)

    read_downloaded_files(
        downloaded_files_path="s3://bucket/downloaded_files.parquet",
        power_table_path="s3://bucket/power.delta",
        metadata_path="s3://bucket/metadata.parquet",
        storage_options={"aws_region": "eu-west-2"},
    )

    assert seen == [{"aws_region": "eu-west-2"}]
