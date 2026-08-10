from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data.read_nged_json import _extract_power_time_series, _extract_time_series_metadata


@pytest.mark.parametrize(
    ("filename", "expected_time_series_id", "expected_name", "expected_units", "expected_h3"),
    [
        ("TimeSeries_10.json", 10, "SPILSBY 33 11kV S STN", "MW", 599423199024775167),
        ("TimeSeries_11.json", 11, "INGOLDMELLS 33 11kV S STN", "MVA", 599422966022799359),
    ],
)
def test_nged_json_to_metadata_df_and_time_series_df(
    filename: str,
    expected_time_series_id: int,
    expected_name: str,
    expected_units: str,
    expected_h3: int,
):
    file_path = Path(__file__).parent / "data" / filename

    with file_path.open("rb") as f:
        json_bytes = f.read()

    df = pl.read_json(json_bytes)
    metadata_df = _extract_time_series_metadata(df)
    time_series_id = metadata_df["time_series_id"].item()
    extracted = _extract_power_time_series(df, time_series_id)
    time_series_df = extracted.dataframe

    TimeSeriesMetadata.validate(metadata_df)
    PowerTimeSeries.validate(time_series_df)
    assert extracted.n_dropped == 0

    assert not metadata_df.is_empty()
    assert not time_series_df.is_empty()
    assert metadata_df["time_series_id"].item() == expected_time_series_id
    assert metadata_df["time_series_name"].item() == expected_name
    assert metadata_df["units"].item() == expected_units
    assert metadata_df["time_series_type"].item() == "Disaggregated Demand"
    assert metadata_df["licence_area"].item() == "EMids"
    assert metadata_df["substation_type"].item() == "Primary"
    assert "POLYGON" in metadata_df["area_wkt"].item()
    assert metadata_df["h3_res_5"].item() == expected_h3


def test_nged_json_to_metadata_df_and_time_series_df_invalid_json():
    # JSON missing required fields
    invalid_json = b'{"some": "data"}'
    df = pl.read_json(invalid_json)
    with pytest.raises((pt.exceptions.DataFrameValidationError, pl.exceptions.ColumnNotFoundError)):
        _extract_time_series_metadata(df)


def test_extract_power_time_series_drops_malformed_time_rows():
    """A row with an implausible or misaligned `time` is dropped and counted, not raised on."""
    raw_json = b"""
    {
        "data": [
            {"endTime": "1840-06-01T00:30:00Z", "value": 1.0},
            {"endTime": "2026-01-01T00:15:00Z", "value": 2.0},
            {"endTime": "2026-01-01T00:30:00Z", "value": 3.0}
        ]
    }
    """
    df = pl.read_json(raw_json)

    extracted = _extract_power_time_series(df, time_series_id=42)

    assert extracted.n_dropped == 2
    assert extracted.dataframe.height == 1
    assert extracted.dataframe["power"].to_list() == [3.0]
    PowerTimeSeries.validate(extracted.dataframe)


def test_nged_json_to_metadata_df_and_time_series_df_invalid_data_types():
    """A `value` that is not a number is a malformed input, so it is rejected, not degraded."""
    invalid_json = (
        b'{"time_series_id": 1, "Area": {"latitude": 50, "longitude": 0},'
        b' "data": [{"endTime": "2026-01-01T00:00:00Z", "value": "not_a_float"}]}'
    )
    df = pl.read_json(invalid_json)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion from `str` to `f32`"):
        _extract_power_time_series(df, time_series_id=1)
