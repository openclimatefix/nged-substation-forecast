from pathlib import Path

import pytest
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df


@pytest.mark.parametrize(
    "filename, expected_time_series_id, expected_name, expected_units, expected_h3",
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

    with open(file_path, "rb") as f:
        json_bytes = f.read()

    metadata_df, time_series_df = nged_json_to_metadata_df_and_time_series_df(json_bytes)

    TimeSeriesMetadata.validate(metadata_df)
    PowerTimeSeries.validate(time_series_df)

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
