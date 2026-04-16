from pathlib import Path

import pytest
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df


@pytest.mark.parametrize(
    "filename, expected_time_series_id", [("TimeSeries_10.json", 10), ("TimeSeries_11.json", 11)]
)
def test_nged_json_to_metadata_df_and_time_series_df(filename: str, expected_time_series_id: int):
    data_dir = Path(
        "tests/data"
    )  # FIXME: Use a path that works correctly no matter if you run pytest in the root uv workspace directory or in packages/nged_data
    file_path = data_dir / filename

    with open(file_path, "rb") as f:
        json_bytes = f.read()

    metadata_df, time_series_df = nged_json_to_metadata_df_and_time_series_df(json_bytes)

    TimeSeriesMetadata.validate(metadata_df)
    PowerTimeSeries.validate(time_series_df)

    assert not metadata_df.is_empty()
    assert not time_series_df.is_empty()
    assert metadata_df["time_series_id"].item() == expected_time_series_id
