import os

from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df


def test_nged_json_to_metadata_df_and_time_series_df_file_10():
    data_dir = "packages/nged_data/tests/data"
    file_path = os.path.join(data_dir, "TimeSeries_10.json")

    with open(file_path, "rb") as f:
        json_bytes = f.read()

    metadata_df, time_series_df = nged_json_to_metadata_df_and_time_series_df(json_bytes)

    assert not metadata_df.is_empty()
    assert not time_series_df.is_empty()
    assert "time_series_id" in metadata_df.columns
    assert "time" in time_series_df.columns
    assert "power" in time_series_df.columns
    assert metadata_df["time_series_id"].item() == 10


def test_nged_json_to_metadata_df_and_time_series_df_file_11():
    data_dir = "packages/nged_data/tests/data"
    file_path = os.path.join(data_dir, "TimeSeries_11.json")

    with open(file_path, "rb") as f:
        json_bytes = f.read()

    metadata_df, time_series_df = nged_json_to_metadata_df_and_time_series_df(json_bytes)

    assert not metadata_df.is_empty()
    assert not time_series_df.is_empty()
    assert "time_series_id" in metadata_df.columns
    assert "time" in time_series_df.columns
    assert "power" in time_series_df.columns
    assert metadata_df["time_series_id"].item() == 11
