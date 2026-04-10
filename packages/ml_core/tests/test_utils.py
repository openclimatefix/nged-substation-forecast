from datetime import date
import polars as pl

from ml_core.utils import _slice_temporal_data


def test_slice_temporal_data_dataframe():
    df = pl.DataFrame(
        {
            "end_time": [
                "2023-01-01T00:00:00",
                "2023-01-02T00:00:00",
                "2023-01-03T00:00:00",
            ],
            "value": [1, 2, 3],
        }
    ).with_columns(pl.col("end_time").str.to_datetime())

    start = date(2023, 1, 1)
    end = date(2023, 1, 2)

    result = _slice_temporal_data(df, start, end, "end_time")

    assert len(result) == 2
    assert result["value"].to_list() == [1, 2]


def test_slice_temporal_data_lazyframe():
    df = pl.LazyFrame(
        {
            "valid_time": [
                "2023-01-01T00:00:00",
                "2023-01-02T00:00:00",
                "2023-01-03T00:00:00",
            ],
            "value": [1, 2, 3],
        }
    ).with_columns(pl.col("valid_time").str.to_datetime())

    start = date(2023, 1, 2)
    end = date(2023, 1, 3)

    result = _slice_temporal_data(df, start, end, "valid_time")

    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert len(collected) == 2
    assert collected["value"].to_list() == [2, 3]


def test_slice_temporal_data_dict():
    df1 = pl.DataFrame(
        {
            "end_time": [
                "2023-01-01T00:00:00",
                "2023-01-02T00:00:00",
            ],
            "value": [1, 2],
        }
    ).with_columns(pl.col("end_time").str.to_datetime())

    df2 = pl.DataFrame(
        {
            "end_time": [
                "2023-01-01T00:00:00",
                "2023-01-02T00:00:00",
            ],
            "value": [3, 4],
        }
    ).with_columns(pl.col("end_time").str.to_datetime())

    data = {"flows": df1, "nwp": df2}

    start = date(2023, 1, 2)
    end = date(2023, 1, 2)

    result = _slice_temporal_data(data, start, end, "end_time")

    assert len(result["flows"]) == 1
    assert result["flows"]["value"].to_list() == [2]
    assert len(result["nwp"]) == 1
    assert result["nwp"]["value"].to_list() == [4]


def test_slice_temporal_data_no_time_col():
    df = pl.DataFrame(
        {
            "other_col": [
                "2023-01-01T00:00:00",
                "2023-01-02T00:00:00",
            ],
            "value": [1, 2],
        }
    )

    start = date(2023, 1, 2)
    end = date(2023, 1, 2)

    result = _slice_temporal_data(df, start, end, "end_time")

    # Should return unchanged
    assert len(result) == 2
    assert result["value"].to_list() == [1, 2]
