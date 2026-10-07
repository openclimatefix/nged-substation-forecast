from datetime import UTC, datetime

import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata
from nged_data.cleaning import flag_nged_power

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)


def _metadata(substation_types: dict[int, str]) -> pt.DataFrame[TimeSeriesMetadata]:
    return (
        pt.DataFrame(
            [
                {
                    "time_series_id": time_series_id,
                    "time_series_name": f"Series {time_series_id}",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": time_series_id,
                    "substation_type": substation_type,
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
                for time_series_id, substation_type in substation_types.items()
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )


def _power(rows: list[tuple[int, datetime, float]]) -> pt.LazyFrame[PowerTimeSeries]:
    frame = pl.DataFrame(
        {
            "time_series_id": [row[0] for row in rows],
            "time": [row[1] for row in rows],
            "power": [row[2] for row in rows],
        },
        schema_overrides={"time_series_id": pl.Int32, "power": pl.Float32},
    ).cast({"time": UTC_DATETIME_DTYPE})
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(PowerTimeSeries)


def _flag(
    rows: list[tuple[int, datetime, float]], substation_types: dict[int, str]
) -> pl.DataFrame:
    flagged = flag_nged_power(_power(rows), _metadata(substation_types)).collect()
    return pl.DataFrame._from_pydf(flagged._df)


@pytest.mark.parametrize("substation_type", ["Primary", "BSP", "GSP"])
def test_substation_zero_flags_every_substation_type(substation_type: str):
    result = _flag([(1, T0, 0.0)], {1: substation_type})
    assert result["drop_reason"].to_list() == ["substation_zero"]


def test_substation_non_zero_reading_is_not_flagged():
    result = _flag([(1, T0, 0.5), (1, T1, -0.5)], {1: "Primary"})
    assert result["drop_reason"].to_list() == [None, None]


def test_generator_zero_is_not_flagged():
    result = _flag([(1, T0, 0.0)], {1: "HV Customer"})
    assert result["drop_reason"].to_list() == [None]


def test_zero_from_series_missing_from_metadata_is_not_flagged():
    result = _flag([(1, T0, 0.0), (2, T0, 0.0)], {1: "Primary"})
    assert result["drop_reason"].to_list() == ["substation_zero", None]


def test_flag_nged_power_returns_exactly_the_input_rows():
    rows = [(1, T0, 0.0), (1, T1, 2.0), (2, T0, 0.0), (3, T0, 4.0)]
    result = _flag(rows, {1: "Primary", 2: "HV Customer"})
    assert result.columns == ["time_series_id", "time", "power", "drop_reason"]
    # `flag_nged_power` promises no row order, so compare sorted rows.
    assert sorted(result.select("time_series_id", "time", "power").rows()) == sorted(rows)
    CleanedPowerTimeSeries.validate(result.sort("time_series_id", "time"))
