from datetime import UTC, datetime, timedelta

import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata
from nged_data import cleaning
from nged_data.cleaning import flag_nged_power

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)


def _half_hour(index: int) -> datetime:
    return T0 + timedelta(minutes=30 * index)


@pytest.fixture(autouse=True)
def no_initial_analogue_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the `initial_analogue_noise` rule off, because the series here are a few rows long.

    A test of that rule sets its own cut-off with `set_initial_analogue_noise_readings`.
    """
    monkeypatch.setattr(cleaning, "INITIAL_ANALOGUE_NOISE_READINGS", 0)


def set_initial_analogue_noise_readings(monkeypatch: pytest.MonkeyPatch, readings: int) -> None:
    monkeypatch.setattr(cleaning, "INITIAL_ANALOGUE_NOISE_READINGS", readings)


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
    # `flag_nged_power` promises no row order (Polars 2 runs a lazy join on the streaming engine),
    # so sort before the tests compare `drop_reason` lists.
    return pl.DataFrame._from_pydf(flagged._df).sort("time_series_id", "time")


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


def test_initial_analogue_noise_flags_the_first_readings_only(monkeypatch: pytest.MonkeyPatch):
    set_initial_analogue_noise_readings(monkeypatch, 2)
    rows = [(1, _half_hour(index), 1.0) for index in range(4)]
    result = _flag(rows, {1: "Primary"})
    assert result["drop_reason"].to_list() == [
        "initial_analogue_noise",
        "initial_analogue_noise",
        None,
        None,
    ]
    CleanedPowerTimeSeries.validate(result)


def test_initial_analogue_noise_counts_readings_not_calendar_time(
    monkeypatch: pytest.MonkeyPatch,
):
    set_initial_analogue_noise_readings(monkeypatch, 2)
    # A 1,000-day gap after the first reading does not count towards the cut-off.
    rows = [(1, _half_hour(0), 1.0), (1, _half_hour(48_000), 1.0), (1, _half_hour(48_001), 1.0)]
    result = _flag(rows, {1: "Primary"})
    assert result["drop_reason"].to_list() == [
        "initial_analogue_noise",
        "initial_analogue_noise",
        None,
    ]


def test_initial_analogue_noise_flags_a_short_series_in_full(monkeypatch: pytest.MonkeyPatch):
    set_initial_analogue_noise_readings(monkeypatch, 5)
    rows = [(1, _half_hour(index), 1.0) for index in range(3)]
    result = _flag(rows, {1: "Primary"})
    assert result["drop_reason"].to_list() == ["initial_analogue_noise"] * 3


def test_initial_analogue_noise_counts_each_series_separately(monkeypatch: pytest.MonkeyPatch):
    set_initial_analogue_noise_readings(monkeypatch, 1)
    # Series 2 starts a day after series 1, and still has its own first reading flagged.
    rows = [
        (1, _half_hour(0), 1.0),
        (1, _half_hour(1), 1.0),
        (2, _half_hour(48), 1.0),
        (2, _half_hour(49), 1.0),
    ]
    result = _flag(rows, {1: "Primary", 2: "HV Customer"})
    assert result["drop_reason"].to_list() == ["initial_analogue_noise", None] * 2


def test_initial_analogue_noise_counts_in_time_order_whatever_the_row_order(
    monkeypatch: pytest.MonkeyPatch,
):
    set_initial_analogue_noise_readings(monkeypatch, 2)
    rows = [(1, _half_hour(index), 1.0) for index in (3, 0, 2, 1)]
    result = _flag(rows, {1: "Primary"})
    assert result["drop_reason"].to_list() == [
        "initial_analogue_noise",
        "initial_analogue_noise",
        None,
        None,
    ]


def test_initial_analogue_noise_moves_its_cut_off_when_earlier_readings_are_back_filled(
    monkeypatch: pytest.MonkeyPatch,
):
    set_initial_analogue_noise_readings(monkeypatch, 1)
    rows = [(1, _half_hour(index), 1.0) for index in (10, 11)]
    back_filled = [(1, _half_hour(0), 1.0), *rows]
    assert _flag(rows, {1: "Primary"})["drop_reason"].to_list() == ["initial_analogue_noise", None]
    assert _flag(back_filled, {1: "Primary"})["drop_reason"].to_list() == [
        "initial_analogue_noise",
        None,
        None,
    ]


def test_initial_analogue_noise_wins_over_substation_zero(monkeypatch: pytest.MonkeyPatch):
    set_initial_analogue_noise_readings(monkeypatch, 1)
    result = _flag([(1, T0, 0.0), (1, T1, 0.0)], {1: "Primary"})
    assert result["drop_reason"].to_list() == ["initial_analogue_noise", "substation_zero"]
    CleanedPowerTimeSeries.validate(result)


def test_initial_analogue_noise_flags_a_series_missing_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    set_initial_analogue_noise_readings(monkeypatch, 1)
    result = _flag([(1, T0, 1.0), (1, T1, 1.0)], {2: "Primary"})
    assert result["drop_reason"].to_list() == ["initial_analogue_noise", None]
