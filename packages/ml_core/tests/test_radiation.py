from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import numpy as np
import polars as pl
import pvlib
import pytest
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from ml_core.features._radiation import LONG_WAVE, SHORT_WAVE, _half_hour_clear_sky
from polars.testing import assert_frame_equal
from scipy.interpolate import PchipInterpolator

COORDINATES: Final[pl.LazyFrame] = pl.DataFrame(
    {"time_series_id": [1], "latitude": [52.0], "longitude": [0.0]}
).lazy()


def _native(
    *,
    start: datetime,
    leads: list[int],
    short_wave: list[float | None],
    long_wave: list[float | None] | None = None,
) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "time_series_id": [1] * len(leads),
            "nwp_init_time": [start] * len(leads),
            "ensemble_member": [0] * len(leads),
            "valid_time": [start + timedelta(hours=lead) for lead in leads],
            SHORT_WAVE: short_wave,
        }
    )
    if long_wave is not None:
        frame = frame.with_columns(pl.Series(name=LONG_WAVE, values=long_wave))
    return frame.cast({name: pl.Float32 for name in (SHORT_WAVE, LONG_WAVE) if name in frame})


def _resample(
    frame: pl.DataFrame,
    *,
    coordinates: pl.LazyFrame = COORDINATES,
    engine: Literal["auto", "streaming"] = "auto",
) -> pl.DataFrame:
    return (
        _upsample_nwp_to_half_hourly(frame.lazy(), solar_coordinates=coordinates)
        .collect(engine=engine)
        .sort("valid_time")
    )


def _reference_sky(*, times: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    samples = []
    for offset in (450, 1350):
        stamps = times - timedelta(seconds=offset)
        position = pvlib.solarposition.get_solarposition(
            time=stamps.to_numpy(), latitude=latitude, longitude=longitude
        )
        cosine = np.maximum(np.cos(np.deg2rad(position["zenith"].to_numpy())), 0)
        samples.append(
            np.where(cosine > 0, 1098 * cosine * np.exp(-0.059 / np.maximum(cosine, 1e-12)), 0)
        )
    return np.mean(samples, axis=0)


@pytest.mark.parametrize(
    ("date", "latitude", "longitude"),
    [
        (datetime(2024, 2, 29, tzinfo=UTC), 49.0, -9.0),
        (datetime(2026, 6, 21, tzinfo=UTC), 61.0, 2.0),
        (datetime(2026, 12, 21, tzinfo=UTC), 52.0, 0.0),
    ],
)
def test_solar_geometry_against_independent_reference(
    date: datetime, latitude: float, longitude: float
) -> None:
    times = pl.datetime_range(date, date + timedelta(days=1), interval="30m", eager=True)
    targets = pl.DataFrame({"time_series_id": [1] * len(times), "valid_time": times})
    coordinates = pl.DataFrame(
        {"time_series_id": [1], "latitude": [latitude], "longitude": [longitude]}
    ).lazy()
    actual = _half_hour_clear_sky(targets=targets.lazy(), coordinates=coordinates).collect()
    actual = actual.sort("valid_time")["_clear_sky"].to_numpy()
    expected = _reference_sky(times=times, latitude=latitude, longitude=longitude)
    # US National Oceanic and Atmospheric Administration (NOAA) fractional-year geometry is
    # approximate; this absolute bound includes its sunrise error without accepting the multi-hour
    # error from treating native means as snapshots.
    np.testing.assert_allclose(actual, expected, atol=12, rtol=0)


@pytest.mark.parametrize("width", [3, 6])
def test_clear_day_timing_and_dark_targets(width: int) -> None:
    start = datetime(2026, 3, 21, tzinfo=UTC)
    times = pl.datetime_range(
        start + timedelta(minutes=30), start + timedelta(days=3), interval="30m", eager=True
    )
    truth = _reference_sky(times=times, latitude=52, longitude=0) * 0.7
    means = truth.reshape(-1, width * 2).mean(axis=1)
    # Six-hour windows are native after day six; move init backward for that case.
    init = start - timedelta(days=6) if width == 6 else start
    leads = [int((start - init).total_seconds() / 3600) + i * width for i in range(len(means) + 1)]
    native = _native(start=init, leads=leads, short_wave=[None, *means.tolist()])
    result = _resample(native)
    interior = result.filter(
        (pl.col("valid_time") > start + timedelta(days=1))
        & (pl.col("valid_time") <= start + timedelta(days=2))
    )
    expected = truth[48:96]
    actual = interior[SHORT_WAVE].to_numpy()
    hours = np.arange(1, 49) / 2
    assert abs(np.average(hours, weights=actual) - np.average(hours, weights=expected)) < 0.06
    np.testing.assert_allclose(actual, expected, atol=12, rtol=0)
    assert actual[hours >= 20].max() == 0


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_energy_transition_first_and_final_windows(engine: Literal["auto", "streaming"]) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    leads = [0, 3, 6, 141, 144, 150, 156, 354, 360]
    native = _native(
        start=start,
        leads=leads,
        short_wave=[None, 5.0, 20.0, 200.0, 10.0, 90.0, 300.0, 30.0, 100.0],
        long_wave=[None, 210.0, 220.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0],
    )
    result = _resample(native, engine=engine)
    assert result.filter(pl.col("valid_time") == start)[SHORT_WAVE][0] is None
    for row in native.iter_rows(named=True):
        lead = int((row["valid_time"] - start).total_seconds() / 3600)
        if lead == 0:
            continue
        width = 3 if lead <= 144 else 6
        window = result.filter(
            (pl.col("valid_time") > row["valid_time"] - timedelta(hours=width))
            & (pl.col("valid_time") <= row["valid_time"])
        )
        assert window.height == width * 2
        assert window[SHORT_WAVE].mean() == pytest.approx(row[SHORT_WAVE], rel=2e-7, abs=1e-5)
    assert result.schema[SHORT_WAVE] == pl.Float32


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_long_wave_midpoint_interpolation_across_schedule_transition(
    engine: Literal["auto", "streaming"],
) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=[138, 141, 144, 150, 156],
        short_wave=[None, 0.0, 0.0, 0.0, 0.0],
        long_wave=[None, 300.0, 330.0, 390.0, 360.0],
    ).drop(SHORT_WAVE)
    other = native.with_columns(
        ensemble_member=pl.lit(50, dtype=pl.Int64), **{LONG_WAVE: pl.col(LONG_WAVE) + 100}
    )
    mixed = pl.concat([native, other]).sample(fraction=1, shuffle=True, seed=5)
    result = _resample(mixed, engine=engine)
    supported = result.filter(
        (pl.col("ensemble_member") == 0) & (pl.col("valid_time") > start + timedelta(hours=138))
    )
    target_hours = (supported["valid_time"] - start).dt.total_seconds().to_numpy() / 3600 - 0.25
    anchors = np.array([139.5, 142.5, 147, 153])
    expected = PchipInterpolator(x=anchors, y=[300, 330, 390, 360])(
        np.clip(target_hours, anchors[0], anchors[-1])
    )
    np.testing.assert_allclose(supported[LONG_WAVE], expected, atol=2e-5, rtol=0)
    np.testing.assert_allclose(
        result.filter(pl.col("ensemble_member") == 50)[LONG_WAVE],
        result.filter(pl.col("ensemble_member") == 0)[LONG_WAVE] + 100,
        equal_nan=True,
    )
    assert supported.schema[LONG_WAVE] == pl.Float32


@pytest.mark.parametrize("absent_row", [False, True])
def test_long_wave_missing_window_separates_midpoint_interpolation(absent_row: bool) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=[0, 3, 6, 9, 12, 15],
        short_wave=[None, 0.0, 0.0, None, 0.0, 0.0],
        long_wave=[None, 200.0, 260.0, None, 350.0, 320.0],
    ).drop(SHORT_WAVE)
    if absent_row:
        native = native.filter(pl.col("valid_time") != start + timedelta(hours=9))
    result = _resample(native)
    gap = (pl.col("valid_time") > start + timedelta(hours=6)) & (
        pl.col("valid_time") <= start + timedelta(hours=9)
    )
    assert result.filter(gap)[LONG_WAVE].null_count() == 6
    for hour, expected in [(6, 260), (9.5, 350), (15, 320)]:
        assert (
            result.filter(pl.col("valid_time") == start + timedelta(hours=hour))[LONG_WAVE][0]
            == expected
        )
    changed = native.with_columns(
        pl.when(pl.col("valid_time") <= start + timedelta(hours=6))
        .then(pl.col(LONG_WAVE) + 100)
        .otherwise(pl.col(LONG_WAVE))
        .alias(LONG_WAVE)
    )
    after_gap = pl.col("valid_time") > start + timedelta(hours=9)
    assert_frame_equal(result.filter(after_gap), _resample(changed).filter(after_gap))


@pytest.mark.parametrize(
    "values",
    [
        [300.0],
        [300.0, 360.0],
        [300.0, 300.0, 360.0, 360.0],
        [300.0, 330.0, 390.0, 420.0],
        [300.0, 301.0, 390.0, 300.0],
    ],
)
def test_long_wave_pchip_matches_scipy_without_overshoot(values: list[float]) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=list(range(0, 3 * len(values) + 1, 3)),
        short_wave=[None, *([0.0] * len(values))],
        long_wave=[None, *values],
    ).drop(SHORT_WAVE)
    result = _resample(native).filter(pl.col("valid_time") > start)
    targets = np.arange(0.25, 3 * len(values), 0.5)
    anchors = np.arange(1.5, 3 * len(values), 3)
    expected = (
        PchipInterpolator(x=anchors, y=values)(np.clip(targets, anchors[0], anchors[-1]))
        if len(values) > 1
        else np.full_like(targets, values[0])
    )
    np.testing.assert_allclose(result[LONG_WAVE], expected, atol=2e-5, rtol=0)
    assert result[LONG_WAVE].to_numpy().min() >= min(values)
    assert result[LONG_WAVE].to_numpy().max() <= max(values)


def test_long_wave_mixed_series_match_independent_resampling() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=[0, 3, 6, 9],
        short_wave=[None, 0.0, 0.0, 0.0],
        long_wave=[None, 200.0, 260.0, 210.0],
    ).drop(SHORT_WAVE)
    other = _native(
        start=start,
        leads=[0, 3, 6, 9],
        short_wave=[None, 0.0, 0.0, 0.0],
        long_wave=[None, 400.0, 310.0, 380.0],
    ).drop(SHORT_WAVE)
    other = other.with_columns(time_series_id=pl.lit(2, dtype=pl.Int64))
    mixed = _resample(pl.concat([native, other]).sample(fraction=1, shuffle=True, seed=5))
    for trajectory in (native, other):
        assert_frame_equal(
            _resample(trajectory),
            mixed.filter(pl.col("time_series_id") == trajectory["time_series_id"][0]),
        )


@pytest.mark.parametrize("absent_row", [False, True])
def test_source_missing_window_is_not_filled(absent_row: bool) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=[0, 3, 6, 9, 12],
        short_wave=[None, 10.0, None, 200.0, 300.0],
        long_wave=[None, 250.0, None, 300.0, 350.0],
    )
    if absent_row:
        native = native.filter(pl.col("valid_time") != start + timedelta(hours=6))
    result = _resample(native)
    missing = result.filter(
        (pl.col("valid_time") > start + timedelta(hours=3))
        & (pl.col("valid_time") <= start + timedelta(hours=6))
    )
    for column in (SHORT_WAVE, LONG_WAVE):
        assert missing[column].null_count() == 6
    assert result.filter(pl.col("valid_time") == start + timedelta(minutes=30))[SHORT_WAVE][0] == 10


def test_midpoint_index_final_shape_and_missing_gap_isolation() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    times = pl.datetime_range(
        start + timedelta(minutes=30), start + timedelta(hours=24), interval="30m", eager=True
    )
    sky = (
        _half_hour_clear_sky(
            targets=pl.DataFrame({"time_series_id": [1] * len(times), "valid_time": times}).lazy(),
            coordinates=COORDINATES,
        )
        .collect()
        .sort("valid_time")["_clear_sky"]
        .to_numpy()
    )
    indices = np.array([0.0, 0.3, 0.4, 0.9, 0.5, 0.2, 0.7, 0.0])
    means = sky.reshape(8, 6).mean(axis=1) * indices
    native = _native(start=start, leads=list(range(0, 25, 3)), short_wave=[None, *means.tolist()])
    result = _resample(native)[SHORT_WAVE].to_numpy()[1:]
    # Independently construct the daylight interpolation and normalise each six-target window.
    daylight = sky.reshape(8, 6).mean(axis=1) >= 20
    midpoints = np.arange(1.5, 24, 3)[daylight]
    ratios = (
        np.array(native[SHORT_WAVE].to_list()[1:])[daylight]
        / sky.reshape(8, 6).mean(axis=1)[daylight]
    )
    target_midpoints = np.arange(0.25, 24, 0.5)
    expected = sky * np.interp(target_midpoints, midpoints, ratios)
    for i, mean in enumerate(means):
        block = slice(i * 6, (i + 1) * 6)
        expected[block] *= mean / expected[block].mean() if expected[block].mean() > 0 else 0
    np.testing.assert_allclose(result, expected, atol=3e-5, rtol=3e-7)
    missing = native.with_columns(
        pl.when(pl.col("valid_time") == start + timedelta(hours=12))
        .then(None)
        .otherwise(pl.col(SHORT_WAVE))
        .alias(SHORT_WAVE)
    )
    changed = missing.with_columns(
        pl.when(pl.col("valid_time") == start + timedelta(hours=9))
        .then(800.0)
        .otherwise(pl.col(SHORT_WAVE))
        .alias(SHORT_WAVE)
    )
    after_gap = pl.col("valid_time") > start + timedelta(hours=12)
    assert_frame_equal(_resample(missing).filter(after_gap), _resample(changed).filter(after_gap))


def test_absent_window_isolates_member_when_other_member_has_complete_support() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start,
        leads=list(range(0, 25, 3)),
        short_wave=[None, 10.0, 30.0, 200.0, 250.0, 300.0, 400.0, 50.0, 0.0],
    )
    missing = native.filter(pl.col("valid_time") != start + timedelta(hours=12))
    complete_member = native.with_columns(ensemble_member=pl.lit(50, dtype=pl.Int64))
    combined = pl.concat([missing, complete_member])
    changed = combined.with_columns(
        pl.when(
            (pl.col("ensemble_member") == 0) & (pl.col("valid_time") == start + timedelta(hours=9))
        )
        .then(800.0)
        .otherwise(pl.col(SHORT_WAVE))
        .alias(SHORT_WAVE)
    )
    after_gap = (pl.col("ensemble_member") == 0) & (
        pl.col("valid_time") > start + timedelta(hours=12)
    )
    assert_frame_equal(_resample(combined).filter(after_gap), _resample(changed).filter(after_gap))


def test_dark_tiny_denominator_and_cropped_first_window() -> None:
    start = datetime(2026, 12, 21, tzinfo=UTC)
    native = _native(start=start, leads=[0, 3, 6, 9, 12], short_wave=[None, 0.0, 0.4, 10.0, 50.0])
    result = _resample(native)
    assert (
        result.filter(
            (pl.col("valid_time") > start) & (pl.col("valid_time") <= start + timedelta(hours=3))
        )[SHORT_WAVE].max()
        == 0
    )
    assert result.filter(
        (pl.col("valid_time") > start + timedelta(hours=3))
        & (pl.col("valid_time") <= start + timedelta(hours=6))
    )[SHORT_WAVE].unique().to_list() == pytest.approx([0.4])
    assert result[SHORT_WAVE].drop_nulls().is_finite().all()
    cropped = native.filter(pl.col("valid_time") >= start + timedelta(hours=9))
    assert_frame_equal(
        _resample(cropped), result.filter(pl.col("valid_time") >= start + timedelta(hours=9))
    )


def test_northern_summer_daylight_blocks_and_group_isolation() -> None:
    start = datetime(2026, 6, 21, tzinfo=UTC)
    init = start - timedelta(days=6)
    coordinates = pl.DataFrame(
        {"time_series_id": [1], "latitude": [61.0], "longitude": [2.0]}
    ).lazy()
    native = _native(
        start=init,
        leads=list(range(144, 193, 6)),
        short_wave=[None, 80.0, 400.0, 500.0, 100.0, 50.0, 200.0, 250.0, 60.0],
    )
    result = _resample(native, coordinates=coordinates)
    changed = native.with_columns(
        pl.when(pl.col("valid_time") <= start + timedelta(days=1))
        .then(pl.col(SHORT_WAVE) * 2)
        .otherwise(pl.col(SHORT_WAVE))
        .alias(SHORT_WAVE)
    )
    second_day = pl.col("valid_time") > start + timedelta(days=1)
    assert_frame_equal(
        result.filter(second_day), _resample(changed, coordinates=coordinates).filter(second_day)
    )
    other = native.with_columns(
        ensemble_member=pl.lit(50, dtype=pl.Int64), **{SHORT_WAVE: pl.col(SHORT_WAVE) * 2}
    )
    mixed = _resample(
        pl.concat([native, other]).sample(fraction=1, shuffle=True, seed=5), coordinates=coordinates
    )
    assert_frame_equal(result, mixed.filter(pl.col("ensemble_member") == 0))
    np.testing.assert_allclose(
        mixed.filter(pl.col("ensemble_member") == 50)[SHORT_WAVE],
        result[SHORT_WAVE] * 2,
        equal_nan=True,
    )


def test_unsupported_lead_and_long_wave_without_coordinates() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(start=start, leads=[0, 4], short_wave=[None, 100.0])
    with pytest.raises(ValueError, match="unsupported_radiation_schedule"):
        _resample(native)
    native = _native(
        start=start, leads=[0, 3, 6], short_wave=[None, 0.0, 0.0], long_wave=[None, 200.0, 300.0]
    ).drop(SHORT_WAVE)
    result = _upsample_nwp_to_half_hourly(native.lazy()).collect()
    assert result.filter(pl.col("valid_time") == start + timedelta(minutes=30))[LONG_WAVE][0] == 200
    with pytest.raises(ValueError, match="solar_coordinates"):
        _upsample_nwp_to_half_hourly(native.with_columns(pl.lit(0.0).alias(SHORT_WAVE)).lazy())


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_weak_dawn_index_cannot_distort_next_window(engine: Literal["auto", "streaming"]) -> None:
    start = datetime(2026, 4, 5, tzinfo=UTC)
    times = pl.datetime_range(
        start + timedelta(minutes=30), start + timedelta(hours=12), interval="30m", eager=True
    )
    # Order explicitly: the geometry aggregation need not retain chronological order.
    sky = (
        _half_hour_clear_sky(
            targets=pl.DataFrame({"time_series_id": [1] * len(times), "valid_time": times}).lazy(),
            coordinates=COORDINATES,
        )
        .collect()
        .sort("valid_time")["_clear_sky"]
        .to_numpy()
    )
    means = sky.reshape(-1, 6).mean(axis=1)
    assert 1 < means[1] < 2
    outputs = []
    for dawn_index in (0.6, 10.0):
        values = [None, 0.0, means[1] * dawn_index, means[2] * 0.6, means[3] * 0.6]
        native = _native(start=start, leads=[0, 3, 6, 9, 12], short_wave=values)
        result = _resample(native, engine=engine)
        dawn = result.filter(
            (pl.col("valid_time") > start + timedelta(hours=3))
            & (pl.col("valid_time") <= start + timedelta(hours=6))
        )[SHORT_WAVE]
        assert dawn.mean() == pytest.approx(values[2], abs=2e-6)
        outputs.append(result.filter(pl.col("valid_time") > start + timedelta(hours=6)))
    assert_frame_equal(*outputs)


@pytest.mark.parametrize("engine", ["auto", "streaming"])
@pytest.mark.parametrize("unknown_model", [False, True])
def test_unsupported_schedule_is_reported_and_serving_retains_other_weather(
    engine: Literal["auto", "streaming"], unknown_model: bool, caplog: pytest.LogCaptureFixture
) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start, leads=[0, 3 if unknown_model else 4], short_wave=[None, 100.0]
    ).with_columns(temperature_2m=pl.Series([10.0, 20.0], dtype=pl.Float32))
    if unknown_model:
        native = native.with_columns(nwp_model_id=pl.lit("unsupported_source"))
    with pytest.raises(ValueError, match="unsupported_radiation_schedule"):
        _upsample_nwp_to_half_hourly(native.lazy(), solar_coordinates=COORDINATES).collect(
            engine=engine
        )
    result = _upsample_nwp_to_half_hourly(
        native.lazy(), solar_coordinates=COORDINATES, strict_radiation_schedule=False
    ).collect(engine=engine)
    assert "unsupported_radiation_schedule" in caplog.text
    assert str(start.year) in caplog.text
    assert result[SHORT_WAVE].null_count() == result.height
    assert result["temperature_2m"].null_count() == 0
    assert result["temperature_2m"].min() == 10
    assert result["temperature_2m"].max() == 20


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_shared_absent_window_isolates_short_wave_segments(
    engine: Literal["auto", "streaming"],
) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    native = _native(
        start=start, leads=[0, 3, 6, 12, 15], short_wave=[None, 0.0, 10.0, 100.0, 80.0]
    )
    changed = native.with_columns(
        pl.when(pl.col("valid_time") == start + timedelta(hours=6))
        .then(1000.0)
        .otherwise(pl.col(SHORT_WAVE))
        .alias(SHORT_WAVE)
    )
    after_gap = pl.col("valid_time") > start + timedelta(hours=9)
    assert_frame_equal(
        _resample(native, engine=engine).filter(after_gap),
        _resample(changed, engine=engine).filter(after_gap),
    )


@pytest.mark.parametrize("engine", ["auto", "streaming"])
@pytest.mark.parametrize(
    "fault", ["missing_site", "null_latitude", "nan_longitude", "infinite_latitude"]
)
def test_invalid_solar_coordinates_are_not_treated_as_dark_weather(
    engine: Literal["auto", "streaming"], fault: str
) -> None:
    start = datetime(2026, 3, 21, tzinfo=UTC)
    native = _native(start=start, leads=[0, 12], short_wave=[None, 400.0])
    coordinates = pl.DataFrame(
        {
            "time_series_id": [2 if fault == "missing_site" else 1],
            "latitude": [
                None
                if fault == "null_latitude"
                else float("inf")
                if fault == "infinite_latitude"
                else 52.0
            ],
            "longitude": [float("nan") if fault == "nan_longitude" else 0.0],
        },
        schema={"time_series_id": pl.Int64, "latitude": pl.Float64, "longitude": pl.Float64},
    )
    with pytest.raises(ValueError, match="invalid_solar_coordinates"):
        _resample(native, coordinates=coordinates.lazy(), engine=engine)
