from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.served_column_checks import (
    check_direct_is_not_a_separation_model,
    check_hourly_value_is_a_backward_mean,
)
from studies.solar import cos_zenith, cos_zenith_hour_mean, extraterrestrial_horizontal, zenith

# A generic coordinate in eastern England, deliberately not one of the metered generators'. The
# geometry is built with pvlib rather than hand-set, because the ratio of the instantaneous to the
# hour-mean cosine is near 1.0 around midday: a hand-set frame lands close to the 5 W m-2 threshold
# by accident, where real geometry over several days misses it by tens of W m-2.
LATITUDE = 52.9
LONGITUDE = 0.1
# The reconstruction check needs only a few days, and each row costs 60 extra solar-position
# evaluations for the hour-mean cosine. The spread check needs enough rows to fill a
# (clearness, zenith) bin 30 deep, and needs no hour-mean at all.
RECONSTRUCTION_HOURS = 24 * 5
SPREAD_HOURS = 24 * 60
CLEAR_SKY_FRACTION = 0.75


def _stamps(n_hours: int) -> pl.Series:
    return pl.Series(
        "time",
        [datetime(2025, 6, 1, tzinfo=UTC) + timedelta(hours=hour) for hour in range(n_hours)],
    ).dt.replace_time_zone("UTC")


def _geometry(*, n_hours: int, with_hour_mean: bool) -> pl.DataFrame:
    stamps = _stamps(n_hours)
    midpoint_zenith = zenith(
        stamps=stamps.dt.offset_by("-30m"), latitude=LATITUDE, longitude=LONGITUDE
    )
    extraterrestrial = extraterrestrial_horizontal(stamps=stamps, zenith_deg=midpoint_zenith)
    frame = pl.DataFrame(
        {
            "time": stamps,
            "solar_zenith_deg": midpoint_zenith,
            "extraterrestrial_horizontal_w_m2": extraterrestrial,
            "ghi_w_m2": extraterrestrial * CLEAR_SKY_FRACTION,
        }
    )
    if not with_hour_mean:
        return frame
    return frame.with_columns(
        cos_zenith_instant=pl.Series(
            cos_zenith(zenith_deg=zenith(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE))
        ),
        cos_zenith_hour_mean=pl.Series(
            cos_zenith_hour_mean(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)
        ),
    )


def _reconstructable() -> pl.DataFrame:
    """A frame whose `_instant` columns are what the conversion would actually serve."""
    frame = _geometry(n_hours=RECONSTRUCTION_HOURS, with_hour_mean=True)
    factor = frame["cos_zenith_instant"].to_numpy() / np.maximum(
        frame["cos_zenith_hour_mean"].to_numpy(), 1e-9
    )
    return frame.with_columns(
        bhi_w_m2=pl.col("ghi_w_m2") * 0.7,
        ghi_instant_w_m2=pl.col("ghi_w_m2") * factor,
        bhi_instant_w_m2=pl.col("ghi_w_m2") * 0.7 * factor,
    )


def test_a_correctly_converted_column_passes():
    check_hourly_value_is_a_backward_mean(frame=_reconstructable())


def test_a_correctly_converted_column_with_real_world_noise_passes():
    # Served columns carry rounding and noise of a few W m-2. A threshold tightened below that
    # would reject every real download, and an exact fixture alone cannot notice.
    noise = np.random.default_rng(0).normal(0.0, 2.0, size=RECONSTRUCTION_HOURS)
    frame = _reconstructable().with_columns(
        ghi_instant_w_m2=pl.col("ghi_instant_w_m2") + noise,
        bhi_instant_w_m2=pl.col("bhi_instant_w_m2") + noise,
    )

    check_hourly_value_is_a_backward_mean(frame=frame)


def test_a_label_naming_the_wrong_hour_raises():
    # The failure the check exists for: the served hourly column is the snapshot rather than the
    # backward mean, which is what a half-hour error in the label looks like from the inside.
    frame = _reconstructable().with_columns(
        ghi_instant_w_m2=pl.col("ghi_w_m2"), bhi_instant_w_m2=pl.col("bhi_w_m2")
    )

    with pytest.raises(ValueError, match="does not reconstruct"):
        check_hourly_value_is_a_backward_mean(frame=frame)


def test_a_conversion_half_an_hour_off_raises():
    # The subtler failure: the served snapshot was converted for an hour half an hour away from
    # the one the label names. The reconstruction misses by about 14 W m-2, which a threshold
    # loosened to tens of W m-2 would let through.
    frame = _reconstructable()
    stamps = _stamps(RECONSTRUCTION_HOURS).dt.offset_by("30m")
    factor = cos_zenith(
        zenith_deg=zenith(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)
    ) / np.maximum(
        cos_zenith_hour_mean(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE), 1e-9
    )
    shifted = frame.with_columns(
        ghi_instant_w_m2=pl.col("ghi_w_m2") * factor,
        bhi_instant_w_m2=pl.col("bhi_w_m2") * factor,
    )

    with pytest.raises(ValueError, match="does not reconstruct"):
        check_hourly_value_is_a_backward_mean(frame=shifted)


@pytest.mark.parametrize(
    ("broken", "source"),
    [("ghi_instant_w_m2", "ghi_w_m2"), ("bhi_instant_w_m2", "bhi_w_m2")],
    ids=["global", "direct"],
)
def test_either_flux_failing_alone_raises(broken: str, source: str):
    frame = _reconstructable().with_columns(pl.col(source).alias(broken))

    with pytest.raises(ValueError, match=f"{source} does not reconstruct"):
        check_hourly_value_is_a_backward_mean(frame=frame)


def _with_direct_fraction(
    fraction: np.ndarray, *, clearness: np.ndarray | None = None
) -> pl.DataFrame:
    frame = _geometry(n_hours=SPREAD_HOURS, with_hour_mean=False)
    if clearness is not None:
        frame = frame.with_columns(
            ghi_w_m2=pl.col("extraterrestrial_horizontal_w_m2") * pl.Series(clearness)
        )
    return frame.with_columns(
        clearness_index=pl.Series(
            np.where(
                frame["extraterrestrial_horizontal_w_m2"].to_numpy() > 0.0,
                frame["ghi_w_m2"].to_numpy()
                / np.maximum(frame["extraterrestrial_horizontal_w_m2"].to_numpy(), 1e-9),
                0.0,
            )
        ),
        bhi_w_m2=pl.col("ghi_w_m2") * pl.Series(fraction),
    )


def test_a_direct_fraction_that_is_a_function_of_clearness_and_geometry_raises():
    # A separation model's direct fraction is by construction a function of the clearness index and
    # the solar zenith angle, so inside a fine bin on those two it is very nearly constant. That is
    # the failure this check exists to catch.
    frame = _geometry(n_hours=SPREAD_HOURS, with_hour_mean=False)
    zenith_deg = frame["solar_zenith_deg"].to_numpy()
    separation_model = np.clip(0.9 - 0.004 * zenith_deg, 0.0, 1.0)

    with pytest.raises(ValueError, match="varies by only"):
        check_direct_is_not_a_separation_model(frame=_with_direct_fraction(separation_model))


def test_a_direct_fraction_that_follows_clearness_raises():
    # A separation model follows the clearness index as well as the zenith. With the clearness
    # alternating between a cloudy and a clear sky, only binning on it separately makes the
    # fraction look constant inside a bin. Two levels keep each bin full enough to score.
    frame = _geometry(n_hours=SPREAD_HOURS, with_hour_mean=False)
    clearness = np.random.default_rng(1).choice([0.325, 0.725], size=SPREAD_HOURS)
    zenith_deg = frame["solar_zenith_deg"].to_numpy()
    separation_model = np.clip(1.4 * clearness - 0.3 - 0.002 * zenith_deg, 0.0, 1.0)

    with pytest.raises(ValueError, match="varies by only"):
        check_direct_is_not_a_separation_model(
            frame=_with_direct_fraction(separation_model, clearness=clearness)
        )


@pytest.mark.parametrize(
    ("within_bin_std", "passes"), [(0.1, True), (0.02, False)], ids=["informative", "near_floor"]
)
def test_the_spread_threshold_sits_between_a_real_field_and_a_separation_model(
    within_bin_std: float, passes: bool
):
    # UKV's published fraction measured a within-bin spread of 0.118 and an Erbs fraction 0.016,
    # so the threshold has to pass the first scale and reject the second.
    half_width = within_bin_std * np.sqrt(3.0)
    noise = np.random.default_rng(0).uniform(-half_width, half_width, size=SPREAD_HOURS)
    frame = _with_direct_fraction(np.full(SPREAD_HOURS, 0.5) + noise)

    if passes:
        check_direct_is_not_a_separation_model(frame=frame)
    else:
        with pytest.raises(ValueError, match="varies by only"):
            check_direct_is_not_a_separation_model(frame=frame)


@pytest.mark.parametrize(
    ("informative_below_zenith_deg", "passes"),
    [(45.0, False), (75.0, True)],
    ids=["three_of_eight_bins", "seven_of_eight_bins"],
)
def test_the_verdict_follows_the_typical_bin_rather_than_the_extreme_one(
    informative_below_zenith_deg: float, passes: bool
):
    # The fixture fills eight zenith bins. A field informative in only three of them is judged a
    # separation model, and one informative in seven is not: one bin at either extreme must not
    # decide the verdict.
    frame = _geometry(n_hours=SPREAD_HOURS, with_hour_mean=False)
    informative = frame["solar_zenith_deg"].to_numpy() < informative_below_zenith_deg
    noise = np.random.default_rng(0).uniform(-0.17, 0.17, size=SPREAD_HOURS)
    frame = _with_direct_fraction(np.where(informative, 0.5 + noise, 0.5))

    if passes:
        check_direct_is_not_a_separation_model(frame=frame)
    else:
        with pytest.raises(ValueError, match="varies by only"):
            check_direct_is_not_a_separation_model(frame=frame)


def test_too_few_rows_per_bin_raises_a_different_error():
    # The function raises ValueError twice, and this guard fires first. Without matching on the
    # message, a thin fixture would satisfy `pytest.raises(ValueError)` for the wrong reason and
    # the spread test below it would never run.
    thin = _with_direct_fraction(np.full(SPREAD_HOURS, 0.5)).head(5)

    with pytest.raises(ValueError, match="cannot run"):
        check_direct_is_not_a_separation_model(frame=thin)
