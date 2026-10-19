"""Tests of the convex model fitted through an MVA meter (`studies.dp_meter`, experiment 110).

Each test replaces the power with a truth made from the model's own design. Where the truth must
reverse flow at midday, the capacity is large enough that solar output exceeds demand, and the
reading is `sqrt(P^2 + Q^2)` with `Q = tan(phi) * D`, so a fit that recovers the sign, the capacity,
and the power factor shows the Gauss-Newton fit climbed out of the mirror basin.
"""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    GamLayout,
    GamSpec,
    TrainingWindow,
    design_matrix,
    fit_gam,
    predict_gam,
    prepare_fit,
    training_rows,
)
from studies.dp_meter import (  # noqa: E402
    CAPACITY_START_MULTIPLIERS,
    demand_column_mask,
    fit_gam_through_meter,
    meter_replica_residuals,
    predict_meter,
)
from studies.dp_operator import DEFAULT_EPSILON, tan_phi_from_power_factor  # noqa: E402
from studies.dp_pinned_capacity import pinned_capacity  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
WINDOW: Final[TrainingWindow] = TrainingWindow(months=18)
NOISE_POINTS: Final[float] = 0.1
SPEC: Final[GamSpec] = GamSpec(ridge=0.01)
POWER_FACTOR: Final[float] = 0.94
"""The truth's power factor, away from the prior's 0.97 so that a fit stuck at its start fails."""


def _truth(*, layout: GamLayout, capacity: dict[int, float]) -> np.ndarray:
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    shape = 40.0 + 6.0 * np.sin(2 * np.pi * np.arange(48) / 48 - 1.0)
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        theta[profile.start + day_type * 48 : profile.start + (day_type + 1) * 48] = shape + offset
    heating = layout.slices["heating"]
    for band, slope in enumerate((0.8, 1.5, 1.0, 1.8)):
        theta[heating.start + 4 + band] = slope
    theta[layout.slices["lighting"]] = (0.5, 0.3, 0.2, 0.1)
    for year, increment in capacity.items():
        theta[layout.slices["capacity"].start + layout.season_years.index(year)] = increment
    return theta


def _cache_with_reading(
    *, cache: FeatureCache, capacity: dict[int, float], reactive: bool
) -> tuple[FeatureCache, GamLayout, np.ndarray, np.ndarray]:
    """Return the cache with the reading made from the truth, the layout, truth, and signed P."""
    layout, _ = prepare_fit(
        cache=cache, site="S1", spec=SPEC, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    theta = _truth(layout=layout, capacity=capacity)
    mask = demand_column_mask(layout=layout)
    features = cache.sites["S1"]
    usable = np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    signed = np.full(len(cache.end_times), np.nan)
    reading = np.full(len(cache.end_times), np.nan)
    tan_phi = tan_phi_from_power_factor(power_factor=POWER_FACTOR) if reactive else 0.0
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        design = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        real = design @ theta
        demand = design[:, mask] @ theta[mask]
        signed[rows] = real
        reading[rows] = np.sqrt(real**2 + (tan_phi * demand) ** 2 + DEFAULT_EPSILON)
    reading = reading + NOISE_POINTS * np.random.default_rng(8).normal(size=len(reading))
    return replace(cache, sites={"S1": replace(features, y=np.abs(reading))}), layout, theta, signed


def _meter(*, cache: FeatureCache, qmode: str):  # noqa: ANN202
    return fit_gam_through_meter(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        qmode=qmode,  # ty: ignore[invalid-argument-type]
    )


def _balanced_accuracy(*, truth_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    true_positive_rate = predicted_positive[truth_positive].mean()
    true_negative_rate = (~predicted_positive[~truth_positive]).mean()
    return float((true_positive_rate + true_negative_rate) / 2.0)


def test_with_zero_reactive_power_and_positive_flow_the_meter_fit_matches_the_plain_fit(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _, signed = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 8.0, 2022: 10.0, 2023: 5.0}, reactive=False
    )
    assert np.nanmin(signed) > 5.0
    plain = fit_gam(
        cache=cache,
        site="S1",
        specs=[SPEC],
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    meter = _meter(cache=cache, qmode="zero")
    rows = np.flatnonzero(
        (cache.end_times > np.datetime64("2024-01-01T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-09-01T00:00", "us"))
        & np.isfinite(cache.sites["S1"].y)
    )
    difference = np.abs(
        predict_gam(cache=cache, fit=plain, index=rows)
        - predict_gam(cache=cache, fit=meter.fit, index=rows)
    )
    assert difference.max() < 1e-5
    assert meter.tan_phi == 0.0
    assert meter.q0 == 0.0


def test_the_learned_fit_recovers_reverse_flow_capacity_and_power_factor(
    dp_synthetic_cache: FeatureCache,
):
    capacity = {2020: 35.0, 2022: 40.0, 2023: 25.0}
    cache, layout, theta, signed = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=capacity, reactive=True
    )
    rows = np.flatnonzero(
        (cache.end_times > np.datetime64("2023-01-01T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-07-01T00:00", "us"))
        & np.isfinite(cache.sites["S1"].y)
    )
    assert (signed[rows] < 0).mean() > 0.03, "the truth must reverse flow often enough to test it"
    meter = _meter(cache=cache, qmode="learned")
    recovered = meter.fit.capacity_points()[2023]
    assert recovered == pytest.approx(sum(capacity.values()), rel=0.15)
    assert meter.power_factor == pytest.approx(POWER_FACTOR, abs=0.015)
    assert meter.tan_phi >= 0.0
    predicted = predict_meter(cache=cache, meter=meter, index=rows)
    accuracy = _balanced_accuracy(
        truth_positive=signed[rows] > 0, predicted_positive=predicted.real_power > 0
    )
    assert accuracy > 0.9
    # The reading is |S|, so the predicted apparent power must track the reading.
    assert np.abs(predicted.apparent_power - cache.sites["S1"].y[rows]).mean() < 0.5
    assert len(meter.starts) == len(CAPACITY_START_MULTIPLIERS) + 1
    assert meter.loss == min(start.loss for start in meter.starts)
    assert np.isfinite(meter.capacity_spread)
    del layout, theta


def test_reading_the_meter_as_signed_power_underestimates_capacity_that_the_meter_fit_recovers(
    dp_synthetic_cache: FeatureCache,
):
    capacity = {2020: 35.0, 2022: 40.0, 2023: 25.0}
    cache, _, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=capacity, reactive=False
    )
    naive = fit_gam(
        cache=cache,
        site="S1",
        specs=[SPEC],
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    meter = _meter(cache=cache, qmode="zero")
    truth = sum(capacity.values())
    naive_error = abs(naive.capacity_points()[2023] - truth)
    meter_error = abs(meter.fit.capacity_points()[2023] - truth)
    assert meter_error < 0.5 * naive_error


def test_the_meter_fit_uses_training_rows_only(dp_synthetic_cache: FeatureCache):
    cache, _, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 8.0, 2022: 10.0, 2023: 5.0}, reactive=False
    )
    features = cache.sites["S1"]
    after = cache.end_times > np.datetime64("2024-07-01T00:00", "us")
    garbled = replace(cache, sites={"S1": replace(features, y=np.where(after, 1e3, features.y))})
    first = _meter(cache=cache, qmode="zero")
    second = _meter(cache=garbled, qmode="zero")
    assert np.array_equal(first.fit.theta, second.fit.theta)


def test_the_meter_fit_refuses_a_pinned_capacity(dp_synthetic_cache: FeatureCache):
    with pytest.raises(ValueError, match="fitted capacity"):
        fit_gam_through_meter(
            cache=dp_synthetic_cache,
            site="S1",
            spec=GamSpec(capacity="pinned"),
            cut=CUT,
            window=WINDOW,
            season_years=SEASON_YEARS,
            qmode="zero",
        )


def test_predictions_follow_the_apparent_power_formula(dp_synthetic_cache: FeatureCache):
    cache, _, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 12.0, 2022: 12.0}, reactive=True
    )
    meter = _meter(cache=cache, qmode="learned")
    rows = np.flatnonzero(np.isfinite(cache.sites["S1"].y))[::997]
    rows = rows[cache.end_times[rows] <= np.datetime64("2024-07-01T00:00", "us")]
    predicted = predict_meter(cache=cache, meter=meter, index=rows)
    reactive = meter.tan_phi * predicted.demand + meter.q0
    assert predicted.reactive_power == pytest.approx(reactive)
    assert predicted.apparent_power == pytest.approx(
        np.sqrt(predicted.real_power**2 + reactive**2 + DEFAULT_EPSILON)
    )
    assert (predicted.apparent_power >= 0.0).all()


def test_a_unity_power_factor_truth_keeps_the_ratio_of_reactive_to_real_power_non_negative(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 35.0, 2022: 40.0, 2023: 25.0}, reactive=False
    )
    meter = _meter(cache=cache, qmode="learned")
    assert meter.tan_phi >= 0.0
    assert meter.power_factor <= 1.0


def test_the_meter_replica_fits_twelve_months_earlier_and_predicts_only_the_year_after(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 8.0, 2022: 10.0, 2023: 5.0}, reactive=False
    )
    final_cut = datetime(2025, 7, 1, tzinfo=UTC)
    replica = meter_replica_residuals(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=final_cut,
        window=WINDOW,
        season_years=SEASON_YEARS,
        qmode="zero",
    )
    assert replica.fit_cut == CUT
    ends = cache.end_times[replica.index]
    assert ends.min() > np.datetime64("2024-07-01T00:00", "us")
    assert ends.max() <= np.datetime64("2025-07-01T00:00", "us")
    assert np.abs(replica.residuals).mean() < 1.0
    assert replica.band.shape == replica.residuals.shape


def test_a_dark_only_window_trains_on_dark_rows_and_a_zero_capacity_removes_the_solar_term(
    dp_synthetic_cache: FeatureCache,
):
    features = dp_synthetic_cache.sites["S1"]
    dark = (features.pv_unit_shape < 1e-6) & np.isfinite(features.pv_unit_shape)
    cache = replace(dp_synthetic_cache, sites={"S1": replace(features, dark=dark)})
    all_rows = training_rows(
        features=cache.sites["S1"], cache=cache, cut=CUT, window=TrainingWindow(months=12)
    )
    dark_rows = training_rows(
        features=cache.sites["S1"],
        cache=cache,
        cut=CUT,
        window=TrainingWindow(months=12, dark_only=True),
    )
    assert 0 < len(dark_rows) < len(all_rows)
    assert cache.sites["S1"].dark[dark_rows].all()
    weeks = [date(2019, 8, 26) + timedelta(days=7 * i) for i in range(370)]
    zero_vintages = pl.DataFrame(
        {
            "time_series_id": features.table.time_series_id,
            "variant": "frozen",
            "origin_monday": date(2019, 9, 2),
            "week_start": weeks,
            "capacity_mw": 0.0,
            "unobserved": False,
        }
    )
    zero = pinned_capacity(vintages=zero_vintages, table=features.table, cut=CUT)
    assert zero.held_points == 0.0
    assert zero.at_rows(cache=cache, index=dark_rows[:10]) == pytest.approx(0.0)
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[GamSpec(capacity="pinned")],
        cut=CUT,
        window=TrainingWindow(months=12, dark_only=True),
        season_years=SEASON_YEARS,
        pinned=zero,
    )
    sunny = np.flatnonzero(
        (features.pv_unit_shape > 0.5)
        & (cache.end_times > np.datetime64("2023-06-01", "us"))
        & (cache.end_times <= np.datetime64("2024-07-01", "us"))
    )[:20]
    reading = predict_gam(cache=cache, fit=fit, index=sunny)
    demand_only = predict_gam(
        cache=cache, fit=fit, index=sunny, columns=np.ones(fit.layout.n_columns, dtype=bool)
    )
    assert reading == pytest.approx(demand_only)
