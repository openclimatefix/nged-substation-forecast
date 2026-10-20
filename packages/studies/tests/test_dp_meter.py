"""Tests of the convex model fitted through an MVA meter (`studies.dp_meter`, experiment 110).

Each test replaces the power with a truth made from the model's own design, or from a plant unlike
the model's. Where the truth must reverse flow at midday, the capacity is large enough that solar
output exceeds demand, and the reading is `sqrt(P^2 + Q^2)` with `Q = tan(phi) * D + q0`, so a fit
that recovers the sign, the capacity, and the power factor shows the Gauss-Newton fit climbed out of
the mirror basin. Where the truth has no reverse flow, the tests check that the reactive pair is
not learned from nothing.
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
    ClipSpec,
    GamLayout,
    GamSpec,
    NodeSpec,
    TrainingWindow,
    derated_unit_shape,
    design_matrix,
    fit_gam,
    predict_gam,
    prepare_fit,
    training_rows,
)
from studies.dp_meter import (  # noqa: E402
    CAPACITY_START_MULTIPLIERS,
    MIN_REVERSE_FLOW_SHARE,
    TAN_PHI_MAX,
    TAN_PHI_PRIOR_MEAN,
    MeterFit,
    QModeType,
    demand_column_mask,
    fit_gam_through_meter,
    meter_replica_residuals,
    predict_meter,
)
from studies.dp_operator import DEFAULT_EPSILON, tan_phi_from_power_factor  # noqa: E402
from studies.dp_pinned_capacity import pinned_capacity  # noqa: E402

from studies import dp_meter  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
WINDOW: Final[TrainingWindow] = TrainingWindow(months=18)
NOISE_POINTS: Final[float] = 0.1
SPEC: Final[GamSpec] = GamSpec(ridge=0.01)
REVERSE_FLOW: Final[dict[int, float]] = {2020: 35.0, 2022: 40.0, 2023: 25.0}
"""Capacity increments large enough that solar output exceeds demand at midday."""
NO_REVERSE_FLOW: Final[dict[int, float]] = {2020: 8.0, 2022: 10.0, 2023: 5.0}
SMOOTHING_HALF_HOURS: Final[int] = 7


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


def _smoothed_plant(*, features, rows: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Return the derated output per point of capacity of a plant unlike the model's.

    The plant's shape is the unit shape smoothed over seven half-hours, a broader day than the
    model's, as an east-west plant is.
    """
    smoothed = np.convolve(
        np.nan_to_num(features.pv_unit_shape),
        np.ones(SMOOTHING_HALF_HOURS) / SMOOTHING_HALF_HOURS,
        mode="same",
    )
    return derated_unit_shape(features=replace(features, pv_unit_shape=smoothed), index=rows)


def _cache_with_reading(
    *,
    cache: FeatureCache,
    capacity: dict[int, float],
    power_factor: float | None,
    q0: float = 0.0,
    mismatched_plant: bool = False,
) -> tuple[FeatureCache, np.ndarray, np.ndarray]:
    """Return the cache with the reading made from the truth, the signed P, and a signed series.

    The signed series is `P` plus the same noise, as a fit to the signed power would see it.
    """
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
    tan_phi = 0.0 if power_factor is None else tan_phi_from_power_factor(power_factor=power_factor)
    years = np.array(layout.season_years)
    increments = theta[layout.slices["capacity"]]
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        design = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        demand = design[:, mask] @ theta[mask]
        if mismatched_plant:
            held = cache.calendar.season_year[rows][:, None] >= years[None, :]
            solar = (held @ increments) * _smoothed_plant(features=features, rows=rows)
            real = demand - solar
        else:
            real = design @ theta
        signed[rows] = real
        reactive = tan_phi * demand + (0.0 if power_factor is None else q0)
        reading[rows] = np.sqrt(real**2 + reactive**2 + DEFAULT_EPSILON)
    noise = NOISE_POINTS * np.random.default_rng(8).normal(size=len(reading))
    metered = replace(cache, sites={"S1": replace(features, y=np.abs(reading + noise))})
    return metered, signed, signed + noise


def _meter(*, cache: FeatureCache, qmode: QModeType) -> MeterFit:
    return fit_gam_through_meter(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        qmode=qmode,
    )


def _signed_capacity(*, cache: FeatureCache, signed_series: np.ndarray) -> float:
    """Return the capacity of the plain fit to the signed series, the reference of gate 3."""
    features = cache.sites["S1"]
    reference = fit_gam(
        cache=replace(cache, sites={"S1": replace(features, y=signed_series)}),
        site="S1",
        specs=[SPEC],
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    return reference.capacity_points()[reference.last_season_year]


def _balanced_accuracy(*, truth_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    true_positive_rate = predicted_positive[truth_positive].mean()
    true_negative_rate = (~predicted_positive[~truth_positive]).mean()
    return float((true_positive_rate + true_negative_rate) / 2.0)


def _scored_rows(*, cache: FeatureCache) -> np.ndarray:
    return np.flatnonzero(
        (cache.end_times > np.datetime64("2023-01-01T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-07-01T00:00", "us"))
        & np.isfinite(cache.sites["S1"].y)
    )


def test_with_zero_reactive_power_and_positive_flow_the_meter_fit_matches_the_plain_fit(
    dp_synthetic_cache: FeatureCache,
):
    cache, signed, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=NO_REVERSE_FLOW, power_factor=None
    )
    assert np.nanmin(signed) > 5.0
    plain = fit_gam(
        cache=cache, site="S1", specs=[SPEC], cut=CUT, window=WINDOW, season_years=SEASON_YEARS
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
    assert (meter.tan_phi, meter.q0, meter.reactive) == (0.0, 0.0, "zero")


def test_the_learned_fit_recovers_reverse_flow_capacity_power_factor_and_q0(
    dp_synthetic_cache: FeatureCache,
):
    truth_power_factor = 0.94
    cache, signed, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=truth_power_factor, q0=-1.5
    )
    rows = _scored_rows(cache=cache)
    assert (signed[rows] < 0).mean() > 0.03, "the truth must reverse flow often enough to test it"
    meter = _meter(cache=cache, qmode="learned")
    assert meter.reactive == "learned"
    assert meter.reverse_flow_share >= MIN_REVERSE_FLOW_SHARE
    assert meter.fit.capacity_points()[2023] == pytest.approx(sum(REVERSE_FLOW.values()), rel=0.15)
    assert meter.power_factor == pytest.approx(truth_power_factor, abs=0.015)
    assert meter.q0 == pytest.approx(-1.5, abs=1.0)
    predicted = predict_meter(cache=cache, meter=meter, index=rows)
    accuracy = _balanced_accuracy(
        truth_positive=signed[rows] > 0, predicted_positive=predicted.real_power > 0
    )
    assert accuracy > 0.9
    assert np.abs(predicted.apparent_power - cache.sites["S1"].y[rows]).mean() < 0.5
    assert len(meter.starts) == len(CAPACITY_START_MULTIPLIERS) + 1
    assert meter.loss == min(start.loss for start in meter.starts)
    assert meter.converged
    assert np.isfinite(meter.capacity_spread)


def test_a_plant_unlike_the_models_with_reverse_flow_still_gives_the_signed_capacity(
    dp_synthetic_cache: FeatureCache,
):
    cache, signed, signed_series = _cache_with_reading(
        cache=dp_synthetic_cache,
        capacity=REVERSE_FLOW,
        power_factor=0.95,
        q0=-3.0,
        mismatched_plant=True,
    )
    assert (signed[_scored_rows(cache=cache)] < 0).mean() > 0.01
    reference = _signed_capacity(cache=cache, signed_series=signed_series)
    meter = _meter(cache=cache, qmode="learned")
    assert meter.reactive == "learned"
    capacity = meter.fit.capacity_points()[meter.fit.last_season_year]
    assert capacity == pytest.approx(reference, rel=0.10)
    assert meter.power_factor == pytest.approx(0.95, abs=0.02)


def test_without_reverse_flow_the_reactive_pair_is_held_at_the_prior_not_learned(
    dp_synthetic_cache: FeatureCache,
):
    cache, signed, signed_series = _cache_with_reading(
        cache=dp_synthetic_cache,
        capacity=NO_REVERSE_FLOW,
        power_factor=0.97,
        q0=2.0,
        mismatched_plant=True,
    )
    assert (signed[_scored_rows(cache=cache)] < 0).mean() < MIN_REVERSE_FLOW_SHARE
    zero = _meter(cache=cache, qmode="zero")
    learned = _meter(cache=cache, qmode="learned")
    assert learned.reactive == "fixed"
    assert learned.tan_phi == pytest.approx(TAN_PHI_PRIOR_MEAN)
    assert learned.q0 == 0.0
    reference = _signed_capacity(cache=cache, signed_series=signed_series)
    zero_capacity = zero.fit.capacity_points()[zero.fit.last_season_year]
    learned_capacity = learned.fit.capacity_points()[learned.fit.last_season_year]
    assert learned_capacity == pytest.approx(zero_capacity, rel=0.10)
    assert learned_capacity == pytest.approx(reference, rel=0.30)


def test_a_learned_pair_stays_inside_its_bounds_when_the_truth_is_outside_them(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache,
        capacity=REVERSE_FLOW,
        power_factor=0.89,
        q0=-9.0,
    )
    meter = _meter(cache=cache, qmode="learned")
    assert meter.reactive == "learned"
    assert meter.converged
    assert meter.tan_phi <= TAN_PHI_MAX + 1e-9
    assert meter.q0 == pytest.approx(-dp_meter.Q0_MAX_POINTS, abs=0.05)
    assert -dp_meter.Q0_MAX_POINTS - 1e-9 <= meter.q0 <= dp_meter.Q0_MAX_POINTS + 1e-9


def test_a_strong_prior_holds_the_learned_pair_at_its_centre_and_the_default_lets_data_move_it(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=0.94, q0=-1.5
    )
    default = _meter(cache=cache, qmode="learned")
    monkeypatch.setattr(dp_meter, "PRIOR_STRENGTH_ROWS", 1e9)
    pinned = _meter(cache=cache, qmode="learned")
    assert abs(default.tan_phi - TAN_PHI_PRIOR_MEAN) > 0.05
    assert pinned.tan_phi == pytest.approx(TAN_PHI_PRIOR_MEAN, abs=0.005)
    assert pinned.q0 == pytest.approx(0.0, abs=0.05)
    assert pinned.converged
    assert max(start.iterations for start in pinned.starts) < 40


def test_reading_the_meter_as_signed_power_misplaces_capacity_that_the_meter_fit_recovers(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=None
    )
    naive = fit_gam(
        cache=cache, site="S1", specs=[SPEC], cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    meter = _meter(cache=cache, qmode="zero")
    truth = sum(REVERSE_FLOW.values())
    naive_error = abs(naive.capacity_points()[2023] - truth)
    meter_error = abs(meter.fit.capacity_points()[2023] - truth)
    assert meter_error < 0.5 * naive_error


@pytest.mark.parametrize("qmode", ["zero", "learned"])
def test_the_meter_fit_uses_training_rows_only(dp_synthetic_cache: FeatureCache, qmode: QModeType):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=0.95, q0=-1.0
    )
    features = cache.sites["S1"]
    after = cache.end_times > np.datetime64("2024-07-01T00:00", "us")
    garbled = replace(
        cache,
        sites={
            "S1": replace(
                features,
                y=np.where(after, 1e3, features.y),
                pv_unit_shape=np.where(after, 0.0, features.pv_unit_shape),
            )
        },
    )
    first = _meter(cache=cache, qmode=qmode)
    second = _meter(cache=garbled, qmode=qmode)
    assert np.array_equal(first.fit.theta, second.fit.theta)
    assert (first.tan_phi, first.q0) == (second.tan_phi, second.q0)


@pytest.mark.parametrize(
    "spec",
    [
        GamSpec(capacity="pinned"),
        GamSpec(capacity="pinned", clip=ClipSpec()),
        GamSpec(capacity="fitted", drift="d1", node=NodeSpec()),
    ],
)
def test_the_meter_fit_refuses_a_pinned_capacity_a_clip_and_a_node(
    dp_synthetic_cache: FeatureCache, spec: GamSpec
):
    with pytest.raises(ValueError, match="fitted capacity and no clip or node"):
        fit_gam_through_meter(
            cache=dp_synthetic_cache,
            site="S1",
            spec=spec,
            cut=CUT,
            window=WINDOW,
            season_years=SEASON_YEARS,
            qmode="zero",
        )


def test_predictions_follow_the_apparent_power_formula(dp_synthetic_cache: FeatureCache):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity={2020: 12.0, 2022: 12.0}, power_factor=0.97
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


def test_the_dark_row_start_runs_and_every_start_converges(dp_synthetic_cache: FeatureCache):
    features = dp_synthetic_cache.sites["S1"]
    dark = (features.pv_unit_shape < 1e-6) & np.isfinite(features.pv_unit_shape)
    with_dark = replace(dp_synthetic_cache, sites={"S1": replace(features, dark=dark)})
    cache, _, _ = _cache_with_reading(cache=with_dark, capacity=REVERSE_FLOW, power_factor=None)
    meter = _meter(cache=cache, qmode="zero")
    names = [start.name for start in meter.starts]
    assert "dark-row demand, no solar" in names
    assert "naive demand, no solar" not in names
    assert meter.converged
    best = min(meter.starts, key=lambda start: start.loss)
    assert best.final_capacity == pytest.approx(sum(REVERSE_FLOW.values()), rel=0.15)


def test_a_start_that_runs_out_of_iterations_is_flagged(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=None
    )
    monkeypatch.setattr(dp_meter, "MAX_ITERATIONS", 1)
    meter = _meter(cache=cache, qmode="zero")
    assert not meter.converged
    assert all(start.iterations == 1 for start in meter.starts)


def test_ones_weights_reproduce_the_unweighted_fit_and_zero_weights_drop_rows(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=REVERSE_FLOW, power_factor=0.97
    )
    _, index = prepare_fit(
        cache=cache, site="S1", spec=SPEC, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    unweighted = _meter(cache=cache, qmode="learned")
    ones = fit_gam_through_meter(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        qmode="learned",
        weights=np.ones(len(index)),
    )
    assert np.allclose(ones.fit.theta, unweighted.fit.theta, atol=1e-9)
    assert ones.tan_phi == pytest.approx(unweighted.tan_phi)
    early_cut = datetime(2024, 1, 1, tzinfo=UTC)
    fixed_start = TrainingWindow(start=datetime(2023, 1, 1, tzinfo=UTC))
    _, long_index = prepare_fit(
        cache=cache, site="S1", spec=SPEC, cut=CUT, window=fixed_start, season_years=SEASON_YEARS
    )
    keep = (
        cache.end_times[long_index] <= np.datetime64(early_cut.replace(tzinfo=None), "us")
    ).astype(np.float64)
    zeroed = fit_gam_through_meter(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=CUT,
        window=fixed_start,
        season_years=SEASON_YEARS,
        qmode="zero",
        weights=keep,
    )
    shorter = fit_gam_through_meter(
        cache=cache,
        site="S1",
        spec=SPEC,
        cut=early_cut,
        window=fixed_start,
        season_years=SEASON_YEARS,
        qmode="zero",
    )
    assert np.allclose(zeroed.fit.theta, shorter.fit.theta, atol=1e-6)


def test_weights_with_the_wrong_length_or_a_negative_entry_are_refused(
    dp_synthetic_cache: FeatureCache,
):
    for weights in (np.ones(3), -np.ones(26256)):
        with pytest.raises(ValueError, match="one non-negative entry per training row"):
            fit_gam_through_meter(
                cache=dp_synthetic_cache,
                site="S1",
                spec=SPEC,
                cut=CUT,
                window=WINDOW,
                season_years=SEASON_YEARS,
                qmode="zero",
                weights=weights,
            )


def test_the_meter_replica_fits_twelve_months_earlier_and_predicts_only_the_year_after(
    dp_synthetic_cache: FeatureCache,
):
    cache, _, _ = _cache_with_reading(
        cache=dp_synthetic_cache, capacity=NO_REVERSE_FLOW, power_factor=None
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
