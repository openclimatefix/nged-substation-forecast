"""Tests of the recent-level assimilation (`studies.dp_assimilation`).

The filter is checked against a dense NumPy Kalman filter that steps through every half-hour, the
information cut by changing every residual after the cut, and the hyperparameter fit by the
likelihood at the truth of a simulated stream.
"""

from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from studies.dp_assimilation import (  # noqa: E402
    N_LEAD_BANDS,
    CorrectionTrace,
    FilterHyperparameters,
    LeadSpread,
    SharedTerm,
    correction_at,
    day_ahead_initialisation,
    fit_filter_hyperparameters,
    fit_site_hyperparameters,
    lead_band_of_days_ahead,
    post_filter_lead_spread,
    replica_corrections,
    run_ewma,
    run_kalman,
    site_log_likelihood,
    stitch_out_of_sample_residuals,
    summed_correction_at,
    usable_cutoff_positions,
)
from studies.dp_gam import ReplicaResiduals, ResidualSpread  # noqa: E402

HYPER: Final[FilterHyperparameters] = FilterHyperparameters(
    log_q=np.log(0.4), logit_phi=3.5, log_sigma_z=np.log(0.8), log_sigma_eps=np.log(0.7)
)
N_HALF_HOURS: Final[int] = 48 * 40


def _bands(*, n: int) -> np.ndarray:
    """Hour bands that run for 12 half-hours at a time, like the real bands."""
    return (np.arange(n) // 12) % 4


def _simulate(*, hyper: FilterHyperparameters, n: int, seed: int) -> np.ndarray:
    """Simulate the model: a random-walk level plus an AR(1) transient plus noise, per band."""
    generator = np.random.default_rng(seed)
    band = _bands(n=n)
    residual = np.full(n, np.nan)
    stationary = hyper.sigma_z / np.sqrt(1.0 - hyper.phi**2)
    for b in range(4):
        level, anomaly = 0.0, stationary * generator.normal()
        previous = None
        for position in np.flatnonzero(band == b):
            gap = 0 if previous is None else position - previous
            previous = position
            level += np.sqrt(hyper.q / 48.0 * gap) * generator.normal()
            decay = hyper.phi**gap
            anomaly = decay * anomaly + stationary * np.sqrt(1 - decay**2) * generator.normal()
            residual[position] = level + anomaly + hyper.sigma_eps * generator.normal()
    return residual


def _dense_reference(
    *, values: np.ndarray, hyper: FilterHyperparameters
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a dense two-state Kalman filter over one band's half-hours, stepping every half-hour.

    `values` has a `NaN` where the band has no reading. Returns the level and transient after each
    reading and the log likelihood.
    """
    stationary = hyper.sigma_z**2 / (1.0 - hyper.phi**2)
    transition = np.diag([1.0, hyper.phi])
    process = np.diag([hyper.q / 48.0, hyper.sigma_z**2])
    state = np.zeros(2)
    covariance = np.diag([100.0, stationary])
    observe = np.array([1.0, 1.0])
    levels, anomalies, log_likelihood, started = [], [], 0.0, False
    for value in values:
        if started:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        if np.isnan(value):
            continue
        started = True
        variance = observe @ covariance @ observe + hyper.sigma_eps**2
        innovation = value - observe @ state
        log_likelihood -= 0.5 * (np.log(2 * np.pi * variance) + innovation**2 / variance)
        gain = covariance @ observe / variance
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, gain) * variance
        levels.append(state[0])
        anomalies.append(state[1])
    return np.array(levels), np.array(anomalies), log_likelihood


def test_the_filter_matches_a_dense_numpy_kalman_filter():
    residual = _simulate(hyper=HYPER, n=N_HALF_HOURS, seed=1)
    residual[np.random.default_rng(2).random(N_HALF_HOURS) < 0.1] = np.nan
    band = _bands(n=N_HALF_HOURS)
    trace = run_kalman(residual=residual, band=band, hyper=HYPER)
    total = 0.0
    for b in range(4):
        first = np.flatnonzero((band == b) & np.isfinite(residual))[0]
        window = np.where(band == b, residual, np.nan)[first:]
        levels, anomalies, log_likelihood = _dense_reference(values=window, hyper=HYPER)
        assert np.allclose(trace.level[b], levels, atol=1e-6)
        assert np.allclose(trace.anomaly[b], anomalies, atol=1e-6)
        total += log_likelihood
    assert trace.log_likelihood == pytest.approx(total, abs=1e-6)
    assert site_log_likelihood(residual=residual, band=band, hyper=HYPER) == pytest.approx(total)


@pytest.mark.parametrize("kind", ["kalman", "ewma"])
def test_changing_any_residual_after_the_cut_leaves_the_forecast_unchanged(kind: str):
    residual = _simulate(hyper=HYPER, n=N_HALF_HOURS, seed=3)
    band = _bands(n=N_HALF_HOURS)
    cutoff_position = N_HALF_HOURS // 2
    cases = np.arange(cutoff_position + 1, cutoff_position + 300)
    cutoff = np.full(len(cases), cutoff_position)

    def forecast(stream: np.ndarray) -> np.ndarray:
        trace = (
            run_kalman(residual=stream, band=band, hyper=HYPER)
            if kind == "kalman"
            else run_ewma(residual=stream, band=band)
        )
        return correction_at(trace=trace, cutoff=cutoff, target=cases, band=band[cases]).total

    altered = residual.copy()
    altered[cutoff_position + 1 :] = np.random.default_rng(4).normal(
        50.0, 20.0, size=N_HALF_HOURS - cutoff_position - 1
    )
    assert np.array_equal(forecast(residual), forecast(altered))
    altered[cutoff_position] += 1.0
    assert not np.array_equal(forecast(residual), forecast(altered))


def test_the_level_follows_a_permanent_step_and_the_transient_releases_a_pulse():
    n = 48 * 60
    band = _bands(n=n)
    step_at = 48 * 20
    stream = np.where(np.arange(n) >= step_at, 3.0, 0.0)
    hyper = FilterHyperparameters(
        log_q=np.log(0.5), logit_phi=3.5, log_sigma_z=np.log(0.3), log_sigma_eps=np.log(0.3)
    )
    trace = run_kalman(residual=stream, band=band, hyper=hyper)
    late = trace.level[1][-1]
    assert late == pytest.approx(3.0, abs=0.3)
    assert abs(trace.anomaly[1][-1]) < 0.3
    pulse = np.zeros(n)
    pulse[(np.arange(n) >= 48 * 20) & (np.arange(n) < 48 * 20 + 6)] = 4.0
    trace = run_kalman(residual=pulse, band=band, hyper=hyper)
    during = trace.anomaly[0][(trace.positions[0] >= 48 * 20) & (trace.positions[0] < 48 * 20 + 6)]
    assert np.abs(during).max() > 1.0  # the filter reacts to the pulse
    assert abs(trace.level[1][-1]) < 0.3
    assert abs(trace.anomaly[1][-1]) < 0.3


def test_the_transient_decays_by_phi_per_half_hour_from_the_newest_reading():
    trace = CorrectionTrace(
        kind="kalman",
        positions=(
            np.array([10, 20]),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        ),
        level=(np.array([1.0, 2.0]), np.array([]), np.array([]), np.array([])),
        anomaly=(np.array([5.0, 4.0]), np.array([]), np.array([]), np.array([])),
        phi=0.9,
        log_likelihood=0.0,
    )
    result = correction_at(
        trace=trace,
        cutoff=np.array([25, 15, 5]),
        target=np.array([30, 30, 30]),
        band=np.zeros(3, dtype=int),
    )
    assert result.level.tolist() == [2.0, 1.0, 0.0]
    assert result.anomaly == pytest.approx([4.0 * 0.9**10, 5.0 * 0.9**20, 0.0])
    assert result.total == pytest.approx(result.level + result.anomaly)


def test_the_ewma_of_a_constant_is_that_constant_and_weights_fall_by_half_each_half_life():
    n = 48 * 60
    band = np.zeros(n, dtype=int)
    constant = run_ewma(residual=np.full(n, 2.5), band=band)
    assert constant.level[0] == pytest.approx(np.full(n, 2.5))
    two = np.full(n, np.nan)
    two[0], two[48 * 14] = 0.0, 1.0
    trace = run_ewma(residual=two, band=band, half_life_days=14.0)
    assert trace.level[0][-1] == pytest.approx(1.0 / 1.5)
    assert np.all(trace.anomaly[0] == 0.0)


def test_the_day_ahead_initialisation_and_cut_follow_the_leads_of_experiment_104():
    end_times = np.arange(
        np.datetime64("2025-06-29T00:30", "us"),
        np.datetime64("2025-07-03T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    valid = np.array(
        [np.datetime64("2025-07-01T00:30", "us"), np.datetime64("2025-07-02T00:00", "us")]
    )
    init = day_ahead_initialisation(end_times=valid, days_ahead=1)
    assert init.tolist() == [np.datetime64("2025-06-30T09:00", "us")] * 2
    leads = (valid - init) / np.timedelta64(1, "h")
    assert leads.tolist() == [15.5, 39.0]
    cutoff = usable_cutoff_positions(end_times=end_times, initialisation=init, delay_hours=3.5)
    assert end_times[cutoff[0]] == np.datetime64("2025-06-30T05:30", "us")
    later = usable_cutoff_positions(end_times=end_times, initialisation=init, delay_hours=24.0)
    assert end_times[later[0]] == np.datetime64("2025-06-29T09:00", "us")
    nothing = usable_cutoff_positions(
        end_times=end_times, initialisation=init[:1], delay_hours=1000.0
    )
    assert nothing[0] == -1


def test_the_lead_bands_cover_days_one_to_seven_and_refuse_others():
    assert [lead_band_of_days_ahead(days_ahead=d) for d in range(1, 8)] == [0, 1, 1, 2, 2, 2, 2]
    for bad in (0, 8):
        with pytest.raises(ValueError, match="between 1 and 7"):
            lead_band_of_days_ahead(days_ahead=bad)


def test_the_true_hyperparameters_beat_wrong_ones_and_the_fit_finds_them_roughly():
    n = 48 * 120
    residual = _simulate(hyper=HYPER, n=n, seed=5)
    band = _bands(n=n)
    truth = site_log_likelihood(residual=residual, band=band, hyper=HYPER)
    for wrong in (
        replace(HYPER, log_q=HYPER.log_q + 3.0),
        replace(HYPER, logit_phi=HYPER.logit_phi - 3.0),
        replace(HYPER, log_sigma_eps=HYPER.log_sigma_eps + 2.0),
    ):
        assert site_log_likelihood(residual=residual, band=band, hyper=wrong) < truth - 10.0
    flat = np.array([0.0, 0.0, 0.0, 0.0])
    fitted = fit_site_hyperparameters(
        residual=residual,
        band=band,
        prior_mean=flat,
        prior_spread=np.full(4, 100.0),
        start=HYPER.as_array() - 0.5,
    )
    assert site_log_likelihood(residual=residual, band=band, hyper=fitted) >= truth - 1.0
    assert fitted.sigma_eps == pytest.approx(HYPER.sigma_eps, rel=0.5)


def test_the_shared_prior_fit_returns_one_set_per_site_and_refuses_no_sites():
    n = 48 * 40
    band = _bands(n=n)
    streams = {"S1": _simulate(hyper=HYPER, n=n, seed=6), "S2": _simulate(hyper=HYPER, n=n, seed=7)}
    fitted = fit_filter_hyperparameters(residuals=streams, band=band)
    assert set(fitted) == {"S1", "S2"}
    assert all(0.0 < h.phi < 1.0 for h in fitted.values())
    with pytest.raises(ValueError, match="no residual streams"):
        fit_filter_hyperparameters(residuals={}, band=band)


def test_the_lead_spread_is_centred_per_lead_band_and_needs_one_array_per_band():
    generator = np.random.default_rng(8)
    n = 4000
    arrays = [generator.normal(loc=2.0 * i, scale=1.0 + i, size=n) for i in range(N_LEAD_BANDS)]
    band = [generator.integers(0, 4, size=n) for _ in range(N_LEAD_BANDS)]
    quarter = [generator.integers(0, 4, size=n) for _ in range(N_LEAD_BANDS)]
    spread = LeadSpread.fit(residuals=arrays, band=band, quarter=quarter)
    quantiles = spread.quantiles(
        mean=np.zeros(5), band=np.zeros(5, dtype=int), quarter=np.zeros(5, dtype=int), lead_band=2
    )
    assert quantiles.shape == (5, 11)
    assert np.allclose(quantiles[:, 5], 0.0)
    assert np.all(np.diff(quantiles, axis=1) >= 0)
    wide = spread.quantiles(
        mean=np.zeros(1), band=np.zeros(1, dtype=int), quarter=np.zeros(1, dtype=int), lead_band=2
    )
    narrow = spread.quantiles(
        mean=np.zeros(1), band=np.zeros(1, dtype=int), quarter=np.zeros(1, dtype=int), lead_band=0
    )
    assert wide[0, -1] - wide[0, 0] > narrow[0, -1] - narrow[0, 0]
    with pytest.raises(ValueError, match="one array per lead band"):
        LeadSpread.fit(residuals=arrays[:2], band=band[:2], quarter=quarter[:2])


def test_a_drifting_level_widens_the_raw_spread_but_not_the_post_filter_spread():
    end_times = np.arange(
        np.datetime64("2024-01-01T00:30", "us"),
        np.datetime64("2025-01-01T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    n = len(end_times)
    generator = np.random.default_rng(9)
    drift = np.linspace(0.0, 40.0, n)
    residual = drift + 0.3 * generator.normal(size=n)
    band = _bands(n=n)
    quarter = ((end_times.astype("datetime64[M]").astype(int) % 12) // 3).astype(int)
    replica = ReplicaResiduals(
        residuals=residual,
        band=band,
        quarter=quarter,
        index=np.arange(n),
        fit_cut=datetime(2024, 1, 1, tzinfo=UTC),
        switching=np.zeros(n, dtype=bool),
    )
    hyper = FilterHyperparameters(
        log_q=np.log(0.5), logit_phi=3.5, log_sigma_z=np.log(0.3), log_sigma_eps=np.log(0.5)
    )
    spread = post_filter_lead_spread(
        end_times=end_times, replica=replica, run=partial(run_kalman, hyper=hyper)
    )
    raw = ResidualSpread.fit(residuals=residual, band=band, quarter=quarter, centre=True)
    post = spread.by_lead_band[0].table[0, 0]
    assert post[-1] - post[0] < 0.4 * (raw.table[0, 0, -1] - raw.table[0, 0, 0])
    far = spread.by_lead_band[2].table[0, 0]
    assert far[-1] - far[0] > post[-1] - post[0] + 0.05


def test_a_flagged_switching_row_stays_in_the_stream_but_leaves_the_spread():
    end_times = np.arange(
        np.datetime64("2024-01-01T00:30", "us"),
        np.datetime64("2024-04-01T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    n = len(end_times)
    residual = np.zeros(n)
    switching = np.zeros(n, dtype=bool)
    switching[48 * 40 : 48 * 45] = True
    residual[switching] = 50.0
    replica = ReplicaResiduals(
        residuals=residual,
        band=_bands(n=n),
        quarter=np.zeros(n, dtype=int),
        index=np.arange(n),
        fit_cut=datetime(2024, 1, 1, tzinfo=UTC),
        switching=switching,
    )
    streams: list[np.ndarray] = []

    def run(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        streams.append(residual.copy())
        return run_ewma(residual=residual, band=band)

    spread = post_filter_lead_spread(end_times=end_times, replica=replica, run=run)
    assert streams[0][switching].min() == 50.0
    assert spread.by_lead_band[0].table.max() < 5.0


def test_a_leak_of_the_target_reading_into_the_spread_would_be_caught():
    # White noise has nothing in its past that predicts the next reading, so no causal correction
    # can shrink the spread below the noise. A filter that saw the target would track it.
    end_times = np.arange(
        np.datetime64("2024-01-01T00:30", "us"),
        np.datetime64("2024-07-01T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    n = len(end_times)
    residual = np.random.default_rng(10).normal(0.0, 1.0, size=n)
    replica = ReplicaResiduals(
        residuals=residual,
        band=_bands(n=n),
        quarter=np.zeros(n, dtype=int),
        index=np.arange(n),
        fit_cut=datetime(2024, 1, 1, tzinfo=UTC),
        switching=np.zeros(n, dtype=bool),
    )
    tight = FilterHyperparameters(
        log_q=np.log(5.0), logit_phi=3.5, log_sigma_z=np.log(1.0), log_sigma_eps=np.log(0.06)
    )
    spread = post_filter_lead_spread(
        end_times=end_times, replica=replica, run=partial(run_kalman, hyper=tight)
    )
    for lead_band in range(N_LEAD_BANDS):
        table = spread.by_lead_band[lead_band].table[0, 0]
        assert table[-1] - table[0] > 0.9 * (4.65 * 1.0) * 0.8


def test_the_stitched_stream_takes_each_fits_residuals_only_after_its_own_cut():
    n = 100
    first = np.full(n, 1.0)
    second = np.full(n, 2.0)
    third = np.full(n, 3.0)
    stream = stitch_out_of_sample_residuals(
        cut_positions=[20, 50, 80], residuals=[first, second, third]
    )
    assert np.isnan(stream[:21]).all()
    assert (stream[21:51] == 1.0).all()
    assert (stream[51:81] == 2.0).all()
    assert (stream[81:] == 3.0).all()
    # Changing a fit's in-sample residuals, at or before its own cut, changes nothing.
    in_sample = second.copy()
    in_sample[:51] = -99.0
    again = stitch_out_of_sample_residuals(
        cut_positions=[20, 50, 80], residuals=[first, in_sample, third]
    )
    assert np.array_equal(stream, again, equal_nan=True)


def test_stitching_refuses_unordered_cuts_missing_fits_and_mismatched_grids():
    with pytest.raises(ValueError, match="strictly ascending"):
        stitch_out_of_sample_residuals(cut_positions=[50, 20], residuals=[np.zeros(9), np.zeros(9)])
    with pytest.raises(ValueError, match="one residual array per cut"):
        stitch_out_of_sample_residuals(cut_positions=[5], residuals=[])
    with pytest.raises(ValueError, match="same grid"):
        stitch_out_of_sample_residuals(cut_positions=[2, 4], residuals=[np.zeros(9), np.zeros(8)])


def test_a_hyperparameter_fit_that_does_not_converge_raises(monkeypatch: pytest.MonkeyPatch):
    class Failed:
        success = False
        message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        x = np.zeros(4)

    monkeypatch.setattr("studies.dp_assimilation.minimize", lambda *args, **kwargs: Failed())
    with pytest.raises(ValueError, match="did not converge"):
        fit_site_hyperparameters(
            residual=np.zeros(48),
            band=np.zeros(48, dtype=int),
            prior_mean=np.zeros(4),
            prior_spread=np.ones(4),
            start=np.zeros(4),
        )


def test_a_residual_outside_the_fitting_window_is_refused():
    n = 48 * 10
    end_times = np.arange(
        np.datetime64("2024-01-01T00:30", "us"),
        np.datetime64("2024-01-11T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    stream = np.full(n, np.nan)
    stream[:48] = 0.1
    stream[-1] = 0.2
    with pytest.raises(ValueError, match="outside the fitting window"):
        fit_filter_hyperparameters(
            residuals={"S1": stream},
            band=_bands(n=n),
            end_times=end_times,
            window=(np.datetime64("2024-01-01", "us"), np.datetime64("2024-01-05", "us")),
        )


def test_the_variance_floor_is_applied_after_each_update(monkeypatch: pytest.MonkeyPatch):
    stream = _simulate(hyper=HYPER, n=48 * 20, seed=12)
    band = _bands(n=len(stream))
    normal = run_kalman(residual=stream, band=band, hyper=HYPER)
    monkeypatch.setattr("studies.dp_assimilation.VARIANCE_FLOOR", -1e9)
    unfloored = run_kalman(residual=stream, band=band, hyper=HYPER)
    assert np.allclose(normal.level[1], unfloored.level[1])  # the default floor never binds here
    monkeypatch.setattr("studies.dp_assimilation.VARIANCE_FLOOR", 1e3)
    floored = run_kalman(residual=stream, band=band, hyper=HYPER)
    assert np.abs(floored.level[1] - normal.level[1]).max() > 1e-3


def test_a_tiny_observation_noise_keeps_the_filter_finite():
    stream = _simulate(hyper=HYPER, n=48 * 20, seed=12)
    tiny = replace(HYPER, log_sigma_eps=np.log(1e-6))
    trace = run_kalman(residual=stream, band=_bands(n=len(stream)), hyper=tiny)
    assert np.isfinite(trace.log_likelihood)


def test_a_fitting_window_without_the_grid_is_refused():
    with pytest.raises(ValueError, match="needs the grid's end_times"):
        fit_filter_hyperparameters(
            residuals={"S1": np.zeros(48)},
            band=np.zeros(48, dtype=int),
            window=(np.datetime64("2024-01-01", "us"), np.datetime64("2024-01-05", "us")),
        )


def _summed_replica() -> tuple[np.ndarray, ReplicaResiduals, np.ndarray, np.ndarray]:
    """Return a 60-day grid, its replica, a local stream, and a shared stream."""
    end_times = np.arange(
        np.datetime64("2024-01-01T00:30", "us"),
        np.datetime64("2024-03-01T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    n = len(end_times)
    generator = np.random.default_rng(21)
    residual = _simulate(hyper=HYPER, n=n, seed=22) + generator.normal(0.0, 0.2, size=n)
    shared = _simulate(hyper=HYPER, n=n, seed=23)
    replica = ReplicaResiduals(
        residuals=residual,
        band=_bands(n=n),
        quarter=np.zeros(n, dtype=int),
        index=np.arange(n),
        fit_cut=datetime(2024, 1, 1, tzinfo=UTC),
        switching=np.zeros(n, dtype=bool),
    )
    return end_times, replica, residual - 0.5 * shared, shared


def _shared_term(*, stream: np.ndarray, weights: tuple[float, ...]) -> SharedTerm:
    return SharedTerm(
        residual=stream, run=partial(run_kalman, hyper=HYPER), weight_by_band=np.array(weights)
    )


def test_a_summed_correction_is_the_local_plus_the_weighted_shared_correction():
    end_times, replica, local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    grid_band = _bands(n=len(end_times))
    weights = (0.5, 1.0, 1.5, 0.0)
    summed = replica_corrections(
        end_times=end_times,
        replica=replica,
        run=run,
        grid_band=grid_band,
        local_residual=local,
        shared=[_shared_term(stream=shared, weights=weights)],
    )
    local_only = replica_corrections(
        end_times=end_times, replica=replica, run=run, grid_band=grid_band, local_residual=local
    )
    shared_only = replica_corrections(
        end_times=end_times, replica=replica, run=run, grid_band=grid_band, local_residual=shared
    )
    for total, local_part, shared_part in zip(summed, local_only, shared_only, strict=True):
        expected = local_part + np.array(weights)[replica.band] * shared_part
        assert np.allclose(total, expected, atol=1e-12)


def test_the_helper_that_sums_a_forecast_correction_matches_the_replica_sum():
    end_times, replica, local, shared = _summed_replica()
    grid_band = _bands(n=len(end_times))
    weights = np.array([0.4, 0.8, 1.2, 1.6])
    run = partial(run_kalman, hyper=HYPER)
    replica_total = replica_corrections(
        end_times=end_times,
        replica=replica,
        run=run,
        grid_band=grid_band,
        local_residual=local,
        shared=[_shared_term(stream=shared, weights=tuple(weights))],
    )[0]
    cutoff = usable_cutoff_positions(
        end_times=end_times,
        initialisation=day_ahead_initialisation(end_times=end_times, days_ahead=1),
        delay_hours=3.5,
    )
    helper = summed_correction_at(
        local_trace=run(residual=local, band=grid_band),
        shared_traces=[(run(residual=shared, band=grid_band), weights)],
        cutoff=cutoff,
        target=replica.index,
        band=replica.band,
    )
    assert np.array_equal(helper.total, replica_total)
    assert np.allclose(helper.total, helper.level + helper.anomaly)


def _before_the_shared_term_existed(
    *, end_times: np.ndarray, replica: ReplicaResiduals, run: partial
) -> LeadSpread:
    """Return the spread as `post_filter_lead_spread` built it before shared terms existed."""
    stream = np.full(len(end_times), np.nan)
    stream[replica.index] = replica.residuals
    grid_band = np.zeros(len(end_times), dtype=np.int64)
    grid_band[replica.index] = replica.band
    trace = run(residual=stream, band=grid_band)
    valid_times = end_times[replica.index]
    kept = (valid_times > valid_times.min() + np.timedelta64(14, "D")) & ~replica.switching
    residuals: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    bands: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    quarters: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    for days_ahead in range(1, 8):
        cutoff = usable_cutoff_positions(
            end_times=end_times,
            initialisation=day_ahead_initialisation(end_times=valid_times, days_ahead=days_ahead),
            delay_hours=3.5,
        )
        correction = correction_at(
            trace=trace, cutoff=cutoff, target=replica.index, band=replica.band
        )
        lead_band = lead_band_of_days_ahead(days_ahead=days_ahead)
        residuals[lead_band].append((replica.residuals - correction.total)[kept])
        bands[lead_band].append(replica.band[kept])
        quarters[lead_band].append(replica.quarter[kept])
    return LeadSpread.fit(
        residuals=[np.concatenate(r) for r in residuals],
        band=[np.concatenate(b) for b in bands],
        quarter=[np.concatenate(q) for q in quarters],
    )


def test_no_shared_term_and_a_zero_weight_both_match_the_spread_from_before_shared_terms():
    end_times, replica, _local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    before = _before_the_shared_term_existed(end_times=end_times, replica=replica, run=run)
    plain = post_filter_lead_spread(end_times=end_times, replica=replica, run=run)
    zero = post_filter_lead_spread(
        end_times=end_times,
        replica=replica,
        run=run,
        grid_band=_bands(n=len(end_times)),
        shared=[_shared_term(stream=shared, weights=(0.0, 0.0, 0.0, 0.0))],
    )
    for left, middle, right in zip(
        before.by_lead_band, plain.by_lead_band, zero.by_lead_band, strict=True
    ):
        assert np.array_equal(left.table, middle.table)
        assert np.array_equal(left.table, right.table)


def test_readings_outside_the_replica_rows_go_to_their_own_band_not_band_zero():
    end_times, replica, _local, _shared = _summed_replica()
    n = len(end_times)
    grid_band = _bands(n=n)
    # A replica that covers only the second half of the grid, and a local stream that also
    # holds the first half, as a stitched stream with a warm-up would.
    second_half = np.arange(n // 2, n)
    half = replace(
        replica,
        residuals=replica.residuals[second_half],
        band=replica.band[second_half],
        quarter=replica.quarter[second_half],
        index=second_half,
        switching=replica.switching[second_half],
    )
    stream = np.where(grid_band == 1, 5.0, 0.0)
    run = partial(run_kalman, hyper=HYPER)
    corrections = replica_corrections(
        end_times=end_times, replica=half, run=run, grid_band=grid_band, local_residual=stream
    )
    in_band_one = half.band == 1
    in_band_two = half.band == 2
    assert np.mean(corrections[0][in_band_one]) > 4.0
    assert abs(np.mean(corrections[0][in_band_two])) < 0.5


def test_a_local_stream_or_shared_term_without_grid_bands_is_refused():
    end_times, replica, local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    with pytest.raises(ValueError, match="needs grid_band"):
        replica_corrections(end_times=end_times, replica=replica, run=run, local_residual=local)
    with pytest.raises(ValueError, match="needs grid_band"):
        replica_corrections(
            end_times=end_times,
            replica=replica,
            run=run,
            shared=[_shared_term(stream=shared, weights=(1.0, 1.0, 1.0, 1.0))],
        )


def test_a_stream_that_does_not_cover_the_grid_and_a_wrong_weight_count_are_refused():
    end_times, replica, local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    grid_band = _bands(n=len(end_times))
    with pytest.raises(ValueError, match="one entry per grid row"):
        replica_corrections(
            end_times=end_times,
            replica=replica,
            run=run,
            grid_band=grid_band,
            local_residual=local[:-1],
        )
    with pytest.raises(ValueError, match="one entry per grid row"):
        replica_corrections(
            end_times=end_times,
            replica=replica,
            run=run,
            grid_band=grid_band[:-1],
            local_residual=local,
        )
    with pytest.raises(ValueError, match="one per hour band"):
        _shared_term(stream=shared, weights=(1.0, 1.0))


@pytest.mark.parametrize("which", ["local", "shared", "both"])
def test_changing_a_local_or_shared_residual_after_each_cases_cutoff_changes_nothing(which: str):
    end_times, replica, local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    grid_band = _bands(n=len(end_times))
    weights = (0.7, 0.7, 0.7, 0.7)

    def corrections(
        *, local_stream: np.ndarray, shared_stream: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        return replica_corrections(
            end_times=end_times,
            replica=replica,
            run=run,
            grid_band=grid_band,
            local_residual=local_stream,
            shared=[_shared_term(stream=shared_stream, weights=weights)],
        )

    base = corrections(local_stream=local, shared_stream=shared)
    # Position 11 of a day is one half-hour after the 05:30 UTC cutoff of a case whose target is
    # on the next UTC day, so a look-ahead of a half-hour or more reads the change.
    after = 48 * 30 + 11
    local_changed, shared_changed = local.copy(), shared.copy()
    if which in {"local", "both"}:
        local_changed[after:] += 25.0
    if which in {"shared", "both"}:
        shared_changed[after:] += 25.0
    changed = corrections(local_stream=local_changed, shared_stream=shared_changed)
    valid_times = end_times[replica.index]
    for days_ahead, (left, right) in enumerate(zip(base, changed, strict=True), start=1):
        cutoff = usable_cutoff_positions(
            end_times=end_times,
            initialisation=day_ahead_initialisation(end_times=valid_times, days_ahead=days_ahead),
            delay_hours=3.5,
        )
        before_the_change = cutoff < after
        assert before_the_change.any()
        assert np.array_equal(left[before_the_change], right[before_the_change])
        assert not np.allclose(left[~before_the_change], right[~before_the_change])


def test_the_shared_term_changes_the_spread_it_is_summed_into():
    end_times, replica, local, shared = _summed_replica()
    run = partial(run_kalman, hyper=HYPER)
    grid_band = _bands(n=len(end_times))
    local_only = post_filter_lead_spread(
        end_times=end_times, replica=replica, run=run, grid_band=grid_band, local_residual=local
    )
    summed = post_filter_lead_spread(
        end_times=end_times,
        replica=replica,
        run=run,
        grid_band=grid_band,
        local_residual=local,
        shared=[_shared_term(stream=shared, weights=(0.5, 0.5, 0.5, 0.5))],
    )
    narrow = local_only.by_lead_band[0].table[0, 0]
    with_shared = summed.by_lead_band[0].table[0, 0]
    assert (with_shared[-1] - with_shared[0]) < (narrow[-1] - narrow[0])
