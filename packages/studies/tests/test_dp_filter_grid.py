"""Tests of the batched Kalman filter grid (`studies.dp_filter_grid`).

Every test compares the grid with `studies.dp_assimilation.run_kalman` and `correction_at`, the CPU
filter that the grid reproduces, on random streams with missing readings.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import (  # noqa: E402
    CorrectionTrace,
    FilterHyperparameters,
    correction_at,
    run_kalman,
)
from studies.dp_filter_grid import (  # noqa: E402
    GridTrace,
    band_slots,
    correction_at_cutoffs,
    resolve_device,
    run_kalman_grid,
)

N_BANDS = 4
N_ROWS = 600
TOLERANCE = 1e-6
LOG_LIKELIHOOD_RELATIVE_TOLERANCE = 1e-9


def _streams(*, seed: int, n_streams: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    residuals = rng.normal(0.0, 2.0, size=(n_streams, N_ROWS)).cumsum(axis=1) * 0.05
    residuals += rng.normal(0.0, 1.0, size=residuals.shape)
    residuals[rng.random(residuals.shape) < 0.2] = np.nan
    residuals[n_streams - 1, :150] = np.nan  # the last stream starts late
    band = (np.arange(N_ROWS) // 12) % N_BANDS
    return residuals, band


def _run(
    *,
    residuals: np.ndarray,
    band: np.ndarray,
    hyper: np.ndarray,
    device: str = "cpu",
    keep_states: bool = True,
) -> GridTrace:
    return run_kalman_grid(
        residuals=residuals,
        band=band,
        hyper=hyper,
        n_bands=N_BANDS,
        device=device,
        keep_states=keep_states,
    )


def _settings() -> np.ndarray:
    return np.array(
        [
            [np.log(0.05), 4.2, np.log(0.35), 0.0],
            [np.log(0.5), 2.0, np.log(1.5), np.log(0.3)],
            [np.log(0.001), 6.0, np.log(0.1), np.log(3.0)],
        ]
    )


def _per_band_settings() -> np.ndarray:
    """Three settings, each with a different setting in every band, shape (3, 4, 4)."""
    shared = _settings()
    return np.stack(
        [shared[[(setting + band) % 3 for band in range(N_BANDS)]] for setting in range(3)]
    )


def _reference(*, residual: np.ndarray, band: np.ndarray, values: np.ndarray) -> CorrectionTrace:
    return run_kalman(
        residual=residual, band=band, hyper=FilterHyperparameters.from_array(values=values)
    )


def test_log_likelihood_and_states_match_run_kalman() -> None:
    residuals, band = _streams(seed=0)
    settings = _settings()
    trace = _run(residuals=residuals, band=band, hyper=settings)
    for stream in range(len(residuals)):
        for setting_index, values in enumerate(settings):
            expected = _reference(residual=residuals[stream], band=band, values=values)
            assert trace.log_likelihood[stream, setting_index].item() == pytest.approx(
                expected.log_likelihood, rel=LOG_LIKELIHOOD_RELATIVE_TOLERANCE
            )
            for band_index in range(N_BANDS):
                # The grid carries the state through missing slots, so compare at the readings.
                slot = np.searchsorted(
                    trace.slots.positions[band_index], expected.positions[band_index]
                )
                got_level = trace.level[stream, setting_index, band_index, slot].numpy()
                got_anomaly = trace.anomaly[stream, setting_index, band_index, slot].numpy()
                assert np.abs(got_level - expected.level[band_index]).max() < TOLERANCE
                assert np.abs(got_anomaly - expected.anomaly[band_index]).max() < TOLERANCE


def test_per_band_settings_match_one_run_per_band() -> None:
    residuals, band = _streams(seed=1, n_streams=1)
    per_band = _per_band_settings()
    trace = _run(residuals=residuals, band=band, hyper=per_band)
    for setting_index in range(len(per_band)):
        total = 0.0
        for band_index in range(N_BANDS):
            only_band = np.where(band == band_index, residuals[0], np.nan)
            total += _reference(
                residual=only_band, band=band, values=per_band[setting_index, band_index]
            ).log_likelihood
        assert trace.log_likelihood[0, setting_index].item() == pytest.approx(
            total, rel=LOG_LIKELIHOOD_RELATIVE_TOLERANCE
        )


def test_per_band_settings_match_through_correction_at_cutoffs() -> None:
    residuals, band = _streams(seed=8, n_streams=2)
    per_band = _per_band_settings()
    trace = _run(residuals=residuals, band=band, hyper=per_band)
    cutoff, target = _cases(seed=9)
    got = correction_at_cutoffs(
        trace=trace, cutoff=cutoff, target=target, band=band[target]
    ).numpy()
    for stream in range(len(residuals)):
        for setting_index in range(len(per_band)):
            expected = np.zeros(len(cutoff))
            for band_index in range(N_BANDS):
                only_band = np.where(band == band_index, residuals[stream], np.nan)
                trace_band = _reference(
                    residual=only_band, band=band, values=per_band[setting_index, band_index]
                )
                in_band = band[target] == band_index
                expected[in_band] = correction_at(
                    trace=trace_band,
                    cutoff=cutoff[in_band],
                    target=target[in_band],
                    band=band[target][in_band],
                ).total
            assert np.abs(got[stream, setting_index] - expected).max() < TOLERANCE


def _cases(*, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cutoff = rng.integers(-5, N_ROWS, size=200)
    target = np.minimum(cutoff + rng.integers(1, 80, size=200), N_ROWS - 1)
    return cutoff, target


def test_keep_states_false_gives_the_same_log_likelihood() -> None:
    residuals, band = _streams(seed=2)
    full = _run(residuals=residuals, band=band, hyper=_settings())
    light = _run(residuals=residuals, band=band, hyper=_settings(), keep_states=False)
    assert light.level.shape[-1] == 0
    assert torch.equal(full.log_likelihood, light.log_likelihood)


def test_correction_at_cutoffs_matches_correction_at() -> None:
    residuals, band = _streams(seed=3)
    settings = _settings()
    trace = _run(residuals=residuals, band=band, hyper=settings)
    cutoff, target = _cases(seed=7)
    got = correction_at_cutoffs(
        trace=trace, cutoff=cutoff, target=target, band=band[target]
    ).numpy()
    for stream in range(len(residuals)):
        for setting_index, values in enumerate(settings):
            reference = _reference(residual=residuals[stream], band=band, values=values)
            expected = correction_at(
                trace=reference, cutoff=cutoff, target=target, band=band[target]
            ).total
            assert np.abs(got[stream, setting_index] - expected).max() < TOLERANCE


def test_correction_matches_when_the_target_is_before_the_newest_reading() -> None:
    residuals, band = _streams(seed=10, n_streams=1)
    trace = _run(residuals=residuals, band=band, hyper=_settings())
    cutoff = np.full(30, 400)
    target = np.arange(300, 330)
    got = correction_at_cutoffs(trace=trace, cutoff=cutoff, target=target, band=band[target])
    for setting_index, values in enumerate(_settings()):
        reference = _reference(residual=residuals[0], band=band, values=values)
        expected = correction_at(
            trace=reference, cutoff=cutoff, target=target, band=band[target]
        ).total
        assert np.abs(got[0, setting_index].numpy() - expected).max() < TOLERANCE


def test_chunked_lookup_equals_one_chunk() -> None:
    residuals, band = _streams(seed=11)
    trace = _run(residuals=residuals, band=band, hyper=_settings())
    cutoff, target = _cases(seed=12)
    one_chunk = correction_at_cutoffs(trace=trace, cutoff=cutoff, target=target, band=band[target])
    n_streams, n_settings = one_chunk.shape[:2]
    limit = n_streams * n_settings * 8 * 220  # holds the 200-case result, but chunks of 55 cases
    few_chunks = correction_at_cutoffs(
        trace=trace, cutoff=cutoff, target=target, band=band[target], max_bytes=limit
    )
    assert torch.allclose(one_chunk, few_chunks, atol=1e-12, rtol=0.0)


def test_result_larger_than_the_limit_raises() -> None:
    residuals, band = _streams(seed=13)
    trace = _run(residuals=residuals, band=band, hyper=_settings())
    cutoff, target = _cases(seed=14)
    with pytest.raises(ValueError, match="look up fewer cases"):
        correction_at_cutoffs(
            trace=trace, cutoff=cutoff, target=target, band=band[target], max_bytes=1000
        )


def test_states_larger_than_the_limit_raise_and_keep_states_false_still_runs() -> None:
    residuals, band = _streams(seed=15)
    with pytest.raises(ValueError, match="keep_states=False"):
        run_kalman_grid(
            residuals=residuals,
            band=band,
            hyper=_settings(),
            n_bands=N_BANDS,
            device="cpu",
            max_bytes=1000,
        )
    light = run_kalman_grid(
        residuals=residuals,
        band=band,
        hyper=_settings(),
        n_bands=N_BANDS,
        device="cpu",
        keep_states=False,
        max_bytes=1000,
    )
    assert light.log_likelihood.shape == (3, 3)


def test_cut_stream_gives_the_same_correction_as_lookup_on_the_full_stream() -> None:
    residuals, band = _streams(seed=4, n_streams=1)
    settings = _settings()
    cut = 300
    full = _run(residuals=residuals, band=band, hyper=settings)
    truncated = residuals.copy()
    truncated[:, cut + 1 :] = np.nan
    short = _run(residuals=truncated, band=band, hyper=settings)
    target = np.arange(cut + 1, cut + 40)
    cutoffs = np.full(len(target), cut)
    assert torch.allclose(
        correction_at_cutoffs(trace=full, cutoff=cutoffs, target=target, band=band[target]),
        correction_at_cutoffs(trace=short, cutoff=cutoffs, target=target, band=band[target]),
        atol=TOLERANCE,
    )


def test_empty_band_all_nan_stream_and_infinite_residual_match_run_kalman() -> None:
    residuals, _ = _streams(seed=16, n_streams=3)
    band = (np.arange(N_ROWS) // 12) % 3  # band 3 has no rows
    residuals[0, :] = np.nan  # an all-missing stream
    residuals[1, 100] = np.inf  # the reference treats a non-finite residual as missing
    trace = _run(residuals=residuals, band=band, hyper=_settings())
    for stream in range(3):
        for setting_index, values in enumerate(_settings()):
            expected = _reference(residual=residuals[stream], band=band, values=values)
            assert trace.log_likelihood[stream, setting_index].item() == pytest.approx(
                expected.log_likelihood, rel=LOG_LIKELIHOOD_RELATIVE_TOLERANCE
            )
    assert trace.log_likelihood[0].tolist() == [0.0, 0.0, 0.0]


def test_band_slots_pads_the_short_band() -> None:
    band = np.array([0, 1, 0, 0, 1])
    slots = band_slots(band=band, n_bands=3)
    assert slots.n_rows.tolist() == [3, 2, 0]
    assert slots.positions[0].tolist() == [0, 2, 3]
    assert slots.positions[1, :2].tolist() == [1, 4]


def test_bad_settings_shape_raises() -> None:
    residuals, band = _streams(seed=5)
    with pytest.raises(ValueError, match="hyper must have shape"):
        _run(residuals=residuals, band=band, hyper=np.zeros((2, 3)))


def test_residuals_not_matching_the_band_length_raise() -> None:
    residuals, band = _streams(seed=5)
    with pytest.raises(ValueError, match="residuals must have shape"):
        _run(residuals=residuals[:, :-1], band=band, hyper=_settings())


def test_non_finite_setting_raises() -> None:
    residuals, band = _streams(seed=5)
    settings = _settings()
    settings[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(residuals=residuals, band=band, hyper=settings)


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(device="cuda").type == "cpu"
    assert resolve_device().type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_matches_cpu() -> None:
    residuals, band = _streams(seed=6)
    cpu = _run(residuals=residuals, band=band, hyper=_settings())
    gpu = _run(residuals=residuals, band=band, hyper=_settings(), device="cuda")
    assert np.abs(cpu.log_likelihood.numpy() - gpu.log_likelihood.cpu().numpy()).max() < TOLERANCE
    cutoff, target = _cases(seed=6)
    cpu_correction = correction_at_cutoffs(
        trace=cpu, cutoff=cutoff, target=target, band=band[target]
    )
    gpu_correction = correction_at_cutoffs(
        trace=gpu, cutoff=cutoff, target=target, band=band[target]
    )
    assert np.abs(cpu_correction.numpy() - gpu_correction.cpu().numpy()).max() < TOLERANCE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_grid_is_faster_than_looping_run_kalman() -> None:
    """A loose floor on the speed-up; `studies/dp_filter_grid_benchmark` prints the real ratio."""
    n_streams, n_rows, n_settings = 8, 8000, 200
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=(n_streams, n_rows))
    band = (np.arange(n_rows) % 48) // 12
    settings = np.column_stack(
        [
            rng.uniform(-4, 1, n_settings),
            rng.uniform(1, 6, n_settings),
            rng.uniform(-2, 1, n_settings),
            rng.uniform(-1, 1, n_settings),
        ]
    )
    _run(residuals=residuals, band=band, hyper=settings, device="cuda", keep_states=False)
    gpu_seconds = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _run(residuals=residuals, band=band, hyper=settings, device="cuda", keep_states=False)
        torch.cuda.synchronize()
        gpu_seconds.append(time.perf_counter() - start)
    start = time.perf_counter()
    for values in settings[:3]:
        _reference(residual=residuals[0], band=band, values=values)
    cpu_seconds = (time.perf_counter() - start) / 3 * n_streams * n_settings
    assert cpu_seconds / min(gpu_seconds) > 3.0
