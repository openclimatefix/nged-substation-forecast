"""Tests of the batched Kalman filter grid (`studies.dp_filter_grid`).

Every test compares the grid with `studies.dp_assimilation.run_kalman` and `correction_at`, the CPU
filter that the grid reproduces, on random streams with missing readings.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import (  # noqa: E402
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


def test_log_likelihood_and_states_match_run_kalman() -> None:
    residuals, band = _streams(seed=0)
    settings = _settings()
    trace = run_kalman_grid(
        residuals=residuals, band=band, hyper=settings, n_bands=N_BANDS, device="cpu"
    )
    for s in range(len(residuals)):
        for h, values in enumerate(settings):
            expected = run_kalman(
                residual=residuals[s],
                band=band,
                hyper=FilterHyperparameters.from_array(values=values),
            )
            assert trace.log_likelihood[s, h].item() == pytest.approx(
                expected.log_likelihood, abs=TOLERANCE, rel=TOLERANCE
            )
            for b in range(N_BANDS):
                n = len(expected.positions[b])
                # The grid carries the state through missing slots, so compare at the readings.
                slot = np.searchsorted(trace.slots.positions[b], expected.positions[b])
                got_level = trace.level[s, h, b, slot].numpy()
                got_anomaly = trace.anomaly[s, h, b, slot].numpy()
                assert n == 0 or np.abs(got_level - expected.level[b]).max() < TOLERANCE
                assert n == 0 or np.abs(got_anomaly - expected.anomaly[b]).max() < TOLERANCE


def test_per_band_settings_match_one_run_per_band() -> None:
    residuals, band = _streams(seed=1, n_streams=1)
    settings = _settings()
    per_band = np.stack(
        [settings[[b % 3, (b + 1) % 3, (b + 2) % 3, b % 3]] for b in range(N_BANDS)]
    )
    per_band = per_band.transpose(1, 0, 2)[:3]  # (H=3, B=4, 4)
    trace = run_kalman_grid(
        residuals=residuals, band=band, hyper=per_band, n_bands=N_BANDS, device="cpu"
    )
    for h in range(3):
        total = 0.0
        for b in range(N_BANDS):
            only_band = np.where(band == b, residuals[0], np.nan)
            reference = run_kalman(
                residual=only_band,
                band=band,
                hyper=FilterHyperparameters.from_array(values=per_band[h, b]),
            )
            total += reference.log_likelihood
        assert trace.log_likelihood[0, h].item() == pytest.approx(
            total, abs=TOLERANCE, rel=TOLERANCE
        )


def test_keep_states_false_gives_the_same_log_likelihood() -> None:
    residuals, band = _streams(seed=2)
    full = _run(residuals=residuals, band=band, hyper=_settings())
    light = _run(residuals=residuals, band=band, hyper=_settings(), keep_states=False)
    assert light.level.shape[-1] == 0
    assert torch.equal(full.log_likelihood, light.log_likelihood)


def test_correction_at_cutoffs_matches_correction_at() -> None:
    residuals, band = _streams(seed=3)
    settings = _settings()
    trace = run_kalman_grid(
        residuals=residuals, band=band, hyper=settings, n_bands=N_BANDS, device="cpu"
    )
    rng = np.random.default_rng(7)
    cutoff = rng.integers(-5, N_ROWS, size=200)
    target = cutoff + rng.integers(1, 80, size=200)
    target = np.minimum(target, N_ROWS - 1)
    case_band = band[target]
    got = correction_at_cutoffs(trace=trace, cutoff=cutoff, target=target, band=case_band).numpy()
    for s in range(len(residuals)):
        for h, values in enumerate(settings):
            reference = run_kalman(
                residual=residuals[s],
                band=band,
                hyper=FilterHyperparameters.from_array(values=values),
            )
            expected = correction_at(
                trace=reference, cutoff=cutoff, target=target, band=case_band
            ).total
            assert np.abs(got[s, h] - expected).max() < TOLERANCE


def test_cut_stream_gives_the_same_correction_as_lookup_on_the_full_stream() -> None:
    residuals, band = _streams(seed=4, n_streams=1)
    settings = _settings()
    cut = 300
    full = run_kalman_grid(
        residuals=residuals, band=band, hyper=settings, n_bands=N_BANDS, device="cpu"
    )
    truncated = residuals.copy()
    truncated[:, cut + 1 :] = np.nan
    short = run_kalman_grid(
        residuals=truncated, band=band, hyper=settings, n_bands=N_BANDS, device="cpu"
    )
    target = np.arange(cut + 1, cut + 40)
    cutoffs = np.full(len(target), cut)
    assert torch.allclose(
        correction_at_cutoffs(trace=full, cutoff=cutoffs, target=target, band=band[target]),
        correction_at_cutoffs(trace=short, cutoff=cutoffs, target=target, band=band[target]),
        atol=TOLERANCE,
    )


def test_band_slots_pads_the_short_band() -> None:
    band = np.array([0, 1, 0, 0, 1])
    slots = band_slots(band=band, n_bands=3)
    assert slots.n_rows.tolist() == [3, 2, 0]
    assert slots.positions[0].tolist() == [0, 2, 3]
    assert slots.positions[1, :2].tolist() == [1, 4]


def test_bad_settings_shape_raises() -> None:
    residuals, band = _streams(seed=5)
    with pytest.raises(ValueError, match="hyper must have shape"):
        run_kalman_grid(
            residuals=residuals, band=band, hyper=np.zeros((2, 3)), n_bands=N_BANDS, device="cpu"
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_grid_is_much_faster_than_looping_run_kalman() -> None:
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
    _run(
        residuals=residuals, band=band, hyper=settings, device="cuda", keep_states=False
    )  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    _run(residuals=residuals, band=band, hyper=settings, device="cuda", keep_states=False)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for values in settings[:3]:
        run_kalman(
            residual=residuals[0], band=band, hyper=FilterHyperparameters.from_array(values=values)
        )
    cpu_seconds = (time.perf_counter() - start) / 3 * n_streams * n_settings
    assert cpu_seconds / gpu_seconds > 5.0
