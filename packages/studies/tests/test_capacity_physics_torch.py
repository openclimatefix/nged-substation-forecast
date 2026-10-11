"""Tests of the variational physics estimator DP1, all on the CPU in float64."""

from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.capacity_estimators import scramble_from_day, synthetic_record  # noqa: E402
from studies.capacity_physics_torch import (  # noqa: E402
    DP1_DEFAULT_PARAMS,
    REFIT_EVERY_DAYS,
    WINDOW_DAYS,
    run_dp1,
    soft_clip,
)

DAYS: Final[int] = 400
FAST: Final = {**DP1_DEFAULT_PARAMS, "first_steps": 150.0, "later_steps": 60.0}
"""Fewer steps than a real run, enough for the synthetic record's clean signal."""


def _run(record, *, sizes: np.ndarray):  # noqa: ANN001, ANN202
    return run_dp1(
        records=[record],
        dc_prior=sizes / record.static_p99_mw,
        ac_limit=np.full(len(sizes), 10.0),
        params=FAST,
        dtype=torch.float64,
    )[0]


def _sizes(plants: int) -> np.ndarray:
    return np.array([1.0 + 0.2 * (p == 0) + 0.1 * p for p in range(plants)])


def test_it_recovers_a_planted_direct_current_capacity_and_its_loss() -> None:
    factor = np.ones((3, DAYS))
    factor[1, 250:320] = 0.7
    record = synthetic_record(capacity_factor=factor, days=DAYS, seed=2)
    estimate = _run(record, sizes=_sizes(3))
    unit = _sizes(3) / record.static_p99_mw
    # Before the loss the capacity is the true one (the prior's own value), within 6%.
    assert estimate.level[1, 240] == pytest.approx(unit[1], rel=0.06)
    # By the end of the loss the last week's capacity has fallen towards 70%.
    assert estimate.level[1, 316] / unit[1] == pytest.approx(0.7, abs=0.1)
    # After it ends, it returns.
    assert estimate.level[1, 380] / unit[1] == pytest.approx(1.0, abs=0.1)
    assert estimate.level[0, 316] / unit[0] == pytest.approx(1.0, abs=0.1)
    assert estimate.sd is not None
    assert np.isfinite(estimate.sd[1, 316])


def test_there_is_no_estimate_before_the_first_full_window_and_it_is_held_between_refits() -> None:
    record = synthetic_record(days=DAYS, seed=3)
    estimate = _run(record, sizes=_sizes(3))
    assert np.isnan(estimate.level[:, : WINDOW_DAYS - 1]).all()
    held = estimate.level[0, WINDOW_DAYS - 1 : WINDOW_DAYS - 1 + REFIT_EVERY_DAYS]
    assert (held == held[0]).all()
    assert estimate.level[0, WINDOW_DAYS - 1 + REFIT_EVERY_DAYS] != held[0]


def test_an_estimate_for_day_d_ignores_every_input_after_day_d() -> None:
    record = synthetic_record(days=DAYS, seed=4)
    # A refit happens on days 62, 69, ... so day 300 reads the refit of day 300 (62 + 34 * 7).
    day = 62 + 34 * REFIT_EVERY_DAYS
    original = _run(record, sizes=_sizes(3))
    after = _run(scramble_from_day(record=record, first_day=day + 1, seed=1), sizes=_sizes(3))
    from_day = _run(scramble_from_day(record=record, first_day=day, seed=1), sizes=_sizes(3))
    assert np.array_equal(original.level[:, day], after.level[:, day])
    assert not np.array_equal(original.level[:, day], from_day.level[:, day])


def test_the_soft_clip_follows_the_smaller_of_the_power_and_the_limit() -> None:
    limit = torch.tensor(2.0, dtype=torch.float64)
    low = soft_clip(power=torch.tensor(0.5, dtype=torch.float64), limit=limit)
    high = soft_clip(power=torch.tensor(5.0, dtype=torch.float64), limit=limit)
    assert float(low) == pytest.approx(0.5, abs=1e-6)
    assert float(high) == pytest.approx(2.0, abs=1e-3)
    assert float(soft_clip(power=torch.tensor(4.0, dtype=torch.float64), limit=limit)) < 2.0 + 1e-9


def test_identical_records_in_one_batch_get_identical_estimates() -> None:
    record = synthetic_record(days=200, seed=5)
    sizes = _sizes(3)
    first, second = run_dp1(
        records=[record, record],
        dc_prior=sizes / record.static_p99_mw,
        ac_limit=np.full(3, 10.0),
        params={**FAST, "first_steps": 60.0, "later_steps": 20.0},
        dtype=torch.float64,
    )
    assert np.allclose(first.level, second.level, rtol=1e-9, equal_nan=True)
    assert first.sd is not None
    assert second.sd is not None
    assert np.allclose(first.sd, second.sd, rtol=1e-9, equal_nan=True)


def test_a_larger_data_weight_lets_the_data_narrow_the_posterior() -> None:
    record = synthetic_record(days=DAYS, seed=4)
    sizes = _sizes(3)
    sds = []
    for weight in (1.0, 30.0):
        estimate = run_dp1(
            records=[record],
            dc_prior=sizes / record.static_p99_mw,
            ac_limit=np.full(3, 10.0),
            params={**FAST, "data_weight": weight},
            dtype=torch.float64,
        )[0]
        assert estimate.sd is not None
        sds.append(float(np.nanmean(estimate.sd[:, 100:] / estimate.level[:, 100:])))
    assert sds[1] < sds[0]
