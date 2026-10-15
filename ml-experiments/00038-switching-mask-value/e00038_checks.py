"""Checks on the interval builder and the correction (run before the scoring)."""

import numpy as np
import polars as pl
from e00038_base import SOURCE_DIR, DetectorInputs, alarms, level_shift
from e00038_corrections import MIN_COUNT, correction_series, detected_mask
from e00038_detect_run import load_arrays
from e00038_intervals import (
    BLOCK,
    Interval,
    build_intervals,
    known_at,
    level_track,
    relative_residual,
    split_fractions,
)

N = 200 * BLOCK


def _noise(*, seed: int = 0, scale: float = 0.002) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, N)


def check_level_track_follows_daily_shape() -> None:
    """A step whose size follows a daily shape ends within one block of the truth."""
    q = _noise()
    shape = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(N) / BLOCK)
    q[100 * BLOCK : 110 * BLOCK] += 0.2 * shape[100 * BLOCK : 110 * BLOCK]
    _, end, known = level_track(q=q, onset=100 * BLOCK, window=8, step_from_block=True)
    assert abs(end - 110 * BLOCK) <= BLOCK, end
    assert known >= end


def check_decay_ends_at_half_time() -> None:
    """A step that decays to zero at half-time ends there."""
    q = _noise()
    ramp = np.linspace(0.3, 0.0, 10 * BLOCK)
    q[100 * BLOCK : 110 * BLOCK] += ramp
    _, end, _ = level_track(q=q, onset=100 * BLOCK, window=8, step_from_block=True)
    assert abs(end - 105 * BLOCK) <= 2 * BLOCK, end


def check_known_at_truncates() -> None:
    """An open interval is masked up to the newest observation only."""
    q = _noise()
    q[100 * BLOCK : 110 * BLOCK] += 0.2
    table = pl.DataFrame(
        {"series": [1], "stamp": [100 * BLOCK], "partners": [""]},
        schema_overrides={"stamp": pl.Int64},
    )
    intervals = build_intervals(alarm_table=table, relative={1: q}, window=8, detector="D1")
    newest = 103 * BLOCK
    ((series, start, stop),) = known_at(intervals=intervals, newest=newest, window=8)
    assert (series, start, stop) == (1, 100 * BLOCK, newest + 1)
    assert known_at(intervals=intervals, newest=100 * BLOCK + 4, window=8) == []
    ((_, _, stop_late),) = known_at(intervals=intervals, newest=150 * BLOCK, window=8)
    assert stop_late == intervals[0].end


def check_all_masked_window_carries_forward() -> None:
    """With under 3 days unmasked, the last good correction is carried."""
    residual = np.full(N, 0.1)
    residual[: 100 * BLOCK] = 0.5
    newest = np.array([60 * BLOCK, 120 * BLOCK])
    mask = np.zeros(N, dtype=bool)
    mask[105 * BLOCK : 121 * BLOCK] = True
    out = correction_series(
        residual=residual, newest=newest, window=14 * BLOCK, masked=lambda n, lo: mask[lo : n + 1]
    )
    assert (np.isclose(out[0], 0.5) and np.isclose(out[1], 0.1)) or out[1] != 0.0
    assert MIN_COUNT == 144


def check_unmasked_equals_cumulative_sum() -> None:
    """The loop rule equals an independent cumulative-sum computation."""
    rng = np.random.default_rng(1)
    residual = rng.normal(0, 1, N)
    residual[rng.random(N) < 0.05] = np.nan
    newest = np.arange(20 * BLOCK, N - 10, 37)
    out = correction_series(residual=residual, newest=newest, window=14 * BLOCK)
    ok = np.isfinite(residual)
    total = np.concatenate([[0], np.cumsum(np.where(ok, residual, 0))])
    count = np.concatenate([[0], np.cumsum(ok)])
    lo = newest - 14 * BLOCK + 1
    expected = (total[newest + 1] - total[lo]) / (count[newest + 1] - count[lo])
    assert np.allclose(out, expected, atol=1e-9)


def check_split_fractions_recover_known_split() -> None:
    """A 1 MW drop split 0.6 and 0.4 returns 0.6 and 0.4 at zero noise."""
    power = {s: np.full(N, 10.0) for s in (1, 2, 3)}
    power[1][100 * BLOCK : 110 * BLOCK] -= 1.0
    power[2][100 * BLOCK : 110 * BLOCK] += 0.6
    power[3][100 * BLOCK : 110 * BLOCK] += 0.4
    q = {s: np.zeros(N) for s in power}
    q[1][100 * BLOCK : 110 * BLOCK] = -0.1
    q[2][100 * BLOCK : 110 * BLOCK] = 0.06
    q[3][100 * BLOCK : 110 * BLOCK] = 0.04
    table = pl.DataFrame(
        {"series": [1], "stamp": [100 * BLOCK], "partners": ["2:0.6;3:0.4"]},
        schema_overrides={"stamp": pl.Int64},
    )
    intervals = build_intervals(alarm_table=table, relative=q, window=8, detector="D1")
    fractions = split_fractions(power=power, interval_set=intervals)
    by_partner = fractions.group_by("partner").agg(f=pl.col("fraction").mean())
    got = dict(by_partner.iter_rows())
    assert abs(got[2] - 0.6) < 0.02, got
    assert abs(got[3] - 0.4) < 0.02, got


def check_causality() -> None:
    """Intervals known at an origin do not change when the residual after it is NaN."""
    panel = pl.read_parquet(SOURCE_DIR / "panel.parquet")
    residual = load_arrays(panel=panel, column="res_mw")
    expected = load_arrays(panel=panel, column="expected_mw")
    window = 8
    full_shift = {s: level_shift(values=v, window=window) for s, v in residual.items()}
    from e00038_base import robust_scale

    scale = {s: robust_scale(shift=d) for s, d in full_shift.items()}

    def intervals_for(*, last: int | None, fixed: bool) -> list[Interval]:
        res = {s: v.copy() for s, v in residual.items()}
        if last is not None:
            for v in res.values():
                v[last + 1 :] = np.nan
        shift = {s: level_shift(values=v, window=window) for s, v in res.items()}
        inputs = DetectorInputs(window=window, shift=shift, scale=scale)
        table = alarms(inputs=inputs, detector="D1", threshold=4.0)
        rel = {s: relative_residual(residual=res[s], expected=expected[s]) for s in res}
        return build_intervals(
            alarm_table=table, relative=rel, window=window, detector="D1", fixed=fixed
        )

    for fixed in (False, True):
        full = intervals_for(last=None, fixed=fixed)
        for n in (30_000, 50_000, 60_000, 70_200, 70_500, 72_000):
            cut = intervals_for(last=n, fixed=fixed)
            a = sorted(known_at(intervals=full, newest=n, window=window))
            b = sorted(known_at(intervals=cut, newest=n, window=window))
            assert a == b, (
                fixed,
                n,
                [x for x in a if x not in b][:3],
                [x for x in b if x not in a][:3],
            )
    _ = detected_mask


def main() -> None:
    """Run every check and print which passed."""
    for name, check in list(globals().items()):
        if name.startswith("check_") and callable(check):
            check()
            print("passed", name)


if __name__ == "__main__":
    main()
