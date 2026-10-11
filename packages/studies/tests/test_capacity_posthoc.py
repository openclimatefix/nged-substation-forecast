"""Tests of the post hoc DSKF and HYST estimators."""

import numpy as np
import pytest
from studies.capacity_estimators import RawEstimate
from studies.capacity_posthoc import deseasonalise_causally, hysteresis_level

DATES = np.datetime64("2020-03-01") + np.arange(4 * 365)


def _seasonal_level() -> np.ndarray:
    doy = (DATES - DATES.astype("datetime64[Y]")).astype(int)
    shape = 1.0 + 0.3 * np.cos(2 * np.pi * doy / 365.0)
    return np.stack([shape * size for size in (1.0, 2.0, 3.0)])


def test_the_seasonal_shape_is_divided_out_from_the_second_calendar_year_on() -> None:
    level = _seasonal_level()
    out = deseasonalise_causally(raw=RawEstimate(level=level, sd=level * 0.1), dates=DATES)
    first_year = DATES.astype("datetime64[Y]") == DATES[0].astype("datetime64[Y]")
    assert np.isnan(out.level[:, first_year]).all()
    later = ~first_year
    relative = out.level[:, later] / np.array([1.0, 2.0, 3.0])[:, None]
    finite = relative[np.isfinite(relative)]
    assert len(finite) > 2 * 365
    assert finite.std() < 0.03
    assert finite == pytest.approx(1.0, abs=0.1)
    assert out.sd is not None
    ratio = (out.sd[:, later] / out.level[:, later])[np.isfinite(out.level[:, later])]
    assert ratio == pytest.approx(0.1, rel=1e-6)


def test_a_change_from_the_middle_of_a_year_alters_nothing_before_the_next_year() -> None:
    level = _seasonal_level()
    changed = level.copy()
    middle = 800
    changed[:, middle:] *= 0.5
    base = deseasonalise_causally(raw=RawEstimate(level=level, sd=None), dates=DATES).level / level
    other = (
        deseasonalise_causally(raw=RawEstimate(level=changed, sd=None), dates=DATES).level / changed
    )
    next_year = np.flatnonzero(
        DATES.astype("datetime64[Y]") > DATES[middle].astype("datetime64[Y]")
    )[0]
    assert np.allclose(base[:, :next_year], other[:, :next_year], equal_nan=True)
    assert not np.allclose(base[:, next_year:], other[:, next_year:], equal_nan=True)


def test_the_scale_holds_until_a_drop_has_lasted_seven_days_then_follows_and_releases() -> None:
    ratio = np.ones(60)
    ratio[10:30] = 0.8
    ratio[30:] = 0.95
    out = hysteresis_level(ratio=ratio)
    assert (out[:16] == 1.0).all()
    assert out[16] == pytest.approx(0.8)
    assert (out[16:30] == 0.8).all()
    assert (out[30:36] == 0.95).all()
    assert out[36] == 1.0
    assert (out[37:] == 1.0).all()


def test_a_short_dip_or_a_small_loss_never_moves_the_scale() -> None:
    short = np.ones(60)
    short[10:16] = 0.7
    small = np.full(60, 0.93)
    assert (hysteresis_level(ratio=short) == 1.0).all()
    assert (hysteresis_level(ratio=small) == 1.0).all()


def test_the_hysteresis_level_ignores_later_days() -> None:
    rng = np.random.default_rng(0)
    ratio = 1.0 - 0.3 * (rng.random(120) < 0.5)
    changed = ratio.copy()
    changed[70:] = 0.2
    assert (hysteresis_level(ratio=ratio)[:70] == hysteresis_level(ratio=changed)[:70]).all()
