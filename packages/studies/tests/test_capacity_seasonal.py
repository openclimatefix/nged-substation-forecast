"""Tests of the seasonal reference and the corrected-ratio estimators."""

import numpy as np
import pytest
from studies.capacity_estimators import scramble_from_day, synthetic_record
from studies.capacity_seasonal import (
    corrected_record,
    daily_log_ratio,
    hm2_scale,
    hyst2_ratio,
    hyst2_scale,
    seasonal_reference,
)

PLANTS = 6
DAYS = 700
FIRST_DAY_AFTER_REFERENCE = 560
"""Day index 560 of a record starting 2023-06-01 is 2024-12-12, after the reference year."""


def _factor(*, plants: int = PLANTS) -> np.ndarray:
    return np.ones((plants, DAYS))


def _month_of(*, record_dates: np.ndarray, month: str) -> np.ndarray:
    return record_dates.astype("datetime64[M]") == np.datetime64(month)


def _median_corrected(
    *, record_dates: np.ndarray, log_ratio: np.ndarray, plant: int, month: str
) -> float:
    selected = _month_of(record_dates=record_dates, month=month)
    return float(np.exp(np.nanmedian(log_ratio[plant, selected])))


def test_a_december_dip_shared_by_all_plants_is_corrected_to_flat() -> None:
    factor = _factor()
    base = synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=1)
    dipped_days = _month_of(record_dates=base.dates, month="2024-12")
    factor[:, dipped_days] = 0.8
    record = synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=1)
    raw = daily_log_ratio(record=record)
    corrected = daily_log_ratio(
        record=corrected_record(record=record, reference=seasonal_reference(record=record))
    )
    for plant in range(PLANTS):
        before = _median_corrected(
            record_dates=record.dates, log_ratio=raw, plant=plant, month="2024-12"
        )
        after = _median_corrected(
            record_dates=record.dates, log_ratio=corrected, plant=plant, month="2024-12"
        )
        reference_level = _median_corrected(
            record_dates=record.dates, log_ratio=corrected, plant=plant, month="2024-06"
        )
        assert before / _median_corrected(
            record_dates=record.dates, log_ratio=raw, plant=plant, month="2024-06"
        ) == pytest.approx(0.8, abs=0.03)
        assert after / reference_level == pytest.approx(1.0, abs=0.02)


def test_a_loss_at_one_plant_survives_the_correction() -> None:
    factor = _factor()
    base = synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=2)
    in_month = _month_of(record_dates=base.dates, month="2025-01")
    factor[0, in_month] = 0.75
    record = synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=2)
    corrected = daily_log_ratio(
        record=corrected_record(record=record, reference=seasonal_reference(record=record))
    )
    low = _median_corrected(
        record_dates=record.dates, log_ratio=corrected, plant=0, month="2025-01"
    )
    normal = _median_corrected(
        record_dates=record.dates, log_ratio=corrected, plant=0, month="2024-06"
    )
    assert low / normal == pytest.approx(0.75, abs=0.03)


def test_the_reference_of_a_day_ignores_every_later_day() -> None:
    record = synthetic_record(plants=PLANTS, days=DAYS, seed=3)
    scrambled = scramble_from_day(record=record, first_day=FIRST_DAY_AFTER_REFERENCE + 1, seed=9)
    kept = slice(0, FIRST_DAY_AFTER_REFERENCE + 1)
    original = seasonal_reference(record=record).log_reference[:, kept]
    after_noise = seasonal_reference(record=scrambled).log_reference[:, kept]
    assert np.array_equal(original, after_noise)


def test_hysteresis_holds_at_static_power_then_follows_a_lasting_drop() -> None:
    factor = _factor()
    factor[0, 600:] = 0.6
    record = synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=4)
    corrected = corrected_record(record=record, reference=seasonal_reference(record=record))
    ratio, _ = hyst2_ratio(corrected=corrected)
    scale = hyst2_scale(record=record, ratio=ratio)
    static = record.static_p99_mw
    assert scale.scale_mw[0, 590] == pytest.approx(static[0])
    assert scale.scale_mw[0, 690] < 0.8 * static[0]


def test_the_hidden_markov_estimator_runs_on_the_corrected_record() -> None:
    record = synthetic_record(plants=PLANTS, days=DAYS, seed=5)
    corrected = corrected_record(record=record, reference=seasonal_reference(record=record))
    scale = hm2_scale(
        record=record,
        corrected=corrected,
        params={"stay_probability": 0.98, "observation_sd": 0.15},
    )
    assert scale.scale_mw.shape == (PLANTS, DAYS)
    assert np.isfinite(scale.scale_mw).all()


def test_the_shared_term_of_a_plant_is_the_median_of_the_other_plants_only() -> None:
    from studies.capacity_seasonal import shared_term

    centred = np.array([[10.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    shared = shared_term(centred=centred)
    assert shared[0, 0] == 1.5
    assert shared[1, 0] == 6.0
    assert shared[2, 0] == 5.5
    assert np.isnan(shared_term(centred=np.array([[1.0], [np.nan]]))[0, 0])
