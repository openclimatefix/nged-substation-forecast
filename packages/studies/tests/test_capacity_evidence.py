"""Tests of the six evidence tools: each fails on the defect it names."""

import inspect
from dataclasses import replace

import numpy as np
import pytest
from studies.capacity_estimators import CapacityRecord, scramble_from_day, synthetic_record
from studies.capacity_evidence import (
    CapOverlay,
    Evidence,
    PlantFit,
    build_evidence,
    find_changepoints,
    stuck_days,
)

PLANTS = 6
DAYS = 700
LAST_DAY = 640
FIT = PlantFit(ac_capacity_mw=1.0, dc_capacity_mw=1.2, tilt_deg=25.0, azimuth_deg=180.0)


def _record(*, factor: np.ndarray | None = None, seed: int = 1) -> CapacityRecord:
    return synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=seed)


def _evidence(
    *, record: CapacityRecord, last_day: int = DAYS - 1, cap: CapOverlay | None = None
) -> Evidence:
    return build_evidence(record=record, plant=0, last_day=last_day, fit=FIT, cap=cap)


def _same(*, first: Evidence, second: Evidence) -> bool:
    arrays = (
        "clear_raw", "clear_corrected", "shared", "own", "neighbour", "hour_table", "envelope",
        "plateau", "clip_share", "gap", "stuck", "energy_mwh", "modelled_mwh",
    )  # fmt: skip
    return (
        all(
            np.array_equal(getattr(first, name), getattr(second, name), equal_nan=True)
            for name in arrays
        )
        and first.changepoints == second.changepoints
    )


def test_a_known_loss_of_twenty_percent_recovers_a_ratio_of_point_eight() -> None:
    factor = np.ones((PLANTS, DAYS))
    factor[0, 450:] = 0.8
    evidence = _evidence(record=_record(factor=factor))
    before = np.nanmedian(evidence.clear_raw[300:440])
    after = np.nanmedian(evidence.clear_raw[460:])
    assert after / before == pytest.approx(0.8, abs=0.01)


def test_changes_in_cloudy_hours_do_not_move_the_clear_sky_ratio() -> None:
    record = _record()
    ghi, clear = record.cams_ghi_w_m2[0], record.clear_sky_w_m2[0]
    cloudy = (clear >= 50) & (ghi < 0.8 * clear)
    power = record.power_mw.copy()
    power[0][cloudy] *= 0.3
    changed = _evidence(record=record.with_power(power_mw=power))
    assert np.array_equal(changed.clear_raw, _evidence(record=record).clear_raw, equal_nan=True)


def test_a_gap_day_and_a_six_hour_stuck_run_are_masked_and_flagged() -> None:
    record = _record()
    power = record.power_mw.copy()
    power[0, 100] = np.nan
    power[0, 200, 8:14] = 0.5
    power[0, 300, 8:13] = 0.5
    evidence = _evidence(record=record.with_power(power_mw=power))
    assert evidence.gap[100]
    assert evidence.stuck[200]
    assert not evidence.stuck[300]
    assert np.isnan(evidence.clear_raw[100])
    assert np.isnan(evidence.clear_raw[200])


def test_a_stuck_run_must_be_positive_and_six_readings_long() -> None:
    power = np.zeros((3, 24))
    power[1, 2:8] = 0.4
    power[2, 0:5] = 0.4
    assert stuck_days(power=power).tolist() == [False, True, False]


def test_the_builder_takes_no_event_and_ignores_which_record_object_holds_the_power() -> None:
    parameters = set(inspect.signature(build_evidence).parameters)
    assert parameters == {"record", "plant", "last_day", "fit", "cap"}
    record = _record()
    copied = replace(record, power_mw=record.power_mw.copy())
    assert _same(first=_evidence(record=record), second=_evidence(record=copied))


def test_every_tool_ignores_inputs_after_the_cut_off_day() -> None:
    record = _record()
    cap = CapOverlay(
        cap_mw=np.full((DAYS, 24), 1.0), binding=np.random.default_rng(0).random((DAYS, 24)) > 0.5
    )
    scrambled = scramble_from_day(record=record, first_day=LAST_DAY + 1, seed=7)
    later = np.arange(DAYS) > LAST_DAY
    noisy_cap = CapOverlay(
        cap_mw=np.where(later[:, None], 9.0, cap.cap_mw),
        binding=np.where(later[:, None], ~cap.binding, cap.binding),
    )
    kept = _evidence(record=record, last_day=LAST_DAY, cap=cap)
    other = _evidence(record=scrambled, last_day=LAST_DAY, cap=noisy_cap)
    assert _same(first=kept, second=other)
    assert kept.cap_binding_share is not None
    assert other.cap_binding_share is not None
    assert np.array_equal(kept.cap_binding_share, other.cap_binding_share, equal_nan=True)
    assert len(kept.dates) == LAST_DAY + 1


def _noisy(*, steps: list[tuple[int, float]], days: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    values = 1.0 + 0.03 * rng.standard_normal(days)
    for day, factor in steps:
        values[day:] *= factor
    return np.datetime64("2024-01-01") + np.arange(days), values


def test_a_planted_step_is_found_within_three_days_with_direction_and_size() -> None:
    dates, values = _noisy(steps=[(200, 0.8)])
    found = find_changepoints(dates=dates, values=values)
    assert len(found) == 1
    assert abs(found[0].day_index - 200) <= 3
    assert found[0].direction == "down"
    assert found[0].size == pytest.approx(-0.2, abs=0.03)
    assert found[0].p_value < 0.05


def test_a_flat_noisy_series_gives_no_candidate_and_two_steps_give_two() -> None:
    dates, flat = _noisy(steps=[])
    assert find_changepoints(dates=dates, values=flat) == ()
    dates, values = _noisy(steps=[(120, 0.8), (260, 1.25)])
    found = find_changepoints(dates=dates, values=values)
    assert [abs(c.day_index - d) <= 3 for c, d in zip(found, (120, 260), strict=True)] == [
        True,
        True,
    ]


def test_a_candidate_near_the_cut_off_day_is_flagged_unconfirmed() -> None:
    dates, values = _noisy(steps=[(388, 0.7)])
    found = find_changepoints(dates=dates, values=values)
    assert found
    assert found[-1].unconfirmed


def test_the_cap_overlay_reports_the_share_of_daylight_hours_it_binds() -> None:
    record = _record()
    binding = np.zeros((DAYS, 24), dtype=bool)
    binding[50] = True
    evidence = _evidence(
        record=record, cap=CapOverlay(cap_mw=np.full((DAYS, 24), 2.0), binding=binding)
    )
    assert evidence.cap_binding_share is not None
    assert evidence.cap_binding_share[50] == pytest.approx(1.0)
    assert evidence.cap_binding_share[51] == pytest.approx(0.0)
    assert evidence.cap_mean_mw is not None
    assert evidence.cap_mean_mw[50] == pytest.approx(2.0)


def _series(*, dip: tuple[int, int, float] | None, days: int = 400) -> np.ndarray:
    values = 1.0 + 0.03 * np.random.default_rng(8).standard_normal(days)
    if dip is not None:
        start, length, size = dip
        values[start : start + length] *= 1.0 - size
    return values


def test_the_local_scan_brackets_a_one_month_loss_and_ignores_a_flat_series() -> None:
    from studies.capacity_evidence import local_scan

    found = local_scan(
        clear_ratio=_series(dip=(150, 30, 0.3)), all_sky_ratio=_series(dip=(150, 30, 0.3))
    )
    assert len(found) == 1
    episode = found[0]
    assert episode.direction == "down"
    assert abs(episode.deepest_start - 150) <= 5
    assert episode.first_day <= 160
    assert episode.last_day >= 170
    assert episode.last_day - episode.first_day < 60
    assert episode.effect == pytest.approx(-0.3, abs=0.04)
    assert episode.all_sky_effect == pytest.approx(-0.3, abs=0.04)
    assert local_scan(clear_ratio=_series(dip=None), all_sky_ratio=_series(dip=None)) == ()


def test_the_local_scan_needs_enough_observations_and_lists_at_most_fifteen_episodes() -> None:
    from studies.capacity_evidence import LOCAL_MAX_EPISODES, local_scan

    sparse = _series(dip=(150, 30, 0.3))
    sparse[100:260] = np.nan
    assert local_scan(clear_ratio=sparse, all_sky_ratio=sparse) == ()
    wavy = 1.0 + 0.4 * np.sign(np.sin(np.arange(2000) * 2 * np.pi / 90))
    assert len(local_scan(clear_ratio=wavy, all_sky_ratio=wavy)) <= LOCAL_MAX_EPISODES


def test_the_local_scan_of_a_cut_record_ignores_later_days() -> None:
    record = _record()
    scrambled = scramble_from_day(
        record=record, first_day=LAST_DAY + 1, seed=3, scramble_static=True
    )
    assert (
        _evidence(record=record, last_day=LAST_DAY).local_episodes
        == _evidence(record=scrambled, last_day=LAST_DAY).local_episodes
    )
