import numpy as np
import pytest
from studies.battery_dispatch import (
    HOURS_PER_HALF_HOUR,
    lp_schedule,
    rank_rule_schedule,
    state_of_charge,
)

# Cheap overnight, dear in the evening: a typical day.
_DAY = np.concatenate(
    [np.full(12, 20.0), np.full(12, 40.0), np.full(12, 60.0), np.full(12, 120.0)]
) + np.linspace(0.0, 0.5, 48)


def test_rank_rule_charges_in_the_cheapest_and_discharges_in_the_dearest_half_hours() -> None:
    schedule = rank_rule_schedule(prices=_DAY, duration_half_hours=4)

    assert np.flatnonzero(schedule == -1.0).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(schedule == 1.0).tolist() == [44, 45, 46, 47]
    assert np.count_nonzero(schedule) == 8


def test_rank_rule_idles_when_the_dearest_block_comes_before_the_cheapest() -> None:
    schedule = rank_rule_schedule(prices=_DAY[::-1].copy(), duration_half_hours=4)

    assert not schedule.any()


def test_rank_rule_idles_when_the_spread_is_below_the_efficiency_threshold() -> None:
    prices = np.where(np.arange(48) < 24, 100.0, 110.0) + np.linspace(0.0, 0.1, 48)

    assert not rank_rule_schedule(prices=prices, duration_half_hours=4).any()


def test_rank_rule_gives_a_day_with_a_missing_price_a_zero_schedule() -> None:
    prices = np.concatenate([_DAY, _DAY])
    prices[60] = np.nan

    schedule = rank_rule_schedule(prices=prices, duration_half_hours=4).reshape(2, 48)

    assert schedule[0].any()
    assert not schedule[1].any()


def test_prices_that_do_not_fill_whole_days_raise() -> None:
    with pytest.raises(ValueError, match="whole days"):
        rank_rule_schedule(prices=np.ones(50), duration_half_hours=4)


def test_lp_schedule_keeps_the_state_of_charge_inside_its_bounds() -> None:
    prices = np.tile(_DAY, 3) * np.repeat([1.0, 0.5, 2.0], 48)

    schedule = lp_schedule(prices=prices, energy_hours=2.0, soc_min=0.1, soc_max=0.9)
    soc = state_of_charge(schedule=schedule, energy_hours=2.0, eta_one_way=0.92, initial_soc=0.1)

    assert soc.min() >= 0.1 - 1e-9
    assert soc.max() <= 0.9 + 1e-9
    assert soc.max() > 0.8
    assert abs(soc[47] - 0.1) < 1e-9


def test_lp_schedule_charges_cheap_and_discharges_dear_without_doing_both_at_once() -> None:
    schedule = lp_schedule(prices=_DAY, energy_hours=2.0)

    assert schedule[:12].sum() < 0
    assert schedule[36:].sum() > 0
    assert schedule[:12].max() <= 0
    assert schedule[36:].min() >= 0
    assert np.abs(schedule).max() <= 1.0 + 1e-9


def test_lp_schedule_with_a_flat_price_does_nothing() -> None:
    assert not np.abs(lp_schedule(prices=np.full(96, 50.0), energy_hours=2.0)).max() > 1e-9


def test_a_low_cycle_cap_binds() -> None:
    uncapped = lp_schedule(prices=_DAY, energy_hours=2.0, cycles_per_day_cap=1.0)
    capped = lp_schedule(prices=_DAY, energy_hours=2.0, cycles_per_day_cap=0.25)

    usable = 0.9 * 2.0
    discharged = np.clip(capped, 0.0, None).sum() * HOURS_PER_HALF_HOUR
    assert discharged == pytest.approx(0.25 * usable)
    # The usable energy leaves the grid side at the one-way efficiency, below the cap of 1 cycle.
    assert np.clip(uncapped, 0.0, None).sum() * HOURS_PER_HALF_HOUR == pytest.approx(usable * 0.92)


def test_lp_schedule_carries_the_state_of_charge_to_the_next_day() -> None:
    schedule = lp_schedule(prices=np.tile(_DAY, 2), energy_hours=2.0, initial_soc=0.5)
    soc = state_of_charge(schedule=schedule, energy_hours=2.0, eta_one_way=0.92, initial_soc=0.5)

    assert soc[47] == pytest.approx(0.5)
    assert soc[95] == pytest.approx(0.5)


def test_rank_rule_charges_in_the_earliest_of_tied_cheapest_half_hours() -> None:
    prices = np.concatenate([np.full(24, 10.0), np.full(24, 100.0)])

    schedule = rank_rule_schedule(prices=prices, duration_half_hours=4)

    assert np.flatnonzero(schedule == -1.0).tolist() == [0, 1, 2, 3]


def test_rank_rule_idles_when_the_efficiency_adjusted_spread_exactly_breaks_even() -> None:
    prices = np.concatenate([np.full(24, 50.0), np.full(24, 100.0)])

    exactly_even = rank_rule_schedule(
        prices=prices, duration_half_hours=4, round_trip_efficiency=0.5
    )
    just_above = rank_rule_schedule(
        prices=prices, duration_half_hours=4, round_trip_efficiency=0.51
    )

    assert not exactly_even.any()
    assert just_above.any()


@pytest.mark.parametrize("duration", [0, 25])
def test_rank_rule_rejects_a_duration_outside_one_to_twenty_four(duration: int) -> None:
    with pytest.raises(ValueError, match="1 to 24"):
        rank_rule_schedule(prices=_DAY, duration_half_hours=duration)


def test_rank_rule_accepts_the_longest_duration_of_twenty_four() -> None:
    prices = np.concatenate([np.full(24, 10.0), np.full(24, 100.0)])

    schedule = rank_rule_schedule(prices=prices, duration_half_hours=24)

    assert (schedule[:24] == -1.0).all()
    assert (schedule[24:] == 1.0).all()


def test_lp_schedule_gives_a_day_with_a_missing_price_a_zero_schedule() -> None:
    prices = np.concatenate([_DAY, _DAY])
    prices[60] = np.nan

    schedule = lp_schedule(prices=prices, energy_hours=2.0).reshape(2, 48)

    assert schedule[0].any()
    assert not schedule[1].any()


def test_lp_schedule_starts_from_the_given_state_of_charge_and_stays_above_the_minimum() -> None:
    # Dear early and cheap late: a battery starting half full discharges first, down to the
    # minimum state of charge, then recharges to end where it started.
    prices = _DAY[::-1].copy()

    schedule = lp_schedule(
        prices=prices, energy_hours=2.0, soc_min=0.1, soc_max=0.9, initial_soc=0.5
    )
    soc = state_of_charge(schedule=schedule, energy_hours=2.0, eta_one_way=0.92, initial_soc=0.5)

    assert schedule[:12].sum() > 0
    assert soc.min() == pytest.approx(0.1)
    assert soc[47] == pytest.approx(0.5)


def test_state_of_charge_applies_the_efficiency_on_the_right_side_of_each_direction() -> None:
    schedule = np.array([-1.0, 1.0])

    soc = state_of_charge(schedule=schedule, energy_hours=2.0, eta_one_way=0.8, initial_soc=0.5)

    # Charging 1 MW for half an hour stores 0.5 * 0.8 MWh, discharging draws 0.5 / 0.8 MWh.
    assert soc[0] == pytest.approx(0.5 + 0.4 / 2.0)
    assert soc[1] == pytest.approx(0.5 + (0.4 - 0.625) / 2.0)
