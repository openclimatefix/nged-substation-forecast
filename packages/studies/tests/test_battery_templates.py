from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from studies.battery_templates import (
    MAX_HOME_EFFICIENCY,
    MIN_HOME_EFFICIENCY,
    SENSITIVITY_LP_SETTINGS,
    TARIFF_WINDOWS,
    DomesticTariffNameType,
    FleetSpec,
    LpSettings,
    TariffNameType,
    TariffWindow,
    agile_days,
    agile_template,
    charge_only_template,
    domestic_template,
    draw_homes,
    effective_duration_hours,
    merchant_template,
    simulate_fleet,
    spread_mean,
    window_coverage,
    window_template,
)

HALF_HOUR = timedelta(minutes=30)
ETA = 0.81


def _grid(*, start: datetime, days: int) -> pl.Series:
    return pl.datetime_range(
        start + HALF_HOUR,
        start + timedelta(days=days),
        interval="30m",
        time_unit="us",
        eager=True,
    ).alias("half_hour_end_time")


def _index(*, grid: pl.Series, start: datetime, instant: datetime) -> int:
    """The grid position of the half-hour that starts at `instant`."""
    return int((instant - start) / HALF_HOUR)


DECEMBER = datetime(2025, 12, 1, tzinfo=UTC)  # UK clock equals UTC
SEPTEMBER = datetime(2025, 9, 1, tzinfo=UTC)  # UK clock is UTC + 1 hour


@pytest.mark.parametrize("name", ["intelligent_octopus_go", "octopus_go", "octopus_flux"])
def test_a_window_template_starts_charging_exactly_at_the_windows_first_half_hour(
    name: TariffNameType,
) -> None:
    grid = _grid(start=DECEMBER, days=10)
    window = TARIFF_WINDOWS[name]

    template = window_template(
        half_hour_end_time=grid, window=window, duration_hours=2.0, round_trip_efficiency=ETA
    )

    window_start = DECEMBER + timedelta(days=4, hours=window.charge_start_hour)
    first = _index(grid=grid, start=DECEMBER, instant=window_start)
    assert template[first] == pytest.approx(-1.0)
    assert template[first - 1] == 0.0


def test_a_window_template_follows_the_uk_clock_through_daylight_saving() -> None:
    grid = _grid(start=SEPTEMBER, days=10)

    template = window_template(
        half_hour_end_time=grid,
        window=TARIFF_WINDOWS["octopus_go"],
        duration_hours=2.0,
        round_trip_efficiency=ETA,
    )

    # Octopus Go starts at 00:30 BST, which is 23:30 UTC on the evening before.
    first = _index(
        grid=grid, start=SEPTEMBER, instant=SEPTEMBER + timedelta(days=4) - timedelta(minutes=30)
    )
    assert template[first] == pytest.approx(-1.0)
    assert template[first - 1] == 0.0


@pytest.mark.parametrize("name", ["intelligent_octopus_go", "octopus_go", "octopus_flux"])
@pytest.mark.parametrize("duration_hours", [0.8, 2.0, 6.0])
def test_a_window_template_discharges_its_charged_energy_times_the_round_trip_efficiency(
    name: TariffNameType, duration_hours: float
) -> None:
    grid = _grid(start=DECEMBER, days=12)
    template = window_template(
        half_hour_end_time=grid,
        window=TARIFF_WINDOWS[name],
        duration_hours=duration_hours,
        round_trip_efficiency=ETA,
    )

    # Six whole cycles lie between 12:00 on day 2 and 12:00 on day 8.
    cycles = template[2 * 48 + 24 : 8 * 48 + 24]
    charged = -cycles[cycles < 0].sum() * 0.5
    discharged = cycles[cycles > 0].sum() * 0.5
    assert charged > 0
    assert discharged == pytest.approx(charged * ETA, rel=1e-9)


def test_a_duration_longer_than_the_windows_allow_is_cut_to_what_they_allow() -> None:
    flux = TARIFF_WINDOWS["octopus_flux"]

    capped = effective_duration_hours(window=flux, duration_hours=6.0, round_trip_efficiency=ETA)

    assert capped == pytest.approx(np.sqrt(ETA) * 3.0)
    assert effective_duration_hours(
        window=flux, duration_hours=1.0, round_trip_efficiency=ETA
    ) == pytest.approx(1.0)


def test_the_red_band_template_runs_on_weekdays_only() -> None:
    grid = _grid(start=DECEMBER, days=14)  # 1 December 2025 is a Monday
    template = window_template(
        half_hour_end_time=grid,
        window=TARIFF_WINDOWS["red_band"],
        duration_hours=1.5,
        round_trip_efficiency=ETA,
    )

    daily_activity = np.abs(template).reshape(14, 48).sum(axis=1)
    assert (daily_activity[[0, 1, 2, 3, 4, 7, 8]] > 0).all()
    assert (daily_activity[[5, 6, 12, 13]] == 0).all()


def test_the_charge_only_template_never_discharges_and_starts_at_the_window() -> None:
    grid = _grid(start=DECEMBER, days=10)
    window = TARIFF_WINDOWS["octopus_flux"]

    template = charge_only_template(half_hour_end_time=grid, window=window, charge_hours=3.0)

    assert template.max() <= 0.0
    first = _index(grid=grid, start=DECEMBER, instant=DECEMBER + timedelta(days=4, hours=2))
    assert template[first] == -1.0
    assert template[first - 1] == 0.0
    assert template[first + 5] == -1.0
    assert template[first + 6] == 0.0


def test_a_spread_of_durations_softens_the_end_of_the_charge_but_keeps_the_energy() -> None:
    grid = _grid(start=DECEMBER, days=10)
    window = TARIFF_WINDOWS["octopus_go"]

    def build(duration: float) -> np.ndarray:
        return window_template(
            half_hour_end_time=grid,
            window=window,
            duration_hours=duration,
            round_trip_efficiency=ETA,
        )

    sharp = build(2.0)
    soft = spread_mean(template_for_duration=build, duration_hours=2.0, sd_log=0.25)

    partial_sharp = ((sharp > -1.0) & (sharp < 0.0)).sum()
    partial_soft = ((soft > -1.0) & (soft < 0.0)).sum()
    assert partial_soft > partial_sharp
    assert soft.sum() == pytest.approx(sharp.sum(), rel=0.2)  # nearly the same net flow
    assert spread_mean(
        template_for_duration=build, duration_hours=2.0, sd_log=0.0
    ) == pytest.approx(sharp)


def test_spread_mean_weights_sum_to_one() -> None:
    mean = spread_mean(
        template_for_duration=lambda duration: np.full(4, 3.0), duration_hours=2.0, sd_log=0.4
    )

    assert mean == pytest.approx(np.full(4, 3.0))


def test_a_merchant_template_charges_cheap_hours_and_discharges_dear_ones_with_losses() -> None:
    day = np.concatenate([np.full(12, 20.0), np.full(24, 50.0), np.full(12, 120.0)])
    prices = np.tile(day, 3) + np.tile(np.linspace(0.0, 0.3, 48), 3)

    template = merchant_template(
        day_ahead_prices=prices, duration_hours=2.0, round_trip_efficiency=ETA
    )

    assert template[:12].mean() < 0.0
    assert template[36:].mean() > 0.0
    charged = -template[template < 0].sum() * 0.5
    discharged = template[template > 0].sum() * 0.5
    assert discharged == pytest.approx(charged * ETA, rel=1e-6)


def _agile_prices(*, start: datetime, days: int, cheapest_local_hour: float) -> pl.DataFrame:
    times = [start + HALF_HOUR * i for i in range(days * 48)]
    local = [t.astimezone(ZoneInfo("Europe/London")) for t in times]
    hours = np.array([t.hour + t.minute / 60 for t in local])
    price = 20.0 + 2.0 * np.abs(hours - cheapest_local_hour)
    price = np.where((hours >= 17) & (hours < 19), 60.0, price)
    return pl.DataFrame({"time": times, "price_inc_vat_p_per_kwh": price}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )


def test_the_agile_template_charges_in_the_delivery_days_cheapest_slot_across_midnight() -> None:
    # The cheapest local slot is 23:30, which belongs to the delivery day that starts at 23:00 on
    # the evening before. A schedule anchored on UTC midnight would not charge there in September.
    grid = _grid(start=SEPTEMBER, days=10)
    prices = _agile_prices(start=SEPTEMBER - timedelta(days=2), days=14, cheapest_local_hour=23.5)

    template = agile_template(
        half_hour_end_time=grid,
        agile_prices=prices,
        duration_hours=1.0,
        round_trip_efficiency=ETA,
    )

    charge_slot = _index(
        grid=grid,
        start=SEPTEMBER,
        instant=SEPTEMBER + timedelta(days=4, hours=22, minutes=30),  # 23:30 BST
    )
    assert template[charge_slot] < 0.0
    discharge = np.flatnonzero(template[48 * 4 : 48 * 5 + 40] > 0) + 48 * 4
    assert len(discharge) > 0


def test_an_agile_day_with_a_missing_price_has_a_zero_schedule() -> None:
    grid = _grid(start=DECEMBER, days=6)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=10, cheapest_local_hour=3.0)
    # Drop one half-hour in the delivery day that runs from 23:00 on 2 December to 23:00 on 3.
    missing = DECEMBER + timedelta(days=2, hours=10)
    prices = prices.filter(pl.col("time") != missing)

    template = agile_template(
        half_hour_end_time=grid,
        agile_prices=prices,
        duration_hours=1.0,
        round_trip_efficiency=ETA,
    )

    day_slice = slice(48 * 2 - 2, 48 * 3 - 2)
    assert not template[day_slice].any()
    assert template[48 * 3 : 48 * 4].any()


def test_window_coverage_is_one_inside_each_window_and_a_share_at_a_half_covered_edge() -> None:
    grid = _grid(start=DECEMBER, days=4)
    window = TariffWindow(
        charge_start_hour=0.75,
        charge_end_hour=5.5,
        discharge_start_hour=16.0,
        discharge_end_hour=19.0,
    )

    charge, discharge = window_coverage(half_hour_end_time=grid, window=window)

    day = 48
    # The half-hour 00:30 to 01:00 is covered from 00:45, so half of it.
    assert charge[day + 1] == pytest.approx(0.5)
    assert charge[day] == 0.0
    assert charge[day + 2 : day + 11].tolist() == [1.0] * 9
    assert charge[day + 11] == 0.0
    assert discharge[day + 32 : day + 38].tolist() == [1.0] * 6
    assert discharge[day + 31] == 0.0
    assert discharge[day + 38] == 0.0


def test_a_window_that_spans_the_spring_clock_change_ends_at_its_wall_clock_time() -> None:
    start = datetime(2026, 3, 28, tzinfo=UTC)  # The clocks go forward at 01:00 on 29 March.
    grid = _grid(start=start, days=3)

    charge, _ = window_coverage(
        half_hour_end_time=grid, window=TARIFF_WINDOWS["intelligent_octopus_go"]
    )

    # The window runs from 23:30 on 28 March (GMT) to 05:30 on 29 March (BST), which is 04:30 UTC.
    last_covered = _index(grid=grid, start=start, instant=datetime(2026, 3, 29, 4, 0, tzinfo=UTC))
    assert charge[last_covered] == 1.0
    assert charge[last_covered + 1] == 0.0
    assert charge[last_covered - 9 : last_covered + 1].sum() == pytest.approx(10.0)


def test_a_weekday_only_window_covers_nothing_at_the_weekend() -> None:
    grid = _grid(start=DECEMBER, days=7)  # Monday 1 December 2025 to Sunday 7 December

    charge, discharge = window_coverage(half_hour_end_time=grid, window=TARIFF_WINDOWS["red_band"])

    day = 48
    assert discharge[2 * day + 32 : 2 * day + 38].tolist() == [1.0] * 6  # Wednesday
    assert not discharge[5 * day : 7 * day].any()  # Saturday and Sunday
    assert not charge[5 * day : 7 * day].any()


def test_agile_days_returns_each_delivery_days_prices_and_their_grid_positions() -> None:
    grid = _grid(start=DECEMBER, days=6)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=10, cheapest_local_hour=3.0)

    day_prices, slot_index = agile_days(half_hour_end_time=grid, agile_prices=prices)

    assert day_prices.shape[1] == 48
    assert slot_index.shape == day_prices.shape
    lookup = dict(
        zip(prices["time"].to_list(), prices["price_inc_vat_p_per_kwh"].to_list(), strict=True)
    )
    checked = 0
    for day in range(day_prices.shape[0]):
        for slot in range(48):
            index = slot_index[day, slot]
            if 0 <= index < len(grid):
                slot_start = DECEMBER + (index) * HALF_HOUR
                assert day_prices[day, slot] == pytest.approx(lookup[slot_start])
                checked += 1
    assert checked > 48 * 4


_SHARES: dict[DomesticTariffNameType, float] = {
    "intelligent_octopus_go": 0.4,
    "octopus_go": 0.3,
    "octopus_flux": 0.1,
    "agile": 0.2,
}


def test_a_fleets_output_is_the_sum_of_its_homes_outputs() -> None:
    grid = _grid(start=DECEMBER, days=10)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=14, cheapest_local_hour=3.0)
    spec = FleetSpec(tariff_shares=_SHARES)

    fleet = simulate_fleet(
        half_hour_end_time=grid,
        agile_prices=prices,
        n_homes=12,
        spec=spec,
        rng=np.random.default_rng(1),
    )

    by_hand = np.zeros(len(grid))
    for home in fleet.homes.iter_rows(named=True):
        by_hand += home["power_mw"] * domestic_template(
            tariff=home["tariff"],
            half_hour_end_time=grid,
            agile_prices=prices,
            duration_hours=home["duration_hours"],
            round_trip_efficiency=home["round_trip_efficiency"],
            spread_sd_log=0.0,
        )
    assert fleet.homes.height == 12
    assert fleet.homes["tariff"].n_unique() > 1
    assert fleet.output_mw == pytest.approx(by_hand)


def test_a_fleet_of_identical_homes_is_one_template_times_the_home_count() -> None:
    grid = _grid(start=DECEMBER, days=10)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=14, cheapest_local_hour=3.0)
    spec = FleetSpec(
        tariff_shares={"octopus_go": 1.0},
        home_power_kw=(4.0, 4.0),
        duration_median_hours=2.0,
        duration_sd_log=0.0,
        round_trip_efficiency_mean=0.82,
        round_trip_efficiency_sd=0.0,
    )

    fleet = simulate_fleet(
        half_hour_end_time=grid,
        agile_prices=prices,
        n_homes=50,
        spec=spec,
        rng=np.random.default_rng(2),
    )

    single = domestic_template(
        tariff="octopus_go",
        half_hour_end_time=grid,
        agile_prices=prices,
        duration_hours=2.0,
        round_trip_efficiency=0.82,
        spread_sd_log=0.0,
    )
    assert fleet.output_mw == pytest.approx(50 * 0.004 * single)


def test_a_merchant_templates_daily_discharge_is_capped_at_its_usable_duration() -> None:
    day = np.concatenate([np.full(12, 20.0), np.full(24, 50.0), np.full(12, 120.0)])
    prices = np.tile(day, 3) + np.tile(np.linspace(0.0, 0.3, 48), 3)

    standard = merchant_template(
        day_ahead_prices=prices, duration_hours=2.0, round_trip_efficiency=ETA
    )
    sensitivity = merchant_template(
        day_ahead_prices=prices,
        duration_hours=2.0,
        round_trip_efficiency=ETA,
        settings=SENSITIVITY_LP_SETTINGS,
    )

    # A strong spread cycles the whole usable energy: the cells swing by `duration` and the grid
    # receives `duration * sqrt(eta)`. A template built from the nameplate energy would cycle 10%
    # less in the standard setting, whose limits are 5% and 95%.
    expected = 2.0 * np.sqrt(ETA)
    assert standard[48:96][standard[48:96] > 0].sum() * 0.5 == pytest.approx(expected, rel=1e-6)
    assert sensitivity[48:96][sensitivity[48:96] > 0].sum() * 0.5 == pytest.approx(
        expected, rel=1e-6
    )


def test_a_daily_tariff_template_repeats_exactly_every_day_including_the_grids_edges() -> None:
    # The charge starts at 23:30 on the evening before, so the grid's first half-hour belongs to a
    # cycle that begins before the grid. A cycle that falls outside the grid must not wrap round.
    grid = _grid(start=DECEMBER, days=10)

    template = window_template(
        half_hour_end_time=grid,
        window=TARIFF_WINDOWS["intelligent_octopus_go"],
        duration_hours=2.0,
        round_trip_efficiency=ETA,
    )

    days = template.reshape(10, 48)
    assert days[0, 0] < 0.0
    assert days[0] == pytest.approx(days[1])
    assert days[9] == pytest.approx(days[1])


def test_a_discharge_that_starts_the_evening_before_the_grid_runs_into_its_first_half_hours() -> (
    None
):
    window = TariffWindow(
        charge_start_hour=2.0,
        charge_end_hour=7.0,
        discharge_start_hour=23.0,
        discharge_end_hour=26.0,
    )
    grid = _grid(start=DECEMBER, days=3)

    template = window_template(
        half_hour_end_time=grid, window=window, duration_hours=3.0, round_trip_efficiency=ETA
    )

    assert template[0] > 0.0  # from the cycle that begins at 23:00 on 30 November


def test_a_charge_that_starts_in_the_grids_last_half_hour_belongs_to_the_next_days_cycle() -> None:
    window = TariffWindow(
        charge_start_hour=-1.0,
        charge_end_hour=5.0,
        discharge_start_hour=16.0,
        discharge_end_hour=20.0,
    )
    grid = _grid(start=DECEMBER, days=3)[:-1]  # the last half-hour ends at 23:30

    template = window_template(
        half_hour_end_time=grid, window=window, duration_hours=1.0, round_trip_efficiency=ETA
    )

    assert template[-1] < 0.0  # from the cycle of 3 December, which charges from 23:00 on the 2nd


def test_the_discharge_window_cuts_the_usable_duration_when_it_is_the_shorter_window() -> None:
    window = TariffWindow(
        charge_start_hour=0.0,
        charge_end_hour=4.0,
        discharge_start_hour=16.0,
        discharge_end_hour=18.0,
    )

    capped = effective_duration_hours(window=window, duration_hours=6.0, round_trip_efficiency=ETA)

    assert capped == pytest.approx(2.0 / np.sqrt(ETA))


def test_a_weekday_only_window_covers_both_ends_of_the_working_week() -> None:
    grid = _grid(start=DECEMBER, days=7)  # Monday 1 December 2025 to Sunday 7 December
    window = TARIFF_WINDOWS["red_band"]

    _, discharge = window_coverage(half_hour_end_time=grid, window=window)

    day = 48
    for weekday in (0, 4):  # Monday and Friday
        assert discharge[weekday * day + 32 : weekday * day + 38].tolist() == [1.0] * 6


def test_spread_mean_is_the_log_normal_mean_of_the_durations() -> None:
    mean = spread_mean(
        template_for_duration=lambda duration: np.array([duration]),
        duration_hours=2.0,
        sd_log=0.3,
    )

    assert mean[0] == pytest.approx(2.0 * np.exp(0.3**2 / 2), rel=1e-3)


@pytest.mark.parametrize("name", ["intelligent_octopus_go", "octopus_go", "octopus_flux"])
def test_a_domestic_template_uses_its_own_tariffs_window_and_spreads_the_duration(
    name: Literal["intelligent_octopus_go", "octopus_go", "octopus_flux"],
) -> None:
    grid = _grid(start=DECEMBER, days=6)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=10, cheapest_local_hour=3.0)

    def window_for(duration: float) -> np.ndarray:
        return window_template(
            half_hour_end_time=grid,
            window=TARIFF_WINDOWS[name],
            duration_hours=duration,
            round_trip_efficiency=ETA,
        )

    def domestic(*, spread_sd_log: float) -> np.ndarray:
        return domestic_template(
            tariff=name,
            half_hour_end_time=grid,
            agile_prices=prices,
            duration_hours=2.0,
            round_trip_efficiency=ETA,
            spread_sd_log=spread_sd_log,
        )

    spread = domestic(spread_sd_log=0.3)

    assert domestic(spread_sd_log=0.0) == pytest.approx(window_for(2.0))
    assert spread == pytest.approx(
        spread_mean(template_for_duration=window_for, duration_hours=2.0, sd_log=0.3)
    )
    assert spread != pytest.approx(window_for(2.0))


def test_a_domestic_agile_template_is_the_agile_price_takers_schedule() -> None:
    grid = _grid(start=DECEMBER, days=6)
    prices = _agile_prices(start=DECEMBER - timedelta(days=2), days=10, cheapest_local_hour=3.0)

    domestic = domestic_template(
        tariff="agile",
        half_hour_end_time=grid,
        agile_prices=prices,
        duration_hours=2.0,
        round_trip_efficiency=ETA,
        spread_sd_log=0.0,
    )

    assert domestic == pytest.approx(
        agile_template(
            half_hour_end_time=grid,
            agile_prices=prices,
            duration_hours=2.0,
            round_trip_efficiency=ETA,
        )
    )


def test_the_price_takers_cycle_cap_comes_from_its_settings() -> None:
    # Two cheap and two dear blocks a day: a cap of one cycle uses one pair, a cap of two both.
    day = np.concatenate(
        [
            np.full(6, 10.0),
            np.full(6, 100.0),
            np.full(6, 10.0),
            np.full(6, 100.0),
            np.full(24, 50.0),
        ]
    )
    prices = np.tile(day, 3) + np.tile(np.linspace(0.0, 0.3, 48), 3)

    def daily_discharge(settings: LpSettings) -> float:
        template = merchant_template(
            day_ahead_prices=prices,
            duration_hours=1.0,
            round_trip_efficiency=ETA,
            settings=settings,
        )
        return float(template[48:96][template[48:96] > 0].sum() * 0.5)

    one = daily_discharge(LpSettings(soc_min=0.0, soc_max=1.0, cycles_per_day_cap=1.0))
    two = daily_discharge(LpSettings(soc_min=0.0, soc_max=1.0, cycles_per_day_cap=2.0))

    assert one == pytest.approx(1.0, rel=1e-6)
    assert two == pytest.approx(2 * np.sqrt(ETA), rel=1e-6)


def test_a_delivery_day_that_starts_the_grid_keeps_its_first_slot() -> None:
    start = DECEMBER - timedelta(hours=1)  # 23:00 on 30 November, the first delivery day's start
    grid = _grid(start=start, days=2)
    prices = _agile_prices(start=start - timedelta(days=1), days=5, cheapest_local_hour=23.0)

    template = agile_template(
        half_hour_end_time=grid,
        agile_prices=prices,
        duration_hours=1.0,
        round_trip_efficiency=ETA,
    )
    _, slot_index = agile_days(half_hour_end_time=grid, agile_prices=prices)

    assert 0 in slot_index
    assert template[0] < 0.0


def test_the_spring_clock_change_leaves_the_delivery_days_missing_slots_empty() -> None:
    start = datetime(2026, 3, 27, 12, tzinfo=UTC)
    grid = _grid(start=start, days=4)
    prices = _agile_prices(start=start - timedelta(days=2), days=8, cheapest_local_hour=3.0)

    day_prices, slot_index = agile_days(half_hour_end_time=grid, agile_prices=prices)

    rows = np.flatnonzero(np.isnan(day_prices).any(axis=1))
    assert len(rows) == 1  # the 23-hour delivery day of 29 March
    row = rows[0]
    assert np.isnan(day_prices[row, 46:]).all()
    assert np.isfinite(day_prices[row, :46]).all()
    assert (slot_index[row, 46:] == -1).all()
    assert (slot_index[row, :46] == slot_index[row, 0] + np.arange(46)).all()


def test_the_drawn_homes_follow_the_tariff_shares_and_stay_within_their_ranges() -> None:
    spec = FleetSpec(
        tariff_shares={"octopus_go": 0.9, "agile": 0.1},
        home_power_kw=(3.0, 5.0),
        round_trip_efficiency_mean=0.8,
        round_trip_efficiency_sd=0.3,
    )

    homes = draw_homes(n_homes=4000, spec=spec, rng=np.random.default_rng(0))

    power = homes["power_mw"].to_numpy()
    efficiency = homes["round_trip_efficiency"].to_numpy()
    assert (homes["tariff"] == "octopus_go").mean() == pytest.approx(0.9, abs=0.02)
    assert power.min() >= 0.003
    assert power.max() <= 0.005
    assert power.max() > 0.0049
    assert efficiency.min() == MIN_HOME_EFFICIENCY
    assert efficiency.max() == MAX_HOME_EFFICIENCY
