from datetime import UTC, datetime

import polars as pl
import pytest
from studies.ifs_single_runs import (
    DomainType,
    clip_radiation,
    last_servable_day,
    native_step_hours,
    served_init_time,
    served_lead_hours,
)


def _served(*, time: datetime, day: int, domain: DomainType) -> tuple[datetime, int]:
    frame = pl.DataFrame({"time": [time]}).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    return frame.select(
        init=served_init_time(time=pl.col("time"), day=day, domain=domain),
        lead=served_lead_hours(time=pl.col("time"), day=day, domain=domain),
    ).row(0)


def test_a_solar_hour_ending_at_midnight_belongs_to_the_day_it_ended():
    init, lead = _served(time=datetime(2025, 3, 11, 0), day=2, domain="solar")

    assert init == datetime(2025, 3, 8, tzinfo=UTC)
    assert lead == 72


def test_a_wind_hour_at_midnight_belongs_to_the_day_it_starts():
    init, lead = _served(time=datetime(2025, 3, 11, 0), day=2, domain="wind")

    assert init == datetime(2025, 3, 9, tzinfo=UTC)
    assert lead == 48


def test_day_zero_reads_the_run_of_the_hours_own_day_and_never_the_freshest_run():
    init, lead = _served(time=datetime(2025, 3, 10, 18), day=0, domain="wind")

    assert init == datetime(2025, 3, 10, tzinfo=UTC)
    assert lead == 18


def test_solar_leads_run_from_24_n_plus_1_to_24_n_plus_24():
    labels = [datetime(2025, 3, 10, hour) for hour in range(1, 24)] + [datetime(2025, 3, 11, 0)]

    leads = [_served(time=label, day=3, domain="solar")[1] for label in labels]

    assert leads == list(range(73, 97))


def test_wind_leads_run_from_24_n_to_24_n_plus_23():
    labels = [datetime(2025, 3, 10, hour) for hour in range(24)]

    leads = [_served(time=label, day=3, domain="wind")[1] for label in labels]

    assert leads == list(range(72, 96))


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_day_nine_is_the_last_day_whose_hours_all_lie_within_the_run(domain: DomainType):
    last_label = datetime(2025, 3, 11, 0) if domain == "solar" else datetime(2025, 3, 10, 23)

    assert last_servable_day(domain=domain) == 9
    assert _served(time=last_label, day=9, domain=domain)[1] <= 240
    assert _served(time=last_label, day=10, domain=domain)[1] > 240


@pytest.mark.parametrize(
    ("lead", "width"), [(0, 1), (90, 1), (91, 3), (144, 3), (145, 6), (240, 6)]
)
def test_native_step_widths_change_after_lead_90_and_144(lead: int, width: int):
    assert native_step_hours(lead_hours=lead) == width


def test_a_lead_beyond_the_run_has_no_native_step():
    with pytest.raises(ValueError, match="241"):
        native_step_hours(lead_hours=241)


def test_radiation_is_clipped_at_zero_and_left_alone_above_it():
    frame = pl.DataFrame({"r": [-1.0, 0.0, 5.5, None]})

    clipped = frame.select(clip_radiation(radiation=pl.col("r")))["r"].to_list()

    assert clipped == [0.0, 0.0, 5.5, None]
