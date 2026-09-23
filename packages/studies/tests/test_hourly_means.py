from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.hourly_means import hourly_from_running_means, hourly_from_snapshots

START = datetime(2025, 6, 1, tzinfo=UTC)


def _hourly_means(*, hours: int) -> list[float]:
    # Every hour's mean differs from its neighbours', so recovering the wrong hour cannot pass.
    return [float(10 * hour + (hour % 7)) for hour in range(1, hours + 1)]


def _running_means(
    *, means: list[float], cycle_hours: int, first_start_hour: int, start: datetime = START
) -> pl.DataFrame:
    # Build what the reanalysis stores: the value labelled h is the mean since the latest start at
    # or before h - 1, with starts every `cycle_hours` hours from `first_start_hour`. `means[i]` is
    # the mean over the hour ending at `start + (i + 1) h`; a label whose window reaches back before
    # `means` begins is left out.
    rows = []
    for index in range(len(means)):
        label = start + timedelta(hours=index + 1)
        step = (label.hour - first_start_hour - 1) % cycle_hours + 1
        if index + 1 - step < 0:
            continue
        window = means[index + 1 - step : index + 1]
        rows.append({"key": "cell", "time": label, "value": sum(window) / len(window)})
    return pl.DataFrame(rows)


def test_running_means_are_de_averaged_into_the_hourly_means():
    means = _hourly_means(hours=24)
    stored = _running_means(means=means, cycle_hours=3, first_start_hour=0)

    recovered = hourly_from_running_means(frame=stored, value_column="value", cycle_hours=3)

    # The first label, 01:00, is step 1 and needs no previous hour, so every hour is recovered.
    assert recovered["value"].to_list() == pytest.approx(means)


def test_starts_off_the_assumed_phase_give_different_hours():
    # Forecasts starting at 01, 04, ... UTC rather than 00, 03, ... is the error the phase guards.
    means = _hourly_means(hours=24)
    stored = _running_means(means=means, cycle_hours=3, first_start_hour=1)

    recovered = hourly_from_running_means(frame=stored, value_column="value", cycle_hours=3)

    expected = {START + timedelta(hours=index + 1): mean for index, mean in enumerate(means)}
    wrong = [
        row["value"] != pytest.approx(expected[row["time"]])
        for row in recovered.iter_rows(named=True)
    ]
    assert sum(wrong) > len(wrong) / 2


def test_midnight_is_the_third_step_of_the_run_starting_at_21_utc():
    # 00 UTC is where a truncating modulo would put step 0 rather than step 3.
    evening = datetime(2025, 5, 31, 20, tzinfo=UTC)
    means = _hourly_means(hours=6)
    stored = _running_means(means=means, cycle_hours=3, first_start_hour=0, start=evening)

    recovered = hourly_from_running_means(frame=stored, value_column="value", cycle_hours=3)

    # means[3] is the hour ending at 00:00, the fourth label after 20:00.
    assert recovered.filter(pl.col("time").dt.hour() == 0)["value"].item() == pytest.approx(
        means[3]
    )


def test_a_step_without_its_previous_hour_is_dropped_not_guessed():
    means = _hourly_means(hours=6)
    stored = _running_means(means=means, cycle_hours=3, first_start_hour=0)
    gap = START + timedelta(hours=1)

    recovered = hourly_from_running_means(
        frame=stored.filter(pl.col("time") != gap), value_column="value", cycle_hours=3
    )

    # 02:00 is step 2 and needed 01:00; 03:00 is step 3 and still has 02:00's stored value.
    assert (START + timedelta(hours=2)) not in recovered["time"].to_list()
    assert recovered.height == 4


def test_padding_rows_are_dropped_before_de_averaging():
    means = _hourly_means(hours=6)
    stored = _running_means(means=means, cycle_hours=3, first_start_hour=0)
    padded = pl.concat(
        [stored, pl.DataFrame({"key": ["cell"], "time": [START], "value": [float("nan")]})]
    )

    recovered = hourly_from_running_means(frame=padded, value_column="value", cycle_hours=3)

    assert recovered["value"].to_list() == pytest.approx(means)


def test_a_negative_de_averaged_value_is_clipped_to_zero():
    stored = pl.DataFrame(
        {
            "key": ["cell", "cell"],
            "time": [START + timedelta(hours=1), START + timedelta(hours=2)],
            "value": [10.0, 4.0],
        }
    )

    recovered = hourly_from_running_means(frame=stored, value_column="value", cycle_hours=3)

    assert recovered["value"].to_list() == [10.0, 0.0]


def test_a_duplicated_hour_raises():
    stored = pl.DataFrame({"key": ["cell", "cell"], "time": [START, START], "value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="more than one row"):
        hourly_from_running_means(frame=stored, value_column="value", cycle_hours=3)


def _snapshots(*, hours: int) -> pl.DataFrame:
    # Each snapshot's value is its own minute of the day, so a mean names the slots it used.
    stamps = [START + timedelta(minutes=30 * slot) for slot in range(2 * hours)]
    return pl.DataFrame(
        {
            "key": "site",
            "time": stamps,
            "ghi": [float(stamp.hour * 60 + stamp.minute) for stamp in stamps],
            "bhi": [float(stamp.hour * 60 + stamp.minute) / 2 for stamp in stamps],
        }
    )


def test_each_hour_averages_the_two_snapshots_before_its_label():
    hourly = hourly_from_snapshots(
        frame=_snapshots(hours=24), value_columns=("ghi", "bhi"), slot_offsets_minutes=(-60, -30)
    )

    noon = hourly.filter(pl.col("time") == START + timedelta(hours=12))
    # 11:00 is minute 660 and 11:30 is minute 690. The snapshots at 11:30 and 12:00 would give 705.
    assert noon["ghi"].item() == 675.0
    assert noon["bhi"].item() == 337.5
    # 48 snapshots from 00:00 to 23:30 complete the 24 hours ending 01:00 to 24:00.
    assert hourly.height == 24


def test_an_hour_missing_a_snapshot_is_dropped():
    snapshots = _snapshots(hours=4).with_columns(
        ghi=pl.when(pl.col("time") == START + timedelta(hours=1, minutes=30))
        .then(float("nan"))
        .otherwise(pl.col("ghi"))
    )

    hourly = hourly_from_snapshots(
        frame=snapshots, value_columns=("ghi", "bhi"), slot_offsets_minutes=(-60, -30)
    )

    assert (START + timedelta(hours=2)) not in hourly["time"].to_list()
    assert hourly.height == 3


def test_snapshots_need_at_least_one_offset():
    with pytest.raises(ValueError, match="at least one"):
        hourly_from_snapshots(
            frame=_snapshots(hours=2), value_columns=("ghi",), slot_offsets_minutes=()
        )
